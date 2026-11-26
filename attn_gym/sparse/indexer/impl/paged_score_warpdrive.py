# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by Attention Gym in 2026: paged scoring with warpdrive roles and channels.

"""Paged lightning-indexer scoring on SM100-family GPUs, written on warpdrive.

Scores ``scale * sum_h(weights[t, h] * ReLU(query[t, h] @ key))`` for every visible logical
candidate of each query into the bounded FP32 slab that the radix selector consumes, and
records each query's global candidate interval ``[0, visible)``. It selects nothing.

Each CTA owns one query row and a strided subset of that row's M128 logical candidate
tiles: tile ``c, c + C, c + 2C, ...`` for CTA ``c`` of ``C``. A tile covers
``128 // page_size`` physical pages, each looked up in the block table and loaded by TMA,
so the grid depends only on host shapes while every CTA past the visible prefix exits
after its prologue. Pages past the visible prefix are never read: a tile's trailing page
slot that starts beyond ``visible`` reloads the tile's last visible page instead, and its
scores are not stored. Rows with ``visible <= topk`` write only their bounds.

Roles and channels of one work item ``(c, row, 0)``::

    load (warp 5) --q: 1 stage, TMA--> mma (warp 4) --acc: 2 TMEM stages--> epilogue (0-3)
                  --k: 3 stages, TMA-->

``q`` carries one epoch per item with tiles and none otherwise; ``k`` and ``acc`` one per
visited tile. UMMA computes ``K[128, D] @ Q[16, D].T`` with heads zero-filled to N16 by TMA.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, cute
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05

from attn_gym._backends.cute import get_device_properties
from attn_gym.utils import cdiv
from attn_gym.warpdrive import (
    Consumer,
    Local,
    OneShot,
    Producer,
    Ring,
    SmemPlan,
    StageView,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    WorkCursor,
    WorkStream,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, get_compile_target, jit_cache

from .cute_kernel_base import IndexerScoreKernelBase

_TILE_CANDIDATES = 128
_HEAD_TILE = 16  # the smallest UMMA N for M128
_IO_DTYPES = {torch.bfloat16: "bfloat16", torch.float16: "float16"}


@dataclass(frozen=True)
class PagedScoreParams:
    """Kernel-prologue values the roles read."""

    tiled_mma: cute.TiledMma
    tma_k: cute.CopyAtom
    k: cute.Tensor  # K TMA coordinates: (page_size, D, pages)
    tma_q: cute.CopyAtom
    q: cute.Tensor  # Q TMA coordinates: (H, D, T)
    sK_pages: cute.Tensor  # K stages as (page_size, D, pages per tile, stage) for TMA
    sK: cute.Tensor  # the same stages in the UMMA A-operand layout
    sQ: cute.Tensor
    acc_layout: cute.Layout
    weights: cute.Tensor
    table: cute.Tensor
    cu_seqlens: cute.Tensor
    lengths: cute.Tensor
    positions: cute.Tensor | None
    scores: cute.Tensor  # (slab rows, capacity)
    bounds: cute.Tensor  # (T, 2)
    row_start: Int32
    ctas_per_row: Int32
    topk: Int32
    ratio: Int32


@dataclass(frozen=True)
class RowWork:
    """One CTA's share of a query row: ``tiles`` M128 tiles starting at ``first``, every
    ``stride``-th tile."""

    row: Int32
    sequence: Int32
    visible: Int32
    first: Int32
    stride: Int32
    tiles: Int32


class PagedWarpdriveScore(IndexerScoreKernelBase):
    """Paged score kernel for one FP16/BF16 ``(heads, head_dim, page_size)`` specialization.

    Reuses the dense score kernels' TMEM epilogue helpers; its own launch replaces theirs.

    ``explicit_positions`` selects the kernel that reads ``query_positions``; the other takes
    None and infers suffix positions. Every count, offset, ``topk`` and ``compress_ratio`` is
    a runtime Int32. Q, K, weights, scores and bounds must be contiguous; Q and K need
    16-byte-aligned bases for TMA. The block table may have any int32-representable strides.
    """

    name_prefix = "paged_warpdrive_score"
    q_stages = 1
    # Measured on GB300 with one wave of CTAs: 2 and 3 stages tie; 3 leaves 2 CTAs per SM.
    k_stages = 3

    def __init__(
        self,
        dtype: str,
        heads: int,
        head_dim: int,
        page_size: int,
        explicit_positions: bool,
    ):
        if heads > _HEAD_TILE or heads % 4:
            raise ValueError(f"PagedWarpdriveScore requires H in 4..{_HEAD_TILE}, H % 4 == 0")
        if head_dim % 64:
            raise ValueError("PagedWarpdriveScore requires D divisible by 64")
        if page_size not in (32, 64, 128):
            raise ValueError("PagedWarpdriveScore requires page_size 32, 64 or 128")
        self.dtype = cutlass.BFloat16 if dtype == "bfloat16" else cutlass.Float16
        self.heads = heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.explicit_positions = explicit_positions
        self.pages_per_tile = _TILE_CANDIDATES // page_size
        self.mma_tile = (_TILE_CANDIDATES, _HEAD_TILE, head_dim)
        operand_bytes = self.dtype.width // 8
        self.smem_plan = SmemPlan()
        k_buffer = self.smem_plan.buffer("K", self.dtype, align=1024)
        q_buffer = self.smem_plan.buffer("Q", self.dtype, align=1024)
        self.k = Ring(
            self.k_stages,
            Tma(_TILE_CANDIDATES * head_dim * operand_bytes),
            Umma,
            buffer=k_buffer,
        )
        self.q = Ring(
            self.q_stages, Tma(_HEAD_TILE * head_dim * operand_bytes), Umma, buffer=q_buffer
        )
        # tcgen05.wait::ld before each release, so MMA cannot overwrite a stage still being read.
        self.acc = Ring(self.acc_stages, Umma, Threads, release_fence="tmem_load")
        self.tmem_plan = TmemPlan().region("acc", _HEAD_TILE, self.acc_stages)
        self.work = WorkStream(schedule=OneShot())

    def resident_ctas(self, device: torch.device) -> int:
        """CTAs that fit on the device at once: SMEM, about 2 KiB of it for barriers, TMEM
        address, alignment and the per-CTA reservation, is the binding limit."""
        properties = get_device_properties(device)
        operand_bytes = self.dtype.width // 8
        cta_bytes = (
            self.k_stages * _TILE_CANDIDATES + _HEAD_TILE
        ) * self.head_dim * operand_bytes + 2048
        per_sm = max(1, properties.shared_memory_per_multiprocessor // cta_bytes)
        return properties.multi_processor_count * per_sm

    def get_name(self) -> str:
        return (
            f"{self.name_prefix}_{self.dtype.__name__.lower()}_h{self.heads}_d{self.head_dim}"
            f"_p{self.page_size}_pos{int(self.explicit_positions)}"
        )

    # ------------------------------------------------------------------ metadata

    @cute.jit
    def find_sequence(self, params: PagedScoreParams, row: Int32, lane: Int32):
        """Return ``(sequence, length, end)`` for the sequence whose query range holds ``row``.

        ``sequence`` is the first ``s`` with ``cu_seqlens[s + 1] > row`` (``B`` for a capacity
        row, skipping empty sequences), found by a 32-ary warp search: each step probes 32
        evenly spaced boundaries, ``ceil(log32(B))`` dependent loads in all. Each probe also
        loads its sequence's length, so the final step returns ``length = seqused_k[s]`` and
        ``end = cu_seqlens[s + 1]`` without another dependent load, unless ``s`` lies just past
        the final step's probes. Offsets stay below the remaining width, so no sum exceeds
        ``B``. Every lane of the warp must call it; the results are warp-uniform.
        """
        sequences = params.lengths.shape[0]
        low = Int32(0)
        high = sequences
        length = Int32(0)
        end = Int32(0)
        found = cutlass.Boolean(False)
        while high > low:
            width = high - low
            span = cute.ceil_div(width, 32)
            offset = lane * span + (span - 1)
            boundary = Int32(0)
            probe_length = Int32(0)
            hit = cutlass.Boolean(False)
            if offset < width:
                boundary = params.cu_seqlens[low + offset + 1]
                probe_length = params.lengths[low + offset]
                hit = boundary <= row
            passed = cute.arch.popc(cute.arch.vote_ballot_sync(hit))
            low += passed * span
            if span == 1:
                # Lane ``passed`` probed the answer, unless every remaining probe passed.
                end = cute.arch.shuffle_sync(boundary, passed)
                length = cute.arch.shuffle_sync(probe_length, passed)
                found = passed < width
            high = low + cutlass.min(span - 1, high - low)
        if low < sequences and not found:
            end = params.cu_seqlens[low + 1]
            length = params.lengths[low]
        return low, length, end

    @cute.jit
    def row_work(self, params: PagedScoreParams, item, lane: Int32) -> RowWork:
        """Decode item ``(c, local row, 0)``; every role derives its trip counts from this."""
        c, local_row, _ = item
        row = params.row_start + local_row
        explicit = Int32(0)
        if cutlass.const_expr(self.explicit_positions):
            explicit = params.positions[row]  # independent of the search, so issued first
        sequence, length, end = self.find_sequence(params, row, lane)
        visible = Int32(0)
        if sequence < params.lengths.shape[0]:
            # Both rules keep position below length, which caps visible at length // ratio.
            if cutlass.const_expr(self.explicit_positions):
                # Clamp before adding one: an INT32_MAX position must not wrap.
                position = cutlass.min(explicit, length - 1)
            else:
                # Suffix rule without forming length - end, which may be negative.
                position = length - (end - row)
            visible = cutlass.max(position + 1, Int32(0)) // params.ratio
        total_tiles = Int32(0)
        if visible > params.topk:
            total_tiles = cute.ceil_div(visible, _TILE_CANDIDATES)
        tiles = Int32(0)
        if c < total_tiles:
            tiles = cute.ceil_div(total_tiles - c, params.ctas_per_row)
        return RowWork(row, sequence, visible, Int32(c), params.ctas_per_row, tiles)

    # ------------------------------------------------------------------ roles

    @role(warps=5)
    @cute.jit
    def load(self, params: PagedScoreParams, *, q: Producer, k: Producer, work: WorkCursor):
        """Load Q once, then each tile's pages; a lane per table entry batches page lookups."""
        cpasync.prefetch_descriptor(params.tma_k)
        cpasync.prefetch_descriptor(params.tma_q)
        lane = cute.arch.lane_idx()
        tiles_per_batch = 32 // self.pages_per_tile
        mma_zero = params.tiled_mma.get_slice(0)
        tKsK, tKgK = cpasync.tma_partition(
            params.tma_k,
            0,
            cute.make_layout(1),
            cute.group_modes(params.sK_pages, 0, 2),
            cute.group_modes(params.k, 0, 2),
        )
        k = k.views(K=StageView(tKsK, axis=2))
        while work.valid:
            share = self.row_work(params, work.item, lane)
            # A CTA without tiles loads no Q: no MMA consumes it.
            if share.tiles > 0:
                gQ = cute.local_tile(
                    params.q[None, None, share.row], self.mma_tile, (0, 0, 0), proj=(None, 1, 1)
                )
                tQsQ, tQgQ = cpasync.tma_partition(
                    params.tma_q,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(params.sQ, 0, 3),
                    cute.group_modes(mma_zero.partition_B(gQ), 0, 3),
                )
                q = q.views(Q=StageView(tQsQ, axis=1))
                with q.produce() as h:
                    cute.copy(params.tma_q, tQgQ, h.view("Q"), tma_bar_ptr=h.barrier)
                page = Int32(0)
                for i in cutlass.range(share.tiles, unroll=0):
                    slot = i % tiles_per_batch
                    if slot == 0:
                        # Lane l looks up page slot l % pages_per_tile of the batch's tile
                        # l // pages_per_tile; a slot past the visible prefix takes the tile's
                        # last visible page, so no unreferenced table entry or page is read.
                        tile = share.first + (i + lane // self.pages_per_tile) * share.stride
                        candidate = tile * _TILE_CANDIDATES + (lane % self.pages_per_tile) * (
                            self.page_size
                        )
                        candidate = cutlass.min(candidate, share.visible - 1)
                        if i + lane // self.pages_per_tile < share.tiles:
                            page = params.table[share.sequence, candidate // self.page_size]
                    with k.produce() as h:
                        for slot_page in cutlass.range_constexpr(self.pages_per_tile):
                            physical = cute.arch.shuffle_sync(
                                page, slot * self.pages_per_tile + slot_page
                            )
                            cute.copy(
                                params.tma_k,
                                tKgK[None, physical],
                                h.view("K")[None, slot_page],
                                tma_bar_ptr=h.barrier,
                            )
            work.advance()
        return q, k, work

    @role(warps=4)
    @cute.jit
    def mma(
        self,
        params: PagedScoreParams,
        *,
        q: Consumer,
        k: Consumer,
        acc: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        """Compute [128, D] @ [16, D].T for each tile into a two-stage FP32 TMEM ring."""
        tiled_mma = params.tiled_mma
        lane = cute.arch.lane_idx()
        accumulator = cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        fragment_k = tiled_mma.make_fragment_A(params.sK)
        q = q.views(Q=StageView(tiled_mma.make_fragment_B(params.sQ), axis=3))
        k = k.views(K=StageView(fragment_k, axis=3))
        while work.valid:
            share = self.row_work(params, work.item, lane)
            if share.tiles > 0:
                q_full = q.wait()  # every K tile of the item multiplies this Q stage
                for _ in cutlass.range(share.tiles, unroll=0):
                    acc_empty = acc.acquire()
                    k_full = k.wait()
                    for d_block in cutlass.range_constexpr(cute.size(fragment_k, mode=[2])):
                        issue_mma = tiled_mma.with_()
                        issue_mma.set(tcgen05.Field.ACCUMULATE, d_block != 0)
                        cute.gemm(
                            issue_mma,
                            accumulator[(None, None, None, acc_empty.index)],
                            k_full.view("K", None, None, d_block),
                            q_full.view("Q", None, None, d_block),
                            accumulator[(None, None, None, acc_empty.index)],
                        )
                    acc_empty.commit()
                    k_full.release()
                q_full.release()
            work.advance()
        return q, k, acc, work

    @role(warps=range(4))
    @cute.jit
    def epilogue(
        self,
        params: PagedScoreParams,
        *,
        acc: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        """Reduce each thread's candidate row of logits and store visible scores."""
        accumulator, tmem_copy, tmem_thread, tile_candidate, logits = self.accumulator_reader(
            params, tmem, me
        )
        rW = cute.make_rmem_tensor((self.heads,), Float32)
        score_scale = Float32(1.0 / math.sqrt(self.heads * self.head_dim))
        lane = cute.arch.lane_idx()
        while work.valid:
            share = self.row_work(params, work.item, lane)
            local_row = work.item[1]
            if share.first == 0 and me.tid == 0:
                params.bounds[share.row, 0] = Int32(0)
                params.bounds[share.row, 1] = share.visible
            if share.tiles > 0:
                for head in cutlass.range_constexpr(self.heads):
                    rW[head] = Float32(params.weights[share.row, head])
            for i in cutlass.range(share.tiles, unroll=0):
                acc_full = acc.wait()
                cute.copy(
                    tmem_copy,
                    tmem_thread.partition_S(accumulator[(None, None, None, acc_full.index)]),
                    logits,
                )
                acc_full.release()
                tile = share.first + i * share.stride
                candidate = tile * _TILE_CANDIDATES + tile_candidate
                if candidate < share.visible:
                    sums = self.add_weighted_relu((Float32(0.0),) * 4, logits, rW, 0, self.heads)
                    params.scores[local_row, candidate] = (
                        (sums[0] + sums[1]) + (sums[2] + sums[3])
                    ) * score_scale
            work.advance()
        return acc, work

    # ------------------------------------------------------------------ launch

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_k: cute.CopyAtom,
        k: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        q: cute.Tensor,
        k_layout: cute.ComposedLayout,
        k_page_layout: cute.ComposedLayout,
        q_layout: cute.ComposedLayout,
        weights: cute.Tensor,
        table: cute.Tensor,
        cu_seqlens: cute.Tensor,
        lengths: cute.Tensor,
        positions: cute.Tensor | None,
        scores: cute.Tensor,
        bounds: cute.Tensor,
        row_start: Int32,
        ctas_per_row: Int32,
        topk: Int32,
        ratio: Int32,
        bound,
    ):
        self.begin_roles(work=bound, layouts={"K": k_layout, "Q": q_layout})
        sK = self.buffer("K")
        accumulator_shape = tiled_mma.partition_shape_C(self.mma_tile[:2])
        params = PagedScoreParams(
            tiled_mma=tiled_mma,
            tma_k=tma_atom_k,
            k=k,
            tma_q=tma_atom_q,
            q=q,
            # The buffer's pointer already carries the swizzle shared by both layouts.
            sK_pages=cute.make_tensor(sK.iterator, k_page_layout.outer),
            sK=sK,
            sQ=self.buffer("Q"),
            acc_layout=tiled_mma.make_fragment_C(
                cute.append(accumulator_shape, self.acc_stages)
            ).layout,
            weights=weights,
            table=table,
            cu_seqlens=cu_seqlens,
            lengths=lengths,
            positions=positions,
            scores=scores,
            bounds=bounds,
            row_start=row_start,
            ctas_per_row=ctas_per_row,
            topk=topk,
            ratio=ratio,
        )
        self.run_roles(params)

    @cute.jit
    def __call__(
        self,
        query: cute.Tensor,
        key_cache: cute.Tensor,
        weights: cute.Tensor,
        table: cute.Tensor,
        cu_seqlens: cute.Tensor,
        lengths: cute.Tensor,
        positions: cute.Tensor | None,
        scores: cute.Tensor,
        bounds: cute.Tensor,
        row_start: Int32,
        rows: Int32,
        ctas_per_row: Int32,
        topk: Int32,
        ratio: Int32,
        stream: cuda.CUstream,
    ):
        """Launch ``(ctas_per_row, rows)`` CTAs for global rows ``row_start + [0, rows)``."""
        tiled_mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                self.dtype,
                Float32,
                (_TILE_CANDIDATES, _HEAD_TILE, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        )
        k_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tile, self.dtype, self.k_stages
        )
        # The UMMA A layout stores each K atom's 128 rows contiguously (1 KiB per 8 rows),
        # then the next K atom: rows of page slot p start p * page_size rows in. The same
        # bytes as (row in page, D, page slot, stage) are one TMA destination per page.
        atom = tcgen05.make_smem_layout_atom(
            sm100_utils.get_smem_layout_atom_ab(
                OperandMajorMode.K, self.dtype, (_TILE_CANDIDATES, self.head_dim)
            ),
            self.dtype,
        )
        k_page_layout = cute.tile_to_shape(
            atom,
            (self.page_size, self.head_dim, self.pages_per_tile, self.k_stages),
            order=(0, 2, 1, 3),
        )
        q_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tile, self.dtype, self.q_stages
        )
        tma_load = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        k_pages = cute.make_tensor(
            key_cache.iterator, cute.select(key_cache.layout, mode=[1, 2, 0])
        )
        tma_atom_k, k_tma = cpasync.make_tiled_tma_atom(
            tma_load,
            k_pages,
            cute.slice_(k_page_layout, (None, None, 0, 0)),
            (self.page_size, self.head_dim),
        )
        q_hdt = cute.make_tensor(query.iterator, cute.select(query.layout, mode=[1, 2, 0]))
        tma_atom_q, q_tma = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load,
            q_hdt,
            cute.select(q_layout, mode=[0, 1, 2]),
            self.mma_tile,
            tiled_mma,
        )
        bound = self.work.bind((ctas_per_row, rows, 1))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            tiled_mma,
            tma_atom_k,
            k_tma,
            tma_atom_q,
            q_tma,
            k_layout,
            k_page_layout,
            q_layout,
            weights,
            table,
            cu_seqlens,
            lengths,
            positions,
            scores,
            bounds,
            row_start,
            ctas_per_row,
            topk,
            ratio,
            bound,
        ).launch(**self.launch_config(bound), stream=stream)


@jit_cache
def compile_paged_score(operation: PagedWarpdriveScore) -> Callable[..., None]:
    """Compile the constructor-keyed kernel with symbolic T, B, pages, capacity and slab rows."""
    sym = cute.sym_int
    tokens, sequences, cu_size, pages, table_pages, capacity, slab_rows = (sym() for _ in range(7))
    io_dtype = operation.dtype

    def compact(dtype, shape, align):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=align,
            use_32bit_stride=True,
        )

    table = cute.runtime.make_fake_tensor(
        Int32,
        (sequences, table_pages),
        stride=(sym(), sym()),
        assumed_align=4,
    )
    return compile_tvm_ffi(
        operation,
        compact(io_dtype, (tokens, operation.heads, operation.head_dim), 16),
        compact(io_dtype, (pages, operation.page_size, operation.head_dim), 16),
        compact(io_dtype, (tokens, operation.heads), io_dtype.width // 8),
        table,
        compact(Int32, (cu_size,), 4),
        compact(Int32, (sequences,), 4),
        compact(Int32, (tokens,), 4) if operation.explicit_positions else None,
        compact(Float32, (slab_rows, capacity), 16),
        compact(Int32, (tokens, 2), 8),
        Int32(0),
        Int32(0),
        Int32(1),
        Int32(0),
        Int32(1),
        gpu_arch=get_compile_target().gpu_arch,
    )


def _ctas_per_row(rows: int, capacity: int, resident_ctas: int) -> int:
    """Split each row's tiles over at most one wave of resident CTAs.

    A second-wave CTA starts only when a first-wave CTA ends, so its tiles would finish
    last; one floored wave measured fastest from 1 to 64 rows of 1K to 32K candidates.
    """
    return max(1, min(resident_ctas // rows, cdiv(capacity, _TILE_CANDIDATES)))


def get_paged_score_runner(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    query_positions: torch.Tensor | None,
    scores: torch.Tensor,
    bounds: torch.Tensor,
    topk: int,
    compress_ratio: int,
    *,
    checked: bool = False,
) -> Callable[[int, int], None]:
    """Compile once and return ``run(start, rows)``, scoring global rows ``[start, start + rows)``.

    ``scores`` is the contiguous FP32 ``[pairs, 2, capacity]`` slab; slab row ``r`` holds
    global row ``start + r``, written only for candidates below ``visible`` and only when
    ``visible > topk``. ``bounds[row]`` receives ``(0, visible)`` for every launched row.
    Call inside the caller's compile-target scope, on the query's device and stream.
    ``checked`` builds warpdrive's bounds-checked variant, for tests.
    """
    capacity = scores.shape[-1]
    operation = PagedWarpdriveScore.get_or_build(
        _IO_DTYPES[query.dtype],
        query.shape[1],
        query.shape[2],
        key_cache.shape[1],
        query_positions is not None,
        checked=checked,
    )
    compiled = compile_paged_score(operation)
    slab = scores.view(-1, capacity)
    query, key_cache, weights = query.detach(), key_cache.detach(), weights.detach()
    resident_ctas = operation.resident_ctas(query.device)

    def run(start: int, rows: int) -> None:
        compiled(
            query,
            key_cache,
            weights,
            block_table,
            cu_seqlens_q,
            seqused_k,
            query_positions,
            slab,
            bounds,
            start,
            rows,
            _ctas_per_row(rows, capacity, resident_ctas),
            topk,
            compress_ratio,
        )

    return run


__all__ = ["PagedWarpdriveScore", "compile_paged_score", "get_paged_score_runner"]
