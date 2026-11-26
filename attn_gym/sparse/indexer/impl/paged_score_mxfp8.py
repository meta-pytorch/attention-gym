# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
#
# The block-scaled MMA issue, scale packing and scale lifetimes are adapted from Attention
# Gym's dense MXFP8 indexer scorer (cute_score_mxfp8.py at 61c61903), whose score pipeline
# adapts the Apache-2.0 cuDNN implementation in cute_score.py (see NOTICE.cudnn and
# LICENSE.Apache-2.0).
# The scale-copy partition is adapted from CUTLASS dense_blockscaled_gemm_persistent.py,
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Its BSD-3-Clause terms follow:
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Native block-scaled MXFP8 paged lightning-indexer scoring on SM100 and SM103.

Scores E4M3FN Q/K, dequantized by E8M0 scales per 32 channels, with the routing, bounds
and epilogue of :mod:`.paged_score_warpdrive`: each CTA owns a strided share of one query
row's M128 logical candidate tiles and writes the same FP32 slab and ``[0, visible)``
bounds. UMMA multiplies FP8 operands with ``kind::mxf8f6f4`` block scales; nothing is
dequantized to BF16.

``q_scale[t, h, g]`` scales channels ``32g .. 32g + 31`` of ``query[t, h]`` and
``k_scale[page, slot, g]`` those of ``key_cache[page, slot]``: K scales live on the same
physical pages as the keys. Roles and channels of one work item ``(c, row, 0)``::

    load (warp 5)   --q: 1 stage, TMA--------> mma (warp 4) --acc: 2 TMEM stages--> epilogue
                    --k: 3 stages, TMA------->              (0-3, inherited)
    scales (warp 6) --qsf: 1 stage, Warps---->
                    --ksf: 3 stages, Warps--->

``qsf`` carries one epoch per item with tiles, as ``q`` does; ``k``, ``ksf`` and ``acc``
one per visited tile. The scale warp looks each tile's pages up like the loader, clamping a
trailing page slot past the visible prefix to the last visible page, so it reads no unused
table entry. It packs four logical scale bytes per row into the native scale-factor layout;
rows past ``visible``, a page ID outside the cache and the heads padding N to 16 take the
neutral scale (E8M0 byte 127). Its ``Warps -> Umma`` handoff proxy-fences the generic stores.
MMA copies the Q scales into TMEM once per item and each tile's K scales into the TMEM
columns of that tile's accumulator stage, so acquiring the accumulator stage also orders the
K-scale overwrite after the previous MMA that read those columns.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float8E8M0FNU, Float32, Int32, Uint32, cute
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05

from attn_gym._backends.cute import get_device_properties
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
    Warps,
    WorkCursor,
    WorkStream,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, get_compile_target, jit_cache

from .paged_score_warpdrive import (
    _HEAD_TILE,
    _TILE_CANDIDATES,
    PagedScoreParams,
    PagedWarpdriveScore,
    _ctas_per_row,
)

_SCALE_GROUP = 32
_HEAD_DIM = 128  # four scale groups: one 32-bit scale word per row
_NEUTRAL_SCALES = 0x7F7F7F7F  # four E8M0 ones
_WEIGHT_DTYPES = {torch.bfloat16: "bfloat16", torch.float32: "float32"}


@dataclass(frozen=True)
class PagedMXFP8ScoreParams(PagedScoreParams):
    q_scale: cute.Tensor  # (T, H, 4) E8M0
    k_scale: cute.Tensor  # (pages, page_size, 4) E8M0
    sQsf: cute.Tensor
    sKsf: cute.Tensor


class PagedMXFP8Score(PagedWarpdriveScore):
    """Paged MXFP8 score kernel for one ``(weights dtype, heads, page_size)`` specialization.

    Inherits the BF16 kernel's row routing, K/Q loader and epilogue; adds the scale role and
    issues block-scaled MMAs. Q and K are compact E4M3FN with 16-byte-aligned bases, the
    scales compact E8M0 with 4-byte-aligned bases, and weights compact BF16 or FP32. D is 128.
    """

    name_prefix = "paged_mxfp8_score"
    sf_cols = 4

    def __init__(
        self,
        weight_dtype: str,
        heads: int,
        head_dim: int,
        page_size: int,
        explicit_positions: bool,
    ):
        if heads > _HEAD_TILE or heads % 4:
            raise ValueError(f"PagedMXFP8Score requires H in 4..{_HEAD_TILE}, H % 4 == 0")
        if head_dim != _HEAD_DIM:
            raise ValueError(f"PagedMXFP8Score requires D={_HEAD_DIM}")
        if page_size not in (32, 64, 128):
            raise ValueError("PagedMXFP8Score requires page_size 32, 64 or 128")
        if weight_dtype not in _WEIGHT_DTYPES.values():
            raise ValueError(f"PagedMXFP8Score weights must be BF16 or FP32, got {weight_dtype}")
        self.dtype = cutlass.Float8E4M3FN
        self.weight_dtype = cutlass.BFloat16 if weight_dtype == "bfloat16" else Float32
        self.heads = heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.explicit_positions = explicit_positions
        self.pages_per_tile = _TILE_CANDIDATES // page_size
        self.mma_tile = (_TILE_CANDIDATES, _HEAD_TILE, head_dim)
        self.smem_plan = SmemPlan()
        self.k = Ring(
            self.k_stages,
            Tma(_TILE_CANDIDATES * head_dim),
            Umma,
            buffer=self.smem_plan.buffer("K", self.dtype, align=1024),
        )
        self.q = Ring(
            self.q_stages,
            Tma(_HEAD_TILE * head_dim),
            Umma,
            buffer=self.smem_plan.buffer("Q", self.dtype, align=1024),
        )
        self.qsf = Ring(
            self.q_stages,
            Warps,
            Umma,
            buffer=self.smem_plan.buffer("Qsf", Float8E8M0FNU, align=128),
        )
        self.ksf = Ring(
            self.k_stages,
            Warps,
            Umma,
            buffer=self.smem_plan.buffer("Ksf", Float8E8M0FNU, align=128),
        )
        self.acc = Ring(self.acc_stages, Umma, Threads, release_fence="tmem_load")
        self.tmem_plan = (
            TmemPlan()
            .region("acc", _HEAD_TILE, self.acc_stages)
            .region("Ksf", self.sf_cols, self.acc_stages)
            .region("Qsf", self.sf_cols)
        )
        self.work = WorkStream(schedule=OneShot())

    def resident_ctas(self, device: torch.device) -> int:
        """CTAs that fit on the device at once, bounded by SMEM as in the BF16 kernel; every
        native scale stage spans 128 rows of 4 bytes, including Q's padded heads."""
        properties = get_device_properties(device)
        scale_bytes = _TILE_CANDIDATES * self.sf_cols
        cta_bytes = (
            self.k_stages * (_TILE_CANDIDATES * self.head_dim + scale_bytes)
            + _HEAD_TILE * self.head_dim
            + scale_bytes
            + 2048
        )
        per_sm = max(1, properties.shared_memory_per_multiprocessor // cta_bytes)
        return properties.multi_processor_count * per_sm

    def get_name(self) -> str:
        weights = "bf16" if self.weight_dtype is cutlass.BFloat16 else "fp32"
        return (
            f"{self.name_prefix}_w{weights}_h{self.heads}_d{self.head_dim}"
            f"_p{self.page_size}_pos{int(self.explicit_positions)}"
        )

    # ------------------------------------------------------------------ scales

    @role(warps=6)
    @cute.jit
    def scales(
        self,
        params: PagedMXFP8ScoreParams,
        *,
        qsf: Producer,
        ksf: Producer,
        work: WorkCursor,
        me: Local,
    ):
        """Pack each item's Q scales, then each tile's K scales, into native SF stages.

        Row ``32 * g + lane`` of a stage holds four group bytes at ``lane * 16 + g * 4``.
        """
        lane = me.lane
        tiles_per_batch = 32 // self.pages_per_tile
        q_words = cute.recast_tensor(params.q_scale, Uint32)  # (T, H, 1)
        k_words = cute.recast_tensor(params.k_scale, Uint32)  # (pages, page_size, 1)
        pages = params.k_scale.shape[0]
        while work.valid:
            share = self.row_work(params, work.item, lane)
            if share.tiles > 0:
                with qsf.produce() as h:
                    native = cute.make_tensor(
                        cute.recast_ptr(
                            h.view("Qsf", write=True, by="threads").iterator, dtype=Uint32
                        ),
                        (4, 32),
                    )
                    # H <= 16, so only the first row group holds heads.
                    value = Uint32(_NEUTRAL_SCALES)
                    if lane < self.heads:
                        value = q_words[share.row, lane, 0]
                    native[0, lane] = value
                    for row_group in cutlass.range_constexpr(1, 4):
                        native[row_group, lane] = Uint32(_NEUTRAL_SCALES)
                page = Int32(0)
                for i in cutlass.range(share.tiles, unroll=0):
                    slot = i % tiles_per_batch
                    if slot == 0:
                        page = self.batch_pages(params, share, i, lane, page)
                    tile = share.first + i * share.stride
                    with ksf.produce() as h:
                        native = cute.make_tensor(
                            cute.recast_ptr(
                                h.view("Ksf", write=True, by="threads").iterator, dtype=Uint32
                            ),
                            (4, 32),
                        )
                        for row_group in cutlass.range_constexpr(4):
                            row = 32 * row_group  # this lane's row is row + lane
                            physical = cute.arch.shuffle_sync(
                                page, slot * self.pages_per_tile + row // self.page_size
                            )
                            candidate = tile * _TILE_CANDIDATES + row + lane
                            value = Uint32(_NEUTRAL_SCALES)
                            if candidate < share.visible and physical >= 0 and physical < pages:
                                value = k_words[physical, (row + lane) % self.page_size, 0]
                            native[row_group, lane] = value
            work.advance()
        return qsf, ksf, work

    @cute.jit
    def copy_scales(self, source: cute.Tensor, destination: cute.Tensor):
        """Partition one native SF stage for a warp's SMEM-to-TMEM copy."""
        atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), Float8E8M0FNU)
        dest = cute.filter_zeros(destination)
        tiled_copy = tcgen05.make_s2t_copy(atom, dest)
        thread = tiled_copy.get_slice(0)
        source_desc = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy, thread.partition_S(cute.filter_zeros(source))
        )
        cute.copy(tiled_copy, source_desc, thread.partition_D(dest))

    # ------------------------------------------------------------------ roles

    @role(warps=4)
    @cute.jit
    def mma(
        self,
        params: PagedMXFP8ScoreParams,
        *,
        q: Consumer,
        k: Consumer,
        qsf: Consumer,
        ksf: Consumer,
        acc: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        """Issue four D32 block-scaled MMAs per tile into a two-stage FP32 TMEM ring."""
        tiled_mma = params.tiled_mma
        lane = cute.arch.lane_idx()
        accumulator = cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        fragment_k = tiled_mma.make_fragment_A(params.sK)
        q = q.views(Q=StageView(tiled_mma.make_fragment_B(params.sQ), axis=3))
        k = k.views(K=StageView(fragment_k, axis=3))
        k_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, self.mma_tile, _SCALE_GROUP, params.sKsf[(None, None, None, 0)].layout
        )
        q_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, self.mma_tile, _SCALE_GROUP, params.sQsf[(None, None, None, 0)].layout
        )
        ksf_base = tmem.ptr("Ksf")
        tQsf = cute.make_tensor(cute.recast_ptr(tmem.ptr("Qsf"), dtype=Float8E8M0FNU), q_layout)
        assert tcgen05.find_tmem_tensor_col_offset(tQsf) == self.sf_cols
        while work.valid:
            share = self.row_work(params, work.item, lane)
            if share.tiles > 0:
                q_full = q.wait()  # every K tile of the item multiplies this Q stage
                with qsf.consume() as scale:
                    self.copy_scales(scale.view("Qsf", read=True, by="umma"), tQsf)
                for _ in cutlass.range(share.tiles, unroll=0):
                    acc_empty = acc.acquire()
                    k_full = k.wait()
                    tKsf = cute.make_tensor(
                        cute.recast_ptr(
                            ksf_base + acc_empty.index * self.sf_cols, dtype=Float8E8M0FNU
                        ),
                        k_layout,
                    )
                    assert tcgen05.find_tmem_tensor_col_offset(tKsf) == self.sf_cols
                    with ksf.consume() as scale:
                        self.copy_scales(scale.view("Ksf", read=True, by="umma"), tKsf)
                    for d_block in cutlass.range_constexpr(cute.size(fragment_k, mode=[2])):
                        issue_mma = tiled_mma.with_()
                        issue_mma.set(tcgen05.Field.ACCUMULATE, d_block != 0)
                        issue_mma.set(tcgen05.Field.SFA, tKsf[(None, None, d_block)].iterator)
                        issue_mma.set(tcgen05.Field.SFB, tQsf[(None, None, d_block)].iterator)
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
        return q, k, qsf, ksf, acc, work

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
        q_scale: cute.Tensor,
        k_scale: cute.Tensor,
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
        self.begin_roles(
            work=bound,
            layouts={
                "K": k_layout,
                "Q": q_layout,
                "Qsf": blockscaled_utils.make_smem_layout_sfb(
                    tiled_mma, self.mma_tile, _SCALE_GROUP, self.q_stages
                ),
                "Ksf": blockscaled_utils.make_smem_layout_sfa(
                    tiled_mma, self.mma_tile, _SCALE_GROUP, self.k_stages
                ),
            },
        )
        sK = self.buffer("K")
        accumulator_shape = tiled_mma.partition_shape_C(self.mma_tile[:2])
        params = PagedMXFP8ScoreParams(
            tiled_mma=tiled_mma,
            tma_k=tma_atom_k,
            k=k,
            tma_q=tma_atom_q,
            q=q,
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
            q_scale=q_scale,
            k_scale=k_scale,
            sQsf=self.buffer("Qsf"),
            sKsf=self.buffer("Ksf"),
        )
        self.run_roles(params)

    @cute.jit
    def __call__(
        self,
        query: cute.Tensor,
        key_cache: cute.Tensor,
        weights: cute.Tensor,
        q_scale: cute.Tensor,
        k_scale: cute.Tensor,
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
        assert query.element_type == self.dtype and key_cache.element_type == self.dtype
        assert weights.element_type == self.weight_dtype
        assert q_scale.element_type == Float8E8M0FNU and k_scale.element_type == Float8E8M0FNU
        tiled_mma = cute.make_tiled_mma(
            tcgen05.MmaMXF8F6F4Op(
                self.dtype,
                self.dtype,
                (_TILE_CANDIDATES, _HEAD_TILE, _SCALE_GROUP),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                OperandMajorMode.K,
                OperandMajorMode.K,
            )
        )
        k_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tile, self.dtype, self.k_stages
        )
        # One 128-byte swizzle atom spans D, so each page slot's rows are contiguous
        # 1 KiB-per-8-row blocks of the UMMA A layout: one TMA destination per page.
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
        # TMA zero-fills heads H .. 15 of the N16 operand; their scales are neutral.
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
            q_scale,
            k_scale,
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
def compile_paged_mxfp8_score(operation: PagedMXFP8Score) -> Callable[..., None]:
    """Compile the constructor-keyed kernel with symbolic T, B, pages, capacity and slab rows."""
    sym = cute.sym_int
    tokens, sequences, cu_size, pages, table_pages, capacity, slab_rows = (sym() for _ in range(7))
    groups = operation.head_dim // _SCALE_GROUP

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
    heads, page_size = operation.heads, operation.page_size
    return compile_tvm_ffi(
        operation,
        compact(operation.dtype, (tokens, heads, operation.head_dim), 16),
        compact(operation.dtype, (pages, page_size, operation.head_dim), 16),
        compact(operation.weight_dtype, (tokens, heads), operation.weight_dtype.width // 8),
        compact(Float8E8M0FNU, (tokens, heads, groups), 4),
        compact(Float8E8M0FNU, (pages, page_size, groups), 4),
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


def get_paged_mxfp8_score_runner(
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
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    *,
    checked: bool = False,
) -> Callable[[int, int], None]:
    """Compile once and return ``run(start, rows)``, scoring global rows ``[start, start + rows)``.

    The slab, bounds and routing contract match ``get_paged_score_runner``. ``query`` and
    ``key_cache`` are E4M3FN; ``q_scale`` ``[T, H, D/32]`` and ``k_scale``
    ``[pages, page_size, D/32]`` are contiguous E8M0. Call inside the caller's compile-target
    scope, on the query's device and stream. ``checked`` builds warpdrive's bounds-checked
    variant, for tests. The caller validates dtypes, shapes, compactness, alignment and the
    SM100/SM103 device; the compiled ABI rejects other dtypes and ranks.
    """
    capacity = scores.shape[-1]
    operation = PagedMXFP8Score.get_or_build(
        _WEIGHT_DTYPES[weights.dtype],
        query.shape[1],
        query.shape[2],
        key_cache.shape[1],
        query_positions is not None,
        checked=checked,
    )
    compiled = compile_paged_mxfp8_score(operation)
    slab = scores.view(-1, capacity)
    query, key_cache, weights = query.detach(), key_cache.detach(), weights.detach()
    resident_ctas = operation.resident_ctas(query.device)

    def run(start: int, rows: int) -> None:
        compiled(
            query,
            key_cache,
            weights,
            q_scale,
            k_scale,
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


__all__ = ["PagedMXFP8Score", "compile_paged_mxfp8_score", "get_paged_mxfp8_score_runner"]
