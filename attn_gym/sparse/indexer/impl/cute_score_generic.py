# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by Attention Gym in 2026: bounded score slabs, general head/dimension tiling, and
# warpdrive roles and channels.

"""Tiled SM100 indexer score generation without shape-sized shared buffers.

One CTA owns one query of the caller's [P, 2, S] FP32 score slab. Candidate, head and
reduction tiles bound SMEM and TMEM independently of H and D. Head, query and batch stay
separate TMA modes, so tails zero-fill without reading another query or batch. This module
allocates no global workspace and selects nothing; masked and inactive slab entries stay
untouched. The packed entrypoint visits only the absolute M128 tiles covering each query's
``candidate_bounds[T, 2]`` interval.

One work item is one query of a slab pair::

    load (warp 5) --ab: 2 stages of K + Q, TMA--> mma (warp 4) --acc: 2 TMEM stages--> epilogue
                                                                                        (0-3)

Per item, ``ab`` carries ``candidate tiles * head tiles * D tiles`` epochs and ``acc``
``candidate tiles * head tiles``. The missing second query of an odd tail visits no tiles.
"""

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.warpdrive import (
    Consumer,
    Local,
    OneShot,
    Producer,
    ProtocolError,
    Ring,
    SmemPlan,
    StageView,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    role,
)

from .cute_kernel_base import IndexerScoreKernelBase


@dataclass(frozen=True)
class GenericScoreParams:
    """Kernel-prologue values the roles read."""

    tiled_mma: cute.TiledMma
    tma_k: cute.CopyAtom
    k: cute.Tensor  # K TMA coordinates: (S, D, B)
    tma_q: cute.CopyAtom
    q: cute.Tensor  # Q TMA coordinates: (H, D, T, B)
    sK: cute.Tensor
    sQ: cute.Tensor
    acc_layout: cute.Layout  # every accumulator stage
    weights: cute.Tensor
    scores: cute.Tensor
    pair_start: cutlass.Numeric
    score_scale: Float32
    candidate_bounds: cute.Tensor | None


class IndexerGenericScoreKernel(IndexerScoreKernelBase, WarpSpecialized):
    """Tiled score kernel for positive even H and positive D divisible by 16 on SM100.

    For each M128 candidate tile and N64 head tile, UMMA accumulates
    ``L[m,h] = sum_d K[m,d] * Q[h,d]`` in FP32 over K64 chunks. Only once D is complete
    does the epilogue add ``weight[h] * max(L[m,h], 0)`` across head tiles; it multiplies
    the final FP32 sum by ``score_scale``.

    Warps 0--3 own 32 candidates each, warp 4 issues UMMA, and warp 5 TMA-loads both
    operands. Two combined operand stages feed a two-stage accumulator ring. Q/K need unit
    D strides and 16-byte-aligned bases and outer slice origins; their other strides are
    independent. Weights may be strided and need only element alignment;
    ``contiguous_weight_heads`` records a promised unit head stride. The caller supplies
    an Int32 ``pair_start`` and dynamic shapes and strides, or Int64 equivalents for wide
    offsets. Item ``(qi, p, 0)`` owns query ``qi`` of flattened pair ``pair_start + p`` and
    slab row ``p``; only existing pairs launch.
    """

    name_prefix = "indexer_score_generic"
    tile_candidates = 128
    tile_heads = 64
    tile_dim = 64
    ab_stages = 2
    acc_stages = 2

    def __init__(
        self,
        heads: int,
        head_dim: int,
        causal: bool,
        use_int64_offsets: bool = False,
        *,
        compress_ratio: int = 1,
        contiguous_weight_heads: bool,
        dtype: type[cutlass.Numeric],
    ):
        """``dtype`` is the Q, K and weights element type: FP16 or BF16."""
        if heads <= 0 or heads % 2 != 0:
            raise ValueError("IndexerGenericScoreKernel requires positive even H")
        if head_dim <= 0 or head_dim % 16 != 0:
            raise ValueError("IndexerGenericScoreKernel requires positive D divisible by 16")
        self.heads = heads
        self.head_dim = head_dim
        self.causal = causal
        self.compress_ratio = compress_ratio
        self.use_int64_offsets = use_int64_offsets
        self.contiguous_weight_heads = contiguous_weight_heads
        self.dtype = dtype
        self.head_tiles = (heads + self.tile_heads - 1) // self.tile_heads
        self.d_tiles = (head_dim + self.tile_dim - 1) // self.tile_dim
        self.mma_tile = (self.tile_candidates, self.tile_heads, self.tile_dim)
        self.smem_plan = SmemPlan()
        k_buffer = self.smem_plan.buffer("K", dtype, align=1024)
        q_buffer = self.smem_plan.buffer("Q", dtype, align=1024)
        stage_bytes = (self.tile_candidates + self.tile_heads) * self.tile_dim * dtype.width // 8
        self.ab = Ring(
            self.ab_stages, Tma(stage_bytes), Umma, buffer={"K": k_buffer, "Q": q_buffer}
        )
        # tcgen05.wait::ld before each release, so MMA cannot overwrite a stage still being read.
        self.acc = Ring(self.acc_stages, Umma, Threads, release_fence="tmem_load")
        self.tmem_plan = TmemPlan().region("acc", self.tile_heads, self.acc_stages)
        self.work = WorkStream(schedule=OneShot())  # item (qi, slab pair, 0), qi fastest

    @cute.jit
    def query_work(self, params: GenericScoreParams, item):
        """Decode work item ``(qi, local_pair, 0)``.

        Every role derives its trip counts from this, so each channel's producer and consumer
        agree on its epoch count.
        Returns ``(batch, query, first_tile, tile_end, interval)``; ``interval`` is the query's
        ``(start, end)`` candidate interval in the packed entrypoint and None otherwise. The
        missing second query of an odd tail visits no tiles.
        """
        qi, local_pair, _ = item
        num_queries = self.upcast_offset(params.weights.shape[1])
        pairs_per_batch = cute.ceil_div(num_queries, 2)
        global_pair = params.pair_start + self.upcast_offset(local_pair)
        batch = global_pair // pairs_per_batch
        query = (global_pair % pairs_per_batch) * 2 + qi
        first_tile = self.upcast_offset(0)
        tile_end = self.upcast_offset(0)
        start = Int32(0)
        end = Int32(0)
        if query < num_queries:
            if cutlass.const_expr(params.candidate_bounds is not None):
                start = params.candidate_bounds[query, 0]
                end = params.candidate_bounds[query, 1]
                first_tile = self.upcast_offset(start // self.tile_candidates)
                tile_end = first_tile
                if end > start:
                    tile_end = self.upcast_offset(cute.ceil_div(end, self.tile_candidates))
            else:
                candidate_end = (
                    self.visible_candidates(query)
                    if self.causal
                    else self.upcast_offset(params.scores.shape[2])
                )
                tile_end = self.upcast_offset(cute.ceil_div(candidate_end, self.tile_candidates))
        interval = None
        if cutlass.const_expr(params.candidate_bounds is not None):
            interval = (start, end)
        return batch, query, first_tile, tile_end, interval

    # ------------------------------------------------------------------ roles

    @role(warps=5)
    @cute.jit
    def load(self, params: GenericScoreParams, *, ab: Producer, work: WorkCursor):
        """Stream K and Q tiles into shared stages; TMA zero-fills the head and D tails."""
        cpasync.prefetch_descriptor(params.tma_k)
        cpasync.prefetch_descriptor(params.tma_q)
        mma_zero = params.tiled_mma.get_slice(0)
        while work.valid:
            batch, query, first_tile, tile_end, _ = self.query_work(params, work.item)
            # K tiles of this batch element: (128, 64, candidate tiles, D tiles).
            # Q tiles of this query: (64, 64, head tiles, D tiles).
            gK = cute.local_tile(
                params.k[None, None, batch], self.mma_tile, (None, 0, None), proj=(1, None, 1)
            )
            gQ = cute.local_tile(
                params.q[None, None, query, batch],
                self.mma_tile,
                (0, None, None),
                proj=(None, 1, 1),
            )
            tKsK, tKgK = cpasync.tma_partition(
                params.tma_k,
                0,
                cute.make_layout(1),
                cute.group_modes(params.sK, 0, 3),
                cute.group_modes(mma_zero.partition_A(gK), 0, 3),
            )
            tQsQ, tQgQ = cpasync.tma_partition(
                params.tma_q,
                0,
                cute.make_layout(1),
                cute.group_modes(params.sQ, 0, 3),
                cute.group_modes(mma_zero.partition_B(gQ), 0, 3),
            )
            ab = ab.views(K=StageView(tKsK, axis=1), Q=StageView(tQsQ, axis=1))
            for candidate_tile in cutlass.range(first_tile, tile_end, unroll=0):
                for head_tile in cutlass.range(self.head_tiles, unroll=0):
                    for d_tile in cutlass.range(self.d_tiles, unroll=0):
                        with ab.produce() as h:
                            cute.copy(
                                params.tma_k,
                                tKgK[None, candidate_tile, self.upcast_offset(d_tile)],
                                h.view("K"),
                                tma_bar_ptr=h.barrier,
                            )
                            cute.copy(
                                params.tma_q,
                                tQgQ[
                                    None,
                                    self.upcast_offset(head_tile),
                                    self.upcast_offset(d_tile),
                                ],
                                h.view("Q"),
                                tma_bar_ptr=h.barrier,
                            )
            work.advance()
        return ab, work

    @role(warps=4)
    @cute.jit
    def mma(
        self,
        params: GenericScoreParams,
        *,
        ab: Consumer,
        acc: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        """Complete D before publishing each [128, 64] FP32 accumulator tile."""
        tiled_mma = params.tiled_mma
        accumulator = cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        fragment_k = tiled_mma.make_fragment_A(params.sK)
        ab = ab.views(
            K=StageView(fragment_k, axis=3),
            Q=StageView(tiled_mma.make_fragment_B(params.sQ), axis=3),
        )
        while work.valid:
            _, _, first_tile, tile_end, _ = self.query_work(params, work.item)
            for _candidate_tile in cutlass.range(first_tile, tile_end, unroll=0):
                for _head_tile in cutlass.range(self.head_tiles, unroll=0):
                    with acc.produce() as acc_empty:
                        for d_tile in cutlass.range(self.d_tiles, unroll=0):
                            with ab.consume() as ab_full:
                                for d_block in cutlass.range_constexpr(
                                    cute.size(fragment_k, mode=[2])
                                ):
                                    issue_mma = tiled_mma.with_()
                                    issue_mma.set(
                                        tcgen05.Field.ACCUMULATE,
                                        cutlass.Boolean(d_tile != Int32(0) or d_block != 0),
                                    )
                                    cute.gemm(
                                        issue_mma,
                                        accumulator[(None, None, None, acc_empty.index)],
                                        ab_full.view("K", None, None, d_block),
                                        ab_full.view("Q", None, None, d_block),
                                        accumulator[(None, None, None, acc_empty.index)],
                                    )
            work.advance()
        return ab, acc, work

    @role(warps=range(4))
    @cute.jit
    def epilogue(
        self,
        params: GenericScoreParams,
        *,
        acc: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        """Keep one candidate's FP32 head sum in registers until its final slab store."""
        if cutlass.const_expr(me.warps != 4):
            raise ProtocolError(
                f"epilogue: one thread per M128 candidate needs 4 warps, got {me.warps}"
            )
        tidx = me.tmem_row
        tiled_mma = params.tiled_mma
        accumulator = cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        acc_tile = accumulator[(None, None, None, 0)]
        tmem_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.tile_heads // 4)),
            Float32,
        )
        tmem_copy = tcgen05.make_tmem_copy(tmem_atom, acc_tile)
        tmem_thread = tmem_copy.get_slice(tidx)
        coordinates = tmem_thread.partition_D(
            tiled_mma.get_slice(tidx).partition_C(cute.make_identity_tensor(self.mma_tile[:2]))
        )
        logits = cute.make_rmem_tensor(
            tmem_thread.partition_D(cute.make_identity_tensor(acc_tile.shape)).shape, Float32
        )
        assert cute.size(logits) == self.tile_heads
        rW = cute.make_rmem_tensor((self.tile_heads,), Float32)
        num_candidates = self.upcast_offset(params.scores.shape[2])
        while work.valid:
            qi, local_pair, _ = work.item
            batch, query, first_tile, tile_end, interval = self.query_work(params, work.item)
            weights = params.weights[batch, query, None]
            for candidate_tile in cutlass.range(first_tile, tile_end, unroll=0):
                candidate = (
                    self.upcast_offset(candidate_tile) * self.tile_candidates + coordinates[0][0]
                )
                valid = candidate < num_candidates
                if cutlass.const_expr(interval is not None):
                    valid = valid and candidate >= interval[0] and candidate < interval[1]
                elif cutlass.const_expr(self.causal):
                    valid = valid and candidate < self.visible_candidates(query)
                sum0 = Float32(0.0)
                sum1 = Float32(0.0)
                sum2 = Float32(0.0)
                sum3 = Float32(0.0)
                for head_tile in cutlass.range(self.head_tiles, unroll=0):
                    acc_full = acc.wait()
                    cute.copy(
                        tmem_copy,
                        tmem_thread.partition_S(accumulator[(None, None, None, acc_full.index)]),
                        logits,
                    )
                    acc_full.release()
                    # Invalid lanes still wait and release: the acc release counts all 128.
                    if valid:
                        for offset in cutlass.range_constexpr(self.tile_heads):
                            head = self.upcast_offset(head_tile) * self.tile_heads + offset
                            weight = Float32(0.0)
                            if head < self.heads:
                                weight = Float32(weights[head])
                            rW[offset] = weight
                        for group in cutlass.range_constexpr(self.tile_heads // 4):
                            offset = group * 4
                            x0 = Float32(logits[offset])
                            x1 = Float32(logits[offset + 1])
                            x2 = Float32(logits[offset + 2])
                            x3 = Float32(logits[offset + 3])
                            x0 = x0 if x0 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
                            x1 = x1 if x1 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
                            x2 = x2 if x2 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
                            x3 = x3 if x3 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
                            sum0 = sum0 + x0 * rW[offset]
                            sum1 = sum1 + x1 * rW[offset + 1]
                            sum2 = sum2 + x2 * rW[offset + 2]
                            sum3 = sum3 + x3 * rW[offset + 3]
                if valid:
                    params.scores[self.upcast_offset(local_pair), qi, candidate] = (
                        (sum0 + sum1) + (sum2 + sum3)
                    ) * params.score_scale
            work.advance()
        return acc, work

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_k: cute.CopyAtom,
        k: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        q: cute.Tensor,
        weights: cute.Tensor,
        scores: cute.Tensor,
        k_layout: cute.ComposedLayout,
        q_layout: cute.ComposedLayout,
        pair_start,
        score_scale: Float32,
        candidate_bounds: cute.Tensor | None,
        bound,
    ):
        """Bind the K/Q buffers, then run the roles."""
        self.begin_roles(work=bound, layouts={"K": k_layout, "Q": q_layout})
        accumulator_shape = tiled_mma.partition_shape_C(self.mma_tile[:2])
        params = GenericScoreParams(
            tiled_mma=tiled_mma,
            tma_k=tma_atom_k,
            k=k,
            tma_q=tma_atom_q,
            q=q,
            sK=self.buffer("K"),
            sQ=self.buffer("Q"),
            acc_layout=tiled_mma.make_fragment_C(
                cute.append(accumulator_shape, self.acc_stages)
            ).layout,
            weights=weights,
            scores=scores,
            pair_start=pair_start,
            score_scale=score_scale,
            candidate_bounds=candidate_bounds,
        )
        self.run_roles(params)

    # ------------------------------------------------------------------- host

    @cute.jit
    def launch(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        weights: cute.Tensor,
        scores: cute.Tensor,
        pair_start,
        score_scale: Float32,
        candidate_bounds: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        """Build the TMA descriptors and launch two CTAs per existing slab pair."""
        assert q.element_type == self.dtype
        assert k.element_type == q.element_type and weights.element_type == q.element_type
        assert scores.element_type == Float32
        q_hdtb = cute.make_tensor(q.iterator, cute.select(q.layout, mode=[2, 3, 1, 0]))
        k_sdb = cute.make_tensor(k.iterator, cute.select(k.layout, mode=[1, 2, 0]))
        tiled_mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                q.element_type,
                Float32,
                (self.tile_candidates, self.tile_heads, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
            )
        )
        k_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tile, k.element_type, self.ab.n
        )
        q_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tile, q.element_type, self.ab.n
        )
        tma_load = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_k, k_tma = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load,
            k_sdb,
            cute.select(k_layout, mode=[0, 1, 2]),
            self.mma_tile,
            tiled_mma,
        )
        tma_atom_q, q_tma = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load,
            q_hdtb,
            cute.select(q_layout, mode=[0, 1, 2]),
            self.mma_tile,
            tiled_mma,
        )
        pair_start = self.upcast_offset(pair_start)
        total_pairs = self.upcast_offset(weights.shape[0]) * cute.ceil_div(
            self.upcast_offset(weights.shape[1]), 2
        )
        # The public launcher slices the slab to the remaining pairs. A longer slab from a
        # direct caller keeps its extra rows untouched.
        pairs = cutlass.min(self.upcast_offset(scores.shape[0]), total_pairs - pair_start)
        bound = self.work.bind((2, cutlass.Int32(pairs), 1))
        packed = "_packed" if cutlass.const_expr(candidate_bounds is not None) else ""
        self.kernel.set_name_prefix(f"{self.get_name()}_{q.element_type.__name__.lower()}{packed}")
        self.kernel(
            tiled_mma,
            tma_atom_k,
            k_tma,
            tma_atom_q,
            q_tma,
            weights,
            scores,
            k_layout,
            q_layout,
            pair_start,
            score_scale,
            candidate_bounds,
            bound,
        ).launch(**self.launch_config(bound), stream=stream)
