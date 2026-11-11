# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by Attention Gym in 2026: bounded score slabs, general head/dimension tiling, and
# warpdrive roles and channels.

"""Tiled SM100 indexer score generation without shape-sized shared buffers.

One CTA owns one query of the caller's [P, 2, S] FP32 score slab. Candidate, head and
reduction tiles bound SMEM and TMEM independently of H and D. Head, query and batch stay
separate TMA modes, so tails zero-fill without reading another query or batch. This module
allocates no global workspace and selects nothing; masked and inactive slab entries stay
untouched. The packed specialization visits only the absolute M128 tiles covering each query's
``candidate_bounds[T, 2]`` interval.

One work item is one query of a slab pair::

    load (warp 5) --ab: 2 stages of K + Q, TMA--> mma (warp 4) --acc: 2 TMEM stages--> epilogue
                                                                                        (0-3)

Per item, ``ab`` carries ``candidate tiles * head tiles * D tiles`` epochs and ``acc``
``candidate tiles * head tiles``. The missing second query of an odd tail visits no tiles.
"""

import cutlass
from cutlass import Float32, Int32, cute
from cutlass.cute.nvgpu import cpasync, tcgen05

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

from .cute_kernel_base import IndexerScoreKernelBase, ScoreParams


class IndexerGenericScoreKernel(IndexerScoreKernelBase):
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
    k_stages = q_stages = 2  # one ring stages both operands

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
            self.k_stages, Tma(stage_bytes), Umma, buffer={"K": k_buffer, "Q": q_buffer}
        )
        # tcgen05.wait::ld before each release, so MMA cannot overwrite a stage still being read.
        self.acc = Ring(self.acc_stages, Umma, Threads, release_fence="tmem_load")
        self.tmem_plan = TmemPlan().region("acc", self.tile_heads, self.acc_stages)
        self.work = WorkStream(schedule=OneShot())

    def q_operand(self, q: cute.Tensor) -> cute.Tensor:
        """Q as (H, D, T, B)."""
        return cute.make_tensor(q.iterator, cute.select(q.layout, mode=[2, 3, 1, 0]))

    def work_space(self, pairs):
        """Item ``(qi, slab pair, 0)``, ``qi`` fastest."""
        return 2, pairs, 1

    @cute.jit
    def query_work(self, params: ScoreParams, item):
        """Decode work item ``(qi, local_pair, 0)``.

        Every role derives its trip counts from this, so each channel's producer and consumer
        agree on its epoch count.
        Returns ``(batch, query, first_tile, tile_end, interval)``; ``interval`` is the query's
        ``(start, end)`` candidate interval in the packed specialization and None otherwise. The
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
    def load(self, params: ScoreParams, *, ab: Producer, work: WorkCursor):
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
        params: ScoreParams,
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
        params: ScoreParams,
        *,
        acc: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        """Keep one candidate's FP32 head sum in registers until its final slab store."""
        accumulator, tmem_copy, tmem_thread, tile_candidate, logits = self.accumulator_reader(
            params, tmem, me
        )
        # With one head tile, the columns past H hold logits of TMA zero-filled Q rows: their
        # ReLU is +0 against weight 0, so each adds +0 to its sum. They are left unloaded, which
        # frees their registers, and one +0 added to the total keeps it bitwise equal: the only
        # value a +0 addition changes is -0 (from an underflowing product), and a total of -0
        # would otherwise survive.
        live_heads = min(cute.round_up(self.heads, 4), self.tile_heads)
        live_loads = cute.ceil_div(live_heads, cute.size(logits, mode=[0]))
        rW = cute.make_rmem_tensor((self.tile_heads,), Float32)
        num_candidates = self.upcast_offset(params.scores.shape[2])
        while work.valid:
            qi, local_pair, _ = work.item
            batch, query, first_tile, tile_end, interval = self.query_work(params, work.item)
            weights = params.weights[batch, query, None]
            for candidate_tile in cutlass.range(first_tile, tile_end, unroll=0):
                candidate = (
                    self.upcast_offset(candidate_tile) * self.tile_candidates + tile_candidate
                )
                valid = candidate < num_candidates
                if cutlass.const_expr(interval is not None):
                    valid = valid and candidate >= interval[0] and candidate < interval[1]
                elif cutlass.const_expr(self.causal):
                    valid = valid and candidate < self.visible_candidates(query)
                sums = (Float32(0.0),) * 4
                for head_tile in cutlass.range(self.head_tiles, unroll=0):
                    acc_full = acc.wait()
                    tmem_stage = tmem_thread.partition_S(
                        accumulator[(None, None, None, acc_full.index)]
                    )
                    if cutlass.const_expr(live_heads == self.tile_heads):
                        cute.copy(tmem_copy, tmem_stage, logits)
                    else:
                        for load in cutlass.range_constexpr(live_loads):
                            cute.copy(
                                tmem_copy,
                                tmem_stage[None, load, None, None],
                                logits[None, load, None, None],
                            )
                    acc_full.release()
                    # Invalid lanes still wait and release: the acc release counts all 128.
                    if valid:
                        for offset in cutlass.range_constexpr(live_heads):
                            head = self.upcast_offset(head_tile) * self.tile_heads + offset
                            weight = Float32(0.0)
                            if head < self.heads:
                                weight = Float32(weights[head])
                            rW[offset] = weight
                        sums = self.add_weighted_relu(sums, logits, rW, 0, live_heads)
                if valid:
                    total = (sums[0] + sums[1]) + (sums[2] + sums[3])
                    if cutlass.const_expr(live_heads < self.tile_heads):
                        total = total + Float32(0.0)
                    params.scores[self.upcast_offset(local_pair), qi, candidate] = (
                        total * params.score_scale
                    )
            work.advance()
        return acc, work
