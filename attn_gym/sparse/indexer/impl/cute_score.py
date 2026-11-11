# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by Attention Gym in 2026: warpdrive roles and channels.

"""Packed two-query SM100 score generation for bounded indexer score slabs.

Adapted from cuDNN Frontend's ``dense_score_recompute_sm100.py`` and
``indexer_score_unified_sm100.py``. K is the M128 A operand and two queries' heads are
packed into the B operand, so one FP32 MMA tile is [128, 2*H]. The epilogue computes
``scale * sum_h(weight * ReLU(Q @ K))`` for each query separately. This module only
generates scores; Top-K and the caller's [P, 2, S] FP32 slab are owned elsewhere.

Head, query and batch stay separate TMA modes despite the packing, so their strides are
independent and the missing query of an odd tail is zero-filled. Only valid rows and
candidates are written. The packed specialization reads each query's candidate interval from
``candidate_bounds[T, 2]`` and visits only the absolute M128 tiles covering the union of
the pair's nonempty intervals.

One work item is one slab pair::

    load (warp 5) --q: 1 stage, TMA --> mma (warp 4) --acc: 2 TMEM stages--> epilogue (0-3)
                  --k: 2 stages, TMA -->

Per item, ``q`` carries one epoch, and ``k`` and ``acc`` one per visited candidate tile:
none for an empty packed interval.
"""

import cutlass
from cutlass import Float32, cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.warpdrive import (
    Consumer,
    Local,
    NamedBarrier,
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


class IndexerScoreKernel(IndexerScoreKernelBase):
    """Packed score kernel for H=32/64, D=128 and FP16/BF16 inputs on SM100.

    Each CTA scores two adjacent queries against M128 candidate tiles. UMMA computes
    ``K[128,D] @ Q[2*H,D].T``; each of the 128 epilogue threads owns one candidate and
    reduces ``weight * ReLU(dot)`` for each query separately. Warp 5 loads Q once and
    streams K through two SMEM stages; warp 4 issues UMMA into a two-stage TMEM ring that
    warps 0--3 drain. Work is ``OneShot``: one item per slab pair, no persistent scheduler.

    Q/K need unit D strides and 16-byte-aligned bases and outer slice origins; their other
    strides are independent. Weights may have any strides and need only element alignment;
    ``contiguous_weight_heads`` records whether the caller's ABI promises a unit head
    stride. ``pair_start`` is a dynamic Int32 scalar, or Int64 with ``use_int64_offsets``,
    which also needs int64 dynamic shapes and strides in the caller's fake signature. Item
    ``p`` owns flattened pair ``pair_start + p`` and slab row ``p``; only existing pairs
    launch. Pair numbering restarts at each batch, so odd T ends a batch with an unpaired
    query.
    """

    name_prefix = "indexer_score"
    tile_candidates = 128
    k_stages = 2
    q_stages = 1

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
        if heads not in (32, 64) or head_dim != 128:
            raise ValueError("IndexerScoreKernel requires H=32/64 and D=128")
        self.heads = heads
        self.head_dim = head_dim
        self.causal = causal
        self.compress_ratio = compress_ratio
        self.use_int64_offsets = use_int64_offsets
        self.contiguous_weight_heads = contiguous_weight_heads
        self.packed_heads = 2 * heads
        self.mma_tile = (self.tile_candidates, self.packed_heads, head_dim)
        self.dtype = dtype
        operand_bytes = dtype.width // 8
        self.smem_plan = SmemPlan()
        k_buffer = self.smem_plan.buffer("K", dtype, align=1024)
        q_buffer = self.smem_plan.buffer("Q", dtype, align=1024)
        self.k = Ring(
            self.k_stages,
            Tma(self.tile_candidates * head_dim * operand_bytes),
            Umma,
            buffer=k_buffer,
        )
        self.q = Ring(
            self.q_stages,
            Tma(self.packed_heads * head_dim * operand_bytes),
            Umma,
            buffer=q_buffer,
        )
        # tcgen05.wait::ld before each release, so MMA cannot overwrite a stage still being read.
        self.acc = Ring(self.acc_stages, Umma, Threads, release_fence="tmem_load")
        self.tmem_plan = TmemPlan().region("acc", self.packed_heads, self.acc_stages)
        self.weights_ready = NamedBarrier(threads=128)  # epilogue sW stores before its loads
        self.work = WorkStream(schedule=OneShot())

    def q_operand(self, q: cute.Tensor) -> cute.Tensor:
        """Q as ((H, T), D, B): the MMA N mode packs (h, t) as ``h + H * t``.

        H and T keep their own strides and extents, so TMA zero-fills the missing query of an
        odd tail.
        """
        q_htdb = cute.make_tensor(q.iterator, cute.select(q.layout, mode=[2, 1, 3, 0]))
        return cute.group_modes(q_htdb, 0, 2)

    def work_space(self, pairs):
        """One item per slab pair."""
        return pairs, 1, 1

    def weight_stage(self, smem) -> cute.Tensor:
        """Both queries' weights, (2*H,) FP32."""
        return smem.allocate_tensor(Float32, cute.make_layout((self.packed_heads,)))

    @cute.jit
    def pair_tiles(self, candidate_bounds, query0, num_queries):
        """Absolute M128 tiles covering the pair's nonempty candidate intervals.

        An odd tail, inactive capacity or an empty document contributes no interval. A pair
        that crosses a document boundary visits the contiguous union of both ranges.
        """
        start0 = candidate_bounds[query0, 0]
        end0 = candidate_bounds[query0, 1]
        start1 = start0
        end1 = start0
        if query0 + 1 < num_queries:
            start1 = candidate_bounds[query0 + 1, 0]
            end1 = candidate_bounds[query0 + 1, 1]
        low = start0
        high = end0
        if end0 <= start0:
            low = start1
            high = end1
        elif end1 > start1:
            low = cutlass.min(start0, start1)
            high = cutlass.max(end0, end1)
        first_tile = low // self.tile_candidates
        tile_end = first_tile
        if high > low:
            tile_end = cute.ceil_div(high, self.tile_candidates)
        return (
            self.upcast_offset(first_tile),
            self.upcast_offset(tile_end),
            ((start0, end0), (start1, end1)),
        )

    @cute.jit
    def pair_work(self, params: ScoreParams, local_pair):
        """Decode slab pair ``local_pair``.

        Every role derives its trip counts from this, so each channel's producer and consumer
        agree on its epoch count.
        Returns ``(batch, batch_pair, query0, first_tile, tile_end, intervals)``; ``intervals``
        holds both queries' ``(start, end)`` candidate intervals in the packed specialization and
        is None otherwise.
        """
        num_queries = self.upcast_offset(params.weights.shape[1])
        pairs_per_batch = cute.ceil_div(num_queries, 2)
        global_pair = params.pair_start + self.upcast_offset(local_pair)
        batch = global_pair // pairs_per_batch
        batch_pair = global_pair % pairs_per_batch
        query0 = batch_pair * 2
        if cutlass.const_expr(params.candidate_bounds is not None):
            first_tile, tile_end, intervals = self.pair_tiles(
                params.candidate_bounds, query0, num_queries
            )
        else:
            query_last = query0 + 1 if query0 + 1 < num_queries else query0
            first_tile = self.upcast_offset(0)
            tile_end = self.upcast_offset(
                cute.ceil_div(self.visible_candidates(query_last), self.tile_candidates)
                if self.causal
                else cute.ceil_div(
                    self.upcast_offset(params.scores.shape[2]), self.tile_candidates
                )
            )
            intervals = None
        return batch, batch_pair, query0, first_tile, tile_end, intervals

    # ------------------------------------------------------------------ roles

    @role(warps=5)
    @cute.jit
    def load(self, params: ScoreParams, *, q: Producer, k: Producer, work: WorkCursor):
        """Load packed Q once per item, then stream its K tiles; TMA zero-fills the tails."""
        cpasync.prefetch_descriptor(params.tma_k)
        cpasync.prefetch_descriptor(params.tma_q)
        mma_zero = params.tiled_mma.get_slice(0)
        while work.valid:
            batch, batch_pair, _, first_tile, tile_end, _ = self.pair_work(params, work.item[0])
            gQ = cute.local_tile(
                params.q[None, None, batch], self.mma_tile, (0, batch_pair, 0), proj=(None, 1, 1)
            )
            tQsQ, tQgQ = cpasync.tma_partition(
                params.tma_q,
                0,
                cute.make_layout(1),
                cute.group_modes(params.sQ, 0, 3),
                cute.group_modes(mma_zero.partition_B(gQ), 0, 3),
            )
            # All candidate tiles of this batch element: (128, D, candidate tiles).
            gK = cute.local_tile(
                params.k[None, None, batch], self.mma_tile, (None, 0, 0), proj=(1, None, 1)
            )
            tKsK, tKgK = cpasync.tma_partition(
                params.tma_k,
                0,
                cute.make_layout(1),
                cute.group_modes(params.sK, 0, 3),
                cute.group_modes(mma_zero.partition_A(gK), 0, 3),
            )
            q = q.views(Q=StageView(tQsQ, axis=1))
            k = k.views(K=StageView(tKsK, axis=1))
            with q.produce() as h:
                cute.copy(params.tma_q, tQgQ, h.view("Q"), tma_bar_ptr=h.barrier)
            for candidate_tile in cutlass.range(first_tile, tile_end, unroll=0):
                with k.produce() as h:
                    cute.copy(
                        params.tma_k,
                        tKgK[None, candidate_tile],
                        h.view("K"),
                        tma_bar_ptr=h.barrier,
                    )
            work.advance()
        return q, k, work

    @role(warps=4)
    @cute.jit
    def mma(
        self,
        params: ScoreParams,
        *,
        q: Consumer,
        k: Consumer,
        acc: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        """Compute [128, D] @ [2*H, D].T into a two-stage FP32 TMEM ring."""
        tiled_mma = params.tiled_mma
        accumulator = cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        fragment_k = tiled_mma.make_fragment_A(params.sK)
        q = q.views(Q=StageView(tiled_mma.make_fragment_B(params.sQ), axis=3))
        k = k.views(K=StageView(fragment_k, axis=3))
        while work.valid:
            _, _, _, first_tile, tile_end, _ = self.pair_work(params, work.item[0])
            q_full = q.wait()  # every K tile of the item multiplies this Q stage
            for _candidate_tile in cutlass.range(first_tile, tile_end, unroll=0):
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
        params: ScoreParams,
        *,
        acc: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        """Load each thread's candidate row of 2*H logits and reduce each query separately."""
        accumulator, tmem_copy, tmem_thread, tile_candidate, logits = self.accumulator_reader(
            params, tmem, me
        )
        rW = cute.make_rmem_tensor((self.packed_heads,), Float32)
        num_queries = self.upcast_offset(params.weights.shape[1])
        num_candidates = self.upcast_offset(params.scores.shape[2])
        while work.valid:
            local_pair = work.item[0]
            batch, _, query0, first_tile, tile_end, intervals = self.pair_work(params, local_pair)
            # OneShot gives each CTA one item, so nothing rewrites sW after this sync.
            if me.tid < self.packed_heads:
                query = query0 + me.tid // self.heads
                value = Float32(0.0)
                if query < num_queries:
                    value = Float32(params.weights[batch, query, me.tid % self.heads])
                params.sW[me.tid] = value
            self.weights_ready.sync()
            cute.autovec_copy(params.sW, rW)

            for candidate_tile in cutlass.range(first_tile, tile_end, unroll=0):
                acc_full = acc.wait()
                cute.copy(
                    tmem_copy,
                    tmem_thread.partition_S(accumulator[(None, None, None, acc_full.index)]),
                    logits,
                )
                acc_full.release()
                candidate = (
                    self.upcast_offset(candidate_tile) * self.tile_candidates + tile_candidate
                )
                for qi in cutlass.range_constexpr(2):
                    query = query0 + qi
                    valid = query < num_queries and candidate < num_candidates
                    if cutlass.const_expr(intervals is not None):
                        start, end = intervals[qi]
                        valid = valid and candidate >= start and candidate < end
                    elif cutlass.const_expr(self.causal):
                        valid = valid and candidate < self.visible_candidates(query)
                    if valid:
                        sums = self.add_weighted_relu(
                            (Float32(0.0),) * 4, logits, rW, qi * self.heads, self.heads
                        )
                        params.scores[self.upcast_offset(local_pair), qi, candidate] = (
                            (sums[0] + sums[1]) + (sums[2] + sums[3])
                        ) * params.score_scale
            work.advance()
        return acc, work
