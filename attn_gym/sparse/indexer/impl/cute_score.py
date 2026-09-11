# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed two-query SM100 score generation for bounded indexer score slabs.

Adapted from cuDNN Frontend's ``dense_score_recompute_sm100.py`` and
``indexer_score_unified_sm100.py``: K is the M128 A operand, while two queries'
heads are packed into the B operand. One FP32 MMA tile has shape [128, 2*H];
the epilogue independently computes ``scale * sum_h(weight * ReLU(Q @ K))``
for each query. This module owns only score generation, not Top-K or allocation
of the caller's [P, 2, T] FP32 slab.

Head, query, and batch remain separate TMA modes despite the logical packing,
so their physical strides are independent and odd query tails are zero-filled.
Only valid rows/candidates are written.
"""

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05


class IndexerScoreKernel:
    """Generate a slab of scores for H=32/64, D=128, FP16/BF16 inputs on SM100.

    Each CTA scores two adjacent queries against M128 candidate tiles. UMMA
    computes ``K[128,D] @ Q[2*H,D].T``; each of the 128 epilogue threads owns
    one candidate and separately reduces ``weight * ReLU(dot)`` for each query.
    Warp 5 loads Q once and streams K through two shared-memory stages; warp 4
    issues UMMA into a two-stage TMEM ring consumed by warps 0--3. Warps 6--7
    participate only in CTA synchronization. There is no persistent scheduler.

    Q/K require unit D strides and 16-byte-aligned bases and outer slice origins;
    their remaining strides are independent. Weights may have arbitrary strides
    and require only element alignment; ``contiguous_weight_heads`` records whether
    the caller's ABI promises a unit head stride. ``pair_start`` is a dynamic Int32
    scalar, or Int64 when ``use_int64_offsets`` is enabled; the latter also requires
    int64 dynamic shapes/strides in the caller's fake signature. Each CTA owns
    flattened pair ``pair_start + blockIdx.x``. Pair numbering restarts at each
    batch boundary, including an unpaired final query for odd T.
    """

    tile_candidates = 128
    k_stages = 2
    acc_stages = 2
    epilogue_threads = 128
    mma_warp = 4
    load_warp = 5
    threads = 256

    def __init__(
        self,
        heads: int,
        head_dim: int,
        causal: bool,
        use_int64_offsets: bool = False,
        *,
        contiguous_weight_heads: bool,
    ):
        if heads not in (32, 64) or head_dim != 128:
            raise ValueError("IndexerScoreKernel requires H=32/64 and D=128")
        self.heads = heads
        self.head_dim = head_dim
        self.causal = causal
        self.use_int64_offsets = use_int64_offsets
        self.contiguous_weight_heads = contiguous_weight_heads
        self.packed_heads = 2 * heads
        self.mma_tile = (self.tile_candidates, self.packed_heads, head_dim)

        @cute.struct
        class SharedStorage:
            k_barriers: cute.struct.MemRange[Int64, self.k_stages * 2]
            q_barriers: cute.struct.MemRange[Int64, 2]
            acc_barriers: cute.struct.MemRange[Int64, self.acc_stages * 2]
            tmem_holding: Int32

        self.SharedStorage = SharedStorage

    def get_name(self) -> str:
        """Return a stable name for the shape, mask, weight layout, and offset width."""
        return (
            f"indexer_score_h{self.heads}_d{self.head_dim}_c{int(self.causal)}_"
            f"i64{int(self.use_int64_offsets)}_wh{int(self.contiguous_weight_heads)}"
        )

    @cute.jit
    def upcast_offset(self, value):
        """Widen origins before address arithmetic in the wide specialization."""
        return Int64(value) if cutlass.const_expr(self.use_int64_offsets) else Int32(value)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        weights: cute.Tensor,
        scores: cute.Tensor,
        pair_start,
        score_scale: Float32,
        stream: cuda.CUstream,
    ):
        """Build TMA descriptors and launch one CTA per output-slab query pair."""
        assert q.element_type in (cutlass.Float16, cutlass.BFloat16)
        assert k.element_type == q.element_type and weights.element_type == q.element_type
        assert scores.element_type == Float32

        # The logical N coordinate is h + H*t, but H/T retain independent physical strides.
        # TMA sees their separate extents, including the zero-filled query of an odd tail.
        q_htdb = cute.make_tensor(q.iterator, cute.select(q.layout, mode=[2, 1, 3, 0]))
        q_packed = cute.group_modes(q_htdb, 0, 2)
        k_sdb = cute.make_tensor(k.iterator, cute.select(k.layout, mode=[1, 2, 0]))
        tiled_mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                q.element_type,
                Float32,
                (self.tile_candidates, self.packed_heads, 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
            )
        )
        k_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tile, k.element_type, self.k_stages
        )
        q_layout = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tile, q.element_type, 1)
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
            q_packed,
            cute.select(q_layout, mode=[0, 1, 2]),
            self.mma_tile,
            tiled_mma,
        )
        self.kernel.set_name_prefix(f"{self.get_name()}_{q.element_type.__name__.lower()}")
        self.kernel(
            tiled_mma,
            tma_atom_k,
            k_tma,
            tma_atom_q,
            q_tma,
            weights,
            scores,
            q.element_type,
            k_layout,
            q_layout,
            self.upcast_offset(pair_start),
            score_scale,
        ).launch(grid=(scores.shape[0], 1, 1), block=(self.threads, 1, 1), stream=stream)

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
        io_dtype: cutlass.Constexpr,
        k_layout: cute.ComposedLayout,
        q_layout: cute.ComposedLayout,
        pair_start,
        score_scale: Float32,
    ):
        """Initialize stage ownership, dispatch warp roles, and retire TMEM."""
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        local_pair, _, _ = cute.arch.block_idx()
        num_queries = self.upcast_offset(weights.shape[1])
        pairs_per_batch = cute.ceil_div(num_queries, 2)
        global_pair = pair_start + self.upcast_offset(local_pair)
        batch = global_pair // pairs_per_batch
        batch_pair = global_pair % pairs_per_batch
        query0 = batch_pair * 2
        active = global_pair < self.upcast_offset(weights.shape[0]) * pairs_per_batch
        query_last = query0 + 1 if query0 + 1 < num_queries else query0
        candidate_tiles = (
            cute.ceil_div(query_last + 1, self.tile_candidates)
            if self.causal
            else cute.ceil_div(num_queries, self.tile_candidates)
        )

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.SharedStorage)
        sK = smem.allocate_tensor(
            io_dtype, k_layout.outer, byte_alignment=128, swizzle=k_layout.inner
        )
        sQ = smem.allocate_tensor(
            io_dtype, q_layout.outer, byte_alignment=128, swizzle=q_layout.inner
        )
        sW = smem.allocate_tensor(Float32, cute.make_layout((self.packed_heads,)))
        if active and tidx < self.packed_heads:
            query = query0 + tidx // self.heads
            value = Float32(0.0)
            if query < num_queries:
                value = Float32(weights[batch, query, tidx % self.heads])
            sW[tidx] = value

        k_producer, k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.k_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=cute.size_in_bytes(io_dtype, cute.select(k_layout, mode=[0, 1, 2])),
            barrier_storage=storage.k_barriers.data_ptr(),
            defer_sync=True,
        ).make_participants()
        q_producer, q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=cute.size_in_bytes(io_dtype, cute.select(q_layout, mode=[0, 1, 2])),
            barrier_storage=storage.q_barriers.data_ptr(),
            defer_sync=True,
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.acc_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.epilogue_threads),
            barrier_storage=storage.acc_barriers.data_ptr(),
            defer_sync=True,
        ).make_participants()
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        accumulator_shape = tiled_mma.partition_shape_C(self.mma_tile[:2])
        accumulator_template = tiled_mma.make_fragment_C(
            cute.append(accumulator_shape, self.acc_stages)
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads),
            allocator_warp_id=self.mma_warp,
        )
        tmem.allocate(utils.get_num_tmem_alloc_cols(accumulator_template))
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        accumulator = cute.make_tensor(tmem_ptr, accumulator_template.layout)

        if active:
            if warp_idx < self.mma_warp:
                self.run_epilogue(
                    tiled_mma,
                    accumulator,
                    acc_consumer,
                    sW,
                    scores,
                    self.upcast_offset(local_pair),
                    tidx,
                    query0,
                    num_queries,
                    candidate_tiles,
                    score_scale,
                )
            elif warp_idx == self.mma_warp:
                self.run_mma(
                    tiled_mma,
                    accumulator,
                    sK,
                    sQ,
                    candidate_tiles,
                    k_consumer,
                    q_consumer,
                    acc_producer,
                )
            elif warp_idx == self.load_warp:
                self.run_load(
                    tiled_mma,
                    tma_atom_k,
                    k[None, None, batch],
                    tma_atom_q,
                    q[None, None, batch],
                    sK,
                    sQ,
                    batch_pair,
                    candidate_tiles,
                    k_producer,
                    q_producer,
                )

        tmem.relinquish_alloc_permit()
        pipeline.NamedBarrier(barrier_id=2, num_threads=self.threads).arrive_and_wait()
        tmem.free(tmem_ptr)

    @cute.jit
    def run_load(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_k: cute.CopyAtom,
        k: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        q: cute.Tensor,
        sK: cute.Tensor,
        sQ: cute.Tensor,
        batch_pair,
        candidate_tiles,
        k_producer,
        q_producer,
    ):
        """Load packed Q once, then stream candidate K tiles with TMA tail zero-fill."""
        cpasync.prefetch_descriptor(tma_atom_k)
        cpasync.prefetch_descriptor(tma_atom_q)
        mma_zero = tiled_mma.get_slice(0)
        gQ = cute.local_tile(q, self.mma_tile, (0, batch_pair, 0), proj=(None, 1, 1))
        s_q, g_q = cpasync.tma_partition(
            tma_atom_q,
            0,
            cute.make_layout(1),
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(mma_zero.partition_B(gQ), 0, 3),
        )
        q_empty = q_producer.acquire_and_advance()
        cute.copy(tma_atom_q, g_q, s_q[(None, q_empty.index)], tma_bar_ptr=q_empty.barrier)

        for candidate_tile in cutlass.range(candidate_tiles, unroll=0):
            gK = cute.local_tile(k, self.mma_tile, (candidate_tile, 0, 0), proj=(1, None, 1))
            s_k, g_k = cpasync.tma_partition(
                tma_atom_k,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 3),
                cute.group_modes(mma_zero.partition_A(gK), 0, 3),
            )
            k_empty = k_producer.acquire_and_advance()
            cute.copy(tma_atom_k, g_k, s_k[(None, k_empty.index)], tma_bar_ptr=k_empty.barrier)
        k_producer.tail()
        q_producer.tail()

    @cute.jit
    def run_mma(
        self,
        tiled_mma: cute.TiledMma,
        accumulator: cute.Tensor,
        sK: cute.Tensor,
        sQ: cute.Tensor,
        candidate_tiles,
        k_consumer,
        q_consumer,
        acc_producer,
    ):
        """Compute [128, D] @ [2*H, D].T into a two-stage FP32 TMEM ring."""
        fragment_k = tiled_mma.make_fragment_A(sK)
        fragment_q = tiled_mma.make_fragment_B(sQ)
        q_full = q_consumer.wait_and_advance()
        for _candidate_tile in cutlass.range(candidate_tiles, unroll=0):
            acc_empty = acc_producer.acquire_and_advance()
            k_full = k_consumer.wait_and_advance()
            for d_block in cutlass.range_constexpr(cute.size(fragment_k, mode=[2])):
                issue_mma = tiled_mma.with_()
                issue_mma.set(tcgen05.Field.ACCUMULATE, d_block != 0)
                cute.gemm(
                    issue_mma,
                    accumulator[(None, None, None, acc_empty.index)],
                    fragment_k[(None, None, d_block, k_full.index)],
                    fragment_q[(None, None, d_block, q_full.index)],
                    accumulator[(None, None, None, acc_empty.index)],
                )
            acc_empty.commit()
            k_full.release()
        q_full.release()
        acc_producer.tail()

    @cute.jit
    def run_epilogue(
        self,
        tiled_mma: cute.TiledMma,
        accumulator: cute.Tensor,
        acc_consumer,
        sW: cute.Tensor,
        scores: cute.Tensor,
        local_pair,
        tidx: Int32,
        query0,
        num_queries,
        candidate_tiles,
        score_scale: Float32,
    ):
        """Load one full head row per thread, independently reduce each packed query."""
        acc_tile = accumulator[(None, None, None, 0)]
        tmem_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.packed_heads // 4)),
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
        assert cute.size(logits) == self.packed_heads
        rW = cute.make_rmem_tensor((self.packed_heads,), Float32)
        cute.autovec_copy(sW, rW)

        for candidate_tile in cutlass.range(candidate_tiles, unroll=0):
            acc_full = acc_consumer.wait_and_advance()
            cute.copy(
                tmem_copy,
                tmem_thread.partition_S(accumulator[(None, None, None, acc_full.index)]),
                logits,
            )
            cute.arch.fence_view_async_tmem_load()
            acc_full.release()
            candidate = (
                self.upcast_offset(candidate_tile) * self.tile_candidates + coordinates[0][0]
            )
            for qi in cutlass.range_constexpr(2):
                query = query0 + qi
                valid = query < num_queries and candidate < num_queries
                if cutlass.const_expr(self.causal):
                    valid = valid and candidate <= query
                if valid:
                    # Four independent FP32 chains keep head reduction latency
                    # bounded without packed-PTX or cross-thread reductions.
                    sum0 = Float32(0.0)
                    sum1 = Float32(0.0)
                    sum2 = Float32(0.0)
                    sum3 = Float32(0.0)
                    for group in cutlass.range_constexpr(self.heads // 4):
                        offset = qi * self.heads + group * 4
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
                    scores[local_pair, qi, candidate] = (
                        (sum0 + sum1) + (sum2 + sum3)
                    ) * score_scale
