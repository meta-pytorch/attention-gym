# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tiled SM100 indexer score generation without shape-sized shared buffers.

One CTA owns one query in the caller's [P, 2, T] FP32 score slab. Candidate,
head, and reduction tiles bound shared-memory and TMEM usage independently of
H and D. Separate head/query/batch TMA modes zero-fill tails without reading
another query or batch. This module allocates no global workspace and performs
no selection; masked and inactive slab entries remain untouched.
"""

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05


class IndexerGenericScoreKernel:
    """Generate scores for positive even H and positive D divisible by 16.

    For each candidate tile M128 and head tile N64, UMMA accumulates
    ``L[m,h] = sum_d K[m,d] * Q[h,d]`` in FP32 across K64 chunks. Only after
    completing D does the epilogue accumulate ``weight[h] * max(L[m,h], 0)``
    across head tiles, then multiply the final FP32 sum by ``score_scale``.

    Warps 0--3 each own 32 candidates, warp 4 issues UMMA, warp 5 loads both
    operands with TMA, and warps 6--7 participate only in CTA synchronization.
    Two combined operand stages feed a two-stage accumulator ring. Inputs are
    contiguous, 16-byte-aligned FP16/BF16. The caller supplies Int32 pair_start
    and dynamic shapes/strides, or their Int64 equivalents for wide offsets.
    """

    tile_candidates = 128
    tile_heads = 64
    tile_dim = 64
    ab_stages = 2
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
    ):
        if heads <= 0 or heads % 2 != 0:
            raise ValueError("IndexerGenericScoreKernel requires positive even H")
        if head_dim <= 0 or head_dim % 16 != 0:
            raise ValueError("IndexerGenericScoreKernel requires positive D divisible by 16")
        self.heads = heads
        self.head_dim = head_dim
        self.causal = causal
        self.use_int64_offsets = use_int64_offsets
        self.head_tiles = (heads + self.tile_heads - 1) // self.tile_heads
        self.d_tiles = (head_dim + self.tile_dim - 1) // self.tile_dim
        self.mma_tile = (self.tile_candidates, self.tile_heads, self.tile_dim)

        @cute.struct
        class SharedStorage:
            ab_barriers: cute.struct.MemRange[Int64, self.ab_stages * 2]
            acc_barriers: cute.struct.MemRange[Int64, self.acc_stages * 2]
            tmem_holding: Int32

        self.SharedStorage = SharedStorage

    def get_name(self) -> str:
        """Return a stable name for the static shape, mask, and offset width."""
        return (
            f"indexer_score_generic_h{self.heads}_d{self.head_dim}_c{int(self.causal)}_"
            f"i64{int(self.use_int64_offsets)}"
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
        stream: cuda_driver.CUstream,
    ):
        """Build boundary-preserving TMA descriptors and launch two CTAs per pair."""
        assert q.element_type in (cutlass.Float16, cutlass.BFloat16)
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
            tiled_mma, self.mma_tile, k.element_type, self.ab_stages
        )
        q_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tile, q.element_type, self.ab_stages
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
        ).launch(
            grid=(self.upcast_offset(scores.shape[0]) * 2, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

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
        """Guard the whole CTA, initialize pipelines, dispatch roles, and retire TMEM."""
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block_id, _, _ = cute.arch.block_idx()
        slab_pair = self.upcast_offset(block_id) // 2
        qi = block_id % 2
        num_queries = self.upcast_offset(weights.shape[1])
        pairs_per_batch = cute.ceil_div(num_queries, 2)
        global_pair = pair_start + slab_pair
        batch = global_pair // pairs_per_batch
        query = (global_pair % pairs_per_batch) * 2 + qi
        active = batch < self.upcast_offset(weights.shape[0]) and query < num_queries

        if active:
            candidate_tiles = (
                cute.ceil_div(query + 1, self.tile_candidates)
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
            ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
                num_stages=self.ab_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                tx_count=(
                    cute.size_in_bytes(io_dtype, cute.select(k_layout, mode=[0, 1, 2]))
                    + cute.size_in_bytes(io_dtype, cute.select(q_layout, mode=[0, 1, 2]))
                ),
                barrier_storage=storage.ab_barriers.data_ptr(),
                defer_sync=True,
            ).make_participants()
            acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
                num_stages=self.acc_stages,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, self.epilogue_threads
                ),
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

            if warp_idx < self.mma_warp:
                self.run_epilogue(
                    tiled_mma,
                    accumulator,
                    acc_consumer,
                    weights[batch, query, None],
                    scores,
                    slab_pair,
                    qi,
                    tidx,
                    query,
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
                    ab_consumer,
                    acc_producer,
                )
            elif warp_idx == self.load_warp:
                self.run_load(
                    tiled_mma,
                    tma_atom_k,
                    k[None, None, batch],
                    tma_atom_q,
                    q[None, None, query, batch],
                    sK,
                    sQ,
                    candidate_tiles,
                    ab_producer,
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
        candidate_tiles,
        ab_producer,
    ):
        """Stream paired K/Q boxes, including zero-filled head and reduction tails."""
        cpasync.prefetch_descriptor(tma_atom_k)
        cpasync.prefetch_descriptor(tma_atom_q)
        mma_zero = tiled_mma.get_slice(0)
        for candidate_tile in cutlass.range(candidate_tiles, unroll=0):
            for head_tile in cutlass.range(self.head_tiles, unroll=0):
                tile_coord = (
                    self.upcast_offset(candidate_tile),
                    self.upcast_offset(head_tile),
                    None,
                )
                gK = cute.local_tile(k, self.mma_tile, tile_coord, proj=(1, None, 1))
                gQ = cute.local_tile(q, self.mma_tile, tile_coord, proj=(None, 1, 1))
                s_k, g_k = cpasync.tma_partition(
                    tma_atom_k,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sK, 0, 3),
                    cute.group_modes(mma_zero.partition_A(gK), 0, 3),
                )
                s_q, g_q = cpasync.tma_partition(
                    tma_atom_q,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sQ, 0, 3),
                    cute.group_modes(mma_zero.partition_B(gQ), 0, 3),
                )
                for d_tile in cutlass.range(self.d_tiles, unroll=0):
                    ab_empty = ab_producer.acquire_and_advance()
                    cute.copy(
                        tma_atom_k,
                        g_k[(None, self.upcast_offset(d_tile))],
                        s_k[(None, ab_empty.index)],
                        tma_bar_ptr=ab_empty.barrier,
                    )
                    cute.copy(
                        tma_atom_q,
                        g_q[(None, self.upcast_offset(d_tile))],
                        s_q[(None, ab_empty.index)],
                        tma_bar_ptr=ab_empty.barrier,
                    )
        ab_producer.tail()

    @cute.jit
    def run_mma(
        self,
        tiled_mma: cute.TiledMma,
        accumulator: cute.Tensor,
        sK: cute.Tensor,
        sQ: cute.Tensor,
        candidate_tiles,
        ab_consumer,
        acc_producer,
    ):
        """Complete D before publishing each [128, 64] FP32 accumulator tile."""
        fragment_k = tiled_mma.make_fragment_A(sK)
        fragment_q = tiled_mma.make_fragment_B(sQ)
        for _candidate_tile in cutlass.range(candidate_tiles, unroll=0):
            for _head_tile in cutlass.range(self.head_tiles, unroll=0):
                acc_empty = acc_producer.acquire_and_advance()
                for d_tile in cutlass.range(self.d_tiles, unroll=0):
                    ab_full = ab_consumer.wait_and_advance()
                    for d_block in cutlass.range_constexpr(cute.size(fragment_k, mode=[2])):
                        issue_mma = tiled_mma.with_()
                        issue_mma.set(
                            tcgen05.Field.ACCUMULATE,
                            cutlass.Boolean(d_tile != Int32(0) or d_block != 0),
                        )
                        cute.gemm(
                            issue_mma,
                            accumulator[(None, None, None, acc_empty.index)],
                            fragment_k[(None, None, d_block, ab_full.index)],
                            fragment_q[(None, None, d_block, ab_full.index)],
                            accumulator[(None, None, None, acc_empty.index)],
                        )
                    ab_full.release()
                acc_empty.commit()
        acc_producer.tail()

    @cute.jit
    def run_epilogue(
        self,
        tiled_mma: cute.TiledMma,
        accumulator: cute.Tensor,
        acc_consumer,
        weights: cute.Tensor,
        scores: cute.Tensor,
        slab_pair,
        qi: Int32,
        tidx: Int32,
        query,
        num_queries,
        candidate_tiles,
        score_scale: Float32,
    ):
        """Keep one candidate's FP32 head sum in registers until its final slab store."""
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

        for candidate_tile in cutlass.range(candidate_tiles, unroll=0):
            candidate = (
                self.upcast_offset(candidate_tile) * self.tile_candidates + coordinates[0][0]
            )
            valid = candidate < num_queries
            if cutlass.const_expr(self.causal):
                valid = valid and candidate <= query
            sum0 = Float32(0.0)
            sum1 = Float32(0.0)
            sum2 = Float32(0.0)
            sum3 = Float32(0.0)
            for head_tile in cutlass.range(self.head_tiles, unroll=0):
                acc_full = acc_consumer.wait_and_advance()
                cute.copy(
                    tmem_copy,
                    tmem_thread.partition_S(accumulator[(None, None, None, acc_full.index)]),
                    logits,
                )
                cute.arch.fence_view_async_tmem_load()
                acc_full.release()
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
                scores[slab_pair, qi, candidate] = ((sum0 + sum1) + (sum2 + sum3)) * score_scale
