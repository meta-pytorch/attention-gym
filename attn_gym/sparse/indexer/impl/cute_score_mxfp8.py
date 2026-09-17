# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
#
# The score pipeline is adapted from the Apache-2.0 cuDNN implementation in cute_score.py.
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

"""Native block-scaled MXFP8 scoring into the indexer's bounded FP32 slab."""

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cuda.bindings import driver as cuda
from cutlass import Float8E8M0FNU, Float32, Uint8, Uint32, cute, pipeline, utils
from cutlass.cute.nvgpu import cpasync, tcgen05

from .cute_score import IndexerScoreKernel


class IndexerMXFP8ScoreKernel(IndexerScoreKernel):
    """Score E4M3 Q/K with logical E8M0 scales and BF16/FP32 weights on SM100/103.

    Operation ledger:
    - Pack K scales [128,4] and Q scales [2*H,4] into native SF shared layout.
    - Copy SF to TMEM; native MMA computes the sum of four scaled D32 dots.
    - Two epilogue warpgroups reduce ``weight * ReLU(dot)``, one query per group.

    Unlike the base kernel, this CTA has 384 threads: warps 0--3 and 4--7
    each cache H weights and drain H logits, warp 8 issues MMA, warp 9 streams
    K through four SMEM stages, and warp 10 loads immutable Q once. Warp 11
    participates only in CTA synchronization.

    Q/K retain the base kernel's TMA alignment and stride contract. Scale tensors
    have shapes [B,T,H,4] and [B,S,4], arbitrary byte-aligned strides, and E8M0
    dtype. Padding scales are one; corresponding TMA-loaded operand data are zero.
    The matching Q/K pipeline owns each scale stage; its FULL barrier needs two
    arrivals because TMA complete-tx does not publish the generic scale stores.
    """

    k_stages = 4
    acc_stages = 3
    epilogue_threads = 256
    mma_warp = 8
    load_warp = 9
    q_load_warp = 10
    threads = 384
    sf_cols = 4

    def __init__(
        self,
        heads: int,
        head_dim: int,
        causal: bool,
        use_int64_offsets: bool = False,
        *,
        compress_ratio: int = 1,
        contiguous_weight_heads: bool = True,
        contiguous_q_scales: bool = False,
        contiguous_k_scales: bool = False,
    ):
        super().__init__(
            heads,
            head_dim,
            causal,
            use_int64_offsets,
            compress_ratio=compress_ratio,
            contiguous_weight_heads=contiguous_weight_heads,
        )

        self.contiguous_q_scales = contiguous_q_scales
        self.contiguous_k_scales = contiguous_k_scales

    def get_name(self) -> str:
        name = super().get_name().replace("indexer_score_", "indexer_score_mxfp8_", 1)
        return f"{name}_qs{int(self.contiguous_q_scales)}_ks{int(self.contiguous_k_scales)}"

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        weights: cute.Tensor,
        q_scale: cute.Tensor,
        k_scale: cute.Tensor,
        scores: cute.Tensor,
        pair_start,
        score_scale: Float32,
        stream: cuda.CUstream,
    ):
        assert q.element_type == cutlass.Float8E4M3FN and k.element_type == q.element_type
        assert weights.element_type in (cutlass.BFloat16, Float32)
        assert q_scale.element_type == Float8E8M0FNU and k_scale.element_type == Float8E8M0FNU
        assert scores.element_type == Float32
        q_htdb = cute.make_tensor(q.iterator, cute.select(q.layout, mode=[2, 1, 3, 0]))
        q_packed = cute.group_modes(q_htdb, 0, 2)
        k_sdb = cute.make_tensor(k.iterator, cute.select(k.layout, mode=[1, 2, 0]))
        tiled_mma = cute.make_tiled_mma(
            tcgen05.MmaMXF8F6F4Op(
                k.element_type,
                q.element_type,
                (self.tile_candidates, self.packed_heads, 32),
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
        self.kernel.set_name_prefix(f"{self.get_name()}_w{weights.element_type.__name__.lower()}")
        self.kernel(
            tiled_mma,
            tma_atom_k,
            k_tma,
            tma_atom_q,
            q_tma,
            weights,
            q_scale,
            k_scale,
            scores,
            k_layout,
            q_layout,
            self.upcast_offset(pair_start),
            score_scale,
        ).launch(
            grid=(scores.shape[0], 1, 1),
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
        q_scale: cute.Tensor,
        k_scale: cute.Tensor,
        scores: cute.Tensor,
        k_layout: cute.ComposedLayout,
        q_layout: cute.ComposedLayout,
        pair_start,
        score_scale: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        local_pair, _, _ = cute.arch.block_idx()
        num_queries = self.upcast_offset(weights.shape[1])
        num_candidates = self.upcast_offset(scores.shape[2])
        pairs_per_batch = cute.ceil_div(num_queries, 2)
        global_pair = pair_start + self.upcast_offset(local_pair)
        batch = global_pair // pairs_per_batch
        batch_pair = global_pair % pairs_per_batch
        query0 = batch_pair * 2
        active = global_pair < self.upcast_offset(weights.shape[0]) * pairs_per_batch
        query_last = query0 + 1 if query0 + 1 < num_queries else query0
        candidate_tiles = (
            cute.ceil_div(self.visible_candidates(query_last), self.tile_candidates)
            if self.causal
            else cute.ceil_div(num_candidates, self.tile_candidates)
        )

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.SharedStorage)
        sK = smem.allocate_tensor(
            cutlass.Float8E4M3FN, k_layout.outer, byte_alignment=128, swizzle=k_layout.inner
        )
        sQ = smem.allocate_tensor(
            cutlass.Float8E4M3FN, q_layout.outer, byte_alignment=128, swizzle=q_layout.inner
        )
        sKsf = smem.allocate_tensor(
            Float8E8M0FNU,
            blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tile, 32, self.k_stages),
            byte_alignment=128,
        )
        sQsf = smem.allocate_tensor(
            Float8E8M0FNU,
            blockscaled_utils.make_smem_layout_sfb(tiled_mma, self.mma_tile, 32, 1),
            byte_alignment=128,
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
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=cute.size_in_bytes(
                cutlass.Float8E4M3FN, cute.select(k_layout, mode=[0, 1, 2])
            ),
            barrier_storage=storage.k_barriers.data_ptr(),
            defer_sync=True,
        ).make_participants()
        q_producer, q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=cute.size_in_bytes(
                cutlass.Float8E4M3FN, cute.select(q_layout, mode=[0, 1, 2])
            ),
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

        accumulator_template = tiled_mma.make_fragment_C(
            cute.append(tiled_mma.partition_shape_C(self.mma_tile[:2]), self.acc_stages)
        )
        k_sf_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, self.mma_tile, 32, sKsf[(None, None, None, 0)].layout
        )
        q_sf_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, self.mma_tile, 32, sQsf[(None, None, None, 0)].layout
        )
        acc_cols = tcgen05.find_tmem_tensor_col_offset(accumulator_template)
        sf_cols = self.sf_cols
        tmem = utils.TmemAllocator(
            storage.tmem_holding.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads),
            allocator_warp_id=self.mma_warp,
        )
        tmem.allocate(1 << (acc_cols + (self.acc_stages + 1) * sf_cols - 1).bit_length())
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        accumulator = cute.make_tensor(tmem_ptr, accumulator_template.layout)
        tKsf = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + acc_cols, dtype=Float8E8M0FNU), k_sf_layout
        )
        tQsf = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + acc_cols + self.acc_stages * sf_cols, dtype=Float8E8M0FNU),
            q_sf_layout,
        )
        assert tcgen05.find_tmem_tensor_col_offset(tKsf) == sf_cols
        assert tcgen05.find_tmem_tensor_col_offset(tQsf) == sf_cols

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
                    num_candidates,
                    candidate_tiles,
                    score_scale,
                )
            elif warp_idx == self.mma_warp:
                self.run_mma(
                    tiled_mma,
                    accumulator,
                    sK,
                    sQ,
                    sKsf,
                    sQsf,
                    tKsf,
                    tQsf,
                    candidate_tiles,
                    k_consumer,
                    q_consumer,
                    acc_producer,
                )
            elif warp_idx == self.load_warp or warp_idx == self.q_load_warp:
                self.run_load(
                    tiled_mma,
                    tma_atom_k,
                    k[None, None, batch],
                    tma_atom_q,
                    q[None, None, batch],
                    sK,
                    sQ,
                    sKsf,
                    sQsf,
                    q_scale[batch, None, None, None],
                    k_scale[batch, None, None],
                    batch_pair,
                    num_queries,
                    num_candidates,
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
        sKsf: cute.Tensor,
        sQsf: cute.Tensor,
        q_scale: cute.Tensor,
        k_scale: cute.Tensor,
        batch_pair,
        num_queries,
        num_candidates,
        candidate_tiles,
        k_producer,
        q_producer,
    ):
        cpasync.prefetch_descriptor(tma_atom_k)
        cpasync.prefetch_descriptor(tma_atom_q)
        mma_zero = tiled_mma.get_slice(0)
        lane = cute.arch.lane_idx()
        # Reinterpret only the layout: logical (row, D, stage) -> native SF bytes.
        # BlockScaledBasicChunk maps (row, group*32) to row%32*16 + row//32*4 + group.
        q_sf = cute.make_tensor(
            cute.recast_ptr(sQsf.iterator, dtype=Uint8),
            blockscaled_utils.make_smem_layout_sf((128, 128), 32, 1),
        )
        k_sf = cute.make_tensor(
            cute.recast_ptr(sKsf.iterator, dtype=Uint8),
            blockscaled_utils.make_smem_layout_sf((128, 128), 32, self.k_stages),
        )
        # Four group bytes are consecutive in both logical rows and each native SF row.
        # A packed signature promises unit group stride and four-byte aligned row origins.
        if cutlass.const_expr(self.contiguous_q_scales):
            q_words = cute.recast_tensor(q_scale, Uint32)
            q_sf_words = cute.make_tensor(
                cute.recast_ptr(sQsf.iterator, dtype=Uint32), cute.make_layout((128, 1))
            )
            q_bytes = None
        else:
            q_words, q_sf_words = None, None
            q_bytes = cute.recast_tensor(q_scale, Uint8)
        if cutlass.const_expr(self.contiguous_k_scales):
            k_words = cute.recast_tensor(k_scale, Uint32)
            k_sf_words = cute.make_tensor(
                cute.recast_ptr(sKsf.iterator, dtype=Uint32),
                cute.make_layout((128, self.k_stages)),
            )
            k_bytes = None
        else:
            k_words, k_sf_words = None, None
            k_bytes = cute.recast_tensor(k_scale, Uint8)
        if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == self.q_load_warp:
            gQ = cute.local_tile(q, self.mma_tile, (0, batch_pair, 0), proj=(None, 1, 1))
            s_q, g_q = cpasync.tma_partition(
                tma_atom_q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 3),
                cute.group_modes(mma_zero.partition_B(gQ), 0, 3),
            )
            q_empty = q_producer.acquire_and_advance()
            for row_group in cutlass.range_constexpr(4):
                row = lane + row_group * 32
                query = batch_pair * 2 + row // self.heads
                if cutlass.const_expr(self.contiguous_q_scales):
                    q_value = Uint32(0x7F7F7F7F)
                    if row < self.packed_heads and query < num_queries:
                        q_value = q_words[query, row % self.heads, 0]
                    q_sf_words[lane * 4 + row_group, q_empty.index] = q_value
                else:
                    for group in cutlass.range_constexpr(4):
                        q_value = Uint8(127)
                        if row < self.packed_heads and query < num_queries:
                            q_value = q_bytes[query, row % self.heads, group]
                        q_sf[row, group * 32, q_empty.index] = q_value
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            cute.copy(tma_atom_q, g_q, s_q[(None, q_empty.index)], tma_bar_ptr=q_empty.barrier)
            # Bulk complete-tx is non-transitive: publish the generic SF stores separately.
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(q_empty.barrier)
            q_producer.tail()
        else:
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
                for row_group in cutlass.range_constexpr(4):
                    row = lane + row_group * 32
                    candidate = self.upcast_offset(candidate_tile) * 128 + row
                    if cutlass.const_expr(self.contiguous_k_scales):
                        value = Uint32(0x7F7F7F7F)
                        if candidate < num_candidates:
                            value = k_words[candidate, 0]
                        k_sf_words[lane * 4 + row_group, k_empty.index] = value
                    else:
                        for group in cutlass.range_constexpr(4):
                            value = Uint8(127)
                            if candidate < num_candidates:
                                value = k_bytes[candidate, group]
                            k_sf[row, group * 32, k_empty.index] = value
                cute.arch.fence_view_async_shared()
                cute.arch.sync_warp()
                cute.copy(tma_atom_k, g_k, s_k[(None, k_empty.index)], tma_bar_ptr=k_empty.barrier)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(k_empty.barrier)
            k_producer.tail()

    @cute.jit
    def scale_copy(self, source: cute.Tensor, destination: cute.Tensor):
        """Partition native scale bytes for a single-warp SMEM-to-TMEM copy."""
        atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), Float8E8M0FNU)
        dest = cute.filter_zeros(destination)
        tiled_copy = tcgen05.make_s2t_copy(atom, dest)
        thread_copy = tiled_copy.get_slice(0)
        source_desc = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy, thread_copy.partition_S(cute.filter_zeros(source))
        )
        return tiled_copy, source_desc, thread_copy.partition_D(dest)

    @cute.jit
    def run_mma(
        self,
        tiled_mma: cute.TiledMma,
        accumulator: cute.Tensor,
        sK: cute.Tensor,
        sQ: cute.Tensor,
        sKsf: cute.Tensor,
        sQsf: cute.Tensor,
        tKsf: cute.Tensor,
        tQsf: cute.Tensor,
        candidate_tiles,
        k_consumer,
        q_consumer,
        acc_producer,
    ):
        fragment_k = tiled_mma.make_fragment_A(sK)
        fragment_q = tiled_mma.make_fragment_B(sQ)
        copy_q, source_q, dest_q = self.scale_copy(sQsf, tQsf)
        q_full = q_consumer.wait_and_advance()
        cute.copy(copy_q, source_q[(None, None, None, None, q_full.index)], dest_q)
        for _candidate_tile in cutlass.range(candidate_tiles, unroll=0):
            # The matching accumulator EMPTY protects this slot's previous MMA and
            # both query-half TMEM loads.
            acc_empty = acc_producer.acquire_and_advance()
            k_full = k_consumer.wait_and_advance()
            tKsf_stage = cute.make_tensor(
                cute.recast_ptr(
                    cute.recast_ptr(tKsf.iterator, dtype=Float32) + acc_empty.index * self.sf_cols,
                    dtype=Float8E8M0FNU,
                ),
                tKsf.layout,
            )
            copy_k, source_k, dest_k = self.scale_copy(sKsf, tKsf_stage)
            cute.copy(copy_k, source_k[(None, None, None, None, k_full.index)], dest_k)
            for d_block in cutlass.range_constexpr(cute.size(fragment_k, mode=[2])):
                issue_mma = tiled_mma.with_()
                issue_mma.set(tcgen05.Field.ACCUMULATE, d_block != 0)
                issue_mma.set(tcgen05.Field.SFA, tKsf_stage[(None, None, d_block)].iterator)
                issue_mma.set(tcgen05.Field.SFB, tQsf[(None, None, d_block)].iterator)
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
        tidx: cutlass.Int32,
        query0,
        num_queries,
        num_candidates,
        candidate_tiles,
        score_scale: Float32,
    ):
        """Each warpgroup drains one query, releases, then reduces with cached weights."""
        qi = cute.arch.make_warp_uniform(tidx // 128)
        tidx = tidx % 128
        # Split the MMA fragment's logical N into (head, query), slice one query, regroup.
        acc_halves = cute.logical_divide(accumulator, ((None, self.heads), None, None, None))
        acc_tile = cute.group_modes(acc_halves[((None, (None, 0)), None, None, 0)], 0, 2)
        tmem_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.heads // 4)), Float32
        )
        tmem_copy = tcgen05.make_tmem_copy(tmem_atom, acc_tile)
        tmem_thread = tmem_copy.get_slice(tidx)
        coord_halves = cute.logical_divide(
            tiled_mma.get_slice(tidx).partition_C(cute.make_identity_tensor(self.mma_tile[:2])),
            ((None, self.heads), None, None),
        )
        coordinates = tmem_thread.partition_D(
            cute.group_modes(coord_halves[((None, (None, 0)), None, None)], 0, 2)
        )
        logits = cute.make_rmem_tensor(
            tmem_thread.partition_D(cute.make_identity_tensor(acc_tile.shape)).shape, Float32
        )
        assert cute.size(logits) == self.heads
        rW = cute.make_rmem_tensor((self.heads,), Float32)
        weight_halves = cute.logical_divide(sW, self.heads)
        cute.autovec_copy(weight_halves[None, qi], rW)
        query = query0 + qi

        for candidate_tile in cutlass.range(candidate_tiles, unroll=0):
            acc_full = acc_consumer.wait_and_advance()
            candidate = (
                self.upcast_offset(candidate_tile) * self.tile_candidates + coordinates[0][0]
            )
            cute.copy(
                tmem_copy,
                tmem_thread.partition_S(
                    cute.group_modes(
                        acc_halves[((None, (None, qi)), None, None, acc_full.index)], 0, 2
                    )
                ),
                logits,
            )
            cute.arch.fence_view_async_tmem_load()
            # EMPTY counts all 256 threads, so neither half can be overwritten
            # until both groups have completed their asynchronous TMEM loads.
            acc_full.release()
            valid = query < num_queries and candidate < num_candidates
            if cutlass.const_expr(self.causal):
                valid = valid and candidate < self.visible_candidates(query)
            if valid:
                # Packed FMAs keep the base kernel's four FP32 chains and final tree.
                sum0 = Float32(0.0)
                sum1 = Float32(0.0)
                sum2 = Float32(0.0)
                sum3 = Float32(0.0)
                for group in cutlass.range_constexpr(self.heads // 4):
                    offset = group * 4
                    x0 = Float32(logits[offset])
                    x1 = Float32(logits[offset + 1])
                    x2 = Float32(logits[offset + 2])
                    x3 = Float32(logits[offset + 3])
                    x0 = x0 if x0 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
                    x1 = x1 if x1 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
                    x2 = x2 if x2 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
                    x3 = x3 if x3 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
                    sum0, sum1 = cute.arch.fma_packed_f32x2(
                        (x0, x1), (rW[offset], rW[offset + 1]), (sum0, sum1)
                    )
                    sum2, sum3 = cute.arch.fma_packed_f32x2(
                        (x2, x3), (rW[offset + 2], rW[offset + 3]), (sum2, sum3)
                    )
                scores[local_pair, qi, candidate] = ((sum0 + sum1) + (sum2 + sum3)) * score_scale
