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

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cuda.bindings import driver as cuda
from cutlass import Float8E8M0FNU, Float32, Uint8, Uint32, cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.warpdrive import (
    Consumer,
    Local,
    NamedBarrier,
    Producer,
    ProtocolError,
    Ring,
    RoleConfig,
    StageView,
    Tmem,
    TmemPlan,
    Umma,
    Warps,
    WorkCursor,
    role,
)

from .cute_kernel_base import ScoreParams
from .cute_score import IndexerScoreKernel


@dataclass(frozen=True)
class MXScoreParams(ScoreParams):
    q_scale: cute.Tensor
    k_scale: cute.Tensor
    sQsf: cute.Tensor
    sKsf: cute.Tensor


class IndexerMXFP8ScoreKernel(IndexerScoreKernel):
    """One CTA scores a pair of queries against M128 candidate tiles at H=32/64, D=128.

    The TMA loader (warp 9) fills Q once and streams K. Warp 10 packs each operand's
    logical E8M0 scales (four per row) into native scale-factor layout in separate rings.
    Their Warps → Umma handoff proxy-fences the generic stores; TMA completion alone
    would not publish them. Warp 8 copies the scales into TMEM and issues four native
    D32 block-scaled MMAs per tile. Warpgroups 0 and 1 each drain one query's logits
    and reduce its heads. The accumulator's EMPTY waits for both query halves, and
    protects the corresponding K-scale TMEM columns from being overwritten as well.
    """

    name_prefix = "indexer_score_mxfp8"
    k_stages = 4
    acc_stages = 3
    sf_cols = 4

    def __init__(
        self,
        heads: int,
        head_dim: int,
        causal: bool,
        use_int64_offsets: bool = False,
        *,
        compress_ratio: int = 1,
        weight_dtype: type[cutlass.Numeric] = cutlass.BFloat16,
        contiguous_q_scales: bool = False,
        contiguous_k_scales: bool = False,
    ):
        super().__init__(
            heads,
            head_dim,
            causal,
            use_int64_offsets,
            compress_ratio=compress_ratio,
            contiguous_weight_heads=False,  # MX stages one scalar weight per thread.
            dtype=cutlass.Float8E4M3FN,
        )
        self.roles = {"load": RoleConfig(9)}
        if weight_dtype not in (cutlass.BFloat16, Float32):
            raise ValueError("MXFP8 indexer weights must be BF16 or FP32")
        self.weight_dtype = weight_dtype
        self.weights_ready = NamedBarrier(threads=256)
        self.contiguous_q_scales = contiguous_q_scales
        self.contiguous_k_scales = contiguous_k_scales
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
        # The handle picks the accumulator slot at runtime. Annotate accesses against the
        # whole (unaliasable) ring rather than attributing them to a static plan slot.
        self.tmem_plan = (
            TmemPlan()
            .region("acc", self.packed_heads * self.acc_stages)
            .region("Ksf", self.sf_cols, self.acc_stages)
            .region("Qsf", self.sf_cols)
        )

    def get_name(self) -> str:
        name = super().get_name()
        return f"{name}_qs{int(self.contiguous_q_scales)}_ks{int(self.contiguous_k_scales)}"

    @cute.jit
    def pack_scales(
        self,
        src: cute.Tensor,
        dst: cute.Tensor,
        first,
        extent,
        lane,
        is_query: cutlass.Constexpr,
        packed: cutlass.Constexpr,
    ):
        """Fill the four group bytes at ``lane * 16 + row_group * 4`` of one native SF stage."""
        if cutlass.const_expr(packed):
            values = cute.recast_tensor(src, Uint32)
            native = cute.make_tensor(cute.recast_ptr(dst.iterator, dtype=Uint32), (4, 32))
        else:
            values = cute.recast_tensor(src, Uint8)
            native = cute.make_tensor(cute.recast_ptr(dst.iterator, dtype=Uint8), (4, 4, 32))
        for row_group in cutlass.range_constexpr(4):
            row = lane + 32 * row_group
            if cutlass.const_expr(is_query):
                query = first + row // self.heads
                coord = (query, row % self.heads)
                valid = row < self.packed_heads and query < extent
            else:
                coord = (first + row,)
                valid = first + row < extent
            # Invalid rows multiply TMA's zero-filled data by one (E8M0 byte 127).
            if cutlass.const_expr(packed):
                value = Uint32(0x7F7F7F7F)
                if valid:
                    value = values[(*coord, 0)]
                native[row_group, lane] = value
            else:
                for group in cutlass.range_constexpr(4):
                    value = Uint8(127)
                    if valid:
                        value = values[(*coord, group)]
                    native[group, row_group, lane] = value

    @role(warps=10)
    @cute.jit
    def scales(
        self, params: MXScoreParams, *, qsf: Producer, ksf: Producer, work: WorkCursor, me: Local
    ):
        while work.valid:
            batch, _, query0, first, end, _ = self.pair_work(params, work.item[0])
            with qsf.produce() as h:
                self.pack_scales(
                    params.q_scale[batch, None, None, None],
                    h.view("Qsf", write=True, by="threads"),
                    query0,
                    params.weights.shape[1],
                    me.lane,
                    True,
                    self.contiguous_q_scales,
                )
            for tile in cutlass.range(first, end, unroll=0):
                with ksf.produce() as h:
                    self.pack_scales(
                        params.k_scale[batch, None, None],
                        h.view("Ksf", write=True, by="threads"),
                        self.upcast_offset(tile) * self.tile_candidates,
                        params.scores.shape[2],
                        me.lane,
                        False,
                        self.contiguous_k_scales,
                    )
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

    @role(warps=8)
    @cute.jit
    def mma(
        self,
        params: MXScoreParams,
        *,
        q: Consumer,
        k: Consumer,
        qsf: Consumer,
        ksf: Consumer,
        acc: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        tiled_mma = params.tiled_mma
        accumulator = cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        fragment_k = tiled_mma.make_fragment_A(params.sK)
        k = k.views(K=StageView(fragment_k, axis=3))
        q = q.views(Q=StageView(tiled_mma.make_fragment_B(params.sQ), axis=3))
        k_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, self.mma_tile, 32, params.sKsf[(None, None, None, 0)].layout
        )
        q_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, self.mma_tile, 32, params.sQsf[(None, None, None, 0)].layout
        )
        ksf_base = tmem.ptr("Ksf")
        tQsf = cute.make_tensor(cute.recast_ptr(tmem.ptr("Qsf"), dtype=Float8E8M0FNU), q_layout)
        assert tcgen05.find_tmem_tensor_col_offset(tQsf) == self.sf_cols
        while work.valid:
            _, _, _, first, end, _ = self.pair_work(params, work.item[0])
            q_full = q.wait()
            with qsf.consume() as scale:
                # Without an MMA tile, no accumulator wait orders this copy before TMEM free.
                if first < end:
                    self.copy_scales(scale.view("Qsf", read=True, by="umma"), tQsf)
            for _tile in cutlass.range(first, end, unroll=0):
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
                tmem.access("acc", write=True, by="umma")
                for d in cutlass.range_constexpr(cute.size(fragment_k, mode=[2])):
                    issue = tiled_mma.with_()
                    issue.set(tcgen05.Field.ACCUMULATE, d != 0)
                    issue.set(tcgen05.Field.SFA, tKsf[(None, None, d)].iterator)
                    issue.set(tcgen05.Field.SFB, tQsf[(None, None, d)].iterator)
                    dest = accumulator[(None, None, None, acc_empty.index)]
                    cute.gemm(
                        issue,
                        dest,
                        k_full.view("K", None, None, d, read=True, by="umma"),
                        q_full.view("Q", None, None, d, read=True, by="umma"),
                        dest,
                    )
                acc_empty.commit()
                k_full.release()
            q_full.release()
            work.advance()
        return q, k, qsf, ksf, acc, work

    @cute.jit
    def reduce_heads(self, logits: cute.Tensor, weights: cute.Tensor):
        """Accumulate four FP32 chains with two packed FMAs per four heads."""
        sum01 = (Float32(0), Float32(0))
        sum23 = (Float32(0), Float32(0))
        for group in cutlass.range_constexpr(self.heads // 4):
            h = 4 * group
            x0, x1, x2, x3 = logits[h], logits[h + 1], logits[h + 2], logits[h + 3]
            x0 = x0 if x0 > 0 else Float32(0)
            x1 = x1 if x1 > 0 else Float32(0)
            x2 = x2 if x2 > 0 else Float32(0)
            x3 = x3 if x3 > 0 else Float32(0)
            sum01 = cute.arch.fma_packed_f32x2((x0, x1), (weights[h], weights[h + 1]), sum01)
            sum23 = cute.arch.fma_packed_f32x2((x2, x3), (weights[h + 2], weights[h + 3]), sum23)
        return (sum01[0] + sum01[1]) + (sum23[0] + sum23[1])

    @role(warps=range(8))
    @cute.jit
    def epilogue(
        self, params: MXScoreParams, *, acc: Consumer, work: WorkCursor, tmem: Tmem, me: Local
    ):
        if cutlass.const_expr(me.warps != 8):
            raise ProtocolError(f"epilogue: two M128 query halves need 8 warps, got {me.warps}")
        qi = me.warp // 4
        tidx = me.tmem_row % 128
        accumulator = cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        halves = cute.logical_divide(accumulator, ((None, self.heads), None, None, None))
        acc_tile = cute.group_modes(halves[((None, (None, 0)), None, None, 0)], 0, 2)
        tmem_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.heads // 4)), Float32
        )
        tmem_copy = tcgen05.make_tmem_copy(tmem_atom, acc_tile)
        thread = tmem_copy.get_slice(tidx)
        coord_halves = cute.logical_divide(
            params.tiled_mma.get_slice(tidx).partition_C(
                cute.make_identity_tensor(self.mma_tile[:2])
            ),
            ((None, self.heads), None, None),
        )
        candidate_lane = thread.partition_D(
            cute.group_modes(coord_halves[((None, (None, 0)), None, None)], 0, 2)
        )[0][0]
        logits = cute.make_rmem_tensor(
            thread.partition_D(cute.make_identity_tensor(acc_tile.shape)).shape, Float32
        )
        assert cute.size(logits) == self.heads
        rW = cute.make_rmem_tensor((self.heads,), Float32)
        while work.valid:
            pair = self.upcast_offset(work.item[0])
            batch, _, query0, first, end, _ = self.pair_work(params, pair)
            if me.tid < self.packed_heads:
                query = query0 + me.tid // self.heads
                value = Float32(0)
                if query < params.weights.shape[1]:
                    value = Float32(params.weights[batch, query, me.tid % self.heads])
                params.sW[me.tid] = value
            self.weights_ready.sync()
            cute.autovec_copy(cute.logical_divide(params.sW, self.heads)[None, qi], rW)
            query = query0 + qi
            for tile in cutlass.range(first, end, unroll=0):
                with acc.consume() as h:
                    tmem.access("acc", read=True)
                    source = cute.group_modes(
                        halves[((None, (None, qi)), None, None, h.index)], 0, 2
                    )
                    cute.copy(tmem_copy, thread.partition_S(source), logits)
                candidate = self.upcast_offset(tile) * self.tile_candidates + candidate_lane
                valid = query < params.weights.shape[1] and candidate < params.scores.shape[2]
                if cutlass.const_expr(self.causal):
                    valid = valid and candidate < self.visible_candidates(query)
                if valid:
                    params.scores[pair, qi, candidate] = (
                        self.reduce_heads(logits, rW) * params.score_scale
                    )
            work.advance()
        return acc, work

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
        bound,
    ):
        smem = self.begin_roles(
            work=bound,
            layouts={
                "K": k_layout,
                "Q": q_layout,
                "Ksf": blockscaled_utils.make_smem_layout_sfa(
                    tiled_mma, self.mma_tile, 32, self.k_stages
                ),
                "Qsf": blockscaled_utils.make_smem_layout_sfb(
                    tiled_mma, self.mma_tile, 32, self.q_stages
                ),
            },
        )
        self.run_roles(
            MXScoreParams(
                tiled_mma=tiled_mma,
                tma_k=tma_atom_k,
                k=k,
                tma_q=tma_atom_q,
                q=q,
                sK=self.buffer("K"),
                sQ=self.buffer("Q"),
                sW=self.weight_stage(smem),
                acc_layout=tiled_mma.make_fragment_C(
                    cute.append(tiled_mma.partition_shape_C(self.mma_tile[:2]), self.acc_stages)
                ).layout,
                weights=weights,
                scores=scores,
                pair_start=pair_start,
                score_scale=score_scale,
                candidate_bounds=None,
                q_scale=q_scale,
                k_scale=k_scale,
                sKsf=self.buffer("Ksf"),
                sQsf=self.buffer("Qsf"),
            )
        )

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
        assert q.element_type == self.dtype and k.element_type == self.dtype
        assert weights.element_type == self.weight_dtype
        assert q_scale.element_type == Float8E8M0FNU and k_scale.element_type == Float8E8M0FNU
        assert scores.element_type == Float32
        tiled_mma = cute.make_tiled_mma(
            tcgen05.MmaMXF8F6F4Op(
                self.dtype,
                self.dtype,
                (*self.mma_tile[:2], 32),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
            )
        )
        k_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tile, self.dtype, self.k_stages
        )
        q_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tile, self.dtype, self.q_stages
        )
        tma_load = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        k_sdb = cute.make_tensor(k.iterator, cute.select(k.layout, mode=[1, 2, 0]))
        tma_atom_k, k_tma = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load, k_sdb, cute.select(k_layout, mode=[0, 1, 2]), self.mma_tile, tiled_mma
        )
        tma_atom_q, q_tma = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load,
            self.q_operand(q),
            cute.select(q_layout, mode=[0, 1, 2]),
            self.mma_tile,
            tiled_mma,
        )
        pair_start = self.upcast_offset(pair_start)
        total_pairs = self.upcast_offset(weights.shape[0]) * cute.ceil_div(
            self.upcast_offset(weights.shape[1]), 2
        )
        pairs = cutlass.min(self.upcast_offset(scores.shape[0]), total_pairs - pair_start)
        bound = self.work.bind(self.work_space(cutlass.Int32(pairs)))
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
            pair_start,
            score_scale,
            bound,
        ).launch(**self.launch_config(bound), stream=stream)
