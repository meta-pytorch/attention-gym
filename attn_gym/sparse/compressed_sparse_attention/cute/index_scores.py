"""SM100 tensor-core index scoring with CSA-compatible BF16 boundaries."""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import torch
from cutlass import Float32, Int32

from cudnn.deepseek_sparse_attention.indexer_forward.indexer_fwd_sm100 import (
    IndexerForwardSm100,
)
from cudnn.deepseek_sparse_attention.utils.compiler import compile_options
from cudnn.deepseek_sparse_attention.utils.runtime import resolve_stream
from cudnn.deepseek_sparse_attention.utils.tensor_conversion import to_cute_tensor


class ExactBf16IndexerForwardSm100(IndexerForwardSm100):
    """Reuse cuDNN's TMA/UMMA pipeline with the CSA score epilogue."""

    @cute.jit
    def _epilogue_warp(
        self,
        q_stage_idx,
        tiled_mma_qk,
        tStS_stage,
        sW,
        sScore_stage,
        S_mbar_ptr,
        Score_store_mbar_ptr,
        m_block,
        batch_idx,
        num_n_blocks,
        seqlen_k,
        seqlen_q,
        tidx,
        epi_s_full_phase,
        score_empty_phase,
        sm_scale,
        sW_base=None,
    ):
        """Match CSA's BF16 boundaries and power-of-two head reduction."""

        tidx_wg = tidx % (cute.arch.WARP_SIZE * 4)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(
            tmem_load_atom,
            tStS_stage,
        ).get_slice(tidx_wg)
        tStS_t2r = thr_tmem_load.partition_S(tStS_stage)

        thr_mma = tiled_mma_qk.get_slice(tidx_wg)
        cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
        tScS_mma = thr_mma.partition_C(cS)
        tScS = thr_tmem_load.partition_D(tScS_mma)
        tSrS_shape = thr_tmem_load.partition_D(
            cute.make_identity_tensor(tStS_stage.shape),
        ).shape
        tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

        weight_base = Int32(0) if sW_base is None else sW_base
        qhpkv = self.qhead_per_kvhead
        ratio = Int32(self.ratio)
        q_global_start = seqlen_k * ratio - seqlen_q

        weights_per_vector = 4
        sW_f32_ptr = cute.make_ptr(
            Float32,
            (sW.iterator + weight_base + q_stage_idx * self.m_block_size).llvm_ptr,
            cute.AddressSpace.smem,
            assumed_align=16,
        )
        sW_f32 = cute.make_tensor(
            sW_f32_ptr,
            cute.make_layout((self.m_block_size // 2,)),
        )
        sW_vectors = cute.logical_divide(
            sW_f32,
            cute.make_layout((weights_per_vector,)),
        )
        rW_f32 = cute.make_rmem_tensor(
            (self.m_block_size // 2,),
            Float32,
        )
        rW_vectors = cute.logical_divide(
            rW_f32,
            cute.make_layout((weights_per_vector,)),
        )
        rW_buffer = cute.make_rmem_tensor(
            (2 * weights_per_vector,),
            self.w_dtype,
        )
        rW_buffer_f32 = cute.recast_tensor(rW_buffer, Float32)

        local_count = num_n_blocks if num_n_blocks < Int32(3) else Int32(3)
        for iter_idx in cutlass.range(num_n_blocks, unroll=1):
            n_block = iter_idx - local_count
            if iter_idx < local_count:
                n_block = num_n_blocks - Int32(1) - iter_idx

            cute.arch.mbarrier_wait(S_mbar_ptr + q_stage_idx, epi_s_full_phase)
            epi_s_full_phase = epi_s_full_phase ^ 1
            if iter_idx == 0:
                for vector in cutlass.range_constexpr(
                    (self.m_block_size // 2) // weights_per_vector,
                ):
                    cute.autovec_copy(
                        sW_vectors[None, vector],
                        rW_vectors[None, vector],
                    )
            cute.copy(thr_tmem_load, tStS_t2r, tSrS)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.mbarrier_arrive(S_mbar_ptr + self.q_stage + q_stage_idx)

            # The reference rounds each per-head QK dot, scale, and weighted
            # contribution to BF16. Transform the TMEM fragment in place so the
            # subsequent tree reduction has exactly the same power-of-two order.
            for query in cutlass.range_constexpr(self.q_tokens_per_tile):
                for head_vector in cutlass.range_constexpr(
                    qhpkv // 2 // weights_per_vector,
                ):
                    cute.autovec_copy(
                        rW_vectors[
                            None,
                            query * (qhpkv // 2 // weights_per_vector) + head_vector,
                        ],
                        rW_buffer_f32,
                    )

                    for item in cutlass.range_constexpr(weights_per_vector):
                        head0 = (head_vector * weights_per_vector + item) * 2
                        index0 = query * qhpkv + head0
                        index1 = index0 + 1

                        value0 = tSrS[index0].to(self.q_dtype).to(Float32)
                        value0 = value0 if value0 > Float32(0.0) else Float32(0.0)
                        value0 = (value0 * Float32(sm_scale)).to(self.q_dtype).to(Float32)
                        tSrS[index0] = (
                            (value0 * rW_buffer[2 * item].to(Float32)).to(self.w_dtype).to(Float32)
                        )

                        value1 = tSrS[index1].to(self.q_dtype).to(Float32)
                        value1 = value1 if value1 > Float32(0.0) else Float32(0.0)
                        value1 = (value1 * Float32(sm_scale)).to(self.q_dtype).to(Float32)
                        tSrS[index1] = (
                            (value1 * rW_buffer[2 * item + 1].to(Float32))
                            .to(self.w_dtype)
                            .to(Float32)
                        )

            scores = cute.make_rmem_tensor(
                (self.q_tokens_per_tile,),
                Float32,
            )
            for query in cutlass.range_constexpr(self.q_tokens_per_tile):
                for stage in cutlass.range_constexpr(int(math.log2(qhpkv))):
                    stride = qhpkv >> (stage + 1)
                    for head in cutlass.range_constexpr(stride):
                        lhs = query * qhpkv + head
                        tSrS[lhs] = tSrS[lhs] + tSrS[lhs + stride]
                scores[query] = tSrS[query * qhpkv]

            if iter_idx > 0:
                cute.arch.mbarrier_wait(
                    Score_store_mbar_ptr + self.q_stage + q_stage_idx,
                    score_empty_phase,
                )
                score_empty_phase = score_empty_phase ^ 1

            kv_offset = tScS[0][0]
            query_base = (self.q_stage * m_block + q_stage_idx) * self.q_tokens_per_tile
            if iter_idx < local_count:
                kv_token = kv_offset + n_block * self.n_block_size
                for query in cutlass.range(
                    self.q_tokens_per_tile,
                    unroll_full=True,
                ):
                    query_token = query_base + query
                    value = -Float32.inf
                    if query_token < seqlen_q and kv_token < seqlen_k:
                        column_limit = (q_global_start + query_token + 1) // ratio
                        if kv_token < column_limit:
                            value = scores[query]
                    scores[query] = value

            sScore_dst_ptr = cute.make_ptr(
                Float32,
                (sScore_stage.iterator + kv_offset).llvm_ptr,
                cute.AddressSpace.smem,
            )
            sScore_dst = cute.make_tensor(
                sScore_dst_ptr,
                cute.make_layout(
                    (self.q_tokens_per_tile,),
                    stride=(self.n_block_size,),
                ),
            )
            cute.autovec_copy(scores, sScore_dst)
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(
                Score_store_mbar_ptr + q_stage_idx,
            )

        if num_n_blocks > 0:
            cute.arch.mbarrier_wait(
                Score_store_mbar_ptr + self.q_stage + q_stage_idx,
                score_empty_phase,
            )
            score_empty_phase = score_empty_phase ^ 1

        return epi_s_full_phase, score_empty_phase


_compile_cache: dict[tuple[object, ...], object] = {}


def exact_bf16_index_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    *,
    ratio: int,
    qhead_per_kv_head: int,
    out: torch.Tensor,
    sm_scale: float,
    current_stream: cuda.CUstream | None = None,
) -> torch.Tensor:
    """Launch the exact-epilogue BSHD scorer into a caller-owned FP32 slab."""

    if not (q.is_contiguous() and k.is_contiguous() and weights.is_contiguous()):
        raise ValueError("Tensor-core index-score inputs must be contiguous.")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("Tensor-core index scoring currently requires BF16 Q and K.")
    if weights.dtype != torch.bfloat16 or out.dtype != torch.float32:
        raise TypeError("Tensor-core index weights must be BF16 and scores FP32.")

    batch, sequence, query_heads, head_dim = q.shape
    key_batch, num_blocks, key_heads, key_dim = k.shape
    if key_batch != batch or key_dim != head_dim or query_heads != qhead_per_kv_head * key_heads:
        raise ValueError("Incompatible tensor-core index-score shapes.")
    if weights.shape != (batch, sequence, query_heads):
        raise ValueError("Incompatible tensor-core index-weight shape.")
    if out.shape != (batch, sequence, num_blocks):
        raise ValueError("Incompatible tensor-core score output shape.")

    m_block_size = 128
    n_block_size = 128
    q_stage = 2
    kv_stage = 4
    compile_key = (
        q.dtype,
        head_dim,
        qhead_per_kv_head,
        ratio,
        m_block_size,
        n_block_size,
        q_stage,
        kv_stage,
    )
    stream = resolve_stream(current_stream)
    if compile_key not in _compile_cache:
        kernel = ExactBf16IndexerForwardSm100(
            head_dim=head_dim,
            qhead_per_kvhead=qhead_per_kv_head,
            ratio=ratio,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            q_stage=q_stage,
            kv_stage=kv_stage,
        )
        _compile_cache[compile_key] = cute.compile(
            kernel,
            to_cute_tensor(q),
            to_cute_tensor(k),
            to_cute_tensor(weights),
            to_cute_tensor(out),
            key_heads,
            Int32(sequence),
            Int32(num_blocks),
            Float32(sm_scale),
            None,
            None,
            stream,
            options=compile_options(),
        )

    out.fill_(float("-inf"))
    _compile_cache[compile_key](
        q,
        k,
        weights,
        out,
        key_heads,
        Int32(sequence),
        Int32(num_blocks),
        Float32(sm_scale),
        None,
        None,
        stream,
    )
    return out


__all__ = ["exact_bf16_index_scores"]
