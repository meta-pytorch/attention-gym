# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# KDA forward intra-chunk diagonal sub-chunk kernel (safe-gate persistent grid).
#
# Computes the diagonal 16x16 Aqk/Akk sub-blocks of each 64-token chunk and
# inverts each diagonal Akk block by forward substitution. The off-diagonal
# blocks and the full 64x64 block-triangular inverse are handled by the CuTe
# K3b/K4b kernels. Only ``chunk_kda_fwd_kernel_intra_sub_chunk_forloop`` is
# consumed by the CuTe forward path
# (attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra).

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset, requires_int64_offsets
from attn_gym.linear.kda.chunk_scheduler import (
    RaggedChunkMetadata,
    load_ragged_chunk_work,
)
from attn_gym.linear.kda.utils import (
    IS_GATHER_SUPPORTED,
    autotune_cache_kwargs,
    exp2,
    gather,
)


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_INT64_OFFSETS": lambda args: requires_int64_offsets(
            args["q"],
            args["k"],
            args["g"],
            args["beta"],
            args["Aqk"],
            args["Akk"],
            args["cu_seqlens"],
            args["chunk_offsets"],
        ),
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=3),
    ],
    key=["BT", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(
    do_not_specialize=[
        "T",
        "num_sequences",
        "q_stride_t",
        "q_stride_h",
        "k_stride_t",
        "k_stride_h",
    ]
)
def chunk_kda_fwd_kernel_intra_sub_chunk_forloop(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_offsets,
    T,
    q_stride_t,
    q_stride_h,
    k_stride_t,
    k_stride_h,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    num_sequences,
    USE_INT64_OFFSETS: tl.constexpr,
    USE_GATHER: tl.constexpr,
    CAUSAL_NORMREF: tl.constexpr = True,
    GRID_NT: tl.constexpr = 0,
    MAX_NT: tl.constexpr = 0,
):
    i_t_start, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_INT64_OFFSETS:
        i_t_start = i_t_start.to(tl.int64)
        i_bh = i_bh.to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    for _iter in range((MAX_NT + GRID_NT - 1) // GRID_NT):
        i_t_orig = i_t_start + _iter * GRID_NT
        _run = i_t_orig < MAX_NT
        if IS_VARLEN and _run:
            _run = i_t_orig < tl.load(chunk_offsets + num_sequences)
        if _run:
            if IS_VARLEN:
                i_n, i_t, token_start, _ = load_ragged_chunk_work(
                    cu_seqlens,
                    chunk_offsets,
                    i_t_orig,
                    num_sequences,
                    BT,
                )
                if USE_INT64_OFFSETS:
                    i_n = i_n.to(tl.int64)
                    i_t = i_t.to(tl.int64)
                    token_start = token_start.to(tl.int64)
                eos = tl.load(cu_seqlens + ptr_offset((i_n,), (1,)) + 1).to(tl.int32)
                if USE_INT64_OFFSETS:
                    eos = eos.to(tl.int64)
                # token_start == bos + i_t * BT; only eos still needs a load for masking.
                bos = token_start - i_t * BT
                T_local = eos - bos
            else:
                i_t = i_t_orig
                bos = i_b * T
                T_local = T

            i_ti = ptr_offset((i_t, i_i), (BT, BC))
            if i_ti < T_local:
                o_c = i_ti + tl.arange(0, BC)
                m_c = o_c < T_local

                q_off = q + bos * q_stride_t + i_h * q_stride_h
                k_off = k + bos * k_stride_t + i_h * k_stride_h
                g_off = g + ptr_offset((bos, i_h), (H * K, K))
                beta_off = beta + ptr_offset((bos, i_h), (H, 1))
                Aqk_off = Aqk + ptr_offset((bos, i_h), (H * BT, BT))
                Akk_off = Akk + ptr_offset((bos, i_h), (H * BC, BC))

                o_k = tl.arange(0, BK)
                m_qkg = m_c[:, None] & (o_k[None, :] < K)
                q_offsets = o_c[:, None] * q_stride_t + o_k[None, :]
                k_offsets = o_c[:, None] * k_stride_t + o_k[None, :]
                g_offsets = ptr_offset(
                    (o_c[:, None], o_k[None, :]),
                    (H * K, 1),
                )
                b_q = tl.load(q_off + q_offsets, mask=m_qkg, other=0.0)
                b_k = tl.load(k_off + k_offsets, mask=m_qkg, other=0.0)
                b_g = tl.load(g_off + g_offsets, mask=m_qkg, other=0.0)
                b_beta = tl.load(
                    beta_off + ptr_offset((o_c,), (H,)),
                    mask=m_c,
                    other=0.0,
                )

                # The gate decay between positions i and j is 2^{g[i] - g[j]}. Splitting it
                # as 2^{g[i]-ref} * 2^{ref-g[j]} turns the decay into a matmul, and the
                # reference cancels in exact arithmetic. It does not cancel in floating
                # point: the two factors round separately, so the reference row is part of
                # the numerics. See NOTE [Causal gate reference].
                normref_idx = 0 if CAUSAL_NORMREF else min(BC // 2, T_local - i_ti - 1)
                if USE_GATHER:
                    b_gn = gather(b_g, tl.full([1, BK], normref_idx, dtype=tl.int16), axis=0)
                else:
                    p_gn = g_off + ptr_offset(
                        (i_ti + normref_idx, o_k),
                        (H * K, 1),
                    )
                    b_gn = tl.load(p_gn, mask=o_k < K, other=0.0)
                    b_gn = b_gn[None, :]

                b_gm = (b_g - b_gn).to(tl.float32)

                b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.0)
                b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.0)

                b_kgt = tl.trans(b_k * b_gk)

                b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
                b_Akk = tl.dot(b_k * b_gq, b_kgt) * b_beta[:, None]

                o_i = tl.arange(0, BC)
                m_Aqk = o_i[:, None] >= o_i[None, :]
                m_Akk = o_i[:, None] > o_i[None, :]
                m_I = o_i[:, None] == o_i[None, :]

                b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
                b_Akk = tl.where(m_Akk, b_Akk, 0.0)

                o_Aqk = ptr_offset((i_i, o_i), (BC, 1))
                p_Aqk = Aqk_off + ptr_offset(
                    (o_c[:, None], o_Aqk[None, :]),
                    (H * BT, 1),
                )
                p_Akk = Akk_off + ptr_offset(
                    (o_c[:, None], o_i[None, :]),
                    (H * BC, 1),
                )
                m_Aqk_store = m_c[:, None] & (o_Aqk[None, :] < BT)
                m_Akk_store = m_c[:, None] & (o_i[None, :] < BC)
                tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), mask=m_Aqk_store)
                tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), mask=m_Akk_store)

                tl.debug_barrier()

                # forward substitution
                b_Ai = -b_Akk
                for i in range(2, min(BC, T_local - i_ti)):
                    b_a = -tl.load(Akk_off + ptr_offset((i_ti + i, o_i), (H * BC, 1)))
                    b_a = tl.where(o_i < i, b_a, 0.0)
                    b_a += tl.sum(b_a[:, None] * b_Ai, 0)
                    b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
                b_Ai += m_I
                tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), mask=m_Akk_store)


# NOTE [Causal gate reference]
# The intra-chunk rebase splits 2^{g_i - g_j} into two separately rounded factors, so the
# reference row chosen for a BC-row subchunk changes results at the rounding level. Two
# choices are available:
#
#   causal (row 0)   never in the future for any query in the subchunk, so a change to a
#                    later token cannot perturb an earlier result. Spans BC-1 = 15 steps.
#   midpoint         halves the span to BC/2 = 8 steps, buying exponent headroom, but the
#                    reference itself moves when future tokens change, so causal-prefix
#                    outputs drift at the rounding level.
#
# Measured on GB300 (agent_space/probe_normref_*.py): both choices land at 1.14x the
# irreducible BF16 output-rounding floor, identical to five significant figures, so the
# midpoint buys no accuracy. Its only advantage is range. The span sets a hard ceiling on
# the supported gate, because the operands must stay inside the FP32 exponent range:
#
#   |lower_bound|_max = 128 / (span_steps * log2(e))
#       causal   128 / (15 * log2 e) = 5.915
#       midpoint 128 / ( 8 * log2 e) = 11.09
#
# Predicted and measured boundaries agree to three decimals. The default lower_bound of -5
# needs 7.213 log2/token against the causal ceiling of 8.533, clearing it by 18%. The
# public gate validates that ceiling; see GATE_SPAN_STEPS in gate_fwd.py.
def chunk_kda_fwd_intra_diagonal(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    metadata: RaggedChunkMetadata | None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the scheduler-aware diagonal intra-chunk stage in isolation."""
    batch, tokens, heads, key_dim = k.shape
    if batch != 1 or key_dim != 128 or chunk_size != 64:
        raise ValueError("the diagonal KDA stage requires B=1, K=128, and chunk_size=64")
    if metadata is not None:
        metadata.validate_chunk_size(chunk_size)
    elif tokens % chunk_size:
        raise ValueError(f"the dense diagonal KDA stage requires complete chunks, got T={tokens}")

    subchunk_size = 16
    subchunks = triton.cdiv(chunk_size, subchunk_size)
    capacity = triton.cdiv(tokens, chunk_size) if metadata is None else metadata.capacity
    grid_chunks = min(
        torch.cuda.get_device_properties(k.device).multi_processor_count,
        capacity,
    )
    Aqk = torch.empty(
        (batch, tokens, heads, chunk_size),
        device=k.device,
        dtype=k.dtype,
    )
    Akkd = torch.empty(
        (batch, tokens, heads, subchunk_size),
        device=k.device,
        dtype=torch.float32,
    )
    if grid_chunks == 0:
        return Aqk, Akkd

    chunk_kda_fwd_kernel_intra_sub_chunk_forloop[(grid_chunks, subchunks, batch * heads)](
        q=q,
        k=k,
        g=g,
        beta=beta,
        Aqk=Aqk,
        Akk=Akkd,
        scale=scale,
        cu_seqlens=None if metadata is None else metadata.cu_seqlens,
        chunk_offsets=None if metadata is None else metadata.chunk_offsets,
        T=tokens,
        q_stride_t=q.stride(1),
        q_stride_h=q.stride(2),
        k_stride_t=k.stride(1),
        k_stride_h=k.stride(2),
        H=heads,
        K=key_dim,
        BT=chunk_size,
        BC=subchunk_size,
        BK=triton.next_power_of_2(key_dim),
        num_sequences=0 if metadata is None else metadata.cu_seqlens.shape[0] - 1,
        USE_GATHER=IS_GATHER_SUPPORTED,
        CAUSAL_NORMREF=True,
        GRID_NT=grid_chunks,
        MAX_NT=capacity,
    )
    return Aqk, Akkd


__all__ = [
    "chunk_kda_fwd_intra_diagonal",
    "chunk_kda_fwd_kernel_intra_sub_chunk_forloop",
]
