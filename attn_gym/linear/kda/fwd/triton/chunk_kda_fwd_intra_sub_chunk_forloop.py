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
# blocks and the full 64x64 block-triangular inverse are handled by the
# architecture-selected K3/K4 stages. ``chunk_kda_fwd_kernel_intra_sub_chunk_forloop``
# is consumed by the composed forward path in ``fwd.cute.chunk_kda_fwd_intra``.

import torch
import triton
import triton.language as tl

from attn_gym._backends.cute.utils import get_device_properties
from attn_gym._backends.triton.utils import ptr_offset, requires_int64_offsets
from attn_gym.linear._delta_rule.triton.chunk_scheduler import (
    RaggedChunkMetadata,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)
from attn_gym.linear.kda.utils import (
    IS_GATHER_SUPPORTED,
    exp2,
    gather,
)

# See NOTE [Causal gate reference].
CAUSAL_REBASE_SHIFT = tl.constexpr(64.0)


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
@triton.jit(
    do_not_specialize=[
        "T",
        "num_sequences",
        "GRID_NT",
        "MAX_NT",
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
    GRID_NT=0,
    MAX_NT=0,
    FASTMATH: tl.constexpr = True,
):
    i_t_start, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_INT64_OFFSETS:
        i_t_start = i_t_start.to(tl.int64)
        i_bh = i_bh.to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    # Stride directly over chunk indices: a runtime grid must not introduce
    # integer division into the persistent loop's trip-count calculation.
    i_t_orig = i_t_start
    while i_t_orig < MAX_NT:
        _run = True
        if IS_VARLEN:
            _run = i_t_orig < load_ragged_chunk_count(chunk_offsets, num_sequences)
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
                b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
                b_Akk = tl.zeros([BC, BC], dtype=tl.float32)
                for i_k in tl.static_range(0, K, BK):
                    o_k = i_k + tl.arange(0, BK)
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
                    if CAUSAL_NORMREF:
                        # Padded rows hold -g_0 >= 0 for nonpositive gates; with positive
                        # gates they can only force a shift, which stays within range.
                        b_span = tl.min(tl.min(b_gm, axis=1), axis=0)
                        b_gm += tl.where(b_span < -CAUSAL_REBASE_SHIFT, CAUSAL_REBASE_SHIFT, 0.0)

                    b_gq = tl.where(m_c[:, None], exp2(b_gm, FASTMATH), 0.0)
                    b_gk = tl.where(m_c[:, None], exp2(-b_gm, FASTMATH), 0.0)

                    b_kgt = tl.trans(b_k * b_gk)

                    b_Aqk = tl.dot(b_q * b_gq, b_kgt, b_Aqk)
                    b_Akk = tl.dot(b_k * b_gq, b_kgt, b_Akk)
                b_Aqk *= scale
                b_Akk *= b_beta[:, None]

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
        i_t_orig += GRID_NT


# NOTE [Causal gate reference]
# The rebase evaluates 2^{g_i - g_j} as 2^{g_i - g_r} * 2^{g_r - g_j}. The factors round
# separately, so the reference row r affects floating-point results. Choosing row 0 keeps
# the reference at or before every query; a midpoint reference reduces the exponent span
# but makes early outputs depend on a future gate through rounding.
#
# A gate contributes at most |lower_bound| * log2(e) per token interval. Row 0 spans 15
# intervals, so its query factors 2^{g_i - g_0} fall to 2^-128 at the 5.914 fused bound
# (``constants.py``), below the 2^-126 FP32 normal range that the flush-to-zero fastmath
# exp2 keeps; that dropped the last rows' diagonal terms (the non-fast exp2 keeps them only as
# subnormals). A key block (the BK channels of one loop step) whose span
# min_{i,c} (g_ic - g_0c) over the subchunk is below -64 is therefore evaluated as
# 2^{g_i - g_0 + 64} * 2^{g_0 - g_j - 64}, as the SM100 CuTe engine evaluates every block.
# Either way both factors stay within 2^±64, so they remain normal while
# |lower_bound| <= 190 / (15 * log2(e)) = 8.78, and Q/K entries must stay below 2^64 in
# magnitude. Leaving flatter blocks unshifted keeps tiny keys there: shifted, a 2^-100 key
# underflows to 2^-164 before its product with a 2^58 query. A shifted block still loses
# gated Q/K entries below 2^-62, like the engine. The span reads later rows, so an earlier
# output can depend on a later gate, but only through rounding of the shifted exponent once
# the span crosses -64. The 8-interval midpoint needs no shift up to 10.9. Custom gate
# producers must honor the fused bound.
#
# The kernel sits at the 255-register limit. With fastmath one block covers all 128 channels;
# the span reduction made the non-fast exp2 spill (1.25x the time), so that mode walks K in
# 32-channel blocks instead.
def chunk_kda_fwd_intra_diagonal(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    metadata: RaggedChunkMetadata | None,
    chunk_size: int = 64,
    fastmath: bool = True,
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
    grid_chunks = min(get_device_properties(k.device).multi_processor_count, capacity)
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
        # See NOTE [Causal gate reference] for the non-fast key block.
        BK=key_dim if fastmath else 32,
        num_sequences=0 if metadata is None else metadata.cu_seqlens.shape[0] - 1,
        USE_GATHER=IS_GATHER_SUPPORTED,
        CAUSAL_NORMREF=True,
        GRID_NT=grid_chunks,
        MAX_NT=capacity,
        FASTMATH=fastmath,
        enable_reflect_ftz=fastmath,
        num_warps=1,
        num_stages=3,
    )
    return Aqk, Akkd


__all__ = [
    "chunk_kda_fwd_intra_diagonal",
    "chunk_kda_fwd_kernel_intra_sub_chunk_forloop",
]
