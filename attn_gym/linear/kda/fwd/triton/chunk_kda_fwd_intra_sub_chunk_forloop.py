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

                # the kda gate is applied as a decay between 2 positions i, j
                # and cumulative log gate is  2^{g[i] − g[j]}
                # but you can do 2^{g[i]−g[j]}  =  2^{g[i]} * 2^{−g[j]} and this
                # turns it into a matmul
                # but then now you have g[i] and g[j] so what if they're really big
                # and overflow?
                # so the trick is you pick a rebase index and you do g[i] - rebase and
                # rebase - g[j] so you always are within bounds because the number is
                # smaller and also mathematically it cancels out
                # when CAUSAL_NORMREF = True, we pick g[0] and when its False we pick
                # the midpoint
                # but the concern w/ the midpoint is that it breaks causality by whatever
                # the rounding error is and also no BI because the midpoint changes
                # so we are setting CAUSAL_NORMREF = True as default for now
                # the only concern with doing topmost is what if the numbers are actually
                # really big (but does this happen in practice? we will see)
                if CAUSAL_NORMREF:
                    normref_idx = 0
                else:
                    normref_idx = min(BC // 2, T_local - i_ti - 1)
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

                # Forward substitution computes M with M = N + N @ M for N = -Akk
                # (strictly lower, so N^BC == 0) and stores the unit inverse
                # (I + Akk)^-1 = I + M. Use the log-depth Neumann factorization
                # (I - N)^-1 = (I + N)(I + N^2)(I + N^4)(I + N^8): register-resident
                # dots instead of a BC-step serial load -> reduce -> select chain.
                # Rows at or past T_local carry zero data and stay identity rows.
                b_N = -b_Akk
                b_Ai = b_N + m_I
                for _double in tl.static_range(3):
                    b_N = tl.sum(b_N[:, :, None] * b_N[None, :, :], 1)
                    b_Ai += tl.sum(b_Ai[:, :, None] * b_N[None, :, :], 1)
                tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), mask=m_Akk_store)


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
        CAUSAL_NORMREF=False,
        GRID_NT=grid_chunks,
        MAX_NT=capacity,
    )
    return Aqk, Akkd


__all__ = [
    "chunk_kda_fwd_intra_diagonal",
    "chunk_kda_fwd_kernel_intra_sub_chunk_forloop",
]
