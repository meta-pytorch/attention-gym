# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
#
# Portions are derived from flash-linear-attention and licensed under the MIT license;
# see https://github.com/fla-org/flash-linear-attention/graphs/contributors.
# The remaining portions use the BSD-style license in the repository root.

"""Scalar-GDN output composition with on-the-fly causal QK."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset
from attn_gym.linear.kda.chunk_scheduler import (
    RaggedChunkMetadata,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)
from attn_gym.linear.kda.utils import autotune_cache_kwargs, exp2


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": 128, "BV": 128}, num_warps=8, num_stages=3),
        triton.Config({"BK": 64, "BV": 64}, num_warps=4, num_stages=3),
        triton.Config({"BK": 32, "BV": 32}, num_warps=2, num_stages=3),
    ],
    key=["H", "HV", "K", "V", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "num_sequences"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    q_stride_t,
    k_stride_t,
    v_stride_t,
    h,
    g,
    o,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    num_sequences,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2).to(tl.int64)
    i_b, i_h = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        if i_t >= load_ragged_chunk_count(chunk_offsets, num_sequences):
            return
        i_tg = i_t
        i_n, i_t, token_start, _valid = load_ragged_chunk_work(
            cu_seqlens, chunk_offsets, i_t, num_sequences, BT
        )
        bos = token_start - i_t * BT
        eos = tl.load(cu_seqlens + ptr_offset((i_n + 1,), (1,))).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # offset calculation
    key_head = i_h // (HV // H)
    q += ptr_offset((bos, key_head), (q_stride_t, K))
    k += ptr_offset((bos, key_head), (k_stride_t, K))
    v += ptr_offset((bos, i_h), (v_stride_t, V))
    o += ptr_offset((bos, i_h), (HV * V, V))
    h += ptr_offset((i_tg, i_h), (HV * K * V, K * V))

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    o_v = i_v * BV + tl.arange(0, BV)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_q = q + ptr_offset((o_t[:, None], o_k[None, :]), (q_stride_t, 1))
        p_k = k + ptr_offset((o_k[:, None], o_t[None, :]), (1, k_stride_t))
        p_h = h + ptr_offset((o_k[:, None], o_v[None, :]), (V, 1))
        m_h = m_k[:, None] & (o_v[None, :] < V)
        # [BT, BK]
        b_q = tl.load(p_q, mask=m_t[:, None] & m_k[None, :], other=0.0)
        # [BK, BT]
        b_k = tl.load(p_k, mask=m_k[:, None] & m_t[None, :], other=0.0)
        b_h = tl.load(p_h, mask=m_h, other=0.0)

        # [BT, BK] @ [BK, BV] -> [BT, BV]
        b_o += tl.dot(b_q, b_h)
        # [BT, BK] @ [BK, BT] -> [BT, BT]
        b_A += tl.dot(b_q, b_k)

    g += ptr_offset((bos, i_h), (HV, 1))
    p_g = g + ptr_offset((o_t,), (HV,))
    b_g = tl.load(p_g, mask=m_t, other=0.0)
    b_o = b_o * exp2(b_g)[:, None]
    decay_mask = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    gate_delta = b_g[:, None] - b_g[None, :]
    decay = tl.where(
        decay_mask,
        exp2(tl.where(decay_mask, gate_delta, 0.0)),
        0.0,
    )
    b_A = b_A * decay
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = v + ptr_offset((o_t[:, None], o_v[None, :]), (v_stride_t, 1))
    p_o = o + ptr_offset((o_t[:, None], o_v[None, :]), (HV * V, 1))

    b_v = tl.load(p_v, mask=m_t[:, None] & (o_v < V)[None, :], other=0.0)
    # to fix mma -> mma layout conversion
    # already solved by triton v3.2 or higher
    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=m_t[:, None] & (o_v < V)[None, :])


def chunk_gdn_fwd_output_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    cumulative_gate: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Compose dense GQA-capable BT64 output without materializing Aqk."""
    batch, tokens, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if k.shape != q.shape or v.shape[:2] != q.shape[:2] or value_heads % key_heads:
        raise ValueError("dense fused chunk GDN requires matching Q/K and H % HK == 0")
    if batch != 1 or tokens % 64 or (key_dim, value_dim) != (128, 128):
        raise ValueError("dense fused chunk GDN requires B=1, complete BT64 chunks, and K=V=128")
    chunks = tokens // 64
    if h.shape != (batch, chunks, value_heads, key_dim, value_dim):
        raise ValueError("h must contain one [K,V] entry state per chunk and head")

    output = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    def grid(meta):
        return (triton.cdiv(value_dim, meta["BV"]), chunks, batch * value_heads)

    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        q_stride_t=q.stride(1),
        k_stride_t=k.stride(1),
        v_stride_t=v.stride(1),
        h=h,
        g=cumulative_gate,
        o=output,
        cu_seqlens=None,
        chunk_offsets=None,
        scale=scale,
        T=tokens,
        num_sequences=0,
        H=key_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BT=64,
    )
    return output


def chunk_gdn_fwd_output_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    cumulative_gate: torch.Tensor,
    scale: float,
    metadata: RaggedChunkMetadata,
) -> torch.Tensor:
    """Compose fixed-capacity packed GQA output without materializing Aqk."""
    metadata.validate_chunk_size(64)
    batch, tokens, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if batch != 1 or k.shape != q.shape or v.shape[:2] != q.shape[:2]:
        raise ValueError("packed fused chunk GDN requires B=1 and matching Q/K token axes")
    if value_heads % key_heads or (key_dim, value_dim) != (128, 128):
        raise ValueError("packed fused chunk GDN requires K=V=128 and H % HK == 0")
    if h.shape != (batch, metadata.capacity, value_heads, key_dim, value_dim):
        raise ValueError("h must contain one fixed-capacity [K,V] entry state per chunk")

    output = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    def grid(meta):
        return (triton.cdiv(value_dim, meta["BV"]), metadata.capacity, value_heads)

    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        q_stride_t=q.stride(1),
        k_stride_t=k.stride(1),
        v_stride_t=v.stride(1),
        h=h,
        g=cumulative_gate,
        o=output,
        cu_seqlens=metadata.cu_seqlens,
        chunk_offsets=metadata.chunk_offsets,
        scale=scale,
        T=tokens,
        num_sequences=metadata.cu_seqlens.shape[0] - 1,
        H=key_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BT=64,
    )
    return output


__all__ = ["chunk_gdn_fwd_output_dense", "chunk_gdn_fwd_output_packed"]
