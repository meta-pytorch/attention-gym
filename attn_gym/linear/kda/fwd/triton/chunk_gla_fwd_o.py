# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
#
# Forward-only GLA output projection, based on
# genai/llama4x/llama4x/ops/fla/ops/gla/chunk.py.

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.triton.utils import can_use_tma, ptr_offset, requires_int64_offsets
from attn_gym.linear.kda.utils import (
    autotune_cache_kwargs,
    exp,
    exp2,
)


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_NUM_CHUNKS": lambda args: args["num_chunks"] is not None,
        "USE_INT64_OFFSETS": lambda args: requires_int64_offsets(
            args["q"],
            args["v"],
            args["g"],
            args["h"],
            args["o"],
            args["A"],
            args["cu_seqlens"],
            args["chunk_indices"],
            args["num_chunks"],
        ),
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BK": 64, "BV": 64}, num_warps=4, num_stages=4),
        triton.Config({"BK": 32, "BV": 64}, num_warps=2, num_stages=4),
        triton.Config({"BK": 32, "BV": 64}, num_warps=4, num_stages=4),
        triton.Config({"BK": 64, "BV": 32}, num_warps=4, num_stages=4),
        triton.Config({"BK": 64, "BV": 64}, num_warps=2, num_stages=4),
        triton.Config({"BK": 64, "BV": 64}, num_warps=8, num_stages=4),
    ],
    key=["H", "K", "V", "T", "BT"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "num_chunks"])
def chunk_gla_fwd_kernel_o(
    q,
    v,
    g,
    h,
    o,
    A,
    cu_seqlens,
    chunk_indices,
    num_chunks,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_NUM_CHUNKS: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_INT64_OFFSETS:
        i_v = i_v.to(tl.int64)
        i_t = i_t.to(tl.int64)
        i_bh = i_bh.to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        if HAS_NUM_CHUNKS and i_t >= tl.load(num_chunks):
            return
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + ptr_offset((i_t, 0), (2, 1))).to(tl.int32),
            tl.load(chunk_indices + ptr_offset((i_t, 1), (2, 1))).to(tl.int32),
        )
        if USE_INT64_OFFSETS:
            i_n = i_n.to(tl.int64)
            i_t = i_t.to(tl.int64)
        bos, eos = (
            tl.load(cu_seqlens + ptr_offset((i_n, 0), (1, 1))).to(tl.int32),
            tl.load(cu_seqlens + ptr_offset((i_n, 1), (1, 1))).to(tl.int32),
        )
        if USE_INT64_OFFSETS:
            bos = bos.to(tl.int64)
            eos = eos.to(tl.int64)
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos = i_b * T

    o_i = tl.arange(0, BT)
    o_t = i_t * BT + o_i
    o_v = i_v * BV + tl.arange(0, BV)
    m_t = o_t < T
    m_v = o_v < V
    m_tv = m_t[:, None] & m_v[None, :]
    m_s = o_i[:, None] >= o_i[None, :]

    q += ptr_offset((bos, i_h), (H * K, K))
    g += ptr_offset((bos, i_h), (H * K, K))
    h += ptr_offset((i_tg, i_h), (H * K * V, K * V))
    v += ptr_offset((bos, i_h), (H * V, V))
    o += ptr_offset((bos, i_h), (H * V, V))
    A += ptr_offset((bos, i_h), (H * BT, BT))

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_qg = m_t[:, None] & m_k[None, :]
        p_q = q + ptr_offset((o_t[:, None], o_k[None, :]), (H * K, 1))
        p_g = g + ptr_offset((o_t[:, None], o_k[None, :]), (H * K, 1))
        p_h = h + ptr_offset((o_k[:, None], o_v[None, :]), (V, 1))

        # [BT, BK]
        b_q = tl.load(p_q, mask=m_qg, other=0.0)
        # [BT, BK]
        b_g = tl.load(p_g, mask=m_qg, other=0.0).to(tl.float32)
        # [BT, BK]
        if USE_EXP2:
            b_qg = (b_q * exp2(b_g)).to(b_q.dtype)
        else:
            b_qg = (b_q * exp(b_g)).to(b_q.dtype)
        # [BK, BV]
        b_h = tl.load(p_h, mask=m_k[:, None] & m_v[None, :], other=0.0)
        # works but dkw, owing to divine benevolence
        # [BT, BV]
        if i_k >= 0:
            b_o += tl.dot(b_qg, b_h.to(b_qg.dtype))
    b_o *= scale
    p_v = v + ptr_offset((o_t[:, None], o_v[None, :]), (H * V, 1))
    p_o = o + ptr_offset((o_t[:, None], o_v[None, :]), (H * V, 1))
    p_A = A + ptr_offset((o_t[:, None], o_i[None, :]), (H * BT, 1))
    # [BT, BV]
    b_v = tl.load(p_v, mask=m_tv, other=0.0)
    # [BT, BT]
    b_A = tl.load(p_A, mask=m_t[:, None], other=0.0)
    b_A = tl.where(m_s, b_A, 0.0).to(b_v.dtype)
    b_o += tl.dot(b_A, b_v)
    tl.store(p_o, b_o.to(o.dtype.element_ty), mask=m_tv)


@triton.jit
def chunk_gla_fwd_kernel_o_tma(
    q_desc,
    v_desc,
    g_desc,
    h_desc,
    o_desc,
    A_desc,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Compose fixed KDA output tiles with TMA-backed tensor descriptors."""
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        b_q = q_desc.load([i_b, i_t * BT, i_h, i_k * BK])
        b_q = tl.reshape(b_q, [BT, BK])
        b_g = g_desc.load([i_b, i_t * BT, i_h, i_k * BK])
        b_g = tl.reshape(b_g, [BT, BK]).to(tl.float32)
        b_qg = (b_q * exp2(b_g)).to(b_q.dtype)

        b_h = h_desc.load([i_b, i_t, i_h, i_k * BK, i_v * BV])
        b_h = tl.reshape(b_h, [BK, BV]).to(b_qg.dtype)
        b_o += tl.dot(b_qg, b_h)

    b_o *= scale
    b_A = A_desc.load([i_b, i_t * BT, i_h, 0])
    b_A = tl.reshape(b_A, [BT, BT])
    o_i = tl.arange(0, BT)
    b_A = tl.where(o_i[:, None] >= o_i[None, :], b_A, 0.0)
    b_v = v_desc.load([i_b, i_t * BT, i_h, i_v * BV])
    b_v = tl.reshape(b_v, [BT, BV])
    b_o += tl.dot(b_A.to(b_v.dtype), b_v)
    o_desc.store(
        [i_b, i_t * BT, i_h, i_v * BV],
        tl.reshape(b_o.to(b_v.dtype), [1, BT, 1, BV]),
    )


def _can_use_tensor_descriptors(*tensors: torch.Tensor) -> bool:
    """Return whether all fixed KDA tensors satisfy host TMA requirements."""
    return all(can_use_tma(tensor) for tensor in tensors)


def chunk_gla_fwd_o_gk(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    scale: float,
    *,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Compose fixed-length KDA intra- and inter-chunk output terms."""
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if tokens % chunk_size:
        raise ValueError(f"the KDA output kernel requires complete chunks, got T={tokens}")
    chunks = tokens // chunk_size
    if g.shape != q.shape:
        raise ValueError("g must have the same shape as q")
    if v.shape != (batch, tokens, heads, value_dim):
        raise ValueError("v must have shape [B, T, H, V]")
    if A.shape != (batch, tokens, heads, chunk_size):
        raise ValueError(
            f"A must have shape {(batch, tokens, heads, chunk_size)}, got {tuple(A.shape)}"
        )
    expected_h_shape = (batch, chunks, heads, key_dim, value_dim)
    if h.shape != expected_h_shape:
        raise ValueError(f"h must have shape {expected_h_shape}, got {tuple(h.shape)}")

    output = torch.empty_like(v)
    if (key_dim, value_dim, chunk_size) == (128, 128, 64) and _can_use_tensor_descriptors(
        q, v, g, h, output, A
    ):
        block_key_dim = 32
        block_value_dim = 64
        chunk_gla_fwd_kernel_o_tma[
            (
                triton.cdiv(value_dim, block_value_dim),
                chunks,
                batch * heads,
            )
        ](
            TensorDescriptor.from_tensor(q, [1, chunk_size, 1, block_key_dim]),
            TensorDescriptor.from_tensor(v, [1, chunk_size, 1, block_value_dim]),
            TensorDescriptor.from_tensor(g, [1, chunk_size, 1, block_key_dim]),
            TensorDescriptor.from_tensor(h, [1, 1, 1, block_key_dim, block_value_dim]),
            TensorDescriptor.from_tensor(output, [1, chunk_size, 1, block_value_dim]),
            TensorDescriptor.from_tensor(A, [1, chunk_size, 1, chunk_size]),
            scale,
            H=heads,
            K=key_dim,
            V=value_dim,
            BT=chunk_size,
            BK=block_key_dim,
            BV=block_value_dim,
            num_warps=2,
            num_stages=3,
        )
    else:

        def grid(meta):
            return (triton.cdiv(value_dim, meta["BV"]), chunks, batch * heads)

        chunk_gla_fwd_kernel_o[grid](
            q=q,
            v=v,
            g=g,
            h=h,
            o=output,
            A=A,
            cu_seqlens=None,
            chunk_indices=None,
            num_chunks=None,
            scale=scale,
            T=tokens,
            H=heads,
            K=key_dim,
            V=value_dim,
            BT=chunk_size,
            USE_EXP2=True,
        )
    return output


__all__ = ["chunk_gla_fwd_o_gk"]
