# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Register-operand (warp MMA) recompute of the KDA W/U intermediates.
#
# This operator is memory-bound at chunk_size=64 (~11 FLOP/byte vs the ~350
# FLOP/byte B200 balance point), so tensor-core operand ceremony is pure
# overhead: `tl.dot` consumes register operands directly and converts the FP32
# gates in flight, avoiding the staging -> convert -> swizzled-SMEM round trip
# the tcgen05/UMMA kernel pays (~40% of its runtime at BT=64).
#
# Computes per chunk (matching the CuTe kernel's contract, including its
# inclusive-tril masking of A):
#   w  = A @ (k * beta * exp2(gk))
#   u  = A @ (v * beta)
#   qg = q * exp2(gk)              (only when q and gk are provided)
#   kg = k * exp2(gk_last - gk)    (only when gk is provided)

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata, load_ragged_chunk_work
from attn_gym.linear.kda.utils import (
    ChunkMetadata,
    autotune_cache_kwargs,
    exp2,
)


@triton.heuristics(
    {
        "STORE_QG": lambda args: args["q"] is not None,
        "HAS_GK": lambda args: args["gk"] is not None,
        "IS_RAGGED": lambda args: args["chunk_offsets"] is not None,
        "HAS_NUM_CHUNKS": lambda args: args["num_chunks"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=w, num_stages=s) for w in [4, 8] for s in [2, 3]],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_RAGGED"],
    **autotune_cache_kwargs,
)
@triton.jit(
    do_not_specialize=[
        "T",
        "num_chunks",
        "num_sequences",
        "q_stride_t",
        "k_stride_t",
        "v_stride_t",
    ]
)
def recompute_w_u_fwd_kernel(
    q,
    k,
    qg,
    kg,
    v,
    beta,
    w,
    u,
    A,
    gk,
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    num_chunks,
    T,
    q_stride_t,
    k_stride_t,
    v_stride_t,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    num_sequences,
    STORE_QG: tl.constexpr,
    HAS_GK: tl.constexpr,
    IS_RAGGED: tl.constexpr,
    HAS_NUM_CHUNKS: tl.constexpr,
):
    """Recompute one chunk's W/U (and optional QG/KG) with register-operand dots."""
    i_c, i_h = tl.program_id(0), tl.program_id(1).to(tl.int64)
    if IS_RAGGED:
        if i_c >= tl.load(chunk_offsets + num_sequences):
            return
        i_n, i_t, _, _ = load_ragged_chunk_work(cu_seqlens, chunk_offsets, i_c, num_sequences, BT)
    else:
        if HAS_NUM_CHUNKS and i_c >= tl.load(num_chunks):
            return
        i_n, i_t = (
            tl.load(chunk_indices + ptr_offset((i_c, 0), (2, 1))).to(tl.int32),
            tl.load(chunk_indices + ptr_offset((i_c, 1), (2, 1))).to(tl.int32),
        )
    bos = tl.load(cu_seqlens + ptr_offset((i_n,), (1,))).to(tl.int64)
    eos = tl.load(cu_seqlens + ptr_offset((i_n,), (1,)) + 1).to(tl.int64)
    T_local = (eos - bos).to(tl.int32)

    o_t = i_t.to(tl.int64) * BT + tl.arange(0, BT)
    m_t = o_t < T_local
    token = bos + o_t

    b_b = tl.load(beta + ptr_offset((token, i_h), (H, 1)), mask=m_t, other=0.0).to(tl.float32)

    o_A = tl.arange(0, BT)
    valid = T_local - i_t.to(tl.int32) * BT
    # Inclusive tril + row/col validity, matching the CuTe kernel's A masking.
    m_A = m_t[:, None] & (o_A[None, :] <= o_A[:, None]) & (o_A[None, :] < valid)
    b_A = tl.load(
        A + ptr_offset((token[:, None], i_h, o_A[None, :]), (H * BT, BT, 1)),
        mask=m_A,
        other=0.0,
    ).to(k.dtype.element_ty)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        b_v = tl.load(
            v + ptr_offset((token[:, None], i_h, o_v[None, :]), (v_stride_t, V, 1)),
            mask=m_v,
            other=0.0,
        )
        b_u = tl.dot(b_A, (b_v * b_b[:, None]).to(b_v.dtype))
        tl.store(
            u + ptr_offset((token[:, None], i_h, o_v[None, :]), (H * V, V, 1)),
            b_u.to(u.dtype.element_ty),
            mask=m_v,
        )

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_tk = m_t[:, None] & m_k[None, :]
        b_k = tl.load(
            k + ptr_offset((token[:, None], i_h, o_k[None, :]), (k_stride_t, K, 1)),
            mask=m_tk,
            other=0.0,
        )
        b_kb = b_k * b_b[:, None]
        if HAS_GK:
            b_gk = tl.load(
                gk + ptr_offset((token[:, None], i_h, o_k[None, :]), (H * K, K, 1)),
                mask=m_tk,
                other=0.0,
            ).to(tl.float32)
            b_kb = b_kb * exp2(b_gk)
            if STORE_QG:
                b_q = tl.load(
                    q + ptr_offset((token[:, None], i_h, o_k[None, :]), (q_stride_t, K, 1)),
                    mask=m_tk,
                    other=0.0,
                )
                tl.store(
                    qg + ptr_offset((token[:, None], i_h, o_k[None, :]), (H * K, K, 1)),
                    (b_q * exp2(b_gk)).to(qg.dtype.element_ty),
                    mask=m_tk,
                )
            last_idx = bos + min(i_t * BT + BT, T_local) - 1
            b_gn = tl.load(
                gk + ptr_offset((last_idx, i_h, o_k), (H * K, K, 1)),
                mask=m_k,
                other=0.0,
            ).to(tl.float32)
            b_kg = b_k * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_gk), 0.0)
            tl.store(
                kg + ptr_offset((token[:, None], i_h, o_k[None, :]), (H * K, K, 1)),
                b_kg.to(kg.dtype.element_ty),
                mask=m_tk,
            )
        b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        tl.store(
            w + ptr_offset((token[:, None], i_h, o_k[None, :]), (H * K, K, 1)),
            b_w.to(w.dtype.element_ty),
            mask=m_tk,
        )


@torch.no_grad()
def recompute_w_u_fwd_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    metadata: ChunkMetadata | RaggedChunkMetadata,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    *,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Launch the register-operand recompute for packed B=1 inputs."""
    batch, tokens, heads, key_dim = k.shape
    value_dim = v.shape[-1]
    if batch != 1:
        raise ValueError(f"recompute_w_u_fwd_triton requires B=1, got B={batch}")
    if A.shape != (batch, tokens, heads, chunk_size):
        raise ValueError(f"A must have shape [1, T, H, {chunk_size}], got {tuple(A.shape)}")
    ragged = isinstance(metadata, RaggedChunkMetadata)
    if ragged:
        metadata.validate_chunk_size(chunk_size)
    # q/k/v may carry a strided token dimension (fused-QKV views); heads must be
    # compact and the channel dimension contiguous. Small tensors stay contiguous.
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor is not None and (
            tensor.stride(-1) != 1 or tensor.stride(-2) != tensor.shape[-1]
        ):
            raise ValueError(f"recompute_w_u_fwd_triton requires compact heads in {name}")
    for name, tensor in (("beta", beta), ("A", A), ("gk", gk)):
        if tensor is not None and not tensor.is_contiguous():
            raise ValueError(f"recompute_w_u_fwd_triton requires contiguous {name}")
    has_gk = gk is not None
    has_q = q is not None and has_gk

    w = k.new_empty(batch, tokens, heads, key_dim)
    u = v.new_empty(batch, tokens, heads, value_dim)
    qg = k.new_empty(batch, tokens, heads, key_dim) if has_q else None
    kg = k.new_empty(batch, tokens, heads, key_dim) if has_gk else None
    chunks = metadata.capacity if ragged else tokens // chunk_size
    if chunks:
        recompute_w_u_fwd_kernel[(chunks, heads)](
            q=q if has_q else None,
            k=k,
            qg=qg,
            kg=kg,
            v=v,
            beta=beta,
            w=w,
            u=u,
            A=A,
            gk=gk,
            cu_seqlens=metadata.cu_seqlens,
            chunk_indices=None if ragged else metadata.chunk_indices,
            chunk_offsets=metadata.chunk_offsets if ragged else None,
            num_chunks=None if ragged else metadata.num_chunks,
            T=tokens,
            q_stride_t=q.stride(1) if has_q else 0,
            k_stride_t=k.stride(1),
            v_stride_t=v.stride(1),
            H=heads,
            K=key_dim,
            V=value_dim,
            BT=chunk_size,
            BK=64,
            BV=64,
            num_sequences=metadata.cu_seqlens.shape[0] - 1 if ragged else 0,
        )
    return w, u, qg, kg


__all__ = ["recompute_w_u_fwd_triton"]
