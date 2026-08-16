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
# inclusive-tril masking of A and its grouped-value-head mapping
# ``key_head = value_head // (H_V // H_K)`` for k/gk):
#   w  = A @ (k * beta * exp2(gk))
#   u  = A @ (v * beta)
#   qg = q * exp2(gk)              (only when q and gk are provided)
#   kg = k * exp2(gk_last - gk)    (only when gk is provided)
#
# dot_precision selects the tensor-core operand precision, mirroring the CuTe
# kernel's knob: "bf16" rounds the fp32 operand products to bf16 before the
# dot; "tf32"/"tf32x3" keep them in fp32 registers and pass
# ``input_precision`` through to `tl.dot` (only meaningful for fp32 operands).
# W stays single-pass tf32 in tf32x3 mode, and tf32x3 requires fp32 A, both
# matching the CuTe contract. Note fp32->tf32 operand conversion truncates
# rather than rounds inside the MMA, a <=1-ulp-of-tf32 difference from the
# CuTe kernel's rounded conversion; both sit inside the mode's accuracy class.

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym._backends.triton.utils import ptr_offset
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata, load_ragged_chunk_work
from attn_gym.linear.kda.utils import autotune_cache_kwargs, exp2

# Compile-time dot-precision selector shared with the launch wrapper.
_PRECISION_MODES = {"bf16": 0, "tf32": 1, "tf32x3": 2}


@triton.heuristics(
    {
        "STORE_QG": lambda args: args["q"] is not None,
        "HAS_GK": lambda args: args["gk"] is not None,
        "IS_RAGGED": lambda args: args["chunk_offsets"] is not None,
    }
)
@triton.autotune(
    configs=[triton.Config({}, num_warps=w, num_stages=s) for w in [4, 8] for s in [2, 3]],
    key=["H", "HV", "K", "V", "BT", "BK", "BV", "IS_RAGGED", "HAS_GK", "STORE_QG", "PRECISION"],
    **autotune_cache_kwargs,
)
@triton.jit(
    do_not_specialize=[
        "T",
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
    chunk_offsets,
    T,
    q_stride_t,
    k_stride_t,
    v_stride_t,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    PRECISION: tl.constexpr,
    num_sequences,
    STORE_QG: tl.constexpr,
    HAS_GK: tl.constexpr,
    IS_RAGGED: tl.constexpr,
):
    """Recompute one chunk's W/U (and optional QG/KG) with register-operand dots."""
    i_c, i_hv = tl.program_id(0), tl.program_id(1).to(tl.int64)
    i_h = i_hv // (HV // H)
    if IS_RAGGED:
        if i_c >= tl.load(chunk_offsets + num_sequences):
            return
        i_n, i_t, token_start, _ = load_ragged_chunk_work(
            cu_seqlens, chunk_offsets, i_c, num_sequences, BT
        )
        # token_start == bos + i_t * BT; only eos still needs a load for masking.
        bos = (token_start - i_t * BT).to(tl.int64)
        eos = tl.load(cu_seqlens + ptr_offset((i_n,), (1,)) + 1).to(tl.int64)
        T_local = (eos - bos).to(tl.int32)
    else:
        i_t = i_c
        bos = 0
        T_local = T

    o_t = i_t.to(tl.int64) * BT + tl.arange(0, BT)
    m_t = o_t < T_local
    token = bos + o_t

    b_b = tl.load(beta + ptr_offset((token, i_hv), (HV, 1)), mask=m_t, other=0.0).to(tl.float32)

    o_A = tl.arange(0, BT)
    valid = T_local - i_t.to(tl.int32) * BT
    # Inclusive tril + row/col validity, matching the CuTe kernel's A masking.
    m_A = m_t[:, None] & (o_A[None, :] <= o_A[:, None]) & (o_A[None, :] < valid)
    b_A_raw = tl.load(
        A + ptr_offset((token[:, None], i_hv, o_A[None, :]), (HV * BT, BT, 1)),
        mask=m_A,
        other=0.0,
    )
    if PRECISION == 0:
        b_A = b_A_raw.to(k.dtype.element_ty)
    else:
        b_A = b_A_raw.to(tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_t[:, None] & (o_v[None, :] < V)
        b_v = tl.load(
            v + ptr_offset((token[:, None], i_hv, o_v[None, :]), (v_stride_t, V, 1)),
            mask=m_v,
            other=0.0,
        )
        b_vb = b_v * b_b[:, None]
        if PRECISION == 0:
            b_u = tl.dot(b_A, b_vb.to(b_v.dtype))
        elif PRECISION == 1:
            b_u = tl.dot(b_A, b_vb, input_precision="tf32")
        else:
            b_u = tl.dot(b_A, b_vb, input_precision="tf32x3")
        tl.store(
            u + ptr_offset((token[:, None], i_hv, o_v[None, :]), (HV * V, V, 1)),
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
                    q + ptr_offset((token[:, None], i_hv, o_k[None, :]), (q_stride_t, K, 1)),
                    mask=m_tk,
                    other=0.0,
                )
                tl.store(
                    qg + ptr_offset((token[:, None], i_hv, o_k[None, :]), (HV * K, K, 1)),
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
                kg + ptr_offset((token[:, None], i_hv, o_k[None, :]), (HV * K, K, 1)),
                b_kg.to(kg.dtype.element_ty),
                mask=m_tk,
            )
        # W stays single-pass tf32 even in tf32x3 mode, matching the CuTe contract.
        if PRECISION == 0:
            b_w = tl.dot(b_A, b_kb.to(b_k.dtype))
        else:
            b_w = tl.dot(b_A, b_kb, input_precision="tf32")
        tl.store(
            w + ptr_offset((token[:, None], i_hv, o_k[None, :]), (HV * K, K, 1)),
            b_w.to(w.dtype.element_ty),
            mask=m_tk,
        )


def recompute_w_u_fwd_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    metadata: RaggedChunkMetadata | None = None,
    q: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    *,
    chunk_size: int = 64,
    dot_precision: str = "bf16",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Launch the register-operand recompute for packed B=1 inputs."""
    batch, tokens, key_heads, key_dim = k.shape
    value_heads, value_dim = v.shape[2], v.shape[3]
    if batch != 1:
        raise ValueError(f"recompute_w_u_fwd_triton requires B=1, got B={batch}")
    if dot_precision not in _PRECISION_MODES:
        raise ValueError(
            f"dot_precision must be one of {sorted(_PRECISION_MODES)}, got {dot_precision!r}"
        )
    if value_heads % key_heads:
        raise ValueError(
            f"value heads ({value_heads}) must be divisible by key heads ({key_heads})"
        )
    if v.shape[:2] != (batch, tokens):
        raise ValueError(f"v must have shape [1, T, H_V, V], got {tuple(v.shape)}")
    if A.shape != (batch, tokens, value_heads, chunk_size):
        raise ValueError(f"A must have shape [1, T, H_V, {chunk_size}], got {tuple(A.shape)}")
    if beta.shape != (batch, tokens, value_heads):
        raise ValueError(f"beta must have shape [1, T, H_V], got {tuple(beta.shape)}")
    if gk is not None and gk.shape != k.shape:
        raise ValueError(f"gk must have shape {tuple(k.shape)}, got {tuple(gk.shape)}")
    if q is not None and q.shape != (batch, tokens, value_heads, key_dim):
        raise ValueError(
            f"q must have shape {(batch, tokens, value_heads, key_dim)}, got {tuple(q.shape)}"
        )
    for name, tensor, dtypes in (
        ("k", k, (torch.bfloat16,)),
        ("v", v, (torch.bfloat16,)),
        ("q", q, (torch.bfloat16,)),
        ("beta", beta, (torch.float32, torch.bfloat16)),
        ("gk", gk, (torch.float32,)),
        ("A", A, (torch.float32, torch.bfloat16)),
    ):
        if tensor is not None and tensor.dtype not in dtypes:
            raise ValueError(f"{name} must be one of {dtypes}, got {tensor.dtype}")
    if dot_precision == "tf32x3" and A.dtype != torch.float32:
        raise ValueError("dot_precision='tf32x3' requires fp32 A")
    # q/k/v may carry a strided token dimension (fused-QKV views); heads must be
    # compact and the channel dimension contiguous. Small tensors stay contiguous.
    for name, tensor in (("q", q), (("k"), k), ("v", v)):
        if tensor is not None and (
            tensor.stride(-1) != 1 or tensor.stride(-2) != tensor.shape[-1]
        ):
            raise ValueError(f"recompute_w_u_fwd_triton requires compact heads in {name}")
    for name, tensor in (("beta", beta), ("A", A), ("gk", gk)):
        if tensor is not None and not tensor.is_contiguous():
            raise ValueError(f"recompute_w_u_fwd_triton requires contiguous {name}")
    if metadata is not None:
        metadata.validate_chunk_size(chunk_size)
    elif tokens % chunk_size:
        raise ValueError(
            "dense recompute_w_u_fwd_triton requires complete chunks, "
            f"got T={tokens} and chunk_size={chunk_size}"
        )
    has_gk = gk is not None
    has_q = q is not None and has_gk

    w = k.new_empty(batch, tokens, value_heads, key_dim)
    u = v.new_empty(batch, tokens, value_heads, value_dim)
    qg = k.new_empty(batch, tokens, value_heads, key_dim) if has_q else None
    kg = k.new_empty(batch, tokens, value_heads, key_dim) if has_gk else None
    # Fake tracing stops at output metadata, matching the sibling stages
    # (chunk_kda_fwd_intra); the autotuned launch must not run.
    if isinstance(k, FakeTensor):
        return w, u, qg, kg
    chunks = metadata.capacity if metadata is not None else tokens // chunk_size
    if chunks:
        recompute_w_u_fwd_kernel[(chunks, value_heads)](
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
            cu_seqlens=None if metadata is None else metadata.cu_seqlens,
            chunk_offsets=None if metadata is None else metadata.chunk_offsets,
            T=tokens,
            q_stride_t=q.stride(1) if has_q else 0,
            k_stride_t=k.stride(1),
            v_stride_t=v.stride(1),
            H=key_heads,
            HV=value_heads,
            K=key_dim,
            V=value_dim,
            BT=chunk_size,
            BK=64,
            BV=64,
            PRECISION=_PRECISION_MODES[dot_precision],
            num_sequences=0 if metadata is None else metadata.cu_seqlens.shape[0] - 1,
        )
    return w, u, qg, kg


__all__ = ["recompute_w_u_fwd_triton"]
