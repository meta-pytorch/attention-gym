# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused O(T) KDA recurrence for decode and inference prefill.

The kernel scans tokens sequentially per (sequence, head, value block), holding the
FP32 recurrent state in registers. It mirrors :func:`naive_recurrent_kda` exactly:
per step the state decays by ``exp2(gate)`` per key channel, a beta-scaled delta
writes the new value, and the query reads the updated state. The operation is
inference-only; use ``chunk_kda`` for training.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym.linear.kda.validation import validate_kda_inputs

_MAX_KEY_DIM = 256


@triton.jit
def kda_recurrent_fwd_kernel(
    q,
    k,
    v,
    gate,
    beta,
    output,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_h = i_nh // H, i_nh % H

    if IS_VARLEN:
        # Assumption: seqlens have been validated prior to call
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos = i_n * T
        eos = bos + T

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_kv = m_k[:, None] & m_v[None, :]

    p_state = i_n * H * K * V + i_h * K * V + o_k[:, None] * V + o_v[None, :]
    if USE_INITIAL_STATE:
        b_state = tl.load(h0 + p_state, mask=m_kv, other=0.0).to(tl.float32)
    else:
        b_state = tl.zeros([BK, BV], dtype=tl.float32)

    for t in range(bos, eos):
        row = t * H + i_h
        b_q = tl.load(q + row * K + o_k, mask=m_k, other=0.0).to(tl.float32) * scale
        b_k = tl.load(k + row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
        b_g = tl.load(gate + row * K + o_k, mask=m_k, other=0.0).to(tl.float32)
        b_beta = tl.load(beta + row).to(tl.float32)
        b_v = tl.load(v + row * V + o_v, mask=m_v, other=0.0).to(tl.float32)

        b_state *= tl.exp2(b_g)[:, None]
        b_delta = (b_v - tl.sum(b_k[:, None] * b_state, 0)) * b_beta
        b_state += b_k[:, None] * b_delta[None, :]
        b_o = tl.sum(b_q[:, None] * b_state, 0)
        tl.store(output + row * V + o_v, b_o.to(output.dtype.element_ty), mask=m_v)

    if STORE_FINAL_STATE:
        tl.store(ht + p_state, b_state, mask=m_kv)


def _launch_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    store_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Allocate outputs and launch the sequential scan over token spans."""
    batch, tokens, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    num_sequences = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    output = torch.empty_like(v, dtype=q.dtype)
    final_state = (
        q.new_empty(num_sequences, heads, key_dim, value_dim, dtype=torch.float32)
        if store_final_state
        else None
    )
    # BV=32 measured faster than 64 on B200 for both decode and prefill
    # (smaller state tiles schedule better; state traffic is identical).
    block_v = min(triton.next_power_of_2(value_dim), 32)
    # One flat launch dimension: sequence-head counts can exceed the 65,535
    # grid-Y limit, while grid-X is effectively unbounded.
    grid = (triton.cdiv(value_dim, block_v) * num_sequences * heads,)
    kda_recurrent_fwd_kernel[grid](
        q,
        k,
        v,
        gate,
        beta,
        output,
        initial_state,
        final_state,
        cu_seqlens,
        scale=key_dim**-0.5,
        T=tokens,
        H=heads,
        K=key_dim,
        V=value_dim,
        BK=triton.next_power_of_2(key_dim),
        BV=block_v,
        USE_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=store_final_state,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=4,
    )
    return output, final_state


_RECURRENT_FWD_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta,"
    " Tensor? initial_state, Tensor? cu_seqlens)"
)
torch.library.define("attn_gym::kda_recurrent_fwd", _RECURRENT_FWD_ARGS + " -> (Tensor, Tensor)")
torch.library.define("attn_gym::kda_recurrent_fwd_no_state", _RECURRENT_FWD_ARGS + " -> Tensor")


def _kda_recurrent_fwd_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, final_state = _launch_recurrent_fwd(
        q, k, v, gate, beta, initial_state, cu_seqlens, store_final_state=True
    )
    assert final_state is not None
    return output, final_state


def _kda_recurrent_fwd_no_state_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    return _launch_recurrent_fwd(
        q, k, v, gate, beta, initial_state, cu_seqlens, store_final_state=False
    )[0]


torch.library.impl("attn_gym::kda_recurrent_fwd", "CUDA", _kda_recurrent_fwd_cuda)
torch.library.impl(
    "attn_gym::kda_recurrent_fwd_no_state", "CUDA", _kda_recurrent_fwd_no_state_cuda
)


@torch.library.register_fake("attn_gym::kda_recurrent_fwd")
def _kda_recurrent_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    del gate, beta, initial_state
    num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    final_state = q.new_empty(
        num_sequences, q.shape[2], q.shape[3], v.shape[-1], dtype=torch.float32
    )
    return torch.empty_like(v, dtype=q.dtype), final_state


@torch.library.register_fake("attn_gym::kda_recurrent_fwd_no_state")
def _kda_recurrent_fwd_no_state_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    del gate, beta, initial_state, cu_seqlens
    return torch.empty_like(v, dtype=q.dtype)


_recurrent_fwd_op = torch.ops.attn_gym.kda_recurrent_fwd.default
_recurrent_fwd_no_state_op = torch.ops.attn_gym.kda_recurrent_fwd_no_state.default


def _validate_recurrent_kda_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> None:
    """Validate the shared contract plus the fused scan's own constraints."""
    validate_kda_inputs(
        q, k, v, gate, beta, initial_state, cu_seqlens, op_name="recurrent_kda", gate_name="gate"
    )
    if q.shape[-1] > _MAX_KEY_DIM:
        raise ValueError(f"recurrent_kda requires K in [1, {_MAX_KEY_DIM}], got {q.shape[-1]}")
    if not q.is_cuda:
        raise ValueError("recurrent_kda requires CUDA tensors")
    data_tensors = (q, k, v, gate, beta)
    if initial_state is not None:
        data_tensors += (initial_state,)
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in data_tensors):
        raise RuntimeError(
            "recurrent_kda is inference-only and has no backward; use chunk_kda for "
            "training or call under torch.no_grad() / torch.inference_mode()"
        )


def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply the fused O(T) KDA delta rule for decode and inference prefill.

    Args:
        q: Queries with shape ``[B, T, H, K]``; scaled by ``1/sqrt(K)`` internally.
        k: Keys with the same shape as ``q``.
        v: Values with shape ``[B, T, H, V]``.
        gate: Per-token log2 decay with the same shape as ``q``, as produced by
            ``bounded_gate_cumsum(chunk_size=1)`` — not the chunk-local
            cumulative gate that ``chunk_kda`` consumes.
        beta: Per-token write gate with shape ``[B, T, H]``.
        initial_state: Optional recurrent state with one ``[H, K, V]`` entry per
            logical sequence.
        cu_seqlens: Optional device-resident int32 offsets selecting packed
            ``[1, T, H, D]`` execution. Repeated offsets are empty padding slots
            whose state passes through unchanged, and the terminal offset may sit
            below the physical token capacity; values past it are outside the
            operation's contract, so fixed-shape CUDA graphs can replay with
            different boundaries and active lengths.
        output_final_state: Also return the final recurrent state. When false,
            the state is neither allocated nor written.

    Returns:
        The output in ``q.dtype`` and, when requested, the FP32 recurrent state.

    The scan computes in FP32 regardless of input dtype. The fused scan is
    inference-only: when autograd is enabled, calls whose data inputs require
    gradients are rejected instead of silently detaching.
    """
    _validate_recurrent_kda_inputs(q, k, v, gate, beta, initial_state, cu_seqlens)
    # The kernel loads every operand through an FP32 register cast, so only the
    # layout needs normalizing here; recurrent states are always produced in FP32.
    q, k, v, gate, beta = (tensor.contiguous() for tensor in (q, k, v, gate, beta))
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    if output_final_state:
        return _recurrent_fwd_op(q, k, v, gate, beta, initial_state, cu_seqlens)
    return _recurrent_fwd_no_state_op(q, k, v, gate, beta, initial_state, cu_seqlens), None


__all__ = ["recurrent_kda"]
