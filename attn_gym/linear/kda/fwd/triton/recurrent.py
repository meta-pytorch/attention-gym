# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inference-only KDA wrappers for recurrent prefill and fused decode.

The vector-gate prefill specialization uses the shared recurrent delta-rule scan;
use ``chunk_kda`` for training.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym.linear._delta_rule.recurrent import GateKind, launch_recurrent_delta_rule_fwd
from attn_gym.linear.kda.ops import recurrent_decode_forward as decode_forward
from attn_gym.linear.kda.ops import recurrent_decode_op as _recurrent_decode_op
from attn_gym.linear.kda.ops import recurrent_forward as forward
from attn_gym.linear.kda.ops import (
    recurrent_fwd_no_state_op as _recurrent_fwd_no_state_op,
)
from attn_gym.linear.kda.ops import recurrent_fwd_op as _recurrent_fwd_op
from attn_gym.linear.kda.ops import (
    recurrent_fwd_paged_op as _recurrent_fwd_paged_op,
)


# Use a flat 1D grid of B * H * ceil(V / BV) programs. Each program owns one
# (batch row, head, V tile), processes all K channels for that tile, and gets
# contiguous K-dimension accesses from the [V, K] state layout.
@triton.jit
def kda_recurrent_decode_kernel(
    packed_qkv,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    output,
    state_cache,
    state_indices,
    lower_bound,
    scale,
    state_batch_stride: tl.constexpr,
    qkv_token_stride: tl.constexpr,
    gate_token_stride: tl.constexpr,
    beta_token_stride: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_h = i_nh // H, i_nh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_vk = m_v[:, None] & m_k[None, :]

    i_state = tl.load(state_indices + i_n).to(tl.int64)
    row = i_n * H + i_h
    if i_state <= 0:
        tl.store(output + row * V + o_v, 0.0, mask=m_v)
        return

    p_state = i_state * state_batch_stride + i_h * V * K + o_v[:, None] * K + o_k[None, :]
    b_state = tl.load(state_cache + p_state, mask=m_vk, other=0.0).to(tl.float32)

    p_qkv = packed_qkv + i_n * qkv_token_stride
    b_q = tl.load(p_qkv + i_h * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_k = tl.load(p_qkv + H * K + i_h * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_v = tl.load(p_qkv + 2 * H * K + i_h * V + o_v, mask=m_v, other=0.0).to(tl.float32)
    b_q *= tl.rsqrt(tl.sum(b_q * b_q) + 1e-6) * scale
    b_k *= tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)

    p_gate = raw_gate + i_n * gate_token_stride + i_h * K + o_k
    b_gate_input = tl.load(p_gate, mask=m_k, other=0.0).to(tl.float32)
    b_gate_input += tl.load(dt_bias + i_h * K + o_k, mask=m_k, other=0.0).to(tl.float32)
    b_a = tl.exp(tl.load(A_log + i_h).to(tl.float32))
    if USE_LOWER_BOUND:
        b_gate = lower_bound * tl.sigmoid(b_a * b_gate_input)
    else:
        b_softplus = tl.where(
            b_gate_input > 20.0,
            b_gate_input,
            tl.where(
                b_gate_input < -10.0,
                tl.exp(b_gate_input),
                tl.log(1.0 + tl.exp(b_gate_input)),
            ),
        )
        b_gate = -b_a * b_softplus

    b_state *= tl.exp(b_gate)[None, :]
    b_delta = b_v - tl.sum(b_state * b_k[None, :], axis=1)
    b_beta = tl.sigmoid(tl.load(raw_beta + i_n * beta_token_stride + i_h).to(tl.float32))
    b_delta *= b_beta
    b_state += b_delta[:, None] * b_k[None, :]
    b_o = tl.sum(b_state * b_q[None, :], axis=1)
    tl.store(output + row * V + o_v, b_o.to(output.dtype.element_ty), mask=m_v)
    tl.store(state_cache + p_state, b_state, mask=m_vk)


def _launch_kda_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    *,
    store_final_state: bool,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    autotune: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch the vector-gate specialization used by recurrent KDA."""
    return launch_recurrent_delta_rule_fwd(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale=q.shape[-1] ** -0.5,
        gate_kind=GateKind.VECTOR,
        store_final_state=store_final_state,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        autotune=autotune,
    )


def _kda_recurrent_fwd_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    autotune: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, final_state = _launch_kda_recurrent_fwd(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        store_final_state=True,
        autotune=autotune,
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
    autotune: bool,
) -> torch.Tensor:
    return _launch_kda_recurrent_fwd(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        store_final_state=False,
        autotune=autotune,
    )[0]


def _kda_recurrent_fwd_paged_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    return _launch_kda_recurrent_fwd(
        q,
        k,
        v,
        gate,
        beta,
        state_cache,
        cu_seqlens,
        store_final_state=True,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        autotune=False,
    )[0]


def _kda_recurrent_decode_cuda(
    packed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    output: torch.Tensor,
    lower_bound: float,
    use_lower_bound: bool,
    scale: float,
) -> None:
    read_only_inputs = (packed_qkv, raw_gate, raw_beta, A_log, dt_bias, state_indices)
    if any(torch._C._overlaps(output, tensor) for tensor in (*read_only_inputs, state_cache)):
        raise ValueError("out must not alias any recurrent_kda_decode input")
    if any(torch._C._overlaps(state_cache, tensor) for tensor in read_only_inputs):
        raise ValueError("state_cache must not alias recurrent_kda_decode read-only inputs")

    batch = packed_qkv.shape[0]
    heads, value_dim, key_dim = state_cache.shape[1:]
    if value_dim <= 32:
        block_v, num_warps = min(triton.next_power_of_2(value_dim), 8), 4
    elif batch * heads <= 8:
        block_v, num_warps = min(triton.next_power_of_2(value_dim), 16), 2
    else:
        block_v, num_warps = min(triton.next_power_of_2(value_dim), 16), 1
    grid = (triton.cdiv(value_dim, block_v) * batch * heads,)
    kda_recurrent_decode_kernel[grid](
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        output,
        state_cache,
        state_indices,
        lower_bound,
        scale=scale,
        state_batch_stride=state_cache.stride(0),
        qkv_token_stride=packed_qkv.stride(0),
        gate_token_stride=raw_gate.stride(1),
        beta_token_stride=raw_beta.stride(1),
        H=heads,
        K=key_dim,
        V=value_dim,
        BK=triton.next_power_of_2(key_dim),
        BV=block_v,
        USE_LOWER_BOUND=use_lower_bound,
        num_warps=num_warps,
    )


__all__ = [
    "_recurrent_decode_op",
    "_recurrent_fwd_no_state_op",
    "_recurrent_fwd_op",
    "_recurrent_fwd_paged_op",
    "decode_forward",
    "forward",
]
