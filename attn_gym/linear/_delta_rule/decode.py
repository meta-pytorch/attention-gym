# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused one-token paged decode shared by scalar- and vector-gated delta rules.

One kernel advances each sequence by a single token: it slices q/k/v out of the packed
post-convolution QKV buffer, applies the gate transform to the raw projections, L2-normalizes
q and k, runs one delta-rule step against the selected cache slot, and writes the slot back —
so serving callers launch no separate elementwise kernels. The recurrence matches the shared
scan in :mod:`attn_gym.linear._delta_rule.recurrent`, with the cache storing the transpose
``[V, K]`` of the scan's logical ``[K, V]`` state:

    state  = exp(gate) * state       # GateKind.SCALAR: one decay per value head
                                     # GateKind.VECTOR: one decay per key channel
    delta  = sigmoid(raw_beta) * (v - k @ state)
    state += outer(k, delta)
    out    = scale * l2norm(q) @ state

Gate transforms (`USE_LOWER_BOUND` selects the first):

    bounded:  lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))
    softplus: -exp(A_log) * softplus(raw_gate + dt_bias)

Callers may pass fewer q/k heads than value heads (multi-value attention): each block of
``H // HK`` consecutive value heads shares one q/k head inside the packed buffer, while the
gate stays per value head.
"""

from __future__ import annotations

from enum import Enum

import torch
import triton
import triton.language as tl

from attn_gym._backends.cute.utils import get_device_properties
from attn_gym._backends.triton.utils import ptr_offset
from attn_gym.linear._delta_rule.recurrent import GateKind


class GateTransform(Enum):
    """Pointwise transform applied to the raw gate projection in-kernel."""

    BOUNDED = "bounded"  # lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))
    SOFTPLUS = "softplus"  # -exp(A_log) * softplus(raw_gate + dt_bias)


def _decode_launch_config(
    value_dim: int, sequence_heads: int, use_hopper_gdn_config: bool
) -> tuple[int, int]:
    """Select the value tile and warp count for one-token decode."""
    if value_dim <= 32:
        return min(triton.next_power_of_2(value_dim), 8), 4
    # H100 measurements become inconclusive above 96 sequence-heads.
    block_v = 8 if use_hopper_gdn_config and sequence_heads < 104 else 16
    return min(triton.next_power_of_2(value_dim), block_v), 2 if sequence_heads <= 8 else 1


# Use a flat 1D grid of B * H * ceil(V / BV) programs. Each program owns one
# (batch row, value head, V tile), processes all K channels for that tile, and gets
# contiguous K-dimension accesses from the [V, K] state layout.
@triton.jit
def _recurrent_delta_rule_decode_kernel(
    packed_qkv,
    raw_gate,
    raw_beta,
    A_log,
    dt_bias,
    output,
    state_cache,
    state_indices,
    has_initial_state,
    lower_bound,
    scale,
    state_batch_stride: tl.constexpr,
    qkv_token_stride: tl.constexpr,
    gate_token_stride: tl.constexpr,
    beta_token_stride: tl.constexpr,
    H: tl.constexpr,
    HK: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
    SCALAR_GATE: tl.constexpr,
    USE_HAS_INITIAL_STATE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_n, i_h = i_nh // H, i_nh % H
    i_hk = i_h // (H // HK)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    m_vk = m_v[:, None] & m_k[None, :]

    i_state = tl.load(state_indices + i_n).to(tl.int64)
    row = i_n * H + i_h
    if i_state <= 0:
        tl.store(output + ptr_offset((row, o_v), (V, 1)), 0.0, mask=m_v)
        return

    p_state = ptr_offset(
        (i_state, i_h, o_v[:, None], o_k[None, :]),
        (state_batch_stride, V * K, K, 1),
    )
    m_state = m_vk
    if USE_HAS_INITIAL_STATE:
        m_state &= tl.load(has_initial_state + i_n)
    b_state = tl.load(state_cache + p_state, mask=m_state, other=0.0).to(tl.float32)

    p_qkv = packed_qkv + i_n * qkv_token_stride
    b_q = tl.load(p_qkv + ptr_offset((i_hk, o_k), (K, 1)), mask=m_k, other=0.0).to(tl.float32)
    b_k = tl.load(p_qkv + HK * K + ptr_offset((i_hk, o_k), (K, 1)), mask=m_k, other=0.0).to(
        tl.float32
    )
    b_v = tl.load(p_qkv + 2 * HK * K + ptr_offset((i_h, o_v), (V, 1)), mask=m_v, other=0.0).to(
        tl.float32
    )
    b_q *= tl.rsqrt(tl.sum(b_q * b_q) + 1e-6) * scale
    b_k *= tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)

    if SCALAR_GATE:
        b_gate_input = tl.load(raw_gate + i_n * gate_token_stride + i_h).to(tl.float32)
        b_gate_input += tl.load(dt_bias + i_h).to(tl.float32)
    else:
        # The gate decays each value head's own state; only q/k are shared across a group.
        p_gate = raw_gate + i_n * gate_token_stride + ptr_offset((i_h, o_k), (K, 1))
        b_gate_input = tl.load(p_gate, mask=m_k, other=0.0).to(tl.float32)
        b_gate_input += tl.load(dt_bias + ptr_offset((i_h, o_k), (K, 1)), mask=m_k, other=0.0).to(
            tl.float32
        )
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

    if SCALAR_GATE:
        b_state *= tl.exp(b_gate)
    else:
        b_state *= tl.exp(b_gate)[None, :]
    b_delta = b_v - tl.sum(b_state * b_k[None, :], axis=1)
    b_beta = tl.sigmoid(tl.load(raw_beta + i_n * beta_token_stride + i_h).to(tl.float32))
    b_delta *= b_beta
    b_state += b_delta[:, None] * b_k[None, :]
    b_o = tl.sum(b_state * b_q[None, :], axis=1)
    tl.store(output + ptr_offset((row, o_v), (V, 1)), b_o.to(output.dtype.element_ty), mask=m_v)
    tl.store(state_cache + p_state, b_state, mask=m_vk)


def launch_recurrent_delta_rule_decode(
    packed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    output: torch.Tensor,
    *,
    gate_kind: GateKind,
    gate_transform: GateTransform,
    key_heads: int,
    lower_bound: float,
    scale: float,
    has_initial_state: torch.Tensor | None,
    op_name: str,
) -> None:
    """Launch a scalar- or vector-gated fused decode specialization.

    Args:
        packed_qkv: ``[B, HK*K | HK*K | H*V]`` per-token buffer; within each section head
            rows are contiguous. ``HK == key_heads`` may divide ``H``.
        raw_gate: Unactivated token-major gate per value head, ``[B, H]`` for
            ``GateKind.SCALAR`` or ``[B, H, K]`` for ``GateKind.VECTOR``.
        raw_beta: Unactivated token-major write gate shaped ``[B, H]``, activated in-kernel
            as ``sigmoid``.
        A_log: FP32 per-head log decay parameter shaped ``[H]``.
        dt_bias: FP32 gate bias, ``[H]`` for scalar or ``[H, K]`` for vector gates.
        state_cache: Mutable FP32 pool shaped ``[num_slots, H, V, K]``; selected slots are
            advanced in place. Slots may have padding between them but each ``[H, V, K]``
            row must be dense.
        state_indices: Contiguous int32 slot indices shaped ``[B]``; nonpositive entries
            are padding that produce zero output and leave the pool untouched.
        output: Preallocated output written in place, one row of ``H * V`` per sequence.
        gate_kind: Gate layout; selects the transform input shapes above.
        gate_transform: Pointwise transform applied to the raw gate projection.
        key_heads: Number of q/k heads packed into ``packed_qkv``.
        lower_bound: Finite nonpositive bound used only by ``GateTransform.BOUNDED``.
        scale: Query scale folded into the in-kernel L2 normalization.
        has_initial_state: Optional per-sequence booleans; False marks freshly assigned
            slots whose bytes are garbage, so the step starts from the zero state and
            overwrites the slot.
    """
    read_only_inputs = (packed_qkv, raw_gate, raw_beta, A_log, dt_bias, state_indices)
    if has_initial_state is not None:
        read_only_inputs += (has_initial_state,)
    if any(torch._C._overlaps(output, tensor) for tensor in (*read_only_inputs, state_cache)):
        raise ValueError(f"out must not alias any {op_name} input")
    if any(torch._C._overlaps(state_cache, tensor) for tensor in read_only_inputs):
        raise ValueError(f"state_cache must not alias {op_name} read-only inputs")

    batch = packed_qkv.shape[0]
    heads, value_dim, key_dim = state_cache.shape[1:]
    assert key_heads > 0 and heads % key_heads == 0
    use_hopper_gdn_config = (
        get_device_properties(packed_qkv.device).major == 9
        and packed_qkv.dtype is torch.bfloat16
        and key_dim == value_dim == 128
        and gate_kind is GateKind.SCALAR
    )
    block_v, num_warps = _decode_launch_config(
        value_dim,
        batch * heads,
        use_hopper_gdn_config,
    )
    grid = (triton.cdiv(value_dim, block_v) * batch * heads,)
    _recurrent_delta_rule_decode_kernel[grid](
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        output,
        state_cache,
        state_indices,
        has_initial_state,
        lower_bound,
        scale=scale,
        state_batch_stride=state_cache.stride(0),
        qkv_token_stride=packed_qkv.stride(0),
        gate_token_stride=raw_gate.stride(0),
        beta_token_stride=raw_beta.stride(0),
        H=heads,
        HK=key_heads,
        K=key_dim,
        V=value_dim,
        BK=triton.next_power_of_2(key_dim),
        BV=block_v,
        USE_LOWER_BOUND=gate_transform is GateTransform.BOUNDED,
        SCALAR_GATE=gate_kind is GateKind.SCALAR,
        USE_HAS_INITIAL_STATE=has_initial_state is not None,
        num_warps=num_warps,
    )


__all__ = ["GateTransform", "launch_recurrent_delta_rule_decode"]
