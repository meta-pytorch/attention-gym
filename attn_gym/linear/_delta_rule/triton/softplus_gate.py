# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Portable fused softplus gate transform ``-exp(A_log) * softplus(raw_gate + dt_bias)``.

One pointwise kernel serves per-head gates ``[B, T, H]`` and per-channel gates ``[B, T, H, D]``:
the launcher views a per-head gate as ``D=1`` and the kernel tiles ``rows = B*T`` by
``channels = H*D``, recovering ``(b, t, h, d)`` through compile-time strides so any input layout
works without copies. Strides are constexpr as elsewhere in the repo: the kernel is
instruction-bound, and runtime strides measured 8-20% slower; the cost is one JIT per distinct
stride set (i.e. per new ``T`` for a contiguous gate). The backward is not purely pointwise:
``d_dt_bias`` and ``d_A_log`` reduce over rows, so each program writes one FP32 partial per
channel and the launcher finishes with ``torch.sum``, keeping the reduction deterministic
without atomics.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from attn_gym._backends.triton.utils import LOG2_E, ptr_offset

LN_2 = tl.constexpr(math.log(2.0))
# Matches ``torch.nn.functional.softplus``: above this input the identity is exact in FP32.
SOFTPLUS_THRESHOLD = tl.constexpr(20.0)
# Elements per program with the channel block capped so narrow per-head gates (H=32) and wide
# per-channel gates (H*128) both fill four warps; a sweep of 512-16K elements and 1-8 warps on
# GB200 found no better point.
ELEMENTS_PER_PROGRAM = 2048
MAX_BLOCK_CHANNELS = 256

# NOTE [Fastmath Softplus]
# The accurate path (libdevice ``exp`` + ``log1p``, agreeing with ``F.softplus`` within FP32
# rounding) issues ~60 instructions per element and caps this kernel at about half of HBM
# bandwidth. The fastmath path uses the MUFU ``ex2.approx``/``lg2.approx`` intrinsics through the
# overflow-free form ``max(s, 0) + log(1 + exp(-|s|))``, which needs no threshold branch and
# reaches the memory roofline. Its absolute error matches FP32 eager (~6e-7 on N(0, 4) inputs);
# relative error is large only where softplus underflows toward zero, which the downstream
# ``exp(gate)`` cannot see. ``tl.log2`` is not a shortcut here: Triton lowers it to a
# polynomial, not ``lg2.approx``.


@triton.jit
def _softplus(s, FASTMATH: tl.constexpr):
    if FASTMATH:  # See NOTE [Fastmath Softplus]
        return tl.maximum(s, 0.0) + libdevice.fast_log2f(1.0 + tl.exp2(-tl.abs(s) * LOG2_E)) * LN_2
    return tl.where(s > SOFTPLUS_THRESHOLD, s, libdevice.log1p(libdevice.exp(s)))


@triton.jit
def _sigmoid(s, FASTMATH: tl.constexpr):
    if FASTMATH:
        return libdevice.fast_dividef(1.0, 1.0 + tl.exp2(-s * LOG2_E))
    return 1.0 / (1.0 + libdevice.exp(-s))


@triton.jit
def _load_gate_operands(
    raw_gate,
    A_log,
    dt_bias,
    o_row,
    o_c,
    m_row,
    m_c,
    T,
    RAW_STRIDES: tl.constexpr,
    A_LOG_STRIDE: tl.constexpr,
    DT_STRIDES: tl.constexpr,
    D: tl.constexpr,
):
    """Return ``(s, amplitude)`` with ``s = raw + dt_bias`` in FP32 and ``amplitude = exp(A_log)``."""
    o_h = o_c // D
    o_d = o_c % D
    raw = tl.load(
        raw_gate
        + ptr_offset(
            ((o_row // T)[:, None], (o_row % T)[:, None], o_h[None, :], o_d[None, :]),
            RAW_STRIDES,
        ),
        mask=m_row[:, None] & m_c[None, :],
        other=0.0,
    ).to(tl.float32)
    bias = tl.load(dt_bias + ptr_offset((o_h, o_d), DT_STRIDES), mask=m_c, other=0.0)
    amplitude = tl.exp(tl.load(A_log + o_h * A_LOG_STRIDE, mask=m_c, other=0.0))
    return raw + bias[None, :], amplitude


@triton.jit
def softplus_gate_fwd_kernel(
    raw_gate,
    A_log,
    dt_bias,
    gate,
    ROWS,
    T,
    RAW_STRIDES: tl.constexpr,
    A_LOG_STRIDE: tl.constexpr,
    DT_STRIDES: tl.constexpr,
    GATE_STRIDES: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    FASTMATH: tl.constexpr,
):
    o_row = tl.program_id(0).to(tl.int64) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    o_c = (tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)).to(tl.int64)
    m_row = o_row < ROWS
    m_c = o_c < C
    s, amplitude = _load_gate_operands(
        raw_gate,
        A_log,
        dt_bias,
        o_row,
        o_c,
        m_row,
        m_c,
        T,
        RAW_STRIDES,
        A_LOG_STRIDE,
        DT_STRIDES,
        D,
    )
    value = -amplitude[None, :] * _softplus(s, FASTMATH)
    tl.store(
        gate
        + ptr_offset(
            ((o_row // T)[:, None], (o_row % T)[:, None], (o_c // D)[None, :], (o_c % D)[None, :]),
            GATE_STRIDES,
        ),
        value,
        mask=m_row[:, None] & m_c[None, :],
    )


@triton.jit
def softplus_gate_bwd_kernel(
    raw_gate,
    A_log,
    dt_bias,
    d_gate,
    d_raw_gate,
    d_A_log_partial,
    d_dt_bias_partial,
    ROWS,
    T,
    RAW_STRIDES: tl.constexpr,
    A_LOG_STRIDE: tl.constexpr,
    DT_STRIDES: tl.constexpr,
    D_GATE_STRIDES: tl.constexpr,
    D_RAW_STRIDES: tl.constexpr,
    C: tl.constexpr,
    D: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    FASTMATH: tl.constexpr,
):
    i_row_block = tl.program_id(0).to(tl.int64)
    o_row = i_row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    o_c = (tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)).to(tl.int64)
    m_row = o_row < ROWS
    m_c = o_c < C
    mask = m_row[:, None] & m_c[None, :]
    s, amplitude = _load_gate_operands(
        raw_gate,
        A_log,
        dt_bias,
        o_row,
        o_c,
        m_row,
        m_c,
        T,
        RAW_STRIDES,
        A_LOG_STRIDE,
        DT_STRIDES,
        D,
    )
    element_offsets = (
        (o_row // T)[:, None],
        (o_row % T)[:, None],
        (o_c // D)[None, :],
        (o_c % D)[None, :],
    )
    grad = tl.load(d_gate + ptr_offset(element_offsets, D_GATE_STRIDES), mask=mask, other=0.0)
    grad = grad.to(tl.float32)

    # gate = -amplitude * softplus(s): d/ds = -amplitude * sigmoid(s); d/dA_log = gate.
    d_s = -amplitude[None, :] * _sigmoid(s, FASTMATH) * grad
    d_amplitude_log = -amplitude[None, :] * _softplus(s, FASTMATH) * grad
    tl.store(
        d_raw_gate + ptr_offset(element_offsets, D_RAW_STRIDES),
        d_s.to(d_raw_gate.dtype.element_ty),
        mask=mask,
    )
    # Masked rows loaded grad == 0, so they contribute exactly zero to the block partials.
    tl.store(d_dt_bias_partial + i_row_block * C + o_c, tl.sum(d_s, 0), mask=m_c)
    tl.store(d_A_log_partial + i_row_block * C + o_c, tl.sum(d_amplitude_log, 0), mask=m_c)


def _launch_config(channels: int) -> tuple[int, int]:
    """Pick ``(BLOCK_ROWS, BLOCK_C)`` so one program covers ``ELEMENTS_PER_PROGRAM`` elements."""
    block_channels = min(triton.next_power_of_2(channels), MAX_BLOCK_CHANNELS)
    return max(1, ELEMENTS_PER_PROGRAM // block_channels), block_channels


def _softplus_gate_fwd_cuda(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    fastmath: bool,
) -> torch.Tensor:
    """Launch the forward over any input layout and return a compact FP32 gate."""
    shape = raw_gate.shape
    if raw_gate.ndim == 3:  # Per-head gate: view as per-channel with D=1.
        raw_gate, dt_bias = raw_gate.unsqueeze(-1), dt_bias.unsqueeze(-1)
    batch, tokens, heads, head_dim = raw_gate.shape
    rows, channels = batch * tokens, heads * head_dim
    gate = torch.empty(raw_gate.shape, device=raw_gate.device, dtype=torch.float32)
    block_rows, block_channels = _launch_config(channels)
    softplus_gate_fwd_kernel[
        (triton.cdiv(rows, block_rows), triton.cdiv(channels, block_channels))
    ](
        raw_gate,
        A_log,
        dt_bias,
        gate,
        rows,
        tokens,
        RAW_STRIDES=raw_gate.stride(),
        A_LOG_STRIDE=A_log.stride(0),
        DT_STRIDES=dt_bias.stride(),
        GATE_STRIDES=gate.stride(),
        C=channels,
        D=head_dim,
        BLOCK_ROWS=block_rows,
        BLOCK_C=block_channels,
        FASTMATH=fastmath,
        num_warps=4,
    )
    return gate.view(shape)


def _softplus_gate_bwd_cuda(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_gate: torch.Tensor,
    fastmath: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the backward; returns ``(d_raw_gate, d_A_log, d_dt_bias)`` with reduced params."""
    shape = raw_gate.shape
    if raw_gate.ndim == 3:  # Per-head gate: view as per-channel with D=1.
        raw_gate, dt_bias, d_gate = (t.unsqueeze(-1) for t in (raw_gate, dt_bias, d_gate))
    batch, tokens, heads, head_dim = raw_gate.shape
    rows, channels = batch * tokens, heads * head_dim
    d_raw_gate = torch.empty_like(raw_gate, memory_format=torch.contiguous_format)
    block_rows, block_channels = _launch_config(channels)
    row_blocks = triton.cdiv(rows, block_rows)
    d_A_log_partial = torch.empty(
        row_blocks, channels, device=raw_gate.device, dtype=torch.float32
    )
    d_dt_bias_partial = torch.empty_like(d_A_log_partial)
    softplus_gate_bwd_kernel[(row_blocks, triton.cdiv(channels, block_channels))](
        raw_gate,
        A_log,
        dt_bias,
        d_gate,
        d_raw_gate,
        d_A_log_partial,
        d_dt_bias_partial,
        rows,
        tokens,
        RAW_STRIDES=raw_gate.stride(),
        A_LOG_STRIDE=A_log.stride(0),
        DT_STRIDES=dt_bias.stride(),
        D_GATE_STRIDES=d_gate.stride(),
        D_RAW_STRIDES=d_raw_gate.stride(),
        C=channels,
        D=head_dim,
        BLOCK_ROWS=block_rows,
        BLOCK_C=block_channels,
        FASTMATH=fastmath,
        num_warps=4,
    )
    d_A_log = d_A_log_partial.view(row_blocks, heads, head_dim).sum((0, 2))
    return d_raw_gate.view(shape), d_A_log, d_dt_bias_partial.sum(0).view(shape[2:])
