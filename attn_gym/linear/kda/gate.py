# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi-style per-token natural-log gate transform."""

from __future__ import annotations

from numbers import Real
from sys import float_info

import torch

from attn_gym.linear.kda.ops import _bound_gate_bwd_op, _bound_gate_fwd_op
from attn_gym.linear.types import Impl, resolve_impl

_SUPPORTED_FUSED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _validate_bound_gate_inputs(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
) -> tuple[int, int, float]:
    """Validate the backend-neutral pointwise transform contract."""
    if raw_gate.ndim != 4:
        raise ValueError(f"raw_gate must have shape [B, T, H, D], got {tuple(raw_gate.shape)}")
    _, _, heads, head_dim = raw_gate.shape
    if not raw_gate.dtype.is_floating_point:
        raise TypeError("raw_gate must use a floating-point dtype")
    if A_log.shape != (heads,) or A_log.dtype != torch.float32:
        raise ValueError(
            f"A_log must be float32 with shape {(heads,)}, "
            f"got {tuple(A_log.shape)} and {A_log.dtype}"
        )
    if dt_bias.shape != (heads, head_dim) or dt_bias.dtype != torch.float32:
        raise ValueError(
            f"dt_bias must be float32 with shape {(heads, head_dim)}, "
            f"got {tuple(dt_bias.shape)} and {dt_bias.dtype}"
        )
    if not all(tensor.device == raw_gate.device for tensor in (A_log, dt_bias)):
        raise ValueError("bound_gate inputs must be on the same device")
    if isinstance(lower_bound, bool) or not isinstance(lower_bound, Real):
        raise TypeError(f"lower_bound must be a real scalar, got {type(lower_bound).__name__}")
    lower_bound = float(lower_bound)
    if not -float_info.max <= lower_bound <= 0.0:
        raise ValueError(f"lower_bound must be finite and nonpositive, got {lower_bound}")
    return heads, head_dim, lower_bound


def _bound_gate_reference(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
    heads: int,
) -> torch.Tensor:
    """Evaluate the transform as ordinary FP32 PyTorch operations."""
    gate_input = raw_gate.float() + dt_bias
    amplitude = A_log.exp().view(1, 1, heads, 1)
    return lower_bound * torch.sigmoid(amplitude * gate_input)


class _BoundGate(torch.autograd.Function):
    """Attach the private CuTeDSL forward and first-order backward operators."""

    @staticmethod
    def forward(ctx, raw_gate, A_log, dt_bias, lower_bound, fastmath):
        gate = _bound_gate_fwd_op(raw_gate, A_log, dt_bias, lower_bound, fastmath)
        ctx.save_for_backward(raw_gate, A_log, dt_bias)
        ctx.lower_bound = lower_bound
        ctx.fastmath = fastmath
        return gate

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_gate):
        raw_gate, A_log, dt_bias = ctx.saved_tensors
        d_raw_gate, dA_log_partial, d_dt_bias = _bound_gate_bwd_op(
            raw_gate,
            A_log,
            dt_bias,
            d_gate,
            ctx.lower_bound,
            ctx.fastmath,
        )
        return d_raw_gate, dA_log_partial.sum((0, 1)), d_dt_bias, None, None


def bound_gate(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float = -5.0,
    fastmath: bool = False,
    impl: Impl | str = Impl.FUSED,
) -> torch.Tensor:
    """Map projection outputs to bounded per-token natural-log decays.

    The transform is ``lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))``.
    Arithmetic and output use FP32; a fused raw-gate gradient matches the input dtype.

    Args:
        raw_gate: Floating-point projection output shaped ``[B, T, H, D]``.
        A_log: FP32 per-head log scale shaped ``[H]``.
        dt_bias: FP32 per-channel bias shaped ``[H, D]``.
        lower_bound: Finite nonpositive gate floor.
        fastmath: Use approximate fused exponentials; rejected by the reference path.
        impl: ``"reference"`` uses ordinary PyTorch. ``"fused"`` uses private CuTeDSL
            kernels and requires CUDA capability 9.0 or newer, ``D=128``, and FP16, BF16,
            or FP32 logits.
    """
    heads, head_dim, lower_bound = _validate_bound_gate_inputs(
        raw_gate,
        A_log,
        dt_bias,
        lower_bound,
    )
    selected_impl = resolve_impl(impl)
    if selected_impl is Impl.REFERENCE:
        if fastmath:
            raise ValueError("fastmath applies only to impl='fused'")
        return _bound_gate_reference(raw_gate, A_log, dt_bias, lower_bound, heads)

    batch, tokens, _, _ = raw_gate.shape
    if head_dim != 128:
        raise ValueError("bound_gate(impl='fused') requires raw_gate with D=128")
    if not torch.compiler.is_compiling() and min(batch, tokens, heads) < 1:
        raise ValueError("raw_gate batch, token, and head dimensions must be nonzero")
    if raw_gate.dtype not in _SUPPORTED_FUSED_DTYPES or not raw_gate.is_cuda:
        raise ValueError("bound_gate(impl='fused') requires CUDA FP16, BF16, or FP32 raw_gate")
    if not isinstance(fastmath, bool):
        raise TypeError(f"fastmath must be bool, got {type(fastmath).__name__}")
    return _BoundGate.apply(raw_gate, A_log, dt_bias, lower_bound, fastmath)
