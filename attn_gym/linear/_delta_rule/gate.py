# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone gate transforms shared by the KDA and GDN training paths.

The decode kernels evaluate the same two transforms in-kernel
(``attn_gym.linear._delta_rule.decode``); this module is the materialized, differentiable
version for training and prefill, where the gate is computed once and reused by the
chunked recurrence. Transform kind and gate shape are independent axes: either kind applies
to per-head ``[B, T, H]`` or per-channel ``[B, T, H, D]`` gates.
"""

from __future__ import annotations

import importlib
from functools import cache
from numbers import Real
from sys import float_info

import torch
import torch.nn.functional as F

from attn_gym._backends.cute.utils import get_device_properties
from attn_gym.linear.types import GateTransform, Impl, resolve_gate_transform, resolve_impl

_SUPPORTED_FUSED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# One operator pair covers both kinds and both fused backends; ``kind`` is an argument, so the
# backend is chosen once per call inside the CUDA implementation on real tensors.
torch.library.define(
    "attn_gym::_gate_transform_fwd",
    "(Tensor raw_gate, Tensor A_log, Tensor dt_bias, str kind, float lower_bound, bool fastmath)"
    " -> Tensor",
)
torch.library.define(
    "attn_gym::_gate_transform_bwd",
    "(Tensor raw_gate, Tensor A_log, Tensor dt_bias, Tensor d_gate, str kind,"
    " float lower_bound, bool fastmath) -> (Tensor, Tensor, Tensor)",
)


def _cute_backend(module: str):
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ImportError(
            "gate_transform(kind='bounded', impl='fused') requires the optional CuTeDSL backend: "
            "pip install attn-gym[linear]"
        ) from error


def _triton_backend():
    try:
        return importlib.import_module("attn_gym.linear._delta_rule.triton.softplus_gate")
    except ImportError as error:
        raise ImportError(
            "gate_transform(kind='softplus', impl='fused') requires CUDA with Triton support"
        ) from error


@cache
def _cute_gate_backend_available() -> bool:
    """Whether the CuTeDSL gate kernels import; probed once, only on an eligible call."""
    try:
        _cute_backend("attn_gym.linear.kda.fwd.cute.gate_fwd")
        _cute_backend("attn_gym.linear.kda.bwd.cute.gate_bwd")
    except ImportError:
        return False
    return True


# NOTE [Softplus Backend Selection]
# The Triton softplus kernel is portable (any CUDA device, both gate shapes, any strides) but
# instruction-bound at roughly half of HBM bandwidth. The CuTeDSL bound-gate kernels reach the
# bf16->fp32 cast roofline and gained a softplus specialization, but they keep the bound-gate
# contract: per-channel gates with D=128, CUDA capability 9.0+, and the optional CuTeDSL
# dependency. Selection runs inside the registered operator on real tensors, so it is invisible
# to torch.compile tracing; tests pin the policy through ``_softplus_uses_cute``.
def _softplus_uses_cute(raw_gate: torch.Tensor) -> bool:
    """Route per-channel D=128 gates on sm90+ to the CuTeDSL kernels when they are installed."""
    return (
        raw_gate.ndim == 4
        and raw_gate.shape[3] == 128
        and get_device_properties(raw_gate.device).major >= 9
        and _cute_gate_backend_available()
    )


def _gate_transform_fwd_cuda(raw_gate, A_log, dt_bias, kind, lower_bound, fastmath):
    transform = GateTransform(kind)
    if transform is GateTransform.SOFTPLUS and not _softplus_uses_cute(raw_gate):
        return _triton_backend()._softplus_gate_fwd_cuda(raw_gate, A_log, dt_bias, fastmath)
    return _cute_backend("attn_gym.linear.kda.fwd.cute.gate_fwd")._gate_transform_fwd_cuda(
        raw_gate, A_log, dt_bias, lower_bound, fastmath, transform
    )


def _gate_transform_bwd_cuda(raw_gate, A_log, dt_bias, d_gate, kind, lower_bound, fastmath):
    transform = GateTransform(kind)
    if transform is GateTransform.SOFTPLUS and not _softplus_uses_cute(raw_gate):
        return _triton_backend()._softplus_gate_bwd_cuda(
            raw_gate, A_log, dt_bias, d_gate, fastmath
        )
    return _cute_backend("attn_gym.linear.kda.bwd.cute.gate_bwd")._gate_transform_bwd_cuda(
        raw_gate, A_log, dt_bias, d_gate, lower_bound, fastmath, transform
    )


torch.library.impl("attn_gym::_gate_transform_fwd", "CUDA", _gate_transform_fwd_cuda)
torch.library.impl("attn_gym::_gate_transform_bwd", "CUDA", _gate_transform_bwd_cuda)


@torch.library.register_fake("attn_gym::_gate_transform_fwd")
def _gate_transform_fwd_fake(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    kind: str,
    lower_bound: float,
    fastmath: bool,
) -> torch.Tensor:
    """Describe the compact FP32 gate output."""
    del A_log, dt_bias, kind, lower_bound, fastmath
    return torch.empty(raw_gate.shape, device=raw_gate.device, dtype=torch.float32)


@torch.library.register_fake("attn_gym::_gate_transform_bwd")
def _gate_transform_bwd_fake(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    d_gate: torch.Tensor,
    kind: str,
    lower_bound: float,
    fastmath: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``d_raw_gate`` in the input dtype; parameter gradients already reduced and contiguous."""
    del d_gate, kind, lower_bound, fastmath
    return (
        torch.empty_like(raw_gate, memory_format=torch.contiguous_format),
        A_log.new_empty(A_log.shape),
        dt_bias.new_empty(dt_bias.shape),
    )


_gate_transform_fwd_op = torch.ops.attn_gym._gate_transform_fwd.default
_gate_transform_bwd_op = torch.ops.attn_gym._gate_transform_bwd.default


class _FusedGate(torch.autograd.Function):
    """Attach the private fused forward and first-order backward operators."""

    @staticmethod
    def forward(ctx, raw_gate, A_log, dt_bias, kind, lower_bound, fastmath):
        ctx.save_for_backward(raw_gate, A_log, dt_bias)
        ctx.kind, ctx.lower_bound, ctx.fastmath = kind, lower_bound, fastmath
        return _gate_transform_fwd_op(raw_gate, A_log, dt_bias, kind, lower_bound, fastmath)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_gate):
        raw_gate, A_log, dt_bias = ctx.saved_tensors
        d_raw_gate, d_A_log, d_dt_bias = _gate_transform_bwd_op(
            raw_gate, A_log, dt_bias, d_gate, ctx.kind, ctx.lower_bound, ctx.fastmath
        )
        return d_raw_gate, d_A_log, d_dt_bias, None, None, None


def _validate_lower_bound(kind: GateTransform, lower_bound: float | None) -> float:
    """Require a finite nonpositive floor for ``bounded`` and forbid one otherwise."""
    if kind is GateTransform.SOFTPLUS:
        if lower_bound is not None:
            raise ValueError("lower_bound applies only to kind='bounded'")
        # Softplus has no floor; a fixed 0.0 keeps the fused compile-cache key stable.
        return 0.0
    if isinstance(lower_bound, bool) or not isinstance(lower_bound, Real):
        raise TypeError(
            f"kind='bounded' requires a real lower_bound, got {type(lower_bound).__name__}"
        )
    lower_bound = float(lower_bound)
    if not -float_info.max <= lower_bound <= 0.0:
        raise ValueError(f"lower_bound must be finite and nonpositive, got {lower_bound}")
    return lower_bound


def _validate_gate_inputs(raw_gate: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor):
    """Validate the backend-neutral contract shared by both gate shapes."""
    if raw_gate.ndim not in (3, 4):
        raise ValueError(
            f"raw_gate must have shape [B, T, H] or [B, T, H, D], got {tuple(raw_gate.shape)}"
        )
    if not raw_gate.dtype.is_floating_point:
        raise TypeError("raw_gate must use a floating-point dtype")
    heads = raw_gate.shape[2]
    if A_log.shape != (heads,) or A_log.dtype != torch.float32:
        raise ValueError(
            f"A_log must be float32 with shape {(heads,)}, "
            f"got {tuple(A_log.shape)} and {A_log.dtype}"
        )
    bias_shape = tuple(raw_gate.shape[2:])
    if dt_bias.shape != bias_shape or dt_bias.dtype != torch.float32:
        raise ValueError(
            f"dt_bias must be float32 with shape {bias_shape}, "
            f"got {tuple(dt_bias.shape)} and {dt_bias.dtype}"
        )
    if not all(tensor.device == raw_gate.device for tensor in (A_log, dt_bias)):
        raise ValueError("gate_transform inputs must be on the same device")


def _gate_transform_reference(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    kind: GateTransform,
    lower_bound: float,
) -> torch.Tensor:
    """Evaluate the transform as ordinary FP32 PyTorch operations."""
    gate_input = raw_gate.float() + dt_bias
    amplitude = A_log.exp().view(1, 1, -1, *([1] * (raw_gate.ndim - 3)))
    if kind is GateTransform.BOUNDED:
        return lower_bound * torch.sigmoid(amplitude * gate_input)
    return -amplitude * F.softplus(gate_input)


def gate_transform(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    kind: GateTransform | str,
    lower_bound: float | None = None,
    fastmath: bool = False,
    impl: Impl | str = Impl.FUSED,
) -> torch.Tensor:
    """Map raw gate projections to per-token natural-log decays.

    ``kind="bounded"`` computes ``lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))``
    (Kimi/KDA convention, range ``(lower_bound, 0)``); ``kind="softplus"`` computes
    ``-exp(A_log) * softplus(raw_gate + dt_bias)`` (Mamba2/GDN convention, unbounded below).
    Arithmetic and output use FP32; a fused raw-gate gradient matches the input dtype. The
    decode kernels evaluate the same transforms in-kernel from the raw projections.

    Args:
        raw_gate: Floating-point projection output shaped ``[B, T, H]`` (per-head gate) or
            ``[B, T, H, D]`` (per-channel gate).
        A_log: FP32 per-head log scale shaped ``[H]``.
        dt_bias: FP32 bias shaped ``[H]`` or ``[H, D]`` to match ``raw_gate``.
        kind: Transform kind; required so a default cannot silently select the wrong
            recurrence convention.
        lower_bound: Finite nonpositive floor, required by ``"bounded"`` and rejected by
            ``"softplus"``.
        fastmath: Use approximate fused exponentials; rejected by the reference path.
        impl: ``"reference"`` uses ordinary PyTorch. ``"fused"`` uses private CuTeDSL kernels
            for per-channel ``D=128`` gates on CUDA capability 9.0+ (both kinds) and a
            portable Triton kernel for every other ``"softplus"`` configuration. Fused paths
            require FP16, BF16, or FP32 ``raw_gate``; per-head ``"bounded"`` gates are
            reference-only.

    Raises:
        ValueError: On shape, ``lower_bound``, or unsupported fused configurations.
    """
    kind = resolve_gate_transform(kind)
    _validate_gate_inputs(raw_gate, A_log, dt_bias)
    lower_bound = _validate_lower_bound(kind, lower_bound)
    if resolve_impl(impl) is Impl.REFERENCE:
        if fastmath:
            raise ValueError("fastmath applies only to impl='fused'")
        return _gate_transform_reference(raw_gate, A_log, dt_bias, kind, lower_bound)

    if not isinstance(fastmath, bool):
        raise TypeError(f"fastmath must be bool, got {type(fastmath).__name__}")
    if raw_gate.dtype not in _SUPPORTED_FUSED_DTYPES or not raw_gate.is_cuda:
        raise ValueError("gate_transform(impl='fused') requires CUDA FP16, BF16, or FP32 raw_gate")
    if not torch.compiler.is_compiling() and min(raw_gate.shape) < 1:
        raise ValueError("gate_transform(impl='fused') requires nonzero raw_gate dimensions")
    if kind is GateTransform.BOUNDED and (raw_gate.ndim != 4 or raw_gate.shape[3] != 128):
        raise ValueError(
            "gate_transform(kind='bounded', impl='fused') requires raw_gate [B, T, H, 128]; "
            "use impl='reference' for other shapes"
        )
    return _FusedGate.apply(raw_gate, A_log, dt_bias, kind.value, lower_bound, fastmath)


__all__ = ["GateTransform", "gate_transform"]
