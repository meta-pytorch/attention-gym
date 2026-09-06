# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Types shared by public linear-attention operations."""

from enum import Enum
from typing import Literal, TypedDict


class BackendOptions(TypedDict, total=False):
    """Backend selection shared by optimized linear-attention operations."""

    backend: Literal["fused", "mega"]
    """Select the repo-local fused backend or the optional Mega backend."""


class KernelOptions(BackendOptions, total=False):
    """KDA backend controls and experimental scheduling options."""

    split_backward: bool
    """Allow KDA Mega to use its approximate split-backward schedule."""

    split_forward: bool
    """Allow KDA Mega to use its approximate forgetting-horizon split forward schedule."""


class Impl(str, Enum):
    """Select a fused or reference implementation without automatic fallback."""

    FUSED = "fused"
    REFERENCE = "reference"


def resolve_impl(impl: Impl | str) -> Impl:
    """Normalize an implementation selector and report the valid values."""
    try:
        return Impl(impl)
    except ValueError:
        valid = ", ".join(repr(member.value) for member in Impl)
        raise ValueError(f"unknown impl {impl!r}; expected one of {valid}") from None


class GateTransform(str, Enum):
    """Pointwise transform from a raw gate projection to a natural-log decay.

    Both kinds apply to per-head (``[B, T, H]``) and per-channel (``[B, T, H, D]``) gates;
    the kind is independent of the gate shape.
    """

    BOUNDED = "bounded"  # lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))
    SOFTPLUS = "softplus"  # -exp(A_log) * softplus(raw_gate + dt_bias)


def resolve_gate_transform(kind: GateTransform | str) -> GateTransform:
    """Normalize a gate transform selector and report the valid values."""
    try:
        return GateTransform(kind)
    except ValueError:
        valid = ", ".join(repr(member.value) for member in GateTransform)
        raise ValueError(
            f"unknown gate transform kind {kind!r}; expected one of {valid}"
        ) from None


__all__ = [
    "BackendOptions",
    "GateTransform",
    "Impl",
    "KernelOptions",
    "resolve_gate_transform",
    "resolve_impl",
]
