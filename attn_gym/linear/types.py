# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Types shared by public linear-attention operations."""

from enum import Enum
from typing import Literal, NamedTuple, TypedDict

import torch

from attn_gym.types import Impl, resolve_impl


class ReplayState(NamedTuple):
    """Persistent token cache used by replay-backed paged KDA.

    The Q, K, and V caches use BF16, gate and beta use FP32, and count uses int32.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    gate: torch.Tensor
    beta: torch.Tensor
    count: torch.Tensor


class BackendOptions(TypedDict, total=False):
    """Backend selection shared by optimized linear-attention operations."""

    backend: Literal["fused", "cudnn"]
    """Select the repo-local fused backend or the optional cuDNN backend."""


class SplitOptions(BackendOptions, total=False):
    """Approximate cuDNN forgetting-horizon split schedules for chunk KDA and chunk GDN.

    Both default to ``False``, require ``backend="cudnn"``, and reject calls with recurrent
    state. They help only when ``B * H`` does not fill the GPU and every head forgets quickly.
    """

    split_backward: bool
    """Split the backward recurrence."""

    split_forward: bool
    """Split the forward recurrence."""


class GDNKernelOptions(SplitOptions, total=False):
    """Chunk GDN backend controls."""

    save_chunk_states: bool
    """Keep the forward's chunk states for the backward instead of recomputing them. Gradients
    are bitwise identical; the cost is about ``(T / 64 + N) * H * K * V + T * H * V`` more
    ``q.dtype`` elements held until the backward (384 MiB at T=16384, N=1, H=32, K=V=128).
    Fused backend only."""


class KernelOptions(SplitOptions, total=False):
    """KDA backend controls and experimental scheduling options."""

    schedule: Literal["auto", "static", "persistent"]
    """Set ``"persistent"`` for CUDA graphs whose replays can carry far fewer tokens than the
    captured capacity: the default static grid pays one CTA per capacity chunk even when empty,
    while the persistent grid strides over active work only and wins below ~1/4 of capacity
    active. Outputs are bitwise identical. Fused backend only; Hopper or newer for packed
    inputs."""


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
    "GDNKernelOptions",
    "GateTransform",
    "Impl",
    "KernelOptions",
    "ReplayState",
    "SplitOptions",
    "resolve_gate_transform",
    "resolve_impl",
]
