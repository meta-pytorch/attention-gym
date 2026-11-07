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


class KernelOptions(SplitOptions, total=False):
    """KDA backend controls and experimental scheduling options."""

    schedule: Literal["auto", "static", "persistent"]
    """Set ``"persistent"`` for CUDA graphs whose replays can carry far fewer tokens than the
    captured capacity: the default static grid pays one CTA per capacity chunk even when empty,
    while the persistent grid strides over active work only and wins below ~1/4 of capacity
    active. Outputs are bitwise identical. Fused backend only; Hopper or newer for packed
    inputs."""


class RecurrentKernelOptions(TypedDict, total=False):
    """Options shared by the fused ``recurrent_gdn`` and ``recurrent_kda`` scans."""

    batch_invariant: bool
    """Give a sequence the same output and final-state bits in any batch: alone, among empty
    slots, or decoded one token per call with an FP32 state pool (a BF16 pool rounds the state
    between calls). Holds on one GPU model and software stack. Without it, the launch config
    follows the batch shape and changes the bits, and ``autotune=False`` does not prevent that.
    The scan uses one formula, sums in a fixed order and turns off FMA fusion, so the config no
    longer matters. Autotuning and prefetching still run. Costs up to ~20% GPU time on long
    prefills and ~9% on decode (GB300). The fused decode ops (``recurrent_gdn_decode``,
    ``recurrent_kda_decode``) do not support it. Defaults to ``False``."""


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
    "RecurrentKernelOptions",
    "ReplayState",
    "SplitOptions",
    "resolve_gate_transform",
    "resolve_impl",
]
