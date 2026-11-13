# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Types shared by public linear-attention operations."""

from enum import Enum
from typing import Literal, NamedTuple

import torch

from attn_gym.linear._delta_rule.constants import DEFAULT_CHUNK_SIZE
from attn_gym.types import TUNING_OPTION_SPEC, Impl, KernelOptionSpec, TuningOptions, resolve_impl


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

    @staticmethod
    def slot_layout(
        num_heads: int, head_dim: int, *, value_dim: int | None = None
    ) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
        """Return per-slot (shape, dtype) pairs in q, k, v, gate, beta, count order.

        The slot dimension is excluded. Value width defaults to the key width.
        Count has one element per slot; paged KDA also accepts a flat count vector.
        """
        value_dim = head_dim if value_dim is None else value_dim
        if num_heads < 1 or head_dim < 1 or value_dim < 1:
            raise ValueError("replay slot head count and dimensions must be positive")
        token_heads = (DEFAULT_CHUNK_SIZE, num_heads)
        key = ((*token_heads, head_dim), torch.bfloat16)
        return (
            key,
            key,
            ((*token_heads, value_dim), torch.bfloat16),
            (key[0], torch.float32),
            (token_heads, torch.float32),
            ((1,), torch.int32),
        )


class BackendOptions(TuningOptions, total=False):
    """Backend selection and tuning shared by optimized linear-attention operations."""

    backend: Literal["fused", "warpdrive"]
    """Select the repo-local fused backend or the optional warpdrive backend."""


class SplitOptions(BackendOptions, total=False):
    """Approximate warpdrive forgetting-horizon split schedules for chunk KDA and chunk GDN.

    Both default to ``False``, require ``backend="warpdrive"``, and reject calls with recurrent
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


class RecurrentKernelOptions(TuningOptions, total=False):
    """Options shared by the fused ``recurrent_gdn`` and ``recurrent_kda`` scans."""

    batch_invariant: bool
    """Keep output and final-state bits identical across batches on the same GPU model and
    software stack. Defaults to ``False`` for speed; ``"autotune": False`` alone is not enough.
    Token-by-token equivalence requires FP32 state; BF16 state rounds between calls.
    Autotuning and prefetching remain available. The fused ``recurrent_gdn_decode`` and
    ``recurrent_kda_decode`` ops do not support this option.
    See ``docs/determinism.md`` for the full contract and performance costs."""


# Accepted values and defaults of the option dicts above; see ``KernelOptionSpec``.
SPLIT_OPTION_SPEC: KernelOptionSpec = {
    **TUNING_OPTION_SPEC,
    "backend": (("fused", "warpdrive"), "fused"),
    "split_backward": (bool, False),
    "split_forward": (bool, False),
}
KERNEL_OPTION_SPEC: KernelOptionSpec = {
    **SPLIT_OPTION_SPEC,
    "schedule": (("auto", "static", "persistent"), "auto"),
}
RECURRENT_OPTION_SPEC: KernelOptionSpec = {**TUNING_OPTION_SPEC, "batch_invariant": (bool, False)}


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
    "TuningOptions",
    "resolve_gate_transform",
    "resolve_impl",
]
