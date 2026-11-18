# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Types shared by public linear-attention operations."""

from enum import Enum
from typing import Literal, NamedTuple, TypedDict, get_args

import torch

from attn_gym.linear._delta_rule.constants import DEFAULT_CHUNK_SIZE
from attn_gym.types import TUNING_OPTION_SPEC, Impl, KernelOptionSpec, TuningOptions, resolve_impl


class StateCheckpoints(NamedTuple):
    """Publish one selected prefix per sequence into a caller-owned recurrent-state pool.

    For sequence i, ``checkpoint_positions[i]`` counts new tokens in this call and
    ``checkpoint_slots[i]`` names the destination. Both are contiguous int32 vectors of
    shape ``[N]`` on the input device. Without replay, only positive multiples of 64
    within the sequence are written; other positions do not round down. Non-positive slots
    and inactive sequences do not write. An invalid position for a selected slot is a caller
    error: debug kernels can assert, and an untouched slot must not be published.

    With KDA replay, the stored state is the last 64-token boundary reached by the requested
    position, counting the slot's previously buffered tokens. No state is published before
    the first completed chunk or past the sequence. To resume mid-chunk, the caller must also
    preserve the trailing replay inputs and count; an exact boundary needs a zero count.

    ``state_pool`` is an FP32 or BF16 ``[slots, H, V, K]`` tensor, contiguous within
    disjoint slots. It may be the running state cache, but no other input. Positive destination
    slots must be in bounds, distinct among writers and, if sharing the cache, not routed by
    this call. Neither request vector may overlap storage the call modifies (including replay
    counts). These are caller guarantees, not host-side checks of device values.

    Runs on the fused backend; the default ``backend="auto"`` selects it for checkpointed
    chunk calls, and an explicit ``"warpdrive"`` raises.
    """

    state_pool: torch.Tensor
    checkpoint_positions: torch.Tensor
    checkpoint_slots: torch.Tensor


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

    backend: Literal["auto", "fused", "warpdrive", "cudnn"]
    """Choose the optimized chunk backend. With CuTeDSL installed, ``"auto"`` prefers warpdrive on
    SM100 and SM103, and on SM107 with CuTeDSL 4.8+. Paged checkpoint writes use Triton.
    Set ``"warpdrive"`` or ``"fused"`` (Triton) to require that backend; ``"cudnn"``,
    warpdrive's former name, is an alias of ``"warpdrive"``.
    See "Chunk backend selection" in ``docs/linear.md``."""


class SplitOptions(BackendOptions, total=False):
    """Approximate warpdrive forgetting-horizon split schedules for chunk KDA and chunk GDN.

    Both default to ``False``, require an explicit ``backend="warpdrive"``, and reject calls with
    recurrent state. They help only when ``B * H`` does not fill the GPU and every head forgets
    quickly.
    """

    split_backward: bool
    """Split the backward recurrence."""

    split_forward: bool
    """Split the forward recurrence."""


IntraChunkDtypes = Literal["tf32+bf16", "tf32", "bf16", "tf32+fp16"]


class KernelOptions(SplitOptions, total=False):
    """KDA backend controls and experimental scheduling options."""

    schedule: Literal["auto", "static", "persistent"]
    """Set ``"persistent"`` for CUDA graphs whose replays can carry far fewer tokens than the
    captured capacity: the default static grid pays one CTA per capacity chunk even when empty,
    while the persistent grid strides over active work only and wins below ~1/4 of capacity
    active. Outputs are bitwise identical. Hopper or newer for packed inputs. The warpdrive
    backend accepts and ignores every value: its persistent grids follow the SM count and
    always stride over the active work only."""

    intra_chunk_dtypes: IntraChunkDtypes
    """Operand precision inside each warpdrive KDA chunk, in forward and backward. The first
    dtype is for products of decayed queries and keys; the second is for the triangular solve
    and its contribution to the output. One dtype sets both. State products still use the
    Q/K/V kernel dtype, and all matrix products accumulate in FP32.

    BF16 kernels accept ``"tf32+bf16"`` (default), ``"tf32"`` or ``"bf16"`` (legacy bits).
    FP16 kernels accept only ``"tf32+fp16"``, their default. Replay casts Q/K/V to BF16 before
    choosing. Triton and the reference implementation validate the option but do not use it.
    See "Operand precision" in ``docs/linear.md``."""


class _StageOptions(TypedDict, total=False):
    batch_invariant: bool
    """Keep each sequence's context-parallel summary bits (``state_summaries`` and
    ``state_grad_summaries``) independent of the rest of the packed stream: the other sequences,
    their count and lengths. Defaults to ``False``, where a long sequence's summaries split into
    pieces by a plan of the stream's shape. ``True`` scans every sequence uncut in both
    directions. Only the warpdrive backend's summaries split, so like the split schedules it
    requires an explicit ``backend="warpdrive"``; the handles' local ``run`` keeps its own
    shape-dependent plan."""


class StageKernelOptions(KernelOptions, _StageOptions, total=False):
    """``chunk_kda_prepare`` and ``chunk_kda_prepare_backward`` options: ``chunk_kda``'s and
    ``batch_invariant``."""


class GDNStageKernelOptions(SplitOptions, _StageOptions, total=False):
    """``chunk_gdn_prepare`` and ``chunk_gdn_prepare_backward`` options: ``chunk_gdn``'s and
    ``batch_invariant``."""


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
    "backend": (("auto", "fused", "warpdrive", "cudnn"), "auto"),
    "split_backward": (bool, False),
    "split_forward": (bool, False),
}
KERNEL_OPTION_SPEC: KernelOptionSpec = {
    **SPLIT_OPTION_SPEC,
    "schedule": (("auto", "static", "persistent"), "auto"),
    # ``None``: the default by input dtype (``resolve_intra_chunk_dtypes``, ``kda/validation.py``).
    "intra_chunk_dtypes": (get_args(IntraChunkDtypes), None),
}
RECURRENT_OPTION_SPEC: KernelOptionSpec = {**TUNING_OPTION_SPEC, "batch_invariant": (bool, False)}
STAGE_OPTION_SPEC: KernelOptionSpec = {**KERNEL_OPTION_SPEC, "batch_invariant": (bool, False)}
GDN_STAGE_OPTION_SPEC: KernelOptionSpec = {**SPLIT_OPTION_SPEC, "batch_invariant": (bool, False)}


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
    "GDNStageKernelOptions",
    "GateTransform",
    "Impl",
    "KernelOptions",
    "RecurrentKernelOptions",
    "ReplayState",
    "SplitOptions",
    "StageKernelOptions",
    "StateCheckpoints",
    "TuningOptions",
    "resolve_gate_transform",
    "resolve_impl",
]
