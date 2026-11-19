"""Host side of :class:`~attn_gym.linear.StateCheckpoints`: validation, op arguments, kernel view.

Kept free of Triton so public APIs validate checkpoints without it; the kernels' store lives in
``_delta_rule/triton/state_checkpoints.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from attn_gym.linear._delta_rule.paged_state import validate_state_pool
from attn_gym.linear._delta_rule.validation import require_inference_only
from attn_gym.linear.types import StateCheckpoints


class CheckpointWrites(NamedTuple):
    """The kernel view of :class:`~attn_gym.linear.StateCheckpoints`.

    Triton specializes the integer fields only on equality to 1, divisibility by 16, and int32
    range, so every interval (a positive multiple of 64 below ``2**31``) and every slot stride
    of one pool layout share a compiled variant.
    """

    state_pool: torch.Tensor
    cu_checkpoint_offsets: torch.Tensor
    every_n_tokens: int
    destinations: torch.Tensor
    slot_stride: int


def checkpoint_writes(checkpoints: StateCheckpoints | None) -> CheckpointWrites | None:
    """Return the kernel argument for optional, already validated checkpoints."""
    if checkpoints is None:
        return None
    return CheckpointWrites(*checkpoints, checkpoints.state_pool.stride(0))


def validate_state_checkpoints(
    checkpoints: StateCheckpoints | None,
    q: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> None:
    """Check snapshot metadata against token-major operands without reading device values."""
    if checkpoints is None:
        return
    validate_state_checkpoints_metadata(
        checkpoints,
        num_sequences=q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1,
        state_shape=(v.shape[2], v.shape[-1], q.shape[-1]),
        device=q.device,
    )


def validate_state_checkpoints_metadata(
    checkpoints: StateCheckpoints,
    *,
    num_sequences: int,
    state_shape: tuple[int, int, int],
    device: torch.device,
) -> None:
    """Validate caller-owned snapshots, including commits that have no token operands."""
    if not isinstance(checkpoints, StateCheckpoints):
        raise TypeError("state_checkpoints must be a StateCheckpoints instance")
    state_pool, offsets, interval, destinations = checkpoints
    if type(interval) is not int:
        raise TypeError("state_checkpoints.every_n_tokens must be an int")
    if interval <= 0 or interval % 64:
        raise ValueError("state_checkpoints.every_n_tokens must be a positive multiple of 64")
    heads, value_dim, key_dim = state_shape
    validate_state_pool(
        state_pool,
        heads=heads,
        value_dim=value_dim,
        key_dim=key_dim,
        device=device,
        name="state_checkpoints.state_pool",
    )
    if (
        destinations.ndim != 1
        or destinations.dtype != torch.int32
        or not destinations.is_contiguous()
        or destinations.device != device
    ):
        raise ValueError(
            "state_checkpoints.destinations must be a contiguous 1-D int32 tensor "
            "on the input device"
        )
    if (
        offsets.shape != (num_sequences + 1,)
        or offsets.dtype != torch.int32
        or not offsets.is_contiguous()
        or offsets.device != device
    ):
        raise ValueError(
            "state_checkpoints.cu_checkpoint_offsets must be a contiguous int32 tensor of shape "
            "[N + 1] on the input device"
        )
    require_inference_only(
        (state_pool,),
        "state_checkpoints is inference-only; call under torch.no_grad() or torch.inference_mode()",
    )


def state_checkpoint_op_args(
    checkpoints: StateCheckpoints | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int, torch.Tensor | None]:
    """Flatten optional snapshots into the trailing arguments of a custom-op schema."""
    return (None, None, 0, None) if checkpoints is None else checkpoints


def state_checkpoints_from_op_args(
    state_pool: torch.Tensor | None,
    offsets: torch.Tensor | None,
    every_n_tokens: int,
    destinations: torch.Tensor | None,
) -> StateCheckpoints | None:
    """Rebuild the snapshots a custom-op schema received as flattened arguments."""
    if state_pool is None and offsets is None and destinations is None:
        return None
    if state_pool is None or offsets is None or destinations is None:
        raise ValueError(
            "state checkpoints require checkpoint_states, cu_checkpoint_offsets, "
            "and checkpoint_destinations together"
        )
    return StateCheckpoints(state_pool, offsets, every_n_tokens, destinations)


__all__ = [
    "CheckpointWrites",
    "checkpoint_writes",
    "state_checkpoint_op_args",
    "state_checkpoints_from_op_args",
    "validate_state_checkpoints",
    "validate_state_checkpoints_metadata",
]
