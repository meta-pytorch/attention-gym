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
    """Per-sequence checkpoint requests plus the pool's physical slot stride."""

    state_pool: torch.Tensor
    checkpoint_positions: torch.Tensor
    checkpoint_slots: torch.Tensor
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
    """Check checkpoint metadata against token-major operands without reading device values."""
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
    """Validate caller-owned checkpoints, including commits that have no token operands."""
    if not isinstance(checkpoints, StateCheckpoints):
        raise TypeError("state_checkpoints must be a StateCheckpoints instance")
    state_pool, positions, slots = checkpoints
    heads, value_dim, key_dim = state_shape
    validate_state_pool(
        state_pool,
        heads=heads,
        value_dim=value_dim,
        key_dim=key_dim,
        device=device,
        name="state_checkpoints.state_pool",
        any_unit_dim_stride=True,
    )
    for name, tensor in (("checkpoint_positions", positions), ("checkpoint_slots", slots)):
        if (
            tensor.shape != (num_sequences,)
            or tensor.dtype != torch.int32
            or not tensor.is_contiguous()
            or tensor.device != device
        ):
            raise ValueError(
                f"state_checkpoints.{name} must be a contiguous int32 tensor of shape [N] "
                "on the input device"
            )
    require_inference_only(
        (state_pool,),
        "state_checkpoints is inference-only; call under torch.no_grad() or torch.inference_mode()",
    )


def state_checkpoint_op_args(
    checkpoints: StateCheckpoints | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Flatten optional checkpoints into the trailing arguments of a custom-op schema."""
    return (None, None, None) if checkpoints is None else checkpoints


def state_checkpoints_from_op_args(
    state_pool: torch.Tensor | None,
    checkpoint_positions: torch.Tensor | None,
    checkpoint_slots: torch.Tensor | None,
) -> StateCheckpoints | None:
    """Rebuild the checkpoints a custom-op schema received as flattened arguments."""
    if state_pool is None and checkpoint_positions is None and checkpoint_slots is None:
        return None
    if state_pool is None or checkpoint_positions is None or checkpoint_slots is None:
        raise ValueError(
            "state checkpoints require checkpoint_states, checkpoint_positions, "
            "and checkpoint_slots together"
        )
    return StateCheckpoints(state_pool, checkpoint_positions, checkpoint_slots)


__all__ = [
    "CheckpointWrites",
    "checkpoint_writes",
    "state_checkpoint_op_args",
    "state_checkpoints_from_op_args",
    "validate_state_checkpoints",
    "validate_state_checkpoints_metadata",
]
