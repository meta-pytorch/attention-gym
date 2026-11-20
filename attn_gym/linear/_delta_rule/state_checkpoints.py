"""Host side of :class:`~attn_gym.linear.StateCheckpoints`: validation and custom-op arguments."""

from __future__ import annotations

import torch

from attn_gym.linear._delta_rule.validation import require_inference_only
from attn_gym.linear.types import StateCheckpoints


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
    states, offsets, interval = checkpoints
    if type(interval) is not int:
        raise TypeError("state_checkpoints.every_n_tokens must be an int")
    if interval <= 0 or interval % 64:
        raise ValueError("state_checkpoints.every_n_tokens must be a positive multiple of 64")
    if states.ndim != 4 or states.shape[1:] != state_shape:
        raise ValueError("state_checkpoints.states must have shape [capacity, H, V, K]")
    if states.dtype != torch.float32:
        raise TypeError("state_checkpoints.states must use float32")
    if not states.is_contiguous():
        raise ValueError("state_checkpoints.states must be contiguous")
    if offsets.shape != (num_sequences + 1,):
        raise ValueError("state_checkpoints.cu_checkpoint_offsets must have shape [N + 1]")
    if offsets.dtype != torch.int32:
        raise TypeError("state_checkpoints.cu_checkpoint_offsets must use int32")
    if not offsets.is_contiguous():
        raise ValueError("state_checkpoints.cu_checkpoint_offsets must be contiguous")
    if states.device != device or offsets.device != device:
        raise ValueError("state_checkpoints tensors must be on the input device")
    require_inference_only(
        (states,),
        "state_checkpoints is inference-only; call under torch.no_grad() or torch.inference_mode()",
    )


def state_checkpoint_op_args(
    checkpoints: StateCheckpoints | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """Flatten optional snapshots into the trailing arguments of a custom-op schema."""
    return (None, None, 0) if checkpoints is None else checkpoints


def state_checkpoints_from_op_args(
    states: torch.Tensor | None, offsets: torch.Tensor | None, every_n_tokens: int
) -> StateCheckpoints | None:
    """Rebuild the snapshots a custom-op schema received as flattened arguments.

    Triton kernels take the result as one tuple argument. Triton specializes its interval only
    on equality to 1, divisibility by 16, and int32 range, so every positive multiple of 64
    below ``2**31`` shares one compiled variant.
    """
    if states is None:
        if offsets is not None:
            raise ValueError("cu_checkpoint_offsets requires checkpoint_states")
        return None
    if offsets is None:
        raise ValueError("checkpoint_states requires cu_checkpoint_offsets")
    return StateCheckpoints(states, offsets, every_n_tokens)


__all__ = [
    "state_checkpoint_op_args",
    "state_checkpoints_from_op_args",
    "validate_state_checkpoints",
    "validate_state_checkpoints_metadata",
]
