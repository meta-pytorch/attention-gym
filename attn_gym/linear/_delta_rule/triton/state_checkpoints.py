"""Device-side access to per-sequence state checkpoints (``CheckpointWrites``)."""

from __future__ import annotations

import triton
import triton.language as tl

from attn_gym.linear._delta_rule.constants import DEFAULT_CHUNK_SIZE

_CHUNK_SIZE = tl.constexpr(DEFAULT_CHUNK_SIZE)


@triton.jit
def load_checkpoint_target(checkpoints, sequence, length):
    """Return the requested boundary and slot; an invalid request never matches a scan step."""
    position = tl.load(checkpoints.checkpoint_positions + sequence)
    slot = tl.load(checkpoints.checkpoint_slots + sequence)
    aligned = position % _CHUNK_SIZE == 0
    tl.device_assert(
        (position > 0) & (position <= length) & aligned,
        "invalid state checkpoint position",
        mask=slot > 0,
    )
    position = tl.where(aligned & (slot > 0), position, 0)
    return position, slot


@triton.jit
def store_checkpoint(checkpoints, slot, slot_offset, value, mask):
    """Store the selected elements of a published state; callers skip non-positive slots."""
    offset = slot.to(tl.int64) * checkpoints.slot_stride + slot_offset
    tl.store(checkpoints.state_pool + offset, value, mask=mask)


__all__ = ["load_checkpoint_target", "store_checkpoint"]
