"""Device-side stores for state-checkpoint rows (``CheckpointWrites``)."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def checkpoint_slot(checkpoints, row, row_mask):
    """Return where checkpoint ``row`` is stored: the tape row, or slot ``destinations[row]``.

    ``row_mask`` marks valid rows, or is ``None`` when every row is valid. Pool slots are int64
    and non-positive for skipped rows.
    """
    slot = row
    if checkpoints.destinations is not None:
        if row_mask is None:
            slot = tl.load(checkpoints.destinations + row).to(tl.int64)
        else:
            slot = tl.load(checkpoints.destinations + row, mask=row_mask, other=0).to(tl.int64)
    return slot


@triton.jit
def store_checkpoint(checkpoints, slot, slot_offset, value, mask):
    """Store ``value`` at ``slot_offset`` within ``slot``, skipping non-positive pool slots.

    ``mask`` selects the stored elements, or is ``None`` to store all of them. The store rounds
    the FP32 value to the destination dtype.
    """
    if checkpoints.destinations is not None:
        mask = slot > 0 if mask is None else mask & (slot > 0)
    tl.store(checkpoints.states + slot * checkpoints.slot_stride + slot_offset, value, mask=mask)


__all__ = ["checkpoint_slot", "store_checkpoint"]
