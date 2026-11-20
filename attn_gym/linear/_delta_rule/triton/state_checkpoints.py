"""Device-side stores for state-checkpoint rows (``CheckpointWrites``)."""

from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def checkpoint_slot(checkpoints, row, row_mask):
    """Return the int64 pool slot of checkpoint ``row``; non-positive slots skip the row.

    ``row_mask`` marks valid rows, or is ``None`` when every row is valid.
    """
    if row_mask is None:
        slot = tl.load(checkpoints.block_table + row)
    else:
        slot = tl.load(checkpoints.block_table + row, mask=row_mask, other=0)
    return slot.to(tl.int64)


@triton.jit
def store_checkpoint(checkpoints, slot, slot_offset, value, mask):
    """Store ``value`` at ``slot_offset`` within pool ``slot`` unless the slot is non-positive.

    ``mask`` selects the stored elements, or is ``None`` to store all of them. The store rounds
    the FP32 value to the pool dtype.
    """
    mask = slot > 0 if mask is None else mask & (slot > 0)
    tl.store(
        checkpoints.state_pool + slot * checkpoints.slot_stride + slot_offset, value, mask=mask
    )


__all__ = ["checkpoint_slot", "store_checkpoint"]
