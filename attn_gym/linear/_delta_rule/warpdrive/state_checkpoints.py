"""Selected-prefix stores at the existing FP32 state handoffs of the paged prefills."""

import cutlass
from cutlass import cute

from attn_gym.linear._delta_rule.constants import DEFAULT_CHUNK_SIZE

from .kda_common import store_tmem_state


@cute.jit
def checkpoint_row(checkpoints, routes, batch, seqlen, head, row):
    """Resolve this work item's request once. A disabled target never matches a boundary."""
    result = None
    if cutlass.const_expr(checkpoints is not None):
        pool, positions, slots = checkpoints
        pos, slot = positions[batch], slots[batch]
        valid = (
            (routes[batch] > 0)
            & (slot > 0)
            & (pos > 0)
            & (pos <= seqlen)
            & (pos % DEFAULT_CHUNK_SIZE == 0)
        )
        result = (cutlass.select_(valid, pos, -1), pool[cutlass.max(slot, 0), head, row, None])
    return result


@cute.jit
def store_checkpoint(
    checkpoint,
    boundary,
    taddr,
    row_valid,
    first: cutlass.Constexpr[int],
    width: cutlass.Constexpr[int],
):
    """Read only columns this role owns, after its update wait and before it decays them again.

    The destination is independent of the running cache's dtype, alignment and slot stride.
    """
    if cutlass.const_expr(checkpoint is not None):
        position, dst = checkpoint
        if position == boundary:
            store_tmem_state(dst, taddr, row_valid, first, width, dst.element_type.width // 8)
