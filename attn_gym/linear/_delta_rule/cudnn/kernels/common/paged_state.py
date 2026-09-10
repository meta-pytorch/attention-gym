# SPDX-License-Identifier: BSD-3-Clause

"""Device-side routing for mutable paged recurrent state."""

import cutlass
from cutlass import cute


@cute.jit
def resolve_paged_state(batch_idx, state_indices, has_initial_state):
    """Resolve one sequence to its state slot and load/store predicates.

    Without routing metadata, the sequence index selects ordinary dense state.
    With routing metadata, non-positive indices are null routes, and an optional
    byte mask distinguishes resumed slots from fresh slots.
    """
    state_slot = batch_idx
    active = cutlass.Boolean(True)
    load_initial = cutlass.Boolean(True)
    clear_empty = cutlass.Boolean(False)
    if cutlass.const_expr(state_indices is not None):
        route = state_indices[batch_idx]
        active = route > 0
        state_slot = route if active else cutlass.Int32(0)
        load_initial = active
        if cutlass.const_expr(has_initial_state is not None):
            has_state = has_initial_state[batch_idx] != 0
            load_initial = active and has_state
            clear_empty = active and has_initial_state[batch_idx] == 0
    return state_slot, active, load_initial, clear_empty


__all__ = ["resolve_paged_state"]
