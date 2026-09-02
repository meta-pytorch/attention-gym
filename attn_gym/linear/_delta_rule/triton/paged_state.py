# SPDX-License-Identifier: BSD-3-Clause

"""Triton lowering of the shared paged-state routing contract."""

import triton
import triton.language as tl


@triton.jit
def resolve_paged_state(
    sequence,
    state_indices,
    has_initial_state,
    USE_STATE_INDICES: tl.constexpr,
    USE_HAS_INITIAL_STATE: tl.constexpr,
):
    """Return the effective slot and active/load predicates for one sequence."""
    slot = sequence
    active = True
    load_initial = True
    if USE_STATE_INDICES:
        route = tl.load(state_indices + sequence).to(tl.int64)
        active = route > 0
        slot = tl.where(active, route, 0)
        load_initial = active
        if USE_HAS_INITIAL_STATE:
            has_state = tl.load(has_initial_state + sequence) != 0
            load_initial = active & has_state
    return slot, active, load_initial


__all__ = ["resolve_paged_state"]
