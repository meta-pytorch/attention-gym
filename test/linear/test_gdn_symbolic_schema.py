"""Packed workspace capacity must remain symbolic at the registered-op boundary."""

import pytest
import torch

from attn_gym.linear.gdn import ops


@pytest.mark.parametrize(
    "operator",
    [ops.chunk_fwd_packed_op, ops.chunk_fwd_packed_with_state_op, ops.chunk_fwd_packed_paged_op],
    ids=["functional", "with-state", "paged"],
)
def test_packed_capacity_schema_is_symbolic(operator: torch._ops.OpOverload):
    assert "SymInt capacity" in str(operator._schema)
