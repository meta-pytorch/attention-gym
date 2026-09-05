"""Reject malformed Mega metadata before launching device-side validation on any CUDA GPU."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from attn_gym.linear.gdn.impl import mega_ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA tensors")


@pytest.mark.parametrize(
    "op",
    [
        mega_ops.chunk_gdn_mega_packed_fwd_op,
        mega_ops.chunk_gdn_mega_packed_fwd_with_initial_state_op,
        mega_ops.chunk_gdn_mega_packed_fwd_with_state_op,
    ],
    ids=["no-state", "initial-only", "with-state"],
)
@pytest.mark.parametrize("invalid", ["rank", "empty", "short", "dtype", "cpu"])
def test_raw_forward_rejects_metadata_before_device_reads(op, invalid: str, monkeypatch):
    """Raw callers must not reach the offset kernel with a malformed metadata tensor."""
    from attn_gym.linear.kda import chunk_scheduler

    scheduler = Mock(side_effect=AssertionError("malformed metadata reached the scheduler"))
    backend = Mock(side_effect=AssertionError("malformed metadata reached the Mega backend"))
    monkeypatch.setattr(chunk_scheduler, "_prepare_ragged_chunk_offsets", scheduler)
    monkeypatch.setattr(mega_ops, "_forward_backend", backend)
    q = torch.zeros(1, 8, 1, 128, dtype=torch.bfloat16, device="cuda")
    gate = torch.zeros(1, 8, 1, device="cuda")
    match invalid:
        case "rank":
            offsets = torch.empty(2, 0, dtype=torch.int32, device="cuda")
        case "empty":
            offsets = torch.empty(0, dtype=torch.int32, device="cuda")
        case "short":
            offsets = torch.zeros(1, dtype=torch.int32, device="cuda")
        case "dtype":
            offsets = torch.tensor([0, 8], dtype=torch.int64, device="cuda")
        case "cpu":
            offsets = torch.tensor([0, 8], dtype=torch.int32)
    args = (q, q, q, gate, gate)
    if op is not mega_ops.chunk_gdn_mega_packed_fwd_op:
        args = (*args, torch.zeros(1, 1, 128, 128, device="cuda"))
    with pytest.raises(TypeError, match="int32 vector on q.device"):
        op(*args, offsets, 128**-0.5)
    scheduler.assert_not_called()
    backend.assert_not_called()
