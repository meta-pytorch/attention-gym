"""Strict compilation regression for dynamic packed KDA sequence counts."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear import chunk_kda
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

TOKENS = 128


def offsets(sequence_count: int) -> torch.Tensor:
    """Split the token capacity into near-equal packed sequence boundaries."""
    step = TOKENS // sequence_count
    lengths = [step] * (sequence_count - 1) + [TOKENS - step * (sequence_count - 1)]
    return cumulative_sequence_offsets(lengths)


def test_compiled_chunk_kda_accepts_varying_packed_sequence_count():
    """Vary sequence counts through the composed core and shared scheduler."""
    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("the fused KDA core requires CUDA capability 8.0 or newer")
    inputs = make_kda_test_inputs(TOKENS)
    compiled = torch.compile(chunk_kda, fullgraph=True, dynamic=True)
    for sequence_count in (2, 4):
        cu_seqlens = offsets(sequence_count)
        expected, expected_state = chunk_kda(*inputs, cu_seqlens=cu_seqlens)
        actual, actual_state = compiled(*inputs, cu_seqlens=cu_seqlens)
        assert expected_state is actual_state is None
        torch.testing.assert_close(actual, expected)
