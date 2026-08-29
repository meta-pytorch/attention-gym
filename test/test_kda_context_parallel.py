"""Small CPU tests for context-parallel affine-summary routing."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from examples.kda_context_parallel import (
    PackedShard,
    _compose_forward_initial_states,
    _compose_reverse_final_states,
)


def _summary(transition: list[list[float]], bias: list[list[float]]) -> torch.Tensor:
    transition_tensor = torch.tensor(transition, dtype=torch.float32).unsqueeze(0)
    bias_tensor = torch.tensor(bias, dtype=torch.float32).unsqueeze(0)
    return torch.cat((bias_tensor, transition_tensor), dim=-2)


def _apply(state: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
    """Independent affine-map oracle for the routing tests."""
    return state @ summary[:, 2:, :] + summary[:, :2, :]


def test_forward_prefix_propagates_one_sequence_across_three_shards():
    summaries = torch.stack(
        (
            _summary([[1, 2], [0, 1]], [[1, 0], [0, 2]]),
            _summary([[2, 0], [3, 1]], [[0, 1], [2, 0]]),
            _summary([[1, 0], [0, 1]], [[9, 9], [9, 9]]),
        )
    )
    shards = (
        PackedShard(cu_seqlens=(0, 2), sequence_ids=(0,)),
        PackedShard(cu_seqlens=(0, 2), sequence_ids=(0,)),
        PackedShard(cu_seqlens=(0, 1, 2), sequence_ids=(0, 1)),
    )
    q = torch.empty(1, 2, 1, 2)
    v = torch.empty_like(q)

    states = _compose_forward_initial_states(summaries, q, v, rank=2, shards=shards)

    expected_first = _apply(_apply(torch.zeros(1, 2, 2), summaries[0]), summaries[1])
    torch.testing.assert_close(states[0], expected_first)
    torch.testing.assert_close(states[1], torch.zeros_like(states[1]))


def test_reverse_suffix_filters_sequences_and_adds_final_state_gradient():
    summaries = torch.stack(
        (
            _summary([[1, 0], [0, 1]], [[9, 9], [9, 9]]),
            _summary([[2, 1], [0, 1]], [[1, 0], [0, 1]]),
            _summary([[1, 0], [2, 3]], [[0, 2], [1, 0]]),
            _summary([[4, 0], [0, 4]], [[7, 7], [7, 7]]),
        )
    )
    shards = (
        PackedShard(cu_seqlens=(0, 1, 2), sequence_ids=(0, 1)),
        PackedShard(cu_seqlens=(0, 2), sequence_ids=(1,)),
        PackedShard(cu_seqlens=(0, 1, 2), sequence_ids=(1, 2)),
        PackedShard(cu_seqlens=(0, 2), sequence_ids=(2,)),
    )
    initial_state = torch.zeros(2, 1, 2, 2)
    d_final_state = torch.arange(8, dtype=torch.float32).reshape_as(initial_state)

    gradients = _compose_reverse_final_states(
        summaries,
        d_final_state=d_final_state,
        initial_state=initial_state,
        rank=0,
        shards=shards,
    )

    suffix = _apply(_apply(torch.zeros(1, 2, 2), summaries[2]), summaries[1])
    expected = d_final_state.clone()
    expected[-1] += suffix
    torch.testing.assert_close(gradients, expected)
