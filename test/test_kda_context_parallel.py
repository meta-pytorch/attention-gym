"""Small CPU tests for context-parallel state routing."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from examples.kda_context_parallel import (
    ContextParallelKDAAttention,
    ContextParallelKDAFunction,
    PackedShard,
)


def _summary(transition: list[list[float]], bias: list[list[float]]) -> torch.Tensor:
    transition_tensor = torch.tensor(transition, dtype=torch.float32).unsqueeze(0)
    bias_tensor = torch.tensor(bias, dtype=torch.float32).unsqueeze(0)
    return torch.cat((bias_tensor, transition_tensor), dim=-2)


def _apply(state: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
    """Independent affine-map oracle for the routing tests."""
    return state @ summary[:, 2:, :] + summary[:, :2, :]


def test_short_conv_halo_spans_multiple_small_shards_and_resets_at_documents():
    global_cu_seqlens = (0, 1, 7, 8)
    shards = (
        PackedShard(cu_seqlens=(0, 1, 2), sequence_ids=(0, 1)),
        PackedShard(cu_seqlens=(0, 2), sequence_ids=(1,)),
        PackedShard(cu_seqlens=(0, 2), sequence_ids=(1,)),
        PackedShard(cu_seqlens=(0, 1, 2), sequence_ids=(1, 2)),
    )
    gathered_tails = torch.tensor(
        [
            [[0.0], [0.0], [1.0]],
            [[0.0], [2.0], [3.0]],
            [[0.0], [4.0], [5.0]],
            [[0.0], [6.0], [7.0]],
        ]
    )

    rank_three = ContextParallelKDAAttention.compose_conv_initial_states(
        gathered_tails,
        shard_tokens=2,
        rank=3,
        shards=shards,
        global_cu_seqlens=global_cu_seqlens,
    )

    torch.testing.assert_close(rank_three[0, :, 0], torch.tensor([3.0, 4.0, 5.0]))
    torch.testing.assert_close(rank_three[1], torch.zeros_like(rank_three[1]))

    reset_boundaries = (0, 2, 8)
    reset_shards = (
        PackedShard(cu_seqlens=(0, 2), sequence_ids=(0,)),
        PackedShard(cu_seqlens=(0, 2), sequence_ids=(1,)),
        PackedShard(cu_seqlens=(0, 2), sequence_ids=(1,)),
        PackedShard(cu_seqlens=(0, 2), sequence_ids=(1,)),
    )
    reset = ContextParallelKDAAttention.compose_conv_initial_states(
        gathered_tails,
        shard_tokens=2,
        rank=1,
        shards=reset_shards,
        global_cu_seqlens=reset_boundaries,
    )
    torch.testing.assert_close(reset, torch.zeros_like(reset))

    partial_boundaries = (0, 3, 8)
    partial_shards = (
        PackedShard(cu_seqlens=(0, 2), sequence_ids=(0,)),
        PackedShard(cu_seqlens=(0, 1, 2), sequence_ids=(0, 1)),
        PackedShard(cu_seqlens=(0, 2), sequence_ids=(1,)),
        PackedShard(cu_seqlens=(0, 2), sequence_ids=(1,)),
    )
    partial = ContextParallelKDAAttention.compose_conv_initial_states(
        gathered_tails,
        shard_tokens=2,
        rank=2,
        shards=partial_shards,
        global_cu_seqlens=partial_boundaries,
    )
    torch.testing.assert_close(partial[0, :, 0], torch.tensor([0.0, 0.0, 3.0]))


def test_short_conv_halo_backward_is_the_transpose_of_forward_routing():
    global_cu_seqlens = (0, 1, 7, 8)
    shards = (
        PackedShard(cu_seqlens=(0, 1, 2), sequence_ids=(0, 1)),
        PackedShard(cu_seqlens=(0, 2), sequence_ids=(1,)),
        PackedShard(cu_seqlens=(0, 2), sequence_ids=(1,)),
        PackedShard(cu_seqlens=(0, 1, 2), sequence_ids=(1, 2)),
    )
    gathered_tails = torch.zeros(4, 3, 1, requires_grad=True)
    rank_two = ContextParallelKDAAttention.compose_conv_initial_states(
        gathered_tails,
        shard_tokens=2,
        rank=2,
        shards=shards,
        global_cu_seqlens=global_cu_seqlens,
    )
    rank_three = ContextParallelKDAAttention.compose_conv_initial_states(
        gathered_tails,
        shard_tokens=2,
        rank=3,
        shards=shards,
        global_cu_seqlens=global_cu_seqlens,
    )
    loss = (rank_two[0, :, 0] * torch.tensor([10.0, 20.0, 30.0])).sum()
    loss += (rank_three[0, :, 0] * torch.tensor([40.0, 50.0, 60.0])).sum()

    (tail_gradients,) = torch.autograd.grad(loss, gathered_tails)

    torch.testing.assert_close(tail_gradients[1, :, 0], torch.tensor([0.0, 20.0, 70.0]))


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

    states = ContextParallelKDAFunction.compose_forward_initial_states(
        summaries, q, v, rank=2, shards=shards
    )

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

    gradients = ContextParallelKDAFunction.compose_reverse_final_states(
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
