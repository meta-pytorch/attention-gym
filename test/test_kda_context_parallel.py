"""Small CPU tests for context-parallel ownership plans and state routing."""

from __future__ import annotations

import pytest
import torch

import attn_gym.linear.context_parallel as cpmod
from attn_gym.linear.context_parallel import (
    ContextParallelPlan,
    Fragment,
    compose_conv_histories,
    compose_entry_states,
    compose_exit_cotangents,
)

VALUE_DIM = 2
KEY_DIM = 2

# Sequences [0,1), [1,7), [7,8) over four ranks of two contiguous tokens each.
CONV_CU_SEQLENS = (0, 1, 7, 8)
CONV_RANGES = [[(0, 2)], [(2, 4)], [(4, 6)], [(6, 8)]]
CONV_PLANS = [
    ContextParallelPlan.from_token_ranges(CONV_CU_SEQLENS, CONV_RANGES, rank) for rank in range(4)
]


def summary(transition: list[list[float]], bias: list[list[float]]) -> torch.Tensor:
    """One-head ``[bias; transition]`` summary."""
    return torch.cat(
        (torch.tensor(bias, dtype=torch.float32), torch.tensor(transition, dtype=torch.float32))
    ).unsqueeze(0)


def apply(state: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    """Independent affine-map oracle for the routing tests."""
    return state @ packed[:, VALUE_DIM:, :] + packed[:, :VALUE_DIM, :]


UNUSED = summary([[7, 7], [7, 7]], [[7, 7], [7, 7]])


def with_unused_slot(first: torch.Tensor) -> torch.Tensor:
    """One rank's two gather slots where only the first is consumed."""
    return torch.stack((first, UNUSED))


@pytest.mark.parametrize(
    ("cu_seqlens", "ranges", "message"),
    [
        pytest.param((0, 4), [[(0, 2)], [(3, 4)]], "tile", id="gap"),
        pytest.param((0, 4), [[(0, 2)], [(1, 4)]], "tile", id="overlap"),
        pytest.param((0, 4), [[(1, 4)]], "tile", id="missing-prefix"),
        pytest.param((0, 4), [[(0, 3)]], "tile", id="missing-suffix"),
        pytest.param((0, 4), [[(0, 2)], [(2, 4), (0, 2)]], "tile", id="duplicate"),
        pytest.param((0, 4), [[(0, 4)], []], "at least one", id="empty-rank"),
        pytest.param((0, 4), [[(2, 2), (0, 4)]], "empty", id="empty-range"),
        pytest.param((0,), [[(0, 4)]], "at least one sequence", id="no-sequences"),
        pytest.param((1, 4), [[(0, 4)]], "start at zero", id="offset-stream"),
        pytest.param((0, 4, 2), [[(0, 4)]], "nondecreasing", id="decreasing-offsets"),
    ],
)
def test_plan_rejects_ranges_that_do_not_tile_the_stream(cu_seqlens, ranges, message):
    with pytest.raises(ValueError, match=message):
        ContextParallelPlan.from_token_ranges(cu_seqlens, ranges, rank=0)


def test_plan_rejects_rank_outside_table():
    with pytest.raises(ValueError, match="rank 2"):
        ContextParallelPlan.from_token_ranges((0, 4), [[(0, 2)], [(2, 4)]], rank=2)


def test_plan_cuts_ranges_at_sequences_and_orders_neighbors():
    # A = [0, 8), B = [8, 12); rank 0 owns A[0:2], then the block [6, 10) spanning A and B.
    ranges = [[(0, 2), (6, 10)], [(2, 6), (10, 12)]]

    plan = ContextParallelPlan.from_token_ranges((0, 8, 12), ranges, rank=0)
    assert plan.fragments == (Fragment(0, 0, 2), Fragment(0, 6, 8), Fragment(1, 8, 10))
    assert plan.table[1] == (Fragment(0, 2, 6), Fragment(1, 10, 12))
    assert plan.cu_seqlens == (0, 2, 4, 6)
    assert plan.slots == 3
    assert plan.predecessors == ((), ((0, 0), (1, 0)), ())
    assert plan.successors == (((0, 1), (1, 0)), (), ((1, 1),))
    assert plan.terminal == (1,)
    assert plan.global_token_ids("cpu").tolist() == [0, 1, 6, 7, 8, 9]

    other = ContextParallelPlan.from_token_ranges((0, 8, 12), ranges, rank=1)
    assert other.slots == 3
    assert other.predecessors == (((0, 0),), ((0, 2),))
    assert other.successors == (((0, 1),), ())
    assert other.terminal == (1,)


def test_plan_skips_empty_sequences_and_chains_adjacent_ranges_on_one_rank():
    # Sequence 1 is empty, so it owns no fragment anywhere while sequence 2 keeps its index.
    ranges = [[(0, 2), (2, 4)], [(4, 8)]]
    plan = ContextParallelPlan.from_token_ranges((0, 4, 4, 8), ranges, rank=0)
    assert plan.fragments == (Fragment(0, 0, 2), Fragment(0, 2, 4))
    assert plan.predecessors == ((), ((0, 0),))
    assert plan.terminal == (1,)
    other = ContextParallelPlan.from_token_ranges((0, 4, 4, 8), ranges, rank=1)
    assert other.fragments == (Fragment(2, 4, 8),)
    assert other.predecessors == ((),)


def test_entry_states_share_prefixes_across_fragments_of_one_sequence(monkeypatch):
    """A rank holding many pieces of one sequence folds each predecessor once, not per piece."""
    pieces = [(i * 4, (i + 1) * 4) for i in range(8)]
    plan = ContextParallelPlan.from_token_ranges((0, 32), [pieces[:4], pieces[4:]], rank=1)
    gathered = torch.randn(2, 4, 1, 3, 2)
    merges = []
    merge_state = cpmod.merge_state
    monkeypatch.setattr(cpmod, "merge_state", lambda s, a: merges.append(1) or merge_state(s, a))
    entry = compose_entry_states(gathered, plan)
    # Fragments 4..7 have 4..7 predecessors each; the shared prefix means 7 merges, not 22.
    assert len(merges) == 7
    expected = torch.zeros(1, 1, 2)
    for owner, slot in plan.predecessors[-1]:
        expected = merge_state(expected, gathered[owner, slot])
    torch.testing.assert_close(entry[-1], expected)


def test_entry_and_exit_folds_follow_fragment_geometry_for_every_fragment():
    """Independently derive each fragment's chain from token order and compare both folds."""
    cu_seqlens = (0, 5, 5, 12, 16)
    ranges = [[(0, 3), (13, 16)], [(3, 7), (11, 13)], [(7, 11)]]
    plans = [ContextParallelPlan.from_token_ranges(cu_seqlens, ranges, rank) for rank in range(3)]
    generator = torch.Generator().manual_seed(3)
    gathered = torch.randn(3, max(p.slots for p in plans), 1, 4, 2, generator=generator)
    everything = [(r, s, f) for r, row in enumerate(plans[0].table) for s, f in enumerate(row)]
    zero = torch.zeros(1, VALUE_DIM, KEY_DIM)
    for plan in plans:
        entries = compose_entry_states(gathered, plan)
        exits = compose_exit_cotangents(gathered, None, plan)
        for index, fragment in enumerate(plan.fragments):
            siblings = [(r, s, f) for r, s, f in everything if f.sequence == fragment.sequence]
            earlier = sorted((f.start, r, s) for r, s, f in siblings if f.start < fragment.start)
            later = sorted((f.start, r, s) for r, s, f in siblings if f.start > fragment.start)
            entry = zero
            for _, r, s in earlier:
                entry = apply(entry, gathered[r, s])
            exit_cotangent = zero
            for _, r, s in reversed(later):
                exit_cotangent = apply(exit_cotangent, gathered[r, s])
            torch.testing.assert_close(entries[index], entry)
            torch.testing.assert_close(exits[index], exit_cotangent)
            assert (index in plan.terminal) == (not later)


def test_plan_handles_leading_and_trailing_empty_sequences():
    plan = ContextParallelPlan.from_token_ranges((0, 0, 4, 4), [[(0, 4)]], rank=0)
    assert plan.fragments == (Fragment(1, 0, 4),)
    assert plan.terminal == (0,)
    with pytest.raises(ValueError, match="empty"):
        ContextParallelPlan.from_token_ranges((0, 0), [[(0, 0)]], rank=0)


def test_entry_states_fold_predecessors_from_zero_in_token_order():
    ranges = [[(0, 2), (9, 10)], [(2, 4)], [(4, 9)]]
    plans = [ContextParallelPlan.from_token_ranges((0, 9, 10), ranges, rank) for rank in range(3)]
    gathered = torch.stack(
        (
            with_unused_slot(summary([[1, 2], [0, 1]], [[1, 0], [0, 2]])),
            with_unused_slot(summary([[2, 0], [3, 1]], [[0, 1], [2, 0]])),
            with_unused_slot(summary([[5, 5], [5, 5]], [[5, 5], [5, 5]])),
        )
    )

    zero = torch.zeros(1, VALUE_DIM, KEY_DIM)
    torch.testing.assert_close(
        compose_entry_states(gathered, plans[2])[0],
        apply(apply(zero, gathered[0, 0]), gathered[1, 0]),
    )
    torch.testing.assert_close(
        compose_entry_states(gathered, plans[0]), torch.zeros(2, 1, VALUE_DIM, KEY_DIM)
    )


def test_exit_cotangents_fold_successors_in_reverse_and_add_the_direct_term():
    # Sequences [0,1), [1,5), [5,8); rank 0 owns the first token of each of the first two.
    ranges = [[(0, 2)], [(2, 4)], [(4, 6)], [(6, 8)]]
    plan = ContextParallelPlan.from_token_ranges((0, 1, 5, 8), ranges, rank=0)
    gathered = torch.stack(
        (
            with_unused_slot(summary([[1, 0], [0, 1]], [[9, 9], [9, 9]])),
            with_unused_slot(summary([[2, 1], [0, 1]], [[1, 0], [0, 1]])),
            with_unused_slot(summary([[1, 0], [2, 3]], [[0, 2], [1, 0]])),
            with_unused_slot(summary([[4, 0], [0, 4]], [[7, 7], [7, 7]])),
        )
    )
    d_final_state = torch.arange(8, dtype=torch.float32).reshape(2, 1, VALUE_DIM, KEY_DIM)

    zero = torch.zeros(1, VALUE_DIM, KEY_DIM)
    # Fragment 1 (sequence 1) continues on ranks 1 then 2; the fold visits the farthest first.
    suffix = apply(apply(zero, gathered[2, 0]), gathered[1, 0])
    expected = d_final_state.clone()
    expected[1] += suffix
    torch.testing.assert_close(compose_exit_cotangents(gathered, d_final_state, plan), expected)
    torch.testing.assert_close(
        compose_exit_cotangents(gathered, None, plan), torch.stack((zero, suffix))
    )


def test_short_conv_histories_span_short_predecessors_and_reset_at_sequences():
    tails = torch.tensor(
        [
            [[[0.0], [0.0], [1.0]], [[0.0], [0.0], [1.5]]],
            [[[0.0], [2.0], [3.0]], [[0.0], [0.0], [0.0]]],
            [[[0.0], [4.0], [5.0]], [[0.0], [0.0], [0.0]]],
            [[[0.0], [0.0], [6.0]], [[0.0], [0.0], [7.0]]],
        ]
    )
    rank_three = compose_conv_histories(tails, CONV_PLANS[3])
    torch.testing.assert_close(rank_three[0, :, 0], torch.tensor([3.0, 4.0, 5.0]))
    torch.testing.assert_close(rank_three[1], torch.zeros_like(rank_three[1]))
    torch.testing.assert_close(
        compose_conv_histories(tails, CONV_PLANS[1])[0, :, 0], torch.tensor([0.0, 0.0, 1.5])
    )


def test_short_conv_history_backward_is_the_transpose_of_forward_routing():
    tails = torch.zeros(4, 2, 3, 1, requires_grad=True)
    rank_two = compose_conv_histories(tails, CONV_PLANS[2])
    rank_three = compose_conv_histories(tails, CONV_PLANS[3])
    loss = (rank_two[0, :, 0] * torch.tensor([10.0, 20.0, 30.0])).sum()
    loss += (rank_three[0, :, 0] * torch.tensor([40.0, 50.0, 60.0])).sum()

    (tail_gradients,) = torch.autograd.grad(loss, tails)

    # Rank 1's last token feeds rank 2's history[2] (weight 30) and rank 3's history[0] (40).
    torch.testing.assert_close(tail_gradients[1, 0, :, 0], torch.tensor([0.0, 20.0, 70.0]))
    # Rank 0 starts both of its sequences yet must stay connected to the gathered tails so its
    # reduce-scatter participates in the collective's backward.
    assert compose_conv_histories(tails, CONV_PLANS[0]).requires_grad
