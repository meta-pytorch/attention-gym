"""Small CPU tests for context-parallel ownership plans and state routing."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.context_parallel import (
    ContextParallelPlan,
    Subsequence,
    compose_conv_histories,
    compose_entry_states,
    compose_exit_cotangents,
)

VALUE_DIM = 2
KEY_DIM = 2

# Sequences [0,1), [1,7), [7,8) over four ranks of two contiguous tokens each, with a width-4
# short convolution (three tokens of history).
CONV_CU_SEQLENS = (0, 1, 7, 8)
CONV_FRAGMENTS = [[(0, 2)], [(2, 4)], [(4, 6)], [(6, 8)]]
CONV_ROUTINGS = [
    ContextParallelPlan.from_fragments(CONV_CU_SEQLENS, CONV_FRAGMENTS, cp_rank).routing(
        "cpu", conv_history=3
    )
    for cp_rank in range(4)
]

# Three ranks, five fragments, over stream layouts that cut them differently.
FRAGMENTS = [[(0, 3), (13, 16)], [(3, 7), (11, 13)], [(7, 11)]]
FRAGMENT_TABLES = [
    pytest.param(FRAGMENTS, id="zigzag"),
    pytest.param([list(reversed(row)) for row in FRAGMENTS], id="reversed-span-order"),
    pytest.param([[(7, 11), (0, 3), (13, 16), (3, 7), (11, 13)]], id="one-rank-reordered"),
    pytest.param([[(0, 16)]], id="one-fragment"),
]
LAYOUTS = [
    pytest.param((0, 5, 5, 12, 16), id="two-cuts-and-an-empty-sequence"),
    pytest.param((0, 16), id="one-sequence"),
    pytest.param((0, 1, 2, 3, 4, 8, 16), id="many-short-documents"),
]


def summary(transition: list[list[float]], bias: list[list[float]]) -> torch.Tensor:
    """One-head ``[bias; transition]`` summary."""
    return torch.cat(
        (torch.tensor(bias, dtype=torch.float32), torch.tensor(transition, dtype=torch.float32))
    ).unsqueeze(0)


def apply(state: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    """Independent affine-map oracle for the routing tests."""
    return state @ packed[:, VALUE_DIM:, :] + packed[:, :VALUE_DIM, :]


def spans(plan: ContextParallelPlan) -> list[tuple[int, int]]:
    """``(first, last)`` local subsequence index of each fragment, in slot order."""
    result, first = [], 0
    for fragment in plan.fragments:
        result.append((first, first + len(fragment) - 1))
        first += len(fragment)
    return result


@pytest.mark.parametrize(
    ("cu_seqlens", "fragments", "message"),
    [
        pytest.param((0, 4), [[(0, 2)], [(3, 4)]], "tile", id="gap"),
        pytest.param((0, 4), [[(0, 2)], [(1, 4)]], "tile", id="overlap"),
        pytest.param((0, 4), [[(1, 4)]], "tile", id="missing-prefix"),
        pytest.param((0, 4), [[(0, 3)]], "tile", id="missing-suffix"),
        pytest.param((0, 4), [[(0, 2)], [(2, 4), (0, 2)]], "tile", id="duplicate"),
        pytest.param((0, 4), [[(0, 4)], []], "at least one", id="empty-rank"),
        pytest.param((0, 4), [[(2, 2), (0, 4)]], "empty", id="empty-fragment"),
        pytest.param((0,), [[(0, 4)]], "at least one sequence", id="no-sequences"),
        pytest.param((1, 4), [[(0, 4)]], "start at zero", id="offset-stream"),
        pytest.param((0, 4, 2), [[(0, 4)]], "nondecreasing", id="decreasing-offsets"),
        pytest.param((0, 0), [[(0, 0)]], "empty", id="empty-stream"),
    ],
)
def test_plan_rejects_fragments_that_do_not_tile_the_stream(cu_seqlens, fragments, message):
    with pytest.raises(ValueError, match=message):
        ContextParallelPlan.from_fragments(cu_seqlens, fragments, cp_rank=0)


def test_plan_rejects_rank_outside_table():
    with pytest.raises(ValueError, match="cp_rank 2"):
        ContextParallelPlan.from_fragments((0, 4), [[(0, 2)], [(2, 4)]], cp_rank=2)


def test_plan_cuts_fragments_at_sequences_and_orders_neighbors():
    # A = [0, 8), B = [8, 12); cp_rank 0 owns A[0:2], then the fragment [6, 10) spanning A and B.
    fragments = [[(0, 2), (6, 10)], [(2, 6), (10, 12)]]

    plan = ContextParallelPlan.from_fragments((0, 8, 12), fragments, cp_rank=0)
    assert plan.subsequences == (Subsequence(0, 0, 2), Subsequence(0, 6, 8), Subsequence(1, 8, 10))
    assert plan.fragments[1] == (Subsequence(0, 6, 8), Subsequence(1, 8, 10))
    assert plan.table[1] == ((Subsequence(0, 2, 6),), (Subsequence(1, 10, 12),))
    assert plan.cu_seqlens == (0, 2, 4, 6)
    assert plan.slots == 2  # one per fragment, not per subsequence
    # Per fragment. A's pieces sit in cp_rank 0 fragment 0, cp_rank 1 fragment 0, then cp_rank 0
    # fragment 1, so fragment 1 has two predecessors and fragment 0 two successors.
    assert plan.predecessors == ((), ((0, 0), (1, 0)))
    assert plan.successors == (((0, 1), (1, 0)), ((1, 1),))
    assert plan.terminal == (1,)
    assert plan.global_token_ids("cpu").tolist() == [0, 1, 6, 7, 8, 9]

    other = ContextParallelPlan.from_fragments((0, 8, 12), fragments, cp_rank=1)
    assert other.slots == 2
    assert other.predecessors == (((0, 0),), ((0, 1),))
    assert other.successors == (((0, 1),), ())
    assert other.terminal == (1,)


def test_plan_skips_empty_sequences_and_links_adjacent_fragments_on_one_rank():
    # Sequence 1 is empty, so it has no subsequence anywhere while sequence 2 keeps its index.
    fragments = [[(0, 2), (2, 4)], [(4, 8)]]
    plan = ContextParallelPlan.from_fragments((0, 4, 4, 8), fragments, cp_rank=0)
    assert plan.subsequences == (Subsequence(0, 0, 2), Subsequence(0, 2, 4))
    assert plan.predecessors == ((), ((0, 0),))
    assert plan.terminal == (1,)
    other = ContextParallelPlan.from_fragments((0, 4, 4, 8), fragments, cp_rank=1)
    assert other.subsequences == (Subsequence(2, 4, 8),)
    assert other.predecessors == ((),)
    # Leading and trailing empty sequences leave one subsequence that ends its sequence.
    plan = ContextParallelPlan.from_fragments((0, 0, 4, 4), [[(0, 4)]], cp_rank=0)
    assert plan.subsequences == (Subsequence(1, 0, 4),)
    assert plan.terminal == (0,)


@pytest.mark.parametrize("fragments", FRAGMENT_TABLES)
@pytest.mark.parametrize("cu_seqlens", LAYOUTS)
def test_entry_and_exit_folds_follow_subsequence_geometry_for_every_subsequence(
    cu_seqlens, fragments
):
    """Independently derive each subsequence's neighbours from token order and compare both folds."""
    plans = [
        ContextParallelPlan.from_fragments(cu_seqlens, fragments, cp_rank)
        for cp_rank in range(len(fragments))
    ]
    generator = torch.Generator().manual_seed(3)
    gathered = torch.randn(len(plans), plans[0].slots, 1, 4, 2, generator=generator)
    everything = [
        (r, s, piece)
        for r, row in enumerate(plans[0].table)
        for s, fragment in enumerate(row)
        for piece in fragment
    ]
    zero = torch.zeros(1, VALUE_DIM, KEY_DIM)
    for plan in plans:
        routing = plan.routing("cpu")
        count = len(plan.subsequences)
        d_final_state = torch.randn(count, 1, VALUE_DIM, KEY_DIM, generator=generator)
        entries = compose_entry_states(gathered, routing)
        exits = compose_exit_cotangents(gathered, torch.zeros_like(d_final_state), routing)
        exits_with_loss = compose_exit_cotangents(gathered, d_final_state, routing)
        for index, subsequence in enumerate(plan.subsequences):
            siblings = [(r, s, f) for r, s, f in everything if f.sequence == subsequence.sequence]
            earlier = sorted(
                (f.start, r, s) for r, s, f in siblings if f.start < subsequence.start
            )
            later = sorted((f.start, r, s) for r, s, f in siblings if f.start > subsequence.start)
            entry = zero
            for _, r, s in earlier:
                entry = apply(entry, gathered[r, s])
            exit_cotangent = zero
            for _, r, s in reversed(later):
                exit_cotangent = apply(exit_cotangent, gathered[r, s])
            torch.testing.assert_close(entries[index], entry)
            torch.testing.assert_close(exits[index], exit_cotangent)
            # The loss's own cotangent adds to what flows back from the successors.
            torch.testing.assert_close(
                exits_with_loss[index], exit_cotangent + d_final_state[index]
            )
            assert (index in plan.terminal) == (not later)


def test_tail_sources_pick_the_last_tokens_of_each_fragment_and_pad_short_ones():
    # Rank 0's last subsequence is token 1 alone (sequence 1 starts there): the row is zero, zero,
    # then span token 1; the zero row is index ``tokens`` (2). Rank 1's fragment is whole.
    assert CONV_ROUTINGS[0].tail_sources.tolist() == [[2, 2, 1]]
    assert CONV_ROUTINGS[1].tail_sources.tolist() == [[2, 0, 1]]


def test_short_conv_histories_span_short_predecessors_and_reset_at_sequences():
    # One slot per fragment (one per rank here) holding the tail of its last subsequence, gathered
    # as [world * slots, history, C]: cp_rank 0's is token 1 alone (sequence 1 starts there),
    # cp_ranks 1 and 2 hold two tokens, cp_rank 3's is token 7 (sequence 2).
    tails = torch.tensor(
        [
            [[0.0], [0.0], [1.5]],
            [[0.0], [2.0], [3.0]],
            [[0.0], [4.0], [5.0]],
            [[0.0], [0.0], [7.0]],
        ]
    )
    rank_three = compose_conv_histories(tails, CONV_ROUTINGS[3])
    torch.testing.assert_close(rank_three[0, :, 0], torch.tensor([3.0, 4.0, 5.0]))
    torch.testing.assert_close(rank_three[1], torch.zeros_like(rank_three[1]))
    torch.testing.assert_close(
        compose_conv_histories(tails, CONV_ROUTINGS[1])[0, :, 0], torch.tensor([0.0, 0.0, 1.5])
    )


def test_short_conv_history_backward_is_the_transpose_of_forward_routing():
    tails = torch.zeros(4, 3, 1, requires_grad=True)
    rank_two = compose_conv_histories(tails, CONV_ROUTINGS[2])
    rank_three = compose_conv_histories(tails, CONV_ROUTINGS[3])
    loss = (rank_two[0, :, 0] * torch.tensor([10.0, 20.0, 30.0])).sum()
    loss += (rank_three[0, :, 0] * torch.tensor([40.0, 50.0, 60.0])).sum()

    (tail_gradients,) = torch.autograd.grad(loss, tails)

    # Rank 1's last token feeds rank 2's history[2] (weight 30) and rank 3's history[0] (40).
    torch.testing.assert_close(tail_gradients[1, :, 0], torch.tensor([0.0, 20.0, 70.0]))
    # Rank 0 starts both of its sequences yet must stay connected to the gathered tails so its
    # reduce-scatter participates in the collective's backward.
    assert compose_conv_histories(tails, CONV_ROUTINGS[0]).requires_grad


@pytest.mark.parametrize("fragments", FRAGMENT_TABLES)
@pytest.mark.parametrize("cu_seqlens", LAYOUTS)
def test_routing_tensors_follow_fragment_geometry(cu_seqlens, fragments):
    """Bounds, predecessors/successors, and source maps are per fragment; padding is unrouted."""
    plans = [
        ContextParallelPlan.from_fragments(cu_seqlens, fragments, cp_rank)
        for cp_rank in range(len(fragments))
    ]
    world, slots = len(plans), plans[0].slots
    generator = torch.Generator().manual_seed(11)
    gathered = torch.randn(world, slots, 1, 4, 2, generator=generator)
    for plan in plans:
        capacity = len(plan.subsequences) + 1
        routing = plan.routing("cpu", max_subsequences=capacity)
        count = len(plan.subsequences)
        assert routing.slots == slots and routing.world == world
        assert routing.predecessors.shape == routing.successors.shape == (slots, world * slots - 1)
        assert routing.tail_sources.shape == (slots, 0) and routing.conv_sources.shape == (
            capacity,
            0,
        )
        assert routing.cu_seqlens.tolist() == [*plan.cu_seqlens] + [plan.cu_seqlens[-1]] * (
            capacity - count
        )
        assert routing.terminal.tolist() == [i in plan.terminal for i in range(capacity)]
        entries = compose_entry_states(gathered, routing)
        exits = compose_exit_cotangents(
            gathered, torch.zeros(capacity, 1, VALUE_DIM, KEY_DIM), routing
        )
        assert entries.shape == exits.shape == (capacity, 1, VALUE_DIM, KEY_DIM)
        assert torch.equal(entries[count:], torch.zeros_like(entries[count:]))
        assert torch.equal(exits[count:], torch.zeros_like(exits[count:]))
        firsts = {first for first, _ in spans(plan)}
        lasts = {last for _, last in spans(plan)}
        for slot, (first, last) in enumerate(spans(plan)):
            # A forward slot summarizes the fragment's last subsequence iff it has successors, and
            # only that subsequence receives the folded successors.
            if plan.successors[slot]:
                assert routing.forward_bounds[slot].tolist() == list(
                    plan.cu_seqlens[last : last + 2]
                )
                assert routing.exit_sources[last].item() == slot
            else:
                assert routing.forward_bounds[slot].tolist() == [0, 0]
                assert routing.exit_sources[last].item() == routing.slots
            # A reverse slot covers the first subsequence iff it has predecessors; its entry state
            # is the folded predecessors and its loss cotangent is what the slot folds in.
            if plan.predecessors[slot]:
                assert routing.reverse_bounds[slot].tolist() == list(
                    plan.cu_seqlens[first : first + 2]
                )
                assert routing.reverse_sources[slot].item() == first
                assert routing.entry_sources[first].item() == slot
            else:
                assert routing.reverse_bounds[slot].tolist() == [0, 0]
                assert routing.entry_sources[first].item() == routing.slots
        for index in range(count):
            if index not in firsts:
                assert routing.entry_sources[index].item() == routing.slots
            if index not in lasts:
                assert routing.exit_sources[index].item() == routing.slots


@pytest.mark.parametrize("fragments", FRAGMENT_TABLES)
@pytest.mark.parametrize("cu_seqlens", LAYOUTS)
@pytest.mark.parametrize("history", [0, 1, 7, 20])
def test_conv_histories_match_global_tokens(cu_seqlens, fragments, history):
    """Halo routing reads preceding tokens within a document, independent of rank/span order."""
    plans = [
        ContextParallelPlan.from_fragments(cu_seqlens, fragments, cp_rank)
        for cp_rank in range(len(fragments))
    ]
    capacity = max(len(plan.subsequences) for plan in plans) + 1
    routings = [
        plan.routing("cpu", slots=plan.slots + 1, max_subsequences=capacity, conv_history=history)
        for plan in plans
    ]
    tokens = torch.randn(cu_seqlens[-1], 2, generator=torch.Generator().manual_seed(19))
    tails = []
    for plan, routing in zip(plans, routings, strict=True):
        span = tokens[plan.global_token_ids("cpu")]
        tails.append(torch.cat((span, span.new_zeros(1, 2)))[routing.tail_sources])
    gathered = torch.cat(tails)
    for plan, routing in zip(plans, routings, strict=True):
        expected = tokens.new_zeros(capacity, history, 2)
        for index, piece in enumerate(plan.subsequences):
            available = min(history, piece.start - cu_seqlens[piece.sequence])
            expected[index, history - available :] = tokens[piece.start - available : piece.start]
        torch.testing.assert_close(compose_conv_histories(gathered, routing), expected)


def test_routing_rejects_layouts_beyond_the_caps():
    plan = ContextParallelPlan.from_fragments((0, 1, 2, 3, 4, 8, 16), FRAGMENTS, cp_rank=0)
    with pytest.raises(ValueError, match="max_subsequences"):
        plan.routing("cpu", max_subsequences=len(plan.subsequences) - 1)
    with pytest.raises(ValueError, match="slots"):
        plan.routing("cpu", slots=1)


def test_slots_cap_pads_every_rank_to_the_same_shapes_across_fragment_tables():
    """Two tables with different fragment counts route identically shaped tensors under one cap."""
    cu_seqlens = (0, 5, 12, 16)
    tables = [FRAGMENTS, [[(0, 6)], [(6, 9), (12, 16)], [(9, 12)]]]  # 5 fragments, then 4
    generator = torch.Generator().manual_seed(7)
    gathered = torch.randn(3, 3, 1, 4, 2, generator=generator)
    for table in tables:
        for cp_rank in range(3):
            plan = ContextParallelPlan.from_fragments(cu_seqlens, table, cp_rank)
            capped = plan.routing("cpu", slots=3, max_subsequences=4, conv_history=2)
            assert capped.slots == 3
            assert capped.forward_bounds.shape == capped.reverse_bounds.shape == (3, 2)
            assert capped.predecessors.shape == capped.successors.shape == (3, 8)  # 3 * 3 - 1
            assert capped.tail_sources.shape == (3, 2) and capped.conv_sources.shape == (4, 2)
            # Padding slots are empty and unrouted, so the folds match the uncapped routing.
            exact = plan.routing("cpu", max_subsequences=4)
            assert torch.equal(capped.forward_bounds[plan.slots :], torch.zeros(3 - plan.slots, 2))
            torch.testing.assert_close(
                compose_entry_states(gathered, capped),
                compose_entry_states(gathered[:, : plan.slots], exact),
            )
