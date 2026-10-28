"""Check block exchange plans on the host."""

import random

import pytest

from attn_gym.distributed.selective_gather import BlockExchangePlan, build_sliding_window_plan
from attn_gym.distributed.selective_gather import BlockLocation as L
from attn_gym.testing.selective_gather import local_ranges


def _plan(assembled_blocks, *, rank, blocks_per_rank):
    return BlockExchangePlan.from_assembled_blocks(
        assembled_blocks, rank=rank, blocks_per_rank=blocks_per_rank, block_tokens=1
    )


def _sequence_order(received, *, world_size, blocks_per_rank):
    """Each rank's own blocks plus ``received[r]``, in rank-major sequence order."""
    return [
        sorted([L(rank, b) for b in range(blocks_per_rank)] + [L(*x) for x in received[rank]])
        for rank in range(world_size)
    ]


def _simulate(plans, blocks, grads):
    """Run the exchange, assembly, and backward reduction with plain lists.

    ``blocks[r]`` and ``grads[r]`` are rank ``r``'s local blocks and the gradients of its
    receive slots. Returns each rank's assembled output and summed local gradients.
    """
    recv = [[None] * plan.max_recv_blocks for plan in plans]
    backward_recv = [[None] * plan.max_send_blocks for plan in plans]
    for owner, plan in enumerate(plans):
        send = [blocks[owner][block] for block in plan.send_source_blocks]
        for peer in range(plan.world_size):
            start, count = plan.send_offsets[peer], plan.send_counts[peer]
            destination = plan.peer_recv_offsets[peer]
            recv[peer][destination : destination + count] = send[start : start + count]
    for consumer, plan in enumerate(plans):
        for peer in range(plan.world_size):
            start, count = plan.recv_offsets[peer], plan.recv_counts[peer]
            destination = plan.peer_send_offsets[peer]
            backward_recv[peer][destination : destination + count] = grads[consumer][
                start : start + count
            ]
    outputs, reduced = [], []
    for rank, plan in enumerate(plans):
        output = [-1] * plan.num_assembled_blocks
        for block, destination in enumerate(plan.local_destination_blocks):
            output[destination] = blocks[rank][block]
        for slot, destination in enumerate(plan.remote_destination_blocks):
            output[destination] = recv[rank][slot]
        outputs.append(output)
        local = [0] * plan.blocks_per_rank
        for slot, block in enumerate(plan.send_source_blocks):
            local[block] += backward_recv[rank][slot]
        reduced.append(local)
    return outputs, reduced


@pytest.mark.parametrize("seed", range(5))
def test_plans_agree_across_ranks(seed: int) -> None:
    rng = random.Random(seed)
    world_size, blocks_per_rank = 4, 5
    # Each rank reads a random set of remote blocks, in a shuffled assembled order.
    assembled_blocks = []
    for rank in range(world_size):
        locations = [
            L(owner, block)
            for owner in range(world_size)
            for block in range(blocks_per_rank)
            if owner == rank or rng.random() < 0.4
        ]
        rng.shuffle(locations)
        assembled_blocks.append(locations)
    plans = [
        _plan(assembled_blocks, rank=rank, blocks_per_rank=blocks_per_rank)
        for rank in range(world_size)
    ]
    # Block value 10 * owner + block names the block; gradient 1000 + 10 * consumer + slot.
    blocks = [
        [10 * owner + block for block in range(blocks_per_rank)] for owner in range(world_size)
    ]
    grads = [
        [1000 + 10 * consumer + slot for slot in range(plans[consumer].max_recv_blocks)]
        for consumer in range(world_size)
    ]
    outputs, reduced = _simulate(plans, blocks, grads)

    def received(consumer):
        return sorted(x for x in assembled_blocks[consumer] if x.owner_rank != consumer)

    num_assembled_blocks = max(map(len, assembled_blocks))
    for rank, plan in enumerate(plans):
        assert plan.num_assembled_blocks == num_assembled_blocks
        expected = [10 * owner + block for owner, block in assembled_blocks[rank]]
        expected += [-1] * (num_assembled_blocks - len(expected))
        assert outputs[rank] == expected

        expected_grad = [0] * blocks_per_rank
        for consumer in range(world_size):
            for slot, (owner, block) in enumerate(received(consumer)):
                if owner == rank:
                    expected_grad[block] += grads[consumer][slot]
        assert reduced[rank] == expected_grad


def test_plan_layout_is_grouped_by_peer() -> None:
    received = [[(1, 3), (1, 0), (2, 1)], [(0, 4), (0, 0), (2, 3)], [(0, 1), (1, 1)], [(0, 2)]]
    assembled_blocks = _sequence_order(received, world_size=4, blocks_per_rank=5)
    plan = _plan(assembled_blocks, rank=2, blocks_per_rank=5)
    assert plan.send_source_blocks == (1, 3)
    assert plan.send_counts == (1, 1, 0, 0)
    assert plan.send_offsets == (0, 1, 2, 2)
    # Rank 0 receives 2 blocks from rank 1 and rank 1 receives 2 from rank 0 before ours.
    assert plan.peer_recv_offsets[:2] == (2, 2)
    assert plan.recv_counts == (1, 1, 0, 0)
    assert plan.recv_offsets == (0, 1, 2, 2)
    # Rank 0 sends 2 blocks to rank 1 and rank 1 sends 2 to rank 0 before ours.
    assert plan.peer_send_offsets[:2] == (2, 2)
    # Rank 0 sends 4 blocks; rank 1 receives 3.
    assert (plan.max_send_blocks, plan.max_recv_blocks) == (4, 3)
    # Rank 2 assembles [(0, 1), (1, 1), own blocks]; ranks 0 and 1 assemble 8 blocks.
    assert plan.num_assembled_blocks == 8
    assert plan.local_destination_blocks == (2, 3, 4, 5, 6)
    assert plan.remote_destination_blocks == (0, 1)


def test_destinations_follow_assembled_order() -> None:
    plan = _plan([[L(1, 0), L(0, 1), L(0, 0)], [L(1, 0), L(1, 1)]], rank=0, blocks_per_rank=2)
    assert plan.num_assembled_blocks == 3
    assert plan.local_destination_blocks == (2, 1)
    assert plan.remote_destination_blocks == (0,)


def test_empty_plan() -> None:
    plan = _plan(
        _sequence_order([[], []], world_size=2, blocks_per_rank=2), rank=0, blocks_per_rank=2
    )
    assert plan.send_source_blocks == plan.remote_destination_blocks == ()
    assert plan.send_counts == plan.recv_counts == (0, 0)
    assert plan.max_send_blocks == plan.max_recv_blocks == 0


@pytest.mark.parametrize(
    ("rank_0_blocks", "message"),
    (
        ([L(0, 0), L(0, 1), L(1, 0), L(1, 0)], "twice"),
        ([L(0, 0), L(0, 1), L(2, 0)], "invalid block"),
        ([L(0, 0), L(0, 1), L(1, 2)], "invalid block"),
        ([L(0, 0), L(1, 0)], "all of its own blocks"),
    ),
)
def test_rejects_invalid_assembled_blocks(rank_0_blocks, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _plan([rank_0_blocks, [L(1, 0), L(1, 1)]], rank=0, blocks_per_rank=2)


@pytest.mark.parametrize(
    ("world_size", "rank", "blocks_per_rank", "message"),
    ((1, 0, 2, "multi-rank"), (2, 2, 2, "multi-rank"), (2, 0, 0, "positive")),
)
def test_rejects_invalid_group(
    world_size: int, rank: int, blocks_per_rank: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _plan([[]] * world_size, rank=rank, blocks_per_rank=blocks_per_rank)


# (global_tokens, world_size, block_tokens, sliding_window_size, head_tail)
SLIDING_WINDOW_CASES = (
    (16, 2, 2, 3, False),  # the docstring example
    (32, 4, 2, 3, True),  # head-tail; ranks receive different counts
    (16, 2, 2, 1, False),  # window 1: no remote blocks
    (32, 4, 2, 1, True),
    (32, 2, 4, 3, False),  # window smaller than a block
    (32, 2, 4, 5, False),  # the first visible key starts a block
    (32, 4, 2, 12, False),  # the window crosses two owners
    (32, 4, 2, 9, True),  # head and tail halos overlap the rank's own blocks
    (16, 2, 2, 40, False),  # window longer than the sequence
    (32, 4, 2, 100, True),
)


@pytest.mark.parametrize("case", SLIDING_WINDOW_CASES)
def test_sliding_window_plan(case) -> None:
    global_tokens, world_size, block_tokens, window, head_tail = case
    ranges = local_ranges(global_tokens, world_size, head_tail=head_tail)
    results = [
        build_sliding_window_plan(
            ranges, rank=rank, block_tokens=block_tokens, sliding_window_size=window
        )
        for rank in range(world_size)
    ]
    plans = [plan for plan, _ in results]
    global_blocks_by_rank = results[0][1]
    assert all(layout == global_blocks_by_rank for _, layout in results)

    # Each rank's global token IDs in local order.
    local_tokens = [[t for start, end in r for t in range(start, end)] for r in ranges]
    for rank, tokens in enumerate(local_tokens):
        # Token oracle: the blocks of every key that some local query can see.
        visible = {
            key // block_tokens
            for query in tokens
            for key in range(max(0, query - window + 1), query + 1)
        }
        assert global_blocks_by_rank[rank] == tuple(sorted(visible))

    # Each block carries its global ID. Assembled block i must hold global_blocks_by_rank[r][i].
    blocks = [tokens[::block_tokens] for tokens in local_tokens]
    blocks = [[token // block_tokens for token in rank_blocks] for rank_blocks in blocks]
    num_assembled = max(map(len, global_blocks_by_rank))
    zero_grads = [[0] * plan.max_recv_blocks for plan in plans]
    outputs, _ = _simulate(plans, blocks, zero_grads)
    for rank, layout in enumerate(global_blocks_by_rank):
        assert plans[rank].num_assembled_blocks == num_assembled
        assert outputs[rank] == list(layout) + [-1] * (num_assembled - len(layout))

    # Consumer c returns 1000 * (c + 1) + g for global block g, so the owner's sum is known.
    grads = [
        [1000 * (c + 1) + outputs[c][slot] for slot in plan.remote_destination_blocks]
        + [0] * (plan.max_recv_blocks - len(plan.remote_destination_blocks))
        for c, plan in enumerate(plans)
    ]
    _, reduced = _simulate(plans, blocks, grads)
    for rank, rank_blocks in enumerate(blocks):
        expected = [
            sum(
                1000 * (c + 1) + block
                for c in range(world_size)
                if c != rank and block in global_blocks_by_rank[c]
            )
            for block in rank_blocks
        ]
        assert reduced[rank] == expected
    if window == 1:
        assert all(plan.max_send_blocks == plan.max_recv_blocks == 0 for plan in plans)


def test_sliding_window_plan_examples() -> None:
    def build(global_tokens, world_size, rank, head_tail):
        ranges = local_ranges(global_tokens, world_size, head_tail=head_tail)
        return build_sliding_window_plan(ranges, rank=rank, block_tokens=2, sliding_window_size=3)

    plan, layout = build(16, 2, 0, head_tail=False)
    assert layout == ((0, 1, 2, 3), (3, 4, 5, 6, 7))
    assert plan.num_assembled_blocks == 5
    plans = [build(32, 4, rank, head_tail=True)[0] for rank in range(4)]
    layout = build(32, 4, 0, head_tail=True)[1]
    assert layout[0] == (0, 1, 13, 14, 15)
    assert layout[3] == (5, 6, 7, 8, 9)
    assert plans[0].num_assembled_blocks == 6
    # Block value 10 * owner + local block: rank 0 receives block 3 of rank 1 as global block 13.
    blocks = [[10 * owner + block for block in range(4)] for owner in range(4)]
    outputs, _ = _simulate(plans, blocks, [[0] * p.max_recv_blocks for p in plans])
    assert outputs[0] == [0, 1, 13, 2, 3, -1]


@pytest.mark.parametrize(
    ("ranges", "rank", "block_tokens", "window", "message"),
    (
        ([[(0, 8)]], 0, 2, 3, "multi-rank"),
        ([[(0, 8)], [(8, 16)]], 2, 2, 3, "multi-rank"),
        ([[(0, 8)], [(8, 16)]], 0, 0, 3, "positive"),
        ([[(0, 8)], [(8, 16)]], 0, 2, 0, "positive"),
        ([[(0, 7)], [(7, 16)]], 0, 2, 3, "whole blocks"),
        ([[(0, 8)], [(6, 14)]], 0, 2, 3, "more than one range"),
        ([[(0, 8)], [(10, 18)]], 0, 2, 3, "tile"),
        ([[(0, 4)], [(4, 16)]], 0, 2, 3, "same number"),
    ),
)
def test_sliding_window_plan_rejects_invalid_inputs(
    ranges, rank, block_tokens, window, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_sliding_window_plan(
            ranges, rank=rank, block_tokens=block_tokens, sliding_window_size=window
        )
