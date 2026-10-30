"""Check block exchange plans on the host."""

import random

import pytest

from attn_gym.distributed.selective_gather import BlockExchangePlan
from attn_gym.distributed.selective_gather import BlockLocation as L


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
