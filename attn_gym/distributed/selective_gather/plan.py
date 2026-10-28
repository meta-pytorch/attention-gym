# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Plan for selective block exchange and assembly.

Each rank lists the blocks of its assembled K/V, the K/V that attention reads. Every rank
builds its plan from the same lists, so all ranks derive the same buffer slots without
communication.

Send slots group this rank's outgoing blocks by consumer, then by block number. Receive
slots group incoming blocks by owner, then by block number. Send slots are entries in
``send_source_blocks``; the local blocks they name need not be contiguous. Receive slots are
buffer rows, so each owner's blocks land in one contiguous range.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple, TypeAlias


class BlockLocation(NamedTuple):
    """Where a K/V block is stored: its owner rank and its block number on that rank."""

    owner_rank: int
    block_number: int


@dataclass(frozen=True)
class BlockExchangePlan:
    """One rank's exchange and assembly layout.

    Build it with `BlockExchangePlan.from_assembled_blocks`. Send and receive counts for this
    rank are zero.

    Attributes:
        rank: This rank in the group.
        world_size: Number of ranks in the group.
        block_tokens: Tokens per block.
        blocks_per_rank: Local blocks on every rank.
        num_assembled_blocks: Blocks in every rank's assembled K/V.
        send_source_blocks: ``[sum(send_counts)]`` local block of each send slot.
        send_offsets: ``[world_size]`` first send slot for each consumer.
        send_counts: ``[world_size]`` blocks sent to each consumer.
        peer_recv_offsets: ``[world_size]`` first receive slot on each consumer for blocks from
            this rank. Forward writes there.
        recv_offsets: ``[world_size]`` first receive slot for each owner.
        recv_counts: ``[world_size]`` blocks received from each owner.
        peer_send_offsets: ``[world_size]`` first send slot on each owner for blocks to this
            rank. Backward writes gradients there.
        local_destination_blocks: ``[blocks_per_rank]`` assembled block of each local block.
        remote_destination_blocks: ``[sum(recv_counts)]`` assembled block of each receive slot.
        max_send_blocks: Largest send count over all ranks.
        max_recv_blocks: Largest receive count over all ranks. Shared buffers are sized by
            these maxima, so they have the same size on every rank.
    """

    rank: int
    world_size: int
    block_tokens: int
    blocks_per_rank: int
    num_assembled_blocks: int
    send_source_blocks: tuple[int, ...]
    send_offsets: tuple[int, ...]
    send_counts: tuple[int, ...]
    peer_recv_offsets: tuple[int, ...]
    recv_offsets: tuple[int, ...]
    recv_counts: tuple[int, ...]
    peer_send_offsets: tuple[int, ...]
    local_destination_blocks: tuple[int, ...]
    remote_destination_blocks: tuple[int, ...]
    max_send_blocks: int
    max_recv_blocks: int

    @classmethod
    def from_assembled_blocks(
        cls,
        assembled_blocks: Sequence[Sequence[BlockLocation]],
        *,
        rank: int,
        blocks_per_rank: int,
        block_tokens: int,
    ) -> "BlockExchangePlan":
        """Build this rank's plan.

        Every rank passes the same arguments except ``rank``, so all ranks derive matching
        buffer slots without communication.

        Args:
            assembled_blocks (Sequence[Sequence[BlockLocation]]): ``assembled_blocks[r][i]``
                is the location of block ``i`` in rank ``r``'s assembled K/V. Each rank lists all
                of its own blocks. Lists may have different lengths: every rank's assembled K/V
                has the length of the longest list, and blocks past the end of a shorter list
                are unused.
            rank (int): This rank.
            blocks_per_rank (int): Local K/V blocks on every rank.
            block_tokens (int): Tokens per block.

        Example:
            Three ranks with two blocks each. Each rank reads its own blocks and the last block
            of the rank before it::

                L = BlockLocation
                assembled_blocks = [
                    [L(0, 0), L(0, 1)],  # rank 0
                    [L(0, 1), L(1, 0), L(1, 1)],  # rank 1
                    [L(1, 1), L(2, 0), L(2, 1)],  # rank 2
                ]

            Rank 1 receives block 1 of rank 0 into assembled block 0, and copies its own blocks
            into assembled blocks 1 and 2. Every assembled K/V has 3 blocks; block 2 is unused
            on rank 0.
        """
        world_size = len(assembled_blocks)
        if world_size <= 1 or not 0 <= rank < world_size:
            raise ValueError("Block exchange needs a valid rank in a multi-rank group.")
        if block_tokens <= 0 or blocks_per_rank <= 0:
            raise ValueError("Block size and blocks per rank must be positive.")

        # transfers holds (owner, block, consumer) for every block that crosses ranks.
        transfers = []
        for consumer, locations in enumerate(assembled_blocks):
            if len(set(locations)) != len(locations):
                raise ValueError(f"Rank {consumer} lists a block twice.")
            for owner, block in locations:
                if not (0 <= owner < world_size and 0 <= block < blocks_per_rank):
                    raise ValueError(f"Rank {consumer} lists an invalid block {(owner, block)}.")
                if owner != consumer:
                    transfers.append((owner, block, consumer))
            if sum(owner == consumer for owner, _ in locations) != blocks_per_rank:
                raise ValueError(f"Rank {consumer} must list all of its own blocks.")

        # transfer_counts[owner][consumer] is the number of blocks from owner to consumer.
        transfer_counts = [[0] * world_size for _ in range(world_size)]
        # Sorted transfers list each consumer's blocks in block-number order.
        send_blocks_by_consumer: list[list[int]] = [[] for _ in range(world_size)]
        recv_blocks: list[BlockLocation] = []
        for owner, block, consumer in sorted(transfers):
            transfer_counts[owner][consumer] += 1
            if owner == rank:
                send_blocks_by_consumer[consumer].append(block)
            if consumer == rank:
                recv_blocks.append(BlockLocation(owner, block))

        position = {location: index for index, location in enumerate(assembled_blocks[rank])}
        local_destinations = tuple(
            position[BlockLocation(rank, b)] for b in range(blocks_per_rank)
        )

        def send_offset(owner: int, consumer: int) -> int:
            return sum(transfer_counts[owner][:consumer])

        def recv_offset(consumer: int, owner: int) -> int:
            return sum(transfer_counts[o][consumer] for o in range(owner))

        peers = range(world_size)
        return cls(
            rank=rank,
            world_size=world_size,
            block_tokens=block_tokens,
            blocks_per_rank=blocks_per_rank,
            num_assembled_blocks=max(map(len, assembled_blocks)),
            send_source_blocks=tuple(b for blocks in send_blocks_by_consumer for b in blocks),
            send_offsets=tuple(send_offset(rank, p) for p in peers),
            send_counts=tuple(transfer_counts[rank]),
            peer_recv_offsets=tuple(recv_offset(p, rank) for p in peers),
            recv_offsets=tuple(recv_offset(rank, p) for p in peers),
            recv_counts=tuple(transfer_counts[p][rank] for p in peers),
            peer_send_offsets=tuple(send_offset(p, rank) for p in peers),
            local_destination_blocks=local_destinations,
            remote_destination_blocks=tuple(position[location] for location in recv_blocks),
            max_send_blocks=max(sum(row) for row in transfer_counts),
            max_recv_blocks=max(sum(transfer_counts[o][c] for o in peers) for c in peers),
        )


TokenRanges: TypeAlias = Sequence[tuple[int, int]]
"""One rank's global ``[start, end)`` token ranges, in local order."""

GlobalBlocks: TypeAlias = tuple[int, ...]
"""Global block IDs of one rank's assembled K/V, in assembled order."""


def build_sliding_window_plan(
    local_ranges: Sequence[TokenRanges],
    *,
    rank: int,
    block_tokens: int,
    sliding_window_size: int,
) -> tuple[BlockExchangePlan, tuple[GlobalBlocks, ...]]:
    """Build one rank's plan for causal sliding-window attention, and every rank's layout.

    Each rank assembles the blocks its queries can see: its own blocks plus the earlier blocks
    inside the window, sorted by global block ID. Blocks are selected for the causal window
    only, so document boundaries never change the layout; the attention mask drops those
    edges.

    Args:
        local_ranges (Sequence[TokenRanges]): ``local_ranges[r]`` lists rank ``r``'s global
            ``[start, end)`` token ranges in local order: concatenated, they are its local K/V
            and Q. The ranges must tile the sequence from token 0 in whole blocks, and every
            rank must hold the same number of blocks.
        rank (int): Rank whose plan is returned.
        block_tokens (int): Tokens per transport block.
        sliding_window_size (int): Maximum visible tokens per query, including the query
            itself. A value of 4 means three earlier tokens plus the current token.

    Returns:
        tuple[BlockExchangePlan, tuple[GlobalBlocks, ...]]: The plan for ``rank``, and the
        global block IDs of every rank's assembled K/V. Assembled block ``i`` of rank ``r``
        holds global block ``global_blocks_by_rank[r][i]``.

    Raises:
        ValueError: If a count is invalid, or the ranges do not tile the sequence in whole
            blocks with the same number of blocks per rank.

    Example:
        Sixteen tokens on two contiguous shards, two tokens per block, window 3::

            plan, global_blocks_by_rank = build_sliding_window_plan(
                [[(0, 8)], [(8, 16)]], rank=1, block_tokens=2, sliding_window_size=3
            )
            # global_blocks_by_rank == ((0, 1, 2, 3), (3, 4, 5, 6, 7))

        Rank 1's first query is token 8, which sees tokens 6 to 8, so rank 1 receives
        global block 3. Both assembled K/V have five blocks; rank 0's last block is unused.
        With head-tail sharding, rank ``r`` would pass its head range, then its tail range.
    """
    world_size = len(local_ranges)
    if world_size <= 1 or not 0 <= rank < world_size:
        raise ValueError("Block exchange needs a valid rank in a multi-rank group.")
    if block_tokens <= 0 or sliding_window_size <= 0:
        raise ValueError("Block size and window size must be positive.")

    # location[g] is where global block g is stored: its owner and its local block number.
    location: dict[int, BlockLocation] = {}
    block_counts = []
    for owner, ranges in enumerate(local_ranges):
        local_block = 0
        for start, end in ranges:
            if not 0 <= start < end or start % block_tokens or end % block_tokens:
                raise ValueError(f"Rank {owner} range {(start, end)} is not made of whole blocks.")
            for block in range(start // block_tokens, end // block_tokens):
                if block in location:
                    raise ValueError(f"Global block {block} is in more than one range.")
                location[block] = BlockLocation(owner, local_block)
                local_block += 1
        block_counts.append(local_block)
    if sorted(location) != list(range(len(location))):
        raise ValueError("The ranges must tile the sequence from token 0.")
    if len(set(block_counts)) != 1:
        raise ValueError("Every rank must hold the same number of blocks.")

    global_blocks_by_rank = []
    for ranges in local_ranges:
        needed: set[int] = set()
        for start, end in ranges:
            first_key = max(0, start - sliding_window_size + 1)
            needed.update(range(first_key // block_tokens, end // block_tokens))
        global_blocks_by_rank.append(tuple(sorted(needed)))
    plan = BlockExchangePlan.from_assembled_blocks(
        [[location[block] for block in blocks] for blocks in global_blocks_by_rank],
        rank=rank,
        blocks_per_rank=block_counts[0],
        block_tokens=block_tokens,
    )
    return plan, tuple(global_blocks_by_rank)
