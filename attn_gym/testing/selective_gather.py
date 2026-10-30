"""Shared helpers for selective block exchange tests."""

from attn_gym.distributed.selective_gather import BlockLocation


def sliding_window_assembled_blocks(
    world_size: int, blocks_per_rank: int, window_blocks: int, *, head_tail: bool
) -> list[list[BlockLocation]]:
    """Compact assembled blocks for a causal sliding window over whole blocks.

    Each block needs the ``window_blocks`` blocks before it in sequence order. Each rank
    assembles the blocks it needs in sequence order. Rank-major: rank ``r`` owns chunk ``r``,
    so it assembles ``[halo, local]``. Head-tail: rank ``r`` owns chunks ``r`` and
    ``2W - 1 - r`` as its first and second half, so it assembles
    ``[head halo, head, tail halo, tail]``.
    """
    chunks = 2 * world_size if head_tail else world_size
    chunk_blocks = blocks_per_rank * world_size // chunks
    # location[g] is where global block g is stored.
    location = {}
    for owner in range(world_size):
        owned_chunks = [owner, chunks - 1 - owner] if head_tail else [owner]
        for index, chunk in enumerate(owned_chunks):
            for offset in range(chunk_blocks):
                location[chunk * chunk_blocks + offset] = BlockLocation(
                    owner, index * chunk_blocks + offset
                )
    assembled_blocks = []
    for rank in range(world_size):
        needed = set()
        for block, (owner, _) in location.items():
            if owner == rank:
                needed.update(range(max(block - window_blocks, 0), block + 1))
        assembled_blocks.append([location[block] for block in sorted(needed)])
    return assembled_blocks
