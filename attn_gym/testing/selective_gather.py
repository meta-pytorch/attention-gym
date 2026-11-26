"""Shared helpers for selective block exchange tests."""


def local_ranges(
    global_tokens: int, world_size: int, *, head_tail: bool
) -> list[list[tuple[int, int]]]:
    """Each rank's global ``[start, end)`` token ranges in local order.

    Contiguous sharding gives rank ``r`` chunk ``r``. Head-tail sharding gives it chunk ``r``,
    then chunk ``2 * world_size - 1 - r``.
    """
    num_chunks = 2 * world_size if head_tail else world_size
    chunk_tokens = global_tokens // num_chunks

    def chunk(index: int) -> tuple[int, int]:
        return (index * chunk_tokens, (index + 1) * chunk_tokens)

    if head_tail:
        return [[chunk(r), chunk(num_chunks - 1 - r)] for r in range(world_size)]
    return [[chunk(r)] for r in range(world_size)]
