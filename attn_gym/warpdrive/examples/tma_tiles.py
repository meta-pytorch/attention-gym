"""Single-CTA TMA partitioning without multicast for the examples' SMEM and global tiles."""

from cutlass import cute
from cutlass.cute.nvgpu import cpasync


def tma_partition_tiles(atom, smem, gmem, tile_rank: int = 3):
    """Partition SMEM and global tiles for a single-CTA TMA copy without multicast.

    The first ``tile_rank`` modes become ``(atom_v, rest_v)``; outer modes, such as stage
    or global tile indices, stay intact. Use rank 3 for ``(MMA, MMA_M, MMA_K)`` operands
    or rank 2 for ``(M, N)`` epilogue tiles. Either input may omit the outer modes.
    """
    return cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0, tile_rank),
        cute.group_modes(gmem, 0, tile_rank),
    )
