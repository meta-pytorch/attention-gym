"""Single-CTA TMA partitioning without multicast for the examples' SMEM and global tiles."""

from cutlass import cute
from cutlass.cute.nvgpu import cpasync


def tma_partition_tiles(atom, smem, gmem, tile_rank: int = 3):
    """``cpasync.tma_partition`` of ``smem`` ``(tile..., STAGE)`` and ``gmem``
    ``(tile..., tiles...)`` into ``((atom_v, rest_v), STAGE)`` and ``((atom_v, rest_v),
    tiles...)``. A tile spans the first ``tile_rank`` modes: ``(MMA, MMA_M, MMA_K)`` of an
    MMA-partitioned operand, ``(M, N)`` of an epilogue tile. Either side may be a tile instead
    of a stack of them (the result then has no outer mode)."""
    return cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0, tile_rank),
        cute.group_modes(gmem, 0, tile_rank),
    )
