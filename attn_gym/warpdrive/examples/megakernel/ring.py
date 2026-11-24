"""The weight ring: its depth in the shared-memory budget, its stage layouts and TMA
partitions, and the load warp's loads of 16 KB stages (``constants``) of weight or KV tiles
into a ``Ring`` the compute warps consume."""

import cutlass
from cutlass import Int64, cute
from cutlass.cute.nvgpu import cpasync

from attn_gym.warpdrive.examples.megakernel.constants import (
    BLOCK,
    EVICT_FIRST,
    HEAD_DIM,
    MAX_STAGES,
    ROWS,
    SMEM_BYTES,
    SMEM_HEADROOM,
    STAGE_BYTES,
    TILE_K,
)


@cute.jit
def load_tiles(w, atom, tiles, row, k0, count: cutlass.Constexpr, buffer: cutlass.Constexpr):
    """``count`` stages of tiles ``(row, k0 + k)`` into the ring's ``buffer`` view. Weights and
    KV rows are read once per launch, hence evict-first in L2."""
    for k in cutlass.range_constexpr(count):
        with w.produce() as h:
            cute.copy(
                atom,
                tiles[None, row, k0 + k],
                h.view(buffer),
                tma_bar_ptr=h.barrier,
                cache_policy=Int64(EVICT_FIRST),
            )
    return w


def ring_stages(fixed: int, stages: int | None) -> int:
    """Depth of a weight ring of ``STAGE_BYTES`` stages beside ``fixed`` bytes of other shared
    memory: ``stages``, by default as many as fit up to ``MAX_STAGES``."""
    fit = (SMEM_BYTES - fixed - SMEM_HEADROOM) // STAGE_BYTES
    stages = min(MAX_STAGES, fit) if stages is None else stages
    if not 2 <= stages <= fit:
        raise ValueError(f"stages must be in [2, {fit}], got {stages}")
    return stages


def weight_stages(stages: int) -> cute.Layout:
    """The ring's ``stages`` weight stages as row-major ``(ROWS, TILE_K, stages)`` bf16 tiles."""
    return cute.make_layout((ROWS, TILE_K, stages), stride=(TILE_K, 1, ROWS * TILE_K))


def kv_stages(stages: int) -> cute.Layout:
    """The ring's ``stages`` KV stages as row-major ``(BLOCK, HEAD_DIM, stages)`` bf16 blocks."""
    return cute.make_layout((BLOCK, HEAD_DIM, stages), stride=(HEAD_DIM, 1, BLOCK * HEAD_DIM))


def tma_partition(atom, smem, gtensor, box: tuple[int, ...]) -> tuple:
    """TMA ``atom``'s ``(SMEM, GMEM)`` partitions: the stages of ``smem`` (box modes first),
    and ``gtensor``'s ``box`` tiles, ``(TMA, tile coordinates...)``."""
    rank = len(box)
    tiles = cute.local_tile(gtensor, box, (None,) * rank)
    return cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0, rank),
        cute.group_modes(tiles, 0, rank),
    )
