"""The weight ring: its depth in the shared-memory budget, its stage layouts, TMA atoms and
partitions, and the load warp's loads of 16 KB stages (``constants``) of weight or KV tiles,
whole or as ``multi_gemv``'s swizzled boxes, into a ``Ring`` the compute warps consume."""

import cutlass
from cutlass import BFloat16, Int64, cute
from cutlass.cute.nvgpu import cpasync

from attn_gym.warpdrive.examples.megakernel.constants import (
    BLOCK,
    BOX,
    EVICT_FIRST,
    HEAD_DIM,
    MAX_STAGES,
    QBOX,
    ROWS,
    SMEM_BYTES,
    SMEM_HEADROOM,
    STAGE_BYTES,
    TILE_K,
)


@cute.jit
def load_tiles(w, atom, tiles, row, k0, count: cutlass.Constexpr, buffer: cutlass.Constexpr,
               policy: cutlass.Constexpr = EVICT_FIRST):  # fmt: skip
    """``count`` stages of tiles ``(row, k0 + k)`` into the ring's ``buffer`` view with the L2
    cache ``policy``: weights and KV rows are read once per launch, hence evict-first."""
    for k in cutlass.range_constexpr(count):
        with w.produce() as h:
            cute.copy(
                atom,
                tiles[None, row, k0 + k],
                h.view(buffer),
                tma_bar_ptr=h.barrier,
                cache_policy=Int64(policy),
            )
    return w


def ring_stages(
    fixed: int, stages: int | None, *, smem_bytes: int = SMEM_BYTES, min_stages: int = 2
) -> int:
    """Depth of a weight ring of ``STAGE_BYTES`` stages beside ``fixed`` bytes of other shared
    memory: ``stages``, by default as many as fit in ``smem_bytes`` up to ``MAX_STAGES``."""
    fit = (smem_bytes - fixed - SMEM_HEADROOM) // STAGE_BYTES
    if fit < min_stages:
        required = fixed + SMEM_HEADROOM + min_stages * STAGE_BYTES
        raise ValueError(
            f"kernel needs at least {required} bytes of shared memory, has {smem_bytes}"
        )
    stages = min(MAX_STAGES, fit) if stages is None else stages
    if not min_stages <= stages <= fit:
        raise ValueError(f"stages must be in [{min_stages}, {fit}], got {stages}")
    return stages


def weight_stages(stages: int) -> cute.Layout:
    """The ring's ``stages`` weight stages as row-major ``(ROWS, TILE_K, stages)`` bf16 tiles."""
    return cute.make_ordered_layout((ROWS, TILE_K, stages), order=(1, 0, 2))


def kv_stages(stages: int) -> cute.Layout:
    """The ring's ``stages`` KV stages as row-major ``(BLOCK, HEAD_DIM, stages)`` bf16 blocks."""
    return cute.make_ordered_layout((BLOCK, HEAD_DIM, stages), order=(1, 0, 2))


def tma_atom(src, box: tuple[int, ...]) -> tuple:
    """TMA ``(atom, tensor)`` loading ``box`` tiles of ``src`` into row-major SMEM."""
    smem = cute.make_ordered_layout(box, order=tuple(reversed(range(len(box)))))
    return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), src, smem, box)


@cute.jit
def load_boxes(w, atom, tiles, row, k0, count: cutlass.Constexpr, buffer: cutlass.Constexpr):
    """``load_tiles`` of ``count`` 16 KB stages, each as ``TILE_K // BOX`` swizzled TMA boxes
    (``multi_gemv``) into the ring's view ``buffer``: box ``i`` of stage ``k`` is tile ``(row,
    (k0 + k) * boxes + i)``."""
    boxes = TILE_K // BOX
    for k in cutlass.range_constexpr(count):
        with w.produce() as h:
            stage = h.view(buffer)
            for i in cutlass.range_constexpr(boxes):
                cute.copy(
                    atom,
                    tiles[None, row, (k0 + k) * boxes + i],
                    stage[None, i],
                    tma_bar_ptr=h.barrier,
                    cache_policy=Int64(EVICT_FIRST),
                )
    return w


def box_layout(dtype, stages: int | None = None):
    """``multi_gemv``'s swizzled TMA box of ``dtype`` (bf16 or 32-bit words): ``(ROWS, box)``
    of 128-byte rows whose 16-byte chunks XOR with the row (``S<3, 4, 3>`` on byte addresses);
    with ``stages``, the weight ring's ``stages`` 16 KB stages as ``(ROWS, box, boxes,
    stages)`` of them."""
    box = BOX if dtype is BFloat16 else QBOX
    shape = (ROWS, box) if stages is None else (ROWS, box, TILE_K // BOX, stages)
    layout = cute.make_ordered_layout(shape, order=(1, 0, 2, 3)[: len(shape)])
    return cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, layout)


def swizzled_atom(src, dtype) -> tuple:
    """TMA ``(atom, tensor)`` loading ``src``'s tiles into ``box_layout(dtype)`` boxes."""
    smem = box_layout(dtype)
    return cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), src, smem, smem.shape)


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
