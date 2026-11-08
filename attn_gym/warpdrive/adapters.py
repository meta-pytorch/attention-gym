"""External scheduler tile orders as warpdrive work mappings.

The adapter supplies ``index -> tile``; the warpdrive schedule distributes indices::

    self.work = WorkStream(mapping=CutlassRaster(swizzle=8), schedule=StaticPersistent())
    self.work = WorkStream(mapping=CutlassRaster(swizzle=8), schedule=ClusterLaunchControl())

``StaticPersistent()`` reproduces the external per-CTA index sequence, not its lifecycle.
For ``Fa4Mapping(SingleTileLPTScheduler)``, use ``OneShot()`` for that sequence. Kernels still
bind ``(m_tiles, n_tiles, l)``; the adapter builds external parameters in the entry point.

Only index-addressable schedulers fit: the tile must be a pure function of one linear index.
Supported orders are CUTLASS ``StaticPersistentTileScheduler`` and FA4
``StaticPersistentTileScheduler``/``SingleTileLPTScheduler`` (``STATIC`` mode). FA4
``SingleTileScheduler`` needs 3-D ``blockIdx``. QuACK and CLC/dynamic schedulers own pipelines,
SMEM and barriers that would duplicate warpdrive's distribution.

External ``initial_work_tile_info``, ``advance_to_next_work``, ``prefetch_next_work``,
``producer_tail`` and ``get_grid_shape`` are not called. The warpdrive schedule owns the grid.
Indices count work units: a ``CtaPair`` has one item per cluster, with FA4 params using a
``(1, 1)`` cluster shape. An extent <= 0 means zero items. FA4 params use at least one tile
and one KV key to avoid division by zero, but the normalized count remains zero.
Both adapters supply an exact Int64 ``count64``; launch rejects counts past ``2**31``.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import cutlass
from cutlass import Int32, Int64

from attn_gym.warpdrive.scheduling import Raster, WorkMap


def _round_up(extent, multiple: int):
    """Round up in Int64 for DSL values to avoid Int32 overflow."""
    if isinstance(extent, int):
        return -(-extent // multiple) * multiple
    return (Int64(extent) + (multiple - 1)) // multiple * multiple


@dataclass(frozen=True)
class CutlassRaster(WorkMap):
    """CUTLASS static-persistent tile order over ``(m_tiles, n_tiles, l)``.

    ``along`` is the raster axis (``raster_along_m``); ``swizzle`` is the band width
    (``swizzle_size``). The other axis is padded to full bands, then visited in
    ``Raster(along, swizzle)`` order. Padded tiles return ``keep=False`` and never reach
    roles. ``bind`` normalizes empty extents like ``Raster``; ``decode`` accepts any index.

    Source: BSD-3 CuTeDSL helper, NVIDIA/cutlass ``e13f5cd``,
    ``examples/python/CuTeDSL/helpers/static_persistent_tile_scheduler.py``.
    """

    swizzle: int = 1
    along: str = "m"
    words = 3
    decodes_any_index = True

    def __post_init__(self):
        if self.along not in ("m", "n"):
            raise ValueError(f"CutlassRaster.along must be 'm' or 'n', got {self.along!r}")
        if type(self.swizzle) is not int or self.swizzle < 1:
            raise ValueError(f"CutlassRaster.swizzle must be a positive int, got {self.swizzle!r}")

    @property
    def skips(self) -> bool:
        return self.swizzle > 1

    def _pad(self, extents, round_up):
        """Round the axis other than ``along`` up to a multiple of ``swizzle``."""
        em, en, el = extents
        if self.along == "m":
            return em, round_up(en), el
        return round_up(em), en, el

    def bind(self, space):
        extents = Raster().bind(space)
        s = self.swizzle
        # Int32 like CUTLASS: it wraps only for spaces whose count64 the launch rejects.
        padded = extents if s == 1 else self._pad(extents, lambda e: (e + s - 1) // s * s)
        return padded, extents

    def count64(self, space):
        _, extents = space
        return Raster().count64(self._pad(extents, lambda e: _round_up(e, self.swizzle)))

    def decode(self, index, space, state):
        padded, (em, en, _) = space
        raster = Raster(along=self.along, swizzle=self.swizzle)
        _, (m, n, li), _ = raster.decode(index, padded, None)
        if self.swizzle == 1:
            return True, (m, n, li), state
        return (m < em) & (n < en), (m, n, li), state

    def get_key(self) -> str:
        return f"cutlasssp{self.along}{self.swizzle}"

    def describe(self) -> str:
        return f"CutlassRaster(along={self.along}, swizzle={self.swizzle})"


def _fa4_tile_scheduler():
    try:
        return importlib.import_module("flash_attn.cute.tile_scheduler")
    except ImportError as exc:
        raise ImportError(
            "Fa4Mapping needs flash_attn.cute (FlashAttention 4, package flash-attn-4)"
        ) from exc


@dataclass(frozen=True)
class Fa4Mapping(WorkMap):
    """FA4 scheduler mapping with ``(block, head, batch)`` items and no split-KV.

    ``scheduler`` must be ``StaticPersistentTileScheduler`` or ``SingleTileLPTScheduler``
    from ``flash_attn.cute.tile_scheduler``; the latter uses ``STATIC`` mode.
    ``space`` is ``(num_block, num_head, num_batch)``, plus ``seqlen_k`` for LPT's L2 sizing.
    Extents normalize like ``Raster``. Blocks count work units, one per ``CtaPair`` cluster;
    FA4 params use a ``(1, 1)`` cluster shape.

    LPT compile choices are fields and part of ``get_key``: ``lpt`` reverses block order,
    ``lpt_global`` uses one L2 section, and ``head_dim``, ``head_dim_v``, ``element_size``
    determine K/V bytes per key. ``StaticPersistentTileScheduler`` ignores these fields.
    """

    scheduler: type
    lpt: bool = True
    lpt_global: bool = False
    head_dim: int = 128
    head_dim_v: int = 128
    element_size: int = 2
    words = 3
    # The params are built over at least one tile and one key, so FA4's divisions never see 0.
    decodes_any_index = True

    def __post_init__(self):
        ts = _fa4_tile_scheduler()
        if self.scheduler not in (ts.StaticPersistentTileScheduler, ts.SingleTileLPTScheduler):
            raise ValueError(
                "Fa4Mapping adapts StaticPersistentTileScheduler or SingleTileLPTScheduler, "
                f"whose tile is a function of one linear index; got {self.scheduler!r}"
            )
        for name in ("head_dim", "head_dim_v", "element_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"Fa4Mapping.{name} must be a positive int, got {value!r}")
        for name in ("lpt", "lpt_global"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"Fa4Mapping.{name} must be a bool")

    @property
    def _persistent(self) -> bool:
        return self.scheduler is _fa4_tile_scheduler().StaticPersistentTileScheduler

    def bind(self, space):
        rank = 3 if self._persistent else 4
        if len(space) != rank:
            what = "(num_block, num_head, num_batch" + ("" if rank == 3 else ", seqlen_k") + ")"
            raise ValueError(f"{self.describe()} space is {what}, got {space!r}")
        extents = Raster().bind(space[:3])  # an empty space is (1, 1, 0)
        num_block, num_head, num_batch = extents
        # Safe FA4 parameters for empty spaces; count64 still reports zero items.
        seqlen_k = Int32(0) if self._persistent else cutlass.max(Int32(space[3]), 1)
        args = _fa4_tile_scheduler().TileSchedulerArguments(
            num_block,
            num_head,
            cutlass.max(num_batch, 1),
            Int32(1),  # num_splits
            seqlen_k,
            self.head_dim,
            self.head_dim_v,
            Int32(0),  # total_q, varlen only
            (1, 1),  # tile_shape_mn, unused by these schedulers
            element_size=self.element_size,
            lpt=self.lpt,
            lpt_global=self.lpt_global,
        )
        return self.scheduler.to_underlying_arguments(args), extents

    def count64(self, space):
        # With a (1, 1) cluster and no splits, both FA4 counts equal the Raster count:
        # num_block * num_head * num_batch, exact in Int64 and zero for empty spaces.
        _, extents = space
        return Raster().count64(extents)

    def decode(self, index, space, state):
        params, _ = space
        if self._persistent:
            scheduler = self.scheduler(params, Int32(index))
        else:
            scheduler = self.scheduler(params, Int32(index), Int32(0))
        return True, tuple(scheduler.get_current_work().tile_idx[:3]), state

    def get_key(self) -> str:
        if self._persistent:
            return "fa4sp"
        flags = ("lpt" if self.lpt else "") + ("global" if self.lpt_global else "")
        sizes = f"{self.head_dim}_{self.head_dim_v}_{self.element_size}"
        return f"fa4single{flags}_{sizes}"

    def describe(self) -> str:
        return f"Fa4Mapping({self.scheduler.__name__})"
