"""External tile schedulers as warpdrive work mappings.

An adapter reuses an external scheduler's ``index -> tile`` mapping and leaves distribution to
the warpdrive index schedule, so it swaps in with one line and runs under every schedule::

    self.work = WorkStream(mapping=CutlassRaster(swizzle=8), schedule=StaticPersistent())
    self.work = WorkStream(mapping=CutlassRaster(swizzle=8), schedule=ClusterLaunchControl())

Composing the adapter with ``StaticPersistent()`` (``Fa4Mapping`` of ``SingleTileLPTScheduler``:
``OneShot()``) reproduces the external scheduler's own per-CTA index sequence, not its
lifecycle. The kernel keeps binding the same ``(m_tiles, n_tiles, l)`` space; the adapter derives
the external space (FA4: its ``Params``) from it in the entry point.

What carries over and what does not:

* Only *index-addressable* schedulers adapt: those whose state is one linear work index and
  whose current tile is a pure function of it. CUTLASS's ``StaticPersistentTileScheduler``
  (its order, reimplemented from the BSD-3 CuTeDSL helper) and FA4's
  ``StaticPersistentTileScheduler`` and ``SingleTileLPTScheduler`` (``STATIC`` mode) are.
  FA4's ``SingleTileScheduler`` reads a 3-D ``blockIdx`` and is rejected; QuACK's and the
  CLC/dynamic schedulers own their own pipelines, SMEM and barriers, which would duplicate the
  layer's distribution.
* The external ``initial_work_tile_info``/``advance_to_next_work``/``prefetch_next_work``/
  ``producer_tail`` and ``get_grid_shape`` are not called: the warpdrive schedule picks indices
  and launches its own grid.
* Indices count warpdrive work units, so a ``CtaPair`` maps one item per cluster: tiles are
  work-unit tiles (FA4's params use a ``(1, 1)`` cluster shape).
* Empty spaces (an extent <= 0) have no items. FA4's params are then built over one tile (and
  an empty KV as one key), so no external arithmetic divides by zero; the item count comes from
  the normalized extents and is 0.
* Both adapters have an exact Int64 count (``count64``), so a space past ``2**31`` items is
  rejected at launch like a ``Raster`` space.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import cutlass
from cutlass import Int32, Int64

from attn_gym.warpdrive.scheduling import Raster, WorkMap


def _round_up(extent, multiple: int):
    """``extent`` rounded up to ``multiple``, in Int64 for DSL values (never wraps Int32)."""
    if isinstance(extent, int):
        return -(-extent // multiple) * multiple
    return (Int64(extent) + (multiple - 1)) // multiple * multiple


@dataclass(frozen=True)
class CutlassRaster(WorkMap):
    """The tile order of CUTLASS's ``StaticPersistentTileScheduler`` over
    ``space = (m_tiles, n_tiles, l)``: ``along`` is the raster axis (CUTLASS ``raster_along_m``)
    and ``swizzle`` the band width in tiles (CUTLASS ``swizzle_size``).

    Reimplemented from the BSD-3 CuTeDSL helper (NVIDIA/cutlass
    ``examples/python/CuTeDSL/helpers/static_persistent_tile_scheduler.py`` at ``e13f5cd``, which
    replaces the deprecated ``cutlass.utils`` copy): its problem layout pads the banded axis to a
    multiple of ``swizzle``, and on that padded space its order is exactly ``Raster(along,
    swizzle)``, whose bands are then all full. CUTLASS hands the padded tiles to the kernel; this
    adapter drops them (``keep = False``), so roles only see tiles inside the space. The tests
    check the order against the helper itself. ``bind`` normalizes extents like ``Raster`` (an
    empty space has no items), so ``decode`` is defined for any index.
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
        """``extents`` with the banded axis (the one ``along`` does not name) rounded up to a
        multiple of ``swizzle``."""
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
        if self.swizzle == 1:  # no padding: every tile is inside the space
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
    """The tile mapping of a FlashAttention 4 scheduler class from
    ``flash_attn.cute.tile_scheduler``: ``StaticPersistentTileScheduler`` or
    ``SingleTileLPTScheduler`` (``STATIC`` mode). Items are ``(block, head, batch)``.

    ``space`` is ``(num_block, num_head, num_batch)``; ``SingleTileLPTScheduler`` also takes
    ``seqlen_k`` as a fourth entry, since it sizes its L2 sections from the K/V bytes per head.
    Extents are normalized like ``Raster``. ``num_block`` counts work-unit blocks (one per
    cluster of a ``CtaPair``): the FA4 params are built with a ``(1, 1)`` cluster shape. Its
    compile-time arguments are fields here, so ``get_key`` names them: ``lpt`` (reverse block
    order), ``lpt_global`` (one L2 section), ``head_dim``/``head_dim_v`` and ``element_size``
    (the K/V bytes per key). ``StaticPersistentTileScheduler`` ignores them. No split-KV.
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
        # An empty space becomes one tile and an empty KV one key (``decodes_any_index``).
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
        # With a (1, 1) cluster and no splits both FA4 counts (``total_blocks_cluster``,
        # ``total_blocks``) are num_block * num_head * num_batch: the Raster count of the
        # normalized extents, exact in Int64 and 0 for an empty space.
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
