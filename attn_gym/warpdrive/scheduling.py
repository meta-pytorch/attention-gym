"""Work scheduling for warp-role kernels: which CTA runs which work item, in what order.

A stream makes two independent compile-time choices, from which the layer derives the launch
grid and the distribution:

* **schedule** (an ``IndexSchedule``): how a CTA gets its next *linear work index*. Schedules
  know nothing about work items.

  ==================================  ======================================  =====================
  schedule                            next index                              launch grid
  ==================================  ======================================  =====================
  ``OneShot()``                       none (one item per CTA)                 ``total`` (see below)
  ``StaticPersistent()``              ``index + gridDim.x``                   ``min(total, max)``
  ``TilePersistent(axis)``            ``coordinate + gridDim[axis]``          tiles, axis capped
  ``AtomicPersistent(stages)``        ``gridDim.x + atomicAdd(counter, 1)``   ``min(total, max)``
  ``ClusterLaunchControl(stages)``    ``clusterlaunchcontrol.try_cancel``     ``total``
  ==================================  ======================================  =====================

  (``max`` is ``max_ctas``.)
* **mapping** (a ``WorkMap``): ``index -> work item``, a tuple of Int32 words that ``unpack()``
  turns into whatever the roles consume. ``Raster`` is the tile mapping: row/column raster and
  L2 swizzle bands. The mapping owns its runtime ``space``, the item count (``count64``), and
  ``decode``, which returns ``(keep, words)``; a mapping that declares ``skips`` may drop empty
  items.
* **launch grid**: ``OneShot`` with the default ``Raster()`` (along m, no swizzle) launches the
  tile space itself, ``(m_tiles, n_tiles, l)`` CTAs (times the cluster size along x), and reads
  the item from ``blockIdx`` with no division. When ``n_tiles`` or ``l`` exceeds the 65535
  limit of ``gridDim.y``/``z`` it falls back to a 1-D grid: the kernel sees
  ``gridDim.y = gridDim.z = 1`` with more than one ``(n, l)`` pair and decodes the index in a
  prologue branch that tile-grid launches skip. A hand-built ``(total, 1, 1)`` grid is also
  correct, just not division-free. That is ``OneShot(grid="auto")``, the default. Its branch
  merges ``blockIdx`` with a decoded index, which takes every value derived from the item off
  the uniform datapath, so the grid can be a static choice instead: ``OneShot(grid="3d")``
  compiles the tile grid without the branch, ``"2d"`` the grid ``(m_tiles * l, n_tiles)``
  for an ``l`` past the limit, and ``"1d"`` the 1-D grid alone; ``OneShot.grid_for(space)``
  picks among them on the host. Every other combination launches a 1-D grid and decodes the
  index.
  ``BoundWork.grid`` covers empty spaces and spaces past Int32.
* **distribution**, derived from the schedule. Index sequences every role can compute
  (``OneShot``, ``StaticPersistent``, ``TilePersistent``) are decoded locally in each role
  with no synchronization.
  Claimed indices (``AtomicPersistent``, ``ClusterLaunchControl``) are claimed by a layer-owned
  scheduler warp, decoded once, and broadcast as ``[index, item words]`` over a ``Ring`` whose
  consumers are every role that takes a ``WorkCursor``. The layer derives the ring's arrival
  counts, payload and fences, and the scheduler's warp (the lowest warp no role claims).

Wherever decoding happens, the stream first runs its readiness hook once (``ready=PdlWait()``)
and steps past items the mapping drops, so every role sees the same item sequence. Only valid
indices are decoded, except by a ``WorkMap.decodes_any_index`` mapping.

A kernel declares ``self.work = WorkStream(mapping=Raster(swizzle=8),
schedule=ClusterLaunchControl())``; roles take ``work: WorkCursor`` and loop::

    while work.valid:
        m, n, l = work.item
        ...
        work.advance()

Tracing checks that each role calls ``advance()`` exactly once, directly in the body of a
``while work.valid:`` loop (not under a branch or nested loop), so every role steps through
the same sequence. A ``StaticPersistent`` or ``TilePersistent`` stream may instead be looped as
``for k in cutlass.range(work.trip_count(), unroll=n): item = work.item_at(k)``, which
``cutlass.range`` can unroll (``WorkCursor.trip_count``).

Clusters (``CtaPair``, ``Cluster``): a work item belongs to a cluster of ``cs`` CTAs. Indices
count clusters (``blockIdx.x // cs``, ``gridDim.x // cs``), and the grid launches ``cs``
consecutive CTAs per cluster (launch cluster ``(cs, 1, 1)``). Local schedules compute the same
sequence in every CTA. Broadcast schedules claim on cluster rank 0 only (a pair's leader): its
scheduler warp publishes each decoded item into the work-ring slot of every CTA with
``st.async``, whose bytes complete each CTA's full barrier, and the consumers in every CTA
release rank 0's empty barrier. CLC cancels whole clusters, so its response decodes to
``ctaid.x // cs``.

Counter lifetime (``AtomicPersistent``): the workspace holds ``[claims, finished]``. Each work
unit's scheduler reports once after its last claim, and the last one to report zeroes both
words. Launches ordered on one stream, including CUDA-graph replays, therefore reuse one
workspace with no host reset. Concurrently running launches must not share a workspace.

Custom schedules subclass the public protocols the built-ins use: ``IndexSchedule`` (next
index), ``WorkMap`` (index -> ``(keep, item)``) and ``Readiness`` (a wait before the first
decode). Their docstrings give each method's contract and whether it runs on the host or the
device. External tile orders also subclass ``WorkMap``; see ``examples/external_maps.py``.
"""

from __future__ import annotations

import functools
import inspect
import itertools
import math
import re
from dataclasses import dataclass, field, fields, is_dataclass

import cutlass
from cutlass import Int32, Int64, Uint32, cute

from attn_gym.warpdrive.channels import Ring, Slots, Threads, Tma
from attn_gym.warpdrive.compat import store_async_dsmem
from attn_gym.warpdrive.protocol import (
    Lineage,
    ProtocolError,
    RoleTrace,
    current_block,
    user_site,
)
from attn_gym.warpdrive.tracing import push_if

AXES = ("m", "n", "l")
INT32_MAX = 2**31 - 1
MAX_GRID_YZ = 65535  # gridDim.y and gridDim.z limit


def _field_key(obj, skip: tuple[str, ...] = ()) -> str:
    """``_<name><value>`` for each dataclass field: the default compile key of a protocol object
    with constructor arguments. Override ``get_key`` for values that are not short tokens."""
    if not is_dataclass(obj):
        return ""
    return "".join(
        f"_{f.name}" + re.sub(r"[^0-9a-z]+", "_", str(getattr(obj, f.name)).lower())
        for f in fields(obj)
        if f.name not in skip
    )


# --------------------------------------------------------------------------- #
# Work mappings: linear index -> work item
# --------------------------------------------------------------------------- #


class WorkMap:
    """Public protocol for ``WorkStream(mapping=...)``: linear work index -> work item.

    Subclass it (as ``Raster`` does) and set or override:

    * ``words`` (class attribute, positive int): Int32 words per item. Broadcast schedules ship
      ``[index, *words]`` through the work ring, so keep it small.
    * ``bind(space)``, host (the ``@cute.jit`` entry point, inside ``WorkStream.bind``):
      normalize the runtime space and return the bound space. That can be any DSL value
      ``extract_mlir_values`` accepts (tuples of ``Int32``, tensors, CuTeDSL params objects);
      it is passed to the kernel and to every method below. ``bind`` must make an empty space
      safe (``Raster`` stores ``(1, 1, 0)``).
    * ``count64(space)``, host: the item count as an Int64, or a Python int for static spaces;
      items are indices ``[0, count)``. It must be exact up to ``2**31``, may saturate there,
      and must not wrap. The layer calls it once, in the entry point, and narrows it to the
      Int32 ``BoundWork.total`` the kernel receives. The launch rejects a count it cannot index
      in Int32 (``BoundWork.grid``), so narrowing never truncates a launched count. An empty
      space counts 0 and runs no items.
    * ``start(space)``, device: the decode state of a decoding site (each role in local mode,
      the scheduler warp in broadcast mode, ``single_item``), made once at its startup after
      the readiness hook, even when the stream is empty. Default ``None``: a stateless mapping.
      A stateful one returns DSL values (``Int32(0)``, not ``0``) in a tuple or ``NamedTuple``;
      a plain dataclass's fields are not carried. ``start`` must be safe for every bound space,
      so read item tables in ``decode``, which is guarded. The state lives in registers of
      every decoding warp: keep it to a few words.
    * ``decode(index, space, state)``, device: returns ``(keep, words, state)``: a boolean,
      ``words`` Int32-convertible values, and the next state with the same structure and types
      (``None`` when stateless). The layer calls it only for valid indices,
      ``0 <= index < total``, after the readiness hook, so it may read per-item tables and a
      device-written length. ``keep = False`` removes the item for every role. Only a mapping
      with ``skips = True`` may return a runtime ``keep``; any other must return the constant
      ``True`` (checked at trace time). A dropped item's words are ignored. Each site passes the
      state its previous decode returned (dropped items included), or ``start``'s. The state is
      a cache, never an input to the answer: from any state reachable from ``start``,
      ``decode`` must return the ``keep`` and kept words that ``start``'s state would, for
      indices in any order (schedules jump; CLC and custom schedules may go backwards). Make it
      fast for increasing indices and correct for the rest.
    * ``decodes_any_index`` (class attribute, default ``False``): set it only when ``decode`` is
      a pure function of launch-time data, defined for every nonnegative Int32 index including
      ``index >= total`` and empty spaces (no table reads, no division by a runtime value that
      may be zero). The layer may then decode speculatively, past the end or before readiness
      on the tile grid, and discard the result; that saves a branch per item on the decoding
      warps. A skipping mapping's ``keep`` is still used only for valid indices, and a
      speculative decode's new state must still be a valid state. A subclass that overrides
      ``decode`` does not inherit the flag.
    * ``unpack(words)``, device: what ``work.item`` returns; by default the tuple of words.
    * ``get_key()``, host: compile-key fragment naming every compile-time choice, never runtime
      extents, in lowercase letters, digits and underscores. ``describe()`` (optional): text
      for ``describe()``.

    ``decode``'s item must be a pure function of the index and space, plus tables written
    before the launch or before ``ready`` returns, so every role that decodes an index gets the
    same item whatever its state; ``keep`` must agree in every role and CTA. A device-built
    work list binds a host-visible capacity as the count and reads its true length in
    ``decode`` (``examples/packed_work.py``); the host never reads device memory.
    """

    words: int = 1
    skips = False
    decodes_any_index = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "decode" in vars(cls):
            # A replacement decoder must opt into speculative decoding itself.
            if "decodes_any_index" not in vars(cls):
                cls.decodes_any_index = False
            params = inspect.signature(cls.decode).parameters
            if len(params) != 4 and not any(p.kind is p.VAR_POSITIONAL for p in params.values()):
                raise TypeError(
                    f"{cls.__name__}.decode takes (self, index, space, state) and returns "
                    "(keep, words, state); a stateless mapping passes state through (None)"
                )

    def bind(self, space):
        return space

    def count64(self, space):
        raise NotImplementedError

    def start(self, space):
        return None

    def decode(self, index, space, state):
        raise NotImplementedError

    def unpack(self, words):
        return tuple(words)

    def get_key(self) -> str:
        raise NotImplementedError

    def describe(self) -> str:
        return type(self).__name__


@dataclass(frozen=True)
class Raster(WorkMap):
    """Tile mapping over ``space = (m_tiles, n_tiles, l)``; ``l`` is always the slowest axis.

    ``along``: the axis consecutive indices advance along ("m": m fastest, like a 2-D launch
    grid with ``blockIdx.x`` over m). ``swizzle``: cut the other axis into bands of this many
    tiles and finish one band before the next. Inside a band its tiles are innermost, so
    ``k * swizzle`` consecutive indices touch ``swizzle`` operand panels on the banded axis
    instead of ``k * swizzle`` (CUTLASS raster + swizzle). The last band may be narrower.

    ``decode`` uses only ``+ - * // %`` and ``min``, so it runs on Python ints (tests) and on
    device ``Int32`` values alike. Its intermediates are nonnegative and bounded by
    ``max(index, item count)``, so it cannot overflow for any index of a space the launch
    admits (``BoundWork.grid``), and it may run on unsigned values (``OneShot(grid="1d")``).

    ``bind`` treats a negative extent as zero and maps an empty space to ``(1, 1, 0)``: no
    items, and every divisor in ``decode`` is positive. ``decode`` is therefore defined for any
    index (``decodes_any_index``) and keeps every item. ``bind`` runs in the host part of the
    jit entry point, so kernels pay nothing for it.

    A static (Python int) ``m_tiles`` or ``n_tiles`` stays a compile-time constant when other
    extents are runtime values, so ``decode`` and the ``OneShot(grid="2d")`` unfold divide
    by a constant. ``l`` is never a divisor; with any runtime extent it is bound as a runtime
    value, which a runtime-empty space clamps to 0 (and its runtime tile extents to 1).
    """

    along: str = "m"
    swizzle: int = 1
    words = 3
    decodes_any_index = True

    def __post_init__(self):
        if self.along not in ("m", "n"):
            raise ValueError(f"Raster.along must be 'm' or 'n', got {self.along!r}")
        if type(self.swizzle) is not int or self.swizzle < 1:
            raise ValueError(f"Raster.swizzle must be a positive int, got {self.swizzle!r}")

    def bind(self, space):
        if len(space) != 3:
            raise ValueError(f"Raster space is (m_tiles, n_tiles, l), got {space!r}")
        static = [e for e in space if isinstance(e, int)]
        if static and min(static) <= 0:  # statically empty: the same rule on the host
            return (1, 1, 0)
        if len(static) == 3:
            return tuple(min(e, INT32_MAX) for e in space)
        # Static tile extents stay constants (positive, checked above). l is never a divisor,
        # so it is always a runtime value that can carry the emptiness of the space.
        *tiles, el = (min(e, INT32_MAX) if isinstance(e, int) else Int32(e) for e in space)
        el = Int32(el)
        runtime = [e for e in tiles if not isinstance(e, int)] + [el]
        # INT32_MAX when every runtime extent is positive, else 0 (clamping the runtime tile
        # extents to 1 and l to 0, so the space has no items).
        cap = cutlass.max(cutlass.min(functools.reduce(cutlass.min, runtime), 1), 0) * INT32_MAX
        em, en = (e if isinstance(e, int) else cutlass.max(cutlass.min(e, cap), 1) for e in tiles)
        return em, en, cutlass.max(cutlass.min(el, cap), 0)

    def count64(self, space):
        # Saturates at 2**31 so neither this nor the launch check's arithmetic wraps Int64.
        em, en, el = space
        limit = INT32_MAX + 1
        if all(isinstance(e, int) for e in space):  # static extents: plain host arithmetic
            return min(min(em * en, limit) * el, limit)
        return cutlass.min(cutlass.min(Int64(em) * en, limit) * el, limit)

    def decode(self, index, space, state):
        em, en, _ = space
        per_l = em * en
        li = index // per_l
        r = index - li * per_l
        e_along, e_band = (em, en) if self.along == "m" else (en, em)
        if self.swizzle == 1:
            b = r // e_along
            a = r - b * e_along
        else:
            # A band never holds more than the whole layer, so this stays within the count.
            band_tiles = cutlass.min(self.swizzle, e_band) * e_along
            start = (r // band_tiles) * self.swizzle
            width = cutlass.min(self.swizzle, e_band - start)
            local = r - start * e_along
            a = local // width
            b = start + (local - a * width)
        m, n = (a, b) if self.along == "m" else (b, a)
        return True, (m, n, li), state

    def get_key(self) -> str:
        return f"{self.along}{self.swizzle}"

    def describe(self) -> str:
        parts = [f"along={self.along}"]
        if self.swizzle > 1:
            parts.append(f"swizzle={self.swizzle}")
        return f"Raster({', '.join(parts)})"


# --------------------------------------------------------------------------- #
# Readiness hooks
# --------------------------------------------------------------------------- #


class Readiness:
    """Public protocol for ``WorkStream(ready=...)``.

    ``wait()`` runs on the device once per decoding site (each role in local mode, the scheduler
    warp in broadcast mode), before the first ``decode``, and must return in every warp that
    calls it. ``get_key()`` (host) names the hook in the compile key.
    """

    def wait(self):
        raise NotImplementedError

    def get_key(self) -> str:
        return type(self).__name__.lower() + _field_key(self)


@dataclass(frozen=True)
class PdlWait(Readiness):
    """Readiness hook: ``griddepcontrol.wait`` before the first mapping-dependent read, for
    mappings that read tables written by the preceding kernel under programmatic dependent
    launch."""

    def wait(self):
        cute.arch.griddepcontrol_wait()

    def get_key(self) -> str:
        return "pdl"


# --------------------------------------------------------------------------- #
# Device-read limits
# --------------------------------------------------------------------------- #


class Limit:
    """Public protocol for ``WorkStream.bind(..., limit=...)``: an item bound read on the device.

    A launch sized on the host for a capacity (a CUDA graph captured for the most chunks or
    sequences a replay may hold) can end its stream at a count only the device knows, such as
    the last entry of a prefix-sum table. ``read()`` runs on the device once per decoding site
    (each role of a local schedule, ``single_item``, a plain kernel's cursor), after the
    readiness hook and before the first decode, and returns an Int32. Every site ends its
    stream at ``min(total, max(read(), 0))`` in the schedule's index order: indices at or past it
    are terminal, are never decoded, and ``single_item()`` reports them invalid. A limit of 0 (or
    below) runs no item; one at or past ``total`` changes nothing.

    Roles read the limit separately, so ``read()`` must return the same value in every role
    and CTA: it may read only memory written before the launch, or before ``ready`` returns,
    and not written during the launch. It must not depend on the role or thread.

    A limit is a DSL value passed into the kernel with the bound stream: implement
    ``__extract_mlir_values__``/``__new_from_mlir_values__`` for the tensors or scalars it holds
    (``LastEntry`` shows the pattern). It is bound in the jit entry point, so it is not part of
    ``WorkStream.get_key()``: a kernel that binds a limit only for some configurations must name
    them in its own compile key, as it does for any other choice of its entry point. Only local
    schedules take a limit: a broadcast schedule's scheduler warp claims indices for every role,
    so it has no per-role bound to apply. ``WorkStream.assignment`` lists the host-side sequence
    and ignores limits.
    """

    def read(self):
        raise NotImplementedError


class LastEntry(Limit):
    """``Limit`` reading ``scale * table[size - 1]``: the terminal entry of a prefix-sum table
    (``chunk_offsets[-1]``, the active chunk count; ``cu_seqlens[-1]``, the active token count).
    ``scale``: static items per counted unit. The table is Int32 with at least one entry, and
    ``scale * table[-1]`` must fit Int32."""

    def __init__(self, table: cute.Tensor, scale: int = 1):
        if type(scale) is not int or scale < 1:
            raise ValueError(f"LastEntry.scale must be a positive int, got {scale!r}")
        self.table, self.scale = table, scale

    def read(self):
        last = self.table[Int32(cute.size(self.table)) - 1]
        return Int32(last) * self.scale

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.table)

    def __new_from_mlir_values__(self, values):
        return LastEntry(cutlass.new_from_mlir_values(self.table, values), self.scale)


# --------------------------------------------------------------------------- #
# Index schedules: how a CTA gets its next linear index
# --------------------------------------------------------------------------- #


class IndexSchedule:
    """Public protocol for ``WorkStream(schedule=...)``: how a work unit gets its next linear
    index.

    A work unit is a CTA or a cluster. Schedules never see work items; they return indices the
    mapping decodes. Every returned index is valid, in ``[0, problem.total)``, or terminal,
    ``>= problem.total``; never negative. Once a unit returns a terminal index its stream has
    ended.

    Class attributes:

    * ``persistent``: a unit may run more than one item, so every managed role must loop. With
      ``persistent = False`` a unit runs at most one item: ``next_index`` must return a terminal
      index (the cursor keeps the first item's words), and raw roles may use ``single_item()``.
    * ``broadcast``: ``False`` computes indices locally in every role with
      ``first_index``/``next_index``. ``True`` claims them with ``claim_next`` on a layer-owned
      scheduler warp, on cluster rank 0 only, and broadcasts the decoded items. The layer owns
      the ring, its counts, the drain, and publication to every CTA of a cluster.
    * ``stages``: work-ring depth of a broadcast schedule, the decoded items buffered ahead of
      the slowest role; a positive int.
    * ``uses_max_ctas``: ``launch_units`` reads the ``max_ctas`` cap, so ``BoundWork.grid``
      requires one.
    * ``uses_workspace``: device code reads ``problem.workspace``, the int32 tensor from
      ``launch_args``.

    Host methods, run in the ``@cute.jit`` entry point (``total`` is the item count):

    * ``launch_units(total, max_units)``: launched work units, at least 1. ``max_units`` is the
      cap in work units, or ``None`` without ``uses_max_ctas``.
    * ``index_limit(total, units, cs)``: the largest grid x extent or index the launch computes,
      from the Int64 ``total`` (``WorkMap.count64``). When it exceeds Int32, ``BoundWork.grid``
      launches a zero-wide grid, which CUDA rejects. Override it when the default
      ``total + units - 1`` (a strided terminal index) does not bound your schedule.
    * ``get_key()``: compile-key fragment naming every compile-time field, in lowercase letters,
      digits and underscores. The default, the lowercased class name, ``stages`` and each other
      dataclass field as ``_<name><value>``, needs overriding when a value is not a short
      token. Also ``describe()``.
    * ``fetch_channels()`` (broadcast): ``{name: Channel}`` of extra layer-owned channels whose
      producer and consumer ends the scheduler warp both hold (CLC's response slot).

    Device methods (``problem`` is the kernel's ``BoundWork``: ``total``, ``space``,
    ``workspace``, ``cs``, ``units()``, ``first_unit()``):

    * ``first_index(problem)``: this unit's first index. It runs in every role and the
      scheduler, so it may depend only on the launch (``blockIdx``, ``gridDim``) and
      ``problem``. Default: ``problem.first_unit()``.
    * ``next_index(index, problem)`` (local): the index after a valid ``index``; the same pure
      function in every role.
    * ``claim_next(fetch, problem, index)`` (broadcast): claim the next index on the scheduler
      warp. ``index`` is the unit's previous index, valid or skipped; ``fetch`` maps each
      ``fetch_channels`` name to its ``(producer end, consumer end)``. Called only when a ring
      slot is free and the previous claim was valid.
    * ``close_claims(fetch, problem)`` (broadcast): runs once, after the unit's last claim, and
      closes claim state only (``AtomicPersistent`` reports and resets its counter). Roles may
      still be computing earlier items then, because ring consumers release a slot on receipt,
      so it is not a teardown, fixup or reduction point.

    All values are Int32; ``index_limit`` rejects extents past Int32 at launch.
    """

    broadcast = False
    persistent = True
    stages = 0
    uses_max_ctas = True
    uses_workspace = False

    def launch_units(self, total, max_units):
        if max_units is None:  # uses_max_ctas = False: one unit per item
            return cutlass.max(total, 1)
        return cutlass.max(cutlass.min(total, max_units), 1)

    def index_limit(self, total, units, cs: int):
        """Largest launch-grid x extent or linear index the launch computes, from the Int64 item
        count ``total`` and ``units`` launched work units. The last unit's terminal index
        overshoots ``total`` by less than the stride, ``units``."""
        return total + units - 1

    def get_key(self) -> str:
        return f"{type(self).__name__.lower()}{self.stages}" + _field_key(self, skip=("stages",))

    def describe(self) -> str:
        return type(self).__name__ + (f"(stages={self.stages})" if self.broadcast else "()")

    def fetch_channels(self) -> dict:
        """Extra layer-owned channels the scheduler warp holds both ends of."""
        return {}

    def first_index(self, problem):
        return problem.first_unit()

    def next_index(self, index, problem):
        raise NotImplementedError

    def claim_next(self, fetch, problem, index):
        raise NotImplementedError

    def close_claims(self, fetch, problem):
        pass


@dataclass(frozen=True)
class OneShot(IndexSchedule):
    """One item per CTA, ``grid = total`` (the classic non-persistent launch).

    ``grid`` chooses, at compile time, the launch grid of a plain ``Raster()`` stream:

    * ``"auto"``: the tile space, or the 1-D grid when ``n_tiles`` or ``l`` exceeds 65535. One
      kernel serves both through a prologue branch on the launch shape. Any other mapping
      launches the 1-D grid.
    * ``"3d"``: the tile space only, the item read from ``blockIdx`` with no branch. Needs
      ``n_tiles`` and ``l`` of at most 65535.
    * ``"2d"``: the grid ``(m_tiles * l, n_tiles, 1)``, for an ``l`` past 65535: ``l`` is
      folded into ``blockIdx.x`` and recovered with one division by ``m_tiles``, and ``n`` is
      ``blockIdx.y``. Needs ``n_tiles`` of at most 65535.
    * ``"1d"``: the 1-D grid only, every index decoded (two divisions for ``Raster()``), for
      any ``Raster``. Other mappings always launch the 1-D grid, under ``"auto"``.

    ``"3d"`` and ``"2d"`` need plain ``Raster()`` and a launch on ``BoundWork.grid``. A space
    past their limit cannot launch: static extents raise a ``ValueError`` from
    ``BoundWork.grid``; runtime extents cannot raise from the jit entry point, so CUDA rejects
    the grid before any CTA runs (``CUDA launch failed: cudaErrorInvalidValue (1)``, ``CUDA
    rejected the launch configuration before running the kernel``).

    A wrapper that must cover every space selects the variant with
    ``OneShot(grid=OneShot.grid_for(space))``: at most three compiled kernels, and two when one
    of ``n_tiles`` and ``l`` is bounded by the model.
    """

    grid: str = "auto"
    persistent = False
    uses_max_ctas = False

    def __post_init__(self):
        if self.grid not in ("auto", "3d", "2d", "1d"):
            raise ValueError(f"OneShot.grid must be 'auto', '3d', '2d' or '1d', got {self.grid!r}")

    @staticmethod
    def grid_for(space) -> str:
        """Host: the cheapest static ``grid`` that launches ``space = (m_tiles, n_tiles, l)``,
        given as Python ints: ``"3d"`` when ``n_tiles`` and ``l`` fit ``gridDim.y``/``z``,
        ``"2d"`` when only ``n_tiles`` does, else ``"1d"``."""
        _, n_tiles, layers = space
        if n_tiles > MAX_GRID_YZ:
            return "1d"
        return "3d" if layers <= MAX_GRID_YZ else "2d"

    def get_key(self) -> str:
        key = super().get_key()  # names the class, and the grid as a field
        return key.removesuffix("_gridauto")  # the default grid adds no suffix

    def describe(self) -> str:
        grid = "" if self.grid == "auto" else f"grid={self.grid!r}"
        return f"{type(self).__name__}({grid})"

    def index_limit(self, total, units, cs: int):
        return total * cs  # the 1-D grid; the terminal index is ``total``

    def next_index(self, index, problem):
        return problem.total


@dataclass(frozen=True)
class StaticPersistent(IndexSchedule):
    """``min(total, max_ctas)`` CTAs; CTA ``b`` runs ``b, b + grid, b + 2 grid, ...``."""

    def next_index(self, index, problem):
        return index + problem.units()


@dataclass(frozen=True)
class TilePersistent(IndexSchedule):
    """Persistent along one ``axis`` of a plain ``Raster()`` tile space, the other two fixed per
    CTA.

    The launch grid is the tile space with ``axis`` capped at ``max_ctas`` units:
    ``(min(m_tiles, max_ctas), n_tiles, l)`` for ``axis="m"``, and likewise for ``"n"`` (on
    ``gridDim.y``) and ``"l"`` (``gridDim.z``). A CTA reads the two other coordinates from
    ``blockIdx`` with no division, so address arithmetic on them keeps compile-time strides, and
    walks its axis coordinate ``c = blockIdx[axis], c + gridDim[axis], ...``. ``max_ctas``
    therefore caps the units along ``axis`` (per slice of the other two axes), not the total
    CTA count; it is a runtime value, so changing it does not recompile.

    ``units`` (a positive int) instead fixes the axis extent of the grid at compile time and makes
    the stride that constant; CTAs past the axis extent run no item, and ``max_ctas`` is not read.
    A compile-time stride lets the loop step by an immediate. It specializes the kernel on
    ``units``, so bucket it (a power of two capped at the SM count, say).

    The stream's index is that axis coordinate, and ``total`` is the axis extent (0 for an empty
    space), so a ``Limit`` bounds the coordinate. Items are decoded without division: the words
    are ``blockIdx`` with the axis coordinate replaced. Needs the plain ``Raster()`` mapping, a
    single-CTA cluster; ``n_tiles`` and ``l`` must fit ``gridDim.y``/``z`` unless that axis is
    the capped one.
    """

    axis: str = "m"
    units: int | None = None

    def __post_init__(self):
        if self.axis not in AXES:
            raise ValueError(f"TilePersistent.axis must be one of {AXES}, got {self.axis!r}")
        if self.units is not None and (type(self.units) is not int or self.units < 1):
            raise ValueError(f"TilePersistent.units must be a positive int, got {self.units!r}")
        if self.units is not None and self.axis != "m" and self.units > MAX_GRID_YZ:
            raise ValueError(
                f"TilePersistent(axis={self.axis!r}) launches units on gridDim.y/z, limited to "
                f"{MAX_GRID_YZ}; got units={self.units}"
            )

    @property
    def uses_max_ctas(self) -> bool:
        return self.units is None

    def get_key(self) -> str:
        units = "" if self.units is None else f"_units{self.units}"
        return f"tilepersistent_axis{self.axis}{units}"

    def describe(self) -> str:
        units = "" if self.units is None else f", units={self.units}"
        return f"TilePersistent(axis={self.axis!r}{units})"

    def launch_units(self, total, max_units):
        if self.units is not None:
            return self.units
        return super().launch_units(total, max_units)

    def first_index(self, problem):
        return cute.arch.block_idx()[AXES.index(self.axis)]

    def next_index(self, index, problem):
        if self.units is not None:
            return index + self.units
        return index + cute.arch.grid_dim()[AXES.index(self.axis)]


@cute.jit
def _claim_atomic(workspace, units):
    old = Int32(0)
    if cute.arch.lane_idx() == 0:
        old = cute.arch.atomic_add(workspace.iterator, Int32(1), scope="gpu")
    return cute.arch.shuffle_sync(old, 0) + units


@cute.jit
def _report_finished(workspace, units):
    # The same lane made every claim of this CTA. The first fence orders those claims before
    # the report; the last reporter's fence orders all reports before its reset, so the reset
    # is coherence-ordered after every claim of this launch.
    if cute.arch.lane_idx() == 0:
        claims = workspace.iterator
        cute.arch.fence_acq_rel_gpu()
        done = cute.arch.atomic_add(claims + 1, Int32(1), scope="gpu")
        grid = units  # one reporting scheduler per work unit (CTA or cluster leader)
        if done == grid - 1:
            cute.arch.fence_acq_rel_gpu()
            cute.arch.atomic_exch(claims, Int32(0), scope="gpu")
            cute.arch.atomic_exch(claims + 1, Int32(0), scope="gpu")
        elif done >= grid:  # workspace dirty or shared with a concurrent launch
            cute.arch.inline_ptx("trap;", read_only_args=[])


@dataclass(frozen=True)
class AtomicPersistent(IndexSchedule):
    """``min(total, max_ctas)`` CTAs; after its first item (``blockIdx.x``) a CTA claims
    ``gridDim.x + atomicAdd(workspace[0], 1)``. ``stages``: decoded items buffered ahead of the
    slowest role. Needs an int32 ``[2]`` zeroed workspace (see the module docstring)."""

    stages: int = 2
    broadcast = True
    uses_workspace = True

    def claim_next(self, fetch, problem, index):
        return _claim_atomic(problem.workspace, problem.units())

    def close_claims(self, fetch, problem):
        _report_finished(problem.workspace, problem.units())


@dataclass(frozen=True)
class ClusterLaunchControl(IndexSchedule):
    """Blackwell cluster launch control: ``grid = total``. A running CTA cancels a CTA that has
    not launched yet with ``clusterlaunchcontrol.try_cancel`` and runs its item instead, so the
    hardware balances load and yields SMs to other kernels with no counter to manage.
    ``stages``: decoded items buffered ahead of the slowest role.

    In a cluster, rank 0 cancels whole clusters; the response is not multicast, and the decoded
    item is published to every CTA. Nothing is in flight when the scheduler warp exits: every
    query was waited, and none follows a failed one (PTX forbids that), so ``close_claims`` has
    nothing to do.
    """

    stages: int = 2
    broadcast = True
    uses_max_ctas = False

    def index_limit(self, total, units, cs: int):
        return total * cs  # the grid; a failed cancel returns the terminal index ``total``

    def fetch_channels(self) -> dict:
        # The scheduler's private response slot: try_cancel completes 16 bytes on the full
        # barrier, like a TMA load. The response is read through the generic proxy and the next
        # try_cancel overwrites it through the async proxy, so the release is fenced: the
        # release_fence a Tma-filled channel derives for thread consumers.
        clc = Slots(1, Tma(16), Threads, payload=(Int32, (4,)))
        clc.internal = True  # completed by try_cancel, not by TMA copies
        return {"clc": clc}

    def claim_next(self, fetch, problem, index):
        prod, cons = fetch["clc"]
        h = prod.acquire(0)
        with cute.arch.elect_one():
            cute.arch.issue_clc_query(h.barrier, h.data.iterator, multicast=False)
        h.commit()  # TMA-like producer: completed by the response bytes
        r = cons.wait(0)
        x, _, _, ok = cute.arch.clc_response(r.data.iterator)
        r.release()
        if cutlass.const_expr(problem.cs > 1):
            x = x // problem.cs  # first CTA of the canceled cluster -> cluster index
        return x * ok + problem.total * (1 - ok)


# --------------------------------------------------------------------------- #
# Declaration and bound work
# --------------------------------------------------------------------------- #


@dataclass
class WorkStream:
    """A kernel's work declaration:
    ``self.work = WorkStream(mapping=Raster(swizzle=8), schedule=ClusterLaunchControl())``.

    ``ready``: readiness hook run once before the first decode, wherever decoding happens.
    ``scheduler_warp``: warp of a broadcast schedule's scheduler role; by default the lowest
    warp no role claims, usually a warpgroup padding warp. Local schedules normalize it to None.
    """

    mapping: WorkMap = field(default_factory=Raster)
    schedule: IndexSchedule = field(default_factory=OneShot)
    ready: Readiness | None = None
    scheduler_warp: int | None = None
    name: str = field(default="", init=False)
    # Its users in warp order, then its scheduler (set by the layer; describe() lists them).
    roles: list = field(default_factory=list, init=False)
    # The declaring kernel (set on attribute assignment); its cluster fixes CTAs per work unit.
    owner: object = field(default=None, init=False, repr=False, compare=False)
    _workspaces: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        for arg, value, protocol, names in (
            ("mapping", self.mapping, WorkMap, "WorkMap"),
            ("schedule", self.schedule, IndexSchedule, "IndexSchedule"),
            ("ready", self.ready, (Readiness, type(None)), "Readiness or None"),
        ):
            if not isinstance(value, protocol):
                raise TypeError(
                    f"WorkStream.{arg}: expected an instance of {names} (subclass the public "
                    f"protocol), got {type(value).__name__}"
                )
        schedule = type(self.schedule)
        needed = "claim_next" if self.schedule.broadcast else "next_index"
        if getattr(schedule, needed) is getattr(IndexSchedule, needed):
            mode = "broadcast" if self.schedule.broadcast else "local"
            raise TypeError(
                f"{schedule.__name__} is a {mode} schedule but does not define {needed}()"
            )
        if type(self.mapping.words) is not int or self.mapping.words < 1:
            raise ValueError("work items need at least one word (a positive int)")
        if self.scheduler_warp is not None and (
            type(self.scheduler_warp) is not int or not 0 <= self.scheduler_warp < 32
        ):
            raise ValueError(
                f"WorkStream.scheduler_warp must be an int in [0, 32), got {self.scheduler_warp!r}"
            )
        if not self.schedule.broadcast:
            self.scheduler_warp = None
        if self.schedule.broadcast and (
            type(self.schedule.stages) is not int or self.schedule.stages <= 0
        ):
            raise ValueError("broadcast schedule stages must be a positive int")
        if isinstance(self.schedule, TilePersistent) and (
            type(self.mapping) is not Raster or self.mapping != Raster()
        ):
            raise ValueError(
                f"{self.schedule.describe()} reads the item from blockIdx: it needs the plain "
                f"Raster() mapping, not {self.describe()}"
            )
        if isinstance(self.schedule, OneShot):
            grid = self.schedule.grid
            if grid in ("3d", "2d") and not _tile_grid(self):
                raise ValueError(
                    f"OneShot(grid={grid!r}) reads the item from blockIdx: it needs OneShot "
                    f"itself and the plain Raster() mapping, not {self.describe()}"
                )
            if grid == "1d" and type(self.mapping) is not Raster:
                raise ValueError(
                    "OneShot(grid='1d') chooses the grid of a Raster stream; "
                    f"{type(self.mapping).__name__} always launches the 1-D grid, so leave "
                    "grid='auto'"
                )

    @property
    def cluster_size(self) -> int:
        """CTAs per work unit: the owning kernel's cluster size (2 for ``CtaPair``)."""
        return 1 if self.owner is None else self.owner.cluster.size

    @property
    def payload_words(self) -> int:
        """Broadcast payload per stage: ``[index, item words]`` padded to 16 bytes."""
        return -(-(1 + self.mapping.words) // 4) * 4

    def channel(self) -> Ring:
        """The broadcast work ring.

        In a cluster, rank 0's scheduler fills every CTA's slot with ``st.async``, which is
        TMA-like: one armed arrival plus payload bytes per CTA. The consumers' releases then
        publish nothing, so the ring derives ``remote_scope="cta"``; a cluster-scope release per
        receiving thread is measurably slower (docs/warpdrive/examples.md).
        """
        payload = (Int32, (self.payload_words,))
        if self.cluster_size == 1:
            return Ring(self.schedule.stages, Threads, Threads, payload=payload)
        # st.async writes through the generic proxy (PTX ISA, st.async), so unlike a TMA fill
        # it needs no proxy fence after the consumers' reads.
        ring = Ring(
            self.schedule.stages,
            Tma(4 * self.payload_words),
            Threads,
            payload=payload,
            release_fence=(),
        )
        ring.internal = True  # leader-only producer filling consumer CTAs (no multicast TMA)
        return ring

    def describe(self) -> str:
        ready = f", ready={type(self.ready).__name__}" if self.ready else ""
        return f"{self.schedule.describe()}, {self.mapping.describe()}{ready}"

    def get_key(self) -> str:
        """Compile-key fragment: only compile-time choices (never extents or SM counts)."""
        key = f"{self.schedule.get_key()}_{self.mapping.get_key()}"
        ready = f"_{self.ready.get_key()}" if self.ready else ""
        return key + ready

    # ---- host, inside the @cute.jit entry point ---------------------------- #

    def bind(self, space, workspace=None, max_ctas=None, limit: Limit | None = None) -> BoundWork:
        """Bind the runtime space (``Raster``: ``(m_tiles, n_tiles, l)``). ``workspace``: the
        int32 tensor from ``launch_args``, holding ``AtomicPersistent``'s counters.
        ``max_ctas``: the runtime persistent grid cap, typically the SM count.
        ``limit``: a ``Limit`` read on the device that ends the stream early (local schedules)."""
        if workspace is None and self.schedule.uses_workspace:
            raise ValueError(f"{self.schedule.describe()} needs a workspace tensor")
        if limit is not None and not isinstance(limit, Limit):
            raise TypeError(
                f"WorkStream.bind(limit=...) takes a Limit, got {type(limit).__name__}"
            )
        if limit is not None and self.schedule.broadcast:
            raise ValueError(
                f"{self.schedule.describe()} claims indices on a scheduler warp; a Limit bounds "
                "local schedules only"
            )
        return BoundWork(self, self.mapping.bind(space), workspace, max_ctas, limit=limit)

    # ---- device, inside a raw role ------------------------------------------ #

    def single_item(self):
        """``valid, item = self.work.single_item()``: this CTA's work item, for a raw role that
        takes no ``WorkCursor``.

        Only for non-persistent local schedules (``OneShot``), where a unit has at most one
        item. It runs the readiness hook, drops items and reads the item the way the cursor
        does, so it stays correct on the 1-D fallback grid, a hand-built ``(total, 1, 1)`` grid
        (``OneShot(grid="3d")`` and ``"2d"`` need ``BoundWork.grid``), a cluster and
        custom mappings. ``valid`` is false for an empty space or a dropped item.
        Call it after ``begin_roles``.
        """
        _check_single_item(self)
        problems = getattr(self.owner, "_ws_problems", {})
        if self.name not in problems:
            raise ProtocolError(
                f"WorkStream {self.name or '?'}: call single_item() inside a role, after "
                "begin_roles()"
            )
        return _single_item(self, problems[self.name])

    # ---- host, plain Python ------------------------------------------------ #

    def launch_args(self, device, max_ctas: int | None = None):
        """``(workspace, max_ctas)`` for the compiled entry point.

        One workspace is cached per (device, current stream): launches on one stream run in
        order, so they may share it. A schedule that reads no workspace gets one placeholder per
        device, which the kernel never reads or writes.

        ``max_ctas`` defaults to the CTAs of every cluster that can be resident at once (the SM
        count for one CTA; fewer for clusters that do not tile the GPCs), so a persistent grid
        runs in one wave.
        """
        import torch

        device = torch.device(device)
        index = device.index if device.index is not None else torch.cuda.current_device()
        device = torch.device("cuda", index)  # "cuda" means the current device: cache by index
        sms = resident_ctas(index, self.cluster_size)
        words = 4
        if not self.schedule.uses_workspace:
            # The kernel never touches it: one tensor per device serves every stream, and it is
            # only zeroed outside a capture, so a CUDA graph on a new stream records no fill.
            ws = self._workspaces.get(device)
            if ws is None:
                with torch.cuda.device(index):  # capturing on that device's current stream
                    capturing = torch.cuda.is_current_stream_capturing()
                new = torch.empty if capturing else torch.zeros
                ws = self._workspaces[device] = new(words, dtype=torch.int32, device=device)
            return ws, max_ctas or sms
        stream = torch.cuda.current_stream(device).cuda_stream
        ws = self._workspaces.get((device, stream))
        if ws is None:
            ws = torch.zeros(words, dtype=torch.int32, device=device)
            self._workspaces[(device, stream)] = ws
        return ws, max_ctas or sms

    def assignment(self, space, max_ctas: int | None = None, decode=None) -> list[list[tuple]]:
        """The items each CTA runs, in order, computed on the host: ``[cta][slot]`` is
        ``(index, item)``.

        Only a local schedule fixes the sequences before the launch: ``OneShot``,
        ``StaticPersistent``, ``TilePersistent``, or a custom one whose ``first_index`` and
        ``next_index`` are plain arithmetic on ``problem``. Claimed indices depend on run-time
        order. The sequences are the schedule's own, run on a host ``problem`` whose ``units()``
        and ``first_unit()`` are ints. ``space`` and ``max_ctas`` are the launch's, as host values
        (``Raster`` takes int extents); a persistent schedule needs ``max_ctas``. CTA ``c`` is
        ``blockIdx.x`` of the 1-D launch grid, in work unit ``c // cs``, and every CTA of a unit
        runs its items. A ``OneShot`` tile-grid launch places unit ``u`` at
        ``(u % m_tiles, (u // m_tiles) % n_tiles, ...)`` instead (``grid="2d"``: at
        ``(u % m_tiles + m_tiles * l, n)``).

        ``decode(index) -> (keep, item)`` is the host counterpart of ``mapping.decode``, for a
        mapping that reads tables or device data. By default a ``decodes_any_index`` mapping
        (a pure function of the space, like ``Raster``) decodes itself and the item is its
        words; any other keeps every index with item ``None``, so a mapping that ``skips``
        needs ``decode``. Items the decode drops are left out, as the roles skip them. A space
        the launch would reject (past Int32) raises.

        ``TilePersistent`` lists its CTAs in launch order (``blockIdx.x`` fastest, then ``y``,
        ``z``); each runs the items along its axis, and ``index`` is the axis coordinate.
        """
        schedule, mapping = self.schedule, self.mapping
        if isinstance(schedule, TilePersistent):
            if decode is not None:
                raise ValueError(f"{schedule.describe()} reads items from blockIdx; drop decode")
            _check_single_cta(self, self.cluster_size)
            return _tile_persistent_assignment(mapping.bind(space), schedule, max_ctas)
        if schedule.broadcast:
            raise ValueError(
                f"{schedule.describe()} claims indices at run time; only a local schedule has a "
                "host-side assignment"
            )
        space = mapping.bind(space)
        count = int(mapping.count64(space))
        total = _narrow(count)
        cs = self.cluster_size
        units = int(schedule.launch_units(total, _unit_cap(schedule, max_ctas, cs)))
        if schedule.index_limit(count, units, cs) > INT32_MAX:
            raise ValueError(f"the space's {count} items exceed what a launch indexes in Int32")
        if decode is None and mapping.decodes_any_index:

            def decode(index):
                keep, words, _ = mapping.decode(index, space, mapping.start(space))
                return keep, tuple(int(w) for w in words)

        elif decode is None and mapping.skips:
            raise ValueError(
                f"{type(mapping).__name__} drops items by data the host cannot read; pass its "
                "host counterpart as decode=(index -> (keep, item))"
            )
        elif decode is None:

            def decode(index):
                return True, None

        rows = []
        for unit in range(units):
            problem = _HostProblem(space, total, cs, units, unit)
            items = []
            index = int(schedule.first_index(problem))
            for _ in range(total):  # a unit runs each index at most once
                if index >= total:
                    break
                keep, item = decode(index)
                if keep:
                    items.append((index, item))
                index = int(schedule.next_index(index, problem))
            rows += [list(items) for _ in range(cs)]
        return rows


def _tile_persistent_assignment(space, schedule, max_ctas) -> list[list[tuple]]:
    """``WorkStream.assignment`` of a ``TilePersistent`` schedule over a bound static space."""
    k = AXES.index(schedule.axis)
    total = space[k] if math.prod(space) > 0 else 0
    if schedule.units is None and max_ctas is None:
        raise ValueError(f"{schedule.describe()} needs max_ctas")
    units = schedule.launch_units(total, max_ctas)
    dims = _tile_persistent_dims(space, schedule.axis, units)
    rows = []
    for z, y, x in itertools.product(*(range(d) for d in reversed(dims))):  # x fastest
        block = (x, y, z)
        rows.append(
            [(c, block[:k] + (c,) + block[k + 1 :]) for c in range(block[k], total, units)]
        )
    return rows


@dataclass(frozen=True)
class _HostProblem:
    """What a local schedule reads of ``BoundWork``, for one work unit on the host
    (``WorkStream.assignment``): the launch's unit count and this unit, as ints."""

    space: object
    total: int
    cs: int
    launched: int
    unit: int
    workspace = None

    def units(self) -> int:
        return self.launched

    def first_unit(self) -> int:
        return self.unit


class BoundWork:
    """A stream bound to its runtime work space (``WorkStream.bind``), passed to the kernel as a
    DSL value: the mapping's space, the Int32 item count ``total`` and, for ``AtomicPersistent``,
    the workspace. ``count64`` (the mapping's Int64 count) and ``max_ctas`` stay on the host."""

    def __init__(self, stream: WorkStream, space, workspace, max_ctas, total=None, limit=None):
        self.stream, self.space, self.workspace, self.max_ctas = stream, space, workspace, max_ctas
        if total is None:  # host: count once; the kernel's copies carry the narrowed count
            self.count64 = stream.mapping.count64(space)
            total = _narrow(self.count64)
            if _tile_persistent(stream):  # the stream's index is the axis coordinate
                extent = space[AXES.index(stream.schedule.axis)]
                if isinstance(self.count64, int):
                    total = extent if self.count64 > 0 else 0
                else:
                    total = Int32(cutlass.select_(self.count64 > 0, Int32(extent), Int32(0)))
        self.total = total
        self.limit = limit  # a Limit, read per decoding site (``_limited``)
        self.cs = stream.cluster_size  # CTAs per work unit

    def _in_units(self, ctas):
        """A CTA count or index along x in work units (clusters of ``cs`` CTAs)."""
        return ctas if self.cs == 1 else ctas // self.cs

    def units(self):
        """Device-side: launched work units along x (CTAs or clusters), the persistent
        schedules' stride. A ``OneShot`` tile grid has more units along y and z."""
        return self._in_units(cute.arch.grid_dim()[0])

    def first_unit(self):
        """Device-side: this CTA's work unit (its cluster's index), as a linear index."""
        bidx, bidy, bidz = cute.arch.block_idx()
        unit = self._in_units(bidx)
        if _tile_grid(self.stream):
            em, en, _ = self.space
            if self.stream.schedule.grid == "2d":
                unit, bidz = _unfold_l(unit, em)
            unit = unit + em * (bidy + en * bidz)
        return unit

    def start(self):
        """Device-side, in the kernel prologue (``begin_roles``): the first item's words of a
        ``_tile_grid`` stream. Every role shares them, which keeps the 1-D fallback branch of
        ``OneShot(grid="auto")`` off the roles' paths."""
        self.first_words = None
        if _tile_persistent(self.stream):
            _check_single_cta(self.stream, self.cs)
            self.first_words = tuple(Int32(b) for b in cute.arch.block_idx())
        if _tile_grid(self.stream):
            bidx, bidy, bidz = cute.arch.block_idx()
            words = Int32(self._in_units(bidx)), Int32(bidy), Int32(bidz)
            match self.stream.schedule.grid:
                case "auto":
                    words = _grid_item(self.stream.mapping, self.space, *words)
                case "2d":
                    m, li = _unfold_l(words[0], self.space[0])
                    words = m, words[1], li
            self.first_words = words

    # ---- device, in a kernel without roles ---------------------------------- #

    def single_item(self):
        """``valid, item = bound.single_item()``: this CTA's item in a plain ``@cute.kernel``
        that received ``bound``, with the semantics of ``WorkStream.single_item``.

        A kernel without roles takes the stream's grid and mapping without
        ``begin_roles``/``run_roles``, so it pays no barrier-init fence or CTA sync.
        """
        _check_plain(self.stream, "single_item")
        _check_single_item(self.stream)
        self.start()
        return _single_item(self.stream, self)

    def cursor(self) -> WorkCursor:
        """``work = bound.cursor()``: a local-schedule cursor for a plain ``@cute.kernel``.

        Loop as a role does (``while work.valid: ...; work.advance()``). One loop runs every
        item, so there is no cross-role agreement to check: tracing does not verify the loop
        shape, and nothing needs draining.
        """
        _check_plain(self.stream, "cursor")
        if self.stream.schedule.broadcast:
            raise ProtocolError(
                f"BoundWork.cursor() needs a local schedule, not "
                f"{self.stream.schedule.describe()}: a broadcast schedule needs a scheduler "
                "warp and a work ring, so declare roles"
            )
        self.start()
        return WorkCursor.create(self.stream, self, None, RoleTrace("kernel", None))

    def _limited(self):
        """Device, at a decoding site after the readiness hook: this work with ``total`` cut
        to ``min(total, limit.read())``, or itself without a limit. The read dominates the
        site's whole stream, so the copy serves its loops without being carried."""
        if self.limit is None:
            return self
        new = BoundWork(self.stream, self.space, self.workspace, None, total=self.total)
        limit = cutlass.max(Int32(self.limit.read()), Int32(0))
        new.total = cutlass.min(Int32(self.total), limit)
        new.first_words = getattr(self, "first_words", None)
        return new

    def __extract_mlir_values__(self):
        values = cutlass.extract_mlir_values(self.space)
        if not isinstance(self.total, int):  # a static total stays a constant
            values += cutlass.extract_mlir_values(self.total)
        if self.stream.schedule.uses_workspace:
            values += cutlass.extract_mlir_values(self.workspace)
        if self.limit is not None:
            values += cutlass.extract_mlir_values(self.limit)
        return values

    def __new_from_mlir_values__(self, values):
        n = len(cutlass.extract_mlir_values(self.space))
        space = cutlass.new_from_mlir_values(self.space, values[:n])
        total = self.total
        if not isinstance(total, int):
            total, n = Int32(values[n]), n + 1
        ws = None
        if self.stream.schedule.uses_workspace:
            k = len(cutlass.extract_mlir_values(self.workspace))
            ws, n = cutlass.new_from_mlir_values(self.workspace, values[n : n + k]), n + k
        limit = None
        if self.limit is not None:
            limit = cutlass.new_from_mlir_values(self.limit, values[n:])
        new = BoundWork(self.stream, space, ws, None, total, limit)
        # Prologue values (begin_roles) dominate every later region, so they carry as is.
        new.first_words = getattr(self, "first_words", None)
        return new

    def grid(self):
        """Launch grid for this schedule: ``(ctas, 1, 1)``, or the tile space for a ``OneShot``
        over plain ``Raster()`` (``tile_grid_dims``); ``max_ctas`` caps CTAs, so a cluster of
        ``cs`` CTAs launches at most ``max_ctas // cs`` clusters. An empty space launches one
        work unit. A space whose ``schedule.index_limit`` exceeds Int32 gets a zero-wide grid,
        which CUDA rejects at launch. ``OneShot(grid="3d")`` and ``"2d"`` have no 1-D
        grid: static extents past their 65535 limit raise here, and runtime ones launch the
        grid as is, which CUDA rejects."""
        schedule = self.stream.schedule
        units = schedule.launch_units(self.total, _unit_cap(schedule, self.max_ctas, self.cs))
        if _tile_grid(self.stream):
            x, y, z = tile_grid_dims(self.space, self.cs, schedule.grid)
        elif _tile_persistent(self.stream):
            _check_single_cta(self.stream, self.cs)
            x, y, z = _tile_persistent_dims(self.space, schedule.axis, units)
        else:
            x, y, z = units * self.cs, 1, 1
        # Every grid extent and index must fit Int32. Past that the grid is zero-wide, so CUDA
        # rejects the launch instead of the kernel covering a wrapped space. This is host code
        # in the jit entry point: kernels pay nothing for the check.
        # TilePersistent indexes its axis coordinate, so the axis extent bounds its indices.
        count = self.total if _tile_persistent(self.stream) else self.count64
        if isinstance(count, int) and isinstance(units, int):  # static extents: plain host ints
            return (x if schedule.index_limit(count, units, self.cs) <= INT32_MAX else 0), y, z
        limit = schedule.index_limit(Int64(count), Int64(units), self.cs)
        return Int32(cutlass.select_(limit <= INT32_MAX, x, Int32(0))), y, z


# --------------------------------------------------------------------------- #
# Device side: decoding, the per-role cursor, and the scheduler role
# --------------------------------------------------------------------------- #


def _unit_cap(schedule, max_ctas, cs: int):
    """``launch_units``'s cap, shared by the launch and ``assignment``: ``max_ctas`` in whole
    work units of ``cs`` CTAs (at least one cluster when ``cs > 1``), or None for a schedule
    without ``uses_max_ctas``."""
    if not schedule.uses_max_ctas:
        return None
    if max_ctas is None:
        raise ValueError(f"{schedule.describe()} needs max_ctas")
    return max_ctas if cs == 1 else cutlass.max(max_ctas // cs, 1)


def _narrow(count):
    """The Int32 item count from ``count64``. A count past ``INT32_MAX`` saturates; the launch
    rejects such a space (``BoundWork.grid``), so no launched kernel sees a truncated count."""
    if isinstance(count, int):
        return min(count, INT32_MAX)
    return Int32(cutlass.min(Int64(count), Int64(INT32_MAX)))


def _state_schema(mapping, state):
    """Trace-time structure of a non-``None`` decode state: its tuple/``NamedTuple`` nesting and
    each leaf's MLIR types. Rejects the leaves the layer's flat carry would drop."""
    name = type(mapping).__name__
    if isinstance(state, tuple):
        return type(state), tuple(_state_schema(mapping, leaf) for leaf in state)
    if state is None or isinstance(state, (bool, int, float)):
        raise ProtocolError(
            f"{name} decode state has a Python value {state!r}; state leaves are DSL values "
            "(Int32(0), not 0), so the cursor can carry them"
        )
    if isinstance(state, (list, dict)):
        raise ProtocolError(
            f"{name} decode state holds a {type(state).__name__}, whose Python entries are not "
            "carried across work items; use a tuple or NamedTuple"
        )
    if is_dataclass(state):
        raise ProtocolError(
            f"{name} decode state holds a dataclass {type(state).__name__}, whose fields are not "
            "carried across work items; use a NamedTuple"
        )
    values = cutlass.extract_mlir_values(state)
    if not values:
        raise ProtocolError(f"{name} decode state holds {state!r}, which carries no DSL values")
    return type(state), tuple(str(v.type) for v in values)


def _start(mapping, space):
    """A decoding site's initial state: ``mapping.start``, checked to be carriable."""
    state = mapping.start(space)
    if state is not None:
        _state_schema(mapping, state)
    return state


def _keep_words(mapping, index, space, state):
    """``(keep, words, state)`` of ``mapping.decode``, with the words as Int32 values."""
    name = type(mapping).__name__
    result = mapping.decode(index, space, state)
    if not isinstance(result, tuple) or len(result) != 3:
        raise ProtocolError(f"{name}.decode must return (keep, words, state)")
    keep, words, next_state = result
    words = tuple(Int32(w) for w in words)
    if len(words) != mapping.words:
        raise ProtocolError(
            f"{name}.decode returned {len(words)} words; its words is {mapping.words}"
        )
    if not mapping.skips and keep is not True:
        raise ProtocolError(
            f"{name}.decode returned keep={keep!r}; a mapping that drops items "
            "sets skips = True (any other returns the constant True)"
        )
    if (state is None) != (next_state is None) or (
        state is not None and _state_schema(mapping, next_state) != _state_schema(mapping, state)
    ):
        raise ProtocolError(
            f"{name}.decode returned a state of another structure or type than it received "
            f"({next_state!r} for {state!r}); the state is carried from item to item"
        )
    return keep, words, next_state


def _decode(stream, index, problem, state):
    """``(words, state)`` of ``index``: decoded for a valid index, zeros (and ``state``) past the
    end. Only a mapping with ``decodes_any_index`` is evaluated unconditionally (no branch)."""
    mapping = stream.mapping
    if _tile_grid(stream):
        return problem.first_words, state
    if _tile_persistent(stream):  # blockIdx with the axis coordinate replaced: no division
        axis = AXES.index(stream.schedule.axis)
        words = problem.first_words
        return words[:axis] + (Int32(index),) + words[axis + 1 :], state
    if mapping.decodes_any_index:
        _, words, state = _keep_words(mapping, index, problem.space, state)
    else:
        _, words, state = _decode_kept(mapping, index, problem.total, problem.space, state)
    return words, state


@cute.jit
def _decode_kept(mapping, index, total, space, state):
    """``(dropped, words, state)`` of a valid index; ``(False, zeros, state)`` past the end,
    which is never decoded (``decode`` may read per-item tables). One decode yields all three."""
    dropped = cutlass.Boolean(False)
    words = tuple(Int32(0) for _ in range(mapping.words))
    if index < total:
        keep, kept, state = _keep_words(mapping, index, space, state)
        dropped = ~cutlass.Boolean(keep)
        # Keep words a plain assignment target: CuTeDSL instruments tuple-unpack targets
        # differently, changing the SASS of kernels whose mapping skips items.
        words = kept
    return dropped, words, state


@cute.jit
def _skip_local(stream, index, problem, state):
    """``(index, words, state)``: the first index at or after ``index`` (in this CTA's local
    sequence) the mapping keeps, or a terminal index, with the words and state its last decode
    returned."""
    step = stream.schedule.next_index  # a method call on `stream` in the loop would carry it
    mapping, total, space = stream.mapping, problem.total, problem.space
    dropped, words, state = _decode_kept(mapping, index, total, space, state)
    while dropped:
        index = Int32(step(index, problem))
        dropped, words, state = _decode_kept(mapping, index, total, space, state)
    return index, words, state


def _local_item(stream, index, problem, state):
    """``(index, words, state)`` of the first kept item at or after ``index`` in this unit's
    local sequence: ``index`` itself unless the mapping skips."""
    if stream.mapping.skips:
        return _skip_local(stream, index, problem, state)
    return index, *_decode(stream, index, problem, state)


def _local_start(stream, first, problem):
    """``(problem, index, words, state)`` a local-mode role starts with, from the unit's first
    index. The readiness hook runs before the limit read, ``start`` and any decode; the returned
    problem carries the limited total. A tile-grid stream takes the prologue's words. The cursor
    and ``single_item`` share this, so a raw role sees what a managed role would."""
    if stream.ready is not None:
        stream.ready.wait()
    problem = problem._limited()
    state = _start(stream.mapping, problem.space)
    if _tile_grid(stream) or _tile_persistent(stream):
        return problem, first, _decode(stream, first, problem, state)[0], state
    if _flat_raster(stream):
        words = _unsigned_raster_words(stream.mapping, first, problem.space)
        return problem, first, words, state
    return problem, *_local_item(stream, first, problem, state)


def _flat_raster(stream) -> bool:
    """``OneShot(grid="1d")`` over ``Raster``: every CTA decodes its one index."""
    return (
        isinstance(stream.schedule, OneShot)
        and stream.schedule.grid == "1d"
        and type(stream.mapping) is Raster
    )


def _unsigned_raster_words(mapping, index, space):
    """``Raster`` words of ``index``, decoded in unsigned arithmetic.

    An index and the bound extents are nonnegative, so the unsigned decode returns the same
    words while its divisions skip the sign handling of Int32 floor division, most of the
    decode's instructions. Only ``OneShot(grid="1d")`` decodes this way; the other schedules
    use the signed decode.
    """
    _, words, _ = mapping.decode(Uint32(index), tuple(Uint32(e) for e in space), None)
    return tuple(Int32(w) for w in words)


def _check_single_item(stream):
    if stream.schedule.persistent or stream.schedule.broadcast:
        raise ProtocolError(
            f"WorkStream {stream.name or '?'}: single_item() needs a non-persistent local "
            f"schedule (OneShot), not {stream.schedule.describe()}; code that runs several "
            "items loops on a WorkCursor"
        )


def _check_plain(stream, method: str):
    """``BoundWork``'s device methods serve kernels without roles: the roles of a
    ``WarpSpecialized`` kernel share the prologue's start values and the loop checks."""
    if stream.owner is not None:
        raise ProtocolError(
            f"BoundWork.{method}() is for kernels without roles; roles of "
            f"{type(stream.owner).__name__} take `{stream.name}: WorkCursor` or call "
            f"self.{stream.name}.single_item()"
        )


def _single_item(stream, problem):
    """``(valid, item)`` of a ``OneShot`` unit, decoded as a local cursor would."""
    first = Int32(stream.schedule.first_index(problem))
    limited = problem.limit is not None
    problem, index, words, _ = _local_start(stream, first, problem)
    valid = index < problem.total
    if _tile_grid(stream) and stream.schedule.grid != "auto" and not limited:
        # The grid holds one CTA per item, so only an empty space's one CTA has no item: a test
        # of the count alone, with no index to compute.
        valid = problem.total > 0
    return valid, stream.mapping.unpack(words)


@dataclass
class _CursorRecord:
    """Trace-time record shared by loop-carried copies of one cursor."""

    role: str
    cursor: str  # the stream's (and the role parameter's) name, for messages
    advances: list = field(default_factory=list)  # (site, block)
    valid_reads: list = field(default_factory=list)  # (block, valid SSA value identity)
    counted: list = field(default_factory=list)  # trip_count() sites: a counted loop
    loop_block: object = None  # the loop body of a counted loop's item_at() calls
    trace: object = None
    lineage: Lineage = field(default_factory=Lineage)

    def check(self):
        if self.counted:  # a counted loop: trip_count() and advance() exclude each other
            if self.loop_block is None:
                raise ProtocolError(
                    f"{self.role}: {self.cursor}.trip_count() at {self.counted[0]} is never "
                    f"looped over; read each item with {self.cursor}.item_at(k) in "
                    f"`for k in cutlass.range({self.cursor}.trip_count()):`"
                )
            return
        if not self.advances:
            raise ProtocolError(
                f"{self.role}: takes a WorkCursor but never calls advance(); loop with "
                f"`while {self.cursor}.valid: ...; {self.cursor}.advance()`"
            )
        if len(self.advances) > 1:
            sites = ", ".join(s for s, _ in self.advances)
            raise ProtocolError(
                f"{self.role}: advance() traced {len(self.advances)} times ({sites}); call it "
                "exactly once, unconditionally, at the end of the work loop body"
            )
        site, block = self.advances[0]
        op = block.owner
        conditions = [
            (cond, value)
            for cond, value in self.valid_reads
            if op.name == "scf.while"
            and block == op.regions[1].blocks[0]
            and cond == op.regions[0].blocks[0]
        ]
        if not conditions:
            raise ProtocolError(
                f"{self.role}: advance() at {site} must run once per iteration of a "
                f"`while {self.cursor}.valid:` loop: call it directly in that loop's body, not "
                "under a nested branch or loop"
            )
        if not any(hash(cond.operations[-1].operands[0]) == value for cond, value in conditions):
            raise ProtocolError(
                f"{self.role}: the work-loop condition must be exactly {self.cursor}.valid; "
                "an extra condition can leave the stream undrained"
            )


@functools.cache
def resident_ctas(index: int, cs: int) -> int:
    """CTAs of the ``cs``-CTA clusters that fit on CUDA device ``index`` at once, one CTA per
    SM.

    One CTA and CTA pairs use the SM count. Larger clusters need not tile the GPCs (a 152-SM
    GB300 fits 36 4-CTA clusters, not 38), so they ask the occupancy calculator,
    ``HardwareInfo``. It compiles and loads a probe kernel on first use: call ``launch_args``
    once before graph capture.
    """
    import torch

    sms = torch.cuda.get_device_properties(index).multi_processor_count
    if cs <= 2:
        return sms
    from cutlass.utils import HardwareInfo

    with torch.cuda.device(index):
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.synchronize()  # makes the device's primary context current
        return HardwareInfo(index).get_max_active_clusters(cs) * cs


def tile_grid_dims(space, cs: int = 1, grid: str = "auto"):
    """Launch grid of a ``_tile_grid`` stream over ``space = (m_tiles, n_tiles, l)``, for its
    ``OneShot.grid`` (``"auto"``, ``"3d"`` or ``"2d"``); one work unit when the space is
    empty (an extent <= 0). Works on Python ints and on DSL extents.

    ``"auto"``: the tile space when ``n_tiles`` and ``l`` fit ``gridDim.y``/``z``, else the 1-D
    grid of ``total`` items. ``"3d"``: the tile space. ``"2d"``:
    ``(m_tiles * l, n_tiles, 1)``. The last two have no alternative: static extents past their
    limit raise, and DSL extents are returned as they are.
    """
    em, en, el = space
    x, y, z = (em * el, en, 1) if grid == "2d" else (em, en, el)
    static = all(isinstance(e, int) for e in space)
    if static and min(space) <= 0:
        return (cs, 1, 1)
    # A static y/z extent past the limit cannot launch, whatever the other extents are.
    too_large = any(isinstance(e, int) and e > MAX_GRID_YZ for e in (y, z))
    if static and not too_large:
        return (x * cs, y, z)
    if static and grid == "auto":
        return (em * en * el * cs, 1, 1)
    if too_large and grid != "auto":
        raise ValueError(
            f"OneShot(grid={grid!r}) cannot launch the space {tuple(space)}: gridDim.y/z are "
            f"limited to {MAX_GRID_YZ}. OneShot.grid_for(space) selects a grid that fits "
            "('2d' or '1d')"
        )
    em, en, el = (Int32(e) for e in space)  # a bound space may mix static tile extents in
    nonempty = (em > 0) & (en > 0) & (el > 0)
    one = Int32(1)
    fits, x_otherwise = nonempty, one
    if grid == "auto":
        fits = nonempty & (en <= MAX_GRID_YZ) & (el <= MAX_GRID_YZ)
        x_otherwise = cutlass.select_(nonempty, em * en * el, one)
    return (
        Int32(cutlass.select_(fits, x, x_otherwise)) * cs,
        Int32(cutlass.select_(fits, Int32(y), one)),
        Int32(cutlass.select_(fits, Int32(z), one)),
    )


@cute.jit
def _unfold_l(x, em):
    """``(m, l)`` of a ``"2d"`` grid's ``x = m + m_tiles * l``. Both operands are
    nonnegative, so the division is unsigned, with no sign handling."""
    li = Int32(Uint32(x) // Uint32(em))
    return x - li * em, li


@cute.jit
def _grid_item(mapping, space, m, n, li):
    """Item words of a ``_tile_grid`` stream: ``blockIdx`` as is on a tile-space grid, or the
    decoded index on any 1-D grid (the fallback, or a hand-built ``(total, 1, 1)`` launch),
    where ``m`` is the linear index and ``n = l = 0``. The launch shape decides, not the
    extents, so both launches are correct; with ``n_tiles = l = 1`` the two grids coincide."""
    decode = mapping.decode  # a method call on `mapping` in the branch would carry it
    _, en, el = space
    _, gdy, gdz = cute.arch.grid_dim()
    if (gdy * gdz == 1) & (en * el > 1):
        _, (dm, dn, dl), _ = decode(m, space, None)
        m, n, li = Int32(dm), Int32(dn), Int32(dl)
    return m, n, li


def _tile_persistent(stream) -> bool:
    """``TilePersistent``: the tile space with one axis capped and walked by each CTA."""
    return isinstance(stream.schedule, TilePersistent)


def _check_single_cta(stream, cs: int):
    if cs != 1:
        raise ProtocolError(
            f"{stream.schedule.describe()} launches single CTAs; a cluster of {cs} CTAs needs "
            "a 1-D schedule"
        )


def _tile_persistent_dims(space, axis: str, units):
    """Launch grid of ``TilePersistent(axis)``: the tile space with ``axis`` replaced by
    ``units``, or one CTA for an empty space. Static extents past the y/z limit raise; DSL ones
    are launched as they are, which CUDA rejects."""
    k = AXES.index(axis)
    dims = list(space)
    dims[k] = units
    # A static y/z extent past the limit raises whatever the other extents are (the axis is
    # capped by its units, checked where they are chosen).
    if any(isinstance(d, int) and d > MAX_GRID_YZ for j, d in enumerate(dims[1:], 1) if j != k):
        raise ValueError(
            f"TilePersistent(axis={axis!r}) cannot launch the space {tuple(space)}: "
            f"gridDim.y/z are limited to {MAX_GRID_YZ}"
        )
    if all(isinstance(e, int) for e in (*space, units)):
        return (1, 1, 1) if min(space) <= 0 else tuple(dims)
    em, en, el = (Int32(e) for e in space)
    nonempty = (em > 0) & (en > 0) & (el > 0)
    return tuple(Int32(cutlass.select_(nonempty, Int32(d), Int32(1))) for d in dims)


def _tile_grid(stream) -> bool:
    """One item per CTA in plain raster order: launch the tile space as the grid (``l`` folded
    into x for ``"2d"``) and take the item from ``blockIdx``. Decoding a 1-D index divides
    by the extents on the load warp's path to its first copy."""
    mapping = stream.mapping
    return (
        type(stream.schedule) is OneShot  # a subclass may override first_index()
        and stream.schedule.grid != "1d"
        and type(mapping) is Raster
        and mapping.along == "m"
        and mapping.swizzle == 1
    )


def _publishes_first(stream) -> bool:
    """Whether the broadcast scheduler also decodes and publishes the first item, so ring
    receivers never decode. It does unless the mapping is a pure any-index decoder without
    hooks, in which case every role decodes ``first_index`` from launch data. Hooks (readiness,
    skipping) can move the first item off ``first_index``."""
    mapping = stream.mapping
    return stream.ready is not None or mapping.skips or not mapping.decodes_any_index


class WorkCursor:
    """One role's position in the work stream: ``valid``, ``item``, ``index``, ``advance()``.

    Also the role annotation: a parameter ``work: WorkCursor`` receives the cursor over the
    stream declared as ``self.work``. It is a DSL value carrying the linear index, the decoded
    item words, the mapping's decode state where this cursor decodes (local mode and the
    scheduler warp), and, in broadcast mode, the role's consumer end of the work ring.
    """

    def __init__(self, stream, problem, end, index, words, rec, state=None):
        self._stream, self._problem, self._end, self._rec = stream, problem, end, rec
        self.index, self._words = index, words
        self._state = state  # WorkMap.decode state of a decoding cursor (None: stateless)
        self._copy = rec.lineage.add()

    @staticmethod
    def create(stream, problem, end, trace, *, scheduler: bool = False):
        """Local mode (``end`` is None): run the hooks and decode here. Broadcast mode: decode
        ``first_index`` locally (a pure any-index mapping without hooks), or receive the first
        item from the scheduler, which publishes it in ``scheduler_role``. In broadcast mode
        only the scheduler's cursor keeps a decode state."""
        rec = _CursorRecord(trace.role_name, stream.name or "work", trace=trace)
        trace.checks.append(rec.check)
        first = Int32(stream.schedule.first_index(problem))
        cur = WorkCursor(stream, problem, end, first, (), rec)
        if end is None:
            cur._problem, cur.index, cur._words, cur._state = _local_start(stream, first, problem)
        elif not _publishes_first(stream):
            cur._words, state = _decode(
                stream, first, problem, _start(stream.mapping, problem.space)
            )
            if scheduler:
                cur._state = state
        elif scheduler:
            # scheduler_role runs the hooks and start(), then publishes and receives the first item
            # (and opens its range through _enter_item()).
            cur._words = (Int32(0),) * stream.mapping.words
            return cur
        else:
            cur._receive()
        cur._enter_item()
        return cur

    def _enter_item(self):
        """Bookkeeping at the start of each item: the item's trace range."""
        trace = self._rec.trace
        if trace.ranges is not None:  # one range per work item, named after the role
            push_if(trace.ranges, self.index < self._problem.total, trace.role_name)

    def __extract_mlir_values__(self):
        values = [self.index.ir_value()] + [w.ir_value() for w in self._words]
        if self._state is not None:
            values += cutlass.extract_mlir_values(self._state)
        if self._end is not None:
            values += self._end.__extract_mlir_values__()
        return values

    def __new_from_mlir_values__(self, values):
        n = 1 + len(self._words)
        words = tuple(Int32(v) for v in values[1:n])
        state = None
        if self._state is not None:
            k = len(cutlass.extract_mlir_values(self._state))
            state, n = cutlass.new_from_mlir_values(self._state, values[n : n + k]), n + k
        end = None if self._end is None else self._end.__new_from_mlir_values__(values[n:])
        stream = self._stream.name or "work"  # the same item in every role: see expr.py
        items = [(values[0], ("item", stream, "index"))]
        items += [(v, ("item", stream, k)) for k, v in enumerate(values[1 : 1 + len(words)])]
        self._rec.trace.item_values += items
        return WorkCursor(
            self._stream,
            self._problem,
            end,
            Int32(values[0]),
            words,
            self._rec,
            state,
        )

    def _check_current(self, returned_by: str | None = None):
        """Reject this copy if a newer copy advanced past it (``Lineage``)."""
        rec = self._rec
        rec.lineage.check(self._copy, rec.trace, rec.cursor, "cursor", returned_by)

    @property
    def valid(self):
        self._check_current()
        valid = self.index < self._problem.total
        # Store a hash, not the Value: CuTeDSL serialization would turn a raw Value in this
        # shared dataclass into loop-carried state instead of trace-only bookkeeping.
        self._rec.valid_reads.append((current_block(), hash(valid.ir_value())))
        return valid

    @property
    def item(self):
        """The current work item (``Raster``: ``(m, n, l)`` tile coordinates)."""
        self._check_current()
        block = current_block()
        while block is not None:
            if any(block == advanced for _, advanced in self._rec.advances):
                raise ProtocolError(
                    f"{self._rec.role}: {self._rec.cursor}.item read after advance() in the same "
                    "iteration "
                    f"at {user_site()}; read the current item before advancing"
                )
            owner = block.owner
            block = owner.operation.block if owner.name != "builtin.module" else None
        return self._stream.mapping.unpack(self._words)

    def advance(self):
        """Move to this CTA's next work item (``valid`` turns false when there is none)."""
        self._check_current()
        if self._rec.counted:
            raise ProtocolError(
                f"{self._rec.role}: advance() at {user_site()} on a cursor counted by "
                f"trip_count() at {self._rec.counted[0]}; loop either with advance() or over "
                "trip_count(), not both"
            )
        self._rec.lineage.advanced(self._copy)
        self._rec.advances.append((user_site(), current_block()))
        trace = self._rec.trace
        trace.item_block = current_block()
        if trace.ranges is not None:
            trace.ranges.pop(trace.role_name, self.index)  # this item's range; payload: index
        stream, problem = self._stream, self._problem
        if self._end is None:
            self.index = Int32(stream.schedule.next_index(self.index, problem))
            # A non-persistent stream has ended: keep the words, nothing past the end is decoded.
            if stream.schedule.persistent:
                self.index, self._words, self._state = _local_item(
                    stream, self.index, problem, self._state
                )
        else:
            self._receive()
        self._enter_item()

    def trip_count(self):
        """Items this cursor runs, for a counted loop that ``cutlass.range`` can unroll::

            for k in cutlass.range(work.trip_count(), unroll=2):
                m, n, l = work.item_at(k)

        Instead of ``while work.valid: ...; work.advance()``, for a local ``StaticPersistent``
        or ``TilePersistent`` stream whose mapping keeps every item, decodes any index and keeps
        no decode state (``Raster``). Item ``k`` is index ``first + k * stride``; the count is
        ``ceil((end - first) / stride)``, or 0 when the first index is already terminal, with
        ``end`` the total or the limit. A cursor runs in one mode: ``advance()`` on a counted
        cursor, or ``trip_count()`` on an advanced one, raises at trace time.
        """
        self._check_current()
        stream, rec = self._stream, self._rec
        why = None
        if self._end is not None:
            why = "its items arrive over a broadcast ring"
        elif type(stream.schedule) not in (StaticPersistent, TilePersistent):
            why = f"{stream.schedule.describe()} has no fixed stride"
        elif not stream.mapping.decodes_any_index or stream.mapping.skips:
            why = f"{type(stream.mapping).__name__} does not decode every index unguarded"
        elif self._state is not None:
            why = f"{type(stream.mapping).__name__} keeps a decode state"
        elif rec.advances:
            why = f"it already advances at {rec.advances[0][0]}"
        elif rec.counted:
            why = f"it is already counted at {rec.counted[0]}, so its items would run twice"
        if why is not None:
            raise ProtocolError(
                f"{rec.role}: trip_count() on {rec.cursor} at {user_site()}: {why}; loop with "
                f"`while {rec.cursor}.valid: ...; {rec.cursor}.advance()`"
            )
        rec.counted.append(user_site())
        remaining = cutlass.max(Int32(self._problem.total) - self.index, Int32(0))
        # The fixed index step, computed once where the loop starts (its value dominates it).
        stride = Int32(stream.schedule.next_index(self.index, self._problem)) - self.index
        rec.stride = stride
        return Int32((Uint32(remaining) + Uint32(stride) - 1) // Uint32(stride))

    def item_at(self, k):
        """Item ``k`` of a counted loop (``trip_count()``): index ``first + k * stride``. Call it
        only in the body of that one loop; before the loop, ``item`` is the first item."""
        self._check_current()
        rec = self._rec
        if not rec.counted:
            raise ProtocolError(
                f"{rec.role}: item_at() at {user_site()} before trip_count(); a counted "
                f"loop runs `for k in cutlass.range({rec.cursor}.trip_count()): ...`"
            )
        block = current_block()
        if block.owner.name != "scf.for" or rec.loop_block not in (None, block):
            raise ProtocolError(
                f"{rec.role}: item_at() at {user_site()} must run directly in the body of the "
                f"one `for k in cutlass.range({rec.cursor}.trip_count()):` loop, not under a "
                "branch, a nested loop or another loop"
            )
        rec.loop_block = rec.trace.item_block = block  # per-item channel uses count from here
        index = self.index + Int32(k) * rec.stride
        words, _ = _decode(self._stream, index, self._problem, None)
        return self._stream.mapping.unpack(words)

    def _receive(self):
        h = self._end.wait()
        self.index = Int32(h.data[0])
        self._words = tuple(Int32(h.data[1 + i]) for i in range(self._stream.mapping.words))
        h.release()


class _Fetch:
    """The scheduler's private schedule channels (``{name: (producer end, consumer end)}``) as one
    DSL value, so a single ``fetch.claim_next()`` call in a loop carries every phase."""

    def __init__(self, schedule, ends: dict, problem):
        self._schedule, self._ends, self._problem = schedule, ends, problem

    def __extract_mlir_values__(self):
        return [v for pc in self._ends.values() for e in pc for v in e.__extract_mlir_values__()]

    def __new_from_mlir_values__(self, values):
        ends = {}
        for k, pair in self._ends.items():
            new = []
            for e in pair:
                n = len(e.__extract_mlir_values__())
                new.append(e.__new_from_mlir_values__(values[:n]))
                values = values[n:]
            ends[k] = tuple(new)
        return _Fetch(self._schedule, ends, self._problem)

    def claim_next(self, index):
        return Int32(self._schedule.claim_next(self._ends, self._problem, index))

    def close_claims(self):
        self._schedule.close_claims(self._ends, self._problem)


def _publish(h, index, words, total):
    chan = h._end._chan
    if isinstance(chan.producer, Tma):  # a cluster's ring: the bytes complete each full barrier
        targets = chan.scope.full_ctas if chan.scope.pair else (0,)  # local: leader-only users
        _publish_cluster(h.data.iterator, h.barrier, [index, *words], targets)
    else:
        _publish_local(h, index, words, total)


@cute.jit
def _publish_local(h, index, words, total):
    with cute.arch.elect_one():
        h.data[0] = index
        if index < total:
            for i in cutlass.range_constexpr(len(words)):
                h.data[1 + i] = words[i]


@cute.jit
def _publish_cluster(slot_ptr, full, values, targets: cutlass.Constexpr):
    """Leader: ``st.async`` the item into the slot of every consumer CTA; the stores complete
    the bytes the producer armed on each CTA's full barrier (``Tma(16 * chunks)``)."""
    words = list(values) + [Int32(0)] * (-len(values) % 4)
    base = slot_ptr.toint()
    with cute.arch.elect_one():
        for c in cutlass.range_constexpr(len(targets)):
            for q in cutlass.range_constexpr(len(words) // 4):
                dst = cute.make_ptr(Int32, base + 16 * q, cute.AddressSpace.smem, assumed_align=16)
                store_async_dsmem(dst, tuple(words[4 * q : 4 * q + 4]), full, targets[c])


@cute.jit
def _publish_kept(h, fetch, stream, problem, index, state):
    """Publish into ``h`` the first kept item at or after ``index``, claiming past dropped
    ones. Returns ``(fetch, state)``: the skip loop rebinds both, so the caller must take them
    back."""
    mapping = stream.mapping
    if cutlass.const_expr(mapping.skips):
        dropped, words, state = _decode_kept(mapping, index, problem.total, problem.space, state)
        while dropped:
            index = fetch.claim_next(index)
            dropped, words, state = _decode_kept(
                mapping, index, problem.total, problem.space, state
            )
    else:
        words, state = _decode(stream, index, problem, state)
    _publish(h, index, words, problem.total)
    return fetch, state


@cute.jit
def scheduler_role(kernel, ring, cursor: WorkCursor, fetch: _Fetch):
    """Layer-owned scheduler warp for broadcast schedules: claim, decode, publish, repeat.

    A claim is made only once a ring slot is free, so a CTA reserves at most ``stages`` items
    beyond what its slowest role has started, and only after the previous claim came back valid
    (CLC forbids querying after a failed query). The scheduler consumes its own ring like every
    other role, which keeps it in step with the CTA.
    """
    stream = cursor._stream
    problem = cursor._problem
    # The decode state travels beside the cursor, which only receives here.
    state, cursor._state = cursor._state, None
    if cutlass.const_expr(_publishes_first(stream)):
        # Publish the first item like a claim: hooks may move it off blockIdx.x, and ring
        # receivers decode no other mapping.
        if cutlass.const_expr(stream.ready is not None):
            stream.ready.wait()
        state = _start(stream.mapping, problem.space)
        with ring.produce() as h0:
            fetch, state = _publish_kept(h0, fetch, stream, problem, cursor.index, state)
        cursor._receive()
        cursor._enter_item()
    while cursor.valid:
        with ring.produce() as h:
            index = fetch.claim_next(cursor.index)
            fetch, state = _publish_kept(h, fetch, stream, problem, index, state)
        cursor.advance()
    fetch.close_claims()
    return ring, cursor
