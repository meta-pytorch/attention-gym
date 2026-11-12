"""Test custom schedules, WorkMap implementations, and raw-role work-item access."""

import itertools
import math
import os
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import Int32, Int64, cute
from test_warpdrive_scheduling import (
    EMPTY_SPACES,
    OVERFLOWING_SPACES,
    SCHEDULES,
    CheckedRaster,
    CountToy,
    SkipOdd,
    StreamToy,
    _count_items,
    check_gemm,
    compile_count_toy,
)

from attn_gym.testing.warpdrive import INVALID_LAUNCH
from attn_gym.warpdrive import (
    AtomicPersistent,
    ClusterLaunchControl,
    CtaPair,
    IndexSchedule,
    LastEntry,
    Limit,
    OneShot,
    PdlWait,
    ProtocolError,
    Raster,
    Readiness,
    SingleCta,
    StaticPersistent,
    TilePersistent,
    WorkCursor,
    WorkMap,
    WorkStream,
    role,
)
from attn_gym.warpdrive.cache import (
    canonical_key,
    compile_tvm_ffi,
    jit_cache,
    make_fake_compact_tensor,
)
from attn_gym.warpdrive.examples.gemm import Gemm
from attn_gym.warpdrive.examples.packed_work import PackedWork
from attn_gym.warpdrive.scheduling import tile_grid_dims

# Custom schedules, written against the public protocol only.


@cute.jit
def _countdown(index, problem):
    """``index`` while it is a valid index, else the terminal index ``total``."""
    return cutlass.select_(index >= 0, index, problem.total)


@dataclass(frozen=True)
class ReverseStatic(IndexSchedule):
    """Static persistent from the back: unit ``b`` runs ``total-1-b, total-1-b-U, ...``.
    Computed locally in every role."""

    def first_index(self, problem):
        return _countdown(problem.total - 1 - problem.first_unit(), problem)

    def next_index(self, index, problem):
        return _countdown(index - problem.units(), problem)


@dataclass(frozen=True)
class ReverseBroadcast(IndexSchedule):
    """The same sequence, claimed on the scheduler warp and broadcast (stateful through the
    previous index ``claim_next`` receives)."""

    stages: int = 2
    broadcast = True

    def first_index(self, problem):
        return _countdown(problem.total - 1 - problem.first_unit(), problem)

    def claim_next(self, fetch, problem, index):
        return _countdown(index - problem.units(), problem)


# Host-side contract


def test_custom_protocol_errors_at_declaration():
    class NoStep(IndexSchedule):
        pass

    class NoClaim(IndexSchedule):
        broadcast = True
        stages = 2

    for kwargs, error, message in [
        ({"schedule": NoStep()}, TypeError, "local schedule but does not define next_index"),
        ({"schedule": NoClaim()}, TypeError, "broadcast schedule but does not define claim_next"),
        ({"schedule": SimpleNamespace(next_index=None)}, TypeError,
         "WorkStream.schedule: expected an instance of IndexSchedule"),
        ({"mapping": SimpleNamespace(words=3)}, TypeError,
         "WorkStream.mapping: expected an instance of WorkMap"),
        ({"ready": object()}, TypeError, "WorkStream.ready: expected an instance of Readiness"),
        ({"schedule": ReverseBroadcast(stages=0)}, ValueError, "broadcast schedule stages"),
    ]:  # fmt: skip
        with pytest.raises(error, match=message):
            WorkStream(**kwargs)


def test_a_schedule_has_one_kind():
    """A schedule has one kind: the layer takes its grid, first item and decode from that kind."""
    with pytest.raises(
        TypeError, match="TileOneShot derives from the schedules TilePersistent, OneShot; a "
    ):

        class TileOneShot(TilePersistent, OneShot):
            pass

    with pytest.raises(
        TypeError, match="ReverseShot derives from the schedules ReverseStatic, OneShot; a "
    ):

        class ReverseShot(ReverseStatic, OneShot):
            pass

    # Deriving from one kind, at any depth, stays a schedule of that kind.
    class Bounded(StaticPersistent):
        pass

    class Deeper(Bounded):
        pass

    assert WorkStream(schedule=Deeper()).get_key() == "deeper0_m1"


@dataclass(frozen=True)
class Strided(IndexSchedule):
    """A schedule with a compile-time field: the default key must name it."""

    stride: int = 1

    def next_index(self, index, problem):
        return index + self.stride * problem.units()


@dataclass(frozen=True)
class NoWait(Readiness):
    """A custom readiness hook (nothing to wait for)."""

    def wait(self):
        pass


def test_custom_schedule_keys_and_launch_needs():
    assert WorkStream(schedule=ReverseStatic()).get_key() == "reversestatic0_m1"
    assert WorkStream(schedule=ReverseBroadcast(3)).get_key() == "reversebroadcast3_m1"
    assert WorkStream(schedule=Strided(2)).get_key() == "strided0_stride2_m1"
    assert WorkStream(ready=NoWait()).get_key() == "oneshot0_m1_nowait"
    # Built-ins take the same default key as custom schedules.
    stream = WorkStream(mapping=Raster(swizzle=8), schedule=ClusterLaunchControl())
    assert stream.get_key() == "clusterlaunchcontrol2_m8"
    assert WorkStream().get_key() == "oneshot0_m1"
    # Every built-in schedule (and its stage count) has its own key.
    keys = {
        WorkStream(mapping=Raster(swizzle=8), schedule=p).get_key() for p in SCHEDULES.values()
    }
    assert len(keys) == len(SCHEDULES)
    # A mapping without describe() is named by its class.
    assert WorkStream(mapping=SplitK()).describe() == "OneShot(), SplitK"
    assert ReverseStatic().uses_max_ctas and not ReverseStatic().uses_workspace
    with pytest.raises(ValueError, match="AtomicPersistent.*workspace"):
        WorkStream(schedule=AtomicPersistent()).bind((1, 1, 1))


@pytest.mark.parametrize("schedule", [OneShot(), StaticPersistent(), ClusterLaunchControl()])
def test_scheduler_warp_only_specializes_broadcast_schedules(schedule):
    kernels = []
    for warp in (None, 30, 31):
        op = Gemm()
        op.work = WorkStream(schedule=schedule, scheduler_warp=warp)
        kernels.append(op)
    keys = {canonical_key(op) for op in kernels}
    assert len(keys) == (3 if schedule.broadcast else 1)
    if not schedule.broadcast:
        assert all(op.work.scheduler_warp is None for op in kernels)
        assert kernels[0].describe() == kernels[1].describe() == kernels[2].describe()


def test_single_item_rejects_persistent_schedules():
    stream = WorkStream(schedule=StaticPersistent())
    with pytest.raises(ProtocolError, match="non-persistent local schedule"):
        stream.single_item()
    with pytest.raises(ProtocolError, match="after begin_roles"):
        WorkStream().single_item()


# GPU: custom schedules on the schedule toy (single CTA) and the count toy (pairs)

CUSTOM = {"reverse_static": ReverseStatic(), "reverse_broadcast": ReverseBroadcast(stages=2)}
BUILTIN = ["oneshot", "static", "atomic", "clc"]


# Stateful decoding: WorkMap.start and the state threaded through decode


class Chain(NamedTuple):
    prev: Int32  # the index this site decoded last, -1 before its first decode
    decodes: Int32  # decodes at this site so far


@dataclass(frozen=True)
class Chained(Raster):
    """Expose each decoding site's previous index and decode count in the item words.

    State-dependent words violate a real mapping's cache contract. Here every site in a
    unit sees the same index sequence, so the roles agree. ``drop_odd`` drops tiles with
    odd m + n.
    """

    drop_odd: bool = False
    words = 5

    @property
    def skips(self) -> bool:
        return self.drop_odd

    def start(self, space):
        return Chain(Int32(-1), Int32(0))

    def decode(self, index, space, state):
        _, (m, n, li), _ = super().decode(index, space, None)
        keep = self.kept((m, n, li)) if self.drop_odd else True
        return keep, (m, n, li, state.prev, state.decodes), Chain(index, state.decodes + 1)

    def kept(self, item):
        m, n, _ = item
        return not self.drop_odd or (m + n) % 2 == 0

    def get_key(self) -> str:
        return f"chained{int(self.drop_odd)}_" + super().get_key()


@dataclass(frozen=True)
class ChainedAnyIndex(Chained):
    """Decodable at any index: broadcast roles decode the first item locally."""

    decodes_any_index = True


@dataclass(frozen=True)
class ChainedTiles(Chained):
    """``Chained`` whose item is the tile, for toys that read ``(m, n, l)``."""

    def unpack(self, words):
        return tuple(words[:3])


class ChainToy(StreamToy):
    """Each role logs ``(index, prev, decodes)`` of each item in ``log[role, cta, k]``."""

    MAX_ITEMS = 32

    @cute.jit
    def visit(self, r: int, index, item, k, ctx):
        _, _, _, prev, decodes = item
        if k < self.MAX_ITEMS:
            bidx, _, _ = cute.arch.block_idx()
            ctx.log[r, bidx, k, 0] = index
            ctx.log[r, bidx, k, 1] = prev
            ctx.log[r, bidx, k, 2] = decodes

    @cute.kernel
    def kernel(self, log: cute.Tensor, bound):
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace(log=log))

    @cute.jit
    def __call__(self, log: cute.Tensor, workspace: cute.Tensor, em: Int32, en: Int32,
                 el: Int32, max_ctas: Int32, stream: cuda.CUstream):  # fmt: skip
        bound = self.work.bind((em, en, el), workspace, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(log, bound).launch(**self.launch_config(bound), stream=stream)


@jit_cache
def compile_chain_toy(op: ChainToy):
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(cutlass.Int32, (2, cute.sym_int(), ChainToy.MAX_ITEMS, 3)),
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),)),
        *(Int32(1) for _ in range(4)),
        name=op.get_name(),
    )


CHAIN_SCHEDULES = {**{name: SCHEDULES[name] for name in BUILTIN}, **CUSTOM}


@pytest.mark.parametrize(
    "mapping, cluster",
    [
        (Chained(swizzle=2), CtaPair()),
        (Chained(swizzle=2, drop_odd=True), CtaPair()),
        (ChainedAnyIndex(swizzle=2), SingleCta()),
    ],
    ids=["chained-pair", "chained_skip-pair", "chained_any_index-single"],
)
@pytest.mark.parametrize("schedule", list(CHAIN_SCHEDULES))
def test_decode_state_follows_each_site(requires_blackwell, schedule, mapping, cluster):
    """Carry state through every decode, including dropped items.

    Local roles decode their unit's sequence. A broadcast scheduler decodes claimed indices;
    roles do not decode broadcast items. Each kept tile runs once per CTA in its unit, and
    both roles see the same items. Empty spaces still initialize every site's state but
    yield no items. Custom schedules also use custom readiness hooks.
    """
    sched = CHAIN_SCHEDULES[schedule]
    op = ChainToy(sched, mapping, cluster, NoWait() if schedule in CUSTOM else None)
    fn = compile_chain_toy(op)
    em, en, el = ext = 9, 7, 2
    total, max_ctas, cs = em * en * el, 16 * cluster.size, cluster.size
    units = min(total, max_ctas // cs) if sched.uses_max_ctas else total
    log = torch.full((2, units * cs, ChainToy.MAX_ITEMS, 3), -2, dtype=torch.int32, device="cuda")
    ws, _ = op.work.launch_args("cuda")
    fn(log, ws, 0, en, el, max_ctas)
    torch.cuda.synchronize()
    assert (log == -2).all() and ws[:2].tolist() == [0, 0]
    fn(log, ws, em, en, el, max_ctas)
    torch.cuda.synchronize()
    log = log.cpu()
    assert torch.equal(log[0], log[1])
    kept = [i for i in range(total) if mapping.kept(Raster(swizzle=2).decode(i, ext, None)[1])]
    seen = sorted(int(i) for i in log[0, :, :, 0].flatten() if i >= 0)
    assert seen == sorted(kept * cs)
    stride = {"static": units, "reverse_static": -units, "reverse_broadcast": -units}.get(schedule)
    for cta in range(units * cs):
        items = [tuple(int(v) for v in row) for row in log[0, cta] if row[0] >= 0]
        if stride is not None:  # a strided sequence: the site decoded every index before it
            unit = cta // cs
            first = unit if stride > 0 else total - 1 - unit
            for index, prev, decodes in items:
                assert decodes == (index - first) // stride
                assert prev == (-1 if index == first else index - stride)
        elif not mapping.drop_odd:  # the site decoded exactly the items it published
            for k, (index, prev, decodes) in enumerate(items):
                assert decodes == k and prev == (items[k - 1][0] if k else -1)
        else:
            assert all(a[2] < b[2] for a, b in itertools.pairwise(items))
            assert all((prev == -1) == (decodes == 0) for _, prev, decodes in items)


# Pairs, empty spaces and overflow for custom schedules and mappings


@pytest.mark.parametrize("cluster", [SingleCta(), CtaPair()], ids=["single", "pair"])
@pytest.mark.parametrize(
    "schedule", [ReverseStatic(), ReverseBroadcast()], ids=["reverse_static", "reverse_broadcast"]
)
def test_swapped_schedules_cover_empty_and_overflowing_spaces(
    requires_blackwell, schedule, cluster
):
    """Every CTA in a unit sees its items; empty spaces run no work.

    Reject spaces past Int32 at launch, using the custom schedule's default index_limit
    and the mapping's Int64 count64. Built-in schedules on the guarded CheckedRaster are
    covered by test_empty_and_overflowing_spaces.
    """
    mapping = Raster(along="n", swizzle=3)
    op = CountToy(schedule, mapping, cluster)
    fn = compile_count_toy(op)
    ws, _ = op.work.launch_args("cuda")
    for space in EMPTY_SPACES:
        assert _count_items(fn, ws, space, 148) == [0, 0], space
    for space in OVERFLOWING_SPACES:
        with pytest.raises(Exception, match=INVALID_LAUNCH):
            _count_items(fn, ws, space, 148)
    want = 5 * 3 * 2 * cluster.size
    assert _count_items(fn, ws, (5, 3, 2), 4) == [want, want]


# Raw roles: the OneShot work item without a cursor


@cute.jit
def _hit(r: int, item, ctx):
    """Count tile ``(m, n, l)`` of ``item`` in ``ctx.hits[r, tile]``."""
    m, n, li = item
    tile = (li * ctx.en + n) * ctx.em + m
    cute.arch.atomic_add(ctx.hits.iterator + (r * ctx.hits.shape[1] + tile), Int32(1))


class RawItemToy(StreamToy):
    """The managed role ``a`` loops the stream; the raw role ``b`` (warps 1-2) reads its work
    item with ``self.work.single_item()``. Both count each tile in ``hits[role, tile]``."""

    def __init__(self, mapping, mode: str, cluster=None, schedule=None):
        super().__init__(schedule or OneShot(), mapping, cluster)
        self.mode = mode

    def get_name(self) -> str:
        return f"{super().get_name()}_{self.mode}"

    @cute.jit
    def visit(self, r: int, index, item, k, ctx):
        _hit(r, item, ctx)

    @role(warps=range(1, 3), raw=True)
    @cute.jit
    def b(self, ctx):
        valid, item = self.work.single_item()
        if valid and cute.arch.lane_idx() == 0 and cute.arch.warp_idx() == 1:
            _hit(1, item, ctx)

    @cute.kernel
    def kernel(self, hits: cute.Tensor, em: Int32, en: Int32, bound):
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace(hits=hits, em=em, en=en))

    @cute.jit
    def __call__(self, hits: cute.Tensor, em: Int32, en: Int32, el: Int32, stream: cuda.CUstream):
        bound = self.work.bind((em, en, el))
        self.kernel.set_name_prefix(self.get_name())
        cfg = self.launch_config(bound)
        if cutlass.const_expr(self.mode == "1d"):  # a hand-built (total * cs, 1, 1) grid
            cfg["grid"] = (cutlass.max(bound.total, 1) * self.cluster.size, 1, 1)
        self.kernel(hits, em, en, bound).launch(**cfg, stream=stream)


@jit_cache
def compile_raw_item_toy(op: RawItemToy):
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(cutlass.Int32, (2, cute.sym_int())),
        Int32(1),
        Int32(1),
        Int32(1),
        name=op.get_name(),
    )


@pytest.mark.parametrize(
    "mapping, mode, ext, cluster, schedule",
    [
        (Raster(), "grid", (5, 3, 2), SingleCta(), OneShot()),
        (Raster(), "grid", (2, 65536, 1), SingleCta(), OneShot()),  # n_tiles past gridDim.y: 1-D fallback
        (Raster(), "1d", (5, 3, 2), SingleCta(), OneShot()),
        (Raster(), "grid", (3, 70000, 1), CtaPair(), OneShot()),
        (Raster(swizzle=3), "grid", (7, 5, 2), CtaPair(), OneShot()),
        (SkipOdd(swizzle=2), "grid", (9, 7, 2), SingleCta(), OneShot()),
        (Raster(), "grid", (0, 3, 1), SingleCta(), OneShot()),  # empty: one unit, no valid item
        # A default WorkMap traps if decoded past the end: one CTA per item, plus drops.
        (CheckedRaster(drop_odd=True), "grid", (9, 7, 2), CtaPair(), OneShot()),
        # A stateful mapping: single_item starts the state like a managed role.
        (ChainedTiles(swizzle=2, drop_odd=True), "grid", (9, 7, 2), CtaPair(), OneShot()),
        (ChainedTiles(swizzle=2), "grid", (0, 3, 1), SingleCta(), OneShot()),
        # Static grids: the roles of a single CTA and of a pair read the same items.
        (Raster(), "grid", (5, 3, 2), CtaPair(), OneShot(grid="3d")),
        (Raster(), "grid", (3, 65535, 1), SingleCta(), OneShot(grid="3d")),
        (Raster(), "grid", (0, 3, 1), CtaPair(), OneShot(grid="3d")),
        (Raster(), "grid", (3, 2, 65537), CtaPair(), OneShot(grid="2d")),
        (Raster(), "grid", (5, 3, 0), SingleCta(), OneShot(grid="2d")),
        (Raster(), "grid", (2, 65536, 1), CtaPair(), OneShot(grid="1d")),
        (Raster(swizzle=3), "grid", (7, 5, 2), CtaPair(), OneShot(grid="1d")),
        (Raster(), "grid", (4, 0, 2), SingleCta(), OneShot(grid="1d")),
    ],
    ids=["tile_grid", "fallback", "hand_1d", "pair_fallback", "pair_swizzle", "skip",
         "empty", "checked_skip", "stateful_skip", "stateful_empty", "pair_3d", "3d_max_n",
         "pair_3d_empty", "pair_2d", "2d_empty", "pair_1d", "pair_1d_swizzle",
         "1d_empty"],
)  # fmt: skip
def test_raw_role_single_item(requires_blackwell, mapping, mode, ext, cluster, schedule):
    check_raw_role_single_item(mapping, mode, ext, cluster, schedule)


def check_raw_role_single_item(mapping, mode, ext, cluster, schedule=None):
    op = RawItemToy(mapping, mode, cluster, schedule)
    em, en, el = ext
    total = max(em * en * el, 0)
    hits = torch.zeros(2, max(total, 1), dtype=torch.int32, device="cuda")
    fn = compile_raw_item_toy(op)
    fn(hits, em, en, el)
    torch.cuda.synchronize()
    kept = [
        0 if hasattr(mapping, "kept") and not mapping.kept((m, n, li)) else cluster.size
        for li in range(el)
        for n in range(en)
        for m in range(em)
    ]
    # The whole buffer: the idle unit of an empty space must count nothing.
    kept = torch.tensor(kept or [0], dtype=torch.int32)
    assert torch.equal(hits[0].cpu(), kept) and torch.equal(hits[1].cpu(), kept)


# Static OneShot grids, and kernels without roles


def test_oneshot_grid_is_a_static_choice():
    grids = ("auto", "3d", "2d", "1d")
    assert [WorkStream(schedule=OneShot(grid=g)).get_key() for g in grids] == [
        "oneshot0_m1",
        "oneshot0_grid3d_m1",
        "oneshot0_grid2d_m1",
        "oneshot0_grid1d_m1",
    ]
    assert WorkStream(schedule=OneShot(grid="3d")).describe() == "OneShot(grid='3d'), " + (
        Raster().describe()
    )
    with pytest.raises(ValueError, match="'auto', '3d', '2d' or '1d'"):
        OneShot(grid="bogus")

    class Shifted(OneShot):
        def first_index(self, problem):
            return problem.first_unit()

    # These grids have no general decode: only the plain raster order is blockIdx.
    for grid in ("3d", "2d"):
        for kwargs in (
            {"mapping": Raster(swizzle=2)},
            {"mapping": Raster(along="n")},
            {"mapping": SkipOdd()},
            {"schedule": Shifted(grid=grid)},
        ):
            with pytest.raises(ValueError, match=rf"OneShot\(grid='{grid}'\) reads the item"):
                WorkStream(**{"schedule": OneShot(grid=grid), **kwargs})
    # "1d" chooses among a Raster's grids; another mapping only has the 1-D one.
    flat = WorkStream(mapping=Raster(swizzle=2), schedule=Shifted(grid="1d"))
    assert flat.get_key() == "shifted0_grid1d_m2"
    assert flat.describe().startswith("Shifted(grid='1d')")
    assert WorkStream(schedule=Shifted()).get_key() == "shifted0_m1"
    for mapping in (SkipOdd(),):
        with pytest.raises(ValueError, match=r"OneShot\(grid='1d'\) chooses the grid"):
            WorkStream(mapping=mapping, schedule=OneShot(grid="1d"))


@pytest.mark.parametrize(
    "space, grid, launch",
    [
        ((7, 65535, 2), "3d", (7, 65535, 2)),
        ((7, 2, 65535), "3d", (7, 2, 65535)),
        ((2**20, 1, 1), "3d", (2**20, 1, 1)),  # gridDim.x has no 65535 limit
        ((0, 3, 1), "3d", (1, 1, 1)),
        ((3, 1, 65536), "2d", (3 * 65536, 1, 1)),
        ((3, 65535, 70000), "2d", (3 * 70000, 65535, 1)),
        ((1, 65536, 1), "1d", (65536, 1, 1)),
        ((2, 65536, 65536), "1d", (2**33, 1, 1)),
    ],
)
def test_grid_for_selects_the_static_grid(space, grid, launch):
    assert OneShot.grid_for(space) == grid
    # The selected variant launches the space (static extents: the launch grid on the host).
    x, y, z = WorkStream(schedule=OneShot(grid=grid)).bind(space).grid()
    assert (x or launch[0], y, z) == launch  # past Int32 the grid is zero-wide
    assert (x == 0) == (math.prod(space) > 2**31 - 1)


def test_static_grids_launch_whole_clusters():
    assert tile_grid_dims((5, 3, 2), 2, "3d") == (10, 3, 2)
    assert tile_grid_dims((5, 3, 70000), 2, "2d") == (2 * 5 * 70000, 3, 1)
    assert tile_grid_dims((5, 0, 70000), 2, "2d") == (2, 1, 1)


@pytest.mark.parametrize(
    "grid, space",
    [
        ("3d", (3, 65536, 1)),
        ("3d", (3, 2, 65536)),
        ("2d", (3, 65536, 2)),
        ("2d", (3, 65536, 65536)),
    ],
)
def test_static_grid_rejects_an_oversized_static_space(grid, space):
    bound = WorkStream(schedule=OneShot(grid=grid)).bind(space)
    with pytest.raises(ValueError, match=r"OneShot\.grid_for\(space\).*'2d' or '1d'"):
        bound.grid()


class PlainToy:
    """A kernel without roles: every CTA counts its items in ``hits[tile]``, read with
    ``bound.single_item()`` (``mode="item"``) or a ``bound.cursor()`` loop. ``static`` gives
    ``(m_tiles, n_tiles)`` compile-time values (None: the runtime argument)."""

    def __init__(self, schedule, mapping=None, mode: str = "item", static=(None, None)):
        self.work = WorkStream(mapping=mapping or Raster(), schedule=schedule)
        self.mode = mode
        self.static = static

    def get_name(self) -> str:
        static = "".join("r" if e is None else str(e) for e in self.static)
        return f"warpdrive_plaintoy_{self.work.get_key()}_{self.mode}_{static}"

    def __cute_cache_key__(self) -> str:
        return self.get_name()

    @cute.kernel
    def kernel(self, hits: cute.Tensor, em: Int32, en: Int32, bound):
        first_thread = cute.arch.thread_idx()[0] == 0
        ctx = SimpleNamespace(hits=hits, em=em, en=en)
        if cutlass.const_expr(self.mode == "item"):
            valid, item = bound.single_item()
            if valid and first_thread:
                _hit(0, item, ctx)
        elif cutlass.const_expr(self.mode.startswith("count")):
            work = bound.cursor()
            trips = work.trip_count()
            if cutlass.const_expr(self.mode == "count_twice"):
                trips = work.trip_count()
            if cutlass.const_expr(self.mode == "count_item_before_loop"):
                _hit(0, work.item_at(0), ctx)
            for k in cutlass.range(trips, unroll=2):
                if cutlass.const_expr(self.mode == "count_item_in_branch"):
                    if first_thread:
                        _hit(0, work.item_at(k), ctx)
                else:
                    item = work.item_at(k)
                    if first_thread:
                        _hit(0, item, ctx)
            if cutlass.const_expr(self.mode == "count_then_advance"):
                work.advance()
        else:
            work = bound.cursor()
            while work.valid:
                if first_thread:
                    _hit(0, work.item, ctx)
                work.advance()
            if cutlass.const_expr(self.mode == "advance_then_count"):
                work.trip_count()

    @cute.jit
    def __call__(
        self,
        hits: cute.Tensor,
        em: Int32,
        en: Int32,
        el: Int32,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        tiles = (e if s is None else s for e, s in zip((em, en), self.static, strict=True))
        bound = self.work.bind((*tiles, el), None, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(hits, em, en, bound).launch(grid=bound.grid(), block=(64, 1, 1), stream=stream)


@jit_cache
def compile_plain_toy(op: PlainToy):
    hits = make_fake_compact_tensor(cutlass.Int32, (1, cute.sym_int()))
    return compile_tvm_ffi(op, hits, Int32(1), Int32(1), Int32(1), Int32(1), name=op.get_name())


def _plain_hits(op, ext, max_ctas=148):
    """Counts per tile; one zero for an empty space, whose idle CTA must count nothing. An item
    past the space would count in the checked padding."""
    size, pad = max(math.prod(ext), 1), 64
    hits = torch.zeros(1, size + pad, dtype=torch.int32, device="cuda")
    compile_plain_toy(op)(hits, *ext, max_ctas)
    torch.cuda.synchronize()
    assert not hits[0, size:].any(), "an item past the space ran"
    return hits[0, :size].cpu()


@pytest.mark.parametrize(
    "schedule, mapping, mode, ext",
    [
        (OneShot(), None, "item", (5, 3, 2)),
        (OneShot(), None, "item", (2, 65536, 1)),  # the runtime 1-D fallback
        (OneShot(grid="3d"), None, "item", (5, 3, 2)),
        (OneShot(grid="3d"), None, "item", (3, 65535, 1)),
        (OneShot(grid="3d"), None, "item", (0, 3, 1)),  # empty: one CTA, no valid item
        (OneShot(grid="2d"), None, "item", (5, 3, 2)),
        (OneShot(grid="2d"), None, "item", (3, 2, 65537)),
        (OneShot(grid="2d"), None, "item", (1, 65535, 3)),
        (OneShot(grid="2d"), None, "item", (5, 3, 0)),
        (OneShot(grid="2d"), None, "cursor", (5, 3, 2)),
        (OneShot(grid="1d"), None, "item", (5, 3, 2)),
        (OneShot(grid="1d"), None, "item", (2, 65536, 1)),
        (OneShot(grid="1d"), None, "item", (3, 2, 65537)),
        (OneShot(grid="1d"), None, "item", (5, 0, 2)),
        (OneShot(), SkipOdd(swizzle=2), "item", (9, 7, 2)),
        (OneShot(grid="3d"), None, "cursor", (5, 3, 2)),
        (OneShot(grid="1d"), Raster(swizzle=3), "cursor", (7, 5, 2)),
        (OneShot(grid="1d"), Raster(along="n"), "item", (7, 5, 3)),
        (StaticPersistent(), None, "cursor", (9, 7, 2)),  # 4 CTAs run 126 items
        (StaticPersistent(), SkipOdd(swizzle=2), "cursor", (9, 7, 2)),
        (StaticPersistent(), None, "cursor", (0, 7, 2)),
        # Static tile extents (Python ints) beside runtime ones; l carries emptiness.
        (OneShot(grid="2d"), None, "static_m", (3, 2, 65537)),
        (OneShot(grid="2d"), None, "static_m", (3, 2, 0)),
        (OneShot(grid="3d"), None, "static_mn", (5, 3, 2)),
        (OneShot(), None, "static_mn", (2, 3, 2)),
        (OneShot(), None, "static_mn", (5, 3, 0)),
        (OneShot(grid="1d"), Raster(swizzle=2), "static_mn", (5, 3, 2)),
        (StaticPersistent(), None, "static_mn", (9, 7, 2)),
        (StaticPersistent(), Raster(along="n", swizzle=3), "static_m", (9, 7, 2)),
        # One axis walked per CTA (max_ctas=4 units along it), the others from blockIdx.
        (TilePersistent("m"), None, "cursor", (9, 3, 2)),
        (TilePersistent("n"), None, "cursor", (3, 9, 2)),
        (TilePersistent("l"), None, "cursor", (3, 2, 9)),
        (TilePersistent("n"), None, "static_mn", (3, 2, 2)),
        (TilePersistent("m"), None, "cursor", (0, 3, 2)),
        (TilePersistent("n"), None, "cursor", (3, 9, 0)),
        # A compile-time stride: the grid's axis is exactly `units`, idle CTAs run nothing.
        (TilePersistent("m", units=4), None, "cursor", (9, 3, 2)),
        (TilePersistent("n", units=16), None, "cursor", (3, 9, 2)),
        (TilePersistent("l", units=3), None, "cursor", (2, 3, 7)),
        # Counted loops (unroll=2): 126 items over 4 CTAs, not a multiple; empty spaces.
        (StaticPersistent(), None, "counted", (9, 7, 2)),
        (StaticPersistent(), None, "counted", (3, 1, 1)),  # fewer items than max_ctas
        (StaticPersistent(), None, "counted", (0, 7, 2)),
        (TilePersistent("n"), None, "counted", (3, 9, 2)),
        (TilePersistent("l"), None, "counted", (2, 3, 2)),  # stride 4 past the extent of 2
        (TilePersistent("m"), None, "counted", (0, 3, 2)),
        (TilePersistent("l", units=3), None, "counted", (2, 3, 7)),  # constant stride
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_plain_kernel_reads_its_work(requires_blackwell, schedule, mapping, mode, ext):
    """A kernel without roles covers the space once, on every static grid and through a
    persistent cursor loop, also with static tile extents."""
    em, en, _ = ext
    static = {"static_m": (em, None), "static_mn": (em, en)}.get(mode, (None, None))
    if mode.startswith("static"):
        mode = "cursor" if schedule.persistent else "item"
    op = PlainToy(schedule, mapping, mode, static)
    em, en, el = ext
    kept = [
        int(mapping is None or not mapping.skips or mapping.kept((m, n, li)))
        for li in range(el)
        for n in range(en)
        for m in range(em)
    ]
    kept = torch.tensor(kept or [0], dtype=torch.int32)
    assert torch.equal(_plain_hits(op, ext, max_ctas=4), kept)


@pytest.mark.parametrize(
    "grid, spaces",
    [("3d", [(2, 65536, 1), (2, 3, 65536)]), ("2d", [(2, 65536, 1), (1, 65536, 3)])],
)
def test_static_grid_rejects_an_oversized_dynamic_space(requires_blackwell, grid, spaces):
    """Host code in the jit entry cannot raise on a runtime extent, so the grid is launched as
    is and CUDA rejects it before any CTA runs; the next launch is unaffected."""
    op = PlainToy(OneShot(grid=grid))
    for ext in spaces:
        with pytest.raises(Exception, match=INVALID_LAUNCH):
            _plain_hits(op, ext)
    assert _plain_hits(op, (5, 3, 2)).tolist() == [1] * 30


def test_static_3d_grid_reads_block_idx_without_a_fallback(
    requires_blackwell, tmp_path, monkeypatch
):
    """``grid="3d"`` takes the item from ``blockIdx`` alone: no branch on the launch shape
    and no division, where ``"auto"`` keeps both for its 1-D fallback. ``"2d"`` adds one
    division."""
    from cutlass.base_dsl.compiler import Compiler

    compile_fn = Compiler.compile

    def ptx_of(grid: str, static=(None, None)) -> str:
        label = grid + "".join(f"_{e}" for e in static if e is not None)
        (tmp_path / label).mkdir()

        def dump_ptx(self, module, pipeline, *args, **kwargs):
            option = f"dump-ptx-path='{tmp_path / label / 'kernel'}' "
            pipeline = pipeline.replace("cute-to-nvvm{", "cute-to-nvvm{" + option, 1)
            return compile_fn(self, module, pipeline, *args, **kwargs)

        monkeypatch.setattr(Compiler, "compile", dump_ptx)
        # Bypass the cache: the PTX needs a compile.
        compile_plain_toy.__wrapped__(PlainToy(OneShot(grid=grid), static=static))
        (ptx,) = (tmp_path / label).glob("*.ptx")
        return ptx.read_text(errors="ignore")

    decodes = re.compile(r"^\s*(div|rem)\.", re.MULTILINE)
    tiles = ptx_of("3d")
    assert all(f"%ctaid.{axis}" in tiles for axis in "xyz")
    assert "%nctaid" not in tiles and not decodes.search(tiles)
    auto = ptx_of("auto")
    assert "%nctaid.y" in auto and decodes.search(auto)
    fold = ptx_of("2d")  # l from blockIdx.x: one unsigned division, n is blockIdx.y
    assert "%ctaid.y" in fold and "%ctaid.z" not in fold and "%nctaid" not in fold
    assert [m.group(0).strip() for m in decodes.finditer(fold)] == ["div."]
    assert "div.u32" in fold
    # A static m_tiles stays a constant: the unfold multiplies instead of dividing.
    fold_static = ptx_of("2d", static=(3, None))
    assert "%ctaid.x" in fold_static and not decodes.search(fold_static)
    flat = ptx_of("1d")  # the 1-D index, decoded unsigned
    assert "%ctaid.y" not in flat and "%nctaid" not in flat
    assert "div.u32" in flat and "div.s32" not in flat and "div.s32" in auto


def test_plain_kernel_access_rejections():
    """``BoundWork.single_item``/``cursor`` serve kernels without roles and local schedules."""
    owned = StreamToy(OneShot()).work.bind((2, 2, 1))
    for method in (owned.single_item, owned.cursor):
        with pytest.raises(ProtocolError, match="is for kernels without roles"):
            method()
    workspace = object()  # never read: the checks run first
    for schedule in (AtomicPersistent(), ClusterLaunchControl()):
        bound = WorkStream(schedule=schedule).bind((2, 2, 1), workspace, 4)
        with pytest.raises(ProtocolError, match="cursor\\(\\) needs a local schedule"):
            bound.cursor()
        with pytest.raises(ProtocolError, match="non-persistent local schedule"):
            bound.single_item()
    with pytest.raises(ProtocolError, match="non-persistent local schedule"):
        WorkStream(schedule=StaticPersistent()).bind((2, 2, 1), None, 4).single_item()


# Device-read limits: WorkStream.bind(..., limit=...).


class LimitToy(StreamToy):
    """Managed roles ``a`` and ``b`` and, for ``mode="item"``, raw role ``c`` reading
    ``single_item()`` count each item of a stream bound with ``limit=LastEntry(table)`` in
    ``hits[role, tile]``."""

    def __init__(self, schedule, mode: str = "cursor"):
        super().__init__(schedule)
        self.mode = mode

    def get_name(self) -> str:
        return f"{super().get_name()}_{self.mode}"

    @cute.jit
    def visit(self, r: int, index, item, k, ctx):
        _hit(r, item, ctx)

    @role(warps=3, raw=True)
    @cute.jit
    def c(self, ctx):
        if cutlass.const_expr(self.mode == "item"):
            valid, item = self.work.single_item()
            if valid and cute.arch.lane_idx() == 0:
                _hit(2, item, ctx)

    @cute.kernel
    def kernel(self, hits: cute.Tensor, em: Int32, en: Int32, bound):
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace(hits=hits, em=em, en=en))

    @cute.jit
    def __call__(
        self,
        hits: cute.Tensor,
        table: cute.Tensor,
        em: Int32,
        en: Int32,
        el: Int32,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        bound = self.work.bind((em, en, el), None, max_ctas, limit=LastEntry(table))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(hits, em, en, bound).launch(**self.launch_config(bound), stream=stream)


class LimitPlainToy(PlainToy):
    """``PlainToy`` (a kernel without roles) with ``limit=LastEntry(table)``."""

    def get_name(self) -> str:
        return super().get_name().replace("plaintoy", "limitplaintoy")

    @cute.jit
    def __call__(
        self,
        hits: cute.Tensor,
        table: cute.Tensor,
        em: Int32,
        en: Int32,
        el: Int32,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        bound = self.work.bind((em, en, el), None, max_ctas, limit=LastEntry(table))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(hits, em, en, bound).launch(grid=bound.grid(), block=(64, 1, 1), stream=stream)


@jit_cache
def compile_limit_toy(op):
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(cutlass.Int32, (3, cute.sym_int())),
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),)),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        name=op.get_name(),
    )


@pytest.mark.parametrize(
    "schedule, mode, plain",
    [
        (StaticPersistent(), "cursor", False),
        (StaticPersistent(), "cursor", True),
        (OneShot(), "item", False),
        (OneShot(grid="3d"), "item", False),
        (OneShot(grid="3d"), "item", True),
        (OneShot(grid="1d"), "cursor", True),
        (OneShot(grid="2d"), "item", True),
        (TilePersistent("n"), "cursor", False),
        (TilePersistent("m"), "cursor", True),
        (StaticPersistent(), "counted", True),
        (TilePersistent("n"), "counted", True),
    ],
    ids=["static_roles", "static_plain", "auto_roles", "3d_roles", "3d_plain", "1d_plain",
         "2d_plain", "tile_n_roles", "tile_m_plain", "static_counted", "tile_n_counted"],
)  # fmt: skip
def test_limit_ends_every_role_at_the_device_count(requires_blackwell, schedule, mode, plain):
    """Every decoding site reads the limit once and ends at ``min(total, limit)``: the first
    ``limit`` indices run, once, in every role; 0 or below runs nothing; a limit at or past the total
    changes nothing. ``TilePersistent``'s index is its axis coordinate. One compile serves every
    limit."""
    op = LimitPlainToy(schedule, mode=mode) if plain else LimitToy(schedule, mode)
    fn = compile_limit_toy(op)
    em, en, el = 5, 3, 2
    total = em * en * el
    for limit in (-3, 0, 7, total, total + 9):
        pad = 64  # an item past the stream's end would count in the padding
        hits = torch.zeros(3, total + pad, dtype=torch.int32, device="cuda")
        # A prefix-sum table: only its last entry is read.
        table = torch.tensor([0, 3, limit], dtype=torch.int32, device="cuda")
        fn(hits, table, em, en, el, 4)
        torch.cuda.synchronize()
        # Plain Raster: the linear index is the tile id ((l * en + n) * em + m).
        tiles = itertools.product(range(el), range(en), range(em))
        index = {
            TilePersistent: lambda li, n, m: (m, n, li)["mnl".index(schedule.axis)],
        }.get(type(schedule), lambda li, n, m: (li * en + n) * em + m)
        expect = torch.tensor([int(index(*t) < limit) for t in tiles] + [0] * pad)
        rows = [0] if plain else [0, 1, 2] if mode == "item" else [0, 1]
        for r in range(3):
            assert torch.equal(hits[r].cpu(), expect.int() if r in rows else 0 * expect.int()), (
                limit,
                r,
            )


def test_tile_persistent_reads_without_division(requires_blackwell, tmp_path, monkeypatch):
    """``TilePersistent`` reads two coordinates from ``blockIdx`` and steps the third by
    ``gridDim``: no division anywhere, and ``max_ctas`` is a runtime value."""
    from cutlass.base_dsl.compiler import Compiler

    compile_fn = Compiler.compile

    def dump_ptx(self, module, pipeline, *args, **kwargs):
        option = f"dump-ptx-path='{tmp_path / 'kernel'}' "
        pipeline = pipeline.replace("cute-to-nvvm{", "cute-to-nvvm{" + option, 1)
        return compile_fn(self, module, pipeline, *args, **kwargs)

    monkeypatch.setattr(Compiler, "compile", dump_ptx)
    compile_plain_toy.__wrapped__(PlainToy(TilePersistent("n"), mode="cursor"))
    (ptx,) = tmp_path.glob("*.ptx")
    text = ptx.read_text(errors="ignore")
    assert "%nctaid.y" in text and all(f"%ctaid.{a}" in text for a in "xyz")
    assert not re.search(r"^\s*(div|rem)\.", text, re.MULTILINE)


def test_tile_persistent_cap_is_runtime(requires_blackwell):
    """One compile serves every ``max_ctas`` (here, the SM count's stand-in) and every space."""
    op = PlainToy(TilePersistent("n"), mode="cursor")
    _plain_hits(op, (3, 9, 2), max_ctas=4)
    before = compile_plain_toy.cache_info()
    for max_ctas, ext in [(1, (3, 9, 2)), (3, (2, 17, 3)), (148, (3, 9, 2)), (7, (1, 1, 1))]:
        expect = torch.ones(math.prod(ext), dtype=torch.int32)
        assert torch.equal(_plain_hits(op, ext, max_ctas=max_ctas), expect), (max_ctas, ext)
    after = compile_plain_toy.cache_info()
    assert after.currsize == before.currsize and after.misses == before.misses


def test_tile_persistent_assignment_keeps_a_folded_grid_order():
    """``TilePersistent("n")`` over ``(pairs, chunks, heads)`` lists its CTAs with the pair
    fastest, then the worker, then the head: CTA ``x + pairs * workers * head`` with
    ``x = worker * pairs + pair`` runs chunks ``worker, worker + workers, ...`` of that pair and
    head."""
    pairs, chunks, heads, workers = 6, 11, 3, 4
    rows = WorkStream(schedule=TilePersistent("n")).assignment(
        (pairs, chunks, heads), max_ctas=workers
    )
    expected = []
    for head in range(heads):
        for x in range(workers * pairs):
            worker, pair = divmod(x, pairs)
            expected.append([(c, (pair, c, head)) for c in range(worker, chunks, workers)])
    assert rows == expected


def test_limit_rejections():
    table = object()  # never read: the checks run first

    class Bad:
        pass

    with pytest.raises(TypeError, match="takes a Limit"):
        WorkStream(schedule=StaticPersistent()).bind((2, 2, 1), None, 4, limit=Bad())
    for schedule in (AtomicPersistent(), ClusterLaunchControl()):
        with pytest.raises(ValueError, match="local schedules only"):
            WorkStream(schedule=schedule).bind((2, 2, 1), table, 4, limit=Limit())
    with pytest.raises(ValueError, match="positive int"):
        LastEntry(table, scale=0)


# The persistent GEMM: one argument swaps the schedule.


@pytest.mark.parametrize("cluster", [None, CtaPair()], ids=["single", "pair"])
@pytest.mark.parametrize("case", list(CUSTOM))
def test_persistent_gemm_swaps_schedules(requires_blackwell, case, cluster):
    for m, n, k, max_ctas in [(300, 520, 136, None), (1100, 1032, 200, 5)]:
        check_gemm(CUSTOM[case], m, n, k, max_ctas=max_ctas, cluster=cluster)


# A device-built work list consumed under programmatic dependent launch


def serialized_launches() -> bool:
    """Launches are serialized by ``CUDA_LAUNCH_BLOCKING`` or an injected CUDA tool."""
    return os.environ.get("CUDA_LAUNCH_BLOCKING") == "1" or bool(
        os.environ.get("CUDA_INJECTION64_PATH")
    )


def _packed_keep(m, n, li):
    return (m + 2 * n + 3 * li) % 3 != 0


class PackedToy(StreamToy):
    """``produce`` (one CTA) lets its dependents launch, waits ``delay`` ns, then compacts the
    items ``(m, n, l)`` of an ``em x en x el`` grid that ``_packed_keep`` accepts into
    ``table``/``length`` (atomics: the row order varies) and stamps its finish time. The
    consumer is launched with PDL right behind it on the same stream; its stream waits with
    ``PdlWait`` and maps rows through ``PackedWork``. Both roles count each item in
    ``hits[role, item]``; each consumer CTA stamps its entry time."""

    PRODUCER_THREADS = 128

    def __init__(self, schedule, cluster=None, mapping=None):
        super().__init__(schedule, mapping or PackedWork(), cluster, PdlWait())

    @cute.jit
    def visit(self, r: int, index, item, k, ctx):
        _hit(r, item, ctx)

    @cute.kernel
    def produce(
        self, table: cute.Tensor, length: cute.Tensor, stamp: cute.Tensor, em, en, el, delay
    ):
        cute.arch.griddepcontrol_launch_dependents()
        tidx, _, _ = cute.arch.thread_idx()
        start = cute.arch.globaltimer_lo()
        while cute.arch.globaltimer_lo() - start < delay:
            pass
        for i in cutlass.range(tidx, em * en * el, self.PRODUCER_THREADS, unroll=1):
            li = i // (em * en)
            n = (i - li * em * en) // em
            m = i - li * em * en - n * em
            if _packed_keep(m, n, li):
                row = cute.arch.atomic_add(length.iterator, Int32(1), scope="gpu")
                table[row, 0] = m
                table[row, 1] = n
                table[row, 2] = li
        cute.arch.sync_threads()
        if tidx == 0:
            stamp[0] = cute.arch.globaltimer_lo()

    @cute.kernel
    def kernel(self, hits: cute.Tensor, entry: cute.Tensor, em: Int32, en: Int32, bound):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        if tidx == 0:
            entry[bidx] = cute.arch.globaltimer_lo()
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace(hits=hits, em=em, en=en))

    @cute.jit
    def __call__(
        self,
        table: cute.Tensor,
        length: cute.Tensor,
        stamp: cute.Tensor,
        hits: cute.Tensor,
        entry: cute.Tensor,
        workspace: cute.Tensor,
        em: Int32,
        en: Int32,
        el: Int32,
        delay: Int32,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        self.produce.set_name_prefix(self.get_name() + "_produce")
        self.produce(table, length, stamp, em, en, el, delay).launch(
            grid=(1, 1, 1), block=(self.PRODUCER_THREADS, 1, 1), stream=stream
        )
        bound = self.work.bind((table, length), workspace, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(hits, entry, em, en, bound).launch(
            **self.launch_config(bound), stream=stream, use_pdl=True
        )


@jit_cache
def compile_packed_toy(op: PackedToy):
    i32 = cutlass.Int32
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(i32, (cute.sym_int(), 3)),
        *(make_fake_compact_tensor(i32, (cute.sym_int(),)) for _ in range(2)),
        make_fake_compact_tensor(i32, (2, cute.sym_int())),
        *(make_fake_compact_tensor(i32, (cute.sym_int(),)) for _ in range(2)),
        *(Int32(1) for _ in range(5)),
    )


@dataclass(frozen=True)
class CachedLength(PackedWork):
    """Cache the device length in decode state during ``start``.

    This is valid only if start runs after the readiness hook.
    """

    def start(self, space):
        _, length = space
        return (Int32(length[0]),)

    def decode(self, index, space, state):
        table, _ = space
        row = tuple(table[index, w] for w in range(self.words))
        return index < state[0], row, state

    def get_key(self) -> str:
        return "cached_" + super().get_key()


@pytest.mark.parametrize("mapping", [PackedWork(), CachedLength()], ids=["packed", "cached"])
@pytest.mark.parametrize("cluster", [SingleCta(), CtaPair()], ids=["single", "pair"])
@pytest.mark.parametrize("schedule", BUILTIN)
def test_device_built_list_under_pdl(requires_blackwell, schedule, cluster, mapping):
    """PDL consumers wait before reading the device-built work list.

    The consumer launches before the producer finishes, then waits in PdlWait. Both roles
    run every kept item once per CTA in its unit and nothing past the device length. Spare
    table rows point to item 0 to expose overreads. CachedLength reads length in start,
    which must follow the wait at every decoding site.
    """
    op = PackedToy(SCHEDULES[schedule], cluster, mapping)
    fn = compile_packed_toy(op)
    em, en, el = 7, 5, 3
    total = em * en * el
    capacity = total + 23
    want = torch.tensor(
        [
            cluster.size if _packed_keep(m, n, li) else 0
            for li in range(el)
            for n in range(en)
            for m in range(em)
        ],
        dtype=torch.int32,
    )
    ws, _ = op.work.launch_args("cuda")
    for max_ctas in (148, 6):
        # A shared GPU may delay the consumer past the producer's spin. Retry only overlap,
        # with a longer spin (0.2 to 3.2 ms); every launch must still produce correct results.
        for attempt in range(5):
            table = torch.zeros(capacity, 3, dtype=torch.int32, device="cuda")
            length = torch.zeros(1, dtype=torch.int32, device="cuda")
            stamp = torch.zeros(1, dtype=torch.int32, device="cuda")
            hits = torch.zeros(2, total, dtype=torch.int32, device="cuda")
            entry = torch.zeros(capacity * cluster.size, dtype=torch.int32, device="cuda")
            delay = 200_000 * 2**attempt
            fn(table, length, stamp, hits, entry, ws, em, en, el, delay, max_ctas)
            torch.cuda.synchronize()
            assert int(length) == int((want > 0).sum())
            assert torch.equal(hits[0].cpu(), want) and torch.equal(hits[1].cpu(), want), max_ctas
            assert ws[:2].tolist() == [0, 0]
            # Tools that serialize launches (compute-sanitizer, CUTracer, CUDA_LAUNCH_BLOCKING)
            # remove the overlap, not the correctness checks above.
            if serialized_launches():
                break
            launched = entry[entry != 0].cpu().long()
            early = ((launched - int(stamp)) + 2**31) % 2**32 - 2**31 < 0
            if early.any():
                break
        else:
            pytest.fail(
                f"the consumer did not overlap the producer in 5 attempts (max_ctas={max_ctas}); "
                "PDL not exercised"
            )


# Split work: (tile, split) items with K ranges


def _wide(value):
    return value if isinstance(value, int) else Int64(value)


@dataclass(frozen=True)
class SplitK(WorkMap):
    """Map indices to split-K items ``(m, n, l, split, begin, end)``.

    Tile ``index // splits`` follows Raster order; ``[begin, end)`` is the split's K-tile
    range. Empty ranges are dropped. The space is ``((m_tiles, n_tiles, l), splits, k_tiles)``
    with ``splits >= 1``. Splits write distinct partials; a later reduction must sum them.
    Neither this mapping nor close_claims performs the reduction.
    """

    words = 6
    skips = True

    def bind(self, space):
        extents, splits, k_tiles = space
        return Raster().bind(extents), splits, k_tiles

    def count64(self, space):
        extents, splits, _ = space
        return _wide(Raster().count64(extents)) * _wide(splits)

    def decode(self, index, space, state):
        extents, splits, k_tiles = space
        tile = index // splits
        split = index - tile * splits
        _, (m, n, li), _ = Raster().decode(tile, extents, None)
        begin = split * k_tiles // splits
        end = (split + 1) * k_tiles // splits
        return begin < end, (m, n, li, split, begin, end), state

    def get_key(self) -> str:
        return "split_k"


K_TILE = 4  # K elements per k tile in the split-K toy


class SplitKToy(StreamToy):
    """Role ``a`` sums ``x[tile, k]`` over its item's K range into ``partial[split, tile]``;
    role ``b`` counts each item in ``hits``."""

    def __init__(self, schedule, cluster=None):
        super().__init__(schedule, SplitK(), cluster)

    @cute.jit
    def visit(self, r: int, index, item, k, ctx):
        m, n, li, split, begin, end = item
        tile = (li * ctx.en + n) * ctx.em + m
        if cutlass.const_expr(r == 0):
            acc = Int32(0)
            for kk in cutlass.range(begin * K_TILE, end * K_TILE, unroll=1):
                acc += ctx.x[tile, kk]
            ctx.partial[split, tile] = acc
        else:
            cute.arch.atomic_add(ctx.hits.iterator + (split * ctx.x.shape[0] + tile), Int32(1))

    @cute.kernel
    def kernel(self, x, partial, hits, em: Int32, en: Int32, bound):
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace(x=x, partial=partial, hits=hits, em=em, en=en))

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        partial: cute.Tensor,
        hits: cute.Tensor,
        workspace: cute.Tensor,
        em: Int32,
        en: Int32,
        el: Int32,
        splits: Int32,
        k_tiles: Int32,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        bound = self.work.bind(((em, en, el), splits, k_tiles), workspace, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(x, partial, hits, em, en, bound).launch(
            **self.launch_config(bound), stream=stream
        )


@jit_cache
def compile_split_k_toy(op: SplitKToy):
    i32 = cutlass.Int32
    return compile_tvm_ffi(
        op,
        *(make_fake_compact_tensor(i32, (cute.sym_int(), cute.sym_int())) for _ in range(2)),
        *(make_fake_compact_tensor(i32, (cute.sym_int(),)) for _ in range(2)),
        *(Int32(1) for _ in range(6)),
    )


@pytest.mark.parametrize("cluster", [SingleCta(), CtaPair()], ids=["single", "pair"])
@pytest.mark.parametrize("schedule", BUILTIN)
def test_split_k_partials_reduce_on_the_host(requires_blackwell, schedule, cluster):
    """Every nonempty (tile, split) item writes its partial once per CTA of its unit; empty
    splits never run (their partials keep the sentinel); summing the written partials per tile
    gives the full K reduction."""
    op = SplitKToy(SCHEDULES[schedule], cluster)
    fn = compile_split_k_toy(op)
    ws, _ = op.work.launch_args("cuda")
    em, en, el = 5, 3, 2
    tiles = em * en * el
    sentinel = 2**30
    for splits, k_tiles, max_ctas in [(3, 7, 148), (5, 3, 4), (4, 4, 148)]:
        torch.manual_seed(splits)
        x = torch.randint(-100, 100, (tiles, k_tiles * K_TILE), dtype=torch.int32, device="cuda")
        partial = torch.full((splits, tiles), sentinel, dtype=torch.int32, device="cuda")
        hits = torch.zeros(splits * tiles, dtype=torch.int32, device="cuda")
        fn(x, partial, hits, ws, em, en, el, splits, k_tiles, max_ctas)
        torch.cuda.synchronize()
        nonempty = torch.tensor(
            [s * k_tiles // splits < (s + 1) * k_tiles // splits for s in range(splits)]
        )
        want_hits = (nonempty.view(-1, 1).int() * cluster.size).expand(splits, tiles)
        assert torch.equal(hits.view(splits, tiles).cpu(), want_hits)
        written = partial.cpu() != sentinel
        assert torch.equal(written, nonempty.view(-1, 1).expand(splits, tiles))
        reduced = torch.where(written, partial.cpu(), 0).sum(0)
        assert torch.equal(reduced, x.cpu().sum(1, dtype=torch.int32))
        assert ws[:2].tolist() == [0, 0]


def test_tile_persistent_units_fix_the_grid_and_stride():
    """``units`` launches exactly that many CTAs along the axis, reads no ``max_ctas`` and
    names itself in the compile key."""
    schedule = TilePersistent("n", units=4)
    assert not schedule.uses_max_ctas and schedule.get_key().endswith("_units4")
    rows = WorkStream(schedule=schedule).assignment((2, 3, 1))
    assert len(rows) == 2 * 4 * 1
    assert rows[:2] == [[(0, (0, 0, 0))], [(0, (1, 0, 0))]]  # CTA (x, y=0): chunk 0 only
    assert rows[-1] == []  # y=3 is past the 3 chunks
    for units in (0, -1, 2.0):
        with pytest.raises(ValueError, match="units must be a positive int"):
            TilePersistent("m", units=units)


def test_tile_persistent_assignment_follows_the_launch():
    """``assignment`` uses the launch's units: one row per CTA of the launch grid, every tile
    once, also for a subclass that changes whether ``max_ctas`` is read. It reads no launch
    resources."""

    class Uncapped(TilePersistent):
        uses_max_ctas = False  # one unit per axis coordinate

    class WithWorkspace(TilePersistent):
        uses_workspace = True

    space = (5, 4, 3)
    for schedule, max_ctas in [
        (TilePersistent("n"), 2),
        (TilePersistent("l", units=2), None),
        (Uncapped("m"), None),
    ]:
        stream = WorkStream(schedule=schedule)
        rows = stream.assignment(space, max_ctas)
        assert len(rows) == math.prod(stream.bind(space, max_ctas=max_ctas).grid())
        items = sorted(item for row in rows for _, item in row)
        assert items == sorted(itertools.product(*map(range, space)))
    reference = WorkStream(schedule=TilePersistent("m")).assignment(space, 2)
    assert WorkStream(schedule=WithWorkspace("m")).assignment(space, 2) == reference


def test_tile_persistent_rejections():
    with pytest.raises(ValueError, match="axis must be one of"):
        TilePersistent("k")
    for mapping in (Raster(swizzle=2), Raster(along="n"), SkipOdd()):
        with pytest.raises(ValueError, match="plain Raster"):
            WorkStream(mapping=mapping, schedule=TilePersistent("m"))
    for max_ctas in (4, Int32(4)):  # a static l past gridDim.z raises with any cap
        bound = WorkStream(schedule=TilePersistent("n")).bind((2, 3, 70000), None, max_ctas)
        with pytest.raises(ValueError, match="gridDim.y/z are limited"):
            bound.grid()
    with pytest.raises(ValueError, match="units on gridDim.y/z"):
        TilePersistent("n", units=70000)
    with pytest.raises(ValueError, match="drop decode"):
        WorkStream(schedule=TilePersistent("m")).assignment((2, 2, 1), 2, decode=lambda i: i)


def test_tile_persistent_launch_bounds_the_axis_not_the_space():
    """The Int32 launch check bounds the axis coordinate: a space of 3e9 items whose
    coordinates all fit launches."""
    bound = WorkStream(schedule=TilePersistent("n")).bind((16, 3_000_000, 64), None, 152)
    assert bound.grid() == (16, 152, 64)


@pytest.mark.parametrize(
    "mode, message",
    [
        ("count_then_advance", r"advance\(\) at .* on a cursor counted by trip_count\(\)"),
        ("advance_then_count", r"trip_count\(\) on .*: it already advances"),
        ("count_twice", r"trip_count\(\) on .*: it is already counted"),
        ("count_item_before_loop", r"item_at\(\) at .* must run directly in the body"),
        ("count_item_in_branch", r"item_at\(\) at .* must run directly in the body"),
    ],
)
def test_counted_loop_rejects_misuse(mode, message):
    """A cursor loops either with advance() or over one trip_count(), reading items with
    item_at() directly in that loop's body; anything else raises when traced."""
    with pytest.raises(ProtocolError, match=message):
        compile_plain_toy.__wrapped__(PlainToy(StaticPersistent(), mode=mode))


def test_counted_loop_rejections():
    """trip_count() needs a fixed stride and an unguarded, stateless decode."""
    for schedule, mapping, why in [
        (OneShot(grid="1d"), None, "has no fixed stride"),
        (StaticPersistent(), SkipOdd(), "does not decode every index unguarded"),
    ]:
        with pytest.raises(ProtocolError, match=why):
            compile_plain_toy.__wrapped__(PlainToy(schedule, mapping, mode="counted"))


class CountedRoleToy(StreamToy):
    """Roles ``a`` (warp 0) and ``b`` (warps 1-2) each loop over ``trip_count()`` and count every
    item in ``hits[role, tile]``."""

    def __init__(self, schedule, mode: str = "ok"):
        super().__init__(schedule)
        self.mode = mode

    @cute.jit
    def _counted(self, r: int, work, ctx):
        trips = work.trip_count()
        if cutlass.const_expr(self.mode == "no_loop" and r == 1):
            return work
        for k in cutlass.range(trips, unroll=2):
            item = work.item_at(k)
            if cute.arch.lane_idx() == 0 and cute.arch.warp_idx() == r:
                _hit(r, item, ctx)
        return work

    @role(warps=0)
    @cute.jit
    def a(self, work: WorkCursor, ctx):
        return self._counted(0, work, ctx)

    @role(warps=range(1, 3))
    @cute.jit
    def b(self, work: WorkCursor, ctx):
        return self._counted(1, work, ctx)

    @cute.kernel
    def kernel(self, hits: cute.Tensor, em: Int32, en: Int32, bound):
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace(hits=hits, em=em, en=en))

    @cute.jit
    def __call__(
        self, hits: cute.Tensor, em: Int32, en: Int32, el: Int32, max_ctas: Int32, stream
    ):
        bound = self.work.bind((em, en, el), None, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(hits, em, en, bound).launch(**self.launch_config(bound), stream=stream)


@pytest.mark.parametrize(
    "schedule", [StaticPersistent(), TilePersistent("n"), TilePersistent("m", units=3)]
)
def test_counted_loop_in_roles(requires_blackwell, schedule):
    """Every role of a warp-specialized kernel runs each item once through a counted loop."""
    fn = compile_tvm_ffi(
        CountedRoleToy(schedule),
        make_fake_compact_tensor(cutlass.Int32, (2, cute.sym_int())),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        name=CountedRoleToy(schedule).get_name(),
    )
    ext = (5, 7, 2)
    hits = torch.zeros(2, math.prod(ext) + 64, dtype=torch.int32, device="cuda")
    fn(hits, *ext, 4)
    torch.cuda.synchronize()
    expect = torch.tensor([1] * math.prod(ext) + [0] * 64, dtype=torch.int32)
    assert torch.equal(hits[0].cpu(), expect) and torch.equal(hits[1].cpu(), expect)


def test_counted_role_must_loop(requires_blackwell):
    """A role's counted cursor whose items are never read with item_at() raises at role exit."""
    op = CountedRoleToy(StaticPersistent(), mode="no_loop")
    hits = make_fake_compact_tensor(cutlass.Int32, (2, cute.sym_int()))
    with pytest.raises(ProtocolError, match=r"trip_count\(\) at .* is never looped over"):
        compile_tvm_ffi(op, hits, Int32(1), Int32(1), Int32(1), Int32(1), name=op.get_name())
