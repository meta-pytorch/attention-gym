import itertools
import math
import random
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import Int32, cute

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.testing.warpdrive import INVALID_LAUNCH
from attn_gym.warpdrive import (
    AtomicPersistent,
    Cluster,
    ClusterLaunchControl,
    CtaPair,
    OneShot,
    PdlWait,
    ProtocolError,
    Raster,
    SingleCta,
    StaticPersistent,
    WarpSpecialized,
    WorkCursor,
    WorkMap,
    WorkStream,
    role,
    scheduling,
)
from attn_gym.warpdrive.examples.persistent_gemm import PersistentGemm
from attn_gym.warpdrive.scheduling import tile_grid_dims

SCHEDULES = {
    "oneshot": OneShot(),
    "static": StaticPersistent(),
    "atomic": AtomicPersistent(stages=2),
    "atomic1": AtomicPersistent(stages=1),
    "clc": ClusterLaunchControl(stages=2),
    "clc1": ClusterLaunchControl(stages=1),
}


# --------------------------------------------------------------------------- #
# CPU: mappings
# --------------------------------------------------------------------------- #


def _all_coords(mapping, ext):
    total = ext[0] * ext[1] * ext[2]
    return [mapping.decode(i, ext, None)[1] for i in range(total)]


EXTENTS = [(1, 1, 1), (1, 7, 1), (7, 1, 2), (5, 3, 1), (16, 16, 1), (13, 29, 3), (3, 100, 1)]
MAPPINGS = [
    Raster(),
    Raster(along="n"),
    Raster(swizzle=2),
    Raster(swizzle=8),
    Raster(along="n", swizzle=3),
    Raster(swizzle=8, reverse=("m",)),
    Raster(along="n", swizzle=5, reverse=("m", "n", "l")),
]


@pytest.mark.parametrize("ext", EXTENTS)
@pytest.mark.parametrize("mapping", MAPPINGS, ids=lambda o: o.get_key())
def test_raster_is_a_bijection(mapping, ext):
    coords = _all_coords(mapping, ext)
    assert sorted(coords) == sorted(itertools.product(*(range(e) for e in ext)))
    assert len(set(coords)) == len(coords)


@pytest.mark.parametrize("ext", [(13, 29, 1), (8, 64, 2), (40, 3, 1)])
def test_raster_along_and_l_slowest(ext):
    em, en, _ = ext
    for i, (m, n, li) in enumerate(_all_coords(Raster(), ext)):
        assert (m, n, li) == (i % em, (i // em) % en, i // (em * en))
    for i, (m, n, li) in enumerate(_all_coords(Raster(along="n"), ext)):
        assert (m, n, li) == ((i // en) % em, i % en, i // (em * en))


@pytest.mark.parametrize("swizzle", [2, 4, 8])
@pytest.mark.parametrize("ext", [(13, 29, 1), (32, 32, 1), (5, 17, 2)])
def test_raster_swizzle_bands(swizzle, ext):
    """Indices fill one band of ``swizzle`` n-columns before the next; within a band the band's
    columns are innermost, so any window of ``k * swizzle`` indices inside a band touches at most
    ``swizzle`` columns and ``k`` (+1) rows."""
    em, en, _ = ext
    coords = _all_coords(Raster(swizzle=swizzle), ext)
    for li in range(ext[2]):
        layer = coords[li * em * en : (li + 1) * em * en]
        bands = [n // swizzle for _, n, _ in layer]
        assert bands == sorted(bands)
        for start in range(0, en, swizzle):
            width = min(swizzle, en - start)
            band = [(m, n) for m, n, _ in layer if start <= n < start + width]
            for j, (m, n) in enumerate(band):
                assert (m, n) == (j // width, start + j % width)


def test_raster_lpt_reverses_m():
    ext = (6, 4, 2)
    plain = _all_coords(Raster(swizzle=2), ext)
    lpt = _all_coords(Raster(swizzle=2, reverse="m"), ext)
    assert [(ext[0] - 1 - m, n, li) for m, n, li in plain] == lpt
    assert lpt[0][0] == ext[0] - 1  # the longest query block goes first


def test_raster_validation():
    with pytest.raises(ValueError):
        Raster(along="k")
    with pytest.raises(ValueError):
        Raster(swizzle=0)
    with pytest.raises(ValueError):
        Raster(reverse=("k",))


# Empty work spaces: a zero extent, or a negative one (treated as empty).
EMPTY_SPACES = [
    (0, 5, 3),
    (5, 0, 3),
    (5, 3, 0),
    (0, 0, 0),
    (0, 70000, 1),  # n_tiles over gridDim.y: the OneShot tile grid would fall back to 1-D
    (-2, -3, 1),  # two negative extents multiply to a positive count
    (-1, 4, 2),
    (3, -5, -1),
    (65536, 65536, 0),  # m * n wraps to 0 in Int32: decode must not divide by it
]


@pytest.mark.parametrize("space", EMPTY_SPACES, ids=str)
@pytest.mark.parametrize("mapping", MAPPINGS, ids=lambda o: o.get_key())
def test_bound_empty_space_has_no_items_and_decodes(mapping, space):
    """Host Int32 arithmetic folds like the device's (and raises on division by zero): a bound
    empty space counts no items, and decode, which runs past the end (``decodes_any_index``),
    never divides by zero."""
    bound = mapping.bind(space)
    assert int(mapping.count64(bound)) == 0
    for index in range(4):
        mapping.decode(Int32(index), bound, None)


@pytest.mark.parametrize(
    "space", [(5, 3, 2), (2**31 - 1, 1, 1), (65537, 65537, 1), (2**31 - 1, 2**31 - 1, 4)]
)
def test_count64_is_exact_within_int32_and_never_wraps(space):
    exact = space[0] * space[1] * space[2]
    size = int(Raster().count64(Raster().bind(space)))
    assert size == exact if exact <= 2**31 - 1 else size > 2**31 - 1


@pytest.mark.parametrize(
    "schedule, units, cs, largest, too_large",
    [
        # Static/atomic: the terminal index total + units - 1.
        (StaticPersistent(), 148, 1, (2**31 - 148, 1, 1), (2**31 - 147, 1, 1)),
        (AtomicPersistent(), 74, 2, (2**31 - 74, 1, 1), (2**31 - 73, 1, 1)),
        # OneShot/ClusterLaunchControl: the 1-D grid, total * cluster size.
        (OneShot(), 2**31 - 1, 1, (2**31 - 1, 1, 1), (2**30, 2, 1)),
        (ClusterLaunchControl(), 2**30 - 1, 2, (2**30 - 1, 1, 1), (2**29, 2, 1)),
    ],
    ids=["static", "atomic_pair", "oneshot", "clc_pair"],
)
def test_launch_admits_spaces_exactly_up_to_int32(schedule, units, cs, largest, too_large):
    """The largest admitted space leaves every index and grid extent at most INT32_MAX; one
    more item would not."""
    for space, admitted in [(largest, True), (too_large, False)]:
        count = Raster().count64(Raster().bind(space))
        assert (int(schedule.index_limit(count, units, cs)) <= 2**31 - 1) == admitted, space


@pytest.mark.parametrize("ext", EXTENTS)
def test_bind_keeps_nonempty_spaces(ext):
    assert tuple(int(e) for e in Raster().bind(ext)) == ext


@pytest.mark.parametrize("mapping", [Raster(swizzle=8), Raster(along="n", swizzle=8)])
def test_swizzled_decode_does_not_overflow_int32(mapping):
    """``swizzle * along-extent`` exceeds Int32 although the item count fits; Int32 decoding
    still matches exact Python-int decoding."""
    ext = (2**29, 3, 1) if mapping.along == "m" else (3, 2**29, 1)
    bound = mapping.bind(ext)
    total = 3 * 2**29
    for index in [0, 5, 2**29 + 7, total - 2**29 - 1, total - 1]:
        got = tuple(int(w) for w in mapping.decode(Int32(index), bound, None)[1])
        assert got == mapping.decode(index, ext, None)[1]


def test_stream_keys_exclude_runtime_values():
    keys = {
        WorkStream(mapping=Raster(swizzle=8), schedule=p).get_key() for p in SCHEDULES.values()
    }
    assert len(keys) == len(SCHEDULES)
    assert WorkStream().get_key() == "oneshot0_m1"


# --------------------------------------------------------------------------- #
# A toy kernel: every role logs, per CTA, the sequence of work items it saw.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SkipOdd(Raster):
    """Raster that drops tiles whose m + n is odd (an empty-item test mapping)."""

    skips = True

    def decode(self, index, space, state):
        _, item, state = super().decode(index, space, state)
        return self.kept(item), item, state

    def kept(self, item):
        m, n, _ = item
        return (m + n) % 2 == 0

    def get_key(self) -> str:
        return "skipodd_" + super().get_key()


SWIZZLED = Raster(along="n", swizzle=3, reverse=("m", "l"))


@cute.jit
def _trap_outside(index, count):
    if (index < 0) | (index >= count):
        cute.arch.inline_ptx("trap;", read_only_args=[])


@dataclass(frozen=True)
class CheckedRaster(WorkMap):
    """A default ``WorkMap`` (no ``decodes_any_index``) over a swizzled Raster space whose decode
    traps on an index outside ``[0, count)``: the layer must never decode past the end, on an
    empty space, or after a dropped item. ``drop_odd`` drops tiles whose m + n is odd."""

    drop_odd: bool = False
    words = 3

    @property
    def skips(self) -> bool:
        return self.drop_odd

    def bind(self, space):
        return Raster().bind(space)

    def count64(self, space):
        return Raster().count64(space)

    def decode(self, index, space, state):
        em, en, el = space
        _trap_outside(index, em * en * el)
        _, item, state = SWIZZLED.decode(index, space, state)
        return self.kept(item), item, state

    def kept(self, item):
        if not self.drop_odd:
            return True
        m, n, _ = item
        return (m + n) % 2 == 0

    def get_key(self) -> str:
        return f"checked{int(self.drop_odd)}"


class ScheduleToy(WarpSpecialized):
    """Roles ``a`` (warp 0) and ``b`` (warps 1-2) write each item's linear tile id into
    ``log[role, cta, k]`` (k = the CTA's k-th item) and bump ``hits[role, tile]``."""

    MAX_ITEMS = 64

    def __init__(self, schedule, mapping=None, mode: str = "ok", regs: bool = False, ready=None):
        self.mode = mode
        self.work = WorkStream(
            mapping=mapping or Raster(swizzle=3), schedule=schedule, ready=ready
        )
        self.regs = regs

    def get_name(self) -> str:
        return f"warpdrive_sched_toy_{self.work.get_key()}_{self.mode}"

    @cute.jit
    def _visit(self, r: int, work, count, ctx):
        m, n, li = work.item
        tile = (li * ctx.en + n) * ctx.em + m
        # Out-of-range coordinates (a misread launch grid) never count, even when their linear
        # id happens to land on a real tile.
        in_range = (m < ctx.em) & (n < ctx.en)
        if cute.arch.lane_idx() == 0 and cute.arch.warp_idx() == r and in_range:
            bidx, _, _ = cute.arch.block_idx()
            if count < self.MAX_ITEMS:
                ctx.log[r, bidx, count] = tile
            cute.arch.atomic_add(ctx.hits.iterator + (r * ctx.hits.shape[1] + tile), Int32(1))

    @role(warps=0)
    @cute.jit
    def a(self, work: WorkCursor, ctx):
        count = Int32(0)
        while work.valid:
            self._visit(0, work, count, ctx)
            count += 1
            if cutlass.const_expr(self.mode == "conditional_advance"):
                if count < self.MAX_ITEMS:  # a runtime predicate: some iterations skip it
                    work.advance()
            else:
                work.advance()
            if cutlass.const_expr(self.mode == "double_advance"):
                work.advance()
        return work

    @role(warps=range(1, 3))
    @cute.jit
    def b(self, work: WorkCursor, ctx):
        count = Int32(0)
        if cutlass.const_expr(self.mode == "no_advance"):
            self._visit(1, work, count, ctx)
        else:
            while work.valid:
                self._visit(1, work, count, ctx)
                count += 1
                work.advance()
        return work

    def space(self, em, en, el):
        """The space bound in the entry point (a test mapping may need more than extents)."""
        return (em, en, el)

    @cute.kernel
    def kernel(self, log: cute.Tensor, hits: cute.Tensor, em: Int32, en: Int32, bound):
        smem = self.begin_roles(work=bound)
        # ~160 KB of SMEM: one CTA per SM, so CLC has unlaunched CTAs to cancel.
        smem.allocate_array(cutlass.Int32, 40000)
        # Extents from the arguments, not bound.space: external mappings bind other spaces.
        self.run_roles(SimpleNamespace(log=log, hits=hits, em=em, en=en))

    @cute.jit
    def __call__(
        self,
        log: cute.Tensor,
        hits: cute.Tensor,
        workspace: cute.Tensor,
        em: Int32,
        en: Int32,
        el: Int32,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        bound = self.work.bind(self.space(em, en, el), workspace, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        # "1d" modes launch a hand-built (total, 1, 1) grid instead of the schedule's.
        grid = (
            (bound.total, 1, 1) if cutlass.const_expr(self.mode.endswith("1d")) else bound.grid()
        )
        self.kernel(log, hits, em, en, bound).launch(
            grid=grid, block=(self.num_threads, 1, 1), stream=stream
        )


@jit_cache
def compile_toy(op: ScheduleToy):
    i32 = cutlass.Int32
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(i32, (2, cute.sym_int(), ScheduleToy.MAX_ITEMS)),
        make_fake_compact_tensor(i32, (2, cute.sym_int())),
        make_fake_compact_tensor(i32, (cute.sym_int(),)),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        name=op.get_name(),
    )


def run_toy(op, ext, max_ctas, reps=1, items=None):
    """``items``: the mapping's item count when it differs from the tile count (padding)."""
    em, en, el = ext
    total = em * en * el
    count = total if items is None else items
    grid = max(min(count, max_ctas), 1) if op.work.schedule.uses_max_ctas else max(count, 1)
    log = torch.full((2, grid, ScheduleToy.MAX_ITEMS), -1, dtype=torch.int32, device="cuda")
    hits = torch.zeros(2, max(total, 1), dtype=torch.int32, device="cuda")
    ws, _ = op.work.launch_args("cuda")
    fn = compile_toy(op)
    for _ in range(reps):
        log.fill_(-1)
        hits.zero_()
        fn(log, hits, ws, em, en, el, max_ctas)
    torch.cuda.synchronize()
    return log.cpu(), hits.cpu()[:, :total], ws.cpu()


TOY_CASES = [
    ((1, 1, 1), 148),  # one tile
    ((5, 3, 1), 148),  # tiles < SMs
    ((13, 29, 3), 148),  # tiles >> SMs
    ((40, 40, 1), 7),  # capped grid: ~229 tiles per CTA for persistent schedules
]


@pytest.mark.parametrize("ext,max_ctas", TOY_CASES)
@pytest.mark.parametrize("schedule", list(SCHEDULES), ids=str)
def test_every_role_sees_every_tile_once(requires_blackwell, schedule, ext, max_ctas):
    op = ScheduleToy(SCHEDULES[schedule])
    log, hits, ws = run_toy(op, ext, max_ctas, reps=3)
    total = ext[0] * ext[1] * ext[2]
    assert (hits == 1).all(), f"tiles missed or repeated: {hits}"
    # Both roles of a CTA see the same sequence (for the first MAX_ITEMS items).
    assert torch.equal(log[0], log[1])
    if (
        SCHEDULES[schedule].persistent
        and total > max_ctas
        and not isinstance(op.work.schedule, ClusterLaunchControl)
    ):
        assert (log[0][:, 1] >= 0).all(), "persistent CTAs should run several tiles"
    if isinstance(op.work.schedule, ClusterLaunchControl) and total > 4 * 148:
        assert (log[0][:, 1] >= 0).any(), "no CLC cancellation succeeded"
    if schedule == "static":
        grid = min(total, max_ctas)
        mapping = op.work.mapping
        for b in range(grid):
            for k in range(min(ScheduleToy.MAX_ITEMS, len(range(b, total, grid)))):
                _, (m, n, li), _ = mapping.decode(b + k * grid, ext, None)
                assert log[0, b, k] == (li * ext[1] + n) * ext[0] + m
    assert ws[:2].tolist() == [0, 0], "atomic counter must self-reset"


@pytest.mark.parametrize(
    "space, cs, grid",
    [
        ((5, 3, 2), 1, (5, 3, 2)),
        ((5, 3, 2), 2, (10, 3, 2)),
        ((0, 0, 1), 1, (1, 1, 1)),  # empty problem: one CTA that sees no item
        ((1000, 0, 1), 1, (1, 1, 1)),  # not one idle CTA per m tile
        ((7, -3, 2), 2, (2, 1, 1)),  # negative extents are empty: one cluster
        ((7, 65535, 65535), 1, (7, 65535, 65535)),
        ((3, 65536, 1), 1, (3 * 65536, 1, 1)),  # n_tiles over gridDim.y: 1-D fallback
        ((3, 2, 65536), 2, (2 * 3 * 2 * 65536, 1, 1)),  # l over gridDim.z
    ],
)
def test_tile_grid_dims_respect_the_grid_limit(space, cs, grid):
    assert tile_grid_dims(space, cs) == grid


@pytest.mark.parametrize("launch", ["schedule", "1d"])
@pytest.mark.parametrize("ext", [(1, 1, 1), (5, 3, 1), (13, 29, 3), (1, 65536, 1)])
def test_oneshot_launches_the_tile_grid(requires_blackwell, ext, launch):
    """OneShot over plain Raster() launches (m_tiles, n_tiles, l) CTAs and reads each item from
    blockIdx, or a 1-D grid it decodes once n_tiles exceeds gridDim.y. A hand-built 1-D launch
    is decoded the same way; every tile still reaches every role once."""
    mode = "tile_grid_1d" if launch == "1d" else "tile_grid"
    op = ScheduleToy(OneShot(), mapping=Raster(), mode=mode)
    _, hits, _ = run_toy(op, ext, 148, reps=2)
    assert (hits == 1).all(), f"tiles missed or repeated: {hits}"


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("schedule", ["oneshot", "static", "atomic", "clc"])
def test_skip_hook_hides_items_from_every_role(requires_blackwell, schedule, ready):
    """Skip (and the readiness hook, a no-op wait outside PDL) in local and broadcast mode."""
    op = ScheduleToy(
        SCHEDULES[schedule],
        mapping=SkipOdd(swizzle=2),
        mode="skip",
        ready=PdlWait() if ready else None,
    )
    ext = (9, 7, 2)
    log, hits, _ = run_toy(op, ext, 16, reps=2)
    assert torch.equal(log[0], log[1])
    want = torch.tensor(
        [
            0 if (m + n) % 2 else 1
            for li in range(ext[2])
            for n in range(ext[1])
            for m in range(ext[0])
        ],
        dtype=torch.int32,
    )
    assert torch.equal(hits[0], want) and torch.equal(hits[1], want)


@dataclass(frozen=True)
class SkipAll(SkipOdd):
    def kept(self, item):
        m, _, _ = item
        return m < 0


@pytest.mark.parametrize("schedule", ["oneshot", "static", "atomic", "clc"])
def test_all_items_skipped_terminates(requires_blackwell, schedule):
    op = ScheduleToy(SCHEDULES[schedule], mapping=SkipAll(), mode="skipall")
    _, hits, ws = run_toy(op, (5, 4, 1), 3, reps=2)
    assert (hits == 0).all() and ws[:2].tolist() == [0, 0]


# --------------------------------------------------------------------------- #
# WorkStream.assignment: the host's view of which CTA runs which items
# --------------------------------------------------------------------------- #


def test_static_assignment_strides_by_the_grid():
    """CTA b of min(total, max_ctas) runs indices b, b + grid, ... in Raster order."""
    rows = WorkStream(schedule=StaticPersistent()).assignment((3, 2, 1), max_ctas=4)
    assert rows == [
        [(0, (0, 0, 0)), (4, (1, 1, 0))],
        [(1, (1, 0, 0)), (5, (2, 1, 0))],
        [(2, (2, 0, 0))],
        [(3, (0, 1, 0))],
    ]


def test_oneshot_assignment_runs_one_item_per_cta():
    rows = WorkStream(mapping=Raster(along="n")).assignment((2, 3, 2))
    tiles = itertools.product(range(2), range(2), range(3))  # (l, m, n): n fastest
    assert rows == [[(i, (m, n, li))] for i, (li, m, n) in enumerate(tiles)]
    assert WorkStream().assignment((0, 4, 1)) == [[]]  # an empty space: one idle CTA


def test_cluster_assignment_gives_every_cta_of_a_unit_its_items():
    """max_ctas counts CTAs: 6 CTAs are 3 pairs, and pair u runs u, u + 3, ..."""
    op = PersistentGemm(cluster=CtaPair())
    rows = op.work.assignment((4, 3, 1), max_ctas=6)
    assert [[i for i, _ in row] for row in rows] == [
        [u, u + 3, u + 6, u + 9] for u in (0, 0, 1, 1, 2, 2)
    ]
    assert all(rows[2 * u] == rows[2 * u + 1] for u in range(3))


def test_assignment_leaves_out_dropped_items():
    """A dropped item leaves its CTA's list and no other: the schedule only sees indices. Only
    a pure (``decodes_any_index``) mapping decodes itself on the host; another one that drops
    items needs its host counterpart (``decode``)."""
    ext, stride = (5, 3, 1), 4
    tiles = [(m, n, 0) for n in range(3) for m in range(5)]
    stream = WorkStream(mapping=SkipOdd(), schedule=StaticPersistent())
    with pytest.raises(ValueError, match="pass its host counterpart as decode"):
        stream.assignment(ext, stride)
    rows = stream.assignment(ext, stride, decode=lambda i: (sum(tiles[i]) % 2 == 0, tiles[i]))
    assert rows == [
        [(i, tiles[i]) for i in range(b, 15, stride) if sum(tiles[i]) % 2 == 0]
        for b in range(stride)
    ]


def test_assignment_needs_a_local_schedule_and_its_cap():
    for schedule in (AtomicPersistent(), ClusterLaunchControl()):
        with pytest.raises(ValueError, match="claims indices at run time"):
            WorkStream(schedule=schedule).assignment((2, 2, 1), max_ctas=4)
    with pytest.raises(ValueError, match="needs max_ctas"):
        WorkStream(schedule=StaticPersistent()).assignment((2, 2, 1))
    with pytest.raises(ValueError, match="exceed what a launch indexes in Int32"):
        WorkStream().assignment((2**16, 2**15, 1))


@pytest.mark.parametrize(
    "schedule,mapping,mode",
    [
        (StaticPersistent(), None, "ok"),
        (OneShot(), None, "ok"),
        (StaticPersistent(), SkipOdd(swizzle=2), "skip"),
        (OneShot(), SkipOdd(swizzle=2), "skip"),
    ],
    ids=["static", "oneshot", "static_skip", "oneshot_skip"],
)
def test_assignment_matches_the_device(requires_blackwell, schedule, mapping, mode):
    """Each CTA logs the tiles it ran, in order; the host assignment lists the same."""
    op = ScheduleToy(schedule, mapping=mapping, mode=mode)
    ext, max_ctas = (9, 7, 2), 16
    log, _, _ = run_toy(op, ext, max_ctas)
    decode = None
    if mapping is not None:  # SkipOdd's decode is plain arithmetic: its own host counterpart
        decode = lambda index: mapping.decode(index, ext, None)[:2]
    rows = op.work.assignment(ext, max_ctas, decode)
    assert len(rows) == log.shape[1]
    for cta, row in enumerate(rows):
        want = [(li * ext[1] + n) * ext[0] + m for _, (m, n, li) in row]
        got = log[0, cta].tolist()
        assert got[: len(want)] == want and set(got[len(want) :]) <= {-1}, cta


class CountToy(WarpSpecialized):
    """Roles ``a`` (warp 0) and ``b`` (warps 1-2), in every CTA of the cluster, count the work
    items they see in ``count[role]``."""

    def __init__(self, schedule, mapping, cluster):
        self.cluster = cluster
        self.work = WorkStream(mapping=mapping, schedule=schedule)

    def get_name(self) -> str:
        return f"warpdrive_count_toy_{self.work.get_key()}_{self.cluster.get_key()}"

    @cute.jit
    def _count(self, r: int, ctx):
        if cute.arch.lane_idx() == 0 and cute.arch.warp_idx() == r:
            cute.arch.atomic_add(ctx.count.iterator + r, Int32(1))

    @role(warps=0)
    @cute.jit
    def a(self, work: WorkCursor, ctx):
        while work.valid:
            self._count(0, ctx)
            work.advance()
        return work

    @role(warps=range(1, 3))
    @cute.jit
    def b(self, work: WorkCursor, ctx):
        while work.valid:
            self._count(1, ctx)
            work.advance()
        return work

    def space(self, em, en, el):
        return (em, en, el)

    @cute.kernel
    def kernel(self, count: cute.Tensor, bound):
        self.begin_roles(work=bound)
        self.run_roles(SimpleNamespace(count=count))

    @cute.jit
    def __call__(
        self,
        count: cute.Tensor,
        workspace: cute.Tensor,
        em: Int32,
        en: Int32,
        el: Int32,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        bound = self.work.bind(self.space(em, en, el), workspace, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(count, bound).launch(**self.launch_config(bound), stream=stream)


@jit_cache
def compile_count_toy(op: CountToy):
    count, ws = (make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),)) for _ in range(2))
    return compile_tvm_ffi(op, count, ws, Int32(1), Int32(1), Int32(1), Int32(1))


COUNT_CASES = {
    "oneshot_tile_grid": (OneShot(), Raster()),
    "oneshot_1d": (OneShot(), SWIZZLED),
    "static": (StaticPersistent(), SWIZZLED),
    "atomic": (AtomicPersistent(stages=2), SWIZZLED),
    "clc": (ClusterLaunchControl(stages=2), SWIZZLED),
    # Skip hooks: local skipping, and the scheduler publishing the first item itself.
    "static_skip": (StaticPersistent(), SkipOdd(swizzle=2)),
    "clc_skip": (ClusterLaunchControl(stages=2), SkipOdd(swizzle=2)),
    # A default WorkMap traps when decoded outside [0, count): no schedule decodes past the end.
    "oneshot_checked": (OneShot(), CheckedRaster()),
    "static_checked": (StaticPersistent(), CheckedRaster()),
    "atomic_checked": (AtomicPersistent(stages=2), CheckedRaster()),
    "clc_checked": (ClusterLaunchControl(stages=2), CheckedRaster()),
    "static_checked_skip": (StaticPersistent(), CheckedRaster(drop_odd=True)),
    "atomic_checked_skip": (AtomicPersistent(stages=1), CheckedRaster(drop_odd=True)),
    "clc_checked_skip": (ClusterLaunchControl(stages=2), CheckedRaster(drop_odd=True)),
}
CLUSTERS = [SingleCta(), CtaPair(), Cluster(2, 2, pair=True)]
CLUSTER_IDS = ["single", "pair", "2x2"]


@dataclass(frozen=True)
class UndeclaredDrop(CheckedRaster):
    def kept(self, item):
        m, _, _ = item
        return m >= 0  # a runtime keep from a mapping that does not declare skips


@dataclass(frozen=True)
class ShortDecode(CheckedRaster):
    def decode(self, index, space, state):
        _, (m, n, _), state = super().decode(index, space, state)
        return True, (m, n), state  # two words for a three-word mapping


def test_decode_word_count_is_checked(requires_blackwell):
    op = CountToy(StaticPersistent(), ShortDecode(), SingleCta())
    with pytest.raises(Exception, match="returned 2 words; its words is 3"):
        compile_count_toy(op)


@dataclass(frozen=True)
class Counter:
    count: Int32


@dataclass(frozen=True)
class BadState(CheckedRaster):
    """A stateful mapping whose state breaks a carry rule (``kind``)."""

    kind: str = "dataclass"

    def start(self, space):
        match self.kind:
            case "dataclass":
                return Counter(Int32(0))
            case "python_int":
                return (0,)
            case "list":
                return ([Int32(0), 0],)
        return (Int32(0),)

    def decode(self, index, space, state):
        keep, words, state = super().decode(index, space, state)
        match self.kind:
            case "grows":
                return keep, words, (*state, Int32(1))
            case "retyped":
                return keep, words, (cutlass.Int64(state[0]),)
            case "two_values":
                return keep, words
        return keep, words, state


@pytest.mark.parametrize(
    "kind, message",
    [
        ("dataclass", "use a NamedTuple"),
        ("python_int", r"Int32\(0\), not 0"),
        ("list", "use a tuple or NamedTuple"),
        ("grows", "another structure or type"),
        ("retyped", "another structure or type"),
        ("two_values", r"must return \(keep, words, state\)"),
    ],
)
def test_decode_state_must_be_carriable(requires_blackwell, kind, message):
    """The cursor carries the state from item to item as flat DSL values: a container whose
    fields would not be carried, a Python number, or a state that changes shape is a trace-time
    error naming the rule."""
    op = CountToy(StaticPersistent(), BadState(kind=kind), SingleCta())
    with pytest.raises(Exception, match=message):
        compile_count_toy(op)


def test_decode_takes_a_state():
    """A mapping written for the stateless signature fails when it is defined."""
    with pytest.raises(TypeError, match=r"takes \(self, index, space, state\)"):

        class Stateless(WorkMap):
            def decode(self, index, space):
                return True, (index,)


def test_overriding_decode_drops_the_any_index_opt_in():
    """A Raster subclass that replaces decode (it might read a table) is decoded only for valid
    indices unless it opts in again; one that keeps Raster's decode keeps the opt-in."""

    class TableRaster(Raster):
        def decode(self, index, space, state):
            return super().decode(index, space, state)

    class OptedIn(TableRaster):
        decodes_any_index = True

    class Renamed(Raster):
        def get_key(self) -> str:
            return "renamed"

    assert Raster.decodes_any_index and Renamed.decodes_any_index and OptedIn.decodes_any_index
    assert not TableRaster.decodes_any_index and not SkipOdd.decodes_any_index


def test_runtime_keep_needs_skips(requires_blackwell):
    """Only a mapping that declares ``skips`` may drop items: the layer would otherwise skip
    no index and run a dropped item."""
    op = CountToy(StaticPersistent(), UndeclaredDrop(), SingleCta())
    with pytest.raises(Exception, match="sets skips = True"):
        compile_count_toy(op)


def _count_items(fn, ws, space, max_ctas):
    count = torch.zeros(2, dtype=torch.int32, device="cuda")
    fn(count, ws, *space, max_ctas)
    torch.cuda.synchronize()
    return count.tolist()


def _check_valid_launch(op, fn, ws):
    """30 items over a capped grid (persistent units run several each): every role in every
    CTA of a unit counts each kept item, and the atomic counter is zeroed again."""
    space = (5, 3, 2)
    mapping = op.work.mapping
    kept = sum(
        not mapping.skips or mapping.kept(item)
        for item in itertools.product(*(range(e) for e in space))
    )
    want = kept * op.cluster.size
    assert _count_items(fn, ws, space, 4) == [want, want]
    assert ws[:2].tolist() == [0, 0]


@pytest.mark.parametrize("cluster", CLUSTERS, ids=CLUSTER_IDS)
@pytest.mark.parametrize("case", list(COUNT_CASES))
def test_empty_space_runs_no_items(requires_blackwell, case, cluster):
    """An empty (or negative) space launches one work unit, every role sees no item, and the
    launch leaves the atomic counter zeroed, so a later nonempty launch covers its space."""
    op = CountToy(*COUNT_CASES[case], cluster)
    fn = compile_count_toy(op)
    ws, _ = op.work.launch_args("cuda")
    for space in EMPTY_SPACES:
        for max_ctas in (148, 1):
            assert _count_items(fn, ws, space, max_ctas) == [0, 0], (space, max_ctas)
            assert ws[:2].tolist() == [0, 0], (space, max_ctas)
    _check_valid_launch(op, fn, ws)


OVERFLOWING_SPACES = [
    (65537, 65537, 1),  # 2**32 + 131073 items wrap to 131073 in Int32
    (2**17, 65535, 1),  # n and l fit gridDim.y/z, so OneShot selects the tile grid
    (2**31 - 1, 2**31 - 1, 4),  # the exact count also wraps Int64
    (2**21, 2**21, 2**22 - 1),  # wraps to 0 in Int32: would run as an empty space
]


@pytest.mark.parametrize("cluster", CLUSTERS, ids=CLUSTER_IDS)
@pytest.mark.parametrize("case", list(COUNT_CASES))
def test_space_overflowing_int32_is_rejected(requires_blackwell, case, cluster):
    """An item count, grid, or persistent terminal index past Int32 makes the launch fail (no
    partial or wrapped coverage), and the next valid launch is unaffected."""
    schedule, mapping = COUNT_CASES[case]
    op = CountToy(schedule, mapping, cluster)
    fn = compile_count_toy(op)
    ws, _ = op.work.launch_args("cuda")
    spaces = list(OVERFLOWING_SPACES)
    if isinstance(schedule, (StaticPersistent, AtomicPersistent)):
        # Fits Int32, but the last unit's terminal index (total - 1 + units) does not (every
        # cluster shape launches more than 10 units here).
        spaces.append((2**31 - 10, 1, 1))
    for space in spaces:
        with pytest.raises(Exception, match=INVALID_LAUNCH):
            _count_items(fn, ws, space, 148)
    _check_valid_launch(op, fn, ws)


@pytest.mark.parametrize("cluster", CLUSTERS, ids=CLUSTER_IDS)
def test_concurrent_launches_use_one_workspace_per_stream(requires_blackwell, cluster):
    """``launch_args()`` caches one workspace per (device, stream): launches on one stream share
    it (they run in order), launches on two streams get distinct workspaces, so concurrent
    atomic launches each cover their own space and each leave their counter zeroed."""
    op = CountToy(*COUNT_CASES["atomic"], cluster)
    fn = compile_count_toy(op)
    streams = [torch.cuda.Stream() for _ in range(2)]
    workspaces = []
    for strm in streams:
        with torch.cuda.stream(strm):
            ws, _ = op.work.launch_args("cuda")
            again, _ = op.work.launch_args("cuda")
            assert again.data_ptr() == ws.data_ptr()
            workspaces.append(ws)
    assert workspaces[0].data_ptr() != workspaces[1].data_ptr()
    space = (9, 7, 3)
    want = math.prod(space) * op.cluster.size
    counts = [torch.zeros(2, dtype=torch.int32, device="cuda") for _ in streams]
    torch.cuda.synchronize()
    for _ in range(4):  # several rounds in flight on both streams at once
        for strm, ws, count in zip(streams, workspaces, counts, strict=True):
            with torch.cuda.stream(strm):
                fn(count, ws, *space, 6)
    torch.cuda.synchronize()
    for ws, count in zip(workspaces, counts, strict=True):
        assert count.tolist() == [4 * want, 4 * want]
        assert ws[:2].tolist() == [0, 0]


@pytest.mark.parametrize("case", ["oneshot_tile_grid", "static", "clc"])
def test_schedules_without_a_workspace_share_an_untouched_placeholder(requires_blackwell, case):
    """Schedules that never read a workspace get one per device, whichever stream asks; the
    kernel leaves it alone, and a CUDA graph captured on a new stream records no fill."""
    op = CountToy(*COUNT_CASES[case], SingleCta())
    fn = compile_count_toy(op)
    count = torch.zeros(2, dtype=torch.int32, device="cuda")
    with torch.cuda.stream(torch.cuda.Stream()):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            ws, sms = op.work.launch_args("cuda")  # first call: inside the capture
            fn(count, ws, 9, 7, 3, sms)
    again, _ = op.work.launch_args("cuda")
    assert again.data_ptr() == ws.data_ptr()
    ws.fill_(0x5EED)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        graph.replay()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    assert not any("memset" in n.lower() for n in names), names
    assert count.tolist() == [9 * 7 * 3] * 2
    assert ws.tolist() == [0x5EED] * ws.numel()


def test_placeholder_is_per_resolved_device(monkeypatch):
    """``"cuda"`` resolves to the current device before the cache lookup, so switching devices
    gets that device's placeholder, ``"cuda"`` and ``"cuda:N"`` share one, and the capture check
    asks the requested device's current stream (devices and allocations mocked)."""
    current, capturing = [0], {0: False, 1: False, 2: True}

    class Device:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            self.saved, current[0] = current[0], self.index

        def __exit__(self, *exc):
            current[0] = self.saved

    def alloc(kind):
        return lambda words, dtype, device: SimpleNamespace(kind=kind, device=device)

    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "device", Device)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[current[0]])
    monkeypatch.setattr(torch, "zeros", alloc("zeros"))
    monkeypatch.setattr(torch, "empty", alloc("empty"))
    monkeypatch.setattr(scheduling, "resident_ctas", lambda index, cs: 148)
    stream = WorkStream(schedule=OneShot())
    on0, _ = stream.launch_args("cuda")
    current[0] = 1
    on1, _ = stream.launch_args("cuda")
    assert (on0.device, on1.device) == (torch.device("cuda", 0), torch.device("cuda", 1))
    assert stream.launch_args("cuda:1")[0] is on1
    current[0] = 0
    assert stream.launch_args("cuda:1")[0] is on1 and stream.launch_args("cuda")[0] is on0
    on2, _ = stream.launch_args("cuda:2")  # device 2 is capturing; the current device is not
    assert (on2.kind, on2.device) == ("empty", torch.device("cuda", 2))
    assert on0.kind == on1.kind == "zeros" and current[0] == 0


def test_record_needs_a_bound_stream(requires_blackwell):
    stream = WorkStream(schedule=StaticPersistent(), record=4)
    with pytest.raises(ValueError, match="before launch_args"):
        stream.launch_args("cuda")


def test_describe_places_scheduler_and_counts(requires_blackwell):
    op = ScheduleToy(ClusterLaunchControl(stages=3))
    op._collect()
    chans = op._ws_chans
    work = chans["work_ring"]
    # consumers: a (32) + b (64) + scheduler (32) threads; producer: scheduler warp.
    assert work.slot_counts("consumer") == [128] * 3
    assert work.slot_counts("producer") == [32] * 3
    assert chans["work_clc"].slot_counts("producer") == [1]
    sched = next(r for r in op._ws_roles if r.name == "work.scheduler")
    assert sched.first_warp == 3
    assert "scheduler warp 3" in op.describe()
    local = ScheduleToy(StaticPersistent())
    local._collect()
    assert "work_ring" not in local._ws_chans
    assert "decoded locally" in local.describe()


@pytest.mark.parametrize(
    "mode,message",
    [
        ("no_advance", "never calls advance"),
        ("double_advance", "advance\\(\\) traced 2 times"),
        ("conditional_advance", "not under a nested branch"),
    ],
)
def test_cursor_protocol_errors(requires_blackwell, mode, message):
    with pytest.raises(ProtocolError, match=message):
        compile_toy(ScheduleToy(StaticPersistent(), mode=mode))


class _IdleRole(WarpSpecialized):
    def __init__(self):
        self.work = WorkStream(schedule=StaticPersistent())

    @role(warps=0)
    @cute.jit
    def a(self, work: WorkCursor, ctx):
        return work

    @role(warps=1)
    @cute.jit
    def b(self, ctx):
        pass


def test_persistent_requires_every_role_to_loop():
    with pytest.raises(ProtocolError, match="do not take `work: WorkCursor`"):
        _IdleRole()._collect()


def test_random_extents_bijection_fuzz():
    rng = random.Random(0)
    for _ in range(200):
        ext = (rng.randint(1, 40), rng.randint(1, 40), rng.randint(1, 3))
        mapping = Raster(
            along=rng.choice("mn"),
            swizzle=rng.randint(1, 12),
            reverse=rng.choice([(), ("m",), ("n",)]),
        )
        coords = _all_coords(mapping, ext)
        assert len(set(coords)) == len(coords) == ext[0] * ext[1] * ext[2]
        assert all(0 <= c < e for xyz in coords for c, e in zip(xyz, ext, strict=True))


# --------------------------------------------------------------------------- #
# Persistent GEMM example: every schedule, ragged shapes, tiles < SMs and >> SMs.
# --------------------------------------------------------------------------- #

GEMM_SHAPES = [
    (1, 8, 8),
    (300, 520, 136),  # ragged M, N, K
    (128, 256, 64),  # exactly one tile
    (1000, 1000, 1000),  # 8 x 4 tiles < SMs
    (4097, 2056, 520),  # 33 x 9 = 297 tiles, two waves, ragged
    (8192, 2048, 128),  # 512 tiles, short K: epilogue-bound, many tiles per CTA
]


def _check_gemm(schedule, m, n, k, max_ctas=None, mapping=None, cluster=None):
    from attn_gym.warpdrive.examples.persistent_gemm import persistent_gemm

    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(a.float(), b.float())
    torch_err = (torch.nn.functional.linear(a, b).float() - ref).abs().max().item()
    out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):  # the second launch reuses the (self-reset) atomic counter
        out.fill_(float("nan"))
        persistent_gemm(
            a, b, out, schedule=schedule, mapping=mapping, max_ctas=max_ctas, cluster=cluster
        )
        err = (out.float() - ref).abs().max().item()
        assert err <= 2 * torch_err + 1e-3, (schedule, (m, n, k), err, torch_err)


@pytest.mark.parametrize("m,n,k", GEMM_SHAPES)
@pytest.mark.parametrize("schedule", ["oneshot", "static", "atomic", "clc"])
def test_persistent_gemm(requires_blackwell, schedule, m, n, k):
    _check_gemm(SCHEDULES[schedule], m, n, k)


@pytest.mark.parametrize("schedule", ["static", "atomic", "atomic1", "clc1"])
@pytest.mark.parametrize("max_ctas", [1, 5])
def test_persistent_gemm_capped_grid(requires_blackwell, schedule, max_ctas):
    """A tiny grid makes every CTA wrap the work, A/B, and accumulator rings many times."""
    _check_gemm(SCHEDULES[schedule], 1100, 1032, 200, max_ctas=max_ctas)


@pytest.mark.parametrize("mapping", [Raster(), Raster(along="n", swizzle=3, reverse=("m", "n"))])
def test_persistent_gemm_mappings(requires_blackwell, mapping):
    _check_gemm(SCHEDULES["atomic"], 1500, 1800, 64, mapping=mapping)


def test_persistent_gemm_compiles_once_across_shapes(requires_blackwell):
    """Tile counts, K, and the grid cap are runtime values: one compile per (schedule, mapping)."""
    from attn_gym.warpdrive.examples.persistent_gemm import (
        PersistentGemm,
        compile_persistent_gemm,
    )

    compile_persistent_gemm.cache_clear()
    op = PersistentGemm(schedule=AtomicPersistent(stages=3), ab_stages=3)
    fns = set()
    # m: 1, multiple of 16, odd, tile edge, multi-wave; n, k: multiples of 8 on both sides of 16.
    for m, n, k, cap in [
        (1, 8, 8, 148),
        (16, 256, 64, 3),
        (129, 264, 136, 148),
        (4096, 520, 24, 64),
    ]:
        torch.manual_seed(m)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        fn = compile_persistent_gemm(op)
        fns.add(id(fn))
        ws, _ = op.work.launch_args("cuda")
        fn(a, b, out, ws, cap)
        ref = torch.nn.functional.linear(a.float(), b.float())
        assert (out.float() - ref).abs().max().item() < 0.05 * ref.abs().max().item() + 1e-2
    info = compile_persistent_gemm.cache_info()
    assert info.currsize == 1 and info.hits + info.misses == 4 and len(fns) == 1


# --------------------------------------------------------------------------- #
# FA4-style forward on the same primitive: persistent schedules.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "max_ctas,turnstile", [(None, False), (3, True)], ids=["sm_grid", "3ctas_turnstile"]
)
@pytest.mark.parametrize("nonuniform", [False, True])
@pytest.mark.parametrize("b,h,sq,sk", [(2, 3, 640, 512), (8, 16, 1024, 384)])
@pytest.mark.parametrize("schedule", ["static", "atomic", "clc"])
def test_fa4_fwd_persistent(
    requires_blackwell, schedule, b, h, sq, sk, nonuniform, max_ctas, turnstile
):
    """(8, 16, 1024) has 256 work items, so most CTAs run two (S/O/Q slots reused); with 3 CTAs
    each runs ~85, wrapping every slot and ring. sk=384 is an odd key-block count."""
    pytest.importorskip("flash_attn.cute")
    from test_warpdrive import fa4_inputs

    from attn_gym.warpdrive.examples.fa4_fwd import fa4_fwd

    q, k, v = fa4_inputs(b, h, sq, sk, nonuniform)
    qf, kf, vf = (x.float().transpose(1, 2) for x in (q, k, v))
    ref = torch.nn.functional.scaled_dot_product_attention(qf, kf, vf).transpose(1, 2)
    ref_lse = torch.logsumexp((qf @ kf.transpose(-1, -2)) / math.sqrt(128), dim=-1)
    tol = 2**-7 * ref.abs().max().item() + 4e-3
    for _ in range(2):
        o, lse = fa4_fwd(
            q, k, v, schedule=SCHEDULES[schedule], max_ctas=max_ctas, turnstile=turnstile
        )
        assert (o.float() - ref).abs().max().item() <= tol
        torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)


def test_persistent_gemm_empty_k(requires_blackwell):
    from attn_gym.warpdrive.examples.persistent_gemm import persistent_gemm

    a = torch.empty(64, 0, device="cuda", dtype=torch.bfloat16)
    b = torch.empty(24, 0, device="cuda", dtype=torch.bfloat16)
    assert (persistent_gemm(a, b) == 0).all()
