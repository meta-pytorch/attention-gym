"""Test custom schedules, external adapters, and raw-role work-item access."""

import importlib
import itertools
import os
import sys
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import Int32, Int64, cute
from cutlass_static_persistent_tile_scheduler import (
    PersistentTileSchedulerParams,
    StaticPersistentTileScheduler,
)
from test_warpdrive_scheduling import (
    EMPTY_SPACES,
    OVERFLOWING_SPACES,
    SCHEDULES,
    CheckedRaster,
    CountToy,
    ScheduleToy,
    SkipOdd,
    StreamToy,
    _count_items,
    check_gemm,
    compile_count_toy,
    compile_toy,
    run_toy,
)

from attn_gym.testing.warpdrive import INVALID_LAUNCH
from attn_gym.warpdrive import (
    AtomicPersistent,
    ClusterLaunchControl,
    CtaPair,
    CutlassRaster,
    Fa4Mapping,
    IndexSchedule,
    OneShot,
    PdlWait,
    ProtocolError,
    Raster,
    Readiness,
    SingleCta,
    StaticPersistent,
    WorkMap,
    WorkStream,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.examples.packed_work import PackedWork

# The vendored CUTLASS oracle still creates V1 fast-divmod divisors (the adapter does not).
pytestmark = pytest.mark.filterwarnings(
    "ignore:FastDivmodDivisor is deprecated:DeprecationWarning"
)


def fa4_tile_scheduler():
    return pytest.importorskip("flash_attn.cute.tile_scheduler")


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


def test_adapters_import_without_flash_attn(monkeypatch):
    """warpdrive imports without flash_attn; only constructing Fa4Mapping needs it."""
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", None)
    monkeypatch.setitem(sys.modules, "flash_attn.cute.tile_scheduler", None)
    for name in [m for m in sys.modules if m.startswith("attn_gym.warpdrive")]:
        monkeypatch.delitem(sys.modules, name)
    warpdrive = importlib.import_module("attn_gym.warpdrive")
    warpdrive.WorkStream(mapping=warpdrive.CutlassRaster(swizzle=4))
    with pytest.raises(ImportError, match="flash-attn-4"):
        warpdrive.Fa4Mapping(object)


def test_adapter_validation():
    with pytest.raises(ValueError, match="along"):
        CutlassRaster(along="k")
    with pytest.raises(ValueError, match="swizzle"):
        CutlassRaster(swizzle=0)
    assert CutlassRaster(swizzle=8).skips and not CutlassRaster().skips
    ts = fa4_tile_scheduler()
    with pytest.raises(ValueError, match="one linear index"):
        Fa4Mapping(ts.SingleTileScheduler)


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


# External schedulers: item order equals the library's own sequence.

MAX_REF = ScheduleToy.MAX_ITEMS


class ExternalReference:
    """Runs an external scheduler the way its library does (``create``,
    ``initial_work_tile_info``, ``advance_to_next_work``) and logs each unit's raw
    ``(m, n, l)`` coordinates in ``log[unit, k]``. CUTLASS: the vendored BSD-3 CuTeDSL helper
    (``cutlass_static_persistent_tile_scheduler.py``) over the space's tiles. FA4: its params
    built here from ``TileSchedulerArguments`` the way FA4's bf16 head-dim-128 forward builds
    them, independently of the adapter's ``bind``; ``lpt`` holds the LPT flags."""

    def __init__(self, adapter, kind: str, seqlen_k: int = 0, lpt=(True, False)):
        self.adapter, self.kind, self.seqlen_k, self.lpt = adapter, kind, seqlen_k, lpt
        self.unit_axis = 2 if kind == "cutlass" else 0  # CUTLASS launches (1, 1, units)

    def fa4_params(self, em, en, el):
        ts = fa4_tile_scheduler()
        args = ts.TileSchedulerArguments(
            num_block=em,
            num_head=en,
            num_batch=el,
            num_splits=Int32(1),
            seqlen_k=Int32(self.seqlen_k),
            headdim=128,
            headdim_v=128,
            total_q=Int32(0),
            tile_shape_mn=(128, 128),
            element_size=2,
            lpt=self.lpt[0],
            lpt_global=self.lpt[1],
        )
        return self.adapter.scheduler.to_underlying_arguments(args)

    def _create(self, params):
        if self.kind == "cutlass":
            return StaticPersistentTileScheduler.create(
                params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
        return self.adapter.scheduler.create(params)

    @cute.kernel
    def kernel(self, log: cute.Tensor, params):
        unit = cute.arch.block_idx()[self.unit_axis]
        scheduler = self._create(params)
        work = scheduler.initial_work_tile_info()
        k = Int32(0)
        while work.is_valid_tile:
            if (cute.arch.thread_idx()[0] == 0) & (k < MAX_REF):
                for a in cutlass.range_constexpr(3):
                    log[unit, k, a] = Int32(work.tile_idx[a])
            k += 1
            scheduler.advance_to_next_work()
            work = scheduler.get_current_work()

    @cute.jit
    def __call__(self, log: cute.Tensor, em: Int32, en: Int32, el: Int32, units: Int32,
                 stream: cuda.CUstream):  # fmt: skip
        if cutlass.const_expr(self.kind == "cutlass"):
            a = self.adapter
            params = PersistentTileSchedulerParams(
                (em, en, el), (1, 1, 1), a.swizzle, a.along == "m"
            )
            grid = (1, 1, units)
        else:
            params = self.fa4_params(em, en, el)
            grid = (units, 1, 1)
        self.kernel(log, params).launch(grid=grid, block=(32, 1, 1), stream=stream)


def reference_sequences(ref: ExternalReference, ext, units: int) -> list[list[int]]:
    """Per unit, the tile ids the external scheduler visits inside the space (CUTLASS hands
    out padded swizzle tiles too; the adapter skips them)."""
    em, en, el = ext
    log = torch.full((units, MAX_REF, 3), -1, dtype=torch.int32, device="cuda")
    i32 = cutlass.Int32
    fn = compile_tvm_ffi(
        ref,
        make_fake_compact_tensor(i32, (cute.sym_int(), MAX_REF, 3)),
        Int32(1),
        Int32(1),
        Int32(1),
        Int32(1),
        name=f"warpdrive_external_reference_{ref.adapter.get_key()}",
    )
    fn(log, em, en, el, units)
    torch.cuda.synchronize()
    out = []
    for row in log.cpu().tolist():
        out.append([(li * en + n) * em + m for m, n, li in row if m >= 0 and m < em and n < en])
    return out


def padded_count(adapter: CutlassRaster, ext) -> int:
    em, en, el = ext
    s = adapter.swizzle
    if adapter.along == "m":
        en = -(-en // s) * s
    else:
        em = -(-em // s) * s
    return em * en * el


def check_external_order(op, ref, ext, max_ctas, items):
    """Every tile reaches both roles once, and each unit's sequence (up to ``MAX_ITEMS``)
    equals the external scheduler's."""
    log, hits, _ = run_toy(op, ext, max_ctas, items=items)
    assert (hits == 1).all(), f"tiles missed or repeated: {hits}"
    assert torch.equal(log[0], log[1])
    units = max(min(items, max_ctas), 1) if op.work.schedule.uses_max_ctas else max(items, 1)
    for b, want in enumerate(reference_sequences(ref, ext, units)):
        want = want[:MAX_REF]
        got = [t for t in log[0, b].tolist() if t >= 0]
        assert got == want, (b, got, want)


EXTERNAL_SHAPES = [((5, 3, 1), 148), ((13, 29, 3), 148), ((7, 17, 2), 5)]


@pytest.mark.parametrize("ext,max_ctas", EXTERNAL_SHAPES)
@pytest.mark.parametrize(
    "adapter",
    [
        CutlassRaster(),
        CutlassRaster(along="n"),
        CutlassRaster(swizzle=3),
        CutlassRaster(swizzle=8, along="n"),
    ],
    ids=lambda a: a.get_key(),
)
def test_cutlass_static_persistent_order(requires_blackwell, adapter, ext, max_ctas):
    op = ScheduleToy(StaticPersistent(), mapping=adapter, mode="ext")
    ref = ExternalReference(adapter, "cutlass")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile_toy(op)
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)], caught
    check_external_order(op, ref, ext, max_ctas, padded_count(adapter, ext))


@pytest.mark.parametrize("ext,max_ctas", EXTERNAL_SHAPES)
def test_fa4_static_persistent_order(requires_blackwell, ext, max_ctas):
    ts = fa4_tile_scheduler()
    adapter = Fa4Mapping(ts.StaticPersistentTileScheduler)
    op = ScheduleToy(StaticPersistent(), mapping=adapter, mode="ext")
    ref = ExternalReference(adapter, "fa4")
    check_external_order(op, ref, ext, max_ctas, ext[0] * ext[1] * ext[2])


# 4096 keys of head dim 128 in bf16 give 16-head L2 sections, so 29 * 3 heads exercise FA4's
# residual section.
SEQLEN_K = 4096


class LptSpace:
    """Binds ``(num_block, num_head, num_batch, seqlen_k)``."""

    seqlen_k = SEQLEN_K

    def space(self, em, en, el):
        return (em, en, el, Int32(self.seqlen_k))


class LptToy(LptSpace, ScheduleToy):
    pass


class LptCountToy(LptSpace, CountToy):
    pass


class EmptyKvCountToy(LptCountToy):
    seqlen_k = 0  # FA4 would divide by the zero K/V bytes per head


@pytest.mark.parametrize("lpt", [(True, False), (False, False), (True, True)], ids=str)
@pytest.mark.parametrize("ext", [(13, 29, 3), (2, 40, 1)])
def test_fa4_lpt_order(requires_blackwell, ext, lpt):
    ts = fa4_tile_scheduler()
    adapter = Fa4Mapping(ts.SingleTileLPTScheduler, lpt=lpt[0], lpt_global=lpt[1])
    op = LptToy(OneShot(), mapping=adapter, mode="ext")
    ref = ExternalReference(adapter, "fa4", SEQLEN_K, lpt)
    check_external_order(op, ref, ext, 148, ext[0] * ext[1] * ext[2])


def test_fa4_adapter_spaces_and_keys():
    ts = fa4_tile_scheduler()
    with pytest.raises(ValueError, match="seqlen_k"):
        Fa4Mapping(ts.SingleTileLPTScheduler).bind((1, 2, 3))
    with pytest.raises(ValueError, match="num_batch\\)"):
        Fa4Mapping(ts.StaticPersistentTileScheduler).bind((1, 2, 3, 4))
    with pytest.raises(ValueError, match="head_dim"):
        Fa4Mapping(ts.SingleTileLPTScheduler, head_dim=0)
    # Every compile-time argument of the LPT scheduler is in the key.
    keys = {
        Fa4Mapping(ts.SingleTileLPTScheduler, **kw).get_key()
        for kw in [{}, {"lpt": False}, {"lpt_global": True}, {"head_dim": 64}, {"element_size": 1}]
    }
    assert len(keys) == 5


# Pairs, empty spaces and overflow for custom schedules and adapters


def fa4_or_none():
    try:
        return importlib.import_module("flash_attn.cute.tile_scheduler")
    except ImportError:
        return None


def count_cases():
    """name -> (toy class, schedule, mapping)."""
    cases = {
        "reverse_static": (CountToy, ReverseStatic(), Raster(along="n", swizzle=3)),
        "reverse_broadcast": (CountToy, ReverseBroadcast(), Raster(along="n", swizzle=3)),
        "cutlass_static": (CountToy, StaticPersistent(), CutlassRaster(swizzle=4)),
        "cutlass_clc": (CountToy, ClusterLaunchControl(), CutlassRaster(swizzle=4, along="n")),
    }
    ts = fa4_or_none()
    if ts is not None:
        static = Fa4Mapping(ts.StaticPersistentTileScheduler)
        lpt_global = Fa4Mapping(ts.SingleTileLPTScheduler, lpt_global=True)
        lpt = Fa4Mapping(ts.SingleTileLPTScheduler)
        cases["fa4_static"] = (CountToy, StaticPersistent(), static)
        cases["fa4_lpt_global"] = (LptCountToy, OneShot(), lpt_global)
        cases["fa4_lpt_empty_kv"] = (EmptyKvCountToy, OneShot(), lpt)
    return cases


@pytest.mark.parametrize("cluster", [SingleCta(), CtaPair()], ids=["single", "pair"])
@pytest.mark.parametrize("case", list(count_cases()))
def test_swapped_schedules_cover_empty_and_overflowing_spaces(requires_blackwell, case, cluster):
    """Every CTA in a unit sees its items; empty spaces run no work.

    Empty KV must not divide by zero in the entry point. Reject spaces past Int32 at launch,
    using the custom schedule's default index_limit and the adapters' Int64 count64.
    """
    toy, schedule, mapping = count_cases()[case]
    op = toy(schedule, mapping, cluster)
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

    def __init__(self, mapping, mode: str, cluster=None):
        super().__init__(OneShot(), mapping, cluster)
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
    "mapping, mode, ext, cluster",
    [
        (Raster(), "grid", (5, 3, 2), SingleCta()),
        (Raster(), "grid", (2, 65536, 1), SingleCta()),  # n_tiles past gridDim.y: 1-D fallback
        (Raster(), "1d", (5, 3, 2), SingleCta()),
        (Raster(), "grid", (3, 70000, 1), CtaPair()),
        (Raster(swizzle=3, reverse="m"), "grid", (7, 5, 2), CtaPair()),
        (SkipOdd(swizzle=2), "grid", (9, 7, 2), SingleCta()),
        (CutlassRaster(swizzle=4), "grid", (7, 5, 2), SingleCta()),
        (Raster(), "grid", (0, 3, 1), SingleCta()),  # empty: one unit, no valid item
        # A default WorkMap traps if decoded past the end: one CTA per item, plus drops.
        (CheckedRaster(drop_odd=True), "grid", (9, 7, 2), CtaPair()),
        # A stateful mapping: single_item starts the state like a managed role.
        (ChainedTiles(swizzle=2, drop_odd=True), "grid", (9, 7, 2), CtaPair()),
        (ChainedTiles(swizzle=2), "grid", (0, 3, 1), SingleCta()),
    ],
    ids=["tile_grid", "fallback", "hand_1d", "pair_fallback", "pair_swizzle", "skip", "cutlass",
         "empty", "checked_skip", "stateful_skip", "stateful_empty"],
)  # fmt: skip
def test_raw_role_single_item(requires_blackwell, mapping, mode, ext, cluster):
    op = RawItemToy(mapping, mode, cluster)
    em, en, el = ext
    total = max(em * en * el, 0)
    hits = torch.zeros(2, max(total, 1), dtype=torch.int32, device="cuda")
    fn = compile_raw_item_toy(op)
    fn(hits, em, en, el)
    torch.cuda.synchronize()
    hits = hits.cpu()[:, :total]
    kept = torch.tensor(
        [
            0 if hasattr(mapping, "kept") and not mapping.kept((m, n, li)) else cluster.size
            for li in range(el)
            for n in range(en)
            for m in range(em)
        ],
        dtype=torch.int32,
    )
    assert torch.equal(hits[0], kept) and torch.equal(hits[1], kept)


# The persistent GEMM: one argument swaps the schedule.


def gemm_schedules():
    """Custom schedules and external mappings (the built-ins run in test_warpdrive_scheduling
    and test_warpdrive_cluster)."""
    ts = fa4_or_none()
    schedules = {
        "reverse_static": (ReverseStatic(), None),
        "reverse_broadcast": (ReverseBroadcast(), None),
        "cutlass_static": (StaticPersistent(), CutlassRaster(swizzle=8)),
        "cutlass_clc": (ClusterLaunchControl(), CutlassRaster(swizzle=2, along="n")),
    }
    if ts is not None:
        adapter = Fa4Mapping(ts.StaticPersistentTileScheduler)
        schedules["fa4_static"] = (StaticPersistent(), adapter)
        schedules["fa4_atomic"] = (AtomicPersistent(), adapter)
    return schedules


GEMM_SCHEDULES = list(gemm_schedules())


@pytest.mark.parametrize("cluster", [None, CtaPair()], ids=["single", "pair"])
@pytest.mark.parametrize("case", GEMM_SCHEDULES)
def test_persistent_gemm_swaps_schedules(requires_blackwell, case, cluster):
    schedule, mapping = gemm_schedules()[case]
    for m, n, k, max_ctas in [(300, 520, 136, None), (1100, 1032, 200, 5)]:
        check_gemm(schedule, m, n, k, max_ctas=max_ctas, mapping=mapping, cluster=cluster)


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


def fa4_example_cases():
    """name -> (schedule, mapping factory, cluster) for the FA4 example's ``mapping=``."""
    return {
        "lpt_static": (StaticPersistent(), lambda ts: Raster(reverse="m"), SingleCta()),
        "fa4_static": (
            StaticPersistent(),
            lambda ts: Fa4Mapping(ts.StaticPersistentTileScheduler),
            SingleCta(),
        ),
        "fa4_clc_pair": (
            ClusterLaunchControl(),
            lambda ts: Fa4Mapping(ts.StaticPersistentTileScheduler),
            CtaPair(),
        ),
    }


@pytest.mark.parametrize("case", list(fa4_example_cases()))
def test_fa4_example_swaps_mapping(requires_blackwell, case):
    """The FA4 example's (query block, head, batch) items take any mapping, including FA4's own
    scheduler order, with unchanged numerics."""
    ts = fa4_tile_scheduler()
    from test_warpdrive import check_fa4, fa4_inputs

    schedule, make_mapping, cluster = fa4_example_cases()[case]
    check_fa4(
        *fa4_inputs(2, 5, 1536, 384),
        schedule=schedule,
        cluster=cluster,
        mapping=make_mapping(ts),
        max_ctas=7,
    )
