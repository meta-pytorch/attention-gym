"""External WorkMap examples match their libraries' orders and run on supported spaces."""

import importlib
import sys
import warnings

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import Int32, cute
from cutlass_static_persistent_tile_scheduler import (
    PersistentTileSchedulerParams,
    StaticPersistentTileScheduler,
)
from test_warpdrive_scheduler_interface import check_raw_role_single_item
from test_warpdrive_scheduling import (
    EMPTY_SPACES,
    OVERFLOWING_SPACES,
    CountToy,
    ScheduleToy,
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
    OneShot,
    Raster,
    SingleCta,
    StaticPersistent,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive.examples.external_maps import CutlassRaster, Fa4Mapping

# The vendored CUTLASS oracle still creates V1 fast-divmod divisors (the example does not).
pytestmark = pytest.mark.filterwarnings(
    "ignore:FastDivmodDivisor is deprecated:DeprecationWarning"
)


def fa4_tile_scheduler():
    return pytest.importorskip("flash_attn.cute.tile_scheduler")


def test_external_maps_import_without_flash_attn(monkeypatch):
    """The core, examples package and mapping module import without flash_attn."""
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", None)
    monkeypatch.setitem(sys.modules, "flash_attn.cute.tile_scheduler", None)
    for name in [m for m in sys.modules if m.startswith("attn_gym.warpdrive")]:
        monkeypatch.delitem(sys.modules, name)
    warpdrive = importlib.import_module("attn_gym.warpdrive")
    importlib.import_module("attn_gym.warpdrive.examples")
    external_maps = importlib.import_module("attn_gym.warpdrive.examples.external_maps")
    warpdrive.WorkStream(mapping=external_maps.CutlassRaster(swizzle=4))
    with pytest.raises(ImportError, match="flash-attn-4"):
        external_maps.Fa4Mapping(object)


def test_adapter_validation():
    with pytest.raises(ValueError, match="along"):
        CutlassRaster(along="k")
    with pytest.raises(ValueError, match="swizzle"):
        CutlassRaster(swizzle=0)
    assert CutlassRaster(swizzle=8).skips and not CutlassRaster().skips
    ts = fa4_tile_scheduler()
    with pytest.raises(ValueError, match="one linear index"):
        Fa4Mapping(ts.SingleTileScheduler)


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


@pytest.mark.parametrize("cluster", [SingleCta(), CtaPair()], ids=["single", "pair"])
def test_fa4_lpt_empty_kv_nonempty_space(requires_blackwell, cluster):
    """Empty KV normalization preserves an accepted nonempty LPT launch."""
    ts = fa4_tile_scheduler()
    op = EmptyKvCountToy(OneShot(), Fa4Mapping(ts.SingleTileLPTScheduler), cluster)
    fn = compile_count_toy(op)
    ws, _ = op.work.launch_args("cuda")
    assert _count_items(fn, ws, (32767, 1, 1), 148) == [32767 * cluster.size] * 2


def fa4_or_none():
    try:
        return importlib.import_module("flash_attn.cute.tile_scheduler")
    except ImportError:
        return None


def count_cases():
    """name -> (toy class, schedule, mapping)."""
    cases = {
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
def test_external_maps_cover_empty_and_normal_spaces(requires_blackwell, case, cluster):
    """Supported external spaces run once per CTA; empty spaces run no work.

    Oversized spaces are tested with core maps, not FA4's narrower host-math domain.
    """
    toy, schedule, mapping = count_cases()[case]
    op = toy(schedule, mapping, cluster)
    fn = compile_count_toy(op)
    ws, _ = op.work.launch_args("cuda")
    for space in EMPTY_SPACES:
        assert _count_items(fn, ws, space, 148) == [0, 0], space
    want = 5 * 3 * 2 * cluster.size
    assert _count_items(fn, ws, (5, 3, 2), 4) == [want, want]


@pytest.mark.parametrize("cluster", [SingleCta(), CtaPair()], ids=["single", "pair"])
@pytest.mark.parametrize(
    "schedule,mapping",
    [
        (StaticPersistent(), CutlassRaster(swizzle=4)),
        (ClusterLaunchControl(), CutlassRaster(swizzle=4, along="n")),
    ],
    ids=["static", "clc"],
)
def test_cutlass_rejects_overflowing_spaces(requires_blackwell, schedule, mapping, cluster):
    op = CountToy(schedule, mapping, cluster)
    fn = compile_count_toy(op)
    ws, _ = op.work.launch_args("cuda")
    for space in OVERFLOWING_SPACES:
        with pytest.raises(Exception, match=INVALID_LAUNCH):
            _count_items(fn, ws, space, 148)


@pytest.mark.parametrize("along", ["m", "n"])
def test_cutlass_padded_count_does_not_wrap(along):
    mapping = CutlassRaster(swizzle=4, along=along)
    extents = (1, 2**31 - 1, 1) if along == "m" else (2**31 - 1, 1, 1)
    bound = mapping.bind(tuple(Int32(e) for e in extents))
    assert int(mapping.count64(bound)) == 2**31


def test_cutlass_raw_role_single_item(requires_blackwell):
    check_raw_role_single_item(CutlassRaster(swizzle=4), "grid", (7, 5, 2), SingleCta())


def gemm_schedules():
    """External example mappings under different index schedules."""
    ts = fa4_or_none()
    schedules = {
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
def test_persistent_gemm_swaps_external_maps(requires_blackwell, case, cluster):
    schedule, mapping = gemm_schedules()[case]
    for m, n, k, max_ctas in [(300, 520, 136, None), (1100, 1032, 200, 5)]:
        check_gemm(schedule, m, n, k, max_ctas=max_ctas, mapping=mapping, cluster=cluster)


def fa4_example_cases():
    """name -> (schedule, mapping factory, cluster) for the FA4 example's ``mapping=``."""
    return {
        "swizzled_static": (StaticPersistent(), lambda ts: Raster(swizzle=2), SingleCta()),
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
