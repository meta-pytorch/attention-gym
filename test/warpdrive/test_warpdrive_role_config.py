"""Constructor-configured roles (``roles=`` overrides): binding, validation, specialization
reuse, and real handoffs."""

from dataclasses import FrozenInstanceError, dataclass, field, replace

import pytest
import torch

pytest.importorskip("cutlass.cute")

import cutlass
from cuda.bindings import driver as cuda
from cutlass import cute

from attn_gym._backends.cute.cache import CacheInfo, jit_cache
from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive import (
    Consumer,
    CtaPair,
    Instance,
    Local,
    Producer,
    ProtocolError,
    Ring,
    RoleConfig,
    Slots,
    Threads,
    WarpSpecialized,
    role,
)

LOAD, CONSUME = RoleConfig(0, regs=40), RoleConfig(range(4, 8), regs=160)  # the defaults


@dataclass(frozen=True)
class CopyCtx:
    out: cute.Tensor
    iterations: cutlass.Int32


class ConfiguredCopy(WarpSpecialized):
    """Declared defaults, overridden per instance through ``roles=``."""

    def __init__(self, roles: dict | None = None):
        self.roles = roles
        self.data = Ring(2, Threads, Threads, payload=(cutlass.Int32, (32,)))

    def get_name(self) -> str:
        # Deliberately independent of roles: the shared compile cache must distinguish them.
        return "warpdrive_configured_copy"

    @role(warps=0, regs=40)
    @cute.jit
    def load(self, data: Producer, ctx: CopyCtx):
        lane = cute.arch.lane_idx()
        for i in cutlass.range(ctx.iterations, unroll=1):
            with data.produce() as h:
                h.data[lane] = i * 32 + lane
        return data

    @role(warps=range(4, 8), regs=160)
    @cute.jit
    def consume(self, data: Consumer, ctx: CopyCtx):
        lane = cute.arch.lane_idx()
        value = cutlass.Int32(0)
        for _ in cutlass.range(ctx.iterations, unroll=1):
            with data.consume() as h:
                value += h.data[lane]
        ctx.out[cute.arch.thread_idx()[0]] = value
        return data

    @cute.kernel
    def kernel(self, out: cute.Tensor, iterations: cutlass.Int32):
        self.begin_roles()
        self.run_roles(CopyCtx(out, iterations))

    @cute.jit
    def __call__(self, out: cute.Tensor, iterations: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(out, iterations).launch(
            grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream
        )


@jit_cache
def compile_copy(op: ConfiguredCopy):
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),), assumed_align=4),
        cutlass.Int32(1),
        name=op.get_name(),
    )


@dataclass(frozen=True)
class CopyRoles:
    load: RoleConfig = field(default_factory=lambda: LOAD)
    consume: RoleConfig = field(default_factory=lambda: CONSUME)


class SelectedCopy(ConfiguredCopy):
    """The same kernel with ``self.roles`` computed in ``__init__`` from a dataclass."""

    def __init__(self, placement: CopyRoles | None = None):
        placement = placement if placement is not None else CopyRoles()
        super().__init__({"load": placement.load, "consume": placement.consume})


def test_config_matches_shorthand_and_normalizes_warps():
    mutable_warps = [range(4), range(4, 8)]
    config = RoleConfig(mutable_warps, regs=192)
    mutable_warps.clear()
    assert config.warps == (range(4), range(4, 8))
    assert RoleConfig(0) == RoleConfig(range(1)) == RoleConfig([range(1)])
    with pytest.raises(FrozenInstanceError):
        config.regs = 184
    assert ConfiguredCopy().role_config_key() == SelectedCopy().role_config_key()
    assert (
        ConfiguredCopy().describe().replace("ConfiguredCopy", "SelectedCopy")
        == SelectedCopy().describe()
    )
    tuned = {"consume": replace(CONSUME, regs=152)}
    assert (
        ConfiguredCopy(tuned).role_config_key()
        == SelectedCopy(CopyRoles(consume=tuned["consume"])).role_config_key()
    )
    assert "4-7    consume     152" in ConfiguredCopy(tuned).describe()


def test_roles_override_diagnostics():
    with pytest.raises(ProtocolError, match="roles names \\['consumer'\\], which are not roles"):
        ConfiguredCopy({"consumer": CONSUME}).describe()
    with pytest.raises(ProtocolError, match="roles\\['load'\\] must be a RoleConfig"):
        ConfiguredCopy({"load": 40}).describe()
    with pytest.raises(ProtocolError, match="roles must be a mapping"):
        ConfiguredCopy(CopyRoles()).describe()
    op = ConfiguredCopy()
    op.roles = {"load": LOAD}  # after construction: not part of the constructor call
    with pytest.raises(TypeError, match="assigned after construction"):
        op.__cute_cache_key__()


def test_resolution_is_per_instance_and_frozen_after_collection():
    class Selected(WarpSpecialized):
        def __init__(self, setting):
            self.roles = {"worker": setting}

        @role(warps=7)
        @cute.jit
        def worker(self, ctx):
            pass

    first, second = Selected(RoleConfig(0)), Selected(RoleConfig(4, regs=40))
    key = first.role_config_key()
    first.roles["worker"] = RoleConfig(2)  # an in-place change after collection: not seen
    assert first.role_config_key() == key
    second.describe()
    assert second.role_config_key() != key


def test_override_instantiates_a_single_warp_default():
    class Instanced(WarpSpecialized):
        def __init__(self):
            self.roles = {"load": RoleConfig([range(1), range(1, 2)])}
            self.data = Slots(2, Threads, Threads)

        @role(warps=0)
        @cute.jit
        def load(self, index: Instance, data: Producer[Instance], ctx):
            return data

        @role(warps=2)
        @cute.jit
        def consume(self, data: Consumer, ctx):
            return data

    op = Instanced()
    op.describe()
    assert op.data.scope.full_counts == {0: [32, 32]}
    assert op.data.scope.empty_counts == {0: [32, 32]}


def test_resolved_placement_drives_pair_scope():
    op = ConfiguredCopy({"load": replace(LOAD, cta="leader")})
    op.cluster = CtaPair()
    # Pair payloads aren't yet supported: use the same graph as a pure handoff.
    op.data = Ring(2, Threads, Threads)
    op.describe()
    assert op.data.scope.full_counts == {0: [32, 32], 1: [32, 32]}
    assert op.data.scope.empty_counts == {0: [256, 256], 1: [1, 1]}


def test_resolved_graph_validation_and_no_implicit_budget():
    with pytest.raises(ValueError, match="claimed by"):
        ConfiguredCopy({"consume": RoleConfig(0, regs=40)}).describe()
    with pytest.raises(ValueError, match="mixed register budgets"):
        ConfiguredCopy({"load": RoleConfig(0), "consume": RoleConfig(1, regs=40)}).describe()
    op = ConfiguredCopy({"load": RoleConfig(0), "consume": RoleConfig(1)})
    assert op.num_threads == 64  # no setmaxnreg-driven warpgroup padding
    assert all(entry[4] is None for entry in op.role_config_key())


@pytest.mark.parametrize("value", [-1, 32, [], range(0, 4, 2)])
def test_invalid_selected_warps(value):
    with pytest.raises(ValueError, match="role warps"):
        RoleConfig(value)


@pytest.mark.parametrize("value", [True, 0, 23, 41, 264, 40.0])
def test_invalid_selected_register_budget(value):
    with pytest.raises(ValueError, match="role regs"):
        RoleConfig(0, regs=value)


def test_raw_role_config_and_key():
    class Raw(WarpSpecialized):
        def __init__(self, raw):
            self.roles = {"worker": RoleConfig(0, raw=raw)}

        @role(warps=0)
        @cute.jit
        def worker(self, ctx):
            pass

    assert Raw(True).role_config_key() != Raw(False).role_config_key()
    assert Raw(True).num_threads == 32


def test_configured_handoffs_and_cache_isolation(requires_blackwell):
    compile_copy.cache_clear()
    configs = [
        None,
        {"load": RoleConfig(0, regs=48), "consume": RoleConfig(range(4, 8), regs=152)},
        {"consume": RoleConfig(range(8, 12), regs=160)},
    ]
    compiled = [compile_copy(ConfiguredCopy(config)) for config in configs]
    assert compile_copy.cache_info().currsize == len(configs)
    assert len({id(fn) for fn in compiled}) == len(configs)
    assert compile_copy(ConfiguredCopy(None)) is compiled[0]
    # The key is the constructor call: spelling the defaults out is another kernel.
    assert compile_copy(ConfiguredCopy({"load": LOAD, "consume": CONSUME})) is not compiled[0]
    assert compile_copy.cache_info().currsize == len(configs) + 1
    info = compile_copy.cache_info()  # disk loads count as hits too
    assert info.hits + info.misses == len(configs) + 2 and info.currsize == len(configs) + 1
    for config, fn in zip(configs, compiled):
        for n in (1, 7, 19):  # first use and ring wraps, without recompiling
            out = torch.full((384,), -1, device="cuda", dtype=torch.int32)
            fn(out, n)
            expected = torch.full_like(out, -1)
            for warp in (config or {}).get("consume", CONSUME).warps[0]:
                lanes = torch.arange(32, device="cuda", dtype=torch.int32)
                expected[warp * 32 : (warp + 1) * 32] = 32 * n * (n - 1) // 2 + n * lanes
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert compile_copy.cache_info().currsize == len(configs) + 1


def test_omitted_registers_emit_no_adjustment(requires_blackwell, monkeypatch, traced_compiles):
    def unexpected_adjustment(*args, **kwargs):
        pytest.fail("regs=None must not emit setmaxnreg")

    monkeypatch.setattr(cute.arch, "setmaxregister_increase", unexpected_adjustment)
    monkeypatch.setattr(cute.arch, "setmaxregister_decrease", unexpected_adjustment)
    op = ConfiguredCopy({"load": RoleConfig(0), "consume": RoleConfig(1)})
    compile_copy.cache_clear()
    compile_copy(op)


def test_cache_key_is_the_resolved_config(tmp_path, monkeypatch):
    """jit_cache keys the resolved roles and the class, not the name (compile only)."""
    monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(tmp_path))
    built = []

    @jit_cache(persistent=False)
    def compile_marker(op: ConfiguredCopy):
        built.append(op.role_config_key())
        return object()

    first = compile_marker(ConfiguredCopy())
    second = compile_marker(ConfiguredCopy({"consume": RoleConfig(range(4, 8), regs=152)}))
    assert first is not second
    assert compile_marker(ConfiguredCopy()) is first
    assert compile_marker(SelectedCopy()) is not first  # same roles, another class
    assert len(built) == 3
    assert compile_marker.cache_info() == CacheInfo(hits=1, misses=3, currsize=3)


def test_fa4_register_bucket(requires_blackwell):
    pytest.importorskip("flash_attn.cute")
    from attn_gym.warpdrive.examples.fa4_fwd import fa4_fwd

    torch.manual_seed(0)
    q = torch.randn(1, 512, 2, 128, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    tuned = {
        "softmax": RoleConfig((range(4), range(4, 8)), regs=184),
        "correction": RoleConfig(range(8, 12), regs=96),
    }
    expected, expected_lse = fa4_fwd(q, k, v)
    actual, actual_lse = fa4_fwd(q, k, v, roles=tuned)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=0)


@dataclass(frozen=True)
class ProbeCtx:
    out: cute.Tensor  # (threads, 6): each thread's Local values at its hardware thread's row
    reps: cutlass.Int32  # a dynamic trip count: the roles read Local inside a loop


def record_local(out: cute.Tensor, me: Local, tmem_row: bool = False):
    row = cute.arch.thread_idx()[0]
    values = (me.tid, me.warp, me.lane, me.warps, me.threads)
    for column, value in enumerate(values + ((me.tmem_row,) if tmem_row else ())):
        out[row, column] = cutlass.Int32(value)


class LocalProbe(WarpSpecialized):
    """A single-warp role, a two-instance role of unequal widths, a raw role and a warpgroup
    role, each recording its role-local indices (the warpgroup also its TMEM rows)."""

    def __init__(self, roles: dict | None = None):
        self.roles = roles

    def get_name(self) -> str:
        return "warpdrive_local_probe"

    @role(warps=0)
    @cute.jit
    def single(self, ctx: ProbeCtx, me: Local):
        for _ in cutlass.range(ctx.reps, unroll=1):
            record_local(ctx.out, me)

    @role(warps=[range(1, 3), range(3, 4)])
    @cute.jit
    def multi(self, ctx: ProbeCtx, *, me: Local):
        for _ in cutlass.range(ctx.reps, unroll=1):
            record_local(ctx.out, me)

    @role(warps=range(4, 6), raw=True)
    @cute.jit
    def bare(self, ctx: ProbeCtx, me: Local):
        for _ in cutlass.range(ctx.reps, unroll=1):
            record_local(ctx.out, me)

    @role(warps=range(6, 10))
    @cute.jit
    def rows(self, ctx: ProbeCtx, me: Local):
        for _ in cutlass.range(ctx.reps, unroll=1):
            record_local(ctx.out, me, tmem_row=True)

    @cute.kernel
    def kernel(self, out: cute.Tensor, reps: cutlass.Int32):
        self.begin_roles()
        self.run_roles(ProbeCtx(out, reps))

    @cute.jit
    def __call__(self, out: cute.Tensor, reps: cutlass.Int32, stream: cuda.CUstream):
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(out, reps).launch(
            grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream
        )


@jit_cache
def compile_probe(op: LocalProbe):
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(), 6), assumed_align=4),
        cutlass.Int32(1),
        name=op.get_name(),
    )


PROBE_DEFAULTS = {
    "single": [range(1)],
    "multi": [range(1, 3), range(3, 4)],
    "bare": [range(4, 6)],
    "rows": [range(6, 10)],
}
PROBE_MOVED = {  # every instance off its default first warp, warps 3, 4 and 8 left idle
    "single": RoleConfig(5),
    "multi": RoleConfig([range(6, 8), range(2, 3)]),
    "bare": RoleConfig(range(2), raw=True),
    "rows": RoleConfig(range(9, 13)),  # not on a warpgroup boundary
}
PROBE_WIDE = {**PROBE_MOVED, "rows": RoleConfig(range(9, 17))}  # 8 warps over 3 warpgroups


@pytest.mark.parametrize(
    "roles", [None, PROBE_MOVED, PROBE_WIDE], ids=["defaults", "moved", "wide"]
)
def test_local_counts_from_each_instance(requires_blackwell, roles):
    op = LocalProbe(roles)
    placement = PROBE_DEFAULTS if roles is None else {k: c.warps for k, c in roles.items()}
    out = torch.full((op.num_threads, 6), -1, device="cuda", dtype=torch.int32)
    compile_probe(op)(out, 3)
    expected = torch.full_like(out, -1)
    for ranges in placement.values():
        for warps in ranges:
            for t in range(32 * warps.start, 32 * warps.stop):
                local = t - 32 * warps.start
                expected[t, :5] = torch.tensor(
                    [local, local // 32, t % 32, len(warps), 32 * len(warps)]
                )
    rows = range(32 * placement["rows"][0].start, 32 * placement["rows"][0].stop)
    expected[rows, 5] = out[rows, 5]  # checked below by what TMEM rows must satisfy
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    tmem_rows = out[rows, 5].cpu()
    assert sorted(tmem_rows.tolist()) == list(range(len(rows)))  # each row once
    hardware_quadrant = torch.tensor([t // 32 % 4 for t in rows])
    assert torch.equal(tmem_rows // 32 % 4, hardware_quadrant)  # lanes the warp can access


def test_tmem_row_needs_whole_warpgroups(requires_blackwell):
    with pytest.raises(ProtocolError, match="role rows: tmem_row needs whole warpgroups"):
        compile_probe(LocalProbe({"rows": RoleConfig(range(6, 8))}))


# Epilogue on the next warpgroup, and on warps 2-5 (no budgets), which straddle two warpgroups:
# the epilogue's TMEM rows follow the hardware warp quadrant, its other indices the role.
GEMM_PLACEMENTS = {
    "next_warpgroup": {
        "load": RoleConfig(0, regs=40),
        "mma": RoleConfig(1, regs=40, cta="leader"),
        "epilogue": RoleConfig(range(4, 8), regs=160),
    },
    "unaligned": {
        "load": RoleConfig(0),
        "mma": RoleConfig(1, cta="leader"),
        "epilogue": RoleConfig(range(2, 6)),
    },
}


@pytest.mark.parametrize("placement", list(GEMM_PLACEMENTS))
@pytest.mark.parametrize("example", ["gemm", "persistent_gemm"])
def test_moved_gemm_roles_match_defaults(requires_blackwell, example, placement):
    from attn_gym.warpdrive.examples.gemm import Gemm, compile_gemm
    from attn_gym.warpdrive.examples.persistent_gemm import persistent_gemm

    roles = GEMM_PLACEMENTS[placement]
    if example == "gemm":
        roles = {**roles, "mma": replace(roles["mma"], cta="all")}

        def run(a, b, roles=None):
            out = a.new_empty(a.shape[0], b.shape[0])
            compile_gemm(Gemm(roles=roles))(a, b, out)
            return out
    else:
        run = persistent_gemm
    torch.manual_seed(0)
    a = torch.randn(300, 136, device="cuda", dtype=torch.bfloat16)  # ragged M, N and K tiles
    b = torch.randn(520, 136, device="cuda", dtype=torch.bfloat16)
    expected = run(a, b)
    torch.testing.assert_close(expected.float(), a.float() @ b.float().T, atol=0.1, rtol=1e-2)
    torch.testing.assert_close(run(a, b, roles=roles), expected, rtol=0, atol=0)


def test_gemm_epilogue_size_is_checked(requires_blackwell):
    from attn_gym.warpdrive.examples.gemm import Gemm, compile_gemm

    roles = {
        "epilogue": RoleConfig(range(8), regs=160),
        "load": RoleConfig(8, regs=40),
        "mma": RoleConfig(9, regs=40),
    }
    with pytest.raises(ProtocolError, match="128 rows with 4 warps, got 8"):
        compile_gemm(Gemm(roles=roles))


def test_fa4_unaligned_softmax(requires_blackwell):
    """Softmax instances on warps 2-5 and 6-9 (no budgets): their TMEM rows follow the
    hardware warp quadrant, so the output matches the default placement bit for bit."""
    pytest.importorskip("flash_attn.cute")
    from attn_gym.warpdrive.examples.fa4_fwd import fa4_fwd

    torch.manual_seed(0)
    q = torch.randn(1, 512, 2, 128, device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    moved = {
        "mma": RoleConfig(0, cta="leader"),
        "epilogue": RoleConfig(1),
        "softmax": RoleConfig((range(2, 6), range(6, 10))),
        "correction": RoleConfig(range(10, 14)),
        "load": RoleConfig(14),
    }
    expected, expected_lse = fa4_fwd(q, k, v)
    actual, actual_lse = fa4_fwd(q, k, v, roles=moved)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=0)
