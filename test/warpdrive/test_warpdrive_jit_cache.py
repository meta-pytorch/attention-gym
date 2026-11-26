"""Test kernel compile keys, shape reuse, source invalidation, and compiler-process pickling."""

import dataclasses
import importlib
import pickle
import sys
import textwrap
import threading
from pathlib import Path

import pytest
import torch

pytest.importorskip("cutlass.cute")

from attn_gym.warpdrive import (
    AtomicPersistent,
    ClusterLaunchControl,
    CtaPair,
    ProtocolError,
    RoleConfig,
    SingleCta,
)
from attn_gym.warpdrive.cache import CacheInfo, CompileTarget, compile_many, get_compile_target
from attn_gym.warpdrive.cache import _key as cute_key
from attn_gym.warpdrive.cache.target import compile_target_scope
from attn_gym.warpdrive.examples.gemm import Gemm, compile_gemm, gemm
from attn_gym.warpdrive.examples.persistent_gemm import (
    PersistentGemm,
    persistent_gemm,
)
from attn_gym.warpdrive.examples.persistent_gemm import (
    compile_persistent_gemm as compile_pgemm,
)


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("WARPDRIVE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CUTE_DSL_NO_CACHE", raising=False)
    return tmp_path


def key_of(op, fn=compile_gemm) -> str:
    return cute_key.make_key(fn, (op,), {}, CompileTarget("test"))


class ReorderedGemm(Gemm):
    """Same name and constructor as Gemm: only the class tells the two apart."""


class ListGemm(Gemm):
    """Takes its stage count in a list, which the caller may change after construction."""

    def __init__(self, stages: list):
        super().__init__(ab_stages=stages[0])


class Tagged:
    """A mixin whose ``__init__`` assigns before the kernel's own."""

    def __init__(self, *args, tag: str = "x", **kwargs):
        self.tag = tag
        super().__init__(*args, **kwargs)


class TaggedGemm(Tagged, Gemm):
    pass


def test_key_names_each_compile_time_choice():
    """Equal configurations share a key; codegen and trace-check settings get distinct keys."""
    base = key_of(Gemm())
    assert key_of(Gemm()) == key_of(Gemm(ab_stages=3)) == base  # defaults are filled in
    changed = [
        Gemm(ab_stages=4),
        Gemm(acc_stages=2),
        Gemm(trace_backend="iket"),
        Gemm(trace_backend="iket", trace_waits=False),
        Gemm(trace_backend="globaltimer"),
        Gemm(trace_backend="globaltimer", trace_events_per_role=64),
        Gemm(alias_check="strict"),
        ReorderedGemm(),
    ]
    keys = {key_of(op) for op in changed}
    assert base not in keys and len(keys) == len(changed)
    # Trace options that cannot change the code do not split the key.
    assert key_of(Gemm(trace_waits=False, trace_events_per_role=8)) == base
    iket = key_of(Gemm(trace_backend="iket"))
    assert key_of(Gemm(trace_backend="iket", trace_events_per_role=8)) == iket
    pgemms = [
        PersistentGemm(),
        PersistentGemm(schedule=ClusterLaunchControl()),
        PersistentGemm(cluster=CtaPair()),
        PersistentGemm(tile_n=224),
    ]
    assert len({key_of(op, compile_pgemm) for op in pgemms}) == len(pgemms)
    stages = [2]
    listed = ListGemm(stages)
    stages[0] = 4  # the kernel keeps 2 stages; its key must too
    assert key_of(listed) != key_of(ListGemm([4])) and listed.ab.n == 2
    assert key_of(TaggedGemm(tag="a")) != key_of(TaggedGemm(tag="b"))  # mixin init assigns


def test_a_stream_belongs_to_one_kernel():
    """A stream handed to another kernel's constructor moves to it; the kernel it was taken
    from can no longer use it."""

    class Borrowing(PersistentGemm):
        def __init__(self, donor):
            super().__init__()
            self.work = donor.work

    first = PersistentGemm()
    second = Borrowing(first)
    assert first.work.owner is second
    second.describe()
    with pytest.raises(ProtocolError, match="work: this WorkStream now belongs to a Borrowing"):
        first.describe()


def test_pickled_kernel_keeps_its_key():
    """Pickle the constructor call, settings included, for a fresh compiler process."""
    op = PersistentGemm(
        schedule=ClusterLaunchControl(),
        cluster=CtaPair(),
        tile_n=224,
        trace_backend="iket",
        trace_waits=False,
        alias_check="off",
    )
    copy = pickle.loads(pickle.dumps(op))
    assert type(copy) is PersistentGemm and copy is not op
    assert (copy.trace_backend, copy.trace_waits, copy.alias_check) == ("iket", False, "off")
    assert key_of(copy, compile_pgemm) == key_of(op, compile_pgemm)
    stages = [2]
    listed = ListGemm(stages)
    stages[0] = 4  # the key froze the call as made; the rebuilt kernel must be that call too
    copy = pickle.loads(pickle.dumps(listed))
    assert copy.ab.n == 2 and key_of(copy) == key_of(listed)


def test_kernel_is_fixed_once_constructed():
    """Every attribute, setting and work-stream field is fixed once construction returns, so
    the constructor call stays the compile key; with_settings builds a changed kernel."""
    op = PersistentGemm(
        schedule=AtomicPersistent(stages=3), cluster=CtaPair(), trace_protocol=True
    )
    for name, value in (
        ("cluster", SingleCta()),
        ("work", PersistentGemm().work),
        ("trace_protocol", False),
        ("alias_check", "off"),
        ("tile_n", 224),
    ):
        with pytest.raises(ProtocolError, match=f"{name} cannot change after construction"):
            setattr(op, name, value)
    with pytest.raises(ProtocolError, match="WorkStream.schedule cannot change"):
        op.work.schedule = ClusterLaunchControl()
    changed = op.with_settings(trace_protocol=False, tile_n=256, alias_check="strict")
    assert (changed.trace_protocol, changed.tile_n, changed.alias_check) == (False, 256, "strict")
    assert changed.cluster == CtaPair() and changed.work.schedule == AtomicPersistent(stages=3)
    assert (op.trace_protocol, op.tile_n, op.alias_check) == (True, 224, "warn")  # untouched
    with pytest.raises(TypeError, match="no setting or parameter 'stages'"):
        op.with_settings(stages=2)


def test_runtime_key_tells_constructor_calls_classes_and_settings_apart():
    """The in-process key holds the constructor call, the class and the late settings, so two
    kernels share a build only when every compile-time choice matches; it collects no roles."""
    from attn_gym.warpdrive.cache.jit import fast_key

    op = Gemm()
    assert fast_key(op) == fast_key(Gemm())
    assert op._ws_roles is None  # keying a launch collects nothing
    strict = Gemm(alias_check="strict")  # a "warn" build must not satisfy a "strict" one
    assert len({fast_key(k) for k in (op, Gemm(ab_stages=4), ReorderedGemm(), strict)}) == 4


def test_cached_launch_of_a_fresh_kernel_restores_without_collecting(
    requires_blackwell, monkeypatch
):
    """A fresh instance served from the in-process cache takes the build's trace record without
    collecting its roles (the launch costs a dict lookup, not a role walk); describe() collects
    on demand and matches the traced instance."""
    from attn_gym.warpdrive import kernel

    # The suite's fence fixture reports through checks(), which collects; measure the launch.
    monkeypatch.setattr(kernel.WarpSpecialized, "_report_trace", lambda op: None)
    traced = Gemm()
    compile_gemm(traced)
    before = compile_gemm.cache_info()
    fresh = Gemm()
    compile_gemm(fresh)
    assert compile_gemm.cache_info().hits == before.hits + 1
    assert fresh._ws_roles is None and fresh._ws_traced
    assert fresh.describe() == traced.describe()


def test_reused_kernel_is_served_from_its_memo(requires_blackwell, monkeypatch):
    """The second call with one instance is a lookup on the instance, with no key built (the
    kernel is immutable, so the memo stays valid). Clearing the cache invalidates all memos;
    switching targets invalidates only the instance making the call."""
    from attn_gym.warpdrive import WarpSpecialized

    keys = []
    runtime_key = WarpSpecialized.__cute_runtime_key__
    monkeypatch.setattr(
        Gemm, "__cute_runtime_key__", lambda op: keys.append(op) or runtime_key(op)
    )
    op = Gemm()
    fn = compile_gemm(op)
    assert len(keys) == 1
    hits = compile_gemm.cache_info().hits
    assert compile_gemm(op) is fn and len(keys) == 1
    assert compile_gemm.cache_info().hits == hits + 1  # counted like any other hit
    compile_gemm.cache_clear()
    fn = compile_gemm(op)
    assert len(keys) == 2
    other = Gemm()
    target = dataclasses.replace(get_compile_target(), sm_count=1)
    with compile_target_scope(target):
        other_fn = compile_gemm(other)
    assert compile_gemm(op) is fn and len(keys) == 3  # the other call did not retire op's memo
    with compile_target_scope(target):
        assert compile_gemm(op) is other_fn
    assert compile_gemm(op) is fn and len(keys) == 5


def test_get_or_build_reuses_one_kernel_per_constructor_call():
    """Kernel.get_or_build() returns one instance per class and constructor call, settings
    included; direct construction always builds a new kernel."""
    shared = Gemm.get_or_build(ab_stages=4)
    assert Gemm.get_or_build(ab_stages=4) is shared
    assert Gemm.get_or_build(ab_stages=2) is not shared
    assert Gemm.get_or_build(ab_stages=4, alias_check="strict") is not shared
    assert ReorderedGemm.get_or_build(ab_stages=4) is not shared
    assert Gemm(ab_stages=4) is not shared
    load, mma = RoleConfig(4, regs=48), RoleConfig(5, regs=48)
    roles = PersistentGemm.get_or_build(roles={"load": load, "mma": mma})
    assert PersistentGemm.get_or_build(roles={"mma": mma, "load": load}) is roles


def test_cache_clear_drops_in_process_builds(requires_blackwell):
    """``cache_clear()`` empties the in-process lookup too: the next launch loads the build
    again instead of reusing one the caller cleared."""
    compile_gemm(Gemm())
    compile_gemm.cache_clear()
    assert compile_gemm.cache_info() == CacheInfo(hits=0, misses=0, currsize=0)
    compile_gemm(Gemm())
    assert compile_gemm.cache_info().currsize == 1


def test_uncopyable_constructor_argument_keeps_the_kernel_usable_but_unkeyed():
    """Keying and pickling require a copyable constructor call.

    An uncopyable argument makes both raise; construction and describe() still work.
    """

    class Locked(Gemm):
        def __init__(self, lock):
            super().__init__()

    op = Locked(threading.Lock())
    assert op.describe().startswith("Locked: 8 warps, 256 threads")
    with pytest.raises(TypeError, match="cannot be copied"):
        key_of(op)
    with pytest.raises(TypeError, match="cannot be copied"):
        pickle.dumps(op)


# A kernel module outside the package, as a test or user file would define one: its own source
# is hashed for its compile function and, through module_source_digest, for its kernel class.
FILL_MODULE = """
from dataclasses import dataclass

import cutlass
from cutlass import cute

from attn_gym.warpdrive.cache import jit_cache
from attn_gym.warpdrive.cache import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive import WarpSpecialized, role

VALUE = {value}


@dataclass(frozen=True)
class FillCtx:
    out: cute.Tensor


class Fill(WarpSpecialized):
    def get_name(self):
        return "warpdrive_jit_cache_fill"

    @role(warps=1)
    @cute.jit
    def fill(self, ctx: FillCtx):
        ctx.out[cute.arch.lane_idx()] = VALUE

    @cute.kernel
    def kernel(self, out: cute.Tensor):
        self.begin_roles()
        self.run_roles(FillCtx(out))

    @cute.jit
    def __call__(self, out: cute.Tensor, stream):
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(out).launch(grid=(1, 1, 1), block=(self.num_threads, 1, 1), stream=stream)


@jit_cache
def compile_fill(op):
    return compile_tvm_ffi(op, make_fake_compact_tensor(cutlass.Int32, (32,), assumed_align=4))
"""


def test_editing_the_kernel_module_recompiles(
    requires_blackwell, isolated_cache, tmp_path_factory
):
    """A warm process loads the kernel from disk; editing its module compiles a new entry, and
    restoring the source finds the old one again."""
    directory = tmp_path_factory.mktemp("kernels")
    source = directory / "warpdrive_fill_kernel.py"
    sys.path.insert(0, str(directory))
    try:

        def load(value: int):
            source.write_text(textwrap.dedent(FILL_MODULE.format(value=value)))
            cute_key.source_fingerprint.cache_clear()
            cute_key.module_source_digest.cache_clear()
            if "warpdrive_fill_kernel" in sys.modules:
                return importlib.reload(sys.modules["warpdrive_fill_kernel"])
            return importlib.import_module("warpdrive_fill_kernel")

        def run(module) -> list[int]:
            out = torch.zeros(32, device="cuda", dtype=torch.int32)
            module.compile_fill(module.Fill())(out)
            return out.tolist()

        module = load(7)
        assert run(module) == [7] * 32
        assert module.compile_fill.cache_info() == CacheInfo(hits=0, misses=1, currsize=1)
        module.compile_fill.cache_clear()  # a warm process: the entry is on disk
        assert run(module) == [7] * 32
        assert module.compile_fill.cache_info() == CacheInfo(hits=1, misses=0, currsize=1)

        module = load(1234)  # another size: a same-size edit within a second reuses the .pyc
        assert run(module) == [1234] * 32
        assert module.compile_fill.cache_info() == CacheInfo(hits=0, misses=1, currsize=1)
        assert len(list(isolated_cache.rglob("*.o"))) == 2

        module = load(7)
        assert run(module) == [7] * 32
        assert module.compile_fill.cache_info() == CacheInfo(hits=1, misses=0, currsize=1)
    finally:
        sys.path.remove(str(directory))
        sys.modules.pop("warpdrive_fill_kernel", None)


# Host-only modules: rendering and trace decoding never run while tracing.
HOST_ONLY = (
    "describe_graph.py",
    "describe_html.py",
    "html_page.py",
    "pipeline_plan.py",
    "schedule_html.py",
    "trace_session.py",
    "examples/schedule_views.py",
    "examples/qwen_megakernel/viz.py",
    "examples/qwen_megakernel/trace.py",
    "examples/qwen_megakernel/schedule.py",
    "examples/qwen_megakernel/runner.py",
)


@pytest.mark.parametrize("example", ["gemm", "persistent_gemm", "fa4_bwd", "qwen_megakernel/core"])
def test_host_only_modules_stay_out_of_compile_keys(example):
    """Renderers, trace decoding, the pipeline plan export and the root package's utilities do
    not invalidate kernels; of the profiler, only its device ops are inside the closure."""
    warpdrive = Path(importlib.import_module("attn_gym.warpdrive").__file__).parent
    profiler = warpdrive / "profiler"
    closure = cute_key.module_closure(warpdrive / "examples" / f"{example}.py")
    assert warpdrive / "kernel.py" in closure
    assert not {warpdrive / name for name in HOST_ONLY} & set(closure)
    assert all(source.is_relative_to(warpdrive) for source in closure)
    assert {path for path in closure if path.is_relative_to(profiler)} == {profiler / "ops.py"}


# M: 1, a multiple of 16, odd, a tile edge, several waves; N and K: multiples of 8 on both sides
# of 16. Each is a runtime value, so every shape reuses the one compiled kernel.
SHAPES = [(1, 8, 8), (16, 256, 64), (129, 264, 136), (128, 520, 24), (4097, 2056, 520)]


@pytest.mark.parametrize(
    "entry, compile_fn",
    [(gemm, compile_gemm), (persistent_gemm, compile_pgemm)],
    ids=["gemm", "persistent_gemm"],
)
def test_entry_points_reuse_one_kernel_across_shapes(requires_blackwell, entry, compile_fn):
    compile_fn.cache_clear()
    for m, n, k in SHAPES:
        torch.manual_seed(m)
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(a.float(), b.float())
        out = entry(a, b)
        assert (out.float() - ref).abs().max().item() < 0.05 * ref.abs().max().item() + 1e-2
    info = compile_fn.cache_info()  # a disk load counts as a hit, a compile as a miss
    assert info.currsize == 1 and info.hits + info.misses == len(SHAPES)


def test_compile_many_builds_kernels_in_a_fresh_process(requires_blackwell, isolated_cache):
    """Compile pickled kernels in workers forked from a fresh compiler process.

    The caller already owns a CUDA context, so forking it would poison the workers.
    The caller only loads artifacts.
    """
    a = torch.randn(300, 136, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(520, 136, device="cuda", dtype=torch.bfloat16)
    ops = [Gemm(ab_stages=2), Gemm(ab_stages=4)]
    compile_gemm.cache_clear()
    target = get_compile_target()
    with compile_target_scope(dataclasses.replace(target, sm_count=1)):
        fns = compile_many(compile_gemm, [(op,) for op in ops], workers=2, target=target)
    assert compile_gemm.cache_info() == CacheInfo(hits=2, misses=0, currsize=2)
    assert len(list(isolated_cache.rglob("*.o"))) == 2
    ref = torch.nn.functional.linear(a.float(), b.float())
    for fn in fns:
        out = torch.empty(300, 520, device="cuda", dtype=torch.bfloat16)
        fn(a, b, out)
        assert (out.float() - ref).abs().max().item() < 0.05 * ref.abs().max().item() + 1e-2
