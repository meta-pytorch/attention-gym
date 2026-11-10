"""Test kernel compile keys, shape reuse, source invalidation, and compiler-process pickling."""

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
)
from attn_gym.warpdrive.cache import CacheInfo, CompileTarget, compile_many
from attn_gym.warpdrive.cache import _key as cute_key
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
    """Equal configurations share a key; codegen and trace-check settings get distinct keys.

    Compile only.
    """
    base = key_of(Gemm())
    assert key_of(Gemm()) == key_of(Gemm(ab_stages=3)) == base  # defaults are filled in
    strict = Gemm()
    strict.alias_check = "strict"
    changed = [
        Gemm(ab_stages=4),
        Gemm(acc_stages=2),
        Gemm().enable_iket(),
        Gemm().enable_iket(waits=False),
        Gemm().enable_trace("globaltimer"),
        Gemm().enable_trace("globaltimer", events_per_role=64),
        strict,
        ReorderedGemm(),
    ]
    keys = {key_of(op) for op in changed}
    assert base not in keys and len(keys) == len(changed)
    # Trace options that cannot change the code do not split the key.
    assert key_of(Gemm().enable_trace(None, waits=False, events_per_role=8)) == base
    assert key_of(Gemm().enable_trace("iket", events_per_role=8)) == key_of(Gemm().enable_iket())
    pgemms = [
        PersistentGemm(),
        PersistentGemm(schedule=ClusterLaunchControl()),
        PersistentGemm(cluster=CtaPair()),
        PersistentGemm(tile_n=224),
    ]
    assert len({key_of(op, compile_pgemm) for op in pgemms}) == len(pgemms)
    reassigned = PersistentGemm()
    reassigned.work = PersistentGemm(schedule=AtomicPersistent(stages=3)).work
    assert key_of(reassigned, compile_pgemm) != key_of(pgemms[0], compile_pgemm)
    stages = [2]
    listed = ListGemm(stages)
    stages[0] = 4  # the kernel keeps 2 stages; its key must too
    assert key_of(listed) != key_of(ListGemm([4])) and listed.ab.n == 2
    assert key_of(TaggedGemm(tag="a")) != key_of(TaggedGemm(tag="b"))  # mixin init assigns
    retiled = PersistentGemm()
    retiled.tile_n = 224  # the constructor call no longer describes the kernel
    with pytest.raises(TypeError, match=r"\['tile_n'\] assigned after construction"):
        key_of(retiled, compile_pgemm)


def test_pickled_kernel_keeps_its_key():
    """Pickle the constructor call and host settings for a fresh compiler process.

    Compile only.
    """
    op = PersistentGemm(schedule=ClusterLaunchControl(), cluster=CtaPair(), tile_n=224)
    op.enable_iket(waits=False).alias_check = "off"
    copy = pickle.loads(pickle.dumps(op))
    assert type(copy) is PersistentGemm and copy is not op
    assert (copy.trace_backend, copy.trace_waits, copy.alias_check) == ("iket", False, "off")
    assert key_of(copy, compile_pgemm) == key_of(op, compile_pgemm)
    moved = PersistentGemm()
    moved.cluster = CtaPair()
    moved.work = PersistentGemm(schedule=AtomicPersistent(stages=3)).work
    copy = pickle.loads(pickle.dumps(moved))
    assert copy.work.owner is copy and copy.cluster == CtaPair()
    assert key_of(copy, compile_pgemm) == key_of(moved, compile_pgemm)
    stages = [2]
    listed = ListGemm(stages)
    stages[0] = 4  # the key froze the call as made; the rebuilt kernel must be that call too
    copy = pickle.loads(pickle.dumps(listed))
    assert copy.ab.n == 2 and key_of(copy) == key_of(listed)


def test_collected_kernel_rejects_cluster_and_stream_reassignment():
    """Freeze cluster, work-stream, and trace_protocol settings at first collection.

    Before describe(), keying, or tracing, they may change. Later changes raise because
    derived channels would be stale.
    """
    op = PersistentGemm()
    op.cluster = CtaPair()
    op.work = PersistentGemm(schedule=AtomicPersistent(stages=3)).work
    op.trace_protocol = True
    before = op.describe()
    with pytest.raises(ProtocolError, match="cluster.*after collection"):
        op.cluster = CtaPair()
    with pytest.raises(ProtocolError, match="work.*after collection"):
        op.work = PersistentGemm(schedule=AtomicPersistent(stages=3)).work
    with pytest.raises(ProtocolError, match="trace_protocol.*after collection"):
        op.trace_protocol = False
    assert op.describe() == before and op.work.roles
    with pytest.raises(AttributeError):
        op.num_warps = 4  # derived from the roles, not a setting
    assert op.__cute_cache_key__()  # a rejected assignment leaves the key intact


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
    "examples/qwen_megakernel_viz.py",
    "examples/qwen_megakernel_trace.py",
)


@pytest.mark.parametrize("example", ["gemm", "persistent_gemm", "fa4_bwd", "qwen_megakernel"])
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
    fns = compile_many(compile_gemm, [(op,) for op in ops], workers=2)
    assert compile_gemm.cache_info() == CacheInfo(hits=2, misses=0, currsize=2)
    assert len(list(isolated_cache.rglob("*.o"))) == 2
    ref = torch.nn.functional.linear(a.float(), b.float())
    for fn in fns:
        out = torch.empty(300, 520, device="cuda", dtype=torch.bfloat16)
        fn(a, b, out)
        assert (out.float() - ref).abs().max().item() < 0.05 * ref.abs().max().item() + 1e-2
