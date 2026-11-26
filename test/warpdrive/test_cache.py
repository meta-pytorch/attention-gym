from __future__ import annotations

import gc
import importlib.util
import multiprocessing
import os
import struct
import sys
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import pytest

from attn_gym.warpdrive.cache import _key as cute_key
from attn_gym.warpdrive.cache import compile as cute_compile
from attn_gym.warpdrive.cache import (
    compile_many,
    compile_tvm_ffi,
    precompile_calls,
    precompile_many,
)
from attn_gym.warpdrive.cache import jit as cute_cache
from attn_gym.warpdrive.cache import target as cute_target

_DRIVER_LOG_ENV = "ATTN_GYM_TEST_CUTE_DRIVER_LOG"
_DRIVER_READY_ENV = "ATTN_GYM_TEST_CUTE_DRIVER_READY"
_DRIVER_EXPECTED_ENV = "ATTN_GYM_TEST_CUTE_DRIVER_EXPECTED"
_HANG_ENV = "ATTN_GYM_TEST_CUTE_HANG"


class CompileConfig(NamedTuple):
    """Typed static config used across the compiler-process boundary."""

    variant: str
    timing: float


class FakeCompiled:
    """Minimal compiled callable that exports a loadable test artifact."""

    def __init__(self, value: str):
        self.value = value

    def __call__(self) -> str:
        return self.value

    def export_to_c(self, object_file_path: str, function_name: str) -> None:
        Path(object_file_path).write_text(f"{function_name}:{self.value}")


def load_fake_compiled(path: Path) -> FakeCompiled:
    """Load the test artifact format written by ``FakeCompiled``."""
    function_name, separator, value = path.read_text().partition(":")
    if separator != ":" or function_name != cute_cache.EXPORT_FUNCTION_NAME:
        raise RuntimeError("corrupt test artifact")
    return FakeCompiled(value)


@cute_cache.jit_cache
def compile_test_variant(config: CompileConfig) -> FakeCompiled:
    """Importable fake compiler used to exercise the fresh compiler process."""
    ready_directory = os.getenv(_DRIVER_READY_ENV)
    expected = int(os.getenv(_DRIVER_EXPECTED_ENV, "0"))
    if ready_directory and expected:
        ready_path = Path(ready_directory)
        ready_path.mkdir(parents=True, exist_ok=True)
        (ready_path / f"{config.variant}-{os.getpid()}.ready").touch()
        deadline = time.monotonic() + 10
        while len(list(ready_path.glob("*.ready"))) < expected:
            if time.monotonic() >= deadline:
                raise RuntimeError("compiler workers did not run concurrently")
            time.sleep(0.01)

    log_path = os.getenv(_DRIVER_LOG_ENV)
    if log_path:
        with Path(log_path).open("a") as log:
            log.write(
                f"{os.getpid()},{os.getppid()},{config.variant},{os.getenv('CUTE_DSL_ARCH', '')}\n"
            )
    return FakeCompiled(config.variant)


@cute_cache.jit_cache
def compile_other_variant(config: CompileConfig) -> FakeCompiled:
    """A second importable fake compiler, for batches that mix compile functions."""
    return compile_test_variant.__wrapped__(config._replace(variant=f"other-{config.variant}"))


@cute_cache.jit_cache(lock_timeout=1.0)
def compile_hanging_variant(config: CompileConfig) -> FakeCompiled:
    """With ``_HANG_ENV`` naming a directory, adds a file named by its pid there and never
    returns, holding its entry's lock; else compiles at once."""
    if hang := os.getenv(_HANG_ENV):
        (Path(hang) / str(os.getpid())).touch()
        while True:
            time.sleep(1)
    return FakeCompiled(config.variant)


@cute_cache.jit_cache
def compile_broken_variant(config: CompileConfig) -> FakeCompiled:
    raise ValueError(f"cannot compile {config.variant}")


def wait_until(condition, timeout: float = 60.0):
    """``condition()``'s first true value, polled until ``timeout`` seconds pass."""
    deadline = time.monotonic() + timeout
    while not (value := condition()):
        if time.monotonic() >= deadline:
            raise AssertionError(f"{condition} still false after {timeout} s")
        time.sleep(0.05)
    return value


def alive(pid: int) -> bool:
    """Whether process ``pid`` runs (a zombie has exited)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rpartition(")")[2].split()[0] != "Z"


def wait_for_processes(processes, timeout: float = 15.0) -> None:
    """Join child processes and fail without leaving a process behind."""
    deadline = time.monotonic() + timeout
    try:
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        failures = [
            (process.pid, process.exitcode) for process in processes if process.exitcode != 0
        ]
        assert failures == []
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)


@pytest.fixture(autouse=True)
def reset_compile_target():
    cute_target.set_compile_target(None)
    yield
    cute_target.set_compile_target(None)


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Use deterministic fingerprints and an isolated persistent cache."""
    cache_directory = tmp_path / "cache"
    monkeypatch.setenv("WARPDRIVE_CACHE_DIR", str(cache_directory))
    monkeypatch.delenv("CUTE_DSL_NO_CACHE", raising=False)
    monkeypatch.setattr(
        cute_cache,
        "_source_fingerprint",
        lambda _fn, _extra_sources=(): "test-source",
    )
    monkeypatch.setattr(
        cute_cache,
        "get_compile_target",
        lambda: cute_target.CompileTarget("test"),
    )
    monkeypatch.setattr(cute_cache, "_load_compiled", load_fake_compiled)
    return cache_directory


def test_compile_tvm_ffi_enforces_the_compile_contract():
    import torch

    with pytest.raises(TypeError, match=r"get_name\(\) or name="):
        compile_tvm_ffi(object())
    with pytest.raises(ValueError, match="lowercase"):
        compile_tvm_ffi(object(), name="Bad-Name")
    with pytest.raises(TypeError, match="fake CuTe tensors"):
        compile_tvm_ffi(object(), {"nested": torch.empty(0)}, name="valid_name")


def test_compile_tvm_ffi_adds_fake_stream_and_typed_option(monkeypatch):
    cute = pytest.importorskip("cutlass.cute")
    from attn_gym.warpdrive.compat import COMPILE_OPTIONS

    fake_stream = object()
    fake_tensor = object()
    observed = {}

    class FakeCompile:
        def __getitem__(self, option):
            observed["option"] = option
            return self

        def __call__(self, entrypoint, *args, **kwargs):
            observed.update(entrypoint=entrypoint, args=args, kwargs=kwargs)
            return "compiled"

    class EntryPoint:
        @staticmethod
        def get_name() -> str:
            return "stable_kernel_name"

        def set_name_prefix(self, name: str) -> None:
            observed["name_prefix"] = name

    monkeypatch.setattr(cute, "compile", FakeCompile())
    monkeypatch.setattr(
        cute.runtime,
        "make_fake_stream",
        lambda *, use_tvm_ffi_env_stream: fake_stream if use_tvm_ffi_env_stream else None,
    )
    entrypoint = EntryPoint()

    assert compile_tvm_ffi(entrypoint, fake_tensor) == "compiled"
    assert observed == {
        "option": (cute.EnableTVMFFI, *COMPILE_OPTIONS),
        "entrypoint": entrypoint,
        "args": (fake_tensor, fake_stream),
        "kwargs": {},
        "name_prefix": "stable_kernel_name",
    }

    compile_tvm_ffi(entrypoint, fake_tensor, opt_level=2)
    *options, level = observed["option"]
    assert options == [cute.EnableTVMFFI, *COMPILE_OPTIONS] and isinstance(level, cute.OptLevel)


FIRST_NAN = struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0]
OTHER_NAN = struct.unpack("!d", bytes.fromhex("7ff8000000000002"))[0]


@pytest.mark.parametrize(
    ("first", "second"),
    [(True, 1), (1, 1.0), (0.0, -0.0), (FIRST_NAN, OTHER_NAN)],
    ids=["bool_int", "int_float", "signed_zero", "nan_payload"],
)
def test_runtime_cache_preserves_scalar_representation(isolated_cache, first, second):
    """Values that compare equal (or, for NaN, unequal) still key by type and bit pattern."""

    def describe(value) -> str:
        if isinstance(value, float):
            return f"float:{struct.pack('!d', value).hex()}"
        return f"{type(value).__name__}:{value}"

    @cute_cache.jit_cache
    def compile_kernel(value) -> FakeCompiled:
        return FakeCompiled(describe(value))

    for value in (first, second, first, second):
        assert compile_kernel(value)() == describe(value)
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=2, misses=2, currsize=2)


def test_explicit_cache_key_is_shared_by_memory_disk_and_precompile(isolated_cache):
    compile_count = 0

    @cute_cache.jit_cache(cache_key=lambda _value: "shared")
    def compile_kernel(value: str) -> FakeCompiled:
        nonlocal compile_count
        compile_count += 1
        return FakeCompiled(value)

    assert not compile_kernel.is_cached("first")
    compile_kernel.precompile("first")
    assert compile_kernel.is_cached("second")
    assert compile_kernel("second")() == "first"
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=1, misses=0, currsize=1)
    compile_kernel.cache_clear()
    assert compile_kernel("third")() == "first"
    assert compile_count == 1
    assert len(list(isolated_cache.rglob("*.o"))) == 1

    @cute_cache.jit_cache(cache_key=lambda _value: [])
    def unhashable(value: str) -> FakeCompiled:
        return FakeCompiled(value)

    with pytest.raises(TypeError, match="cache_key must return a hashable value"):
        unhashable("value")


def test_uncached_calls_bypass_both_caches(isolated_cache):
    compile_count = 0

    @cute_cache.jit_cache(uncached=lambda key: key.startswith("trace"))
    def compile_kernel(key: str) -> FakeCompiled:
        nonlocal compile_count
        compile_count += 1
        return FakeCompiled(key)

    assert compile_kernel("trace")() == "trace"
    assert compile_kernel("trace")() == "trace"
    assert compile_kernel("plain")() == "plain"
    assert compile_kernel("plain")() == "plain"
    assert compile_count == 3
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=1, misses=3, currsize=1)
    assert len(list(isolated_cache.rglob("*.o"))) == 1


class TracedConfig:
    """A compile argument that stores trace results, like a warpdrive kernel."""

    def __init__(self, variant: str):
        self.variant = variant
        self.found = None

    def __cute_cache_key__(self) -> str:
        return self.variant

    def __cute_trace_record__(self) -> str:
        return self.found

    def __cute_restore_trace__(self, record: str) -> None:
        self.found = record


def test_a_class_argument_keys_by_name_and_keeps_no_trace_record(isolated_cache):
    """A class argument keys by name, even if its instances store trace records.

    Cold, memory, and disk calls must not call the class's unbound trace hooks.
    """

    @cute_cache.jit_cache
    def compile_kernel(kind: type) -> FakeCompiled:
        return FakeCompiled(kind.__name__)

    assert compile_kernel(TracedConfig)() == "TracedConfig"
    assert compile_kernel(TracedConfig)() == "TracedConfig"
    assert compile_kernel(CompileConfig)() == "CompileConfig"
    compile_kernel.cache_clear()  # also resets the counts
    assert compile_kernel(TracedConfig)() == "TracedConfig"  # from disk
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=1, misses=0, currsize=1)


def test_cached_calls_restore_the_trace_record(isolated_cache):
    compile_count = 0

    @cute_cache.jit_cache
    def compile_kernel(config: TracedConfig) -> FakeCompiled:
        nonlocal compile_count
        compile_count += 1
        config.found = f"traced {config.variant}"
        return FakeCompiled(config.variant)

    first = TracedConfig("a")
    assert compile_kernel(first)() == "a" and first.found == "traced a"
    again = TracedConfig("a")  # memory hit, another instance
    assert compile_kernel(again)() == "a" and again.found == "traced a"
    compile_kernel.cache_clear()
    warm = TracedConfig("a")  # disk hit
    assert compile_kernel(warm)() == "a" and warm.found == "traced a"
    assert compile_count == 1
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=1, misses=0, currsize=1)

    next(isolated_cache.rglob("*.trace")).write_bytes(b"truncated")
    compile_kernel.cache_clear()
    rebuilt = TracedConfig("a")
    assert compile_kernel(rebuilt)() == "a" and rebuilt.found == "traced a"
    assert compile_count == 2  # an unreadable record recompiles and republishes it
    assert compile_kernel.is_cached(TracedConfig("a"))
    next(isolated_cache.rglob("*.trace")).unlink()
    assert not compile_kernel.is_cached(TracedConfig("a"))


class MemoOwner:
    """A reusable argument with a ``__cute_memo__``, as a warpdrive kernel has."""

    def __init__(self):
        self.__cute_memo__ = {}

    def __cute_cache_key__(self) -> str:
        return "owner"


def test_retired_builds_leave_the_memos_of_live_arguments(isolated_cache, monkeypatch):
    """``cache_clear()`` and a new compile target remove this function's builds from the memos
    of arguments still alive (a kernel kept by ``get_or_build``), so launch -> clear -> launch
    cycles hold one build and a cleared one is freed; another function's entries stay."""

    @cute_cache.jit_cache(persistent=False)
    def compile_kernel(owner: MemoOwner) -> FakeCompiled:
        return FakeCompiled("kernel")

    @cute_cache.jit_cache(persistent=False)
    def compile_other(owner: MemoOwner) -> FakeCompiled:
        return FakeCompiled("other")

    first_target = target = cute_target.CompileTarget("first")
    monkeypatch.setattr(cute_cache, "get_compile_target", lambda: target)
    owner = MemoOwner()
    other = compile_other(owner)
    first = weakref.ref(compile_kernel(owner))
    for _ in range(3):
        compile_kernel.cache_clear()
        compile_kernel(owner)
    gc.collect()
    assert first() is None
    assert len(owner.__cute_memo__) == 2  # one build of each function
    target = cute_target.CompileTarget("second")
    current = weakref.ref(compile_kernel(owner))  # retires the other target's build
    compile_kernel.cache_clear()
    gc.collect()
    assert current() is None
    assert len(owner.__cute_memo__) == 1
    target = first_target
    assert compile_other(owner) is other


def test_runtime_cache_includes_compile_target(isolated_cache, monkeypatch):
    compile_count = 0
    target = cute_target.CompileTarget("first")
    monkeypatch.setattr(cute_cache, "get_compile_target", lambda: target)

    @cute_cache.jit_cache
    def compile_kernel(value: str) -> FakeCompiled:
        nonlocal compile_count
        compile_count += 1
        return FakeCompiled(cute_cache.get_compile_target().device_type)

    assert compile_kernel("shared")() == "first"
    target = cute_target.CompileTarget("second")
    assert compile_kernel("shared")() == "second"
    target = cute_target.CompileTarget("first")
    assert compile_kernel("shared")() == "first"
    assert compile_count == 2
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=1, misses=2, currsize=2)


def test_runtime_key_fuzz_matches_persistent_key_equivalence(isolated_cache):
    same_nan, other_nan = FIRST_NAN, OTHER_NAN
    same_nan_copy = struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0]
    calls = [
        ((True,), {}),
        ((1,), {}),
        ((1.0,), {}),
        ((0.0,), {}),
        ((-0.0,), {}),
        ((same_nan,), {}),
        ((same_nan_copy,), {}),
        ((other_nan,), {}),
        ((CompileConfig("shared", 1.0),), {}),
        ((["nested", {"value": True}],), {}),
        ((["nested", {"value": 1}],), {}),
        (({same_nan: "value"},), {}),
        (({same_nan: "value", same_nan_copy: "value"},), {}),
        ((), {"first": 1, "second": 2}),
        ((), {"second": 2, "first": 1}),
    ]
    target = cute_target.CompileTarget("test")

    def compile_kernel(value=None, **kwargs):
        return value, kwargs

    runtime_keys = [cute_cache._make_runtime_key(*call, target) for call in calls]
    persistent_keys = [cute_cache._make_key(compile_kernel, *call, target) for call in calls]
    for left in range(len(calls)):
        for right in range(len(calls)):
            if runtime_keys[left] == runtime_keys[right]:
                assert persistent_keys[left] == persistent_keys[right]

    assert runtime_keys[5] == runtime_keys[6]
    assert runtime_keys[5] != runtime_keys[7]
    assert runtime_keys[-2] == runtime_keys[-1]

    bool_target = cute_target.CompileTarget("test", sm_count=True)
    int_target = cute_target.CompileTarget("test", sm_count=1)
    assert cute_cache._make_runtime_key((), {}, bool_target) != cute_cache._make_runtime_key(
        (), {}, int_target
    )


def test_runtime_cache_snapshots_mutable_arguments(isolated_cache):
    compile_count = 0

    @cute_cache.jit_cache
    def compile_kernel(values: list[str]) -> FakeCompiled:
        nonlocal compile_count
        compile_count += 1
        return FakeCompiled(values[0])

    values = ["first"]
    assert compile_kernel(values)() == "first"
    assert compile_kernel(["first"])() == "first"
    values[0] = "second"
    assert compile_kernel(values)() == "second"
    assert compile_kernel(["second"])() == "second"
    compile_kernel.cache_clear()
    assert compile_kernel(["first"])() == "first"
    assert compile_count == 2


def test_same_key_compiles_once_across_threads(isolated_cache):
    compile_count = 0
    count_lock = threading.Lock()

    @cute_cache.jit_cache
    def compile_kernel(key: str) -> FakeCompiled:
        nonlocal compile_count
        with count_lock:
            compile_count += 1
        time.sleep(0.1)
        return FakeCompiled(key)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(compile_kernel, ["shared"] * 8))

    assert compile_count == 1
    assert [result() for result in results] == ["shared"] * 8
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=7, misses=1, currsize=1)
    assert len(list(isolated_cache.rglob("*.o"))) == 1


def test_distinct_keys_serialize_compilation_within_process(isolated_cache):
    active_compiles = 0
    max_active_compiles = 0
    count_lock = threading.Lock()

    @cute_cache.jit_cache
    def compile_kernel(key: str) -> FakeCompiled:
        nonlocal active_compiles, max_active_compiles
        with count_lock:
            active_compiles += 1
            max_active_compiles = max(max_active_compiles, active_compiles)
        time.sleep(0.1)
        with count_lock:
            active_compiles -= 1
        return FakeCompiled(key)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(compile_kernel, ["first", "second"]))

    assert {result() for result in results} == {"first", "second"}
    assert max_active_compiles == 1
    assert compile_kernel.cache_info().misses == 2


def test_corrupt_artifact_is_recompiled(isolated_cache):
    compile_count = 0

    @cute_cache.jit_cache
    def compile_kernel(key: str) -> FakeCompiled:
        nonlocal compile_count
        compile_count += 1
        return FakeCompiled(key)

    assert compile_kernel("key")() == "key"
    compile_kernel.cache_clear()
    artifact = next(isolated_cache.rglob("*.o"))
    artifact.write_bytes(b"corrupt")

    assert compile_kernel("key")() == "key"
    assert compile_count == 2
    assert artifact.read_text() == f"{cute_cache.EXPORT_FUNCTION_NAME}:key"


def test_failed_export_does_not_publish_partial_artifact(isolated_cache):
    class FailedExport(FakeCompiled):
        def export_to_c(self, object_file_path: str, function_name: str) -> None:
            Path(object_file_path).write_bytes(b"partial")
            raise RuntimeError("export failed")

    @cute_cache.jit_cache
    def compile_kernel() -> FailedExport:
        return FailedExport("value")

    assert compile_kernel()() == "value"
    assert not list(isolated_cache.rglob("*.o"))


def test_persistent_cache_can_be_disabled_with_an_option(isolated_cache):
    compile_count = 0

    @cute_cache.jit_cache(persistent=False)
    def compile_kernel() -> FakeCompiled:
        nonlocal compile_count
        compile_count += 1
        return FakeCompiled("value")

    compile_kernel()
    compile_kernel.cache_clear()
    compile_kernel()

    assert compile_count == 2
    assert not list(isolated_cache.rglob("*.o"))


def test_bad_fork_check_precedes_cached_target_lookup(monkeypatch):
    """Reject a child that inherited a target cached by its parent."""
    import torch

    class Properties:
        major = 10
        minor = 0
        name = "test-gpu"
        multi_processor_count = 100

    bad_fork = False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: Properties())
    monkeypatch.setattr(torch.cuda, "_is_in_bad_fork", lambda: bad_fork, raising=False)
    cute_target._query_compile_target.cache_clear()
    try:
        assert cute_target.detect_compile_target() == cute_target.CompileTarget(
            device_type="cuda",
            capability=(10, 0),
            name="test-gpu",
            sm_count=100,
        )
        bad_fork = True
        with pytest.raises(RuntimeError, match="forked CUDA child"):
            cute_target.detect_compile_target()
    finally:
        cute_target._query_compile_target.cache_clear()


@pytest.mark.parametrize(
    ("configured_arch", "physical", "expected"),
    (
        ("sm_90a", (10, 0), (9, 0)),
        ("sm_100a", (9, 0), (10, 0)),
        ("sm_103a", (10, 0), (10, 3)),
        (None, (10, 3), (10, 3)),
        (None, None, None),
        ("native", (10, 0), ValueError),
    ),
)
def test_compile_target_effective_capability(configured_arch, physical, expected):
    """Prefer an explicit code-generation architecture over the physical GPU."""
    target = cute_target.CompileTarget(
        device_type="cuda",
        configured_arch=configured_arch,
        capability=physical,
    )
    if expected is ValueError:
        with pytest.raises(ValueError, match="configured CUDA architecture"):
            _ = target.effective_capability
    else:
        assert target.effective_capability == expected


def test_concurrent_compile_scopes_isolate_cache_keys_and_restore_target():
    default, first, second = map(cute_target.CompileTarget, ("default", "first", "second"))
    cute_target.set_compile_target(default)
    barrier = threading.Barrier(2, timeout=10)

    config = TracedConfig("shared")
    config.__cute_memo__ = {}

    @cute_cache.jit_cache(persistent=False)
    def compile_kernel(op: TracedConfig) -> str:
        op.found = cute_target.get_compile_target().device_type
        return op.found

    # Warm both builds on separate instances; only their lookups share mutable trace state.
    for target in (first, second):
        with cute_target.compile_target_scope(target):
            assert compile_kernel(TracedConfig("shared")) == target.device_type

    def scoped_compile(target: cute_target.CompileTarget) -> str:
        with cute_target.compile_target_scope(target):
            barrier.wait()
            result = compile_kernel(config)
            barrier.wait()
            return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(scoped_compile, (first, second))) == ["first", "second"]
    assert cute_target.get_compile_target() == default
    with cute_target.compile_target_scope(first):
        with (
            pytest.raises(RuntimeError, match="compile failed"),
            cute_target.compile_target_scope(second),
        ):
            assert compile_kernel(config) == "second"
            raise RuntimeError("compile failed")
        assert compile_kernel(config) == config.found == "first"
    assert compile_kernel(config) == config.found == "default"


def test_explicit_target_avoids_device_discovery(monkeypatch):
    target = cute_target.CompileTarget(
        device_type="cuda",
        capability=(10, 0),
        name="test-gpu",
        sm_count=100,
    )

    def unexpected_discovery():
        raise AssertionError("CUDA discovery was called")

    monkeypatch.setattr(cute_target, "detect_compile_target", unexpected_discovery)
    cute_target.set_compile_target(target)

    assert cute_target.get_compile_target() == target


@pytest.mark.skipif(sys.platform == "win32", reason="fork is required")
def test_compile_many_forks_parallel_workers_once_per_cold_variant(tmp_path, monkeypatch):
    """Cold variants compile in a fresh driver's workers using the caller's cache directory.

    ``set_cache_path`` sets that directory. A warm batch starts no driver.
    """
    pytest.importorskip("cutlass")
    pytest.importorskip("tvm_ffi")
    cache_directory = tmp_path / "cache"
    log_path = tmp_path / "compiles.log"
    configs = tuple(CompileConfig(str(index), float(index + 1)) for index in range(2))
    monkeypatch.setenv("WARPDRIVE_CACHE_DIR", str(tmp_path / "environment-cache"))
    monkeypatch.setattr(cute_cache, "_cache_path", None)  # restored after the test
    cute_cache.set_cache_path(cache_directory)
    monkeypatch.delenv("CUTE_DSL_NO_CACHE", raising=False)
    monkeypatch.setenv(_DRIVER_LOG_ENV, str(log_path))
    monkeypatch.setenv(_DRIVER_READY_ENV, str(tmp_path / "ready"))
    monkeypatch.setenv(_DRIVER_EXPECTED_ENV, str(len(configs)))
    monkeypatch.setattr(cute_cache, "_load_compiled", load_fake_compiled)
    compile_test_variant.cache_clear()
    target = cute_target.CompileTarget(
        device_type="cuda", capability=(10, 0), name="test-gpu", sm_count=100
    )
    calls = [(config,) for config in configs]

    compiled = compile_many(compile_test_variant, calls, workers=len(configs), target=target)

    assert [fn() for fn in compiled] == [config.variant for config in configs]
    assert cute_target.get_compile_target() == target  # unscoped explicit target is the default
    records = [line.split(",") for line in log_path.read_text().splitlines()]
    assert sorted(record[2] for record in records) == sorted(c.variant for c in configs)
    assert len({record[0] for record in records}) == len(configs)
    driver_pids = {int(record[1]) for record in records}
    assert len(driver_pids) == 1 and os.getpid() not in driver_pids
    assert {record[3] for record in records} == {"sm_100a"}
    assert not (tmp_path / "environment-cache").exists()

    def unexpected_driver(*args, **kwargs):
        raise AssertionError("a fully warm batch started a compiler process")

    compile_test_variant.cache_clear()
    monkeypatch.setattr(cute_compile.subprocess, "Popen", unexpected_driver)
    default, other = map(cute_target.CompileTarget, ("default", "other"))
    cute_target.set_compile_target(default)
    with cute_target.compile_target_scope(target):
        precompile_many(compile_test_variant, calls)
    assert cute_target.get_compile_target() == default
    with cute_target.compile_target_scope(other):
        compiled = compile_many(compile_test_variant, calls, workers=len(configs), target=target)
        assert cute_target.get_compile_target() == other
    assert cute_target.get_compile_target() == default
    assert [fn() for fn in compiled] == [config.variant for config in configs]
    assert compile_test_variant.cache_info() == cute_cache.CacheInfo(hits=2, misses=0, currsize=2)


@pytest.mark.parametrize("bulk", [compile_many, precompile_many])
def test_rejected_bulk_compile_keeps_default_target(monkeypatch, bulk):
    default = cute_target.CompileTarget("default")
    cute_target.set_compile_target(default)
    monkeypatch.setenv("CUTE_DSL_NO_CACHE", "1")
    with pytest.raises(RuntimeError, match="requires the disk cache"):
        bulk(
            compile_test_variant,
            [(CompileConfig("one", 0.0),)],
            target=cute_target.CompileTarget("B"),
        )
    assert cute_target.get_compile_target() is default


@pytest.fixture
def driver_cache(tmp_path, monkeypatch) -> cute_target.CompileTarget:
    """An empty disk cache for compiler processes to fill; the target (sm_100a) they compile
    for."""
    pytest.importorskip("cutlass")
    pytest.importorskip("tvm_ffi")
    monkeypatch.setattr(cute_cache, "_cache_path", None)  # restored after the test
    cute_cache.set_cache_path(tmp_path / "cache")
    monkeypatch.delenv("CUTE_DSL_NO_CACHE", raising=False)
    return cute_target.CompileTarget(
        device_type="cuda", capability=(10, 0), name="test-gpu", sm_count=100
    )


@pytest.mark.skipif(sys.platform == "win32", reason="fork is required")
def test_precompile_calls_mixes_compile_functions_in_one_driver(
    tmp_path, monkeypatch, driver_cache
):
    """Calls to two compile functions compile concurrently in one driver's workers; progress
    counts the cold calls up to all of them, and a warm batch reports nothing."""
    log_path = tmp_path / "compiles.log"
    monkeypatch.setenv(_DRIVER_LOG_ENV, str(log_path))
    monkeypatch.setenv(_DRIVER_READY_ENV, str(tmp_path / "ready"))
    monkeypatch.setenv(_DRIVER_EXPECTED_ENV, "2")
    config = CompileConfig("a", 1.0)
    calls = [(compile_test_variant, config), (compile_other_variant, config)]
    reports = []

    precompile_calls(calls, workers=2, target=driver_cache, progress=lambda *r: reports.append(r))

    records = [line.split(",") for line in log_path.read_text().splitlines()]
    assert sorted(record[2] for record in records) == ["a", "other-a"]
    assert len({record[1] for record in records}) == 1, "one driver"
    assert reports == sorted(reports) and reports[-1] == (2, 2)
    assert all(fn.is_cached(*args) for fn, *args in calls)
    reports.clear()
    precompile_calls(calls, target=driver_cache, progress=lambda *r: reports.append(r))
    assert reports == []


@pytest.mark.skipif(sys.platform == "win32", reason="fork is required")
def test_precompile_calls_compiles_a_single_call_for_the_requested_target(
    tmp_path, monkeypatch, driver_cache
):
    """One cold call compiles in a compiler process too, for the batch's target, not the
    caller's CuTeDSL architecture, so it lands under the target's key."""
    log_path = tmp_path / "compiles.log"
    monkeypatch.setenv(_DRIVER_LOG_ENV, str(log_path))
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_90a")
    config = CompileConfig("single", 1.0)
    precompile_calls([(compile_test_variant, config)], target=driver_cache)
    ((pid, _, variant, arch),) = (line.split(",") for line in log_path.read_text().splitlines())
    assert (variant, arch) == ("single", "sm_100a") and int(pid) != os.getpid()
    assert compile_test_variant.is_cached(config)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded.*:DeprecationWarning")
@pytest.mark.parametrize("stop", ["timeout", "interrupt", "caller killed"])
def test_a_stopped_batch_leaves_no_worker_or_lock(tmp_path, monkeypatch, driver_cache, stop):
    """A batch stopped while its worker compiles, by its timeout, Ctrl-C between progress
    reports or its caller's death, kills the worker, which frees the entry's lock."""
    hang = tmp_path / "hang"
    hang.mkdir()
    monkeypatch.setenv(_HANG_ENV, str(hang))
    config = CompileConfig("hang", 1.0)
    calls = [(compile_hanging_variant, config)]
    started = lambda: [int(path.name) for path in hang.iterdir()]

    def progress(done, total):
        wait_until(started)  # the worker holds the entry's lock
        if stop == "interrupt":
            raise KeyboardInterrupt

    if stop == "caller killed":
        caller = multiprocessing.get_context("fork").Process(
            target=precompile_calls, args=(calls,), kwargs={"target": driver_cache}
        )
        caller.start()
        wait_until(started)
        caller.kill()
        caller.join()
    else:
        error = KeyboardInterrupt if stop == "interrupt" else RuntimeError
        with pytest.raises(error, match=None if stop == "interrupt" else "timed out"):
            precompile_calls(calls, target=driver_cache, timeout=0.1, progress=progress)
    (worker,) = started()
    wait_until(lambda: not alive(worker), timeout=10)

    monkeypatch.delenv(_HANG_ENV)
    cute_target.set_compile_target(driver_cache)
    compile_hanging_variant.precompile(config)  # CacheLockError if a worker still held the lock
    assert compile_hanging_variant.is_cached(config)


CONFIGS = [CompileConfig("x", 1.0), CompileConfig("y", 1.0)]


@pytest.mark.skipif(sys.platform == "win32", reason="fork is required")
def test_a_batch_the_disk_cache_cannot_hold_raises_a_storage_error(tmp_path, driver_cache):
    """Calls that cannot lock or write their entries (the cache directory is a file) raise
    ``CacheStorageError``, which callers recover from by compiling without the cache."""
    (tmp_path / "file").touch()
    cute_cache.set_cache_path(tmp_path / "file")
    with pytest.raises(cute_cache.CacheStorageError, match="could not open cache lock"):
        precompile_calls([(compile_test_variant, c) for c in CONFIGS], target=driver_cache)


@pytest.mark.skipif(sys.platform == "win32", reason="fork is required")
def test_a_batch_that_fails_to_compile_raises_the_compiler_error(driver_cache):
    with pytest.raises(RuntimeError, match="cannot compile x") as raised:
        precompile_calls([(compile_broken_variant, c) for c in CONFIGS], target=driver_cache)
    assert not isinstance(raised.value, cute_cache.CacheStorageError)


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl.flock and fork are required")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded.*:DeprecationWarning")
def test_processes_compile_each_key_once_then_the_parent_reloads(isolated_cache):
    """Same-key processes compile once; distinct keys compile concurrently.

    The barrier requires both compilers to run at once. The parent reloads every
    variant from disk.
    """
    context = multiprocessing.get_context("fork")
    variants = ["0", "1", "0"]
    compile_count = context.Value("i", 0)
    compile_barrier = context.Barrier(len(set(variants)), timeout=10)
    start = context.Event()

    @cute_cache.jit_cache
    def compile_kernel(variant: str) -> FakeCompiled:
        with compile_count.get_lock():
            compile_count.value += 1
        compile_barrier.wait()
        return FakeCompiled(variant)

    def worker(variant: str) -> None:
        start.wait()
        assert compile_kernel(variant)() == variant

    processes = [context.Process(target=worker, args=(variant,)) for variant in variants]
    for process in processes:
        process.start()
    start.set()
    wait_for_processes(processes)

    assert compile_count.value == 2
    assert len(list(isolated_cache.rglob("*.o"))) == 2
    for variant in ("0", "1"):
        assert compile_kernel(variant)() == variant
    assert compile_count.value == 2
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=2, misses=0, currsize=2)


# A synthetic package: ``pkg.entry.compile_kernel`` reaches every module except the three excluded
# below, through each import form the source key must follow.
SYNTHETIC_PACKAGE = {
    "__init__.py": "",
    "entry.py": """
import pkg.absolute
import pkg.aliased as aliased
import pkg.sub.leaf
from . import relative
from pkg.lazymap import thing
from pkg.nested import VALUE
from pkg.reexport import NAME
from pkg.starred import *

from attn_gym.warpdrive.cache import jit_cache


@jit_cache
def compile_kernel():
    from pkg import local


def report():
    import importlib

    return importlib.import_module(".hostonly", __package__)
""",
    "absolute.py": "from pkg import cycle\n",
    "cycle.py": "from pkg import absolute\n",
    "aliased.py": "VALUE = 1\n",
    "starred.py": "VALUE = 1\n",
    "relative.py": "VALUE = 1\n",
    "local.py": "VALUE = 1\n",
    # Excluded: ``import pkg.sub.leaf`` runs this ``__init__`` only implicitly.
    "sub/__init__.py": "from pkg import unrelated\n",
    "sub/leaf.py": "from ..relative import VALUE\n",
    "nested/__init__.py": "from . import child\nfrom .. import upper\nVALUE = 1\n",
    "nested/child.py": "VALUE = 1\n",
    "upper.py": "VALUE = 1\n",
    "reexport/__init__.py": "from .impl import NAME\n",
    "reexport/impl.py": "NAME = 1\n",
    # Lazy exports name their owners only as strings (see attn_gym/linear/_lazy.py).
    "lazymap.py": """
EXPORTS = {"thing": "pkg.lazy_target"}


def __getattr__(name):
    import importlib

    if name not in EXPORTS:
        raise AttributeError(name)
    return getattr(importlib.import_module(EXPORTS[name]), name)
""",
    "lazy_target.py": "thing = 1\n",
    # Excluded: nothing imports it.
    "unrelated.py": "VALUE = 1\n",
    # Excluded: imported by relative name at call time, the opt-out for host-only code.
    "hostonly.py": "VALUE = 1\n",
}


@pytest.fixture
def synthetic_package(tmp_path, monkeypatch):
    """Import ``SYNTHETIC_PACKAGE`` as ``pkg`` and treat it as the package the key follows."""
    root = tmp_path / "pkg"
    for name, source in SYNTHETIC_PACKAGE.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(cute_key, "_PACKAGE_ROOT", root)
    yield root
    for name in [name for name in sys.modules if name == "pkg" or name.startswith("pkg.")]:
        del sys.modules[name]


@pytest.mark.parametrize("module", sorted(SYNTHETIC_PACKAGE))
def test_cache_namespace_changes_only_with_imported_sources(synthetic_package, module):
    """Editing a module the compile function transitively imports recompiles; others do not."""
    pytest.importorskip("cutlass")
    compile_kernel = importlib.import_module("pkg.entry").compile_kernel
    baseline = compile_kernel.cache_namespace()
    with (synthetic_package / module).open("a") as source:
        source.write("EDITED = 1\n")
    cute_key.source_fingerprint.cache_clear()

    changed = compile_kernel.cache_namespace() != baseline
    assert changed == (module not in {"sub/__init__.py", "unrelated.py", "hostonly.py"})


def test_callers_outside_the_package_key_on_their_package_imports(
    synthetic_package, tmp_path_factory, monkeypatch
):
    """A compile function outside the package keys on its own file and package imports."""
    pytest.importorskip("cutlass")
    caller = tmp_path_factory.mktemp("outside") / "outside_kernels.py"
    caller.write_text(
        "import pkg.sub.leaf\nfrom attn_gym.warpdrive.cache.jit import jit_cache\n\n"
        "@jit_cache\ndef compile_kernel():\n    pass\n"
    )
    spec = importlib.util.spec_from_file_location("outside_kernels", caller)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "outside_kernels", module)  # Removed at teardown.
    spec.loader.exec_module(module)
    compile_kernel = module.compile_kernel

    def namespace_after_editing(path: Path) -> str:
        with path.open("a") as source:
            source.write("EDITED = 1\n")
        cute_key.source_fingerprint.cache_clear()
        return compile_kernel.cache_namespace()

    baseline = compile_kernel.cache_namespace()
    assert namespace_after_editing(synthetic_package / "unrelated.py") == baseline
    package_edited = namespace_after_editing(synthetic_package / "relative.py")
    assert package_edited != baseline
    assert namespace_after_editing(caller) != package_edited


def test_module_source_digest_follows_the_module_closure(synthetic_package):
    """A class's module digest changes with the modules it imports, as a namespace does."""
    importlib.import_module("pkg.entry")
    cute_key.module_source_digest.cache_clear()
    baseline = cute_key.module_source_digest("pkg.entry")
    assert cute_key.module_source_digest("builtins") is None

    def digest_after_editing(module: str) -> str:
        with (synthetic_package / module).open("a") as source:
            source.write("EDITED = 1\n")
        cute_key.module_source_digest.cache_clear()
        return cute_key.module_source_digest("pkg.entry")

    assert digest_after_editing("unrelated.py") == baseline
    assert digest_after_editing("relative.py") != baseline


def test_kernel_constants_are_part_of_the_kernel_key():
    """Inlined constants outside the compile module must still affect the key."""
    package = cute_key._PACKAGE_ROOT
    closure = cute_key.module_closure(package / "linear/_delta_rule/warpdrive/kda_prefill.py")
    assert package / "linear/_delta_rule/warpdrive/kda_common.py" in closure
    assert package / "linear/short_conv/cute.py" not in closure
