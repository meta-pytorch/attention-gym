from __future__ import annotations

import importlib
import multiprocessing
import os
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import pytest

from attn_gym._backends.cute import cache as cute_cache
from attn_gym._backends.cute import compile as cute_compile
from attn_gym._backends.cute import target as cute_target
from attn_gym._backends.cute.tune import benchmark_gpu, run_tunable, tune
from attn_gym._backends.cute.utils import compile_tvm_ffi
from attn_gym.utils import ceildiv

_DRIVER_LOG_ENV = "ATTN_GYM_TEST_CUTE_DRIVER_LOG"
_DRIVER_READY_ENV = "ATTN_GYM_TEST_CUTE_DRIVER_READY"
_DRIVER_EXPECTED_ENV = "ATTN_GYM_TEST_CUTE_DRIVER_EXPECTED"


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


def measure_return_value(fn) -> float:
    """Use a launch's return value as a deterministic unit-test timing."""
    return float(fn())


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


def wait_for_processes(processes, timeout: float = 15.0) -> None:
    """Join child processes and fail without leaving a process behind."""
    deadline = time.monotonic() + timeout
    try:
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        failures = [(process.pid, process.exitcode) for process in processes if process.exitcode]
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
    monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(cache_directory))
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


def test_cache_directory_can_be_overridden(tmp_path, monkeypatch):
    configured_path = tmp_path / "custom-cache"
    monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(configured_path))

    assert cute_cache.get_cache_path() == configured_path


def test_ceildiv():
    assert ceildiv(0, 3) == 0
    assert ceildiv(1, 3) == 1
    assert ceildiv(7, 3) == 3


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
        "option": cute.EnableTVMFFI,
        "entrypoint": entrypoint,
        "args": (fake_tensor, fake_stream),
        "kwargs": {},
        "name_prefix": "stable_kernel_name",
    }


def test_benchmark_gpu_uses_inductor_benchmarker(monkeypatch):
    import torch._inductor.runtime.benchmarking as inductor_benchmarking

    launches = []
    observed = {}

    class FakeBenchmarker:
        def benchmark_gpu(
            self,
            fn,
            estimation_iters,
            memory_warmup_iters,
            benchmark_iters,
            max_benchmark_duration,
            return_mode,
            is_vetted_benchmarking,
        ):
            fn()
            observed.update(
                estimation_iters=estimation_iters,
                memory_warmup_iters=memory_warmup_iters,
                benchmark_iters=benchmark_iters,
                max_benchmark_duration=max_benchmark_duration,
                return_mode=return_mode,
                is_vetted_benchmarking=is_vetted_benchmarking,
            )
            return [0.003, 0.001, 0.002]

    monkeypatch.setattr(inductor_benchmarking, "benchmarker", FakeBenchmarker())

    timing = benchmark_gpu(
        lambda: launches.append("launch"),
        estimation_iters=2,
        memory_warmup_iters=3,
        benchmark_iters=4,
        max_benchmark_duration=5,
    )

    assert timing == pytest.approx(0.002)
    assert launches == ["launch"]
    assert observed == {
        "estimation_iters": 2,
        "memory_warmup_iters": 3,
        "benchmark_iters": 4,
        "max_benchmark_duration": 5,
        "return_mode": "all",
        "is_vetted_benchmarking": True,
    }


def test_tune_sequential_option_caches_every_named_tuple_config(isolated_cache):
    compile_count = 0
    configs = (
        CompileConfig("slow", 3.0),
        CompileConfig("fast", 1.0),
        CompileConfig("medium", 2.0),
    )

    @cute_cache.jit_cache
    def compile_kernel(config: CompileConfig) -> FakeCompiled:
        nonlocal compile_count
        compile_count += 1
        return FakeCompiled(config.variant)

    benchmark_order = []

    def launch(compiled: FakeCompiled, config: CompileConfig) -> float:
        assert compiled() == config.variant
        benchmark_order.append(config.variant)
        return config.timing

    best = tune(
        configs,
        compile_kernel,
        launch,
        benchmark=measure_return_value,
        parallel_compile=False,
    )

    assert best is configs[1]
    assert benchmark_order == ["slow", "fast", "medium"]
    assert compile_count == len(configs)
    assert len(list(isolated_cache.rglob("*.o"))) == len(configs)

    compile_kernel.cache_clear()
    benchmark_order.clear()
    assert (
        tune(
            configs,
            compile_kernel,
            launch,
            benchmark=measure_return_value,
            parallel_compile=False,
        )
        is configs[1]
    )
    assert benchmark_order == ["slow", "fast", "medium"]
    assert compile_count == len(configs)
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(
        hits=len(configs),
        misses=0,
        currsize=len(configs),
    )


def test_run_tunable_uses_kernel_convention(isolated_cache):
    candidates = (
        CompileConfig("default", 3.0),
        CompileConfig("fast", 1.0),
        CompileConfig("medium", 2.0),
    )
    compile_count = 0
    config_requests = []
    default_requests = []
    launches = []

    class ToyKernel:
        @staticmethod
        def default_config(prefix: str, _launch_log, *, target):
            default_requests.append((prefix, target))
            return candidates[0]

        @staticmethod
        def tuning_key(prefix: str, _launch_log, *, target):
            assert prefix == "toy" and isinstance(target, cute_target.CompileTarget)
            return ()

        @staticmethod
        def configs(prefix: str, _launch_log):
            config_requests.append(prefix)
            return candidates

        @staticmethod
        @cute_cache.jit_cache
        def compile(prefix: str, config: CompileConfig) -> FakeCompiled:
            nonlocal compile_count
            compile_count += 1
            return FakeCompiled(f"{prefix}:{config.variant}")

        @staticmethod
        def compile_call(config: CompileConfig, prefix: str, _launches):
            return prefix, config

        @staticmethod
        def launch(
            compiled: FakeCompiled,
            config: CompileConfig,
            prefix: str,
            launch_log,
        ) -> float:
            assert compiled() == f"{prefix}:{config.variant}"
            launch_log.append(config.variant)
            return config.timing

    result, best = run_tunable(
        ToyKernel,
        "toy",
        launches,
        autotune=True,
        benchmark=measure_return_value,
        parallel_compile=False,
    )

    assert best is candidates[1]
    assert result == best.timing
    assert launches == [config.variant for config in candidates] + [best.variant]
    assert config_requests == ["toy"]
    assert default_requests == []
    assert compile_count == len(candidates)
    assert len(list(isolated_cache.rglob("*.o"))) == len(candidates)

    launches.clear()
    result, selected = run_tunable(ToyKernel, "toy", launches)
    assert selected is candidates[0]
    assert result == selected.timing
    assert launches == [selected.variant]
    assert config_requests == ["toy"]
    assert len(default_requests) == 1
    assert default_requests[0][0] == "toy"
    assert compile_count == len(candidates)

    launches.clear()
    override = (candidates[2],)
    result, selected = run_tunable(
        ToyKernel,
        "toy",
        launches,
        autotune=True,
        configs=override,
        benchmark=measure_return_value,
        parallel_compile=False,
    )
    assert selected is override[0]
    assert result == selected.timing
    assert launches == [selected.variant, selected.variant]
    assert config_requests == ["toy"]

    with pytest.raises(ValueError, match="either config= or autotune=True"):
        run_tunable(
            ToyKernel,
            "toy",
            launches,
            config=candidates[0],
            autotune=True,
        )
    with pytest.raises(ValueError, match="configs= requires autotune=True"):
        run_tunable(ToyKernel, "toy", launches, configs=candidates)

    class InvalidCompileCall(ToyKernel):
        @staticmethod
        def compile_call(_config, prefix, _launch_log):
            return prefix

    with pytest.raises(TypeError, match="tuple of positional static arguments"):
        run_tunable(InvalidCompileCall, "toy", launches)


def test_run_tunable_custom_benchmark_bypasses_cached_winner(isolated_cache, monkeypatch):
    """Always execute an explicitly requested timing policy."""
    tune_module = importlib.import_module("attn_gym._backends.cute.tune")
    candidates = (CompileConfig("slow", 3.0), CompileConfig("fast", 1.0))

    class ToyKernel:
        @staticmethod
        def default_config(*, target):
            assert target.device_type == "test"
            return candidates[0]

        @staticmethod
        def tuning_key(*, target):
            assert target.device_type == "test"
            return ()

        @staticmethod
        def configs():
            return candidates

        @staticmethod
        @cute_cache.jit_cache
        def compile(config: CompileConfig) -> FakeCompiled:
            return FakeCompiled(config.variant)

        @staticmethod
        def compile_call(config: CompileConfig):
            return (config,)

        @staticmethod
        def launch(compiled: FakeCompiled, config: CompileConfig) -> float:
            assert compiled() == config.variant
            return config.timing

    target = cute_target.CompileTarget("test")
    monkeypatch.setattr(tune_module, "benchmark_gpu", measure_return_value)
    _, cached = run_tunable(ToyKernel, autotune=True, parallel_compile=False, target=target)
    assert cached is candidates[1]

    benchmarked = []

    def reverse_policy(fn) -> float:
        timing = float(fn())
        benchmarked.append(timing)
        return -timing

    _, selected = run_tunable(
        ToyKernel,
        autotune=True,
        benchmark=reverse_policy,
        parallel_compile=False,
        target=target,
    )
    assert selected is candidates[0]
    assert benchmarked == [config.timing for config in candidates]


def test_run_tunable_keys_winners_by_runtime_workload(isolated_cache, monkeypatch):
    """Separate winner decisions without recompiling shape-polymorphic binaries."""
    tune_module = importlib.import_module("attn_gym._backends.cute.tune")
    candidates = (CompileConfig("small", 0.0), CompileConfig("large", 0.0))
    compile_count = 0
    measurements = []

    class WorkloadAwareKernel:
        @staticmethod
        def default_config(_work_units: int, *, target):
            assert target.device_type == "test"
            return candidates[0]

        @staticmethod
        def tuning_key(work_units: int, *, target):
            assert target.device_type == "test"
            return (work_units,)

        @staticmethod
        def configs(_work_units: int):
            return candidates

        @staticmethod
        @cute_cache.jit_cache
        def compile(config: CompileConfig) -> FakeCompiled:
            nonlocal compile_count
            compile_count += 1
            return FakeCompiled(config.variant)

        @staticmethod
        def compile_call(config: CompileConfig, _work_units: int):
            return (config,)

        @staticmethod
        def launch(compiled: FakeCompiled, config: CompileConfig, work_units: int) -> float:
            assert compiled() == config.variant
            expected = "small" if work_units == 1 else "large"
            return 1.0 if config.variant == expected else 2.0

    def measure(fn) -> float:
        timing = float(fn())
        measurements.append(timing)
        return timing

    target = cute_target.CompileTarget("test")
    monkeypatch.setattr(tune_module, "benchmark_gpu", measure)
    monkeypatch.setattr(tune_module, "_WINNERS", {})
    monkeypatch.setattr(tune_module, "_WINNERS_FAST", {})

    _, small = run_tunable(
        WorkloadAwareKernel, 1, autotune=True, parallel_compile=False, target=target
    )
    _, large = run_tunable(
        WorkloadAwareKernel, 2, autotune=True, parallel_compile=False, target=target
    )
    assert small is candidates[0]
    assert large is candidates[1]
    assert measurements == [1.0, 2.0, 2.0, 1.0]
    assert compile_count == len(candidates)
    assert len(list(isolated_cache.rglob("*.o"))) == len(candidates)

    run_tunable(WorkloadAwareKernel, 1, autotune=True, parallel_compile=False, target=target)
    assert measurements == [1.0, 2.0, 2.0, 1.0]

    monkeypatch.setattr(tune_module, "_WINNERS", {})
    monkeypatch.setattr(tune_module, "_WINNERS_FAST", {})
    _, persisted = run_tunable(
        WorkloadAwareKernel, 2, autotune=True, parallel_compile=False, target=target
    )
    assert persisted == candidates[1]
    assert measurements == [1.0, 2.0, 2.0, 1.0]
    assert compile_count == len(candidates)


def test_run_tunable_sets_target_before_candidate_generation(isolated_cache):
    """Generate target-aware candidates from the explicitly requested device."""
    config = CompileConfig("target-aware", 1.0)
    target = cute_target.CompileTarget(device_type="cuda", capability=(10, 3))
    observed_targets = []

    class TargetAwareKernel:
        @staticmethod
        def default_config(*, target):
            observed_targets.append(target)
            return config

        @staticmethod
        def tuning_key(*, target):
            observed_targets.append(target)
            return ()

        @staticmethod
        def configs():
            observed_targets.append(cute_target.get_compile_target())
            return (config,)

        @staticmethod
        @cute_cache.jit_cache
        def compile(candidate: CompileConfig) -> FakeCompiled:
            return FakeCompiled(candidate.variant)

        @staticmethod
        def compile_call(candidate: CompileConfig):
            return (candidate,)

        @staticmethod
        def launch(compiled: FakeCompiled, candidate: CompileConfig) -> float:
            assert compiled() == candidate.variant
            return candidate.timing

    result, selected = run_tunable(
        TargetAwareKernel,
        autotune=True,
        benchmark=measure_return_value,
        parallel_compile=False,
        target=target,
    )

    assert selected is config
    assert result == config.timing
    assert observed_targets == [target]


def test_runtime_cache_skips_persistent_key_for_immutable_arguments(isolated_cache, monkeypatch):
    make_key_calls = 0
    original_make_key = cute_cache._make_key

    def counting_make_key(*args, **kwargs):
        nonlocal make_key_calls
        make_key_calls += 1
        return original_make_key(*args, **kwargs)

    monkeypatch.setattr(cute_cache, "_make_key", counting_make_key)

    @cute_cache.jit_cache
    def compile_kernel(config: CompileConfig) -> FakeCompiled:
        return FakeCompiled(config.variant)

    assert compile_kernel(CompileConfig("shared", 1.0))() == "shared"
    assert compile_kernel(CompileConfig("shared", 1.0))() == "shared"
    assert make_key_calls == 1
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=1, misses=1, currsize=1)


@pytest.mark.parametrize(
    ("first", "second"),
    [(True, 1), (1, 1.0), (0.0, -0.0)],
)
def test_runtime_cache_preserves_scalar_representation(isolated_cache, first, second):
    def describe(value) -> str:
        if isinstance(value, float):
            return f"float:{value.hex()}"
        return f"{type(value).__name__}:{value}"

    @cute_cache.jit_cache
    def compile_kernel(value) -> FakeCompiled:
        return FakeCompiled(describe(value))

    assert compile_kernel(first)() == describe(first)
    assert compile_kernel(second)() == describe(second)
    assert compile_kernel(first)() == describe(first)
    assert compile_kernel(second)() == describe(second)
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=2, misses=2, currsize=2)


def test_explicit_cache_key_is_shared_by_memory_and_disk(isolated_cache):
    compile_count = 0

    @cute_cache.jit_cache(cache_key=lambda _value: "shared")
    def compile_kernel(value: str) -> FakeCompiled:
        nonlocal compile_count
        compile_count += 1
        return FakeCompiled(value)

    assert compile_kernel("first")() == "first"
    assert compile_kernel("second")() == "first"
    compile_kernel.cache_clear()
    assert compile_kernel("third")() == "first"
    assert compile_count == 1
    assert len(list(isolated_cache.rglob("*.o"))) == 1


def test_explicit_cache_key_is_shared_by_precompile_and_lookup(isolated_cache):
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
    assert compile_count == 1
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(hits=1, misses=0, currsize=1)


def test_explicit_cache_key_must_be_hashable():
    @cute_cache.jit_cache(cache_key=lambda _value: [])
    def compile_kernel(value: str) -> FakeCompiled:
        return FakeCompiled(value)

    with pytest.raises(TypeError, match="cache_key must return a hashable value"):
        compile_kernel("value")


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
    same_nan = struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0]
    same_nan_copy = struct.unpack("!d", bytes.fromhex("7ff8000000000001"))[0]
    other_nan = struct.unpack("!d", bytes.fromhex("7ff8000000000002"))[0]
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


def test_persistent_cache_can_be_disabled_with_an_option(tmp_path, monkeypatch):
    cache_directory = tmp_path / "cache"
    monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(cache_directory))
    monkeypatch.setattr(
        cute_cache,
        "get_compile_target",
        lambda: cute_target.CompileTarget("test"),
    )
    monkeypatch.setattr(
        cute_cache,
        "_source_fingerprint",
        lambda _fn, _extra_sources=(): "test-source",
    )
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
    assert not list(cache_directory.rglob("*.o"))


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
    ),
)
def test_compile_target_effective_capability(configured_arch, physical, expected):
    """Prefer an explicit code-generation architecture over the physical GPU."""
    target = cute_target.CompileTarget(
        device_type="cuda",
        configured_arch=configured_arch,
        capability=physical,
    )
    assert target.effective_capability == expected


def test_compile_target_rejects_invalid_configured_architecture():
    target = cute_target.CompileTarget(device_type="cuda", configured_arch="native")
    with pytest.raises(ValueError, match="configured CUDA architecture"):
        _ = target.effective_capability


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
def test_fresh_compile_driver_forks_parallel_workers(tmp_path, monkeypatch):
    pytest.importorskip("cutlass")
    pytest.importorskip("tvm_ffi")
    cache_directory = tmp_path / "cache"
    log_path = tmp_path / "compiles.log"
    ready_directory = tmp_path / "ready"
    configs = tuple(CompileConfig(str(index), float(index + 1)) for index in range(2))
    shared = CompileConfig("shared", 0.5)
    monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(cache_directory))
    monkeypatch.delenv("CUTE_DSL_NO_CACHE", raising=False)
    monkeypatch.setenv(_DRIVER_LOG_ENV, str(log_path))
    monkeypatch.setenv(_DRIVER_READY_ENV, str(ready_directory))
    monkeypatch.setenv(_DRIVER_EXPECTED_ENV, str(len(configs)))
    monkeypatch.setattr(cute_cache, "_load_compiled", load_fake_compiled)
    compile_test_variant.cache_clear()
    target = cute_target.CompileTarget(
        device_type="cuda",
        capability=(10, 0),
        name="test-gpu",
        sm_count=100,
    )
    benchmark_order = []

    def launch(compiled: FakeCompiled, config: CompileConfig) -> float:
        assert compiled() == config.variant
        benchmark_order.append(config.variant)
        return config.timing

    best = tune(
        configs + (shared,) * 2,
        compile_test_variant,
        launch,
        benchmark=measure_return_value,
        workers=len(configs),
        target=target,
    )

    assert best is shared
    assert benchmark_order == [config.variant for config in configs] + ["shared"] * 2
    records = [line.split(",") for line in log_path.read_text().splitlines()]
    variant_records = [record for record in records if record[2] != "shared"]
    worker_pids = {int(record[0]) for record in variant_records}
    driver_pids = {int(record[1]) for record in records}
    assert {record[2] for record in variant_records} == {config.variant for config in configs}
    assert sum(record[2] == "shared" for record in records) == 1
    assert len(worker_pids) == len(configs)
    assert len(driver_pids) == 1
    assert os.getpid() not in driver_pids
    assert {record[3] for record in records} == {"sm_100a"}
    assert len(list(cache_directory.rglob("*.o"))) == len(configs) + 1

    def unexpected_driver(*args, **kwargs):
        raise AssertionError("a fully warm tune started a compiler process")

    compile_test_variant.cache_clear()
    benchmark_order.clear()
    monkeypatch.setattr(cute_compile.subprocess, "run", unexpected_driver)
    assert (
        tune(
            configs + (shared,),
            compile_test_variant,
            launch,
            benchmark=measure_return_value,
            workers=len(configs),
            target=target,
        )
        is shared
    )
    assert benchmark_order == [config.variant for config in configs] + ["shared"]


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl.flock and fork are required")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded.*:DeprecationWarning")
def test_same_key_compiles_once_across_processes(isolated_cache):
    context = multiprocessing.get_context("fork")
    process_count = 2
    compile_count = context.Value("i", 0)
    start = context.Event()

    @cute_cache.jit_cache
    def compile_kernel(key: str) -> FakeCompiled:
        with compile_count.get_lock():
            compile_count.value += 1
        time.sleep(0.2)
        return FakeCompiled(key)

    def worker() -> None:
        start.wait()
        assert compile_kernel("shared")() == "shared"

    processes = [context.Process(target=worker) for _ in range(process_count)]
    for process in processes:
        process.start()
    start.set()
    wait_for_processes(processes)

    assert compile_count.value == 1
    assert len(list(isolated_cache.rglob("*.o"))) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="fcntl.flock and fork are required")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded.*:DeprecationWarning")
@pytest.mark.parametrize("process_count", [1, 2])
def test_parallel_population_then_sequential_reload(isolated_cache, process_count):
    """Workers compile distinct variants before the parent reloads them sequentially."""
    context = multiprocessing.get_context("fork")
    compile_count = context.Value("i", 0)
    compile_barrier = context.Barrier(process_count, timeout=10)

    @cute_cache.jit_cache
    def compile_kernel(variant: str) -> FakeCompiled:
        with compile_count.get_lock():
            compile_count.value += 1
        compile_barrier.wait()
        return FakeCompiled(variant)

    def run_wave() -> None:
        start = context.Event()

        def worker(variant: int) -> None:
            start.wait()
            expected = str(variant)
            assert compile_kernel(expected)() == expected

        processes = [
            context.Process(target=worker, args=(variant,)) for variant in range(process_count)
        ]
        for process in processes:
            process.start()
        start.set()
        wait_for_processes(processes)

    run_wave()
    assert compile_count.value == process_count
    assert len(list(isolated_cache.rglob("*.o"))) == process_count

    compile_kernel.cache_clear()
    for variant in range(process_count):
        expected = str(variant)
        assert compile_kernel(expected)() == expected

    assert compile_count.value == process_count
    assert compile_kernel.cache_info() == cute_cache.CacheInfo(
        hits=process_count,
        misses=0,
        currsize=process_count,
    )


def test_run_tunable_winner_cache_round_trip(tmp_path, monkeypatch):
    """Winners persist across processes-worth of state and reject stale candidates."""
    tune_module = importlib.import_module("attn_gym._backends.cute.tune")

    monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tune_module, "_WINNERS", {})

    key = "0" * 64
    tune_module._store_winner(key, {"grid": 4})
    assert tune_module._load_winner(key) == {"grid": 4}

    # A fresh process sees only the disk copy.
    monkeypatch.setattr(tune_module, "_WINNERS", {})
    assert tune_module._load_winner(key) == {"grid": 4}

    # Corrupt and truncated payloads degrade to re-tuning rather than failing.
    for payload in (b"not a pickle", b"", b"\x80\x04"):
        monkeypatch.setattr(tune_module, "_WINNERS", {})
        tune_module._winner_path(key).write_bytes(payload)
        assert tune_module._load_winner(key) is None


def test_winner_cache_honors_cutedsl_no_cache(tmp_path, monkeypatch):
    """Keep local memoization while disabling winner-file reads and writes."""
    tune_module = importlib.import_module("attn_gym._backends.cute.tune")
    monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("CUTE_DSL_NO_CACHE", raising=False)
    monkeypatch.setattr(tune_module, "_WINNERS", {})

    key = "1" * 64
    tune_module._store_winner(key, "disk-winner")
    path = tune_module._winner_path(key)
    disk_payload = path.read_bytes()

    monkeypatch.setenv("CUTE_DSL_NO_CACHE", "1")
    monkeypatch.setattr(tune_module, "_WINNERS", {})
    assert tune_module._load_winner(key) is None

    tune_module._store_winner(key, "memory-winner")
    assert tune_module._load_winner(key) == "memory-winner"
    assert path.read_bytes() == disk_payload


def test_winner_key_tracks_candidates_and_namespace():
    """Changing the candidate set or kernel namespace re-tunes."""
    tune_module = importlib.import_module("attn_gym._backends.cute.tune")

    class Kernel:
        @staticmethod
        def compile_call(config, capacity):
            return (config, capacity)

    class _FakeCompile:
        @staticmethod
        def cache_namespace():
            return _FakeCompile.namespace

        namespace = "namespace-a"

    Kernel.compile = _FakeCompile
    base = tune_module._winner_key(Kernel, [1, 2], (64,), ())
    assert tune_module._winner_key(Kernel, [1, 2], (64,), ()) == base
    assert tune_module._winner_key(Kernel, [1, 2, 3], (64,), ()) != base
    assert tune_module._winner_key(Kernel, [1, 2], (128,), ()) != base
    assert tune_module._winner_key(Kernel, [1, 2], (64,), ("other",)) != base
    _FakeCompile.namespace = "namespace-b"
    assert tune_module._winner_key(Kernel, [1, 2], (64,), ()) != base
