"""Explicit compile-and-benchmark tuning for cached CuTeDSL variants."""

from __future__ import annotations

import functools
import inspect
import math
import statistics
from collections.abc import Callable, Iterable
from typing import Any, Protocol, TypeVar

from .compile import (
    DEFAULT_COMPILE_TIMEOUT_SECONDS,
    _materialize_calls,
    compile_many,
)
from .target import CompileTarget, set_compile_target

ConfigT = TypeVar("ConfigT")
CompiledT = TypeVar("CompiledT")


class TunableKernel(Protocol[ConfigT, CompiledT]):
    """Convention consumed by :func:`run_tunable`."""

    default_config: ConfigT

    def configs(self, *runtime_args: Any) -> Iterable[ConfigT]: ...

    def compile(self, *args: Any) -> CompiledT: ...

    def compile_call(
        self,
        config: ConfigT,
        *runtime_args: Any,
    ) -> tuple[Any, ...]: ...

    def launch(
        self,
        compiled: CompiledT,
        config: ConfigT,
        *runtime_args: Any,
    ) -> Any: ...


def benchmark_gpu(
    fn: Callable[[], Any],
    *,
    estimation_iters: int = 5,
    memory_warmup_iters: int = 100,
    benchmark_iters: int = 100,
    max_benchmark_duration: int = 25,
) -> float:
    """Return median GPU latency in milliseconds using Inductor's benchmarker.

    Inductor performs CUDA-event timing and L2-cache warmup/flush work. Keyword
    support is detected at runtime so this helper remains usable across the
    Torch versions supported by Attention Gym.
    """
    from torch._inductor.runtime.benchmarking import benchmarker

    supported = inspect.signature(benchmarker.benchmark_gpu).parameters
    options = {
        "estimation_iters": estimation_iters,
        "memory_warmup_iters": memory_warmup_iters,
        "benchmark_iters": benchmark_iters,
        "max_benchmark_duration": max_benchmark_duration,
        "return_mode": "all",
        "is_vetted_benchmarking": True,
    }
    timings = benchmarker.benchmark_gpu(
        fn,
        **{name: value for name, value in options.items() if name in supported},
    )
    if isinstance(timings, list):
        if not timings:
            raise RuntimeError("Inductor benchmarker returned no GPU timings")
        return float(statistics.median(timings))
    return float(timings)


def tune(
    configs: Iterable[ConfigT],
    compile_fn: Callable[..., CompiledT],
    launch: Callable[[CompiledT, ConfigT], Any],
    *,
    benchmark: Callable[[Callable[[], Any]], float] | None = None,
    compile_call: Callable[[ConfigT], tuple[Any, ...]] | None = None,
    parallel_compile: bool = True,
    workers: int | None = None,
    target: CompileTarget | None = None,
    timeout: float | None = DEFAULT_COMPILE_TIMEOUT_SECONDS,
) -> ConfigT:
    """Compile candidates, benchmark GPU launches sequentially, and return the fastest.

    The common case passes each config as the compile function's sole argument
    and uses the default Inductor GPU benchmarker::

        best = tune(
            configs,
            compile_kernel,
            lambda kernel, config: kernel(*runtime_arguments),
        )

    Cold variants can be populated concurrently. Module-scope ``NamedTuple``
    and dataclass configs work directly. A warm invocation skips the compiler
    process, loads every requested artifact, and benchmarks again.
    ``benchmark`` may replace the default timing policy; it receives a zero-arg
    callable for one candidate launch. Kernels with destructive or stateful
    inputs should use that hook to restore state between measurements.

    Use ``compile_call`` when the compile function needs additional static
    arguments::

        best = tune(
            configs,
            compile_kernel,
            launch,
            compile_call=lambda config: (head_dim, config),
        )
    """
    candidates = list(configs)
    if not candidates:
        raise ValueError("tune() requires at least one config")
    if workers is not None and workers < 1:
        raise ValueError(f"workers must be positive, got {workers}")
    if not callable(getattr(compile_fn, "precompile", None)):
        raise TypeError("tune() requires a compile function decorated with jit_cache")

    if compile_call is None:
        calls = [(config,) for config in candidates]
    else:
        calls = _materialize_calls(compile_call(config) for config in candidates)

    if parallel_compile and len(calls) > 1:
        compiled_candidates = compile_many(
            compile_fn,
            calls,
            workers=workers,
            target=target,
            timeout=timeout,
        )
    else:
        if target is not None:
            set_compile_target(target)
        compiled_candidates = [compile_fn(*args) for args in calls]

    measure = benchmark_gpu if benchmark is None else benchmark
    timings = []
    for compiled, config in zip(compiled_candidates, candidates):
        timing = float(measure(functools.partial(launch, compiled, config)))
        if math.isnan(timing):
            raise ValueError(f"benchmark returned NaN for config {config!r}")
        timings.append(timing)
    return candidates[min(range(len(candidates)), key=timings.__getitem__)]


def run_tunable(
    kernel: TunableKernel[ConfigT, CompiledT],
    *runtime_args: Any,
    config: ConfigT | None = None,
    autotune: bool = False,
    configs: Iterable[ConfigT] | None = None,
    benchmark: Callable[[Callable[[], Any]], float] | None = None,
    parallel_compile: bool = True,
    workers: int | None = None,
    target: CompileTarget | None = None,
    timeout: float | None = DEFAULT_COMPILE_TIMEOUT_SECONDS,
) -> tuple[Any, ConfigT]:
    """Select, compile, and launch an op following the tunable-kernel convention.

    ``kernel`` supplies ``default_config``, an input-aware
    ``configs(*runtime_args)`` method, a cached ``compile`` function, and
    ``compile_call``/``launch`` adapters. This keeps candidate generation and
    fake-tensor ABI construction in the kernel implementation while public
    PyTorch entrypoints only choose between a default, explicit config, or
    autotuning. Passing ``configs=`` overrides the kernel's candidate method.

    The return value is ``(launch_result, selected_config)``. Autotuning still
    compiles candidates in parallel and benchmarks launches sequentially, then
    performs one final launch with the winner.
    """
    if autotune and config is not None:
        raise ValueError("pass either config= or autotune=True, not both")
    if not autotune and configs is not None:
        raise ValueError("configs= requires autotune=True")

    compile_fn = kernel.compile
    if not callable(getattr(compile_fn, "precompile", None)):
        raise TypeError("run_tunable() requires kernel.compile decorated with jit_cache")
    if autotune:
        candidates = kernel.configs(*runtime_args) if configs is None else configs

        def launch(compiled: CompiledT, candidate: ConfigT) -> Any:
            return kernel.launch(compiled, candidate, *runtime_args)

        selected = tune(
            candidates,
            compile_fn,
            launch,
            benchmark=benchmark,
            compile_call=lambda candidate: kernel.compile_call(candidate, *runtime_args),
            parallel_compile=parallel_compile,
            workers=workers,
            target=target,
            timeout=timeout,
        )
    else:
        selected = kernel.default_config if config is None else config
        if target is not None:
            set_compile_target(target)

    call_args = _materialize_calls((kernel.compile_call(selected, *runtime_args),))[0]
    compiled = compile_fn(*call_args)
    return kernel.launch(compiled, selected, *runtime_args), selected


__all__ = ["TunableKernel", "benchmark_gpu", "run_tunable", "tune"]
