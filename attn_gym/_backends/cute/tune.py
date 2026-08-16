"""Explicit compile-and-benchmark tuning for cached CuTeDSL variants."""

from __future__ import annotations

import functools
import hashlib
import inspect
import math
import os
import pickle
import statistics
import threading
from collections.abc import Callable, Iterable
from typing import Any, Protocol, TypeVar

from .cache import cache_enabled
from .compile import (
    DEFAULT_COMPILE_TIMEOUT_SECONDS,
    _materialize_calls,
    compile_many,
)
from .target import CompileTarget, get_compile_target, set_compile_target

ConfigT = TypeVar("ConfigT")
_WINNERS_LOCK = threading.Lock()
_WINNERS: dict[str, Any] = {}
# Hot-path memo in front of the hashed disk key: compile calls are small tuples
# of static scalars, so they key a plain dict directly. Kernel sources cannot
# change within a process, so the source namespace only guards the disk entry.
_WINNERS_FAST: dict[Any, Any] = {}
CompiledT = TypeVar("CompiledT")


class TunableKernel(Protocol[ConfigT, CompiledT]):
    """Convention consumed by :func:`run_tunable`."""

    def default_config(
        self,
        *runtime_args: Any,
        target: CompileTarget,
    ) -> ConfigT: ...

    def tuning_key(
        self,
        *runtime_args: Any,
        target: CompileTarget,
    ) -> tuple[Any, ...]: ...

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
    Candidates are benchmarked in iteration order. ``benchmark`` may replace
    the default timing policy; it receives a zero-arg callable for one candidate
    launch. Destructive or stateful kernels must restore state in that callback.
    Since this function only returns a config, callers can restore state again
    before launching the winner; do not use :func:`run_tunable` for such kernels
    because its winner launch is unconditional.

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


def _winner_key(
    kernel: Any,
    candidates: list[Any],
    runtime_args: tuple[Any, ...],
    tuning_key: tuple[Any, ...],
) -> str:
    """Identify one tuning decision independently from codegen specialization."""
    from ._key import _canonicalize
    from .target import get_compile_target

    name = getattr(kernel, "__qualname__", type(kernel).__qualname__)
    payload = (
        # Source and target namespace: editing the kernel or moving to another
        # GPU model must re-tune rather than reuse a stale winner.
        kernel.compile.cache_namespace(),
        _canonicalize(get_compile_target()),
        getattr(kernel, "__module__", type(kernel).__module__),
        name,
        tuple(
            _canonicalize(kernel.compile_call(candidate, *runtime_args))
            for candidate in candidates
        ),
        _canonicalize(tuning_key),
    )
    return hashlib.sha256(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)).hexdigest()


def _winner_path(key: str):
    from .cache import get_cache_path

    return get_cache_path() / "winners" / f"{key}.pickle"


def _load_winner(key: str) -> Any | None:
    """Return a previously selected config for this tuning decision, if any."""
    with _WINNERS_LOCK:
        if key in _WINNERS:
            return _WINNERS[key]
    if not cache_enabled():
        return None
    path = _winner_path(key)
    try:
        winner = pickle.loads(path.read_bytes())
    except (EOFError, OSError, pickle.PickleError):
        return None
    with _WINNERS_LOCK:
        _WINNERS[key] = winner
    return winner


def _store_winner(key: str, winner: Any) -> None:
    """Persist a selected config in memory and on disk (best effort)."""
    with _WINNERS_LOCK:
        _WINNERS[key] = winner
    if not cache_enabled():
        return
    path = _winner_path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(pickle.dumps(winner, protocol=pickle.HIGHEST_PROTOCOL))
        os.replace(temporary, path)
    except OSError:
        pass


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

    ``kernel`` supplies input-aware ``default_config(*runtime_args, target=...)``,
    ``tuning_key(*runtime_args, target=...)``, and ``configs(*runtime_args)``
    methods, a cached ``compile`` function, and ``compile_call``/``launch``
    adapters. This keeps candidate generation and
    fake-tensor ABI construction in the kernel implementation while public
    PyTorch entrypoints only choose between a default, explicit config, or
    autotuning. Passing ``configs=`` overrides the kernel's candidate method.

    The return value is ``(launch_result, selected_config)``. Autotuning
    compiles candidates in parallel, benchmarks launches sequentially, then
    performs one final launch with the winner. Winners are cached in memory and
    under ``<cache>/winners`` keyed by the kernel and its compile-relevant
    arguments, so repeat invocations skip benchmarking; changing the kernel
    source, candidate set, static dimensions, or ``tuning_key`` re-tunes.
    ``tuning_key`` may use host-visible tensor metadata and target facts, but
    must not read device values or synchronize; return ``()`` when compile
    identities already distinguish every tuning decision. Passing a custom
    ``benchmark=`` bypasses winner reuse so that timing policy always runs.
    Therefore this convenience API requires repeatable, non-destructive
    launches. An explicit ``target`` is installed process-wide before candidate
    generation and remains active after this function returns.
    """
    if autotune and config is not None:
        raise ValueError("pass either config= or autotune=True, not both")
    if not autotune and configs is not None:
        raise ValueError("configs= requires autotune=True")

    compile_fn = kernel.compile
    if not callable(getattr(compile_fn, "precompile", None)):
        raise TypeError("run_tunable() requires kernel.compile decorated with jit_cache")
    if target is not None:
        set_compile_target(target)
    resolved_target = get_compile_target()
    if autotune:
        # Custom timing policies always benchmark rather than inheriting a winner
        # selected under different criteria.
        reuse_winner = benchmark is None
        tuning_key = (
            kernel.tuning_key(*runtime_args, target=resolved_target) if reuse_winner else None
        )
        if tuning_key is not None and not isinstance(tuning_key, tuple):
            raise TypeError("tuning_key() must return a tuple of host-static values")
        selected = None
        fast_key = None
        if reuse_winner and configs is None:
            default = kernel.default_config(*runtime_args, target=resolved_target)
            try:
                fast_key = (
                    id(kernel),
                    resolved_target,
                    tuning_key,
                    kernel.compile_call(default, *runtime_args),
                )
                selected = _WINNERS_FAST.get(fast_key)
            except TypeError:
                fast_key = None
        if selected is None:
            candidates = list(kernel.configs(*runtime_args) if configs is None else configs)
            winner_key = (
                _winner_key(kernel, candidates, runtime_args, tuning_key)
                if tuning_key is not None
                else None
            )
            selected = None if winner_key is None else _load_winner(winner_key)
            # A cached winner outside the candidate set is stale (config space changed).
            if selected is None or selected not in candidates:

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
                if winner_key is not None:
                    _store_winner(winner_key, selected)
            if fast_key is not None:
                _WINNERS_FAST[fast_key] = selected
    else:
        selected = (
            kernel.default_config(*runtime_args, target=resolved_target)
            if config is None
            else config
        )

    call_args = _materialize_calls((kernel.compile_call(selected, *runtime_args),))[0]
    compiled = compile_fn(*call_args)
    return kernel.launch(compiled, selected, *runtime_args), selected


__all__ = ["TunableKernel", "benchmark_gpu", "run_tunable", "tune"]
