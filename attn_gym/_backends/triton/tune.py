"""Tune Triton kernels with :func:`run_tunable` instead of ``triton.autotune``, or pin them.

``triton.autotune`` and ``triton.heuristics`` rebuild argument dictionaries on every launch;
``TritonTuner`` memoizes the winner under an explicit key and launches the kernel directly.

Every tuned kernel also has a pinned launch that never benchmarks, for callers that need the
same configuration in every process: ``PinnedConfigKernel`` (``tuner.pinned`` for a
``TritonTuner``). The caller chooses between the two; see ``attn_gym._backends.autotune``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import triton
from triton.compiler.errors import CompileTimeAssertionFailure
from triton.runtime.autotuner import Autotuner, Heuristics
from triton.runtime.errors import OutOfResources, PTXASError

from attn_gym._backends.cute import run_tunable

# ``Config.all_kwargs()`` as sorted items: hashable and picklable for winner caching.
TritonCandidate = tuple[tuple[str, Any], ...]
Grid = tuple[int, ...] | Callable[[dict[str, Any]], tuple[int, ...]]


class TritonTuner:
    """Tune a ``@triton.jit`` kernel over ``triton.Config`` candidates with ``run_tunable``.

    Launch with ``tuner[grid](*args, **kwargs)`` as with the kernel itself; ``grid`` may be a
    tuple or ``grid(meta)`` over the arguments and the candidate's meta-parameters.

    Like ``triton.autotune``, a candidate that fails to compile or load while tuning (more
    shared memory, tensor memory, or threads than the device has) is skipped.

    ``tuner.pinned[grid](*args, **kwargs)`` launches the heuristic default instead: the first
    candidate, in list order after ``prune``, that the device can run. It never benchmarks or
    reads the winner cache, so its choice is the same in every process on one GPU model.

    Args:
        kernel: The ``@triton.jit`` function, without autotune or heuristics decorators.
        configs: Candidates to benchmark, the heuristic default first.
        key: Maps the launch arguments to the host-static values that decide the winner, like
            ``triton.autotune(key=...)``. It must also decide what ``prune`` keeps: the tuned
            winner and the pinned candidate are both remembered per key.
        prune: Optional ``prune(configs, args)`` dropping candidates invalid for these arguments.
        reset: Optional ``reset(grid, args) -> restore`` for kernels that mutate their inputs;
            see ``run_tunable``'s ``benchmark_reset``.
    """

    # Errors triton.autotune also treats as an unusable candidate rather than a failure.
    benchmark_errors = (OutOfResources, CompileTimeAssertionFailure, PTXASError)

    def __init__(
        self,
        kernel: Any,
        configs: Sequence[triton.Config],
        *,
        key: Callable[[dict[str, Any]], tuple[Any, ...]],
        prune: Callable[[list[triton.Config], dict[str, Any]], list[triton.Config]] | None = None,
        reset: Callable[[Grid, dict[str, Any]], Callable[[], Any]] | None = None,
    ):
        def compile_kernel(*_static_args: Any) -> Any:
            return kernel  # Triton compiles on first launch and caches the binary itself.

        # Winners are namespaced by the kernel source and its @triton.jit dependencies.
        compile_kernel.cache_namespace = lambda: kernel.cache_key  # type: ignore[attr-defined]
        self.compile = compile_kernel
        self._kernel = kernel
        self._arg_names = kernel.arg_names
        self._configs = list(configs)
        self._candidates = {id(c): tuple(sorted(c.all_kwargs().items())) for c in self._configs}
        self._key = key
        self._prune = prune
        if reset is not None:
            self.benchmark_reset = reset
        self._pinned_candidates: dict[tuple[Any, ...], TritonCandidate] = {}
        self.pinned = PinnedConfigKernel(self)

    def __getitem__(self, grid: Grid) -> Callable[..., None]:
        return lambda *args, **kwargs: run_tunable(
            self, grid, self.bind(args, kwargs), autotune=True
        )

    def launch_pinned(self, grid: Grid, args: dict[str, Any]) -> None:
        """Launch the first candidate the device can run, without benchmarking."""
        key = self._key(args)
        candidate = self._pinned_candidates.get(key)
        if candidate is not None:
            self.launch(self._kernel, candidate, grid, args)
            return
        candidates = self.configs(grid, args)
        if not candidates:
            raise ValueError("a pinned launch requires at least one candidate")
        for candidate in candidates:
            try:
                self.launch(self._kernel, candidate, grid, args)
            except self.benchmark_errors as candidate_error:
                error = candidate_error
                continue
            self._pinned_candidates[key] = candidate
            return
        raise error

    def bind(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Name positional launch arguments by the kernel's parameters."""
        return {**dict(zip(self._arg_names, args)), **kwargs}

    def configs(self, _grid: Grid, args: dict[str, Any]) -> list[TritonCandidate]:
        configs = self._configs if self._prune is None else self._prune(self._configs, args)
        return [self._candidates[id(config)] for config in configs]

    def default_config(self, _grid: Grid, _args: dict, *, target: Any) -> TritonCandidate:
        # run_tunable is only asked to tune, so the default only probes its winner memo.
        return self._candidates[id(self._configs[0])]

    def tuning_key(self, _grid: Grid, args: dict[str, Any], *, target: Any) -> tuple[Any, ...]:
        return self._key(args)

    def compile_call(self, candidate: TritonCandidate, *_runtime_args: Any) -> tuple[Any, ...]:
        return (candidate,)

    def launch(self, kernel: Any, candidate: TritonCandidate, grid: Grid, args: dict) -> None:
        meta = dict(candidate)
        if callable(grid):
            grid = grid({**args, **meta})
        kernel[grid](**args, **meta)


class PinnedConfigKernel:
    """Launch a tuned kernel's heuristic default without benchmarking or reading winner caches.

    A ``TritonTuner`` uses the first candidate left by ``prune`` that the device can run.
    A ``triton.autotune`` kernel uses its first config as listed, without pruning or fallback;
    that config must be valid for every accepted input. Outer ``triton.heuristics`` still
    apply to the autotuned kernel. Launch with ``pinned[grid](*args, **kwargs)``.
    """

    def __init__(self, kernel: Any):
        if isinstance(kernel, TritonTuner):
            self._launch = lambda grid, args, kwargs: kernel.launch_pinned(
                grid, kernel.bind(args, kwargs)
            )
            return
        heuristic_values = None
        if isinstance(kernel, Heuristics):
            heuristic_values = kernel.values
            kernel = kernel.fn
        if not isinstance(kernel, Autotuner):
            raise TypeError(f"expected a tuned kernel, got {type(kernel).__name__}")
        config_kwargs = kernel.configs[0].all_kwargs()
        jit_kernel = (
            triton.heuristics(heuristic_values)(kernel.fn) if heuristic_values else kernel.fn
        )
        self._launch = lambda grid, args, kwargs: jit_kernel[grid](
            *args, **kwargs, **config_kwargs
        )

    def __getitem__(self, grid: Any) -> Callable[..., None]:
        return lambda *args, **kwargs: self._launch(grid, args, kwargs)


__all__ = ["PinnedConfigKernel", "TritonTuner"]
