"""Tune Triton kernels with :func:`run_tunable` instead of ``triton.autotune``, or pin them.

``triton.autotune`` and ``triton.heuristics`` rebuild argument dictionaries on every launch;
``TritonTuner`` memoizes the winner under an explicit key and launches the kernel directly.

Every launch says whether it may tune: ``tuner[grid](..., autotune=False)`` never benchmarks.
Given the same launch arguments and software stack, it selects the same configuration on a
given GPU model. Callers resolve the flag; see ``attn_gym._backends.autotune``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import triton
from triton.compiler.errors import CompileTimeAssertionFailure
from triton.runtime.errors import OutOfResources, PTXASError
from triton.runtime.jit import JITFunction
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.tune import run_tunable

# ``Config.all_kwargs()`` as sorted items: hashable and picklable for winner caching.
TritonCandidate = tuple[tuple[str, Any], ...]
Grid = tuple[int, ...] | Callable[[dict[str, Any]], tuple[int, ...]]


class TritonTuner:
    """Tune a ``@triton.jit`` kernel over ``triton.Config`` candidates with ``run_tunable``.

    Launch with ``tuner[grid](*args, autotune=..., **kwargs)``, otherwise as with the kernel
    itself; ``grid`` may be a tuple or ``grid(meta)`` over the arguments and the candidate's
    meta-parameters. The required ``autotune`` keyword selects the launch:

    - ``True`` reuses a cached winner, or benchmarks the candidates if none is cached.
    - ``False`` launches the heuristic default: the first candidate, in list order after
      ``prune``, that the device can run. It never benchmarks or reads the winner cache, so
      its choice is the same in every process on one GPU model.

    Like ``triton.autotune``, either launch skips a candidate that fails to compile or load
    (more shared memory, tensor memory, or threads than the device has).

    Args:
        kernel: The ``@triton.jit`` function, without autotune or heuristics decorators.
        configs: Candidates to benchmark, the heuristic default first.
        key: Maps the launch arguments to the host-static values that decide the winner, like
            ``triton.autotune(key=...)``. It must also determine what ``prune`` keeps: winners
            are remembered per GPU model and key, pinned candidates per ``(device, key)``.
        prune: Optional ``prune(configs, args)`` returning the candidates valid for these
            arguments in pinning order; it may reorder them as well as drop some.
        reset: Optional ``reset(grid, args) -> restore`` for kernels that mutate their inputs;
            see ``run_tunable``'s ``benchmark_reset``.
    """

    # Errors triton.autotune also treats as an unusable candidate rather than a failure.
    benchmark_errors = (OutOfResources, CompileTimeAssertionFailure, PTXASError)

    def __init__(
        self,
        kernel: JITFunction,
        configs: Sequence[triton.Config],
        *,
        key: Callable[[dict[str, Any]], tuple[Any, ...]],
        prune: Callable[[list[triton.Config], dict[str, Any]], list[triton.Config]] | None = None,
        reset: Callable[[Grid, dict[str, Any]], Callable[[], Any]] | None = None,
    ):
        def compile_kernel(*_static_args: Any) -> Any:
            return kernel  # Triton compiles on first launch and caches the binary itself.

        self.jit_function = kernel
        # Winners are namespaced by the kernel source and its @triton.jit dependencies.
        compile_kernel.cache_namespace = lambda: kernel.cache_key  # type: ignore[attr-defined]
        self.compile = compile_kernel
        self._arg_names = kernel.arg_names
        self._configs = list(configs)
        self._candidates = {id(c): tuple(sorted(c.all_kwargs().items())) for c in self._configs}
        self._key = key
        self._prune = prune
        if reset is not None:
            self.benchmark_reset = reset
        self._pinned_candidates: dict[tuple[int, tuple[Any, ...]], TritonCandidate] = {}

    def __getitem__(self, grid: Grid) -> Callable[..., None]:
        def launch(*args: Any, autotune: bool, **kwargs: Any) -> None:
            bound = self.bind(args, kwargs)
            if autotune:
                run_tunable(self, grid, bound, autotune=True)
            else:
                self.launch_pinned(grid, bound)

        return launch

    def launch_pinned(self, grid: Grid, args: dict[str, Any]) -> None:
        """Launch the first candidate the device can run, without benchmarking."""
        # Devices differ in resource limits, so one's usable candidate may not fit another.
        key = (triton.runtime.driver.active.get_current_device(), self._key(args))
        remembered = self._pinned_candidates.get(key)
        candidates = self.configs(grid, args) if remembered is None else [remembered]
        error: Exception = ValueError("a pinned launch requires at least one candidate")
        for candidate in candidates:
            try:
                self.launch(self.compile(), candidate, grid, args)
            except self.benchmark_errors as candidate_error:
                error = candidate_error
                continue
            self._pinned_candidates[key] = candidate
            return
        raise error

    def bind(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
        """Name positional launch arguments by the kernel's parameters."""
        return {**dict(zip(self._arg_names, args)), **kwargs} if args else kwargs

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
        launch_args = {**args, **dict(candidate)}
        kernel[grid(launch_args) if callable(grid) else grid](**launch_args)


def tuning_key(tensor: str, *names: str) -> Callable[[dict[str, Any]], tuple[Any, ...]]:
    """Key a ``TritonTuner`` by the dtype of argument ``tensor`` and the arguments ``names``.

    Unlike ``triton.autotune``'s implicit dtype key, this also sees through TMA descriptors.
    """

    def key(args: dict[str, Any]) -> tuple[Any, ...]:
        value = args[tensor]
        dtype = (value.base if isinstance(value, TensorDescriptor) else value).dtype
        return (dtype, *(args[name] for name in names))

    return key


__all__ = ["TritonTuner", "tuning_key"]
