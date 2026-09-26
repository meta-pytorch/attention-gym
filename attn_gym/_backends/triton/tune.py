"""Tune Triton kernels with :func:`run_tunable` instead of ``triton.autotune``.

``triton.autotune`` and ``triton.heuristics`` rebuild argument dictionaries on every launch;
``TritonTuner`` memoizes the winner under an explicit key and launches the kernel directly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import triton

from attn_gym._backends.cute import run_tunable

# ``Config.all_kwargs()`` as sorted items: hashable and picklable for winner caching.
TritonCandidate = tuple[tuple[str, Any], ...]
Grid = tuple[int, ...] | Callable[[dict[str, Any]], tuple[int, ...]]


class TritonTuner:
    """Tune a ``@triton.jit`` kernel over ``triton.Config`` candidates with ``run_tunable``.

    Launch with ``tuner[grid](**args)`` as with the kernel itself; ``grid`` may be a tuple or
    ``grid(meta)`` over the arguments and the candidate's meta-parameters.

    Args:
        kernel: The ``@triton.jit`` function, without autotune or heuristics decorators.
        configs: Candidates to benchmark.
        key: Maps the launch arguments to the host-static values that decide the winner, like
            ``triton.autotune(key=...)``.
        prune: Optional ``prune(configs, args)`` dropping candidates invalid for these arguments.
        reset: Optional ``reset(grid, args) -> restore`` for kernels that mutate their inputs;
            see ``run_tunable``'s ``benchmark_reset``.
    """

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
        self._configs = list(configs)
        self._candidates = {id(c): tuple(sorted(c.all_kwargs().items())) for c in self._configs}
        self._key = key
        self._prune = prune
        if reset is not None:
            self.benchmark_reset = reset

    def __getitem__(self, grid: Grid) -> Callable[..., None]:
        return lambda **args: run_tunable(self, grid, args, autotune=True)

    def configs(self, _grid: Grid, args: dict[str, Any]) -> list[TritonCandidate]:
        configs = self._configs if self._prune is None else self._prune(self._configs, args)
        return [self._candidates[id(config)] for config in configs]

    def default_config(self, _grid: Grid, _args: dict, *, target: Any) -> TritonCandidate:
        # TritonTuner always tunes, so the default only probes run_tunable's winner memo.
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


__all__ = ["TritonTuner"]
