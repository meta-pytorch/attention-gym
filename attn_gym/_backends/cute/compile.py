"""Process orchestration for parallel CuTeDSL compilation.

The application discovers its target once, starts a fresh compiler process,
and asks it to fork compiler workers. Workers populate the disk cache;
the application then loads artifacts sequentially without sharing CUDA state
or CuTeDSL compiler singletons across processes.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

from .target import CompileTarget, get_compile_target, set_compile_target

T = TypeVar("T")
DEFAULT_COMPILE_TIMEOUT_SECONDS = 15 * 60


def _materialize_calls(calls: Iterable[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """Materialize positional compile calls and reject ambiguous scalar values."""
    result = list(calls)
    if not all(isinstance(call, tuple) for call in result):
        raise TypeError("each compile call must be a tuple of positional static arguments")
    return result


def _worker_count(requested: int | None, call_count: int) -> int:
    requested = min(4, os.cpu_count() or 1) if requested is None else requested
    if requested < 1:
        raise ValueError(f"workers must be positive, got {requested}")
    return min(requested, call_count)


def _function_reference(fn: Callable[..., Any]) -> tuple[str, str]:
    module = fn.__module__
    qualname = fn.__qualname__
    if module == "__main__" or "<locals>" in qualname:
        raise ValueError(
            "parallel CuTeDSL compile functions must be decorated at module scope "
            "in an importable module"
        )
    if not callable(getattr(fn, "precompile", None)):
        raise TypeError("precompile_many() requires a function decorated with jit_cache")
    return module, qualname


def _module_root(fn: Callable[..., Any]) -> str | None:
    module = sys.modules.get(fn.__module__)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    root = Path(module_file).resolve().parent
    levels = len(fn.__module__.split("."))
    if Path(module_file).name != "__init__.py":
        levels -= 1
    for _ in range(levels):
        root = root.parent
    return str(root)


def _import_paths(fn: Callable[..., Any]) -> list[str]:
    paths = []
    module_root = _module_root(fn)
    candidates = ([module_root] if module_root else []) + list(sys.path)
    for path in candidates:
        resolved = os.getcwd() if path == "" else os.fspath(path)
        if resolved not in paths:
            paths.append(resolved)
    return paths


def _driver_error(completed: subprocess.CompletedProcess[str]) -> str:
    details = []
    if completed.stdout:
        details.append("compiler process stdout:\n" + completed.stdout)
    if completed.stderr:
        details.append("compiler process stderr:\n" + completed.stderr)
    return "\n".join(details) or f"compiler process exited with status {completed.returncode}"


def precompile_many(
    fn: Callable[..., Any],
    calls: Iterable[tuple[Any, ...]],
    *,
    workers: int | None = None,
    target: CompileTarget | None = None,
    timeout: float | None = DEFAULT_COMPILE_TIMEOUT_SECONDS,
) -> None:
    """Populate many variants through a fresh compiler process and forked workers.

    Each item in ``calls`` is a tuple of positional static arguments; use
    ``(config,)`` for a one-argument compile function. Values must be pickleable,
    and custom config classes must live at module scope so the fresh compiler
    process can import them.
    """
    call_args = _materialize_calls(calls)
    if not call_args:
        return
    if sys.platform == "win32":  # pragma: no cover - CuTeDSL currently targets Linux.
        raise RuntimeError("parallel CuTeDSL precompilation requires fork support")

    module, qualname = _function_reference(fn)
    disk_cache_enabled = getattr(fn, "disk_cache_enabled", None)
    if callable(disk_cache_enabled) and not disk_cache_enabled():
        raise RuntimeError("parallel CuTeDSL precompilation requires the disk cache")

    target = target or get_compile_target()
    set_compile_target(target)
    is_cached = getattr(fn, "is_cached", None)
    pending_calls = (
        [args for args in call_args if not is_cached(*args)] if callable(is_cached) else call_args
    )
    if not pending_calls:
        return

    request = {
        "module": module,
        "qualname": qualname,
        "workers": _worker_count(workers, len(pending_calls)),
        "target": asdict(target),
        "sys_path": _import_paths(fn),
    }

    driver = Path(__file__).with_name("_compile_driver.py")
    with tempfile.TemporaryDirectory(prefix="attention-gym-cute-compile-") as directory:
        request_path = Path(directory) / "request.json"
        calls_path = Path(directory) / "calls.pkl"
        request_path.write_text(
            json.dumps(request, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        # This trusted, one-shot payload is created and consumed inside the
        # private TemporaryDirectory; persistent cache entries are never pickled.
        calls_path.write_bytes(pickle.dumps(pending_calls, protocol=pickle.HIGHEST_PROTOCOL))
        try:
            completed = subprocess.run(
                [sys.executable, str(driver), str(request_path), str(calls_path)],
                check=False,
                capture_output=True,
                text=True,
                env={**os.environ, "TORCH_WARM_POOL": "0"},
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"parallel CuTeDSL precompilation timed out after {timeout} seconds"
            ) from error

        if completed.returncode != 0:
            raise RuntimeError(
                "parallel CuTeDSL precompilation failed:\n" + _driver_error(completed)
            )


def compile_many(
    fn: Callable[..., T],
    calls: Iterable[tuple[Any, ...]],
    *,
    workers: int | None = None,
    target: CompileTarget | None = None,
    timeout: float | None = DEFAULT_COMPILE_TIMEOUT_SECONDS,
) -> list[T]:
    """Compile variants in parallel, then load them sequentially in the caller."""
    call_args = _materialize_calls(calls)
    precompile_many(
        fn,
        call_args,
        workers=workers,
        target=target,
        timeout=timeout,
    )
    return [fn(*args) for args in call_args]
