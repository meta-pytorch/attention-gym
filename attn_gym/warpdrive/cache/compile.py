"""Parallel CuTeDSL compilation.

The caller resolves its compile target once and starts a fresh compiler process, which forks
workers that fill the disk cache. The caller then loads the entries one at a time. No CUDA
state or CuTeDSL compiler singleton is shared across processes. One batch may mix compile
functions (``precompile_calls``), so a model's kernels of several kinds share one process.

The compiler process leads its own process group, which its workers join: stopping a batch
kills the group, and the process kills it when its caller dies.
"""

from __future__ import annotations

import json
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

from .jit import CacheStorageError, get_cache_path
from .target import CompileTarget, get_compile_target, set_compile_target

T = TypeVar("T")
DEFAULT_COMPILE_TIMEOUT_SECONDS = 15 * 60
PROGRESS_INTERVAL_SECONDS = 0.25
# The compiler process's exit status when its calls failed only to lock or write the disk cache.
CACHE_STORAGE_EXIT = 3


def _materialize_calls(calls: Iterable[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """List the calls, each a tuple of positional arguments; a bare value is ambiguous."""
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


def _import_paths(fns: list[Callable[..., Any]]) -> list[str]:
    paths = []
    module_roots = [root for root in map(_module_root, fns) if root]
    for path in module_roots + list(sys.path):
        resolved = os.getcwd() if path == "" else os.fspath(path)
        if resolved not in paths:
            paths.append(resolved)
    return paths


def _driver_error(returncode: int, stdout: str, stderr: str) -> str:
    details = []
    if stdout:
        details.append("compiler process stdout:\n" + stdout)
    if stderr:
        details.append("compiler process stderr:\n" + stderr)
    return "\n".join(details) or f"compiler process exited with status {returncode}"


def _run_driver(
    command: list[str], timeout: float | None, poll: Callable[[], None]
) -> tuple[int, str, str]:
    """Run the compiler process, calling ``poll`` between waits; its exit status and output.

    Its forked workers hold cache locks and its output pipes, so a timeout, an interruption or a
    failing ``poll`` kills its whole process group before draining the pipes.
    """
    deadline = None if timeout is None else time.monotonic() + timeout
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "TORCH_WARM_POOL": "0"},
        start_new_session=True,
    ) as process:
        try:
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=PROGRESS_INTERVAL_SECONDS)
                    return process.returncode, stdout, stderr
                except subprocess.TimeoutExpired:
                    poll()
                    if deadline is not None and time.monotonic() >= deadline:
                        raise RuntimeError(
                            f"parallel CuTeDSL precompilation timed out after {timeout} seconds"
                        ) from None
        except BaseException:
            # The group, named by the driver's pid, outlives the driver while a worker is left.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:  # the group already exited
                pass
            process.communicate()
            raise


def precompile_many(
    fn: Callable[..., Any],
    calls: Iterable[tuple[Any, ...]],
    *,
    workers: int | None = None,
    target: CompileTarget | None = None,
    timeout: float | None = DEFAULT_COMPILE_TIMEOUT_SECONDS,
) -> None:
    """Fill the disk cache for ``calls`` from a fresh compiler process and forked workers.

    Each call is a tuple of positional static arguments; use ``(config,)`` for a one-argument
    compile function. Values must pickle, and custom config classes must live at module scope
    so the compiler process can import them. Calls already on disk are skipped.
    """
    precompile_calls(
        [(fn, *args) for args in _materialize_calls(calls)],
        workers=workers,
        target=target,
        timeout=timeout,
    )


def precompile_calls(
    calls: Iterable[tuple[Any, ...]],
    *,
    workers: int | None = None,
    target: CompileTarget | None = None,
    timeout: float | None = DEFAULT_COMPILE_TIMEOUT_SECONDS,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """``precompile_many`` of calls to several compile functions, each call ``(fn, *args)``.

    ``progress(done, total)``, if given, reports how many of the ``total`` calls not yet on
    disk have landed there, while the compiler process runs and once it finishes; never when
    every call is already cached. A timeout, an interruption or a ``progress`` exception kills
    the compiler process and its workers. Raises ``CacheStorageError`` when the calls failed
    only to lock or write the disk cache (an unwritable cache directory, say), so a caller can
    compile them without it instead.
    """
    call_args = _materialize_calls(calls)
    if not call_args:
        return
    if sys.platform == "win32":  # pragma: no cover - CuTeDSL currently targets Linux.
        raise RuntimeError("parallel CuTeDSL precompilation requires fork support")

    fns = list(dict.fromkeys(fn for fn, *_ in call_args))  # each once, in call order
    references = [_function_reference(fn) for fn in fns]
    if not all(fn.disk_cache_enabled() for fn in fns):
        raise RuntimeError("parallel CuTeDSL precompilation requires the disk cache")

    target = target or get_compile_target()
    set_compile_target(target)
    pending = [(fn, args) for fn, *args in call_args if not fn.is_cached(*args)]
    if not pending:
        return

    def report() -> None:
        if progress is not None:
            progress(sum(fn.is_cached(*args) for fn, args in pending), len(pending))

    request = {
        "functions": references,
        "workers": _worker_count(workers, len(pending)),
        "target": asdict(target),
        # The driver fills the cache this process reads, including one set by set_cache_path().
        "cache_path": os.fspath(get_cache_path()),
        "sys_path": _import_paths(fns),
        "caller_pid": os.getpid(),
    }
    # The driver resolves each function once; a call names its function by index.
    index = {fn: i for i, fn in enumerate(fns)}
    indexed = [(index[fn], tuple(args)) for fn, args in pending]

    driver = Path(__file__).with_name("_compile_driver.py")
    with tempfile.TemporaryDirectory(prefix="warpdrive-compile-") as directory:
        request_path = Path(directory) / "request.json"
        calls_path = Path(directory) / "calls.pkl"
        request_path.write_text(
            json.dumps(request, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        # These pickled calls are written and read once inside this private directory. The
        # only pickles in persistent entries are trace records, which sit beside an object file
        # that is loaded as code anyway.
        calls_path.write_bytes(pickle.dumps(indexed, protocol=pickle.HIGHEST_PROTOCOL))
        command = [sys.executable, str(driver), str(request_path), str(calls_path)]
        returncode, stdout, stderr = _run_driver(command, timeout, report)

    if returncode == CACHE_STORAGE_EXIT:
        raise CacheStorageError(
            "parallel CuTeDSL precompilation could not use the disk cache:\n"
            + _driver_error(returncode, stdout, stderr)
        )
    if returncode != 0:
        raise RuntimeError(
            "parallel CuTeDSL precompilation failed:\n" + _driver_error(returncode, stdout, stderr)
        )
    report()


def compile_many(
    fn: Callable[..., T],
    calls: Iterable[tuple[Any, ...]],
    *,
    workers: int | None = None,
    target: CompileTarget | None = None,
    timeout: float | None = DEFAULT_COMPILE_TIMEOUT_SECONDS,
) -> list[T]:
    """Compile ``calls`` in parallel, then load them one at a time in the caller."""
    call_args = _materialize_calls(calls)
    precompile_many(
        fn,
        call_args,
        workers=workers,
        target=target,
        timeout=timeout,
    )
    return [fn(*args) for args in call_args]
