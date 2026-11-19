"""Compiler process that ``precompile_many`` starts in a fresh interpreter.

It installs the caller's import paths and compile target, resolves the compile functions and
loads the pickled compile calls (each ``(function index, args)``), then forks workers. A worker
receives only an index into the calls it inherited. It leads the process group its workers join,
and kills that group if its caller dies, which signals sent to the caller's group no longer reach.
"""

from __future__ import annotations

import importlib
import json
import os
import pickle
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path
from typing import Any

CALLER_POLL_SECONDS = 0.5

_compile_fns: list[Any] = []
_compile_calls: list[tuple[int, tuple[Any, ...]]] = []


def _resolve(module_name: str, qualname: str) -> Any:
    value: Any = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    return value


def _precompile_call(index: int) -> None:
    if not _compile_fns:
        raise RuntimeError("compiler worker was forked without compile functions")
    fn_index, args = _compile_calls[index]
    _compile_fns[fn_index].precompile(*args)


def main(request_path: Path, calls_path: Path) -> int:
    global _compile_calls, _compile_fns
    if os.getpgrp() != os.getpid():
        raise SystemExit("the compiler process must lead its own process group")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    inherited_paths = request["sys_path"]
    sys.path[:] = inherited_paths + [path for path in sys.path if path not in inherited_paths]

    try:
        import torch._thread_safe_fork  # noqa: F401
    except ImportError:
        pass

    target_fields = request["target"]
    capability = target_fields.get("capability")
    if capability is not None:
        target_fields["capability"] = tuple(capability)
    # CuTeDSL reads CUTE_DSL_ARCH when cutlass is first imported, and importing attn_gym
    # imports cutlass: set it before the imports below.
    if request["gpu_arch"] is not None:
        os.environ["CUTE_DSL_ARCH"] = request["gpu_arch"]

    from attn_gym.warpdrive.cache.compile import CACHE_STORAGE_EXIT
    from attn_gym.warpdrive.cache.jit import CacheStorageError, set_cache_path
    from attn_gym.warpdrive.cache.target import CompileTarget, set_compile_target

    set_compile_target(CompileTarget(**target_fields))
    set_cache_path(request["cache_path"])

    _compile_fns = [_resolve(module, qualname) for module, qualname in request["functions"]]
    if not all(callable(getattr(fn, "precompile", None)) for fn in _compile_fns):
        raise TypeError("compile function is not decorated with jit_cache")

    # Safe to unpickle: the parent wrote this file in its private TemporaryDirectory. It never
    # comes from a persistent cache or an outside caller.
    _compile_calls = pickle.loads(calls_path.read_bytes())

    # Import and hash sources once, before forking, so workers inherit the result. The target
    # is already set, so this does no CUDA device discovery.
    for fn in _compile_fns:
        fn.prepare_cache()

    context = get_context("fork")
    with ProcessPoolExecutor(
        max_workers=request["workers"],
        mp_context=context,
    ) as pool:
        futures = [pool.submit(_precompile_call, index) for index in range(len(_compile_calls))]
        running = set(futures)
        while running:
            _, running = wait(running, timeout=CALLER_POLL_SECONDS)
            if os.getppid() != request["caller_pid"]:  # orphaned: stop the workers too
                os.killpg(0, signal.SIGKILL)
    errors = [error for future in futures if (error := future.exception()) is not None]
    if compile_errors := [e for e in errors if not isinstance(e, CacheStorageError)]:
        raise compile_errors[0]
    if errors:
        print("\n".join(map(str, errors)), file=sys.stderr)
        return CACHE_STORAGE_EXIT
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: _compile_driver.py REQUEST CALLS")
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
