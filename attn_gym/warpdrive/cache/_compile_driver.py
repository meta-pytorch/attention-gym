"""Compiler process that ``precompile_many`` starts in a fresh interpreter.

It installs the caller's import paths and compile target, loads the pickled compile calls,
then forks workers. A worker receives only an index into the calls it inherited.
"""

from __future__ import annotations

import importlib
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any

_compile_fn: Any | None = None
_compile_calls: list[tuple[Any, ...]] = []


def _resolve(module_name: str, qualname: str) -> Any:
    value: Any = importlib.import_module(module_name)
    for component in qualname.split("."):
        value = getattr(value, component)
    return value


def _precompile_call(index: int) -> None:
    if _compile_fn is None:
        raise RuntimeError("compiler worker was forked without a compile function")
    _compile_fn.precompile(*_compile_calls[index])


def main(request_path: Path, calls_path: Path) -> int:
    global _compile_calls, _compile_fn
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
    if target_fields.get("configured_arch") is not None:
        os.environ["CUTE_DSL_ARCH"] = target_fields["configured_arch"]
    elif capability is not None:
        major, minor = capability
        suffix = "a" if major >= 9 else ""
        os.environ["CUTE_DSL_ARCH"] = f"sm_{major}{minor}{suffix}"

    from attn_gym.warpdrive.cache.jit import set_cache_path
    from attn_gym.warpdrive.cache.target import CompileTarget, set_compile_target

    set_compile_target(CompileTarget(**target_fields))
    set_cache_path(request["cache_path"])

    _compile_fn = _resolve(request["module"], request["qualname"])
    if not callable(getattr(_compile_fn, "precompile", None)):
        raise TypeError("compile function is not decorated with jit_cache")

    # Safe to unpickle: the parent wrote this file in its private TemporaryDirectory. It never
    # comes from a persistent cache or an outside caller.
    _compile_calls = pickle.loads(calls_path.read_bytes())

    # Import and hash sources once, before forking, so workers inherit the result. The target
    # is already set, so this does no CUDA device discovery.
    prepare_cache = getattr(_compile_fn, "prepare_cache", None)
    if callable(prepare_cache):
        prepare_cache()

    context = get_context("fork")
    with ProcessPoolExecutor(
        max_workers=request["workers"],
        mp_context=context,
    ) as pool:
        futures = [pool.submit(_precompile_call, index) for index in range(len(_compile_calls))]
        for future in futures:
            future.result()
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: _compile_driver.py REQUEST CALLS")
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
