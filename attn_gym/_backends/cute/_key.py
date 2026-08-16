"""Deterministic source and invocation keys for the CuTeDSL cache."""

from __future__ import annotations

import dataclasses
import dis
import enum
import functools
import hashlib
import inspect
import os
import pickle
import platform
import sys
import threading
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .target import CompileTarget

CACHE_FORMAT_VERSION = 4


def _hash_file(hasher: Any, label: str, source: Path) -> None:
    content = source.read_bytes()
    encoded_label = label.encode()
    hasher.update(len(encoded_label).to_bytes(8, "little"))
    hasher.update(encoded_label)
    hasher.update(len(content).to_bytes(8, "little"))
    hasher.update(content)


def _hash_source_tree(hasher: Any, root: Path) -> None:
    for source in sorted(root.rglob("*.py")):
        if source.is_file():
            _hash_file(hasher, source.relative_to(root).as_posix(), source)


@functools.cache
def source_fingerprint(
    fn: Callable[..., Any],
    extra_sources: tuple[str, ...] = (),
) -> str:
    """Fingerprint the compile function, CuTe sources, and host/runtime ABI."""
    import cutlass
    import torch
    import tvm_ffi

    hasher = hashlib.sha256()
    stamps = (
        CACHE_FORMAT_VERSION,
        sys.implementation.cache_tag,
        platform.system(),
        platform.machine(),
        cutlass.__version__,
        tvm_ffi.__version__,
        torch.__version__,
        torch.version.cuda,
    )
    hasher.update(pickle.dumps(stamps, protocol=pickle.HIGHEST_PROTOCOL))

    module = sys.modules.get(fn.__module__)
    module_file = getattr(module, "__file__", None)
    if module_file is not None:
        source = Path(module_file).resolve()
        if source.suffix == ".py" and source.is_file():
            _hash_file(hasher, fn.__module__, source)

    package_root = Path(__file__).resolve().parents[2]
    for source in sorted(package_root.rglob("*.py")):
        relative_path = source.relative_to(package_root)
        if "cute" in relative_path.parts:
            _hash_file(hasher, relative_path.as_posix(), source)

    for extra_source in extra_sources:
        path = Path(extra_source).expanduser().resolve()
        if path.is_dir():
            _hash_source_tree(hasher, path)
        elif path.is_file():
            _hash_file(hasher, str(path), path)
        else:
            raise FileNotFoundError(f"extra CuTeDSL cache source does not exist: {path}")

    cutlass_root = Path(cutlass.__file__).resolve().parent
    for relative_path in (
        "__init__.py",
        "base_dsl/compiler.py",
        "base_dsl/dsl.py",
        "cutlass_dsl/tvm_ffi_provider.py",
        "cute/runtime.py",
    ):
        source = cutlass_root / relative_path
        if source.is_file():
            _hash_file(hasher, f"cutlass/{relative_path}", source)

    codegen_environment = tuple(
        (name, os.getenv(name))
        for name in (
            "CUTE_DSL_ARCH",
            "CUTE_DSL_COMPILER_OPT",
            "CUTE_DSL_ENABLE_ASSERTIONS",
            "CUTE_DSL_ENABLE_TVM_FFI",
            "CUTE_DSL_LIBS",
            "CUTE_DSL_LINEINFO",
        )
    )
    hasher.update(pickle.dumps(codegen_environment, protocol=pickle.HIGHEST_PROTOCOL))
    return hasher.hexdigest()


def _is_named_tuple(item: Any) -> bool:
    fields = getattr(type(item), "_fields", None)
    return isinstance(item, tuple) and isinstance(fields, tuple)


@functools.cache
def _unwrapped(function: types.FunctionType) -> types.FunctionType:
    """Resolve decorator wrappers once; wrapper chains are fixed at definition."""
    return inspect.unwrap(function)


@functools.cache
def _function_static_parts(function: types.FunctionType) -> tuple[str, tuple[str, ...]]:
    """Fetch a function's source and referenced global names once per function."""
    try:
        source = inspect.getsource(function)
    except (OSError, TypeError) as error:
        raise TypeError(
            f"function {function.__qualname__!r} has no stable cache key; it must be "
            "defined in a source file so compiled kernels can key on its code"
        ) from error
    global_names = tuple(
        sorted(
            {
                instruction.argval
                for instruction in dis.get_instructions(function)
                if instruction.opname == "LOAD_GLOBAL"
            }
        )
    )
    return source, global_names


_FUNCTIONS_IN_PROGRESS = threading.local()


def _canonicalize_function(function: types.FunctionType) -> tuple[Any, ...]:
    """Encode a function by content: source plus current closure and global values.

    Compiled CuTeDSL kernels inline the traced expressions, so the identity must
    cover everything that shapes the generated code: the source text, captured
    closure cells, and referenced module-global values (imported modules and
    builtins are stable references and are skipped). Values are re-read on every
    key computation, so mutating a captured global recompiles instead of
    silently reusing a kernel traced with the old value.
    """
    unwrapped = _unwrapped(function)
    in_progress = getattr(_FUNCTIONS_IN_PROGRESS, "stack", None)
    if in_progress is None:
        in_progress = _FUNCTIONS_IN_PROGRESS.stack = set()
    if id(unwrapped) in in_progress:
        raise TypeError(
            f"function {unwrapped.__qualname__!r} participates in a reference cycle "
            "and has no stable cache key"
        )
    in_progress.add(id(unwrapped))
    try:
        source, global_names = _function_static_parts(unwrapped)
        cells = tuple(_canonicalize(cell.cell_contents) for cell in unwrapped.__closure__ or ())
        module_globals = unwrapped.__globals__
        globals_used = tuple(
            (name, _canonicalize(module_globals[name]))
            for name in global_names
            if name in module_globals and not isinstance(module_globals[name], types.ModuleType)
        )
        # Default values appear in the source text, but a *mutable* default
        # object can change behavior after definition without changing it.
        defaults = _canonicalize(unwrapped.__defaults__)
        keyword_defaults = _canonicalize(unwrapped.__kwdefaults__)
    finally:
        in_progress.discard(id(unwrapped))
    return ("function", source, cells, globals_used, defaults, keyword_defaults)


def function_cache_key(function: Callable[..., Any]) -> tuple[Any, ...]:
    """Return the content-addressed cache identity of a traced function."""
    return _canonicalize_function(function)


def _pickle_sort_key(item: Any) -> bytes:
    return pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)


def _canonicalize(item: Any) -> Any:
    custom_key = getattr(item, "__attention_gym_cache_key__", None)
    if custom_key is not None:
        value = custom_key() if callable(custom_key) else custom_key
        return (
            "custom",
            type(item).__module__,
            type(item).__qualname__,
            _canonicalize(value),
        )
    if _is_named_tuple(item):
        return (
            "named_tuple",
            type(item).__module__,
            type(item).__qualname__,
            tuple((name, _canonicalize(getattr(item, name))) for name in type(item)._fields),
        )
    if dataclasses.is_dataclass(item) and not isinstance(item, type):
        return (
            "dataclass",
            type(item).__module__,
            type(item).__qualname__,
            tuple(
                (field.name, _canonicalize(getattr(item, field.name)))
                for field in dataclasses.fields(item)
            ),
        )
    if isinstance(item, enum.Enum):
        return ("enum", type(item).__module__, type(item).__qualname__, item.name)
    if isinstance(item, types.FunctionType):
        return _canonicalize_function(item)
    if isinstance(item, type):
        return ("type", item.__module__, item.__qualname__)
    if isinstance(item, Path):
        return ("path", str(item))
    if isinstance(item, tuple):
        return ("tuple", tuple(_canonicalize(value) for value in item))
    if isinstance(item, list):
        return ("list", tuple(_canonicalize(value) for value in item))
    if isinstance(item, dict):
        entries = [(_canonicalize(key), _canonicalize(value)) for key, value in item.items()]
        return ("dict", tuple(sorted(entries, key=lambda entry: _pickle_sort_key(entry[0]))))
    if isinstance(item, set):
        values = [_canonicalize(value) for value in item]
        return ("set", tuple(sorted(values, key=_pickle_sort_key)))
    if isinstance(item, frozenset):
        values = [_canonicalize(value) for value in item]
        return ("frozenset", tuple(sorted(values, key=_pickle_sort_key)))
    try:
        pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
    except (pickle.PickleError, TypeError) as error:
        raise TypeError(
            f"CuTeDSL cache argument of type {type(item).__qualname__} has no stable key; "
            "pass static pickleable values or define __attention_gym_cache_key__"
        ) from error
    return item


def make_runtime_key(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: CompileTarget,
) -> bytes:
    """Encode the persistent invocation identity for process-local lookup."""
    key_data = (_canonicalize(args), _canonicalize(kwargs), target)
    return pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)


def make_key(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: CompileTarget,
) -> str:
    """Hash one compile invocation and its complete target contract."""
    key_data = (
        CACHE_FORMAT_VERSION,
        fn.__module__,
        fn.__qualname__,
        _canonicalize(args),
        _canonicalize(kwargs),
        target,
    )
    encoded = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
    return hashlib.sha256(encoded).hexdigest()
