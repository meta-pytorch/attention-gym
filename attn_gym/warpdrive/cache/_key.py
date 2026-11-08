"""Deterministic source and invocation keys for the CuTeDSL cache."""

from __future__ import annotations

import ast
import dataclasses
import dis
import enum
import functools
import hashlib
import importlib.util
import inspect
import os
import pickle
import platform
import re
import sys
import threading
import types
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from .target import CompileTarget

CACHE_FORMAT_VERSION = 5

# The top-level package that ships this module. Keys hash the modules under it a kernel imports.
_PACKAGE_ROOT = Path(__file__).resolve().parents[__name__.count(".") - 1]
# String literals shaped like dotted module names count as imports, e.g. import_module("pkg.x").
_MODULE_NAME = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")


def _hash_file(hasher: Any, label: str, source: Path) -> None:
    content = source.read_bytes()
    encoded_label = label.encode()
    hasher.update(len(encoded_label).to_bytes(8, "little"))
    hasher.update(encoded_label)
    hasher.update(len(content).to_bytes(8, "little"))
    hasher.update(content)


def _imported_names(source: Path, package: str) -> Iterator[str]:
    """Yield a static superset of the dotted module names ``source`` may import.

    Counts ``import``/``from`` statements anywhere in the file, function-local and
    ``TYPE_CHECKING`` ones included, and string literals shaped like module names, such as
    ``importlib.import_module("pkg.module")``. ``from pkg import name`` yields both ``pkg``,
    which may define ``name``, and ``pkg.name``, which may be a submodule.

    Relative names in strings (``importlib.import_module(".mod", __package__)``) are not
    followed. That is how host-only code that tracing never runs stays out of the key (see
    "Adding CuTeDSL Kernels" in AGENTS.md).
    """
    for node in ast.walk(ast.parse(source.read_bytes(), filename=str(source))):
        match node:
            case ast.Import(names=aliases):
                yield from (alias.name for alias in aliases)
            case ast.ImportFrom(module=module, level=level, names=aliases) if package or not level:
                base = importlib.util.resolve_name("." * level + (module or ""), package)
                yield base
                yield from (f"{base}.{alias.name}" for alias in aliases)
            case ast.Constant(value=str(value)) if _MODULE_NAME.fullmatch(value):
                yield value


@functools.cache
def _direct_imports(source: Path) -> frozenset[Path]:
    """Return the package modules ``source`` may import.

    Parent ``__init__`` files that ``import pkg.sub.module`` runs implicitly are not followed;
    only an explicit import of the package reaches their names.
    """
    top = _PACKAGE_ROOT.parent
    # A caller outside the package cannot reach it by relative import.
    package = ".".join(source.parent.relative_to(top).parts) if source.is_relative_to(top) else ""
    found: set[Path] = set()
    for name in _imported_names(source, package):
        if name.partition(".")[0] != _PACKAGE_ROOT.name:
            continue
        module = top.joinpath(*name.split("."))
        for path in (module.with_suffix(".py"), module / "__init__.py"):
            if path.is_file():
                found.add(path)
                break
    return frozenset(found)


def module_closure(source: Path) -> tuple[Path, ...]:
    """Return ``source`` and every package module it transitively imports, sorted.

    A compiled kernel inlines whatever package code runs while tracing, module-level constants
    included, so every module the compile function's module can reach by import is part of its
    source identity. Unrelated package modules are not.
    """
    seen: set[Path] = set()
    pending = [source.resolve()]
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(_direct_imports(current) - seen)
    return tuple(sorted(seen))


def _hash_module_closure(hasher: Any, module_name: str) -> bool:
    """Hash an imported module's ``module_closure``; False when it has no Python source."""
    module_file = getattr(sys.modules.get(module_name), "__file__", None)
    if module_file is None or not module_file.endswith(".py"):
        return False
    for path in module_closure(Path(module_file)):
        # Package-relative labels let checkouts and installs share entries. Only the caller
        # itself can live outside the package.
        if path.is_relative_to(_PACKAGE_ROOT):
            _hash_file(hasher, path.relative_to(_PACKAGE_ROOT).as_posix(), path)
        else:
            _hash_file(hasher, module_name, path)
    return True


@functools.cache
def module_source_digest(module_name: str) -> str | None:
    """Hash an imported module's ``module_closure``; None for a module without Python source.

    A class passed as a compile argument keys by its module and name only. Include this digest
    in its ``__cute_cache_key__`` so that editing a class defined outside the compile
    function's imports (a test's kernel subclass, say) changes the key, without naming that
    module in ``extra_sources`` up front.
    """
    hasher = hashlib.sha256()
    return hasher.hexdigest() if _hash_module_closure(hasher, module_name) else None


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

    _hash_module_closure(hasher, fn.__module__)

    for extra_source in extra_sources:
        path = Path(extra_source).expanduser().resolve()
        if path.is_dir():
            for file in sorted(path.rglob("*.py")):
                _hash_file(hasher, file.relative_to(path).as_posix(), file)
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

    # The architecture is keyed through CompileTarget, not here.
    codegen_environment = tuple(
        (name, os.getenv(name))
        for name in (
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
    """Unwrap decorators once; a wrapper chain is fixed at definition."""
    return inspect.unwrap(function)


@functools.cache
def _function_static_parts(function: types.FunctionType) -> tuple[str, tuple[str, ...]]:
    """A function's source and the global names it loads, read once per function."""
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
    """Key a function by content: its source and current closure and global values.

    A compiled kernel inlines the traced expressions, so the key covers everything that shapes
    the generated code: source text, closure cells and referenced module globals. Imported
    modules and builtins are stable references and are skipped. Values are re-read for every
    key, so mutating a captured global recompiles rather than reusing a kernel traced with the
    old value.
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
        # Defaults appear in the source, but a mutable default can change after definition.
        defaults = _canonicalize(unwrapped.__defaults__)
        keyword_defaults = _canonicalize(unwrapped.__kwdefaults__)
    finally:
        in_progress.discard(id(unwrapped))
    return ("function", source, cells, globals_used, defaults, keyword_defaults)


def function_cache_key(function: Callable[..., Any]) -> tuple[Any, ...]:
    """Return the content-addressed cache key of a traced function."""
    return _canonicalize_function(function)


def _pickle_sort_key(item: Any) -> bytes:
    return pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)


def _canonicalize(item: Any) -> Any:
    # A class passed as a value keys by name, even if its instances define the hook.
    custom_key = None if isinstance(item, type) else getattr(item, "__cute_cache_key__", None)
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
            "pass static pickleable values or define __cute_cache_key__"
        ) from error
    return item


def canonical_key(item: Any) -> Any:
    """Return the content-addressed, picklable encoding the cache keys ``item`` by.

    Freezing it early (warpdrive freezes a kernel's constructor call) keeps later in-place
    changes to ``item`` out of the key.
    """
    return _canonicalize(item)


def make_runtime_key(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: CompileTarget,
) -> bytes:
    """Encode a call's canonical arguments and target as a process-local cache key."""
    key_data = (_canonicalize(args), _canonicalize(kwargs), target)
    return pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)


def make_key(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target: CompileTarget,
) -> str:
    """Hash one compile call and its target into a disk-entry key."""
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
