"""Persistent cache for TVM-FFI CuTeDSL kernels.

``jit_cache`` decorates a module-level function that calls ``cute.compile``. Its static
arguments and the ``CompileTarget`` form the cache key. A cold entry compiles under a per-key
file lock shared across processes and is published atomically; a warm entry loads the object
file. Arguments that define ``__cute_trace_record__`` and ``__cute_restore_trace__`` get back
the host-side results of the trace that built the kernel on every cached call. That record is
pickled beside the object file, which is loaded as trusted code anyway.

Entries live under ``get_cache_path()``: ``set_cache_path()``, else ``WARPDRIVE_CACHE_DIR``, else
``$XDG_CACHE_HOME/warpdrive``. Other behavior is expressed through ``jit_cache`` options. The
upstream ``CUTE_DSL_NO_CACHE`` switch is still respected.
"""

from __future__ import annotations

import ctypes
import enum
import errno
import functools
import logging
import math
import os
import pickle
import tempfile
import threading
import time
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import torch

from ._key import make_key as _make_key
from ._key import make_runtime_key as _make_runtime_key
from ._key import source_fingerprint as _source_fingerprint
from .target import CompileTarget, get_compile_target

if TYPE_CHECKING:
    from typing_extensions import Self

try:
    import fcntl
except ImportError:  # pragma: no cover - CuTeDSL currently targets Linux.
    fcntl = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

EXPORT_FUNCTION_NAME = "func"
DEFAULT_LOCK_TIMEOUT_SECONDS = 600.0

P = ParamSpec("P")
T = TypeVar("T")

_compile_lock = threading.Lock()
_compile_lock_pid = os.getpid()
_runtime_library_handles: list[Any] = []
_runtime_library_lock = threading.Lock()
_runtime_library_lock_pid = os.getpid()


@dataclass(frozen=True)
class CacheInfo:
    """Process-local cache statistics for one decorated compile function."""

    hits: int
    misses: int
    currsize: int


@dataclass(frozen=True)
class _CacheEntry:
    key: str
    object_path: Path
    lock_path: Path

    @property
    def trace_path(self) -> Path:
        return self.object_path.with_suffix(".trace")


class CacheLockError(RuntimeError):
    """Raised when a persistent cache entry cannot be locked."""


class _FileLock:
    """Hold an exclusive advisory file lock with a bounded wait."""

    def __init__(self, path: Path, timeout: float):
        self.path = path
        self.timeout = timeout
        self.fd = -1

    def __enter__(self) -> Self:
        if fcntl is None:
            raise CacheLockError("persistent CuTeDSL caching requires fcntl.flock")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as error:
            raise CacheLockError(f"could not open cache lock {self.path}") from error

        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EAGAIN):
                    self._close()
                    raise CacheLockError(f"could not lock cache entry {self.path}") from error
                if time.monotonic() >= deadline:
                    self._close()
                    raise CacheLockError(
                        f"timed out waiting for cache lock {self.path}"
                    ) from error
                time.sleep(0.05)

    def __exit__(self, *exc: object) -> None:
        if self.fd >= 0:
            assert fcntl is not None
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self._close()

    def _close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


_cache_path: Path | None = None


def set_cache_path(path: str | os.PathLike[str] | None) -> None:
    """Store persistent artifacts under ``path``; ``None`` restores the environment default."""
    global _cache_path
    _cache_path = None if path is None else Path(path).expanduser()


def get_cache_path() -> Path:
    """Return the root directory used for persistent CuTeDSL artifacts."""
    if _cache_path is not None:
        return _cache_path
    configured = os.getenv("WARPDRIVE_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_cache = os.getenv("XDG_CACHE_HOME")
    cache_home = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return cache_home / "warpdrive"


def cache_enabled() -> bool:
    """False when ``CUTE_DSL_NO_CACHE`` is set to anything but 0/false/no/off (any case)."""
    value = os.getenv("CUTE_DSL_NO_CACHE")
    return value is None or value.lower() in {"0", "false", "no", "off"}


def _cache_entry(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    extra_sources: tuple[str, ...],
    target: CompileTarget,
) -> _CacheEntry:
    key = _make_key(fn, args, kwargs, target)
    directory = get_cache_path() / _source_fingerprint(fn, extra_sources)
    return _CacheEntry(key, directory / f"{key}.o", directory / f"{key}.lock")


def _compile(fn: Callable[P, T], args: tuple[Any, ...], kwargs: dict[str, Any]) -> T:
    """Run one CuTeDSL compile at a time in this process; a forked child gets a new lock."""
    global _compile_lock, _compile_lock_pid
    current_pid = os.getpid()
    if current_pid != _compile_lock_pid:
        _compile_lock = threading.Lock()
        _compile_lock_pid = current_pid
    with _compile_lock:
        return fn(*args, **kwargs)


def _ensure_runtime_libraries_global() -> None:
    """Load the CuTeDSL runtime libraries RTLD_GLOBAL so cached objects resolve symbols."""
    global _runtime_library_lock, _runtime_library_lock_pid
    if _runtime_library_handles or not hasattr(ctypes, "RTLD_GLOBAL"):
        return
    current_pid = os.getpid()
    if current_pid != _runtime_library_lock_pid:
        _runtime_library_lock = threading.Lock()
        _runtime_library_lock_pid = current_pid
    with _runtime_library_lock:
        if _runtime_library_handles:
            return
        from cutlass import cute

        for library_path in cute.runtime.find_runtime_libraries(enable_tvm_ffi=False):
            path = Path(library_path)
            if path.exists():
                _runtime_library_handles.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))


def _load_compiled(path: Path) -> Any:
    """Load a TVM-FFI callable from an exported object file."""
    from cutlass import cute

    _ensure_runtime_libraries_global()
    module = cute.runtime.load_module(str(path), enable_tvm_ffi=True)
    return module[EXPORT_FUNCTION_NAME]


def _trace_holders(args: tuple[Any, ...], kwargs: dict[str, Any]) -> list[Any]:
    """Arguments that keep trace-time results: positional first, then keywords by name.

    A class passed as a value is not one, even if its instances are; its hooks are unbound.
    """
    values = (*args, *(kwargs[name] for name in sorted(kwargs)))
    return [
        value
        for value in values
        if not isinstance(value, type) and callable(getattr(value, "__cute_trace_record__", None))
    ]


def _trace_records(holders: list[Any]) -> tuple[Any, ...] | None:
    """Capture what tracing left on ``holders``; None when no argument keeps any."""
    return tuple(holder.__cute_trace_record__() for holder in holders) if holders else None


def _restore_traces(holders: list[Any], records: tuple[Any, ...] | None) -> None:
    """Give a cached call's arguments the results of the trace that built the kernel."""
    for holder, record in zip(holders, records or ()):
        holder.__cute_restore_trace__(record)


def _publish_records(records: tuple[Any, ...], destination: Path) -> None:
    """Atomically publish an entry's trace records; callers do so before its object file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp.trace", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as artifact:
            pickle.dump(records, artifact, protocol=pickle.HIGHEST_PROTOCOL)
            artifact.flush()
            os.fsync(artifact.fileno())
        os.replace(temporary_name, destination)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _load_records(path: Path) -> tuple[Any, ...] | None:
    """Load an entry's trace records, or None (recompile) if they are unreadable.

    That covers a missing or truncated file and a stale record, such as one naming a class that
    has moved.
    """
    try:
        records = pickle.loads(path.read_bytes())
    except (
        OSError,
        EOFError,
        ImportError,
        AttributeError,
        IndexError,
        TypeError,
        ValueError,
        pickle.UnpicklingError,
    ):
        return None
    return records if isinstance(records, tuple) else None


def _publish_compiled(compiled: Any, destination: Path) -> None:
    """Export a compiled callable and atomically publish its object file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".tmp.o", dir=destination.parent
    )
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        compiled.export_to_c(
            object_file_path=str(temporary_path),
            function_name=EXPORT_FUNCTION_NAME,
        )
        if temporary_path.stat().st_size == 0:
            raise RuntimeError("CuTeDSL exported an empty object file")
        with temporary_path.open("rb") as artifact:
            os.fsync(artifact.fileno())
        os.replace(temporary_path, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


_PLAIN_TYPES = frozenset({int, bool, str, bytes, type(None), torch.dtype, torch.device})


@functools.cache
def _target_key(target: CompileTarget) -> bytes:
    """Pickle a target once; it is fixed for the process but keys every launch."""
    return pickle.dumps(target, protocol=pickle.HIGHEST_PROTOCOL)


def _fast_key(item: Any) -> tuple[Any, ...] | None:
    """Key a plain static value without pickling, or return None to use the pickled key.

    Each value is tagged with its type, so values equal across types (``True == 1``) stay
    apart, as in the pickled key. Floats key on ``float.hex`` so ``0.0`` and ``-0.0`` differ;
    every NaN payload hexes to ``'nan'``, so NaN takes the pickled key.
    """
    item_type = type(item)
    # One set lookup, not a chain of isinstance patterns: this runs per value per launch.
    match item:
        case _ if item_type in _PLAIN_TYPES or isinstance(item, (type, enum.Enum)):
            return (item_type, item)
        case float() if not math.isnan(item):
            return (item_type, item.hex())
        case tuple() if item_type is tuple:  # NamedTuples keep their type via the pickled key.
            parts = tuple(map(_fast_key, item))
            return None if None in parts else (tuple, *parts)
        case _:
            return None


def _artifact_exists(path: Path) -> bool:
    try:
        return path.stat().st_size > 0
    except OSError:
        return False


def jit_cache(
    fn: Callable[P, T] | None = None,
    *,
    persistent: bool = True,
    lock_timeout: float = DEFAULT_LOCK_TIMEOUT_SECONDS,
    extra_sources: Iterable[str | os.PathLike[str]] = (),
    cache_key: Callable[P, Hashable] | None = None,
    uncached: Callable[P, bool] | None = None,
) -> Callable[P, T]:
    """Cache a ``cute.compile`` function in memory and optionally on disk.

    The arguments must be static values that fully determine the generated code. Their
    canonical encoding is the process-local key; source hashing and disk paths are computed
    only after a process-local miss. A disk entry is invalidated by edits to the function's
    module and every module it transitively imports from the package that ships warpdrive
    (``_key.module_closure``), not by unrelated modules.

    An argument with ``__cute_trace_record__()`` (a picklable record of what tracing left on
    it) and ``__cute_restore_trace__(record)`` gets the record of the build restored on every
    call served from memory or disk.

    Args:
        persistent: False keeps only the process-local cache.
        lock_timeout: Seconds to wait for another process's lock on an entry before compiling
            without the disk cache (``precompile`` raises ``CacheLockError`` instead).
        extra_sources: Inputs import analysis cannot see, such as files read while tracing or
            code outside the package.
        cache_key: Explicit structural key computed from the arguments, used instead of them.
        uncached: Calls for which it returns true compile without reading or filling either
            cache and count as misses. Use it when the caller needs the compile's trace-time
            side effects, such as a recorded protocol trace.
    """
    if fn is None:
        return functools.partial(
            jit_cache,
            persistent=persistent,
            lock_timeout=lock_timeout,
            extra_sources=extra_sources,
            cache_key=cache_key,
            uncached=uncached,
        )
    if lock_timeout <= 0:
        raise ValueError(f"lock_timeout must be positive, got {lock_timeout}")
    source_paths = tuple(os.fspath(Path(path).expanduser().resolve()) for path in extra_sources)
    # Values are (compiled, the build's trace records or None).
    memory_cache: dict[str, tuple[T, Any]] = {}
    runtime_cache: dict[Any, tuple[T, Any]] = {}
    static_cache: dict[Any, T] = {}
    key_locks: dict[str, threading.Lock] = {}
    state_lock = threading.RLock()
    cache_pid = os.getpid()
    hits = 0
    misses = 0

    def reset_after_fork() -> None:
        nonlocal cache_pid, hits, misses, state_lock
        current_pid = os.getpid()
        if current_pid == cache_pid:
            return
        state_lock = threading.RLock()
        memory_cache.clear()
        runtime_cache.clear()
        static_cache.clear()
        key_locks.clear()
        hits = 0
        misses = 0
        cache_pid = current_pid

    def remember(
        key: str,
        built: tuple[T, Any],
        *,
        runtime_key: Any,
        hit: bool,
        holders: list[Any],
    ) -> T:
        nonlocal hits, misses
        with state_lock:
            memory_cache[key] = built
            runtime_cache[runtime_key] = built
            if hit:
                hits += 1
            else:
                misses += 1
        if hit:
            _restore_traces(holders, built[1])
        return built[0]

    def compile_traced(
        args: tuple[Any, ...], kwargs: dict[str, Any], holders: list[Any]
    ) -> tuple[T, Any]:
        compiled = _compile(fn, args, kwargs)
        return compiled, _trace_records(holders)

    def compile_and_publish(
        entry: _CacheEntry,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        strict: bool,
    ) -> tuple[T, Any]:
        built = compile_traced(args, kwargs, _trace_holders(args, kwargs))
        try:
            if built[1] is not None:
                _publish_records(built[1], entry.trace_path)
            _publish_compiled(built[0], entry.object_path)
        except Exception:
            if strict:
                raise
            logger.warning(
                "Could not persist CuTeDSL cache artifact %s",
                entry.object_path,
                exc_info=True,
            )
        return built

    def entry_complete(entry: _CacheEntry, holders: list[Any]) -> bool:
        if not _artifact_exists(entry.object_path):
            return False
        return not holders or _artifact_exists(entry.trace_path)

    def load_from_disk(entry: _CacheEntry, holders: list[Any]) -> tuple[T, Any] | None:
        if not entry_complete(entry, holders):
            return None
        records = _load_records(entry.trace_path) if holders else None
        if holders and (records is None or len(records) != len(holders)):
            return None
        try:
            return _load_compiled(entry.object_path), records
        except (AttributeError, KeyError, OSError, RuntimeError, ValueError):
            return None

    def disk_cache_enabled() -> bool:
        return persistent and cache_enabled()

    def key_arguments(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if cache_key is None:
            return args, kwargs
        static_key = cache_key(*args, **kwargs)
        try:
            hash(static_key)
        except TypeError as error:
            raise TypeError("jit_cache cache_key must return a hashable value") from error
        return (static_key,), {}

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        nonlocal hits, misses
        reset_after_fork()
        if uncached is not None and uncached(*args, **kwargs):
            compiled = _compile(fn, args, kwargs)
            with state_lock:
                misses += 1
            return compiled
        target = get_compile_target()
        key_args, key_kwargs = key_arguments(args, kwargs)
        runtime_key = _fast_key((key_args, tuple(key_kwargs.items())))
        if runtime_key is None:
            runtime_key = _make_runtime_key(key_args, key_kwargs, target)
        else:
            runtime_key = (_target_key(target), runtime_key)
        holders = _trace_holders(args, kwargs)
        with state_lock:
            built = runtime_cache.get(runtime_key)
            if built is not None:
                hits += 1
        if built is not None:
            _restore_traces(holders, built[1])
            return built[0]

        entry = _cache_entry(fn, key_args, key_kwargs, source_paths, target)
        cached = {"runtime_key": runtime_key, "holders": holders}
        with state_lock:
            if entry.key in memory_cache:
                return remember(entry.key, memory_cache[entry.key], hit=True, **cached)
            key_lock = key_locks.setdefault(entry.key, threading.Lock())

        with key_lock:
            reset_after_fork()
            with state_lock:
                if entry.key in memory_cache:
                    return remember(entry.key, memory_cache[entry.key], hit=True, **cached)

            if not disk_cache_enabled():
                built = compile_traced(args, kwargs, holders)
                return remember(entry.key, built, hit=False, **cached)

            built = load_from_disk(entry, holders)
            if built is not None:
                return remember(entry.key, built, hit=True, **cached)

            try:
                with _FileLock(entry.lock_path, lock_timeout):
                    built = load_from_disk(entry, holders)
                    if built is not None:
                        return remember(entry.key, built, hit=True, **cached)
                    if entry.object_path.exists():
                        logger.warning(
                            "Replacing corrupt CuTeDSL cache artifact %s",
                            entry.object_path,
                        )
                        entry.object_path.unlink(missing_ok=True)
                    built = compile_and_publish(entry, args, kwargs, strict=False)
                    return remember(entry.key, built, hit=False, **cached)
            except CacheLockError:
                logger.warning(
                    "CuTeDSL cache lock unavailable for key %s; compiling without disk cache",
                    entry.key,
                    exc_info=True,
                )
                built = compile_traced(args, kwargs, holders)
                return remember(entry.key, built, hit=False, **cached)

    def by_static_key(static_key: Hashable, compile_call: Callable[[], T]) -> T:
        """Return the launch cached under a caller-computed ``static_key``.

        ``compile_call`` must call this function and runs only on a miss, so a warm launch skips
        building and hashing the full argument key. The caller guarantees that ``static_key``
        fully determines ``compile_call``'s arguments; the compile target is added here. A hit
        counts in ``cache_info`` like any other but restores no trace records, so use it only
        where the compiled arguments are the ones reused.
        """
        nonlocal hits
        reset_after_fork()
        key = (get_compile_target(), static_key)
        with state_lock:
            compiled = static_cache.get(key)
            if compiled is not None:
                hits += 1
                return compiled
        compiled = compile_call()
        with state_lock:
            static_cache[key] = compiled
        return compiled

    def by_args(*args: P.args) -> T:
        """``by_static_key`` keyed by the positional arguments themselves.

        Each value is tagged with its type (``True`` and ``1`` stay apart) and floats key on
        ``float.hex``. Values inside nested tuples key by equality, so each position must hold
        one type.
        """
        types = tuple(map(type, args))
        if float in types:
            key = tuple(arg.hex() if type(arg) is float else arg for arg in args)
        else:
            key = args
        return by_static_key((types, key), lambda: wrapper(*args))

    def precompile(*args: P.args, **kwargs: P.kwargs) -> None:
        """Populate one disk entry without loading or returning the artifact."""
        if not disk_cache_enabled():
            raise RuntimeError("parallel CuTeDSL precompilation requires the disk cache")
        key_args, key_kwargs = key_arguments(args, kwargs)
        entry = _cache_entry(fn, key_args, key_kwargs, source_paths, get_compile_target())
        holders = _trace_holders(args, kwargs)
        if entry_complete(entry, holders):
            return
        with _FileLock(entry.lock_path, lock_timeout):
            if entry_complete(entry, holders):
                return
            entry.object_path.unlink(missing_ok=True)
            compile_and_publish(entry, args, kwargs, strict=True)

    def is_cached(*args: P.args, **kwargs: P.kwargs) -> bool:
        """Whether this call has a nonempty disk entry, with trace records if it needs them."""
        key_args, key_kwargs = key_arguments(args, kwargs)
        entry = _cache_entry(fn, key_args, key_kwargs, source_paths, get_compile_target())
        return entry_complete(entry, _trace_holders(args, kwargs))

    def prepare_cache() -> None:
        """Compute the source fingerprint before compiler workers fork, so they inherit it."""
        _source_fingerprint(fn, source_paths)

    def cache_clear() -> None:
        """Clear this function's process-local entries and statistics."""
        nonlocal hits, misses
        reset_after_fork()
        with state_lock:
            memory_cache.clear()
            runtime_cache.clear()
            static_cache.clear()
            key_locks.clear()
            hits = 0
            misses = 0

    def cache_info() -> CacheInfo:
        """Return this function's process-local hit and miss counters."""
        reset_after_fork()
        with state_lock:
            return CacheInfo(hits=hits, misses=misses, currsize=len(memory_cache))

    def cache_namespace() -> str:
        """The source fingerprint naming the directory of this function's disk entries."""
        return _source_fingerprint(fn, source_paths)

    wrapper.cache_namespace = cache_namespace  # type: ignore[attr-defined]
    wrapper.by_static_key = by_static_key  # type: ignore[attr-defined]
    wrapper.by_args = by_args  # type: ignore[attr-defined]
    wrapper.precompile = precompile  # type: ignore[attr-defined]
    wrapper.is_cached = is_cached  # type: ignore[attr-defined]
    wrapper.disk_cache_enabled = disk_cache_enabled  # type: ignore[attr-defined]
    wrapper.prepare_cache = prepare_cache  # type: ignore[attr-defined]
    wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
    wrapper.cache_info = cache_info  # type: ignore[attr-defined]
    return wrapper
