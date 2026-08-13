"""Persistent cache primitives for TVM-FFI CuTeDSL kernels.

``jit_cache`` decorates a module-level function that calls ``cute.compile``.
Its static arguments and ``CompileTarget`` determine the cache key. Cold
entries are compiled under a per-key process lock and atomically exported;
warm entries are loaded from the shared object file.

``ATTN_GYM_CUTE_CACHE_DIR`` is the only Attention Gym environment override.
Other behavior is expressed through ``jit_cache`` and tuning options. The
upstream ``CUTE_DSL_NO_CACHE`` switch is still respected.
"""

from __future__ import annotations

import ctypes
import errno
import functools
import logging
import os
import tempfile
import threading
import time
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from typing_extensions import Self

from ._key import make_key as _make_key
from ._key import make_runtime_key as _make_runtime_key
from ._key import source_fingerprint as _source_fingerprint
from .target import CompileTarget, get_compile_target

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


def get_cache_path() -> Path:
    """Return the root directory used for persistent CuTeDSL artifacts."""
    configured = os.getenv("ATTN_GYM_CUTE_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_cache = os.getenv("XDG_CACHE_HOME")
    cache_home = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return cache_home / "attention_gym" / "cute"


def cache_enabled() -> bool:
    """Return whether the upstream CuTeDSL no-cache switch is inactive."""
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
    """Serialize CuTeDSL compilation within one process."""
    global _compile_lock, _compile_lock_pid
    current_pid = os.getpid()
    if current_pid != _compile_lock_pid:
        _compile_lock = threading.Lock()
        _compile_lock_pid = current_pid
    with _compile_lock:
        return fn(*args, **kwargs)


def _ensure_runtime_libraries_global() -> None:
    """Preload runtime libraries for symbols referenced by cached objects."""
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
) -> Callable[P, T]:
    """Cache a ``cute.compile`` function in memory and optionally on disk.

    The function's arguments must be static values that completely determine
    generated code. Their canonical encoding provides the process-local key;
    ``cache_key`` may define an explicit structural key instead. Persistent
    hashing and path construction occur only after a process-local miss.
    ``persistent=False`` keeps only the process-local cache. ``extra_sources``
    explicitly adds downstream files or trees to source invalidation.
    """
    if fn is None:
        return functools.partial(
            jit_cache,
            persistent=persistent,
            lock_timeout=lock_timeout,
            extra_sources=extra_sources,
            cache_key=cache_key,
        )
    if lock_timeout <= 0:
        raise ValueError(f"lock_timeout must be positive, got {lock_timeout}")
    source_paths = tuple(os.fspath(Path(path).expanduser().resolve()) for path in extra_sources)
    memory_cache: dict[str, T] = {}
    runtime_cache: dict[bytes, T] = {}
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
        key_locks.clear()
        hits = 0
        misses = 0
        cache_pid = current_pid

    def remember(
        key: str,
        compiled: T,
        *,
        runtime_key: bytes,
        hit: bool,
    ) -> T:
        nonlocal hits, misses
        with state_lock:
            memory_cache[key] = compiled
            runtime_cache[runtime_key] = compiled
            if hit:
                hits += 1
            else:
                misses += 1
        return compiled

    def compile_and_publish(
        entry: _CacheEntry,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        strict: bool,
    ) -> T:
        compiled = _compile(fn, args, kwargs)
        try:
            _publish_compiled(compiled, entry.object_path)
        except Exception:
            if strict:
                raise
            logger.warning(
                "Could not persist CuTeDSL cache artifact %s",
                entry.object_path,
                exc_info=True,
            )
        return compiled

    def load_from_disk(entry: _CacheEntry) -> T | None:
        if not _artifact_exists(entry.object_path):
            return None
        try:
            return _load_compiled(entry.object_path)
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
        nonlocal hits
        reset_after_fork()
        target = get_compile_target()
        key_args, key_kwargs = key_arguments(args, kwargs)
        runtime_key = _make_runtime_key(key_args, key_kwargs, target)
        with state_lock:
            if runtime_key in runtime_cache:
                hits += 1
                return runtime_cache[runtime_key]

        entry = _cache_entry(fn, key_args, key_kwargs, source_paths, target)
        with state_lock:
            if entry.key in memory_cache:
                return remember(
                    entry.key,
                    memory_cache[entry.key],
                    runtime_key=runtime_key,
                    hit=True,
                )
            key_lock = key_locks.setdefault(entry.key, threading.Lock())

        with key_lock:
            reset_after_fork()
            with state_lock:
                if entry.key in memory_cache:
                    return remember(
                        entry.key,
                        memory_cache[entry.key],
                        runtime_key=runtime_key,
                        hit=True,
                    )

            if not disk_cache_enabled():
                return remember(
                    entry.key,
                    _compile(fn, args, kwargs),
                    runtime_key=runtime_key,
                    hit=False,
                )

            compiled = load_from_disk(entry)
            if compiled is not None:
                return remember(entry.key, compiled, runtime_key=runtime_key, hit=True)

            try:
                with _FileLock(entry.lock_path, lock_timeout):
                    compiled = load_from_disk(entry)
                    if compiled is not None:
                        return remember(entry.key, compiled, runtime_key=runtime_key, hit=True)
                    if entry.object_path.exists():
                        logger.warning(
                            "Replacing corrupt CuTeDSL cache artifact %s",
                            entry.object_path,
                        )
                        entry.object_path.unlink(missing_ok=True)
                    compiled = compile_and_publish(entry, args, kwargs, strict=False)
                    return remember(entry.key, compiled, runtime_key=runtime_key, hit=False)
            except CacheLockError:
                logger.warning(
                    "CuTeDSL cache lock unavailable for key %s; compiling without disk cache",
                    entry.key,
                    exc_info=True,
                )
                return remember(
                    entry.key,
                    _compile(fn, args, kwargs),
                    runtime_key=runtime_key,
                    hit=False,
                )

    def precompile(*args: P.args, **kwargs: P.kwargs) -> None:
        """Populate one disk entry without loading or returning the artifact."""
        if not disk_cache_enabled():
            raise RuntimeError("parallel CuTeDSL precompilation requires the disk cache")
        key_args, key_kwargs = key_arguments(args, kwargs)
        entry = _cache_entry(fn, key_args, key_kwargs, source_paths, get_compile_target())
        if _artifact_exists(entry.object_path):
            return
        with _FileLock(entry.lock_path, lock_timeout):
            if _artifact_exists(entry.object_path):
                return
            entry.object_path.unlink(missing_ok=True)
            compile_and_publish(entry, args, kwargs, strict=True)

    def is_cached(*args: P.args, **kwargs: P.kwargs) -> bool:
        """Return whether a nonempty disk artifact exists for this invocation."""
        key_args, key_kwargs = key_arguments(args, kwargs)
        entry = _cache_entry(fn, key_args, key_kwargs, source_paths, get_compile_target())
        return _artifact_exists(entry.object_path)

    def prepare_cache() -> None:
        """Warm source fingerprinting before compiler workers are forked."""
        _source_fingerprint(fn, source_paths)

    def cache_clear() -> None:
        """Clear this function's process-local entries and statistics."""
        nonlocal hits, misses
        reset_after_fork()
        with state_lock:
            memory_cache.clear()
            runtime_cache.clear()
            key_locks.clear()
            hits = 0
            misses = 0

    def cache_info() -> CacheInfo:
        """Return this function's process-local hit and miss counters."""
        reset_after_fork()
        with state_lock:
            return CacheInfo(hits=hits, misses=misses, currsize=len(memory_cache))

    wrapper.precompile = precompile  # type: ignore[attr-defined]
    wrapper.is_cached = is_cached  # type: ignore[attr-defined]
    wrapper.disk_cache_enabled = disk_cache_enabled  # type: ignore[attr-defined]
    wrapper.prepare_cache = prepare_cache  # type: ignore[attr-defined]
    wrapper.cache_clear = cache_clear  # type: ignore[attr-defined]
    wrapper.cache_info = cache_info  # type: ignore[attr-defined]
    return wrapper
