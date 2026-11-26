"""CUDA target metadata that forked compiler workers can inherit safely."""

from __future__ import annotations

import functools
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class CompileTarget:
    """Device facts that may affect generated code or launch policy."""

    device_type: str
    configured_arch: str | None = None
    capability: tuple[int, int] | None = None
    name: str | None = None
    sm_count: int | None = None

    @property
    def gpu_arch(self) -> str | None:
        if self.configured_arch is not None or self.capability is None:
            return self.configured_arch
        major, minor = self.capability
        return f"sm_{major}{minor}{'a' if major >= 9 else ''}"

    @property
    def effective_capability(self) -> tuple[int, int] | None:
        """The capability code is generated for: ``configured_arch`` if set, else the device's."""
        if self.configured_arch is None:
            return self.capability
        arch = self.configured_arch.removeprefix("sm_").removesuffix("a")
        if not arch.isdecimal() or len(arch) < 2:
            raise ValueError(f"invalid configured CUDA architecture {self.configured_arch!r}")
        return divmod(int(arch), 10)


_scoped_target: ContextVar[CompileTarget | None] = ContextVar("compile_target", default=None)
_target: CompileTarget | None = None
_target_lock = threading.Lock()
_target_lock_pid = os.getpid()


def _lock() -> threading.Lock:
    global _target_lock, _target_lock_pid
    current_pid = os.getpid()
    if current_pid != _target_lock_pid:
        _target_lock = threading.Lock()
        _target_lock_pid = current_pid
    return _target_lock


def set_compile_target(target: CompileTarget | None) -> None:
    """Set target metadata for this process and subsequently forked workers."""
    global _target
    with _lock():
        _target = target


@contextmanager
def compile_target_scope(target: CompileTarget) -> Iterator[None]:
    """Override ``get_compile_target()`` in this context without changing the process default."""
    token = _scoped_target.set(target)
    try:
        yield
    finally:
        _scoped_target.reset(token)


def _reject_bad_fork() -> None:
    """Raise in a forked CUDA child, even when the device query is already cached."""
    import torch

    is_bad_fork = getattr(torch.cuda, "_is_in_bad_fork", lambda: False)
    if is_bad_fork():
        raise RuntimeError(
            "cannot discover a CuTeDSL target in a forked CUDA child; pass a "
            "CompileTarget or use precompile_many(), which uses a fresh compiler process"
        )


@functools.lru_cache
def _query_compile_target(device_index: int, configured_arch: str | None) -> CompileTarget:
    """Query the driver once per device; hot launchers re-resolve every call."""
    import torch

    properties = torch.cuda.get_device_properties(device_index)
    return CompileTarget(
        device_type="cuda",
        configured_arch=configured_arch,
        capability=(properties.major, properties.minor),
        name=properties.name,
        sm_count=properties.multi_processor_count,
    )


def detect_compile_target(device: int | None = None) -> CompileTarget:
    """Query target metadata in a process allowed to use the CUDA driver."""
    import torch

    _reject_bad_fork()
    configured_arch = os.getenv("CUTE_DSL_ARCH")
    if not torch.cuda.is_available():
        return CompileTarget(device_type="none", configured_arch=configured_arch)
    device_index = torch.cuda.current_device() if device is None else device
    return _query_compile_target(device_index, configured_arch)


def get_compile_target() -> CompileTarget:
    """Return the scoped target, or the process default (detected on first use)."""
    global _target
    target = _scoped_target.get() or _target
    if target is not None:
        return target
    with _lock():
        if _target is None:
            _target = detect_compile_target()
        return _target
