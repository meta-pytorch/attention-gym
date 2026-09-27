# SPDX-License-Identifier: BSD-3-Clause

"""Attention Gym persistent-cache adapter for the vendored cudnn-frontend GDN builders.

The vendored builders compile over live placeholder tensors and key each compile by a static
tuple of dtypes, flags, and geometry. ``persistent_compile`` routes those ``cute.compile`` calls
through ``jit_cache`` with that tuple as the structural key, so compiled launches are reused
across processes. The whole vendored package is hashed as the source input because the builders
reach kernel modules through constexpr module arguments that import analysis cannot see.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from pathlib import Path
from typing import Any

from attn_gym._backends.cute import jit_cache

_PACKAGE_SOURCES = tuple(sorted(str(path) for path in Path(__file__).parent.rglob("*.py")))


def _static_key(name: str, key: Hashable, compile_fn: Callable, *args, **kwargs) -> Hashable:
    return name, key


@jit_cache(cache_key=_static_key, extra_sources=_PACKAGE_SOURCES)
def persistent_compile(name: str, key: Hashable, compile_fn: Callable, *args, **kwargs) -> Any:
    """Compile ``compile_fn(*args, **kwargs)`` once per ``(name, key)`` and compile target."""
    return compile_fn(*args, **kwargs)
