"""Opinionated primitives for standalone CuTeDSL kernels."""

from __future__ import annotations

import re
from typing import Any

_VALID_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


def ceildiv(number: int, divisor: int) -> int:
    """Return ``ceil(number / divisor)`` using integer arithmetic."""
    return -(number // -divisor)


def _contains_torch_tensor(value: Any) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_torch_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_torch_tensor(key) or _contains_torch_tensor(item)
            for key, item in value.items()
        )
    return False


def compile_tvm_ffi(
    entrypoint: Any,
    *compile_args: Any,
    name: str | None = None,
) -> Any:
    """Compile a fake-tensor signature with the canonical TVM-FFI stream ABI.

    ``compile_args`` describe the callable's runtime signature and must omit the
    stream. Tensor arguments should be fake CuTe tensors, never runtime Torch
    tensors. This helper appends a fake environment stream, enables TVM-FFI with
    a typed compile option, and gives the outer artifact a stable name.

    Class-based entrypoints should expose ``get_name()``. Free-function
    entrypoints may instead provide ``name=`` explicitly.
    """
    if name is None:
        get_name = getattr(entrypoint, "get_name", None)
        if not callable(get_name):
            raise TypeError("compile_tvm_ffi() requires entrypoint.get_name() or name=")
        name = get_name()
    if not isinstance(name, str) or _VALID_NAME.fullmatch(name) is None:
        raise ValueError(
            "CuTeDSL compile names must start with a lowercase letter and contain "
            f"only lowercase letters, digits, and underscores; got {name!r}"
        )
    if any(_contains_torch_tensor(arg) for arg in compile_args):
        raise TypeError("compile_tvm_ffi() accepts fake CuTe tensors, not runtime Torch tensors")

    from cutlass import cute

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile[cute.EnableTVMFFI](
        entrypoint,
        *compile_args,
        stream,
        _name_prefix=name,
    )


__all__ = ["ceildiv", "compile_tvm_ffi"]
