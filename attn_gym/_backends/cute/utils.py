"""Opinionated primitives for standalone CuTeDSL kernels."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import torch

from attn_gym._backends.triton.utils import requires_int64_offsets

_VALID_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
TMA_ALIGNMENT_BYTES = 16


def ceildiv(number: int, divisor: int) -> int:
    """Return ``ceil(number / divisor)`` using integer arithmetic."""
    return -(number // -divisor)


def _contains_torch_tensor(value: Any) -> bool:
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


@lru_cache(maxsize=8)
def get_device_properties(device: torch.device) -> Any:
    """Return cached CUDA properties for a device."""
    return torch.cuda.get_device_properties(device)


def requires_int64_abi(*tensors: torch.Tensor | None) -> bool:
    """Return whether any tensor needs the Int64 fake-signature specialization.

    TVM-FFI must represent every declared runtime stride, including the outer
    stride of a size-1 batch mode whose coordinate never leaves zero, so this
    is stricter than the reachable-offset bound alone. Callers may omit the
    contiguous ``[num_sequences + 1]`` int32 routing arrays: their extents are
    bounded by ``MAX_NUM_SEQUENCES`` and cannot approach either limit.
    """
    return requires_int64_offsets(*tensors) or any(
        tensor is not None and any(abs(stride) > 2**31 - 1 for stride in tensor.stride())
        for tensor in tensors
    )


def tensor_supports_tma(tensor: torch.Tensor) -> bool:
    """Return whether a CUDA tensor has a TMA-compatible aligned strided layout.

    The innermost dimension must be contiguous. The base pointer and every outer stride,
    measured in bytes, must satisfy ``TMA_ALIGNMENT_BYTES``.
    """
    if not tensor.is_cuda:
        return False
    element_size = tensor.element_size()
    return (
        tensor.ndim > 0
        and tensor.stride(-1) == 1
        and tensor.data_ptr() % TMA_ALIGNMENT_BYTES == 0
        and all(
            stride * element_size % TMA_ALIGNMENT_BYTES == 0 for stride in tensor.stride()[:-1]
        )
    )


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

    # Requires CuTeDSL >= 4.5.0 for `set_name_prefix` (`_name_prefix=` was removed in 4.6.2).
    # Class-based entrypoints hold the jit wrapper on their `__call__`.
    jit_wrapper = entrypoint if hasattr(entrypoint, "set_name_prefix") else entrypoint.__call__
    jit_wrapper.set_name_prefix(name)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile[cute.EnableTVMFFI](
        entrypoint,
        *compile_args,
        stream,
    )


__all__ = [
    "TMA_ALIGNMENT_BYTES",
    "ceildiv",
    "compile_tvm_ffi",
    "get_device_properties",
    "tensor_supports_tma",
]
