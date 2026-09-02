"""Opinionated primitives for standalone CuTeDSL kernels."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import torch

_VALID_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
TMA_ALIGNMENT_BYTES = 16


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
    """Check both reachable cosize and every stride exposed through the CuTe ABI.

    Unlike cosize, TVM-FFI must represent a size-1 mode's unreachable stride.
    Bounded int32 routing arrays may be omitted by callers.
    """
    for tensor in tensors:
        if tensor is None:
            continue
        strides = tensor.stride()
        if any(abs(stride) > 2**31 - 1 for stride in strides):
            return True
        if (
            tensor.numel()
            and 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, strides)) > 2**31
        ):
            return True
    return False


def tensor_supports_contiguous_dim(
    tensor: torch.Tensor,
    *,
    dim: int = -1,
    alignment_bytes: int = 1,
) -> bool:
    """Return whether one tensor mode is contiguous with aligned slice origins.

    ``alignment_bytes`` applies to the base pointer and every other mode's stride in
    bytes, so every slice along ``dim`` starts at that alignment. Use element-size
    alignment for the general scalar/gather path and a vector width such as 16 for a
    vectorized specialization.
    """
    if tensor.ndim == 0 or not -tensor.ndim <= dim < tensor.ndim or not 1 <= alignment_bytes:
        return False
    dim %= tensor.ndim
    element_size = tensor.element_size()
    return (
        tensor.stride(dim) == 1
        and tensor.data_ptr() % alignment_bytes == 0
        and all(
            index == dim or stride * element_size % alignment_bytes == 0
            for index, stride in enumerate(tensor.stride())
        )
    )


def make_fake_strided_tensor(
    dtype: Any,
    shape: tuple[Any, ...],
    *,
    contiguous_dim: int = -1,
    stride_divisibility: int = 1,
    assumed_align: int | None = None,
    use_int64_strides: bool = True,
) -> Any:
    """Create a fake tensor with one contiguous mode and dynamic other strides.

    ``stride_divisibility`` is measured in elements. When ``assumed_align`` is omitted,
    it is derived from that divisibility and the element width, matching the weakest
    alignment promised by the dynamic stride layout.
    """
    if not shape:
        raise ValueError("make_fake_strided_tensor requires at least one dimension")
    if not -len(shape) <= contiguous_dim < len(shape):
        raise ValueError(f"contiguous_dim is out of range for rank {len(shape)}")
    if not 1 <= stride_divisibility:
        raise ValueError("stride_divisibility must be positive")
    if assumed_align is not None and assumed_align < 1:
        raise ValueError("assumed_align must be positive")
    contiguous_dim %= len(shape)
    from cutlass import cute

    sym_int = cute.sym_int64 if use_int64_strides else cute.sym_int
    strides = tuple(
        1 if index == contiguous_dim else sym_int(divisibility=stride_divisibility)
        for index in range(len(shape))
    )
    if assumed_align is None:
        alignment_bits = stride_divisibility * dtype.width
        if alignment_bits % 8:
            raise ValueError("sub-byte fake tensors require an explicit assumed_align")
        assumed_align = max(1, alignment_bits // 8)
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=strides,
        assumed_align=assumed_align,
    )


def tensor_supports_tma(tensor: torch.Tensor) -> bool:
    """Return whether a CUDA tensor has a TMA-compatible aligned row layout."""
    return tensor.is_cuda and tensor_supports_contiguous_dim(
        tensor,
        alignment_bytes=TMA_ALIGNMENT_BYTES,
    )


def tensor_supports_tma_rows(tensor: torch.Tensor) -> bool:
    """Return whether the last two modes form aligned, compact rows for TMA kernels."""
    return (
        tensor.ndim >= 2 and tensor.stride(-2) == tensor.shape[-1] and tensor_supports_tma(tensor)
    )


def normalize_tma_rows(tensor: torch.Tensor) -> torch.Tensor:
    """Copy unless the tensor satisfies the aligned, compact-row TMA contract."""
    if tensor_supports_tma_rows(tensor):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def normalize_compact_tensor(
    tensor: torch.Tensor,
    *,
    alignment_bytes: int = 128,
) -> torch.Tensor:
    """Copy unless the full tensor is compact with aligned slice origins."""
    if tensor.is_contiguous() and tensor_supports_contiguous_dim(
        tensor,
        alignment_bytes=alignment_bytes,
    ):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


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
    "compile_tvm_ffi",
    "get_device_properties",
    "make_fake_strided_tensor",
    "normalize_compact_tensor",
    "normalize_tma_rows",
    "tensor_supports_contiguous_dim",
    "tensor_supports_tma",
    "tensor_supports_tma_rows",
]
