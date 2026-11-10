"""Opinionated primitives for standalone CuTeDSL kernels."""

from __future__ import annotations

import operator
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache, lru_cache
from typing import Any, NamedTuple

import torch

TMA_ALIGNMENT_BYTES = 16
_CUTE_DTYPES = {
    torch.float16: ("Float16", "fp16"),
    torch.bfloat16: ("BFloat16", "bf16"),
    torch.float32: ("Float32", "fp32"),
}


class CuteDType(NamedTuple):
    """A Torch storage dtype's CuTeDSL scalar type and kernel-name tag."""

    cute_type: type
    name: str


@cache
def cute_dtype(dtype: torch.dtype | type) -> CuteDType:
    """Describe an fp16, bf16, or fp32 Torch or CuTeDSL dtype for kernel compilation."""
    import cutlass
    import cutlass.torch

    if not isinstance(dtype, torch.dtype):
        dtype = cutlass.torch.dtype(dtype)
    if dtype not in _CUTE_DTYPES:
        raise TypeError(f"no CuTeDSL storage type for {dtype}")
    type_name, name = _CUTE_DTYPES[dtype]
    return CuteDType(getattr(cutlass, type_name), name)


@lru_cache(maxsize=8)
def get_device_properties(device: torch.device | int) -> Any:
    """Return cached CUDA properties for a device or device ordinal."""
    return torch.cuda.get_device_properties(device)


@contextmanager
def initialized_cuda_device(tensor: torch.Tensor) -> Iterator[None]:
    """Make ``tensor``'s device current for a TVM-FFI launch and restore the caller's device."""
    previous_device = torch.cuda.current_device()
    torch.cuda.set_device(tensor.device)
    try:
        yield
    finally:
        torch.cuda.set_device(previous_device)


def requires_int64_abi(*tensors: torch.Tensor | None) -> bool:
    """Check both reachable cosize and every stride exposed through the CuTe ABI.

    Unlike cosize, TVM-FFI must represent a size-1 mode's unreachable stride.
    Bounded int32 routing arrays may be omitted by callers.
    """
    # Builtin reductions instead of generator expressions: this runs on every launch.
    for tensor in tensors:
        if tensor is None or not tensor.ndim:
            continue
        strides = tensor.stride()
        if max(strides) > 2**31 - 1 or min(strides) < 1 - 2**31:
            return True
        cosize = 1 + sum(map(operator.mul, tensor.shape, strides)) - sum(strides)
        if tensor.numel() and cosize > 2**31:
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


def validate_tma_tensor(
    name: str, tensor: torch.Tensor, *, alignment: int = TMA_ALIGNMENT_BYTES
) -> None:
    """Reject layouts that cannot satisfy the aligned TMA tensor contract."""
    if tensor.data_ptr() % alignment:
        raise ValueError(f"{name} data pointer must be {alignment}-byte aligned")
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} innermost stride must be one")
    element_bytes = tensor.element_size()
    for stride in tensor.stride()[:-1]:
        if stride < 0:
            raise ValueError(f"{name} outer strides must be nonnegative")
        if stride * element_bytes % TMA_ALIGNMENT_BYTES:
            raise ValueError(
                f"{name} outer byte strides must be multiples of {TMA_ALIGNMENT_BYTES}"
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


def normalize_tma_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Copy unless the tensor satisfies the aligned TMA contract.

    The copy is a ``clone``: ``.contiguous()`` returns a contiguous view unchanged even when its
    storage offset breaks the alignment.
    """
    if tensor_supports_tma(tensor):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


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


__all__ = [
    "TMA_ALIGNMENT_BYTES",
    "get_device_properties",
    "initialized_cuda_device",
    "normalize_compact_tensor",
    "normalize_tma_rows",
    "tensor_supports_contiguous_dim",
    "tensor_supports_tma",
    "tensor_supports_tma_rows",
    "validate_tma_tensor",
]
