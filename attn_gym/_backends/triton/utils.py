"""Small utilities shared by hand-written Triton kernels."""

from collections.abc import Sequence

import torch
import triton
import triton.language as tl


@triton.jit
def ptr_offset(indices, strides: tl.constexpr):
    """Compute a broadcasted signed 64-bit offset from index and stride tuples."""
    tl.static_assert(len(indices) == len(strides), "indices and strides must have equal length")
    offset = 0
    for axis in tl.static_range(len(strides)):
        offset += tl.cast(indices[axis], tl.int64) * strides[axis]
    return offset


def storage_cosize(shape: Sequence[int], strides: Sequence[int]) -> int:
    """Return the storage extent of a nonnegative-strided logical layout.

    Analogous to CuTe's ``cosize``: a layout with a zero-sized dimension has
    extent zero; otherwise the result is one plus the largest offset from the
    logical origin.
    """
    if len(shape) != len(strides):
        raise ValueError("shape and strides must have equal length")

    cosize = 1
    is_empty = False
    for size, stride in zip(shape, strides):
        if size < 0:
            raise ValueError("shape dimensions must be nonnegative")
        if stride < 0:
            raise ValueError("strides must be nonnegative")
        is_empty |= size == 0
        if size > 0:
            cosize += (size - 1) * stride
    return 0 if is_empty else cosize


def can_use_tma(tensor: torch.Tensor) -> bool:
    """Return whether a tensor can be accessed with a TMA tensor descriptor."""
    if not tensor.is_cuda or torch.cuda.get_device_capability(tensor.device)[0] < 9:
        return False

    element_size = tensor.element_size()
    storage_is_aligned = (
        torch.compiler.is_compiling() or tensor.storage_offset() * element_size % 16 == 0
    )
    return (
        tensor.shape[-1] <= 256
        and tensor.stride(-1) == 1
        and storage_is_aligned
        and all(stride * element_size % 16 == 0 for stride in tensor.stride()[:-1])
    )
