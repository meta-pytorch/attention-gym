"""Small utilities shared by hand-written Triton kernels."""

import torch
import triton
import triton.language as tl


@triton.jit
def ptr_offset(indices, strides: tl.constexpr):
    """Compute a broadcasted linear offset from index and stride tuples."""
    tl.static_assert(len(indices) == len(strides), "indices and strides must have equal length")
    offset = 0
    for axis in tl.static_range(len(strides)):
        offset += indices[axis] * strides[axis]
    return offset


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
