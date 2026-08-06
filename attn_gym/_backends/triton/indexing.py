"""Pointer-indexing helpers shared by hand-written Triton kernels."""

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
