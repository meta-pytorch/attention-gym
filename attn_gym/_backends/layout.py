"""Strided-layout arithmetic shared by the Triton and CuTe backends.

Imports only the standard library: either backend imports it without pulling in the other's
toolchain (the reference APIs run with neither Triton nor CuTeDSL installed).
"""

from __future__ import annotations

from collections.abc import Sequence


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


def layout_overlaps(shape: Sequence[int], strides: Sequence[int]) -> bool:
    """Whether two indices of a nonnegative-strided layout address one element.

    Ordered by stride, a layout is injective when each dimension of size > 1 steps past the
    ``storage_cosize`` of the dimensions inside it. This is sufficient, not necessary: an
    interleaving such as strides ``(2, 3)`` over sizes ``(3, 2)`` is injective but reported as
    overlapping. Writes through views of one buffer need no more than this.
    """
    inner_shape: list[int] = []
    inner_strides: list[int] = []
    for stride, size in sorted(zip(strides, shape)):
        if size > 1 and stride < storage_cosize(inner_shape, inner_strides):
            return True
        inner_shape.append(size)
        inner_strides.append(stride)
    return False
