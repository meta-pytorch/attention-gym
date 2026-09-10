# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Modified by Attention Gym in 2026: the generic and 64-byte swizzles unused by the vendored
# kernels were removed.


import cutlass
from cutlass import cute


@cute.jit
def swizzle_xor_128b(row, col_elem, *, elem_bytes: cutlass.Constexpr[int] = 2):
    return col_elem ^ ((row & 7) * cutlass.const_expr(16 // elem_bytes))


@cute.jit
def swizzle_box_offset_128b(
    row,
    col,
    *,
    box_rows: cutlass.Constexpr[int],
    elem_bytes: cutlass.Constexpr[int] = 2,
):
    """Map a logical coordinate into segment-major 128-byte-swizzled storage."""
    box_cols = cutlass.const_expr(128 // elem_bytes)
    box = col // box_cols
    col_in_box = col - box * box_cols
    return (
        box * box_rows * box_cols
        + row * box_cols
        + swizzle_xor_128b(row, col_in_box, elem_bytes=elem_bytes)
    )


@cute.jit
def swizzle_xor_32b(row, col_elem, *, elem_bytes: cutlass.Constexpr[int] = 2):
    chunk_elems = 16 // elem_bytes
    chunk_idx = col_elem // chunk_elems
    in_chunk = col_elem % chunk_elems
    swz_chunk = chunk_idx ^ ((row >> 2) & 1)
    return swz_chunk * chunk_elems + in_chunk


@cute.jit
def swizzle_box_offset_32b(
    row,
    col,
    *,
    box_rows: cutlass.Constexpr[int],
    elem_bytes: cutlass.Constexpr[int] = 2,
):
    """Map a logical coordinate into segment-major 32-byte-swizzled storage."""
    box_cols = cutlass.const_expr(32 // elem_bytes)
    box = col // box_cols
    col_in_box = col - box * box_cols
    return (
        box * box_rows * box_cols
        + row * box_cols
        + swizzle_xor_32b(row, col_in_box, elem_bytes=elem_bytes)
    )


@cute.jit
def swizzle_lin_128b(
    lin, *, row_stride_log2: cutlass.Constexpr[int], elem_bytes: cutlass.Constexpr[int] = 2
):
    chunk_log2 = cutlass.const_expr((16 // elem_bytes).bit_length() - 1)
    shift = cutlass.const_expr(row_stride_log2 - chunk_log2)
    mask = cutlass.const_expr(0x7 << chunk_log2)
    return lin ^ ((lin >> shift) & mask)
