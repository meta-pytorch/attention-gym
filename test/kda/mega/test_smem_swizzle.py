"""Pointwise validation for Mega shared-memory layout helpers."""

from __future__ import annotations

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")

from attn_gym._backends.cute import compile_tvm_ffi
from attn_gym._backends.cute.cache import jit_cache
from attn_gym.linear._delta_rule.mega.kernels.tile_dsl.swizzle import (
    swizzle_box_offset_32b,
    swizzle_box_offset_128b,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the Mega SMEM layout test requires SM100 or SM103",
)

_SW128_ROWS = 16
_SW128_COLS = 128
_SW128_ELEMENTS = _SW128_ROWS * _SW128_COLS
_SW128_STATE_ROWS = 128
_SW128_STATE_COLS = 128
_SW128_STATE_ELEMENTS = _SW128_STATE_ROWS * _SW128_STATE_COLS
_SW32_ROWS = 16
_SW32_COLS = 16
_SW32_ELEMENTS = _SW32_ROWS * _SW32_COLS
_THREADS = 256
_MAX_ELEMENTS = max(_SW128_ELEMENTS, _SW128_STATE_ELEMENTS, _SW32_ELEMENTS)


@cute.kernel
def write_swizzled_offsets(
    output_128b_f16: cute.Tensor,
    output_128b_f32: cute.Tensor,
    output_128b_state: cute.Tensor,
    output_32b: cute.Tensor,
) -> None:
    """Write every logical coordinate's physical offset for each live layout."""
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx = cute.arch.block_idx()[0]
    index = block_idx * _THREADS + thread_idx
    if index < _SW128_ELEMENTS:
        row = index // _SW128_COLS
        col = index % _SW128_COLS
        output_128b_f16[index] = swizzle_box_offset_128b(
            row,
            col,
            box_rows=_SW128_ROWS,
        )
        output_128b_f32[index] = swizzle_box_offset_128b(
            row,
            col,
            box_rows=_SW128_ROWS,
            elem_bytes=4,
        )
    if index < _SW128_STATE_ELEMENTS:
        row = index // _SW128_STATE_COLS
        col = index % _SW128_STATE_COLS
        output_128b_state[index] = swizzle_box_offset_128b(
            row,
            col,
            box_rows=_SW128_STATE_ROWS,
        )
    if index < _SW32_ELEMENTS:
        row = index // _SW32_COLS
        col = index % _SW32_COLS
        output_32b[index] = swizzle_box_offset_32b(row, col, box_rows=_SW32_ROWS)


@cute.jit
def launch_swizzled_offsets(
    output_128b_f16: cute.Tensor,
    output_128b_f32: cute.Tensor,
    output_128b_state: cute.Tensor,
    output_32b: cute.Tensor,
    stream,
) -> None:
    """Launch the pointwise layout-map test kernel."""
    write_swizzled_offsets(
        output_128b_f16,
        output_128b_f32,
        output_128b_state,
        output_32b,
    ).launch(
        grid=((_MAX_ELEMENTS + _THREADS - 1) // _THREADS, 1, 1),
        block=(_THREADS, 1, 1),
        stream=stream,
    )


def fake_output(elements: int):
    """Create one compact int32 TVM-FFI output signature."""
    return cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (elements,),
        stride_order=(0,),
        assumed_align=16,
    )


@jit_cache
def compile_swizzled_offsets():
    """Compile the fixed SW128 and SW32 offset-map test."""
    return compile_tvm_ffi(
        launch_swizzled_offsets,
        fake_output(_SW128_ELEMENTS),
        fake_output(_SW128_ELEMENTS),
        fake_output(_SW128_STATE_ELEMENTS),
        fake_output(_SW32_ELEMENTS),
        name="mega_swizzle_box_offset_test",
    )


def reference_offset_128b(
    row: int,
    col: int,
    *,
    box_rows: int,
    elem_bytes: int = 2,
) -> int:
    """Evaluate the original segment-major SW128 address expression."""
    box_cols = 128 // elem_bytes
    box = col // box_cols
    col_in_box = col - box * box_cols
    return (
        box * box_rows * box_cols
        + row * box_cols
        + (col_in_box ^ ((row & 7) * (16 // elem_bytes)))
    )


def reference_offset_32b(row: int, col: int) -> int:
    """Evaluate the original S<1,3,3> 16x16 address expression."""
    return row * _SW32_COLS + (col ^ (((row >> 2) & 1) * 8))


def assert_bijective_offsets(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Require pointwise equality and dense one-to-one physical coverage."""
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.sort().values, torch.arange(actual.numel(), device="cuda"))


def test_swizzle_box_offsets_match_original_layouts() -> None:
    output_128b_f16 = torch.empty(_SW128_ELEMENTS, dtype=torch.int32, device="cuda")
    output_128b_f32 = torch.empty(_SW128_ELEMENTS, dtype=torch.int32, device="cuda")
    output_128b_state = torch.empty(_SW128_STATE_ELEMENTS, dtype=torch.int32, device="cuda")
    output_32b = torch.empty(_SW32_ELEMENTS, dtype=torch.int32, device="cuda")
    compile_swizzled_offsets()(
        output_128b_f16,
        output_128b_f32,
        output_128b_state,
        output_32b,
    )

    expected_128b_f16 = torch.tensor(
        [
            reference_offset_128b(row, col, box_rows=_SW128_ROWS)
            for row in range(_SW128_ROWS)
            for col in range(_SW128_COLS)
        ],
        dtype=torch.int32,
        device="cuda",
    )
    expected_128b_f32 = torch.tensor(
        [
            reference_offset_128b(row, col, box_rows=_SW128_ROWS, elem_bytes=4)
            for row in range(_SW128_ROWS)
            for col in range(_SW128_COLS)
        ],
        dtype=torch.int32,
        device="cuda",
    )
    expected_128b_state = torch.tensor(
        [
            reference_offset_128b(row, col, box_rows=_SW128_STATE_ROWS)
            for row in range(_SW128_STATE_ROWS)
            for col in range(_SW128_STATE_COLS)
        ],
        dtype=torch.int32,
        device="cuda",
    )
    expected_32b = torch.tensor(
        [reference_offset_32b(row, col) for row in range(_SW32_ROWS) for col in range(_SW32_COLS)],
        dtype=torch.int32,
        device="cuda",
    )
    assert_bijective_offsets(output_128b_f16, expected_128b_f16)
    assert_bijective_offsets(output_128b_f32, expected_128b_f32)
    assert_bijective_offsets(output_128b_state, expected_128b_state)
    assert_bijective_offsets(output_32b, expected_32b)

    lane_starts = []
    for lane in range(32):
        lhs_row = lane % 8 + (8 if (lane // 8) % 2 else 0)
        lhs_col_offset = 8 if lane // 8 >= 2 else 0
        rhs_row = lane % 8 + (8 if lane // 16 else 0)
        rhs_col_offset = 8 if (lane // 8) % 2 else 0
        for k_block in range(8):
            lane_starts.extend(
                (
                    reference_offset_128b(
                        lhs_row,
                        k_block * 16 + lhs_col_offset,
                        box_rows=_SW128_ROWS,
                    ),
                    reference_offset_128b(
                        rhs_row,
                        k_block * 16 + rhs_col_offset,
                        box_rows=_SW128_ROWS,
                    ),
                )
            )
    assert all(offset % 8 == 0 for offset in lane_starts)
