"""Pointwise validation for Mega shared-memory layout helpers."""

from __future__ import annotations

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
cute = pytest.importorskip("cutlass.cute")

from attn_gym._backends.cute import compile_tvm_ffi
from attn_gym._backends.cute.cache import jit_cache
from attn_gym.linear._delta_rule.mega.kernels.tile_dsl.swizzle import (
    swizzle_box_offset_128b,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the Mega SMEM layout test requires SM100 or SM103",
)

_ROWS = 16
_COLS = 128
_ELEMENTS = _ROWS * _COLS
_THREADS = 256


@cute.kernel
def write_swizzled_offsets(output: cute.Tensor) -> None:
    """Write the physical offset for every logical coordinate in a 16x128 tile."""
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx = cute.arch.block_idx()[0]
    index = block_idx * _THREADS + thread_idx
    if index < _ELEMENTS:
        row = index // _COLS
        col = index % _COLS
        output[index] = swizzle_box_offset_128b(row, col, box_rows=_ROWS)


@cute.jit
def launch_swizzled_offsets(output: cute.Tensor, stream) -> None:
    """Launch the pointwise layout-map test kernel."""
    write_swizzled_offsets(output).launch(
        grid=((_ELEMENTS + _THREADS - 1) // _THREADS, 1, 1),
        block=(_THREADS, 1, 1),
        stream=stream,
    )


@jit_cache
def compile_swizzled_offsets():
    """Compile the fixed 16x128 offset-map test."""
    output = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (_ELEMENTS,),
        stride_order=(0,),
        assumed_align=16,
    )
    return compile_tvm_ffi(
        launch_swizzled_offsets,
        output,
        name="mega_swizzle_box_offset_128b_test",
    )


def reference_offset(row: int, col: int) -> int:
    """Evaluate the original segment-major SW128 address expression."""
    segment = col // 64
    col_in_segment = col - segment * 64
    return segment * (_ROWS * 64) + row * 64 + (col_in_segment ^ ((row & 7) * 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_swizzle_box_offset_128b_matches_original_layout() -> None:
    output = torch.empty(_ELEMENTS, dtype=torch.int32, device="cuda")
    compile_swizzled_offsets()(output)

    expected = torch.tensor(
        [reference_offset(row, col) for row in range(_ROWS) for col in range(_COLS)],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert torch.equal(output.sort().values, torch.arange(_ELEMENTS, device="cuda"))

    lane_starts = []
    for lane in range(32):
        lhs_row = lane % 8 + (8 if (lane // 8) % 2 else 0)
        lhs_col_offset = 8 if lane // 8 >= 2 else 0
        rhs_row = lane % 8 + (8 if lane // 16 else 0)
        rhs_col_offset = 8 if (lane // 8) % 2 else 0
        for k_block in range(8):
            lane_starts.extend(
                (
                    reference_offset(lhs_row, k_block * 16 + lhs_col_offset),
                    reference_offset(rhs_row, k_block * 16 + rhs_col_offset),
                )
            )
    assert all(offset % 8 == 0 for offset in lane_starts)
