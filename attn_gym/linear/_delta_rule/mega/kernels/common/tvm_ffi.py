# SPDX-License-Identifier: BSD-3-Clause

"""Fake-tensor signature builders for the Mega TVM-FFI launchers."""

from typing import Any

import cutlass
from cutlass.cute.runtime import make_fake_compact_tensor

from attn_gym._backends.cute import make_fake_strided_tensor

WORK_ITEM_FIELDS = 8


def make_compact_signature_tensor(
    dtype: Any,
    shape: tuple[Any, ...],
    *,
    assumed_align: int,
):
    """Create a row-major compact signature tensor."""
    return make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=assumed_align,
    )


def make_strided_signature_tensor(
    dtype: Any,
    shape: tuple[Any, ...],
    *,
    assumed_align: int,
    use_int64_offsets: bool,
):
    """Create a last-dimension-contiguous tensor with aligned dynamic outer strides."""
    element_bytes = dtype.width // 8
    if assumed_align % element_bytes:
        raise ValueError("assumed alignment must be a multiple of the element width")
    return make_fake_strided_tensor(
        dtype,
        shape,
        stride_divisibility=assumed_align // element_bytes,
        assumed_align=assumed_align,
        use_int64_strides=use_int64_offsets,
    )


def make_cu_seqlens_signature(entries: Any, *, assumed_align: int = 8):
    """Create the compact int32 cumulative-sequence-length signature."""
    return make_compact_signature_tensor(
        cutlass.Int32,
        (entries,),
        assumed_align=assumed_align,
    )


def make_work_items_signature(rows: Any):
    """Create a compact work-item table signature."""
    return make_compact_signature_tensor(
        cutlass.Int32,
        (rows, WORK_ITEM_FIELDS),
        assumed_align=4,
    )


def make_counter_signature(entries: Any = 1):
    """Create a compact int32 work-count or scheduler-counter signature."""
    return make_compact_signature_tensor(
        cutlass.Int32,
        (entries,),
        assumed_align=4,
    )


def make_workspace_signature(words: Any):
    """Create the aligned int64 tensor-map workspace signature."""
    return make_compact_signature_tensor(
        cutlass.Int64,
        (words,),
        assumed_align=128,
    )


def validate_cu_seqlens(cu_seqlens: Any, *, assumed_align: int) -> None:
    """Validate the cumulative-sequence-length ABI before cache selection."""
    if str(cu_seqlens.dtype) != "torch.int32":
        raise ValueError(f"cu_seqlens must have dtype torch.int32, got {cu_seqlens.dtype}")
    if cu_seqlens.data_ptr() % assumed_align:
        raise ValueError(f"cu_seqlens data pointer must be {assumed_align}-byte aligned")
