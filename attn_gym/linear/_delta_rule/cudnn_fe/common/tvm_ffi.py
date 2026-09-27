# SPDX-License-Identifier: BSD-3-Clause

"""Fake-tensor signature builders for the cuDNN TVM-FFI launchers."""

from typing import Any

import cutlass
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor

from attn_gym._backends.cute import make_fake_strided_tensor

# v1.30 adds seed and final-state destinations to the work-item ABI.
WORK_ITEM_FIELDS = 10


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
    stride_divisibility: int | None = None,
):
    """Create a last-dimension-contiguous tensor with dynamic outer strides.

    Use ``stride_divisibility=1`` to retain the upstream dynamic-layout ABI;
    by default every outer row promises the same alignment as the base pointer.
    """
    element_bytes = dtype.width // 8
    if assumed_align % element_bytes:
        raise ValueError("assumed alignment must be a multiple of the element width")
    return make_fake_strided_tensor(
        dtype,
        shape,
        stride_divisibility=(
            assumed_align // element_bytes if stride_divisibility is None else stride_divisibility
        ),
        assumed_align=assumed_align,
        use_int64_strides=use_int64_offsets,
    )


def make_dynamic_signature_tensor(
    dtype: Any,
    rank: int,
    *,
    assumed_align: int,
    use_int64_offsets: bool = False,
):
    """Match upstream ``mark_layout_dynamic(leading_dim=rank - 1)`` exactly.

    Legacy outer strides are always int64 with no divisibility promise. The
    optional wide ABI also widens shapes; the default retains int32 shapes.
    """
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    return make_strided_signature_tensor(
        dtype,
        tuple(sym_int() for _ in range(rank)),
        assumed_align=assumed_align,
        use_int64_offsets=True,
        stride_divisibility=1,
    )


def make_cu_seqlens_signature(entries: Any, *, assumed_align: int = 8):
    """Create the compact int32 cumulative-sequence-length signature."""
    return make_compact_signature_tensor(
        cutlass.Int32,
        (entries,),
        assumed_align=assumed_align,
    )


def make_paged_route_signatures(sequences: Any, *, has_initial_state: bool):
    """Create the shared route and optional seed-mask signatures."""
    indices = make_compact_signature_tensor(cutlass.Int32, (sequences,), assumed_align=4)
    mask = (
        make_compact_signature_tensor(cutlass.Uint8, (sequences,), assumed_align=1)
        if has_initial_state
        else None
    )
    return indices, mask


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
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 1 or not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be a nonempty compact vector")
    if str(cu_seqlens.dtype) != "torch.int32":
        raise ValueError(f"cu_seqlens must have dtype torch.int32, got {cu_seqlens.dtype}")
    if cu_seqlens.data_ptr() % assumed_align:
        raise ValueError(f"cu_seqlens data pointer must be {assumed_align}-byte aligned")
