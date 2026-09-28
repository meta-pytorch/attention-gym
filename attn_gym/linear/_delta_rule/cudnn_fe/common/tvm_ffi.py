# SPDX-License-Identifier: BSD-3-Clause

"""Fake-tensor signature builders for the cuDNN TVM-FFI launchers."""

from typing import Any

import cutlass
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor

from attn_gym._backends.cute import make_fake_strided_tensor

# v1.30 adds seed and final-state destinations to the work-item ABI.
WORK_ITEM_FIELDS = 10

SignatureSpec = tuple[str, str, int | tuple[int, ...], int] | tuple[str, str, int, int, int]
_SIGNATURE_DTYPES = {
    "bfloat16": cutlass.BFloat16,
    "float16": cutlass.Float16,
    "float32": cutlass.Float32,
    "int32": cutlass.Int32,
    "int64": cutlass.Int64,
    "uint8": cutlass.Uint8,
}


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


def signature_spec(
    tensor: Any,
    *,
    assumed_align: int,
    compact: bool = False,
    mode3_divisibility: int = 0,
) -> SignatureSpec | None:
    """Project a live placeholder to a static, pickleable description of its legacy ABI.

    Fully dynamic layouts retain only rank; compact tables retain only their
    static trailing dimensions. Token counts and dynamic strides never enter a key.
    """
    if tensor is None:
        return None
    dtype = str(tensor.dtype).removeprefix("torch.")
    if dtype not in _SIGNATURE_DTYPES:
        raise ValueError(f"unsupported signature dtype {dtype}")
    if compact:
        if mode3_divisibility:
            raise ValueError("compact and mode3 layouts are mutually exclusive")
        return ("compact", dtype, tuple(int(dim) for dim in tensor.shape[1:]), assumed_align)
    if mode3_divisibility:
        if len(tensor.shape) != 4 or mode3_divisibility < 1:
            raise ValueError("mode3 signatures require rank 4 and positive divisibility")
        return ("mode3", dtype, 4, assumed_align, mode3_divisibility)
    return ("dynamic", dtype, len(tensor.shape), assumed_align)


def signature_key(dynamic: tuple = (), compact: tuple = ()) -> tuple:
    """Hashable projection of every tensor fact ``signature_spec`` reads: dtype and rank of
    ``dynamic`` placeholders, dtype and static trailing extents of ``compact`` tables. With a
    builder's static flags it keys the specs without building them; passing a tensor the specs
    omit only makes the key more specific."""
    return (
        tuple(None if t is None else (t.dtype, t.dim()) for t in dynamic),
        tuple(None if t is None else (t.dtype, t.shape[1:]) for t in compact),
    )


def make_signature(spec: SignatureSpec | None, *, use_int64_offsets: bool = False):
    """Materialize a legacy fake signature inside a module-level cached compiler."""
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    match spec:
        case None:
            return None
        case ("compact", dtype, tail, align):
            return make_compact_signature_tensor(
                _SIGNATURE_DTYPES[dtype],
                (sym_int(), *tail),
                assumed_align=align,
            )
        case ("dynamic", dtype, rank, align):
            return make_dynamic_signature_tensor(
                _SIGNATURE_DTYPES[dtype],
                rank,
                assumed_align=align,
                use_int64_offsets=use_int64_offsets,
            )
        case ("mode3", dtype, rank, align, divisibility):
            shape = tuple(sym_int() for _ in range(rank - 1)) + (
                sym_int(divisibility=divisibility),
            )
            return make_strided_signature_tensor(
                _SIGNATURE_DTYPES[dtype],
                shape,
                assumed_align=align,
                use_int64_offsets=True,
                stride_divisibility=divisibility,
            )
        case _:
            raise ValueError(f"unsupported signature specification {spec}")


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
