# SPDX-License-Identifier: BSD-3-Clause

"""Launch-contract checks shared by the cuDNN kernel configs and hosts.

Every check runs on host metadata before cache selection, so an invalid config or tensor fails
with a ``ValueError`` instead of compiling a kernel whose barriers or TMA descriptors cannot work.
"""

import functools
from collections.abc import Callable, Sequence

import cutlass
import torch

from attn_gym._backends.cute.utils import validate_tma_tensor

from .thd import TENSOR_MAP_QWORDS
from .tvm_ffi import WORK_ITEM_FIELDS

NAMED_BARRIER_IDS = range(1, 16)  # 0 is the CTA-wide barrier
STATE_DIMS = (64, 128)
TMEM_COLUMNS = 512
# Base-pointer residue kept in a memoized check's key; every checked alignment divides it.
_ALIGNMENT_RESIDUE = 1024
_MAX_MEMOIZED_CHECKS = 256


def _check_key_item(value):
    if isinstance(value, torch.Tensor):
        return (
            value.shape,
            value.stride(),
            value.dtype,
            value.device,
            value.data_ptr() % _ALIGNMENT_RESIDUE,
        )
    return value


def memoize_launch_check(check: Callable[..., None]) -> Callable[..., None]:
    """Skip a keyword-only launch check whose inputs repeat a call that already passed.

    The check may read only tensor metadata (shape, strides, dtype, device, base alignment up to
    ``_ALIGNMENT_RESIDUE`` bytes), tensor presence and the other hashable arguments, plus the
    active CUDA device; all of them form the key. Failing calls are never recorded, so an
    invalid launch raises the same error every time.
    """
    passed: set = set()

    @functools.wraps(check)
    def memoized(**kwargs) -> None:
        key = (
            torch.cuda.current_device() if torch.cuda.is_initialized() else None,
            tuple(kwargs),
            *map(_check_key_item, kwargs.values()),
        )
        if key in passed:
            return
        check(**kwargs)
        if len(passed) >= _MAX_MEMOIZED_CHECKS:
            passed.clear()
        passed.add(key)

    return memoized


def validate_kernel_domain(kernel: str, io_dtype, dims: Sequence[int], **positive: int) -> None:
    """Check the io dtype (Float16/BFloat16), head dims in ``STATE_DIMS`` and positive counts."""
    if io_dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise ValueError(f"{kernel} serves Float16/BFloat16 io, got {io_dtype}")
    if any(dim not in STATE_DIMS for dim in dims):
        raise ValueError(f"{kernel} serves head dims in {STATE_DIMS}, got {tuple(dims)}")
    for name, value in positive.items():
        if value < 1:
            raise ValueError(f"{kernel} needs a positive {name}, got {value}")


def validate_tmem_columns(kernel: str, columns: int) -> None:
    """Check that a TMEM layout fits the 512-column allocation."""
    if columns > TMEM_COLUMNS:
        raise ValueError(f"{kernel} TMEM layout needs {columns} > {TMEM_COLUMNS} columns")


def validate_warp_roles(groups: Sequence[Sequence[int]], singles: Sequence[int]) -> int:
    """Check that warp roles tile ``0..n-1`` and every compute group is one aligned warpgroup
    (its TMEM lane quarter is ``warp_id % 4``); return the warp count ``n``."""
    roles = [*(warp for group in groups for warp in group), *singles]
    if sorted(roles) != list(range(len(roles))):
        raise ValueError(f"warp roles must be disjoint and cover IDs 0..n-1, got {roles}")
    for group in groups:
        if tuple(group) != tuple(range(group[0], group[0] + 4)) or group[0] % 4:
            raise ValueError(f"compute groups must be aligned four-warp warpgroups, got {group}")
    return len(roles)


def validate_named_barriers(threads_per_cta: int, **barriers: tuple[int, int]) -> None:
    """Check ``name=(barrier_id, thread_count)`` named barriers: distinct hardware IDs and
    whole-warp participant counts that fit in the CTA."""
    ids = [barrier_id for barrier_id, _ in barriers.values()]
    if len(set(ids)) != len(ids) or any(barrier_id not in NAMED_BARRIER_IDS for barrier_id in ids):
        raise ValueError(f"named barrier IDs must be distinct and in [1, 16), got {barriers}")
    for name, (_, threads) in barriers.items():
        if threads <= 0 or threads % 32 or threads > threads_per_cta:
            raise ValueError(
                f"{name} barrier needs a whole-warp thread count in (0, {threads_per_cta}], "
                f"got {threads}"
            )


def checkpoint_capacity_bound(tokens: int, num_sequences: int, interval: int) -> int:
    """Checkpoint rows any packing of ``tokens`` into ``num_sequences`` sequences can emit with
    one row per started ``interval`` (read from metadata only, so graph-safe)."""
    nonempty = min(tokens, num_sequences)
    return nonempty + (tokens - nonempty) // interval


def validate_seqlens(cu_seqlens: torch.Tensor) -> int:
    """Check the ``[B + 1]`` int32/int64 cumulative-length vector; return ``B``."""
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be a compact vector with at least two entries")
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"cu_seqlens must have dtype int32 or int64, got {cu_seqlens.dtype}")
    if cu_seqlens.data_ptr() % cu_seqlens.element_size():
        raise ValueError("cu_seqlens data pointer must be element aligned")
    return cu_seqlens.numel() - 1


def validate_tensor(
    name: str,
    tensor: torch.Tensor | None,
    shape: Sequence[int | None],
    dtypes: Sequence[str],
    *,
    align: int = 16,
    tma: bool = False,
    compact: bool = False,
    min_rows: int = 0,
) -> None:
    """Check one launch tensor: ``shape`` (``None`` = any extent), dtype name, ``min_rows`` for
    mode 0, base alignment, unit inner stride, and nonnegative (TMA: 16-byte) outer strides."""
    if tensor is None:
        raise ValueError(f"{name} is required")
    if tensor.ndim != len(shape) or any(
        want is not None and want != got for want, got in zip(shape, tensor.shape)
    ):
        want = tuple("*" if extent is None else extent for extent in shape)
        raise ValueError(f"{name} must have shape {want}, got {tuple(tensor.shape)}")
    if tensor.shape[0] < min_rows:
        raise ValueError(f"{name} requires at least {min_rows} rows, got {tensor.shape[0]}")
    if str(tensor.dtype).removeprefix("torch.") not in dtypes:
        raise ValueError(f"{name} must have dtype in {tuple(dtypes)}, got {tensor.dtype}")
    if tma:
        validate_tma_tensor(name, tensor, alignment=align)
        return
    if tensor.data_ptr() % align:
        raise ValueError(f"{name} data pointer must be {align}-byte aligned")
    if tensor.stride(-1) != 1 or any(stride < 0 for stride in tensor.stride()):
        raise ValueError(f"{name} needs a unit inner stride and nonnegative outer strides")
    if compact and not tensor.is_contiguous():
        raise ValueError(f"{name} must be compact")


def validate_work_table(work_items, work_count, scheduler_counter, *, min_rows: int) -> None:
    """Check the ``[rows, WORK_ITEM_FIELDS]`` item table, its count cell and the ticket counter."""
    validate_tensor(
        "work_items",
        work_items,
        (None, WORK_ITEM_FIELDS),
        ("int32",),
        compact=True,
        min_rows=min_rows,
    )
    validate_tensor("work_count", work_count, (None,), ("int32",), align=4, min_rows=1)
    validate_tensor(
        "scheduler_counter", scheduler_counter, (None,), ("int32",), align=4, min_rows=1
    )


def validate_workspace(name: str, workspace, desc_arrays: int, batch: int) -> None:
    """Check a TMA-descriptor workspace holding ``desc_arrays`` per-batch descriptor arrays (the
    ``tensormap_workspace_bytes`` size of a module with ``TENSORMAP_DESC_ARRAYS = desc_arrays``)."""
    words = (TENSOR_MAP_QWORDS * 8 * desc_arrays * batch + 128) // 8
    validate_tensor(name, workspace, (None,), ("int64",), align=128, min_rows=words)
