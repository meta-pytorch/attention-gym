# SPDX-License-Identifier: BSD-3-Clause
#
# Host helpers and launch-contract checks derived in 2026 from the cudnn-frontend v1.30.0 kernel
# modules (common/host.py, common/launch.py, kda_prefill_f16, gdn_bprop_f16, gdn_tinv_f16).

"""Host-side launch contract of the warpdrive GDN and KDA launch bundles.

Dtype mapping and the operand checks the bundles run on tensor metadata before selecting a
compiled ABI, so an invalid plan fails with a ``ValueError`` instead of compiling a kernel whose
TMA descriptors cannot work. Nothing here traces or launches a kernel.
"""

import functools
from collections.abc import Callable, Sequence

import cutlass
import torch

from attn_gym._backends.cute.utils import validate_tma_tensor
from attn_gym.linear._delta_rule.paged_state import state_row_alignment

from .gdn_tinv import B_T as GDN_B_T
from .gdn_tinv import tinv_rows
from .work_tables import WORK_ITEM_FIELDS

IO_DTYPES = ("bfloat16", "float16")
SCALAR_DTYPES = ("float32", "bfloat16", "float16")
STATE_DTYPES = ("float32", "bfloat16")

_DTYPES = {
    "bfloat16": cutlass.BFloat16,
    "float16": cutlass.Float16,
    "float32": cutlass.Float32,
}
_OP_DTYPE_NAMES = {cutlass.Float16: "fp16", cutlass.BFloat16: "bf16", cutlass.Float32: "fp32"}
# Base-pointer residue kept in a memoized check's key; every host-checked alignment divides it.
# Larger would split keys on allocator placement (caching-allocator blocks are 512-byte rounded).
_ALIGNMENT_RESIDUE = 128
_MAX_MEMOIZED_CHECKS = 256


# ---- dtypes ----------------------------------------------------------------------------------


def get_dtype(dtype):
    """Map an exact Torch dtype name to its CuTeDSL type."""
    name = str(dtype).removeprefix("torch.")
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(
            f"Unsupported dtype {dtype}, expected bfloat16, float16 or float32"
        ) from None


def dtype_name(dtype) -> str:
    """Bare dtype name (``"float32"``, ``"int64"``, ...) of a Torch dtype, CuTeDSL type or name."""
    if isinstance(dtype, type):
        return dtype.__name__.lower()
    return str(dtype).split(".")[-1]


def op_dtype_name(dtype) -> str:
    """The op constructors' name (``"fp16"``, ``"bf16"``, ``"fp32"``) of a CuTeDSL type."""
    return _OP_DTYPE_NAMES[dtype]


def tma_divisibility(dtype) -> int:
    """Elements per 16 bytes of a CuTeDSL type: the outer-stride divisibility a compile signature
    may assume for a public TMA operand, which the host checks only to 16 bytes. Promising more
    (e.g. 8 fp32 elements) rejects legal strided inputs at launch."""
    return 16 // (dtype.width // 8)


# ---- generic checks --------------------------------------------------------------------------


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

    memoized.cache_clear = passed.clear
    return memoized


def validate_cuda_tensors(reference: torch.Tensor, **tensors: torch.Tensor | None) -> None:
    """Reject cross-device or inactive-device launches before selecting a compiled ABI."""
    if not reference.is_cuda:
        raise ValueError("the warpdrive launch bundles require CUDA tensors")
    if torch.cuda.current_device() != reference.get_device():
        raise ValueError("the active CUDA device must match the input device")
    for name, tensor in tensors.items():
        if tensor is not None and tensor.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}")


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


def validate_state_series(name, tensor, rows, heads_out, dim_v, dim_k, dtypes, *, align=16):
    """Check a ``[rows, HO, V, K]`` state or checkpoint series with at least ``rows`` rows."""
    validate_tensor(
        name, tensor, (None, heads_out, dim_v, dim_k), dtypes, align=align, tma=True, min_rows=rows
    )


# ---- KDA forward -----------------------------------------------------------------------------


def validate_kda_forward_operands(q, k, v, gate, beta, o, cu_seqlens):
    """Check the packed KDA forward operands: ``[T, H, D]`` q/k/v, ``[T, HO, DV]`` output, the
    ``[T, HO, DK]`` channel gate and ``[T, HO]`` beta (fp32 or the io dtype); return
    ``(tokens, heads_out, num_seqs)``."""
    validate_cuda_tensors(q, k=k, v=v, gate=gate, beta=beta, o=o, cu_seqlens=cu_seqlens)
    num_seqs = validate_seqlens(cu_seqlens)
    if q.ndim != 3 or v.ndim != 3 or gate.ndim != 3:
        raise ValueError("q, k, v and o must be packed [T, H, D] and gate [T, HO, DK]")
    tokens, dim_k, dim_v, heads_out = q.shape[0], q.shape[2], v.shape[2], gate.shape[1]
    io = (str(q.dtype).removeprefix("torch."),)
    if io[0] not in IO_DTYPES:
        raise ValueError(f"q must have dtype in {IO_DTYPES}, got {q.dtype}")
    for name, tensor, shape in (
        ("q", q, (tokens, None, dim_k)),
        ("k", k, (tokens, None, dim_k)),
        ("v", v, (tokens, None, dim_v)),
        ("o", o, (tokens, heads_out, dim_v)),
    ):
        validate_tensor(name, tensor, shape, io, tma=True)
    for name, heads in (("q", q.shape[1]), ("k", k.shape[1]), ("v", v.shape[1])):
        if heads <= 0 or heads_out % heads:
            raise ValueError(f"{name} heads ({heads}) must divide the output heads ({heads_out})")
    validate_tensor("gate", gate, (tokens, heads_out, dim_k), ("float32", *io), tma=True)
    validate_tensor("beta", beta, (tokens, heads_out), ("float32", *io))
    return tokens, heads_out, num_seqs


def paged_state_align(pool: torch.Tensor) -> int:
    """The state row alignment a paged prefill compiles for: 16 bytes, or the element size for a
    pool whose rows are not all 16-byte aligned (NOTE [Unaligned state rows] in kda_common)."""
    return 16 if state_row_alignment(pool) == 16 else pool.element_size()


def validate_kda_forward_states(q, v, heads_out, num_seqs, *, align: int = 16, **states):
    """Check ``name=(state, routed)`` ``[rows, HO, V, K]`` states: at least ``num_seqs`` rows
    unless routed through an index table (a paged pool or seed/final indices), and ``align``-byte
    aligned bases."""
    validate_cuda_tensors(q, **{name: state for name, (state, _) in states.items()})
    for name, (state, routed) in states.items():
        if state is not None:
            validate_tensor(
                name,
                state,
                (None, heads_out, v.shape[2], q.shape[2]),
                STATE_DTYPES,
                align=align,
                min_rows=0 if routed else num_seqs,
            )


# ---- GDN backward ----------------------------------------------------------------------------


def validate_gdn_bwd_operands(
    q,
    k,
    v,
    do,
    gate,
    beta,
    cu_seqlens,
    *,
    dq=None,
    dk=None,
    dv=None,
    dgate=None,
    dbeta=None,
):
    """Check the packed GDN backward operands shared by the standalone and bundled hosts (every
    ``None`` optional is skipped); return ``(tokens, heads_out, num_seqs)``."""
    validate_cuda_tensors(
        q,
        k=k,
        v=v,
        do=do,
        gate=gate,
        beta=beta,
        cu_seqlens=cu_seqlens,
        dq=dq,
        dk=dk,
        dv=dv,
        dgate=dgate,
        dbeta=dbeta,
    )
    num_seqs = validate_seqlens(cu_seqlens)
    if q.ndim != 3 or v.ndim != 3 or gate.ndim != 2:
        raise ValueError("q, k, v, do must be packed [T, H, D] and gate, beta [T, HO]")
    tokens, dim_k, dim_v, heads_out = q.shape[0], q.shape[2], v.shape[2], gate.shape[1]
    io = (str(q.dtype).removeprefix("torch."),)
    if io[0] not in IO_DTYPES:
        raise ValueError(f"q must have dtype in {IO_DTYPES}, got {q.dtype}")
    for name, tensor, shape in (
        ("q", q, (tokens, None, dim_k)),
        ("k", k, (tokens, None, dim_k)),
        ("v", v, (tokens, None, dim_v)),
        ("do", do, (tokens, heads_out, dim_v)),
        ("dq", dq, (tokens, heads_out, dim_k)),
        ("dk", dk, (tokens, heads_out, dim_k)),
        ("dv", dv, (tokens, heads_out, dim_v)),
    ):
        if tensor is not None or name in ("q", "k", "v"):
            validate_tensor(name, tensor, shape, io, tma=True)
    for name, heads in (("q", q.shape[1]), ("k", k.shape[1]), ("v", v.shape[1])):
        if heads <= 0 or heads_out % heads:
            raise ValueError(f"{name} heads ({heads}) must divide the output heads ({heads_out})")
    validate_tensor("gate", gate, (tokens, heads_out), SCALAR_DTYPES)
    validate_tensor("beta", beta, (tokens, heads_out), ("float32", *io))
    if dgate is not None:
        validate_tensor("dgate", dgate, (tokens, heads_out), ("float32",))
    if dbeta is not None:
        validate_tensor(
            "dbeta", dbeta, (tokens, heads_out), (str(beta.dtype).removeprefix("torch."),)
        )
    return tokens, heads_out, num_seqs


def validate_gdn_bwd_bundle(
    q,
    k,
    v,
    do,
    dq,
    dk,
    dv,
    gate,
    beta,
    cu_seqlens,
    checkpoints,
    *,
    num_pieces,
    b_t,
    dgate=None,
    dbeta=None,
):
    """Check the operands every bundled GDN backward host (warmup/uncut and chain) shares: the
    bprop's ``num_pieces`` checkpoint series (one per sequence, or per chain piece); return
    ``(tokens, heads_out, num_seqs)``."""
    if b_t != GDN_B_T or num_pieces < 1:
        raise ValueError(f"bundled backward hosts need b_t={GDN_B_T} and a positive piece count")
    tokens, heads_out, num_seqs = validate_gdn_bwd_operands(
        q,
        k,
        v,
        do,
        gate,
        beta,
        cu_seqlens,
        dq=dq,
        dk=dk,
        dv=dv,
        dgate=dgate,
        dbeta=dbeta,
    )
    validate_cuda_tensors(q, checkpoints=checkpoints)
    io = str(q.dtype).removeprefix("torch.")
    validate_state_series(
        "checkpoints",
        checkpoints,
        tinv_rows(tokens, num_pieces),
        heads_out,
        v.shape[2],
        q.shape[2],
        (io,),
    )
    return tokens, heads_out, num_seqs


def validate_gdn_tinv(tinv, row_table, row_count, tokens, num_seqs, heads_out, io_dtype):
    """Check the GDN chunk-factor tiles (one row per chunk plus one per sequence) and the row
    table the recompute and bprop summary read."""
    rows = tinv_rows(tokens, num_seqs)
    validate_tensor(
        "tinv",
        tinv,
        (None, heads_out, GDN_B_T, GDN_B_T),
        (io_dtype,),
        align=128,
        tma=True,
        min_rows=rows,
    )
    validate_tensor("tinv_rows", row_table, (None, 4), ("int32",), compact=True, min_rows=rows)
    validate_tensor("tinv_row_count", row_count, (None,), ("int32",), align=4, min_rows=1)
