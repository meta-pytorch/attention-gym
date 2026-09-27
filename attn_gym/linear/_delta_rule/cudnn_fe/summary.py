# SPDX-License-Identifier: BSD-3-Clause

"""Native v1.30 affine state summaries for context-parallel KDA.

Forward maps pack ``[B; A]`` for ``exit = entry @ A + B``; reverse maps pack ``[C; R]`` for
``d_entry = d_exit @ R + C``. One ``kda_summary`` launch writes both B (the zero-seeded final
state H) and A (the stored-domain transition ``M_buf``); one ``kda_bprop_summary`` launch writes
C (the entry cotangent under a zero exit cotangent). Every work item is the whole, uncut
sequence, so the arithmetic is independent of sequence ownership and packed token count.

Bounds select whole packed sequences on device: a Triton selector writes one uncut work item
per (sequence, head) and neutralizes unrequested sequences to zero width, and a gather maps each
requested bound to its sequence-indexed result. Bounds values may change between CUDA Graph
replays; nothing here synchronizes with the host.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import cutlass
import torch
import triton
import triton.language as tl

from attn_gym._backends.cute.utils import (
    get_device_properties,
    initialized_cuda_device,
    requires_int64_abi,
)
from attn_gym.utils import ceildiv

from .common.host import get_dtype
from .common.split_k import WORK_ITEM_FIELDS
from .kernel import kda_bprop_summary_f16, kda_summary_f16
from .plan import int32, workspace

# Staged KDA tensors: natural-log FP32 channel gate, FP32 post-sigmoid beta, normalized keys.
_GATE = {
    "l2norm": False,
    "safe_gate": False,
    "log_gate": True,
    "beta_sigmoid": False,
    "allow_neg_eigval": False,
}
_GATE_SCALE_LOG2 = kda_summary_f16.DEFAULT_GATE_LOWER_BOUND * kda_summary_f16.LOG2_E


@triton.jit
def _select_summary_work(
    cu, bounds, work, count, HEADS: tl.constexpr, ROWS: tl.constexpr, FIELDS: tl.constexpr
):
    """Write one uncut native work item per sequence/head, zero-width unless a bound selects it.

    ``ROWS == 0`` selects every sequence. Row layout follows ``common/split_k.py``:
    ``[seq, head, write_start, write_end, compute_start, compute_end, token_start, token_stop,
    final_dst, dstate_dst]``.
    """
    seq = tl.program_id(0)
    head = tl.program_id(1)
    start, stop = tl.load(cu + seq), tl.load(cu + seq + 1)
    if ROWS == 0:
        selected = True
    else:
        rows = tl.arange(0, triton.next_power_of_2(ROWS))
        lo = tl.load(bounds + 2 * rows, rows < ROWS, other=-1)
        hi = tl.load(bounds + 2 * rows + 1, rows < ROWS, other=-1)
        selected = tl.sum(((lo == start) & (hi == stop) & (lo < hi)).to(tl.int32), 0) > 0
    stop = tl.where(selected, stop, start)
    chunks = tl.cdiv(stop - start, 16)
    fields = tl.arange(0, 16)
    row = tl.where((fields == 0) | (fields >= 8), seq, tl.where(fields == 1, head, 0))
    row = tl.where((fields == 3) | (fields == 5), chunks, row)
    row = tl.where(fields == 6, start, tl.where(fields == 7, stop, row))
    tl.store(work + (seq * HEADS + head) * FIELDS + fields, row, fields < FIELDS)
    if (seq == 0) & (head == 0):
        tl.store(count, tl.num_programs(0) * HEADS)


@triton.jit
def _gather_summary_rows(
    maps,
    cu,
    bounds,
    output,
    HEADS: tl.constexpr,
    SEQUENCES: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Map each subsequence bound to its summary, or synthesize the empty identity.

    A nonempty row that is not a subsequence (NOTE [Summary ranges are subsequences] in
    ``attn_gym.linear.kda.stages``) is filled with NaN so the mistake surfaces at the first
    compose instead of silently using another subsequence's map.
    """
    row, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start, stop = tl.load(bounds + row * 2), tl.load(bounds + row * 2 + 1)
    seqs = tl.arange(0, triton.next_power_of_2(SEQUENCES))
    lo = tl.load(cu + seqs, seqs < SEQUENCES, other=-1)
    hi = tl.load(cu + seqs + 1, seqs < SEQUENCES, other=-1)
    matches = (lo == start) & (hi == stop)
    seq = tl.max(tl.where(matches, seqs, 0), 0)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    inside = offsets < (V + K) * K
    value = tl.load(maps + (seq * HEADS + head) * ((V + K) * K) + offsets, inside)
    # [0; I]: diagonal ones live only in the lower K rows.
    identity = (offsets // K == V + offsets % K).to(tl.float32)
    value = tl.where(start == stop, identity, value)
    valid = (start == stop) | (tl.sum(matches.to(tl.int32), 0) > 0)
    value = tl.where(valid, value, float("nan"))
    tl.store(output + (row * HEADS + head) * ((V + K) * K) + offsets, value, inside)


def _summary_rows(
    maps: torch.Tensor, cu_seqlens: torch.Tensor, bounds: torch.Tensor | None
) -> torch.Tensor:
    """Gather selected sequence maps into bound order after the native kernels finish."""
    if bounds is None:
        return maps
    sequences, heads, rows, k = maps.shape
    selected = torch.empty(bounds.shape[0], heads, rows, k, device=maps.device, dtype=maps.dtype)
    if bounds.shape[0]:
        block = 1024
        _gather_summary_rows[(bounds.shape[0], heads, ceildiv(rows * k, block))](
            maps, cu_seqlens, bounds, selected, heads, sequences, rows - k, k, block
        )
    return selected


@dataclass(frozen=True)
class _Launch:
    """Device-owned scheduling buffers for one selected uncut summary launch."""

    work_items: torch.Tensor
    work_count: torch.Tensor
    counters: torch.Tensor
    workspace: torch.Tensor
    num_sm: int

    @classmethod
    def prepare(
        cls,
        kernel: ModuleType,
        cu_seqlens: torch.Tensor,
        heads: int,
        bounds: torch.Tensor | None,
    ) -> _Launch:
        device = cu_seqlens.device
        sequences = cu_seqlens.shape[0] - 1
        work_items = torch.empty(
            sequences * heads, WORK_ITEM_FIELDS, dtype=torch.int32, device=device
        )
        work_count = int32(1, device)
        _select_summary_work[(sequences, heads)](
            cu_seqlens,
            work_count if bounds is None else bounds,
            work_items,
            work_count,
            heads,
            0 if bounds is None else bounds.shape[0],
            WORK_ITEM_FIELDS,
        )
        return cls(
            work_items,
            work_count,
            int32(2, device),
            workspace(kernel, sequences, device),
            get_device_properties(device).multi_processor_count,
        )

    def reset(self) -> None:
        """The work-stealing counters must be zero before every launch, including replays."""
        self.counters.zero_()


def _forward_summary(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state: torch.Tensor,
    transition: torch.Tensor,
    launch: _Launch,
) -> None:
    """Write H into ``state`` and stored-domain M into ``transition`` for every work item."""
    use_int64_offsets = requires_int64_abi(
        k,
        v,
        gate,
        beta,
        cu_seqlens,
        state,
        transition,
        launch.work_items,
        launch.work_count,
        launch.counters,
        launch.workspace,
    )
    cu_dtype = cutlass.Int64 if cu_seqlens.dtype == torch.int64 else cutlass.Int32
    prologue = kda_summary_f16._compile_kda_summary_prologue(
        get_dtype(k.dtype),
        get_dtype(gate.dtype),
        cu_dtype,
        False,
        False,
        use_int64_offsets,
    )
    compiled = kda_summary_f16._compile_kda_summary(
        get_dtype(k.dtype),
        get_dtype(gate.dtype),
        get_dtype(beta.dtype),
        None,
        None,
        cu_dtype,
        None,
        k.shape[-1],
        v.shape[-1],
        _GATE["l2norm"],
        _GATE["safe_gate"],
        _GATE_SCALE_LOG2,
        _GATE["log_gate"],
        _GATE["beta_sigmoid"],
        _GATE["allow_neg_eigval"],
        launch.num_sm,
        use_int64_offsets,
    )
    launch.reset()
    prologue(
        k,
        v,
        gate,
        cu_seqlens,
        None,
        launch.work_count,
        launch.work_items,
        None,
        launch.workspace,
    )
    compiled(
        k,
        v,
        gate,
        None,
        None,
        beta,
        cu_seqlens,
        None,
        state,
        transition,
        launch.work_items,
        launch.work_count,
        launch.counters,
        launch.workspace,
    )


def _reverse_summary(
    q: torch.Tensor,
    k: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    d_initial_state: torch.Tensor,
    d_final_state: torch.Tensor | None,
    launch: _Launch,
) -> None:
    """Write the entry cotangent for every work item; ``d_final_state`` None seeds zero."""
    use_int64_offsets = requires_int64_abi(
        q,
        k,
        gate,
        beta,
        d_output,
        cu_seqlens,
        d_initial_state,
        d_final_state,
        launch.work_items,
        launch.work_count,
        launch.counters,
        launch.workspace,
    )
    cu_dtype = cutlass.Int64 if cu_seqlens.dtype == torch.int64 else cutlass.Int32
    prologue = kda_bprop_summary_f16._compile_kda_bprop_summary_prologue(
        get_dtype(q.dtype),
        get_dtype(gate.dtype),
        cu_dtype,
        False,
        False,
        False,
        use_int64_offsets,
    )
    compiled = kda_bprop_summary_f16._compile_kda_bprop_summary(
        get_dtype(q.dtype),
        get_dtype(gate.dtype),
        None,
        None,
        cu_dtype,
        get_dtype(beta.dtype),
        get_dtype(d_initial_state.dtype),
        None if d_final_state is None else get_dtype(d_final_state.dtype),
        d_final_state is not None,
        _GATE["l2norm"],
        _GATE["safe_gate"],
        _GATE_SCALE_LOG2,
        _GATE["log_gate"],
        _GATE["beta_sigmoid"],
        _GATE["allow_neg_eigval"],
        1,
        1,
        q.shape[-1],
        d_output.shape[-1],
        launch.num_sm,
        use_int64_offsets,
    )
    launch.reset()
    prologue(
        q,
        k,
        gate,
        d_output,
        cu_seqlens,
        None,
        launch.work_count,
        launch.work_items,
        None,
        launch.workspace,
    )
    compiled(
        cutlass.Int32(1),
        cutlass.Int32(1),
        None,
        None,
        beta,
        cu_seqlens,
        d_initial_state,
        d_final_state,
        launch.work_items,
        launch.work_count,
        launch.counters,
        launch.workspace,
        scale,
    )


def _validate(tensors: dict[str, torch.Tensor], cu_seqlens: torch.Tensor, bounds) -> None:
    k = tensors["k"]
    if k.ndim != 4 or k.shape[0] != 1 or k.shape[-1] not in (64, 128):
        raise ValueError("k must have shape [1, T, H, K] with K in {64, 128}")
    for name in ("q", "v", "d_output"):
        if name in tensors and tensors[name].shape[:3] != k.shape[:3]:
            raise ValueError(f"{name} must match k's token and head extents")
    if "v" in tensors and tensors["v"].shape[-1] not in (64, 128):
        raise ValueError("v must have V in {64, 128}")
    if tensors["gate"].shape != k.shape or tensors["beta"].shape != k.shape[:3]:
        raise ValueError("gate and beta must match k's token and head extents")
    if k.dtype not in (torch.float16, torch.bfloat16) or any(
        tensors[name].dtype != k.dtype for name in ("q", "v", "d_output") if name in tensors
    ):
        raise TypeError("q, k, v, and d_output must share dtype float16 or bfloat16")
    if tensors["gate"].dtype != torch.float32 or tensors["beta"].dtype != torch.float32:
        raise TypeError("gate and beta must be float32")
    if not k.is_cuda or any(t.device != k.device for t in (*tensors.values(), cu_seqlens)):
        raise ValueError("all inputs must be CUDA tensors on k.device")
    if bounds is not None and (
        bounds.ndim != 2
        or bounds.shape[1] != 2
        or bounds.dtype != torch.int32
        or bounds.device != k.device
        or not bounds.is_contiguous()
    ):
        raise ValueError("bounds must be contiguous int32 [R, 2] on k.device")


def _identity_maps(sequences: int, heads: int, v: int, k: int, device) -> torch.Tensor:
    """``[0; I]`` per sequence/head: the map every skipped or empty sequence keeps."""
    maps = torch.zeros(sequences, heads, v + k, k, dtype=torch.float32, device=device)
    maps[:, :, v:].diagonal(dim1=-2, dim2=-1).fill_(1)
    return maps


def _launches(tokens: int, bounds: torch.Tensor | None) -> bool:
    return tokens != 0 and (bounds is None or bounds.shape[0] != 0)


def build_state_summaries(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    bounds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return FP32 ``[R, H, V + K, K]`` maps packed as ``[B; A]`` for whole packed sequences.

    Inputs use the staged ``[1, T, H, D]`` layout with FP32 natural-log gate and FP32 beta.
    Optional contiguous int32 ``bounds[R, 2]`` on ``k.device`` select, reorder, or duplicate
    consecutive ``cu_seqlens`` pairs; without bounds one row per sequence is returned. Empty
    intervals yield ``[0; I]``, nonempty rows that are not a sequence yield NaN, and zero rows
    yield no maps. Bounds values stay on device and may change during CUDA Graph replay.
    """
    _validate({"k": k, "v": v, "gate": gate, "beta": beta}, cu_seqlens, bounds)
    with initialized_cuda_device(k):
        sequences, heads = cu_seqlens.shape[0] - 1, k.shape[2]
        dim_v, dim_k = v.shape[-1], k.shape[-1]
        maps = _identity_maps(sequences, heads, dim_v, dim_k, k.device)
        if _launches(k.shape[1], bounds):
            launch = _Launch.prepare(kda_summary_f16, cu_seqlens, heads, bounds)
            _forward_summary(
                k[0],
                v[0],
                gate[0],
                beta[0],
                cu_seqlens,
                maps[:, :, :dim_v],
                maps[:, :, dim_v:],
                launch,
            )
        return _summary_rows(maps, cu_seqlens, bounds)


def build_state_grad_summaries(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    *,
    transpose_forward_transition: bool = True,
    bounds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return FP32 ``[R, H, V + K, K]`` maps packed as ``[C; R]`` for whole packed sequences.

    C is the native entry cotangent under a zero exit cotangent. R is ``A.T`` from the forward
    summary (the exact adjoint of right-multiplication by A) or, with
    ``transpose_forward_transition=False``, a second reverse launch seeded with the identity exit
    cotangent and zero ``d_output`` (requires ``V == K``). Inputs and bounds follow
    :func:`build_state_summaries`.
    """
    _validate(
        {"q": q, "k": k, "v": v, "gate": gate, "beta": beta, "d_output": d_output},
        cu_seqlens,
        bounds,
    )
    with initialized_cuda_device(q):
        sequences, heads = cu_seqlens.shape[0] - 1, q.shape[2]
        dim_v, dim_k = d_output.shape[-1], k.shape[-1]
        maps = _identity_maps(sequences, heads, dim_v, dim_k, q.device)
        if not _launches(q.shape[1], bounds):
            return _summary_rows(maps, cu_seqlens, bounds)
        if transpose_forward_transition:
            forward = _Launch.prepare(kda_summary_f16, cu_seqlens, heads, bounds)
            state = torch.empty(
                sequences, heads, dim_v, dim_k, dtype=torch.float32, device=q.device
            )
            transition = torch.empty(
                sequences, heads, dim_k, dim_k, dtype=torch.float32, device=q.device
            )
            _forward_summary(k[0], v[0], gate[0], beta[0], cu_seqlens, state, transition, forward)
            maps[:, :, dim_v:].copy_(transition.transpose(-1, -2))
        elif dim_v != dim_k:
            raise ValueError("the identity reverse probe requires V == K")
        launch = _Launch.prepare(kda_bprop_summary_f16, cu_seqlens, heads, bounds)
        _reverse_summary(
            q[0],
            k[0],
            gate[0],
            beta[0],
            d_output[0],
            cu_seqlens,
            scale,
            maps[:, :, :dim_v],
            None,
            launch,
        )
        if not transpose_forward_transition:
            identity = (
                torch.eye(dim_k, device=q.device).expand(sequences, heads, -1, -1).contiguous()
            )
            _reverse_summary(
                q[0],
                k[0],
                gate[0],
                beta[0],
                torch.zeros_like(d_output[0]),
                cu_seqlens,
                scale,
                maps[:, :, dim_v:],
                identity,
                launch,
            )
        return _summary_rows(maps, cu_seqlens, bounds)


__all__ = ["build_state_grad_summaries", "build_state_summaries"]
