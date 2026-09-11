# SPDX-License-Identifier: BSD-3-Clause

"""Native BT16 affine probes for whole packed sequences.

Forward maps pack [B; A] for ``exit = entry @ A + B``; reverse maps pack [C; R] for
``d_entry = d_exit @ R + C``. Native state/operand rounding makes these approximate affine
maps, not bitwise reconstruction of an ordinary unsharded cuDNN run. Every probe is unsplit,
so its arithmetic is independent of sequence ownership and packed token count.
"""

from __future__ import annotations

from types import ModuleType

import torch
import triton
import triton.language as tl

from attn_gym._backends.cute.utils import initialized_cuda_device
from attn_gym.utils import ceildiv

from .forward import validate_available
from .kernels import kda_bprop_f16, kda_recompute_f16
from .kernels.common.host import tensormap_workspace_bytes
from .schedule import CudnnSchedule, prepare_cudnn_schedule


@triton.jit
def _select_summary_work(cu, bounds, work, count, HEADS: tl.constexpr, ROWS: tl.constexpr):
    """Keep one native work item per sequence/head, neutralizing unrequested sequences."""
    seq = tl.program_id(0)
    head = tl.program_id(1)
    start, stop = tl.load(cu + seq), tl.load(cu + seq + 1)
    rows = tl.arange(0, triton.next_power_of_2(ROWS))
    lo = tl.load(bounds + 2 * rows, rows < ROWS, other=-1)
    hi = tl.load(bounds + 2 * rows + 1, rows < ROWS, other=-1)
    selected = tl.sum(((lo == start) & (hi == stop) & (lo < hi)).to(tl.int32), 0) > 0
    stop = tl.where(selected, stop, start)
    chunks = tl.cdiv(stop - start, 16)
    # Native row: [seq, head, wstart, wend, cstart, cend, token_start, token_stop].
    fields = tl.arange(0, 8)
    row = tl.where(fields == 0, seq, tl.where(fields == 1, head, 0))
    row = tl.where((fields == 3) | (fields == 5), chunks, row)
    row = tl.where(fields == 6, start, tl.where(fields == 7, stop, row))
    tl.store(work + (seq * HEADS + head) * 8 + fields, row)
    if (seq == 0) & (head == 0):
        tl.store(count, tl.num_programs(0) * HEADS)


@triton.jit
def _gather_summary_rows(
    maps, cu, bounds, output, HEADS: tl.constexpr, SEQUENCES: tl.constexpr, BLOCK: tl.constexpr
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
    value = tl.load(maps + (seq * HEADS + head) * (256 * 128) + offsets)
    # [0; I]: diagonal ones live only in the lower 128 rows.
    identity = (offsets // 128 == 128 + offsets % 128).to(tl.float32)
    value = tl.where(start == stop, identity, value)
    valid = (start == stop) | (tl.sum(matches.to(tl.int32), 0) > 0)
    value = tl.where(valid, value, float("nan"))
    tl.store(output + (row * HEADS + head) * (256 * 128) + offsets, value)


def build_cudnn_state_summaries(
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    bounds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return FP32 ``[N, H, 256, 128]`` maps packed as ``[B; A]`` for subsequences.

    B is the no-state final state; A is the identity-seeded, zero-value final state.
    Inputs use staged ``[1, T, H, 128]`` layout, FP32 gate/beta, and normalized keys.
    Optional contiguous int32 ``bounds[R, 2]`` on k.device select, reorder, or duplicate
    consecutive cu_seqlens pairs. Empty intervals yield ``[0; I]``; zero rows yield no maps.
    Bounds values stay on device and may change during CUDA Graph replay.
    """
    maps = _build_cudnn_state_probes(k, value, gate, beta, cu_seqlens, bounds, bias=True)
    return _summary_rows(maps, cu_seqlens, bounds)


def _summary_rows(
    maps: torch.Tensor, cu_seqlens: torch.Tensor, bounds: torch.Tensor | None
) -> torch.Tensor:
    """Gather selected probes only after native kernels finish writing by sequence index."""
    if bounds is None:
        return maps
    sequences, heads = maps.shape[:2]
    selected = torch.empty(bounds.shape[0], heads, 256, 128, device=maps.device, dtype=maps.dtype)
    if bounds.shape[0]:
        with initialized_cuda_device(maps):
            _gather_summary_rows[(bounds.shape[0], heads, 32)](
                maps, cu_seqlens, bounds, selected, heads, sequences, 1024
            )
    return selected


def _prepare_summary_launch(
    gate: torch.Tensor,
    cu_seqlens: torch.Tensor,
    bounds: torch.Tensor | None,
    kernel: ModuleType,
) -> tuple[CudnnSchedule, torch.Tensor]:
    """Allocate native launch buffers and select subsequence work without a host sync.

    With bounds, the caller must reset counters before each probe and disable prologue
    ordering; otherwise the prologue generates all work and resets counters itself.
    """
    sequences, heads = cu_seqlens.shape[0] - 1, gate.shape[2]
    schedule = prepare_cudnn_schedule(
        gate,
        cu_seqlens,
        tile_tokens=16,
        counter_count=2,
        split=False,
        stream=torch.cuda.current_stream(gate.device).cuda_stream,
    )
    if bounds is not None:
        _select_summary_work[(sequences, heads)](
            cu_seqlens, bounds, schedule.work_items, schedule.work_count, heads, bounds.shape[0]
        )
    workspace = torch.empty(
        ceildiv(tensormap_workspace_bytes(kernel, sequences), 8),
        dtype=torch.int64,
        device=gate.device,
    )
    return schedule, workspace


def _build_cudnn_state_probes(
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    bounds: torch.Tensor | None,
    *,
    bias: bool,
) -> torch.Tensor:
    """Probe selected sequences into full-size storage indexed by the native sequence IDs.

    ``bias=False`` skips the zero-state probe and returns only the transition ``A``.
    """
    validate_available(k)
    if k.ndim != 4 or k.shape[0] != 1 or k.shape[-1] != 128:
        raise ValueError("k must have shape [1, T, H, 128]")
    if value.shape != k.shape or gate.shape != k.shape or beta.shape != k.shape[:3]:
        raise ValueError("value, gate, and beta must match k's token and head extents")
    if k.dtype not in (torch.float16, torch.bfloat16) or value.dtype != k.dtype:
        raise TypeError("k and value must share dtype float16 or bfloat16")
    if gate.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("gate and beta must be float32")
    if any(t.device != k.device for t in (value, gate, beta, cu_seqlens)):
        raise ValueError("all inputs must be on k.device")
    if bounds is not None and (
        bounds.ndim != 2
        or bounds.shape[1] != 2
        or bounds.dtype != torch.int32
        or bounds.device != k.device
        or not bounds.is_contiguous()
    ):
        raise ValueError("bounds must be contiguous int32 [R, 2] on k.device")

    with initialized_cuda_device(k):
        num_sequences, heads = cu_seqlens.shape[0] - 1, k.shape[2]
        summaries = torch.zeros(
            num_sequences, heads, 256, 128, dtype=torch.float32, device=k.device
        )
        summaries[:, :, 128:, :].diagonal(dim1=-2, dim2=-1).fill_(1)
        if k.shape[1] != 0 and (bounds is None or bounds.shape[0] != 0):
            schedule, workspace = _prepare_summary_launch(
                gate, cu_seqlens, bounds, kda_recompute_f16
            )
            for transition_only in (False, True) if bias else (True,):
                if bounds is not None:
                    schedule.counters.zero_()
                kda_recompute_f16.chunk_kda_recompute_sm100(
                    k[0],
                    None if transition_only else value[0],
                    gate[0],
                    beta[0],
                    cu_seqlens,
                    initial_state=None,
                    output_state=summaries[:, :, 128:, :]
                    if transition_only
                    else summaries[:, :, :128, :],
                    work_items=schedule.work_items,
                    work_count=schedule.work_count,
                    sched_ctr=schedule.counters,
                    sched_all=schedule.counters,
                    order_in_prologue=bounds is None,
                    tensormap_workspace=workspace,
                    transition_only=transition_only,
                )
        return summaries


def build_cudnn_state_grad_summaries(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    *,
    transpose_forward_transition: bool = False,
    bounds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return subsequence FP32 ``[N, H, 256, 128]`` maps packed as ``[C; R]``.

    C is the native entry cotangent with zero exit cotangent. R uses an identity exit
    cotangent and zero d_output. ``transpose_forward_transition=True`` instead uses A.T
    from the forward probe: algebraically the adjoint, but not native backward rounding.
    Inputs and bounds follow :func:`build_cudnn_state_summaries`; selection applies to both
    probes and empty intervals yield ``[0; I]``.
    """
    validate_available(q)
    with initialized_cuda_device(q):
        sequences, heads = cu_seqlens.numel() - 1, q.shape[2]
        if transpose_forward_transition:
            maps = _build_cudnn_state_probes(k, value, gate, beta, cu_seqlens, bounds, bias=False)
            maps[:, :, 128:].copy_(maps[:, :, 128:].transpose(-1, -2).contiguous())
        else:
            maps = torch.zeros(sequences, heads, 256, 128, device=q.device)
            maps[:, :, 128:].diagonal(dim1=-2, dim2=-1).fill_(1)
        if q.shape[1] == 0 or (bounds is not None and bounds.shape[0] == 0):
            return _summary_rows(maps, cu_seqlens, bounds)
        schedule, workspace = _prepare_summary_launch(gate, cu_seqlens, bounds, kda_bprop_f16)
        # The dH recurrence ignores V/checkpoints; token gradients are computed but discarded.
        # Zero-stride TMA checkpoint rows all address one tile, avoiding a state recompute.
        checkpoints = torch.zeros(1, heads, 128, 128, device=q.device, dtype=q.dtype).expand(
            q.shape[1] // 16 + sequences, -1, -1, -1
        )
        scratch = [torch.empty_like(t[0]) for t in (q, k, value, gate, beta)]
        for transition in (False,) if transpose_forward_transition else (False, True):
            if bounds is not None:
                schedule.counters.zero_()
            exit_state = None
            do = d_output
            if transition:
                exit_state = torch.eye(128, device=q.device).expand(sequences, heads, 128, 128)
                do = torch.zeros_like(d_output[:, :1]).expand_as(d_output)
            kda_bprop_f16.chunk_kda_bwd_sm100(
                q[0],
                k[0],
                value[0],
                gate[0],
                beta[0],
                do[0],
                checkpoints,
                *scratch,
                cu_seqlens,
                scale,
                use_initial_state=True,
                d_initial_state=maps[:, :, 128:] if transition else maps[:, :, :128],
                d_final_state=exit_state,
                work_items=schedule.work_items,
                work_count=schedule.work_count,
                sched_ctr=schedule.counters if sequences * heads <= schedule.num_sms else None,
                sched_all=schedule.counters,
                order_in_prologue=bounds is None,
                tensormap_workspace=workspace,
            )
        return _summary_rows(maps, cu_seqlens, bounds)
