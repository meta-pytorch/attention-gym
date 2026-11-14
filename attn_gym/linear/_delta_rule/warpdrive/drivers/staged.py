# SPDX-License-Identifier: BSD-3-Clause

"""Staged KDA and GDN plans that run the local pass over the split summaries' pieces.

When the context-parallel summaries split (NOTE [Summary split cost] in ``plan.py``), their
piece maps already hold everything the local pass needs to start every piece at once. NOTE
[Staged piece plans]::

    forward:  maps    = forward summary of every piece      ([H_j; M_j])
              prefix  = map chain, X_j = [B_<j; A_<j]        (composite: the summary)
              run     = X_j(seed) = seed @ A_<j + B_<j, then one prefill over the pieces
    backward: maps    = forward and reverse summaries       ([H_j; M_j], C_j)
              suffix  = transposed map chain, [C_>j; R_>j]   (composite: the summary)
              states  = state chain from the seed, recompute every piece's checkpoints
              run     = dX_j(d_final) = d_final @ R_>j + C_>j, then one bprop over the pieces

``prepare`` (``prepare_backward``) builds the maps once; ``state_summaries`` gathers their
composites and ``run`` only applies the entry state (exit cotangent), so the summaries cost the
staged pass nothing beyond the compositions. GDN's pieces share one ``gdn_tinv`` pass. The
pieces follow the summaries' span, a function of the call's shapes and SM count, and every
sequence is planned whatever the bounds, so the results are deterministic and independent of
call order and bounds; they are not the chunk op's bits, whose own plan for the shape differs.

The maps (and the backward's checkpoints) are built once from the inputs at ``prepare``
(``prepare_backward``), while ``run`` also reads the saved inputs themselves, so a split handle's
inputs and offsets must not change before its last use. A CUDA graph that replays a split handle
over new input values must capture ``prepare`` (``prepare_backward``) too, as the
context-parallel ops do. Unsplit handles rerun their summaries from the saved inputs on every
call.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from attn_gym._backends.cute.utils import initialized_cuda_device, requires_int64_abi
from attn_gym.linear._delta_rule.triton.group_sum import group_sum

from .. import (
    gdn_bprop,
    gdn_bprop_summary,
    gdn_prefill,
    gdn_recompute,
    gdn_summary,
    kda_bprop,
    kda_prefill,
    kda_recompute,
    piece_chain,
)
from ..kda_common import B_T, TORCH_NAMES
from ..ragged import new_tma_slots
from . import summary
from .plan import _num_sm
from .summary import (
    CHAIN_ROWS,
    GDN_FORWARD_COST,
    GDN_REVERSE_COST,
    KDA_FORWARD_COST,
    KDA_REVERSE_COST,
    _Launch,
)


@triton.jit(do_not_specialize=["sequences"])
def _apply_maps(
    maps,
    state,
    out,
    main_rows,
    sequences,
    state_stride_seq,
    state_stride_head,
    state_stride_row,
    HEADS: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    """``out[p, h] = state[s, h] @ maps[p, h, V:] + maps[p, h, :V]`` in 3xTF32 for every piece
    ``p`` of sequence ``s`` (``main_rows``, the ``piece_chain`` table); pieces past the planned
    ones are skipped. ``SEQ_BLOCK`` is a power of two covering ``sequences``: a split plan has
    at most half an SM count of them (NOTE [Summary split cost])."""
    piece, head = tl.program_id(0), tl.program_id(1)
    rows = tl.program_id(2) * BM + tl.arange(0, BM)
    seqs = tl.arange(0, SEQ_BLOCK)
    starts = tl.load(main_rows + seqs, seqs < sequences, other=2**30)
    if piece * HEADS < tl.load(main_rows + sequences):
        seq = tl.sum((starts <= piece * HEADS).to(tl.int32), 0) - 1
        cols = tl.arange(0, K)
        size = (V + K) * K
        base = maps + (piece.to(tl.int64) * HEADS + head) * size
        source = state + seq.to(tl.int64) * state_stride_seq + head * state_stride_head
        acc = tl.load(base + rows[:, None] * K + cols[None, :])
        for kb in tl.static_range(0, K, BK):
            ks = kb + tl.arange(0, BK)
            a = tl.load(source + rows[:, None] * state_stride_row + ks[None, :])
            t = tl.load(base + (V + ks[:, None]) * K + cols[None, :])
            acc = tl.dot(a, t, acc, input_precision="tf32x3")
        target = out + (piece.to(tl.int64) * HEADS + head) * (V * K)
        tl.store(target + rows[:, None] * K + cols[None, :], acc)


@triton.jit(do_not_specialize=["sequences", "tokens"])
def _zero_tail(rows, cu, sequences, tokens, ROW: tl.constexpr, BLOCK: tl.constexpr):
    """Zero ``rows`` (``tokens`` rows of ``ROW`` elements) past the stream end ``cu[sequences]``,
    which no piece writes, in a grid-stride loop."""
    elements = tl.arange(0, BLOCK)
    last = tokens.to(tl.int64) * ROW
    first = tl.load(cu + sequences).to(tl.int64) * ROW + tl.program_id(0) * BLOCK
    for offset in range(first, last, tl.num_programs(0) * BLOCK):
        tl.store(rows + offset + elements, 0.0, offset + elements < last)


def _zero_tail_rows(rows: torch.Tensor, cu_seqlens: torch.Tensor) -> None:
    """Zero the token rows of ``rows`` past the stream end."""
    tokens = rows.shape[0]
    _zero_tail[(_num_sm(rows.device),)](
        rows, cu_seqlens, cu_seqlens.shape[0] - 1, tokens, rows[0].numel(), 1024
    )


def _entry_states(
    maps: torch.Tensor, state: torch.Tensor | None, launch: _Launch, sequences: int
) -> torch.Tensor:
    """Every piece's ``[pieces, H, V, K]`` entry state (exit cotangent) from its sequence's
    ``state`` through the piece maps: their bias rows alone when ``state`` is None."""
    pieces, heads, rows, dim_k = maps.shape
    dim_v = rows - dim_k
    if state is None:
        return maps[:, :, :dim_v]
    out = torch.empty(pieces, heads, dim_v, dim_k, dtype=torch.float32, device=maps.device)
    seq_block = triton.next_power_of_2(_num_sm(maps.device) // 2)
    assert sequences <= seq_block, "a split plan has at most half an SM count of sequences"
    block = 32
    _apply_maps[(pieces, heads, dim_v // block)](
        maps,
        state,
        out,
        launch.main_rows,
        sequences,
        state.stride(0),
        state.stride(1),
        state.stride(2),
        heads,
        dim_v,
        dim_k,
        seq_block,
        block,
        32,
    )
    return out


def _launch(
    cu_seqlens: torch.Tensor,
    shape: tuple[int, int, int],
    cost,
    chunk: int,
    tensors: tuple[torch.Tensor | None, ...],
) -> _Launch | None:
    """The split plan over every sequence of ``shape = (tokens, heads, V)`` (square ``V x V``
    states), None when the summaries do not split or the operands or the plan's own buffers
    (checkpoints, piece maps) need 64-bit offsets, which the standalone launches do not
    compile."""
    tokens, heads, dim_v = shape
    sequences = cu_seqlens.shape[0] - 1
    span = summary.summary_span(tokens, sequences, heads, cu_seqlens.device, b_t=chunk, cost=cost)
    if not span or requires_int64_abi(*tensors):
        return None
    pieces = _Launch.capacity(tokens, sequences, chunk, span)
    rows = max(tokens // chunk + pieces, 2 * pieces)  # checkpoints; [B; A] maps (V = K)
    if rows * heads * dim_v * dim_v > 2**31 - 1:
        return None
    return _Launch.prepare(cu_seqlens, tokens, heads, None, chunk, cost, destinations=True)


def _check_state(state: torch.Tensor | None, shape: tuple, device, name: str) -> None:
    """``state`` is None or an FP32 ``shape`` tensor on ``device`` with 16-byte aligned,
    unit-stride rows (``_normalize_warpdrive_state``)."""
    if state is None:
        return
    if tuple(state.shape) != shape or state.dtype != torch.float32:
        raise TypeError(f"{name} must be float32 with shape {shape}")
    if state.device != device or state.stride(-1) != 1 or state.data_ptr() % 16:
        raise ValueError(f"{name} must be on {device} with 16-byte aligned, unit-stride rows")


def _on_device(fn):
    """Launch on the first tensor argument's device."""

    @functools.wraps(fn)
    def wrapper(first: torch.Tensor, *args, **kwargs):
        with initialized_cuda_device(first):
            return fn(first, *args, **kwargs)

    return wrapper


def _scheduler(device, size: int = 1) -> torch.Tensor:
    """The zeroed claim counters of one persistent launch."""
    return torch.zeros(size, dtype=torch.int32, device=device)


def _start(pieces, maps, state, cu_seqlens, shape, device, name, store: bool):
    """Every piece's entry state (or exit cotangent) through ``maps`` from the sequences'
    ``state`` (``shape = (H, V, K)``), and the ``[N, H, V, K]`` buffer the sequences' last (or
    first) pieces store into when ``store``: every sequence, empty ones included, owns one."""
    sequences = cu_seqlens.shape[0] - 1
    _check_state(state, (sequences, *shape), device, name)
    entry = _entry_states(maps, state, pieces.launch, sequences)
    stored = torch.empty(sequences, *shape, dtype=torch.float32, device=device) if store else None
    return entry, stored


class ForwardPieces(NamedTuple):
    """The forward maps of one staged call; see NOTE [Staged piece plans]."""

    launch: _Launch
    prefix: torch.Tensor  # [pieces, H, V + K, K]: [B_<j; A_<j]
    maps: torch.Tensor  # [N, H, V + K, K]: each sequence's [B; A]
    tinv: torch.Tensor | None = None  # GDN: the pieces' chunk inverses


def _forward_pieces(
    launch: _Launch, sequences: int, heads: int, shape: tuple[int, int], device, summarize
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(prefix, maps)`` from a variant's ``summarize(H, M)`` over every piece."""
    dim_v, dim_k = shape
    bias = torch.empty(launch.map_rows, heads, dim_v, dim_k, dtype=torch.float32, device=device)
    transition = torch.empty(
        launch.map_rows, heads, dim_k, dim_k, dtype=torch.float32, device=device
    )
    summarize(bias, transition)
    prefix = torch.empty(
        launch.map_rows, heads, dim_v + dim_k, dim_k, dtype=torch.float32, device=device
    )
    maps = torch.empty(sequences, heads, dim_v + dim_k, dim_k, dtype=torch.float32, device=device)
    piece_chain.state_chain(
        bias, transition, prefix, launch.main_rows, rows=CHAIN_ROWS, final=maps
    )
    return prefix, maps


@_on_device
def forward_pieces(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    fastmath: bool,
) -> ForwardPieces | None:
    """Piece maps for packed ``[T, H, D]`` operands (natural-log FP32 channel gate, FP32 beta),
    None when the forward summaries do not split."""
    tokens, heads, dim_k = k.shape
    dim_v = v.shape[-1]
    launch = _launch(cu_seqlens, (tokens, heads, dim_v), KDA_FORWARD_COST, B_T, (k, v, gate, beta))
    if launch is None:
        return None

    def summarize(bias, transition) -> None:
        summary._forward_summary(k, v, gate, beta, bias, transition, launch, fastmath=fastmath)

    sequences = cu_seqlens.shape[0] - 1
    shape = (dim_v, dim_k)
    return ForwardPieces(
        launch, *_forward_pieces(launch, sequences, heads, shape, k.device, summarize)
    )


@_on_device
def forward_from_pieces(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pieces: ForwardPieces,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    fastmath: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The local forward over the pieces: ``(o, final_state)`` from FP32 ``[N, H, V, K]``
    entry states (zero when None)."""
    tokens, heads, dim_k = q.shape
    dim_v = v.shape[-1]
    states, final = _start(
        pieces, pieces.prefix, initial_state, cu_seqlens, (heads, dim_v, dim_k), q.device,
        "initial_state", output_final_state,
    )  # fmt: skip
    o = torch.empty(tokens, heads, dim_v, dtype=q.dtype, device=q.device)
    launch = pieces.launch
    op = kda_prefill.KdaPrefill(
        TORCH_NAMES[q.dtype],
        TORCH_NAMES[gate.dtype],
        dim_k,
        dim_v,
        1,
        True,
        output_final_state,
        0,
        _num_sm(q.device),
        fastmath,
    )
    fn = kda_prefill.compile_kda_prefill(op, TORCH_NAMES[beta.dtype], "fp32")
    fn(
        q, k, v, gate, beta, states, o, final, None, None, launch.work_items, launch.work_count,
        _scheduler(q.device), new_tma_slots(op.tma_slot_rows(heads), q.device), scale,
    )  # fmt: skip
    return o, final


class BackwardPieces(NamedTuple):
    """The reverse maps and checkpoints of one staged backward; see NOTE [Staged piece plans]."""

    launch: _Launch
    suffix: torch.Tensor  # [pieces, H, V + K, K]: [C_>j; R_>j]
    maps: torch.Tensor  # [N, H, V + K, K]: each sequence's [C; R]
    checkpoints: torch.Tensor
    tinv: torch.Tensor | None = None  # GDN: the pieces' chunk inverses


def _reverse_pieces(
    launch: _Launch,
    sequences: int,
    heads: int,
    shape: tuple[int, int],
    checkpoints: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    overlap: bool,
    forward_summary,
    reverse_summary,
    recompute,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(suffix, maps)`` from a variant's summaries over every piece; ``checkpoints`` filled.

    ``forward_summary(H, M)`` and ``reverse_summary(C)`` write the piece maps and
    ``recompute(X, checkpoints)`` the checkpoints from every piece's entry state. The reverse
    summary and its map chain run on a side stream beside the state chain and the recompute,
    and with ``overlap`` beside the forward summary too (``SummaryCost.kernels``)."""
    dim_v, dim_k = shape
    device = checkpoints.device

    def state(dim: int = dim_v) -> torch.Tensor:
        return torch.empty(launch.map_rows, heads, dim, dim_k, dtype=torch.float32, device=device)

    bias, transition, entry, states, suffix = (
        state(),
        state(dim_k),
        state(),
        state(),
        state(dim_v + dim_k),
    )
    maps = torch.empty(sequences, heads, dim_v + dim_k, dim_k, dtype=torch.float32, device=device)
    # Two forks, not fork_join_streams: the side's map chain also waits for the forward summary.
    current, side = torch.cuda.current_stream(device), summary._side_stream(device)
    if not overlap:
        forward_summary(bias, transition)
    side.wait_stream(current)
    with torch.cuda.stream(side):
        reverse_summary(entry)
    if overlap:
        forward_summary(bias, transition)
        side.wait_stream(current)
    with torch.cuda.stream(side):
        piece_chain.state_chain(
            entry, transition, suffix, launch.main_rows, transpose=True, rows=CHAIN_ROWS,
            final=maps,
        )  # fmt: skip
    piece_chain.state_chain(bias, transition, states, launch.main_rows, seed=initial_state)
    recompute(states, checkpoints)
    current.wait_stream(side)
    return suffix, maps


@_on_device
def backward_pieces(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    fastmath: bool,
) -> BackwardPieces | None:
    """Reverse maps and checkpoints for packed ``[T, H, D]`` operands from FP32 ``[N, H, V,
    K]`` entry states (zero when None), None when the reverse summaries do not split."""
    tokens, heads, dim_k = k.shape
    dim_v = v.shape[-1]
    device = k.device
    tensors = (q, k, v, gate, beta, d_output)
    launch = _launch(cu_seqlens, (tokens, heads, dim_v), KDA_REVERSE_COST, B_T, tensors)
    if launch is None:
        return None
    sequences = cu_seqlens.shape[0] - 1
    _check_state(initial_state, (sequences, heads, dim_v, dim_k), device, "initial_state")
    rows = kda_recompute.checkpoint_rows(tokens, launch.map_rows)
    checkpoints = torch.empty(rows, heads, dim_v, dim_k, dtype=k.dtype, device=device)

    def forward_summary(bias, transition) -> None:
        summary._forward_summary(k, v, gate, beta, bias, transition, launch, fastmath=fastmath)

    def reverse_summary(entry) -> None:
        summary._reverse_summary(
            q, k, gate, beta, d_output, cu_seqlens, scale, entry, None, launch, fastmath=fastmath
        )

    def recompute(states, checkpoints) -> None:
        op = kda_recompute.KdaRecompute(
            TORCH_NAMES[k.dtype], TORCH_NAMES[gate.dtype], dim_k, dim_v, True, _num_sm(device),
            fastmath,
        )  # fmt: skip
        kda_recompute.compile_kda_recompute(op, TORCH_NAMES[beta.dtype], "fp32")(
            k, v, gate, beta, states, checkpoints, launch.work_items, launch.work_count,
            _scheduler(device), new_tma_slots(op.tma_slot_rows(heads), device),
        )  # fmt: skip

    suffix, maps = _reverse_pieces(
        launch, sequences, heads, (dim_v, dim_k), checkpoints, initial_state,
        overlap=KDA_REVERSE_COST.kernels > 1, forward_summary=forward_summary,
        reverse_summary=reverse_summary, recompute=recompute,
    )  # fmt: skip
    return BackwardPieces(launch, suffix, maps, checkpoints)


@_on_device
def backward_from_pieces(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pieces: BackwardPieces,
    *,
    scale: float,
    d_final_state: torch.Tensor | None,
    d_initial_state: bool,
    fastmath: bool,
) -> tuple[torch.Tensor, ...]:
    """The local backward over the pieces: ``(dq, dk, dv, dgate, dbeta, d_initial_state)`` from
    FP32 ``[N, H, V, K]`` exit cotangents (zero when None); the last is None unless
    ``d_initial_state``."""
    heads, dim_k = q.shape[1:]
    dim_v = v.shape[-1]
    exits, dstate0 = _start(
        pieces, pieces.suffix, d_final_state, cu_seqlens, (heads, dim_v, dim_k), q.device,
        "d_final_state", d_initial_state,
    )  # fmt: skip
    dq, dk, dv, dgate, dbeta = (torch.empty_like(t) for t in (q, k, v, gate, beta))
    _zero_tail_rows(dgate, cu_seqlens)
    launch = pieces.launch
    op = kda_bprop.KdaBprop(
        TORCH_NAMES[q.dtype], TORCH_NAMES[gate.dtype], dim_k, dim_v, True, True, d_initial_state,
        _num_sm(q.device), fastmath,
    )  # fmt: skip
    kda_bprop.compile_kda_bprop(op, TORCH_NAMES[beta.dtype])(
        q, k, v, d_output, gate, beta, pieces.checkpoints, dq, dk, dv, dgate, dbeta, exits,
        dstate0, launch.work_items, launch.work_count, _scheduler(q.device),
        new_tma_slots(op.tma_slot_rows(heads), q.device), scale, None,
    )  # fmt: skip
    return dq, dk, dv, dgate, dbeta, dstate0


@_on_device
def gdn_forward_pieces(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> ForwardPieces | None:
    """:func:`forward_pieces` for GDN: ``k`` ``[T, HK, K]``, ``v`` ``[T, H, V]``, FP32
    natural-log ``gate`` and ``beta`` ``[T, H]``."""
    tokens, heads, dim_v = v.shape
    tensors = (k, v, gate, beta)
    launch = _launch(
        cu_seqlens, (tokens, heads, dim_v), GDN_FORWARD_COST, gdn_summary.B_T, tensors
    )
    if launch is None:
        return None
    tinv = summary._gdn_tinv(k, gate, beta, launch.cu_pieces)
    items = (launch.work_items, launch.work_count)

    def summarize(bias, transition) -> None:
        gdn_summary.gdn_summary(k, v, gate, tinv, bias, transition, *items)

    sequences = cu_seqlens.shape[0] - 1
    shape = (dim_v, k.shape[-1])
    prefix, maps = _forward_pieces(launch, sequences, heads, shape, k.device, summarize)
    return ForwardPieces(launch, prefix, maps, tinv)


@_on_device
def gdn_forward_from_pieces(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pieces: ForwardPieces,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """:func:`forward_from_pieces` for GDN."""
    tokens, heads, dim_v = v.shape
    dim_k = k.shape[-1]
    states, final = _start(
        pieces, pieces.prefix, initial_state, cu_seqlens, (heads, dim_v, dim_k), q.device,
        "initial_state", output_final_state,
    )  # fmt: skip
    o = torch.empty(tokens, heads, dim_v, dtype=q.dtype, device=q.device)
    launch = pieces.launch
    op = gdn_prefill.GdnPrefillGmem(
        TORCH_NAMES[q.dtype], "fp32", dim_k, dim_v, 1, True, output_final_state, False
    )
    _, sms = op.work.launch_args(q.device)
    gdn_prefill.compile_gdn_prefill(op)(
        q, k, v, gate, None, o, states, final, pieces.tinv, launch.work_items, launch.work_count,
        _scheduler(q.device, 2), new_tma_slots(op.tma_slot_rows(heads), q.device), scale, sms,
    )  # fmt: skip
    return o, final


@_on_device
def gdn_backward_pieces(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
) -> BackwardPieces | None:
    """:func:`backward_pieces` for GDN: ``q``/``k`` ``[T, HK, K]``, ``v``/``d_output`` ``[T, H,
    V]``, FP32 natural-log ``gate`` and ``beta`` ``[T, H]``."""
    tokens, heads, dim_v = v.shape
    dim_k = k.shape[-1]
    device = k.device
    tensors = (q, k, v, gate, beta, d_output)
    launch = _launch(
        cu_seqlens, (tokens, heads, dim_v), GDN_REVERSE_COST, gdn_summary.B_T, tensors
    )
    if launch is None:
        return None
    sequences = cu_seqlens.shape[0] - 1
    _check_state(initial_state, (sequences, heads, dim_v, dim_k), device, "initial_state")
    tinv = summary._gdn_tinv(k, gate, beta, launch.cu_pieces)
    rows = gdn_recompute.checkpoint_rows(tokens, launch.map_rows)
    checkpoints = torch.empty(rows, heads, dim_v, dim_k, dtype=k.dtype, device=device)
    items = (launch.work_items, launch.work_count)

    def forward_summary(bias, transition) -> None:
        gdn_summary.gdn_summary(k, v, gate, tinv, bias, transition, *items)

    def reverse_summary(entry) -> None:
        gdn_bprop_summary.gdn_bprop_summary(q, k, gate, d_output, tinv, entry, *items, scale=scale)

    def recompute(states, checkpoints) -> None:
        gdn_recompute.gdn_recompute(k, v, gate, tinv, *items, checkpoints, state_in=states)

    suffix, maps = _reverse_pieces(
        launch, sequences, heads, (dim_v, dim_k), checkpoints, initial_state,
        overlap=GDN_REVERSE_COST.kernels > 1, forward_summary=forward_summary,
        reverse_summary=reverse_summary, recompute=recompute,
    )  # fmt: skip
    return BackwardPieces(launch, suffix, maps, checkpoints, tinv)


@_on_device
def gdn_backward_from_pieces(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    pieces: BackwardPieces,
    *,
    scale: float,
    d_final_state: torch.Tensor | None,
    d_initial_state: bool,
) -> tuple[torch.Tensor, ...]:
    """:func:`backward_from_pieces` for GDN; ``dq``/``dk`` at the key head count (grouped heads
    reduced in a fixed order), FP32 ``dgate``/``dbeta``."""
    tokens, heads, dim_v = v.shape
    key_heads, dim_k = k.shape[1:]
    exits, dstate0 = _start(
        pieces, pieces.suffix, d_final_state, cu_seqlens, (heads, dim_v, dim_k), q.device,
        "d_final_state", d_initial_state,
    )  # fmt: skip
    dq, dk = (torch.empty(tokens, heads, dim_k, dtype=q.dtype, device=q.device) for _ in "qk")
    dv, dgate, dbeta = (torch.empty_like(t) for t in (v, gate, beta))
    _zero_tail_rows(dgate, cu_seqlens)
    launch = pieces.launch
    gdn_bprop.gdn_bprop(
        q, k, v, gate, beta, d_output, pieces.checkpoints, launch.work_items, launch.work_count,
        dq, dk, dv, dgate, dbeta, scale=scale, use_initial_state=True, dstate_in=exits,
        dstate0=dstate0,
    )  # fmt: skip
    groups = heads // key_heads
    if groups > 1:
        dq, dk = (group_sum(grad, groups, out_dtype=q.dtype) for grad in (dq, dk))
    return dq, dk, dv, dgate, dbeta, dstate0


def gather(pieces: ForwardPieces | BackwardPieces, cu_seqlens, bounds) -> torch.Tensor:
    """The summaries of ``bounds`` rows from the composed maps, a new tensor (see
    :func:`summary._summary_rows`)."""
    if bounds is None:
        return pieces.maps.clone()
    summary.validate_bounds(bounds, pieces.maps.device)
    with initialized_cuda_device(pieces.maps):
        return summary._summary_rows(pieces.maps, cu_seqlens, bounds)


__all__ = [
    "BackwardPieces",
    "ForwardPieces",
    "backward_from_pieces",
    "backward_pieces",
    "forward_from_pieces",
    "forward_pieces",
    "gather",
    "gdn_backward_from_pieces",
    "gdn_backward_pieces",
    "gdn_forward_from_pieces",
    "gdn_forward_pieces",
]
