# SPDX-License-Identifier: BSD-3-Clause

"""Staged KDA and GDN plans that run the local pass over the split summaries' pieces.

When the context-parallel summaries split (NOTE [Summary split cost] in ``plan.py``), their
piece maps already hold everything the local pass needs to start every piece at once. NOTE
[Staged piece plans]::

    forward:  plan    = the summaries' pieces (:func:`plan`), kept on the tape
              maps    = forward summary of every piece      ([H_j; M_j])
              prefix  = map chain, X_j = [B_<j; A_<j]        (composite: the summary)
              run     = e_j = seed @ A_<j + B_<j (e_0 = seed), then one prefill over the pieces
    backward: plan    = the tape's
              maps    = forward and reverse summaries       ([H_j; M_j], C_j)
              prefix  = the forward's, recomposed bit for bit; recompute every piece's
                        checkpoints from the forward's e_j
              suffix  = transposed map chain, [C_>j; R_>j]
              summary = R = A^T, C = C_0 + sum_j C_j @ A_<j^T
              run     = d_final @ R_>j + C_>j into one bprop over the pieces, and
                        d_seed = d_final @ R + C

The backward thus differentiates the forward that ran: the same pieces whatever its own options
or operands, checkpoints from the forward's entry states, and the seed cotangent through the
forward's prefix transposed (NOTE [Reverse maps are the forward's adjoint] in ``summary.py``).
A sequence's first piece starts from the seed itself, and an empty sequence passes its state and
cotangent through, bit for bit. The plan exists whenever the summaries split; the staged passes
run it only when their 32-bit launches can address the operands and the plan's buffers, which
the forward and the backward decide on the same tape operands.

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
from collections.abc import Callable
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
)
from ..kda_common import B_T, TORCH_NAMES
from ..ragged import new_tma_slots
from . import summary
from .plan import SummaryCost, _num_sm
from .summary import GDN_SUMMARY_COST, KDA_SUMMARY_COST, _ForwardMaps, _Launch


@triton.jit(do_not_specialize=["sequences", "heads", "pieces"])
def _apply_maps(
    maps,
    state,
    out,
    main_rows,
    sequence_maps,
    sequence_out,
    cu,
    pieces,
    sequences,
    heads,
    state_stride_seq,
    state_stride_head,
    state_stride_row,
    V: tl.constexpr,
    K: tl.constexpr,
    LAST: tl.constexpr,
    SEQUENCE_SLOTS: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    """``out[p, h] = state[s, h] @ maps[p, h, V:] + maps[p, h, :V]`` in 3xTF32 for every
    planned piece ``p < pieces`` of sequence ``s`` (``main_rows``, the ``piece_chain`` table);
    the sequence's first (``LAST``: last) piece, whose map is the identity, copies ``state[s,
    h]``. With ``SEQUENCE_SLOTS``, slot ``pieces + s`` does the same for sequence ``s`` with
    ``sequence_maps`` into ``sequence_out``, an empty sequence copying. ``SEQ_BLOCK`` is
    ``summary.split_sequence_block``."""
    slot, head = tl.program_id(0), tl.program_id(1)
    rows = tl.program_id(2) * BM + tl.arange(0, BM)
    cols = tl.arange(0, K)
    if SEQUENCE_SLOTS and slot >= pieces:
        seq = slot - pieces
        active = seq < sequences
        identity = tl.load(cu + seq) == tl.load(cu + seq + 1)
        index = seq.to(tl.int64) * heads + head
        base = sequence_maps + index * ((V + K) * K)
        target = sequence_out + index * (V * K)
    else:
        seqs = tl.arange(0, SEQ_BLOCK)
        starts = tl.load(main_rows + seqs, seqs < sequences, other=2**30)
        row = slot * heads
        active = row < tl.load(main_rows + sequences)
        seq = tl.sum((starts <= row).to(tl.int32), 0) - 1
        if LAST:
            identity = tl.load(main_rows + seq + 1) == row + heads
        else:
            identity = tl.load(main_rows + seq) == row
        index = slot.to(tl.int64) * heads + head
        base = maps + index * ((V + K) * K)
        target = out + index * (V * K)
    if active:
        source = state + seq.to(tl.int64) * state_stride_seq + head * state_stride_head
        tile = target + rows[:, None] * K + cols[None, :]
        if identity:
            tl.store(tile, tl.load(source + rows[:, None] * state_stride_row + cols[None, :]))
        else:
            acc = tl.load(base + rows[:, None] * K + cols[None, :])
            for kb in tl.static_range(0, K, BK):
                ks = kb + tl.arange(0, BK)
                a = tl.load(source + rows[:, None] * state_stride_row + ks[None, :])
                t = tl.load(base + (V + ks[:, None]) * K + cols[None, :])
                acc = tl.dot(a, t, acc, input_precision="tf32x3")
            tl.store(tile, acc)


@triton.jit(do_not_specialize=["sequences", "tokens", "row"])
def _zero_tail(rows, cu, sequences, tokens, row, BLOCK: tl.constexpr):
    """Zero ``rows`` (``tokens`` rows of ``row`` elements) past the stream end ``cu[sequences]``,
    which no piece writes, in a grid-stride loop."""
    elements = tl.arange(0, BLOCK)
    last = tokens.to(tl.int64) * row
    first = tl.load(cu + sequences).to(tl.int64) * row + tl.program_id(0) * BLOCK
    for offset in range(first, last, tl.num_programs(0) * BLOCK):
        tl.store(rows + offset + elements, 0.0, offset + elements < last)


def _zero_tail_rows(rows: torch.Tensor, cu_seqlens: torch.Tensor) -> None:
    """Zero the token rows of ``rows`` past the stream end."""
    tokens = rows.shape[0]
    _zero_tail[(_num_sm(rows.device),)](
        rows, cu_seqlens, cu_seqlens.shape[0] - 1, tokens, rows[0].numel(), 1024
    )


def _piece_states(
    launch: _Launch,
    maps: torch.Tensor,
    state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    name: str,
    *,
    last: bool = False,
    out: torch.Tensor | None = None,
    sequence_maps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Every piece's ``[pieces, H, V, K]`` entry state (``last``: exit cotangent) from its
    sequence's ``state`` through the piece ``maps`` (into ``out`` when given), their bias rows
    alone when ``state`` is None; the sequence's first (last) piece, whose map is the identity,
    takes ``state`` itself. With ``sequence_maps`` the same launch also applies each sequence's
    own map (an empty sequence's state passes through), else the second result is None."""
    pieces, heads, rows, dim_k = maps.shape
    dim_v = rows - dim_k
    sequences = cu_seqlens.shape[0] - 1
    _check_state(state, (sequences, heads, dim_v, dim_k), maps.device, name)
    if state is None:
        return maps[:, :, :dim_v], None if sequence_maps is None else sequence_maps[:, :, :dim_v]

    def fp32(slots: int) -> torch.Tensor:
        return torch.empty(slots, heads, dim_v, dim_k, dtype=torch.float32, device=maps.device)

    out = fp32(pieces) if out is None else out
    sequence_out = None if sequence_maps is None else fp32(sequences)
    slots = pieces if sequence_maps is None else pieces + sequences
    block = 32
    _apply_maps[(slots, heads, dim_v // block)](
        maps, state, out, launch.main_rows, sequence_maps, sequence_out, cu_seqlens, pieces,
        sequences, heads, *state.stride()[:3], dim_v, dim_k, last, sequence_maps is not None,
        summary.split_sequence_block(sequences, maps.device), block, 32,
    )  # fmt: skip
    return out, sequence_out


def _check_entry(
    initial_state: torch.Tensor | None, cu_seqlens: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> None:
    """Check a backward's entry states before it allocates or launches anything."""
    shape = (cu_seqlens.shape[0] - 1, v.shape[1], v.shape[2], k.shape[-1])
    _check_state(initial_state, shape, k.device, "initial_state")


def _sequence_states(
    cu_seqlens: torch.Tensor, heads: int, dim_v: int, dim_k: int, device: torch.device
) -> torch.Tensor:
    """The ``[N, H, V, K]`` buffer the sequences' last pieces store their final states into:
    every sequence, empty ones included, owns one."""
    sequences = cu_seqlens.shape[0] - 1
    return torch.empty(sequences, heads, dim_v, dim_k, dtype=torch.float32, device=device)


class _Variant(NamedTuple):
    """What a staged variant's plan depends on."""

    cost: SummaryCost
    chunk: int  # tokens per summary chunk
    checkpoint_rows: Callable[[int, int], int]  # (tokens, pieces) -> recompute checkpoint rows


KDA = _Variant(KDA_SUMMARY_COST, B_T, kda_recompute.checkpoint_rows)
GDN = _Variant(GDN_SUMMARY_COST, gdn_summary.B_T, gdn_recompute.checkpoint_rows)


def plan(variant: _Variant, v: torch.Tensor, cu_seqlens: torch.Tensor) -> _Launch | None:
    """The split plan of the summaries over every sequence of packed ``v`` ``[T, H, V]``, None
    when they run uncut. A forward resolves it once and its tape keeps it (:func:`plan_tables`,
    :func:`restore_plan`), so the backward's summaries and staged passes run the same pieces."""
    tokens, heads, _ = v.shape
    sequences = cu_seqlens.shape[0] - 1
    with initialized_cuda_device(v):
        device = cu_seqlens.device
        span = summary.summary_span(
            tokens, sequences, heads, device, b_t=variant.chunk, cost=variant.cost
        )
        if not span:
            return None
        return _Launch.prepare(
            cu_seqlens, tokens, heads, None, variant.chunk, variant.cost, destinations=True
        )


def plan_tables(launch: _Launch | None) -> tuple[torch.Tensor | None, ...]:
    """A tape's ``plan_work_items, plan_work_count, plan_cu_pieces, plan_main_rows``."""
    return (None,) * 4 if launch is None else launch.saved()


def restore_plan(tape: NamedTuple, *, batch_invariant: bool) -> _Launch | None:
    """The plan of a tape with :func:`plan_tables`' fields; a backward's ``batch_invariant``
    request must not contradict a split one."""
    if tape.plan_work_items is None:
        return None
    if batch_invariant:
        raise ValueError(
            "kernel_options['batch_invariant']=True does not match the saved tape, whose "
            "forward split its summaries"
        )
    return _Launch.restore(
        tape.plan_work_items, tape.plan_work_count, tape.plan_cu_pieces, tape.plan_main_rows
    )


def _launchable(variant: _Variant, launch: _Launch | None, *operands: torch.Tensor) -> bool:
    """Whether the staged passes can run the split ``launch`` over ``(q, k, v, gate, beta)``:
    their standalone launches compile 32-bit offsets only, for the operands and for the plan's
    own buffers (checkpoints, piece maps). Forward and backward decide on the same operands; a
    compact ``d_output`` shaped like ``v`` never needs more than ``v``."""
    if launch is None or requires_int64_abi(*operands):
        return False
    _, k, v = operands[:3]
    tokens, heads, dim_v = v.shape
    dim_k = k.shape[-1]
    checkpoints = variant.checkpoint_rows(tokens, launch.map_rows) * dim_v * dim_k
    maps = launch.map_rows * (dim_v + dim_k) * dim_k
    return max(checkpoints, maps) * heads <= 2**31 - 1


def _check_state(state: torch.Tensor | None, shape: tuple, device, name: str) -> None:
    """``state`` is None or an FP32 ``shape`` tensor on ``device`` with 16-byte aligned,
    unit-stride rows (``_normalize_warpdrive_state``)."""
    if state is None:
        return
    if tuple(state.shape) != shape or state.dtype != torch.float32:
        raise TypeError(f"{name} must be float32 with shape {shape}")
    if state.device != device or state.stride(-1) != 1 or state.data_ptr() % 16:
        raise ValueError(f"{name} must be on {device} with 16-byte aligned, unit-stride rows")


def _on_device(fn: Callable) -> Callable:
    """Launch on the first tensor argument's device."""

    @functools.wraps(fn)
    def wrapper(first: torch.Tensor, *args, **kwargs):
        with initialized_cuda_device(first):
            return fn(first, *args, **kwargs)

    return wrapper


def _scheduler(device, size: int = 1) -> torch.Tensor:
    """The zeroed claim counters of one persistent launch."""
    return torch.zeros(size, dtype=torch.int32, device=device)


class ForwardPieces(NamedTuple):
    """The forward maps of one staged call; see NOTE [Staged piece plans]."""

    launch: _Launch
    prefix: torch.Tensor  # [pieces, H, V + K, K]: [B_<j; A_<j]
    maps: torch.Tensor  # [N, H, V + K, K]: each sequence's [B; A]
    tinv: torch.Tensor | None = None  # GDN: the pieces' chunk inverses


def _forward_pieces(
    launch: _Launch,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    summarize: Callable[[torch.Tensor, torch.Tensor], None],
    *,
    fastmath: bool = True,
    tinv: torch.Tensor | None = None,
) -> ForwardPieces:
    """Summarize every piece with a variant's ``summarize(H, M)`` and compose the maps as the
    split summaries do."""
    heads, dim_v = v.shape[1:]
    dim_k = k.shape[-1]
    forward = _ForwardMaps.allocate(
        launch, cu_seqlens.shape[0] - 1, heads, dim_v, dim_k, k.device, fastmath=fastmath
    )
    summarize(forward.bias, forward.transition)
    prefix = torch.empty(
        launch.map_rows, heads, dim_v + dim_k, dim_k, dtype=torch.float32, device=k.device
    )
    forward.compose(launch, prefix=prefix)
    return ForwardPieces(launch, prefix, forward.maps, tinv)


@_on_device
def forward_pieces(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    launch: _Launch | None,
    *,
    fastmath: bool,
) -> ForwardPieces | None:
    """Piece maps of the forward's ``launch`` (:func:`plan`) for packed ``[T, H, D]`` operands
    (natural-log FP32 channel gate, FP32 beta), None when the staged passes cannot run it."""
    if not _launchable(KDA, launch, q, k, v, gate, beta):
        return None

    def summarize(bias: torch.Tensor, transition: torch.Tensor) -> None:
        summary._forward_summary(k, v, gate, beta, bias, transition, launch, fastmath=fastmath)

    return _forward_pieces(launch, k, v, cu_seqlens, summarize, fastmath=fastmath)


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
    launch = pieces.launch
    states, _ = _piece_states(launch, pieces.prefix, initial_state, cu_seqlens, "initial_state")
    final = None
    if output_final_state:
        final = _sequence_states(cu_seqlens, heads, dim_v, dim_k, q.device)
    o = torch.empty(tokens, heads, dim_v, dtype=q.dtype, device=q.device)
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


def _backward_pieces(
    launch: _Launch,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor | None,
    checkpoint_rows: int,
    *,
    forward_summary: Callable[[torch.Tensor, torch.Tensor], None],
    reverse_summary: Callable[[torch.Tensor], None],
    recompute: Callable[[torch.Tensor, torch.Tensor], None],
    fastmath: bool = True,
    tinv: torch.Tensor | None = None,
) -> BackwardPieces:
    """The reverse maps of a variant's summaries over every piece (``summary._reverse_maps``),
    and the checkpoints ``recompute(states, checkpoints)`` writes from every piece's entry
    state: the forward's bits, from the same summaries, composition and seed application.
    ``initial_state`` was checked by :func:`_check_entry`."""
    heads, dim_v = v.shape[1:]
    dim_k = k.shape[-1]
    device = k.device
    shape = (cu_seqlens.shape[0] - 1, heads, dim_v, dim_k)
    checkpoints = torch.empty(checkpoint_rows, heads, dim_v, dim_k, dtype=k.dtype, device=device)
    forward = _ForwardMaps.allocate(
        launch, shape[0], heads, dim_v, dim_k, device, fastmath=fastmath
    )
    entries = None
    if initial_state is not None:
        entries = torch.empty(launch.map_rows, *shape[1:], dtype=torch.float32, device=device)

    def local(prefix: torch.Tensor) -> None:
        states, _ = _piece_states(
            launch, prefix, initial_state, cu_seqlens, "initial_state", out=entries
        )
        recompute(states, checkpoints)

    maps, suffix = summary._reverse_maps(
        launch, forward, forward_summary, reverse_summary, local=local
    )
    return BackwardPieces(launch, suffix, maps, checkpoints, tinv)


@_on_device
def backward_pieces(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    launch: _Launch | None,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
    fastmath: bool,
) -> BackwardPieces | None:
    """Reverse maps and checkpoints of the forward's ``launch`` for packed ``[T, H, D]``
    operands from FP32 ``[N, H, V, K]`` entry states (zero when None), None when the staged
    passes cannot run it (as the forward decided)."""
    if not _launchable(KDA, launch, q, k, v, gate, beta):
        return None
    assert not requires_int64_abi(d_output)
    _check_entry(initial_state, cu_seqlens, k, v)
    tokens, heads, dim_k = k.shape
    dim_v = v.shape[-1]
    device = k.device

    def forward_summary(bias: torch.Tensor, transition: torch.Tensor) -> None:
        summary._forward_summary(k, v, gate, beta, bias, transition, launch, fastmath=fastmath)

    def reverse_summary(entry: torch.Tensor) -> None:
        summary._reverse_summary(
            q, k, gate, beta, d_output, cu_seqlens, scale, entry, None, launch, fastmath=fastmath
        )

    def recompute(states: torch.Tensor, checkpoints: torch.Tensor) -> None:
        op = kda_recompute.KdaRecompute(
            TORCH_NAMES[k.dtype], TORCH_NAMES[gate.dtype], dim_k, dim_v, True, _num_sm(device),
            fastmath,
        )  # fmt: skip
        kda_recompute.compile_kda_recompute(op, TORCH_NAMES[beta.dtype], "fp32")(
            k, v, gate, beta, states, checkpoints, launch.work_items, launch.work_count,
            _scheduler(device), new_tma_slots(op.tma_slot_rows(heads), device),
        )  # fmt: skip

    return _backward_pieces(
        launch, k, v, cu_seqlens, initial_state, KDA.checkpoint_rows(tokens, launch.map_rows),
        forward_summary=forward_summary, reverse_summary=reverse_summary, recompute=recompute,
        fastmath=fastmath,
    )  # fmt: skip


def _exit_cotangents(
    cu_seqlens: torch.Tensor,
    pieces: BackwardPieces,
    d_final_state: torch.Tensor | None,
    d_initial_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Every piece's exit cotangent through its suffix map, and, when ``d_initial_state``, each
    sequence's entry cotangent ``d_final_state @ R + C`` through its summary (an empty
    sequence's is ``d_final_state`` itself): the adjoint of the forward's seed application
    (NOTE [Reverse maps are the forward's adjoint])."""
    exits, entry = _piece_states(
        pieces.launch, pieces.suffix, d_final_state, cu_seqlens, "d_final_state", last=True,
        sequence_maps=pieces.maps if d_initial_state else None,
    )  # fmt: skip
    if d_final_state is None and entry is not None:
        entry = entry.clone()  # C itself, not a view of the handle's maps
    return exits, entry


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
    launch = pieces.launch
    exits, dstate0 = _exit_cotangents(cu_seqlens, pieces, d_final_state, d_initial_state)
    dq, dk, dv, dgate, dbeta = (torch.empty_like(t) for t in (q, k, v, gate, beta))
    _zero_tail_rows(dgate, cu_seqlens)
    op = kda_bprop.KdaBprop(
        TORCH_NAMES[q.dtype], TORCH_NAMES[gate.dtype], dim_k, dim_v, True, True, False,
        _num_sm(q.device), fastmath,
    )  # fmt: skip
    kda_bprop.compile_kda_bprop(op, TORCH_NAMES[beta.dtype])(
        q, k, v, d_output, gate, beta, pieces.checkpoints, dq, dk, dv, dgate, dbeta, exits,
        None, launch.work_items, launch.work_count, _scheduler(q.device),
        new_tma_slots(op.tma_slot_rows(heads), q.device), scale, None,
    )  # fmt: skip
    return dq, dk, dv, dgate, dbeta, dstate0


@_on_device
def gdn_forward_pieces(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    launch: _Launch | None,
) -> ForwardPieces | None:
    """:func:`forward_pieces` for GDN: ``q``/``k`` ``[T, HK, K]``, ``v`` ``[T, H, V]``, FP32
    natural-log ``gate`` and ``beta`` ``[T, H]``."""
    if not _launchable(GDN, launch, q, k, v, gate, beta):
        return None
    tinv = summary._gdn_tinv(k, gate, beta, launch.cu_pieces)
    items = (launch.work_items, launch.work_count)

    def summarize(bias: torch.Tensor, transition: torch.Tensor) -> None:
        gdn_summary.gdn_summary(k, v, gate, tinv, bias, transition, *items)

    return _forward_pieces(launch, k, v, cu_seqlens, summarize, tinv=tinv)


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
    launch = pieces.launch
    states, _ = _piece_states(launch, pieces.prefix, initial_state, cu_seqlens, "initial_state")
    final = None
    if output_final_state:
        final = _sequence_states(cu_seqlens, heads, dim_v, dim_k, q.device)
    o = torch.empty(tokens, heads, dim_v, dtype=q.dtype, device=q.device)
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
    launch: _Launch | None,
    *,
    scale: float,
    initial_state: torch.Tensor | None,
) -> BackwardPieces | None:
    """:func:`backward_pieces` for GDN: ``q``/``k`` ``[T, HK, K]``, ``v``/``d_output`` ``[T, H,
    V]``, FP32 natural-log ``gate`` and ``beta`` ``[T, H]``."""
    if not _launchable(GDN, launch, q, k, v, gate, beta):
        return None
    assert not requires_int64_abi(d_output)
    _check_entry(initial_state, cu_seqlens, k, v)
    tinv = summary._gdn_tinv(k, gate, beta, launch.cu_pieces)
    items = (launch.work_items, launch.work_count)

    def forward_summary(bias: torch.Tensor, transition: torch.Tensor) -> None:
        gdn_summary.gdn_summary(k, v, gate, tinv, bias, transition, *items)

    def reverse_summary(entry: torch.Tensor) -> None:
        gdn_bprop_summary.gdn_bprop_summary(q, k, gate, d_output, tinv, entry, *items, scale=scale)

    def recompute(states: torch.Tensor, checkpoints: torch.Tensor) -> None:
        gdn_recompute.gdn_recompute(k, v, gate, tinv, *items, checkpoints, state_in=states)

    return _backward_pieces(
        launch, k, v, cu_seqlens, initial_state, GDN.checkpoint_rows(v.shape[0], launch.map_rows),
        forward_summary=forward_summary, reverse_summary=reverse_summary, recompute=recompute,
        tinv=tinv,
    )  # fmt: skip


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
    tokens, heads = v.shape[:2]
    key_heads, dim_k = k.shape[1:]
    launch = pieces.launch
    exits, dstate0 = _exit_cotangents(cu_seqlens, pieces, d_final_state, d_initial_state)
    dq, dk = (torch.empty(tokens, heads, dim_k, dtype=q.dtype, device=q.device) for _ in "qk")
    dv, dgate, dbeta = (torch.empty_like(t) for t in (v, gate, beta))
    _zero_tail_rows(dgate, cu_seqlens)
    gdn_bprop.gdn_bprop(
        q, k, v, gate, beta, d_output, pieces.checkpoints, launch.work_items, launch.work_count,
        dq, dk, dv, dgate, dbeta, scale=scale, use_initial_state=True, dstate_in=exits,
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
    "plan",
    "plan_tables",
    "restore_plan",
]
