# SPDX-License-Identifier: BSD-3-Clause

"""Native v1.30 affine state summaries for context-parallel KDA and GDN.

Forward maps pack ``[B; A]`` for ``exit = entry @ A + B``; reverse maps pack ``[C; R]`` for
``d_entry = d_exit @ R + C``. One ``kda_summary`` launch writes both B (the zero-seeded final
state H) and A (the stored-domain transition ``M_buf``); one ``kda_bprop_summary`` launch writes
C (the entry cotangent under a zero exit cotangent). GDN does the same in its own 64-token chunks
with ``gdn_summary`` and ``gdn_bprop_summary``, after one ``gdn_tinv`` pass for the chunk
inverses both read.

Each call either runs every sequence uncut or splits them (``plan.summary_span``, NOTE [Summary
split cost]). Uncut, one work item per (sequence, head) walks the whole sequence, so the
arithmetic is independent of sequence ownership and packed token count. Split, a planner cuts
every selected sequence into pieces of ``span`` chunks (the last one shorter), the summary
kernels write one map per (piece, head), and ``piece_chain.state_chain`` composes each
sequence's piece maps in order (``[B; A] = [0; I]`` chained through every piece, reversed for
``[C; R]``) in 3xTF32. ``span`` follows the call's shapes and SM count, so a sequence's bits
depend on its own tokens and those, not on the other sequences or the bounds.

Bounds select whole packed sequences on device: the selector (uncut) or the planner (split)
writes work items only for requested sequences, and a gather maps each requested bound to its
sequence-indexed result. Bounds values may change between CUDA Graph replays; nothing here
synchronizes with the host.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from attn_gym._backends.cute.utils import initialized_cuda_device
from attn_gym.linear._delta_rule.wd_support import HEAD_DIMS
from attn_gym.utils import cdiv, fork_join_streams

from .. import (
    gdn_bprop_summary,
    gdn_summary,
    gdn_tinv,
    kda_bprop_summary,
    kda_common,
    kda_summary,
    piece_chain,
)
from ..work_tables import WORK_ITEM_FIELDS
from .plan import SummaryCost, int32, summary_span

# Staged tensors: natural-log FP32 gate (per channel for KDA, per head for GDN), FP32
# post-sigmoid beta, normalized keys.


@triton.jit(debug=True, do_not_specialize=["rows", "tokens"])
def _select_summary_work(
    cu,
    bounds,
    work,
    count,
    rows,
    tokens,
    HEADS: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    FIELDS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Write one uncut native work item per sequence/head, zero-width unless a bound selects it.

    ``rows == 0`` selects every sequence; ``ROW_BLOCK`` is a power of two covering ``rows``, the
    only shape the kernel specializes on besides the summary kernel's ``CHUNK`` size. Row layout
    follows ``work_tables.py``: ``[seq, head, write_start, write_end, compute_start, compute_end,
    token_start, token_stop, final_dst, dstate_dst]``. Like the other warpdrive prologues
    (``check_packed_offsets``), it aborts with ``invalid packed cu_seqlens`` unless the offsets
    start at zero, never decrease, and end within the ``tokens`` packed tokens, so the summary
    kernels never address through them.
    """
    seq = tl.program_id(0)
    head = tl.program_id(1)
    start, stop = tl.load(cu + seq), tl.load(cu + seq + 1)
    valid = (start <= stop) & ((seq != 0) | (start == 0))
    valid &= (seq != tl.num_programs(0) - 1) | (stop <= tokens)
    tl.device_assert(valid, "invalid packed cu_seqlens")
    bound = tl.arange(0, ROW_BLOCK)
    lo = tl.load(bounds + 2 * bound, bound < rows, other=-1)
    hi = tl.load(bounds + 2 * bound + 1, bound < rows, other=-1)
    matched = tl.sum(((lo == start) & (hi == stop) & (lo < hi)).to(tl.int32), 0) > 0
    stop = tl.where((rows == 0) | matched, stop, start)
    chunks = tl.cdiv(stop - start, CHUNK)
    fields = tl.arange(0, 16)
    row = tl.where((fields == 0) | (fields >= 8), seq, tl.where(fields == 1, head, 0))
    row = tl.where((fields == 3) | (fields == 5), chunks, row)
    row = tl.where(fields == 6, start, tl.where(fields == 7, stop, row))
    tl.store(work + (seq * HEADS + head) * FIELDS + fields, row, fields < FIELDS)
    if (seq == 0) & (head == 0):
        tl.store(count, tl.num_programs(0) * HEADS)


@triton.jit(debug=True, do_not_specialize=["sequences", "rows", "tokens", "span", "capacity"])
def _plan_summary_pieces(
    cu,
    bounds,
    cu_pieces,
    main_rows,
    work,
    count,
    sequences,
    rows,
    tokens,
    span,
    capacity,
    HEADS: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    PIECE_BLOCK: tl.constexpr,
    FIELDS: tl.constexpr,
    CHUNK: tl.constexpr,
):
    """Cut every selected sequence into pieces of ``span`` chunks, the last one shorter.

    One program walks the sequences in order. ``cu_pieces[: capacity + 1]`` gets the flat piece
    starts padded with the stream end, so the pieces tile the stream like a ``cu_seqlens`` (an
    unselected or empty sequence is one piece); sequence ``s`` owns piece rows ``main_rows[s] ..
    main_rows[s + 1]`` (pieces times ``HEADS``, the ``piece_chain`` table). Work items, one per
    (selected piece, head) and named by piece, go to the first ``count[0]`` rows of ``work``.
    ``ROW_BLOCK`` and ``PIECE_BLOCK`` are powers of two covering ``rows`` and any sequence's
    pieces. The offsets are checked as in :func:`_select_summary_work`.
    """
    bound = tl.arange(0, ROW_BLOCK)
    lo = tl.load(bounds + 2 * bound, bound < rows, other=-1)
    hi = tl.load(bounds + 2 * bound + 1, bound < rows, other=-1)
    pieces = tl.arange(0, PIECE_BLOCK)
    fields = tl.arange(0, 16)
    piece = 0
    item = 0
    for seq in range(sequences):
        start, stop = tl.load(cu + seq), tl.load(cu + seq + 1)
        valid = (start <= stop) & ((seq != 0) | (start == 0))
        valid &= (seq != sequences - 1) | (stop <= tokens)
        tl.device_assert(valid, "invalid packed cu_seqlens")
        matched = tl.sum(((lo == start) & (hi == stop) & (lo < hi)).to(tl.int32), 0) > 0
        selected = (rows == 0) | matched
        chunks = tl.cdiv(stop - start, CHUNK)
        count_here = tl.where(selected, tl.maximum(tl.cdiv(chunks, span), 1), 1)
        piece_start = tl.minimum(start + pieces.to(tl.int64) * span * CHUNK, stop)
        piece_stop = tl.minimum(piece_start + span * CHUNK, stop)
        own = pieces < count_here
        tl.store(cu_pieces + piece + pieces, piece_start.to(tl.int32), own)
        tl.store(main_rows + seq, piece * HEADS)
        piece_chunks = tl.cdiv(piece_stop - piece_start, CHUNK)
        index = (piece + pieces)[:, None]
        row = tl.where((fields == 0) | (fields >= 8), index, 0)
        row = tl.where((fields == 3) | (fields == 5), piece_chunks[:, None], row)
        row = tl.where(fields == 6, piece_start[:, None], row)
        row = tl.where(fields == 7, piece_stop[:, None], row).to(tl.int32)
        mask = (own & selected)[:, None] & (fields < FIELDS)
        for head in range(HEADS):
            head_row = tl.where(fields == 1, head, row)
            offsets = (item + pieces[:, None] * HEADS + head) * FIELDS + fields
            tl.store(work + offsets, head_row, mask)
        item += tl.where(selected, count_here * HEADS, 0)
        piece += count_here
    tl.store(main_rows + sequences, piece * HEADS)
    end = tl.load(cu + sequences)
    for tail in range(piece, capacity + 1, PIECE_BLOCK):
        tl.store(cu_pieces + tail + pieces, end, tail + pieces <= capacity)
    tl.store(count, item)


@triton.jit(do_not_specialize=["sequences"])
def _gather_summary_rows(
    maps,
    cu,
    bounds,
    output,
    sequences,
    HEADS: tl.constexpr,
    SEQ_BLOCK: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Map each subsequence bound to its summary, or synthesize the empty identity.

    ``SEQ_BLOCK`` is a power of two covering ``sequences``, the only shape the kernel specializes
    on. A nonempty row that is not a subsequence (NOTE [Summary ranges are subsequences] in
    ``attn_gym.linear.kda.stages``) is filled with NaN so the mistake surfaces at the first
    compose instead of silently using another subsequence's map.
    """
    row, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    start, stop = tl.load(bounds + row * 2), tl.load(bounds + row * 2 + 1)
    seqs = tl.arange(0, SEQ_BLOCK)
    lo = tl.load(cu + seqs, seqs < sequences, other=-1)
    hi = tl.load(cu + seqs + 1, seqs < sequences, other=-1)
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
        _gather_summary_rows[(bounds.shape[0], heads, cdiv(rows * k, block))](
            maps,
            cu_seqlens,
            bounds,
            selected,
            sequences,
            heads,
            triton.next_power_of_2(sequences),
            rows - k,
            k,
            block,
        )
    return selected


@dataclass(frozen=True)
class _Launch:
    """Device-owned scheduling buffers for one selected summary launch: uncut, or split into
    pieces whose maps the chain composes through ``main_rows``."""

    work_items: torch.Tensor
    work_count: torch.Tensor
    counters: torch.Tensor
    map_rows: int  # rows of the kernels' maps: pieces when split, else sequences
    cu_pieces: torch.Tensor | None = None  # split: the pieces as packed offsets
    main_rows: torch.Tensor | None = None  # split: each sequence's piece rows

    @property
    def split(self) -> bool:
        return self.cu_pieces is not None

    @classmethod
    def prepare(
        cls,
        cu_seqlens: torch.Tensor,
        tokens: int,
        heads: int,
        bounds: torch.Tensor | None,
        chunk: int,
        cost: SummaryCost | None,
    ) -> _Launch:
        device = cu_seqlens.device
        sequences = cu_seqlens.shape[0] - 1
        rows = 0 if bounds is None else bounds.shape[0]
        row_block = max(1, triton.next_power_of_2(rows))
        span = summary_span(tokens, sequences, heads, device, b_t=chunk, cost=cost) if cost else 0
        work_count = int32(1, device)
        if not span:
            work_items = torch.empty(
                sequences * heads, WORK_ITEM_FIELDS, dtype=torch.int32, device=device
            )
            _select_summary_work[(sequences, heads)](
                cu_seqlens,
                work_count if bounds is None else bounds,
                work_items,
                work_count,
                rows,
                tokens,
                heads,
                row_block,
                WORK_ITEM_FIELDS,
                chunk,
            )
            return cls(work_items, work_count, int32(2, device), sequences)
        # Sequence s has at most ceil(chunks_s / span) pieces, and the chunks add up to at most
        # ceil(tokens / chunk) + sequences (each sequence may end in a partial chunk).
        per_sequence = cdiv(cdiv(tokens, chunk), span)
        capacity = cdiv(cdiv(tokens, chunk) + sequences, span) + sequences
        cu_pieces = int32(capacity + 1, device)
        main_rows = int32(sequences + 1, device)
        work_items = torch.empty(
            capacity * heads, WORK_ITEM_FIELDS, dtype=torch.int32, device=device
        )
        _plan_summary_pieces[(1,)](
            cu_seqlens,
            work_count if bounds is None else bounds,
            cu_pieces,
            main_rows,
            work_items,
            work_count,
            sequences,
            rows,
            tokens,
            span,
            capacity,
            heads,
            row_block,
            triton.next_power_of_2(per_sequence),
            WORK_ITEM_FIELDS,
            chunk,
        )
        return cls(work_items, work_count, int32(2, device), capacity, cu_pieces, main_rows)

    def reset(self) -> None:
        """The work-stealing counters must be zero before every launch, including replays."""
        self.counters.zero_()


def _forward_summary(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    transition: torch.Tensor,
    launch: _Launch,
    *,
    fastmath: bool,
) -> None:
    """Write H into ``state`` and stored-domain M into ``transition`` for every work item."""
    launch.reset()
    kda_summary.kda_summary(
        k,
        v,
        gate,
        beta,
        state,
        transition,
        launch.work_items,
        launch.work_count,
        launch.counters,
        fastmath=fastmath,
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
    *,
    fastmath: bool,
) -> None:
    """Write the entry cotangent for every work item; ``d_final_state`` None seeds zero."""
    kda_bprop_summary.kda_bprop_summary(
        q,
        k,
        gate,
        beta,
        d_output,
        d_initial_state,
        launch.work_items,
        launch.work_count,
        scale=scale,
        d_final_state=d_final_state,
        fastmath=fastmath,
    )


def _validate(tensors: dict[str, torch.Tensor], cu_seqlens: torch.Tensor, bounds) -> None:
    k = tensors["k"]
    if k.ndim != 4 or k.shape[0] != 1 or k.shape[-1] not in HEAD_DIMS:
        raise ValueError(f"k must have shape [1, T, H, K] with K in {HEAD_DIMS}")
    for name in ("q", "v", "d_output"):
        if name in tensors and tensors[name].shape[:3] != k.shape[:3]:
            raise ValueError(f"{name} must match k's token and head extents")
    if "v" in tensors and tensors["v"].shape[-1] not in HEAD_DIMS:
        raise ValueError(f"v must have V in {HEAD_DIMS}")
    if tensors["gate"].shape != k.shape or tensors["beta"].shape != k.shape[:3]:
        raise ValueError("gate and beta must match k's token and head extents")
    _validate_operands(tensors, cu_seqlens, bounds)


def _validate_gdn(tensors: dict[str, torch.Tensor], cu_seqlens: torch.Tensor, bounds) -> None:
    """GDN shapes: ``q``/``k`` ``[1, T, HK, K]``, ``v``/``d_output`` ``[1, T, H, V]`` with ``HK``
    dividing ``H``, and per-head ``gate``/``beta`` ``[1, T, H]``."""
    k, v = tensors["k"], tensors["v"]
    if k.ndim != 4 or k.shape[0] != 1 or k.shape[-1] not in HEAD_DIMS:
        raise ValueError(f"k must have shape [1, T, HK, K] with K in {HEAD_DIMS}")
    if v.ndim != 4 or v.shape[:2] != k.shape[:2] or v.shape[-1] not in HEAD_DIMS:
        raise ValueError(f"v must have shape [1, T, H, V] with V in {HEAD_DIMS}")
    if v.shape[2] % k.shape[2]:
        raise ValueError("the value head count must be a multiple of the key head count")
    if "q" in tensors and tensors["q"].shape != k.shape:
        raise ValueError("q must match k's shape")
    if "d_output" in tensors and tensors["d_output"].shape != v.shape:
        raise ValueError("d_output must match v's shape")
    if tensors["gate"].shape != v.shape[:3] or tensors["beta"].shape != v.shape[:3]:
        raise ValueError("gate and beta must match v's token and head extents")
    _validate_operands(tensors, cu_seqlens, bounds)


def _validate_operands(tensors: dict[str, torch.Tensor], cu_seqlens: torch.Tensor, bounds) -> None:
    """Dtype, device and ``bounds`` checks shared by both variants."""
    k = tensors["k"]
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


# Split costs per variant and direction (NOTE [Summary split cost]). A reverse step pays both
# summary kernels, which share one composing chain; GDN runs the two concurrently.
KDA_FORWARD_COST = SummaryCost(compose=2.0, overhead=3.0, fill=2.0)
KDA_REVERSE_COST = SummaryCost(compose=0.8, overhead=2.0, fill=0.0)
GDN_FORWARD_COST = SummaryCost(compose=1.8, overhead=2.0, fill=3.0)
GDN_REVERSE_COST = SummaryCost(compose=1.0, overhead=2.0, fill=0.75, kernels=2)
# The composing chain is latency bound, so its finest tensor-core blocks run fastest.
CHAIN_ROWS = 16


def _compose(
    bias: torch.Tensor,
    transition: torch.Tensor,
    launch: _Launch,
    sequences: int,
    *,
    reverse: bool,
) -> torch.Tensor:
    """Each sequence's piece maps (bias ``[pieces, H, V, K]``, transition ``[pieces, H, K, K]``)
    composed into its ``[S, H, V + K, K]`` map: ``[B; A]`` through the pieces in order, or with
    ``reverse`` ``[C; R]`` through their transposes last piece first (``R = A.T``)."""
    pieces, heads, dim_v, dim_k = bias.shape
    maps = torch.empty(
        sequences, heads, dim_v + dim_k, dim_k, dtype=torch.float32, device=bias.device
    )
    piece_chain.state_chain(
        bias,
        transition,
        torch.empty(  # unused
            pieces, heads, dim_v + dim_k, dim_k, dtype=torch.float32, device=bias.device
        ),
        launch.main_rows,
        transpose=reverse,
        rows=CHAIN_ROWS,
        final=maps,
    )
    return maps


def build_state_summaries(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    bounds: torch.Tensor | None = None,
    fastmath: bool = True,
) -> torch.Tensor:
    """Return FP32 ``[R, H, V + K, K]`` maps packed as ``[B; A]`` for whole packed sequences.

    Inputs use the staged ``[1, T, H, D]`` layout with FP32 natural-log gate and FP32 beta.
    Optional contiguous int32 ``bounds[R, 2]`` on ``k.device`` select, reorder, or duplicate
    consecutive ``cu_seqlens`` pairs; without bounds one row per sequence is returned. Empty
    intervals yield ``[0; I]``, nonempty rows that are not a sequence yield NaN, and zero rows
    yield no maps. Bounds values stay on device and may change during CUDA Graph replay.
    ``fastmath=False`` selects the non-flushing gate exponentials.
    """
    _validate({"k": k, "v": v, "gate": gate, "beta": beta}, cu_seqlens, bounds)
    with initialized_cuda_device(k):
        sequences, heads = cu_seqlens.shape[0] - 1, k.shape[2]
        dim_v, dim_k = v.shape[-1], k.shape[-1]
        if not _launches(k.shape[1], bounds):
            maps = _identity_maps(sequences, heads, dim_v, dim_k, k.device)
            return _summary_rows(maps, cu_seqlens, bounds)
        launch = _Launch.prepare(
            cu_seqlens, k.shape[1], heads, bounds, kda_common.B_T, KDA_FORWARD_COST
        )
        inputs = (k[0], v[0], gate[0], beta[0])
        if launch.split:
            bias = torch.empty(
                launch.map_rows, heads, dim_v, dim_k, dtype=torch.float32, device=k.device
            )
            transition = torch.empty(
                launch.map_rows, heads, dim_k, dim_k, dtype=torch.float32, device=k.device
            )
            _forward_summary(*inputs, bias, transition, launch, fastmath=fastmath)
            maps = _compose(bias, transition, launch, sequences, reverse=False)
        else:
            maps = _identity_maps(sequences, heads, dim_v, dim_k, k.device)
            _forward_summary(
                *inputs, maps[:, :, :dim_v], maps[:, :, dim_v:], launch, fastmath=fastmath
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
    fastmath: bool = True,
) -> torch.Tensor:
    """Return FP32 ``[R, H, V + K, K]`` maps packed as ``[C; R]`` for whole packed sequences.

    C is the native entry cotangent under a zero exit cotangent. R is ``A.T`` from the forward
    summary (the exact adjoint of right-multiplication by A) or, with
    ``transpose_forward_transition=False``, a second reverse launch seeded with the identity exit
    cotangent and zero ``d_output`` (requires ``V == K``; never split). Inputs, bounds and
    ``fastmath`` follow :func:`build_state_summaries`.
    """
    _validate(
        {"q": q, "k": k, "v": v, "gate": gate, "beta": beta, "d_output": d_output},
        cu_seqlens,
        bounds,
    )
    with initialized_cuda_device(q):
        sequences, heads = cu_seqlens.shape[0] - 1, q.shape[2]
        dim_v, dim_k = d_output.shape[-1], k.shape[-1]
        if not _launches(q.shape[1], bounds):
            maps = _identity_maps(sequences, heads, dim_v, dim_k, q.device)
            return _summary_rows(maps, cu_seqlens, bounds)
        if not transpose_forward_transition and dim_v != dim_k:
            raise ValueError("the identity reverse probe requires V == K")
        cost = KDA_REVERSE_COST if transpose_forward_transition else None
        launch = _Launch.prepare(cu_seqlens, q.shape[1], heads, bounds, kda_common.B_T, cost)
        rows = launch.map_rows
        if launch.split:
            maps = torch.empty(  # the pieces' C
                rows, heads, dim_v, dim_k, dtype=torch.float32, device=q.device
            )
        else:
            maps = _identity_maps(sequences, heads, dim_v, dim_k, q.device)
        if transpose_forward_transition:
            state = torch.empty(  # unused
                rows, heads, dim_v, dim_k, dtype=torch.float32, device=q.device
            )
            transition = torch.empty(
                rows, heads, dim_k, dim_k, dtype=torch.float32, device=q.device
            )
            _forward_summary(
                k[0], v[0], gate[0], beta[0], state, transition, launch, fastmath=fastmath
            )
        operands = (q[0], k[0], gate[0], beta[0])
        _reverse_summary(
            *operands,
            d_output[0],
            cu_seqlens,
            scale,
            maps[:, :, :dim_v],
            None,
            launch,
            fastmath=fastmath,
        )
        if launch.split:
            maps = _compose(maps, transition, launch, sequences, reverse=True)
        elif transpose_forward_transition:
            maps[:, :, dim_v:].copy_(transition.transpose(-1, -2))
        else:
            identity = (
                torch.eye(dim_k, dtype=torch.float32, device=q.device)
                .expand(rows, heads, -1, -1)
                .contiguous()
            )
            _reverse_summary(
                *operands,
                torch.zeros_like(d_output[0]),
                cu_seqlens,
                scale,
                maps[:, :, dim_v:],
                identity,
                launch,
                fastmath=fastmath,
            )
        return _summary_rows(maps, cu_seqlens, bounds)


@functools.cache
def _side_stream(device: torch.device) -> torch.cuda.Stream:
    return torch.cuda.Stream(device)


def _gdn_tinv(
    k: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor, cu_seqlens: torch.Tensor
) -> torch.Tensor:
    """``gdn_tinv``'s chunk-inverse tiles for every sequence (or piece) of ``cu_seqlens``: both
    GDN summaries read them."""
    tokens, heads = gate.shape
    rows = gdn_tinv.tinv_rows(tokens, cu_seqlens.shape[0] - 1)
    tinv = torch.empty(rows, heads, gdn_tinv.B_T, gdn_tinv.B_T, dtype=k.dtype, device=k.device)
    row_table = int32(rows * 4, k.device).view(rows, 4)
    gdn_tinv.gdn_tinv(k, gate, beta, cu_seqlens, tinv, row_table, int32(1, k.device))
    return tinv


def _gdn_launch(
    k: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    bounds: torch.Tensor | None,
    cost: SummaryCost,
) -> tuple[_Launch, torch.Tensor]:
    """The selected work and the chunk inverses of its sequences or pieces."""
    launch = _Launch.prepare(cu_seqlens, k.shape[0], gate.shape[1], bounds, gdn_summary.B_T, cost)
    tinv = _gdn_tinv(k, gate, beta, launch.cu_pieces if launch.split else cu_seqlens)
    return launch, tinv


def build_gdn_state_summaries(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    bounds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return FP32 ``[R, H, V + K, K]`` GDN maps packed as ``[B; A]`` for whole packed sequences.

    Inputs use the staged layout: ``k`` ``[1, T, HK, K]``, ``v`` ``[1, T, H, V]``, FP32
    natural-log ``gate`` and FP32 ``beta`` ``[1, T, H]``. Bounds follow
    :func:`build_state_summaries`.
    """
    _validate_gdn({"k": k, "v": v, "gate": gate, "beta": beta}, cu_seqlens, bounds)
    with initialized_cuda_device(k):
        sequences, heads = cu_seqlens.shape[0] - 1, v.shape[2]
        dim_v, dim_k = v.shape[-1], k.shape[-1]
        if not _launches(k.shape[1], bounds):
            maps = _identity_maps(sequences, heads, dim_v, dim_k, k.device)
            return _summary_rows(maps, cu_seqlens, bounds)
        k, v, gate, beta = k[0], v[0], gate[0], beta[0]
        launch, tinv = _gdn_launch(k, gate, beta, cu_seqlens, bounds, GDN_FORWARD_COST)
        items = (launch.work_items, launch.work_count)
        if launch.split:
            bias = torch.empty(
                launch.map_rows, heads, dim_v, dim_k, dtype=torch.float32, device=k.device
            )
            transition = torch.empty(
                launch.map_rows, heads, dim_k, dim_k, dtype=torch.float32, device=k.device
            )
            gdn_summary.gdn_summary(k, v, gate, tinv, bias, transition, *items)
            maps = _compose(bias, transition, launch, sequences, reverse=False)
        else:
            # Every sequence has a work item, so the kernel writes every map, empty ones as
            # [0; I].
            maps = torch.empty(
                sequences, heads, dim_v + dim_k, dim_k, dtype=torch.float32, device=k.device
            )
            gdn_summary.gdn_summary(
                k, v, gate, tinv, maps[:, :, :dim_v], maps[:, :, dim_v:], *items
            )
        return _summary_rows(maps, cu_seqlens, bounds)


def build_gdn_state_grad_summaries(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    *,
    bounds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return FP32 ``[R, H, V + K, K]`` GDN maps packed as ``[C; R]`` for whole packed sequences.

    C is the native entry cotangent under a zero exit cotangent and R is ``A.T`` from the forward
    summary. Inputs follow :func:`build_gdn_state_summaries`, with ``q`` shaped like ``k`` and
    ``d_output`` like ``v``; bounds follow :func:`build_state_summaries`.
    """
    _validate_gdn(
        {"q": q, "k": k, "v": v, "gate": gate, "beta": beta, "d_output": d_output},
        cu_seqlens,
        bounds,
    )
    with initialized_cuda_device(q):
        sequences, heads = cu_seqlens.shape[0] - 1, v.shape[2]
        dim_v, dim_k = v.shape[-1], k.shape[-1]
        if not _launches(q.shape[1], bounds):
            maps = _identity_maps(sequences, heads, dim_v, dim_k, q.device)
            return _summary_rows(maps, cu_seqlens, bounds)
        q, k, v, gate, beta, d_output = (t[0] for t in (q, k, v, gate, beta, d_output))
        launch, tinv = _gdn_launch(k, gate, beta, cu_seqlens, bounds, GDN_REVERSE_COST)
        rows = launch.map_rows
        # Split, the pieces' C; uncut, the maps themselves.
        map_rows = dim_v if launch.split else dim_v + dim_k
        maps = torch.empty(rows, heads, map_rows, dim_k, dtype=torch.float32, device=q.device)
        items = (launch.work_items, launch.work_count)

        def transition() -> torch.Tensor:
            exit_state = torch.empty(  # unused
                rows, heads, dim_v, dim_k, dtype=torch.float32, device=q.device
            )
            transition = torch.empty(
                rows, heads, dim_k, dim_k, dtype=torch.float32, device=q.device
            )
            gdn_summary.gdn_summary(k, v, gate, tinv, exit_state, transition, *items)
            return transition

        def bias() -> None:
            gdn_bprop_summary.gdn_bprop_summary(
                q, k, gate, d_output, tinv, maps[:, :, :dim_v], *items, scale=scale
            )

        # Uncut, both passes run one CTA per selected (sequence, head), a fraction of the SMs:
        # overlap.
        _, transition = fork_join_streams(
            torch.cuda.current_stream(q.device), _side_stream(q.device), transition, bias
        )
        if launch.split:
            maps = _compose(maps, transition, launch, sequences, reverse=True)
        else:
            maps[:, :, dim_v:].copy_(transition.transpose(-1, -2))
        return _summary_rows(maps, cu_seqlens, bounds)


__all__ = [
    "build_gdn_state_grad_summaries",
    "build_gdn_state_summaries",
    "build_state_grad_summaries",
    "build_state_summaries",
]
