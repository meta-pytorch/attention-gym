# SPDX-License-Identifier: BSD-3-Clause

"""Native v1.30 affine state summaries for context-parallel KDA and GDN.

Forward maps pack ``[B; A]`` for ``exit = entry @ A + B``; reverse maps pack ``[C; R]`` for
``d_entry = d_exit @ R + C``. One ``kda_summary`` launch writes both B (the zero-seeded final
state H) and A (the stored-domain transition ``M_buf``); one ``kda_bprop_summary`` launch writes
C (the entry cotangent under a zero exit cotangent). GDN does the same in its own 64-token chunks
with ``gdn_summary`` and ``gdn_bprop_summary``, after one ``gdn_tinv`` pass for the chunk
inverses both read.

The host chooses one piece length for the call from its shape and the GPU's SM count, the
same for both directions so the reverse transition is the forward one transposed bit for bit.
It never uses gate values or the requested bounds to make that decision; ``batch_invariant``
pins it to zero. A length of zero means one block scans each whole (sequence, head).
Otherwise, blocks summarize pieces in parallel and the state chain combines their maps in order
(last piece first for backward).

On the GPU, bounds select whole sequences. Both paths give an unselected sequence a
zero-length work item, so its map is ``[0; I]``. The final gather returns maps in the
requested order. Changing just the bounds during CUDA Graph replay changes which maps are
returned, not the bits of a selected map.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import torch
import triton
import triton.language as tl

from attn_gym._backends.cute.utils import initialized_cuda_device
from attn_gym.linear._delta_rule.triton.chunk_scheduler import assert_packed_offsets
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
    assert_packed_offsets(start, stop, seq, tokens, True)
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
    DESTINATIONS: tl.constexpr,
):
    """Cut every selected sequence into pieces of ``span`` chunks, the last one shorter.

    Keep two indices: where a piece's map lives, and which work item computes it.
    Every sequence gets at least one piece, even if empty or unselected. ``cu_pieces[p]`` is
    piece ``p``'s token offset (unused offsets repeat the stream end). ``main_rows[s:s+2]`` gives
    sequence ``s``'s start and end in the map table, where row ``p * HEADS + h`` means piece
    ``p``, head ``h``. ``work[:count[0]]`` lists the work items' piece IDs and heads: one per
    selected piece and head, and a zero-length one per head of an unselected sequence. With
    ``DESTINATIONS`` the items follow the chain-mode convention of ``work_tables`` instead: the
    sequence's last piece names it as ``final_dst`` and its first as ``dstate_dst``, the others
    ``-1``. ``ROW_BLOCK`` and ``PIECE_BLOCK`` are powers of two covering ``rows`` and any
    sequence's pieces. The offsets are checked as in :func:`_select_summary_work`.
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
        # Before any store: a later sequence's check would come after this one's writes.
        assert_packed_offsets(start, stop, seq, tokens, True)
        matched = tl.sum(((lo == start) & (hi == stop) & (lo < hi)).to(tl.int32), 0) > 0
        selected = (rows == 0) | matched
        chunks = tl.cdiv(stop - start, CHUNK)
        count_here = tl.where(selected, tl.maximum(tl.cdiv(chunks, span), 1), 1)
        piece_start = tl.minimum(start + pieces.to(tl.int64) * span * CHUNK, stop)
        own = pieces < count_here
        tl.store(cu_pieces + piece + pieces, piece_start.to(tl.int32), own)
        # An unselected sequence's one piece gets a zero-length item, whose map is [0; I].
        piece_stop = tl.minimum(piece_start + span * CHUNK, tl.where(selected, stop, start))
        tl.store(main_rows + seq, piece * HEADS)
        piece_chunks = tl.cdiv(piece_stop - piece_start, CHUNK)
        index = (piece + pieces)[:, None]
        row = tl.where((fields == 0) | (fields >= 8), index, 0)
        if DESTINATIONS:
            row = tl.where((fields == 8) & (pieces[:, None] != count_here - 1), -1, row)
            row = tl.where((fields == 9) & (pieces[:, None] != 0), -1, row)
            row = tl.where((fields >= 8) & (row >= 0), seq, row)
        row = tl.where((fields == 3) | (fields == 5), piece_chunks[:, None], row)
        row = tl.where(fields == 6, piece_start[:, None], row)
        row = tl.where(fields == 7, piece_stop[:, None], row).to(tl.int32)
        mask = own[:, None] & (fields < FIELDS)
        for head in range(HEADS):
            head_row = tl.where(fields == 1, head, row)
            offsets = (item + pieces[:, None] * HEADS + head) * FIELDS + fields
            tl.store(work + offsets, head_row, mask)
        item += count_here * HEADS
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
        *,
        destinations: bool = False,
    ) -> _Launch:
        """``destinations`` (split only) follows :func:`_plan_summary_pieces`: the staged plans
        run their prefill or backward over the same work items."""
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
        per_sequence = cdiv(cdiv(tokens, chunk), span)
        capacity = cls.capacity(tokens, sequences, chunk, span)
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
            destinations,
        )
        return cls(work_items, work_count, int32(2, device), capacity, cu_pieces, main_rows)

    @staticmethod
    def capacity(tokens: int, sequences: int, chunk: int, span: int) -> int:
        """Pieces a split plan may fill: sequence ``s`` has at most ``ceil(chunks_s / span)``,
        and the chunks add up to at most ``ceil(tokens / chunk) + sequences`` (each sequence may
        end in a partial chunk)."""
        return cdiv(cdiv(tokens, chunk) + sequences, span) + sequences

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
    validate_bounds(bounds, k.device)


def validate_bounds(bounds: torch.Tensor | None, device: torch.device) -> None:
    if bounds is not None and (
        bounds.ndim != 2
        or bounds.shape[1] != 2
        or bounds.dtype != torch.int32
        or bounds.device != device
        or not bounds.is_contiguous()
    ):
        raise ValueError("bounds must be contiguous int32 [R, 2] on the inputs' device")


def _identity_maps(sequences: int, heads: int, v: int, k: int, device) -> torch.Tensor:
    """``[0; I]`` per sequence/head: the map every skipped or empty sequence keeps."""
    maps = torch.zeros(sequences, heads, v + k, k, dtype=torch.float32, device=device)
    maps[:, :, v:].diagonal(dim1=-2, dim2=-1).fill_(1)
    return maps


def _launches(tokens: int, bounds: torch.Tensor | None) -> bool:
    return tokens != 0 and (bounds is None or bounds.shape[0] != 0)


# Split costs per variant (NOTE [Summary split cost]). The reverse summaries run the forward's
# plan, so their R is the forward's A transposed bit for bit.
KDA_SUMMARY_COST = SummaryCost(compose=2.0, overhead=3.0, fill=2.0)
GDN_SUMMARY_COST = SummaryCost(compose=1.8, overhead=2.0, fill=3.0)
# The composing chain is latency bound, so its finest tensor-core blocks run fastest. Their
# 3xTF32 products round subnormal values, so fastmath=False composes on the FP32 FMA walk.
CHAIN_ROWS = 16
FP32_CHAIN_ROWS = 8


def _fp32(*shape: int, device) -> torch.Tensor:
    return torch.empty(shape, dtype=torch.float32, device=device)


class _ForwardMaps(NamedTuple):
    """Each sequence's ``[B; A]`` (``maps``) and the forward summary kernels' outputs: per piece
    when split (composed by :meth:`compose`), else views of the maps."""

    bias: torch.Tensor
    transition: torch.Tensor
    maps: torch.Tensor
    chain_rows: int  # state rows per chain CTA, for these maps and the reverse's C

    @classmethod
    def allocate(
        cls, launch: _Launch, sequences: int, heads: int, dim_v: int, dim_k: int, device, *,
        identity: bool, fastmath: bool = True,
    ) -> _ForwardMaps:  # fmt: skip
        """``identity`` prefills uncut maps with ``[0; I]`` for kernels that skip empty items;
        ``fastmath=False`` composes on the FP32 walk."""
        chain_rows = CHAIN_ROWS if fastmath else FP32_CHAIN_ROWS
        maps = (
            _identity_maps(sequences, heads, dim_v, dim_k, device)
            if identity and not launch.split
            else _fp32(sequences, heads, dim_v + dim_k, dim_k, device=device)
        )
        if not launch.split:
            return cls(maps[:, :, :dim_v], maps[:, :, dim_v:], maps, chain_rows)
        bias = _fp32(launch.map_rows, heads, dim_v, dim_k, device=device)
        transition = _fp32(launch.map_rows, heads, dim_k, dim_k, device=device)
        return cls(bias, transition, maps, chain_rows)

    def compose(self, launch: _Launch, *, transition_only: bool = False) -> None:
        """Split, chain each sequence's piece maps through its pieces in order into ``maps``:
        only A, with the bits of the whole map's, when ``transition_only``."""
        if launch.split:
            piece_chain.state_chain(
                self.bias, self.transition, None, launch.main_rows, rows=self.chain_rows,
                final=self.maps, first_row=self.bias.shape[2] if transition_only else 0,
            )  # fmt: skip


def _reverse_maps(
    launch: _Launch,
    forward: _ForwardMaps,
    forward_summary: Callable[[torch.Tensor, torch.Tensor], None],
    reverse_summary: Callable[[torch.Tensor], None],
) -> torch.Tensor:
    """Each sequence's ``[C; R]`` beside its ``forward`` maps.

    ``forward_summary(bias, transition)`` and the forward maps' chain run on a side stream next
    to ``reverse_summary(entry)``, which writes the entry cotangents C (per piece when split).
    Split, C composes through the pieces' transposed transitions last piece first, waiting only
    for the forward summary. R is the composed A transposed, the exact adjoint with the
    forward's bits, not the different rounding of a transposed composition."""
    sequences, heads, rows, dim_k = forward.maps.shape
    dim_v, device = rows - dim_k, forward.maps.device
    maps = _fp32(sequences, heads, rows, dim_k, device=device)
    if launch.split:
        entry = _fp32(launch.map_rows, heads, dim_v, dim_k, device=device)
    else:
        entry = maps[:, :, :dim_v].zero_()
    summarized = torch.cuda.Event()

    def side() -> None:
        forward_summary(forward.bias, forward.transition)
        summarized.record()
        forward.compose(launch, transition_only=True)

    def current() -> None:
        reverse_summary(entry)
        if launch.split:
            torch.cuda.current_stream(device).wait_event(summarized)
            piece_chain.state_chain(
                entry, forward.transition, None, launch.main_rows, transpose=True,
                rows=forward.chain_rows, final=maps[:, :, :dim_v],
            )  # fmt: skip

    # Every buffer is allocated on the current stream before the fork and outlives the join.
    fork_join_streams(torch.cuda.current_stream(device), _side_stream(device), side, current)
    maps[:, :, dim_v:].copy_(forward.maps[:, :, dim_v:].transpose(-1, -2))
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
    batch_invariant: bool = False,
) -> torch.Tensor:
    """Return FP32 ``[R, H, V + K, K]`` maps packed as ``[B; A]`` for whole packed sequences.

    Inputs use the staged ``[1, T, H, D]`` layout with FP32 natural-log gate and FP32 beta.
    Optional contiguous int32 ``bounds[R, 2]`` on ``k.device`` select, reorder, or duplicate
    consecutive ``cu_seqlens`` pairs; without bounds one row per sequence is returned. Empty
    intervals yield ``[0; I]``, nonempty rows that are not a sequence yield NaN, and zero rows
    yield no maps. Bounds values stay on device and may change during CUDA Graph replay.
    ``fastmath=False`` selects the non-flushing gate exponentials. ``batch_invariant`` scans
    every sequence uncut, so a sequence's bits do not depend on the other sequences.
    """
    _validate({"k": k, "v": v, "gate": gate, "beta": beta}, cu_seqlens, bounds)
    with initialized_cuda_device(k):
        sequences, heads = cu_seqlens.shape[0] - 1, k.shape[2]
        dim_v, dim_k = v.shape[-1], k.shape[-1]
        if not _launches(k.shape[1], bounds):
            maps = _identity_maps(sequences, heads, dim_v, dim_k, k.device)
            return _summary_rows(maps, cu_seqlens, bounds)
        cost = None if batch_invariant else KDA_SUMMARY_COST
        launch = _Launch.prepare(cu_seqlens, k.shape[1], heads, bounds, kda_common.B_T, cost)
        forward = _ForwardMaps.allocate(
            launch, sequences, heads, dim_v, dim_k, k.device, identity=True, fastmath=fastmath
        )
        _forward_summary(
            k[0], v[0], gate[0], beta[0], forward.bias, forward.transition, launch,
            fastmath=fastmath,
        )  # fmt: skip
        forward.compose(launch)
        maps = forward.maps
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
    batch_invariant: bool = False,
) -> torch.Tensor:
    """Return FP32 ``[R, H, V + K, K]`` maps packed as ``[C; R]`` for whole packed sequences.

    C is the native entry cotangent under a zero exit cotangent. R is ``A.T`` for the A of
    :func:`build_state_summaries` on the same inputs, bit for bit (the exact adjoint of
    right-multiplication by A), so the reverse runs the forward's plan; with
    ``transpose_forward_transition=False`` R is instead a second reverse launch seeded with the
    identity exit cotangent and zero ``d_output`` (requires ``V == K``; never split). Inputs,
    bounds, ``fastmath`` and ``batch_invariant`` follow :func:`build_state_summaries`.
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
        split = transpose_forward_transition and not batch_invariant
        cost = KDA_SUMMARY_COST if split else None
        launch = _Launch.prepare(cu_seqlens, q.shape[1], heads, bounds, kda_common.B_T, cost)
        operands = (q[0], k[0], gate[0], beta[0])

        def reverse(entry: torch.Tensor, d_out: torch.Tensor, exit_cotangent=None) -> None:
            _reverse_summary(
                *operands,
                d_out,
                cu_seqlens,
                scale,
                entry,
                exit_cotangent,
                launch,
                fastmath=fastmath,
            )

        if transpose_forward_transition:
            forward = _ForwardMaps.allocate(
                launch, sequences, heads, dim_v, dim_k, q.device, identity=True, fastmath=fastmath
            )

            def forward_summary(bias: torch.Tensor, transition: torch.Tensor) -> None:
                _forward_summary(
                    k[0], v[0], gate[0], beta[0], bias, transition, launch, fastmath=fastmath
                )

            maps = _reverse_maps(
                launch, forward, forward_summary, lambda entry: reverse(entry, d_output[0])
            )
        else:
            maps = _identity_maps(sequences, heads, dim_v, dim_k, q.device)
            reverse(maps[:, :, :dim_v], d_output[0])
            identity = (
                torch.eye(dim_k, dtype=torch.float32, device=q.device)
                .expand(sequences, heads, -1, -1)
                .contiguous()
            )
            reverse(maps[:, :, dim_v:], torch.zeros_like(d_output[0]), identity)
        return _summary_rows(maps, cu_seqlens, bounds)


@functools.cache
def _side_stream(device: torch.device) -> torch.cuda.Stream:
    return torch.cuda.Stream(device)


# NOTE [Summary-only chunk inverses]
# A context-parallel rank summarizes only the documents that continue on its neighbours, usually
# its first and last, yet the chunk-inverse pass both GDN summaries read used to run over every
# local document: with many short documents it was ~40% of the summary launch. The bounds now
# reach gdn_tinv's row-table warp, which lists only the sequences (or pieces) overlapping a
# summarized range. Tiles keep their rows (row = start // 64 + sequence + chunk, from the table
# entry, not its position), the count stays on the device, and the grid is still shape-sized, so
# a CUDA Graph replays for other bounds. GPU time per call on GB300, first + last document
# summarized, before -> after: 16 x 2048 tokens at 16 heads 120 -> 88 us forward and 201 -> 168
# us reverse; 64 x 512 at 16 heads 210 -> 170 / 353 -> 317; 8 x 4096 at 32 heads 194 -> 143 /
# 340 -> 289; 4 x 1024 at 8 heads unchanged (42 / 78). The staged local passes read every tile
# and keep the full table.


def _gdn_tinv(
    k: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    bounds: torch.Tensor | None = None,
) -> torch.Tensor:
    """``gdn_tinv``'s chunk-inverse tiles for the sequences (or pieces) of ``cu_seqlens``: both
    GDN summaries read them. With ``bounds``, only those overlapping a summarized range: the
    rest of a rank's documents would cost up to ~45% of a summary launch (NOTE [Summary-only
    chunk inverses])."""
    tokens, heads = gate.shape
    rows = gdn_tinv.tinv_rows(tokens, cu_seqlens.shape[0] - 1)
    tinv = torch.empty(rows, heads, gdn_tinv.B_T, gdn_tinv.B_T, dtype=k.dtype, device=k.device)
    row_table = int32(rows * 4, k.device).view(rows, 4)
    gdn_tinv.gdn_tinv(
        k, gate, beta, cu_seqlens, tinv, row_table, int32(1, k.device), bounds=bounds
    )
    return tinv


def _gdn_launch(
    k: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    bounds: torch.Tensor | None,
    cost: SummaryCost | None,
) -> tuple[_Launch, torch.Tensor]:
    """The selected work and the chunk inverses of its sequences or pieces."""
    launch = _Launch.prepare(cu_seqlens, k.shape[0], gate.shape[1], bounds, gdn_summary.B_T, cost)
    tinv = _gdn_tinv(k, gate, beta, launch.cu_pieces if launch.split else cu_seqlens, bounds)
    return launch, tinv


def build_gdn_state_summaries(
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    bounds: torch.Tensor | None = None,
    batch_invariant: bool = False,
) -> torch.Tensor:
    """Return FP32 ``[R, H, V + K, K]`` GDN maps packed as ``[B; A]`` for whole packed sequences.

    Inputs use the staged layout: ``k`` ``[1, T, HK, K]``, ``v`` ``[1, T, H, V]``, FP32
    natural-log ``gate`` and FP32 ``beta`` ``[1, T, H]``. Bounds and ``batch_invariant``
    follow :func:`build_state_summaries`.
    """
    _validate_gdn({"k": k, "v": v, "gate": gate, "beta": beta}, cu_seqlens, bounds)
    with initialized_cuda_device(k):
        sequences, heads = cu_seqlens.shape[0] - 1, v.shape[2]
        dim_v, dim_k = v.shape[-1], k.shape[-1]
        if not _launches(k.shape[1], bounds):
            maps = _identity_maps(sequences, heads, dim_v, dim_k, k.device)
            return _summary_rows(maps, cu_seqlens, bounds)
        k, v, gate, beta = k[0], v[0], gate[0], beta[0]
        cost = None if batch_invariant else GDN_SUMMARY_COST
        launch, tinv = _gdn_launch(k, gate, beta, cu_seqlens, bounds, cost)
        # Every sequence has a work item, so the kernel writes every map, empty ones as [0; I].
        forward = _ForwardMaps.allocate(
            launch, sequences, heads, dim_v, dim_k, k.device, identity=False
        )
        gdn_summary.gdn_summary(
            k, v, gate, tinv, forward.bias, forward.transition, launch.work_items,
            launch.work_count,
        )  # fmt: skip
        forward.compose(launch)
        maps = forward.maps
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
    batch_invariant: bool = False,
) -> torch.Tensor:
    """Return FP32 ``[R, H, V + K, K]`` GDN maps packed as ``[C; R]`` for whole packed sequences.

    C is the native entry cotangent under a zero exit cotangent and R is ``A.T`` for the A of
    :func:`build_gdn_state_summaries`, bit for bit, so the reverse runs the forward's plan.
    Inputs follow :func:`build_gdn_state_summaries`, with ``q`` shaped like ``k`` and
    ``d_output`` like ``v``; bounds and ``batch_invariant`` follow :func:`build_state_summaries`.
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
        cost = None if batch_invariant else GDN_SUMMARY_COST
        launch, tinv = _gdn_launch(k, gate, beta, cu_seqlens, bounds, cost)
        items = (launch.work_items, launch.work_count)
        forward = _ForwardMaps.allocate(
            launch, sequences, heads, dim_v, dim_k, q.device, identity=False
        )

        def forward_summary(bias: torch.Tensor, transition: torch.Tensor) -> None:
            gdn_summary.gdn_summary(k, v, gate, tinv, bias, transition, *items)

        def reverse_summary(entry: torch.Tensor) -> None:
            gdn_bprop_summary.gdn_bprop_summary(
                q, k, gate, d_output, tinv, entry, *items, scale=scale
            )

        maps = _reverse_maps(launch, forward, forward_summary, reverse_summary)
        return _summary_rows(maps, cu_seqlens, bounds)


__all__ = [
    "build_gdn_state_grad_summaries",
    "build_gdn_state_summaries",
    "build_state_grad_summaries",
    "build_state_summaries",
]
