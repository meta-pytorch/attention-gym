# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Device-side work planning and composition for the affine-summary kernels.

The summary kernels (``_delta_rule/cute/affine_summary_fwd`` / ``_rev`` and the Triton fallbacks)
scan chunks as a serial chain, one CTA per (column tile, head, work item). This module owns the
``work`` table those kernels read and the fold that turns their outputs back into one summary per
range. Both run on the device with static shapes, so a CUDA Graph captured around them replays
for any layout with the same number of ranges. *Chunk*, *range*, and *column tile* are defined in
``affine_summary_fwd``.

    work item     a run of whole chunks ``[chunk_begin, chunk_end)`` of one range, the unit one
                  CTA scans as a serial chain from ``[0 | I]``. Row ``w`` of the ``int32 [W, 4]``
                  table is ``(start_token, chunk_begin, chunk_end, range_length)``;
                  ``range_ids[w]`` names its range. Unused rows are zero with range id ``R`` and
                  scan nothing.
    budget        how many items to cut the span into; ``W = budget + R`` bounds the row count
                  because every range boundary can split one more share.
    partial       the ``[H, V + K, K]`` map one work item produces, indexed by ``w``.

Planning splits the *flat* chunk axis (all ranges' chunks laid end to end) into ``budget`` equal
shares and cuts a share wherever it crosses a range boundary, so a long range among short ones
receives proportionally more items and every item stays inside one range::

    bounds = [[0, 96], [96, 288]]        # chunks: 2 + 3 = 5 on the flat axis
    budget = 3                           # shares [0, 2) [2, 4) [4, 5)
    work   = [(0,   0, 2,  96),          # range 0, chunks [0, 2) = tokens [0, 96)
              (96,  0, 2, 192),          # range 1, chunks [0, 2) = tokens [96, 224)
              (96,  2, 3, 192),          # range 1, chunk  [2, 3) = tokens [224, 288)
              (0, 0, 0, 0), (0, 0, 0, 0)]  # W = 3 + 2 = 5; unused rows have range id 2

Composition is ``functools.reduce(compose_summaries, partials_of_range)`` from
``attn_gym.linear.context_parallel``, run on the device: ``(A0, B0)`` then ``(A1, B1)`` gives
``(A0 @ A1, B0 @ A1 + B1)``. Forward folds a range's items in chunk order; reverse folds them
backwards, the order a cotangent flows. A range with no items gets the identity.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

CHUNK_SIZE = 64


@triton.jit
def load_work_item(
    work,
    work_index,
    BT: tl.constexpr,
    WHOLE_RANGES: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Load ``(start, stop, chunk_begin, chunk_end)`` from a whole-range or planned row.

    ``WHOLE_RANGES`` selects unsplit versus planned rows. When true (budget 1), ``work``
    is the original ``(start, stop)`` bounds and each item scans
    an entire range, without planning or composition. When false, rows contain
    ``(start, chunk_begin, chunk_end, length)`` and the partial maps are composed afterward.
    Both paths support variable-length ranges, empty ranges, and partial tail chunks.

    Chunk indices are relative to the range's ``start``; ``stop`` remains its full token end
    so only its final chunk is masked. Unused planned rows return an empty chunk interval.
    Callers widen ``work_index`` before indexing when ``USE_INT64_OFFSETS`` is set.
    """
    if WHOLE_RANGES:
        start = tl.load(work + 2 * work_index)
        stop = tl.load(work + 2 * work_index + 1)
        if USE_INT64_OFFSETS:
            start = start.to(tl.int64)
            stop = stop.to(tl.int64)
        chunk_begin = 0
        chunk_end = (stop - start + BT - 1) // BT
    else:
        start = tl.load(work + 4 * work_index)
        chunk_begin = tl.load(work + 4 * work_index + 1)
        chunk_end = tl.load(work + 4 * work_index + 2)
        length = tl.load(work + 4 * work_index + 3)
        if USE_INT64_OFFSETS:
            start = start.to(tl.int64)
            chunk_begin = chunk_begin.to(tl.int64)
            chunk_end = chunk_end.to(tl.int64)
            length = length.to(tl.int64)
        stop = start + length
    return start, stop, chunk_begin, chunk_end


@triton.jit
def plan_work_items_kernel(
    bounds,
    work,
    range_ids,
    R,
    BUDGET,
    W,
    BT: tl.constexpr,
):
    """One program cuts the ranges' chunks into ``BUDGET`` equal shares of the flat chunk axis.

    A share that crosses a range boundary is cut there, so every item stays inside one range.
    Items are emitted range-major, so one range's items are contiguous and in chunk order (the
    module docstring has the row layout and a worked example). The flat chunk count and the
    ``share * total`` products are int64: ``bounds`` is int32, but their sum times the budget need
    not fit.
    """
    total = tl.zeros((), dtype=tl.int64)
    for r_id in range(R):
        total += (tl.load(bounds + 2 * r_id + 1) - tl.load(bounds + 2 * r_id) + BT - 1) // BT
    row = 0
    offset = tl.zeros((), dtype=tl.int64)
    for r_id in range(R):
        start = tl.load(bounds + 2 * r_id)
        length = tl.load(bounds + 2 * r_id + 1) - start
        chunks = (length + BT - 1) // BT
        for share in range(BUDGET):
            lo = tl.maximum(share * total // BUDGET, offset) - offset
            hi = tl.minimum((share + 1) * total // BUDGET, offset + chunks) - offset
            if hi > lo:
                tl.store(work + 4 * row, start)
                tl.store(work + 4 * row + 1, lo.to(tl.int32))
                tl.store(work + 4 * row + 2, hi.to(tl.int32))
                tl.store(work + 4 * row + 3, length)
                tl.store(range_ids + row, r_id)
                row += 1
        offset += chunks
    for w in range(row, W):
        tl.store(work + 4 * w, 0)
        tl.store(work + 4 * w + 1, 0)
        tl.store(work + 4 * w + 2, 0)
        tl.store(work + 4 * w + 3, 0)
        tl.store(range_ids + w, R)


def plan_work_items(bounds: torch.Tensor, budget: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut the ranges' chunks into ``budget`` work items on the device, none crossing a range.

    Returns the ``work: int32 [W, 4]`` table and ``range_ids: int32 [W]``, ``W = budget + R``
    (see the module docstring). Nothing reaches the host, so this composes under CUDA Graph
    capture.
    """
    ranges = bounds.shape[0]
    width = budget + ranges
    work = torch.empty(width, 4, dtype=torch.int32, device=bounds.device)
    range_ids = torch.empty(width, dtype=torch.int32, device=bounds.device)
    plan_work_items_kernel[(1,)](
        bounds, work, range_ids, ranges, budget, width, BT=CHUNK_SIZE, num_warps=1
    )
    return work, range_ids


def work_table(bounds: torch.Tensor, budget: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """The kernels' work table for ``budget`` items, and each row's range id.

    With one item per range the table is ``bounds`` itself and ``range_ids`` is ``None``:
    the kernels' ``whole_ranges`` specialization reads ``(start, stop)`` rows and their
    partials are the summaries, so a short single-range summary stays one launch.
    """
    if budget == 1:
        return bounds, None
    return plan_work_items(bounds, budget)


@triton.jit
def compose_work_items_kernel(
    partials,
    range_ids,
    out,
    W: tl.constexpr,
    W_PADDED: tl.constexpr,
    REVERSE: tl.constexpr,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
):
    """Compose one range's partials for one head and one row tile of the output map.

    A *row tile* is ``BM`` rows of the range's ``[V + K, K]`` map: rows ``< V`` are bias rows
    and pick up each item's bias, rows ``>= V`` are transition rows and do not. Because
    ``(A0, B0) ∘ (A1, B1) = (A0 @ A1, B0 @ A1 + B1)`` right-multiplies by the next item's
    transition, every output row depends only on the same row of the accumulator, so a program
    needs its own ``BM`` rows plus each item's full ``[K, K]`` transition. A range's items are
    contiguous in ``range_ids``, so the loop runs over ``[first, first + count)`` with no
    branch and the transition loads pipeline. Dots run as three-pass TF32 (fp32-accurate), so the
    caller's ``float32_matmul_precision`` never leaks in. Row offsets are int64 so large
    ``[W, H, V + K, K]`` buffers address correctly.
    """
    range_id = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    row0 = tl.program_id(2) * BM
    rows = row0 + tl.arange(0, BM)
    cols = tl.arange(0, K)
    is_bias = rows < V

    slots = tl.arange(0, W_PADDED)
    ids = tl.load(range_ids + slots, mask=slots < W, other=-1)
    mine = ids == range_id
    count = tl.sum(mine.to(tl.int32), axis=0)
    first = tl.min(tl.where(mine, slots, W), axis=0)

    # Identity: zero bias rows, a one on the diagonal of the transition rows.
    acc = tl.where((rows[:, None] - V) == cols[None, :], 1.0, 0.0).to(tl.float32)
    acc = tl.where(is_bias[:, None], 0.0, acc)
    for step in tl.range(0, count, num_stages=2):
        if REVERSE:
            item = first + count - 1 - step
        else:
            item = first + step
        base = partials + (item.to(tl.int64) * H + head) * (V + K) * K
        transition = tl.load(base + (V + cols[:, None]) * K + cols[None, :])  # [K, K]
        bias = tl.load(base + rows[:, None] * K + cols[None, :], mask=is_bias[:, None], other=0.0)
        acc = tl.dot(acc, transition, input_precision="tf32x3") + bias
    tl.store(
        out + (range_id.to(tl.int64) * H + head) * (V + K) * K + rows[:, None] * K + cols[None, :],
        acc,
    )


def compose_work_items(
    partials: torch.Tensor, range_ids: torch.Tensor | None, ranges: int, *, reverse: bool
) -> torch.Tensor:
    """Fold ``[W, H, V + K, K]`` partials into one ``[R, H, V + K, K]`` summary per range.

    The device form of ``reduce(compose_summaries, items)`` per range (see
    ``attn_gym.linear.context_parallel``); ``reverse`` applies the later item first, the order a
    cotangent flows. A range with no items gets the identity. ``range_ids=None`` (one item
    per range, see ``work_table``) returns the partials as they are.
    """
    if range_ids is None:
        return partials
    width, heads, packed, key_dim = partials.shape
    value_dim = packed - key_dim
    out = torch.empty(ranges, heads, packed, key_dim, dtype=torch.float32, device=partials.device)
    # Measured on GB200 for an 18-item chain: 16-row tiles with 8 warps are 2x faster than 32-row
    # tiles with 4; more warps hide the per-item transition load, more programs spread the chain.
    block_rows = 16
    compose_work_items_kernel[(ranges, heads, packed // block_rows)](
        partials,
        range_ids,
        out,
        W=width,
        W_PADDED=triton.next_power_of_2(width),
        REVERSE=reverse,
        H=heads,
        V=value_dim,
        K=key_dim,
        BM=block_rows,
        num_warps=8,
    )
    return out


__all__ = ["compose_work_items", "load_work_item", "plan_work_items", "work_table"]
