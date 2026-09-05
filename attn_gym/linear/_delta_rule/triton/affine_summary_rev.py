# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Portable Triton reverse affine summaries for delta-rule context parallelism.

Each program scans one work item (row layout in ``work_items.py``), keeping one
FP32 ``[128, BN]`` augmented-state tile resident while scanning BT64 chunks backward. Raw compact-layout pointers avoid host
tensor descriptors, allocations, and synchronization in the launch path, so a
warmed specialization can be captured directly by a CUDA Graph.

The fp32-valued ``tl.dot`` operands (state and corrected write gradient) are
split into hi/lo halves of the I/O dtype and accumulated in two passes, exactly
as the forward summary treats its state and residual, so the reverse transition
is the transpose of the map the forward summary actually evaluates.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset, requires_int64_offsets

_CHUNK_SIZE = 64
_KEY_DIM = 128
_VALUE_DIM = 128
_SUMMARY_DIM = _VALUE_DIM + _KEY_DIM


def _select_block_columns(heads: int, capability: tuple[int, int]) -> int:
    """Select the measured Hopper column tile or a conservative portable default."""
    if capability != (9, 0):
        return 16
    if heads <= 4:
        return 8
    if heads <= 8:
        return 16
    return 32


@triton.jit
def affine_summary_rev_kernel(
    qg,
    kg,
    w,
    dout,
    aqk,
    cumulative_gate,
    work,
    out,
    scale,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
    WHOLE_RANGES: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Scan one work item, head, and augmented-state column tile backward through its chunks.

    ``work_items.py`` defines the work-row layout. The partial last chunk is masked so
    tokens past the range's end contribute nothing; unused rows store the identity.
    """
    head = tl.program_id(0)
    column_tile = tl.program_id(1)
    work_index = tl.program_id(2)
    if USE_INT64_OFFSETS:
        head = head.to(tl.int64)
        column_tile = column_tile.to(tl.int64)
        work_index = work_index.to(tl.int64)
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

    key = tl.arange(0, K)
    row = tl.arange(0, BT)
    column = column_tile * BN + tl.arange(0, BN)
    state = tl.where(column[None, :] == V + key[:, None], 1.0, 0.0).to(tl.float32)

    for chunk in range(chunk_end - 1, chunk_begin - 1, -1):
        token_start = start + chunk * BT
        token = token_start + row
        valid = token[:, None] < stop

        token_key_offset = ptr_offset(
            (token[:, None], head, key[None, :]),
            (H * K, K, 1),
        )
        kg_tile = tl.load(kg + token_key_offset, mask=valid, other=0.0)
        state_hi = state.to(kg_tile.dtype)
        state_lo = (state - state_hi.to(tl.float32)).to(kg_tile.dtype)
        corrected = tl.dot(kg_tile, state_hi)
        corrected = tl.dot(kg_tile, state_lo, acc=corrected)

        dout_tile = tl.load(
            dout
            + ptr_offset(
                (token[:, None], head, column[None, :]),
                (H * V, V, 1),
            ),
            mask=valid & (column[None, :] < V),
            other=0.0,
        )
        aqk_tile = tl.load(
            aqk
            + ptr_offset(
                (token[:, None], head, row[None, :]),
                (H * BT, BT, 1),
            ),
            mask=valid,
            other=0.0,
        )
        corrected = tl.dot(tl.trans(aqk_tile), dout_tile, acc=corrected)

        gate_offset = ptr_offset(
            (tl.minimum(token_start + BT - 1, stop - 1), head, key),
            (H * K, K, 1),
        )
        state *= tl.exp2(tl.load(cumulative_gate + gate_offset))[:, None]

        qg_tile = tl.load(qg + token_key_offset, mask=valid, other=0.0)
        state += tl.dot(tl.trans(qg_tile), dout_tile) * scale
        w_tile = tl.load(w + token_key_offset, mask=valid, other=0.0)
        corrected_hi = corrected.to(w_tile.dtype)
        corrected_lo = (corrected - corrected_hi.to(tl.float32)).to(w_tile.dtype)
        state -= tl.dot(tl.trans(w_tile), corrected_hi)
        state -= tl.dot(tl.trans(w_tile), corrected_lo)

    out_offset = ptr_offset(
        (work_index, head, column[None, :], key[:, None]),
        (H * (V + K) * K, (V + K) * K, K, 1),
    )
    tl.store(out + out_offset, state)


def launch_affine_summary_rev(
    qg: torch.Tensor,
    kg: torch.Tensor,
    w: torch.Tensor,
    dout: torch.Tensor,
    aqk: torch.Tensor,
    cumulative_gate: torch.Tensor,
    scale: float,
    work: torch.Tensor,
    partials: torch.Tensor,
    capability: tuple[int, int],
    *,
    whole_ranges: bool,
) -> None:
    """Scan one item per work row into ``partials[W, H, V + K, K]``."""
    heads = qg.shape[2]
    block_columns = _select_block_columns(heads, capability)
    affine_summary_rev_kernel[(heads, _SUMMARY_DIM // block_columns, work.shape[0])](
        qg,
        kg,
        w,
        dout,
        aqk,
        cumulative_gate,
        work,
        partials,
        float(scale),
        H=heads,
        K=_KEY_DIM,
        V=_VALUE_DIM,
        BT=_CHUNK_SIZE,
        BN=block_columns,
        WHOLE_RANGES=whole_ranges,
        USE_INT64_OFFSETS=requires_int64_offsets(
            qg,
            kg,
            w,
            dout,
            aqk,
            cumulative_gate,
            partials,
        ),
        num_warps=8 if block_columns == 32 else 4,
        num_stages=2,
    )
