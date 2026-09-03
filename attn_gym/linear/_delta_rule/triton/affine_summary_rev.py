# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Portable Triton reverse affine summaries for delta-rule context parallelism.

Each program keeps one FP32 ``[128, BN]`` augmented-state tile resident while
scanning complete BT64 chunks backward. Raw compact-layout pointers avoid host
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


@triton.jit(do_not_specialize=["T"])
def affine_summary_rev_kernel(
    qg,
    kg,
    w,
    dout,
    aqk,
    cumulative_gate,
    out,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Scan one head and augmented-state column tile backward through BT64 chunks."""
    head = tl.program_id(0)
    column_tile = tl.program_id(1)
    if USE_INT64_OFFSETS:
        head = head.to(tl.int64)
        column_tile = column_tile.to(tl.int64)

    key = tl.arange(0, K)
    row = tl.arange(0, BT)
    column = column_tile * BN + tl.arange(0, BN)
    state = tl.where(column[None, :] == V + key[:, None], 1.0, 0.0).to(tl.float32)

    for chunk in range(T // BT - 1, -1, -1):
        if USE_INT64_OFFSETS:
            token_start = chunk.to(tl.int64) * BT
        else:
            token_start = chunk * BT
        token = token_start + row

        token_key_offset = ptr_offset(
            (token[:, None], head, key[None, :]),
            (H * K, K, 1),
        )
        kg_tile = tl.load(kg + token_key_offset)
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
            mask=column[None, :] < V,
            other=0.0,
        )
        aqk_tile = tl.load(
            aqk
            + ptr_offset(
                (token[:, None], head, row[None, :]),
                (H * BT, BT, 1),
            )
        )
        corrected = tl.dot(tl.trans(aqk_tile), dout_tile, acc=corrected)

        gate_offset = ptr_offset(
            (token_start + BT - 1, head, key),
            (H * K, K, 1),
        )
        state *= tl.exp2(tl.load(cumulative_gate + gate_offset))[:, None]

        qg_tile = tl.load(qg + token_key_offset)
        state += tl.dot(tl.trans(qg_tile), dout_tile) * scale
        w_tile = tl.load(w + token_key_offset)
        corrected_hi = corrected.to(w_tile.dtype)
        corrected_lo = (corrected - corrected_hi.to(tl.float32)).to(w_tile.dtype)
        state -= tl.dot(tl.trans(w_tile), corrected_hi)
        state -= tl.dot(tl.trans(w_tile), corrected_lo)

    out_offset = ptr_offset(
        (head, column[None, :], key[:, None]),
        ((V + K) * K, K, 1),
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
    out: torch.Tensor,
    capability: tuple[int, int],
) -> None:
    """Launch a validated, BT64-padded summary into a preallocated FP32 output."""
    _, tokens, heads, _ = qg.shape
    block_columns = _select_block_columns(heads, capability)
    affine_summary_rev_kernel[(heads, _SUMMARY_DIM // block_columns)](
        qg,
        kg,
        w,
        dout,
        aqk,
        cumulative_gate,
        out,
        float(scale),
        tokens,
        H=heads,
        K=_KEY_DIM,
        V=_VALUE_DIM,
        BT=_CHUNK_SIZE,
        BN=block_columns,
        USE_INT64_OFFSETS=requires_int64_offsets(
            qg,
            kg,
            w,
            dout,
            aqk,
            cumulative_gate,
            out,
        ),
        num_warps=8 if block_columns == 32 else 4,
        num_stages=2,
    )
