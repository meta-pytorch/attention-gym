# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Portable Triton forward affine summaries for delta-rule context parallelism.

Each program keeps one FP32 ``[128, BN]`` augmented-state tile resident while
scanning complete BT64 chunks. Raw compact-layout pointers avoid host tensor
descriptors, allocations, and synchronization in the launch path, so a warmed
specialization can be captured directly by a CUDA Graph.
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
    if heads <= 8:
        return 8
    if heads <= 24:
        return 16
    return 32


@triton.jit(do_not_specialize=["T"])
def affine_summary_fwd_kernel(
    kg,
    w,
    u,
    cumulative_gate,
    out,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BN: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Scan one head and augmented-state column tile through all BT64 chunks."""
    head = tl.program_id(0)
    column_tile = tl.program_id(1)
    if USE_INT64_OFFSETS:
        head = head.to(tl.int64)
        column_tile = column_tile.to(tl.int64)

    key = tl.arange(0, K)
    row = tl.arange(0, BT)
    column = column_tile * BN + tl.arange(0, BN)
    state = tl.where(column[None, :] == V + key[:, None], 1.0, 0.0).to(tl.float32)

    for chunk in tl.range(0, T // BT):
        if USE_INT64_OFFSETS:
            token_start = chunk.to(tl.int64) * BT
        else:
            token_start = chunk * BT
        token = token_start + row

        token_key_offset = ptr_offset(
            (token[:, None], head, key[None, :]),
            (H * K, K, 1),
        )
        w_tile = tl.load(w + token_key_offset)
        u_tile = tl.load(
            u
            + ptr_offset(
                (token[:, None], head, column[None, :]),
                (H * V, V, 1),
            ),
            mask=column[None, :] < V,
            other=0.0,
        )
        state_hi = state.to(w_tile.dtype)
        state_lo = (state - state_hi.to(tl.float32)).to(w_tile.dtype)
        tmp = u_tile.to(tl.float32) - tl.dot(w_tile, state_hi)
        tmp -= tl.dot(w_tile, state_lo)

        gate_offset = ptr_offset(
            (token_start + BT - 1, head, key),
            (H * K, K, 1),
        )
        state *= tl.exp2(tl.load(cumulative_gate + gate_offset))[:, None]

        kg_tile = tl.load(kg + token_key_offset)
        tmp_hi = tmp.to(kg_tile.dtype)
        tmp_lo = (tmp - tmp_hi.to(tl.float32)).to(kg_tile.dtype)
        state = tl.dot(tl.trans(kg_tile), tmp_hi, acc=state)
        state = tl.dot(tl.trans(kg_tile), tmp_lo, acc=state)

    out_offset = ptr_offset(
        (head, column[None, :], key[:, None]),
        ((V + K) * K, K, 1),
    )
    tl.store(out + out_offset, state)


def launch_affine_summary_fwd(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    cumulative_gate: torch.Tensor,
    out: torch.Tensor,
    capability: tuple[int, int],
) -> None:
    """Launch a validated, BT64-padded summary into a preallocated FP32 output."""
    _, tokens, heads, _ = kg.shape
    block_columns = _select_block_columns(heads, capability)
    affine_summary_fwd_kernel[(heads, _SUMMARY_DIM // block_columns)](
        kg,
        w,
        u,
        cumulative_gate,
        out,
        tokens,
        H=heads,
        K=_KEY_DIM,
        V=_VALUE_DIM,
        BT=_CHUNK_SIZE,
        BN=block_columns,
        USE_INT64_OFFSETS=requires_int64_offsets(kg, w, u, cumulative_gate, out),
        num_warps=8 if block_columns == 32 else 4,
        num_stages=2,
    )
