# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Internal natural-log to cumulative-log2 KDA gate scan.

Dense and packed routing use separate kernels because a shared constexpr branch changes
the FP32 scan lowering enough to break route-independent rounding. One registered op
still owns both routes and both scan directions.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset
from attn_gym.linear.kda.chunk_scheduler import (
    chunk_capacity,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)
from attn_gym.linear.kda.constants import DEFAULT_CHUNK_SIZE, LOG2_E


@triton.jit(do_not_specialize=["T"])
def _plain_gate_scan_dense_kernel(
    values,
    output,
    T,
    scale,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    REVERSE: tl.constexpr,
):
    """Scan one chunk from an ordinary dense batch."""
    chunk = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    dim_block = tl.program_id(2).to(tl.int64)
    token = chunk * BT + tl.arange(0, BT)[:, None]
    channel = dim_block * BD + tl.arange(0, BD)[None, :]
    mask = (token < T) & (channel < D)
    input_offsets = ptr_offset((0, token, head, channel), X_STRIDES)
    output_offsets = ptr_offset((0, token, head, channel), Y_STRIDES)
    gate = tl.load(values + input_offsets, mask=mask, other=0.0).to(tl.float32)
    cumulative = tl.cumsum(gate, axis=0, reverse=REVERSE) * scale.to(tl.float32)
    tl.store(output + output_offsets, cumulative, mask=mask)


@triton.jit(do_not_specialize=["num_sequences"])
def _plain_gate_scan_ragged_kernel(
    values,
    output,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    scale,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    REVERSE: tl.constexpr,
):
    """Scan one sequence-local packed chunk."""
    chunk = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    dim_block = tl.program_id(2).to(tl.int64)
    active_chunks = load_ragged_chunk_count(chunk_offsets, num_sequences)
    if chunk >= active_chunks:
        return

    _, _, token_begin, valid_tokens = load_ragged_chunk_work(
        cu_seqlens,
        chunk_offsets,
        chunk,
        num_sequences,
        BT,
    )
    token_offset = tl.arange(0, BT)[:, None]
    token = token_begin.to(tl.int64) + token_offset
    channel = dim_block * BD + tl.arange(0, BD)[None, :]
    mask = (token_offset < valid_tokens) & (channel < D)
    input_offsets = ptr_offset((0, token, head, channel), X_STRIDES)
    output_offsets = ptr_offset((0, token, head, channel), Y_STRIDES)
    gate = tl.load(values + input_offsets, mask=mask, other=0.0).to(tl.float32)
    cumulative = tl.cumsum(gate, axis=0, reverse=REVERSE) * scale.to(tl.float32)
    tl.store(output + output_offsets, cumulative, mask=mask)


def _plain_gate_scan_cuda(
    values: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    reverse: bool,
) -> torch.Tensor:
    """Launch the internal dense or packed gate scan."""
    _, tokens, heads, head_dim = values.shape
    is_ragged = cu_seqlens is not None
    # Packed reverse scans do not visit inactive capacity; its gradient must be zero
    # before upstream parameter reductions consume it.
    factory = torch.zeros_like if is_ragged and reverse else torch.empty_like
    output = factory(values, memory_format=torch.contiguous_format)
    block_dim = 32
    if is_ragged:
        assert chunk_offsets is not None
        num_sequences = cu_seqlens.shape[0] - 1
        chunks = chunk_capacity(tokens, num_sequences, DEFAULT_CHUNK_SIZE)
        _plain_gate_scan_ragged_kernel[(chunks, heads, triton.cdiv(head_dim, block_dim))](
            values,
            output,
            cu_seqlens,
            chunk_offsets,
            num_sequences,
            LOG2_E,
            X_STRIDES=(0, *values.stride()[1:]),
            Y_STRIDES=(0, *output.stride()[1:]),
            D=head_dim,
            BT=DEFAULT_CHUNK_SIZE,
            BD=block_dim,
            REVERSE=reverse,
            num_warps=2,
            num_stages=3,
        )
    else:
        assert chunk_offsets is None and values.shape[0] == 1
        _plain_gate_scan_dense_kernel[
            (
                triton.cdiv(tokens, DEFAULT_CHUNK_SIZE),
                heads,
                triton.cdiv(head_dim, block_dim),
            )
        ](
            values,
            output,
            tokens,
            LOG2_E,
            X_STRIDES=(0, *values.stride()[1:]),
            Y_STRIDES=(0, *output.stride()[1:]),
            D=head_dim,
            BT=DEFAULT_CHUNK_SIZE,
            BD=block_dim,
            REVERSE=reverse,
            num_warps=2,
            num_stages=3,
        )
    return output


__all__ = ["_plain_gate_scan_cuda"]
