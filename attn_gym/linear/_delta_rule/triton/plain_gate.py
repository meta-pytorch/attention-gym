# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Internal natural-log to cumulative-log2 gate scan shared by delta-rule variants.

Dense and packed routing use separate kernels because a shared constexpr branch changes
the FP32 scan lowering enough to break route-independent rounding. One registered op
still owns both routes and both scan directions.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset
from attn_gym.linear._delta_rule.constants import DEFAULT_CHUNK_SIZE, LOG2_E
from attn_gym.linear._delta_rule.triton.chunk_scheduler import (
    chunk_capacity,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)


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


@triton.jit
def _zero_inactive_token_tail(
    output,
    cu_seqlens,
    num_sequences,
    T,
    worker,
    workers,
    head,
    channel,
    Y_STRIDES: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    D: tl.constexpr,
):
    """Zero this head/channel block of the inactive capacity ``[cu_seqlens[-1], T)``.

    ``worker`` of ``workers`` programs takes every ``workers``-th ``BT`` token block of the tail,
    so any positive number of programs covers it.
    """
    tail_begin = tl.load(cu_seqlens + num_sequences).to(tl.int64)
    tail_blocks = tl.cdiv(T - tail_begin, BT)
    zeros = tl.zeros((BT, BD), dtype=tl.float32)
    for block in tl.range(worker, tail_blocks, workers):
        token = tail_begin + block * BT + tl.arange(0, BT)[:, None]
        mask = (token < T) & (channel < D)
        tl.store(output + ptr_offset((0, token, head, channel), Y_STRIDES), zeros, mask=mask)


@triton.jit(do_not_specialize=["num_sequences", "T"])
def _plain_gate_scan_ragged_kernel(
    values,
    output,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    T,
    scale,
    X_STRIDES: tl.constexpr,
    Y_STRIDES: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    REVERSE: tl.constexpr,
):
    """Scan one sequence-local packed chunk.

    Programs past the active chunk count exit, except in a reverse scan, where they zero the
    inactive capacity ``[cu_seqlens[-1], T)``: that gradient feeds parameter reductions and must
    be defined. The launcher guarantees at least one such program.
    """
    chunk = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    dim_block = tl.program_id(2).to(tl.int64)
    channel = dim_block * BD + tl.arange(0, BD)[None, :]
    active_chunks = load_ragged_chunk_count(chunk_offsets, num_sequences)
    if chunk >= active_chunks:
        if REVERSE:
            _zero_inactive_token_tail(
                output,
                cu_seqlens,
                num_sequences,
                T,
                chunk - active_chunks,
                tl.num_programs(0) - active_chunks,
                head,
                channel,
                Y_STRIDES,
                BT,
                BD,
                D,
            )
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
    with torch.cuda.device(values.device):
        _, tokens, heads, head_dim = values.shape
        is_ragged = cu_seqlens is not None
        output = torch.empty_like(values, memory_format=torch.contiguous_format)
        block_dim = 32
        if is_ragged:
            assert chunk_offsets is not None
            num_sequences = cu_seqlens.shape[0] - 1
            chunks = chunk_capacity(tokens, num_sequences, DEFAULT_CHUNK_SIZE)
            # The reverse scan zeroes the inactive capacity from the programs past the active
            # chunk count instead of a whole-output memset; one extra program guarantees such a
            # program exists even when the active chunks fill the capacity.
            if reverse:
                chunks += 1
            _plain_gate_scan_ragged_kernel[(chunks, heads, triton.cdiv(head_dim, block_dim))](
                values,
                output,
                cu_seqlens,
                chunk_offsets,
                num_sequences,
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
