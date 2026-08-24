# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton backward for the KDA intra-chunk attention matrix."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.triton.utils import can_use_tma, ptr_offset, requires_int64_offsets
from attn_gym.linear.kda.chunk_scheduler import (
    GridScheduler,
    RaggedChunkMetadata,
    ScheduleKind,
    ScheduleRequest,
    decode_ragged_task,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
    load_ragged_task_count,
)


def uses_tensor_descriptors(args) -> bool:
    """Return whether the launch uses the TMA specialization."""
    return isinstance(args["v"], TensorDescriptor)


def requires_wide_offsets(args) -> bool:
    """Return whether a pointer launch needs int64 offset arithmetic."""
    if uses_tensor_descriptors(args):
        return False
    return requires_int64_offsets(
        args["v"],
        args["do"],
        args["dA"],
        args["cu_seqlens"],
        args["chunk_offsets"],
    )


@triton.jit
def daqk_tile_tma(
    v_desc,
    do_desc,
    dA_desc,
    scale,
    batch,
    row,
    head,
    V: tl.constexpr,
    BT: tl.constexpr,
):
    """Differentiate one complete chunk through TMA descriptors."""
    dA = tl.zeros([BT, BT], dtype=tl.float32)
    for value_tile in range(tl.cdiv(V, 64)):
        v = v_desc.load([batch, row, head, value_tile * 64])
        v = tl.trans(tl.reshape(v, [BT, 64]))
        d_output = do_desc.load([batch, row, head, value_tile * 64])
        d_output = tl.reshape(d_output, [BT, 64])
        dA += tl.dot(d_output, v)

    token = tl.arange(0, BT)
    dA = tl.where(token[:, None] >= token[None, :], dA * scale, 0.0)
    dA_desc.store([batch, row, head, 0], tl.reshape(dA, [1, BT, 1, BT]))


@triton.jit
def daqk_tile_pointer(
    v,
    do,
    dA,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    chunk,
    batch_head,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    num_sequences,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Compute one pointer-addressed dense or packed dAqk tile."""
    if USE_INT64_OFFSETS:
        chunk = chunk.to(tl.int64)
        batch_head = batch_head.to(tl.int64)
    batch, head = batch_head // H, batch_head % H

    if IS_VARLEN:
        if chunk >= load_ragged_chunk_count(chunk_offsets, num_sequences):
            return
        sequence, chunk, token_start, _ = load_ragged_chunk_work(
            cu_seqlens,
            chunk_offsets,
            chunk,
            num_sequences,
            BT,
        )
        if USE_INT64_OFFSETS:
            sequence = sequence.to(tl.int64)
            chunk = chunk.to(tl.int64)
            token_start = token_start.to(tl.int64)
        sequence_end = tl.load(cu_seqlens + ptr_offset((sequence + 1,), (1,))).to(tl.int32)
        if USE_INT64_OFFSETS:
            sequence_end = sequence_end.to(tl.int64)
        sequence_start = token_start - chunk * BT
        T = sequence_end - sequence_start
    else:
        sequence_start = batch * T

    v += ptr_offset((sequence_start, head), (H * V, V))
    do += ptr_offset((sequence_start, head), (H * V, V))
    dA += ptr_offset((sequence_start, head), (H * BT, BT))

    token = tl.arange(0, BT)
    token_offset = chunk * BT + token
    token_mask = token_offset < T
    dA_tile = tl.zeros([BT, BT], dtype=tl.float32)
    for value_tile in range(tl.cdiv(V, 64)):
        value = value_tile * 64 + tl.arange(0, 64)
        v_tile = tl.load(
            v + ptr_offset((value[:, None], token_offset[None, :]), (1, H * V)),
            mask=(value[:, None] < V) & token_mask[None, :],
            other=0.0,
        )
        d_output = tl.load(
            do + ptr_offset((token_offset[:, None], value[None, :]), (H * V, 1)),
            mask=token_mask[:, None] & (value[None, :] < V),
            other=0.0,
        )
        dA_tile += tl.dot(d_output, v_tile)

    dA_tile = tl.where(token[:, None] >= token[None, :], dA_tile * scale, 0.0)
    tl.store(
        dA + ptr_offset((token_offset[:, None], token[None, :]), (H * BT, 1)),
        dA_tile,
        mask=token_mask[:, None],
    )


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_TMA": uses_tensor_descriptors,
        "USE_INT64_OFFSETS": requires_wide_offsets,
    }
)
@triton.jit(do_not_specialize=["T", "num_sequences"])
def chunk_kda_bwd_kernel_dAqk(
    v,
    do,
    dA,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    num_sequences,
    USE_TMA: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Compute dAqk with pointer or TMA specialization."""
    chunk, batch_head = tl.program_id(0), tl.program_id(1)
    if USE_TMA:
        batch, head = batch_head // H, batch_head % H
        daqk_tile_tma(v, do, dA, scale, batch, chunk * BT, head, V, BT)
    else:
        daqk_tile_pointer(
            v,
            do,
            dA,
            cu_seqlens,
            chunk_offsets,
            scale,
            T,
            chunk,
            batch_head,
            H,
            V,
            BT,
            IS_VARLEN,
            num_sequences,
            USE_INT64_OFFSETS,
        )


def can_use_tensor_descriptors(*tensors: torch.Tensor) -> bool:
    """Return whether all tensors satisfy host TMA requirements."""
    return all(can_use_tma(tensor) for tensor in tensors)


@triton.jit
def compose_daqk_ragged_task(
    v_desc,
    do_desc,
    dA_desc,
    v,
    do,
    dA,
    cu_seqlens,
    chunk_offsets,
    scale,
    global_chunk,
    head,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    num_sequences,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Compute one active packed chunk, using pointers for partial tails."""
    _, _, token_start, valid_tokens = load_ragged_chunk_work(
        cu_seqlens,
        chunk_offsets,
        global_chunk,
        num_sequences,
        BT,
    )
    if valid_tokens == BT:
        daqk_tile_tma(v_desc, do_desc, dA_desc, scale, 0, token_start, head, V, BT)
        return

    if USE_INT64_OFFSETS:
        token_start = token_start.to(tl.int64)
        head = head.to(tl.int64)
    token = tl.arange(0, BT)
    token_offset = token_start + token
    token_mask = token < valid_tokens
    dA_tile = tl.zeros([BT, BT], dtype=tl.float32)
    for value_tile in range(tl.cdiv(V, 64)):
        value = value_tile * 64 + tl.arange(0, 64)
        value_mask = value < V
        v_tile = tl.load(
            v + ptr_offset((token_offset[None, :], head, value[:, None]), (H * V, V, 1)),
            mask=value_mask[:, None] & token_mask[None, :],
            other=0.0,
        )
        d_output = tl.load(
            do + ptr_offset((token_offset[:, None], head, value[None, :]), (H * V, V, 1)),
            mask=token_mask[:, None] & value_mask[None, :],
            other=0.0,
        )
        dA_tile += tl.dot(d_output, v_tile)

    dA_tile = tl.where(token[:, None] >= token[None, :], dA_tile * scale, 0.0)
    tl.store(
        dA + ptr_offset((token_offset[:, None], head, token[None, :]), (H * BT, BT, 1)),
        dA_tile,
        mask=token_mask[:, None],
    )


@triton.jit(do_not_specialize=["num_sequences"])
def chunk_kda_bwd_kernel_dAqk_ragged_tma(
    v_desc,
    do_desc,
    dA_desc,
    v,
    do,
    dA,
    cu_seqlens,
    chunk_offsets,
    scale,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    num_sequences,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Launch one CTA per packed capacity slot."""
    global_chunk, head = tl.program_id(0), tl.program_id(1)
    if global_chunk >= load_ragged_chunk_count(chunk_offsets, num_sequences):
        return
    compose_daqk_ragged_task(
        v_desc,
        do_desc,
        dA_desc,
        v,
        do,
        dA,
        cu_seqlens,
        chunk_offsets,
        scale,
        global_chunk,
        head,
        H,
        V,
        BT,
        num_sequences,
        USE_INT64_OFFSETS,
    )


@triton.jit(do_not_specialize=["num_sequences", "num_workers"])
def chunk_kda_bwd_kernel_dAqk_ragged_tma_persistent(
    v_desc,
    do_desc,
    dA_desc,
    v,
    do,
    dA,
    cu_seqlens,
    chunk_offsets,
    scale,
    H: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    num_sequences,
    num_workers,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Stride a bounded worker grid over active packed chunk/head tasks."""
    worker = tl.program_id(0)
    total_tasks = load_ragged_task_count(chunk_offsets, num_sequences, H)
    for task in tl.range(worker, total_tasks, num_workers, num_stages=1):
        global_chunk, head = decode_ragged_task(task, H)
        compose_daqk_ragged_task(
            v_desc,
            do_desc,
            dA_desc,
            v,
            do,
            dA,
            cu_seqlens,
            chunk_offsets,
            scale,
            global_chunk.to(tl.int32),
            head.to(tl.int32),
            H,
            V,
            BT,
            num_sequences,
            USE_INT64_OFFSETS,
        )


def launch_daqk_ragged(
    v: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    chunk_size: int,
    metadata: RaggedChunkMetadata,
    schedule: ScheduleRequest,
) -> torch.Tensor:
    """Launch the packed dAqk specialization."""
    _, tokens, heads, value_dim = v.shape
    dA = torch.empty(1, tokens, heads, chunk_size, dtype=torch.float32, device=v.device)
    if metadata.capacity == 0:
        return dA

    use_tma = (value_dim, chunk_size) == (128, 64) and can_use_tensor_descriptors(v, do, dA)
    resolved = GridScheduler(metadata).resolve_flat(
        schedule,
        heads,
        v.device,
        eligible=use_tma,
        requirement="the packed dAqk TMA path: V=128, chunk_size=64, and TMA-capable tensors",
    )
    if not use_tma:
        chunk_kda_bwd_kernel_dAqk[(metadata.capacity, heads)](
            v=v,
            do=do,
            dA=dA,
            cu_seqlens=metadata.cu_seqlens,
            chunk_offsets=metadata.chunk_offsets,
            scale=scale,
            T=tokens,
            H=heads,
            V=value_dim,
            BT=chunk_size,
            num_sequences=metadata.cu_seqlens.shape[0] - 1,
            num_warps=2,
            num_stages=3,
        )
        return dA

    v_desc = TensorDescriptor.from_tensor(v, [1, chunk_size, 1, 64])
    do_desc = TensorDescriptor.from_tensor(do, [1, chunk_size, 1, 64])
    dA_desc = TensorDescriptor.from_tensor(dA, [1, chunk_size, 1, chunk_size])
    args = (
        v_desc,
        do_desc,
        dA_desc,
        v,
        do,
        dA,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        scale,
    )
    kwargs = {
        "H": heads,
        "V": value_dim,
        "BT": chunk_size,
        "num_sequences": metadata.cu_seqlens.shape[0] - 1,
        "USE_INT64_OFFSETS": requires_int64_offsets(
            v,
            do,
            dA,
            metadata.cu_seqlens,
            metadata.chunk_offsets,
        ),
        "num_warps": 2,
        "num_stages": 3,
    }
    if resolved.kind is ScheduleKind.PERSISTENT:
        chunk_kda_bwd_kernel_dAqk_ragged_tma_persistent[(resolved.workers,)](
            *args, num_workers=resolved.workers, **kwargs
        )
    else:
        chunk_kda_bwd_kernel_dAqk_ragged_tma[(metadata.capacity, heads)](*args, **kwargs)
    return dA


def chunk_kda_bwd_daqk(
    v: torch.Tensor,
    do: torch.Tensor,
    scale: float,
    *,
    chunk_size: int = 64,
    metadata: RaggedChunkMetadata | None = None,
    schedule: ScheduleRequest = ScheduleRequest.AUTO,
) -> torch.Tensor:
    """Differentiate the intra-chunk attention matrix without materializing dV."""
    batch, tokens, heads, value_dim = v.shape
    if do.shape != v.shape:
        raise ValueError("do must have the same shape as v")
    if metadata is not None:
        metadata.validate_chunk_size(chunk_size)
        if batch != 1:
            raise ValueError("packed dAqk metadata requires batch size 1")
        return launch_daqk_ragged(v, do, scale, chunk_size, metadata, schedule)
    if batch != 1 or value_dim != 128 or chunk_size != 64 or tokens % chunk_size:
        raise ValueError("dense dAqk requires B=1, V=128, chunk_size=64, and complete chunks")

    dAqk = torch.empty(batch, tokens, heads, chunk_size, dtype=torch.float32, device=v.device)
    if tokens == 0:
        return dAqk
    if can_use_tensor_descriptors(v, do, dAqk):
        v_arg = TensorDescriptor.from_tensor(v, [1, chunk_size, 1, 64])
        do_arg = TensorDescriptor.from_tensor(do, [1, chunk_size, 1, 64])
        dA_arg = TensorDescriptor.from_tensor(dAqk, [1, chunk_size, 1, chunk_size])
    else:
        v_arg, do_arg, dA_arg = v, do, dAqk
    chunk_kda_bwd_kernel_dAqk[(tokens // chunk_size, heads)](
        v=v_arg,
        do=do_arg,
        dA=dA_arg,
        cu_seqlens=None,
        chunk_offsets=None,
        scale=scale,
        T=tokens,
        H=heads,
        V=value_dim,
        BT=chunk_size,
        num_sequences=0,
        num_warps=2,
        num_stages=3,
    )
    return dAqk


__all__ = ["chunk_kda_bwd_daqk"]
