# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
#
# Portions are derived from flash-linear-attention and licensed under the MIT license;
# see https://github.com/fla-org/flash-linear-attention/graphs/contributors.
# The remaining portions use the BSD-style license in the repository root.

"""Scalar-gate delta-state backward specialized for BT64 and K=V=128."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset
from attn_gym.linear.kda.chunk_scheduler import (
    RaggedChunkMetadata,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
    load_ragged_sequence_work,
)


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["num_sequences"])
def chunk_gdn_bwd_dv_local_kernel(
    q,
    k,
    q_stride_t,
    k_stride_t,
    cumulative_gate,
    d_output,
    dv_local,
    cu_seqlens,
    chunk_offsets,
    scale,
    num_sequences,
    H: tl.constexpr,
    HK: tl.constexpr,
    GROUPS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Compute the causal within-chunk value gradient for one value head."""
    chunk = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    if IS_VARLEN:
        if chunk >= load_ragged_chunk_count(chunk_offsets, num_sequences):
            return
        _sequence, _local_chunk, token_start, valid_tokens = load_ragged_chunk_work(
            cu_seqlens,
            chunk_offsets,
            chunk,
            num_sequences,
            BT,
        )
        token_start = token_start.to(tl.int64)
    else:
        token_start = chunk * BT
        valid_tokens = BT

    row = tl.arange(0, BT)
    token = token_start + row
    token_mask = row < valid_tokens
    key_head = head // GROUPS
    gate = tl.load(
        cumulative_gate + ptr_offset((token, head), (H, 1)),
        mask=token_mask,
        other=0.0,
    ).to(tl.float32)

    attention = tl.zeros((BT, BT), dtype=tl.float32)
    for key_start in range(0, K, BK):
        key = key_start + tl.arange(0, BK)
        q_offset = ptr_offset((token[:, None], key_head, key[None, :]), (q_stride_t, K, 1))
        k_offset = ptr_offset((token[:, None], key_head, key[None, :]), (k_stride_t, K, 1))
        q_tile = tl.load(q + q_offset, mask=token_mask[:, None], other=0.0)
        k_tile = tl.load(k + k_offset, mask=token_mask[:, None], other=0.0)
        attention += tl.dot(k_tile, tl.trans(q_tile)) * scale

    active = token_mask[:, None] & token_mask[None, :]
    causal_transpose = row[:, None] <= row[None, :]
    gate_delta = tl.where(active, gate[None, :] - gate[:, None], 0.0)
    attention = tl.where(
        causal_transpose & active,
        attention * tl.exp2(gate_delta),
        0.0,
    )

    for value_start in range(0, V, BV):
        value = value_start + tl.arange(0, BV)
        output_offset = ptr_offset((token[:, None], head, value[None, :]), (H * V, V, 1))
        output_tile = tl.load(
            d_output + output_offset,
            mask=token_mask[:, None],
            other=0.0,
        )
        value_gradient = tl.dot(attention.to(output_tile.dtype), output_tile)
        tl.store(
            dv_local + output_offset,
            value_gradient.to(dv_local.dtype.element_ty),
            mask=token_mask[:, None],
        )


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "STORE_INITIAL_STATE_GRADIENT": lambda args: args["d_initial_state"] is not None,
        "USE_FINAL_STATE_GRADIENT": lambda args: args["d_final_state"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "num_sequences"])
def chunk_gdn_bwd_delta_h_kernel(
    q,
    k,
    q_stride_t,
    k_stride_t,
    w,
    cumulative_gate,
    d_output,
    d_final_state,
    d_initial_state,
    dv_local,
    dh,
    dv,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    num_sequences,
    H: tl.constexpr,
    HK: tl.constexpr,
    GROUPS: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STORE_INITIAL_STATE_GRADIENT: tl.constexpr,
    USE_FINAL_STATE_GRADIENT: tl.constexpr,
):
    """Traverse one sequence/head/value tile while keeping its state gradient resident."""
    sequence_head = tl.program_id(0).to(tl.int64)
    value_tile = tl.program_id(1).to(tl.int64)
    sequence = sequence_head // H
    head = sequence_head % H
    if IS_VARLEN:
        sequence_begin, sequence_end, chunk_begin = load_ragged_sequence_work(
            cu_seqlens, chunk_offsets, sequence
        )
        sequence_begin = sequence_begin.to(tl.int64)
        sequence_end = sequence_end.to(tl.int64)
        chunk_begin = chunk_begin.to(tl.int64)
        chunk_count = (sequence_end - sequence_begin + BT - 1) // BT
    else:
        sequence_begin = 0
        sequence_end = T
        chunk_begin = 0
        chunk_count = T // BT

    row = tl.arange(0, BT)
    value = value_tile * BV + tl.arange(0, BV)
    key_1 = tl.arange(0, BK)
    key_2 = BK + key_1
    key_head = head // GROUPS
    d_state_1 = tl.zeros((BK, BV), dtype=tl.float32)
    d_state_2 = tl.zeros((BK, BV), dtype=tl.float32)

    state_base = ptr_offset((sequence, head), (H * V * K, V * K))
    if USE_FINAL_STATE_GRADIENT:
        final_offset_1 = state_base + ptr_offset((value[:, None], key_1[None, :]), (K, 1))
        final_offset_2 = state_base + ptr_offset((value[:, None], key_2[None, :]), (K, 1))
        d_state_1 += tl.trans(tl.load(d_final_state + final_offset_1)).to(tl.float32)
        d_state_2 += tl.trans(tl.load(d_final_state + final_offset_2)).to(tl.float32)

    for local_chunk in range(chunk_count - 1, -1, -1):
        global_chunk = chunk_begin + local_chunk
        token_start = sequence_begin + local_chunk * BT
        valid_tokens = tl.minimum(BT, sequence_end - token_start)
        token = token_start + row
        token_mask = row < valid_tokens

        dh_base = ptr_offset((global_chunk, head), (H * K * V, K * V))
        dh_offset_1 = dh_base + ptr_offset((key_1[:, None], value[None, :]), (V, 1))
        dh_offset_2 = dh_base + ptr_offset((key_2[:, None], value[None, :]), (V, 1))
        tl.store(dh + dh_offset_1, d_state_1.to(dh.dtype.element_ty))
        tl.store(dh + dh_offset_2, d_state_2.to(dh.dtype.element_ty))

        gate_offset = ptr_offset((token, head), (H, 1))
        gate = tl.load(
            cumulative_gate + gate_offset,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)
        final_gate = tl.sum(tl.where(row == valid_tokens - 1, gate, 0.0), axis=0)
        restored_decay = tl.where(
            token_mask,
            tl.exp2(tl.where(token_mask, final_gate - gate, 0.0)),
            0.0,
        )

        q_base = ptr_offset((token[:, None], key_head), (q_stride_t, K))
        k_base = ptr_offset((token[:, None], key_head), (k_stride_t, K))
        k_tile_1 = tl.load(
            k + k_base + key_1[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        k_tile_2 = tl.load(
            k + k_base + key_2[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        value_gradient = tl.dot(k_tile_1, d_state_1.to(k_tile_1.dtype))
        value_gradient += tl.dot(k_tile_2, d_state_2.to(k_tile_2.dtype))
        value_gradient *= restored_decay[:, None]

        value_offset = ptr_offset((token[:, None], head, value[None, :]), (H * V, V, 1))
        value_gradient += tl.load(
            dv_local + value_offset,
            mask=token_mask[:, None],
            other=0.0,
        )
        tl.store(
            dv + value_offset,
            value_gradient.to(dv.dtype.element_ty),
            mask=token_mask[:, None],
        )

        state_decay = tl.exp2(final_gate)
        d_state_1 *= state_decay
        d_state_2 *= state_decay
        output_tile = tl.load(
            d_output + value_offset,
            mask=token_mask[:, None],
            other=0.0,
        )
        query_decay = tl.where(token_mask, tl.exp2(gate), 0.0)
        q_tile_1 = tl.load(
            q + q_base + key_1[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        q_tile_2 = tl.load(
            q + q_base + key_2[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        q_tile_1 = (q_tile_1 * query_decay[:, None]).to(q_tile_1.dtype)
        q_tile_2 = (q_tile_2 * query_decay[:, None]).to(q_tile_2.dtype)

        w_base = ptr_offset((token[:, None], head), (H * K, K))
        w_tile_1 = tl.load(
            w + w_base + key_1[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        w_tile_2 = tl.load(
            w + w_base + key_2[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        d_state_1 += tl.dot(tl.trans(q_tile_1), output_tile) * scale
        d_state_1 -= tl.dot(tl.trans(w_tile_1), value_gradient.to(w_tile_1.dtype))
        d_state_2 += tl.dot(tl.trans(q_tile_2), output_tile) * scale
        d_state_2 -= tl.dot(tl.trans(w_tile_2), value_gradient.to(w_tile_2.dtype))

    if STORE_INITIAL_STATE_GRADIENT:
        initial_offset_1 = state_base + ptr_offset((value[:, None], key_1[None, :]), (K, 1))
        initial_offset_2 = state_base + ptr_offset((value[:, None], key_2[None, :]), (K, 1))
        tl.store(d_initial_state + initial_offset_1, tl.trans(d_state_1))
        tl.store(d_initial_state + initial_offset_2, tl.trans(d_state_2))


def chunk_gdn_bwd_delta_h(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cumulative_gate: torch.Tensor,
    d_output: torch.Tensor,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    metadata: RaggedChunkMetadata | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Compute delta-state, initial-state, and corrected-value gradients.

    Dense inputs use one complete B=1 token stream. Packed inputs use fixed-capacity
    ragged metadata; inactive chunk slots and unowned physical tokens remain zero.
    Public state tensors use ``[N, H, V, K]``, while ``dh`` uses ``[chunk, H, K, V]``.
    """
    bt = 64
    key_dim = value_dim = 128
    block_key = block_value = 64

    if q.ndim != 4 or k.shape != q.shape:
        raise ValueError("q and k must have matching [B,T,HK,K] shapes")
    batch, tokens, key_heads, qk_dim = q.shape
    if batch != 1 or qk_dim != key_dim:
        raise ValueError("chunk_gdn_bwd_delta_h requires B=1 and K=128")
    if d_output.ndim != 4:
        raise ValueError("d_output must have shape [B,T,H,V]")
    value_heads = d_output.shape[2]
    if d_output.shape != (batch, tokens, value_heads, value_dim):
        raise ValueError("d_output must match the Q/K token axes and have V=128")
    if key_heads == 0 or value_heads % key_heads:
        raise ValueError("the number of value heads must be divisible by the number of q/k heads")
    if w.shape != (batch, tokens, value_heads, key_dim):
        raise ValueError("w must have shape [B,T,H,K]")
    if cumulative_gate.shape != (batch, tokens, value_heads):
        raise ValueError("cumulative_gate must have shape [B,T,H]")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("q, k, w, and d_output must use float16 or bfloat16")
    if not (k.dtype == w.dtype == d_output.dtype == q.dtype):
        raise TypeError("q, k, w, and d_output must have matching dtypes")

    if metadata is None:
        if tokens % bt:
            raise ValueError("dense chunk_gdn_bwd_delta_h requires complete BT64 chunks")
        num_sequences = 1
        chunk_slots = tokens // bt
        cu_seqlens = chunk_offsets = None
    else:
        metadata.validate_chunk_size(bt)
        if (
            metadata.cu_seqlens.ndim != 1
            or metadata.cu_seqlens.shape[0] < 2
            or metadata.chunk_offsets.shape != metadata.cu_seqlens.shape
        ):
            raise ValueError("packed metadata tensors must have matching [N+1] shapes")
        if metadata.capacity < 0:
            raise ValueError("packed metadata capacity must be nonnegative")
        if (
            metadata.cu_seqlens.dtype != torch.int32
            or metadata.chunk_offsets.dtype != torch.int32
            or not metadata.cu_seqlens.is_contiguous()
            or not metadata.chunk_offsets.is_contiguous()
        ):
            raise ValueError("packed metadata tensors must be contiguous int32")
        num_sequences = metadata.cu_seqlens.shape[0] - 1
        chunk_slots = metadata.capacity
        cu_seqlens = metadata.cu_seqlens
        chunk_offsets = metadata.chunk_offsets

    state_shape = (num_sequences, value_heads, value_dim, key_dim)
    if initial_state is not None and initial_state.shape != state_shape:
        raise ValueError(f"initial_state must have shape {state_shape}")
    if d_final_state is not None and d_final_state.shape != state_shape:
        raise ValueError(f"d_final_state must have shape {state_shape}")

    tensors = (w, cumulative_gate, d_output)
    optional_tensors = (() if d_final_state is None else (d_final_state,)) + (
        () if initial_state is None else (initial_state,)
    )
    metadata_tensors = () if metadata is None else (cu_seqlens, chunk_offsets)
    if any(tensor.device != q.device for tensor in tensors + optional_tensors + metadata_tensors):
        raise ValueError("all tensors must be on the same device")
    if any(not tensor.is_contiguous() for tensor in tensors + optional_tensors):
        raise ValueError("chunk_gdn_bwd_delta_h requires contiguous tensors")

    output_factory = torch.zeros if metadata is not None else torch.empty
    dv_local = output_factory(d_output.shape, dtype=d_output.dtype, device=d_output.device)
    dv = output_factory(d_output.shape, dtype=d_output.dtype, device=d_output.device)
    dh = output_factory(
        (chunk_slots, value_heads, key_dim, value_dim),
        dtype=q.dtype,
        device=q.device,
    )
    d_initial_state = (
        torch.empty(state_shape, dtype=torch.float32, device=q.device)
        if initial_state is not None
        else None
    )

    groups = value_heads // key_heads
    if chunk_slots:
        chunk_gdn_bwd_dv_local_kernel[(chunk_slots, value_heads)](
            q,
            k,
            q.stride(1),
            k.stride(1),
            cumulative_gate,
            d_output,
            dv_local,
            cu_seqlens,
            chunk_offsets,
            scale,
            num_sequences,
            H=value_heads,
            HK=key_heads,
            GROUPS=groups,
            K=key_dim,
            V=value_dim,
            BT=bt,
            BK=32,
            BV=32,
            num_warps=4,
            num_stages=2,
        )

    chunk_gdn_bwd_delta_h_kernel[(num_sequences * value_heads, value_dim // block_value)](
        q,
        k,
        q.stride(1),
        k.stride(1),
        w,
        cumulative_gate,
        d_output,
        d_final_state,
        d_initial_state,
        dv_local,
        dh,
        dv,
        cu_seqlens,
        chunk_offsets,
        scale,
        tokens,
        num_sequences,
        H=value_heads,
        HK=key_heads,
        GROUPS=groups,
        K=key_dim,
        V=value_dim,
        BT=bt,
        BK=block_key,
        BV=block_value,
        num_warps=4,
        num_stages=2,
    )
    return dh, d_initial_state, dv


__all__ = ["chunk_gdn_bwd_delta_h"]
