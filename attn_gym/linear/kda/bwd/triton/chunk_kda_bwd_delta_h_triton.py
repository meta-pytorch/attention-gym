# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton KDA reverse delta-H recurrence for the fixed BT64, K=V=128 path.

One ``(sequence, head, BV=64)`` Triton program computes each chunk's
``Aqk^T @ dO`` contribution and keeps the K-by-BV state cotangent in FP32 while
traversing chunks in reverse. Packed tails use masked pointer accesses, and
capacity slots without runtime work remain zero.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym._backends.triton.utils import ptr_offset, requires_int64_offsets
from attn_gym.linear.kda.chunk_scheduler import RaggedChunkMetadata, load_ragged_sequence_work

_CHUNK_SIZE = 64
_HEAD_DIM = 128
_VALUE_DIM = 128
_BLOCK_KEY = 64
_BLOCK_VALUE = 64
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


@triton.jit(do_not_specialize=["T"])
def chunk_kda_bwd_delta_h_triton_kernel(
    qg,
    kg,
    w,
    d_output,
    aqk,
    gk,
    d_final_state,
    d_initial_state,
    dh,
    dv,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_RAGGED: tl.constexpr,
    USE_FINAL_STATE: tl.constexpr,
    STORE_INITIAL_STATE: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Traverse one sequence/head/value tile with an FP32 state cotangent."""
    sequence_head = tl.program_id(0)
    value_tile = tl.program_id(1)
    if USE_INT64_OFFSETS:
        sequence_head = sequence_head.to(tl.int64)
        value_tile = value_tile.to(tl.int64)
        T = T.to(tl.int64)
    sequence = sequence_head // H
    head = sequence_head % H

    if IS_RAGGED:
        sequence_begin, sequence_end, chunk_begin = load_ragged_sequence_work(
            cu_seqlens,
            chunk_offsets,
            sequence,
        )
        if USE_INT64_OFFSETS:
            sequence_begin = sequence_begin.to(tl.int64)
            sequence_end = sequence_end.to(tl.int64)
            chunk_begin = chunk_begin.to(tl.int64)
        chunk_count = (sequence_end - sequence_begin + BT - 1) // BT
    else:
        sequence_begin = T * 0
        sequence_end = T
        chunk_begin = T * 0
        chunk_count = T // BT

    row = tl.arange(0, BT)
    key_1 = tl.arange(0, BK)
    key_2 = BK + key_1
    value = value_tile * BV + tl.arange(0, BV)
    value_mask = value < V
    d_state_1 = tl.zeros((BK, BV), dtype=tl.float32)
    d_state_2 = tl.zeros((BK, BV), dtype=tl.float32)

    state_base = ptr_offset((sequence, head), (H * V * K, V * K))
    if USE_FINAL_STATE:
        final_offset_1 = state_base + ptr_offset(
            (value[None, :], key_1[:, None]),
            (K, 1),
        )
        final_offset_2 = state_base + ptr_offset(
            (value[None, :], key_2[:, None]),
            (K, 1),
        )
        d_state_1 += tl.load(
            d_final_state + final_offset_1,
            mask=value_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        d_state_2 += tl.load(
            d_final_state + final_offset_2,
            mask=value_mask[None, :],
            other=0.0,
        ).to(tl.float32)

    for local_chunk in range(chunk_count - 1, -1, -1):
        global_chunk = chunk_begin + local_chunk
        token_start = sequence_begin + local_chunk * BT
        valid_tokens = tl.minimum(BT, sequence_end - token_start)
        token = token_start + row
        token_mask = row < valid_tokens

        dh_base = ptr_offset((global_chunk, head), (H * K * V, K * V))
        dh_offset_1 = dh_base + ptr_offset(
            (key_1[:, None], value[None, :]),
            (V, 1),
        )
        dh_offset_2 = dh_base + ptr_offset(
            (key_2[:, None], value[None, :]),
            (V, 1),
        )
        tl.store(
            dh + dh_offset_1,
            d_state_1.to(dh.dtype.element_ty),
            mask=value_mask[None, :],
        )
        tl.store(
            dh + dh_offset_2,
            d_state_2.to(dh.dtype.element_ty),
            mask=value_mask[None, :],
        )

        key_base = ptr_offset((token[:, None], head), (H * K, K))
        kg_tile_1 = tl.load(
            kg + key_base + key_1[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        kg_tile_2 = tl.load(
            kg + key_base + key_2[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        value_gradient = tl.dot(kg_tile_1, d_state_1.to(kg_tile_1.dtype))
        value_gradient += tl.dot(kg_tile_2, d_state_2.to(kg_tile_2.dtype))

        value_offset = ptr_offset(
            (token[:, None], head, value[None, :]),
            (H * V, V, 1),
        )
        output_tile = tl.load(
            d_output + value_offset,
            mask=token_mask[:, None] & value_mask[None, :],
            other=0.0,
        )
        causal = row[:, None] >= row[None, :]
        aqk_tile = tl.load(
            aqk
            + ptr_offset(
                (token[:, None], head, row[None, :]),
                (H * BT, BT, 1),
            ),
            mask=token_mask[:, None] & token_mask[None, :] & causal,
            other=0.0,
        )
        value_gradient += tl.dot(tl.trans(aqk_tile), output_tile)
        tl.store(
            dv + value_offset,
            value_gradient.to(dv.dtype.element_ty),
            mask=token_mask[:, None] & value_mask[None, :],
        )

        last_token = token_start + valid_tokens - 1
        gate_base = ptr_offset((last_token, head), (H * K, K))
        gate_1 = tl.load(gk + gate_base + key_1).to(tl.float32)
        gate_2 = tl.load(gk + gate_base + key_2).to(tl.float32)
        d_state_1 *= tl.exp2(gate_1)[:, None]
        d_state_2 *= tl.exp2(gate_2)[:, None]

        qg_tile_1 = tl.load(
            qg + key_base + key_1[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        qg_tile_2 = tl.load(
            qg + key_base + key_2[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        d_state_1 += tl.dot(tl.trans(qg_tile_1), output_tile) * scale
        d_state_2 += tl.dot(tl.trans(qg_tile_2), output_tile) * scale

        w_tile_1 = tl.load(
            w + key_base + key_1[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        w_tile_2 = tl.load(
            w + key_base + key_2[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        narrowed_value_gradient = value_gradient.to(w_tile_1.dtype)
        d_state_1 -= tl.dot(tl.trans(w_tile_1), narrowed_value_gradient)
        d_state_2 -= tl.dot(tl.trans(w_tile_2), narrowed_value_gradient)

    if STORE_INITIAL_STATE:
        initial_offset_1 = state_base + ptr_offset(
            (value[None, :], key_1[:, None]),
            (K, 1),
        )
        initial_offset_2 = state_base + ptr_offset(
            (value[None, :], key_2[:, None]),
            (K, 1),
        )
        tl.store(
            d_initial_state + initial_offset_1,
            d_state_1,
            mask=value_mask[None, :],
        )
        tl.store(
            d_initial_state + initial_offset_2,
            d_state_2,
            mask=value_mask[None, :],
        )


def chunk_kda_bwd_delta_h_triton(
    qg: torch.Tensor,
    kg: torch.Tensor,
    w: torch.Tensor,
    d_output: torch.Tensor,
    aqk: torch.Tensor,
    *,
    gk: torch.Tensor,
    initial_state: torch.Tensor | None,
    d_final_state: torch.Tensor | None,
    scale: float,
    metadata: RaggedChunkMetadata | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Run the Triton KDA reverse delta-H stage.

    Args:
        qg: Gated queries with shape ``[1, T, H, 128]``.
        kg: Gated keys with the same shape and dtype as ``qg``.
        w: Delta-rule weights with the same shape and dtype as ``qg``.
        d_output: Output cotangent with shape ``[1, T, H, 128]``.
        aqk: Intra-chunk factor with shape ``[1, T, H, 64]``.
        gk: FP32 cumulative key gate with shape ``[1, T, H, 128]``.
        initial_state: Optional FP32 state ``[N, H, 128, 128]``. Its presence
            requests the initial-state cotangent; its values are not read.
        d_final_state: Optional FP32 final-state cotangent with the state shape.
        scale: Query/output contribution scale.
        metadata: Packed BT64 routing, or ``None`` for complete dense chunks.

    Returns:
        ``(dh, d_initial_state, dv)`` where ``dh`` has shape
        ``[1, capacity, H, 128, 128]`` in the input dtype,
        ``d_initial_state`` is optional FP32 ``[N, H, 128, 128]``, and ``dv``
        matches ``d_output``.
    """
    if qg.ndim != 4:
        raise ValueError(f"qg must be 4D, got shape {tuple(qg.shape)}")
    batch, tokens, heads, key_dim = qg.shape
    expected_key_shape = (1, tokens, heads, _HEAD_DIM)
    expected_value_shape = (1, tokens, heads, _VALUE_DIM)
    if batch != 1 or heads < 1 or key_dim != _HEAD_DIM:
        raise ValueError("Hopper delta-H requires qg shape [1, T, H, 128] with H >= 1")
    if kg.shape != expected_key_shape or w.shape != expected_key_shape:
        raise ValueError(f"kg and w must have shape {expected_key_shape}")
    if d_output.shape != expected_value_shape:
        raise ValueError(f"d_output must have shape {expected_value_shape}")
    if aqk.shape != (1, tokens, heads, _CHUNK_SIZE):
        raise ValueError(f"aqk must have shape {(1, tokens, heads, _CHUNK_SIZE)}")
    if gk.shape != expected_key_shape:
        raise ValueError(f"gk must have shape {expected_key_shape}")
    if qg.dtype not in _SUPPORTED_DTYPES or any(
        tensor.dtype != qg.dtype for tensor in (kg, w, d_output, aqk)
    ):
        raise TypeError("qg, kg, w, d_output, and aqk must share dtype float16 or bfloat16")
    if gk.dtype != torch.float32:
        raise TypeError("gk must use float32")

    inputs = (qg, kg, w, d_output, aqk, gk)
    if not qg.is_cuda or any(tensor.device != qg.device for tensor in inputs):
        raise ValueError("Hopper delta-H requires all inputs on the same CUDA device")
    if any(not tensor.is_contiguous() for tensor in inputs):
        raise ValueError("Hopper delta-H requires contiguous inputs")

    if metadata is None:
        if tokens % _CHUNK_SIZE:
            raise ValueError(f"dense Hopper delta-H requires complete BT64 chunks, got T={tokens}")
        cu_seqlens = None
        chunk_offsets = None
        num_sequences = 1
        capacity = tokens // _CHUNK_SIZE
    else:
        metadata.validate_chunk_size(_CHUNK_SIZE)
        cu_seqlens = metadata.cu_seqlens
        chunk_offsets = metadata.chunk_offsets
        num_sequences = cu_seqlens.shape[0] - 1
        capacity = metadata.capacity

    state_shape = (num_sequences, heads, _VALUE_DIM, _HEAD_DIM)
    for name, state in (("initial_state", initial_state), ("d_final_state", d_final_state)):
        if state is not None and state.shape != state_shape:
            raise ValueError(f"{name} must have shape {state_shape}, got {tuple(state.shape)}")
        if state is not None and (state.dtype != torch.float32 or state.device != qg.device):
            raise TypeError(f"{name} must be float32 on qg.device")
    if d_final_state is not None and not d_final_state.is_contiguous():
        raise TypeError("d_final_state must be contiguous")

    output_factory = torch.zeros if metadata is not None else torch.empty
    dh = output_factory(
        (1, capacity, heads, _HEAD_DIM, _VALUE_DIM),
        dtype=qg.dtype,
        device=qg.device,
    )
    dv = output_factory(d_output.shape, dtype=d_output.dtype, device=d_output.device)
    d_initial_state = (
        torch.empty(state_shape, dtype=torch.float32, device=qg.device)
        if initial_state is not None
        else None
    )
    if isinstance(qg, FakeTensor):
        return dh, d_initial_state, dv

    use_int64_offsets = requires_int64_offsets(
        qg,
        kg,
        w,
        d_output,
        aqk,
        gk,
        d_final_state,
        dh,
        d_initial_state,
        dv,
        cu_seqlens,
        chunk_offsets,
    )
    chunk_kda_bwd_delta_h_triton_kernel[(num_sequences * heads, _VALUE_DIM // _BLOCK_VALUE)](
        qg,
        kg,
        w,
        d_output,
        aqk,
        gk,
        d_final_state,
        d_initial_state,
        dh,
        dv,
        cu_seqlens,
        chunk_offsets,
        float(scale),
        tokens,
        H=heads,
        K=_HEAD_DIM,
        V=_VALUE_DIM,
        BT=_CHUNK_SIZE,
        BK=_BLOCK_KEY,
        BV=_BLOCK_VALUE,
        IS_RAGGED=metadata is not None,
        USE_FINAL_STATE=d_final_state is not None,
        STORE_INITIAL_STATE=initial_state is not None,
        USE_INT64_OFFSETS=use_int64_offsets,
        num_warps=4,
        num_stages=2,
    )
    return dh, d_initial_state, dv


__all__ = ["chunk_kda_bwd_delta_h_triton"]
