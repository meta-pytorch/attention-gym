# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness-first Triton KDA WY/dQKG backward for BT64 and K=V=128.

The implementation follows the KDA inverse's channelwise-gated VJP rather than
scalar GDN's gate algebra. Its three stages compute ``dQ/dW`` from state products,
apply ``A.T`` to form ``dK/dV/dG/dBeta``, then evaluate
``dA = -A.T @ tril(beta * (dV @ V.T + dW @ KG.T), -1) @ A.T``. Matrix products
accumulate in FP32, while the internal dW staging tensor and public dV result use
the FP16/BF16 input dtype.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym._backends.triton.utils import ptr_offset, requires_int64_offsets
from attn_gym.linear.kda.chunk_scheduler import (
    RaggedChunkMetadata,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)
from attn_gym.linear.kda.utils import masked_exp2

_CHUNK_SIZE = 64
_HEAD_DIM = 128
_VALUE_DIM = 128
_BLOCK_KEY = 32
_BLOCK_VALUE = 32
_KEY_TILES = _HEAD_DIM // _BLOCK_KEY
_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


@triton.jit
def _load_chunk_work(
    global_chunk,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    BT: tl.constexpr,
    IS_RAGGED: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Decode one known-active dense or packed chunk."""
    if IS_RAGGED:
        _, _, token_start, valid_tokens = load_ragged_chunk_work(
            cu_seqlens,
            chunk_offsets,
            global_chunk,
            num_sequences,
            BT,
        )
    else:
        if USE_INT64_OFFSETS:
            global_chunk = global_chunk.to(tl.int64)
        token_start = global_chunk * BT
        valid_tokens = global_chunk * 0 + BT
    if USE_INT64_OFFSETS:
        token_start = token_start.to(tl.int64)
        valid_tokens = valid_tokens.to(tl.int64)
    return token_start, valid_tokens


@triton.jit(do_not_specialize=["num_sequences"])
def chunk_kda_bwd_wy_direct_triton_kernel(
    q,
    k,
    v_new,
    gate,
    h,
    d_output,
    dh,
    dv,
    dq,
    dk,
    dg,
    dw,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    scale,
    q_stride_t,
    q_stride_h,
    k_stride_t,
    k_stride_h,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_RAGGED: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    FASTMATH: tl.constexpr,
):
    """Compute dQ, the state part of dK/dG, and low-precision dW."""
    global_chunk = tl.program_id(0)
    head = tl.program_id(1)
    key_tile = tl.program_id(2)
    if IS_RAGGED and global_chunk >= load_ragged_chunk_count(chunk_offsets, num_sequences):
        return
    token_start, valid_tokens = _load_chunk_work(
        global_chunk,
        cu_seqlens,
        chunk_offsets,
        num_sequences,
        BT,
        IS_RAGGED,
        USE_INT64_OFFSETS,
    )
    if USE_INT64_OFFSETS:
        global_chunk = global_chunk.to(tl.int64)
        head = head.to(tl.int64)
        key_tile = key_tile.to(tl.int64)

    row = tl.arange(0, BT)
    key = key_tile * BK + tl.arange(0, BK)
    token = token_start + row
    token_mask = row < valid_tokens
    key_mask = key < K
    matrix_mask = token_mask[:, None] & key_mask[None, :]

    dq_accumulator = tl.zeros((BT, BK), dtype=tl.float32)
    dk_accumulator = tl.zeros((BT, BK), dtype=tl.float32)
    dw_accumulator = tl.zeros((BT, BK), dtype=tl.float32)
    state_gate_gradient = tl.zeros((BK,), dtype=tl.float32)

    for value_start in tl.static_range(0, V, BV):
        value = value_start + tl.arange(0, BV)
        value_mask = value < V
        token_value_offset = ptr_offset(
            (token[:, None], head, value[None, :]),
            (H * V, V, 1),
        )
        state_offset = ptr_offset(
            (global_chunk, head, key[:, None], value[None, :]),
            (H * K * V, K * V, V, 1),
        )
        value_tile_mask = token_mask[:, None] & value_mask[None, :]
        state_tile_mask = key_mask[:, None] & value_mask[None, :]

        output_tile = tl.load(
            d_output + token_value_offset,
            mask=value_tile_mask,
            other=0.0,
        )
        v_new_tile = tl.load(v_new + token_value_offset, mask=value_tile_mask, other=0.0)
        dv_tile = tl.load(dv + token_value_offset, mask=value_tile_mask, other=0.0)
        h_tile = tl.load(h + state_offset, mask=state_tile_mask, other=0.0)
        dh_tile = tl.load(dh + state_offset, mask=state_tile_mask, other=0.0)

        dq_accumulator += tl.dot(output_tile, tl.trans(h_tile))
        dk_accumulator += tl.dot(v_new_tile, tl.trans(dh_tile))
        dw_accumulator -= tl.dot(dv_tile, tl.trans(h_tile))
        state_gate_gradient += tl.sum(h_tile.to(tl.float32) * dh_tile, axis=1)

    q_offset = ptr_offset(
        (token[:, None], head, key[None, :]),
        (q_stride_t, q_stride_h, 1),
    )
    k_offset = ptr_offset(
        (token[:, None], head, key[None, :]),
        (k_stride_t, k_stride_h, 1),
    )
    compact_key_offset = ptr_offset(
        (token[:, None], head, key[None, :]),
        (H * K, K, 1),
    )
    q_tile = tl.load(q + q_offset, mask=matrix_mask, other=0.0)
    k_tile = tl.load(k + k_offset, mask=matrix_mask, other=0.0)
    gate_tile = tl.load(gate + compact_key_offset, mask=matrix_mask, other=0.0).to(tl.float32)

    last_token = token_start + valid_tokens - 1
    last_gate_offset = ptr_offset((last_token, head, key), (H * K, K, 1))
    last_gate = tl.load(gate + last_gate_offset, mask=key_mask, other=0.0).to(tl.float32)
    gate_exp = masked_exp2(gate_tile, matrix_mask, FASTMATH)
    reverse_decay = masked_exp2(last_gate[None, :] - gate_tile, matrix_mask, FASTMATH)

    dq_tile = dq_accumulator * gate_exp * scale
    dk_state = dk_accumulator * reverse_decay
    k_dk_state = k_tile.to(tl.float32) * dk_state
    dg_tile = q_tile.to(tl.float32) * dq_tile - k_dk_state
    last_gradient = state_gate_gradient * masked_exp2(last_gate, key_mask, FASTMATH)
    last_gradient += tl.sum(k_dk_state, axis=0)
    dg_tile += tl.where(row[:, None] == valid_tokens - 1, last_gradient[None, :], 0.0)

    tl.store(dq + compact_key_offset, dq_tile, mask=matrix_mask)
    tl.store(dk + compact_key_offset, dk_state, mask=matrix_mask)
    tl.store(dg + compact_key_offset, dg_tile, mask=matrix_mask)
    tl.store(
        dw + compact_key_offset,
        dw_accumulator.to(dw.dtype.element_ty),
        mask=matrix_mask,
    )


@triton.jit(do_not_specialize=["num_sequences"])
def chunk_kda_bwd_wy_apply_triton_kernel(
    k,
    v,
    gate,
    beta,
    inverse,
    dw,
    dv,
    dk,
    d_value,
    dg,
    db,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    k_stride_t,
    k_stride_h,
    v_stride_t,
    v_stride_h,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_RAGGED: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    FASTMATH: tl.constexpr,
):
    """Apply A^T to dW/dV and emit dK/dV/dG/dBeta in one chunk program."""
    global_chunk = tl.program_id(0)
    head = tl.program_id(1)
    if IS_RAGGED and global_chunk >= load_ragged_chunk_count(chunk_offsets, num_sequences):
        return
    token_start, valid_tokens = _load_chunk_work(
        global_chunk,
        cu_seqlens,
        chunk_offsets,
        num_sequences,
        BT,
        IS_RAGGED,
        USE_INT64_OFFSETS,
    )
    if USE_INT64_OFFSETS:
        global_chunk = global_chunk.to(tl.int64)
        head = head.to(tl.int64)

    row = tl.arange(0, BT)
    column = tl.arange(0, BT)
    token = token_start + row
    token_mask = row < valid_tokens
    inverse_mask = token_mask[:, None] & (column[None, :] < valid_tokens)
    inverse_offset = ptr_offset(
        (token[:, None], head, column[None, :]),
        (H * BT, BT, 1),
    )
    inverse_tile = tl.load(inverse + inverse_offset, mask=inverse_mask, other=0.0)
    beta_offset = ptr_offset((token, head), (H, 1))
    beta_tile = tl.load(beta + beta_offset, mask=token_mask, other=0.0).to(tl.float32)
    db_accumulator = tl.zeros((BT,), dtype=tl.float32)

    for key_start in tl.static_range(0, K, BK):
        key = key_start + tl.arange(0, BK)
        key_mask = key < K
        key_matrix_mask = token_mask[:, None] & key_mask[None, :]
        compact_key_offset = ptr_offset(
            (token[:, None], head, key[None, :]),
            (H * K, K, 1),
        )
        dw_tile = tl.load(dw + compact_key_offset, mask=key_matrix_mask, other=0.0)
        dkgb = tl.dot(tl.trans(inverse_tile), dw_tile)
        k_offset = ptr_offset(
            (token[:, None], head, key[None, :]),
            (k_stride_t, k_stride_h, 1),
        )
        k_tile = tl.load(k + k_offset, mask=key_matrix_mask, other=0.0)
        gate_tile = tl.load(
            gate + compact_key_offset,
            mask=key_matrix_mask,
            other=0.0,
        ).to(tl.float32)
        gate_exp = masked_exp2(gate_tile, key_matrix_mask, FASTMATH)
        kg = k_tile.to(tl.float32) * gate_exp
        dk_tile = tl.load(dk + compact_key_offset, mask=key_matrix_mask, other=0.0)
        dg_tile = tl.load(dg + compact_key_offset, mask=key_matrix_mask, other=0.0)
        dk_tile += dkgb * (beta_tile[:, None] * gate_exp)
        dg_tile += kg * dkgb * beta_tile[:, None]
        db_accumulator += tl.sum(dkgb * kg, axis=1)
        tl.store(dk + compact_key_offset, dk_tile, mask=key_matrix_mask)
        tl.store(dg + compact_key_offset, dg_tile, mask=key_matrix_mask)

    for value_start in tl.static_range(0, V, BV):
        value = value_start + tl.arange(0, BV)
        value_mask = value < V
        value_matrix_mask = token_mask[:, None] & value_mask[None, :]
        compact_value_offset = ptr_offset(
            (token[:, None], head, value[None, :]),
            (H * V, V, 1),
        )
        dv_tile = tl.load(dv + compact_value_offset, mask=value_matrix_mask, other=0.0)
        dvb = tl.dot(tl.trans(inverse_tile), dv_tile)
        v_offset = ptr_offset(
            (token[:, None], head, value[None, :]),
            (v_stride_t, v_stride_h, 1),
        )
        v_tile = tl.load(v + v_offset, mask=value_matrix_mask, other=0.0)
        db_accumulator += tl.sum(dvb * v_tile.to(tl.float32), axis=1)
        tl.store(
            d_value + compact_value_offset,
            (dvb * beta_tile[:, None]).to(d_value.dtype.element_ty),
            mask=value_matrix_mask,
        )

    tl.store(db + beta_offset, db_accumulator, mask=token_mask)


@triton.jit(do_not_specialize=["num_sequences"])
def chunk_kda_bwd_wy_dA_triton_kernel(
    k,
    v,
    gate,
    beta,
    inverse,
    dw,
    dv,
    d_inverse,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    k_stride_t,
    k_stride_h,
    v_stride_t,
    v_stride_h,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_RAGGED: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    FASTMATH: tl.constexpr,
):
    """Form the strict-lower dAkk VJP for one chunk and head."""
    global_chunk = tl.program_id(0)
    head = tl.program_id(1)
    if IS_RAGGED and global_chunk >= load_ragged_chunk_count(chunk_offsets, num_sequences):
        return
    token_start, valid_tokens = _load_chunk_work(
        global_chunk,
        cu_seqlens,
        chunk_offsets,
        num_sequences,
        BT,
        IS_RAGGED,
        USE_INT64_OFFSETS,
    )
    if USE_INT64_OFFSETS:
        global_chunk = global_chunk.to(tl.int64)
        head = head.to(tl.int64)

    row = tl.arange(0, BT)
    column = tl.arange(0, BT)
    row_token = token_start + row
    column_token = token_start + column
    row_mask = row < valid_tokens
    column_mask = column < valid_tokens
    matrix_mask = row_mask[:, None] & column_mask[None, :]
    dA_repr = tl.zeros((BT, BT), dtype=tl.float32)

    for value_start in tl.static_range(0, V, BV):
        value = value_start + tl.arange(0, BV)
        value_mask = value < V
        dv_offset = ptr_offset(
            (row_token[:, None], head, value[None, :]),
            (H * V, V, 1),
        )
        v_offset = ptr_offset(
            (column_token[:, None], head, value[None, :]),
            (v_stride_t, v_stride_h, 1),
        )
        dv_tile = tl.load(
            dv + dv_offset,
            mask=row_mask[:, None] & value_mask[None, :],
            other=0.0,
        )
        v_tile = tl.load(
            v + v_offset,
            mask=column_mask[:, None] & value_mask[None, :],
            other=0.0,
        )
        dA_repr += tl.dot(dv_tile, tl.trans(v_tile))

    for key_start in tl.static_range(0, K, BK):
        key = key_start + tl.arange(0, BK)
        key_mask = key < K
        dw_offset = ptr_offset(
            (row_token[:, None], head, key[None, :]),
            (H * K, K, 1),
        )
        k_offset = ptr_offset(
            (column_token[:, None], head, key[None, :]),
            (k_stride_t, k_stride_h, 1),
        )
        gate_offset = ptr_offset(
            (column_token[:, None], head, key[None, :]),
            (H * K, K, 1),
        )
        dw_tile = tl.load(
            dw + dw_offset,
            mask=row_mask[:, None] & key_mask[None, :],
            other=0.0,
        )
        k_tile = tl.load(
            k + k_offset,
            mask=column_mask[:, None] & key_mask[None, :],
            other=0.0,
        )
        gate_mask = column_mask[:, None] & key_mask[None, :]
        gate_tile = tl.load(gate + gate_offset, mask=gate_mask, other=0.0).to(tl.float32)
        kg_tile = (k_tile.to(tl.float32) * masked_exp2(gate_tile, gate_mask, FASTMATH)).to(
            k_tile.dtype
        )
        dA_repr += tl.dot(dw_tile, tl.trans(kg_tile))

    beta_offset = ptr_offset((column_token, head), (H, 1))
    beta_column = tl.load(beta + beta_offset, mask=column_mask, other=0.0).to(tl.float32)
    strict_lower = (row[:, None] > column[None, :]) & matrix_mask
    dA_repr = tl.where(strict_lower, dA_repr * beta_column[None, :], 0.0)

    inverse_offset = ptr_offset(
        (row_token[:, None], head, column[None, :]),
        (H * BT, BT, 1),
    )
    inverse_tile = tl.load(inverse + inverse_offset, mask=matrix_mask, other=0.0)
    inverse_fp32 = inverse_tile.to(tl.float32)
    right_product = tl.dot(dA_repr, tl.trans(inverse_fp32), input_precision="tf32")
    result = -tl.dot(tl.trans(inverse_fp32), right_product, input_precision="tf32")
    result = tl.where(strict_lower, result, 0.0)
    tl.store(d_inverse + inverse_offset, result, mask=row_mask[:, None])


def chunk_kda_bwd_wy_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    d_output: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    metadata: RaggedChunkMetadata | None,
    *,
    scale: float,
    fastmath: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Differentiate Triton KDA's WY representation and chunk-level state path.

    Args:
        q: Queries with shape ``[1, T, H, 128]``.
        k: Keys with the same shape and dtype as ``q``.
        v: Values with the same shape and dtype as ``q``.
        v_new: Recomputed WY values with the same shape and dtype as ``q``.
        g: FP32 cumulative per-channel log2 gate with shape ``[1, T, H, 128]``.
        beta: FP32 update strength with shape ``[1, T, H]``.
        A: Recomputed inverse with shape ``[1, T, H, 64]`` in the QKV dtype.
        h: Chunk-start state tape ``[1, capacity, H, 128, 128]`` in the QKV dtype.
        d_output: Output cotangent with the same shape and dtype as ``v``.
        dh: Chunk state cotangent tape with the same shape and dtype as ``h``.
        dv: Incoming WY-value cotangent with the same shape and dtype as ``v``.
        metadata: Packed BT64 routing, or ``None`` for complete dense chunks.
        scale: Query/output scale; it multiplies only ``dq``.
        fastmath: Use approximate exponentials instead of libdevice exponentials.

    Returns:
        ``(dq, dk, dv, dg, db, dA)``. ``dq``, ``dk``, ``dg``, ``db``, and ``dA``
        are FP32; the returned ``dv`` uses the QKV dtype. Packed capacity outside
        active sequences is undefined and must not be consumed.
    """
    if q.ndim != 4:
        raise ValueError(f"q must be 4D, got shape {tuple(q.shape)}")
    batch, tokens, heads, key_dim = q.shape
    token_shape = (1, tokens, heads, _HEAD_DIM)
    if batch != 1 or heads < 1 or key_dim != _HEAD_DIM:
        raise ValueError("Triton WY requires q shape [1, T, H, 128] with H >= 1")
    for name, tensor in (
        ("k", k),
        ("v", v),
        ("v_new", v_new),
        ("d_output", d_output),
        ("dv", dv),
    ):
        if tensor.shape != token_shape:
            raise ValueError(f"{name} must have shape {token_shape}, got {tuple(tensor.shape)}")
    if g.shape != token_shape:
        raise ValueError(f"g must have shape {token_shape}, got {tuple(g.shape)}")
    if beta.shape != token_shape[:3]:
        raise ValueError(f"beta must have shape {token_shape[:3]}, got {tuple(beta.shape)}")
    if A.shape != (1, tokens, heads, _CHUNK_SIZE):
        raise ValueError(
            f"A must have shape {(1, tokens, heads, _CHUNK_SIZE)}, got {tuple(A.shape)}"
        )

    low_precision_inputs = (k, v, v_new, A, d_output, dv)
    if q.dtype not in _SUPPORTED_DTYPES or any(
        tensor.dtype != q.dtype for tensor in low_precision_inputs
    ):
        raise TypeError("q, k, v, v_new, A, d_output, and dv must share dtype float16 or bfloat16")
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("g and beta must use float32")
    inputs = (q, k, v, v_new, g, beta, A, d_output, dv)
    if not q.is_cuda or any(tensor.device != q.device for tensor in inputs):
        raise ValueError("Triton WY requires all inputs on the same CUDA device")
    if any(tensor.stride(-1) != 1 for tensor in (q, k, v)):
        raise ValueError("q, k, and v must be contiguous in their last dimension")
    if any(not tensor.is_contiguous() for tensor in (v_new, g, beta, A, d_output, dv)):
        raise ValueError("v_new, g, beta, A, d_output, and dv must be contiguous")

    if metadata is None:
        if tokens % _CHUNK_SIZE:
            raise ValueError(f"dense Triton WY requires complete BT64 chunks, got T={tokens}")
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

    state_shape = (1, capacity, heads, _HEAD_DIM, _VALUE_DIM)
    for name, tensor in (("h", h), ("dh", dh)):
        if tensor.shape != state_shape:
            raise ValueError(f"{name} must have shape {state_shape}, got {tuple(tensor.shape)}")
        if tensor.dtype != q.dtype or tensor.device != q.device or not tensor.is_contiguous():
            raise TypeError(f"{name} must be contiguous on q.device with dtype {q.dtype}")

    # Each active output element is written before reuse. Inactive data may feed only
    # inactive outputs, which caller gradient barriers discard at the packed boundary.
    dq = torch.empty_like(g, memory_format=torch.contiguous_format)
    dk = torch.empty_like(g, memory_format=torch.contiguous_format)
    d_value = torch.empty_like(v, memory_format=torch.contiguous_format)
    dg = torch.empty_like(g, memory_format=torch.contiguous_format)
    db = torch.empty_like(beta, memory_format=torch.contiguous_format)
    dA = torch.empty_like(A, dtype=torch.float32, memory_format=torch.contiguous_format)
    if isinstance(q, FakeTensor) or tokens == 0:
        return dq, dk, d_value, dg, db, dA

    dw = torch.empty_like(q, memory_format=torch.contiguous_format)
    use_int64_offsets = requires_int64_offsets(
        q,
        k,
        v,
        v_new,
        g,
        beta,
        A,
        h,
        d_output,
        dh,
        dv,
        dq,
        dk,
        d_value,
        dg,
        db,
        dA,
        dw,
        cu_seqlens,
        chunk_offsets,
    )
    ragged = metadata is not None

    direct_grid = (capacity, heads, _KEY_TILES)
    chunk_kda_bwd_wy_direct_triton_kernel[direct_grid](
        q[0],
        k[0],
        v_new[0],
        g[0],
        h[0],
        d_output[0],
        dh[0],
        dv[0],
        dq[0],
        dk[0],
        dg[0],
        dw[0],
        cu_seqlens,
        chunk_offsets,
        num_sequences,
        float(scale),
        q.stride(1),
        q.stride(2),
        k.stride(1),
        k.stride(2),
        H=heads,
        K=_HEAD_DIM,
        V=_VALUE_DIM,
        BT=_CHUNK_SIZE,
        BK=_BLOCK_KEY,
        BV=_BLOCK_VALUE,
        IS_RAGGED=ragged,
        USE_INT64_OFFSETS=use_int64_offsets,
        FASTMATH=fastmath,
        num_warps=4,
        num_stages=2,
    )
    chunk_kda_bwd_wy_apply_triton_kernel[(capacity, heads)](
        k[0],
        v[0],
        g[0],
        beta[0],
        A[0],
        dw[0],
        dv[0],
        dk[0],
        d_value[0],
        dg[0],
        db[0],
        cu_seqlens,
        chunk_offsets,
        num_sequences,
        k.stride(1),
        k.stride(2),
        v.stride(1),
        v.stride(2),
        H=heads,
        K=_HEAD_DIM,
        V=_VALUE_DIM,
        BT=_CHUNK_SIZE,
        BK=_BLOCK_KEY,
        BV=_BLOCK_VALUE,
        IS_RAGGED=ragged,
        USE_INT64_OFFSETS=use_int64_offsets,
        FASTMATH=fastmath,
        num_warps=4,
        num_stages=2,
    )
    chunk_kda_bwd_wy_dA_triton_kernel[(capacity, heads)](
        k[0],
        v[0],
        g[0],
        beta[0],
        A[0],
        dw[0],
        dv[0],
        dA[0],
        cu_seqlens,
        chunk_offsets,
        num_sequences,
        k.stride(1),
        k.stride(2),
        v.stride(1),
        v.stride(2),
        H=heads,
        K=_HEAD_DIM,
        V=_VALUE_DIM,
        BT=_CHUNK_SIZE,
        BK=_BLOCK_KEY,
        BV=_BLOCK_VALUE,
        IS_RAGGED=ragged,
        USE_INT64_OFFSETS=use_int64_offsets,
        FASTMATH=fastmath,
        num_warps=4,
        num_stages=2,
    )
    return dq, dk, d_value, dg, db, dA


__all__ = ["chunk_kda_bwd_wy_triton"]
