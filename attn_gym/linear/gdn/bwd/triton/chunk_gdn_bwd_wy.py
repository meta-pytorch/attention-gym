# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
#
# Portions are derived from flash-linear-attention and licensed under the MIT license;
# see https://github.com/fla-org/flash-linear-attention/graphs/contributors.
# The remaining portions use the BSD-style license in the repository root.
"""Triton backward kernels for the scalar-GDN WY representation.

``cumulative_gate`` contains per-chunk cumulative gates in log2 space.  The
intermediate gate derivatives below intentionally omit ln(2): after the local
reverse cumulative sum, they are gradients for the original natural-log gate.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset
from attn_gym.linear.kda.chunk_scheduler import (
    RaggedChunkMetadata,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)


@triton.jit
def _load_chunk_work(
    chunk,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    if IS_VARLEN:
        if chunk >= load_ragged_chunk_count(chunk_offsets, num_sequences):
            return -1, 0
        _, _, token_start, valid_tokens = load_ragged_chunk_work(
            cu_seqlens,
            chunk_offsets,
            chunk,
            num_sequences,
            BT,
        )
        return token_start, valid_tokens
    return chunk * BT, BT


@triton.jit
def _load_last_gate(gate, valid_tokens, stride, BT: tl.constexpr):
    rows = tl.arange(0, BT)
    return tl.sum(
        tl.load(gate + ptr_offset((rows,), (stride,)), mask=rows == valid_tokens - 1, other=0.0),
        axis=0,
    )


@triton.jit
def chunk_gdn_bwd_dqkwg_kernel(
    q,
    k,
    q_stride_t,
    k_stride_t,
    v_new,
    d_w,
    cumulative_gate,
    h,
    d_output,
    dh,
    d_v_in,
    d_q,
    d_k,
    d_gate_direct,
    cu_seqlens,
    chunk_offsets,
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
    SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    key_tile = tl.program_id(0)
    chunk = tl.program_id(1)
    head = tl.program_id(2)

    token_start, valid_tokens = _load_chunk_work(
        chunk,
        cu_seqlens,
        chunk_offsets,
        num_sequences,
        BT,
        IS_VARLEN,
    )
    if token_start < 0:
        return

    rows = tl.arange(0, BT)
    keys = key_tile * BK + tl.arange(0, BK)
    token = token_start + rows
    token_mask = rows < valid_tokens
    key_head = head // GROUPS

    q_offset = ptr_offset((token[:, None], key_head, keys[None, :]), (q_stride_t, K, 1))
    k_offset = ptr_offset((token[:, None], key_head, keys[None, :]), (k_stride_t, K, 1))
    v_offset = ptr_offset((token[:, None], head), (H * V, V))
    h_offset = ptr_offset((chunk, head), (H * K * V, K * V))

    b_dq = tl.zeros((BT, BK), tl.float32)
    b_dk = tl.zeros((BT, BK), tl.float32)
    b_dw = tl.zeros((BT, BK), tl.float32)
    b_ds = tl.zeros((BT, BT), tl.float32)
    b_dg_last = tl.zeros((1,), tl.float32)

    for value_start in range(0, V, BV):
        values = value_start + tl.arange(0, BV)
        value_mask = values < V
        value_offsets = v_offset + values[None, :]
        h_values = h_offset + ptr_offset((keys[None, :], values[:, None]), (V, 1))

        b_h = tl.load(
            h + h_values,
            mask=value_mask[:, None] & (keys[None, :] < K),
            other=0.0,
        )
        b_dh = tl.load(
            dh + h_values,
            mask=value_mask[:, None] & (keys[None, :] < K),
            other=0.0,
        )
        b_do = tl.load(
            d_output + value_offsets,
            mask=token_mask[:, None] & value_mask[None, :],
            other=0.0,
        )
        b_v_new = tl.load(
            v_new + value_offsets,
            mask=token_mask[:, None] & value_mask[None, :],
            other=0.0,
        )
        b_du = tl.load(
            d_v_in + value_offsets,
            mask=token_mask[:, None] & value_mask[None, :],
            other=0.0,
        )

        # Accumulate the state term of dGate in FP32 rather than the source dtype.
        b_dg_last += tl.sum(b_h.to(tl.float32) * b_dh.to(tl.float32))
        b_ds += tl.dot(b_do, tl.trans(b_v_new))
        b_dq += tl.dot(b_do, b_h)
        b_dk += tl.dot(b_v_new.to(b_dh.dtype), b_dh)
        b_dw -= tl.dot(b_du, b_h)

    gate_offsets = ptr_offset((token, head), (H, 1))
    b_gate = tl.load(cumulative_gate + gate_offsets, mask=token_mask, other=0.0).to(tl.float32)
    b_gate_last = _load_last_gate(
        cumulative_gate + ptr_offset((token_start, head), (H, 1)),
        valid_tokens,
        H,
        BT,
    ).to(tl.float32)
    b_q = tl.load(q + q_offset, mask=token_mask[:, None], other=0.0)
    b_k = tl.load(k + k_offset, mask=token_mask[:, None], other=0.0)

    b_dq *= tl.exp2(b_gate)[:, None] * SCALE
    b_dk *= tl.where(token_mask, tl.exp2(b_gate_last - b_gate), 0.0)[:, None]
    b_dg_last *= tl.exp2(b_gate_last)
    b_dg_last += tl.sum(b_dk * b_k)

    causal = rows[:, None] >= rows[None, :]
    active = token_mask[:, None] & token_mask[None, :]
    gate_delta = tl.where(active, b_gate[:, None] - b_gate[None, :], 0.0)
    b_ds = tl.where(causal & active, b_ds * tl.exp2(gate_delta), 0.0) * SCALE
    b_dq += tl.dot(b_ds.to(b_k.dtype), b_k)
    b_dk += tl.dot(tl.trans(b_ds).to(b_q.dtype), b_q)

    b_dg = tl.sum(b_dq * b_q, axis=1) - tl.sum(b_dk * b_k, axis=1)
    b_dg += tl.where(rows == valid_tokens - 1, b_dg_last, 0.0)

    d_q_offset = ptr_offset((token[:, None], head, keys[None, :]), (H * K, K, 1))
    d_gate_offset = ptr_offset((key_tile, token, head), (T * H, H, 1))
    tl.store(d_q + d_q_offset, b_dq, mask=token_mask[:, None])
    tl.store(d_k + d_q_offset, b_dk, mask=token_mask[:, None])
    tl.store(d_w + d_q_offset, b_dw, mask=token_mask[:, None])
    tl.store(d_gate_direct + d_gate_offset, b_dg, mask=token_mask)


@triton.jit
def chunk_gdn_bwd_wy_kernel(
    k,
    v,
    k_stride_t,
    v_stride_t,
    d_w,
    cumulative_gate,
    beta,
    inverse,
    d_v_in,
    d_k,
    d_v,
    d_gate,
    d_beta,
    cu_seqlens,
    chunk_offsets,
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
    chunk = tl.program_id(0)
    head = tl.program_id(1)

    token_start, valid_tokens = _load_chunk_work(
        chunk,
        cu_seqlens,
        chunk_offsets,
        num_sequences,
        BT,
        IS_VARLEN,
    )
    if token_start < 0:
        return

    rows = tl.arange(0, BT)
    columns = tl.arange(0, BT)
    token = token_start + rows
    token_mask = rows < valid_tokens
    matrix_mask = token_mask[:, None] & token_mask[None, :]
    key_head = head // GROUPS

    gate_offsets = ptr_offset((token, head), (H, 1))
    b_gate = tl.load(cumulative_gate + gate_offsets, mask=token_mask, other=0.0).to(tl.float32)
    b_beta = tl.load(beta + gate_offsets, mask=token_mask, other=0.0).to(tl.float32)
    b_inverse = tl.load(
        inverse + ptr_offset((token[None, :], head, rows[:, None]), (H * BT, BT, 1)),
        mask=matrix_mask,
        other=0.0,
    )

    b_dinverse = tl.zeros((BT, BT), tl.float32)
    b_dbeta = tl.zeros((BT,), tl.float32)
    b_dgate = tl.zeros((BT,), tl.float32)
    b_gate_exp = tl.exp2(b_gate)

    for key_start in range(0, K, BK):
        keys = key_start + tl.arange(0, BK)
        key_mask = keys < K
        k_offsets = ptr_offset((token[:, None], key_head, keys[None, :]), (k_stride_t, K, 1))
        w_offsets = ptr_offset((token[:, None], head, keys[None, :]), (H * K, K, 1))
        b_k = tl.load(k + k_offsets, mask=token_mask[:, None] & key_mask[None, :], other=0.0)
        b_dw = tl.load(
            d_w + w_offsets,
            mask=token_mask[:, None] & key_mask[None, :],
            other=0.0,
        )
        b_kbg = b_k * (b_beta * b_gate_exp)[:, None]
        b_dinverse += tl.dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype))

        b_dkbg = tl.dot(b_inverse, b_dw)
        b_dk = b_dkbg * (b_beta * b_gate_exp)[:, None]
        b_dbeta += tl.sum(b_dkbg * b_k * b_gate_exp[:, None], axis=1)
        b_dgate += tl.sum(b_dkbg * b_kbg, axis=1)
        tl.store(d_k + w_offsets, b_dk, mask=token_mask[:, None] & key_mask[None, :])

    for value_start in range(0, V, BV):
        values = value_start + tl.arange(0, BV)
        value_mask = values < V
        v_offsets = ptr_offset((token[:, None], head, values[None, :]), (v_stride_t, V, 1))
        gradient_offsets = ptr_offset((token[:, None], head, values[None, :]), (H * V, V, 1))
        b_v = tl.load(v + v_offsets, mask=token_mask[:, None] & value_mask[None, :], other=0.0)
        b_du = tl.load(
            d_v_in + gradient_offsets,
            mask=token_mask[:, None] & value_mask[None, :],
            other=0.0,
        )
        b_vb = b_v * b_beta[:, None]
        b_dinverse += tl.dot(b_du, tl.trans(b_vb).to(b_du.dtype))

        b_dvb = tl.dot(b_inverse, b_du)
        b_dbeta += tl.sum(b_dvb * b_v, axis=1)
        tl.store(
            d_v + gradient_offsets,
            b_dvb * b_beta[:, None],
            mask=token_mask[:, None] & value_mask[None, :],
        )

    strict_lower = (rows[:, None] > columns[None, :]) & matrix_mask
    b_dinverse = tl.where(strict_lower, b_dinverse, 0.0)
    b_dinverse = tl.dot(b_dinverse.to(b_inverse.dtype), b_inverse)
    b_dinverse = tl.dot(b_inverse, b_dinverse.to(b_inverse.dtype))
    inverse_gate_delta = tl.where(
        strict_lower,
        b_gate[:, None] - b_gate[None, :],
        0.0,
    )
    b_dinverse = tl.where(
        strict_lower,
        -b_dinverse * tl.exp2(inverse_gate_delta),
        0.0,
    ).to(b_inverse.dtype)

    for key_start in range(0, K, BK):
        keys = key_start + tl.arange(0, BK)
        key_mask = keys < K
        k_offsets = ptr_offset((token[:, None], key_head, keys[None, :]), (k_stride_t, K, 1))
        dk_offsets = ptr_offset((token[:, None], head, keys[None, :]), (H * K, K, 1))
        b_k = tl.load(k + k_offsets, mask=token_mask[:, None] & key_mask[None, :], other=0.0)
        b_kb = b_k * b_beta[:, None]
        b_dkb = tl.dot(b_dinverse, b_k)
        b_dbeta += tl.sum(b_dkb * b_k, axis=1)
        b_dk = b_dkb * b_beta[:, None]
        b_dk += tl.trans(tl.dot(tl.trans(b_kb).to(b_dinverse.dtype), b_dinverse))
        b_dk += tl.load(d_k + dk_offsets, mask=token_mask[:, None] & key_mask[None, :], other=0.0)
        tl.store(d_k + dk_offsets, b_dk, mask=token_mask[:, None] & key_mask[None, :])

        b_kkt = tl.dot(b_k, tl.trans(b_k)) * b_beta[:, None]
        b_adin = b_dinverse.to(tl.float32) * b_kkt
        b_dgate += tl.sum(b_adin, axis=1) - tl.sum(b_adin, axis=0)

    scalar_offsets = ptr_offset((token, head), (H, 1))
    tl.store(d_beta + scalar_offsets, b_dbeta, mask=token_mask)
    tl.store(d_gate + scalar_offsets, b_dgate, mask=token_mask)


@triton.jit
def chunk_gdn_bwd_gate_cumsum_kernel(
    d_gate_direct,
    d_gate_wy,
    d_gate,
    cu_seqlens,
    chunk_offsets,
    T,
    num_sequences,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    chunk = tl.program_id(0)
    head = tl.program_id(1)

    token_start, valid_tokens = _load_chunk_work(
        chunk,
        cu_seqlens,
        chunk_offsets,
        num_sequences,
        BT,
        IS_VARLEN,
    )
    if token_start < 0:
        return

    rows = tl.arange(0, BT)
    token = token_start + rows
    token_mask = rows < valid_tokens
    scalar_offsets = ptr_offset((token, head), (H, 1))
    direct_stride = T * H
    b_dgate = tl.load(d_gate_direct + scalar_offsets, mask=token_mask, other=0.0)
    b_dgate += tl.load(
        d_gate_direct + direct_stride + scalar_offsets,
        mask=token_mask,
        other=0.0,
    )
    b_dgate += tl.load(d_gate_wy + scalar_offsets, mask=token_mask, other=0.0)
    b_dgate = tl.where(token_mask, b_dgate, 0.0)
    b_dgate = tl.cumsum(b_dgate, axis=0, reverse=True)
    tl.store(d_gate + scalar_offsets, b_dgate, mask=token_mask)


def chunk_gdn_bwd_wy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    v_new: torch.Tensor,
    w: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    inverse: torch.Tensor,
    h: torch.Tensor,
    d_output: torch.Tensor,
    dh: torch.Tensor,
    dv: torch.Tensor,
    metadata: RaggedChunkMetadata | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiate scalar-GDN's chunk output and WY representation.

    ``w`` supplies the forward WY layout; its values are not reread because
    this routine produces the internal ``dW`` staging tensor. ``dv`` is the
    cotangent of WY's ``U`` representation (including the recurrence/output
    contributions assembled by the caller). The returned value gradient is
    with respect to the original ``v``. Only active packed rows are part of the public contract;
    callers own masking of inactive capacity.
    """
    bt = 64
    key_dim = value_dim = 128
    block_key = block_value = 64

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B, T, H, D]")
    batch, tokens, key_heads, qk_dim = q.shape
    value_heads = v.shape[2]
    if (
        batch != 1
        or k.shape != (batch, tokens, key_heads, key_dim)
        or qk_dim != key_dim
        or v.shape != (batch, tokens, value_heads, value_dim)
        or v_new.shape != v.shape
        or d_output.shape != v.shape
        or dv.shape != v.shape
        or w.shape != (batch, tokens, value_heads, key_dim)
        or cumulative_gate.shape != (batch, tokens, value_heads)
        or beta.shape != (batch, tokens, value_heads)
        or inverse.shape != (batch, tokens, value_heads, bt)
    ):
        raise ValueError("scalar GDN requires B=1, BT=64, and K=V=128")
    if value_heads % key_heads:
        raise ValueError("the number of value heads must be divisible by the number of q/k heads")
    if not all(
        tensor.is_contiguous()
        for tensor in (
            v_new,
            w,
            cumulative_gate,
            beta,
            inverse,
            h,
            d_output,
            dh,
            dv,
        )
    ):
        raise ValueError("chunk_gdn_bwd_wy requires contiguous intermediate tensors")

    if metadata is None:
        if tokens % bt:
            raise ValueError("dense scalar GDN requires a sequence length divisible by 64")
        chunk_slots = tokens // bt
        cu_seqlens = chunk_offsets = None
    else:
        metadata.validate_chunk_size(bt)
        chunk_slots = metadata.capacity
        cu_seqlens = metadata.cu_seqlens
        chunk_offsets = metadata.chunk_offsets

    state_shape = (batch, chunk_slots, value_heads, key_dim, value_dim)
    compact_state_shape = state_shape[1:]
    if h.shape not in (state_shape, compact_state_shape) or dh.shape not in (
        state_shape,
        compact_state_shape,
    ):
        raise ValueError("h and dh must have layout [chunk, H, K, V] (optionally prefixed by B=1)")

    groups = value_heads // key_heads
    output_factory = torch.zeros if metadata is not None else torch.empty
    output_kwargs = {"device": q.device}
    d_q_expanded = output_factory(
        (batch, tokens, value_heads, key_dim), dtype=torch.float32, **output_kwargs
    )
    d_k_direct = output_factory(
        (batch, tokens, value_heads, key_dim), dtype=torch.float32, **output_kwargs
    )
    d_k_wy = output_factory(
        (batch, tokens, value_heads, key_dim), dtype=torch.float32, **output_kwargs
    )
    d_w = output_factory(w.shape, dtype=w.dtype, **output_kwargs)
    d_value = output_factory(v.shape, dtype=v.dtype, **output_kwargs)
    d_gate_direct = output_factory(
        (key_dim // block_key, batch, tokens, value_heads),
        dtype=cumulative_gate.dtype,
        **output_kwargs,
    )
    d_gate_wy = output_factory(cumulative_gate.shape, dtype=cumulative_gate.dtype, **output_kwargs)
    d_gate = output_factory(cumulative_gate.shape, dtype=cumulative_gate.dtype, **output_kwargs)
    d_beta = output_factory(beta.shape, dtype=beta.dtype, **output_kwargs)

    compact_value_strides = (
        tokens * value_heads * value_dim,
        value_heads * value_dim,
        value_dim,
        1,
    )
    assert dv.stride() == d_value.stride() == compact_value_strides, (
        "dV buffers must match the compact [B,T,H,V] kernel ABI"
    )

    if chunk_slots:
        direct_grid = (key_dim // block_key, chunk_slots, value_heads)
        chunk_gdn_bwd_dqkwg_kernel[direct_grid](
            q,
            k,
            q.stride(1),
            k.stride(1),
            v_new,
            d_w,
            cumulative_gate,
            h,
            d_output,
            dh,
            dv,
            d_q_expanded,
            d_k_direct,
            d_gate_direct,
            cu_seqlens,
            chunk_offsets,
            tokens,
            0 if metadata is None else metadata.cu_seqlens.shape[0] - 1,
            H=value_heads,
            HK=key_heads,
            GROUPS=groups,
            K=key_dim,
            V=value_dim,
            BT=bt,
            BK=block_key,
            BV=block_value,
            SCALE=scale,
            IS_VARLEN=metadata is not None,
            num_warps=4,
            num_stages=2,
        )
        chunk_gdn_bwd_wy_kernel[(chunk_slots, value_heads)](
            k,
            v,
            k.stride(1),
            v.stride(1),
            d_w,
            cumulative_gate,
            beta,
            inverse,
            dv,
            d_k_wy,
            d_value,
            d_gate_wy,
            d_beta,
            cu_seqlens,
            chunk_offsets,
            0 if metadata is None else metadata.cu_seqlens.shape[0] - 1,
            H=value_heads,
            HK=key_heads,
            GROUPS=groups,
            K=key_dim,
            V=value_dim,
            BT=bt,
            BK=block_key,
            BV=block_value,
            IS_VARLEN=metadata is not None,
            num_warps=4,
            num_stages=2,
        )
        chunk_gdn_bwd_gate_cumsum_kernel[(chunk_slots, value_heads)](
            d_gate_direct,
            d_gate_wy,
            d_gate,
            cu_seqlens,
            chunk_offsets,
            tokens,
            0 if metadata is None else metadata.cu_seqlens.shape[0] - 1,
            H=value_heads,
            BT=bt,
            IS_VARLEN=metadata is not None,
            num_warps=2,
            num_stages=2,
        )

    d_k_expanded = d_k_direct + d_k_wy
    if groups == 1:
        return d_q_expanded.to(q.dtype), d_k_expanded.to(k.dtype), d_value, d_gate, d_beta
    return (
        d_q_expanded.view(batch, tokens, key_heads, groups, key_dim).sum(dim=3).to(q.dtype),
        d_k_expanded.view(batch, tokens, key_heads, groups, key_dim).sum(dim=3).to(k.dtype),
        d_value,
        d_gate,
        d_beta,
    )


__all__ = ["chunk_gdn_bwd_wy"]
