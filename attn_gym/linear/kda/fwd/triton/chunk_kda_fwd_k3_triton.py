# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton KDA K3 off-diagonal factors over the existing BT64/BC16 ABI."""

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
_SUBCHUNK_SIZE = 16
_HEAD_DIM = 128
_OFFDIAGONAL_BLOCKS = 6


@triton.jit
def _store_k3_block(
    aqk,
    akk_od,
    aqk_block,
    akk_block,
    row_valid,
    col_valid,
    token_start,
    global_chunk,
    head,
    aqk_stride_t,
    aqk_stride_h,
    scale,
    H: tl.constexpr,
    BC: tl.constexpr,
    PAIR: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    """Store one K3 block into the existing Aqk and temporary Akk layouts."""
    row = tl.arange(0, BC)
    output_rows = token_start + ROW_BLOCK * BC + row
    output_cols = COL_BLOCK * BC + row
    aqk_value = tl.where(col_valid[None, :], aqk_block * scale, 0.0)
    tl.store(
        aqk
        + ptr_offset(
            (output_rows[:, None], head, output_cols[None, :]),
            (aqk_stride_t, aqk_stride_h, 1),
        ),
        aqk_value,
        mask=row_valid[:, None],
    )

    element = row[:, None] * BC + row[None, :]
    output_row = global_chunk * 6 + PAIR
    output_col = head * BC * BC + element
    tl.store(
        akk_od + ptr_offset((output_row, output_col), (H * BC * BC, 1)),
        tl.where(row_valid[:, None] & col_valid[None, :], akk_block, 0.0),
    )


@triton.jit
def _zero_aqk_block(
    aqk,
    row_valid,
    token_start,
    head,
    aqk_stride_t,
    aqk_stride_h,
    BC: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    COL_BLOCK: tl.constexpr,
):
    """Define one block-strictly-upper Aqk tile as zero."""
    row = tl.arange(0, BC)
    output_rows = token_start + ROW_BLOCK * BC + row
    output_cols = COL_BLOCK * BC + row
    tl.store(
        aqk
        + ptr_offset(
            (output_rows[:, None], head, output_cols[None, :]),
            (aqk_stride_t, aqk_stride_h, 1),
        ),
        0.0,
        mask=row_valid[:, None],
    )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["num_sequences"])
def chunk_kda_fwd_k3_triton_kernel(
    q,
    k,
    gate,
    beta,
    aqk,
    akk_od,
    cu_seqlens,
    chunk_offsets,
    num_sequences,
    scale,
    q_stride_t,
    q_stride_h,
    k_stride_t,
    k_stride_h,
    gate_stride_t,
    gate_stride_h,
    beta_stride_t,
    beta_stride_h,
    aqk_stride_t,
    aqk_stride_h,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Produce six off-diagonal Aqk/Akk blocks for one chunk and head."""
    global_chunk = tl.program_id(0)
    head = tl.program_id(1)
    if USE_INT64_OFFSETS:
        global_chunk = global_chunk.to(tl.int64)
        head = head.to(tl.int64)

    if IS_VARLEN:
        active_chunks = load_ragged_chunk_count(chunk_offsets, num_sequences)
        if global_chunk >= active_chunks:
            return
        _, _, token_start, valid_tokens = load_ragged_chunk_work(
            cu_seqlens,
            chunk_offsets,
            global_chunk,
            num_sequences,
            BT,
        )
        if USE_INT64_OFFSETS:
            token_start = token_start.to(tl.int64)
            valid_tokens = valid_tokens.to(tl.int64)
    else:
        token_start = global_chunk * BT
        valid_tokens = BT

    row = tl.arange(0, BC)
    key = tl.arange(0, BK)
    row_valid0 = row < tl.maximum(tl.minimum(valid_tokens, BC), 0)
    row_valid1 = row < tl.maximum(tl.minimum(valid_tokens - BC, BC), 0)
    row_valid2 = row < tl.maximum(tl.minimum(valid_tokens - 2 * BC, BC), 0)
    row_valid3 = row < tl.maximum(tl.minimum(valid_tokens - 3 * BC, BC), 0)
    has_block1 = valid_tokens > BC
    has_block2 = valid_tokens > 2 * BC
    has_block3 = valid_tokens > 3 * BC

    # BT64 has four BC16 blocks and therefore exactly six strict-lower pairs.
    # Keep the fixed MMA graph explicit so each accumulator maps directly to the K4 ABI.
    aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    akk10 = tl.zeros([BC, BC], dtype=tl.float32)
    akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    akk21 = tl.zeros([BC, BC], dtype=tl.float32)
    akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    for key_tile in tl.static_range(0, K, BK):
        key_offset = key_tile + key
        key_valid = key_offset < K

        block0_rows = token_start + row
        block1_rows = token_start + BC + row
        block2_rows = token_start + 2 * BC + row
        block3_rows = token_start + 3 * BC + row

        q1_ptr = q + ptr_offset(
            (block1_rows[:, None], head, key_offset[None, :]),
            (q_stride_t, q_stride_h, 1),
        )
        q2_ptr = q + ptr_offset(
            (block2_rows[:, None], head, key_offset[None, :]),
            (q_stride_t, q_stride_h, 1),
        )
        q3_ptr = q + ptr_offset(
            (block3_rows[:, None], head, key_offset[None, :]),
            (q_stride_t, q_stride_h, 1),
        )
        k0_ptr = k + ptr_offset(
            (block0_rows[:, None], head, key_offset[None, :]),
            (k_stride_t, k_stride_h, 1),
        )
        k1_ptr = k + ptr_offset(
            (block1_rows[:, None], head, key_offset[None, :]),
            (k_stride_t, k_stride_h, 1),
        )
        k2_ptr = k + ptr_offset(
            (block2_rows[:, None], head, key_offset[None, :]),
            (k_stride_t, k_stride_h, 1),
        )
        k3_ptr = k + ptr_offset(
            (block3_rows[:, None], head, key_offset[None, :]),
            (k_stride_t, k_stride_h, 1),
        )
        gate0_ptr = gate + ptr_offset(
            (block0_rows[:, None], head, key_offset[None, :]),
            (gate_stride_t, gate_stride_h, 1),
        )
        gate1_ptr = gate + ptr_offset(
            (block1_rows[:, None], head, key_offset[None, :]),
            (gate_stride_t, gate_stride_h, 1),
        )
        gate2_ptr = gate + ptr_offset(
            (block2_rows[:, None], head, key_offset[None, :]),
            (gate_stride_t, gate_stride_h, 1),
        )
        gate3_ptr = gate + ptr_offset(
            (block3_rows[:, None], head, key_offset[None, :]),
            (gate_stride_t, gate_stride_h, 1),
        )

        mask0 = row_valid0[:, None] & key_valid[None, :]
        mask1 = row_valid1[:, None] & key_valid[None, :]
        mask2 = row_valid2[:, None] & key_valid[None, :]
        mask3 = row_valid3[:, None] & key_valid[None, :]
        q1 = tl.load(q1_ptr, mask=mask1, other=0.0)
        q2 = tl.load(q2_ptr, mask=mask2, other=0.0)
        q3 = tl.load(q3_ptr, mask=mask3, other=0.0)
        k0 = tl.load(k0_ptr, mask=mask0, other=0.0)
        k1 = tl.load(k1_ptr, mask=mask1, other=0.0)
        k2 = tl.load(k2_ptr, mask=mask2, other=0.0)
        k3 = tl.load(k3_ptr, mask=mask3, other=0.0)
        gate0 = tl.load(gate0_ptr, mask=mask0, other=0.0).to(tl.float32)
        gate1 = tl.load(gate1_ptr, mask=mask1, other=0.0).to(tl.float32)
        gate2 = tl.load(gate2_ptr, mask=mask2, other=0.0).to(tl.float32)
        gate3 = tl.load(gate3_ptr, mask=mask3, other=0.0).to(tl.float32)

        reference1 = tl.load(
            gate
            + ptr_offset(
                (token_start + BC, head, key_offset),
                (gate_stride_t, gate_stride_h, 1),
            ),
            mask=has_block1 & key_valid,
            other=0.0,
        ).to(tl.float32)
        reference2 = tl.load(
            gate
            + ptr_offset(
                (token_start + 2 * BC, head, key_offset),
                (gate_stride_t, gate_stride_h, 1),
            ),
            mask=has_block2 & key_valid,
            other=0.0,
        ).to(tl.float32)
        reference3 = tl.load(
            gate
            + ptr_offset(
                (token_start + 3 * BC, head, key_offset),
                (gate_stride_t, gate_stride_h, 1),
            ),
            mask=has_block3 & key_valid,
            other=0.0,
        ).to(tl.float32)

        row_scale1 = masked_exp2(gate1 - reference1[None, :], mask1)
        row_scale2 = masked_exp2(gate2 - reference2[None, :], mask2)
        row_scale3 = masked_exp2(gate3 - reference3[None, :], mask3)
        qg1 = (q1 * row_scale1).to(q.dtype.element_ty)
        qg2 = (q2 * row_scale2).to(q.dtype.element_ty)
        qg3 = (q3 * row_scale3).to(q.dtype.element_ty)
        kg1 = (k1 * row_scale1).to(k.dtype.element_ty)
        kg2 = (k2 * row_scale2).to(k.dtype.element_ty)
        kg3 = (k3 * row_scale3).to(k.dtype.element_ty)

        k0g1 = (k0 * masked_exp2(reference1[None, :] - gate0, mask0)).to(k.dtype.element_ty)
        k0g2 = (k0 * masked_exp2(reference2[None, :] - gate0, mask0)).to(k.dtype.element_ty)
        k1g2 = (k1 * masked_exp2(reference2[None, :] - gate1, mask1)).to(k.dtype.element_ty)
        k0g3 = (k0 * masked_exp2(reference3[None, :] - gate0, mask0)).to(k.dtype.element_ty)
        k1g3 = (k1 * masked_exp2(reference3[None, :] - gate1, mask1)).to(k.dtype.element_ty)
        k2g3 = (k2 * masked_exp2(reference3[None, :] - gate2, mask2)).to(k.dtype.element_ty)

        aqk10 += tl.dot(qg1, tl.trans(k0g1))
        aqk20 += tl.dot(qg2, tl.trans(k0g2))
        aqk21 += tl.dot(qg2, tl.trans(k1g2))
        aqk30 += tl.dot(qg3, tl.trans(k0g3))
        aqk31 += tl.dot(qg3, tl.trans(k1g3))
        aqk32 += tl.dot(qg3, tl.trans(k2g3))
        akk10 += tl.dot(kg1, tl.trans(k0g1))
        akk20 += tl.dot(kg2, tl.trans(k0g2))
        akk21 += tl.dot(kg2, tl.trans(k1g2))
        akk30 += tl.dot(kg3, tl.trans(k0g3))
        akk31 += tl.dot(kg3, tl.trans(k1g3))
        akk32 += tl.dot(kg3, tl.trans(k2g3))

    beta_row = tl.arange(0, BC)
    beta1 = tl.load(
        beta
        + ptr_offset(
            (token_start + BC + beta_row, head),
            (beta_stride_t, beta_stride_h),
        ),
        mask=row_valid1,
        other=0.0,
    ).to(tl.float32)
    beta2 = tl.load(
        beta
        + ptr_offset(
            (token_start + 2 * BC + beta_row, head),
            (beta_stride_t, beta_stride_h),
        ),
        mask=row_valid2,
        other=0.0,
    ).to(tl.float32)
    beta3 = tl.load(
        beta
        + ptr_offset(
            (token_start + 3 * BC + beta_row, head),
            (beta_stride_t, beta_stride_h),
        ),
        mask=row_valid3,
        other=0.0,
    ).to(tl.float32)

    _store_k3_block(
        aqk,
        akk_od,
        aqk10,
        akk10 * beta1[:, None],
        row_valid1,
        row_valid0,
        token_start,
        global_chunk,
        head,
        aqk_stride_t,
        aqk_stride_h,
        scale,
        H,
        BC,
        0,
        1,
        0,
    )
    _store_k3_block(
        aqk,
        akk_od,
        aqk20,
        akk20 * beta2[:, None],
        row_valid2,
        row_valid0,
        token_start,
        global_chunk,
        head,
        aqk_stride_t,
        aqk_stride_h,
        scale,
        H,
        BC,
        1,
        2,
        0,
    )
    _store_k3_block(
        aqk,
        akk_od,
        aqk21,
        akk21 * beta2[:, None],
        row_valid2,
        row_valid1,
        token_start,
        global_chunk,
        head,
        aqk_stride_t,
        aqk_stride_h,
        scale,
        H,
        BC,
        2,
        2,
        1,
    )
    _store_k3_block(
        aqk,
        akk_od,
        aqk30,
        akk30 * beta3[:, None],
        row_valid3,
        row_valid0,
        token_start,
        global_chunk,
        head,
        aqk_stride_t,
        aqk_stride_h,
        scale,
        H,
        BC,
        3,
        3,
        0,
    )
    _store_k3_block(
        aqk,
        akk_od,
        aqk31,
        akk31 * beta3[:, None],
        row_valid3,
        row_valid1,
        token_start,
        global_chunk,
        head,
        aqk_stride_t,
        aqk_stride_h,
        scale,
        H,
        BC,
        4,
        3,
        1,
    )
    _store_k3_block(
        aqk,
        akk_od,
        aqk32,
        akk32 * beta3[:, None],
        row_valid3,
        row_valid2,
        token_start,
        global_chunk,
        head,
        aqk_stride_t,
        aqk_stride_h,
        scale,
        H,
        BC,
        5,
        3,
        2,
    )
    _zero_aqk_block(aqk, row_valid0, token_start, head, aqk_stride_t, aqk_stride_h, BC, 0, 1)
    _zero_aqk_block(aqk, row_valid0, token_start, head, aqk_stride_t, aqk_stride_h, BC, 0, 2)
    _zero_aqk_block(aqk, row_valid0, token_start, head, aqk_stride_t, aqk_stride_h, BC, 0, 3)
    _zero_aqk_block(aqk, row_valid1, token_start, head, aqk_stride_t, aqk_stride_h, BC, 1, 2)
    _zero_aqk_block(aqk, row_valid1, token_start, head, aqk_stride_t, aqk_stride_h, BC, 1, 3)
    _zero_aqk_block(aqk, row_valid2, token_start, head, aqk_stride_t, aqk_stride_h, BC, 2, 3)


def chunk_kda_fwd_k3b_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    aqk: torch.Tensor,
    scale: float,
    metadata: RaggedChunkMetadata | None,
) -> torch.Tensor:
    """Complete the caller-owned Aqk tensor and return the temporary Akk blocks."""
    batch, tokens, heads, key_dim = q.shape
    capacity = tokens // _CHUNK_SIZE if metadata is None else metadata.capacity
    if batch != 1 or key_dim != _HEAD_DIM or k.shape != q.shape:
        raise ValueError("Triton K3 requires B=1 and matching Q/K with K=128")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype:
        raise TypeError("Triton K3 requires matching FP16 or BF16 Q/K")
    if gate.shape != q.shape or gate.dtype != torch.float32:
        raise TypeError("Triton K3 requires an FP32 gate shaped like Q")
    if beta.shape != q.shape[:3] or beta.dtype != torch.float32:
        raise TypeError("Triton K3 requires FP32 beta shaped [1, T, H]")
    if aqk.shape != (batch, tokens, heads, _CHUNK_SIZE) or aqk.dtype != q.dtype:
        raise TypeError("Triton K3 requires Aqk shaped [1, T, H, 64] in the Q dtype")
    if metadata is None and tokens % _CHUNK_SIZE:
        raise ValueError("dense Triton K3 requires complete 64-token chunks")
    if metadata is not None:
        metadata.validate_chunk_size(_CHUNK_SIZE)

    akk_od = torch.empty(
        capacity * _OFFDIAGONAL_BLOCKS,
        heads * _SUBCHUNK_SIZE * _SUBCHUNK_SIZE,
        dtype=torch.float32,
        device=q.device,
    )
    if isinstance(q, FakeTensor) or capacity == 0:
        return akk_od

    q_view, k_view, gate_view = q[0], k[0], gate[0]
    beta_view, output_view = beta[0], aqk[0]
    cu_seqlens = None if metadata is None else metadata.cu_seqlens
    chunk_offsets = None if metadata is None else metadata.chunk_offsets
    chunk_kda_fwd_k3_triton_kernel[(capacity, heads)](
        q_view,
        k_view,
        gate_view,
        beta_view,
        output_view,
        akk_od,
        cu_seqlens,
        chunk_offsets,
        0 if metadata is None else metadata.cu_seqlens.shape[0] - 1,
        float(scale),
        q_view.stride(0),
        q_view.stride(1),
        k_view.stride(0),
        k_view.stride(1),
        gate_view.stride(0),
        gate_view.stride(1),
        beta_view.stride(0),
        beta_view.stride(1),
        output_view.stride(0),
        output_view.stride(1),
        H=heads,
        K=key_dim,
        BT=_CHUNK_SIZE,
        BC=_SUBCHUNK_SIZE,
        BK=32,
        USE_INT64_OFFSETS=requires_int64_offsets(
            q_view,
            k_view,
            gate_view,
            beta_view,
            output_view,
            akk_od,
            cu_seqlens,
            chunk_offsets,
        ),
        num_warps=4,
        num_stages=2,
    )
    return akk_od


__all__ = ["chunk_kda_fwd_k3b_triton"]
