# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
#
# Portions are derived from flash-linear-attention and licensed under the MIT license;
# see https://github.com/fla-org/flash-linear-attention/graphs/contributors.
# The remaining portions use the BSD-style license in the repository root.

"""Numerically safe scalar-GDN intra factors and W/U/kg production."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym.linear.kda.chunk_scheduler import (
    RaggedChunkMetadata,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)
from attn_gym.linear.kda.utils import autotune_cache_kwargs, exp2

SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32")


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.autotune(
    configs=[
        triton.Config({"BK": block_key}, num_warps=num_warps)
        for block_key in (32, 64)
        for num_warps in (1, 2, 4)
    ],
    key=["H", "HV", "K", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "num_sequences"])
def chunk_gdn_fwd_kkt_solve_kernel(
    k,
    g,
    beta,
    A,
    cu_seqlens,
    chunk_offsets,
    T,
    num_sequences,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel: compute beta * K @ K^T (lower triangular) + solve_tril (I+A)^{-1} in one pass.

    This kernel fuses chunk_scaled_dot_kkt_fwd and solve_tril into a single kernel,
    avoiding the HBM round-trip for the intermediate A matrix.

    Steps:
    1. Compute all 10 lower-triangular [BC, BC] blocks of beta * K @ K^T in registers
    2. Apply gate and beta scaling
    3. Forward substitution on diagonal blocks
    4. Block merge to get full (I+A)^{-1}
    5. Write result to A (output)
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1)
    i_b, i_h = i_bh // HV, i_bh % HV

    if IS_VARLEN:
        if i_t >= load_ragged_chunk_count(chunk_offsets, num_sequences):
            return
        i_n, i_t, token_start, _valid = load_ragged_chunk_work(
            cu_seqlens, chunk_offsets, i_t, num_sequences, BT
        )
        bos = token_start - i_t * BT
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    k += (bos * H + i_h // (HV // H)) * K
    A += (bos * HV + i_h) * BT

    o_i = tl.arange(0, BC)
    m_tc0 = (i_tc0 + o_i) < T
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T

    # load beta for each sub-chunk
    p_b0 = beta + bos * HV + i_h + (i_tc0 + o_i) * HV
    p_b1 = beta + bos * HV + i_h + (i_tc1 + o_i) * HV
    p_b2 = beta + bos * HV + i_h + (i_tc2 + o_i) * HV
    p_b3 = beta + bos * HV + i_h + (i_tc3 + o_i) * HV
    b_b0 = tl.load(p_b0, mask=m_tc0, other=0.0).to(tl.float32)
    b_b1 = tl.load(p_b1, mask=m_tc1, other=0.0).to(tl.float32)
    b_b2 = tl.load(p_b2, mask=m_tc2, other=0.0).to(tl.float32)
    b_b3 = tl.load(p_b3, mask=m_tc3, other=0.0).to(tl.float32)

    p_g0 = g + bos * HV + i_h + (i_tc0 + o_i) * HV
    p_g1 = g + bos * HV + i_h + (i_tc1 + o_i) * HV
    p_g2 = g + bos * HV + i_h + (i_tc2 + o_i) * HV
    p_g3 = g + bos * HV + i_h + (i_tc3 + o_i) * HV
    b_g0 = tl.load(p_g0, mask=m_tc0, other=0.0).to(tl.float32)
    b_g1 = tl.load(p_g1, mask=m_tc1, other=0.0).to(tl.float32)
    b_g2 = tl.load(p_g2, mask=m_tc2, other=0.0).to(tl.float32)
    b_g3 = tl.load(p_g3, mask=m_tc3, other=0.0).to(tl.float32)

    ############################################################################
    # Step 1: compute all 10 lower-triangular [BC, BC] blocks of K @ K^T
    ############################################################################

    # 4 diagonal blocks
    b_A00 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A11 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A22 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A33 = tl.zeros([BC, BC], dtype=tl.float32)

    # 6 off-diagonal blocks
    b_A10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_A32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        p_k0 = k + (i_tc0 + o_i)[:, None] * (H * K) + o_k[None, :]
        b_k0 = tl.load(p_k0, mask=m_tc0[:, None] & (o_k[None, :] < K), other=0.0)
        # diagonal block 0
        b_A00 += tl.dot(b_k0, tl.trans(b_k0))

        if i_tc1 < T:
            p_k1 = k + (i_tc1 + o_i)[:, None] * (H * K) + o_k[None, :]
            b_k1 = tl.load(p_k1, mask=m_tc1[:, None] & (o_k[None, :] < K), other=0.0)
            # diagonal block 1
            b_A11 += tl.dot(b_k1, tl.trans(b_k1))
            # off-diagonal (1,0)
            b_A10 += tl.dot(b_k1, tl.trans(b_k0))

            if i_tc2 < T:
                p_k2 = k + (i_tc2 + o_i)[:, None] * (H * K) + o_k[None, :]
                b_k2 = tl.load(p_k2, mask=m_tc2[:, None] & (o_k[None, :] < K), other=0.0)
                # diagonal block 2
                b_A22 += tl.dot(b_k2, tl.trans(b_k2))
                # off-diagonal (2,0), (2,1)
                b_A20 += tl.dot(b_k2, tl.trans(b_k0))
                b_A21 += tl.dot(b_k2, tl.trans(b_k1))

                if i_tc3 < T:
                    p_k3 = k + (i_tc3 + o_i)[:, None] * (H * K) + o_k[None, :]
                    b_k3 = tl.load(p_k3, mask=m_tc3[:, None] & (o_k[None, :] < K), other=0.0)
                    # diagonal block 3
                    b_A33 += tl.dot(b_k3, tl.trans(b_k3))
                    # off-diagonal (3,0), (3,1), (3,2)
                    b_A30 += tl.dot(b_k3, tl.trans(b_k0))
                    b_A31 += tl.dot(b_k3, tl.trans(b_k1))
                    b_A32 += tl.dot(b_k3, tl.trans(b_k2))

    ############################################################################
    # Step 2: apply gate and beta scaling
    ############################################################################

    # apply gate, beta scaling, and masking
    # m_d: strictly lower triangular mask for diagonal blocks
    # m_tc: boundary mask to prevent NaN from 0 * inf (IEEE 754) when
    #   out-of-bounds g loads as 0 via boundary_check and exp2(0 - g_inbounds) overflows
    m_d = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    mask00 = m_d & m_tc0[:, None] & m_tc0[None, :]
    mask11 = m_d & m_tc1[:, None] & m_tc1[None, :]
    mask22 = m_d & m_tc2[:, None] & m_tc2[None, :]
    mask33 = m_d & m_tc3[:, None] & m_tc3[None, :]
    mask10 = m_tc1[:, None] & m_tc0[None, :]
    mask20 = m_tc2[:, None] & m_tc0[None, :]
    mask21 = m_tc2[:, None] & m_tc1[None, :]
    mask30 = m_tc3[:, None] & m_tc0[None, :]
    mask31 = m_tc3[:, None] & m_tc1[None, :]
    mask32 = m_tc3[:, None] & m_tc2[None, :]
    b_A00 *= tl.where(mask00, exp2(tl.where(mask00, b_g0[:, None] - b_g0[None, :], 0.0)), 0.0)
    b_A11 *= tl.where(mask11, exp2(tl.where(mask11, b_g1[:, None] - b_g1[None, :], 0.0)), 0.0)
    b_A22 *= tl.where(mask22, exp2(tl.where(mask22, b_g2[:, None] - b_g2[None, :], 0.0)), 0.0)
    b_A33 *= tl.where(mask33, exp2(tl.where(mask33, b_g3[:, None] - b_g3[None, :], 0.0)), 0.0)
    b_A10 *= tl.where(mask10, exp2(tl.where(mask10, b_g1[:, None] - b_g0[None, :], 0.0)), 0.0)
    b_A20 *= tl.where(mask20, exp2(tl.where(mask20, b_g2[:, None] - b_g0[None, :], 0.0)), 0.0)
    b_A21 *= tl.where(mask21, exp2(tl.where(mask21, b_g2[:, None] - b_g1[None, :], 0.0)), 0.0)
    b_A30 *= tl.where(mask30, exp2(tl.where(mask30, b_g3[:, None] - b_g0[None, :], 0.0)), 0.0)
    b_A31 *= tl.where(mask31, exp2(tl.where(mask31, b_g3[:, None] - b_g1[None, :], 0.0)), 0.0)
    b_A32 *= tl.where(mask32, exp2(tl.where(mask32, b_g3[:, None] - b_g2[None, :], 0.0)), 0.0)

    # diagonal blocks: scaled by beta
    b_A00 = b_A00 * b_b0[:, None]
    b_A11 = b_A11 * b_b1[:, None]
    b_A22 = b_A22 * b_b2[:, None]
    b_A33 = b_A33 * b_b3[:, None]

    # off-diagonal blocks: full block, scaled by beta
    b_A10 = b_A10 * b_b1[:, None]
    b_A20 = b_A20 * b_b2[:, None]
    b_A21 = b_A21 * b_b2[:, None]
    b_A30 = b_A30 * b_b3[:, None]
    b_A31 = b_A31 * b_b3[:, None]
    b_A32 = b_A32 * b_b3[:, None]

    ############################################################################
    # Step 3: forward substitution on diagonal blocks -> (I + A_diag)^{-1}
    #
    # Same algorithm as solve_tril, but rows are extracted from in-register
    # [BC, BC] tensor via tl.sum(tl.where(mask, tensor, 0), 0) instead of
    # tl.load from HBM.
    ############################################################################

    b_Ai00 = -b_A00
    b_Ai11 = -b_A11
    b_Ai22 = -b_A22
    b_Ai33 = -b_A33

    for i in range(2, min(BC, T - i_tc0)):
        b_a00 = tl.sum(tl.where((o_i == i)[:, None], -b_A00, 0.0), 0)
        b_a00 = tl.where(o_i < i, b_a00, 0.0)
        b_a00 = b_a00 + tl.sum(b_a00[:, None] * b_Ai00, 0)
        b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
    for i in range(2, min(BC, T - i_tc1)):
        b_a11 = tl.sum(tl.where((o_i == i)[:, None], -b_A11, 0.0), 0)
        b_a11 = tl.where(o_i < i, b_a11, 0.0)
        b_a11 = b_a11 + tl.sum(b_a11[:, None] * b_Ai11, 0)
        b_Ai11 = tl.where((o_i == i)[:, None], b_a11, b_Ai11)
    for i in range(2, min(BC, T - i_tc2)):
        b_a22 = tl.sum(tl.where((o_i == i)[:, None], -b_A22, 0.0), 0)
        b_a22 = tl.where(o_i < i, b_a22, 0.0)
        b_a22 = b_a22 + tl.sum(b_a22[:, None] * b_Ai22, 0)
        b_Ai22 = tl.where((o_i == i)[:, None], b_a22, b_Ai22)
    for i in range(2, min(BC, T - i_tc3)):
        b_a33 = tl.sum(tl.where((o_i == i)[:, None], -b_A33, 0.0), 0)
        b_a33 = tl.where(o_i < i, b_a33, 0.0)
        b_a33 = b_a33 + tl.sum(b_a33[:, None] * b_Ai33, 0)
        b_Ai33 = tl.where((o_i == i)[:, None], b_a33, b_Ai33)

    b_Ai00 += m_I
    b_Ai11 += m_I
    b_Ai22 += m_I
    b_Ai33 += m_I

    ############################################################################
    # Step 4: block merge -> full (I + A)^{-1}
    ############################################################################

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_A10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai21 = -tl.dot(
        tl.dot(b_Ai22, b_A21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai11,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai32 = -tl.dot(
        tl.dot(b_Ai33, b_A32, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai22,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    b_Ai20 = -tl.dot(
        b_Ai22,
        tl.dot(b_A20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_A21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai31 = -tl.dot(
        b_Ai33,
        tl.dot(b_A31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_A32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    b_Ai30 = -tl.dot(
        b_Ai33,
        tl.dot(b_A30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_A31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
        + tl.dot(b_A32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )

    ############################################################################
    # Step 5: store full (I + A)^{-1} to output A
    ############################################################################

    p_A00 = A + (i_tc0 + o_i)[:, None] * (HV * BT) + o_i[None, :]
    p_A10 = A + (i_tc1 + o_i)[:, None] * (HV * BT) + o_i[None, :]
    p_A11 = A + (i_tc1 + o_i)[:, None] * (HV * BT) + (BC + o_i)[None, :]
    p_A20 = A + (i_tc2 + o_i)[:, None] * (HV * BT) + o_i[None, :]
    p_A21 = A + (i_tc2 + o_i)[:, None] * (HV * BT) + (BC + o_i)[None, :]
    p_A22 = A + (i_tc2 + o_i)[:, None] * (HV * BT) + (2 * BC + o_i)[None, :]
    p_A30 = A + (i_tc3 + o_i)[:, None] * (HV * BT) + o_i[None, :]
    p_A31 = A + (i_tc3 + o_i)[:, None] * (HV * BT) + (BC + o_i)[None, :]
    p_A32 = A + (i_tc3 + o_i)[:, None] * (HV * BT) + (2 * BC + o_i)[None, :]
    p_A33 = A + (i_tc3 + o_i)[:, None] * (HV * BT) + (3 * BC + o_i)[None, :]

    m_A0 = m_tc0[:, None] & (o_i[None, :] < BT)
    m_A1 = m_tc1[:, None] & (o_i[None, :] < BT)
    m_A2 = m_tc2[:, None] & (o_i[None, :] < BT)
    m_A3 = m_tc3[:, None] & (o_i[None, :] < BT)
    m_A11 = m_tc1[:, None] & ((BC + o_i)[None, :] < BT)
    m_A21 = m_tc2[:, None] & ((BC + o_i)[None, :] < BT)
    m_A22 = m_tc2[:, None] & ((2 * BC + o_i)[None, :] < BT)
    m_A31 = m_tc3[:, None] & ((BC + o_i)[None, :] < BT)
    m_A32 = m_tc3[:, None] & ((2 * BC + o_i)[None, :] < BT)
    m_A33 = m_tc3[:, None] & ((3 * BC + o_i)[None, :] < BT)

    tl.store(p_A00, b_Ai00.to(A.dtype.element_ty), mask=m_A0)
    tl.store(p_A10, b_Ai10.to(A.dtype.element_ty), mask=m_A1)
    tl.store(p_A11, b_Ai11.to(A.dtype.element_ty), mask=m_A11)
    tl.store(p_A20, b_Ai20.to(A.dtype.element_ty), mask=m_A2)
    tl.store(p_A21, b_Ai21.to(A.dtype.element_ty), mask=m_A21)
    tl.store(p_A22, b_Ai22.to(A.dtype.element_ty), mask=m_A22)
    tl.store(p_A30, b_Ai30.to(A.dtype.element_ty), mask=m_A3)
    tl.store(p_A31, b_Ai31.to(A.dtype.element_ty), mask=m_A31)
    tl.store(p_A32, b_Ai32.to(A.dtype.element_ty), mask=m_A32)
    tl.store(p_A33, b_Ai33.to(A.dtype.element_ty), mask=m_A33)


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "STORE_QG": lambda args: args["q"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "num_sequences"])
def scalar_recompute_w_u_kg_kernel(
    q,
    k,
    v,
    beta,
    inverse,
    cumulative_gate,
    w,
    u,
    restored_k,
    qg,
    cu_seqlens,
    chunk_offsets,
    T,
    num_sequences,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    STORE_QG: tl.constexpr,
):
    """Compute W/U and kg while K, beta, gate, and the solved transition are resident."""
    chunk = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // HV
    head = batch_head % HV
    key_head = head // (HV // H)
    row = tl.arange(0, BT)
    column = tl.arange(0, BT)
    if IS_VARLEN:
        if chunk >= load_ragged_chunk_count(chunk_offsets, num_sequences):
            return
        _sequence, _local_chunk, token_start, valid_tokens = load_ragged_chunk_work(
            cu_seqlens, chunk_offsets, chunk, num_sequences, BT
        )
        global_token = token_start + row
        token_mask = row < valid_tokens
    else:
        global_token = batch * T + chunk * BT + row
        token_mask = row < BT
        valid_tokens = BT

    beta_row = tl.load(beta + global_token * HV + head, mask=token_mask, other=0.0).to(tl.float32)
    gate = tl.load(cumulative_gate + global_token * HV + head, mask=token_mask, other=0.0).to(
        tl.float32
    )
    inverse_tile = tl.load(
        inverse + global_token[:, None] * (HV * BT) + head * BT + column[None, :],
        mask=token_mask[:, None],
        other=0.0,
    )

    for value_block in range(0, V, BV):
        value = value_block + tl.arange(0, BV)
        v_tile = tl.load(
            v + global_token[:, None] * (HV * V) + head * V + value[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        u_tile = tl.dot(inverse_tile, (v_tile * beta_row[:, None]).to(v_tile.dtype))
        tl.store(
            u + global_token[:, None] * (HV * V) + head * V + value[None, :],
            u_tile.to(u.dtype.element_ty),
            mask=token_mask[:, None],
        )

    final_gate = tl.sum(tl.where(row == valid_tokens - 1, gate, 0.0), axis=0)
    gate_exp = tl.exp2(gate)
    restore = tl.exp2(final_gate - gate)
    for key_block in range(0, K, BK):
        feature = key_block + tl.arange(0, BK)
        k_tile = tl.load(
            k + global_token[:, None] * (H * K) + key_head * K + feature[None, :],
            mask=token_mask[:, None],
            other=0.0,
        )
        weighted_k = k_tile * (beta_row * gate_exp)[:, None]
        w_tile = tl.dot(inverse_tile, weighted_k.to(k_tile.dtype))
        output_offset = global_token[:, None] * (HV * K) + head * K + feature[None, :]
        tl.store(w + output_offset, w_tile.to(w.dtype.element_ty), mask=token_mask[:, None])
        tl.store(
            restored_k + output_offset,
            (k_tile * restore[:, None]).to(k_tile.dtype),
            mask=token_mask[:, None],
        )
        if STORE_QG:
            q_tile = tl.load(
                q + global_token[:, None] * (H * K) + key_head * K + feature[None, :],
                mask=token_mask[:, None],
                other=0.0,
            )
            tl.store(
                qg + output_offset,
                (q_tile * gate_exp[:, None]).to(q_tile.dtype),
                mask=token_mask[:, None],
            )


def chunk_gdn_fwd_intra_dense(
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run dense GQA-capable BT64 scalar KKT/solve and W/U/kg."""
    batch, tokens, key_heads, key_dim = k.shape
    value_heads, value_dim = v.shape[2:]
    if (
        tokens % 64
        or key_dim != 128
        or value_dim != 128
        or v.shape[:2] != k.shape[:2]
        or value_heads % key_heads
    ):
        raise ValueError("dense fused chunk GDN requires BT64, K=V=128, and H % HK == 0")
    if (
        cumulative_gate.shape != (batch, tokens, value_heads)
        or beta.shape != cumulative_gate.shape
    ):
        raise ValueError("cumulative_gate and beta must follow value heads")
    chunks = tokens // 64
    inverse = torch.zeros(batch, tokens, value_heads, 64, dtype=k.dtype, device=k.device)
    chunk_gdn_fwd_kkt_solve_kernel[(chunks, batch * value_heads)](
        k=k,
        g=cumulative_gate,
        beta=beta,
        A=inverse,
        cu_seqlens=None,
        chunk_offsets=None,
        T=tokens,
        num_sequences=0,
        H=key_heads,
        HV=value_heads,
        K=key_dim,
        BT=64,
        BC=16,
    )

    w = k.new_empty(batch, tokens, value_heads, key_dim)
    u = torch.empty_like(v)
    restored_k = torch.empty_like(w)
    scalar_recompute_w_u_kg_kernel[(chunks, batch * value_heads)](
        q=None,
        k=k,
        v=v,
        beta=beta,
        inverse=inverse,
        cumulative_gate=cumulative_gate,
        w=w,
        u=u,
        restored_k=restored_k,
        qg=None,
        cu_seqlens=None,
        chunk_offsets=None,
        T=tokens,
        num_sequences=0,
        H=key_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BT=64,
        BK=64,
        BV=64,
        num_warps=4,
        num_stages=4,
    )
    return w, u, restored_k, inverse


def chunk_gdn_fwd_intra_packed(
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    metadata: RaggedChunkMetadata,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run fixed-capacity packed scalar KKT/solve and W/U/kg."""
    metadata.validate_chunk_size(64)
    batch, tokens, key_heads, key_dim = k.shape
    value_heads, value_dim = v.shape[2:]
    if (
        batch != 1
        or key_dim != 128
        or value_dim != 128
        or v.shape[:2] != k.shape[:2]
        or value_heads % key_heads
    ):
        raise ValueError("packed fused chunk GDN requires B=1, K=V=128, and H % HK == 0")
    if (
        cumulative_gate.shape != (batch, tokens, value_heads)
        or beta.shape != cumulative_gate.shape
    ):
        raise ValueError("cumulative_gate and beta must follow value heads")

    inverse = torch.zeros(batch, tokens, value_heads, 64, dtype=k.dtype, device=k.device)
    chunk_gdn_fwd_kkt_solve_kernel[(metadata.capacity, value_heads)](
        k=k,
        g=cumulative_gate,
        beta=beta,
        A=inverse,
        cu_seqlens=metadata.cu_seqlens,
        chunk_offsets=metadata.chunk_offsets,
        T=tokens,
        num_sequences=metadata.cu_seqlens.shape[0] - 1,
        H=key_heads,
        HV=value_heads,
        K=key_dim,
        BT=64,
        BC=16,
    )

    w = k.new_zeros(batch, tokens, value_heads, key_dim)
    u = torch.zeros_like(v)
    restored_k = torch.zeros_like(w)
    scalar_recompute_w_u_kg_kernel[(metadata.capacity, value_heads)](
        q=None,
        k=k,
        v=v,
        beta=beta,
        inverse=inverse,
        cumulative_gate=cumulative_gate,
        w=w,
        u=u,
        restored_k=restored_k,
        qg=None,
        cu_seqlens=metadata.cu_seqlens,
        chunk_offsets=metadata.chunk_offsets,
        T=tokens,
        num_sequences=metadata.cu_seqlens.shape[0] - 1,
        H=key_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BT=64,
        BK=64,
        BV=64,
        num_warps=4,
        num_stages=4,
    )
    return w, u, restored_k, inverse


def chunk_gdn_recompute_w_u_qg_kg(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    inverse: torch.Tensor,
    metadata: RaggedChunkMetadata | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute scalar W/U/qg/kg from the saved inverse without rebuilding KKT."""
    batch, tokens, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[2:]
    if (
        k.shape != q.shape
        or v.shape[:2] != q.shape[:2]
        or value_heads % key_heads
        or (key_dim, value_dim) != (128, 128)
    ):
        raise ValueError("fused chunk GDN recompute requires K=V=128 and H % HK == 0")
    if metadata is not None:
        metadata.validate_chunk_size(64)
        if batch != 1:
            raise ValueError("packed fused chunk GDN recompute requires B=1")
    elif tokens % 64:
        raise ValueError("dense fused chunk GDN recompute requires complete BT64 chunks")

    factory = torch.zeros if metadata is not None else torch.empty
    w = factory((batch, tokens, value_heads, key_dim), dtype=k.dtype, device=k.device)
    u = factory(v.shape, dtype=v.dtype, device=v.device)
    restored_k = factory(w.shape, dtype=w.dtype, device=w.device)
    qg = factory(w.shape, dtype=w.dtype, device=w.device)
    chunks = tokens // 64 if metadata is None else metadata.capacity
    cu_seqlens = None if metadata is None else metadata.cu_seqlens
    chunk_offsets = None if metadata is None else metadata.chunk_offsets
    num_sequences = 0 if metadata is None else metadata.cu_seqlens.shape[0] - 1
    scalar_recompute_w_u_kg_kernel[(chunks, batch * value_heads)](
        q=q,
        k=k,
        v=v,
        beta=beta,
        inverse=inverse,
        cumulative_gate=cumulative_gate,
        w=w,
        u=u,
        restored_k=restored_k,
        qg=qg,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=tokens,
        num_sequences=num_sequences,
        H=key_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BT=64,
        BK=64,
        BV=64,
        num_warps=4,
        num_stages=4,
    )
    return w, u, qg, restored_k


__all__ = [
    "chunk_gdn_fwd_intra_dense",
    "chunk_gdn_fwd_intra_packed",
    "chunk_gdn_recompute_w_u_qg_kg",
]
