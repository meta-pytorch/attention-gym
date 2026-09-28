# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe.

"""Blockwise inverse of the beta-folded chunk matrix ``I + strict_lower(M)``."""

import cutlass
import cutlass.experimental.primitives as nvvm
from cutlass import cute

from ..tile_dsl.mma import mma_step, mma_step_k8
from ..tile_dsl.pointwise import fp32_to_fp16, movmatrix_16b, opaque_i32_zero
from ..tile_dsl.swizzle import swizzle_xor_128b


@cute.jit
def invert_diagonal_NxN(cfg, in_base, out_base, d_idx, tidx, N: int = 8):
    """Gauss-Jordan inversion of one diagonal NxN block, ``in_base`` -> ``out_base`` (f16 SMEM)."""
    tidx_in_group = tidx % N
    BT = cfg.b_t

    row_coord = d_idx * N + tidx_in_group
    row_off = row_coord * BT + swizzle_xor_128b(row_coord, d_idx * N)
    row_ptr_in = in_base + row_off
    row_ptr = out_base + row_off

    row = [(row_ptr_in + j).load().to(cutlass.Float32) for j in range(N)]
    for i in cutlass.range_constexpr(N):
        row[i] = cutlass.Float32(1.0) if tidx_in_group == i else row[i]
    for src_row in cutlass.range_constexpr(N - 1):
        row_scale = -row[src_row]
        for i in cutlass.range_constexpr(src_row):
            shfl_val = nvvm.shfl_sync(
                0xFFFFFFFF, row[i], src_row, 0b1100000011111, kind=nvvm.Shfl.IDX
            )
            row[i] = row[i] + row_scale * shfl_val if tidx_in_group > src_row else row[i]
        row[src_row] = row_scale if tidx_in_group > src_row else row[src_row]

    for j in cutlass.range_constexpr(N):
        (row_ptr + j).store(row[j].to(cfg.io_dtype))


@cute.jit
def blockwise_diagonal_8x8_to_16x16(cfg, base, raw_base, d_idx, lane_idx):
    """Off-diagonal correction 8x8 -> 16x16 (C <- -D^{-1} C A^{-1}); raw C from ``raw_base``, writes on ``base``."""
    BT = cfg.b_t
    row_lo = d_idx + lane_idx % 8
    row_hi = row_lo + 8
    off_d_inv = row_hi * BT + swizzle_xor_128b(row_hi, d_idx + 8)
    off_c = row_hi * BT + swizzle_xor_128b(row_hi, d_idx)
    off_a_inv = row_lo * BT + swizzle_xor_128b(row_lo, d_idx)
    d_inv_frag = nvvm.ldmatrix(base + off_d_inv, 1, nvvm.MMALayout.ROW)
    c_frag = nvvm.ldmatrix(raw_base + off_c, 1, nvvm.MMALayout.COL)

    # ---- T = -(D^-1 @ C) -------------------------------------------------------------
    c_regs = cutlass.Array(cutlass.Float32, 4, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(4):
        c_regs[i] = cutlass.Float32(0.0)
    mma_step_k8(
        c_regs, [d_inv_frag, d_inv_frag], [c_frag], k_step=0, M=16, N=8, ab_dtype=cfg.io_dtype
    )
    for i in cutlass.range_constexpr(4):
        c_regs[i] = -c_regs[i]
    a_pack = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(2)]

    # ---- C = T @ A^-1 ----------------------------------------------------------------
    a_inv_frag = nvvm.ldmatrix(base + off_a_inv, 1, nvvm.MMALayout.COL)
    o_regs = cutlass.Array(cutlass.Float32, 4, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(4):
        o_regs[i] = cutlass.Float32(0.0)
    mma_step_k8(o_regs, a_pack, [a_inv_frag], k_step=0, M=16, N=8, ab_dtype=cfg.io_dtype)
    o_pack = fp32_to_fp16(o_regs[0], o_regs[1], dtype=cfg.io_dtype)

    # ---- store corrected C -----------------------------------------------------------
    nvvm.stmatrix(base + off_c, o_pack, nvvm.MMALayout.ROW)


@cute.jit
def blockwise_diagonal_16x16_to_32x32(cfg, base, raw_base, d_idx, lane_idx):
    """Off-diagonal correction 16x16 -> 32x32 (raw C from ``raw_base``)."""
    BT = cfg.b_t
    lane_row = lane_idx % 16
    lane_col = (lane_idx // 16) * 8
    row_lo = d_idx + lane_row
    row_hi = row_lo + 16
    off_d_inv = row_hi * BT + swizzle_xor_128b(row_hi, d_idx + 16 + lane_col)
    off_c = row_hi * BT + swizzle_xor_128b(row_hi, d_idx + lane_col)
    off_a_inv = row_lo * BT + swizzle_xor_128b(row_lo, d_idx + lane_col)
    d_inv_frags = list(nvvm.ldmatrix(base + off_d_inv, 4, nvvm.MMALayout.ROW))
    c_frags = list(nvvm.ldmatrix(raw_base + off_c, 4, nvvm.MMALayout.COL))

    # ---- T = -(D^-1 @ C) -------------------------------------------------------------
    c_regs = cutlass.Array(cutlass.Float32, 8, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(8):
        c_regs[i] = cutlass.Float32(0.0)
    mma_step(c_regs, d_inv_frags, c_frags, k_step=0, M=16, N=16, ab_dtype=cfg.io_dtype)
    for i in cutlass.range_constexpr(8):
        c_regs[i] = -c_regs[i]
    a_pack = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]

    # ---- C = T @ A^-1 ----------------------------------------------------------------
    a_inv_frags = list(nvvm.ldmatrix(base + off_a_inv, 4, nvvm.MMALayout.COL))
    o_regs = cutlass.Array(cutlass.Float32, 8, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(8):
        o_regs[i] = cutlass.Float32(0.0)
    mma_step(o_regs, a_pack, a_inv_frags, k_step=0, M=16, N=16, ab_dtype=cfg.io_dtype)
    o_pack = [fp32_to_fp16(o_regs[2 * j], o_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]

    # ---- store corrected C -----------------------------------------------------------
    nvvm.stmatrix(base + off_c, o_pack, nvvm.MMALayout.ROW)


@cute.jit
def blockwise_diagonal_32x32_to_64x64(
    cfg, base, raw_base, warp_id, lane_idx, barrier_id, barrier_threads
):
    """Off-diagonal correction 32x32 -> 64x64 (2 warps, one 16-row M-band each; raw C from ``raw_base``);
    the two bands meet on named barrier ``barrier_id`` (``barrier_threads`` arrivals) before the store."""
    band = warp_id % 2
    BT = cfg.b_t
    lane_row = lane_idx % 16
    lane_col = (lane_idx // 16) * 8
    row_d_inv = 32 + band * 16 + lane_row
    d_inv_frags = []
    for vs in cutlass.range_constexpr(2):
        d_inv_frags += list(
            nvvm.ldmatrix(
                base + row_d_inv * BT + swizzle_xor_128b(row_d_inv, 32 + vs * 16 + lane_col),
                4,
                nvvm.MMALayout.ROW,
            )
        )
    c_frags = []
    for vs in cutlass.range_constexpr(4):
        row_c = 32 + (vs // 2) * 16 + lane_row
        c_frags += list(
            nvvm.ldmatrix(
                raw_base + row_c * BT + swizzle_xor_128b(row_c, (vs % 2) * 16 + lane_col),
                4,
                nvvm.MMALayout.COL,
            )
        )

    # ---- T = -(D^-1 @ C) -------------------------------------------------------------
    c_regs = cutlass.Array(cutlass.Float32, 16, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(16):
        c_regs[i] = cutlass.Float32(0.0)
    for ks in cutlass.range_constexpr(2):
        mma_step(
            c_regs,
            d_inv_frags,
            c_frags[ks * 8 : ks * 8 + 8],
            k_step=ks,
            M=16,
            N=32,
            ab_dtype=cfg.io_dtype,
        )
    for i in cutlass.range_constexpr(16):
        c_regs[i] = -c_regs[i]
    a_pack = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(8)]

    # ---- C = T @ A^-1 ----------------------------------------------------------------
    a_inv_frags = []
    for vs in cutlass.range_constexpr(4):
        row_a_inv = (vs // 2) * 16 + lane_row
        a_inv_frags += list(
            nvvm.ldmatrix(
                base + row_a_inv * BT + swizzle_xor_128b(row_a_inv, (vs % 2) * 16 + lane_col),
                4,
                nvvm.MMALayout.COL,
            )
        )
    o_regs = cutlass.Array(cutlass.Float32, 16, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(16):
        o_regs[i] = cutlass.Float32(0.0)
    for ks in cutlass.range_constexpr(2):
        mma_step(
            o_regs,
            a_pack,
            a_inv_frags[ks * 8 : ks * 8 + 8],
            k_step=ks,
            M=16,
            N=32,
            ab_dtype=cfg.io_dtype,
        )
    o_pack = [fp32_to_fp16(o_regs[2 * j], o_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(8)]

    # ---- store corrected C -----------------------------------------------------------
    nvvm.barrier_cta_sync_aligned(barrier_id, thread_count=barrier_threads)
    nvvm.stmatrix(
        base + row_d_inv * BT + swizzle_xor_128b(row_d_inv, lane_col),
        o_pack[0:4],
        nvvm.MMALayout.ROW,
    )
    nvvm.stmatrix(
        base + row_d_inv * BT + swizzle_xor_128b(row_d_inv, 16 + lane_col),
        o_pack[4:8],
        nvvm.MMALayout.ROW,
    )


@cute.jit
def invert_unit_lower_16x16_fragments(cfg, l_regs, tinv_acc, lane_idx):
    """Blockwise inverse of ``I + L`` for one 16x16 strictly lower ``L`` held as an mma.sync m16n16 fp32 accumulator
    fragment (``l_regs``): the 4x4 diagonal blocks by their two-term series, then the 4 -> 8 and 8 -> 16 corrections
    ``C <- -D^-1 C A^-1`` on the same fragments; writes the fp32 inverse to ``tinv_acc``."""
    row_lo = lane_idx // 4
    col_lo = 2 * (lane_idx % 4)
    on_diagonal_block = (lane_idx // 16) == ((lane_idx % 4) // 2)
    below_diagonal_block = ((lane_idx // 16) == 1) and (((lane_idx % 4) // 2) == 0)
    zero = opaque_i32_zero()

    # ---- L4 = blockdiag_4(L), S = L4 @ L4 ---------------------------------------------
    l4 = cutlass.Array(cutlass.Float32, 8, alignment=16)
    for i in cutlass.range_constexpr(8):
        if cutlass.const_expr((i % 4) // 2 == i // 4):
            l4[i] = l_regs[i] if on_diagonal_block else cutlass.Float32(0.0)
        else:
            l4[i] = cutlass.Float32(0.0)
    l4_a0 = fp32_to_fp16(l4[0], l4[1], dtype=cfg.io_dtype)
    l4_a3 = fp32_to_fp16(l4[6], l4[7], dtype=cfg.io_dtype)
    s = cutlass.Array(cutlass.Float32, 8, alignment=16)
    for i in cutlass.range_constexpr(8):
        s[i] = cutlass.Float32(0.0)
    mma_step(
        s,
        (l4_a0, zero, zero, l4_a3),
        (movmatrix_16b(l4_a0), zero, zero, movmatrix_16b(l4_a3)),
        k_step=0,
        M=16,
        N=16,
        ab_dtype=cfg.io_dtype,
    )

    # ---- D4 = I - L4 + S - L4 @ S -------------------------------------------------------
    d4 = cutlass.Array(cutlass.Float32, 8, alignment=16)
    for i in cutlass.range_constexpr(8):
        if cutlass.const_expr((i % 4) // 2 == i // 4):
            eye = cutlass.Float32(1.0) if row_lo == col_lo + (i % 2) else cutlass.Float32(0.0)
            d4[i] = eye - l4[i] + s[i]
        else:
            d4[i] = cutlass.Float32(0.0)
    neg_s_b0 = movmatrix_16b(fp32_to_fp16(-s[0], -s[1], dtype=cfg.io_dtype))
    neg_s_b3 = movmatrix_16b(fp32_to_fp16(-s[6], -s[7], dtype=cfg.io_dtype))
    mma_step(
        d4,
        (l4_a0, zero, zero, l4_a3),
        (neg_s_b0, zero, zero, neg_s_b3),
        k_step=0,
        M=16,
        N=16,
        ab_dtype=cfg.io_dtype,
    )

    # ---- C4 = subdiag_4(L), M = D4 @ C4 -------------------------------------------------
    c4_lo0 = l_regs[0] if below_diagonal_block else cutlass.Float32(0.0)
    c4_lo1 = l_regs[1] if below_diagonal_block else cutlass.Float32(0.0)
    c4_hi0 = l_regs[6] if below_diagonal_block else cutlass.Float32(0.0)
    c4_hi1 = l_regs[7] if below_diagonal_block else cutlass.Float32(0.0)
    c4_b0 = movmatrix_16b(fp32_to_fp16(c4_lo0, c4_lo1, dtype=cfg.io_dtype))
    c4_b3 = movmatrix_16b(fp32_to_fp16(c4_hi0, c4_hi1, dtype=cfg.io_dtype))
    d4_a0 = fp32_to_fp16(d4[0], d4[1], dtype=cfg.io_dtype)
    d4_a3 = fp32_to_fp16(d4[6], d4[7], dtype=cfg.io_dtype)
    m = cutlass.Array(cutlass.Float32, 8, alignment=16)
    for i in cutlass.range_constexpr(8):
        m[i] = cutlass.Float32(0.0)
    mma_step(
        m,
        (d4_a0, zero, zero, d4_a3),
        (c4_b0, zero, zero, c4_b3),
        k_step=0,
        M=16,
        N=16,
        ab_dtype=cfg.io_dtype,
    )

    # ---- D8 = D4 - M @ D4 ---------------------------------------------------------------
    d8 = cutlass.Array(cutlass.Float32, 8, alignment=16)
    for i in cutlass.range_constexpr(8):
        d8[i] = d4[i]
    neg_m_a0 = fp32_to_fp16(-m[0], -m[1], dtype=cfg.io_dtype)
    neg_m_a3 = fp32_to_fp16(-m[6], -m[7], dtype=cfg.io_dtype)
    mma_step(
        d8,
        (neg_m_a0, zero, zero, neg_m_a3),
        (movmatrix_16b(d4_a0), zero, zero, movmatrix_16b(d4_a3)),
        k_step=0,
        M=16,
        N=16,
        ab_dtype=cfg.io_dtype,
    )

    # ---- C8 = L[8:16, 0:8], M = D8 @ C8 -------------------------------------------------
    c8_b1 = movmatrix_16b(fp32_to_fp16(l_regs[2], l_regs[3], dtype=cfg.io_dtype))
    d8_a0 = fp32_to_fp16(d8[0], d8[1], dtype=cfg.io_dtype)
    d8_a3 = fp32_to_fp16(d8[6], d8[7], dtype=cfg.io_dtype)
    m8 = cutlass.Array(cutlass.Float32, 4, alignment=16)
    for i in cutlass.range_constexpr(4):
        m8[i] = cutlass.Float32(0.0)
    mma_step(
        m8, (d8_a0, zero, zero, d8_a3), (zero, c8_b1), k_step=0, M=16, N=8, ab_dtype=cfg.io_dtype
    )

    # ---- T_inv = D8 - M @ D8 ------------------------------------------------------------
    for i in cutlass.range_constexpr(8):
        tinv_acc[i] = d8[i]
    neg_m8_a1 = fp32_to_fp16(-m8[2], -m8[3], dtype=cfg.io_dtype)
    mma_step(
        tinv_acc,
        (zero, neg_m8_a1, zero, zero),
        (movmatrix_16b(d8_a0), zero),
        k_step=0,
        M=16,
        N=8,
        ab_dtype=cfg.io_dtype,
    )
