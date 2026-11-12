# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/common/
# blockwise_inverse.py), rewritten by Attention Gym in 2026 on CuTeDSL 4.6.2 APIs: the tiles are
# flat CuTe SMEM tensors addressed through frost.ptx instead of cutlass.experimental pointers.

"""Blockwise inverse of the beta-folded chunk matrix ``I + strict_lower(M)``.

Every function works in place on one ``b_t x b_t`` 16-bit row-major tile with a 128-byte swizzle
(``ptx.swizzle_xor_128b``), given as a flat CuTe SMEM tensor. The steps double the inverted
diagonal: 8x8 Gauss-Jordan blocks, then ``C <- -D^-1 C A^-1`` corrections 8 -> 16 -> 32 -> 64.
"""

import cutlass
from cutlass import cute

from .ptx import (
    fp32_to_fp16,
    ldmatrix,
    mma_step,
    mma_step_k8,
    movmatrix_16b,
    opaque_i32_zero,
    shfl_idx,
    stmatrix,
    swizzle_xor_128b,
)

# Segmented shuffle within 8-lane groups: c = ((32 - 8) << 8) | 31.
_SEGMENT_8 = 0b1100000011111


@cute.jit
def _addr(tile, offset):
    return tile.iterator.toint() + offset * cutlass.Int32(2)


@cute.jit
def invert_diagonal_8x8(tile, d_idx, tid, b_t: cutlass.Constexpr[int], dtype: cutlass.Constexpr):
    """Gauss-Jordan inversion of diagonal block ``d_idx`` (8 lanes per block, one row each)."""
    n = 8
    tid_in_group = tid % n
    row_coord = d_idx * n + tid_in_group
    row_off = row_coord * b_t + swizzle_xor_128b(row_coord, d_idx * n)
    row = [tile[row_off + j].to(cutlass.Float32) for j in range(n)]
    for i in cutlass.range_constexpr(n):
        row[i] = cutlass.Float32(1.0) if tid_in_group == i else row[i]
    for src_row in cutlass.range_constexpr(n - 1):
        row_scale = -row[src_row]
        for i in cutlass.range_constexpr(src_row):
            shfl_val = shfl_idx(row[i], src_row, _SEGMENT_8)
            row[i] = row[i] + row_scale * shfl_val if tid_in_group > src_row else row[i]
        row[src_row] = row_scale if tid_in_group > src_row else row[src_row]
    for j in cutlass.range_constexpr(n):
        tile[row_off + j] = row[j].to(dtype)


@cute.jit
def blockwise_8x8_to_16x16(
    tile, d_idx, lane, b_t: cutlass.Constexpr[int], dtype: cutlass.Constexpr
):
    """Off-diagonal correction of the 16x16 block at row/column ``d_idx`` (one warp)."""
    row_lo = d_idx + lane % 8
    row_hi = row_lo + 8
    off_d_inv = row_hi * b_t + swizzle_xor_128b(row_hi, d_idx + 8)
    off_c = row_hi * b_t + swizzle_xor_128b(row_hi, d_idx)
    off_a_inv = row_lo * b_t + swizzle_xor_128b(row_lo, d_idx)
    d_inv_frag = ldmatrix(_addr(tile, off_d_inv), 1)[0]
    c_frag = ldmatrix(_addr(tile, off_c), 1, trans=True)[0]

    # T = -(D^-1 @ C)
    c_regs = cute.make_rmem_tensor((4,), cutlass.Float32)
    for i in cutlass.range_constexpr(4):
        c_regs[i] = cutlass.Float32(0.0)
    mma_step_k8(c_regs, [d_inv_frag, d_inv_frag], [c_frag], k_step=0, M=16, N=8, dtype=dtype)
    for i in cutlass.range_constexpr(4):
        c_regs[i] = -c_regs[i]
    a_pack = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=dtype) for j in range(2)]

    # C = T @ A^-1
    a_inv_frag = ldmatrix(_addr(tile, off_a_inv), 1, trans=True)[0]
    o_regs = cute.make_rmem_tensor((4,), cutlass.Float32)
    for i in cutlass.range_constexpr(4):
        o_regs[i] = cutlass.Float32(0.0)
    mma_step_k8(o_regs, a_pack, [a_inv_frag], k_step=0, M=16, N=8, dtype=dtype)
    stmatrix(_addr(tile, off_c), [fp32_to_fp16(o_regs[0], o_regs[1], dtype=dtype)])


@cute.jit
def blockwise_16x16_to_32x32(
    tile, d_idx, lane, b_t: cutlass.Constexpr[int], dtype: cutlass.Constexpr
):
    """Off-diagonal correction of the 32x32 block at row/column ``d_idx`` (one warp)."""
    lane_row = lane % 16
    lane_col = (lane // 16) * 8
    row_lo = d_idx + lane_row
    row_hi = row_lo + 16
    off_d_inv = row_hi * b_t + swizzle_xor_128b(row_hi, d_idx + 16 + lane_col)
    off_c = row_hi * b_t + swizzle_xor_128b(row_hi, d_idx + lane_col)
    off_a_inv = row_lo * b_t + swizzle_xor_128b(row_lo, d_idx + lane_col)
    d_inv_frags = ldmatrix(_addr(tile, off_d_inv), 4)
    c_frags = ldmatrix(_addr(tile, off_c), 4, trans=True)

    c_regs = cute.make_rmem_tensor((8,), cutlass.Float32)
    for i in cutlass.range_constexpr(8):
        c_regs[i] = cutlass.Float32(0.0)
    mma_step(c_regs, d_inv_frags, c_frags, k_step=0, M=16, N=16, dtype=dtype)
    for i in cutlass.range_constexpr(8):
        c_regs[i] = -c_regs[i]
    a_pack = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=dtype) for j in range(4)]

    a_inv_frags = ldmatrix(_addr(tile, off_a_inv), 4, trans=True)
    o_regs = cute.make_rmem_tensor((8,), cutlass.Float32)
    for i in cutlass.range_constexpr(8):
        o_regs[i] = cutlass.Float32(0.0)
    mma_step(o_regs, a_pack, a_inv_frags, k_step=0, M=16, N=16, dtype=dtype)
    o_pack = [fp32_to_fp16(o_regs[2 * j], o_regs[2 * j + 1], dtype=dtype) for j in range(4)]
    stmatrix(_addr(tile, off_c), o_pack)


@cute.jit
def blockwise_32x32_to_64x64(
    tile, band, lane, barrier, b_t: cutlass.Constexpr[int], dtype: cutlass.Constexpr
):
    """Off-diagonal correction of the 64x64 tile (two warps, one 16-row band each).

    Both bands read the raw C block before either stores, so they meet on ``barrier`` (a
    ``NamedBarrier`` over the compute group) first."""
    lane_row = lane % 16
    lane_col = (lane // 16) * 8
    row_d_inv = 32 + band * 16 + lane_row
    d_inv_frags = []
    for vs in cutlass.range_constexpr(2):
        d_inv_frags += ldmatrix(
            _addr(tile, row_d_inv * b_t + swizzle_xor_128b(row_d_inv, 32 + vs * 16 + lane_col)), 4
        )
    c_frags = []
    for vs in cutlass.range_constexpr(4):
        row_c = 32 + (vs // 2) * 16 + lane_row
        c_frags += ldmatrix(
            _addr(tile, row_c * b_t + swizzle_xor_128b(row_c, (vs % 2) * 16 + lane_col)),
            4,
            trans=True,
        )

    c_regs = cute.make_rmem_tensor((16,), cutlass.Float32)
    for i in cutlass.range_constexpr(16):
        c_regs[i] = cutlass.Float32(0.0)
    for ks in cutlass.range_constexpr(2):
        mma_step(
            c_regs, d_inv_frags, c_frags[ks * 8 : ks * 8 + 8], k_step=ks, M=16, N=32, dtype=dtype
        )
    for i in cutlass.range_constexpr(16):
        c_regs[i] = -c_regs[i]
    a_pack = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=dtype) for j in range(8)]

    a_inv_frags = []
    for vs in cutlass.range_constexpr(4):
        row_a_inv = (vs // 2) * 16 + lane_row
        a_inv_frags += ldmatrix(
            _addr(tile, row_a_inv * b_t + swizzle_xor_128b(row_a_inv, (vs % 2) * 16 + lane_col)),
            4,
            trans=True,
        )
    o_regs = cute.make_rmem_tensor((16,), cutlass.Float32)
    for i in cutlass.range_constexpr(16):
        o_regs[i] = cutlass.Float32(0.0)
    for ks in cutlass.range_constexpr(2):
        mma_step(
            o_regs, a_pack, a_inv_frags[ks * 8 : ks * 8 + 8], k_step=ks, M=16, N=32, dtype=dtype
        )
    o_pack = [fp32_to_fp16(o_regs[2 * j], o_regs[2 * j + 1], dtype=dtype) for j in range(8)]

    barrier.sync(aligned=True)
    stmatrix(_addr(tile, row_d_inv * b_t + swizzle_xor_128b(row_d_inv, lane_col)), o_pack[0:4])
    stmatrix(
        _addr(tile, row_d_inv * b_t + swizzle_xor_128b(row_d_inv, 16 + lane_col)), o_pack[4:8]
    )


@cute.jit
def invert_unit_lower_16x16_fragments(l_regs, lane, dtype: cutlass.Constexpr):
    """Inverse of ``I + L`` for one strictly lower 16x16 ``L`` held as an m16n16 mma.sync fp32
    accumulator fragment (8 values per lane), for the BT = 16 KDA kernels: the 4x4 diagonal
    blocks by their two-term series, then the 4 -> 8 and 8 -> 16 corrections
    ``C <- -D^-1 C A^-1`` on the same fragments. Returns the fp32 inverse fragment."""
    row_lo = lane // 4
    col_lo = 2 * (lane % 4)
    on_diagonal_block = (lane // 16) == ((lane % 4) // 2)
    below_diagonal_block = ((lane // 16) == 1) and (((lane % 4) // 2) == 0)
    zero = opaque_i32_zero()
    f0 = cutlass.Float32(0.0)

    # ---- L4 = blockdiag_4(L), S = L4 @ L4 -------------------------------------------
    l4 = cute.make_rmem_tensor((8,), cutlass.Float32)
    for i in cutlass.range_constexpr(8):
        if cutlass.const_expr((i % 4) // 2 == i // 4):
            l4[i] = l_regs[i] if on_diagonal_block else f0
        else:
            l4[i] = f0
    l4_a0 = fp32_to_fp16(l4[0], l4[1], dtype=dtype)
    l4_a3 = fp32_to_fp16(l4[6], l4[7], dtype=dtype)
    s = cute.make_rmem_tensor((8,), cutlass.Float32)
    for i in cutlass.range_constexpr(8):
        s[i] = f0
    mma_step(
        s,
        (l4_a0, zero, zero, l4_a3),
        (movmatrix_16b(l4_a0), zero, zero, movmatrix_16b(l4_a3)),
        k_step=0,
        M=16,
        N=16,
        dtype=dtype,
    )

    # ---- D4 = I - L4 + S - L4 @ S -----------------------------------------------------
    d4 = cute.make_rmem_tensor((8,), cutlass.Float32)
    for i in cutlass.range_constexpr(8):
        if cutlass.const_expr((i % 4) // 2 == i // 4):
            eye = cutlass.Float32(1.0) if row_lo == col_lo + (i % 2) else f0
            d4[i] = eye - l4[i] + s[i]
        else:
            d4[i] = f0
    neg_s_b0 = movmatrix_16b(fp32_to_fp16(-s[0], -s[1], dtype=dtype))
    neg_s_b3 = movmatrix_16b(fp32_to_fp16(-s[6], -s[7], dtype=dtype))
    mma_step(
        d4,
        (l4_a0, zero, zero, l4_a3),
        (neg_s_b0, zero, zero, neg_s_b3),
        k_step=0,
        M=16,
        N=16,
        dtype=dtype,
    )

    # ---- C4 = subdiag_4(L), M = D4 @ C4 -----------------------------------------------
    c4_lo0 = l_regs[0] if below_diagonal_block else f0
    c4_lo1 = l_regs[1] if below_diagonal_block else f0
    c4_hi0 = l_regs[6] if below_diagonal_block else f0
    c4_hi1 = l_regs[7] if below_diagonal_block else f0
    c4_b0 = movmatrix_16b(fp32_to_fp16(c4_lo0, c4_lo1, dtype=dtype))
    c4_b3 = movmatrix_16b(fp32_to_fp16(c4_hi0, c4_hi1, dtype=dtype))
    d4_a0 = fp32_to_fp16(d4[0], d4[1], dtype=dtype)
    d4_a3 = fp32_to_fp16(d4[6], d4[7], dtype=dtype)
    m = cute.make_rmem_tensor((8,), cutlass.Float32)
    for i in cutlass.range_constexpr(8):
        m[i] = f0
    mma_step(
        m,
        (d4_a0, zero, zero, d4_a3),
        (c4_b0, zero, zero, c4_b3),
        k_step=0,
        M=16,
        N=16,
        dtype=dtype,
    )

    # ---- D8 = D4 - M @ D4 -------------------------------------------------------------
    d8 = cute.make_rmem_tensor((8,), cutlass.Float32)
    for i in cutlass.range_constexpr(8):
        d8[i] = d4[i]
    neg_m_a0 = fp32_to_fp16(-m[0], -m[1], dtype=dtype)
    neg_m_a3 = fp32_to_fp16(-m[6], -m[7], dtype=dtype)
    mma_step(
        d8,
        (neg_m_a0, zero, zero, neg_m_a3),
        (movmatrix_16b(d4_a0), zero, zero, movmatrix_16b(d4_a3)),
        k_step=0,
        M=16,
        N=16,
        dtype=dtype,
    )

    # ---- C8 = L[8:16, 0:8], M = D8 @ C8 -----------------------------------------------
    c8_b1 = movmatrix_16b(fp32_to_fp16(l_regs[2], l_regs[3], dtype=dtype))
    d8_a0 = fp32_to_fp16(d8[0], d8[1], dtype=dtype)
    d8_a3 = fp32_to_fp16(d8[6], d8[7], dtype=dtype)
    m8 = cute.make_rmem_tensor((4,), cutlass.Float32)
    for i in cutlass.range_constexpr(4):
        m8[i] = f0
    mma_step(m8, (d8_a0, zero, zero, d8_a3), (zero, c8_b1), k_step=0, M=16, N=8, dtype=dtype)

    # ---- T_inv = D8 - M @ D8 ----------------------------------------------------------
    tinv = cute.make_rmem_tensor((8,), cutlass.Float32)
    for i in cutlass.range_constexpr(8):
        tinv[i] = d8[i]
    neg_m8_a1 = fp32_to_fp16(-m8[2], -m8[3], dtype=dtype)
    mma_step(
        tinv,
        (zero, neg_m8_a1, zero, zero),
        (movmatrix_16b(d8_a0), zero),
        k_step=0,
        M=16,
        N=8,
        dtype=dtype,
    )
    return tinv
