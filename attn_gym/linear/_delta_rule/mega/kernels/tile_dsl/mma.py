# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Modified by Attention Gym in 2026: block-scale, multi-CTA, and ldmatrix B-fragment helpers
# unused by the vendored kernels were removed.



import cutlass
from cutlass import cute
from cutlass.cute.arch.nvvm_wrappers import inline_ptx
from cutlass.experimental import primitives as nvvm


@cute.jit
def mma_ss(
    desc,
    desc_a_base,
    desc_b_base,
    tmem_c,
    accumulate: bool = False,
    k_start: int = 0,
    k_count=None,
):
    """Issue the K loop of one SMEM x SMEM tcgen05 MMA from the electing thread."""
    intra_a = desc.smem_advance_A_intra
    intra_b = desc.smem_advance_B_intra
    subtile_a = desc.smem_subtile_A
    subtile_b = desc.smem_subtile_B
    sps_a = desc.sps_A
    sps_b = desc.sps_B
    k_count = desc.num_k_steps if cutlass.const_expr(k_count is None) else k_count
    enable_input_d = accumulate
    for kk in cutlass.range_constexpr(k_count):
        k = k_start + kk
        inc_a = (intra_a * (k % sps_a) + subtile_a * (k // sps_a)) >> 4
        inc_b = (intra_b * (k % sps_b) + subtile_b * (k // sps_b)) >> 4
        da = desc_a_base + inc_a
        db = desc_b_base + inc_b
        if nvvm.elect_sync():
            nvvm.tcgen05_mma(
                desc.kind, nvvm.CTAGroup.CTA_1, tmem_c, da, db, desc.idesc, enable_input_d
            )
        enable_input_d = True


@cute.jit
def mma_ts_step(desc, tmem_a_base, desc_b_base, tmem_c, k_idx: int, accumulate):
    """Issue one K step of a TMEM x SMEM tcgen05 MMA from the electing thread."""
    dp = tmem_a_base.subview(k_idx * desc.tmem_advance_A)
    db = desc_b_base + ((desc.smem_advance_B_intra * k_idx) >> 4)
    if nvvm.elect_sync():
        nvvm.tcgen05_mma(desc.kind, nvvm.CTAGroup.CTA_1, tmem_c, dp, db, desc.idesc, accumulate)


@cute.jit
def mma_step(
    acc,
    a_frag,
    b_frag,
    *,
    k_step: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    ab_dtype: cutlass.Constexpr[type[cutlass.Numeric]] = cutlass.Float16,
):
    if cutlass.const_expr(M % 16 != 0):
        raise ValueError(f"mma_step: M must be a multiple of 16, got M={M}")
    if cutlass.const_expr(ab_dtype != cutlass.Float16 and ab_dtype != cutlass.BFloat16):
        raise TypeError(f"mma_step: ab_dtype must be Float16 or BFloat16, got {ab_dtype}")
    M_BLOCKS = M // 16
    N_FRAGS = N // 8
    a_stride = len(a_frag) // M_BLOCKS

    ab_tag = "f16" if cutlass.const_expr(ab_dtype == cutlass.Float16) else "bf16"
    mma_ptx = (
        f"mma.sync.aligned.m16n8k16.row.col.f32.{ab_tag}.{ab_tag}.f32"
        " {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};"
    )

    for m_block in cutlass.range_constexpr(M_BLOCKS):
        a_off = m_block * a_stride + k_step * 4
        a0 = a_frag[a_off + 0]
        a1 = a_frag[a_off + 1]
        a2 = a_frag[a_off + 2]
        a3 = a_frag[a_off + 3]
        acc_base = m_block * N_FRAGS * 4
        for n_frag in cutlass.range_constexpr(N_FRAGS):
            b0 = b_frag[n_frag * 2 + 0]
            b1 = b_frag[n_frag * 2 + 1]
            s_off = acc_base + n_frag * 4
            c0, c1, c2, c3 = inline_ptx(
                mma_ptx,
                write_only_types=[
                    cutlass.Float32,
                    cutlass.Float32,
                    cutlass.Float32,
                    cutlass.Float32,
                ],
                read_only_args=[
                    a0,
                    a1,
                    a2,
                    a3,
                    b0,
                    b1,
                    acc[s_off + 0],
                    acc[s_off + 1],
                    acc[s_off + 2],
                    acc[s_off + 3],
                ],
            )
            acc[s_off + 0] = c0
            acc[s_off + 1] = c1
            acc[s_off + 2] = c2
            acc[s_off + 3] = c3


@cute.jit
def mma_step_k8(
    acc,
    a_frag,
    b_frag,
    *,
    k_step: cutlass.Constexpr[int],
    M: cutlass.Constexpr[int],
    N: cutlass.Constexpr[int],
    ab_dtype: cutlass.Constexpr[type[cutlass.Numeric]] = cutlass.Float16,
):
    if cutlass.const_expr(M % 16 != 0):
        raise ValueError(f"mma_step_k8: M must be a multiple of 16, got M={M}")
    if cutlass.const_expr(ab_dtype != cutlass.Float16 and ab_dtype != cutlass.BFloat16):
        raise TypeError(f"mma_step_k8: ab_dtype must be Float16 or BFloat16, got {ab_dtype}")
    M_BLOCKS = M // 16
    N_FRAGS = N // 8
    a_stride = len(a_frag) // M_BLOCKS

    ab_tag = "f16" if cutlass.const_expr(ab_dtype == cutlass.Float16) else "bf16"
    mma_ptx = (
        f"mma.sync.aligned.m16n8k8.row.col.f32.{ab_tag}.{ab_tag}.f32"
        " {$0,$1,$2,$3}, {$4,$5}, {$6}, {$7,$8,$9,$10};"
    )

    for m_block in cutlass.range_constexpr(M_BLOCKS):
        a_off = m_block * a_stride + k_step * 2
        a0 = a_frag[a_off + 0]
        a1 = a_frag[a_off + 1]
        acc_base = m_block * N_FRAGS * 4
        for n_frag in cutlass.range_constexpr(N_FRAGS):
            b0 = b_frag[n_frag]
            s_off = acc_base + n_frag * 4
            c0, c1, c2, c3 = inline_ptx(
                mma_ptx,
                write_only_types=[
                    cutlass.Float32,
                    cutlass.Float32,
                    cutlass.Float32,
                    cutlass.Float32,
                ],
                read_only_args=[
                    a0,
                    a1,
                    b0,
                    acc[s_off + 0],
                    acc[s_off + 1],
                    acc[s_off + 2],
                    acc[s_off + 3],
                ],
            )
            acc[s_off + 0] = c0
            acc[s_off + 1] = c1
            acc[s_off + 2] = c2
            acc[s_off + 3] = c3
