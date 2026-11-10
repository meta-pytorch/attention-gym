# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors
# NOTE: do NOT use `from __future__ import annotations` — cute.struct
# requires eager-evaluated annotations.


import cuda.bindings.driver as cuda
import cutlass
from cutlass import Boolean, Float32, Int32, cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_tensor
from cutlass.cutlass_dsl import Constexpr

from attn_gym._backends.cute.utils import cute_dtype
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_intra import (
    _MIN_ALIGN_BYTES,
    _MIN_ALIGN_ELEMENTS_16BIT,
    BC,
    BT,
    K_PHASES,
    KC_TOTAL,
    KEY_DIM,
    KEY_DIM_PER_CTA,
    SUBCHUNKS,
    _copy_qkg_smem_cpasync,
    _cvt_half2_f32,
    _hmma_load_b_group_staged,
    _hmma_load_b_groups_smem_ldmatrix,
    _hmma_load_da,
    _hmma_load_da_pair,
    _hmma_stage_b_group,
    _ld_global_f32x2_pred,
    _ld_global_f32x4_lo2_pred,
    _mma_tf32_m16n8k8,
    _st_global_16b_ldmatrix_epilogue_16x32,
    _st_global_f32_b32_pred,
    _st_global_f32_triton_shared_epilogue_16x32,
    _st_shared_dg_triton_swizzle,
)
from attn_gym.linear.kda.constants import LN2, is_sm100_kda_capability
from attn_gym.linear.kda.fwd.cute.chunk_scheduler_cute import (
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)
from attn_gym.warpdrive.cache import (
    compile_tvm_ffi,
    get_compile_target,
    jit_cache,
)
from attn_gym.warpdrive.compat import SmemAllocator


@cute.kernel
def legacy_chunk_kda_bwd_intra_hmma_grid_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mG: cute.Tensor,
    mBeta: cute.Tensor,
    mdAqk: cute.Tensor,
    mdAkk: cute.Tensor,
    mDq: cute.Tensor,
    mDk: cute.Tensor,
    mDb2: cute.Tensor,
    mDg: cute.Tensor,
    mDq2: cute.Tensor,
    mDk2: cute.Tensor,
    mDg2: cute.Tensor,
    mCuSeqlens: cute.Tensor | None,
    mChunkOffsets: cute.Tensor | None,
    grid_chunks: Int32,
    ragged: Constexpr,
    use_int64_offsets: Constexpr,
    use_packed_f32x2: Constexpr,
    use_stmatrix: Constexpr,
    fastmath: Constexpr = True,
):
    tidx, _, _ = cute.arch.thread_idx()
    # Grid is (KC_TOTAL, grid_chunks, H), mirroring Triton's for-loop variant.
    # kc and head are real grid axes; only the chunk axis is capped at
    # grid_chunks and walked by each CTA. These per-(kc, head) coordinates are
    # loop-invariant, so they are computed once outside the chunk loop.
    kc_idx, chunk_start, head_idx = cute.arch.block_idx()
    gid = tidx // 4
    tid = tidx % 4
    k_phase = kc_idx // SUBCHUNKS
    subchunk_idx = kc_idx - k_phase * SUBCHUNKS
    k_start = k_phase * KEY_DIM_PER_CTA
    row_base = subchunk_idx * BC
    row0 = row_base + gid
    row1 = row0 + 8

    # Both bounds are runtime values: packed replays read the active count
    # on-device, while dense launches derive it from the symbolic token extent.
    if cutlass.const_expr(ragged):
        active_chunks = load_ragged_chunk_count(mChunkOffsets)
    else:
        active_chunks = cute.size(mQ.shape[1]) // BT
    iters = (active_chunks - chunk_start + grid_chunks - 1) // grid_chunks

    # SMEM is allocated once and reused across the chunk-loop iterations.
    smem = SmemAllocator()
    sQ_tile = smem.allocate_tensor(
        element_type=mQ.element_type,
        layout=cute.make_layout((BT * KEY_DIM_PER_CTA,), stride=(1,)),
        byte_alignment=128,
        swizzle=None,
    )
    sK_tile = smem.allocate_tensor(
        element_type=mK.element_type,
        layout=cute.make_layout((BT * KEY_DIM_PER_CTA,), stride=(1,)),
        byte_alignment=128,
        swizzle=None,
    )
    sG_tile = smem.allocate_tensor(
        element_type=cutlass.Float32,
        layout=cute.make_layout((BT * KEY_DIM_PER_CTA,), stride=(1,)),
        byte_alignment=128,
        swizzle=None,
    )
    sBeta_tile = smem.allocate_tensor(
        element_type=cutlass.Float32,
        layout=cute.make_layout((BT,), stride=(1,)),
        byte_alignment=128,
        swizzle=None,
    )
    sEpi_tile = smem.allocate_tensor(
        element_type=mDq2.element_type,
        layout=cute.make_layout((2048,), stride=(1,)),
        byte_alignment=128,
        swizzle=None,
    )

    for chunk_iter in cutlass.range(iters, unroll=1):
        chunk_block = chunk_start + chunk_iter * grid_chunks
        if cutlass.const_expr(ragged):
            _, _, row_start, valid = load_ragged_chunk_work(
                mCuSeqlens,
                mChunkOffsets,
                chunk_block,
                Int32(BT),
            )
        else:
            row_start = chunk_block * BT
            valid = Int32(BT)
        if cutlass.const_expr(use_int64_offsets):
            row_start = cutlass.Int64(row_start)
        # Match Triton's default safe-gate normalization reference
        # (causal_gate_normref=False): use the middle row of this BC subchunk,
        # clamped for partial chunks. This keeps local diagonal dA scaling
        # finite for dense causal dA without relying on a near-diagonal dA
        # property.
        ref_prev = cutlass.min(row_base, valid - 1)
        ref_qk = cutlass.min(row_base + BC // 2, valid - 1)
        ref_future = cutlass.min(row_base + BC, valid) - 1
        daqk_row_stride = mdAqk.layout.stride[1]
        daqk_head_stride = mdAqk.layout.stride[2]
        dakk_row_stride = mdAkk.layout.stride[1]
        dakk_head_stride = mdAkk.layout.stride[2]
        dq_row_stride = mDq.layout.stride[1]
        dq_head_stride = mDq.layout.stride[2]
        dq2_row_stride = mDq2.layout.stride[1]
        dq2_head_stride = mDq2.layout.stride[2]
        dk_row_stride = mDk.layout.stride[1]
        dk_head_stride = mDk.layout.stride[2]
        dk2_row_stride = mDk2.layout.stride[1]
        dk2_head_stride = mDk2.layout.stride[2]
        dg_row_stride = mDg.layout.stride[1]
        dg_head_stride = mDg.layout.stride[2]
        dg2_row_stride = mDg2.layout.stride[1]
        dg2_head_stride = mDg2.layout.stride[2]
        db2_k_stride = mDb2.layout.stride[0]
        db2_row_stride = mDb2.layout.stride[1]
        db2_head_stride = mDb2.layout.stride[2]

        z = Float32(0.0)
        beta_row0 = tidx
        beta_row1 = tidx + 32
        beta_val0 = z
        beta_val1 = z
        if (beta_row0 >= row_base) & (beta_row0 < valid):
            beta_val0 = mBeta[0, row_start + beta_row0, head_idx]
        if (beta_row1 >= row_base) & (beta_row1 < valid):
            beta_val1 = mBeta[0, row_start + beta_row1, head_idx]
        sBeta_tile[beta_row0] = beta_val0
        sBeta_tile[beta_row1] = beta_val1
        _copy_qkg_smem_cpasync(
            sQ_tile,
            sK_tile,
            sG_tile,
            mQ,
            mK,
            mG,
            row_start,
            head_idx,
            k_start,
            valid,
            tidx,
        )
        gref_qk0 = sG_tile[ref_qk * KEY_DIM_PER_CTA + gid]
        gref_qk1 = sG_tile[ref_qk * KEY_DIM_PER_CTA + 8 + gid]
        gref_qk2 = sG_tile[ref_qk * KEY_DIM_PER_CTA + 16 + gid]
        gref_qk3 = sG_tile[ref_qk * KEY_DIM_PER_CTA + 24 + gid]
        gref_prev0 = sG_tile[ref_prev * KEY_DIM_PER_CTA + gid]
        gref_prev1 = sG_tile[ref_prev * KEY_DIM_PER_CTA + 8 + gid]
        gref_prev2 = sG_tile[ref_prev * KEY_DIM_PER_CTA + 16 + gid]
        gref_prev3 = sG_tile[ref_prev * KEY_DIM_PER_CTA + 24 + gid]
        gref_future0 = sG_tile[ref_future * KEY_DIM_PER_CTA + gid]
        gref_future1 = sG_tile[ref_future * KEY_DIM_PER_CTA + 8 + gid]
        gref_future2 = sG_tile[ref_future * KEY_DIM_PER_CTA + 16 + gid]
        gref_future3 = sG_tile[ref_future * KEY_DIM_PER_CTA + 24 + gid]
        q00 = z
        q01 = z
        q02 = z
        q03 = z
        q10 = z
        q11 = z
        q12 = z
        q13 = z
        q20 = z
        q21 = z
        q22 = z
        q23 = z
        q30 = z
        q31 = z
        q32 = z
        q33 = z
        k00 = z
        k01 = z
        k02 = z
        k03 = z
        k10 = z
        k11 = z
        k12 = z
        k13 = z
        k20 = z
        k21 = z
        k22 = z
        k23 = z
        k30 = z
        k31 = z
        k32 = z
        k33 = z
        qd00 = z
        qd01 = z
        qd02 = z
        qd03 = z
        qd10 = z
        qd11 = z
        qd12 = z
        qd13 = z
        qd20 = z
        qd21 = z
        qd22 = z
        qd23 = z
        qd30 = z
        qd31 = z
        qd32 = z
        qd33 = z
        kd00 = z
        kd01 = z
        kd02 = z
        kd03 = z
        kd10 = z
        kd11 = z
        kd12 = z
        kd13 = z
        kd20 = z
        kd21 = z
        kd22 = z
        kd23 = z
        kd30 = z
        kd31 = z
        kd32 = z
        kd33 = z
        t00 = z
        t01 = z
        t02 = z
        t03 = z
        t10 = z
        t11 = z
        t12 = z
        t13 = z
        t20 = z
        t21 = z
        t22 = z
        t23 = z
        t30 = z
        t31 = z
        t32 = z
        t33 = z
        d00 = z
        d01 = z
        d02 = z
        d03 = z
        d10 = z
        d11 = z
        d12 = z
        d13 = z
        d20 = z
        d21 = z
        d22 = z
        d23 = z
        d30 = z
        d31 = z
        d32 = z
        d33 = z

        for j_block in cutlass.range(subchunk_idx):
            key_base = j_block * BC
            for k_outer in cutlass.range_constexpr(2):
                key0 = key_base + k_outer * 8 + 2 * tid
                key1 = key0 + 1
                aq0, aq2 = _hmma_load_da_pair(
                    mdAqk,
                    row_start,
                    head_idx,
                    row0,
                    key0,
                    valid,
                    daqk_row_stride,
                    daqk_head_stride,
                )
                ak0, ak2 = _hmma_load_da_pair(
                    mdAkk,
                    row_start,
                    head_idx,
                    row0,
                    key0,
                    valid,
                    dakk_row_stride,
                    dakk_head_stride,
                )
                aq1, aq3 = _hmma_load_da_pair(
                    mdAqk,
                    row_start,
                    head_idx,
                    row1,
                    key0,
                    valid,
                    daqk_row_stride,
                    daqk_head_stride,
                )
                ak1, ak3 = _hmma_load_da_pair(
                    mdAkk,
                    row_start,
                    head_idx,
                    row1,
                    key0,
                    valid,
                    dakk_row_stride,
                    dakk_head_stride,
                )
                kb00, kb01, kb10, kb11, kb20, kb21, kb30, kb31 = _hmma_load_b_groups_smem_ldmatrix(
                    sK_tile, key_base, k_outer, tidx
                )

                for group in cutlass.range_constexpr(4):
                    col_a = group * 8
                    col_load = col_a + gid
                    gref = gref_prev0
                    if group == 1:
                        gref = gref_prev1
                    if group == 2:
                        gref = gref_prev2
                    if group == 3:
                        gref = gref_prev3
                    gk0 = sG_tile[key0 * KEY_DIM_PER_CTA + col_load]
                    gk1 = sG_tile[key1 * KEY_DIM_PER_CTA + col_load]
                    kb0 = kb00
                    kb1 = kb01
                    if group == 1:
                        kb0 = kb10
                        kb1 = kb11
                    if group == 2:
                        kb0 = kb20
                        kb1 = kb21
                    if group == 3:
                        kb0 = kb30
                        kb1 = kb31
                    b0 = kb0 * cute.math.exp2(gref - gk0, fastmath=fastmath)
                    b1 = kb1 * cute.math.exp2(gref - gk1, fastmath=fastmath)
                    b0 = b0 if key0 < valid else z
                    b1 = b1 if key1 < valid else z
                    _hmma_stage_b_group(sEpi_tile, b0, b1, tidx, use_stmatrix)
                    cute.arch.sync_warp()
                    b0, b1 = _hmma_load_b_group_staged(sEpi_tile, tidx)
                    if group == 0:
                        q00, q01, q02, q03 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, q00, q01, q02, q03
                        )
                        k00, k01, k02, k03 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, b0, b1, k00, k01, k02, k03
                        )
                    if group == 1:
                        q10, q11, q12, q13 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, q10, q11, q12, q13
                        )
                        k10, k11, k12, k13 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, b0, b1, k10, k11, k12, k13
                        )
                    if group == 2:
                        q20, q21, q22, q23 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, q20, q21, q22, q23
                        )
                        k20, k21, k22, k23 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, b0, b1, k20, k21, k22, k23
                        )
                    if group == 3:
                        q30, q31, q32, q33 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, q30, q31, q32, q33
                        )
                        k30, k31, k32, k33 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, b0, b1, k30, k31, k32, k33
                        )

        # Keep i == j out of the gate-factored MMAs. Its gate-independent
        # dq/dk/dbeta contributions are restored directly in the epilogue.
        key_base = row_base
        for k_outer in cutlass.range_constexpr(2):
            key0 = key_base + k_outer * 8 + 2 * tid
            key1 = key0 + 1
            keep00 = (k_outer * 8 + 2 * tid) < gid
            keep01 = (k_outer * 8 + 2 * tid + 1) < gid
            keep10 = (k_outer * 8 + 2 * tid) < (gid + 8)
            keep11 = (k_outer * 8 + 2 * tid + 1) < (gid + 8)
            aq0, aq2 = _hmma_load_da_pair(
                mdAqk,
                row_start,
                head_idx,
                row0,
                key0,
                valid,
                daqk_row_stride,
                daqk_head_stride,
                keep00,
                keep01,
            )
            ak0, ak2 = _hmma_load_da_pair(
                mdAkk,
                row_start,
                head_idx,
                row0,
                key0,
                valid,
                dakk_row_stride,
                dakk_head_stride,
                keep00,
                keep01,
            )
            aq1, aq3 = _hmma_load_da_pair(
                mdAqk,
                row_start,
                head_idx,
                row1,
                key0,
                valid,
                daqk_row_stride,
                daqk_head_stride,
                keep10,
                keep11,
            )
            ak1, ak3 = _hmma_load_da_pair(
                mdAkk,
                row_start,
                head_idx,
                row1,
                key0,
                valid,
                dakk_row_stride,
                dakk_head_stride,
                keep10,
                keep11,
            )
            kb00, kb01, kb10, kb11, kb20, kb21, kb30, kb31 = _hmma_load_b_groups_smem_ldmatrix(
                sK_tile, key_base, k_outer, tidx
            )

            for group in cutlass.range_constexpr(4):
                col_a = group * 8
                col_load = col_a + gid
                gref = gref_qk0
                if group == 1:
                    gref = gref_qk1
                if group == 2:
                    gref = gref_qk2
                if group == 3:
                    gref = gref_qk3
                gk0 = sG_tile[key0 * KEY_DIM_PER_CTA + col_load]
                gk1 = sG_tile[key1 * KEY_DIM_PER_CTA + col_load]
                kb0 = kb00
                kb1 = kb01
                if group == 1:
                    kb0 = kb10
                    kb1 = kb11
                if group == 2:
                    kb0 = kb20
                    kb1 = kb21
                if group == 3:
                    kb0 = kb30
                    kb1 = kb31
                b0 = kb0 * cute.math.exp2(gref - gk0, fastmath=fastmath)
                b1 = kb1 * cute.math.exp2(gref - gk1, fastmath=fastmath)
                b0 = b0 if key0 < valid else z
                b1 = b1 if key1 < valid else z
                _hmma_stage_b_group(sEpi_tile, b0, b1, tidx, use_stmatrix)
                cute.arch.sync_warp()
                b0, b1 = _hmma_load_b_group_staged(sEpi_tile, tidx)
                if group == 0:
                    qd00, qd01, qd02, qd03 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, qd00, qd01, qd02, qd03
                    )
                    kd00, kd01, kd02, kd03 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, b0, b1, kd00, kd01, kd02, kd03
                    )
                if group == 1:
                    qd10, qd11, qd12, qd13 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, qd10, qd11, qd12, qd13
                    )
                    kd10, kd11, kd12, kd13 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, b0, b1, kd10, kd11, kd12, kd13
                    )
                if group == 2:
                    qd20, qd21, qd22, qd23 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, qd20, qd21, qd22, qd23
                    )
                    kd20, kd21, kd22, kd23 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, b0, b1, kd20, kd21, kd22, kd23
                    )
                if group == 3:
                    qd30, qd31, qd32, qd33 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, qd30, qd31, qd32, qd33
                    )
                    kd30, kd31, kd32, kd33 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, b0, b1, kd30, kd31, kd32, kd33
                    )

        for j_block in cutlass.range(subchunk_idx + 1, SUBCHUNKS):
            q_base = j_block * BC
            for k_outer in cutlass.range_constexpr(2):
                query0 = q_base + k_outer * 8 + 2 * tid
                query1 = query0 + 1
                aq0 = _hmma_load_da(mdAqk, row_start, head_idx, query0, row0, valid)
                aq1 = _hmma_load_da(mdAqk, row_start, head_idx, query0, row1, valid)
                aq2 = _hmma_load_da(mdAqk, row_start, head_idx, query1, row0, valid)
                aq3 = _hmma_load_da(mdAqk, row_start, head_idx, query1, row1, valid)
                ak0 = _hmma_load_da(mdAkk, row_start, head_idx, query0, row0, valid)
                ak1 = _hmma_load_da(mdAkk, row_start, head_idx, query0, row1, valid)
                ak2 = _hmma_load_da(mdAkk, row_start, head_idx, query1, row0, valid)
                ak3 = _hmma_load_da(mdAkk, row_start, head_idx, query1, row1, valid)
                bq0 = sBeta_tile[query0]
                bq1 = sBeta_tile[query1]
                qq00, qq01, qq10, qq11, qq20, qq21, qq30, qq31 = _hmma_load_b_groups_smem_ldmatrix(
                    sQ_tile, q_base, k_outer, tidx
                )
                kk00, kk01, kk10, kk11, kk20, kk21, kk30, kk31 = _hmma_load_b_groups_smem_ldmatrix(
                    sK_tile, q_base, k_outer, tidx
                )

                for group in cutlass.range_constexpr(4):
                    col_a = group * 8
                    col_load = col_a + gid
                    gref = gref_future0
                    if group == 1:
                        gref = gref_future1
                    if group == 2:
                        gref = gref_future2
                    if group == 3:
                        gref = gref_future3
                    gq0 = sG_tile[query0 * KEY_DIM_PER_CTA + col_load]
                    gq1 = sG_tile[query1 * KEY_DIM_PER_CTA + col_load]
                    qq0 = qq00
                    qq1 = qq01
                    kk0 = kk00
                    kk1 = kk01
                    if group == 1:
                        qq0 = qq10
                        qq1 = qq11
                        kk0 = kk10
                        kk1 = kk11
                    if group == 2:
                        qq0 = qq20
                        qq1 = qq21
                        kk0 = kk20
                        kk1 = kk21
                    if group == 3:
                        qq0 = qq30
                        qq1 = qq31
                        kk0 = kk30
                        kk1 = kk31
                    e0 = cute.math.exp2(gq0 - gref, fastmath=fastmath)
                    e1 = cute.math.exp2(gq1 - gref, fastmath=fastmath)
                    b0 = qq0 * e0
                    b1 = qq1 * e1
                    c0 = kk0 * bq0 * e0
                    c1 = kk1 * bq1 * e1
                    b0 = b0 if query0 < valid else z
                    b1 = b1 if query1 < valid else z
                    c0 = c0 if query0 < valid else z
                    c1 = c1 if query1 < valid else z
                    if group == 0:
                        t00, t01, t02, t03 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, t00, t01, t02, t03
                        )
                        t00, t01, t02, t03 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, c0, c1, t00, t01, t02, t03
                        )
                    if group == 1:
                        t10, t11, t12, t13 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, t10, t11, t12, t13
                        )
                        t10, t11, t12, t13 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, c0, c1, t10, t11, t12, t13
                        )
                    if group == 2:
                        t20, t21, t22, t23 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, t20, t21, t22, t23
                        )
                        t20, t21, t22, t23 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, c0, c1, t20, t21, t22, t23
                        )
                    if group == 3:
                        t30, t31, t32, t33 = _mma_tf32_m16n8k8(
                            aq0, aq1, aq2, aq3, b0, b1, t30, t31, t32, t33
                        )
                        t30, t31, t32, t33 = _mma_tf32_m16n8k8(
                            ak0, ak1, ak2, ak3, c0, c1, t30, t31, t32, t33
                        )

        for k_outer in cutlass.range_constexpr(2):
            query0 = row_base + k_outer * 8 + 2 * tid
            query1 = query0 + 1
            keep00 = gid < (k_outer * 8 + 2 * tid)
            keep01 = gid < (k_outer * 8 + 2 * tid + 1)
            keep10 = (gid + 8) < (k_outer * 8 + 2 * tid)
            keep11 = (gid + 8) < (k_outer * 8 + 2 * tid + 1)
            aq0 = _hmma_load_da(mdAqk, row_start, head_idx, query0, row0, valid, keep00)
            aq1 = _hmma_load_da(mdAqk, row_start, head_idx, query0, row1, valid, keep10)
            aq2 = _hmma_load_da(mdAqk, row_start, head_idx, query1, row0, valid, keep01)
            aq3 = _hmma_load_da(mdAqk, row_start, head_idx, query1, row1, valid, keep11)
            ak0 = _hmma_load_da(mdAkk, row_start, head_idx, query0, row0, valid, keep00)
            ak1 = _hmma_load_da(mdAkk, row_start, head_idx, query0, row1, valid, keep10)
            ak2 = _hmma_load_da(mdAkk, row_start, head_idx, query1, row0, valid, keep01)
            ak3 = _hmma_load_da(mdAkk, row_start, head_idx, query1, row1, valid, keep11)
            bq0 = sBeta_tile[query0]
            bq1 = sBeta_tile[query1]
            qq00, qq01, qq10, qq11, qq20, qq21, qq30, qq31 = _hmma_load_b_groups_smem_ldmatrix(
                sQ_tile, row_base, k_outer, tidx
            )
            kk00, kk01, kk10, kk11, kk20, kk21, kk30, kk31 = _hmma_load_b_groups_smem_ldmatrix(
                sK_tile, row_base, k_outer, tidx
            )

            for group in cutlass.range_constexpr(4):
                col_a = group * 8
                col_load = col_a + gid
                gref = gref_qk0
                if group == 1:
                    gref = gref_qk1
                if group == 2:
                    gref = gref_qk2
                if group == 3:
                    gref = gref_qk3
                gq0 = sG_tile[query0 * KEY_DIM_PER_CTA + col_load]
                gq1 = sG_tile[query1 * KEY_DIM_PER_CTA + col_load]
                qq0 = qq00
                qq1 = qq01
                kk0 = kk00
                kk1 = kk01
                if group == 1:
                    qq0 = qq10
                    qq1 = qq11
                    kk0 = kk10
                    kk1 = kk11
                if group == 2:
                    qq0 = qq20
                    qq1 = qq21
                    kk0 = kk20
                    kk1 = kk21
                if group == 3:
                    qq0 = qq30
                    qq1 = qq31
                    kk0 = kk30
                    kk1 = kk31
                e0 = cute.math.exp2(gq0 - gref, fastmath=fastmath)
                e1 = cute.math.exp2(gq1 - gref, fastmath=fastmath)
                b0 = qq0 * e0
                b1 = qq1 * e1
                c0 = kk0 * bq0 * e0
                c1 = kk1 * bq1 * e1
                b0 = b0 if query0 < valid else z
                b1 = b1 if query1 < valid else z
                c0 = c0 if query0 < valid else z
                c1 = c1 if query1 < valid else z
                if group == 0:
                    d00, d01, d02, d03 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, d00, d01, d02, d03
                    )
                    d00, d01, d02, d03 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, c0, c1, d00, d01, d02, d03
                    )
                if group == 1:
                    d10, d11, d12, d13 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, d10, d11, d12, d13
                    )
                    d10, d11, d12, d13 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, c0, c1, d10, d11, d12, d13
                    )
                if group == 2:
                    d20, d21, d22, d23 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, d20, d21, d22, d23
                    )
                    d20, d21, d22, d23 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, c0, c1, d20, d21, d22, d23
                    )
                if group == 3:
                    d30, d31, d32, d33 = _mma_tf32_m16n8k8(
                        aq0, aq1, aq2, aq3, b0, b1, d30, d31, d32, d33
                    )
                    d30, d31, d32, d33 = _mma_tf32_m16n8k8(
                        ak0, ak1, ak2, ak3, c0, c1, d30, d31, d32, d33
                    )

        daq_diag0 = z
        daq_diag1 = z
        dak_diag0 = z
        dak_diag1 = z
        if tid == 0:
            daq_diag0 = _hmma_load_da(mdAqk, row_start, head_idx, row0, row0, valid)
            daq_diag1 = _hmma_load_da(mdAqk, row_start, head_idx, row1, row1, valid)
            dak_diag0 = _hmma_load_da(mdAkk, row_start, head_idx, row0, row0, valid)
            dak_diag1 = _hmma_load_da(mdAkk, row_start, head_idx, row1, row1, valid)
        diagonal_lane = gid * 4
        daq_diag0 = cute.arch.shuffle_sync(daq_diag0, diagonal_lane)
        daq_diag1 = cute.arch.shuffle_sync(daq_diag1, diagonal_lane)
        dak_diag0 = cute.arch.shuffle_sync(dak_diag0, diagonal_lane)
        dak_diag1 = cute.arch.shuffle_sync(dak_diag1, diagonal_lane)
        db_acc0 = z
        db_acc1 = z
        dq_top0 = Int32(0)
        dq_top1 = Int32(0)
        dq_top2 = Int32(0)
        dq_top3 = Int32(0)
        dq_bottom0 = Int32(0)
        dq_bottom1 = Int32(0)
        dq_bottom2 = Int32(0)
        dq_bottom3 = Int32(0)
        dk_top0 = Int32(0)
        dk_top1 = Int32(0)
        dk_top2 = Int32(0)
        dk_top3 = Int32(0)
        dk_bottom0 = Int32(0)
        dk_bottom1 = Int32(0)
        dk_bottom2 = Int32(0)
        dk_bottom3 = Int32(0)
        dg_top00 = z
        dg_top01 = z
        dg_top10 = z
        dg_top11 = z
        dg_top20 = z
        dg_top21 = z
        dg_top30 = z
        dg_top31 = z
        dg_bottom00 = z
        dg_bottom01 = z
        dg_bottom10 = z
        dg_bottom11 = z
        dg_bottom20 = z
        dg_bottom21 = z
        dg_bottom30 = z
        dg_bottom31 = z
        for group in cutlass.range_constexpr(4):
            col_a = group * 8
            col0 = k_start + col_a + 2 * tid
            local_col0 = col_a + 2 * tid
            dq_word0 = Int32(0)
            dq_word1 = Int32(0)
            dk_word0 = Int32(0)
            dk_word1 = Int32(0)
            dg_top0 = z
            dg_top1 = z
            dg_bottom0 = z
            dg_bottom1 = z
            gref_prev_offset = ref_prev * KEY_DIM_PER_CTA + local_col0
            gref_qk_offset = ref_qk * KEY_DIM_PER_CTA + local_col0
            gref_t_offset = ref_future * KEY_DIM_PER_CTA + local_col0
            gref_prev0 = sG_tile[gref_prev_offset]
            gref_prev1 = sG_tile[gref_prev_offset + 1]
            gref_qk0 = sG_tile[gref_qk_offset]
            gref_qk1 = sG_tile[gref_qk_offset + 1]
            gref_t0 = sG_tile[gref_t_offset]
            gref_t1 = sG_tile[gref_t_offset + 1]
            qk_from_prev0 = cute.math.exp2(gref_prev0 - gref_qk0, fastmath=fastmath)
            qk_from_prev1 = cute.math.exp2(gref_prev1 - gref_qk1, fastmath=fastmath)
            tref0 = cute.math.exp2(gref_t0 - gref_qk0, fastmath=fastmath)
            tref1 = cute.math.exp2(gref_t1 - gref_qk1, fastmath=fastmath)

            for lane_row in cutlass.range_constexpr(2):
                row = row0 if lane_row == 0 else row1
                q_acc0 = q00 if lane_row == 0 else q02
                q_acc1 = q01 if lane_row == 0 else q03
                k_acc0 = k00 if lane_row == 0 else k02
                k_acc1 = k01 if lane_row == 0 else k03
                q_diag0 = qd00 if lane_row == 0 else qd02
                q_diag1 = qd01 if lane_row == 0 else qd03
                k_diag0 = kd00 if lane_row == 0 else kd02
                k_diag1 = kd01 if lane_row == 0 else kd03
                t_acc0 = t00 if lane_row == 0 else t02
                t_acc1 = t01 if lane_row == 0 else t03
                d_acc0 = d00 if lane_row == 0 else d02
                d_acc1 = d01 if lane_row == 0 else d03
                if group == 1:
                    q_acc0 = q10 if lane_row == 0 else q12
                    q_acc1 = q11 if lane_row == 0 else q13
                    k_acc0 = k10 if lane_row == 0 else k12
                    k_acc1 = k11 if lane_row == 0 else k13
                    q_diag0 = qd10 if lane_row == 0 else qd12
                    q_diag1 = qd11 if lane_row == 0 else qd13
                    k_diag0 = kd10 if lane_row == 0 else kd12
                    k_diag1 = kd11 if lane_row == 0 else kd13
                    t_acc0 = t10 if lane_row == 0 else t12
                    t_acc1 = t11 if lane_row == 0 else t13
                    d_acc0 = d10 if lane_row == 0 else d12
                    d_acc1 = d11 if lane_row == 0 else d13
                if group == 2:
                    q_acc0 = q20 if lane_row == 0 else q22
                    q_acc1 = q21 if lane_row == 0 else q23
                    k_acc0 = k20 if lane_row == 0 else k22
                    k_acc1 = k21 if lane_row == 0 else k23
                    q_diag0 = qd20 if lane_row == 0 else qd22
                    q_diag1 = qd21 if lane_row == 0 else qd23
                    k_diag0 = kd20 if lane_row == 0 else kd22
                    k_diag1 = kd21 if lane_row == 0 else kd23
                    t_acc0 = t20 if lane_row == 0 else t22
                    t_acc1 = t21 if lane_row == 0 else t23
                    d_acc0 = d20 if lane_row == 0 else d22
                    d_acc1 = d21 if lane_row == 0 else d23
                if group == 3:
                    q_acc0 = q30 if lane_row == 0 else q32
                    q_acc1 = q31 if lane_row == 0 else q33
                    k_acc0 = k30 if lane_row == 0 else k32
                    k_acc1 = k31 if lane_row == 0 else k33
                    q_diag0 = qd30 if lane_row == 0 else qd32
                    q_diag1 = qd31 if lane_row == 0 else qd33
                    k_diag0 = kd30 if lane_row == 0 else kd32
                    k_diag1 = kd31 if lane_row == 0 else kd33
                    t_acc0 = t30 if lane_row == 0 else t32
                    t_acc1 = t31 if lane_row == 0 else t33
                    d_acc0 = d30 if lane_row == 0 else d32
                    d_acc1 = d31 if lane_row == 0 else d33

                row_valid = Boolean(row < valid)
                out_row = row_start + row
                row_offset = row * KEY_DIM_PER_CTA + local_col0
                qval0 = sQ_tile[row_offset].to(Float32)
                qval1 = sQ_tile[row_offset + 1].to(Float32)
                kval0 = sK_tile[row_offset].to(Float32)
                kval1 = sK_tile[row_offset + 1].to(Float32)
                grow0 = sG_tile[row_offset]
                grow1 = sG_tile[row_offset + 1]
                beta_row = sBeta_tile[row]
                qscale_prev0 = cute.math.exp2(grow0 - gref_prev0, fastmath=fastmath)
                qscale_prev1 = cute.math.exp2(grow1 - gref_prev1, fastmath=fastmath)
                qscale_diag0 = qscale_prev0 * qk_from_prev0
                qscale_diag1 = qscale_prev1 * qk_from_prev1
                dscale0 = cute.math.exp2(gref_qk0 - grow0, fastmath=fastmath)
                dscale1 = cute.math.exp2(gref_qk1 - grow1, fastmath=fastmath)
                tscale0 = dscale0 * tref0
                tscale1 = dscale1 * tref1
                dq_in0, dq_in1 = _ld_global_f32x4_lo2_pred(
                    mDq.iterator + out_row * dq_row_stride + head_idx * dq_head_stride + col0,
                    row_valid,
                )
                dk_in0, dk_in1 = _ld_global_f32x4_lo2_pred(
                    mDk.iterator + out_row * dk_row_stride + head_idx * dk_head_stride + col0,
                    row_valid,
                )
                dg_in0, dg_in1 = _ld_global_f32x2_pred(
                    mDg.iterator + out_row * dg_row_stride + head_idx * dg_head_stride + col0,
                    row_valid,
                )
                dq_strict0 = q_acc0 * qscale_prev0 + q_diag0 * qscale_diag0
                dq_strict1 = q_acc1 * qscale_prev1 + q_diag1 * qscale_diag1
                daq_diag = daq_diag0 if lane_row == 0 else daq_diag1
                dak_diag = dak_diag0 if lane_row == 0 else dak_diag1
                dq_add0 = dq_strict0 + daq_diag * kval0
                dq_add1 = dq_strict1 + daq_diag * kval1
                if cutlass.const_expr(use_packed_f32x2):
                    dq_out0, dq_out1 = cute.arch.add_packed_f32x2(
                        (dq_in0, dq_in1), (dq_add0, dq_add1)
                    )
                else:
                    dq_out0, dq_out1 = dq_in0 + dq_add0, dq_in1 + dq_add1
                if lane_row == 0:
                    dq_word0 = _cvt_half2_f32(dq_out0, dq_out1, mDq2.element_type)
                else:
                    dq_word1 = _cvt_half2_f32(dq_out0, dq_out1, mDq2.element_type)
                dk_qk0 = k_acc0 * qscale_prev0 + k_diag0 * qscale_diag0
                dk_qk1 = k_acc1 * qscale_prev1 + k_diag1 * qscale_diag1
                dkt0 = d_acc0 * dscale0 + t_acc0 * tscale0
                dkt1 = d_acc1 * dscale1 + t_acc1 * tscale1
                dk_beta0 = dk_qk0 * beta_row
                dk_beta1 = dk_qk1 * beta_row
                dk_diag_scale = Float32(2.0) * dak_diag * beta_row
                dk_add0 = dk_beta0 + dkt0 + daq_diag * qval0 + dk_diag_scale * kval0
                dk_add1 = dk_beta1 + dkt1 + daq_diag * qval1 + dk_diag_scale * kval1
                if cutlass.const_expr(use_packed_f32x2):
                    dk_out0, dk_out1 = cute.arch.add_packed_f32x2(
                        (dk_in0, dk_in1), (dk_add0, dk_add1)
                    )
                else:
                    dk_out0, dk_out1 = dk_in0 + dk_add0, dk_in1 + dk_add1
                if lane_row == 0:
                    dk_word0 = _cvt_half2_f32(dk_out0, dk_out1, mDk2.element_type)
                else:
                    dk_word1 = _cvt_half2_f32(dk_out0, dk_out1, mDk2.element_type)
                dg_out0 = (dg_in0 + dq_strict0 * qval0 + (dk_beta0 - dkt0) * kval0) * Float32(LN2)
                dg_out1 = (dg_in1 + dq_strict1 * qval1 + (dk_beta1 - dkt1) * kval1) * Float32(LN2)
                if lane_row == 0:
                    dg_top0 = dg_out0
                    dg_top1 = dg_out1
                else:
                    dg_bottom0 = dg_out0
                    dg_bottom1 = dg_out1
                db_pair = (
                    dk_qk0 * kval0 + dk_qk1 * kval1 + dak_diag * (kval0 * kval0 + kval1 * kval1)
                )
                if lane_row == 0:
                    db_acc0 = db_acc0 + db_pair
                else:
                    db_acc1 = db_acc1 + db_pair

            if group == 0:
                dq_top0 = dq_word0
                dq_bottom0 = dq_word1
                dk_top0 = dk_word0
                dk_bottom0 = dk_word1
                dg_top00 = dg_top0
                dg_top01 = dg_top1
                dg_bottom00 = dg_bottom0
                dg_bottom01 = dg_bottom1
            if group == 1:
                dq_top1 = dq_word0
                dq_bottom1 = dq_word1
                dk_top1 = dk_word0
                dk_bottom1 = dk_word1
                dg_top10 = dg_top0
                dg_top11 = dg_top1
                dg_bottom10 = dg_bottom0
                dg_bottom11 = dg_bottom1
            if group == 2:
                dq_top2 = dq_word0
                dq_bottom2 = dq_word1
                dk_top2 = dk_word0
                dk_bottom2 = dk_word1
                dg_top20 = dg_top0
                dg_top21 = dg_top1
                dg_bottom20 = dg_bottom0
                dg_bottom21 = dg_bottom1
            if group == 3:
                dq_top3 = dq_word0
                dq_bottom3 = dq_word1
                dk_top3 = dk_word0
                dk_bottom3 = dk_word1
                dg_top30 = dg_top0
                dg_top31 = dg_top1
                dg_bottom30 = dg_bottom0
                dg_bottom31 = dg_bottom1

        _st_shared_dg_triton_swizzle(
            sEpi_tile,
            0,
            dg_top00,
            dg_top01,
            dg_bottom00,
            dg_bottom01,
            tidx,
        )
        _st_shared_dg_triton_swizzle(
            sEpi_tile,
            1,
            dg_top10,
            dg_top11,
            dg_bottom10,
            dg_bottom11,
            tidx,
        )
        _st_shared_dg_triton_swizzle(
            sEpi_tile,
            2,
            dg_top20,
            dg_top21,
            dg_bottom20,
            dg_bottom21,
            tidx,
        )
        _st_shared_dg_triton_swizzle(
            sEpi_tile,
            3,
            dg_top30,
            dg_top31,
            dg_bottom30,
            dg_bottom31,
            tidx,
        )

        _st_global_f32_triton_shared_epilogue_16x32(
            sEpi_tile,
            mDg2.iterator
            + (row_start + row_base) * dg2_row_stride
            + head_idx * dg2_head_stride
            + k_start,
            tidx,
            row_base=row_base,
            valid=valid,
            row_stride=dg2_row_stride,
        )
        _st_global_16b_ldmatrix_epilogue_16x32(
            sEpi_tile,
            mDq2.iterator
            + (row_start + row_base) * dq2_row_stride
            + head_idx * dq2_head_stride
            + k_start,
            dq_top0,
            dq_top1,
            dq_top2,
            dq_top3,
            dq_bottom0,
            dq_bottom1,
            dq_bottom2,
            dq_bottom3,
            tidx,
            row_base=row_base,
            valid=valid,
            row_stride=dq2_row_stride,
        )
        _st_global_16b_ldmatrix_epilogue_16x32(
            sEpi_tile,
            mDk2.iterator
            + (row_start + row_base) * dk2_row_stride
            + head_idx * dk2_head_stride
            + k_start,
            dk_top0,
            dk_top1,
            dk_top2,
            dk_top3,
            dk_bottom0,
            dk_bottom1,
            dk_bottom2,
            dk_bottom3,
            tidx,
            row_base=row_base,
            valid=valid,
            row_stride=dk2_row_stride,
        )

        db_acc0 = db_acc0 + cute.arch.shuffle_sync_bfly(db_acc0, 1)
        db_acc0 = db_acc0 + cute.arch.shuffle_sync_bfly(db_acc0, 2)
        db_acc1 = db_acc1 + cute.arch.shuffle_sync_bfly(db_acc1, 1)
        db_acc1 = db_acc1 + cute.arch.shuffle_sync_bfly(db_acc1, 2)
        _st_global_f32_b32_pred(
            mDb2.iterator
            + k_phase * db2_k_stride
            + (row_start + row0) * db2_row_stride
            + head_idx * db2_head_stride,
            db_acc0,
            Boolean((tid == 0) & (row0 < valid)),
        )
        _st_global_f32_b32_pred(
            mDb2.iterator
            + k_phase * db2_k_stride
            + (row_start + row1) * db2_row_stride
            + head_idx * db2_head_stride,
            db_acc1,
            Boolean((tid == 0) & (row1 < valid)),
        )

        # Reused SMEM (sQ/sK/sG/sBeta/sEpi) is overwritten by the next
        # persistent iteration's staging; fence reads of this iteration first.
        cute.arch.barrier()


class LegacyChunkKdaBwdIntraHmmaGrid:
    """Launch the persistent dense or ragged-scheduler kernel."""

    def __init__(
        self,
        ragged: bool,
        use_int64_offsets: bool,
        use_packed_f32x2: bool,
        use_stmatrix: bool,
        fastmath: bool = True,
    ):
        self.ragged = ragged
        self.use_int64_offsets = use_int64_offsets
        self.use_packed_f32x2 = use_packed_f32x2
        self.use_stmatrix = use_stmatrix
        self.fastmath = fastmath

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mG: cute.Tensor,
        mBeta: cute.Tensor,
        mdAqk: cute.Tensor,
        mdAkk: cute.Tensor,
        mDq: cute.Tensor,
        mDk: cute.Tensor,
        mDb2: cute.Tensor,
        mDg: cute.Tensor,
        mDq2: cute.Tensor,
        mDk2: cute.Tensor,
        mDg2: cute.Tensor,
        mCuSeqlens: cute.Tensor | None,
        mChunkOffsets: cute.Tensor | None,
        grid_chunks: Int32,
        stream: cuda.CUstream = None,
    ):
        legacy_chunk_kda_bwd_intra_hmma_grid_kernel.set_name_prefix(
            f"legacy_cutlass_dsl_chunk_kda_bwd_intra_fm{int(self.fastmath)}"
        )
        legacy_chunk_kda_bwd_intra_hmma_grid_kernel(
            mQ,
            mK,
            mG,
            mBeta,
            mdAqk,
            mdAkk,
            mDq,
            mDk,
            mDb2,
            mDg,
            mDq2,
            mDk2,
            mDg2,
            mCuSeqlens,
            mChunkOffsets,
            grid_chunks,
            self.ragged,
            self.use_int64_offsets,
            self.use_packed_f32x2,
            self.use_stmatrix,
            self.fastmath,
        ).launch(
            grid=(KC_TOTAL, grid_chunks, cute.size(mQ.shape[2])),
            block=(32, 1, 1),
            stream=stream,
        )


@jit_cache
def legacy_compile_chunk_kda_bwd_intra(
    heads: int,
    ragged: bool,
    io_type: type[cutlass.Numeric],
    use_int64_offsets: bool = False,
    fastmath: bool = True,
):
    """Compile one persistent intra-chunk backward specialization."""
    target = get_compile_target()
    capability = target.effective_capability
    if target.device_type != "cuda" or capability is None or capability < (8, 0):
        raise ValueError(f"KDA backward intra requires CUDA capability >= 8.0; got {target}")
    op = LegacyChunkKdaBwdIntraHmmaGrid(
        ragged=ragged,
        use_int64_offsets=use_int64_offsets,
        use_packed_f32x2=is_sm100_kda_capability(capability),
        use_stmatrix=capability >= (9, 0),
        fastmath=fastmath,
    )
    tokens, sequences = cute.sym_int(), cute.sym_int()
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int

    def normal(dtype, shape):
        return make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=128,
        )

    def column_token_head(dtype, columns: int):
        return make_fake_compact_tensor(
            dtype,
            (columns, tokens, heads),
            stride_order=(0, 2, 1),
            assumed_align=128,
        )

    def strided_column_token_head(dtype, columns: int):
        """Transposed view of a [1, T, H, columns] input with runtime token/head strides."""
        return make_fake_tensor(
            dtype,
            (columns, tokens, heads),
            stride=(
                1,
                sym_int(divisibility=_MIN_ALIGN_ELEMENTS_16BIT),
                sym_int(divisibility=_MIN_ALIGN_ELEMENTS_16BIT),
            ),
            assumed_align=_MIN_ALIGN_BYTES,
        )

    q = strided_column_token_head(io_type, KEY_DIM)
    k = strided_column_token_head(io_type, KEY_DIM)
    g = column_token_head(cutlass.Float32, KEY_DIM)
    beta = normal(cutlass.Float32, (1, tokens, heads))
    dAqk = column_token_head(cutlass.Float32, BT)
    dAkk = column_token_head(cutlass.Float32, BT)
    dq = column_token_head(cutlass.Float32, KEY_DIM)
    dk = column_token_head(cutlass.Float32, KEY_DIM)
    db_partial = normal(cutlass.Float32, (K_PHASES, tokens, heads))
    dg = column_token_head(cutlass.Float32, KEY_DIM)
    dq2 = column_token_head(io_type, KEY_DIM)
    dk2 = column_token_head(io_type, KEY_DIM)
    dg2 = column_token_head(cutlass.Float32, KEY_DIM)
    cu_seqlens = normal(cutlass.Int32, (sequences,)) if ragged else None
    chunk_offsets = normal(cutlass.Int32, (sequences,)) if ragged else None
    return compile_tvm_ffi(
        op,
        q,
        k,
        g,
        beta,
        dAqk,
        dAkk,
        dq,
        dk,
        db_partial,
        dg,
        dq2,
        dk2,
        dg2,
        cu_seqlens,
        chunk_offsets,
        Int32(1),
        name=(
            f"legacy_kda_bwd_intra_h{heads}_{cute_dtype(io_type).name}_rg{int(ragged)}"
            f"_i64{int(use_int64_offsets)}_fm{int(fastmath)}"
        ),
    )
