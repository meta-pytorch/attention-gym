# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# KDA forward intra-chunk diagonal sub-chunk kernel (safe-gate persistent grid).
#
# Computes the diagonal 16x16 Aqk/Akk sub-blocks of each 64-token chunk and
# inverts each diagonal Akk block by forward substitution. The off-diagonal
# blocks and the full 64x64 block-triangular inverse are handled by the CuTe
# K3b/K4b kernels. Only ``chunk_kda_fwd_kernel_intra_sub_chunk_forloop`` is
# consumed by the CuTe forward path
# (attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra).

import triton
import triton.language as tl

from attn_gym.linear.kda.utils import (
    autotune_cache_kwargs,
    exp2,
    gather,
)


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_NUM_CHUNKS": lambda args: args["num_chunks"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=3),
    ],
    key=["BT", "BC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "num_chunks"])
def chunk_kda_fwd_kernel_intra_sub_chunk_forloop(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    num_chunks,
    T,
    H: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_NUM_CHUNKS: tl.constexpr,
    USE_GATHER: tl.constexpr,
    CAUSAL_NORMREF: tl.constexpr = True,
    GRID_NT: tl.constexpr = 0,
    MAX_NT: tl.constexpr = 0,
):
    i_t_start, i_i, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    for _iter in range((MAX_NT + GRID_NT - 1) // GRID_NT):
        i_t_orig = i_t_start + _iter * GRID_NT
        _run = i_t_orig < MAX_NT
        if IS_VARLEN and HAS_NUM_CHUNKS and _run:
            _run = i_t_orig < tl.load(num_chunks)
        if _run:
            if IS_VARLEN:
                i_n, i_t = (
                    tl.load(chunk_indices + i_t_orig * 2).to(tl.int32),
                    tl.load(chunk_indices + i_t_orig * 2 + 1).to(tl.int32),
                )
                bos, eos = (
                    tl.load(cu_seqlens + i_n).to(tl.int32),
                    tl.load(cu_seqlens + i_n + 1).to(tl.int32),
                )
                T_local = eos - bos
            else:
                i_t = i_t_orig
                bos, eos = i_b * T, i_b * T + T
                T_local = T

            i_ti = i_t * BT + i_i * BC
            if i_ti < T_local:
                o_c = i_ti + tl.arange(0, BC)
                m_c = o_c < T_local

                q_off = q + (bos * H + i_h) * K
                k_off = k + (bos * H + i_h) * K
                g_off = g + (bos * H + i_h) * K
                beta_off = beta + bos * H + i_h
                Aqk_off = Aqk + (bos * H + i_h) * BT
                Akk_off = Akk + (bos * H + i_h) * BC

                p_q = tl.make_block_ptr(
                    q_off, (T_local, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0)
                )
                p_k = tl.make_block_ptr(
                    k_off, (T_local, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0)
                )
                p_g = tl.make_block_ptr(
                    g_off, (T_local, K), (H * K, 1), (i_ti, 0), (BC, BK), (1, 0)
                )

                p_beta = tl.make_block_ptr(beta_off, (T_local,), (H,), (i_ti,), (BC,), (0,))

                b_q = tl.load(p_q, boundary_check=(0, 1))
                b_k = tl.load(p_k, boundary_check=(0, 1))
                b_g = tl.load(p_g, boundary_check=(0, 1))
                b_beta = tl.load(p_beta, boundary_check=(0,))

                if CAUSAL_NORMREF:
                    normref_idx = 0
                else:
                    normref_idx = min(BC // 2, T_local - i_ti - 1)
                if USE_GATHER:
                    b_gn = gather(b_g, tl.full([1, BK], normref_idx, dtype=tl.int16), axis=0)
                else:
                    p_gn = g_off + (i_ti + normref_idx) * H * K + tl.arange(0, BK)
                    b_gn = tl.load(p_gn, mask=tl.arange(0, BK) < K, other=0.0)
                    b_gn = b_gn[None, :]

                b_gm = (b_g - b_gn).to(tl.float32)

                b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.0)
                b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.0)

                b_kgt = tl.trans(b_k * b_gk)

                b_Aqk = tl.dot(b_q * b_gq, b_kgt) * scale
                b_Akk = tl.dot(b_k * b_gq, b_kgt) * b_beta[:, None]

                o_i = tl.arange(0, BC)
                m_Aqk = o_i[:, None] >= o_i[None, :]
                m_Akk = o_i[:, None] > o_i[None, :]
                m_I = o_i[:, None] == o_i[None, :]

                b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
                b_Akk = tl.where(m_Akk, b_Akk, 0.0)

                p_Aqk = tl.make_block_ptr(
                    Aqk_off,
                    (T_local, BT),
                    (H * BT, 1),
                    (i_ti, i_i * BC),
                    (BC, BC),
                    (1, 0),
                )
                p_Akk = tl.make_block_ptr(
                    Akk_off, (T_local, BC), (H * BC, 1), (i_ti, 0), (BC, BC), (1, 0)
                )
                tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), boundary_check=(0, 1))
                tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), boundary_check=(0, 1))

                tl.debug_barrier()

                # forward substitution
                b_Ai = -b_Akk
                for i in range(2, min(BC, T_local - i_ti)):
                    b_a = -tl.load(Akk_off + (i_ti + i) * H * BC + o_i)
                    b_a = tl.where(o_i < i, b_a, 0.0)
                    b_a += tl.sum(b_a[:, None] * b_Ai, 0)
                    b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
                b_Ai += m_I
                tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), boundary_check=(0, 1))


__all__ = ["chunk_kda_fwd_kernel_intra_sub_chunk_forloop"]
