# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This kernel is derived from cuDNN, NVIDIA Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe.

"""
Chunked Gated Delta Net (GDN) chunk-factor pass (the T pass) for SM100 / SM103 / SM107 (Cutlass primitives): the Beta-folded
chunk inverse tile that the U GEMMs of the prefill / recompute / summary / bprop kernels consume, one BT x BT tile per
(chunk, head), in bf16 / fp16 arithmetic.

Algorithm overview (per chunk c of sequence b, tokens [cC, (c+1)C); heads independent):
  Inputs : K[BT,DK], Gate[BT] (scalar gate), Beta[BT] (scalar LR)
  Output : tinv[row, head, BT, BT] (io dtype), row = cu[b] * expand_num // BT + b + c; the rows between one sequence's
           last chunk and the next base are padding (rows = total_tokens * expand_num // BT + num_seqs, tinv_rows)

  Preprocessing (gate warp, registers):
    cumsumlog[t]     = sum_{l=0}^{t} log2(Gate_l)             cumulative log2 of gates (safe-gate / log / linear)
    Beta[t]          = sigmoid(beta_t) [* 2]                   when the kernel applies the sigmoid

  KK GEMM        : W_kk[2BT,2BT] = [K0;K1] @ [K0;K1]^T   (one M = N = 2 BT GEMM per pair; member m is diagonal block m)
  KK epilogue    : M_kk[i,j] = W_kk[i,j] * exp2(cumsumlog[i] - cumsumlog[j]) * Beta[i]   (i >= j, else 0)
  Inverse        : T_inv = (I + strict_lower(M_kk))^-1   blockwise 8 -> 16 -> 32 -> 64, both members in parallel
  Beta scaling   : tile[i,j] = T_inv[i,j] * Beta[j]      in place, then the TMA store

Tiles run in PAIRS (one KK GEMM and one accumulator stage per pair); CTA c owns the contiguous items
[c * n_items // num_ctas, (c + 1) * n_items // num_ctas) of the row table's (chunk row, head) tiles, one row's heads then
the next row's; the per-batch K descriptors zero-fill the rows past the sequence end.

SMEM layout (stage counts live in gdn_tinv_config.py; sizes at DK = 128):
  Buffer                       Size (B)  Stages
  K (two-box stage)               32768       4    <-- both members' BT x DK boxes
  tile ring                        8192   2 x 4    <-- one 4-stage ring per compute group
  cumsumlog / Beta                  512   2 x 4    <-- both members, one 4-stage ring per compute group

TMEM layout (512 columns):
  Buffer                  Cols
  KK pair acc         4 x 128     <-- 2 BT fp32 columns per pair in flight

Row table: the chain prologue (chain path) or this kernel's one-warp prologue builds one (chunk, batch, batch_start,
batch_end) entry per valid chunk row from cu_seqlens and emits the per-batch K and tinv descriptor arrays.

Warp assignments (12 warps = 384 threads):
  warps 0-3     : compute group 0 - pairs 0, 2, ..: KK epilogue x2, pair inverse, Beta column scaling x2
  warps 4-7     : compute group 1 - pairs 1, 3, ..: the same
  warp  8       : TMA load warp  - K tiles of both members into the K ring
  warp  9       : MMA warp       - one [K0;K1] @ [K0;K1]^T GEMM per pair; TMEM lifecycle
  warp  10      : epilogue warp  - tile TMA stores through the tinv map, in pair order
  warp  11      : gate warp      - Gate cumsum + Beta of both members into the group's stage
"""

import functools
from dataclasses import dataclass, replace
from typing import NamedTuple, Optional, Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import cutlass.experimental.cuda.tensor_map as tma
from cutlass.cute.runtime import from_dlpack

from ..common.thd import emit_seq_descs, emit_tile_seq_descs, TENSOR_MAP_QWORDS
from ..common.split_k import expanded_cu_seqlen
from ..common.host import get_dtype
from ..common.blockwise_inverse import (
    blockwise_diagonal_8x8_to_16x16,
    blockwise_diagonal_16x16_to_32x32,
    blockwise_diagonal_32x32_to_64x64,
    invert_diagonal_NxN,
)
from .._compat import probe
from .._compat import multiprocessor_count

RCP_LN2 = 1.4426950408889634  # 1/ln(2): natural-log gates -> the kernel's log2 domain
from ..tile_dsl.barrier import MBarrier, Producer, launch_dependent_grids, wait_on_dependent_grids
from ..tile_dsl.handles import MmaDesc, SmemTile, tma_slice_runtime_desc
from ..tile_dsl.mma import mma_ss
from ..tile_dsl.pointwise import f16x2_to_f32, fadd2, fmul2, fp32_to_fp16, opaque_f32_zero, sigmoid, softplus2
from ..tile_dsl.swizzle import swizzle_xor_128b
from ..tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait, tma_tensormap_acquire
from .gdn_tinv_config import CFG

USE_PDL = True
KEY_DIMS = (64, 128)


class GdnTinvBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_k_ready: MBarrier
    mb_k_done: MBarrier

    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier

    mb_acc_ready: MBarrier
    mb_acc_done: MBarrier

    mb_tile_ready: MBarrier
    mb_tile_done: MBarrier


def make_bars(cfg) -> GdnTinvBars:
    """GdnTinvBars factory."""
    ONE_LANE = 1
    MMA_ARRIVERS = len([cfg.tcgen05_mma_warp_id])
    GATE_WARP = cfg.threads_per_warp * len([cfg.load_gate_warp_id])
    GROUP_WARPS = len(cfg.compute_group_warp_ids[0])
    GROUP_THREADS = cfg.threads_per_warp * GROUP_WARPS
    GATE_RING = len(cfg.compute_group_warp_ids) * cfg.smem_gate_stages
    TILE_RING = len(cfg.compute_group_warp_ids) * cfg.smem_tile_stages

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=16)

    return GdnTinvBars(
        mb_k_ready=MBarrier(alloc(cfg.smem_k_stages), try_wait=True, stages=cfg.smem_k_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_done=MBarrier(alloc(cfg.smem_k_stages), try_wait=True, stages=cfg.smem_k_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_gate_ready=MBarrier(alloc(GATE_RING), try_wait=True, stages=GATE_RING, init_count=GATE_WARP, producer=Producer.THREAD),
        mb_gate_done=MBarrier(alloc(GATE_RING), try_wait=True, stages=GATE_RING, init_count=GROUP_THREADS, producer=Producer.THREAD),
        mb_acc_ready=MBarrier(alloc(cfg.tmem_acc_stages), try_wait=True, stages=cfg.tmem_acc_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_acc_done=MBarrier(alloc(cfg.tmem_acc_stages), try_wait=True, stages=cfg.tmem_acc_stages, init_count=GROUP_WARPS, producer=Producer.THREAD),
        mb_tile_ready=MBarrier(alloc(TILE_RING), try_wait=True, stages=TILE_RING, init_count=ONE_LANE, producer=Producer.THREAD),
        mb_tile_done=MBarrier(alloc(TILE_RING), try_wait=True, stages=TILE_RING, init_count=ONE_LANE, producer=Producer.THREAD),
    )


@cute.jit
def tmastg_warp(
    cfg,
    n_tiles,
    n_pairs,
    item_begin,
    mRows,
    desc_tinv_base,
    sTinv_tma,
    bars,
    heads_out,
):
    """Epilogue warp role (warp 10): every published tile SMEM -> GMEM (TMA store through its batch's tinv descriptor)
    in pair order; a tile stage returns to its compute group once the store has read it."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    NG = len(cfg.compute_group_warp_ids)
    TS = cfg.smem_tile_stages
    zero = cutlass.Int32(0)
    elect_one = nvvm.elect_sync()

    # ---- descriptor acquire, one per batch of the block ------------------------------
    r0 = item_begin // heads_out
    head0 = item_begin - r0 * heads_out.divisor
    if n_tiles > cutlass.Int32(0):
        batch_first = mRows[r0, 1]
        batch_last = mRows[(item_begin + n_tiles - cutlass.Int32(1)) // heads_out, 1]
        if elect_one:
            for b in cutlass.range(batch_first, batch_last + cutlass.Int32(1), unroll=1):
                tma_tensormap_acquire((desc_tinv_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic))
    # ---- (row, head) of pair p's first member: two items per pair, carried with one wrap, no division ----
    pair_rows = cutlass.Int32(2) // heads_out
    pair_heads = cutlass.Int32(2) - pair_rows * heads_out.divisor
    for p in cutlass.range(n_pairs):
        group = p % cutlass.Int32(NG)
        j = p // cutlass.Int32(NG)
        t0 = 2 * p
        have_m1 = t0 + cutlass.Int32(1) < n_tiles
        wrap1 = head0 + cutlass.Int32(1) == heads_out.divisor
        r1 = r0 + cutlass.Int32(1) if have_m1 and wrap1 else r0
        head1 = (cutlass.Int32(0) if wrap1 else head0 + cutlass.Int32(1)) if have_m1 else head0
        tile_row0 = mRows[r0, 0]
        tile_row1 = mRows[r1, 0]
        desc_tinv0 = (desc_tinv_base + mRows[r0, 1] * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        desc_tinv1 = (desc_tinv_base + mRows[r1, 1] * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        tile_seq0 = 2 * j
        tile_seq1 = tile_seq0 + cutlass.Int32(1)
        tile_slot0 = group * cutlass.Int32(TS) + tile_seq0 % cutlass.Int32(TS)
        tile_slot1 = group * cutlass.Int32(TS) + tile_seq1 % cutlass.Int32(TS)

        # ---- chunk-factor tile store -------------------------------------------------
        bars.mb_tile_ready[tile_slot0].wait((tile_seq0 // cutlass.Int32(TS)) & cutlass.Int32(1))
        if elect_one:
            tma_store_tile(sTinv_tma[tile_slot0], tma_slice_runtime_desc(desc_tinv0, zero, zero, head0, tile_row0))
        if have_m1:
            bars.mb_tile_ready[tile_slot1].wait((tile_seq1 // cutlass.Int32(TS)) & cutlass.Int32(1))
            if elect_one:
                tma_store_tile(sTinv_tma[tile_slot1], tma_slice_runtime_desc(desc_tinv1, zero, zero, head1, tile_row1))
        if elect_one:
            tma_store_commit()
            tma_store_wait(0)
            bars.mb_tile_done[tile_slot0].arrive()
            if have_m1:
                bars.mb_tile_done[tile_slot1].arrive()
        head0 = head0 + pair_heads
        r0 = r0 + pair_rows
        wrap0 = head0 >= heads_out.divisor
        head0 = head0 - heads_out.divisor if wrap0 else head0
        r0 = r0 + cutlass.Int32(1) if wrap0 else r0


@cute.jit
def gate_warp(
    cfg,
    n_tiles,
    n_pairs,
    item_begin,
    mRows,
    lane_idx,
    mGate,
    mA_log,
    mDt_bias,
    mBeta,
    sCumsumlog,
    sBeta,
    bars,
    heads_out,
):
    """Gate producer (warp 11): the gate cumsum and beta of both members of pair p into stage (p // groups) % SMEM_GATE_STAGES
    of group p % groups's ring, in pair order; every lane arrives on the stage's ready barrier after its own stores, the
    group frees the stage after its beta column scaling."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    NG = len(cfg.compute_group_warp_ids)
    GS = cfg.smem_gate_stages
    n_cols = cfg.b_t // cfg.threads_per_warp
    oob_neutral = cutlass.Float32(0.0) if cutlass.const_expr(cfg.log_gate) else cutlass.Float32(1.0)
    r0 = item_begin // heads_out
    head0 = item_begin - r0 * heads_out.divisor
    pair_rows = cutlass.Int32(2) // heads_out
    pair_heads = cutlass.Int32(2) - pair_rows * heads_out.divisor
    for p in cutlass.range(n_pairs):
        group = p % cutlass.Int32(NG)
        j = p // cutlass.Int32(NG)
        gate_slot = group * cutlass.Int32(GS) + j % cutlass.Int32(GS)
        t0 = 2 * p
        have_m1 = t0 + cutlass.Int32(1) < n_tiles
        wrap1 = head0 + cutlass.Int32(1) == heads_out.divisor
        r1 = r0 + cutlass.Int32(1) if wrap1 else r0
        head1 = cutlass.Int32(0) if wrap1 else head0 + cutlass.Int32(1)
        bars.mb_gate_done[gate_slot].wait(((j // cutlass.Int32(GS)) & cutlass.Int32(1)) ^ cutlass.Int32(1))
        for member in cutlass.range_constexpr(2):
            member_live = cutlass.Boolean(True) if cutlass.const_expr(member == 0) else have_m1
            if member_live:
                r = r0 if cutlass.const_expr(member == 0) else r1
                head_idx = head0 if cutlass.const_expr(member == 0) else head1
                chunk_offset = mRows[r, 2] + mRows[r, 0] * cutlass.Int32(cfg.b_t)
                member_end = mRows[r, 3]

                # ---- safe-gate per-head parameters: a = -exp(A_log) / ln 2, bias -----
                if cutlass.const_expr(cfg.safe_gate and mA_log is None):
                    a = opaque_f32_zero() - cutlass.Float32(RCP_LN2)
                else:
                    a = cutlass.Float32(0.0)
                bias = cutlass.Float32(0.0)
                if cutlass.const_expr(cfg.safe_gate and mA_log is not None):
                    a = -cute.math.exp2(mA_log[head_idx].to(cutlass.Float32) * cutlass.Float32(RCP_LN2), fastmath=True) * cutlass.Float32(RCP_LN2)
                if cutlass.const_expr(cfg.safe_gate and mDt_bias is not None):
                    bias = mDt_bias[head_idx].to(cutlass.Float32)

                # ---- Gate load: GMEM -> registers (OOB neutral: 1.0 -> log2 = 0.0) ----
                tok_lo = chunk_offset + lane_idx
                tok_hi = tok_lo + cutlass.Int32(cfg.threads_per_warp)
                pos_valid_lo = tok_lo < member_end
                pos_valid_hi = tok_hi < member_end
                if cutlass.const_expr(cfg.expand_num > 1):
                    row_lo = tok_lo // cutlass.Int32(cfg.expand_num)
                    row_hi = tok_hi // cutlass.Int32(cfg.expand_num)
                    gate_valid_lo = pos_valid_lo and (tok_lo - row_lo * cutlass.Int32(cfg.expand_num) == 0)
                    gate_valid_hi = pos_valid_hi and (tok_hi - row_hi * cutlass.Int32(cfg.expand_num) == 0)
                    gate_row_end = member_end // cutlass.Int32(cfg.expand_num)
                    raw_lo = mGate[cutlass.min(row_lo, gate_row_end - 1), head_idx]
                    raw_hi = mGate[cutlass.min(row_hi, gate_row_end - 1), head_idx]
                else:
                    gate_valid_lo = pos_valid_lo
                    gate_valid_hi = pos_valid_hi
                    raw_lo = mGate[cutlass.min(tok_lo, member_end - 1), head_idx]
                    raw_hi = mGate[cutlass.min(tok_hi, member_end - 1), head_idx]
                gate_valid = [gate_valid_lo, gate_valid_hi]
                gate_vals = [raw_lo.to(cutlass.Float32) if gate_valid_lo else oob_neutral, raw_hi.to(cutlass.Float32) if gate_valid_hi else oob_neutral]

                # ---- Gate transform into the log2 domain -----------------------------
                if cutlass.const_expr(cfg.safe_gate):
                    for col in cutlass.range_constexpr(0, n_cols, 2):
                        biased_lo, biased_hi = fadd2(gate_vals[col], gate_vals[col + 1], bias, bias)
                        sp_lo, sp_hi = softplus2(biased_lo, biased_hi)
                        contrib_lo, contrib_hi = fmul2(sp_lo, sp_hi, a, a)
                        gate_vals[col] = contrib_lo if gate_valid[col] else cutlass.Float32(0.0)
                        gate_vals[col + 1] = contrib_hi if gate_valid[col + 1] else cutlass.Float32(0.0)
                elif cutlass.const_expr(cfg.log_gate):
                    rcp_ln2 = opaque_f32_zero() + cutlass.Float32(RCP_LN2)
                    for col in cutlass.range_constexpr(0, n_cols, 2):
                        gate_vals[col], gate_vals[col + 1] = fmul2(gate_vals[col], gate_vals[col + 1], rcp_ln2, rcp_ln2)
                else:
                    floor = cutlass.Float32(1e-10)
                    for col in cutlass.range_constexpr(0, n_cols, 2):
                        shifted_lo, shifted_hi = fadd2(gate_vals[col], gate_vals[col + 1], floor, floor)
                        gate_vals[col] = cute.math.log2(shifted_lo, fastmath=True)
                        gate_vals[col + 1] = cute.math.log2(shifted_hi, fastmath=True)

                # ---- 64-token cumsum: warp scan, then the column carry ---------------
                for offset in [1, 2, 4, 8, 16]:
                    for col in cutlass.range_constexpr(n_cols):
                        n = nvvm.shfl_sync(0xFFFFFFFF, gate_vals[col], offset, 0, kind=nvvm.Shfl.UP)
                        if lane_idx >= offset:
                            gate_vals[col] = gate_vals[col] + n
                for col in cutlass.range_constexpr(1, n_cols):
                    last_v = nvvm.shfl_sync(
                        0xFFFFFFFF,
                        gate_vals[col - 1],
                        cfg.threads_per_warp - 1,
                        cfg.threads_per_warp - 1,
                        kind=nvvm.Shfl.IDX,
                    )
                    gate_vals[col] += last_v
                for col in cutlass.range_constexpr(n_cols):
                    pos = lane_idx + col * cfg.threads_per_warp
                    sCumsumlog[pos, 0, member, gate_slot] = gate_vals[col]

                # ---- Beta load: GMEM -> registers ------------------------------------
                raws = [mBeta[cutlass.min(tok_lo, member_end - 1), head_idx], mBeta[cutlass.min(tok_hi, member_end - 1), head_idx]]
                valids = [pos_valid_lo, pos_valid_hi]
                for col in cutlass.range_constexpr(n_cols):
                    pos = lane_idx + col * cfg.threads_per_warp
                    beta_value = raws[col].to(cutlass.Float32)
                    if cutlass.const_expr(cfg.beta_sigmoid):
                        beta_value = (sigmoid(beta_value) * (2.0 if cfg.allow_neg_eigval else 1.0)).to(mBeta.element_type).to(cutlass.Float32)
                    sBeta[pos, 0, member, gate_slot] = beta_value if valids[col] else cutlass.Float32(0.0)
        bars.mb_gate_ready[gate_slot].arrive()
        head0 = head0 + pair_heads
        r0 = r0 + pair_rows
        wrap0 = head0 >= heads_out.divisor
        head0 = head0 - heads_out.divisor if wrap0 else head0
        r0 = r0 + cutlass.Int32(1) if wrap0 else r0


@cute.jit
def tcgen05_mma_warp(
    cfg,
    n_pairs,
    tmem_col,
    sK,
    bars,
):
    """MMA issuer role (warp 9): K(S) @ K^T of both members into accumulator stage p % TMEM_ACC_STAGES."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    KS = cfg.smem_k_stages
    AS = cfg.tmem_acc_stages
    bytes_per_element = cfg.io_dtype.width // 8
    elect_one = nvvm.elect_sync()
    tmem_base = tmem_col

    # ---- chunk-invariant GEMM descriptor ---------------------------------------------
    instruction_descriptor_kk = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=2 * cfg.b_t, m_dim=2 * cfg.b_t
    )
    bmm_k_k_desc = MmaDesc(
        M=2 * cfg.b_t,
        N=2 * cfg.b_t,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_kk,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    KQ_SEG = (2 * cfg.b_t * 64 * bytes_per_element) >> 4
    KQ_SUBTILES = bmm_k_k_desc.num_subtiles
    KQ_STEPS = bmm_k_k_desc.steps_per_subtile
    ACC_STAGE_COLS = cfg.b_t
    for p in cutlass.range(n_pairs):
        k_stage = p % cutlass.Int32(KS)
        k_phase = (p // cutlass.Int32(KS)) & cutlass.Int32(1)
        acc_stage = p % cutlass.Int32(AS)
        acc_done_phase = ((p // cutlass.Int32(AS)) & cutlass.Int32(1)) ^ cutlass.Int32(1)
        bars.mb_k_ready[k_stage].wait(k_phase)

        # ---- KK pair = K pair(S) @ K pair^T ------------------------------------------
        # one M = N = 2 b_t GEMM over [K0;K1]; member m's K_m K_m^T is diagonal block m
        bars.mb_acc_done[acc_stage].wait(acc_done_phase)
        desc_kq = sK[k_stage].desc()
        acc = nvvm.make_tmem_ptr(tmem_base + acc_stage * cutlass.Int32(2 * ACC_STAGE_COLS), cutlass.Float32)
        for subtile in cutlass.range_constexpr(KQ_SUBTILES):
            subtile_offset = subtile * KQ_SEG
            mma_ss(bmm_k_k_desc, desc_kq + subtile_offset, desc_kq + subtile_offset, acc, accumulate=subtile > 0, k_count=KQ_STEPS)
        if elect_one:
            bars.mb_acc_ready[acc_stage].arrive(cta_group=1)
            bars.mb_k_done[k_stage].arrive(cta_group=1)


@cute.jit
def tmaldg_warp(
    cfg,
    n_tiles,
    n_pairs,
    item_begin,
    mRows,
    desc_k_base,
    sK_tma,
    bars,
    heads_out,
    k_ratio,
):
    """TMA-LDG warp role (warp 8): K tiles of pair p into K stage p % SMEM_K_STAGES, as far
    ahead as the ring allows."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    KS = cfg.smem_k_stages
    half_elements = cfg.b_t * 64
    zero = cutlass.Int32(0)
    elect_one = nvvm.elect_sync()

    # ---- descriptor acquire, one per batch of the block ------------------------------
    r0 = item_begin // heads_out
    head_o0 = item_begin - r0 * heads_out.divisor
    if n_tiles > cutlass.Int32(0):
        batch_first = mRows[r0, 1]
        batch_last = mRows[(item_begin + n_tiles - cutlass.Int32(1)) // heads_out, 1]
        if elect_one:
            for b in cutlass.range(batch_first, batch_last + cutlass.Int32(1), unroll=1):
                tma_tensormap_acquire((desc_k_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic))
    pair_rows = cutlass.Int32(2) // heads_out
    pair_heads = cutlass.Int32(2) - pair_rows * heads_out.divisor
    for p in cutlass.range(n_pairs):
        k_stage = p % cutlass.Int32(KS)
        done_phase = ((p // cutlass.Int32(KS)) & cutlass.Int32(1)) ^ cutlass.Int32(1)
        t0 = 2 * p
        have_m1 = t0 + cutlass.Int32(1) < n_tiles
        wrap1 = head_o0 + cutlass.Int32(1) == heads_out.divisor
        r1 = r0 + cutlass.Int32(1) if have_m1 and wrap1 else r0
        head_o1 = (cutlass.Int32(0) if wrap1 else head_o0 + cutlass.Int32(1)) if have_m1 else head_o0
        token0 = mRows[r0, 0] * cutlass.Int32(cfg.b_t)
        token1 = mRows[r1, 0] * cutlass.Int32(cfg.b_t)
        desc_k0 = (desc_k_base + mRows[r0, 1] * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        desc_k1 = (desc_k_base + mRows[r1, 1] * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(cutlass.AddressSpace.generic)
        head_k0 = head_o0 // k_ratio
        head_k1 = head_o1 // k_ratio

        # ---- K load ------------------------------------------------------------------
        mb_k_ready_stage = bars.mb_k_ready[k_stage]
        bars.mb_k_done[k_stage].wait(done_phase)
        if elect_one:
            mb_k_ready_stage.arrive(n_bytes=cutlass.Int32(cfg.tma_k_bytes) * (cutlass.Int32(2) if have_m1 else cutlass.Int32(1)))
        tma_load_tile(sK_tma[k_stage], tma_slice_runtime_desc(desc_k0, zero, head_k0, token0), mb_k_ready_stage.smem_ptr)
        if have_m1:
            tma_load_tile(
                sK_tma[k_stage].shifted(half_elements), tma_slice_runtime_desc(desc_k1, zero, head_k1, token1), mb_k_ready_stage.smem_ptr
            )
        head_o0 = head_o0 + pair_heads
        r0 = r0 + pair_rows
        wrap0 = head_o0 >= heads_out.divisor
        head_o0 = head_o0 - heads_out.divisor if wrap0 else head_o0
        r0 = r0 + cutlass.Int32(1) if wrap0 else r0


@cute.jit
def compute_warp_group(
    cfg,
    group: cutlass.Constexpr,
    n_tiles,
    n_pairs,
    tidx,
    warp_id,
    lane_idx,
    tmem_col,
    sTinv,
    sCumsumlog,
    sBeta,
    bars,
):
    """Compute warp-group role (4 warps): pairs group, group + groups, ...: each pair's KK epilogues, blockwise inverse and
    beta column scaling into the group's tile ring; ``cfg.inverse_barrier_id`` is this group's named barrier."""
    nvvm.setmaxregister(cfg.num_regs_compute, nvvm.SetMaxRegisterAction.INCREASE)
    NG = len(cfg.compute_group_warp_ids)
    AS = cfg.tmem_acc_stages
    TS = cfg.smem_tile_stages
    GS = cfg.smem_gate_stages
    tmem_row_lo_addr = cutlass.Int32(0)
    tmem_row_hi_addr = cutlass.Int32(16) << 16
    ACC_STAGE_COLS = cfg.b_t
    storer = tidx == cutlass.Int32(cfg.compute_group_warp_ids[group][0] * cfg.threads_per_warp)
    elect_one = nvvm.elect_sync()
    tmem_base = tmem_col

    mask_zero = opaque_f32_zero()

    # ---- lane geometry ---------------------------------------------------------------
    inverse_local_warp = warp_id % 2
    pair_half = warp_id // 2
    half_row_base = inverse_local_warp * 32
    store_row = warp_id * 16 + lane_idx % 16
    store_col = (lane_idx // 16) * 8
    store_row_frag = lane_idx % 16
    num_vals = 32
    FRAG_COLS = 16
    ACC_N_FRAGS = cfg.b_t // FRAG_COLS
    row_u0_lo = half_row_base + lane_idx // 4
    row_u0_hi = row_u0_lo + 8
    row_u1_lo = row_u0_lo + 16
    row_u1_hi = row_u0_lo + 24

    n_pairs_g = (n_pairs - cutlass.Int32(group) + cutlass.Int32(NG - 1)) // cutlass.Int32(NG)

    for j in cutlass.range(n_pairs_g):
        p = j * cutlass.Int32(NG) + cutlass.Int32(group)
        stage = j % cutlass.Int32(GS)
        gate_slot = cutlass.Int32(group * GS) + stage
        acc_stage = p % cutlass.Int32(AS)
        acc_phase = (p // cutlass.Int32(AS)) & cutlass.Int32(1)
        t0 = 2 * p
        have_m1 = t0 + cutlass.Int32(1) < n_tiles
        tile_seq0 = 2 * j
        tile_seq1 = tile_seq0 + cutlass.Int32(1)
        tile_slot0 = cutlass.Int32(group * TS) + tile_seq0 % cutlass.Int32(TS)
        tile_slot1 = cutlass.Int32(group * TS) + tile_seq1 % cutlass.Int32(TS)
        tinv0_base = sTinv[tile_slot0].base
        tinv1_base = sTinv[tile_slot1].base

        # ---- the pair's Gate cumsum / Beta stage, filled by the gate warp ------------
        bars.mb_gate_ready[gate_slot].wait((j // cutlass.Int32(GS)) & cutlass.Int32(1))

        # ---- Gate rows for this warp's KK member role --------------------------------
        do_kk = have_m1 or pair_half == 0
        gate0_idx = cutlass.Int32(0)
        gate1_idx = cutlass.Int32(1) if have_m1 else cutlass.Int32(0)
        kk_gate_idx = gate1_idx if pair_half == 1 else gate0_idx

        kk_row_cumsumlog = []
        for r in (row_u0_lo, row_u0_hi, row_u1_lo, row_u1_hi):
            kk_row_cumsumlog.append(sCumsumlog[r, 0, kk_gate_idx, stage])
        kk_col_cumsumlog = []
        for g in cutlass.range_constexpr(8):
            for b in cutlass.range_constexpr(2):
                chunk_col = (lane_idx % 4) * 2 + g * 8 + b
                kk_col_cumsumlog.append(sCumsumlog[chunk_col, 0, kk_gate_idx, stage])

        decay_t_kk = []
        for u in cutlass.range_constexpr(2):
            for k in cutlass.range_constexpr(num_vals):
                hi_row = ((k // 2) % 2) == 1
                chunk_row_u0 = row_u0_hi if cutlass.const_expr(hi_row) else row_u0_lo
                chunk_row_u1 = row_u1_hi if cutlass.const_expr(hi_row) else row_u1_lo
                chunk_row = chunk_row_u1 if cutlass.const_expr(u == 1) else chunk_row_u0
                chunk_col = (lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2)
                is_lower = chunk_row >= chunk_col
                row_cumsumlog = kk_row_cumsumlog[u * 2 + (1 if hi_row else 0)]
                col = (k // 4) * 2 + (k % 2)
                decay_t_kk.append(cute.math.exp2(row_cumsumlog - kk_col_cumsumlog[col], fastmath=True) if is_lower else mask_zero)

        beta0_idx = cutlass.Int32(0)
        beta1_idx = cutlass.Int32(1) if have_m1 else cutlass.Int32(0)
        kk_beta_idx = beta1_idx if pair_half == 1 else beta0_idx
        kk_beta = []
        for r in (row_u0_lo, row_u0_hi, row_u1_lo, row_u1_hi):
            kk_beta.append(sBeta[r, 0, kk_beta_idx, stage])

        # ---- the pair's tile stages are free once the epilogue warp has read them ----
        bars.mb_tile_done[tile_slot0].wait(((tile_seq0 // cutlass.Int32(TS)) & cutlass.Int32(1)) ^ cutlass.Int32(1))
        if have_m1:
            bars.mb_tile_done[tile_slot1].wait(((tile_seq1 // cutlass.Int32(TS)) & cutlass.Int32(1)) ^ cutlass.Int32(1))

        # ---- KK epilogue: M kk[i,j] = W kk[i,j] * T[i,j] * Beta[i] -------------------
        acc0_idx = cutlass.Int32(0)
        acc1_idx = cutlass.Int32(1) if have_m1 else cutlass.Int32(0)
        kk_acc_idx = acc1_idx if pair_half == 1 else acc0_idx
        kk_base = tinv1_base if pair_half == 1 else tinv0_base
        acc_col = tmem_base + acc_stage * cutlass.Int32(2 * ACC_STAGE_COLS) + kk_acc_idx * ACC_STAGE_COLS
        bars.mb_acc_ready[acc_stage].wait(acc_phase)
        if do_kk:
            kk_vec_lo = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(tmem_row_lo_addr + acc_col, cutlass.Float32), num=8)
            kk_vec_hi = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(tmem_row_hi_addr + acc_col, cutlass.Float32), num=8)
            nvvm.tcgen05_wait("load")
            if elect_one:
                bars.mb_acc_done[acc_stage].arrive()
            for u in cutlass.range_constexpr(2):
                kk_vec = kk_vec_hi if cutlass.const_expr(u == 1) else kk_vec_lo
                kk_pack = []
                for k in cutlass.range_constexpr(num_vals // 2):
                    row_beta = kk_beta[u * 2 + 1] if cutlass.const_expr((k % 2) == 1) else kk_beta[u * 2]
                    p0, p1 = fmul2(kk_vec[2 * k], kk_vec[2 * k + 1], decay_t_kk[u * num_vals + 2 * k], decay_t_kk[u * num_vals + 2 * k + 1])
                    v0, v1 = fmul2(p0, p1, row_beta, row_beta)
                    kk_pack.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
                st_row = half_row_base + u * 16 + store_row_frag
                for c in cutlass.range_constexpr(ACC_N_FRAGS):
                    nvvm.stmatrix(
                        kk_base + st_row * cfg.b_t + swizzle_xor_128b(st_row, store_col + c * FRAG_COLS),
                        [kk_pack[c * 4 + 0], kk_pack[c * 4 + 1], kk_pack[c * 4 + 2], kk_pack[c * 4 + 3]],
                        nvvm.MMALayout.ROW,
                    )
        else:
            if elect_one:
                bars.mb_acc_done[acc_stage].arrive()

        # ---- pair inverse: warps 0-1 own matrix 0, warps 2-3 matrix 1 ----------------
        inv_base = tinv0_base
        if have_m1:
            inv_base = tinv1_base if warp_id >= 2 else tinv0_base
        do_inv = have_m1 or warp_id < 2

        # ---- diagonal 8x8 inverse ----------------------------------------------------
        nvvm.barrier_cta_sync_aligned(cfg.inverse_barrier_id, thread_count=cfg.inverse_barrier_threads)
        if do_inv:
            invert_diagonal_NxN(cfg, inv_base, inv_base, (inverse_local_warp * cfg.threads_per_warp + lane_idx) // 8, tidx, 8)
        nvvm.barrier_cta_sync_aligned(cfg.inverse_barrier_id, thread_count=cfg.inverse_barrier_threads)

        # ---- 8x8 -> 16x16 ------------------------------------------------------------
        blockwise_diagonal_8x8_to_16x16(cfg, tinv0_base, tinv0_base, warp_id * 16, lane_idx)
        if have_m1:
            blockwise_diagonal_8x8_to_16x16(cfg, tinv1_base, tinv1_base, warp_id * 16, lane_idx)
        nvvm.barrier_cta_sync_aligned(cfg.inverse_barrier_id, thread_count=cfg.inverse_barrier_threads)

        # ---- 16x16 -> 32x32 ----------------------------------------------------------
        if do_inv:
            blockwise_diagonal_16x16_to_32x32(cfg, inv_base, inv_base, inverse_local_warp * 32, lane_idx)
        nvvm.barrier_cta_sync_aligned(cfg.inverse_barrier_id, thread_count=cfg.inverse_barrier_threads)

        # ---- 32x32 -> 64x64 ----------------------------------------------------------
        blockwise_diagonal_32x32_to_64x64(cfg, inv_base, inv_base, inverse_local_warp, lane_idx, cfg.inverse_barrier_id, cfg.inverse_barrier_threads)
        nvvm.barrier_cta_sync_aligned(cfg.inverse_barrier_id, thread_count=cfg.inverse_barrier_threads)

        # ---- Beta column scaling, member 0: T^-1[i,j] *= Beta[j] ---------------------
        beta_col = []
        for k in cutlass.range_constexpr(num_vals):
            beta_col.append(sBeta[(lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2), 0, beta0_idx, stage])
        tinv_frags = []
        for c in cutlass.range_constexpr(ACC_N_FRAGS):
            tinv_frags += list(nvvm.ldmatrix(tinv0_base + store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS), 4, nvvm.MMALayout.ROW))
        tinv_pack = []
        for jj in cutlass.range_constexpr(num_vals // 2):
            lo, hi = f16x2_to_f32(tinv_frags[jj], dtype=cfg.io_dtype)
            s0, s1 = fmul2(lo, hi, beta_col[2 * jj], beta_col[2 * jj + 1])
            tinv_pack.append(fp32_to_fp16(s0, s1, dtype=cfg.io_dtype))
        for c in cutlass.range_constexpr(ACC_N_FRAGS):
            nvvm.stmatrix(
                tinv0_base + store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS),
                [tinv_pack[c * 4 + 0], tinv_pack[c * 4 + 1], tinv_pack[c * 4 + 2], tinv_pack[c * 4 + 3]],
                nvvm.MMALayout.ROW,
            )

        if have_m1:
            # ---- Beta column scaling, member 1: T^-1[i,j] *= Beta[j] -----------------
            beta_col = []
            for k in cutlass.range_constexpr(num_vals):
                beta_col.append(sBeta[(lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2), 0, beta1_idx, stage])
            tinv_frags = []
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                tinv_frags += list(
                    nvvm.ldmatrix(tinv1_base + store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS), 4, nvvm.MMALayout.ROW)
                )
            tinv_pack = []
            for jj in cutlass.range_constexpr(num_vals // 2):
                lo, hi = f16x2_to_f32(tinv_frags[jj], dtype=cfg.io_dtype)
                s0, s1 = fmul2(lo, hi, beta_col[2 * jj], beta_col[2 * jj + 1])
                tinv_pack.append(fp32_to_fp16(s0, s1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    tinv1_base + store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS),
                    [tinv_pack[c * 4 + 0], tinv_pack[c * 4 + 1], tinv_pack[c * 4 + 2], tinv_pack[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )

        bars.mb_gate_done[gate_slot].arrive()

        # ---- publish the finished tiles to the epilogue warp -------------------------
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.barrier_cta_sync_aligned(cfg.inverse_barrier_id, thread_count=cfg.inverse_barrier_threads)
        if storer:
            bars.mb_tile_ready[tile_slot0].arrive()
            if have_m1:
                bars.mb_tile_ready[tile_slot1].arrive()


@cute.jit
def emit_tinv_rows(
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    cu_seqlens: cute.Tensor,
    mRows: cute.Tensor,
    mCount: cute.Tensor,
    lane_idx,
) -> None:
    """One warp: the chunk-factor pass's row table, one (chunk, batch, batch_start, batch_end) entry per valid chunk row in
    tile-row order, and the valid row count; lanes scan the chunk counts of 32 sequences at a time and write their rows."""
    n_batch = cutlass.Int32(cu_seqlens.shape[0]) - cutlass.Int32(1)
    carry = cutlass.Int32(0)
    for b0 in cutlass.range(0, n_batch, 32, unroll=1):
        b = b0 + lane_idx
        n_chunks = cutlass.Int32(0)
        batch_start = cutlass.Int32(0)
        batch_end = cutlass.Int32(0)
        if b < n_batch:
            batch_start = expanded_cu_seqlen(expand_num, cu_seqlens, b)
            batch_end = expanded_cu_seqlen(expand_num, cu_seqlens, b + cutlass.Int32(1))
            n_chunks = (batch_end - batch_start + cutlass.Int32(b_t - 1)) // cutlass.Int32(b_t)
        incl = n_chunks
        for offset in [1, 2, 4, 8, 16]:
            n = nvvm.shfl_sync(0xFFFFFFFF, incl, offset, 0, kind=nvvm.Shfl.UP)
            if lane_idx >= offset:
                incl = incl + n
        base = carry + incl - n_chunks
        for j in cutlass.range(32):
            n_j = nvvm.shfl_sync(0xFFFFFFFF, n_chunks, j, 31, kind=nvvm.Shfl.IDX)
            base_j = nvvm.shfl_sync(0xFFFFFFFF, base, j, 31, kind=nvvm.Shfl.IDX)
            start_j = nvvm.shfl_sync(0xFFFFFFFF, batch_start, j, 31, kind=nvvm.Shfl.IDX)
            end_j = nvvm.shfl_sync(0xFFFFFFFF, batch_end, j, 31, kind=nvvm.Shfl.IDX)
            for c in cutlass.range(lane_idx, n_j, 32, unroll=1):
                mRows[base_j + c, 0] = c
                mRows[base_j + c, 1] = b0 + j
                mRows[base_j + c, 2] = start_j
                mRows[base_j + c, 3] = end_j
        carry = carry + nvvm.shfl_sync(0xFFFFFFFF, incl, 31, 31, kind=nvvm.Shfl.IDX)
    if lane_idx == 0:
        mCount[0] = carry


@cute.jit
def build_descs_body(
    widx,
    base_k,
    base_tinv,
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    k: cute.Tensor,
    tinv: cute.Tensor,
    n_batch: cutlass.Int32,
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
) -> None:
    """Per-batch descriptor-array build on one warp, the K array then the tinv array."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_k_arr = cute.make_tensor(desc_workspace.iterator, cute.make_layout((arr_words,), stride=(1,)))
    desc_tinv_arr = cute.make_tensor(desc_workspace.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,)))
    if widx == 0:
        emit_seq_descs(base_k, desc_k_arr, cu_seqlens, k, n_batch, 2, expand_num, lanes=32)
        emit_tile_seq_descs(base_tinv, desc_tinv_arr, cu_seqlens, tinv, n_batch, b_t, 3, expand_num, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


@cute.kernel
def frost_gdn_tinv_prologue(
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    base_k: cutlass.GridConstant[tma.TensorMap],
    base_tinv: cutlass.GridConstant[tma.TensorMap],
    desc_words: cute.Tensor,
    cu_seqlens: cute.Tensor,
    k: cute.Tensor,
    tinv: cute.Tensor,
    mRows: cute.Tensor,
    mCount: cute.Tensor,
):
    """One warp: emit the per-batch K and tinv descriptor arrays (from the kernel-parameter maps the launch encoded) into
    the caller's workspace (a CUDA-graph replay re-emits from the captured maps), then build the row table."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
        launch_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    widx = cutlass.Int32(tidx) // cutlass.Int32(32)
    lane_idx = cutlass.Int32(tidx) % cutlass.Int32(32)
    n_batch = cutlass.Int32(cu_seqlens.shape[0]) - cutlass.Int32(1)
    build_descs_body(widx, base_k, base_tinv, desc_words, cu_seqlens, k, tinv, n_batch, b_t, expand_num)
    emit_tinv_rows(b_t, expand_num, cu_seqlens, mRows, mCount, lane_idx)


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    publish_desc: cutlass.Constexpr[bool],
    k: cute.Tensor,
    k_desc: cute.Tensor,
    gate: cute.Tensor,
    a_log: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    tinv: cute.Tensor,
    rows: cute.Tensor,
    row_count: cute.Tensor,
    stream: cuda.CUstream,
):
    # ---- grid: one CTA per SM, capped by the tile count ------------------------------
    heads_out = cutlass.Int32(gate.shape[1])
    k_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(k.shape[1]))
    num_ctas = cutlass.min(cutlass.Int32(cfg.num_sm), cutlass.Int32(tinv.shape[0]) * heads_out)

    # ---- SMEM sizing: per-buffer element cosizes -------------------------------------
    bytes_per_element = cfg.io_dtype.width // 8
    k_stage_elements = 2 * cfg.b_t * cfg.d_k
    tile_elements = cfg.b_t * cfg.b_t
    cfg.k_cosize = k_stage_elements * cfg.smem_k_stages
    cfg.tile_cosize = tile_elements * len(cfg.compute_group_warp_ids) * cfg.smem_tile_stages

    cfg.tma_k_bytes = cfg.b_t * cfg.d_k * bytes_per_element
    cfg.tile_bytes = tile_elements * bytes_per_element

    # ---- base K and tinv maps (the prologue emits their per-batch arrays) ------------
    box_elems = 128 // bytes_per_element
    swizzle_128b = tma.TensorMapSwizzle.s128b
    k_headed = cute.make_tensor(k.iterator, cute.make_layout((k.shape[0], k.shape[1], k.shape[2]), stride=(k.stride[0], k.stride[1], 1)))
    base_desc_k = tma.create_tensor_map_tiled_from_view(k_headed, box_dims=(cfg.b_t, 1, box_elems), stride_order=(2, 1, 0), swizzle=swizzle_128b)
    tinv_tiles = cute.make_tensor(
        tinv.iterator,
        cute.make_layout((tinv.shape[0], tinv.shape[1], tinv.shape[2], tinv.shape[3]), stride=(tinv.stride[0], tinv.stride[1], tinv.stride[2], 1)),
    )
    base_desc_tinv = tma.create_tensor_map_tiled_from_view(tinv_tiles, box_dims=(1, 1, cfg.b_t, cfg.b_t), stride_order=(3, 2, 1, 0), swizzle=swizzle_128b)

    # ---- launch ----------------------------------------------------------------------
    if cutlass.const_expr(publish_desc):
        frost_gdn_tinv_prologue(cfg.b_t, cfg.expand_num, base_desc_k, base_desc_tinv, k_desc, cu_seqlens, k, tinv, rows, row_count).launch(
            grid=(1, 1, 1), block=(cfg.threads_per_warp, 1, 1), stream=stream, use_pdl=USE_PDL
        )
    frost_gdn_tinv(
        cfg,
        cute.FastDivmodDivisorV2(heads_out),
        k_ratio,
        cute.FastDivmodDivisorV2(num_ctas),
        k_desc,
        gate,
        a_log,
        dt_bias,
        beta,
        rows,
        row_count,
        tinv,
        cutlass.Int32(cu_seqlens.shape[0] - 1),
    ).launch(
        grid=(num_ctas, 1, 1),
        block=(cfg.threads_per_cta, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
        min_blocks_per_mp=1,
    )


@cute.kernel
def frost_gdn_tinv(
    cfg: cutlass.Constexpr,
    heads_out: cute.FastDivmodDivisorV2,
    k_ratio: cute.FastDivmodDivisorV2,
    num_ctas: cute.FastDivmodDivisorV2,
    mDesc: cute.Tensor,
    mGate: cute.Tensor,
    mA_log: Optional[cute.Tensor],
    mDt_bias: Optional[cute.Tensor],
    mBeta: cute.Tensor,
    mRows: cute.Tensor,
    mCount: cute.Tensor,
    mTinv: cute.Tensor,
    num_batches: cutlass.Int32,
):
    """Chunk-factor kernel: persistent CTAs each own a contiguous block of the row table's (chunk row, head) tiles,
    warp-specialized pair loop over the block."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = tidx % cutlass.Int32(cfg.threads_per_warp)
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    arr_words = num_batches * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_k_base = mDesc.iterator.raw_ptr()
    desc_tinv_base = desc_k_base + arr_words
    n_items = cutlass.Int32(mCount[0]) * heads_out.divisor
    item_begin = (bidx * n_items) // num_ctas
    n_tiles = ((bidx + cutlass.Int32(1)) * n_items) // num_ctas - item_begin
    n_pairs = (n_tiles + cutlass.Int32(1)) // cutlass.Int32(2)
    NG = len(cfg.compute_group_warp_ids)
    GS = cfg.smem_gate_stages

    SMEM = cutlass.AddressSpace.smem

    SWZ = 2
    LEAD = 16
    STRIDE = 8 * 128
    sK_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.k_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sK = SmemTile(
        base=sK_raw.data_ptr(),
        elems_per_stage=(cfg.k_cosize // cfg.smem_k_stages),
        stages=cfg.smem_k_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_tma = SmemTile(
        base=sK_raw,
        elems_per_stage=(cfg.k_cosize // cfg.smem_k_stages),
        stages=cfg.smem_k_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_k // 64,
        tma_granu_elems=64,
        tma_subtile_stride_elems=2 * cfg.b_t * 64,
    )
    sTinv_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.tile_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sTinv = SmemTile(
        base=sTinv_raw.data_ptr(),
        elems_per_stage=(cfg.tile_cosize // (NG * cfg.smem_tile_stages)),
        stages=NG * cfg.smem_tile_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sTinv_tma = SmemTile(
        base=sTinv_raw,
        elems_per_stage=(cfg.tile_cosize // (NG * cfg.smem_tile_stages)),
        stages=NG * cfg.smem_tile_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=1,
        tma_granu_elems=cfg.b_t,
        tma_subtile_stride_elems=cfg.b_t * cfg.b_t,
    )
    bars = make_bars(cfg)
    tmem_base_slot = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=16)
    cumsumlog_smem_layout_staged = cute.make_layout((cfg.b_t, 1, 2, NG * GS))
    cumsumlog_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
    beta_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
    sCumsumlog = cute.make_tensor(
        cute.make_ptr(cutlass.Float32, cumsumlog_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=128),
        cumsumlog_smem_layout_staged,
    )
    sBeta = cute.make_tensor(
        cute.make_ptr(cutlass.Float32, beta_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=128),
        cumsumlog_smem_layout_staged,
    )

    # ---- mbarrier init (thread 0) ----------------------------------------------------
    if tidx == 0:
        for s in range(cfg.smem_k_stages):
            bars.mb_k_ready[s].init()
            bars.mb_k_done[s].init()
        for s in range(NG * cfg.smem_gate_stages):
            bars.mb_gate_ready[s].init()
            bars.mb_gate_done[s].init()
        for s in range(cfg.tmem_acc_stages):
            bars.mb_acc_ready[s].init()
            bars.mb_acc_done[s].init()
        for s in range(NG * cfg.smem_tile_stages):
            bars.mb_tile_ready[s].init()
            bars.mb_tile_done[s].init()

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    if warp_idx == 0:
        nvvm.tcgen05_alloc(tmem_base_slot, cutlass.Int32(cfg.tmem_columns), group=nvvm.CTAGroup.CTA_1)
        nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync()
    tmem_col = tmem_base_slot.load() & 0xFFFF

    # ---- warp specialization ---------------------------------------------------------
    if warp_idx == cfg.tma_k_warp_id:
        tmaldg_warp(
            cfg,
            n_tiles,
            n_pairs,
            item_begin,
            mRows,
            desc_k_base=desc_k_base,
            sK_tma=sK_tma,
            bars=bars,
            heads_out=heads_out,
            k_ratio=k_ratio,
        )
    elif warp_idx == cfg.tcgen05_mma_warp_id:
        tcgen05_mma_warp(
            cfg,
            n_pairs,
            tmem_col=tmem_col,
            sK=sK,
            bars=bars,
        )
    elif warp_idx == cfg.epilogue_warp_id:
        tmastg_warp(
            cfg,
            n_tiles,
            n_pairs,
            item_begin,
            mRows,
            desc_tinv_base=desc_tinv_base,
            sTinv_tma=sTinv_tma,
            bars=bars,
            heads_out=heads_out,
        )
    elif warp_idx == cfg.load_gate_warp_id:
        gate_warp(
            cfg,
            n_tiles,
            n_pairs,
            item_begin,
            mRows,
            lane_idx=lane_idx,
            mGate=mGate,
            mA_log=mA_log,
            mDt_bias=mDt_bias,
            mBeta=mBeta,
            sCumsumlog=sCumsumlog,
            sBeta=sBeta,
            bars=bars,
            heads_out=heads_out,
        )
    else:
        for group in cutlass.range_constexpr(NG):
            if warp_idx // cutlass.Int32(len(cfg.compute_group_warp_ids[0])) == group:
                compute_warp_group(
                    replace(cfg, inverse_barrier_id=1 + group),
                    group,
                    n_tiles,
                    n_pairs,
                    tidx,
                    warp_id=warp_idx - cutlass.Int32(cfg.compute_group_warp_ids[group][0]),
                    lane_idx=lane_idx,
                    tmem_col=tmem_col,
                    sTinv=sTinv,
                    sCumsumlog=cute.make_tensor(
                        cute.make_ptr(
                            cutlass.Float32,
                            cumsumlog_raw.data_ptr().toint() + group * GS * 2 * cfg.b_t * 4,
                            mem_space=cute.AddressSpace.smem,
                            assumed_align=128,
                        ),
                        cute.make_layout((cfg.b_t, 1, 2, GS)),
                    ),
                    sBeta=cute.make_tensor(
                        cute.make_ptr(
                            cutlass.Float32, beta_raw.data_ptr().toint() + group * GS * 2 * cfg.b_t * 4, mem_space=cute.AddressSpace.smem, assumed_align=128
                        ),
                        cute.make_layout((cfg.b_t, 1, 2, GS)),
                    ),
                    bars=bars,
                )

    nvvm.barrier_cta_sync()
    if warp_idx == 0:
        nvvm.tcgen05_dealloc(nvvm.make_tmem_ptr(tmem_col, cutlass.Int8), cutlass.Int32(cfg.tmem_columns), group=nvvm.CTAGroup.CTA_1)
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@dataclass
class GdnTinvCfg:
    """Per-compile chunk-factor kernel knob (``build_cfg``): the dtype / head-count / gate-flag fields are the ``cute.compile``
    cache keys, the rest derives from ``CFG``; ``host`` stamps the shape-derived fields at trace time.
    """

    io_dtype: Type[cutlass.Numeric]
    acc_dtype: Type[cutlass.Numeric]
    d_k: int
    log_gate: bool = False
    safe_gate: bool = False
    beta_sigmoid: bool = False
    allow_neg_eigval: bool = False

    # ---- fixed constants stamped from CFG at build time ------------------------------
    b_t: int = CFG.B_T
    expand_num: int = 1
    num_sm: int = 0  # SM count of the device the plan is built for
    compute_group_warp_ids: Tuple[Tuple[int, ...], ...] = CFG.COMPUTE_GROUP_WARP_IDS
    load_gate_warp_id: int = CFG.LOAD_GATE_WARP_ID
    tma_k_warp_id: int = CFG.TMA_K_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    num_regs_compute: int = 0
    num_regs_other: int = CFG.NUM_REGS_OTHER
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0

    # ---- named barrier slots (0 is the CTA-wide sync; compute group g uses 1 + g) ----
    inverse_barrier_id: int = 1
    inverse_barrier_threads: int = 0

    # ---- SMEM / TMEM stage counts ----------------------------------------------------
    smem_k_stages: int = CFG.SMEM_K_STAGES
    smem_tile_stages: int = CFG.SMEM_TILE_STAGES
    smem_gate_stages: int = CFG.SMEM_GATE_STAGES
    tmem_acc_stages: int = CFG.TMEM_ACC_STAGES
    tmem_columns: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- stamped by host at trace time (shape-derived) -------------------------------
    k_cosize: int = 0
    tile_cosize: int = 0
    tma_k_bytes: int = 0
    tile_bytes: int = 0


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    *,
    num_sm: int,
    log_gate: bool = False,
    safe_gate: bool = False,
    beta_sigmoid: bool = False,
    allow_neg_eigval: bool = False,
    d_k: int,
    expand_num: int = 1,
) -> GdnTinvCfg:
    """Build the per-compile ``GdnTinvCfg`` (io_dtype in {Float16, BFloat16}; acc is always Float32)."""
    if d_k not in KEY_DIMS:
        raise ValueError(f"the chunk-factor pass serves d_k in {KEY_DIMS}, got d_k={d_k}")
    cfg = GdnTinvCfg(
        io_dtype=io_dtype,
        acc_dtype=cutlass.Float32,
        num_sm=num_sm,
        log_gate=log_gate,
        safe_gate=safe_gate,
        beta_sigmoid=beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        d_k=d_k,
        expand_num=expand_num,
    )
    n_groups = len(cfg.compute_group_warp_ids)
    n_group_warps = len(cfg.compute_group_warp_ids[0])
    other_threads = cfg.threads_per_warp * 4
    compute_threads = cfg.threads_per_warp * n_groups * n_group_warps
    cfg.threads_per_cta = other_threads + compute_threads
    cfg.num_regs_compute = min(256, (cfg.threads_per_cta * CFG.LAUNCH_REGS - other_threads * cfg.num_regs_other) // compute_threads // 8 * 8)
    cfg.inverse_barrier_threads = cfg.threads_per_warp * n_group_warps
    cfg.tmem_columns = cfg.tmem_acc_stages * 2 * cfg.b_t
    return cfg


def tinv_rows(total_tokens: int, num_seqs: int, expand_num: int = 1, b_t: int = CFG.B_T) -> int:
    """Tile rows of the ``tinv`` buffer: one row per chunk plus one padding row per sequence."""
    return total_tokens * expand_num // b_t + num_seqs


TENSORMAP_DESC_ARRAYS = 2  # per-batch runtime TMA descriptors: K, tinv


# ---------------------------------------------------------------------------


@functools.cache
def get_compiled_cache(
    io_dtype_str: str,
    cu_dtype_str: str,
    gate_dtype_str: str,
    a_log_dtype_str: str,
    dt_bias_dtype_str: str,
    beta_dtype_str: str,
    device: int,
    DK: int,
    expand_num: int,
    log_gate: bool,
    safe_gate: bool,
    beta_sigmoid: bool,
    allow_neg_eigval: bool,
    publish_desc: bool,
):
    """Return a mutable dict that lazily stores the compiled kernel."""
    return {}


def compile(
    io_dtype,
    log_gate: bool = False,
    safe_gate: bool = False,
    beta_sigmoid: bool = False,
    allow_neg_eigval: bool = False,
    publish_desc: bool = True,
    *,
    num_sm: int,
    d_k: int,
    expand_num: int = 1,
    k_cute,
    workspace_cute,
    gate_cute,
    a_log_cute=None,
    dt_bias_cute=None,
    beta_cute,
    cu_seqlens_cute,
    tinv_cute,
    rows_cute,
    row_count_cute,
    stream,
):
    """JIT-compile the chunk-factor pass for one static config."""
    cfg = build_cfg(
        io_dtype,
        num_sm=num_sm,
        log_gate=log_gate,
        safe_gate=safe_gate,
        beta_sigmoid=beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        d_k=d_k,
        expand_num=expand_num,
    )

    return cute.compile(
        host,
        cfg,
        bool(publish_desc),
        k_cute,
        workspace_cute,
        gate_cute,
        a_log_cute,
        dt_bias_cute,
        beta_cute,
        cu_seqlens_cute,
        tinv_cute,
        rows_cute,
        row_count_cute,
        stream,
        options="--enable-tvm-ffi --opt-level 2",
    )


def chunk_gdn_tinv(
    k,
    gate,
    beta,
    cu_seqlens,
    tinv,
    *,
    log_gate: bool = False,
    safe_gate: bool = False,
    a_log=None,
    dt_bias=None,
    use_beta_sigmoid: bool = False,
    allow_neg_eigval: bool = False,
    expand_num: int = 1,
    workspace,
    row_table,
    row_count,
    device: int,
    stream,
    publish_desc: bool = True,
):
    """Execute the GDN chunk-factor pass (THD / varlen entry), compiled once per static config and replayed; tensors are
    DLPack CUDA tensors with a stride-1 innermost dim.  Two launches per call: the one-warp prologue emits the per-batch
    K and tinv descriptor arrays into ``workspace``, then the pass reads them.
    k: ``(total_tokens * expand_num, HK, DK)`` normalized keys, row and head strides multiples of 8 elements
    gate: raw linear alpha, natural-log decay under ``log_gate``, or raw logits under ``safe_gate`` (``a_log`` / ``dt_bias`` per head)
    beta: ``(total_tokens * expand_num, HO)`` post-sigmoid, or raw logits under ``use_beta_sigmoid``
    tinv: ``(tinv_rows(total_tokens, num_seqs, expand_num), HO, B_T, B_T)`` io dtype, contiguous; padding rows are left untouched
    row_table / row_count: ``(rows, 4)`` / ``(1,)`` int32 row table (chunk, batch, batch_start, batch_end per valid row, and
    the count), written by the prologue, or by the caller's prologue under ``publish_desc=False``
    expand_num: GDP's ``num_householder`` timeline factor (1 = off)
    publish_desc: False when the caller's prologue already emitted the descriptor arrays and the row table (no prologue launch)
    """
    HK = k.shape[1]
    HO = gate.shape[1]
    DK = k.shape[2]
    B = cu_seqlens.shape[0] - 1
    if workspace.shape[0] < TENSORMAP_DESC_ARRAYS * B * TENSOR_MAP_QWORDS:
        raise ValueError(
            f"workspace holds {workspace.shape[0]} int64 words, the K and tinv descriptor arrays need {TENSORMAP_DESC_ARRAYS * B * TENSOR_MAP_QWORDS}"
        )
    if tuple(tinv.shape[1:]) != (HO, CFG.B_T, CFG.B_T):
        raise ValueError(f"tinv must be (rows, {HO}, {CFG.B_T}, {CFG.B_T}), got {tuple(tinv.shape)}")
    if tinv.shape[0] < tinv_rows(gate.shape[0], B, expand_num):
        raise ValueError(f"tinv has {tinv.shape[0]} rows, needs {tinv_rows(gate.shape[0], B, expand_num)}")
    if tinv.shape[0] * HO * multiprocessor_count(device) >= 2**31:
        raise ValueError(f"the item block arithmetic needs rows * HO * num_sm < 2^31, got {tinv.shape[0]} * {HO} * {multiprocessor_count(device)}")
    if tuple(row_table.shape) != (tinv.shape[0], 4) or row_count.shape[0] < 1:
        raise ValueError(f"the row table must be ({tinv.shape[0]}, 4) int32 plus a 1-entry count, got {tuple(row_table.shape)} / {tuple(row_count.shape)}")
    _, _, k_strides, _, _ = probe(k)
    if k_strides is None:
        k_strides = (HK * DK, DK, 1)
    if k_strides[2] != 1 or k_strides[1] % 8 != 0 or k_strides[0] % 8 != 0:
        raise ValueError(f"k needs a stride-1 head dim and 16-byte aligned row / head strides, got strides {k_strides}")
    if not safe_gate:
        a_log = None
        dt_bias = None
    io_dtype = get_dtype(k.dtype)

    cu_stream = cuda.CUstream(int(stream))
    cache = get_compiled_cache(
        str(k.dtype),
        str(cu_seqlens.dtype),
        str(gate.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(beta.dtype),
        device,
        DK,
        expand_num,
        log_gate,
        safe_gate,
        use_beta_sigmoid,
        allow_neg_eigval,
        publish_desc,
    )

    if "compiled" not in cache:
        k_cute = from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2)
        workspace_cute = from_dlpack(workspace, assumed_align=128).mark_layout_dynamic()
        gate_cute = from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        a_log_cute = from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None
        dt_bias_cute = from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None
        beta_cute = from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic()
        tinv_cute = from_dlpack(tinv, assumed_align=128).mark_layout_dynamic(leading_dim=3)
        rows_cute = from_dlpack(row_table, assumed_align=16).mark_layout_dynamic(leading_dim=1)
        row_count_cute = from_dlpack(row_count, assumed_align=4).mark_layout_dynamic()

        cache["compiled"] = compile(
            io_dtype,
            log_gate,
            safe_gate,
            use_beta_sigmoid,
            allow_neg_eigval,
            publish_desc,
            d_k=DK,
            expand_num=expand_num,
            num_sm=multiprocessor_count(device),
            k_cute=k_cute,
            workspace_cute=workspace_cute,
            gate_cute=gate_cute,
            a_log_cute=a_log_cute,
            dt_bias_cute=dt_bias_cute,
            beta_cute=beta_cute,
            cu_seqlens_cute=cu_seqlens_cute,
            tinv_cute=tinv_cute,
            rows_cute=rows_cute,
            row_count_cute=row_count_cute,
            stream=cu_stream,
        )

    cache["compiled"](k, workspace, gate, a_log, dt_bias, beta, cu_seqlens, tinv, row_table, row_count, cu_stream)
    return cache


def run_tinv(cache, k, gate, beta, cu_seqlens, tinv, workspace, stream, a_log=None, dt_bias=None, *, row_table, row_count) -> None:
    """Replay the compiled plan (the prologue launch, then the pass).  The caller owns the
    contract, which the plan validated at build, so nothing here raises."""
    cu_stream = cuda.CUstream(int(stream))
    cache["compiled"](k, workspace, gate, a_log, dt_bias, beta, cu_seqlens, tinv, row_table, row_count, cu_stream)


frost_gdn_tinv_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_gdn_tinv.set_name_prefix("cudnn", remove_cutlass_symbol=False)
