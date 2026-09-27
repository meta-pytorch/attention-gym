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
Chunked Gated Delta Net (GDN) bprop state-summary kernel for SM100 / SM103 / SM107 (Cutlass primitives): the reverse
state-gradient recurrence alone (no dQ/dK/dV/dGate/dBeta, no V, no checkpoints, no l2norm Jacobian); the single output
is d_initial_state = dh0 (M^T dht + G, or G alone when d_final_state is absent).

Algorithm overview (per chunk c, iterated c = compute_end-1 .. write_start):
  Inputs : Q[BT,DK], K[BT,DK], dO[BT,DV], Gate[BT] (scalar gate), T_inv[BT,BT] (the chunk factor tile of the T pass)
  State  : dstate[DK,DV]  (state gradient, held in TMEM, seeded from d_final_state or zero, accumulated backward)

  MMA order.  A = the staged attention matrix (CG0's A epilogue); dO' = dO * cumprod * scale (CG0 restage):
  QK          : W_qk[BT,BT]  = Q  @ K^T          -> shared acc   (the A tile; the next chunk's QK is issued early)
  dV inter    : dV[DV,BT]    = dstate^T(TMEM) @ K^T -> the dV/dK slot (CG1 rescales it by the decay in place)
  dU intra    : dV/dK slot  += dO^T(SMEM) @ A(SMEM)
  dstate Q-term : dstate    += dO'^T(TMEM) @ Q
  dY          : dY[DV,BT]    = dU^T(TMEM f16) @ T(SMEM) -> shared acc
  dstate K-term : dstate    += dY'^T(TMEM f16) @ K   (dY' = -cumprod * dY)

  Epilogue (CG2): dstate carried to the next chunk as an f16 pack with the total-decay rescale; the item owning chunk 0
  stores d_initial_state.

GDP (expand_num > 1): q and dO stay on the compact token timeline, one 64-token tile of each per block of expand_num
chunks held in SMEM; CG0 masks the dO' stage and the A tile to the tokens whose readout sub-token lies in the chunk
(slot = expand_num * (token - first token of the chunk) + phase, every other row zero) from the token tables the gate
warp publishes per chunk (readout slot or -1, cumsum / cumprod at the slot or 0); q_step > 1 reads the phase rows of an
expanded q buffer through a strided descriptor.

SMEM layout (stage counts live in gdn_bprop_summary_config.py; sizes at DK = DV = 128):
  Buffer                       Size (B)  Stages
  Q                               16384       2
  K                               16384       4
  dO                              16384       2
  T_inv                            8192       1
  A tile                           8192       1
  cumsumlog / cumprod               256       2
  token slot / cumsum / cumprod   3 x 256     2    <-- expand_num > 1 only
  scheduler ticket ring               4       2    <-- next-tile publish ring

TMEM layout (512 columns):
  Buffer                  Cols
  dstate acc              128     <-- DKxDV fp32
  dV/dK acc                64
  dstate input             64     <-- f16 packed
  shared acc x2           128     <-- QK / dY
  shared input x2          64     <-- dO' / dU / dY' (f16 packed)
  Y                        32

Warp assignments (16 warps = 512 threads):
  warps 0-3     : compute group 0 - T-pairwise, dO' stage, A epilogue
  warps 4-7     : compute group 1 - dV rescale, dU / dY' stagings
  warps 8-11    : compute group 2 - dstate seed, f16 pack + total-decay rescale, d_initial_state store
  warp  12      : MMA warp       - issues the GEMMs
  warp  13      : TMA load warp  - loads Q, K, dO and the chunk-factor tiles
  warp  14      : gate warp      - loads Gate (double-buffered, backward order)
  warp  15      : epilogue warp  - scheduler-only shell
"""

import functools
from dataclasses import dataclass
from typing import NamedTuple, Optional, Type, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import cutlass.experimental.cuda.tensor_map as tma
from cutlass.cute.runtime import from_dlpack

from ..common.thd import emit_seq_descs, emit_tile_seq_descs, TENSOR_MAP_QWORDS
from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, decode_work_item, order_body
from ..common.host import get_dtype

RCP_LN2 = 1.4426950408889634  # 1/ln(2): natural-log gates -> the kernel's log2 domain
from ..tile_dsl.barrier import (
    MBarrier,
    Producer,
    PipelineState,
    advance,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from ..tile_dsl.handles import MmaDesc, SmemTile, tma_slice_runtime_desc
from ..tile_dsl.mma import mma_ss, mma_ts_step
from ..tile_dsl.pointwise import f16x2_to_f32, fadd2, fmul2, fp32_to_fp16, opaque_f32_zero, softplus2
from ..tile_dsl.swizzle import swizzle_xor_128b
from ..tile_dsl.tma import (
    tma_load_tile,
    tma_tensormap_acquire,
)
from .gdn_bprop_summary_config import CFG

USE_PDL = True


class GdnBpropSummaryBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_q_ready: MBarrier
    mb_q_mma_done: MBarrier
    mb_k_ready: MBarrier
    mb_k_mma_done: MBarrier
    mb_do_ready: MBarrier
    mb_do_mma_done: MBarrier
    mb_do_cg0_done: MBarrier

    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier

    mb_dstate_acc_ready: MBarrier
    mb_dstate_scale_acc_done: MBarrier
    mb_du_scale_acc_ready: MBarrier
    mb_du_scale_acc_done: MBarrier
    mb_du_total_acc_ready: MBarrier
    mb_du_total_acc_done: MBarrier
    mb_a_acc_ready: MBarrier
    mb_dy_acc_ready: MBarrier

    mb_dstate_input_ready: MBarrier
    mb_dstate_input_done: MBarrier
    mb_do_prime_input_ready: MBarrier
    mb_du_input_ready: MBarrier
    mb_dyp_input_ready: MBarrier

    mb_t_inv_ready: MBarrier
    mb_t_inv_done: MBarrier
    mb_a_ready: MBarrier
    mb_a_done: MBarrier

    mb_tmem_done: MBarrier
    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier


def make_bars(cfg) -> GdnBpropSummaryBars:
    """GdnBpropSummaryBars factory."""
    ONE_LANE = 1
    MMA_ARRIVERS = len([cfg.tcgen05_mma_warp_id])
    GATE_WARP = cfg.threads_per_warp * len([cfg.load_gate_beta_warp_id])
    CG0_THREADS = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    CG1_THREADS = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    CG2_THREADS = cfg.threads_per_warp * len(cfg.compute_group_2_warp_ids)
    ALL_COMPUTE = CG0_THREADS + CG1_THREADS + CG2_THREADS

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=16)

    return GdnBpropSummaryBars(
        mb_q_ready=MBarrier(alloc(cfg.smem_q_stages), try_wait=True, stages=cfg.smem_q_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_q_mma_done=MBarrier(
            alloc(cfg.smem_q_stages), try_wait=True, stages=cfg.smem_q_stages, init_count=MMA_ARRIVERS * cfg.expand_num, producer=Producer.MMA_COMMIT
        ),
        mb_k_ready=MBarrier(alloc(cfg.smem_k_stages), try_wait=True, stages=cfg.smem_k_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_mma_done=MBarrier(alloc(cfg.smem_k_stages), try_wait=True, stages=cfg.smem_k_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_do_ready=MBarrier(alloc(cfg.smem_do_stages), try_wait=True, stages=cfg.smem_do_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_do_mma_done=MBarrier(
            alloc(cfg.smem_do_stages), try_wait=True, stages=cfg.smem_do_stages, init_count=MMA_ARRIVERS * cfg.expand_num, producer=Producer.MMA_COMMIT
        ),
        mb_do_cg0_done=MBarrier(
            alloc(cfg.smem_do_stages), try_wait=True, stages=cfg.smem_do_stages, init_count=CG0_THREADS * cfg.expand_num, producer=Producer.THREAD
        ),
        mb_gate_ready=MBarrier(alloc(cfg.smem_gate_stages), try_wait=True, stages=cfg.smem_gate_stages, init_count=GATE_WARP, producer=Producer.THREAD),
        mb_gate_done=MBarrier(alloc(cfg.smem_gate_stages), try_wait=True, stages=cfg.smem_gate_stages, init_count=ALL_COMPUTE, producer=Producer.THREAD),
        mb_dstate_acc_ready=MBarrier(
            alloc(cfg.tmem_dstate_acc_stages), try_wait=True, stages=cfg.tmem_dstate_acc_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_dstate_scale_acc_done=MBarrier(
            alloc(cfg.tmem_dstate_acc_stages), try_wait=True, stages=cfg.tmem_dstate_acc_stages, init_count=CG2_THREADS, producer=Producer.THREAD
        ),
        mb_du_scale_acc_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_du_scale_acc_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_du_total_acc_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_du_total_acc_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_a_acc_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dy_acc_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_dstate_input_ready=MBarrier(
            alloc(cfg.tmem_dstate_input_stages), try_wait=True, stages=cfg.tmem_dstate_input_stages, init_count=CG2_THREADS, producer=Producer.THREAD
        ),
        mb_dstate_input_done=MBarrier(
            alloc(cfg.tmem_dstate_input_stages), try_wait=True, stages=cfg.tmem_dstate_input_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_do_prime_input_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_du_input_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dyp_input_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_t_inv_ready=MBarrier(alloc(cfg.smem_t_inv_stages), try_wait=True, stages=cfg.smem_t_inv_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_t_inv_done=MBarrier(
            alloc(cfg.smem_t_inv_stages), try_wait=True, stages=cfg.smem_t_inv_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_a_ready=MBarrier(alloc(cfg.smem_a_stages), try_wait=True, stages=cfg.smem_a_stages, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_a_done=MBarrier(alloc(cfg.smem_a_stages), try_wait=True, stages=cfg.smem_a_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_tmem_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=ALL_COMPUTE, producer=Producer.THREAD),
        mb_scheduler_ready=MBarrier(alloc(cfg.scheduler_stages), try_wait=True, stages=cfg.scheduler_stages, init_count=ONE_LANE, producer=Producer.THREAD),
        mb_scheduler_done=MBarrier(alloc(cfg.scheduler_stages), try_wait=True, stages=cfg.scheduler_stages, init_count=15, producer=Producer.THREAD),
    )


@cute.jit
def scheduler_publish_next(cfg, bars, sScheduler, mScheduler, scheduler_state, num_ctas, elect_one):
    """TMA-LDG-warp side: pull the next tile off the global ticket, publish it."""
    bars.mb_scheduler_done[scheduler_state.idx].wait(scheduler_state.phase)
    if elect_one:
        fetched = cutlass.Int32(nvvm.atomicrmw("add", mScheduler.iterator, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu"))
        sScheduler[scheduler_state.idx] = num_ctas + fetched
    nvvm.bar_warp_sync(cute.arch.FULL_MASK)
    next_tile = sScheduler[scheduler_state.idx]
    if elect_one:
        bars.mb_scheduler_ready[scheduler_state.idx].arrive()
    return next_tile, advance(scheduler_state, cfg.scheduler_stages)


@cute.jit
def scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one):
    """Consumer side: read the TMA-LDG warp's published next tile."""
    bars.mb_scheduler_ready[scheduler_state.idx].wait(scheduler_state.phase)
    next_tile = sScheduler[scheduler_state.idx]
    if elect_one:
        bars.mb_scheduler_done[scheduler_state.idx].arrive()
    return next_tile, advance(scheduler_state, cfg.scheduler_stages)


@cute.jit
def tmastg_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    sScheduler,
    bars,
):
    """Scheduler-only shell for warp 15; it arrives on ``mb_scheduler_done``
    so that barrier's 15-warp population holds."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    scheduler_state = PipelineState.start(phase=0)

    elect_one = nvvm.elect_sync()
    tile_idx = cutlass.Int32(bidx)

    while tile_idx < total_tiles:
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)


@cute.jit
def gate_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tidx,
    mGate,
    mA_log,
    mDt_bias,
    sCumsumlog,
    sCumprod,
    sTokSlot,
    sTokCumsum,
    sTokCumprod,
    sScheduler,
    bars,
):
    """Gate load warp role (warp 14): per-chunk Gate G->S loads and, under expand_num > 1, the chunk's block-token tables
    (readout slot, cumsum and cumprod at the slot); the stage recycle waits on the consumers' done barrier."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    elect_one = nvvm.elect_sync()

    gate_index = PipelineState.start(phase=1)
    gate_store_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)

    lane_idx = tidx % cfg.threads_per_warp

    n_cols = cfg.b_t // cfg.threads_per_warp
    if cutlass.const_expr(cfg.safe_gate and mA_log is None):
        a = opaque_f32_zero() - cutlass.Float32(RCP_LN2)
    else:
        a = cutlass.Float32(0.0)
    bias = cutlass.Float32(0.0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_item_chunks = compute_end - write_start
        if cutlass.const_expr(cfg.safe_gate and mA_log is not None):
            if num_item_chunks > 0:
                a = -cute.math.exp2(mA_log[head_idx].to(cutlass.Float32) * cutlass.Float32(RCP_LN2), fastmath=True) * cutlass.Float32(RCP_LN2)
        if cutlass.const_expr(cfg.safe_gate and mDt_bias is not None):
            if num_item_chunks > 0:
                bias = mDt_bias[head_idx].to(cutlass.Float32)

        for rev_idx in cutlass.range(num_item_chunks + 1):
            # ---- prefetch the NEXT chunk's Gate --------------------------------------
            if rev_idx < num_item_chunks:
                chunk_offset = batch_start + (compute_end - 1 - rev_idx) * cfg.b_t
                gGateSeq = mGate[None, head_idx]
                gGate = cute.domain_offset((chunk_offset,), gGateSeq)
                gate_idx = gate_index.idx
                gate_index = advance(gate_index, cfg.smem_gate_stages)
                pos_valid = [(chunk_offset + lane_idx + col * cfg.threads_per_warp) < batch_end for col in range(n_cols)]

                # ---- Gate load: GMEM -> SMEM (OOB neutral: 1.0 -> log2 = 0.0) --------
                oob_neutral = cutlass.Float32(0.0) if cutlass.const_expr(cfg.log_gate) else cutlass.Float32(1.0)
                if cutlass.const_expr(cfg.expand_num > 1):
                    gate_toks = [chunk_offset + lane_idx + col * cfg.threads_per_warp for col in range(n_cols)]
                    gate_rows = [tok // cutlass.Int32(cfg.expand_num) for tok in gate_toks]
                    gate_valid = [valid and (tok - row * cutlass.Int32(cfg.expand_num) == 0) for tok, row, valid in zip(gate_toks, gate_rows, pos_valid)]
                    gate_vals = [
                        gGateSeq[row if valid else cutlass.Int32(0)].to(cutlass.Float32) if valid else oob_neutral for row, valid in zip(gate_rows, gate_valid)
                    ]
                else:
                    gate_valid = pos_valid
                    gate_vals = [
                        gGate[min(lane_idx + col * cfg.threads_per_warp, batch_end - chunk_offset - 1)].to(cutlass.Float32) if pos_valid[col] else oob_neutral
                        for col in range(n_cols)
                    ]

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
                    sCumsumlog[pos, 0, gate_idx] = gate_vals[col]
                    sCumprod[pos, 0, gate_idx] = cute.math.exp2(gate_vals[col], fastmath=True)

                if cutlass.const_expr(cfg.expand_num > 1):
                    # token tables: the block's token t reads out at slot R * (t - lo) + phi of this chunk when
                    # lo <= t < lo + n, else nowhere (slot -1, cumsum and cumprod 0)
                    ld_chunk = compute_end - 1 - rev_idx
                    gdp_sub = ld_chunk % cutlass.Int32(cfg.expand_num)
                    gdp_lo = (cutlass.Int32(cfg.b_t) * gdp_sub) // cutlass.Int32(cfg.expand_num)
                    gdp_n = (cutlass.Int32(cfg.b_t) * (gdp_sub + 1)) // cutlass.Int32(cfg.expand_num) - gdp_lo
                    gdp_phi = cutlass.Int32(cfg.expand_num) * gdp_lo + cutlass.Int32(cfg.expand_num - 1) - cutlass.Int32(cfg.b_t) * gdp_sub
                    nvvm.bar_warp_sync(cute.arch.FULL_MASK)
                    for col in cutlass.range_constexpr(n_cols):
                        tok = lane_idx + col * cfg.threads_per_warp
                        member = tok >= gdp_lo and tok < gdp_lo + gdp_n
                        slot = cutlass.Int32(cfg.expand_num) * (tok - gdp_lo) + gdp_phi
                        slot = slot if member else cutlass.Int32(0)
                        sTokSlot[tok, 0, gate_idx] = slot if member else cutlass.Int32(-1)
                        sTokCumsum[tok, 0, gate_idx] = sCumsumlog[slot, 0, gate_idx] if member else cutlass.Float32(0.0)
                        sTokCumprod[tok, 0, gate_idx] = sCumprod[slot, 0, gate_idx] if member else cutlass.Float32(0.0)

                bars.mb_gate_ready[gate_idx].arrive()

            # ---- Gate stage recycle --------------------------------------------------
            if rev_idx > 0:
                gate_store_idx = gate_store_index.idx
                bars.mb_gate_done[gate_store_idx].wait(gate_store_index.phase)
                gate_store_index = advance(gate_store_index, cfg.smem_gate_stages)
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)


@cute.jit
def tcgen05_mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tmem_base_slot,
    sQ,
    sQ_trans,
    sK,
    sK_trans,
    sdO_trans,
    sTinv_trans,
    sA,
    sA_trans,
    sScheduler,
    bars,
):
    """MMA issuer role (warp 12): every tcgen05 GEMM of the reverse dstate recurrence.  Per work item the first chunk's QK
    GEMM is issued before the chunk loop; per chunk the order is dV inter, dstate Q-term, dU intra, QK of the next chunk,
    dY, dstate K-term."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    du_total_done_index = PipelineState.start(phase=1)
    du_scale_index = PipelineState.start(phase=0)
    dstate_acc_index = PipelineState.start(phase=0 if cfg.use_dstate_in else 1)
    k_index = PipelineState.start(phase=0)
    q_index = PipelineState.start(phase=0)
    tinv_index = PipelineState.start(phase=0)
    do_index = PipelineState.start(phase=0)
    a_index = PipelineState.start(phase=0)
    do_prime_input_ready = PipelineState.start(phase=0)
    du_input_ready = PipelineState.start(phase=0)
    dyp_input_ready = PipelineState.start(phase=0)
    dstate_input_index = PipelineState.start(phase=0)

    elect_one = nvvm.elect_sync()

    nvvm.tcgen05_alloc(tmem_base_slot, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    tmem_row << 16
    (tmem_row + 16) << 16

    # ---- chunk-invariant GEMM descriptors --------------------------------------------
    bytes_per_element = cfg.io_dtype.width // 8
    instruction_descriptor_qk = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.b_t,
    )
    bmm_q_k_desc = MmaDesc(
        M=cfg.b_t,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_qk,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    tmem_shared_acc_col = tmem_col + cfg.tmem_shared_acc_offset
    tmem_shared_input_col = tmem_col + cfg.tmem_shared_input_offset
    SHARED_INP_STAGE_COLS = cfg.b_t // 2
    tmem_do_prime_col = tmem_shared_input_col
    tmem_du_col = tmem_shared_input_col + SHARED_INP_STAGE_COLS
    tmem_dyp_col = tmem_du_col
    ACC_STAGE_COLS = cfg.b_t
    tmem_acc_a = tmem_shared_acc_col
    tmem_acc_b = tmem_shared_acc_col + ACC_STAGE_COLS
    tmem_dy_col = tmem_acc_a
    tmem_a_col = tmem_acc_b

    instruction_descriptor_dv = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_dstate_k_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_dv,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    tmem_dstate_input_col = tmem_col + cfg.tmem_dstate_input_offset
    tmem_dvdk_acc_col = tmem_col + cfg.tmem_dvdk_acc_offset
    DSTATE_INP_STAGE_COLS = cfg.d_k // 2

    instruction_descriptor_du = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        a_major=1,
        b_major=1,
    )
    bmm_do_a_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        atranspose=True,
        cta_group=1,
        idesc=instruction_descriptor_du,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    instruction_descriptor_dstate_upd = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_k,
        m_dim=cfg.d_v,
        b_major=1,
    )
    bmm_do_prime_q_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.d_k,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_dstate_upd,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_dy_prime_k_desc = bmm_do_prime_q_desc
    tmem_dstate_acc_col = tmem_col + cfg.tmem_dstate_acc_offset

    instruction_descriptor_dy = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        b_major=1,
    )
    bmm_du_t_inv_trans_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_dy,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    do_prime_input_ptr = nvvm.make_tmem_ptr(tmem_do_prime_col, cutlass.Int8)
    du_input_ptr = nvvm.make_tmem_ptr(tmem_du_col, cutlass.Int8)
    dyp_input_ptr = nvvm.make_tmem_ptr(tmem_dyp_col, cutlass.Int8)
    a_acc_ptr = nvvm.make_tmem_ptr(tmem_a_col, cutlass.Float32)
    dy_acc_ptr = nvvm.make_tmem_ptr(tmem_dy_col, cutlass.Float32)
    dstate_acc_ptr = nvvm.make_tmem_ptr(tmem_dstate_acc_col, cutlass.Float32)
    dvdk_acc_ptr = nvvm.make_tmem_ptr(tmem_dvdk_acc_col, cutlass.Float32)

    # ---- warp-top descriptors --------------------------------------------------------
    d_q0 = sQ[0].desc()
    d_k0 = sK[0].desc()
    d_k_trans0 = sK_trans[0].desc()
    d_q_trans0 = sQ_trans[0].desc()
    d_do_trans0 = sdO_trans[0].desc()
    d_tinv_trans0 = sTinv_trans[0].desc()
    d_a_trans0 = sA_trans[0].desc()
    K_STAGE_BYTES = (cfg.k_cosize // cfg.smem_k_stages) * bytes_per_element
    Q_STAGE_BYTES = (cfg.q_cosize // cfg.smem_q_stages) * bytes_per_element
    DO_STAGE_BYTES = (cfg.do_cosize // cfg.smem_do_stages) * bytes_per_element

    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_item_chunks = compute_end - write_start

        # ---- QK of the item's first chunk --------------------------------------------
        k_cur = k_index.idx
        q_cur = q_index.idx
        do_cur = do_index.idx
        block_last = True
        if cutlass.const_expr(cfg.expand_num > 1):
            # the q / dO slots are per block; sub is the chunk's position in its block (R-1 at the head, 0 at the
            # last chunk), their done barriers take R commits per phase, one per chunk, and the item's partial first
            # and last blocks are padded with idle commits so every block use completes exactly one phase
            sub = (compute_end - 1) % cutlass.Int32(cfg.expand_num)
            first_block_start = ((compute_end - 1) // cutlass.Int32(cfg.expand_num)) * cutlass.Int32(cfg.expand_num)
            last_block_end = (write_start // cutlass.Int32(cfg.expand_num) + 1) * cutlass.Int32(cfg.expand_num)
            first_block_start = first_block_start if first_block_start > write_start else write_start
            last_block_end = last_block_end if last_block_end < compute_end else compute_end
            single_block = first_block_start <= write_start
            pad_pre = (cutlass.Int32(cfg.expand_num) - num_item_chunks) if single_block else (cutlass.Int32(cfg.expand_num) - (compute_end - first_block_start))
            pad_post = cutlass.Int32(0) if single_block else (cutlass.Int32(cfg.expand_num) - (last_block_end - write_start))
            block_last = sub == 0
            if num_item_chunks > 0:
                for _ in cutlass.range(pad_pre):
                    if elect_one:
                        bars.mb_q_mma_done[q_cur].arrive(cta_group=1)
                        bars.mb_do_mma_done[do_cur].arrive(cta_group=1)
        if num_item_chunks > 0:
            bars.mb_k_ready[k_cur].wait(k_index.phase)
            k_index = advance(k_index, cfg.smem_k_stages)
            bars.mb_q_ready[q_cur].wait(q_index.phase)
            if cutlass.const_expr(cfg.expand_num == 1):
                q_index = advance(q_index, cfg.smem_q_stages)
            mma_ss(
                bmm_q_k_desc,
                d_q0.advance_start_address(q_cur * Q_STAGE_BYTES),
                d_k0.advance_start_address(k_cur * K_STAGE_BYTES),
                a_acc_ptr,
                accumulate=False,
            )
            if elect_one:
                bars.mb_a_acc_ready[0].arrive(cta_group=1)

        # ---- chunks compute_end-1 .. write_start (backward) --------------------------
        for rev_idx in cutlass.range(num_item_chunks):
            compute_end - 1 - rev_idx
            have_dstate = cutlass.Boolean(True) if cutlass.const_expr(cfg.use_dstate_in) else rev_idx > 0
            desc_k = d_k0.advance_start_address(k_cur * K_STAGE_BYTES)
            desc_k_trans = d_k_trans0.advance_start_address(k_cur * K_STAGE_BYTES)
            desc_q_trans = d_q_trans0.advance_start_address(q_cur * Q_STAGE_BYTES)

            # ---- dV inter = dstate^T(T) @ K ------------------------------------------
            dstate_input_idx = dstate_input_index.idx
            if have_dstate:
                bars.mb_dstate_input_ready[dstate_input_idx].wait(dstate_input_index.phase)
                dstate_input_index = advance(dstate_input_index, cfg.tmem_dstate_input_stages)
            bars.mb_du_total_acc_done[0].wait(du_total_done_index.phase)
            du_total_done_index = advance(du_total_done_index, 1)

            if have_dstate:
                dstate_a_ptr = nvvm.make_tmem_ptr(tmem_dstate_input_col + dstate_input_idx * DSTATE_INP_STAGE_COLS, cutlass.Int8)
                for i in cutlass.range_constexpr(bmm_dstate_k_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_dstate_k_desc.sps_B):
                        mma_ts_step(
                            bmm_dstate_k_desc,
                            dstate_a_ptr.subview(i * bmm_dstate_k_desc.sps_B * bmm_dstate_k_desc.tmem_advance_A),
                            desc_k + i * (bmm_dstate_k_desc.smem_subtile_B >> 4),
                            dvdk_acc_ptr,
                            k,
                            cutlass.Boolean(i + k > 0),
                        )
                if elect_one:
                    bars.mb_du_scale_acc_ready[0].arrive(cta_group=1)
                    bars.mb_dstate_input_done[dstate_input_idx].arrive(cta_group=1)

            # ---- dstate Q-term += dO'^T(T) @ Q ---------------------------------------
            bars.mb_do_prime_input_ready[0].wait(do_prime_input_ready.phase)
            do_prime_input_ready = advance(do_prime_input_ready, 1)
            dstate_idx = dstate_acc_index.idx
            bars.mb_dstate_scale_acc_done[dstate_idx].wait(dstate_acc_index.phase)
            dstate_acc_index = advance(dstate_acc_index, cfg.tmem_dstate_acc_stages)

            for i in cutlass.range_constexpr(bmm_do_prime_q_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_do_prime_q_desc.sps_B):
                    mma_ts_step(
                        bmm_do_prime_q_desc,
                        do_prime_input_ptr.subview(i * bmm_do_prime_q_desc.sps_B * bmm_do_prime_q_desc.tmem_advance_A),
                        desc_q_trans + i * (bmm_do_prime_q_desc.smem_subtile_B >> 4),
                        dstate_acc_ptr,
                        k,
                        cutlass.Boolean(True) if cutlass.const_expr(i + k > 0) else have_dstate,
                    )
            if elect_one:
                bars.mb_q_mma_done[q_cur].arrive(cta_group=1)

            # ---- dU intra += dO^T(S) @ A (the block's dO tile) -----------------------
            if cutlass.const_expr(cfg.expand_num > 1):
                bars.mb_do_ready[do_cur].wait(do_index.phase)
            else:
                do_cur = do_index.idx
                bars.mb_do_ready[do_cur].wait(do_index.phase)
                do_index = advance(do_index, cfg.smem_do_stages)
            du_a_idx = a_index.idx
            bars.mb_a_ready[du_a_idx].wait(a_index.phase)
            a_index = advance(a_index, cfg.smem_a_stages)
            if have_dstate:
                bars.mb_du_scale_acc_done[0].wait(du_scale_index.phase)
                du_scale_index = advance(du_scale_index, 1)

            desc_do_trans = d_do_trans0.advance_start_address(do_cur * DO_STAGE_BYTES)
            desc_a_trans = d_a_trans0
            mma_ss(
                bmm_do_a_desc,
                desc_do_trans,
                desc_a_trans,
                dvdk_acc_ptr,
                accumulate=have_dstate,
            )
            if elect_one:
                bars.mb_du_total_acc_ready[0].arrive(cta_group=1)
                bars.mb_a_done[du_a_idx].arrive(cta_group=1)
                bars.mb_do_mma_done[do_cur].arrive(cta_group=1)

            # ---- QK of the next chunk = Q(S) @ K^T -----------------------------------
            k_next = k_cur
            q_next = q_cur
            if rev_idx + 1 < num_item_chunks:
                k_next = k_index.idx
                bars.mb_k_ready[k_next].wait(k_index.phase)
                k_index = advance(k_index, cfg.smem_k_stages)
                if cutlass.const_expr(cfg.expand_num > 1):
                    q_next = (q_cur ^ cutlass.Int32(1)) if block_last else q_cur
                    q_next_phase = (q_index.phase ^ q_cur) if block_last else q_index.phase
                    bars.mb_q_ready[q_next].wait(q_next_phase)
                else:
                    q_next = q_index.idx
                    bars.mb_q_ready[q_next].wait(q_index.phase)
                    q_index = advance(q_index, cfg.smem_q_stages)
                mma_ss(
                    bmm_q_k_desc,
                    d_q0.advance_start_address(q_next * Q_STAGE_BYTES),
                    d_k0.advance_start_address(k_next * K_STAGE_BYTES),
                    a_acc_ptr,
                    accumulate=False,
                )
                if elect_one:
                    bars.mb_a_acc_ready[0].arrive(cta_group=1)

            # ---- dY = dU^T(T) @ T ----------------------------------------------------
            bars.mb_du_input_ready[0].wait(du_input_ready.phase)
            du_input_ready = advance(du_input_ready, 1)
            tinv_idx = tinv_index.idx
            bars.mb_t_inv_ready[tinv_idx].wait(tinv_index.phase)
            tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)

            desc_tinv_trans = d_tinv_trans0.advance_start_address(tinv_idx * cfg.tma_tinv_bytes)
            for i in cutlass.range_constexpr(bmm_du_t_inv_trans_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_du_t_inv_trans_desc.sps_B):
                    mma_ts_step(
                        bmm_du_t_inv_trans_desc,
                        du_input_ptr.subview(i * bmm_du_t_inv_trans_desc.sps_B * bmm_du_t_inv_trans_desc.tmem_advance_A),
                        desc_tinv_trans + i * (bmm_du_t_inv_trans_desc.smem_subtile_B >> 4),
                        dy_acc_ptr,
                        k,
                        cutlass.Boolean(i + k > 0),
                    )
            if elect_one:
                bars.mb_dy_acc_ready[0].arrive(cta_group=1)
                bars.mb_t_inv_done[tinv_idx].arrive(cta_group=1)

            # ---- dstate K-term += dY'^T(T) @ K ---------------------------------------
            bars.mb_dyp_input_ready[0].wait(dyp_input_ready.phase)
            dyp_input_ready = advance(dyp_input_ready, 1)

            for i in cutlass.range_constexpr(bmm_dy_prime_k_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_dy_prime_k_desc.sps_B):
                    mma_ts_step(
                        bmm_dy_prime_k_desc,
                        dyp_input_ptr.subview(i * bmm_dy_prime_k_desc.sps_B * bmm_dy_prime_k_desc.tmem_advance_A),
                        desc_k_trans + i * (bmm_dy_prime_k_desc.smem_subtile_B >> 4),
                        dstate_acc_ptr,
                        k,
                        cutlass.Boolean(True),
                    )
            if elect_one:
                bars.mb_dstate_acc_ready[dstate_idx].arrive(cta_group=1)
                bars.mb_k_mma_done[k_cur].arrive(cta_group=1)
            k_cur = k_next
            if cutlass.const_expr(cfg.expand_num > 1):
                # q and dO share the block state (both rings hold 2 block tiles); placed after the chunk's last GEMM
                # issue so it overlaps the K-term
                q_index = PipelineState(idx=(q_cur ^ cutlass.Int32(1)) if block_last else q_cur, phase=(q_index.phase ^ q_cur) if block_last else q_index.phase)
                do_index = q_index
                q_cur = q_index.idx
                do_cur = q_cur
                sub = cutlass.Int32(cfg.expand_num - 1) if block_last else sub - cutlass.Int32(1)
                block_last = sub == 0
            else:
                q_cur = q_next

        if cutlass.const_expr(cfg.expand_num > 1):
            if num_item_chunks > 0:
                q_last = q_index.idx ^ cutlass.Int32(1)
                do_last = q_last
                for _ in cutlass.range(pad_post):
                    if elect_one:
                        bars.mb_q_mma_done[q_last].arrive(cta_group=1)
                        bars.mb_do_mma_done[do_last].arrive(cta_group=1)

        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    bars.mb_du_total_acc_done[0].wait(du_total_done_index.phase)
    bars.mb_tmem_done[0].wait(0)
    nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
    nvvm.tcgen05_dealloc(
        nvvm.make_tmem_ptr(tmem_col, cutlass.Int8),
        cutlass.Int32(512),
        group=nvvm.CTAGroup.CTA_1,
    )


@cute.jit
def tmaldg_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sQ_raw,
    sK_raw,
    sdO_raw,
    sTinv_raw,
    desc_tinv_base,
    desc_q_base,
    desc_k_base,
    desc_do_base,
    mScheduler,
    sScheduler,
    bars,
    q_ratio,
    k_ratio,
):
    """TMA-LDG warp role (warp 13): every Q/K/dO TMA load and the chunk-factor tile TMA loads; a dO stage is recycled
    after both its readers (the dU intra GEMM and CG0) release it."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    q_index = PipelineState.start(phase=1)
    k_index = PipelineState.start(phase=1)
    do_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=1)
    tinv_index = PipelineState.start(phase=1)
    sTinv_tma = SmemTile(
        base=sTinv_raw,
        elems_per_stage=cfg.t_inv_cosize // cfg.smem_t_inv_stages,
        stages=cfg.smem_t_inv_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=1,
        tma_granu_elems=cfg.b_t,
        tma_subtile_stride_elems=cfg.b_t * cfg.b_t,
    )

    elect_one = nvvm.elect_sync()
    tile_idx = cutlass.Int32(bidx)
    bytes_per_element = cfg.io_dtype.width // 8
    box_elems = 128 // bytes_per_element
    bt = cfg.b_t
    q_stage_elements = cfg.q_cosize // cfg.smem_q_stages
    k_stage_elements = cfg.k_cosize // cfg.smem_k_stages
    sQ_tma = SmemTile(
        base=sQ_raw,
        elems_per_stage=q_stage_elements,
        stages=cfg.smem_q_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_k // box_elems,
        tma_granu_elems=box_elems,
        tma_subtile_stride_elems=bt * box_elems,
    )
    sK_tma = SmemTile(
        base=sK_raw,
        elems_per_stage=k_stage_elements,
        stages=cfg.smem_k_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_k // box_elems,
        tma_granu_elems=box_elems,
        tma_subtile_stride_elems=bt * box_elems,
    )
    do_stage_elements = cfg.do_cosize // cfg.smem_do_stages
    sdO_tma = SmemTile(
        base=sdO_raw,
        elems_per_stage=do_stage_elements,
        stages=cfg.smem_do_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_v // box_elems,
        tma_granu_elems=box_elems,
        tma_subtile_stride_elems=cfg.b_t * box_elems,
    )
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o = head_idx
        head_q = head_idx // q_ratio
        head_k = head_idx // k_ratio
        slot = batch_idx * desc_qwords
        desc_q_slot = (desc_q_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_do_slot = (desc_do_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_do_slot)
        desc_tinv_slot = (desc_tinv_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_tinv_slot)

        if cutlass.const_expr(cfg.expand_num > 1):
            sub = (compute_end - 1) % cutlass.Int32(cfg.expand_num)
        for rev_idx in cutlass.range(compute_end - write_start):
            chunk_idx = compute_end - 1 - rev_idx
            tok_coord = chunk_idx * cutlass.Int32(cfg.b_t)
            block_head = True
            block_token_coord = tok_coord
            if cutlass.const_expr(cfg.expand_num > 1):
                block_head = (sub == cutlass.Int32(cfg.expand_num - 1)) or rev_idx == 0
                block_token_coord = (chunk_idx // cutlass.Int32(cfg.expand_num)) * cutlass.Int32(cfg.b_t)
                sub = cutlass.Int32(cfg.expand_num - 1) if sub == 0 else sub - cutlass.Int32(1)

            # ---- K load --------------------------------------------------------------
            k_idx = k_index.idx
            bars.mb_k_mma_done[k_idx].wait(k_index.phase)
            k_index = advance(k_index, cfg.smem_k_stages)
            if elect_one:
                bars.mb_k_ready[k_idx].arrive(n_bytes=cfg.tma_k_bytes)
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), head_k, tok_coord)
            tma_load_tile(sK_tma[k_idx], k_slice, bars.mb_k_ready[k_idx].smem_ptr)

            # ---- Q load: the block's 64 compact tokens -------------------------------
            if block_head:
                q_idx = q_index.idx
                bars.mb_q_mma_done[q_idx].wait(q_index.phase)
                q_index = advance(q_index, cfg.smem_q_stages)
                if elect_one:
                    bars.mb_q_ready[q_idx].arrive(n_bytes=cfg.tma_q_bytes)
                q_slice = tma_slice_runtime_desc(desc_q_slot, cutlass.Int32(0), head_q, block_token_coord)
                tma_load_tile(sQ_tma[q_idx], q_slice, bars.mb_q_ready[q_idx].smem_ptr)

            # ---- dO load: the block's 64 compact tokens ------------------------------
            if block_head:
                do_idx = do_index.idx
                bars.mb_do_mma_done[do_idx].wait(do_index.phase)
                bars.mb_do_cg0_done[do_idx].wait(do_index.phase)
                do_index = advance(do_index, cfg.smem_do_stages)
                if elect_one:
                    bars.mb_do_ready[do_idx].arrive(n_bytes=cfg.tma_do_bytes)
                do_slice = tma_slice_runtime_desc(desc_do_slot, cutlass.Int32(0), head_o, block_token_coord)
                tma_load_tile(sdO_tma[do_idx], do_slice, bars.mb_do_ready[do_idx].smem_ptr)

            # ---- chunk-factor tile load ----------------------------------------------
            tinv_idx = tinv_index.idx
            bars.mb_t_inv_done[tinv_idx].wait(tinv_index.phase)
            tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
            if elect_one:
                bars.mb_t_inv_ready[tinv_idx].arrive(n_bytes=cfg.tma_tinv_bytes)
            tinv_row = chunk_idx
            tinv_slice = tma_slice_runtime_desc(desc_tinv_slot, cutlass.Int32(0), cutlass.Int32(0), head_idx, tinv_row)
            tma_load_tile(sTinv_tma[tinv_idx], tinv_slice, bars.mb_t_inv_ready[tinv_idx].smem_ptr)

        next_tile, scheduler_state = scheduler_publish_next(cfg, bars, sScheduler, mScheduler, scheduler_state, num_ctas, elect_one)
        tile_idx = next_tile

    for _ in range(cfg.smem_q_stages):
        bars.mb_q_mma_done[q_index.idx].wait(q_index.phase)
        q_index = advance(q_index, cfg.smem_q_stages)
    for _ in range(cfg.smem_k_stages):
        bars.mb_k_mma_done[k_index.idx].wait(k_index.phase)
        k_index = advance(k_index, cfg.smem_k_stages)
    for _ in range(cfg.smem_do_stages):
        bars.mb_do_mma_done[do_index.idx].wait(do_index.phase)
        bars.mb_do_cg0_done[do_index.idx].wait(do_index.phase)
        do_index = advance(do_index, cfg.smem_do_stages)
    for _ in range(cfg.smem_t_inv_stages):
        bars.mb_t_inv_done[tinv_index.idx].wait(tinv_index.phase)
        tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.jit
def compute0_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tidx,
    tmem_base_slot,
    scale,
    sdO,
    sCumsumlog,
    sCumprod,
    sTokSlot,
    sTokCumsum,
    sTokCumprod,
    sA,
    sScheduler,
    bars,
):
    """Compute warp-group 0 role (warps 0-3): stages each chunk's dO' into the shared input columns and builds its
    attention matrix."""

    nvvm.setmaxregister(cfg.num_regs_compute_group_0, nvvm.SetMaxRegisterAction.INCREASE)
    elect_one = nvvm.elect_sync()

    gate_index = PipelineState.start(phase=0)
    a_index = PipelineState.start(phase=1)
    cg0_a_ready = PipelineState.start(phase=0)
    do_index = PipelineState.start(phase=0)
    do_cur = do_index.idx

    num_threads_cg0 = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    cg0_tidx = tidx % num_threads_cg0
    warp_id = cg0_tidx // cfg.threads_per_warp
    lane_idx = cg0_tidx % cfg.threads_per_warp
    store_row = warp_id * 16 + lane_idx % 16
    store_col = (lane_idx // 16) * 8

    cfg.io_dtype.width // 8
    num_vals = 32
    FRAG_COLS = 16
    ACC_N_FRAGS = cfg.b_t // FRAG_COLS
    ACC_STAGE_COLS = cfg.b_t
    mask_zero = opaque_f32_zero()

    # ---- dO fragment lane decode -----------------------------------------------------
    dv_halves = cutlass.const_expr(cfg.d_v // 64)
    frag_row = cg0_tidx % 8 + (cg0_tidx // 16 % 2) * 8
    frag_col = (cg0_tidx // 8 % 2) * 8 + (cg0_tidx // 32 % 2) * 32
    frag_segment = (cg0_tidx // 64) * (cfg.b_t * 64)
    if cutlass.const_expr(cfg.d_v == 128):
        dv_frag_col = frag_col
        dv_frag_segment = frag_segment
    else:
        dv_frag_col = (cg0_tidx // 32) * 16 + (cg0_tidx // 8 % 2) * 8
        dv_frag_segment = 0
    do_stage_elements = cfg.do_cosize // cfg.smem_do_stages
    sdO_base = sdO[0].base

    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    (tmem_row + 16) << 16
    tmem_shared_acc_col = tmem_col + cfg.tmem_shared_acc_offset
    tmem_acc_b = tmem_shared_acc_col + ACC_STAGE_COLS
    tmem_a_col = tmem_acc_b
    tmem_shared_input_col = tmem_col + cfg.tmem_shared_input_offset
    tmem_do_prime_col = tmem_shared_input_col

    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_item_chunks = compute_end - write_start

        block_last = True
        if cutlass.const_expr(cfg.expand_num > 1):
            sub = (compute_end - 1) % cutlass.Int32(cfg.expand_num)
            first_block_start = ((compute_end - 1) // cutlass.Int32(cfg.expand_num)) * cutlass.Int32(cfg.expand_num)
            last_block_end = (write_start // cutlass.Int32(cfg.expand_num) + 1) * cutlass.Int32(cfg.expand_num)
            first_block_start = first_block_start if first_block_start > write_start else write_start
            last_block_end = last_block_end if last_block_end < compute_end else compute_end
            single_block = first_block_start <= write_start
            pad_pre = (cutlass.Int32(cfg.expand_num) - num_item_chunks) if single_block else (cutlass.Int32(cfg.expand_num) - (compute_end - first_block_start))
            pad_post = cutlass.Int32(0) if single_block else (cutlass.Int32(cfg.expand_num) - (last_block_end - write_start))
            block_last = sub == 0
            if num_item_chunks > 0:
                for _ in cutlass.range(pad_pre):
                    bars.mb_do_cg0_done[do_index.idx].arrive()
        for chunk_idx in cutlass.range(num_item_chunks):

            # ---- T-pairwise ----------------------------------------------------------
            gate_idx = gate_index.idx
            bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)

            col_cumsums = []
            for g in cutlass.range_constexpr(8):
                for b in cutlass.range_constexpr(2):
                    col_cumsums.append(sCumsumlog[(lane_idx % 4) * 2 + g * 8 + b, 0, gate_idx])
            if cutlass.const_expr(cfg.expand_num > 1):
                # tile rows are the block's tokens; the gate warp's tables carry each row's readout slot in this
                # chunk (-1 = none) and its cumsum / cumprod there (0 = none)
                row_cumsums = []
                row_slot = []
                for r in cutlass.range_constexpr(2):
                    tokr = warp_id * 16 + lane_idx // 4 + r * 8
                    row_cumsums.append(sTokCumsum[tokr, 0, gate_idx])
                    row_slot.append(sTokSlot[tokr, 0, gate_idx])
                decay_t = []
                for k in cutlass.range_constexpr(num_vals):
                    rp = cutlass.const_expr((k // 2) % 2)
                    chunk_col = (lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2)
                    decay_t.append(
                        cute.math.exp2(row_cumsums[rp] - col_cumsums[(k // 4) * 2 + (k % 2)], fastmath=True) if chunk_col <= row_slot[rp] else mask_zero
                    )
                cumprod_fp32 = []
                for g in cutlass.range_constexpr(8):
                    for b in cutlass.range_constexpr(2):
                        cumprod_fp32.append(sTokCumprod[(lane_idx % 4) * 2 + g * 8 + b, 0, gate_idx])
                cumprod_vals = [cumprod_fp32[(k // 4) * 2 + (k % 2)] for k in range(num_vals)]
            else:
                row_cumsums = []
                for r in cutlass.range_constexpr(2):
                    row_cumsums.append(sCumsumlog[warp_id * 16 + lane_idx // 4 + r * 8, 0, gate_idx])
                decay_t = []
                for k in cutlass.range_constexpr(num_vals):
                    chunk_row = warp_id * 16 + lane_idx // 4 + ((k // 2) % 2) * 8
                    chunk_col = (lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2)
                    decay_t.append(
                        cute.math.exp2(row_cumsums[(k // 2) % 2] - col_cumsums[(k // 4) * 2 + (k % 2)], fastmath=True) if chunk_row >= chunk_col else mask_zero
                    )
                cumprod_fp32 = []
                for g in cutlass.range_constexpr(8):
                    for b in cutlass.range_constexpr(2):
                        cumprod_fp32.append(sCumprod[(lane_idx % 4) * 2 + g * 8 + b, 0, gate_idx])
                cumprod_vals = [cumprod_fp32[(k // 4) * 2 + (k % 2)] for k in range(num_vals)]

            # ---- A acc ready (the QK GEMM of this chunk committed) -------------------
            a_idx = a_index.idx
            a_phase = a_index.phase
            a_index = advance(a_index, cfg.smem_a_stages)
            bars.mb_a_acc_ready[0].wait(cg0_a_ready.phase)
            cg0_a_ready = advance(cg0_a_ready, 1)

            # ---- dO fragment preload (the block's tile, held for its chunks) ---------
            do_cur = do_index.idx
            bars.mb_do_ready[do_cur].wait(do_index.phase)
            if cutlass.const_expr(cfg.expand_num == 1):
                do_index = advance(do_index, cfg.smem_do_stages)
            do_raw_frag = []
            for half in cutlass.range_constexpr(dv_halves):
                do_words_raw = []
                for block in cutlass.range_constexpr(4):
                    do_raw = nvvm.ldmatrix(
                        (
                            sdO_base
                            + do_cur * do_stage_elements
                            + dv_frag_segment
                            + (frag_row + block * 16) * 64
                            + swizzle_xor_128b(frag_row + block * 16, dv_frag_col + half * 16)
                        ),
                        4,
                        nvvm.MMALayout.COL,
                    )
                    for i in cutlass.range_constexpr(4):
                        do_words_raw.append(do_raw[i])
                do_raw_frag.append(do_words_raw)
            bars.mb_do_cg0_done[do_cur].arrive()

            # ---- dO' stage: dO * cumprod * scale -> shared input TMEM ----------------
            for half in cutlass.range_constexpr(dv_halves):
                do_pack = []
                for j in cutlass.range_constexpr(16):
                    lo, hi = f16x2_to_f32(do_raw_frag[half][j], dtype=cfg.io_dtype)
                    p0, p1 = fmul2(lo, hi, cumprod_vals[2 * j], cumprod_vals[2 * j + 1])
                    q0, q1 = fmul2(p0, p1, scale, scale)
                    do_pack.append(fp32_to_fp16(q0, q1, dtype=cfg.io_dtype))
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_do_prime_col, cutlass.Int32),
                    cutlass.Vector.from_elements(tuple(do_pack), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_do_prime_input_ready[0].arrive()

            # ---- A epilogue: A[i,j] = W qk[i,j] * T[i,j] * scale ---------------------
            a_base = sA[a_idx].base
            a_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + tmem_a_col, cutlass.Float32), num=8)
            a_pack = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(a_vec[2 * k], a_vec[2 * k + 1], decay_t[2 * k], decay_t[2 * k + 1])
                v0, v1 = fmul2(p0, p1, scale, scale)
                a_pack.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            bars.mb_a_done[a_idx].wait(a_phase)
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    a_base + store_row * cfg.b_t + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS),
                    [a_pack[c * 4 + 0], a_pack[c * 4 + 1], a_pack[c * 4 + 2], a_pack[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_a_ready[a_idx].arrive()

            # ---- Gate stage release --------------------------------------------------
            bars.mb_gate_done[gate_idx].arrive()
            if cutlass.const_expr(cfg.expand_num > 1):
                do_index = PipelineState(
                    idx=(do_index.idx ^ cutlass.Int32(1)) if block_last else do_index.idx,
                    phase=(do_index.phase ^ do_index.idx) if block_last else do_index.phase,
                )
                sub = cutlass.Int32(cfg.expand_num - 1) if block_last else sub - cutlass.Int32(1)
                block_last = sub == 0
        if cutlass.const_expr(cfg.expand_num > 1):
            if num_item_chunks > 0:
                do_last = do_index.idx ^ cutlass.Int32(1)
                for _ in cutlass.range(pad_post):
                    bars.mb_do_cg0_done[do_last].arrive()
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)
    for _ in range(cfg.smem_a_stages):
        bars.mb_a_done[a_index.idx].wait(a_index.phase)
        a_index = advance(a_index, cfg.smem_a_stages)

    bars.mb_tmem_done[0].arrive()


@cute.jit
def compute1_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tidx,
    warp_idx,
    tmem_base_slot,
    scale,
    sCumsumlog,
    sCumprod,
    sScheduler,
    bars,
):
    """Compute warp-group 1 role (warps 4-7): persistent scheduler loop
    running each chunk's dV decay rescale and the dU / dY' stagings."""

    nvvm.setmaxregister(cfg.num_regs_compute_group_1, nvvm.SetMaxRegisterAction.INCREASE)
    elect_one = nvvm.elect_sync()

    gate_index = PipelineState.start(phase=0)
    cg1_dy_ready = PipelineState.start(phase=0)
    cg1_du_scale_ready = PipelineState.start(phase=0)
    cg1_du_total_ready = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)

    num_threads_cg1 = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    cg1_tidx = tidx % num_threads_cg1
    lane_idx = cg1_tidx % cfg.threads_per_warp

    SHARED_INP_STAGE_COLS = cfg.b_t // 2
    dv_halves = cutlass.const_expr(cfg.d_v // 64)
    tile_idx = cutlass.Int32(bidx)

    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    tmem_row << 16
    (tmem_row + 16) << 16
    tmem_dvdk_acc_col = tmem_col + cfg.tmem_dvdk_acc_offset
    tmem_shared_acc_col = tmem_col + cfg.tmem_shared_acc_offset
    tmem_shared_input_col = tmem_col + cfg.tmem_shared_input_offset
    tmem_du_col = tmem_shared_input_col + SHARED_INP_STAGE_COLS
    tmem_dyp_col = tmem_du_col
    tmem_acc_a = tmem_shared_acc_col
    tmem_dy_col = tmem_acc_a

    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_item_chunks = compute_end - write_start

        # ---- chunks NT-1 .. 0 (backward) ---------------------------------------------
        for rev_idx in cutlass.range(num_item_chunks):
            compute_end - 1 - rev_idx
            have_dstate = cutlass.Boolean(True) if cutlass.const_expr(cfg.use_dstate_in) else rev_idx > 0
            gate_idx = gate_index.idx
            bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)

            num_vals = 32
            cumprod_fp32 = []
            for g in cutlass.range_constexpr(8):
                for b in cutlass.range_constexpr(2):
                    cumprod_fp32.append(sCumprod[(lane_idx % 4) * 2 + g * 8 + b, 0, gate_idx])
            cumprod_vals = [cumprod_fp32[(k // 4) * 2 + (k % 2)] for k in range(num_vals)]

            # ---- dV inter ------------------------------------------------------------
            if have_dstate:
                last_cumsumlog = sCumsumlog[cfg.b_t - 1, 0, gate_idx]
                col_cumsumlog_fp32 = []
                for g in cutlass.range_constexpr(8):
                    for b in cutlass.range_constexpr(2):
                        col_cumsumlog_fp32.append(sCumsumlog[(lane_idx % 4) * 2 + g * 8 + b, 0, gate_idx])
                decay_scale_fp32 = []
                for i in cutlass.range_constexpr(16):
                    decay_scale_fp32.append(cute.math.exp2(last_cumsumlog - col_cumsumlog_fp32[i], fastmath=True))
                decay_scale_vals = [decay_scale_fp32[(k // 4) * 2 + (k % 2)] for k in range(num_vals)]
                bars.mb_du_scale_acc_ready[0].wait(cg1_du_scale_ready.phase)
                cg1_du_scale_ready = advance(cg1_du_scale_ready, 1)
                dv_ptrs = []
                dv_vecs = []
                for half in cutlass.range_constexpr(dv_halves):
                    dv_ptrs.append(nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_dvdk_acc_col, cutlass.Float32))
                    dv_vecs.append(nvvm.tcgen05_ld("16x256b", dv_ptrs[half], num=8))
                for half in cutlass.range_constexpr(dv_halves):
                    dv_scaled = []
                    for j in cutlass.range_constexpr(16):
                        s0, s1 = fmul2(dv_vecs[half][2 * j], dv_vecs[half][2 * j + 1], decay_scale_vals[2 * j], decay_scale_vals[2 * j + 1])
                        dv_scaled += [s0, s1]
                    nvvm.tcgen05_st("16x256b", dv_ptrs[half], cutlass.Vector.from_elements(tuple(dv_scaled), cutlass.Float32))
                nvvm.tcgen05_wait("store")
                bars.mb_du_scale_acc_done[0].arrive()

            # ---- dU stage: dV acc -> TMEM f16 ----------------------------------------
            bars.mb_du_total_acc_ready[0].wait(cg1_du_total_ready.phase)
            cg1_du_total_ready = advance(cg1_du_total_ready, 1)
            for half in cutlass.range_constexpr(dv_halves):
                du_vec = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_dvdk_acc_col, cutlass.Float32), num=8)
                du_pack = [fp32_to_fp16(du_vec[2 * j], du_vec[2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_du_col, cutlass.Int32),
                    cutlass.Vector.from_elements(tuple(du_pack), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_du_input_ready[0].arrive()
            bars.mb_du_total_acc_done[0].arrive()

            # ---- dY ------------------------------------------------------------------
            bars.mb_dy_acc_ready[0].wait(cg1_dy_ready.phase)
            cg1_dy_ready = advance(cg1_dy_ready, 1)
            dy_regs = []
            for half in cutlass.range_constexpr(dv_halves):
                dy_vec = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_dy_col, cutlass.Float32),
                    num=8,
                )
                dy_regs.append([dy_vec[k] for k in range(32)])

            # ---- dY' = -cumprod * dY -> f16 shared input -----------------------------
            neg_one = cutlass.Float32(-1.0)
            cumprod_neg_vals = [cumprod_vals[k] * neg_one for k in range(32)]
            for half in cutlass.range_constexpr(dv_halves):
                dyp = []
                for j in cutlass.range_constexpr(16):
                    n0, n1 = fmul2(dy_regs[half][2 * j], dy_regs[half][2 * j + 1], cumprod_neg_vals[2 * j], cumprod_neg_vals[2 * j + 1])
                    dyp += [n0, n1]
                dyp_pack = [fp32_to_fp16(dyp[2 * j], dyp[2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_dyp_col, cutlass.Int32),
                    cutlass.Vector.from_elements(tuple(dyp_pack), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_dyp_input_ready[0].arrive()

            # ---- Gate stage release --------------------------------------------------
            bars.mb_gate_done[gate_idx].arrive()

        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    bars.mb_tmem_done[0].arrive()


@cute.jit
def compute2_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    mWorkItems,
    mDstate0_out,
    mDstate_in,
    tidx,
    tmem_base_slot,
    sCumprod,
    sScheduler,
    bars,
):
    """Compute warp-group 2 role (warps 8-11): d_final_state seeding, the per-chunk f16 pack of the dstate accumulator into
    the input columns with its rescale by the next chunk's total decay, and the d_initial_state store + zero-chunk passthrough."""

    nvvm.setmaxregister(cfg.num_regs_compute_group_2, nvvm.SetMaxRegisterAction.DECREASE)
    elect_one = nvvm.elect_sync()

    dstate_acc_index = PipelineState.start(phase=0)
    dstate_input_index = PipelineState.start(phase=1)
    gate_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)

    num_threads_cg2 = cfg.threads_per_warp * len(cfg.compute_group_2_warp_ids)
    cg2_tidx = tidx % num_threads_cg2

    if cutlass.const_expr(cfg.d_v == 128):
        dstate_gmem_row = cg2_tidx
        dstate_row_valid = cutlass.Boolean(True)
    else:
        dstate_gmem_row = (cg2_tidx // 32) * 16 + (cg2_tidx % 32) % 16
        dstate_row_valid = (cg2_tidx % 32) < 16
    ldtm_width = 32
    sttm_width = 16
    num_ldtms = cutlass.const_expr(cfg.d_k // ldtm_width)
    tile_idx = cutlass.Int32(bidx)

    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    (tmem_row + 16) << 16
    tmem_dstate_acc_col = tmem_col + cfg.tmem_dstate_acc_offset
    tmem_dstate_input_col = tmem_col + cfg.tmem_dstate_input_offset

    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_item_chunks = compute_end - write_start

        # ---- first chunk: gate stage, dstate seed ------------------------------------
        if num_item_chunks > 0:
            gate_idx = gate_index.idx
            bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)
            if cutlass.const_expr(cfg.use_dstate_in):
                cumprod_top = sCumprod[sCumprod.shape[0] - 1, 0, gate_idx]
                gDstate_in = mDstate_in[None, None, head_idx, batch_idx]
                seed_from_dstate_in = compute_end == batch_num_chunks
                dstate_src = (gDstate_in.iterator + gDstate_in.layout((dstate_gmem_row, 0))).raw_ptr()
                seed_vw = cutlass.const_expr(16 // (mDstate_in.element_type.width // 8))
                dstate_input_idx = dstate_input_index.idx
                bars.mb_dstate_input_done[dstate_input_idx].wait(dstate_input_index.phase)
                dstate_input_index = advance(dstate_input_index, cfg.tmem_dstate_input_stages)
                for i in cutlass.range_constexpr(cfg.d_k // 16):
                    seed_block = cutlass.Array(cutlass.Float32, 16, alignment=16)
                    for t in cutlass.range_constexpr(16):
                        seed_block[t] = cutlass.Float32(0.0)
                    if seed_from_dstate_in:
                        for g in cutlass.range_constexpr(16 // seed_vw):
                            seed_chunk = (dstate_src + i * 16 + g * seed_vw).load(count=seed_vw, alignment=16)
                            for t in cutlass.range_constexpr(seed_vw):
                                seed_block[g * seed_vw + t] = seed_chunk[t].to(cutlass.Float32)
                    seed_pack = cutlass.Array(cutlass.Int32, 8, alignment=16)
                    for pc in cutlass.range_constexpr(8):
                        seed_pack[pc] = fp32_to_fp16(seed_block[2 * pc], seed_block[2 * pc + 1], dtype=cfg.io_dtype)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_input_col + i * 8, cutlass.Int32),
                        seed_pack[0:8],
                    )
                    seed_scaled = cutlass.Array(cutlass.Float32, 16, alignment=16)
                    for pc in cutlass.range_constexpr(8):
                        seed_scaled[2 * pc], seed_scaled[2 * pc + 1] = fmul2(seed_block[2 * pc], seed_block[2 * pc + 1], cumprod_top, cumprod_top)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_acc_col + i * 16, cutlass.Float32),
                        seed_scaled[0:16],
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_input_ready[dstate_input_idx].arrive()
                bars.mb_dstate_scale_acc_done[dstate_acc_index.idx].arrive()
            bars.mb_gate_done[gate_idx].arrive()

        for rev_idx in cutlass.range(num_item_chunks):
            abs_chunk = compute_end - 1 - rev_idx

            # ---- NEXT-CHUNK dstate prep: f16 pack + total-decay rescale --------------
            if abs_chunk >= write_start + 1:
                dstate_idx = dstate_acc_index.idx
                bars.mb_dstate_acc_ready[dstate_idx].wait(dstate_acc_index.phase)
                dstate_acc_index = advance(dstate_acc_index, cfg.tmem_dstate_acc_stages)
                dstate_input_idx = dstate_input_index.idx
                bars.mb_dstate_input_done[dstate_input_idx].wait(dstate_input_index.phase)
                dstate_input_index = advance(dstate_input_index, cfg.tmem_dstate_input_stages)
                gate_idx = gate_index.idx
                bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
                gate_index = advance(gate_index, cfg.smem_gate_stages)
                cumprod_top = sCumprod[sCumprod.shape[0] - 1, 0, gate_idx]
                for i in cutlass.range_constexpr(num_ldtms):
                    dstate_vec = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_acc_col + i * ldtm_width, cutlass.Float32), num=32)
                    dstate_pack = [fp32_to_fp16(dstate_vec[2 * j], dstate_vec[2 * j + 1], dtype=cfg.io_dtype) for j in range(16)]
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_input_col + i * sttm_width, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(dstate_pack), cutlass.Int32),
                    )
                    dstate_rescaled = []
                    for j in cutlass.range_constexpr(16):
                        h0, h1 = fmul2(dstate_vec[2 * j], dstate_vec[2 * j + 1], cumprod_top, cumprod_top)
                        dstate_rescaled += [h0, h1]
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_acc_col + i * ldtm_width, cutlass.Float32),
                        cutlass.Vector.from_elements(tuple(dstate_rescaled), cutlass.Float32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_input_ready[dstate_input_idx].arrive()
                bars.mb_dstate_scale_acc_done[dstate_idx].arrive()
                bars.mb_gate_done[gate_idx].arrive()

        # ---- dstate store ------------------------------------------------------------
        if num_item_chunks > 0:
            dstate_idx = dstate_acc_index.idx
            bars.mb_dstate_acc_ready[dstate_idx].wait(dstate_acc_index.phase)
            dstate_acc_index = advance(dstate_acc_index, cfg.tmem_dstate_acc_stages)
            if write_start == 0:
                gDstate0 = mDstate0_out[None, None, head_idx, batch_idx]
                dstate0_vw = 16 // (mDstate0_out.element_type.width // 8)
                dstate0_dst = (gDstate0.iterator + gDstate0.layout((dstate_gmem_row, 0))).raw_ptr()
                for i in cutlass.range_constexpr(num_ldtms):
                    dstate0_vec = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_dstate_acc_col + i * ldtm_width, cutlass.Float32), num=32)
                    if dstate_row_valid:
                        for g in cutlass.range_constexpr(ldtm_width // dstate0_vw):
                            (dstate0_dst + i * ldtm_width + g * dstate0_vw).store(
                                cutlass.Vector.from_elements(
                                    tuple(dstate0_vec[g * dstate0_vw + t].to(mDstate0_out.element_type) for t in range(dstate0_vw)), mDstate0_out.element_type
                                ),
                                alignment=16,
                            )
            if cutlass.const_expr(not cfg.use_dstate_in):
                bars.mb_dstate_scale_acc_done[dstate_idx].arrive()
        else:
            write_passthrough = write_start == 0
            if write_passthrough:
                gDstate0 = mDstate0_out[None, None, head_idx, batch_idx]
                if cutlass.const_expr(cfg.use_dstate_in):
                    gDstate_in = mDstate_in[None, None, head_idx, batch_idx]
                    for i in cutlass.range_constexpr(num_ldtms):
                        for kk in cutlass.range_constexpr(32):
                            if dstate_row_valid:
                                gDstate0[dstate_gmem_row, i * ldtm_width + kk] = gDstate_in[dstate_gmem_row, i * ldtm_width + kk]
                else:
                    for i in cutlass.range_constexpr(num_ldtms):
                        for kk in cutlass.range_constexpr(32):
                            if dstate_row_valid:
                                gDstate0[dstate_gmem_row, i * ldtm_width + kk] = cutlass.Float32(0.0).to(mDstate0_out.element_type)

        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    bars.mb_tmem_done[0].arrive()

    for _ in range(cfg.tmem_dstate_input_stages):
        bars.mb_dstate_input_done[dstate_input_index.idx].wait(dstate_input_index.phase)
        dstate_input_index = advance(dstate_input_index, cfg.tmem_dstate_input_stages)


@cute.jit
def build_descs_body(
    widx,
    base_q,
    base_k,
    base_do,
    base_tinv,
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    do_: cute.Tensor,
    tinv: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    q_step: cutlass.Constexpr[int] = 1,
) -> None:
    """Per-batch descriptor-array build, one warp per array (the tinv array on the K warp), after the prologue's order pass.
    q and dO address the compact
    token timeline (``expand_num`` scales k alone); ``q_step`` > 1 reads the phase rows of an expanded q buffer."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    sub0 = cute.make_tensor(desc_workspace.iterator, cute.make_layout((arr_words,), stride=(1,)))
    sub1 = cute.make_tensor(desc_workspace.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub2 = cute.make_tensor(desc_workspace.iterator + 2 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub3 = cute.make_tensor(desc_workspace.iterator + 3 * arr_words, cute.make_layout((arr_words,), stride=(1,)))

    if widx == 0:
        if cutlass.const_expr(q_step > 1):
            q_phase = cute.make_tensor(
                q.iterator + cutlass.Int32(q_step - 1) * q.stride[0],
                cute.make_layout((q.shape[0] // cutlass.Int32(q_step), q.shape[1], q.shape[2]), stride=(q.stride[0] * cutlass.Int32(q_step), q.stride[1], 1)),
            )
            emit_seq_descs(base_q, sub0, cu_seqlens, q_phase, n_batch, 2, 1, lanes=32)
        else:
            emit_seq_descs(base_q, sub0, cu_seqlens, q, n_batch, 2, 1, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 1:
        emit_seq_descs(base_k, sub1, cu_seqlens, k, n_batch, 2, expand_num, lanes=32)
        if cutlass.const_expr(tinv is not None):
            emit_tile_seq_descs(base_tinv, sub3, cu_seqlens, tinv, n_batch, b_t, 3, expand_num, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 2:
        emit_seq_descs(base_do, sub2, cu_seqlens, do_, n_batch, 2, 1, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


@cute.kernel
def frost_gdn_bprop_summary_prologue(
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    q_step: cutlass.Constexpr[int],
    base_q: cutlass.GridConstant[tma.TensorMap],
    base_k: cutlass.GridConstant[tma.TensorMap],
    base_do: cutlass.GridConstant[tma.TensorMap],
    base_tinv: cutlass.GridConstant[tma.TensorMap],
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    do_: cute.Tensor,
    tinv: Optional[cute.Tensor],
    mStaging: Optional[cute.Tensor],
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mScheduler: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
) -> None:
    """Two-CTA prologue: under ``run_order`` (this kernel is the first work-item-table consumer) block 0 LPT-orders the
    table and zeroes both consumers' scheduler rings (:func:`order_body`); block 1 builds the per-batch TMA-descriptor
    arrays (:func:`build_descs_body`)."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
        launch_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    if bidx == cutlass.Int32(0):
        if cutlass.const_expr(run_order):
            sKey = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
            sIdx = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
            sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
            n_heads_out = cutlass.Int32(do_.shape[1])
            order_body(
                order_gen,
                b_t,
                ORDER_THREADS,
                ORDER_ELEMENTS,
                tidx,
                n_heads_out,
                n_heads_out * n_batch,
                cu_seqlens,
                mStaging,
                mCount,
                mWorkItems,
                mScheduler,
                sKey,
                sIdx,
                sSpread,
                expand_num=expand_num,
            )
    else:
        build_descs_body(
            widx,
            base_q,
            base_k,
            base_do,
            base_tinv,
            desc_workspace,
            cu_seqlens,
            q,
            k,
            do_,
            tinv,
            n_batch,
            b_t,
            expand_num,
            q_step,
        )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    expand_num: cutlass.Constexpr[int],
    q_step: cutlass.Constexpr[int],
    q: cute.Tensor,
    k: cute.Tensor,
    do_: cute.Tensor,
    cu_seqlens: cute.Tensor,
    work_item_staging: Optional[cute.Tensor],
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler_all: Optional[cute.Tensor],
    tinv: Optional[cute.Tensor],
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
):
    """One-launch prologue: LPT-order the work items (``run_order``, when this kernel is the backward pair's first table
    consumer) and build the per-(b,h) TMA-descriptor arrays (Q, K, dO) into ``tensormap_workspace``."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    h_v = do_.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = h_q if h_q >= h_v else h_v
    d_v = do_.shape[2]
    bytes_per_element = io_dtype.width // 8
    box_elems = 128 // bytes_per_element
    bt = b_t

    q_row_stride, q_head_stride = q.stride[0], q.stride[1]
    k_row_stride, k_head_stride = k.stride[0], k.stride[1]

    d_k = q.shape[2]
    if cutlass.const_expr(q_step > 1):
        q_headed = cute.make_tensor(
            q.iterator + cutlass.Int32(q_step - 1) * q_row_stride,
            cute.make_layout((q.shape[0] // cutlass.Int32(q_step), h_q, d_k), stride=(q_row_stride * cutlass.Int32(q_step), q_head_stride, 1)),
        )
    else:
        q_headed = cute.make_tensor(q.iterator, cute.make_layout((q.shape[0], h_q, d_k), stride=(q_row_stride, q_head_stride, 1)))
    k_headed = cute.make_tensor(k.iterator, cute.make_layout((k.shape[0], h_k, d_k), stride=(k_row_stride, k_head_stride, 1)))

    do_headed = cute.make_tensor(do_.iterator, cute.make_layout((d_v, heads_out, do_.shape[0]), stride=(1, do_.stride[1], do_.stride[0])))
    swizzle_128b = tma.TensorMapSwizzle.s128b
    base_desc_q = tma.create_tensor_map_tiled_from_view(q_headed, box_dims=(bt, 1, box_elems), stride_order=(2, 1, 0), swizzle=swizzle_128b)
    base_desc_k = tma.create_tensor_map_tiled_from_view(k_headed, box_dims=(bt, 1, box_elems), stride_order=(2, 1, 0), swizzle=swizzle_128b)
    base_desc_do = tma.create_tensor_map_tiled_from_view(do_headed, box_dims=(box_elems, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)

    base_desc_tinv = base_desc_k
    if cutlass.const_expr(tinv is not None):
        tinv_tiles = cute.make_tensor(
            tinv.iterator,
            cute.make_layout((tinv.shape[0], tinv.shape[1], tinv.shape[2], tinv.shape[3]), stride=(tinv.stride[0], tinv.stride[1], tinv.stride[2], 1)),
        )
        base_desc_tinv = tma.create_tensor_map_tiled_from_view(tinv_tiles, box_dims=(1, 1, bt, bt), stride_order=(3, 2, 1, 0), swizzle=swizzle_128b)
    frost_gdn_bprop_summary_prologue(
        run_order,
        order_gen,
        b_t,
        expand_num,
        q_step,
        base_desc_q,
        base_desc_k,
        base_desc_do,
        base_desc_tinv,
        tensormap_workspace,
        cu_seqlens,
        q,
        k,
        do_,
        tinv,
        work_item_staging,
        work_count,
        work_items,
        scheduler_all,
        cutlass.Int32(batch_size),
    ).launch(grid=(2, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    q: cute.Tensor,
    k: cute.Tensor,
    gate: cute.Tensor,
    a_log: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    do_: cute.Tensor,
    cu_seqlens: cute.Tensor,
    dstate0: cute.Tensor,
    dstate_in: Optional[cute.Tensor],
    tinv: cute.Tensor,
    work_items: Optional[cute.Tensor],
    work_count: Optional[cute.Tensor],
    scheduler_counter: cute.Tensor,
    scale: cutlass.Float32,
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
):
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = cutlass.Int32(gate.shape[1])
    q_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(q.shape[1]))
    k_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(k.shape[1]))

    # ---- SMEM sizing: per-buffer element cosizes -------------------------------------
    bytes_per_element = cfg.io_dtype.width // 8
    q_tile_elements = cfg.b_t * cfg.d_k
    k_tile_elements = cfg.b_t * cfg.d_k
    do_tile_elements = cfg.d_v * cfg.b_t
    tinv_tile_elements = cfg.b_t * cfg.b_t
    a_tile_elements = cfg.b_t * cfg.b_t
    cfg.q_cosize = q_tile_elements * cfg.smem_q_stages
    cfg.k_cosize = k_tile_elements * cfg.smem_k_stages
    cfg.do_cosize = do_tile_elements * cfg.smem_do_stages
    cfg.t_inv_cosize = tinv_tile_elements * cfg.smem_t_inv_stages
    cfg.a_cosize = a_tile_elements * cfg.smem_a_stages

    cumsumlog_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_gate_stages))

    cfg.tma_q_bytes = q_tile_elements * bytes_per_element
    cfg.tma_k_bytes = k_tile_elements * bytes_per_element
    cfg.tma_do_bytes = do_tile_elements * bytes_per_element
    cfg.tma_tinv_bytes = tinv_tile_elements * bytes_per_element

    num_descs = batch_size

    # ---- launch ----------------------------------------------------------------------
    grid_shape = (cfg.max_active_clusters, 1, 1)

    frost_gdn_bprop_summary(
        cfg,
        q_ratio,
        k_ratio,
        gate,
        a_log,
        dt_bias,
        cu_seqlens,
        scale,
        cumsumlog_smem_layout_staged,
        q,
        k,
        do_,
        dstate0,
        dstate_in,
        tinv,
        work_items,
        work_count,
        scheduler_counter,
        tensormap_workspace,
        cutlass.Int32(num_descs),
    ).launch(
        grid=grid_shape,
        block=(cfg.threads_per_cta, 1, 1),
        cluster=cfg.cluster_shape_mnk,
        stream=stream,
        use_pdl=USE_PDL,
        min_blocks_per_mp=1,
    )


@cute.kernel
def frost_gdn_bprop_summary(
    cfg: cutlass.Constexpr,
    q_ratio: cute.FastDivmodDivisorV2,
    k_ratio: cute.FastDivmodDivisorV2,
    mGate: cute.Tensor,
    mA_log: Optional[cute.Tensor],
    mDt_bias: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    scale: cutlass.Float32,
    cumsumlog_smem_layout_staged: cute.Layout,
    mQ,
    mK,
    mdO,
    mDstate0,
    mDstate_in,
    mTinv: cute.Tensor,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
):
    """Main GDN bprop state-summary chunked kernel (warp-specialized
    persistent body)."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]

    total_tiles = mCount[0]
    if cutlass.const_expr(mDstate0 is not None):
        mDstate0 = cute.make_tensor(
            mDstate0.iterator,
            cute.make_layout(
                (mDstate0.shape[2], mDstate0.shape[3], mDstate0.shape[1], mDstate0.shape[0]),
                stride=(mDstate0.stride[2], mDstate0.stride[3], mDstate0.stride[1], mDstate0.stride[0]),
            ),
        )
    if cutlass.const_expr(mDstate_in is not None):
        mDstate_in = cute.make_tensor(
            mDstate_in.iterator,
            cute.make_layout(
                (mDstate_in.shape[2], mDstate_in.shape[3], mDstate_in.shape[1], mDstate_in.shape[0]),
                stride=(mDstate_in.stride[2], mDstate_in.stride[3], mDstate_in.stride[1], mDstate_in.stride[0]),
            ),
        )

    desc_base_words = tensormap_workspace.iterator.raw_ptr()
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)
    arr_words = n_desc * desc_qwords
    desc_q_base = desc_base_words
    desc_k_base = desc_base_words + arr_words
    desc_do_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_tinv_base = desc_base_words + cutlass.Int32(3) * arr_words

    SMEM = cutlass.AddressSpace.smem
    bars = make_bars(cfg)
    sScheduler = cutlass.Array(cutlass.Int32, cfg.scheduler_stages, space=cutlass.AddressSpace.smem, alignment=16)
    tmem_base_slot = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=16)
    cumsumlog_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
    cumprod_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
    sTokSlot = None
    sTokCumsum = None
    sTokCumprod = None
    if cutlass.const_expr(cfg.expand_num > 1):
        tok_slot_raw = cutlass.Array(cutlass.Int32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
        tok_cumsum_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
        tok_cumprod_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
        sTokSlot = cute.make_tensor(
            cute.make_ptr(cutlass.Int32, tok_slot_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=128), cumsumlog_smem_layout_staged
        )
        sTokCumsum = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, tok_cumsum_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=128),
            cumsumlog_smem_layout_staged,
        )
        sTokCumprod = cute.make_tensor(
            cute.make_ptr(cutlass.Float32, tok_cumprod_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=128),
            cumsumlog_smem_layout_staged,
        )

    SWZ = 2
    LEAD = 16
    STRIDE = 8 * 128
    KT_LEAD = cfg.b_t * 128
    V_LEAD = cfg.b_t * 128
    sQ_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.q_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sQ = SmemTile(
        base=sQ_raw.data_ptr(),
        elems_per_stage=(cfg.q_cosize // cfg.smem_q_stages),
        stages=cfg.smem_q_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
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
    sK_trans = SmemTile(
        base=sK_raw.data_ptr(),
        elems_per_stage=(cfg.k_cosize // cfg.smem_k_stages),
        stages=cfg.smem_k_stages,
        leading_byte_offset=KT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sQ_trans = SmemTile(
        base=sQ_raw.data_ptr(),
        elems_per_stage=(cfg.q_cosize // cfg.smem_q_stages),
        stages=cfg.smem_q_stages,
        leading_byte_offset=KT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdO_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.do_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sdO_trans = SmemTile(
        base=sdO_raw.data_ptr(),
        elems_per_stage=(cfg.do_cosize // cfg.smem_do_stages),
        stages=cfg.smem_do_stages,
        leading_byte_offset=V_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdO = SmemTile(
        base=sdO_raw.data_ptr(),
        elems_per_stage=(cfg.do_cosize // cfg.smem_do_stages),
        stages=cfg.smem_do_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sTinv_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.t_inv_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sTinv_trans = SmemTile(
        base=sTinv_raw.data_ptr(),
        elems_per_stage=(cfg.t_inv_cosize // cfg.smem_t_inv_stages),
        stages=cfg.smem_t_inv_stages,
        leading_byte_offset=(cfg.b_t // 2) * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sA_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.a_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sA = SmemTile(
        base=sA_raw.data_ptr(),
        elems_per_stage=(cfg.a_cosize // cfg.smem_a_stages),
        stages=cfg.smem_a_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sA_trans = SmemTile(
        base=sA_raw.data_ptr(),
        elems_per_stage=(cfg.a_cosize // cfg.smem_a_stages),
        stages=cfg.smem_a_stages,
        leading_byte_offset=(cfg.b_t // 2) * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sCumsumlog = cute.make_tensor(
        cute.make_ptr(cutlass.Float32, cumsumlog_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=128),
        cumsumlog_smem_layout_staged,
    )
    sCumprod = cute.make_tensor(
        cute.make_ptr(cutlass.Float32, cumprod_raw.data_ptr().toint(), mem_space=cute.AddressSpace.smem, assumed_align=128),
        cumsumlog_smem_layout_staged,
    )

    # ---- mbarrier init (all threads) -------------------------------------------------
    for s in range(cfg.scheduler_stages):
        bars.mb_scheduler_ready[s].init()
        bars.mb_scheduler_done[s].init()
    for s in range(cfg.smem_q_stages):
        bars.mb_q_ready[s].init()
        bars.mb_q_mma_done[s].init()
    for s in range(cfg.smem_k_stages):
        bars.mb_k_ready[s].init()
        bars.mb_k_mma_done[s].init()
    for s in range(cfg.smem_do_stages):
        bars.mb_do_ready[s].init()
        bars.mb_do_mma_done[s].init()
        bars.mb_do_cg0_done[s].init()
    for s in range(cfg.smem_gate_stages):
        bars.mb_gate_ready[s].init()
        bars.mb_gate_done[s].init()
    for s in range(cfg.tmem_dstate_acc_stages):
        bars.mb_dstate_acc_ready[s].init()
        bars.mb_dstate_scale_acc_done[s].init()
    for b in (
        bars.mb_du_scale_acc_ready,
        bars.mb_du_scale_acc_done,
        bars.mb_du_total_acc_ready,
        bars.mb_du_total_acc_done,
    ):
        b[0].init()
    for b in (
        bars.mb_a_acc_ready,
        bars.mb_dy_acc_ready,
    ):
        b[0].init()
    for s in range(cfg.smem_t_inv_stages):
        bars.mb_t_inv_ready[s].init()
        bars.mb_t_inv_done[s].init()
    for s in range(cfg.smem_a_stages):
        bars.mb_a_ready[s].init()
        bars.mb_a_done[s].init()
    for s in range(cfg.tmem_dstate_input_stages):
        bars.mb_dstate_input_ready[s].init()
        bars.mb_dstate_input_done[s].init()
    for b in (
        bars.mb_do_prime_input_ready,
        bars.mb_du_input_ready,
        bars.mb_dyp_input_ready,
    ):
        b[0].init()
    bars.mb_tmem_done[0].init()

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # ---- warp specialization ---------------------------------------------------------
    if warp_idx >= cfg.compute_group_0_warp_ids[0] and warp_idx <= cfg.compute_group_0_warp_ids[-1]:
        compute0_warp_group(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tidx,
            tmem_base_slot=tmem_base_slot,
            scale=scale,
            sdO=sdO,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sTokSlot=sTokSlot,
            sTokCumsum=sTokCumsum,
            sTokCumprod=sTokCumprod,
            sA=sA,
            sScheduler=sScheduler,
            bars=bars,
        )

    if warp_idx >= cfg.compute_group_1_warp_ids[0] and warp_idx <= cfg.compute_group_1_warp_ids[-1]:
        compute1_warp_group(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tidx,
            warp_idx=warp_idx,
            tmem_base_slot=tmem_base_slot,
            scale=scale,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sScheduler=sScheduler,
            bars=bars,
        )

    if warp_idx >= cfg.compute_group_2_warp_ids[0] and warp_idx <= cfg.compute_group_2_warp_ids[-1]:
        compute2_warp_group(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            mWorkItems,
            mDstate0,
            mDstate_in,
            tidx,
            tmem_base_slot=tmem_base_slot,
            sCumprod=sCumprod,
            sScheduler=sScheduler,
            bars=bars,
        )

    elif warp_idx == cfg.tcgen05_mma_warp_id:
        tcgen05_mma_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tmem_base_slot=tmem_base_slot,
            sQ=sQ,
            sQ_trans=sQ_trans,
            sK=sK,
            sK_trans=sK_trans,
            sdO_trans=sdO_trans,
            sTinv_trans=sTinv_trans,
            sA=sA,
            sA_trans=sA_trans,
            sScheduler=sScheduler,
            bars=bars,
        )

    elif warp_idx == cfg.tma_qkv_warp_id:
        tmaldg_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sQ_raw=sQ_raw,
            sK_raw=sK_raw,
            sdO_raw=sdO_raw,
            sTinv_raw=sTinv_raw,
            desc_tinv_base=desc_tinv_base,
            desc_q_base=desc_q_base,
            desc_k_base=desc_k_base,
            desc_do_base=desc_do_base,
            mScheduler=mScheduler,
            sScheduler=sScheduler,
            bars=bars,
            q_ratio=q_ratio,
            k_ratio=k_ratio,
        )

    if warp_idx == cfg.load_gate_beta_warp_id:
        gate_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tidx,
            mGate=mGate,
            mA_log=mA_log,
            mDt_bias=mDt_bias,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sTokSlot=sTokSlot,
            sTokCumsum=sTokCumsum,
            sTokCumprod=sTokCumprod,
            sScheduler=sScheduler,
            bars=bars,
        )
    if warp_idx == cfg.epilogue_warp_id:
        tmastg_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            sScheduler=sScheduler,
            bars=bars,
        )


@dataclass
class GdnBpropSummaryCfg:
    """Per-compile bprop state-summary kernel knob (``build_cfg``): the dtype / GQA fields are the ``cute.compile`` cache
    keys, the rest derives from ``CFG``; ``host`` stamps the shape-derived fields at trace time.
    """

    use_dstate_in: bool
    io_dtype: Type[cutlass.Numeric]
    acc_dtype: Type[cutlass.Numeric]
    max_active_clusters: int
    d_k: int
    d_v: int
    log_gate: bool = False
    safe_gate: bool = False

    # ---- fixed constants stamped from CFG at build time ------------------------------
    b_t: int = CFG.B_T
    expand_num: int = 1
    compute_group_0_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_1_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    compute_group_2_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_2_WARP_IDS
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    tma_qkv_warp_id: int = CFG.TMA_QKV_WARP_ID
    load_gate_beta_warp_id: int = CFG.LOAD_GATE_BETA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_compute_group_2: int = CFG.NUM_REGS_COMPUTE_GROUP_2
    num_regs_other: int = CFG.NUM_REGS_OTHER
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0
    cluster_shape_mnk: Tuple[int, int, int] = CFG.CLUSTER_SHAPE_MNK
    scheduler_stages: int = CFG.SMEM_SCHEDULER_STAGES

    # ---- named barrier slots (ids 1-6; 0 is the CTA-wide sync) -----------------------
    tmem_lifecycle_barrier_id: int = 1
    tmem_user_threads: int = 0
    init_state_store_barrier_threads: int = 0
    cg1_barrier_id: int = 5
    cg1_barrier_threads: int = 0
    cg2_barrier_id: int = 6
    cg2_barrier_threads: int = 0

    # ---- SMEM / TMEM stage counts + TMEM column offsets ------------------------------
    smem_q_stages: int = CFG.SMEM_Q_STAGES
    smem_k_stages: int = CFG.SMEM_K_STAGES
    smem_do_stages: int = CFG.SMEM_DO_STAGES
    smem_t_inv_stages: int = CFG.SMEM_T_INV_STAGES
    smem_a_stages: int = CFG.SMEM_A_STAGES
    smem_gate_stages: int = CFG.SMEM_GATE_STAGES
    tmem_dstate_acc_stages: int = CFG.TMEM_DH_ACC_STAGES
    tmem_dvdk_acc_stages: int = CFG.TMEM_DVDK_ACC_STAGES
    tmem_dstate_input_stages: int = CFG.TMEM_DH_INP_STAGES
    tmem_shared_input_stages: int = CFG.TMEM_SHARED_INP_STAGES
    tmem_shared_acc_stages: int = CFG.TMEM_SHARED_ACC_STAGES
    tmem_dstate_acc_offset: int = 0
    tmem_dvdk_acc_offset: int = 0
    tmem_dstate_input_offset: int = 0
    tmem_shared_acc_offset: int = 0
    tmem_shared_input_offset: int = 0
    tmem_y_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- stamped by host at trace time (shape-derived) -------------------------------
    q_cosize: int = 0
    k_cosize: int = 0
    do_cosize: int = 0
    t_inv_cosize: int = 0
    a_cosize: int = 0
    tma_q_bytes: int = 0
    tma_k_bytes: int = 0
    tma_do_bytes: int = 0
    tma_tinv_bytes: int = 0


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    *,
    max_active_clusters: int,
    use_dstate_in: bool = False,
    log_gate: bool = False,
    safe_gate: bool = False,
    d_k: int,
    d_v: int,
    expand_num: int = 1,
) -> GdnBpropSummaryCfg:
    """Build the per-compile ``GdnBpropSummaryCfg`` (io_dtype in {Float16, BFloat16}; acc is always Float32)."""
    if expand_num > 1 and (CFG.SMEM_Q_STAGES != 2 or CFG.SMEM_DO_STAGES != 2):
        raise ValueError("the compact q / dO block state assumes 2-stage q and dO rings")
    cfg = GdnBpropSummaryCfg(
        use_dstate_in=use_dstate_in,
        io_dtype=io_dtype,
        acc_dtype=cutlass.Float32,
        max_active_clusters=max_active_clusters,
        log_gate=log_gate,
        safe_gate=safe_gate,
        d_k=d_k,
        d_v=d_v,
        expand_num=expand_num,
    )
    n_cg0 = len(cfg.compute_group_0_warp_ids)
    n_cg1 = len(cfg.compute_group_1_warp_ids)
    n_cg2 = len(cfg.compute_group_2_warp_ids)
    cfg.threads_per_cta = cfg.threads_per_warp * (4 + n_cg0 + n_cg1 + n_cg2)
    cfg.tmem_user_threads = cfg.threads_per_warp * (1 + n_cg0 + n_cg1 + n_cg2)
    cfg.init_state_store_barrier_threads = cfg.threads_per_warp * n_cg1
    cfg.cg1_barrier_threads = cfg.threads_per_warp * n_cg1
    cfg.cg2_barrier_threads = cfg.threads_per_warp * n_cg2
    cfg.tmem_dstate_acc_offset = 0
    cfg.tmem_dvdk_acc_offset = cfg.tmem_dstate_acc_offset + cfg.tmem_dstate_acc_stages * cfg.d_k
    cfg.tmem_dstate_input_offset = cfg.tmem_dvdk_acc_offset + cfg.tmem_dvdk_acc_stages * cfg.b_t
    cfg.tmem_shared_acc_offset = cfg.tmem_dstate_input_offset + cfg.tmem_dstate_input_stages * max(cfg.d_k // 2, cfg.b_t)
    cfg.tmem_shared_input_offset = cfg.tmem_shared_acc_offset + cfg.tmem_shared_acc_stages * cfg.b_t
    cfg.tmem_y_offset = cfg.tmem_shared_input_offset + cfg.tmem_shared_input_stages * (cfg.b_t // 2)
    return cfg


TENSORMAP_DESC_ARRAYS = 4  # per-batch runtime TMA descriptors: Q, K, dO, tinv


# ---------------------------------------------------------------------------


@functools.cache
def get_compiled_cache(
    io_dtype_str: str,
    cu_dtype_str: str,
    gate_dtype_str: str,
    a_log_dtype_str: str,
    dt_bias_dtype_str: str,
    dstate_in_dtype_str: str,
    dstate0_dtype_str: str,
    device: int,
    num_sm: int,
    DK: int,
    DV: int,
    expand_num: int,
    use_dstate_in: bool = False,
    log_gate: bool = False,
    safe_gate: bool = False,
    run_order: bool = False,
    order_gen: bool = False,
    q_step: int = 1,
):
    """Return a mutable dict that lazily stores the compiled kernel."""
    return {}


def compile(
    io_dtype,
    use_dstate_in: bool = False,
    log_gate: bool = False,
    safe_gate: bool = False,
    *,
    num_sm: int,
    d_k: int,
    d_v: int,
    expand_num: int = 1,
    q_cute,
    k_cute,
    gate_cute,
    a_log_cute=None,
    dt_bias_cute=None,
    do_cute,
    cu_seqlens_cute,
    dstate0_cute,
    dstate_in_cute=None,
    tinv_cute,
    work_items_cute=None,
    work_count_cute=None,
    scheduler_counter_cute=None,
    scale=None,
    workspace_cute=None,
    stream=None,
):
    """JIT-compile the chunked GDN bprop state-summary kernel for one static
    config."""
    cfg = build_cfg(
        io_dtype,
        max_active_clusters=num_sm,
        use_dstate_in=use_dstate_in,
        log_gate=log_gate,
        safe_gate=safe_gate,
        d_k=d_k,
        d_v=d_v,
        expand_num=expand_num,
    )

    return cute.compile(
        host,
        cfg,
        q_cute,
        k_cute,
        gate_cute,
        a_log_cute,
        dt_bias_cute,
        do_cute,
        cu_seqlens_cute,
        dstate0_cute,
        dstate_in_cute,
        tinv_cute,
        work_items_cute,
        work_count_cute,
        scheduler_counter_cute,
        scale,
        workspace_cute,
        stream,
        options="--enable-tvm-ffi --opt-level 2",
    )


def chunk_gdn_bwd_summary(
    q,
    k,
    gate,
    do,
    d_initial_state,
    cu_seqlens,
    scale: float,
    *,
    d_final_state=None,
    work_items=None,
    work_count=None,
    scheduler_counter=None,
    scheduler_all=None,
    work_item_scratch=None,
    order_in_prologue: bool = False,
    log_gate: bool = False,
    safe_gate: bool = False,
    a_log=None,
    dt_bias=None,
    tinv,
    expand_num: int = 1,
    q_step: int = 1,
    workspace,
    device: int,
    num_sm: int,
    stream,
    own_prologue: bool = True,
) -> None:
    """Execute the chunked GDN bprop state-summary kernel (THD / varlen entry), compiled once per static config and
    replayed; tensors are DLPack CUDA tensors with a stride-1 innermost dim.  ``d_initial_state`` is the only output.
    gate: raw linear alpha, natural-log decay under ``log_gate``, or raw logits under ``safe_gate`` (``-exp(a_log) * softplus(gate + dt_bias)``)
    work_items / work_count: ``common/split_k.py`` table ``(max_items, 8)`` int32 and ``(1,)`` count (REQUIRED); an item
        computes chunks ``[write_start, compute_end)`` backward and only the item owning chunk 0 stores
    tinv: the chunk-factor tiles of ``gdn_tinv_f16.chunk_gdn_tinv`` over the same k / gate / beta / cu_seqlens / expand_num
    expand_num: scales every device-side ``cu_seqlens`` value for k and the chunk bounds (GDP; 1 = off); q (already
        l2-normalized) and dO stay on the compact token timeline
    q_step: row step of the phase rows inside ``q`` (1 = compact buffer)
    own_prologue: False skips the prologue launch when the chain prologue already ordered the table and built the descriptors
    """
    q.shape[1]
    k.shape[1]
    do.shape[1]
    DK = q.shape[2]
    DV = do.shape[2]
    cu_seqlens.shape[0] - 1
    io_dtype = get_dtype(q.dtype)

    cu_stream = cuda.CUstream(int(stream))

    if scheduler_counter is None:
        raise ValueError("scheduler_counter is required")
    run_order = bool(order_in_prologue)
    order_gen = run_order and work_item_scratch is None
    if run_order and scheduler_all is None:
        raise ValueError("order_in_prologue requires scheduler_all (the prologue zeroes both consumers' sched rings)")
    if not safe_gate:
        a_log = None
        dt_bias = None
    cache = get_compiled_cache(
        str(q.dtype),
        str(cu_seqlens.dtype),
        str(gate.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(d_final_state.dtype) if d_final_state is not None else "none",
        str(d_initial_state.dtype),
        device,
        num_sm,
        DK,
        DV,
        expand_num,
        d_final_state is not None,
        log_gate,
        safe_gate,
        run_order,
        order_gen,
        q_step,
    )

    if "compiled" not in cache:
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()
        workspace_cute = from_dlpack(workspace, assumed_align=128).mark_layout_dynamic()
        tinv_cute = from_dlpack(tinv, assumed_align=128).mark_layout_dynamic(leading_dim=3)

        dstate0_cute = from_dlpack(d_initial_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        dstate_in_cute = None
        if d_final_state is not None:
            dstate_in_cute = from_dlpack(d_final_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        work_items_cute = from_dlpack(work_items, assumed_align=16)
        work_items_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_count_cute = from_dlpack(work_count, assumed_align=4).mark_layout_dynamic()
        scheduler_counter_cute = from_dlpack(scheduler_counter, assumed_align=4).mark_layout_dynamic()
        a_log_cute = from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None
        dt_bias_cute = from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None
        cache["compiled"] = compile(
            io_dtype,
            use_dstate_in=d_final_state is not None,
            log_gate=log_gate,
            safe_gate=safe_gate,
            num_sm=num_sm,
            d_k=DK,
            d_v=DV,
            expand_num=expand_num,
            q_cute=from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            k_cute=from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            gate_cute=from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            a_log_cute=a_log_cute,
            dt_bias_cute=dt_bias_cute,
            do_cute=from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            cu_seqlens_cute=cu_seqlens_cute,
            dstate0_cute=dstate0_cute,
            dstate_in_cute=dstate_in_cute,
            tinv_cute=tinv_cute,
            work_items_cute=work_items_cute,
            work_count_cute=work_count_cute,
            scheduler_counter_cute=scheduler_counter_cute,
            scale=scale,
            workspace_cute=workspace_cute,
            stream=cu_stream,
        )

    compiled = cache["compiled"]

    if own_prologue and "prologue" not in cache:
        cu_placeholder = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()
        staging_placeholder = None
        if run_order and not order_gen:
            staging_placeholder = from_dlpack(work_item_scratch, assumed_align=16)
            staging_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_count_placeholder = from_dlpack(work_count, assumed_align=4).mark_layout_dynamic()
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        scheduler_all_placeholder = None
        if run_order:
            scheduler_all_placeholder = from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic()
        workspace_placeholder = from_dlpack(workspace, assumed_align=128).mark_layout_dynamic()
        tinv_placeholder = from_dlpack(tinv, assumed_align=128).mark_layout_dynamic(leading_dim=3) if tinv is not None else None
        cache["prologue"] = cute.compile(
            prologue,
            io_dtype,
            CFG.B_T,
            run_order,
            order_gen,
            expand_num,
            q_step,
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            cu_placeholder,
            staging_placeholder,
            work_count_placeholder,
            work_items_placeholder,
            scheduler_all_placeholder,
            tinv_placeholder,
            workspace_placeholder,
            cu_stream,
            options="--enable-tvm-ffi --opt-level 2",
        )
    if own_prologue:
        cache["prologue"](
            q,
            k,
            do,
            cu_seqlens,
            work_item_scratch if (run_order and not order_gen) else None,
            work_count,
            work_items,
            scheduler_all if run_order else None,
            tinv,
            workspace,
            cu_stream,
        )
    compiled(
        q,
        k,
        gate,
        a_log,
        dt_bias,
        do,
        cu_seqlens,
        d_initial_state,
        d_final_state,
        tinv,
        work_items,
        work_count,
        scheduler_counter,
        scale,
        workspace,
        cu_stream,
    )
    return cache


def run_bwd_summary(
    cache,
    q,
    k,
    gate,
    do,
    cu_seqlens,
    d_initial_state,
    d_final_state,
    work_items,
    work_count,
    scheduler_counter,
    scheduler_all,
    work_item_scratch,
    tensormap_workspace,
    scale,
    stream,
    a_log=None,
    dt_bias=None,
    *,
    tinv,
    own_prologue=True,
) -> None:
    """Replay the compiled plan: the prologue launch, then the main launch.  The plan validated the contract at build, so
    nothing here raises."""
    cu_stream = cuda.CUstream(int(stream))
    if own_prologue:
        cache["prologue"](
            q,
            k,
            do,
            cu_seqlens,
            work_item_scratch,
            work_count,
            work_items,
            scheduler_all,
            tinv,
            tensormap_workspace,
            cu_stream,
        )
    cache["compiled"](
        q,
        k,
        gate,
        a_log,
        dt_bias,
        do,
        cu_seqlens,
        d_initial_state,
        d_final_state,
        tinv,
        work_items,
        work_count,
        scheduler_counter,
        scale,
        tensormap_workspace,
        cu_stream,
    )


frost_gdn_bprop_summary_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_gdn_bprop_summary.set_name_prefix("cudnn", remove_cutlass_symbol=False)
