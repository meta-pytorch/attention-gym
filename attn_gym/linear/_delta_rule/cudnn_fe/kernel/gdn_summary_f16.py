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
Chunked Gated Delta Net (GDN) fused state-summary kernel for SM100 / SM103 / SM107 (Cutlass primitives): the recompute's state
pipeline run twice per chunk on one K / Gate stream, chain H (the state from zero or initial_state, consuming V) and
chain M (the identity seed with V = 0, the piece transition), one persistent CTA per (work item, head).

Algorithm overview (per chunk c, tokens [cC, (c+1)C), for each chain X in {H, M}):
  Inputs : K[BT,DK], V[BT,DV] (chain H only), Gate[BT] (scalar gate), T_inv[BT,BT] (the chunk factor tile of
           gdn_tinv_f16.py, Beta folded in)
  State  : S_X[DK,DV]  (recurrent state, held in TMEM; H seeded from zero / initial_state, M from the identity)

  Preprocessing (gate warp):
    cumsumlog[t]     = sum_{l=0}^{t} log(Gate_l)              cumulative log of gates
    cumprod[t]       = exp(cumsumlog[t])                       cumulative product of gates

  K*state GEMM   : KS_X[BT,DV] = K  @ S_X        (key applied to state)
  U GEMM         : U_X[BT,DV]  = T_inv @ Y_X     (corrected value vectors), Y_H = V - KS_H, Y_M = 0 - KS_M
  KV update GEMM : S_upd_X[DK,DV] = K^T @ (decay .* U_X)  (state update, BT contraction)

  Epilogue:
    S_X       = cumprod[BT-1] * S_X + S_upd_X          (update state in TMEM)
    output_state = S_H, output_transition = S_M        (stored domain M_buf = M^T; X_final = X_init @ M_buf + X_H)

Work-item contract as the recompute: chunks [compute_start, write_end), seed at compute_start == 0, store at
write_end == batch_num_chunks; empty items pass initial_state (or zero) and the identity through.

SMEM layout (stage counts live in gdn_summary_config.py; sizes at DK = DV = 128):
  Buffer                       Size (B)  Stages
  K                               16384       3
  V                               16384       3
  T_inv                            8192       3
  cumsumlog / cumprod               256       3
  scheduler ticket ring               4       2    <-- next-tile publish ring

TMEM layout (512 columns):
  Buffer                  Cols
  state H                 128     <-- DKxDV fp32
  state M                 128
  state input H            64     <-- fp16 state staging (K*state A operand)
  state input M            64
  acc H                    64     <-- KS then U of chain H
  acc M                    64     <-- KS then U of chain M

Warp assignments (12 warps = 384 threads):
  warps 0-3     : chain M epilogues - state stage/rescale, Y = 0 - K*state, U epilogue, transition store
  warps 4-7     : chain H epilogues - state stage/rescale, Y = V - K*state, U epilogue, state store
  warp  8       : Gate loads
  warp  9       : TMA load warp  - loads K, V and the chunk-factor tiles
  warp  10      : MMA warp       - both chains' K*state/U/KV per chunk; TMEM lifecycle
  warp  11      : register pool  - setmaxnreg.dec only; its share feeds the chain groups
"""

from dataclasses import dataclass, replace
from typing import NamedTuple, Optional, Type, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
import cutlass.experimental.cuda.tensor_map as tma
from cutlass.cute.arch.nvvm_wrappers import inline_ptx

from ..common.thd import emit_seq_descs, emit_tile_seq_descs, TENSOR_MAP_QWORDS
from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, decode_head, decode_work_item, order_body

RCP_LN2 = 1.4426950408889634  # 1/ln(2): natural-log gates -> the kernel's log2 domain
from ..tile_dsl.barrier import (
    MBarrier,
    Producer,
    PipelineState,
    advance,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from attn_gym._backends.cute.compat import SmemAllocator

from ..tile_dsl.handles import MmaDesc, SmemTile, smem_data_ptr, tma_slice_runtime_desc
from ..tile_dsl.mma import mma_ts
from ..tile_dsl.pointwise import fadd2, fmul2, fp32_to_fp16, opaque_f32_zero, opaque_i32_zero, softplus2, sub_f16x2
from ..tile_dsl.swizzle import swizzle_xor_128b
from ..tile_dsl.tma import tma_load_tile, tma_tensormap_acquire
from .gdn_summary_config import CFG

USE_PDL = True
STATE_DIMS = (64, 128)


class GdnSummaryBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_k_ready: MBarrier
    mb_k_done: MBarrier
    mb_v_ready: MBarrier
    mb_v_done: MBarrier
    mb_tinv_ready: MBarrier
    mb_tinv_done: MBarrier

    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier

    mb_state_input_h_ready: MBarrier
    mb_k_state_acc_h_ready: MBarrier
    mb_y_input_h_ready: MBarrier
    mb_u_acc_h_ready: MBarrier
    mb_decay_u_input_h_ready: MBarrier
    mb_state_acc_h_ready: MBarrier
    mb_state_input_m_ready: MBarrier
    mb_k_state_acc_m_ready: MBarrier
    mb_y_input_m_ready: MBarrier
    mb_u_acc_m_ready: MBarrier
    mb_decay_u_input_m_ready: MBarrier
    mb_state_acc_m_ready: MBarrier

    mb_tmem_done: MBarrier
    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier


def make_bars(cfg) -> GdnSummaryBars:
    """GdnSummaryBars factory."""
    ONE_LANE = 1
    MMA_ARRIVERS = len([cfg.tcgen05_mma_warp_id])
    GATE_WARP = cfg.threads_per_warp * len([cfg.load_gate_warp_id])
    CHAIN_THREADS = cfg.threads_per_warp * len(cfg.chain_h_warp_ids)
    BOTH_CHAINS = cfg.threads_per_warp * (len(cfg.chain_h_warp_ids) + len(cfg.chain_m_warp_ids))
    CONSUMER_WARPS = len(cfg.chain_h_warp_ids) + len(cfg.chain_m_warp_ids) + len([cfg.load_gate_warp_id, cfg.tcgen05_mma_warp_id])

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=16)

    return GdnSummaryBars(
        mb_k_ready=MBarrier(alloc(cfg.smem_k_stages), try_wait=True, stages=cfg.smem_k_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_k_done=MBarrier(alloc(cfg.smem_k_stages), try_wait=True, stages=cfg.smem_k_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_v_ready=MBarrier(alloc(cfg.smem_v_stages), try_wait=True, stages=cfg.smem_v_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_v_done=MBarrier(alloc(cfg.smem_v_stages), try_wait=True, stages=cfg.smem_v_stages, init_count=CHAIN_THREADS, producer=Producer.THREAD),
        mb_tinv_ready=MBarrier(alloc(cfg.smem_t_inv_stages), try_wait=True, stages=cfg.smem_t_inv_stages, init_count=ONE_LANE, producer=Producer.TMA_LOAD),
        mb_tinv_done=MBarrier(alloc(cfg.smem_t_inv_stages), try_wait=True, stages=cfg.smem_t_inv_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_gate_ready=MBarrier(alloc(cfg.smem_gate_stages), try_wait=True, stages=cfg.smem_gate_stages, init_count=GATE_WARP, producer=Producer.THREAD),
        mb_gate_done=MBarrier(alloc(cfg.smem_gate_stages), try_wait=True, stages=cfg.smem_gate_stages, init_count=BOTH_CHAINS, producer=Producer.THREAD),
        mb_state_input_h_ready=MBarrier(
            alloc(cfg.tmem_state_input_stages), try_wait=True, stages=cfg.tmem_state_input_stages, init_count=CHAIN_THREADS, producer=Producer.THREAD
        ),
        mb_k_state_acc_h_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_y_input_h_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CHAIN_THREADS, producer=Producer.THREAD),
        mb_u_acc_h_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_decay_u_input_h_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CHAIN_THREADS, producer=Producer.THREAD),
        mb_state_acc_h_ready=MBarrier(
            alloc(cfg.tmem_state_acc_stages), try_wait=True, stages=cfg.tmem_state_acc_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_state_input_m_ready=MBarrier(
            alloc(cfg.tmem_state_input_stages), try_wait=True, stages=cfg.tmem_state_input_stages, init_count=CHAIN_THREADS, producer=Producer.THREAD
        ),
        mb_k_state_acc_m_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_y_input_m_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CHAIN_THREADS, producer=Producer.THREAD),
        mb_u_acc_m_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT),
        mb_decay_u_input_m_ready=MBarrier(alloc(1), try_wait=True, stages=1, init_count=CHAIN_THREADS, producer=Producer.THREAD),
        mb_state_acc_m_ready=MBarrier(
            alloc(cfg.tmem_state_acc_stages), try_wait=True, stages=cfg.tmem_state_acc_stages, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_tmem_done=MBarrier(alloc(1), try_wait=True, stages=1, init_count=BOTH_CHAINS, producer=Producer.THREAD),
        mb_scheduler_ready=MBarrier(alloc(cfg.scheduler_stages), try_wait=True, stages=cfg.scheduler_stages, init_count=1, producer=Producer.THREAD),
        mb_scheduler_done=MBarrier(
            alloc(cfg.scheduler_stages), try_wait=True, stages=cfg.scheduler_stages, init_count=CONSUMER_WARPS, producer=Producer.THREAD
        ),
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
    sScheduler,
    bars,
):
    """Gate producer (warp 8): persistent scheduler loop + the cumsum/cumprod
    chunk loads (the recompute's gate path; beta is folded into the chunk factor)."""

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    elect_one = nvvm.elect_sync()

    gate_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=0)

    lane_idx = tidx % cfg.threads_per_warp

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
        head_o, v_offset = decode_head(cfg, head_idx)
        n_local = write_end - compute_start
        if cutlass.const_expr(cfg.safe_gate and mA_log is not None):
            if n_local > 0:
                a = -cute.math.exp2(mA_log[head_o].to(cutlass.Float32) * cutlass.Float32(RCP_LN2), fastmath=True) * cutlass.Float32(RCP_LN2)
        if cutlass.const_expr(cfg.safe_gate and mDt_bias is not None):
            if n_local > 0:
                bias = mDt_bias[head_o].to(cutlass.Float32)
        if n_local > 0:
            for local_idx in cutlass.range(n_local):
                # ---- Gate load: GMEM -> SMEM (OOB neutral: 1.0 -> log2 = 0.0) --------
                chunk_idx = compute_start + local_idx
                n_cols = cfg.b_t // cfg.threads_per_warp
                chunk_offset = batch_start + chunk_idx * cfg.b_t
                gGateSeq = mGate[None, head_o]

                gate_idx = gate_index.idx
                gate_phase = gate_index.phase
                gate_index = advance(gate_index, cfg.smem_gate_stages)

                oob_neutral = cutlass.Float32(0.0) if cutlass.const_expr(cfg.log_gate) else cutlass.Float32(1.0)
                toks = [chunk_offset + lane_idx + col * cfg.threads_per_warp for col in range(n_cols)]
                pos_valid = [tok < batch_end for tok in toks]
                if cutlass.const_expr(cfg.expand_num > 1):
                    gate_rows = [tok // cutlass.Int32(cfg.expand_num) for tok in toks]
                    gate_valid = [valid and (tok - row * cutlass.Int32(cfg.expand_num) == 0) for tok, row, valid in zip(toks, gate_rows, pos_valid)]
                    gate_row_end = batch_end // cutlass.Int32(cfg.expand_num)
                    gate_vals = [
                        gGateSeq[cutlass.min(row, gate_row_end - 1)].to(cutlass.Float32) if valid else oob_neutral for row, valid in zip(gate_rows, gate_valid)
                    ]
                else:
                    gate_valid = pos_valid
                    gate_vals = [gGateSeq[cutlass.min(tok, batch_end - 1)].to(cutlass.Float32) if valid else oob_neutral for tok, valid in zip(toks, pos_valid)]

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

                bars.mb_gate_done[gate_idx].wait(gate_phase)
                for col in cutlass.range_constexpr(n_cols):
                    pos = lane_idx + col * cfg.threads_per_warp
                    sCumsumlog[pos, 0, gate_idx] = gate_vals[col]
                    sCumprod[pos, 0, gate_idx] = cute.math.exp2(gate_vals[col], fastmath=True)

                bars.mb_gate_ready[gate_idx].arrive()
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    for _ in range(cfg.smem_gate_stages):
        bars.mb_gate_done[gate_index.idx].wait(gate_index.phase)
        gate_index = advance(gate_index, cfg.smem_gate_stages)


@cute.jit
def tcgen05_mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tmem_base_slot,
    sK,
    sK_trans,
    sTinv,
    sScheduler,
    bars,
):
    """MMA issuer role (warp 10): persistent scheduler loop issuing both chains'
    tcgen05 GEMMs, interleaved per chunk."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    k_index = PipelineState.start(phase=0)
    tinv_index = PipelineState.start(phase=0)
    state_input_h = PipelineState.start(phase=0)
    state_input_m = PipelineState.start(phase=0)
    y_input_h = PipelineState.start(phase=0)
    y_input_m = PipelineState.start(phase=0)
    decay_u_input_h = PipelineState.start(phase=0)
    decay_u_input_m = PipelineState.start(phase=0)

    elect_one = nvvm.elect_sync()

    nvvm.tcgen05_alloc(tmem_base_slot, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)

    # ---- chunk-invariant GEMM descriptors (rows = DV for chain H, DK for chain M) ----
    bytes_per_element = cfg.io_dtype.width // 8
    instruction_descriptor_k_state_h = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=cfg.b_t, m_dim=cfg.d_v
    )
    instruction_descriptor_k_state_m = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=cfg.b_t, m_dim=cfg.d_k
    )
    bmm_state_k_desc_h = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_k_state_h,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_state_k_desc_m = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_k_state_m,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_y_t_inv_desc_h = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_k_state_h,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_y_t_inv_desc_m = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_k_state_m,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    instruction_descriptor_kv_h = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=cfg.d_k, m_dim=cfg.d_v, b_major=1
    )
    instruction_descriptor_kv_m = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32, a_dtype=cfg.io_dtype, b_dtype=cfg.io_dtype, n_dim=cfg.d_k, m_dim=cfg.d_k, b_major=1
    )
    bmm_decay_u_k_desc_h = MmaDesc(
        M=cfg.d_v,
        N=cfg.d_k,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_kv_h,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_decay_u_k_desc_m = MmaDesc(
        M=cfg.d_k,
        N=cfg.d_k,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_kv_m,
        kind=nvvm.Tcgen05MMAKind.F16,
    )

    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    state_h_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_state_h_offset, cutlass.Float32)
    state_m_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_state_m_offset, cutlass.Float32)
    state_input_h_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_state_input_h_offset, cutlass.Int8)
    state_input_m_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_state_input_m_offset, cutlass.Int8)
    acc_h_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_acc_h_offset, cutlass.Float32)
    acc_m_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_acc_m_offset, cutlass.Float32)

    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        n_local = write_end - compute_start
        seed_state = compute_start == 0

        for local_idx in cutlass.range(n_local):  # noqa: B007
            if cutlass.const_expr(cfg.use_initial_state):
                have_state_h = local_idx > 0 or seed_state
            else:
                have_state_h = local_idx > 0
            have_state_m = local_idx > 0 or seed_state

            k_idx = k_index.idx
            bars.mb_k_ready[k_idx].wait(k_index.phase)
            k_index = advance(k_index, cfg.smem_k_stages)
            desc_k = sK[k_idx].desc()
            desc_k_trans = sK_trans[k_idx].desc()

            # ---- k state H = state H(T) @ K^T, k state M = state M(T) @ K^T ----------
            if have_state_h:
                bars.mb_state_input_h_ready[state_input_h.idx].wait(state_input_h.phase)
                state_input_h = advance(state_input_h, cfg.tmem_state_input_stages)
                mma_ts(bmm_state_k_desc_h, state_input_h_ptr, desc_k, acc_h_ptr, accumulate=False)
                if elect_one:
                    bars.mb_k_state_acc_h_ready[0].arrive(cta_group=1)
            if have_state_m:
                bars.mb_state_input_m_ready[state_input_m.idx].wait(state_input_m.phase)
                state_input_m = advance(state_input_m, cfg.tmem_state_input_stages)
                mma_ts(bmm_state_k_desc_m, state_input_m_ptr, desc_k, acc_m_ptr, accumulate=False)
                if elect_one:
                    bars.mb_k_state_acc_m_ready[0].arrive(cta_group=1)

            # ---- U H = Y H(T) @ T^-1, U M = Y M(T) @ T^-1 ----------------------------
            tinv_idx = tinv_index.idx
            bars.mb_tinv_ready[tinv_idx].wait(tinv_index.phase)
            tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
            desc_tinv = sTinv[tinv_idx].desc()

            bars.mb_y_input_h_ready[0].wait(y_input_h.phase)
            y_input_h = advance(y_input_h, 1)
            mma_ts(bmm_y_t_inv_desc_h, state_input_h_ptr, desc_tinv, acc_h_ptr, accumulate=False)
            if elect_one:
                bars.mb_u_acc_h_ready[0].arrive(cta_group=1)

            bars.mb_y_input_m_ready[0].wait(y_input_m.phase)
            y_input_m = advance(y_input_m, 1)
            mma_ts(bmm_y_t_inv_desc_m, state_input_m_ptr, desc_tinv, acc_m_ptr, accumulate=False)
            if elect_one:
                bars.mb_u_acc_m_ready[0].arrive(cta_group=1)
                bars.mb_tinv_done[tinv_idx].arrive(cta_group=1)

            # ---- state H += decayed U H(T) @ K, state M += decayed U M(T) @ K --------
            bars.mb_decay_u_input_h_ready[0].wait(decay_u_input_h.phase)
            decay_u_input_h = advance(decay_u_input_h, 1)
            mma_ts(bmm_decay_u_k_desc_h, state_input_h_ptr, desc_k_trans, state_h_ptr, accumulate=have_state_h)
            if elect_one:
                bars.mb_state_acc_h_ready[0].arrive(cta_group=1)

            bars.mb_decay_u_input_m_ready[0].wait(decay_u_input_m.phase)
            decay_u_input_m = advance(decay_u_input_m, 1)
            mma_ts(bmm_decay_u_k_desc_m, state_input_m_ptr, desc_k_trans, state_m_ptr, accumulate=have_state_m)
            if elect_one:
                bars.mb_state_acc_m_ready[0].arrive(cta_group=1)
                bars.mb_k_done[k_idx].arrive(cta_group=1)

        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

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
    sK_raw,
    sV_raw,
    sTinv_raw,
    desc_tinv_base,
    desc_k_base,
    desc_v_base,
    mScheduler,
    sScheduler,
    bars,
    k_ratio,
    v_ratio,
):
    """TMA-LDG warp role (warp 9): persistent scheduler loop + per-chunk K / V
    TMA loads and the chunk-factor tile TMA loads."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    k_index = PipelineState.start(phase=1)
    v_index = PipelineState.start(phase=1)
    tinv_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=1)

    elect_one = nvvm.elect_sync()
    tile_idx = cutlass.Int32(bidx)
    bytes_per_element = cfg.io_dtype.width // 8
    box_elems = 128 // bytes_per_element
    bt = cfg.b_t
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
    sK_tma = SmemTile(
        base=sK_raw,
        elems_per_stage=cfg.k_cosize // cfg.smem_k_stages,
        stages=cfg.smem_k_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_k // box_elems,
        tma_granu_elems=box_elems,
        tma_subtile_stride_elems=bt * box_elems,
    )
    sV_tma = SmemTile(
        base=sV_raw,
        elems_per_stage=cfg.v_cosize // cfg.smem_v_stages,
        stages=cfg.smem_v_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_v // box_elems,
        tma_granu_elems=box_elems,
        tma_subtile_stride_elems=bt * box_elems,
    )
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )

        head_o, v_offset = decode_head(cfg, head_idx)
        head_k = head_idx // k_ratio
        head_v = head_idx // v_ratio
        slot = batch_idx * desc_qwords
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_v_slot)
        desc_tinv_slot = (desc_tinv_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_tinv_slot)

        for chunk_idx in cutlass.range(compute_start, write_end):
            tok_coord = chunk_idx * cutlass.Int32(cfg.b_t)

            # ---- K load --------------------------------------------------------------
            k_idx = k_index.idx
            bars.mb_k_done[k_idx].wait(k_index.phase)
            k_index = advance(k_index, cfg.smem_k_stages)
            if elect_one:
                bars.mb_k_ready[k_idx].arrive(n_bytes=cfg.tma_k_bytes)
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), head_k, tok_coord)
            tma_load_tile(sK_tma[k_idx], k_slice, bars.mb_k_ready[k_idx].smem_ptr)

            # ---- chunk-factor tile load ----------------------------------------------
            tinv_idx = tinv_index.idx
            bars.mb_tinv_done[tinv_idx].wait(tinv_index.phase)
            tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
            if elect_one:
                bars.mb_tinv_ready[tinv_idx].arrive(n_bytes=cfg.tma_tinv_bytes)
            tinv_slice = tma_slice_runtime_desc(desc_tinv_slot, cutlass.Int32(0), cutlass.Int32(0), head_o, chunk_idx)
            tma_load_tile(sTinv_tma[tinv_idx], tinv_slice, bars.mb_tinv_ready[tinv_idx].smem_ptr)

            # ---- V load --------------------------------------------------------------
            v_idx = v_index.idx
            bars.mb_v_done[v_idx].wait(v_index.phase)
            v_index = advance(v_index, cfg.smem_v_stages)
            if elect_one:
                bars.mb_v_ready[v_idx].arrive(n_bytes=cfg.tma_v_bytes)
            v_slice = tma_slice_runtime_desc(desc_v_slot, v_offset, head_v, tok_coord)
            tma_load_tile(sV_tma[v_idx], v_slice, bars.mb_v_ready[v_idx].smem_ptr)

        tile_idx, scheduler_state = scheduler_publish_next(cfg, bars, sScheduler, mScheduler, scheduler_state, num_ctas, elect_one)

    for _ in range(cfg.smem_k_stages):
        bars.mb_k_done[k_index.idx].wait(k_index.phase)
        k_index = advance(k_index, cfg.smem_k_stages)
    for _ in range(cfg.smem_t_inv_stages):
        bars.mb_tinv_done[tinv_index.idx].wait(tinv_index.phase)
        tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
    for _ in range(cfg.smem_v_stages):
        bars.mb_v_done[v_index.idx].wait(v_index.phase)
        v_index = advance(v_index, cfg.smem_v_stages)
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.jit
def chain_warp_group(
    cfg,
    is_h: cutlass.Constexpr[bool],
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tidx,
    tmem_base_slot,
    sV_raw,
    sCumsumlog,
    sCumprod,
    mState_init,
    mState_out,
    sScheduler,
    mb_state_input_ready,
    mb_k_state_acc_ready,
    mb_y_input_ready,
    mb_u_acc_ready,
    mb_decay_u_input_ready,
    mb_state_acc_ready,
    bars,
):
    """One chain's epilogue group (4 warps): owns that chain's recurrent state from seed to final store.  ``is_h`` selects
    chain H (zero / ``initial_state`` seed, Y = V - cumprod .* KS) against chain M (identity seed, Y = -(cumprod .* KS))."""
    nvvm.setmaxregister(cfg.num_regs_chain, nvvm.SetMaxRegisterAction.INCREASE)

    v_index = PipelineState.start(phase=0)
    gate_index = PipelineState.start(phase=0)
    state_acc_index = PipelineState.start(phase=0)
    k_state_ready_index = PipelineState.start(phase=0)
    u_acc_ready_index = PipelineState.start(phase=0)

    group_threads = cfg.threads_per_warp * len(cfg.chain_h_warp_ids)
    group_tidx = tidx % group_threads
    lane_idx = group_tidx % cfg.threads_per_warp

    elect_one = nvvm.elect_sync()
    state_input_cnt = cutlass.Int32(0)
    ldtm_width = 32
    sttm_width = 16
    num_ldtms = cutlass.const_expr(cfg.d_k // ldtm_width)
    rows = cutlass.const_expr(cfg.d_v if is_h else cfg.d_k)
    row_halves = cutlass.const_expr(rows // 64)
    v_row = group_tidx % 8 + (group_tidx // 16 % 2) * 8
    if cutlass.const_expr(rows == 128):
        v_col = (group_tidx // 8 % 2) * 8 + (group_tidx // 32 % 2) * 32
        v_segment = (group_tidx // 64) * (cfg.b_t * 64)
        state_gmem_row = group_tidx
        state_row_valid = cutlass.Boolean(True)
    else:
        v_col = (group_tidx // 32) * 16 + (group_tidx // 8 % 2) * 8
        v_segment = 0
        state_gmem_row = (group_tidx // 32) * 16 + (group_tidx % 32) % 16
        state_row_valid = (group_tidx % 32) < 16
    v_stage_elements = cfg.v_cosize // cfg.smem_v_stages
    sV_base = smem_data_ptr(sV_raw)
    num_vals = 32

    nvvm.barrier_cta_sync_aligned(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    if cutlass.const_expr(is_h):
        tmem_state_col = tmem_col + cfg.tmem_state_h_offset
        tmem_state_input_col = tmem_col + cfg.tmem_state_input_h_offset
        tmem_acc_col = tmem_col + cfg.tmem_acc_h_offset
    else:
        tmem_state_col = tmem_col + cfg.tmem_state_m_offset
        tmem_state_input_col = tmem_col + cfg.tmem_state_input_m_offset
        tmem_acc_col = tmem_col + cfg.tmem_acc_m_offset

    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o, v_offset = decode_head(cfg, head_idx)
        if cutlass.const_expr(is_h):
            gmem_row = state_gmem_row + v_offset
            row_store = state_row_valid
        elif cutlass.const_expr(cfg.tiles_per_head > 1):
            gmem_row = state_gmem_row
            row_store = state_row_valid and v_offset == cutlass.Int32(0)
        else:
            gmem_row = state_gmem_row
            row_store = state_row_valid
        n_local = write_end - compute_start
        seed_state = compute_start == 0
        if n_local > 0:
            # ---- state seed: initial state / identity -> fp32 state TMEM -------------
            if cutlass.const_expr(is_h):
                if cutlass.const_expr(cfg.use_initial_state):
                    gState_init = mState_init[None, None, head_o, batch_idx]
                    if seed_state:
                        seed_vw = 16 // (mState_init.element_type.width // 8)
                        seed_src = (gState_init.iterator + gState_init.layout((gmem_row, 0))).raw_ptr()
                        for i in cutlass.range_constexpr(num_ldtms):
                            words = []
                            for g in cutlass.range_constexpr(ldtm_width // seed_vw):
                                seed_chunk = (seed_src + i * ldtm_width + g * seed_vw).load(count=seed_vw, alignment=16)
                                for t in cutlass.range_constexpr(seed_vw):
                                    words.append(seed_chunk[t].to(cfg.acc_dtype))
                            nvvm.tcgen05_st(
                                "32x32b",
                                nvvm.make_tmem_ptr(row_lo_addr + tmem_state_col + i * ldtm_width, cutlass.Float32),
                                cutlass.Vector.from_elements(tuple(words), cutlass.Float32),
                            )
                        nvvm.tcgen05_wait("store")
            else:
                if seed_state:
                    # ---- identity seed (one bound to the tile) -----------------------
                    one = inline_ptx(
                        "mov.b32 $0, $1;",
                        write_only_types=[cutlass.Float32],
                        read_only_args=[opaque_f32_zero() + cutlass.Float32(1.0), batch_idx],
                    )
                    for i in cutlass.range_constexpr(num_ldtms):
                        words = []
                        for k in cutlass.range_constexpr(32):
                            words.append(one if state_gmem_row == i * ldtm_width + k else cutlass.Float32(0.0))
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(row_lo_addr + tmem_state_col + i * ldtm_width, cutlass.Float32),
                            cutlass.Vector.from_elements(tuple(words), cutlass.Float32),
                        )
                    nvvm.tcgen05_wait("store")

            for local_idx in cutlass.range(n_local):  # noqa: B007
                if cutlass.const_expr(is_h and not cfg.use_initial_state):
                    valid_state = local_idx > 0
                else:
                    valid_state = local_idx > 0 or seed_state

                gate_idx = gate_index.idx
                bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
                gate_index = advance(gate_index, cfg.smem_gate_stages)
                cumprod_total = sCumprod[sCumprod.shape[0] - 1, 0, gate_idx]

                # ---- state stage + rescale -------------------------------------------
                if valid_state:
                    if local_idx > 0:
                        mb_state_acc_ready[state_acc_index.idx].wait(state_acc_index.phase)
                        state_acc_index = advance(state_acc_index, cfg.tmem_state_acc_stages)

                    state_input_stage_idx = state_input_cnt % cfg.tmem_state_input_stages
                    state_vecs = [
                        nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_state_col + i * ldtm_width, cutlass.Float32), num=32)
                        for i in range(num_ldtms)
                    ]
                    state_regs = [[state_vecs[i][k] for i in range(num_ldtms)] for k in range(32)]
                    for i in cutlass.range_constexpr(num_ldtms):
                        state_pack = [fp32_to_fp16(state_regs[2 * j][i], state_regs[2 * j + 1][i], dtype=cfg.io_dtype) for j in range(16)]
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(row_lo_addr + tmem_state_input_col + i * sttm_width, cutlass.Int32),
                            cutlass.Vector.from_elements(tuple(state_pack), cutlass.Int32),
                        )
                    nvvm.tcgen05_wait("store")
                    mb_state_input_ready[state_input_stage_idx].arrive()
                    state_input_cnt = state_input_cnt + 1

                    for i in cutlass.range_constexpr(num_ldtms):
                        state_scaled = []
                        for j in cutlass.range_constexpr(16):
                            s0, s1 = fmul2(state_regs[2 * j][i], state_regs[2 * j + 1][i], cumprod_total, cumprod_total)
                            state_scaled += [s0, s1]
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(row_lo_addr + tmem_state_col + i * ldtm_width, cutlass.Float32),
                            cutlass.Vector.from_elements(tuple(state_scaled), cutlass.Float32),
                        )
                    nvvm.tcgen05_wait("store")

                # ---- per-row Gate register builds ------------------------------------
                cumprod_vals = []
                for k in cutlass.range_constexpr(num_vals):
                    cumprod_vals.append(sCumprod[(lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2), 0, gate_idx])
                last_cumsumlog = sCumsumlog[cfg.b_t - 1, 0, gate_idx]
                cumsumlog_vals = []
                for k in cutlass.range_constexpr(num_vals):
                    cumsumlog_vals.append(sCumsumlog[(lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2), 0, gate_idx])
                decay_scale_vals = []
                for k in cutlass.range_constexpr(0, num_vals, 2):
                    d0, d1 = fadd2(last_cumsumlog, last_cumsumlog, -cumsumlog_vals[k], -cumsumlog_vals[k + 1])
                    decay_scale_vals.append(cute.math.exp2(d0, fastmath=True))
                    decay_scale_vals.append(cute.math.exp2(d1, fastmath=True))
                bars.mb_gate_done[gate_idx].arrive()

                # ---- Y = V - k state (packed 16-bit) ---------------------------------
                if cutlass.const_expr(is_h):
                    v_idx = v_index.idx
                    bars.mb_v_ready[v_idx].wait(v_index.phase)
                    v_index = advance(v_index, cfg.smem_v_stages)

                    v_frag = []
                    for half in cutlass.range_constexpr(row_halves):
                        v_words = []
                        for block in cutlass.range_constexpr(4):
                            v_raw = nvvm.ldmatrix(
                                (
                                    sV_base
                                    + v_idx * v_stage_elements
                                    + v_segment
                                    + (v_row + block * 16) * 64
                                    + swizzle_xor_128b(v_row + block * 16, v_col + half * 16)
                                ),
                                4,
                                nvvm.MMALayout.COL,
                            )
                            for i in cutlass.range_constexpr(4):
                                v_words.append(v_raw[i])
                        v_frag.append(v_words)
                else:
                    zero_word = opaque_i32_zero()
                    v_frag = []
                    for half in cutlass.range_constexpr(row_halves):
                        v_frag.append([zero_word for _ in range(16)])
                if valid_state:
                    mb_k_state_acc_ready[0].wait(k_state_ready_index.phase)
                    k_state_ready_index = advance(k_state_ready_index, 1)

                    for half in cutlass.range_constexpr(row_halves):
                        k_state_vec = nvvm.tcgen05_ld(
                            "16x256b",
                            nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_acc_col, cutlass.Float32),
                            num=8,
                        )
                        for j in cutlass.range_constexpr(16):
                            s0, s1 = fmul2(k_state_vec[2 * j], k_state_vec[2 * j + 1], cumprod_vals[2 * j], cumprod_vals[2 * j + 1])
                            k_state_pack = fp32_to_fp16(s0, s1, dtype=cfg.io_dtype)
                            v_frag[half][j] = sub_f16x2(v_frag[half][j], k_state_pack, cfg.io_dtype)
                for half in cutlass.range_constexpr(row_halves):
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_state_input_col, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(v_frag[half]), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                mb_y_input_ready[0].arrive()
                if cutlass.const_expr(is_h):
                    bars.mb_v_done[v_idx].arrive()

                # ---- U epilogue + decayed-U publish ----------------------------------
                mb_u_acc_ready[0].wait(u_acc_ready_index.phase)
                u_acc_ready_index = advance(u_acc_ready_index, 1)

                u_acc_vecs = []
                for half in cutlass.range_constexpr(row_halves):
                    u_acc_vecs.append(
                        nvvm.tcgen05_ld(
                            "16x256b",
                            nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_acc_col, cutlass.Float32),
                            num=8,
                        )
                    )
                u_acc_regs = [[u_acc_vecs[h][k] for h in range(row_halves)] for k in range(32)]

                for half in cutlass.range_constexpr(row_halves):
                    for j in cutlass.range_constexpr(16):
                        u_acc_regs[2 * j][half], u_acc_regs[2 * j + 1][half] = fmul2(
                            u_acc_regs[2 * j][half], u_acc_regs[2 * j + 1][half], decay_scale_vals[2 * j], decay_scale_vals[2 * j + 1]
                        )
                    decay_pack = [fp32_to_fp16(u_acc_regs[2 * j][half], u_acc_regs[2 * j + 1][half], dtype=cfg.io_dtype) for j in range(16)]
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(((tmem_row + half * 16) << 16) + tmem_state_input_col, cutlass.Int32),
                        cutlass.Vector.from_elements(tuple(decay_pack), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                mb_decay_u_input_ready[0].arrive()

        # ---- final state store: TMEM -> GMEM -----------------------------------------
        if n_local > 0:
            mb_state_acc_ready[state_acc_index.idx].wait(state_acc_index.phase)
            state_acc_index = advance(state_acc_index, cfg.tmem_state_acc_stages)
            if write_end == batch_num_chunks:
                gState_out = mState_out[None, None, head_o, batch_idx]
                state_vw = 16 // (mState_out.element_type.width // 8)
                state_dst = (gState_out.iterator + gState_out.layout((gmem_row, 0))).raw_ptr()
                for i in cutlass.range_constexpr(num_ldtms):
                    state_vec = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + tmem_state_col + i * ldtm_width, cutlass.Float32), num=32)
                    if row_store:
                        for g in cutlass.range_constexpr(ldtm_width // state_vw):
                            (state_dst + i * ldtm_width + g * state_vw).store(
                                cutlass.Vector.from_elements(
                                    tuple(state_vec[g * state_vw + t].to(mState_out.element_type) for t in range(state_vw)), mState_out.element_type
                                ),
                                alignment=16,
                            )
        else:
            if write_end == batch_num_chunks:
                gState_out = mState_out[None, None, head_o, batch_idx]
                if cutlass.const_expr(is_h and cfg.use_initial_state):
                    gState_in = mState_init[None, None, head_o, batch_idx]
                    for r in cutlass.range(num_ldtms * ldtm_width):
                        if row_store:
                            gState_out[gmem_row, r] = gState_in[gmem_row, r]
                else:
                    for i in cutlass.range_constexpr(num_ldtms):
                        for k in cutlass.range_constexpr(32):
                            if row_store:
                                gState_out[gmem_row, i * ldtm_width + k] = cutlass.Float32(0.0).to(mState_out.element_type)
                    if cutlass.const_expr(not is_h):
                        if row_store:
                            gState_out[gmem_row, gmem_row] = cutlass.Float32(1.0).to(mState_out.element_type)

        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    bars.mb_tmem_done[0].arrive()


@cute.jit
def build_descs_body(
    widx,
    base_k,
    base_v,
    base_tinv,
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    tinv: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
) -> None:
    """Per-batch descriptor-array build, one warp per array (K, V), the tinv array on the K warp."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_k_arr = cute.make_tensor(desc_workspace.iterator, cute.make_layout((arr_words,), stride=(1,)))
    desc_v_arr = cute.make_tensor(desc_workspace.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_tinv_arr = cute.make_tensor(desc_workspace.iterator + 2 * arr_words, cute.make_layout((arr_words,), stride=(1,)))

    if widx == 0:
        emit_seq_descs(base_k, desc_k_arr, cu_seqlens, k, n_batch, 2, expand_num, lanes=32)
        if cutlass.const_expr(tinv is not None):
            emit_tile_seq_descs(base_tinv, desc_tinv_arr, cu_seqlens, tinv, n_batch, b_t, 3, expand_num, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 1:
        emit_seq_descs(base_v, desc_v_arr, cu_seqlens, v, n_batch, 2, expand_num, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


@cute.kernel
def frost_gdn_summary_prologue(
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    tiles_per_head: cutlass.Constexpr[int],
    base_k: cutlass.GridConstant[tma.TensorMap],
    base_v: cutlass.GridConstant[tma.TensorMap],
    base_tinv: cutlass.GridConstant[tma.TensorMap],
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    tinv: Optional[cute.Tensor],
    mStaging: Optional[cute.Tensor],
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mScheduler: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
) -> None:
    """Two-CTA prologue: under ``run_order`` block 0 LPT-orders the work-item table and zeroes the scheduler rings
    (:func:`order_body`); block 1 builds the per-batch K and V TMA-descriptor arrays (:func:`build_descs_body`)."""
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
            n_heads_out = cutlass.Int32(gate.shape[1])
            if cutlass.const_expr(tiles_per_head > 1):
                n_heads_out = n_heads_out * cutlass.Int32(tiles_per_head)
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
            base_k,
            base_v,
            base_tinv,
            desc_workspace,
            cu_seqlens,
            k,
            v,
            tinv,
            n_batch,
            b_t,
            expand_num,
        )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    expand_num: cutlass.Constexpr[int],
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    cu_seqlens: cute.Tensor,
    work_item_staging: Optional[cute.Tensor],
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler_all: Optional[cute.Tensor],
    tinv: Optional[cute.Tensor],
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
    tiles_per_head: cutlass.Constexpr[int] = 1,
):
    """One-launch prologue: order the work items (``run_order``) and build the
    2 per-batch TMA-descriptor arrays (K, V) into ``tensormap_workspace``."""
    h_k = k.shape[1]
    h_v = v.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    d_v = v.shape[2]
    bytes_per_element = io_dtype.width // 8
    box_elems = 128 // bytes_per_element
    bt = b_t

    k_row_stride, k_head_stride = k.stride[0], k.stride[1]
    v_row_stride, v_head_stride = v.stride[0], v.stride[1]

    seqlen = k.shape[0]
    d_k = k.shape[2]
    k_headed = cute.make_tensor(k.iterator, cute.make_layout((seqlen, h_k, d_k), stride=(k_row_stride, k_head_stride, 1)))
    v_headed = cute.make_tensor(v.iterator, cute.make_layout((d_v, h_v, seqlen), stride=(1, v_head_stride, v_row_stride)))
    swizzle_128b = tma.TensorMapSwizzle.s128b
    base_desc_k = tma.create_tensor_map_tiled_from_view(k_headed, box_dims=(bt, 1, box_elems), stride_order=(2, 1, 0), swizzle=swizzle_128b)
    base_desc_v = tma.create_tensor_map_tiled_from_view(v_headed, box_dims=(box_elems, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b)

    base_desc_tinv = base_desc_k
    if cutlass.const_expr(tinv is not None):
        tinv_tiles = cute.make_tensor(
            tinv.iterator,
            cute.make_layout((tinv.shape[0], tinv.shape[1], tinv.shape[2], tinv.shape[3]), stride=(tinv.stride[0], tinv.stride[1], tinv.stride[2], 1)),
        )
        base_desc_tinv = tma.create_tensor_map_tiled_from_view(tinv_tiles, box_dims=(1, 1, bt, bt), stride_order=(3, 2, 1, 0), swizzle=swizzle_128b)
    frost_gdn_summary_prologue(
        run_order,
        order_gen,
        b_t,
        expand_num,
        tiles_per_head,
        base_desc_k,
        base_desc_v,
        base_desc_tinv,
        tensormap_workspace,
        cu_seqlens,
        k,
        v,
        gate,
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
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    a_log: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    tinv: cute.Tensor,
    state_in: Optional[cute.Tensor],
    state_out: cute.Tensor,
    transition_out: cute.Tensor,
    work_items: cute.Tensor,
    work_count: cute.Tensor,
    scheduler_counter: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
):
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = cutlass.Int32(gate.shape[1])
    if cutlass.const_expr(cfg.tiles_per_head > 1):
        heads_out = heads_out * cutlass.Int32(cfg.tiles_per_head)
    k_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(k.shape[1]))
    v_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(v.shape[1]))

    # ---- head-major views: (tokens, dim, heads) k / q, (dim, tokens, heads) v / o, (V, K, heads, seqs) states ----
    k = cute.make_tensor(k.iterator, cute.make_layout((k.shape[0], k.shape[2], k.shape[1]), stride=(k.stride[0], k.stride[2], k.stride[1])))
    v = cute.make_tensor(v.iterator, cute.make_layout((v.shape[2], v.shape[0], v.shape[1]), stride=(v.stride[2], v.stride[0], v.stride[1])))
    if cutlass.const_expr(state_in is not None):
        state_in = cute.make_tensor(
            state_in.iterator,
            cute.make_layout(
                (state_in.shape[2], state_in.shape[3], state_in.shape[1], state_in.shape[0]),
                stride=(state_in.stride[2], state_in.stride[3], state_in.stride[1], state_in.stride[0]),
            ),
        )
    state_out = cute.make_tensor(
        state_out.iterator,
        cute.make_layout(
            (state_out.shape[2], state_out.shape[3], state_out.shape[1], state_out.shape[0]),
            stride=(state_out.stride[2], state_out.stride[3], state_out.stride[1], state_out.stride[0]),
        ),
    )
    transition_out = cute.make_tensor(
        transition_out.iterator,
        cute.make_layout(
            (transition_out.shape[2], transition_out.shape[3], transition_out.shape[1], transition_out.shape[0]),
            stride=(transition_out.stride[2], transition_out.stride[3], transition_out.stride[1], transition_out.stride[0]),
        ),
    )

    cumsumlog_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_gate_stages))

    @cute.struct
    class SharedStorage:
        k: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, cfg.k_cosize], cfg.buffer_align_bytes]
        v: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, cfg.v_cosize], cfg.buffer_align_bytes]
        t_inv: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, cfg.t_inv_cosize], cfg.buffer_align_bytes]

    num_descs = batch_size

    # ---- launch ----------------------------------------------------------------------
    grid_shape = (cfg.max_active_clusters, 1, 1)

    frost_gdn_summary(
        cfg,
        SharedStorage,
        k_ratio,
        v_ratio,
        gate,
        a_log,
        dt_bias,
        cu_seqlens,
        tinv,
        state_in,
        state_out,
        transition_out,
        work_items,
        work_count,
        scheduler_counter,
        cumsumlog_smem_layout_staged,
        k,
        v,
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
def frost_gdn_summary(
    cfg: cutlass.Constexpr,
    shared_type: cutlass.Constexpr,
    k_ratio: cute.FastDivmodDivisorV2,
    v_ratio: cute.FastDivmodDivisorV2,
    mGate: cute.Tensor,
    mA_log: Optional[cute.Tensor],
    mDt_bias: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    mTinv: cute.Tensor,
    mState_init: Optional[cute.Tensor],
    mState_out: cute.Tensor,
    mTransition_out: cute.Tensor,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor,
    cumsumlog_smem_layout_staged: cute.Layout,
    mK,
    mV,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
):
    """Fused GDN state-summary kernel: warp-specialized dispatch over (batch,
    head) work items, two recurrences per CTA."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]

    total_tiles = mCount[0]

    desc_base_words = tensormap_workspace.iterator.raw_ptr()
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)
    arr_words = n_desc * desc_qwords
    desc_k_base = desc_base_words
    desc_v_base = desc_base_words + arr_words
    desc_tinv_base = desc_base_words + cutlass.Int32(2) * arr_words

    SMEM = cutlass.AddressSpace.smem

    SWZ = 2
    LEAD = 16
    STRIDE = 8 * 128
    KT_LEAD = cfg.b_t * 128
    # Barrier, scheduler and gate staging arrays keep stable raw pointers; the 1024-aligned tile
    # buffers share one SharedStorage allocation placed after them.
    bars = make_bars(cfg)
    tmem_base_slot = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=16)
    sScheduler = cutlass.Array(cutlass.Int32, cfg.scheduler_stages, space=SMEM, alignment=16)
    cumsumlog_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
    cumprod_raw = cutlass.Array(cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128)
    storage = SmemAllocator().allocate(shared_type)
    sK_raw = storage.k.get_tensor(cute.make_layout((cfg.k_cosize,)))
    sK = SmemTile(
        base=sK_raw,
        elems_per_stage=(cfg.k_cosize // cfg.smem_k_stages),
        stages=cfg.smem_k_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_trans = SmemTile(
        base=sK_raw,
        elems_per_stage=(cfg.k_cosize // cfg.smem_k_stages),
        stages=cfg.smem_k_stages,
        leading_byte_offset=KT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sV_raw = storage.v.get_tensor(cute.make_layout((cfg.v_cosize,)))
    sTinv_raw = storage.t_inv.get_tensor(cute.make_layout((cfg.t_inv_cosize,)))
    sTinv = SmemTile(
        base=sTinv_raw,
        elems_per_stage=(cfg.t_inv_cosize // cfg.smem_t_inv_stages),
        stages=cfg.smem_t_inv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sCumsumlog = cute.make_tensor(
        cute.make_ptr(cutlass.Float32, smem_data_ptr(cumsumlog_raw).toint(), mem_space=cute.AddressSpace.smem, assumed_align=128),
        cumsumlog_smem_layout_staged,
    )
    sCumprod = cute.make_tensor(
        cute.make_ptr(cutlass.Float32, smem_data_ptr(cumprod_raw).toint(), mem_space=cute.AddressSpace.smem, assumed_align=128),
        cumsumlog_smem_layout_staged,
    )

    # ---- mbarrier init (all threads) -------------------------------------------------
    for s in range(cfg.smem_k_stages):
        bars.mb_k_ready[s].init()
        bars.mb_k_done[s].init()
    for s in range(cfg.smem_v_stages):
        bars.mb_v_ready[s].init()
        bars.mb_v_done[s].init()
    for s in range(cfg.smem_t_inv_stages):
        bars.mb_tinv_ready[s].init()
        bars.mb_tinv_done[s].init()
    for s in range(cfg.smem_gate_stages):
        bars.mb_gate_ready[s].init()
        bars.mb_gate_done[s].init()
    for s in range(cfg.tmem_state_input_stages):
        bars.mb_state_input_h_ready[s].init()
    for s in range(cfg.tmem_state_acc_stages):
        bars.mb_state_acc_h_ready[s].init()
    bars.mb_k_state_acc_h_ready[0].init()
    bars.mb_y_input_h_ready[0].init()
    bars.mb_u_acc_h_ready[0].init()
    bars.mb_decay_u_input_h_ready[0].init()
    for s in range(cfg.tmem_state_input_stages):
        bars.mb_state_input_m_ready[s].init()
    for s in range(cfg.tmem_state_acc_stages):
        bars.mb_state_acc_m_ready[s].init()
    bars.mb_k_state_acc_m_ready[0].init()
    bars.mb_y_input_m_ready[0].init()
    bars.mb_u_acc_m_ready[0].init()
    bars.mb_decay_u_input_m_ready[0].init()
    for s in range(cfg.scheduler_stages):
        bars.mb_scheduler_ready[s].init()
        bars.mb_scheduler_done[s].init()
    bars.mb_tmem_done[0].init()

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # ---- warp specialization ---------------------------------------------------------
    if warp_idx >= cfg.chain_m_warp_ids[0] and warp_idx <= cfg.chain_m_warp_ids[-1]:
        chain_warp_group(
            cfg,
            False,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tidx,
            tmem_base_slot=tmem_base_slot,
            sV_raw=sV_raw,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            mState_init=None,
            mState_out=mTransition_out,
            sScheduler=sScheduler,
            mb_state_input_ready=bars.mb_state_input_m_ready,
            mb_k_state_acc_ready=bars.mb_k_state_acc_m_ready,
            mb_y_input_ready=bars.mb_y_input_m_ready,
            mb_u_acc_ready=bars.mb_u_acc_m_ready,
            mb_decay_u_input_ready=bars.mb_decay_u_input_m_ready,
            mb_state_acc_ready=bars.mb_state_acc_m_ready,
            bars=bars,
        )

    if warp_idx >= cfg.chain_h_warp_ids[0] and warp_idx <= cfg.chain_h_warp_ids[-1]:
        chain_warp_group(
            cfg,
            True,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tidx,
            tmem_base_slot=tmem_base_slot,
            sV_raw=sV_raw,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            mState_init=mState_init,
            mState_out=mState_out,
            sScheduler=sScheduler,
            mb_state_input_ready=bars.mb_state_input_h_ready,
            mb_k_state_acc_ready=bars.mb_k_state_acc_h_ready,
            mb_y_input_ready=bars.mb_y_input_h_ready,
            mb_u_acc_ready=bars.mb_u_acc_h_ready,
            mb_decay_u_input_ready=bars.mb_decay_u_input_h_ready,
            mb_state_acc_ready=bars.mb_state_acc_h_ready,
            bars=bars,
        )

    elif warp_idx == cfg.load_gate_warp_id:
        gate_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            tidx=tidx,
            mGate=mGate,
            mA_log=mA_log,
            mDt_bias=mDt_bias,
            sCumsumlog=sCumsumlog,
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
            sK=sK,
            sK_trans=sK_trans,
            sTinv=sTinv,
            sScheduler=sScheduler,
            bars=bars,
        )

    elif warp_idx == cfg.tma_warp_id:
        tmaldg_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sK_raw=sK_raw,
            sV_raw=sV_raw,
            sTinv_raw=sTinv_raw,
            desc_tinv_base=desc_tinv_base,
            desc_k_base=desc_k_base,
            desc_v_base=desc_v_base,
            mScheduler=mScheduler,
            sScheduler=sScheduler,
            bars=bars,
            k_ratio=k_ratio,
            v_ratio=v_ratio,
        )

    elif warp_idx == cfg.register_pool_warp_id:
        nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)


@dataclass(frozen=True)
class GdnSummaryCfg:
    """Per-compile fused-summary kernel knob (``build_cfg``): the dtype / GQA / state-flag fields are the ``cute.compile``
    cache keys, the rest derives from ``CFG``.
    """

    io_dtype: Type[cutlass.Numeric]
    acc_dtype: Type[cutlass.Numeric]
    state_dtype: Type[cutlass.Numeric]
    max_active_clusters: int
    use_initial_state: bool
    d_k: int
    d_v: int
    log_gate: bool = False
    safe_gate: bool = False
    tiles_per_head: int = 1
    scheduler_stages: int = CFG.SMEM_SCHEDULER_STAGES

    # ---- fixed constants stamped from CFG at build time ------------------------------
    b_t: int = CFG.B_T
    expand_num: int = 1
    chain_m_warp_ids: Tuple[int, ...] = CFG.CHAIN_M_WARP_IDS
    chain_h_warp_ids: Tuple[int, ...] = CFG.CHAIN_H_WARP_IDS
    load_gate_warp_id: int = CFG.LOAD_GATE_WARP_ID
    tma_warp_id: int = CFG.TMA_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    register_pool_warp_id: int = CFG.REGISTER_POOL_WARP_ID
    num_regs_chain: int = CFG.NUM_REGS_CHAIN
    num_regs_other: int = CFG.NUM_REGS_OTHER
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0
    cluster_shape_mnk: Tuple[int, int, int] = CFG.CLUSTER_SHAPE_MNK

    # ---- named barrier slot (0 is the CTA-wide sync) ---------------------------------
    tmem_lifecycle_barrier_id: int = 1
    tmem_user_threads: int = 0

    # ---- SMEM / TMEM stage counts + TMEM column offsets ------------------------------
    smem_k_stages: int = CFG.SMEM_K_STAGES
    smem_v_stages: int = CFG.SMEM_V_STAGES
    smem_t_inv_stages: int = CFG.SMEM_T_INV_STAGES
    smem_gate_stages: int = CFG.SMEM_GATE_STAGES
    tmem_state_acc_stages: int = CFG.TMEM_STATE_ACC_STAGES
    tmem_state_input_stages: int = CFG.TMEM_STATE_INPUT_STAGES
    tmem_state_h_offset: int = 0
    tmem_state_m_offset: int = 0
    tmem_state_input_h_offset: int = 0
    tmem_state_input_m_offset: int = 0
    tmem_acc_h_offset: int = 0
    tmem_acc_m_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- derived by build_cfg (SMEM cosizes, TMA transaction bytes) --------------------
    k_cosize: int = 0
    v_cosize: int = 0
    t_inv_cosize: int = 0
    tma_k_bytes: int = 0
    tma_v_bytes: int = 0
    tma_tinv_bytes: int = 0


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    state_dtype: Type[cutlass.Numeric],
    *,
    max_active_clusters: int,
    use_initial_state: bool,
    log_gate: bool = False,
    safe_gate: bool = False,
    d_k: int,
    d_v: int,
    expand_num: int = 1,
    tiles_per_head: int = 1,
) -> GdnSummaryCfg:
    """Build the per-compile ``GdnSummaryCfg`` (io_dtype in {Float16, BFloat16}; acc is always Float32).  ``tiles_per_head`` > 1
    runs every gate head as ``tiles_per_head`` tiles of ``d_v`` value columns of H each; chain M is stored by the first tile."""
    if d_k not in STATE_DIMS or d_v not in STATE_DIMS:
        raise ValueError(f"fused GDN summary serves DK, DV in {STATE_DIMS}, got DK={d_k} DV={d_v}")
    cfg = GdnSummaryCfg(
        io_dtype=io_dtype,
        acc_dtype=cutlass.Float32,
        state_dtype=state_dtype,
        max_active_clusters=max_active_clusters,
        use_initial_state=use_initial_state,
        log_gate=log_gate,
        safe_gate=safe_gate,
        d_k=d_k,
        d_v=d_v,
        expand_num=expand_num,
        tiles_per_head=tiles_per_head,
    )
    n_chain = len(cfg.chain_h_warp_ids) + len(cfg.chain_m_warp_ids)
    tmem_state_m_offset = cfg.tmem_state_acc_stages * cfg.d_k
    tmem_state_input_h_offset = tmem_state_m_offset + cfg.tmem_state_acc_stages * cfg.d_k
    tmem_state_input_m_offset = tmem_state_input_h_offset + cfg.tmem_state_input_stages * (cfg.d_k // 2)
    tmem_acc_h_offset = tmem_state_input_m_offset + cfg.tmem_state_input_stages * (cfg.d_k // 2)
    tmem_acc_m_offset = tmem_acc_h_offset + cfg.b_t
    if tmem_acc_m_offset + cfg.b_t > 512:
        raise ValueError(f"TMEM layout exceeds 512 columns: {tmem_acc_m_offset + cfg.b_t}")
    # ---- SMEM sizing: per-buffer element cosizes and TMA transaction bytes -----------------
    bytes_per_element = io_dtype.width // 8
    k_tile_elements = cfg.b_t * cfg.d_k
    v_tile_elements = cfg.d_v * cfg.b_t
    tinv_tile_elements = cfg.b_t * cfg.b_t
    return replace(
        cfg,
        threads_per_cta=cfg.threads_per_warp * (4 + n_chain),
        tmem_user_threads=cfg.threads_per_warp * (1 + n_chain),
        tmem_state_h_offset=0,
        tmem_state_m_offset=tmem_state_m_offset,
        tmem_state_input_h_offset=tmem_state_input_h_offset,
        tmem_state_input_m_offset=tmem_state_input_m_offset,
        tmem_acc_h_offset=tmem_acc_h_offset,
        tmem_acc_m_offset=tmem_acc_m_offset,
        k_cosize=k_tile_elements * cfg.smem_k_stages,
        v_cosize=v_tile_elements * cfg.smem_v_stages,
        t_inv_cosize=tinv_tile_elements * cfg.smem_t_inv_stages,
        tma_k_bytes=k_tile_elements * bytes_per_element,
        tma_v_bytes=v_tile_elements * bytes_per_element,
        tma_tinv_bytes=tinv_tile_elements * bytes_per_element,
    )


TENSORMAP_DESC_ARRAYS = 3  # per-batch runtime TMA descriptors: K, V, tinv


frost_gdn_summary_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_gdn_summary.set_name_prefix("cudnn", remove_cutlass_symbol=False)
