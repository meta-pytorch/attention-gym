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
# attn_gym.linear._delta_rule.cudnn_fe; operand SMEM in lead/tail SharedStorage structs (v1.30
# order) read through smem_data_ptr and named swizzle-box checkpoint offsets; mbarriers initialized
# by one thread, with the scheduler-done count derived from the warp count; a frozen cfg whose
# build_cfg derives the SMEM cosizes and TMA byte counts; paged state (null/fresh/resumed routes
# resolved through paged_state, route metadata threaded to the work-table prologue); the
# upstream-only expand_num, safe_gate/A_log/dt_bias, beta-sigmoid, and negative-eigenvalue paths
# removed; Ruff formatting.

"""
Chunked Gated Delta Net (GDN) prefill kernel for SM100 / SM103 / SM107 (Cutlass primitives)
with optional per-chunk state-checkpoint output.

Algorithm overview (per chunk c, tokens [cC, (c+1)C)):
  Inputs : Q[BT,DK], K[BT,DK], V[BT,DV], Gate[BT] (scalar gate), Beta[BT] (scalar LR)
  State  : S_prev[DK,DV]  (recurrent state, held in TMEM)

  Preprocessing (gate warp; T_pairwise in compute group 0 registers):
    cumsumlog[t]     = sum_{l=0}^{t} log(gate_l)              cumulative log of gates
    cumprod[t]       = exp(cumsumlog[t])                       cumulative product of gates
    T_pairwise[i,j]  = cumprod[i] / cumprod[j]  (i>=j)       inter-token transfer weights
    (stored in registers; 128 regs/thread)

  QK GEMM        : W_qk[BT,BT]  = Q  @ K^T       (output attention scores)
  K*state GEMM   : KS[BT,DV] = K  @ S_prev    (key applied to state)
  Q*state GEMM   : QS[BT,DV] = Q  @ S_prev    (inter-chunk output, before T scaling)
  U GEMM         : U[BT,DV]  = T_inv @ Y       (corrected value vectors)
                   where T_inv = (I + M_kk)^{-1},  M_kk[i,j] = T[i,j]*Beta[i]*(K K^T)[i,j]  (the chunk factor tile of gdn_tinv_f16.py)
  QKV GEMM       : O_intra[BT,DV] = W_qkv @ U    (intra-chunk output)
                   where W_qkv = T*Beta*W_qk (the A tile)
  KV update GEMM : S_upd[DK,DV] = K^T @ (decay .* U)  (state update, BT contraction)
                   where Y[BT,DV] = V - KS    (delta rule residuals, after decay)

  Epilogue:
    O[BT,DV]  = O_intra + T_col * QS             (combine intra + inter)
    S_next    = cumprod[BT-1] * S_prev + S_upd        (update state in TMEM)

Chunks run in PAIRS; odd counts pad with a neutral zero-filled chunk.

SMEM layout (227 KB SM100 / SM103 / SM107 cap; stage counts live in gdn_prefill_config.py;
enable_checkpoints compiles trim the K + Q stages to fit the checkpoint buffer):
  Buffer                       Size (B)  Stages
  K + Q (two-box stage)           32768       4
  V                               16384       2
  T_inv                            8192       3
  A tile output                    8192       3
  O store                         16384       1
  checkpoint staging            DK*DV*2       1    <-- enable_checkpoints only
  cumsumlog / cumprod / Beta        256       3
  scheduler ticket ring               4       2    <-- next-tile publish ring

TMEM layout (512 columns):
  Buffer                  Cols
  state                   128     <-- DKxDV fp32 = 128x128x4B
  Q*state / O acc          64     <-- BTxDV fp32 accumulator
  state input                64     <-- fp16 state staging (K*state/Q*state A operand)
  cg0 shared acc          128     <-- 2-stage ring: QK0/QK1 (Q K^T at M = b_t, rows 0..63 of each stage;
                                      KK0/KK1 then QK0/QK1 as one M = 2 b_t GEMM with tinv_source
                                      "compute")
  cg1 shared acc           64     <-- 1-stage ring: KS then U
  Y + U input / decayed-U input    64  <-- slot 0 = Y then U input, slot 1 = decayed U (b16)

``tinv_source`` selects the chunk factor's producer.  "gmem": the tiles of ``gdn_tinv_f16.py``;
the TMA warp TMA-loads one 8192 B tile per chunk into the T_inv ring at the V cadence, the MMA
warp issues Q K^T and compute group 0 runs the A epilogues (16 rows of each member per warp).
"compute": compute group 0 builds the factor itself from the fused [K;Q] K^T GEMM (the KK
epilogue, the blockwise inverse over a chunk pair, the beta scaling) and the gate warp loads beta;
no T pass runs.

Warp assignments (12 warps = 384 threads):
  warps 0-3     : compute group 0 - T-pairwise x2, A epilogue x2 (+ KK epilogue x2 and the
                                    pair inverse with tinv_source "compute")
  warps 4-7     : compute group 1 - state restage/rescale, Y = V - K*state,
                                    state*Q_epi, U_epi, QKV_epilogue
  warp  8       : Gate loads (+ Beta loads with tinv_source "compute")
  warp  9       : TMA load warp  - loads Q, K, V
  warp  10      : MMA warp       - QK (or fused KK/QK) pairs + K*state/Q*state/U/QKV/KV per chunk;
                                    TMEM lifecycle
  warp  11      : epilogue warp  - O then checkpoint TMA stores
"""

from dataclasses import dataclass, replace
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.experimental.cuda.tensor_map as tma
import cutlass.experimental.primitives as nvvm
from cutlass import cute

from ..common.blockwise_inverse import (
    blockwise_diagonal_8x8_to_16x16,
    blockwise_diagonal_16x16_to_32x32,
    blockwise_diagonal_32x32_to_64x64,
    invert_diagonal_NxN,
)
from ..common.paged_state import resolve_paged_state
from ..common.split_k import (
    ORDER_CAPACITY,
    ORDER_ELEMENTS,
    ORDER_THREADS,
    WORK_ITEM_FINAL_DST,
    decode_head,
    decode_work_item,
    order_body,
)
from ..common.thd import (
    TENSOR_MAP_QWORDS,
    emit_checkpoint_seq_descs,
    emit_seq_descs,
    emit_tile_seq_descs,
)

RCP_LN2 = 1.4426950408889634  # 1/ln(2): natural-log gates -> the kernel's log2 domain
from attn_gym._backends.cute.compat import SmemAllocator

from ..tile_dsl.barrier import (
    MBarrier,
    PipelineState,
    Producer,
    advance,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from ..tile_dsl.handles import MmaDesc, SmemTile, smem_data_ptr, tma_slice_runtime_desc
from ..tile_dsl.mma import desc_opaque, mma_ss, mma_ts_step
from ..tile_dsl.pointwise import (
    f16x2_to_f32,
    fadd2,
    fmul2,
    fp32_to_fp16,
    opaque_f32_zero,
    sub_f16x2,
)
from ..tile_dsl.swizzle import swizzle_box_offset_128b, swizzle_xor_128b
from ..tile_dsl.tma import (
    tma_load_tile,
    tma_store_commit,
    tma_store_tile,
    tma_store_wait,
    tma_tensormap_acquire,
)
from .gdn_prefill_config import CFG

USE_PDL = True
TINV_SOURCES = ("compute", "gmem")


class GdnPrefillBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_kq_ready: MBarrier
    mb_kq_done: MBarrier
    mb_v_ready: MBarrier
    mb_v_done: MBarrier

    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier
    mb_beta_ready: MBarrier
    mb_beta_done: MBarrier
    mb_gate_cumsum_done: MBarrier

    mb_state_acc_ready: MBarrier
    mb_o_acc_ready: MBarrier
    mb_o_final_acc_ready: MBarrier
    mb_o_state_scale_acc_done: MBarrier
    mb_cg0_acc_ready: MBarrier
    mb_cg0_acc_done: MBarrier

    mb_state_input_ready: MBarrier
    mb_y_input_ready: MBarrier
    mb_u_input_ready: MBarrier
    mb_decay_u_input_ready: MBarrier

    mb_t_inv_ready: MBarrier
    mb_t_inv_done: MBarrier
    mb_a_ready: MBarrier
    mb_a_done: MBarrier

    mb_k_state_acc_ready: MBarrier
    mb_u_acc_ready: MBarrier

    mb_o_tmastg_ready: MBarrier
    mb_o_tmastg_done: MBarrier

    mb_checkpoint_tmastg_ready: MBarrier
    mb_checkpoint_tmastg_done: MBarrier

    mb_tmem_done: MBarrier

    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier


def make_bars(cfg) -> GdnPrefillBars:
    """GdnPrefillBars factory."""
    ONE_LANE = 1
    MMA_ARRIVERS = len([cfg.tcgen05_mma_warp_id])
    KQ_RELEASE_SITES = 1
    GATE_WARP = cfg.threads_per_warp * len([cfg.load_gate_warp_id])
    EPI_WARP = cfg.threads_per_warp * len([cfg.epilogue_warp_id])
    cfg.threads_per_warp * len([cfg.tma_qkv_warp_id])
    CG0_THREADS = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    CG1_THREADS = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    CG0_PLUS_CG1 = CG0_THREADS + CG1_THREADS

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=16)

    return GdnPrefillBars(
        mb_kq_ready=MBarrier(
            alloc(cfg.smem_kq_stages),
            try_wait=True,
            stages=cfg.smem_kq_stages,
            init_count=ONE_LANE,
            producer=Producer.TMA_LOAD,
        ),
        mb_kq_done=MBarrier(
            alloc(cfg.smem_kq_stages),
            try_wait=True,
            stages=cfg.smem_kq_stages,
            init_count=KQ_RELEASE_SITES,
            producer=Producer.MMA_COMMIT,
        ),
        mb_v_ready=MBarrier(
            alloc(cfg.smem_v_stages),
            try_wait=True,
            stages=cfg.smem_v_stages,
            init_count=ONE_LANE,
            producer=Producer.TMA_LOAD,
        ),
        mb_v_done=MBarrier(
            alloc(cfg.smem_v_stages),
            try_wait=True,
            stages=cfg.smem_v_stages,
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_gate_ready=MBarrier(
            alloc(cfg.smem_gate_stages),
            try_wait=True,
            stages=cfg.smem_gate_stages,
            init_count=GATE_WARP,
            producer=Producer.THREAD,
        ),
        mb_gate_done=MBarrier(
            alloc(cfg.smem_gate_stages),
            try_wait=True,
            stages=cfg.smem_gate_stages,
            init_count=CG0_PLUS_CG1,
            producer=Producer.THREAD,
        ),
        mb_beta_ready=MBarrier(
            alloc(cfg.smem_beta_stages),
            try_wait=True,
            stages=cfg.smem_beta_stages,
            init_count=GATE_WARP,
            producer=Producer.THREAD,
        ),
        mb_beta_done=MBarrier(
            alloc(cfg.smem_beta_stages),
            try_wait=True,
            stages=cfg.smem_beta_stages,
            init_count=CG0_THREADS,
            producer=Producer.THREAD,
        ),
        mb_gate_cumsum_done=MBarrier(
            alloc(cfg.smem_gate_stages),
            try_wait=True,
            stages=cfg.smem_gate_stages,
            init_count=(CG0_THREADS if cfg.tinv_source == "gmem" else CG0_THREADS // 2)
            + CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_state_acc_ready=MBarrier(
            alloc(cfg.tmem_state_acc_stages),
            try_wait=True,
            stages=cfg.tmem_state_acc_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_o_acc_ready=MBarrier(
            alloc(cfg.tmem_q_state_acc_stages),
            try_wait=True,
            stages=cfg.tmem_q_state_acc_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_o_final_acc_ready=MBarrier(
            alloc(cfg.tmem_q_state_acc_stages),
            try_wait=True,
            stages=cfg.tmem_q_state_acc_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_o_state_scale_acc_done=MBarrier(
            alloc(cfg.tmem_q_state_acc_stages),
            try_wait=True,
            stages=cfg.tmem_q_state_acc_stages,
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_cg0_acc_ready=MBarrier(
            alloc(cfg.tmem_cg0_acc_stages),
            try_wait=True,
            stages=cfg.tmem_cg0_acc_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_cg0_acc_done=MBarrier(
            alloc(cfg.tmem_cg0_acc_stages),
            try_wait=True,
            stages=cfg.tmem_cg0_acc_stages,
            init_count=CG0_THREADS if cfg.tinv_source == "gmem" else CG0_THREADS // 2,
            producer=Producer.THREAD,
        ),
        mb_state_input_ready=MBarrier(
            alloc(cfg.tmem_state_input_stages),
            try_wait=True,
            stages=cfg.tmem_state_input_stages,
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_y_input_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_u_input_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_decay_u_input_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_t_inv_ready=MBarrier(
            alloc(cfg.smem_t_inv_stages),
            try_wait=True,
            stages=cfg.smem_t_inv_stages,
            init_count=ONE_LANE if cfg.tinv_source == "gmem" else CG0_THREADS,
            producer=Producer.TMA_LOAD if cfg.tinv_source == "gmem" else Producer.THREAD,
        ),
        mb_t_inv_done=MBarrier(
            alloc(cfg.smem_t_inv_stages),
            try_wait=True,
            stages=cfg.smem_t_inv_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_a_ready=MBarrier(
            alloc(cfg.smem_a_stages),
            try_wait=True,
            stages=cfg.smem_a_stages,
            init_count=CG0_THREADS if cfg.tinv_source == "gmem" else CG0_THREADS // 2,
            producer=Producer.THREAD,
        ),
        mb_a_done=MBarrier(
            alloc(cfg.smem_a_stages),
            try_wait=True,
            stages=cfg.smem_a_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_k_state_acc_ready=MBarrier(
            alloc(1),
            try_wait=True,
            stages=1,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_u_acc_ready=MBarrier(
            alloc(1),
            try_wait=True,
            stages=1,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_o_tmastg_ready=MBarrier(
            alloc(cfg.smem_o_stages),
            try_wait=True,
            stages=cfg.smem_o_stages,
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_o_tmastg_done=MBarrier(
            alloc(cfg.smem_o_stages),
            try_wait=True,
            stages=cfg.smem_o_stages,
            init_count=EPI_WARP,
            producer=Producer.THREAD,
        ),
        mb_checkpoint_tmastg_ready=MBarrier(
            alloc(cfg.smem_checkpoint_stages),
            try_wait=True,
            stages=cfg.smem_checkpoint_stages,
            init_count=len(cfg.compute_group_1_warp_ids),
            producer=Producer.THREAD,
        ),
        mb_checkpoint_tmastg_done=MBarrier(
            alloc(cfg.smem_checkpoint_stages),
            try_wait=True,
            stages=cfg.smem_checkpoint_stages,
            init_count=EPI_WARP,
            producer=Producer.THREAD,
        ),
        mb_tmem_done=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_scheduler_ready=MBarrier(
            alloc(cfg.scheduler_stages),
            try_wait=True,
            stages=cfg.scheduler_stages,
            init_count=1,
            producer=Producer.THREAD,
        ),
        mb_scheduler_done=MBarrier(
            alloc(cfg.scheduler_stages),
            try_wait=True,
            stages=cfg.scheduler_stages,
            init_count=cfg.threads_per_cta // cfg.threads_per_warp
            - 1,  # every warp but the TMA producer
            producer=Producer.THREAD,
        ),
    )


@cute.jit
def scheduler_publish_next(
    cfg, bars, sScheduler, mScheduler, scheduler_state, num_ctas, elect_one
):
    """TMA-LDG-warp side: pull the next tile off the global ticket, publish it."""
    bars.mb_scheduler_done[scheduler_state.idx].wait(scheduler_state.phase)
    if elect_one:
        fetched = cutlass.Int32(
            nvvm.atomicrmw(
                "add", mScheduler.iterator, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu"
            )
        )
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
    cu_seqlens,
    mWorkItems,
    checkpoint_every_n_tokens,
    tidx,
    sO_raw,
    sCheckpoint_raw,
    desc_o_base,
    desc_checkpoint_base,
    sScheduler,
    bars,
):
    """Epilogue warp role (warp 11): persistent scheduler loop issuing the
    per-chunk O and state-checkpoint TMA stores."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    o_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)

    tidx % cfg.threads_per_warp

    elect_one = nvvm.elect_sync()
    tile_idx = cutlass.Int32(bidx)
    bytes_per_element = cfg.io_dtype.width // 8
    box_elems = 128 // bytes_per_element
    sO_tma = SmemTile(
        base=sO_raw,
        elems_per_stage=(cfg.o_cosize // cfg.smem_o_stages),
        stages=cfg.smem_o_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_v // box_elems,
        tma_granu_elems=box_elems,
        tma_subtile_stride_elems=cfg.b_t * box_elems,
    )
    if cutlass.const_expr(cfg.enable_checkpoints):
        checkpoint_box_elems = 64
        sCheckpoint_tma = SmemTile(
            base=sCheckpoint_raw,
            elems_per_stage=(cfg.checkpoint_cosize // cfg.smem_checkpoint_stages),
            stages=cfg.smem_checkpoint_stages,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=0,
            tma_loads_per_tile=cfg.d_k // checkpoint_box_elems,
            tma_granu_elems=checkpoint_box_elems,
            tma_subtile_stride_elems=cfg.d_v * checkpoint_box_elems,
        )
        checkpoint_store_cnt = cutlass.Int32(0)
        checkpoint_chunks = checkpoint_every_n_tokens // cutlass.Int32(cfg.b_t)
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)

    while tile_idx < total_tiles:
        (
            batch_idx,
            head_idx,
            _batch_start,
            _batch_end,
            _batch_seqlen,
            _batch_num_chunks,
            write_start,
            write_end,
            compute_start,
            _compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        n_local = write_end - compute_start

        head_o, v_offset = decode_head(cfg, head_idx)
        slot = batch_idx * desc_qwords
        desc_o_slot = (desc_o_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_o_slot)
        if cutlass.const_expr(cfg.enable_checkpoints):
            desc_checkpoint_slot = (desc_checkpoint_base + slot).tospace(
                cutlass.AddressSpace.generic
            )
            checkpoint_coord = (
                write_start + checkpoint_chunks - cutlass.Int32(1)
            ) // checkpoint_chunks
            checkpoint_mod = (compute_start + cutlass.Int32(1)) % checkpoint_chunks
            if elect_one:
                tma_tensormap_acquire(desc_checkpoint_slot)

        if n_local > 0:
            if cutlass.const_expr(cfg.enable_checkpoints):
                if write_start == 0:
                    checkpoint_stage = checkpoint_store_cnt % cfg.smem_checkpoint_stages
                    checkpoint_phase = (
                        checkpoint_store_cnt // cfg.smem_checkpoint_stages
                    ) & cutlass.Int32(1)
                    bars.mb_checkpoint_tmastg_ready[checkpoint_stage].wait(checkpoint_phase)
                    checkpoint_slice = tma_slice_runtime_desc(
                        desc_checkpoint_slot, cutlass.Int32(0), v_offset, checkpoint_coord, head_o
                    )
                    tma_store_tile(sCheckpoint_tma[checkpoint_stage], checkpoint_slice)
                    tma_store_commit()
                    tma_store_wait(0)
                    bars.mb_checkpoint_tmastg_done[checkpoint_stage].arrive()
                    checkpoint_coord += 1
                    checkpoint_store_cnt = checkpoint_store_cnt + 1
            for local_idx in cutlass.range(n_local):
                chunk_idx = compute_start + local_idx

                did_o = cutlass.Int32(0)
                o_participates = cutlass.Boolean(True)
                o_idx = cutlass.Int32(0)
                if o_participates:
                    o_idx = o_index.idx
                    bars.mb_o_tmastg_ready[o_idx].wait(o_index.phase)
                    o_index = advance(o_index, cfg.smem_o_stages)

                if o_participates and chunk_idx >= write_start and chunk_idx < write_end:
                    tok_coord = chunk_idx * cutlass.Int32(cfg.b_t)
                    o_slice = tma_slice_runtime_desc(desc_o_slot, v_offset, head_o, tok_coord)
                    tma_store_tile(sO_tma[o_idx], o_slice)
                    tma_store_commit()
                    did_o = cutlass.Int32(1)

                did_checkpoint = cutlass.Int32(0)
                if cutlass.const_expr(cfg.enable_checkpoints):
                    checkpoint_stage = checkpoint_store_cnt % cfg.smem_checkpoint_stages
                    checkpoint_phase = (
                        checkpoint_store_cnt // cfg.smem_checkpoint_stages
                    ) & cutlass.Int32(1)
                    if chunk_idx >= write_start - 1 and chunk_idx < write_end - 1:
                        if checkpoint_mod == 0:
                            bars.mb_checkpoint_tmastg_ready[checkpoint_stage].wait(
                                checkpoint_phase
                            )
                            checkpoint_slice = tma_slice_runtime_desc(
                                desc_checkpoint_slot,
                                cutlass.Int32(0),
                                v_offset,
                                checkpoint_coord,
                                head_o,
                            )
                            tma_store_tile(sCheckpoint_tma[checkpoint_stage], checkpoint_slice)
                            tma_store_commit()
                            checkpoint_coord += 1
                            did_checkpoint = cutlass.Int32(1)
                    checkpoint_mod = checkpoint_mod + cutlass.Int32(1)
                    checkpoint_mod = (
                        cutlass.Int32(0) if checkpoint_mod == checkpoint_chunks else checkpoint_mod
                    )

                if cutlass.const_expr(cfg.enable_checkpoints):
                    if did_o == 1 and did_checkpoint == 1:
                        tma_store_wait(1)
                        bars.mb_o_tmastg_done[o_idx].arrive()
                        tma_store_wait(0)
                        bars.mb_checkpoint_tmastg_done[checkpoint_stage].arrive()
                        checkpoint_store_cnt = checkpoint_store_cnt + 1
                    if did_o == 1 and did_checkpoint == 0:
                        tma_store_wait(0)
                        bars.mb_o_tmastg_done[o_idx].arrive()
                    if did_o == 0:
                        if did_checkpoint == 1:
                            tma_store_wait(0)
                            bars.mb_checkpoint_tmastg_done[checkpoint_stage].arrive()
                            checkpoint_store_cnt = checkpoint_store_cnt + 1
                        if o_participates:
                            bars.mb_o_tmastg_done[o_idx].arrive()
                else:
                    if did_o == 1:
                        tma_store_wait(0)
                    if o_participates:
                        bars.mb_o_tmastg_done[o_idx].arrive()

        tile_idx, scheduler_state = scheduler_next_tile(
            cfg, bars, sScheduler, scheduler_state, elect_one
        )


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
    mBeta,
    sCumsumlog,
    sCumprod,
    sBeta,
    sScheduler,
    bars,
):
    """Gate producer (warp 8): persistent scheduler loop + the cumsum/cumprod
    chunk loads, and the Beta chunk loads with tinv_source "compute"."""

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    elect_one = nvvm.elect_sync()

    gate_index = PipelineState.start(phase=1)
    beta_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=0)

    lane_idx = tidx % cfg.threads_per_warp

    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        (
            _batch_idx,
            head_idx,
            batch_start,
            batch_end,
            _batch_seqlen,
            _batch_num_chunks,
            _write_start,
            write_end,
            compute_start,
            _compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        head_o, _v_offset = decode_head(cfg, head_idx)
        n_local = write_end - compute_start
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

                oob_neutral = (
                    cutlass.Float32(0.0)
                    if cutlass.const_expr(cfg.log_gate)
                    else cutlass.Float32(1.0)
                )
                toks = [
                    chunk_offset + lane_idx + col * cfg.threads_per_warp for col in range(n_cols)
                ]
                pos_valid = [tok < batch_end for tok in toks]
                gate_vals = [
                    gGateSeq[cutlass.min(tok, batch_end - 1)].to(cutlass.Float32)
                    if valid
                    else oob_neutral
                    for tok, valid in zip(toks, pos_valid)
                ]

                if cutlass.const_expr(
                    cfg.tinv_source == "compute" and mBeta.element_type != cutlass.Float32
                ):
                    gBeta = cute.domain_offset((chunk_offset,), mBeta[None, head_o])
                    beta_raw = [
                        gBeta[lane_idx + col * cfg.threads_per_warp].to(cutlass.Float32)
                        if pos_valid[col]
                        else cutlass.Float32(0.0)
                        for col in range(n_cols)
                    ]
                if cutlass.const_expr(cfg.log_gate):
                    rcp_ln2 = opaque_f32_zero() + cutlass.Float32(RCP_LN2)
                    for col in cutlass.range_constexpr(0, n_cols, 2):
                        gate_vals[col], gate_vals[col + 1] = fmul2(
                            gate_vals[col], gate_vals[col + 1], rcp_ln2, rcp_ln2
                        )
                else:
                    floor = cutlass.Float32(1e-10)
                    for col in cutlass.range_constexpr(0, n_cols, 2):
                        shifted_lo, shifted_hi = fadd2(
                            gate_vals[col], gate_vals[col + 1], floor, floor
                        )
                        gate_vals[col] = cute.math.log2(shifted_lo, fastmath=True)
                        gate_vals[col + 1] = cute.math.log2(shifted_hi, fastmath=True)
                for offset in [1, 2, 4, 8, 16]:
                    for col in cutlass.range_constexpr(n_cols):
                        n = nvvm.shfl_sync(
                            0xFFFFFFFF, gate_vals[col], offset, 0, kind=nvvm.Shfl.UP
                        )
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

                # ---- Beta load: GMEM -> SMEM (per-element cp.async) ------------------
                if cutlass.const_expr(cfg.tinv_source == "compute"):
                    gBeta = cute.domain_offset((chunk_offset,), mBeta[None, head_o])
                    beta_idx = beta_index.idx
                    bars.mb_beta_done[beta_idx].wait(beta_index.phase)
                    beta_index = advance(beta_index, cfg.smem_beta_stages)
                    if cutlass.const_expr(mBeta.element_type != cutlass.Float32):
                        for col in cutlass.range_constexpr(n_cols):
                            pos = lane_idx + col * cfg.threads_per_warp
                            beta_value = beta_raw[col]
                            sBeta[pos, 0, beta_idx] = beta_value
                        bars.mb_beta_ready[beta_idx].arrive()
                    else:
                        for col in cutlass.range_constexpr(n_cols):
                            pos = lane_idx + col * cfg.threads_per_warp
                            src = gBeta.iterator + gBeta.layout((pos,))
                            dst = sBeta.iterator + sBeta.layout((pos, 0, beta_idx))
                            cp_size = cutlass.Int32(4) * cutlass.Int32(pos_valid[col])
                            nvvm.cp_async_shared_global(
                                dst, src, 4, nvvm.LoadCacheModifier.CA, cp_size=cp_size
                            )
                        nvvm.cp_async_mbarrier_arrive(
                            bars.mb_beta_ready[beta_idx].smem_ptr, noinc=True
                        )
        tile_idx, scheduler_state = scheduler_next_tile(
            cfg, bars, sScheduler, scheduler_state, elect_one
        )

    for _ in range(cfg.smem_gate_stages):
        bars.mb_gate_done[gate_index.idx].wait(gate_index.phase)
        gate_index = advance(gate_index, cfg.smem_gate_stages)
    if cutlass.const_expr(cfg.tinv_source == "compute"):
        for _ in range(cfg.smem_beta_stages):
            bars.mb_beta_done[beta_index.idx].wait(beta_index.phase)
            beta_index = advance(beta_index, cfg.smem_beta_stages)


@cute.jit
def issue_cg0_gemm(
    cfg,
    bmm_kq_k_desc,
    bmm_q_k_desc,
    desc_kq_cg0,
    member: int,
    acc_cg0,
    KQ_SEG,
    KQ_BOX,
    KQ_SUBTILES,
    KQ_STEPS,
):
    """Compute group 0's shared-accumulator GEMM for one chunk of a KQ stage
    (member 0 stages [K;Q], member 1 [Q;K]): the fused [K;Q](S) @ K^T at
    M = 2 b_t, or Q(S) @ K^T alone at M = b_t when the chunk factor comes from GMEM."""
    k_off = 0 if member == 0 else KQ_BOX
    if cutlass.const_expr(cfg.tinv_source == "gmem"):
        q_off = KQ_BOX if member == 0 else 0
        for subtile in cutlass.range_constexpr(KQ_SUBTILES):
            subtile_offset = subtile * KQ_SEG
            mma_ss(
                bmm_q_k_desc,
                desc_kq_cg0 + (q_off + subtile_offset),
                desc_kq_cg0 + (k_off + subtile_offset),
                acc_cg0,
                accumulate=subtile > 0,
                k_count=KQ_STEPS,
            )
    else:
        for subtile in cutlass.range_constexpr(KQ_SUBTILES):
            subtile_offset = subtile * KQ_SEG
            mma_ss(
                bmm_kq_k_desc,
                desc_kq_cg0 + subtile_offset,
                desc_kq_cg0 + (k_off + subtile_offset),
                acc_cg0,
                accumulate=subtile > 0,
                k_count=KQ_STEPS,
            )


@cute.jit
def tcgen05_mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    tmem_base_slot,
    sKQ,
    sKQ_trans,
    sTinv,
    sA,
    sScheduler,
    bars,
):
    """MMA issuer role (warp 10): persistent scheduler loop issuing every
    tcgen05 GEMM."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    o_acc_index = PipelineState.start(phase=1)
    o_state_scale_index = PipelineState.start(phase=0)
    kv_acc_index = PipelineState.start(phase=1)
    kq_index = PipelineState.start(phase=0)
    cg0_acc_index = PipelineState.start(phase=1)
    kq_cg0_index = PipelineState.start(phase=0)
    tinv_index = PipelineState.start(phase=0)
    a_index = PipelineState.start(phase=0)
    state_input_index = PipelineState.start(phase=0)
    y_input_ready = PipelineState.start(phase=0)
    u_input_ready = PipelineState.start(phase=0)
    decay_u_input_ready = PipelineState.start(phase=0)

    elect_one = nvvm.elect_sync()

    nvvm.tcgen05_alloc(tmem_base_slot, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads
    )

    # ---- chunk-invariant GEMM descriptors --------------------------------------------
    instruction_descriptor_qk = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=2 * cfg.b_t,
    )
    bmm_kq_k_desc = MmaDesc(
        M=2 * cfg.b_t,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=cfg.io_dtype.width // 8,
        bpe_b=cfg.io_dtype.width // 8,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_qk,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    instruction_descriptor_q_k = nvvm.Tcgen05InstrDesc.build(
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
        bpe_a=cfg.io_dtype.width // 8,
        bpe_b=cfg.io_dtype.width // 8,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_q_k,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bytes_per_element = cfg.io_dtype.width // 8
    instruction_descriptor_q_state = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_state_k_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_q_state,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_state_q_desc = bmm_state_k_desc
    instruction_descriptor_qkv_ts = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_u_a_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_qkv_ts,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_y_t_inv_desc = bmm_u_a_desc
    instruction_descriptor_kv = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_k,
        m_dim=cfg.d_v,
        b_major=1,
    )
    bmm_decay_u_k_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.d_k,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_kv,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    KQ_SEG = (2 * cfg.b_t * 64 * bytes_per_element) >> 4
    KQ_BOX = (cfg.b_t * 64 * bytes_per_element) >> 4
    KQ_SUBTILES = bmm_kq_k_desc.num_subtiles
    KQ_STEPS = bmm_kq_k_desc.steps_per_subtile
    KQ_A_SEG = KQ_STEPS * bmm_state_k_desc.tmem_advance_A

    ACC_STAGE_COLS = cfg.b_t
    KV_ACC_STAGE_COLS = cfg.d_k
    STATE_INP_STAGE_COLS = cfg.d_k // 2
    INP_SLOT_COLS = cfg.b_t // 2

    tmem_cg0_acc_col = (tmem_base_slot.load() & 0xFFFF) + cfg.tmem_cg0_acc_offset

    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        (
            _batch_idx,
            _head_idx,
            _batch_start,
            _batch_end,
            _batch_seqlen,
            _batch_num_chunks,
            _write_start,
            write_end,
            compute_start,
            _compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        n_local = write_end - compute_start

        # ---- [K;Q] K^T or Q K^T of the first pair ------------------------------------
        if n_local > 0:
            cg0_acc_idx = cg0_acc_index.idx
            bars.mb_cg0_acc_done[cg0_acc_idx].wait(cg0_acc_index.phase)
            cg0_acc_index = advance(cg0_acc_index, cfg.tmem_cg0_acc_stages)
            kq_cg0_idx = kq_cg0_index.idx
            bars.mb_kq_ready[kq_cg0_idx].wait(kq_cg0_index.phase)
            kq_cg0_index = advance(kq_cg0_index, cfg.smem_kq_stages)
            desc_kq_cg0 = desc_opaque(sKQ[kq_cg0_idx].desc())
            acc_cg0 = nvvm.make_tmem_ptr(tmem_cg0_acc_col + cg0_acc_idx * cfg.b_t, cutlass.Float32)
            issue_cg0_gemm(
                cfg,
                bmm_kq_k_desc,
                bmm_q_k_desc,
                desc_kq_cg0,
                0,
                acc_cg0,
                KQ_SEG,
                KQ_BOX,
                KQ_SUBTILES,
                KQ_STEPS,
            )
            if elect_one:
                bars.mb_cg0_acc_ready[cg0_acc_idx].arrive(cta_group=1)
        if n_local > 1:
            cg0_acc_idx = cg0_acc_index.idx
            bars.mb_cg0_acc_done[cg0_acc_idx].wait(cg0_acc_index.phase)
            cg0_acc_index = advance(cg0_acc_index, cfg.tmem_cg0_acc_stages)
            kq_cg0_idx = kq_cg0_index.idx
            bars.mb_kq_ready[kq_cg0_idx].wait(kq_cg0_index.phase)
            kq_cg0_index = advance(kq_cg0_index, cfg.smem_kq_stages)
            desc_kq_cg0 = desc_opaque(sKQ[kq_cg0_idx].desc())
            acc_cg0 = nvvm.make_tmem_ptr(tmem_cg0_acc_col + cg0_acc_idx * cfg.b_t, cutlass.Float32)
            issue_cg0_gemm(
                cfg,
                bmm_kq_k_desc,
                bmm_q_k_desc,
                desc_kq_cg0,
                1,
                acc_cg0,
                KQ_SEG,
                KQ_BOX,
                KQ_SUBTILES,
                KQ_STEPS,
            )
            if elect_one:
                bars.mb_cg0_acc_ready[cg0_acc_idx].arrive(cta_group=1)

        for peel in cutlass.range_constexpr(1 if cfg.use_initial_state else 2):
            have_state = (
                cutlass.Boolean(True)
                if cutlass.const_expr(cfg.use_initial_state)
                else cutlass.const_expr(peel == 1)
            )
            peel_stop = (
                n_local
                if cutlass.const_expr(peel == 1 or cfg.use_initial_state)
                else cutlass.min(n_local, 1)
            )
            for local_idx in cutlass.range(peel, peel_stop, 1):
                kq_idx = kq_index.idx
                member = local_idx & 1
                state_input_idx = state_input_index.idx
                q_state_acc_idx = o_acc_index.idx
                tinv_idx = tinv_index.idx
                a_idx = a_index.idx
                o_scale_idx = o_state_scale_index.idx
                kv_acc_idx = kv_acc_index.idx
                kq_member_off = member * KQ_BOX
                desc_k = desc_opaque(sKQ[kq_idx].desc() + kq_member_off)
                desc_q = desc_opaque(sKQ[kq_idx].desc() + (KQ_BOX - kq_member_off))
                desc_tinv = desc_opaque(sTinv[tinv_idx].desc())
                desc_a = desc_opaque(sA[a_idx].desc())
                desc_k_trans = desc_opaque(sKQ_trans[kq_idx].desc() + kq_member_off)
                tmem_col = tmem_base_slot.load() & 0xFFFF
                tmem_input_col = tmem_col + cfg.tmem_y_decay_u_input_offset
                y_input_ptr = nvvm.make_tmem_ptr(tmem_input_col, cutlass.Int8)
                u_input_ptr = nvvm.make_tmem_ptr(tmem_input_col, cutlass.Int8)
                decay_u_input_ptr = nvvm.make_tmem_ptr(
                    tmem_input_col + INP_SLOT_COLS, cutlass.Int8
                )
                k_state_acc_ptr = nvvm.make_tmem_ptr(
                    tmem_col + cfg.tmem_cg1_acc_offset, cutlass.Float32
                )
                u_acc_ptr = k_state_acc_ptr
                state_a_ptr = nvvm.make_tmem_ptr(
                    tmem_col
                    + cfg.tmem_state_input_offset
                    + state_input_idx * STATE_INP_STAGE_COLS,
                    cutlass.Int8,
                )
                q_state_acc_ptr = nvvm.make_tmem_ptr(
                    tmem_col + cfg.tmem_q_state_acc_offset + q_state_acc_idx * ACC_STAGE_COLS,
                    cutlass.Float32,
                )
                qkv_acc_ptr = nvvm.make_tmem_ptr(
                    tmem_col + cfg.tmem_q_state_acc_offset + o_scale_idx * ACC_STAGE_COLS,
                    cutlass.Float32,
                )
                state_acc_ptr = nvvm.make_tmem_ptr(
                    tmem_col + cfg.tmem_state_acc_offset + kv_acc_idx * KV_ACC_STAGE_COLS,
                    cutlass.Float32,
                )

                kq_index = advance(kq_index, cfg.smem_kq_stages)

                # ---- Q K^T lookahead (member 1) = Q(S) @ K^T -------------------------
                if member == 1:
                    if local_idx + 2 < n_local:
                        cg0_acc_idx = cg0_acc_index.idx
                        bars.mb_cg0_acc_done[cg0_acc_idx].wait(cg0_acc_index.phase)
                        cg0_acc_index = advance(cg0_acc_index, cfg.tmem_cg0_acc_stages)
                        kq_cg0_idx = kq_cg0_index.idx
                        bars.mb_kq_ready[kq_cg0_idx].wait(kq_cg0_index.phase)
                        kq_cg0_index = advance(kq_cg0_index, cfg.smem_kq_stages)
                        desc_kq_cg0 = desc_opaque(sKQ[kq_cg0_idx].desc())
                        acc_cg0 = nvvm.make_tmem_ptr(
                            tmem_cg0_acc_col + cg0_acc_idx * cfg.b_t, cutlass.Float32
                        )
                        issue_cg0_gemm(
                            cfg,
                            bmm_kq_k_desc,
                            bmm_q_k_desc,
                            desc_kq_cg0,
                            1,
                            acc_cg0,
                            KQ_SEG,
                            KQ_BOX,
                            KQ_SUBTILES,
                            KQ_STEPS,
                        )
                        if elect_one:
                            bars.mb_cg0_acc_ready[cg0_acc_idx].arrive(cta_group=1)

                # ---- k state^T = state^T(T) @ K^T ------------------------------------
                if have_state:
                    bars.mb_state_input_ready[state_input_idx].wait(state_input_index.phase)
                    state_input_index = advance(state_input_index, cfg.tmem_state_input_stages)

                    for subtile in cutlass.range_constexpr(KQ_SUBTILES):
                        subtile_offset = subtile * KQ_SEG
                        state_a_subtile = state_a_ptr.subview(subtile * KQ_A_SEG)
                        for k in cutlass.range_constexpr(KQ_STEPS):
                            mma_ts_step(
                                bmm_state_k_desc,
                                state_a_subtile,
                                desc_k + subtile_offset,
                                k_state_acc_ptr,
                                k,
                                cutlass.Boolean(subtile > 0 or k > 0),
                            )
                    if elect_one:
                        bars.mb_k_state_acc_ready[0].arrive(cta_group=1)

                # ---- q state^T = state^T(T) @ Q^T ------------------------------------
                block_head = cutlass.Boolean(True)
                if block_head:
                    o_acc_index = advance(o_acc_index, cfg.tmem_q_state_acc_stages)
                if have_state and block_head:
                    for subtile in cutlass.range_constexpr(KQ_SUBTILES):
                        subtile_offset = subtile * KQ_SEG
                        state_a_subtile = state_a_ptr.subview(subtile * KQ_A_SEG)
                        for k in cutlass.range_constexpr(KQ_STEPS):
                            mma_ts_step(
                                bmm_state_q_desc,
                                state_a_subtile,
                                desc_q + subtile_offset,
                                q_state_acc_ptr,
                                k,
                                cutlass.Boolean(subtile > 0 or k > 0),
                            )
                    if elect_one:
                        bars.mb_o_acc_ready[q_state_acc_idx].arrive(cta_group=1)

                # ---- U^T = Y^T(T) @ (T^-1)^T -----------------------------------------
                bars.mb_t_inv_ready[tinv_idx].wait(tinv_index.phase)
                tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
                bars.mb_y_input_ready[0].wait(y_input_ready.phase)
                y_input_ready = advance(y_input_ready, 1)
                for k in cutlass.range_constexpr(cfg.b_t // 16):
                    mma_ts_step(
                        bmm_y_t_inv_desc,
                        y_input_ptr,
                        desc_tinv,
                        u_acc_ptr,
                        k,
                        cutlass.Boolean(k > 0),
                    )
                if elect_one:
                    bars.mb_u_acc_ready[0].arrive(cta_group=1)
                    bars.mb_t_inv_done[tinv_idx].arrive(cta_group=1)

                # ---- Q K^T lookahead (member 0) = Q(S) @ K^T -------------------------
                if member == 0:
                    if local_idx + 2 < n_local:
                        cg0_acc_idx = cg0_acc_index.idx
                        bars.mb_cg0_acc_done[cg0_acc_idx].wait(cg0_acc_index.phase)
                        cg0_acc_index = advance(cg0_acc_index, cfg.tmem_cg0_acc_stages)
                        kq_cg0_idx = kq_cg0_index.idx
                        bars.mb_kq_ready[kq_cg0_idx].wait(kq_cg0_index.phase)
                        kq_cg0_index = advance(kq_cg0_index, cfg.smem_kq_stages)
                        desc_kq_cg0 = desc_opaque(sKQ[kq_cg0_idx].desc())
                        acc_cg0 = nvvm.make_tmem_ptr(
                            tmem_cg0_acc_col + cg0_acc_idx * cfg.b_t, cutlass.Float32
                        )
                        issue_cg0_gemm(
                            cfg,
                            bmm_kq_k_desc,
                            bmm_q_k_desc,
                            desc_kq_cg0,
                            0,
                            acc_cg0,
                            KQ_SEG,
                            KQ_BOX,
                            KQ_SUBTILES,
                            KQ_STEPS,
                        )
                        if elect_one:
                            bars.mb_cg0_acc_ready[cg0_acc_idx].arrive(cta_group=1)

                # ---- O^T += U input^T(T) @ A^T ---------------------------------------
                bars.mb_a_ready[a_idx].wait(a_index.phase)
                a_index = advance(a_index, cfg.smem_a_stages)
                qkv_block_head = cutlass.Boolean(True)
                if have_state and qkv_block_head:
                    bars.mb_o_state_scale_acc_done[o_scale_idx].wait(o_state_scale_index.phase)
                    o_state_scale_index = advance(o_state_scale_index, cfg.tmem_q_state_acc_stages)
                bars.mb_u_input_ready[0].wait(u_input_ready.phase)
                u_input_ready = advance(u_input_ready, 1)
                qkv_accumulate = have_state if qkv_block_head else cutlass.Boolean(True)
                for k in cutlass.range_constexpr(cfg.b_t // 16):
                    mma_ts_step(
                        bmm_u_a_desc,
                        u_input_ptr,
                        desc_a,
                        qkv_acc_ptr,
                        k,
                        cutlass.Boolean(True) if cutlass.const_expr(k > 0) else qkv_accumulate,
                    )
                if elect_one:
                    bars.mb_a_done[a_idx].arrive(cta_group=1)
                    block_tail = cutlass.Boolean(True)
                    if block_tail:
                        bars.mb_o_final_acc_ready[o_scale_idx].arrive(cta_group=1)

                # ---- state^T += decayed U^T(T) @ K -----------------------------------
                bars.mb_decay_u_input_ready[0].wait(decay_u_input_ready.phase)
                decay_u_input_ready = advance(decay_u_input_ready, 1)
                kv_acc_index = advance(kv_acc_index, cfg.tmem_state_acc_stages)
                for k in cutlass.range_constexpr(cfg.b_t // 16):
                    mma_ts_step(
                        bmm_decay_u_k_desc,
                        decay_u_input_ptr,
                        desc_k_trans,
                        state_acc_ptr,
                        k,
                        cutlass.Boolean(True) if cutlass.const_expr(k > 0) else have_state,
                    )
                if elect_one:
                    bars.mb_state_acc_ready[kv_acc_idx].arrive(cta_group=1)
                    bars.mb_kq_done[kq_idx].arrive(cta_group=1)

        tile_idx, scheduler_state = scheduler_next_tile(
            cfg, bars, sScheduler, scheduler_state, elect_one
        )

    bars.mb_tmem_done[0].wait(0)
    nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
    nvvm.tcgen05_dealloc(
        nvvm.make_tmem_ptr(tmem_base_slot.load() & 0xFFFF, cutlass.Int8),
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
    sKQ_raw,
    sV_raw,
    sTinv_raw,
    desc_tinv_base,
    tidx,
    desc_q_base,
    desc_k_base,
    desc_v_base,
    mScheduler,
    sScheduler,
    bars,
    q_ratio,
    k_ratio,
    v_ratio,
):
    """TMA-LDG warp role (warp 9): persistent scheduler loop + per-chunk
    Q/K/V G->S TMA loads, plus the chunk-factor tile TMA loads at the V
    cadence when the chunk factor comes from GMEM."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    kq_index = PipelineState.start(phase=1)
    v_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=1)
    if cutlass.const_expr(cfg.tinv_source == "gmem"):
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
    tidx % cfg.threads_per_warp
    tile_idx = cutlass.Int32(bidx)
    bytes_per_element = cfg.io_dtype.width // 8
    box_elems = 128 // bytes_per_element
    bt = cfg.b_t
    kq_stage_elements = cfg.kq_cosize // cfg.smem_kq_stages
    kq_box_elements = kq_stage_elements // (2 * (cfg.d_k // box_elems))
    sKQ_lo_tma = SmemTile(
        base=sKQ_raw,
        elems_per_stage=kq_stage_elements,
        stages=cfg.smem_kq_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_k // box_elems,
        tma_granu_elems=box_elems,
        tma_subtile_stride_elems=2 * bt * box_elems,
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
        (
            batch_idx,
            head_idx,
            _batch_start,
            _batch_end,
            _batch_seqlen,
            _batch_num_chunks,
            _write_start,
            write_end,
            compute_start,
            _compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)

        head_o, v_offset = decode_head(cfg, head_idx)
        head_q = head_idx // q_ratio
        head_k = head_idx // k_ratio
        head_v = head_idx // v_ratio
        slot = batch_idx * desc_qwords
        desc_q_slot = (desc_q_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_v_slot)
        if cutlass.const_expr(cfg.tinv_source == "gmem"):
            desc_tinv_slot = (desc_tinv_base + slot).tospace(cutlass.AddressSpace.generic)
            if elect_one:
                tma_tensormap_acquire(desc_tinv_slot)

        if write_end > compute_start:
            kq_idx = kq_index.idx
            bars.mb_kq_done[kq_idx].wait(kq_index.phase)
            kq_index = advance(kq_index, cfg.smem_kq_stages)
            if elect_one:
                bars.mb_kq_ready[kq_idx].arrive(n_bytes=cfg.tma_kq_bytes)
            tok_coord = compute_start * cutlass.Int32(cfg.b_t)
            q_coord = tok_coord
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), head_k, tok_coord)
            kq_tile = sKQ_lo_tma[kq_idx]
            tma_load_tile(kq_tile, k_slice, bars.mb_kq_ready[kq_idx].smem_ptr)
            q_slice = tma_slice_runtime_desc(desc_q_slot, cutlass.Int32(0), head_q, q_coord)
            tma_load_tile(
                kq_tile.shifted(kq_box_elements), q_slice, bars.mb_kq_ready[kq_idx].smem_ptr
            )
            for chunk_idx in cutlass.range(compute_start + 1, write_end):
                tok_coord = chunk_idx * cutlass.Int32(cfg.b_t)

                # ---- K + Q load (interleaved) ----------------------------------------
                kq_idx = kq_index.idx
                bars.mb_kq_done[kq_idx].wait(kq_index.phase)
                kq_index = advance(kq_index, cfg.smem_kq_stages)
                if elect_one:
                    bars.mb_kq_ready[kq_idx].arrive(n_bytes=cfg.tma_kq_bytes)
                member = (chunk_idx - compute_start) & 1
                k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), head_k, tok_coord)
                kq_tile = sKQ_lo_tma[kq_idx]
                q_coord = tok_coord
                q_slice = tma_slice_runtime_desc(desc_q_slot, cutlass.Int32(0), head_q, q_coord)
                if member == 0:
                    tma_load_tile(kq_tile, k_slice, bars.mb_kq_ready[kq_idx].smem_ptr)
                    tma_load_tile(
                        kq_tile.shifted(kq_box_elements),
                        q_slice,
                        bars.mb_kq_ready[kq_idx].smem_ptr,
                    )
                else:
                    tma_load_tile(kq_tile, q_slice, bars.mb_kq_ready[kq_idx].smem_ptr)
                    tma_load_tile(
                        kq_tile.shifted(kq_box_elements),
                        k_slice,
                        bars.mb_kq_ready[kq_idx].smem_ptr,
                    )

                # ---- V load ----------------------------------------------------------
                v_idx = v_index.idx
                bars.mb_v_done[v_idx].wait(v_index.phase)
                v_index = advance(v_index, cfg.smem_v_stages)
                if elect_one:
                    bars.mb_v_ready[v_idx].arrive(n_bytes=cfg.tma_v_bytes)
                v_tok = (chunk_idx - 1) * cutlass.Int32(cfg.b_t)
                v_slice = tma_slice_runtime_desc(desc_v_slot, v_offset, head_v, v_tok)
                tma_load_tile(sV_tma[v_idx], v_slice, bars.mb_v_ready[v_idx].smem_ptr)

                # ---- chunk-factor tile load ------------------------------------------
                if cutlass.const_expr(cfg.tinv_source == "gmem"):
                    tinv_idx = tinv_index.idx
                    bars.mb_t_inv_done[tinv_idx].wait(tinv_index.phase)
                    tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
                    if elect_one:
                        bars.mb_t_inv_ready[tinv_idx].arrive(n_bytes=cfg.tma_tinv_bytes)
                    tinv_row = chunk_idx - cutlass.Int32(1)
                    tinv_slice = tma_slice_runtime_desc(
                        desc_tinv_slot, cutlass.Int32(0), cutlass.Int32(0), head_o, tinv_row
                    )
                    tma_load_tile(
                        sTinv_tma[tinv_idx], tinv_slice, bars.mb_t_inv_ready[tinv_idx].smem_ptr
                    )

            v_idx = v_index.idx
            bars.mb_v_done[v_idx].wait(v_index.phase)
            v_index = advance(v_index, cfg.smem_v_stages)
            if elect_one:
                bars.mb_v_ready[v_idx].arrive(n_bytes=cfg.tma_v_bytes)
            v_tok = (write_end - cutlass.Int32(1)) * cutlass.Int32(cfg.b_t)
            v_slice = tma_slice_runtime_desc(desc_v_slot, v_offset, head_v, v_tok)
            tma_load_tile(sV_tma[v_idx], v_slice, bars.mb_v_ready[v_idx].smem_ptr)
            if cutlass.const_expr(cfg.tinv_source == "gmem"):
                tinv_idx = tinv_index.idx
                bars.mb_t_inv_done[tinv_idx].wait(tinv_index.phase)
                tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
                if elect_one:
                    bars.mb_t_inv_ready[tinv_idx].arrive(n_bytes=cfg.tma_tinv_bytes)
                tinv_row = write_end - cutlass.Int32(1)
                tinv_slice = tma_slice_runtime_desc(
                    desc_tinv_slot, cutlass.Int32(0), cutlass.Int32(0), head_o, tinv_row
                )
                tma_load_tile(
                    sTinv_tma[tinv_idx], tinv_slice, bars.mb_t_inv_ready[tinv_idx].smem_ptr
                )

        tile_idx, scheduler_state = scheduler_publish_next(
            cfg, bars, sScheduler, mScheduler, scheduler_state, num_ctas, elect_one
        )

    for _ in range(cfg.smem_kq_stages):
        bars.mb_kq_done[kq_index.idx].wait(kq_index.phase)
        kq_index = advance(kq_index, cfg.smem_kq_stages)
    for _ in range(cfg.smem_v_stages):
        bars.mb_v_done[v_index.idx].wait(v_index.phase)
        v_index = advance(v_index, cfg.smem_v_stages)
    if cutlass.const_expr(cfg.tinv_source == "gmem"):
        for _ in range(cfg.smem_t_inv_stages):
            bars.mb_t_inv_done[tinv_index.idx].wait(tinv_index.phase)
            tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.jit
def compute0_warp_group_gmem(
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
    sA,
    sScheduler,
    bars,
):
    """Compute warp-group 0 role (warps 0-3) with the chunk factor bulk-loaded
    from GMEM: only the A epilogues remain, every warp staging 16 rows of each
    chunk's Q K^T (the M = b_t accumulator occupies lanes 0-15 of every
    sub-partition)."""

    nvvm.setmaxregister(cfg.num_regs_compute_group_0, nvvm.SetMaxRegisterAction.INCREASE)
    elect_one = nvvm.elect_sync()

    gate_index = PipelineState.start(phase=0)
    cg0_acc_ready = PipelineState.start(phase=0)
    a_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=0)

    num_threads_cg0 = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    cg0_tidx = tidx % num_threads_cg0
    warp_id = warp_idx - cfg.compute_group_0_warp_ids[0]
    lane_idx = cg0_tidx % cfg.threads_per_warp
    store_row = warp_id * 16 + lane_idx % 16
    store_col = (lane_idx // 16) * 8
    row_lo = warp_id * 16 + lane_idx // 4
    row_hi = row_lo + 8

    num_vals = 32
    FRAG_COLS = 16
    ACC_N_FRAGS = cfg.b_t // FRAG_COLS
    ACC_STAGE_COLS = cfg.b_t
    mask_zero = opaque_f32_zero()
    tile_idx = cutlass.Int32(bidx)

    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads
    )
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    tmem_cg0_acc_col = tmem_col + cfg.tmem_cg0_acc_offset
    while tile_idx < total_tiles:
        (
            _batch_idx,
            _head_idx,
            _batch_start,
            _batch_end,
            _batch_seqlen,
            _batch_num_chunks,
            _write_start,
            write_end,
            compute_start,
            _compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        n_local = write_end - compute_start
        n_pairs = (n_local + 1) // 2

        for pair_i in cutlass.range(n_pairs):
            have_m1 = pair_i * 2 + 1 < n_local

            gate0_idx = gate_index.idx
            bars.mb_gate_ready[gate0_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)
            gate1_idx = gate0_idx
            if have_m1:
                gate1_idx = gate_index.idx
                bars.mb_gate_ready[gate1_idx].wait(gate_index.phase)
                gate_index = advance(gate_index, cfg.smem_gate_stages)

            for member in cutlass.range_constexpr(2):
                member_active = (
                    cutlass.Boolean(True) if cutlass.const_expr(member == 0) else have_m1
                )
                if member_active:
                    a_gate_idx = gate0_idx if cutlass.const_expr(member == 0) else gate1_idx

                    # ---- Gate rows for this chunk's A tile ---------------------------
                    a_cumsumlog_rows = [
                        sCumsumlog[row_lo, 0, a_gate_idx],
                        sCumsumlog[row_hi, 0, a_gate_idx],
                    ]
                    a_cumsumlog_cols = []
                    for g in cutlass.range_constexpr(8):
                        for b in cutlass.range_constexpr(2):
                            chunk_col = (lane_idx % 4) * 2 + g * 8 + b
                            a_cumsumlog_cols.append(sCumsumlog[chunk_col, 0, a_gate_idx])
                    decay_t_a = []
                    for k in cutlass.range_constexpr(num_vals):
                        hi_row = ((k // 2) % 2) == 1
                        chunk_row = row_hi if cutlass.const_expr(hi_row) else row_lo
                        chunk_col = (lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2)
                        is_lower = chunk_row >= chunk_col
                        col = (k // 4) * 2 + (k % 2)
                        decay_t_a.append(
                            cute.math.exp2(
                                a_cumsumlog_rows[1 if hi_row else 0] - a_cumsumlog_cols[col],
                                fastmath=True,
                            )
                            if is_lower
                            else mask_zero
                        )

                    # ---- A epilogue: A[i,j] = W qk[i,j] * T[i,j] * scale -------------
                    acc_idx = cg0_acc_ready.idx
                    acc_phase = cg0_acc_ready.phase
                    cg0_acc_ready = advance(cg0_acc_ready, cfg.tmem_cg0_acc_stages)
                    a_idx = a_index.idx
                    a_phase = a_index.phase
                    a_index = advance(a_index, cfg.smem_a_stages)
                    bars.mb_cg0_acc_ready[acc_idx].wait(acc_phase)
                    a_vec = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(
                            row_lo_addr + tmem_cg0_acc_col + acc_idx * ACC_STAGE_COLS,
                            cutlass.Float32,
                        ),
                        num=8,
                    )
                    nvvm.tcgen05_wait("load")
                    bars.mb_cg0_acc_done[acc_idx].arrive()
                    a_pack = []
                    for k in cutlass.range_constexpr(num_vals // 2):
                        p0, p1 = fmul2(
                            a_vec[2 * k], a_vec[2 * k + 1], decay_t_a[2 * k], decay_t_a[2 * k + 1]
                        )
                        v0, v1 = fmul2(p0, p1, scale, scale)
                        a_pack.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
                    bars.mb_a_done[a_idx].wait(a_phase)
                    a_base = smem_data_ptr(sA[a_idx].base)
                    for c in cutlass.range_constexpr(ACC_N_FRAGS):
                        nvvm.stmatrix(
                            a_base
                            + store_row * cfg.b_t
                            + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS),
                            [
                                a_pack[c * 4 + 0],
                                a_pack[c * 4 + 1],
                                a_pack[c * 4 + 2],
                                a_pack[c * 4 + 3],
                            ],
                            nvvm.MMALayout.ROW,
                        )
                    nvvm.fence_proxy("async.shared", space="cta")
                    bars.mb_a_ready[a_idx].arrive()

            bars.mb_gate_done[gate0_idx].arrive()
            if have_m1:
                bars.mb_gate_done[gate1_idx].arrive()

        tile_idx, scheduler_state = scheduler_next_tile(
            cfg, bars, sScheduler, scheduler_state, elect_one
        )
    for _ in range(cfg.smem_a_stages):
        bars.mb_a_done[a_index.idx].wait(a_index.phase)
        a_index = advance(a_index, cfg.smem_a_stages)


@cute.jit
def cg0_pair(
    cfg,
    have_m1: cutlass.Constexpr[bool],
    gate_index,
    beta_index,
    cg0_acc_ready,
    tinv_index,
    a_index,
    pair_half,
    warp_id,
    lane_idx,
    inverse_local_warp,
    half_row_base,
    store_row,
    store_col,
    store_row_frag,
    cg0_tidx,
    row_lo_addr,
    row_hi_addr,
    tmem_cg0_acc_col,
    scale,
    mask_zero,
    sCumsumlog,
    sBeta,
    sTinv,
    sA,
    bars,
):
    """One chunk pair of compute group 0 with the chunk factor built in-kernel: both members' KK
    epilogues, the pair inverse, the beta scaling and the A epilogues.  ``have_m1`` False is the
    odd tail: member 0 alone, warps 0-1 own the single inverse and warps 2-3 the single A epilogue,
    every ring advances once."""
    num_vals = 32
    FRAG_COLS = 16
    ACC_N_FRAGS = cfg.b_t // FRAG_COLS
    ACC_STAGE_COLS = cfg.b_t

    do_kk = have_m1 or pair_half == 0
    do_a = have_m1 or pair_half == 1

    # ---- Gate rows for this warp's KK / A member roles -------------------------------
    gate0_idx = gate_index.idx
    bars.mb_gate_ready[gate0_idx].wait(gate_index.phase)
    gate_index = advance(gate_index, cfg.smem_gate_stages)
    gate1_idx = gate0_idx
    if have_m1:
        gate1_idx = gate_index.idx
        bars.mb_gate_ready[gate1_idx].wait(gate_index.phase)
        gate_index = advance(gate_index, cfg.smem_gate_stages)
    kk_gate_idx = gate1_idx if pair_half == 1 else gate0_idx
    a_gate_idx = gate0_idx if pair_half == 1 else gate1_idx

    row_u0_lo = half_row_base + lane_idx // 4
    row_u0_hi = row_u0_lo + 8
    row_u1_lo = row_u0_lo + 16
    row_u1_hi = row_u0_lo + 24

    kk_cumsumlog_rows = []
    for r in (row_u0_lo, row_u0_hi, row_u1_lo, row_u1_hi):
        kk_cumsumlog_rows.append(sCumsumlog[r, 0, kk_gate_idx])
    kk_cumsumlog_cols = []
    for g in cutlass.range_constexpr(8):
        for b in cutlass.range_constexpr(2):
            chunk_col = (lane_idx % 4) * 2 + g * 8 + b
            kk_cumsumlog_cols.append(sCumsumlog[chunk_col, 0, kk_gate_idx])

    decay_t_kk = []
    for u in cutlass.range_constexpr(2):
        for k in cutlass.range_constexpr(num_vals):
            hi_row = ((k // 2) % 2) == 1
            chunk_row_u0 = row_u0_hi if cutlass.const_expr(hi_row) else row_u0_lo
            chunk_row_u1 = row_u1_hi if cutlass.const_expr(hi_row) else row_u1_lo
            chunk_row = chunk_row_u1 if cutlass.const_expr(u == 1) else chunk_row_u0
            chunk_col = (lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2)
            is_lower = chunk_row >= chunk_col
            kk_row_cumsumlog = kk_cumsumlog_rows[u * 2 + (1 if hi_row else 0)]
            col = (k // 4) * 2 + (k % 2)
            decay_t_kk.append(
                cute.math.exp2(kk_row_cumsumlog - kk_cumsumlog_cols[col], fastmath=True)
                if is_lower
                else mask_zero
            )
    beta0_idx = beta_index.idx
    bars.mb_beta_ready[beta0_idx].wait(beta_index.phase)
    beta_index = advance(beta_index, cfg.smem_beta_stages)
    beta1_idx = beta0_idx
    if have_m1:
        beta1_idx = beta_index.idx
        bars.mb_beta_ready[beta1_idx].wait(beta_index.phase)
        beta_index = advance(beta_index, cfg.smem_beta_stages)
    kk_beta_idx = beta1_idx if pair_half == 1 else beta0_idx
    kk_beta = []
    for r in (row_u0_lo, row_u0_hi, row_u1_lo, row_u1_hi):
        kk_beta.append(sBeta[r, 0, kk_beta_idx])

    # ---- KK epilogue: M kk[i,j] = W kk[i,j] * T[i,j] * Beta[i] -----------------------
    acc0_idx = cg0_acc_ready.idx
    acc0_phase = cg0_acc_ready.phase
    cg0_acc_ready = advance(cg0_acc_ready, cfg.tmem_cg0_acc_stages)
    acc1_idx = acc0_idx
    acc1_phase = acc0_phase
    if have_m1:
        acc1_idx = cg0_acc_ready.idx
        acc1_phase = cg0_acc_ready.phase
        cg0_acc_ready = advance(cg0_acc_ready, cfg.tmem_cg0_acc_stages)
    kk_acc_idx = acc1_idx if pair_half == 1 else acc0_idx
    kk_acc_phase = acc1_phase if pair_half == 1 else acc0_phase
    a_acc_idx = acc0_idx if pair_half == 1 else acc1_idx

    tinv0_idx = tinv_index.idx
    tinv0_phase = tinv_index.phase
    tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
    tinv1_idx = tinv0_idx
    tinv1_phase = tinv0_phase
    if have_m1:
        tinv1_idx = tinv_index.idx
        tinv1_phase = tinv_index.phase
        tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
    kk_tinv_idx = tinv1_idx if pair_half == 1 else tinv0_idx
    kk_tinv_phase = tinv1_phase if pair_half == 1 else tinv0_phase

    tinv0_base = smem_data_ptr(sTinv[tinv0_idx].base)
    tinv1_base = smem_data_ptr(sTinv[tinv1_idx].base)
    kk_base = tinv1_base if pair_half == 1 else tinv0_base
    if do_kk:
        bars.mb_cg0_acc_ready[kk_acc_idx].wait(kk_acc_phase)
        kk_vec_lo = nvvm.tcgen05_ld(
            "16x256b",
            nvvm.make_tmem_ptr(
                row_lo_addr + tmem_cg0_acc_col + kk_acc_idx * ACC_STAGE_COLS, cutlass.Float32
            ),
            num=8,
        )
        kk_vec_hi = nvvm.tcgen05_ld(
            "16x256b",
            nvvm.make_tmem_ptr(
                row_hi_addr + tmem_cg0_acc_col + kk_acc_idx * ACC_STAGE_COLS, cutlass.Float32
            ),
            num=8,
        )
        bars.mb_t_inv_done[kk_tinv_idx].wait(kk_tinv_phase)
        for u in cutlass.range_constexpr(2):
            kk_vec = kk_vec_hi if cutlass.const_expr(u == 1) else kk_vec_lo
            kk_pack = []
            for k in cutlass.range_constexpr(num_vals // 2):
                b0 = kk_beta[u * 2 + 1] if cutlass.const_expr((k % 2) == 1) else kk_beta[u * 2]
                p0, p1 = fmul2(
                    kk_vec[2 * k],
                    kk_vec[2 * k + 1],
                    decay_t_kk[u * num_vals + 2 * k],
                    decay_t_kk[u * num_vals + 2 * k + 1],
                )
                v0, v1 = fmul2(p0, p1, b0, b0)
                kk_pack.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            st_row = half_row_base + u * 16 + store_row_frag
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    kk_base
                    + st_row * cfg.b_t
                    + swizzle_xor_128b(st_row, store_col + c * FRAG_COLS),
                    [
                        kk_pack[c * 4 + 0],
                        kk_pack[c * 4 + 1],
                        kk_pack[c * 4 + 2],
                        kk_pack[c * 4 + 3],
                    ],
                    nvvm.MMALayout.ROW,
                )

    # ---- pair inverse: warps 0-1 own matrix 0, warps 2-3 matrix 1 --------------------
    inv_base = tinv0_base
    if have_m1:
        inv_base = tinv1_base if warp_id >= 2 else tinv0_base
    do_inv = have_m1 or warp_id < 2

    # ---- diagonal 8x8 inverse --------------------------------------------------------
    nvvm.barrier_cta_sync_aligned(
        cfg.inverse_barrier_id,
        thread_count=cfg.inverse_barrier_threads,
    )
    if do_inv:
        invert_diagonal_NxN(
            cfg,
            inv_base,
            inv_base,
            (inverse_local_warp * cfg.threads_per_warp + lane_idx) // 8,
            cg0_tidx,
            8,
        )
    nvvm.barrier_cta_sync_aligned(
        cfg.inverse_barrier_id,
        thread_count=cfg.inverse_barrier_threads,
    )

    # ---- 8x8 -> 16x16 ----------------------------------------------------------------
    blockwise_diagonal_8x8_to_16x16(cfg, tinv0_base, tinv0_base, warp_id * 16, lane_idx)
    if have_m1:
        blockwise_diagonal_8x8_to_16x16(cfg, tinv1_base, tinv1_base, warp_id * 16, lane_idx)
    nvvm.barrier_cta_sync_aligned(
        cfg.inverse_barrier_id,
        thread_count=cfg.inverse_barrier_threads,
    )

    # ---- 16x16 -> 32x32 --------------------------------------------------------------
    if do_inv:
        blockwise_diagonal_16x16_to_32x32(
            cfg, inv_base, inv_base, inverse_local_warp * 32, lane_idx
        )
    nvvm.barrier_cta_sync_aligned(
        cfg.inverse_barrier_id,
        thread_count=cfg.inverse_barrier_threads,
    )

    # ---- 32x32 -> 64x64 --------------------------------------------------------------
    blockwise_diagonal_32x32_to_64x64(
        cfg,
        inv_base,
        inv_base,
        inverse_local_warp,
        lane_idx,
        cfg.inverse_barrier_id,
        cfg.inverse_barrier_threads,
    )
    nvvm.barrier_cta_sync_aligned(
        cfg.inverse_barrier_id,
        thread_count=cfg.inverse_barrier_threads,
    )

    # ---- Beta column scaling, member 0: T^-1[i,j] *= Beta[j] -------------------------
    beta_col = []
    for k in cutlass.range_constexpr(num_vals):
        beta_col.append(sBeta[(lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2), 0, beta0_idx])
    tinv_frags = []
    for c in cutlass.range_constexpr(ACC_N_FRAGS):
        tinv_frags += list(
            nvvm.ldmatrix(
                tinv0_base
                + store_row * cfg.b_t
                + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS),
                4,
                nvvm.MMALayout.ROW,
            )
        )
    tinv_pack = []
    for j in cutlass.range_constexpr(num_vals // 2):
        lo, hi = f16x2_to_f32(tinv_frags[j], dtype=cfg.io_dtype)
        s0, s1 = fmul2(lo, hi, beta_col[2 * j], beta_col[2 * j + 1])
        tinv_pack.append(fp32_to_fp16(s0, s1, dtype=cfg.io_dtype))
    for c in cutlass.range_constexpr(ACC_N_FRAGS):
        nvvm.stmatrix(
            tinv0_base
            + store_row * cfg.b_t
            + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS),
            [
                tinv_pack[c * 4 + 0],
                tinv_pack[c * 4 + 1],
                tinv_pack[c * 4 + 2],
                tinv_pack[c * 4 + 3],
            ],
            nvvm.MMALayout.ROW,
        )
    nvvm.fence_proxy("async.shared", space="cta")
    bars.mb_t_inv_ready[tinv0_idx].arrive()
    bars.mb_beta_done[beta0_idx].arrive()

    if have_m1:
        # ---- Beta column scaling, member 1: T^-1[i,j] *= Beta[j] ---------------------
        beta_col = []
        for k in cutlass.range_constexpr(num_vals):
            beta_col.append(sBeta[(lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2), 0, beta1_idx])
        tinv_frags = []
        for c in cutlass.range_constexpr(ACC_N_FRAGS):
            tinv_frags += list(
                nvvm.ldmatrix(
                    tinv1_base
                    + store_row * cfg.b_t
                    + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS),
                    4,
                    nvvm.MMALayout.ROW,
                )
            )
        tinv_pack = []
        for j in cutlass.range_constexpr(num_vals // 2):
            lo, hi = f16x2_to_f32(tinv_frags[j], dtype=cfg.io_dtype)
            s0, s1 = fmul2(lo, hi, beta_col[2 * j], beta_col[2 * j + 1])
            tinv_pack.append(fp32_to_fp16(s0, s1, dtype=cfg.io_dtype))
        for c in cutlass.range_constexpr(ACC_N_FRAGS):
            nvvm.stmatrix(
                tinv1_base
                + store_row * cfg.b_t
                + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS),
                [
                    tinv_pack[c * 4 + 0],
                    tinv_pack[c * 4 + 1],
                    tinv_pack[c * 4 + 2],
                    tinv_pack[c * 4 + 3],
                ],
                nvvm.MMALayout.ROW,
            )
        nvvm.fence_proxy("async.shared", space="cta")
        bars.mb_t_inv_ready[tinv1_idx].arrive()
        bars.mb_beta_done[beta1_idx].arrive()

    # ---- A-member decay tile ---------------------------------------------------------
    a_cumsumlog_rows = []
    for r in (row_u0_lo, row_u0_hi, row_u1_lo, row_u1_hi):
        a_cumsumlog_rows.append(sCumsumlog[r, 0, a_gate_idx])
    a_cumsumlog_cols = []
    for g in cutlass.range_constexpr(8):
        for b in cutlass.range_constexpr(2):
            chunk_col = (lane_idx % 4) * 2 + g * 8 + b
            a_cumsumlog_cols.append(sCumsumlog[chunk_col, 0, a_gate_idx])
    decay_t_a = []
    for u in cutlass.range_constexpr(2):
        for k in cutlass.range_constexpr(num_vals):
            hi_row = ((k // 2) % 2) == 1
            chunk_row_u0 = row_u0_hi if cutlass.const_expr(hi_row) else row_u0_lo
            chunk_row_u1 = row_u1_hi if cutlass.const_expr(hi_row) else row_u1_lo
            chunk_row = chunk_row_u1 if cutlass.const_expr(u == 1) else chunk_row_u0
            chunk_col = (lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2)
            is_lower = chunk_row >= chunk_col
            a_row_cumsumlog = a_cumsumlog_rows[u * 2 + (1 if hi_row else 0)]
            col = (k // 4) * 2 + (k % 2)
            decay_t_a.append(
                cute.math.exp2(a_row_cumsumlog - a_cumsumlog_cols[col], fastmath=True)
                if is_lower
                else mask_zero
            )
    bars.mb_gate_done[gate0_idx].arrive()
    if have_m1:
        bars.mb_gate_done[gate1_idx].arrive()

    # ---- A epilogue (opposite member, both halves in parallel) -----------------------
    a0_idx = a_index.idx
    a0_phase = a_index.phase
    a_index = advance(a_index, cfg.smem_a_stages)
    a1_idx = a0_idx
    a1_phase = a0_phase
    if have_m1:
        a1_idx = a_index.idx
        a1_phase = a_index.phase
        a_index = advance(a_index, cfg.smem_a_stages)
    my_a_idx = a0_idx if pair_half == 1 else a1_idx
    my_a_phase = a0_phase if pair_half == 1 else a1_phase

    if do_a:
        a_base = smem_data_ptr(sA[my_a_idx].base)
        a_vec_lo = nvvm.tcgen05_ld(
            "16x256b",
            nvvm.make_tmem_ptr(
                row_lo_addr + tmem_cg0_acc_col + a_acc_idx * ACC_STAGE_COLS, cutlass.Float32
            ),
            num=8,
        )
        a_vec_hi = nvvm.tcgen05_ld(
            "16x256b",
            nvvm.make_tmem_ptr(
                row_hi_addr + tmem_cg0_acc_col + a_acc_idx * ACC_STAGE_COLS, cutlass.Float32
            ),
            num=8,
        )
        nvvm.tcgen05_wait("load")
        bars.mb_cg0_acc_done[a_acc_idx].arrive()
        bars.mb_a_done[my_a_idx].wait(my_a_phase)
        for u in cutlass.range_constexpr(2):
            a_vec = a_vec_hi if cutlass.const_expr(u == 1) else a_vec_lo
            a_pack = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(
                    a_vec[2 * k],
                    a_vec[2 * k + 1],
                    decay_t_a[u * num_vals + 2 * k],
                    decay_t_a[u * num_vals + 2 * k + 1],
                )
                v0, v1 = fmul2(p0, p1, scale, scale)
                a_pack.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            st_row = half_row_base + u * 16 + store_row_frag
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    a_base
                    + st_row * cfg.b_t
                    + swizzle_xor_128b(st_row, store_col + c * FRAG_COLS),
                    [a_pack[c * 4 + 0], a_pack[c * 4 + 1], a_pack[c * 4 + 2], a_pack[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )
        nvvm.fence_proxy("async.shared", space="cta")
        bars.mb_a_ready[my_a_idx].arrive()
    return gate_index, beta_index, cg0_acc_ready, tinv_index, a_index


@cute.jit
def compute0_warp_group(
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
    sBeta,
    sTinv,
    sA,
    sScheduler,
    bars,
):
    """Compute warp-group 0 role (warps 0-3) with the chunk factor built
    in-kernel: each chunk pair's KK epilogues, blockwise inverse, beta
    scaling and A epilogues."""

    nvvm.setmaxregister(cfg.num_regs_compute_group_0, nvvm.SetMaxRegisterAction.INCREASE)
    elect_one = nvvm.elect_sync()

    gate_index = PipelineState.start(phase=0)
    beta_index = PipelineState.start(phase=0)
    cg0_acc_ready = PipelineState.start(phase=0)
    tinv_index = PipelineState.start(phase=1)
    a_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=0)

    num_threads_cg0 = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    cg0_tidx = tidx % num_threads_cg0
    warp_id = warp_idx - cfg.compute_group_0_warp_ids[0]
    lane_idx = cg0_tidx % cfg.threads_per_warp
    inverse_local_warp = warp_id % 2
    store_row = warp_id * 16 + lane_idx % 16
    store_col = (lane_idx // 16) * 8

    pair_half = warp_id // 2
    half_row_base = inverse_local_warp * 32
    store_row_frag = lane_idx % 16
    mask_zero = opaque_f32_zero()
    chunk_row_lo = warp_id * 16 + lane_idx // 4
    chunk_row_lo + 8
    tile_idx = cutlass.Int32(bidx)

    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads
    )
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    row_hi_addr = (tmem_row + 16) << 16
    tmem_cg0_acc_col = tmem_col + cfg.tmem_cg0_acc_offset
    while tile_idx < total_tiles:
        (
            _batch_idx,
            _head_idx,
            _batch_start,
            _batch_end,
            _batch_seqlen,
            _batch_num_chunks,
            _write_start,
            write_end,
            compute_start,
            _compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        n_local = write_end - compute_start

        n_pairs_full = n_local // 2
        for pair_i in cutlass.range(n_pairs_full):
            gate_index, beta_index, cg0_acc_ready, tinv_index, a_index = cg0_pair(
                cfg,
                True,
                gate_index,
                beta_index,
                cg0_acc_ready,
                tinv_index,
                a_index,
                pair_half,
                warp_id,
                lane_idx,
                inverse_local_warp,
                half_row_base,
                store_row,
                store_col,
                store_row_frag,
                cg0_tidx,
                row_lo_addr,
                row_hi_addr,
                tmem_cg0_acc_col,
                scale,
                mask_zero,
                sCumsumlog,
                sBeta,
                sTinv,
                sA,
                bars,
            )
        if n_local - n_pairs_full * 2 == 1:
            gate_index, beta_index, cg0_acc_ready, tinv_index, a_index = cg0_pair(
                cfg,
                False,
                gate_index,
                beta_index,
                cg0_acc_ready,
                tinv_index,
                a_index,
                pair_half,
                warp_id,
                lane_idx,
                inverse_local_warp,
                half_row_base,
                store_row,
                store_col,
                store_row_frag,
                cg0_tidx,
                row_lo_addr,
                row_hi_addr,
                tmem_cg0_acc_col,
                scale,
                mask_zero,
                sCumsumlog,
                sBeta,
                sTinv,
                sA,
                bars,
            )
        tile_idx, scheduler_state = scheduler_next_tile(
            cfg, bars, sScheduler, scheduler_state, elect_one
        )
    for _ in range(cfg.smem_t_inv_stages):
        bars.mb_t_inv_done[tinv_index.idx].wait(tinv_index.phase)
        tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
    for _ in range(cfg.smem_a_stages):
        bars.mb_a_done[a_index.idx].wait(a_index.phase)
        a_index = advance(a_index, cfg.smem_a_stages)


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
    sV_trans,
    sCumsumlog,
    sCumprod,
    sO,
    sCheckpoint_raw,
    mState_init,
    mState_out,
    mSeedIndices,
    mFinalIndices,
    mHasInitialState,
    checkpoint_every_n_tokens,
    sScheduler,
    bars,
):
    """Compute warp-group 1 role (warps 4-7): persistent scheduler loop
    running the per-chunk state-update and output epilogues.

    With ``cfg.paged_state`` the state tensors alias one pool and ``mSeedIndices[batch_idx]``
    selects the slot: non-positive routes are null (zero output, pool untouched), a zero
    ``mHasInitialState[batch_idx]`` seeds from zero and, for an empty sequence, clears the slot."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_1, nvvm.SetMaxRegisterAction.INCREASE)

    v_index = PipelineState.start(phase=0)
    gate_index = PipelineState.start(phase=0)
    kv_acc_index = PipelineState.start(phase=0)
    o_acc_ready_index = PipelineState.start(phase=0)
    o_final_acc_ready_index = PipelineState.start(phase=0)
    k_state_ready_index = PipelineState.start(phase=0)
    u_acc_ready_index = PipelineState.start(phase=0)
    o_index = PipelineState.start(phase=1)

    num_threads_cg1 = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    cg1_tidx = tidx % num_threads_cg1
    lane_idx = cg1_tidx % cfg.threads_per_warp

    elect_one = nvvm.elect_sync()
    state_input_cnt = cutlass.Int32(0)
    ldtm_width = 32
    sttm_width = 16
    num_ldtms = cutlass.const_expr(cfg.d_k // ldtm_width)
    ACC_STAGE_COLS = cfg.b_t
    INP_SLOT_COLS = cfg.b_t // 2
    dv_halves = cutlass.const_expr(cfg.d_v // 64)
    v_o_row = cg1_tidx % 8 + (cg1_tidx // 16 % 2) * 8
    if cutlass.const_expr(cfg.d_v == 128):
        o_col_base = (cg1_tidx // 32 % 2) * 32
        v_o_segment = (cg1_tidx // 64) * (cfg.b_t * 64)
        state_gmem_row = cg1_tidx
        state_row_valid = cutlass.Boolean(True)
    else:
        o_col_base = (cg1_tidx // 32) * 16
        v_o_segment = 0
        state_gmem_row = (cg1_tidx // 32) * 16 + (cg1_tidx % 32) % 16
        state_row_valid = (cg1_tidx % 32) < 16
    state_halves_n = cutlass.const_expr(cfg.d_k // 64)
    state_sub_base = (cg1_tidx // cfg.threads_per_warp) * 16
    state_half_row = (cg1_tidx % cfg.threads_per_warp) // 4
    state_half_col = (cg1_tidx % cfg.threads_per_warp) % 4 * 2
    v_o_col = o_col_base + (cg1_tidx // 8 % 2) * 8
    v_stage_elements = cfg.v_cosize // cfg.smem_v_stages
    o_stage_elements = cfg.o_cosize // cfg.smem_o_stages
    sV_base = smem_data_ptr(sV_trans[0].base)
    sO_base = smem_data_ptr(sO[0].base)
    num_vals = 32

    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads
    )
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    (tmem_row + 16) << 16
    tmem_state_col = tmem_col + cfg.tmem_state_acc_offset
    tmem_state_input_col = tmem_col + cfg.tmem_state_input_offset
    tmem_q_state_col = tmem_col + cfg.tmem_q_state_acc_offset
    tmem_input_col = tmem_col + cfg.tmem_y_decay_u_input_offset
    tmem_k_state_col = tmem_col + cfg.tmem_cg1_acc_offset
    tmem_u_acc_col = tmem_k_state_col
    tmem_y_input_col = tmem_input_col
    tmem_u_input_col = tmem_input_col
    tmem_decay_v_col = tmem_input_col + INP_SLOT_COLS
    if cutlass.const_expr(cfg.enable_checkpoints):
        sCheckpoint_base = smem_data_ptr(sCheckpoint_raw)
        checkpoint_cnt = cutlass.Int32(0)
        cg1_tidx % 8 + (cg1_tidx // 16 % 2) * 8
        (cg1_tidx // 8 % 2) * 8 + (cg1_tidx // 32 % 2) * 32

    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        (
            batch_idx,
            head_idx,
            _batch_start,
            _batch_end,
            _batch_seqlen,
            _batch_num_chunks,
            write_start,
            write_end,
            compute_start,
            _compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        head_o, v_offset = decode_head(cfg, head_idx)
        # Attention Gym modification: paged state resolves the sequence's pool slot and its
        # load/store predicates once per work item (null routes never touch the pool, fresh
        # routes zero-seed, empty fresh routes clear their slot); dense calls index directly.
        seed_row = batch_idx
        if cutlass.const_expr(cfg.paged_state):
            seed_row, state_active, load_initial, clear_empty = resolve_paged_state(
                batch_idx, mSeedIndices, mHasInitialState
            )
        elif cutlass.const_expr(mSeedIndices is not None):
            seed_row = cutlass.Int32(mSeedIndices[batch_idx])
        n_local = write_end - compute_start
        if cutlass.const_expr(cfg.enable_checkpoints):
            checkpoint_chunks = checkpoint_every_n_tokens // cutlass.Int32(cfg.b_t)
            checkpoint_mod = compute_start % checkpoint_chunks
        if n_local > 0:
            if cutlass.const_expr(cfg.use_initial_state):
                gState_init = mState_init[None, None, head_o, seed_row]
                seed_state = compute_start == 0
                if cutlass.const_expr(cfg.paged_state):
                    seed_state = seed_state and load_initial
                if seed_state:
                    seed_vw = 16 // (mState_init.element_type.width // 8)
                    seed_src = (
                        gState_init.iterator + gState_init.layout((state_gmem_row + v_offset, 0))
                    ).raw_ptr()
                    for i in cutlass.range_constexpr(num_ldtms):
                        words = []
                        for g in cutlass.range_constexpr(ldtm_width // seed_vw):
                            seed_chunk = (seed_src + i * ldtm_width + g * seed_vw).load(
                                count=seed_vw, alignment=16
                            )
                            for t in cutlass.range_constexpr(seed_vw):
                                words.append(seed_chunk[t].to(cfg.acc_dtype))
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                row_lo_addr + tmem_state_col + i * ldtm_width, cutlass.Float32
                            ),
                            cutlass.Vector.from_elements(tuple(words), cutlass.Float32),
                        )
                    nvvm.tcgen05_wait("store")
                else:
                    for i in cutlass.range_constexpr(num_ldtms):
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                row_lo_addr + tmem_state_col + i * ldtm_width, cutlass.Float32
                            ),
                            cutlass.Vector.from_elements(
                                tuple(cutlass.Float32(0.0) for _ in range(32)), cutlass.Float32
                            ),
                        )
                    nvvm.tcgen05_wait("store")

            for peel in cutlass.range_constexpr(1 if cfg.use_initial_state else 2):
                peel_stop = (
                    n_local
                    if cutlass.const_expr(peel == 1 or cfg.use_initial_state)
                    else cutlass.min(n_local, 1)
                )
                for local_idx in cutlass.range(peel, peel_stop, 1):
                    chunk_idx = compute_start + local_idx
                    if cutlass.const_expr(cfg.enable_checkpoints):
                        do_checkpoint_now = checkpoint_mod == 0
                        checkpoint_mod = checkpoint_mod + cutlass.Int32(1)
                        checkpoint_mod = (
                            cutlass.Int32(0)
                            if checkpoint_mod == checkpoint_chunks
                            else checkpoint_mod
                        )
                    if cutlass.const_expr(cfg.enable_checkpoints and not cfg.use_initial_state):
                        if chunk_idx == 0 and write_start == 0:
                            checkpoint_stage = checkpoint_cnt % cfg.smem_checkpoint_stages
                            checkpoint_phase_done = cutlass.Int32(1) ^ (
                                (checkpoint_cnt // cfg.smem_checkpoint_stages) & cutlass.Int32(1)
                            )
                            bars.mb_checkpoint_tmastg_done[checkpoint_stage].wait(
                                checkpoint_phase_done
                            )
                            checkpoint_zero_ptr = cutlass.inttoptr(
                                (sCheckpoint_base + checkpoint_stage * cfg.d_k * cfg.d_v).toint(),
                                cutlass.AddressSpace.smem,
                                cutlass.Int32,
                            )
                            for z in cutlass.range_constexpr(
                                cfg.d_k * cfg.d_v // 2 // num_threads_cg1
                            ):
                                (checkpoint_zero_ptr + cg1_tidx + z * num_threads_cg1).store(
                                    cutlass.Int32(0)
                                )
                            nvvm.fence_proxy("async.shared", space="cta")
                            if elect_one:
                                bars.mb_checkpoint_tmastg_ready[checkpoint_stage].arrive()
                            checkpoint_cnt = checkpoint_cnt + 1
                    valid_state = (
                        cutlass.Boolean(True)
                        if cutlass.const_expr(cfg.use_initial_state)
                        else cutlass.const_expr(peel == 1)
                    )

                    gate_idx = gate_index.idx
                    bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
                    gate_index = advance(gate_index, cfg.smem_gate_stages)
                    cumprod_total = sCumprod[sCumprod.shape[0] - 1, 0, gate_idx]

                    # ---- state stage + rescale ---------------------------------------
                    if valid_state:
                        if cutlass.const_expr(cfg.use_initial_state):
                            if local_idx > 0:
                                bars.mb_state_acc_ready[kv_acc_index.idx].wait(kv_acc_index.phase)
                                kv_acc_index = advance(kv_acc_index, cfg.tmem_state_acc_stages)
                        else:
                            bars.mb_state_acc_ready[kv_acc_index.idx].wait(kv_acc_index.phase)
                            kv_acc_index = advance(kv_acc_index, cfg.tmem_state_acc_stages)

                        state_input_stage_idx = state_input_cnt % cfg.tmem_state_input_stages
                        state_regs = [
                            [cutlass.Float32(0.0) for _ in range(num_ldtms)] for _ in range(32)
                        ]
                        state_halves = []
                        if cutlass.const_expr(cfg.d_v == 64):
                            for hh in cutlass.range_constexpr(state_halves_n):
                                state_halves.append(
                                    nvvm.tcgen05_ld(
                                        "16x256b",
                                        nvvm.make_tmem_ptr(
                                            row_lo_addr + tmem_state_col + hh * 64, cutlass.Float32
                                        ),
                                        num=8,
                                    )
                                )
                            state_pack = [
                                fp32_to_fp16(
                                    state_halves[hh][2 * j],
                                    state_halves[hh][2 * j + 1],
                                    dtype=cfg.io_dtype,
                                )
                                for hh in range(state_halves_n)
                                for j in range(16)
                            ]
                            nvvm.tcgen05_st(
                                "16x128b",
                                nvvm.make_tmem_ptr(
                                    row_lo_addr + tmem_state_input_col, cutlass.Int32
                                ),
                                cutlass.Vector.from_elements(tuple(state_pack), cutlass.Int32),
                            )
                        else:
                            for i in cutlass.range_constexpr(num_ldtms):
                                state_vec = nvvm.tcgen05_ld(
                                    "32x32b",
                                    nvvm.make_tmem_ptr(
                                        row_lo_addr + tmem_state_col + i * ldtm_width,
                                        cutlass.Float32,
                                    ),
                                    num=32,
                                )
                                for k in cutlass.range_constexpr(32):
                                    state_regs[k][i] = state_vec[k]
                            for i in cutlass.range_constexpr(num_ldtms):
                                state_pack = [
                                    fp32_to_fp16(
                                        state_regs[2 * j][i],
                                        state_regs[2 * j + 1][i],
                                        dtype=cfg.io_dtype,
                                    )
                                    for j in range(16)
                                ]
                                nvvm.tcgen05_st(
                                    "32x32b",
                                    nvvm.make_tmem_ptr(
                                        row_lo_addr + tmem_state_input_col + i * sttm_width,
                                        cutlass.Int32,
                                    ),
                                    cutlass.Vector.from_elements(tuple(state_pack), cutlass.Int32),
                                )
                        nvvm.tcgen05_wait("store")
                        bars.mb_state_input_ready[state_input_stage_idx].arrive()
                        state_input_cnt = state_input_cnt + 1

                        if cutlass.const_expr(cfg.enable_checkpoints):
                            # ---- state checkpoint ------------------------------------
                            do_checkpoint = do_checkpoint_now and chunk_idx < write_end
                            do_checkpoint = do_checkpoint and chunk_idx >= write_start
                            if do_checkpoint:
                                checkpoint_stage = checkpoint_cnt % cfg.smem_checkpoint_stages
                                checkpoint_phase_done = cutlass.Int32(1) ^ (
                                    (checkpoint_cnt // cfg.smem_checkpoint_stages)
                                    & cutlass.Int32(1)
                                )
                                bars.mb_checkpoint_tmastg_done[checkpoint_stage].wait(
                                    checkpoint_phase_done
                                )
                                checkpoint_stage_base = checkpoint_stage * cfg.d_k * cfg.d_v
                                if cutlass.const_expr(cfg.d_v == 64):
                                    for hh in cutlass.range_constexpr(state_halves_n):
                                        for j in cutlass.range_constexpr(16):
                                            ck_row = (
                                                state_sub_base
                                                + state_half_row
                                                + cutlass.Int32(8 * (j % 2))
                                            )
                                            col = state_half_col + cutlass.Int32(8 * (j // 2))
                                            checkpoint_addr = (
                                                checkpoint_stage_base
                                                + hh * (cfg.d_v * 64)
                                                + ck_row * 64
                                                + swizzle_xor_128b(ck_row, col)
                                            )
                                            (
                                                smem_data_ptr(sCheckpoint_raw) + checkpoint_addr
                                            ).store(
                                                cutlass.Vector.from_elements(
                                                    (state_pack[hh * 16 + j],), cutlass.Int32
                                                ).bitcast(cfg.io_dtype),
                                                alignment=4,
                                            )
                                elif state_row_valid:
                                    for i in cutlass.range_constexpr(num_ldtms):
                                        for g in cutlass.range_constexpr(ldtm_width // 8):
                                            packs = tuple(
                                                fp32_to_fp16(
                                                    state_regs[g * 8 + 2 * t][i],
                                                    state_regs[g * 8 + 2 * t + 1][i],
                                                    dtype=cfg.io_dtype,
                                                )
                                                for t in range(4)
                                            )
                                            col = i * ldtm_width + g * 8
                                            checkpoint_addr = (
                                                checkpoint_stage_base
                                                + swizzle_box_offset_128b(
                                                    state_gmem_row, col, box_rows=cfg.d_v
                                                )
                                            )
                                            (
                                                smem_data_ptr(sCheckpoint_raw) + checkpoint_addr
                                            ).store(
                                                cutlass.Vector.from_elements(
                                                    packs, cutlass.Int32
                                                ).bitcast(cfg.io_dtype),
                                                alignment=16,
                                            )
                                nvvm.fence_proxy("async.shared", space="cta")
                                if elect_one:
                                    bars.mb_checkpoint_tmastg_ready[checkpoint_stage].arrive()
                                checkpoint_cnt = checkpoint_cnt + 1

                        if cutlass.const_expr(cfg.d_v == 64):
                            for hh in cutlass.range_constexpr(state_halves_n):
                                state_scaled = []
                                for j in cutlass.range_constexpr(16):
                                    s0, s1 = fmul2(
                                        state_halves[hh][2 * j],
                                        state_halves[hh][2 * j + 1],
                                        cumprod_total,
                                        cumprod_total,
                                    )
                                    state_scaled += [s0, s1]
                                nvvm.tcgen05_st(
                                    "16x256b",
                                    nvvm.make_tmem_ptr(
                                        row_lo_addr + tmem_state_col + hh * 64, cutlass.Float32
                                    ),
                                    cutlass.Vector.from_elements(
                                        tuple(state_scaled), cutlass.Float32
                                    ),
                                )
                        else:
                            for i in cutlass.range_constexpr(num_ldtms):
                                state_scaled = []
                                for j in cutlass.range_constexpr(16):
                                    s0, s1 = fmul2(
                                        state_regs[2 * j][i],
                                        state_regs[2 * j + 1][i],
                                        cumprod_total,
                                        cumprod_total,
                                    )
                                    state_scaled += [s0, s1]
                                nvvm.tcgen05_st(
                                    "32x32b",
                                    nvvm.make_tmem_ptr(
                                        row_lo_addr + tmem_state_col + i * ldtm_width,
                                        cutlass.Float32,
                                    ),
                                    cutlass.Vector.from_elements(
                                        tuple(state_scaled), cutlass.Float32
                                    ),
                                )
                        nvvm.tcgen05_wait("store")

                    # ---- per-row Gate register builds --------------------------------
                    cumprod_vals = []
                    for k in cutlass.range_constexpr(num_vals):
                        cumprod_vals.append(
                            sCumprod[(lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2), 0, gate_idx]
                        )
                    last_cumsumlog = sCumsumlog[cfg.b_t - 1, 0, gate_idx]
                    cumsumlog_vals = []
                    for k in cutlass.range_constexpr(num_vals):
                        cumsumlog_vals.append(
                            sCumsumlog[(lane_idx % 4) * 2 + ((k // 4) * 8 + k % 2), 0, gate_idx]
                        )
                    decay_scale_vals = []
                    for k in cutlass.range_constexpr(0, num_vals, 2):
                        d0, d1 = fadd2(
                            last_cumsumlog,
                            last_cumsumlog,
                            -cumsumlog_vals[k],
                            -cumsumlog_vals[k + 1],
                        )
                        decay_scale_vals.append(cute.math.exp2(d0, fastmath=True))
                        decay_scale_vals.append(cute.math.exp2(d1, fastmath=True))
                    bars.mb_gate_done[gate_idx].arrive()

                    # ---- Y = V - k state (packed 16-bit) -----------------------------
                    v_idx = v_index.idx
                    bars.mb_v_ready[v_idx].wait(v_index.phase)
                    v_index = advance(v_index, cfg.smem_v_stages)

                    v_frag = []
                    for half in cutlass.range_constexpr(dv_halves):
                        v_words = []
                        for block in cutlass.range_constexpr(4):
                            v_raw = nvvm.ldmatrix(
                                (
                                    sV_base
                                    + v_idx * v_stage_elements
                                    + v_o_segment
                                    + (v_o_row + block * 16) * 64
                                    + swizzle_xor_128b(v_o_row + block * 16, v_o_col + half * 16)
                                ),
                                4,
                                nvvm.MMALayout.COL,
                            )
                            for i in cutlass.range_constexpr(4):
                                v_words.append(v_raw[i])
                        v_frag.append(v_words)
                    if valid_state:
                        bars.mb_k_state_acc_ready[0].wait(k_state_ready_index.phase)
                        k_state_ready_index = advance(k_state_ready_index, 1)

                        for half in cutlass.range_constexpr(dv_halves):
                            k_state_vec = nvvm.tcgen05_ld(
                                "16x256b",
                                nvvm.make_tmem_ptr(
                                    ((tmem_row + half * 16) << 16) + tmem_k_state_col,
                                    cutlass.Float32,
                                ),
                                num=8,
                            )
                            for j in cutlass.range_constexpr(16):
                                s0, s1 = fmul2(
                                    k_state_vec[2 * j],
                                    k_state_vec[2 * j + 1],
                                    cumprod_vals[2 * j],
                                    cumprod_vals[2 * j + 1],
                                )
                                k_state_word = fp32_to_fp16(s0, s1, dtype=cfg.io_dtype)
                                v_frag[half][j] = sub_f16x2(
                                    v_frag[half][j], k_state_word, cfg.io_dtype
                                )
                    for half in cutlass.range_constexpr(dv_halves):
                        nvvm.tcgen05_st(
                            "16x128b",
                            nvvm.make_tmem_ptr(
                                ((tmem_row + half * 16) << 16) + tmem_y_input_col, cutlass.Int32
                            ),
                            cutlass.Vector.from_elements(tuple(v_frag[half]), cutlass.Int32),
                        )
                    nvvm.tcgen05_wait("store")
                    bars.mb_y_input_ready[0].arrive()

                    # ---- q state epilogue: q state *= cumprod * scale ----------------
                    rescale_head = cutlass.Boolean(True)
                    if valid_state and rescale_head:
                        q_state_idx = o_acc_ready_index.idx
                        bars.mb_o_acc_ready[q_state_idx].wait(o_acc_ready_index.phase)
                        o_acc_ready_index = advance(o_acc_ready_index, cfg.tmem_q_state_acc_stages)

                        o_scale_vals = cumprod_vals
                        q_state_ptrs = []
                        q_state_vecs = []
                        for half in cutlass.range_constexpr(dv_halves):
                            q_state_ptrs.append(
                                nvvm.make_tmem_ptr(
                                    ((tmem_row + half * 16) << 16)
                                    + tmem_q_state_col
                                    + q_state_idx * ACC_STAGE_COLS,
                                    cutlass.Float32,
                                )
                            )
                            q_state_vecs.append(
                                nvvm.tcgen05_ld("16x256b", q_state_ptrs[half], num=8)
                            )
                        for half in cutlass.range_constexpr(dv_halves):
                            q_state_scaled = []
                            for j in cutlass.range_constexpr(16):
                                p0, p1 = fmul2(
                                    q_state_vecs[half][2 * j],
                                    q_state_vecs[half][2 * j + 1],
                                    o_scale_vals[2 * j],
                                    o_scale_vals[2 * j + 1],
                                )
                                s0, s1 = fmul2(p0, p1, scale, scale)
                                q_state_scaled += [s0, s1]
                            nvvm.tcgen05_st(
                                "16x256b",
                                q_state_ptrs[half],
                                cutlass.Vector.from_elements(
                                    tuple(q_state_scaled), cutlass.Float32
                                ),
                            )
                        nvvm.tcgen05_wait("store")
                        bars.mb_o_state_scale_acc_done[q_state_idx].arrive()

                    # ---- U epilogue + decayed-U publish ------------------------------
                    bars.mb_u_acc_ready[0].wait(u_acc_ready_index.phase)
                    u_acc_ready_index = advance(u_acc_ready_index, 1)
                    bars.mb_v_done[v_idx].arrive()

                    u_vecs = []
                    for half in cutlass.range_constexpr(dv_halves):
                        u_vecs.append(
                            nvvm.tcgen05_ld(
                                "16x256b",
                                nvvm.make_tmem_ptr(
                                    ((tmem_row + half * 16) << 16) + tmem_u_acc_col,
                                    cutlass.Float32,
                                ),
                                num=8,
                            )
                        )
                    u_regs = [[u_vecs[h][k] for h in range(dv_halves)] for k in range(32)]
                    for half in cutlass.range_constexpr(dv_halves):
                        u_pack = [
                            fp32_to_fp16(
                                u_regs[2 * j][half], u_regs[2 * j + 1][half], dtype=cfg.io_dtype
                            )
                            for j in range(16)
                        ]
                        nvvm.tcgen05_st(
                            "16x128b",
                            nvvm.make_tmem_ptr(
                                ((tmem_row + half * 16) << 16) + tmem_u_input_col, cutlass.Int32
                            ),
                            cutlass.Vector.from_elements(tuple(u_pack), cutlass.Int32),
                        )
                    nvvm.tcgen05_wait("store")
                    bars.mb_u_input_ready[0].arrive()

                    for half in cutlass.range_constexpr(dv_halves):
                        for j in cutlass.range_constexpr(16):
                            u_regs[2 * j][half], u_regs[2 * j + 1][half] = fmul2(
                                u_regs[2 * j][half],
                                u_regs[2 * j + 1][half],
                                decay_scale_vals[2 * j],
                                decay_scale_vals[2 * j + 1],
                            )
                        decay_pack = [
                            fp32_to_fp16(
                                u_regs[2 * j][half], u_regs[2 * j + 1][half], dtype=cfg.io_dtype
                            )
                            for j in range(16)
                        ]
                        nvvm.tcgen05_st(
                            "16x128b",
                            nvvm.make_tmem_ptr(
                                ((tmem_row + half * 16) << 16) + tmem_decay_v_col, cutlass.Int32
                            ),
                            cutlass.Vector.from_elements(tuple(decay_pack), cutlass.Int32),
                        )
                    nvvm.tcgen05_wait("store")
                    bars.mb_decay_u_input_ready[0].arrive()

                    # ---- output store: O acc TMEM -> SMEM ----------------------------
                    readout_tail = cutlass.Boolean(True)
                    if readout_tail:
                        o_scale_idx = o_final_acc_ready_index.idx
                        bars.mb_o_final_acc_ready[o_scale_idx].wait(o_final_acc_ready_index.phase)
                        o_final_acc_ready_index = advance(
                            o_final_acc_ready_index, cfg.tmem_q_state_acc_stages
                        )

                        o_regs = []
                        for half in cutlass.range_constexpr(dv_halves):
                            o_vec = nvvm.tcgen05_ld(
                                "16x256b",
                                nvvm.make_tmem_ptr(
                                    ((tmem_row + half * 16) << 16)
                                    + tmem_q_state_col
                                    + o_scale_idx * ACC_STAGE_COLS,
                                    cutlass.Float32,
                                ),
                                num=8,
                            )
                            o_regs.append([o_vec[k] for k in range(32)])
                        nvvm.tcgen05_wait("load")
                        if cutlass.const_expr(cfg.paged_state):
                            # Null-route padding may hold NaNs: select, never multiply, to emit
                            # zeros.
                            o_regs = [
                                [val if state_active else cutlass.Float32(0.0) for val in regs]
                                for regs in o_regs
                            ]
                        o_idx = o_index.idx
                        bars.mb_o_tmastg_done[o_idx].wait(o_index.phase)
                        o_index = advance(o_index, cfg.smem_o_stages)
                        for half in cutlass.range_constexpr(dv_halves):
                            for block in cutlass.range_constexpr(4):
                                o_pack = [
                                    fp32_to_fp16(
                                        o_regs[half][8 * block + 2 * j],
                                        o_regs[half][8 * block + 2 * j + 1],
                                        dtype=cfg.io_dtype,
                                    )
                                    for j in range(4)
                                ]
                                nvvm.stmatrix(
                                    (
                                        sO_base
                                        + o_idx * o_stage_elements
                                        + v_o_segment
                                        + (v_o_row + block * 16) * 64
                                        + swizzle_xor_128b(
                                            v_o_row + block * 16, v_o_col + half * 16
                                        )
                                    ),
                                    o_pack,
                                    nvvm.MMALayout.COL,
                                )
                        nvvm.fence_proxy("async.shared", space="cta")
                        bars.mb_o_tmastg_ready[o_idx].arrive()

        # ---- final state store: TMEM -> GMEM -----------------------------------------
        if n_local > 0:
            kv_last_idx = kv_acc_index.idx
            bars.mb_state_acc_ready[kv_last_idx].wait(kv_acc_index.phase)
            kv_acc_index = advance(kv_acc_index, cfg.tmem_state_acc_stages)
            if cutlass.const_expr(cfg.store_final_state):
                if cutlass.const_expr(cfg.paged_state):
                    store_final = state_active
                    final_row = seed_row
                else:
                    final_dst = mWorkItems[tile_idx, WORK_ITEM_FINAL_DST]
                    store_final = final_dst >= 0
                    final_row = final_dst
                    if cutlass.const_expr(mFinalIndices is not None):
                        if store_final:
                            final_row = cutlass.Int32(mFinalIndices[final_dst])
                if store_final:
                    gState_out = mState_out[None, None, head_o, final_row]
                    state_vw = 16 // (mState_out.element_type.width // 8)
                    state_dst = (
                        gState_out.iterator + gState_out.layout((state_gmem_row + v_offset, 0))
                    ).raw_ptr()
                    for i in cutlass.range_constexpr(num_ldtms):
                        state_vec = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                row_lo_addr + tmem_state_col + i * ldtm_width, cutlass.Float32
                            ),
                            num=32,
                        )
                        if state_row_valid:
                            for g in cutlass.range_constexpr(ldtm_width // state_vw):
                                (state_dst + i * ldtm_width + g * state_vw).store(
                                    cutlass.Vector.from_elements(
                                        tuple(
                                            state_vec[g * state_vw + t].to(mState_out.element_type)
                                            for t in range(state_vw)
                                        ),
                                        mState_out.element_type,
                                    ),
                                    alignment=16,
                                )
        else:
            if cutlass.const_expr(cfg.store_final_state):
                if cutlass.const_expr(cfg.paged_state):
                    # Only a fresh route clears its slot; resumed and null routes leave the pool.
                    write_empty = state_active and clear_empty
                    final_row = seed_row
                else:
                    final_dst = mWorkItems[tile_idx, WORK_ITEM_FINAL_DST]
                    write_empty = final_dst >= 0
                    final_row = final_dst
                    if cutlass.const_expr(mFinalIndices is not None):
                        if write_empty:
                            final_row = cutlass.Int32(mFinalIndices[final_dst])
                if write_empty:
                    gState_out = mState_out[None, None, head_o, final_row]
                    if cutlass.const_expr(cfg.use_initial_state and not cfg.paged_state):
                        gState_in = mState_init[None, None, head_o, seed_row]
                        for i in cutlass.range_constexpr(num_ldtms):
                            for k in cutlass.range_constexpr(32):
                                if state_row_valid:
                                    seed_val = gState_in[
                                        state_gmem_row + v_offset, i * ldtm_width + k
                                    ]
                                    if cutlass.const_expr(
                                        mState_init.element_type != mState_out.element_type
                                    ):
                                        seed_val = seed_val.to(mState_out.element_type)
                                    gState_out[state_gmem_row + v_offset, i * ldtm_width + k] = (
                                        seed_val
                                    )
                    else:
                        for i in cutlass.range_constexpr(num_ldtms):
                            for k in cutlass.range_constexpr(32):
                                if state_row_valid:
                                    gState_out[state_gmem_row + v_offset, i * ldtm_width + k] = (
                                        cutlass.Float32(0.0).to(mState_out.element_type)
                                    )

        tile_idx, scheduler_state = scheduler_next_tile(
            cfg, bars, sScheduler, scheduler_state, elect_one
        )

    bars.mb_tmem_done[0].arrive()

    for _ in range(cfg.smem_o_stages):
        bars.mb_o_tmastg_done[o_index.idx].wait(o_index.phase)
        o_index = advance(o_index, cfg.smem_o_stages)
    if cutlass.const_expr(cfg.enable_checkpoints):
        for _ in range(cfg.smem_checkpoint_stages):
            checkpoint_stage = checkpoint_cnt % cfg.smem_checkpoint_stages
            checkpoint_phase_done = cutlass.Int32(1) ^ (
                (checkpoint_cnt // cfg.smem_checkpoint_stages) & cutlass.Int32(1)
            )
            bars.mb_checkpoint_tmastg_done[checkpoint_stage].wait(checkpoint_phase_done)
            checkpoint_cnt = checkpoint_cnt + 1


@cute.jit
def build_descs_body(
    widx,
    base_q,
    base_k,
    base_v,
    base_o,
    base_checkpoint,
    base_tinv,
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor | None,
    state_checkpoints_out: cute.Tensor | None,
    tinv: cute.Tensor | None,
    n_batch: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    b_t: cutlass.Constexpr[int],
) -> None:
    """Per-batch descriptor-array build, one warp per array (the tinv array on the K warp). Runs
    inside the prologue kernel after its order pass; warps past the array count fall through the
    widx guards."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    sub0 = cute.make_tensor(desc_workspace.iterator, cute.make_layout((arr_words,), stride=(1,)))
    sub1 = cute.make_tensor(
        desc_workspace.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    sub2 = cute.make_tensor(
        desc_workspace.iterator + 2 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    sub3 = cute.make_tensor(
        desc_workspace.iterator + 3 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    sub4 = cute.make_tensor(
        desc_workspace.iterator + 4 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    sub5 = cute.make_tensor(
        desc_workspace.iterator + 5 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )

    if widx == 0:
        emit_seq_descs(base_q, sub0, cu_seqlens, q, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 1:
        emit_seq_descs(base_k, sub1, cu_seqlens, k, n_batch, 2, lanes=32)
        if cutlass.const_expr(tinv is not None):
            emit_tile_seq_descs(base_tinv, sub5, cu_seqlens, tinv, n_batch, b_t, 3, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 2:
        emit_seq_descs(base_v, sub2, cu_seqlens, v, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if cutlass.const_expr(o is not None):
        if widx == 3:
            emit_seq_descs(base_o, sub3, cu_seqlens, o, n_batch, 2, lanes=32)
            nvvm.fence_proxy_release(
                nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
            )
    if cutlass.const_expr(state_checkpoints_out is not None):
        if widx == 4:
            emit_checkpoint_seq_descs(
                base_checkpoint,
                sub4,
                cu_seqlens,
                state_checkpoints_out,
                n_batch,
                checkpoint_every_n,
                2,
                lanes=32,
            )
            nvvm.fence_proxy_release(
                nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
            )


@cute.kernel
def frost_gdn_prefill_prologue(
    order_gen: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    tiles_per_head: cutlass.Constexpr[int],
    base_q: cutlass.GridConstant[tma.TensorMap],
    base_k: cutlass.GridConstant[tma.TensorMap],
    base_v: cutlass.GridConstant[tma.TensorMap],
    base_o: cutlass.GridConstant[tma.TensorMap],
    base_checkpoint: cutlass.GridConstant[tma.TensorMap],
    base_tinv: cutlass.GridConstant[tma.TensorMap],
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor | None,
    state_checkpoints_out: cute.Tensor | None,
    tinv: cute.Tensor | None,
    mStaging: cute.Tensor | None,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mScheduler: cute.Tensor,
    n_batch: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    mStateIndices: cute.Tensor | None,
    mHasInitialState: cute.Tensor | None,
) -> None:
    """Two-CTA prologue: block 0 LPT-orders the work-item table and zeroes the
    scheduler rings via :func:`order_body`, block 1 builds the per-batch
    TMA-descriptor arrays via :func:`build_descs_body`, one warp per array."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
        launch_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    if bidx == cutlass.Int32(0):
        sKey = cutlass.Array(
            cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16
        )
        sIdx = cutlass.Array(
            cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16
        )
        sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
        n_heads_out = cutlass.Int32(q.shape[1] if q.shape[1] >= v.shape[1] else v.shape[1])
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
            mStateIndices=mStateIndices,
            mHasInitialState=mHasInitialState,
        )
    else:
        build_descs_body(
            widx,
            base_q,
            base_k,
            base_v,
            base_o,
            base_checkpoint,
            base_tinv,
            desc_workspace,
            cu_seqlens,
            q,
            k,
            v,
            o,
            state_checkpoints_out,
            tinv,
            n_batch,
            checkpoint_every_n,
            b_t,
        )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    order_gen: cutlass.Constexpr[bool],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: cute.Tensor | None,
    cu_seqlens: cute.Tensor,
    state_checkpoints_out: cute.Tensor | None,
    work_item_staging: cute.Tensor | None,
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler_counter: cute.Tensor,
    checkpoint_every_n: cutlass.Int32,
    tinv: cute.Tensor | None,
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
    tiles_per_head: cutlass.Constexpr[int] = 1,
    state_indices: cute.Tensor | None = None,
    has_initial_state: cute.Tensor | None = None,
):
    """One-launch prologue: LPT-order the work items and build the per-(b,h)
    TMA-descriptor arrays (Q, K, V, O, checkpoints) into
    ``tensormap_workspace``.  Paged ``state_indices`` / ``has_initial_state`` keep the empty
    sequences whose fresh slot the main kernel must clear in the compacted uncut table."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    h_v = v.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = h_q if h_q >= h_v else h_v
    d_v = v.shape[2]
    bytes_per_element = io_dtype.width // 8
    box_elems = 128 // bytes_per_element
    bt = b_t

    q_row_stride, q_head_stride = q.stride[0], q.stride[1]
    k_row_stride, k_head_stride = k.stride[0], k.stride[1]
    v_row_stride, v_head_stride = v.stride[0], v.stride[1]

    seqlen = q.shape[0]
    d_k = q.shape[2]
    q_headed = cute.make_tensor(
        q.iterator, cute.make_layout((seqlen, h_q, d_k), stride=(q_row_stride, q_head_stride, 1))
    )
    k_headed = cute.make_tensor(
        k.iterator, cute.make_layout((seqlen, h_k, d_k), stride=(k_row_stride, k_head_stride, 1))
    )
    v_headed = cute.make_tensor(
        v.iterator, cute.make_layout((d_v, h_v, seqlen), stride=(1, v_head_stride, v_row_stride))
    )
    swizzle_128b = tma.TensorMapSwizzle.s128b
    base_desc_q = tma.create_tensor_map_tiled_from_view(
        q_headed, box_dims=(bt, 1, box_elems), stride_order=(2, 1, 0), swizzle=swizzle_128b
    )
    base_desc_k = tma.create_tensor_map_tiled_from_view(
        k_headed, box_dims=(bt, 1, box_elems), stride_order=(2, 1, 0), swizzle=swizzle_128b
    )
    base_desc_v = tma.create_tensor_map_tiled_from_view(
        v_headed, box_dims=(box_elems, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b
    )

    base_desc_o = base_desc_v
    if cutlass.const_expr(o is not None):
        o_headed = cute.make_tensor(
            o.iterator,
            cute.make_layout((d_v, heads_out, o.shape[0]), stride=(1, o.stride[1], o.stride[0])),
        )
        base_desc_o = tma.create_tensor_map_tiled_from_view(
            o_headed, box_dims=(box_elems, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b
        )

    base_desc_checkpoint = base_desc_v
    if cutlass.const_expr(state_checkpoints_out is not None):
        d_v_state = state_checkpoints_out.shape[2]
        d_k_state = state_checkpoints_out.shape[3]
        checkpoint_box_elems = 128 // (state_checkpoints_out.element_type.width // 8)
        checkpoint_view = cute.make_tensor(
            state_checkpoints_out.iterator,
            cute.make_layout(
                (d_k_state, d_v_state, state_checkpoints_out.shape[0], heads_out),
                stride=(
                    state_checkpoints_out.stride[3],
                    state_checkpoints_out.stride[2],
                    state_checkpoints_out.stride[0],
                    state_checkpoints_out.stride[1],
                ),
            ),
        )
        base_desc_checkpoint = tma.create_tensor_map_tiled_from_view(
            checkpoint_view,
            box_dims=(checkpoint_box_elems, d_v_state // tiles_per_head, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=swizzle_128b,
        )

    base_desc_tinv = base_desc_k
    if cutlass.const_expr(tinv is not None):
        tinv_tiles = cute.make_tensor(
            tinv.iterator,
            cute.make_layout(
                (tinv.shape[0], tinv.shape[1], tinv.shape[2], tinv.shape[3]),
                stride=(tinv.stride[0], tinv.stride[1], tinv.stride[2], 1),
            ),
        )
        base_desc_tinv = tma.create_tensor_map_tiled_from_view(
            tinv_tiles, box_dims=(1, 1, bt, bt), stride_order=(3, 2, 1, 0), swizzle=swizzle_128b
        )
    frost_gdn_prefill_prologue(
        order_gen,
        b_t,
        tiles_per_head,
        base_desc_q,
        base_desc_k,
        base_desc_v,
        base_desc_o,
        base_desc_checkpoint,
        base_desc_tinv,
        tensormap_workspace,
        cu_seqlens,
        q,
        k,
        v,
        o,
        state_checkpoints_out,
        tinv,
        work_item_staging,
        work_count,
        work_items,
        scheduler_counter,
        cutlass.Int32(batch_size),
        checkpoint_every_n,
        state_indices,
        has_initial_state,
    ).launch(grid=(2, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    beta: cute.Tensor | None,
    o: cute.Tensor | None,
    cu_seqlens: cute.Tensor,
    state_in: cute.Tensor | None,
    state_out: cute.Tensor | None,
    seed_indices: cute.Tensor | None,
    final_indices: cute.Tensor | None,
    has_initial_state: cute.Tensor | None,
    tinv: cute.Tensor | None,
    work_items: cute.Tensor | None,
    work_count: cute.Tensor | None,
    scheduler_counter: cute.Tensor,
    checkpoint_every_n_tokens: cutlass.Int32,
    scale: cutlass.Float32,
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
):
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = cutlass.Int32(gate.shape[1])
    if cutlass.const_expr(cfg.tiles_per_head > 1):
        heads_out = heads_out * cutlass.Int32(cfg.tiles_per_head)
    q_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(q.shape[1]))
    k_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(k.shape[1]))
    v_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(v.shape[1]))

    # ---- head-major views: (tokens, dim, heads) k / q, (dim, tokens, heads) v / o, (V, K, heads,
    # seqs) states
    q = cute.make_tensor(
        q.iterator,
        cute.make_layout(
            (q.shape[0], q.shape[2], q.shape[1]), stride=(q.stride[0], q.stride[2], q.stride[1])
        ),
    )
    k = cute.make_tensor(
        k.iterator,
        cute.make_layout(
            (k.shape[0], k.shape[2], k.shape[1]), stride=(k.stride[0], k.stride[2], k.stride[1])
        ),
    )
    v = cute.make_tensor(
        v.iterator,
        cute.make_layout(
            (v.shape[2], v.shape[0], v.shape[1]), stride=(v.stride[2], v.stride[0], v.stride[1])
        ),
    )
    if cutlass.const_expr(o is not None):
        o = cute.make_tensor(
            o.iterator,
            cute.make_layout(
                (o.shape[2], o.shape[0], o.shape[1]),
                stride=(o.stride[2], o.stride[0], o.stride[1]),
            ),
        )
    if cutlass.const_expr(state_in is not None):
        state_in = cute.make_tensor(
            state_in.iterator,
            cute.make_layout(
                (state_in.shape[2], state_in.shape[3], state_in.shape[1], state_in.shape[0]),
                stride=(
                    state_in.stride[2],
                    state_in.stride[3],
                    state_in.stride[1],
                    state_in.stride[0],
                ),
            ),
        )
    if cutlass.const_expr(state_out is not None):
        state_out = cute.make_tensor(
            state_out.iterator,
            cute.make_layout(
                (state_out.shape[2], state_out.shape[3], state_out.shape[1], state_out.shape[0]),
                stride=(
                    state_out.stride[2],
                    state_out.stride[3],
                    state_out.stride[1],
                    state_out.stride[0],
                ),
            ),
        )

    cumsumlog_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_gate_stages))
    beta_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_beta_stages))

    # Two tile structs keep the v1.30 SMEM order: O, checkpoint and K+Q, then the barrier and
    # staging arrays, then T_inv, A and V. Moving the arrays ahead of the tiles slows the kernel
    # (~3% at 10x4096: 3.7x the shared-load bank conflicts).
    @cute.struct
    class LeadStorage:
        output: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.o_cosize], cfg.buffer_align_bytes
        ]
        checkpoint: cute.struct.Align[
            cute.struct.MemRange[
                cfg.io_dtype, cfg.checkpoint_cosize if cfg.enable_checkpoints else 0
            ],
            cfg.buffer_align_bytes,
        ]
        kq: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.kq_cosize], cfg.buffer_align_bytes
        ]

    @cute.struct
    class TailStorage:
        t_inv: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.t_inv_cosize], cfg.buffer_align_bytes
        ]
        a: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.a_cosize], cfg.buffer_align_bytes
        ]
        v: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.v_cosize], cfg.buffer_align_bytes
        ]

    num_descs = batch_size

    # ---- launch ----------------------------------------------------------------------
    grid_shape = (cfg.max_active_clusters, 1, 1)

    frost_gdn_prefill(
        cfg,
        LeadStorage,
        TailStorage,
        q_ratio,
        k_ratio,
        v_ratio,
        gate,
        beta,
        cu_seqlens,
        state_in,
        state_out,
        seed_indices,
        final_indices,
        has_initial_state,
        tinv,
        work_items,
        work_count,
        scheduler_counter,
        checkpoint_every_n_tokens,
        scale,
        cumsumlog_smem_layout_staged,
        beta_smem_layout_staged,
        q,
        k,
        v,
        o,
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
def frost_gdn_prefill(
    cfg: cutlass.Constexpr,
    lead_storage_type: cutlass.Constexpr,
    tail_storage_type: cutlass.Constexpr,
    q_ratio: cute.FastDivmodDivisorV2,
    k_ratio: cute.FastDivmodDivisorV2,
    v_ratio: cute.FastDivmodDivisorV2,
    mGate: cute.Tensor,
    mBeta: cute.Tensor | None,
    cu_seqlens: cute.Tensor,
    mState_init: cute.Tensor | None,
    mState_out: cute.Tensor | None,
    mSeedIndices: cute.Tensor | None,
    mFinalIndices: cute.Tensor | None,
    mHasInitialState: cute.Tensor | None,
    mTinv: cute.Tensor | None,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor,
    checkpoint_every_n_tokens: cutlass.Int32,
    scale: cutlass.Float32,
    cumsumlog_smem_layout_staged: cute.Layout,
    beta_smem_layout_staged: cute.Layout,
    mQ,
    mK,
    mV,
    mO,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
):
    """Main GDN chunked kernel: warp-specialized persistent tile loop."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]

    total_tiles = mCount[0]
    if cutlass.const_expr(cfg.paged_state):
        assert mSeedIndices is not None, "paged_state routes every sequence through mSeedIndices"
        assert cfg.use_initial_state and cfg.store_final_state, (
            "paged_state reads and writes the pool"
        )
    else:
        assert mHasInitialState is None, "mHasInitialState requires paged_state"

    desc_base_words = tensormap_workspace.iterator.raw_ptr()
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)
    arr_words = n_desc * desc_qwords
    desc_q_base = desc_base_words
    desc_k_base = desc_base_words + arr_words
    desc_v_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_o_base = desc_base_words + cutlass.Int32(3) * arr_words
    desc_checkpoint_base = desc_base_words + cutlass.Int32(4) * arr_words
    desc_tinv_base = desc_base_words + cutlass.Int32(5) * arr_words

    SMEM = cutlass.AddressSpace.smem

    SWZ = 2
    LEAD = 16
    STRIDE = 8 * 128
    KT_LEAD = cfg.b_t * 128
    V_LEAD = cfg.b_t * 128
    # SMEM in v1.30 order (see LeadStorage): lead tiles, barrier/scheduler/staging arrays, tail
    # tiles.
    lead = SmemAllocator().allocate(lead_storage_type)
    sO_raw = lead.output.get_tensor(cute.make_layout((cfg.o_cosize,)))
    sO = SmemTile(
        base=sO_raw,
        elems_per_stage=(cfg.o_cosize // cfg.smem_o_stages),
        stages=cfg.smem_o_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    if cutlass.const_expr(cfg.enable_checkpoints):
        sCheckpoint_raw = lead.checkpoint.get_tensor(cute.make_layout((cfg.checkpoint_cosize,)))
    else:
        sCheckpoint_raw = None
    sKQ_raw = lead.kq.get_tensor(cute.make_layout((cfg.kq_cosize,)))
    sKQ = SmemTile(
        base=sKQ_raw,
        elems_per_stage=(cfg.kq_cosize // cfg.smem_kq_stages),
        stages=cfg.smem_kq_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sKQ_trans = SmemTile(
        base=sKQ_raw,
        elems_per_stage=(cfg.kq_cosize // cfg.smem_kq_stages),
        stages=cfg.smem_kq_stages,
        leading_byte_offset=2 * KT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    bars = make_bars(cfg)
    tmem_base_slot = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=16)
    sScheduler = cutlass.Array(cutlass.Int32, cfg.scheduler_stages, space=SMEM, alignment=16)
    cumsumlog_raw = cutlass.Array(
        cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128
    )
    cumprod_raw = cutlass.Array(
        cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128
    )
    beta_raw = cutlass.Array(
        cutlass.Float32, cute.cosize(beta_smem_layout_staged), space=SMEM, alignment=128
    )
    tail = SmemAllocator().allocate(tail_storage_type)
    sTinv_raw = tail.t_inv.get_tensor(cute.make_layout((cfg.t_inv_cosize,)))
    sTinv = SmemTile(
        base=sTinv_raw,
        elems_per_stage=(cfg.t_inv_cosize // cfg.smem_t_inv_stages),
        stages=cfg.smem_t_inv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sA_raw = tail.a.get_tensor(cute.make_layout((cfg.a_cosize,)))
    sA = SmemTile(
        base=sA_raw,
        elems_per_stage=(cfg.a_cosize // cfg.smem_a_stages),
        stages=cfg.smem_a_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sV_raw = tail.v.get_tensor(cute.make_layout((cfg.v_cosize,)))
    sV_trans = SmemTile(
        base=sV_raw,
        elems_per_stage=(cfg.v_cosize // cfg.smem_v_stages),
        stages=cfg.smem_v_stages,
        leading_byte_offset=V_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sCumsumlog = cute.make_tensor(
        cute.make_ptr(
            cutlass.Float32,
            smem_data_ptr(cumsumlog_raw).toint(),
            mem_space=cute.AddressSpace.smem,
            assumed_align=128,
        ),
        cumsumlog_smem_layout_staged,
    )
    sCumprod = cute.make_tensor(
        cute.make_ptr(
            cutlass.Float32,
            smem_data_ptr(cumprod_raw).toint(),
            mem_space=cute.AddressSpace.smem,
            assumed_align=128,
        ),
        cumsumlog_smem_layout_staged,
    )
    sBeta = cute.make_tensor(
        cute.make_ptr(
            cutlass.Float32,
            smem_data_ptr(beta_raw).toint(),
            mem_space=cute.AddressSpace.smem,
            assumed_align=128,
        ),
        beta_smem_layout_staged,
    )

    # ---- mbarrier init (one thread; every thread fences and syncs below) ----------------
    # Attention Gym modification: a single initializer instead of a racy all-thread init.
    if tidx == 0:
        for s in range(cfg.smem_kq_stages):
            bars.mb_kq_ready[s].init()
            bars.mb_kq_done[s].init()
        for s in range(cfg.smem_v_stages):
            bars.mb_v_ready[s].init()
            bars.mb_v_done[s].init()
        for s in range(cfg.smem_gate_stages):
            bars.mb_gate_ready[s].init()
            bars.mb_gate_done[s].init()
        for s in range(cfg.smem_beta_stages):
            bars.mb_beta_ready[s].init()
            bars.mb_beta_done[s].init()
        for s in range(cfg.tmem_state_acc_stages):
            bars.mb_state_acc_ready[s].init()
        for s in range(cfg.tmem_q_state_acc_stages):
            bars.mb_o_acc_ready[s].init()
            bars.mb_o_final_acc_ready[s].init()
            bars.mb_o_state_scale_acc_done[s].init()
        for s in range(cfg.tmem_cg0_acc_stages):
            bars.mb_cg0_acc_ready[s].init()
            bars.mb_cg0_acc_done[s].init()
        bars.mb_k_state_acc_ready[0].init()
        bars.mb_u_acc_ready[0].init()
        for s in range(cfg.smem_t_inv_stages):
            bars.mb_t_inv_ready[s].init()
            bars.mb_t_inv_done[s].init()
        for s in range(cfg.smem_a_stages):
            bars.mb_a_ready[s].init()
            bars.mb_a_done[s].init()
        for s in range(cfg.tmem_state_input_stages):
            bars.mb_state_input_ready[s].init()
        for b in (bars.mb_y_input_ready, bars.mb_u_input_ready, bars.mb_decay_u_input_ready):
            b[0].init()
        for s in range(cfg.smem_o_stages):
            bars.mb_o_tmastg_ready[s].init()
            bars.mb_o_tmastg_done[s].init()
        for s in range(cfg.smem_checkpoint_stages):
            bars.mb_checkpoint_tmastg_ready[s].init()
            bars.mb_checkpoint_tmastg_done[s].init()
        for s in range(cfg.scheduler_stages):
            bars.mb_scheduler_ready[s].init()
            bars.mb_scheduler_done[s].init()
        bars.mb_tmem_done[0].init()

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # ---- warp specialization ---------------------------------------------------------
    if (
        warp_idx >= cfg.compute_group_0_warp_ids[0]
        and warp_idx <= cfg.compute_group_0_warp_ids[-1]
    ):
        if cutlass.const_expr(cfg.tinv_source == "gmem"):
            compute0_warp_group_gmem(
                cfg,
                total_tiles,
                bidx,
                num_ctas,
                cu_seqlens,
                mWorkItems,
                tidx,
                warp_idx,
                tmem_base_slot=tmem_base_slot,
                scale=scale,
                sCumsumlog=sCumsumlog,
                sA=sA,
                sScheduler=sScheduler,
                bars=bars,
            )
        else:
            compute0_warp_group(
                cfg,
                total_tiles,
                bidx,
                num_ctas,
                cu_seqlens,
                mWorkItems,
                tidx,
                warp_idx,
                tmem_base_slot=tmem_base_slot,
                scale=scale,
                sCumsumlog=sCumsumlog,
                sBeta=sBeta,
                sTinv=sTinv,
                sA=sA,
                sScheduler=sScheduler,
                bars=bars,
            )

    if (
        warp_idx >= cfg.compute_group_1_warp_ids[0]
        and warp_idx <= cfg.compute_group_1_warp_ids[-1]
    ):
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
            sV_trans=sV_trans,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sO=sO,
            sCheckpoint_raw=sCheckpoint_raw,
            mState_init=mState_init,
            mState_out=mState_out,
            mSeedIndices=mSeedIndices,
            mFinalIndices=mFinalIndices,
            mHasInitialState=mHasInitialState,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            sScheduler=sScheduler,
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
            mBeta=mBeta,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sBeta=sBeta,
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
            sKQ=sKQ,
            sKQ_trans=sKQ_trans,
            sTinv=sTinv,
            sA=sA,
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
            sKQ_raw=sKQ_raw,
            sV_raw=sV_raw,
            sTinv_raw=sTinv_raw,
            desc_tinv_base=desc_tinv_base,
            tidx=tidx,
            desc_q_base=desc_q_base,
            desc_k_base=desc_k_base,
            desc_v_base=desc_v_base,
            mScheduler=mScheduler,
            sScheduler=sScheduler,
            bars=bars,
            q_ratio=q_ratio,
            k_ratio=k_ratio,
            v_ratio=v_ratio,
        )

    if warp_idx == cfg.epilogue_warp_id:
        tmastg_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            tidx=tidx,
            sO_raw=sO_raw,
            sCheckpoint_raw=sCheckpoint_raw,
            desc_o_base=desc_o_base,
            desc_checkpoint_base=desc_checkpoint_base,
            sScheduler=sScheduler,
            bars=bars,
        )


@dataclass(frozen=True)
class GdnPrefillCfg:
    """Per-compile GDN kernel knob, built by ``build_cfg``.

    The per-compile parameters (dtypes, GQA, state flags) are the
    ``cute.compile`` cache keys; the rest is derived from the module-global
    ``CFG`` constants.  Passed ``cfg``-first (a ``cutlass.Constexpr``) into ``host`` /
    ``kernel`` and every warp body.
    """

    io_dtype: type[cutlass.Numeric]
    acc_dtype: type[cutlass.Numeric]
    state_dtype: type[cutlass.Numeric]
    max_active_clusters: int
    use_initial_state: bool
    store_final_state: bool
    enable_checkpoints: bool
    d_k: int
    d_v: int
    log_gate: bool = False
    tinv_source: str = "compute"
    tiles_per_head: int = 1
    # Attention Gym: the seed and final indices address one mutable pool (see
    # common/paged_state.py).
    paged_state: bool = False
    scheduler_stages: int = CFG.SMEM_SCHEDULER_STAGES

    # ---- fixed constants stamped from CFG at build time ------------------------------
    b_t: int = CFG.B_T
    compute_group_0_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_1_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    load_gate_warp_id: int = CFG.LOAD_GATE_WARP_ID
    tma_qkv_warp_id: int = CFG.TMA_QKV_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_other: int = CFG.NUM_REGS_OTHER
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0
    cluster_shape_mnk: tuple[int, int, int] = CFG.CLUSTER_SHAPE_MNK

    # ---- named barrier slots (ids 1-4; 0 is the CTA-wide sync) -----------------------
    tmem_lifecycle_barrier_id: int = 1
    tmem_user_threads: int = 0
    inverse_barrier_id: int = 2
    inverse_barrier_threads: int = 0

    # ---- SMEM / TMEM stage counts + TMEM column offsets ------------------------------
    smem_kq_stages: int = CFG.SMEM_KQ_STAGES
    smem_v_stages: int = CFG.SMEM_V_STAGES
    smem_t_inv_stages: int = CFG.SMEM_T_INV_STAGES
    smem_a_stages: int = CFG.SMEM_A_STAGES
    smem_o_stages: int = CFG.SMEM_O_STAGES
    smem_checkpoint_stages: int = 1
    smem_gate_stages: int = CFG.SMEM_GATE_STAGES
    smem_beta_stages: int = 3
    tmem_state_acc_stages: int = CFG.TMEM_KV_ACC_STAGES
    tmem_q_state_acc_stages: int = CFG.TMEM_Q_STATE_ACC_STAGES
    tmem_state_input_stages: int = CFG.TMEM_STATE_INP_STAGES
    tmem_cg0_acc_stages: int = CFG.TMEM_CG0_ACC_STAGES
    tmem_cg1_acc_stages: int = CFG.TMEM_CG1_ACC_STAGES
    tmem_state_acc_offset: int = 0
    tmem_q_state_acc_offset: int = 0
    tmem_state_input_offset: int = 0
    tmem_cg0_acc_offset: int = 0
    tmem_cg1_acc_offset: int = 0
    tmem_y_decay_u_input_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- derived by build_cfg (SMEM cosizes, TMA transaction bytes) --------------------
    kq_cosize: int = 0
    v_cosize: int = 0
    t_inv_cosize: int = 0
    a_cosize: int = 0
    o_cosize: int = 0
    checkpoint_cosize: int = 0
    tma_kq_bytes: int = 0
    tma_v_bytes: int = 0
    tma_o_bytes: int = 0
    tma_tinv_bytes: int = 0


def build_cfg(
    io_dtype: type[cutlass.Numeric],
    state_dtype: type[cutlass.Numeric],
    *,
    max_active_clusters: int,
    use_initial_state: bool,
    store_final_state: bool = True,
    enable_checkpoints: bool = False,
    log_gate: bool = False,
    tinv_source: str = "compute",
    d_k: int,
    d_v: int,
    tiles_per_head: int = 1,
    paged_state: bool = False,
) -> GdnPrefillCfg:
    """Build the per-compile ``GdnPrefillCfg`` (io_dtype in {Float16, BFloat16}; acc is
    always Float32).  ``tinv_source`` "compute" builds the chunk factor in compute
    group 0; "gmem" bulk-loads the tiles of ``gdn_tinv_f16.py``.  ``tiles_per_head`` > 1 runs every
    gate head as ``tiles_per_head`` tiles of ``d_v`` value columns each (the d_v split)."""
    if tinv_source not in TINV_SOURCES:
        raise ValueError(f"tinv_source must be one of {TINV_SOURCES}, got {tinv_source!r}")
    if paged_state and not (use_initial_state and store_final_state):
        raise ValueError(
            "paged_state reads and writes one state pool: use_initial_state and store_final_state"
        )
    stages = {"smem_checkpoint_stages": 1}
    if enable_checkpoints and d_v != 64:
        stages["smem_kq_stages"] = 3
    cfg = GdnPrefillCfg(
        io_dtype=io_dtype,
        acc_dtype=cutlass.Float32,
        state_dtype=state_dtype,
        max_active_clusters=max_active_clusters,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        log_gate=log_gate,
        tinv_source=tinv_source,
        d_k=d_k,
        d_v=d_v,
        tiles_per_head=tiles_per_head,
        paged_state=paged_state,
        **stages,
    )
    n_cg0 = len(cfg.compute_group_0_warp_ids)
    n_cg1 = len(cfg.compute_group_1_warp_ids)
    tmem_q_state_acc_offset = cfg.tmem_state_acc_stages * cfg.d_k
    tmem_state_input_offset = tmem_q_state_acc_offset + cfg.tmem_q_state_acc_stages * cfg.b_t
    tmem_cg0_acc_offset = tmem_state_input_offset + cfg.tmem_state_input_stages * (cfg.d_k // 2)
    tmem_cg1_acc_offset = tmem_cg0_acc_offset + cfg.tmem_cg0_acc_stages * cfg.b_t
    # ---- SMEM sizing: per-buffer element cosizes and TMA transaction bytes -----------------
    bytes_per_element = io_dtype.width // 8
    kq_tile_elements = 2 * cfg.b_t * cfg.d_k
    v_tile_elements = cfg.d_v * cfg.b_t
    tinv_tile_elements = cfg.b_t * cfg.b_t
    a_tile_elements = cfg.b_t * cfg.b_t
    o_tile_elements = cfg.d_v * cfg.b_t
    return replace(
        cfg,
        threads_per_cta=cfg.threads_per_warp * (4 + n_cg0 + n_cg1),
        tmem_user_threads=cfg.threads_per_warp * (1 + n_cg0 + n_cg1),
        inverse_barrier_threads=cfg.threads_per_warp * n_cg0,
        tmem_state_acc_offset=0,
        tmem_q_state_acc_offset=tmem_q_state_acc_offset,
        tmem_state_input_offset=tmem_state_input_offset,
        tmem_cg0_acc_offset=tmem_cg0_acc_offset,
        tmem_cg1_acc_offset=tmem_cg1_acc_offset,
        tmem_y_decay_u_input_offset=tmem_cg1_acc_offset + cfg.tmem_cg1_acc_stages * cfg.b_t,
        kq_cosize=kq_tile_elements * cfg.smem_kq_stages,
        v_cosize=v_tile_elements * cfg.smem_v_stages,
        t_inv_cosize=tinv_tile_elements * cfg.smem_t_inv_stages,
        a_cosize=a_tile_elements * cfg.smem_a_stages,
        o_cosize=o_tile_elements * cfg.smem_o_stages,
        checkpoint_cosize=cfg.d_k * cfg.d_v * cfg.smem_checkpoint_stages,
        tma_kq_bytes=kq_tile_elements * bytes_per_element,
        tma_v_bytes=v_tile_elements * bytes_per_element,
        tma_o_bytes=o_tile_elements * bytes_per_element,
        tma_tinv_bytes=tinv_tile_elements * bytes_per_element,
    )


TENSORMAP_DESC_ARRAYS = 6  # per-batch runtime TMA descriptors: Q, K, V, O, checkpoints, tinv


# ---------------------------------------------------------------------------


frost_gdn_prefill_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_gdn_prefill.set_name_prefix("cudnn", remove_cutlass_symbol=False)
