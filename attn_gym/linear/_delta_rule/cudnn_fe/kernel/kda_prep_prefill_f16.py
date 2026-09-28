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
# attn_gym.linear._delta_rule.cudnn_fe; register arrays, SMEM storage, and swizzle offsets restyled.

"""
Prep-fed chunked Kimi Delta Attention (KDA) prefill for SM100 / SM103 / SM107: the BT = 16 recurrence over the records of
kda_prep_f16 (K decay k_decay, Q decay q_decay, T, A, diag), one CTA per (sequence, head, d_v half) at the tile counts where the slot budget is 2, with
optional per-chunk state-checkpoint output.

Algorithm overview (per chunk c, tokens [cC, (c+1)C); beta and the chunk inverse are folded into the records):
  Inputs : k_decay[BT,DK], q_decay[BT,DK], T[BT,DK], A[BT,BT], diag[DK] = exp2(g[BT-1,:]), V[BT,DV]
  State  : S_prev[DK,DV]  (recurrent state, held in TMEM, fp32 carry; a b16 copy is the state GEMMs' A operand)
  K*state GEMM   : KS[BT,DV]  = k_decay @ S_prev
  Q*state GEMM   : QS[BT,DV]  = q_decay @ S_prev
  Y              : Y = V - KS                                     (b16, the U input slot)
  KV update GEMM : S_upd[DK,DV] = T^T @ Y                         (the T record in the K restore slot; left then right key half)
  QKV GEMM       : O_acc[BT,DV] = QS + A @ Y
  Epilogue:
    O[BT,DV]  = scale * O_acc                          (drained to SMEM, TMA-stored)
    S_next    = diag .* S_prev + S_upd                 (per-channel decay of the state in TMEM, then the update)

SMEM layout (stage counts live in kda_prefill_config.py and build_cfg; sizes at DK = 128, DV = 64, bf16 io):
  Buffer                       Size (B)  Stages
  V (raw)                          2048       8
  diag (fp32)                       512       8
  k_decay / q_decay / T records  3 x 4096       4
  A (intermediate)               1024       4
  O store                          2048       2
  checkpoint staging              16384       2    <-- enable_checkpoints only
  scheduler ticket ring               4       8    <-- next-tile publish ring

TMEM layout (512 columns allocated):
  Buffer                  Cols
  state                   128     <-- DKxDV fp32 (doubles as the final state acc)
  state input              64     <-- b16 state staging (K*state / Q*state A operand)
  Q*state / O acc      2 x 16     <-- BTxDV fp32, 2-stage ring
  K*state acc              16
  U input                   8     <-- b16 packed Y

Warp assignments (16 warps = 512 threads):
  warps 0-3     : CG0 - left state half of every chunk: b16 pack, fp32 decay, checkpoint rows
  warps 4-7     : CG1 - the O drain: O acc -> scaled b16 SMEM
  warps 8-11    : CG2 - state seed, right state half, Y = V - KS, checkpoint rows, final state
  warp  12      : scheduler warp - the scheduler protocol only
  warp  13      : MMA warp       - every tcgen05 GEMM; TMEM lifecycle
  warp  14      : TMA load warp  - V and the five records
  warp  15      : epilogue warp  - O and checkpoint TMA stores
"""

from dataclasses import dataclass
from typing import NamedTuple, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as nvvm
import cutlass.cute as cute

from attn_gym._backends.cute.compat import SmemAllocator

from ..common.split_k import ORDER_CAPACITY, ORDER_ELEMENTS, ORDER_THREADS, WORK_ITEM_FINAL_DST, decode_head, decode_work_item, order_body
from ..common.thd import TENSOR_MAP_QWORDS, emit_checkpoint_seq_descs, emit_seq_descs, emit_tile_seq_descs
from . import kda_prep_f16
from .gdn_tinv_f16 import emit_tinv_rows
from .kda_prefill_config import CFG

from ..tile_dsl.barrier import (
    advance,
    launch_dependent_grids,
    MBarrier,
    PipelineState,
    Producer,
    wait_on_dependent_grids,
)
from ..tile_dsl.handles import MmaDesc, SmemTile, smem_data_ptr, tma_slice_runtime_desc
from ..tile_dsl.mma import desc_opaque, mma_ts_step
from ..tile_dsl.swizzle import swizzle_box_offset_128b, swizzle_xor_128b
from ..tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait, tma_tensormap_acquire
from ..tile_dsl.pointwise import sigmoid, opaque_f32_zero, opaque_i32, fmul2, fp32_to_fp16, sub_f16x2

USE_PDL = True

LOG2_E: float = 1.4426950408889634
DEFAULT_GATE_LOWER_BOUND: float = -5.0
L2_NORM_EPS: float = 1.0e-12


class KdaPrefillBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_raw_ready: MBarrier
    mb_raw_done: MBarrier

    mb_o_acc_ready: MBarrier
    mb_o_acc_done: MBarrier
    mb_state_k_acc_ready: MBarrier

    mb_state_input_cg2_ready: MBarrier
    mb_state_input_cg0_ready: MBarrier
    mb_u_input_ready: MBarrier

    mb_a_done: MBarrier
    mb_k_decay_inv_cg0_ready: MBarrier
    mb_decay_tcgen05_done: MBarrier
    mb_k_restore_done: MBarrier

    mb_state_acc_cg0_done: MBarrier
    mb_state_acc_cg2_done: MBarrier
    mb_tmem_done: MBarrier

    mb_o_tmastg_ready: MBarrier
    mb_o_tmastg_done: MBarrier

    mb_checkpoint_tmastg_ready: MBarrier
    mb_checkpoint_tmastg_done: MBarrier

    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier


def make_bars(cfg) -> KdaPrefillBars:
    """KdaPrefillBars constructor."""

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=8)

    CG0_WARPS = len(cfg.compute_group_0_warp_ids)
    CG2_WARPS = len(cfg.compute_group_2_warp_ids)
    CG1_WARPS = len(cfg.compute_group_1_warp_ids)

    return KdaPrefillBars(
        mb_raw_ready=MBarrier(alloc(cfg.smem_raw_bar_stages), spin=True, stages=cfg.smem_raw_bar_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_raw_done=MBarrier(alloc(cfg.smem_raw_stages), spin=True, stages=cfg.smem_raw_stages, init_count=CG0_WARPS + CG2_WARPS, producer=Producer.THREAD),
        mb_o_acc_ready=MBarrier(alloc(cfg.tmem_q_state_acc_stages), spin=True, stages=cfg.tmem_q_state_acc_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_o_acc_done=MBarrier(
            alloc(cfg.tmem_q_state_acc_stages), spin=True, stages=cfg.tmem_q_state_acc_stages, init_count=CG1_WARPS, producer=Producer.THREAD
        ),
        mb_state_k_acc_ready=MBarrier(alloc(1), spin=True, stages=1, init_count=1, producer=Producer.MMA_COMMIT),
        mb_state_input_cg2_ready=MBarrier(alloc(1), spin=True, stages=1, init_count=CG2_WARPS, producer=Producer.THREAD),
        mb_state_input_cg0_ready=MBarrier(alloc(1), spin=True, stages=1, init_count=CG0_WARPS, producer=Producer.THREAD),
        mb_u_input_ready=MBarrier(alloc(1), spin=True, stages=1, init_count=CG2_WARPS + CG0_WARPS, producer=Producer.THREAD),
        mb_a_done=MBarrier(alloc(cfg.smem_intermediate_stages), spin=True, stages=cfg.smem_intermediate_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_k_decay_inv_cg0_ready=MBarrier(alloc(cfg.smem_decay_stages), spin=True, stages=cfg.smem_decay_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_decay_tcgen05_done=MBarrier(alloc(cfg.smem_decay_stages), spin=True, stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_k_restore_done=MBarrier(alloc(cfg.smem_decay_stages), spin=True, stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_state_acc_cg0_done=MBarrier(alloc(cfg.smem_decay_stages), spin=True, stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_state_acc_cg2_done=MBarrier(alloc(cfg.smem_decay_stages), spin=True, stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_tmem_done=MBarrier(alloc(1), spin=True, stages=1, init_count=CG2_WARPS + CG1_WARPS, producer=Producer.THREAD),
        mb_o_tmastg_ready=MBarrier(alloc(cfg.smem_o_stages), spin=True, stages=cfg.smem_o_stages, init_count=CG1_WARPS, producer=Producer.THREAD),
        mb_o_tmastg_done=MBarrier(alloc(cfg.smem_o_stages), spin=True, stages=cfg.smem_o_stages, init_count=1, producer=Producer.THREAD),
        mb_checkpoint_tmastg_ready=MBarrier(
            alloc(cfg.smem_checkpoint_stages),
            spin=True,
            stages=cfg.smem_checkpoint_stages,
            init_count=CG0_WARPS + CG2_WARPS,
            producer=Producer.THREAD,
        ),
        mb_checkpoint_tmastg_done=MBarrier(
            alloc(cfg.smem_checkpoint_stages), spin=True, stages=cfg.smem_checkpoint_stages, init_count=1, producer=Producer.THREAD
        ),
        mb_scheduler_ready=MBarrier(alloc(cfg.scheduler_stages), spin=True, stages=cfg.scheduler_stages, init_count=1, producer=Producer.THREAD),
        mb_scheduler_done=MBarrier(alloc(cfg.scheduler_stages), spin=True, stages=cfg.scheduler_stages, init_count=15, producer=Producer.THREAD),
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
def epilogue_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    lane_idx,
    mO,
    sO_raw,
    sCheckpoint_raw,
    desc_o_base,
    desc_checkpoint_base,
    checkpoint_every_n_tokens,
    bars,
) -> None:
    """Epilogue warp role (warp 15): persistent scheduler loop computing the
    causal A tile via register MMA and draining the O/checkpoint TMA stores."""
    elect_one = nvvm.elect_sync()
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    if cutlass.const_expr(cfg.enable_checkpoints):
        sCheckpoint_tma = SmemTile(
            base=sCheckpoint_raw,
            elems_per_stage=(cfg.d_k * cfg.d_v),
            stages=cfg.smem_checkpoint_stages,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=0,
            tma_loads_per_tile=(cfg.d_k // 64),
            tma_granu_elems=64,
            tma_subtile_stride_elems=cfg.d_v * 64,
        )
    checkpoint_ready_index = PipelineState.start(phase=0)
    sO_tma = SmemTile(
        base=sO_raw,
        elems_per_stage=(cfg.b_t * cfg.d_v),
        stages=cfg.smem_o_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_v // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=cfg.b_t * 64,
    )

    cum_chunk_base = cutlass.Int32(0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o, v_offset = decode_head(cfg, head_idx)
        o_slot = batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
        desc_o_slot = (desc_o_base + o_slot).tospace(cutlass.AddressSpace.generic)
        if cutlass.const_expr(cfg.enable_checkpoints):
            desc_checkpoint_slot = (desc_checkpoint_base + o_slot).tospace(cutlass.AddressSpace.generic)
            checkpoint_chunks = checkpoint_every_n_tokens // cutlass.Int32(cfg.b_t)
            checkpoint_quotient = (compute_start + cutlass.Int32(1)) // checkpoint_chunks
            checkpoint_remainder = (compute_start + cutlass.Int32(1)) % checkpoint_chunks
            if elect_one:
                tma_tensormap_acquire(desc_checkpoint_slot)
        if elect_one:
            tma_tensormap_acquire(desc_o_slot)
        num_chunks_tile = write_end - compute_start
        if cutlass.const_expr(cfg.enable_checkpoints):
            if num_chunks_tile > 0 and write_start == 0:
                checkpoint_stage = checkpoint_ready_index.idx
                bars.mb_checkpoint_tmastg_ready[checkpoint_stage].wait(checkpoint_ready_index.phase)
                checkpoint_ready_index = advance(checkpoint_ready_index, cfg.smem_checkpoint_stages)
                checkpoint_slice = tma_slice_runtime_desc(desc_checkpoint_slot, cutlass.Int32(0), v_offset, cutlass.Int32(0), head_o)
                tma_store_tile(sCheckpoint_tma[checkpoint_stage], checkpoint_slice)
                tma_store_commit()
                tma_store_wait(0)
                if nvvm.elect_sync():
                    bars.mb_checkpoint_tmastg_done[checkpoint_stage].arrive()
        for local_chunk_idx in cutlass.range(num_chunks_tile, unroll=1):
            chunk_idx = compute_start + local_chunk_idx
            cum_chunk = cum_chunk_base + local_chunk_idx
            chunk_count = cutlass.Uint32(cum_chunk)

            # ---- checkpoint + O store: checkpoint first ------------------------------
            if local_chunk_idx > 0:
                output_chunk = chunk_idx - cutlass.Int32(1)
                output_chunk_start = output_chunk * cfg.b_t
                o_stage = cutlass.Int32((chunk_count - cutlass.Uint32(1)) % cfg.smem_o_stages)
                did_checkpoint = cutlass.Int32(0)
                checkpoint_stage = cutlass.Int32(0)
                if cutlass.const_expr(cfg.enable_checkpoints):
                    # ---- checkpoint store --------------------------------------------
                    do_checkpoint = checkpoint_remainder == 0
                    do_checkpoint = do_checkpoint and chunk_idx >= write_start
                    checkpoint_stage = checkpoint_ready_index.idx
                    if do_checkpoint:
                        bars.mb_checkpoint_tmastg_ready[checkpoint_ready_index.idx].wait(checkpoint_ready_index.phase)
                        checkpoint_ready_index = advance(checkpoint_ready_index, cfg.smem_checkpoint_stages)
                        checkpoint_entry = checkpoint_quotient
                        checkpoint_slice = tma_slice_runtime_desc(desc_checkpoint_slot, cutlass.Int32(0), v_offset, checkpoint_entry, head_o)
                        tma_store_tile(sCheckpoint_tma[checkpoint_stage], checkpoint_slice)
                        tma_store_commit()
                        did_checkpoint = cutlass.Int32(1)
                    checkpoint_remainder = checkpoint_remainder + cutlass.Int32(1)
                    if checkpoint_remainder == checkpoint_chunks:
                        checkpoint_remainder = cutlass.Int32(0)
                        checkpoint_quotient = checkpoint_quotient + cutlass.Int32(1)
                bars.mb_o_tmastg_ready[o_stage].wait(cutlass.Int32(((chunk_count - cutlass.Uint32(1)) // cfg.smem_o_stages) % 2))
                o_slice = tma_slice_runtime_desc(desc_o_slot, v_offset, head_o, output_chunk_start)
                did_o = cutlass.Int32(0)
                if output_chunk >= write_start:
                    tma_store_tile(sO_tma[o_stage], o_slice)
                    tma_store_commit()
                    did_o = cutlass.Int32(1)
                if cutlass.const_expr(cfg.enable_checkpoints):
                    if did_checkpoint == 1 and did_o == 1:
                        tma_store_wait(1)
                        if nvvm.elect_sync():
                            bars.mb_checkpoint_tmastg_done[checkpoint_stage].arrive()
                        tma_store_wait(0)
                        if nvvm.elect_sync():
                            bars.mb_o_tmastg_done[o_stage].arrive()
                    if did_checkpoint == 1 and did_o == 0:
                        tma_store_wait(0)
                        if nvvm.elect_sync():
                            bars.mb_checkpoint_tmastg_done[checkpoint_stage].arrive()
                            bars.mb_o_tmastg_done[o_stage].arrive()
                    if did_checkpoint == 0:
                        if did_o == 1:
                            tma_store_wait(0)
                        if nvvm.elect_sync():
                            bars.mb_o_tmastg_done[o_stage].arrive()
                else:
                    tma_store_wait(0)
                    if nvvm.elect_sync():
                        bars.mb_o_tmastg_done[o_stage].arrive()

        # ---- last computed chunk store -----------------------------------------------
        if num_chunks_tile > 0:
            output_chunk = write_end - cutlass.Int32(1)
            last_cum_chunk = cum_chunk_base + num_chunks_tile - cutlass.Int32(1)
            output_chunk_start = output_chunk * cfg.b_t
            o_stage = last_cum_chunk % cfg.smem_o_stages
            bars.mb_o_tmastg_ready[o_stage].wait((last_cum_chunk // cfg.smem_o_stages) % 2)
            o_slice = tma_slice_runtime_desc(desc_o_slot, v_offset, head_o, output_chunk_start)
            tma_store_tile(sO_tma[o_stage], o_slice)
            tma_store_commit()
            tma_store_wait(0)
            if nvvm.elect_sync():
                bars.mb_o_tmastg_done[o_stage].arrive()
        cum_chunk_base += num_chunks_tile
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)


@cute.jit
def scheduler_warp(
    cfg,
    total_tiles,
    bidx,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    bars,
) -> None:
    """Scheduler warp role (warp 12): the scheduler protocol only."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    elect_one = nvvm.elect_sync()
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)


@cute.jit
def tcgen05_mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    tmem_base_slot,
    sIntermediate,
    sK_decay,
    sK_restore_trans,
    sQ_decay,
    bars,
) -> None:
    """tcgen05-MMA warp role (warp 13): persistent scheduler loop issuing
    every tcgen05 GEMM and owning the TMEM lifecycle."""
    elect_one = nvvm.elect_sync()
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    nvvm.tcgen05_alloc(tmem_base_slot, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    state_input_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_input_offset, cutlass.Int8)
    state_k_acc_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_k_acc_offset, cutlass.Float32)
    u_input_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_u_input_offset, cutlass.Int8)
    state_dst_cg0_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_acc_offset, cutlass.Float32)
    state_update_n = cutlass.const_expr(cfg.d_k // 2 if cfg.d_k // 2 >= 64 else cfg.d_k)
    state_update_split = cutlass.const_expr(state_update_n < cfg.d_k)
    k_restore_right_bytes = cutlass.const_expr(cfg.b_t * 128)
    state_dst_cg2_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_acc_offset + state_update_n, cutlass.Float32)
    state_input_cg2_index = PipelineState.start(phase=0)
    state_input_cg0_index = PipelineState.start(phase=0)
    u_input_index = PipelineState.start(phase=0)
    k_decay_ready = PipelineState.start(phase=0)
    intermediate_ready = PipelineState.start(phase=0)
    o_acc_free = PipelineState.start(phase=1)

    # ---- chunk-invariant GEMM descriptors --------------------------------------------
    bytes_per_element = cfg.io_dtype.width // 8
    instruction_descriptor_acc = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        b_major=0,
    )
    instruction_descriptor_final_state = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=state_update_n,
        m_dim=cfg.d_v,
        b_major=1,
    )
    bmm_state_k_decay_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_acc,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_state_q_decay_desc = bmm_state_k_decay_desc
    bmm_u_a_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_acc,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_u_k_restore_desc = MmaDesc(
        M=cfg.d_v,
        N=state_update_n,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        cta_group=1,
        idesc=instruction_descriptor_final_state,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    STATE_A_SEG = bmm_state_k_decay_desc.sps_B * bmm_state_k_decay_desc.tmem_advance_A
    STATE_B_SEG = bmm_state_k_decay_desc.smem_subtile_B >> 4
    STATE_K_STEPS_CG0 = bmm_state_k_decay_desc.num_k_steps // 2
    cum_chunk_base = cutlass.Int32(0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_chunks_tile = write_end - compute_start
        if cutlass.const_expr(cfg.use_initial_state):
            seed_state = compute_start == 0
        for local_chunk_idx in cutlass.range(num_chunks_tile, unroll=1):
            cum_chunk_base + local_chunk_idx
            if cutlass.const_expr(cfg.use_initial_state):
                have_state = local_chunk_idx > 0 or seed_state
            else:
                have_state = local_chunk_idx > 0
            q_state_acc_stage = o_acc_free.idx
            decay_stage = k_decay_ready.idx
            intermediate_stage = intermediate_ready.idx
            sK_decay_stage = sK_decay[decay_stage]
            sQ_decay_stage = sQ_decay[decay_stage]
            sK_restore_stage = sK_restore_trans[decay_stage]
            sIntermediate_stage = sIntermediate[intermediate_stage]
            desc_k_decay = desc_opaque(sK_decay_stage.desc())
            desc_q_decay = desc_opaque(sQ_decay_stage.desc())
            desc_k_restore = desc_opaque(sK_restore_stage.desc())
            if cutlass.const_expr(state_update_split):
                desc_k_restore_right = desc_k_restore.advance_start_address(k_restore_right_bytes)
            desc_a = desc_opaque(sIntermediate_stage.desc())
            q_state_acc_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_q_state_acc_offset + q_state_acc_stage * cfg.b_t, cutlass.Float32)

            # ---- k state = state(T) @ K decay^T --------------------------------------
            bars.mb_k_decay_inv_cg0_ready[decay_stage].wait(k_decay_ready.phase)
            k_decay_ready = advance(k_decay_ready, cfg.smem_decay_stages)
            if have_state:
                bars.mb_state_input_cg0_ready.wait(state_input_cg0_index.phase)
                state_input_cg0_index = advance(state_input_cg0_index, 1)

                for f in cutlass.range_constexpr(bmm_state_k_decay_desc.num_k_steps):
                    if cutlass.const_expr(f == STATE_K_STEPS_CG0):
                        bars.mb_state_input_cg2_ready.wait(state_input_cg2_index.phase)
                        state_input_cg2_index = advance(state_input_cg2_index, 1)
                    s = f // bmm_state_k_decay_desc.sps_B
                    k = f - s * bmm_state_k_decay_desc.sps_B
                    mma_ts_step(
                        bmm_state_k_decay_desc,
                        state_input_ptr.subview(s * STATE_A_SEG),
                        desc_k_decay + s * STATE_B_SEG,
                        state_k_acc_ptr,
                        k,
                        cutlass.Boolean(f > 0),
                        issue_mma=elect_one,
                    )

                if elect_one:
                    bars.mb_state_k_acc_ready.arrive(cta_group=1)

            # ---- q state = state(T) @ Q decay^T --------------------------------------
            bars.mb_o_acc_done[q_state_acc_stage].wait(o_acc_free.phase)
            o_acc_free = advance(o_acc_free, cfg.tmem_q_state_acc_stages)
            if have_state:
                for s in cutlass.range_constexpr(bmm_state_q_decay_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_state_q_decay_desc.sps_B):
                        mma_ts_step(
                            bmm_state_q_decay_desc,
                            state_input_ptr.subview(s * STATE_A_SEG),
                            desc_q_decay + s * STATE_B_SEG,
                            q_state_acc_ptr,
                            k,
                            cutlass.Boolean(s + k > 0),
                            issue_mma=elect_one,
                        )

            if elect_one:
                bars.mb_decay_tcgen05_done[decay_stage].arrive(cta_group=1)

            # ---- final state += U(T) @ K restore, left then right key half -----------
            bars.mb_u_input_ready.wait(u_input_index.phase)
            u_input_index = advance(u_input_index, 1)
            mma_ts_step(bmm_u_k_restore_desc, u_input_ptr, desc_k_restore, state_dst_cg0_ptr, 0, have_state, issue_mma=elect_one)
            if elect_one:
                bars.mb_state_acc_cg0_done[decay_stage].arrive(cta_group=1)
            if cutlass.const_expr(state_update_split):
                mma_ts_step(bmm_u_k_restore_desc, u_input_ptr, desc_k_restore_right, state_dst_cg2_ptr, 0, have_state, issue_mma=elect_one)
            if elect_one:
                bars.mb_k_restore_done[decay_stage].arrive(cta_group=1)
                bars.mb_state_acc_cg2_done[decay_stage].arrive(cta_group=1)

            # ---- O += U(T) @ A -------------------------------------------------------
            intermediate_ready = advance(intermediate_ready, cfg.smem_intermediate_stages)
            mma_ts_step(bmm_u_a_desc, u_input_ptr, desc_a, q_state_acc_ptr, 0, have_state, issue_mma=elect_one)
            if elect_one:
                bars.mb_o_acc_ready[q_state_acc_stage].arrive(cta_group=1)
                bars.mb_a_done[intermediate_stage].arrive(cta_group=1)

        cum_chunk_base += num_chunks_tile
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)
    bars.mb_tmem_done[0].wait(0)
    nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
    nvvm.tcgen05_dealloc(
        nvvm.make_tmem_ptr(tmem_base, cutlass.Int8),
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
    mScheduler,
    sScheduler,
    lane_idx,
    sV_raw,
    sGate_raw,
    desc_v_base,
    desc_gate_base,
    desc_k_decay_base,
    desc_q_decay_base,
    desc_t_base,
    desc_a_base,
    desc_diag_base,
    sK_decay_raw,
    sQ_decay_raw,
    sK_restore_raw,
    sIntermediate_raw,
    bars,
    v_ratio,
) -> None:
    """TMA-LDG warp role (warp 14): persistent scheduler loop issuing the per-chunk V load and the five records of
    kda_prep_f16 (k_decay -> K decay, q_decay -> Q decay, T -> K restore, A -> the A slot, diag -> the gate stage)."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    raw_index = PipelineState.start(phase=1)
    raw_bar_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=1)

    elect_one = nvvm.elect_sync()
    sV_tma = SmemTile(
        base=sV_raw,
        elems_per_stage=(cfg.d_v * cfg.b_t),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_v // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
    cutlass.const_expr(128 // (cfg.gate_dtype.width // 8))
    sGate_tma = SmemTile(
        base=sGate_raw,
        elems_per_stage=(cfg.gate_cosize // cfg.smem_raw_stages),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(1),
        tma_granu_elems=(cfg.d_k),
        tma_subtile_stride_elems=(cfg.b_t * 32),
    )
    bytes_per_element = cfg.io_dtype.width // 8
    sK_decay_tma = SmemTile(
        base=sK_decay_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_decay_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
    sQ_decay_tma = SmemTile(
        base=sQ_decay_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_decay_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
    sK_restore_tma = SmemTile(
        base=sK_restore_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_decay_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
    sA_tma = SmemTile(
        base=sIntermediate_raw,
        elems_per_stage=(2 * cfg.b_t * cfg.b_t),
        stages=cfg.smem_intermediate_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=1,
        tma_granu_elems=(cfg.b_t * cfg.b_t),
        tma_subtile_stride_elems=0,
    )
    cum_chunk_base = cutlass.Int32(0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o, v_offset = decode_head(cfg, head_idx)
        head_v = head_idx // v_ratio
        slot = batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        (desc_gate_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_decay_slot = (desc_k_decay_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_q_decay_slot = (desc_q_decay_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_t_slot = (desc_t_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_a_slot = (desc_a_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_diag_slot = (desc_diag_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_k_decay_slot)
            tma_tensormap_acquire(desc_q_decay_slot)
            tma_tensormap_acquire(desc_t_slot)
            tma_tensormap_acquire(desc_a_slot)
            tma_tensormap_acquire(desc_diag_slot)
            tma_tensormap_acquire(desc_v_slot)

        for chunk_idx in cutlass.range(compute_start, write_end, 1, unroll=1):
            chunk_start = chunk_idx * cfg.b_t

            # ---- V + diag record: one transaction barrier per stage ------------------
            bars.mb_raw_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_raw_ready[raw_bar_index.idx].arrive(n_bytes=cfg.tma_v_bytes + cfg.tma_gate_bytes)
            raw_ready_ptr = bars.mb_raw_ready[raw_bar_index.idx].smem_ptr
            v_slice = tma_slice_runtime_desc(desc_v_slot, v_offset, head_v, chunk_start)
            tma_load_tile(sV_tma[raw_index.idx], v_slice, raw_ready_ptr)

            chunk_count = cutlass.Uint32(cum_chunk_base + (chunk_idx - compute_start))
            decay_stage = cutlass.Int32(chunk_count % cfg.smem_decay_stages)
            decay_free_parity = cutlass.Int32(((chunk_count // cfg.smem_decay_stages) + 1) % 2)
            intermediate_stage = cutlass.Int32(chunk_count % cfg.smem_intermediate_stages)
            intermediate_free_parity = cutlass.Int32(((chunk_count // cfg.smem_intermediate_stages) + 1) % 2)

            # ---- diag record into the gate stage -------------------------------------
            diag_slice = tma_slice_runtime_desc(desc_diag_slot, cutlass.Int32(0), head_o, chunk_idx)
            tma_load_tile(sGate_tma[raw_index.idx], diag_slice, raw_ready_ptr)

            # ---- k_decay, q_decay, T, A records on the decay stage's one barrier -------------
            bars.mb_decay_tcgen05_done[decay_stage].wait(decay_free_parity)
            bars.mb_k_restore_done[decay_stage].wait(decay_free_parity)
            bars.mb_a_done[intermediate_stage].wait(intermediate_free_parity)
            if elect_one:
                bars.mb_k_decay_inv_cg0_ready[decay_stage].arrive(n_bytes=3 * cfg.tma_k_bytes + cfg.b_t * cfg.b_t * bytes_per_element)
            k_decay_slice = tma_slice_runtime_desc(desc_k_decay_slot, cutlass.Int32(0), cutlass.Int32(0), head_o, chunk_idx)
            tma_load_tile(sK_decay_tma[decay_stage], k_decay_slice, bars.mb_k_decay_inv_cg0_ready[decay_stage].smem_ptr)
            q_decay_slice = tma_slice_runtime_desc(desc_q_decay_slot, cutlass.Int32(0), cutlass.Int32(0), head_o, chunk_idx)
            tma_load_tile(sQ_decay_tma[decay_stage], q_decay_slice, bars.mb_k_decay_inv_cg0_ready[decay_stage].smem_ptr)
            t_slice = tma_slice_runtime_desc(desc_t_slot, cutlass.Int32(0), cutlass.Int32(0), head_o, chunk_idx)
            tma_load_tile(sK_restore_tma[decay_stage], t_slice, bars.mb_k_decay_inv_cg0_ready[decay_stage].smem_ptr)
            a_slice = tma_slice_runtime_desc(desc_a_slot, cutlass.Int32(0), head_o, chunk_idx)
            tma_load_tile(sA_tma[intermediate_stage], a_slice, bars.mb_k_decay_inv_cg0_ready[decay_stage].smem_ptr)

            raw_index = advance(raw_index, cfg.smem_raw_stages)
            raw_bar_index = advance(raw_bar_index, cfg.smem_raw_bar_stages)
        cum_chunk_base += write_end - compute_start
        tile_idx, scheduler_state = scheduler_publish_next(cfg, bars, sScheduler, mScheduler, scheduler_state, num_ctas, elect_one)
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.jit
def gate_scale(cfg, raw_gate: cutlass.Float32) -> cutlass.Float32:
    """Map raw gate to the log2-domain decay increment used by KDA."""

    if cutlass.const_expr(cfg.safe_gate):
        return cfg.gate_scale_log2 * sigmoid(raw_gate)
    if cutlass.const_expr(cfg.log_gate):
        return raw_gate * cutlass.Float32(LOG2_E)
    return cute.math.log2(raw_gate + cutlass.Float32(1e-10), fastmath=True)


@cute.jit
def compute0_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    lane_idx,
    warp_idx,
    sGate_exchange_raw,
    sCheckpoint_raw,
    tmem_base_slot,
    checkpoint_every_n_tokens,
    bars,
) -> None:
    """CG0 warp role (warps 0-3): the left state half of every chunk, b16 pack, fp32 decay and checkpoint rows."""
    nvvm.setmaxregister(
        cfg.num_regs_compute_group_0,
        nvvm.SetMaxRegisterAction.INCREASE if cfg.num_regs_compute_group_0 >= 65536 // cfg.threads_per_cta else nvvm.SetMaxRegisterAction.DECREASE,
    )
    elect_one = nvvm.elect_sync()
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    row_lo_addr = (tmem_base >> 16) << 16
    state_col_id = tmem_col + cfg.tmem_state_acc_offset
    packed_col_id = tmem_col + cfg.tmem_state_input_offset

    scheduler_state = PipelineState.start(phase=0)

    cg0_warp = warp_idx - cfg.compute_group_0_warp_ids[0]
    dk_halves = cutlass.const_expr(cfg.d_k // 64)
    if cutlass.const_expr(cfg.enable_checkpoints):
        checkpoint_chunks = checkpoint_every_n_tokens // cutlass.Int32(cfg.b_t)
        checkpoint_row_base = cutlass.Int32(0)
        sCheckpoint_ptr = smem_data_ptr(sCheckpoint_raw)
        if cutlass.const_expr(cfg.d_v == 128):
            checkpoint_row_dim = cg0_warp * cfg.threads_per_warp + lane_idx
            checkpoint_row_valid = cutlass.Boolean(True)
        else:
            checkpoint_row_dim = cg0_warp * 16 + lane_idx // 4
    if cutlass.const_expr(cfg.d_v != 128):
        pack_col = opaque_i32(2 * (lane_idx % 4))
        opaque_i32(2 * (lane_idx % 4) ^ 4)
    cum_chunk_base = cutlass.Int32(0)
    tile_idx = cutlass.Int32(bidx)
    opaque_f32_zero() + cutlass.Float32(1.0)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o, v_offset = decode_head(cfg, head_idx)
        num_chunks_tile = write_end - compute_start
        if cutlass.const_expr(cfg.enable_checkpoints):
            checkpoint_lo = compute_start + cutlass.Int32(1)
            checkpoint_lo = write_start if write_start > checkpoint_lo else checkpoint_lo
            checkpoint_seed_rows = cutlass.Int32(1) if write_start == 0 else cutlass.Int32(0)
            checkpoint_lo_quotient = (checkpoint_lo - cutlass.Int32(1)) // checkpoint_chunks
        for local_chunk_idx in cutlass.range(num_chunks_tile, unroll=1):
            chunk_idx = compute_start + local_chunk_idx
            cum_chunk = cum_chunk_base + local_chunk_idx
            chunk_count = cutlass.Uint32(cum_chunk)
            cutlass.Int32(chunk_count % cfg.smem_decay_stages)
            raw_stage = cutlass.Int32(chunk_count % cfg.smem_raw_stages)
            raw_bar_stage = cutlass.Int32(chunk_count % cfg.smem_raw_bar_stages)
            raw_bar_parity = cutlass.Int32((chunk_count // cfg.smem_raw_bar_stages) % 2)
            sGate_exchange_ptr = smem_data_ptr(sGate_exchange_raw) + raw_stage * cfg.d_k

            bars.mb_raw_ready[raw_bar_stage].wait(raw_bar_parity)

            # ---- state stage, left key half: pack, publish, fp32 decay ---------------
            if cum_chunk > 0:
                update_count = chunk_count - cutlass.Uint32(1)
                bars.mb_state_acc_cg0_done[cutlass.Int32(update_count % cfg.smem_decay_stages)].wait(cutlass.Int32((update_count // cfg.smem_decay_stages) % 2))
            if local_chunk_idx > 0:
                if cutlass.const_expr(cfg.d_v == 128):
                    l_state_vecs = []
                    for b in cutlass.range_constexpr(dk_halves):
                        l_state_vecs.append(nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + state_col_id + b * 32, cutlass.Float32), num=32))
                    l_packed_blocks = []
                    for b in cutlass.range_constexpr(dk_halves):
                        l_packed = cute.make_rmem_tensor((16,), cutlass.Int32)
                        for packed_col in cutlass.range_constexpr(16):
                            l_packed[packed_col] = fp32_to_fp16(l_state_vecs[b][2 * packed_col], l_state_vecs[b][2 * packed_col + 1], dtype=cfg.io_dtype)
                        nvvm.tcgen05_st("32x32b", nvvm.make_tmem_ptr(row_lo_addr + packed_col_id + b * 16, cutlass.Int8), l_packed.load())
                        l_packed_blocks.append(l_packed)
                else:
                    l_state = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + state_col_id, cutlass.Float32), num=cfg.d_k // 16)
                    l_packed = cute.make_rmem_tensor((cfg.d_k // 8,), cutlass.Int32)
                    for m in cutlass.range_constexpr(cfg.d_k // 16):
                        l_packed[2 * m] = fp32_to_fp16(l_state[4 * m], l_state[4 * m + 1], dtype=cfg.io_dtype)
                        l_packed[2 * m + 1] = fp32_to_fp16(l_state[4 * m + 2], l_state[4 * m + 3], dtype=cfg.io_dtype)
                    nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_lo_addr + packed_col_id, cutlass.Int8), l_packed.load())
                nvvm.tcgen05_wait("store")
                if nvvm.elect_sync():
                    bars.mb_state_input_cg0_ready.arrive()

                # ---- fp32 decay of the left key half: state *= exp2(g last) ----------
                if cutlass.const_expr(cfg.d_v == 128):
                    for b in cutlass.range_constexpr(dk_halves):
                        l_scaled = []
                        for scale_group in cutlass.range_constexpr(8):
                            scale_dim = b * 32 + scale_group * 4
                            scale_segment = scale_dim // 32
                            (
                                scale_segment * (cfg.b_t * 32)
                                + (cfg.b_t - 1) * 32
                                + swizzle_xor_128b(cfg.b_t - 1 ^ scale_segment, scale_dim - scale_segment * 32, elem_bytes=4)
                            )
                            l_scale_frag = (sGate_exchange_ptr + scale_dim).load(count=4, alignment=16)
                            for t in cutlass.range_constexpr(2):
                                l_s0, l_s1 = fmul2(
                                    l_state_vecs[b][scale_group * 4 + 2 * t],
                                    l_state_vecs[b][scale_group * 4 + 2 * t + 1],
                                    l_scale_frag[2 * t],
                                    l_scale_frag[2 * t + 1],
                                )
                                l_scaled += [l_s0, l_s1]
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(row_lo_addr + state_col_id + b * 32, cutlass.Float32),
                            cutlass.Vector.from_elements(tuple(l_scaled), cutlass.Float32),
                        )
                else:
                    l_scaled = []
                    for m in cutlass.range_constexpr(cfg.d_k // 16):
                        scale_dim = 8 * m
                        l_scale_frag = (sGate_exchange_ptr + scale_dim + pack_col).load(count=2, alignment=8)
                        l_s0, l_s1 = fmul2(l_state[4 * m], l_state[4 * m + 1], l_scale_frag[0], l_scale_frag[1])
                        l_s2, l_s3 = fmul2(l_state[4 * m + 2], l_state[4 * m + 3], l_scale_frag[0], l_scale_frag[1])
                        l_scaled += [l_s0, l_s1, l_s2, l_s3]
                    nvvm.tcgen05_st(
                        "16x256b",
                        nvvm.make_tmem_ptr(row_lo_addr + state_col_id, cutlass.Float32),
                        cutlass.Vector.from_elements(tuple(l_scaled), cutlass.Float32),
                    )
                nvvm.tcgen05_wait("store")

                # ---- checkpoint row, left key half: the packed operand words ---------
                if cutlass.const_expr(cfg.enable_checkpoints):
                    checkpoint_row = chunk_idx % checkpoint_chunks == 0
                    checkpoint_row = checkpoint_row and chunk_idx >= write_start
                    if checkpoint_row:
                        checkpoint_row_idx = (
                            checkpoint_row_base + checkpoint_seed_rows + (chunk_idx - cutlass.Int32(1)) // checkpoint_chunks - checkpoint_lo_quotient
                        )
                        checkpoint_row_stage = checkpoint_row_idx % cutlass.Int32(cfg.smem_checkpoint_stages)
                        bars.mb_checkpoint_tmastg_done[checkpoint_row_stage].wait(
                            (checkpoint_row_idx // cutlass.Int32(cfg.smem_checkpoint_stages) + cutlass.Int32(1)) % 2
                        )
                        checkpoint_row_addr = checkpoint_row_stage * (cfg.d_k * cfg.d_v)
                        if cutlass.const_expr(cfg.d_v == 128):
                            if checkpoint_row_valid:
                                for b in cutlass.range_constexpr(dk_halves):
                                    for word_group in cutlass.range_constexpr(4):
                                        dk = b * 32 + word_group * 8
                                        row_addr = checkpoint_row_addr + swizzle_box_offset_128b(checkpoint_row_dim, dk, box_rows=cfg.d_v)
                                        (sCheckpoint_ptr + row_addr).store(
                                            cutlass.Vector.from_elements(
                                                tuple(l_packed_blocks[b][word_group * 4 + t] for t in range(4)), cutlass.Int32
                                            ).bitcast(cfg.io_dtype),
                                            alignment=16,
                                        )
                        else:
                            for m in cutlass.range_constexpr(cfg.d_k // 16):
                                dk = 8 * m
                                for h in cutlass.range_constexpr(2):
                                    row_addr = (
                                        checkpoint_row_addr
                                        + (dk // 64) * (cfg.d_v * 64)
                                        + (checkpoint_row_dim + 8 * h) * 64
                                        + swizzle_xor_128b(checkpoint_row_dim + 8 * h, dk % 64 + 2 * (lane_idx % 4), elem_bytes=2)
                                    )
                                    (sCheckpoint_ptr + row_addr).store(
                                        cutlass.Vector.from_elements((l_packed[2 * m + h],), cutlass.Int32).bitcast(cfg.io_dtype), alignment=4
                                    )
                        nvvm.fence_proxy("async.shared", space="cta")
                        if nvvm.elect_sync():
                            bars.mb_checkpoint_tmastg_ready[checkpoint_row_stage].arrive()
            if nvvm.elect_sync():
                bars.mb_raw_done[raw_stage].arrive()
                bars.mb_u_input_ready.arrive()
        if cutlass.const_expr(cfg.enable_checkpoints):
            if num_chunks_tile > 0:
                checkpoint_row_base = checkpoint_row_base + checkpoint_seed_rows + (write_end - cutlass.Int32(1)) // checkpoint_chunks - checkpoint_lo_quotient
        cum_chunk_base += num_chunks_tile
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)


@cute.jit
def compute1_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    lane_idx,
    warp_idx,
    tmem_base_slot,
    mO,
    sO_raw,
    scale,
    bars,
) -> None:
    """CG1 warp role (warps 4-7): the O drain, every chunk's O accumulator scaled into the b16 O staging tile."""
    nvvm.setmaxregister(
        cfg.num_regs_compute_group_1,
        nvvm.SetMaxRegisterAction.INCREASE if cfg.num_regs_compute_group_1 >= 65536 // cfg.threads_per_cta else nvvm.SetMaxRegisterAction.DECREASE,
    )
    elect_one = nvvm.elect_sync()
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    row_hi_addr = (tmem_row + 16) << 16
    sO_ptr = smem_data_ptr(sO_raw)
    ov_row_coord = (lane_idx // 16) * 8 + (lane_idx & 7)
    ov_col_offset = ((lane_idx // 8) & 1) * 8
    cg1_warp = warp_idx % len(cfg.compute_group_1_warp_ids)
    if cutlass.const_expr(cfg.d_v == 128):
        value_dim_base = cg1_warp * cfg.threads_per_warp
    else:
        value_dim_base = cg1_warp * 16
    q_state_col_base = tmem_col + cfg.tmem_q_state_acc_offset
    ov_swizzle_off_lo = swizzle_box_offset_128b(ov_row_coord, value_dim_base + ov_col_offset, box_rows=cfg.b_t)
    ov_swizzle_off_hi = swizzle_box_offset_128b(ov_row_coord, value_dim_base + 16 + ov_col_offset, box_rows=cfg.b_t)
    cg1_index = PipelineState.start(phase=1)
    q_state_drain_index = PipelineState.start(phase=0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        num_chunks_tile = write_end - compute_start
        for local_chunk_idx in cutlass.range(num_chunks_tile, unroll=1):
            drain_o_stage = cg1_index.idx
            drain_o_parity = cg1_index.phase
            drain_q_state_stage = q_state_drain_index.idx
            drain_q_state_parity = q_state_drain_index.phase
            cg1_index = advance(cg1_index, cfg.smem_o_stages)
            q_state_drain_index = advance(q_state_drain_index, cfg.tmem_q_state_acc_stages)
            projection_col_id = q_state_col_base + drain_q_state_stage * cfg.b_t
            bars.mb_o_tmastg_done[drain_o_stage].wait(drain_o_parity)
            bars.mb_o_acc_ready[drain_q_state_stage].wait(drain_q_state_parity)
            loaded_vec_lo = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + projection_col_id, cutlass.Float32), num=2)
            if cutlass.const_expr(cfg.d_v == 128):
                loaded_vec_hi = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_hi_addr + projection_col_id, cutlass.Float32), num=2)

            # ---- output drain: O acc -> scaled b16 SMEM ------------------------------
            o_pack_lo = cute.make_rmem_tensor((4,), cutlass.Int32)
            o_pack_hi = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                scaled0_0, scaled0_1 = fmul2(loaded_vec_lo[2 * reg_idx], loaded_vec_lo[2 * reg_idx + 1], scale, scale)
                o_pack_lo[reg_idx] = fp32_to_fp16(scaled0_0, scaled0_1, dtype=mO.element_type)
                if cutlass.const_expr(cfg.d_v == 128):
                    scaled1_0, scaled1_1 = fmul2(loaded_vec_hi[2 * reg_idx], loaded_vec_hi[2 * reg_idx + 1], scale, scale)
                    o_pack_hi[reg_idx] = fp32_to_fp16(scaled1_0, scaled1_1, dtype=mO.element_type)
            nvvm.stmatrix(
                sO_ptr + drain_o_stage * (cfg.b_t * cfg.d_v) + ov_swizzle_off_lo,
                o_pack_lo.load(),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            if cutlass.const_expr(cfg.d_v == 128):
                nvvm.stmatrix(
                    sO_ptr + drain_o_stage * (cfg.b_t * cfg.d_v) + ov_swizzle_off_hi,
                    o_pack_hi.load(),
                    nvvm.MMALayout.COL,
                    shape=nvvm.StoreShape.M8N8,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            if nvvm.elect_sync():
                bars.mb_o_acc_done[drain_q_state_stage].arrive()
                bars.mb_o_tmastg_ready[drain_o_stage].arrive()
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    if nvvm.elect_sync():
        bars.mb_tmem_done[0].arrive()


@cute.jit
def compute2_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    lane_idx,
    tmem_base_slot,
    warp_idx,
    mState_out,
    mState_init,
    mSeedIndices,
    mFinalIndices,
    mO,
    sO_raw,
    sV_raw,
    sCheckpoint_raw,
    sGate_exchange_raw,
    checkpoint_every_n_tokens,
    scale,
    bars,
) -> None:
    """CG2 warp role (warps 8-11): the state seed, the right state half, Y = V - KS, checkpoint rows and the final state."""
    nvvm.setmaxregister(
        cfg.num_regs_compute_group_2,
        nvvm.SetMaxRegisterAction.INCREASE if cfg.num_regs_compute_group_2 >= 65536 // cfg.threads_per_cta else nvvm.SetMaxRegisterAction.DECREASE,
    )
    elect_one = nvvm.elect_sync()

    if cutlass.const_expr(cfg.enable_checkpoints):
        checkpoint_chunks = checkpoint_every_n_tokens // cutlass.Int32(cfg.b_t)
        checkpoint_row_count = cutlass.Int32(0)

    smem_data_ptr(sO_raw)
    sCheckpoint_ptr = smem_data_ptr(sCheckpoint_raw) if cutlass.const_expr(cfg.enable_checkpoints) else smem_data_ptr(sO_raw)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    ov_row_coord = (lane_idx // 16) * 8 + (lane_idx & 7)
    ov_col_offset = ((lane_idx // 8) & 1) * 8
    if cutlass.const_expr(cfg.d_v == 128):
        tmem_subpartition = warp_idx % (cfg.d_v // cfg.threads_per_warp)
        value_dim = tmem_subpartition * cfg.threads_per_warp + lane_idx
        value_dim_base = tmem_subpartition * cfg.threads_per_warp
        state_row_valid = cutlass.Boolean(True)
    else:
        cg2_warp = warp_idx % len(cfg.compute_group_2_warp_ids)
        value_dim = cg2_warp * 16 + lane_idx % 16
        value_dim_base = cg2_warp * 16
        state_row_valid = lane_idx < 16
        pack_row = cg2_warp * 16 + lane_idx // 4
    row_lo_addr = tmem_row << 16
    row_hi_addr = (tmem_row + 16) << 16
    state_col_id = tmem_col + cfg.tmem_state_acc_offset
    packed_col_id = tmem_col + cfg.tmem_state_input_offset
    state_k_col_id = tmem_col + cfg.tmem_state_k_acc_offset
    y_dst_col_id = tmem_col + (cfg.tmem_u_input_offset)
    tmem_col + cfg.tmem_q_state_acc_offset
    state_blocks_per_half = cutlass.const_expr(cfg.d_k // 32)
    ov_swizzle_off_lo = swizzle_box_offset_128b(ov_row_coord, value_dim_base + ov_col_offset, box_rows=cfg.b_t)
    ov_swizzle_off_hi = swizzle_box_offset_128b(ov_row_coord, value_dim_base + 16 + ov_col_offset, box_rows=cfg.b_t)
    state_k_acc_index = PipelineState.start(phase=0)
    PipelineState.start(phase=0)
    state_update_index = PipelineState.start(phase=0)
    raw_index = PipelineState.start(phase=0)
    raw_bar_index = PipelineState.start(phase=0)  # even-depth ready/beta-ring slot
    if cutlass.const_expr(cfg.d_v != 128):
        pack_col = opaque_i32(2 * (lane_idx % 4))
        opaque_i32(2 * (lane_idx % 4) ^ 4)
    cum_chunk_base = cutlass.Int32(0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, batch_seqlen, batch_num_chunks, write_start, write_end, compute_start, compute_end = decode_work_item(
            cfg, tile_idx, mWorkItems
        )
        head_o, v_offset = decode_head(cfg, head_idx)
        num_chunks_tile = write_end - compute_start
        if cutlass.const_expr(cfg.enable_checkpoints):
            checkpoint_phase = compute_start % checkpoint_chunks
            checkpoint_stage = checkpoint_row_count % cutlass.Int32(cfg.smem_checkpoint_stages)
            checkpoint_stage_parity = (checkpoint_row_count // cutlass.Int32(cfg.smem_checkpoint_stages) + cutlass.Int32(1)) % 2

        if num_chunks_tile > 0:
            # ---- first chunk: seed state TMEM from mState init -----------------------
            seed_from_initial_state = compute_start == 0
            sV_ptr = smem_data_ptr(sV_raw) + raw_index.idx * (cfg.d_v * cfg.b_t)

            # ---- state seed: initial state GMEM -> packed b16 TMEM + fp32 state TMEM ----
            if cutlass.const_expr(mState_init is not None):
                if seed_from_initial_state:
                    seed_row = batch_idx
                    if cutlass.const_expr(mSeedIndices is not None):
                        seed_row = cutlass.Int32(mSeedIndices[batch_idx])
                    seed_vw = 16 // (mState_init.element_type.width // 8)
                    seed_src = (mState_init.iterator + mState_init.layout((seed_row, head_o, value_dim + v_offset, 0))).raw_ptr()
                    bars.mb_raw_ready[raw_bar_index.idx].wait(raw_bar_index.phase)
                    seed_exchange_ptr = smem_data_ptr(sGate_exchange_raw) + raw_index.idx * cfg.d_k
                    if cutlass.const_expr(cfg.enable_checkpoints):
                        seed_stage_base = checkpoint_stage * (cfg.d_k * cfg.d_v)
                    for init_half in cutlass.range_constexpr(2):
                        init_blocks_lo = init_half * state_blocks_per_half
                        init_blocks_hi = init_blocks_lo + state_blocks_per_half
                        seed_vecs = []
                        for i in cutlass.range_constexpr(init_blocks_lo, init_blocks_hi):
                            seed_block = []
                            for g in cutlass.range_constexpr(16 // seed_vw):
                                seed_chunk = (seed_src + i * 16 + g * seed_vw).load(count=seed_vw, alignment=16)
                                for t in cutlass.range_constexpr(seed_vw):
                                    seed_block.append(seed_chunk[t].to(cutlass.Float32))
                            seed_vecs.append(seed_block)

                        seed_packed_blocks = []
                        for i in cutlass.range_constexpr(init_blocks_lo, init_blocks_hi):
                            seed_pack = cute.make_rmem_tensor((8,), cutlass.Int32)
                            for packed_col in cutlass.range_constexpr(8):
                                seed_pack[packed_col] = fp32_to_fp16(
                                    seed_vecs[i - init_blocks_lo][2 * packed_col], seed_vecs[i - init_blocks_lo][2 * packed_col + 1], dtype=cfg.io_dtype
                                )
                            nvvm.tcgen05_st(
                                "32x32b",
                                nvvm.make_tmem_ptr(row_lo_addr + packed_col_id + i * 8, cutlass.Int8),
                                seed_pack.load(),
                            )
                            seed_packed_blocks.append(seed_pack)
                        nvvm.tcgen05_wait("store")
                        if cutlass.const_expr(init_half == 0):
                            if nvvm.elect_sync():
                                bars.mb_state_input_cg0_ready.arrive()
                        else:
                            if nvvm.elect_sync():
                                bars.mb_state_input_cg2_ready.arrive()

                        # ---- fp32 decay of the seed half: state = seed * exp2(g last) ----
                        for i in cutlass.range_constexpr(init_blocks_lo, init_blocks_hi):
                            seed_scaled = []
                            for scale_group in cutlass.range_constexpr(4):
                                scale_dim = i * 16 + scale_group * 4
                                scale_segment = scale_dim // 32
                                (
                                    scale_segment * (cfg.b_t * 32)
                                    + (cfg.b_t - 1) * 32
                                    + swizzle_xor_128b(cfg.b_t - 1 ^ scale_segment, scale_dim - scale_segment * 32, elem_bytes=4)
                                )
                                seed_scale_frag = (seed_exchange_ptr + scale_dim).load(count=4, alignment=16)
                                for t in cutlass.range_constexpr(2):
                                    seed_s0, seed_s1 = fmul2(
                                        seed_vecs[i - init_blocks_lo][scale_group * 4 + 2 * t],
                                        seed_vecs[i - init_blocks_lo][scale_group * 4 + 2 * t + 1],
                                        seed_scale_frag[2 * t],
                                        seed_scale_frag[2 * t + 1],
                                    )
                                    seed_scaled += [seed_s0, seed_s1]
                            nvvm.tcgen05_st(
                                "32x32b",
                                nvvm.make_tmem_ptr(row_lo_addr + state_col_id + i * 16, cutlass.Float32),
                                cutlass.Vector.from_elements(tuple(seed_scaled), cutlass.Float32),
                            )

                        # ---- seed checkpoint row half: the packed operand words ------
                        if cutlass.const_expr(cfg.enable_checkpoints):
                            if write_start == 0:
                                if cutlass.const_expr(init_half == 0):
                                    bars.mb_checkpoint_tmastg_done[checkpoint_stage].wait(checkpoint_stage_parity)
                                if state_row_valid:
                                    for i in cutlass.range_constexpr(init_blocks_lo, init_blocks_hi):
                                        for word_group in cutlass.range_constexpr(2):
                                            dk = i * 16 + word_group * 8
                                            seed_row_addr = seed_stage_base + swizzle_box_offset_128b(value_dim, dk, box_rows=cfg.d_v)
                                            (sCheckpoint_ptr + seed_row_addr).store(
                                                cutlass.Vector.from_elements(
                                                    tuple(seed_packed_blocks[i - init_blocks_lo][word_group * 4 + t] for t in range(4)), cutlass.Int32
                                                ).bitcast(cfg.io_dtype),
                                                alignment=16,
                                            )
                    nvvm.tcgen05_wait("store")
                    if cutlass.const_expr(cfg.enable_checkpoints):
                        if write_start == 0:
                            nvvm.fence_proxy("async.shared", space="cta")
                            if nvvm.elect_sync():
                                bars.mb_checkpoint_tmastg_ready[checkpoint_stage].arrive()
                                bars.mb_checkpoint_tmastg_ready[checkpoint_stage].arrive()
                            checkpoint_row_count = checkpoint_row_count + cutlass.Int32(1)

            if cutlass.const_expr(cfg.enable_checkpoints and mState_init is None):
                if write_start == 0:
                    bars.mb_checkpoint_tmastg_done[checkpoint_stage].wait(checkpoint_stage_parity)
                    checkpoint_stage_base = checkpoint_stage * (cfg.d_k * cfg.d_v)
                    zero_packs = tuple(cutlass.Int32(0) for _ in range(4))
                    if state_row_valid:
                        for i in cutlass.range_constexpr(cfg.d_k // 16):
                            for g in cutlass.range_constexpr(2):
                                dk = i * 16 + g * 8
                                checkpoint_addr = checkpoint_stage_base + swizzle_box_offset_128b(value_dim, dk, box_rows=cfg.d_v)
                                (sCheckpoint_ptr + checkpoint_addr).store(
                                    cutlass.Vector.from_elements(zero_packs, cutlass.Int32).bitcast(cfg.io_dtype), alignment=16
                                )
                    nvvm.fence_proxy("async.shared", space="cta")
                    if nvvm.elect_sync():
                        bars.mb_checkpoint_tmastg_ready[checkpoint_stage].arrive()
                        bars.mb_checkpoint_tmastg_ready[checkpoint_stage].arrive()
                    checkpoint_row_count = checkpoint_row_count + cutlass.Int32(1)

            # ---- Y stage: Y = Beta * (V - k state) -----------------------------------
            bars.mb_raw_ready[raw_bar_index.idx].wait(raw_bar_index.phase)
            raw_v_frag_lo = nvvm.ldmatrix(
                sV_ptr + ov_swizzle_off_lo,
                4,
                nvvm.MMALayout.COL,
            )
            if cutlass.const_expr(cfg.d_v == 128):
                raw_v_frag_hi = nvvm.ldmatrix(
                    sV_ptr + ov_swizzle_off_hi,
                    4,
                    nvvm.MMALayout.COL,
                )
            cute.make_rmem_tensor((4,), cutlass.Int32)

            y_lo = [cutlass.Int32(0) for _ in range(4)]
            y_hi = [cutlass.Int32(0) for _ in range(4)]
            if cutlass.const_expr(mState_init is not None):
                if seed_from_initial_state:
                    bars.mb_state_k_acc_ready.wait(state_k_acc_index.phase)
                    state_k_acc_index = advance(state_k_acc_index, 1)
                    state_k_vec_lo = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + state_k_col_id, cutlass.Float32), num=2)
                    for reg_idx in cutlass.range_constexpr(4):
                        frag_pair = reg_idx * 2
                        state_k_lo = fp32_to_fp16(state_k_vec_lo[frag_pair], state_k_vec_lo[frag_pair + 1], dtype=cfg.io_dtype)
                        diff_lo = sub_f16x2(raw_v_frag_lo[reg_idx], state_k_lo, cfg.io_dtype)
                        y_lo[reg_idx] = diff_lo
                    if cutlass.const_expr(cfg.d_v == 128):
                        state_k_vec_hi = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_hi_addr + state_k_col_id, cutlass.Float32), num=2)
                        for reg_idx in cutlass.range_constexpr(4):
                            frag_pair = reg_idx * 2
                            state_k_hi = fp32_to_fp16(state_k_vec_hi[frag_pair], state_k_vec_hi[frag_pair + 1], dtype=cfg.io_dtype)
                            diff_hi = sub_f16x2(raw_v_frag_hi[reg_idx], state_k_hi, cfg.io_dtype)
                            y_hi[reg_idx] = diff_hi
                else:
                    for reg_idx in cutlass.range_constexpr(4):
                        y_lo[reg_idx] = raw_v_frag_lo[reg_idx]
                        if cutlass.const_expr(cfg.d_v == 128):
                            y_hi[reg_idx] = raw_v_frag_hi[reg_idx]
            else:
                for reg_idx in cutlass.range_constexpr(4):
                    y_lo[reg_idx] = raw_v_frag_lo[reg_idx]
                    if cutlass.const_expr(cfg.d_v == 128):
                        y_hi[reg_idx] = raw_v_frag_hi[reg_idx]

            y_input_pack_lo = cute.make_rmem_tensor((4,), cutlass.Int32)
            y_input_pack_hi = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                y_input_pack_lo[reg_idx] = y_lo[reg_idx]
                y_input_pack_hi[reg_idx] = y_hi[reg_idx]
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_lo_addr + y_dst_col_id, cutlass.Int8), y_input_pack_lo.load())
            if cutlass.const_expr(cfg.d_v == 128):
                nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_hi_addr + y_dst_col_id, cutlass.Int8), y_input_pack_hi.load())
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_raw_done[raw_index.idx].arrive()
                bars.mb_u_input_ready.arrive()

            if cutlass.const_expr(cfg.enable_checkpoints):
                checkpoint_phase = checkpoint_phase + cutlass.Int32(1)
                if checkpoint_phase == checkpoint_chunks:
                    checkpoint_phase = cutlass.Int32(0)
            raw_index = advance(raw_index, cfg.smem_raw_stages)
            raw_bar_index = advance(raw_bar_index, cfg.smem_raw_bar_stages)

        for local_chunk_idx in cutlass.range(1, num_chunks_tile, 1, unroll=1):
            chunk_idx = compute_start + local_chunk_idx
            cum_chunk_base + local_chunk_idx
            raw_stage = raw_index.idx
            raw_bar_stage = raw_bar_index.idx
            sV_ptr = smem_data_ptr(sV_raw) + raw_stage * (cfg.d_v * cfg.b_t)
            raw_bar_phase = raw_bar_index.phase
            bars.mb_raw_ready[raw_bar_stage].wait(raw_bar_phase)
            raw_index = advance(raw_index, cfg.smem_raw_stages)
            raw_bar_index = advance(raw_bar_index, cfg.smem_raw_bar_stages)

            # ---- state stage, right key half: pack, publish, fp32 decay --------------
            bars.mb_state_acc_cg2_done[state_update_index.idx].wait(state_update_index.phase)
            state_update_index = advance(state_update_index, cfg.smem_decay_stages)
            if cutlass.const_expr(cfg.d_v == 128):
                state_vecs = []
                for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                    state_vecs.append(nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_lo_addr + state_col_id + i * 16, cutlass.Float32), num=16))
            else:
                state_right = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + state_col_id + cfg.d_k // 2, cutlass.Float32), num=cfg.d_k // 16)

            if cutlass.const_expr(cfg.d_v == 128):
                packed_blocks = []
                for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                    packed_state = cute.make_rmem_tensor((8,), cutlass.Int32)
                    for packed_col in cutlass.range_constexpr(8):
                        packed_state[packed_col] = fp32_to_fp16(
                            state_vecs[i - state_blocks_per_half][2 * packed_col], state_vecs[i - state_blocks_per_half][2 * packed_col + 1], dtype=cfg.io_dtype
                        )
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + packed_col_id + i * 8, cutlass.Int8),
                        packed_state.load(),
                    )
                    packed_blocks.append(packed_state)
            else:
                packed_right = cute.make_rmem_tensor((cfg.d_k // 8,), cutlass.Int32)
                for m in cutlass.range_constexpr(cfg.d_k // 16):
                    packed_right[2 * m] = fp32_to_fp16(state_right[4 * m], state_right[4 * m + 1], dtype=cfg.io_dtype)
                    packed_right[2 * m + 1] = fp32_to_fp16(state_right[4 * m + 2], state_right[4 * m + 3], dtype=cfg.io_dtype)
                nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_lo_addr + packed_col_id + cfg.d_k // 4, cutlass.Int8), packed_right.load())

            # ---- fp32 decay of the right key half: state *= exp2(g last) -------------
            sGate_exchange_ptr = smem_data_ptr(sGate_exchange_raw) + raw_stage * cfg.d_k
            if cutlass.const_expr(cfg.d_v == 128):
                scaled_blocks = []
                for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                    scaled = []
                    for scale_group in cutlass.range_constexpr(4):
                        scale_dim = i * 16 + scale_group * 4
                        scale_segment = scale_dim // 32
                        (
                            scale_segment * (cfg.b_t * 32)
                            + (cfg.b_t - 1) * 32
                            + swizzle_xor_128b(cfg.b_t - 1 ^ scale_segment, scale_dim - scale_segment * 32, elem_bytes=4)
                        )
                        scale_frag = (sGate_exchange_ptr + scale_dim).load(count=4, alignment=16)
                        for t in cutlass.range_constexpr(2):
                            s0, s1 = fmul2(
                                state_vecs[i - state_blocks_per_half][scale_group * 4 + 2 * t],
                                state_vecs[i - state_blocks_per_half][scale_group * 4 + 2 * t + 1],
                                scale_frag[2 * t],
                                scale_frag[2 * t + 1],
                            )
                            scaled += [s0, s1]
                    scaled_blocks.append(scaled)
            else:
                scaled_right = []
                for m in cutlass.range_constexpr(cfg.d_k // 16):
                    scale_dim = cfg.d_k // 2 + 8 * m
                    scale_frag = (sGate_exchange_ptr + scale_dim + pack_col).load(count=2, alignment=8)
                    s0, s1 = fmul2(state_right[4 * m], state_right[4 * m + 1], scale_frag[0], scale_frag[1])
                    s2, s3 = fmul2(state_right[4 * m + 2], state_right[4 * m + 3], scale_frag[0], scale_frag[1])
                    scaled_right += [s0, s1, s2, s3]
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_state_input_cg2_ready.arrive()
            if cutlass.const_expr(cfg.d_v == 128):
                for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + state_col_id + i * 16, cutlass.Float32),
                        cutlass.Vector.from_elements(tuple(scaled_blocks[i - state_blocks_per_half]), cutlass.Float32),
                    )
            else:
                nvvm.tcgen05_st(
                    "16x256b",
                    nvvm.make_tmem_ptr(row_lo_addr + state_col_id + cfg.d_k // 2, cutlass.Float32),
                    cutlass.Vector.from_elements(tuple(scaled_right), cutlass.Float32),
                )

            # ---- checkpoint row, right key half: the packed operand words ------------
            if cutlass.const_expr(cfg.enable_checkpoints):
                checkpoint_row = checkpoint_phase == 0
                checkpoint_row = checkpoint_row and chunk_idx >= write_start
                if checkpoint_row:
                    checkpoint_row_idx = checkpoint_row_count
                    checkpoint_row_stage = checkpoint_row_idx % cutlass.Int32(cfg.smem_checkpoint_stages)
                    bars.mb_checkpoint_tmastg_done[checkpoint_row_stage].wait(
                        (checkpoint_row_idx // cutlass.Int32(cfg.smem_checkpoint_stages) + cutlass.Int32(1)) % 2
                    )
                    checkpoint_row_addr = checkpoint_row_stage * (cfg.d_k * cfg.d_v)
                    if cutlass.const_expr(cfg.d_v == 128):
                        if state_row_valid:
                            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                                for word_group in cutlass.range_constexpr(2):
                                    dk = i * 16 + word_group * 8
                                    row_addr = checkpoint_row_addr + swizzle_box_offset_128b(value_dim, dk, box_rows=cfg.d_v)
                                    (sCheckpoint_ptr + row_addr).store(
                                        cutlass.Vector.from_elements(
                                            tuple(packed_blocks[i - state_blocks_per_half][word_group * 4 + t] for t in range(4)), cutlass.Int32
                                        ).bitcast(cfg.io_dtype),
                                        alignment=16,
                                    )
                    else:
                        for m in cutlass.range_constexpr(cfg.d_k // 16):
                            dk = cfg.d_k // 2 + 8 * m
                            for h in cutlass.range_constexpr(2):
                                row_addr = (
                                    checkpoint_row_addr
                                    + (dk // 64) * (cfg.d_v * 64)
                                    + (pack_row + 8 * h) * 64
                                    + swizzle_xor_128b(pack_row + 8 * h, dk % 64 + 2 * (lane_idx % 4), elem_bytes=2)
                                )
                                (sCheckpoint_ptr + row_addr).store(
                                    cutlass.Vector.from_elements((packed_right[2 * m + h],), cutlass.Int32).bitcast(cfg.io_dtype), alignment=4
                                )
                    nvvm.fence_proxy("async.shared", space="cta")
                    if nvvm.elect_sync():
                        bars.mb_checkpoint_tmastg_ready[checkpoint_row_stage].arrive()
                    checkpoint_row_count = checkpoint_row_count + cutlass.Int32(1)
                checkpoint_phase = checkpoint_phase + cutlass.Int32(1)
                if checkpoint_phase == checkpoint_chunks:
                    checkpoint_phase = cutlass.Int32(0)

            # ---- Y stage: Y = Beta * (V - k state) -----------------------------------
            raw_v_frag_lo = nvvm.ldmatrix(
                sV_ptr + ov_swizzle_off_lo,
                4,
                nvvm.MMALayout.COL,
            )
            if cutlass.const_expr(cfg.d_v == 128):
                raw_v_frag_hi = nvvm.ldmatrix(
                    sV_ptr + ov_swizzle_off_hi,
                    4,
                    nvvm.MMALayout.COL,
                )
            bars.mb_state_k_acc_ready.wait(state_k_acc_index.phase)

            state_k_vec_lo = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_lo_addr + state_k_col_id, cutlass.Float32), num=2)
            if cutlass.const_expr(cfg.d_v == 128):
                state_k_vec_hi = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_hi_addr + state_k_col_id, cutlass.Float32), num=2)

            cute.make_rmem_tensor((4,), cutlass.Int32)
            y_input_pack_lo = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                frag_pair = reg_idx * 2
                state_k_val0, state_k_val1 = state_k_vec_lo[frag_pair], state_k_vec_lo[frag_pair + 1]
                state_k_pair = fp32_to_fp16(state_k_val0, state_k_val1, dtype=cfg.io_dtype)
                diff_pair = sub_f16x2(raw_v_frag_lo[reg_idx], state_k_pair, cfg.io_dtype)
                y_input_pack_lo[reg_idx] = diff_pair

            y_input_pack_hi = cute.make_rmem_tensor((4,), cutlass.Int32)
            if cutlass.const_expr(cfg.d_v == 128):
                for reg_idx in cutlass.range_constexpr(4):
                    frag_pair = reg_idx * 2
                    state_k_val0, state_k_val1 = state_k_vec_hi[frag_pair], state_k_vec_hi[frag_pair + 1]
                    state_k_pair = fp32_to_fp16(state_k_val0, state_k_val1, dtype=cfg.io_dtype)
                    diff_pair = sub_f16x2(raw_v_frag_hi[reg_idx], state_k_pair, cfg.io_dtype)
                    y_input_pack_hi[reg_idx] = diff_pair

            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_lo_addr + y_dst_col_id, cutlass.Int8), y_input_pack_lo.load())
            if cutlass.const_expr(cfg.d_v == 128):
                nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_hi_addr + y_dst_col_id, cutlass.Int8), y_input_pack_hi.load())
            nvvm.tcgen05_wait("store")
            state_k_acc_index = advance(state_k_acc_index, 1)
            if nvvm.elect_sync():
                bars.mb_u_input_ready.arrive()
                bars.mb_raw_done[raw_stage].arrive()

        if num_chunks_tile > 0:
            bars.mb_state_acc_cg2_done[state_update_index.idx].wait(state_update_index.phase)
            state_update_index = advance(state_update_index, cfg.smem_decay_stages)

        final_dst = mWorkItems[tile_idx, WORK_ITEM_FINAL_DST]

        # ---- final state store: TMEM -> GMEM -----------------------------------------
        if cutlass.const_expr(mState_out is not None):
            if batch_seqlen > 0:
                if final_dst >= 0:
                    final_row = final_dst
                    if cutlass.const_expr(mFinalIndices is not None):
                        final_row = cutlass.Int32(mFinalIndices[final_dst])
                    state_vw = 16 // (mState_out.element_type.width // 8)
                    state_dst = (mState_out.iterator + mState_out.layout((final_row, head_o, value_dim + v_offset, 0))).raw_ptr()
                    for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                        loaded = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(row_lo_addr + (tmem_col + cfg.tmem_state_acc_offset + key_block_start), cutlass.Float32),
                            num=32,
                        )

                        if state_row_valid:
                            for g in cutlass.range_constexpr(32 // state_vw):
                                (state_dst + key_block_start + g * state_vw).store(
                                    cutlass.Vector.from_elements(
                                        tuple(loaded[g * state_vw + t].to(mState_out.element_type) for t in range(state_vw)),
                                        mState_out.element_type,
                                    ),
                                    alignment=16,
                                )
            else:
                if state_row_valid and final_dst >= 0:
                    final_row = final_dst
                    if cutlass.const_expr(mFinalIndices is not None):
                        final_row = cutlass.Int32(mFinalIndices[final_dst])
                    seed_row = batch_idx
                    if cutlass.const_expr(mSeedIndices is not None):
                        seed_row = cutlass.Int32(mSeedIndices[batch_idx])
                    for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                        for col in cutlass.range_constexpr(32):
                            key_dim = key_block_start + col
                            if cutlass.const_expr(mState_init is not None):
                                mState_out[final_row, head_o, value_dim + v_offset, key_dim] = mState_init[seed_row, head_o, value_dim + v_offset, key_dim].to(
                                    mState_out.element_type
                                )
                            else:
                                mState_out[final_row, head_o, value_dim + v_offset, key_dim] = cutlass.Float32(0.0).to(mState_out.element_type)
        cum_chunk_base += num_chunks_tile
        tile_idx, scheduler_state = scheduler_next_tile(cfg, bars, sScheduler, scheduler_state, elect_one)

    if nvvm.elect_sync():
        bars.mb_tmem_done[0].arrive()


@cute.jit
def build_descs_body(
    widx,
    base_q,
    base_k,
    base_v,
    base_gate,
    base_o,
    base_checkpoint,
    base_k_decay,
    base_q_decay,
    base_t,
    base_a,
    base_diag,
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    o: cute.Tensor,
    state_checkpoints: cute.Tensor | None,
    prep_k_decay: cute.Tensor | None,
    prep_q_decay: cute.Tensor | None,
    prep_t: cute.Tensor | None,
    prep_a: cute.Tensor | None,
    prep_diag: cute.Tensor | None,
    n_batch: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    b_t: cutlass.Constexpr[int],
) -> None:
    """Per-batch descriptor-array build, one warp per array. Runs inside the
    prologue kernel after its order pass; warps past the array count fall
    through the widx guards."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_q_arr = cute.make_tensor(desc_workspace.iterator, cute.make_layout((arr_words,), stride=(1,)))
    desc_k_arr = cute.make_tensor(desc_workspace.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_v_arr = cute.make_tensor(desc_workspace.iterator + 2 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_gate_arr = cute.make_tensor(desc_workspace.iterator + 3 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_o_arr = cute.make_tensor(desc_workspace.iterator + 4 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_checkpoint_arr = cute.make_tensor(desc_workspace.iterator + 5 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_k_decay_arr = cute.make_tensor(desc_workspace.iterator + 6 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_q_decay_arr = cute.make_tensor(desc_workspace.iterator + 7 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_t_arr = cute.make_tensor(desc_workspace.iterator + 8 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_a_arr = cute.make_tensor(desc_workspace.iterator + 9 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_diag_arr = cute.make_tensor(desc_workspace.iterator + 10 * arr_words, cute.make_layout((arr_words,), stride=(1,)))

    if widx == 0:
        if nvvm.elect_sync():
            emit_seq_descs(base_q, desc_q_arr, cu_seqlens, q, n_batch, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 1:
        if nvvm.elect_sync():
            emit_seq_descs(base_k, desc_k_arr, cu_seqlens, k, n_batch, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 2:
        if nvvm.elect_sync():
            emit_seq_descs(base_v, desc_v_arr, cu_seqlens, v, n_batch, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 3:
        if nvvm.elect_sync():
            emit_seq_descs(base_gate, desc_gate_arr, cu_seqlens, gate, n_batch, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 4:
        if nvvm.elect_sync():
            emit_seq_descs(base_o, desc_o_arr, cu_seqlens, o, n_batch, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if cutlass.const_expr(state_checkpoints is not None):
        if widx == 5:
            if nvvm.elect_sync():
                emit_checkpoint_seq_descs(base_checkpoint, desc_checkpoint_arr, cu_seqlens, state_checkpoints, n_batch, checkpoint_every_n, 2)
                nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 0:
        emit_tile_seq_descs(base_k_decay, desc_k_decay_arr, cu_seqlens, prep_k_decay, n_batch, b_t, 3, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 1:
        emit_tile_seq_descs(base_q_decay, desc_q_decay_arr, cu_seqlens, prep_q_decay, n_batch, b_t, 3, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 2:
        emit_tile_seq_descs(base_t, desc_t_arr, cu_seqlens, prep_t, n_batch, b_t, 3, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 3:
        emit_tile_seq_descs(base_a, desc_a_arr, cu_seqlens, prep_a, n_batch, b_t, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 4:
        emit_tile_seq_descs(base_diag, desc_diag_arr, cu_seqlens, prep_diag, n_batch, b_t, 2, lanes=32)
        nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


@cute.kernel
def frost_kda_prep_prefill_prologue(
    order_gen: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    tiles_per_head: cutlass.Constexpr[int],
    base_q: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_k: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_v: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_gate: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_o: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_checkpoint: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_k_decay: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_q_decay: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_t: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_a: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_diag: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    o: cute.Tensor,
    state_checkpoints: cute.Tensor | None,
    prep_k_decay: cute.Tensor | None,
    prep_q_decay: cute.Tensor | None,
    prep_t: cute.Tensor | None,
    prep_a: cute.Tensor | None,
    prep_diag: cute.Tensor | None,
    mStaging: cute.Tensor | None,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mScheduler: cute.Tensor,
    n_batch: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    prep_base_q: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    prep_base_k: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    prep_base_gate: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    prep_base_k_decay: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    prep_base_q_decay: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    prep_base_t: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    prep_desc_words: cute.Tensor,
    prep_rows: cute.Tensor,
    prep_count: cute.Tensor,
) -> None:
    """Three-CTA prologue (block 2 emits kda_prep's descriptor arrays and its row table, so the prep launches without a
    prologue of its own): block 0 LPT-orders the work-item table and zeroes the
    scheduler rings via :func:`order_body`, block 1 builds the per-batch
    TMA-descriptor arrays via :func:`build_descs_body`, one warp per array."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
        launch_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    lane_idx = tidx % cutlass.Int32(32)
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    sKey = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
    sIdx = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
    sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
    n_heads_out = cutlass.Int32(gate.shape[1])
    if cutlass.const_expr(tiles_per_head > 1):
        n_heads_out = n_heads_out * cutlass.Int32(tiles_per_head)
    if bidx == cutlass.Int32(0):
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
            num_ctas=num_ctas,
        )
    elif bidx == cutlass.Int32(1):
        build_descs_body(
            widx,
            base_q,
            base_k,
            base_v,
            base_gate,
            base_o,
            base_checkpoint,
            base_k_decay,
            base_q_decay,
            base_t,
            base_a,
            base_diag,
            desc_workspace,
            cu_seqlens,
            q,
            k,
            v,
            gate,
            o,
            state_checkpoints,
            prep_k_decay,
            prep_q_decay,
            prep_t,
            prep_a,
            prep_diag,
            n_batch,
            checkpoint_every_n,
            b_t,
        )
    else:
        kda_prep_f16.build_descs_body(
            widx,
            prep_base_q,
            prep_base_k,
            prep_base_gate,
            prep_base_k_decay,
            prep_base_q_decay,
            prep_base_t,
            prep_desc_words,
            cu_seqlens,
            q,
            k,
            gate,
            prep_k_decay,
            prep_q_decay,
            prep_t,
            n_batch,
            b_t,
        )
        if widx == cutlass.Int32(0):
            emit_tinv_rows(b_t, 1, cu_seqlens, prep_rows, prep_count, lane_idx)


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    num_ctas: cutlass.Constexpr[int],
    order_gen: cutlass.Constexpr[bool],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    o: cute.Tensor,
    state_checkpoints: cute.Tensor | None,
    cu_seqlens: cute.Tensor,
    work_item_staging: cute.Tensor | None,
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler_counter: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    checkpoint_every_n: cutlass.Int32,
    stream: cuda_driver.CUstream,
    tiles_per_head: cutlass.Constexpr[int] = 1,
    prep_k_decay: cute.Tensor | None = None,
    prep_q_decay: cute.Tensor | None = None,
    prep_t: cute.Tensor | None = None,
    prep_a: cute.Tensor | None = None,
    prep_diag: cute.Tensor | None = None,
    prep_cfg: cutlass.Constexpr = None,
    prep_desc_words: cute.Tensor | None = None,
    prep_rows: cute.Tensor | None = None,
    prep_row_count: cute.Tensor | None = None,
):
    """One-launch prologue: LPT-order the work items and build the 6
    per-(batch, head) TMA-descriptor arrays (q, k, v, gate, o,
    state_checkpoints) into ``tensormap_workspace``, one block each."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    h_v = v.shape[1]
    n_heads_out = gate.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    d_k = q.shape[2]
    d_v = v.shape[2]
    bytes_per_element = io_dtype.width // 8
    box_elems = 128 // bytes_per_element
    seqlen = q.shape[0]

    q_headed = cute.make_tensor(q.iterator, cute.make_layout((d_k, h_q, seqlen), stride=(1, q.stride[1], q.stride[0])))
    k_headed = cute.make_tensor(k.iterator, cute.make_layout((d_k, h_k, seqlen), stride=(1, k.stride[1], k.stride[0])))
    v_headed = cute.make_tensor(v.iterator, cute.make_layout((d_v, h_v, seqlen), stride=(1, v.stride[1], v.stride[0])))
    gate_headed = cute.make_tensor(gate.iterator, cute.make_layout((d_k, n_heads_out, seqlen), stride=(1, gate.stride[1], gate.stride[0])))
    o_headed = cute.make_tensor(o.iterator, cute.make_layout((d_v, n_heads_out, seqlen), stride=(1, o.stride[1], o.stride[0])))

    swizzle = cuda.TensorMapSwizzle.s128b
    base_q = cuda.create_tensor_map_tiled_from_view(q_headed, box_dims=(box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle)
    base_k = cuda.create_tensor_map_tiled_from_view(k_headed, box_dims=(box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle)
    base_v = cuda.create_tensor_map_tiled_from_view(v_headed, box_dims=(box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle)
    gate_box_elems = 128 // (gate.element_type.width // 8)
    base_gate = cuda.create_tensor_map_tiled_from_view(gate_headed, box_dims=(gate_box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle)
    base_o = cuda.create_tensor_map_tiled_from_view(o_headed, box_dims=(box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle)

    base_checkpoint = base_o
    if cutlass.const_expr(state_checkpoints is not None):
        d_v_state = state_checkpoints.shape[2]
        d_k_state = state_checkpoints.shape[3]
        checkpoint_view = cute.make_tensor(
            state_checkpoints.iterator,
            cute.make_layout(
                (d_k_state, d_v_state, state_checkpoints.shape[0], n_heads_out),
                stride=(state_checkpoints.stride[3], state_checkpoints.stride[2], state_checkpoints.stride[0], state_checkpoints.stride[1]),
            ),
        )
        base_checkpoint = cuda.create_tensor_map_tiled_from_view(
            checkpoint_view, box_dims=(box_elems, d_v_state // tiles_per_head, 1, 1), stride_order=(0, 1, 2, 3), swizzle=swizzle
        )
    base_k_decay = base_o
    base_q_decay = base_o
    base_t = base_o
    base_a = base_o
    base_diag = base_o
    rows = prep_k_decay.shape[0]
    record_maps = []
    for rec in (prep_k_decay, prep_q_decay, prep_t):
        rec_view = cute.make_tensor(rec.iterator, cute.make_layout((d_k, b_t, n_heads_out, rows), stride=(1, rec.stride[2], rec.stride[1], rec.stride[0])))
        record_maps.append(cuda.create_tensor_map_tiled_from_view(rec_view, box_dims=(box_elems, b_t, 1, 1), stride_order=(0, 1, 2, 3), swizzle=swizzle))
    base_k_decay, base_q_decay, base_t = record_maps
    a_view = cute.make_tensor(prep_a.iterator, cute.make_layout((b_t * b_t // 2, n_heads_out, rows), stride=(1, prep_a.stride[1], prep_a.stride[0])))
    base_a = cuda.create_tensor_map_tiled_from_view(a_view, box_dims=(b_t * b_t // 2, 1, 1), stride_order=(0, 1, 2), swizzle=cuda.TensorMapSwizzle.none)
    diag_view = cute.make_tensor(prep_diag.iterator, cute.make_layout((d_k, n_heads_out, rows), stride=(1, prep_diag.stride[1], prep_diag.stride[0])))
    base_diag = cuda.create_tensor_map_tiled_from_view(diag_view, box_dims=(d_k, 1, 1), stride_order=(0, 1, 2), swizzle=cuda.TensorMapSwizzle.none)
    prep_base_q, prep_base_k, prep_base_gate, prep_record_maps = kda_prep_f16.prep_base_maps(prep_cfg, q, k, gate, prep_k_decay, prep_q_decay, prep_t)
    frost_kda_prep_prefill_prologue(
        order_gen,
        b_t,
        num_ctas,
        tiles_per_head,
        base_q,
        base_k,
        base_v,
        base_gate,
        base_o,
        base_checkpoint,
        base_k_decay,
        base_q_decay,
        base_t,
        base_a,
        base_diag,
        tensormap_workspace,
        cu_seqlens,
        q,
        k,
        v,
        gate,
        o,
        state_checkpoints,
        prep_k_decay,
        prep_q_decay,
        prep_t,
        prep_a,
        prep_diag,
        work_item_staging,
        work_count,
        work_items,
        scheduler_counter,
        cutlass.Int32(batch_size),
        checkpoint_every_n,
        prep_base_q,
        prep_base_k,
        prep_base_gate,
        prep_record_maps[0],
        prep_record_maps[1],
        prep_record_maps[2],
        prep_desc_words,
        prep_rows,
        prep_row_count,
    ).launch(grid=(3, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    raw_gate: cute.Tensor,
    a_log: cute.Tensor | None,
    dt_bias: cute.Tensor | None,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    initial_state: cute.Tensor | None,
    out: cute.Tensor,
    final_state: cute.Tensor | None,
    seed_indices: cute.Tensor | None,
    final_indices: cute.Tensor | None,
    work_items: cute.Tensor | None,
    work_count: cute.Tensor | None,
    scheduler_counter: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    checkpoint_every_n_tokens: cutlass.Int32,
    scale: cutlass.Float32,
    stream,
) -> None:
    heads_out = cutlass.Int32(raw_gate.shape[1])
    if cutlass.const_expr(cfg.tiles_per_head > 1):
        heads_out = heads_out * cutlass.Int32(cfg.tiles_per_head)
    v_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(v.shape[1]))
    num_sequences = cu_seqlens.shape[0] - 1

    @cute.struct
    class SharedStorage:
        k_decay: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, cfg.k_decay_cosize], cfg.buffer_align_bytes]
        q_decay: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, cfg.q_decay_cosize], cfg.buffer_align_bytes]
        k_restore: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, cfg.k_restore_cosize], cfg.buffer_align_bytes]
        intermediate: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, cfg.intermediate_cosize], cfg.buffer_align_bytes]
        v: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, cfg.v_cosize], cfg.buffer_align_bytes]
        gate: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, cfg.gate_cosize], 1024]
        output: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, cfg.o_cosize], cfg.buffer_align_bytes]
        if cfg.enable_checkpoints:
            checkpoint: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.smem_checkpoint_stages * cfg.d_k * cfg.d_v], cfg.buffer_align_bytes
            ]

    # ---- launch ----------------------------------------------------------------------
    grid_shape = (cfg.max_active_clusters, 1, 1)
    frost_kda_prep_prefill(
        cfg,
        SharedStorage,
        v_ratio,
        tensormap_workspace,
        cutlass.Int32(num_sequences),
        q,
        k,
        v,
        raw_gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        initial_state,
        out,
        final_state,
        seed_indices,
        final_indices,
        work_items,
        work_count,
        scheduler_counter,
        scale,
        checkpoint_every_n_tokens,
    ).launch(
        grid=grid_shape,
        block=(cfg.threads_per_cta, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
        min_blocks_per_mp=1,
    )


@cute.kernel
def frost_kda_prep_prefill(
    cfg: cutlass.Constexpr,
    shared_type: cutlass.Constexpr,
    v_ratio: cute.FastDivmodDivisorV2,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mGate: cute.Tensor,
    mA_log: cute.Tensor | None,
    mDt_bias: cute.Tensor | None,
    mBeta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    mState_init: cute.Tensor | None,
    mO: cute.Tensor,
    mState_out: cute.Tensor | None,
    mSeedIndices: cute.Tensor | None,
    mFinalIndices: cute.Tensor | None,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor,
    scale: cutlass.Float32,
    checkpoint_every_n_tokens: cutlass.Int32,
) -> None:
    """BT=16 KDA forward kernel (persistent, tile-scheduled)."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()

    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = tidx % cfg.threads_per_warp

    total_tiles = mCount[0]
    desc_base_words = tensormap_workspace.iterator.raw_ptr()
    arr_words = n_desc * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_v_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_gate_base = desc_base_words + cutlass.Int32(3) * arr_words
    desc_o_base = desc_base_words + cutlass.Int32(4) * arr_words
    desc_checkpoint_base = desc_base_words + cutlass.Int32(5) * arr_words
    desc_k_decay_base = desc_base_words + cutlass.Int32(6) * arr_words
    desc_q_decay_base = desc_base_words + cutlass.Int32(7) * arr_words
    desc_t_base = desc_base_words + cutlass.Int32(8) * arr_words
    desc_a_base = desc_base_words + cutlass.Int32(9) * arr_words
    desc_diag_base = desc_base_words + cutlass.Int32(10) * arr_words

    # Barrier/control rings stay declaration-ordered; data buffers share one storage allocation.
    SMEM = cutlass.AddressSpace.smem
    bars = make_bars(cfg)
    tmem_base_slot = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=4)
    sScheduler = cutlass.Array(cutlass.Int32, cfg.scheduler_stages, space=SMEM, alignment=16)
    storage = SmemAllocator().allocate(shared_type)
    sK_decay_raw = storage.k_decay.get_tensor(cute.make_layout((cfg.k_decay_cosize,)))
    sQ_decay_raw = storage.q_decay.get_tensor(cute.make_layout((cfg.q_decay_cosize,)))
    sK_restore_raw = storage.k_restore.get_tensor(cute.make_layout((cfg.k_restore_cosize,)))
    sIntermediate_raw = storage.intermediate.get_tensor(cute.make_layout((cfg.intermediate_cosize,)))
    sV_raw = storage.v.get_tensor(cute.make_layout((cfg.v_cosize,)))
    sGate_raw = storage.gate.get_tensor(cute.make_layout((cfg.gate_cosize,)))
    smem_data_ptr(sGate_raw)
    sGate_exchange_raw = sGate_raw
    sO_raw = storage.output.get_tensor(cute.make_layout((cfg.o_cosize,)))
    sCheckpoint_raw = (
        storage.checkpoint.get_tensor(cute.make_layout((cfg.smem_checkpoint_stages * cfg.d_k * cfg.d_v,)))
        if cutlass.const_expr(cfg.enable_checkpoints)
        else sO_raw
    )
    sK_decay = SmemTile(
        base=sK_decay_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_decay_stages,
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    sQ_decay = SmemTile(
        base=sQ_decay_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_decay_stages,
        leading_byte_offset=16,
        stride_byte_offset=1024,
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    sK_restore_trans = SmemTile(
        base=sK_restore_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_decay_stages,
        leading_byte_offset=(cfg.b_t * 128),
        stride_byte_offset=(8 * 128),
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    sIntermediate = SmemTile(
        base=sIntermediate_raw,
        elems_per_stage=(2 * cfg.b_t * cfg.b_t),
        stages=cfg.smem_intermediate_stages,
        leading_byte_offset=16,
        stride_byte_offset=(8 * cfg.b_t * 2),
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_32B,
    )

    elect_one = nvvm.elect_sync()

    # ---- mbarrier init (one lane per owning role) ------------------------------------
    if warp_idx == cfg.tma_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_raw_bar_stages):
                bars.mb_raw_ready[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_raw_stages):
                bars.mb_raw_done[stage].init()
    elif warp_idx == cfg.tcgen05_mma_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.tmem_q_state_acc_stages):
                bars.mb_o_acc_ready[stage].init()
                bars.mb_o_acc_done[stage].init()
            bars.mb_state_k_acc_ready.init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_state_acc_cg0_done[stage].init()
                bars.mb_state_acc_cg2_done[stage].init()
            bars.mb_state_input_cg2_ready.init()
            bars.mb_state_input_cg0_ready.init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_decay_tcgen05_done[stage].init()
                bars.mb_k_restore_done[stage].init()
            bars.mb_u_input_ready.init()
            bars.mb_tmem_done[0].init()
    elif warp_idx == cfg.scheduler_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_intermediate_stages):
                bars.mb_a_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_k_decay_inv_cg0_ready[stage].init()
    elif warp_idx == cfg.epilogue_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_o_stages):
                bars.mb_o_tmastg_ready[stage].init()
                bars.mb_o_tmastg_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.scheduler_stages):
                bars.mb_scheduler_ready[stage].init()
                bars.mb_scheduler_done[stage].init()
            if cutlass.const_expr(cfg.enable_checkpoints):
                for stage in cutlass.range_constexpr(cfg.smem_checkpoint_stages):
                    bars.mb_checkpoint_tmastg_ready[stage].init()
                    bars.mb_checkpoint_tmastg_done[stage].init()
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync(0, thread_count=cfg.threads_per_cta)

    # ---- warp specialization ---------------------------------------------------------
    if warp_idx == cfg.tma_warp_id:
        tmaldg_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            mScheduler,
            sScheduler,
            lane_idx,
            sV_raw,
            sGate_raw,
            desc_v_base,
            desc_gate_base,
            desc_k_decay_base,
            desc_q_decay_base,
            desc_t_base,
            desc_a_base,
            desc_diag_base,
            sK_decay_raw,
            sQ_decay_raw,
            sK_restore_raw,
            sIntermediate_raw,
            bars,
            v_ratio=v_ratio,
        )
    elif warp_idx == cfg.scheduler_warp_id:
        scheduler_warp(
            cfg,
            total_tiles,
            bidx,
            cu_seqlens,
            mWorkItems,
            sScheduler,
            bars,
        )
    elif warp_idx == cfg.tcgen05_mma_warp_id:
        tcgen05_mma_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sScheduler,
            tmem_base_slot,
            sIntermediate,
            sK_decay,
            sK_restore_trans,
            sQ_decay,
            bars,
        )
    elif warp_idx == cfg.epilogue_warp_id:
        epilogue_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sScheduler,
            lane_idx,
            mO,
            sO_raw,
            sCheckpoint_raw,
            desc_o_base,
            desc_checkpoint_base,
            checkpoint_every_n_tokens,
            bars,
        )
    elif warp_idx >= cfg.compute_group_0_warp_ids[0] and warp_idx <= cfg.compute_group_0_warp_ids[-1]:
        compute0_warp_group(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sScheduler,
            lane_idx,
            warp_idx,
            sGate_exchange_raw,
            sCheckpoint_raw,
            tmem_base_slot,
            checkpoint_every_n_tokens,
            bars,
        )
    elif warp_idx >= cfg.compute_group_2_warp_ids[0] and warp_idx <= cfg.compute_group_2_warp_ids[-1]:
        compute2_warp_group(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sScheduler,
            lane_idx,
            tmem_base_slot,
            warp_idx,
            mState_out,
            mState_init,
            mSeedIndices,
            mFinalIndices,
            mO,
            sO_raw,
            sV_raw,
            sCheckpoint_raw,
            sGate_exchange_raw,
            checkpoint_every_n_tokens,
            scale,
            bars,
        )
    elif warp_idx >= cfg.compute_group_1_warp_ids[0] and warp_idx <= cfg.compute_group_1_warp_ids[-1]:
        compute1_warp_group(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sScheduler,
            lane_idx,
            warp_idx,
            tmem_base_slot,
            mO,
            sO_raw,
            scale,
            bars,
        )


@dataclass
class KdaPrepPrefillCfg:
    """Kernel cfg (fixed BT=16 schedule constants; derived TMEM column offsets
    and SMEM buffer cosizes are stamped by ``build_cfg``; per-stage sizes are
    inlined at the use sites).  Passed ``cfg``-first (a ``cutlass.Constexpr``)
    into ``host`` / ``kernel`` and every warp body."""

    io_dtype: Type[cutlass.Numeric]
    state_dtype: Type[cutlass.Numeric]
    gate_dtype: Type[cutlass.Numeric]
    use_initial_state: bool
    store_final_state: bool
    enable_checkpoints: bool
    l2norm: bool
    safe_gate: bool
    gate_scale_log2: float
    log_gate: bool
    beta_sigmoid: bool
    allow_neg_eigval: bool
    max_active_clusters: int
    d_k: int
    d_v: int
    tiles_per_head: int = 1
    scheduler_stages: int = CFG.SMEM_SCHEDULER_STAGES

    compute_group_0_warp_ids: tuple[int, ...] = (0, 1, 2, 3)
    compute_group_2_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    compute_group_1_warp_ids: tuple[int, ...] = (4, 5, 6, 7)
    scheduler_warp_id: int = CFG.REGISTER_MMA_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    tma_warp_id: int = CFG.TMA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    b_t: int = CFG.B_T
    threads_per_warp: int = CFG.THREADS_PER_WARP
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES
    threads_per_cta: int = 0
    tmem_user_threads: int = 0
    tmem_lifecycle_barrier_id: int = 3
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_2: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_other: int = CFG.NUM_REGS_OTHER

    # ---- SMEM / TMEM ring stage counts -----------------------------------------------
    smem_raw_stages: int = CFG.SMEM_RAW_STAGES
    smem_raw_bar_stages: int = 0  # ready/beta-ring mbar depth: raw rounded up to even
    smem_o_stages: int = CFG.SMEM_O_STAGES
    smem_checkpoint_stages: int = 1
    smem_decay_stages: int = CFG.SMEM_DECAY_STAGES
    smem_intermediate_stages: int = CFG.SMEM_INTERMEDIATE_STAGES
    tmem_q_state_acc_stages: int = CFG.TMEM_Q_STATE_ACC_STAGES

    # ---- TMEM column offsets (state doubles as the final state acc) ------------------
    tmem_state_acc_offset: int = 0
    tmem_state_input_offset: int = 0
    tmem_q_state_acc_offset: int = 0
    tmem_state_k_acc_offset: int = 0
    tmem_u_input_offset: int = 0

    # ---- SMEM buffer cosizes ---------------------------------------------------------
    v_cosize: int = 0
    gate_cosize: int = 0
    gate_stage_elems: int = 0
    k_decay_cosize: int = 0
    q_decay_cosize: int = 0
    k_restore_cosize: int = 0
    o_cosize: int = 0

    # ---- TMA transaction bytes per stage ---------------------------------------------
    tma_k_bytes: int = 0
    tma_v_bytes: int = 0
    tma_gate_bytes: int = 0
    intermediate_cosize: int = 0


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    state_dtype: Type[cutlass.Numeric],
    gate_dtype: Type[cutlass.Numeric],
    *,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_scale_log2: float,
    log_gate: bool = True,
    beta_sigmoid: bool,
    allow_neg_eigval: bool,
    max_active_clusters: int,
    d_k: int,
    d_v: int,
    tiles_per_head: int = 1,
) -> KdaPrepPrefillCfg:
    """Build the per-compile ``KdaPrepPrefillCfg`` (io_dtype in {Float16, BFloat16}); fills the derived TMEM column
    offsets and SMEM buffer cosizes.  ``tiles_per_head`` > 1 runs every gate head as ``tiles_per_head`` tiles of ``d_v`` value
    columns each (the d_v split)."""
    cfg = KdaPrepPrefillCfg(
        io_dtype=io_dtype,
        state_dtype=state_dtype,
        gate_dtype=gate_dtype,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        l2norm=l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        log_gate=log_gate,
        beta_sigmoid=beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        max_active_clusters=max_active_clusters,
        d_k=d_k,
        d_v=d_v,
        tiles_per_head=tiles_per_head,
    )
    if enable_checkpoints:
        cfg.smem_checkpoint_stages = 2
    cfg.smem_raw_bar_stages = cfg.smem_raw_stages + (cfg.smem_raw_stages % 2)
    cfg.threads_per_cta = 16 * cfg.threads_per_warp
    cfg.tmem_user_threads = (
        1 + len(cfg.compute_group_2_warp_ids) + len(cfg.compute_group_0_warp_ids) + len(cfg.compute_group_1_warp_ids)
    ) * cfg.threads_per_warp
    if len(cfg.compute_group_0_warp_ids) != len(cfg.compute_group_2_warp_ids):
        raise ValueError("the state halves are packed by CG0 and by CG2: their warp counts must match")

    cfg.tmem_state_input_offset = cfg.tmem_state_acc_offset + cfg.d_k
    cfg.tmem_q_state_acc_offset = cfg.tmem_state_input_offset + (cfg.d_k // 2)
    cfg.tmem_state_k_acc_offset = cfg.tmem_q_state_acc_offset + cfg.tmem_q_state_acc_stages * cfg.b_t
    cfg.tmem_u_input_offset = cfg.tmem_state_k_acc_offset + cfg.b_t
    assert (cfg.tmem_u_input_offset + (cfg.b_t // 2)) <= 512

    cfg.v_cosize = cfg.smem_raw_stages * cfg.d_v * cfg.b_t
    cfg.gate_cosize = cfg.smem_raw_stages * cfg.d_k * cfg.b_t * (cfg.gate_dtype.width // 8) // 4
    cfg.gate_stage_elems = cfg.d_k * cfg.b_t
    cfg.k_decay_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.q_decay_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.k_restore_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.o_cosize = cfg.smem_o_stages * cfg.b_t * cfg.d_v
    cfg.intermediate_cosize = cfg.smem_intermediate_stages * 2 * cfg.b_t * cfg.b_t
    cfg.tma_k_bytes = cfg.d_k * cfg.b_t * (cfg.io_dtype.width // 8)
    cfg.tma_v_bytes = cfg.d_v * cfg.b_t * (cfg.io_dtype.width // 8)
    cfg.tma_gate_bytes = cfg.d_k * cfg.b_t * (cfg.gate_dtype.width // 8)
    cfg.gate_cosize = cfg.smem_raw_stages * cfg.d_k
    cfg.gate_stage_elems = cfg.d_k
    cfg.tma_gate_bytes = cfg.d_k * 4
    cfg.smem_decay_stages = 4
    cfg.smem_intermediate_stages = 4
    cfg.k_decay_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.q_decay_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.k_restore_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.intermediate_cosize = cfg.smem_intermediate_stages * 2 * cfg.b_t * cfg.b_t
    return cfg


TENSORMAP_DESC_ARRAYS = 11  # per-batch runtime TMA descriptors: Q, K, V, Gate, O, state_checkpoints, k_decay, q_decay, t, a, diag


# ---------------------------------------------------------------------------


frost_kda_prep_prefill_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_kda_prep_prefill.set_name_prefix("cudnn", remove_cutlass_symbol=False)
