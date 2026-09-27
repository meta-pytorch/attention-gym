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
# attn_gym.linear._delta_rule.cudnn_fe. The T_inv and V rings live in a SharedStorage struct, raw
# SMEM pointers go through smem_data_ptr, and the checkpoint staging offset uses
# swizzle_box_offset_128b.
# Modified by Attention Gym in 2026: frozen cfg (build_cfg derives the SMEM cosizes and TMA byte
# counts) and jit_cache fake-tensor TVM-FFI compiles for the standalone launcher.

"""
Chunked Gated Delta Net (GDN) recompute (state/checkpoint-only) kernel for SM100 / SM103 / SM107
(Cutlass primitives): the recurrent state and its checkpoint series, with no Q or O path.

Algorithm overview (per chunk c, tokens [cC, (c+1)C)):
  Inputs : K[BT,DK], V[BT,DV], Gate[BT] (scalar gate), T_inv[BT,BT] (chunk factor tile)
  State  : S_prev[DK,DV]  (recurrent state, held in TMEM)

  Preprocessing (gate warp):
    cumsumlog[t]     = sum_{l=0}^{t} log(Gate_l)              cumulative log of gates
    cumprod[t]       = exp(cumsumlog[t])                       cumulative product of gates

  K*state GEMM   : KS[BT,DV] = K  @ S_prev    (key applied to state)
  U GEMM         : U[BT,DV]  = T_inv @ Y       (corrected value vectors)
                   where T_inv = (I + M_kk)^{-1},  M_kk[i,j] = T[i,j]*Beta[i]*(K K^T)[i,j]  (the chunk factor tile of
                   gdn_tinv_f16.py, beta folded in)
  KV update GEMM : S_upd[DK,DV] = K^T @ (decay .* U)  (state update, BT contraction)
                   where Y[BT,DV] = V - KS    (delta rule residuals, after decay)

  Epilogue:
    S_next    = cumprod[BT-1] * S_prev + S_upd        (update state in TMEM)

Each chunk owns one K stage and lands in box (chunk parity) of it.

SMEM layout (stage counts live in gdn_recompute_config.py;
enable_checkpoints compiles trim K stages to fit the checkpoint buffer):
  Buffer                       Size (B)  Stages
  K (two-box stage)               32768       4
  V                               16384       2
  T_inv                            8192       3
  checkpoint staging            DK*DV*2       1    <-- enable_checkpoints only
  cumsumlog / cumprod               256       3
  scheduler ticket ring               4       2    <-- next-tile publish ring

TMEM layout (512 columns):
  Buffer                  Cols
  state                   128     <-- DKxDV fp32 = 128x128x4B
  state input                64     <-- fp16 state staging (K*state A operand)
  cg1 shared acc           64     <-- 1-stage ring: KS then U
  Y / decayed-U input        64     <-- slot 0 = Y (V - K*state), slot 1 = decayed U (b16)

The chunk factor comes from the tiles of ``gdn_tinv_f16.py`` (same k / gate / beta / cu_seqlens);
the TMA warp loads one 8192 B tile per chunk into the T_inv ring through the tiles' per-batch
tensor map, built by the prologue like K's.

Warp assignments (8 warps = 256 threads):
  warps 0-3     : compute group 1 - state restage/rescale, Y = V - K*state,
                                    U epilogue
  warp  4       : Gate loads
  warp  5       : TMA load warp  - loads K, V and the chunk-factor tiles
  warp  6       : MMA warp       - K*state/U/KV per chunk; TMEM lifecycle
  warp  7       : epilogue warp  - checkpoint TMA stores
"""

from dataclasses import dataclass, replace
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.experimental.cuda.tensor_map as tma
import cutlass.experimental.primitives as nvvm
from cutlass import cute

from attn_gym._backends.cute.compat import SmemAllocator

from ..common.launch import (
    validate_named_barriers,
    validate_warp_roles,
)
from ..common.split_k import (
    ORDER_CAPACITY,
    ORDER_ELEMENTS,
    ORDER_THREADS,
    decode_head,
    decode_work_item,
    expanded_cu_seqlen,
    gen_interval_items,
    order_body,
)
from ..common.thd import (
    TENSOR_MAP_QWORDS,
    emit_checkpoint_seq_descs,
    emit_seq_descs,
    emit_tile_seq_descs,
)

RCP_LN2 = 1.4426950408889634  # 1/ln(2): natural-log gates -> the kernel's log2 domain
from ..tile_dsl.barrier import (
    MBarrier,
    PipelineState,
    Producer,
    advance,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from ..tile_dsl.handles import MmaDesc, SmemTile, smem_data_ptr, tma_slice_runtime_desc
from ..tile_dsl.mma import mma_ts_step
from ..tile_dsl.pointwise import (
    fadd2,
    fmul2,
    fp32_to_fp16,
    opaque_f32_zero,
    opaque_i32_zero,
    softplus2,
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
from .gdn_recompute_config import CFG

USE_PDL = True
STATE_DIMS = (64, 128)
TMEM_COLUMNS = 512
IO_DTYPES = ("bfloat16", "float16")
SCALAR_DTYPES = ("float32", "bfloat16", "float16")
STATE_DTYPES = ("float32", "bfloat16")


class GdnRecomputeBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_kq_ready: MBarrier
    mb_kq_done: MBarrier
    mb_v_ready: MBarrier
    mb_v_done: MBarrier

    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier

    mb_state_acc_ready: MBarrier

    mb_state_input_ready: MBarrier
    mb_y_input_ready: MBarrier
    mb_decay_u_input_ready: MBarrier

    mb_t_inv_ready: MBarrier
    mb_t_inv_done: MBarrier

    mb_k_state_acc_ready: MBarrier
    mb_u_acc_ready: MBarrier

    mb_checkpoint_tmastg_ready: MBarrier
    mb_checkpoint_tmastg_done: MBarrier

    mb_tmem_done: MBarrier

    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier


def make_bars(cfg) -> GdnRecomputeBars:
    """GdnRecomputeBars factory."""
    ONE_LANE = 1
    MMA_ARRIVERS = len([cfg.tcgen05_mma_warp_id])
    KQ_RELEASE_SITES = 1
    GATE_WARP = cfg.threads_per_warp * len([cfg.load_gate_warp_id])
    EPI_WARP = cfg.threads_per_warp * len([cfg.epilogue_warp_id])
    CG1_THREADS = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    CONSUMER_WARPS = len(cfg.compute_group_1_warp_ids) + len(
        [cfg.load_gate_warp_id, cfg.tcgen05_mma_warp_id, cfg.epilogue_warp_id]
    )

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=16)

    return GdnRecomputeBars(
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
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_state_acc_ready=MBarrier(
            alloc(cfg.tmem_state_acc_stages),
            try_wait=True,
            stages=cfg.tmem_state_acc_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
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
        mb_decay_u_input_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_t_inv_ready=MBarrier(
            alloc(cfg.smem_t_inv_stages),
            try_wait=True,
            stages=cfg.smem_t_inv_stages,
            init_count=ONE_LANE,
            producer=Producer.TMA_LOAD,
        ),
        mb_t_inv_done=MBarrier(
            alloc(cfg.smem_t_inv_stages),
            try_wait=True,
            stages=cfg.smem_t_inv_stages,
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
            init_count=CONSUMER_WARPS,
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
    sCheckpoint_raw,
    desc_checkpoint_base,
    sScheduler,
    bars,
):
    """Epilogue warp role (warp 7): persistent scheduler loop issuing the
    per-chunk checkpoint TMA stores."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    scheduler_state = PipelineState.start(phase=0)

    tidx % cfg.threads_per_warp

    elect_one = nvvm.elect_sync()
    tile_idx = cutlass.Int32(bidx)

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
        if cutlass.const_expr(cfg.enable_checkpoints):
            desc_checkpoint_slot = (desc_checkpoint_base + slot).tospace(
                cutlass.AddressSpace.generic
            )
            checkpoint_coord = (
                write_start + checkpoint_chunks - cutlass.Int32(1)
            ) // checkpoint_chunks
            checkpoint_mod = compute_start % checkpoint_chunks
            if elect_one:
                tma_tensormap_acquire(desc_checkpoint_slot)

        if n_local > 0:
            for local_idx in cutlass.range(n_local):
                chunk_idx = compute_start + local_idx

                did_checkpoint = cutlass.Int32(0)
                if cutlass.const_expr(cfg.enable_checkpoints):
                    checkpoint_stage = checkpoint_store_cnt % cfg.smem_checkpoint_stages
                    checkpoint_phase = (
                        checkpoint_store_cnt // cfg.smem_checkpoint_stages
                    ) & cutlass.Int32(1)
                    if chunk_idx >= write_start and chunk_idx < write_end:
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

                    if did_checkpoint == 1:
                        tma_store_wait(0)
                        bars.mb_checkpoint_tmastg_done[checkpoint_stage].arrive()
                        checkpoint_store_cnt = checkpoint_store_cnt + 1

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
    mA_log,
    mDt_bias,
    sCumsumlog,
    sCumprod,
    sScheduler,
    bars,
):
    """Gate producer: persistent scheduler loop + the cumsum/cumprod chunk loads."""

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
        if cutlass.const_expr(cfg.safe_gate and mA_log is not None):
            if n_local > 0:
                a = -cute.math.exp2(
                    mA_log[head_o].to(cutlass.Float32) * cutlass.Float32(RCP_LN2), fastmath=True
                ) * cutlass.Float32(RCP_LN2)
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

                oob_neutral = (
                    cutlass.Float32(0.0)
                    if cutlass.const_expr(cfg.log_gate)
                    else cutlass.Float32(1.0)
                )
                toks = [
                    chunk_offset + lane_idx + col * cfg.threads_per_warp for col in range(n_cols)
                ]
                pos_valid = [tok < batch_end for tok in toks]
                if cutlass.const_expr(cfg.expand_num > 1):
                    gate_rows = [tok // cutlass.Int32(cfg.expand_num) for tok in toks]
                    gate_valid = [
                        valid and (tok - row * cutlass.Int32(cfg.expand_num) == 0)
                        for tok, row, valid in zip(toks, gate_rows, pos_valid)
                    ]
                    gate_row_end = batch_end // cutlass.Int32(cfg.expand_num)
                    gate_vals = [
                        gGateSeq[cutlass.min(row, gate_row_end - 1)].to(cutlass.Float32)
                        if valid
                        else oob_neutral
                        for row, valid in zip(gate_rows, gate_valid)
                    ]
                else:
                    gate_valid = pos_valid
                    gate_vals = [
                        gGateSeq[cutlass.min(tok, batch_end - 1)].to(cutlass.Float32)
                        if valid
                        else oob_neutral
                        for tok, valid in zip(toks, pos_valid)
                    ]

                if cutlass.const_expr(cfg.safe_gate):
                    for col in cutlass.range_constexpr(0, n_cols, 2):
                        biased_lo, biased_hi = fadd2(
                            gate_vals[col], gate_vals[col + 1], bias, bias
                        )
                        sp_lo, sp_hi = softplus2(biased_lo, biased_hi)
                        contrib_lo, contrib_hi = fmul2(sp_lo, sp_hi, a, a)
                        gate_vals[col] = contrib_lo if gate_valid[col] else cutlass.Float32(0.0)
                        gate_vals[col + 1] = (
                            contrib_hi if gate_valid[col + 1] else cutlass.Float32(0.0)
                        )
                elif cutlass.const_expr(cfg.log_gate):
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
        tile_idx, scheduler_state = scheduler_next_tile(
            cfg, bars, sScheduler, scheduler_state, elect_one
        )

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
    sKQ,
    sKQ_trans,
    sTinv,
    sScheduler,
    bars,
):
    """MMA issuer role: persistent scheduler loop issuing every tcgen05 GEMM."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    kv_acc_index = PipelineState.start(phase=1)
    kq_index = PipelineState.start(phase=0)
    tinv_index = PipelineState.start(phase=0)
    state_input_index = PipelineState.start(phase=0)
    y_input_ready = PipelineState.start(phase=0)
    decay_u_input_ready = PipelineState.start(phase=0)

    elect_one = nvvm.elect_sync()

    nvvm.tcgen05_alloc(tmem_base_slot, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads
    )

    # ---- chunk-invariant GEMM descriptors --------------------------------------------
    bytes_per_element = cfg.io_dtype.width // 8
    instruction_descriptor_k_state = nvvm.Tcgen05InstrDesc.build(
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
        idesc=instruction_descriptor_k_state,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    instruction_descriptor_u_ts = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_y_t_inv_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_u_ts,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
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
    KQ_SUBTILES = bmm_state_k_desc.num_subtiles
    KQ_STEPS = bmm_state_k_desc.steps_per_subtile
    KQ_A_SEG = KQ_STEPS * bmm_state_k_desc.tmem_advance_A

    KV_ACC_STAGE_COLS = cfg.d_k
    STATE_INP_STAGE_COLS = cfg.d_k // 2
    INP_SLOT_COLS = cfg.b_t // 2

    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    tmem_row << 16
    (tmem_row + 16) << 16
    tmem_state_col = tmem_col + cfg.tmem_state_acc_offset
    tmem_state_input_col = tmem_col + cfg.tmem_state_input_offset
    tmem_input_col = tmem_col + cfg.tmem_y_decay_u_input_offset
    y_input_ptr = nvvm.make_tmem_ptr(tmem_input_col, cutlass.Int8)
    decay_u_input_ptr = nvvm.make_tmem_ptr(tmem_input_col + INP_SLOT_COLS, cutlass.Int8)
    k_state_acc_ptr = nvvm.make_tmem_ptr(tmem_col + cfg.tmem_cg1_acc_offset, cutlass.Float32)
    u_acc_ptr = k_state_acc_ptr

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

        if cutlass.const_expr(cfg.use_initial_state or cfg.seed_identity):
            seed_state = compute_start == 0
        for local_idx in cutlass.range(n_local):
            if cutlass.const_expr(cfg.seed_checkpoints):
                have_state = cutlass.Boolean(True)
            elif cutlass.const_expr(cfg.use_initial_state or cfg.seed_identity):
                have_state = local_idx > 0 or seed_state
            else:
                have_state = local_idx > 0

            kq_idx = kq_index.idx
            member = local_idx & 1
            state_input_idx = state_input_index.idx
            tinv_idx = tinv_index.idx
            kv_acc_idx = kv_acc_index.idx
            kq_member_off = member * KQ_BOX
            desc_k = sKQ[kq_idx].desc() + kq_member_off
            desc_tinv = sTinv[tinv_idx].desc()
            desc_k_trans = sKQ_trans[kq_idx].desc() + kq_member_off
            state_a_ptr = nvvm.make_tmem_ptr(
                tmem_state_input_col + state_input_idx * STATE_INP_STAGE_COLS, cutlass.Int8
            )
            state_acc_ptr = nvvm.make_tmem_ptr(
                tmem_state_col + kv_acc_idx * KV_ACC_STAGE_COLS, cutlass.Float32
            )

            # ---- K tile landed -------------------------------------------------------
            bars.mb_kq_ready[kq_idx].wait(kq_index.phase)
            kq_index = advance(kq_index, cfg.smem_kq_stages)

            # ---- k state = state(T) @ K^T --------------------------------------------
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

            # ---- U = Y(T) @ T^-1 -----------------------------------------------------
            bars.mb_t_inv_ready[tinv_idx].wait(tinv_index.phase)
            tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
            bars.mb_y_input_ready[0].wait(y_input_ready.phase)
            y_input_ready = advance(y_input_ready, 1)
            for k in cutlass.range_constexpr(cfg.b_t // 16):
                mma_ts_step(
                    bmm_y_t_inv_desc, y_input_ptr, desc_tinv, u_acc_ptr, k, cutlass.Boolean(k > 0)
                )
            if elect_one:
                bars.mb_u_acc_ready[0].arrive(cta_group=1)
                bars.mb_t_inv_done[tinv_idx].arrive(cta_group=1)

            # ---- state += decayed U(T) @ K -------------------------------------------
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
    sKQ_raw,
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
    """TMA-LDG warp role: persistent scheduler loop + per-chunk K/V G->S TMA
    loads, plus the chunk-factor tile TMA loads."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    kq_index = PipelineState.start(phase=1)
    v_index = PipelineState.start(phase=1)
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
        head_k = head_idx // k_ratio
        head_v = head_idx // v_ratio
        slot = batch_idx * desc_qwords
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_k_slot)
            if cutlass.const_expr(not cfg.v_is_zero):
                tma_tensormap_acquire(desc_v_slot)
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
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), head_k, tok_coord)
            kq_tile = sKQ_lo_tma[kq_idx]
            tma_load_tile(kq_tile, k_slice, bars.mb_kq_ready[kq_idx].smem_ptr)
            for chunk_idx in cutlass.range(compute_start + 1, write_end):
                tok_coord = chunk_idx * cutlass.Int32(cfg.b_t)

                # ---- K load ----------------------------------------------------------
                kq_idx = kq_index.idx
                bars.mb_kq_done[kq_idx].wait(kq_index.phase)
                kq_index = advance(kq_index, cfg.smem_kq_stages)
                if elect_one:
                    bars.mb_kq_ready[kq_idx].arrive(n_bytes=cfg.tma_kq_bytes)
                member = (chunk_idx - compute_start) & 1
                k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), head_k, tok_coord)
                kq_tile = sKQ_lo_tma[kq_idx]
                if member == 0:
                    tma_load_tile(kq_tile, k_slice, bars.mb_kq_ready[kq_idx].smem_ptr)
                else:
                    tma_load_tile(
                        kq_tile.shifted(kq_box_elements),
                        k_slice,
                        bars.mb_kq_ready[kq_idx].smem_ptr,
                    )

                # ---- V load ----------------------------------------------------------
                if cutlass.const_expr(not cfg.v_is_zero):
                    v_idx = v_index.idx
                    bars.mb_v_done[v_idx].wait(v_index.phase)
                    v_index = advance(v_index, cfg.smem_v_stages)
                    if elect_one:
                        bars.mb_v_ready[v_idx].arrive(n_bytes=cfg.tma_v_bytes)
                    v_tok = (chunk_idx - 1) * cutlass.Int32(cfg.b_t)
                    v_slice = tma_slice_runtime_desc(desc_v_slot, v_offset, head_v, v_tok)
                    tma_load_tile(sV_tma[v_idx], v_slice, bars.mb_v_ready[v_idx].smem_ptr)

                # ---- chunk-factor tile load ------------------------------------------
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

            if cutlass.const_expr(not cfg.v_is_zero):
                v_idx = v_index.idx
                bars.mb_v_done[v_idx].wait(v_index.phase)
                v_index = advance(v_index, cfg.smem_v_stages)
                if elect_one:
                    bars.mb_v_ready[v_idx].arrive(n_bytes=cfg.tma_v_bytes)
                v_tok = (write_end - cutlass.Int32(1)) * cutlass.Int32(cfg.b_t)
                v_slice = tma_slice_runtime_desc(desc_v_slot, v_offset, head_v, v_tok)
                tma_load_tile(sV_tma[v_idx], v_slice, bars.mb_v_ready[v_idx].smem_ptr)
            tinv_idx = tinv_index.idx
            bars.mb_t_inv_done[tinv_idx].wait(tinv_index.phase)
            tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
            if elect_one:
                bars.mb_t_inv_ready[tinv_idx].arrive(n_bytes=cfg.tma_tinv_bytes)
            tinv_row = write_end - cutlass.Int32(1)
            tinv_slice = tma_slice_runtime_desc(
                desc_tinv_slot, cutlass.Int32(0), cutlass.Int32(0), head_o, tinv_row
            )
            tma_load_tile(sTinv_tma[tinv_idx], tinv_slice, bars.mb_t_inv_ready[tinv_idx].smem_ptr)

        tile_idx, scheduler_state = scheduler_publish_next(
            cfg, bars, sScheduler, mScheduler, scheduler_state, num_ctas, elect_one
        )

    for _ in range(cfg.smem_kq_stages):
        bars.mb_kq_done[kq_index.idx].wait(kq_index.phase)
        kq_index = advance(kq_index, cfg.smem_kq_stages)
    if cutlass.const_expr(not cfg.v_is_zero):
        for _ in range(cfg.smem_v_stages):
            bars.mb_v_done[v_index.idx].wait(v_index.phase)
            v_index = advance(v_index, cfg.smem_v_stages)
    for _ in range(cfg.smem_t_inv_stages):
        bars.mb_t_inv_done[tinv_index.idx].wait(tinv_index.phase)
        tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


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
    sV_trans,
    sCumsumlog,
    sCumprod,
    sCheckpoint_raw,
    mState_init,
    mState_out,
    mSeedCheckpoints,
    checkpoint_every_n_tokens,
    seed_every_n_tokens,
    sScheduler,
    bars,
):
    """Compute warp-group 1 role (warps 0-3): persistent scheduler loop
    owning the recurrent state from seed to final store."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_1, nvvm.SetMaxRegisterAction.INCREASE)

    v_index = PipelineState.start(phase=0)
    gate_index = PipelineState.start(phase=0)
    kv_acc_index = PipelineState.start(phase=0)
    k_state_ready_index = PipelineState.start(phase=0)
    u_acc_ready_index = PipelineState.start(phase=0)

    num_threads_cg1 = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    cg1_tidx = tidx % num_threads_cg1
    lane_idx = cg1_tidx % cfg.threads_per_warp

    elect_one = nvvm.elect_sync()
    state_input_cnt = cutlass.Int32(0)
    ldtm_width = 32
    sttm_width = 16
    num_ldtms = cutlass.const_expr(cfg.d_k // ldtm_width)
    INP_SLOT_COLS = cfg.b_t // 2
    dv_halves = cutlass.const_expr(cfg.d_v // 64)
    v_row = cg1_tidx % 8 + (cg1_tidx // 16 % 2) * 8
    if cutlass.const_expr(cfg.d_v == 128):
        v_col = (cg1_tidx // 8 % 2) * 8 + (cg1_tidx // 32 % 2) * 32
        v_segment = (cg1_tidx // 64) * (cfg.b_t * 64)
        state_gmem_row = cg1_tidx
        state_row_valid = cutlass.Boolean(True)
    else:
        v_col = (cg1_tidx // 32) * 16 + (cg1_tidx // 8 % 2) * 8
        v_segment = 0
        state_gmem_row = (cg1_tidx // 32) * 16 + (cg1_tidx % 32) % 16
        state_row_valid = (cg1_tidx % 32) < 16
    v_stage_elements = cfg.v_cosize // cfg.smem_v_stages
    sV_base = sV_trans[0].base
    num_vals = 32

    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads
    )
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    tmem_state_col = tmem_col + cfg.tmem_state_acc_offset
    tmem_state_input_col = tmem_col + cfg.tmem_state_input_offset
    tmem_input_col = tmem_col + cfg.tmem_y_decay_u_input_offset
    tmem_k_state_col = tmem_col + cfg.tmem_cg1_acc_offset
    tmem_u_acc_col = tmem_k_state_col
    tmem_y_input_col = tmem_input_col
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
            batch_num_chunks,
            write_start,
            write_end,
            compute_start,
            _compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        head_o, v_offset = decode_head(cfg, head_idx)
        n_local = write_end - compute_start
        if cutlass.const_expr(cfg.enable_checkpoints):
            checkpoint_chunks = checkpoint_every_n_tokens // cutlass.Int32(cfg.b_t)
            checkpoint_mod = compute_start % checkpoint_chunks
        if n_local > 0:
            if cutlass.const_expr(cfg.seed_checkpoints):
                seed_row = write_start // (seed_every_n_tokens // cutlass.Int32(cfg.b_t))
                seed_b = cutlass.Int32(0)
                while seed_b < batch_idx:
                    seed_len = expanded_cu_seqlen(
                        cfg.expand_num, cu_seqlens, seed_b + 1
                    ) - expanded_cu_seqlen(cfg.expand_num, cu_seqlens, seed_b)
                    seed_row = (
                        seed_row
                        + (seed_len + seed_every_n_tokens - cutlass.Int32(1))
                        // seed_every_n_tokens
                    )
                    seed_b = seed_b + cutlass.Int32(1)
                gSeed = mSeedCheckpoints[None, None, head_o, seed_row]
                checkpoint_vw = 16 // (mSeedCheckpoints.element_type.width // 8)
                checkpoint_src = (
                    gSeed.iterator + gSeed.layout((state_gmem_row + v_offset, 0))
                ).raw_ptr()
                for i in cutlass.range_constexpr(num_ldtms):
                    seed_words = []
                    for checkpoint_group in cutlass.range_constexpr(ldtm_width // checkpoint_vw):
                        checkpoint_chunk = (
                            checkpoint_src + i * ldtm_width + checkpoint_group * checkpoint_vw
                        ).load(count=checkpoint_vw, alignment=16)
                        for checkpoint_elem in cutlass.range_constexpr(checkpoint_vw):
                            seed_words.append(checkpoint_chunk[checkpoint_elem].to(cfg.acc_dtype))
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            row_lo_addr + tmem_state_col + i * ldtm_width, cutlass.Float32
                        ),
                        cutlass.Vector.from_elements(tuple(seed_words), cutlass.Float32),
                    )
                nvvm.tcgen05_wait("store")
            if cutlass.const_expr(cfg.use_initial_state or cfg.seed_identity):
                if cutlass.const_expr(cfg.use_initial_state):
                    gState_init = mState_init[None, None, head_o, batch_idx]
                    seed_vw = 16 // (mState_init.element_type.width // 8)
                    seed_src = (
                        gState_init.iterator + gState_init.layout((state_gmem_row + v_offset, 0))
                    ).raw_ptr()
                seed_state = compute_start == 0
                if seed_state:
                    for i in cutlass.range_constexpr(num_ldtms):
                        words = []
                        if cutlass.const_expr(cfg.seed_identity):
                            for k in cutlass.range_constexpr(32):
                                words.append(
                                    cutlass.Float32(1.0)
                                    if state_gmem_row == i * ldtm_width + k
                                    else cutlass.Float32(0.0)
                                )
                        else:
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

            for local_idx in cutlass.range(n_local):
                chunk_idx = compute_start + local_idx
                if cutlass.const_expr(cfg.enable_checkpoints):
                    do_checkpoint_now = checkpoint_mod == 0
                    checkpoint_mod = checkpoint_mod + cutlass.Int32(1)
                    checkpoint_mod = (
                        cutlass.Int32(0) if checkpoint_mod == checkpoint_chunks else checkpoint_mod
                    )
                if cutlass.const_expr(
                    cfg.enable_checkpoints
                    and not cfg.use_initial_state
                    and not cfg.seed_checkpoints
                ):
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
                valid_state = local_idx > 0
                if cutlass.const_expr(cfg.use_initial_state or cfg.seed_identity):
                    valid_state = local_idx > 0 or seed_state
                if cutlass.const_expr(cfg.seed_checkpoints):
                    valid_state = cutlass.Boolean(True)

                gate_idx = gate_index.idx
                bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
                gate_index = advance(gate_index, cfg.smem_gate_stages)
                cumprod_total = sCumprod[sCumprod.shape[0] - 1, 0, gate_idx]

                # ---- state stage + rescale -------------------------------------------
                if valid_state:
                    if local_idx > 0:
                        bars.mb_state_acc_ready[kv_acc_index.idx].wait(kv_acc_index.phase)
                        kv_acc_index = advance(kv_acc_index, cfg.tmem_state_acc_stages)

                    state_input_stage_idx = state_input_cnt % cfg.tmem_state_input_stages
                    state_vecs = [
                        nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                row_lo_addr + tmem_state_col + i * ldtm_width, cutlass.Float32
                            ),
                            num=32,
                        )
                        for i in range(num_ldtms)
                    ]
                    state_regs = [[state_vecs[i][k] for i in range(num_ldtms)] for k in range(32)]
                    for i in cutlass.range_constexpr(num_ldtms):
                        state_pack = [
                            fp32_to_fp16(
                                state_regs[2 * j][i], state_regs[2 * j + 1][i], dtype=cfg.io_dtype
                            )
                            for j in range(16)
                        ]
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                row_lo_addr + tmem_state_input_col + i * sttm_width, cutlass.Int32
                            ),
                            cutlass.Vector.from_elements(tuple(state_pack), cutlass.Int32),
                        )
                    nvvm.tcgen05_wait("store")
                    bars.mb_state_input_ready[state_input_stage_idx].arrive()
                    state_input_cnt = state_input_cnt + 1

                    if cutlass.const_expr(cfg.enable_checkpoints):
                        # ---- state checkpoint ----------------------------------------
                        do_checkpoint = do_checkpoint_now and chunk_idx < write_end
                        do_checkpoint = do_checkpoint and chunk_idx >= write_start
                        if do_checkpoint:
                            checkpoint_stage = checkpoint_cnt % cfg.smem_checkpoint_stages
                            checkpoint_phase_done = cutlass.Int32(1) ^ (
                                (checkpoint_cnt // cfg.smem_checkpoint_stages) & cutlass.Int32(1)
                            )
                            bars.mb_checkpoint_tmastg_done[checkpoint_stage].wait(
                                checkpoint_phase_done
                            )
                            checkpoint_stage_base = checkpoint_stage * cfg.d_k * cfg.d_v
                            if state_row_valid:
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
                                        (smem_data_ptr(sCheckpoint_raw) + checkpoint_addr).store(
                                            cutlass.Vector.from_elements(
                                                packs, cutlass.Int32
                                            ).bitcast(cfg.io_dtype),
                                            alignment=16,
                                        )
                            nvvm.fence_proxy("async.shared", space="cta")
                            if elect_one:
                                bars.mb_checkpoint_tmastg_ready[checkpoint_stage].arrive()
                            checkpoint_cnt = checkpoint_cnt + 1

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
                                row_lo_addr + tmem_state_col + i * ldtm_width, cutlass.Float32
                            ),
                            cutlass.Vector.from_elements(tuple(state_scaled), cutlass.Float32),
                        )
                    nvvm.tcgen05_wait("store")

                # ---- per-row Gate register builds ------------------------------------
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
                        last_cumsumlog, last_cumsumlog, -cumsumlog_vals[k], -cumsumlog_vals[k + 1]
                    )
                    decay_scale_vals.append(cute.math.exp2(d0, fastmath=True))
                    decay_scale_vals.append(cute.math.exp2(d1, fastmath=True))
                bars.mb_gate_done[gate_idx].arrive()

                # ---- Y = V - k state (packed 16-bit) ---------------------------------
                if cutlass.const_expr(cfg.v_is_zero):
                    zero_word = opaque_i32_zero()
                    v_frag = []
                    for half in cutlass.range_constexpr(dv_halves):
                        v_frag.append([zero_word for _ in range(16)])
                else:
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
                if valid_state:
                    bars.mb_k_state_acc_ready[0].wait(k_state_ready_index.phase)
                    k_state_ready_index = advance(k_state_ready_index, 1)

                    for half in cutlass.range_constexpr(dv_halves):
                        k_state_vec = nvvm.tcgen05_ld(
                            "16x256b",
                            nvvm.make_tmem_ptr(
                                ((tmem_row + half * 16) << 16) + tmem_k_state_col, cutlass.Float32
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
                            k_state_pack = fp32_to_fp16(s0, s1, dtype=cfg.io_dtype)
                            v_frag[half][j] = sub_f16x2(
                                v_frag[half][j], k_state_pack, cfg.io_dtype
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

                # ---- U epilogue + decayed-U publish ----------------------------------
                bars.mb_u_acc_ready[0].wait(u_acc_ready_index.phase)
                u_acc_ready_index = advance(u_acc_ready_index, 1)
                if cutlass.const_expr(not cfg.v_is_zero):
                    bars.mb_v_done[v_idx].arrive()

                u_acc_vecs = []
                for half in cutlass.range_constexpr(dv_halves):
                    u_acc_vecs.append(
                        nvvm.tcgen05_ld(
                            "16x256b",
                            nvvm.make_tmem_ptr(
                                ((tmem_row + half * 16) << 16) + tmem_u_acc_col, cutlass.Float32
                            ),
                            num=8,
                        )
                    )
                u_acc_regs = [[u_acc_vecs[h][k] for h in range(dv_halves)] for k in range(32)]

                for half in cutlass.range_constexpr(dv_halves):
                    for j in cutlass.range_constexpr(16):
                        u_acc_regs[2 * j][half], u_acc_regs[2 * j + 1][half] = fmul2(
                            u_acc_regs[2 * j][half],
                            u_acc_regs[2 * j + 1][half],
                            decay_scale_vals[2 * j],
                            decay_scale_vals[2 * j + 1],
                        )
                    decay_pack = [
                        fp32_to_fp16(
                            u_acc_regs[2 * j][half],
                            u_acc_regs[2 * j + 1][half],
                            dtype=cfg.io_dtype,
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

        # ---- final state store: TMEM -> GMEM -----------------------------------------
        if n_local > 0:
            kv_last_idx = kv_acc_index.idx
            bars.mb_state_acc_ready[kv_last_idx].wait(kv_acc_index.phase)
            kv_acc_index = advance(kv_acc_index, cfg.tmem_state_acc_stages)
            if cutlass.const_expr(cfg.store_final_state):
                if write_end == batch_num_chunks:
                    gState_out = mState_out[None, None, head_o, batch_idx]
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
                write_passthrough = write_end == batch_num_chunks
                if write_passthrough:
                    gState_out = mState_out[None, None, head_o, batch_idx]
                    if cutlass.const_expr(cfg.use_initial_state):
                        gState_in = mState_init[None, None, head_o, batch_idx]
                        for r in cutlass.range(num_ldtms * ldtm_width):
                            if state_row_valid:
                                gState_out[state_gmem_row + v_offset, r] = gState_in[
                                    state_gmem_row + v_offset, r
                                ]
                    elif cutlass.const_expr(cfg.seed_identity):
                        for i in cutlass.range_constexpr(num_ldtms):
                            for k in cutlass.range_constexpr(32):
                                diag = (
                                    cutlass.Float32(1.0)
                                    if state_gmem_row == i * ldtm_width + k
                                    else cutlass.Float32(0.0)
                                )
                                if state_row_valid:
                                    gState_out[state_gmem_row, i * ldtm_width + k] = diag.to(
                                        cfg.state_dtype
                                    )
                    else:
                        for i in cutlass.range_constexpr(num_ldtms):
                            for k in cutlass.range_constexpr(32):
                                if state_row_valid:
                                    gState_out[state_gmem_row + v_offset, i * ldtm_width + k] = (
                                        cutlass.Float32(0.0).to(cfg.state_dtype)
                                    )

        tile_idx, scheduler_state = scheduler_next_tile(
            cfg, bars, sScheduler, scheduler_state, elect_one
        )

    bars.mb_tmem_done[0].arrive()

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
    base_k,
    base_v,
    base_checkpoint,
    base_tinv,
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    state_checkpoints_out: cute.Tensor | None,
    tinv: cute.Tensor | None,
    n_batch: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
) -> None:
    """Per-batch descriptor-array build, one warp per array (the tinv array on the K warp). Runs
    inside the prologue kernel after its order pass; warps past the array count fall through the
    widx guards."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_k_arr = cute.make_tensor(
        desc_workspace.iterator, cute.make_layout((arr_words,), stride=(1,))
    )
    desc_v_arr = cute.make_tensor(
        desc_workspace.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    desc_checkpoint_arr = cute.make_tensor(
        desc_workspace.iterator + 2 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    desc_tinv_arr = cute.make_tensor(
        desc_workspace.iterator + 3 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )

    if widx == 0:
        emit_seq_descs(base_k, desc_k_arr, cu_seqlens, k, n_batch, 2, expand_num, lanes=32)
        if cutlass.const_expr(tinv is not None):
            emit_tile_seq_descs(
                base_tinv, desc_tinv_arr, cu_seqlens, tinv, n_batch, b_t, 3, expand_num, lanes=32
            )
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 1:
        emit_seq_descs(base_v, desc_v_arr, cu_seqlens, v, n_batch, 2, expand_num, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if cutlass.const_expr(state_checkpoints_out is not None):
        if widx == 2:
            emit_checkpoint_seq_descs(
                base_checkpoint,
                desc_checkpoint_arr,
                cu_seqlens,
                state_checkpoints_out,
                n_batch,
                checkpoint_every_n,
                2,
                expand_num,
                lanes=32,
            )
            nvvm.fence_proxy_release(
                nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
            )


@cute.kernel
def frost_gdn_recompute_prologue(
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    gen_intervals: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    tiles_per_head: cutlass.Constexpr[int],
    base_k: cutlass.GridConstant[tma.TensorMap],
    base_v: cutlass.GridConstant[tma.TensorMap],
    base_checkpoint: cutlass.GridConstant[tma.TensorMap],
    base_tinv: cutlass.GridConstant[tma.TensorMap],
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    state_checkpoints_out: cute.Tensor | None,
    tinv: cute.Tensor | None,
    mStaging: cute.Tensor | None,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mScheduler: cute.Tensor | None,
    n_batch: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    seed_span_chunks: cutlass.Int32,
) -> None:
    """Two-CTA prologue. Block 0 owns the item phase: under ``run_order`` this
    kernel is the first work-item-table consumer, so it LPT-orders the table
    and zeroes both consumers' scheduler rings via :func:`order_body`; under
    ``gen_intervals`` it synthesizes one checkpoint-seeded work item per
    ``seed_span_chunks`` chunks (a whole number of checkpoint intervals) of
    every (batch, head) tile.  Block 1 builds the per-batch TMA-descriptor
    arrays via :func:`build_descs_body`, one warp per array."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
        launch_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    if bidx == cutlass.Int32(0):
        if cutlass.const_expr(gen_intervals):
            n_heads_out = cutlass.Int32(gate.shape[1])
            if cutlass.const_expr(tiles_per_head > 1):
                n_heads_out = n_heads_out * cutlass.Int32(tiles_per_head)
            gen_interval_items(
                b_t,
                ORDER_THREADS,
                tidx,
                n_heads_out,
                n_heads_out * n_batch,
                seed_span_chunks,
                cu_seqlens,
                mCount,
                mWorkItems,
                mScheduler,
                expand_num,
            )
        if cutlass.const_expr(run_order):
            sKey = cutlass.Array(
                cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16
            )
            sIdx = cutlass.Array(
                cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16
            )
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
            base_checkpoint,
            base_tinv,
            desc_workspace,
            cu_seqlens,
            k,
            v,
            state_checkpoints_out,
            tinv,
            n_batch,
            checkpoint_every_n,
            b_t,
            expand_num,
        )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    gen_intervals: cutlass.Constexpr[bool],
    expand_num: cutlass.Constexpr[int],
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    cu_seqlens: cute.Tensor,
    state_checkpoints_out: cute.Tensor | None,
    work_item_staging: cute.Tensor | None,
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    scheduler_all: cute.Tensor | None,
    checkpoint_every_n: cutlass.Int32,
    seed_span_chunks: cutlass.Int32,
    tinv: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
    tiles_per_head: cutlass.Constexpr[int] = 1,
):
    """One-launch prologue: LPT-order the work items (with ``run_order``, when
    this kernel is the backward pair's first table consumer) and build the 3
    per-batch TMA-descriptor arrays (K, V, checkpoints) into
    ``tensormap_workspace``."""
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
    k_headed = cute.make_tensor(
        k.iterator, cute.make_layout((seqlen, h_k, d_k), stride=(k_row_stride, k_head_stride, 1))
    )
    v_headed = cute.make_tensor(
        v.iterator, cute.make_layout((d_v, h_v, seqlen), stride=(1, v_head_stride, v_row_stride))
    )
    swizzle_128b = tma.TensorMapSwizzle.s128b
    base_desc_k = tma.create_tensor_map_tiled_from_view(
        k_headed, box_dims=(bt, 1, box_elems), stride_order=(2, 1, 0), swizzle=swizzle_128b
    )
    base_desc_v = tma.create_tensor_map_tiled_from_view(
        v_headed, box_dims=(box_elems, 1, bt), stride_order=(0, 1, 2), swizzle=swizzle_128b
    )

    base_desc_checkpoint = base_desc_v
    if cutlass.const_expr(state_checkpoints_out is not None):
        d_v_state = state_checkpoints_out.shape[2]
        d_k_state = state_checkpoints_out.shape[3]
        checkpoint_box_elems = 128 // (state_checkpoints_out.element_type.width // 8)
        checkpoint_view = cute.make_tensor(
            state_checkpoints_out.iterator,
            cute.make_layout(
                (
                    d_k_state,
                    d_v_state,
                    state_checkpoints_out.shape[0],
                    state_checkpoints_out.shape[1],
                ),
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
    frost_gdn_recompute_prologue(
        run_order,
        order_gen,
        gen_intervals,
        b_t,
        expand_num,
        tiles_per_head,
        base_desc_k,
        base_desc_v,
        base_desc_checkpoint,
        base_desc_tinv,
        tensormap_workspace,
        cu_seqlens,
        k,
        v,
        gate,
        state_checkpoints_out,
        tinv,
        work_item_staging,
        work_count,
        work_items,
        scheduler_all,
        cutlass.Int32(batch_size),
        checkpoint_every_n,
        seed_span_chunks,
    ).launch(grid=(2, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    a_log: cute.Tensor | None,
    dt_bias: cute.Tensor | None,
    cu_seqlens: cute.Tensor,
    state_in: cute.Tensor | None,
    state_out: cute.Tensor | None,
    seed_state_checkpoints: cute.Tensor | None,
    tinv: cute.Tensor,
    work_items: cute.Tensor | None,
    work_count: cute.Tensor | None,
    scheduler_counter: cute.Tensor,
    checkpoint_every_n_tokens: cutlass.Int32,
    seed_every_n_tokens: cutlass.Int32,
    tensormap_workspace: cute.Tensor,
    stream: cuda.CUstream,
):
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = cutlass.Int32(gate.shape[1])
    if cutlass.const_expr(cfg.tiles_per_head > 1):
        heads_out = heads_out * cutlass.Int32(cfg.tiles_per_head)
    k_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(k.shape[1]))
    v_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(v.shape[1]))
    if cutlass.const_expr(cfg.v_is_zero):
        v_ratio = cute.FastDivmodDivisorV2(cutlass.Int32(1))

    # ---- head-major views: (tokens, dim, heads) k / q, (dim, tokens, heads) v / o, (V, K, heads,
    # seqs) states
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
    if cutlass.const_expr(seed_state_checkpoints is not None):
        seed_state_checkpoints = cute.make_tensor(
            seed_state_checkpoints.iterator,
            cute.make_layout(
                (
                    seed_state_checkpoints.shape[2],
                    seed_state_checkpoints.shape[3],
                    seed_state_checkpoints.shape[1],
                    seed_state_checkpoints.shape[0],
                ),
                stride=(
                    seed_state_checkpoints.stride[2],
                    seed_state_checkpoints.stride[3],
                    seed_state_checkpoints.stride[1],
                    seed_state_checkpoints.stride[0],
                ),
            ),
        )

    cumsumlog_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_gate_stages))

    num_descs = batch_size

    @cute.struct
    class SharedStorage:
        t_inv: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.t_inv_cosize], cfg.buffer_align_bytes
        ]
        v: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.v_cosize], cfg.buffer_align_bytes
        ]

    # ---- launch ----------------------------------------------------------------------
    grid_shape = (cfg.max_active_clusters, 1, 1)

    frost_gdn_recompute(
        cfg,
        SharedStorage,
        k_ratio,
        v_ratio,
        gate,
        a_log,
        dt_bias,
        cu_seqlens,
        state_in,
        state_out,
        seed_state_checkpoints,
        tinv,
        work_items,
        work_count,
        scheduler_counter,
        checkpoint_every_n_tokens,
        seed_every_n_tokens,
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
def frost_gdn_recompute(
    cfg: cutlass.Constexpr,
    shared_type: cutlass.Constexpr,
    k_ratio: cute.FastDivmodDivisorV2,
    v_ratio: cute.FastDivmodDivisorV2,
    mGate: cute.Tensor,
    mA_log: cute.Tensor | None,
    mDt_bias: cute.Tensor | None,
    cu_seqlens: cute.Tensor,
    mState_init: cute.Tensor | None,
    mState_out: cute.Tensor | None,
    mSeedCheckpoints: cute.Tensor | None,
    mTinv: cute.Tensor,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor,
    checkpoint_every_n_tokens: cutlass.Int32,
    seed_every_n_tokens: cutlass.Int32,
    cumsumlog_smem_layout_staged: cute.Layout,
    mK,
    mV,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
):
    """Main GDN chunked kernel: warp-specialized dispatch over (batch, head)
    tiles."""
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
    desc_checkpoint_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_tinv_base = desc_base_words + cutlass.Int32(3) * arr_words

    SMEM = cutlass.AddressSpace.smem

    SWZ = 2
    LEAD = 16
    STRIDE = 8 * 128
    KT_LEAD = cfg.b_t * 128
    V_LEAD = cfg.b_t * 128
    if cutlass.const_expr(cfg.enable_checkpoints):
        sCheckpoint_raw = cutlass.Array(
            cfg.io_dtype,
            cfg.checkpoint_cosize,
            space=cutlass.AddressSpace.smem,
            alignment=cfg.buffer_align_bytes,
        )
    else:
        sCheckpoint_raw = None
    sKQ_raw = cutlass.Array(
        cfg.io_dtype,
        cfg.kq_cosize,
        space=cutlass.AddressSpace.smem,
        alignment=cfg.buffer_align_bytes,
    )
    sKQ = SmemTile(
        base=smem_data_ptr(sKQ_raw),
        elems_per_stage=(cfg.kq_cosize // cfg.smem_kq_stages),
        stages=cfg.smem_kq_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sKQ_trans = SmemTile(
        base=smem_data_ptr(sKQ_raw),
        elems_per_stage=(cfg.kq_cosize // cfg.smem_kq_stages),
        stages=cfg.smem_kq_stages,
        leading_byte_offset=2 * KT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    # The checkpoint and K buffers above, the barrier/control rings, and the scalar gate rings
    # stay declaration-ordered Arrays so the SMEM layout matches the vendored kernel; the
    # T_inv and V rings share one storage allocation.
    bars = make_bars(cfg)
    tmem_base_slot = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=16)
    sScheduler = cutlass.Array(cutlass.Int32, cfg.scheduler_stages, space=SMEM, alignment=16)
    cumsumlog_raw = cutlass.Array(
        cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128
    )
    cumprod_raw = cutlass.Array(
        cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged), space=SMEM, alignment=128
    )
    storage = SmemAllocator().allocate(shared_type)
    sTinv_raw = storage.t_inv.get_tensor(cute.make_layout((cfg.t_inv_cosize,)))
    sTinv = SmemTile(
        base=smem_data_ptr(sTinv_raw),
        elems_per_stage=(cfg.t_inv_cosize // cfg.smem_t_inv_stages),
        stages=cfg.smem_t_inv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sV_raw = storage.v.get_tensor(cute.make_layout((cfg.v_cosize,)))
    sV_trans = SmemTile(
        base=smem_data_ptr(sV_raw),
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

    # ---- mbarrier init (one thread; the fence + CTA barrier publish it) ---------------
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
        for s in range(cfg.tmem_state_acc_stages):
            bars.mb_state_acc_ready[s].init()
        bars.mb_k_state_acc_ready[0].init()
        bars.mb_u_acc_ready[0].init()
        for s in range(cfg.smem_t_inv_stages):
            bars.mb_t_inv_ready[s].init()
            bars.mb_t_inv_done[s].init()
        for s in range(cfg.tmem_state_input_stages):
            bars.mb_state_input_ready[s].init()
        for b in (bars.mb_y_input_ready, bars.mb_decay_u_input_ready):
            b[0].init()
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
            sV_trans=sV_trans,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sCheckpoint_raw=sCheckpoint_raw,
            mState_init=mState_init,
            mState_out=mState_out,
            mSeedCheckpoints=mSeedCheckpoints,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            seed_every_n_tokens=seed_every_n_tokens,
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
            sKQ=sKQ,
            sKQ_trans=sKQ_trans,
            sTinv=sTinv,
            sScheduler=sScheduler,
            bars=bars,
        )

    elif warp_idx == cfg.tma_kv_warp_id:
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
            desc_k_base=desc_k_base,
            desc_v_base=desc_v_base,
            mScheduler=mScheduler,
            sScheduler=sScheduler,
            bars=bars,
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
            sCheckpoint_raw=sCheckpoint_raw,
            desc_checkpoint_base=desc_checkpoint_base,
            sScheduler=sScheduler,
            bars=bars,
        )


@dataclass(frozen=True)
class GdnRecomputeCfg:
    """Per-compile GDN kernel knob, built by ``build_cfg``.

    The per-compile parameters (dtypes, GQA, state flags) are the compile
    cache keys; the rest, including the SMEM cosizes and TMA byte counts, is
    derived by ``build_cfg`` from the module-global ``CFG`` constants.  Passed ``cfg``-first (a
    ``cutlass.Constexpr``) into ``host`` / ``kernel`` and every warp body.
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
    seed_checkpoints: bool = False
    log_gate: bool = False
    safe_gate: bool = False
    seed_identity: bool = False
    v_is_zero: bool = False
    tiles_per_head: int = 1
    scheduler_stages: int = CFG.SMEM_SCHEDULER_STAGES

    # ---- fixed constants stamped from CFG at build time ------------------------------
    b_t: int = CFG.B_T
    expand_num: int = 1
    compute_group_1_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    load_gate_warp_id: int = CFG.LOAD_GATE_WARP_ID
    tma_kv_warp_id: int = CFG.TMA_KV_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_other: int = CFG.NUM_REGS_OTHER
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0
    cluster_shape_mnk: tuple[int, int, int] = CFG.CLUSTER_SHAPE_MNK

    # ---- named barrier slots (ids 1-4; 0 is the CTA-wide sync) -----------------------
    tmem_lifecycle_barrier_id: int = 1
    tmem_user_threads: int = 0

    # ---- SMEM / TMEM stage counts + TMEM column offsets ------------------------------
    smem_kq_stages: int = CFG.SMEM_KQ_STAGES
    smem_v_stages: int = CFG.SMEM_V_STAGES
    smem_t_inv_stages: int = CFG.SMEM_T_INV_STAGES
    smem_checkpoint_stages: int = 1
    smem_gate_stages: int = CFG.SMEM_GATE_STAGES
    tmem_state_acc_stages: int = CFG.TMEM_KV_ACC_STAGES
    tmem_state_input_stages: int = CFG.TMEM_STATE_INP_STAGES
    tmem_cg1_acc_stages: int = CFG.TMEM_CG1_ACC_STAGES
    tmem_state_acc_offset: int = 0
    tmem_state_input_offset: int = 0
    tmem_cg1_acc_offset: int = 0
    tmem_y_decay_u_input_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- SMEM cosizes and TMA byte counts, derived by build_cfg ----------------------
    kq_cosize: int = 0
    v_cosize: int = 0
    t_inv_cosize: int = 0
    checkpoint_cosize: int = 0
    tma_kq_bytes: int = 0
    tma_v_bytes: int = 0
    tma_tinv_bytes: int = 0


def build_cfg(
    io_dtype: type[cutlass.Numeric],
    state_dtype: type[cutlass.Numeric],
    *,
    max_active_clusters: int,
    use_initial_state: bool,
    store_final_state: bool = True,
    enable_checkpoints: bool = False,
    seed_checkpoints: bool = False,
    log_gate: bool = False,
    safe_gate: bool = False,
    seed_identity: bool = False,
    v_is_zero: bool = False,
    d_k: int,
    d_v: int,
    expand_num: int = 1,
    tiles_per_head: int = 1,
) -> GdnRecomputeCfg:
    """Build the per-compile ``GdnRecomputeCfg`` (io_dtype in {Float16, BFloat16}; acc is
    always Float32).  ``tiles_per_head`` > 1 runs every gate head as ``tiles_per_head`` tiles of
    ``d_v`` value columns each."""
    if (
        io_dtype not in (cutlass.Float16, cutlass.BFloat16)
        or d_k not in STATE_DIMS
        or d_v not in STATE_DIMS
    ):
        raise ValueError(
            f"recompute serves Float16/BFloat16 with d_k, d_v in {STATE_DIMS}, got {io_dtype}, "
            f"{d_k}, {d_v}"
        )
    if max_active_clusters <= 0 or expand_num < 1 or tiles_per_head < 1:
        raise ValueError("max_active_clusters, expand_num and tiles_per_head must be positive")
    cfg = GdnRecomputeCfg(
        io_dtype=io_dtype,
        acc_dtype=cutlass.Float32,
        state_dtype=state_dtype,
        max_active_clusters=max_active_clusters,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        seed_checkpoints=seed_checkpoints,
        log_gate=log_gate,
        safe_gate=safe_gate,
        seed_identity=seed_identity,
        v_is_zero=v_is_zero,
        d_k=d_k,
        d_v=d_v,
        expand_num=expand_num,
        tiles_per_head=tiles_per_head,
    )
    smem_kq_stages = 3 if enable_checkpoints else cfg.smem_kq_stages
    num_regs_compute_group_1 = cfg.num_regs_compute_group_1
    num_regs_other = cfg.num_regs_other
    if not (use_initial_state or seed_identity):
        num_regs_compute_group_1 = 232
        num_regs_other = 48
    smem_checkpoint_stages = 1
    n_cg1 = len(cfg.compute_group_1_warp_ids)
    warp = cfg.threads_per_warp
    # Attention Gym modification: the warp count derives from the validated role table.
    n_warps = validate_warp_roles(
        (cfg.compute_group_1_warp_ids,),
        (cfg.load_gate_warp_id, cfg.tma_kv_warp_id, cfg.tcgen05_mma_warp_id, cfg.epilogue_warp_id),
    )
    tmem_state_input_offset = cfg.tmem_state_acc_stages * cfg.d_k
    tmem_cg1_acc_offset = tmem_state_input_offset + cfg.tmem_state_input_stages * (cfg.d_k // 2)
    bytes_per_element = io_dtype.width // 8
    kq_tile = 2 * cfg.b_t * cfg.d_k
    v_tile = cfg.d_v * cfg.b_t
    tinv_tile = cfg.b_t * cfg.b_t
    cfg = replace(
        cfg,
        smem_checkpoint_stages=smem_checkpoint_stages,
        smem_kq_stages=smem_kq_stages,
        num_regs_compute_group_1=num_regs_compute_group_1,
        num_regs_other=num_regs_other,
        threads_per_cta=warp * n_warps,
        tmem_user_threads=warp * (1 + n_cg1),
        tmem_state_acc_offset=0,
        tmem_state_input_offset=tmem_state_input_offset,
        tmem_cg1_acc_offset=tmem_cg1_acc_offset,
        tmem_y_decay_u_input_offset=tmem_cg1_acc_offset + cfg.tmem_cg1_acc_stages * cfg.b_t,
        kq_cosize=kq_tile * smem_kq_stages,
        v_cosize=v_tile * cfg.smem_v_stages,
        t_inv_cosize=tinv_tile * cfg.smem_t_inv_stages,
        checkpoint_cosize=cfg.d_k * cfg.d_v * smem_checkpoint_stages,
        tma_kq_bytes=(kq_tile // 2) * bytes_per_element,
        tma_v_bytes=v_tile * bytes_per_element,
        tma_tinv_bytes=tinv_tile * bytes_per_element,
    )
    validate_named_barriers(
        cfg.threads_per_cta, tmem_lifecycle=(cfg.tmem_lifecycle_barrier_id, cfg.tmem_user_threads)
    )
    if cfg.tmem_y_decay_u_input_offset + cfg.b_t > TMEM_COLUMNS:
        raise ValueError(
            f"recompute TMEM layout needs {cfg.tmem_y_decay_u_input_offset + cfg.b_t} > "
            f"{TMEM_COLUMNS} columns"
        )
    return cfg


TENSORMAP_DESC_ARRAYS = 4  # per-batch runtime TMA descriptors: K, V, checkpoints, tinv


frost_gdn_recompute_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_gdn_recompute.set_name_prefix("cudnn", remove_cutlass_symbol=False)
