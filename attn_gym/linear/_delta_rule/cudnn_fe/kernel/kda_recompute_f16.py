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
# Register arrays are cute rmem tensors, SMEM data buffers live in a SharedStorage struct, and raw
# SMEM pointers go through smem_data_ptr (S1-S3). It compiles through persisted jit_cache
# fake-tensor TVM-FFI signatures and launches on the current Torch stream.

"""
Chunked Kimi Delta Attention (KDA) recompute (state/checkpoint-only) kernel for SM100 / SM103 /
SM107 (Cutlass primitives): the BT = 16 schedule with a per-key-channel decay, the recurrent state
and its checkpoint series, with no Q or O path.

Algorithm overview (per chunk c, tokens [cC, (c+1)C)):
  Inputs : K[BT,DK], V[BT,DV], Gate[BT,DK] (per-channel gate), Beta[BT] (scalar LR)
  State  : S_prev[DK,DV]  (recurrent state, held in TMEM, fp32 carry; seeded from zero, initial_state, the identity
           (chain M) or a coarse checkpoint row (series recompute))

  Preprocessing (compute group 0, two ping-pong groups of four warps):
    g[t,d]           = sum_{l=0}^{t} log2(Gate_ld)             per-channel cumulative log2 of gates (safe-gate / log)
    K decay[t,d]     = K[t,d] * exp2(+g[t,d])                    (KK A operand, K*state B operand)
    K inv[t,d]       = K[t,d] * exp2(-g[t,d])                    (KK / A tile B operand)
    K restore[t,d]   = K[t,d] * exp2(g[BT-1,d] - g[t,d])         (state update B operand)
    (optional in-kernel Q/K L2-norm folds 1/|q|, 1/|k| into the operands)

  KK (register MMA) : W_kk[BT,BT] = K decay @ K inv^T;  L = Beta * tril(W_kk, -1)
  T_inv (register MMA) : T_inv = (I + L)^-1 blockwise, 4x4 diagonal blocks then the 4 -> 8 and 8 ->
                         16 corrections
  K*state GEMM   : KS[BT,DV] = K decay @ S_prev    (key applied to state)
  U GEMM         : U[BT,DV]  = T_inv @ Y,  Y = Beta * (V - KS)
  KV update GEMM : S_upd[DK,DV] = K restore^T @ U   (state update, BT contraction, left then right key half)

  Epilogue:
    S_next    = exp2(g[BT-1,:]) .* S_prev + S_upd      (per-channel decay of the state in TMEM, then the update)
    checkpoint rows every checkpoint_every_n_tokens; the final state at write_end ==
    batch_num_chunks

SMEM layout (stage counts live in kda_recompute_config.py; enable_checkpoints compiles trim the raw
stages to 6 and add the checkpoint buffer; sizes at DK = DV = 128, bf16 io, fp32 Gate):
  Buffer                       Size (B)  Stages
  K / V (raw)                  2 x 4096       8
  Gate (raw)                       8192       8    <-- bf16 Gate: 4096 plus a 4-stage fp32 exchange ring
  Beta                               64       8
  K decay / K inv / K restore  3 x 4096       2
  T_inv (intermediate)             1024       2
  checkpoint staging              32768       2    <-- enable_checkpoints only (aliases V otherwise)
  scheduler ticket ring               4       8    <-- next-tile publish ring

TMEM layout (512 columns allocated):
  Buffer                  Cols
  state                   128     <-- DKxDV fp32 (doubles as the final state acc)
  state input              64     <-- f16 state staging (K*state A operand)
  K*state acc              16
  U acc                    16
  Y input                   8     <-- f16 packed
  U input                   8

Warp assignments (16 warps = 512 threads):
  warps 0-7     : compute group 0 - Gate prefix scan, decay/restore operands, left state halves (two ping-pong groups)
  warps 8-11    : compute group 1 - state seed, right state halves, Y / U staging, checkpoint rows, final state store
  warp  12      : register-MMA warp - KK and the blockwise T_inv
  warp  13      : MMA warp       - every tcgen05 GEMM; TMEM lifecycle
  warp  14      : TMA load warp  - loads K, V, Gate; stages Beta
  warp  15      : epilogue warp  - checkpoint TMA stores
"""

from dataclasses import dataclass, replace
from typing import NamedTuple

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.experimental.primitives as nvvm
from cutlass import cute
from cutlass.experimental import cuda

from attn_gym._backends.cute.compat import SmemAllocator

from ..common.blockwise_inverse import invert_unit_lower_16x16_fragments
from ..common.split_k import (
    ORDER_CAPACITY,
    ORDER_ELEMENTS,
    ORDER_THREADS,
    decode_work_item,
    expanded_cu_seqlen,
    gen_interval_items,
    order_body,
)
from ..common.thd import TENSOR_MAP_QWORDS, emit_checkpoint_seq_descs, emit_seq_descs
from ..tile_dsl.barrier import (
    MBarrier,
    PipelineState,
    Producer,
    advance,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
from ..tile_dsl.handles import MmaDesc, SmemTile, smem_data_ptr, tma_slice_runtime_desc
from ..tile_dsl.mma import desc_opaque, mma_step, mma_ts_step
from ..tile_dsl.pointwise import (
    beta_residual_f16x2,
    fadd2,
    ffma2,
    fmul2,
    fp32_to_fp16,
    opaque_f32_zero,
    opaque_i32,
    opaque_i32_zero,
    sigmoid,
)
from ..tile_dsl.swizzle import swizzle_xor_32b, swizzle_xor_128b
from ..tile_dsl.tma import (
    tma_load_tile,
    tma_store_commit,
    tma_store_tile,
    tma_store_wait,
    tma_tensormap_acquire,
)
from .kda_bprop_f16 import _validate_roles
from .kda_recompute_config import CFG

USE_PDL = True

LOG2_E: float = 1.4426950408889634
DEFAULT_GATE_LOWER_BOUND: float = -5.0
L2_NORM_EPS: float = 1.0e-12


class KdaRecomputeBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_raw_ready: MBarrier
    mb_raw_done: MBarrier
    mb_gate_exchange_ready: MBarrier

    mb_beta_ready: MBarrier
    mb_beta_done: MBarrier

    mb_state_k_acc_ready: MBarrier
    mb_u_acc_ready: MBarrier

    mb_state_input_cg1_ready: MBarrier
    mb_state_input_cg0_ready: MBarrier
    mb_y_input_ready: MBarrier
    mb_u_input_ready: MBarrier

    mb_t_inv_ready: MBarrier
    mb_t_inv_done: MBarrier
    mb_qk_scale_ready: MBarrier
    mb_k_decay_inv_cg0_ready: MBarrier
    mb_decay_tcgen05_done: MBarrier
    mb_decay_register_mma_done: MBarrier
    mb_k_restore_done: MBarrier

    mb_state_acc_cg0_done: MBarrier
    mb_state_acc_cg1_done: MBarrier
    mb_tmem_done: MBarrier

    mb_checkpoint_tmastg_ready: MBarrier
    mb_checkpoint_tmastg_done: MBarrier

    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier


def make_bars(cfg) -> KdaRecomputeBars:
    """KdaRecomputeBars constructor."""

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=8)

    CG0_GROUP_WARPS = cfg.cg0_warps_per_group
    CG1_WARPS = len(cfg.compute_group_1_warp_ids)

    return KdaRecomputeBars(
        mb_raw_ready=MBarrier(
            alloc(cfg.smem_raw_stages),
            try_wait=True,
            stages=cfg.smem_raw_stages,
            init_count=1,
            producer=Producer.TMA_LOAD,
        ),
        mb_raw_done=MBarrier(
            alloc(cfg.smem_raw_stages),
            try_wait=True,
            stages=cfg.smem_raw_stages,
            init_count=CG0_GROUP_WARPS + CG1_WARPS,
            producer=Producer.THREAD,
        ),
        mb_gate_exchange_ready=MBarrier(
            alloc(cfg.smem_raw_stages),
            try_wait=True,
            stages=cfg.smem_raw_stages,
            init_count=CG0_GROUP_WARPS,
            producer=Producer.THREAD,
        ),
        mb_beta_ready=MBarrier(
            alloc(cfg.smem_raw_stages),
            try_wait=True,
            stages=cfg.smem_raw_stages,
            init_count=1,
            producer=Producer.THREAD,
        ),
        mb_beta_done=MBarrier(
            alloc(cfg.smem_raw_stages),
            try_wait=True,
            stages=cfg.smem_raw_stages,
            init_count=1 + CG1_WARPS,
            producer=Producer.THREAD,
        ),
        mb_state_k_acc_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=1, producer=Producer.MMA_COMMIT
        ),
        mb_u_acc_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=1, producer=Producer.MMA_COMMIT
        ),
        mb_state_input_cg1_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1_WARPS, producer=Producer.THREAD
        ),
        mb_state_input_cg0_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG0_GROUP_WARPS, producer=Producer.THREAD
        ),
        mb_y_input_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1_WARPS, producer=Producer.THREAD
        ),
        mb_u_input_ready=MBarrier(
            alloc(1),
            try_wait=True,
            stages=1,
            init_count=CG1_WARPS + CG0_GROUP_WARPS,
            producer=Producer.THREAD,
        ),
        mb_t_inv_ready=MBarrier(
            alloc(cfg.smem_intermediate_stages),
            try_wait=True,
            stages=cfg.smem_intermediate_stages,
            init_count=1,
            producer=Producer.THREAD,
        ),
        mb_t_inv_done=MBarrier(
            alloc(cfg.smem_intermediate_stages),
            try_wait=True,
            stages=cfg.smem_intermediate_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
        ),
        mb_qk_scale_ready=MBarrier(
            alloc(cfg.qk_scale_ready_stages),
            try_wait=True,
            stages=cfg.qk_scale_ready_stages,
            init_count=CG0_GROUP_WARPS,
            producer=Producer.THREAD,
        ),
        mb_k_decay_inv_cg0_ready=MBarrier(
            alloc(cfg.smem_decay_stages),
            try_wait=True,
            stages=cfg.smem_decay_stages,
            init_count=CG0_GROUP_WARPS,
            producer=Producer.THREAD,
        ),
        mb_decay_tcgen05_done=MBarrier(
            alloc(cfg.smem_decay_stages),
            try_wait=True,
            stages=cfg.smem_decay_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
        ),
        mb_decay_register_mma_done=MBarrier(
            alloc(cfg.smem_decay_stages),
            try_wait=True,
            stages=cfg.smem_decay_stages,
            init_count=1,
            producer=Producer.THREAD,
        ),
        mb_k_restore_done=MBarrier(
            alloc(cfg.smem_decay_stages),
            try_wait=True,
            stages=cfg.smem_decay_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
        ),
        mb_state_acc_cg0_done=MBarrier(
            alloc(cfg.smem_decay_stages),
            try_wait=True,
            stages=cfg.smem_decay_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
        ),
        mb_state_acc_cg1_done=MBarrier(
            alloc(cfg.smem_decay_stages),
            try_wait=True,
            stages=cfg.smem_decay_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
        ),
        mb_tmem_done=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1_WARPS, producer=Producer.THREAD
        ),
        mb_checkpoint_tmastg_ready=MBarrier(
            alloc(cfg.smem_checkpoint_stages),
            try_wait=True,
            stages=cfg.smem_checkpoint_stages,
            init_count=CG0_GROUP_WARPS + CG1_WARPS,
            producer=Producer.THREAD,
        ),
        mb_checkpoint_tmastg_done=MBarrier(
            alloc(cfg.smem_checkpoint_stages),
            try_wait=True,
            stages=cfg.smem_checkpoint_stages,
            init_count=1,
            producer=Producer.THREAD,
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
            init_count=cfg.scheduler_consumer_warps,
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
def epilogue_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    sCheckpoint_raw,
    desc_checkpoint_base,
    checkpoint_every_n_tokens,
    bars,
) -> None:
    """Epilogue warp role (warp 15): persistent scheduler loop issuing the
    per-chunk checkpoint TMA stores."""
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
        if cutlass.const_expr(cfg.enable_checkpoints):
            head_o = head_idx
            checkpoint_slot = batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
            desc_checkpoint_slot = (desc_checkpoint_base + checkpoint_slot).tospace(
                cutlass.AddressSpace.generic
            )
            checkpoint_chunks = checkpoint_every_n_tokens // cutlass.Int32(cfg.b_t)
            checkpoint_quotient = (compute_start + cutlass.Int32(1)) // checkpoint_chunks
            checkpoint_remainder = (compute_start + cutlass.Int32(1)) % checkpoint_chunks
            if elect_one:
                tma_tensormap_acquire(desc_checkpoint_slot)
            num_chunks_tile = write_end - compute_start
            first_row_store = num_chunks_tile > 0 and write_start == 0
            if cutlass.const_expr(cfg.seed_checkpoints):
                first_row_store = num_chunks_tile > 0
            if first_row_store:
                checkpoint_stage = checkpoint_ready_index.idx
                bars.mb_checkpoint_tmastg_ready[checkpoint_stage].wait(
                    checkpoint_ready_index.phase
                )
                checkpoint_ready_index = advance(
                    checkpoint_ready_index, cfg.smem_checkpoint_stages
                )
                checkpoint_slice = tma_slice_runtime_desc(
                    desc_checkpoint_slot, cutlass.Int32(0), cutlass.Int32(0), write_start, head_o
                )
                tma_store_tile(sCheckpoint_tma[checkpoint_stage], checkpoint_slice)
                tma_store_commit()
                tma_store_wait(0)
                if nvvm.elect_sync():
                    bars.mb_checkpoint_tmastg_done[checkpoint_stage].arrive()
            for local_chunk_idx in cutlass.range(num_chunks_tile, unroll=1):
                chunk_idx = compute_start + local_chunk_idx
                if local_chunk_idx > 0:
                    # ---- checkpoint store --------------------------------------------
                    do_checkpoint = checkpoint_remainder == 0
                    do_checkpoint = do_checkpoint and chunk_idx >= write_start
                    if do_checkpoint:
                        checkpoint_stage = checkpoint_ready_index.idx
                        bars.mb_checkpoint_tmastg_ready[checkpoint_stage].wait(
                            checkpoint_ready_index.phase
                        )
                        checkpoint_ready_index = advance(
                            checkpoint_ready_index, cfg.smem_checkpoint_stages
                        )
                        checkpoint_entry = checkpoint_quotient
                        checkpoint_slice = tma_slice_runtime_desc(
                            desc_checkpoint_slot,
                            cutlass.Int32(0),
                            cutlass.Int32(0),
                            checkpoint_entry,
                            head_o,
                        )
                        tma_store_tile(sCheckpoint_tma[checkpoint_stage], checkpoint_slice)
                        tma_store_commit()
                        tma_store_wait(0)
                        if nvvm.elect_sync():
                            bars.mb_checkpoint_tmastg_done[checkpoint_stage].arrive()
                    checkpoint_remainder = checkpoint_remainder + cutlass.Int32(1)
                    if checkpoint_remainder == checkpoint_chunks:
                        checkpoint_remainder = cutlass.Int32(0)
                        checkpoint_quotient = checkpoint_quotient + cutlass.Int32(1)
        tile_idx, scheduler_state = scheduler_next_tile(
            cfg, bars, sScheduler, scheduler_state, elect_one
        )


@cute.jit
def register_mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    lane_idx,
    sK_inv_raw,
    sIntermediate_raw,
    sBeta_raw,
    sK_decay_raw,
    bars,
) -> None:
    """Register-MMA warp role (warp 12): persistent scheduler loop computing the
    register-MMA blockwise T_inv."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    elect_one = nvvm.elect_sync()

    # ---- ldmatrix/stmatrix lane decode -----------------------------------------------
    k_inv_row_coord = lane_idx % 8 + (cutlass.Int32(8) if (lane_idx // 16) else cutlass.Int32(0))
    k_inv_col_offset = cutlass.Int32(8) if ((lane_idx // 8) % 2) else cutlass.Int32(0)
    k_decay_row_coord = lane_idx % 8 + (
        cutlass.Int32(8) if ((lane_idx // 8) % 2) else cutlass.Int32(0)
    )
    k_decay_col_offset = cutlass.Int32(8) if ((lane_idx // 8) // 2) else cutlass.Int32(0)
    t_inv_row_coord = lane_idx & 7
    t_inv_col_coord = cutlass.Int32(0)
    if (lane_idx // 8) & 1:
        t_inv_row_coord = t_inv_row_coord + cutlass.Int32(8)
    if lane_idx // 8 >= 2:
        t_inv_col_coord = cutlass.Int32(8)
    t_inv_idx = t_inv_row_coord * cfg.b_t + swizzle_xor_32b(t_inv_row_coord, t_inv_col_coord)
    k_inv_frag_offsets = [
        opaque_i32(
            k_inv_row_coord * 64
            + swizzle_xor_128b(k_inv_row_coord, i * 16 + k_inv_col_offset, elem_bytes=2)
        )
        for i in range(4)
    ]
    k_decay_frag_offsets = [
        opaque_i32(
            swizzle_xor_128b(
                k_decay_row_coord,
                k_decay_row_coord * 64 + i * 16 + k_decay_col_offset,
                elem_bytes=2,
            )
        )
        for i in range(4)
    ]
    cum_chunk_base = cutlass.Int32(0)
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
        num_chunks_tile = write_end - compute_start
        for local_chunk_idx in cutlass.range(num_chunks_tile, unroll=1):
            cum_chunk = cum_chunk_base + local_chunk_idx
            chunk_count = cutlass.Uint32(cum_chunk)
            decay_stage = cutlass.Int32(chunk_count % cfg.smem_decay_stages)
            decay_parity = cutlass.Int32((chunk_count // cfg.smem_decay_stages) % 2)
            intermediate_stage = cutlass.Int32(chunk_count % cfg.smem_intermediate_stages)
            intermediate_free_parity = cutlass.Int32(
                ((chunk_count // cfg.smem_intermediate_stages) + 1) % 2
            )
            raw_stage = cutlass.Int32(chunk_count % cfg.smem_raw_stages)
            raw_parity = cutlass.Int32((chunk_count // cfg.smem_raw_stages) % 2)
            sBeta_ptr = smem_data_ptr(sBeta_raw) + raw_stage * cfg.b_t
            sK_inv_ptr = smem_data_ptr(sK_inv_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = smem_data_ptr(sK_decay_raw) + decay_stage * (cfg.d_k * cfg.b_t)
            sIntermediate_ptr = smem_data_ptr(sIntermediate_raw) + intermediate_stage * (
                2 * cfg.b_t * cfg.b_t
            )

            bars.mb_k_decay_inv_cg0_ready[decay_stage].wait(decay_parity)

            # ---- KK = K decay @ K inv^T ----------------------------------------------
            kk_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                kk_acc[accum_idx] = cutlass.Float32(0.0)

            for i in cutlass.range_constexpr(cfg.d_k // 16):
                k_inv_frag = nvvm.ldmatrix(
                    sK_inv_ptr + k_inv_frag_offsets[i % 4] + (i // 4) * (cfg.b_t * 64),
                    4,
                    nvvm.MMALayout.ROW,
                )
                k_decay_frag = nvvm.ldmatrix(
                    sK_decay_ptr + k_decay_frag_offsets[i % 4] + (i // 4) * (cfg.b_t * 64),
                    4,
                    nvvm.MMALayout.ROW,
                )

                mma_step(
                    kk_acc,
                    (k_decay_frag[0], k_decay_frag[1], k_decay_frag[2], k_decay_frag[3]),
                    (k_inv_frag[0], k_inv_frag[1], k_inv_frag[2], k_inv_frag[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )

            # ---- L = Beta * tril(KK, -1) ---------------------------------------------
            bars.mb_beta_ready[raw_stage].wait(raw_parity)
            row_lo = lane_idx // 4
            row_hi = row_lo + cutlass.Int32(8)
            beta_lo = (sBeta_ptr + row_lo).load().to(cutlass.Float32)
            beta_hi = (sBeta_ptr + row_hi).load().to(cutlass.Float32)
            l_regs = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                row_coord = row_hi if cutlass.const_expr(accum_idx % 4 >= 2) else row_lo
                col_coord = (accum_idx // 4) * 8 + 2 * (lane_idx % 4)
                if cutlass.const_expr(accum_idx % 2 == 1):
                    col_coord = col_coord + cutlass.Int32(1)
                l_regs[accum_idx] = (
                    kk_acc[accum_idx] if row_coord > col_coord else cutlass.Float32(0.0)
                )
            for pair in cutlass.range_constexpr(4):
                beta_scale = beta_hi if cutlass.const_expr(pair % 2 == 1) else beta_lo
                l_regs[2 * pair], l_regs[2 * pair + 1] = fmul2(
                    l_regs[2 * pair], l_regs[2 * pair + 1], beta_scale, beta_scale
                )
            if nvvm.elect_sync():
                bars.mb_beta_done[raw_stage].arrive()

            # ---- T_inv = (I + L)^-1 --------------------------------------------------
            tinv_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            invert_unit_lower_16x16_fragments(cfg, l_regs, tinv_acc, lane_idx)

            bars.mb_t_inv_done[intermediate_stage].wait(intermediate_free_parity)
            nvvm.stmatrix(
                sIntermediate_ptr + (cfg.b_t * cfg.b_t) + t_inv_idx,
                [
                    fp32_to_fp16(tinv_acc[0], tinv_acc[1], dtype=cfg.io_dtype),
                    fp32_to_fp16(tinv_acc[2], tinv_acc[3], dtype=cfg.io_dtype),
                    fp32_to_fp16(tinv_acc[4], tinv_acc[5], dtype=cfg.io_dtype),
                    fp32_to_fp16(tinv_acc[6], tinv_acc[7], dtype=cfg.io_dtype),
                ],
                nvvm.MMALayout.ROW,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.fence_proxy("async.shared", space="cta")
            if nvvm.elect_sync():
                bars.mb_t_inv_ready[intermediate_stage].arrive()
                bars.mb_decay_register_mma_done[decay_stage].arrive()
        cum_chunk_base += num_chunks_tile
        tile_idx, scheduler_state = scheduler_next_tile(
            cfg, bars, sScheduler, scheduler_state, elect_one
        )


@cute.jit
def tcgen05_mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sScheduler,
    sTmem_base,
    sIntermediate,
    sK_decay,
    sK_restore_trans,
    bars,
) -> None:
    """tcgen05-MMA warp role (warp 13): persistent scheduler loop issuing
    every state GEMM and owning the TMEM lifecycle."""
    elect_one = nvvm.elect_sync()
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    nvvm.tcgen05_alloc(sTmem_base, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = sTmem_base.load()
    state_input_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_input_offset, cutlass.Int8)
    state_k_acc_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_k_acc_offset, cutlass.Float32)
    u_acc_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_u_acc_offset, cutlass.Float32)
    y_input_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_y_input_offset, cutlass.Int8)
    u_input_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_u_input_offset, cutlass.Int8)
    state_dst_cg0_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_acc_offset, cutlass.Float32)
    state_update_n = cutlass.const_expr(cfg.d_k // 2 if cfg.d_k // 2 >= 64 else cfg.d_k)
    state_update_split = cutlass.const_expr(state_update_n < cfg.d_k)
    k_restore_right_bytes = cutlass.const_expr(cfg.b_t * 128)
    state_dst_cg1_ptr = nvvm.make_tmem_ptr(
        tmem_base + cfg.tmem_state_acc_offset + state_update_n, cutlass.Float32
    )
    state_input_cg1_index = PipelineState.start(phase=0)
    state_input_cg0_index = PipelineState.start(phase=0)
    y_input_index = PipelineState.start(phase=0)
    u_input_index = PipelineState.start(phase=0)
    qk_scale_index = PipelineState.start(phase=0)
    k_decay_ready = PipelineState.start(phase=0)
    t_inv_ready = PipelineState.start(phase=0)

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
    bmm_y_t_inv_desc = MmaDesc(
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
        num_chunks_tile = write_end - compute_start
        if cutlass.const_expr(cfg.use_initial_state or cfg.seed_identity):
            seed_state = compute_start == 0
        for local_chunk_idx in cutlass.range(num_chunks_tile, unroll=1):
            cum_chunk_base + local_chunk_idx
            if cutlass.const_expr(cfg.seed_checkpoints):
                have_state = cutlass.Boolean(True)
            elif cutlass.const_expr(cfg.use_initial_state or cfg.seed_identity):
                have_state = local_chunk_idx > 0 or seed_state
            else:
                have_state = local_chunk_idx > 0
            decay_stage = k_decay_ready.idx
            intermediate_stage = t_inv_ready.idx
            sK_decay_stage = sK_decay[decay_stage]
            sK_restore_stage = sK_restore_trans[decay_stage]
            sIntermediate_stage = sIntermediate[intermediate_stage]
            desc_k_decay = desc_opaque(sK_decay_stage.desc())
            desc_k_restore = desc_opaque(sK_restore_stage.desc())
            if cutlass.const_expr(state_update_split):
                desc_k_restore_right = desc_k_restore.advance_start_address(k_restore_right_bytes)
            desc_t_inv = desc_opaque(sIntermediate_stage.shifted(cfg.b_t * cfg.b_t).desc())

            # ---- k state = state(T) @ K decay^T --------------------------------------
            bars.mb_k_decay_inv_cg0_ready[decay_stage].wait(k_decay_ready.phase)
            k_decay_ready = advance(k_decay_ready, cfg.smem_decay_stages)
            if have_state:
                bars.mb_state_input_cg0_ready.wait(state_input_cg0_index.phase)
                state_input_cg0_index = advance(state_input_cg0_index, 1)

                for f in cutlass.range_constexpr(bmm_state_k_decay_desc.num_k_steps):
                    if cutlass.const_expr(f == STATE_K_STEPS_CG0):
                        bars.mb_state_input_cg1_ready.wait(state_input_cg1_index.phase)
                        state_input_cg1_index = advance(state_input_cg1_index, 1)
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

            if elect_one:
                bars.mb_decay_tcgen05_done[decay_stage].arrive(cta_group=1)

            bars.mb_qk_scale_ready[qk_scale_index.idx].wait(qk_scale_index.phase)

            # ---- U = Y(T) @ T^-1 -----------------------------------------------------
            bars.mb_t_inv_ready[intermediate_stage].wait(t_inv_ready.phase)
            bars.mb_y_input_ready.wait(y_input_index.phase)
            y_input_index = advance(y_input_index, 1)
            mma_ts_step(
                bmm_y_t_inv_desc,
                y_input_ptr,
                desc_t_inv,
                u_acc_ptr,
                0,
                cutlass.Boolean(False),
                issue_mma=elect_one,
            )
            if elect_one:
                bars.mb_t_inv_done[intermediate_stage].arrive(cta_group=1)
                bars.mb_u_acc_ready.arrive(cta_group=1)

            # ---- final state += U(T) @ K restore, left then right key half -----------
            bars.mb_u_input_ready.wait(u_input_index.phase)
            u_input_index = advance(u_input_index, 1)
            mma_ts_step(
                bmm_u_k_restore_desc,
                u_input_ptr,
                desc_k_restore,
                state_dst_cg0_ptr,
                0,
                have_state,
                issue_mma=elect_one,
            )
            if elect_one:
                bars.mb_state_acc_cg0_done[decay_stage].arrive(cta_group=1)
            if cutlass.const_expr(state_update_split):
                mma_ts_step(
                    bmm_u_k_restore_desc,
                    u_input_ptr,
                    desc_k_restore_right,
                    state_dst_cg1_ptr,
                    0,
                    have_state,
                    issue_mma=elect_one,
                )
            if elect_one:
                bars.mb_k_restore_done[decay_stage].arrive(cta_group=1)
                bars.mb_state_acc_cg1_done[decay_stage].arrive(cta_group=1)

            t_inv_ready = advance(t_inv_ready, cfg.smem_intermediate_stages)
            qk_scale_index = advance(qk_scale_index, cfg.qk_scale_ready_stages)

        cum_chunk_base += num_chunks_tile
        tile_idx, scheduler_state = scheduler_next_tile(
            cfg, bars, sScheduler, scheduler_state, elect_one
        )
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
    sK_raw,
    sV_raw,
    sGate_raw,
    desc_k_base,
    desc_v_base,
    desc_gate_base,
    bars,
    k_ratio,
    v_ratio,
) -> None:
    """TMA-LDG warp role (warp 14): persistent scheduler loop issuing the
    per-chunk K/V/Gate G->S loads."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    raw_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=1)

    elect_one = nvvm.elect_sync()
    sK_tma = SmemTile(
        base=sK_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
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
    gate_box_elems = cutlass.const_expr(128 // (cfg.gate_dtype.width // 8))
    sGate_tma = SmemTile(
        base=sGate_raw,
        elems_per_stage=(cfg.gate_cosize // cfg.smem_raw_stages),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // gate_box_elems),
        tma_granu_elems=gate_box_elems,
        tma_subtile_stride_elems=(cfg.b_t * 32),
    )
    tile_idx = cutlass.Int32(bidx)
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
        head_o = head_idx
        head_k = head_idx // k_ratio
        head_v = head_idx // v_ratio
        slot = batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_gate_slot = (desc_gate_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_k_slot)
            if cutlass.const_expr(not cfg.v_is_zero):
                tma_tensormap_acquire(desc_v_slot)
            tma_tensormap_acquire(desc_gate_slot)
        for chunk_idx in cutlass.range(compute_start, write_end, 1, unroll=1):
            chunk_start = chunk_idx * cfg.b_t

            # ---- K / Gate / V loads: one transaction barrier per stage ---------------
            bars.mb_raw_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_raw_ready[raw_index.idx].arrive(
                    n_bytes=cfg.tma_k_bytes
                    + cfg.tma_gate_bytes
                    + (0 if cfg.v_is_zero else cfg.tma_v_bytes)
                )
            raw_ready_ptr = bars.mb_raw_ready[raw_index.idx].smem_ptr
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), head_k, chunk_start)
            tma_load_tile(sK_tma[raw_index.idx], k_slice, raw_ready_ptr)
            gate_slice = tma_slice_runtime_desc(
                desc_gate_slot, cutlass.Int32(0), head_o, chunk_start
            )
            tma_load_tile(sGate_tma[raw_index.idx], gate_slice, raw_ready_ptr)
            if cutlass.const_expr(not cfg.v_is_zero):
                v_slice = tma_slice_runtime_desc(
                    desc_v_slot, cutlass.Int32(0), head_v, chunk_start
                )
                tma_load_tile(sV_tma[raw_index.idx], v_slice, raw_ready_ptr)

            raw_index = advance(raw_index, cfg.smem_raw_stages)
        tile_idx, scheduler_state = scheduler_publish_next(
            cfg, bars, sScheduler, mScheduler, scheduler_state, num_ctas, elect_one
        )
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
    mA_log,
    mDt_bias,
    sK_inv_raw,
    sGate_exchange_raw,
    sGate_load_ptr,
    mBeta,
    sBeta_raw,
    sK_raw,
    sK_decay_raw,
    sK_restore_raw,
    sCheckpoint_raw,
    sTmem_base,
    checkpoint_every_n_tokens,
    bars,
) -> None:
    """CG0 warp-group role (warps 0-7): persistent scheduler loop running the
    Gate prefix scan and staging the decay/restore operands."""
    nvvm.setmaxregister(
        cfg.num_regs_compute_group_0,
        nvvm.SetMaxRegisterAction.INCREASE
        if cfg.num_regs_compute_group_0 >= 65536 // cfg.threads_per_cta
        else nvvm.SetMaxRegisterAction.DECREASE,
    )
    elect_one = nvvm.elect_sync()
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = sTmem_base.load()
    tmem_col = tmem_base & 0xFFFF
    row_lo_addr = (tmem_base >> 16) << 16
    state_col_id = tmem_col + cfg.tmem_state_acc_offset
    packed_col_id = tmem_col + cfg.tmem_state_input_offset

    scheduler_state = PipelineState.start(phase=0)

    cg0_warp = warp_idx - cfg.compute_group_0_warp_ids[0]
    cg0_local_warp = cg0_warp % cfg.cg0_warps_per_group
    dk_halves = cutlass.const_expr(cfg.d_k // 64)
    channel_rows = cutlass.const_expr(cfg.d_k // cfg.cg0_warps_per_group)
    store_rows = cutlass.const_expr(cfg.b_t * channel_rows // cfg.threads_per_warp)
    store_row_base = (lane_idx // cutlass.Int32(channel_rows)) * cutlass.Int32(store_rows)

    cg0_group_id = cg0_warp // cfg.cg0_warps_per_group
    channel_dim = cg0_local_warp * cfg.threads_per_warp + lane_idx
    if cutlass.const_expr(channel_rows < cfg.threads_per_warp):
        channel_dim = cg0_local_warp * channel_rows + lane_idx % channel_rows
    cg0_a_log_exp = cutlass.Float32(1.0)
    cg0_dt_bias_value = cutlass.Float32(0.0)
    if cutlass.const_expr(cfg.enable_checkpoints):
        checkpoint_chunks = checkpoint_every_n_tokens // cutlass.Int32(cfg.b_t)
        checkpoint_row_base = cutlass.Int32(0)
        sCheckpoint_ptr = smem_data_ptr(sCheckpoint_raw)
        if cutlass.const_expr(cfg.d_v == 128):
            checkpoint_row_dim = cg0_local_warp * cfg.threads_per_warp + lane_idx
            checkpoint_row_valid = cutlass.Boolean(True)
        else:
            checkpoint_row_dim = cg0_local_warp * 16 + lane_idx % 16
            checkpoint_row_valid = lane_idx < 16
    prefix_segment = channel_dim // 32
    prefix_seg_base = prefix_segment * (cfg.b_t * 32)
    prefix_col = channel_dim - prefix_segment * 32
    prefix_row_offsets = [
        opaque_i32(
            prefix_seg_base + swizzle_xor_128b(j ^ prefix_segment, prefix_col, elem_bytes=4)
        )
        for j in range(8)
    ]
    if cutlass.const_expr(cfg.gate_dtype == cutlass.Float32):
        gate_row_offsets = [
            opaque_i32(prefix_seg_base + swizzle_xor_128b(j, prefix_col, elem_bytes=4))
            for j in range(8)
        ]
    else:
        raw_segment = channel_dim // 64
        raw_seg_base = raw_segment * (cfg.b_t * 64)
        raw_col = channel_dim - raw_segment * 64
        gate_row_offsets = [
            opaque_i32(raw_seg_base + swizzle_xor_128b(j, raw_col, elem_bytes=2)) for j in range(8)
        ]
    cum_chunk_base = cutlass.Int32(0)
    tile_idx = cutlass.Int32(bidx)
    opaque_one = opaque_f32_zero() + cutlass.Float32(1.0)
    while tile_idx < total_tiles:
        (
            _batch_idx,
            head_idx,
            batch_start,
            _batch_end,
            batch_seqlen,
            _batch_num_chunks,
            write_start,
            write_end,
            compute_start,
            _compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        head_o = head_idx
        num_chunks_tile = write_end - compute_start
        if cutlass.const_expr(cfg.enable_checkpoints):
            checkpoint_lo = compute_start + cutlass.Int32(1)
            checkpoint_lo = write_start if write_start > checkpoint_lo else checkpoint_lo
            checkpoint_seed_rows = cutlass.Int32(1) if write_start == 0 else cutlass.Int32(0)
            if cutlass.const_expr(cfg.seed_checkpoints):
                checkpoint_seed_rows = cutlass.Int32(1)
            checkpoint_lo_quotient = (checkpoint_lo - cutlass.Int32(1)) // checkpoint_chunks
        if cutlass.const_expr(mA_log is not None):
            if num_chunks_tile > 0:
                cg0_a_log_exp = cute.math.exp2(
                    mA_log[head_o].to(cutlass.Float32) * LOG2_E, fastmath=True
                )
        if cutlass.const_expr(mDt_bias is not None):
            if num_chunks_tile > 0:
                cg0_dt_bias_value = mDt_bias[head_o, channel_dim].to(cutlass.Float32)
        nvvm.barrier_cta_sync(
            cfg.cg0_tile_entry_barrier_id,
            thread_count=cfg.cg0_group_count * cfg.cg0_threads_per_group,
        )
        for local_chunk_idx in cutlass.range(
            cg0_group_id, num_chunks_tile, cfg.cg0_group_count, unroll=1
        ):
            chunk_idx = compute_start + local_chunk_idx
            cum_chunk = cum_chunk_base + local_chunk_idx
            chunk_count = cutlass.Uint32(cum_chunk)
            chunk_start = chunk_idx * cfg.b_t
            decay_stage = cutlass.Int32(chunk_count % cfg.smem_decay_stages)
            raw_stage = cutlass.Int32(chunk_count % cfg.smem_raw_stages)
            raw_parity = cutlass.Int32((chunk_count // cfg.smem_raw_stages) % 2)
            raw_free_parity = cutlass.Int32(((chunk_count // cfg.smem_raw_stages) + 1) % 2)
            qk_scale_ready_stage = cutlass.Int32(chunk_count % cfg.qk_scale_ready_stages)
            decay_free_parity = cutlass.Int32(((chunk_count // cfg.smem_decay_stages) + 1) % 2)
            exchange_stage = cutlass.Int32(chunk_count % cfg.gate_exchange_stages)
            sK_ptr = smem_data_ptr(sK_raw) + raw_stage * (cfg.d_k * cfg.b_t)
            sGate_ptr = sGate_load_ptr + raw_stage * cfg.gate_stage_elems
            sGate_exchange_ptr = smem_data_ptr(sGate_exchange_raw) + exchange_stage * (
                cfg.d_k * cfg.b_t
            )
            sK_inv_ptr = smem_data_ptr(sK_inv_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = smem_data_ptr(sK_decay_raw) + decay_stage * (cfg.d_k * cfg.b_t)
            sK_restore_ptr = smem_data_ptr(sK_restore_raw) + decay_stage * (cfg.d_k * cfg.b_t)

            # ---- Beta scalars --------------------------------------------------------
            if cg0_local_warp == 0:
                bars.mb_beta_done[raw_stage].wait(raw_free_parity)
                if lane_idx < cfg.b_t:
                    token_idx = chunk_idx * cfg.b_t + lane_idx
                    beta_value = cutlass.Float32(0.0)
                    if token_idx < batch_seqlen:
                        beta_value = mBeta[batch_start + token_idx, head_o].to(cutlass.Float32)
                        if cutlass.const_expr(cfg.beta_sigmoid):
                            beta_value = (
                                (sigmoid(beta_value) * (2.0 if cfg.allow_neg_eigval else 1.0))
                                .to(mBeta.element_type)
                                .to(cutlass.Float32)
                            )
                    sBeta_raw[raw_stage * cfg.b_t + lane_idx] = beta_value
                if nvvm.elect_sync():
                    bars.mb_beta_ready[raw_stage].arrive()
            bars.mb_raw_ready[raw_stage].wait(raw_parity)

            row_group_start = cg0_local_warp * (cfg.b_t // cfg.cg0_warps_per_group)
            lane_row_group = lane_idx // 8
            lane_in_row_group = lane_idx - lane_row_group * 8
            decay_row = row_group_start + lane_row_group

            # ---- Gate prefix scan ----------------------------------------------------
            gate_raw = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
            if cutlass.const_expr(cfg.gate_dtype == cutlass.Float32):
                for row in cutlass.range_constexpr(cfg.b_t):
                    gate_raw[row] = (sGate_ptr + (gate_row_offsets[row % 8] + row * 32)).load()
            else:
                for row in cutlass.range_constexpr(cfg.b_t):
                    gate_raw[row] = (
                        (sGate_ptr + (gate_row_offsets[row % 8] + row * 64))
                        .load()
                        .to(cutlass.Float32)
                    )
            g_prefix_regs = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
            if cutlass.const_expr(cfg.safe_gate):
                for row in cutlass.range_constexpr(cfg.b_t):
                    g_prefix_regs[row] = gate_scale(
                        cfg, cg0_a_log_exp * (gate_raw[row] + cg0_dt_bias_value)
                    )
            else:
                for row in cutlass.range_constexpr(cfg.b_t):
                    g_prefix_regs[row] = gate_scale(cfg, gate_raw[row])

            # ---- ragged tail chunk: padded rows carry no decay -----------------------
            if chunk_start + cutlass.Int32(cfg.b_t) > batch_seqlen:
                for row in cutlass.range_constexpr(cfg.b_t):
                    g_prefix_regs[row] = (
                        cutlass.Float32(0.0)
                        if chunk_start + cutlass.Int32(row) >= batch_seqlen
                        else g_prefix_regs[row]
                    )

            prefix_acc = cutlass.Float32(0.0)
            for row_pair in cutlass.range_constexpr(cfg.b_t // 2):
                row0 = row_pair * 2
                row1 = row0 + 1
                gate0 = g_prefix_regs[row0]
                gate1 = g_prefix_regs[row1]
                prefix0, row_pair_sum = fadd2(prefix_acc, gate0, gate0, gate1)
                prefix1 = prefix_acc + row_pair_sum
                g_prefix_regs[row0] = prefix0
                g_prefix_regs[row1] = prefix1
                prefix_acc = prefix1

            # ---- exp2(g): stage prefixes + final-token decay -------------------------
            for row in cutlass.range_constexpr(cfg.b_t):
                g_prefix_regs[row] = cute.math.exp2(g_prefix_regs[row], fastmath=True)

            for row_off in cutlass.range_constexpr(store_rows):
                if cutlass.const_expr(channel_rows < cfg.threads_per_warp):
                    row = store_row_base + cutlass.Int32(row_off)
                    value = (
                        g_prefix_regs[row_off + cfg.b_t // 2]
                        if lane_idx >= cutlass.Int32(channel_rows)
                        else g_prefix_regs[row_off]
                    )
                    prefix_idx = prefix_seg_base + swizzle_xor_128b(
                        row ^ prefix_segment, row * 32 + prefix_col, elem_bytes=4
                    )
                else:
                    value = g_prefix_regs[row_off]
                    prefix_idx = prefix_row_offsets[row_off % 8] + row_off * 32
                (sGate_exchange_ptr + prefix_idx).store(value)
            nvvm.barrier_cta_sync(
                cfg.cg0_group_sync_barrier_base_id + cg0_group_id,
                thread_count=cfg.cg0_threads_per_group,
            )
            if nvvm.elect_sync():
                bars.mb_gate_exchange_ready[raw_stage].arrive()

            k_inv_pack = cute.make_rmem_tensor((dk_halves * 4,), cutlass.Int32)
            k_restore_pack = cute.make_rmem_tensor((dk_halves * 4,), cutlass.Int32)
            raw_k_regs = cute.make_rmem_tensor((dk_halves * 8,), cutlass.Float32)

            # ---- optional K L2-norm + K inv stage ------------------------------------
            if cutlass.const_expr(cfg.l2norm):
                kk_lo = opaque_f32_zero()
                kk_hi = opaque_f32_zero()
            for dim_half in cutlass.range_constexpr(dk_halves):
                dim_base = dim_half * 64 + lane_in_row_group * 8
                reg_base = dim_half * 8
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                raw_f16_idx = (
                    f16_segment * (cfg.b_t * 64)
                    + decay_row * 64
                    + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                )
                raw_k_frag = (sK_ptr + raw_f16_idx).load(count=8, alignment=16)
                raw_k_vec_f32 = raw_k_frag.to(cutlass.Float32)
                for dim_offset in cutlass.range_constexpr(8):
                    k_val = raw_k_vec_f32[dim_offset]
                    raw_k_regs[reg_base + dim_offset] = k_val
                if cutlass.const_expr(cfg.l2norm):
                    for dim_pair in cutlass.range_constexpr(4):
                        k_even = raw_k_vec_f32[2 * dim_pair]
                        k_odd = raw_k_vec_f32[2 * dim_pair + 1]
                        kk_lo, kk_hi = ffma2(k_even, k_odd, k_even, k_odd, kk_lo, kk_hi)

            k_inv_norm = opaque_one
            if cutlass.const_expr(cfg.l2norm):
                k_sum_sq = kk_lo + kk_hi
                k_sum_sq = k_sum_sq + cutlass.Float32(
                    nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 4, 31, kind=nvvm.Shfl.BFLY)
                )
                k_sum_sq = k_sum_sq + cutlass.Float32(
                    nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 2, 31, kind=nvvm.Shfl.BFLY)
                )
                k_sum_sq = k_sum_sq + cutlass.Float32(
                    nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 1, 31, kind=nvvm.Shfl.BFLY)
                )
                norm_floor_sq = cutlass.Float32(L2_NORM_EPS * L2_NORM_EPS)
                k_inv_norm = cute.math.rsqrt(cute.math.max(k_sum_sq, norm_floor_sq), fastmath=True)

            # ---- decay/restore operands: exp2(+-g) applied per key channel -----------
            exp_g_regs = cute.make_rmem_tensor((dk_halves * 8,), cutlass.Float32)
            for dim_half in cutlass.range_constexpr(dk_halves):
                dim_base = dim_half * 64 + lane_in_row_group * 8
                reg_base = dim_half * 8
                for f32_group in cutlass.range_constexpr(2):
                    f32_dim_base = dim_base + f32_group * 4
                    f32_segment = f32_dim_base // 32
                    f32_segment_dim = f32_dim_base - f32_segment * 32
                    g_prefix_idx = (
                        f32_segment * (cfg.b_t * 32)
                        + decay_row * 32
                        + swizzle_xor_128b(decay_row ^ f32_segment, f32_segment_dim, elem_bytes=4)
                    )
                    exp_g_frag = (sGate_exchange_ptr + g_prefix_idx).load(count=4, alignment=16)
                    f32_reg_base = reg_base + f32_group * 4
                    for j in cutlass.range_constexpr(4):
                        exp_g_regs[f32_reg_base + j] = exp_g_frag[j]

            for dim_half in cutlass.range_constexpr(dk_halves):
                dim_base = dim_half * 64 + lane_in_row_group * 8
                reg_base = dim_half * 8

                # ---- K decay + K inv + K restore operands: K * exp2(+g), K * exp2(-g), K * exp2(g
                # last - g)
                exp_g_last_half = cute.make_rmem_tensor((8,), cutlass.Float32)
                for f32_group in cutlass.range_constexpr(2):
                    f32_dim_base = dim_base + f32_group * 4
                    f32_segment = f32_dim_base // 32
                    f32_segment_dim = f32_dim_base - f32_segment * 32
                    exp_g_last_idx = (
                        f32_segment * (cfg.b_t * 32)
                        + (cfg.b_t - 1) * 32
                        + swizzle_xor_128b(
                            cfg.b_t - 1 ^ f32_segment, f32_segment_dim, elem_bytes=4
                        )
                    )
                    exp_g_last_frag = (sGate_exchange_ptr + exp_g_last_idx).load(
                        count=4, alignment=16
                    )
                    for j in cutlass.range_constexpr(4):
                        exp_g_last_half[f32_group * 4 + j] = exp_g_last_frag[j]
                k_decay_pack = cute.make_rmem_tensor((4,), cutlass.Int32)
                for pair_idx in cutlass.range_constexpr(4):
                    dim0 = pair_idx * 2
                    dim1 = dim0 + 1
                    raw_reg_idx0 = reg_base + dim0
                    raw_reg_idx1 = reg_base + dim1
                    k_value0, k_value1 = fmul2(
                        raw_k_regs[raw_reg_idx0], raw_k_regs[raw_reg_idx1], k_inv_norm, k_inv_norm
                    )
                    k_decay0, k_decay1 = fmul2(
                        k_value0, k_value1, exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1]
                    )
                    k_decay_pack[pair_idx] = fp32_to_fp16(k_decay0, k_decay1, dtype=cfg.io_dtype)
                    exp_neg_g0 = cute.math.rcp(exp_g_regs[raw_reg_idx0], approx=True, ftz=True)
                    exp_neg_g1 = cute.math.rcp(exp_g_regs[raw_reg_idx1], approx=True, ftz=True)
                    k_inv0, k_inv1 = fmul2(k_value0, k_value1, exp_neg_g0, exp_neg_g1)
                    k_inv_pack[dim_half * 4 + pair_idx] = fp32_to_fp16(
                        k_inv0, k_inv1, dtype=cfg.io_dtype
                    )
                    k_restore0, k_restore1 = fmul2(
                        k_inv0, k_inv1, exp_g_last_half[dim0], exp_g_last_half[dim1]
                    )
                    k_restore_pack[dim_half * 4 + pair_idx] = fp32_to_fp16(
                        k_restore0, k_restore1, dtype=cfg.io_dtype
                    )

                k_inv_vec = cutlass.Vector.from_elements(
                    (
                        k_inv_pack[dim_half * 4],
                        k_inv_pack[dim_half * 4 + 1],
                        k_inv_pack[dim_half * 4 + 2],
                        k_inv_pack[dim_half * 4 + 3],
                    ),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                k_decay_vec = cutlass.Vector.from_elements(
                    (
                        k_decay_pack[0],
                        k_decay_pack[1],
                        k_decay_pack[2],
                        k_decay_pack[3],
                    ),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                if cutlass.const_expr(dim_half == 0):
                    bars.mb_decay_register_mma_done[decay_stage].wait(decay_free_parity)
                    bars.mb_decay_tcgen05_done[decay_stage].wait(decay_free_parity)
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                k_inv_swizzled_idx = (
                    f16_segment * (cfg.b_t * 64)
                    + decay_row * 64
                    + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                )
                (sK_inv_ptr + k_inv_swizzled_idx).store(k_inv_vec, alignment=16)
                decay_col = dim_base
                decay_segment = decay_col // 64
                decay_swizzled_idx = decay_segment * (cfg.b_t * 64) + swizzle_xor_128b(
                    decay_row, decay_row * 64 + decay_col - decay_segment * 64, elem_bytes=2
                )
                (sK_decay_ptr + decay_swizzled_idx).store(k_decay_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            if nvvm.elect_sync():
                bars.mb_k_decay_inv_cg0_ready[decay_stage].arrive()

            # ---- K restore operand store ---------------------------------------------
            bars.mb_k_restore_done[decay_stage].wait(decay_free_parity)
            for dim_half in cutlass.range_constexpr(dk_halves):
                dim_base = dim_half * 64 + lane_in_row_group * 8
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                k_restore_idx = (
                    f16_segment * (cfg.b_t * 64)
                    + decay_row * 64
                    + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                )
                k_restore_vec = cutlass.Vector.from_elements(
                    (
                        k_restore_pack[dim_half * 4],
                        k_restore_pack[dim_half * 4 + 1],
                        k_restore_pack[dim_half * 4 + 2],
                        k_restore_pack[dim_half * 4 + 3],
                    ),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                (sK_restore_ptr + k_restore_idx).store(k_restore_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            if nvvm.elect_sync():
                bars.mb_qk_scale_ready[qk_scale_ready_stage].arrive()

            # ---- state stage, left key half: pack, publish, fp32 decay ---------------
            if cum_chunk > 0:
                update_count = chunk_count - cutlass.Uint32(1)
                bars.mb_state_acc_cg0_done[
                    cutlass.Int32(update_count % cfg.smem_decay_stages)
                ].wait(cutlass.Int32((update_count // cfg.smem_decay_stages) % 2))
            if local_chunk_idx > 0:
                l_state_vecs = []
                for b in cutlass.range_constexpr(dk_halves):
                    l_state_vecs.append(
                        nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                row_lo_addr + state_col_id + b * 32, cutlass.Float32
                            ),
                            num=32,
                        )
                    )
                l_packed_blocks = []
                for b in cutlass.range_constexpr(dk_halves):
                    l_packed = cute.make_rmem_tensor((16,), cutlass.Int32)
                    for packed_col in cutlass.range_constexpr(16):
                        l_packed[packed_col] = fp32_to_fp16(
                            l_state_vecs[b][2 * packed_col],
                            l_state_vecs[b][2 * packed_col + 1],
                            dtype=cfg.io_dtype,
                        )
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + packed_col_id + b * 16, cutlass.Int8),
                        l_packed.load(),
                    )
                    l_packed_blocks.append(l_packed)
                nvvm.tcgen05_wait("store")
                if nvvm.elect_sync():
                    bars.mb_state_input_cg0_ready.arrive()

                # ---- fp32 decay of the left key half: state *= exp2(g last) ----------
                for b in cutlass.range_constexpr(dk_halves):
                    l_scaled = []
                    for scale_group in cutlass.range_constexpr(8):
                        scale_dim = b * 32 + scale_group * 4
                        scale_segment = scale_dim // 32
                        scale_idx = (
                            scale_segment * (cfg.b_t * 32)
                            + (cfg.b_t - 1) * 32
                            + swizzle_xor_128b(
                                cfg.b_t - 1 ^ scale_segment,
                                scale_dim - scale_segment * 32,
                                elem_bytes=4,
                            )
                        )
                        l_scale_frag = (sGate_exchange_ptr + scale_idx).load(count=4, alignment=16)
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
                nvvm.tcgen05_wait("store")

                # ---- checkpoint row, left key half: the packed operand words ---------
                if cutlass.const_expr(cfg.enable_checkpoints):
                    checkpoint_row = chunk_idx % checkpoint_chunks == 0
                    checkpoint_row = checkpoint_row and chunk_idx >= write_start
                    if checkpoint_row:
                        checkpoint_row_idx = (
                            checkpoint_row_base
                            + checkpoint_seed_rows
                            + (chunk_idx - cutlass.Int32(1)) // checkpoint_chunks
                            - checkpoint_lo_quotient
                        )
                        checkpoint_row_stage = checkpoint_row_idx % cutlass.Int32(
                            cfg.smem_checkpoint_stages
                        )
                        bars.mb_checkpoint_tmastg_done[checkpoint_row_stage].wait(
                            (
                                checkpoint_row_idx // cutlass.Int32(cfg.smem_checkpoint_stages)
                                + cutlass.Int32(1)
                            )
                            % 2
                        )
                        checkpoint_row_addr = checkpoint_row_stage * (cfg.d_k * cfg.d_v)
                        if checkpoint_row_valid:
                            for b in cutlass.range_constexpr(dk_halves):
                                for word_group in cutlass.range_constexpr(4):
                                    dk = b * 32 + word_group * 8
                                    row_addr = (
                                        checkpoint_row_addr
                                        + (dk // 64) * (cfg.d_v * 64)
                                        + checkpoint_row_dim * 64
                                        + swizzle_xor_128b(
                                            checkpoint_row_dim, dk % 64, elem_bytes=2
                                        )
                                    )
                                    (sCheckpoint_ptr + row_addr).store(
                                        cutlass.Vector.from_elements(
                                            tuple(
                                                l_packed_blocks[b][word_group * 4 + t]
                                                for t in range(4)
                                            ),
                                            cutlass.Int32,
                                        ).bitcast(cfg.io_dtype),
                                        alignment=16,
                                    )
                        nvvm.fence_proxy("async.shared", space="cta")
                        if nvvm.elect_sync():
                            bars.mb_checkpoint_tmastg_ready[checkpoint_row_stage].arrive()
            if nvvm.elect_sync():
                bars.mb_raw_done[raw_stage].arrive()
                bars.mb_u_input_ready.arrive()
        if cutlass.const_expr(cfg.enable_checkpoints):
            if num_chunks_tile > 0:
                checkpoint_row_base = (
                    checkpoint_row_base
                    + checkpoint_seed_rows
                    + (write_end - cutlass.Int32(1)) // checkpoint_chunks
                    - checkpoint_lo_quotient
                )
        cum_chunk_base += num_chunks_tile
        tile_idx, scheduler_state = scheduler_next_tile(
            cfg, bars, sScheduler, scheduler_state, elect_one
        )


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
    sTmem_base,
    warp_idx,
    mState_out,
    mState_init,
    mSeedCheckpoints,
    sBeta_raw,
    sV_raw,
    sCheckpoint_raw,
    sGate_exchange_raw,
    checkpoint_every_n_tokens,
    seed_every_n_tokens,
    bars,
) -> None:
    """CG1 warp-group role (warps 8-11): persistent scheduler loop staging the
    value-side TMEM operands and storing the checkpoint/final states."""
    nvvm.setmaxregister(
        cfg.num_regs_compute_group_1,
        nvvm.SetMaxRegisterAction.INCREASE
        if cfg.num_regs_compute_group_1 >= 65536 // cfg.threads_per_cta
        else nvvm.SetMaxRegisterAction.DECREASE,
    )
    elect_one = nvvm.elect_sync()

    if cutlass.const_expr(cfg.enable_checkpoints):
        checkpoint_chunks = checkpoint_every_n_tokens // cutlass.Int32(cfg.b_t)
        checkpoint_row_count = cutlass.Int32(0)

    sCheckpoint_ptr = (
        smem_data_ptr(sCheckpoint_raw)
        if cutlass.const_expr(cfg.enable_checkpoints)
        else smem_data_ptr(sV_raw)
    )
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = sTmem_base.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16

    # ---- ldmatrix.x4 COL lane decode for the V loads ---------------------------------
    ov_row_coord = (lane_idx // 16) * 8 + (lane_idx & 7)
    ov_col_offset = ((lane_idx // 8) & 1) * 8
    if cutlass.const_expr(cfg.d_v == 128):
        tmem_subpartition = warp_idx % (cfg.d_v // cfg.threads_per_warp)
        value_dim = tmem_subpartition * cfg.threads_per_warp + lane_idx
        value_dim_base = tmem_subpartition * cfg.threads_per_warp
        state_row_valid = cutlass.Boolean(True)
    else:
        cg1_warp = warp_idx % len(cfg.compute_group_1_warp_ids)
        value_dim = cg1_warp * 16 + lane_idx % 16
        value_dim_base = cg1_warp * 16
        state_row_valid = lane_idx < 16
    row_lo_addr = tmem_row << 16
    row_hi_addr = (tmem_row + 16) << 16
    state_col_id = tmem_col + cfg.tmem_state_acc_offset
    packed_col_id = tmem_col + cfg.tmem_state_input_offset
    statek_col_id = tmem_col + cfg.tmem_state_k_acc_offset
    y_input_col_id = tmem_col + cfg.tmem_y_input_offset
    u_acc_addr = row_lo_addr + tmem_col + cfg.tmem_u_acc_offset
    u_input_addr = row_lo_addr + tmem_col + cfg.tmem_u_input_offset
    v_swizzle_off_lo = (
        (value_dim_base + ov_col_offset) // 64 * (cfg.b_t * 64)
        + ov_row_coord * 64
        + swizzle_xor_128b(ov_row_coord, (value_dim_base + ov_col_offset) % 64, elem_bytes=2)
    )
    v_swizzle_off_hi = (
        (value_dim_base + 16 + ov_col_offset) // 64 * (cfg.b_t * 64)
        + ov_row_coord * 64
        + swizzle_xor_128b(ov_row_coord, (value_dim_base + 16 + ov_col_offset) % 64, elem_bytes=2)
    )
    state_k_acc_index = PipelineState.start(phase=0)
    u_acc_index = PipelineState.start(phase=0)
    state_update_index = PipelineState.start(phase=0)
    raw_index = PipelineState.start(phase=0)
    state_blocks_per_half = cutlass.const_expr(cfg.d_k // 32)
    cum_chunk_base = cutlass.Int32(0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        (
            batch_idx,
            head_idx,
            _batch_start,
            _batch_end,
            batch_seqlen,
            batch_num_chunks,
            write_start,
            write_end,
            compute_start,
            _compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        head_o = head_idx
        num_chunks_tile = write_end - compute_start
        if cutlass.const_expr(cfg.enable_checkpoints):
            checkpoint_phase = compute_start % checkpoint_chunks
            checkpoint_stage = checkpoint_row_count % cutlass.Int32(cfg.smem_checkpoint_stages)
            checkpoint_stage_parity = (
                checkpoint_row_count // cutlass.Int32(cfg.smem_checkpoint_stages)
                + cutlass.Int32(1)
            ) % 2

        if num_chunks_tile > 0:
            # ---- first chunk: seed state TMEM from mState init -----------------------
            seed_from_initial_state = compute_start == 0
            if cutlass.const_expr(cfg.seed_checkpoints):
                seed_from_initial_state = cutlass.Boolean(True)
            sV_ptr = smem_data_ptr(sV_raw) + raw_index.idx * (cfg.d_v * cfg.b_t)
            sBeta_ptr = smem_data_ptr(sBeta_raw) + raw_index.idx * cfg.b_t

            # ---- state seed: coarse checkpoint GMEM -> packed b16 TMEM + fp32 state TMEM ----
            if cutlass.const_expr(cfg.seed_checkpoints):
                seed_row = write_start // (seed_every_n_tokens // cutlass.Int32(cfg.b_t))
                seed_b = cutlass.Int32(0)
                while seed_b < batch_idx:
                    seed_len = expanded_cu_seqlen(1, cu_seqlens, seed_b + 1) - expanded_cu_seqlen(
                        1, cu_seqlens, seed_b
                    )
                    seed_row = (
                        seed_row
                        + (seed_len + seed_every_n_tokens - cutlass.Int32(1))
                        // seed_every_n_tokens
                    )
                    seed_b = seed_b + cutlass.Int32(1)
                seed_width = 16 // (mSeedCheckpoints.element_type.width // 8)
                seed_ptr = (
                    mSeedCheckpoints.iterator
                    + mSeedCheckpoints.layout((seed_row, head_o, value_dim, 0))
                ).raw_ptr()
                bars.mb_gate_exchange_ready[raw_index.idx].wait(raw_index.phase)
                seed_exchange_ptr = smem_data_ptr(sGate_exchange_raw) + (
                    cum_chunk_base % cfg.gate_exchange_stages
                ) * (cfg.d_k * cfg.b_t)
                seed_stage_base = checkpoint_stage * (cfg.d_k * cfg.d_v)
                for seed_half in cutlass.range_constexpr(2):
                    seed_blocks_lo = seed_half * state_blocks_per_half
                    seed_blocks_hi = seed_blocks_lo + state_blocks_per_half
                    seed_vecs = []
                    for i in cutlass.range_constexpr(seed_blocks_lo, seed_blocks_hi):
                        seed_words = []
                        for g in cutlass.range_constexpr(16 // seed_width):
                            seed_load = (seed_ptr + i * 16 + g * seed_width).load(
                                count=seed_width, alignment=16
                            )
                            for t in cutlass.range_constexpr(seed_width):
                                seed_words.append(seed_load[t].to(cutlass.Float32))
                        seed_vecs.append(seed_words)

                    seed_packed_blocks = []
                    for i in cutlass.range_constexpr(seed_blocks_lo, seed_blocks_hi):
                        seed_pack = cute.make_rmem_tensor((8,), cutlass.Int32)
                        for packed_col in cutlass.range_constexpr(8):
                            seed_pack[packed_col] = fp32_to_fp16(
                                seed_vecs[i - seed_blocks_lo][2 * packed_col],
                                seed_vecs[i - seed_blocks_lo][2 * packed_col + 1],
                                dtype=cfg.io_dtype,
                            )
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(row_lo_addr + packed_col_id + i * 8, cutlass.Int8),
                            seed_pack.load(),
                        )
                        seed_packed_blocks.append(seed_pack)
                    nvvm.tcgen05_wait("store")
                    if cutlass.const_expr(seed_half == 0):
                        if nvvm.elect_sync():
                            bars.mb_state_input_cg0_ready.arrive()
                    else:
                        if nvvm.elect_sync():
                            bars.mb_state_input_cg1_ready.arrive()

                    # ---- fp32 decay of the seed half: state = seed * exp2(g last) ----
                    for i in cutlass.range_constexpr(seed_blocks_lo, seed_blocks_hi):
                        seed_scaled = []
                        for seed_scale_group in cutlass.range_constexpr(4):
                            seed_scale_dim = i * 16 + seed_scale_group * 4
                            seed_scale_segment = seed_scale_dim // 32
                            seed_scale_idx = (
                                seed_scale_segment * (cfg.b_t * 32)
                                + (cfg.b_t - 1) * 32
                                + swizzle_xor_128b(
                                    cfg.b_t - 1 ^ seed_scale_segment,
                                    seed_scale_dim - seed_scale_segment * 32,
                                    elem_bytes=4,
                                )
                            )
                            seed_scale_frag = (seed_exchange_ptr + seed_scale_idx).load(
                                count=4, alignment=16
                            )
                            for t in cutlass.range_constexpr(2):
                                seed_s0, seed_s1 = fmul2(
                                    seed_vecs[i - seed_blocks_lo][seed_scale_group * 4 + 2 * t],
                                    seed_vecs[i - seed_blocks_lo][
                                        seed_scale_group * 4 + 2 * t + 1
                                    ],
                                    seed_scale_frag[2 * t],
                                    seed_scale_frag[2 * t + 1],
                                )
                                seed_scaled += [seed_s0, seed_s1]
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                row_lo_addr + state_col_id + i * 16, cutlass.Float32
                            ),
                            cutlass.Vector.from_elements(tuple(seed_scaled), cutlass.Float32),
                        )

                    # ---- seed checkpoint row half: the packed operand words ----------
                    if cutlass.const_expr(seed_half == 0):
                        bars.mb_checkpoint_tmastg_done[checkpoint_stage].wait(
                            checkpoint_stage_parity
                        )
                    if state_row_valid:
                        for i in cutlass.range_constexpr(seed_blocks_lo, seed_blocks_hi):
                            for word_group in cutlass.range_constexpr(2):
                                dk = i * 16 + word_group * 8
                                seed_row_addr = (
                                    seed_stage_base
                                    + (dk // 64) * (cfg.d_v * 64)
                                    + value_dim * 64
                                    + swizzle_xor_128b(value_dim, dk % 64, elem_bytes=2)
                                )
                                (sCheckpoint_ptr + seed_row_addr).store(
                                    cutlass.Vector.from_elements(
                                        tuple(
                                            seed_packed_blocks[i - seed_blocks_lo][
                                                word_group * 4 + t
                                            ]
                                            for t in range(4)
                                        ),
                                        cutlass.Int32,
                                    ).bitcast(cfg.io_dtype),
                                    alignment=16,
                                )
                nvvm.tcgen05_wait("store")
                nvvm.fence_proxy("async.shared", space="cta")
                if nvvm.elect_sync():
                    bars.mb_checkpoint_tmastg_ready[checkpoint_stage].arrive()
                    bars.mb_checkpoint_tmastg_ready[checkpoint_stage].arrive()
                checkpoint_row_count = checkpoint_row_count + cutlass.Int32(1)

            # ---- state seed: initial state GMEM -> packed b16 TMEM + fp32 state TMEM ----
            if cutlass.const_expr(mState_init is not None or cfg.seed_identity):
                if seed_from_initial_state:
                    bars.mb_gate_exchange_ready[raw_index.idx].wait(raw_index.phase)
                    init_exchange_ptr = smem_data_ptr(sGate_exchange_raw) + (
                        cum_chunk_base % cfg.gate_exchange_stages
                    ) * (cfg.d_k * cfg.b_t)
                    if cutlass.const_expr(cfg.enable_checkpoints):
                        init_stage_base = checkpoint_stage * (cfg.d_k * cfg.d_v)
                    for init_half in cutlass.range_constexpr(2):
                        init_blocks_lo = init_half * state_blocks_per_half
                        init_blocks_hi = init_blocks_lo + state_blocks_per_half
                        if cutlass.const_expr(cfg.seed_identity):
                            init_state_vecs = []
                            for i in cutlass.range_constexpr(init_blocks_lo, init_blocks_hi):
                                init_state_block = []
                                for j in cutlass.range_constexpr(16):
                                    init_state_block.append(
                                        cutlass.Float32(1.0)
                                        if value_dim == i * 16 + j
                                        else cutlass.Float32(0.0)
                                    )
                                init_state_vecs.append(init_state_block)
                        else:
                            seed_vw = 16 // (mState_init.element_type.width // 8)
                            seed_src = (
                                mState_init.iterator
                                + mState_init.layout((batch_idx, head_o, value_dim, 0))
                            ).raw_ptr()
                            init_state_vecs = []
                            for i in cutlass.range_constexpr(init_blocks_lo, init_blocks_hi):
                                init_state_block = []
                                for g in cutlass.range_constexpr(16 // seed_vw):
                                    seed_chunk = (seed_src + i * 16 + g * seed_vw).load(
                                        count=seed_vw, alignment=16
                                    )
                                    for t in cutlass.range_constexpr(seed_vw):
                                        init_state_block.append(seed_chunk[t].to(cutlass.Float32))
                                init_state_vecs.append(init_state_block)

                        init_packed_blocks = []
                        for i in cutlass.range_constexpr(init_blocks_lo, init_blocks_hi):
                            init_state_pack = cute.make_rmem_tensor((8,), cutlass.Int32)
                            for packed_col in cutlass.range_constexpr(8):
                                init_state_pack[packed_col] = fp32_to_fp16(
                                    init_state_vecs[i - init_blocks_lo][2 * packed_col],
                                    init_state_vecs[i - init_blocks_lo][2 * packed_col + 1],
                                    dtype=cfg.io_dtype,
                                )
                            nvvm.tcgen05_st(
                                "32x32b",
                                nvvm.make_tmem_ptr(
                                    row_lo_addr + packed_col_id + i * 8, cutlass.Int8
                                ),
                                init_state_pack.load(),
                            )
                            init_packed_blocks.append(init_state_pack)
                        nvvm.tcgen05_wait("store")
                        if cutlass.const_expr(init_half == 0):
                            if nvvm.elect_sync():
                                bars.mb_state_input_cg0_ready.arrive()
                        else:
                            if nvvm.elect_sync():
                                bars.mb_state_input_cg1_ready.arrive()

                        # ---- fp32 decay of the seed half: state = seed * exp2(g last) ----
                        for i in cutlass.range_constexpr(init_blocks_lo, init_blocks_hi):
                            init_state_scaled = []
                            for init_scale_group in cutlass.range_constexpr(4):
                                init_scale_dim = i * 16 + init_scale_group * 4
                                init_scale_segment = init_scale_dim // 32
                                init_scale_idx = (
                                    init_scale_segment * (cfg.b_t * 32)
                                    + (cfg.b_t - 1) * 32
                                    + swizzle_xor_128b(
                                        cfg.b_t - 1 ^ init_scale_segment,
                                        init_scale_dim - init_scale_segment * 32,
                                        elem_bytes=4,
                                    )
                                )
                                init_scale_frag = (init_exchange_ptr + init_scale_idx).load(
                                    count=4, alignment=16
                                )
                                for t in cutlass.range_constexpr(2):
                                    init_s0, init_s1 = fmul2(
                                        init_state_vecs[i - init_blocks_lo][
                                            init_scale_group * 4 + 2 * t
                                        ],
                                        init_state_vecs[i - init_blocks_lo][
                                            init_scale_group * 4 + 2 * t + 1
                                        ],
                                        init_scale_frag[2 * t],
                                        init_scale_frag[2 * t + 1],
                                    )
                                    init_state_scaled += [init_s0, init_s1]
                            nvvm.tcgen05_st(
                                "32x32b",
                                nvvm.make_tmem_ptr(
                                    row_lo_addr + state_col_id + i * 16, cutlass.Float32
                                ),
                                cutlass.Vector.from_elements(
                                    tuple(init_state_scaled), cutlass.Float32
                                ),
                            )

                        # ---- seed checkpoint row half: the packed operand words ------
                        if cutlass.const_expr(cfg.enable_checkpoints):
                            if write_start == 0:
                                if cutlass.const_expr(init_half == 0):
                                    bars.mb_checkpoint_tmastg_done[checkpoint_stage].wait(
                                        checkpoint_stage_parity
                                    )
                                if state_row_valid:
                                    for i in cutlass.range_constexpr(
                                        init_blocks_lo, init_blocks_hi
                                    ):
                                        for word_group in cutlass.range_constexpr(2):
                                            dk = i * 16 + word_group * 8
                                            init_row_addr = (
                                                init_stage_base
                                                + (dk // 64) * (cfg.d_v * 64)
                                                + value_dim * 64
                                                + swizzle_xor_128b(
                                                    value_dim, dk % 64, elem_bytes=2
                                                )
                                            )
                                            (sCheckpoint_ptr + init_row_addr).store(
                                                cutlass.Vector.from_elements(
                                                    tuple(
                                                        init_packed_blocks[i - init_blocks_lo][
                                                            word_group * 4 + t
                                                        ]
                                                        for t in range(4)
                                                    ),
                                                    cutlass.Int32,
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

            if cutlass.const_expr(
                cfg.enable_checkpoints
                and mState_init is None
                and not cfg.seed_checkpoints
                and not cfg.seed_identity
            ):
                if write_start == 0:
                    bars.mb_checkpoint_tmastg_done[checkpoint_stage].wait(checkpoint_stage_parity)
                    checkpoint_stage_base = checkpoint_stage * (cfg.d_k * cfg.d_v)
                    zero_packs = tuple(cutlass.Int32(0) for _ in range(4))
                    if state_row_valid:
                        for i in cutlass.range_constexpr(cfg.d_k // 16):
                            for g in cutlass.range_constexpr(2):
                                dk = i * 16 + g * 8
                                checkpoint_addr = (
                                    checkpoint_stage_base
                                    + (dk // 64) * (cfg.d_v * 64)
                                    + value_dim * 64
                                    + swizzle_xor_128b(value_dim, dk % 64, elem_bytes=2)
                                )
                                (sCheckpoint_ptr + checkpoint_addr).store(
                                    cutlass.Vector.from_elements(
                                        zero_packs, cutlass.Int32
                                    ).bitcast(cfg.io_dtype),
                                    alignment=16,
                                )
                    nvvm.fence_proxy("async.shared", space="cta")
                    if nvvm.elect_sync():
                        bars.mb_checkpoint_tmastg_ready[checkpoint_stage].arrive()
                        bars.mb_checkpoint_tmastg_ready[checkpoint_stage].arrive()
                    checkpoint_row_count = checkpoint_row_count + cutlass.Int32(1)

            # ---- Y stage: Y = Beta * (V - k state) -----------------------------------
            if cutlass.const_expr(cfg.v_is_zero):
                zero_word = opaque_i32_zero()
                raw_v_frag_lo = [zero_word for _ in range(4)]
                if cutlass.const_expr(cfg.d_v == 128):
                    raw_v_frag_hi = [zero_word for _ in range(4)]
            else:
                bars.mb_raw_ready[raw_index.idx].wait(raw_index.phase)
                raw_v_frag_lo = nvvm.ldmatrix(
                    sV_ptr + v_swizzle_off_lo,
                    4,
                    nvvm.MMALayout.COL,
                )
                if cutlass.const_expr(cfg.d_v == 128):
                    raw_v_frag_hi = nvvm.ldmatrix(
                        sV_ptr + v_swizzle_off_hi,
                        4,
                        nvvm.MMALayout.COL,
                    )
            bars.mb_beta_ready[raw_index.idx].wait(raw_index.phase)
            # FP32 beta per token pair; the residual v - k state is subtracted and scaled in FP32
            # and packed once for the MMA (packing the residual lost it against a large
            # contraction).
            beta_regs = cute.make_rmem_tensor((8,), cutlass.Float32)
            for reg_idx in cutlass.range_constexpr(4):
                token0 = ((reg_idx // 2) * 4 + (lane_idx & 3)) * 2
                beta_regs[2 * reg_idx] = (sBeta_ptr + token0).load().to(cutlass.Float32)
                beta_regs[2 * reg_idx + 1] = (sBeta_ptr + token0 + 1).load().to(cutlass.Float32)

            y_lo = [cutlass.Int32(0) for _ in range(4)]
            y_hi = [cutlass.Int32(0) for _ in range(4)]
            if cutlass.const_expr(
                mState_init is not None or cfg.seed_checkpoints or cfg.seed_identity
            ):
                if seed_from_initial_state:
                    bars.mb_state_k_acc_ready.wait(state_k_acc_index.phase)
                    state_k_acc_index = advance(state_k_acc_index, 1)
                    state_k_vec_lo = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(row_lo_addr + statek_col_id, cutlass.Float32),
                        num=2,
                    )
                    for reg_idx in cutlass.range_constexpr(4):
                        lo, hi = 2 * reg_idx, 2 * reg_idx + 1
                        y_lo[reg_idx], _, _ = beta_residual_f16x2(
                            raw_v_frag_lo[reg_idx],
                            beta_regs[lo],
                            beta_regs[hi],
                            state_k_vec_lo[lo],
                            state_k_vec_lo[hi],
                            dtype=cfg.io_dtype,
                        )
                    if cutlass.const_expr(cfg.d_v == 128):
                        state_k_vec_hi = nvvm.tcgen05_ld(
                            "16x256b",
                            nvvm.make_tmem_ptr(row_hi_addr + statek_col_id, cutlass.Float32),
                            num=2,
                        )
                        for reg_idx in cutlass.range_constexpr(4):
                            lo, hi = 2 * reg_idx, 2 * reg_idx + 1
                            y_hi[reg_idx], _, _ = beta_residual_f16x2(
                                raw_v_frag_hi[reg_idx],
                                beta_regs[lo],
                                beta_regs[hi],
                                state_k_vec_hi[lo],
                                state_k_vec_hi[hi],
                                dtype=cfg.io_dtype,
                            )
                else:
                    for reg_idx in cutlass.range_constexpr(4):
                        lo, hi = 2 * reg_idx, 2 * reg_idx + 1
                        y_lo[reg_idx], _, _ = beta_residual_f16x2(
                            raw_v_frag_lo[reg_idx],
                            beta_regs[lo],
                            beta_regs[hi],
                            dtype=cfg.io_dtype,
                        )
                        if cutlass.const_expr(cfg.d_v == 128):
                            y_hi[reg_idx], _, _ = beta_residual_f16x2(
                                raw_v_frag_hi[reg_idx],
                                beta_regs[lo],
                                beta_regs[hi],
                                dtype=cfg.io_dtype,
                            )
            else:
                for reg_idx in cutlass.range_constexpr(4):
                    lo, hi = 2 * reg_idx, 2 * reg_idx + 1
                    y_lo[reg_idx], _, _ = beta_residual_f16x2(
                        raw_v_frag_lo[reg_idx], beta_regs[lo], beta_regs[hi], dtype=cfg.io_dtype
                    )
                    if cutlass.const_expr(cfg.d_v == 128):
                        y_hi[reg_idx], _, _ = beta_residual_f16x2(
                            raw_v_frag_hi[reg_idx],
                            beta_regs[lo],
                            beta_regs[hi],
                            dtype=cfg.io_dtype,
                        )

            y_input_pack_lo = cute.make_rmem_tensor((4,), cutlass.Int32)
            y_input_pack_hi = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                y_input_pack_lo[reg_idx] = y_lo[reg_idx]
                y_input_pack_hi[reg_idx] = y_hi[reg_idx]
            nvvm.tcgen05_st(
                "16x128b",
                nvvm.make_tmem_ptr(row_lo_addr + y_input_col_id, cutlass.Int8),
                y_input_pack_lo.load(),
            )
            if cutlass.const_expr(cfg.d_v == 128):
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(row_hi_addr + y_input_col_id, cutlass.Int8),
                    y_input_pack_hi.load(),
                )
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_beta_done[raw_index.idx].arrive()
                bars.mb_raw_done[raw_index.idx].arrive()
                bars.mb_y_input_ready.arrive()

            # ---- U stage: u acc TMEM -> packed b16 U input TMEM ----------------------
            bars.mb_u_acc_ready.wait(u_acc_index.phase)
            u_acc_vals = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(u_acc_addr, cutlass.Float32),
                num=cfg.b_t,
            )

            u_input_pack = cute.make_rmem_tensor(((cfg.b_t // 2),), cutlass.Int32)
            for packed_col in cutlass.range_constexpr(cfg.b_t // 2):
                token0 = packed_col * 2
                token1 = token0 + 1
                u_input_pack[packed_col] = fp32_to_fp16(
                    u_acc_vals[token0], u_acc_vals[token1], dtype=cfg.io_dtype
                )

            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(u_input_addr, cutlass.Int8),
                u_input_pack.load(),
            )
            nvvm.tcgen05_wait("store")
            u_acc_index = advance(u_acc_index, 1)
            if nvvm.elect_sync():
                bars.mb_u_input_ready.arrive()

            if cutlass.const_expr(cfg.enable_checkpoints):
                checkpoint_phase = checkpoint_phase + cutlass.Int32(1)
                if checkpoint_phase == checkpoint_chunks:
                    checkpoint_phase = cutlass.Int32(0)
            raw_index = advance(raw_index, cfg.smem_raw_stages)

        for local_chunk_idx in cutlass.range(1, num_chunks_tile, 1, unroll=1):
            chunk_idx = compute_start + local_chunk_idx
            cum_chunk = cum_chunk_base + local_chunk_idx
            raw_stage = raw_index.idx
            raw_phase = raw_index.phase
            sV_ptr = smem_data_ptr(sV_raw) + raw_stage * (cfg.d_v * cfg.b_t)
            sBeta_ptr = smem_data_ptr(sBeta_raw) + raw_stage * cfg.b_t
            if cutlass.const_expr(not cfg.v_is_zero):
                bars.mb_raw_ready[raw_stage].wait(raw_phase)
            bars.mb_beta_ready[raw_stage].wait(raw_phase)
            raw_index = advance(raw_index, cfg.smem_raw_stages)

            # ---- state stage, right key half: pack, publish, fp32 decay --------------
            bars.mb_state_acc_cg1_done[state_update_index.idx].wait(state_update_index.phase)
            state_update_index = advance(state_update_index, cfg.smem_decay_stages)
            state_vecs = []
            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                state_vecs.append(
                    nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + state_col_id + i * 16, cutlass.Float32),
                        num=16,
                    )
                )

            packed_blocks = []
            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                packed_state = cute.make_rmem_tensor((8,), cutlass.Int32)
                for packed_col in cutlass.range_constexpr(8):
                    packed_state[packed_col] = fp32_to_fp16(
                        state_vecs[i - state_blocks_per_half][2 * packed_col],
                        state_vecs[i - state_blocks_per_half][2 * packed_col + 1],
                        dtype=cfg.io_dtype,
                    )
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(row_lo_addr + packed_col_id + i * 8, cutlass.Int8),
                    packed_state.load(),
                )
                packed_blocks.append(packed_state)

            # ---- fp32 decay of the right key half: state *= exp2(g last) -------------
            bars.mb_gate_exchange_ready[raw_stage].wait(raw_phase)
            sGate_exchange_ptr = smem_data_ptr(sGate_exchange_raw) + (
                cum_chunk % cfg.gate_exchange_stages
            ) * (cfg.d_k * cfg.b_t)
            scaled_blocks = []
            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                scaled = []
                for scale_group in cutlass.range_constexpr(4):
                    scale_dim = i * 16 + scale_group * 4
                    scale_segment = scale_dim // 32
                    scale_idx = (
                        scale_segment * (cfg.b_t * 32)
                        + (cfg.b_t - 1) * 32
                        + swizzle_xor_128b(
                            cfg.b_t - 1 ^ scale_segment,
                            scale_dim - scale_segment * 32,
                            elem_bytes=4,
                        )
                    )
                    scale_frag = (sGate_exchange_ptr + scale_idx).load(count=4, alignment=16)
                    for t in cutlass.range_constexpr(2):
                        s0, s1 = fmul2(
                            state_vecs[i - state_blocks_per_half][scale_group * 4 + 2 * t],
                            state_vecs[i - state_blocks_per_half][scale_group * 4 + 2 * t + 1],
                            scale_frag[2 * t],
                            scale_frag[2 * t + 1],
                        )
                        scaled += [s0, s1]
                scaled_blocks.append(scaled)
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_state_input_cg1_ready.arrive()
            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(row_lo_addr + state_col_id + i * 16, cutlass.Float32),
                    cutlass.Vector.from_elements(
                        tuple(scaled_blocks[i - state_blocks_per_half]), cutlass.Float32
                    ),
                )

            # ---- checkpoint row, right key half: the packed operand words ------------
            if cutlass.const_expr(cfg.enable_checkpoints):
                checkpoint_row = checkpoint_phase == 0
                checkpoint_row = checkpoint_row and chunk_idx >= write_start
                if checkpoint_row:
                    checkpoint_row_idx = checkpoint_row_count
                    checkpoint_row_stage = checkpoint_row_idx % cutlass.Int32(
                        cfg.smem_checkpoint_stages
                    )
                    bars.mb_checkpoint_tmastg_done[checkpoint_row_stage].wait(
                        (
                            checkpoint_row_idx // cutlass.Int32(cfg.smem_checkpoint_stages)
                            + cutlass.Int32(1)
                        )
                        % 2
                    )
                    checkpoint_row_addr = checkpoint_row_stage * (cfg.d_k * cfg.d_v)
                    if state_row_valid:
                        for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                            for word_group in cutlass.range_constexpr(2):
                                dk = i * 16 + word_group * 8
                                row_addr = (
                                    checkpoint_row_addr
                                    + (dk // 64) * (cfg.d_v * 64)
                                    + value_dim * 64
                                    + swizzle_xor_128b(value_dim, dk % 64, elem_bytes=2)
                                )
                                (sCheckpoint_ptr + row_addr).store(
                                    cutlass.Vector.from_elements(
                                        tuple(
                                            packed_blocks[i - state_blocks_per_half][
                                                word_group * 4 + t
                                            ]
                                            for t in range(4)
                                        ),
                                        cutlass.Int32,
                                    ).bitcast(cfg.io_dtype),
                                    alignment=16,
                                )
                    nvvm.fence_proxy("async.shared", space="cta")
                    if nvvm.elect_sync():
                        bars.mb_checkpoint_tmastg_ready[checkpoint_row_stage].arrive()
                    checkpoint_row_count = checkpoint_row_count + cutlass.Int32(1)
                checkpoint_phase = checkpoint_phase + cutlass.Int32(1)
                if checkpoint_phase == checkpoint_chunks:
                    checkpoint_phase = cutlass.Int32(0)

            # ---- Y stage: Y = Beta * (V - k state) -----------------------------------
            if cutlass.const_expr(cfg.v_is_zero):
                zero_word = opaque_i32_zero()
                raw_v_frag_lo = [zero_word for _ in range(4)]
                if cutlass.const_expr(cfg.d_v == 128):
                    raw_v_frag_hi = [zero_word for _ in range(4)]
            else:
                raw_v_frag_lo = nvvm.ldmatrix(
                    sV_ptr + v_swizzle_off_lo,
                    4,
                    nvvm.MMALayout.COL,
                )
                if cutlass.const_expr(cfg.d_v == 128):
                    raw_v_frag_hi = nvvm.ldmatrix(
                        sV_ptr + v_swizzle_off_hi,
                        4,
                        nvvm.MMALayout.COL,
                    )

            # ---- read k state acc ----------------------------------------------------
            bars.mb_state_k_acc_ready.wait(state_k_acc_index.phase)
            state_k_vec_lo = nvvm.tcgen05_ld(
                "16x256b", nvvm.make_tmem_ptr(row_lo_addr + statek_col_id, cutlass.Float32), num=2
            )
            if cutlass.const_expr(cfg.d_v == 128):
                state_k_vec_hi = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(row_hi_addr + statek_col_id, cutlass.Float32),
                    num=2,
                )

            # FP32 beta per token pair; the residual v - k state is subtracted and scaled in FP32
            # and packed once for the MMA (packing the residual lost it against a large
            # contraction).
            beta_regs = cute.make_rmem_tensor((8,), cutlass.Float32)
            for reg_idx in cutlass.range_constexpr(4):
                token0 = ((reg_idx // 2) * 4 + (lane_idx & 3)) * 2
                beta_regs[2 * reg_idx] = (sBeta_ptr + token0).load().to(cutlass.Float32)
                beta_regs[2 * reg_idx + 1] = (sBeta_ptr + token0 + 1).load().to(cutlass.Float32)
            y_input_pack_lo = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                lo, hi = 2 * reg_idx, 2 * reg_idx + 1
                y_input_pack_lo[reg_idx], _, _ = beta_residual_f16x2(
                    raw_v_frag_lo[reg_idx],
                    beta_regs[lo],
                    beta_regs[hi],
                    state_k_vec_lo[lo],
                    state_k_vec_lo[hi],
                    dtype=cfg.io_dtype,
                )

            y_input_pack_hi = cute.make_rmem_tensor((4,), cutlass.Int32)
            if cutlass.const_expr(cfg.d_v == 128):
                for reg_idx in cutlass.range_constexpr(4):
                    lo, hi = 2 * reg_idx, 2 * reg_idx + 1
                    y_input_pack_hi[reg_idx], _, _ = beta_residual_f16x2(
                        raw_v_frag_hi[reg_idx],
                        beta_regs[lo],
                        beta_regs[hi],
                        state_k_vec_hi[lo],
                        state_k_vec_hi[hi],
                        dtype=cfg.io_dtype,
                    )

            nvvm.tcgen05_st(
                "16x128b",
                nvvm.make_tmem_ptr(row_lo_addr + y_input_col_id, cutlass.Int8),
                y_input_pack_lo.load(),
            )
            if cutlass.const_expr(cfg.d_v == 128):
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(row_hi_addr + y_input_col_id, cutlass.Int8),
                    y_input_pack_hi.load(),
                )
            nvvm.tcgen05_wait("store")
            state_k_acc_index = advance(state_k_acc_index, 1)
            if nvvm.elect_sync():
                bars.mb_y_input_ready.arrive()
                bars.mb_beta_done[raw_stage].arrive()
                bars.mb_raw_done[raw_stage].arrive()

            # ---- U stage: u acc TMEM -> packed b16 U input TMEM ----------------------
            bars.mb_u_acc_ready.wait(u_acc_index.phase)
            u_acc_vals = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(u_acc_addr, cutlass.Float32),
                num=cfg.b_t,
            )

            u_input_pack = cute.make_rmem_tensor(((cfg.b_t // 2),), cutlass.Int32)
            for packed_col in cutlass.range_constexpr(cfg.b_t // 2):
                token0 = packed_col * 2
                token1 = token0 + 1
                u_input_pack[packed_col] = fp32_to_fp16(
                    u_acc_vals[token0], u_acc_vals[token1], dtype=cfg.io_dtype
                )

            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(u_input_addr, cutlass.Int8),
                u_input_pack.load(),
            )
            nvvm.tcgen05_wait("store")
            u_acc_index = advance(u_acc_index, 1)
            if nvvm.elect_sync():
                bars.mb_u_input_ready.arrive()

        if num_chunks_tile > 0:
            bars.mb_state_acc_cg1_done[state_update_index.idx].wait(state_update_index.phase)
            state_update_index = advance(state_update_index, cfg.smem_decay_stages)

        owns_final = write_end == batch_num_chunks

        # ---- final state store: TMEM -> GMEM -----------------------------------------
        if cutlass.const_expr(mState_out is not None):
            if batch_seqlen > 0:
                if owns_final:
                    state_vw = 16 // (mState_out.element_type.width // 8)
                    state_dst = (
                        mState_out.iterator + mState_out.layout((batch_idx, head_o, value_dim, 0))
                    ).raw_ptr()
                    for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                        loaded = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                row_lo_addr
                                + (tmem_col + cfg.tmem_state_acc_offset + key_block_start),
                                cutlass.Float32,
                            ),
                            num=32,
                        )

                        if state_row_valid:
                            for g in cutlass.range_constexpr(32 // state_vw):
                                (state_dst + key_block_start + g * state_vw).store(
                                    cutlass.Vector.from_elements(
                                        tuple(
                                            loaded[g * state_vw + t].to(mState_out.element_type)
                                            for t in range(state_vw)
                                        ),
                                        mState_out.element_type,
                                    ),
                                    alignment=16,
                                )
            else:
                if state_row_valid:
                    for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                        for col in cutlass.range_constexpr(32):
                            key_dim = key_block_start + col
                            if cutlass.const_expr(mState_init is not None):
                                mState_out[batch_idx, head_o, value_dim, key_dim] = mState_init[
                                    batch_idx, head_o, value_dim, key_dim
                                ]
                            elif cutlass.const_expr(cfg.seed_identity):
                                diag = (
                                    cutlass.Float32(1.0)
                                    if value_dim == key_dim
                                    else cutlass.Float32(0.0)
                                )
                                mState_out[batch_idx, head_o, value_dim, key_dim] = diag.to(
                                    mState_out.element_type
                                )
                            else:
                                mState_out[batch_idx, head_o, value_dim, key_dim] = (
                                    cutlass.Float32(0.0).to(mState_out.element_type)
                                )
        cum_chunk_base += num_chunks_tile
        tile_idx, scheduler_state = scheduler_next_tile(
            cfg, bars, sScheduler, scheduler_state, elect_one
        )

    if nvvm.elect_sync():
        bars.mb_tmem_done[0].arrive()


@cute.jit
def build_descs_body(
    widx,
    base_k,
    base_v,
    base_gate,
    base_checkpoint,
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    state_checkpoints: cute.Tensor | None,
    n_batch: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
) -> None:
    """Per-batch descriptor-array build, one warp per array. Runs inside the
    prologue kernel after its order pass; warps past the array count fall
    through the widx guards."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_words_k = cute.make_tensor(
        desc_workspace.iterator, cute.make_layout((arr_words,), stride=(1,))
    )
    desc_words_v = cute.make_tensor(
        desc_workspace.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    desc_words_gate = cute.make_tensor(
        desc_workspace.iterator + 2 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    desc_words_checkpoint = cute.make_tensor(
        desc_workspace.iterator + 3 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )

    if widx == 0:
        if nvvm.elect_sync():
            emit_seq_descs(base_k, desc_words_k, cu_seqlens, k, n_batch, 2)
            nvvm.fence_proxy_release(
                nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
            )
    if widx == 1:
        if nvvm.elect_sync():
            emit_seq_descs(base_v, desc_words_v, cu_seqlens, v, n_batch, 2)
            nvvm.fence_proxy_release(
                nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
            )
    if widx == 2:
        if nvvm.elect_sync():
            emit_seq_descs(base_gate, desc_words_gate, cu_seqlens, gate, n_batch, 2)
            nvvm.fence_proxy_release(
                nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
            )
    if cutlass.const_expr(state_checkpoints is not None):
        if widx == 3:
            if nvvm.elect_sync():
                emit_checkpoint_seq_descs(
                    base_checkpoint,
                    desc_words_checkpoint,
                    cu_seqlens,
                    state_checkpoints,
                    n_batch,
                    checkpoint_every_n,
                    2,
                )
                nvvm.fence_proxy_release(
                    nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
                )


@cute.kernel
def frost_kda_recompute_prologue(
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    gen_intervals: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    base_k: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_v: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_gate: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_checkpoint: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    state_checkpoints: cute.Tensor | None,
    mStaging: cute.Tensor | None,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor | None,
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
            )
    else:
        build_descs_body(
            widx,
            base_k,
            base_v,
            base_gate,
            base_checkpoint,
            desc_workspace,
            cu_seqlens,
            k,
            v,
            gate,
            state_checkpoints,
            n_batch,
            checkpoint_every_n,
        )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    gen_intervals: cutlass.Constexpr[bool],
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    state_checkpoints: cute.Tensor | None,
    cu_seqlens: cute.Tensor,
    work_item_staging: cute.Tensor | None,
    work_count: cute.Tensor,
    work_items: cute.Tensor | None,
    scheduler_all: cute.Tensor | None,
    tensormap_workspace: cute.Tensor,
    checkpoint_every_n: cutlass.Int32,
    seed_span_chunks: cutlass.Int32,
    stream: cuda_driver.CUstream,
):
    """One-launch prologue: LPT-order the work items (when ``run_order``) and
    build the per-batch K/V/Gate/checkpoint TMA-descriptor arrays into
    ``tensormap_workspace``."""
    h_k = k.shape[1]
    h_v = v.shape[1]
    ho = gate.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    d_k = k.shape[2]
    d_v = v.shape[2]
    bytes_per_element = io_dtype.width // 8
    box_elems = 128 // bytes_per_element
    seqlen = k.shape[0]

    k_headed = cute.make_tensor(
        k.iterator, cute.make_layout((d_k, h_k, seqlen), stride=(1, k.stride[1], k.stride[0]))
    )
    v_headed = cute.make_tensor(
        v.iterator, cute.make_layout((d_v, h_v, seqlen), stride=(1, v.stride[1], v.stride[0]))
    )
    gate_headed = cute.make_tensor(
        gate.iterator,
        cute.make_layout((d_k, ho, seqlen), stride=(1, gate.stride[1], gate.stride[0])),
    )

    swizzle = cuda.TensorMapSwizzle.s128b
    base_k = cuda.create_tensor_map_tiled_from_view(
        k_headed, box_dims=(box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    base_v = cuda.create_tensor_map_tiled_from_view(
        v_headed, box_dims=(box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    gate_box_elems = 128 // (gate.element_type.width // 8)
    base_gate = cuda.create_tensor_map_tiled_from_view(
        gate_headed, box_dims=(gate_box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )

    base_checkpoint = base_gate
    if cutlass.const_expr(state_checkpoints is not None):
        d_v_state = state_checkpoints.shape[2]
        d_k_state = state_checkpoints.shape[3]
        checkpoint_view = cute.make_tensor(
            state_checkpoints.iterator,
            cute.make_layout(
                (d_k_state, d_v_state, state_checkpoints.shape[0], ho),
                stride=(
                    state_checkpoints.stride[3],
                    state_checkpoints.stride[2],
                    state_checkpoints.stride[0],
                    state_checkpoints.stride[1],
                ),
            ),
        )
        base_checkpoint = cuda.create_tensor_map_tiled_from_view(
            checkpoint_view,
            box_dims=(box_elems, d_v_state, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=swizzle,
        )
    frost_kda_recompute_prologue(
        run_order,
        order_gen,
        gen_intervals,
        b_t,
        base_k,
        base_v,
        base_gate,
        base_checkpoint,
        tensormap_workspace,
        cu_seqlens,
        k,
        v,
        gate,
        state_checkpoints,
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
    raw_gate: cute.Tensor,
    a_log: cute.Tensor | None,
    dt_bias: cute.Tensor | None,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    initial_state: cute.Tensor | None,
    final_state: cute.Tensor | None,
    seed_state_checkpoints: cute.Tensor | None,
    work_items: cute.Tensor | None,
    work_count: cute.Tensor | None,
    scheduler_counter: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    checkpoint_every_n_tokens: cutlass.Int32,
    seed_every_n_tokens: cutlass.Int32,
    stream,
) -> None:
    heads_out = cutlass.Int32(raw_gate.shape[1])
    k_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(k.shape[1]))
    v_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(v.shape[1]))
    if cutlass.const_expr(cfg.v_is_zero):
        v_ratio = cute.FastDivmodDivisorV2(cutlass.Int32(1))
    num_sequences = cu_seqlens.shape[0] - 1
    gate_exchange_elements = 0 if cfg.gate_dtype == cutlass.Float32 else cfg.gate_exchange_cosize
    checkpoint_elements = (
        cfg.smem_checkpoint_stages * cfg.d_k * cfg.d_v if cfg.enable_checkpoints else 0
    )

    @cute.struct
    class SharedStorage:
        k_decay: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.k_decay_cosize], cfg.buffer_align_bytes
        ]
        k_restore: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.k_restore_cosize], cfg.buffer_align_bytes
        ]
        intermediate: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.intermediate_cosize], cfg.buffer_align_bytes
        ]
        k: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.k_cosize], cfg.buffer_align_bytes
        ]
        v: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.v_cosize], cfg.buffer_align_bytes
        ]
        gate: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, cfg.gate_cosize], 1024]
        gate_exchange: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, gate_exchange_elements], 1024
        ]
        k_inv: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.k_inv_cosize], cfg.buffer_align_bytes
        ]
        checkpoint: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, checkpoint_elements], cfg.buffer_align_bytes
        ]

    # ---- launch ----------------------------------------------------------------------
    grid_shape = (cfg.max_active_clusters, 1, 1)
    frost_kda_recompute(
        cfg,
        SharedStorage,
        k_ratio,
        v_ratio,
        tensormap_workspace,
        cutlass.Int32(num_sequences),
        k,
        v,
        raw_gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        initial_state,
        final_state,
        seed_state_checkpoints,
        work_items,
        work_count,
        scheduler_counter,
        checkpoint_every_n_tokens,
        seed_every_n_tokens,
    ).launch(
        grid=grid_shape,
        block=(cfg.threads_per_cta, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
        min_blocks_per_mp=1,
    )


@cute.kernel
def frost_kda_recompute(
    cfg: cutlass.Constexpr,
    shared_type: cutlass.Constexpr,
    k_ratio: cute.FastDivmodDivisorV2,
    v_ratio: cute.FastDivmodDivisorV2,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mGate: cute.Tensor,
    mA_log: cute.Tensor | None,
    mDt_bias: cute.Tensor | None,
    mBeta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    mState_init: cute.Tensor | None,
    mState_out: cute.Tensor | None,
    mSeedCheckpoints: cute.Tensor | None,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor,
    checkpoint_every_n_tokens: cutlass.Int32,
    seed_every_n_tokens: cutlass.Int32,
) -> None:
    """BT=16 KDA recompute (state/checkpoints-only) persistent kernel body: every warp
    role runs a tile-scheduler loop over the tiles."""
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
    desc_k_base = desc_base_words
    desc_v_base = desc_base_words + arr_words
    desc_gate_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_checkpoint_base = desc_base_words + cutlass.Int32(3) * arr_words

    SMEM = cutlass.AddressSpace.smem
    bars = make_bars(cfg)
    sTmem_base = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=4)
    sScheduler = cutlass.Array(cutlass.Int32, cfg.scheduler_stages, space=SMEM, alignment=16)
    storage = SmemAllocator().allocate(shared_type)
    sK_decay_raw = storage.k_decay.get_tensor(cute.make_layout((cfg.k_decay_cosize,)))
    sK_restore_raw = storage.k_restore.get_tensor(cute.make_layout((cfg.k_restore_cosize,)))
    sIntermediate_raw = storage.intermediate.get_tensor(
        cute.make_layout((cfg.intermediate_cosize,))
    )
    sK_raw = storage.k.get_tensor(cute.make_layout((cfg.k_cosize,)))
    sV_raw = storage.v.get_tensor(cute.make_layout((cfg.v_cosize,)))
    sGate_raw = storage.gate.get_tensor(cute.make_layout((cfg.gate_cosize,)))
    if cutlass.const_expr(cfg.gate_dtype == cutlass.Float32):
        sGate_load_ptr = smem_data_ptr(sGate_raw)
        sGate_exchange_raw = sGate_raw
    else:
        sGate_load_ptr = cute.make_ptr(
            cfg.gate_dtype, smem_data_ptr(sGate_raw).toint(), mem_space=SMEM, assumed_align=1024
        )
        sGate_exchange_raw = storage.gate_exchange.get_tensor(
            cute.make_layout((cfg.gate_exchange_cosize,))
        )
    sK_inv_raw = storage.k_inv.get_tensor(cute.make_layout((cfg.k_inv_cosize,)))
    sBeta_raw = cutlass.Array(
        cutlass.Float32, cfg.beta_cosize, space=SMEM, alignment=cfg.buffer_align_bytes
    )
    sCheckpoint_raw = (
        storage.checkpoint.get_tensor(
            cute.make_layout((cfg.smem_checkpoint_stages * cfg.d_k * cfg.d_v,))
        )
        if cutlass.const_expr(cfg.enable_checkpoints)
        else sV_raw
    )
    sK_decay = SmemTile(
        base=sK_decay_raw,
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
            for stage in cutlass.range_constexpr(cfg.smem_raw_stages):
                bars.mb_raw_ready[stage].init()
                bars.mb_beta_ready[stage].init()
                bars.mb_beta_done[stage].init()
                bars.mb_raw_done[stage].init()
                bars.mb_gate_exchange_ready[stage].init()
    elif warp_idx == cfg.tcgen05_mma_warp_id:
        if elect_one:
            bars.mb_state_k_acc_ready.init()
            bars.mb_u_acc_ready.init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_state_acc_cg0_done[stage].init()
                bars.mb_state_acc_cg1_done[stage].init()
            bars.mb_state_input_cg1_ready.init()
            bars.mb_state_input_cg0_ready.init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_decay_tcgen05_done[stage].init()
                bars.mb_decay_register_mma_done[stage].init()
                bars.mb_k_restore_done[stage].init()
            bars.mb_y_input_ready.init()
            bars.mb_u_input_ready.init()
            bars.mb_tmem_done[0].init()
    elif warp_idx == cfg.register_mma_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_intermediate_stages):
                bars.mb_t_inv_ready[stage].init()
                bars.mb_t_inv_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.qk_scale_ready_stages):
                bars.mb_qk_scale_ready[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_k_decay_inv_cg0_ready[stage].init()
    elif warp_idx == cfg.epilogue_warp_id:
        if elect_one:
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
            sK_raw,
            sV_raw,
            sGate_raw,
            desc_k_base,
            desc_v_base,
            desc_gate_base,
            bars,
            k_ratio=k_ratio,
            v_ratio=v_ratio,
        )
    elif warp_idx == cfg.register_mma_warp_id:
        register_mma_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sScheduler,
            lane_idx,
            sK_inv_raw,
            sIntermediate_raw,
            sBeta_raw,
            sK_decay_raw,
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
            sTmem_base,
            sIntermediate,
            sK_decay,
            sK_restore_trans,
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
            sCheckpoint_raw,
            desc_checkpoint_base,
            checkpoint_every_n_tokens,
            bars,
        )
    elif (
        warp_idx >= cfg.compute_group_0_warp_ids[0]
        and warp_idx <= cfg.compute_group_0_warp_ids[-1]
    ):
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
            mA_log,
            mDt_bias,
            sK_inv_raw,
            sGate_exchange_raw,
            sGate_load_ptr,
            mBeta,
            sBeta_raw,
            sK_raw,
            sK_decay_raw,
            sK_restore_raw,
            sCheckpoint_raw,
            sTmem_base,
            checkpoint_every_n_tokens,
            bars,
        )
    elif (
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
            sScheduler,
            lane_idx,
            sTmem_base,
            warp_idx,
            mState_out,
            mState_init,
            mSeedCheckpoints,
            sBeta_raw,
            sV_raw,
            sCheckpoint_raw,
            sGate_exchange_raw,
            checkpoint_every_n_tokens,
            seed_every_n_tokens,
            bars,
        )


@dataclass(frozen=True)
class KdaRecomputeCfg:
    """Kernel cfg (fixed BT=16 schedule constants; derived TMEM column offsets
    and SMEM buffer cosizes are stamped by ``build_cfg``; per-stage sizes are
    inlined at the use sites).  Passed ``cfg``-first (a ``cutlass.Constexpr``)
    into ``host`` / ``kernel`` and every warp body."""

    io_dtype: type[cutlass.Numeric]
    state_dtype: type[cutlass.Numeric]
    gate_dtype: type[cutlass.Numeric]
    use_initial_state: bool
    store_final_state: bool
    enable_checkpoints: bool
    seed_checkpoints: bool
    l2norm: bool
    safe_gate: bool
    gate_scale_log2: float
    log_gate: bool
    beta_sigmoid: bool
    allow_neg_eigval: bool
    max_active_clusters: int
    d_k: int
    d_v: int
    seed_identity: bool = False
    v_is_zero: bool = False
    scheduler_stages: int = CFG.SMEM_SCHEDULER_STAGES

    compute_group_0_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_1_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    register_mma_warp_id: int = CFG.REGISTER_MMA_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    tma_warp_id: int = CFG.TMA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    b_t: int = CFG.B_T
    threads_per_warp: int = CFG.THREADS_PER_WARP
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES
    threads_per_cta: int = 0
    cg0_group_count: int = 2
    cg0_warps_per_group: int = 4
    cg0_threads_per_group: int = 0
    cg0_group_sync_barrier_base_id: int = 1  # CG0 group g syncs on named-barrier id 1 + g
    cg0_tile_entry_barrier_id: int = 5  # CG0-wide (both groups) work-item entry sync
    tmem_user_threads: int = 0
    tmem_lifecycle_barrier_id: int = 3
    scheduler_consumer_warps: int = 0
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_other: int = CFG.NUM_REGS_OTHER

    # ---- SMEM / TMEM ring stage counts -----------------------------------------------
    smem_raw_stages: int = CFG.SMEM_RAW_STAGES
    smem_checkpoint_stages: int = 1
    smem_decay_stages: int = CFG.SMEM_DECAY_STAGES
    smem_intermediate_stages: int = CFG.SMEM_INTERMEDIATE_STAGES
    qk_scale_ready_stages: int = CFG.QK_SCALE_READY_STAGES

    # ---- TMEM column offsets (state doubles as the final state acc) ------------------
    tmem_state_acc_offset: int = 0
    tmem_state_input_offset: int = 0
    tmem_state_k_acc_offset: int = 0
    tmem_u_acc_offset: int = 0
    tmem_y_input_offset: int = 0
    tmem_u_input_offset: int = 0

    # ---- SMEM buffer cosizes ---------------------------------------------------------
    k_cosize: int = 0
    v_cosize: int = 0
    gate_cosize: int = 0
    gate_stage_elems: int = 0
    gate_exchange_stages: int = 0
    gate_exchange_cosize: int = 0
    beta_cosize: int = 0
    k_inv_cosize: int = 0
    k_decay_cosize: int = 0
    k_restore_cosize: int = 0

    # ---- TMA transaction bytes per stage ---------------------------------------------
    tma_k_bytes: int = 0
    tma_v_bytes: int = 0
    tma_gate_bytes: int = 0
    intermediate_cosize: int = 0


def build_cfg(
    io_dtype: type[cutlass.Numeric],
    state_dtype: type[cutlass.Numeric],
    gate_dtype: type[cutlass.Numeric],
    *,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
    seed_checkpoints: bool = False,
    l2norm: bool,
    safe_gate: bool,
    gate_scale_log2: float,
    log_gate: bool = True,
    beta_sigmoid: bool,
    allow_neg_eigval: bool,
    max_active_clusters: int,
    seed_identity: bool = False,
    v_is_zero: bool = False,
    d_k: int,
    d_v: int,
) -> KdaRecomputeCfg:
    """Build the per-compile KdaRecomputeCfg (io_dtype in {Float16, BFloat16});
    fills the derived TMEM column offsets and SMEM buffer cosizes."""
    cfg = KdaRecomputeCfg(
        io_dtype=io_dtype,
        state_dtype=state_dtype,
        gate_dtype=gate_dtype,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        seed_checkpoints=seed_checkpoints,
        l2norm=l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        log_gate=log_gate,
        beta_sigmoid=beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        max_active_clusters=max_active_clusters,
        seed_identity=seed_identity,
        v_is_zero=v_is_zero,
        d_k=d_k,
        d_v=d_v,
    )
    if enable_checkpoints:
        cfg = replace(cfg, smem_raw_stages=6, smem_checkpoint_stages=2)
    if cfg.smem_raw_stages % 2 != 0:
        raise ValueError(
            "smem_raw_stages must be even: the CG0 ping-pong groups alias parity waits on odd "
            "rings"
        )
    if cfg.cg0_warps_per_group != len(cfg.compute_group_1_warp_ids):
        raise ValueError(
            "the state halves are packed by one CG0 group and by CG1: their warp counts must match"
        )

    role_ids = (
        *cfg.compute_group_0_warp_ids,
        *cfg.compute_group_1_warp_ids,
        cfg.register_mma_warp_id,
        cfg.tcgen05_mma_warp_id,
        cfg.tma_warp_id,
        cfg.epilogue_warp_id,
    )
    if len(cfg.compute_group_0_warp_ids) != cfg.cg0_group_count * cfg.cg0_warps_per_group:
        raise ValueError(
            "compute group 0 must hold cg0_group_count groups of cg0_warps_per_group warps"
        )
    scheduler_consumer_warps = _validate_roles(
        role_ids,
        (
            *(cfg.cg0_group_sync_barrier_base_id + g for g in range(cfg.cg0_group_count)),
            cfg.cg0_tile_entry_barrier_id,
            cfg.tmem_lifecycle_barrier_id,
        ),
        16,
    )
    raw, b_t, d_k, d_v = cfg.smem_raw_stages, cfg.b_t, cfg.d_k, cfg.d_v
    io_bytes, gate_bytes = cfg.io_dtype.width // 8, cfg.gate_dtype.width // 8
    tmem_state_input_offset = cfg.tmem_state_acc_offset + d_k
    tmem_state_k_acc_offset = tmem_state_input_offset + (d_k // 2)
    tmem_u_acc_offset = tmem_state_k_acc_offset + b_t
    tmem_y_input_offset = tmem_u_acc_offset + b_t
    tmem_u_input_offset = tmem_y_input_offset + (b_t // 2)
    assert (tmem_u_input_offset + (b_t // 2)) <= 512
    gate_exchange_stages = raw if gate_dtype == cutlass.Float32 else 4
    gate_exchange_cosize = 0 if gate_dtype == cutlass.Float32 else gate_exchange_stages * d_k * b_t
    return replace(
        cfg,
        threads_per_cta=16 * cfg.threads_per_warp,
        scheduler_consumer_warps=scheduler_consumer_warps,
        cg0_threads_per_group=cfg.cg0_warps_per_group * cfg.threads_per_warp,
        tmem_user_threads=(
            1 + len(cfg.compute_group_1_warp_ids) + len(cfg.compute_group_0_warp_ids)
        )
        * cfg.threads_per_warp,
        tmem_state_input_offset=tmem_state_input_offset,
        tmem_state_k_acc_offset=tmem_state_k_acc_offset,
        tmem_u_acc_offset=tmem_u_acc_offset,
        tmem_y_input_offset=tmem_y_input_offset,
        tmem_u_input_offset=tmem_u_input_offset,
        k_cosize=raw * d_k * b_t,
        v_cosize=raw * d_v * b_t,
        gate_cosize=raw * d_k * b_t * gate_bytes // 4,
        gate_stage_elems=d_k * b_t,
        gate_exchange_stages=gate_exchange_stages,
        gate_exchange_cosize=gate_exchange_cosize,
        beta_cosize=raw * b_t,
        k_inv_cosize=cfg.smem_decay_stages * b_t * d_k,
        k_decay_cosize=cfg.smem_decay_stages * d_k * b_t,
        k_restore_cosize=cfg.smem_decay_stages * d_k * b_t,
        intermediate_cosize=cfg.smem_intermediate_stages * 2 * b_t * b_t,
        tma_k_bytes=d_k * b_t * io_bytes,
        tma_v_bytes=d_v * b_t * io_bytes,
        tma_gate_bytes=d_k * b_t * gate_bytes,
    )


TENSORMAP_DESC_ARRAYS = 4  # per-batch runtime TMA descriptors: K, V, Gate, state_checkpoints


# ---------------------------------------------------------------------------


frost_kda_recompute_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_kda_recompute.set_name_prefix("cudnn", remove_cutlass_symbol=False)
