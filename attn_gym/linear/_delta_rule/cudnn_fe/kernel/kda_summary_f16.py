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
# attn_gym.linear._delta_rule.cudnn_fe; restyled register tensors (cute.make_rmem_tensor), one
# SharedStorage struct, smem_data_ptr, swizzle_box_offset_{128b,32b} swizzle offsets, and call
# sites of the pruned tile_dsl helpers; upstream standalone host (get_compiled_cache, compile,
# chunk_kda_summary, run_summary) replaced by KdaSummaryOp and @jit_cache fake-tensor TVM-FFI
# compiles of the kernel and prologue (+int64 variants); frozen cfg with launch-contract,
# warp-role, named-barrier, and TMEM/SMEM validation; FP32 delta residual until the MMA pack;
# upstream-only constexpr knobs pruned (safe_gate, A_log/dt_bias, beta sigmoid, allow_neg_eigval,
# Q/K L2 norm); Ruff formatting.

"""
Chunked Kimi Delta Attention (KDA) fused state-summary kernel for SM100 / SM103 / SM107 (Cutlass
primitives): the BT = 16 recompute pipeline run twice per chunk on one K / Gate / Beta stream,
chain H (the state from zero or initial_state, consuming V) and chain M (the identity seed with V =
0, the piece transition), one persistent CTA per (piece, head).

Algorithm overview (per chunk c, tokens [cC, (c+1)C), for each chain X in {H, M}):
  Inputs : K[BT,DK], V[BT,DV] (chain H only), Gate[BT,DK] (per-channel gate), Beta[BT] (scalar LR)
  State  : S_X[DK,DV]  (recurrent state, held in TMEM, fp32 carry; H seeded from zero / initial_state, M from I)

  Preprocessing (compute group 0, two ping-pong groups of four warps):
    g[t,d]           = sum_{l=0}^{t} log2(Gate_ld)             per-channel cumulative log2 of gates (log)
    K decay[t,d]     = K[t,d] * exp2(+g[t,d])                    (KK A operand, K*state B operand)
    K inv[t,d]       = K[t,d] * exp2(-g[t,d])                    (KK / A tile B operand)
    K restore[t,d]   = K[t,d] * exp2(g[BT-1,d] - g[t,d])         (state update B operand)

  KK (register MMA) : W_kk[BT,BT] = K decay @ K inv^T;  L = Beta * tril(W_kk, -1)   (shared by both chains)
  T_inv (register MMA) : T_inv = (I + L)^-1 blockwise, 4x4 diagonal blocks then the 4 -> 8 and 8 ->
                         16 corrections
  K*state GEMM   : KS_X[BT,DV] = K decay @ S_X
  U GEMM         : U_X[BT,DV]  = T_inv @ Y_X,  Y_H = Beta * (V - KS_H), Y_M = Beta * (0 - KS_M)
  KV update GEMM : S_upd_X[DK,DV] = K restore^T @ U_X   (H then M per stage, so chain M trails H by one stage)

  Epilogue:
    S_X       = exp2(g[BT-1,:]) .* S_X + S_upd_X
    output_state = S_H, output_transition = S_M        (stored domain M_buf = M^T; X_final = X_init @ M_buf + X_H)

An item seeds when compute_start == 0 and stores when write_end == batch_num_chunks; an empty item
passes initial_state (or zero) through as H and writes M = I.

SMEM layout (stage counts live in kda_summary_config.py; sizes at DK = DV = 128, bf16 io, fp32
Gate):
  Buffer                       Size (B)  Stages
  K / V (raw)                  2 x 4096       8
  Gate (raw)                       8192       8    <-- bf16 Gate: 4096 plus a 4-stage fp32 exchange ring
  Beta                               64       8
  K decay / K inv / K restore  3 x 4096       2
  T_inv (intermediate)             1024       2
  scheduler ticket ring               4       8    <-- next-tile publish ring

TMEM layout (512 columns allocated; chain H at 0, chain M at 240):
  Buffer                  Cols
  state                   128     <-- DKxDV fp32 (doubles as the final state acc)
  state input              64     <-- f16 state staging (K*state A operand)
  K*state acc              16
  U acc                    16
  Y input                   8     <-- f16 packed
  U input                   8

Warp assignments (16 warps = 512 threads):
  warps 0-7     : compute group 0 - Gate prefix scan, shared operands, left key halves of both states (two ping-pong
                                    groups)
  warps 8-11    : compute group 1 - seeds, right key halves, Y / U inputs of both chains, H / M stores
  warp  12      : register-MMA warp - KK and T_inv of the even cumulative chunks
  warp  13      : MMA warp       - every tcgen05 GEMM of both chains; TMEM lifecycle
  warp  14      : TMA load warp  - loads K, V, Gate; stages Beta
  warp  15      : register-MMA twin - KK and T_inv of the odd cumulative chunks
"""

from dataclasses import dataclass, replace
from typing import NamedTuple

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.experimental.primitives as nvvm
from cutlass import cute
from cutlass.experimental import cuda

from attn_gym._backends.cute import compile_tvm_ffi, jit_cache
from attn_gym._backends.cute.compat import SmemAllocator

from ..common.blockwise_inverse import invert_unit_lower_16x16_fragments
from ..common.launch import validate_kernel_domain, validate_named_barriers, validate_warp_roles
from ..common.split_k import (
    ORDER_CAPACITY,
    ORDER_ELEMENTS,
    ORDER_THREADS,
    decode_work_item,
    order_body,
)
from ..common.thd import TENSOR_MAP_QWORDS, emit_seq_descs
from ..common.tvm_ffi import (
    WORK_ITEM_FIELDS,
    make_compact_signature_tensor,
    make_dynamic_signature_tensor,
)
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
    fmul2,
    fp32_to_fp16,
    opaque_f32_zero,
    opaque_i32,
    opaque_i32_zero,
)
from ..tile_dsl.swizzle import swizzle_box_offset_32b, swizzle_box_offset_128b, swizzle_xor_128b
from ..tile_dsl.tma import tma_load_tile, tma_tensormap_acquire
from .kda_summary_config import CFG

USE_PDL = True
STATE_DIMS = (64, 128)

LOG2_E: float = 1.4426950408889634
DEFAULT_GATE_LOWER_BOUND: float = -5.0


class KdaSummaryBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring: the shared
    stream and operand rings, then each recurrence's own handoffs, chain H then chain M."""

    mb_raw_ready: MBarrier
    mb_raw_done: MBarrier
    mb_gate_exchange_ready: MBarrier

    mb_beta_ready: MBarrier
    mb_beta_done: MBarrier

    mb_t_inv_ready: MBarrier
    mb_t_inv_done: MBarrier
    mb_qk_scale_ready: MBarrier
    mb_k_decay_inv_cg0_ready: MBarrier
    mb_decay_tcgen05_done: MBarrier
    mb_decay_register_mma_done: MBarrier
    mb_k_restore_done: MBarrier

    mb_tmem_done: MBarrier

    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier

    mb_state_k_acc_h_ready: MBarrier
    mb_u_acc_h_ready: MBarrier
    mb_state_input_h_cg1_ready: MBarrier
    mb_state_input_h_cg0_ready: MBarrier
    mb_y_input_h_ready: MBarrier
    mb_u_input_h_ready: MBarrier
    mb_state_acc_h_cg0_done: MBarrier
    mb_state_acc_h_cg1_done: MBarrier
    mb_state_k_acc_m_ready: MBarrier
    mb_u_acc_m_ready: MBarrier
    mb_state_input_m_cg1_ready: MBarrier
    mb_state_input_m_cg0_ready: MBarrier
    mb_y_input_m_ready: MBarrier
    mb_u_input_m_ready: MBarrier
    mb_state_acc_m_cg0_done: MBarrier
    mb_state_acc_m_cg1_done: MBarrier


def make_bars(cfg) -> KdaSummaryBars:
    """KdaSummaryBars constructor."""

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=8)

    CG0_GROUP_WARPS = cfg.cg0_warps_per_group
    CG1_WARPS = len(cfg.compute_group_1_warp_ids)

    return KdaSummaryBars(
        mb_raw_ready=MBarrier(
            alloc(cfg.smem_raw_stages),
            spin=True,
            stages=cfg.smem_raw_stages,
            init_count=1,
            producer=Producer.TMA_LOAD,
        ),
        mb_raw_done=MBarrier(
            alloc(cfg.smem_raw_stages),
            spin=True,
            stages=cfg.smem_raw_stages,
            init_count=CG0_GROUP_WARPS + CG1_WARPS,
            producer=Producer.THREAD,
        ),
        mb_gate_exchange_ready=MBarrier(
            alloc(cfg.smem_raw_stages),
            spin=True,
            stages=cfg.smem_raw_stages,
            init_count=CG0_GROUP_WARPS,
            producer=Producer.THREAD,
        ),
        mb_beta_ready=MBarrier(
            alloc(cfg.smem_raw_stages),
            spin=True,
            stages=cfg.smem_raw_stages,
            init_count=1,
            producer=Producer.THREAD,
        ),
        mb_beta_done=MBarrier(
            alloc(cfg.smem_raw_stages),
            spin=True,
            stages=cfg.smem_raw_stages,
            init_count=1 + CG1_WARPS,
            producer=Producer.THREAD,
        ),
        mb_t_inv_ready=MBarrier(
            alloc(cfg.smem_intermediate_stages),
            spin=True,
            stages=cfg.smem_intermediate_stages,
            init_count=1,
            producer=Producer.THREAD,
        ),
        mb_t_inv_done=MBarrier(
            alloc(cfg.smem_intermediate_stages),
            spin=True,
            stages=cfg.smem_intermediate_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
        ),
        mb_qk_scale_ready=MBarrier(
            alloc(cfg.qk_scale_ready_stages),
            spin=True,
            stages=cfg.qk_scale_ready_stages,
            init_count=CG0_GROUP_WARPS,
            producer=Producer.THREAD,
        ),
        mb_k_decay_inv_cg0_ready=MBarrier(
            alloc(cfg.smem_decay_stages),
            spin=True,
            stages=cfg.smem_decay_stages,
            init_count=CG0_GROUP_WARPS,
            producer=Producer.THREAD,
        ),
        mb_decay_tcgen05_done=MBarrier(
            alloc(cfg.smem_decay_stages),
            spin=True,
            stages=cfg.smem_decay_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
        ),
        mb_decay_register_mma_done=MBarrier(
            alloc(cfg.smem_decay_stages),
            spin=True,
            stages=cfg.smem_decay_stages,
            init_count=1,
            producer=Producer.THREAD,
        ),
        mb_k_restore_done=MBarrier(
            alloc(cfg.smem_decay_stages),
            spin=True,
            stages=cfg.smem_decay_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
        ),
        mb_tmem_done=MBarrier(
            alloc(1), spin=True, stages=1, init_count=CG1_WARPS, producer=Producer.THREAD
        ),
        mb_scheduler_ready=MBarrier(
            alloc(cfg.scheduler_stages),
            spin=True,
            stages=cfg.scheduler_stages,
            init_count=1,
            producer=Producer.THREAD,
        ),
        mb_scheduler_done=MBarrier(
            alloc(cfg.scheduler_stages),
            spin=True,
            stages=cfg.scheduler_stages,
            init_count=cfg.threads_per_cta // cfg.threads_per_warp
            - 1,  # every warp but the TMA publisher
            producer=Producer.THREAD,
        ),
        mb_state_k_acc_h_ready=MBarrier(
            alloc(1), spin=True, stages=1, init_count=1, producer=Producer.MMA_COMMIT
        ),
        mb_u_acc_h_ready=MBarrier(
            alloc(1), spin=True, stages=1, init_count=1, producer=Producer.MMA_COMMIT
        ),
        mb_state_input_h_cg1_ready=MBarrier(
            alloc(1), spin=True, stages=1, init_count=CG1_WARPS, producer=Producer.THREAD
        ),
        mb_state_input_h_cg0_ready=MBarrier(
            alloc(1), spin=True, stages=1, init_count=CG0_GROUP_WARPS, producer=Producer.THREAD
        ),
        mb_y_input_h_ready=MBarrier(
            alloc(1), spin=True, stages=1, init_count=CG1_WARPS, producer=Producer.THREAD
        ),
        mb_u_input_h_ready=MBarrier(
            alloc(1),
            spin=True,
            stages=1,
            init_count=CG1_WARPS + CG0_GROUP_WARPS,
            producer=Producer.THREAD,
        ),
        mb_state_acc_h_cg0_done=MBarrier(
            alloc(cfg.smem_decay_stages),
            spin=True,
            stages=cfg.smem_decay_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
        ),
        mb_state_acc_h_cg1_done=MBarrier(
            alloc(cfg.smem_decay_stages),
            spin=True,
            stages=cfg.smem_decay_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
        ),
        mb_state_k_acc_m_ready=MBarrier(
            alloc(1), spin=True, stages=1, init_count=1, producer=Producer.MMA_COMMIT
        ),
        mb_u_acc_m_ready=MBarrier(
            alloc(1), spin=True, stages=1, init_count=1, producer=Producer.MMA_COMMIT
        ),
        mb_state_input_m_cg1_ready=MBarrier(
            alloc(1), spin=True, stages=1, init_count=CG1_WARPS, producer=Producer.THREAD
        ),
        mb_state_input_m_cg0_ready=MBarrier(
            alloc(1), spin=True, stages=1, init_count=CG0_GROUP_WARPS, producer=Producer.THREAD
        ),
        mb_y_input_m_ready=MBarrier(
            alloc(1), spin=True, stages=1, init_count=CG1_WARPS, producer=Producer.THREAD
        ),
        mb_u_input_m_ready=MBarrier(
            alloc(1),
            spin=True,
            stages=1,
            init_count=CG1_WARPS + CG0_GROUP_WARPS,
            producer=Producer.THREAD,
        ),
        mb_state_acc_m_cg0_done=MBarrier(
            alloc(cfg.smem_decay_stages),
            spin=True,
            stages=cfg.smem_decay_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
        ),
        mb_state_acc_m_cg1_done=MBarrier(
            alloc(cfg.smem_decay_stages),
            spin=True,
            stages=cfg.smem_decay_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
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
    chunk_parity: cutlass.Int32,
) -> None:
    """Chunk-inverse warp role (warps 12 and 15): persistent scheduler loop computing the
    register-MMA blockwise T_inv, shared by both chains."""
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
    t_inv_idx = swizzle_box_offset_32b(t_inv_row_coord, t_inv_col_coord, box_rows=cfg.b_t)
    k_inv_frag_offsets = [
        opaque_i32(
            swizzle_box_offset_128b(k_inv_row_coord, i * 16 + k_inv_col_offset, box_rows=cfg.b_t)
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
        # Keep the same ring parity across tasks; an odd task must not skip a barrier generation.
        first_chunk = chunk_parity ^ (cum_chunk_base % 2)
        for local_chunk_idx in cutlass.range(first_chunk, num_chunks_tile, 2, unroll=1):
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
    """tcgen05-MMA warp role (warp 13): issues every state GEMM of both chains (H first, M second
    at every stage) and owns the TMEM lifecycle."""
    elect_one = nvvm.elect_sync()
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    nvvm.tcgen05_alloc(sTmem_base, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = sTmem_base.load()
    state_update_n = cutlass.const_expr(cfg.d_k // 2 if cfg.d_k // 2 >= 64 else cfg.d_k)
    state_update_split = cutlass.const_expr(state_update_n < cfg.d_k)
    k_restore_right_bytes = cutlass.const_expr(cfg.b_t * 128)
    state_input_ptr_h = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_input_h_offset, cutlass.Int8)
    state_k_acc_ptr_h = nvvm.make_tmem_ptr(
        tmem_base + cfg.tmem_state_k_acc_h_offset, cutlass.Float32
    )
    u_acc_ptr_h = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_u_acc_h_offset, cutlass.Float32)
    y_input_ptr_h = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_y_input_h_offset, cutlass.Int8)
    u_input_ptr_h = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_u_input_h_offset, cutlass.Int8)
    state_dst_cg0_ptr_h = nvvm.make_tmem_ptr(
        tmem_base + cfg.tmem_state_acc_h_offset, cutlass.Float32
    )
    state_dst_cg1_ptr_h = nvvm.make_tmem_ptr(
        tmem_base + cfg.tmem_state_acc_h_offset + state_update_n, cutlass.Float32
    )
    state_input_ptr_m = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_input_m_offset, cutlass.Int8)
    state_k_acc_ptr_m = nvvm.make_tmem_ptr(
        tmem_base + cfg.tmem_state_k_acc_m_offset, cutlass.Float32
    )
    u_acc_ptr_m = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_u_acc_m_offset, cutlass.Float32)
    y_input_ptr_m = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_y_input_m_offset, cutlass.Int8)
    u_input_ptr_m = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_u_input_m_offset, cutlass.Int8)
    state_dst_cg0_ptr_m = nvvm.make_tmem_ptr(
        tmem_base + cfg.tmem_state_acc_m_offset, cutlass.Float32
    )
    state_dst_cg1_ptr_m = nvvm.make_tmem_ptr(
        tmem_base + cfg.tmem_state_acc_m_offset + state_update_n, cutlass.Float32
    )
    state_input_cg1_index_h = PipelineState.start(phase=0)
    state_input_cg0_index_h = PipelineState.start(phase=0)
    state_input_cg1_index_m = PipelineState.start(phase=0)
    state_input_cg0_index_m = PipelineState.start(phase=0)
    y_input_index = PipelineState.start(phase=0)
    u_input_index = PipelineState.start(phase=0)
    qk_scale_index = PipelineState.start(phase=0)
    k_decay_ready = PipelineState.start(phase=0)
    t_inv_ready = PipelineState.start(phase=0)

    # ---- chunk-invariant GEMM descriptors (rows = DV for chain H, DK for chain M) ----
    bytes_per_element = cfg.io_dtype.width // 8
    instruction_descriptor_acc_h = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        b_major=0,
    )
    instruction_descriptor_acc_m = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        b_major=0,
    )
    instruction_descriptor_final_state_h = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=state_update_n,
        m_dim=cfg.d_v,
        b_major=1,
    )
    instruction_descriptor_final_state_m = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=state_update_n,
        m_dim=cfg.d_k,
        b_major=1,
    )
    bmm_state_k_decay_desc_h = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_acc_h,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_state_k_decay_desc_m = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_acc_m,
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
        cta_group=1,
        idesc=instruction_descriptor_acc_h,
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
        cta_group=1,
        idesc=instruction_descriptor_acc_m,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_u_k_restore_desc_h = MmaDesc(
        M=cfg.d_v,
        N=state_update_n,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        cta_group=1,
        idesc=instruction_descriptor_final_state_h,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_u_k_restore_desc_m = MmaDesc(
        M=cfg.d_k,
        N=state_update_n,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        cta_group=1,
        idesc=instruction_descriptor_final_state_m,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    STATE_A_SEG = bmm_state_k_decay_desc_h.sps_B * bmm_state_k_decay_desc_h.tmem_advance_A
    STATE_B_SEG = bmm_state_k_decay_desc_h.smem_subtile_B >> 4
    STATE_K_STEPS_CG0 = bmm_state_k_decay_desc_h.num_k_steps // 2
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
        seed_state = compute_start == 0
        for local_chunk_idx in cutlass.range(num_chunks_tile, unroll=1):
            if cutlass.const_expr(cfg.use_initial_state):
                have_state_h = local_chunk_idx > 0 or seed_state
            else:
                have_state_h = local_chunk_idx > 0
            have_state_m = local_chunk_idx > 0 or seed_state
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

            # ---- k state H = state H(T) @ K decay^T, k state M = state M(T) @ K decay^T --
            bars.mb_k_decay_inv_cg0_ready[decay_stage].wait(k_decay_ready.phase)
            k_decay_ready = advance(k_decay_ready, cfg.smem_decay_stages)
            if have_state_h:
                bars.mb_state_input_h_cg0_ready.wait(state_input_cg0_index_h.phase)
                state_input_cg0_index_h = advance(state_input_cg0_index_h, 1)

                for f in cutlass.range_constexpr(bmm_state_k_decay_desc_h.num_k_steps):
                    if cutlass.const_expr(f == STATE_K_STEPS_CG0):
                        bars.mb_state_input_h_cg1_ready.wait(state_input_cg1_index_h.phase)
                        state_input_cg1_index_h = advance(state_input_cg1_index_h, 1)
                    s = f // bmm_state_k_decay_desc_h.sps_B
                    k = f - s * bmm_state_k_decay_desc_h.sps_B
                    mma_ts_step(
                        bmm_state_k_decay_desc_h,
                        state_input_ptr_h.subview(s * STATE_A_SEG),
                        desc_k_decay + s * STATE_B_SEG,
                        state_k_acc_ptr_h,
                        k,
                        cutlass.Boolean(f > 0),
                        issue_mma=elect_one,
                    )

                if elect_one:
                    bars.mb_state_k_acc_h_ready.arrive(cta_group=1)
            if have_state_m:
                bars.mb_state_input_m_cg0_ready.wait(state_input_cg0_index_m.phase)
                state_input_cg0_index_m = advance(state_input_cg0_index_m, 1)

                for f in cutlass.range_constexpr(bmm_state_k_decay_desc_m.num_k_steps):
                    if cutlass.const_expr(f == STATE_K_STEPS_CG0):
                        bars.mb_state_input_m_cg1_ready.wait(state_input_cg1_index_m.phase)
                        state_input_cg1_index_m = advance(state_input_cg1_index_m, 1)
                    s = f // bmm_state_k_decay_desc_m.sps_B
                    k = f - s * bmm_state_k_decay_desc_m.sps_B
                    mma_ts_step(
                        bmm_state_k_decay_desc_m,
                        state_input_ptr_m.subview(s * STATE_A_SEG),
                        desc_k_decay + s * STATE_B_SEG,
                        state_k_acc_ptr_m,
                        k,
                        cutlass.Boolean(f > 0),
                        issue_mma=elect_one,
                    )

                if elect_one:
                    bars.mb_state_k_acc_m_ready.arrive(cta_group=1)

            if elect_one:
                bars.mb_decay_tcgen05_done[decay_stage].arrive(cta_group=1)

            bars.mb_qk_scale_ready[qk_scale_index.idx].wait(qk_scale_index.phase)

            # ---- U H = Y H(T) @ T^-1, U M = Y M(T) @ T^-1 ----------------------------
            bars.mb_t_inv_ready[intermediate_stage].wait(t_inv_ready.phase)
            bars.mb_y_input_h_ready.wait(y_input_index.phase)
            mma_ts_step(
                bmm_y_t_inv_desc_h,
                y_input_ptr_h,
                desc_t_inv,
                u_acc_ptr_h,
                0,
                cutlass.Boolean(False),
                issue_mma=elect_one,
            )
            if elect_one:
                bars.mb_u_acc_h_ready.arrive(cta_group=1)
            bars.mb_y_input_m_ready.wait(y_input_index.phase)
            mma_ts_step(
                bmm_y_t_inv_desc_m,
                y_input_ptr_m,
                desc_t_inv,
                u_acc_ptr_m,
                0,
                cutlass.Boolean(False),
                issue_mma=elect_one,
            )
            if elect_one:
                bars.mb_u_acc_m_ready.arrive(cta_group=1)
                bars.mb_t_inv_done[intermediate_stage].arrive(cta_group=1)
            y_input_index = advance(y_input_index, 1)

            # ---- state H += U H(T) @ K restore, then state M: left then right key half --
            bars.mb_u_input_h_ready.wait(u_input_index.phase)
            mma_ts_step(
                bmm_u_k_restore_desc_h,
                u_input_ptr_h,
                desc_k_restore,
                state_dst_cg0_ptr_h,
                0,
                have_state_h,
                issue_mma=elect_one,
            )
            if elect_one:
                bars.mb_state_acc_h_cg0_done[decay_stage].arrive(cta_group=1)
            if cutlass.const_expr(state_update_split):
                mma_ts_step(
                    bmm_u_k_restore_desc_h,
                    u_input_ptr_h,
                    desc_k_restore_right,
                    state_dst_cg1_ptr_h,
                    0,
                    have_state_h,
                    issue_mma=elect_one,
                )
            if elect_one:
                bars.mb_state_acc_h_cg1_done[decay_stage].arrive(cta_group=1)
            bars.mb_u_input_m_ready.wait(u_input_index.phase)
            mma_ts_step(
                bmm_u_k_restore_desc_m,
                u_input_ptr_m,
                desc_k_restore,
                state_dst_cg0_ptr_m,
                0,
                have_state_m,
                issue_mma=elect_one,
            )
            if elect_one:
                bars.mb_state_acc_m_cg0_done[decay_stage].arrive(cta_group=1)
            if cutlass.const_expr(state_update_split):
                mma_ts_step(
                    bmm_u_k_restore_desc_m,
                    u_input_ptr_m,
                    desc_k_restore_right,
                    state_dst_cg1_ptr_m,
                    0,
                    have_state_m,
                    issue_mma=elect_one,
                )
            if elect_one:
                bars.mb_k_restore_done[decay_stage].arrive(cta_group=1)
                bars.mb_state_acc_m_cg1_done[decay_stage].arrive(cta_group=1)
            u_input_index = advance(u_input_index, 1)

            t_inv_ready = advance(t_inv_ready, cfg.smem_intermediate_stages)
            qk_scale_index = advance(qk_scale_index, cfg.qk_scale_ready_stages)

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
            tma_tensormap_acquire(desc_v_slot)
            tma_tensormap_acquire(desc_gate_slot)
        for chunk_idx in cutlass.range(compute_start, write_end, 1, unroll=1):
            chunk_start = chunk_idx * cfg.b_t

            # ---- K / Gate / V loads: one transaction barrier per stage ---------------
            bars.mb_raw_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_raw_ready[raw_index.idx].arrive(
                    n_bytes=cfg.tma_k_bytes + cfg.tma_gate_bytes + cfg.tma_v_bytes
                )
            raw_ready_ptr = bars.mb_raw_ready[raw_index.idx].smem_ptr
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), head_k, chunk_start)
            tma_load_tile(sK_tma[raw_index.idx], k_slice, raw_ready_ptr)
            gate_slice = tma_slice_runtime_desc(
                desc_gate_slot, cutlass.Int32(0), head_o, chunk_start
            )
            tma_load_tile(sGate_tma[raw_index.idx], gate_slice, raw_ready_ptr)
            v_slice = tma_slice_runtime_desc(desc_v_slot, cutlass.Int32(0), head_v, chunk_start)
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
    sK_inv_raw,
    sGate_exchange_raw,
    sGate_load_ptr,
    mBeta,
    sBeta_raw,
    sK_raw,
    sK_decay_raw,
    sK_restore_raw,
    sTmem_base,
    bars,
) -> None:
    """CG0 warp-group role (warps 0-7): persistent scheduler loop running the gate prefix scan,
    staging the decay / restore operands and carrying the left key half of both states."""
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
    state_col_id_h = tmem_col + cfg.tmem_state_acc_h_offset
    packed_col_id_h = tmem_col + cfg.tmem_state_input_h_offset
    state_col_id_m = tmem_col + cfg.tmem_state_acc_m_offset
    packed_col_id_m = tmem_col + cfg.tmem_state_input_m_offset

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
            _write_start,
            write_end,
            compute_start,
            _compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        head_o = head_idx
        num_chunks_tile = write_end - compute_start
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

            # ---- K inv stage ---------------------------------------------------------
            for dim_half in cutlass.range_constexpr(dk_halves):
                dim_base = dim_half * 64 + lane_in_row_group * 8
                reg_base = dim_half * 8
                raw_f16_idx = swizzle_box_offset_128b(decay_row, dim_base, box_rows=cfg.b_t)
                raw_k_frag = (sK_ptr + raw_f16_idx).load(count=8, alignment=16)
                raw_k_vec_f32 = raw_k_frag.to(cutlass.Float32)
                for dim_offset in cutlass.range_constexpr(8):
                    k_val = raw_k_vec_f32[dim_offset]
                    raw_k_regs[reg_base + dim_offset] = k_val

            k_inv_norm = opaque_one

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
                k_inv_swizzled_idx = swizzle_box_offset_128b(decay_row, dim_base, box_rows=cfg.b_t)
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
                k_restore_idx = swizzle_box_offset_128b(decay_row, dim_base, box_rows=cfg.b_t)
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
                bars.mb_state_acc_h_cg0_done[
                    cutlass.Int32(update_count % cfg.smem_decay_stages)
                ].wait(cutlass.Int32((update_count // cfg.smem_decay_stages) % 2))
            if local_chunk_idx > 0:
                l_state_vecs = []
                for b in cutlass.range_constexpr(dk_halves):
                    l_state_vecs.append(
                        nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                row_lo_addr + state_col_id_h + b * 32, cutlass.Float32
                            ),
                            num=32,
                        )
                    )
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
                        nvvm.make_tmem_ptr(row_lo_addr + packed_col_id_h + b * 16, cutlass.Int8),
                        l_packed.load(),
                    )
                nvvm.tcgen05_wait("store")
                if nvvm.elect_sync():
                    bars.mb_state_input_h_cg0_ready.arrive()

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
                        nvvm.make_tmem_ptr(row_lo_addr + state_col_id_h + b * 32, cutlass.Float32),
                        cutlass.Vector.from_elements(tuple(l_scaled), cutlass.Float32),
                    )
                nvvm.tcgen05_wait("store")

            # ---- state stage, left key half: pack, publish, fp32 decay ---------------
            if cum_chunk > 0:
                update_count = chunk_count - cutlass.Uint32(1)
                bars.mb_state_acc_m_cg0_done[
                    cutlass.Int32(update_count % cfg.smem_decay_stages)
                ].wait(cutlass.Int32((update_count // cfg.smem_decay_stages) % 2))
            if local_chunk_idx > 0:
                l_state_vecs = []
                for b in cutlass.range_constexpr(dk_halves):
                    l_state_vecs.append(
                        nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                row_lo_addr + state_col_id_m + b * 32, cutlass.Float32
                            ),
                            num=32,
                        )
                    )
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
                        nvvm.make_tmem_ptr(row_lo_addr + packed_col_id_m + b * 16, cutlass.Int8),
                        l_packed.load(),
                    )
                nvvm.tcgen05_wait("store")
                if nvvm.elect_sync():
                    bars.mb_state_input_m_cg0_ready.arrive()

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
                        nvvm.make_tmem_ptr(row_lo_addr + state_col_id_m + b * 32, cutlass.Float32),
                        cutlass.Vector.from_elements(tuple(l_scaled), cutlass.Float32),
                    )
                nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_raw_done[raw_stage].arrive()
                bars.mb_u_input_h_ready.arrive()
                bars.mb_u_input_m_ready.arrive()
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
    mTransition,
    mState_init,
    sBeta_raw,
    sV_raw,
    sGate_exchange_raw,
    bars,
) -> None:
    """CG1 warp-group role (warps 8-11): persistent scheduler loop seeding both states, packing and
    rescaling the right key half of both, staging the Y / U inputs of both chains (H first, M
    second at every stage) and storing H and M."""
    nvvm.setmaxregister(
        cfg.num_regs_compute_group_1,
        nvvm.SetMaxRegisterAction.INCREASE
        if cfg.num_regs_compute_group_1 >= 65536 // cfg.threads_per_cta
        else nvvm.SetMaxRegisterAction.DECREASE,
    )
    elect_one = nvvm.elect_sync()

    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = sTmem_base.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16

    # ---- ldmatrix.x4 COL lane decode for the V loads ---------------------------------
    ov_row_coord = (lane_idx // 16) * 8 + (lane_idx & 7)
    ov_col_offset = ((lane_idx // 8) & 1) * 8
    cg1_warp = warp_idx - cfg.compute_group_1_warp_ids[0]
    row_lanes_h = cutlass.const_expr(cfg.threads_per_warp if cfg.d_v == 128 else 16)
    row_lanes_m = cutlass.const_expr(cfg.threads_per_warp if cfg.d_k == 128 else 16)
    value_dim = cg1_warp * row_lanes_h + lane_idx % row_lanes_h
    value_dim_base = cg1_warp * row_lanes_h
    row_valid_h = lane_idx < cutlass.Int32(row_lanes_h)
    key_dim = cg1_warp * row_lanes_m + lane_idx % row_lanes_m
    row_valid_m = lane_idx < cutlass.Int32(row_lanes_m)
    row_lo_addr = tmem_row << 16
    row_hi_addr = (tmem_row + 16) << 16
    state_blocks_per_half = cutlass.const_expr(cfg.d_k // 32)
    state_col_id_h = tmem_col + cfg.tmem_state_acc_h_offset
    packed_col_id_h = tmem_col + cfg.tmem_state_input_h_offset
    statek_col_id_h = tmem_col + cfg.tmem_state_k_acc_h_offset
    y_input_col_id_h = tmem_col + cfg.tmem_y_input_h_offset
    u_acc_addr_h = row_lo_addr + tmem_col + cfg.tmem_u_acc_h_offset
    u_input_addr_h = row_lo_addr + tmem_col + cfg.tmem_u_input_h_offset
    state_col_id_m = tmem_col + cfg.tmem_state_acc_m_offset
    packed_col_id_m = tmem_col + cfg.tmem_state_input_m_offset
    statek_col_id_m = tmem_col + cfg.tmem_state_k_acc_m_offset
    y_input_col_id_m = tmem_col + cfg.tmem_y_input_m_offset
    u_acc_addr_m = row_lo_addr + tmem_col + cfg.tmem_u_acc_m_offset
    u_input_addr_m = row_lo_addr + tmem_col + cfg.tmem_u_input_m_offset
    v_swizzle_off_lo = swizzle_box_offset_128b(
        ov_row_coord, value_dim_base + ov_col_offset, box_rows=cfg.b_t
    )
    v_swizzle_off_hi = swizzle_box_offset_128b(
        ov_row_coord, value_dim_base + 16 + ov_col_offset, box_rows=cfg.b_t
    )
    state_k_acc_index_h = PipelineState.start(phase=0)
    state_k_acc_index_m = PipelineState.start(phase=0)
    u_acc_index = PipelineState.start(phase=0)
    state_update_index = PipelineState.start(phase=0)
    raw_index = PipelineState.start(phase=0)
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
            _write_start,
            write_end,
            compute_start,
            _compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        head_o = head_idx
        num_chunks_tile = write_end - compute_start

        if num_chunks_tile > 0:
            seed_state = compute_start == 0
            sV_ptr = smem_data_ptr(sV_raw) + raw_index.idx * (cfg.d_v * cfg.b_t)
            sBeta_ptr = smem_data_ptr(sBeta_raw) + raw_index.idx * cfg.b_t

            if seed_state:
                bars.mb_gate_exchange_ready[raw_index.idx].wait(raw_index.phase)
                seed_exchange_ptr = smem_data_ptr(sGate_exchange_raw) + (
                    cum_chunk_base % cfg.gate_exchange_stages
                ) * (cfg.d_k * cfg.b_t)

                # ---- state seed: initial state GMEM -> packed b16 TMEM + fp32 state TMEM ----
                if cutlass.const_expr(mState_init is not None):
                    seed_vw = 16 // (mState_init.element_type.width // 8)
                    seed_src = (
                        mState_init.iterator
                        + mState_init.layout((batch_idx, head_o, value_dim, 0))
                    ).raw_ptr()
                    for seed_half in cutlass.range_constexpr(2):
                        seed_blocks_lo = seed_half * state_blocks_per_half
                        seed_blocks_hi = seed_blocks_lo + state_blocks_per_half
                        seed_vecs = []
                        for i in cutlass.range_constexpr(seed_blocks_lo, seed_blocks_hi):
                            seed_block = []
                            for g in cutlass.range_constexpr(16 // seed_vw):
                                seed_chunk = (seed_src + i * 16 + g * seed_vw).load(
                                    count=seed_vw, alignment=16
                                )
                                for t in cutlass.range_constexpr(seed_vw):
                                    seed_block.append(seed_chunk[t].to(cutlass.Float32))
                            seed_vecs.append(seed_block)

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
                                nvvm.make_tmem_ptr(
                                    row_lo_addr + packed_col_id_h + i * 8, cutlass.Int8
                                ),
                                seed_pack.load(),
                            )
                        nvvm.tcgen05_wait("store")
                        if cutlass.const_expr(seed_half == 0):
                            if nvvm.elect_sync():
                                bars.mb_state_input_h_cg0_ready.arrive()
                        else:
                            if nvvm.elect_sync():
                                bars.mb_state_input_h_cg1_ready.arrive()

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
                                        seed_vecs[i - seed_blocks_lo][
                                            seed_scale_group * 4 + 2 * t
                                        ],
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
                                    row_lo_addr + state_col_id_h + i * 16, cutlass.Float32
                                ),
                                cutlass.Vector.from_elements(tuple(seed_scaled), cutlass.Float32),
                            )
                    nvvm.tcgen05_wait("store")

                # ---- identity seed: one -> packed b16 TMEM + fp32 state TMEM ---------
                seed_zero = opaque_f32_zero()
                seed_one = seed_zero + cutlass.Float32(1.0)
                for seed_half in cutlass.range_constexpr(2):
                    seed_blocks_lo = seed_half * state_blocks_per_half
                    seed_blocks_hi = seed_blocks_lo + state_blocks_per_half
                    seed_vecs = []
                    for i in cutlass.range_constexpr(seed_blocks_lo, seed_blocks_hi):
                        seed_block = []
                        for j in cutlass.range_constexpr(16):
                            seed_block.append(seed_one if key_dim == i * 16 + j else seed_zero)
                        seed_vecs.append(seed_block)

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
                            nvvm.make_tmem_ptr(
                                row_lo_addr + packed_col_id_m + i * 8, cutlass.Int8
                            ),
                            seed_pack.load(),
                        )
                    nvvm.tcgen05_wait("store")
                    if cutlass.const_expr(seed_half == 0):
                        if nvvm.elect_sync():
                            bars.mb_state_input_m_cg0_ready.arrive()
                    else:
                        if nvvm.elect_sync():
                            bars.mb_state_input_m_cg1_ready.arrive()

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
                                row_lo_addr + state_col_id_m + i * 16, cutlass.Float32
                            ),
                            cutlass.Vector.from_elements(tuple(seed_scaled), cutlass.Float32),
                        )
                nvvm.tcgen05_wait("store")

            # ---- Y stage: Y = Beta * (V - k state) -----------------------------------
            bars.mb_raw_ready[raw_index.idx].wait(raw_index.phase)
            raw_v_frag_lo = nvvm.ldmatrix(sV_ptr + v_swizzle_off_lo, 4, nvvm.MMALayout.COL)
            raw_v_frag_hi = raw_v_frag_lo
            if cutlass.const_expr(cfg.d_v == 128):
                raw_v_frag_hi = nvvm.ldmatrix(sV_ptr + v_swizzle_off_hi, 4, nvvm.MMALayout.COL)
            bars.mb_beta_ready[raw_index.idx].wait(raw_index.phase)
            # Attention Gym modification (B6): beta stays FP32 and each residual is formed in FP32,
            # rounded once into the b16 MMA operand.
            beta_regs = []
            for reg_idx in cutlass.range_constexpr(4):
                token0 = ((reg_idx // 2) * 4 + (lane_idx & 3)) * 2
                beta0 = (sBeta_ptr + token0).load().to(cutlass.Float32)
                beta1 = (sBeta_ptr + token0 + 1).load().to(cutlass.Float32)
                beta_regs.append((beta0, beta1))
            if cutlass.const_expr(mState_init is not None):
                have_state_h = seed_state
            else:
                have_state_h = cutlass.Boolean(False)
            y_lo = [cutlass.Int32(0) for _ in range(4)]
            y_hi = [cutlass.Int32(0) for _ in range(4)]
            if have_state_h:
                bars.mb_state_k_acc_h_ready.wait(state_k_acc_index_h.phase)
                state_k_acc_index_h = advance(state_k_acc_index_h, 1)
                state_k_vec_lo = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(row_lo_addr + statek_col_id_h, cutlass.Float32),
                    num=2,
                )
                for reg_idx in cutlass.range_constexpr(4):
                    frag_pair = reg_idx * 2
                    y_lo[reg_idx] = beta_residual_f16x2(
                        raw_v_frag_lo[reg_idx],
                        *beta_regs[reg_idx],
                        state_k_vec_lo[frag_pair],
                        state_k_vec_lo[frag_pair + 1],
                        dtype=cfg.io_dtype,
                    )[0]
                if cutlass.const_expr(cfg.d_v == 128):
                    state_k_vec_hi = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(row_hi_addr + statek_col_id_h, cutlass.Float32),
                        num=2,
                    )
                    for reg_idx in cutlass.range_constexpr(4):
                        frag_pair = reg_idx * 2
                        y_hi[reg_idx] = beta_residual_f16x2(
                            raw_v_frag_hi[reg_idx],
                            *beta_regs[reg_idx],
                            state_k_vec_hi[frag_pair],
                            state_k_vec_hi[frag_pair + 1],
                            dtype=cfg.io_dtype,
                        )[0]
            else:
                for reg_idx in cutlass.range_constexpr(4):
                    y_lo[reg_idx] = beta_residual_f16x2(
                        raw_v_frag_lo[reg_idx], *beta_regs[reg_idx], dtype=cfg.io_dtype
                    )[0]
                    y_hi[reg_idx] = beta_residual_f16x2(
                        raw_v_frag_hi[reg_idx], *beta_regs[reg_idx], dtype=cfg.io_dtype
                    )[0]

            y_input_pack_lo = cute.make_rmem_tensor((4,), cutlass.Int32)
            y_input_pack_hi = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                y_input_pack_lo[reg_idx] = y_lo[reg_idx]
                y_input_pack_hi[reg_idx] = y_hi[reg_idx]
            nvvm.tcgen05_st(
                "16x128b",
                nvvm.make_tmem_ptr(row_lo_addr + y_input_col_id_h, cutlass.Int8),
                y_input_pack_lo.load(),
            )
            if cutlass.const_expr(cfg.d_v == 128):
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(row_hi_addr + y_input_col_id_h, cutlass.Int8),
                    y_input_pack_hi.load(),
                )
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_y_input_h_ready.arrive()

            # ---- Y stage: Y = Beta * (0 - k state) -----------------------------------
            zero_word = opaque_i32_zero()
            zero_v_frag_lo = [zero_word for _ in range(4)]
            zero_v_frag_hi = [zero_word for _ in range(4)]
            y_lo = [cutlass.Int32(0) for _ in range(4)]
            y_hi = [cutlass.Int32(0) for _ in range(4)]
            if seed_state:
                bars.mb_state_k_acc_m_ready.wait(state_k_acc_index_m.phase)
                state_k_acc_index_m = advance(state_k_acc_index_m, 1)
                state_k_vec_lo = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(row_lo_addr + statek_col_id_m, cutlass.Float32),
                    num=2,
                )
                for reg_idx in cutlass.range_constexpr(4):
                    frag_pair = reg_idx * 2
                    y_lo[reg_idx] = beta_residual_f16x2(
                        zero_v_frag_lo[reg_idx],
                        *beta_regs[reg_idx],
                        state_k_vec_lo[frag_pair],
                        state_k_vec_lo[frag_pair + 1],
                        dtype=cfg.io_dtype,
                    )[0]
                if cutlass.const_expr(cfg.d_k == 128):
                    state_k_vec_hi = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(row_hi_addr + statek_col_id_m, cutlass.Float32),
                        num=2,
                    )
                    for reg_idx in cutlass.range_constexpr(4):
                        frag_pair = reg_idx * 2
                        y_hi[reg_idx] = beta_residual_f16x2(
                            zero_v_frag_hi[reg_idx],
                            *beta_regs[reg_idx],
                            state_k_vec_hi[frag_pair],
                            state_k_vec_hi[frag_pair + 1],
                            dtype=cfg.io_dtype,
                        )[0]
            else:
                for reg_idx in cutlass.range_constexpr(4):
                    y_lo[reg_idx] = beta_residual_f16x2(
                        zero_v_frag_lo[reg_idx], *beta_regs[reg_idx], dtype=cfg.io_dtype
                    )[0]
                    y_hi[reg_idx] = beta_residual_f16x2(
                        zero_v_frag_hi[reg_idx], *beta_regs[reg_idx], dtype=cfg.io_dtype
                    )[0]

            y_input_pack_lo = cute.make_rmem_tensor((4,), cutlass.Int32)
            y_input_pack_hi = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                y_input_pack_lo[reg_idx] = y_lo[reg_idx]
                y_input_pack_hi[reg_idx] = y_hi[reg_idx]
            nvvm.tcgen05_st(
                "16x128b",
                nvvm.make_tmem_ptr(row_lo_addr + y_input_col_id_m, cutlass.Int8),
                y_input_pack_lo.load(),
            )
            if cutlass.const_expr(cfg.d_k == 128):
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(row_hi_addr + y_input_col_id_m, cutlass.Int8),
                    y_input_pack_hi.load(),
                )
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_y_input_m_ready.arrive()
            if nvvm.elect_sync():
                bars.mb_beta_done[raw_index.idx].arrive()
                bars.mb_raw_done[raw_index.idx].arrive()

            # ---- U stage: u acc TMEM -> packed b16 U input TMEM ----------------------
            u_acc_phase = u_acc_index.phase
            bars.mb_u_acc_h_ready.wait(u_acc_phase)
            u_acc_vals = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(u_acc_addr_h, cutlass.Float32),
                num=cfg.b_t,
            )
            u_input_pack = cute.make_rmem_tensor((cfg.b_t // 2,), cutlass.Int32)
            for packed_col in cutlass.range_constexpr(cfg.b_t // 2):
                token0 = packed_col * 2
                token1 = token0 + 1
                u_input_pack[packed_col] = fp32_to_fp16(
                    u_acc_vals[token0], u_acc_vals[token1], dtype=cfg.io_dtype
                )
            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(u_input_addr_h, cutlass.Int8),
                u_input_pack.load(),
            )
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_u_input_h_ready.arrive()

            # ---- U stage: u acc TMEM -> packed b16 U input TMEM ----------------------
            bars.mb_u_acc_m_ready.wait(u_acc_phase)
            u_acc_vals = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(u_acc_addr_m, cutlass.Float32),
                num=cfg.b_t,
            )
            u_input_pack = cute.make_rmem_tensor((cfg.b_t // 2,), cutlass.Int32)
            for packed_col in cutlass.range_constexpr(cfg.b_t // 2):
                token0 = packed_col * 2
                token1 = token0 + 1
                u_input_pack[packed_col] = fp32_to_fp16(
                    u_acc_vals[token0], u_acc_vals[token1], dtype=cfg.io_dtype
                )
            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(u_input_addr_m, cutlass.Int8),
                u_input_pack.load(),
            )
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_u_input_m_ready.arrive()
            u_acc_index = advance(u_acc_index, 1)
            raw_index = advance(raw_index, cfg.smem_raw_stages)

        for local_chunk_idx in cutlass.range(1, num_chunks_tile, 1, unroll=1):
            cum_chunk = cum_chunk_base + local_chunk_idx
            raw_stage = raw_index.idx
            raw_phase = raw_index.phase
            sV_ptr = smem_data_ptr(sV_raw) + raw_stage * (cfg.d_v * cfg.b_t)
            sBeta_ptr = smem_data_ptr(sBeta_raw) + raw_stage * cfg.b_t
            bars.mb_raw_ready[raw_stage].wait(raw_phase)
            bars.mb_beta_ready[raw_stage].wait(raw_phase)
            raw_index = advance(raw_index, cfg.smem_raw_stages)

            # ---- state stage, right key half: pack, publish, fp32 decay --------------
            sGate_exchange_ptr = smem_data_ptr(sGate_exchange_raw) + (
                cum_chunk % cfg.gate_exchange_stages
            ) * (cfg.d_k * cfg.b_t)
            bars.mb_state_acc_h_cg1_done[state_update_index.idx].wait(state_update_index.phase)
            state_vecs = []
            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                state_vecs.append(
                    nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + state_col_id_h + i * 16, cutlass.Float32),
                        num=16,
                    )
                )

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
                    nvvm.make_tmem_ptr(row_lo_addr + packed_col_id_h + i * 8, cutlass.Int8),
                    packed_state.load(),
                )

            # ---- fp32 decay of the right key half: state *= exp2(g last) -------------
            bars.mb_gate_exchange_ready[raw_stage].wait(raw_phase)
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
                bars.mb_state_input_h_cg1_ready.arrive()
            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(row_lo_addr + state_col_id_h + i * 16, cutlass.Float32),
                    cutlass.Vector.from_elements(
                        tuple(scaled_blocks[i - state_blocks_per_half]), cutlass.Float32
                    ),
                )

            # ---- state stage, right key half: pack, publish, fp32 decay --------------
            bars.mb_state_acc_m_cg1_done[state_update_index.idx].wait(state_update_index.phase)
            state_vecs = []
            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                state_vecs.append(
                    nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + state_col_id_m + i * 16, cutlass.Float32),
                        num=16,
                    )
                )

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
                    nvvm.make_tmem_ptr(row_lo_addr + packed_col_id_m + i * 8, cutlass.Int8),
                    packed_state.load(),
                )

            # ---- fp32 decay of the right key half: state *= exp2(g last) -------------
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
                bars.mb_state_input_m_cg1_ready.arrive()
            for i in cutlass.range_constexpr(state_blocks_per_half, cfg.d_k // 16):
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr(row_lo_addr + state_col_id_m + i * 16, cutlass.Float32),
                    cutlass.Vector.from_elements(
                        tuple(scaled_blocks[i - state_blocks_per_half]), cutlass.Float32
                    ),
                )
            state_update_index = advance(state_update_index, cfg.smem_decay_stages)

            # ---- Y stage: Y = Beta * (V - k state) -----------------------------------
            have_state = cutlass.Boolean(True)
            raw_v_frag_lo = nvvm.ldmatrix(sV_ptr + v_swizzle_off_lo, 4, nvvm.MMALayout.COL)
            raw_v_frag_hi = raw_v_frag_lo
            if cutlass.const_expr(cfg.d_v == 128):
                raw_v_frag_hi = nvvm.ldmatrix(sV_ptr + v_swizzle_off_hi, 4, nvvm.MMALayout.COL)
            # Attention Gym modification (B6): beta stays FP32 and each residual is formed in FP32,
            # rounded once into the b16 MMA operand.
            beta_regs = []
            for reg_idx in cutlass.range_constexpr(4):
                token0 = ((reg_idx // 2) * 4 + (lane_idx & 3)) * 2
                beta0 = (sBeta_ptr + token0).load().to(cutlass.Float32)
                beta1 = (sBeta_ptr + token0 + 1).load().to(cutlass.Float32)
                beta_regs.append((beta0, beta1))
            y_lo = [cutlass.Int32(0) for _ in range(4)]
            y_hi = [cutlass.Int32(0) for _ in range(4)]
            if have_state:
                bars.mb_state_k_acc_h_ready.wait(state_k_acc_index_h.phase)
                state_k_acc_index_h = advance(state_k_acc_index_h, 1)
                state_k_vec_lo = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(row_lo_addr + statek_col_id_h, cutlass.Float32),
                    num=2,
                )
                for reg_idx in cutlass.range_constexpr(4):
                    frag_pair = reg_idx * 2
                    y_lo[reg_idx] = beta_residual_f16x2(
                        raw_v_frag_lo[reg_idx],
                        *beta_regs[reg_idx],
                        state_k_vec_lo[frag_pair],
                        state_k_vec_lo[frag_pair + 1],
                        dtype=cfg.io_dtype,
                    )[0]
                if cutlass.const_expr(cfg.d_v == 128):
                    state_k_vec_hi = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(row_hi_addr + statek_col_id_h, cutlass.Float32),
                        num=2,
                    )
                    for reg_idx in cutlass.range_constexpr(4):
                        frag_pair = reg_idx * 2
                        y_hi[reg_idx] = beta_residual_f16x2(
                            raw_v_frag_hi[reg_idx],
                            *beta_regs[reg_idx],
                            state_k_vec_hi[frag_pair],
                            state_k_vec_hi[frag_pair + 1],
                            dtype=cfg.io_dtype,
                        )[0]
            else:
                for reg_idx in cutlass.range_constexpr(4):
                    y_lo[reg_idx] = beta_residual_f16x2(
                        raw_v_frag_lo[reg_idx], *beta_regs[reg_idx], dtype=cfg.io_dtype
                    )[0]
                    y_hi[reg_idx] = beta_residual_f16x2(
                        raw_v_frag_hi[reg_idx], *beta_regs[reg_idx], dtype=cfg.io_dtype
                    )[0]

            y_input_pack_lo = cute.make_rmem_tensor((4,), cutlass.Int32)
            y_input_pack_hi = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                y_input_pack_lo[reg_idx] = y_lo[reg_idx]
                y_input_pack_hi[reg_idx] = y_hi[reg_idx]
            nvvm.tcgen05_st(
                "16x128b",
                nvvm.make_tmem_ptr(row_lo_addr + y_input_col_id_h, cutlass.Int8),
                y_input_pack_lo.load(),
            )
            if cutlass.const_expr(cfg.d_v == 128):
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(row_hi_addr + y_input_col_id_h, cutlass.Int8),
                    y_input_pack_hi.load(),
                )
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_y_input_h_ready.arrive()

            # ---- Y stage: Y = Beta * (0 - k state) -----------------------------------
            zero_word = opaque_i32_zero()
            zero_v_frag_lo = [zero_word for _ in range(4)]
            zero_v_frag_hi = [zero_word for _ in range(4)]
            y_lo = [cutlass.Int32(0) for _ in range(4)]
            y_hi = [cutlass.Int32(0) for _ in range(4)]
            if have_state:
                bars.mb_state_k_acc_m_ready.wait(state_k_acc_index_m.phase)
                state_k_acc_index_m = advance(state_k_acc_index_m, 1)
                state_k_vec_lo = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(row_lo_addr + statek_col_id_m, cutlass.Float32),
                    num=2,
                )
                for reg_idx in cutlass.range_constexpr(4):
                    frag_pair = reg_idx * 2
                    y_lo[reg_idx] = beta_residual_f16x2(
                        zero_v_frag_lo[reg_idx],
                        *beta_regs[reg_idx],
                        state_k_vec_lo[frag_pair],
                        state_k_vec_lo[frag_pair + 1],
                        dtype=cfg.io_dtype,
                    )[0]
                if cutlass.const_expr(cfg.d_k == 128):
                    state_k_vec_hi = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(row_hi_addr + statek_col_id_m, cutlass.Float32),
                        num=2,
                    )
                    for reg_idx in cutlass.range_constexpr(4):
                        frag_pair = reg_idx * 2
                        y_hi[reg_idx] = beta_residual_f16x2(
                            zero_v_frag_hi[reg_idx],
                            *beta_regs[reg_idx],
                            state_k_vec_hi[frag_pair],
                            state_k_vec_hi[frag_pair + 1],
                            dtype=cfg.io_dtype,
                        )[0]
            else:
                for reg_idx in cutlass.range_constexpr(4):
                    y_lo[reg_idx] = beta_residual_f16x2(
                        zero_v_frag_lo[reg_idx], *beta_regs[reg_idx], dtype=cfg.io_dtype
                    )[0]
                    y_hi[reg_idx] = beta_residual_f16x2(
                        zero_v_frag_hi[reg_idx], *beta_regs[reg_idx], dtype=cfg.io_dtype
                    )[0]

            y_input_pack_lo = cute.make_rmem_tensor((4,), cutlass.Int32)
            y_input_pack_hi = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                y_input_pack_lo[reg_idx] = y_lo[reg_idx]
                y_input_pack_hi[reg_idx] = y_hi[reg_idx]
            nvvm.tcgen05_st(
                "16x128b",
                nvvm.make_tmem_ptr(row_lo_addr + y_input_col_id_m, cutlass.Int8),
                y_input_pack_lo.load(),
            )
            if cutlass.const_expr(cfg.d_k == 128):
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(row_hi_addr + y_input_col_id_m, cutlass.Int8),
                    y_input_pack_hi.load(),
                )
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_y_input_m_ready.arrive()
            if nvvm.elect_sync():
                bars.mb_beta_done[raw_stage].arrive()
                bars.mb_raw_done[raw_stage].arrive()

            # ---- U stage: u acc TMEM -> packed b16 U input TMEM ----------------------
            u_acc_phase = u_acc_index.phase
            bars.mb_u_acc_h_ready.wait(u_acc_phase)
            u_acc_vals = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(u_acc_addr_h, cutlass.Float32),
                num=cfg.b_t,
            )
            u_input_pack = cute.make_rmem_tensor((cfg.b_t // 2,), cutlass.Int32)
            for packed_col in cutlass.range_constexpr(cfg.b_t // 2):
                token0 = packed_col * 2
                token1 = token0 + 1
                u_input_pack[packed_col] = fp32_to_fp16(
                    u_acc_vals[token0], u_acc_vals[token1], dtype=cfg.io_dtype
                )
            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(u_input_addr_h, cutlass.Int8),
                u_input_pack.load(),
            )
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_u_input_h_ready.arrive()

            # ---- U stage: u acc TMEM -> packed b16 U input TMEM ----------------------
            bars.mb_u_acc_m_ready.wait(u_acc_phase)
            u_acc_vals = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(u_acc_addr_m, cutlass.Float32),
                num=cfg.b_t,
            )
            u_input_pack = cute.make_rmem_tensor((cfg.b_t // 2,), cutlass.Int32)
            for packed_col in cutlass.range_constexpr(cfg.b_t // 2):
                token0 = packed_col * 2
                token1 = token0 + 1
                u_input_pack[packed_col] = fp32_to_fp16(
                    u_acc_vals[token0], u_acc_vals[token1], dtype=cfg.io_dtype
                )
            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(u_input_addr_m, cutlass.Int8),
                u_input_pack.load(),
            )
            nvvm.tcgen05_wait("store")
            if nvvm.elect_sync():
                bars.mb_u_input_m_ready.arrive()
            u_acc_index = advance(u_acc_index, 1)

        if num_chunks_tile > 0:
            bars.mb_state_acc_h_cg1_done[state_update_index.idx].wait(state_update_index.phase)
            bars.mb_state_acc_m_cg1_done[state_update_index.idx].wait(state_update_index.phase)
            state_update_index = advance(state_update_index, cfg.smem_decay_stages)

        owns_final = write_end == batch_num_chunks

        # ---- final state stores; empty items pass the seeds through ------------------
        if batch_seqlen > 0:
            if owns_final:
                # ---- final state store: TMEM -> GMEM ---------------------------------
                state_vw = 16 // (mState_out.element_type.width // 8)
                state_dst = (
                    mState_out.iterator + mState_out.layout((batch_idx, head_o, value_dim, 0))
                ).raw_ptr()
                for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                    loaded = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            row_lo_addr + state_col_id_h + key_block_start, cutlass.Float32
                        ),
                        num=32,
                    )
                    for g in cutlass.range_constexpr(32 // state_vw):
                        if row_valid_h:
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

                # ---- final state store: TMEM -> GMEM ---------------------------------
                state_vw = 16 // (mTransition.element_type.width // 8)
                state_dst = (
                    mTransition.iterator + mTransition.layout((batch_idx, head_o, key_dim, 0))
                ).raw_ptr()
                for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                    loaded = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            row_lo_addr + state_col_id_m + key_block_start, cutlass.Float32
                        ),
                        num=32,
                    )
                    for g in cutlass.range_constexpr(32 // state_vw):
                        if row_valid_m:
                            (state_dst + key_block_start + g * state_vw).store(
                                cutlass.Vector.from_elements(
                                    tuple(
                                        loaded[g * state_vw + t].to(mTransition.element_type)
                                        for t in range(state_vw)
                                    ),
                                    mTransition.element_type,
                                ),
                                alignment=16,
                            )
        else:
            h_vw = 16 // (mState_out.element_type.width // 8)
            h_dst = (
                mState_out.iterator + mState_out.layout((batch_idx, head_o, value_dim, 0))
            ).raw_ptr()
            if cutlass.const_expr(mState_init is not None):
                seed_vw = 16 // (mState_init.element_type.width // 8)
                seed_src = (
                    mState_init.iterator + mState_init.layout((batch_idx, head_o, value_dim, 0))
                ).raw_ptr()
                for g in cutlass.range_constexpr(cfg.d_k // seed_vw):
                    seed_chunk = (seed_src + g * seed_vw).load(count=seed_vw, alignment=16)
                    for t in cutlass.range_constexpr(seed_vw):
                        if row_valid_h:
                            (h_dst + g * seed_vw + t).store(
                                seed_chunk[t].to(mState_out.element_type)
                            )
            else:
                h_zero = cutlass.Vector.from_elements(
                    tuple(cutlass.Float32(0.0).to(mState_out.element_type) for _ in range(h_vw)),
                    mState_out.element_type,
                )
                for g in cutlass.range_constexpr(cfg.d_k // h_vw):
                    if row_valid_h:
                        (h_dst + g * h_vw).store(h_zero, alignment=16)
            m_vw = 16 // (mTransition.element_type.width // 8)
            m_dst = (
                mTransition.iterator + mTransition.layout((batch_idx, head_o, key_dim, 0))
            ).raw_ptr()
            m_zero = cutlass.Vector.from_elements(
                tuple(cutlass.Float32(0.0).to(mTransition.element_type) for _ in range(m_vw)),
                mTransition.element_type,
            )
            for g in cutlass.range_constexpr(cfg.d_k // m_vw):
                if row_valid_m:
                    (m_dst + g * m_vw).store(m_zero, alignment=16)
            if row_valid_m:
                mTransition[batch_idx, head_o, key_dim, key_dim] = cutlass.Float32(1.0).to(
                    mTransition.element_type
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
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    n_batch: cutlass.Int32,
) -> None:
    """Per-batch descriptor-array build inside the prologue kernel after its order pass, one warp
    per array; warps past the array count fall through the widx guards."""
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

    if widx == 0:
        emit_seq_descs(base_k, desc_words_k, cu_seqlens, k, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 1:
        emit_seq_descs(base_v, desc_words_v, cu_seqlens, v, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 2:
        emit_seq_descs(base_gate, desc_words_gate, cu_seqlens, gate, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )


@cute.kernel
def frost_kda_summary_prologue(
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    base_k: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_v: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_gate: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    mStaging: cute.Tensor | None,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor | None,
    mScheduler: cute.Tensor | None,
    n_batch: cutlass.Int32,
) -> None:
    """Two-CTA prologue: under ``run_order`` block 0 LPT-orders the work-item table and zeroes the
    scheduler rings (:func:`order_body`); block 1 builds the per-batch TMA-descriptor arrays
    (:func:`build_descs_body`), one warp per array."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
        launch_dependent_grids()
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    if bidx == cutlass.Int32(0):
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
            desc_workspace,
            cu_seqlens,
            k,
            v,
            gate,
            n_batch,
        )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    cu_seqlens: cute.Tensor,
    work_item_staging: cute.Tensor | None,
    work_count: cute.Tensor,
    work_items: cute.Tensor | None,
    scheduler_all: cute.Tensor | None,
    tensormap_workspace: cute.Tensor,
    stream: cuda_driver.CUstream,
):
    """One-launch prologue: LPT-order the work items (``run_order``) and build the per-batch K / V
    / gate TMA-descriptor arrays into ``tensormap_workspace``."""
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

    frost_kda_summary_prologue(
        run_order,
        order_gen,
        b_t,
        base_k,
        base_v,
        base_gate,
        tensormap_workspace,
        cu_seqlens,
        k,
        v,
        gate,
        work_item_staging,
        work_count,
        work_items,
        scheduler_all,
        cutlass.Int32(batch_size),
    ).launch(grid=(2, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)


def _name_float(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m").replace("+", "")


class KdaSummaryOp:
    """The fused summary launch for one frozen ``KdaSummaryCfg``; standalone or nested into the
    chain hosts."""

    def __init__(self, cfg: "KdaSummaryCfg", use_int64_offsets: bool = False):
        self.cfg = cfg
        self.use_int64_offsets = use_int64_offsets

    def get_name(self) -> str:
        cfg = self.cfg
        flags = "".join(
            str(int(flag))
            for flag in (
                cfg.use_initial_state,
                cfg.log_gate,
            )
        )
        dtypes = "_".join(t.__name__.lower() for t in (cfg.io_dtype, cfg.gate_dtype))
        return (
            f"kda_cudnn_summary_{dtypes}_k{cfg.d_k}_v{cfg.d_v}_f{flags}"
            f"_g{_name_float(cfg.gate_scale_log2)}"
            f"_sm{cfg.max_active_clusters}_i64{int(self.use_int64_offsets)}"
        )

    @cute.jit
    def __call__(
        self,
        k: cute.Tensor,
        v: cute.Tensor,
        raw_gate: cute.Tensor,
        beta: cute.Tensor,
        cu_seqlens: cute.Tensor,
        initial_state: cute.Tensor | None,
        final_state: cute.Tensor,
        transition: cute.Tensor,
        work_items: cute.Tensor,
        work_count: cute.Tensor,
        scheduler_counter: cute.Tensor,
        tensormap_workspace: cute.Tensor,
        stream,
    ) -> None:
        cfg = self.cfg
        heads_out = cutlass.Int32(raw_gate.shape[1])
        k_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(k.shape[1]))
        v_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(v.shape[1]))
        num_sequences = cu_seqlens.shape[0] - 1

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
                cute.struct.MemRange[cutlass.Float32, cfg.gate_exchange_cosize], 1024
            ]  # 0 for fp32 Gate
            k_inv: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.k_inv_cosize], cfg.buffer_align_bytes
            ]
            beta: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cfg.beta_cosize], cfg.buffer_align_bytes
            ]

        frost_kda_summary.set_name_prefix(self.get_name())
        # ---- launch ----------------------------------------------------------------------
        grid_shape = (cfg.max_active_clusters, 1, 1)
        frost_kda_summary(
            cfg,
            SharedStorage,
            k_ratio,
            v_ratio,
            tensormap_workspace,
            cutlass.Int32(num_sequences),
            k,
            v,
            raw_gate,
            beta,
            cu_seqlens,
            initial_state,
            final_state,
            transition,
            work_items,
            work_count,
            scheduler_counter,
        ).launch(
            grid=grid_shape,
            block=(cfg.threads_per_cta, 1, 1),
            stream=stream,
            use_pdl=USE_PDL,
            min_blocks_per_mp=1,
        )


@cute.kernel
def frost_kda_summary(
    cfg: cutlass.Constexpr,
    shared_type: cutlass.Constexpr,
    k_ratio: cute.FastDivmodDivisorV2,
    v_ratio: cute.FastDivmodDivisorV2,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
    mK: cute.Tensor,
    mV: cute.Tensor,
    mGate: cute.Tensor,
    mBeta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    mState_init: cute.Tensor | None,
    mState_out: cute.Tensor,
    mTransition: cute.Tensor,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor,
) -> None:
    """BT=16 KDA fused H + M summary persistent kernel body: every warp role runs a
    tile-scheduler loop over the tiles."""
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

    # Barrier/control rings stay declaration-ordered; data buffers share one storage allocation.
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
    sBeta_raw = storage.beta.get_tensor(cute.make_layout((cfg.beta_cosize,)))
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
            bars.mb_state_k_acc_h_ready.init()
            bars.mb_u_acc_h_ready.init()
            bars.mb_state_input_h_cg1_ready.init()
            bars.mb_state_input_h_cg0_ready.init()
            bars.mb_y_input_h_ready.init()
            bars.mb_u_input_h_ready.init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_state_acc_h_cg0_done[stage].init()
                bars.mb_state_acc_h_cg1_done[stage].init()
            bars.mb_state_k_acc_m_ready.init()
            bars.mb_u_acc_m_ready.init()
            bars.mb_state_input_m_cg1_ready.init()
            bars.mb_state_input_m_cg0_ready.init()
            bars.mb_y_input_m_ready.init()
            bars.mb_u_input_m_ready.init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_state_acc_m_cg0_done[stage].init()
                bars.mb_state_acc_m_cg1_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_decay_tcgen05_done[stage].init()
                bars.mb_decay_register_mma_done[stage].init()
                bars.mb_k_restore_done[stage].init()
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
    elif warp_idx == cfg.register_mma_twin_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.scheduler_stages):
                bars.mb_scheduler_ready[stage].init()
                bars.mb_scheduler_done[stage].init()
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
            cutlass.Int32(0),
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
    elif warp_idx == cfg.register_mma_twin_warp_id:
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
            cutlass.Int32(1),
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
            sK_inv_raw,
            sGate_exchange_raw,
            sGate_load_ptr,
            mBeta,
            sBeta_raw,
            sK_raw,
            sK_decay_raw,
            sK_restore_raw,
            sTmem_base,
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
            mTransition,
            mState_init,
            sBeta_raw,
            sV_raw,
            sGate_exchange_raw,
            bars,
        )


@dataclass(frozen=True)
class KdaSummaryCfg:
    """Kernel cfg: fixed BT=16 schedule constants plus the TMEM column offsets and SMEM cosizes
    stamped by ``build_cfg``; passed ``cfg``-first as a ``cutlass.Constexpr`` into ``host`` /
    ``kernel`` and every warp body."""

    io_dtype: type[cutlass.Numeric]
    gate_dtype: type[cutlass.Numeric]
    use_initial_state: bool
    gate_scale_log2: float
    log_gate: bool
    max_active_clusters: int
    d_k: int
    d_v: int
    scheduler_stages: int = CFG.SMEM_SCHEDULER_STAGES

    compute_group_0_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_1_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    register_mma_warp_id: int = CFG.REGISTER_MMA_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    tma_warp_id: int = CFG.TMA_WARP_ID
    register_mma_twin_warp_id: int = CFG.REGISTER_MMA_TWIN_WARP_ID
    b_t: int = CFG.B_T
    threads_per_warp: int = CFG.THREADS_PER_WARP
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES
    threads_per_cta: int = 0  # derived by build_cfg
    cg0_group_count: int = 2
    cg0_warps_per_group: int = 4
    cg0_threads_per_group: int = 0  # derived by build_cfg
    cg0_group_sync_barrier_base_id: int = 1  # CG0 group g syncs on named-barrier id 1 + g
    cg0_tile_entry_barrier_id: int = 5  # CG0-wide (both groups) work-item entry sync
    tmem_user_threads: int = 0  # derived by build_cfg
    tmem_lifecycle_barrier_id: int = 3
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_other: int = CFG.NUM_REGS_OTHER

    # ---- SMEM / TMEM ring stage counts -----------------------------------------------
    smem_raw_stages: int = CFG.SMEM_RAW_STAGES
    smem_decay_stages: int = CFG.SMEM_DECAY_STAGES
    smem_intermediate_stages: int = CFG.SMEM_INTERMEDIATE_STAGES
    qk_scale_ready_stages: int = CFG.QK_SCALE_READY_STAGES

    # ---- TMEM column offsets of the two chains ---------------------------------------
    tmem_state_acc_h_offset: int = 0  # derived by build_cfg
    tmem_state_input_h_offset: int = 0  # derived by build_cfg
    tmem_state_k_acc_h_offset: int = 0  # derived by build_cfg
    tmem_u_acc_h_offset: int = 0  # derived by build_cfg
    tmem_y_input_h_offset: int = 0  # derived by build_cfg
    tmem_u_input_h_offset: int = 0  # derived by build_cfg
    tmem_state_acc_m_offset: int = 0  # derived by build_cfg
    tmem_state_input_m_offset: int = 0  # derived by build_cfg
    tmem_state_k_acc_m_offset: int = 0  # derived by build_cfg
    tmem_u_acc_m_offset: int = 0  # derived by build_cfg
    tmem_y_input_m_offset: int = 0  # derived by build_cfg
    tmem_u_input_m_offset: int = 0  # derived by build_cfg

    # ---- SMEM buffer cosizes ---------------------------------------------------------
    k_cosize: int = 0  # derived by build_cfg
    v_cosize: int = 0  # derived by build_cfg
    gate_cosize: int = 0  # derived by build_cfg
    gate_stage_elems: int = 0  # derived by build_cfg
    gate_exchange_stages: int = 0  # derived by build_cfg
    gate_exchange_cosize: int = 0  # derived by build_cfg
    beta_cosize: int = 0  # derived by build_cfg
    k_inv_cosize: int = 0  # derived by build_cfg
    k_decay_cosize: int = 0  # derived by build_cfg
    k_restore_cosize: int = 0  # derived by build_cfg

    # ---- TMA transaction bytes per stage ---------------------------------------------
    tma_k_bytes: int = 0  # derived by build_cfg
    tma_v_bytes: int = 0  # derived by build_cfg
    tma_gate_bytes: int = 0  # derived by build_cfg
    intermediate_cosize: int = 0  # derived by build_cfg


def build_cfg(
    io_dtype: type[cutlass.Numeric],
    gate_dtype: type[cutlass.Numeric],
    *,
    use_initial_state: bool,
    gate_scale_log2: float,
    log_gate: bool = True,
    max_active_clusters: int,
    d_k: int,
    d_v: int,
) -> KdaSummaryCfg:
    """Build the per-compile ``KdaSummaryCfg`` (io_dtype in {Float16, BFloat16});
    fills the derived TMEM column offsets and SMEM buffer cosizes."""
    validate_kernel_domain(
        "KDA summary", io_dtype, (d_k, d_v), max_active_clusters=max_active_clusters
    )
    cfg = KdaSummaryCfg(
        io_dtype=io_dtype,
        gate_dtype=gate_dtype,
        use_initial_state=use_initial_state,
        gate_scale_log2=gate_scale_log2,
        log_gate=log_gate,
        max_active_clusters=max_active_clusters,
        d_k=d_k,
        d_v=d_v,
    )
    if cfg.d_k not in STATE_DIMS or cfg.d_v not in STATE_DIMS:
        raise ValueError(
            f"the fused KDA summary serves DK, DV in {STATE_DIMS}, got DK={cfg.d_k} DV={cfg.d_v}"
        )
    if cfg.smem_raw_stages % 2 != 0:
        raise ValueError(
            "smem_raw_stages must be even: the CG0 ping-pong groups alias parity waits on odd "
            "rings"
        )
    if cfg.cg0_warps_per_group != len(cfg.compute_group_1_warp_ids):
        raise ValueError(
            "the state halves are packed by one CG0 group and by CG1: their warp counts must match"
        )
    cg0, per_group = cfg.compute_group_0_warp_ids, cfg.cg0_warps_per_group
    if len(cg0) != cfg.cg0_group_count * per_group:
        raise ValueError(
            "compute group 0 must hold cg0_group_count groups of cg0_warps_per_group warps"
        )
    n_warps = validate_warp_roles(
        (
            *(cg0[g * per_group : (g + 1) * per_group] for g in range(cfg.cg0_group_count)),
            cfg.compute_group_1_warp_ids,
        ),
        (
            cfg.register_mma_warp_id,
            cfg.tcgen05_mma_warp_id,
            cfg.tma_warp_id,
            cfg.register_mma_twin_warp_id,
        ),
    )
    b_t, d_k, d_v, raw = cfg.b_t, cfg.d_k, cfg.d_v, cfg.smem_raw_stages
    io_bytes, gate_bytes = cfg.io_dtype.width // 8, cfg.gate_dtype.width // 8
    offsets = {}
    base = 0
    for chain in ("h", "m"):
        offsets[f"tmem_state_acc_{chain}_offset"] = base
        offsets[f"tmem_state_input_{chain}_offset"] = base + d_k
        offsets[f"tmem_state_k_acc_{chain}_offset"] = base + d_k + d_k // 2
        offsets[f"tmem_u_acc_{chain}_offset"] = base + d_k + d_k // 2 + b_t
        offsets[f"tmem_y_input_{chain}_offset"] = base + d_k + d_k // 2 + 2 * b_t
        offsets[f"tmem_u_input_{chain}_offset"] = base + d_k + d_k // 2 + 2 * b_t + b_t // 2
        base = offsets[f"tmem_u_input_{chain}_offset"] + b_t // 2
    if base > 512:
        raise ValueError(f"TMEM layout exceeds 512 columns: {base}")
    # A Float32 gate exchanges in place (empty exchange buffer over the raw gate stages).
    gate_exchange_stages = raw if gate_dtype == cutlass.Float32 else 4
    gate_exchange_cosize = 0 if gate_dtype == cutlass.Float32 else gate_exchange_stages * d_k * b_t
    cfg = replace(
        cfg,
        threads_per_cta=n_warps * cfg.threads_per_warp,
        cg0_threads_per_group=cfg.cg0_warps_per_group * cfg.threads_per_warp,
        tmem_user_threads=(
            1 + len(cfg.compute_group_1_warp_ids) + len(cfg.compute_group_0_warp_ids)
        )
        * cfg.threads_per_warp,
        **offsets,
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
    validate_named_barriers(
        cfg.threads_per_cta,
        **{
            f"cg0_group_{g}": (cfg.cg0_group_sync_barrier_base_id + g, cfg.cg0_threads_per_group)
            for g in range(cfg.cg0_group_count)
        },
        cg0_tile_entry=(
            cfg.cg0_tile_entry_barrier_id,
            cfg.cg0_group_count * cfg.cg0_threads_per_group,
        ),
        tmem_lifecycle=(cfg.tmem_lifecycle_barrier_id, cfg.tmem_user_threads),
    )
    return cfg


TENSORMAP_DESC_ARRAYS = 3  # per-batch runtime TMA descriptors: K, V, Gate


# ---------------------------------------------------------------------------


def _dynamic(dtype, rank: int, align: int, use_int64_offsets: bool):
    """The legacy ``from_dlpack(t, assumed_align=align).mark_layout_dynamic()`` placeholder (last
    mode contiguous)."""
    return make_dynamic_signature_tensor(
        dtype, rank, assumed_align=align, use_int64_offsets=use_int64_offsets
    )


def _work_table(use_int64_offsets: bool):
    """The legacy work-item placeholder: compact ``[rows, WORK_ITEM_FIELDS]`` int32, only the row
    count dynamic."""
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    return make_compact_signature_tensor(
        cutlass.Int32, (sym_int(), WORK_ITEM_FIELDS), assumed_align=16
    )


@jit_cache
def _compile_kda_summary(
    io_dtype,
    gate_dtype,
    beta_dtype,
    cu_seqlens_dtype,
    state_in_dtype,
    d_k: int,
    d_v: int,
    gate_scale_log2: float,
    log_gate: bool,
    num_sm: int,
    use_int64_offsets: bool,
):
    """Compile the fused summary for one static config.  ``state_in_dtype`` is None when the
    tensor is absent.  Launch with live tensors in ``KdaSummaryOp.__call__`` order (k, v, gate,
    beta, cu_seqlens, initial_state, final_state, transition, work_items, work_count,
    scheduler_counter, tensormap_workspace) on the current Torch stream."""
    cfg = build_cfg(
        io_dtype,
        gate_dtype,
        use_initial_state=state_in_dtype is not None,
        gate_scale_log2=gate_scale_log2,
        log_gate=log_gate,
        max_active_clusters=num_sm,
        d_k=d_k,
        d_v=d_v,
    )
    i64 = use_int64_offsets
    cu_align = 8 if cu_seqlens_dtype is cutlass.Int64 else 4
    return compile_tvm_ffi(
        KdaSummaryOp(cfg, use_int64_offsets),
        _dynamic(io_dtype, 3, 16, i64),  # k
        _dynamic(io_dtype, 3, 16, i64),  # v
        _dynamic(gate_dtype, 3, 16, i64),  # gate
        _dynamic(beta_dtype, 2, 4, i64),  # beta
        _dynamic(cu_seqlens_dtype, 1, cu_align, i64),  # cu_seqlens
        _dynamic(state_in_dtype, 4, 16, i64) if state_in_dtype is not None else None,
        _dynamic(cutlass.Float32, 4, 16, i64),  # final_state (H)
        _dynamic(cutlass.Float32, 4, 16, i64),  # transition (M)
        _work_table(i64),  # work_items
        _dynamic(cutlass.Int32, 1, 4, i64),  # work_count
        _dynamic(cutlass.Int32, 1, 4, i64),  # scheduler_counter
        _dynamic(cutlass.Int64, 1, 128, i64),  # tensormap_workspace
        opt_level=2,
    )


@jit_cache
def _compile_kda_summary_prologue(
    io_dtype,
    gate_dtype,
    cu_seqlens_dtype,
    run_order: bool,
    order_gen: bool,
    use_int64_offsets: bool,
):
    """Compile the summary prologue for one static config.  Launch with live tensors in
    ``prologue`` order (k, v, gate, cu_seqlens, work_item_staging, work_count, work_items,
    scheduler_all, tensormap_workspace); the staging table is None unless ``run_order and not
    order_gen`` and ``scheduler_all`` is None unless ``run_order``."""
    i64 = use_int64_offsets
    cu_align = 8 if cu_seqlens_dtype is cutlass.Int64 else 4
    dtypes = "_".join(t.__name__.lower() for t in (io_dtype, gate_dtype, cu_seqlens_dtype))
    return compile_tvm_ffi(
        prologue,
        io_dtype,
        CFG.B_T,
        run_order,
        order_gen,
        _dynamic(io_dtype, 3, 16, i64),  # k
        _dynamic(io_dtype, 3, 16, i64),  # v
        _dynamic(gate_dtype, 3, 16, i64),  # gate
        _dynamic(cu_seqlens_dtype, 1, cu_align, i64),  # cu_seqlens
        _work_table(i64) if run_order and not order_gen else None,  # work_item_staging
        _dynamic(cutlass.Int32, 1, 4, i64),  # work_count
        _work_table(i64),  # work_items
        _dynamic(cutlass.Int32, 1, 4, i64) if run_order else None,  # scheduler_all
        _dynamic(cutlass.Int64, 1, 128, i64),  # tensormap_workspace
        name=(
            f"kda_cudnn_summary_prologue_{dtypes}_r{int(run_order)}_o{int(order_gen)}"
            f"_i64{int(i64)}"
        ),
        opt_level=2,
    )


frost_kda_summary_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
