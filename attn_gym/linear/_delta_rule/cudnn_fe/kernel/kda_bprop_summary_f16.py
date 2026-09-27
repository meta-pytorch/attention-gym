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
# Restyled register tensors, shared storage, and shared-memory pointer access. Fake-tensor TVM-FFI
# compilation is persisted through jit_cache.

"""
Chunked Kimi Delta Attention (KDA) bprop state-summary kernel for SM100 / SM103 / SM107 (Cutlass
primitives): the BT = 16 reverse state-gradient recurrence alone over q, k, Gate (+ a_log / dt_bias
under safe_gate), Beta, dO and an optional d_final_state seed (zero when absent); the single output
is d_initial_state.

Algorithm overview (per chunk c, iterated c = compute_end-1 .. write_start):
  Inputs : Q[BT,DK], K[BT,DK], dO[BT,DV], Gate[BT,DK] (per-channel gate), Beta[BT] (scalar LR)
  State  : dstate[DK,DV]  (state gradient, held in TMEM, seeded from d_final_state or zero, accumulated backward)

  Preprocessing (compute group 0): g, K decay, K inv, Q decay as in the prefill, the decay scale
  exp2(g[BT-1,:]). Register MMA (warp 12): KK = K decay @ K inv^T; L = Beta * tril(KK, -1); T_inv
  by the blockwise inverse. Register MMA (warp 15): A = tril(Q decay @ K inv^T, 0).

  MMA order (tcgen05, warp 13; (S) = SMEM operand, (T) = TMEM operand):
  dU inter      : dU inter = dstate input(T) @ K inv
  dU intra      : dU intra += dO^T(S) @ A
  dstate Q-term : dstate Q-term += dO^T(S) @ Q decay
  dY            : dY = dU(T) @ T^-1
  dstate K-term : dstate K-term += -Beta.dY(T) @ K decay

  Epilogue (compute group 1): dstate captured for the next chunk (f16 pack); the item owning chunk
  0 stores d_initial_state.

SMEM layout (stage counts live in kda_bprop_config.py and the kernel cfg; sizes at DK = DV = 128,
bf16 io, fp32 Gate):
  Buffer                       Size (B)  Stages
  Q / K (raw)                  2 x 4096       2
  dO (raw)                         4096       2
  Gate (raw)                       8192       2
  Beta                               64       4
  K decay / K inv / Q decay    3 x 4096       2
  decay scale                       512       2
  T_inv + A (intermediate)         1024       2
  scheduler ticket ring               4       8    <-- next-tile publish ring

TMEM layout (512 columns allocated):
  Buffer                  Cols
  dstate acc              128     <-- DKxDV fp32
  dstate input             64     <-- f16 packed
  dU acc                   16
  dY acc                   16
  -Beta.dY input            8     <-- f16 packed
  dU input                  8

Warp assignments (16 warps = 512 threads):
  warps 0-3     : compute group 0 - Gate prefix scan, decay operands
  warps 4-7     : compute group 1 - dstate seed, dU / -Beta.dY stagings, dstate capture, d_initial_state store
  warps 8-11    : exit after init
  warp  12      : register-MMA warp - KK and T_inv
  warp  13      : MMA warp       - every tcgen05 GEMM; TMEM lifecycle
  warp  14      : TMA load warp  - loads Q, K, Gate, dO
  warp  15      : epilogue warp  - register-MMA A tile
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
from ..tile_dsl.mma import mma_ss, mma_step, mma_ts_step
from ..tile_dsl.pointwise import (
    fadd2,
    ffma2,
    fmul2,
    fp32_to_fp16,
    opaque_f32_zero,
    sigmoid,
)
from ..tile_dsl.swizzle import swizzle_xor_32b, swizzle_xor_128b
from ..tile_dsl.tma import tma_load_tile, tma_tensormap_acquire
from .kda_bprop_config import CFG
from .kda_bprop_f16 import _validate_roles

USE_PDL = True

LOG2_E: float = 1.4426950408889634
DEFAULT_GATE_LOWER_BOUND: float = -5.0
L2_NORM_EPS: float = 1.0e-12


class KdaBpropSummaryBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_raw_ready: MBarrier
    mb_raw_done: MBarrier
    mb_do_ready: MBarrier
    mb_do_done: MBarrier

    mb_beta_ready: MBarrier
    mb_beta_done: MBarrier

    mb_du_acc_ready: MBarrier
    mb_dy_acc_ready: MBarrier

    mb_du_input_ready: MBarrier
    mb_neg_beta_dy_input_ready: MBarrier

    mb_k_decay_inv_ready: MBarrier
    mb_q_decay_ready: MBarrier
    mb_decay_done: MBarrier
    mb_decay_scale_ready: MBarrier
    mb_t_inv_ready: MBarrier
    mb_t_inv_done: MBarrier
    mb_a_ready: MBarrier
    mb_a_done: MBarrier

    mb_dstate_acc_ready: MBarrier
    mb_dstate_input_ready: MBarrier

    mb_dstate0_acc_stored: MBarrier
    mb_tmem_done: MBarrier

    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier


def make_bars(cfg) -> KdaBpropSummaryBars:
    """KdaBpropSummaryBars constructor."""

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=8)

    WARP = cfg.threads_per_warp
    CG0 = len(cfg.compute_group_0_warp_ids) * WARP
    CG1 = len(cfg.compute_group_1_warp_ids) * WARP
    MMA = 1

    return KdaBpropSummaryBars(
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
            init_count=CG0,
            producer=Producer.THREAD,
        ),
        mb_do_ready=MBarrier(
            alloc(cfg.smem_raw_stages),
            try_wait=True,
            stages=cfg.smem_raw_stages,
            init_count=1,
            producer=Producer.TMA_LOAD,
        ),
        mb_do_done=MBarrier(
            alloc(cfg.smem_raw_stages),
            try_wait=True,
            stages=cfg.smem_raw_stages,
            init_count=CG1,
            producer=Producer.THREAD,
        ),
        mb_beta_ready=MBarrier(
            alloc(cfg.smem_beta_stages),
            try_wait=True,
            stages=cfg.smem_beta_stages,
            init_count=WARP,
            producer=Producer.THREAD,
        ),
        mb_beta_done=MBarrier(
            alloc(cfg.smem_beta_stages),
            try_wait=True,
            stages=cfg.smem_beta_stages,
            init_count=WARP + CG1,
            producer=Producer.THREAD,
        ),
        mb_du_acc_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=MMA, producer=Producer.MMA_COMMIT
        ),
        mb_dy_acc_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=MMA, producer=Producer.MMA_COMMIT
        ),
        mb_du_input_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1, producer=Producer.THREAD
        ),
        mb_neg_beta_dy_input_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1, producer=Producer.THREAD
        ),
        mb_k_decay_inv_ready=MBarrier(
            alloc(cfg.smem_decay_stages),
            try_wait=True,
            stages=cfg.smem_decay_stages,
            init_count=CG0,
            producer=Producer.THREAD,
        ),
        mb_q_decay_ready=MBarrier(
            alloc(cfg.smem_decay_stages),
            try_wait=True,
            stages=cfg.smem_decay_stages,
            init_count=CG0,
            producer=Producer.THREAD,
        ),
        mb_decay_done=MBarrier(
            alloc(cfg.smem_decay_stages),
            try_wait=True,
            stages=cfg.smem_decay_stages,
            init_count=MMA,
            producer=Producer.MMA_COMMIT,
        ),
        mb_decay_scale_ready=MBarrier(
            alloc(cfg.smem_decay_stages),
            try_wait=True,
            stages=cfg.smem_decay_stages,
            init_count=CG0,
            producer=Producer.THREAD,
        ),
        mb_t_inv_ready=MBarrier(
            alloc(cfg.smem_intermediate_stages),
            try_wait=True,
            stages=cfg.smem_intermediate_stages,
            init_count=WARP,
            producer=Producer.THREAD,
        ),
        mb_t_inv_done=MBarrier(
            alloc(cfg.smem_intermediate_stages),
            try_wait=True,
            stages=cfg.smem_intermediate_stages,
            init_count=MMA,
            producer=Producer.MMA_COMMIT,
        ),
        mb_a_ready=MBarrier(
            alloc(cfg.smem_intermediate_stages),
            try_wait=True,
            stages=cfg.smem_intermediate_stages,
            init_count=WARP,
            producer=Producer.THREAD,
        ),
        mb_a_done=MBarrier(
            alloc(cfg.smem_intermediate_stages),
            try_wait=True,
            stages=cfg.smem_intermediate_stages,
            init_count=MMA,
            producer=Producer.MMA_COMMIT,
        ),
        mb_dstate_acc_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=MMA, producer=Producer.MMA_COMMIT
        ),
        mb_dstate_input_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1, producer=Producer.THREAD
        ),
        mb_dstate0_acc_stored=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1, producer=Producer.THREAD
        ),
        mb_tmem_done=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1, producer=Producer.THREAD
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
    lane_idx,
    sK_inv_raw,
    sQ_decay_raw,
    sIntermediate_raw,
    bars,
) -> None:
    """Epilogue warp role (warp 15): the register-MMA A tile per chunk."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    elect_one = nvvm.elect_sync()

    # ---- ldmatrix/stmatrix lane decode -----------------------------------------------
    b_row_coord = lane_idx % 8 + (cutlass.Int32(8) if (lane_idx // 16) else cutlass.Int32(0))
    b_col_offset = cutlass.Int32(8) if ((lane_idx // 8) % 2) else cutlass.Int32(0)
    a_row_coord = lane_idx % 8 + (cutlass.Int32(8) if ((lane_idx // 8) % 2) else cutlass.Int32(0))
    a_col_offset = cutlass.Int32(8) if ((lane_idx // 8) // 2) else cutlass.Int32(0)
    intermediate_row_coord = lane_idx & 7
    intermediate_col_coord = cutlass.Int32(0)
    if (lane_idx // 8) & 1:
        intermediate_row_coord = intermediate_row_coord + cutlass.Int32(8)
    if lane_idx // 8 >= 2:
        intermediate_col_coord = cutlass.Int32(8)
    intermediate_idx = intermediate_row_coord * cfg.b_t + swizzle_xor_32b(
        intermediate_row_coord, intermediate_col_coord
    )
    row_lo = lane_idx // 4
    row_hi = row_lo + cutlass.Int32(8)

    tril_incl_mask = cutlass.Int32(0)
    for accum_idx in cutlass.range_constexpr(8):
        row_coord = row_hi if cutlass.const_expr(accum_idx % 4 >= 2) else row_lo
        col_coord = (accum_idx // 4) * 8 + 2 * (lane_idx % 4)
        if cutlass.const_expr(accum_idx % 2 == 1):
            col_coord = col_coord + cutlass.Int32(1)
        tril_incl_mask = tril_incl_mask | (
            cutlass.Int32(1 << accum_idx) if row_coord >= col_coord else cutlass.Int32(0)
        )
    chunk_serial_base = cutlass.Int32(0)

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
            write_start,
            _write_end,
            _compute_start,
            compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        num_compute_chunks = compute_end - write_start
        for rev_idx in cutlass.range(num_compute_chunks, unroll=1):
            chunk_serial = chunk_serial_base + rev_idx
            decay_stage = chunk_serial % cfg.smem_decay_stages
            intermediate_stage = chunk_serial % cfg.smem_intermediate_stages
            sK_inv_ptr = smem_data_ptr(sK_inv_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sQ_decay_ptr = smem_data_ptr(sQ_decay_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sIntermediate_ptr = smem_data_ptr(sIntermediate_raw) + intermediate_stage * (
                cfg.intermediate_tiles * cfg.b_t * cfg.b_t
            )

            # ---- A = tril(Q decay @ K inv^T, 0) --------------------------------------
            bars.mb_a_done[intermediate_stage].wait(
                ((chunk_serial // cfg.smem_intermediate_stages) + 1) % 2
            )
            bars.mb_q_decay_ready[decay_stage].wait((chunk_serial // cfg.smem_decay_stages) % 2)
            a_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                a_acc[accum_idx] = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(cfg.d_k // 16):
                a_col = i * 16 + a_col_offset
                a_seg = a_col // 64
                q_decay_frag = nvvm.ldmatrix(
                    sQ_decay_ptr
                    + a_seg * (cfg.b_t * 64)
                    + a_row_coord * 64
                    + swizzle_xor_128b(a_row_coord, a_col - a_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                b_col = i * 16 + b_col_offset
                b_seg = b_col // 64
                k_inv_frag = nvvm.ldmatrix(
                    sK_inv_ptr
                    + b_seg * (cfg.b_t * 64)
                    + b_row_coord * 64
                    + swizzle_xor_128b(b_row_coord, b_col - b_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                mma_step(
                    a_acc,
                    (q_decay_frag[0], q_decay_frag[1], q_decay_frag[2], q_decay_frag[3]),
                    (k_inv_frag[0], k_inv_frag[1], k_inv_frag[2], k_inv_frag[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )
            for accum_idx in cutlass.range_constexpr(8):
                a_acc[accum_idx] = (
                    a_acc[accum_idx] if (tril_incl_mask >> accum_idx) & 1 else cutlass.Float32(0.0)
                )
            nvvm.stmatrix(
                sIntermediate_ptr + intermediate_idx,
                [
                    fp32_to_fp16(a_acc[0], a_acc[1], dtype=cfg.io_dtype),
                    fp32_to_fp16(a_acc[2], a_acc[3], dtype=cfg.io_dtype),
                    fp32_to_fp16(a_acc[4], a_acc[5], dtype=cfg.io_dtype),
                    fp32_to_fp16(a_acc[6], a_acc[7], dtype=cfg.io_dtype),
                ],
                nvvm.MMALayout.ROW,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_a_ready[intermediate_stage].arrive()

        chunk_serial_base += num_compute_chunks
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
    sK_decay_raw,
    sK_inv_raw,
    sIntermediate_raw,
    sBeta_raw,
    bars,
) -> None:
    """Register-MMA warp role (warp 12): the blockwise T_inv register MMAs in
    chunk order."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    elect_one = nvvm.elect_sync()

    # ---- ldmatrix/stmatrix lane decode -----------------------------------------------
    b_row_coord = lane_idx % 8 + (cutlass.Int32(8) if (lane_idx // 16) else cutlass.Int32(0))
    b_col_offset = cutlass.Int32(8) if ((lane_idx // 8) % 2) else cutlass.Int32(0)
    a_row_coord = lane_idx % 8 + (cutlass.Int32(8) if ((lane_idx // 8) % 2) else cutlass.Int32(0))
    a_col_offset = cutlass.Int32(8) if ((lane_idx // 8) // 2) else cutlass.Int32(0)
    intermediate_row_coord = lane_idx & 7
    intermediate_col_coord = cutlass.Int32(0)
    if (lane_idx // 8) & 1:
        intermediate_row_coord = intermediate_row_coord + cutlass.Int32(8)
    if lane_idx // 8 >= 2:
        intermediate_col_coord = cutlass.Int32(8)
    intermediate_idx = intermediate_row_coord * cfg.b_t + swizzle_xor_32b(
        intermediate_row_coord, intermediate_col_coord
    )
    row_lo = lane_idx // 4
    row_hi = row_lo + cutlass.Int32(8)

    # tril bitmask: bit i = row > col for accum index i
    tril_strict_mask = cutlass.Int32(0)
    for accum_idx in cutlass.range_constexpr(8):
        row_coord = row_hi if cutlass.const_expr(accum_idx % 4 >= 2) else row_lo
        col_coord = (accum_idx // 4) * 8 + 2 * (lane_idx % 4)
        if cutlass.const_expr(accum_idx % 2 == 1):
            col_coord = col_coord + cutlass.Int32(1)
        tril_strict_mask = tril_strict_mask | (
            cutlass.Int32(1 << accum_idx) if row_coord > col_coord else cutlass.Int32(0)
        )

    chunk_serial_base = cutlass.Int32(0)
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
            write_start,
            _write_end,
            _compute_start,
            compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        num_compute_chunks = compute_end - write_start
        for rev_idx in cutlass.range(num_compute_chunks, unroll=1):
            chunk_serial = chunk_serial_base + rev_idx
            decay_stage = chunk_serial % cfg.smem_decay_stages
            intermediate_stage = chunk_serial % cfg.smem_intermediate_stages
            sBeta_ptr = smem_data_ptr(sBeta_raw) + (chunk_serial % cfg.smem_beta_stages) * cfg.b_t
            sK_inv_ptr = smem_data_ptr(sK_inv_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = smem_data_ptr(sK_decay_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sIntermediate_ptr = smem_data_ptr(sIntermediate_raw) + intermediate_stage * (
                cfg.intermediate_tiles * cfg.b_t * cfg.b_t
            )

            bars.mb_t_inv_done[intermediate_stage].wait(
                ((chunk_serial // cfg.smem_intermediate_stages) + 1) % 2
            )

            # ---- KK = K decay @ K inv^T ----------------------------------------------
            bars.mb_k_decay_inv_ready[decay_stage].wait(
                (chunk_serial // cfg.smem_decay_stages) % 2
            )
            kk_a_row = a_row_coord
            kk_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                kk_acc[accum_idx] = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(cfg.d_k // 16):
                a_col = i * 16 + a_col_offset
                a_seg = a_col // 64
                k_decay_frag = nvvm.ldmatrix(
                    sK_decay_ptr
                    + a_seg * (cfg.b_t * 64)
                    + kk_a_row * 64
                    + swizzle_xor_128b(kk_a_row, a_col - a_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                b_col = i * 16 + b_col_offset
                b_seg = b_col // 64
                k_inv_frag = nvvm.ldmatrix(
                    sK_inv_ptr
                    + b_seg * (cfg.b_t * 64)
                    + b_row_coord * 64
                    + swizzle_xor_128b(b_row_coord, b_col - b_seg * 64, elem_bytes=2),
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
            bars.mb_beta_ready[chunk_serial % cfg.smem_beta_stages].wait(
                (chunk_serial // cfg.smem_beta_stages) % 2
            )
            beta_lo = (sBeta_ptr + row_lo).load().to(cutlass.Float32)
            beta_hi = (sBeta_ptr + row_hi).load().to(cutlass.Float32)
            bars.mb_beta_done[chunk_serial % cfg.smem_beta_stages].arrive()
            l_regs = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                beta_scale = beta_lo if accum_idx % 4 < 2 else beta_hi
                lower = (
                    kk_acc[accum_idx]
                    if (tril_strict_mask >> accum_idx) & 1
                    else cutlass.Float32(0.0)
                )
                l_regs[accum_idx] = lower * beta_scale

            # ---- T_inv = (I + L)^-1 --------------------------------------------------
            tinv_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            invert_unit_lower_16x16_fragments(cfg, l_regs, tinv_acc, lane_idx)

            nvvm.stmatrix(
                sIntermediate_ptr + 1 * (cfg.b_t * cfg.b_t) + intermediate_idx,
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
            bars.mb_t_inv_ready[intermediate_stage].arrive()
        chunk_serial_base += num_compute_chunks
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
    tmem_base_holder,
    sK_decay,
    sK_inv,
    sK_inv_trans,
    sDo,
    sDo_trans,
    sQ_decay_trans,
    sK_decay_trans,
    sIntermediate,
    bars,
) -> None:
    """tcgen05-MMA warp role (warp 13): issues every tcgen05 GEMM and owns
    the TMEM lifecycle."""
    elect_one = nvvm.elect_sync()

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    nvvm.tcgen05_alloc(tmem_base_holder, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_holder.load()
    bytes_per_element = cfg.io_dtype.width // 8

    # ---- chunk-invariant GEMM descriptors --------------------------------------------
    instruction_descriptor_mv_nt = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_dstate_k_inv_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_mv_nt,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    instruction_descriptor_du_at = nvvm.Tcgen05InstrDesc.build(
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
        idesc=instruction_descriptor_du_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    instruction_descriptor_dstate_q_at = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_k,
        m_dim=cfg.d_v,
        a_major=1,
        b_major=1,
    )
    bmm_do_q_decay_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.d_k,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        atranspose=True,
        cta_group=1,
        idesc=instruction_descriptor_dstate_q_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    instruction_descriptor_mv_nt_t = nvvm.Tcgen05InstrDesc.build(
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
        cta_group=1,
        idesc=instruction_descriptor_mv_nt_t,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    instruction_descriptor_dstate_k = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_k,
        m_dim=cfg.d_v,
        b_major=1,
    )
    bmm_dy_k_decay_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.d_k,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        cta_group=1,
        idesc=instruction_descriptor_dstate_k,
        kind=nvvm.Tcgen05MMAKind.F16,
    )

    dstate_input_index = PipelineState.start(phase=0)
    du_input_index = PipelineState.start(phase=0)
    neg_beta_dy_index = PipelineState.start(phase=0)

    do_seg = (cfg.b_t * cfg.d_v * (cfg.io_dtype.width // 8)) >> 4
    op_seg = (cfg.b_t * cfg.d_k * (cfg.io_dtype.width // 8)) >> 4
    intermediate_seg = (
        cfg.intermediate_tiles * cfg.b_t * cfg.b_t * (cfg.io_dtype.width // 8)
    ) >> 4
    intermediate_slot = (cfg.b_t * cfg.b_t * (cfg.io_dtype.width // 8)) >> 4
    d_do_trans0 = sDo_trans[0].desc()
    d_qd_trans0 = sQ_decay_trans[0].desc()
    d_kd_trans0 = sK_decay_trans[0].desc()
    d_ki_trans0 = sK_inv_trans[0].desc()
    d_int0 = sIntermediate[0].desc()
    sK_decay[0].desc()
    sDo[0].desc()
    d_ki0 = sK_inv[0].desc()
    dstate0_index = PipelineState.start(phase=0)

    chunk_serial_base = cutlass.Int32(0)
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
            write_start,
            _write_end,
            _compute_start,
            compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        num_compute_chunks = compute_end - write_start
        for rev_idx in cutlass.range(num_compute_chunks, unroll=1):
            chunk_serial = chunk_serial_base + rev_idx
            decay_stage = chunk_serial % cfg.smem_decay_stages
            intermediate_stage = chunk_serial % cfg.smem_intermediate_stages
            decay_phase = (chunk_serial // cfg.smem_decay_stages) % 2
            intermediate_phase = (chunk_serial // cfg.smem_intermediate_stages) % 2
            has_dstate = cutlass.Boolean(rev_idx > 0)
            if cutlass.const_expr(cfg.use_dstate_in):
                has_dstate = cutlass.Boolean(True)
            raw_stage_idx = chunk_serial % cfg.smem_raw_stages

            # ---- stage-derived operand descriptors -----------------------------------
            decay_op_off = decay_stage * op_seg
            d_do_trans = d_do_trans0 + raw_stage_idx * do_seg
            d_qd_trans = d_qd_trans0 + decay_op_off
            d_kd_trans = d_kd_trans0 + decay_op_off
            d_ki_trans0 + decay_op_off
            d_int = d_int0 + intermediate_stage * intermediate_seg
            d_int_tinv = d_int + intermediate_slot

            # ---- decay + dO operand guards -------------------------------------------
            bars.mb_k_decay_inv_ready[decay_stage].wait(decay_phase)
            bars.mb_do_ready[raw_stage_idx].wait((chunk_serial // cfg.smem_raw_stages) % 2)

            # ---- dU inter = dstate input(T) @ K inv ----------------------------------
            if has_dstate:
                bars.mb_dstate_input_ready.wait(dstate_input_index.phase)
                a_ptr = nvvm.make_tmem_ptr(
                    (tmem_base + cfg.tmem_dstate_input_offset), cutlass.Int8
                )
                b_desc = d_ki0 + decay_op_off
                c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_du_acc_offset), cutlass.Float32)
                for i in cutlass.range_constexpr(bmm_dstate_k_inv_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_dstate_k_inv_desc.sps_B):
                        mma_ts_step(
                            bmm_dstate_k_inv_desc,
                            a_ptr.subview(
                                i
                                * bmm_dstate_k_inv_desc.sps_B
                                * bmm_dstate_k_inv_desc.tmem_advance_A
                            ),
                            b_desc + i * (bmm_dstate_k_inv_desc.smem_subtile_B >> 4),
                            c_ptr,
                            k,
                            cutlass.Boolean(i + k > 0),
                        )
                dstate_input_index = advance(dstate_input_index, 1)

            # ---- dU intra += dO^T(S) @ A ---------------------------------------------
            bars.mb_a_ready[intermediate_stage].wait(intermediate_phase)
            mma_ss(
                bmm_do_a_desc,
                d_do_trans,
                d_int,
                nvvm.make_tmem_ptr((tmem_base + cfg.tmem_du_acc_offset), cutlass.Float32),
                accumulate=has_dstate,
            )
            if elect_one:
                bars.mb_du_acc_ready.arrive(cta_group=1)
                bars.mb_a_done[intermediate_stage].arrive(cta_group=1)

            # ---- dstate Q-term += dO^T(S) @ Q decay ----------------------------------
            bars.mb_q_decay_ready[decay_stage].wait(decay_phase)
            mma_ss(
                bmm_do_q_decay_desc,
                d_do_trans,
                d_qd_trans,
                nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dstate_acc_offset), cutlass.Float32),
                accumulate=has_dstate,
            )

            # ---- dY = dU(T) @ T^-1 ---------------------------------------------------
            bars.mb_t_inv_ready[intermediate_stage].wait(intermediate_phase)
            bars.mb_du_input_ready.wait(du_input_index.phase)
            du_input_index = advance(du_input_index, 1)
            a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_du_input_offset), cutlass.Int8)
            dy_b_desc = d_int_tinv
            c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dy_acc_offset), cutlass.Float32)
            for i in cutlass.range_constexpr(bmm_du_t_inv_trans_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_du_t_inv_trans_desc.sps_B):
                    mma_ts_step(
                        bmm_du_t_inv_trans_desc,
                        a_ptr.subview(
                            i
                            * bmm_du_t_inv_trans_desc.sps_B
                            * bmm_du_t_inv_trans_desc.tmem_advance_A
                        ),
                        dy_b_desc + i * (bmm_du_t_inv_trans_desc.smem_subtile_B >> 4),
                        c_ptr,
                        k,
                        cutlass.Boolean(i + k > 0),
                    )
            if elect_one:
                bars.mb_dy_acc_ready.arrive(cta_group=1)
                bars.mb_t_inv_done[intermediate_stage].arrive(cta_group=1)

            # ---- dstate K-term += -Beta.dY(T) @ K decay ------------------------------
            bars.mb_neg_beta_dy_input_ready.wait(neg_beta_dy_index.phase)
            neg_beta_dy_index = advance(neg_beta_dy_index, 1)
            a_ptr = nvvm.make_tmem_ptr(
                (tmem_base + cfg.tmem_neg_beta_dy_input_offset), cutlass.Int8
            )
            b_desc = d_kd_trans
            c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dstate_acc_offset), cutlass.Float32)
            for i in cutlass.range_constexpr(bmm_dy_k_decay_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_dy_k_decay_desc.sps_B):
                    mma_ts_step(
                        bmm_dy_k_decay_desc,
                        a_ptr.subview(
                            i * bmm_dy_k_decay_desc.sps_B * bmm_dy_k_decay_desc.tmem_advance_A
                        ),
                        b_desc + i * (bmm_dy_k_decay_desc.smem_subtile_B >> 4),
                        c_ptr,
                        k,
                        cutlass.Boolean(True),
                    )
            if elect_one:
                bars.mb_dstate_acc_ready.arrive(cta_group=1)
                bars.mb_decay_done[decay_stage].arrive(cta_group=1)

        # ---- tile end: dstate0 store wait --------------------------------------------
        if num_compute_chunks > 0:
            bars.mb_dstate0_acc_stored.wait(dstate0_index.phase)
            dstate0_index = advance(dstate0_index, 1)
        chunk_serial_base += num_compute_chunks
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
    sQ_raw,
    sK_raw,
    sGate_raw,
    sDo_raw,
    desc_q_base,
    desc_k_base,
    desc_gate_base,
    desc_do_base,
    bars,
    q_ratio,
    k_ratio,
) -> None:
    """TMA-LDG warp role (warp 14): persistent tile-scheduler loop issuing
    every G->S TMA load."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    raw_index = PipelineState.start(phase=1)
    scheduler_state = PipelineState.start(phase=1)

    elect_one = nvvm.elect_sync()
    sQ_tma = SmemTile(
        base=sQ_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
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
    gate_box_elems = cutlass.const_expr(128 // (cfg.gate_dtype.width // 8))
    sGate_tma = SmemTile(
        base=sGate_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // gate_box_elems),
        tma_granu_elems=gate_box_elems,
        tma_subtile_stride_elems=(cfg.b_t * 32),
    )
    sDo_tma = SmemTile(
        base=sDo_raw,
        elems_per_stage=(cfg.d_v * cfg.b_t),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_v // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
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
            write_start,
            _write_end,
            _compute_start,
            compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        head_o = head_idx
        head_q = head_idx // q_ratio
        head_k = head_idx // k_ratio
        slot = batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
        desc_q_slot = (desc_q_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_gate_slot = (desc_gate_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_do_slot = (desc_do_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_gate_slot)
            tma_tensormap_acquire(desc_do_slot)
        num_compute_chunks = compute_end - write_start
        for rev_idx in cutlass.range(num_compute_chunks, unroll=1):
            chunk_idx = compute_end - cutlass.Int32(1) - rev_idx
            chunk_start = chunk_idx * cfg.b_t

            # ---- Q / K / Gate loads: one transaction barrier per stage ---------------
            bars.mb_raw_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_raw_ready[raw_index.idx].arrive(
                    n_bytes=cfg.tma_q_bytes + cfg.tma_k_bytes + cfg.tma_gate_bytes
                )
            raw_ready_ptr = bars.mb_raw_ready[raw_index.idx].smem_ptr
            q_slice = tma_slice_runtime_desc(desc_q_slot, cutlass.Int32(0), head_q, chunk_start)
            tma_load_tile(sQ_tma[raw_index.idx], q_slice, raw_ready_ptr)
            k_slice = tma_slice_runtime_desc(desc_k_slot, cutlass.Int32(0), head_k, chunk_start)
            tma_load_tile(sK_tma[raw_index.idx], k_slice, raw_ready_ptr)
            gate_slice = tma_slice_runtime_desc(
                desc_gate_slot, cutlass.Int32(0), head_o, chunk_start
            )
            tma_load_tile(sGate_tma[raw_index.idx], gate_slice, raw_ready_ptr)

            # ---- dO load -------------------------------------------------------------
            bars.mb_do_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_do_ready[raw_index.idx].arrive(n_bytes=cfg.tma_do_bytes)
            do_slice = tma_slice_runtime_desc(desc_do_slot, cutlass.Int32(0), head_o, chunk_start)
            tma_load_tile(
                sDo_tma[raw_index.idx], do_slice, bars.mb_do_ready[raw_index.idx].smem_ptr
            )
            raw_index = advance(raw_index, cfg.smem_raw_stages)
        next_tile, scheduler_state = scheduler_publish_next(
            cfg, bars, sScheduler, mScheduler, scheduler_state, num_ctas, elect_one
        )
        tile_idx = next_tile
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
    scale,
    mA_log,
    mDt_bias,
    mBeta,
    sBeta_raw,
    sK_inv_raw,
    sGate_raw,
    sGate_load_ptr,
    sK_raw,
    sQ_raw,
    sK_decay_raw,
    sQ_decay_raw,
    sDecay_scale_raw,
    bars,
) -> None:
    """WG0 warp role (warps 0-3): persistent tile-scheduler loop running the gate prefix scan and
    materializing the decay operands into tcgen05 SMEM for every chunk."""
    nvvm.setmaxregister(
        cfg.num_regs_compute_group_0,
        nvvm.SetMaxRegisterAction.INCREASE
        if cfg.num_regs_compute_group_0 >= 65536 // cfg.threads_per_cta
        else nvvm.SetMaxRegisterAction.DECREASE,
    )
    elect_one = nvvm.elect_sync()
    cg0_warp = warp_idx - cfg.compute_group_0_warp_ids[0]
    dk_halves = cutlass.const_expr(cfg.d_k // 64)
    channel_rows = cutlass.const_expr(cfg.d_k // len(cfg.compute_group_0_warp_ids))
    channel_active = cutlass.Boolean(True)
    channel_dim = cg0_warp * cfg.threads_per_warp + lane_idx
    if cutlass.const_expr(channel_rows < cfg.threads_per_warp):
        channel_active = lane_idx < cutlass.Int32(channel_rows)
        channel_dim = cg0_warp * channel_rows + lane_idx % channel_rows
    cg0_a_log_exp = cutlass.Float32(1.0)
    cg0_dt_bias_value = cutlass.Float32(0.0)
    chunk_serial_base = cutlass.Int32(0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        (
            _batch_idx,
            head_idx,
            batch_start,
            _batch_end,
            batch_seqlen,
            _batch_num_chunks,
            write_start,
            _write_end,
            _compute_start,
            compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        num_compute_chunks = compute_end - write_start
        if cutlass.const_expr(mA_log is not None):
            if num_compute_chunks > 0:
                cg0_a_log_exp = cute.math.exp2(
                    mA_log[head_idx].to(cutlass.Float32) * LOG2_E, fastmath=True
                )
        if cutlass.const_expr(mDt_bias is not None):
            if num_compute_chunks > 0:
                cg0_dt_bias_value = mDt_bias[head_idx, channel_dim].to(cutlass.Float32)
        for rev_idx in cutlass.range(num_compute_chunks, unroll=1):
            chunk_idx = compute_end - cutlass.Int32(1) - rev_idx
            chunk_serial = chunk_serial_base + rev_idx
            chunk_start = chunk_idx * cfg.b_t
            decay_stage = chunk_serial % cfg.smem_decay_stages
            raw_stage = chunk_serial % cfg.smem_raw_stages
            sQ_ptr = smem_data_ptr(sQ_raw) + raw_stage * (cfg.d_k * cfg.b_t)
            sK_ptr = smem_data_ptr(sK_raw) + raw_stage * (cfg.d_k * cfg.b_t)
            sGate_ptr = sGate_load_ptr + raw_stage * cfg.gate_stage_elems
            sGate_exchange_ptr = smem_data_ptr(sGate_raw) + raw_stage * (cfg.d_k * cfg.b_t)
            sK_inv_ptr = smem_data_ptr(sK_inv_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = smem_data_ptr(sK_decay_raw) + decay_stage * (cfg.d_k * cfg.b_t)
            sQ_decay_ptr = smem_data_ptr(sQ_decay_raw) + decay_stage * (cfg.d_k * cfg.b_t)
            sDecay_scale_ptr = smem_data_ptr(sDecay_scale_raw) + decay_stage * cfg.d_k

            # ---- Beta scalars --------------------------------------------------------
            if cg0_warp == 0:
                beta_stage = chunk_serial % cfg.smem_beta_stages
                bars.mb_beta_done[beta_stage].wait(
                    ((chunk_serial // cfg.smem_beta_stages) + 1) % 2
                )
                if lane_idx < cfg.b_t:
                    token_idx = chunk_start + lane_idx
                    beta_value = cutlass.Float32(0.0)
                    if token_idx < batch_seqlen:
                        beta_value = mBeta[batch_start + token_idx, head_idx].to(cutlass.Float32)
                        if cutlass.const_expr(cfg.beta_sigmoid):
                            beta_value = (
                                (sigmoid(beta_value) * (2.0 if cfg.allow_neg_eigval else 1.0))
                                .to(mBeta.element_type)
                                .to(cutlass.Float32)
                            )
                    sBeta_raw[beta_stage * cfg.b_t + lane_idx] = beta_value
                bars.mb_beta_ready[beta_stage].arrive()

            bars.mb_raw_ready[raw_stage].wait((chunk_serial // cfg.smem_raw_stages) % 2)

            row_group_start = cg0_warp * (cfg.b_t // len(cfg.compute_group_0_warp_ids))
            lane_row_group = lane_idx // 8
            lane_in_row_group = lane_idx - lane_row_group * 8
            decay_row = row_group_start + lane_row_group

            # ---- Gate prefix scan: cumulative log-gate per key channel ---------------
            f32_segment = channel_dim // 32
            f32_segment_dim = channel_dim - f32_segment * 32
            if cutlass.const_expr(cfg.gate_dtype != cutlass.Float32):
                raw_segment = channel_dim // 64
                raw_seg_base = raw_segment * (cfg.b_t * 64)
                raw_col = channel_dim - raw_segment * 64
            prefix_acc = cutlass.Float32(0.0)
            exp_g_last = cutlass.Float32(0.0)
            if cutlass.const_expr(cfg.gate_dtype == cutlass.Float32):
                for row_pair in cutlass.range_constexpr(cfg.b_t // 2):
                    row0 = row_pair * 2
                    row1 = row0 + 1
                    prefix_idx0 = (
                        f32_segment * (cfg.b_t * 32)
                        + row0 * 32
                        + swizzle_xor_128b(row0, f32_segment_dim, elem_bytes=4)
                    )
                    prefix_idx1 = (
                        f32_segment * (cfg.b_t * 32)
                        + row1 * 32
                        + swizzle_xor_128b(row1, f32_segment_dim, elem_bytes=4)
                    )
                    gate0 = (sGate_ptr + prefix_idx0).load()
                    gate1 = (sGate_ptr + prefix_idx1).load()
                    token_idx0 = chunk_idx * cutlass.Int32(cfg.b_t) + row0
                    token_idx1 = chunk_idx * cutlass.Int32(cfg.b_t) + row1
                    if cutlass.const_expr(cfg.safe_gate):
                        if token_idx0 < batch_seqlen:
                            gate0 = gate_scale(cfg, cg0_a_log_exp * (gate0 + cg0_dt_bias_value))
                        else:
                            gate0 = cutlass.Float32(0.0)
                        if token_idx1 < batch_seqlen:
                            gate1 = gate_scale(cfg, cg0_a_log_exp * (gate1 + cg0_dt_bias_value))
                        else:
                            gate1 = cutlass.Float32(0.0)
                    else:
                        if token_idx0 < batch_seqlen:
                            gate0 = gate_scale(cfg, gate0)
                        else:
                            gate0 = cutlass.Float32(0.0)
                        if token_idx1 < batch_seqlen:
                            gate1 = gate_scale(cfg, gate1)
                        else:
                            gate1 = cutlass.Float32(0.0)
                    prefix0, row_pair_sum = fadd2(prefix_acc, gate0, gate0, gate1)
                    prefix1 = prefix_acc + row_pair_sum
                    exp_g0 = cute.math.exp2(prefix0, fastmath=True)
                    exp_g1 = cute.math.exp2(prefix1, fastmath=True)
                    if channel_active:
                        (
                            sGate_exchange_ptr
                            + f32_segment * (cfg.b_t * 32)
                            + row0 * 32
                            + swizzle_xor_128b(row0 ^ f32_segment, f32_segment_dim, elem_bytes=4)
                        ).store(exp_g0)
                        (
                            sGate_exchange_ptr
                            + f32_segment * (cfg.b_t * 32)
                            + row1 * 32
                            + swizzle_xor_128b(row1 ^ f32_segment, f32_segment_dim, elem_bytes=4)
                        ).store(exp_g1)
                    prefix_acc = prefix1
                    exp_g_last = exp_g1
            else:
                gate_raw = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
                for row_pair in cutlass.range_constexpr(cfg.b_t // 2):
                    row0 = row_pair * 2
                    row1 = row0 + 1
                    raw_idx0 = raw_seg_base + swizzle_xor_128b(
                        row0, row0 * 64 + raw_col, elem_bytes=2
                    )
                    raw_idx1 = raw_seg_base + swizzle_xor_128b(
                        row1, row1 * 64 + raw_col, elem_bytes=2
                    )
                    gate0 = (sGate_ptr + raw_idx0).load().to(cutlass.Float32)
                    gate1 = (sGate_ptr + raw_idx1).load().to(cutlass.Float32)
                    token_idx0 = chunk_idx * cutlass.Int32(cfg.b_t) + row0
                    token_idx1 = chunk_idx * cutlass.Int32(cfg.b_t) + row1
                    if cutlass.const_expr(cfg.safe_gate):
                        if token_idx0 < batch_seqlen:
                            gate0 = gate_scale(cfg, cg0_a_log_exp * (gate0 + cg0_dt_bias_value))
                        else:
                            gate0 = cutlass.Float32(0.0)
                        if token_idx1 < batch_seqlen:
                            gate1 = gate_scale(cfg, cg0_a_log_exp * (gate1 + cg0_dt_bias_value))
                        else:
                            gate1 = cutlass.Float32(0.0)
                    else:
                        if token_idx0 < batch_seqlen:
                            gate0 = gate_scale(cfg, gate0)
                        else:
                            gate0 = cutlass.Float32(0.0)
                        if token_idx1 < batch_seqlen:
                            gate1 = gate_scale(cfg, gate1)
                        else:
                            gate1 = cutlass.Float32(0.0)
                    gate_raw[row0] = gate0
                    gate_raw[row1] = gate1
                nvvm.barrier_cta_sync(cfg.cg0_sync_barrier_id, thread_count=cfg.cg0_threads)
                for row_pair in cutlass.range_constexpr(cfg.b_t // 2):
                    row0 = row_pair * 2
                    row1 = row0 + 1
                    prefix_idx0 = (
                        f32_segment * (cfg.b_t * 32)
                        + row0 * 32
                        + swizzle_xor_128b(row0 ^ f32_segment, f32_segment_dim, elem_bytes=4)
                    )
                    prefix_idx1 = (
                        f32_segment * (cfg.b_t * 32)
                        + row1 * 32
                        + swizzle_xor_128b(row1 ^ f32_segment, f32_segment_dim, elem_bytes=4)
                    )
                    gate0 = gate_raw[row0]
                    gate1 = gate_raw[row1]
                    prefix0, row_pair_sum = fadd2(prefix_acc, gate0, gate0, gate1)
                    prefix1 = prefix_acc + row_pair_sum
                    exp_g0 = cute.math.exp2(prefix0, fastmath=True)
                    exp_g1 = cute.math.exp2(prefix1, fastmath=True)
                    if channel_active:
                        (sGate_exchange_ptr + prefix_idx0).store(exp_g0)
                        (sGate_exchange_ptr + prefix_idx1).store(exp_g1)
                    prefix_acc = prefix1
                    exp_g_last = exp_g1

            # ---- decay-slot guard ----------------------------------------------------
            operand_done_phase = ((chunk_serial // cfg.smem_decay_stages) + 1) % 2
            bars.mb_decay_done[decay_stage].wait(operand_done_phase)

            # ---- decay scale: exp2(g last) per key channel ---------------------------
            if channel_active:
                (sDecay_scale_ptr + channel_dim).store(exp_g_last)
            bars.mb_decay_scale_ready[decay_stage].arrive()

            nvvm.barrier_cta_sync(cfg.cg0_sync_barrier_id, thread_count=cfg.cg0_threads)

            k_inv_pack = cute.make_rmem_tensor((dk_halves * 4,), cutlass.Int32)
            raw_q_regs = cute.make_rmem_tensor((dk_halves * 8,), cutlass.Float32)
            raw_k_regs = cute.make_rmem_tensor((dk_halves * 8,), cutlass.Float32)

            # ---- optional Q/K L2-norm ------------------------------------------------
            if cutlass.const_expr(cfg.l2norm):
                qk0_lo = opaque_f32_zero()
                qk0_hi = opaque_f32_zero()
                qk1_lo = opaque_f32_zero()
                qk1_hi = opaque_f32_zero()
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
                raw_q_frag = (sQ_ptr + raw_f16_idx).load(count=8, alignment=16)
                raw_k_frag = (sK_ptr + raw_f16_idx).load(count=8, alignment=16)
                raw_q_frag_f32 = raw_q_frag.to(cutlass.Float32)
                raw_k_frag_f32 = raw_k_frag.to(cutlass.Float32)
                for dim_offset in cutlass.range_constexpr(8):
                    q_val = raw_q_frag_f32[dim_offset]
                    k_val = raw_k_frag_f32[dim_offset]
                    raw_q_regs[reg_base + dim_offset] = q_val
                    raw_k_regs[reg_base + dim_offset] = k_val
                    if cutlass.const_expr(cfg.l2norm):
                        if cutlass.const_expr(dim_offset % 2 == 0):
                            qk0_lo, qk0_hi = ffma2(q_val, k_val, q_val, k_val, qk0_lo, qk0_hi)
                        else:
                            qk1_lo, qk1_hi = ffma2(q_val, k_val, q_val, k_val, qk1_lo, qk1_hi)

            nvvm.fence_proxy("async.shared", space="cta")

            q_inv_norm = opaque_f32_zero() + cutlass.Float32(1.0)
            k_inv_norm = opaque_f32_zero() + cutlass.Float32(1.0)
            if cutlass.const_expr(cfg.l2norm):
                q_sum_sq = qk0_lo + qk1_lo
                k_sum_sq = qk0_hi + qk1_hi
                q_sum_sq = q_sum_sq + cutlass.Float32(
                    nvvm.shfl_sync(0xFFFFFFFF, q_sum_sq, 4, 31, kind=nvvm.Shfl.BFLY)
                )
                q_sum_sq = q_sum_sq + cutlass.Float32(
                    nvvm.shfl_sync(0xFFFFFFFF, q_sum_sq, 2, 31, kind=nvvm.Shfl.BFLY)
                )
                q_sum_sq = q_sum_sq + cutlass.Float32(
                    nvvm.shfl_sync(0xFFFFFFFF, q_sum_sq, 1, 31, kind=nvvm.Shfl.BFLY)
                )
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
                q_inv_norm = cute.math.rsqrt(cute.math.max(q_sum_sq, norm_floor_sq), fastmath=True)
                k_inv_norm = cute.math.rsqrt(cute.math.max(k_sum_sq, norm_floor_sq), fastmath=True)
            q_stage_norm = q_inv_norm * scale

            # ---- decay operands: exp2(+-g) applied per key channel -------------------
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
                    exp_g_regs[f32_reg_base] = exp_g_frag[0]
                    exp_g_regs[f32_reg_base + 1] = exp_g_frag[1]
                    exp_g_regs[f32_reg_base + 2] = exp_g_frag[2]
                    exp_g_regs[f32_reg_base + 3] = exp_g_frag[3]
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_raw_done[raw_stage].arrive()

            for dim_half in cutlass.range_constexpr(dk_halves):
                dim_base = dim_half * 64 + lane_in_row_group * 8
                reg_base = dim_half * 8
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64

                # ---- K decay + K inv operands: exp2(+g) * K, exp2(-g) * K ------------
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
                    (k_decay_pack[0], k_decay_pack[1], k_decay_pack[2], k_decay_pack[3]),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                op_idx = (
                    f16_segment * (cfg.b_t * 64)
                    + decay_row * 64
                    + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                )
                (sK_inv_ptr + op_idx).store(k_inv_vec, alignment=16)
                (sK_decay_ptr + op_idx).store(k_decay_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_k_decay_inv_ready[decay_stage].arrive()

            # ---- Q decay operand -----------------------------------------------------
            for dim_half in cutlass.range_constexpr(dk_halves):
                dim_base = dim_half * 64 + lane_in_row_group * 8
                reg_base = dim_half * 8
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                q_decay_pack = cute.make_rmem_tensor((4,), cutlass.Int32)
                for pair_idx in cutlass.range_constexpr(4):
                    dim0 = pair_idx * 2
                    dim1 = dim0 + 1
                    raw_reg_idx0 = reg_base + dim0
                    raw_reg_idx1 = reg_base + dim1
                    q_value0, q_value1 = fmul2(
                        raw_q_regs[raw_reg_idx0],
                        raw_q_regs[raw_reg_idx1],
                        q_stage_norm,
                        q_stage_norm,
                    )
                    q_decay0, q_decay1 = fmul2(
                        q_value0, q_value1, exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1]
                    )
                    q_decay_pack[pair_idx] = fp32_to_fp16(q_decay0, q_decay1, dtype=cfg.io_dtype)

                q_decay_vec = cutlass.Vector.from_elements(
                    (q_decay_pack[0], q_decay_pack[1], q_decay_pack[2], q_decay_pack[3]),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                op_idx = (
                    f16_segment * (cfg.b_t * 64)
                    + decay_row * 64
                    + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                )
                (sQ_decay_ptr + op_idx).store(q_decay_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_q_decay_ready[decay_stage].arrive()
        chunk_serial_base += num_compute_chunks
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
    tmem_base_holder,
    warp_idx,
    mDstate0,
    mDstate_in,
    sBeta_raw,
    sDecay_scale_raw,
    bars,
) -> None:
    """WG1 warp role (warps 4-7): the value-side TMEM staging for the reverse
    dstate recurrence, the d_final_state seed, and the d_initial_state store."""
    nvvm.setmaxregister(
        cfg.num_regs_compute_group_1,
        nvvm.SetMaxRegisterAction.INCREASE
        if cfg.num_regs_compute_group_1 >= 65536 // cfg.threads_per_cta
        else nvvm.SetMaxRegisterAction.DECREASE,
    )
    elect_one = nvvm.elect_sync()
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_holder.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    if cutlass.const_expr(cfg.d_v == 128):
        tmem_subpartition = warp_idx % (cfg.d_v // cfg.threads_per_warp)
        value_dim = tmem_subpartition * cfg.threads_per_warp + lane_idx
        state_row_valid = cutlass.Boolean(True)
    else:
        cg1_warp = warp_idx % len(cfg.compute_group_1_warp_ids)
        value_dim = cg1_warp * 16 + lane_idx % 16
        state_row_valid = lane_idx < 16

    raw_index = PipelineState.start(phase=0)
    du_acc_index = PipelineState.start(phase=0)
    dy_acc_index = PipelineState.start(phase=0)
    dstate_ready_index = PipelineState.start(phase=0)

    chunk_serial_base = cutlass.Int32(0)
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
            _write_end,
            _compute_start,
            compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        num_compute_chunks = compute_end - write_start

        # ---- dstate seed: GMEM -> TMEM -----------------------------------------------
        if cutlass.const_expr(cfg.use_dstate_in):
            if num_compute_chunks > 0:
                seed_true = compute_end == batch_num_chunks
                seed_stage = chunk_serial_base % cfg.smem_decay_stages
                bars.mb_decay_scale_ready[seed_stage].wait(
                    (chunk_serial_base // cfg.smem_decay_stages) % 2
                )
                seed_scale_ptr = smem_data_ptr(sDecay_scale_raw) + seed_stage * cfg.d_k
                row_lo_addr = tmem_row << 16
                seed_vw = 16 // (mDstate_in.element_type.width // 8)
                dstate_src = (
                    mDstate_in.iterator + mDstate_in.layout((batch_idx, head_idx, value_dim, 0))
                ).raw_ptr()
                for i in cutlass.range(cfg.d_k // 16, unroll=1):
                    seed_block = cute.make_rmem_tensor((16,), cutlass.Float32)
                    for g in cutlass.range_constexpr(16 // seed_vw):
                        seed_chunk = (dstate_src + i * 16 + g * seed_vw).load(
                            count=seed_vw, alignment=16
                        )
                        for t in cutlass.range_constexpr(seed_vw):
                            dval = seed_chunk[t].to(cutlass.Float32)
                            seed_block[g * seed_vw + t] = (
                                dval if seed_true else cutlass.Float32(0.0)
                            )
                    for g in cutlass.range_constexpr(4):
                        seed_frag = (seed_scale_ptr + i * 16 + g * 4).load(count=4, alignment=16)
                        for t in cutlass.range_constexpr(2):
                            seed_block[g * 4 + 2 * t], seed_block[g * 4 + 2 * t + 1] = fmul2(
                                seed_block[g * 4 + 2 * t],
                                seed_block[g * 4 + 2 * t + 1],
                                seed_frag[2 * t],
                                seed_frag[2 * t + 1],
                            )
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            row_lo_addr + (tmem_col + cfg.tmem_dstate_acc_offset + i * 16),
                            cutlass.Float32,
                        ),
                        seed_block.load(),
                    )
                    seed_pack = cute.make_rmem_tensor((8,), cutlass.Int32)
                    for pc in cutlass.range_constexpr(8):
                        seed_pack[pc] = fp32_to_fp16(
                            seed_block[2 * pc], seed_block[2 * pc + 1], dtype=cfg.io_dtype
                        )
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            row_lo_addr + (tmem_col + cfg.tmem_dstate_input_offset + i * 8),
                            cutlass.Int8,
                        ),
                        seed_pack.load(),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_input_ready.arrive()

        for rev_idx in cutlass.range(num_compute_chunks, unroll=1):
            chunk_serial = chunk_serial_base + rev_idx
            sBeta_ptr = smem_data_ptr(sBeta_raw) + (chunk_serial % cfg.smem_beta_stages) * cfg.b_t
            row_lo_addr = tmem_row << 16
            row_hi_addr = (tmem_row + 16) << 16

            # ---- dU stage: dU acc -> TMEM f16 ----------------------------------------
            bars.mb_du_acc_ready.wait(du_acc_index.phase)
            du_acc_index = advance(du_acc_index, 1)
            du_col_id = tmem_col + cfg.tmem_du_acc_offset
            du_vec_lo = nvvm.tcgen05_ld(
                "16x256b", nvvm.make_tmem_ptr(row_lo_addr + du_col_id, cutlass.Float32), num=2
            )
            if cutlass.const_expr(cfg.d_v == 128):
                du_vec_hi = nvvm.tcgen05_ld(
                    "16x256b", nvvm.make_tmem_ptr(row_hi_addr + du_col_id, cutlass.Float32), num=2
                )

            du_pack_lo = cute.make_rmem_tensor((4,), cutlass.Int32)
            du_pack_hi = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                frag_pair = reg_idx * 2
                du_pack_lo[reg_idx] = fp32_to_fp16(
                    du_vec_lo[frag_pair], du_vec_lo[frag_pair + 1], dtype=cfg.io_dtype
                )
                if cutlass.const_expr(cfg.d_v == 128):
                    du_pack_hi[reg_idx] = fp32_to_fp16(
                        du_vec_hi[frag_pair], du_vec_hi[frag_pair + 1], dtype=cfg.io_dtype
                    )
            nvvm.tcgen05_st(
                "16x128b",
                nvvm.make_tmem_ptr(
                    row_lo_addr + (tmem_col + cfg.tmem_du_input_offset), cutlass.Int8
                ),
                du_pack_lo.load(),
            )
            if cutlass.const_expr(cfg.d_v == 128):
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(
                        row_hi_addr + (tmem_col + cfg.tmem_du_input_offset), cutlass.Int8
                    ),
                    du_pack_hi.load(),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_du_input_ready.arrive()

            # ---- dY read -------------------------------------------------------------
            bars.mb_dy_acc_ready.wait(dy_acc_index.phase)
            dy_acc_index = advance(dy_acc_index, 1)
            dy_col_id = tmem_col + cfg.tmem_dy_acc_offset
            dy_vec_lo = nvvm.tcgen05_ld(
                "16x256b", nvvm.make_tmem_ptr(row_lo_addr + dy_col_id, cutlass.Float32), num=2
            )
            if cutlass.const_expr(cfg.d_v == 128):
                dy_vec_hi = nvvm.tcgen05_ld(
                    "16x256b", nvvm.make_tmem_ptr(row_hi_addr + dy_col_id, cutlass.Float32), num=2
                )

            # ---- Beta scalars --------------------------------------------------------
            bars.mb_beta_ready[chunk_serial % cfg.smem_beta_stages].wait(
                (chunk_serial // cfg.smem_beta_stages) % 2
            )
            beta_c0 = (sBeta_ptr + (lane_idx % 4) * 2).load().to(cutlass.Float32)
            beta_c1 = (sBeta_ptr + (lane_idx % 4) * 2 + 1).load().to(cutlass.Float32)
            beta_c8 = (sBeta_ptr + (lane_idx % 4) * 2 + 8).load().to(cutlass.Float32)
            beta_c9 = (sBeta_ptr + (lane_idx % 4) * 2 + 9).load().to(cutlass.Float32)
            bars.mb_beta_done[chunk_serial % cfg.smem_beta_stages].arrive()
            beta_dy_regs_lo = cute.make_rmem_tensor((8,), cutlass.Float32)
            beta_dy_regs_hi = cute.make_rmem_tensor((8,), cutlass.Float32)
            for e2 in cutlass.range_constexpr(4):
                e = 2 * e2
                b_lo = beta_c8 if cutlass.const_expr(e >= 4) else beta_c0
                b_hi = beta_c9 if cutlass.const_expr(e >= 4) else beta_c1
                beta_dy_regs_lo[e], beta_dy_regs_lo[e + 1] = fmul2(
                    dy_vec_lo[e], dy_vec_lo[e + 1], b_lo, b_hi
                )
                if cutlass.const_expr(cfg.d_v == 128):
                    beta_dy_regs_hi[e], beta_dy_regs_hi[e + 1] = fmul2(
                        dy_vec_hi[e], dy_vec_hi[e + 1], b_lo, b_hi
                    )

            # ---- -Beta.dY -> TMEM ----------------------------------------------------
            neg_beta_dy_regs_lo = cute.make_rmem_tensor((8,), cutlass.Float32)
            neg_beta_dy_regs_hi = cute.make_rmem_tensor((8,), cutlass.Float32)
            for e in cutlass.range_constexpr(8):
                neg_beta_dy_regs_lo[e] = -beta_dy_regs_lo[e]
                if cutlass.const_expr(cfg.d_v == 128):
                    neg_beta_dy_regs_hi[e] = -beta_dy_regs_hi[e]
            neg_beta_dy_pack_lo = cute.make_rmem_tensor((4,), cutlass.Int32)
            neg_beta_dy_pack_hi = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                frag_pair = reg_idx * 2
                neg_beta_dy_pack_lo[reg_idx] = fp32_to_fp16(
                    neg_beta_dy_regs_lo[frag_pair],
                    neg_beta_dy_regs_lo[frag_pair + 1],
                    dtype=cfg.io_dtype,
                )
                if cutlass.const_expr(cfg.d_v == 128):
                    neg_beta_dy_pack_hi[reg_idx] = fp32_to_fp16(
                        neg_beta_dy_regs_hi[frag_pair],
                        neg_beta_dy_regs_hi[frag_pair + 1],
                        dtype=cfg.io_dtype,
                    )
            nvvm.tcgen05_st(
                "16x128b",
                nvvm.make_tmem_ptr(
                    row_lo_addr + (tmem_col + cfg.tmem_neg_beta_dy_input_offset), cutlass.Int8
                ),
                neg_beta_dy_pack_lo.load(),
            )
            if cutlass.const_expr(cfg.d_v == 128):
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(
                        row_hi_addr + (tmem_col + cfg.tmem_neg_beta_dy_input_offset), cutlass.Int8
                    ),
                    neg_beta_dy_pack_hi.load(),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_neg_beta_dy_input_ready.arrive()

            # ---- dstate capture for the next chunk -----------------------------------
            bars.mb_dstate_acc_ready.wait(dstate_ready_index.phase)
            dstate_ready_index = advance(dstate_ready_index, 1)
            bars.mb_do_done[raw_index.idx].arrive()
            if rev_idx + cutlass.Int32(1) < num_compute_chunks:
                next_serial = chunk_serial + cutlass.Int32(1)
                next_stage = next_serial % cfg.smem_decay_stages
                bars.mb_decay_scale_ready[next_stage].wait(
                    (next_serial // cfg.smem_decay_stages) % 2
                )
                next_scale_ptr = smem_data_ptr(sDecay_scale_raw) + next_stage * cfg.d_k
                row_lo_addr = tmem_row << 16
                for i in cutlass.range(cfg.d_k // 32, unroll=1):
                    dstate_vec = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            row_lo_addr + (tmem_col + cfg.tmem_dstate_acc_offset + i * 32),
                            cutlass.Float32,
                        ),
                        num=32,
                    )
                    dstate_scaled = cute.make_rmem_tensor((32,), cutlass.Float32)
                    for g in cutlass.range_constexpr(8):
                        scale_frag = (next_scale_ptr + i * 32 + g * 4).load(count=4, alignment=16)
                        for t in cutlass.range_constexpr(2):
                            dstate_scaled[g * 4 + 2 * t], dstate_scaled[g * 4 + 2 * t + 1] = fmul2(
                                dstate_vec[g * 4 + 2 * t],
                                dstate_vec[g * 4 + 2 * t + 1],
                                scale_frag[2 * t],
                                scale_frag[2 * t + 1],
                            )
                    dstate_pack = cute.make_rmem_tensor((16,), cutlass.Int32)
                    for pc in cutlass.range_constexpr(16):
                        dstate_pack[pc] = fp32_to_fp16(
                            dstate_scaled[2 * pc], dstate_scaled[2 * pc + 1], dtype=cfg.io_dtype
                        )
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            row_lo_addr + (tmem_col + cfg.tmem_dstate_input_offset + i * 16),
                            cutlass.Int8,
                        ),
                        dstate_pack.load(),
                    )
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            row_lo_addr + (tmem_col + cfg.tmem_dstate_acc_offset + i * 32),
                            cutlass.Float32,
                        ),
                        dstate_scaled.load(),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_input_ready.arrive()
            raw_index = advance(raw_index, cfg.smem_raw_stages)

        # ---- tile end: dstate0 store / zero-length pass-through ----------------------
        if num_compute_chunks > 0:
            if write_start == 0:
                row_lo_addr = tmem_row << 16
                dstate0_vw = 16 // (mDstate0.element_type.width // 8)
                dstate0_dst = (
                    mDstate0.iterator + mDstate0.layout((batch_idx, head_idx, value_dim, 0))
                ).raw_ptr()
                for i in cutlass.range_constexpr(cfg.d_k // 32):
                    dstate0_vec = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            row_lo_addr + (tmem_col + cfg.tmem_dstate_acc_offset + i * 32),
                            cutlass.Float32,
                        ),
                        num=32,
                    )
                    if state_row_valid:
                        for g in cutlass.range_constexpr(32 // dstate0_vw):
                            (dstate0_dst + i * 32 + g * dstate0_vw).store(
                                cutlass.Vector.from_elements(
                                    tuple(
                                        dstate0_vec[g * dstate0_vw + t].to(mDstate0.element_type)
                                        for t in range(dstate0_vw)
                                    ),
                                    mDstate0.element_type,
                                ),
                                alignment=16,
                            )
        else:
            if state_row_valid:
                for key_dim_base in cutlass.range_constexpr(0, cfg.d_k, 32):
                    for kk_i in cutlass.range_constexpr(32):
                        kd = key_dim_base + kk_i
                        if cutlass.const_expr(cfg.use_dstate_in):
                            mDstate0[batch_idx, head_idx, value_dim, kd] = mDstate_in[
                                batch_idx, head_idx, value_dim, kd
                            ]
                        else:
                            mDstate0[batch_idx, head_idx, value_dim, kd] = cutlass.Float32(0.0).to(
                                mDstate0.element_type
                            )
        if num_compute_chunks > 0:
            bars.mb_dstate0_acc_stored.arrive()
        chunk_serial_base += num_compute_chunks
        tile_idx, scheduler_state = scheduler_next_tile(
            cfg, bars, sScheduler, scheduler_state, elect_one
        )

    bars.mb_tmem_done[0].arrive()


@cute.jit
def build_descs_body(
    widx,
    base_q,
    base_k,
    base_gate,
    base_do,
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    gate: cute.Tensor,
    do: cute.Tensor,
    n_batch: cutlass.Int32,
) -> None:
    """Per-batch descriptor-array build inside the prologue kernel after its order pass, one warp
    per array; warps past the array count fall through the widx guards."""
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

    if widx == 0:
        emit_seq_descs(base_q, sub0, cu_seqlens, q, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 1:
        emit_seq_descs(base_k, sub1, cu_seqlens, k, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 2:
        emit_seq_descs(base_gate, sub2, cu_seqlens, gate, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 3:
        emit_seq_descs(base_do, sub3, cu_seqlens, do, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )


@cute.kernel
def frost_kda_bprop_summary_prologue(
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    base_q: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_k: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_gate: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_do: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    gate: cute.Tensor,
    do: cute.Tensor,
    mStaging: cute.Tensor | None,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor | None,
    mScheduler: cute.Tensor | None,
    n_batch: cutlass.Int32,
) -> None:
    """Two-CTA prologue: under ``run_order`` block 0 LPT-orders the work-item table and zeroes both
    consumers' scheduler rings (:func:`order_body`); block 1 builds the per-batch TMA-descriptor
    arrays (:func:`build_descs_body`), one warp per array."""
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
            base_q,
            base_k,
            base_gate,
            base_do,
            desc_workspace,
            cu_seqlens,
            q,
            k,
            gate,
            do,
            n_batch,
        )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    q: cute.Tensor,
    k: cute.Tensor,
    gate: cute.Tensor,
    do: cute.Tensor,
    cu_seqlens: cute.Tensor,
    work_item_staging: cute.Tensor | None,
    work_count: cute.Tensor,
    work_items: cute.Tensor | None,
    scheduler_all: cute.Tensor | None,
    tensormap_workspace: cute.Tensor,
    stream: cuda_driver.CUstream,
):
    """One-launch prologue: LPT-order the work items (``run_order``) and build the 4 per-(batch,
    head) capped TMA-descriptor arrays into ``tensormap_workspace`` (sequence-relative coordinates;
    tail loads zero-fill in hardware)."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    ho = gate.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    d_k = q.shape[2]
    d_v = do.shape[2]
    bytes_per_element = io_dtype.width // 8
    box_elems = 128 // bytes_per_element
    seqlen = q.shape[0]

    q_headed = cute.make_tensor(
        q.iterator, cute.make_layout((d_k, h_q, seqlen), stride=(1, q.stride[1], q.stride[0]))
    )
    k_headed = cute.make_tensor(
        k.iterator, cute.make_layout((d_k, h_k, seqlen), stride=(1, k.stride[1], k.stride[0]))
    )
    gate_headed = cute.make_tensor(
        gate.iterator,
        cute.make_layout((d_k, ho, seqlen), stride=(1, gate.stride[1], gate.stride[0])),
    )
    do_headed = cute.make_tensor(
        do.iterator, cute.make_layout((d_v, ho, seqlen), stride=(1, do.stride[1], do.stride[0]))
    )

    swizzle = cuda.TensorMapSwizzle.s128b
    base_q = cuda.create_tensor_map_tiled_from_view(
        q_headed, box_dims=(box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    base_k = cuda.create_tensor_map_tiled_from_view(
        k_headed, box_dims=(box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    gate_box_elems = 128 // (gate.element_type.width // 8)
    base_gate = cuda.create_tensor_map_tiled_from_view(
        gate_headed, box_dims=(gate_box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    base_do = cuda.create_tensor_map_tiled_from_view(
        do_headed, box_dims=(box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )

    frost_kda_bprop_summary_prologue(
        run_order,
        order_gen,
        b_t,
        base_q,
        base_k,
        base_gate,
        base_do,
        tensormap_workspace,
        cu_seqlens,
        q,
        k,
        gate,
        do,
        work_item_staging,
        work_count,
        work_items,
        scheduler_all,
        cutlass.Int32(batch_size),
    ).launch(grid=(2, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    q_ratio: cutlass.Int32,
    k_ratio: cutlass.Int32,
    a_log: cute.Tensor | None,
    dt_bias: cute.Tensor | None,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    d_initial_state: cute.Tensor,
    d_final_state: cute.Tensor | None,
    work_items: cute.Tensor | None,
    work_count: cute.Tensor | None,
    scheduler_counter: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    scale: cutlass.Float32,
    stream,
) -> None:
    q_ratio = cute.FastDivmodDivisorV2(q_ratio)
    k_ratio = cute.FastDivmodDivisorV2(k_ratio)
    num_sequences = cu_seqlens.shape[0] - 1

    @cute.struct
    class SharedStorage:
        k_decay: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.operand_cosize], cfg.buffer_align_bytes
        ]
        k_inv: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.operand_cosize], cfg.buffer_align_bytes
        ]
        q_decay: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.operand_cosize], cfg.buffer_align_bytes
        ]
        decay_scale: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, cfg.decay_scale_cosize], 16
        ]
        intermediate: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.intermediate_cosize], cfg.buffer_align_bytes
        ]
        do: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.raw_v_cosize], cfg.buffer_align_bytes
        ]
        q: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.raw_qk_cosize], cfg.buffer_align_bytes
        ]
        k: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.raw_qk_cosize], cfg.buffer_align_bytes
        ]
        gate: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, cfg.raw_gate_cosize], 1024]

    # ---- launch ----------------------------------------------------------------------
    n_desc = num_sequences
    grid_shape = (cfg.max_active_clusters, 1, 1)
    frost_kda_bprop_summary(
        cfg,
        SharedStorage,
        q_ratio,
        k_ratio,
        tensormap_workspace,
        n_desc,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        d_initial_state,
        d_final_state,
        work_items,
        work_count,
        scheduler_counter,
        scale,
    ).launch(
        grid=grid_shape,
        block=(cfg.threads_per_cta, 1, 1),
        stream=stream,
        use_pdl=USE_PDL,
        min_blocks_per_mp=1,
    )


@cute.kernel
def frost_kda_bprop_summary(
    cfg: cutlass.Constexpr,
    shared_type: cutlass.Constexpr,
    q_ratio: cute.FastDivmodDivisorV2,
    k_ratio: cute.FastDivmodDivisorV2,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
    mA_log: cute.Tensor | None,
    mDt_bias: cute.Tensor | None,
    mBeta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    mDstate0: cute.Tensor,
    mDstate_in: cute.Tensor | None,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor,
    scale: cutlass.Float32,
) -> None:
    """BT=16 KDA reverse state-gradient summary kernel (persistent, 16 warps;
    warps 8-11 exit after init)."""
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
    desc_q_base = desc_base_words
    desc_k_base = desc_base_words + arr_words
    desc_gate_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_do_base = desc_base_words + cutlass.Int32(3) * arr_words

    SMEM = cutlass.AddressSpace.smem
    bars = make_bars(cfg)
    tmem_base_holder = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=4)
    sScheduler = cutlass.Array(cutlass.Int32, cfg.scheduler_stages, space=SMEM, alignment=16)
    bytes_per_element = cfg.io_dtype.width // 8
    SWZ = 2
    LEAD = 16
    STRIDE = 8 * 128

    sBeta_raw = cutlass.Array(
        cutlass.Float32,
        cfg.smem_beta_stages * cfg.b_t,
        space=SMEM,
        alignment=64,
    )
    storage = SmemAllocator().allocate(shared_type)
    sK_decay_raw = storage.k_decay.get_tensor(cute.make_layout((cfg.operand_cosize,)))
    sK_inv_raw = storage.k_inv.get_tensor(cute.make_layout((cfg.operand_cosize,)))
    sQ_decay_raw = storage.q_decay.get_tensor(cute.make_layout((cfg.operand_cosize,)))
    sDecay_scale_raw = storage.decay_scale.get_tensor(cute.make_layout((cfg.decay_scale_cosize,)))
    sIntermediate_raw = storage.intermediate.get_tensor(
        cute.make_layout((cfg.intermediate_cosize,))
    )
    sDo_raw = storage.do.get_tensor(cute.make_layout((cfg.raw_v_cosize,)))
    sQ_raw = storage.q.get_tensor(cute.make_layout((cfg.raw_qk_cosize,)))
    sK_raw = storage.k.get_tensor(cute.make_layout((cfg.raw_qk_cosize,)))
    sGate_raw = storage.gate.get_tensor(cute.make_layout((cfg.raw_gate_cosize,)))
    if cutlass.const_expr(cfg.gate_dtype == cutlass.Float32):
        sGate_load_ptr = smem_data_ptr(sGate_raw)
    else:
        sGate_load_ptr = cute.make_ptr(
            cfg.gate_dtype, smem_data_ptr(sGate_raw).toint(), mem_space=SMEM, assumed_align=1024
        )

    sK_decay = SmemTile(
        base=smem_data_ptr(sK_decay_raw).toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bytes_per_element,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_inv = SmemTile(
        base=smem_data_ptr(sK_inv_raw).toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bytes_per_element,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDo = SmemTile(
        base=smem_data_ptr(sDo_raw).toint(),
        elems_per_stage=((cfg.raw_v_cosize) // (cfg.smem_raw_stages)) * bytes_per_element,
        stages=cfg.smem_raw_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDo_trans = SmemTile(
        base=smem_data_ptr(sDo_raw).toint(),
        elems_per_stage=((cfg.raw_v_cosize) // (cfg.smem_raw_stages)) * bytes_per_element,
        stages=cfg.smem_raw_stages,
        leading_byte_offset=cfg.b_t * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )

    sQ_decay_trans = SmemTile(
        base=smem_data_ptr(sQ_decay_raw).toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bytes_per_element,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=cfg.b_t * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_inv_trans = SmemTile(
        base=smem_data_ptr(sK_inv_raw).toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bytes_per_element,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=cfg.b_t * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_decay_trans = SmemTile(
        base=smem_data_ptr(sK_decay_raw).toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bytes_per_element,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=cfg.b_t * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sIntermediate = SmemTile(
        base=sIntermediate_raw,
        elems_per_stage=(cfg.intermediate_tiles * cfg.b_t * cfg.b_t),
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
                bars.mb_raw_done[stage].init()
                bars.mb_do_ready[stage].init()
                bars.mb_do_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_beta_stages):
                bars.mb_beta_ready[stage].init()
                bars.mb_beta_done[stage].init()
    elif warp_idx == cfg.tcgen05_mma_warp_id:
        if elect_one:
            bars.mb_du_acc_ready.init()
            bars.mb_du_input_ready.init()
            bars.mb_dy_acc_ready.init()
            bars.mb_neg_beta_dy_input_ready.init()
            bars.mb_dstate_acc_ready.init()
            bars.mb_dstate_input_ready.init()
            bars.mb_dstate0_acc_stored.init()
            bars.mb_tmem_done[0].init()
    elif warp_idx == cfg.register_mma_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_k_decay_inv_ready[stage].init()
                bars.mb_q_decay_ready[stage].init()
                bars.mb_decay_done[stage].init()
                bars.mb_decay_scale_ready[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_intermediate_stages):
                bars.mb_t_inv_ready[stage].init()
                bars.mb_a_ready[stage].init()
                bars.mb_a_done[stage].init()
                bars.mb_t_inv_done[stage].init()
    elif warp_idx == cfg.epilogue_warp_id:
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
            sQ_raw,
            sK_raw,
            sGate_raw,
            sDo_raw,
            desc_q_base,
            desc_k_base,
            desc_gate_base,
            desc_do_base,
            bars,
            q_ratio=q_ratio,
            k_ratio=k_ratio,
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
            sK_decay_raw,
            sK_inv_raw,
            sIntermediate_raw,
            sBeta_raw,
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
            tmem_base_holder,
            sK_decay,
            sK_inv,
            sK_inv_trans,
            sDo,
            sDo_trans,
            sQ_decay_trans,
            sK_decay_trans,
            sIntermediate,
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
            sK_inv_raw,
            sQ_decay_raw,
            sIntermediate_raw,
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
            scale,
            mA_log,
            mDt_bias,
            mBeta,
            sBeta_raw,
            sK_inv_raw,
            sGate_raw,
            sGate_load_ptr,
            sK_raw,
            sQ_raw,
            sK_decay_raw,
            sQ_decay_raw,
            sDecay_scale_raw,
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
            tmem_base_holder,
            warp_idx,
            mDstate0,
            mDstate_in,
            sBeta_raw,
            sDecay_scale_raw,
            bars,
        )


@dataclass(frozen=True)
class KdaBpropSummaryCfg:
    """Kernel cfg (fixed BT=16 schedule constants; derived TMEM column offsets
    and SMEM buffer cosizes are stamped by ``build_cfg``)."""

    io_dtype: type[cutlass.Numeric]
    gate_dtype: type[cutlass.Numeric]
    use_dstate_in: bool
    l2norm: bool
    safe_gate: bool
    gate_scale_log2: float
    log_gate: bool
    beta_sigmoid: bool
    allow_neg_eigval: bool
    max_active_clusters: int
    d_k: int
    d_v: int
    scheduler_stages: int = 8

    # ---- fixed constants stamped from CFG at build time ------------------------------
    compute_group_0_warp_ids: tuple = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_1_warp_ids: tuple = CFG.COMPUTE_GROUP_1_WARP_IDS
    register_mma_warp_id: int = CFG.REGISTER_MMA_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    tma_warp_id: int = CFG.TMA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    b_t: int = CFG.B_T
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_other: int = CFG.NUM_REGS_OTHER

    # ---- named barrier slots (0 is the CTA-wide sync) --------------------------------
    cg0_sync_barrier_id: int = 1
    cg0_threads: int = 0
    tmem_lifecycle_barrier_id: int = 3
    scheduler_consumer_warps: int = 0
    tmem_user_threads: int = 0

    # ---- SMEM / TMEM stage counts + TMEM column offsets ------------------------------
    smem_raw_stages: int = CFG.SMEM_RAW_STAGES
    smem_decay_stages: int = CFG.SMEM_DECAY_STAGES
    smem_intermediate_stages: int = CFG.SMEM_INTERMEDIATE_STAGES
    smem_beta_stages: int = 4
    intermediate_tiles: int = 2
    tmem_dstate_acc_offset: int = 0
    tmem_dstate_input_offset: int = 0
    tmem_du_acc_offset: int = 0
    tmem_dy_acc_offset: int = 0
    tmem_neg_beta_dy_input_offset: int = 0
    tmem_du_input_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- buffer cosizes / TMA bytes stamped at build time ----------------------------
    raw_qk_cosize: int = 0
    raw_v_cosize: int = 0
    raw_gate_cosize: int = 0
    gate_stage_elems: int = 0
    operand_cosize: int = 0
    decay_scale_cosize: int = 0
    intermediate_cosize: int = 0

    # ---- TMA transaction bytes per stage ---------------------------------------------
    tma_q_bytes: int = 0
    tma_k_bytes: int = 0
    tma_gate_bytes: int = 0
    tma_do_bytes: int = 0


def build_cfg(
    io_dtype: type[cutlass.Numeric],
    gate_dtype: type[cutlass.Numeric],
    *,
    use_dstate_in: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_scale_log2: float,
    log_gate: bool = True,
    beta_sigmoid: bool,
    allow_neg_eigval: bool,
    max_active_clusters: int,
    d_k: int,
    d_v: int,
) -> KdaBpropSummaryCfg:
    cfg = KdaBpropSummaryCfg(
        io_dtype=io_dtype,
        gate_dtype=gate_dtype,
        use_dstate_in=use_dstate_in,
        l2norm=l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        log_gate=log_gate,
        beta_sigmoid=beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        max_active_clusters=max_active_clusters,
        d_k=d_k,
        d_v=d_v,
    )
    # The bprop role map minus compute group 2: warps 8-11 launch (upstream 16-warp CTA) and exit
    # after the role dispatch, so twelve warps consume the scheduler ring.
    role_ids = (
        *cfg.compute_group_0_warp_ids,
        *cfg.compute_group_1_warp_ids,
        cfg.register_mma_warp_id,
        cfg.tcgen05_mma_warp_id,
        cfg.tma_warp_id,
        cfg.epilogue_warp_id,
    )
    scheduler_consumer_warps = _validate_roles(
        role_ids, (cfg.cg0_sync_barrier_id, cfg.tmem_lifecycle_barrier_id), 16
    )
    threads_per_cta = 16 * cfg.threads_per_warp
    cg0_threads = len(cfg.compute_group_0_warp_ids) * cfg.threads_per_warp
    tmem_user_threads = (1 + len(cfg.compute_group_1_warp_ids)) * cfg.threads_per_warp

    tmem_dstate_input_offset = cfg.d_k
    tmem_du_acc_offset = tmem_dstate_input_offset + cfg.d_k // 2
    tmem_dy_acc_offset = tmem_du_acc_offset + cfg.b_t
    tmem_neg_beta_dy_input_offset = tmem_dy_acc_offset + cfg.b_t
    tmem_du_input_offset = tmem_neg_beta_dy_input_offset + cfg.b_t // 2
    assert tmem_du_input_offset + cfg.b_t // 2 <= 512

    return replace(
        cfg,
        threads_per_cta=threads_per_cta,
        scheduler_consumer_warps=scheduler_consumer_warps,
        cg0_threads=cg0_threads,
        tmem_user_threads=tmem_user_threads,
        tmem_dstate_acc_offset=0,
        tmem_dstate_input_offset=tmem_dstate_input_offset,
        tmem_du_acc_offset=tmem_du_acc_offset,
        tmem_dy_acc_offset=tmem_dy_acc_offset,
        tmem_neg_beta_dy_input_offset=tmem_neg_beta_dy_input_offset,
        tmem_du_input_offset=tmem_du_input_offset,
        raw_qk_cosize=cfg.smem_raw_stages * cfg.d_k * cfg.b_t,
        raw_v_cosize=cfg.smem_raw_stages * cfg.d_v * cfg.b_t,
        raw_gate_cosize=cfg.smem_raw_stages * cfg.d_k * cfg.b_t,
        gate_stage_elems=(cfg.d_k * cfg.b_t) * (4 // (cfg.gate_dtype.width // 8)),
        operand_cosize=cfg.smem_decay_stages * cfg.b_t * cfg.d_k,
        decay_scale_cosize=cfg.smem_decay_stages * cfg.d_k,
        intermediate_cosize=cfg.smem_intermediate_stages
        * cfg.intermediate_tiles
        * cfg.b_t
        * cfg.b_t,
        tma_q_bytes=cfg.d_k * cfg.b_t * (cfg.io_dtype.width // 8),
        tma_k_bytes=cfg.d_k * cfg.b_t * (cfg.io_dtype.width // 8),
        tma_gate_bytes=cfg.d_k * cfg.b_t * (cfg.gate_dtype.width // 8),
        tma_do_bytes=cfg.d_v * cfg.b_t * (cfg.io_dtype.width // 8),
    )


TENSORMAP_DESC_ARRAYS = 4  # per-batch runtime TMA descriptors: Q, K, Gate, dO


# ---------------------------------------------------------------------------


def _dtype_name(dtype) -> str:
    return "none" if dtype is None else dtype.__name__.lower()


class KdaBpropSummaryOp:
    """Standalone KDA bprop-summary launch over ``host`` for one static config."""

    def __init__(self, cfg: KdaBpropSummaryCfg, use_int64_offsets: bool = False, dtypes: str = ""):
        self.cfg = cfg
        self.use_int64_offsets = use_int64_offsets
        self.dtypes = dtypes

    def get_name(self) -> str:
        cfg = self.cfg
        flags = "".join(
            str(int(flag))
            for flag in (
                cfg.use_dstate_in,
                cfg.l2norm,
                cfg.safe_gate,
                cfg.log_gate,
                cfg.beta_sigmoid,
                cfg.allow_neg_eigval,
            )
        )
        return (
            f"kda_cudnn_bprop_summary_{cfg.io_dtype.__name__.lower()}"
            f"_{cfg.gate_dtype.__name__.lower()}"
            f"_f{flags}_k{cfg.d_k}_v{cfg.d_v}_{self.dtypes}_sm{cfg.max_active_clusters}"
            f"_i64{int(self.use_int64_offsets)}"
        )

    @cute.jit
    def __call__(
        self,
        q_ratio: cutlass.Int32,
        k_ratio: cutlass.Int32,
        a_log: cute.Tensor | None,
        dt_bias: cute.Tensor | None,
        beta: cute.Tensor,
        cu_seqlens: cute.Tensor,
        d_initial_state: cute.Tensor,
        d_final_state: cute.Tensor | None,
        work_items: cute.Tensor | None,
        work_count: cute.Tensor | None,
        scheduler_counter: cute.Tensor,
        tensormap_workspace: cute.Tensor,
        scale: cutlass.Float32,
        stream,
    ) -> None:
        host(
            self.cfg,
            q_ratio,
            k_ratio,
            a_log,
            dt_bias,
            beta,
            cu_seqlens,
            d_initial_state,
            d_final_state,
            work_items,
            work_count,
            scheduler_counter,
            tensormap_workspace,
            scale,
            stream,
        )


@jit_cache
def _compile_kda_bprop_summary(
    io_dtype,
    gate_dtype,
    a_log_dtype,
    dt_bias_spec,
    cu_seqlens_dtype,
    beta_dtype,
    dstate0_dtype,
    dstate_in_dtype,
    use_dstate_in,
    l2norm,
    safe_gate,
    gate_scale_log2,
    log_gate,
    beta_sigmoid,
    allow_neg_eigval,
    q_ratio,
    k_ratio,
    d_k,
    d_v,
    num_sm,
    use_int64_offsets,
):
    cfg = build_cfg(
        io_dtype,
        gate_dtype,
        use_dstate_in=use_dstate_in,
        l2norm=l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        log_gate=log_gate,
        beta_sigmoid=beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        max_active_clusters=num_sm,
        d_k=d_k,
        d_v=d_v,
    )
    dyn = lambda dtype, rank, align: make_dynamic_signature_tensor(
        dtype, rank, assumed_align=align, use_int64_offsets=use_int64_offsets
    )
    dt_bias_dtype = None if dt_bias_spec is None else dt_bias_spec[0]
    dt_bias_rank = 0 if dt_bias_spec is None else dt_bias_spec[1]
    dtypes = "_".join(
        (
            _dtype_name(a_log_dtype),
            _dtype_name(dt_bias_dtype),
            _dtype_name(cu_seqlens_dtype),
            _dtype_name(beta_dtype),
            _dtype_name(dstate0_dtype),
            _dtype_name(dstate_in_dtype),
        )
    )
    return compile_tvm_ffi(
        KdaBpropSummaryOp(cfg, use_int64_offsets, dtypes),
        cutlass.Int32(0),
        cutlass.Int32(0),
        None if a_log_dtype is None else dyn(a_log_dtype, 1, 4),
        None if dt_bias_dtype is None else dyn(dt_bias_dtype, dt_bias_rank, 16),
        dyn(beta_dtype, 2, 4),
        dyn(cu_seqlens_dtype, 1, 8 if cu_seqlens_dtype == cutlass.Int64 else 4),
        dyn(dstate0_dtype, 4, 16),
        None if dstate_in_dtype is None else dyn(dstate_in_dtype, 4, 16),
        make_compact_signature_tensor(
            cutlass.Int32, (cute.sym_int(), WORK_ITEM_FIELDS), assumed_align=16
        ),
        dyn(cutlass.Int32, 1, 4),
        dyn(cutlass.Int32, 1, 4),
        dyn(cutlass.Int64, 1, 128),
        cutlass.Float32(0),
        opt_level=2,
    )


@jit_cache
def _compile_kda_bprop_summary_prologue(
    io_dtype,
    gate_dtype,
    cu_seqlens_dtype,
    run_order,
    order_gen,
    has_sched,
    use_int64_offsets,
):
    dyn = lambda dtype, rank, align: make_dynamic_signature_tensor(
        dtype, rank, assumed_align=align, use_int64_offsets=use_int64_offsets
    )
    items = lambda: make_compact_signature_tensor(
        cutlass.Int32, (cute.sym_int(), WORK_ITEM_FIELDS), assumed_align=16
    )
    io = _dtype_name(io_dtype)
    gate = _dtype_name(gate_dtype)
    cu = _dtype_name(cu_seqlens_dtype)
    return compile_tvm_ffi(
        prologue,
        io_dtype,
        CFG.B_T,
        run_order,
        order_gen,
        dyn(io_dtype, 3, 16),
        dyn(io_dtype, 3, 16),
        dyn(gate_dtype, 3, 16),
        dyn(io_dtype, 3, 16),
        dyn(cu_seqlens_dtype, 1, 8 if cu_seqlens_dtype == cutlass.Int64 else 4),
        items() if run_order and not order_gen else None,
        dyn(cutlass.Int32, 1, 4),
        items(),
        dyn(cutlass.Int32, 1, 4) if has_sched else None,
        dyn(cutlass.Int64, 1, 128),
        name=(
            f"kda_cudnn_bprop_summary_prologue_{io}_{gate}_{cu}_r{int(run_order)}"
            f"o{int(order_gen)}s{int(has_sched)}_i64{int(use_int64_offsets)}"
        ),
        opt_level=2,
    )


frost_kda_bprop_summary_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_kda_bprop_summary.set_name_prefix("cudnn", remove_cutlass_symbol=False)
