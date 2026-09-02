# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# This kernel is derived from cuDNN, NVIDIA Corporation.
# Modified by Attention Gym in 2026: imports were relocated into this package.
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

"""
Chunked Gated Delta Net (GDN) prefill kernel for Blackwell SM100 (Cutlass primitives)
with optional per-chunk state-checkpoint output.

Algorithm overview (per chunk c, tokens [cC, (c+1)C)):
  Inputs : Q[BT,DK], K[BT,DK], V[BT,DV], Gate[BT] (scalar gate), Beta[BT] (scalar LR)
  State  : S_prev[DK,DV]  (recurrent state, held in TMEM)

  Preprocessing (compute warp group 0):
    cumsumlog[t]     = sum_{l=0}^{t} log(gate_l)              cumulative log of gates
    cumprod[t]       = exp(cumsumlog[t])                       cumulative product of gates
    T_pairwise[i,j]  = cumprod[i] / cumprod[j]  (i>=j)       inter-token transfer weights
    (stored in registers; 128 regs/thread)

  GEMM 1 - KK   : W_kk[BT,BT]  = K  @ K^T       (lower-triangular intra scores)
  GEMM 2 - QK   : W_qk[BT,BT]  = Q  @ K^T       (output attention scores)
  GEMM 3 - K*state : KS[BT,DV] = K  @ S_prev    (key applied to state)
  GEMM 4 - Q*state : QS[BT,DV] = Q  @ S_prev    (inter-chunk output, before T scaling)
  GEMM 5 - U       : U[BT,DV]  = T_inv @ Y       (corrected value vectors)
                      where T_inv = (I + M_kk)^{-1},  M_kk[i,j] = T[i,j]*Beta[i]*W_kk[i,j]  (lower-tri, hierarchical blockwise inverse)
  GEMM 6 - QKV  : O_intra[BT,DV] = W_qkv @ U    (intra-chunk output)
                   where W_qkv = T*Beta*W_qk (the A tile)
  GEMM 7 - KV update : S_upd[DK,DV] = K^T @ (decay .* U)  (state update, BT contraction)
                        where Y[BT,DV] = V - KS    (delta rule residuals, after decay)

  Epilogue:
    O[BT,DV]  = O_intra + T_col * QS             (combine intra + inter)
    S_next    = cumprod[BT-1] * S_prev + S_upd        (update state in TMEM)

Chunks run in PAIRS (CG0 warp halves invert chunk 0 / chunk 1 in parallel);
odd counts pad with a neutral zero-filled chunk.

SMEM layout (227 KB = full; stage counts live in gdn_prefill_config.py;
enable_checkpoints compiles trim K/V stages to fit the checkpoint buffer):
  Buffer                       Size (B)  Stages
  Q                               16384       3
  K                               16384       4
  V                               16384       3
  T_inv                            8192       3
  A tile output                    8192       3
  O store                         16384       2
  checkpoint staging            DK*DV*2       1    <-- enable_checkpoints only
  cumsumlog / cumprod / Beta        256       3
  sched ticket ring                   4       2    <-- dyn_sched publish ring

TMEM layout (512 columns):
  Buffer                  Cols
  state                   128     <-- DKxDV fp32 = 128x128x4B
  Q*state / O acc          64     <-- BTxDV fp32 accumulator
  state inp                64     <-- fp16 state staging (GEMMs 3/4 A operand)
  cg0 shared acc          128     <-- 2-stage ring: KK0/KK1 then QK0/QK1
  cg1 shared acc           64     <-- 1-stage ring: KS then U
  Y + U input / decayed-U inp    64  <-- slot 0 = Y then U input, slot 1 = decayed U (b16)

Warp assignments (12 warps = 384 threads):
  warps 0-3     : compute group 0 - T-pairwise x2, KK_epi x2, pair inverse,
                                    A_epi x2
  warps 4-7     : compute group 1 - state restage/rescale, Y = V - K*state,
                                    state*Q_epi, U_epi, QKV_epilogue
  warp  8       : Gate/Beta loads
  warp  9       : TMA load warp  - loads Q, K, V
  warp  10      : MMA warp       - fused KK/QK pairs + K*state/Q*state/U/QKV/KV per chunk;
                                    TMEM lifecycle
  warp  11      : epilogue warp  - O then checkpoint TMA stores
"""

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import NamedTuple, Optional, Tuple, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as nvvm
from cutlass.cutlass_dsl import min

from attn_gym._backends.cute import compile_tvm_ffi
from attn_gym._backends.cute.utils import requires_int64_abi

from .common.elementwise import softplus
from .common.host import get_dtype
from .common.paged_state import resolve_paged_state
from .common.split_k import (
    ORDER_CAPACITY,
    ORDER_ELEMS,
    ORDER_THREADS,
    decode_work_item,
    order_body,
)
from .common.thd import (
    TENSOR_MAP_QWORDS,
    emit_checkpoint_seq_descs,
    emit_seq_descs,
    emit_seq_load_descs,
)
from .common.tvm_ffi import (
    make_compact_signature_tensor,
    make_counter_signature,
    make_cu_seqlens_signature,
    make_paged_route_signatures,
    make_strided_signature_tensor,
    make_work_items_signature,
    make_workspace_signature,
    validate_cu_seqlens,
)
from .compat import (
    checkpoint_capacity_bound,
    current_device,
    get_device_properties,
    tensor_device_index,
    validate_tma_tensor,
)
from .gdn_prefill_config import CFG
from .tile_dsl.barrier import MBarrier, PipelineState, Producer, advance
from .tile_dsl.handles import MmaDesc, SmemTile, smem_data_ptr, tma_slice_runtime_desc
from .tile_dsl.mma import mma_ss, mma_step, mma_step_k8, mma_ts_step
from .tile_dsl.pointwise import (
    f16x2_to_f32,
    fadd2,
    fmul2,
    fp32_to_fp16,
    opaque_f32_zero,
    sub_f16x2,
)
from .tile_dsl.swizzle import swizzle_lin_128b, swizzle_xor_128b
from .tile_dsl.tma import (
    tma_load_tile,
    tma_store_commit,
    tma_store_tile,
    tma_store_wait,
    tma_tensormap_acquire,
)

RCP_LN2: float = 1.4426950408889634  # natural-log gates -> kernel's log2 domain


class GdnBars(NamedTuple):
    """GDN pipeline mbarrier inventory.

    Every pipeline is a ``_ready``/``_done`` MBarrier pair over one ring: a
    slot is acquired for filling by waiting ``_done`` and committed by
    arriving ``_ready``; the reading side waits ``_ready`` and releases the
    slot by arriving ``_done``.
    """

    mb_kq_ready: MBarrier
    mb_kq_done: MBarrier
    mb_v_ready: MBarrier
    mb_v_done: MBarrier

    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier
    mb_beta_ready: MBarrier
    mb_beta_done: MBarrier

    mb_state_acc_ready: MBarrier
    mb_state_acc_scale_done: MBarrier
    mb_o_acc_ready: MBarrier
    mb_o_final_acc_ready: MBarrier
    mb_o_state_scale_acc_done: MBarrier
    mb_cg0_acc_ready: MBarrier
    mb_cg0_acc_done: MBarrier

    mb_state_inp_ready: MBarrier
    mb_y_inp_ready: MBarrier
    mb_u_inp_ready: MBarrier
    mb_decay_u_inp_ready: MBarrier

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

    mb_sched_ready: MBarrier
    mb_sched_done: MBarrier


def make_gdn_bars(cfg) -> GdnBars:
    """GdnBars factory.  MUST be called from inside ``kernel`` (allocates SMEM)."""
    ONE_LANE = 1
    MMA_ARRIVERS = len([cfg.mma_warp_id])
    KQ_RELEASE_SITES = 1
    GATE_WARP = cfg.threads_per_warp * len([cfg.load_gate_beta_warp_id])
    EPI_WARP = cfg.threads_per_warp * len([cfg.epilogue_warp_id])
    CG0_THREADS = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    CG1_THREADS = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    CG0_PLUS_CG1 = CG0_THREADS + CG1_THREADS

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=16)

    return GdnBars(
        mb_kq_ready=MBarrier(
            alloc(cfg.smem_kq_stages),
            stages=cfg.smem_kq_stages,
            init_count=ONE_LANE,
            producer=Producer.TMA_LOAD,
        ),
        mb_kq_done=MBarrier(
            alloc(cfg.smem_kq_stages),
            stages=cfg.smem_kq_stages,
            init_count=KQ_RELEASE_SITES,
            producer=Producer.MMA_COMMIT,
        ),
        mb_v_ready=MBarrier(
            alloc(cfg.smem_v_stages),
            stages=cfg.smem_v_stages,
            init_count=ONE_LANE,
            producer=Producer.TMA_LOAD,
        ),
        mb_v_done=MBarrier(
            alloc(cfg.smem_v_stages),
            stages=cfg.smem_v_stages,
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_gate_ready=MBarrier(
            alloc(cfg.smem_gate_stages),
            stages=cfg.smem_gate_stages,
            init_count=GATE_WARP,
            producer=Producer.THREAD,
        ),
        mb_gate_done=MBarrier(
            alloc(cfg.smem_gate_stages),
            stages=cfg.smem_gate_stages,
            init_count=CG0_PLUS_CG1,
            producer=Producer.THREAD,
        ),
        mb_beta_ready=MBarrier(
            alloc(cfg.smem_beta_stages),
            stages=cfg.smem_beta_stages,
            init_count=GATE_WARP,
            producer=Producer.THREAD,
        ),
        mb_beta_done=MBarrier(
            alloc(cfg.smem_beta_stages),
            stages=cfg.smem_beta_stages,
            init_count=CG0_THREADS,
            producer=Producer.THREAD,
        ),
        mb_state_acc_ready=MBarrier(
            alloc(cfg.tmem_state_acc_stages),
            stages=cfg.tmem_state_acc_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_state_acc_scale_done=MBarrier(
            alloc(cfg.tmem_state_acc_stages),
            stages=cfg.tmem_state_acc_stages,
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_o_acc_ready=MBarrier(
            alloc(cfg.tmem_q_state_acc_stages),
            stages=cfg.tmem_q_state_acc_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_o_final_acc_ready=MBarrier(
            alloc(cfg.tmem_q_state_acc_stages),
            stages=cfg.tmem_q_state_acc_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_o_state_scale_acc_done=MBarrier(
            alloc(cfg.tmem_q_state_acc_stages),
            stages=cfg.tmem_q_state_acc_stages,
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_cg0_acc_ready=MBarrier(
            alloc(cfg.tmem_cg0_acc_stages),
            stages=cfg.tmem_cg0_acc_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_cg0_acc_done=MBarrier(
            alloc(cfg.tmem_cg0_acc_stages),
            stages=cfg.tmem_cg0_acc_stages,
            init_count=CG0_THREADS // 2,
            producer=Producer.THREAD,
        ),
        mb_state_inp_ready=MBarrier(
            alloc(cfg.tmem_state_inp_stages),
            stages=cfg.tmem_state_inp_stages,
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_y_inp_ready=MBarrier(
            alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_u_inp_ready=MBarrier(
            alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_decay_u_inp_ready=MBarrier(
            alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_t_inv_ready=MBarrier(
            alloc(cfg.smem_t_inv_stages),
            stages=cfg.smem_t_inv_stages,
            init_count=CG0_THREADS,
            producer=Producer.THREAD,
        ),
        mb_t_inv_done=MBarrier(
            alloc(cfg.smem_t_inv_stages),
            stages=cfg.smem_t_inv_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_a_ready=MBarrier(
            alloc(cfg.smem_a_stages),
            stages=cfg.smem_a_stages,
            init_count=CG0_THREADS // 2,
            producer=Producer.THREAD,
        ),
        mb_a_done=MBarrier(
            alloc(cfg.smem_a_stages),
            stages=cfg.smem_a_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_k_state_acc_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_u_acc_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_o_tmastg_ready=MBarrier(
            alloc(cfg.smem_o_stages),
            stages=cfg.smem_o_stages,
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_o_tmastg_done=MBarrier(
            alloc(cfg.smem_o_stages),
            stages=cfg.smem_o_stages,
            init_count=EPI_WARP,
            producer=Producer.THREAD,
        ),
        mb_checkpoint_tmastg_ready=MBarrier(
            alloc(cfg.smem_checkpoint_stages),
            stages=cfg.smem_checkpoint_stages,
            init_count=len(cfg.compute_group_1_warp_ids),
            producer=Producer.THREAD,
        ),
        mb_checkpoint_tmastg_done=MBarrier(
            alloc(cfg.smem_checkpoint_stages),
            stages=cfg.smem_checkpoint_stages,
            init_count=EPI_WARP,
            producer=Producer.THREAD,
        ),
        mb_tmem_done=MBarrier(
            alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_sched_ready=MBarrier(
            alloc(cfg.sched_stages),
            stages=cfg.sched_stages,
            init_count=1,
            producer=Producer.THREAD,
        ),
        mb_sched_done=MBarrier(
            alloc(cfg.sched_stages),
            stages=cfg.sched_stages,
            init_count=11,
            producer=Producer.THREAD,
        ),
    )


@cute.jit
def invert_diagonal_NxN(cfg, base_int, d, tidx, N: int = 8):
    """Gauss-Jordan inversion of one diagonal NxN block in-place (f16 SMEM)."""
    tidx_in_group = tidx % N
    BT = cfg.b_t

    row_lin_base = (d * N + tidx_in_group) * BT + d * N
    row_phys = swizzle_lin_128b(row_lin_base, row_stride_log2=6)
    row_ptr = (
        cute.make_ptr(
            cfg.io_dtype,
            base_int,
            mem_space=cute.AddressSpace.smem,
            assumed_align=cfg.buffer_align_bytes,
        )
        + row_phys
    )

    row = [(row_ptr + j).load().to(cutlass.Float32) for j in range(N)]
    for i in cutlass.range_constexpr(N):
        row[i] = cutlass.Float32(1.0) if tidx_in_group == i else row[i]
    for src_row in cutlass.range_constexpr(N - 1):
        row_scale = -row[src_row]
        for i in cutlass.range_constexpr(src_row):
            shfl_val = nvvm.shfl_sync(
                0xFFFFFFFF, row[i], src_row, 0b1100000011111, kind=nvvm.Shfl.IDX
            )
            row[i] = row[i] + row_scale * shfl_val if tidx_in_group > src_row else row[i]
        row[src_row] = row_scale if tidx_in_group > src_row else row[src_row]

    for j in cutlass.range_constexpr(N):
        (row_ptr + j).store(row[j].to(cfg.io_dtype))


@cute.jit
def blockwise_diagonal_8x8_to_16x16(cfg, base_int, d0, lane_id):
    """Off-diagonal correction 8x8 -> 16x16 (C <- -D^{-1} C A^{-1})."""
    bpe = cfg.io_dtype.width // 8
    ldsm_x1_lane_off = (lane_id % 8) * 64
    d = nvvm.ldmatrix(
        cutlass.inttoptr(
            base_int
            + swizzle_lin_128b((d0 + 8) * 64 + d0 + 8 + ldsm_x1_lane_off, row_stride_log2=6) * bpe,
            cutlass.AddressSpace.smem,
            cutlass.BFloat16,
        ),
        1,
        nvvm.MMALayout.ROW,
    )
    c = nvvm.ldmatrix(
        cutlass.inttoptr(
            base_int
            + swizzle_lin_128b((d0 + 8) * 64 + d0 + ldsm_x1_lane_off, row_stride_log2=6) * bpe,
            cutlass.AddressSpace.smem,
            cutlass.BFloat16,
        ),
        1,
        nvvm.MMALayout.COL,
    )

    # ---- T = -(D^{-1} @ C) -------------------------------------------------------
    c_regs = cutlass.Array(cutlass.Float32, 4, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(4):
        c_regs[i] = cutlass.Float32(0.0)
    mma_step_k8(c_regs, [d, d], [c], k_step=0, M=16, N=8, ab_dtype=cfg.io_dtype)
    for i in cutlass.range_constexpr(4):
        c_regs[i] = -c_regs[i]
    a_pack = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(2)]

    # ---- C = T @ A^{-1} ----------------------------------------------------------
    ai_frag = nvvm.ldmatrix(
        cutlass.inttoptr(
            base_int + swizzle_lin_128b(d0 * 64 + d0 + ldsm_x1_lane_off, row_stride_log2=6) * bpe,
            cutlass.AddressSpace.smem,
            cutlass.BFloat16,
        ),
        1,
        nvvm.MMALayout.COL,
    )
    o_regs = cutlass.Array(cutlass.Float32, 4, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(4):
        o_regs[i] = cutlass.Float32(0.0)
    mma_step_k8(o_regs, a_pack, [ai_frag], k_step=0, M=16, N=8, ab_dtype=cfg.io_dtype)
    o_pack = fp32_to_fp16(o_regs[0], o_regs[1], dtype=cfg.io_dtype)

    # ---- store corrected C -------------------------------------------------------
    nvvm.stmatrix(
        cutlass.inttoptr(
            base_int
            + swizzle_lin_128b((d0 + 8) * 64 + d0 + ldsm_x1_lane_off, row_stride_log2=6) * bpe,
            cutlass.AddressSpace.smem,
            cutlass.BFloat16,
        ),
        o_pack,
        nvvm.MMALayout.ROW,
    )


@cute.jit
def blockwise_diagonal_16x16_to_32x32(cfg, base_int, d0, lane_id):
    """Off-diagonal correction 16x16 -> 32x32."""
    bpe = cfg.io_dtype.width // 8
    ldsm_x4_lane_off = (lane_id % 16) * 64 + (lane_id // 16) * 8
    d = list(
        nvvm.ldmatrix(
            cutlass.inttoptr(
                base_int
                + swizzle_lin_128b((d0 + 16) * 64 + d0 + 16 + ldsm_x4_lane_off, row_stride_log2=6)
                * bpe,
                cutlass.AddressSpace.smem,
                cutlass.BFloat16,
            ),
            4,
            nvvm.MMALayout.ROW,
        )
    )
    c = list(
        nvvm.ldmatrix(
            cutlass.inttoptr(
                base_int
                + swizzle_lin_128b((d0 + 16) * 64 + d0 + ldsm_x4_lane_off, row_stride_log2=6)
                * bpe,
                cutlass.AddressSpace.smem,
                cutlass.BFloat16,
            ),
            4,
            nvvm.MMALayout.COL,
        )
    )

    # ---- T = -(D^{-1} @ C) -------------------------------------------------------
    c_regs = cutlass.Array(cutlass.Float32, 8, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(8):
        c_regs[i] = cutlass.Float32(0.0)
    mma_step(c_regs, d, c, k_step=0, M=16, N=16, ab_dtype=cfg.io_dtype)
    for i in cutlass.range_constexpr(8):
        c_regs[i] = -c_regs[i]
    a_pack = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]

    # ---- C = T @ A^{-1} ----------------------------------------------------------
    ai_frag = list(
        nvvm.ldmatrix(
            cutlass.inttoptr(
                base_int
                + swizzle_lin_128b(d0 * 64 + d0 + ldsm_x4_lane_off, row_stride_log2=6) * bpe,
                cutlass.AddressSpace.smem,
                cutlass.BFloat16,
            ),
            4,
            nvvm.MMALayout.COL,
        )
    )
    o_regs = cutlass.Array(cutlass.Float32, 8, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(8):
        o_regs[i] = cutlass.Float32(0.0)
    mma_step(o_regs, a_pack, ai_frag, k_step=0, M=16, N=16, ab_dtype=cfg.io_dtype)
    o_pack = [fp32_to_fp16(o_regs[2 * j], o_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]

    # ---- store corrected C -------------------------------------------------------
    nvvm.stmatrix(
        cutlass.inttoptr(
            base_int
            + swizzle_lin_128b((d0 + 16) * 64 + d0 + ldsm_x4_lane_off, row_stride_log2=6) * bpe,
            cutlass.AddressSpace.smem,
            cutlass.BFloat16,
        ),
        o_pack,
        nvvm.MMALayout.ROW,
    )


@cute.jit
def blockwise_diagonal_32x32_to_64x64(cfg, base_int, warp_id, lane_id):
    """Off-diagonal correction 32x32 -> 64x64 (2 warps, one 16-row M-band each)."""
    band = warp_id % 2
    bpe = cfg.io_dtype.width // 8
    ldsm_x4_lane_off = (lane_id % 16) * 64 + (lane_id // 16) * 8
    a_frags = []
    for vs in cutlass.range_constexpr(2):
        a_frags += list(
            nvvm.ldmatrix(
                cutlass.inttoptr(
                    base_int
                    + swizzle_lin_128b(
                        (32 + band * 16) * 64 + 32 + vs * 16 + ldsm_x4_lane_off, row_stride_log2=6
                    )
                    * bpe,
                    cutlass.AddressSpace.smem,
                    cutlass.BFloat16,
                ),
                4,
                nvvm.MMALayout.ROW,
            )
        )
    b_frags = []
    for vs in cutlass.range_constexpr(4):
        b_frags += list(
            nvvm.ldmatrix(
                cutlass.inttoptr(
                    base_int
                    + swizzle_lin_128b(
                        (32 + (vs // 2) * 16) * 64 + (vs % 2) * 16 + ldsm_x4_lane_off,
                        row_stride_log2=6,
                    )
                    * bpe,
                    cutlass.AddressSpace.smem,
                    cutlass.BFloat16,
                ),
                4,
                nvvm.MMALayout.COL,
            )
        )

    # ---- T = -(D^{-1} @ C) -------------------------------------------------------
    c_regs = cutlass.Array(cutlass.Float32, 16, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(16):
        c_regs[i] = cutlass.Float32(0.0)
    for ks in cutlass.range_constexpr(2):
        mma_step(
            c_regs,
            a_frags,
            b_frags[ks * 8 : ks * 8 + 8],
            k_step=ks,
            M=16,
            N=32,
            ab_dtype=cfg.io_dtype,
        )
    for i in cutlass.range_constexpr(16):
        c_regs[i] = -c_regs[i]
    a_pack = [fp32_to_fp16(c_regs[2 * j], c_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(8)]

    # ---- C = T @ A^{-1} ----------------------------------------------------------
    ai_frags = []
    for vs in cutlass.range_constexpr(4):
        ai_frags += list(
            nvvm.ldmatrix(
                cutlass.inttoptr(
                    base_int
                    + swizzle_lin_128b(
                        ((vs // 2) * 16) * 64 + (vs % 2) * 16 + ldsm_x4_lane_off, row_stride_log2=6
                    )
                    * bpe,
                    cutlass.AddressSpace.smem,
                    cutlass.BFloat16,
                ),
                4,
                nvvm.MMALayout.COL,
            )
        )
    o_regs = cutlass.Array(cutlass.Float32, 16, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(16):
        o_regs[i] = cutlass.Float32(0.0)
    for ks in cutlass.range_constexpr(2):
        mma_step(
            o_regs,
            a_pack,
            ai_frags[ks * 8 : ks * 8 + 8],
            k_step=ks,
            M=16,
            N=32,
            ab_dtype=cfg.io_dtype,
        )
    o_pack = [fp32_to_fp16(o_regs[2 * j], o_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(8)]

    # ---- store corrected C -------------------------------------------------------
    nvvm.barrier_cta_sync_aligned(
        cfg.inverse_barrier_id,
        thread_count=cfg.inverse_barrier_threads,
    )
    nvvm.stmatrix(
        cutlass.inttoptr(
            base_int
            + swizzle_lin_128b((32 + band * 16) * 64 + ldsm_x4_lane_off, row_stride_log2=6) * bpe,
            cutlass.AddressSpace.smem,
            cutlass.BFloat16,
        ),
        o_pack[0:4],
        nvvm.MMALayout.ROW,
    )
    nvvm.stmatrix(
        cutlass.inttoptr(
            base_int
            + swizzle_lin_128b((32 + band * 16) * 64 + 16 + ldsm_x4_lane_off, row_stride_log2=6)
            * bpe,
            cutlass.AddressSpace.smem,
            cutlass.BFloat16,
        ),
        o_pack[4:8],
        nvvm.MMALayout.ROW,
    )


# ---- Dynamic tile scheduler ------------------------------------------------------


@cute.jit
def sched_publish_next(cfg, bars, sSched, mSched, sched_state, tile_idx, num_ctas):
    """TMA-LDG-warp side: pull the next tile off the global ticket, publish it."""
    if cutlass.const_expr(cfg.dyn_sched):
        bars.mb_sched_done[sched_state.idx].wait(sched_state.phase)
        if nvvm.elect_sync():
            fetched = cutlass.Int32(
                nvvm.atomicrmw(
                    "add", mSched.iterator, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu"
                )
            )
            sSched[sched_state.idx] = num_ctas + fetched
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        next_tile = sSched[sched_state.idx]
        if nvvm.elect_sync():
            bars.mb_sched_ready[sched_state.idx].arrive()
        return next_tile, advance(sched_state, cfg.sched_stages)
    return tile_idx + num_ctas, sched_state


@cute.jit
def sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas):
    """Consumer side: read the TMA-LDG warp's published next tile."""
    if cutlass.const_expr(cfg.dyn_sched):
        bars.mb_sched_ready[sched_state.idx].wait(sched_state.phase)
        next_tile = sSched[sched_state.idx]
        if nvvm.elect_sync():
            bars.mb_sched_done[sched_state.idx].arrive()
        return next_tile, advance(sched_state, cfg.sched_stages)
    return tile_idx + num_ctas, sched_state


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
    sSched,
    bars,
):
    """Epilogue warp role (warp 11): persistent scheduler loop issuing the
    per-chunk O and state-checkpoint TMA stores."""
    elect_one = nvvm.elect_sync()
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    o_index = PipelineState.start(phase=0)
    lidx = tidx % cfg.threads_per_warp
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)

    bpe = cfg.io_dtype.width // 8
    elems_per_128b = 128 // bpe
    sO_tma = SmemTile(
        base=sO_raw,
        elems_per_stage=(cfg.o_cosize // cfg.smem_o_stages),
        stages=cfg.smem_o_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=elems_per_128b,
        tma_subtile_stride_elems=4096,
    )
    if cutlass.const_expr(cfg.enable_checkpoints):
        checkpoint_elems_per_128b = 64
        sCheckpoint_tma = SmemTile(
            base=sCheckpoint_raw,
            elems_per_stage=(cfg.checkpoint_cosize // cfg.smem_checkpoint_stages),
            stages=cfg.smem_checkpoint_stages,
            leading_byte_offset=0,
            stride_byte_offset=0,
            layout=0,
            tma_loads_per_tile=cfg.d_v // checkpoint_elems_per_128b,
            tma_granu_elems=checkpoint_elems_per_128b,
            tma_subtile_stride_elems=cfg.d_k * checkpoint_elems_per_128b,
        )
        checkpoint_store_cnt = cutlass.Int32(0)
        ckpt_chunks = checkpoint_every_n_tokens // cutlass.Int32(cfg.b_t)
    heads_out = cutlass.Int32(cfg.n_heads_out)
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)

    while tile_idx < total_tiles:
        (
            batch_idx,
            head_idx,
            batch_start,
            batch_end,
            seqlen_b,
            num_chunks_b,
            wstart,
            wend,
            cstart,
            cend,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        n_local = wend - cstart

        head_o = head_idx
        slot = batch_idx * desc_qwords
        desc_o_slot = (desc_o_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_o_slot)
        if cutlass.const_expr(cfg.enable_checkpoints):
            desc_checkpoint_slot = (desc_checkpoint_base + slot).tospace(
                cutlass.AddressSpace.generic
            )
            checkpoint_coord = (wstart + ckpt_chunks - cutlass.Int32(1)) // ckpt_chunks
            checkpoint_mod = (cstart + cutlass.Int32(1)) % ckpt_chunks
            if elect_one:
                tma_tensormap_acquire(desc_checkpoint_slot)

        if n_local > 0:
            if cutlass.const_expr(cfg.enable_checkpoints):
                if wstart == 0:
                    checkpoint_stage = checkpoint_store_cnt % cfg.smem_checkpoint_stages
                    checkpoint_phase = (
                        checkpoint_store_cnt // cfg.smem_checkpoint_stages
                    ) & cutlass.Int32(1)
                    bars.mb_checkpoint_tmastg_ready[checkpoint_stage].wait(checkpoint_phase)
                    checkpoint_slice = tma_slice_runtime_desc(
                        desc_checkpoint_slot,
                        cutlass.Int32(0),
                        cutlass.Int32(0),
                        checkpoint_coord,
                        head_o,
                    )
                    tma_store_tile(
                        sCheckpoint_tma[checkpoint_stage], checkpoint_slice, acquire=False
                    )
                    tma_store_commit()
                    tma_store_wait(0)
                    bars.mb_checkpoint_tmastg_done[checkpoint_stage].arrive()
                    checkpoint_coord += 1
                    checkpoint_store_cnt = checkpoint_store_cnt + 1
            for local_idx in cutlass.range(n_local):
                chunk_idx = cstart + local_idx

                did_o = cutlass.Int32(0)
                o_idx = o_index.idx
                bars.mb_o_tmastg_ready[o_idx].wait(o_index.phase)
                o_index = advance(o_index, cfg.smem_o_stages)

                if chunk_idx >= wstart and chunk_idx < wend:
                    tok_coord = chunk_idx * cutlass.Int32(cfg.b_t)
                    o_slice = tma_slice_runtime_desc(
                        desc_o_slot, cutlass.Int32(0), head_o, tok_coord
                    )
                    tma_store_tile(sO_tma[o_idx], o_slice, acquire=False)
                    tma_store_commit()
                    did_o = cutlass.Int32(1)

                did_checkpoint = cutlass.Int32(0)
                if cutlass.const_expr(cfg.enable_checkpoints):
                    checkpoint_stage = checkpoint_store_cnt % cfg.smem_checkpoint_stages
                    checkpoint_phase = (
                        checkpoint_store_cnt // cfg.smem_checkpoint_stages
                    ) & cutlass.Int32(1)
                    if chunk_idx >= wstart - 1 and chunk_idx < wend - 1:
                        if checkpoint_mod == 0:
                            bars.mb_checkpoint_tmastg_ready[checkpoint_stage].wait(
                                checkpoint_phase
                            )
                            checkpoint_slice = tma_slice_runtime_desc(
                                desc_checkpoint_slot,
                                cutlass.Int32(0),
                                cutlass.Int32(0),
                                checkpoint_coord,
                                head_o,
                            )
                            tma_store_tile(
                                sCheckpoint_tma[checkpoint_stage], checkpoint_slice, acquire=False
                            )
                            tma_store_commit()
                            checkpoint_coord += 1
                            did_checkpoint = cutlass.Int32(1)
                    checkpoint_mod = checkpoint_mod + cutlass.Int32(1)
                    checkpoint_mod = (
                        cutlass.Int32(0) if checkpoint_mod == ckpt_chunks else checkpoint_mod
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
                        bars.mb_o_tmastg_done[o_idx].arrive()
                else:
                    if did_o == 1:
                        tma_store_wait(0)
                    bars.mb_o_tmastg_done[o_idx].arrive()

        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


@cute.jit
def gate_beta_warp(
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
    mBeta,
    sCumsumlog,
    sCumprod,
    sBeta,
    sSched,
    bars,
):
    """Gate/Beta producer (warp 8): persistent scheduler loop + the
    cumsum/cumprod/Beta chunk loads."""

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    gate_index = PipelineState.start(phase=1)
    beta_index = PipelineState.start(phase=1)
    lidx = tidx % cfg.threads_per_warp

    a_l2 = cutlass.Float32(0.0)
    bias = cutlass.Float32(0.0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        (
            batch_idx,
            head_idx,
            batch_start,
            batch_end,
            seqlen_b,
            num_chunks_b,
            wstart,
            wend,
            cstart,
            cend,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        n_local = wend - cstart
        if cutlass.const_expr(cfg.safe_gate):
            if n_local > 0:
                # per-head transform constants, fixed for the whole tile
                a_l2 = -cute.math.exp2(
                    mA_log[head_idx].to(cutlass.Float32) * cutlass.Float32(RCP_LN2), fastmath=True
                ) * cutlass.Float32(RCP_LN2)
                bias = mDt_bias[head_idx].to(cutlass.Float32)
        if n_local > 0:
            for local_idx in cutlass.range(n_local):
                # ---- Gate load: GMEM -> SMEM (OOB neutral) -----------------------
                chunk_idx = cstart + local_idx
                n_cols = cfg.b_t // cfg.threads_per_warp
                chunk_offset = batch_start + chunk_idx * cfg.b_t
                gGateSeq = mGate[None, head_idx]
                gBeta = cute.domain_offset((chunk_offset,), mBeta[None, head_idx])

                gate_idx = gate_index.idx
                gate_phase = gate_index.phase
                gate_index = advance(gate_index, cfg.smem_gate_stages)

                pos_valid = [None] * n_cols
                gate_vals = [cutlass.Float32(0.0)] * n_cols
                oob_neutral = (
                    cutlass.Float32(0.0)
                    if cutlass.const_expr(cfg.log_gate)
                    else cutlass.Float32(1.0)
                )
                for col in cutlass.range_constexpr(n_cols):
                    tok = chunk_offset + lidx + col * cfg.threads_per_warp
                    pos_valid[col] = cute.elem_less(tok, batch_end)
                    tok_clamped = min(tok, batch_end - 1)
                    gate_vals[col] = gGateSeq[tok_clamped] if pos_valid[col] else oob_neutral

                if cutlass.const_expr(cfg.safe_gate):
                    # raw logits -> log2-domain decay: a_l2 * softplus(g + bias) (split-K scan arithmetic)
                    for col in cutlass.range_constexpr(n_cols):
                        contrib = a_l2 * softplus(gate_vals[col] + bias)
                        gate_vals[col] = contrib if pos_valid[col] else cutlass.Float32(0.0)
                elif cutlass.const_expr(cfg.log_gate):
                    for col in cutlass.range_constexpr(n_cols):
                        gate_vals[col] = gate_vals[col] * cutlass.Float32(RCP_LN2)
                else:
                    for col in cutlass.range_constexpr(n_cols):
                        gate_vals[col] = cute.math.log2(gate_vals[col] + 1e-10, fastmath=True)
                for offset in [1, 2, 4, 8, 16]:
                    for col in cutlass.range_constexpr(n_cols):
                        n = nvvm.shfl_sync(
                            0xFFFFFFFF, gate_vals[col], offset, 0, kind=nvvm.Shfl.UP
                        )
                        if lidx >= offset:
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
                    pos = lidx + col * cfg.threads_per_warp
                    sCumsumlog[pos, 0, gate_idx] = gate_vals[col]
                    sCumprod[pos, 0, gate_idx] = cute.math.exp2(gate_vals[col], fastmath=True)
                bars.mb_gate_ready[gate_idx].arrive()

                # ---- Beta load: GMEM -> SMEM (per-element cp.async) --------------------------
                beta_idx = beta_index.idx
                bars.mb_beta_done[beta_idx].wait(beta_index.phase)
                beta_index = advance(beta_index, cfg.smem_beta_stages)
                if cutlass.const_expr(cfg.beta_sigmoid):
                    # io-dtype logits -> sigmoid (tanh identity) -> fp32 SMEM
                    for col in cutlass.range_constexpr(n_cols):
                        pos = lidx + col * cfg.threads_per_warp
                        beta_value = cutlass.Float32(0.0)
                        if pos_valid[col]:
                            beta_value = gBeta[pos].to(cutlass.Float32)
                            half = cutlass.Float32(0.5)
                            beta_value = (
                                (cute.math.tanh(beta_value * half, approx=True) * half + half)
                                .to(mBeta.element_type)
                                .to(cutlass.Float32)
                            )
                        sBeta[pos, 0, beta_idx] = beta_value
                    bars.mb_beta_ready[beta_idx].arrive()
                else:
                    for col in cutlass.range_constexpr(n_cols):
                        pos = lidx + col * cfg.threads_per_warp
                        src = gBeta.iterator + gBeta.layout((pos,))
                        dst = sBeta.iterator + sBeta.layout((pos, 0, beta_idx))
                        cp_size = cutlass.Int32(4) * cutlass.Int32(pos_valid[col])
                        nvvm.cp_async_shared_global(
                            dst, src, 4, nvvm.LoadCacheModifier.CA, cp_size=cp_size
                        )
                    nvvm.cp_async_mbarrier_arrive(
                        bars.mb_beta_ready[beta_idx].smem_ptr, noinc=True
                    )
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)

    for _ in range(cfg.smem_gate_stages):
        bars.mb_gate_done[gate_index.idx].wait(gate_index.phase)
        gate_index = advance(gate_index, cfg.smem_gate_stages)
    for _ in range(cfg.smem_beta_stages):
        bars.mb_beta_done[beta_index.idx].wait(beta_index.phase)
        beta_index = advance(beta_index, cfg.smem_beta_stages)


@cute.jit
def mma_warp(
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
    sSched,
    bars,
):
    """MMA issuer role (warp 10): persistent scheduler loop issuing every
    tcgen05 GEMM."""
    elect_one = nvvm.elect_sync()

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    o_acc_index = PipelineState.start(phase=1)
    o_state_scale_index = PipelineState.start(phase=0)
    kv_acc_index = PipelineState.start(phase=1)
    kq_index = PipelineState.start(phase=0)
    cg0_acc_index = PipelineState.start(phase=1)
    kq_fused_index = PipelineState.start(phase=0)
    tinv_index = PipelineState.start(phase=0)
    a_index = PipelineState.start(phase=0)
    state_inp_index = PipelineState.start(phase=0)
    y_inp_ready = PipelineState.start(phase=0)
    u_inp_ready = PipelineState.start(phase=0)
    decay_u_inp_ready = PipelineState.start(phase=0)

    nvvm.tcgen05_alloc(tmem_base_slot, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_alloc_barrier_id,
        thread_count=cfg.tmem_alloc_barrier_threads,
    )

    # ---- chunk-invariant GEMM descriptors ----------------------------------------
    idesc_qk = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=2 * cfg.b_t,
    )
    bmm_qk_desc = MmaDesc(
        M=2 * cfg.b_t,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=cfg.io_dtype.width // 8,
        bpe_b=cfg.io_dtype.width // 8,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_qk,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bpe = cfg.io_dtype.width // 8
    idesc_q_state = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_q_state_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=idesc_q_state,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_qkv_ts = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_qkv_ts_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=idesc_qkv_ts,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_kv = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_v,
        m_dim=cfg.d_k,
        b_major=1,
    )
    bmm_kv_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.d_v,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=idesc_kv,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    KQ_SEG = (2 * cfg.b_t * 64 * bpe) >> 4
    KQ_BOX = (cfg.b_t * 64 * bpe) >> 4
    KQ_HALF_K = (cfg.d_k // 16) // 2
    KQ_A_HALF = KQ_HALF_K * bmm_q_state_desc.tmem_advance_A

    ACC_STAGE_COLS = cfg.b_t
    KV_ACC_STAGE_COLS = cfg.d_v
    STATE_INP_STAGE_COLS = cfg.d_k // 2
    INP_SLOT_COLS = cfg.b_t // 2

    tmem_base = tmem_base_slot.load()
    tmem_cg0_acc_col_f = tmem_base + cfg.tmem_cg0_acc_offset
    tmem_state_col = tmem_base + cfg.tmem_state_acc_offset
    tmem_q_state_col = tmem_base + cfg.tmem_q_state_acc_offset
    tmem_state_inp_col = tmem_base + cfg.tmem_state_inp_offset
    tmem_inp_col = tmem_base + cfg.tmem_y_decay_u_inp_offset
    y_inp_ptr = nvvm.make_tmem_ptr(tmem_inp_col, cutlass.Int8)
    u_inp_ptr = nvvm.make_tmem_ptr(tmem_inp_col, cutlass.Int8)
    decay_u_inp_ptr = nvvm.make_tmem_ptr(tmem_inp_col + INP_SLOT_COLS, cutlass.Int8)
    k_state_acc_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_cg1_acc_offset, cutlass.Float32)
    u_acc_ptr = k_state_acc_ptr

    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        (
            batch_idx,
            head_idx,
            batch_start,
            batch_end,
            seqlen_b,
            num_chunks_b,
            wstart,
            wend,
            cstart,
            cend,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        n_local = wend - cstart

        # ---- fused KK^T/QK^T pair 0: each member issued ahead of the loop -------
        if n_local > 0:
            f0_acc_idx = cg0_acc_index.idx
            bars.mb_cg0_acc_done[f0_acc_idx].wait(cg0_acc_index.phase)
            cg0_acc_index = advance(cg0_acc_index, cfg.tmem_cg0_acc_stages)
            kqf_idx = kq_fused_index.idx
            bars.mb_kq_ready[kqf_idx].wait(kq_fused_index.phase)
            kq_fused_index = advance(kq_fused_index, cfg.smem_kq_stages)
            desc_kqf = sKQ[kqf_idx].desc()
            acc_cg0 = nvvm.make_tmem_ptr(
                tmem_cg0_acc_col_f + f0_acc_idx * cfg.b_t, cutlass.Float32
            )
            mma_ss(bmm_qk_desc, desc_kqf, desc_kqf, acc_cg0, accumulate=False, k_count=KQ_HALF_K)
            mma_ss(
                bmm_qk_desc,
                desc_kqf + KQ_SEG,
                desc_kqf + KQ_SEG,
                acc_cg0,
                accumulate=True,
                k_count=KQ_HALF_K,
            )
            if elect_one:
                bars.mb_cg0_acc_ready[f0_acc_idx].arrive(cta_group=1)
        if n_local > 1:
            pend_acc_idx = cg0_acc_index.idx
            bars.mb_cg0_acc_done[pend_acc_idx].wait(cg0_acc_index.phase)
            cg0_acc_index = advance(cg0_acc_index, cfg.tmem_cg0_acc_stages)
            kqf_idx = kq_fused_index.idx
            bars.mb_kq_ready[kqf_idx].wait(kq_fused_index.phase)
            kq_fused_index = advance(kq_fused_index, cfg.smem_kq_stages)
            desc_kqf = sKQ[kqf_idx].desc()
            desc_kqf_b = desc_kqf + KQ_BOX
            acc_cg0 = nvvm.make_tmem_ptr(
                tmem_cg0_acc_col_f + pend_acc_idx * cfg.b_t, cutlass.Float32
            )
            mma_ss(bmm_qk_desc, desc_kqf, desc_kqf_b, acc_cg0, accumulate=False, k_count=KQ_HALF_K)
            mma_ss(
                bmm_qk_desc,
                desc_kqf + KQ_SEG,
                desc_kqf_b + KQ_SEG,
                acc_cg0,
                accumulate=True,
                k_count=KQ_HALF_K,
            )
            if elect_one:
                bars.mb_cg0_acc_ready[pend_acc_idx].arrive(cta_group=1)

        for local_idx in cutlass.range(n_local):  # noqa: B007
            if cutlass.const_expr(cfg.use_initial_state):
                if local_idx == 0:
                    kv_acc_index = advance(kv_acc_index, cfg.tmem_state_acc_stages)
            have_state = (
                cutlass.Boolean(True)
                if cutlass.const_expr(cfg.use_initial_state)
                else local_idx > 0
            )

            kq_idx = kq_index.idx
            member = local_idx & 1
            state_inp_idx = state_inp_index.idx
            q_state_acc_idx = o_acc_index.idx
            tinv_idx = tinv_index.idx
            a_idx = a_index.idx
            o_scale_idx = o_state_scale_index.idx
            kv_acc_idx = kv_acc_index.idx
            kq_member_off = member * KQ_BOX
            desc_k = sKQ[kq_idx].desc() + kq_member_off
            desc_q = sKQ[kq_idx].desc() + (KQ_BOX - kq_member_off)
            desc_tinv = sTinv[tinv_idx].desc()
            desc_a = sA[a_idx].desc()
            desc_kt = sKQ_trans[kq_idx].desc() + kq_member_off
            state_a_ptr = nvvm.make_tmem_ptr(
                tmem_state_inp_col + state_inp_idx * STATE_INP_STAGE_COLS, cutlass.Int8
            )
            q_state_acc_ptr = nvvm.make_tmem_ptr(
                tmem_q_state_col + q_state_acc_idx * ACC_STAGE_COLS, cutlass.Float32
            )
            qkv_acc_ptr = nvvm.make_tmem_ptr(
                tmem_q_state_col + o_scale_idx * ACC_STAGE_COLS, cutlass.Float32
            )
            state_acc_ptr = nvvm.make_tmem_ptr(
                tmem_state_col + kv_acc_idx * KV_ACC_STAGE_COLS, cutlass.Float32
            )

            kq_index = advance(kq_index, cfg.smem_kq_stages)

            # ---- QK/KK lookahead (member 1) = [Q;K](S) @ K^T ---------------------
            if member == 1:
                if local_idx + 2 < n_local:
                    pend_acc_idx = cg0_acc_index.idx
                    bars.mb_cg0_acc_done[pend_acc_idx].wait(cg0_acc_index.phase)
                    cg0_acc_index = advance(cg0_acc_index, cfg.tmem_cg0_acc_stages)
                    kqf_idx = kq_fused_index.idx
                    bars.mb_kq_ready[kqf_idx].wait(kq_fused_index.phase)
                    kq_fused_index = advance(kq_fused_index, cfg.smem_kq_stages)
                    desc_kqf = sKQ[kqf_idx].desc()
                    desc_kqf_b = desc_kqf + KQ_BOX
                    acc_cg0 = nvvm.make_tmem_ptr(
                        tmem_cg0_acc_col_f + pend_acc_idx * cfg.b_t, cutlass.Float32
                    )
                    mma_ss(
                        bmm_qk_desc,
                        desc_kqf,
                        desc_kqf_b,
                        acc_cg0,
                        accumulate=False,
                        k_count=KQ_HALF_K,
                    )
                    mma_ss(
                        bmm_qk_desc,
                        desc_kqf + KQ_SEG,
                        desc_kqf_b + KQ_SEG,
                        acc_cg0,
                        accumulate=True,
                        k_count=KQ_HALF_K,
                    )
                    if elect_one:
                        bars.mb_cg0_acc_ready[pend_acc_idx].arrive(cta_group=1)

            # ---- K*state^T (GEMM 3) = state^T(T) @ K^T ------------------------------------
            if have_state:
                bars.mb_state_inp_ready[state_inp_idx].wait(state_inp_index.phase)
                state_inp_index = advance(state_inp_index, cfg.tmem_state_inp_stages)

                for k in cutlass.range_constexpr(KQ_HALF_K):
                    mma_ts_step(
                        bmm_q_state_desc,
                        state_a_ptr,
                        desc_k,
                        k_state_acc_ptr,
                        k,
                        cutlass.Boolean(k > 0),
                    )
                for k in cutlass.range_constexpr(KQ_HALF_K):
                    mma_ts_step(
                        bmm_q_state_desc,
                        state_a_ptr.subview(KQ_A_HALF),
                        desc_k + KQ_SEG,
                        k_state_acc_ptr,
                        k,
                        cutlass.Boolean(True),
                    )
                if elect_one:
                    bars.mb_k_state_acc_ready[0].arrive(cta_group=1)

            # ---- Q*state^T (GEMM 4) = state^T(T) @ Q^T ------------------------------------
            o_acc_index = advance(o_acc_index, cfg.tmem_q_state_acc_stages)
            if have_state:
                for k in cutlass.range_constexpr(KQ_HALF_K):
                    mma_ts_step(
                        bmm_q_state_desc,
                        state_a_ptr,
                        desc_q,
                        q_state_acc_ptr,
                        k,
                        cutlass.Boolean(k > 0),
                    )
                for k in cutlass.range_constexpr(KQ_HALF_K):
                    mma_ts_step(
                        bmm_q_state_desc,
                        state_a_ptr.subview(KQ_A_HALF),
                        desc_q + KQ_SEG,
                        q_state_acc_ptr,
                        k,
                        cutlass.Boolean(True),
                    )
                if elect_one:
                    bars.mb_o_acc_ready[q_state_acc_idx].arrive(cta_group=1)

            # ---- U^T (GEMM 5) = Y^T(T) @ T_inv^T ---------------------------------------
            bars.mb_t_inv_ready[tinv_idx].wait(tinv_index.phase)
            tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
            bars.mb_y_inp_ready[0].wait(y_inp_ready.phase)
            y_inp_ready = advance(y_inp_ready, 1)
            for k in cutlass.range_constexpr(cfg.b_t // 16):
                mma_ts_step(
                    bmm_qkv_ts_desc, y_inp_ptr, desc_tinv, u_acc_ptr, k, cutlass.Boolean(k > 0)
                )
            if elect_one:
                bars.mb_u_acc_ready[0].arrive(cta_group=1)
                bars.mb_t_inv_done[tinv_idx].arrive(cta_group=1)

            # ---- KK/QK lookahead (member 0) = [K;Q](S) @ K^T ---------------------
            if member == 0:
                if local_idx + 2 < n_local:
                    f0_acc_idx = cg0_acc_index.idx
                    bars.mb_cg0_acc_done[f0_acc_idx].wait(cg0_acc_index.phase)
                    cg0_acc_index = advance(cg0_acc_index, cfg.tmem_cg0_acc_stages)
                    kqf_idx = kq_fused_index.idx
                    bars.mb_kq_ready[kqf_idx].wait(kq_fused_index.phase)
                    kq_fused_index = advance(kq_fused_index, cfg.smem_kq_stages)
                    desc_kqf = sKQ[kqf_idx].desc()
                    acc_cg0 = nvvm.make_tmem_ptr(
                        tmem_cg0_acc_col_f + f0_acc_idx * cfg.b_t, cutlass.Float32
                    )
                    mma_ss(
                        bmm_qk_desc,
                        desc_kqf,
                        desc_kqf,
                        acc_cg0,
                        accumulate=False,
                        k_count=KQ_HALF_K,
                    )
                    mma_ss(
                        bmm_qk_desc,
                        desc_kqf + KQ_SEG,
                        desc_kqf + KQ_SEG,
                        acc_cg0,
                        accumulate=True,
                        k_count=KQ_HALF_K,
                    )
                    if elect_one:
                        bars.mb_cg0_acc_ready[f0_acc_idx].arrive(cta_group=1)

            # ---- O^T (GEMM 6) += U input^T(T) @ A^T ------------------------------
            bars.mb_a_ready[a_idx].wait(a_index.phase)
            a_index = advance(a_index, cfg.smem_a_stages)
            if have_state:
                bars.mb_o_state_scale_acc_done[o_scale_idx].wait(o_state_scale_index.phase)
                o_state_scale_index = advance(o_state_scale_index, cfg.tmem_q_state_acc_stages)
            bars.mb_u_inp_ready[0].wait(u_inp_ready.phase)
            u_inp_ready = advance(u_inp_ready, 1)
            for k in cutlass.range_constexpr(cfg.b_t // 16):
                mma_ts_step(
                    bmm_qkv_ts_desc,
                    u_inp_ptr,
                    desc_a,
                    qkv_acc_ptr,
                    k,
                    cutlass.Boolean(True) if cutlass.const_expr(k > 0) else have_state,
                )
            if elect_one:
                bars.mb_a_done[a_idx].arrive(cta_group=1)
                bars.mb_o_final_acc_ready[o_scale_idx].arrive(cta_group=1)

            # ---- state^T (GEMM 7) += decayed U^T(T) @ K ------------------------------
            bars.mb_decay_u_inp_ready[0].wait(decay_u_inp_ready.phase)
            decay_u_inp_ready = advance(decay_u_inp_ready, 1)
            kv_acc_index = advance(kv_acc_index, cfg.tmem_state_acc_stages)
            for k in cutlass.range_constexpr(cfg.b_t // 16):
                mma_ts_step(
                    bmm_kv_desc,
                    decay_u_inp_ptr,
                    desc_kt,
                    state_acc_ptr,
                    k,
                    cutlass.Boolean(True) if cutlass.const_expr(k > 0) else have_state,
                )
            if elect_one:
                bars.mb_state_acc_ready[kv_acc_idx].arrive(cta_group=1)
                bars.mb_kq_done[kq_idx].arrive(cta_group=1)

        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)

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
    n_desc,
    cu_seqlens,
    mWorkItems,
    sKQ_raw,
    sV_raw,
    desc_q_base,
    desc_k_base,
    desc_v_base,
    mSched,
    sSched,
    bars,
):
    """TMA-LDG warp role (warp 9): persistent scheduler loop + per-chunk
    Q/K/V G->S TMA loads."""
    elect_one = nvvm.elect_sync()
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    kq_index = PipelineState.start(phase=1)
    v_index = PipelineState.start(phase=1)
    sched_state = PipelineState.start(phase=1)
    tile_idx = cutlass.Int32(bidx)

    bpe = cfg.io_dtype.width // 8
    elems_per_128b = 128 // bpe
    bt = cfg.b_t
    kq_stage_elems = cfg.kq_cosize // cfg.smem_kq_stages
    kq_box_elems = kq_stage_elems // 4
    sKQ_lo_tma = SmemTile(
        base=sKQ_raw,
        elems_per_stage=kq_stage_elems,
        stages=cfg.smem_kq_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=elems_per_128b,
        tma_subtile_stride_elems=2 * bt * elems_per_128b,
    )
    sV_tma = SmemTile(
        base=sV_raw,
        elems_per_stage=cfg.v_cosize // cfg.smem_v_stages,
        stages=cfg.smem_v_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=elems_per_128b,
        tma_subtile_stride_elems=4096,
    )
    heads_out = cutlass.Int32(cfg.n_heads_out)
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)
    packed_tokens = cutlass.Int32(cu_seqlens[n_desc])

    while tile_idx < total_tiles:
        (
            batch_idx,
            head_idx,
            batch_start,
            batch_end,
            seqlen_b,
            num_chunks_b,
            wstart,
            wend,
            cstart,
            cend,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)

        head_q = head_idx if cfg.q_ratio == 1 else head_idx // cutlass.Int32(cfg.q_ratio)
        head_k = head_idx if cfg.k_ratio == 1 else head_idx // cutlass.Int32(cfg.k_ratio)
        head_v = head_idx if cfg.v_ratio == 1 else head_idx // cutlass.Int32(cfg.v_ratio)
        slot = batch_idx * desc_qwords
        desc_q_slot = (desc_q_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_v_slot)
        use_packed_coords = batch_end == packed_tokens

        if wend > cstart:
            kq_idx = kq_index.idx
            bars.mb_kq_done[kq_idx].wait(kq_index.phase)
            kq_index = advance(kq_index, cfg.smem_kq_stages)
            if elect_one:
                bars.mb_kq_ready[kq_idx].arrive(n_bytes=cfg.tma_kq_bytes)
            tok_coord = cstart * cutlass.Int32(cfg.b_t)
            input_tok_coord = batch_start + tok_coord if use_packed_coords else tok_coord
            k_slice = tma_slice_runtime_desc(
                desc_k_slot, cutlass.Int32(0), head_k, input_tok_coord
            )
            q_slice = tma_slice_runtime_desc(
                desc_q_slot, cutlass.Int32(0), head_q, input_tok_coord
            )
            kq_tile = sKQ_lo_tma[kq_idx]
            tma_load_tile(kq_tile, k_slice, bars.mb_kq_ready[kq_idx].smem_ptr, acquire=False)
            tma_load_tile(
                kq_tile.shifted(kq_box_elems),
                q_slice,
                bars.mb_kq_ready[kq_idx].smem_ptr,
                acquire=False,
            )
            for chunk_idx in cutlass.range(cstart + 1, wend):
                tok_coord = chunk_idx * cutlass.Int32(cfg.b_t)
                input_tok_coord = batch_start + tok_coord if use_packed_coords else tok_coord

                # ---- K + Q interleaved -------------------------------------------
                kq_idx = kq_index.idx
                bars.mb_kq_done[kq_idx].wait(kq_index.phase)
                kq_index = advance(kq_index, cfg.smem_kq_stages)
                if elect_one:
                    bars.mb_kq_ready[kq_idx].arrive(n_bytes=cfg.tma_kq_bytes)
                member = (chunk_idx - cstart) & 1
                k_slice = tma_slice_runtime_desc(
                    desc_k_slot, cutlass.Int32(0), head_k, input_tok_coord
                )
                q_slice = tma_slice_runtime_desc(
                    desc_q_slot, cutlass.Int32(0), head_q, input_tok_coord
                )
                kq_tile = sKQ_lo_tma[kq_idx]
                if member == 0:
                    tma_load_tile(
                        kq_tile, k_slice, bars.mb_kq_ready[kq_idx].smem_ptr, acquire=False
                    )
                    tma_load_tile(
                        kq_tile.shifted(kq_box_elems),
                        q_slice,
                        bars.mb_kq_ready[kq_idx].smem_ptr,
                        acquire=False,
                    )
                else:
                    tma_load_tile(
                        kq_tile, q_slice, bars.mb_kq_ready[kq_idx].smem_ptr, acquire=False
                    )
                    tma_load_tile(
                        kq_tile.shifted(kq_box_elems),
                        k_slice,
                        bars.mb_kq_ready[kq_idx].smem_ptr,
                        acquire=False,
                    )

                # ---- V load ------------------------------------------------------
                v_idx = v_index.idx
                bars.mb_v_done[v_idx].wait(v_index.phase)
                v_index = advance(v_index, cfg.smem_v_stages)
                if elect_one:
                    bars.mb_v_ready[v_idx].arrive(n_bytes=cfg.tma_v_bytes)
                v_tok = (chunk_idx - 1) * cutlass.Int32(cfg.b_t)
                input_v_tok = batch_start + v_tok if use_packed_coords else v_tok
                v_slice = tma_slice_runtime_desc(
                    desc_v_slot, cutlass.Int32(0), head_v, input_v_tok
                )
                tma_load_tile(
                    sV_tma[v_idx], v_slice, bars.mb_v_ready[v_idx].smem_ptr, acquire=False
                )

            v_idx = v_index.idx
            bars.mb_v_done[v_idx].wait(v_index.phase)
            v_index = advance(v_index, cfg.smem_v_stages)
            if elect_one:
                bars.mb_v_ready[v_idx].arrive(n_bytes=cfg.tma_v_bytes)
            v_tok = (wend - cutlass.Int32(1)) * cutlass.Int32(cfg.b_t)
            input_v_tok = batch_start + v_tok if use_packed_coords else v_tok
            v_slice = tma_slice_runtime_desc(desc_v_slot, cutlass.Int32(0), head_v, input_v_tok)
            tma_load_tile(sV_tma[v_idx], v_slice, bars.mb_v_ready[v_idx].smem_ptr, acquire=False)

        tile_idx, sched_state = sched_publish_next(
            cfg, bars, sSched, mSched, sched_state, tile_idx, num_ctas
        )

    for _ in range(cfg.smem_kq_stages):
        bars.mb_kq_done[kq_index.idx].wait(kq_index.phase)
        kq_index = advance(kq_index, cfg.smem_kq_stages)
    for _ in range(cfg.smem_v_stages):
        bars.mb_v_done[v_index.idx].wait(v_index.phase)
        v_index = advance(v_index, cfg.smem_v_stages)


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
    sCumsumlog,
    sBeta,
    sTinv,
    sA,
    sCheckpoint_raw,
    checkpoint_every_n_tokens,
    sSched,
    bars,
):
    """Compute warp-group 0 role (warps 0-3): persistent scheduler loop
    computing each chunk pair's T_inv and A epilogues."""

    nvvm.setmaxregister(cfg.num_regs_compute_group_0, nvvm.SetMaxRegisterAction.INCREASE)
    gate_index = PipelineState.start(phase=0)
    beta_index = PipelineState.start(phase=0)
    cg0_acc_ready = PipelineState.start(phase=0)
    tinv_index = PipelineState.start(phase=1)
    a_index = PipelineState.start(phase=1)

    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_alloc_barrier_id,
        thread_count=cfg.tmem_alloc_barrier_threads,
    )
    tmem_base = tmem_base_slot.load()

    num_threads_cg0 = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    cg0_tidx = tidx % num_threads_cg0
    warp_id = cg0_tidx // cfg.threads_per_warp
    lane_id = cg0_tidx % cfg.threads_per_warp
    inverse_local_warp = warp_id % 2
    pair_half = warp_id // 2
    half_row_base = inverse_local_warp * 32
    bpe = cfg.io_dtype.width // 8
    num_vals = 32
    FRAG_COLS = 16
    ACC_N_FRAGS = cfg.b_t // FRAG_COLS
    store_row = warp_id * 16 + lane_id % 16
    store_row_frag = lane_id % 16
    store_col = (lane_id // 16) * 8
    tmem_warp_row = warp_id * cfg.threads_per_warp
    tmem_cg0_acc_col = tmem_base + cfg.tmem_cg0_acc_offset
    ACC_STAGE_COLS = cfg.b_t
    mask_zero = opaque_f32_zero()
    crow_lo = warp_id * 16 + lane_id // 4
    crow_hi = crow_lo + 8

    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        (
            batch_idx,
            head_idx,
            batch_start,
            batch_end,
            seqlen_b,
            num_chunks_b,
            wstart,
            wend,
            cstart,
            cend,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        n_local = wend - cstart
        n_pairs = (n_local + 1) // 2

        for pair_i in cutlass.range(n_pairs):
            # An odd chunk count leaves the last pair with member 0 only; have_m1
            # is uniform across CG0, so the shared barriers below stay aligned.
            have_m1 = pair_i * 2 + 1 < n_local
            do_kk = have_m1 or pair_half == 0
            do_a = have_m1 or pair_half == 1

            # ---- Gate rows for this warp's KK / A member roles -------------------
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

            row_u0_lo = half_row_base + lane_id // 4
            row_u0_hi = row_u0_lo + 8
            row_u1_lo = row_u0_lo + 16
            row_u1_hi = row_u0_lo + 24

            kk_cumsumlog_rows = []
            a_cumsumlog_rows = []
            for r in (row_u0_lo, row_u0_hi, row_u1_lo, row_u1_hi):
                kk_cumsumlog_rows.append(sCumsumlog[r, 0, kk_gate_idx])
                a_cumsumlog_rows.append(sCumsumlog[r, 0, a_gate_idx])
            kk_cumsumlog_cols = []
            a_cumsumlog_cols = []
            for g in cutlass.range_constexpr(8):
                for b in cutlass.range_constexpr(2):
                    ccol = (lane_id % 4) * 2 + g * 8 + b
                    kk_cumsumlog_cols.append(sCumsumlog[ccol, 0, kk_gate_idx])
                    a_cumsumlog_cols.append(sCumsumlog[ccol, 0, a_gate_idx])

            decay_t_kk = []
            decay_t_a = []
            for u in cutlass.range_constexpr(2):
                for k in cutlass.range_constexpr(num_vals):
                    hi_row = ((k // 2) % 2) == 1
                    crow_u0 = row_u0_hi if cutlass.const_expr(hi_row) else row_u0_lo
                    crow_u1 = row_u1_hi if cutlass.const_expr(hi_row) else row_u1_lo
                    crow = crow_u1 if cutlass.const_expr(u == 1) else crow_u0
                    ccol = (lane_id % 4) * 2 + ((k // 4) * 8 + k % 2)
                    is_lower = crow >= ccol
                    kk_row_cumsumlog = kk_cumsumlog_rows[u * 2 + (1 if hi_row else 0)]
                    a_row_cumsumlog = a_cumsumlog_rows[u * 2 + (1 if hi_row else 0)]
                    col = (k // 4) * 2 + (k % 2)
                    decay_t_kk.append(
                        cute.math.exp2(kk_row_cumsumlog - kk_cumsumlog_cols[col], fastmath=True)
                        if is_lower
                        else mask_zero
                    )
                    decay_t_a.append(
                        cute.math.exp2(a_row_cumsumlog - a_cumsumlog_cols[col], fastmath=True)
                        if is_lower
                        else mask_zero
                    )
            bars.mb_gate_done[gate0_idx].arrive()
            if have_m1:
                bars.mb_gate_done[gate1_idx].arrive()

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

            # ---- KK_epi (each warp pair stages its own member) -------------------
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

            tinv0_base = smem_data_ptr(sTinv[tinv0_idx].base).toint()
            tinv1_base = smem_data_ptr(sTinv[tinv1_idx].base).toint()
            kk_base = tinv1_base if pair_half == 1 else tinv0_base
            if do_kk:
                bars.mb_cg0_acc_ready[kk_acc_idx].wait(kk_acc_phase)
                kk_vec0 = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(
                        (tmem_warp_row << 16) + tmem_cg0_acc_col + kk_acc_idx * ACC_STAGE_COLS,
                        cutlass.Float32,
                    ),
                    num=8,
                )
                kk_vec1 = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(
                        ((tmem_warp_row + 16) << 16)
                        + tmem_cg0_acc_col
                        + kk_acc_idx * ACC_STAGE_COLS,
                        cutlass.Float32,
                    ),
                    num=8,
                )
                bars.mb_t_inv_done[kk_tinv_idx].wait(kk_tinv_phase)
                for u in cutlass.range_constexpr(2):
                    kk_vec = kk_vec1 if cutlass.const_expr(u == 1) else kk_vec0
                    kk_pack = []
                    for k in cutlass.range_constexpr(num_vals // 2):
                        b0 = (
                            kk_beta[u * 2 + 1]
                            if cutlass.const_expr((k % 2) == 1)
                            else kk_beta[u * 2]
                        )
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
                            cutlass.inttoptr(
                                kk_base
                                + (
                                    st_row * cfg.b_t
                                    + swizzle_xor_128b(st_row, store_col + c * FRAG_COLS)
                                )
                                * bpe,
                                cutlass.AddressSpace.smem,
                                cutlass.BFloat16,
                            ),
                            [
                                kk_pack[c * 4 + 0],
                                kk_pack[c * 4 + 1],
                                kk_pack[c * 4 + 2],
                                kk_pack[c * 4 + 3],
                            ],
                            nvvm.MMALayout.ROW,
                        )

            # ---- pair inverse: warps 0-1 own matrix 0, warps 2-3 matrix 1 --------
            # With no member 1 there is one matrix, so warps 2-3 idle through the
            # per-warp steps; the barriers below stay unconditional.
            inv_base = tinv0_base
            if have_m1:
                inv_base = tinv1_base if warp_id >= 2 else tinv0_base
            do_inv = have_m1 or warp_id < 2

            # diagonal 8x8 Gauss-Jordan, all four warps
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )
            if do_inv:
                invert_diagonal_NxN(
                    cfg,
                    inv_base,
                    (inverse_local_warp * cfg.threads_per_warp + lane_id) // 8,
                    cg0_tidx,
                    8,
                )
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            # 8x8 -> 16x16 (both matrices per warp)
            blockwise_diagonal_8x8_to_16x16(cfg, tinv0_base, warp_id * 16, lane_id)
            if have_m1:
                blockwise_diagonal_8x8_to_16x16(cfg, tinv1_base, warp_id * 16, lane_id)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            # 16x16 -> 32x32, one tile per warp within the group
            if do_inv:
                blockwise_diagonal_16x16_to_32x32(cfg, inv_base, inverse_local_warp * 32, lane_id)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            # 32x32 -> 64x64, two warps per matrix
            # all four warps: this step carries its own CG0 barrier, and on the tail
            # inv_base is matrix 0 for every warp so the duplicate band is identical
            blockwise_diagonal_32x32_to_64x64(cfg, inv_base, inverse_local_warp, lane_id)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            # ---- Beta column-scaling + publish, stage 0 --------------------------
            beta_col = []
            for k in cutlass.range_constexpr(num_vals):
                beta_col.append(sBeta[(lane_id % 4) * 2 + ((k // 4) * 8 + k % 2), 0, beta0_idx])
            tinv_frags = []
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                tinv_frags += list(
                    nvvm.ldmatrix(
                        cutlass.inttoptr(
                            tinv0_base
                            + (
                                store_row * cfg.b_t
                                + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)
                            )
                            * bpe,
                            cutlass.AddressSpace.smem,
                            cutlass.BFloat16,
                        ),
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
                    cutlass.inttoptr(
                        tinv0_base
                        + (
                            store_row * cfg.b_t
                            + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)
                        )
                        * bpe,
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    ),
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
                # ---- Beta column-scaling + publish, stage 1 --------------------------
                beta_col = []
                for k in cutlass.range_constexpr(num_vals):
                    beta_col.append(
                        sBeta[(lane_id % 4) * 2 + ((k // 4) * 8 + k % 2), 0, beta1_idx]
                    )
                tinv_frags = []
                for c in cutlass.range_constexpr(ACC_N_FRAGS):
                    tinv_frags += list(
                        nvvm.ldmatrix(
                            cutlass.inttoptr(
                                tinv1_base
                                + (
                                    store_row * cfg.b_t
                                    + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)
                                )
                                * bpe,
                                cutlass.AddressSpace.smem,
                                cutlass.BFloat16,
                            ),
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
                        cutlass.inttoptr(
                            tinv1_base
                            + (
                                store_row * cfg.b_t
                                + swizzle_xor_128b(store_row, store_col + c * FRAG_COLS)
                            )
                            * bpe,
                            cutlass.AddressSpace.smem,
                            cutlass.BFloat16,
                        ),
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

            # ---- A_epi (opposite member, both halves in parallel) ----------------
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
                a_base = smem_data_ptr(sA[my_a_idx].base).toint()
                a_vec0 = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(
                        (tmem_warp_row << 16) + tmem_cg0_acc_col + a_acc_idx * ACC_STAGE_COLS,
                        cutlass.Float32,
                    ),
                    num=8,
                )
                a_vec1 = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(
                        ((tmem_warp_row + 16) << 16)
                        + tmem_cg0_acc_col
                        + a_acc_idx * ACC_STAGE_COLS,
                        cutlass.Float32,
                    ),
                    num=8,
                )
                nvvm.tcgen05_wait("load")
                bars.mb_cg0_acc_done[a_acc_idx].arrive()
                bars.mb_a_done[my_a_idx].wait(my_a_phase)
                for u in cutlass.range_constexpr(2):
                    a_vec = a_vec1 if cutlass.const_expr(u == 1) else a_vec0
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
                            cutlass.inttoptr(
                                a_base
                                + (
                                    st_row * cfg.b_t
                                    + swizzle_xor_128b(st_row, store_col + c * FRAG_COLS)
                                )
                                * bpe,
                                cutlass.AddressSpace.smem,
                                cutlass.BFloat16,
                            ),
                            [
                                a_pack[c * 4 + 0],
                                a_pack[c * 4 + 1],
                                a_pack[c * 4 + 2],
                                a_pack[c * 4 + 3],
                            ],
                            nvvm.MMALayout.ROW,
                        )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_a_ready[my_a_idx].arrive()

        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)
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
    sV,
    sCumsumlog,
    sCumprod,
    sBeta,
    sO,
    sCheckpoint_raw,
    mState_init,
    mState_out,
    mStateIndices,
    mHasInitialState,
    checkpoint_every_n_tokens,
    sSched,
    bars,
):
    """Compute warp-group 1 role (warps 4-7): persistent scheduler loop
    running the per-chunk state-update and output epilogues.

    With ``cfg.paged_state`` the state tensors are one shared pool and
    ``mStateIndices[batch_idx]`` selects the slot. Non-positive routes are
    null: they zero the output tile and never touch the pool. A false
    ``mHasInitialState[batch_idx]`` seeds a fresh slot from zero instead of
    the pool contents."""
    elect_one = nvvm.elect_sync()

    v_index = PipelineState.start(phase=0)
    gate_index = PipelineState.start(phase=0)
    kv_acc_index = PipelineState.start(phase=0)
    o_acc_ready_index = PipelineState.start(phase=0)
    o_final_acc_ready_index = PipelineState.start(phase=0)
    k_state_ready_index = PipelineState.start(phase=0)
    u_acc_ready_index = PipelineState.start(phase=0)
    state_acc_seed_index = PipelineState.start(phase=1)
    o_index = PipelineState.start(phase=1)
    state_inp_cnt = cutlass.Int32(0)
    kv_done_idx = cutlass.Int32(0)

    nvvm.setmaxregister(cfg.num_regs_compute_group_1, nvvm.SetMaxRegisterAction.INCREASE)
    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_alloc_barrier_id,
        thread_count=cfg.tmem_alloc_barrier_threads,
    )
    tmem_base = tmem_base_slot.load()

    num_threads_cg1 = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    cg1_tidx = tidx % num_threads_cg1
    lane_id = cg1_tidx % cfg.threads_per_warp
    tmem_warp_row = (cg1_tidx // cfg.threads_per_warp) * cfg.threads_per_warp
    ldtm_width = 32
    sttm_width = ldtm_width // 2
    num_state_subs = cutlass.const_expr(cfg.d_v // ldtm_width)
    tmem_state_col = tmem_base + cfg.tmem_state_acc_offset
    tmem_state_inp_col = tmem_base + cfg.tmem_state_inp_offset
    tmem_q_state_col = tmem_base + cfg.tmem_q_state_acc_offset
    tmem_inp_col = tmem_base + cfg.tmem_y_decay_u_inp_offset
    ACC_STAGE_COLS = cfg.b_t
    INP_SLOT_COLS = cfg.b_t // 2
    tmem_k_state_col = tmem_base + cfg.tmem_cg1_acc_offset
    tmem_u_acc_col = tmem_k_state_col
    tmem_y_inp_col = tmem_inp_col
    tmem_u_inp_col = tmem_inp_col
    tmem_decay_v_col = tmem_inp_col + INP_SLOT_COLS
    v_o_smem_tok = cg1_tidx % 8 + (cg1_tidx // 16 % 2) * 8
    v_o_smem_col = (cg1_tidx // 8 % 2) * 8 + (cg1_tidx // 32 % 2) * 32
    v_o_smem_subtile_off = (cg1_tidx // 64) * 4096
    v_stage_elems = cfg.v_cosize // cfg.smem_v_stages
    o_stage_elems = cfg.o_cosize // cfg.smem_o_stages
    sV_base = cute.make_ptr(
        cfg.io_dtype,
        smem_data_ptr(sV[0].base).toint(),
        mem_space=cute.AddressSpace.smem,
        assumed_align=cfg.buffer_align_bytes,
    )
    sO_base = cute.make_ptr(
        cfg.io_dtype,
        smem_data_ptr(sO[0].base).toint(),
        mem_space=cute.AddressSpace.smem,
        assumed_align=cfg.buffer_align_bytes,
    )
    num_vals = 32
    if cutlass.const_expr(cfg.enable_checkpoints):
        sCheckpoint_base_int = smem_data_ptr(sCheckpoint_raw).toint()
        checkpoint_cnt = cutlass.Int32(0)
        checkpoint_smem_row = cg1_tidx % 8 + (cg1_tidx // 16 % 2) * 8
        checkpoint_smem_col = (cg1_tidx // 8 % 2) * 8 + (cg1_tidx // 32 % 2) * 32

    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        (
            batch_idx,
            head_idx,
            batch_start,
            batch_end,
            seqlen_b,
            num_chunks_b,
            wstart,
            wend,
            cstart,
            cend,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        n_local = wend - cstart
        if cutlass.const_expr(cfg.enable_checkpoints):
            ckpt_chunks = checkpoint_every_n_tokens // cutlass.Int32(cfg.b_t)
            checkpoint_mod = cstart % ckpt_chunks
        state_slot, state_valid, seed_from_pool, clear_empty = resolve_paged_state(
            batch_idx, mStateIndices, mHasInitialState
        )
        if n_local > 0:
            if cutlass.const_expr(cfg.use_initial_state):
                # ---- initial-state seed: initial_state GMEM -> state TMEM ---------------
                gState_init = mState_init[None, None, head_idx, state_slot]
                kv_init_idx = state_acc_seed_index.idx
                bars.mb_state_acc_scale_done[kv_init_idx].wait(state_acc_seed_index.phase)
                state_acc_seed_index = advance(state_acc_seed_index, cfg.tmem_state_acc_stages)
                seed_from_initial_state = cstart == 0
                if cutlass.const_expr(cfg.paged_state):
                    seed_from_initial_state = seed_from_initial_state and seed_from_pool
                if seed_from_initial_state:
                    for sub in cutlass.range_constexpr(num_state_subs):
                        words = []
                        for k in cutlass.range_constexpr(32):
                            v = gState_init[cg1_tidx, sub * ldtm_width + k]
                            if cutlass.const_expr(cfg.state_dtype != cfg.acc_dtype):
                                v = v.to(cfg.acc_dtype)
                            words.append(v)
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                (tmem_warp_row << 16) + tmem_state_col + sub * ldtm_width,
                                cutlass.Float32,
                            ),
                            cutlass.Vector.from_elements(tuple(words), cutlass.Float32),
                        )
                else:
                    for sub in cutlass.range_constexpr(num_state_subs):
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                (tmem_warp_row << 16) + tmem_state_col + sub * ldtm_width,
                                cutlass.Float32,
                            ),
                            cutlass.Vector.from_elements(
                                tuple(cutlass.Float32(0.0) for _ in range(32)), cutlass.Float32
                            ),
                        )
                nvvm.tcgen05_wait("store")

                nvvm.barrier_cta_sync_aligned(
                    cfg.init_state_store_barrier_id,
                    thread_count=cfg.init_state_store_barrier_threads,
                )

            for local_idx in cutlass.range(n_local):  # noqa: B007
                chunk_idx = cstart + local_idx
                if cutlass.const_expr(cfg.enable_checkpoints):
                    do_checkpoint_now = checkpoint_mod == 0
                    checkpoint_mod = checkpoint_mod + cutlass.Int32(1)
                    checkpoint_mod = (
                        cutlass.Int32(0) if checkpoint_mod == ckpt_chunks else checkpoint_mod
                    )
                if cutlass.const_expr(cfg.enable_checkpoints and not cfg.use_initial_state):
                    if chunk_idx == 0 and wstart == 0:
                        checkpoint_stage = checkpoint_cnt % cfg.smem_checkpoint_stages
                        checkpoint_phase_done = cutlass.Int32(1) ^ (
                            (checkpoint_cnt // cfg.smem_checkpoint_stages) & cutlass.Int32(1)
                        )
                        bars.mb_checkpoint_tmastg_done[checkpoint_stage].wait(
                            checkpoint_phase_done
                        )
                        checkpoint_zero_ptr = cutlass.inttoptr(
                            sCheckpoint_base_int + checkpoint_stage * cfg.d_k * cfg.d_v * 2,
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
                if cutlass.const_expr(cfg.use_initial_state):
                    valid_state = cutlass.Boolean(True)
                    state_acc_seed_index = advance(state_acc_seed_index, cfg.tmem_state_acc_stages)

                gate_idx = gate_index.idx
                bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
                gate_index = advance(gate_index, cfg.smem_gate_stages)
                cumprod_total = sCumprod[sCumprod.shape[0] - 1, 0, gate_idx]

                # ---- state restage + rescale -------------------------------------
                if valid_state:
                    kv_idx = kv_acc_index.idx
                    # CG1 writes the seeded state itself; only subsequent states
                    # arrive through the MMA-owned state accumulator barrier.
                    if local_idx > 0:
                        bars.mb_state_acc_ready[kv_idx].wait(kv_acc_index.phase)
                        kv_acc_index = advance(kv_acc_index, cfg.tmem_state_acc_stages)
                    kv_done_idx = kv_idx

                    state_regs = [
                        [cutlass.Float32(0.0) for _ in range(num_state_subs)] for _ in range(32)
                    ]
                    state_inp_stage_idx = state_inp_cnt % cfg.tmem_state_inp_stages
                    state_vecs = []
                    for sub in cutlass.range_constexpr(num_state_subs):
                        state_vecs.append(
                            nvvm.tcgen05_ld(
                                "32x32b",
                                nvvm.make_tmem_ptr(
                                    (tmem_warp_row << 16) + tmem_state_col + sub * ldtm_width,
                                    cutlass.Float32,
                                ),
                                num=32,
                            )
                        )
                    for sub in cutlass.range_constexpr(num_state_subs):
                        for k in cutlass.range_constexpr(32):
                            state_regs[k][sub] = state_vecs[sub][k]
                        state_pack = [
                            fp32_to_fp16(
                                state_regs[2 * j][sub],
                                state_regs[2 * j + 1][sub],
                                dtype=cfg.io_dtype,
                            )
                            for j in range(16)
                        ]
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                (tmem_warp_row << 16) + tmem_state_inp_col + sub * sttm_width,
                                cutlass.Int32,
                            ),
                            cutlass.Vector.from_elements(tuple(state_pack), cutlass.Int32),
                        )
                    nvvm.tcgen05_wait("store")
                    bars.mb_state_inp_ready[state_inp_stage_idx].arrive()
                    state_inp_cnt = state_inp_cnt + 1

                    if cutlass.const_expr(cfg.enable_checkpoints):
                        # ---- state checkpoint ----------------------------------------
                        do_checkpoint = do_checkpoint_now and chunk_idx < wend
                        do_checkpoint = do_checkpoint and chunk_idx >= wstart
                        if do_checkpoint:
                            checkpoint_stage = checkpoint_cnt % cfg.smem_checkpoint_stages
                            checkpoint_phase_done = cutlass.Int32(1) ^ (
                                (checkpoint_cnt // cfg.smem_checkpoint_stages) & cutlass.Int32(1)
                            )
                            bars.mb_checkpoint_tmastg_done[checkpoint_stage].wait(
                                checkpoint_phase_done
                            )
                            checkpoint_stage_base = checkpoint_stage * cfg.d_k * cfg.d_v
                            for sub in cutlass.range_constexpr(num_state_subs):
                                checkpoint_vec = nvvm.tcgen05_ld(
                                    "32x32b",
                                    nvvm.make_tmem_ptr(
                                        (tmem_warp_row << 16) + tmem_state_col + sub * ldtm_width,
                                        cutlass.Float32,
                                    ),
                                    num=32,
                                )
                                for g in cutlass.range_constexpr(ldtm_width // 8):
                                    packs = tuple(
                                        fp32_to_fp16(
                                            checkpoint_vec[g * 8 + 2 * t],
                                            checkpoint_vec[g * 8 + 2 * t + 1],
                                            dtype=cfg.io_dtype,
                                        )
                                        for t in range(4)
                                    )
                                    dk = sub * ldtm_width + g * 8
                                    checkpoint_addr = (
                                        checkpoint_stage_base
                                        + (dk // 64) * (cfg.d_v * 64)
                                        + cg1_tidx * 64
                                        + swizzle_xor_128b(cg1_tidx, dk % 64)
                                    )
                                    (smem_data_ptr(sCheckpoint_raw) + checkpoint_addr).store(
                                        cutlass.Vector.from_elements(packs, cutlass.Int32).bitcast(
                                            cfg.io_dtype
                                        ),
                                        alignment=16,
                                    )
                            nvvm.fence_proxy("async.shared", space="cta")
                            if elect_one:
                                bars.mb_checkpoint_tmastg_ready[checkpoint_stage].arrive()
                            checkpoint_cnt = checkpoint_cnt + 1

                    for sub in cutlass.range_constexpr(num_state_subs):
                        state_scaled = []
                        for j in cutlass.range_constexpr(16):
                            s0, s1 = fmul2(
                                state_regs[2 * j][sub],
                                state_regs[2 * j + 1][sub],
                                cumprod_total,
                                cumprod_total,
                            )
                            state_scaled += [s0, s1]
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                (tmem_warp_row << 16) + tmem_state_col + sub * ldtm_width,
                                cutlass.Float32,
                            ),
                            cutlass.Vector.from_elements(tuple(state_scaled), cutlass.Float32),
                        )
                    nvvm.tcgen05_wait("store")
                    bars.mb_state_acc_scale_done[kv_idx].arrive()

                # ---- per-row Gate register builds --------------------------------
                cumprod_vals = []
                for k in cutlass.range_constexpr(num_vals):
                    cumprod_vals.append(
                        sCumprod[(lane_id % 4) * 2 + ((k // 4) * 8 + k % 2), 0, gate_idx]
                    )
                last_cumsumlog = sCumsumlog[cfg.b_t - 1, 0, gate_idx]
                cumsumlog_vals = []
                for k in cutlass.range_constexpr(num_vals):
                    cumsumlog_vals.append(
                        sCumsumlog[(lane_id % 4) * 2 + ((k // 4) * 8 + k % 2), 0, gate_idx]
                    )
                decay_scale_vals = []
                for k in cutlass.range_constexpr(0, num_vals, 2):
                    d0, d1 = fadd2(
                        last_cumsumlog, last_cumsumlog, -cumsumlog_vals[k], -cumsumlog_vals[k + 1]
                    )
                    decay_scale_vals.append(cute.math.exp2(d0, fastmath=True))
                    decay_scale_vals.append(cute.math.exp2(d1, fastmath=True))
                bars.mb_gate_done[gate_idx].arrive()

                # ---- Y = V - K*state (packed 16-bit) -----------------------------
                v_idx = v_index.idx
                bars.mb_v_ready[v_idx].wait(v_index.phase)
                v_index = advance(v_index, cfg.smem_v_stages)

                v_frags = [[cutlass.Int32(0), cutlass.Int32(0)] for _ in range(16)]
                for c in cutlass.range_constexpr(8):
                    m0 = cutlass.const_expr(c % 4)
                    sub = cutlass.const_expr(c // 4)
                    v_frag = nvvm.ldmatrix(
                        (
                            sV_base
                            + v_idx * v_stage_elems
                            + v_o_smem_subtile_off
                            + (v_o_smem_tok + m0 * 16) * 64
                            + swizzle_xor_128b(v_o_smem_tok + m0 * 16, v_o_smem_col + sub * 16)
                        ).raw_ptr(),
                        4,
                        nvvm.MMALayout.COL,
                    )
                    for i in cutlass.range_constexpr(4):
                        v_frags[4 * m0 + i][sub] = v_frag[i]
                if valid_state:
                    bars.mb_k_state_acc_ready[0].wait(k_state_ready_index.phase)
                    k_state_ready_index = advance(k_state_ready_index, 1)

                    for sub in cutlass.range_constexpr(2):
                        k_state_vec = nvvm.tcgen05_ld(
                            "16x256b",
                            nvvm.make_tmem_ptr(
                                ((tmem_warp_row + sub * 16) << 16) + tmem_k_state_col,
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
                            v_frags[j][sub] = sub_f16x2(
                                v_frags[j][sub], k_state_word, cfg.io_dtype
                            )
                for sub in cutlass.range_constexpr(2):
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(
                            ((tmem_warp_row + sub * 16) << 16) + tmem_y_inp_col, cutlass.Int32
                        ),
                        cutlass.Vector.from_elements(
                            tuple(v_frags[j][sub] for j in range(16)), cutlass.Int32
                        ),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_y_inp_ready[0].arrive()

                # ---- state*Q_epi: Q*state *= cumprod * scale ----------------
                if valid_state:
                    q_state_idx = o_acc_ready_index.idx
                    bars.mb_o_acc_ready[q_state_idx].wait(o_acc_ready_index.phase)
                    o_acc_ready_index = advance(o_acc_ready_index, cfg.tmem_q_state_acc_stages)

                    q_state_ptrs = []
                    q_state_vecs = []
                    for sub in cutlass.range_constexpr(2):
                        q_state_ptrs.append(
                            nvvm.make_tmem_ptr(
                                ((tmem_warp_row + sub * 16) << 16)
                                + tmem_q_state_col
                                + q_state_idx * ACC_STAGE_COLS,
                                cutlass.Float32,
                            )
                        )
                        q_state_vecs.append(nvvm.tcgen05_ld("16x256b", q_state_ptrs[sub], num=8))
                    for sub in cutlass.range_constexpr(2):
                        q_state_scaled = []
                        for j in cutlass.range_constexpr(16):
                            p0, p1 = fmul2(
                                q_state_vecs[sub][2 * j],
                                q_state_vecs[sub][2 * j + 1],
                                cumprod_vals[2 * j],
                                cumprod_vals[2 * j + 1],
                            )
                            s0, s1 = fmul2(p0, p1, scale, scale)
                            q_state_scaled += [s0, s1]
                        nvvm.tcgen05_st(
                            "16x256b",
                            q_state_ptrs[sub],
                            cutlass.Vector.from_elements(tuple(q_state_scaled), cutlass.Float32),
                        )
                    nvvm.tcgen05_wait("store")

                    bars.mb_o_state_scale_acc_done[q_state_idx].arrive()

                # ---- U_epi + decayed-U publish -----------------------------------
                bars.mb_u_acc_ready[0].wait(u_acc_ready_index.phase)
                u_acc_ready_index = advance(u_acc_ready_index, 1)
                bars.mb_v_done[v_idx].arrive()

                u_regs = [[cutlass.Float32(0.0), cutlass.Float32(0.0)] for _ in range(32)]
                u_vecs = []
                for sub in cutlass.range_constexpr(2):
                    u_vecs.append(
                        nvvm.tcgen05_ld(
                            "16x256b",
                            nvvm.make_tmem_ptr(
                                ((tmem_warp_row + sub * 16) << 16) + tmem_u_acc_col,
                                cutlass.Float32,
                            ),
                            num=8,
                        )
                    )
                for sub in cutlass.range_constexpr(2):
                    for k in cutlass.range_constexpr(32):
                        u_regs[k][sub] = u_vecs[sub][k]

                    u_pack = [
                        fp32_to_fp16(
                            u_regs[2 * j][sub], u_regs[2 * j + 1][sub], dtype=cfg.io_dtype
                        )
                        for j in range(16)
                    ]
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(
                            ((tmem_warp_row + sub * 16) << 16) + tmem_u_inp_col, cutlass.Int32
                        ),
                        cutlass.Vector.from_elements(tuple(u_pack), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_u_inp_ready[0].arrive()

                for sub in cutlass.range_constexpr(2):
                    for j in cutlass.range_constexpr(16):
                        u_regs[2 * j][sub], u_regs[2 * j + 1][sub] = fmul2(
                            u_regs[2 * j][sub],
                            u_regs[2 * j + 1][sub],
                            decay_scale_vals[2 * j],
                            decay_scale_vals[2 * j + 1],
                        )
                    decay_pack = [
                        fp32_to_fp16(
                            u_regs[2 * j][sub], u_regs[2 * j + 1][sub], dtype=cfg.io_dtype
                        )
                        for j in range(16)
                    ]
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(
                            ((tmem_warp_row + sub * 16) << 16) + tmem_decay_v_col, cutlass.Int32
                        ),
                        cutlass.Vector.from_elements(tuple(decay_pack), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_decay_u_inp_ready[0].arrive()

                # ---- QKV_epilogue: O acc TMEM -> sO SMEM -------------------------
                o_scale_idx = o_final_acc_ready_index.idx
                bars.mb_o_final_acc_ready[o_scale_idx].wait(o_final_acc_ready_index.phase)
                o_final_acc_ready_index = advance(
                    o_final_acc_ready_index, cfg.tmem_q_state_acc_stages
                )

                o_regs = []
                for sub in cutlass.range_constexpr(2):
                    o_vec = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(
                            ((tmem_warp_row + sub * 16) << 16)
                            + tmem_q_state_col
                            + o_scale_idx * ACC_STAGE_COLS,
                            cutlass.Float32,
                        ),
                        num=8,
                    )
                    o_regs.append([o_vec[k] for k in range(32)])
                nvvm.tcgen05_wait("load")
                if cutlass.const_expr(cfg.paged_state):
                    # Selection, rather than multiplication, keeps null routes zero
                    # even if their padding tokens contain NaNs.
                    o_regs = [
                        [val if state_valid else cutlass.Float32(0.0) for val in regs]
                        for regs in o_regs
                    ]
                o_idx = o_index.idx
                bars.mb_o_tmastg_done[o_idx].wait(o_index.phase)
                o_index = advance(o_index, cfg.smem_o_stages)
                for sub in cutlass.range_constexpr(2):
                    for m0 in cutlass.range_constexpr(4):
                        o_pack = [
                            fp32_to_fp16(
                                o_regs[sub][8 * m0 + 2 * j],
                                o_regs[sub][8 * m0 + 2 * j + 1],
                                dtype=cfg.io_dtype,
                            )
                            for j in range(4)
                        ]
                        nvvm.stmatrix(
                            (
                                sO_base
                                + o_idx * o_stage_elems
                                + v_o_smem_subtile_off
                                + (v_o_smem_tok + m0 * 16) * 64
                                + swizzle_xor_128b(v_o_smem_tok + m0 * 16, v_o_smem_col + sub * 16)
                            ).raw_ptr(),
                            o_pack,
                            nvvm.MMALayout.COL,
                        )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_o_tmastg_ready[o_idx].arrive()

        # ---- final state: state TMEM -> GMEM -----------------------------------
        if n_local > 0:
            kv_last_idx = kv_acc_index.idx
            bars.mb_state_acc_ready[kv_last_idx].wait(kv_acc_index.phase)
            kv_acc_index = advance(kv_acc_index, cfg.tmem_state_acc_stages)
            if cutlass.const_expr(cfg.store_final_state):
                store_final = wend == num_chunks_b
                if cutlass.const_expr(cfg.paged_state):
                    store_final = store_final and state_valid
                if store_final:
                    gState_out = mState_out[None, None, head_idx, state_slot]
                    for sub in cutlass.range_constexpr(num_state_subs):
                        state_vec = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                (tmem_warp_row << 16) + tmem_state_col + sub * ldtm_width,
                                cutlass.Float32,
                            ),
                            num=32,
                        )
                        for k in cutlass.range_constexpr(32):
                            val = state_vec[k]
                            if cutlass.const_expr(cfg.state_dtype != cfg.acc_dtype):
                                val = val.to(cfg.state_dtype)
                            gState_out[cg1_tidx, sub * ldtm_width + k] = val
                bars.mb_state_acc_scale_done[kv_last_idx].arrive()
            else:
                bars.mb_state_acc_scale_done[kv_last_idx].arrive()
        else:
            if cutlass.const_expr(cfg.store_final_state):
                write_passthrough = wend == num_chunks_b
                if cutlass.const_expr(cfg.paged_state):
                    # An empty sequence on a fresh slot clears it; a resumed slot is
                    # already in place and a null route is left untouched.
                    write_passthrough = write_passthrough and clear_empty
                if write_passthrough:
                    gState_out = mState_out[None, None, head_idx, state_slot]
                    if cutlass.const_expr(cfg.use_initial_state and not cfg.paged_state):
                        gState_in = mState_init[None, None, head_idx, state_slot]
                        for sub in cutlass.range_constexpr(num_state_subs):
                            for k in cutlass.range_constexpr(32):
                                gState_out[cg1_tidx, sub * ldtm_width + k] = gState_in[
                                    cg1_tidx, sub * ldtm_width + k
                                ]
                    else:
                        for sub in cutlass.range_constexpr(num_state_subs):
                            for k in cutlass.range_constexpr(32):
                                gState_out[cg1_tidx, sub * ldtm_width + k] = cutlass.Float32(
                                    0.0
                                ).to(cfg.state_dtype)

        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)

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
    desc_ws: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: Optional[cute.Tensor],
    state_checkpoints_out: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    q_row_stride: cutlass.Int64,
    k_row_stride: cutlass.Int64,
    v_row_stride: cutlass.Int64,
    o_row_stride: cutlass.Int64,
    checkpoint_row_stride: cutlass.Int64,
    checkpoint_every_n: cutlass.Int32,
) -> None:
    """Per-batch descriptor-array build, one warp per array. Runs inside the
    prologue kernel after its order pass; warps past the array count fall
    through the widx guards."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    sub0 = cute.make_tensor(desc_ws.iterator, cute.make_layout((arr_words,), stride=(1,)))
    sub1 = cute.make_tensor(
        desc_ws.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    sub2 = cute.make_tensor(
        desc_ws.iterator + 2 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    sub3 = cute.make_tensor(
        desc_ws.iterator + 3 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    sub4 = cute.make_tensor(
        desc_ws.iterator + 4 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )

    if widx == 0:
        if nvvm.elect_sync():
            emit_seq_load_descs(base_q, sub0, cu_seqlens, q, n_batch, q_row_stride, 2)
            nvvm.fence_proxy_release(
                nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
            )
    if widx == 1:
        if nvvm.elect_sync():
            emit_seq_load_descs(base_k, sub1, cu_seqlens, k, n_batch, k_row_stride, 2)
            nvvm.fence_proxy_release(
                nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
            )
    if widx == 2:
        if nvvm.elect_sync():
            emit_seq_load_descs(base_v, sub2, cu_seqlens, v, n_batch, v_row_stride, 2)
            nvvm.fence_proxy_release(
                nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
            )
    if cutlass.const_expr(o is not None):
        if widx == 3:
            if nvvm.elect_sync():
                emit_seq_descs(base_o, sub3, cu_seqlens, o, n_batch, o_row_stride, 2)
                nvvm.fence_proxy_release(
                    nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
                )
    if cutlass.const_expr(state_checkpoints_out is not None):
        if widx == 4:
            if nvvm.elect_sync():
                emit_checkpoint_seq_descs(
                    base_checkpoint,
                    sub4,
                    cu_seqlens,
                    state_checkpoints_out,
                    n_batch,
                    checkpoint_row_stride,
                    checkpoint_every_n,
                    2,
                )
                nvvm.fence_proxy_release(
                    nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
                )


@cute.kernel
def prologue_kernel(
    order_gen: cutlass.Constexpr[bool],
    has_sched: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    base_q: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_k: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_v: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_o: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_checkpoint: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    desc_ws: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: Optional[cute.Tensor],
    state_checkpoints_out: Optional[cute.Tensor],
    mStaging: Optional[cute.Tensor],
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mSched: Optional[cute.Tensor],
    n_batch: cutlass.Int32,
    q_row_stride: cutlass.Int64,
    k_row_stride: cutlass.Int64,
    v_row_stride: cutlass.Int64,
    o_row_stride: cutlass.Int64,
    checkpoint_row_stride: cutlass.Int64,
    checkpoint_every_n: cutlass.Int32,
) -> None:
    """Single-CTA prologue: LPT-order the work-item table and zero the sched
    rings via :func:`order_body`, then build the per-batch TMA-descriptor
    arrays via :func:`build_descs_body`, one warp per array (the extra warps
    only take part in the order phase)."""
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    sKey = cutlass.Array(
        cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16
    )
    sIdx = cutlass.Array(
        cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16
    )
    sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
    n_heads_out = cutlass.Int32(q.shape[1] if q.shape[1] >= v.shape[1] else v.shape[1])
    order_body(
        order_gen,
        has_sched,
        b_t,
        ORDER_THREADS,
        ORDER_ELEMS,
        tidx,
        n_heads_out,
        n_heads_out * n_batch,
        cu_seqlens,
        mStaging,
        mCount,
        mWorkItems,
        mSched,
        sKey,
        sIdx,
        sSpread,
    )
    build_descs_body(
        widx,
        base_q,
        base_k,
        base_v,
        base_o,
        base_checkpoint,
        desc_ws,
        cu_seqlens,
        q,
        k,
        v,
        o,
        state_checkpoints_out,
        n_batch,
        q_row_stride,
        k_row_stride,
        v_row_stride,
        o_row_stride,
        checkpoint_row_stride,
        checkpoint_every_n,
    )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    order_gen: cutlass.Constexpr[bool],
    has_sched: cutlass.Constexpr[bool],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    o: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    state_checkpoints_out: Optional[cute.Tensor],
    work_item_staging: Optional[cute.Tensor],
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    sched_ctr: Optional[cute.Tensor],
    checkpoint_every_n: cutlass.Int32,
    tensormap_workspace: cute.Tensor,
    stream: cuda_driver.CUstream,
):
    """One-launch prologue: LPT-order the work items and build the 5
    per-(b,h) TMA-descriptor arrays (Q, K, V, O, checkpoints) into
    ``tensormap_workspace``."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    h_v = v.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = h_q if h_q >= h_v else h_v
    d_v = v.shape[2]
    bpe = io_dtype.width // 8
    elems_per_128b = 128 // bpe
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
    swz128 = cuda.TensorMapSwizzle.s128b
    base_desc_q = cuda.create_tensor_map_tiled_from_view(
        q_headed, box_dims=(bt, 1, elems_per_128b), stride_order=(2, 1, 0), swizzle=swz128
    )
    base_desc_k = cuda.create_tensor_map_tiled_from_view(
        k_headed, box_dims=(bt, 1, elems_per_128b), stride_order=(2, 1, 0), swizzle=swz128
    )
    base_desc_v = cuda.create_tensor_map_tiled_from_view(
        v_headed, box_dims=(elems_per_128b, 1, bt), stride_order=(0, 1, 2), swizzle=swz128
    )

    base_desc_o = base_desc_v
    if cutlass.const_expr(o is not None):
        o_headed = cute.make_tensor(
            o.iterator,
            cute.make_layout((d_v, heads_out, seqlen), stride=(1, o.stride[1], o.stride[0])),
        )
        base_desc_o = cuda.create_tensor_map_tiled_from_view(
            o_headed, box_dims=(elems_per_128b, 1, bt), stride_order=(0, 1, 2), swizzle=swz128
        )

    base_desc_checkpoint = base_desc_v
    if cutlass.const_expr(state_checkpoints_out is not None):
        d_k_state = state_checkpoints_out.shape[2]
        d_v_state = state_checkpoints_out.shape[3]
        checkpoint_elems_per_128b = 128 // (state_checkpoints_out.element_type.width // 8)
        checkpoint_view = cute.make_tensor(
            state_checkpoints_out.iterator,
            cute.make_layout(
                (d_v_state, d_k_state, state_checkpoints_out.shape[0], heads_out),
                stride=(
                    state_checkpoints_out.stride[3],
                    state_checkpoints_out.stride[2],
                    state_checkpoints_out.stride[0],
                    state_checkpoints_out.stride[1],
                ),
            ),
        )
        base_desc_checkpoint = cuda.create_tensor_map_tiled_from_view(
            checkpoint_view,
            box_dims=(checkpoint_elems_per_128b, d_k_state, 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=swz128,
        )

    prologue_kernel(
        order_gen,
        has_sched,
        b_t,
        base_desc_q,
        base_desc_k,
        base_desc_v,
        base_desc_o,
        base_desc_checkpoint,
        tensormap_workspace,
        cu_seqlens,
        q,
        k,
        v,
        o,
        state_checkpoints_out,
        work_item_staging,
        work_count,
        work_items,
        sched_ctr,
        cutlass.Int32(batch_size),
        cutlass.Int64(q_row_stride),
        cutlass.Int64(k_row_stride),
        cutlass.Int64(v_row_stride),
        cutlass.Int64(o.stride[0] if o is not None else 0),
        cutlass.Int64(state_checkpoints_out.stride[0] if state_checkpoints_out is not None else 0),
        checkpoint_every_n,
    ).launch(grid=(1, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream)


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    a_log: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    beta: cute.Tensor,
    o: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    state_in: Optional[cute.Tensor],
    state_out: Optional[cute.Tensor],
    state_indices: Optional[cute.Tensor],
    has_initial_state: Optional[cute.Tensor],
    work_items: Optional[cute.Tensor],
    work_count: Optional[cute.Tensor],
    sched_ctr: Optional[cute.Tensor],
    checkpoint_every_n_tokens: cutlass.Int32,
    scale: cutlass.Float32,
    tensormap_workspace: cute.Tensor,
    stream: cuda_driver.CUstream,
):
    h_q = q.shape[1]
    h_k = k.shape[1]
    h_v = v.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    heads_out = h_q if h_q >= h_v else h_v

    # ---- GQA reshapes: fold the head group into a --------------------------------
    if cutlass.const_expr(cfg.is_GQA):
        h_r = h_q // h_v
        h_qv = h_v
        q = cute.make_tensor(
            q.iterator,
            cute.make_layout(
                (q.shape[0], q.shape[2], (h_r, h_v)),
                stride=(q.stride[0], q.stride[2], (q.stride[1], h_r * q.stride[1])),
            ),
        )
        k = cute.make_tensor(
            k.iterator,
            cute.make_layout(
                (k.shape[0], k.shape[2], (h_r, h_v)),
                stride=(k.stride[0], k.stride[2], (0, k.stride[1])),
            ),
        )
        v = cute.make_tensor(
            v.iterator,
            cute.make_layout(
                (v.shape[2], v.shape[0], (h_r, h_v)),
                stride=(v.stride[2], v.stride[0], (0, v.stride[1])),
            ),
        )
    else:
        h_r = h_v // h_q
        h_qv = h_q
        q = cute.make_tensor(
            q.iterator,
            cute.make_layout(
                (q.shape[0], q.shape[2], (h_r, h_q)),
                stride=(q.stride[0], q.stride[2], (0, q.stride[1])),
            ),
        )
        k = cute.make_tensor(
            k.iterator,
            cute.make_layout(
                (k.shape[0], k.shape[2], (h_r, h_q)),
                stride=(k.stride[0], k.stride[2], (0, k.stride[1])),
            ),
        )
        v = cute.make_tensor(
            v.iterator,
            cute.make_layout(
                (v.shape[2], v.shape[0], (h_r, h_q)),
                stride=(v.stride[2], v.stride[0], (v.stride[1], h_r * v.stride[1])),
            ),
        )

    gate = cute.make_tensor(
        gate.iterator,
        cute.make_layout(
            (gate.shape[0], (h_r, h_qv)),
            stride=(gate.stride[0], (gate.stride[1], h_r * gate.stride[1])),
        ),
    )
    beta = cute.make_tensor(
        beta.iterator,
        cute.make_layout(
            (beta.shape[0], (h_r, h_qv)),
            stride=(beta.stride[0], (beta.stride[1], h_r * beta.stride[1])),
        ),
    )
    if cutlass.const_expr(o is not None):
        o = cute.make_tensor(
            o.iterator,
            cute.make_layout(
                (o.shape[2], o.shape[0], (h_r, h_qv)),
                stride=(o.stride[2], o.stride[0], (o.stride[1], h_r * o.stride[1])),
            ),
        )
    if cutlass.const_expr(state_in is not None):
        state_in = cute.make_tensor(
            state_in.iterator,
            cute.make_layout(
                (state_in.shape[2], state_in.shape[3], (h_r, h_qv), state_in.shape[0]),
                stride=(
                    state_in.stride[2],
                    state_in.stride[3],
                    (state_in.stride[1], h_r * state_in.stride[1]),
                    state_in.stride[0],
                ),
            ),
        )
    if cutlass.const_expr(state_out is not None):
        state_out = cute.make_tensor(
            state_out.iterator,
            cute.make_layout(
                (state_out.shape[2], state_out.shape[3], (h_r, h_qv), state_out.shape[0]),
                stride=(
                    state_out.stride[2],
                    state_out.stride[3],
                    (state_out.stride[1], h_r * state_out.stride[1]),
                    state_out.stride[0],
                ),
            ),
        )

    # ---- SMEM sizing: per-buffer element cosizes ---------------------------------
    bpe = cfg.io_dtype.width // 8
    kq_tile_elems = 2 * cfg.b_t * cfg.d_k
    v_tile_elems = cfg.d_v * cfg.b_t
    tinv_tile_elems = cfg.b_t * cfg.b_t
    a_tile_elems = cfg.b_t * cfg.b_t
    o_tile_elems = cfg.d_v * cfg.b_t
    cfg.kq_cosize = kq_tile_elems * cfg.smem_kq_stages
    cfg.v_cosize = v_tile_elems * cfg.smem_v_stages
    cfg.t_inv_cosize = tinv_tile_elems * cfg.smem_t_inv_stages
    cfg.a_cosize = a_tile_elems * cfg.smem_a_stages
    cfg.o_cosize = o_tile_elems * cfg.smem_o_stages
    cfg.checkpoint_cosize = (
        cfg.d_k * cfg.d_v * cfg.smem_checkpoint_stages if cfg.enable_checkpoints else 1
    )

    cumsumlog_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_gate_stages))
    beta_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_beta_stages))

    cfg.tma_kq_bytes = kq_tile_elems * bpe
    cfg.tma_v_bytes = v_tile_elems * bpe
    cfg.tma_o_bytes = o_tile_elems * bpe

    cfg.n_heads_out = heads_out
    cfg.q_ratio = heads_out // h_q
    cfg.k_ratio = heads_out // h_k
    cfg.v_ratio = heads_out // h_v
    num_descs = batch_size

    # ---- launch ------------------------------------------------------------------
    # CUDA-graph-stable launch: fixed SM-count grid; shapes ride on buffer contents
    grid_shape = (cfg.max_active_clusters, 1, 1)

    @cute.struct
    class SharedStorage:
        output: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.o_cosize], cfg.buffer_align_bytes
        ]
        checkpoint: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.checkpoint_cosize], cfg.buffer_align_bytes
        ]
        kq: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.kq_cosize], cfg.buffer_align_bytes
        ]
        cumsumlog: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged)], 128
        ]
        cumprod: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged)], 128
        ]
        beta: cute.struct.Align[
            cute.struct.MemRange[cutlass.Float32, cute.cosize(beta_smem_layout_staged)], 128
        ]
        t_inv: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.t_inv_cosize], cfg.buffer_align_bytes
        ]
        a: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.a_cosize], cfg.buffer_align_bytes
        ]
        v: cute.struct.Align[
            cute.struct.MemRange[cfg.io_dtype, cfg.v_cosize], cfg.buffer_align_bytes
        ]

    kernel(
        cfg,
        SharedStorage,
        gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        state_in,
        state_out,
        state_indices,
        has_initial_state,
        work_items,
        work_count,
        sched_ctr,
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
        min_blocks_per_mp=1,
    )


@cute.kernel
def kernel(
    cfg: cutlass.Constexpr,
    shared_type: cutlass.Constexpr,
    mGate: cute.Tensor,
    mA_log: Optional[cute.Tensor],
    mDt_bias: Optional[cute.Tensor],
    mBeta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    mState_init: Optional[cute.Tensor],
    mState_out: Optional[cute.Tensor],
    mStateIndices: Optional[cute.Tensor],
    mHasInitialState: Optional[cute.Tensor],
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mSched: Optional[cute.Tensor],
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
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]

    total_tiles = mCount[0]
    if cutlass.const_expr(cfg.dyn_sched):
        assert mSched is not None, "mSched must be provided if dyn_sched is True"

    if cutlass.const_expr(cfg.use_initial_state):
        assert mState_init is not None, "mState_init must be provided if use_initial_state is True"
    else:
        assert mState_init is None, "mState_init must be None if use_initial_state is False"
    if cutlass.const_expr(cfg.store_final_state):
        assert mState_out is not None, "mState_out must be provided if store_final_state is True"
    else:
        assert mState_out is None, "mState_out must be None if store_final_state is False"
    if cutlass.const_expr(cfg.paged_state):
        assert mStateIndices is not None, "mStateIndices must be provided if paged_state is True"
    else:
        assert mStateIndices is None, "mStateIndices must be None if paged_state is False"
    if cutlass.const_expr(not cfg.paged_state):
        assert mHasInitialState is None, "mHasInitialState requires paged_state"

    desc_base_words = tensormap_workspace.iterator.raw_ptr()
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)
    arr_words = n_desc * desc_qwords
    desc_q_base = desc_base_words
    desc_k_base = desc_base_words + arr_words
    desc_v_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_o_base = desc_base_words + cutlass.Int32(3) * arr_words
    desc_checkpoint_base = desc_base_words + cutlass.Int32(4) * arr_words

    SMEM = cutlass.AddressSpace.smem

    SWZ = 2
    LEAD = 16
    STRIDE = 8 * 128
    KT_LEAD = (cfg.d_v // 2) * 128
    V_LEAD = (cfg.d_v // 2) * 128
    storage = cutlass.utils.SmemAllocator().allocate(shared_type)
    sO_raw = storage.output.get_tensor(cute.make_layout((cfg.o_cosize,)))
    sCheckpoint_raw = storage.checkpoint.get_tensor(cute.make_layout((cfg.checkpoint_cosize,)))
    sKQ_raw = storage.kq.get_tensor(cute.make_layout((cfg.kq_cosize,)))
    cumsumlog_raw = storage.cumsumlog.get_tensor(
        cute.make_layout((cute.cosize(cumsumlog_smem_layout_staged),))
    )
    cumprod_raw = storage.cumprod.get_tensor(
        cute.make_layout((cute.cosize(cumsumlog_smem_layout_staged),))
    )
    beta_raw = storage.beta.get_tensor(cute.make_layout((cute.cosize(beta_smem_layout_staged),)))
    sTinv_raw = storage.t_inv.get_tensor(cute.make_layout((cfg.t_inv_cosize,)))
    sA_raw = storage.a.get_tensor(cute.make_layout((cfg.a_cosize,)))
    sV_raw = storage.v.get_tensor(cute.make_layout((cfg.v_cosize,)))
    sO = SmemTile(
        base=sO_raw,
        elems_per_stage=cfg.o_cosize // cfg.smem_o_stages,
        stages=cfg.smem_o_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sKQ = SmemTile(
        base=sKQ_raw,
        elems_per_stage=cfg.kq_cosize // cfg.smem_kq_stages,
        stages=cfg.smem_kq_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sKQ_trans = SmemTile(
        base=sKQ_raw,
        elems_per_stage=cfg.kq_cosize // cfg.smem_kq_stages,
        stages=cfg.smem_kq_stages,
        leading_byte_offset=2 * KT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    bars = make_gdn_bars(cfg)
    tmem_base_slot = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=16)
    sSched = cutlass.Array(cutlass.Int32, cfg.sched_stages, space=SMEM, alignment=16)
    sTinv = SmemTile(
        base=sTinv_raw,
        elems_per_stage=cfg.t_inv_cosize // cfg.smem_t_inv_stages,
        stages=cfg.smem_t_inv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sA = SmemTile(
        base=sA_raw,
        elems_per_stage=cfg.a_cosize // cfg.smem_a_stages,
        stages=cfg.smem_a_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sV = SmemTile(
        base=sV_raw,
        elems_per_stage=cfg.v_cosize // cfg.smem_v_stages,
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

    # ---- mbarrier init -----------------------------------------------------------
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
            bars.mb_state_acc_scale_done[s].init()
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
        for s in range(cfg.tmem_state_inp_stages):
            bars.mb_state_inp_ready[s].init()
        for b in (bars.mb_y_inp_ready, bars.mb_u_inp_ready, bars.mb_decay_u_inp_ready):
            b[0].init()
        for s in range(cfg.smem_o_stages):
            bars.mb_o_tmastg_ready[s].init()
            bars.mb_o_tmastg_done[s].init()
        for s in range(cfg.smem_checkpoint_stages):
            bars.mb_checkpoint_tmastg_ready[s].init()
            bars.mb_checkpoint_tmastg_done[s].init()
        for s_ in range(cfg.sched_stages):
            bars.mb_sched_ready[s_].init()
            bars.mb_sched_done[s_].init()
        bars.mb_tmem_done[0].init()

    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # ---- warp specialization -----------------------------------------------------

    if (
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
            tidx,
            tmem_base_slot=tmem_base_slot,
            scale=scale,
            sCumsumlog=sCumsumlog,
            sBeta=sBeta,
            sTinv=sTinv,
            sA=sA,
            sCheckpoint_raw=sCheckpoint_raw,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            sSched=sSched,
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
            sV=sV,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sBeta=sBeta,
            sO=sO,
            sCheckpoint_raw=sCheckpoint_raw,
            mState_init=mState_init,
            mState_out=mState_out,
            mStateIndices=mStateIndices,
            mHasInitialState=mHasInitialState,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            sSched=sSched,
            bars=bars,
        )

    elif warp_idx == cfg.load_gate_beta_warp_id:
        gate_beta_warp(
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
            mBeta=mBeta,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sBeta=sBeta,
            sSched=sSched,
            bars=bars,
        )

    elif warp_idx == cfg.mma_warp_id:
        mma_warp(
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
            sSched=sSched,
            bars=bars,
        )

    elif warp_idx == cfg.tma_qkv_warp_id:
        tmaldg_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            n_desc,
            cu_seqlens,
            mWorkItems,
            sKQ_raw=sKQ_raw,
            sV_raw=sV_raw,
            desc_q_base=desc_q_base,
            desc_k_base=desc_k_base,
            desc_v_base=desc_v_base,
            mSched=mSched,
            sSched=sSched,
            bars=bars,
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
            sSched=sSched,
            bars=bars,
        )


@dataclass
class GdnCfg:
    """Per-compile GDN kernel knob, built by ``build_cfg``.

    The per-compile parameters (dtypes, GQA, state flags) are the
    ``cute.compile`` cache keys; the rest is derived from the module-global
    ``CFG`` constants.  ``host`` stamps the shape-derived fields at trace
    time.  Passed ``cfg``-first (a ``cutlass.Constexpr``) into ``host`` /
    ``kernel`` and every warp body.
    """

    io_dtype: Type[cutlass.Numeric]
    acc_dtype: Type[cutlass.Numeric]
    state_dtype: Type[cutlass.Numeric]
    max_active_clusters: int
    is_GQA: bool
    use_initial_state: bool
    store_final_state: bool
    enable_checkpoints: bool
    paged_state: bool = False
    log_gate: bool = False
    safe_gate: bool = False
    beta_sigmoid: bool = False
    dyn_sched: bool = False
    sched_stages: int = CFG.SMEM_SCHED_STAGES

    # ---- fixed constants stamped from CFG by build_cfg ---------------------------
    b_t: int = CFG.B_T
    d_k: int = CFG.D_K
    d_v: int = CFG.D_V
    compute_group_0_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_1_warp_ids: Tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    load_gate_beta_warp_id: int = CFG.LOAD_GATE_BETA_WARP_ID
    tma_qkv_warp_id: int = CFG.TMA_QKV_WARP_ID
    mma_warp_id: int = CFG.MMA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_other: int = CFG.NUM_REGS_OTHER
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0
    cluster_shape_mnk: Tuple[int, int, int] = CFG.CLUSTER_SHAPE_MNK

    # ---- named barrier slots (ids 1-4; 0 is the CTA-wide sync) -------------------
    tmem_alloc_barrier_id: int = 1
    tmem_alloc_barrier_threads: int = 0
    inverse_barrier_id: int = 2
    inverse_barrier_threads: int = 0
    init_state_store_barrier_id: int = 4
    init_state_store_barrier_threads: int = 0

    # ---- SMEM / TMEM stage counts + TMEM column offsets --------------------------
    smem_kq_stages: int = CFG.SMEM_KQ_STAGES
    smem_v_stages: int = CFG.SMEM_V_STAGES
    smem_t_inv_stages: int = CFG.SMEM_T_INV_STAGES
    smem_a_stages: int = CFG.SMEM_A_STAGES
    smem_o_stages: int = CFG.SMEM_O_STAGES
    smem_checkpoint_stages: int = 1
    smem_gate_stages: int = CFG.SMEM_GATE_STAGES
    smem_beta_stages: int = CFG.SMEM_BETA_STAGES
    tmem_state_acc_stages: int = CFG.TMEM_KV_ACC_STAGES
    tmem_q_state_acc_stages: int = CFG.TMEM_Q_STATE_ACC_STAGES
    tmem_state_inp_stages: int = CFG.TMEM_STATE_INP_STAGES
    tmem_cg0_acc_stages: int = CFG.TMEM_CG0_ACC_STAGES
    tmem_cg1_acc_stages: int = CFG.TMEM_CG1_ACC_STAGES
    tmem_state_acc_offset: int = 0
    tmem_q_state_acc_offset: int = 0
    tmem_state_inp_offset: int = 0
    tmem_cg0_acc_offset: int = 0
    tmem_cg1_acc_offset: int = 0
    tmem_y_decay_u_inp_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- stamped by host at trace time (shape-derived) --------------------------
    kq_cosize: int = 0
    v_cosize: int = 0
    t_inv_cosize: int = 0
    a_cosize: int = 0
    o_cosize: int = 0
    checkpoint_cosize: int = 0
    tma_kq_bytes: int = 0
    tma_v_bytes: int = 0
    tma_o_bytes: int = 0
    n_heads_out: int = 0
    q_ratio: int = 1
    k_ratio: int = 1
    v_ratio: int = 1


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    state_dtype: Type[cutlass.Numeric],
    *,
    max_active_clusters: int,
    is_GQA: bool,
    use_initial_state: bool,
    store_final_state: bool = True,
    enable_checkpoints: bool = False,
    log_gate: bool = False,
    safe_gate: bool = False,
    beta_sigmoid: bool = False,
    dyn_sched: bool = False,
    paged_state: bool = False,
) -> GdnCfg:
    """Build the per-compile ``GdnCfg`` (io_dtype ∈ {Float16, BFloat16};
    acc is always Float32)."""
    if io_dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise ValueError(
            f"io_dtype={io_dtype} not supported; only Float16 and BFloat16 are supported"
        )
    if paged_state and not (use_initial_state and store_final_state):
        raise ValueError("paged_state requires use_initial_state and store_final_state")
    cfg = GdnCfg(
        io_dtype=io_dtype,
        acc_dtype=cutlass.Float32,
        state_dtype=state_dtype,
        max_active_clusters=max_active_clusters,
        is_GQA=is_GQA,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        paged_state=paged_state,
        log_gate=log_gate,
        safe_gate=safe_gate,
        beta_sigmoid=beta_sigmoid,
        dyn_sched=dyn_sched,
    )
    cfg.smem_checkpoint_stages = 1
    # The base four-stage KQ allocation exceeds SM100's 227 KiB dynamic-SMEM limit
    # after the structured shared-storage alignment required by the current DSL.
    # Checkpointing needs one additional 128x128 state tile, so it uses two stages.
    cfg.smem_kq_stages = 2 if enable_checkpoints else 3
    if not use_initial_state:
        cfg.num_regs_compute_group_1 = 232
        cfg.num_regs_other = 48
    n_cg0 = len(cfg.compute_group_0_warp_ids)
    n_cg1 = len(cfg.compute_group_1_warp_ids)
    cfg.threads_per_cta = cfg.threads_per_warp * (4 + n_cg0 + n_cg1)
    cfg.tmem_alloc_barrier_threads = cfg.threads_per_warp * (1 + n_cg0 + n_cg1)
    cfg.inverse_barrier_threads = cfg.threads_per_warp * n_cg0
    cfg.init_state_store_barrier_threads = cfg.threads_per_warp * n_cg1
    cfg.tmem_state_acc_offset = 0
    cfg.tmem_q_state_acc_offset = cfg.tmem_state_acc_offset + cfg.tmem_state_acc_stages * 128
    cfg.tmem_state_inp_offset = cfg.tmem_q_state_acc_offset + cfg.tmem_q_state_acc_stages * 64
    cfg.tmem_cg0_acc_offset = cfg.tmem_state_inp_offset + cfg.tmem_state_inp_stages * 64
    cfg.tmem_cg1_acc_offset = cfg.tmem_cg0_acc_offset + cfg.tmem_cg0_acc_stages * 64
    cfg.tmem_y_decay_u_inp_offset = cfg.tmem_cg1_acc_offset + cfg.tmem_cg1_acc_stages * 64
    return cfg


TENSORMAP_DESC_ARRAYS = 5  # per-batch runtime TMA descriptors: Q, K, V, O, checkpoints
TENSORMAP_STATIC_SLOTS = 0


# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def get_compiled_cache(
    io_dtype_str: str,
    state_dtype_str: str,
    HQ: int,
    HK: int,
    HV: int,
    is_gqa: bool,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
    log_gate: bool,
    safe_gate: bool,
    beta_sigmoid: bool,
    dyn_sched: bool,
    order_gen: bool,
    use_int64_offsets: bool,
    paged_state: bool,
    has_initial_state_mask: bool,
    device_index: int,
    device_major: int,
    device_minor: int,
    num_sm: int,
):
    """Return a mutable dict that lazily stores the compiled kernel."""
    return {}


def compile(
    io_dtype,
    state_dtype,
    is_gqa: bool,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
    log_gate: bool,
    safe_gate: bool,
    beta_sigmoid: bool,
    q_ratio: int,
    k_ratio: int,
    v_ratio: int,
    n_heads_out: int,
    dyn_sched: bool = False,
    use_int64_offsets: bool = False,
    *,
    num_sm: int,
    checkpoint_every_n_tokens: int,
    scale: float,
    paged_state: bool = False,
    has_initial_state_mask: bool = False,
):
    """JIT-compile one fake-tensor TVM-FFI GDN prefill signature.

    ``has_initial_state_mask`` only selects whether the paged signature carries the
    optional per-sequence seed mask; the kernel keys on the tensor's presence.
    """
    cfg = build_cfg(
        io_dtype,
        state_dtype,
        max_active_clusters=num_sm,
        is_GQA=is_gqa,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        log_gate=log_gate,
        safe_gate=safe_gate,
        beta_sigmoid=beta_sigmoid,
        dyn_sched=dyn_sched,
        paged_state=paged_state,
    )
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    tokens, sequence_entries, sequences, work_rows, sched_entries, workspace_words = (
        sym_int() for _ in range(6)
    )
    # The paged pool has its own slot count, decoupled from the sequence count.
    state_rows = sym_int() if paged_state else sequences
    tma_tensor = partial(
        make_strided_signature_tensor,
        assumed_align=16,
        use_int64_offsets=use_int64_offsets,
    )
    beta_dtype = io_dtype if beta_sigmoid else cutlass.Float32
    state_input_signature = (
        tma_tensor(state_dtype, (state_rows, n_heads_out, 128, 128))
        if use_initial_state
        else None
    )
    state_output_signature = (
        state_input_signature
        if paged_state
        else tma_tensor(state_dtype, (state_rows, n_heads_out, 128, 128))
        if store_final_state
        else None
    )
    if paged_state:
        state_indices_signature, state_mask_signature = make_paged_route_signatures(
            sequences, has_initial_state=has_initial_state_mask
        )
    else:
        state_indices_signature = state_mask_signature = None
    return compile_tvm_ffi(
        host,
        cfg,
        tma_tensor(io_dtype, (tokens, n_heads_out // q_ratio, 128)),
        tma_tensor(io_dtype, (tokens, n_heads_out // k_ratio, 128)),
        tma_tensor(io_dtype, (tokens, n_heads_out // v_ratio, 128)),
        make_strided_signature_tensor(
            cutlass.Float32,
            (tokens, n_heads_out),
            assumed_align=4,
            use_int64_offsets=use_int64_offsets,
        ),
        make_compact_signature_tensor(cutlass.Float32, (n_heads_out,), assumed_align=4)
        if safe_gate
        else None,
        make_compact_signature_tensor(cutlass.Float32, (n_heads_out,), assumed_align=4)
        if safe_gate
        else None,
        make_strided_signature_tensor(
            beta_dtype,
            (tokens, n_heads_out),
            assumed_align=beta_dtype.width // 8,
            use_int64_offsets=use_int64_offsets,
        ),
        tma_tensor(io_dtype, (tokens, n_heads_out, 128)),
        make_cu_seqlens_signature(sequence_entries),
        state_input_signature,
        state_output_signature,
        state_indices_signature,
        state_mask_signature,
        make_work_items_signature(work_rows),
        make_counter_signature(),
        make_counter_signature(sched_entries) if dyn_sched else None,
        cutlass.Int32(checkpoint_every_n_tokens),
        cutlass.Float32(scale),
        make_workspace_signature(workspace_words),
        name=(
            f"gdn_mega_prefill_{io_dtype.__name__.lower()}_{state_dtype.__name__.lower()}"
            f"_hq{n_heads_out // q_ratio}_hk{n_heads_out // k_ratio}"
            f"_hv{n_heads_out // v_ratio}_g{int(is_gqa)}"
            f"_i{int(use_initial_state)}f{int(store_final_state)}c{int(enable_checkpoints)}"
            f"_l{int(log_gate)}s{int(safe_gate)}b{int(beta_sigmoid)}"
            f"_d{int(dyn_sched)}_i64{int(use_int64_offsets)}"
            f"_p{int(paged_state)}m{int(has_initial_state_mask)}"
        ),
    )


def _validate_scalar_tensor(name, tensor) -> None:
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} innermost stride must be one")
    if any(stride < 0 for stride in tensor.stride()[:-1]):
        raise ValueError(f"{name} outer strides must be nonnegative")


def chunk_gdn_sm100(
    q,
    k,
    v,
    gate,
    beta,
    output,
    cu_seqlens,
    initial_state,
    output_state,
    scale: float,
    checkpoint_every_n_tokens: int = 0,
    output_state_checkpoints=None,
    work_items=None,
    work_count=None,
    sched_ctr=None,
    log_gate: bool = False,
    safe_gate: bool = False,
    a_log=None,
    dt_bias=None,
    use_beta_sigmoid: bool = False,
    work_item_scratch=None,
    *,
    tensormap_workspace,
    state_indices=None,
    has_initial_state=None,
) -> None:
    """Execute the Blackwell BT=64 scalar-GDN prefill kernel.

    ``gate`` is scalar ``(total_tokens, HO)``. With ``log_gate=True``, its values are
    natural-log decays; the kernel converts them to the log2 domain internally.
    Outer tensor strides are dynamic, while innermost dimensions must be contiguous.

    ``state_indices`` switches to paged mode: ``initial_state`` and ``output_state``
    must then alias one ``(slots, HO, 128, 128)`` pool. Non-positive routes are null;
    the optional uint8 ``has_initial_state`` mask selects resumed versus fresh slots.
    """
    for name, tensor in (("q", q), ("k", k), ("v", v), ("output", output)):
        validate_tma_tensor(name, tensor)
    _validate_scalar_tensor("gate", gate)
    _validate_scalar_tensor("beta", beta)
    for name, tensor in (("initial_state", initial_state), ("output_state", output_state)):
        if tensor is not None:
            validate_tma_tensor(name, tensor)
    if output_state_checkpoints is not None:
        validate_tma_tensor("output_state_checkpoints", output_state_checkpoints)
    validate_cu_seqlens(cu_seqlens, assumed_align=8)
    if tensormap_workspace.data_ptr() % 128:
        raise ValueError("tensormap_workspace data pointer must be 128-byte aligned")

    tokens, h_q, d_k = q.shape
    h_k = k.shape[1]
    h_v = v.shape[1]
    h_out = max(h_q, h_v)
    if d_k != 128 or k.shape != (tokens, h_k, 128) or v.shape != (tokens, h_v, 128):
        raise ValueError("q, k, and v must have shape (T, H, 128)")
    if gate.shape != (tokens, h_out) or beta.shape != (tokens, h_out):
        raise ValueError("gate and beta must have shape (T, HO)")
    if output.shape != (tokens, h_out, 128):
        raise ValueError("output must have shape (T, HO, 128)")
    if checkpoint_every_n_tokens < 0:
        raise ValueError("checkpoint interval must be zero or a positive tile multiple")
    enable_checkpoints = checkpoint_every_n_tokens > 0
    if enable_checkpoints and checkpoint_every_n_tokens % CFG.B_T:
        raise ValueError(f"checkpoint interval must be a positive multiple of {CFG.B_T}")
    if enable_checkpoints:
        if output_state_checkpoints is None:
            raise ValueError("checkpoint_every_n_tokens > 0 requires output_state_checkpoints")
        if output_state_checkpoints.shape[1:] != (h_out, 128, 128):
            raise ValueError("output_state_checkpoints must have shape (N, HO, 128, 128)")
        required_checkpoint_rows = checkpoint_capacity_bound(
            tokens, cu_seqlens.shape[0] - 1, checkpoint_every_n_tokens
        )
        if output_state_checkpoints.shape[0] < required_checkpoint_rows:
            raise ValueError(
                f"output_state_checkpoints requires at least {required_checkpoint_rows} rows, "
                f"got {output_state_checkpoints.shape[0]}"
            )
        if output_state_checkpoints.dtype != q.dtype:
            raise ValueError("output_state_checkpoints dtype must match q.dtype")
    if work_items is None or work_count is None:
        raise ValueError(
            "work_items/work_count are required (the split-table stage builds them for every launch)"
        )
    if safe_gate and (a_log is None or dt_bias is None):
        raise ValueError("safe_gate requires a_log and dt_bias")
    if not safe_gate:
        a_log = None
        dt_bias = None
    if a_log is not None:
        _validate_scalar_tensor("a_log", a_log)
    if dt_bias is not None:
        _validate_scalar_tensor("dt_bias", dt_bias)

    use_initial_state = initial_state is not None
    store_final_state = output_state is not None
    if (
        initial_state is not None
        and output_state is not None
        and initial_state.dtype != output_state.dtype
    ):
        raise ValueError("initial_state and output_state dtypes must match")
    paged_state = state_indices is not None
    has_initial_state_mask = has_initial_state is not None
    if paged_state and (initial_state is None or initial_state is not output_state):
        raise ValueError("paged mode requires one aliased input/output state pool")
    if initial_state is not None:
        state_dtype_src = initial_state.dtype
    elif output_state is not None:
        state_dtype_src = output_state.dtype
    else:
        state_dtype_src = "float32"

    for name, heads in (("HQ", h_q), ("HK", h_k), ("HV", h_v)):
        if h_out % heads:
            raise ValueError(f"{name}={heads} must divide output heads {h_out}")
    q_ratio = h_out // h_q
    k_ratio = h_out // h_k
    v_ratio = h_out // h_v
    is_gqa = h_q >= h_v
    dyn_sched = sched_ctr is not None
    order_gen = work_item_scratch is None
    checkpoints_for_descs = output_state_checkpoints if enable_checkpoints else None
    use_int64_offsets = requires_int64_abi(
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        output,
        cu_seqlens,
        initial_state,
        output_state,
        checkpoints_for_descs,
        work_item_scratch,
        work_items,
        work_count,
        sched_ctr,
        tensormap_workspace,
        state_indices,
        has_initial_state,
    )
    device_index = tensor_device_index(q)
    if current_device() != device_index:
        raise ValueError("the active CUDA device must match q.device")
    device_properties = get_device_properties(device_index)
    num_sm = device_properties.multi_processor_count
    cache = get_compiled_cache(
        str(q.dtype),
        str(state_dtype_src),
        h_q,
        h_k,
        h_v,
        is_gqa,
        use_initial_state,
        store_final_state,
        enable_checkpoints,
        log_gate,
        safe_gate,
        use_beta_sigmoid,
        dyn_sched,
        order_gen,
        use_int64_offsets,
        paged_state,
        has_initial_state_mask,
        device_index,
        device_properties.major,
        device_properties.minor,
        num_sm,
    )

    io_dtype = get_dtype(q.dtype)
    if "compiled" not in cache:
        cache["compiled"] = compile(
            io_dtype,
            get_dtype(state_dtype_src),
            is_gqa,
            use_initial_state,
            store_final_state,
            enable_checkpoints,
            log_gate,
            safe_gate,
            use_beta_sigmoid,
            q_ratio,
            k_ratio,
            v_ratio,
            h_out,
            dyn_sched,
            use_int64_offsets,
            num_sm=num_sm,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            scale=scale,
            paged_state=paged_state,
            has_initial_state_mask=has_initial_state_mask,
        )

    if "prologue" not in cache:
        sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
        tokens, sequence_entries, checkpoint_rows, work_rows, sched_entries, workspace_words = (
            sym_int() for _ in range(6)
        )
        tma_tensor = partial(
            make_strided_signature_tensor,
            assumed_align=16,
            use_int64_offsets=use_int64_offsets,
        )
        work_items_signature = make_work_items_signature(work_rows)
        cache["prologue"] = compile_tvm_ffi(
            prologue,
            io_dtype,
            CFG.B_T,
            order_gen,
            dyn_sched,
            tma_tensor(io_dtype, (tokens, h_q, 128)),
            tma_tensor(io_dtype, (tokens, h_k, 128)),
            tma_tensor(io_dtype, (tokens, h_v, 128)),
            tma_tensor(io_dtype, (tokens, h_out, 128)),
            make_cu_seqlens_signature(sequence_entries),
            tma_tensor(io_dtype, (checkpoint_rows, h_out, 128, 128))
            if checkpoints_for_descs is not None
            else None,
            work_items_signature if not order_gen else None,
            make_counter_signature(),
            work_items_signature,
            make_counter_signature(sched_entries) if dyn_sched else None,
            cutlass.Int32(checkpoint_every_n_tokens),
            make_workspace_signature(workspace_words),
            name=(
                f"gdn_mega_prefill_prologue_{io_dtype.__name__.lower()}"
                f"_hq{h_q}_hk{h_k}_hv{h_v}_ho{h_out}"
                f"_c{int(enable_checkpoints)}o{int(order_gen)}d{int(dyn_sched)}"
                f"_i64{int(use_int64_offsets)}"
            ),
        )
    cache["prologue"](
        q,
        k,
        v,
        output,
        cu_seqlens,
        checkpoints_for_descs,
        work_item_scratch if not order_gen else None,
        work_count,
        work_items,
        sched_ctr,
        checkpoint_every_n_tokens,
        tensormap_workspace,
    )
    cache["compiled"](
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        output,
        cu_seqlens,
        initial_state if use_initial_state else None,
        output_state if store_final_state else None,
        state_indices,
        has_initial_state,
        work_items,
        work_count,
        sched_ctr,
        checkpoint_every_n_tokens,
        scale,
        tensormap_workspace,
    )
