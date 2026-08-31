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
# Modified by Attention Gym in 2026: imports were relocated into this package.

"""
Chunked Gated Delta Net (GDN) BPROP kernel for Blackwell SM100 (Cutlass primitives).

Algorithm overview (per chunk c, iterated c = NT-1 .. 0):
  Inputs : Q[BT,DK], K[BT,DK], V[BT,DV], dO[BT,DV], S_c[DK,DV] (= checkpoint entry c-1,
           the forward state ENTERING chunk c), Gate[BT], Beta[BT]
  State  : dstate[DK,DV]  (state gradient, held in TMEM, accumulated backward)

  MMA order.  A = the staged attention matrix
  (CG0's A epilogue, sA); dO' = dO * cumprod_vals * scale (CG1 restage):
  KK          : W_kk[BT,BT]  = K  @ K^T        -> shared acc   (for T)
  QK          : W_qk[BT,BT]  = Q  @ K^T        -> shared acc   (for the A tile)
  dV inter    : dV[DV,BT]    = dstate^T(TMEM) @ K^T -> the dV/dK slot
                (acc=False; CG1 then scales it by decay_scale_vals IN PLACE)
  KS          : KS[BT,DV]    = S^T(SMEM) @ K^T  -> shared acc
  dU intra    : dV/dK slot  += dO^T(SMEM) @ A(SMEM)  (waits CG1's
                in-place scale -> the accumulate is the reference's
                dv2 = dv_intra + exp2(g_last-g)*(k@dh))
  dstate Q-term : dstate    += dO'^T(TMEM) @ Q  (acc=False on the
                first backward chunk = dstate init; the reference's
                dh = dh*exp2(g_last) + (q*scale*exp2(g))^T @ do)
  dY          : dY[DV,BT]    = dU^T(TMEM f16) @ T(SMEM) -> shared acc
                (dY == dV == d(Y); T read TRANSPOSED vs prefill)
  dA          : dA_eff[BT,BT]= dO(SMEM) @ U^T(SMEM) -> shared acc
                (sV holds U; CG0 masks it -> sDa for dQ/dK)
  dM core     : dM[BT,BT]    = dY(SMEM) @ U^T(SMEM) -- the WY
                inverse backward T^T dT T^T collapsed via
                T^T dU = dY and T Y = U (Beta folds through
                sTinv's column scale into both factors); CG0
                applies -strict(2^{g_i-g_j}) -> sDm (the reference's dA22;
                both betas cancel against the K rows in the dM-terms)
  dstate update-term : dstate += dY'^T(TMEM f16) @ K  (dY' = -cumprod_vals
                * dY: the -(W^T @ dY) term; ready commits HERE)
  dK dM-terms : dV/dK slot  += K^T @ dM^T + K^T @ dM  (the reference's
                dk += dA22^T@K_beta / dk_beta += dA22@K folded, the
                beta row scale cancelling with K_beta = beta*K)
  dK state-path : SY[BT,DV]    = S^T(SMEM) @ dY^T(SMEM) -> the shared f16
                input columns (dO'/dU/dY' dead by then); CG1 reads it as
                -cumprod_vals[t] * (dY @ S^T)[t,:] and adds it to the banked
                inter+attn dK terms

  Per chunk CG1: restages dO' and dU and dY' -> the f16 input columns
  (dY' overwrites dU), computes Y = V - cumprod_vals*(K @ state) in registers and stages
  Y and g_k_state to their dedicated f16 TMEM slots (the U GEMM and
  the dV-pass read them), stages Q^T over the Y slot after the dV pass,
  loads dY and stages it plain to sdV (the dV output).

SMEM layout (~226 KB of the 227 KB SM100 cap):
  Buffer                           Size (B)  Stages
  Q                                16384     1
  K                                32768     2      <-- double-buffered (prefetch next chunk)
  V                                16384     1      <-- overwritten in place by U
  dO                               16384     1
  state (checkpoint entry c-1)     32768     1      <-- io-dtype [DK,DV], TMA-loaded
  T_inv                            8192      1      <-- inverse OUTPUT (upper tri = kernel-start zeros)
  KK (strict-masked M_kk)          8192      1      <-- KK epi's only store; inverse input + dGate/dBeta
  A staging / sDa                  8192      1      <-- ALIAS: A then the masked dA
  dM staging (sDm)                 8192      1      <-- Step 8 -> dK dM-terms
  dstate_entry (sDstate)                   32768     1      <-- f16 restage, dK-inter's A
  dQ store staging                 16384     1
  dK store staging                 16384     1
  dV store staging                 16384     1
  cumsumlog / cumprod / Beta       3 x 512   2      <-- in-place dGate/dBeta staging

TMEM layout (512 cols; EXACTLY the prefill map, state->dstate, O->dV):
  cols 0-128   : dstate accumulator (fp32)     <-- prefill: state acc
  cols 128-192 : dV/dK accumulator (fp32)      <-- one slot, five sequential
                 per-chunk productions (dV inter -> dU intra -> dK inter
                 -> dK attn -> dK dM-terms), each with its own mbar pair
  cols 192-256 : dstate input (f16 packed)     <-- prefill: state_inp
  cols 256-384 : shared accumulators x2        <-- KK / A / k_state / U / dY / dA
                                                    / dM core
  cols 384-448 : shared inputs x2 (f16 packed) <-- dO' / dU / dY'; the dK
                 state-path acc overwrites them after their last GEMM reads
                 (CG1's state-path readout precedes its next-chunk restages)
  cols 448-512 : Y (448) + g_k_state (480) f16 slots until the dV pass, then
                 Q^T (448) until CG1's dQ dot reads it

Warp assignments (12 warps = 384 threads):
  warps 0-3     : compute group 0 - T-pairwise, KK epi, A epi, inverse
  warps 4-7     : compute group 1 - dV epilogue, dstate scale + f16 restage
                                    (later: Y = V - K*state -> TMEM, dY staging)
  warp  8       : MMA warp       - issues the GEMMs
  warp  9       : TMA load warp  - loads Q, K (double-buf), V, dO, state(=checkpoint)
  warp  10      : gate warp      - loads Gate/Beta (double-buffered,
                                    backward order) + stores dGate/dBeta
  warp  11      : epilogue warp   - store dQ, dK, dV to global memory
"""

from dataclasses import dataclass
from functools import cache, partial
from typing import NamedTuple

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.experimental.primitives as nvvm
from cutlass import cute
from cutlass.experimental import cuda

from attn_gym._backends.cute import compile_tvm_ffi
from attn_gym._backends.cute.utils import requires_int64_abi

from .common.elementwise import softplus
from .common.host import get_dtype
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
    make_counter_signature,
    make_cu_seqlens_signature,
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
from .gdn_bprop_config import CFG
from .tile_dsl.barrier import (
    MBarrier,
    PipelineState,
    Producer,
    advance,
)
from .tile_dsl.handles import MmaDesc, SmemTile, smem_data_ptr, tma_slice_runtime_desc
from .tile_dsl.mma import mma_ss, mma_step, mma_step_k8, mma_ts_step
from .tile_dsl.pointwise import (
    f16x2_to_f32,
    fadd2,
    ffma2,
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

RCP_LN2 = 1.4426950408889634  # 1/ln(2): natural-log gates -> the kernel's log2 domain


@cute.jit
def smem_64x64_offset(row, col):
    """Return the swizzled in-row offset for a 64-column shared-memory tile."""
    return swizzle_xor_128b(row, col)


@cute.jit
def smem_64x64_linear_offset(offset):
    """Convert a row-major BT=64 offset to the shared-memory swizzle."""
    return swizzle_lin_128b(offset, row_stride_log2=6)


class GdnBwdBars(NamedTuple):
    """GDN bprop pipeline mbarrier inventory.

    Every pipeline is a ``_ready``/``_done`` MBarrier pair over one ring: a
    slot is acquired for filling by waiting ``_done`` and committed by
    arriving ``_ready``; the reading side waits ``_ready`` and releases the
    slot by arriving ``_done``.

    Operand buffers read by both the MMA warp and a compute group carry a
    SPLIT done pair (``_mma_done`` MMA_COMMIT + ``_cg?_done`` THREAD) and the
    TMA warp waits BOTH: a plain arrive from the MMA warp fires at MMA issue,
    not completion, which would release the buffer for reload mid-GEMM.
    """

    mb_q_ready: MBarrier
    mb_q_mma_done: MBarrier
    mb_q_cg1_done: MBarrier
    mb_k_ready: MBarrier
    mb_k_mma_done: MBarrier
    mb_k_cg0_done: MBarrier
    mb_v_ready: MBarrier
    mb_v_mma_done: MBarrier
    mb_do_ready: MBarrier
    mb_do_mma_done: MBarrier
    mb_state_ready: MBarrier
    mb_state_mma_done: MBarrier

    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier
    mb_beta_ready: MBarrier
    mb_beta_done: MBarrier

    mb_dstate_acc_ready: MBarrier
    mb_dstate_scale_acc_done: MBarrier
    mb_du_scale_acc_ready: MBarrier
    mb_du_scale_acc_done: MBarrier
    mb_du_total_acc_ready: MBarrier
    mb_dk_scale_acc_ready: MBarrier
    mb_dk_scale_acc_done: MBarrier
    mb_dk_attn_acc_ready: MBarrier
    mb_dk_attn_acc_done: MBarrier
    mb_dk_total_acc_ready: MBarrier
    mb_dk_total_acc_done: MBarrier
    mb_dq_acc_scale_ready: MBarrier
    mb_dq_acc_scale_done: MBarrier
    mb_dq_acc_total_ready: MBarrier
    mb_dq_acc_total_done: MBarrier
    mb_kk_acc_ready: MBarrier
    mb_kk_acc_done: MBarrier
    mb_a_acc_ready: MBarrier
    mb_k_state_acc_ready: MBarrier
    mb_u_acc_ready: MBarrier
    mb_dy_acc_ready: MBarrier
    mb_da_acc_ready: MBarrier
    mb_dm_acc_ready: MBarrier
    mb_dm_acc_done: MBarrier
    mb_dk_state_path_acc_ready: MBarrier

    mb_dstate_inp_ready: MBarrier
    mb_dstate_inp_done: MBarrier
    mb_do_prime_inp_ready: MBarrier
    mb_du_inp_ready: MBarrier
    mb_dyp_inp_ready: MBarrier
    mb_y_ready: MBarrier

    mb_t_inv_ready: MBarrier
    mb_a_ready: MBarrier
    mb_a_done: MBarrier
    mb_u_ready: MBarrier
    mb_dstate_smem_ready: MBarrier
    mb_state_dot_dstate_done: MBarrier
    mb_da_ready: MBarrier

    mb_dbeta_cg1_ready: MBarrier
    mb_dgate_cg1_ready: MBarrier

    mb_dq_tmastg_ready: MBarrier
    mb_dq_tmastg_done: MBarrier
    mb_dk_tmastg_ready: MBarrier
    mb_dk_tmastg_done: MBarrier
    mb_dv_tmastg_ready: MBarrier
    mb_dv_tmastg_done: MBarrier
    mb_sdv_done: MBarrier

    mb_tmem_done: MBarrier
    mb_sched_ready: MBarrier
    mb_sched_done: MBarrier


def make_gdn_bars(cfg) -> GdnBwdBars:
    """GdnBwdBars factory.  MUST be called from inside ``kernel`` (allocates SMEM;
    the mbar rings sit ahead of the gate scalar arrays and data buffers)."""
    ONE_LANE = 1
    MMA_ARRIVERS = len([cfg.mma_warp_id])
    GATE_WARP = cfg.threads_per_warp * len([cfg.load_gate_beta_warp_id])
    EPI_WARP = cfg.threads_per_warp * len([cfg.epilogue_warp_id])
    CG0_THREADS = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    CG1_THREADS = cfg.threads_per_warp * len(cfg.compute_group_1_warp_ids)
    CG0_PLUS_CG1 = CG0_THREADS + CG1_THREADS

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=16)

    return GdnBwdBars(
        mb_q_ready=MBarrier(
            alloc(cfg.smem_q_stages),
            stages=cfg.smem_q_stages,
            init_count=ONE_LANE,
            producer=Producer.TMA_LOAD,
        ),
        mb_q_mma_done=MBarrier(
            alloc(cfg.smem_q_stages),
            stages=cfg.smem_q_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_q_cg1_done=MBarrier(
            alloc(cfg.smem_q_stages),
            stages=cfg.smem_q_stages,
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_k_ready=MBarrier(
            alloc(cfg.smem_k_stages),
            stages=cfg.smem_k_stages,
            init_count=ONE_LANE,
            producer=Producer.TMA_LOAD,
        ),
        mb_k_mma_done=MBarrier(
            alloc(cfg.smem_k_stages),
            stages=cfg.smem_k_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_k_cg0_done=MBarrier(
            alloc(cfg.smem_k_stages),
            stages=cfg.smem_k_stages,
            init_count=CG0_THREADS,
            producer=Producer.THREAD,
        ),
        mb_v_ready=MBarrier(
            alloc(cfg.smem_v_stages),
            stages=cfg.smem_v_stages,
            init_count=ONE_LANE,
            producer=Producer.TMA_LOAD,
        ),
        mb_v_mma_done=MBarrier(
            alloc(cfg.smem_v_stages),
            stages=cfg.smem_v_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_do_ready=MBarrier(
            alloc(cfg.smem_do_stages),
            stages=cfg.smem_do_stages,
            init_count=ONE_LANE,
            producer=Producer.TMA_LOAD,
        ),
        mb_do_mma_done=MBarrier(
            alloc(cfg.smem_do_stages),
            stages=cfg.smem_do_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_state_ready=MBarrier(
            alloc(cfg.smem_state_stages),
            stages=cfg.smem_state_stages,
            init_count=ONE_LANE,
            producer=Producer.TMA_LOAD,
        ),
        mb_state_mma_done=MBarrier(
            alloc(cfg.smem_state_stages),
            stages=cfg.smem_state_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
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
            init_count=CG0_PLUS_CG1,
            producer=Producer.THREAD,
        ),
        mb_dstate_acc_ready=MBarrier(
            alloc(cfg.tmem_dstate_acc_stages),
            stages=cfg.tmem_dstate_acc_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_dstate_scale_acc_done=MBarrier(
            alloc(cfg.tmem_dstate_acc_stages),
            stages=cfg.tmem_dstate_acc_stages,
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        # five sequential per-chunk productions share the dV/dK TMEM slot
        mb_du_scale_acc_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_du_scale_acc_done=MBarrier(
            alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_du_total_acc_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_dk_scale_acc_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_dk_scale_acc_done=MBarrier(
            alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD
        ),
        mb_dk_attn_acc_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_dk_attn_acc_done=MBarrier(
            alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD
        ),
        mb_dk_total_acc_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_dk_total_acc_done=MBarrier(
            alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_dq_acc_scale_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_dq_acc_scale_done=MBarrier(
            alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD
        ),
        mb_dq_acc_total_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_dq_acc_total_done=MBarrier(
            alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        # shared accumulators at STATIC columns: group A holds
        # KK -> k_state -> dY -> dM core, group B holds A -> U -> dA.
        mb_kk_acc_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_kk_acc_done=MBarrier(
            alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD
        ),
        mb_a_acc_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_k_state_acc_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_u_acc_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_dy_acc_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_da_acc_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_dm_acc_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_dm_acc_done=MBarrier(
            alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD
        ),
        mb_dk_state_path_acc_ready=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_dstate_inp_ready=MBarrier(
            alloc(cfg.tmem_dstate_inp_stages),
            stages=cfg.tmem_dstate_inp_stages,
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_dstate_inp_done=MBarrier(
            alloc(cfg.tmem_dstate_inp_stages),
            stages=cfg.tmem_dstate_inp_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        # f16 input restages at STATIC columns: dO' alone; dU and dY' overlap
        mb_do_prime_inp_ready=MBarrier(
            alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_du_inp_ready=MBarrier(
            alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_dyp_inp_ready=MBarrier(
            alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_y_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_t_inv_ready=MBarrier(
            alloc(cfg.smem_t_inv_stages),
            stages=cfg.smem_t_inv_stages,
            init_count=CG0_THREADS,
            producer=Producer.THREAD,
        ),
        mb_a_ready=MBarrier(
            alloc(cfg.smem_a_stages),
            stages=cfg.smem_a_stages,
            init_count=CG0_THREADS,
            producer=Producer.THREAD,
        ),
        mb_a_done=MBarrier(
            alloc(cfg.smem_a_stages),
            stages=cfg.smem_a_stages,
            init_count=MMA_ARRIVERS,
            producer=Producer.MMA_COMMIT,
        ),
        mb_u_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_dstate_smem_ready=MBarrier(
            alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_state_dot_dstate_done=MBarrier(
            alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD
        ),
        mb_da_ready=MBarrier(alloc(1), stages=1, init_count=CG0_THREADS, producer=Producer.THREAD),
        mb_dbeta_cg1_ready=MBarrier(
            alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_dgate_cg1_ready=MBarrier(
            alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD
        ),
        mb_dq_tmastg_ready=MBarrier(
            alloc(cfg.smem_dq_stages),
            stages=cfg.smem_dq_stages,
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_dq_tmastg_done=MBarrier(
            alloc(cfg.smem_dq_stages),
            stages=cfg.smem_dq_stages,
            init_count=EPI_WARP,
            producer=Producer.THREAD,
        ),
        mb_dk_tmastg_ready=MBarrier(
            alloc(cfg.smem_dk_stages),
            stages=cfg.smem_dk_stages,
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_dk_tmastg_done=MBarrier(
            alloc(cfg.smem_dk_stages),
            stages=cfg.smem_dk_stages,
            init_count=EPI_WARP,
            producer=Producer.THREAD,
        ),
        mb_dv_tmastg_ready=MBarrier(
            alloc(cfg.smem_dv_stages),
            stages=cfg.smem_dv_stages,
            init_count=CG1_THREADS,
            producer=Producer.THREAD,
        ),
        mb_dv_tmastg_done=MBarrier(
            alloc(cfg.smem_dv_stages),
            stages=cfg.smem_dv_stages,
            init_count=EPI_WARP,
            producer=Producer.THREAD,
        ),
        mb_sdv_done=MBarrier(
            alloc(1), stages=1, init_count=MMA_ARRIVERS, producer=Producer.MMA_COMMIT
        ),
        mb_tmem_done=MBarrier(
            alloc(1), stages=1, init_count=CG0_THREADS + CG1_THREADS, producer=Producer.THREAD
        ),
        mb_sched_ready=MBarrier(
            alloc(cfg.sched_stages),
            stages=cfg.sched_stages,
            init_count=ONE_LANE,
            producer=Producer.THREAD,
        ),
        mb_sched_done=MBarrier(
            alloc(cfg.sched_stages),
            stages=cfg.sched_stages,
            init_count=11,
            producer=Producer.THREAD,
        ),
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
def invert_diagonal_NxN(cfg, in_base, out_base, d, tidx, N: int = 8):
    """Gauss-Jordan inversion of one diagonal NxN block, ``in_base`` -> ``out_base`` (f16 SMEM)."""
    tidx_in_group = tidx % N
    BT = cfg.b_t

    row_lin_base = (d * N + tidx_in_group) * BT + d * N
    row_phys = smem_64x64_linear_offset(row_lin_base)
    row_ptr_in = (
        cute.make_ptr(
            cfg.io_dtype,
            in_base,
            mem_space=cute.AddressSpace.smem,
            assumed_align=cfg.buffer_align_bytes,
        )
        + row_phys
    )
    row_ptr = (
        cute.make_ptr(
            cfg.io_dtype,
            out_base,
            mem_space=cute.AddressSpace.smem,
            assumed_align=cfg.buffer_align_bytes,
        )
        + row_phys
    )

    row = [(row_ptr_in + j).load().to(cutlass.Float32) for j in range(N)]
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
def warp_reduce_scatter_frag_16_elems(vals, lane_id):
    """Reduce-scatter 16 fragment token-partials (tcol = (lane%4)*2 +
    (j//2)*8 + (j%2)); lane L returns the sums of tokens (L//4)*8 + (L%4)*2 and +1."""
    cur = vals
    for k in cutlass.range_constexpr(3):
        off = cutlass.const_expr(4 << k)
        hi_lane = (lane_id // off) % 2 == 1
        nxt = []
        for i in cutlass.range_constexpr(len(cur)):
            if cutlass.const_expr((i & 2) == 0):
                lo = cur[i]
                hi = cur[i + 2]
                send = lo if hi_lane else hi
                recv = nvvm.shfl_sync(0xFFFFFFFF, send, off, 31, kind=nvvm.Shfl.BFLY)
                keep = hi if hi_lane else lo
                nxt.append(keep + recv)
        cur = nxt
    return cur[0], cur[1]


@cute.jit
def blockwise_diagonal_8x8_to_16x16(cfg, base_int, raw_base, d0, lane_id):
    """Off-diagonal correction 8x8 -> 16x16 (C <- -D^{-1} C A^{-1}); raw C from ``raw_base``, writes on ``base_int``."""
    bpe = cfg.io_dtype.width // 8
    ldsm_x1_off = (lane_id % 8) * 64
    d = nvvm.ldmatrix(
        cutlass.inttoptr(
            base_int + smem_64x64_linear_offset((d0 + 8) * 64 + d0 + 8 + ldsm_x1_off) * bpe,
            cutlass.AddressSpace.smem,
            cutlass.BFloat16,
        ),
        1,
        nvvm.MMALayout.ROW,
    )
    c = nvvm.ldmatrix(
        cutlass.inttoptr(
            raw_base + smem_64x64_linear_offset((d0 + 8) * 64 + d0 + ldsm_x1_off) * bpe,
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
    ai = nvvm.ldmatrix(
        cutlass.inttoptr(
            base_int + smem_64x64_linear_offset(d0 * 64 + d0 + ldsm_x1_off) * bpe,
            cutlass.AddressSpace.smem,
            cutlass.BFloat16,
        ),
        1,
        nvvm.MMALayout.COL,
    )
    o_regs = cutlass.Array(cutlass.Float32, 4, alignment=16, space=cutlass.AddressSpace.rmem)
    for i in cutlass.range_constexpr(4):
        o_regs[i] = cutlass.Float32(0.0)
    mma_step_k8(o_regs, a_pack, [ai], k_step=0, M=16, N=8, ab_dtype=cfg.io_dtype)
    o_pack = fp32_to_fp16(o_regs[0], o_regs[1], dtype=cfg.io_dtype)

    # ---- store corrected C -------------------------------------------------------
    nvvm.stmatrix(
        cutlass.inttoptr(
            base_int + smem_64x64_linear_offset((d0 + 8) * 64 + d0 + ldsm_x1_off) * bpe,
            cutlass.AddressSpace.smem,
            cutlass.BFloat16,
        ),
        o_pack,
        nvvm.MMALayout.ROW,
    )


@cute.jit
def blockwise_diagonal_16x16_to_32x32(cfg, base_int, raw_base, d0, lane_id):
    """Off-diagonal correction 16x16 -> 32x32 (raw C from ``raw_base``)."""
    bpe = cfg.io_dtype.width // 8
    ldsm_x4_off = (lane_id % 16) * 64 + (lane_id // 16) * 8
    d = list(
        nvvm.ldmatrix(
            cutlass.inttoptr(
                base_int + smem_64x64_linear_offset((d0 + 16) * 64 + d0 + 16 + ldsm_x4_off) * bpe,
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
                raw_base + smem_64x64_linear_offset((d0 + 16) * 64 + d0 + ldsm_x4_off) * bpe,
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
    ai = list(
        nvvm.ldmatrix(
            cutlass.inttoptr(
                base_int + smem_64x64_linear_offset(d0 * 64 + d0 + ldsm_x4_off) * bpe,
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
    mma_step(o_regs, a_pack, ai, k_step=0, M=16, N=16, ab_dtype=cfg.io_dtype)
    o_pack = [fp32_to_fp16(o_regs[2 * j], o_regs[2 * j + 1], dtype=cfg.io_dtype) for j in range(4)]

    # ---- store corrected C -------------------------------------------------------
    nvvm.stmatrix(
        cutlass.inttoptr(
            base_int + smem_64x64_linear_offset((d0 + 16) * 64 + d0 + ldsm_x4_off) * bpe,
            cutlass.AddressSpace.smem,
            cutlass.BFloat16,
        ),
        o_pack,
        nvvm.MMALayout.ROW,
    )


@cute.jit
def blockwise_diagonal_32x32_to_64x64(cfg, base_int, raw_base, warp_id, lane_id):
    """Off-diagonal correction 32x32 -> 64x64 (2 warps, one 16-row M-band each; raw C from ``raw_base``)."""
    band = warp_id % 2
    bpe = cfg.io_dtype.width // 8
    ldsm_x4_off = (lane_id % 16) * 64 + (lane_id // 16) * 8
    a_frags = []
    for vs in cutlass.range_constexpr(2):
        a_frags += list(
            nvvm.ldmatrix(
                cutlass.inttoptr(
                    base_int
                    + smem_64x64_linear_offset((32 + band * 16) * 64 + 32 + vs * 16 + ldsm_x4_off)
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
                    raw_base
                    + smem_64x64_linear_offset(
                        (32 + (vs // 2) * 16) * 64 + (vs % 2) * 16 + ldsm_x4_off
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
                    + smem_64x64_linear_offset(((vs // 2) * 16) * 64 + (vs % 2) * 16 + ldsm_x4_off)
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
        cfg.inverse_inner_barrier_id,
        thread_count=cfg.inverse_inner_barrier_threads,
    )
    nvvm.stmatrix(
        cutlass.inttoptr(
            base_int + smem_64x64_linear_offset((32 + band * 16) * 64 + ldsm_x4_off) * bpe,
            cutlass.AddressSpace.smem,
            cutlass.BFloat16,
        ),
        o_pack[0:4],
        nvvm.MMALayout.ROW,
    )
    nvvm.stmatrix(
        cutlass.inttoptr(
            base_int + smem_64x64_linear_offset((32 + band * 16) * 64 + 16 + ldsm_x4_off) * bpe,
            cutlass.AddressSpace.smem,
            cutlass.BFloat16,
        ),
        o_pack[4:8],
        nvvm.MMALayout.ROW,
    )


@cute.jit
def tmastg_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sdQ_raw,
    sdK_raw,
    sdV_raw,
    desc_dq_base,
    desc_dk_base,
    desc_dv_base,
    sSched,
    bars,
):
    """TMA-STG warp role (warp 11): persistent tile-scheduler loop + per-chunk
    dQ/dK/dV TMA bulk-stores from the SMEM staging buffers to global memory."""
    elect_one = nvvm.elect_sync()
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    dq_index = PipelineState.start(phase=0)
    dk_index = PipelineState.start(phase=0)
    dv_index = PipelineState.start(phase=0)

    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1

    bpe = cfg.io_dtype.width // 8
    granule_elems = 128 // bpe
    sdQ_tma = SmemTile(
        base=sdQ_raw,
        elems_per_stage=(cfg.dq_cosize // cfg.smem_dq_stages),
        stages=cfg.smem_dq_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granule_elems,
        tma_subtile_stride_elems=4096,
    )
    sdK_tma = SmemTile(
        base=sdK_raw,
        elems_per_stage=(cfg.dk_cosize // cfg.smem_dk_stages),
        stages=cfg.smem_dk_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granule_elems,
        tma_subtile_stride_elems=4096,
    )
    sdV_tma = SmemTile(
        base=sdV_raw,
        elems_per_stage=(cfg.dv_cosize // cfg.smem_dv_stages),
        stages=cfg.smem_dv_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granule_elems,
        tma_subtile_stride_elems=4096,
    )
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

        head_q = head_idx if cfg.q_ratio == 1 else head_idx // cutlass.Int32(cfg.q_ratio)
        head_k = head_idx if cfg.k_ratio == 1 else head_idx // cutlass.Int32(cfg.k_ratio)
        head_v = head_idx if cfg.v_ratio == 1 else head_idx // cutlass.Int32(cfg.v_ratio)
        slot = batch_idx * desc_qwords
        desc_dq_slot = (desc_dq_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_dk_slot = (desc_dk_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_dv_slot = (desc_dv_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_dq_slot)
            tma_tensormap_acquire(desc_dk_slot)
            tma_tensormap_acquire(desc_dv_slot)

        for rev_idx in cutlass.range(cend - wstart):
            chunk_idx = cend - 1 - rev_idx
            tok_coord = chunk_idx * cutlass.Int32(cfg.b_t)

            dv_idx = dv_index.idx
            bars.mb_dv_tmastg_ready[dv_idx].wait(dv_index.phase)
            dv_index = advance(dv_index, cfg.smem_dv_stages)
            dv_slice = tma_slice_runtime_desc(desc_dv_slot, cutlass.Int32(0), head_v, tok_coord)
            if chunk_idx < wend:
                tma_store_tile(sdV_tma[dv_idx], dv_slice, acquire=False)
                tma_store_commit()

            dq_idx = dq_index.idx
            bars.mb_dq_tmastg_ready[dq_idx].wait(dq_index.phase)
            dq_index = advance(dq_index, cfg.smem_dq_stages)
            dq_slice = tma_slice_runtime_desc(desc_dq_slot, cutlass.Int32(0), head_q, tok_coord)
            if chunk_idx < wend:
                tma_store_tile(sdQ_tma[dq_idx], dq_slice, acquire=False)
                tma_store_commit()

            dk_idx = dk_index.idx
            bars.mb_dk_tmastg_ready[dk_idx].wait(dk_index.phase)
            dk_index = advance(dk_index, cfg.smem_dk_stages)
            dk_slice = tma_slice_runtime_desc(desc_dk_slot, cutlass.Int32(0), head_k, tok_coord)
            if chunk_idx < wend:
                tma_store_tile(sdK_tma[dk_idx], dk_slice, acquire=False)
                tma_store_commit()

            tma_store_wait(2)
            bars.mb_dv_tmastg_done[dv_idx].arrive()
            tma_store_wait(1)
            bars.mb_dq_tmastg_done[dq_idx].arrive()
            tma_store_wait(0)
            bars.mb_dk_tmastg_done[dk_idx].arrive()
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
    mDgate,
    mDbeta,
    sCumsumlog,
    sCumprod,
    sBeta,
    sSched,
    bars,
):
    """Gate/beta LOAD + STORE warp role (warp 10): per-chunk Gate/Beta G->S
    loads and the dGate/dBeta stores."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    gate_index = PipelineState.start(phase=1)
    beta_index = PipelineState.start(phase=1)
    gate_store_index = PipelineState.start(phase=0)
    beta_store_index = PipelineState.start(phase=0)
    lidx = tidx % cfg.threads_per_warp
    n_cols = cfg.b_t // cfg.threads_per_warp
    a_l2 = cutlass.Float32(0.0)
    bias = cutlass.Float32(0.0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
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
        num_item_chunks = cend - wstart
        if cutlass.const_expr(cfg.safe_gate):
            if num_item_chunks > 0:
                # per-head transform constants, fixed for the whole tile
                a_l2 = -cute.math.exp2(
                    mA_log[head_idx].to(cutlass.Float32) * cutlass.Float32(RCP_LN2), fastmath=True
                ) * cutlass.Float32(RCP_LN2)
                bias = mDt_bias[head_idx].to(cutlass.Float32)
        # dGate/dBeta ownership: mask stores past the item's write range
        write_end = batch_start + wend * cfg.b_t
        write_end = min(batch_end, write_end)

        # ---- prefetch: the FIRST backward chunk's Gate/Beta ----------------------
        if num_item_chunks > 0:
            chunk_offset = batch_start + (cend - 1) * cfg.b_t
            gGate = cute.domain_offset((chunk_offset,), mGate[None, head_idx])
            gBeta = cute.domain_offset((chunk_offset,), mBeta[None, head_idx])
            gate_idx = gate_index.idx
            gate_index = advance(gate_index, cfg.smem_gate_stages)
            pos_valid = [None] * n_cols
            for col in cutlass.range_constexpr(n_cols):
                pos = lidx + col * cfg.threads_per_warp
                pos_valid[col] = cute.elem_less(chunk_offset + pos, batch_end)

            # ---- Gate load: GMEM -> SMEM (OOB neutral: 1.0 -> log2 = 0.0) --------
            gate_vals = [cutlass.Float32(0.0)] * n_cols
            for col in cutlass.range_constexpr(n_cols):
                pos = lidx + col * cfg.threads_per_warp
                oob_neutral = (
                    cutlass.Float32(0.0)
                    if cutlass.const_expr(cfg.log_gate)
                    else cutlass.Float32(1.0)
                )
                gate_vals[col] = gGate[pos] if pos_valid[col] else oob_neutral

            if cutlass.const_expr(cfg.safe_gate):
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
                    n = nvvm.shfl_sync(0xFFFFFFFF, gate_vals[col], offset, 0, kind=nvvm.Shfl.UP)
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

            for col in cutlass.range_constexpr(n_cols):
                pos = lidx + col * cfg.threads_per_warp
                sCumsumlog[pos, 0, gate_idx] = gate_vals[col]
                sCumprod[pos, 0, gate_idx] = cute.math.exp2(gate_vals[col], fastmath=True)

            bars.mb_gate_ready[gate_idx].arrive()

            # ---- Beta load: GMEM -> SMEM (per-element cp.async) ------------------
            beta_idx = beta_index.idx
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
                nvvm.cp_async_mbarrier_arrive(bars.mb_beta_ready[beta_idx].smem_ptr, noinc=True)

        for rev_idx in cutlass.range(num_item_chunks):
            # ---- prefetch the NEXT chunk's Gate/Beta -----------------------------
            if rev_idx + 1 < num_item_chunks:
                chunk_offset = batch_start + (cend - 2 - rev_idx) * cfg.b_t
                gGate = cute.domain_offset((chunk_offset,), mGate[None, head_idx])
                gBeta = cute.domain_offset((chunk_offset,), mBeta[None, head_idx])
                gate_idx = gate_index.idx
                gate_index = advance(gate_index, cfg.smem_gate_stages)
                pos_valid = [None] * n_cols
                for col in cutlass.range_constexpr(n_cols):
                    pos = lidx + col * cfg.threads_per_warp
                    pos_valid[col] = cute.elem_less(chunk_offset + pos, batch_end)

                gate_vals = [cutlass.Float32(0.0)] * n_cols
                for col in cutlass.range_constexpr(n_cols):
                    pos = lidx + col * cfg.threads_per_warp
                    oob_neutral = (
                        cutlass.Float32(0.0)
                        if cutlass.const_expr(cfg.log_gate)
                        else cutlass.Float32(1.0)
                    )
                    gate_vals[col] = gGate[pos] if pos_valid[col] else oob_neutral

                if cutlass.const_expr(cfg.safe_gate):
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

                for col in cutlass.range_constexpr(n_cols):
                    pos = lidx + col * cfg.threads_per_warp
                    sCumsumlog[pos, 0, gate_idx] = gate_vals[col]
                    sCumprod[pos, 0, gate_idx] = cute.math.exp2(gate_vals[col], fastmath=True)

                bars.mb_gate_ready[gate_idx].arrive()

                beta_idx = beta_index.idx
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

            # ---- store-ready wait + in-place store back --------------------------
            st_offset = batch_start + (cend - 1 - rev_idx) * cfg.b_t
            gGate_st = cute.domain_offset((st_offset,), mDgate[None, head_idx])
            gBeta_st = cute.domain_offset((st_offset,), mDbeta[None, head_idx])
            g_st_idx = gate_store_index.idx
            bars.mb_gate_done[g_st_idx].wait(gate_store_index.phase)
            gate_store_index = advance(gate_store_index, cfg.smem_gate_stages)
            dgate_vals = [cutlass.Float32(0.0)] * n_cols
            for col in cutlass.range_constexpr(n_cols):
                pos = lidx + col * cfg.threads_per_warp
                dgate_vals[col] = sCumsumlog[pos, 0, g_st_idx]
            for offset in [1, 2, 4, 8, 16]:
                for col in cutlass.range_constexpr(n_cols):
                    n = nvvm.shfl_sync(
                        0xFFFFFFFF, dgate_vals[col], offset, 31, kind=nvvm.Shfl.DOWN
                    )
                    if lidx < cfg.threads_per_warp - offset:
                        dgate_vals[col] = dgate_vals[col] + n
            for col in cutlass.range_constexpr(n_cols - 1):
                rev_col = cutlass.const_expr(n_cols - 2 - col)
                later_total = nvvm.shfl_sync(
                    0xFFFFFFFF, dgate_vals[rev_col + 1], 0, 0, kind=nvvm.Shfl.IDX
                )
                dgate_vals[rev_col] = dgate_vals[rev_col] + later_total
            for col in cutlass.range_constexpr(n_cols):
                pos = lidx + col * cfg.threads_per_warp
                if cute.elem_less(st_offset + pos, write_end):
                    gGate_st[pos] = dgate_vals[col]
            b_st_idx = beta_store_index.idx
            bars.mb_beta_done[b_st_idx].wait(beta_store_index.phase)
            beta_store_index = advance(beta_store_index, cfg.smem_beta_stages)
            for col in cutlass.range_constexpr(n_cols):
                pos = lidx + col * cfg.threads_per_warp
                if cute.elem_less(st_offset + pos, write_end):
                    if cutlass.const_expr(cfg.beta_sigmoid):
                        gBeta_st[pos] = sBeta[pos, 0, b_st_idx].to(mDbeta.element_type)
                    else:
                        gBeta_st[pos] = sBeta[pos, 0, b_st_idx]
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


@cute.jit
def mma_warp(
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
    sV,
    sV_kmaj,
    sdO,
    sdO_kmaj,
    sState,
    sState_kmaj,
    sTinv,
    sTinv_trans,
    sA,
    sA_trans,
    sDa,
    sDa_trans,
    sDstate,
    sDm,
    sDm_trans,
    sdV_kmaj,
    sSched,
    bars,
):
    """MMA issuer role (warp 8): persistent scheduler loop issuing every
    tcgen05 GEMM."""
    elect_one = nvvm.elect_sync()

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    kk_acc_index = PipelineState.start(phase=0)
    dk_total_index = PipelineState.start(phase=1)
    du_scale_index = PipelineState.start(phase=0)
    dk_scale_index = PipelineState.start(phase=0)
    dk_attn_index = PipelineState.start(phase=0)
    dstate_acc_index = PipelineState.start(phase=0 if cfg.use_dstate_in else 1)
    k_index = PipelineState.start(phase=0)
    q_index = PipelineState.start(phase=0)
    state_index = PipelineState.start(phase=0)
    v_index = PipelineState.start(phase=0)
    tinv_index = PipelineState.start(phase=0)
    do_index = PipelineState.start(phase=0)
    y_ready_index = PipelineState.start(phase=0)
    u_index = PipelineState.start(phase=0)
    da_ready_index = PipelineState.start(phase=0)
    dv_ready_index = PipelineState.start(phase=0)
    dm_index = PipelineState.start(phase=0)
    dstate_smem_index = PipelineState.start(phase=0)
    dq_scale_index = PipelineState.start(phase=0)
    dq_total_index = PipelineState.start(phase=1)
    a_index = PipelineState.start(phase=0)
    do_prime_inp_ready = PipelineState.start(phase=0)
    du_inp_ready = PipelineState.start(phase=0)
    dyp_inp_ready = PipelineState.start(phase=0)
    dstate_inp_index = PipelineState.start(phase=0)

    nvvm.tcgen05_alloc(tmem_base_slot, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_alloc_barrier_id,
        thread_count=cfg.tmem_alloc_barrier_threads,
    )
    tmem_base = tmem_base_slot.load()

    # ---- chunk-invariant GEMM descriptors ----------------------------------------
    bpe = cfg.io_dtype.width // 8
    idesc_qk = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.b_t,
    )
    bmm_qk_desc = MmaDesc(
        M=cfg.b_t,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_qk,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    tmem_shared_acc_col = tmem_base + cfg.tmem_shared_acc_offset
    tmem_shared_inp_col = tmem_base + cfg.tmem_shared_inp_offset
    SHARED_INP_STAGE_COLS = cfg.b_t // 2
    tmem_do_prime_col = tmem_shared_inp_col
    tmem_du_col = tmem_shared_inp_col + SHARED_INP_STAGE_COLS
    tmem_dyp_col = tmem_du_col
    ACC_STAGE_COLS = cfg.b_t
    tmem_acc_a = tmem_shared_acc_col
    tmem_acc_b = tmem_shared_acc_col + ACC_STAGE_COLS
    tmem_kk_col = tmem_acc_a
    tmem_k_state_col = tmem_acc_a
    tmem_dy_col = tmem_acc_a
    tmem_dm_core_col = tmem_acc_a
    tmem_a_col = tmem_acc_b
    tmem_u_col = tmem_acc_b
    tmem_da_col = tmem_acc_b
    tmem_dk_state_path_col = tmem_shared_inp_col

    idesc_dv = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_dv_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=idesc_dv,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    tmem_dstate_inp_col = tmem_base + cfg.tmem_dstate_inp_offset
    tmem_dvdk_acc_col = tmem_base + cfg.tmem_dvdk_acc_offset
    DSTATE_INP_STAGE_COLS = cfg.d_k // 2

    idesc_k_state = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        a_major=1,
    )
    bmm_k_state_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=True,
        cta_group=1,
        idesc=idesc_k_state,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_k_state_kmaj = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_k_state_kmaj_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=idesc_k_state_kmaj,
        kind=nvvm.Tcgen05MMAKind.F16,
    )

    idesc_du = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        a_major=1,
        b_major=1,
    )
    bmm_du_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=True,
        cta_group=1,
        idesc=idesc_du,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dstate_upd = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_v,
        m_dim=cfg.d_k,
        b_major=1,
    )
    bmm_dstate_upd_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.d_v,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=idesc_dstate_upd,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    SHARED_INP_STAGE_COLS = cfg.b_t // 2
    tmem_do_prime_col = tmem_shared_inp_col
    tmem_du_col = tmem_shared_inp_col + SHARED_INP_STAGE_COLS
    tmem_dyp_col = tmem_du_col
    tmem_shared_inp_col = tmem_base + cfg.tmem_shared_inp_offset
    tmem_dstate_acc_col = tmem_base + cfg.tmem_dstate_acc_offset
    tmem_y_col = tmem_base + cfg.tmem_y_offset

    idesc_dy = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        b_major=1,
    )
    bmm_dy_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=False,
        cta_group=1,
        idesc=idesc_dy,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_da = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.b_t,
    )
    bmm_da_desc = MmaDesc(
        M=cfg.b_t,
        N=cfg.b_t,
        K=cfg.d_v,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_da,
        kind=nvvm.Tcgen05MMAKind.F16,
    )

    idesc_u = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_u_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=idesc_u,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dqdk_inter_at = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
    )
    bmm_dqdk_inter_at_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.d_v,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=True,
        cta_group=1,
        idesc=idesc_dqdk_inter_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dka = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
        b_major=1,
    )
    bmm_dka_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=True,
        cta_group=1,
        idesc=idesc_dka,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dqa = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
    )
    bmm_dqa_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=True,
        cta_group=1,
        idesc=idesc_dqa,
        kind=nvvm.Tcgen05MMAKind.F16,
    )

    do_prime_inp_ptr = nvvm.make_tmem_ptr(tmem_do_prime_col, cutlass.Int8)
    y_inp_ptr = nvvm.make_tmem_ptr(tmem_y_col, cutlass.Int8)
    du_inp_ptr = nvvm.make_tmem_ptr(tmem_du_col, cutlass.Int8)
    dyp_inp_ptr = nvvm.make_tmem_ptr(tmem_dyp_col, cutlass.Int8)
    kk_acc_ptr = nvvm.make_tmem_ptr(tmem_kk_col, cutlass.Float32)
    a_acc_ptr = nvvm.make_tmem_ptr(tmem_a_col, cutlass.Float32)
    k_state_acc_ptr = nvvm.make_tmem_ptr(tmem_k_state_col, cutlass.Float32)
    u_acc_ptr = nvvm.make_tmem_ptr(tmem_u_col, cutlass.Float32)
    dy_acc_ptr = nvvm.make_tmem_ptr(tmem_dy_col, cutlass.Float32)
    da_acc_ptr = nvvm.make_tmem_ptr(tmem_da_col, cutlass.Float32)
    dm_core_acc_ptr = nvvm.make_tmem_ptr(tmem_dm_core_col, cutlass.Float32)
    dk_state_path_acc_ptr = nvvm.make_tmem_ptr(tmem_dk_state_path_col, cutlass.Float32)
    dstate_acc_ptr = nvvm.make_tmem_ptr(tmem_dstate_acc_col, cutlass.Float32)
    dq_acc_ptr = nvvm.make_tmem_ptr(tmem_dstate_inp_col, cutlass.Float32)
    dvdk_acc_ptr = nvvm.make_tmem_ptr(tmem_dvdk_acc_col, cutlass.Float32)

    # ---- warp-top descriptors (1-stage tiles are loop-constant; K advances) ----
    d_q0 = sQ[0].desc()
    d_k0 = sK[0].desc()
    d_k_trans0 = sK_trans[0].desc()
    d_q_trans0 = sQ_trans[0].desc()
    d_do0 = sdO[0].desc()
    d_do_kmaj0 = sdO_kmaj[0].desc()
    d_state0 = sState[0].desc()
    d_state_kmaj0 = sState_kmaj[0].desc()
    d_tinv0 = sTinv[0].desc()
    d_tinv_trans0 = sTinv_trans[0].desc()
    d_a_trans0 = sA_trans[0].desc()
    d_v_kmaj0 = sV_kmaj[0].desc()
    d_dv_kmaj0 = sdV_kmaj[0].desc()
    d_da0 = sDa[0].desc()
    d_da_trans0 = sDa_trans[0].desc()
    d_dm0 = sDm[0].desc()
    d_dm_trans0 = sDm_trans[0].desc()
    d_dstate0 = sDstate[0].desc()
    K_STAGE_BYTES = (cfg.k_cosize // cfg.smem_k_stages) * (cfg.io_dtype.width // 8)

    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
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
        num_item_chunks = cend - wstart

        # ---- chunks NT-2 .. 0 (backward): full body ------------------------------
        for rev_idx in cutlass.range(num_item_chunks):
            chunk_idx = cend - 1 - rev_idx
            have_dstate = (
                cutlass.Boolean(True) if cutlass.const_expr(cfg.use_dstate_in) else rev_idx > 0
            )

            # ---- KK = K(S) @ K^T -------------------------------------------------
            k_idx = k_index.idx
            bars.mb_k_ready[k_idx].wait(k_index.phase)
            k_index = advance(k_index, cfg.smem_k_stages)

            desc_k = d_k0.advance_start_address(k_idx * K_STAGE_BYTES)
            mma_ss(
                bmm_qk_desc,
                desc_k,
                desc_k,
                kk_acc_ptr,
                accumulate=False,
            )

            if elect_one:
                bars.mb_kk_acc_ready[0].arrive(cta_group=1)

            # ---- QK = Q(S) @ K^T -------------------------------------------------
            q_idx = q_index.idx
            bars.mb_q_ready[q_idx].wait(q_index.phase)
            q_index = advance(q_index, cfg.smem_q_stages)

            desc_q = d_q0
            mma_ss(
                bmm_qk_desc,
                desc_q,
                desc_k,
                a_acc_ptr,
                accumulate=False,
            )

            if elect_one:
                bars.mb_a_acc_ready[0].arrive(cta_group=1)

            # ---- k_state = state^T(S) @ K^T -----------------------------------------------
            state_idx = state_index.idx
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_state_ready[state_idx].wait(state_index.phase)
                state_index = advance(state_index, cfg.smem_state_stages)
                bars.mb_kk_acc_done[0].wait(kk_acc_index.phase)
                kk_acc_index = advance(kk_acc_index, 1)
                desc_state = d_state_kmaj0
                mma_ss(
                    bmm_k_state_kmaj_desc,
                    desc_state,
                    desc_k,
                    k_state_acc_ptr,
                    accumulate=False,
                )
                if elect_one:
                    bars.mb_k_state_acc_ready[0].arrive(cta_group=1)

            # ---- dV inter = dstate^T(T) @ K ------------------------------------------
            dstate_inp_idx = dstate_inp_index.idx
            if have_dstate:
                bars.mb_dstate_inp_ready[dstate_inp_idx].wait(dstate_inp_index.phase)
                dstate_inp_index = advance(dstate_inp_index, cfg.tmem_dstate_inp_stages)
            bars.mb_dk_total_acc_done[0].wait(dk_total_index.phase)
            dk_total_index = advance(dk_total_index, 1)

            if have_dstate:
                dstate_a_ptr = nvvm.make_tmem_ptr(
                    tmem_dstate_inp_col + dstate_inp_idx * DSTATE_INP_STAGE_COLS, cutlass.Int8
                )
                for sub in cutlass.range_constexpr(bmm_dv_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_dv_desc.sps_B):
                        mma_ts_step(
                            bmm_dv_desc,
                            dstate_a_ptr.subview(
                                sub * bmm_dv_desc.sps_B * bmm_dv_desc.tmem_advance_A
                            ),
                            desc_k + sub * (bmm_dv_desc.smem_subtile_B >> 4),
                            dvdk_acc_ptr,
                            k,
                            cutlass.Boolean(sub + k > 0),
                        )
                if elect_one:
                    bars.mb_du_scale_acc_ready[0].arrive(cta_group=1)
                    bars.mb_dstate_inp_done[dstate_inp_idx].arrive(cta_group=1)

            # ---- dQ inter = state(S) @ dO^T ------------------------------------------
            do_idx = do_index.idx
            bars.mb_do_ready[do_idx].wait(do_index.phase)
            do_index = advance(do_index, cfg.smem_do_stages)
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_dq_acc_total_done[0].wait(dq_total_index.phase)
                dq_total_index = advance(dq_total_index, 1)
                desc_state_kmaj_dq_inter = d_state0
                desc_do_kmaj_dq_inter = d_do_kmaj0
                mma_ss(
                    bmm_dqdk_inter_at_desc,
                    desc_state_kmaj_dq_inter,
                    desc_do_kmaj_dq_inter,
                    dq_acc_ptr,
                    accumulate=False,
                )
                if elect_one:
                    bars.mb_dq_acc_scale_ready[0].arrive(cta_group=1)

            # ---- dU intra += dO^T(S) @ A -----------------------------------------
            du_a_idx = a_index.idx
            bars.mb_a_ready[du_a_idx].wait(a_index.phase)
            a_index = advance(a_index, cfg.smem_a_stages)
            if have_dstate:
                bars.mb_du_scale_acc_done[0].wait(du_scale_index.phase)
                du_scale_index = advance(du_scale_index, 1)

            desc_do_mnmaj = d_do0
            desc_a_t = d_a_trans0
            mma_ss(
                bmm_du_desc,
                desc_do_mnmaj,
                desc_a_t,
                dvdk_acc_ptr,
                accumulate=have_dstate,
            )
            if elect_one:
                bars.mb_du_total_acc_ready[0].arrive(cta_group=1)

            # ---- dstate update += dO'^T(T) @ Q ---------------------------------------
            bars.mb_do_prime_inp_ready[0].wait(do_prime_inp_ready.phase)
            do_prime_inp_ready = advance(do_prime_inp_ready, 1)
            dstate_idx = dstate_acc_index.idx
            bars.mb_dstate_scale_acc_done[dstate_idx].wait(dstate_acc_index.phase)
            dstate_acc_index = advance(dstate_acc_index, cfg.tmem_dstate_acc_stages)

            desc_q_t = d_q_trans0
            for sub in cutlass.range_constexpr(bmm_dstate_upd_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_dstate_upd_desc.sps_B):
                    mma_ts_step(
                        bmm_dstate_upd_desc,
                        do_prime_inp_ptr.subview(
                            sub * bmm_dstate_upd_desc.sps_B * bmm_dstate_upd_desc.tmem_advance_A
                        ),
                        desc_q_t + sub * (bmm_dstate_upd_desc.smem_subtile_B >> 4),
                        dstate_acc_ptr,
                        k,
                        cutlass.Boolean(True) if cutlass.const_expr(sub + k > 0) else have_dstate,
                    )

            # ---- U^T recompute = Y^T(T) @ T^T ------------------------------------
            tinv_idx = tinv_index.idx
            bars.mb_t_inv_ready[tinv_idx].wait(tinv_index.phase)
            tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
            bars.mb_y_ready[0].wait(y_ready_index.phase)
            y_ready_index = advance(y_ready_index, 1)
            v_idx = v_index.idx
            v_index = advance(v_index, cfg.smem_v_stages)

            desc_tinv = d_tinv0
            for sub in cutlass.range_constexpr(bmm_u_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_u_desc.sps_B):
                    mma_ts_step(
                        bmm_u_desc,
                        y_inp_ptr.subview(sub * bmm_u_desc.sps_B * bmm_u_desc.tmem_advance_A),
                        desc_tinv + sub * (bmm_u_desc.smem_subtile_B >> 4),
                        u_acc_ptr,
                        k,
                        cutlass.Boolean(sub + k > 0),
                    )
            if elect_one:
                bars.mb_u_acc_ready[0].arrive(cta_group=1)

            # ---- dY = dU^T(T) @ T ------------------------------------------------
            bars.mb_du_inp_ready[0].wait(du_inp_ready.phase)
            du_inp_ready = advance(du_inp_ready, 1)

            desc_tinv_t = d_tinv_trans0
            for sub in cutlass.range_constexpr(bmm_dy_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_dy_desc.sps_B):
                    mma_ts_step(
                        bmm_dy_desc,
                        du_inp_ptr.subview(sub * bmm_dy_desc.sps_B * bmm_dy_desc.tmem_advance_A),
                        desc_tinv_t + sub * (bmm_dy_desc.smem_subtile_B >> 4),
                        dy_acc_ptr,
                        k,
                        cutlass.Boolean(sub + k > 0),
                    )
            if elect_one:
                bars.mb_dy_acc_ready[0].arrive(cta_group=1)

            # ---- dK_inter = dstate_entry(S) @ U^T ------------------------------------
            bars.mb_u_ready[0].wait(u_index.phase)
            u_index = advance(u_index, 1)
            if have_dstate:
                bars.mb_dstate_smem_ready[0].wait(dstate_smem_index.phase)
                dstate_smem_index = advance(dstate_smem_index, 1)

                desc_dstate = d_dstate0
                desc_u_kmaj_dk_inter = d_v_kmaj0
                mma_ss(
                    bmm_dqdk_inter_at_desc,
                    desc_dstate,
                    desc_u_kmaj_dk_inter,
                    dvdk_acc_ptr,
                    accumulate=False,
                )
                if elect_one:
                    bars.mb_dk_scale_acc_ready[0].arrive(cta_group=1)

            # ---- dA_eff = dO(S) @ U^T --------------------------------------------
            desc_do_kmaj_da = d_do_kmaj0
            desc_u_kmaj_da = d_v_kmaj0
            mma_ss(
                bmm_da_desc,
                desc_do_kmaj_da,
                desc_u_kmaj_da,
                da_acc_ptr,
                accumulate=False,
            )
            if elect_one:
                bars.mb_da_acc_ready[0].arrive(cta_group=1)
                bars.mb_do_mma_done[do_idx].arrive(cta_group=1)

            # ---- dM core = dY(S) @ U^T -------------------------------------------
            dv_ready_idx = dv_ready_index.idx
            bars.mb_dv_tmastg_ready[dv_ready_idx].wait(dv_ready_index.phase)
            dv_ready_index = advance(dv_ready_index, cfg.smem_dv_stages)

            desc_dy_kmaj_dm = d_dv_kmaj0
            desc_u_kmaj_dm = d_v_kmaj0
            mma_ss(
                bmm_da_desc,
                desc_dy_kmaj_dm,
                desc_u_kmaj_dm,
                dm_core_acc_ptr,
                accumulate=False,
            )
            if elect_one:
                bars.mb_dm_acc_ready[0].arrive(cta_group=1)
                bars.mb_v_mma_done[v_idx].arrive(cta_group=1)

            # ---- dstate update-term += dY'^T(T) @ K --------------------------------------
            bars.mb_dyp_inp_ready[0].wait(dyp_inp_ready.phase)
            dyp_inp_ready = advance(dyp_inp_ready, 1)

            desc_k_t = d_k_trans0.advance_start_address(k_idx * K_STAGE_BYTES)
            for sub in cutlass.range_constexpr(bmm_dstate_upd_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_dstate_upd_desc.sps_B):
                    mma_ts_step(
                        bmm_dstate_upd_desc,
                        dyp_inp_ptr.subview(
                            sub * bmm_dstate_upd_desc.sps_B * bmm_dstate_upd_desc.tmem_advance_A
                        ),
                        desc_k_t + sub * (bmm_dstate_upd_desc.smem_subtile_B >> 4),
                        dstate_acc_ptr,
                        k,
                        cutlass.Boolean(True),
                    )
            if elect_one:
                bars.mb_dstate_acc_ready[dstate_idx].arrive(cta_group=1)

            # ---- dK attn += Q^T(S) @ dA ------------------------------------------
            if have_dstate:
                bars.mb_dk_scale_acc_done[0].wait(dk_scale_index.phase)
                dk_scale_index = advance(dk_scale_index, 1)
            bars.mb_da_ready[0].wait(da_ready_index.phase)
            da_ready_index = advance(da_ready_index, 1)

            desc_q_mnmaj_dk_attn = d_q_trans0
            desc_da_t = d_da_trans0
            mma_ss(
                bmm_dka_desc,
                desc_q_mnmaj_dk_attn,
                desc_da_t,
                dvdk_acc_ptr,
                accumulate=have_dstate,
            )
            if elect_one:
                bars.mb_dk_attn_acc_ready[0].arrive(cta_group=1)
                bars.mb_q_mma_done[q_idx].arrive(cta_group=1)

            # ---- dQ attn += K^T(S) @ dA ------------------------------------------
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_dq_acc_scale_done[0].wait(dq_scale_index.phase)
                dq_scale_index = advance(dq_scale_index, 1)
            if chunk_idx < FIRST_STATE_CHUNK:
                bars.mb_dq_acc_total_done[0].wait(dq_total_index.phase)
                dq_total_index = advance(dq_total_index, 1)
            desc_k_mnmaj_dq_attn = d_k_trans0.advance_start_address(k_idx * K_STAGE_BYTES)
            desc_da = d_da0
            if chunk_idx >= FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_dqa_desc,
                    desc_k_mnmaj_dq_attn,
                    desc_da,
                    dq_acc_ptr,
                    accumulate=True,
                )
            if chunk_idx < FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_dqa_desc,
                    desc_k_mnmaj_dq_attn,
                    desc_da,
                    dq_acc_ptr,
                    accumulate=False,
                )
            if elect_one:
                bars.mb_dq_acc_total_ready[0].arrive(cta_group=1)
                bars.mb_a_done[du_a_idx].arrive(cta_group=1)

            # ---- dK state-path = state(S) @ dY^T -----------------------------------------
            if chunk_idx >= FIRST_STATE_CHUNK:
                desc_state_kmaj_spath = d_state0
                desc_dy_kmaj_spath = d_dv_kmaj0
                mma_ss(
                    bmm_dqdk_inter_at_desc,
                    desc_state_kmaj_spath,
                    desc_dy_kmaj_spath,
                    dk_state_path_acc_ptr,
                    accumulate=False,
                )
                if elect_one:
                    bars.mb_dk_state_path_acc_ready[0].arrive(cta_group=1)
                    bars.mb_state_mma_done[state_idx].arrive(cta_group=1)
            if elect_one:
                bars.mb_sdv_done[0].arrive(cta_group=1)

            # ---- dK dM-terms += K^T(S) @ dM^T + K^T(S) @ dM ----------------------
            bars.mb_dm_acc_done[0].wait(dm_index.phase)
            dm_index = advance(dm_index, 1)
            bars.mb_dk_attn_acc_done[0].wait(dk_attn_index.phase)
            dk_attn_index = advance(dk_attn_index, 1)
            desc_k_mnmaj_dm_terms = d_k_trans0.advance_start_address(k_idx * K_STAGE_BYTES)
            desc_dm_t = d_dm_trans0
            mma_ss(
                bmm_dka_desc,
                desc_k_mnmaj_dm_terms,
                desc_dm_t,
                dvdk_acc_ptr,
                accumulate=True,
            )
            desc_dm = d_dm0
            mma_ss(
                bmm_dqa_desc,
                desc_k_mnmaj_dm_terms,
                desc_dm,
                dvdk_acc_ptr,
                accumulate=True,
            )
            if elect_one:
                bars.mb_dk_total_acc_ready[0].arrive(cta_group=1)
                bars.mb_k_mma_done[k_idx].arrive(cta_group=1)

        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)

    bars.mb_dk_total_acc_done[0].wait(dk_total_index.phase)

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
    sQ_raw,
    sK_raw,
    sV_raw,
    sdO_raw,
    sState_raw,
    desc_q_base,
    desc_k_base,
    desc_v_base,
    desc_do_base,
    desc_checkpoint_base,
    sSched,
    mSched,
    bars,
):
    """TMA-LDG warp role (warp 9): persistent tile loop issuing every
    Q/K/V/dO/state TMA load."""
    elect_one = nvvm.elect_sync()
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    q_index = PipelineState.start(phase=1)
    k_index = PipelineState.start(phase=1)
    v_index = PipelineState.start(phase=1)
    do_index = PipelineState.start(phase=1)
    state_index = PipelineState.start(phase=1)
    sched_state = PipelineState.start(phase=1)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1

    bpe = cfg.io_dtype.width // 8
    granule_elems = 128 // bpe
    bt = cfg.b_t
    q_stage_elems = cfg.q_cosize // cfg.smem_q_stages
    k_stage_elems = cfg.k_cosize // cfg.smem_k_stages
    sQ_tma = SmemTile(
        base=sQ_raw,
        elems_per_stage=q_stage_elems,
        stages=cfg.smem_q_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granule_elems,
        tma_subtile_stride_elems=bt * granule_elems,
    )
    sK_tma = SmemTile(
        base=sK_raw,
        elems_per_stage=k_stage_elems,
        stages=cfg.smem_k_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granule_elems,
        tma_subtile_stride_elems=bt * granule_elems,
    )
    sV_tma = SmemTile(
        base=sV_raw,
        elems_per_stage=0,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granule_elems,
        tma_subtile_stride_elems=4096,
    )
    sdO_tma = SmemTile(
        base=sdO_raw,
        elems_per_stage=0,
        stages=1,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=2,
        tma_granu_elems=granule_elems,
        tma_subtile_stride_elems=4096,
    )
    sCheckpoint_tma = SmemTile(
        base=sState_raw,
        elems_per_stage=(cfg.state_cosize // cfg.smem_state_stages),
        stages=cfg.smem_state_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=cfg.d_v // 64,
        tma_granu_elems=64,
        tma_subtile_stride_elems=cfg.d_k * 64,
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

        head_o = head_idx
        head_q = head_idx if cfg.q_ratio == 1 else head_idx // cutlass.Int32(cfg.q_ratio)
        head_k = head_idx if cfg.k_ratio == 1 else head_idx // cutlass.Int32(cfg.k_ratio)
        head_v = head_idx if cfg.v_ratio == 1 else head_idx // cutlass.Int32(cfg.v_ratio)
        slot = batch_idx * desc_qwords
        desc_q_slot = (desc_q_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_do_slot = (desc_do_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_checkpoint_slot = (desc_checkpoint_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_v_slot)
            tma_tensormap_acquire(desc_do_slot)
            tma_tensormap_acquire(desc_checkpoint_slot)
        use_packed_coords = batch_end == packed_tokens

        for rev_idx in cutlass.range(cend - wstart):
            chunk_idx = cend - 1 - rev_idx
            tok_coord = chunk_idx * cutlass.Int32(cfg.b_t)
            input_tok_coord = batch_start + tok_coord if use_packed_coords else tok_coord

            # ---- K load ----------------------------------------------------------
            k_idx = k_index.idx
            bars.mb_k_mma_done[k_idx].wait(k_index.phase)
            bars.mb_k_cg0_done[k_idx].wait(k_index.phase)
            k_index = advance(k_index, cfg.smem_k_stages)
            if elect_one:
                bars.mb_k_ready[k_idx].arrive(n_bytes=cfg.tma_k_bytes)
            k_slice = tma_slice_runtime_desc(
                desc_k_slot, cutlass.Int32(0), head_k, input_tok_coord
            )
            tma_load_tile(sK_tma[k_idx], k_slice, bars.mb_k_ready[k_idx].smem_ptr, acquire=False)

            # ---- Q load ----------------------------------------------------------
            q_idx = q_index.idx
            bars.mb_q_mma_done[q_idx].wait(q_index.phase)
            bars.mb_q_cg1_done[q_idx].wait(q_index.phase)
            q_index = advance(q_index, cfg.smem_q_stages)
            if elect_one:
                bars.mb_q_ready[q_idx].arrive(n_bytes=cfg.tma_q_bytes)
            q_slice = tma_slice_runtime_desc(
                desc_q_slot, cutlass.Int32(0), head_q, input_tok_coord
            )
            tma_load_tile(sQ_tma[q_idx], q_slice, bars.mb_q_ready[q_idx].smem_ptr, acquire=False)

            # ---- V load ----------------------------------------------------------
            v_idx = v_index.idx
            bars.mb_v_mma_done[v_idx].wait(v_index.phase)
            v_index = advance(v_index, cfg.smem_v_stages)
            if elect_one:
                bars.mb_v_ready[v_idx].arrive(n_bytes=cfg.tma_v_bytes)
            v_slice = tma_slice_runtime_desc(
                desc_v_slot, cutlass.Int32(0), head_v, input_tok_coord
            )
            tma_load_tile(sV_tma[v_idx], v_slice, bars.mb_v_ready[v_idx].smem_ptr, acquire=False)

            # ---- dO load ---------------------------------------------------------
            do_idx = do_index.idx
            bars.mb_do_mma_done[do_idx].wait(do_index.phase)
            do_index = advance(do_index, cfg.smem_do_stages)
            if elect_one:
                bars.mb_do_ready[do_idx].arrive(n_bytes=cfg.tma_do_bytes)
            do_slice = tma_slice_runtime_desc(
                desc_do_slot, cutlass.Int32(0), head_o, input_tok_coord
            )
            tma_load_tile(
                sdO_tma[do_idx], do_slice, bars.mb_do_ready[do_idx].smem_ptr, acquire=False
            )

            # ---- entering state ----------
            if chunk_idx >= FIRST_STATE_CHUNK:
                state_idx = state_index.idx
                bars.mb_state_mma_done[state_idx].wait(state_index.phase)
                state_index = advance(state_index, cfg.smem_state_stages)
                if elect_one:
                    bars.mb_state_ready[state_idx].arrive(n_bytes=cfg.tma_state_bytes)
                checkpoint_slice = tma_slice_runtime_desc(
                    desc_checkpoint_slot, cutlass.Int32(0), cutlass.Int32(0), chunk_idx, head_o
                )
                tma_load_tile(
                    sCheckpoint_tma[state_idx],
                    checkpoint_slice,
                    bars.mb_state_ready[state_idx].smem_ptr,
                    acquire=False,
                )

        tile_idx, sched_state = sched_publish_next(
            cfg, bars, sSched, mSched, sched_state, tile_idx, num_ctas
        )

    for _ in range(cfg.smem_q_stages):
        bars.mb_q_mma_done[q_index.idx].wait(q_index.phase)
        bars.mb_q_cg1_done[q_index.idx].wait(q_index.phase)
        q_index = advance(q_index, cfg.smem_q_stages)
    for _ in range(cfg.smem_k_stages):
        bars.mb_k_mma_done[k_index.idx].wait(k_index.phase)
        bars.mb_k_cg0_done[k_index.idx].wait(k_index.phase)
        k_index = advance(k_index, cfg.smem_k_stages)
    for _ in range(cfg.smem_v_stages):
        bars.mb_v_mma_done[v_index.idx].wait(v_index.phase)
        v_index = advance(v_index, cfg.smem_v_stages)
    for _ in range(cfg.smem_do_stages):
        bars.mb_do_mma_done[do_index.idx].wait(do_index.phase)
        do_index = advance(do_index, cfg.smem_do_stages)


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
    sCumprod,
    sBeta,
    sTinv,
    sKK,
    sA,
    sDa,
    sDm,
    sK,
    sdQ,
    sDstate,
    sstate_flat,
    sdstate_flat,
    sSched,
    bars,
):
    """Compute warp-group 0 role (warps 0-3): persistent scheduler loop
    building each chunk's blockwise-inverse T and attention matrices."""

    nvvm.setmaxregister(cfg.num_regs_compute_group_0, nvvm.SetMaxRegisterAction.INCREASE)
    gate_index = PipelineState.start(phase=0)
    beta_index = PipelineState.start(phase=0)
    a_index = PipelineState.start(phase=1)
    tinv_index = PipelineState.start(phase=1)
    da_acc_index = PipelineState.start(phase=0)
    dm_ready_index = PipelineState.start(phase=0)
    cg0_dbeta_index = PipelineState.start(phase=0)

    nvvm.barrier_cta_sync_aligned(
        cfg.tmem_alloc_barrier_id,
        thread_count=cfg.tmem_alloc_barrier_threads,
    )
    tmem_base = tmem_base_slot.load()

    num_threads_cg0 = cfg.threads_per_warp * len(cfg.compute_group_0_warp_ids)
    cg0_tidx = tidx % num_threads_cg0
    warp_id = cg0_tidx // cfg.threads_per_warp
    lane_id = cg0_tidx % cfg.threads_per_warp
    bpe = cfg.io_dtype.width // 8
    num_vals = 32
    FRAG_COLS = 16
    ACC_N_FRAGS = cfg.b_t // FRAG_COLS
    store_row = warp_id * 16 + lane_id % 16
    store_col = (lane_id // 16) * 8
    tmem_warp_row = warp_id * cfg.threads_per_warp
    tmem_shared_acc_col = tmem_base + cfg.tmem_shared_acc_offset
    tmem_shared_inp_col = tmem_base + cfg.tmem_shared_inp_offset
    SHARED_INP_STAGE_COLS = cfg.b_t // 2
    tmem_do_prime_col = tmem_shared_inp_col
    tmem_du_col = tmem_shared_inp_col + SHARED_INP_STAGE_COLS
    tmem_dyp_col = tmem_du_col
    ACC_STAGE_COLS = cfg.b_t
    tmem_acc_a = tmem_shared_acc_col
    tmem_acc_b = tmem_shared_acc_col + ACC_STAGE_COLS
    tmem_kk_col = tmem_acc_a
    tmem_k_state_col = tmem_acc_a
    tmem_dy_col = tmem_acc_a
    tmem_dm_core_col = tmem_acc_a
    tmem_a_col = tmem_acc_b
    tmem_u_col = tmem_acc_b
    tmem_da_col = tmem_acc_b
    tmem_dk_state_path_col = tmem_shared_inp_col
    acc_zero = cfg.acc_dtype(0.0)
    mask_zero = opaque_f32_zero()
    frag_row = cg0_tidx % 8 + (cg0_tidx // 16 % 2) * 8
    frag_col = (cg0_tidx // 8 % 2) * 8 + (cg0_tidx // 32 % 2) * 32
    frag_slab_off = (cg0_tidx // 64) * 4096
    sK_base_p = cute.make_ptr(
        cfg.io_dtype,
        sK[0].base,
        mem_space=cute.AddressSpace.smem,
        assumed_align=cfg.buffer_align_bytes,
    )
    k_stage_elems_cg0 = cfg.k_cosize // cfg.smem_k_stages
    sdQ_base = cute.make_ptr(
        cfg.io_dtype,
        sdQ[0].base,
        mem_space=cute.AddressSpace.smem,
        assumed_align=cfg.buffer_align_bytes,
    )
    sdH_parts_p = cute.make_ptr(
        cfg.io_dtype,
        sDstate[0].base,
        mem_space=cute.AddressSpace.smem,
        assumed_align=cfg.buffer_align_bytes,
    )
    # dGate fold scratch in sKK
    skk_red = cute.make_ptr(
        cutlass.Float32,
        sKK[0].base,
        mem_space=cute.AddressSpace.smem,
        assumed_align=cfg.buffer_align_bytes,
    )
    cg0_k_index = PipelineState.start(phase=0)
    cg0_dgate_index = PipelineState.start(phase=0)
    tmem_dvdk_acc_col = tmem_base + cfg.tmem_dvdk_acc_offset
    tmem_dstate_inp_col = tmem_base + cfg.tmem_dstate_inp_offset
    cg0_kk_ready = PipelineState.start(phase=0)
    cg0_a_ready = PipelineState.start(phase=0)
    cg0_dk_scale_ready = PipelineState.start(phase=0)
    cg0_dq_scale_ready = PipelineState.start(phase=0)
    cg0_dstate_smem_ready = PipelineState.start(phase=0)
    cg0_dk_attn_ready = PipelineState.start(phase=0)
    DSTATE_IN0 = 1 if cfg.use_dstate_in else 0

    tinv_zero_ptr = cute.make_ptr(
        cutlass.Int32,
        sTinv[0].base,
        mem_space=cute.AddressSpace.smem,
        assumed_align=cfg.buffer_align_bytes,
    )
    for z in cutlass.range_constexpr(cfg.t_inv_cosize * bpe // 4 // num_threads_cg0):
        (tinv_zero_ptr + cg0_tidx + z * num_threads_cg0).store(cutlass.Int32(0))

    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
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
        num_item_chunks = cend - wstart

        for chunk_idx in cutlass.range(num_item_chunks):
            # ---- T-pairwise ------------------------------------------------------
            gate_idx = gate_index.idx
            bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)

            row_cs = []
            for r in cutlass.range_constexpr(2):
                row_cs.append(sCumsumlog[warp_id * 16 + lane_id // 4 + r * 8, 0, gate_idx])
            col_cs = []
            for g in cutlass.range_constexpr(8):
                for b in cutlass.range_constexpr(2):
                    col_cs.append(sCumsumlog[(lane_id % 4) * 2 + g * 8 + b, 0, gate_idx])
            decay_t = []
            decay_t_strict = []
            for k in cutlass.range_constexpr(num_vals):
                crow = warp_id * 16 + lane_id // 4 + ((k // 2) % 2) * 8
                ccol = (lane_id % 4) * 2 + ((k // 4) * 8 + k % 2)
                decay_t.append(
                    cute.math.exp2(
                        row_cs[(k // 2) % 2] - col_cs[(k // 4) * 2 + (k % 2)], fastmath=True
                    )
                    if crow >= ccol
                    else mask_zero
                )
                decay_t_strict.append(mask_zero if crow == ccol else decay_t[k])
            last_cs = sCumsumlog[cfg.b_t - 1, 0, gate_idx]
            decay_scale_fp32 = []
            for i in cutlass.range_constexpr(16):
                decay_scale_fp32.append(cute.math.exp2(last_cs - col_cs[i], fastmath=True))
            decay_scale_vals = [decay_scale_fp32[(k // 4) * 2 + (k % 2)] for k in range(num_vals)]
            cumprod_total = sCumprod[sCumprod.shape[0] - 1, 0, gate_idx]

            beta_idx = beta_index.idx
            bars.mb_beta_ready[beta_idx].wait(beta_index.phase)
            beta_index = advance(beta_index, cfg.smem_beta_stages)

            gBeta = []
            for k in cutlass.range_constexpr(num_vals):
                crow = warp_id * 16 + lane_id // 4 + ((k // 2) % 2) * 8
                gBeta.append(sBeta[crow, 0, beta_idx])

            # ---- KK epi:  M_kk[i,j] = W_kk[i,j] * T_strict[i,j] * Beta[i] --------
            tinv_idx = tinv_index.idx
            tinv_index = advance(tinv_index, cfg.smem_t_inv_stages)
            bars.mb_kk_acc_ready[0].wait(cg0_kk_ready.phase)
            cg0_kk_ready = advance(cg0_kk_ready, 1)

            tinv_base = sTinv[tinv_idx].base
            kk_base = sKK[tinv_idx].base
            kk_vec = nvvm.tcgen05_ld(
                "16x256b",
                nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_kk_col, cutlass.Float32),
                num=8,
            )
            kk_pack = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(
                    kk_vec[2 * k],
                    kk_vec[2 * k + 1],
                    decay_t_strict[2 * k],
                    decay_t_strict[2 * k + 1],
                )
                v0, v1 = fmul2(p0, p1, gBeta[2 * k], gBeta[2 * k + 1])
                kk_pack.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    cutlass.inttoptr(
                        kk_base
                        + (
                            store_row * cfg.b_t
                            + smem_64x64_offset(store_row, store_col + c * FRAG_COLS)
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
            if chunk_idx < cend - FIRST_STATE_CHUNK:
                bars.mb_kk_acc_done[0].arrive()

            # ---- A epi:  A[i,j] = W_qk[i,j] * T[i,j] * scale ---------------------
            a_idx = a_index.idx
            a_phase = a_index.phase
            a_index = advance(a_index, cfg.smem_a_stages)
            bars.mb_a_acc_ready[0].wait(cg0_a_ready.phase)
            cg0_a_ready = advance(cg0_a_ready, 1)

            a_base = sA[a_idx].base
            a_vec = nvvm.tcgen05_ld(
                "16x256b",
                nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_a_col, cutlass.Float32),
                num=8,
            )
            a_pack = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(a_vec[2 * k], a_vec[2 * k + 1], decay_t[2 * k], decay_t[2 * k + 1])
                v0, v1 = fmul2(p0, p1, scale, scale)
                a_pack.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            bars.mb_a_done[a_idx].wait(a_phase)
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    cutlass.inttoptr(
                        a_base
                        + (
                            store_row * cfg.b_t
                            + smem_64x64_offset(store_row, store_col + c * FRAG_COLS)
                        )
                        * bpe,
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    ),
                    [a_pack[c * 4 + 0], a_pack[c * 4 + 1], a_pack[c * 4 + 2], a_pack[c * 4 + 3]],
                    nvvm.MMALayout.ROW,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_a_ready[a_idx].arrive()

            # ---- blockwise inverse:  T_inv = -------------------------------------
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )
            if warp_id < 2:
                invert_diagonal_NxN(cfg, kk_base, tinv_base, cg0_tidx // 8, cg0_tidx, 8)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            blockwise_diagonal_8x8_to_16x16(cfg, tinv_base, kk_base, warp_id * 16, lane_id)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            if warp_id < 2:
                blockwise_diagonal_16x16_to_32x32(cfg, tinv_base, kk_base, warp_id * 32, lane_id)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            if warp_id < 2:
                blockwise_diagonal_32x32_to_64x64(cfg, tinv_base, kk_base, warp_id, lane_id)
            nvvm.barrier_cta_sync_aligned(
                cfg.inverse_barrier_id,
                thread_count=cfg.inverse_barrier_threads,
            )

            # ---- Beta column-scaling in place:  T_inv[i,j] *= Beta[j] ------------
            beta_col = []
            for k in cutlass.range_constexpr(num_vals):
                beta_col.append(sBeta[(lane_id % 4) * 2 + ((k // 4) * 8 + k % 2), 0, beta_idx])
            tinv_frags = []
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                tinv_frags += list(
                    nvvm.ldmatrix(
                        cutlass.inttoptr(
                            tinv_base
                            + (
                                store_row * cfg.b_t
                                + smem_64x64_offset(store_row, store_col + c * FRAG_COLS)
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
                        tinv_base
                        + (
                            store_row * cfg.b_t
                            + smem_64x64_offset(store_row, store_col + c * FRAG_COLS)
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
            bars.mb_t_inv_ready[tinv_idx].arrive()

            # ---- dQ inter rescale ------------------------------------------------
            if chunk_idx < cend - FIRST_STATE_CHUNK:
                cumprod_fp32 = []
                for g in cutlass.range_constexpr(8):
                    for b in cutlass.range_constexpr(2):
                        cumprod_fp32.append(sCumprod[(lane_id % 4) * 2 + g * 8 + b, 0, gate_idx])
                cumprod_vals = [cumprod_fp32[(k // 4) * 2 + (k % 2)] for k in range(num_vals)]
                bars.mb_dq_acc_scale_ready[0].wait(cg0_dq_scale_ready.phase)
                cg0_dq_scale_ready = advance(cg0_dq_scale_ready, 1)
                dqi_ptrs = []
                dqi_vecs = []
                for sub in cutlass.range_constexpr(2):
                    dqi_ptrs.append(
                        nvvm.make_tmem_ptr(
                            ((tmem_warp_row + sub * 16) << 16) + tmem_dstate_inp_col,
                            cutlass.Float32,
                        )
                    )
                    dqi_vecs.append(nvvm.tcgen05_ld("16x256b", dqi_ptrs[sub], num=8))
                for sub in cutlass.range_constexpr(2):
                    dqi_scaled = []
                    for j in cutlass.range_constexpr(16):
                        p0, p1 = fmul2(
                            dqi_vecs[sub][2 * j],
                            dqi_vecs[sub][2 * j + 1],
                            cumprod_vals[2 * j],
                            cumprod_vals[2 * j + 1],
                        )
                        s0, s1 = fmul2(p0, p1, scale, scale)
                        dqi_scaled += [s0, s1]
                    nvvm.tcgen05_st(
                        "16x256b",
                        dqi_ptrs[sub],
                        cutlass.Vector.from_elements(tuple(dqi_scaled), cutlass.Float32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dq_acc_scale_done[0].arrive()

            # ---- dGate_last state dot dstate term ----------------------------------------
            if chunk_idx + DSTATE_IN0 >= 1:
                bars.mb_dstate_smem_ready[0].wait(cg0_dstate_smem_ready.phase)
                cg0_dstate_smem_ready = advance(cg0_dstate_smem_ready, 1)
            sdstate_base = sdstate_flat.iterator.toint()
            sstate_base = sstate_flat.iterator.toint()
            state_dot_dstate_lo = [
                opaque_f32_zero(),
                opaque_f32_zero(),
                opaque_f32_zero(),
                opaque_f32_zero(),
            ]
            state_dot_dstate_hi = [
                opaque_f32_zero(),
                opaque_f32_zero(),
                opaque_f32_zero(),
                opaque_f32_zero(),
            ]
            for oct_ in cutlass.range_constexpr(cfg.d_v // 8):
                state_dot_off = (
                    (oct_ // 8) * (cfg.d_k * 64)
                    + cg0_tidx * 64
                    + smem_64x64_offset(cg0_tidx, (oct_ % 8) * 8)
                )
                dstate_frag = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Int32,
                        sdstate_base + state_dot_off * bpe,
                        mem_space=cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout(4),
                ).load()
                state_frag = cute.make_tensor(
                    cute.make_ptr(
                        cutlass.Int32,
                        sstate_base + state_dot_off * bpe,
                        mem_space=cute.AddressSpace.smem,
                        assumed_align=16,
                    ),
                    cute.make_layout(4),
                ).load()
                for w in cutlass.range_constexpr(4):
                    dstate_lo, dstate_hi = f16x2_to_f32(dstate_frag[w], dtype=cfg.io_dtype)
                    state_lo, state_hi = f16x2_to_f32(state_frag[w], dtype=cfg.io_dtype)
                    state_dot_dstate_lo[w], state_dot_dstate_hi[w] = ffma2(
                        dstate_lo,
                        dstate_hi,
                        state_lo,
                        state_hi,
                        state_dot_dstate_lo[w],
                        state_dot_dstate_hi[w],
                    )
            state_dot_dstate = (
                (state_dot_dstate_lo[0] + state_dot_dstate_lo[1])
                + (state_dot_dstate_lo[2] + state_dot_dstate_lo[3])
            ) + (
                (state_dot_dstate_hi[0] + state_dot_dstate_hi[1])
                + (state_dot_dstate_hi[2] + state_dot_dstate_hi[3])
            )
            for off in [1, 2, 4, 8, 16]:
                state_dot_dstate += nvvm.shfl_sync(
                    0xFFFFFFFF, state_dot_dstate, off, 31, kind=nvvm.Shfl.BFLY
                )
            bars.mb_state_dot_dstate_done[0].arrive()

            # ---- dK inter rescale ------------------------------------------------
            cg0_k_idx = cg0_k_index.idx
            cg0_k_index = advance(cg0_k_index, cfg.smem_k_stages)
            k_dot_dk_inter = cutlass.Float32(0.0)
            if chunk_idx + DSTATE_IN0 >= 1:
                bars.mb_dk_scale_acc_ready[0].wait(cg0_dk_scale_ready.phase)
                cg0_dk_scale_ready = advance(cg0_dk_scale_ready, 1)
                k_dot_dk_inter_lo = [
                    opaque_f32_zero(),
                    opaque_f32_zero(),
                    opaque_f32_zero(),
                    opaque_f32_zero(),
                ]
                k_dot_dk_inter_hi = [
                    opaque_f32_zero(),
                    opaque_f32_zero(),
                    opaque_f32_zero(),
                    opaque_f32_zero(),
                ]
                dki_ptrs = []
                dki_vecs = []
                for sub in cutlass.range_constexpr(2):
                    dki_ptrs.append(
                        nvvm.make_tmem_ptr(
                            ((tmem_warp_row + sub * 16) << 16) + tmem_dvdk_acc_col, cutlass.Float32
                        )
                    )
                    dki_vecs.append(nvvm.tcgen05_ld("16x256b", dki_ptrs[sub], num=8))
                for sub in cutlass.range_constexpr(2):
                    dki_scaled = []
                    for j in cutlass.range_constexpr(16):
                        s0, s1 = fmul2(
                            dki_vecs[sub][2 * j],
                            dki_vecs[sub][2 * j + 1],
                            decay_scale_vals[2 * j],
                            decay_scale_vals[2 * j + 1],
                        )
                        dki_scaled += [s0, s1]
                    nvvm.tcgen05_st(
                        "16x256b",
                        dki_ptrs[sub],
                        cutlass.Vector.from_elements(tuple(dki_scaled), cutlass.Float32),
                    )
                    for m0 in cutlass.range_constexpr(4):
                        frag_addr = (
                            frag_slab_off
                            + (frag_row + m0 * 16) * 64
                            + smem_64x64_offset(frag_row + m0 * 16, frag_col + sub * 16)
                        )
                        k_frag = nvvm.ldmatrix(
                            (sK_base_p + cg0_k_idx * k_stage_elems_cg0 + frag_addr).raw_ptr(),
                            4,
                            nvvm.MMALayout.COL,
                        )
                        for i in cutlass.range_constexpr(4):
                            k_lo, k_hi = f16x2_to_f32(k_frag[i], dtype=cfg.io_dtype)
                            k_dot_dk_inter_lo[i], k_dot_dk_inter_hi[i] = ffma2(
                                dki_scaled[8 * m0 + 2 * i],
                                dki_scaled[8 * m0 + 2 * i + 1],
                                k_lo,
                                k_hi,
                                k_dot_dk_inter_lo[i],
                                k_dot_dk_inter_hi[i],
                            )
                nvvm.tcgen05_wait("store")
                bars.mb_dk_scale_acc_done[0].arrive()
                k_dot_dk_inter = (
                    (k_dot_dk_inter_lo[0] + k_dot_dk_inter_lo[1])
                    + (k_dot_dk_inter_lo[2] + k_dot_dk_inter_lo[3])
                ) + (
                    (k_dot_dk_inter_hi[0] + k_dot_dk_inter_hi[1])
                    + (k_dot_dk_inter_hi[2] + k_dot_dk_inter_hi[3])
                )
                for off in [1, 2, 4, 8, 16]:
                    k_dot_dk_inter += nvvm.shfl_sync(
                        0xFFFFFFFF, k_dot_dk_inter, off, 31, kind=nvvm.Shfl.BFLY
                    )

            # ---- dA epilogue -----------------------------------------------------
            bars.mb_da_acc_ready[0].wait(da_acc_index.phase)
            da_acc_index = advance(da_acc_index, 1)

            da_base = sDa[0].base
            da_vec = nvvm.tcgen05_ld(
                "16x256b",
                nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_da_col, cutlass.Float32),
                num=8,
            )
            da_pack = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(
                    da_vec[2 * k], da_vec[2 * k + 1], decay_t[2 * k], decay_t[2 * k + 1]
                )
                v0, v1 = fmul2(p0, p1, scale, scale)
                da_pack.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    cutlass.inttoptr(
                        da_base
                        + (
                            store_row * cfg.b_t
                            + smem_64x64_offset(store_row, store_col + c * FRAG_COLS)
                        )
                        * bpe,
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    ),
                    [
                        da_pack[c * 4 + 0],
                        da_pack[c * 4 + 1],
                        da_pack[c * 4 + 2],
                        da_pack[c * 4 + 3],
                    ],
                    nvvm.MMALayout.ROW,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_da_ready[0].arrive()

            # ---- dM epilogue -----------------------------------------------------
            bars.mb_dm_acc_ready[0].wait(dm_ready_index.phase)
            dm_ready_index = advance(dm_ready_index, 1)
            dm_base = sDm[0].base
            dm_vec = nvvm.tcgen05_ld(
                "16x256b",
                nvvm.make_tmem_ptr((tmem_warp_row << 16) + tmem_dm_core_col, cutlass.Float32),
                num=8,
            )

            dm_pack = []
            for k in cutlass.range_constexpr(num_vals // 2):
                p0, p1 = fmul2(
                    dm_vec[2 * k],
                    dm_vec[2 * k + 1],
                    decay_t_strict[2 * k],
                    decay_t_strict[2 * k + 1],
                )
                v0 = cutlass.Float32(0.0) - p0
                v1 = cutlass.Float32(0.0) - p1
                dm_pack.append(fp32_to_fp16(v0, v1, dtype=cfg.io_dtype))
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                nvvm.stmatrix(
                    cutlass.inttoptr(
                        dm_base
                        + (
                            store_row * cfg.b_t
                            + smem_64x64_offset(store_row, store_col + c * FRAG_COLS)
                        )
                        * bpe,
                        cutlass.AddressSpace.smem,
                        cutlass.BFloat16,
                    ),
                    [
                        dm_pack[c * 4 + 0],
                        dm_pack[c * 4 + 1],
                        dm_pack[c * 4 + 2],
                        dm_pack[c * 4 + 3],
                    ],
                    nvvm.MMALayout.ROW,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dm_acc_done[0].arrive()

            # ---- dK attn read ----------------------------------------------------
            bars.mb_dk_attn_acc_ready[0].wait(cg0_dk_attn_ready.phase)
            cg0_dk_attn_ready = advance(cg0_dk_attn_ready, 1)
            dks_regs = []
            for sub in cutlass.range_constexpr(2):
                dks_vec = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(
                        ((tmem_warp_row + sub * 16) << 16) + tmem_dvdk_acc_col, cutlass.Float32
                    ),
                    num=8,
                )
                dks_regs.append([dks_vec[k] for k in range(32)])
            nvvm.tcgen05_wait("load")
            bars.mb_dk_attn_acc_done[0].arrive()

            # ---- dBeta/dGate M-terms: E = dM_core ⊙ M_kk(sKK, strict-masked). ----
            kk_frag = []
            for c in cutlass.range_constexpr(ACC_N_FRAGS):
                kk_frag += list(
                    nvvm.ldmatrix(
                        cutlass.inttoptr(
                            kk_base
                            + (
                                store_row * cfg.b_t
                                + smem_64x64_offset(store_row, store_col + c * FRAG_COLS)
                            )
                            * bpe,
                            cutlass.AddressSpace.smem,
                            cutlass.BFloat16,
                        ),
                        4,
                        nvvm.MMALayout.ROW,
                    )
                )
            binv_row = [
                cute.math.rcp(gBeta[0] + cutlass.Float32(1e-10), approx=True, ftz=True),
                cute.math.rcp(gBeta[2] + cutlass.Float32(1e-10), approx=True, ftz=True),
            ]
            row_acc = [cutlass.Float32(0.0)] * 8
            col_part = [opaque_f32_zero() for _ in range(16)]
            for j in cutlass.range_constexpr(num_vals // 2):
                klo, khi = f16x2_to_f32(kk_frag[j], dtype=cfg.io_dtype)
                binv_j = binv_row[j % 2]
                p_lo, p_hi = fmul2(dm_vec[2 * j], dm_vec[2 * j + 1], klo, khi)
                e_lo, e_hi = fmul2(p_lo, p_hi, binv_j, binv_j)
                row_acc[(j % 2) * 4 + (j // 2) % 4] += e_lo + e_hi
                c0 = cutlass.const_expr((j // 2) * 2)
                col_part[c0], col_part[c0 + 1] = fadd2(col_part[c0], col_part[c0 + 1], e_lo, e_hi)
            row_part = [
                (row_acc[0] + row_acc[1]) + (row_acc[2] + row_acc[3]),
                (row_acc[4] + row_acc[5]) + (row_acc[6] + row_acc[7]),
            ]
            for off in [1, 2]:
                for rp in cutlass.range_constexpr(2):
                    row_part[rp] += nvvm.shfl_sync(
                        0xFFFFFFFF, row_part[rp], off, 31, kind=nvvm.Shfl.BFLY
                    )
            bars.mb_dbeta_cg1_ready[0].wait(cg0_dbeta_index.phase)
            cg0_dbeta_index = advance(cg0_dbeta_index, 1)
            if lane_id % 4 == 0:
                for rp in cutlass.range_constexpr(2):
                    crow_r = warp_id * 16 + lane_id // 4 + rp * 8
                    db = sBeta[crow_r, 0, beta_idx] - row_part[rp] * binv_row[rp]
                    if cutlass.const_expr(cfg.beta_sigmoid):
                        b = gBeta[2 * rp]
                        db = db * (b - b * b)
                    sBeta[crow_r, 0, beta_idx] = db

            # ---- part reductions -------------------------------------------------
            if chunk_idx + DSTATE_IN0 >= FIRST_STATE_CHUNK:
                part_k = [acc_zero] * 16
                for sub in cutlass.range_constexpr(2):
                    for m0 in cutlass.range_constexpr(4):
                        frag_addr = (
                            frag_slab_off
                            + (frag_row + m0 * 16) * 64
                            + smem_64x64_offset(frag_row + m0 * 16, frag_col + sub * 16)
                        )
                        k_frag = nvvm.ldmatrix(
                            (sK_base_p + cg0_k_idx * k_stage_elems_cg0 + frag_addr).raw_ptr(),
                            4,
                            nvvm.MMALayout.COL,
                        )
                        for i in cutlass.range_constexpr(4):
                            k_lo, k_hi = f16x2_to_f32(k_frag[i], dtype=cfg.io_dtype)
                            frag_e0 = cutlass.const_expr(8 * m0 + 2 * i)
                            part_e0 = cutlass.const_expr((frag_e0 // 4) * 2 + (frag_e0 % 2))
                            if cutlass.const_expr(sub == 0 and i % 2 == 0):
                                part_k[part_e0], part_k[part_e0 + 1] = fmul2(
                                    dks_regs[sub][frag_e0], dks_regs[sub][frag_e0 + 1], k_lo, k_hi
                                )
                            else:
                                part_k[part_e0], part_k[part_e0 + 1] = ffma2(
                                    dks_regs[sub][frag_e0],
                                    dks_regs[sub][frag_e0 + 1],
                                    k_lo,
                                    k_hi,
                                    part_k[part_e0],
                                    part_k[part_e0 + 1],
                                )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_k_cg0_done[cg0_k_idx].arrive()

                dgate_part = [col_part[j] - part_k[j] for j in range(16)]
                dgate_part_lo, dgate_part_hi = warp_reduce_scatter_frag_16_elems(
                    dgate_part, lane_id
                )
                dgate_last_w = k_dot_dk_inter + (
                    (cumprod_total * state_dot_dstate if chunk_idx + DSTATE_IN0 >= 1 else acc_zero)
                    if chunk_idx < cend - FIRST_STATE_CHUNK
                    else acc_zero
                )
                tok0 = (lane_id // 4) * 8 + (lane_id % 4) * 2
                (skk_red + warp_id * 64 + tok0).store(dgate_part_lo)
                (skk_red + warp_id * 64 + tok0 + 1).store(
                    dgate_part_hi + dgate_last_w if lane_id == 31 else dgate_part_hi
                )

                bars.mb_dgate_cg1_ready[0].wait(cg0_dgate_index.phase)
                cg0_dgate_index = advance(cg0_dgate_index, 1)
                if lane_id % 4 == 0:
                    for rp in cutlass.range_constexpr(2):
                        crow_r = warp_id * 16 + lane_id // 4 + rp * 8
                        sCumsumlog[crow_r, 0, gate_idx] = (
                            sCumsumlog[crow_r, 0, gate_idx] - row_part[rp]
                        )
                nvvm.barrier_cta_sync_aligned(
                    cfg.inverse_barrier_id, thread_count=cfg.inverse_barrier_threads
                )
                if cg0_tidx < 64:
                    dgate_sum = (
                        (skk_red + cg0_tidx).load()
                        + (skk_red + 64 + cg0_tidx).load()
                        + (skk_red + 128 + cg0_tidx).load()
                        + (skk_red + 192 + cg0_tidx).load()
                    )
                    sCumsumlog[cg0_tidx, 0, gate_idx] = (
                        sCumsumlog[cg0_tidx, 0, gate_idx] + dgate_sum
                    )

            if chunk_idx + DSTATE_IN0 < FIRST_STATE_CHUNK:
                part_k = [acc_zero] * 16
                for sub in cutlass.range_constexpr(2):
                    for m0 in cutlass.range_constexpr(4):
                        frag_addr = (
                            frag_slab_off
                            + (frag_row + m0 * 16) * 64
                            + smem_64x64_offset(frag_row + m0 * 16, frag_col + sub * 16)
                        )
                        k_frag = nvvm.ldmatrix(
                            (sK_base_p + cg0_k_idx * k_stage_elems_cg0 + frag_addr).raw_ptr(),
                            4,
                            nvvm.MMALayout.COL,
                        )
                        for i in cutlass.range_constexpr(4):
                            k_lo, k_hi = f16x2_to_f32(k_frag[i], dtype=cfg.io_dtype)
                            frag_e0 = cutlass.const_expr(8 * m0 + 2 * i)
                            part_e0 = cutlass.const_expr((frag_e0 // 4) * 2 + (frag_e0 % 2))
                            if cutlass.const_expr(sub == 0 and i % 2 == 0):
                                part_k[part_e0], part_k[part_e0 + 1] = fmul2(
                                    dks_regs[sub][frag_e0], dks_regs[sub][frag_e0 + 1], k_lo, k_hi
                                )
                            else:
                                part_k[part_e0], part_k[part_e0 + 1] = ffma2(
                                    dks_regs[sub][frag_e0],
                                    dks_regs[sub][frag_e0 + 1],
                                    k_lo,
                                    k_hi,
                                    part_k[part_e0],
                                    part_k[part_e0 + 1],
                                )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_k_cg0_done[cg0_k_idx].arrive()

                dgate_part = [col_part[j] - part_k[j] for j in range(16)]
                dgate_part_lo, dgate_part_hi = warp_reduce_scatter_frag_16_elems(
                    dgate_part, lane_id
                )
                tok0 = (lane_id // 4) * 8 + (lane_id % 4) * 2
                (skk_red + warp_id * 64 + tok0).store(dgate_part_lo)
                (skk_red + warp_id * 64 + tok0 + 1).store(dgate_part_hi)

                bars.mb_dgate_cg1_ready[0].wait(cg0_dgate_index.phase)
                cg0_dgate_index = advance(cg0_dgate_index, 1)
                if lane_id % 4 == 0:
                    for rp in cutlass.range_constexpr(2):
                        crow_r = warp_id * 16 + lane_id // 4 + rp * 8
                        sCumsumlog[crow_r, 0, gate_idx] = (
                            sCumsumlog[crow_r, 0, gate_idx] - row_part[rp]
                        )
                nvvm.barrier_cta_sync_aligned(
                    cfg.inverse_barrier_id, thread_count=cfg.inverse_barrier_threads
                )
                if cg0_tidx < 64:
                    dgate_sum = (
                        (skk_red + cg0_tidx).load()
                        + (skk_red + 64 + cg0_tidx).load()
                        + (skk_red + 128 + cg0_tidx).load()
                        + (skk_red + 192 + cg0_tidx).load()
                    )
                    sCumsumlog[cg0_tidx, 0, gate_idx] = (
                        sCumsumlog[cg0_tidx, 0, gate_idx] + dgate_sum
                    )

            bars.mb_gate_done[gate_idx].arrive()
            bars.mb_beta_done[beta_idx].arrive()
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)
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
    mDstate0_out,
    mDstate_in,
    tidx,
    warp_idx,
    tmem_base_slot,
    scale,
    sQ,
    sK,
    sV,
    sdO,
    sCumsumlog,
    sCumprod,
    sBeta,
    sdQ,
    sdK,
    sdV,
    sDstate,
    sDa,
    sDm,
    sSched,
    bars,
):
    """Compute warp-group 1 role (warps 4-7): persistent scheduler loop
    running each chunk's gradient epilogues and stagings."""

    v_index = PipelineState.start(phase=0)
    do_index = PipelineState.start(phase=0)
    gate_index = PipelineState.start(phase=0)
    cg1_k_state_ready = PipelineState.start(phase=0)
    cg1_u_ready = PipelineState.start(phase=0)
    cg1_dy_ready = PipelineState.start(phase=0)
    cg1_du_scale_ready = PipelineState.start(phase=0)
    cg1_du_total_ready = PipelineState.start(phase=0)
    cg1_dk_total_ready = PipelineState.start(phase=0)
    dstate_acc_index = PipelineState.start(phase=0)
    cg1_state_dot_dstate_index = PipelineState.start(phase=0)
    dq_index = PipelineState.start(phase=1)
    cg1_beta_index = PipelineState.start(phase=0)
    sdv_done_index = PipelineState.start(phase=1)
    cg1_dk_state_path_ready = PipelineState.start(phase=0)
    dq_total_ready_index = PipelineState.start(phase=0)
    dstate_inp_index = PipelineState.start(phase=1)
    dk_index = PipelineState.start(phase=1)
    dv_index = PipelineState.start(phase=1)

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
    tmem_dstate_acc_col = tmem_base + cfg.tmem_dstate_acc_offset
    tmem_dstate_inp_col = tmem_base + cfg.tmem_dstate_inp_offset
    tmem_dvdk_acc_col = tmem_base + cfg.tmem_dvdk_acc_offset
    tmem_shared_acc_col = tmem_base + cfg.tmem_shared_acc_offset
    tmem_shared_inp_col = tmem_base + cfg.tmem_shared_inp_offset
    SHARED_INP_STAGE_COLS = cfg.b_t // 2
    tmem_do_prime_col = tmem_shared_inp_col
    tmem_du_col = tmem_shared_inp_col + SHARED_INP_STAGE_COLS
    tmem_dyp_col = tmem_du_col
    tmem_y_col = tmem_base + cfg.tmem_y_offset
    tmem_g_k_state_col = tmem_y_col + SHARED_INP_STAGE_COLS
    ACC_STAGE_COLS = cfg.b_t
    tmem_acc_a = tmem_shared_acc_col
    tmem_acc_b = tmem_shared_acc_col + ACC_STAGE_COLS
    tmem_kk_col = tmem_acc_a
    tmem_k_state_col = tmem_acc_a
    tmem_dy_col = tmem_acc_a
    tmem_dm_core_col = tmem_acc_a
    tmem_a_col = tmem_acc_b
    tmem_u_col = tmem_acc_b
    tmem_da_col = tmem_acc_b
    tmem_dk_state_path_col = tmem_shared_inp_col
    frag_row = cg1_tidx % 8 + (cg1_tidx // 16 % 2) * 8
    frag_col = (cg1_tidx // 8 % 2) * 8 + (cg1_tidx // 32 % 2) * 32
    frag_slab_off = (cg1_tidx // 64) * 4096
    dv_stage_elems = cfg.dv_cosize // cfg.smem_dv_stages
    sdV_base = cute.make_ptr(
        cfg.io_dtype,
        sdV[0].base,
        mem_space=cute.AddressSpace.smem,
        assumed_align=cfg.buffer_align_bytes,
    )
    v_stage_elems = cfg.v_cosize // cfg.smem_v_stages
    sV_base = cute.make_ptr(
        cfg.io_dtype,
        sV[0].base,
        mem_space=cute.AddressSpace.smem,
        assumed_align=cfg.buffer_align_bytes,
    )
    sQ_base = cute.make_ptr(
        cfg.io_dtype,
        sQ[0].base,
        mem_space=cute.AddressSpace.smem,
        assumed_align=cfg.buffer_align_bytes,
    )
    do_stage_elems = cfg.do_cosize // cfg.smem_do_stages
    sdO_base = cute.make_ptr(
        cfg.io_dtype,
        sdO[0].base,
        mem_space=cute.AddressSpace.smem,
        assumed_align=cfg.buffer_align_bytes,
    )
    dk_stage_elems = cfg.dk_cosize // cfg.smem_dk_stages
    dq_stage_elems = cfg.dq_cosize // cfg.smem_dq_stages
    sdQ_base = cute.make_ptr(
        cfg.io_dtype,
        sdQ[0].base,
        mem_space=cute.AddressSpace.smem,
        assumed_align=cfg.buffer_align_bytes,
    )
    sdK_base = cute.make_ptr(
        cfg.io_dtype,
        sdK[0].base,
        mem_space=cute.AddressSpace.smem,
        assumed_align=cfg.buffer_align_bytes,
    )
    sDstate_base_int = sDstate[0].base
    sred_base = cute.make_ptr(
        cutlass.Float32,
        sdQ[0].base,
        mem_space=cute.AddressSpace.smem,
        assumed_align=cfg.buffer_align_bytes,
    )
    sdstate_red = cute.make_ptr(
        cutlass.Float32,
        sDstate[0].base,
        mem_space=cute.AddressSpace.smem,
        assumed_align=cfg.buffer_align_bytes,
    )
    dstate_done_idx = cutlass.Int32(0)
    cg1_warp_id = cg1_tidx // cfg.threads_per_warp

    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
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
        num_item_chunks = cend - wstart

        # ---- d_final_state prologue ----------------------------------------------
        if cutlass.const_expr(cfg.use_dstate_in):
            if num_item_chunks > 0:
                gDstate_in = mDstate_in[None, None, head_idx, batch_idx]
                seed_from_dstate_in = cend == num_chunks_b
                dstate_inp_idx = dstate_inp_index.idx
                bars.mb_dstate_inp_done[dstate_inp_idx].wait(dstate_inp_index.phase)
                dstate_inp_index = advance(dstate_inp_index, cfg.tmem_dstate_inp_stages)
                for sub in cutlass.range_constexpr(num_state_subs):
                    dstate_in_vals = []
                    for kk in cutlass.range_constexpr(ldtm_width):
                        v = gDstate_in[cg1_tidx, sub * ldtm_width + kk]
                        v = v if seed_from_dstate_in else cutlass.Float32(0.0)
                        dstate_in_vals.append(v)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            (tmem_warp_row << 16) + tmem_dstate_acc_col + sub * ldtm_width,
                            cutlass.Float32,
                        ),
                        cutlass.Vector.from_elements(tuple(dstate_in_vals), cutlass.Float32),
                    )
                    dstate_in_pack = [
                        fp32_to_fp16(
                            dstate_in_vals[2 * j], dstate_in_vals[2 * j + 1], dtype=cfg.io_dtype
                        )
                        for j in range(16)
                    ]
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            (tmem_warp_row << 16) + tmem_dstate_inp_col + sub * sttm_width,
                            cutlass.Int32,
                        ),
                        cutlass.Vector.from_elements(tuple(dstate_in_pack), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_inp_ready[dstate_inp_idx].arrive()

                for sub in cutlass.range_constexpr(num_state_subs):
                    dstate_smem_vec = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            (tmem_warp_row << 16) + tmem_dstate_acc_col + sub * ldtm_width,
                            cutlass.Float32,
                        ),
                        num=32,
                    )
                    for g in cutlass.range_constexpr(ldtm_width // 8):
                        dstate_smem_pack = tuple(
                            fp32_to_fp16(
                                dstate_smem_vec[g * 8 + 2 * t],
                                dstate_smem_vec[g * 8 + 2 * t + 1],
                                dtype=cfg.io_dtype,
                            )
                            for t in range(4)
                        )
                        dstate_dk = sub * ldtm_width + g * 8
                        dstate_addr = (
                            (dstate_dk // 64) * (cfg.d_v * 64)
                            + cg1_tidx * 64
                            + smem_64x64_offset(cg1_tidx, dstate_dk % 64)
                        )
                        cutlass.inttoptr(
                            sDstate_base_int + dstate_addr * 2,
                            cutlass.AddressSpace.smem,
                            cfg.io_dtype,
                        ).store(
                            cutlass.Vector.from_elements(dstate_smem_pack, cutlass.Int32).bitcast(
                                cfg.io_dtype
                            ),
                            alignment=16,
                        )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dstate_smem_ready[0].arrive()

        # ---- chunks NT-1 .. 0 (backward) ------------------------------------------
        for rev_idx in cutlass.range(num_item_chunks):
            chunk_idx = cend - 1 - rev_idx
            have_dstate = (
                cutlass.Boolean(True) if cutlass.const_expr(cfg.use_dstate_in) else rev_idx > 0
            )
            gate_idx = gate_index.idx
            bars.mb_gate_ready[gate_idx].wait(gate_index.phase)
            gate_index = advance(gate_index, cfg.smem_gate_stages)

            beta_idx = cg1_beta_index.idx
            bars.mb_beta_ready[beta_idx].wait(cg1_beta_index.phase)
            cg1_beta_index = advance(cg1_beta_index, cfg.smem_beta_stages)

            # ---- dstate rescale: dstate *= this chunk's cumprod --------------------------
            if have_dstate:
                cumprod_top = sCumprod[sCumprod.shape[0] - 1, 0, gate_idx]
                for sub in cutlass.range_constexpr(num_state_subs):
                    dstate_rescale_vec = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            (tmem_warp_row << 16) + tmem_dstate_acc_col + sub * ldtm_width,
                            cutlass.Float32,
                        ),
                        num=32,
                    )
                    dstate_rescaled = []
                    for j in cutlass.range_constexpr(16):
                        h0, h1 = fmul2(
                            dstate_rescale_vec[2 * j],
                            dstate_rescale_vec[2 * j + 1],
                            cumprod_top,
                            cumprod_top,
                        )
                        dstate_rescaled += [h0, h1]
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            (tmem_warp_row << 16) + tmem_dstate_acc_col + sub * ldtm_width,
                            cutlass.Float32,
                        ),
                        cutlass.Vector.from_elements(tuple(dstate_rescaled), cutlass.Float32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_scale_acc_done[dstate_done_idx].arrive()

            num_vals = 32
            cumprod_fp32 = []
            for g in cutlass.range_constexpr(8):
                for b in cutlass.range_constexpr(2):
                    cumprod_fp32.append(sCumprod[(lane_id % 4) * 2 + g * 8 + b, 0, gate_idx])
            cumprod_vals = [cumprod_fp32[(k // 4) * 2 + (k % 2)] for k in range(num_vals)]

            # ---- dO' restage: dO * cumprod_vals * scale -> shared_inp TMEM -----------
            do_idx = do_index.idx
            bars.mb_do_ready[do_idx].wait(do_index.phase)
            do_index = advance(do_index, cfg.smem_do_stages)
            do_regs = [[cutlass.Float32(0.0), cutlass.Float32(0.0)] for _ in range(32)]
            for c in cutlass.range_constexpr(8):
                m0 = cutlass.const_expr(c % 4)
                sub = cutlass.const_expr(c // 4)
                do_frag = nvvm.ldmatrix(
                    (
                        sdO_base
                        + do_idx * do_stage_elems
                        + frag_slab_off
                        + (frag_row + m0 * 16) * 64
                        + smem_64x64_offset(frag_row + m0 * 16, frag_col + sub * 16)
                    ).raw_ptr(),
                    4,
                    nvvm.MMALayout.COL,
                )
                for i in cutlass.range_constexpr(4):
                    lo, hi = f16x2_to_f32(do_frag[i], dtype=cfg.io_dtype)
                    p0, p1 = fmul2(
                        lo, hi, cumprod_vals[8 * m0 + 2 * i], cumprod_vals[8 * m0 + 2 * i + 1]
                    )
                    do_regs[8 * m0 + 2 * i][sub], do_regs[8 * m0 + 2 * i + 1][sub] = fmul2(
                        p0, p1, scale, scale
                    )
            for sub in cutlass.range_constexpr(2):
                do_pack = [
                    fp32_to_fp16(do_regs[2 * j][sub], do_regs[2 * j + 1][sub], dtype=cfg.io_dtype)
                    for j in range(16)
                ]
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(
                        ((tmem_warp_row + sub * 16) << 16) + tmem_do_prime_col, cutlass.Int32
                    ),
                    cutlass.Vector.from_elements(tuple(do_pack), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_do_prime_inp_ready[0].arrive()

            # ---- dV inter: in-place decay scale ----------------------------------
            if have_dstate:
                last_cumsumlog = sCumsumlog[cfg.b_t - 1, 0, gate_idx]
                col_cs_fp32 = []
                for g in cutlass.range_constexpr(8):
                    for b in cutlass.range_constexpr(2):
                        col_cs_fp32.append(sCumsumlog[(lane_id % 4) * 2 + g * 8 + b, 0, gate_idx])
                decay_scale_fp32 = []
                for i in cutlass.range_constexpr(16):
                    decay_scale_fp32.append(
                        cute.math.exp2(last_cumsumlog - col_cs_fp32[i], fastmath=True)
                    )
                decay_scale_vals = [
                    decay_scale_fp32[(k // 4) * 2 + (k % 2)] for k in range(num_vals)
                ]
                bars.mb_du_scale_acc_ready[0].wait(cg1_du_scale_ready.phase)
                cg1_du_scale_ready = advance(cg1_du_scale_ready, 1)
                dv_ptrs = []
                dv_vecs = []
                for sub in cutlass.range_constexpr(2):
                    dv_ptrs.append(
                        nvvm.make_tmem_ptr(
                            ((tmem_warp_row + sub * 16) << 16) + tmem_dvdk_acc_col, cutlass.Float32
                        )
                    )
                    dv_vecs.append(nvvm.tcgen05_ld("16x256b", dv_ptrs[sub], num=8))
                for sub in cutlass.range_constexpr(2):
                    dv_scaled = []
                    for j in cutlass.range_constexpr(16):
                        s0, s1 = fmul2(
                            dv_vecs[sub][2 * j],
                            dv_vecs[sub][2 * j + 1],
                            decay_scale_vals[2 * j],
                            decay_scale_vals[2 * j + 1],
                        )
                        dv_scaled += [s0, s1]
                    nvvm.tcgen05_st(
                        "16x256b",
                        dv_ptrs[sub],
                        cutlass.Vector.from_elements(tuple(dv_scaled), cutlass.Float32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_du_scale_acc_done[0].arrive()

            # ---- Y staging:  Y = V - cumprod*(K @ state) -> f16 TMEM slots -----------
            v_idx = v_index.idx
            bars.mb_v_ready[v_idx].wait(v_index.phase)
            v_index = advance(v_index, cfg.smem_v_stages)
            if chunk_idx >= FIRST_STATE_CHUNK:
                v_frags = [[cutlass.Int32(0), cutlass.Int32(0)] for _ in range(16)]
                for c in cutlass.range_constexpr(8):
                    m0 = cutlass.const_expr(c % 4)
                    sub = cutlass.const_expr(c // 4)
                    v_frag = nvvm.ldmatrix(
                        (
                            sV_base
                            + v_idx * v_stage_elems
                            + frag_slab_off
                            + (frag_row + m0 * 16) * 64
                            + smem_64x64_offset(frag_row + m0 * 16, frag_col + sub * 16)
                        ).raw_ptr(),
                        4,
                        nvvm.MMALayout.COL,
                    )
                    for i in cutlass.range_constexpr(4):
                        v_frags[4 * m0 + i][sub] = v_frag[i]
                bars.mb_k_state_acc_ready[0].wait(cg1_k_state_ready.phase)
                cg1_k_state_ready = advance(cg1_k_state_ready, 1)
                for sub in cutlass.range_constexpr(2):
                    k_state_vec = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(
                            ((tmem_warp_row + sub * 16) << 16) + tmem_k_state_col, cutlass.Float32
                        ),
                        num=8,
                    )
                    g_k_state_pack = []
                    y_pack = []
                    for j in cutlass.range_constexpr(16):
                        g0, g1 = fmul2(
                            k_state_vec[2 * j],
                            k_state_vec[2 * j + 1],
                            cumprod_vals[2 * j],
                            cumprod_vals[2 * j + 1],
                        )
                        g_k_state_word = fp32_to_fp16(g0, g1, dtype=cfg.io_dtype)
                        g_k_state_pack.append(g_k_state_word)
                        y_pack.append(sub_f16x2(v_frags[j][sub], g_k_state_word, cfg.io_dtype))
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(
                            ((tmem_warp_row + sub * 16) << 16) + tmem_y_col, cutlass.Int32
                        ),
                        cutlass.Vector.from_elements(tuple(y_pack), cutlass.Int32),
                    )
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(
                            ((tmem_warp_row + sub * 16) << 16) + tmem_g_k_state_col, cutlass.Int32
                        ),
                        cutlass.Vector.from_elements(tuple(g_k_state_pack), cutlass.Int32),
                    )
            if chunk_idx < FIRST_STATE_CHUNK:
                for sub in cutlass.range_constexpr(2):
                    v_pack = []
                    for m0 in cutlass.range_constexpr(4):
                        v_frag = nvvm.ldmatrix(
                            (
                                sV_base
                                + v_idx * v_stage_elems
                                + frag_slab_off
                                + (frag_row + m0 * 16) * 64
                                + smem_64x64_offset(frag_row + m0 * 16, frag_col + sub * 16)
                            ).raw_ptr(),
                            4,
                            nvvm.MMALayout.COL,
                        )
                        for i in cutlass.range_constexpr(4):
                            v_pack.append(v_frag[i])
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(
                            ((tmem_warp_row + sub * 16) << 16) + tmem_y_col, cutlass.Int32
                        ),
                        cutlass.Vector.from_elements(tuple(v_pack), cutlass.Int32),
                    )
            nvvm.tcgen05_wait("store")
            bars.mb_y_ready[0].arrive()

            # ---- dU restage: dV acc ----------------------------------------------
            bars.mb_du_total_acc_ready[0].wait(cg1_du_total_ready.phase)
            cg1_du_total_ready = advance(cg1_du_total_ready, 1)
            for sub in cutlass.range_constexpr(2):
                du_vec = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(
                        ((tmem_warp_row + sub * 16) << 16) + tmem_dvdk_acc_col, cutlass.Float32
                    ),
                    num=8,
                )
                du_pack = [
                    fp32_to_fp16(du_vec[2 * j], du_vec[2 * j + 1], dtype=cfg.io_dtype)
                    for j in range(16)
                ]
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(
                        ((tmem_warp_row + sub * 16) << 16) + tmem_du_col, cutlass.Int32
                    ),
                    cutlass.Vector.from_elements(tuple(du_pack), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_du_inp_ready[0].arrive()

            # ---- U readout -------------------------------------------------------
            bars.mb_u_acc_ready[0].wait(cg1_u_ready.phase)
            cg1_u_ready = advance(cg1_u_ready, 1)
            u_regs = []
            for sub in cutlass.range_constexpr(2):
                u_vec = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(
                        ((tmem_warp_row + sub * 16) << 16) + tmem_u_col, cutlass.Float32
                    ),
                    num=8,
                )
                u_regs.append([u_vec[k] for k in range(32)])
            for sub in cutlass.range_constexpr(2):
                for m0 in cutlass.range_constexpr(4):
                    u_pack = [
                        fp32_to_fp16(
                            u_regs[sub][8 * m0 + 2 * j],
                            u_regs[sub][8 * m0 + 2 * j + 1],
                            dtype=cfg.io_dtype,
                        )
                        for j in range(4)
                    ]
                    nvvm.stmatrix(
                        (
                            sV_base
                            + v_idx * v_stage_elems
                            + frag_slab_off
                            + (frag_row + m0 * 16) * 64
                            + smem_64x64_offset(frag_row + m0 * 16, frag_col + sub * 16)
                        ).raw_ptr(),
                        u_pack,
                        nvvm.MMALayout.COL,
                    )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_u_ready[0].arrive()

            # ---- dY --------------------------------------------------------------
            bars.mb_dy_acc_ready[0].wait(cg1_dy_ready.phase)
            cg1_dy_ready = advance(cg1_dy_ready, 1)
            dy_regs = []
            for sub in cutlass.range_constexpr(2):
                dy_vec = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(
                        ((tmem_warp_row + sub * 16) << 16) + tmem_dy_col, cutlass.Float32
                    ),
                    num=8,
                )
                dy_regs.append([dy_vec[k] for k in range(32)])

            # ---- dY' = -cumprod_vals * dY -> f16 shared_inp --------------------------
            neg_one = cutlass.Float32(-1.0)
            cumprod_neg_vals = [cumprod_vals[k] * neg_one for k in range(32)]
            for sub in cutlass.range_constexpr(2):
                dyp = []
                for j in cutlass.range_constexpr(16):
                    n0, n1 = fmul2(
                        dy_regs[sub][2 * j],
                        dy_regs[sub][2 * j + 1],
                        cumprod_neg_vals[2 * j],
                        cumprod_neg_vals[2 * j + 1],
                    )
                    dyp += [n0, n1]
                dyp_pack = [
                    fp32_to_fp16(dyp[2 * j], dyp[2 * j + 1], dtype=cfg.io_dtype) for j in range(16)
                ]
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(
                        ((tmem_warp_row + sub * 16) << 16) + tmem_dyp_col, cutlass.Int32
                    ),
                    cutlass.Vector.from_elements(tuple(dyp_pack), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_dyp_inp_ready[0].arrive()

            # ---- dV staging: dV = dY -> the sdV slot -----------------------------
            dv_stg_idx = dv_index.idx
            bars.mb_dv_tmastg_done[dv_stg_idx].wait(dv_index.phase)
            dv_index = advance(dv_index, cfg.smem_dv_stages)
            bars.mb_sdv_done[0].wait(sdv_done_index.phase)
            sdv_done_index = advance(sdv_done_index, 1)
            for sub in cutlass.range_constexpr(2):
                for m0 in cutlass.range_constexpr(4):
                    dv_pack = [
                        fp32_to_fp16(
                            dy_regs[sub][8 * m0 + 2 * j],
                            dy_regs[sub][8 * m0 + 2 * j + 1],
                            dtype=cfg.io_dtype,
                        )
                        for j in range(4)
                    ]
                    nvvm.stmatrix(
                        (
                            sdV_base
                            + dv_stg_idx * dv_stage_elems
                            + frag_slab_off
                            + (frag_row + m0 * 16) * 64
                            + smem_64x64_offset(frag_row + m0 * 16, frag_col + sub * 16)
                        ).raw_ptr(),
                        dv_pack,
                        nvvm.MMALayout.COL,
                    )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dv_tmastg_ready[dv_stg_idx].arrive()
            dq_stg_idx = dq_index.idx
            bars.mb_dq_tmastg_done[dq_stg_idx].wait(dq_index.phase)
            dq_index = advance(dq_index, cfg.smem_dq_stages)
            sred = sred_base + dq_stg_idx * (dq_stage_elems // 2)

            # ---- dBeta/dGate V-terms: dBeta_t += rowsum(dV ⊙ Y)_t / Beta_t -------
            part_y = [cutlass.Float32(0.0)] * 16
            for sub in cutlass.range_constexpr(2):
                y_vec = nvvm.tcgen05_ld(
                    "16x128b",
                    nvvm.make_tmem_ptr(
                        ((tmem_warp_row + sub * 16) << 16) + tmem_y_col, cutlass.Int32
                    ),
                    num=8,
                )
                for j in cutlass.range_constexpr(16):
                    lo, hi = f16x2_to_f32(y_vec[j], dtype=cfg.io_dtype)
                    frag_e0 = cutlass.const_expr(2 * j)
                    part_e0 = cutlass.const_expr((frag_e0 // 4) * 2 + (frag_e0 % 2))
                    if cutlass.const_expr(sub == 0 and j % 2 == 0):
                        part_y[part_e0], part_y[part_e0 + 1] = fmul2(
                            dy_regs[sub][frag_e0], dy_regs[sub][frag_e0 + 1], lo, hi
                        )
                    else:
                        part_y[part_e0], part_y[part_e0 + 1] = ffma2(
                            dy_regs[sub][frag_e0],
                            dy_regs[sub][frag_e0 + 1],
                            lo,
                            hi,
                            part_y[part_e0],
                            part_y[part_e0 + 1],
                        )
            py_lo, py_hi = warp_reduce_scatter_frag_16_elems(part_y, lane_id)
            vt_tok0 = (lane_id // 4) * 8 + (lane_id % 4) * 2
            if chunk_idx >= FIRST_STATE_CHUNK:
                part_g = [cutlass.Float32(0.0)] * 16
                for sub in cutlass.range_constexpr(2):
                    g_k_state_vec = nvvm.tcgen05_ld(
                        "16x128b",
                        nvvm.make_tmem_ptr(
                            ((tmem_warp_row + sub * 16) << 16) + tmem_g_k_state_col, cutlass.Int32
                        ),
                        num=8,
                    )
                    for j in cutlass.range_constexpr(16):
                        lo, hi = f16x2_to_f32(g_k_state_vec[j], dtype=cfg.io_dtype)
                        frag_e0 = cutlass.const_expr(2 * j)
                        part_e0 = cutlass.const_expr((frag_e0 // 4) * 2 + (frag_e0 % 2))
                        if cutlass.const_expr(sub == 0 and j % 2 == 0):
                            part_g[part_e0], part_g[part_e0 + 1] = fmul2(
                                dy_regs[sub][frag_e0], dy_regs[sub][frag_e0 + 1], lo, hi
                            )
                        else:
                            part_g[part_e0], part_g[part_e0 + 1] = ffma2(
                                dy_regs[sub][frag_e0],
                                dy_regs[sub][frag_e0 + 1],
                                lo,
                                hi,
                                part_g[part_e0],
                                part_g[part_e0 + 1],
                            )
                pg_lo, pg_hi = warp_reduce_scatter_frag_16_elems(part_g, lane_id)
                (sred + cg1_warp_id * 64 + vt_tok0).store(py_lo)
                (sred + cg1_warp_id * 64 + vt_tok0 + 1).store(py_hi)
                (sred + 256 + cg1_warp_id * 64 + vt_tok0).store(pg_lo)
                (sred + 256 + cg1_warp_id * 64 + vt_tok0 + 1).store(pg_hi)
                nvvm.barrier_cta_sync_aligned(
                    cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads
                )
                if cg1_tidx < 64:
                    binv_t = cute.math.rcp(
                        sBeta[cg1_tidx, 0, beta_idx] + cutlass.Float32(1e-10),
                        approx=True,
                        ftz=True,
                    )
                    ysum = (
                        (sred + cg1_tidx).load()
                        + (sred + 64 + cg1_tidx).load()
                        + (sred + 128 + cg1_tidx).load()
                        + (sred + 192 + cg1_tidx).load()
                    )
                    gsum = (
                        (sred + 256 + cg1_tidx).load()
                        + (sred + 320 + cg1_tidx).load()
                        + (sred + 384 + cg1_tidx).load()
                        + (sred + 448 + cg1_tidx).load()
                    )
                    sBeta[cg1_tidx, 0, beta_idx] = ysum * binv_t
                    sCumsumlog[cg1_tidx, 0, gate_idx] = cutlass.Float32(0.0) - gsum
            if chunk_idx < FIRST_STATE_CHUNK:
                (sred + cg1_warp_id * 64 + vt_tok0).store(py_lo)
                (sred + cg1_warp_id * 64 + vt_tok0 + 1).store(py_hi)
                nvvm.barrier_cta_sync_aligned(
                    cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads
                )
                if cg1_tidx < 64:
                    binv_t = cute.math.rcp(
                        sBeta[cg1_tidx, 0, beta_idx] + cutlass.Float32(1e-10),
                        approx=True,
                        ftz=True,
                    )
                    ysum = (
                        (sred + cg1_tidx).load()
                        + (sred + 64 + cg1_tidx).load()
                        + (sred + 128 + cg1_tidx).load()
                        + (sred + 192 + cg1_tidx).load()
                    )
                    sBeta[cg1_tidx, 0, beta_idx] = ysum * binv_t
                    sCumsumlog[cg1_tidx, 0, gate_idx] = cutlass.Float32(0.0)
            bars.mb_beta_done[beta_idx].arrive()
            bars.mb_dbeta_cg1_ready[0].arrive()
            nvvm.barrier_cta_sync_aligned(cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads)

            # ---- Q fragments held in registers -----------------------------------
            q_frag = []
            for sub in cutlass.range_constexpr(2):
                q_words = []
                for m0 in cutlass.range_constexpr(4):
                    q_f16 = nvvm.ldmatrix(
                        (
                            sQ_base
                            + frag_slab_off
                            + (frag_row + m0 * 16) * 64
                            + smem_64x64_offset(frag_row + m0 * 16, frag_col + sub * 16)
                        ).raw_ptr(),
                        4,
                        nvvm.MMALayout.COL,
                    )
                    for i in cutlass.range_constexpr(4):
                        q_words.append(q_f16[i])
                q_frag.append(q_words)
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(
                        ((tmem_warp_row + sub * 16) << 16) + tmem_y_col, cutlass.Int32
                    ),
                    cutlass.Vector.from_elements(tuple(q_words), cutlass.Int32),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_q_cg1_done[0].arrive()

            # ---- dQ final read -> sdQ --------------------------------------------
            bars.mb_dq_acc_total_ready[0].wait(dq_total_ready_index.phase)
            dq_total_ready_index = advance(dq_total_ready_index, 1)
            dq_regs = []
            for sub in cutlass.range_constexpr(2):
                dq_vec = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(
                        ((tmem_warp_row + sub * 16) << 16) + tmem_dstate_inp_col, cutlass.Float32
                    ),
                    num=8,
                )
                dq_regs.append([dq_vec[k] for k in range(32)])
                for m0 in cutlass.range_constexpr(4):
                    frag_addr = (
                        frag_slab_off
                        + (frag_row + m0 * 16) * 64
                        + smem_64x64_offset(frag_row + m0 * 16, frag_col + sub * 16)
                    )
                    dq_pack = [
                        fp32_to_fp16(
                            dq_vec[8 * m0 + 2 * j], dq_vec[8 * m0 + 2 * j + 1], dtype=cfg.io_dtype
                        )
                        for j in range(4)
                    ]
                    nvvm.stmatrix(
                        (sdQ_base + dq_stg_idx * dq_stage_elems + frag_addr).raw_ptr(),
                        dq_pack,
                        nvvm.MMALayout.COL,
                    )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dq_acc_total_done[0].arrive()
            bars.mb_dq_tmastg_ready[dq_stg_idx].arrive()

            # ---- dQ dot (part_q) -------------------------------------------------
            part_q = [cutlass.Float32(0.0)] * 16
            for sub in cutlass.range_constexpr(2):
                for m0 in cutlass.range_constexpr(4):
                    for i in cutlass.range_constexpr(4):
                        q_lo, q_hi = f16x2_to_f32(q_frag[sub][4 * m0 + i], dtype=cfg.io_dtype)
                        frag_e0 = cutlass.const_expr(8 * m0 + 2 * i)
                        part_e0 = cutlass.const_expr((frag_e0 // 4) * 2 + (frag_e0 % 2))
                        if cutlass.const_expr(sub == 0 and i % 2 == 0):
                            part_q[part_e0], part_q[part_e0 + 1] = fmul2(
                                dq_regs[sub][frag_e0], dq_regs[sub][frag_e0 + 1], q_lo, q_hi
                            )
                        else:
                            part_q[part_e0], part_q[part_e0 + 1] = ffma2(
                                dq_regs[sub][frag_e0],
                                dq_regs[sub][frag_e0 + 1],
                                q_lo,
                                q_hi,
                                part_q[part_e0],
                                part_q[part_e0 + 1],
                            )
            part_q_lo, part_q_hi = warp_reduce_scatter_frag_16_elems(part_q, lane_id)
            tok0 = (lane_id // 4) * 8 + (lane_id % 4) * 2
            (sdstate_red + (cg1_tidx // 32) * 64 + tok0).store(part_q_lo)
            (sdstate_red + (cg1_tidx // 32) * 64 + tok0 + 1).store(part_q_hi)
            nvvm.barrier_cta_sync_aligned(cfg.cg1_barrier_id, thread_count=cfg.cg1_barrier_threads)
            if cg1_tidx < 64:
                pq_sum = (
                    (sdstate_red + cg1_tidx).load()
                    + (sdstate_red + 64 + cg1_tidx).load()
                    + (sdstate_red + 128 + cg1_tidx).load()
                    + (sdstate_red + 192 + cg1_tidx).load()
                )
                sCumsumlog[cg1_tidx, 0, gate_idx] = sCumsumlog[cg1_tidx, 0, gate_idx] + pq_sum
            bars.mb_gate_done[gate_idx].arrive()
            bars.mb_dgate_cg1_ready[0].arrive()

            # ---- NEXT-CHUNK dstate prep ----------------------------------------------
            if chunk_idx >= wstart + 1:
                dstate_idx = dstate_acc_index.idx
                bars.mb_dstate_acc_ready[dstate_idx].wait(dstate_acc_index.phase)
                dstate_acc_index = advance(dstate_acc_index, cfg.tmem_dstate_acc_stages)
                dstate_done_idx = dstate_idx
                dstate_inp_idx = dstate_inp_index.idx
                bars.mb_dstate_inp_done[dstate_inp_idx].wait(dstate_inp_index.phase)
                dstate_inp_index = advance(dstate_inp_index, cfg.tmem_dstate_inp_stages)
                dstate_regs = [
                    [cutlass.Float32(0.0) for _ in range(num_state_subs)] for _ in range(32)
                ]
                for sub in cutlass.range_constexpr(num_state_subs):
                    dstate_vec = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            (tmem_warp_row << 16) + tmem_dstate_acc_col + sub * ldtm_width,
                            cutlass.Float32,
                        ),
                        num=32,
                    )
                    for k in cutlass.range_constexpr(32):
                        dstate_regs[k][sub] = dstate_vec[k]

                    dstate_pack = [
                        fp32_to_fp16(
                            dstate_regs[2 * j][sub],
                            dstate_regs[2 * j + 1][sub],
                            dtype=cfg.io_dtype,
                        )
                        for j in range(16)
                    ]
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            (tmem_warp_row << 16) + tmem_dstate_inp_col + sub * sttm_width,
                            cutlass.Int32,
                        ),
                        cutlass.Vector.from_elements(tuple(dstate_pack), cutlass.Int32),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_inp_ready[dstate_inp_idx].arrive()

            # ---- dK fold ---------------------------------------------------------
            dk_stg_idx = dk_index.idx
            bars.mb_dk_tmastg_done[dk_stg_idx].wait(dk_index.phase)
            dk_index = advance(dk_index, cfg.smem_dk_stages)
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_dk_state_path_acc_ready[0].wait(cg1_dk_state_path_ready.phase)
                cg1_dk_state_path_ready = advance(cg1_dk_state_path_ready, 1)
                dk_state_path_vecs = []
                for sub in cutlass.range_constexpr(2):
                    dk_state_path_vecs.append(
                        nvvm.tcgen05_ld(
                            "16x256b",
                            nvvm.make_tmem_ptr(
                                ((tmem_warp_row + sub * 16) << 16) + tmem_dk_state_path_col,
                                cutlass.Float32,
                            ),
                            num=8,
                        )
                    )
                nvvm.tcgen05_wait("load")
                dk_state_path_regs = []
                for sub in cutlass.range_constexpr(2):
                    dk_state_path_row = []
                    for j in cutlass.range_constexpr(16):
                        n0, n1 = fmul2(
                            dk_state_path_vecs[sub][2 * j],
                            dk_state_path_vecs[sub][2 * j + 1],
                            cumprod_neg_vals[2 * j],
                            cumprod_neg_vals[2 * j + 1],
                        )
                        dk_state_path_row += [n0, n1]
                    dk_state_path_regs.append(dk_state_path_row)

                bars.mb_dk_total_acc_ready[0].wait(cg1_dk_total_ready.phase)
                cg1_dk_total_ready = advance(cg1_dk_total_ready, 1)
                dmr_vecs = []
                for sub in cutlass.range_constexpr(2):
                    dmr_vecs.append(
                        nvvm.tcgen05_ld(
                            "16x256b",
                            nvvm.make_tmem_ptr(
                                ((tmem_warp_row + sub * 16) << 16) + tmem_dvdk_acc_col,
                                cutlass.Float32,
                            ),
                            num=8,
                        )
                    )
                nvvm.tcgen05_wait("load")
                bars.mb_dk_total_acc_done[0].arrive()
                for sub in cutlass.range_constexpr(2):
                    dk_sum = [dmr_vecs[sub][k] + dk_state_path_regs[sub][k] for k in range(32)]
                    for m0 in cutlass.range_constexpr(4):
                        dk_pack = [
                            fp32_to_fp16(
                                dk_sum[8 * m0 + 2 * j],
                                dk_sum[8 * m0 + 2 * j + 1],
                                dtype=cfg.io_dtype,
                            )
                            for j in range(4)
                        ]
                        nvvm.stmatrix(
                            (
                                sdK_base
                                + dk_stg_idx * dk_stage_elems
                                + frag_slab_off
                                + (frag_row + m0 * 16) * 64
                                + smem_64x64_offset(frag_row + m0 * 16, frag_col + sub * 16)
                            ).raw_ptr(),
                            dk_pack,
                            nvvm.MMALayout.COL,
                        )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dk_tmastg_ready[dk_stg_idx].arrive()
            if chunk_idx < FIRST_STATE_CHUNK:
                bars.mb_dk_total_acc_ready[0].wait(cg1_dk_total_ready.phase)
                cg1_dk_total_ready = advance(cg1_dk_total_ready, 1)
                dmr_vecs = []
                for sub in cutlass.range_constexpr(2):
                    dmr_vecs.append(
                        nvvm.tcgen05_ld(
                            "16x256b",
                            nvvm.make_tmem_ptr(
                                ((tmem_warp_row + sub * 16) << 16) + tmem_dvdk_acc_col,
                                cutlass.Float32,
                            ),
                            num=8,
                        )
                    )
                nvvm.tcgen05_wait("load")
                bars.mb_dk_total_acc_done[0].arrive()
                for sub in cutlass.range_constexpr(2):
                    for m0 in cutlass.range_constexpr(4):
                        dk_pack = [
                            fp32_to_fp16(
                                dmr_vecs[sub][8 * m0 + 2 * j],
                                dmr_vecs[sub][8 * m0 + 2 * j + 1],
                                dtype=cfg.io_dtype,
                            )
                            for j in range(4)
                        ]
                        nvvm.stmatrix(
                            (
                                sdK_base
                                + dk_stg_idx * dk_stage_elems
                                + frag_slab_off
                                + (frag_row + m0 * 16) * 64
                                + smem_64x64_offset(frag_row + m0 * 16, frag_col + sub * 16)
                            ).raw_ptr(),
                            dk_pack,
                            nvvm.MMALayout.COL,
                        )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dk_tmastg_ready[dk_stg_idx].arrive()

            # ---- dstate prep ---------------------------------------------------------
            if chunk_idx >= wstart + 1:
                bars.mb_state_dot_dstate_done[0].wait(cg1_state_dot_dstate_index.phase)
                cg1_state_dot_dstate_index = advance(cg1_state_dot_dstate_index, 1)
                for sub in cutlass.range_constexpr(num_state_subs):
                    dstate_smem_vec = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            (tmem_warp_row << 16) + tmem_dstate_acc_col + sub * ldtm_width,
                            cutlass.Float32,
                        ),
                        num=32,
                    )
                    for g in cutlass.range_constexpr(ldtm_width // 8):
                        dstate_smem_pack = tuple(
                            fp32_to_fp16(
                                dstate_smem_vec[g * 8 + 2 * t],
                                dstate_smem_vec[g * 8 + 2 * t + 1],
                                dtype=cfg.io_dtype,
                            )
                            for t in range(4)
                        )
                        dstate_dk = sub * ldtm_width + g * 8
                        dstate_addr = (
                            (dstate_dk // 64) * (cfg.d_v * 64)
                            + cg1_tidx * 64
                            + smem_64x64_offset(cg1_tidx, dstate_dk % 64)
                        )
                        cutlass.inttoptr(
                            sDstate_base_int + dstate_addr * 2,
                            cutlass.AddressSpace.smem,
                            cfg.io_dtype,
                        ).store(
                            cutlass.Vector.from_elements(dstate_smem_pack, cutlass.Int32).bitcast(
                                cfg.io_dtype
                            ),
                            alignment=16,
                        )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dstate_smem_ready[0].arrive()

            if chunk_idx < wstart + 1:
                cg1_state_dot_dstate_index = advance(cg1_state_dot_dstate_index, 1)

        # ---- dstate drain: with an initial state this is d_initial_state ----------------------
        if num_item_chunks > 0:
            dstate_idx = dstate_acc_index.idx
            bars.mb_dstate_acc_ready[dstate_idx].wait(dstate_acc_index.phase)
            dstate_acc_index = advance(dstate_acc_index, cfg.tmem_dstate_acc_stages)
            if cutlass.const_expr(cfg.use_dstate0):
                if wstart == 0:
                    gDstate0 = mDstate0_out[None, None, head_idx, batch_idx]
                    for sub in cutlass.range_constexpr(num_state_subs):
                        dstate0_vec = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                (tmem_warp_row << 16) + tmem_dstate_acc_col + sub * ldtm_width,
                                cutlass.Float32,
                            ),
                            num=32,
                        )
                        for kk in cutlass.range_constexpr(32):
                            gDstate0[cg1_tidx, sub * ldtm_width + kk] = dstate0_vec[kk]
            if cutlass.const_expr(not cfg.use_dstate_in):
                bars.mb_dstate_scale_acc_done[dstate_idx].arrive()
        else:
            if cutlass.const_expr(cfg.use_dstate0):
                write_passthrough = wstart == 0
                if write_passthrough:
                    gDstate0 = mDstate0_out[None, None, head_idx, batch_idx]
                    if cutlass.const_expr(cfg.use_dstate_in):
                        gDstate_in = mDstate_in[None, None, head_idx, batch_idx]
                        for sub in cutlass.range_constexpr(num_state_subs):
                            for kk in cutlass.range_constexpr(32):
                                gDstate0[cg1_tidx, sub * ldtm_width + kk] = gDstate_in[
                                    cg1_tidx, sub * ldtm_width + kk
                                ]
                    else:
                        for sub in cutlass.range_constexpr(num_state_subs):
                            for kk in cutlass.range_constexpr(32):
                                gDstate0[cg1_tidx, sub * ldtm_width + kk] = cutlass.Float32(0.0)

        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)

    bars.mb_tmem_done[0].arrive()

    for _ in range(cfg.tmem_dstate_inp_stages):
        bars.mb_dstate_inp_done[dstate_inp_index.idx].wait(dstate_inp_index.phase)
        dstate_inp_index = advance(dstate_inp_index, cfg.tmem_dstate_inp_stages)
    for _ in range(cfg.smem_dk_stages):
        bars.mb_dk_tmastg_done[dk_index.idx].wait(dk_index.phase)
        dk_index = advance(dk_index, cfg.smem_dk_stages)
    for _ in range(cfg.smem_dv_stages):
        bars.mb_dv_tmastg_done[dv_index.idx].wait(dv_index.phase)
        dv_index = advance(dv_index, cfg.smem_dv_stages)


@cute.jit
def build_descs_body(
    widx,
    base_q,
    base_k,
    base_v,
    base_do,
    base_checkpoint,
    base_dq,
    base_dk,
    base_dv,
    desc_ws: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    state_checkpoints: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    n_batch: cutlass.Int32,
    q_row_stride: cutlass.Int64,
    k_row_stride: cutlass.Int64,
    v_row_stride: cutlass.Int64,
    do_row_stride: cutlass.Int64,
    checkpoint_row_stride: cutlass.Int64,
    checkpoint_every_n: cutlass.Int32,
    dq_row_stride: cutlass.Int64,
    dk_row_stride: cutlass.Int64,
    dv_row_stride: cutlass.Int64,
) -> None:
    """Build sequence-relative runtime TMA descriptors for each sequence."""
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
    sub5 = cute.make_tensor(
        desc_ws.iterator + 5 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    sub6 = cute.make_tensor(
        desc_ws.iterator + 6 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    sub7 = cute.make_tensor(
        desc_ws.iterator + 7 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )

    if widx == 0 and nvvm.elect_sync():
        emit_seq_load_descs(base_q, sub0, cu_seqlens, q, n_batch, q_row_stride, 2)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 1 and nvvm.elect_sync():
        emit_seq_load_descs(base_k, sub1, cu_seqlens, k, n_batch, k_row_stride, 2)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 2 and nvvm.elect_sync():
        emit_seq_load_descs(base_v, sub2, cu_seqlens, v, n_batch, v_row_stride, 2)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 3 and nvvm.elect_sync():
        emit_seq_load_descs(base_do, sub3, cu_seqlens, do, n_batch, do_row_stride, 2)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 4 and nvvm.elect_sync():
        emit_checkpoint_seq_descs(
            base_checkpoint,
            sub4,
            cu_seqlens,
            state_checkpoints,
            n_batch,
            checkpoint_row_stride,
            checkpoint_every_n,
            2,
        )
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 5 and nvvm.elect_sync():
        emit_seq_descs(base_dq, sub5, cu_seqlens, dq, n_batch, dq_row_stride, 2)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 6 and nvvm.elect_sync():
        emit_seq_descs(base_dk, sub6, cu_seqlens, dk, n_batch, dk_row_stride, 2)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 7 and nvvm.elect_sync():
        emit_seq_descs(base_dv, sub7, cu_seqlens, dv, n_batch, dv_row_stride, 2)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )


@cute.kernel
def prologue_kernel(
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    has_sched: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    base_q: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_k: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_v: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_do: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_checkpoint: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_dq: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_dk: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_dv: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    desc_ws: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    state_checkpoints: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    work_item_staging: cute.Tensor | None,
    work_count: cute.Tensor,
    work_items: cute.Tensor | None,
    sched_all: cute.Tensor | None,
    n_batch: cutlass.Int32,
    q_row_stride: cutlass.Int64,
    k_row_stride: cutlass.Int64,
    v_row_stride: cutlass.Int64,
    do_row_stride: cutlass.Int64,
    checkpoint_row_stride: cutlass.Int64,
    checkpoint_every_n: cutlass.Int32,
    dq_row_stride: cutlass.Int64,
    dk_row_stride: cutlass.Int64,
    dv_row_stride: cutlass.Int64,
) -> None:
    """Order work and build the sequence-relative descriptor arrays."""
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    if cutlass.const_expr(run_order):
        s_key = cutlass.Array(
            cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16
        )
        s_idx = cutlass.Array(
            cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16
        )
        s_spread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
        heads_out = cutlass.Int32(do.shape[1])
        order_body(
            order_gen,
            has_sched,
            b_t,
            ORDER_THREADS,
            ORDER_ELEMS,
            tidx,
            heads_out,
            heads_out * n_batch,
            cu_seqlens,
            work_item_staging,
            work_count,
            work_items,
            sched_all,
            s_key,
            s_idx,
            s_spread,
        )
    build_descs_body(
        widx,
        base_q,
        base_k,
        base_v,
        base_do,
        base_checkpoint,
        base_dq,
        base_dk,
        base_dv,
        desc_ws,
        cu_seqlens,
        q,
        k,
        v,
        do,
        state_checkpoints,
        dq,
        dk,
        dv,
        n_batch,
        q_row_stride,
        k_row_stride,
        v_row_stride,
        do_row_stride,
        checkpoint_row_stride,
        checkpoint_every_n,
        dq_row_stride,
        dk_row_stride,
        dv_row_stride,
    )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    has_sched: cutlass.Constexpr[bool],
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    state_checkpoints: cute.Tensor,
    cu_seqlens: cute.Tensor,
    work_item_staging: cute.Tensor | None,
    work_count: cute.Tensor,
    work_items: cute.Tensor | None,
    sched_all: cute.Tensor | None,
    tensormap_workspace: cute.Tensor,
    stream: cuda_driver.CUstream,
) -> None:
    """Build all TMA maps with sequence-relative token coordinates."""
    h_q, h_k, h_v = q.shape[1], k.shape[1], v.shape[1]
    heads_out = do.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    d_k, d_v = q.shape[2], v.shape[2]
    seqlen = q.shape[0]
    granule_elems = 128 // (io_dtype.width // 8)

    q_headed = cute.make_tensor(
        q.iterator, cute.make_layout((d_k, h_q, seqlen), stride=(1, q.stride[1], q.stride[0]))
    )
    k_headed = cute.make_tensor(
        k.iterator, cute.make_layout((d_k, h_k, seqlen), stride=(1, k.stride[1], k.stride[0]))
    )
    v_headed = cute.make_tensor(
        v.iterator, cute.make_layout((d_v, h_v, seqlen), stride=(1, v.stride[1], v.stride[0]))
    )
    do_headed = cute.make_tensor(
        do.iterator,
        cute.make_layout((d_v, heads_out, seqlen), stride=(1, do.stride[1], do.stride[0])),
    )
    dq_headed = cute.make_tensor(
        dq.iterator, cute.make_layout((d_k, h_q, seqlen), stride=(1, dq.stride[1], dq.stride[0]))
    )
    dk_headed = cute.make_tensor(
        dk.iterator, cute.make_layout((d_k, h_k, seqlen), stride=(1, dk.stride[1], dk.stride[0]))
    )
    dv_headed = cute.make_tensor(
        dv.iterator, cute.make_layout((d_v, h_v, seqlen), stride=(1, dv.stride[1], dv.stride[0]))
    )
    swizzle = cuda.TensorMapSwizzle.s128b
    base_q = cuda.create_tensor_map_tiled_from_view(
        q_headed, box_dims=(granule_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    base_k = cuda.create_tensor_map_tiled_from_view(
        k_headed, box_dims=(granule_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    base_v = cuda.create_tensor_map_tiled_from_view(
        v_headed, box_dims=(granule_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    base_do = cuda.create_tensor_map_tiled_from_view(
        do_headed, box_dims=(granule_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    base_dq = cuda.create_tensor_map_tiled_from_view(
        dq_headed, box_dims=(granule_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    base_dk = cuda.create_tensor_map_tiled_from_view(
        dk_headed, box_dims=(granule_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    base_dv = cuda.create_tensor_map_tiled_from_view(
        dv_headed, box_dims=(granule_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    checkpoint_view = cute.make_tensor(
        state_checkpoints.iterator,
        cute.make_layout(
            (
                state_checkpoints.shape[3],
                state_checkpoints.shape[2],
                state_checkpoints.shape[0],
                heads_out,
            ),
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
        box_dims=(64, state_checkpoints.shape[2], 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=swizzle,
    )

    prologue_kernel(
        run_order,
        order_gen,
        has_sched,
        b_t,
        base_q,
        base_k,
        base_v,
        base_do,
        base_checkpoint,
        base_dq,
        base_dk,
        base_dv,
        tensormap_workspace,
        cu_seqlens,
        q,
        k,
        v,
        do,
        state_checkpoints,
        dq,
        dk,
        dv,
        work_item_staging,
        work_count,
        work_items,
        sched_all,
        cutlass.Int32(batch_size),
        cutlass.Int64(q.stride[0]),
        cutlass.Int64(k.stride[0]),
        cutlass.Int64(v.stride[0]),
        cutlass.Int64(do.stride[0]),
        cutlass.Int64(state_checkpoints.stride[0]),
        cutlass.Int32(b_t),
        cutlass.Int64(dq.stride[0]),
        cutlass.Int64(dk.stride[0]),
        cutlass.Int64(dv.stride[0]),
    ).launch(grid=(1, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream)


class GdnBpropOp:
    """Compile and launch one static BT=64 scalar-GDN backward configuration."""

    def __init__(self, cfg: "GdnBwdCfg", use_int64_offsets: bool = False):
        self.cfg = cfg
        self.use_int64_offsets = use_int64_offsets

    def get_name(self) -> str:
        cfg = self.cfg
        return (
            f"gdn_mega_bprop_{cfg.io_dtype.__name__.lower()}_h{cfg.n_heads_out}"
            f"_q{cfg.q_ratio}_k{cfg.k_ratio}_v{cfg.v_ratio}"
            f"_s{int(cfg.use_dstate_in)}z{int(cfg.use_dstate0)}"
            f"l{int(cfg.log_gate)}g{int(cfg.safe_gate)}b{int(cfg.beta_sigmoid)}"
            f"i{int(cfg.use_initial_state)}d{int(cfg.dyn_sched)}"
            f"_sm{cfg.max_active_clusters}_i64{int(self.use_int64_offsets)}"
        )

    @cute.jit
    def __call__(
        self,
        gate: cute.Tensor,
        a_log: cute.Tensor | None,
        dt_bias: cute.Tensor | None,
        beta: cute.Tensor,
        dgate: cute.Tensor,
        dbeta: cute.Tensor,
        cu_seqlens: cute.Tensor,
        d_initial_state: cute.Tensor | None,
        d_final_state: cute.Tensor | None,
        work_items: cute.Tensor,
        work_count: cute.Tensor,
        sched_ctr: cute.Tensor | None,
        tensormap_workspace: cute.Tensor,
        scale: cutlass.Float32,
        stream,
    ) -> None:
        cfg = self.cfg
        num_sequences = cu_seqlens.shape[0] - 1

        @cute.struct
        class SharedStorage:
            q: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.q_cosize], cfg.buffer_align_bytes
            ]
            k: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.k_cosize], cfg.buffer_align_bytes
            ]
            do: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.do_cosize], cfg.buffer_align_bytes
            ]
            state: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.state_cosize], cfg.buffer_align_bytes
            ]
            t_inv: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.t_inv_cosize], cfg.buffer_align_bytes
            ]
            kk: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.t_inv_cosize], cfg.buffer_align_bytes
            ]
            a: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.a_cosize], cfg.buffer_align_bytes
            ]
            dm: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.a_cosize // cfg.smem_a_stages],
                cfg.buffer_align_bytes,
            ]
            v: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.v_cosize], cfg.buffer_align_bytes
            ]
            dstate: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.d_k * cfg.d_v], cfg.buffer_align_bytes
            ]
            dq: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.dq_cosize], cfg.buffer_align_bytes
            ]
            dk: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.dk_cosize], cfg.buffer_align_bytes
            ]
            dv: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.dv_cosize], cfg.buffer_align_bytes
            ]

        kernel.set_name_prefix(self.get_name())
        kernel(
            cfg,
            SharedStorage,
            tensormap_workspace,
            cutlass.Int32(num_sequences),
            gate,
            a_log,
            dt_bias,
            beta,
            dgate,
            dbeta,
            cu_seqlens,
            d_initial_state,
            d_final_state,
            work_items,
            work_count,
            sched_ctr,
            scale,
        ).launch(
            grid=(cfg.max_active_clusters, 1, 1),
            block=(cfg.threads_per_cta, 1, 1),
            cluster=cfg.cluster_shape_mnk,
            stream=stream,
            min_blocks_per_mp=1,
        )


@cute.kernel
def kernel(
    cfg: cutlass.Constexpr,
    shared_type: cutlass.Constexpr,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
    mGate: cute.Tensor,
    mA_log: cute.Tensor | None,
    mDt_bias: cute.Tensor | None,
    mBeta: cute.Tensor,
    mDgate: cute.Tensor,
    mDbeta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    mDstate0: cute.Tensor | None,
    mDstate_in: cute.Tensor | None,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mSched: cute.Tensor | None,
    scale: cutlass.Float32,
) -> None:
    """Main scalar-GDN BT=64 backward kernel with persistent warp roles."""
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]
    total_tiles = mCount[0]
    assert mGate.element_type == cutlass.Float32
    assert mBeta.element_type == cutlass.Float32
    assert mDgate.element_type == cutlass.Float32
    assert mDbeta.element_type == cutlass.Float32
    assert cu_seqlens.element_type in (cutlass.Int32, cutlass.Int64)
    if cutlass.const_expr(cfg.dyn_sched):
        assert mSched is not None, "mSched must be provided if dyn_sched is True"

    if cutlass.const_expr(mDstate0 is not None):
        mDstate0 = cute.make_tensor(
            mDstate0.iterator,
            cute.make_layout(
                (mDstate0.shape[2], mDstate0.shape[3], mDstate0.shape[1], mDstate0.shape[0]),
                stride=(
                    mDstate0.stride[2],
                    mDstate0.stride[3],
                    mDstate0.stride[1],
                    mDstate0.stride[0],
                ),
            ),
        )
    if cutlass.const_expr(mDstate_in is not None):
        mDstate_in = cute.make_tensor(
            mDstate_in.iterator,
            cute.make_layout(
                (
                    mDstate_in.shape[2],
                    mDstate_in.shape[3],
                    mDstate_in.shape[1],
                    mDstate_in.shape[0],
                ),
                stride=(
                    mDstate_in.stride[2],
                    mDstate_in.stride[3],
                    mDstate_in.stride[1],
                    mDstate_in.stride[0],
                ),
            ),
        )

    desc_base_words = tensormap_workspace.iterator.raw_ptr()
    desc_qwords = cutlass.Int32(TENSOR_MAP_QWORDS)
    arr_words = n_desc * desc_qwords
    desc_q_base = desc_base_words
    desc_k_base = desc_base_words + arr_words
    desc_v_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_do_base = desc_base_words + cutlass.Int32(3) * arr_words
    desc_checkpoint_base = desc_base_words + cutlass.Int32(4) * arr_words
    desc_dq_base = desc_base_words + cutlass.Int32(5) * arr_words
    desc_dk_base = desc_base_words + cutlass.Int32(6) * arr_words
    desc_dv_base = desc_base_words + cutlass.Int32(7) * arr_words

    SMEM = cutlass.AddressSpace.smem
    bars = make_gdn_bars(cfg)
    sSched = cutlass.Array(
        cutlass.Int32, cfg.sched_stages, space=cutlass.AddressSpace.smem, alignment=16
    )
    tmem_base_slot = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=16)
    cumsumlog_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_gate_stages))
    beta_smem_layout_staged = cute.make_layout((cfg.b_t, 1, cfg.smem_beta_stages))
    # Keep the scalar rings outside the 1 KiB-aligned operand struct so the
    # double-buffered GDN pipeline fits in Blackwell's shared-memory limit.
    cumsumlog_raw = cutlass.Array(
        cutlass.Float32,
        cute.cosize(cumsumlog_smem_layout_staged),
        space=SMEM,
        alignment=128,
    )
    cumprod_raw = cutlass.Array(
        cutlass.Float32,
        cute.cosize(cumsumlog_smem_layout_staged),
        space=SMEM,
        alignment=128,
    )
    beta_raw = cutlass.Array(
        cutlass.Float32,
        cute.cosize(beta_smem_layout_staged),
        space=SMEM,
        alignment=128,
    )
    storage = cutlass.utils.SmemAllocator().allocate(shared_type)

    bpe = cfg.io_dtype.width // 8
    SWZ = 2
    LEAD = 16
    STRIDE = 8 * 128
    KT_LEAD = (cfg.d_v // 2) * 128
    V_LEAD = (cfg.d_v // 2) * 128
    STATE_LEAD = cfg.d_k * 128
    sQ_raw = storage.q.get_tensor(cute.make_layout((cfg.q_cosize,)))
    sQ = SmemTile(
        base=smem_data_ptr(sQ_raw).toint(),
        elems_per_stage=(cfg.q_cosize // cfg.smem_q_stages) * bpe,
        stages=cfg.smem_q_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_raw = storage.k.get_tensor(cute.make_layout((cfg.k_cosize,)))
    sK = SmemTile(
        base=smem_data_ptr(sK_raw).toint(),
        elems_per_stage=(cfg.k_cosize // cfg.smem_k_stages) * bpe,
        stages=cfg.smem_k_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_trans = SmemTile(
        base=smem_data_ptr(sK_raw).toint(),
        elems_per_stage=(cfg.k_cosize // cfg.smem_k_stages) * bpe,
        stages=cfg.smem_k_stages,
        leading_byte_offset=KT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sQ_trans = SmemTile(
        base=smem_data_ptr(sQ_raw).toint(),
        elems_per_stage=(cfg.q_cosize // cfg.smem_q_stages) * bpe,
        stages=cfg.smem_q_stages,
        leading_byte_offset=KT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdO_raw = storage.do.get_tensor(cute.make_layout((cfg.do_cosize,)))
    sdO = SmemTile(
        base=smem_data_ptr(sdO_raw).toint(),
        elems_per_stage=(cfg.do_cosize // cfg.smem_do_stages) * bpe,
        stages=cfg.smem_do_stages,
        leading_byte_offset=V_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdO_kmaj = SmemTile(
        base=smem_data_ptr(sdO_raw).toint(),
        elems_per_stage=(cfg.do_cosize // cfg.smem_do_stages) * bpe,
        stages=cfg.smem_do_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sState_raw = storage.state.get_tensor(cute.make_layout((cfg.state_cosize,)))
    sState = SmemTile(
        base=smem_data_ptr(sState_raw).toint(),
        elems_per_stage=(cfg.state_cosize // cfg.smem_state_stages) * bpe,
        stages=cfg.smem_state_stages,
        leading_byte_offset=STATE_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sState_kmaj = SmemTile(
        base=smem_data_ptr(sState_raw).toint(),
        elems_per_stage=(cfg.state_cosize // cfg.smem_state_stages) * bpe,
        stages=cfg.smem_state_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sTinv_raw = storage.t_inv.get_tensor(cute.make_layout((cfg.t_inv_cosize,)))
    sTinv = SmemTile(
        base=smem_data_ptr(sTinv_raw).toint(),
        elems_per_stage=(cfg.t_inv_cosize // cfg.smem_t_inv_stages) * bpe,
        stages=cfg.smem_t_inv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sTinv_trans = SmemTile(
        base=smem_data_ptr(sTinv_raw).toint(),
        elems_per_stage=(cfg.t_inv_cosize // cfg.smem_t_inv_stages) * bpe,
        stages=cfg.smem_t_inv_stages,
        leading_byte_offset=(cfg.b_t // 2) * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sKK_raw = storage.kk.get_tensor(cute.make_layout((cfg.t_inv_cosize,)))
    sKK = SmemTile(
        base=smem_data_ptr(sKK_raw).toint(),
        elems_per_stage=(cfg.t_inv_cosize // cfg.smem_t_inv_stages) * bpe,
        stages=cfg.smem_t_inv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sA_raw = storage.a.get_tensor(cute.make_layout((cfg.a_cosize,)))
    sA = SmemTile(
        base=smem_data_ptr(sA_raw).toint(),
        elems_per_stage=(cfg.a_cosize // cfg.smem_a_stages) * bpe,
        stages=cfg.smem_a_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sA_trans = SmemTile(
        base=smem_data_ptr(sA_raw).toint(),
        elems_per_stage=(cfg.a_cosize // cfg.smem_a_stages) * bpe,
        stages=cfg.smem_a_stages,
        leading_byte_offset=(cfg.b_t // 2) * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDa = SmemTile(
        base=smem_data_ptr(sA_raw).toint(),
        elems_per_stage=(cfg.a_cosize // cfg.smem_a_stages) * bpe,
        stages=1,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDa_trans = SmemTile(
        base=smem_data_ptr(sA_raw).toint(),
        elems_per_stage=(cfg.a_cosize // cfg.smem_a_stages) * bpe,
        stages=1,
        leading_byte_offset=(cfg.b_t // 2) * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDm_raw = storage.dm.get_tensor(cute.make_layout((cfg.a_cosize // cfg.smem_a_stages,)))
    sDm = SmemTile(
        base=smem_data_ptr(sDm_raw).toint(),
        elems_per_stage=(cfg.a_cosize // cfg.smem_a_stages) * bpe,
        stages=1,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDm_trans = SmemTile(
        base=smem_data_ptr(sDm_raw).toint(),
        elems_per_stage=(cfg.a_cosize // cfg.smem_a_stages) * bpe,
        stages=1,
        leading_byte_offset=(cfg.b_t // 2) * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    # sub-bank split: V + sDstate + dQ + dK + dV allocated last
    sV_raw = storage.v.get_tensor(cute.make_layout((cfg.v_cosize,)))
    sV = SmemTile(
        base=smem_data_ptr(sV_raw).toint(),
        elems_per_stage=(cfg.v_cosize // cfg.smem_v_stages) * bpe,
        stages=cfg.smem_v_stages,
        leading_byte_offset=V_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sV_kmaj = SmemTile(
        base=smem_data_ptr(sV_raw).toint(),
        elems_per_stage=(cfg.v_cosize // cfg.smem_v_stages) * bpe,
        stages=cfg.smem_v_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDstate_raw = storage.dstate.get_tensor(cute.make_layout((cfg.d_k * cfg.d_v,)))
    sDstate = SmemTile(
        base=smem_data_ptr(sDstate_raw).toint(),
        elems_per_stage=cfg.d_k * cfg.d_v * bpe,
        stages=1,
        leading_byte_offset=STATE_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdQ_raw = storage.dq.get_tensor(cute.make_layout((cfg.dq_cosize,)))
    sdQ = SmemTile(
        base=smem_data_ptr(sdQ_raw).toint(),
        elems_per_stage=(cfg.dq_cosize // cfg.smem_dq_stages) * bpe,
        stages=cfg.smem_dq_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdK_raw = storage.dk.get_tensor(cute.make_layout((cfg.dk_cosize,)))
    sdK = SmemTile(
        base=smem_data_ptr(sdK_raw).toint(),
        elems_per_stage=(cfg.dk_cosize // cfg.smem_dk_stages) * bpe,
        stages=cfg.smem_dk_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdV_raw = storage.dv.get_tensor(cute.make_layout((cfg.dv_cosize,)))
    sdV = SmemTile(
        base=smem_data_ptr(sdV_raw).toint(),
        elems_per_stage=(cfg.dv_cosize // cfg.smem_dv_stages) * bpe,
        stages=cfg.smem_dv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sdV_kmaj = sdV
    sdstate_flat = cute.make_tensor(
        cute.make_ptr(
            cfg.io_dtype,
            smem_data_ptr(sDstate_raw).toint(),
            mem_space=cute.AddressSpace.smem,
            assumed_align=cfg.buffer_align_bytes,
        ),
        cute.make_layout(cfg.d_k * cfg.d_v),
    )
    sstate_flat = cute.make_tensor(
        cute.make_ptr(
            cfg.io_dtype,
            smem_data_ptr(sState_raw).toint(),
            mem_space=cute.AddressSpace.smem,
            assumed_align=cfg.buffer_align_bytes,
        ),
        cute.make_layout(cfg.state_cosize),
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

    # ---- mbarrier init (all threads) ---------------------------------------------
    for s_ in range(cfg.sched_stages):
        bars.mb_sched_ready[s_].init()
        bars.mb_sched_done[s_].init()
    for s in range(cfg.smem_q_stages):
        bars.mb_q_ready[s].init()
        bars.mb_q_mma_done[s].init()
        bars.mb_q_cg1_done[s].init()
    for s in range(cfg.smem_k_stages):
        bars.mb_k_ready[s].init()
        bars.mb_k_mma_done[s].init()
        bars.mb_k_cg0_done[s].init()
    for s in range(cfg.smem_v_stages):
        bars.mb_v_ready[s].init()
        bars.mb_v_mma_done[s].init()
    for s in range(cfg.smem_do_stages):
        bars.mb_do_ready[s].init()
        bars.mb_do_mma_done[s].init()
    for s in range(cfg.smem_state_stages):
        bars.mb_state_ready[s].init()
        bars.mb_state_mma_done[s].init()
    for s in range(cfg.smem_gate_stages):
        bars.mb_gate_ready[s].init()
        bars.mb_gate_done[s].init()
    for s in range(cfg.smem_beta_stages):
        bars.mb_beta_ready[s].init()
        bars.mb_beta_done[s].init()
    for s in range(cfg.tmem_dstate_acc_stages):
        bars.mb_dstate_acc_ready[s].init()
        bars.mb_dstate_scale_acc_done[s].init()
    for b in (
        bars.mb_du_scale_acc_ready,
        bars.mb_du_scale_acc_done,
        bars.mb_du_total_acc_ready,
        bars.mb_dk_scale_acc_ready,
        bars.mb_dk_scale_acc_done,
        bars.mb_dk_attn_acc_ready,
        bars.mb_dk_attn_acc_done,
        bars.mb_dk_total_acc_ready,
        bars.mb_dk_total_acc_done,
    ):
        b[0].init()
    for b in (
        bars.mb_kk_acc_ready,
        bars.mb_kk_acc_done,
        bars.mb_a_acc_ready,
        bars.mb_k_state_acc_ready,
        bars.mb_u_acc_ready,
        bars.mb_dy_acc_ready,
    ):
        b[0].init()
    for s in range(cfg.smem_t_inv_stages):
        bars.mb_t_inv_ready[s].init()
    for s in range(cfg.smem_a_stages):
        bars.mb_a_ready[s].init()
        bars.mb_a_done[s].init()
    for s in range(cfg.tmem_dstate_inp_stages):
        bars.mb_dstate_inp_ready[s].init()
        bars.mb_dstate_inp_done[s].init()
    for b in (
        bars.mb_do_prime_inp_ready,
        bars.mb_du_inp_ready,
        bars.mb_dyp_inp_ready,
    ):
        b[0].init()
    for s in range(cfg.smem_dq_stages):
        bars.mb_dq_tmastg_ready[s].init()
        bars.mb_dq_tmastg_done[s].init()
    for s in range(cfg.smem_dk_stages):
        bars.mb_dk_tmastg_ready[s].init()
        bars.mb_dk_tmastg_done[s].init()
    for s in range(cfg.smem_dv_stages):
        bars.mb_dv_tmastg_ready[s].init()
        bars.mb_dv_tmastg_done[s].init()
    bars.mb_y_ready[0].init()
    bars.mb_sdv_done[0].init()
    bars.mb_u_ready[0].init()
    bars.mb_dstate_smem_ready[0].init()
    bars.mb_da_ready[0].init()
    bars.mb_dq_acc_scale_ready[0].init()
    bars.mb_dq_acc_scale_done[0].init()
    bars.mb_dq_acc_total_ready[0].init()
    bars.mb_dq_acc_total_done[0].init()
    bars.mb_da_acc_ready[0].init()
    bars.mb_dm_acc_ready[0].init()
    bars.mb_dm_acc_done[0].init()
    bars.mb_dbeta_cg1_ready[0].init()
    bars.mb_dgate_cg1_ready[0].init()
    bars.mb_state_dot_dstate_done[0].init()
    bars.mb_dk_state_path_acc_ready[0].init()
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
            sCumprod=sCumprod,
            sBeta=sBeta,
            sTinv=sTinv,
            sKK=sKK,
            sA=sA,
            sDa=sDa,
            sDm=sDm,
            sK=sK,
            sdQ=sdQ,
            sDstate=sDstate,
            sstate_flat=sstate_flat,
            sdstate_flat=sdstate_flat,
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
            mDstate0,
            mDstate_in,
            tidx,
            warp_idx=warp_idx,
            tmem_base_slot=tmem_base_slot,
            scale=scale,
            sQ=sQ,
            sK=sK,
            sV=sV,
            sdO=sdO,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sBeta=sBeta,
            sdQ=sdQ,
            sdK=sdK,
            sdV=sdV,
            sDstate=sDstate,
            sDa=sDa,
            sDm=sDm,
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
            sQ=sQ,
            sQ_trans=sQ_trans,
            sK=sK,
            sK_trans=sK_trans,
            sV=sV,
            sV_kmaj=sV_kmaj,
            sdO=sdO,
            sdO_kmaj=sdO_kmaj,
            sState=sState,
            sState_kmaj=sState_kmaj,
            sTinv=sTinv,
            sTinv_trans=sTinv_trans,
            sA=sA,
            sA_trans=sA_trans,
            sDa=sDa,
            sDa_trans=sDa_trans,
            sDstate=sDstate,
            sDm=sDm,
            sDm_trans=sDm_trans,
            sdV_kmaj=sdV_kmaj,
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
            mSched=mSched,
            sQ_raw=sQ_raw,
            sK_raw=sK_raw,
            sV_raw=sV_raw,
            sdO_raw=sdO_raw,
            sState_raw=sState_raw,
            desc_q_base=desc_q_base,
            desc_k_base=desc_k_base,
            desc_v_base=desc_v_base,
            desc_do_base=desc_do_base,
            desc_checkpoint_base=desc_checkpoint_base,
            sSched=sSched,
            bars=bars,
        )

    if warp_idx == cfg.load_gate_beta_warp_id:
        gate_beta_warp(
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
            mBeta=mBeta,
            mDgate=mDgate,
            mDbeta=mDbeta,
            sCumsumlog=sCumsumlog,
            sCumprod=sCumprod,
            sBeta=sBeta,
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
            sdQ_raw=sdQ_raw,
            sdK_raw=sdK_raw,
            sdV_raw=sdV_raw,
            desc_dq_base=desc_dq_base,
            desc_dk_base=desc_dk_base,
            desc_dv_base=desc_dv_base,
            sSched=sSched,
            bars=bars,
        )


@dataclass
class GdnBwdCfg:
    """Static BT=64 scalar-GDN backward-kernel configuration."""

    io_dtype: type[cutlass.Numeric]
    use_initial_state: bool
    use_dstate_in: bool
    use_dstate0: bool
    q_ratio: int
    k_ratio: int
    v_ratio: int
    n_heads_out: int
    max_active_clusters: int
    dyn_sched: bool = False
    log_gate: bool = True
    safe_gate: bool = False
    beta_sigmoid: bool = False
    acc_dtype: type[cutlass.Numeric] = cutlass.Float32

    b_t: int = CFG.B_T
    d_k: int = CFG.D_K
    d_v: int = CFG.D_V
    compute_group_0_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_1_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    mma_warp_id: int = CFG.MMA_WARP_ID
    tma_qkv_warp_id: int = CFG.TMA_QKV_WARP_ID
    load_gate_beta_warp_id: int = CFG.LOAD_GATE_BETA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_other: int = CFG.NUM_REGS_OTHER
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0
    cluster_shape_mnk: tuple[int, int, int] = CFG.CLUSTER_SHAPE_MNK
    sched_stages: int = CFG.SMEM_SCHED_STAGES

    tmem_alloc_barrier_id: int = 1
    tmem_alloc_barrier_threads: int = 0
    inverse_barrier_id: int = 2
    inverse_barrier_threads: int = 0
    inverse_inner_barrier_id: int = 3
    inverse_inner_barrier_threads: int = 0
    init_state_store_barrier_id: int = 4
    init_state_store_barrier_threads: int = 0
    cg1_barrier_id: int = 5
    cg1_barrier_threads: int = 0

    smem_q_stages: int = CFG.SMEM_Q_STAGES
    smem_k_stages: int = CFG.SMEM_K_STAGES
    smem_v_stages: int = CFG.SMEM_V_STAGES
    smem_do_stages: int = 1
    smem_state_stages: int = 1
    smem_t_inv_stages: int = CFG.SMEM_T_INV_STAGES
    smem_a_stages: int = CFG.SMEM_A_STAGES
    smem_dq_stages: int = 1
    smem_dk_stages: int = 1
    smem_dv_stages: int = 1
    smem_gate_stages: int = 2
    smem_beta_stages: int = 2
    tmem_dstate_acc_stages: int = CFG.TMEM_DH_ACC_STAGES
    tmem_dvdk_acc_stages: int = CFG.TMEM_DVDK_ACC_STAGES
    tmem_dstate_inp_stages: int = CFG.TMEM_DH_INP_STAGES
    tmem_shared_inp_stages: int = CFG.TMEM_SHARED_INP_STAGES
    tmem_shared_acc_stages: int = CFG.TMEM_SHARED_ACC_STAGES
    tmem_dstate_acc_offset: int = 0
    tmem_dvdk_acc_offset: int = 0
    tmem_dstate_inp_offset: int = 0
    tmem_shared_acc_offset: int = 0
    tmem_shared_inp_offset: int = 0
    tmem_y_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    q_cosize: int = 0
    k_cosize: int = 0
    v_cosize: int = 0
    do_cosize: int = 0
    state_cosize: int = 0
    t_inv_cosize: int = 0
    a_cosize: int = 0
    dq_cosize: int = 0
    dk_cosize: int = 0
    dv_cosize: int = 0
    tma_q_bytes: int = 0
    tma_k_bytes: int = 0
    tma_v_bytes: int = 0
    tma_do_bytes: int = 0
    tma_state_bytes: int = 0


def build_cfg(
    io_dtype: type[cutlass.Numeric],
    *,
    use_initial_state: bool,
    use_dstate_in: bool,
    use_dstate0: bool,
    q_ratio: int,
    k_ratio: int,
    v_ratio: int,
    n_heads_out: int,
    max_active_clusters: int,
    dyn_sched: bool = False,
) -> GdnBwdCfg:
    """Build the fixed BT=64 natural-log scalar-GDN configuration."""
    if io_dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise ValueError(f"io_dtype={io_dtype} is unsupported; expected Float16 or BFloat16")
    if min(q_ratio, k_ratio, v_ratio, n_heads_out, max_active_clusters) <= 0:
        raise ValueError("head ratios, head count, and active-cluster count must be positive")

    cfg = GdnBwdCfg(
        io_dtype=io_dtype,
        use_initial_state=use_initial_state,
        use_dstate_in=use_dstate_in,
        use_dstate0=use_dstate0,
        q_ratio=q_ratio,
        k_ratio=k_ratio,
        v_ratio=v_ratio,
        n_heads_out=n_heads_out,
        max_active_clusters=max_active_clusters,
        dyn_sched=dyn_sched,
    )
    role_ids = (
        *cfg.compute_group_0_warp_ids,
        *cfg.compute_group_1_warp_ids,
        cfg.mma_warp_id,
        cfg.tma_qkv_warp_id,
        cfg.load_gate_beta_warp_id,
        cfg.epilogue_warp_id,
    )
    if tuple(sorted(role_ids)) != tuple(range(len(role_ids))):
        raise ValueError("warp roles must be disjoint and cover contiguous IDs from zero")
    for group in (cfg.compute_group_0_warp_ids, cfg.compute_group_1_warp_ids):
        if len(group) != 4 or group != tuple(range(group[0], group[-1] + 1)):
            raise ValueError("the fixed schedule requires contiguous four-warp compute groups")

    cfg.threads_per_cta = len(role_ids) * cfg.threads_per_warp
    cfg.tmem_alloc_barrier_threads = (
        1 + len(cfg.compute_group_0_warp_ids) + len(cfg.compute_group_1_warp_ids)
    ) * cfg.threads_per_warp
    cfg.inverse_barrier_threads = len(cfg.compute_group_0_warp_ids) * cfg.threads_per_warp
    cfg.inverse_inner_barrier_threads = 2 * cfg.threads_per_warp
    cfg.init_state_store_barrier_threads = len(cfg.compute_group_1_warp_ids) * cfg.threads_per_warp
    cfg.cg1_barrier_threads = len(cfg.compute_group_1_warp_ids) * cfg.threads_per_warp

    cfg.tmem_dstate_acc_offset = 0
    cfg.tmem_dvdk_acc_offset = cfg.tmem_dstate_acc_offset + cfg.tmem_dstate_acc_stages * 128
    cfg.tmem_dstate_inp_offset = cfg.tmem_dvdk_acc_offset + cfg.tmem_dvdk_acc_stages * 64
    cfg.tmem_shared_acc_offset = cfg.tmem_dstate_inp_offset + cfg.tmem_dstate_inp_stages * 64
    cfg.tmem_shared_inp_offset = cfg.tmem_shared_acc_offset + cfg.tmem_shared_acc_stages * 64
    cfg.tmem_y_offset = cfg.tmem_shared_inp_offset + cfg.tmem_shared_inp_stages * (cfg.b_t // 2)
    if cfg.tmem_y_offset + cfg.b_t // 2 > 512:
        raise ValueError("BT=64 GDN TMEM layout exceeds 512 columns")

    cfg.q_cosize = cfg.smem_q_stages * cfg.b_t * cfg.d_k
    cfg.k_cosize = cfg.smem_k_stages * cfg.b_t * cfg.d_k
    cfg.v_cosize = cfg.smem_v_stages * cfg.b_t * cfg.d_v
    cfg.do_cosize = cfg.smem_do_stages * cfg.b_t * cfg.d_v
    cfg.state_cosize = cfg.smem_state_stages * cfg.d_k * cfg.d_v
    cfg.t_inv_cosize = cfg.smem_t_inv_stages * cfg.b_t * cfg.b_t
    cfg.a_cosize = cfg.smem_a_stages * cfg.b_t * cfg.b_t
    cfg.dq_cosize = cfg.smem_dq_stages * cfg.b_t * cfg.d_k
    cfg.dk_cosize = cfg.smem_dk_stages * cfg.b_t * cfg.d_k
    cfg.dv_cosize = cfg.smem_dv_stages * cfg.b_t * cfg.d_v
    element_bytes = io_dtype.width // 8
    cfg.tma_q_bytes = cfg.b_t * cfg.d_k * element_bytes
    cfg.tma_k_bytes = cfg.b_t * cfg.d_k * element_bytes
    cfg.tma_v_bytes = cfg.b_t * cfg.d_v * element_bytes
    cfg.tma_do_bytes = cfg.b_t * cfg.d_v * element_bytes
    cfg.tma_state_bytes = cfg.d_k * cfg.d_v * element_bytes
    return cfg


TENSORMAP_DESC_ARRAYS = 8
TENSORMAP_STATIC_SLOTS = 0


@cache
def get_compiled_cache(
    io_dtype_str: str,
    h_q: int,
    h_k: int,
    h_v: int,
    use_dstate_in: bool,
    use_dstate0: bool,
    use_initial_state: bool,
    dyn_sched: bool,
    run_order: bool,
    order_gen: bool,
    use_int64_offsets: bool,
    device_index: int,
    device_major: int,
    device_minor: int,
    num_sm: int,
):
    return {}


def _validate_scalar_fp32(name, tensor, expected_shape) -> None:
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(tensor.shape)}")
    if str(tensor.dtype) != "torch.float32":
        raise ValueError(f"{name} must have dtype torch.float32, got {tensor.dtype}")
    if tensor.data_ptr() % 4:
        raise ValueError(f"{name} data pointer must be 4-byte aligned")
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} innermost stride must be one")
    if any(stride < 0 for stride in tensor.stride()[:-1]):
        raise ValueError(f"{name} outer strides must be nonnegative")


def chunk_gdn_bwd_sm100(
    q,
    k,
    v,
    gate,
    beta,
    do,
    state_checkpoints,
    dq,
    dk,
    dv,
    dgate,
    dbeta,
    cu_seqlens,
    scale: float,
    *,
    use_initial_state: bool = False,
    d_initial_state=None,
    d_final_state=None,
    work_items=None,
    work_count=None,
    sched_ctr=None,
    sched_all=None,
    work_item_scratch=None,
    order_in_prologue: bool = False,
    tensormap_workspace,
) -> None:
    """Run BT=64 scalar-GDN backward for packed ``[T, H, D]`` tensors.

    ``gate``, ``beta``, ``dgate``, and ``dbeta`` are scalar ``[T, H]`` float32
    tensors. ``gate`` is the natural logarithm of the decay. Checkpoints and
    optional initial/final-state gradients use ``[rows, H, V, K]`` and
    ``[B, H, V, K]`` layouts, respectively.
    """
    tma_tensors = (
        ("q", q),
        ("k", k),
        ("v", v),
        ("do", do),
        ("state_checkpoints", state_checkpoints),
        ("dq", dq),
        ("dk", dk),
        ("dv", dv),
    )
    for name, tensor in tma_tensors:
        validate_tma_tensor(name, tensor)
    for name, tensor in (("d_initial_state", d_initial_state), ("d_final_state", d_final_state)):
        if tensor is not None:
            validate_tma_tensor(name, tensor)
    validate_cu_seqlens(cu_seqlens, assumed_align=8)
    if tensormap_workspace.data_ptr() % 128:
        raise ValueError("tensormap_workspace data pointer must be 128-byte aligned")
    if str(tensormap_workspace.dtype) != "torch.int64":
        raise ValueError("tensormap_workspace must have dtype torch.int64")
    if work_items is None or work_count is None:
        raise ValueError("work_items and work_count are required")
    if str(work_items.dtype) != "torch.int32" or work_items.ndim != 2 or work_items.shape[1] != 8:
        raise ValueError("work_items must be an int32 [rows, 8] tensor")
    if str(work_count.dtype) != "torch.int32" or work_count.numel() != 1:
        raise ValueError("work_count must be a one-element int32 tensor")
    if work_items.stride(-1) != 1 or work_items.data_ptr() % 4 or work_count.data_ptr() % 4:
        raise ValueError("work_items and work_count must be compact and 4-byte aligned")
    if sched_ctr is not None and (
        str(sched_ctr.dtype) != "torch.int32" or sched_ctr.data_ptr() % 4
    ):
        raise ValueError("sched_ctr must be a 4-byte-aligned int32 tensor")

    tokens, h_q, d_k = q.shape
    h_k, h_v = k.shape[1], v.shape[1]
    h_out = max(h_q, h_v)
    if d_k != CFG.D_K or k.shape != (tokens, h_k, CFG.D_K) or v.shape != (tokens, h_v, CFG.D_V):
        raise ValueError("q, k, and v must have shape (T, H, 128)")
    if do.shape != (tokens, h_out, CFG.D_V):
        raise ValueError("do must have shape (T, H, 128) at the output head count")
    if h_q != h_out or h_k != h_out:
        raise ValueError("chunk_gdn_bwd_sm100 requires q and k head ratios of one")
    for name, tensor, expected_shape in (
        ("dq", dq, q.shape),
        ("dk", dk, k.shape),
        ("dv", dv, v.shape),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must match its corresponding input shape")
    _validate_scalar_fp32("gate", gate, (tokens, h_out))
    _validate_scalar_fp32("beta", beta, (tokens, h_out))
    _validate_scalar_fp32("dgate", dgate, gate.shape)
    _validate_scalar_fp32("dbeta", dbeta, beta.shape)

    for name, heads in (("HQ", h_q), ("HK", h_k), ("HV", h_v)):
        if h_out % heads:
            raise ValueError(f"{name}={heads} must divide H={h_out}")
    expected_state_shape = (None, h_out, CFG.D_V, CFG.D_K)
    if state_checkpoints.ndim != 4 or state_checkpoints.shape[1:] != expected_state_shape[1:]:
        raise ValueError("state_checkpoints must have shape [rows, H, 128, 128]")
    required_checkpoint_rows = checkpoint_capacity_bound(tokens, cu_seqlens.shape[0] - 1, CFG.B_T)
    if state_checkpoints.shape[0] < required_checkpoint_rows:
        raise ValueError(
            f"state_checkpoints requires at least {required_checkpoint_rows} rows, got {state_checkpoints.shape[0]}"
        )
    if str(state_checkpoints.dtype) != str(q.dtype):
        raise ValueError("state_checkpoints dtype must match q")
    state_shape = (cu_seqlens.shape[0] - 1, h_out, CFG.D_V, CFG.D_K)
    for name, tensor in (("d_initial_state", d_initial_state), ("d_final_state", d_final_state)):
        if tensor is not None:
            if tensor.shape != state_shape or str(tensor.dtype) != "torch.float32":
                raise ValueError(f"{name} must have shape {state_shape} and dtype torch.float32")
    if any(tensor.device != q.device for _, tensor in tma_tensors):
        raise ValueError("all TMA tensors must be on q.device")
    scalar_tensors = (
        gate,
        beta,
        dgate,
        dbeta,
        cu_seqlens,
        work_items,
        work_count,
        tensormap_workspace,
    )
    optional_tensors = (d_initial_state, d_final_state, sched_ctr, sched_all, work_item_scratch)
    if any(
        tensor.device != q.device
        for tensor in (*scalar_tensors, *optional_tensors)
        if tensor is not None
    ):
        raise ValueError("all tensors must be on q.device")

    use_dstate_in = d_final_state is not None
    use_dstate0 = d_initial_state is not None
    dyn_sched = sched_ctr is not None
    run_order = order_in_prologue
    order_gen = run_order and work_item_scratch is None
    has_sched = run_order
    if run_order and sched_all is None:
        raise ValueError("order_in_prologue requires sched_all")

    use_int64_offsets = requires_int64_abi(
        q,
        k,
        v,
        gate,
        beta,
        do,
        state_checkpoints,
        dq,
        dk,
        dv,
        dgate,
        dbeta,
        cu_seqlens,
        d_initial_state,
        d_final_state,
        work_item_scratch,
        work_items,
        work_count,
        sched_ctr,
        sched_all,
        tensormap_workspace,
    )
    device_index = tensor_device_index(q)
    if current_device() != device_index:
        raise ValueError("the active CUDA device must match q.device")
    device_properties = get_device_properties(device_index)
    num_sm = device_properties.multi_processor_count
    cache = get_compiled_cache(
        str(q.dtype),
        h_q,
        h_k,
        h_v,
        use_dstate_in,
        use_dstate0,
        use_initial_state,
        dyn_sched,
        run_order,
        order_gen,
        use_int64_offsets,
        device_index,
        device_properties.major,
        device_properties.minor,
        num_sm,
    )

    io_dtype = get_dtype(q.dtype)
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    if "compiled" not in cache:
        cfg = build_cfg(
            io_dtype,
            use_initial_state=use_initial_state,
            use_dstate_in=use_dstate_in,
            use_dstate0=use_dstate0,
            q_ratio=h_out // h_q,
            k_ratio=h_out // h_k,
            v_ratio=h_out // h_v,
            n_heads_out=h_out,
            max_active_clusters=num_sm,
            dyn_sched=dyn_sched,
        )
        op = GdnBpropOp(cfg, use_int64_offsets)
        tokens_sym, sequence_entries, sequences, work_rows, sched_entries, workspace_words = (
            sym_int() for _ in range(6)
        )
        scalar_tensor = partial(
            make_strided_signature_tensor,
            assumed_align=4,
            use_int64_offsets=use_int64_offsets,
        )
        cache["compiled"] = compile_tvm_ffi(
            op,
            scalar_tensor(cutlass.Float32, (tokens_sym, h_out)),
            None,
            None,
            scalar_tensor(cutlass.Float32, (tokens_sym, h_out)),
            scalar_tensor(cutlass.Float32, (tokens_sym, h_out)),
            scalar_tensor(cutlass.Float32, (tokens_sym, h_out)),
            make_cu_seqlens_signature(sequence_entries),
            make_strided_signature_tensor(
                cutlass.Float32,
                (sequences, h_out, CFG.D_V, CFG.D_K),
                assumed_align=16,
                use_int64_offsets=use_int64_offsets,
            )
            if use_dstate0
            else None,
            make_strided_signature_tensor(
                cutlass.Float32,
                (sequences, h_out, CFG.D_V, CFG.D_K),
                assumed_align=16,
                use_int64_offsets=use_int64_offsets,
            )
            if use_dstate_in
            else None,
            make_work_items_signature(work_rows),
            make_counter_signature(),
            make_counter_signature(sched_entries) if dyn_sched else None,
            make_workspace_signature(workspace_words),
            cutlass.Float32(scale),
        )

    if "prologue" not in cache:
        (
            tokens_sym,
            checkpoint_rows,
            sequence_entries,
            work_rows,
            sched_entries,
            workspace_words,
        ) = (sym_int() for _ in range(6))
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
            run_order,
            order_gen,
            has_sched,
            tma_tensor(io_dtype, (tokens_sym, h_q, CFG.D_K)),
            tma_tensor(io_dtype, (tokens_sym, h_k, CFG.D_K)),
            tma_tensor(io_dtype, (tokens_sym, h_v, CFG.D_V)),
            tma_tensor(io_dtype, (tokens_sym, h_out, CFG.D_V)),
            tma_tensor(io_dtype, (tokens_sym, h_q, CFG.D_K)),
            tma_tensor(io_dtype, (tokens_sym, h_k, CFG.D_K)),
            tma_tensor(io_dtype, (tokens_sym, h_v, CFG.D_V)),
            tma_tensor(io_dtype, (checkpoint_rows, h_out, CFG.D_V, CFG.D_K)),
            make_cu_seqlens_signature(sequence_entries),
            work_items_signature if run_order and not order_gen else None,
            make_counter_signature(),
            work_items_signature,
            make_counter_signature(sched_entries) if has_sched else None,
            make_workspace_signature(workspace_words),
            name=(
                f"gdn_mega_bprop_prologue_{io_dtype.__name__.lower()}"
                f"_hq{h_q}_hk{h_k}_hv{h_v}_ho{h_out}_r{int(run_order)}"
                f"o{int(order_gen)}d{int(dyn_sched)}_i64{int(use_int64_offsets)}"
            ),
        )

    cache["prologue"](
        q,
        k,
        v,
        do,
        dq,
        dk,
        dv,
        state_checkpoints,
        cu_seqlens,
        work_item_scratch if run_order and not order_gen else None,
        work_count,
        work_items,
        sched_all if run_order else None,
        tensormap_workspace,
    )
    cache["compiled"](
        gate,
        None,
        None,
        beta,
        dgate,
        dbeta,
        cu_seqlens,
        d_initial_state,
        d_final_state,
        work_items,
        work_count,
        sched_ctr,
        tensormap_workspace,
        scale,
    )
