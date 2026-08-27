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

"""Chunked Kimi Delta Attention (KDA) BPROP kernel for Blackwell SM100/SM103
(Cutlass DSL), BT=16 tiling with a per-key-channel decay.  Framework-neutral
entry ``chunk_kda_bwd_sm100``.

Algorithm overview (per chunk c, iterated c = NT-1 .. 0; within-chunk
log2-domain gate cumsum G[t,d], eG = 2^G, eGl = 2^G[BT-1]):
  Inputs : Q/K[BT,DK], V/dO[BT,DV], S = state_checkpoints[c] (state ENTERING chunk c, KV),
           Gate[BT,DK], Beta[BT]
  State  : dH[DV,DK] (state gradient, fp32 TMEM, accumulated backward)

  Operands (WG0, prefill recompute): K_decay = eG.K, K_inv = K/eG,
  K_restore = (eGl/eG).K, Q_decay = eG.Q, diag(eGl).

  Forward recompute: A_kk = K_decay@K_inv^T; T_inv = (I + Beta.tril(A_kk,-1))^-1
  (register Neumann); A = tril_incl(Q_decay@K_inv^T); Y = Beta.(V - S^T K_decay);
  U = T_inv@Y.

  Backward math:
  dU  = K_restore@dH + A^T@dO (Q_decay carries scale, so A does too)
  dY  = T_inv^T@dU                    dV = Beta.dY
  dA  = tril_incl(dO@U^T) (unscaled)  dM = dY@U^T        dM_strict = Beta_row.strict(dM)
  dQ  = eG.scale.(dO@S^T + dA@K_inv)
  dK  = eG.dK_decay + dK_inv/eG + (eGl/eG).dK_restore  where (sign-flipped parts)
        dK_decay part = (Beta.dY)@S^T + dM_strict@K_inv   (= -dK_decay)
        dK_inv part = dA^T@(scale.Q_decay) - dM_strict^T@K_decay   (= dK_inv; one TMEM
                acc, the minus rides the staged -dM_strict tile)
        dK_restore part = U@dH^T                     (= +dK_restore)
  dBeta = sum_v dY.Y / Beta - sum_{j<i} dM.tril(A_kk,-1)
  dGate[t,d] = Q_n.dQ_pre + K_n.(-eG.dK_decay_part + dK_inv_part/eG
            - (eGl/eG).dK_restore_part)
  dGate_last[d] = eGl.sum_v(dH.S) + sum_t K_n.(eGl/eG).dK_restore_part
  dGate = suffix-sum(dGate + dGate_last at row BT-1) (WG2 in-register reverse cumsum)
  dH <- diag-GEMM(eGl).dH + (scale.Q_decay)^T@dO - K_decay^T@(Beta.dY)

ABI: state_checkpoints `[total_checkpoints, HO, DV, DK]` (VK, k contiguous) io
dtype, the plain per-chunk checkpoint series (entry `c` = state entering chunk
c, so row 0 is the initial state or zeros); dq/dk/dv io at HO heads; dgate `[T, HO, DK]` fp32
(natural-log gate domain; with SAFE_GATE the gradient stays wrt the
transformed log-decay); dbeta `[T, HO]` fp32, io dtype with BETA_SIGMOID
(post-sigmoid space, or wrt the raw logits under BETA_SIGMOID).  Gate
arrives natural-log fp32 unless SAFE_GATE (safe-gate transform from raw gate
+ a_log/dt_bias); beta arrives post-sigmoid fp32, or io-dtype logits with
BETA_SIGMOID; d_initial_state / d_final_state fp32 `[N, HO, DV, DK]`
(V-major).

Warp assignments (16 warps = 512 threads):
  warps 0-3  : WG0 - Gate prefix scan + decay/restore operands (all chunks) + Beta scalar gather
  warps 4-7  : WG1 - value-side TMEM staging, dstate capture, dBeta
  warps 8-11 : WG2 - dQ/dK part drain, dGate assembly, reverse cumsum, dGate
  warp  12   : super-MMA - register KK/A/dA/dM + Neumann inverse
  warp  13   : tcgen05-MMA - the 15-GEMM backward schedule
  warp  14   : TMA load - Q/K/V/Gate/dO/entering-state loads
  warp  15   : epilogue - dQ/dK/dV/dGate TMA stores
"""

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import NamedTuple, Type

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.experimental.cuda as cuda
import cutlass.experimental.primitives as nvvm
import cutlass.cute as cute

from attn_gym._backends.cute import compile_tvm_ffi
from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.linear._delta_rule.mega.kernels.common.split_k import ORDER_CAPACITY, ORDER_ELEMS, ORDER_THREADS, decode_work_item, order_body
from attn_gym.linear._delta_rule.mega.kernels.common.host import get_dtype
from attn_gym.linear._delta_rule.mega.kernels.compat import (
    checkpoint_capacity_bound,
    current_device,
    get_device_properties,
    tensor_device_index,
    validate_tma_tensor,
)
from attn_gym.linear._delta_rule.mega.kernels.common.thd import (
    TENSOR_MAP_QWORDS,
    emit_checkpoint_seq_descs,
    emit_seq_descs,
    emit_seq_load_descs,
)
from attn_gym.linear._delta_rule.mega.kernels.common.tvm_ffi import (
    make_compact_signature_tensor,
    make_counter_signature,
    make_cu_seqlens_signature,
    make_strided_signature_tensor,
    make_work_items_signature,
    make_workspace_signature,
    validate_cu_seqlens,
)
from .kda_bprop_config import CFG

from attn_gym.linear._delta_rule.mega.kernels.tile_dsl.barrier import (
    advance,
    MBarrier,
    PipelineState,
    Producer,
)
from attn_gym.linear._delta_rule.mega.kernels.tile_dsl.handles import (
    MmaDesc,
    SmemTile,
    smem_data_ptr,
    tma_slice_runtime_desc,
)
from attn_gym.linear._delta_rule.mega.kernels.tile_dsl.mma import mma_ss, mma_step, mma_ts_step
from attn_gym.linear._delta_rule.mega.kernels.tile_dsl.swizzle import swizzle_lin_S, swizzle_xor_128b
from attn_gym.linear._delta_rule.mega.kernels.tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait, tma_tensormap_acquire
from attn_gym.linear._delta_rule.mega.kernels.tile_dsl.pointwise import (
    f16x2_to_f32,
    fadd2,
    ffma2,
    fmul2,
    fp32_to_fp16,
    movmatrix_16b,
    mul_f16x2,
    opaque_f32_zero,
    sub_f16x2,
)

LOG2_E: float = 1.4426950408889634


DEFAULT_GATE_LOWER_BOUND: float = -5.0


L2_NORM_EPS: float = 1.0e-12


class KdaBwdBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring.  Consumers
    track ``(idx, phase)`` inline; the producer tag selects the arrive
    lowering (``TMA_LOAD``/``MMA_COMMIT``/``THREAD``).

    Buffers read by both the MMA warp and a compute/warp group carry mixed
    arrive counts (one MMA commit + N thread arrivers) so the producer only
    reuses the slot once every reader is done."""

    mb_q_ready: MBarrier
    mb_q_done: MBarrier
    mb_k_ready: MBarrier
    mb_k_done: MBarrier
    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier
    mb_do_ready: MBarrier
    mb_do_done: MBarrier
    mb_v_ready: MBarrier
    mb_v_done: MBarrier
    mb_state_ready: MBarrier
    mb_state_done: MBarrier
    mb_state_cg0_done: MBarrier

    mb_beta_ready: MBarrier
    mb_beta_done: MBarrier

    mb_state_k_acc_ready: MBarrier
    mb_du_acc_ready: MBarrier
    mb_u_acc_ready: MBarrier
    mb_dy_acc_ready: MBarrier
    mb_dq_acc_ready: MBarrier
    mb_dk_decay_part_acc_ready: MBarrier
    mb_dk_inv_part_acc_ready: MBarrier
    mb_dk_restore_part_acc_ready: MBarrier
    mb_dqk_acc_done: MBarrier

    mb_qk_raw_ready: MBarrier
    mb_qk_raw_done: MBarrier
    mb_state_inp_ready: MBarrier
    mb_state_inp_done: MBarrier
    mb_state_inp_cg2_done: MBarrier
    mb_y_inp_ready: MBarrier
    mb_du_inp_ready: MBarrier
    mb_neg_beta_dy_inp_ready: MBarrier

    mb_k_decay_inv_ready: MBarrier
    mb_q_decay_k_restore_ready: MBarrier
    mb_decay_done: MBarrier
    mb_t_inv_ready: MBarrier
    mb_t_inv_done: MBarrier
    mb_a_ready: MBarrier
    mb_a_done: MBarrier
    mb_da_ready: MBarrier
    mb_da_done: MBarrier
    mb_dm_ready: MBarrier
    mb_dm_done: MBarrier
    mb_u_smem_ready: MBarrier
    mb_dy_smem_ready: MBarrier
    mb_dbeta_m_ready: MBarrier

    mb_dstate_acc_ready: MBarrier
    mb_dstate_inp_ready: MBarrier
    mb_dstate_smem_ready: MBarrier
    mb_dstate_smem_done: MBarrier
    mb_dstate_smem_cg2_done: MBarrier

    mb_dq_tmastg_ready: MBarrier
    mb_dq_tmastg_done: MBarrier
    mb_dk_tmastg_ready: MBarrier
    mb_dk_tmastg_done: MBarrier
    mb_dv_tmastg_ready: MBarrier
    mb_dv_tmastg_done: MBarrier
    mb_dgate_tmastg_ready: MBarrier
    mb_dgate_tmastg_done: MBarrier

    mb_dstate0_acc_stored: MBarrier
    mb_tmem_done: MBarrier

    mb_sched_ready: MBarrier
    mb_sched_done: MBarrier


def make_kda_bwd_bars(cfg) -> KdaBwdBars:
    """Bars factory.  MUST be called from inside ``kernel`` (allocates the
    mbarrier rings in SMEM ahead of the data buffers)."""

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=8)

    WARP = cfg.threads_per_warp
    CG0 = len(cfg.compute_group_0_warp_ids) * WARP
    CG2 = len(cfg.compute_group_2_warp_ids) * WARP
    CG1 = len(cfg.compute_group_1_warp_ids) * WARP
    SCHED_CONSUMER_WARPS = cfg.threads_per_cta // WARP - 1
    MMA = 1

    return KdaBwdBars(
        mb_q_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_q_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0, producer=Producer.THREAD),
        mb_k_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_k_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0, producer=Producer.THREAD),
        mb_gate_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_gate_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0 + CG2, producer=Producer.THREAD),
        mb_do_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_do_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=WARP, producer=Producer.THREAD),
        mb_v_ready=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_v_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG1, producer=Producer.THREAD),
        mb_state_ready=MBarrier(alloc(cfg.smem_state_stages), stages=cfg.smem_state_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_state_done=MBarrier(alloc(cfg.smem_state_stages), stages=cfg.smem_state_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_state_cg0_done=MBarrier(alloc(cfg.smem_state_stages), stages=cfg.smem_state_stages, init_count=CG0, producer=Producer.THREAD),
        mb_beta_ready=MBarrier(alloc(cfg.smem_beta_stages), stages=cfg.smem_beta_stages, init_count=WARP, producer=Producer.THREAD),
        mb_beta_done=MBarrier(alloc(cfg.smem_beta_stages), stages=cfg.smem_beta_stages, init_count=WARP + CG1, producer=Producer.THREAD),
        mb_state_k_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_du_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_u_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dy_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dq_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dk_decay_part_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dk_inv_part_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dk_restore_part_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dqk_acc_done=MBarrier(alloc(1), stages=1, init_count=CG2, producer=Producer.THREAD),
        mb_qk_raw_ready=MBarrier(alloc(cfg.tmem_qk_raw_stages), stages=cfg.tmem_qk_raw_stages, init_count=CG0, producer=Producer.THREAD),
        mb_qk_raw_done=MBarrier(alloc(cfg.tmem_qk_raw_stages), stages=cfg.tmem_qk_raw_stages, init_count=CG2, producer=Producer.THREAD),
        mb_state_inp_ready=MBarrier(alloc(2), stages=2, init_count=CG0, producer=Producer.THREAD),
        mb_state_inp_done=MBarrier(alloc(2), stages=2, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_state_inp_cg2_done=MBarrier(alloc(2), stages=2, init_count=CG2, producer=Producer.THREAD),
        mb_y_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_du_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_neg_beta_dy_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_k_decay_inv_ready=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=CG0, producer=Producer.THREAD),
        mb_q_decay_k_restore_ready=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=CG0, producer=Producer.THREAD),
        mb_decay_done=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_t_inv_ready=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=WARP, producer=Producer.THREAD),
        mb_t_inv_done=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_a_ready=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=WARP, producer=Producer.THREAD),
        mb_a_done=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_da_ready=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=WARP, producer=Producer.THREAD),
        mb_da_done=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dm_ready=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=WARP, producer=Producer.THREAD),
        mb_dm_done=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_u_smem_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_dy_smem_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_dbeta_m_ready=MBarrier(alloc(1), stages=1, init_count=WARP, producer=Producer.THREAD),
        mb_dstate_acc_ready=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dstate_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_dstate_smem_ready=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_dstate_smem_done=MBarrier(alloc(1), stages=1, init_count=MMA, producer=Producer.MMA_COMMIT),
        mb_dstate_smem_cg2_done=MBarrier(alloc(1), stages=1, init_count=CG2, producer=Producer.THREAD),
        mb_dq_tmastg_ready=MBarrier(alloc(cfg.smem_dq_stages), stages=cfg.smem_dq_stages, init_count=CG2, producer=Producer.THREAD),
        mb_dq_tmastg_done=MBarrier(alloc(cfg.smem_dq_stages), stages=cfg.smem_dq_stages, init_count=WARP, producer=Producer.THREAD),
        mb_dk_tmastg_ready=MBarrier(alloc(cfg.smem_dk_stages), stages=cfg.smem_dk_stages, init_count=CG2, producer=Producer.THREAD),
        mb_dk_tmastg_done=MBarrier(alloc(cfg.smem_dk_stages), stages=cfg.smem_dk_stages, init_count=WARP, producer=Producer.THREAD),
        mb_dv_tmastg_ready=MBarrier(alloc(cfg.smem_dv_stages), stages=cfg.smem_dv_stages, init_count=CG1, producer=Producer.THREAD),
        mb_dv_tmastg_done=MBarrier(alloc(cfg.smem_dv_stages), stages=cfg.smem_dv_stages, init_count=WARP, producer=Producer.THREAD),
        mb_dgate_tmastg_ready=MBarrier(alloc(cfg.smem_dgate_stages), stages=cfg.smem_dgate_stages, init_count=CG2, producer=Producer.THREAD),
        mb_dgate_tmastg_done=MBarrier(alloc(cfg.smem_dgate_stages), stages=cfg.smem_dgate_stages, init_count=WARP, producer=Producer.THREAD),
        mb_dstate0_acc_stored=MBarrier(alloc(1), stages=1, init_count=CG1, producer=Producer.THREAD),
        mb_tmem_done=MBarrier(alloc(1), stages=1, init_count=CG1 + CG2, producer=Producer.THREAD),
        mb_sched_ready=MBarrier(alloc(cfg.sched_stages), stages=cfg.sched_stages, init_count=1, producer=Producer.THREAD),
        mb_sched_done=MBarrier(
            alloc(cfg.sched_stages),
            stages=cfg.sched_stages,
            init_count=SCHED_CONSUMER_WARPS,
            producer=Producer.THREAD,
        ),
    )


# ---- Dynamic tile scheduler ------------------------------------------------------


@cute.jit
def sched_publish_next(cfg, bars, sSched, mSched, sched_state, tile_idx, num_ctas):
    """TMA-warp side: pull the next tile off the global ticket, publish it."""
    if cutlass.const_expr(cfg.dyn_sched):
        bars.mb_sched_done[sched_state.idx].wait(sched_state.phase)
        if nvvm.elect_sync():
            fetched = cutlass.Int32(nvvm.atomicrmw("add", mSched.iterator, cutlass.Int32(1), mem_order="relaxed", syncscope="gpu"))
            sSched[sched_state.idx] = num_ctas + fetched
        nvvm.bar_warp_sync(cute.arch.FULL_MASK)
        next_tile = sSched[sched_state.idx]
        if nvvm.elect_sync():
            bars.mb_sched_ready[sched_state.idx].arrive()
        return next_tile, advance(sched_state, cfg.sched_stages)
    return tile_idx + num_ctas, sched_state


@cute.jit
def sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas):
    """Consumer side: read the TMA warp's published next tile."""
    if cutlass.const_expr(cfg.dyn_sched):
        bars.mb_sched_ready[sched_state.idx].wait(sched_state.phase)
        next_tile = sSched[sched_state.idx]
        if nvvm.elect_sync():
            bars.mb_sched_done[sched_state.idx].arrive()
        return next_tile, advance(sched_state, cfg.sched_stages)
    return tile_idx + num_ctas, sched_state


# ---- Warp bodies -----------------------------------------------------------------


@cute.jit
def epilogue_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sSched,
    lane,
    sK_inv_raw,
    sQ_decay_raw,
    sDo_raw,
    sU_raw,
    sIntermediate_raw,
    sDq_raw,
    sDk_raw,
    sDv_raw,
    sDgate_raw,
    desc_dq_base,
    desc_dk_base,
    desc_dv_base,
    desc_dgate_base,
    bars,
) -> None:
    """Epilogue warp role (warp 15): the register-MMA A/dA tiles and the
    dQ/dK/dV/dGate TMA stores, in chunk order with a one-behind store
    ladder."""
    elect_one = nvvm.elect_sync()

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    # ---- ldmatrix/stmatrix lane decode -------------------------------------------
    rhs_row_coord = lane % 8 + (cutlass.Int32(8) if (lane // 16) else cutlass.Int32(0))
    rhs_col_offset = cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0)
    lhs_row_coord = lane % 8 + (cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0))
    lhs_col_offset = cutlass.Int32(8) if ((lane // 8) // 2) else cutlass.Int32(0)
    stsm_row_coord = lane & 7
    stsm_col_coord = cutlass.Int32(0)
    if (lane // 8) & 1:
        stsm_row_coord = stsm_row_coord + cutlass.Int32(8)
    if lane // 8 >= 2:
        stsm_col_coord = cutlass.Int32(8)
    stsm_idx = swizzle_lin_S(stsm_row_coord * cfg.b_t + stsm_col_coord, bbits=1, mbase=3, sshift=3)
    row_lo = lane // 4
    row_hi = row_lo + cutlass.Int32(8)

    tril_incl_mask = cutlass.Int32(0)
    for accum_idx in cutlass.range_constexpr(8):
        row_coord = row_hi if cutlass.const_expr(accum_idx % 4 >= 2) else row_lo
        col_coord = (accum_idx // 4) * 8 + 2 * (lane % 4)
        if cutlass.const_expr(accum_idx % 2 == 1):
            col_coord = col_coord + cutlass.Int32(1)
        tril_incl_mask = tril_incl_mask | (cutlass.Int32(1 << accum_idx) if row_coord >= col_coord else cutlass.Int32(0))
    raw_index = PipelineState.start(phase=0)
    u_index = PipelineState.start(phase=0)
    chunk_serial_base = cutlass.Int32(0)

    sDq_tma = SmemTile(
        base=sDq_raw,
        elems_per_stage=(cfg.b_t * cfg.d_k),
        stages=cfg.smem_dq_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=cfg.b_t * 64,
    )
    sDk_tma = SmemTile(
        base=sDk_raw,
        elems_per_stage=(cfg.b_t * cfg.d_k),
        stages=cfg.smem_dk_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=cfg.b_t * 64,
    )
    sDv_tma = SmemTile(
        base=sDv_raw,
        elems_per_stage=(cfg.b_t * cfg.d_v),
        stages=cfg.smem_dv_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_v // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=cfg.b_t * 64,
    )
    sDgate_tma = SmemTile(
        base=sDgate_raw,
        elems_per_stage=(cfg.b_t * cfg.d_k),
        stages=cfg.smem_dgate_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 32),
        tma_granu_elems=32,
        tma_subtile_stride_elems=cfg.b_t * 32,
    )
    dq_index = PipelineState.start(phase=0)
    dk_index = PipelineState.start(phase=0)
    dv_index = PipelineState.start(phase=0)
    dgate_index = PipelineState.start(phase=0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        head_o = head_idx
        slot = batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
        if elect_one:
            desc_dq_slot = (desc_dq_base + slot).tospace(cutlass.AddressSpace.generic)
            desc_dk_slot = (desc_dk_base + slot).tospace(cutlass.AddressSpace.generic)
            desc_dv_slot = (desc_dv_base + slot).tospace(cutlass.AddressSpace.generic)
            desc_dgate_slot = (desc_dgate_base + slot).tospace(cutlass.AddressSpace.generic)
            tma_tensormap_acquire(desc_dq_slot)
            tma_tensormap_acquire(desc_dk_slot)
            tma_tensormap_acquire(desc_dv_slot)
            tma_tensormap_acquire(desc_dgate_slot)
        num_compute_chunks = cend - wstart
        pend_start = cutlass.Int32(0)
        pend_writes = cutlass.Boolean(False)
        for rev_idx in cutlass.range(num_compute_chunks, unroll=1):
            chunk_idx = cend - cutlass.Int32(1) - rev_idx
            chunk_start = chunk_idx * cfg.b_t
            writes = chunk_idx < wend
            chunk_serial = chunk_serial_base + rev_idx
            decay_stage = chunk_serial % cfg.smem_decay_stages
            intermediate_stage = chunk_serial % cfg.smem_intermediate_stages
            raw_stage = raw_index.idx
            sK_inv_ptr = smem_data_ptr(sK_inv_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sQ_decay_ptr = smem_data_ptr(sQ_decay_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sDo_ptr = smem_data_ptr(sDo_raw) + raw_stage * (cfg.d_v * cfg.b_t)
            sIntermediate_ptr = smem_data_ptr(sIntermediate_raw) + intermediate_stage * (cfg.intermediate_tiles * cfg.b_t * cfg.b_t)

            # ---- A = tril_incl(Q_decay @ K_inv^T) --------------------------------
            bars.mb_a_done[intermediate_stage].wait(((chunk_serial // cfg.smem_intermediate_stages) + 1) % 2)
            bars.mb_q_decay_k_restore_ready[decay_stage].wait((chunk_serial // cfg.smem_decay_stages) % 2)
            a_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                a_acc[accum_idx] = cutlass.Float32(0.0)
            for k_block in cutlass.range_constexpr(cfg.d_k // 16):
                a_col = k_block * 16 + lhs_col_offset
                a_seg = a_col // 64
                a_frag = nvvm.ldmatrix(
                    sQ_decay_ptr + a_seg * (cfg.b_t * 64) + lhs_row_coord * 64 + swizzle_xor_128b(lhs_row_coord, a_col - a_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                b_col = k_block * 16 + rhs_col_offset
                b_seg = b_col // 64
                b_frag = nvvm.ldmatrix(
                    sK_inv_ptr + b_seg * (cfg.b_t * 64) + rhs_row_coord * 64 + swizzle_xor_128b(rhs_row_coord, b_col - b_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                mma_step(
                    a_acc,
                    (a_frag[0], a_frag[1], a_frag[2], a_frag[3]),
                    (b_frag[0], b_frag[1], b_frag[2], b_frag[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )
            for accum_idx in cutlass.range_constexpr(8):
                a_acc[accum_idx] = a_acc[accum_idx] if (tril_incl_mask >> accum_idx) & 1 else cutlass.Float32(0.0)
            nvvm.stmatrix(
                sIntermediate_ptr + stsm_idx,
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

            # ---- dA = tril_incl(dO @ U^T) ----------------------------------------
            bars.mb_u_smem_ready.wait(u_index.phase)
            u_index = advance(u_index, 1)
            da_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                da_acc[accum_idx] = cutlass.Float32(0.0)
            for k_block in cutlass.range_constexpr(cfg.d_v // 16):
                a_col = k_block * 16 + lhs_col_offset
                a_seg = a_col // 64
                a_frag = nvvm.ldmatrix(
                    sDo_ptr + a_seg * (cfg.b_t * 64) + lhs_row_coord * 64 + swizzle_xor_128b(lhs_row_coord, a_col - a_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                b_col = k_block * 16 + rhs_col_offset
                b_seg = b_col // 64
                b_frag = nvvm.ldmatrix(
                    smem_data_ptr(sU_raw) + b_seg * (cfg.b_t * 64) + rhs_row_coord * 64 + swizzle_xor_128b(rhs_row_coord, b_col - b_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                mma_step(
                    da_acc,
                    (a_frag[0], a_frag[1], a_frag[2], a_frag[3]),
                    (b_frag[0], b_frag[1], b_frag[2], b_frag[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )
            # fence: the dO/U ldmatrix reads must complete before this release
            # licenses the TMA reload (sDo)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_do_done[raw_stage].arrive()
            for accum_idx in cutlass.range_constexpr(8):
                da_acc[accum_idx] = da_acc[accum_idx] if (tril_incl_mask >> accum_idx) & 1 else cutlass.Float32(0.0)
            bars.mb_da_done[intermediate_stage].wait(((chunk_serial // cfg.smem_intermediate_stages) + 1) % 2)
            nvvm.stmatrix(
                sIntermediate_ptr + 2 * (cfg.b_t * cfg.b_t) + stsm_idx,
                [
                    fp32_to_fp16(da_acc[0], da_acc[1], dtype=cfg.io_dtype),
                    fp32_to_fp16(da_acc[2], da_acc[3], dtype=cfg.io_dtype),
                    fp32_to_fp16(da_acc[4], da_acc[5], dtype=cfg.io_dtype),
                    fp32_to_fp16(da_acc[6], da_acc[7], dtype=cfg.io_dtype),
                ],
                nvvm.MMALayout.ROW,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_da_ready[intermediate_stage].arrive()
            raw_index = advance(raw_index, cfg.smem_raw_stages)

            # ---- dQ/dK/dGate/dV: previous chunk, one-behind store ladder ---------
            if rev_idx > 0:
                bars.mb_dq_tmastg_ready[dq_index.idx].wait(dq_index.phase)
                if pend_writes:
                    desc_dq_slot = (desc_dq_base + slot).tospace(cutlass.AddressSpace.generic)
                    dq_slice = tma_slice_runtime_desc(desc_dq_slot, cutlass.Int32(0), head_o, pend_start)
                    tma_store_tile(sDq_tma[dq_index.idx], dq_slice, acquire=False)
                    tma_store_commit()
                bars.mb_dk_tmastg_ready[dk_index.idx].wait(dk_index.phase)
                if pend_writes:
                    desc_dk_slot = (desc_dk_base + slot).tospace(cutlass.AddressSpace.generic)
                    dk_slice = tma_slice_runtime_desc(desc_dk_slot, cutlass.Int32(0), head_o, pend_start)
                    tma_store_tile(sDk_tma[dk_index.idx], dk_slice, acquire=False)
                    tma_store_commit()
                bars.mb_dgate_tmastg_ready[dgate_index.idx].wait(dgate_index.phase)
                if pend_writes:
                    desc_dgate_slot = (desc_dgate_base + slot).tospace(cutlass.AddressSpace.generic)
                    dgate_slice = tma_slice_runtime_desc(desc_dgate_slot, cutlass.Int32(0), head_o, pend_start)
                    tma_store_tile(sDgate_tma[dgate_index.idx], dgate_slice, acquire=False)
                    tma_store_commit()
                bars.mb_dv_tmastg_ready[dv_index.idx].wait(dv_index.phase)
                if pend_writes:
                    desc_dv_slot = (desc_dv_base + slot).tospace(cutlass.AddressSpace.generic)
                    dv_slice = tma_slice_runtime_desc(desc_dv_slot, cutlass.Int32(0), head_o, pend_start)
                    tma_store_tile(sDv_tma[dv_index.idx], dv_slice, acquire=False)
                    tma_store_commit()
                tma_store_wait(3)
                bars.mb_dq_tmastg_done[dq_index.idx].arrive()
                tma_store_wait(2)
                bars.mb_dk_tmastg_done[dk_index.idx].arrive()
                tma_store_wait(1)
                bars.mb_dgate_tmastg_done[dgate_index.idx].arrive()
                tma_store_wait(0)
                bars.mb_dv_tmastg_done[dv_index.idx].arrive()
                dq_index = advance(dq_index, cfg.smem_dq_stages)
                dk_index = advance(dk_index, cfg.smem_dk_stages)
                dgate_index = advance(dgate_index, cfg.smem_dgate_stages)
                dv_index = advance(dv_index, cfg.smem_dv_stages)
            pend_start = chunk_start
            pend_writes = writes

        # ---- tile tail: drain the last chunk's dQ/dK/dGate/dV --------------------
        if num_compute_chunks > 0:
            bars.mb_dq_tmastg_ready[dq_index.idx].wait(dq_index.phase)
            if pend_writes:
                desc_dq_slot = (desc_dq_base + slot).tospace(cutlass.AddressSpace.generic)
                dq_slice = tma_slice_runtime_desc(desc_dq_slot, cutlass.Int32(0), head_o, pend_start)
                tma_store_tile(sDq_tma[dq_index.idx], dq_slice, acquire=False)
                tma_store_commit()
            bars.mb_dk_tmastg_ready[dk_index.idx].wait(dk_index.phase)
            if pend_writes:
                desc_dk_slot = (desc_dk_base + slot).tospace(cutlass.AddressSpace.generic)
                dk_slice = tma_slice_runtime_desc(desc_dk_slot, cutlass.Int32(0), head_o, pend_start)
                tma_store_tile(sDk_tma[dk_index.idx], dk_slice, acquire=False)
                tma_store_commit()
            bars.mb_dgate_tmastg_ready[dgate_index.idx].wait(dgate_index.phase)
            if pend_writes:
                desc_dgate_slot = (desc_dgate_base + slot).tospace(cutlass.AddressSpace.generic)
                dgate_slice = tma_slice_runtime_desc(desc_dgate_slot, cutlass.Int32(0), head_o, pend_start)
                tma_store_tile(sDgate_tma[dgate_index.idx], dgate_slice, acquire=False)
                tma_store_commit()
            bars.mb_dv_tmastg_ready[dv_index.idx].wait(dv_index.phase)
            if pend_writes:
                desc_dv_slot = (desc_dv_base + slot).tospace(cutlass.AddressSpace.generic)
                dv_slice = tma_slice_runtime_desc(desc_dv_slot, cutlass.Int32(0), head_o, pend_start)
                tma_store_tile(sDv_tma[dv_index.idx], dv_slice, acquire=False)
                tma_store_commit()
            tma_store_wait(3)
            bars.mb_dq_tmastg_done[dq_index.idx].arrive()
            tma_store_wait(2)
            bars.mb_dk_tmastg_done[dk_index.idx].arrive()
            tma_store_wait(1)
            bars.mb_dgate_tmastg_done[dgate_index.idx].arrive()
            tma_store_wait(0)
            bars.mb_dv_tmastg_done[dv_index.idx].arrive()
            dq_index = advance(dq_index, cfg.smem_dq_stages)
            dk_index = advance(dk_index, cfg.smem_dk_stages)
            dgate_index = advance(dgate_index, cfg.smem_dgate_stages)
            dv_index = advance(dv_index, cfg.smem_dv_stages)

        chunk_serial_base += num_compute_chunks
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


@cute.jit
def super_mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sSched,
    lane,
    sK_decay_raw,
    sK_inv_raw,
    sU_raw,
    sDy_raw,
    sIntermediate_raw,
    sBeta_raw,
    sBetaM_raw,
    bars,
) -> None:
    """Super-MMA warp role (warp 12): the Neumann T_inv and dM register MMAs
    plus the dBeta M-term row sums, in chunk order."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    sdy_index = PipelineState.start(phase=0)

    # ---- ldmatrix lane decode ----------------------------------------------------
    rhs_row_coord = lane % 8 + (cutlass.Int32(8) if (lane // 16) else cutlass.Int32(0))
    rhs_col_offset = cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0)
    lhs_row_coord = lane % 8 + (cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0))
    lhs_col_offset = cutlass.Int32(8) if ((lane // 8) // 2) else cutlass.Int32(0)
    stsm_row_coord = lane & 7
    stsm_col_coord = cutlass.Int32(0)
    if (lane // 8) & 1:
        stsm_row_coord = stsm_row_coord + cutlass.Int32(8)
    if lane // 8 >= 2:
        stsm_col_coord = cutlass.Int32(8)
    stsm_idx = swizzle_lin_S(stsm_row_coord * cfg.b_t + stsm_col_coord, bbits=1, mbase=3, sshift=3)
    row_lo = lane // 4
    row_hi = row_lo + cutlass.Int32(8)

    # hoisted tril bitmasks: bit i = row > col / row == col for accum index i
    tril_strict_mask = cutlass.Int32(0)
    eye_mask = cutlass.Int32(0)
    for accum_idx in cutlass.range_constexpr(8):
        row_coord = row_hi if cutlass.const_expr(accum_idx % 4 >= 2) else row_lo
        col_coord = (accum_idx // 4) * 8 + 2 * (lane % 4)
        if cutlass.const_expr(accum_idx % 2 == 1):
            col_coord = col_coord + cutlass.Int32(1)
        tril_strict_mask = tril_strict_mask | (cutlass.Int32(1 << accum_idx) if row_coord > col_coord else cutlass.Int32(0))
        eye_mask = eye_mask | (cutlass.Int32(1 << accum_idx) if row_coord == col_coord else cutlass.Int32(0))

    chunk_serial_base = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        num_compute_chunks = cend - wstart
        for rev_idx in cutlass.range(num_compute_chunks, unroll=1):
            chunk_serial = chunk_serial_base + rev_idx
            decay_stage = chunk_serial % cfg.smem_decay_stages
            intermediate_stage = chunk_serial % cfg.smem_intermediate_stages
            sBeta_ptr = smem_data_ptr(sBeta_raw) + (chunk_serial % cfg.smem_beta_stages) * cfg.b_t
            sK_inv_ptr = smem_data_ptr(sK_inv_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = smem_data_ptr(sK_decay_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sIntermediate_ptr = smem_data_ptr(sIntermediate_raw) + intermediate_stage * (cfg.intermediate_tiles * cfg.b_t * cfg.b_t)

            bars.mb_t_inv_done[intermediate_stage].wait(((chunk_serial // cfg.smem_intermediate_stages) + 1) % 2)

            # ---- KK = K_decay @ K_inv^T ------------------------------------------
            bars.mb_k_decay_inv_ready[decay_stage].wait((chunk_serial // cfg.smem_decay_stages) % 2)
            kk_lhs_row = lhs_row_coord
            kk_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                kk_acc[accum_idx] = cutlass.Float32(0.0)
            for k_block in cutlass.range_constexpr(cfg.d_k // 16):
                a_col = k_block * 16 + lhs_col_offset
                a_seg = a_col // 64
                a_frag = nvvm.ldmatrix(
                    sK_decay_ptr + a_seg * (cfg.b_t * 64) + kk_lhs_row * 64 + swizzle_xor_128b(kk_lhs_row, a_col - a_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                b_col = k_block * 16 + rhs_col_offset
                b_seg = b_col // 64
                b_frag = nvvm.ldmatrix(
                    sK_inv_ptr + b_seg * (cfg.b_t * 64) + rhs_row_coord * 64 + swizzle_xor_128b(rhs_row_coord, b_col - b_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                mma_step(
                    kk_acc,
                    (a_frag[0], a_frag[1], a_frag[2], a_frag[3]),
                    (b_frag[0], b_frag[1], b_frag[2], b_frag[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )

            # ---- L = Beta * tril(KK, -1) -----------------------------------------
            bars.mb_beta_ready[chunk_serial % cfg.smem_beta_stages].wait((chunk_serial // cfg.smem_beta_stages) % 2)
            beta_lo = (sBeta_ptr + row_lo).load().to(cutlass.Float32)
            beta_hi = (sBeta_ptr + row_hi).load().to(cutlass.Float32)
            bars.mb_beta_done[chunk_serial % cfg.smem_beta_stages].arrive()
            l_regs = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                beta_scale = beta_lo if accum_idx % 4 < 2 else beta_hi
                lower = kk_acc[accum_idx] if (tril_strict_mask >> accum_idx) & 1 else cutlass.Float32(0.0)
                l_regs[accum_idx] = lower * beta_scale
            l_a0 = fp32_to_fp16(l_regs[0], l_regs[1], dtype=cfg.io_dtype)
            l_a1 = fp32_to_fp16(l_regs[2], l_regs[3], dtype=cfg.io_dtype)
            l_a2 = fp32_to_fp16(l_regs[4], l_regs[5], dtype=cfg.io_dtype)
            l_a3 = fp32_to_fp16(l_regs[6], l_regs[7], dtype=cfg.io_dtype)
            l_values = cutlass.Vector.from_elements((l_a0, l_a1, l_a2, l_a3), cutlass.Int32).bitcast(cfg.io_dtype).to(cutlass.Float32)

            tinv_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                eye = cutlass.Float32(1.0) if (eye_mask >> accum_idx) & 1 else cutlass.Float32(0.0)
                tinv_acc[accum_idx] = eye - l_values[accum_idx]

            lpow_a0, lpow_a1, lpow_a2, lpow_a3 = l_a0, l_a1, l_a2, l_a3
            mov_lpow0, mov_lpow1, mov_lpow2, mov_lpow3 = movmatrix_16b(l_a0), movmatrix_16b(l_a1), movmatrix_16b(l_a2), movmatrix_16b(l_a3)
            for _round in cutlass.range_constexpr(3):
                sq_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
                for accum_idx in cutlass.range_constexpr(8):
                    sq_acc[accum_idx] = cutlass.Float32(0.0)
                mma_step(
                    sq_acc,
                    (lpow_a0, lpow_a1, lpow_a2, lpow_a3),
                    (mov_lpow0, mov_lpow1, mov_lpow2, mov_lpow3),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )
                lpow_a0 = fp32_to_fp16(sq_acc[0], sq_acc[1], dtype=cfg.io_dtype)
                lpow_a1 = fp32_to_fp16(sq_acc[2], sq_acc[3], dtype=cfg.io_dtype)
                lpow_a2 = fp32_to_fp16(sq_acc[4], sq_acc[5], dtype=cfg.io_dtype)
                lpow_a3 = fp32_to_fp16(sq_acc[6], sq_acc[7], dtype=cfg.io_dtype)
                mov_lpow0, mov_lpow1, mov_lpow2, mov_lpow3 = movmatrix_16b(lpow_a0), movmatrix_16b(lpow_a1), movmatrix_16b(lpow_a2), movmatrix_16b(lpow_a3)
                upd_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
                for accum_idx in cutlass.range_constexpr(8):
                    upd_acc[accum_idx] = cutlass.Float32(0.0)
                tinv_p0 = fp32_to_fp16(tinv_acc[0], tinv_acc[1], dtype=cfg.io_dtype)
                tinv_p1 = fp32_to_fp16(tinv_acc[2], tinv_acc[3], dtype=cfg.io_dtype)
                tinv_p2 = fp32_to_fp16(tinv_acc[4], tinv_acc[5], dtype=cfg.io_dtype)
                tinv_p3 = fp32_to_fp16(tinv_acc[6], tinv_acc[7], dtype=cfg.io_dtype)
                mma_step(
                    upd_acc,
                    (tinv_p0, tinv_p1, tinv_p2, tinv_p3),
                    (mov_lpow0, mov_lpow1, mov_lpow2, mov_lpow3),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )
                tinv_lo0, tinv_hi0 = f16x2_to_f32(tinv_p0, dtype=cfg.io_dtype)
                tinv_lo1, tinv_hi1 = f16x2_to_f32(tinv_p1, dtype=cfg.io_dtype)
                tinv_lo2, tinv_hi2 = f16x2_to_f32(tinv_p2, dtype=cfg.io_dtype)
                tinv_lo3, tinv_hi3 = f16x2_to_f32(tinv_p3, dtype=cfg.io_dtype)
                tinv_acc[0], tinv_acc[1] = fadd2(tinv_lo0, tinv_hi0, upd_acc[0], upd_acc[1])
                tinv_acc[2], tinv_acc[3] = fadd2(tinv_lo1, tinv_hi1, upd_acc[2], upd_acc[3])
                tinv_acc[4], tinv_acc[5] = fadd2(tinv_lo2, tinv_hi2, upd_acc[4], upd_acc[5])
                tinv_acc[6], tinv_acc[7] = fadd2(tinv_lo3, tinv_hi3, upd_acc[6], upd_acc[7])

            nvvm.stmatrix(
                sIntermediate_ptr + 1 * (cfg.b_t * cfg.b_t) + stsm_idx,
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

            # ---- dM = dY @ U^T ---------------------------------------------------
            bars.mb_dm_done[intermediate_stage].wait(((chunk_serial // cfg.smem_intermediate_stages) + 1) % 2)
            bars.mb_dy_smem_ready.wait(sdy_index.phase)
            sdy_index = advance(sdy_index, 1)
            dm_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                dm_acc[accum_idx] = cutlass.Float32(0.0)
            for k_block in cutlass.range_constexpr(cfg.d_v // 16):
                a_col = k_block * 16 + lhs_col_offset
                a_seg = a_col // 64
                a_frag = nvvm.ldmatrix(
                    smem_data_ptr(sDy_raw) + a_seg * (cfg.b_t * 64) + lhs_row_coord * 64 + swizzle_xor_128b(lhs_row_coord, a_col - a_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                b_col = k_block * 16 + rhs_col_offset
                b_seg = b_col // 64
                b_frag = nvvm.ldmatrix(
                    smem_data_ptr(sU_raw) + b_seg * (cfg.b_t * 64) + rhs_row_coord * 64 + swizzle_xor_128b(rhs_row_coord, b_col - b_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                mma_step(
                    dm_acc,
                    (a_frag[0], a_frag[1], a_frag[2], a_frag[3]),
                    (b_frag[0], b_frag[1], b_frag[2], b_frag[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )
            # ---- dM_strict = Beta_row . strict(dM) -------------------------------
            dm_strict_regs = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                beta_scale = beta_lo if accum_idx % 4 < 2 else beta_hi
                val = dm_acc[accum_idx] * beta_scale if (tril_strict_mask >> accum_idx) & 1 else cutlass.Float32(0.0)
                dm_strict_regs[accum_idx] = val
            w0 = fp32_to_fp16(dm_strict_regs[0], dm_strict_regs[1], dtype=cfg.io_dtype)
            w1 = fp32_to_fp16(dm_strict_regs[2], dm_strict_regs[3], dtype=cfg.io_dtype)
            w2 = fp32_to_fp16(dm_strict_regs[4], dm_strict_regs[5], dtype=cfg.io_dtype)
            w3 = fp32_to_fp16(dm_strict_regs[6], dm_strict_regs[7], dtype=cfg.io_dtype)
            nvvm.stmatrix(sIntermediate_ptr + 3 * (cfg.b_t * cfg.b_t) + stsm_idx, [w0, w1, w2, w3], nvvm.MMALayout.ROW, shape=nvvm.StoreShape.M8N8)
            nw0 = fp32_to_fp16(-dm_strict_regs[0], -dm_strict_regs[1], dtype=cfg.io_dtype)
            nw1 = fp32_to_fp16(-dm_strict_regs[2], -dm_strict_regs[3], dtype=cfg.io_dtype)
            nw2 = fp32_to_fp16(-dm_strict_regs[4], -dm_strict_regs[5], dtype=cfg.io_dtype)
            nw3 = fp32_to_fp16(-dm_strict_regs[6], -dm_strict_regs[7], dtype=cfg.io_dtype)
            nvvm.stmatrix(sIntermediate_ptr + 4 * (cfg.b_t * cfg.b_t) + stsm_idx, [nw0, nw1, nw2, nw3], nvvm.MMALayout.ROW, shape=nvvm.StoreShape.M8N8)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dm_ready[intermediate_stage].arrive()

            # ---- M-term: bsum = sum strict(dM . KK) ------------------------------
            bsum_lo = cutlass.Float32(0.0)
            bsum_hi = cutlass.Float32(0.0)
            for accum_idx in cutlass.range_constexpr(8):
                e = dm_acc[accum_idx] * kk_acc[accum_idx] if (tril_strict_mask >> accum_idx) & 1 else cutlass.Float32(0.0)
                if cutlass.const_expr(accum_idx % 4 < 2):
                    bsum_lo = bsum_lo + e
                else:
                    bsum_hi = bsum_hi + e
            bsum_lo = bsum_lo + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, bsum_lo, 1, 31, kind=nvvm.Shfl.BFLY))
            bsum_lo = bsum_lo + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, bsum_lo, 2, 31, kind=nvvm.Shfl.BFLY))
            bsum_hi = bsum_hi + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, bsum_hi, 1, 31, kind=nvvm.Shfl.BFLY))
            bsum_hi = bsum_hi + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, bsum_hi, 2, 31, kind=nvvm.Shfl.BFLY))
            if lane % 4 == 0:
                sBetaM_raw[row_lo] = -bsum_lo
                sBetaM_raw[row_hi] = -bsum_hi
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dbeta_m_ready.arrive()
        chunk_serial_base += num_compute_chunks
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


@cute.jit
def tcgen05_mma_warp(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sSched,
    tmem_base_holder,
    sState_alt,
    sState_direct,
    sK_decay_lead16,
    sK_inv_lead16,
    sK_inv_amaj,
    sK_restore_lead16,
    sDo_lead16,
    sDo_amaj,
    sQ_decay_trans,
    sK_decay_trans,
    sU_lead16,
    sDv_lead16,
    sDstate_alt,
    sIntermediate,
    sState_scale_diag,
    bars,
) -> None:
    """tcgen05-MMA warp role (warp 13): issues every tcgen05 GEMM and owns
    the TMEM lifecycle."""
    elect_one = nvvm.elect_sync()

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    nvvm.tcgen05_alloc(tmem_base_holder, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_holder.load()
    bpe = cfg.io_dtype.width // 8

    # ---- chunk-invariant GEMM descriptors ----------------------------------------
    idesc_mv_nt = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    idesc_state_k_at = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        a_major=1,
    )
    bmm_state_k_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=True,
        cta_group=1,
        idesc=idesc_state_k_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_state_k_kmaj = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_state_k_kmaj_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=idesc_state_k_kmaj,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_dvinter_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_mv_nt,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_du_at = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        a_major=1,
        b_major=1,
    )
    bmm_du_at_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=True,
        cta_group=1,
        idesc=idesc_du_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dstate_q_at = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_k,
        m_dim=cfg.d_v,
        a_major=1,
        b_major=1,
    )
    bmm_dstate_q_at_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.d_k,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=True,
        cta_group=1,
        idesc=idesc_dstate_q_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_qk_ts_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_mv_nt,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_mv_nt_t = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        b_major=1,
    )
    bmm_qk_ts_t_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        cta_group=1,
        idesc=idesc_mv_nt_t,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_diag = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=16,
        m_dim=cfg.d_v,
    )
    bmm_diag_desc = MmaDesc(
        M=cfg.d_v,
        N=16,
        K=16,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_diag,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dstate_k = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_k,
        m_dim=cfg.d_v,
        b_major=1,
    )
    bmm_dstate_k_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.d_k,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        cta_group=1,
        idesc=idesc_dstate_k,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dstate = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
    )
    bmm_dstate_ts_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.d_v,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_dstate,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dstate_at = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
    )
    bmm_dstate_at_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.d_v,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=True,
        cta_group=1,
        idesc=idesc_dstate_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dgp = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
    )
    bmm_dgrad_ts_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_dgp,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dgp_at = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
    )
    bmm_dgrad_at_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        atranspose=True,
        cta_group=1,
        idesc=idesc_dgp_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    idesc_dgp_at_t = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
        b_major=1,
    )
    bmm_dgrad_at_t_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        atranspose=True,
        cta_group=1,
        idesc=idesc_dgp_at_t,
        kind=nvvm.Tcgen05MMAKind.F16,
    )

    state_index = PipelineState.start(phase=0)
    y_inp_index = PipelineState.start(phase=0)
    dstate_inp_index = PipelineState.start(phase=0)
    du_inp_index = PipelineState.start(phase=0)
    neg_beta_dy_index = PipelineState.start(phase=0)
    u_smem_index = PipelineState.start(phase=0)
    dstate_smem_index = PipelineState.start(phase=0)
    parts_done_index = PipelineState.start(phase=1)

    do_seg = (cfg.b_t * cfg.d_v * (cfg.io_dtype.width // 8)) >> 4
    op_seg = (cfg.b_t * cfg.d_k * (cfg.io_dtype.width // 8)) >> 4
    intermediate_seg = (cfg.intermediate_tiles * cfg.b_t * cfg.b_t * (cfg.io_dtype.width // 8)) >> 4
    intermediate_slot = (cfg.b_t * cfg.b_t * (cfg.io_dtype.width // 8)) >> 4
    diag_seg = ((cfg.d_k // 16) * 256 * (cfg.io_dtype.width // 8)) >> 4
    dv_seg = (cfg.b_t * cfg.d_v * (cfg.io_dtype.width // 8)) >> 4
    d_do_amaj0 = sDo_amaj[0].desc()
    d_qd_trans0 = sQ_decay_trans[0].desc()
    d_kd_trans0 = sK_decay_trans[0].desc()
    d_ki_amaj0 = sK_inv_amaj[0].desc()
    d_int0 = sIntermediate[0].desc()
    d_kd_lead0 = sK_decay_lead16[0].desc()
    d_do_lead0 = sDo_lead16[0].desc()
    d_kr_lead0 = sK_restore_lead16[0].desc()
    d_diag0 = sState_scale_diag[0].desc()
    d_dv_lead0 = sDv_lead16[0].desc()
    d_dstate_alt0 = sDstate_alt[0].desc()
    d_u_lead0 = sU_lead16[0].desc()
    assert cfg.smem_state_stages == 1
    d_state_alt0 = sState_alt[0].desc()
    d_state_direct0 = sState_direct[0].desc()
    dstate0_index = PipelineState.start(phase=0)

    chunk_serial_base = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        num_compute_chunks = cend - wstart
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

            # ---- stage-derived operand descriptors -------------------------------
            decay_op_off = decay_stage * op_seg
            d_do_amaj = d_do_amaj0 + raw_stage_idx * do_seg
            d_qd_trans = d_qd_trans0 + decay_op_off
            d_kd_trans = d_kd_trans0 + decay_op_off
            d_ki_amaj = d_ki_amaj0 + decay_op_off
            d_int = d_int0 + intermediate_stage * intermediate_seg
            d_int_tinv = d_int + intermediate_slot
            d_int_da = d_int + 2 * intermediate_slot
            d_int_dm = d_int + 3 * intermediate_slot
            d_int_ndm = d_int + 4 * intermediate_slot
            chunk_idx = cend - cutlass.Int32(1) - rev_idx

            # ---- state_k = state(S) @ K_decay^T --------------------------------------
            bars.mb_k_decay_inv_ready[decay_stage].wait(decay_phase)
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_state_ready[state_index.idx].wait(state_index.phase)
                mma_ss(
                    bmm_state_k_kmaj_desc,
                    d_state_direct0,
                    d_kd_lead0 + decay_op_off,
                    nvvm.make_tmem_ptr((tmem_base + cfg.tmem_state_k_acc_offset), cutlass.Float32),
                    accumulate=False,
                )
                if elect_one:
                    bars.mb_state_k_acc_ready.arrive(cta_group=1)
                    bars.mb_state_done[state_index.idx].arrive(cta_group=1)
                state_index = advance(state_index, cfg.smem_state_stages)

            # ---- dQ inter = state(T) @ dO^T ------------------------------------------
            bars.mb_dqk_acc_done.wait(parts_done_index.phase)
            parts_done_index = advance(parts_done_index, 1)
            bars.mb_state_inp_ready[chunk_serial % 2].wait((chunk_serial // 2) % 2)
            bars.mb_do_ready[raw_stage_idx].wait((chunk_serial // cfg.smem_raw_stages) % 2)
            if chunk_idx >= FIRST_STATE_CHUNK:
                a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_state_inp_offset + (chunk_serial % 2) * (cfg.d_v // 2)), cutlass.Int8)
                b_desc = d_do_lead0 + raw_stage_idx * do_seg
                c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dq_acc_offset), cutlass.Float32)
                for sub in cutlass.range_constexpr(bmm_dstate_ts_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_dstate_ts_desc.sps_B):
                        mma_ts_step(
                            bmm_dstate_ts_desc,
                            a_ptr.subview(sub * bmm_dstate_ts_desc.sps_B * bmm_dstate_ts_desc.tmem_advance_A),
                            b_desc + sub * (bmm_dstate_ts_desc.smem_subtile_B >> 4),
                            c_ptr,
                            k,
                            cutlass.Boolean(sub + k > 0),
                        )

            # ---- dU inter = dstate(T) @ K_restore ------------------------------------
            bars.mb_q_decay_k_restore_ready[decay_stage].wait(decay_phase)
            if has_dstate:
                bars.mb_dstate_inp_ready.wait(dstate_inp_index.phase)
                a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dstate_inp_offset), cutlass.Int8)
                b_desc = d_kr_lead0 + decay_op_off
                c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_du_acc_offset), cutlass.Float32)
                for sub in cutlass.range_constexpr(bmm_dvinter_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_dvinter_desc.sps_B):
                        mma_ts_step(
                            bmm_dvinter_desc,
                            a_ptr.subview(sub * bmm_dvinter_desc.sps_B * bmm_dvinter_desc.tmem_advance_A),
                            b_desc + sub * (bmm_dvinter_desc.smem_subtile_B >> 4),
                            c_ptr,
                            k,
                            cutlass.Boolean(sub + k > 0),
                        )

            # ---- dstate decay = dstate(T) @ diag(eGl) ------------------------------------
            if has_dstate:
                desc_diag = d_diag0 + decay_stage * diag_seg
                for k_block in cutlass.range_constexpr(cfg.d_k // 16):
                    a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dstate_inp_offset) + k_block * 8, cutlass.Int8)
                    b_desc = desc_diag.advance_start_address(k_block * 256 * 2)
                    c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dstate_acc_offset) + k_block * 16, cutlass.Float32)
                    for sub in cutlass.range_constexpr(bmm_diag_desc.num_subtiles_B):
                        for k in cutlass.range_constexpr(bmm_diag_desc.sps_B):
                            mma_ts_step(
                                bmm_diag_desc,
                                a_ptr.subview(sub * bmm_diag_desc.sps_B * bmm_diag_desc.tmem_advance_A),
                                b_desc + sub * (bmm_diag_desc.smem_subtile_B >> 4),
                                c_ptr,
                                k,
                                cutlass.Boolean(sub + k > 0),
                            )
                dstate_inp_index = advance(dstate_inp_index, 1)

            # ---- dU intra += dO^T(S) @ A -----------------------------------------
            bars.mb_a_ready[intermediate_stage].wait(intermediate_phase)
            mma_ss(
                bmm_du_at_desc,
                d_do_amaj,
                d_int,
                nvvm.make_tmem_ptr((tmem_base + cfg.tmem_du_acc_offset), cutlass.Float32),
                accumulate=has_dstate,
            )
            if elect_one:
                bars.mb_du_acc_ready.arrive(cta_group=1)
                bars.mb_a_done[intermediate_stage].arrive(cta_group=1)

            # ---- dstate q-term += dO^T(S) @ Q_decay ----------------------------------
            mma_ss(
                bmm_dstate_q_at_desc,
                d_do_amaj,
                d_qd_trans,
                nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dstate_acc_offset), cutlass.Float32),
                accumulate=has_dstate,
            )

            # ---- U = Y(T) @ T_inv ------------------------------------------------
            bars.mb_t_inv_ready[intermediate_stage].wait(intermediate_phase)
            bars.mb_y_inp_ready.wait(y_inp_index.phase)
            y_inp_index = advance(y_inp_index, 1)
            a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_y_inp_offset), cutlass.Int8)
            b_desc = d_int_tinv
            c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_u_acc_offset), cutlass.Float32)
            for sub in cutlass.range_constexpr(bmm_qk_ts_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_qk_ts_desc.sps_B):
                    mma_ts_step(
                        bmm_qk_ts_desc,
                        a_ptr.subview(sub * bmm_qk_ts_desc.sps_B * bmm_qk_ts_desc.tmem_advance_A),
                        b_desc + sub * (bmm_qk_ts_desc.smem_subtile_B >> 4),
                        c_ptr,
                        k,
                        cutlass.Boolean(sub + k > 0),
                    )
            if elect_one:
                bars.mb_u_acc_ready.arrive(cta_group=1)

            # ---- dY = dU(T) @ T_inv ----------------------------------------------
            bars.mb_du_inp_ready.wait(du_inp_index.phase)
            du_inp_index = advance(du_inp_index, 1)
            a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_du_inp_offset), cutlass.Int8)
            dy_b_desc = d_int_tinv
            c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dy_acc_offset), cutlass.Float32)
            for sub in cutlass.range_constexpr(bmm_qk_ts_t_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_qk_ts_t_desc.sps_B):
                    mma_ts_step(
                        bmm_qk_ts_t_desc,
                        a_ptr.subview(sub * bmm_qk_ts_t_desc.sps_B * bmm_qk_ts_t_desc.tmem_advance_A),
                        dy_b_desc + sub * (bmm_qk_ts_t_desc.smem_subtile_B >> 4),
                        c_ptr,
                        k,
                        cutlass.Boolean(sub + k > 0),
                    )
            if elect_one:
                bars.mb_dy_acc_ready.arrive(cta_group=1)
                bars.mb_t_inv_done[intermediate_stage].arrive(cta_group=1)

            # ---- dK_restore part = dstate(S) @ U^T -----------------------------------
            bars.mb_u_smem_ready.wait(u_smem_index.phase)
            u_smem_index = advance(u_smem_index, 1)
            if has_dstate:
                bars.mb_dstate_smem_ready.wait(dstate_smem_index.phase)
                dstate_smem_index = advance(dstate_smem_index, 1)
                mma_ss(
                    bmm_dstate_at_desc,
                    d_dstate_alt0,
                    d_u_lead0,
                    nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dk_restore_acc_offset), cutlass.Float32),
                    accumulate=False,
                )
                if elect_one:
                    bars.mb_dk_restore_part_acc_ready.arrive(cta_group=1)
                    bars.mb_dstate_smem_done.arrive(cta_group=1)

            # ---- dstate dY-term += -Beta.dY(T) @ K_decay -----------------------------
            bars.mb_neg_beta_dy_inp_ready.wait(neg_beta_dy_index.phase)
            neg_beta_dy_index = advance(neg_beta_dy_index, 1)
            a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_neg_beta_dy_inp_offset), cutlass.Int8)
            b_desc = d_kd_trans
            c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dstate_acc_offset), cutlass.Float32)
            for sub in cutlass.range_constexpr(bmm_dstate_k_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_dstate_k_desc.sps_B):
                    mma_ts_step(
                        bmm_dstate_k_desc,
                        a_ptr.subview(sub * bmm_dstate_k_desc.sps_B * bmm_dstate_k_desc.tmem_advance_A),
                        b_desc + sub * (bmm_dstate_k_desc.smem_subtile_B >> 4),
                        c_ptr,
                        k,
                        cutlass.Boolean(True),
                    )
            if elect_one:
                bars.mb_dstate_acc_ready.arrive(cta_group=1)

            # ---- dK_inv part = scale.Q_decay^T(S) @ dA ---------------------------
            bars.mb_da_ready[intermediate_stage].wait(intermediate_phase)
            mma_ss(
                bmm_dgrad_at_t_desc,
                d_qd_trans,
                d_int_da,
                nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dk_inv_acc_offset), cutlass.Float32),
                accumulate=False,
            )

            # ---- dQ attn += K_inv^T(S) @ dA^T ------------------------------------
            if chunk_idx >= FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_dgrad_at_desc,
                    d_ki_amaj,
                    d_int_da,
                    nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dq_acc_offset), cutlass.Float32),
                    accumulate=True,
                )
            if chunk_idx < FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_dgrad_at_desc,
                    d_ki_amaj,
                    d_int_da,
                    nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dq_acc_offset), cutlass.Float32),
                    accumulate=False,
                )
            if elect_one:
                bars.mb_dq_acc_ready.arrive(cta_group=1)
                bars.mb_da_done[intermediate_stage].arrive(cta_group=1)

            # ---- dK_decay part = state(T) @ (Beta.dY)^T ------------------------------
            bars.mb_dv_tmastg_ready[chunk_serial % cfg.smem_dv_stages].wait((chunk_serial // cfg.smem_dv_stages) % 2)
            if chunk_idx >= FIRST_STATE_CHUNK:
                a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_state_inp_offset + (chunk_serial % 2) * (cfg.d_v // 2)), cutlass.Int8)
                b_desc = d_dv_lead0 + (chunk_serial % cfg.smem_dv_stages) * dv_seg
                c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dk_decay_acc_offset), cutlass.Float32)
                for sub in cutlass.range_constexpr(bmm_dstate_ts_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_dstate_ts_desc.sps_B):
                        mma_ts_step(
                            bmm_dstate_ts_desc,
                            a_ptr.subview(sub * bmm_dstate_ts_desc.sps_B * bmm_dstate_ts_desc.tmem_advance_A),
                            b_desc + sub * (bmm_dstate_ts_desc.smem_subtile_B >> 4),
                            c_ptr,
                            k,
                            cutlass.Boolean(sub + k > 0),
                        )
            if elect_one:
                bars.mb_state_inp_done[chunk_serial % 2].arrive(cta_group=1)

            # ---- dK_inv part += K_decay^T(S) @ -dM_strict ------------------------
            bars.mb_dm_ready[intermediate_stage].wait(intermediate_phase)
            mma_ss(
                bmm_dgrad_at_t_desc,
                d_kd_trans,
                d_int_ndm,
                nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dk_inv_acc_offset), cutlass.Float32),
                accumulate=True,
            )
            if elect_one:
                bars.mb_dk_inv_part_acc_ready.arrive(cta_group=1)

            # ---- dK_decay part += K_inv^T(S) @ dM_strict^T -----------------------
            if chunk_idx >= FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_dgrad_at_desc,
                    d_ki_amaj,
                    d_int_dm,
                    nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dk_decay_acc_offset), cutlass.Float32),
                    accumulate=True,
                )
            if chunk_idx < FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_dgrad_at_desc,
                    d_ki_amaj,
                    d_int_dm,
                    nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dk_decay_acc_offset), cutlass.Float32),
                    accumulate=False,
                )
            if elect_one:
                bars.mb_dk_decay_part_acc_ready.arrive(cta_group=1)
                bars.mb_dm_done[intermediate_stage].arrive(cta_group=1)
                bars.mb_decay_done[decay_stage].arrive(cta_group=1)

        # ---- tile end: WG1's dstate0 drain gates the next tile's dstate reuse ------------
        bars.mb_dstate0_acc_stored.wait(dstate0_index.phase)
        dstate0_index = advance(dstate0_index, 1)
        chunk_serial_base += num_compute_chunks
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
    mSched,
    sSched,
    lane,
    sQ_raw,
    sK_raw,
    sV_raw,
    sGate_raw,
    sDo_raw,
    sState_raw,
    desc_q_base,
    desc_k_base,
    desc_v_base,
    desc_gate_base,
    desc_do_base,
    desc_checkpoint_base,
    bars,
) -> None:
    """TMA-LDG warp role (warp 14): persistent tile-scheduler loop issuing
    every G->S TMA load."""
    elect_one = nvvm.elect_sync()

    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

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
    sGate_tma = SmemTile(
        base=sGate_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_raw_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 32),
        tma_granu_elems=32,
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
    sState_tma = SmemTile(
        base=sState_raw,
        elems_per_stage=(cfg.d_k * cfg.d_v),
        stages=cfg.smem_state_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_v // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=cfg.d_k * 64,
    )
    raw_index = PipelineState.start(phase=1)
    state_index = PipelineState.start(phase=1)
    sched_state = PipelineState.start(phase=1)
    packed_tokens = cutlass.Int32(cu_seqlens[n_desc])
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        next_tile, sched_state = sched_publish_next(cfg, bars, sSched, mSched, sched_state, tile_idx, num_ctas)
        head_o = head_idx
        head_q = head_idx if cfg.q_ratio == 1 else head_idx // cutlass.Int32(cfg.q_ratio)
        head_k = head_idx if cfg.k_ratio == 1 else head_idx // cutlass.Int32(cfg.k_ratio)
        head_v = head_idx if cfg.v_ratio == 1 else head_idx // cutlass.Int32(cfg.v_ratio)
        slot = batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
        desc_q_slot = (desc_q_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_gate_slot = (desc_gate_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_do_slot = (desc_do_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_checkpoint_slot = (desc_checkpoint_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_v_slot)
            tma_tensormap_acquire(desc_gate_slot)
            tma_tensormap_acquire(desc_do_slot)
            tma_tensormap_acquire(desc_checkpoint_slot)
        use_packed_coords = batch_end == packed_tokens
        num_compute_chunks = cend - wstart
        for rev_idx in cutlass.range(num_compute_chunks, unroll=1):
            chunk_idx = cend - cutlass.Int32(1) - rev_idx
            chunk_start = chunk_idx * cfg.b_t
            input_chunk_start = (
                batch_start + chunk_start if use_packed_coords else chunk_start
            )

            # ---- Q load ----------------------------------------------------------
            bars.mb_q_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_q_ready[raw_index.idx].arrive(n_bytes=cfg.tma_q_bytes)
            q_slice = tma_slice_runtime_desc(
                desc_q_slot, cutlass.Int32(0), head_q, input_chunk_start
            )
            tma_load_tile(sQ_tma[raw_index.idx], q_slice, bars.mb_q_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- K load ----------------------------------------------------------
            bars.mb_k_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_k_ready[raw_index.idx].arrive(n_bytes=cfg.tma_k_bytes)
            k_slice = tma_slice_runtime_desc(
                desc_k_slot, cutlass.Int32(0), head_k, input_chunk_start
            )
            tma_load_tile(sK_tma[raw_index.idx], k_slice, bars.mb_k_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- Gate load -------------------------------------------------------
            bars.mb_gate_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_gate_ready[raw_index.idx].arrive(n_bytes=cfg.tma_gate_bytes)
            gate_slice = tma_slice_runtime_desc(
                desc_gate_slot, cutlass.Int32(0), head_o, input_chunk_start
            )
            tma_load_tile(sGate_tma[raw_index.idx], gate_slice, bars.mb_gate_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- dO load ---------------------------------------------------------
            bars.mb_do_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_do_ready[raw_index.idx].arrive(n_bytes=cfg.tma_do_bytes)
            do_slice = tma_slice_runtime_desc(
                desc_do_slot, cutlass.Int32(0), head_o, input_chunk_start
            )
            tma_load_tile(sDo_tma[raw_index.idx], do_slice, bars.mb_do_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- V load ----------------------------------------------------------
            bars.mb_v_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_v_ready[raw_index.idx].arrive(n_bytes=cfg.tma_v_bytes)
            v_slice = tma_slice_runtime_desc(
                desc_v_slot, cutlass.Int32(0), head_v, input_chunk_start
            )
            tma_load_tile(sV_tma[raw_index.idx], v_slice, bars.mb_v_ready[raw_index.idx].smem_ptr, acquire=False)

            # ---- entering state ----
            if chunk_idx >= FIRST_STATE_CHUNK:
                state_idx = state_index.idx
                bars.mb_state_cg0_done[state_idx].wait(state_index.phase)
                bars.mb_state_done[state_idx].wait(state_index.phase)
                state_index = advance(state_index, cfg.smem_state_stages)
                if elect_one:
                    bars.mb_state_ready[state_idx].arrive(n_bytes=cfg.tma_state_bytes)
                state_slice = tma_slice_runtime_desc(desc_checkpoint_slot, cutlass.Int32(0), cutlass.Int32(0), chunk_idx, head_o)
                tma_load_tile(sState_tma[state_idx], state_slice, bars.mb_state_ready[state_idx].smem_ptr, acquire=False)
            raw_index = advance(raw_index, cfg.smem_raw_stages)
        tile_idx = next_tile


@cute.jit
def gate_scale(cfg, raw_gate: cutlass.Float32) -> cutlass.Float32:
    """Map raw gate to the log2-domain decay increment used by KDA."""

    if cutlass.const_expr(cfg.safe_gate):
        half = cutlass.Float32(0.5)
        sigmoid = cute.math.tanh(raw_gate * half, approx=True) * half + half
        return cfg.gate_scale_log2 * sigmoid
    # Default ABI: Gate arrives in natural-log space
    return raw_gate * cutlass.Float32(LOG2_E)


@cute.jit
def compute0_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sSched,
    lane,
    tmem_base_holder,
    warp_idx,
    scale,
    mA_log,
    mDt_bias,
    mBeta,
    sBeta_raw,
    sK_inv_raw,
    sGate_raw,
    sK_raw,
    sQ_raw,
    sState_raw,
    sNorm_raw,
    sK_decay_raw,
    sK_restore_raw,
    sQ_decay_raw,
    sState_scale_diag_raw,
    bars,
) -> None:
    """WG0 warp role (warps 0-3): persistent tile-scheduler loop + gate prefix
    scan and the decay/restore operand materialization into tcgen05 SMEM for
    EVERY chunk (no ping-pong: the backward pipeline is drain-bound).  Also
    stashes the per-row Q/K inverse norms for WG2's dGate assembly and copies
    H -> TMEM f16 at the chunk tail."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_0, nvvm.SetMaxRegisterAction.INCREASE)
    cg0_warp = warp_idx - cfg.compute_group_0_warp_ids[0]
    prefix_dim = cg0_warp * cfg.threads_per_warp + lane
    cg0_a_log_exp = cutlass.Float32(1.0)
    cg0_dt_bias_value = cutlass.Float32(0.0)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_holder.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    tmem_subpartition = warp_idx % (cfg.d_v // cfg.threads_per_warp)
    value_dim = tmem_subpartition * cfg.threads_per_warp + lane
    row_addr = (tmem_row + tmem_subpartition * cfg.threads_per_warp) << 16
    state_index = PipelineState.start(phase=0)
    chunk_serial_base = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        num_compute_chunks = cend - wstart
        if cutlass.const_expr(cfg.safe_gate):
            if num_compute_chunks > 0:
                cg0_a_log_exp = cute.math.exp2(mA_log[head_idx].to(cutlass.Float32) * LOG2_E, fastmath=True)
                cg0_dt_bias_value = mDt_bias[head_idx, prefix_dim].to(cutlass.Float32)
        for rev_idx in cutlass.range(num_compute_chunks, unroll=1):
            chunk_idx = cend - cutlass.Int32(1) - rev_idx
            chunk_serial = chunk_serial_base + rev_idx
            chunk_start = chunk_idx * cfg.b_t
            decay_stage = chunk_serial % cfg.smem_decay_stages
            raw_stage = chunk_serial % cfg.smem_raw_stages
            sQ_ptr = smem_data_ptr(sQ_raw) + raw_stage * (cfg.d_k * cfg.b_t)
            sK_ptr = smem_data_ptr(sK_raw) + raw_stage * (cfg.d_k * cfg.b_t)
            sGate_ptr = smem_data_ptr(sGate_raw) + raw_stage * (cfg.d_k * cfg.b_t)
            sK_inv_ptr = smem_data_ptr(sK_inv_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = smem_data_ptr(sK_decay_raw) + decay_stage * (cfg.d_k * cfg.b_t)
            sQ_decay_ptr = smem_data_ptr(sQ_decay_raw) + decay_stage * (cfg.d_k * cfg.b_t)
            sK_restore_ptr = smem_data_ptr(sK_restore_raw) + decay_stage * (cfg.d_k * cfg.b_t)
            sState_scale_diag_ptr = smem_data_ptr(sState_scale_diag_raw) + decay_stage * ((cfg.d_k // 16) * 256)

            # ---- beta scalars: gathered in the inputs-wait shadow ----------------
            if cg0_warp == 0:
                beta_stage = chunk_serial % cfg.smem_beta_stages
                bars.mb_beta_done[beta_stage].wait(((chunk_serial // cfg.smem_beta_stages) + 1) % 2)
                if lane < cfg.b_t:
                    token_idx = chunk_start + lane
                    beta_value = cutlass.Float32(0.0)
                    if token_idx < seqlen_b:
                        beta_value = mBeta[batch_start + token_idx, head_idx].to(cutlass.Float32)
                        if cutlass.const_expr(cfg.beta_sigmoid):
                            half = cutlass.Float32(0.5)
                            beta_value = (cute.math.tanh(beta_value * half, approx=True) * half + half).to(mBeta.element_type).to(cutlass.Float32)
                    sBeta_raw[beta_stage * cfg.b_t + lane] = beta_value
                bars.mb_beta_ready[beta_stage].arrive()

            bars.mb_gate_ready[raw_stage].wait((chunk_serial // cfg.smem_raw_stages) % 2)
            bars.mb_q_ready[raw_stage].wait((chunk_serial // cfg.smem_raw_stages) % 2)
            bars.mb_k_ready[raw_stage].wait((chunk_serial // cfg.smem_raw_stages) % 2)

            row_group_start = cg0_warp * (cfg.b_t // len(cfg.compute_group_0_warp_ids))
            lane_row_group = lane // 8
            lane_in_row_group = lane - lane_row_group * 8
            decay_row = row_group_start + lane_row_group

            g_prefix_ptr = sGate_ptr
            prefix_dim = cg0_warp * cfg.threads_per_warp + lane
            # ---- gate prefix scan: cumulative log-gate per key channel -----------
            gate_raw = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
            for row in cutlass.range_constexpr(cfg.b_t):
                f32_segment = prefix_dim // 32
                f32_segment_dim = prefix_dim - f32_segment * 32
                prefix_idx = f32_segment * (cfg.b_t * 32) + row * 32 + swizzle_xor_128b(row, f32_segment_dim, elem_bytes=4)
                gate_raw[row] = (sGate_ptr + prefix_idx).load()
            g_prefix_regs = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
            if cutlass.const_expr(cfg.safe_gate):
                valid_rows = seqlen_b - chunk_idx * cutlass.Int32(cfg.b_t)
                valid_mask = cutlass.vector.create_mask([cfg.b_t], [valid_rows])
                for row_pair in cutlass.range_constexpr(cfg.b_t // 2):
                    row0 = row_pair * 2
                    row1 = row0 + 1
                    gate0 = cg0_a_log_exp * (gate_raw[row0] + cg0_dt_bias_value)
                    gate1 = cg0_a_log_exp * (gate_raw[row1] + cg0_dt_bias_value)
                    gate0 = gate_scale(
                        cfg,
                        gate0,
                    )
                    gate1 = gate_scale(
                        cfg,
                        gate1,
                    )
                    gate_pair = cutlass.Vector.from_elements((gate0, gate1), cutlass.Float32)
                    gate_pair = cutlass.vector.where(valid_mask[row0 : row1 + 1], gate_pair, 0.0)
                    g_prefix_regs[row0] = gate_pair[0]
                    g_prefix_regs[row1] = gate_pair[1]
            else:
                for row in cutlass.range_constexpr(cfg.b_t):
                    gate = gate_raw[row]
                    token_idx = chunk_idx * cutlass.Int32(cfg.b_t) + cutlass.Int32(row)
                    if token_idx < seqlen_b:
                        gate = gate_scale(
                            cfg,
                            gate,
                        )
                    else:
                        gate = cutlass.Float32(0.0)
                    g_prefix_regs[row] = gate

            prefix_acc = cutlass.Float32(0.0)
            for row_pair in cutlass.range_constexpr(cfg.b_t // 2):
                row0 = row_pair * 2
                row1 = row0 + 1
                gate0 = g_prefix_regs[row0]
                gate1 = g_prefix_regs[row1]
                pair_vec = nvvm.add_packed_f32x2(
                    cutlass.Vector.from_elements((prefix_acc, gate0), cutlass.Float32),
                    cutlass.Vector.from_elements((gate0, gate1), cutlass.Float32),
                    ftz=False,
                    rnd="rn",
                )
                prefix0, row_pair_sum = cutlass.Float32(pair_vec[0]), cutlass.Float32(pair_vec[1])
                prefix1 = prefix_acc + row_pair_sum
                g_prefix_regs[row0] = prefix0
                g_prefix_regs[row1] = prefix1
                prefix_acc = prefix1

            for row in cutlass.range_constexpr(cfg.b_t):
                g_prefix_regs[row] = cute.math.exp2(g_prefix_regs[row], fastmath=True)

            exp_g_last = g_prefix_regs[cfg.b_t - 1]
            # ---- decay-slot guard: previous use fully consumed -------------------
            operand_done_phase = ((chunk_serial // cfg.smem_decay_stages) + 1) % 2
            bars.mb_decay_done[decay_stage].wait(operand_done_phase)

            for row in cutlass.range_constexpr(cfg.b_t):
                f32_segment = prefix_dim // 32
                f32_segment_dim = prefix_dim - f32_segment * 32
                prefix_idx = f32_segment * (cfg.b_t * 32) + row * 32 + swizzle_xor_128b(row, f32_segment_dim, elem_bytes=4)
                (sGate_ptr + prefix_idx).store(g_prefix_regs[row])

            # ---- state-scale diag: stage exp2(g_last) decay blocks ---------------
            block = prefix_dim // cutlass.Int32(16)
            coord = prefix_dim - block * cutlass.Int32(16)
            linear_idx = block * cutlass.Int32(256) + coord * cutlass.Int32(16) + coord
            diag_idx = swizzle_lin_S(linear_idx, bbits=1, mbase=3, sshift=3)
            sState_scale_diag_ptr[diag_idx] = exp_g_last.to(cfg.io_dtype)

            # ---- raw Q/K: SMEM -> TMEM ring (channel-major, for WG2) -------------
            qk_raw_stage = chunk_serial % cfg.tmem_qk_raw_stages
            bars.mb_qk_raw_done[qk_raw_stage].wait(((chunk_serial // cfg.tmem_qk_raw_stages) + 1) % 2)
            raw_seg = prefix_dim // 64
            raw_dim = prefix_dim - raw_seg * 64
            q_raw_words = cute.make_rmem_tensor((cfg.b_t // 2,), cutlass.Int32)
            k_raw_words = cute.make_rmem_tensor((cfg.b_t // 2,), cutlass.Int32)
            for t2 in cutlass.range_constexpr(cfg.b_t // 2):
                t0 = 2 * t2
                ridx0 = raw_seg * (cfg.b_t * 64) + t0 * 64 + swizzle_xor_128b(t0, raw_dim, elem_bytes=2)
                ridx1 = raw_seg * (cfg.b_t * 64) + (t0 + 1) * 64 + swizzle_xor_128b(t0 + 1, raw_dim, elem_bytes=2)
                q0 = (sQ_ptr + ridx0).load().to(cutlass.Float32)
                q1 = (sQ_ptr + ridx1).load().to(cutlass.Float32)
                k0 = (sK_ptr + ridx0).load().to(cutlass.Float32)
                k1 = (sK_ptr + ridx1).load().to(cutlass.Float32)
                q_raw_words[t2] = fp32_to_fp16(q0, q1, dtype=cfg.io_dtype)
                k_raw_words[t2] = fp32_to_fp16(k0, k1, dtype=cfg.io_dtype)
            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_qraw_inp_offset + qk_raw_stage * (cfg.b_t // 2)), cutlass.Int8),
                q_raw_words.load(),
            )
            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_kraw_inp_offset + qk_raw_stage * (cfg.b_t // 2)), cutlass.Int8),
                k_raw_words.load(),
            )
            nvvm.tcgen05_wait("store")
            bars.mb_qk_raw_ready[qk_raw_stage].arrive()

            nvvm.barrier_cta_sync(cfg.cg0_sync_barrier_id, thread_count=cfg.cg0_threads)

            k_inv_pack = cute.make_rmem_tensor((2 * 4,), cutlass.Int32)
            raw_q_regs = cute.make_rmem_tensor((2 * 8,), cutlass.Float32)
            raw_k_regs = cute.make_rmem_tensor((2 * 8,), cutlass.Float32)
            # ---- optional Q/K L2-norm --------------------------------------------
            if cutlass.const_expr(cfg.l2norm):
                qk0_lo = opaque_f32_zero()
                qk0_hi = opaque_f32_zero()
                qk1_lo = opaque_f32_zero()
                qk1_hi = opaque_f32_zero()
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                raw_f16_idx = f16_segment * (cfg.b_t * 64) + decay_row * 64 + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
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
            bars.mb_q_done[raw_stage].arrive()
            bars.mb_k_done[raw_stage].arrive()

            q_inv_norm = opaque_f32_zero() + cutlass.Float32(1.0)
            k_inv_norm = opaque_f32_zero() + cutlass.Float32(1.0)
            if cutlass.const_expr(cfg.l2norm):
                q_sum_sq = qk0_lo + qk1_lo
                k_sum_sq = qk0_hi + qk1_hi
                q_sum_sq = q_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, q_sum_sq, 4, 31, kind=nvvm.Shfl.BFLY))
                q_sum_sq = q_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, q_sum_sq, 2, 31, kind=nvvm.Shfl.BFLY))
                q_sum_sq = q_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, q_sum_sq, 1, 31, kind=nvvm.Shfl.BFLY))
                k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 4, 31, kind=nvvm.Shfl.BFLY))
                k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 2, 31, kind=nvvm.Shfl.BFLY))
                k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 1, 31, kind=nvvm.Shfl.BFLY))
                norm_floor_sq = cutlass.Float32(L2_NORM_EPS * L2_NORM_EPS)
                q_inv_norm = cute.math.rsqrt(cute.math.max(q_sum_sq, norm_floor_sq), fastmath=True)
                k_inv_norm = cute.math.rsqrt(cute.math.max(k_sum_sq, norm_floor_sq), fastmath=True)
                if lane_in_row_group == 0:
                    sNorm_raw[(chunk_serial % cfg.tmem_qk_raw_stages) * (2 * cfg.b_t) + decay_row] = q_inv_norm
                    sNorm_raw[(chunk_serial % cfg.tmem_qk_raw_stages) * (2 * cfg.b_t) + cfg.b_t + decay_row] = k_inv_norm
            q_stage_norm = q_inv_norm * scale

            # ---- decay/restore operands: exp2(+-g) applied per key channel -------
            exp_g_regs = cute.make_rmem_tensor((2 * 8,), cutlass.Float32)
            exp_g_last_regs = cute.make_rmem_tensor((2 * 8,), cutlass.Float32)
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                for f32_group in cutlass.range_constexpr(2):
                    f32_dim_base = dim_base + f32_group * 4
                    f32_segment = f32_dim_base // 32
                    f32_segment_dim = f32_dim_base - f32_segment * 32
                    g_prefix_idx = f32_segment * (cfg.b_t * 32) + decay_row * 32 + swizzle_xor_128b(decay_row, f32_segment_dim, elem_bytes=4)
                    exp_g_frag = (g_prefix_ptr + g_prefix_idx).load(count=4, alignment=16)
                    exp_g_last_idx = f32_segment * (cfg.b_t * 32) + (cfg.b_t - 1) * 32 + swizzle_xor_128b((cfg.b_t - 1), f32_segment_dim, elem_bytes=4)
                    exp_g_last_frag = (g_prefix_ptr + exp_g_last_idx).load(count=4, alignment=16)
                    f32_reg_base = reg_base + f32_group * 4
                    exp_g_regs[f32_reg_base] = exp_g_frag[0]
                    exp_g_regs[f32_reg_base + 1] = exp_g_frag[1]
                    exp_g_regs[f32_reg_base + 2] = exp_g_frag[2]
                    exp_g_regs[f32_reg_base + 3] = exp_g_frag[3]
                    exp_g_last_regs[f32_reg_base] = exp_g_last_frag[0]
                    exp_g_last_regs[f32_reg_base + 1] = exp_g_last_frag[1]
                    exp_g_last_regs[f32_reg_base + 2] = exp_g_last_frag[2]
                    exp_g_last_regs[f32_reg_base + 3] = exp_g_last_frag[3]
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_gate_done[raw_stage].arrive()

            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                # ---- K decay + K inv operands: exp2(+g) * K / exp2(-g) * K -------
                k_decay_pack = cute.make_rmem_tensor((4,), cutlass.Int32)
                for pair_idx in cutlass.range_constexpr(4):
                    dim0 = pair_idx * 2
                    dim1 = dim0 + 1
                    raw_reg_idx0 = reg_base + dim0
                    raw_reg_idx1 = reg_base + dim1
                    k_value0, k_value1 = fmul2(raw_k_regs[raw_reg_idx0], raw_k_regs[raw_reg_idx1], k_inv_norm, k_inv_norm)
                    k_pair = fp32_to_fp16(k_value0, k_value1, dtype=cfg.io_dtype)
                    exp_g_pair = fp32_to_fp16(exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1], dtype=cfg.io_dtype)
                    k_decay_pack[pair_idx] = mul_f16x2(k_pair, exp_g_pair, cfg.io_dtype)
                    exp_neg_g0 = cute.math.rcp(exp_g_regs[raw_reg_idx0], approx=True, ftz=True)
                    exp_neg_g1 = cute.math.rcp(exp_g_regs[raw_reg_idx1], approx=True, ftz=True)
                    exp_neg_pair = fp32_to_fp16(exp_neg_g0, exp_neg_g1, dtype=cfg.io_dtype)
                    k_inv_pack[dim_half * 4 + pair_idx] = mul_f16x2(k_pair, exp_neg_pair, cfg.io_dtype)

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
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                op_idx = f16_segment * (cfg.b_t * 64) + decay_row * 64 + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                (sK_inv_ptr + op_idx).store(k_inv_vec, alignment=16)
                (sK_decay_ptr + op_idx).store(k_decay_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_k_decay_inv_ready[decay_stage].arrive()

            # ---- Q decay + K_restore operands ------------------------------------
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                q_decay_pack = cute.make_rmem_tensor((4,), cutlass.Int32)
                k_restore_pack = cute.make_rmem_tensor((4,), cutlass.Int32)
                for pair_idx in cutlass.range_constexpr(4):
                    dim0 = pair_idx * 2
                    dim1 = dim0 + 1
                    raw_reg_idx0 = reg_base + dim0
                    raw_reg_idx1 = reg_base + dim1
                    q_value0, q_value1 = fmul2(raw_q_regs[raw_reg_idx0], raw_q_regs[raw_reg_idx1], q_stage_norm, q_stage_norm)
                    q_pair = fp32_to_fp16(q_value0, q_value1, dtype=cfg.io_dtype)
                    exp_g_pair = fp32_to_fp16(exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1], dtype=cfg.io_dtype)
                    q_decay_pack[pair_idx] = mul_f16x2(q_pair, exp_g_pair, cfg.io_dtype)
                    exp_g_last_pair = fp32_to_fp16(exp_g_last_regs[raw_reg_idx0], exp_g_last_regs[raw_reg_idx1], dtype=cfg.io_dtype)
                    k_restore_pack[pair_idx] = mul_f16x2(k_inv_pack[dim_half * 4 + pair_idx], exp_g_last_pair, cfg.io_dtype)

                q_decay_vec = cutlass.Vector.from_elements(
                    (q_decay_pack[0], q_decay_pack[1], q_decay_pack[2], q_decay_pack[3]),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                k_restore_vec = cutlass.Vector.from_elements(
                    (k_restore_pack[0], k_restore_pack[1], k_restore_pack[2], k_restore_pack[3]),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                op_idx = f16_segment * (cfg.b_t * 64) + decay_row * 64 + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                (sQ_decay_ptr + op_idx).store(q_decay_vec, alignment=16)
                (sK_restore_ptr + op_idx).store(k_restore_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_q_decay_k_restore_ready[decay_stage].arrive()

            # ---- state copy: SMEM -> TMEM f16 ----------------------------------------
            bars.mb_state_inp_done[chunk_serial % 2].wait(((chunk_serial // 2) + 1) % 2)
            bars.mb_state_inp_cg2_done[chunk_serial % 2].wait(((chunk_serial // 2) + 1) % 2)
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_state_ready[state_index.idx].wait(state_index.phase)
                state_src = smem_data_ptr(sState_raw) + state_index.idx * (cfg.d_k * cfg.d_v)
                for v_seg in cutlass.range_constexpr(2):
                    for v_col8 in cutlass.range_constexpr(8):
                        state_frag = cutlass.Vector.from_elements(
                            tuple(
                                (
                                    state_src
                                    + (value_dim // 64) * (cfg.d_v * 64)
                                    + (v_seg * 64 + v_col8 * 8 + e) * 64
                                    + swizzle_xor_128b(v_seg * 64 + v_col8 * 8 + e, value_dim % 64, elem_bytes=2)
                                ).load()
                                for e in range(8)
                            ),
                            cfg.io_dtype,
                        )
                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                row_addr + (tmem_col + cfg.tmem_state_inp_offset + (chunk_serial % 2) * (cfg.d_v // 2) + v_seg * 32 + v_col8 * 4), cutlass.Int8
                            ),
                            state_frag.bitcast(cutlass.Int32),
                        )
                nvvm.tcgen05_wait("store")
                bars.mb_state_cg0_done[state_index.idx].arrive()
                state_index = advance(state_index, cfg.smem_state_stages)
            bars.mb_state_inp_ready[chunk_serial % 2].arrive()
        chunk_serial_base += num_compute_chunks
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


@cute.jit
def compute1_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sSched,
    lane,
    tmem_base_holder,
    warp_idx,
    mDbeta,
    mDstate0,
    mDstate_in,
    sV_raw,
    sDo_raw,
    sBeta_raw,
    sU_raw,
    sDy_raw,
    sDv_raw,
    sDstate_raw,
    sRed_raw,
    sBetaM_raw,
    bars,
) -> None:
    """WG1 warp role (warps 4-7): the value-side TMEM staging."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_1, nvvm.SetMaxRegisterAction.INCREASE)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_holder.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    tmem_subpartition = warp_idx % (cfg.d_v // cfg.threads_per_warp)
    token_row_coord = (lane // 16) * 8 + (lane & 7)
    value_col_offset = ((lane // 8) & 1) * 8
    value_dim = tmem_subpartition * cfg.threads_per_warp + lane
    value_dim_base = tmem_subpartition * cfg.threads_per_warp
    cg1_tidx = warp_idx % 4 * cfg.threads_per_warp + lane

    raw_index = PipelineState.start(phase=0)
    state_k_index = PipelineState.start(phase=0)
    u_acc_index = PipelineState.start(phase=0)
    du_acc_index = PipelineState.start(phase=0)
    dy_acc_index = PipelineState.start(phase=0)
    dstate_ready_index = PipelineState.start(phase=0)
    dstate_smem_done_index = PipelineState.start(phase=1)
    dbeta_m_index = PipelineState.start(phase=0)
    dv_done_index = PipelineState.start(phase=1)

    chunk_serial_base = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        num_compute_chunks = cend - wstart

        # ---- dht seeding: dh acc + dh_inp f16 + sdH ------------------------------
        if cutlass.const_expr(cfg.use_dstate_in):
            if num_compute_chunks > 0:
                seed_true = cend == num_chunks_b
                bars.mb_dstate_smem_done.wait(dstate_smem_done_index.phase)
                bars.mb_dstate_smem_cg2_done.wait(dstate_smem_done_index.phase)
                dstate_smem_done_index = advance(dstate_smem_done_index, 1)
                row_addr = (tmem_row + tmem_subpartition * cfg.threads_per_warp) << 16
                dstate_src = (mDstate_in.iterator + mDstate_in.layout((batch_idx, head_idx, value_dim, 0))).raw_ptr()
                for sub in cutlass.range_constexpr(cfg.d_k // 16):
                    seed_block = cute.make_rmem_tensor((16,), cutlass.Float32)
                    for g in cutlass.range_constexpr(4):
                        seed_chunk = (dstate_src + sub * 16 + g * 4).load(count=4, alignment=16)
                        for t in cutlass.range_constexpr(4):
                            dval = seed_chunk[t].to(cutlass.Float32)
                            seed_block[g * 4 + t] = dval if seed_true else cutlass.Float32(0.0)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dstate_acc_offset + sub * 16), cutlass.Float32),
                        seed_block.load(),
                    )
                    seed_pack = cute.make_rmem_tensor((8,), cutlass.Int32)
                    for pc in cutlass.range_constexpr(8):
                        seed_pack[pc] = fp32_to_fp16(seed_block[2 * pc], seed_block[2 * pc + 1], dtype=cfg.io_dtype)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dstate_inp_offset + sub * 8), cutlass.Int8),
                        seed_pack.load(),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_inp_ready.arrive()

                # ---- dht seed -> sdH: re-read dh_inp after the TMEM publish ------
                for sub in cutlass.range_constexpr(cfg.d_k // 16):
                    dstate_words = nvvm.tcgen05_ld(
                        "32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dstate_inp_offset + sub * 8), cutlass.Float32), num=8
                    )
                    for half in cutlass.range_constexpr(2):
                        d_base = sub * 16 + half * 8
                        h_vec = cutlass.Vector.from_elements(
                            (dstate_words[half * 4], dstate_words[half * 4 + 1], dstate_words[half * 4 + 2], dstate_words[half * 4 + 3]),
                            cutlass.Float32,
                        ).bitcast(cfg.io_dtype)
                        h_addr = (d_base // 64) * (cfg.d_v * 64) + value_dim * 64 + swizzle_xor_128b(value_dim, d_base % 64, elem_bytes=2)
                        (smem_data_ptr(sDstate_raw) + h_addr).store(h_vec, alignment=16)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dstate_smem_ready.arrive()

        for rev_idx in cutlass.range(num_compute_chunks, unroll=1):
            chunk_idx = cend - cutlass.Int32(1) - rev_idx
            chunk_serial = chunk_serial_base + rev_idx
            sV_ptr = smem_data_ptr(sV_raw) + raw_index.idx * (cfg.d_v * cfg.b_t)
            sDo_ptr = smem_data_ptr(sDo_raw) + raw_index.idx * (cfg.d_v * cfg.b_t)
            sBeta_ptr = smem_data_ptr(sBeta_raw) + (chunk_serial % cfg.smem_beta_stages) * cfg.b_t
            row_addr_lo = tmem_row << 16
            row_addr_hi = (tmem_row + 16) << 16
            row_id0 = tmem_row + value_dim_base
            row_id1 = row_id0 + 16
            has_dstate = cutlass.Boolean(rev_idx > 0)
            if cutlass.const_expr(cfg.use_dstate_in):
                has_dstate = cutlass.Boolean(True)

            # ---- Y staging: Y = Beta * (V - state_k) -----------------------------
            bars.mb_v_ready[raw_index.idx].wait((chunk_serial // cfg.smem_raw_stages) % 2)
            projection_col_id = tmem_col + cfg.tmem_state_k_acc_offset
            input_col_id = tmem_col + cfg.tmem_y_inp_offset
            raw_v_frag0 = nvvm.ldmatrix(
                sV_ptr
                + (value_dim_base + value_col_offset) // 64 * (cfg.b_t * 64)
                + token_row_coord * 64
                + swizzle_xor_128b(token_row_coord, (value_dim_base + value_col_offset) % 64, elem_bytes=2),
                4,
                nvvm.MMALayout.COL,
            )
            raw_v_frag1 = nvvm.ldmatrix(
                sV_ptr
                + (value_dim_base + 16 + value_col_offset) // 64 * (cfg.b_t * 64)
                + token_row_coord * 64
                + swizzle_xor_128b(token_row_coord, (value_dim_base + 16 + value_col_offset) % 64, elem_bytes=2),
                4,
                nvvm.MMALayout.COL,
            )
            # fence: the V ldmatrix reads must complete before this release
            # licenses the TMA reload (sV)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_v_done[raw_index.idx].arrive()

            bars.mb_beta_ready[chunk_serial % cfg.smem_beta_stages].wait((chunk_serial // cfg.smem_beta_stages) % 2)
            beta_pairs = cute.make_rmem_tensor((2,), cutlass.Int32)
            for half in cutlass.range_constexpr(2):
                token0 = ((half * 4 + (lane & 3)) ^ 4) * 2
                beta0 = (sBeta_ptr + token0).load().to(cutlass.Float32)
                beta1 = (sBeta_ptr + token0 + 1).load().to(cutlass.Float32)
                beta_pairs[half] = fp32_to_fp16(beta0, beta1, dtype=cfg.io_dtype)
            diff_w0 = cute.make_rmem_tensor((4,), cutlass.Int32)
            diff_w1 = cute.make_rmem_tensor((4,), cutlass.Int32)
            y_inp_pack0 = cute.make_rmem_tensor((4,), cutlass.Int32)
            y_inp_pack1 = cute.make_rmem_tensor((4,), cutlass.Int32)
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_state_k_acc_ready.wait(state_k_index.phase)
                state_k_index = advance(state_k_index, 1)
                state_k_vec0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id0 << 16) + projection_col_id, cutlass.Float32), num=2)
                state_k_vec1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id1 << 16) + projection_col_id, cutlass.Float32), num=2)
                for reg_idx in cutlass.range_constexpr(4):
                    raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                    frag_pair = (reg_idx ^ 2) * 2
                    state_k_pair = fp32_to_fp16(state_k_vec0[frag_pair], state_k_vec0[frag_pair + 1], dtype=cfg.io_dtype)
                    diff_pair = sub_f16x2(raw_v_frag0[raw_matrix], state_k_pair, cfg.io_dtype)
                    diff_w0[reg_idx ^ 2] = diff_pair
                    y_inp_pack0[reg_idx ^ 2] = mul_f16x2(beta_pairs[reg_idx // 2], diff_pair, cfg.io_dtype)
                for reg_idx in cutlass.range_constexpr(4):
                    raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                    frag_pair = (reg_idx ^ 2) * 2
                    state_k_pair = fp32_to_fp16(state_k_vec1[frag_pair], state_k_vec1[frag_pair + 1], dtype=cfg.io_dtype)
                    diff_pair = sub_f16x2(raw_v_frag1[raw_matrix], state_k_pair, cfg.io_dtype)
                    diff_w1[reg_idx ^ 2] = diff_pair
                    y_inp_pack1[reg_idx ^ 2] = mul_f16x2(beta_pairs[reg_idx // 2], diff_pair, cfg.io_dtype)
            if chunk_idx < FIRST_STATE_CHUNK:
                for reg_idx in cutlass.range_constexpr(4):
                    raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                    diff_w0[reg_idx ^ 2] = raw_v_frag0[raw_matrix]
                    y_inp_pack0[reg_idx ^ 2] = mul_f16x2(beta_pairs[reg_idx // 2], raw_v_frag0[raw_matrix], cfg.io_dtype)
                for reg_idx in cutlass.range_constexpr(4):
                    raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                    diff_w1[reg_idx ^ 2] = raw_v_frag1[raw_matrix]
                    y_inp_pack1[reg_idx ^ 2] = mul_f16x2(beta_pairs[reg_idx // 2], raw_v_frag1[raw_matrix], cfg.io_dtype)
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_addr_lo + input_col_id, cutlass.Int8), y_inp_pack0.load())
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_addr_hi + input_col_id, cutlass.Int8), y_inp_pack1.load())
            nvvm.tcgen05_wait("store")
            bars.mb_y_inp_ready.arrive()

            # ---- dU restage: dU acc -> TMEM f16 A operand ------------------------
            bars.mb_du_acc_ready.wait(du_acc_index.phase)
            du_acc_index = advance(du_acc_index, 1)
            du_col_id = tmem_col + cfg.tmem_du_acc_offset
            du_vec0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id0 << 16) + du_col_id, cutlass.Float32), num=2)
            du_vec1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id1 << 16) + du_col_id, cutlass.Float32), num=2)

            du_pack0 = cute.make_rmem_tensor((4,), cutlass.Int32)
            du_pack1 = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                frag_pair = reg_idx * 2
                du_pack0[reg_idx] = fp32_to_fp16(du_vec0[frag_pair], du_vec0[frag_pair + 1], dtype=cfg.io_dtype)
                du_pack1[reg_idx] = fp32_to_fp16(du_vec1[frag_pair], du_vec1[frag_pair + 1], dtype=cfg.io_dtype)
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_addr_lo + (tmem_col + cfg.tmem_du_inp_offset), cutlass.Int8), du_pack0.load())
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_addr_hi + (tmem_col + cfg.tmem_du_inp_offset), cutlass.Int8), du_pack1.load())
            nvvm.tcgen05_wait("store")
            bars.mb_du_inp_ready.arrive()

            # ---- U readback -> sU ------------------------------------------------
            bars.mb_u_acc_ready.wait(u_acc_index.phase)
            u_acc_index = advance(u_acc_index, 1)
            u_col_id = tmem_col + cfg.tmem_u_acc_offset
            u_vec0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id0 << 16) + u_col_id, cutlass.Float32), num=2)
            u_vec1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id1 << 16) + u_col_id, cutlass.Float32), num=2)

            u_pack0 = cute.make_rmem_tensor((4,), cutlass.Int32)
            u_pack1 = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                u_pack0[reg_idx] = fp32_to_fp16(u_vec0[2 * reg_idx], u_vec0[2 * reg_idx + 1], dtype=cfg.io_dtype)
                u_pack1[reg_idx] = fp32_to_fp16(u_vec1[2 * reg_idx], u_vec1[2 * reg_idx + 1], dtype=cfg.io_dtype)
            nvvm.stmatrix(
                smem_data_ptr(sU_raw)
                + (value_dim_base + value_col_offset) // 64 * (cfg.b_t * 64)
                + token_row_coord * 64
                + swizzle_xor_128b(token_row_coord, (value_dim_base + value_col_offset) % 64, elem_bytes=2),
                u_pack0.load(),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.stmatrix(
                smem_data_ptr(sU_raw)
                + (value_dim_base + 16 + value_col_offset) // 64 * (cfg.b_t * 64)
                + token_row_coord * 64
                + swizzle_xor_128b(token_row_coord, (value_dim_base + 16 + value_col_offset) % 64, elem_bytes=2),
                u_pack1.load(),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_u_smem_ready.arrive()

            # ---- dY readback -------------------------------------------------------
            bars.mb_dy_acc_ready.wait(dy_acc_index.phase)
            dy_acc_index = advance(dy_acc_index, 1)
            dy_col_id = tmem_col + cfg.tmem_dy_acc_offset
            dy_vec0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id0 << 16) + dy_col_id, cutlass.Float32), num=2)
            dy_vec1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr((row_id1 << 16) + dy_col_id, cutlass.Float32), num=2)

            # ---- dY -> sdY: pack + store + publish (super warp's dM operand) -------
            addr_lo0 = (
                (value_dim_base + value_col_offset) // 64 * (cfg.b_t * 64)
                + token_row_coord * 64
                + swizzle_xor_128b(token_row_coord, (value_dim_base + value_col_offset) % 64, elem_bytes=2)
            )
            addr_lo1 = (
                (value_dim_base + 16 + value_col_offset) // 64 * (cfg.b_t * 64)
                + token_row_coord * 64
                + swizzle_xor_128b(token_row_coord, (value_dim_base + 16 + value_col_offset) % 64, elem_bytes=2)
            )
            dy_pack0 = cute.make_rmem_tensor((4,), cutlass.Int32)
            dy_pack1 = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                dy_pack0[reg_idx] = fp32_to_fp16(dy_vec0[2 * reg_idx], dy_vec0[2 * reg_idx + 1], dtype=cfg.io_dtype)
                dy_pack1[reg_idx] = fp32_to_fp16(dy_vec1[2 * reg_idx], dy_vec1[2 * reg_idx + 1], dtype=cfg.io_dtype)
            nvvm.stmatrix(smem_data_ptr(sDy_raw) + addr_lo0, dy_pack0.load(), nvvm.MMALayout.COL, shape=nvvm.StoreShape.M8N8)
            nvvm.stmatrix(smem_data_ptr(sDy_raw) + addr_lo1, dy_pack1.load(), nvvm.MMALayout.COL, shape=nvvm.StoreShape.M8N8)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dy_smem_ready.arrive()

            # ---- beta scalars -> beta.dY -> sdV (epilogue TMA + dK_decay MMA operand) ---
            beta_c0 = (sBeta_ptr + (lane % 4) * 2).load().to(cutlass.Float32)
            beta_c1 = (sBeta_ptr + (lane % 4) * 2 + 1).load().to(cutlass.Float32)
            beta_c8 = (sBeta_ptr + (lane % 4) * 2 + 8).load().to(cutlass.Float32)
            beta_c9 = (sBeta_ptr + (lane % 4) * 2 + 9).load().to(cutlass.Float32)
            # latched before the stage is released; the dbeta store runs after the refill
            beta_self = cutlass.Float32(0.0)
            if cutlass.const_expr(cfg.beta_sigmoid):
                if cg1_tidx < cfg.b_t:
                    beta_self = (sBeta_ptr + cg1_tidx).load().to(cutlass.Float32)
            bars.mb_beta_done[chunk_serial % cfg.smem_beta_stages].arrive()
            beta_dy_regs0 = cute.make_rmem_tensor((8,), cutlass.Float32)
            beta_dy_regs1 = cute.make_rmem_tensor((8,), cutlass.Float32)
            for e2 in cutlass.range_constexpr(4):
                e = 2 * e2
                b_lo = beta_c8 if cutlass.const_expr(e >= 4) else beta_c0
                b_hi = beta_c9 if cutlass.const_expr(e >= 4) else beta_c1
                beta_dy_regs0[e], beta_dy_regs0[e + 1] = fmul2(dy_vec0[e], dy_vec0[e + 1], b_lo, b_hi)
                beta_dy_regs1[e], beta_dy_regs1[e + 1] = fmul2(dy_vec1[e], dy_vec1[e + 1], b_lo, b_hi)

            # ---- -beta.dY -> TMEM: A operand of the dH ds-term ---------------------
            neg_beta_dy_regs0 = cute.make_rmem_tensor((8,), cutlass.Float32)
            neg_beta_dy_regs1 = cute.make_rmem_tensor((8,), cutlass.Float32)
            for e in cutlass.range_constexpr(8):
                neg_beta_dy_regs0[e] = -beta_dy_regs0[e]
                neg_beta_dy_regs1[e] = -beta_dy_regs1[e]
            neg_beta_dy_pack0 = cute.make_rmem_tensor((4,), cutlass.Int32)
            neg_beta_dy_pack1 = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                frag_pair = reg_idx * 2
                neg_beta_dy_pack0[reg_idx] = fp32_to_fp16(neg_beta_dy_regs0[frag_pair], neg_beta_dy_regs0[frag_pair + 1], dtype=cfg.io_dtype)
                neg_beta_dy_pack1[reg_idx] = fp32_to_fp16(neg_beta_dy_regs1[frag_pair], neg_beta_dy_regs1[frag_pair + 1], dtype=cfg.io_dtype)
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_addr_lo + (tmem_col + cfg.tmem_neg_beta_dy_inp_offset), cutlass.Int8), neg_beta_dy_pack0.load())
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(row_addr_hi + (tmem_col + cfg.tmem_neg_beta_dy_inp_offset), cutlass.Int8), neg_beta_dy_pack1.load())
            nvvm.tcgen05_wait("store")
            bars.mb_neg_beta_dy_inp_ready.arrive()

            # ---- dBeta v-term parts: dY.(V - state_k), diff_w from Y staging -------
            tok4 = cute.make_rmem_tensor((4,), cutlass.Float32)
            for s in cutlass.range_constexpr(4):
                tok4[s] = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(4):
                d0_lo, d0_hi = f16x2_to_f32(diff_w0[j], dtype=cfg.io_dtype)
                d1_lo, d1_hi = f16x2_to_f32(diff_w1[j], dtype=cfg.io_dtype)
                s_lo = cutlass.const_expr(2 * (j // 2))
                s_hi = cutlass.const_expr(2 * (j // 2) + 1)
                t_lo, t_hi = ffma2(dy_vec0[2 * j], dy_vec0[2 * j + 1], d0_lo, d0_hi, tok4[s_lo], tok4[s_hi])
                t_lo, t_hi = ffma2(dy_vec1[2 * j], dy_vec1[2 * j + 1], d1_lo, d1_hi, t_lo, t_hi)
                tok4[s_lo] = t_lo
                tok4[s_hi] = t_hi

            # ---- beta.dY -> sdV (epilogue TMA + dK_decay MMA operand) ------------
            beta_dy_pack0 = cute.make_rmem_tensor((4,), cutlass.Int32)
            beta_dy_pack1 = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                beta_dy_pack0[reg_idx] = fp32_to_fp16(beta_dy_regs0[2 * reg_idx], beta_dy_regs0[2 * reg_idx + 1], dtype=cfg.io_dtype)
                beta_dy_pack1[reg_idx] = fp32_to_fp16(beta_dy_regs1[2 * reg_idx], beta_dy_regs1[2 * reg_idx + 1], dtype=cfg.io_dtype)
            dv_stage = chunk_serial % cfg.smem_dv_stages
            sdv_stage_base = dv_stage * (cfg.b_t * cfg.d_v)
            bars.mb_dv_tmastg_done[dv_stage].wait(dv_done_index.phase)
            dv_done_index = advance(dv_done_index, cfg.smem_dv_stages)
            nvvm.stmatrix(
                smem_data_ptr(sDv_raw) + sdv_stage_base + addr_lo0,
                beta_dy_pack0.load(),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.stmatrix(
                smem_data_ptr(sDv_raw) + sdv_stage_base + addr_lo1,
                beta_dy_pack1.load(),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dv_tmastg_ready[dv_stage].arrive()

            # ---- dBeta = sum_v dY.(V - state_k) + M-term -------------------------
            nvvm.barrier_cta_sync(cfg.cg1_sync_barrier_id, thread_count=cfg.cg1_threads)
            bars.mb_dbeta_m_ready.wait(dbeta_m_index.phase)
            dbeta_m_index = advance(dbeta_m_index, 1)
            for off in cutlass.range_constexpr(3):
                step = cutlass.const_expr(4 << off)
                for s in cutlass.range_constexpr(4):
                    tok4[s] = tok4[s] + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, tok4[s], step, 31, kind=nvvm.Shfl.BFLY))
            if lane < 4:
                for s in cutlass.range_constexpr(4):
                    sRed_raw[(warp_idx % 4) * cfg.b_t + (lane % 4) * 2 + (s % 2) + 8 * (s // 2)] = tok4[s]
            nvvm.barrier_cta_sync(cfg.cg1_sync_barrier_id, thread_count=cfg.cg1_threads)
            if cg1_tidx < cfg.b_t:
                acc = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(4):
                    acc = acc + sRed_raw[w * cfg.b_t + cg1_tidx]
                db_val = acc + sBetaM_raw[cg1_tidx]
                if cutlass.const_expr(cfg.beta_sigmoid):
                    db_val = db_val * (beta_self - beta_self * beta_self)
                token_idx = chunk_idx * cutlass.Int32(cfg.b_t) + cg1_tidx
                if token_idx < seqlen_b and chunk_idx < wend:
                    mDbeta[batch_start + token_idx, head_idx] = db_val.to(mDbeta.element_type)
            nvvm.barrier_cta_sync(cfg.cg1_sync_barrier_id, thread_count=cfg.cg1_threads)

            # ---- dH capture for the next -----------------------------------------
            bars.mb_dstate_acc_ready.wait(dstate_ready_index.phase)
            dstate_ready_index = advance(dstate_ready_index, 1)
            if rev_idx + cutlass.Int32(1) < num_compute_chunks:
                bars.mb_dstate_smem_done.wait(dstate_smem_done_index.phase)
                bars.mb_dstate_smem_cg2_done.wait(dstate_smem_done_index.phase)
                dstate_smem_done_index = advance(dstate_smem_done_index, 1)
                row_addr = (tmem_row + tmem_subpartition * cfg.threads_per_warp) << 16
                for sub in cutlass.range_constexpr(cfg.d_k // 32):
                    dstate_vec = nvvm.tcgen05_ld(
                        "32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dstate_acc_offset + sub * 32), cutlass.Float32), num=32
                    )
                    dstate_pack = cute.make_rmem_tensor((16,), cutlass.Int32)
                    for pc in cutlass.range_constexpr(16):
                        dstate_pack[pc] = fp32_to_fp16(dstate_vec[2 * pc], dstate_vec[2 * pc + 1], dtype=cfg.io_dtype)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dstate_inp_offset + sub * 16), cutlass.Int8),
                        dstate_pack.load(),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_dstate_inp_ready.arrive()

                # ---- dh_inp -> sdH: re-read after the TMEM publish ---------------
                for sub in cutlass.range_constexpr(cfg.d_k // 32):
                    dstate_words = nvvm.tcgen05_ld(
                        "32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dstate_inp_offset + sub * 16), cutlass.Float32), num=16
                    )
                    for half in cutlass.range_constexpr(4):
                        d_base = sub * 32 + half * 8
                        h_vec = cutlass.Vector.from_elements(
                            (dstate_words[half * 4], dstate_words[half * 4 + 1], dstate_words[half * 4 + 2], dstate_words[half * 4 + 3]),
                            cutlass.Float32,
                        ).bitcast(cfg.io_dtype)
                        h_addr = (d_base // 64) * (cfg.d_v * 64) + value_dim * 64 + swizzle_xor_128b(value_dim, d_base % 64, elem_bytes=2)
                        (smem_data_ptr(sDstate_raw) + h_addr).store(h_vec, alignment=16)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dstate_smem_ready.arrive()
            raw_index = advance(raw_index, cfg.smem_raw_stages)

        # ---- tile end: dS0 drain / zero-length pass-through ----------------------
        if cutlass.const_expr(mDstate0 is not None):
            if num_compute_chunks > 0:
                if wstart == 0:
                    row_addr = (tmem_row + tmem_subpartition * cfg.threads_per_warp) << 16
                    dstate0_dst = (mDstate0.iterator + mDstate0.layout((batch_idx, head_idx, value_dim, 0))).raw_ptr()
                    for sub in cutlass.range_constexpr(cfg.d_k // 32):
                        dstate0_vec = nvvm.tcgen05_ld(
                            "32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dstate_acc_offset + sub * 32), cutlass.Float32), num=32
                        )
                        for g in cutlass.range_constexpr(8):
                            (dstate0_dst + sub * 32 + g * 4).store(
                                cutlass.Vector.from_elements(tuple(dstate0_vec[g * 4 + t] for t in range(4)), cutlass.Float32),
                                alignment=16,
                            )
            else:
                for key_dim_base in cutlass.range_constexpr(0, cfg.d_k, 32):
                    for kk_i in cutlass.range_constexpr(32):
                        kd = key_dim_base + kk_i
                        if cutlass.const_expr(cfg.use_dstate_in):
                            mDstate0[batch_idx, head_idx, value_dim, kd] = mDstate_in[batch_idx, head_idx, value_dim, kd]
                        else:
                            mDstate0[batch_idx, head_idx, value_dim, kd] = cutlass.Float32(0.0)
        bars.mb_dstate0_acc_stored.arrive()
        chunk_serial_base += num_compute_chunks
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)

    bars.mb_tmem_done[0].arrive()


@cute.jit
def compute2_warp_group(
    cfg,
    total_tiles,
    bidx,
    num_ctas,
    cu_seqlens,
    mWorkItems,
    sSched,
    lane,
    tmem_base_holder,
    warp_idx,
    sGate_raw,
    sNorm_raw,
    sDq_raw,
    sDk_raw,
    sRed1_raw,
    sDstate_raw,
    sDgate_raw,
    scale,
    bars,
) -> None:
    """WG2 warp role (warps 8-11): the gradient drain.  Each thread owns one
    key channel d for all 16 tokens: reads the dQ accumulator and the four dK
    parts from TMEM, assembles dQ/dK with the per-channel gate factors (raw
    Q/K arrive through WG0's TMEM ring), applies the in-kernel L2-norm
    backward row projection, assembles the per-channel dGate including the
    g_last terms, reverse-cumsums it in registers, stages dGate for the
    epilogue's TMA store, and stages dQ/dK for the epilogue's TMA stores."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_2, nvvm.SetMaxRegisterAction.INCREASE)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_holder.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    tmem_subpartition = warp_idx % 4
    channel = tmem_subpartition * cfg.threads_per_warp + lane
    row_addr = (tmem_row + tmem_subpartition * cfg.threads_per_warp) << 16
    cg2_tidx = channel

    raw_index = PipelineState.start(phase=0)
    dq_acc_index = PipelineState.start(phase=0)
    dk_decay_part_index = PipelineState.start(phase=0)
    dk_inv_part_index = PipelineState.start(phase=0)
    dk_restore_part_index = PipelineState.start(phase=0)
    dgate_last_dstate_smem_index = PipelineState.start(phase=0)
    chunk_serial_base = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        num_compute_chunks = cend - wstart
        for rev_idx in cutlass.range(num_compute_chunks, unroll=1):
            chunk_idx = cend - cutlass.Int32(1) - rev_idx
            chunk_serial = chunk_serial_base + rev_idx
            chunk_start = chunk_idx * cfg.b_t
            raw_stage = chunk_serial % cfg.smem_raw_stages
            decay_stage = chunk_serial % cfg.smem_decay_stages
            has_dstate = cutlass.Boolean(rev_idx > 0)
            if cutlass.const_expr(cfg.use_dstate_in):
                has_dstate = cutlass.Boolean(True)
            sGate_ptr = smem_data_ptr(sGate_raw) + raw_stage * (cfg.d_k * cfg.b_t)
            writes = chunk_idx < wend

            # ---- gate landed: CG0 publishes the decay ring only after consuming
            # the gate TMA, so this wait is CG2's visibility guard for sGate ------
            bars.mb_k_decay_inv_ready[decay_stage].wait((chunk_serial // cfg.smem_decay_stages) % 2)

            # ---- per-channel gate factors ----------------------------------------
            f32_seg = channel // 32
            f32_dim = channel - f32_seg * 32
            f16_seg = channel // 64
            f16_dim = channel - f16_seg * 64
            eg = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
            for t in cutlass.range_constexpr(cfg.b_t):
                eg[t] = (sGate_ptr + f32_seg * (cfg.b_t * 32) + t * 32 + swizzle_xor_128b(t, f32_dim, elem_bytes=4)).load()
            egl = eg[cfg.b_t - 1]
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_gate_done[raw_stage].arrive()

            # ---- staged raw Q/K: TMEM ring cols for this chunk -------------------
            qk_raw_stage = chunk_serial % cfg.tmem_qk_raw_stages
            qraw_col = tmem_col + cfg.tmem_qraw_inp_offset + qk_raw_stage * (cfg.b_t // 2)
            kraw_col = tmem_col + cfg.tmem_kraw_inp_offset + qk_raw_stage * (cfg.b_t // 2)
            norm_base = qk_raw_stage * (2 * cfg.b_t)
            bars.mb_qk_raw_ready[qk_raw_stage].wait((chunk_serial // cfg.tmem_qk_raw_stages) % 2)

            # ---- dGate_last hdot: sum_v sdH[v, c] * S0[c, v] ---------------------
            dgate_last_val = cutlass.Float32(0.0)
            bars.mb_state_inp_ready[chunk_serial % 2].wait((chunk_serial // 2) % 2)
            if has_dstate:
                bars.mb_dstate_smem_ready.wait(dgate_last_dstate_smem_index.phase)
                dgate_last_dstate_smem_index = advance(dgate_last_dstate_smem_index, 1)
                for pl in cutlass.range_constexpr(2):
                    for row_half in cutlass.range_constexpr(2):
                        state_vec = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                row_addr + (tmem_col + cfg.tmem_state_inp_offset + (chunk_serial % 2) * (cfg.d_v // 2) + pl * 32 + row_half * 16),
                                cutlass.Float32,
                            ),
                            num=16,
                        )
                        hacc = cute.make_rmem_tensor((8,), cutlass.Float32)
                        for i in cutlass.range_constexpr(8):
                            hacc[i] = opaque_f32_zero()
                        for j in cutlass.range_constexpr(16):
                            v0 = pl * 64 + row_half * 32 + 2 * j
                            state_pair = cutlass.Vector.from_elements((state_vec[j],), cutlass.Float32).bitcast(cfg.io_dtype)
                            dstate_addr0 = (channel // 64) * (cfg.d_v * 64) + v0 * 64 + swizzle_xor_128b(v0, channel % 64, elem_bytes=2)
                            dstate_addr1 = (channel // 64) * (cfg.d_v * 64) + (v0 + 1) * 64 + swizzle_xor_128b(v0 + 1, channel % 64, elem_bytes=2)
                            hval0 = (smem_data_ptr(sDstate_raw) + dstate_addr0).load().to(cutlass.Float32)
                            hval1 = (smem_data_ptr(sDstate_raw) + dstate_addr1).load().to(cutlass.Float32)
                            hacc[(2 * j) % 8] = hacc[(2 * j) % 8] + hval0 * state_pair[0].to(cutlass.Float32)
                            hacc[(2 * j + 1) % 8] = hacc[(2 * j + 1) % 8] + hval1 * state_pair[1].to(cutlass.Float32)
                        pa0, pb0 = fadd2(hacc[0], hacc[2], hacc[4], hacc[6])
                        pa1, pb1 = fadd2(hacc[1], hacc[3], hacc[5], hacc[7])
                        part_a, part_b = fadd2(pa0, pb0, pa1, pb1)
                        dgate_last_val = dgate_last_val + (part_a + part_b)
                bars.mb_dstate_smem_cg2_done.arrive()
            bars.mb_state_inp_cg2_done[chunk_serial % 2].arrive()

            # ---- part-drain accumulators -------------------------------------------
            dq_n = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
            dk_n = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
            dgate_regs = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
            dgate_last_acc = cute.make_rmem_tensor((4,), cutlass.Float32)
            for i in cutlass.range_constexpr(4):
                dgate_last_acc[i] = opaque_f32_zero()
            for t in cutlass.range_constexpr(cfg.b_t):
                dk_n[t] = cutlass.Float32(0.0)

            # ---- dK_restore part drain: (eGl/eG) scale + dGate_last K-dot ----------
            if has_dstate:
                bars.mb_dk_restore_part_acc_ready.wait(dk_restore_part_index.phase)
                dk_restore_part_index = advance(dk_restore_part_index, 1)
                dk_restore_part_vec = nvvm.tcgen05_ld(
                    "32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dk_restore_acc_offset), cutlass.Float32), num=cfg.b_t
                )
                kr_words = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + kraw_col, cutlass.Float32), num=cfg.b_t // 2)
                for t in cutlass.range_constexpr(cfg.b_t):
                    dk_hat = egl * cute.math.rcp(eg[t], approx=True, ftz=True) * dk_restore_part_vec[t]
                    dk_n[t] = dk_hat
                    k_pair = cutlass.Vector.from_elements((kr_words[t // 2],), cutlass.Float32).bitcast(cfg.io_dtype)
                    k_v = k_pair[t % 2].to(cutlass.Float32)
                    if cutlass.const_expr(cfg.l2norm):
                        k_v = k_v * sNorm_raw[norm_base + cfg.b_t + t]
                    dgate_last_acc[t % 4] = dgate_last_acc[t % 4] + k_v * dk_hat

            # ---- dQ acc drain: eG.scale ---------------------------------------------
            bars.mb_dq_acc_ready.wait(dq_acc_index.phase)
            dq_acc_index = advance(dq_acc_index, 1)
            dq_vec = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dq_acc_offset), cutlass.Float32), num=cfg.b_t)
            for t2 in cutlass.range_constexpr(cfg.b_t // 2):
                t = 2 * t2
                es_lo, es_hi = fmul2(eg[t], eg[t + 1], scale, scale)
                dq_n[t], dq_n[t + 1] = fmul2(es_lo, es_hi, dq_vec[t], dq_vec[t + 1])

            # ---- dK_inv part drain: (dA - dM) term, 1/eG scale ----------------------
            bars.mb_dk_inv_part_acc_ready.wait(dk_inv_part_index.phase)
            dk_inv_part_index = advance(dk_inv_part_index, 1)
            dk_inv_part_vec = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dk_inv_acc_offset), cutlass.Float32), num=cfg.b_t)
            for t in cutlass.range_constexpr(cfg.b_t):
                dk_n[t] = dk_n[t] + dk_inv_part_vec[t] * cute.math.rcp(eg[t], approx=True, ftz=True)

            # ---- dK_decay part drain: -eG scale, seeds dGate ------------------------
            bars.mb_dk_decay_part_acc_ready.wait(dk_decay_part_index.phase)
            dk_decay_part_index = advance(dk_decay_part_index, 1)
            dk_decay_part_vec = nvvm.tcgen05_ld(
                "32x32b", nvvm.make_tmem_ptr(row_addr + (tmem_col + cfg.tmem_dk_decay_acc_offset), cutlass.Float32), num=cfg.b_t
            )
            for t in cutlass.range_constexpr(cfg.b_t):
                dgate_regs[t] = -eg[t] * dk_decay_part_vec[t]
                dk_n[t] = dk_n[t] + dgate_regs[t]

            nvvm.tcgen05_wait("load")
            bars.mb_dqk_acc_done.arrive()

            # ---- dGate finalize --------------------------------------------------
            qf_words = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + qraw_col, cutlass.Float32), num=cfg.b_t // 2)
            kf_words = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + kraw_col, cutlass.Float32), num=cfg.b_t // 2)
            for t in cutlass.range_constexpr(cfg.b_t):
                q_pair = cutlass.Vector.from_elements((qf_words[t // 2],), cutlass.Float32).bitcast(cfg.io_dtype)
                k_pair = cutlass.Vector.from_elements((kf_words[t // 2],), cutlass.Float32).bitcast(cfg.io_dtype)
                q_v = q_pair[t % 2].to(cutlass.Float32)
                k_v = k_pair[t % 2].to(cutlass.Float32)
                if cutlass.const_expr(cfg.l2norm):
                    q_v = q_v * sNorm_raw[norm_base + t]
                    k_v = k_v * sNorm_raw[norm_base + cfg.b_t + t]
                dgate_regs[t] = q_v * dq_n[t] + k_v * (cutlass.Float32(2.0) * dgate_regs[t] - dk_n[t])
            dgate_regs[cfg.b_t - 1] = dgate_regs[cfg.b_t - 1] + ((dgate_last_acc[0] + dgate_last_acc[1]) + (dgate_last_acc[2] + dgate_last_acc[3]))

            # ---- L2-norm backward row projection ---------------------------------
            if cutlass.const_expr(cfg.l2norm):
                for grad, qk_col, inv_off in ((dq_n, qraw_col, 0), (dk_n, kraw_col, cfg.b_t)):
                    dots = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
                    for half in cutlass.range_constexpr(2):
                        p_words = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + qk_col + half * (cfg.b_t // 4), cutlass.Float32), num=cfg.b_t // 4)
                        for tt2 in cutlass.range_constexpr(cfg.b_t // 4):
                            t = half * (cfg.b_t // 2) + 2 * tt2
                            p_pair = cutlass.Vector.from_elements((p_words[tt2],), cutlass.Float32).bitcast(cfg.io_dtype)
                            gp_lo, gp_hi = fmul2(grad[t], grad[t + 1], p_pair[0].to(cutlass.Float32), p_pair[1].to(cutlass.Float32))
                            dots[t], dots[t + 1] = fmul2(gp_lo, gp_hi, sNorm_raw[norm_base + inv_off + t], sNorm_raw[norm_base + inv_off + t + 1])
                    for off in cutlass.range_constexpr(5):
                        step = cutlass.const_expr(1 << off)
                        for t2 in cutlass.range_constexpr(cfg.b_t // 2):
                            t = 2 * t2
                            sh_lo = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, dots[t], step, 31, kind=nvvm.Shfl.BFLY))
                            sh_hi = cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, dots[t + 1], step, 31, kind=nvvm.Shfl.BFLY))
                            dots[t], dots[t + 1] = fadd2(dots[t], dots[t + 1], sh_lo, sh_hi)
                    if lane == 0:
                        for t in cutlass.range_constexpr(cfg.b_t):
                            sRed1_raw[tmem_subpartition * cfg.b_t + t] = dots[t]
                    nvvm.barrier_cta_sync(cfg.cg2_sync_barrier_id, thread_count=cfg.cg2_threads)
                    for half in cutlass.range_constexpr(2):
                        a_words = nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + qk_col + half * (cfg.b_t // 4), cutlass.Float32), num=cfg.b_t // 4)
                        for tt2 in cutlass.range_constexpr(cfg.b_t // 4):
                            t = half * (cfg.b_t // 2) + 2 * tt2
                            a_pair = cutlass.Vector.from_elements((a_words[tt2],), cutlass.Float32).bitcast(cfg.io_dtype)
                            dot_lo, dot_hi = fadd2(sRed1_raw[t], sRed1_raw[t + 1], sRed1_raw[cfg.b_t + t], sRed1_raw[cfg.b_t + t + 1])
                            dot_lo, dot_hi = fadd2(dot_lo, dot_hi, sRed1_raw[2 * cfg.b_t + t], sRed1_raw[2 * cfg.b_t + t + 1])
                            dot_lo, dot_hi = fadd2(dot_lo, dot_hi, sRed1_raw[3 * cfg.b_t + t], sRed1_raw[3 * cfg.b_t + t + 1])
                            norm_lo = sNorm_raw[norm_base + inv_off + t]
                            norm_hi = sNorm_raw[norm_base + inv_off + t + 1]
                            an_lo, an_hi = fmul2(a_pair[0].to(cutlass.Float32), a_pair[1].to(cutlass.Float32), norm_lo, norm_hi)
                            sub_lo = grad[t] - an_lo * dot_lo
                            sub_hi = grad[t + 1] - an_hi * dot_hi
                            grad[t], grad[t + 1] = fmul2(sub_lo, sub_hi, norm_lo, norm_hi)
                    nvvm.barrier_cta_sync(cfg.cg2_sync_barrier_id, thread_count=cfg.cg2_threads)

            bars.mb_qk_raw_done[qk_raw_stage].arrive()

            # ---- stage dQ/dK for the epilogue TMA stores -------------------------
            dq_stage = chunk_serial % cfg.smem_dq_stages
            dk_stage = chunk_serial % cfg.smem_dk_stages
            bars.mb_dq_tmastg_done[dq_stage].wait(((chunk_serial // cfg.smem_dq_stages) + 1) % 2)
            bars.mb_dk_tmastg_done[dk_stage].wait(((chunk_serial // cfg.smem_dk_stages) + 1) % 2)
            dq_base = dq_stage * (cfg.b_t * cfg.d_k)
            dk_base = dk_stage * (cfg.b_t * cfg.d_k)
            for t in cutlass.range_constexpr(cfg.b_t):
                out_idx = f16_seg * (cfg.b_t * 64) + t * 64 + swizzle_xor_128b(t, f16_dim, elem_bytes=2)
                (smem_data_ptr(sDq_raw) + dq_base + out_idx).store(dq_n[t].to(cfg.io_dtype))
                (smem_data_ptr(sDk_raw) + dk_base + out_idx).store(dk_n[t].to(cfg.io_dtype))
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dq_tmastg_ready[dq_stage].arrive()
            bars.mb_dk_tmastg_ready[dk_stage].arrive()

            # ---- dGate_last add --------------------------------------------------
            if has_dstate:
                if chunk_idx >= FIRST_STATE_CHUNK:
                    dgate_regs[cfg.b_t - 1] = dgate_regs[cfg.b_t - 1] + egl * dgate_last_val

            # ---- dGate reverse cumsum --------------------------------------------
            suffix = cutlass.Float32(0.0)
            for rt in cutlass.range_constexpr(cfg.b_t):
                t = cfg.b_t - 1 - rt
                suffix = suffix + dgate_regs[t]
                dgate_regs[t] = suffix

            # ---- stage dGate for the epilogue TMA store --------------------------
            dgate_stage = chunk_serial % cfg.smem_dgate_stages
            bars.mb_dgate_tmastg_done[dgate_stage].wait(((chunk_serial // cfg.smem_dgate_stages) + 1) % 2)
            for t in cutlass.range_constexpr(cfg.b_t):
                dgate_idx = f32_seg * (cfg.b_t * 32) + t * 32 + swizzle_xor_128b(t, f32_dim, elem_bytes=4)
                (smem_data_ptr(sDgate_raw) + dgate_idx).store(dgate_regs[t])
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dgate_tmastg_ready[dgate_stage].arrive()
            raw_index = advance(raw_index, cfg.smem_raw_stages)
        chunk_serial_base += num_compute_chunks
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)

    bars.mb_tmem_done[0].arrive()


# ---------------------------------------------------------------------------
# Host-side assembly
# ---------------------------------------------------------------------------


@cute.jit
def build_descs_body(
    widx,
    base_q,
    base_k,
    base_v,
    base_gate,
    base_do,
    base_dq,
    base_dk,
    base_dv,
    base_dgate,
    base_checkpoint,
    desc_ws: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    do: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dgate: cute.Tensor,
    state_checkpoints: cute.Tensor,
    n_batch: cutlass.Int32,
    q_row_stride: cutlass.Int64,
    k_row_stride: cutlass.Int64,
    v_row_stride: cutlass.Int64,
    gate_row_stride: cutlass.Int64,
    do_row_stride: cutlass.Int64,
    dq_row_stride: cutlass.Int64,
    dk_row_stride: cutlass.Int64,
    dv_row_stride: cutlass.Int64,
    dgate_row_stride: cutlass.Int64,
    checkpoint_row_stride: cutlass.Int64,
    checkpoint_every_n: cutlass.Int32,
) -> None:
    """Per-batch descriptor-array build, one warp per array. Runs inside the
    prologue kernel after its order pass; warps past the array count fall
    through the widx guards."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    sub0 = cute.make_tensor(desc_ws.iterator, cute.make_layout((arr_words,), stride=(1,)))
    sub1 = cute.make_tensor(desc_ws.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub2 = cute.make_tensor(desc_ws.iterator + 2 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub3 = cute.make_tensor(desc_ws.iterator + 3 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub4 = cute.make_tensor(desc_ws.iterator + 4 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub5 = cute.make_tensor(desc_ws.iterator + 5 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub6 = cute.make_tensor(desc_ws.iterator + 6 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub7 = cute.make_tensor(desc_ws.iterator + 7 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub8 = cute.make_tensor(desc_ws.iterator + 8 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    sub9 = cute.make_tensor(desc_ws.iterator + 9 * arr_words, cute.make_layout((arr_words,), stride=(1,)))

    if widx == 0:
        if nvvm.elect_sync():
            emit_seq_load_descs(base_q, sub0, cu_seqlens, q, n_batch, q_row_stride, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 1:
        if nvvm.elect_sync():
            emit_seq_load_descs(base_k, sub1, cu_seqlens, k, n_batch, k_row_stride, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 2:
        if nvvm.elect_sync():
            emit_seq_load_descs(base_v, sub2, cu_seqlens, v, n_batch, v_row_stride, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 3:
        if nvvm.elect_sync():
            emit_seq_load_descs(
                base_gate, sub3, cu_seqlens, gate, n_batch, gate_row_stride, 2
            )
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 4:
        if nvvm.elect_sync():
            emit_seq_load_descs(base_do, sub4, cu_seqlens, do, n_batch, do_row_stride, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 5:
        if nvvm.elect_sync():
            emit_seq_descs(base_dq, sub5, cu_seqlens, dq, n_batch, dq_row_stride, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 6:
        if nvvm.elect_sync():
            emit_seq_descs(base_dk, sub6, cu_seqlens, dk, n_batch, dk_row_stride, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 7:
        if nvvm.elect_sync():
            emit_seq_descs(base_dv, sub7, cu_seqlens, dv, n_batch, dv_row_stride, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 8:
        if nvvm.elect_sync():
            emit_seq_descs(base_dgate, sub8, cu_seqlens, dgate, n_batch, dgate_row_stride, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)
    if widx == 9:
        if nvvm.elect_sync():
            emit_checkpoint_seq_descs(base_checkpoint, sub9, cu_seqlens, state_checkpoints, n_batch, checkpoint_row_stride, checkpoint_every_n, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


@cute.kernel
def prologue_kernel(
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
    has_sched: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    base_q: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_k: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_v: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_gate: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_do: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_dq: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_dk: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_dv: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_dgate: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_checkpoint: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    desc_ws: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    do: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dgate: cute.Tensor,
    state_checkpoints: cute.Tensor,
    mStaging: cute.Tensor | None,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor | None,
    mSched: cute.Tensor | None,
    n_batch: cutlass.Int32,
    q_row_stride: cutlass.Int64,
    k_row_stride: cutlass.Int64,
    v_row_stride: cutlass.Int64,
    gate_row_stride: cutlass.Int64,
    do_row_stride: cutlass.Int64,
    dq_row_stride: cutlass.Int64,
    dk_row_stride: cutlass.Int64,
    dv_row_stride: cutlass.Int64,
    dgate_row_stride: cutlass.Int64,
    checkpoint_row_stride: cutlass.Int64,
    checkpoint_every_n: cutlass.Int32,
) -> None:
    """Single-CTA prologue. Under ``run_order`` this kernel is the first
    work-item-table consumer, so it LPT-orders the table and zeroes both
    consumers' sched rings via :func:`order_body`; it then builds the
    per-batch TMA-descriptor arrays via :func:`build_descs_body`, one warp
    per array (the extra warps only take part in the order phase)."""
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
    if cutlass.const_expr(run_order):
        sKey = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sIdx = cutlass.Array(cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16)
        sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
        n_heads_out = cutlass.Int32(gate.shape[1])
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
        base_gate,
        base_do,
        base_dq,
        base_dk,
        base_dv,
        base_dgate,
        base_checkpoint,
        desc_ws,
        cu_seqlens,
        q,
        k,
        v,
        gate,
        do,
        dq,
        dk,
        dv,
        dgate,
        state_checkpoints,
        n_batch,
        q_row_stride,
        k_row_stride,
        v_row_stride,
        gate_row_stride,
        do_row_stride,
        dq_row_stride,
        dk_row_stride,
        dv_row_stride,
        dgate_row_stride,
        checkpoint_row_stride,
        checkpoint_every_n,
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
    gate: cute.Tensor,
    do: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dgate: cute.Tensor,
    state_checkpoints: cute.Tensor,
    cu_seqlens: cute.Tensor,
    work_item_staging: cute.Tensor | None,
    work_count: cute.Tensor,
    work_items: cute.Tensor | None,
    sched_all: cute.Tensor | None,
    tensormap_workspace: cute.Tensor,
    stream: cuda_driver.CUstream,
):
    """One-launch prologue: LPT-order the work items (when ``run_order``) and
    build the 11 per-(batch, head) capped TMA-descriptor arrays into
    ``tensormap_workspace`` (sequence-relative coordinates; tail loads
    zero-fill and tail stores clip in hardware)."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    h_v = v.shape[1]
    ho = gate.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    d_k = q.shape[2]
    d_v = v.shape[2]
    bpe = io_dtype.width // 8
    tma_granu_elems = 128 // bpe
    seqlen = q.shape[0]

    q_headed = cute.make_tensor(q.iterator, cute.make_layout((d_k, h_q, seqlen), stride=(1, q.stride[1], q.stride[0])))
    k_headed = cute.make_tensor(k.iterator, cute.make_layout((d_k, h_k, seqlen), stride=(1, k.stride[1], k.stride[0])))
    v_headed = cute.make_tensor(v.iterator, cute.make_layout((d_v, h_v, seqlen), stride=(1, v.stride[1], v.stride[0])))
    gate_headed = cute.make_tensor(gate.iterator, cute.make_layout((d_k, ho, seqlen), stride=(1, gate.stride[1], gate.stride[0])))
    do_headed = cute.make_tensor(do.iterator, cute.make_layout((d_v, ho, seqlen), stride=(1, do.stride[1], do.stride[0])))
    dq_headed = cute.make_tensor(dq.iterator, cute.make_layout((d_k, ho, seqlen), stride=(1, dq.stride[1], dq.stride[0])))
    dk_headed = cute.make_tensor(dk.iterator, cute.make_layout((d_k, ho, seqlen), stride=(1, dk.stride[1], dk.stride[0])))
    dv_headed = cute.make_tensor(dv.iterator, cute.make_layout((d_v, ho, seqlen), stride=(1, dv.stride[1], dv.stride[0])))
    dgate_headed = cute.make_tensor(dgate.iterator, cute.make_layout((d_k, ho, seqlen), stride=(1, dgate.stride[1], dgate.stride[0])))

    swz = cuda.TensorMapSwizzle.s128b
    base_q = cuda.create_tensor_map_tiled_from_view(q_headed, box_dims=(tma_granu_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_k = cuda.create_tensor_map_tiled_from_view(k_headed, box_dims=(tma_granu_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_v = cuda.create_tensor_map_tiled_from_view(v_headed, box_dims=(tma_granu_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_gate = cuda.create_tensor_map_tiled_from_view(gate_headed, box_dims=(32, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_do = cuda.create_tensor_map_tiled_from_view(do_headed, box_dims=(tma_granu_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_dq = cuda.create_tensor_map_tiled_from_view(dq_headed, box_dims=(tma_granu_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_dk = cuda.create_tensor_map_tiled_from_view(dk_headed, box_dims=(tma_granu_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_dv = cuda.create_tensor_map_tiled_from_view(dv_headed, box_dims=(tma_granu_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_dgate = cuda.create_tensor_map_tiled_from_view(dgate_headed, box_dims=(32, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)

    checkpoint_view = cute.make_tensor(
        state_checkpoints.iterator,
        cute.make_layout(
            (d_v, d_k, state_checkpoints.shape[0], ho),
            stride=(state_checkpoints.stride[3], state_checkpoints.stride[2], state_checkpoints.stride[0], state_checkpoints.stride[1]),
        ),
    )
    base_checkpoint = cuda.create_tensor_map_tiled_from_view(checkpoint_view, box_dims=(64, d_k, 1, 1), stride_order=(0, 1, 2, 3), swizzle=swz)

    prologue_kernel(
        run_order,
        order_gen,
        has_sched,
        b_t,
        base_q,
        base_k,
        base_v,
        base_gate,
        base_do,
        base_dq,
        base_dk,
        base_dv,
        base_dgate,
        base_checkpoint,
        tensormap_workspace,
        cu_seqlens,
        q,
        k,
        v,
        gate,
        do,
        dq,
        dk,
        dv,
        dgate,
        state_checkpoints,
        work_item_staging,
        work_count,
        work_items,
        sched_all,
        cutlass.Int32(batch_size),
        cutlass.Int64(q.stride[0]),
        cutlass.Int64(k.stride[0]),
        cutlass.Int64(v.stride[0]),
        cutlass.Int64(gate.stride[0]),
        cutlass.Int64(do.stride[0]),
        cutlass.Int64(dq.stride[0]),
        cutlass.Int64(dk.stride[0]),
        cutlass.Int64(dv.stride[0]),
        cutlass.Int64(dgate.stride[0]),
        cutlass.Int64(state_checkpoints.stride[0]),
        cutlass.Int32(b_t),
    ).launch(grid=(1, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream)


class KdaBpropOp:
    """Compile and launch one static mega KDA backward configuration."""

    def __init__(self, cfg: "KdaBwdCfg", use_int64_offsets: bool = False):
        self.cfg = cfg
        self.use_int64_offsets = use_int64_offsets

    def get_name(self) -> str:
        cfg = self.cfg
        gate_scale = (
            f"_gs{float(cfg.gate_scale_log2).hex()}"
            .replace("-", "m")
            .replace(".", "p")
            .replace("+", "")
            if cfg.safe_gate
            else ""
        )
        return (
            f"kda_mega_bprop_{cfg.io_dtype.__name__.lower()}_h{cfg.n_heads_out}"
            f"_q{cfg.q_ratio}_k{cfg.k_ratio}_v{cfg.v_ratio}"
            f"_s{int(cfg.use_dstate_in)}z{int(cfg.use_dstate0)}"
            f"l{int(cfg.l2norm)}g{int(cfg.safe_gate)}b{int(cfg.beta_sigmoid)}"
            f"i{int(cfg.use_initial_state)}d{int(cfg.dyn_sched)}"
            f"_sm{cfg.max_active_clusters}_i64{int(self.use_int64_offsets)}{gate_scale}"
        )

    @cute.jit
    def __call__(
        self,
        a_log: cute.Tensor | None,
        dt_bias: cute.Tensor | None,
        beta: cute.Tensor,
        state_checkpoints: cute.Tensor,
        dgate: cute.Tensor,
        dbeta: cute.Tensor,
        cu_seqlens: cute.Tensor,
        d_initial_state: cute.Tensor | None,
        d_final_state: cute.Tensor | None,
        work_items: cute.Tensor | None,
        work_count: cute.Tensor | None,
        sched_ctr: cute.Tensor | None,
        tensormap_workspace: cute.Tensor,
        scale: cutlass.Float32,
        stream,
    ) -> None:
        cfg = self.cfg
        num_sequences = cu_seqlens.shape[0] - 1

        @cute.struct
        class SharedStorage:
            state: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.state_cosize], cfg.buffer_align_bytes
            ]
            dstate: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.d_k * cfg.d_v], cfg.buffer_align_bytes
            ]
            k_decay: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.operand_cosize], cfg.buffer_align_bytes
            ]
            k_inv: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.operand_cosize], cfg.buffer_align_bytes
            ]
            k_restore: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.operand_cosize], cfg.buffer_align_bytes
            ]
            q_decay: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.operand_cosize], cfg.buffer_align_bytes
            ]
            state_scale_diag: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.diag_cosize], cfg.buffer_align_bytes
            ]
            intermediate: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.intermediate_cosize],
                cfg.buffer_align_bytes,
            ]
            do: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.raw_v_cosize], cfg.buffer_align_bytes
            ]
            dv: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.dv_cosize], cfg.buffer_align_bytes
            ]
            q: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.raw_qk_cosize], cfg.buffer_align_bytes
            ]
            k: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.raw_qk_cosize], cfg.buffer_align_bytes
            ]
            v: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.raw_v_cosize], cfg.buffer_align_bytes
            ]
            gate: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cfg.raw_gate_cosize], 1024
            ]
            dy: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.b_t * cfg.d_v], cfg.buffer_align_bytes
            ]
            u: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.b_t * cfg.d_v], cfg.buffer_align_bytes
            ]
            dq: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.dq_cosize], cfg.buffer_align_bytes
            ]
            dk: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.dk_cosize], cfg.buffer_align_bytes
            ]
            dgate: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cfg.dgate_cosize], 1024
            ]

        kernel.set_name_prefix(self.get_name())
        kernel(
            cfg,
            SharedStorage,
            tensormap_workspace,
            num_sequences,
            a_log,
            dt_bias,
            beta,
            cu_seqlens,
            dgate,
            dbeta,
            d_initial_state,
            d_final_state,
            work_items,
            work_count,
            sched_ctr,
            scale,
        ).launch(
            grid=(cfg.max_active_clusters, 1, 1),
            block=(cfg.threads_per_cta, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )


@cute.kernel
def kernel(
    cfg: cutlass.Constexpr,
    shared_type: cutlass.Constexpr,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
    mA_log: cute.Tensor | None,
    mDt_bias: cute.Tensor | None,
    mBeta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    mDgate: cute.Tensor,
    mDbeta: cute.Tensor,
    mDstate0: cute.Tensor | None,
    mDstate_in: cute.Tensor | None,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mSched: cute.Tensor | None,
    scale: cutlass.Float32,
) -> None:
    """BT=16 KDA backward kernel (persistent, 16 warps)."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % cfg.threads_per_warp

    total_tiles = mCount[0]
    beta_expected = cfg.io_dtype if cutlass.const_expr(cfg.beta_sigmoid) else cutlass.Float32
    assert mBeta.element_type == beta_expected
    assert cu_seqlens.element_type in (cutlass.Int32, cutlass.Int64)
    assert mDgate.element_type == cutlass.Float32 and mDbeta.element_type == beta_expected

    desc_base_words = tensormap_workspace.iterator.raw_ptr()
    arr_words = n_desc * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_q_base = desc_base_words
    desc_k_base = desc_base_words + arr_words
    desc_v_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_gate_base = desc_base_words + cutlass.Int32(3) * arr_words
    desc_do_base = desc_base_words + cutlass.Int32(4) * arr_words
    desc_dq_base = desc_base_words + cutlass.Int32(5) * arr_words
    desc_dk_base = desc_base_words + cutlass.Int32(6) * arr_words
    desc_dv_base = desc_base_words + cutlass.Int32(7) * arr_words
    desc_dgate_base = desc_base_words + cutlass.Int32(8) * arr_words
    desc_checkpoint_base = desc_base_words + cutlass.Int32(9) * arr_words

    SMEM = cutlass.AddressSpace.smem
    bars = make_kda_bwd_bars(cfg)
    tmem_base_holder = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=4)
    sSched = cutlass.Array(cutlass.Int32, cfg.sched_stages, space=SMEM, alignment=16)
    bpe = cfg.io_dtype.width // 8
    SWZ = 2
    LEAD = 16
    STRIDE = 8 * 128
    STATE_ALT_LEAD = cfg.d_v * 128

    # sub-bank split: tcgen05-descriptor operands low, generic-client buffers high
    sBeta_raw = cutlass.Array(cutlass.Float32, cfg.smem_beta_stages * cfg.b_t, space=SMEM, alignment=64)
    sNorm_raw = cutlass.Array(cutlass.Float32, cfg.tmem_qk_raw_stages * 2 * cfg.b_t, space=SMEM, alignment=64)
    sRed_raw = cutlass.Array(cutlass.Float32, 4 * cfg.b_t, space=SMEM, alignment=64)
    sRed1_raw = cutlass.Array(cutlass.Float32, 4 * cfg.b_t, space=SMEM, alignment=64)
    sBetaM_raw = cutlass.Array(cutlass.Float32, cfg.b_t, space=SMEM, alignment=64)
    storage = cutlass.utils.SmemAllocator().allocate(shared_type)
    sState_raw = storage.state.get_tensor(cute.make_layout((cfg.state_cosize,)))
    sDstate_raw = storage.dstate.get_tensor(cute.make_layout((cfg.d_k * cfg.d_v,)))
    sK_decay_raw = storage.k_decay.get_tensor(cute.make_layout((cfg.operand_cosize,)))
    sK_inv_raw = storage.k_inv.get_tensor(cute.make_layout((cfg.operand_cosize,)))
    sK_restore_raw = storage.k_restore.get_tensor(cute.make_layout((cfg.operand_cosize,)))
    sQ_decay_raw = storage.q_decay.get_tensor(cute.make_layout((cfg.operand_cosize,)))
    sState_scale_diag_raw = storage.state_scale_diag.get_tensor(
        cute.make_layout((cfg.diag_cosize,))
    )
    sIntermediate_raw = storage.intermediate.get_tensor(
        cute.make_layout((cfg.intermediate_cosize,))
    )
    sDo_raw = storage.do.get_tensor(cute.make_layout((cfg.raw_v_cosize,)))
    sDv_raw = storage.dv.get_tensor(cute.make_layout((cfg.dv_cosize,)))
    sQ_raw = storage.q.get_tensor(cute.make_layout((cfg.raw_qk_cosize,)))
    sK_raw = storage.k.get_tensor(cute.make_layout((cfg.raw_qk_cosize,)))
    sV_raw = storage.v.get_tensor(cute.make_layout((cfg.raw_v_cosize,)))
    sGate_raw = storage.gate.get_tensor(cute.make_layout((cfg.raw_gate_cosize,)))
    sDy_raw = storage.dy.get_tensor(cute.make_layout((cfg.b_t * cfg.d_v,)))
    sU_raw = storage.u.get_tensor(cute.make_layout((cfg.b_t * cfg.d_v,)))
    sDq_raw = storage.dq.get_tensor(cute.make_layout((cfg.dq_cosize,)))
    sDk_raw = storage.dk.get_tensor(cute.make_layout((cfg.dk_cosize,)))
    sDgate_raw = storage.dgate.get_tensor(cute.make_layout((cfg.dgate_cosize,)))

    sState_alt = SmemTile(
        base=smem_data_ptr(sState_raw).toint(),
        elems_per_stage=((cfg.state_cosize) // (cfg.smem_state_stages)) * bpe,
        stages=cfg.smem_state_stages,
        leading_byte_offset=STATE_ALT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sState_direct = SmemTile(
        base=smem_data_ptr(sState_raw).toint(),
        elems_per_stage=((cfg.state_cosize) // (cfg.smem_state_stages)) * bpe,
        stages=cfg.smem_state_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_decay_lead16 = SmemTile(
        base=smem_data_ptr(sK_decay_raw).toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bpe,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_inv_lead16 = SmemTile(
        base=smem_data_ptr(sK_inv_raw).toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bpe,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_restore_lead16 = SmemTile(
        base=smem_data_ptr(sK_restore_raw).toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bpe,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDo_lead16 = SmemTile(
        base=smem_data_ptr(sDo_raw).toint(),
        elems_per_stage=((cfg.raw_v_cosize) // (cfg.smem_raw_stages)) * bpe,
        stages=cfg.smem_raw_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDo_amaj = SmemTile(
        base=smem_data_ptr(sDo_raw).toint(),
        elems_per_stage=((cfg.raw_v_cosize) // (cfg.smem_raw_stages)) * bpe,
        stages=cfg.smem_raw_stages,
        leading_byte_offset=cfg.b_t * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sU_lead16 = SmemTile(
        base=smem_data_ptr(sU_raw).toint(),
        elems_per_stage=((cfg.b_t * cfg.d_v) // (1)) * bpe,
        stages=1,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDv_lead16 = SmemTile(
        base=smem_data_ptr(sDv_raw).toint(),
        elems_per_stage=((cfg.dv_cosize) // (cfg.smem_dv_stages)) * bpe,
        stages=cfg.smem_dv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDstate_alt = SmemTile(
        base=smem_data_ptr(sDstate_raw).toint(),
        elems_per_stage=((cfg.d_k * cfg.d_v) // (1)) * bpe,
        stages=1,
        leading_byte_offset=STATE_ALT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )

    sQ_decay_trans = SmemTile(
        base=smem_data_ptr(sQ_decay_raw).toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bpe,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=cfg.b_t * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_inv_amaj = SmemTile(
        base=smem_data_ptr(sK_inv_raw).toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bpe,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=cfg.b_t * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sK_decay_trans = SmemTile(
        base=smem_data_ptr(sK_decay_raw).toint(),
        elems_per_stage=((cfg.operand_cosize) // (cfg.smem_decay_stages)) * bpe,
        stages=cfg.smem_decay_stages,
        leading_byte_offset=cfg.b_t * 128,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sState_scale_diag = SmemTile(
        base=sState_scale_diag_raw,
        elems_per_stage=((cfg.d_k // 16) * 256),
        stages=cfg.smem_decay_stages,
        leading_byte_offset=16,
        stride_byte_offset=(8 * 16 * 2),
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_32B,
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
    if warp_idx == cfg.tma_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_raw_stages):
                bars.mb_q_ready[stage].init()
                bars.mb_q_done[stage].init()
                bars.mb_k_ready[stage].init()
                bars.mb_k_done[stage].init()
                bars.mb_gate_ready[stage].init()
                bars.mb_gate_done[stage].init()
                bars.mb_do_ready[stage].init()
                bars.mb_do_done[stage].init()
                bars.mb_v_ready[stage].init()
                bars.mb_v_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_beta_stages):
                bars.mb_beta_ready[stage].init()
                bars.mb_beta_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_state_stages):
                bars.mb_state_ready[stage].init()
                bars.mb_state_done[stage].init()
                bars.mb_state_cg0_done[stage].init()
            for stage in cutlass.range_constexpr(2):
                bars.mb_state_inp_ready[stage].init()
                bars.mb_state_inp_done[stage].init()
                bars.mb_state_inp_cg2_done[stage].init()
    elif warp_idx == cfg.tcgen05_mma_warp_id:
        if elect_one:
            bars.mb_state_k_acc_ready.init()
            bars.mb_y_inp_ready.init()
            bars.mb_u_acc_ready.init()
            bars.mb_u_smem_ready.init()
            bars.mb_du_acc_ready.init()
            bars.mb_du_inp_ready.init()
            bars.mb_dy_acc_ready.init()
            bars.mb_neg_beta_dy_inp_ready.init()
            bars.mb_dy_smem_ready.init()
            bars.mb_dstate_acc_ready.init()
            bars.mb_dstate_inp_ready.init()
            bars.mb_dstate_smem_ready.init()
            bars.mb_dstate_smem_done.init()
            bars.mb_dstate_smem_cg2_done.init()
            bars.mb_dq_acc_ready.init()
            bars.mb_dk_decay_part_acc_ready.init()
            bars.mb_dk_inv_part_acc_ready.init()
            bars.mb_dk_restore_part_acc_ready.init()
            bars.mb_dqk_acc_done.init()
            bars.mb_dbeta_m_ready.init()
            bars.mb_dstate0_acc_stored.init()
            bars.mb_tmem_done[0].init()
    elif warp_idx == cfg.super_mma_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_k_decay_inv_ready[stage].init()
                bars.mb_q_decay_k_restore_ready[stage].init()
                bars.mb_decay_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.tmem_qk_raw_stages):
                bars.mb_qk_raw_ready[stage].init()
                bars.mb_qk_raw_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_intermediate_stages):
                bars.mb_t_inv_ready[stage].init()
                bars.mb_a_ready[stage].init()
                bars.mb_da_ready[stage].init()
                bars.mb_dm_ready[stage].init()
                bars.mb_a_done[stage].init()
                bars.mb_t_inv_done[stage].init()
                bars.mb_da_done[stage].init()
                bars.mb_dm_done[stage].init()
    elif warp_idx == cfg.epilogue_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_dq_stages):
                bars.mb_dq_tmastg_ready[stage].init()
                bars.mb_dq_tmastg_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_dk_stages):
                bars.mb_dk_tmastg_ready[stage].init()
                bars.mb_dk_tmastg_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_dgate_stages):
                bars.mb_dgate_tmastg_ready[stage].init()
                bars.mb_dgate_tmastg_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_dv_stages):
                bars.mb_dv_tmastg_ready[stage].init()
                bars.mb_dv_tmastg_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.sched_stages):
                bars.mb_sched_ready[stage].init()
                bars.mb_sched_done[stage].init()
    diag_zero = cfg.io_dtype(0.0)
    for diag_idx in cutlass.range(tidx, cfg.diag_cosize, cfg.threads_per_cta, unroll=1):
        sState_scale_diag_raw[diag_idx] = diag_zero
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync(0, thread_count=cfg.threads_per_cta)
    if warp_idx == cfg.tma_warp_id:
        tmaldg_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            n_desc,
            cu_seqlens,
            mWorkItems,
            mSched,
            sSched,
            lane,
            sQ_raw,
            sK_raw,
            sV_raw,
            sGate_raw,
            sDo_raw,
            sState_raw,
            desc_q_base,
            desc_k_base,
            desc_v_base,
            desc_gate_base,
            desc_do_base,
            desc_checkpoint_base,
            bars,
        )
    elif warp_idx == cfg.super_mma_warp_id:
        super_mma_warp(
            cfg,
            total_tiles,
            bidx,
            num_ctas,
            cu_seqlens,
            mWorkItems,
            sSched,
            lane,
            sK_decay_raw,
            sK_inv_raw,
            sU_raw,
            sDy_raw,
            sIntermediate_raw,
            sBeta_raw,
            sBetaM_raw,
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
            sSched,
            tmem_base_holder,
            sState_alt,
            sState_direct,
            sK_decay_lead16,
            sK_inv_lead16,
            sK_inv_amaj,
            sK_restore_lead16,
            sDo_lead16,
            sDo_amaj,
            sQ_decay_trans,
            sK_decay_trans,
            sU_lead16,
            sDv_lead16,
            sDstate_alt,
            sIntermediate,
            sState_scale_diag,
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
            sSched,
            lane,
            sK_inv_raw,
            sQ_decay_raw,
            sDo_raw,
            sU_raw,
            sIntermediate_raw,
            sDq_raw,
            sDk_raw,
            sDv_raw,
            sDgate_raw,
            desc_dq_base,
            desc_dk_base,
            desc_dv_base,
            desc_dgate_base,
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
            sSched,
            lane,
            tmem_base_holder,
            warp_idx,
            scale,
            mA_log,
            mDt_bias,
            mBeta,
            sBeta_raw,
            sK_inv_raw,
            sGate_raw,
            sK_raw,
            sQ_raw,
            sState_raw,
            sNorm_raw,
            sK_decay_raw,
            sK_restore_raw,
            sQ_decay_raw,
            sState_scale_diag_raw,
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
            sSched,
            lane,
            tmem_base_holder,
            warp_idx,
            sGate_raw,
            sNorm_raw,
            sDq_raw,
            sDk_raw,
            sRed1_raw,
            sDstate_raw,
            sDgate_raw,
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
            sSched,
            lane,
            tmem_base_holder,
            warp_idx,
            mDbeta,
            mDstate0,
            mDstate_in,
            sV_raw,
            sDo_raw,
            sBeta_raw,
            sU_raw,
            sDy_raw,
            sDv_raw,
            sDstate_raw,
            sRed_raw,
            sBetaM_raw,
            bars,
        )


@dataclass
class KdaBwdCfg:
    """Kernel cfg (fixed BT=16 schedule constants; derived TMEM column offsets
    and SMEM buffer cosizes are stamped by ``build_cfg``)."""

    io_dtype: Type[cutlass.Numeric]
    use_dstate_in: bool
    use_dstate0: bool
    l2norm: bool
    safe_gate: bool
    gate_scale_log2: float
    beta_sigmoid: bool
    use_initial_state: bool
    q_ratio: int
    k_ratio: int
    v_ratio: int
    n_heads_out: int
    max_active_clusters: int
    dyn_sched: bool = False
    sched_stages: int = 8

    # ---- fixed constants stamped from CFG by build_cfg ---------------------------
    compute_group_0_warp_ids: tuple = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_2_warp_ids: tuple = CFG.COMPUTE_GROUP_2_WARP_IDS
    compute_group_1_warp_ids: tuple = CFG.COMPUTE_GROUP_1_WARP_IDS
    super_mma_warp_id: int = CFG.SUPER_MMA_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    tma_warp_id: int = CFG.TMA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    b_t: int = CFG.B_T
    d_k: int = CFG.D_K
    d_v: int = CFG.D_V
    threads_per_warp: int = CFG.THREADS_PER_WARP
    threads_per_cta: int = 0
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_compute_group_2: int = CFG.NUM_REGS_COMPUTE_GROUP_2
    num_regs_other: int = CFG.NUM_REGS_OTHER

    # ---- named barrier slots (ids 1-4; 0 is the CTA-wide sync) -------------------
    cg0_sync_barrier_id: int = 1
    cg0_threads: int = 0
    cg2_sync_barrier_id: int = 2
    cg2_threads: int = 0
    tmem_lifecycle_barrier_id: int = 3
    tmem_user_threads: int = 0
    cg1_sync_barrier_id: int = 4
    cg1_threads: int = 0

    # ---- SMEM / TMEM stage counts + TMEM column offsets --------------------------
    smem_raw_stages: int = CFG.SMEM_RAW_STAGES
    smem_state_stages: int = CFG.SMEM_S_STAGES
    smem_decay_stages: int = CFG.SMEM_DECAY_STAGES
    smem_intermediate_stages: int = CFG.SMEM_INTERMEDIATE_STAGES
    smem_dq_stages: int = CFG.SMEM_DQ_STAGES
    smem_dk_stages: int = CFG.SMEM_DK_STAGES
    smem_dgate_stages: int = CFG.SMEM_DGATE_STAGES
    smem_dv_stages: int = CFG.SMEM_DV_STAGES
    smem_beta_stages: int = 4
    intermediate_tiles: int = 5
    tmem_dstate_acc_offset: int = 0
    tmem_dstate_inp_offset: int = 0
    tmem_state_k_acc_offset: int = 0
    tmem_u_acc_offset: int = 0
    tmem_du_acc_offset: int = 0
    tmem_dy_acc_offset: int = 0
    tmem_dq_acc_offset: int = 0
    tmem_dk_decay_acc_offset: int = 0
    tmem_dk_inv_acc_offset: int = 0
    tmem_dk_restore_acc_offset: int = 0
    tmem_qk_raw_stages: int = 4
    tmem_qraw_inp_offset: int = 0
    tmem_kraw_inp_offset: int = 0
    tmem_y_inp_offset: int = 0
    tmem_du_inp_offset: int = 0
    tmem_neg_beta_dy_inp_offset: int = 0
    tmem_state_inp_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- buffer cosizes / TMA bytes stamped by build_cfg -------------------------
    raw_qk_cosize: int = 0
    raw_v_cosize: int = 0
    raw_gate_cosize: int = 0
    operand_cosize: int = 0
    diag_cosize: int = 0
    intermediate_cosize: int = 0
    state_cosize: int = 0
    dq_cosize: int = 0
    dk_cosize: int = 0
    dgate_cosize: int = 0
    dv_cosize: int = 0

    # TMA transaction bytes per stage
    tma_q_bytes: int = 0
    tma_k_bytes: int = 0
    tma_gate_bytes: int = 0
    tma_do_bytes: int = 0
    tma_v_bytes: int = 0
    tma_state_bytes: int = 0


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    *,
    use_dstate_in: bool,
    use_dstate0: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_scale_log2: float,
    beta_sigmoid: bool,
    use_initial_state: bool,
    q_ratio: int,
    k_ratio: int,
    v_ratio: int,
    n_heads_out: int,
    max_active_clusters: int,
    dyn_sched: bool = False,
) -> KdaBwdCfg:
    if io_dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise ValueError(f"io_dtype={io_dtype} not supported; only Float16 and BFloat16 are supported")
    cfg = KdaBwdCfg(
        io_dtype=io_dtype,
        use_dstate_in=use_dstate_in,
        use_dstate0=use_dstate0,
        l2norm=l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        beta_sigmoid=beta_sigmoid,
        use_initial_state=use_initial_state,
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
        *cfg.compute_group_2_warp_ids,
        cfg.super_mma_warp_id,
        cfg.tcgen05_mma_warp_id,
        cfg.tma_warp_id,
        cfg.epilogue_warp_id,
    )
    if tuple(sorted(role_ids)) != tuple(range(len(role_ids))):
        raise ValueError("warp roles must be disjoint and cover contiguous IDs from zero")
    for group in (
        cfg.compute_group_0_warp_ids,
        cfg.compute_group_1_warp_ids,
        cfg.compute_group_2_warp_ids,
    ):
        if len(group) != 4 or group != tuple(range(group[0], group[-1] + 1)):
            raise ValueError("the fixed schedule requires contiguous four-warp compute groups")
    cfg.threads_per_cta = len(role_ids) * cfg.threads_per_warp
    cfg.cg0_threads = len(cfg.compute_group_0_warp_ids) * cfg.threads_per_warp
    cfg.cg2_threads = len(cfg.compute_group_2_warp_ids) * cfg.threads_per_warp
    cfg.cg1_threads = len(cfg.compute_group_1_warp_ids) * cfg.threads_per_warp
    cfg.tmem_user_threads = (
        1 + len(cfg.compute_group_2_warp_ids) + len(cfg.compute_group_1_warp_ids) + len(cfg.compute_group_0_warp_ids)
    ) * cfg.threads_per_warp
    named_barrier_ids = (
        cfg.cg0_sync_barrier_id,
        cfg.cg2_sync_barrier_id,
        cfg.tmem_lifecycle_barrier_id,
        cfg.cg1_sync_barrier_id,
    )
    if (
        any(not 1 <= barrier_id <= 15 for barrier_id in named_barrier_ids)
        or len(set(named_barrier_ids)) != len(named_barrier_ids)
    ):
        raise ValueError("named barrier IDs must be disjoint values in [1, 15]")

    cfg.tmem_dstate_acc_offset = 0
    cfg.tmem_dstate_inp_offset = cfg.d_k
    cfg.tmem_state_inp_offset = cfg.tmem_dstate_inp_offset + cfg.d_k // 2
    cfg.tmem_state_k_acc_offset = cfg.tmem_state_inp_offset + cfg.d_v
    cfg.tmem_u_acc_offset = cfg.tmem_state_k_acc_offset + cfg.b_t
    cfg.tmem_du_acc_offset = cfg.tmem_u_acc_offset + cfg.b_t
    cfg.tmem_dy_acc_offset = cfg.tmem_state_k_acc_offset
    cfg.tmem_dq_acc_offset = cfg.tmem_du_acc_offset + cfg.b_t
    cfg.tmem_dk_decay_acc_offset = cfg.tmem_dq_acc_offset + cfg.b_t
    cfg.tmem_dk_inv_acc_offset = cfg.tmem_dk_decay_acc_offset + cfg.b_t
    cfg.tmem_dk_restore_acc_offset = cfg.tmem_dk_inv_acc_offset + cfg.b_t
    cfg.tmem_y_inp_offset = cfg.tmem_dk_restore_acc_offset + cfg.b_t
    cfg.tmem_neg_beta_dy_inp_offset = cfg.tmem_y_inp_offset
    cfg.tmem_du_inp_offset = cfg.tmem_y_inp_offset + cfg.b_t // 2
    cfg.tmem_qraw_inp_offset = cfg.tmem_du_inp_offset + cfg.b_t // 2
    cfg.tmem_kraw_inp_offset = cfg.tmem_qraw_inp_offset + cfg.tmem_qk_raw_stages * (cfg.b_t // 2)
    assert cfg.tmem_kraw_inp_offset + cfg.tmem_qk_raw_stages * (cfg.b_t // 2) <= 512

    cfg.raw_qk_cosize = cfg.smem_raw_stages * cfg.d_k * cfg.b_t
    cfg.raw_v_cosize = cfg.smem_raw_stages * cfg.d_v * cfg.b_t
    cfg.raw_gate_cosize = cfg.smem_raw_stages * cfg.d_k * cfg.b_t
    cfg.operand_cosize = cfg.smem_decay_stages * cfg.b_t * cfg.d_k
    cfg.diag_cosize = cfg.smem_decay_stages * (cfg.d_k // 16) * 256
    cfg.intermediate_cosize = cfg.smem_intermediate_stages * cfg.intermediate_tiles * cfg.b_t * cfg.b_t
    cfg.state_cosize = cfg.smem_state_stages * cfg.d_k * cfg.d_v
    cfg.dq_cosize = cfg.smem_dq_stages * cfg.b_t * cfg.d_k
    cfg.dk_cosize = cfg.smem_dk_stages * cfg.b_t * cfg.d_k
    cfg.dgate_cosize = cfg.smem_dgate_stages * cfg.b_t * cfg.d_k
    cfg.dv_cosize = cfg.smem_dv_stages * cfg.b_t * cfg.d_v
    cfg.tma_state_bytes = cfg.d_k * cfg.d_v * (io_dtype.width // 8)
    cfg.tma_q_bytes = cfg.d_k * cfg.b_t * (cfg.io_dtype.width // 8)
    cfg.tma_k_bytes = cfg.d_k * cfg.b_t * (cfg.io_dtype.width // 8)
    cfg.tma_gate_bytes = cfg.d_k * cfg.b_t * 4
    cfg.tma_do_bytes = cfg.d_v * cfg.b_t * (cfg.io_dtype.width // 8)
    cfg.tma_v_bytes = cfg.d_v * cfg.b_t * (cfg.io_dtype.width // 8)
    return cfg


TENSORMAP_DESC_ARRAYS = 10  # per-batch runtime TMA descriptors: Q, K, V, Gate, dO, state_checkpoints, dQ, dK, dV, dGate
TENSORMAP_STATIC_SLOTS = 0


# ---- Torch adapter / host-side compilation ---------------------------------------


@lru_cache(maxsize=None)
def get_compiled_cache(
    io_dtype_str: str,
    HQ: int,
    HK: int,
    HV: int,
    use_dstate_in: bool,
    use_dstate0: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_lower_bound: float,
    beta_sigmoid: bool,
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


def chunk_kda_bwd_sm100(
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
    use_qk_l2norm_in_kernel: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: float = DEFAULT_GATE_LOWER_BOUND,
    a_log=None,
    dt_bias=None,
    use_beta_sigmoid: bool = False,
    work_items=None,
    work_count=None,
    sched_ctr=None,
    sched_all=None,
    work_item_scratch=None,
    order_in_prologue: bool = False,
    tensormap_workspace,
) -> None:
    """Execute the Blackwell BT=16 chunked KDA backward kernel.

    TMA operands require an aligned contiguous innermost mode; outer strides may vary.

    Args:
        q: ``(total_tokens, HQ, DK)`` float16/bfloat16
        k: ``(total_tokens, HK, DK)`` float16/bfloat16
        v: ``(total_tokens, HV, DV)`` float16/bfloat16
        gate: ``(total_tokens, HO, DK)`` fp32.  Natural-log per-channel decay
              unless ``safe_gate``, which applies the safe-gate transform
              ``lower_bound * sigmoid(exp(a_log) * (gate + dt_bias))``.
        beta: ``(total_tokens, HO)``.  Post-sigmoid float32, or io-dtype
              logits when ``use_beta_sigmoid``
        do: ``(total_tokens, HO, DV)`` io dtype
        state_checkpoints: ``(total_checkpoints, HO, DV, DK)`` io dtype (VK, k
            contiguous), the PLAIN per-chunk checkpoint series: sequence-local
            entry ``c`` is the state ENTERING chunk c of sequence b, so row 0
            is the initial state (or zeros when the forward had none)
        dq/dk/dv: io dtype at ``HO = max(HQ, HV)`` heads, pre-allocated
        dgate: ``(total_tokens, HO, DK)`` fp32 (dL/d ln alpha), pre-allocated.
            With ``safe_gate`` this stays the gradient wrt the TRANSFORMED
            log-decay; a host-side helper converts to d(raw gate) afterward
        dbeta: ``(total_tokens, HO)`` fp32, or io dtype with
            ``use_beta_sigmoid``, pre-allocated.  Gradient wrt the post-sigmoid
            beta, or wrt the raw logits under ``use_beta_sigmoid`` (the kernel
            folds the sigmoid derivative into its own dbeta write)
        cu_seqlens: ``(num_seqs + 1,)`` int32
        scale: attention scale factor
        use_initial_state: the forward ran with an initial state, so chunk 0
            has an entering state to load from ``state_checkpoints`` row 0
        d_initial_state: fp32 ``(num_seqs, HO, DV, DK)`` OUT (dL/d initial state), or None
        d_final_state: fp32 ``(num_seqs, HO, DV, DK)`` IN (dL/d final state)
        use_qk_l2norm_in_kernel: q/k arrive raw; the kernel normalizes for the
            recompute math and chains the L2-norm backward into dq/dk
        safe_gate: interpret ``gate`` through the safe-gate transform
        a_log: ``(HO,)`` float32, safe-gate per-head log-amplitude (None = 0)
        dt_bias: ``(HO, DK)`` float32, safe-gate channel bias (None = 0)
        use_beta_sigmoid: ``beta`` holds logits; sigmoid in-kernel
        work_items/work_count: split-K table (``common/split_k.py``, REQUIRED;
            an uncut table row is the whole (b, h) sequence); each item
            computes chunks ``[wstart, cend)`` backward and writes
            gradients only for ``[wstart, wend)``
        sched_ctr: ``(2,)`` int32 zeroed scratch enabling the dynamic
            (work-stealing) tile scheduler
        tensormap_workspace: ``tensormap_workspace_bytes(module, B)`` bytes,
            128-byte aligned, for the per-(batch, head) TMA-descriptor
            arrays (tail chunks clip/zero-fill in hardware)
    """
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("gate", gate),
        ("do", do),
        ("state_checkpoints", state_checkpoints),
        ("dq", dq),
        ("dk", dk),
        ("dv", dv),
        ("dgate", dgate),
    ):
        validate_tma_tensor(name, tensor)
    for name, tensor in (("d_initial_state", d_initial_state), ("d_final_state", d_final_state)):
        if tensor is not None:
            validate_tma_tensor(name, tensor)
    validate_cu_seqlens(cu_seqlens, assumed_align=8)
    if tensormap_workspace.data_ptr() % 128:
        raise ValueError("tensormap_workspace data pointer must be 128-byte aligned")

    tokens, HQ, d_k = q.shape
    HK, HV = k.shape[1], v.shape[1]
    HO = max(HQ, HV)
    if d_k != 128 or k.shape != (tokens, HK, 128) or v.shape != (tokens, HV, 128):
        raise ValueError("q, k, and v must have shape (T, H, 128)")
    if gate.shape != (tokens, HO, 128) or beta.shape != (tokens, HO):
        raise ValueError("gate and beta must match the output head count and token extent")
    if do.shape != (tokens, HO, 128):
        raise ValueError("do must have shape (T, HO, 128)")
    expected_grad_shapes = ((dq, q.shape), (dk, k.shape), (dv, v.shape), (dgate, gate.shape))
    if any(tensor.shape != expected for tensor, expected in expected_grad_shapes):
        raise ValueError("dq, dk, dv, and dgate must match their corresponding inputs")
    if dbeta.shape != beta.shape:
        raise ValueError("dbeta must match beta")
    use_dstate_in = d_final_state is not None
    use_dstate0 = d_initial_state is not None
    if work_items is None or work_count is None:
        raise ValueError("work_items/work_count are required (the split-table stage builds them for every launch)")
    dyn_sched = sched_ctr is not None
    run_order = order_in_prologue
    order_gen = order_in_prologue and work_item_scratch is None
    has_sched = run_order
    if run_order and sched_all is None:
        raise ValueError("order in the prologue requires sched_all (the prologue zeroes both consumers' sched rings)")
    required_checkpoint_rows = checkpoint_capacity_bound(
        q.shape[0], cu_seqlens.shape[0] - 1, CFG.B_T
    )
    if state_checkpoints.shape[0] < required_checkpoint_rows:
        raise ValueError(
            f"state_checkpoints requires at least {required_checkpoint_rows} rows, "
            f"got {state_checkpoints.shape[0]}"
        )
    if str(state_checkpoints.dtype).split(".")[-1] != str(q.dtype).split(".")[-1]:
        raise ValueError(f"state_checkpoints dtype must match the io dtype: got {state_checkpoints.dtype} with io {q.dtype}")
    for name, hh in (("HQ", HQ), ("HK", HK), ("HV", HV)):
        if HO % hh != 0:
            raise ValueError(f"{name}={hh} must divide {HO}")
    B = cu_seqlens.shape[0] - 1
    gate_scale_log2 = gate_lower_bound * LOG2_E

    if safe_gate and (a_log is None or dt_bias is None):
        raise ValueError("safe_gate requires a_log and dt_bias")
    if dt_bias is not None:
        validate_tma_tensor("dt_bias", dt_bias)
    if not safe_gate:
        a_log = None
        dt_bias = None

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
        a_log,
        dt_bias,
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
        HQ,
        HK,
        HV,
        use_dstate_in,
        use_dstate0,
        use_qk_l2norm_in_kernel,
        safe_gate,
        gate_lower_bound,
        use_beta_sigmoid,
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
            use_dstate_in=use_dstate_in,
            use_dstate0=use_dstate0,
            l2norm=use_qk_l2norm_in_kernel,
            safe_gate=safe_gate,
            gate_scale_log2=gate_scale_log2,
            beta_sigmoid=use_beta_sigmoid,
            use_initial_state=use_initial_state,
            q_ratio=HO // HQ,
            k_ratio=HO // HK,
            v_ratio=HO // HV,
            n_heads_out=HO,
            max_active_clusters=num_sm,
            dyn_sched=dyn_sched,
        )
        op = KdaBpropOp(cfg, use_int64_offsets)
        (
            tokens,
            checkpoint_rows,
            sequence_entries,
            sequences,
            work_rows,
            sched_entries,
            workspace_words,
        ) = (sym_int() for _ in range(7))

        tma_tensor = partial(
            make_strided_signature_tensor,
            assumed_align=16,
            use_int64_offsets=use_int64_offsets,
        )
        beta_dtype = io_dtype if use_beta_sigmoid else cutlass.Float32
        cache["compiled"] = compile_tvm_ffi(
            op,
            make_compact_signature_tensor(
                cutlass.Float32,
                (HO,),
                assumed_align=4,
            )
            if safe_gate
            else None,
            tma_tensor(cutlass.Float32, (HO, 128)) if safe_gate else None,
            make_strided_signature_tensor(
                beta_dtype,
                (tokens, HO),
                assumed_align=beta_dtype.width // 8,
                use_int64_offsets=use_int64_offsets,
            ),
            tma_tensor(io_dtype, (checkpoint_rows, HO, 128, 128)),
            tma_tensor(cutlass.Float32, (tokens, HO, 128)),
            make_strided_signature_tensor(
                beta_dtype,
                (tokens, HO),
                assumed_align=beta_dtype.width // 8,
                use_int64_offsets=use_int64_offsets,
            ),
            make_cu_seqlens_signature(sequence_entries),
            tma_tensor(cutlass.Float32, (sequences, HO, 128, 128)) if use_dstate0 else None,
            tma_tensor(cutlass.Float32, (sequences, HO, 128, 128)) if use_dstate_in else None,
            make_work_items_signature(work_rows),
            make_counter_signature(),
            make_counter_signature(sched_entries) if dyn_sched else None,
            make_workspace_signature(workspace_words),
            cutlass.Float32(scale),
        )

    if "prologue" not in cache:
        tokens, checkpoint_rows, sequence_entries, work_rows, sched_entries, workspace_words = (
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
            run_order,
            order_gen,
            has_sched,
            tma_tensor(io_dtype, (tokens, HQ, 128)),
            tma_tensor(io_dtype, (tokens, HK, 128)),
            tma_tensor(io_dtype, (tokens, HV, 128)),
            tma_tensor(cutlass.Float32, (tokens, HO, 128)),
            tma_tensor(io_dtype, (tokens, HO, 128)),
            tma_tensor(io_dtype, (tokens, HQ, 128)),
            tma_tensor(io_dtype, (tokens, HK, 128)),
            tma_tensor(io_dtype, (tokens, HV, 128)),
            tma_tensor(cutlass.Float32, (tokens, HO, 128)),
            tma_tensor(io_dtype, (checkpoint_rows, HO, 128, 128)),
            make_cu_seqlens_signature(sequence_entries),
            work_items_signature if run_order and not order_gen else None,
            make_counter_signature(),
            work_items_signature,
            make_counter_signature(sched_entries) if has_sched else None,
            make_workspace_signature(workspace_words),
            name=(
                f"kda_mega_bprop_prologue_{io_dtype.__name__.lower()}"
                f"_hq{HQ}_hk{HK}_hv{HV}_ho{HO}_r{int(run_order)}"
                f"o{int(order_gen)}d{int(dyn_sched)}_i64{int(use_int64_offsets)}"
            ),
        )
    cache["prologue"](
        q,
        k,
        v,
        gate,
        do,
        dq,
        dk,
        dv,
        dgate,
        state_checkpoints,
        cu_seqlens,
        work_item_scratch if run_order else None,
        work_count,
        work_items,
        sched_all if run_order else None,
        tensormap_workspace,
    )
    cache["compiled"](
        a_log,
        dt_bias,
        beta,
        state_checkpoints,
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
