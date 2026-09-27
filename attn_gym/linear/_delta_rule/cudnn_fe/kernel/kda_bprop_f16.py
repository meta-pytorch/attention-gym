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
# attn_gym.linear._delta_rule.cudnn_fe; removed safe_gate, a_log/dt_bias, beta_sigmoid,
# allow_neg_eigval and l2norm knobs.
# Restyled register tensors, shared storage, and shared pointers.

"""
Chunked Kimi Delta Attention (KDA) BPROP kernel for SM100 / SM103 / SM107 (Cutlass primitives): the
BT = 16 schedule with a per-key-channel decay, mirroring the prefill's 16-warp specialization.

Algorithm overview (per chunk c, iterated c = NT-1 .. 0):
  Inputs : Q[BT,DK], K[BT,DK], V[BT,DV], dO[BT,DV], S_c[DK,DV] (the forward state ENTERING chunk c,
           from the checkpoint series), Gate[BT,DK] (per-channel gate), Beta[BT] (scalar LR)
  State  : dstate[DK,DV]  (state gradient, held in TMEM, accumulated backward)

  Preprocessing (compute group 0): g, K decay, K inv, Q decay as in the prefill (no K restore), the
  decay scale exp2(g[BT-1,:]), the raw Q/K rows into a TMEM ring for compute group 2, the state
  entry copied SMEM -> TMEM f16. Register MMA (warp 12): KK = K
  decay @ K inv^T; L = Beta * tril(KK, -1); T_inv by the blockwise inverse; dM = dY @ U^T; dM
  strict = Beta row . strict(dM); the dBeta M-term row sums. Register MMA (warp 15): A = tril(Q
  decay @ K inv^T, 0); dA = tril(dO @ U^T, 0); the dA diagonal for compute group 2.

  MMA order (tcgen05, warp 13; (S) = SMEM operand, (T) = TMEM operand, as the MMA warp's signposts
  read):
  k state       : k state = state(S) @ K decay^T
  dQ inter      : dQ inter = state(T) @ dO^T
  dU inter      : dU inter = dstate input(T) @ K inv
  dU intra      : dU intra += dO^T(S) @ A
  dstate Q-term : dstate Q-term += dO^T(S) @ Q decay
  U             : U = Y(T) @ T^-1,  Y = Beta * (V - k state)
  dY            : dY = dU(T) @ T^-1
  dK restore    : dK restore part = dstate(S) @ U^T
  dstate K-term : dstate K-term += -Beta.dY(T) @ K decay
  dK inv        : dK inv part = scale.Q decay^T(S) @ dA
  dQ attn       : dQ attn += K inv^T(S) @ dA^T
  dK decay      : dK decay part = state(T) @ (Beta.dY)^T
  dK inv        : dK inv part += K decay^T(S) @ -dM strict
  dK decay      : dK decay part += K inv^T(S) @ dM strict^T

  Epilogue (compute group 2, one key channel per thread): dQ = eG.scale * dQ acc; dK = the restore
  / inv / decay parts with their exp2(+-g) scales and the dA diagonal; the L2-norm backward row
  projection; dGate per channel including the g_last terms, reverse-cumsummed in registers.
  Compute group 1: dBeta = sum over v of dY.(V - k state) + the M-term, dV = Beta.dY.  The epilogue
  warp stores dQ / dK / dV / dGate through a one-behind TMA ladder.

SMEM layout (stage counts live in kda_bprop_config.py; sizes at DK = DV = 128, bf16 io, fp32 Gate):
  Buffer                       Size (B)  Stages
  Q / K (raw)                  2 x 4096       2
  V / dO (raw)                 2 x 4096       2
  Gate (raw)                       8192       2
  K decay / K inv / Q decay    3 x 4096       2
  decay scale                       512       2
  T_inv / A / dA / dM (intermediate) 2560     2    <-- five BT x BT tiles per stage
  state (checkpoint entry)        32768       1    <-- io dtype [DK,DV], TMA-loaded
  dstate f16 restage              32768       1
  U / dY                       2 x 4096       1
  dQ / dK store staging        2 x 4096       1
  dGate store staging              8192       1    <-- Gate dtype
  dV store staging                 4096       2
  fp32 side vectors             ~18 KB        -    <-- Beta ring, Q/K inverse norms, reductions, the dK projection
                                                       scratch (2 x BT x 128 floats), Beta M-term,
                                                       dA diagonal ring

TMEM layout (512 columns allocated):
  Buffer                  Cols
  dstate acc              128     <-- DKxDV fp32
  dstate input             64     <-- f16 packed
  state input             128     <-- f16 packed state entry
  K*state / dY acc         16     <-- one slot, K*state then dY
  U acc                    16
  dU acc                   16
  dQ acc                   16
  dK decay / inv / restore acc  3 x 16
  Y / -Beta.dY input        8     <-- one f16 slot, Y then -Beta.dY
  dU input                  8
  raw Q / raw K rings   2 x 32    <-- 4-stage f16 rings for compute group 2

Warp assignments (16 warps = 512 threads):
  warps 0-3     : compute group 0 - Gate prefix scan, decay operands, raw Q/K ring, state copy
  warps 4-7     : compute group 1 - dstate seed, Y / dU / -Beta.dY stagings, U / dY reads, dBeta, dV, dstate capture
  warps 8-11    : compute group 2 - dQ / dK assembly, L2-norm projection, dGate assembly + reverse cumsum, store staging
  warp  12      : register-MMA warp - KK, T_inv, dM and the dBeta M-term
  warp  13      : MMA warp       - every tcgen05 GEMM; TMEM lifecycle
  warp  14      : TMA load warp  - loads Q, K, Gate, dO, V and the entering state
  warp  15      : epilogue warp  - register-MMA A / dA tiles, dQ / dK / dV / dGate TMA stores
"""

from dataclasses import dataclass, replace
from typing import NamedTuple

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.experimental.primitives as nvvm
from cutlass import cute
from cutlass.experimental import cuda

from attn_gym._backends.cute.compat import SmemAllocator
from attn_gym._backends.cute.utils import (
    requires_int64_abi,  # noqa: F401  (module-level ABI selector)
)

from ..common.blockwise_inverse import invert_unit_lower_16x16_fragments
from ..common.split_k import (
    ORDER_CAPACITY,
    ORDER_ELEMENTS,
    ORDER_THREADS,
    WORK_ITEM_DSTATE_DST,
    decode_work_item,
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
from ..tile_dsl.mma import desc_opaque, mma_ss, mma_step, mma_ts_step
from ..tile_dsl.pointwise import (
    beta_residual_f16x2,
    fadd2,
    ffma2,
    fmul2,
    fp32_to_fp16,
    opaque_f32_zero,
    pack_u16x2,
)
from ..tile_dsl.swizzle import swizzle_xor_32b, swizzle_xor_128b
from ..tile_dsl.tma import (
    tma_load_tile,
    tma_store_commit,
    tma_store_tile,
    tma_store_wait,
    tma_tensormap_acquire,
)
from .kda_bprop_config import CFG

USE_PDL = True

LOG2_E: float = 1.4426950408889634
DEFAULT_GATE_LOWER_BOUND: float = -5.0


class KdaBpropBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_raw_ready: MBarrier
    mb_raw_done: MBarrier
    mb_do_ready: MBarrier
    mb_do_done: MBarrier
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
    mb_state_input_ready: MBarrier
    mb_state_input_done: MBarrier
    mb_state_input_cg2_done: MBarrier
    mb_y_input_ready: MBarrier
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
    mb_da_ready: MBarrier
    mb_da_done: MBarrier
    mb_da_diag_ready: MBarrier
    mb_da_diag_done: MBarrier
    mb_dm_ready: MBarrier
    mb_dm_done: MBarrier
    mb_u_smem_ready: MBarrier
    mb_dy_smem_ready: MBarrier
    mb_dbeta_m_ready: MBarrier

    mb_dstate_acc_ready: MBarrier
    mb_dstate_input_ready: MBarrier
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

    mb_scheduler_ready: MBarrier
    mb_scheduler_done: MBarrier


def make_bars(cfg) -> KdaBpropBars:
    """KdaBpropBars constructor."""

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=8)

    WARP = cfg.threads_per_warp
    CG0 = len(cfg.compute_group_0_warp_ids) * WARP
    CG2 = len(cfg.compute_group_2_warp_ids) * WARP
    CG1 = len(cfg.compute_group_1_warp_ids) * WARP
    MMA = 1

    return KdaBpropBars(
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
            init_count=CG0 + CG2 + CG1,
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
            init_count=WARP,
            producer=Producer.THREAD,
        ),
        mb_state_ready=MBarrier(
            alloc(cfg.smem_state_stages),
            try_wait=True,
            stages=cfg.smem_state_stages,
            init_count=1,
            producer=Producer.TMA_LOAD,
        ),
        mb_state_done=MBarrier(
            alloc(cfg.smem_state_stages),
            try_wait=True,
            stages=cfg.smem_state_stages,
            init_count=MMA,
            producer=Producer.MMA_COMMIT,
        ),
        mb_state_cg0_done=MBarrier(
            alloc(cfg.smem_state_stages),
            try_wait=True,
            stages=cfg.smem_state_stages,
            init_count=CG0,
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
        mb_state_k_acc_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=MMA, producer=Producer.MMA_COMMIT
        ),
        mb_du_acc_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=MMA, producer=Producer.MMA_COMMIT
        ),
        mb_u_acc_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=MMA, producer=Producer.MMA_COMMIT
        ),
        mb_dy_acc_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=MMA, producer=Producer.MMA_COMMIT
        ),
        mb_dq_acc_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=MMA, producer=Producer.MMA_COMMIT
        ),
        mb_dk_decay_part_acc_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=MMA, producer=Producer.MMA_COMMIT
        ),
        mb_dk_inv_part_acc_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=MMA, producer=Producer.MMA_COMMIT
        ),
        mb_dk_restore_part_acc_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=MMA, producer=Producer.MMA_COMMIT
        ),
        mb_dqk_acc_done=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG2, producer=Producer.THREAD
        ),
        mb_qk_raw_ready=MBarrier(
            alloc(cfg.tmem_qk_raw_stages),
            try_wait=True,
            stages=cfg.tmem_qk_raw_stages,
            init_count=CG0,
            producer=Producer.THREAD,
        ),
        mb_qk_raw_done=MBarrier(
            alloc(cfg.tmem_qk_raw_stages),
            try_wait=True,
            stages=cfg.tmem_qk_raw_stages,
            init_count=CG2,
            producer=Producer.THREAD,
        ),
        mb_state_input_ready=MBarrier(
            alloc(2), try_wait=True, stages=2, init_count=CG0, producer=Producer.THREAD
        ),
        mb_state_input_done=MBarrier(
            alloc(2), try_wait=True, stages=2, init_count=MMA, producer=Producer.MMA_COMMIT
        ),
        mb_state_input_cg2_done=MBarrier(
            alloc(2), try_wait=True, stages=2, init_count=CG2, producer=Producer.THREAD
        ),
        mb_y_input_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1, producer=Producer.THREAD
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
        mb_da_ready=MBarrier(
            alloc(cfg.smem_intermediate_stages),
            try_wait=True,
            stages=cfg.smem_intermediate_stages,
            init_count=WARP,
            producer=Producer.THREAD,
        ),
        mb_da_done=MBarrier(
            alloc(cfg.smem_intermediate_stages),
            try_wait=True,
            stages=cfg.smem_intermediate_stages,
            init_count=MMA,
            producer=Producer.MMA_COMMIT,
        ),
        mb_da_diag_ready=MBarrier(
            alloc(cfg.smem_da_diag_stages),
            try_wait=True,
            stages=cfg.smem_da_diag_stages,
            init_count=WARP,
            producer=Producer.THREAD,
        ),
        mb_da_diag_done=MBarrier(
            alloc(cfg.smem_da_diag_stages),
            try_wait=True,
            stages=cfg.smem_da_diag_stages,
            init_count=CG2,
            producer=Producer.THREAD,
        ),
        mb_dm_ready=MBarrier(
            alloc(cfg.smem_intermediate_stages),
            try_wait=True,
            stages=cfg.smem_intermediate_stages,
            init_count=WARP,
            producer=Producer.THREAD,
        ),
        mb_dm_done=MBarrier(
            alloc(cfg.smem_intermediate_stages),
            try_wait=True,
            stages=cfg.smem_intermediate_stages,
            init_count=MMA,
            producer=Producer.MMA_COMMIT,
        ),
        mb_u_smem_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1, producer=Producer.THREAD
        ),
        mb_dy_smem_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1, producer=Producer.THREAD
        ),
        mb_dbeta_m_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=WARP, producer=Producer.THREAD
        ),
        mb_dstate_acc_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=MMA, producer=Producer.MMA_COMMIT
        ),
        mb_dstate_input_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1, producer=Producer.THREAD
        ),
        mb_dstate_smem_ready=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1, producer=Producer.THREAD
        ),
        mb_dstate_smem_done=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=MMA, producer=Producer.MMA_COMMIT
        ),
        mb_dstate_smem_cg2_done=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG2, producer=Producer.THREAD
        ),
        mb_dq_tmastg_ready=MBarrier(
            alloc(cfg.smem_dq_stages),
            try_wait=True,
            stages=cfg.smem_dq_stages,
            init_count=CG2,
            producer=Producer.THREAD,
        ),
        mb_dq_tmastg_done=MBarrier(
            alloc(cfg.smem_dq_stages),
            try_wait=True,
            stages=cfg.smem_dq_stages,
            init_count=WARP,
            producer=Producer.THREAD,
        ),
        mb_dk_tmastg_ready=MBarrier(
            alloc(cfg.smem_dk_stages),
            try_wait=True,
            stages=cfg.smem_dk_stages,
            init_count=CG2,
            producer=Producer.THREAD,
        ),
        mb_dk_tmastg_done=MBarrier(
            alloc(cfg.smem_dk_stages),
            try_wait=True,
            stages=cfg.smem_dk_stages,
            init_count=WARP,
            producer=Producer.THREAD,
        ),
        mb_dv_tmastg_ready=MBarrier(
            alloc(cfg.smem_dv_stages),
            try_wait=True,
            stages=cfg.smem_dv_stages,
            init_count=CG1,
            producer=Producer.THREAD,
        ),
        mb_dv_tmastg_done=MBarrier(
            alloc(cfg.smem_dv_stages),
            try_wait=True,
            stages=cfg.smem_dv_stages,
            init_count=WARP,
            producer=Producer.THREAD,
        ),
        mb_dgate_tmastg_ready=MBarrier(
            alloc(cfg.smem_dgate_stages),
            try_wait=True,
            stages=cfg.smem_dgate_stages,
            init_count=CG2,
            producer=Producer.THREAD,
        ),
        mb_dgate_tmastg_done=MBarrier(
            alloc(cfg.smem_dgate_stages),
            try_wait=True,
            stages=cfg.smem_dgate_stages,
            init_count=WARP,
            producer=Producer.THREAD,
        ),
        mb_dstate0_acc_stored=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1, producer=Producer.THREAD
        ),
        mb_tmem_done=MBarrier(
            alloc(1), try_wait=True, stages=1, init_count=CG1 + CG2, producer=Producer.THREAD
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
    sDo_raw,
    sU_raw,
    sIntermediate_raw,
    sDaDiag_raw,
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
    tril_strict_mask = cutlass.Int32(0)
    for accum_idx in cutlass.range_constexpr(8):
        row_coord = row_hi if cutlass.const_expr(accum_idx % 4 >= 2) else row_lo
        col_coord = (accum_idx // 4) * 8 + 2 * (lane_idx % 4)
        if cutlass.const_expr(accum_idx % 2 == 1):
            col_coord = col_coord + cutlass.Int32(1)
        tril_incl_mask = tril_incl_mask | (
            cutlass.Int32(1 << accum_idx) if row_coord >= col_coord else cutlass.Int32(0)
        )
        tril_strict_mask = tril_strict_mask | (
            cutlass.Int32(1 << accum_idx) if row_coord > col_coord else cutlass.Int32(0)
        )
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
    dgate_box_elems = cutlass.const_expr(128 // (cfg.gate_dtype.width // 8))
    sDgate_tma = SmemTile(
        base=sDgate_raw,
        elems_per_stage=(cfg.b_t * cfg.d_k),
        stages=cfg.smem_dgate_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // dgate_box_elems),
        tma_granu_elems=dgate_box_elems,
        tma_subtile_stride_elems=cfg.b_t * dgate_box_elems,
    )
    dq_index = PipelineState.start(phase=0)
    dk_index = PipelineState.start(phase=0)
    dv_index = PipelineState.start(phase=0)
    dgate_index = PipelineState.start(phase=0)
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
            _compute_start,
            compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
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
        num_compute_chunks = compute_end - write_start
        pend_start = cutlass.Int32(0)
        pend_writes = cutlass.Boolean(False)
        for rev_idx in cutlass.range(num_compute_chunks + 1, unroll=1):
            chunk_idx = compute_end - cutlass.Int32(1) - rev_idx
            chunk_start = chunk_idx * cfg.b_t
            writes = chunk_idx < write_end
            chunk_serial = chunk_serial_base + rev_idx
            decay_stage = chunk_serial % cfg.smem_decay_stages
            intermediate_stage = chunk_serial % cfg.smem_intermediate_stages
            raw_stage = raw_index.idx
            sK_inv_ptr = smem_data_ptr(sK_inv_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sQ_decay_ptr = smem_data_ptr(sQ_decay_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sDo_ptr = smem_data_ptr(sDo_raw) + raw_stage * (cfg.d_v * cfg.b_t)
            sIntermediate_ptr = smem_data_ptr(sIntermediate_raw) + intermediate_stage * (
                cfg.intermediate_tiles * cfg.b_t * cfg.b_t
            )

            if rev_idx < num_compute_chunks:
                # ---- A = tril(Q decay @ K inv^T, 0) ----------------------------------
                bars.mb_a_done[intermediate_stage].wait(
                    ((chunk_serial // cfg.smem_intermediate_stages) + 1) % 2
                )
                bars.mb_q_decay_ready[decay_stage].wait(
                    (chunk_serial // cfg.smem_decay_stages) % 2
                )
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
                        a_acc[accum_idx]
                        if (tril_incl_mask >> accum_idx) & 1
                        else cutlass.Float32(0.0)
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

                # ---- dA = tril(dO @ U^T, 0) ------------------------------------------
                bars.mb_u_smem_ready.wait(u_index.phase)
                u_index = advance(u_index, 1)
                da_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
                for accum_idx in cutlass.range_constexpr(8):
                    da_acc[accum_idx] = cutlass.Float32(0.0)
                for i in cutlass.range_constexpr(cfg.d_v // 16):
                    a_col = i * 16 + a_col_offset
                    a_seg = a_col // 64
                    do_frag = nvvm.ldmatrix(
                        sDo_ptr
                        + a_seg * (cfg.b_t * 64)
                        + a_row_coord * 64
                        + swizzle_xor_128b(a_row_coord, a_col - a_seg * 64, elem_bytes=2),
                        4,
                        nvvm.MMALayout.ROW,
                    )
                    b_col = i * 16 + b_col_offset
                    b_seg = b_col // 64
                    u_frag = nvvm.ldmatrix(
                        smem_data_ptr(sU_raw)
                        + b_seg * (cfg.b_t * 64)
                        + b_row_coord * 64
                        + swizzle_xor_128b(b_row_coord, b_col - b_seg * 64, elem_bytes=2),
                        4,
                        nvvm.MMALayout.ROW,
                    )
                    mma_step(
                        da_acc,
                        (do_frag[0], do_frag[1], do_frag[2], do_frag[3]),
                        (u_frag[0], u_frag[1], u_frag[2], u_frag[3]),
                        k_step=0,
                        M=16,
                        N=16,
                        ab_dtype=cfg.io_dtype,
                    )
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_do_done[raw_stage].arrive()

                # ---- dA diagonal -> WG2 ----------------------------------------------
                da_diag_stage = chunk_serial % cfg.smem_da_diag_stages
                bars.mb_da_diag_done[da_diag_stage].wait(
                    ((chunk_serial // cfg.smem_da_diag_stages) + 1) % 2
                )
                for accum_idx in cutlass.range_constexpr(8):
                    diag_row = row_hi if cutlass.const_expr(accum_idx % 4 >= 2) else row_lo
                    diag_col = (accum_idx // 4) * 8 + 2 * (lane_idx % 4)
                    if cutlass.const_expr(accum_idx % 2 == 1):
                        diag_col = diag_col + cutlass.Int32(1)
                    if diag_row == diag_col:
                        sDaDiag_raw[da_diag_stage * cfg.b_t + diag_row] = da_acc[accum_idx]
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_da_diag_ready[da_diag_stage].arrive()
                for accum_idx in cutlass.range_constexpr(8):
                    da_acc[accum_idx] = (
                        da_acc[accum_idx]
                        if (tril_strict_mask >> accum_idx) & 1
                        else cutlass.Float32(0.0)
                    )
                bars.mb_da_done[intermediate_stage].wait(
                    ((chunk_serial // cfg.smem_intermediate_stages) + 1) % 2
                )
                nvvm.stmatrix(
                    sIntermediate_ptr + 2 * (cfg.b_t * cfg.b_t) + intermediate_idx,
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

            # ---- dQ/dK/dGate/dV: one-behind store ladder -----------------------------
            if rev_idx > 0:
                bars.mb_dq_tmastg_ready[dq_index.idx].wait(dq_index.phase)
                if pend_writes:
                    desc_dq_slot = (desc_dq_base + slot).tospace(cutlass.AddressSpace.generic)
                    dq_slice = tma_slice_runtime_desc(
                        desc_dq_slot, cutlass.Int32(0), head_o, pend_start
                    )
                    tma_store_tile(sDq_tma[dq_index.idx], dq_slice)
                    tma_store_commit()
                bars.mb_dk_tmastg_ready[dk_index.idx].wait(dk_index.phase)
                if pend_writes:
                    desc_dk_slot = (desc_dk_base + slot).tospace(cutlass.AddressSpace.generic)
                    dk_slice = tma_slice_runtime_desc(
                        desc_dk_slot, cutlass.Int32(0), head_o, pend_start
                    )
                    tma_store_tile(sDk_tma[dk_index.idx], dk_slice)
                    tma_store_commit()
                bars.mb_dgate_tmastg_ready[dgate_index.idx].wait(dgate_index.phase)
                if pend_writes:
                    desc_dgate_slot = (desc_dgate_base + slot).tospace(
                        cutlass.AddressSpace.generic
                    )
                    dgate_slice = tma_slice_runtime_desc(
                        desc_dgate_slot, cutlass.Int32(0), head_o, pend_start
                    )
                    tma_store_tile(sDgate_tma[dgate_index.idx], dgate_slice)
                    tma_store_commit()
                bars.mb_dv_tmastg_ready[dv_index.idx].wait(dv_index.phase)
                if pend_writes:
                    desc_dv_slot = (desc_dv_base + slot).tospace(cutlass.AddressSpace.generic)
                    dv_slice = tma_slice_runtime_desc(
                        desc_dv_slot, cutlass.Int32(0), head_o, pend_start
                    )
                    tma_store_tile(sDv_tma[dv_index.idx], dv_slice)
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
    sU_raw,
    sDy_raw,
    sIntermediate_raw,
    sBeta_raw,
    sBetaM_raw,
    bars,
) -> None:
    """Register-MMA warp role (warp 12): the blockwise T_inv and dM register MMAs
    plus the dBeta M-term row sums, in chunk order."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    elect_one = nvvm.elect_sync()
    sdy_index = PipelineState.start(phase=0)

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

            # ---- dM = dY @ U^T -------------------------------------------------------
            bars.mb_dm_done[intermediate_stage].wait(
                ((chunk_serial // cfg.smem_intermediate_stages) + 1) % 2
            )
            bars.mb_dy_smem_ready.wait(sdy_index.phase)
            sdy_index = advance(sdy_index, 1)
            dm_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                dm_acc[accum_idx] = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(cfg.d_v // 16):
                a_col = i * 16 + a_col_offset
                a_seg = a_col // 64
                dy_frag = nvvm.ldmatrix(
                    smem_data_ptr(sDy_raw)
                    + a_seg * (cfg.b_t * 64)
                    + a_row_coord * 64
                    + swizzle_xor_128b(a_row_coord, a_col - a_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                b_col = i * 16 + b_col_offset
                b_seg = b_col // 64
                u_frag = nvvm.ldmatrix(
                    smem_data_ptr(sU_raw)
                    + b_seg * (cfg.b_t * 64)
                    + b_row_coord * 64
                    + swizzle_xor_128b(b_row_coord, b_col - b_seg * 64, elem_bytes=2),
                    4,
                    nvvm.MMALayout.ROW,
                )
                mma_step(
                    dm_acc,
                    (dy_frag[0], dy_frag[1], dy_frag[2], dy_frag[3]),
                    (u_frag[0], u_frag[1], u_frag[2], u_frag[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )

            # ---- dM strict = Beta row . strict(dM) -----------------------------------
            dm_strict_regs = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                beta_scale = beta_lo if accum_idx % 4 < 2 else beta_hi
                val = (
                    dm_acc[accum_idx] * beta_scale
                    if (tril_strict_mask >> accum_idx) & 1
                    else cutlass.Float32(0.0)
                )
                dm_strict_regs[accum_idx] = val
            w0 = fp32_to_fp16(dm_strict_regs[0], dm_strict_regs[1], dtype=cfg.io_dtype)
            w1 = fp32_to_fp16(dm_strict_regs[2], dm_strict_regs[3], dtype=cfg.io_dtype)
            w2 = fp32_to_fp16(dm_strict_regs[4], dm_strict_regs[5], dtype=cfg.io_dtype)
            w3 = fp32_to_fp16(dm_strict_regs[6], dm_strict_regs[7], dtype=cfg.io_dtype)
            nvvm.stmatrix(
                sIntermediate_ptr + 3 * (cfg.b_t * cfg.b_t) + intermediate_idx,
                [w0, w1, w2, w3],
                nvvm.MMALayout.ROW,
                shape=nvvm.StoreShape.M8N8,
            )
            nw0 = fp32_to_fp16(-dm_strict_regs[0], -dm_strict_regs[1], dtype=cfg.io_dtype)
            nw1 = fp32_to_fp16(-dm_strict_regs[2], -dm_strict_regs[3], dtype=cfg.io_dtype)
            nw2 = fp32_to_fp16(-dm_strict_regs[4], -dm_strict_regs[5], dtype=cfg.io_dtype)
            nw3 = fp32_to_fp16(-dm_strict_regs[6], -dm_strict_regs[7], dtype=cfg.io_dtype)
            nvvm.stmatrix(
                sIntermediate_ptr + 4 * (cfg.b_t * cfg.b_t) + intermediate_idx,
                [nw0, nw1, nw2, nw3],
                nvvm.MMALayout.ROW,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dm_ready[intermediate_stage].arrive()

            # ---- M-term: bsum = sum strict(dM . KK) ----------------------------------
            bsum_lo = cutlass.Float32(0.0)
            bsum_hi = cutlass.Float32(0.0)
            for accum_idx in cutlass.range_constexpr(8):
                e = (
                    dm_acc[accum_idx] * kk_acc[accum_idx]
                    if (tril_strict_mask >> accum_idx) & 1
                    else cutlass.Float32(0.0)
                )
                if cutlass.const_expr(accum_idx % 4 < 2):
                    bsum_lo = bsum_lo + e
                else:
                    bsum_hi = bsum_hi + e
            bsum_lo = bsum_lo + cutlass.Float32(
                nvvm.shfl_sync(0xFFFFFFFF, bsum_lo, 1, 31, kind=nvvm.Shfl.BFLY)
            )
            bsum_lo = bsum_lo + cutlass.Float32(
                nvvm.shfl_sync(0xFFFFFFFF, bsum_lo, 2, 31, kind=nvvm.Shfl.BFLY)
            )
            bsum_hi = bsum_hi + cutlass.Float32(
                nvvm.shfl_sync(0xFFFFFFFF, bsum_hi, 1, 31, kind=nvvm.Shfl.BFLY)
            )
            bsum_hi = bsum_hi + cutlass.Float32(
                nvvm.shfl_sync(0xFFFFFFFF, bsum_hi, 2, 31, kind=nvvm.Shfl.BFLY)
            )
            if lane_idx % 4 == 0:
                sBetaM_raw[row_lo] = -bsum_lo
                sBetaM_raw[row_hi] = -bsum_hi
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dbeta_m_ready.arrive()
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
    sState_trans,
    sState,
    sK_decay,
    sK_inv,
    sK_inv_trans,
    sDo,
    sDo_trans,
    sQ_decay_trans,
    sK_decay_trans,
    sU,
    sDv,
    sDstate_trans,
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
    instruction_descriptor_state_k = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
    )
    bmm_state_k_decay_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        atranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_state_k,
        kind=nvvm.Tcgen05MMAKind.F16,
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
    bmm_y_t_inv_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_mv_nt,
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
    instruction_descriptor_dstate = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
    )
    bmm_state_dy_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.d_v,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=instruction_descriptor_dstate,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_state_do_desc = bmm_state_dy_desc
    instruction_descriptor_dstate_at = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
    )
    bmm_dstate_u_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.d_v,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        atranspose=True,
        cta_group=1,
        idesc=instruction_descriptor_dstate_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
    )
    instruction_descriptor_dgp_at = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
    )
    bmm_k_inv_dm_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=False,
        atranspose=True,
        cta_group=1,
        idesc=instruction_descriptor_dgp_at,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_k_inv_da_desc = bmm_k_inv_dm_desc
    instruction_descriptor_dgp_at_t = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_k,
        a_major=1,
        b_major=1,
    )
    bmm_k_decay_dm_desc = MmaDesc(
        M=cfg.d_k,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bytes_per_element,
        bpe_b=bytes_per_element,
        tile_k_hw=16,
        btranspose=True,
        atranspose=True,
        cta_group=1,
        idesc=instruction_descriptor_dgp_at_t,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_q_decay_da_desc = bmm_k_decay_dm_desc

    state_index = PipelineState.start(phase=0)
    y_input_index = PipelineState.start(phase=0)
    dstate_input_index = PipelineState.start(phase=0)
    du_input_index = PipelineState.start(phase=0)
    neg_beta_dy_index = PipelineState.start(phase=0)
    u_smem_index = PipelineState.start(phase=0)
    dstate_smem_index = PipelineState.start(phase=0)
    parts_done_index = PipelineState.start(phase=1)

    do_seg = (cfg.b_t * cfg.d_v * (cfg.io_dtype.width // 8)) >> 4
    op_seg = (cfg.b_t * cfg.d_k * (cfg.io_dtype.width // 8)) >> 4
    intermediate_seg = (
        cfg.intermediate_tiles * cfg.b_t * cfg.b_t * (cfg.io_dtype.width // 8)
    ) >> 4
    intermediate_slot = (cfg.b_t * cfg.b_t * (cfg.io_dtype.width // 8)) >> 4
    dv_seg = (cfg.b_t * cfg.d_v * (cfg.io_dtype.width // 8)) >> 4
    d_do_trans0 = sDo_trans[0].desc()
    d_qd_trans0 = sQ_decay_trans[0].desc()
    d_kd_trans0 = sK_decay_trans[0].desc()
    d_ki_trans0 = sK_inv_trans[0].desc()
    d_int0 = sIntermediate[0].desc()
    d_kd0 = sK_decay[0].desc()
    d_do0 = sDo[0].desc()
    d_ki0 = sK_inv[0].desc()
    d_dv0 = sDv[0].desc()
    d_dstate_trans0 = sDstate_trans[0].desc()
    d_u0 = sU[0].desc()
    assert cfg.smem_state_stages == 1
    sState_trans[0].desc()
    d_state0 = sState[0].desc()
    dstate0_index = PipelineState.start(phase=0)

    chunk_serial_base = cutlass.Int32(0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
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
            d_ki_trans = d_ki_trans0 + decay_op_off
            d_int = d_int0 + intermediate_stage * intermediate_seg
            d_int_tinv = d_int + intermediate_slot
            d_int_da = d_int + 2 * intermediate_slot
            d_int_dm = d_int + 3 * intermediate_slot
            d_int_ndm = d_int + 4 * intermediate_slot
            d_state = desc_opaque(d_state0)
            d_dstate_trans = desc_opaque(d_dstate_trans0)
            d_u = desc_opaque(d_u0)
            chunk_idx = compute_end - cutlass.Int32(1) - rev_idx

            # ---- k state = state(S) @ K decay^T --------------------------------------
            bars.mb_k_decay_inv_ready[decay_stage].wait(decay_phase)
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_state_ready[state_index.idx].wait(state_index.phase)
                mma_ss(
                    bmm_state_k_decay_desc,
                    d_state,
                    d_kd0 + decay_op_off,
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
            bars.mb_state_input_ready[chunk_serial % 2].wait((chunk_serial // 2) % 2)
            bars.mb_do_ready[raw_stage_idx].wait((chunk_serial // cfg.smem_raw_stages) % 2)
            if chunk_idx >= FIRST_STATE_CHUNK:
                a_ptr = nvvm.make_tmem_ptr(
                    (
                        tmem_base
                        + cfg.tmem_state_input_offset
                        + (chunk_serial % 2) * (cfg.d_v // 2)
                    ),
                    cutlass.Int8,
                )
                b_desc = d_do0 + raw_stage_idx * do_seg
                c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dq_acc_offset), cutlass.Float32)
                for i in cutlass.range_constexpr(bmm_state_do_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_state_do_desc.sps_B):
                        mma_ts_step(
                            bmm_state_do_desc,
                            a_ptr.subview(
                                i * bmm_state_do_desc.sps_B * bmm_state_do_desc.tmem_advance_A
                            ),
                            b_desc + i * (bmm_state_do_desc.smem_subtile_B >> 4),
                            c_ptr,
                            k,
                            cutlass.Boolean(i + k > 0),
                        )

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

            # ---- U = Y(T) @ T^-1 -----------------------------------------------------
            bars.mb_t_inv_ready[intermediate_stage].wait(intermediate_phase)
            bars.mb_y_input_ready.wait(y_input_index.phase)
            y_input_index = advance(y_input_index, 1)
            a_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_y_input_offset), cutlass.Int8)
            b_desc = d_int_tinv
            c_ptr = nvvm.make_tmem_ptr((tmem_base + cfg.tmem_u_acc_offset), cutlass.Float32)
            for i in cutlass.range_constexpr(bmm_y_t_inv_desc.num_subtiles_B):
                for k in cutlass.range_constexpr(bmm_y_t_inv_desc.sps_B):
                    mma_ts_step(
                        bmm_y_t_inv_desc,
                        a_ptr.subview(
                            i * bmm_y_t_inv_desc.sps_B * bmm_y_t_inv_desc.tmem_advance_A
                        ),
                        b_desc + i * (bmm_y_t_inv_desc.smem_subtile_B >> 4),
                        c_ptr,
                        k,
                        cutlass.Boolean(i + k > 0),
                    )
            if elect_one:
                bars.mb_u_acc_ready.arrive(cta_group=1)

            # ---- dY = dU(T) @ T^-1 ---------------------------------------------------
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

            # ---- dK restore part = dstate(S) @ U^T -----------------------------------
            bars.mb_u_smem_ready.wait(u_smem_index.phase)
            u_smem_index = advance(u_smem_index, 1)
            if has_dstate:
                bars.mb_dstate_smem_ready.wait(dstate_smem_index.phase)
                dstate_smem_index = advance(dstate_smem_index, 1)
                mma_ss(
                    bmm_dstate_u_desc,
                    d_dstate_trans,
                    d_u,
                    nvvm.make_tmem_ptr(
                        (tmem_base + cfg.tmem_dk_restore_acc_offset), cutlass.Float32
                    ),
                    accumulate=False,
                )
                if elect_one:
                    bars.mb_dk_restore_part_acc_ready.arrive(cta_group=1)
                    bars.mb_dstate_smem_done.arrive(cta_group=1)

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

            # ---- dK inv part = scale.Q decay^T(S) @ dA -------------------------------
            bars.mb_da_ready[intermediate_stage].wait(intermediate_phase)
            mma_ss(
                bmm_q_decay_da_desc,
                d_qd_trans,
                d_int_da,
                nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dk_inv_acc_offset), cutlass.Float32),
                accumulate=False,
            )

            # ---- dQ attn += K inv^T(S) @ dA^T ----------------------------------------
            if chunk_idx >= FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_k_inv_da_desc,
                    d_ki_trans,
                    d_int_da,
                    nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dq_acc_offset), cutlass.Float32),
                    accumulate=True,
                )
            if chunk_idx < FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_k_inv_da_desc,
                    d_ki_trans,
                    d_int_da,
                    nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dq_acc_offset), cutlass.Float32),
                    accumulate=False,
                )
            if elect_one:
                bars.mb_dq_acc_ready.arrive(cta_group=1)
                bars.mb_da_done[intermediate_stage].arrive(cta_group=1)

            # ---- dK decay part = state(T) @ (Beta.dY)^T ------------------------------
            bars.mb_dv_tmastg_ready[chunk_serial % cfg.smem_dv_stages].wait(
                (chunk_serial // cfg.smem_dv_stages) % 2
            )
            if chunk_idx >= FIRST_STATE_CHUNK:
                a_ptr = nvvm.make_tmem_ptr(
                    (
                        tmem_base
                        + cfg.tmem_state_input_offset
                        + (chunk_serial % 2) * (cfg.d_v // 2)
                    ),
                    cutlass.Int8,
                )
                b_desc = d_dv0 + (chunk_serial % cfg.smem_dv_stages) * dv_seg
                c_ptr = nvvm.make_tmem_ptr(
                    (tmem_base + cfg.tmem_dk_decay_acc_offset), cutlass.Float32
                )
                for i in cutlass.range_constexpr(bmm_state_dy_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_state_dy_desc.sps_B):
                        mma_ts_step(
                            bmm_state_dy_desc,
                            a_ptr.subview(
                                i * bmm_state_dy_desc.sps_B * bmm_state_dy_desc.tmem_advance_A
                            ),
                            b_desc + i * (bmm_state_dy_desc.smem_subtile_B >> 4),
                            c_ptr,
                            k,
                            cutlass.Boolean(i + k > 0),
                        )
            if elect_one:
                bars.mb_state_input_done[chunk_serial % 2].arrive(cta_group=1)

            # ---- dK inv part += K decay^T(S) @ -dM strict ----------------------------
            bars.mb_dm_ready[intermediate_stage].wait(intermediate_phase)
            mma_ss(
                bmm_k_decay_dm_desc,
                d_kd_trans,
                d_int_ndm,
                nvvm.make_tmem_ptr((tmem_base + cfg.tmem_dk_inv_acc_offset), cutlass.Float32),
                accumulate=True,
            )
            if elect_one:
                bars.mb_dk_inv_part_acc_ready.arrive(cta_group=1)

            # ---- dK decay part += K inv^T(S) @ dM strict^T ---------------------------
            if chunk_idx >= FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_k_inv_dm_desc,
                    d_ki_trans,
                    d_int_dm,
                    nvvm.make_tmem_ptr(
                        (tmem_base + cfg.tmem_dk_decay_acc_offset), cutlass.Float32
                    ),
                    accumulate=True,
                )
            if chunk_idx < FIRST_STATE_CHUNK:
                mma_ss(
                    bmm_k_inv_dm_desc,
                    d_ki_trans,
                    d_int_dm,
                    nvvm.make_tmem_ptr(
                        (tmem_base + cfg.tmem_dk_decay_acc_offset), cutlass.Float32
                    ),
                    accumulate=False,
                )
            if elect_one:
                bars.mb_dk_decay_part_acc_ready.arrive(cta_group=1)
                bars.mb_dm_done[intermediate_stage].arrive(cta_group=1)
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
    q_ratio,
    k_ratio,
    v_ratio,
) -> None:
    """TMA-LDG warp role (warp 14): persistent tile-scheduler loop issuing
    every G->S TMA load."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)

    raw_index = PipelineState.start(phase=1)
    state_index = PipelineState.start(phase=1)
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
    sState_tma = SmemTile(
        base=sState_raw,
        elems_per_stage=(cfg.d_k * cfg.d_v),
        stages=cfg.smem_state_stages,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=cfg.d_v * 64,
    )
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
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
        head_v = head_idx // v_ratio
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
        num_compute_chunks = compute_end - write_start
        for rev_idx in cutlass.range(num_compute_chunks, unroll=1):
            chunk_idx = compute_end - cutlass.Int32(1) - rev_idx
            chunk_start = chunk_idx * cfg.b_t

            # ---- Q / K / Gate / V loads: one transaction barrier per stage -----------
            bars.mb_raw_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_raw_ready[raw_index.idx].arrive(
                    n_bytes=cfg.tma_q_bytes
                    + cfg.tma_k_bytes
                    + cfg.tma_gate_bytes
                    + cfg.tma_v_bytes
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
            v_slice = tma_slice_runtime_desc(desc_v_slot, cutlass.Int32(0), head_v, chunk_start)
            tma_load_tile(sV_tma[raw_index.idx], v_slice, raw_ready_ptr)

            # ---- dO load -------------------------------------------------------------
            bars.mb_do_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_do_ready[raw_index.idx].arrive(n_bytes=cfg.tma_do_bytes)
            do_slice = tma_slice_runtime_desc(desc_do_slot, cutlass.Int32(0), head_o, chunk_start)
            tma_load_tile(
                sDo_tma[raw_index.idx], do_slice, bars.mb_do_ready[raw_index.idx].smem_ptr
            )

            # ---- entering state ------------------------------------------------------
            if chunk_idx >= FIRST_STATE_CHUNK:
                state_idx = state_index.idx
                bars.mb_state_cg0_done[state_idx].wait(state_index.phase)
                bars.mb_state_done[state_idx].wait(state_index.phase)
                state_index = advance(state_index, cfg.smem_state_stages)
                if elect_one:
                    bars.mb_state_ready[state_idx].arrive(n_bytes=cfg.tma_state_bytes)
                state_slice = tma_slice_runtime_desc(
                    desc_checkpoint_slot, cutlass.Int32(0), cutlass.Int32(0), chunk_idx, head_o
                )
                tma_load_tile(
                    sState_tma[state_idx], state_slice, bars.mb_state_ready[state_idx].smem_ptr
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
    tmem_base_holder,
    warp_idx,
    scale,
    mBeta,
    sBeta_raw,
    sK_inv_raw,
    sGate_raw,
    sGate_load_ptr,
    sK_raw,
    sQ_raw,
    sState_raw,
    sNorm_raw,
    sK_decay_raw,
    sQ_decay_raw,
    sDecay_scale_raw,
    bars,
) -> None:
    """WG0 warp role (warps 0-3): persistent tile-scheduler loop, gate prefix
    scan, and the decay operand materialization into tcgen05 SMEM for every
    chunk.  Also stashes the per-row Q/K inverse norms for WG2's dGate
    assembly and copies H -> TMEM f16 at the chunk tail."""
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
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_holder.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    row_lo_addr = tmem_row << 16
    state_index = PipelineState.start(phase=0)
    chunk_serial_base = cutlass.Int32(0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
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
                for row_pair in cutlass.range(cfg.b_t // 2, unroll=1):
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

            # ---- raw Q/K: SMEM -> TMEM ring (channel-major) --------------------------
            qk_raw_stage = chunk_serial % cfg.tmem_qk_raw_stages
            bars.mb_qk_raw_done[qk_raw_stage].wait(
                ((chunk_serial // cfg.tmem_qk_raw_stages) + 1) % 2
            )
            raw_seg = channel_dim // 64
            raw_dim = channel_dim - raw_seg * 64
            q_raw_words = cute.make_rmem_tensor((cfg.b_t // 2,), cutlass.Int32)
            k_raw_words = cute.make_rmem_tensor((cfg.b_t // 2,), cutlass.Int32)
            for t2 in cutlass.range_constexpr(cfg.b_t // 2):
                t0 = 2 * t2
                ridx0 = (
                    raw_seg * (cfg.b_t * 64)
                    + t0 * 64
                    + swizzle_xor_128b(t0, raw_dim, elem_bytes=2)
                )
                ridx1 = (
                    raw_seg * (cfg.b_t * 64)
                    + (t0 + 1) * 64
                    + swizzle_xor_128b(t0 + 1, raw_dim, elem_bytes=2)
                )
                q0 = (sQ_ptr + ridx0).load().bitcast(cutlass.Uint16)
                q1 = (sQ_ptr + ridx1).load().bitcast(cutlass.Uint16)
                k0 = (sK_ptr + ridx0).load().bitcast(cutlass.Uint16)
                k1 = (sK_ptr + ridx1).load().bitcast(cutlass.Uint16)
                q_raw_words[t2] = pack_u16x2(q0, q1)
                k_raw_words[t2] = pack_u16x2(k0, k1)
            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(
                    row_lo_addr
                    + (tmem_col + cfg.tmem_qraw_input_offset + qk_raw_stage * (cfg.b_t // 2)),
                    cutlass.Int8,
                ),
                q_raw_words.load(),
            )
            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(
                    row_lo_addr
                    + (tmem_col + cfg.tmem_kraw_input_offset + qk_raw_stage * (cfg.b_t // 2)),
                    cutlass.Int8,
                ),
                k_raw_words.load(),
            )
            nvvm.tcgen05_wait("store")
            bars.mb_qk_raw_ready[qk_raw_stage].arrive()

            nvvm.barrier_cta_sync(cfg.cg0_sync_barrier_id, thread_count=cfg.cg0_threads)

            k_inv_pack = cute.make_rmem_tensor((dk_halves * 4,), cutlass.Int32)
            raw_q_regs = cute.make_rmem_tensor((dk_halves * 8,), cutlass.Float32)
            raw_k_regs = cute.make_rmem_tensor((dk_halves * 8,), cutlass.Float32)

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

            nvvm.fence_proxy("async.shared", space="cta")

            q_inv_norm = opaque_f32_zero() + cutlass.Float32(1.0)
            k_inv_norm = opaque_f32_zero() + cutlass.Float32(1.0)
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
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
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
                f16_segment = dim_base // 64
                f16_segment_dim = dim_base - f16_segment * 64
                op_idx = (
                    f16_segment * (cfg.b_t * 64)
                    + decay_row * 64
                    + swizzle_xor_128b(decay_row, f16_segment_dim, elem_bytes=2)
                )
                (sQ_decay_ptr + op_idx).store(q_decay_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_q_decay_ready[decay_stage].arrive()

            # ---- state copy: SMEM -> TMEM f16 ----------------------------------------
            bars.mb_state_input_done[chunk_serial % 2].wait(((chunk_serial // 2) + 1) % 2)
            bars.mb_state_input_cg2_done[chunk_serial % 2].wait(((chunk_serial // 2) + 1) % 2)
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_state_ready[state_index.idx].wait(state_index.phase)
                state_src = smem_data_ptr(sState_raw) + state_index.idx * (cfg.d_k * cfg.d_v)
                ldm_row_coord = (lane_idx // 16) * 8 + (lane_idx & 7)
                ldm_col_offset = ((lane_idx // 8) & 1) * 8
                k_base = cg0_warp * channel_rows
                k_seg_off = (k_base // 64) * (cfg.d_v * 64)
                row_hi_addr = row_lo_addr + (16 << 16)
                state_col = (
                    tmem_col + cfg.tmem_state_input_offset + (chunk_serial % 2) * (cfg.d_v // 2)
                )
                for dv_blk in cutlass.range_constexpr(cfg.d_v // 16):
                    dv_row = dv_blk * 16 + ldm_row_coord
                    frag_lo = nvvm.ldmatrix(
                        state_src
                        + k_seg_off
                        + dv_row * 64
                        + swizzle_xor_128b(dv_row, (k_base + ldm_col_offset) % 64, elem_bytes=2),
                        4,
                        nvvm.MMALayout.COL,
                    )
                    nvvm.tcgen05_st(
                        "16x128b",
                        nvvm.make_tmem_ptr(row_lo_addr + (state_col + dv_blk * 8), cutlass.Int8),
                        frag_lo,
                    )
                    if cutlass.const_expr(channel_rows == cfg.threads_per_warp):
                        frag_hi = nvvm.ldmatrix(
                            state_src
                            + k_seg_off
                            + dv_row * 64
                            + swizzle_xor_128b(
                                dv_row, (k_base + 16 + ldm_col_offset) % 64, elem_bytes=2
                            ),
                            4,
                            nvvm.MMALayout.COL,
                        )
                        nvvm.tcgen05_st(
                            "16x128b",
                            nvvm.make_tmem_ptr(
                                row_hi_addr + (state_col + dv_blk * 8), cutlass.Int8
                            ),
                            frag_hi,
                        )
                nvvm.tcgen05_wait("store")
                bars.mb_state_cg0_done[state_index.idx].arrive()
                state_index = advance(state_index, cfg.smem_state_stages)
            bars.mb_state_input_ready[chunk_serial % 2].arrive()
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
    sReduction_raw,
    sBetaM_raw,
    sDecay_scale_raw,
    bars,
) -> None:
    """WG1 warp role (warps 4-7): the value-side TMEM staging."""
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
    cg1_tidx = warp_idx % 4 * cfg.threads_per_warp + lane_idx

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
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
    while tile_idx < total_tiles:
        (
            batch_idx,
            head_idx,
            batch_start,
            _batch_end,
            batch_seqlen,
            batch_num_chunks,
            write_start,
            write_end,
            _compute_start,
            compute_end,
        ) = decode_work_item(cfg, tile_idx, mWorkItems)
        num_compute_chunks = compute_end - write_start

        # ---- dstate seed: GMEM -> TMEM + SMEM ----------------------------------------
        if cutlass.const_expr(cfg.use_dstate_in):
            if num_compute_chunks > 0:
                seed_true = compute_end == batch_num_chunks
                bars.mb_dstate_smem_done.wait(dstate_smem_done_index.phase)
                bars.mb_dstate_smem_cg2_done.wait(dstate_smem_done_index.phase)
                dstate_smem_done_index = advance(dstate_smem_done_index, 1)
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

                # ---- dstate seed: TMEM -> sDstate ------------------------------------
                for i in cutlass.range_constexpr(cfg.d_k // 16):
                    dstate_words = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            row_lo_addr + (tmem_col + cfg.tmem_dstate_input_offset + i * 8),
                            cutlass.Float32,
                        ),
                        num=8,
                    )
                    if state_row_valid:
                        for half in cutlass.range_constexpr(2):
                            d_base = i * 16 + half * 8
                            h_vec = cutlass.Vector.from_elements(
                                (
                                    dstate_words[half * 4],
                                    dstate_words[half * 4 + 1],
                                    dstate_words[half * 4 + 2],
                                    dstate_words[half * 4 + 3],
                                ),
                                cutlass.Float32,
                            ).bitcast(cfg.io_dtype)
                            h_addr = (
                                (d_base // 64) * (cfg.d_v * 64)
                                + value_dim * 64
                                + swizzle_xor_128b(value_dim, d_base % 64, elem_bytes=2)
                            )
                            (smem_data_ptr(sDstate_raw) + h_addr).store(h_vec, alignment=16)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dstate_smem_ready.arrive()

        for rev_idx in cutlass.range(num_compute_chunks, unroll=1):
            chunk_idx = compute_end - cutlass.Int32(1) - rev_idx
            chunk_serial = chunk_serial_base + rev_idx
            sV_ptr = smem_data_ptr(sV_raw) + raw_index.idx * (cfg.d_v * cfg.b_t)
            smem_data_ptr(sDo_raw) + raw_index.idx * (cfg.d_v * cfg.b_t)
            sBeta_ptr = smem_data_ptr(sBeta_raw) + (chunk_serial % cfg.smem_beta_stages) * cfg.b_t
            row_lo_addr = tmem_row << 16
            row_hi_addr = (tmem_row + 16) << 16
            cutlass.Boolean(rev_idx > 0)
            if cutlass.const_expr(cfg.use_dstate_in):
                cutlass.Boolean(True)

            # ---- Y stage: Y = Beta * (V - k state) -----------------------------------
            bars.mb_raw_ready[raw_index.idx].wait((chunk_serial // cfg.smem_raw_stages) % 2)
            projection_col_id = tmem_col + cfg.tmem_state_k_acc_offset
            input_col_id = tmem_col + cfg.tmem_y_input_offset
            raw_v_frag_lo = nvvm.ldmatrix(
                sV_ptr
                + (value_dim_base + ov_col_offset) // 64 * (cfg.b_t * 64)
                + ov_row_coord * 64
                + swizzle_xor_128b(
                    ov_row_coord, (value_dim_base + ov_col_offset) % 64, elem_bytes=2
                ),
                4,
                nvvm.MMALayout.COL,
            )
            if cutlass.const_expr(cfg.d_v == 128):
                raw_v_frag_hi = nvvm.ldmatrix(
                    sV_ptr
                    + (value_dim_base + 16 + ov_col_offset) // 64 * (cfg.b_t * 64)
                    + ov_row_coord * 64
                    + swizzle_xor_128b(
                        ov_row_coord, (value_dim_base + 16 + ov_col_offset) % 64, elem_bytes=2
                    ),
                    4,
                    nvvm.MMALayout.COL,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_raw_done[raw_index.idx].arrive()

            bars.mb_beta_ready[chunk_serial % cfg.smem_beta_stages].wait(
                (chunk_serial // cfg.smem_beta_stages) % 2
            )
            # FP32 beta per token pair (pair p of the ldmatrix fragment reads tokens of half p //
            # 2).
            beta_regs = cute.make_rmem_tensor((4,), cutlass.Float32)
            for half in cutlass.range_constexpr(2):
                token0 = (half * 4 + (lane_idx & 3)) * 2
                beta_regs[2 * half] = (sBeta_ptr + token0).load().to(cutlass.Float32)
                beta_regs[2 * half + 1] = (sBeta_ptr + token0 + 1).load().to(cutlass.Float32)
            # FP32 residual V - k state, kept for the dBeta v-term below; only the Beta-scaled MMA
            # operand is packed (the packed residual lost small residuals against a large
            # contraction).
            diff_w_lo = cute.make_rmem_tensor((8,), cutlass.Float32)
            diff_w_hi = cute.make_rmem_tensor((8,), cutlass.Float32)
            y_input_pack_lo = cute.make_rmem_tensor((4,), cutlass.Int32)
            y_input_pack_hi = cute.make_rmem_tensor((4,), cutlass.Int32)
            if chunk_idx >= FIRST_STATE_CHUNK:
                bars.mb_state_k_acc_ready.wait(state_k_index.phase)
                state_k_index = advance(state_k_index, 1)
                state_k_vec_lo = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(row_lo_addr + projection_col_id, cutlass.Float32),
                    num=2,
                )
                for reg_idx in cutlass.range_constexpr(4):
                    lo, hi = 2 * reg_idx, 2 * reg_idx + 1
                    b_lo, b_hi = 2 * (reg_idx // 2), 2 * (reg_idx // 2) + 1
                    y_input_pack_lo[reg_idx], diff_w_lo[lo], diff_w_lo[hi] = beta_residual_f16x2(
                        raw_v_frag_lo[reg_idx],
                        beta_regs[b_lo],
                        beta_regs[b_hi],
                        state_k_vec_lo[lo],
                        state_k_vec_lo[hi],
                        dtype=cfg.io_dtype,
                    )
                if cutlass.const_expr(cfg.d_v == 128):
                    state_k_vec_hi = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(row_hi_addr + projection_col_id, cutlass.Float32),
                        num=2,
                    )
                    for reg_idx in cutlass.range_constexpr(4):
                        lo, hi = 2 * reg_idx, 2 * reg_idx + 1
                        b_lo, b_hi = 2 * (reg_idx // 2), 2 * (reg_idx // 2) + 1
                        y_input_pack_hi[reg_idx], diff_w_hi[lo], diff_w_hi[hi] = (
                            beta_residual_f16x2(
                                raw_v_frag_hi[reg_idx],
                                beta_regs[b_lo],
                                beta_regs[b_hi],
                                state_k_vec_hi[lo],
                                state_k_vec_hi[hi],
                                dtype=cfg.io_dtype,
                            )
                        )
            if chunk_idx < FIRST_STATE_CHUNK:
                for reg_idx in cutlass.range_constexpr(4):
                    lo, hi = 2 * reg_idx, 2 * reg_idx + 1
                    b_lo, b_hi = 2 * (reg_idx // 2), 2 * (reg_idx // 2) + 1
                    y_input_pack_lo[reg_idx], diff_w_lo[lo], diff_w_lo[hi] = beta_residual_f16x2(
                        raw_v_frag_lo[reg_idx],
                        beta_regs[b_lo],
                        beta_regs[b_hi],
                        dtype=cfg.io_dtype,
                    )
                if cutlass.const_expr(cfg.d_v == 128):
                    for reg_idx in cutlass.range_constexpr(4):
                        lo, hi = 2 * reg_idx, 2 * reg_idx + 1
                        b_lo, b_hi = 2 * (reg_idx // 2), 2 * (reg_idx // 2) + 1
                        y_input_pack_hi[reg_idx], diff_w_hi[lo], diff_w_hi[hi] = (
                            beta_residual_f16x2(
                                raw_v_frag_hi[reg_idx],
                                beta_regs[b_lo],
                                beta_regs[b_hi],
                                dtype=cfg.io_dtype,
                            )
                        )
            nvvm.tcgen05_st(
                "16x128b",
                nvvm.make_tmem_ptr(row_lo_addr + input_col_id, cutlass.Int8),
                y_input_pack_lo.load(),
            )
            if cutlass.const_expr(cfg.d_v == 128):
                nvvm.tcgen05_st(
                    "16x128b",
                    nvvm.make_tmem_ptr(row_hi_addr + input_col_id, cutlass.Int8),
                    y_input_pack_hi.load(),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_y_input_ready.arrive()

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

            # ---- U read: TMEM -> sU --------------------------------------------------
            bars.mb_u_acc_ready.wait(u_acc_index.phase)
            u_acc_index = advance(u_acc_index, 1)
            u_col_id = tmem_col + cfg.tmem_u_acc_offset
            u_vec_lo = nvvm.tcgen05_ld(
                "16x256b", nvvm.make_tmem_ptr(row_lo_addr + u_col_id, cutlass.Float32), num=2
            )
            if cutlass.const_expr(cfg.d_v == 128):
                u_vec_hi = nvvm.tcgen05_ld(
                    "16x256b", nvvm.make_tmem_ptr(row_hi_addr + u_col_id, cutlass.Float32), num=2
                )

            u_pack_lo = cute.make_rmem_tensor((4,), cutlass.Int32)
            u_pack_hi = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                u_pack_lo[reg_idx] = fp32_to_fp16(
                    u_vec_lo[2 * reg_idx], u_vec_lo[2 * reg_idx + 1], dtype=cfg.io_dtype
                )
                if cutlass.const_expr(cfg.d_v == 128):
                    u_pack_hi[reg_idx] = fp32_to_fp16(
                        u_vec_hi[2 * reg_idx], u_vec_hi[2 * reg_idx + 1], dtype=cfg.io_dtype
                    )
            nvvm.stmatrix(
                smem_data_ptr(sU_raw)
                + (value_dim_base + ov_col_offset) // 64 * (cfg.b_t * 64)
                + ov_row_coord * 64
                + swizzle_xor_128b(
                    ov_row_coord, (value_dim_base + ov_col_offset) % 64, elem_bytes=2
                ),
                u_pack_lo.load(),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            if cutlass.const_expr(cfg.d_v == 128):
                nvvm.stmatrix(
                    smem_data_ptr(sU_raw)
                    + (value_dim_base + 16 + ov_col_offset) // 64 * (cfg.b_t * 64)
                    + ov_row_coord * 64
                    + swizzle_xor_128b(
                        ov_row_coord, (value_dim_base + 16 + ov_col_offset) % 64, elem_bytes=2
                    ),
                    u_pack_hi.load(),
                    nvvm.MMALayout.COL,
                    shape=nvvm.StoreShape.M8N8,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_u_smem_ready.arrive()

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

            # ---- dY -> sdY -----------------------------------------------------------
            dy_addr_lo = (
                (value_dim_base + ov_col_offset) // 64 * (cfg.b_t * 64)
                + ov_row_coord * 64
                + swizzle_xor_128b(
                    ov_row_coord, (value_dim_base + ov_col_offset) % 64, elem_bytes=2
                )
            )
            dy_addr_hi = (
                (value_dim_base + 16 + ov_col_offset) // 64 * (cfg.b_t * 64)
                + ov_row_coord * 64
                + swizzle_xor_128b(
                    ov_row_coord, (value_dim_base + 16 + ov_col_offset) % 64, elem_bytes=2
                )
            )
            dy_pack_lo = cute.make_rmem_tensor((4,), cutlass.Int32)
            dy_pack_hi = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                dy_pack_lo[reg_idx] = fp32_to_fp16(
                    dy_vec_lo[2 * reg_idx], dy_vec_lo[2 * reg_idx + 1], dtype=cfg.io_dtype
                )
                if cutlass.const_expr(cfg.d_v == 128):
                    dy_pack_hi[reg_idx] = fp32_to_fp16(
                        dy_vec_hi[2 * reg_idx], dy_vec_hi[2 * reg_idx + 1], dtype=cfg.io_dtype
                    )
            nvvm.stmatrix(
                smem_data_ptr(sDy_raw) + dy_addr_lo,
                dy_pack_lo.load(),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            if cutlass.const_expr(cfg.d_v == 128):
                nvvm.stmatrix(
                    smem_data_ptr(sDy_raw) + dy_addr_hi,
                    dy_pack_hi.load(),
                    nvvm.MMALayout.COL,
                    shape=nvvm.StoreShape.M8N8,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dy_smem_ready.arrive()

            # ---- Beta scalars -> Beta.dY -> sdV --------------------------------------
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

            # ---- dBeta v-term part: dY.(V - k state) ---------------------------------
            token4 = cute.make_rmem_tensor((4,), cutlass.Float32)
            for s in cutlass.range_constexpr(4):
                token4[s] = cutlass.Float32(0.0)
            for j in cutlass.range_constexpr(4):
                slot = cutlass.const_expr(2 * (j // 2))
                lo, hi = ffma2(
                    dy_vec_lo[2 * j],
                    dy_vec_lo[2 * j + 1],
                    diff_w_lo[2 * j],
                    diff_w_lo[2 * j + 1],
                    token4[slot],
                    token4[slot + 1],
                )
                if cutlass.const_expr(cfg.d_v == 128):
                    lo, hi = ffma2(
                        dy_vec_hi[2 * j],
                        dy_vec_hi[2 * j + 1],
                        diff_w_hi[2 * j],
                        diff_w_hi[2 * j + 1],
                        lo,
                        hi,
                    )
                token4[slot] = lo
                token4[slot + 1] = hi

            # ---- Beta.dY -> sdV ------------------------------------------------------
            beta_dy_pack_lo = cute.make_rmem_tensor((4,), cutlass.Int32)
            beta_dy_pack_hi = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                beta_dy_pack_lo[reg_idx] = fp32_to_fp16(
                    beta_dy_regs_lo[2 * reg_idx],
                    beta_dy_regs_lo[2 * reg_idx + 1],
                    dtype=cfg.io_dtype,
                )
                if cutlass.const_expr(cfg.d_v == 128):
                    beta_dy_pack_hi[reg_idx] = fp32_to_fp16(
                        beta_dy_regs_hi[2 * reg_idx],
                        beta_dy_regs_hi[2 * reg_idx + 1],
                        dtype=cfg.io_dtype,
                    )
            dv_stage = chunk_serial % cfg.smem_dv_stages
            sdv_stage_base = dv_stage * (cfg.b_t * cfg.d_v)
            bars.mb_dv_tmastg_done[dv_stage].wait(dv_done_index.phase)
            dv_done_index = advance(dv_done_index, cfg.smem_dv_stages)
            nvvm.stmatrix(
                smem_data_ptr(sDv_raw) + sdv_stage_base + dy_addr_lo,
                beta_dy_pack_lo.load(),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            if cutlass.const_expr(cfg.d_v == 128):
                nvvm.stmatrix(
                    smem_data_ptr(sDv_raw) + sdv_stage_base + dy_addr_hi,
                    beta_dy_pack_hi.load(),
                    nvvm.MMALayout.COL,
                    shape=nvvm.StoreShape.M8N8,
                )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dv_tmastg_ready[dv_stage].arrive()

            # ---- dBeta = sum over v of dY.(V - k state) + M-term ---------------------
            nvvm.barrier_cta_sync(cfg.cg1_sync_barrier_id, thread_count=cfg.cg1_threads)
            bars.mb_dbeta_m_ready.wait(dbeta_m_index.phase)
            dbeta_m_index = advance(dbeta_m_index, 1)
            for off in cutlass.range_constexpr(3):
                step = cutlass.const_expr(4 << off)
                for s in cutlass.range_constexpr(4):
                    token4[s] = token4[s] + cutlass.Float32(
                        nvvm.shfl_sync(0xFFFFFFFF, token4[s], step, 31, kind=nvvm.Shfl.BFLY)
                    )
            if lane_idx < 4:
                for s in cutlass.range_constexpr(4):
                    sReduction_raw[
                        (warp_idx % 4) * cfg.b_t + (lane_idx % 4) * 2 + (s % 2) + 8 * (s // 2)
                    ] = token4[s]
            nvvm.barrier_cta_sync(cfg.cg1_sync_barrier_id, thread_count=cfg.cg1_threads)
            if cg1_tidx < cfg.b_t:
                acc = cutlass.Float32(0.0)
                for w in cutlass.range_constexpr(4):
                    acc = acc + sReduction_raw[w * cfg.b_t + cg1_tidx]
                db_val = acc + sBetaM_raw[cg1_tidx]
                token_idx = chunk_idx * cutlass.Int32(cfg.b_t) + cg1_tidx
                if token_idx < batch_seqlen and chunk_idx < write_end:
                    mDbeta[batch_start + token_idx, head_idx] = db_val.to(mDbeta.element_type)
            nvvm.barrier_cta_sync(cfg.cg1_sync_barrier_id, thread_count=cfg.cg1_threads)

            # ---- dstate capture for the next chunk -----------------------------------
            bars.mb_dstate_acc_ready.wait(dstate_ready_index.phase)
            dstate_ready_index = advance(dstate_ready_index, 1)
            if rev_idx + cutlass.Int32(1) < num_compute_chunks:
                bars.mb_dstate_smem_done.wait(dstate_smem_done_index.phase)
                bars.mb_dstate_smem_cg2_done.wait(dstate_smem_done_index.phase)
                dstate_smem_done_index = advance(dstate_smem_done_index, 1)
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

                # ---- dstate input: TMEM -> sDstate -----------------------------------
                for i in cutlass.range(cfg.d_k // 32, unroll=1):
                    dstate_words = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            row_lo_addr + (tmem_col + cfg.tmem_dstate_input_offset + i * 16),
                            cutlass.Float32,
                        ),
                        num=16,
                    )
                    if state_row_valid:
                        for half in cutlass.range_constexpr(4):
                            d_base = i * 32 + half * 8
                            h_vec = cutlass.Vector.from_elements(
                                (
                                    dstate_words[half * 4],
                                    dstate_words[half * 4 + 1],
                                    dstate_words[half * 4 + 2],
                                    dstate_words[half * 4 + 3],
                                ),
                                cutlass.Float32,
                            ).bitcast(cfg.io_dtype)
                            h_addr = (
                                (d_base // 64) * (cfg.d_v * 64)
                                + value_dim * 64
                                + swizzle_xor_128b(value_dim, d_base % 64, elem_bytes=2)
                            )
                            (smem_data_ptr(sDstate_raw) + h_addr).store(h_vec, alignment=16)
                nvvm.fence_proxy("async.shared", space="cta")
                bars.mb_dstate_smem_ready.arrive()
            raw_index = advance(raw_index, cfg.smem_raw_stages)

        # ---- tile end: dstate0 store / zero-length pass-through ----------------------
        if cutlass.const_expr(mDstate0 is not None):
            dstate_dst = mWorkItems[tile_idx, WORK_ITEM_DSTATE_DST]
            if num_compute_chunks > 0:
                if dstate_dst >= 0:
                    row_lo_addr = tmem_row << 16
                    dstate0_vw = 16 // (mDstate0.element_type.width // 8)
                    dstate0_dst = (
                        mDstate0.iterator + mDstate0.layout((dstate_dst, head_idx, value_dim, 0))
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
                                            dstate0_vec[g * dstate0_vw + t].to(
                                                mDstate0.element_type
                                            )
                                            for t in range(dstate0_vw)
                                        ),
                                        mDstate0.element_type,
                                    ),
                                    alignment=16,
                                )
            else:
                if state_row_valid and dstate_dst >= 0:
                    for key_dim_base in cutlass.range_constexpr(0, cfg.d_k, 32):
                        for kk_i in cutlass.range_constexpr(32):
                            kd = key_dim_base + kk_i
                            if cutlass.const_expr(cfg.use_dstate_in):
                                mDstate0[dstate_dst, head_idx, value_dim, kd] = mDstate_in[
                                    batch_idx, head_idx, value_dim, kd
                                ].to(mDstate0.element_type)
                            else:
                                mDstate0[dstate_dst, head_idx, value_dim, kd] = cutlass.Float32(
                                    0.0
                                ).to(mDstate0.element_type)
        if num_compute_chunks > 0:
            bars.mb_dstate0_acc_stored.arrive()
        chunk_serial_base += num_compute_chunks
        tile_idx, scheduler_state = scheduler_next_tile(
            cfg, bars, sScheduler, scheduler_state, elect_one
        )

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
    tmem_base_holder,
    warp_idx,
    mGate,
    sGate_raw,
    sNorm_raw,
    sDaDiag_raw,
    sDq_raw,
    sDk_raw,
    sRed1_raw,
    sProj_raw,
    sDstate_raw,
    sDgate_raw,
    scale,
    bars,
) -> None:
    """WG2 warp role (warps 8-11): the gradient drain.  Each thread owns one
    key channel d for all 16 tokens.  It reads the dQ accumulator and the four
    dK parts from TMEM, assembles dQ/dK with the per-channel gate scales (raw
    Q/K arrive through WG0's TMEM ring), applies the in-kernel L2-norm
    backward row projection, assembles the per-channel dGate including the
    g_last terms, reverse-cumsums it in registers, and stages dQ/dK/dGate for
    the epilogue's TMA stores."""
    nvvm.setmaxregister(
        cfg.num_regs_compute_group_2,
        nvvm.SetMaxRegisterAction.INCREASE
        if cfg.num_regs_compute_group_2 >= 65536 // cfg.threads_per_cta
        else nvvm.SetMaxRegisterAction.DECREASE,
    )
    elect_one = nvvm.elect_sync()
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_holder.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    cg2_warp = warp_idx - cfg.compute_group_2_warp_ids[0]
    linear_gate = cutlass.const_expr(not cfg.log_gate)
    channel_rows = cutlass.const_expr(cfg.d_k // len(cfg.compute_group_2_warp_ids))
    split_channels = cutlass.const_expr(channel_rows < cfg.threads_per_warp)
    channel = cg2_warp * cfg.threads_per_warp + lane_idx
    if cutlass.const_expr(split_channels):
        fragment_quad = lane_idx % 4
        quad_base = lane_idx - fragment_quad
        channel_lo = cg2_warp * channel_rows + lane_idx // 4
        channel_hi = channel_lo + cutlass.Int32(8)
    dv_halves = cutlass.const_expr(cfg.d_v // 64)
    row_lo_addr = tmem_row << 16

    raw_index = PipelineState.start(phase=0)
    dq_acc_index = PipelineState.start(phase=0)
    dk_decay_part_index = PipelineState.start(phase=0)
    dk_inv_part_index = PipelineState.start(phase=0)
    dk_restore_part_index = PipelineState.start(phase=0)
    dgate_last_dstate_smem_index = PipelineState.start(phase=0)
    chunk_serial_base = cutlass.Int32(0)
    scheduler_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    FIRST_STATE_CHUNK = 0 if cfg.use_initial_state else 1
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
        for rev_idx in cutlass.range(num_compute_chunks, unroll=1):
            chunk_idx = compute_end - cutlass.Int32(1) - rev_idx
            chunk_serial = chunk_serial_base + rev_idx
            chunk_start = chunk_idx * cfg.b_t
            raw_stage = chunk_serial % cfg.smem_raw_stages
            decay_stage = chunk_serial % cfg.smem_decay_stages
            has_dstate = cutlass.Boolean(rev_idx > 0)
            if cutlass.const_expr(cfg.use_dstate_in):
                has_dstate = cutlass.Boolean(True)
            sGate_exchange_ptr = smem_data_ptr(sGate_raw) + raw_stage * (cfg.d_k * cfg.b_t)

            bars.mb_k_decay_inv_ready[decay_stage].wait(
                (chunk_serial // cfg.smem_decay_stages) % 2
            )

            # ---- per-channel gate scales --------------------------------------------
            if cutlass.const_expr(split_channels):
                eg_frag = cute.make_rmem_tensor((8,), cutlass.Float32)
                for cs in cutlass.range_constexpr(2):
                    channel_cs = channel_lo if cs == 0 else channel_hi
                    seg_cs = channel_cs // 32
                    dim_cs = channel_cs - seg_cs * 32
                    for rep in cutlass.range_constexpr(2):
                        for n in cutlass.range_constexpr(2):
                            token = fragment_quad * 2 + cutlass.Int32(rep * 8 + n)
                            eg_frag[rep * 4 + cs * 2 + n] = (
                                sGate_exchange_ptr
                                + seg_cs * (cfg.b_t * 32)
                                + token * 32
                                + swizzle_xor_128b(token ^ seg_cs, dim_cs, elem_bytes=4)
                            ).load()
            else:
                f32_seg = channel // 32
                f32_dim = channel - f32_seg * 32
                f16_seg = channel // 64
                f16_dim = channel - f16_seg * 64
                eg = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
                for t in cutlass.range_constexpr(cfg.b_t):
                    eg[t] = (
                        sGate_exchange_ptr
                        + f32_seg * (cfg.b_t * 32)
                        + t * 32
                        + swizzle_xor_128b(t ^ f32_seg, f32_dim, elem_bytes=4)
                    ).load()
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_raw_done[raw_stage].arrive()

            # ---- staged raw Q/K: TMEM ring cols for this chunk -----------------------
            qk_raw_stage = chunk_serial % cfg.tmem_qk_raw_stages
            qraw_col = tmem_col + cfg.tmem_qraw_input_offset + qk_raw_stage * (cfg.b_t // 2)
            kraw_col = tmem_col + cfg.tmem_kraw_input_offset + qk_raw_stage * (cfg.b_t // 2)
            bars.mb_qk_raw_ready[qk_raw_stage].wait((chunk_serial // cfg.tmem_qk_raw_stages) % 2)

            # ---- dGate last dot: sum over v of dstate[v, c] * state[c, v] ------------
            dgate_last_val = cutlass.Float32(0.0)
            if cutlass.const_expr(split_channels):
                dgate_last_frag = cute.make_rmem_tensor((2,), cutlass.Float32)
                for cs in cutlass.range_constexpr(2):
                    dgate_last_frag[cs] = cutlass.Float32(0.0)
            bars.mb_state_input_ready[chunk_serial % 2].wait((chunk_serial // 2) % 2)
            if has_dstate:
                bars.mb_dstate_smem_ready.wait(dgate_last_dstate_smem_index.phase)
                dgate_last_dstate_smem_index = advance(dgate_last_dstate_smem_index, 1)
                if cutlass.const_expr(split_channels):
                    state_vec = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(
                            row_lo_addr
                            + (
                                tmem_col
                                + cfg.tmem_state_input_offset
                                + (chunk_serial % 2) * (cfg.d_v // 2)
                            ),
                            cutlass.Float32,
                        ),
                        num=cfg.d_v // 16,
                    )
                    hacc = cute.make_rmem_tensor((8,), cutlass.Float32)
                    for i in cutlass.range_constexpr(8):
                        hacc[i] = opaque_f32_zero()
                    for cs in cutlass.range_constexpr(2):
                        channel_cs = channel_lo if cs == 0 else channel_hi
                        for rep in cutlass.range_constexpr(cfg.d_v // 16):
                            for n in cutlass.range_constexpr(2):
                                word_col = fragment_quad * 2 + cutlass.Int32(rep * 8 + n)
                                v0 = word_col * 2
                                state_pair = cutlass.Vector.from_elements(
                                    (state_vec[rep * 4 + cs * 2 + n],), cutlass.Float32
                                ).bitcast(cfg.io_dtype)
                                dstate_addr0 = v0 * 64 + swizzle_xor_128b(
                                    v0, channel_cs, elem_bytes=2
                                )
                                dstate_addr1 = (v0 + 1) * 64 + swizzle_xor_128b(
                                    v0 + 1, channel_cs, elem_bytes=2
                                )
                                hval0 = (
                                    (smem_data_ptr(sDstate_raw) + dstate_addr0)
                                    .load()
                                    .to(cutlass.Float32)
                                )
                                hval1 = (
                                    (smem_data_ptr(sDstate_raw) + dstate_addr1)
                                    .load()
                                    .to(cutlass.Float32)
                                )
                                bucket = cs * 4 + (rep % 2) * 2 + n
                                hacc[bucket] = hacc[bucket] + hval0 * state_pair[0].to(
                                    cutlass.Float32
                                )
                                hacc[bucket] = hacc[bucket] + hval1 * state_pair[1].to(
                                    cutlass.Float32
                                )
                    for cs in cutlass.range_constexpr(2):
                        part = (hacc[cs * 4 + 0] + hacc[cs * 4 + 1]) + (
                            hacc[cs * 4 + 2] + hacc[cs * 4 + 3]
                        )
                        part = part + cutlass.Float32(
                            nvvm.shfl_sync(0xFFFFFFFF, part, 1, 31, kind=nvvm.Shfl.BFLY)
                        )
                        part = part + cutlass.Float32(
                            nvvm.shfl_sync(0xFFFFFFFF, part, 2, 31, kind=nvvm.Shfl.BFLY)
                        )
                        dgate_last_frag[cs] = part
                else:
                    for trip in cutlass.range(2 * dv_halves, unroll=1):
                        state_vec = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr(
                                row_lo_addr
                                + (
                                    tmem_col
                                    + cfg.tmem_state_input_offset
                                    + (chunk_serial % 2) * (cfg.d_v // 2)
                                    + trip * 16
                                ),
                                cutlass.Float32,
                            ),
                            num=16,
                        )
                        hacc = cute.make_rmem_tensor((8,), cutlass.Float32)
                        for i in cutlass.range_constexpr(8):
                            hacc[i] = opaque_f32_zero()
                        for j in cutlass.range_constexpr(16):
                            v0 = trip * 32 + 2 * j
                            state_pair = cutlass.Vector.from_elements(
                                (state_vec[j],), cutlass.Float32
                            ).bitcast(cfg.io_dtype)
                            dstate_addr0 = (
                                (channel // 64) * (cfg.d_v * 64)
                                + v0 * 64
                                + swizzle_xor_128b(2 * j, channel % 64, elem_bytes=2)
                            )
                            dstate_addr1 = (
                                (channel // 64) * (cfg.d_v * 64)
                                + (v0 + 1) * 64
                                + swizzle_xor_128b(2 * j + 1, channel % 64, elem_bytes=2)
                            )
                            hval0 = (
                                (smem_data_ptr(sDstate_raw) + dstate_addr0)
                                .load()
                                .to(cutlass.Float32)
                            )
                            hval1 = (
                                (smem_data_ptr(sDstate_raw) + dstate_addr1)
                                .load()
                                .to(cutlass.Float32)
                            )
                            hacc[(2 * j) % 8] = hacc[(2 * j) % 8] + hval0 * state_pair[0].to(
                                cutlass.Float32
                            )
                            hacc[(2 * j + 1) % 8] = hacc[(2 * j + 1) % 8] + hval1 * state_pair[
                                1
                            ].to(cutlass.Float32)
                        pa0, pb0 = fadd2(hacc[0], hacc[2], hacc[4], hacc[6])
                        pa1, pb1 = fadd2(hacc[1], hacc[3], hacc[5], hacc[7])
                        part_a, part_b = fadd2(pa0, pb0, pa1, pb1)
                        dgate_last_val = dgate_last_val + (part_a + part_b)
                bars.mb_dstate_smem_cg2_done.arrive()
            bars.mb_state_input_cg2_done[chunk_serial % 2].arrive()

            # ---- part-store accumulators ---------------------------------------------
            if cutlass.const_expr(split_channels):
                dq_frag = cute.make_rmem_tensor((8,), cutlass.Float32)
                dk_frag = cute.make_rmem_tensor((8,), cutlass.Float32)
                dgate_frag = cute.make_rmem_tensor((8,), cutlass.Float32)
                dgate_last_kdot = cute.make_rmem_tensor((4,), cutlass.Float32)
                for i in cutlass.range_constexpr(4):
                    dgate_last_kdot[i] = opaque_f32_zero()
                for i in cutlass.range_constexpr(8):
                    dk_frag[i] = cutlass.Float32(0.0)
            else:
                dq_n = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
                dk_n = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
                dgate_regs = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
                dgate_last_acc = cute.make_rmem_tensor((4,), cutlass.Float32)
                for i in cutlass.range_constexpr(4):
                    dgate_last_acc[i] = opaque_f32_zero()
                for t in cutlass.range_constexpr(cfg.b_t):
                    dk_n[t] = cutlass.Float32(0.0)

            # ---- dK restore part store: (eGl/eG) scale + dGate last K-dot ------------
            if has_dstate:
                bars.mb_dk_restore_part_acc_ready.wait(dk_restore_part_index.phase)
                dk_restore_part_index = advance(dk_restore_part_index, 1)
                if cutlass.const_expr(split_channels):
                    dk_restore_vec = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(
                            row_lo_addr + (tmem_col + cfg.tmem_dk_restore_acc_offset),
                            cutlass.Float32,
                        ),
                        num=cfg.b_t // 8,
                    )
                    kr_words = nvvm.tcgen05_ld(
                        "16x256b",
                        nvvm.make_tmem_ptr(row_lo_addr + kraw_col, cutlass.Float32),
                        num=cfg.b_t // 16,
                    )
                    for cs in cutlass.range_constexpr(2):
                        for rep in cutlass.range_constexpr(2):
                            source_lane = quad_base + fragment_quad // 2 + cutlass.Int32(rep * 2)
                            word_n0 = cutlass.Float32(
                                nvvm.shfl_sync(
                                    0xFFFFFFFF,
                                    kr_words[cs * 2],
                                    source_lane,
                                    31,
                                    kind=nvvm.Shfl.IDX,
                                )
                            )
                            word_n1 = cutlass.Float32(
                                nvvm.shfl_sync(
                                    0xFFFFFFFF,
                                    kr_words[cs * 2 + 1],
                                    source_lane,
                                    31,
                                    kind=nvvm.Shfl.IDX,
                                )
                            )
                            k_word = word_n1 if fragment_quad % 2 == 1 else word_n0
                            k_pair = cutlass.Vector.from_elements(
                                (k_word,), cutlass.Float32
                            ).bitcast(cfg.io_dtype)
                            for n in cutlass.range_constexpr(2):
                                k = rep * 4 + cs * 2 + n
                                token = fragment_quad * 2 + cutlass.Int32(rep * 8 + n)
                                dk_hat = (
                                    cute.math.rcp(eg_frag[k], approx=True, ftz=True)
                                    * dk_restore_vec[k]
                                )
                                dk_frag[k] = dk_hat
                                k_v = k_pair[n].to(cutlass.Float32)
                                dgate_last_kdot[cs * 2 + n] = (
                                    dgate_last_kdot[cs * 2 + n] + k_v * dk_hat
                                )
                else:
                    dk_restore_part_vec = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(
                            row_lo_addr + (tmem_col + cfg.tmem_dk_restore_acc_offset),
                            cutlass.Float32,
                        ),
                        num=cfg.b_t,
                    )
                    kr_words = nvvm.tcgen05_ld(
                        "32x32b",
                        nvvm.make_tmem_ptr(row_lo_addr + kraw_col, cutlass.Float32),
                        num=cfg.b_t // 2,
                    )
                    for t in cutlass.range_constexpr(cfg.b_t):
                        dk_hat = (
                            cute.math.rcp(eg[t], approx=True, ftz=True) * dk_restore_part_vec[t]
                        )
                        dk_n[t] = dk_hat
                        k_pair = cutlass.Vector.from_elements(
                            (kr_words[t // 2],), cutlass.Float32
                        ).bitcast(cfg.io_dtype)
                        k_v = k_pair[t % 2].to(cutlass.Float32)
                        dgate_last_acc[t % 4] = dgate_last_acc[t % 4] + k_v * dk_hat

            # ---- dQ acc store: eG.scale ----------------------------------------------
            bars.mb_dq_acc_ready.wait(dq_acc_index.phase)
            dq_acc_index = advance(dq_acc_index, 1)
            if cutlass.const_expr(split_channels):
                dq_vec = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(
                        row_lo_addr + (tmem_col + cfg.tmem_dq_acc_offset), cutlass.Float32
                    ),
                    num=cfg.b_t // 8,
                )
                for k2 in cutlass.range_constexpr(4):
                    k = 2 * k2
                    lo, hi = fmul2(eg_frag[k], eg_frag[k + 1], scale, scale)
                    dq_frag[k], dq_frag[k + 1] = fmul2(lo, hi, dq_vec[k], dq_vec[k + 1])
            else:
                dq_vec = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(
                        row_lo_addr + (tmem_col + cfg.tmem_dq_acc_offset), cutlass.Float32
                    ),
                    num=cfg.b_t,
                )
                for t2 in cutlass.range_constexpr(cfg.b_t // 2):
                    t = 2 * t2
                    lo, hi = fmul2(eg[t], eg[t + 1], scale, scale)
                    dq_n[t], dq_n[t + 1] = fmul2(lo, hi, dq_vec[t], dq_vec[t + 1])

            # ---- dK inv part store: (dA - dM) term, 1/eG scale -----------------------
            bars.mb_dk_inv_part_acc_ready.wait(dk_inv_part_index.phase)
            dk_inv_part_index = advance(dk_inv_part_index, 1)
            if cutlass.const_expr(split_channels):
                dk_inv_vec = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(
                        row_lo_addr + (tmem_col + cfg.tmem_dk_inv_acc_offset), cutlass.Float32
                    ),
                    num=cfg.b_t // 8,
                )
                for k in cutlass.range_constexpr(8):
                    dk_frag[k] = dk_frag[k] + dk_inv_vec[k] * cute.math.rcp(
                        eg_frag[k], approx=True, ftz=True
                    )
            else:
                dk_inv_part_vec = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(
                        row_lo_addr + (tmem_col + cfg.tmem_dk_inv_acc_offset), cutlass.Float32
                    ),
                    num=cfg.b_t,
                )
                for t in cutlass.range_constexpr(cfg.b_t):
                    dk_n[t] = dk_n[t] + dk_inv_part_vec[t] * cute.math.rcp(
                        eg[t], approx=True, ftz=True
                    )

            # ---- dK decay part store: -eG scale --------------------------------------
            bars.mb_dk_decay_part_acc_ready.wait(dk_decay_part_index.phase)
            dk_decay_part_index = advance(dk_decay_part_index, 1)
            if cutlass.const_expr(split_channels):
                dk_decay_vec = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(
                        row_lo_addr + (tmem_col + cfg.tmem_dk_decay_acc_offset), cutlass.Float32
                    ),
                    num=cfg.b_t // 8,
                )
                for k in cutlass.range_constexpr(8):
                    dgate_frag[k] = -eg_frag[k] * dk_decay_vec[k]
                    dk_frag[k] = dk_frag[k] + dgate_frag[k]
            else:
                dk_decay_part_vec = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(
                        row_lo_addr + (tmem_col + cfg.tmem_dk_decay_acc_offset), cutlass.Float32
                    ),
                    num=cfg.b_t,
                )
                for t in cutlass.range_constexpr(cfg.b_t):
                    dgate_regs[t] = -eg[t] * dk_decay_part_vec[t]
                    dk_n[t] = dk_n[t] + dgate_regs[t]

            nvvm.tcgen05_wait("load")
            bars.mb_dqk_acc_done.arrive()

            # ---- dA diagonal: Gamma-free dQ/dK terms ---------------------------------
            da_diag_stage = chunk_serial % cfg.smem_da_diag_stages
            bars.mb_da_diag_ready[da_diag_stage].wait(
                (chunk_serial // cfg.smem_da_diag_stages) % 2
            )
            if cutlass.const_expr(split_channels):
                da_diag_frag = cute.make_rmem_tensor((4,), cutlass.Float32)
                for rep in cutlass.range_constexpr(2):
                    for n in cutlass.range_constexpr(2):
                        token = fragment_quad * 2 + cutlass.Int32(rep * 8 + n)
                        da_diag_frag[rep * 2 + n] = (
                            sDaDiag_raw[da_diag_stage * cfg.b_t + token] * scale
                        )
            else:
                da_diag = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
                for t in cutlass.range_constexpr(cfg.b_t):
                    da_diag[t] = sDaDiag_raw[da_diag_stage * cfg.b_t + t] * scale
            bars.mb_da_diag_done[da_diag_stage].arrive()

            # ---- dGate finalize ------------------------------------------------------
            if cutlass.const_expr(split_channels):
                qf_words = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(row_lo_addr + qraw_col, cutlass.Float32),
                    num=cfg.b_t // 16,
                )
                kf_words = nvvm.tcgen05_ld(
                    "16x256b",
                    nvvm.make_tmem_ptr(row_lo_addr + kraw_col, cutlass.Float32),
                    num=cfg.b_t // 16,
                )
                for cs in cutlass.range_constexpr(2):
                    for rep in cutlass.range_constexpr(2):
                        source_lane = quad_base + fragment_quad // 2 + cutlass.Int32(rep * 2)
                        q_word_n0 = cutlass.Float32(
                            nvvm.shfl_sync(
                                0xFFFFFFFF, qf_words[cs * 2], source_lane, 31, kind=nvvm.Shfl.IDX
                            )
                        )
                        q_word_n1 = cutlass.Float32(
                            nvvm.shfl_sync(
                                0xFFFFFFFF,
                                qf_words[cs * 2 + 1],
                                source_lane,
                                31,
                                kind=nvvm.Shfl.IDX,
                            )
                        )
                        k_word_n0 = cutlass.Float32(
                            nvvm.shfl_sync(
                                0xFFFFFFFF, kf_words[cs * 2], source_lane, 31, kind=nvvm.Shfl.IDX
                            )
                        )
                        k_word_n1 = cutlass.Float32(
                            nvvm.shfl_sync(
                                0xFFFFFFFF,
                                kf_words[cs * 2 + 1],
                                source_lane,
                                31,
                                kind=nvvm.Shfl.IDX,
                            )
                        )
                        q_word = q_word_n1 if fragment_quad % 2 == 1 else q_word_n0
                        k_word = k_word_n1 if fragment_quad % 2 == 1 else k_word_n0
                        q_pair = cutlass.Vector.from_elements((q_word,), cutlass.Float32).bitcast(
                            cfg.io_dtype
                        )
                        k_pair = cutlass.Vector.from_elements((k_word,), cutlass.Float32).bitcast(
                            cfg.io_dtype
                        )
                        for n in cutlass.range_constexpr(2):
                            k = rep * 4 + cs * 2 + n
                            token = fragment_quad * 2 + cutlass.Int32(rep * 8 + n)
                            q_v = q_pair[n].to(cutlass.Float32)
                            k_v = k_pair[n].to(cutlass.Float32)
                            dq_frag[k] = dq_frag[k] + da_diag_frag[rep * 2 + n] * k_v
                            dk_frag[k] = dk_frag[k] + da_diag_frag[rep * 2 + n] * q_v
                            dgate_frag[k] = q_v * dq_frag[k] + k_v * (
                                cutlass.Float32(2.0) * dgate_frag[k] - dk_frag[k]
                            )
                for cs in cutlass.range_constexpr(2):
                    kdot = dgate_last_kdot[cs * 2] + dgate_last_kdot[cs * 2 + 1]
                    kdot = kdot + cutlass.Float32(
                        nvvm.shfl_sync(0xFFFFFFFF, kdot, 1, 31, kind=nvvm.Shfl.BFLY)
                    )
                    kdot = kdot + cutlass.Float32(
                        nvvm.shfl_sync(0xFFFFFFFF, kdot, 2, 31, kind=nvvm.Shfl.BFLY)
                    )
                    if fragment_quad == 3:
                        dgate_frag[4 + cs * 2 + 1] = dgate_frag[4 + cs * 2 + 1] + kdot
            else:
                qf_words = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(row_lo_addr + qraw_col, cutlass.Float32),
                    num=cfg.b_t // 2,
                )
                kf_words = nvvm.tcgen05_ld(
                    "32x32b",
                    nvvm.make_tmem_ptr(row_lo_addr + kraw_col, cutlass.Float32),
                    num=cfg.b_t // 2,
                )
                for t in cutlass.range_constexpr(cfg.b_t):
                    q_pair = cutlass.Vector.from_elements(
                        (qf_words[t // 2],), cutlass.Float32
                    ).bitcast(cfg.io_dtype)
                    k_pair = cutlass.Vector.from_elements(
                        (kf_words[t // 2],), cutlass.Float32
                    ).bitcast(cfg.io_dtype)
                    q_v = q_pair[t % 2].to(cutlass.Float32)
                    k_v = k_pair[t % 2].to(cutlass.Float32)
                    dq_n[t] = dq_n[t] + da_diag[t] * k_v
                    dk_n[t] = dk_n[t] + da_diag[t] * q_v
                    dgate_regs[t] = q_v * dq_n[t] + k_v * (
                        cutlass.Float32(2.0) * dgate_regs[t] - dk_n[t]
                    )
                dgate_regs[cfg.b_t - 1] = dgate_regs[cfg.b_t - 1] + (
                    (dgate_last_acc[0] + dgate_last_acc[1])
                    + (dgate_last_acc[2] + dgate_last_acc[3])
                )

            bars.mb_qk_raw_done[qk_raw_stage].arrive()

            # ---- stage dQ/dK for the epilogue TMA stores -----------------------------
            dq_stage = chunk_serial % cfg.smem_dq_stages
            dk_stage = chunk_serial % cfg.smem_dk_stages
            bars.mb_dq_tmastg_done[dq_stage].wait(((chunk_serial // cfg.smem_dq_stages) + 1) % 2)
            bars.mb_dk_tmastg_done[dk_stage].wait(((chunk_serial // cfg.smem_dk_stages) + 1) % 2)
            dq_base = dq_stage * (cfg.b_t * cfg.d_k)
            dk_base = dk_stage * (cfg.b_t * cfg.d_k)
            if cutlass.const_expr(split_channels):
                for cs in cutlass.range_constexpr(2):
                    channel_cs = channel_lo if cs == 0 else channel_hi
                    for rep in cutlass.range_constexpr(2):
                        for n in cutlass.range_constexpr(2):
                            k = rep * 4 + cs * 2 + n
                            token = fragment_quad * 2 + cutlass.Int32(rep * 8 + n)
                            out_idx = token * 64 + swizzle_xor_128b(
                                token, channel_cs, elem_bytes=2
                            )
                            (smem_data_ptr(sDq_raw) + dq_base + out_idx).store(
                                dq_frag[k].to(cfg.io_dtype)
                            )
                            (smem_data_ptr(sDk_raw) + dk_base + out_idx).store(
                                dk_frag[k].to(cfg.io_dtype)
                            )
            else:
                for t in cutlass.range_constexpr(cfg.b_t):
                    out_idx = (
                        f16_seg * (cfg.b_t * 64)
                        + t * 64
                        + swizzle_xor_128b(t, f16_dim, elem_bytes=2)
                    )
                    (smem_data_ptr(sDq_raw) + dq_base + out_idx).store(dq_n[t].to(cfg.io_dtype))
                    (smem_data_ptr(sDk_raw) + dk_base + out_idx).store(dk_n[t].to(cfg.io_dtype))
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dq_tmastg_ready[dq_stage].arrive()
            bars.mb_dk_tmastg_ready[dk_stage].arrive()

            # ---- dGate last add ------------------------------------------------------
            if has_dstate:
                if chunk_idx >= FIRST_STATE_CHUNK:
                    if cutlass.const_expr(split_channels):
                        if fragment_quad == 3:
                            for cs in cutlass.range_constexpr(2):
                                dgate_frag[4 + cs * 2 + 1] = (
                                    dgate_frag[4 + cs * 2 + 1] + dgate_last_frag[cs]
                                )
                    else:
                        dgate_regs[cfg.b_t - 1] = dgate_regs[cfg.b_t - 1] + dgate_last_val

            # ---- dGate reverse cumsum ------------------------------------------------
            if cutlass.const_expr(split_channels):
                for cs in cutlass.range_constexpr(2):
                    pair_sum = cute.make_rmem_tensor((2,), cutlass.Float32)
                    higher = cute.make_rmem_tensor((2,), cutlass.Float32)
                    block_total = cute.make_rmem_tensor((2,), cutlass.Float32)
                    for rep in cutlass.range_constexpr(2):
                        pair_sum[rep] = (
                            dgate_frag[rep * 4 + cs * 2] + dgate_frag[rep * 4 + cs * 2 + 1]
                        )
                        quad_1 = cutlass.Float32(
                            nvvm.shfl_sync(
                                0xFFFFFFFF,
                                pair_sum[rep],
                                quad_base + cutlass.Int32(1),
                                31,
                                kind=nvvm.Shfl.IDX,
                            )
                        )
                        quad_2 = cutlass.Float32(
                            nvvm.shfl_sync(
                                0xFFFFFFFF,
                                pair_sum[rep],
                                quad_base + cutlass.Int32(2),
                                31,
                                kind=nvvm.Shfl.IDX,
                            )
                        )
                        quad_3 = cutlass.Float32(
                            nvvm.shfl_sync(
                                0xFFFFFFFF,
                                pair_sum[rep],
                                quad_base + cutlass.Int32(3),
                                31,
                                kind=nvvm.Shfl.IDX,
                            )
                        )
                        after = quad_1 if fragment_quad < 1 else cutlass.Float32(0.0)
                        after = after + (quad_2 if fragment_quad < 2 else cutlass.Float32(0.0))
                        after = after + (quad_3 if fragment_quad < 3 else cutlass.Float32(0.0))
                        higher[rep] = after
                        total = pair_sum[rep] + cutlass.Float32(
                            nvvm.shfl_sync(0xFFFFFFFF, pair_sum[rep], 1, 31, kind=nvvm.Shfl.BFLY)
                        )
                        total = total + cutlass.Float32(
                            nvvm.shfl_sync(0xFFFFFFFF, total, 2, 31, kind=nvvm.Shfl.BFLY)
                        )
                        block_total[rep] = total
                    tail = dgate_frag[4 + cs * 2 + 1] + higher[1]
                    dgate_frag[4 + cs * 2 + 1] = tail
                    dgate_frag[4 + cs * 2] = dgate_frag[4 + cs * 2] + tail
                    head = dgate_frag[cs * 2 + 1] + higher[0]
                    dgate_frag[cs * 2 + 1] = head + block_total[1]
                    dgate_frag[cs * 2] = (dgate_frag[cs * 2] + head) + block_total[1]
            else:
                suffix = cutlass.Float32(0.0)
                for rt in cutlass.range_constexpr(cfg.b_t):
                    t = cfg.b_t - 1 - rt
                    suffix = suffix + dgate_regs[t]
                    dgate_regs[t] = suffix

            # ---- stage dGate for the epilogue TMA store ------------------------------
            dgate_stage = chunk_serial % cfg.smem_dgate_stages
            bars.mb_dgate_tmastg_done[dgate_stage].wait(
                ((chunk_serial // cfg.smem_dgate_stages) + 1) % 2
            )
            dgate_base = dgate_stage * (cfg.b_t * cfg.d_k)
            if cutlass.const_expr(split_channels):
                for cs in cutlass.range_constexpr(2):
                    channel_cs = channel_lo if cs == 0 else channel_hi
                    for rep in cutlass.range_constexpr(2):
                        for n in cutlass.range_constexpr(2):
                            k = rep * 4 + cs * 2 + n
                            token = fragment_quad * 2 + cutlass.Int32(rep * 8 + n)
                            if cutlass.const_expr(linear_gate):
                                token_idx = chunk_start + token
                                if token_idx < batch_seqlen:
                                    dgate_frag[k] = dgate_frag[k] * cute.math.rcp(
                                        mGate[batch_start + token_idx, head_idx, channel_cs].to(
                                            cutlass.Float32
                                        )
                                        + cutlass.Float32(1e-10),
                                        approx=True,
                                        ftz=True,
                                    )
                            if cutlass.const_expr(cfg.gate_dtype == cutlass.Float32):
                                seg_cs = channel_cs // 32
                                dim_cs = channel_cs - seg_cs * 32
                                dgate_idx = (
                                    seg_cs * (cfg.b_t * 32)
                                    + token * 32
                                    + swizzle_xor_128b(token, dim_cs, elem_bytes=4)
                                )
                                (smem_data_ptr(sDgate_raw) + dgate_base + dgate_idx).store(
                                    dgate_frag[k]
                                )
                            else:
                                dgate_idx = token * 64 + swizzle_xor_128b(
                                    token, channel_cs, elem_bytes=2
                                )
                                (smem_data_ptr(sDgate_raw) + dgate_base + dgate_idx).store(
                                    dgate_frag[k].to(cfg.gate_dtype)
                                )
            else:
                if cutlass.const_expr(linear_gate):
                    for t in cutlass.range_constexpr(cfg.b_t):
                        token_idx = chunk_start + cutlass.Int32(t)
                        if token_idx < batch_seqlen:
                            dgate_regs[t] = dgate_regs[t] * cute.math.rcp(
                                mGate[batch_start + token_idx, head_idx, channel].to(
                                    cutlass.Float32
                                )
                                + cutlass.Float32(1e-10),
                                approx=True,
                                ftz=True,
                            )
                if cutlass.const_expr(cfg.gate_dtype == cutlass.Float32):
                    for t in cutlass.range_constexpr(cfg.b_t):
                        dgate_idx = (
                            f32_seg * (cfg.b_t * 32)
                            + t * 32
                            + swizzle_xor_128b(t, f32_dim, elem_bytes=4)
                        )
                        (smem_data_ptr(sDgate_raw) + dgate_base + dgate_idx).store(dgate_regs[t])
                else:
                    dgate_seg = channel // 64
                    dgate_dim = channel - dgate_seg * 64
                    for t in cutlass.range_constexpr(cfg.b_t):
                        dgate_idx = (
                            dgate_seg * (cfg.b_t * 64)
                            + t * 64
                            + swizzle_xor_128b(t, dgate_dim, elem_bytes=2)
                        )
                        (smem_data_ptr(sDgate_raw) + dgate_base + dgate_idx).store(
                            dgate_regs[t].to(cfg.gate_dtype)
                        )
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_dgate_tmastg_ready[dgate_stage].arrive()
            raw_index = advance(raw_index, cfg.smem_raw_stages)
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
    base_v,
    base_gate,
    base_do,
    base_dq,
    base_dk,
    base_dv,
    base_dgate,
    base_checkpoint,
    desc_workspace: cute.Tensor,
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
    checkpoint_every_n: cutlass.Int32,
) -> None:
    """Per-batch descriptor-array build, one warp per array. Runs inside the
    prologue kernel after its order pass; warps past the array count fall
    through the widx guards."""
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
    sub6 = cute.make_tensor(
        desc_workspace.iterator + 6 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    sub7 = cute.make_tensor(
        desc_workspace.iterator + 7 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    sub8 = cute.make_tensor(
        desc_workspace.iterator + 8 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    sub9 = cute.make_tensor(
        desc_workspace.iterator + 9 * arr_words, cute.make_layout((arr_words,), stride=(1,))
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
        emit_seq_descs(base_v, sub2, cu_seqlens, v, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 3:
        emit_seq_descs(base_gate, sub3, cu_seqlens, gate, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 4:
        emit_seq_descs(base_do, sub4, cu_seqlens, do, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 5:
        emit_seq_descs(base_dq, sub5, cu_seqlens, dq, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 6:
        emit_seq_descs(base_dk, sub6, cu_seqlens, dk, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 7:
        emit_seq_descs(base_dv, sub7, cu_seqlens, dv, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 8:
        emit_seq_descs(base_dgate, sub8, cu_seqlens, dgate, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 9:
        emit_checkpoint_seq_descs(
            base_checkpoint,
            sub9,
            cu_seqlens,
            state_checkpoints,
            n_batch,
            checkpoint_every_n,
            2,
            lanes=32,
        )
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )


@cute.kernel
def frost_kda_bprop_prologue(
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
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
    desc_workspace: cute.Tensor,
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
    mScheduler: cute.Tensor | None,
    n_batch: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
) -> None:
    """Two-CTA prologue. Under ``run_order`` this kernel is the first
    work-item-table consumer, so block 0 LPT-orders the table and zeroes both
    consumers' scheduler rings via :func:`order_body`; block 1 builds the
    per-batch TMA-descriptor arrays via :func:`build_descs_body`, one warp
    per array."""
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
            base_v,
            base_gate,
            base_do,
            base_dq,
            base_dk,
            base_dv,
            base_dgate,
            base_checkpoint,
            desc_workspace,
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
            checkpoint_every_n,
        )


@cute.jit
def prologue(
    io_dtype: cutlass.Constexpr,
    b_t: cutlass.Constexpr[int],
    run_order: cutlass.Constexpr[bool],
    order_gen: cutlass.Constexpr[bool],
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
    scheduler_all: cute.Tensor | None,
    tensormap_workspace: cute.Tensor,
    stream: cuda_driver.CUstream,
):
    """One-launch prologue: LPT-order the work items (when ``run_order``) and
    build the 10 per-(batch, head) capped TMA-descriptor arrays into
    ``tensormap_workspace`` (sequence-relative coordinates; tail loads
    zero-fill and tail stores clip in hardware)."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    h_v = v.shape[1]
    ho = gate.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    d_k = q.shape[2]
    d_v = v.shape[2]
    bytes_per_element = io_dtype.width // 8
    box_elems = 128 // bytes_per_element
    seqlen = q.shape[0]

    q_headed = cute.make_tensor(
        q.iterator, cute.make_layout((d_k, h_q, seqlen), stride=(1, q.stride[1], q.stride[0]))
    )
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
    do_headed = cute.make_tensor(
        do.iterator, cute.make_layout((d_v, ho, seqlen), stride=(1, do.stride[1], do.stride[0]))
    )
    dq_headed = cute.make_tensor(
        dq.iterator, cute.make_layout((d_k, ho, seqlen), stride=(1, dq.stride[1], dq.stride[0]))
    )
    dk_headed = cute.make_tensor(
        dk.iterator, cute.make_layout((d_k, ho, seqlen), stride=(1, dk.stride[1], dk.stride[0]))
    )
    dv_headed = cute.make_tensor(
        dv.iterator, cute.make_layout((d_v, ho, seqlen), stride=(1, dv.stride[1], dv.stride[0]))
    )
    dgate_headed = cute.make_tensor(
        dgate.iterator,
        cute.make_layout((d_k, ho, seqlen), stride=(1, dgate.stride[1], dgate.stride[0])),
    )

    swizzle = cuda.TensorMapSwizzle.s128b
    base_q = cuda.create_tensor_map_tiled_from_view(
        q_headed, box_dims=(box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
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
    base_do = cuda.create_tensor_map_tiled_from_view(
        do_headed, box_dims=(box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    base_dq = cuda.create_tensor_map_tiled_from_view(
        dq_headed, box_dims=(box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    base_dk = cuda.create_tensor_map_tiled_from_view(
        dk_headed, box_dims=(box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    base_dv = cuda.create_tensor_map_tiled_from_view(
        dv_headed, box_dims=(box_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swizzle
    )
    base_dgate = cuda.create_tensor_map_tiled_from_view(
        dgate_headed,
        box_dims=(128 // (dgate.element_type.width // 8), 1, b_t),
        stride_order=(0, 1, 2),
        swizzle=swizzle,
    )

    checkpoint_view = cute.make_tensor(
        state_checkpoints.iterator,
        cute.make_layout(
            (d_k, d_v, state_checkpoints.shape[0], ho),
            stride=(
                state_checkpoints.stride[3],
                state_checkpoints.stride[2],
                state_checkpoints.stride[0],
                state_checkpoints.stride[1],
            ),
        ),
    )
    base_checkpoint = cuda.create_tensor_map_tiled_from_view(
        checkpoint_view, box_dims=(64, d_v, 1, 1), stride_order=(0, 1, 2, 3), swizzle=swizzle
    )

    frost_kda_bprop_prologue(
        run_order,
        order_gen,
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
        scheduler_all,
        cutlass.Int32(batch_size),
        cutlass.Int32(b_t),
    ).launch(grid=(2, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)


@cute.jit
def host(
    cfg: cutlass.Constexpr,
    q_ratio: cutlass.Int32,
    k_ratio: cutlass.Int32,
    v_ratio: cutlass.Int32,
    beta: cute.Tensor,
    gate_main: cute.Tensor | None,
    state_checkpoints: cute.Tensor,
    dgate: cute.Tensor,
    dbeta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    d_initial_state: cute.Tensor | None,
    d_final_state: cute.Tensor | None,
    work_items: cute.Tensor | None,
    work_count: cute.Tensor | None,
    scheduler_counter: cute.Tensor,
    tensormap_workspace: cute.Tensor,
    scale: cutlass.Float32,
    stream,
) -> None:
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
        gate: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, cfg.raw_gate_cosize], 1024]
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
        dgate: cute.struct.Align[cute.struct.MemRange[cfg.gate_dtype, cfg.dgate_cosize], 1024]

    q_ratio = cute.FastDivmodDivisorV2(q_ratio)
    k_ratio = cute.FastDivmodDivisorV2(k_ratio)
    v_ratio = cute.FastDivmodDivisorV2(v_ratio)
    num_sequences = cu_seqlens.shape[0] - 1

    # ---- launch ----------------------------------------------------------------------
    n_desc = num_sequences
    grid_shape = (cfg.max_active_clusters, 1, 1)
    frost_kda_bprop(
        cfg,
        SharedStorage,
        q_ratio,
        k_ratio,
        v_ratio,
        tensormap_workspace,
        n_desc,
        beta,
        gate_main,
        cu_seqlens,
        dgate,
        dbeta,
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
def frost_kda_bprop(
    cfg: cutlass.Constexpr,
    shared_type: cutlass.Constexpr,
    q_ratio: cute.FastDivmodDivisorV2,
    k_ratio: cute.FastDivmodDivisorV2,
    v_ratio: cute.FastDivmodDivisorV2,
    tensormap_workspace: cute.Tensor,
    n_desc: cutlass.Int32,
    mBeta: cute.Tensor,
    mGate: cute.Tensor | None,
    cu_seqlens: cute.Tensor,
    mDgate: cute.Tensor,
    mDbeta: cute.Tensor,
    mDstate0: cute.Tensor | None,
    mDstate_in: cute.Tensor | None,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mScheduler: cute.Tensor,
    scale: cutlass.Float32,
) -> None:
    """BT=16 KDA backward kernel (persistent, 16 warps)."""
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
    desc_v_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_gate_base = desc_base_words + cutlass.Int32(3) * arr_words
    desc_do_base = desc_base_words + cutlass.Int32(4) * arr_words
    desc_dq_base = desc_base_words + cutlass.Int32(5) * arr_words
    desc_dk_base = desc_base_words + cutlass.Int32(6) * arr_words
    desc_dv_base = desc_base_words + cutlass.Int32(7) * arr_words
    desc_dgate_base = desc_base_words + cutlass.Int32(8) * arr_words
    desc_checkpoint_base = desc_base_words + cutlass.Int32(9) * arr_words

    SMEM = cutlass.AddressSpace.smem
    bars = make_bars(cfg)
    tmem_base_holder = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=4)
    sScheduler = cutlass.Array(cutlass.Int32, cfg.scheduler_stages, space=SMEM, alignment=16)
    bytes_per_element = cfg.io_dtype.width // 8
    SWZ = 2
    LEAD = 16
    STRIDE = 8 * 128
    STATE_ALT_LEAD = cfg.d_v * 128

    sBeta_raw = cutlass.Array(
        cutlass.Float32, cfg.smem_beta_stages * cfg.b_t, space=SMEM, alignment=64
    )
    sNorm_raw = cutlass.Array(
        cutlass.Float32, cfg.tmem_qk_raw_stages * 2 * cfg.b_t, space=SMEM, alignment=64
    )
    sReduction_raw = cutlass.Array(cutlass.Float32, 4 * cfg.b_t, space=SMEM, alignment=64)
    sRed1_raw = cutlass.Array(cutlass.Float32, 4 * cfg.b_t, space=SMEM, alignment=64)
    sProj_raw = cutlass.Array(
        cutlass.Float32, 2 * cfg.b_t * cfg.cg2_threads, space=SMEM, alignment=64
    )
    sBetaM_raw = cutlass.Array(cutlass.Float32, cfg.b_t, space=SMEM, alignment=64)
    sDaDiag_raw = cutlass.Array(
        cutlass.Float32, cfg.smem_da_diag_stages * cfg.b_t, space=SMEM, alignment=64
    )
    storage = SmemAllocator().allocate(shared_type)
    sState_raw = storage.state.get_tensor(cute.make_layout((cfg.state_cosize,)))
    sDstate_raw = storage.dstate.get_tensor(cute.make_layout((cfg.d_k * cfg.d_v,)))
    sK_decay_raw = storage.k_decay.get_tensor(cute.make_layout((cfg.operand_cosize,)))
    sK_inv_raw = storage.k_inv.get_tensor(cute.make_layout((cfg.operand_cosize,)))
    sQ_decay_raw = storage.q_decay.get_tensor(cute.make_layout((cfg.operand_cosize,)))
    sDecay_scale_raw = storage.decay_scale.get_tensor(cute.make_layout((cfg.decay_scale_cosize,)))
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
    if cutlass.const_expr(cfg.gate_dtype == cutlass.Float32):
        sGate_load_ptr = smem_data_ptr(sGate_raw)
    else:
        sGate_load_ptr = cute.make_ptr(
            cfg.gate_dtype, smem_data_ptr(sGate_raw).toint(), mem_space=SMEM, assumed_align=1024
        )
    sState_trans = SmemTile(
        base=smem_data_ptr(sState_raw).toint(),
        elems_per_stage=((cfg.state_cosize) // (cfg.smem_state_stages)) * bytes_per_element,
        stages=cfg.smem_state_stages,
        leading_byte_offset=STATE_ALT_LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sState = SmemTile(
        base=smem_data_ptr(sState_raw).toint(),
        elems_per_stage=((cfg.state_cosize) // (cfg.smem_state_stages)) * bytes_per_element,
        stages=cfg.smem_state_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
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
    sU = SmemTile(
        base=smem_data_ptr(sU_raw).toint(),
        elems_per_stage=((cfg.b_t * cfg.d_v) // (1)) * bytes_per_element,
        stages=1,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDv = SmemTile(
        base=smem_data_ptr(sDv_raw).toint(),
        elems_per_stage=((cfg.dv_cosize) // (cfg.smem_dv_stages)) * bytes_per_element,
        stages=cfg.smem_dv_stages,
        leading_byte_offset=LEAD,
        stride_byte_offset=STRIDE,
        layout=SWZ,
    )
    sDstate_trans = SmemTile(
        base=smem_data_ptr(sDstate_raw).toint(),
        elems_per_stage=((cfg.d_k * cfg.d_v) // (1)) * bytes_per_element,
        stages=1,
        leading_byte_offset=STATE_ALT_LEAD,
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
            for stage in cutlass.range_constexpr(cfg.smem_state_stages):
                bars.mb_state_ready[stage].init()
                bars.mb_state_done[stage].init()
                bars.mb_state_cg0_done[stage].init()
            for stage in cutlass.range_constexpr(2):
                bars.mb_state_input_ready[stage].init()
                bars.mb_state_input_done[stage].init()
                bars.mb_state_input_cg2_done[stage].init()
    elif warp_idx == cfg.tcgen05_mma_warp_id:
        if elect_one:
            bars.mb_state_k_acc_ready.init()
            bars.mb_y_input_ready.init()
            bars.mb_u_acc_ready.init()
            bars.mb_u_smem_ready.init()
            bars.mb_du_acc_ready.init()
            bars.mb_du_input_ready.init()
            bars.mb_dy_acc_ready.init()
            bars.mb_neg_beta_dy_input_ready.init()
            bars.mb_dy_smem_ready.init()
            bars.mb_dstate_acc_ready.init()
            bars.mb_dstate_input_ready.init()
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
    elif warp_idx == cfg.register_mma_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_k_decay_inv_ready[stage].init()
                bars.mb_q_decay_ready[stage].init()
                bars.mb_decay_done[stage].init()
                bars.mb_decay_scale_ready[stage].init()
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
            for stage in cutlass.range_constexpr(cfg.smem_da_diag_stages):
                bars.mb_da_diag_ready[stage].init()
                bars.mb_da_diag_done[stage].init()
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
            q_ratio=q_ratio,
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
            sScheduler,
            tmem_base_holder,
            sState_trans,
            sState,
            sK_decay,
            sK_inv,
            sK_inv_trans,
            sDo,
            sDo_trans,
            sQ_decay_trans,
            sK_decay_trans,
            sU,
            sDv,
            sDstate_trans,
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
            sDo_raw,
            sU_raw,
            sIntermediate_raw,
            sDaDiag_raw,
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
            tmem_base_holder,
            warp_idx,
            scale,
            mBeta,
            sBeta_raw,
            sK_inv_raw,
            sGate_raw,
            sGate_load_ptr,
            sK_raw,
            sQ_raw,
            sState_raw,
            sNorm_raw,
            sK_decay_raw,
            sQ_decay_raw,
            sDecay_scale_raw,
            bars,
        )
    elif (
        warp_idx >= cfg.compute_group_2_warp_ids[0]
        and warp_idx <= cfg.compute_group_2_warp_ids[-1]
    ):
        compute2_warp_group(
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
            mGate,
            sGate_raw,
            sNorm_raw,
            sDaDiag_raw,
            sDq_raw,
            sDk_raw,
            sRed1_raw,
            sProj_raw,
            sDstate_raw,
            sDgate_raw,
            scale,
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
            sReduction_raw,
            sBetaM_raw,
            sDecay_scale_raw,
            bars,
        )


@dataclass(frozen=True)
class KdaBpropCfg:
    """Kernel cfg (fixed BT=16 schedule constants; derived TMEM column offsets
    and SMEM buffer cosizes are stamped by ``build_cfg``)."""

    io_dtype: type[cutlass.Numeric]
    gate_dtype: type[cutlass.Numeric]
    use_dstate_in: bool
    use_dstate0: bool
    gate_scale_log2: float
    log_gate: bool
    use_initial_state: bool
    max_active_clusters: int
    d_k: int
    d_v: int
    scheduler_stages: int = 8

    # ---- fixed constants stamped from CFG at build time ------------------------------
    compute_group_0_warp_ids: tuple = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_2_warp_ids: tuple = CFG.COMPUTE_GROUP_2_WARP_IDS
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
    num_regs_compute_group_2: int = CFG.NUM_REGS_COMPUTE_GROUP_2
    num_regs_other: int = CFG.NUM_REGS_OTHER

    # ---- named barrier slots (ids 1-4; 0 is the CTA-wide sync) -----------------------
    cg0_sync_barrier_id: int = 1
    cg0_threads: int = 0
    cg2_sync_barrier_id: int = 2
    cg2_threads: int = 0
    tmem_lifecycle_barrier_id: int = 3
    tmem_user_threads: int = 0
    cg1_sync_barrier_id: int = 4
    cg1_threads: int = 0
    scheduler_consumer_warps: int = 0

    # ---- SMEM / TMEM stage counts + TMEM column offsets ------------------------------
    smem_raw_stages: int = CFG.SMEM_RAW_STAGES
    smem_state_stages: int = CFG.SMEM_S_STAGES
    smem_decay_stages: int = CFG.SMEM_DECAY_STAGES
    smem_intermediate_stages: int = CFG.SMEM_INTERMEDIATE_STAGES
    smem_da_diag_stages: int = CFG.SMEM_DA_DIAG_STAGES
    smem_dq_stages: int = CFG.SMEM_DQ_STAGES
    smem_dk_stages: int = CFG.SMEM_DK_STAGES
    smem_dgate_stages: int = CFG.SMEM_DGATE_STAGES
    smem_dv_stages: int = CFG.SMEM_DV_STAGES
    smem_beta_stages: int = 4
    intermediate_tiles: int = 5
    tmem_dstate_acc_offset: int = 0
    tmem_dstate_input_offset: int = 0
    tmem_state_k_acc_offset: int = 0
    tmem_u_acc_offset: int = 0
    tmem_du_acc_offset: int = 0
    tmem_dy_acc_offset: int = 0
    tmem_dq_acc_offset: int = 0
    tmem_dk_decay_acc_offset: int = 0
    tmem_dk_inv_acc_offset: int = 0
    tmem_dk_restore_acc_offset: int = 0
    tmem_qk_raw_stages: int = 4
    tmem_qraw_input_offset: int = 0
    tmem_kraw_input_offset: int = 0
    tmem_y_input_offset: int = 0
    tmem_du_input_offset: int = 0
    tmem_neg_beta_dy_input_offset: int = 0
    tmem_state_input_offset: int = 0
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- buffer cosizes / TMA bytes stamped at build time ----------------------------
    raw_qk_cosize: int = 0
    raw_v_cosize: int = 0
    raw_gate_cosize: int = 0
    gate_stage_elems: int = 0
    operand_cosize: int = 0
    decay_scale_cosize: int = 0
    intermediate_cosize: int = 0
    state_cosize: int = 0
    dq_cosize: int = 0
    dk_cosize: int = 0
    dgate_cosize: int = 0
    dv_cosize: int = 0

    # ---- TMA transaction bytes per stage ---------------------------------------------
    tma_q_bytes: int = 0
    tma_k_bytes: int = 0
    tma_gate_bytes: int = 0
    tma_do_bytes: int = 0
    tma_v_bytes: int = 0
    tma_state_bytes: int = 0


def _validate_roles(role_ids: tuple, named_barrier_ids: tuple, cta_warps: int) -> int:
    """Check the warp-role map and named-barrier slots; return the scheduler consumer count.

    Every role warp except the TMA producer elects one arrival on ``mb_scheduler_done`` (the
    upstream literal 15 for 16 role warps, 11 for 12), so an edited role map cannot silently
    leave the scheduler ring under- or over-subscribed.
    """
    if len(set(role_ids)) != len(role_ids) or any(not 0 <= w < cta_warps for w in role_ids):
        raise ValueError("warp roles must be disjoint IDs below the CTA warp count")
    if len(set(named_barrier_ids)) != len(named_barrier_ids) or any(
        not 1 <= barrier_id <= 15 for barrier_id in named_barrier_ids
    ):
        raise ValueError("named barrier IDs must be disjoint values in [1, 15]")
    return len(role_ids) - 1


def build_cfg(
    io_dtype: type[cutlass.Numeric],
    gate_dtype: type[cutlass.Numeric],
    *,
    use_dstate_in: bool,
    use_dstate0: bool,
    gate_scale_log2: float,
    log_gate: bool = True,
    use_initial_state: bool,
    max_active_clusters: int,
    d_k: int,
    d_v: int,
) -> KdaBpropCfg:
    cfg = KdaBpropCfg(
        io_dtype=io_dtype,
        gate_dtype=gate_dtype,
        use_dstate_in=use_dstate_in,
        use_dstate0=use_dstate0,
        gate_scale_log2=gate_scale_log2,
        log_gate=log_gate,
        use_initial_state=use_initial_state,
        max_active_clusters=max_active_clusters,
        d_k=d_k,
        d_v=d_v,
    )
    role_ids = (
        *cfg.compute_group_0_warp_ids,
        *cfg.compute_group_1_warp_ids,
        *cfg.compute_group_2_warp_ids,
        cfg.register_mma_warp_id,
        cfg.tcgen05_mma_warp_id,
        cfg.tma_warp_id,
        cfg.epilogue_warp_id,
    )
    for group in (
        cfg.compute_group_0_warp_ids,
        cfg.compute_group_1_warp_ids,
        cfg.compute_group_2_warp_ids,
    ):
        if len(group) != 4 or group != tuple(range(group[0], group[-1] + 1)):
            raise ValueError("the fixed schedule requires contiguous four-warp compute groups")
    scheduler_consumer_warps = _validate_roles(
        role_ids,
        (
            cfg.cg0_sync_barrier_id,
            cfg.cg2_sync_barrier_id,
            cfg.tmem_lifecycle_barrier_id,
            cfg.cg1_sync_barrier_id,
        ),
        16,
    )
    b_t, d_k, d_v, io_bytes = cfg.b_t, cfg.d_k, cfg.d_v, cfg.io_dtype.width // 8
    tmem_dstate_input_offset = d_k
    tmem_state_input_offset = tmem_dstate_input_offset + d_k // 2
    tmem_state_k_acc_offset = tmem_state_input_offset + d_v
    tmem_u_acc_offset = tmem_state_k_acc_offset + b_t
    tmem_du_acc_offset = tmem_u_acc_offset + b_t
    tmem_dq_acc_offset = tmem_du_acc_offset + b_t
    tmem_dk_decay_acc_offset = tmem_dq_acc_offset + b_t
    tmem_dk_inv_acc_offset = tmem_dk_decay_acc_offset + b_t
    tmem_dk_restore_acc_offset = tmem_dk_inv_acc_offset + b_t
    tmem_y_input_offset = tmem_dk_restore_acc_offset + b_t
    tmem_du_input_offset = tmem_y_input_offset + b_t // 2
    tmem_qraw_input_offset = tmem_du_input_offset + b_t // 2
    tmem_kraw_input_offset = tmem_qraw_input_offset + cfg.tmem_qk_raw_stages * (b_t // 2)
    assert tmem_kraw_input_offset + cfg.tmem_qk_raw_stages * (b_t // 2) <= 512
    return replace(
        cfg,
        threads_per_cta=16 * cfg.threads_per_warp,
        scheduler_consumer_warps=scheduler_consumer_warps,
        cg0_threads=len(cfg.compute_group_0_warp_ids) * cfg.threads_per_warp,
        cg2_threads=len(cfg.compute_group_2_warp_ids) * cfg.threads_per_warp,
        cg1_threads=len(cfg.compute_group_1_warp_ids) * cfg.threads_per_warp,
        tmem_user_threads=(
            1
            + len(cfg.compute_group_2_warp_ids)
            + len(cfg.compute_group_1_warp_ids)
            + len(cfg.compute_group_0_warp_ids)
        )
        * cfg.threads_per_warp,
        tmem_dstate_acc_offset=0,
        tmem_dstate_input_offset=tmem_dstate_input_offset,
        tmem_state_input_offset=tmem_state_input_offset,
        tmem_state_k_acc_offset=tmem_state_k_acc_offset,
        tmem_u_acc_offset=tmem_u_acc_offset,
        tmem_du_acc_offset=tmem_du_acc_offset,
        tmem_dy_acc_offset=tmem_state_k_acc_offset,
        tmem_dq_acc_offset=tmem_dq_acc_offset,
        tmem_dk_decay_acc_offset=tmem_dk_decay_acc_offset,
        tmem_dk_inv_acc_offset=tmem_dk_inv_acc_offset,
        tmem_dk_restore_acc_offset=tmem_dk_restore_acc_offset,
        tmem_y_input_offset=tmem_y_input_offset,
        tmem_neg_beta_dy_input_offset=tmem_y_input_offset,
        tmem_du_input_offset=tmem_du_input_offset,
        tmem_qraw_input_offset=tmem_qraw_input_offset,
        tmem_kraw_input_offset=tmem_kraw_input_offset,
        raw_qk_cosize=cfg.smem_raw_stages * d_k * b_t,
        raw_v_cosize=cfg.smem_raw_stages * d_v * b_t,
        raw_gate_cosize=cfg.smem_raw_stages * d_k * b_t,
        gate_stage_elems=(d_k * b_t) * (4 // (cfg.gate_dtype.width // 8)),
        operand_cosize=cfg.smem_decay_stages * b_t * d_k,
        decay_scale_cosize=cfg.smem_decay_stages * d_k,
        intermediate_cosize=cfg.smem_intermediate_stages * cfg.intermediate_tiles * b_t * b_t,
        state_cosize=cfg.smem_state_stages * d_k * d_v,
        dq_cosize=cfg.smem_dq_stages * b_t * d_k,
        dk_cosize=cfg.smem_dk_stages * b_t * d_k,
        dgate_cosize=cfg.smem_dgate_stages * b_t * d_k,
        dv_cosize=cfg.smem_dv_stages * b_t * d_v,
        tma_state_bytes=d_k * d_v * io_bytes,
        tma_q_bytes=d_k * b_t * io_bytes,
        tma_k_bytes=d_k * b_t * io_bytes,
        tma_gate_bytes=d_k * b_t * (cfg.gate_dtype.width // 8),
        tma_do_bytes=d_v * b_t * io_bytes,
        tma_v_bytes=d_v * b_t * io_bytes,
    )


# per-batch runtime TMA descriptors: Q, K, V, Gate, dO, state_checkpoints, dQ, dK, dV, dGate
TENSORMAP_DESC_ARRAYS = 10


# ---------------------------------------------------------------------------


frost_kda_bprop_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_kda_bprop.set_name_prefix("cudnn", remove_cutlass_symbol=False)
