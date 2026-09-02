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

"""Chunked Kimi Delta Attention (KDA) prefill kernel for Blackwell SM100/SM103
(Cutlass DSL), BT=16 tiling with a per-key-channel decay.  Framework-neutral
entry ``chunk_kda_sm100``.

Persistent kernel: the grid is the SM count and every warp role
runs a tile-scheduler loop (``decode_work_item``); a tile is one (batch,
head) sequence, or one split-K work item computing chunks ``[cstart, wend)``
and writing O only for the owned ``[wstart, wend)`` (see
``common/split_k.py``; warmup chunks rebuild the incoming state from
zero).  All ring stage/phase bookkeeping runs on cumulative per-CTA chunk
counters so pipelines flow seamlessly across tiles.

Pipeline (direct CUTLASS primitives, chunk_idx-size 16 KDA schedule):

  load Q/K/V/Gate/Beta
  optional in-kernel L2-norm of Q/K (L2NORM specialization)
  exp2(G), exp2(-G), stage final-token exp2(G) as exp2(G_last)
  super-MMA: KK/A/Neumann inverse (T_inv) + apply Beta
  tcgen05-MMA: state*K/state*Q/U solve/state update/O
  store O and final state

ABI: q `[T, HQ, DK]`, k `[T, HK, DK]`, v `[T, HV, DV]`, gate
`[T, HO, DK]` fp32 (natural-log decay unless SAFE_GATE, which applies the
safe-gate transform from raw gate + a_log/dt_bias), beta `[T, HO]` fp32
post-sigmoid, cu_seqlens int32, states `[N, HO, DV, DK]` (VK, k contiguous).
GQA/GVA head broadcast follows repeat_interleave: source head =
head_idx // (HO // H_x).  State presence, L2NORM, SAFE_GATE, and the head ratios
are compile-time specializations.

Warp assignments (16 warps = 512 threads):
  warps 0-7  : compute group 0 - Gate prefix scan + decay/restore operands
  warps 8-11 : compute group 1 - TMEM value side, O drain, state stores
  warp  12   : super-MMA       - register-MMA KK^T + Neumann inverse
  warp  13   : tcgen05-MMA     - the six state GEMMs + the TMEM lifecycle
  warp  14   : TMA load        - per-chunk input G->S loads
  warp  15   : epilogue        - register-MMA A + the O TMA store

SMEM layout (~221 KB total):
  Buffer                    Bytes  Stages
  Q / K / V raw             32768  8       <-- SW128 TMA ring (io dtype)
  gate raw                  65536  8       <-- fp32 prefix-scan source
  beta                        512  8       <-- fp32 per-token scalars
  K_inv                      8192  2       <-- token-major ldmatrix/tcgen05 B operand
  K decay / Q decay       2x 8192  2       <-- tcgen05 SW128 K-box-major A/B operands
  K restore                  8192  2       <-- tcgen05 B operand for the state update
  state-scale diag          16384  4       <-- per-K-atom decay diagonal blocks
  intermediate (A / T_inv)    2048  2       <-- SW32 16x16 register-MMA tiles
  O staging                  8192  2       <-- W128 output drain

TMEM layout (272 of 512 columns):
  Buffer          Cols     Purpose
  state           0-127    state[DK,DV] fp32 recurrent state
  state inp       128-191  packed b16 A operand view of the state
  q_state_acc      192-223  2-stage state*Q -> O accumulator
  state_k_acc     224-239  state*K fp32 accumulator
  u_acc           240-255  U fp32 accumulator
  y_inp           256-263  packed b16 Y staging: Beta * (V - state*K)
  u_inp           264-271  packed b16 U input (b16 U repack)

GEMM schedule (tcgen05-MMA warp, in issue order per chunk):
  state*K -> state_k_acc
  state*Q -> q_state_acc (the O acc)
  state decay (diag blocks)
  U = Y(T) @ T_inv -> u_acc
  final_state += U @ K_restore
  O += A @ U -> q_state_acc

Requires the public CuTeDSL 4.7 API, including `cutlass.experimental.*`.
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

from .common.split_k import ORDER_CAPACITY, ORDER_ELEMS, ORDER_THREADS, decode_work_item, order_body
from .common.host import get_dtype
from .common.paged_state import resolve_paged_state
from .compat import (
    current_device,
    get_device_properties,
    tensor_device_index,
    validate_tma_tensor,
)
from .common.thd import (
    TENSOR_MAP_QWORDS,
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
from .kda_prefill_config import CFG

from .tile_dsl.barrier import (
    advance,
    MBarrier,
    PipelineState,
    Producer,
)
from .tile_dsl.handles import MmaDesc, SmemTile, smem_data_ptr, tma_slice_runtime_desc
from .tile_dsl.mma import mma_step, mma_ts_step
from .tile_dsl.swizzle import swizzle_box_offset_32b, swizzle_box_offset_128b
from .tile_dsl.tma import tma_load_tile, tma_store_commit, tma_store_tile, tma_store_wait, tma_tensormap_acquire
from .tile_dsl.pointwise import (
    opaque_f32_zero,
    f16x2_to_f32,
    fadd2,
    fmul2,
    ffma2,
    movmatrix_16b,
    mul_f16x2,
    fp32_to_fp16,
    sub_f16x2,
)

LOG2_E: float = 1.4426950408889634


DEFAULT_GATE_LOWER_BOUND: float = -5.0


# Host-side API defaults.


L2_NORM_EPS: float = 1.0e-12


class KdaBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring.  Consumers
    track ``(idx, phase)`` inline; the producer tag selects
    the arrive lowering (``TMA_LOAD``/``MMA_COMMIT``/``THREAD``)."""

    mb_q_ready: MBarrier
    mb_q_done: MBarrier
    mb_k_ready: MBarrier
    mb_k_done: MBarrier
    mb_v_ready: MBarrier
    mb_v_done: MBarrier
    mb_gate_ready: MBarrier
    mb_gate_done: MBarrier

    mb_beta_ready: MBarrier
    mb_beta_done: MBarrier

    mb_o_acc_ready: MBarrier
    mb_o_acc_done: MBarrier
    mb_state_k_acc_ready: MBarrier
    mb_u_acc_ready: MBarrier

    mb_state_inp_ready: MBarrier
    mb_y_inp_ready: MBarrier
    mb_u_inp_ready: MBarrier

    mb_t_inv_ready: MBarrier
    mb_t_inv_done: MBarrier
    mb_a_ready: MBarrier
    mb_a_done: MBarrier
    mb_qk_scale_ready: MBarrier
    mb_state_scale_diag_done: MBarrier
    mb_k_decay_inv_cg0_ready: MBarrier
    mb_decay_tcgen05_done: MBarrier
    mb_decay_super_done: MBarrier
    mb_k_restore_done: MBarrier

    mb_state_acc_done: MBarrier
    mb_tmem_done: MBarrier

    mb_o_tmastg_ready: MBarrier
    mb_o_tmastg_done: MBarrier

    mb_sched_ready: MBarrier
    mb_sched_done: MBarrier


def make_kda_bars(cfg) -> KdaBars:
    """Bars factory.  MUST be called from inside ``kernel`` (allocates the
    mbarrier rings in SMEM ahead of the data buffers)."""

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=8)

    WARP = cfg.threads_per_warp
    CG0_GROUP_THREADS = cfg.cg0_warps_per_group * WARP
    CG1_THREADS = len(cfg.compute_group_1_warp_ids) * WARP
    SCHED_CONSUMER_WARPS = cfg.threads_per_cta // WARP - 1

    return KdaBars(
        mb_q_ready=MBarrier(alloc(cfg.smem_raw_bar_stages), stages=cfg.smem_raw_bar_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_q_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0_GROUP_THREADS, producer=Producer.THREAD),
        mb_k_ready=MBarrier(alloc(cfg.smem_raw_bar_stages), stages=cfg.smem_raw_bar_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_k_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0_GROUP_THREADS, producer=Producer.THREAD),
        mb_v_ready=MBarrier(alloc(cfg.smem_raw_bar_stages), stages=cfg.smem_raw_bar_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_v_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_gate_ready=MBarrier(alloc(cfg.smem_raw_bar_stages), stages=cfg.smem_raw_bar_stages, init_count=1, producer=Producer.TMA_LOAD),
        mb_gate_done=MBarrier(alloc(cfg.smem_raw_stages), stages=cfg.smem_raw_stages, init_count=CG0_GROUP_THREADS, producer=Producer.THREAD),
        mb_beta_ready=MBarrier(alloc(cfg.smem_raw_bar_stages), stages=cfg.smem_raw_bar_stages, init_count=WARP, producer=Producer.THREAD),
        mb_beta_done=MBarrier(alloc(cfg.smem_raw_bar_stages), stages=cfg.smem_raw_bar_stages, init_count=WARP + CG1_THREADS, producer=Producer.THREAD),
        mb_o_acc_ready=MBarrier(alloc(1), stages=1, init_count=1, producer=Producer.MMA_COMMIT),
        mb_o_acc_done=MBarrier(alloc(cfg.tmem_q_state_acc_stages), stages=cfg.tmem_q_state_acc_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_state_k_acc_ready=MBarrier(alloc(1), stages=1, init_count=1, producer=Producer.MMA_COMMIT),
        mb_u_acc_ready=MBarrier(alloc(1), stages=1, init_count=1, producer=Producer.MMA_COMMIT),
        mb_state_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_y_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_u_inp_ready=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_t_inv_ready=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=WARP, producer=Producer.THREAD),
        mb_t_inv_done=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_a_ready=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=WARP, producer=Producer.THREAD),
        mb_a_done=MBarrier(alloc(cfg.smem_intermediate_stages), stages=cfg.smem_intermediate_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_qk_scale_ready=MBarrier(
            alloc(cfg.qk_scale_ready_stages),
            stages=cfg.qk_scale_ready_stages,
            init_count=CG0_GROUP_THREADS,
            producer=Producer.THREAD,
        ),
        mb_state_scale_diag_done=MBarrier(
            alloc(cfg.smem_state_scale_diag_stages),
            stages=cfg.smem_state_scale_diag_stages,
            init_count=1,
            producer=Producer.MMA_COMMIT,
        ),
        mb_k_decay_inv_cg0_ready=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=CG0_GROUP_THREADS, producer=Producer.THREAD),
        mb_decay_tcgen05_done=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_decay_super_done=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=2 * WARP, producer=Producer.THREAD),
        mb_k_restore_done=MBarrier(alloc(cfg.smem_decay_stages), stages=cfg.smem_decay_stages, init_count=1, producer=Producer.MMA_COMMIT),
        mb_state_acc_done=MBarrier(alloc(1), stages=1, init_count=1, producer=Producer.MMA_COMMIT),
        mb_tmem_done=MBarrier(alloc(1), stages=1, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_o_tmastg_ready=MBarrier(alloc(cfg.smem_o_stages), stages=cfg.smem_o_stages, init_count=CG1_THREADS, producer=Producer.THREAD),
        mb_o_tmastg_done=MBarrier(alloc(cfg.smem_o_stages), stages=cfg.smem_o_stages, init_count=WARP, producer=Producer.THREAD),
        # The TMA warp publishes each scheduler slot. Every other warp elects one done arrival.
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
    desc_q_base,
    desc_k_base,
    desc_v_base,
    desc_gate_base,
    bars,
) -> None:
    """TMA-LDG warp role (warp 14): persistent scheduler loop issuing the
    per-chunk Q/K/V/Gate G->S loads."""
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
        tma_loads_per_tile=(cfg.d_k // 64),
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
    raw_index = PipelineState.start(phase=1)
    raw_bar_index = PipelineState.start(phase=0)
    sched_state = PipelineState.start(phase=1)
    packed_tokens = cutlass.Int32(cu_seqlens[n_desc])
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        head_o = head_idx
        head_q = head_idx if cfg.q_ratio == 1 else head_idx // cutlass.Int32(cfg.q_ratio)
        head_k = head_idx if cfg.k_ratio == 1 else head_idx // cutlass.Int32(cfg.k_ratio)
        head_v = head_idx if cfg.v_ratio == 1 else head_idx // cutlass.Int32(cfg.v_ratio)
        slot = batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
        desc_q_slot = (desc_q_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_k_slot = (desc_k_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_v_slot = (desc_v_base + slot).tospace(cutlass.AddressSpace.generic)
        desc_gate_slot = (desc_gate_base + slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_q_slot)
            tma_tensormap_acquire(desc_k_slot)
            tma_tensormap_acquire(desc_v_slot)
            tma_tensormap_acquire(desc_gate_slot)
        use_packed_coords = batch_end == packed_tokens
        for chunk_idx in cutlass.range(cstart, wend, 1, unroll=1):
            chunk_start = chunk_idx * cfg.b_t
            input_chunk_start = (
                batch_start + chunk_start if use_packed_coords else chunk_start
            )

            # ---- Q load ----------------------------------------------------------
            bars.mb_q_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_q_ready[raw_bar_index.idx].arrive(n_bytes=cfg.tma_q_bytes)
            q_slice = tma_slice_runtime_desc(
                desc_q_slot, cutlass.Int32(0), head_q, input_chunk_start
            )
            tma_load_tile(
                sQ_tma[raw_index.idx],
                q_slice,
                bars.mb_q_ready[raw_bar_index.idx].smem_ptr,
                acquire=False,
            )

            # ---- K load ----------------------------------------------------------
            bars.mb_k_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_k_ready[raw_bar_index.idx].arrive(n_bytes=cfg.tma_k_bytes)
            k_slice = tma_slice_runtime_desc(
                desc_k_slot, cutlass.Int32(0), head_k, input_chunk_start
            )
            tma_load_tile(
                sK_tma[raw_index.idx],
                k_slice,
                bars.mb_k_ready[raw_bar_index.idx].smem_ptr,
                acquire=False,
            )

            # ---- Gate load -------------------------------------------------------
            bars.mb_gate_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_gate_ready[raw_bar_index.idx].arrive(n_bytes=cfg.tma_gate_bytes)
            gate_slice = tma_slice_runtime_desc(
                desc_gate_slot, cutlass.Int32(0), head_o, input_chunk_start
            )
            tma_load_tile(
                sGate_tma[raw_index.idx],
                gate_slice,
                bars.mb_gate_ready[raw_bar_index.idx].smem_ptr,
                acquire=False,
            )

            # ---- V load ----------------------------------------------------------
            bars.mb_v_done[raw_index.idx].wait(raw_index.phase)
            if elect_one:
                bars.mb_v_ready[raw_bar_index.idx].arrive(n_bytes=cfg.tma_v_bytes)
            v_slice = tma_slice_runtime_desc(
                desc_v_slot, cutlass.Int32(0), head_v, input_chunk_start
            )
            tma_load_tile(
                sV_tma[raw_index.idx],
                v_slice,
                bars.mb_v_ready[raw_bar_index.idx].smem_ptr,
                acquire=False,
            )

            raw_index = advance(raw_index, cfg.smem_raw_stages)
            raw_bar_index = advance(raw_bar_index, cfg.smem_raw_bar_stages)
        tile_idx, sched_state = sched_publish_next(cfg, bars, sSched, mSched, sched_state, tile_idx, num_ctas)


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
    sK_inv_raw,
    sIntermediate_raw,
    sBeta_raw,
    sK_decay_raw,
    bars,
) -> None:
    """Super-MMA warp role (warp 12): persistent scheduler loop computing the
    Neumann-series T_inv via register MMA."""
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    raw_index = PipelineState.start(phase=0)
    t_inv_free = PipelineState.start(phase=1)
    k_decay_ready = PipelineState.start(phase=0)

    # ---- ldmatrix/stmatrix lane decode -------------------------------------------
    rhs_row_coord = lane % 8 + (cutlass.Int32(8) if (lane // 16) else cutlass.Int32(0))
    rhs_col_offset = cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0)
    lhs_row_coord = lane % 8 + (cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0))
    lhs_col_offset = cutlass.Int32(8) if ((lane // 8) // 2) else cutlass.Int32(0)
    decay_key_mask = cutlass.Int32(8)
    stsm_row_coord = lane & 7
    stsm_col_coord = cutlass.Int32(0)
    if (lane // 8) & 1:
        stsm_row_coord = stsm_row_coord + cutlass.Int32(8)
    if lane // 8 >= 2:
        stsm_col_coord = cutlass.Int32(8)
    stsm_idx = swizzle_box_offset_32b(
        stsm_row_coord,
        stsm_col_coord ^ (cfg.b_t // 2),
        box_rows=cfg.b_t,
    )
    cum_chunk_base = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        num_chunks_tile = wend - cstart  # processed chunks; ring bookkeeping runs on cum_chunk_base + local_chunk_idx
        for local_chunk_idx in cutlass.range(num_chunks_tile, unroll=1):
            cum_chunk = cum_chunk_base + local_chunk_idx
            decay_stage = k_decay_ready.idx
            intermediate_stage = t_inv_free.idx
            sBeta_ptr = smem_data_ptr(sBeta_raw) + raw_index.idx * cfg.b_t
            sK_inv_ptr = smem_data_ptr(sK_inv_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = smem_data_ptr(sK_decay_raw) + decay_stage * (cfg.d_k * cfg.b_t)
            sIntermediate_ptr = smem_data_ptr(sIntermediate_raw) + intermediate_stage * (2 * cfg.b_t * cfg.b_t)

            bars.mb_k_decay_inv_cg0_ready[decay_stage].wait(k_decay_ready.phase)
            k_decay_ready = advance(k_decay_ready, cfg.smem_decay_stages)

            # ---- KK = K_decay @ K_inv^T ------------------------------------------
            kk_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                kk_acc[accum_idx] = cutlass.Float32(0.0)

            for k_block in cutlass.range_constexpr((cfg.d_k // 16)):
                # Load B operand
                k_inv_col = k_block * 16 + rhs_col_offset
                rhs_offset = swizzle_box_offset_128b(
                    rhs_row_coord, k_inv_col, box_rows=cfg.b_t
                )
                rhs_frag = nvvm.ldmatrix(sK_inv_ptr + rhs_offset, 4, nvvm.MMALayout.ROW)
                # Load A operand
                storage_key = (k_block * 16 + lhs_col_offset) ^ decay_key_mask
                lhs_offset = swizzle_box_offset_128b(
                    lhs_row_coord, storage_key, box_rows=cfg.b_t
                )
                kk_lhs_frag = nvvm.ldmatrix(
                    sK_decay_ptr + lhs_offset, 4, nvvm.MMALayout.ROW
                )

                mma_step(
                    kk_acc,
                    (kk_lhs_frag[0], kk_lhs_frag[1], kk_lhs_frag[2], kk_lhs_frag[3]),
                    (rhs_frag[0], rhs_frag[1], rhs_frag[2], rhs_frag[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )

            # ---- L = Beta * tril(KK, -1) fragment --------------------------------
            bars.mb_beta_ready[raw_index.idx].wait(raw_index.phase)
            row_lo = lane // 4
            row_hi = row_lo + cutlass.Int32(8)
            beta_lo = (sBeta_ptr + row_lo).load().to(cutlass.Float32)
            beta_hi = (sBeta_ptr + row_hi).load().to(cutlass.Float32)
            l_regs = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                row_coord = row_hi if cutlass.const_expr(accum_idx % 4 >= 2) else row_lo
                col_coord = (accum_idx // 4) * 8 + 2 * (lane % 4)
                if cutlass.const_expr(accum_idx % 2 == 1):
                    col_coord = col_coord + cutlass.Int32(1)
                l_regs[accum_idx] = kk_acc[accum_idx] if row_coord > col_coord else cutlass.Float32(0.0)
            for pair in cutlass.range_constexpr(4):
                beta_scale = beta_hi if cutlass.const_expr(pair % 2 == 1) else beta_lo
                l_regs[2 * pair], l_regs[2 * pair + 1] = fmul2(l_regs[2 * pair], l_regs[2 * pair + 1], beta_scale, beta_scale)
            bars.mb_beta_done[raw_index.idx].arrive()
            l_a0 = fp32_to_fp16(l_regs[0], l_regs[1], dtype=cfg.io_dtype)
            l_a1 = fp32_to_fp16(l_regs[2], l_regs[3], dtype=cfg.io_dtype)
            l_a2 = fp32_to_fp16(l_regs[4], l_regs[5], dtype=cfg.io_dtype)
            l_a3 = fp32_to_fp16(l_regs[6], l_regs[7], dtype=cfg.io_dtype)
            l_values = cutlass.Vector.from_elements((l_a0, l_a1, l_a2, l_a3), cutlass.Int32).bitcast(cfg.io_dtype).to(cutlass.Float32)

            # ---- T_inv = I - L, then three Neumann doubling rounds ---------------
            tinv_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                row_coord = row_lo
                if cutlass.const_expr(accum_idx % 4 >= 2):
                    row_coord = row_hi
                col_coord = (accum_idx // 4) * 8 + 2 * (lane % 4)
                if cutlass.const_expr(accum_idx % 2 == 1):
                    col_coord = col_coord + cutlass.Int32(1)
                eye = cutlass.Float32(1.0) if row_coord == col_coord else cutlass.Float32(0.0)
                tinv_acc[accum_idx] = eye - l_values[accum_idx]

            lpow_a0, lpow_a1, lpow_a2, lpow_a3 = l_a0, l_a1, l_a2, l_a3
            mov_lpow0, mov_lpow1, mov_lpow2, mov_lpow3 = movmatrix_16b(l_a0), movmatrix_16b(l_a1), movmatrix_16b(l_a2), movmatrix_16b(l_a3)
            for _round in cutlass.range_constexpr(3):
                # ---- Lpow = Lpow @ Lpow ------------------------------------------
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
                # ---- T_inv += T_inv @ Lpow ---------------------------------------
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

            bars.mb_t_inv_done[intermediate_stage].wait(t_inv_free.phase)
            t_inv_free = advance(t_inv_free, cfg.smem_intermediate_stages)
            nvvm.stmatrix(
                sIntermediate_ptr + (cfg.b_t * cfg.b_t) + stsm_idx,
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
            bars.mb_decay_super_done[decay_stage].arrive()
            raw_index = advance(raw_index, cfg.smem_raw_bar_stages)
        cum_chunk_base += num_chunks_tile
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
    tmem_base_slot,
    sIntermediate,
    sK_decay,
    sK_restore,
    sQ_decay,
    sState_scale_diag,
    bars,
) -> None:
    """tcgen05-MMA warp role (warp 13): persistent scheduler loop issuing
    every tcgen05 GEMM and owning the TMEM lifecycle."""
    elect_one = nvvm.elect_sync()
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
    nvvm.tcgen05_alloc(tmem_base_slot, cutlass.Int32(512), group=nvvm.CTAGroup.CTA_1)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    state_inp_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_inp_offset, cutlass.Int8)
    state_dsts = tuple(nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_acc_offset + k * 16, cutlass.Float32) for k in range(cfg.d_k // 16))
    state_k_acc_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_k_acc_offset, cutlass.Float32)
    u_acc_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_u_acc_offset, cutlass.Float32)
    y_inp_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_y_inp_offset, cutlass.Int8)
    u_inp_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_u_inp_offset, cutlass.Int8)
    state_dst_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_state_acc_offset, cutlass.Float32)
    state_inp_index = PipelineState.start(phase=0)
    y_inp_index = PipelineState.start(phase=0)
    u_inp_index = PipelineState.start(phase=0)
    qk_scale_index = PipelineState.start(phase=0)
    k_decay_ready = PipelineState.start(phase=0)
    intermediate_ready = PipelineState.start(phase=0)
    o_acc_free = PipelineState.start(phase=1)

    # ---- chunk-invariant GEMM descriptors ----------------------------------------
    bpe = cfg.io_dtype.width // 8
    idesc_acc = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.b_t,
        m_dim=cfg.d_v,
        b_major=0,
    )
    idesc_diag = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=16,
        m_dim=cfg.d_v,
        b_major=0,
    )
    idesc_final_state = nvvm.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=cfg.io_dtype,
        b_dtype=cfg.io_dtype,
        n_dim=cfg.d_k,
        m_dim=cfg.d_v,
        b_major=1,
    )
    bmm_state_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.d_k,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_acc,
        kind=nvvm.Tcgen05MMAKind.F16,
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
    bmm_intermediate_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.b_t,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=False,
        cta_group=1,
        idesc=idesc_acc,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    bmm_final_state_desc = MmaDesc(
        M=cfg.d_v,
        N=cfg.d_k,
        K=cfg.b_t,
        bpe_a=bpe,
        bpe_b=bpe,
        tile_k_hw=16,
        btranspose=True,
        cta_group=1,
        idesc=idesc_final_state,
        kind=nvvm.Tcgen05MMAKind.F16,
    )
    STATE_A_SEG = bmm_state_desc.sps_B * bmm_state_desc.tmem_advance_A
    STATE_B_SEG = bmm_state_desc.smem_subtile_B >> 4
    cum_chunk_base = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        num_chunks_tile = wend - cstart
        for local_chunk_idx in cutlass.range(num_chunks_tile, unroll=1):
            cum_chunk = cum_chunk_base + local_chunk_idx
            have_state = cutlass.Boolean(True) if cutlass.const_expr(cfg.use_initial_state) else local_chunk_idx > 0
            q_state_acc_stage = o_acc_free.idx
            decay_stage = k_decay_ready.idx
            state_scale_diag_stage = qk_scale_index.idx
            intermediate_stage = intermediate_ready.idx
            sK_decay_stage = sK_decay[decay_stage]
            sQ_decay_stage = sQ_decay[decay_stage]
            sK_restore_stage = sK_restore[decay_stage]
            sState_scale_diag_stage = sState_scale_diag[state_scale_diag_stage]
            sIntermediate_stage = sIntermediate[intermediate_stage]

            # ---- state_k = S(T) @ K_decay^T --------------------------------------
            bars.mb_k_decay_inv_cg0_ready[decay_stage].wait(k_decay_ready.phase)
            k_decay_ready = advance(k_decay_ready, cfg.smem_decay_stages)
            if have_state:
                bars.mb_state_inp_ready.wait(state_inp_index.phase)
                state_inp_index = advance(state_inp_index, 1)
                desc_k_decay = sK_decay_stage.desc()

                for s in cutlass.range_constexpr(bmm_state_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_state_desc.sps_B):
                        mma_ts_step(
                            bmm_state_desc,
                            state_inp_ptr.subview(s * STATE_A_SEG),
                            desc_k_decay + s * STATE_B_SEG,
                            state_k_acc_ptr,
                            k,
                            cutlass.Boolean(s + k > 0),
                        )

                if elect_one:
                    bars.mb_state_k_acc_ready.arrive(cta_group=1)

            # ---- q_state = state(T) @ Q_decay^T ---------------------------------
            bars.mb_qk_scale_ready[qk_scale_index.idx].wait(qk_scale_index.phase)
            bars.mb_o_acc_done[q_state_acc_stage].wait(o_acc_free.phase)
            o_acc_free = advance(o_acc_free, cfg.tmem_q_state_acc_stages)
            q_state_acc_ptr = nvvm.make_tmem_ptr(tmem_base + cfg.tmem_q_state_acc_offset + q_state_acc_stage * cfg.b_t, cutlass.Float32)
            if have_state:
                desc_q_decay = sQ_decay_stage.desc()
                for s in cutlass.range_constexpr(bmm_state_desc.num_subtiles_B):
                    for k in cutlass.range_constexpr(bmm_state_desc.sps_B):
                        mma_ts_step(
                            bmm_state_desc,
                            state_inp_ptr.subview(s * STATE_A_SEG),
                            desc_q_decay + s * STATE_B_SEG,
                            q_state_acc_ptr,
                            k,
                            cutlass.Boolean(s + k > 0),
                        )

            if elect_one:
                bars.mb_decay_tcgen05_done[decay_stage].arrive(cta_group=1)

            # ---- S decay = S(T) @ diag(exp2(G_last)) ---------
            if have_state:
                desc_diag = sState_scale_diag_stage.desc()
                for k_block in cutlass.range_constexpr(cfg.d_k // 16):
                    mma_ts_step(
                        bmm_diag_desc,
                        state_inp_ptr.subview(k_block * bmm_diag_desc.tmem_advance_A),
                        desc_diag.advance_start_address(k_block * 256 * 2),
                        state_dsts[k_block],
                        0,
                        cutlass.Boolean(False),
                    )

            if elect_one:
                bars.mb_state_scale_diag_done[state_scale_diag_stage].arrive(cta_group=1)

            # ---- U = Y(T) @ T_inv ------------------------------------------------
            bars.mb_t_inv_ready[intermediate_stage].wait(intermediate_ready.phase)
            bars.mb_y_inp_ready.wait(y_inp_index.phase)
            y_inp_index = advance(y_inp_index, 1)
            d_int = sIntermediate_stage.shifted((cfg.b_t * cfg.b_t)).desc()
            mma_ts_step(bmm_intermediate_desc, y_inp_ptr, d_int, u_acc_ptr, 0, cutlass.Boolean(False))
            if elect_one:
                bars.mb_t_inv_done[intermediate_stage].arrive(cta_group=1)
                bars.mb_u_acc_ready.arrive(cta_group=1)

            # ---- final_state += U(T) @ K_restore ---------------------------------
            bars.mb_u_inp_ready.wait(u_inp_index.phase)
            u_inp_index = advance(u_inp_index, 1)
            desc_k_restore = sK_restore_stage.desc()

            mma_ts_step(bmm_final_state_desc, u_inp_ptr, desc_k_restore, state_dst_ptr, 0, have_state)
            if elect_one:
                bars.mb_k_restore_done[decay_stage].arrive(cta_group=1)
                bars.mb_state_acc_done.arrive(cta_group=1)

            # ---- O += U(T) @ A ---------------------------------------------------
            bars.mb_a_ready[intermediate_stage].wait(intermediate_ready.phase)
            intermediate_ready = advance(intermediate_ready, cfg.smem_intermediate_stages)
            d_int = sIntermediate_stage.desc()
            mma_ts_step(
                bmm_intermediate_desc,
                u_inp_ptr,
                d_int,
                nvvm.make_tmem_ptr(tmem_base + cfg.tmem_q_state_acc_offset + q_state_acc_stage * cfg.b_t, cutlass.Float32),
                0,
                have_state,
            )
            if elect_one:
                bars.mb_o_acc_ready.arrive(cta_group=1)
                bars.mb_a_done[intermediate_stage].arrive(cta_group=1)
            qk_scale_index = advance(qk_scale_index, cfg.smem_state_scale_diag_stages)

        cum_chunk_base += num_chunks_tile
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)
    bars.mb_tmem_done[0].wait(0)
    nvvm.tcgen05_relinquish_alloc_permit(group=nvvm.CTAGroup.CTA_1)
    nvvm.tcgen05_dealloc(
        nvvm.make_tmem_ptr(tmem_base, cutlass.Int8),
        cutlass.Int32(512),
        group=nvvm.CTAGroup.CTA_1,
    )


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
    mO,
    sK_inv_raw,
    sO_raw,
    sIntermediate_raw,
    sQ_decay_raw,
    desc_o_base,
    bars,
) -> None:
    """Epilogue warp role (warp 15): compute causal A and drain O stores."""
    elect_one = nvvm.elect_sync()
    nvvm.setmaxregister(cfg.num_regs_other, nvvm.SetMaxRegisterAction.DECREASE)
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
    qk_scale_index = PipelineState.start(phase=0)
    a_free = PipelineState.start(phase=1)

    # ---- ldmatrix/stmatrix lane decode -------------------------------------------
    rhs_row_coord = lane % 8 + (cutlass.Int32(8) if (lane // 16) else cutlass.Int32(0))
    rhs_col_offset = cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0)
    lhs_row_coord = lane % 8 + (cutlass.Int32(8) if ((lane // 8) % 2) else cutlass.Int32(0))
    lhs_col_offset = cutlass.Int32(8) if ((lane // 8) // 2) else cutlass.Int32(0)
    decay_key_mask = cutlass.Int32(8)
    stsm_row_coord = lane & 7
    stsm_col_coord = cutlass.Int32(0)
    if (lane // 8) & 1:
        stsm_row_coord = stsm_row_coord + cutlass.Int32(8)
    if lane // 8 >= 2:
        stsm_col_coord = cutlass.Int32(8)
    stsm_idx = swizzle_box_offset_32b(
        stsm_row_coord,
        stsm_col_coord ^ (cfg.b_t // 2),
        box_rows=cfg.b_t,
    )
    cum_chunk_base = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        head_o = head_idx
        o_slot = batch_idx * cutlass.Int32(TENSOR_MAP_QWORDS)
        desc_o_slot = (desc_o_base + o_slot).tospace(cutlass.AddressSpace.generic)
        if elect_one:
            tma_tensormap_acquire(desc_o_slot)
        num_chunks_tile = wend - cstart
        for local_chunk_idx in cutlass.range(num_chunks_tile, unroll=1):
            chunk_idx = cstart + local_chunk_idx
            cum_chunk = cum_chunk_base + local_chunk_idx
            decay_stage = cum_chunk % cfg.smem_decay_stages
            intermediate_stage = a_free.idx

            sK_inv_ptr = smem_data_ptr(sK_inv_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sQ_decay_ptr = smem_data_ptr(sQ_decay_raw) + decay_stage * (cfg.d_k * cfg.b_t)
            sIntermediate_ptr = smem_data_ptr(sIntermediate_raw) + intermediate_stage * (2 * cfg.b_t * cfg.b_t)

            bars.mb_qk_scale_ready[qk_scale_index.idx].wait(qk_scale_index.phase)

            # ---- A = Q_decay @ K_inv^T ------------------------------------------
            a_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                a_acc[accum_idx] = cutlass.Float32(0.0)

            for k_block in cutlass.range_constexpr((cfg.d_k // 16)):
                # Load B operand
                k_inv_col = k_block * 16 + rhs_col_offset
                rhs_offset = swizzle_box_offset_128b(
                    rhs_row_coord, k_inv_col, box_rows=cfg.b_t
                )
                rhs_frag = nvvm.ldmatrix(sK_inv_ptr + rhs_offset, 4, nvvm.MMALayout.ROW)
                # Load A operand
                storage_key = (k_block * 16 + lhs_col_offset) ^ decay_key_mask
                lhs_offset = swizzle_box_offset_128b(
                    lhs_row_coord, storage_key, box_rows=cfg.b_t
                )
                a_lhs_frag = nvvm.ldmatrix(
                    sQ_decay_ptr + lhs_offset, 4, nvvm.MMALayout.ROW
                )

                mma_step(
                    a_acc,
                    (a_lhs_frag[0], a_lhs_frag[1], a_lhs_frag[2], a_lhs_frag[3]),
                    (rhs_frag[0], rhs_frag[1], rhs_frag[2], rhs_frag[3]),
                    k_step=0,
                    M=16,
                    N=16,
                    ab_dtype=cfg.io_dtype,
                )

            for accum_idx in cutlass.range_constexpr(8):
                row_coord = lane // 4
                if cutlass.const_expr(accum_idx % 4 >= 2):
                    row_coord = row_coord + cutlass.Int32(8)
                col_coord = (accum_idx // 4) * 8 + 2 * (lane % 4)
                if cutlass.const_expr(accum_idx % 2 == 1):
                    col_coord = col_coord + cutlass.Int32(1)
                a_acc[accum_idx] = a_acc[accum_idx] if row_coord >= col_coord else cutlass.Float32(0.0)

            bars.mb_a_done[intermediate_stage].wait(a_free.phase)
            a_free = advance(a_free, cfg.smem_intermediate_stages)
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
            bars.mb_decay_super_done[decay_stage].arrive()
            qk_scale_index = advance(qk_scale_index, cfg.qk_scale_ready_stages)

            # ---- O drain ---------------------------------------------------------
            if local_chunk_idx > 0:
                output_chunk = chunk_idx - cutlass.Int32(1)
                output_chunk_start = output_chunk * cfg.b_t
                o_stage = (cum_chunk - cutlass.Int32(1)) % cfg.smem_o_stages
                bars.mb_o_tmastg_ready[o_stage].wait(((cum_chunk - cutlass.Int32(1)) // cfg.smem_o_stages) % 2)
                o_slice = tma_slice_runtime_desc(desc_o_slot, cutlass.Int32(0), head_o, output_chunk_start)
                if output_chunk >= wstart:
                    tma_store_tile(sO_tma[o_stage], o_slice, acquire=False)
                    tma_store_commit()
                tma_store_wait(0)
                bars.mb_o_tmastg_done[o_stage].arrive()

        # ---- last computed chunk drain (always owned: it is wend - 1) ------------
        if num_chunks_tile > 0:
            output_chunk = wend - cutlass.Int32(1)
            last_cum_chunk = cum_chunk_base + num_chunks_tile - cutlass.Int32(1)
            output_chunk_start = output_chunk * cfg.b_t
            o_stage = last_cum_chunk % cfg.smem_o_stages
            bars.mb_o_tmastg_ready[o_stage].wait((last_cum_chunk // cfg.smem_o_stages) % 2)
            o_slice = tma_slice_runtime_desc(desc_o_slot, cutlass.Int32(0), head_o, output_chunk_start)
            tma_store_tile(sO_tma[o_stage], o_slice, acquire=False)
            tma_store_commit()
            tma_store_wait(0)
            bars.mb_o_tmastg_done[o_stage].arrive()
        cum_chunk_base += num_chunks_tile
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)


@cute.jit
def gate_scale(cfg, raw_gate: cutlass.Float32) -> cutlass.Float32:
    """Map raw gate to the log2-domain decay increment used by KDA."""

    if cutlass.const_expr(cfg.safe_gate):
        half = cutlass.Float32(0.5)
        sigmoid = cute.math.tanh(raw_gate * half, approx=True) * half + half
        return cfg.gate_scale_log2 * sigmoid
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
    warp_idx,
    mQ,
    mA_log,
    mDt_bias,
    sK_inv_raw,
    sGate_raw,
    mBeta,
    sBeta_raw,
    sK_raw,
    sQ_raw,
    sK_decay_raw,
    sK_restore_raw,
    sQ_decay_raw,
    sState_scale_diag_raw,
    bars,
) -> None:
    """CG0 warp role (warps 0-7, two ping-pong groups): persistent scheduler
    loop for the Gate prefix scan and decay/restore operand materialization."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_0, nvvm.SetMaxRegisterAction.INCREASE)
    cg0_warp = warp_idx - cfg.compute_group_0_warp_ids[0]
    cg0_group_id = cg0_warp // cfg.cg0_warps_per_group
    cg0_local_warp = cg0_warp % cfg.cg0_warps_per_group
    prefix_dim = cg0_local_warp * cfg.threads_per_warp + lane
    cg0_a_log_exp = cutlass.Float32(1.0)
    cg0_dt_bias_value = cutlass.Float32(0.0)
    cum_chunk_base = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    opaque_one = opaque_f32_zero() + cutlass.Float32(1.0)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        head_o = head_idx
        num_chunks_tile = wend - cstart
        if cutlass.const_expr(cfg.safe_gate):
            if num_chunks_tile > 0:
                cg0_a_log_exp = cute.math.exp2(mA_log[head_o].to(cutlass.Float32) * LOG2_E, fastmath=True)
                cg0_dt_bias_value = mDt_bias[head_o, prefix_dim].to(cutlass.Float32)
        # tile entry: both ping-pong groups inherit each other's delivery proofs (parity-swap guard)
        nvvm.barrier_cta_sync(cfg.cg0_tile_entry_barrier_id, thread_count=cfg.cg0_group_count * cfg.cg0_threads_per_group)
        group_cum_chunk_start = cum_chunk_base + cutlass.Int32(cg0_group_id)
        diag_ring_idx = group_cum_chunk_start % cutlass.Int32(cfg.smem_state_scale_diag_stages)
        diag_ring_phase = (group_cum_chunk_start // cutlass.Int32(cfg.smem_state_scale_diag_stages)) % cutlass.Int32(2)
        for local_chunk_idx in cutlass.range(cg0_group_id, num_chunks_tile, cfg.cg0_group_count, unroll=1):
            chunk_idx = cstart + local_chunk_idx
            cum_chunk = cum_chunk_base + local_chunk_idx
            chunk_start = chunk_idx * cfg.b_t
            decay_stage = cum_chunk % cfg.smem_decay_stages
            raw_stage = cum_chunk % cfg.smem_raw_stages
            raw_bar_stage = cum_chunk % cfg.smem_raw_bar_stages
            state_scale_diag_stage = diag_ring_idx
            qk_scale_ready_stage = state_scale_diag_stage
            sQ_ptr = smem_data_ptr(sQ_raw) + raw_stage * (cfg.d_k * cfg.b_t)
            sK_ptr = smem_data_ptr(sK_raw) + raw_stage * (cfg.d_k * cfg.b_t)
            sGate_ptr = smem_data_ptr(sGate_raw) + raw_stage * (cfg.d_k * cfg.b_t)
            sK_inv_ptr = smem_data_ptr(sK_inv_raw) + decay_stage * (cfg.b_t * cfg.d_k)
            sK_decay_ptr = smem_data_ptr(sK_decay_raw) + decay_stage * (cfg.d_k * cfg.b_t)
            sQ_decay_ptr = smem_data_ptr(sQ_decay_raw) + decay_stage * (cfg.d_k * cfg.b_t)
            sK_restore_ptr = smem_data_ptr(sK_restore_raw) + decay_stage * (cfg.d_k * cfg.b_t)
            sState_scale_diag_ptr = smem_data_ptr(sState_scale_diag_raw) + state_scale_diag_stage * ((cfg.d_k // 16) * 256)

            # ---- Beta scalars ---------------------------------------------------
            if cg0_local_warp == 0:
                bars.mb_beta_done[raw_bar_stage].wait(((cum_chunk // cfg.smem_raw_bar_stages) + 1) % 2)
                if lane < cfg.b_t:
                    token_idx = chunk_idx * cfg.b_t + lane
                    beta_value = cutlass.Float32(0.0)
                    if token_idx < seqlen_b:
                        beta_value = mBeta[batch_start + token_idx, head_o].to(cutlass.Float32)
                        if cutlass.const_expr(cfg.beta_sigmoid):
                            half = cutlass.Float32(0.5)
                            beta_value = (cute.math.tanh(beta_value * half, approx=True) * half + half).to(mBeta.element_type).to(cutlass.Float32)
                    sBeta_raw[raw_bar_stage * cfg.b_t + lane] = beta_value
                bars.mb_beta_ready[raw_bar_stage].arrive()
            bars.mb_gate_ready[raw_bar_stage].wait((cum_chunk // cfg.smem_raw_bar_stages) % 2)

            row_group_start = cg0_local_warp * (cfg.b_t // cfg.cg0_warps_per_group)
            lane_row_group = lane // 8
            lane_in_row_group = lane - lane_row_group * 8
            decay_row = row_group_start + lane_row_group
            decay_key_mask = cutlass.Int32(8)

            prefix_dim = cg0_local_warp * cfg.threads_per_warp + lane

            # ---- Gate prefix scan -----------------------------------------------
            gate_raw = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
            for row in cutlass.range_constexpr(cfg.b_t):
                prefix_idx = swizzle_box_offset_128b(
                    row,
                    prefix_dim,
                    box_rows=cfg.b_t,
                    elem_bytes=4,
                )
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
                prefix0, row_pair_sum = fadd2(prefix_acc, gate0, gate0, gate1)
                prefix1 = prefix_acc + row_pair_sum
                g_prefix_regs[row0] = prefix0
                g_prefix_regs[row1] = prefix1
                prefix_acc = prefix1

            # ---- exp2(G): stage prefixes + final-token decay ---------------------
            for row in cutlass.range_constexpr(cfg.b_t):
                g_prefix_regs[row] = cute.math.exp2(g_prefix_regs[row], fastmath=True)

            exp_g_last = g_prefix_regs[cfg.b_t - 1]
            for row in cutlass.range_constexpr(cfg.b_t):
                prefix_idx = swizzle_box_offset_128b(
                    row,
                    prefix_dim,
                    box_rows=cfg.b_t,
                    elem_bytes=4,
                )
                (sGate_ptr + prefix_idx).store(g_prefix_regs[row])

            # ---- state-scale diag: stage exp2(G_last) decay blocks ---------------
            bars.mb_state_scale_diag_done[state_scale_diag_stage].wait(diag_ring_phase ^ cutlass.Int32(1))
            block = prefix_dim // cutlass.Int32(16)
            coord = prefix_dim - block * cutlass.Int32(16)
            storage_col = coord ^ cutlass.Int32((cfg.b_t // 2))
            diag_idx = block * cutlass.Int32(256) + swizzle_box_offset_32b(
                coord,
                storage_col,
                box_rows=16,
            )
            sState_scale_diag_ptr[diag_idx] = exp_g_last.to(cfg.io_dtype)

            nvvm.barrier_cta_sync(cfg.cg0_group_sync_barrier_base_id + cg0_group_id, thread_count=cfg.cg0_threads_per_group)

            bars.mb_q_ready[raw_bar_stage].wait((cum_chunk // cfg.smem_raw_bar_stages) % 2)
            bars.mb_k_ready[raw_bar_stage].wait((cum_chunk // cfg.smem_raw_bar_stages) % 2)
            k_inv_pack = cute.make_rmem_tensor((2 * 4,), cutlass.Int32)
            raw_q_regs = cute.make_rmem_tensor((2 * 8,), cutlass.Float32)
            raw_k_regs = cute.make_rmem_tensor((2 * 8,), cutlass.Float32)

            # ---- optional Q/K L2-norm -------------------------------------------
            if cutlass.const_expr(cfg.l2norm):
                q_sq_even = opaque_f32_zero()
                k_sq_even = opaque_f32_zero()
                q_sq_odd = opaque_f32_zero()
                k_sq_odd = opaque_f32_zero()
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                raw_f16_idx = swizzle_box_offset_128b(
                    decay_row,
                    dim_base,
                    box_rows=cfg.b_t,
                )
                raw_q_frag = (sQ_ptr + raw_f16_idx).load(count=8, alignment=16)
                raw_k_frag = (sK_ptr + raw_f16_idx).load(count=8, alignment=16)
                raw_q_vec_f32 = raw_q_frag.to(cutlass.Float32)
                raw_k_vec_f32 = raw_k_frag.to(cutlass.Float32)
                for dim_offset in cutlass.range_constexpr(8):
                    q_val = raw_q_vec_f32[dim_offset]
                    k_val = raw_k_vec_f32[dim_offset]
                    raw_q_regs[reg_base + dim_offset] = q_val
                    raw_k_regs[reg_base + dim_offset] = k_val
                    if cutlass.const_expr(cfg.l2norm):
                        if cutlass.const_expr(dim_offset % 2 == 0):
                            q_sq_even, k_sq_even = ffma2(q_val, k_val, q_val, k_val, q_sq_even, k_sq_even)
                        else:
                            q_sq_odd, k_sq_odd = ffma2(q_val, k_val, q_val, k_val, q_sq_odd, k_sq_odd)

            q_inv_norm = opaque_one
            k_inv_norm = opaque_one
            if cutlass.const_expr(cfg.l2norm):
                q_sum_sq = q_sq_even + q_sq_odd
                k_sum_sq = k_sq_even + k_sq_odd
                q_sum_sq = q_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, q_sum_sq, 4, 31, kind=nvvm.Shfl.BFLY))
                q_sum_sq = q_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, q_sum_sq, 2, 31, kind=nvvm.Shfl.BFLY))
                q_sum_sq = q_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, q_sum_sq, 1, 31, kind=nvvm.Shfl.BFLY))
                k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 4, 31, kind=nvvm.Shfl.BFLY))
                k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 2, 31, kind=nvvm.Shfl.BFLY))
                k_sum_sq = k_sum_sq + cutlass.Float32(nvvm.shfl_sync(0xFFFFFFFF, k_sum_sq, 1, 31, kind=nvvm.Shfl.BFLY))
                norm_floor_sq = cutlass.Float32(L2_NORM_EPS * L2_NORM_EPS)
                q_inv_norm = cute.math.rsqrt(cute.math.max(q_sum_sq, norm_floor_sq), fastmath=True)
                k_inv_norm = cute.math.rsqrt(cute.math.max(k_sum_sq, norm_floor_sq), fastmath=True)

            # ---- decay/restore operands: exp2(+-G) ------------------------------
            exp_g_regs = cute.make_rmem_tensor((2 * 8,), cutlass.Float32)
            exp_g_last_regs = cute.make_rmem_tensor((2 * 8,), cutlass.Float32)
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                for f32_group in cutlass.range_constexpr(2):
                    f32_dim_base = dim_base + f32_group * 4
                    g_prefix_idx = swizzle_box_offset_128b(
                        decay_row,
                        f32_dim_base,
                        box_rows=cfg.b_t,
                        elem_bytes=4,
                    )
                    exp_g_frag = (sGate_ptr + g_prefix_idx).load(count=4, alignment=16)
                    exp_g_last_idx = swizzle_box_offset_128b(
                        cfg.b_t - 1,
                        f32_dim_base,
                        box_rows=cfg.b_t,
                        elem_bytes=4,
                    )
                    exp_g_last_frag = (sGate_ptr + exp_g_last_idx).load(count=4, alignment=16)
                    f32_reg_base = reg_base + f32_group * 4
                    for j in cutlass.range_constexpr(4):
                        exp_g_regs[f32_reg_base + j] = exp_g_frag[j]
                        exp_g_last_regs[f32_reg_base + j] = exp_g_last_frag[j]

            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8

                # ---- K decay + K_inv operands: K * exp2(+G) and K * exp2(-G) -----
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
                    (
                        k_decay_pack[0],
                        k_decay_pack[1],
                        k_decay_pack[2],
                        k_decay_pack[3],
                    ),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                if cutlass.const_expr(dim_half == 0):
                    operand_done_phase = ((cum_chunk // cfg.smem_decay_stages) + 1) % 2
                    bars.mb_decay_super_done[decay_stage].wait(operand_done_phase)
                    bars.mb_decay_tcgen05_done[decay_stage].wait(operand_done_phase)
                k_inv_swizzled_idx = swizzle_box_offset_128b(
                    decay_row,
                    dim_base,
                    box_rows=cfg.b_t,
                )
                (sK_inv_ptr + k_inv_swizzled_idx).store(k_inv_vec, alignment=16)
                storage_key = dim_base ^ decay_key_mask
                decay_swizzled_idx = swizzle_box_offset_128b(
                    decay_row,
                    storage_key,
                    box_rows=cfg.b_t,
                )
                (sK_decay_ptr + decay_swizzled_idx).store(k_decay_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_k_decay_inv_cg0_ready[decay_stage].arrive()
            bars.mb_q_done[raw_stage].arrive()
            bars.mb_k_done[raw_stage].arrive()
            bars.mb_gate_done[raw_stage].arrive()

            # ---- Q_decay operand: Q * q_inv_norm --------------------------------
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                q_decay_pack = cute.make_rmem_tensor((4,), cutlass.Int32)
                for pair_idx in cutlass.range_constexpr(4):
                    dim0 = pair_idx * 2
                    dim1 = dim0 + 1
                    raw_reg_idx0 = reg_base + dim0
                    raw_reg_idx1 = reg_base + dim1
                    q_value0, q_value1 = fmul2(raw_q_regs[raw_reg_idx0], raw_q_regs[raw_reg_idx1], q_inv_norm, q_inv_norm)
                    q_pair = fp32_to_fp16(q_value0, q_value1, dtype=cfg.io_dtype)
                    exp_g_pair = fp32_to_fp16(exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1], dtype=cfg.io_dtype)
                    q_decay_pack[pair_idx] = mul_f16x2(q_pair, exp_g_pair, cfg.io_dtype)

                q_decay_vec = cutlass.Vector.from_elements(
                    (
                        q_decay_pack[0],
                        q_decay_pack[1],
                        q_decay_pack[2],
                        q_decay_pack[3],
                    ),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                storage_key = dim_base ^ decay_key_mask
                decay_swizzled_idx = swizzle_box_offset_128b(
                    decay_row,
                    storage_key,
                    box_rows=cfg.b_t,
                )
                (sQ_decay_ptr + decay_swizzled_idx).store(q_decay_vec, alignment=16)

            # ---- K_restore operand: K_inv * exp_g_last --------------------------
            bars.mb_k_restore_done[decay_stage].wait(((cum_chunk // cfg.smem_decay_stages + 1) % 2))
            for dim_half in cutlass.range_constexpr(2):
                dim_base = dim_half * (cfg.d_k // 2) + lane_in_row_group * 8
                reg_base = dim_half * 8
                k_restore_pack = cute.make_rmem_tensor((4,), cutlass.Int32)
                for pair_idx in cutlass.range_constexpr(4):
                    dim0 = pair_idx * 2
                    dim1 = dim0 + 1
                    exp_g_last_pair = fp32_to_fp16(exp_g_last_regs[reg_base + dim0], exp_g_last_regs[reg_base + dim1], dtype=cfg.io_dtype)
                    k_restore_pack[pair_idx] = mul_f16x2(k_inv_pack[dim_half * 4 + pair_idx], exp_g_last_pair, cfg.io_dtype)
                storage_row = decay_row ^ (cfg.b_t // 2)
                k_restore_idx = swizzle_box_offset_128b(
                    storage_row,
                    dim_base,
                    box_rows=cfg.b_t,
                )
                k_restore_vec = cutlass.Vector.from_elements(
                    (
                        k_restore_pack[0],
                        k_restore_pack[1],
                        k_restore_pack[2],
                        k_restore_pack[3],
                    ),
                    cutlass.Int32,
                ).bitcast(cfg.io_dtype)
                (sK_restore_ptr + k_restore_idx).store(k_restore_vec, alignment=16)
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_qk_scale_ready[qk_scale_ready_stage].arrive()
            diag_ring_idx = diag_ring_idx + cutlass.Int32(cfg.cg0_group_count)
            wrapped = diag_ring_idx >= cutlass.Int32(cfg.smem_state_scale_diag_stages)
            diag_ring_idx = diag_ring_idx - cutlass.Int32(cfg.smem_state_scale_diag_stages) if wrapped else diag_ring_idx
            diag_ring_phase = diag_ring_phase ^ (cutlass.Int32(1) if wrapped else cutlass.Int32(0))
        cum_chunk_base += num_chunks_tile
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
    tmem_base_slot,
    warp_idx,
    mState_out,
    mState_init,
    mStateIndices,
    mHasInitialState,
    mO,
    sO_raw,
    sBeta_raw,
    sV_raw,
    scale,
    bars,
) -> None:
    """CG1 warp role (warps 8-11): persistent scheduler loop for the
    value-side TMEM staging, output drain, and state stores."""
    nvvm.setmaxregister(cfg.num_regs_compute_group_1, nvvm.SetMaxRegisterAction.INCREASE)
    sO_ptr = smem_data_ptr(sO_raw)
    nvvm.barrier_cta_sync(cfg.tmem_lifecycle_barrier_id, thread_count=cfg.tmem_user_threads)
    tmem_base = tmem_base_slot.load()
    tmem_col = tmem_base & 0xFFFF
    tmem_row = tmem_base >> 16
    tmem_subpartition = warp_idx % (cfg.d_v // cfg.threads_per_warp)
    ov_token_coord = (lane // 16) * 8 + (lane & 7)
    ov_col_coord = ((lane // 8) & 1) * 8
    row_id = tmem_row + tmem_subpartition * cfg.threads_per_warp
    value_dim = tmem_subpartition * cfg.threads_per_warp + lane
    value_dim_base = tmem_subpartition * cfg.threads_per_warp
    row_addr = row_id << 16
    row16_addr = (row_id + 16) << 16
    st_row_addr = tmem_row << 16
    st_row16_addr = (tmem_row + 16) << 16
    state_col_id = tmem_col + cfg.tmem_state_acc_offset
    packed_col_id = tmem_col + cfg.tmem_state_inp_offset
    state_k_col_id = tmem_col + cfg.tmem_state_k_acc_offset
    y_inp_col_id = tmem_col + cfg.tmem_y_inp_offset
    u_acc_addr = row_addr + tmem_col + cfg.tmem_u_acc_offset
    u_inp_addr = st_row_addr + tmem_col + cfg.tmem_u_inp_offset
    q_state_col_base = tmem_col + cfg.tmem_q_state_acc_offset
    ov_swz_off0 = swizzle_box_offset_128b(
        ov_token_coord,
        value_dim_base + ov_col_coord,
        box_rows=cfg.b_t,
    )
    ov_swz_off = swizzle_box_offset_128b(
        ov_token_coord,
        value_dim_base + 16 + ov_col_coord,
        box_rows=cfg.b_t,
    )
    state_k_acc_index = PipelineState.start(phase=0)
    u_acc_index = PipelineState.start(phase=0)
    o_acc_index = PipelineState.start(phase=0)
    state_upd_index = PipelineState.start(phase=0)
    raw_index = PipelineState.start(phase=0)
    raw_bar_index = PipelineState.start(phase=0)  # even-depth ready/beta-ring slot (decoupled from the data ring)
    cum_chunk_base = cutlass.Int32(0)
    sched_state = PipelineState.start(phase=0)
    tile_idx = cutlass.Int32(bidx)
    while tile_idx < total_tiles:
        batch_idx, head_idx, batch_start, batch_end, seqlen_b, num_chunks_b, _wstart, wend, cstart, cend = decode_work_item(cfg, tile_idx, mWorkItems)
        head_o = head_idx
        num_chunks_tile = wend - cstart
        state_slot, state_valid, seed_from_pool, clear_empty = resolve_paged_state(
            batch_idx, mStateIndices, mHasInitialState
        )

        if num_chunks_tile > 0:
            # ---- first chunk: seed state TMEM from mState_init ----------
            seed_from_initial_state = cstart == 0
            if cutlass.const_expr(mStateIndices is not None):
                seed_from_initial_state = seed_from_initial_state and seed_from_pool
            if cutlass.const_expr(mState_init is not None):
                if seed_from_initial_state:
                    seed_vw = 16 // (mState_init.element_type.width // 8)
                    seed_src = (mState_init.iterator + mState_init.layout((state_slot, head_o, value_dim, 0))).raw_ptr()
                    for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                        state_block = cute.make_rmem_tensor((32,), cutlass.Float32)
                        for g in cutlass.range_constexpr(32 // seed_vw):
                            seed_chunk = (seed_src + key_block_start + g * seed_vw).load(count=seed_vw, alignment=16)
                            for t in cutlass.range_constexpr(seed_vw):
                                state_block[g * seed_vw + t] = seed_chunk[t].to(cutlass.Float32)

                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr((row_id << 16) + (tmem_col + cfg.tmem_state_acc_offset + key_block_start), cutlass.Float32),
                            state_block.load(),
                        )
                else:
                    for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                        state_block = cute.make_rmem_tensor((32,), cutlass.Float32)
                        for col in cutlass.range_constexpr(32):
                            state_block[col] = cutlass.Float32(0.0)

                        nvvm.tcgen05_st(
                            "32x32b",
                            nvvm.make_tmem_ptr((row_id << 16) + (tmem_col + cfg.tmem_state_acc_offset + key_block_start), cutlass.Float32),
                            state_block.load(),
                        )
            if cutlass.const_expr(mState_init is not None):
                nvvm.tcgen05_wait("store")
            sV_ptr = smem_data_ptr(sV_raw) + raw_index.idx * (cfg.d_v * cfg.b_t)
            sBeta_ptr = smem_data_ptr(sBeta_raw) + raw_bar_index.idx * cfg.b_t

            # ---- state repack: acc TMEM -> packed b16 TMEM ----------------------
            if cutlass.const_expr(mState_init is not None):
                state_vecs = []
                for sub in cutlass.range_constexpr(cfg.d_k // 16):
                    state_vecs.append(nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + state_col_id + sub * 16, cutlass.Float32), num=16))

                for sub in cutlass.range_constexpr(cfg.d_k // 16):
                    packed_state = cute.make_rmem_tensor((8,), cutlass.Int32)
                    for packed_col in cutlass.range_constexpr(8):
                        source_pair = packed_col ^ 4
                        packed_state[packed_col] = fp32_to_fp16(state_vecs[sub][2 * source_pair], state_vecs[sub][2 * source_pair + 1], dtype=cfg.io_dtype)
                    nvvm.tcgen05_st(
                        "32x32b",
                        nvvm.make_tmem_ptr((tmem_row << 16) + packed_col_id + sub * 8, cutlass.Int8),
                        packed_state.load(),
                    )
                nvvm.tcgen05_wait("store")
                bars.mb_state_inp_ready.arrive()

            # ---- Y staging: Y = Beta * (V - state*K) -----------------------------
            bars.mb_v_ready[raw_bar_index.idx].wait(raw_bar_index.phase)
            raw_v_frag0 = nvvm.ldmatrix(
                sV_ptr + ov_swz_off0,
                4,
                nvvm.MMALayout.COL,
            )
            raw_v_frag1 = nvvm.ldmatrix(
                sV_ptr + ov_swz_off,
                4,
                nvvm.MMALayout.COL,
            )
            bars.mb_beta_ready[raw_bar_index.idx].wait(raw_bar_index.phase)
            if cutlass.const_expr(mState_init is not None):
                bars.mb_state_k_acc_ready.wait(state_k_acc_index.phase)

                state_k_vec0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_addr + state_k_col_id, cutlass.Float32), num=2)
                state_k_vec1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row16_addr + state_k_col_id, cutlass.Float32), num=2)

            beta_pack = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                token0 = (((reg_idx // 2) * 4 + (lane & 3)) ^ 4) * 2
                beta0 = (sBeta_ptr + token0).load().to(cutlass.Float32)
                beta1 = (sBeta_ptr + token0 + 1).load().to(cutlass.Float32)
                beta_pack[reg_idx] = fp32_to_fp16(beta0, beta1, dtype=cfg.io_dtype)
            y_inp_pack0 = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                frag_pair = (reg_idx ^ 2) * 2
                if cutlass.const_expr(mState_init is not None):
                    state_k_val0, state_k_val1 = state_k_vec0[frag_pair], state_k_vec0[frag_pair + 1]
                    state_k_pair = fp32_to_fp16(state_k_val0, state_k_val1, dtype=cfg.io_dtype)
                    diff_pair = sub_f16x2(
                        raw_v_frag0[raw_matrix],
                        state_k_pair,
                        cfg.io_dtype,
                    )
                else:
                    diff_pair = raw_v_frag0[raw_matrix]
                y_inp_pack0[reg_idx] = mul_f16x2(
                    beta_pack[reg_idx],
                    diff_pair,
                    cfg.io_dtype,
                )

            y_inp_pack1 = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                frag_pair = (reg_idx ^ 2) * 2
                if cutlass.const_expr(mState_init is not None):
                    state_k_val0, state_k_val1 = state_k_vec1[frag_pair], state_k_vec1[frag_pair + 1]
                    state_k_pair = fp32_to_fp16(state_k_val0, state_k_val1, dtype=cfg.io_dtype)
                    diff_pair = sub_f16x2(
                        raw_v_frag1[raw_matrix],
                        state_k_pair,
                        cfg.io_dtype,
                    )
                else:
                    diff_pair = raw_v_frag1[raw_matrix]
                y_inp_pack1[reg_idx] = mul_f16x2(
                    beta_pack[reg_idx],
                    diff_pair,
                    cfg.io_dtype,
                )

            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(st_row_addr + y_inp_col_id, cutlass.Int8), y_inp_pack0.load())
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(st_row16_addr + y_inp_col_id, cutlass.Int8), y_inp_pack1.load())
            nvvm.tcgen05_wait("store")
            if cutlass.const_expr(mState_init is not None):
                state_k_acc_index = advance(state_k_acc_index, 1)
            bars.mb_v_done[raw_index.idx].arrive()
            bars.mb_beta_done[raw_bar_index.idx].arrive()
            bars.mb_y_inp_ready.arrive()

            # ---- U repack: u_acc TMEM -> packed b16 U input TMEM ----------------
            bars.mb_u_acc_ready.wait(u_acc_index.phase)
            u_vals = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(u_acc_addr, cutlass.Float32),
                num=cfg.b_t,
            )

            u_inp_pack = cute.make_rmem_tensor((cfg.b_t // 2,), cutlass.Int32)
            for packed_col in cutlass.range_constexpr((cfg.b_t // 2)):
                source_pair = packed_col ^ 4
                token0 = source_pair * 2
                token1 = token0 + 1
                u_inp_pack[packed_col] = fp32_to_fp16(u_vals[token0], u_vals[token1], dtype=cfg.io_dtype)

            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(u_inp_addr, cutlass.Int8),
                u_inp_pack.load(),
            )
            nvvm.tcgen05_wait("store")
            u_acc_index = advance(u_acc_index, 1)
            bars.mb_u_inp_ready.arrive()

            raw_index = advance(raw_index, cfg.smem_raw_stages)
            raw_bar_index = advance(raw_bar_index, cfg.smem_raw_bar_stages)

        for local_chunk_idx in cutlass.range(1, num_chunks_tile, 1, unroll=1):
            cum_chunk = cum_chunk_base + local_chunk_idx
            sV_ptr = smem_data_ptr(sV_raw) + raw_index.idx * (cfg.d_v * cfg.b_t)
            sBeta_ptr = smem_data_ptr(sBeta_raw) + raw_bar_index.idx * cfg.b_t

            prev_cum_chunk = cum_chunk - cutlass.Int32(1)
            prev_o_stage = prev_cum_chunk % cfg.smem_o_stages
            prev_q_state_acc_stage = prev_cum_chunk % cfg.tmem_q_state_acc_stages
            prev_o_stage_base = prev_o_stage * (cfg.b_t * cfg.d_v)

            # ---- state repack: acc TMEM -> packed b16 TMEM ----------------------
            bars.mb_state_acc_done.wait(state_upd_index.phase)
            state_upd_index = advance(state_upd_index, 1)
            state_vecs = []
            for sub in cutlass.range_constexpr(cfg.d_k // 16):
                state_vecs.append(nvvm.tcgen05_ld("32x32b", nvvm.make_tmem_ptr(row_addr + state_col_id + sub * 16, cutlass.Float32), num=16))

            for sub in cutlass.range_constexpr(cfg.d_k // 16):
                packed_state = cute.make_rmem_tensor((8,), cutlass.Int32)
                for packed_col in cutlass.range_constexpr(8):
                    source_pair = packed_col ^ 4
                    packed_state[packed_col] = fp32_to_fp16(state_vecs[sub][2 * source_pair], state_vecs[sub][2 * source_pair + 1], dtype=cfg.io_dtype)
                nvvm.tcgen05_st(
                    "32x32b",
                    nvvm.make_tmem_ptr((tmem_row << 16) + packed_col_id + sub * 8, cutlass.Int8),
                    packed_state.load(),
                )
            nvvm.tcgen05_wait("store")
            bars.mb_state_inp_ready.arrive()

            bars.mb_o_acc_ready.wait(o_acc_index.phase)
            o_acc_index = advance(o_acc_index, 1)
            projection_col_id = q_state_col_base + prev_q_state_acc_stage * cfg.b_t
            loaded_vec0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_addr + projection_col_id, cutlass.Float32), num=2)
            loaded_vec1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row16_addr + projection_col_id, cutlass.Float32), num=2)

            # ---- output drain: O acc TMEM -> scaled b16 SMEM --------------------
            stsm_pack0 = cute.make_rmem_tensor((4,), cutlass.Int32)
            stsm_pack1 = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                scaled0_0, scaled0_1 = fmul2(loaded_vec0[2 * reg_idx], loaded_vec0[2 * reg_idx + 1], scale, scale)
                scaled1_0, scaled1_1 = fmul2(loaded_vec1[2 * reg_idx], loaded_vec1[2 * reg_idx + 1], scale, scale)
                if cutlass.const_expr(mStateIndices is not None):
                    scaled0_0 = scaled0_0 if state_valid else cutlass.Float32(0.0)
                    scaled0_1 = scaled0_1 if state_valid else cutlass.Float32(0.0)
                    scaled1_0 = scaled1_0 if state_valid else cutlass.Float32(0.0)
                    scaled1_1 = scaled1_1 if state_valid else cutlass.Float32(0.0)
                stsm_pack0[reg_idx] = fp32_to_fp16(scaled0_0, scaled0_1, dtype=mO.element_type)
                stsm_pack1[reg_idx] = fp32_to_fp16(scaled1_0, scaled1_1, dtype=mO.element_type)

            bars.mb_o_tmastg_done[prev_o_stage].wait(((prev_cum_chunk // cfg.smem_o_stages) + 1) % 2)
            nvvm.stmatrix(
                sO_ptr + prev_o_stage_base + ov_swz_off0,
                stsm_pack0.load(),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.stmatrix(
                sO_ptr + prev_o_stage_base + ov_swz_off,
                stsm_pack1.load(),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            bars.mb_o_acc_done[prev_q_state_acc_stage].arrive()
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_o_tmastg_ready[prev_o_stage].arrive()

            # ---- Y staging: Y = Beta * (V - state*K) -----------------------------
            bars.mb_v_ready[raw_bar_index.idx].wait(raw_bar_index.phase)
            raw_v_frag0 = nvvm.ldmatrix(
                sV_ptr + ov_swz_off0,
                4,
                nvvm.MMALayout.COL,
            )
            raw_v_frag1 = nvvm.ldmatrix(
                sV_ptr + ov_swz_off,
                4,
                nvvm.MMALayout.COL,
            )
            bars.mb_beta_ready[raw_bar_index.idx].wait(raw_bar_index.phase)
            bars.mb_state_k_acc_ready.wait(state_k_acc_index.phase)

            state_k_vec0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_addr + state_k_col_id, cutlass.Float32), num=2)
            state_k_vec1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row16_addr + state_k_col_id, cutlass.Float32), num=2)

            beta_pack = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                token0 = (((reg_idx // 2) * 4 + (lane & 3)) ^ 4) * 2
                beta0 = (sBeta_ptr + token0).load().to(cutlass.Float32)
                beta1 = (sBeta_ptr + token0 + 1).load().to(cutlass.Float32)
                beta_pack[reg_idx] = fp32_to_fp16(beta0, beta1, dtype=cfg.io_dtype)
            y_inp_pack0 = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                frag_pair = (reg_idx ^ 2) * 2
                state_k_val0, state_k_val1 = state_k_vec0[frag_pair], state_k_vec0[frag_pair + 1]
                state_k_pair = fp32_to_fp16(state_k_val0, state_k_val1, dtype=cfg.io_dtype)
                diff_pair = sub_f16x2(
                    raw_v_frag0[raw_matrix],
                    state_k_pair,
                    cfg.io_dtype,
                )
                y_inp_pack0[reg_idx] = mul_f16x2(
                    beta_pack[reg_idx],
                    diff_pair,
                    cfg.io_dtype,
                )

            y_inp_pack1 = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                raw_matrix = (1 - (reg_idx // 2)) * 2 + (reg_idx & 1)
                frag_pair = (reg_idx ^ 2) * 2
                state_k_val0, state_k_val1 = state_k_vec1[frag_pair], state_k_vec1[frag_pair + 1]
                state_k_pair = fp32_to_fp16(state_k_val0, state_k_val1, dtype=cfg.io_dtype)
                diff_pair = sub_f16x2(
                    raw_v_frag1[raw_matrix],
                    state_k_pair,
                    cfg.io_dtype,
                )
                y_inp_pack1[reg_idx] = mul_f16x2(
                    beta_pack[reg_idx],
                    diff_pair,
                    cfg.io_dtype,
                )

            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(st_row_addr + y_inp_col_id, cutlass.Int8), y_inp_pack0.load())
            nvvm.tcgen05_st("16x128b", nvvm.make_tmem_ptr(st_row16_addr + y_inp_col_id, cutlass.Int8), y_inp_pack1.load())
            nvvm.tcgen05_wait("store")
            state_k_acc_index = advance(state_k_acc_index, 1)
            bars.mb_v_done[raw_index.idx].arrive()
            bars.mb_beta_done[raw_bar_index.idx].arrive()
            bars.mb_y_inp_ready.arrive()

            # ---- U repack: u_acc TMEM -> packed b16 U input TMEM ----------------
            bars.mb_u_acc_ready.wait(u_acc_index.phase)
            u_vals = nvvm.tcgen05_ld(
                "32x32b",
                nvvm.make_tmem_ptr(u_acc_addr, cutlass.Float32),
                num=cfg.b_t,
            )

            u_inp_pack = cute.make_rmem_tensor((cfg.b_t // 2,), cutlass.Int32)
            for packed_col in cutlass.range_constexpr((cfg.b_t // 2)):
                source_pair = packed_col ^ 4
                token0 = source_pair * 2
                token1 = token0 + 1
                u_inp_pack[packed_col] = fp32_to_fp16(u_vals[token0], u_vals[token1], dtype=cfg.io_dtype)

            nvvm.tcgen05_st(
                "32x32b",
                nvvm.make_tmem_ptr(u_inp_addr, cutlass.Int8),
                u_inp_pack.load(),
            )
            nvvm.tcgen05_wait("store")
            u_acc_index = advance(u_acc_index, 1)
            bars.mb_u_inp_ready.arrive()

            raw_index = advance(raw_index, cfg.smem_raw_stages)
            raw_bar_index = advance(raw_bar_index, cfg.smem_raw_bar_stages)

        if num_chunks_tile > 0:
            bars.mb_state_acc_done.wait(state_upd_index.phase)
            state_upd_index = advance(state_upd_index, 1)
            last_cum_chunk = cum_chunk_base + num_chunks_tile - cutlass.Int32(1)
            final_o_stage = last_cum_chunk % cfg.smem_o_stages
            final_q_state_acc_stage = last_cum_chunk % cfg.tmem_q_state_acc_stages
            final_o_stage_base = final_o_stage * (cfg.b_t * cfg.d_v)

            bars.mb_o_acc_ready.wait(o_acc_index.phase)
            o_acc_index = advance(o_acc_index, 1)
            projection_col_id = q_state_col_base + final_q_state_acc_stage * cfg.b_t

            loaded_vec0 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row_addr + projection_col_id, cutlass.Float32), num=2)
            loaded_vec1 = nvvm.tcgen05_ld("16x256b", nvvm.make_tmem_ptr(row16_addr + projection_col_id, cutlass.Float32), num=2)

            # ---- output drain: O acc TMEM -> scaled b16 SMEM --------------------
            stsm_pack0 = cute.make_rmem_tensor((4,), cutlass.Int32)
            stsm_pack1 = cute.make_rmem_tensor((4,), cutlass.Int32)
            for reg_idx in cutlass.range_constexpr(4):
                scaled0_0, scaled0_1 = fmul2(loaded_vec0[2 * reg_idx], loaded_vec0[2 * reg_idx + 1], scale, scale)
                scaled1_0, scaled1_1 = fmul2(loaded_vec1[2 * reg_idx], loaded_vec1[2 * reg_idx + 1], scale, scale)
                if cutlass.const_expr(mStateIndices is not None):
                    scaled0_0 = scaled0_0 if state_valid else cutlass.Float32(0.0)
                    scaled0_1 = scaled0_1 if state_valid else cutlass.Float32(0.0)
                    scaled1_0 = scaled1_0 if state_valid else cutlass.Float32(0.0)
                    scaled1_1 = scaled1_1 if state_valid else cutlass.Float32(0.0)
                stsm_pack0[reg_idx] = fp32_to_fp16(scaled0_0, scaled0_1, dtype=mO.element_type)
                stsm_pack1[reg_idx] = fp32_to_fp16(scaled1_0, scaled1_1, dtype=mO.element_type)

            bars.mb_o_tmastg_done[final_o_stage].wait(((last_cum_chunk // cfg.smem_o_stages) + 1) % 2)
            nvvm.stmatrix(
                sO_ptr + final_o_stage_base + ov_swz_off0,
                stsm_pack0.load(),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            nvvm.stmatrix(
                sO_ptr + final_o_stage_base + ov_swz_off,
                stsm_pack1.load(),
                nvvm.MMALayout.COL,
                shape=nvvm.StoreShape.M8N8,
            )
            bars.mb_o_acc_done[final_q_state_acc_stage].arrive()
            nvvm.fence_proxy("async.shared", space="cta")
            bars.mb_o_tmastg_ready[final_o_stage].arrive()

        owns_final = wend == num_chunks_b
        if cutlass.const_expr(mStateIndices is not None):
            owns_final = owns_final and state_valid

        # ---- final-state drain: state acc TMEM -> GMEM ---------------------------
        if cutlass.const_expr(mState_out is not None):
            if seqlen_b > 0:
                if owns_final:
                    state_vw = 16 // (mState_out.element_type.width // 8)
                    state_dst = (mState_out.iterator + mState_out.layout((state_slot, head_o, value_dim, 0))).raw_ptr()
                    for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                        loaded = nvvm.tcgen05_ld(
                            "32x32b",
                            nvvm.make_tmem_ptr((row_id << 16) + (tmem_col + cfg.tmem_state_acc_offset + key_block_start), cutlass.Float32),
                            num=32,
                        )

                        for g in cutlass.range_constexpr(32 // state_vw):
                            (state_dst + key_block_start + g * state_vw).store(
                                cutlass.Vector.from_elements(
                                    tuple(loaded[g * state_vw + t].to(mState_out.element_type) for t in range(state_vw)),
                                    mState_out.element_type,
                                ),
                                alignment=16,
                            )
            else:
                write_empty = cutlass.Boolean(True)
                if cutlass.const_expr(mStateIndices is not None):
                    write_empty = clear_empty
                if write_empty:
                    for key_block_start in cutlass.range_constexpr(0, cfg.d_k, 32):
                        for col in cutlass.range_constexpr(32):
                            key_dim = key_block_start + col
                            if cutlass.const_expr(mState_init is not None and mStateIndices is None):
                                mState_out[state_slot, head_o, value_dim, key_dim] = mState_init[state_slot, head_o, value_dim, key_dim]
                            else:
                                mState_out[state_slot, head_o, value_dim, key_dim] = cutlass.Float32(0.0).to(mState_out.element_type)
        cum_chunk_base += num_chunks_tile
        tile_idx, sched_state = sched_next_tile(cfg, bars, sSched, sched_state, tile_idx, num_ctas)

    bars.mb_tmem_done[0].arrive()


class KdaPrefillOp:
    """Compile and launch one static mega KDA prefill configuration."""

    def __init__(self, cfg: "KdaCfg", use_int64_offsets: bool = False):
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
            f"kda_mega_prefill_{cfg.io_dtype.__name__.lower()}"
            f"_{cfg.state_dtype.__name__.lower()}_h{cfg.n_heads_out}"
            f"_q{cfg.q_ratio}_k{cfg.k_ratio}_v{cfg.v_ratio}"
            f"_i{int(cfg.use_initial_state)}f{int(cfg.store_final_state)}"
            f"p{int(cfg.paged_state)}m{int(cfg.has_initial_state_mask)}l{int(cfg.l2norm)}"
            f"g{int(cfg.safe_gate)}b{int(cfg.beta_sigmoid)}d{int(cfg.dyn_sched)}"
            f"_sm{cfg.max_active_clusters}_i64{int(self.use_int64_offsets)}{gate_scale}"
        )

    @cute.jit
    def __call__(
        self,
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
        state_indices: cute.Tensor | None,
        has_initial_state: cute.Tensor | None,
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
            k_decay: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.k_decay_cosize],
                cfg.buffer_align_bytes,
            ]
            q_decay: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.q_decay_cosize],
                cfg.buffer_align_bytes,
            ]
            k_restore: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.k_restore_cosize],
                cfg.buffer_align_bytes,
            ]
            intermediate: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.intermediate_cosize],
                cfg.buffer_align_bytes,
            ]
            q: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.q_cosize], cfg.buffer_align_bytes
            ]
            k: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.k_cosize], cfg.buffer_align_bytes
            ]
            v: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.v_cosize], cfg.buffer_align_bytes
            ]
            gate: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cfg.gate_cosize], 1024
            ]
            state_scale_diag: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.state_scale_diag_cosize],
                cfg.buffer_align_bytes,
            ]
            k_inv: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.k_inv_cosize], cfg.buffer_align_bytes
            ]
            output: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.o_cosize], cfg.buffer_align_bytes
            ]

        kernel.set_name_prefix(self.get_name())
        kernel(
            cfg,
            SharedStorage,
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
            state_indices,
            has_initial_state,
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
    mStateIndices: cute.Tensor | None,
    mHasInitialState: cute.Tensor | None,
    mWorkItems: cute.Tensor,
    mCount: cute.Tensor,
    mSched: cute.Tensor | None,
    scale: cutlass.Float32,
) -> None:
    """BT=16 KDA forward kernel (persistent, tile-scheduled)."""

    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    num_ctas = cute.arch.grid_dim()[0]
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane = tidx % cfg.threads_per_warp

    total_tiles = mCount[0]
    if cutlass.const_expr(cfg.dyn_sched):
        assert mSched is not None and mSched.element_type == cutlass.Int32
    assert mQ.element_type == cfg.io_dtype and mK.element_type == cfg.io_dtype and mV.element_type == cfg.io_dtype
    assert mGate.element_type == cutlass.Float32
    beta_expected = cfg.io_dtype if cutlass.const_expr(cfg.beta_sigmoid) else cutlass.Float32
    assert mBeta.element_type == beta_expected
    assert cu_seqlens.element_type in (cutlass.Int32, cutlass.Int64)
    if cutlass.const_expr(cfg.use_initial_state):
        assert mState_init is not None and mState_init.element_type in (cutlass.BFloat16, cutlass.Float32)
    else:
        assert mState_init is None, "mState_init must be None if use_initial_state is False"
    if cutlass.const_expr(cfg.store_final_state):
        assert mState_out is not None and mState_out.element_type in (cutlass.BFloat16, cutlass.Float32)
    else:
        assert mState_out is None, "mState_out must be None if store_final_state is False"
    if cutlass.const_expr(mState_init is not None and mState_out is not None):
        assert mState_init.element_type == mState_out.element_type
    if cutlass.const_expr(cfg.paged_state):
        assert mStateIndices is not None, "mStateIndices must be provided in paged mode"
    else:
        assert mStateIndices is None and mHasInitialState is None
    desc_base_words = tensormap_workspace.iterator.raw_ptr()
    arr_words = n_desc * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_q_base = desc_base_words
    desc_k_base = desc_base_words + arr_words
    desc_v_base = desc_base_words + cutlass.Int32(2) * arr_words
    desc_gate_base = desc_base_words + cutlass.Int32(3) * arr_words
    desc_o_base = desc_base_words + cutlass.Int32(4) * arr_words

    # Barrier/control rings stay declaration-ordered; data buffers share one storage allocation.
    SMEM = cutlass.AddressSpace.smem
    bars = make_kda_bars(cfg)
    tmem_base_slot = cutlass.Array(cutlass.Int32, 1, space=SMEM, alignment=4)
    sSched = cutlass.Array(cutlass.Int32, cfg.sched_stages, space=SMEM, alignment=16)
    storage = cutlass.utils.SmemAllocator().allocate(shared_type)
    sK_decay_raw = storage.k_decay.get_tensor(cute.make_layout((cfg.k_decay_cosize,)))
    sQ_decay_raw = storage.q_decay.get_tensor(cute.make_layout((cfg.q_decay_cosize,)))
    sK_restore_raw = storage.k_restore.get_tensor(cute.make_layout((cfg.k_restore_cosize,)))
    sIntermediate_raw = storage.intermediate.get_tensor(
        cute.make_layout((cfg.intermediate_cosize,))
    )
    sQ_raw = storage.q.get_tensor(cute.make_layout((cfg.q_cosize,)))
    sK_raw = storage.k.get_tensor(cute.make_layout((cfg.k_cosize,)))
    sV_raw = storage.v.get_tensor(cute.make_layout((cfg.v_cosize,)))
    sGate_raw = storage.gate.get_tensor(cute.make_layout((cfg.gate_cosize,)))
    sState_scale_diag_raw = storage.state_scale_diag.get_tensor(
        cute.make_layout((cfg.state_scale_diag_cosize,))
    )
    sK_inv_raw = storage.k_inv.get_tensor(cute.make_layout((cfg.k_inv_cosize,)))
    sO_raw = storage.output.get_tensor(cute.make_layout((cfg.o_cosize,)))
    sBeta_raw = cutlass.Array(
        cutlass.Float32,
        cfg.beta_cosize,
        space=SMEM,
        alignment=cfg.buffer_align_bytes,
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
    sK_restore = SmemTile(
        base=sK_restore_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=cfg.smem_decay_stages,
        leading_byte_offset=(cfg.b_t * (cfg.d_v // 2) * 2),
        stride_byte_offset=(8 * (cfg.d_v // 2) * 2),
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    sState_scale_diag = SmemTile(
        base=sState_scale_diag_raw,
        elems_per_stage=((cfg.d_k // 16) * 256),
        stages=cfg.smem_state_scale_diag_stages,
        leading_byte_offset=16,
        stride_byte_offset=(8 * 16 * 2),
        layout=nvvm.Tcgen05SmemSwizzle.SWIZZLE_32B,
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
    if warp_idx == cfg.tma_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_raw_bar_stages):
                bars.mb_q_ready[stage].init()
                bars.mb_k_ready[stage].init()
                bars.mb_v_ready[stage].init()
                bars.mb_gate_ready[stage].init()
                bars.mb_beta_ready[stage].init()
                bars.mb_beta_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_raw_stages):
                bars.mb_q_done[stage].init()
                bars.mb_k_done[stage].init()
                bars.mb_v_done[stage].init()
                bars.mb_gate_done[stage].init()
    elif warp_idx == cfg.tcgen05_mma_warp_id:
        if elect_one:
            bars.mb_o_acc_ready.init()
            for stage in cutlass.range_constexpr(cfg.tmem_q_state_acc_stages):
                bars.mb_o_acc_done[stage].init()
            bars.mb_state_k_acc_ready.init()
            bars.mb_u_acc_ready.init()
            bars.mb_state_acc_done.init()
            bars.mb_state_inp_ready.init()
            for stage in cutlass.range_constexpr(cfg.smem_state_scale_diag_stages):
                bars.mb_state_scale_diag_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_decay_tcgen05_done[stage].init()
                bars.mb_decay_super_done[stage].init()
                bars.mb_k_restore_done[stage].init()
            bars.mb_y_inp_ready.init()
            bars.mb_u_inp_ready.init()
            bars.mb_tmem_done[0].init()
    elif warp_idx == cfg.super_mma_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_intermediate_stages):
                bars.mb_t_inv_ready[stage].init()
                bars.mb_a_ready[stage].init()
                bars.mb_t_inv_done[stage].init()
                bars.mb_a_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.qk_scale_ready_stages):
                bars.mb_qk_scale_ready[stage].init()
            for stage in cutlass.range_constexpr(cfg.smem_decay_stages):
                bars.mb_k_decay_inv_cg0_ready[stage].init()
    elif warp_idx == cfg.epilogue_warp_id:
        if elect_one:
            for stage in cutlass.range_constexpr(cfg.smem_o_stages):
                bars.mb_o_tmastg_ready[stage].init()
                bars.mb_o_tmastg_done[stage].init()
            for stage in cutlass.range_constexpr(cfg.sched_stages):
                bars.mb_sched_ready[stage].init()
                bars.mb_sched_done[stage].init()
    diag_zero = cfg.io_dtype(0.0)
    for diag_idx in cutlass.range(tidx, cfg.state_scale_diag_cosize, cfg.threads_per_cta, unroll=1):
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
            desc_q_base,
            desc_k_base,
            desc_v_base,
            desc_gate_base,
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
            sSched,
            tmem_base_slot,
            sIntermediate,
            sK_decay,
            sK_restore,
            sQ_decay,
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
            mO,
            sK_inv_raw,
            sO_raw,
            sIntermediate_raw,
            sQ_decay_raw,
            desc_o_base,
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
            warp_idx,
            mQ,
            mA_log,
            mDt_bias,
            sK_inv_raw,
            sGate_raw,
            mBeta,
            sBeta_raw,
            sK_raw,
            sQ_raw,
            sK_decay_raw,
            sK_restore_raw,
            sQ_decay_raw,
            sState_scale_diag_raw,
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
            tmem_base_slot,
            warp_idx,
            mState_out,
            mState_init,
            mStateIndices,
            mHasInitialState,
            mO,
            sO_raw,
            sBeta_raw,
            sV_raw,
            scale,
            bars,
        )


@dataclass
class KdaCfg:
    """Kernel cfg (fixed BT=16 schedule constants; derived TMEM column offsets
    and SMEM buffer cosizes are stamped by ``build_cfg``; per-stage sizes are
    inlined at the use sites).  Owned by ``KdaPrefillOp`` and passed into
    ``kernel`` and every warp body."""

    io_dtype: Type[cutlass.Numeric]
    state_dtype: Type[cutlass.Numeric]
    use_initial_state: bool
    store_final_state: bool
    paged_state: bool
    has_initial_state_mask: bool
    l2norm: bool
    safe_gate: bool
    gate_scale_log2: float
    beta_sigmoid: bool
    q_ratio: int
    k_ratio: int
    v_ratio: int
    n_heads_out: int
    max_active_clusters: int
    dyn_sched: bool = False
    sched_stages: int = CFG.SMEM_SCHED_STAGES

    compute_group_0_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_0_WARP_IDS
    compute_group_1_warp_ids: tuple[int, ...] = CFG.COMPUTE_GROUP_1_WARP_IDS
    super_mma_warp_id: int = CFG.SUPER_MMA_WARP_ID
    tcgen05_mma_warp_id: int = CFG.TCGEN05_MMA_WARP_ID
    tma_warp_id: int = CFG.TMA_WARP_ID
    epilogue_warp_id: int = CFG.EPILOGUE_WARP_ID
    b_t: int = CFG.B_T
    d_k: int = CFG.D_K
    d_v: int = CFG.D_V
    threads_per_warp: int = CFG.THREADS_PER_WARP
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES
    threads_per_cta: int = 0
    cg0_group_count: int = 0  # derived by build_cfg
    cg0_warps_per_group: int = 4
    cg0_threads_per_group: int = 0
    cg0_group_sync_barrier_base_id: int = 1  # CG0 group g syncs on nbar id 1 + g
    cg0_tile_entry_barrier_id: int = 5  # CG0-wide (both groups) work-item entry sync
    tmem_user_threads: int = 0
    tmem_lifecycle_barrier_id: int = 3
    num_regs_compute_group_0: int = CFG.NUM_REGS_COMPUTE_GROUP_0
    num_regs_compute_group_1: int = CFG.NUM_REGS_COMPUTE_GROUP_1
    num_regs_other: int = CFG.NUM_REGS_OTHER

    # ---- SMEM / TMEM ring stage counts -------------------------------------------
    smem_raw_stages: int = CFG.SMEM_RAW_STAGES
    smem_raw_bar_stages: int = 0  # ready/beta-ring mbar depth: raw rounded up to even (CG0 ping-pong parity)
    smem_o_stages: int = CFG.SMEM_O_STAGES
    smem_decay_stages: int = CFG.SMEM_DECAY_STAGES
    smem_intermediate_stages: int = CFG.SMEM_INTERMEDIATE_STAGES
    smem_state_scale_diag_stages: int = CFG.SMEM_STATE_SCALE_DIAG_STAGES
    qk_scale_ready_stages: int = CFG.QK_SCALE_READY_STAGES
    tmem_q_state_acc_stages: int = CFG.TMEM_Q_STATE_ACC_STAGES

    # ---- TMEM column offsets (state doubles as the final_state acc) --------------
    tmem_state_acc_offset: int = 0
    tmem_state_inp_offset: int = 0
    tmem_q_state_acc_offset: int = 0
    tmem_state_k_acc_offset: int = 0
    tmem_u_acc_offset: int = 0
    tmem_y_inp_offset: int = 0
    tmem_u_inp_offset: int = 0

    # ---- SMEM buffer cosizes -----------------------------------------------------
    q_cosize: int = 0
    k_cosize: int = 0
    v_cosize: int = 0
    gate_cosize: int = 0
    beta_cosize: int = 0
    k_inv_cosize: int = 0
    k_decay_cosize: int = 0
    q_decay_cosize: int = 0
    k_restore_cosize: int = 0
    state_scale_diag_cosize: int = 0
    o_cosize: int = 0

    # TMA transaction bytes per stage
    tma_q_bytes: int = 0
    tma_k_bytes: int = 0
    tma_v_bytes: int = 0
    tma_gate_bytes: int = 0
    intermediate_cosize: int = 0


def build_cfg(
    io_dtype: Type[cutlass.Numeric],
    state_dtype: Type[cutlass.Numeric],
    *,
    use_initial_state: bool,
    store_final_state: bool,
    paged_state: bool,
    has_initial_state_mask: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_scale_log2: float,
    beta_sigmoid: bool,
    q_ratio: int,
    k_ratio: int,
    v_ratio: int,
    n_heads_out: int,
    max_active_clusters: int,
    dyn_sched: bool = False,
) -> KdaCfg:
    """Build the per-compile ``KdaCfg`` (io_dtype in {Float16, BFloat16});
    fills the derived TMEM column offsets and SMEM buffer cosizes."""
    if io_dtype not in (cutlass.Float16, cutlass.BFloat16):
        raise ValueError(f"io_dtype={io_dtype} not supported; only Float16 and BFloat16 are supported")
    if paged_state and not (use_initial_state and store_final_state):
        raise ValueError("paged_state requires an aliased input/output state pool")
    cfg = KdaCfg(
        io_dtype=io_dtype,
        state_dtype=state_dtype,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        paged_state=paged_state,
        has_initial_state_mask=has_initial_state_mask,
        l2norm=l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        beta_sigmoid=beta_sigmoid,
        q_ratio=q_ratio,
        k_ratio=k_ratio,
        v_ratio=v_ratio,
        n_heads_out=n_heads_out,
        max_active_clusters=max_active_clusters,
        dyn_sched=dyn_sched,
    )
    cfg.smem_raw_bar_stages = cfg.smem_raw_stages + (cfg.smem_raw_stages % 2)
    role_ids = (
        *cfg.compute_group_0_warp_ids,
        *cfg.compute_group_1_warp_ids,
        cfg.super_mma_warp_id,
        cfg.tcgen05_mma_warp_id,
        cfg.tma_warp_id,
        cfg.epilogue_warp_id,
    )
    if tuple(sorted(role_ids)) != tuple(range(len(role_ids))):
        raise ValueError("warp roles must be disjoint and cover contiguous IDs from zero")
    for group in (cfg.compute_group_0_warp_ids, cfg.compute_group_1_warp_ids):
        if not group or group != tuple(range(group[0], group[-1] + 1)):
            raise ValueError("compute warp groups must be nonempty and use contiguous IDs")
    cfg.threads_per_cta = len(role_ids) * cfg.threads_per_warp
    if cfg.cg0_warps_per_group != 4:
        raise ValueError("the fixed schedule requires four warps per compute group 0 subgroup")
    if len(cfg.compute_group_0_warp_ids) % cfg.cg0_warps_per_group:
        raise ValueError("compute group 0 must divide evenly into ping-pong warp groups")
    cfg.cg0_group_count = len(cfg.compute_group_0_warp_ids) // cfg.cg0_warps_per_group
    if cfg.cg0_group_count != 2:
        raise ValueError("the fixed schedule requires two compute group 0 ping-pong subgroups")
    cfg.cg0_threads_per_group = cfg.cg0_warps_per_group * cfg.threads_per_warp
    cfg.tmem_user_threads = (1 + len(cfg.compute_group_1_warp_ids)) * cfg.threads_per_warp
    named_barrier_ids = (
        *range(
            cfg.cg0_group_sync_barrier_base_id,
            cfg.cg0_group_sync_barrier_base_id + cfg.cg0_group_count,
        ),
        cfg.tmem_lifecycle_barrier_id,
        cfg.cg0_tile_entry_barrier_id,
    )
    if (
        any(not 1 <= barrier_id <= 15 for barrier_id in named_barrier_ids)
        or len(set(named_barrier_ids)) != len(named_barrier_ids)
    ):
        raise ValueError("named barrier IDs must be disjoint values in [1, 15]")
    if cfg.smem_state_scale_diag_stages != cfg.qk_scale_ready_stages:
        raise ValueError("diag and qk-scale ready rings must share their rolling stage")

    cfg.tmem_state_inp_offset = cfg.tmem_state_acc_offset + cfg.d_k
    cfg.tmem_q_state_acc_offset = cfg.tmem_state_inp_offset + (cfg.d_k // 2)
    cfg.tmem_state_k_acc_offset = cfg.tmem_q_state_acc_offset + cfg.tmem_q_state_acc_stages * cfg.b_t
    cfg.tmem_u_acc_offset = cfg.tmem_state_k_acc_offset + cfg.b_t
    cfg.tmem_y_inp_offset = cfg.tmem_u_acc_offset + cfg.b_t
    cfg.tmem_u_inp_offset = cfg.tmem_y_inp_offset + (cfg.b_t // 2)
    assert (cfg.tmem_u_inp_offset + (cfg.b_t // 2)) <= 512

    cfg.q_cosize = cfg.smem_raw_stages * cfg.d_k * cfg.b_t
    cfg.k_cosize = cfg.smem_raw_stages * cfg.d_k * cfg.b_t
    cfg.v_cosize = cfg.smem_raw_stages * cfg.d_v * cfg.b_t
    cfg.gate_cosize = cfg.smem_raw_stages * cfg.d_k * cfg.b_t
    cfg.beta_cosize = cfg.smem_raw_bar_stages * cfg.b_t
    cfg.k_inv_cosize = cfg.smem_decay_stages * cfg.b_t * cfg.d_k
    cfg.k_decay_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.q_decay_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.k_restore_cosize = cfg.smem_decay_stages * cfg.d_k * cfg.b_t
    cfg.state_scale_diag_cosize = cfg.smem_state_scale_diag_stages * (cfg.d_k // 16) * 256
    cfg.o_cosize = cfg.smem_o_stages * cfg.b_t * cfg.d_v
    cfg.intermediate_cosize = cfg.smem_intermediate_stages * 2 * cfg.b_t * cfg.b_t
    cfg.tma_q_bytes = cfg.d_k * cfg.b_t * (cfg.io_dtype.width // 8)
    cfg.tma_k_bytes = cfg.d_k * cfg.b_t * (cfg.io_dtype.width // 8)
    cfg.tma_v_bytes = cfg.d_v * cfg.b_t * (cfg.io_dtype.width // 8)
    cfg.tma_gate_bytes = cfg.d_k * cfg.b_t * 4
    return cfg


TENSORMAP_DESC_ARRAYS = 5  # per-batch runtime TMA descriptors: Q, K, V, Gate, O
TENSORMAP_STATIC_SLOTS = 0


@cute.jit
def build_descs_body(
    widx,
    base_q,
    base_k,
    base_v,
    base_gate,
    base_o,
    desc_ws: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    o: cute.Tensor,
    n_batch: cutlass.Int32,
    q_token_stride: cutlass.Int64,
    k_token_stride: cutlass.Int64,
    v_token_stride: cutlass.Int64,
    gate_token_stride: cutlass.Int64,
    o_token_stride: cutlass.Int64,
) -> None:
    """Per-batch descriptor-array build, one warp per array. Runs inside the
    prologue kernel after its order pass; warps past the array count fall
    through the widx guards."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_q_arr = cute.make_tensor(desc_ws.iterator, cute.make_layout((arr_words,), stride=(1,)))
    desc_k_arr = cute.make_tensor(desc_ws.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_v_arr = cute.make_tensor(desc_ws.iterator + 2 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_gate_arr = cute.make_tensor(desc_ws.iterator + 3 * arr_words, cute.make_layout((arr_words,), stride=(1,)))
    desc_o_arr = cute.make_tensor(desc_ws.iterator + 4 * arr_words, cute.make_layout((arr_words,), stride=(1,)))

    if widx == 0:
        if nvvm.elect_sync():
            emit_seq_load_descs(
                base_q, desc_q_arr, cu_seqlens, q, n_batch, q_token_stride, 2
            )
            nvvm.fence_proxy_release(
                nvvm.MemScope.GPU,
                from_proxy=nvvm.Proxy.GENERIC,
                to_proxy=nvvm.Proxy.TENSORMAP,
            )
    if widx == 1:
        if nvvm.elect_sync():
            emit_seq_load_descs(
                base_k, desc_k_arr, cu_seqlens, k, n_batch, k_token_stride, 2
            )
            nvvm.fence_proxy_release(
                nvvm.MemScope.GPU,
                from_proxy=nvvm.Proxy.GENERIC,
                to_proxy=nvvm.Proxy.TENSORMAP,
            )
    if widx == 2:
        if nvvm.elect_sync():
            emit_seq_load_descs(
                base_v, desc_v_arr, cu_seqlens, v, n_batch, v_token_stride, 2
            )
            nvvm.fence_proxy_release(
                nvvm.MemScope.GPU,
                from_proxy=nvvm.Proxy.GENERIC,
                to_proxy=nvvm.Proxy.TENSORMAP,
            )
    if widx == 3:
        if nvvm.elect_sync():
            emit_seq_load_descs(
                base_gate,
                desc_gate_arr,
                cu_seqlens,
                gate,
                n_batch,
                gate_token_stride,
                2,
            )
            nvvm.fence_proxy_release(
                nvvm.MemScope.GPU,
                from_proxy=nvvm.Proxy.GENERIC,
                to_proxy=nvvm.Proxy.TENSORMAP,
            )
    if widx == 4:
        if nvvm.elect_sync():
            emit_seq_descs(base_o, desc_o_arr, cu_seqlens, o, n_batch, o_token_stride, 2)
            nvvm.fence_proxy_release(nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP)


@cute.kernel
def prologue_kernel(
    order_gen: cutlass.Constexpr[bool],
    has_sched: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    base_q: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_k: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_v: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_gate: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    base_o: cutlass.GridConstant[cuda.tensor_map.TensorMap],
    desc_ws: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    o: cute.Tensor,
    mStaging: cute.Tensor | None,
    mCount: cute.Tensor,
    mWorkItems: cute.Tensor,
    mSched: cute.Tensor | None,
    n_batch: cutlass.Int32,
    q_token_stride: cutlass.Int64,
    k_token_stride: cutlass.Int64,
    v_token_stride: cutlass.Int64,
    gate_token_stride: cutlass.Int64,
    o_token_stride: cutlass.Int64,
) -> None:
    """Single-CTA prologue: LPT-order the work-item table and zero the sched
    rings via :func:`order_body`, then build the per-batch TMA-descriptor
    arrays via :func:`build_descs_body`, one warp per array (the extra warps
    only take part in the order phase)."""
    tidx, _, _ = cute.arch.thread_idx()
    tidx = cutlass.Int32(tidx)
    widx = tidx // cutlass.Int32(32)
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
        base_o,
        desc_ws,
        cu_seqlens,
        q,
        k,
        v,
        gate,
        o,
        n_batch,
        q_token_stride,
        k_token_stride,
        v_token_stride,
        gate_token_stride,
        o_token_stride,
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
    gate: cute.Tensor,
    o: cute.Tensor,
    cu_seqlens: cute.Tensor,
    work_item_staging: cute.Tensor | None,
    work_count: cute.Tensor,
    work_items: cute.Tensor,
    sched_ctr: cute.Tensor | None,
    tensormap_workspace: cute.Tensor,
    stream: cuda_driver.CUstream,
):
    """Order work items and build per-sequence Q/K/V/gate/O descriptors."""
    h_q = q.shape[1]
    h_k = k.shape[1]
    h_v = v.shape[1]
    n_heads_out = gate.shape[1]
    batch_size = cu_seqlens.shape[0] - 1
    d_k = q.shape[2]
    d_v = v.shape[2]
    bpe = io_dtype.width // 8
    tma_granu_elems = 128 // bpe
    seqlen = q.shape[0]

    q_headed = cute.make_tensor(q.iterator, cute.make_layout((d_k, h_q, seqlen), stride=(1, q.stride[1], q.stride[0])))
    k_headed = cute.make_tensor(k.iterator, cute.make_layout((d_k, h_k, seqlen), stride=(1, k.stride[1], k.stride[0])))
    v_headed = cute.make_tensor(v.iterator, cute.make_layout((d_v, h_v, seqlen), stride=(1, v.stride[1], v.stride[0])))
    gate_headed = cute.make_tensor(gate.iterator, cute.make_layout((d_k, n_heads_out, seqlen), stride=(1, gate.stride[1], gate.stride[0])))
    o_headed = cute.make_tensor(o.iterator, cute.make_layout((d_v, n_heads_out, seqlen), stride=(1, o.stride[1], o.stride[0])))

    swz = cuda.TensorMapSwizzle.s128b
    base_q = cuda.create_tensor_map_tiled_from_view(q_headed, box_dims=(tma_granu_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_k = cuda.create_tensor_map_tiled_from_view(k_headed, box_dims=(tma_granu_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_v = cuda.create_tensor_map_tiled_from_view(v_headed, box_dims=(tma_granu_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_gate = cuda.create_tensor_map_tiled_from_view(gate_headed, box_dims=(32, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)
    base_o = cuda.create_tensor_map_tiled_from_view(o_headed, box_dims=(tma_granu_elems, 1, b_t), stride_order=(0, 1, 2), swizzle=swz)

    prologue_kernel(
        order_gen,
        has_sched,
        b_t,
        base_q,
        base_k,
        base_v,
        base_gate,
        base_o,
        tensormap_workspace,
        cu_seqlens,
        q,
        k,
        v,
        gate,
        o,
        work_item_staging,
        work_count,
        work_items,
        sched_ctr,
        cutlass.Int32(batch_size),
        cutlass.Int64(q.stride[0]),
        cutlass.Int64(k.stride[0]),
        cutlass.Int64(v.stride[0]),
        cutlass.Int64(gate.stride[0]),
        cutlass.Int64(o.stride[0]),
    ).launch(grid=(1, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream)


# ---- Torch adapter / host-side compilation ---------------------------------------


@lru_cache(maxsize=None)
def get_compiled_cache(
    io_dtype_str: str,
    state_dtype_str: str,
    HQ: int,
    HK: int,
    HV: int,
    use_initial_state: bool,
    store_final_state: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_lower_bound: float,
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
    use_initial_state: bool,
    store_final_state: bool,
    l2norm: bool,
    safe_gate: bool,
    gate_scale_log2: float,
    beta_sigmoid: bool,
    q_ratio: int,
    k_ratio: int,
    v_ratio: int,
    n_heads_out: int,
    dyn_sched: bool = False,
    use_int64_offsets: bool = False,
    *,
    num_sm: int,
    scale: float,
    paged_state: bool = False,
    has_initial_state_mask: bool = False,
):
    """JIT-compile one fake-tensor TVM-FFI prefill signature."""
    cfg = build_cfg(
        io_dtype,
        state_dtype,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        paged_state=paged_state,
        has_initial_state_mask=has_initial_state_mask,
        l2norm=l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        beta_sigmoid=beta_sigmoid,
        q_ratio=q_ratio,
        k_ratio=k_ratio,
        v_ratio=v_ratio,
        n_heads_out=n_heads_out,
        max_active_clusters=num_sm,
        dyn_sched=dyn_sched,
    )
    op = KdaPrefillOp(cfg, use_int64_offsets)
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    tokens, sequence_entries, sequences, work_rows, sched_entries, workspace_words = (
        sym_int() for _ in range(6)
    )
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
        op,
        tma_tensor(io_dtype, (tokens, n_heads_out // q_ratio, 128)),
        tma_tensor(io_dtype, (tokens, n_heads_out // k_ratio, 128)),
        tma_tensor(io_dtype, (tokens, n_heads_out // v_ratio, 128)),
        tma_tensor(cutlass.Float32, (tokens, n_heads_out, 128)),
        make_compact_signature_tensor(
            cutlass.Float32,
            (n_heads_out,),
            assumed_align=4,
        )
        if safe_gate
        else None,
        tma_tensor(cutlass.Float32, (n_heads_out, 128)) if safe_gate else None,
        make_strided_signature_tensor(
            beta_dtype,
            (tokens, n_heads_out),
            assumed_align=beta_dtype.width // 8,
            use_int64_offsets=use_int64_offsets,
        ),
        make_cu_seqlens_signature(sequence_entries),
        state_input_signature,
        tma_tensor(io_dtype, (tokens, n_heads_out, 128)),
        state_output_signature,
        state_indices_signature,
        state_mask_signature,
        make_work_items_signature(work_rows),
        make_counter_signature(),
        make_counter_signature(sched_entries) if dyn_sched else None,
        make_workspace_signature(workspace_words),
        cutlass.Float32(scale),
    )


def chunk_kda_sm100(
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
    use_qk_l2norm_in_kernel: bool = False,
    safe_gate: bool = False,
    gate_lower_bound: float = DEFAULT_GATE_LOWER_BOUND,
    a_log=None,
    dt_bias=None,
    use_beta_sigmoid_in_kernel: bool = False,
    work_items=None,
    work_count=None,
    sched_ctr=None,
    work_item_scratch=None,
    *,
    tensormap_workspace,
    state_indices=None,
    has_initial_state=None,
) -> None:
    """Execute the Blackwell BT=16 chunked KDA prefill kernel.

    All tensors must be on the same CUDA device with a stride-1 innermost
    dim; outer strides are free (padded / permuted views are read through
    the TMA descriptors and dynamic layouts).

    Args:
        q: ``(total_tokens, HQ, DK)`` float16/bfloat16
        k: ``(total_tokens, HK, DK)`` float16/bfloat16
        v: ``(total_tokens, HV, DV)`` float16/bfloat16
        gate: ``(total_tokens, HO, DK)`` float32.  Natural-log decay unless
              ``safe_gate``, which applies the safe-gate transform
              ``lower_bound * sigmoid(exp(a_log) * (gate + dt_bias))``.
        beta: ``(total_tokens, HO)``.  Post-sigmoid float32, or io-dtype
              logits when ``use_beta_sigmoid_in_kernel``
        output: ``(total_tokens, HO, DV)`` float16/bfloat16, pre-allocated
        cu_seqlens: ``(num_seqs + 1,)`` int32
        initial_state: ``(num_seqs, HO, DV, DK)`` float32/bfloat16, or None
        output_state: ``(num_seqs, HO, DV, DK)`` float32/bfloat16, or None
        scale: attention scale factor (must not be 0)
        use_qk_l2norm_in_kernel: L2-normalize q/k rows inside the kernel
        safe_gate: interpret ``gate`` through the safe-gate transform
        a_log: ``(HO,)`` float32, safe-gate per-head log-amplitude (None = 0)
        dt_bias: ``(HO, DK)`` float32, safe-gate channel bias (None = 0)
        use_beta_sigmoid_in_kernel: ``beta`` holds logits; sigmoid in-kernel
        work_items: ``(max_items, 8)`` int32 work-item table from
            ``common/split_k.py`` (REQUIRED; an uncut table row is the whole
            (b, h) sequence). Each item computes chunks ``[cstart, wend)``
            and writes O only for ``[wstart, wend)``.
        work_count: ``(1,)`` int32 device-side item count (REQUIRED)
        sched_ctr: ``(2,)`` int32 device scratch ``[ticket, done]`` enabling
            the dynamic (work-stealing) tile scheduler; must be zeroed before
            every launch (``build_split_table`` does this when it is passed as
            ``sched_ctr``).  None keeps the static CTA stride.
    """
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("v", v),
        ("gate", gate),
        ("output", output),
    ):
        validate_tma_tensor(name, tensor)
    for name, tensor in (
        ("initial_state", initial_state),
        ("output_state", output_state),
    ):
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
    if output.shape != (tokens, HO, 128):
        raise ValueError("output must have shape (T, HO, 128)")
    use_initial_state = initial_state is not None
    store_final_state = output_state is not None
    paged_state = state_indices is not None
    has_initial_state_mask = has_initial_state is not None
    if paged_state and (initial_state is None or initial_state is not output_state):
        raise ValueError("paged mode requires one aliased input/output state pool")
    if work_items is None or work_count is None:
        raise ValueError("work_items/work_count are required (the split-table stage builds them for every launch)")
    dyn_sched = sched_ctr is not None
    order_gen = work_item_scratch is None

    has_sched = dyn_sched
    if initial_state is not None:
        state_dtype_src = initial_state.dtype
    elif output_state is not None:
        state_dtype_src = output_state.dtype
    else:
        state_dtype_src = "float32"

    for name, h in (("HQ", HQ), ("HK", HK), ("HV", HV)):
        if HO % h != 0:
            raise ValueError(f"{name}={h} must divide sab heads {HO}")
    q_ratio = HO // HQ
    k_ratio = HO // HK
    v_ratio = HO // HV
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
        a_log,
        dt_bias,
        beta,
        output,
        cu_seqlens,
        initial_state,
        output_state,
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
        HQ,
        HK,
        HV,
        use_initial_state,
        store_final_state,
        use_qk_l2norm_in_kernel,
        safe_gate,
        gate_lower_bound,
        use_beta_sigmoid_in_kernel,
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
            use_initial_state,
            store_final_state,
            use_qk_l2norm_in_kernel,
            safe_gate,
            gate_scale_log2,
            use_beta_sigmoid_in_kernel,
            q_ratio,
            k_ratio,
            v_ratio,
            HO,
            dyn_sched,
            use_int64_offsets,
            num_sm=num_sm,
            scale=scale,
            paged_state=paged_state,
            has_initial_state_mask=has_initial_state_mask,
        )

    if "prologue" not in cache:
        sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
        tokens, sequence_entries, work_rows, sched_entries, workspace_words = (
            sym_int() for _ in range(5)
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
            has_sched,
            tma_tensor(io_dtype, (tokens, HQ, 128)),
            tma_tensor(io_dtype, (tokens, HK, 128)),
            tma_tensor(io_dtype, (tokens, HV, 128)),
            tma_tensor(cutlass.Float32, (tokens, HO, 128)),
            tma_tensor(io_dtype, (tokens, HO, 128)),
            make_cu_seqlens_signature(sequence_entries),
            work_items_signature if not order_gen else None,
            make_counter_signature(),
            work_items_signature,
            make_counter_signature(sched_entries) if has_sched else None,
            make_workspace_signature(workspace_words),
            name=(
                f"kda_mega_prefill_prologue_{io_dtype.__name__.lower()}"
                f"_hq{HQ}_hk{HK}_hv{HV}_ho{HO}"
                f"_o{int(order_gen)}d{int(dyn_sched)}_i64{int(use_int64_offsets)}"
            ),
        )
    cache["prologue"](
        q,
        k,
        v,
        gate,
        output,
        cu_seqlens,
        work_item_scratch if not order_gen else None,
        work_count,
        work_items,
        sched_ctr,
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
        cu_seqlens,
        initial_state if use_initial_state else None,
        output,
        output_state if store_final_state else None,
        state_indices,
        has_initial_state,
        work_items,
        work_count,
        sched_ctr,
        tensormap_workspace,
        scale,
    )
