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
# attn_gym.linear._delta_rule.cudnn_fe; register arrays are rmem tensors, the SMEM data buffers
# share one SharedStorage, and segment-major swizzle offsets use swizzle_box_offset_128b. The
# frozen cfg launches through KdaPrepOp, whose name encodes every static flag.

"""
KDA prep for SM100 / SM103 / SM107 (Cutlass primitives): the per-(chunk, head) prep records of the
BT = 16 schedule, computed once ahead of the prep-fed prefill (kda_prep_prefill_f16); gdn2_prep_f16
is the GDN-2 twin.

Per chunk c of head h (tokens [16c, 16c + 16) of its sequence), with the prefill's compute-mode
arithmetic:
  g[t, d]        = cumulative log2 gate per key channel (safe-gate / natural-log / linear, padded rows carry none)
  K decay[t, d]  = K_n[t, d] * exp2(+g[t, d])                    record k_decay   (the prefill's sK_decay image)
  Q decay[t, d]  = Q_n[t, d] * exp2(+g[t, d])                    record q_decay   (the prefill's sQ_decay image)
  K inv[t, d]    = K_n[t, d] * exp2(-g[t, d])                    internal
  K restore[t, d]= K_n[t, d] * exp2(g[15, d] - g[t, d])          internal
  diag[d]        = exp2(g[15, d])                                record diag (fp32)
  KK = K decay @ K inv^T;  L = Beta[i] * tril(KK, -1);  inverse = (I + L)^-1 (the 4 -> 8 -> 16
  blockwise inverse)
  inverse_beta[i, j]    = inverse[i, j] * Beta[j]
  T[j, k]      = sum_i inverse_beta[i, j] * K restore[i, k]           record t    (the prefill's sK_restore image)
  A              = tril(Q decay @ K inv^T);  A[t, j] = sum_i A[t, i] * inverse_beta[i, j]   record a (SW32 image, 512 B)
  (K_n / Q_n are the optionally L2-normalized rows.)

The prep-fed prefill then runs S_next = diag .* S + W @ (V - K decay @ S) and O = scale * (Q decay
@ S + A @ (V - K decay @ S)) with every record TMA-loaded raw into the SMEM slot the compute mode
used to fill.

Warp assignments (4 warps = 128 threads per CTA, CTAS_PER_SM CTAs per SM, FlashInfer's kernel_prep
model): phase 1 = the prefill's CG0 operand materialization (the four warps own the whole chunk),
phase 2 = the record math split FlashInfer-style:
  warp 0 : KK / L / inverse / inverse_beta, the chunk's beta row, and the raw TMA loads of item j +
           SMEM_RAW_STAGES once item j's
           phase 1 has consumed the stage
  warp 1 : one key half of T; lane 0 issues the k_decay / q_decay / t TMA stores after phase 2 and
           holds the next item at the
           gate-scan barrier until the record slot's previous stores have been read
  warp 2 : A then A
  warp 3 : the other key half of T
No register redistribution: the launch's occupancy floor (min_blocks_per_mp = CTAS_PER_SM) sets the
register budget.
"""

from dataclasses import dataclass, replace
from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.experimental.cuda.tensor_map as tma
import cutlass.experimental.primitives as nvvm
from cutlass import cute

from attn_gym._backends.cute.compat import SmemAllocator

from ..common.blockwise_inverse import invert_unit_lower_16x16_fragments
from ..common.launch import validate_kernel_domain, validate_named_barriers
from ..common.thd import TENSOR_MAP_QWORDS, emit_seq_descs, emit_tile_seq_descs
from ..tile_dsl.barrier import MBarrier, Producer, launch_dependent_grids, wait_on_dependent_grids
from ..tile_dsl.handles import SmemTile, smem_data_ptr, tma_slice_runtime_desc
from ..tile_dsl.mma import mma_step
from ..tile_dsl.pointwise import (
    fadd2,
    ffma2,
    fmul2,
    fp32_to_fp16,
    movmatrix_16b,
    opaque_f32_zero,
    opaque_i32,
    sigmoid,
)
from ..tile_dsl.swizzle import swizzle_box_offset_128b, swizzle_xor_32b, swizzle_xor_128b
from ..tile_dsl.tma import (
    tma_load_tile,
    tma_store_commit,
    tma_store_tile,
    tma_store_wait,
    tma_tensormap_acquire,
)
from .kda_prep_config import CFG

USE_PDL = True
LOG2_E: float = 1.4426950408889634
DEFAULT_GATE_LOWER_BOUND: float = -5.0
L2_NORM_EPS: float = 1.0e-12


class KdaPrepBars(NamedTuple):
    """Every inter-warp handoff as an ``MBarrier`` over its ring."""

    mb_raw_ready: MBarrier


def make_bars(cfg) -> KdaPrepBars:
    """KdaPrepBars constructor."""
    ONE_LANE = 1
    RS = cfg.smem_raw_stages

    def alloc(n):
        return cutlass.Array(cutlass.Int64, n, space=cutlass.AddressSpace.smem, alignment=16)

    return KdaPrepBars(
        mb_raw_ready=MBarrier(
            alloc(RS), spin=True, stages=RS, init_count=ONE_LANE, producer=Producer.TMA_LOAD
        ),
    )


@cute.jit
def gate_scale(cfg, raw_gate: cutlass.Float32) -> cutlass.Float32:
    """Map raw gate to the log2-domain decay increment used by KDA."""

    if cutlass.const_expr(cfg.safe_gate):
        return cfg.gate_scale_log2 * sigmoid(raw_gate)
    if cutlass.const_expr(cfg.log_gate):
        return raw_gate * cutlass.Float32(LOG2_E)
    return cute.math.log2(raw_gate + cutlass.Float32(1e-10), fastmath=True)


@cute.jit
def compute_warp_group(
    cfg,
    n_items,
    item_begin,
    mRows,
    warp_id,
    lane_idx,
    mA_log,
    mDt_bias,
    mBeta,
    mA,
    mDiag,
    sQ_raw,
    sK_raw,
    sGate_load_ptr,
    sExchange_raw,
    sK_inv_raw,
    sK_restore_raw,
    sIntermediate_raw,
    sBetaG_raw,
    sK_decay_raw,
    sQ_decay_raw,
    sW_raw,
    bars,
    heads_out,
    q_ratio,
    k_ratio,
    sQ_tma,
    sK_tma,
    sGate_tma,
    sK_decay_tma,
    sQ_decay_tma,
    sW_tma,
    desc_q_base,
    desc_k_base,
    desc_gate_base,
    desc_k_decay_base,
    desc_q_decay_base,
    desc_t_base,
):
    """The CTA's four warps over its items: phase 1 materializes the chunk's operands (the
    prefill's CG0 body), phase 2 folds the records: warp 0 KK -> L -> inverse -> inverse_beta (it
    also gathers the chunk's beta row and issues the raw loads of item j + SMEM_RAW_STAGES), warp 2
    A then A, warps 1 and 3 the two key halves of T (lane 0 of warp 1 issues the record stores)."""
    RS = cfg.smem_raw_stages
    RECS = cfg.smem_record_stages
    barrier_id = cfg.barrier_id
    warp_local = warp_id
    dk_halves = cutlass.const_expr(cfg.d_k // 64)
    channel_rows = cutlass.const_expr(cfg.d_k // cfg.compute_warps)
    store_rows = cutlass.const_expr(cfg.b_t * channel_rows // cfg.threads_per_warp)
    store_row_base = (lane_idx // cutlass.Int32(channel_rows)) * cutlass.Int32(store_rows)
    channel_dim = warp_local * cfg.threads_per_warp + lane_idx
    if cutlass.const_expr(channel_rows < cfg.threads_per_warp):
        channel_dim = warp_local * channel_rows + lane_idx % channel_rows
    opaque_one = opaque_f32_zero() + cutlass.Float32(1.0)
    zero = cutlass.Int32(0)

    sK_inv_ptr = smem_data_ptr(sK_inv_raw)
    sK_restore_ptr = smem_data_ptr(sK_restore_raw)
    sIntermediate_ptr = smem_data_ptr(sIntermediate_raw)
    sBetaG_ptr = smem_data_ptr(sBetaG_raw)

    # ---- phase 1 lane geometry (the prefill's CG0) -----------------------------------
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
    row_group_start = warp_local * (cfg.b_t // cfg.compute_warps)
    lane_row_group = lane_idx // 8
    lane_in_row_group = lane_idx - lane_row_group * 8
    decay_row = row_group_start + lane_row_group

    # ---- phase 2 ldmatrix / stmatrix lane decode (the prefill's warps 12 and 15) -----
    k_inv_row_coord = lane_idx % 8 + (cutlass.Int32(8) if (lane_idx // 16) else cutlass.Int32(0))
    k_inv_col_offset = cutlass.Int32(8) if ((lane_idx // 8) % 2) else cutlass.Int32(0)
    a_row_coord = lane_idx % 8 + (cutlass.Int32(8) if ((lane_idx // 8) % 2) else cutlass.Int32(0))
    a_col_offset = cutlass.Int32(8) if ((lane_idx // 8) // 2) else cutlass.Int32(0)
    inverse_idx = a_row_coord * cfg.b_t + swizzle_xor_32b(a_row_coord, a_col_offset)
    k_inv_frag_offsets = [
        opaque_i32(
            k_inv_row_coord * 64
            + swizzle_xor_128b(k_inv_row_coord, i * 16 + k_inv_col_offset, elem_bytes=2)
        )
        for i in range(4)
    ]
    a_frag_offsets = [
        opaque_i32(
            swizzle_xor_128b(a_row_coord, a_row_coord * 64 + i * 16 + a_col_offset, elem_bytes=2)
        )
        for i in range(4)
    ]
    t_groups = cutlass.const_expr(cfg.d_k // 16)
    t_offsets = [
        opaque_i32(
            (n // 4) * (cfg.b_t * 64)
            + a_row_coord * 64
            + swizzle_xor_128b(a_row_coord, (n % 4) * 16 + a_col_offset, elem_bytes=2)
        )
        for n in range(t_groups)
    ]
    row_lo = lane_idx // 4
    row_hi = row_lo + cutlass.Int32(8)
    col_pair = 2 * (lane_idx % 4)

    # ---- descriptor acquire, one per batch of the block; the first raw stages filled ----
    if n_items > cutlass.Int32(0):
        batch_first = mRows[item_begin // heads_out, 1]
        batch_last = mRows[(item_begin + n_items - cutlass.Int32(1)) // heads_out, 1]
        if warp_local == 0:
            if nvvm.elect_sync():
                for b in cutlass.range(batch_first, batch_last + cutlass.Int32(1), unroll=1):
                    tma_tensormap_acquire(
                        (desc_q_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(
                            cutlass.AddressSpace.generic
                        )
                    )
                    tma_tensormap_acquire(
                        (desc_k_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(
                            cutlass.AddressSpace.generic
                        )
                    )
                    tma_tensormap_acquire(
                        (desc_gate_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(
                            cutlass.AddressSpace.generic
                        )
                    )
            for s0 in cutlass.range(cutlass.min(cutlass.Int32(RS), n_items), unroll=1):
                load_item = item_begin + s0
                load_row = load_item // heads_out
                load_head = load_item - load_row * heads_out.divisor
                load_stage = s0 % cutlass.Int32(RS)
                load_chunk = mRows[load_row, 0]
                load_batch = mRows[load_row, 1] * cutlass.Int32(TENSOR_MAP_QWORDS)
                load_token = load_chunk * cutlass.Int32(cfg.b_t)
                load_desc_q = (desc_q_base + load_batch).tospace(cutlass.AddressSpace.generic)
                load_desc_k = (desc_k_base + load_batch).tospace(cutlass.AddressSpace.generic)
                load_desc_gate = (desc_gate_base + load_batch).tospace(
                    cutlass.AddressSpace.generic
                )
                load_head_q = load_head // q_ratio
                load_head_k = load_head // k_ratio
                if nvvm.elect_sync():
                    bars.mb_raw_ready[load_stage].arrive(
                        n_bytes=cfg.tma_q_bytes + cfg.tma_k_bytes + cfg.tma_gate_bytes
                    )
                tma_load_tile(
                    sQ_tma[load_stage],
                    tma_slice_runtime_desc(load_desc_q, zero, load_head_q, load_token),
                    bars.mb_raw_ready[load_stage].smem_ptr,
                )
                tma_load_tile(
                    sK_tma[load_stage],
                    tma_slice_runtime_desc(load_desc_k, zero, load_head_k, load_token),
                    bars.mb_raw_ready[load_stage].smem_ptr,
                )
                tma_load_tile(
                    sGate_tma[load_stage],
                    tma_slice_runtime_desc(load_desc_gate, zero, load_head, load_token),
                    bars.mb_raw_ready[load_stage].smem_ptr,
                )
        if warp_local == 1:
            if lane_idx == 0:
                for b in cutlass.range(batch_first, batch_last + cutlass.Int32(1), unroll=1):
                    tma_tensormap_acquire(
                        (desc_k_decay_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(
                            cutlass.AddressSpace.generic
                        )
                    )
                    tma_tensormap_acquire(
                        (desc_q_decay_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(
                            cutlass.AddressSpace.generic
                        )
                    )
                    tma_tensormap_acquire(
                        (desc_t_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)).tospace(
                            cutlass.AddressSpace.generic
                        )
                    )
    for j in cutlass.range(n_items):
        s = j
        item = item_begin + s
        r = item // heads_out
        head_idx = item - r * heads_out.divisor
        chunk_idx = mRows[r, 0]
        batch_start = mRows[r, 2]
        batch_seqlen = mRows[r, 3] - batch_start
        chunk_start = chunk_idx * cfg.b_t
        tile_row = batch_start // cutlass.Int32(cfg.b_t) + mRows[r, 1] + chunk_idx
        raw_stage = s % cutlass.Int32(RS)
        raw_parity = (s // cutlass.Int32(RS)) & cutlass.Int32(1)
        rec_slot = j % cutlass.Int32(RECS)
        sQ_ptr = smem_data_ptr(sQ_raw) + raw_stage * (cfg.d_k * cfg.b_t)
        sK_ptr = smem_data_ptr(sK_raw) + raw_stage * (cfg.d_k * cfg.b_t)
        sGate_ptr = sGate_load_ptr + raw_stage * cfg.gate_stage_elems
        if cutlass.const_expr(cfg.gate_dtype == cutlass.Float32):
            sGate_exchange_ptr = smem_data_ptr(sExchange_raw) + raw_stage * (cfg.d_k * cfg.b_t)
        else:
            sGate_exchange_ptr = smem_data_ptr(sExchange_raw)
        sK_decay_ptr = smem_data_ptr(sK_decay_raw) + rec_slot * (cfg.d_k * cfg.b_t)
        sQ_decay_ptr = smem_data_ptr(sQ_decay_raw) + rec_slot * (cfg.d_k * cfg.b_t)
        sW_ptr = smem_data_ptr(sW_raw) + rec_slot * (cfg.d_k * cfg.b_t)

        # ---- safe-gate per-head parameters -------------------------------------------
        a_log_exp = cutlass.Float32(1.0)
        dt_bias_value = cutlass.Float32(0.0)
        if cutlass.const_expr(mA_log is not None):
            a_log_exp = cute.math.exp2(
                mA_log[head_idx].to(cutlass.Float32) * LOG2_E, fastmath=True
            )
        if cutlass.const_expr(mDt_bias is not None):
            dt_bias_value = mDt_bias[head_idx, channel_dim].to(cutlass.Float32)

        # ---- the chunk's beta row: gathered by warp 0, consumed after the operands
        beta_value = cutlass.Float32(0.0)
        if warp_local == 0:
            if lane_idx < cfg.b_t:
                if chunk_start + lane_idx < batch_seqlen:
                    beta_value = mBeta[batch_start + chunk_start + lane_idx, head_idx].to(
                        cutlass.Float32
                    )
                    if cutlass.const_expr(cfg.beta_sigmoid):
                        beta_value = (
                            (sigmoid(beta_value) * (2.0 if cfg.allow_neg_eigval else 1.0))
                            .to(mBeta.element_type)
                            .to(cutlass.Float32)
                        )

        bars.mb_raw_ready[raw_stage].wait(raw_parity)

        # ---- Gate prefix scan ----------------------------------------------------
        gate_raw = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
        if cutlass.const_expr(cfg.gate_dtype == cutlass.Float32):
            for row in cutlass.range_constexpr(cfg.b_t):
                gate_raw[row] = (sGate_ptr + (gate_row_offsets[row % 8] + row * 32)).load()
        else:
            for row in cutlass.range_constexpr(cfg.b_t):
                gate_raw[row] = (
                    (sGate_ptr + (gate_row_offsets[row % 8] + row * 64)).load().to(cutlass.Float32)
                )
        g_prefix_regs = cute.make_rmem_tensor((cfg.b_t,), cutlass.Float32)
        if cutlass.const_expr(cfg.safe_gate):
            for row in cutlass.range_constexpr(cfg.b_t):
                g_prefix_regs[row] = gate_scale(cfg, a_log_exp * (gate_raw[row] + dt_bias_value))
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
        if warp_local == 1:
            if lane_idx == 0:
                tma_store_wait(RECS - 1)
        nvvm.barrier_cta_sync(barrier_id, thread_count=cfg.threads_per_cta)

        # ---- diag record: exp2(g[15, d]) per key channel ---------------------------
        mDiag[tile_row, head_idx, channel_dim] = g_prefix_regs[cfg.b_t - 1]

        k_inv_pack = cute.make_rmem_tensor((dk_halves * 4,), cutlass.Int32)
        k_restore_pack = cute.make_rmem_tensor((dk_halves * 4,), cutlass.Int32)
        raw_q_regs = cute.make_rmem_tensor((dk_halves * 8,), cutlass.Float32)
        raw_k_regs = cute.make_rmem_tensor((dk_halves * 8,), cutlass.Float32)

        # ---- optional Q/K L2-norm ------------------------------------------------
        if cutlass.const_expr(cfg.l2norm):
            q_sq_even = opaque_f32_zero()
            k_sq_even = opaque_f32_zero()
            q_sq_odd = opaque_f32_zero()
            k_sq_odd = opaque_f32_zero()
        for dim_half in cutlass.range_constexpr(dk_halves):
            dim_base = dim_half * 64 + lane_in_row_group * 8
            reg_base = dim_half * 8
            raw_f16_idx = swizzle_box_offset_128b(decay_row, dim_base, box_rows=cfg.b_t)
            raw_q_frag = (sQ_ptr + raw_f16_idx).load(count=8, alignment=16)
            raw_k_frag = (sK_ptr + raw_f16_idx).load(count=8, alignment=16)
            raw_q_vec_f32 = raw_q_frag.to(cutlass.Float32)
            raw_k_vec_f32 = raw_k_frag.to(cutlass.Float32)
            for dim_offset in cutlass.range_constexpr(8):
                raw_q_regs[reg_base + dim_offset] = raw_q_vec_f32[dim_offset]
                raw_k_regs[reg_base + dim_offset] = raw_k_vec_f32[dim_offset]
            if cutlass.const_expr(cfg.l2norm):
                for pair_idx in cutlass.range_constexpr(4):
                    q_val0 = raw_q_vec_f32[2 * pair_idx]
                    q_val1 = raw_q_vec_f32[2 * pair_idx + 1]
                    k_val0 = raw_k_vec_f32[2 * pair_idx]
                    k_val1 = raw_k_vec_f32[2 * pair_idx + 1]
                    q_sq_even, q_sq_odd = ffma2(
                        q_val0, q_val1, q_val0, q_val1, q_sq_even, q_sq_odd
                    )
                    k_sq_even, k_sq_odd = ffma2(
                        k_val0, k_val1, k_val0, k_val1, k_sq_even, k_sq_odd
                    )

        q_inv_norm = opaque_one
        k_inv_norm = opaque_one
        if cutlass.const_expr(cfg.l2norm):
            q_sum_sq = q_sq_even + q_sq_odd
            k_sum_sq = k_sq_even + k_sq_odd
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
                for jj in cutlass.range_constexpr(4):
                    exp_g_regs[f32_reg_base + jj] = exp_g_frag[jj]

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
                    + swizzle_xor_128b(cfg.b_t - 1 ^ f32_segment, f32_segment_dim, elem_bytes=4)
                )
                exp_g_last_frag = (sGate_exchange_ptr + exp_g_last_idx).load(count=4, alignment=16)
                for jj in cutlass.range_constexpr(4):
                    exp_g_last_half[f32_group * 4 + jj] = exp_g_last_frag[jj]
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
            k_inv_swizzled_idx = swizzle_box_offset_128b(decay_row, dim_base, box_rows=cfg.b_t)
            (sK_inv_ptr + k_inv_swizzled_idx).store(k_inv_vec, alignment=16)
            decay_col = dim_base
            decay_segment = decay_col // 64
            decay_swizzled_idx = decay_segment * (cfg.b_t * 64) + swizzle_xor_128b(
                decay_row, decay_row * 64 + decay_col - decay_segment * 64, elem_bytes=2
            )
            (sK_decay_ptr + decay_swizzled_idx).store(k_decay_vec, alignment=16)

        # ---- Q decay operand: Q * q inv norm -------------------------------------
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
                    raw_q_regs[raw_reg_idx0], raw_q_regs[raw_reg_idx1], q_inv_norm, q_inv_norm
                )
                q_decay0, q_decay1 = fmul2(
                    q_value0, q_value1, exp_g_regs[raw_reg_idx0], exp_g_regs[raw_reg_idx1]
                )
                q_decay_pack[pair_idx] = fp32_to_fp16(q_decay0, q_decay1, dtype=cfg.io_dtype)

            q_decay_vec = cutlass.Vector.from_elements(
                (
                    q_decay_pack[0],
                    q_decay_pack[1],
                    q_decay_pack[2],
                    q_decay_pack[3],
                ),
                cutlass.Int32,
            ).bitcast(cfg.io_dtype)
            decay_col = dim_base
            decay_segment = decay_col // 64
            decay_swizzled_idx = decay_segment * (cfg.b_t * 64) + swizzle_xor_128b(
                decay_row, decay_row * 64 + decay_col - decay_segment * 64, elem_bytes=2
            )
            (sQ_decay_ptr + decay_swizzled_idx).store(q_decay_vec, alignment=16)

        # ---- K restore operand store ---------------------------------------------
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

        # ---- the chunk's beta into the group's row; the raw stage is consumed ----------
        if warp_local == 0:
            if lane_idx < cfg.b_t:
                (sBetaG_ptr + lane_idx).store(beta_value)
        nvvm.barrier_cta_sync(barrier_id, thread_count=cfg.threads_per_cta)
        if warp_local == 0:
            if j + cutlass.Int32(RS) < n_items:
                load_index = j + cutlass.Int32(RS)
                load_item = item_begin + load_index
                load_row = load_item // heads_out
                load_head = load_item - load_row * heads_out.divisor
                load_stage = load_index % cutlass.Int32(RS)
                load_chunk = mRows[load_row, 0]
                load_batch = mRows[load_row, 1] * cutlass.Int32(TENSOR_MAP_QWORDS)
                load_token = load_chunk * cutlass.Int32(cfg.b_t)
                load_desc_q = (desc_q_base + load_batch).tospace(cutlass.AddressSpace.generic)
                load_desc_k = (desc_k_base + load_batch).tospace(cutlass.AddressSpace.generic)
                load_desc_gate = (desc_gate_base + load_batch).tospace(
                    cutlass.AddressSpace.generic
                )
                load_head_q = load_head // q_ratio
                load_head_k = load_head // k_ratio
                if nvvm.elect_sync():
                    bars.mb_raw_ready[load_stage].arrive(
                        n_bytes=cfg.tma_q_bytes + cfg.tma_k_bytes + cfg.tma_gate_bytes
                    )
                tma_load_tile(
                    sQ_tma[load_stage],
                    tma_slice_runtime_desc(load_desc_q, zero, load_head_q, load_token),
                    bars.mb_raw_ready[load_stage].smem_ptr,
                )
                tma_load_tile(
                    sK_tma[load_stage],
                    tma_slice_runtime_desc(load_desc_k, zero, load_head_k, load_token),
                    bars.mb_raw_ready[load_stage].smem_ptr,
                )
                tma_load_tile(
                    sGate_tma[load_stage],
                    tma_slice_runtime_desc(load_desc_gate, zero, load_head, load_token),
                    bars.mb_raw_ready[load_stage].smem_ptr,
                )

        # ---- phase 2, warp 0: KK = K decay @ K inv^T, L, T^-1, inverse_beta ----------------
        if warp_local == 0:
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
                    sK_decay_ptr + a_frag_offsets[i % 4] + (i // 4) * (cfg.b_t * 64),
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
            beta_lo = (sBetaG_ptr + row_lo).load()
            beta_hi = (sBetaG_ptr + row_hi).load()
            l_regs = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                row_coord = row_hi if cutlass.const_expr(accum_idx % 4 >= 2) else row_lo
                col_coord = (accum_idx // 4) * 8 + col_pair
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

            # ---- T^-1 = (I + L)^-1 ---------------------------------------------------
            inverse_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            invert_unit_lower_16x16_fragments(cfg, l_regs, inverse_acc, lane_idx)

            # ---- inverse_beta = T^-1 .* Beta[col] into the group's intermediate tile ---------
            for pair in cutlass.range_constexpr(4):
                col_coord = (pair // 2) * 8 + col_pair
                beta_c0 = (sBetaG_ptr + col_coord).load()
                beta_c1 = (sBetaG_ptr + col_coord + cutlass.Int32(1)).load()
                inverse_acc[2 * pair], inverse_acc[2 * pair + 1] = fmul2(
                    inverse_acc[2 * pair], inverse_acc[2 * pair + 1], beta_c0, beta_c1
                )
            nvvm.stmatrix(
                sIntermediate_ptr + inverse_idx,
                [
                    fp32_to_fp16(inverse_acc[0], inverse_acc[1], dtype=cfg.io_dtype),
                    fp32_to_fp16(inverse_acc[2], inverse_acc[3], dtype=cfg.io_dtype),
                    fp32_to_fp16(inverse_acc[4], inverse_acc[5], dtype=cfg.io_dtype),
                    fp32_to_fp16(inverse_acc[6], inverse_acc[7], dtype=cfg.io_dtype),
                ],
                nvvm.MMALayout.ROW,
                shape=nvvm.StoreShape.M8N8,
            )

        # ---- phase 2, warp 2: A = tril(Q decay @ K inv^T) ------------------------------
        a_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
        for accum_idx in cutlass.range_constexpr(8):
            a_acc[accum_idx] = cutlass.Float32(0.0)
        if warp_local == 2:
            for i in cutlass.range_constexpr(cfg.d_k // 16):
                k_inv_frag = nvvm.ldmatrix(
                    sK_inv_ptr + k_inv_frag_offsets[i % 4] + (i // 4) * (cfg.b_t * 64),
                    4,
                    nvvm.MMALayout.ROW,
                )
                q_decay_frag = nvvm.ldmatrix(
                    sQ_decay_ptr + a_frag_offsets[i % 4] + (i // 4) * (cfg.b_t * 64),
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
                row_coord = row_lo
                if cutlass.const_expr(accum_idx % 4 >= 2):
                    row_coord = row_hi
                col_coord = (accum_idx // 4) * 8 + col_pair
                if cutlass.const_expr(accum_idx % 2 == 1):
                    col_coord = col_coord + cutlass.Int32(1)
                a_acc[accum_idx] = (
                    a_acc[accum_idx] if row_coord >= col_coord else cutlass.Float32(0.0)
                )
        # ---- phase 2, warps 1 and 3: K restore fragments transposed while warp 0 inverts ----
        kr_b = [cutlass.Int32(0) for _ in range(4 * (t_groups // 2))]
        for half in cutlass.range_constexpr(2):
            if warp_local == 1 + 2 * half:
                for n_local in cutlass.range_constexpr(t_groups // 2):
                    kr = nvvm.ldmatrix(
                        sK_restore_ptr + t_offsets[half * (t_groups // 2) + n_local],
                        4,
                        nvvm.MMALayout.ROW,
                    )
                    kr_b[4 * n_local] = movmatrix_16b(kr[0])
                    kr_b[4 * n_local + 1] = movmatrix_16b(kr[1])
                    kr_b[4 * n_local + 2] = movmatrix_16b(kr[2])
                    kr_b[4 * n_local + 3] = movmatrix_16b(kr[3])
        nvvm.barrier_cta_sync(barrier_id, thread_count=cfg.threads_per_cta)

        # ---- phase 2, warps 1-3: inverse_beta fragments -----------------------------------------
        ainv = nvvm.ldmatrix(sIntermediate_ptr + inverse_idx, 4, nvvm.MMALayout.ROW)

        # ---- warp 2: A = A @ inverse_beta -> the a record (SW32 order, packed words) ---------
        if warp_local == 2:
            a_record_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
            for accum_idx in cutlass.range_constexpr(8):
                a_record_acc[accum_idx] = cutlass.Float32(0.0)
            mma_step(
                a_record_acc,
                (
                    fp32_to_fp16(a_acc[0], a_acc[1], dtype=cfg.io_dtype),
                    fp32_to_fp16(a_acc[2], a_acc[3], dtype=cfg.io_dtype),
                    fp32_to_fp16(a_acc[4], a_acc[5], dtype=cfg.io_dtype),
                    fp32_to_fp16(a_acc[6], a_acc[7], dtype=cfg.io_dtype),
                ),
                (
                    movmatrix_16b(ainv[0]),
                    movmatrix_16b(ainv[1]),
                    movmatrix_16b(ainv[2]),
                    movmatrix_16b(ainv[3]),
                ),
                k_step=0,
                M=16,
                N=16,
                ab_dtype=cfg.io_dtype,
            )
            for pair in cutlass.range_constexpr(4):
                row_coord = row_hi if cutlass.const_expr(pair % 2 == 1) else row_lo
                col_coord = (pair // 2) * 8 + col_pair
                word = (row_coord * cfg.b_t + swizzle_xor_32b(row_coord, col_coord)) // 2
                mA[tile_row, head_idx, word] = fp32_to_fp16(
                    a_record_acc[2 * pair], a_record_acc[2 * pair + 1], dtype=cfg.io_dtype
                )

        # ---- warps 1 and 3: T = inverse_beta^T @ K restore, key halves [0, 64) and [64, 128) --
        for half in cutlass.range_constexpr(2):
            if warp_local == 1 + 2 * half:
                t0 = movmatrix_16b(ainv[0])
                t1 = movmatrix_16b(ainv[2])
                t2 = movmatrix_16b(ainv[1])
                t3 = movmatrix_16b(ainv[3])
                for n_local in cutlass.range_constexpr(t_groups // 2):
                    n_group = half * (t_groups // 2) + n_local
                    t_acc = cute.make_rmem_tensor((8,), cutlass.Float32)
                    for accum_idx in cutlass.range_constexpr(8):
                        t_acc[accum_idx] = cutlass.Float32(0.0)
                    mma_step(
                        t_acc,
                        (t0, t1, t2, t3),
                        (
                            kr_b[4 * n_local],
                            kr_b[4 * n_local + 1],
                            kr_b[4 * n_local + 2],
                            kr_b[4 * n_local + 3],
                        ),
                        k_step=0,
                        M=16,
                        N=16,
                        ab_dtype=cfg.io_dtype,
                    )
                    nvvm.stmatrix(
                        sW_ptr + t_offsets[n_group],
                        [
                            fp32_to_fp16(t_acc[0], t_acc[1], dtype=cfg.io_dtype),
                            fp32_to_fp16(t_acc[2], t_acc[3], dtype=cfg.io_dtype),
                            fp32_to_fp16(t_acc[4], t_acc[5], dtype=cfg.io_dtype),
                            fp32_to_fp16(t_acc[6], t_acc[7], dtype=cfg.io_dtype),
                        ],
                        nvvm.MMALayout.ROW,
                        shape=nvvm.StoreShape.M8N8,
                    )

        # ---- the chunk's k_decay / q_decay / t tiles SMEM -> GMEM
        nvvm.fence_proxy("async.shared", space="cta")
        nvvm.barrier_cta_sync(barrier_id, thread_count=cfg.threads_per_cta)
        if warp_local == 1:
            if lane_idx == 0:
                store_item = item_begin + s
                store_row = store_item // heads_out
                store_head = store_item - store_row * heads_out.divisor
                store_chunk = mRows[store_row, 0]
                store_batch = mRows[store_row, 1] * cutlass.Int32(TENSOR_MAP_QWORDS)
                tma_store_tile(
                    sK_decay_tma[rec_slot],
                    tma_slice_runtime_desc(
                        (desc_k_decay_base + store_batch).tospace(cutlass.AddressSpace.generic),
                        zero,
                        zero,
                        store_head,
                        store_chunk,
                    ),
                )
                tma_store_tile(
                    sQ_decay_tma[rec_slot],
                    tma_slice_runtime_desc(
                        (desc_q_decay_base + store_batch).tospace(cutlass.AddressSpace.generic),
                        zero,
                        zero,
                        store_head,
                        store_chunk,
                    ),
                )
                tma_store_tile(
                    sW_tma[rec_slot],
                    tma_slice_runtime_desc(
                        (desc_t_base + store_batch).tospace(cutlass.AddressSpace.generic),
                        zero,
                        zero,
                        store_head,
                        store_chunk,
                    ),
                )
                tma_store_commit()
    if warp_local == 1:
        if lane_idx == 0:
            tma_store_wait(0)


@cute.jit
def build_descs_body(
    widx,
    base_q,
    base_k,
    base_gate,
    base_k_decay,
    base_q_decay,
    base_t,
    desc_workspace: cute.Tensor,
    cu_seqlens: cute.Tensor,
    q: cute.Tensor,
    k: cute.Tensor,
    gate: cute.Tensor,
    k_decay: cute.Tensor,
    q_decay: cute.Tensor,
    t: cute.Tensor,
    n_batch: cutlass.Int32,
    b_t: cutlass.Constexpr[int],
) -> None:
    """Per-batch descriptor-array build, one warp per array: Q, K, Gate, k_decay, q_decay, t."""
    arr_words = n_batch * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_q_arr = cute.make_tensor(
        desc_workspace.iterator, cute.make_layout((arr_words,), stride=(1,))
    )
    desc_k_arr = cute.make_tensor(
        desc_workspace.iterator + arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    desc_gate_arr = cute.make_tensor(
        desc_workspace.iterator + 2 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    desc_k_decay_arr = cute.make_tensor(
        desc_workspace.iterator + 3 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    desc_q_decay_arr = cute.make_tensor(
        desc_workspace.iterator + 4 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    desc_t_arr = cute.make_tensor(
        desc_workspace.iterator + 5 * arr_words, cute.make_layout((arr_words,), stride=(1,))
    )
    if widx == 0:
        emit_seq_descs(base_q, desc_q_arr, cu_seqlens, q, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 1:
        emit_seq_descs(base_k, desc_k_arr, cu_seqlens, k, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 2:
        emit_seq_descs(base_gate, desc_gate_arr, cu_seqlens, gate, n_batch, 2, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 3:
        emit_tile_seq_descs(
            base_k_decay, desc_k_decay_arr, cu_seqlens, k_decay, n_batch, b_t, 3, lanes=32
        )
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 4:
        emit_tile_seq_descs(
            base_q_decay, desc_q_decay_arr, cu_seqlens, q_decay, n_batch, b_t, 3, lanes=32
        )
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )
    if widx == 5:
        emit_tile_seq_descs(base_t, desc_t_arr, cu_seqlens, t, n_batch, b_t, 3, lanes=32)
        nvvm.fence_proxy_release(
            nvvm.MemScope.GPU, from_proxy=nvvm.Proxy.GENERIC, to_proxy=nvvm.Proxy.TENSORMAP
        )


def prep_base_maps(cfg, q, k, gate, k_decay, q_decay, t):
    """Trace-time helper: the prep's base TensorMaps (Q, K, Gate, k_decay, q_decay, t) whose
    per-batch arrays the prep-fed prefill prologue emits for it."""
    bytes_per_element = cfg.io_dtype.width // 8
    gate_bytes_per_element = cfg.gate_dtype.width // 8
    box_elems = 128 // bytes_per_element
    swizzle_128b = tma.TensorMapSwizzle.s128b
    q_headed = cute.make_tensor(
        q.iterator,
        cute.make_layout(
            (q.shape[0], q.shape[1], q.shape[2]), stride=(q.stride[0], q.stride[1], 1)
        ),
    )
    k_headed = cute.make_tensor(
        k.iterator,
        cute.make_layout(
            (k.shape[0], k.shape[1], k.shape[2]), stride=(k.stride[0], k.stride[1], 1)
        ),
    )
    gate_headed = cute.make_tensor(
        gate.iterator,
        cute.make_layout(
            (gate.shape[0], gate.shape[1], gate.shape[2]),
            stride=(gate.stride[0], gate.stride[1], 1),
        ),
    )
    base_desc_q = tma.create_tensor_map_tiled_from_view(
        q_headed, box_dims=(cfg.b_t, 1, box_elems), stride_order=(2, 1, 0), swizzle=swizzle_128b
    )
    base_desc_k = tma.create_tensor_map_tiled_from_view(
        k_headed, box_dims=(cfg.b_t, 1, box_elems), stride_order=(2, 1, 0), swizzle=swizzle_128b
    )
    base_desc_gate = tma.create_tensor_map_tiled_from_view(
        gate_headed,
        box_dims=(cfg.b_t, 1, 128 // gate_bytes_per_element),
        stride_order=(2, 1, 0),
        swizzle=swizzle_128b,
    )
    record_maps = []
    for rec in (k_decay, q_decay, t):
        rec_view = cute.make_tensor(
            rec.iterator,
            cute.make_layout(
                (rec.shape[0], rec.shape[1], rec.shape[2], rec.shape[3]),
                stride=(rec.stride[0], rec.stride[1], rec.stride[2], 1),
            ),
        )
        record_maps.append(
            tma.create_tensor_map_tiled_from_view(
                rec_view,
                box_dims=(1, 1, cfg.b_t, box_elems),
                stride_order=(3, 2, 1, 0),
                swizzle=swizzle_128b,
            )
        )
    return base_desc_q, base_desc_k, base_desc_gate, record_maps


def _name_float(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m").replace("+", "")


class KdaPrepOp:
    """The prep launch for one frozen ``KdaPrepCfg``; nested into the warmup forward host."""

    def __init__(self, cfg: "KdaPrepCfg", use_int64_offsets: bool = False):
        self.cfg = cfg
        self.use_int64_offsets = use_int64_offsets

    def get_name(self) -> str:
        cfg = self.cfg
        flags = "".join(
            str(int(flag))
            for flag in (
                cfg.l2norm,
                cfg.safe_gate,
                cfg.log_gate,
                cfg.beta_sigmoid,
                cfg.allow_neg_eigval,
            )
        )
        dtypes = "_".join(t.__name__.lower() for t in (cfg.io_dtype, cfg.gate_dtype))
        return (
            f"kda_cudnn_prep_{dtypes}_k{cfg.d_k}_f{flags}_g{_name_float(cfg.gate_scale_log2)}"
            f"_sm{cfg.num_sm}_i64{int(self.use_int64_offsets)}"
        )

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        desc_words: cute.Tensor,
        gate: cute.Tensor,
        a_log: cute.Tensor | None,
        dt_bias: cute.Tensor | None,
        beta: cute.Tensor,
        cu_seqlens: cute.Tensor,
        k_decay: cute.Tensor,
        q_decay: cute.Tensor,
        t: cute.Tensor,
        a: cute.Tensor,
        diag: cute.Tensor,
        rows: cute.Tensor,
        row_count: cute.Tensor,
        stream: cuda.CUstream,
    ):
        cfg = self.cfg
        # ---- grid: ctas_per_sm CTAs per SM, capped by the item count ----------------------
        heads_out = cutlass.Int32(gate.shape[1])
        q_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(q.shape[1]))
        k_ratio = cute.FastDivmodDivisorV2(heads_out // cutlass.Int32(k.shape[1]))
        num_ctas = cutlass.min(
            cutlass.Int32(cfg.num_sm * cfg.ctas_per_sm),
            cutlass.Int32(k_decay.shape[0]) * heads_out,
        )

        # ---- SMEM sizing -------------------------------------------------------------------
        tile_elems = cfg.d_k * cfg.b_t
        raw_elems = cfg.smem_raw_stages * tile_elems
        record_elems = cfg.smem_record_stages * tile_elems
        exchange_elems = (
            0 if cfg.gate_dtype == cutlass.Float32 else tile_elems
        )  # fp32 gates exchange in place
        align = cfg.buffer_align_bytes

        @cute.struct
        class SharedStorage:
            q: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, raw_elems], align]
            k: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, raw_elems], align]
            gate: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, cfg.gate_cosize], align]
            exchange: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, exchange_elems], align
            ]
            k_inv: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, tile_elems], align]
            k_restore: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, tile_elems], align]
            intermediate: cute.struct.Align[
                cute.struct.MemRange[cfg.io_dtype, cfg.b_t * cfg.b_t], align
            ]
            beta: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, cfg.b_t], 128]
            k_decay: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, record_elems], align]
            q_decay: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, record_elems], align]
            w: cute.struct.Align[cute.struct.MemRange[cfg.io_dtype, record_elems], align]

        frost_kda_prep.set_name_prefix(self.get_name())
        # ---- launch ----------------------------------------------------------------------
        frost_kda_prep(
            cfg,
            SharedStorage,
            cute.FastDivmodDivisorV2(heads_out),
            q_ratio,
            k_ratio,
            cute.FastDivmodDivisorV2(num_ctas),
            desc_words,
            a_log,
            dt_bias,
            beta,
            rows,
            row_count,
            a,
            diag,
            cutlass.Int32(cu_seqlens.shape[0] - 1),
        ).launch(
            grid=(num_ctas, 1, 1),
            block=(cfg.threads_per_cta, 1, 1),
            stream=stream,
            use_pdl=USE_PDL,
            min_blocks_per_mp=cfg.ctas_per_sm,
        )


@cute.kernel
def frost_kda_prep(
    cfg: cutlass.Constexpr,
    shared_type: cutlass.Constexpr,
    heads_out: cute.FastDivmodDivisorV2,
    q_ratio: cute.FastDivmodDivisorV2,
    k_ratio: cute.FastDivmodDivisorV2,
    num_ctas: cute.FastDivmodDivisorV2,
    mDesc: cute.Tensor,
    mA_log: cute.Tensor | None,
    mDt_bias: cute.Tensor | None,
    mBeta: cute.Tensor,
    mRows: cute.Tensor,
    mCount: cute.Tensor,
    mA: cute.Tensor,
    mDiag: cute.Tensor,
    num_batches: cutlass.Int32,
):
    """BT = 16 KDA prep (persistent item blocks, CTAS_PER_SM CTAs per SM)."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()

    tidx, _, _ = cute.arch.thread_idx()
    bidx = cute.arch.block_idx()[0]
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = tidx % cfg.threads_per_warp

    desc_base = mDesc.iterator.raw_ptr()
    arr_words = num_batches * cutlass.Int32(TENSOR_MAP_QWORDS)
    desc_q_base = desc_base
    desc_k_base = desc_q_base + arr_words
    desc_gate_base = desc_q_base + cutlass.Int32(2) * arr_words
    desc_k_decay_base = desc_q_base + cutlass.Int32(3) * arr_words
    desc_q_decay_base = desc_q_base + cutlass.Int32(4) * arr_words
    desc_t_base = desc_q_base + cutlass.Int32(5) * arr_words
    n_all = cutlass.Int32(mCount[0]) * heads_out.divisor
    items_per_cta = n_all // num_ctas
    spare_items = n_all - items_per_cta * num_ctas.divisor
    spare_quotient = (bidx * spare_items) // num_ctas
    spare_remainder = bidx * spare_items - spare_quotient * num_ctas.divisor
    item_begin = bidx * items_per_cta + spare_quotient
    n_items = items_per_cta + (
        cutlass.Int32(1) if spare_remainder + spare_items >= num_ctas.divisor else cutlass.Int32(0)
    )
    RS = cfg.smem_raw_stages
    RECS = cfg.smem_record_stages

    SMEM = cutlass.AddressSpace.smem
    # The data buffers share one storage allocation; the barrier ring keeps its v1.30 slot after
    # them.
    storage = SmemAllocator().allocate(shared_type)
    tile_elems = cfg.d_k * cfg.b_t
    sQ_raw = storage.q.get_tensor(cute.make_layout((RS * tile_elems,)))
    sK_raw = storage.k.get_tensor(cute.make_layout((RS * tile_elems,)))
    sGate_raw = storage.gate.get_tensor(cute.make_layout((cfg.gate_cosize,)))
    if cutlass.const_expr(cfg.gate_dtype == cutlass.Float32):
        sGate_load_ptr = smem_data_ptr(sGate_raw)
    else:
        sGate_load_ptr = cute.make_ptr(
            cfg.gate_dtype, smem_data_ptr(sGate_raw).toint(), mem_space=SMEM, assumed_align=1024
        )
    if cutlass.const_expr(cfg.gate_dtype == cutlass.Float32):
        sExchange_raw = sGate_raw
    else:
        sExchange_raw = storage.exchange.get_tensor(cute.make_layout((tile_elems,)))
    sK_inv_raw = storage.k_inv.get_tensor(cute.make_layout((tile_elems,)))
    sK_restore_raw = storage.k_restore.get_tensor(cute.make_layout((tile_elems,)))
    sIntermediate_raw = storage.intermediate.get_tensor(cute.make_layout((cfg.b_t * cfg.b_t,)))
    sBetaG_raw = storage.beta.get_tensor(cute.make_layout((cfg.b_t,)))
    sK_decay_raw = storage.k_decay.get_tensor(cute.make_layout((RECS * tile_elems,)))
    sQ_decay_raw = storage.q_decay.get_tensor(cute.make_layout((RECS * tile_elems,)))
    sW_raw = storage.w.get_tensor(cute.make_layout((RECS * tile_elems,)))
    sQ_tma = SmemTile(
        base=sQ_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=RS,
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
        stages=RS,
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
        elems_per_stage=(cfg.gate_cosize // RS),
        stages=RS,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // gate_box_elems),
        tma_granu_elems=gate_box_elems,
        tma_subtile_stride_elems=(cfg.b_t * 32),
    )
    sK_decay_tma = SmemTile(
        base=sK_decay_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=RECS,
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
        stages=RECS,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
    sW_tma = SmemTile(
        base=sW_raw,
        elems_per_stage=(cfg.d_k * cfg.b_t),
        stages=RECS,
        leading_byte_offset=0,
        stride_byte_offset=0,
        layout=0,
        tma_loads_per_tile=(cfg.d_k // 64),
        tma_granu_elems=64,
        tma_subtile_stride_elems=(cfg.b_t * 64),
    )
    bars = make_bars(cfg)

    # ---- mbarrier init (thread 0) ----------------------------------------------------
    if tidx == 0:
        for s in range(RS):
            bars.mb_raw_ready[s].init()
    nvvm.fence_mbarrier_init()
    nvvm.barrier_cta_sync()

    # ---- the CTA's four warps -----------------------------------------------------------
    compute_warp_group(
        cfg,
        n_items,
        item_begin,
        mRows,
        warp_idx,
        lane_idx,
        mA_log,
        mDt_bias,
        mBeta,
        mA,
        mDiag,
        sQ_raw,
        sK_raw,
        sGate_load_ptr,
        sExchange_raw,
        sK_inv_raw,
        sK_restore_raw,
        sIntermediate_raw,
        sBetaG_raw,
        sK_decay_raw,
        sQ_decay_raw,
        sW_raw,
        bars,
        heads_out,
        q_ratio,
        k_ratio,
        sQ_tma,
        sK_tma,
        sGate_tma,
        sK_decay_tma,
        sQ_decay_tma,
        sW_tma,
        desc_q_base,
        desc_k_base,
        desc_gate_base,
        desc_k_decay_base,
        desc_q_decay_base,
        desc_t_base,
    )
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@dataclass(frozen=True)
class KdaPrepCfg:
    """Kernel cfg (fixed BT = 16 schedule constants; the derived SMEM sizes are stamped by
    ``build_cfg``).  Passed ``cfg``-first (a ``cutlass.Constexpr``) into ``KdaPrepOp`` / ``kernel``
    and every warp body."""

    io_dtype: type[cutlass.Numeric]
    gate_dtype: type[cutlass.Numeric]
    num_sm: int
    l2norm: bool
    safe_gate: bool
    gate_scale_log2: float
    log_gate: bool
    beta_sigmoid: bool
    allow_neg_eigval: bool
    d_k: int
    b_t: int = CFG.B_T
    compute_warps: int = CFG.COMPUTE_WARPS
    threads_per_warp: int = CFG.THREADS_PER_WARP
    ctas_per_sm: int = CFG.CTAS_PER_SM
    threads_per_cta: int = 0  # derived by build_cfg
    barrier_id: int = 1
    smem_raw_stages: int = CFG.SMEM_RAW_STAGES
    smem_record_stages: int = CFG.SMEM_RECORD_STAGES
    buffer_align_bytes: int = CFG.BUFFER_ALIGN_BYTES

    # ---- derived by build_cfg ---------------------------------------------------------
    gate_stage_elems: int = 0  # derived by build_cfg
    gate_cosize: int = 0  # derived by build_cfg
    tma_q_bytes: int = 0  # derived by build_cfg
    tma_k_bytes: int = 0  # derived by build_cfg
    tma_gate_bytes: int = 0  # derived by build_cfg


def build_cfg(
    io_dtype: type[cutlass.Numeric],
    gate_dtype: type[cutlass.Numeric],
    *,
    num_sm: int,
    l2norm: bool,
    safe_gate: bool,
    gate_scale_log2: float,
    log_gate: bool = True,
    beta_sigmoid: bool,
    allow_neg_eigval: bool,
    d_k: int,
) -> KdaPrepCfg:
    """Build the per-compile ``KdaPrepCfg`` (io_dtype in {Float16, BFloat16}; gate fp32 or the io
    dtype)."""
    validate_kernel_domain("KDA prep", io_dtype, (d_k,), num_sm=num_sm)
    if gate_dtype not in (cutlass.Float32, io_dtype):
        raise ValueError(f"KDA prep reads a Float32 or io-dtype gate, got {gate_dtype}")
    cfg = KdaPrepCfg(
        io_dtype=io_dtype,
        gate_dtype=gate_dtype,
        num_sm=num_sm,
        l2norm=l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        log_gate=log_gate,
        beta_sigmoid=beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        d_k=d_k,
    )
    bytes_per_element = io_dtype.width // 8
    tile = d_k * cfg.b_t
    stage_bytes = 2 * tile * bytes_per_element + tile * (gate_dtype.width // 8) + 4 * cfg.b_t
    fixed_bytes = (
        2 * tile * bytes_per_element
        + cfg.b_t * cfg.b_t * bytes_per_element
        + 4 * cfg.b_t
        + (0 if gate_dtype == cutlass.Float32 else 4 * tile)
    )
    record_bytes = 3 * cfg.smem_record_stages * tile * bytes_per_element
    smem_bytes = (
        cfg.smem_raw_stages * stage_bytes
        + fixed_bytes
        + record_bytes
        + CFG.SMEM_CTA_RESERVED_BYTES
    )
    if cfg.ctas_per_sm * smem_bytes > CFG.SMEM_PER_SM_BYTES:
        raise ValueError(
            f"{cfg.ctas_per_sm} CTAs of {smem_bytes} B do not fit the SM ({CFG.SMEM_PER_SM_BYTES} "
            f"B)"
        )
    cfg = replace(
        cfg,
        threads_per_cta=cfg.threads_per_warp * cfg.compute_warps,
        gate_stage_elems=tile,
        gate_cosize=cfg.smem_raw_stages * tile * (gate_dtype.width // 8) // 4,
        tma_q_bytes=tile * bytes_per_element,
        tma_k_bytes=tile * bytes_per_element,
        tma_gate_bytes=tile * (gate_dtype.width // 8),
    )
    validate_named_barriers(cfg.threads_per_cta, compute=(cfg.barrier_id, cfg.threads_per_cta))
    return cfg


TENSORMAP_DESC_ARRAYS = 6  # per-batch runtime TMA descriptors: Q, K, Gate, k_decay, q_decay, t
