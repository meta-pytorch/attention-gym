# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe.

"""Safe-gate backward helper: convert the main backward kernels' dGate
(gradient wrt the transformed log-decay) into the raw-logit gradient in
place, and produce the gate-parameter gradients dA_log / ddt_bias.

Two forms:
  - scalar (GDN):       g = -exp(A_log[h]) * softplus(g_raw + dt_bias[h])
  - per-channel (KDA):  g = lb * sigmoid(exp(a_log) * (g_raw + dt_bias))

a_log and dt_bias are each optional (an absent a_log means exp(a_log) = 1, an
absent dt_bias means bias 0); an absent parameter's gradient and partial carve
are None and not produced, while the in-place dGate rewrite always runs.

The parameter gradients are cross-token sums, done deterministically: every
fp32 accumulator owns a statically assigned (head[, channel], token-slice)
and partials are combined through fixed-shape trees only, the bracketing is
a pure function of the shapes, never of scheduling, so results are bitwise
stable run to run.  Two stages with a launch boundary as the grid barrier:
a partial pass over token stripes (the same pass rewrites dGate -> dg_raw),
then a finisher that folds the stripe partials.
"""

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass.cute.runtime import from_dlpack

from ..tile_dsl.pointwise import fadd2, ffma2, fmul2, lane_group_sum, opaque_f32_zero, sigmoid, sigmoid2, softplus
from ..tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from ..tile_dsl.tma import ld_global_v2, ld_global_v4, st_global_v2, st_global_v4

from .host import get_dtype
from .._compat import current_device

USE_PDL = True

GATE_BWD_BLOCKS = 128  # channel-gate token stripes (partials carve = GATE_BWD_BLOCKS * HO * d_k fp32)
CHANNEL_BLOCK = 128  # channel-gate partial-kernel CTA threads
SCALAR_BLOCK_CAP = 8192  # scalar-gate stripe ceiling
SCALAR_SLICE_TOKENS = 16  # scalar-gate target tokens per stripe
SCALAR_HEAD_TILE = 32  # scalar-gate heads per block: one warp, tiled over grid.y


def scalar_gate_blocks(n_tokens: int) -> int:
    """Scalar-gate stripe count: shape-only (so the summation bracketing is
    deterministic) and scaling with the token axis."""
    return min(SCALAR_BLOCK_CAP, max(1, -(-n_tokens // SCALAR_SLICE_TOKENS)))


@cute.kernel
def frost_scalar_gate_bwd_partial(
    mDg: cute.Tensor,
    mG: cute.Tensor,
    mALog: cute.Tensor | None,
    mDtBias: cute.Tensor | None,
    mPartA: cute.Tensor | None,
    mPartDt: cute.Tensor | None,
    n_tokens: cutlass.Int32,
    h_o: cutlass.Int32,
    slice_len: cutlass.Int32,
) -> None:
    """Grid (stripes, head tiles), block (SCALAR_HEAD_TILE,): thread h walks
    stripe g's token slice in order, rewriting dGate -> dg_raw in place and
    accumulating the (g, h) partials for dA_log (dGate * g_transformed) and
    ddt_bias (dg_raw).  mPartA is None iff mALog is, mPartDt iff mDtBias is."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    bid = cute.arch.block_idx()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    stripe_idx = cutlass.Int32(bid[0])
    h = cutlass.Int32(bid[1]) * cutlass.Int32(SCALAR_HEAD_TILE) + tidx
    if h < h_o:
        if cutlass.const_expr(mALog is not None):
            neg_exp_a = -cute.math.exp(mALog[h].to(cutlass.Float32), fastmath=True)
        else:
            neg_exp_a = -(opaque_f32_zero() + cutlass.Float32(1.0))
        if cutlass.const_expr(mDtBias is not None):
            bias = mDtBias[h].to(cutlass.Float32)
        else:
            bias = opaque_f32_zero()
        t = stripe_idx * slice_len
        t_end = t + slice_len
        if t_end > n_tokens:
            t_end = n_tokens
        acc_a = cutlass.Float32(0.0)
        acc_dt = cutlass.Float32(0.0)
        while t < t_end:
            d_gate = mDg[t, h].to(cutlass.Float32)
            y = mG[t, h].to(cutlass.Float32) + bias
            dg_raw = d_gate * neg_exp_a * sigmoid(y)
            if cutlass.const_expr(mPartA is not None):
                acc_a += d_gate * (neg_exp_a * softplus(y))
            if cutlass.const_expr(mPartDt is not None):
                acc_dt += dg_raw
            mDg[t, h] = dg_raw.to(mDg.element_type)
            t += 1
        if cutlass.const_expr(mPartA is not None):
            mPartA[stripe_idx * h_o + h] = acc_a
        if cutlass.const_expr(mPartDt is not None):
            mPartDt[stripe_idx * h_o + h] = acc_dt
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.kernel
def frost_scalar_gate_bwd_finish(
    mPartA: cute.Tensor | None,
    mPartDt: cute.Tensor | None,
    mDA: cute.Tensor | None,
    mDDt: cute.Tensor | None,
    h_o: cutlass.Int32,
    n_blocks: cutlass.Int32,
) -> None:
    """Grid (HO,), block (32,): head h's stripe partials fold as 8 fixed
    interleaved chains per lane (independent accumulators), then the 8 chains
    pairwise and one butterfly tree across the 32 lanes, a fixed-shape
    bracketing regardless of the stripe count.  mDA is None iff mPartA is,
    mDDt iff mPartDt is; at least one pair is present."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    bid = cute.arch.block_idx()
    lane_idx = cutlass.Int32(cute.arch.thread_idx()[0])
    h = cutlass.Int32(bid[0])
    a8 = cutlass.Array(cutlass.Float32, 8)
    d8 = cutlass.Array(cutlass.Float32, 8)
    for j in cutlass.range_constexpr(8):
        a8[j] = cutlass.Float32(0.0)
        d8[j] = cutlass.Float32(0.0)
    s = lane_idx
    while s + cutlass.Int32(224) < n_blocks:
        for j in cutlass.range_constexpr(8):
            idx = (s + cutlass.Int32(32 * j)) * h_o + h
            if cutlass.const_expr(mPartA is not None):
                part_a_value = mPartA[idx]
            else:
                part_a_value = cutlass.Float32(0.0)
            if cutlass.const_expr(mPartDt is not None):
                part_dt_value = mPartDt[idx]
            else:
                part_dt_value = cutlass.Float32(0.0)
            a8[j], d8[j] = fadd2(a8[j], d8[j], part_a_value, part_dt_value)
        s += cutlass.Int32(256)
    while s < n_blocks:
        tail_idx = s * h_o + h
        if cutlass.const_expr(mPartA is not None):
            tail_a = mPartA[tail_idx]
        else:
            tail_a = cutlass.Float32(0.0)
        if cutlass.const_expr(mPartDt is not None):
            tail_dt = mPartDt[tail_idx]
        else:
            tail_dt = cutlass.Float32(0.0)
        a8[0], d8[0] = fadd2(a8[0], d8[0], tail_a, tail_dt)
        s += cutlass.Int32(32)
    p0a, p0d = fadd2(a8[0], d8[0], a8[1], d8[1])
    p1a, p1d = fadd2(a8[2], d8[2], a8[3], d8[3])
    p2a, p2d = fadd2(a8[4], d8[4], a8[5], d8[5])
    p3a, p3d = fadd2(a8[6], d8[6], a8[7], d8[7])
    q0a, q0d = fadd2(p0a, p0d, p1a, p1d)
    q1a, q1d = fadd2(p2a, p2d, p3a, p3d)
    acc_a, acc_dt = fadd2(q0a, q0d, q1a, q1d)
    acc_a = lane_group_sum(acc_a, 32)
    acc_dt = lane_group_sum(acc_dt, 32)
    if cutlass.const_expr(mDA is not None):
        if lane_idx == 0:
            mDA[h] = acc_a.to(mDA.element_type)
    if cutlass.const_expr(mDDt is not None):
        if lane_idx == 0:
            mDDt[h] = acc_dt.to(mDDt.element_type)
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.kernel
def frost_channel_gate_bwd_partial(
    d_k: cutlass.Constexpr[int],
    mDg: cute.Tensor,
    mG: cute.Tensor,
    mALog: cute.Tensor | None,
    mDtBias: cute.Tensor | None,
    mPartA: cute.Tensor | None,
    mPartDt: cute.Tensor | None,
    n_tokens: cutlass.Int32,
    h_o: cutlass.Int32,
    slice_len: cutlass.Int32,
    lower_bound: cutlass.Float32,
) -> None:
    """Grid (GATE_BWD_BLOCKS, HO), block (CHANNEL_BLOCK,) = ``phases``
    token-phased row groups of ``d_k // 4`` lanes; a lane owns channels
    [d0, d0 + 4) through 128-bit loads and stores.  Group p walks tokens
    t0 + p, t0 + p + phases, ... of the stripe: with z = exp(a_log) *
    (g_raw + dt_bias) and w_ = dGate * lb * sig(z)(1-sig(z)), dg_raw =
    w_ * exp(a_log) (rewritten in place), the dA_log integrand is w_ * z,
    the ddt_bias one dg_raw.  The ``phases`` group partials per channel
    fold p = 0..phases-1 through SMEM (fixed order) into the (g, h)
    partial slots.  a_log is one scalar per head; dt_bias is per
    (head, channel).  mPartA is None iff mALog is, mPartDt iff mDtBias is."""
    lanes_per_row = d_k // 4
    phases = CHANNEL_BLOCK // lanes_per_row
    active = phases * lanes_per_row
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    bid = cute.arch.block_idx()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    stripe_idx = cutlass.Int32(bid[0])
    h = cutlass.Int32(bid[1])
    phase_id = tidx // cutlass.Int32(lanes_per_row)
    d0 = (tidx % cutlass.Int32(lanes_per_row)) * cutlass.Int32(4)
    sA = cutlass.Array(cutlass.Float32, phases * d_k, space=cutlass.AddressSpace.smem, alignment=16)
    sDt = cutlass.Array(cutlass.Float32, phases * d_k, space=cutlass.AddressSpace.smem, alignment=16)
    if tidx < cutlass.Int32(active):
        if cutlass.const_expr(mALog is not None):
            exp_a = cute.math.exp(mALog[h].to(cutlass.Float32), fastmath=True)
        else:
            exp_a = opaque_f32_zero() + cutlass.Float32(1.0)
        bias = cutlass.Array(cutlass.Float32, 4)
        for q in cutlass.range_constexpr(4):
            if cutlass.const_expr(mDtBias is not None):
                bias[q] = mDtBias[h, d0 + cutlass.Int32(q)].to(cutlass.Float32)
            else:
                bias[q] = opaque_f32_zero()
        acc_a = cutlass.Array(cutlass.Float32, 4)
        acc_dt = cutlass.Array(cutlass.Float32, 4)
        out = cutlass.Array(cutlass.Float32, 4)
        for q in cutlass.range_constexpr(4):
            acc_a[q] = cutlass.Float32(0.0)
            acc_dt[q] = cutlass.Float32(0.0)
        dg_base = mDg.iterator.toint()
        g_base = mG.iterator.toint()
        gate_elem_bytes = cutlass.const_expr(mG.element_type.width // 8)
        dg_s0 = cutlass.Int64(mDg.stride[0])
        dg_s1 = cutlass.Int64(mDg.stride[1])
        g_s0 = cutlass.Int64(mG.stride[0])
        g_s1 = cutlass.Int64(mG.stride[1])
        t = stripe_idx * slice_len + phase_id
        t_end = stripe_idx * slice_len + slice_len
        if t_end > n_tokens:
            t_end = n_tokens
        while t < t_end:
            dg_addr = dg_base + (cutlass.Int64(t) * dg_s0 + cutlass.Int64(h) * dg_s1 + cutlass.Int64(d0)) * cutlass.Int64(gate_elem_bytes)
            g_addr = g_base + (cutlass.Int64(t) * g_s0 + cutlass.Int64(h) * g_s1 + cutlass.Int64(d0)) * cutlass.Int64(gate_elem_bytes)
            if cutlass.const_expr(mG.element_type == cutlass.Float32):
                dgv = ld_global_v4(dg_addr, cutlass.Float32)
                gvv = ld_global_v4(g_addr, cutlass.Float32)
            else:
                dw0, dw1 = ld_global_v2(dg_addr, cutlass.Int32)
                gw0, gw1 = ld_global_v2(g_addr, cutlass.Int32)
                dfrag = cutlass.Vector.from_elements((dw0, dw1), cutlass.Int32).bitcast(mG.element_type).to(cutlass.Float32)
                gfrag = cutlass.Vector.from_elements((gw0, gw1), cutlass.Int32).bitcast(mG.element_type).to(cutlass.Float32)
                dgv = (dfrag[0], dfrag[1], dfrag[2], dfrag[3])
                gvv = (gfrag[0], gfrag[1], gfrag[2], gfrag[3])
            for p in cutlass.range_constexpr(2):
                i = 2 * p
                j = i + 1
                one = cutlass.Float32(1.0)
                y_lo, y_hi = fadd2(gvv[i], gvv[j], bias[i], bias[j])
                z_lo, z_hi = fmul2(exp_a, exp_a, y_lo, y_hi)
                sig_lo, sig_hi = sigmoid2(z_lo, z_hi)
                c_lo, c_hi = fadd2(one, one, -sig_lo, -sig_hi)
                k_lo, k_hi = fmul2(sig_lo, sig_hi, c_lo, c_hi)
                b_lo, b_hi = fmul2(dgv[i], dgv[j], lower_bound, lower_bound)
                w_lo, w_hi = fmul2(b_lo, b_hi, k_lo, k_hi)
                raw_lo, raw_hi = fmul2(w_lo, w_hi, exp_a, exp_a)
                if cutlass.const_expr(mPartA is not None):
                    acc_a[i], acc_a[j] = ffma2(w_lo, w_hi, z_lo, z_hi, acc_a[i], acc_a[j])
                if cutlass.const_expr(mPartDt is not None):
                    acc_dt[i], acc_dt[j] = fadd2(acc_dt[i], acc_dt[j], raw_lo, raw_hi)
                out[i] = raw_lo
                out[j] = raw_hi
            if cutlass.const_expr(mG.element_type == cutlass.Float32):
                st_global_v4(dg_addr, out, cutlass.Float32)
            else:
                words = cutlass.Vector.from_elements((out[0], out[1], out[2], out[3]), cutlass.Float32).to(mG.element_type).bitcast(cutlass.Int32)
                st_global_v2(dg_addr, words, cutlass.Int32)
            t += cutlass.Int32(phases)
        for q in cutlass.range_constexpr(4):
            if cutlass.const_expr(mPartA is not None):
                sA[phase_id * cutlass.Int32(d_k) + d0 + cutlass.Int32(q)] = acc_a[q]
            if cutlass.const_expr(mPartDt is not None):
                sDt[phase_id * cutlass.Int32(d_k) + d0 + cutlass.Int32(q)] = acc_dt[q]
    nvvm.barrier_cta_sync()
    if cutlass.const_expr(mPartA is not None or mPartDt is not None):
        if tidx < cutlass.Int32(lanes_per_row):
            for q in cutlass.range_constexpr(4):
                d = d0 + cutlass.Int32(q)
                base = (stripe_idx * h_o + h) * cutlass.Int32(d_k) + d
                if cutlass.const_expr(mPartA is not None):
                    fa = sA[d]
                    for p in cutlass.range_constexpr(1, phases):
                        fa = fa + sA[cutlass.Int32(p * d_k) + d]
                    mPartA[base] = fa
                if cutlass.const_expr(mPartDt is not None):
                    fdt = sDt[d]
                    for p in cutlass.range_constexpr(1, phases):
                        fdt = fdt + sDt[cutlass.Int32(p * d_k) + d]
                    mPartDt[base] = fdt
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.kernel
def frost_channel_gate_bwd_finish(
    d_k: cutlass.Constexpr[int],
    mPartA: cute.Tensor | None,
    mPartDt: cute.Tensor | None,
    mDA: cute.Tensor | None,
    mDDt: cute.Tensor | None,
    h_o: cutlass.Int32,
) -> None:
    """Grid (HO,), block (ceil(d_k/32)*32,): thread d folds its channel
    column over the GATE_BWD_BLOCKS stripe partials as 8 fixed interleaved
    chains (independent accumulators) folded pairwise.
    ddt_bias is per (head, channel), so thread d stores its column outright;
    dA_log is per head, so the channel axis folds through a fixed tree (warp
    butterflies, then the warp sums in index order).  mDA is None iff mPartA
    is, mDDt iff mPartDt is; at least one pair is present."""
    n_warps = (d_k + 31) // 32
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    bid = cute.arch.block_idx()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    h = cutlass.Int32(bid[0])
    d = tidx
    warp_id = tidx // cutlass.Int32(32)
    lane_idx = tidx % cutlass.Int32(32)
    a8 = cutlass.Array(cutlass.Float32, 8)
    d8 = cutlass.Array(cutlass.Float32, 8)
    for j in cutlass.range_constexpr(8):
        a8[j] = cutlass.Float32(0.0)
        d8[j] = cutlass.Float32(0.0)
    if d < cutlass.Int32(d_k):
        c = cutlass.Int32(0)
        while c < cutlass.Int32(GATE_BWD_BLOCKS // 8):
            for j in cutlass.range_constexpr(8):
                idx = ((c * cutlass.Int32(8) + cutlass.Int32(j)) * h_o + h) * cutlass.Int32(d_k) + d
                if cutlass.const_expr(mPartA is not None):
                    a8[j] = a8[j] + mPartA[idx]
                if cutlass.const_expr(mPartDt is not None):
                    d8[j] = d8[j] + mPartDt[idx]
            c += 1
    if cutlass.const_expr(mDDt is not None):
        col_dt = ((d8[0] + d8[1]) + (d8[2] + d8[3])) + ((d8[4] + d8[5]) + (d8[6] + d8[7]))
        if d < cutlass.Int32(d_k):
            mDDt[h, d] = col_dt.to(mDDt.element_type)
    if cutlass.const_expr(mDA is not None):
        col_a = ((a8[0] + a8[1]) + (a8[2] + a8[3])) + ((a8[4] + a8[5]) + (a8[6] + a8[7]))
        sWa = cutlass.Array(cutlass.Float32, n_warps, space=cutlass.AddressSpace.smem, alignment=16)
        va = lane_group_sum(col_a, 32)
        if lane_idx == 0:
            sWa[warp_id] = va
        nvvm.barrier_cta_sync()
        if tidx == 0:
            col = sWa[0]
            for w in cutlass.range_constexpr(1, n_warps):
                col = col + sWa[w]
            mDA[h] = col.to(mDA.element_type)
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.jit
def scalar_gate_bwd_launch(
    d_gate: cute.Tensor,
    g_raw: cute.Tensor,
    a_log: cute.Tensor | None,
    dt_bias: cute.Tensor | None,
    part_a: cute.Tensor | None,
    part_dt: cute.Tensor | None,
    d_a_log: cute.Tensor | None,
    d_dt_bias: cute.Tensor | None,
    n_tokens: cutlass.Int32,
    h_o: cutlass.Int32,
    slice_len: cutlass.Int32,
    n_blocks: cutlass.Int32,
    head_tiles: cutlass.Int32,
    stream: cuda.CUstream,
):
    frost_scalar_gate_bwd_partial(d_gate, g_raw, a_log, dt_bias, part_a, part_dt, n_tokens, h_o, slice_len).launch(
        grid=(n_blocks, head_tiles, 1), block=(SCALAR_HEAD_TILE, 1, 1), stream=stream, use_pdl=USE_PDL
    )
    if cutlass.const_expr(d_a_log is not None or d_dt_bias is not None):
        frost_scalar_gate_bwd_finish(part_a, part_dt, d_a_log, d_dt_bias, h_o, n_blocks).launch(
            grid=(h_o, 1, 1), block=(32, 1, 1), stream=stream, use_pdl=USE_PDL
        )


@cute.jit
def channel_gate_bwd_launch(
    d_k: cutlass.Constexpr[int],
    d_gate: cute.Tensor,
    g_raw: cute.Tensor,
    a_log: cute.Tensor | None,
    dt_bias: cute.Tensor | None,
    part_a: cute.Tensor | None,
    part_dt: cute.Tensor | None,
    d_a_log: cute.Tensor | None,
    d_dt_bias: cute.Tensor | None,
    n_tokens: cutlass.Int32,
    h_o: cutlass.Int32,
    slice_len: cutlass.Int32,
    lower_bound: cutlass.Float32,
    stream: cuda.CUstream,
):
    frost_channel_gate_bwd_partial(d_k, d_gate, g_raw, a_log, dt_bias, part_a, part_dt, n_tokens, h_o, slice_len, lower_bound).launch(
        grid=(GATE_BWD_BLOCKS, h_o, 1), block=(CHANNEL_BLOCK, 1, 1), stream=stream, use_pdl=USE_PDL
    )
    if cutlass.const_expr(d_a_log is not None or d_dt_bias is not None):
        frost_channel_gate_bwd_finish(d_k, part_a, part_dt, d_a_log, d_dt_bias, h_o).launch(
            grid=(h_o, 1, 1), block=(((d_k + 31) // 32) * 32, 1, 1), stream=stream, use_pdl=USE_PDL
        )


@functools.cache
def gate_bwd_cache(key):
    return {}


def scalar_gate_bwd(d_gate, g_raw, a_log, dt_bias, d_a_log, d_dt_bias, part_a, part_dt, *, stream):
    """Scalar-gate backward: rewrite d_gate [total, HO] (gate dtype) in place from
    transformed-space to raw-logit space and fill d_a_log/d_dt_bias (HO,).
    part_a/part_dt are (scalar_gate_blocks(total) * HO,) fp32 workspace carves.
    a_log/dt_bias may be None (unit amplitude / zero bias); the matching
    d_*/part_* are then None and that gradient is not produced."""
    n_tokens, h_o = (int(dim) for dim in d_gate.shape)
    n_blocks = scalar_gate_blocks(n_tokens)
    head_tiles = -(-h_o // SCALAR_HEAD_TILE)
    slice_len = (n_tokens + n_blocks - 1) // n_blocks
    cache = gate_bwd_cache(
        (
            "gdn",
            str(g_raw.dtype),
            str(a_log.dtype) if a_log is not None else "none",
            str(dt_bias.dtype) if dt_bias is not None else "none",
            d_a_log is not None,
            d_dt_bias is not None,
            part_a is not None,
            part_dt is not None,
            current_device(),
        )
    )
    cu_stream = cuda.CUstream(int(stream))
    tensors = (d_gate, g_raw, a_log, dt_bias, part_a, part_dt, d_a_log, d_dt_bias)
    if "compiled" not in cache:
        traced = [from_dlpack(t, assumed_align=4) for t in tensors[:2]]
        traced = [tr.mark_layout_dynamic(leading_dim=1) for tr in traced]
        traced += [from_dlpack(t, assumed_align=4).mark_layout_dynamic(leading_dim=0) if t is not None else None for t in tensors[2:]]
        cache["compiled"] = cute.compile(
            scalar_gate_bwd_launch,
            *traced,
            cutlass.Int32(n_tokens),
            cutlass.Int32(h_o),
            cutlass.Int32(slice_len),
            cutlass.Int32(n_blocks),
            cutlass.Int32(head_tiles),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["compiled"](*tensors, n_tokens, h_o, slice_len, n_blocks, head_tiles, cu_stream)


def channel_gate_bwd(d_gate, g_raw, a_log, dt_bias, d_a_log, d_dt_bias, part_a, part_dt, gate_lower_bound, *, stream):
    """Per-channel-gate backward: rewrite d_gate [total, HO, d_k] (gate dtype) in
    place and fill d_a_log/d_dt_bias at their parameter shapes ((HO,) params
    get the channel axis folded in the finisher).  a_log/dt_bias may be None
    (unit amplitude / zero bias); the matching d_*/part_* are then None and
    that gradient is not produced."""
    n_tokens, h_o, d_k = (int(dim) for dim in d_gate.shape)
    g_vector_bytes = 4 * (get_dtype(g_raw.dtype).width // 8)
    slice_len = (n_tokens + GATE_BWD_BLOCKS - 1) // GATE_BWD_BLOCKS
    cache = gate_bwd_cache(
        (
            "channel",
            str(g_raw.dtype),
            str(a_log.dtype) if a_log is not None else "none",
            str(dt_bias.dtype) if dt_bias is not None else "none",
            d_a_log is not None,
            d_dt_bias is not None,
            part_a is not None,
            part_dt is not None,
            d_k,
            current_device(),
        )
    )
    cu_stream = cuda.CUstream(int(stream))
    tensors = (d_gate, g_raw, a_log, dt_bias, part_a, part_dt, d_a_log, d_dt_bias)
    args = (n_tokens, h_o, slice_len)
    if "compiled" not in cache:
        traced = [from_dlpack(t, assumed_align=g_vector_bytes).mark_layout_dynamic(leading_dim=2) for t in (d_gate, g_raw)]
        traced += [from_dlpack(t, assumed_align=4).mark_layout_dynamic(leading_dim=len(t.shape) - 1) if t is not None else None for t in tensors[2:]]
        cache["compiled"] = cute.compile(
            channel_gate_bwd_launch,
            d_k,
            *traced,
            *(cutlass.Int32(a) for a in args),
            cutlass.Float32(gate_lower_bound),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    cache["compiled"](*tensors, *args, float(gate_lower_bound), cu_stream)


frost_scalar_gate_bwd_partial.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_scalar_gate_bwd_finish.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_channel_gate_bwd_partial.set_name_prefix("cudnn", remove_cutlass_symbol=False)
frost_channel_gate_bwd_finish.set_name_prefix("cudnn", remove_cutlass_symbol=False)
