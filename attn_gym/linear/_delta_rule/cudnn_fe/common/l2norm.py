# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe.

"""Q/K row L2 normalization helper for GDN.  The main kernels consume the
normalized workspace copies through their usual descriptors.

Normalize every d-element q/k row (d in {64, 128}) into compact io-dtype
workspace buffers and store the fp32 inverse norms.  The backward projection
through the normalize Jacobian runs inside the bprop kernel's dQ/dK epilogues.
"""

from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from .._compat import current_device
from ..tile_dsl.pointwise import f16x2_to_f32, fmul2, fp32_to_fp16, l2norm_inv, lane_group_sum
from ..tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from ..tile_dsl.tma import ld_global_v4, st_global, st_global_v4

USE_PDL = True

THREADS_PER_CTA = 512
FWD_LANES = 4
FWD_ROWS_PER_GROUP = 1


@cute.kernel
def frost_l2norm_qk(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mQn: cute.Tensor,
    mKn: cute.Tensor,
    mInvQ: cute.Tensor,
    mInvK: cute.Tensor,
    n_q_rows: cutlass.Int32,
    n_rows: cutlass.Int32,
    h_q: cutlass.Int32,
    h_k: cutlass.Int32,
    d: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    expand_phase: cutlass.Constexpr[int],
    expand_fill: cutlass.Constexpr[bool],
) -> None:
    """Grid over all q rows then all k rows, FWD_LANES lanes x (d // FWD_LANES)
    elements per row, FWD_ROWS_PER_GROUP consecutive rows per lane group: fp32 sums
    of squares, rsqrt with the shared epsilon floor, normalized rows to the
    compact io workspace, inverse norms to their fp32 slots.  Accesses are
    indexed chunk-major, so the lanes of one access cover a contiguous span of
    the row.  Tail rows clamp their loads and skip stores."""
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
    vec_chunks = cutlass.const_expr(d // (FWD_LANES * 8))
    bid = cute.arch.block_idx()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    grp = tidx // cutlass.Int32(FWD_LANES)
    lane_idx = tidx % cutlass.Int32(FWD_LANES)
    row0 = (cutlass.Int32(bid[0]) * cutlass.Int32(THREADS_PER_CTA // FWD_LANES) + grp) * cutlass.Int32(FWD_ROWS_PER_GROUP)
    rows = []
    workspace_addrs = []
    nrm_addrs = []
    vals = []
    for r in cutlass.range_constexpr(FWD_ROWS_PER_GROUP):
        row = row0 + cutlass.Int32(r)
        row_r = row if row < n_rows else n_rows - cutlass.Int32(1)
        src_addr = cutlass.Int64(0)
        workspace_addr = cutlass.Int64(0)
        nrm_addr = cutlass.Int64(0)
        on_phase = cutlass.Boolean(True)
        if row_r < n_q_rows:
            if cutlass.const_expr(expand_fill):
                expanded_row = row_r // h_q
                h = row_r % h_q
                t = expanded_row // cutlass.Int32(expand_num)
                on_phase = expanded_row - t * cutlass.Int32(expand_num) == cutlass.Int32(expand_phase)
                out_row = cutlass.Int64(row_r)
            else:
                t = row_r // h_q
                h = row_r % h_q
                if cutlass.const_expr(expand_num > 1):
                    out_row = (t * cutlass.Int64(expand_num) + cutlass.Int64(expand_phase)) * cutlass.Int64(h_q) + cutlass.Int64(h)
                else:
                    out_row = cutlass.Int64(row_r)
            src_elements = t * cutlass.Int64(mQ.stride[0]) + h * cutlass.Int64(mQ.stride[1])
            src_addr = mQ.iterator.toint() + src_elements * cutlass.Int64(2)
            workspace_addr = mQn.iterator.toint() + out_row * cutlass.Int64(d) * cutlass.Int64(2)
            nrm_addr = mInvQ.iterator.toint() + out_row * cutlass.Int64(4)
        else:
            k_row = row_r - n_q_rows
            t = k_row // h_k
            h = k_row % h_k
            src_elements = t * cutlass.Int64(mK.stride[0]) + h * cutlass.Int64(mK.stride[1])
            src_addr = mK.iterator.toint() + src_elements * cutlass.Int64(2)
            workspace_addr = mKn.iterator.toint() + cutlass.Int64(k_row) * cutlass.Int64(d) * cutlass.Int64(2)
            nrm_addr = mInvK.iterator.toint() + cutlass.Int64(k_row) * cutlass.Int64(4)
        chunks = []
        for c in cutlass.range_constexpr(vec_chunks):
            voff = (cutlass.Int32(c) * cutlass.Int32(FWD_LANES) + lane_idx) * cutlass.Int32(16)
            packed = [cutlass.Int32(0)] * 4
            if on_phase:
                packed = list(ld_global_v4(src_addr + voff.to(cutlass.Int64), cutlass.Int32))
            pairs = [f16x2_to_f32(w, dtype=mQ.element_type) for w in packed]
            chunks.append(tuple(v for pair in pairs for v in pair))
        rows.append(row)
        workspace_addrs.append(workspace_addr)
        nrm_addrs.append(nrm_addr)
        vals.append(chunks)
    for r in cutlass.range_constexpr(FWD_ROWS_PER_GROUP):
        acc = cutlass.Float32(0.0)
        for c in cutlass.range_constexpr(vec_chunks):
            v = vals[r][c]
            acc = acc + (((v[0] * v[0] + v[1] * v[1]) + (v[2] * v[2] + v[3] * v[3])) + ((v[4] * v[4] + v[5] * v[5]) + (v[6] * v[6] + v[7] * v[7])))
        inv = l2norm_inv(lane_group_sum(acc, FWD_LANES))
        if rows[r] < n_rows:
            for c in cutlass.range_constexpr(vec_chunks):
                voff = (cutlass.Int32(c) * cutlass.Int32(FWD_LANES) + lane_idx) * cutlass.Int32(16)
                f = vals[r][c]
                scaled = [fmul2(f[i], f[i + 1], inv, inv) for i in range(0, len(f), 2)]
                packed = [fp32_to_fp16(lo, hi, dtype=mQ.element_type) for lo, hi in scaled]
                st_global_v4(workspace_addrs[r] + voff.to(cutlass.Int64), packed, cutlass.Int32)
            if lane_idx == cutlass.Int32(0):
                st_global(nrm_addrs[r], inv, cutlass.Float32)
    if cutlass.const_expr(USE_PDL):
        launch_dependent_grids()


@cute.jit
def l2norm_qk_launch(
    q: cute.Tensor,
    k: cute.Tensor,
    q_n: cute.Tensor,
    k_n: cute.Tensor,
    inv_q: cute.Tensor,
    inv_k: cute.Tensor,
    n_q_rows: cutlass.Int32,
    n_rows: cutlass.Int32,
    h_q: cutlass.Int32,
    h_k: cutlass.Int32,
    n_blocks: cutlass.Int32,
    d: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    expand_phase: cutlass.Constexpr[int],
    expand_fill: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    frost_l2norm_qk(q, k, q_n, k_n, inv_q, inv_k, n_q_rows, n_rows, h_q, h_k, d, expand_num, expand_phase, expand_fill).launch(
        grid=(n_blocks, 1, 1), block=(THREADS_PER_CTA, 1, 1), stream=stream, use_pdl=USE_PDL
    )


compiled_cache = {}


class L2NormQkRecipe(NamedTuple):
    """Build-time facts of one q/k normalize launch.  Produced by
    :func:`build_l2norm_qk`."""

    compiled: object
    n_q_rows: int
    n_rows: int
    h_q: int
    h_k: int
    n_blocks: int


def run_l2norm_qk(r, q, k, q_n, k_n, inv_q, inv_k, stream) -> None:
    """The lowered normalize launch: no validation, no key build."""
    r.compiled(q, k, q_n, k_n, inv_q, inv_k, r.n_q_rows, r.n_rows, r.h_q, r.h_k, r.n_blocks, cuda.CUstream(int(stream)))


def build_l2norm_qk(q, k, q_n, k_n, inv_q, inv_k, *, expand_num=1, expand_phase=0, expand_fill=False, skip_q=False, stream) -> L2NormQkRecipe:
    """Compile (cached), run once, and bake the q/k normalize: rows into the
    compact io workspace copies, fp32 inverse norms to their slots.  Sources
    are read through their own strides; ``expand_num > 1`` writes the q rows
    onto sub-token ``expand_phase`` of the expanded workspace, leaving the
    off-phase rows untouched.  ``expand_fill`` instead walks every expanded q
    row and normalizes a zero row on the off-phase ones, so q_n and inv_q come
    out fully written on the expanded timeline.  ``skip_q`` walks the k rows
    alone (the state summaries take no q); q, q_n and inv_q are then never
    read and may alias the k buffers."""
    total, h_q, d = (int(s_) for s_ in q.shape)
    total_k, h_k, d_k = (int(s_) for s_ in k.shape)
    ROWS = (THREADS_PER_CTA // FWD_LANES) * FWD_ROWS_PER_GROUP
    n_q_rows = 0 if skip_q else total * h_q * (int(expand_num) if expand_fill else 1)
    n_rows = n_q_rows + total_k * h_k
    args = (n_q_rows, n_rows, h_q, h_k, (n_rows + ROWS - 1) // ROWS)
    cu_stream = cuda.CUstream(int(stream))
    key = ("fwd", str(q.dtype), int(expand_num), int(expand_phase), bool(expand_fill), d, current_device())
    if key not in compiled_cache:
        tensors = (q, k, q_n, k_n, inv_q, inv_k)
        compiled_cache[key] = cute.compile(
            l2norm_qk_launch,
            *(from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=lead) for t, lead in zip(tensors, (2, 2, 2, 2, 1, 1))),
            *(cutlass.Int32(a) for a in args),
            d,
            int(expand_num),
            int(expand_phase),
            bool(expand_fill),
            cu_stream,
            options="--enable-tvm-ffi",
        )
    r = L2NormQkRecipe(compiled_cache[key], *args)
    run_l2norm_qk(r, q, k, q_n, k_n, inv_q, inv_k, stream)
    return r


frost_l2norm_qk.set_name_prefix("cudnn", remove_cutlass_symbol=False)
