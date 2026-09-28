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

"""One compiled launch for the KDA warmup and uncut backward: the split-K table (warmup only), the recompute prologue and
the checkpoint-series recompute (unless the forward's per-chunk series is passed back), the bprop prologue and the bprop,
issued from a single host at ``--opt-level 2``, the level of every KDA module, its prologues and the split table
(``opt_level``), so every nested kernel is the standalone one.  Every kernel, its host and the tensor placeholder each host was
compiled with are the standalone modules' own; a buffer two hosts read through different placeholder types is passed twice
(the table's 4-byte compact views of work_items, work_count, item_scratch and cu_seqlens; the gate when the bprop reads it
as a linear alpha)."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from ..common import split_k
from ..common.host import get_dtype
from . import kda_bprop_f16, kda_recompute_f16

warmup_backward_cache = {}


@cute.jit
def warmup_backward_host(
    split: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    scan_rows: cutlass.Constexpr[int],
    log_gate: cutlass.Constexpr[bool],
    safe_gate: cutlass.Constexpr[bool],
    gate_channels: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    warmup_cap: cutlass.Constexpr[int],
    full_scan: cutlass.Constexpr[bool],
    n_heads_out: cutlass.Int32,
    num_sms: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    recompute: cutlass.Constexpr[bool],
    recompute_orders: cutlass.Constexpr[bool],
    recompute_order_gen: cutlass.Constexpr[bool],
    coarse: cutlass.Constexpr[bool],
    bwd_orders: cutlass.Constexpr[bool],
    bwd_order_gen: cutlass.Constexpr[bool],
    recompute_cfg: cutlass.Constexpr,
    bprop_cfg: cutlass.Constexpr,
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    gate_scale_log2: cutlass.Float32,
    n_scan_ctas: cutlass.Int32,
    n_scan_blocks: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    seed_span_chunks: cutlass.Int32,
    seed_every_n: cutlass.Int32,
    scale: cutlass.Float32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    gate: cute.Tensor,
    gate_table: Optional[cute.Tensor],
    gate_main: Optional[cute.Tensor],
    beta: cute.Tensor,
    a_log: Optional[cute.Tensor],
    a_log_table: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    dt_bias_table: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    cu_seqlens_table: cute.Tensor,
    checkpoints: cute.Tensor,
    seed_checkpoints: Optional[cute.Tensor],
    state_in: Optional[cute.Tensor],
    dgate: cute.Tensor,
    dbeta: cute.Tensor,
    dstate0: Optional[cute.Tensor],
    dstate_in: Optional[cute.Tensor],
    work_items: cute.Tensor,
    work_items_table: cute.Tensor,
    work_count: cute.Tensor,
    work_count_table: cute.Tensor,
    series_items: Optional[cute.Tensor],
    series_count: Optional[cute.Tensor],
    staging_recompute: Optional[cute.Tensor],
    staging_bprop: Optional[cute.Tensor],
    item_scratch: Optional[cute.Tensor],
    chunk_scratch: Optional[cute.Tensor],
    scheduler_all: cute.Tensor,
    scheduler_all_recompute: Optional[cute.Tensor],
    scheduler_all_bprop: Optional[cute.Tensor],
    scheduler_recompute: cute.Tensor,
    scheduler_bwd: cute.Tensor,
    recompute_words: Optional[cute.Tensor],
    bprop_words: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    heads_out = cutlass.Int32(gate.shape[1])
    q_ratio = heads_out // cutlass.Int32(q.shape[1])
    k_ratio = heads_out // cutlass.Int32(k.shape[1])
    v_ratio = heads_out // cutlass.Int32(v.shape[1])
    if cutlass.const_expr(split):
        split_k.launch(
            split,
            b_t,
            scan_rows,
            log_gate,
            safe_gate,
            gate_channels,
            overhead_chunks,
            expand_num,
            warmup_cap,
            full_scan,
            n_heads_out,
            num_sms,
            n_tiles,
            ideal_chunks,
            batch_size,
            log2_thresh,
            gate_scale_log2,
            gate_table,
            a_log_table,
            dt_bias_table,
            cu_seqlens_table,
            chunk_scratch,
            item_scratch,
            work_items_table,
            work_count_table,
            scheduler_all,
            n_scan_ctas,
            n_scan_blocks,
            n_walk_ctas,
            stream,
        )
    if cutlass.const_expr(recompute):
        kda_recompute_f16.prologue(
            io_dtype,
            b_t,
            recompute_orders,
            recompute_order_gen,
            coarse,
            k,
            v,
            gate,
            checkpoints,
            cu_seqlens,
            staging_recompute,
            series_count,
            series_items,
            scheduler_all_recompute,
            recompute_words,
            checkpoint_every_n,
            seed_span_chunks,
            stream,
        )
        kda_recompute_f16.host(
            recompute_cfg,
            k,
            v,
            gate,
            a_log,
            dt_bias,
            beta,
            cu_seqlens,
            state_in,
            None,
            seed_checkpoints,
            series_items,
            series_count,
            scheduler_recompute,
            recompute_words,
            checkpoint_every_n,
            seed_every_n,
            stream,
        )
    kda_bprop_f16.prologue(
        io_dtype,
        b_t,
        bwd_orders,
        bwd_order_gen,
        q,
        k,
        v,
        gate,
        do,
        dq,
        dk,
        dv,
        dgate,
        checkpoints,
        cu_seqlens,
        staging_bprop,
        work_count,
        work_items,
        scheduler_all_bprop,
        bprop_words,
        stream,
    )
    kda_bprop_f16.host(
        bprop_cfg,
        q_ratio,
        k_ratio,
        v_ratio,
        a_log,
        dt_bias,
        beta,
        gate_main,
        checkpoints,
        dgate,
        dbeta,
        cu_seqlens,
        dstate0,
        dstate_in,
        work_items,
        work_count,
        scheduler_bwd,
        bprop_words,
        scale,
        stream,
    )


def build_warmup_backward(
    *,
    q,
    k,
    v,
    do,
    dq,
    dk,
    dv,
    gate,
    beta,
    a_log,
    dt_bias,
    cu_seqlens,
    checkpoints,
    seed_checkpoints,
    state_in,
    use_initial_state,
    dgate,
    dbeta,
    dstate0,
    dstate_in,
    work_items,
    work_count,
    series_items,
    series_count,
    item_scratch,
    chunk_scratch,
    scheduler_all,
    scheduler_recompute,
    scheduler_bwd,
    recompute_words,
    bprop_words,
    split,
    n_tiles,
    ideal_chunks,
    num_sm,
    b_t,
    recompute,
    recompute_orders,
    coarse,
    bwd_orders,
    seed_span_tokens,
    seed_every_n_tokens,
    log_gate,
    safe_gate,
    gate_lower_bound,
    use_qk_l2norm,
    use_beta_sigmoid,
    allow_neg_eigval,
    scale,
    device,
    stream,
):
    """Compile (cached per static config) the warmup or uncut backward launch over the buffers of one plan.  The
    placeholders repeat the marks of the standalone builds so every kernel compiles as it does there; the recompute and
    bprop prologues read the split table's item scratch as their ordering staging when the plan hands it to them
    (``recompute_orders`` / ``bwd_orders`` with ``split``)."""
    _HQ, DK = q.shape[1], q.shape[2]
    k.shape[1]
    _HV, DV = v.shape[1], v.shape[2]
    gate.shape[1]
    if not safe_gate:
        a_log = None
        dt_bias = None
    facts = split_k.split_table_facts(
        gate,
        cu_seqlens,
        split=split,
        n_tiles=n_tiles,
        ideal_chunks=ideal_chunks,
        num_sms=num_sm,
        b_t=b_t,
        log2_threshold=None,
        log_gate=log_gate,
        safe_gate=safe_gate,
        gate_lower_bound=gate_lower_bound if safe_gate else None,
        expand_num=1,
    )
    io_dtype = get_dtype(q.dtype)
    gate_dtype = get_dtype(gate.dtype)
    gate_scale_log2 = float(gate_lower_bound) * kda_bprop_f16.LOG2_E
    gate_main = not log_gate and not safe_gate
    recompute_staging = recompute and recompute_orders and split
    recompute_order_gen = recompute_orders and not split
    bwd_staging = bwd_orders and split
    bwd_order_gen = bwd_orders and not split
    key = (
        str(q.dtype),
        str(cu_seqlens.dtype),
        str(gate.dtype),
        str(beta.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(state_in.dtype) if state_in is not None else "none",
        str(dstate0.dtype) if dstate0 is not None else "none",
        str(dstate_in.dtype) if dstate_in is not None else "none",
        int(device),
        int(num_sm),
        DK,
        DV,
        int(b_t),
        bool(split),
        facts.scan_rows,
        bool(recompute),
        bool(recompute_orders),
        bool(coarse),
        bool(bwd_orders),
        bool(use_initial_state),
        bool(log_gate),
        bool(safe_gate),
        float(gate_lower_bound),
        bool(use_qk_l2norm),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
    )
    if key not in warmup_backward_cache:
        flags = dict(
            l2norm=use_qk_l2norm,
            safe_gate=safe_gate,
            gate_scale_log2=gate_scale_log2,
            log_gate=log_gate,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            max_active_clusters=num_sm,
            d_k=DK,
        )
        recompute_cfg = None
        if recompute:
            recompute_cfg = kda_recompute_f16.build_cfg(
                io_dtype,
                get_dtype(state_in.dtype) if state_in is not None else cutlass.Float32,
                gate_dtype,
                use_initial_state=state_in is not None,
                store_final_state=False,
                enable_checkpoints=True,
                seed_checkpoints=coarse,
                d_v=DV,
                **flags,
            )
        bprop_cfg = kda_bprop_f16.build_cfg(
            io_dtype,
            gate_dtype,
            use_dstate_in=dstate_in is not None,
            use_dstate0=dstate0 is not None,
            use_initial_state=use_initial_state,
            d_v=DV,
            **flags,
        )
        gate_table_placeholder = None
        dt_bias_table_placeholder = None
        staging_placeholder = None
        item_scratch_placeholder = None
        chunk_scratch_placeholder = None
        if split:
            gate_table_placeholder = from_dlpack(gate, assumed_align=8 if facts.gate_elem_bytes == 2 else 4).mark_layout_dynamic(
                leading_dim=len(gate.shape) - 1
            )
            staging_placeholder = from_dlpack(item_scratch, assumed_align=16)
            staging_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            item_scratch_placeholder = from_dlpack(item_scratch, assumed_align=4)
            item_scratch_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
            chunk_scratch_placeholder = from_dlpack(chunk_scratch, assumed_align=4)
            chunk_scratch_placeholder.mark_layout_dynamic(leading_dim=1)
        if dt_bias is not None:
            dt_bias_table_placeholder = from_dlpack(dt_bias, assumed_align=4)
            dt_bias_table_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=tuple(range(len(dt_bias.shape))), divisibility=1)
        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_items_table_placeholder = from_dlpack(work_items, assumed_align=4)
        work_items_table_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_count_table_placeholder = from_dlpack(work_count, assumed_align=4)
        work_count_table_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0,), divisibility=1)
        series_items_placeholder = None
        if recompute:
            series_items_placeholder = from_dlpack(series_items, assumed_align=16)
            series_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        warmup_backward_cache[key] = cute.compile(
            warmup_backward_host,
            facts.split,
            facts.b_t,
            facts.scan_rows,
            facts.log_gate,
            facts.safe_gate,
            facts.gate_channels,
            facts.overhead_chunks,
            facts.expand_num,
            facts.warmup_cap,
            facts.full_scan,
            cutlass.Int32(facts.n_heads_out),
            facts.num_sms,
            io_dtype,
            bool(recompute),
            bool(recompute_orders),
            recompute_order_gen,
            bool(coarse),
            bool(bwd_orders),
            bwd_order_gen,
            recompute_cfg,
            bprop_cfg,
            cutlass.Int32(facts.n_tiles),
            cutlass.Int32(facts.ideal_chunks),
            cutlass.Int32(facts.batch_size),
            cutlass.Float32(facts.log2_threshold),
            cutlass.Float32(facts.gate_scale_log2),
            cutlass.Int32(facts.n_scan_ctas),
            cutlass.Int32(facts.n_scan_blocks),
            cutlass.Int32(facts.n_walk_ctas),
            cutlass.Int32(int(b_t)),
            cutlass.Int32(int(seed_span_tokens or seed_every_n_tokens) // int(b_t)),
            cutlass.Int32(int(seed_every_n_tokens)),
            float(scale),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(do, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dq, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dk, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dv, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            gate_table_placeholder,
            from_dlpack(gate, assumed_align=4).mark_layout_dynamic(leading_dim=2) if gate_main else None,
            from_dlpack(beta, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=16).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            dt_bias_table_placeholder,
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(seed_checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3) if coarse else None,
            from_dlpack(state_in, assumed_align=16).mark_layout_dynamic(leading_dim=3) if state_in is not None else None,
            from_dlpack(dgate, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(dbeta, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(dstate0, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dstate0 is not None else None,
            from_dlpack(dstate_in, assumed_align=16).mark_layout_dynamic(leading_dim=3) if dstate_in is not None else None,
            work_items_placeholder,
            work_items_table_placeholder,
            from_dlpack(work_count, assumed_align=4).mark_layout_dynamic(),
            work_count_table_placeholder,
            series_items_placeholder,
            from_dlpack(series_count, assumed_align=4).mark_layout_dynamic() if recompute else None,
            staging_placeholder if recompute_staging else None,
            staging_placeholder if bwd_staging else None,
            item_scratch_placeholder,
            chunk_scratch_placeholder,
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic() if recompute and (recompute_orders or coarse) else None,
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic() if bwd_orders else None,
            from_dlpack(scheduler_recompute, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_bwd, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(recompute_words, assumed_align=128).mark_layout_dynamic() if recompute else None,
            from_dlpack(bprop_words, assumed_align=128).mark_layout_dynamic(),
            cuda.CUstream(int(stream)),
            options="--enable-tvm-ffi --opt-level 2",
        )
    return warmup_backward_cache[key], facts


def run_warmup_backward(
    compiled,
    facts,
    *,
    q,
    k,
    v,
    do,
    dq,
    dk,
    dv,
    gate,
    beta,
    a_log,
    dt_bias,
    cu_seqlens,
    checkpoints,
    seed_checkpoints,
    state_in,
    dgate,
    dbeta,
    dstate0,
    dstate_in,
    work_items,
    work_count,
    series_items,
    series_count,
    item_scratch,
    chunk_scratch,
    scheduler_all,
    scheduler_recompute,
    scheduler_bwd,
    recompute_words,
    bprop_words,
    b_t,
    recompute,
    recompute_orders,
    coarse,
    bwd_orders,
    seed_span_tokens,
    seed_every_n_tokens,
    scale,
    stream,
) -> None:
    """Replay the warmup or uncut backward: one crossing into the DSL.  The plan validated the contract at build, so
    nothing here raises."""
    compiled(
        facts.n_heads_out,
        facts.n_tiles,
        facts.ideal_chunks,
        facts.batch_size,
        facts.log2_threshold,
        facts.gate_scale_log2,
        facts.n_scan_ctas,
        facts.n_scan_blocks,
        facts.n_walk_ctas,
        int(b_t),
        int(seed_span_tokens or seed_every_n_tokens) // int(b_t),
        int(seed_every_n_tokens),
        float(scale),
        q,
        k,
        v,
        do,
        dq,
        dk,
        dv,
        gate,
        gate if facts.split else None,
        gate if not facts.log_gate and not facts.safe_gate else None,
        beta,
        a_log if facts.safe_gate else None,
        a_log if facts.safe_gate else None,
        dt_bias if facts.safe_gate else None,
        dt_bias if facts.safe_gate else None,
        cu_seqlens,
        cu_seqlens,
        checkpoints,
        seed_checkpoints if coarse else None,
        state_in,
        dgate,
        dbeta,
        dstate0,
        dstate_in,
        work_items,
        work_items,
        work_count,
        work_count,
        series_items if recompute else None,
        series_count if recompute else None,
        item_scratch if recompute and recompute_orders and facts.split else None,
        item_scratch if bwd_orders and facts.split else None,
        item_scratch if facts.split else None,
        chunk_scratch if facts.split else None,
        scheduler_all,
        scheduler_all if recompute and (recompute_orders or coarse) else None,
        scheduler_all if bwd_orders else None,
        scheduler_recompute,
        scheduler_bwd,
        recompute_words if recompute else None,
        bprop_words,
        cuda.CUstream(int(stream)),
    )
