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
# attn_gym.linear._delta_rule.cudnn_fe. The host compiles through a persisted ``jit_cache`` function
# over fake TVM-FFI signatures and launches on the current Torch stream.

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

from attn_gym._backends.cute import compile_tvm_ffi, jit_cache

from ..common import split_k
from ..common.host import get_dtype
from ..common.tvm_ffi import (
    WORK_ITEM_FIELDS,
    make_compact_signature_tensor,
    make_counter_signature,
    make_dynamic_signature_tensor,
)
from . import kda_bprop_f16, kda_recompute_f16


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


@jit_cache
def _compile_warmup_backward(
    facts_static: tuple,
    io_dtype,
    gate_dtype,
    beta_dtype,
    a_log_dtype,
    dt_bias_spec,
    state_dtype,
    cu_seqlens_dtype,
    d_k: int,
    d_v: int,
    recompute: bool,
    recompute_orders: bool,
    coarse: bool,
    bwd_orders: bool,
    use_initial_state: bool,
    has_state_in: bool,
    has_dstate0: bool,
    has_dstate_in: bool,
    log_gate: bool,
    safe_gate: bool,
    gate_scale_log2: float,
    use_qk_l2norm: bool,
    use_beta_sigmoid: bool,
    allow_neg_eigval: bool,
    use_int64_offsets: bool,
):
    """Compile the warmup / uncut backward host over fake tensors that repeat the standalone
    kernels' dynamic-layout marks (int32 extents, int64 outer strides, last mode contiguous)."""
    (
        split,
        b_t,
        scan_rows,
        gate_channels,
        overhead_chunks,
        expand_num,
        warmup_cap,
        full_scan,
        num_sms,
    ) = facts_static
    sym_int = cute.sym_int

    def strided(dtype, rank, align):
        return make_dynamic_signature_tensor(
            dtype, rank, assumed_align=align, use_int64_offsets=use_int64_offsets
        )

    def table(align):
        return make_compact_signature_tensor(
            cutlass.Int32, (sym_int(), WORK_ITEM_FIELDS), assumed_align=align
        )

    gate_main = not log_gate and not safe_gate
    recompute_order_gen = recompute_orders and not split
    bwd_order_gen = bwd_orders and not split
    recompute_staging = recompute and recompute_orders and split
    bwd_staging = bwd_orders and split
    flags = dict(
        l2norm=use_qk_l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        log_gate=log_gate,
        beta_sigmoid=use_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        max_active_clusters=num_sms,
        d_k=d_k,
    )
    recompute_cfg = None
    if recompute:
        recompute_cfg = kda_recompute_f16.build_cfg(
            io_dtype,
            state_dtype if has_state_in else cutlass.Float32,
            gate_dtype,
            use_initial_state=has_state_in,
            store_final_state=False,
            enable_checkpoints=True,
            seed_checkpoints=coarse,
            d_v=d_v,
            **flags,
        )
    bprop_cfg = kda_bprop_f16.build_cfg(
        io_dtype,
        gate_dtype,
        use_dstate_in=has_dstate_in,
        use_dstate0=has_dstate0,
        use_initial_state=use_initial_state,
        d_v=d_v,
        **flags,
    )
    dt_bias_table = None
    if dt_bias_spec is not None:
        dt_bias_dtype, dt_bias_tail = dt_bias_spec
        dt_bias_table = make_compact_signature_tensor(
            dt_bias_dtype, (sym_int(), *dt_bias_tail), assumed_align=4
        )
    static = (
        split,
        b_t,
        scan_rows,
        gate_channels,
        overhead_chunks,
        expand_num,
        warmup_cap,
        full_scan,
        num_sms,
        d_k,
        d_v,
        recompute,
        recompute_orders,
        coarse,
        bwd_orders,
        use_initial_state,
        has_state_in,
        has_dstate0,
        has_dstate_in,
        log_gate,
        safe_gate,
        use_qk_l2norm,
        use_beta_sigmoid,
        allow_neg_eigval,
        use_int64_offsets,
    )
    dtype_names = "_".join(
        "none" if dtype is None else dtype.__name__.lower()
        for dtype in (io_dtype, gate_dtype, beta_dtype, a_log_dtype, state_dtype, cu_seqlens_dtype)
    )
    name = (
        "kda_warmup_backward_" + "_".join(str(int(flag)) for flag in static) + f"_{dtype_names}"
        f"_g{str(gate_scale_log2).replace('.', 'p').replace('-', 'm')}"
    )
    dt_bias_rank = 1 if dt_bias_spec is None else 1 + len(dt_bias_spec[1])
    return compile_tvm_ffi(
        warmup_backward_host,
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
        cutlass.Int32(0),
        num_sms,
        io_dtype,
        recompute,
        recompute_orders,
        recompute_order_gen,
        coarse,
        bwd_orders,
        bwd_order_gen,
        recompute_cfg,
        bprop_cfg,
        *(cutlass.Int32(0) for _ in range(3)),
        cutlass.Float32(0),
        cutlass.Float32(0),
        *(cutlass.Int32(0) for _ in range(6)),
        cutlass.Float32(0),
        strided(io_dtype, 3, 16),  # q
        strided(io_dtype, 3, 16),  # k
        strided(io_dtype, 3, 16),  # v
        strided(io_dtype, 3, 16),  # do
        strided(io_dtype, 3, 16),  # dq
        strided(io_dtype, 3, 16),  # dk
        strided(io_dtype, 3, 16),  # dv
        strided(gate_dtype, 3, 16),  # gate
        strided(gate_dtype, 3, 8 if gate_dtype.width == 16 else 4) if split else None,  # gate_table
        strided(gate_dtype, 3, 4) if gate_main else None,  # gate_main
        strided(beta_dtype, 2, 4),  # beta
        strided(a_log_dtype, 1, 4) if a_log_dtype is not None else None,  # a_log
        strided(a_log_dtype, 1, 4) if a_log_dtype is not None else None,  # a_log_table
        strided(dt_bias_spec[0], dt_bias_rank, 16) if dt_bias_spec is not None else None,  # dt_bias
        dt_bias_table,
        strided(cu_seqlens_dtype, 1, 8 if cu_seqlens_dtype is cutlass.Int64 else 4),  # cu_seqlens
        strided(cu_seqlens_dtype, 1, 4),  # cu_seqlens_table
        strided(io_dtype, 4, 16),  # checkpoints
        strided(io_dtype, 4, 16) if coarse else None,  # seed_checkpoints
        strided(state_dtype, 4, 16) if has_state_in else None,  # state_in
        strided(gate_dtype, 3, 16),  # dgate
        strided(beta_dtype, 2, 4),  # dbeta
        strided(cutlass.Float32, 4, 16) if has_dstate0 else None,  # dstate0
        strided(cutlass.Float32, 4, 16) if has_dstate_in else None,  # dstate_in
        table(16),  # work_items
        table(4),  # work_items_table
        strided(cutlass.Int32, 1, 4),  # work_count
        make_counter_signature(sym_int()),  # work_count_table
        table(16) if recompute else None,  # series_items
        strided(cutlass.Int32, 1, 4) if recompute else None,  # series_count
        table(16) if recompute_staging else None,  # staging_recompute
        table(16) if bwd_staging else None,  # staging_bprop
        table(4) if split else None,  # item_scratch
        strided(cutlass.Float32, 2, 4) if split else None,  # chunk_scratch
        strided(cutlass.Int32, 1, 4),  # scheduler_all
        strided(cutlass.Int32, 1, 4) if recompute and (recompute_orders or coarse) else None,
        strided(cutlass.Int32, 1, 4) if bwd_orders else None,
        strided(cutlass.Int32, 1, 4),  # scheduler_recompute
        strided(cutlass.Int32, 1, 4),  # scheduler_bwd
        strided(cutlass.Int64, 1, 128) if recompute else None,  # recompute_words
        strided(cutlass.Int64, 1, 128),  # bprop_words
        name=name,
        opt_level=2,
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
    """Compile (persisted per static config) the warmup or uncut backward launch over the buffers of one plan.  The
    fake signatures repeat the marks of the standalone builds so every kernel compiles as it does there; the recompute and
    bprop prologues read the split table's item scratch as their ordering staging when the plan hands it to them
    (``recompute_orders`` / ``bwd_orders`` with ``split``)."""
    DK = q.shape[2]
    DV = v.shape[2]
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
    shared = (gate, beta, a_log, dt_bias, cu_seqlens, checkpoints, seed_checkpoints, state_in)
    recompute_tensors = (k, v, series_items, series_count, scheduler_recompute, recompute_words)
    bprop_tensors = (
        q, k, v, do, dq, dk, dv, dgate, dbeta, dstate0, dstate_in, work_items, work_count,
        item_scratch, chunk_scratch, scheduler_all, scheduler_bwd, bprop_words,
    )
    # Each kernel module owns its ABI selector (tests monkeypatch them); the bundled host
    # widens every signature when either kernel needs int64 offsets.
    use_int64_offsets = kda_bprop_f16.requires_int64_abi(
        *(t for t in shared + bprop_tensors if t is not None)
    ) or (
        recompute
        and kda_recompute_f16.requires_int64_abi(
            *(t for t in shared + recompute_tensors if t is not None)
        )
    )
    compiled = _compile_warmup_backward(
        (
            bool(facts.split),
            int(facts.b_t),
            int(facts.scan_rows),
            int(facts.gate_channels),
            int(facts.overhead_chunks),
            int(facts.expand_num),
            int(facts.warmup_cap),
            bool(facts.full_scan),
            int(facts.num_sms),
        ),
        get_dtype(q.dtype),
        get_dtype(gate.dtype),
        get_dtype(beta.dtype),
        get_dtype(a_log.dtype) if a_log is not None else None,
        (get_dtype(dt_bias.dtype), tuple(int(n) for n in dt_bias.shape[1:]))
        if dt_bias is not None
        else None,
        get_dtype(state_in.dtype) if state_in is not None else cutlass.Float32,
        cutlass.Int64 if str(cu_seqlens.dtype).endswith("int64") else cutlass.Int32,
        int(DK),
        int(DV),
        bool(recompute),
        bool(recompute_orders),
        bool(coarse),
        bool(bwd_orders),
        bool(use_initial_state),
        state_in is not None,
        dstate0 is not None,
        dstate_in is not None,
        bool(log_gate),
        bool(safe_gate),
        float(gate_lower_bound) * kda_bprop_f16.LOG2_E,
        bool(use_qk_l2norm),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        use_int64_offsets,
    )
    return compiled, facts


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
    """Replay the warmup or uncut backward: one crossing into the DSL on the current Torch stream.  The plan validated
    the contract at build, so nothing here raises."""
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
    )
