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
# Modified by Attention Gym in 2026: persistent jit_cache compile on fake-tensor TVM-FFI signatures
# (legacy placeholder ABI) with an int64-shape variant.

"""One compiled launch for everything ahead of the bprop on the GDN warmup and uncut backward: the T pass (with its own
descriptor prologue), the split-K table (warmup only), the recompute prologue and the checkpoint-series recompute (unless
the forward's per-chunk series is passed back) and the bprop prologue, all at ``--opt-level 2`` like their standalone
builds; the bprop itself keeps its standalone ``--opt-level 2`` compile (the bprop module's ``chunk_gdn_bwd`` /
``run_bwd`` without their prologue), so the call sequence is two crossings into the DSL instead of six.  Every kernel, its
host and the tensor placeholder each host was compiled with are the standalone modules' own; a buffer two hosts read
through different placeholder types is passed twice (the table's 4-byte compact views of work_items, work_count and
item_scratch; the bprop prologue's 4-byte cu_seqlens).  The bprop module is a constexpr argument: GDP at d_v = 64 runs
the gdp_bprop_v64_f16 fork, whose prologue takes no T-pass tiles."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from attn_gym._backends.cute import compile_tvm_ffi, jit_cache
from attn_gym._backends.cute.utils import requires_int64_abi

from ..common import split_k
from ..common.host import get_dtype
from ..common.tvm_ffi import make_signature, signature_spec
from . import gdn_bprop_f16, gdn_recompute_f16, gdn_tinv_f16


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
    bprop: cutlass.Constexpr,
    compact_qdo: cutlass.Constexpr[bool],
    tinv_pass: cutlass.Constexpr[bool],
    recompute: cutlass.Constexpr[bool],
    recompute_orders: cutlass.Constexpr[bool],
    recompute_order_gen: cutlass.Constexpr[bool],
    coarse: cutlass.Constexpr[bool],
    bwd_orders: cutlass.Constexpr[bool],
    bwd_order_gen: cutlass.Constexpr[bool],
    tinv_cfg: cutlass.Constexpr,
    recompute_cfg: cutlass.Constexpr,
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
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    gate: cute.Tensor,
    gate_table: Optional[cute.Tensor],
    beta: cute.Tensor,
    a_log: Optional[cute.Tensor],
    a_log_table: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    dt_bias_table: Optional[cute.Tensor],
    cu_seqlens: cute.Tensor,
    cu_seqlens_table: cute.Tensor,
    cu_seqlens_bprop: cute.Tensor,
    tinv: Optional[cute.Tensor],
    tinv_words: Optional[cute.Tensor],
    tinv_rows: Optional[cute.Tensor],
    tinv_row_count: Optional[cute.Tensor],
    checkpoints: cute.Tensor,
    seed_checkpoints: Optional[cute.Tensor],
    state_in: Optional[cute.Tensor],
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
    recompute_words: Optional[cute.Tensor],
    bprop_words: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    if cutlass.const_expr(tinv_pass):
        gdn_tinv_f16.host(tinv_cfg, True, k, tinv_words, gate, a_log, dt_bias, beta, cu_seqlens, tinv, tinv_rows, tinv_row_count, stream)
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
        gdn_recompute_f16.prologue(
            io_dtype,
            b_t,
            recompute_orders,
            recompute_order_gen,
            coarse,
            expand_num,
            k,
            v,
            gate,
            cu_seqlens,
            checkpoints,
            staging_recompute,
            series_count,
            series_items,
            scheduler_all_recompute,
            checkpoint_every_n,
            seed_span_chunks,
            tinv,
            recompute_words,
            stream,
        )
        gdn_recompute_f16.host(
            recompute_cfg,
            k,
            v,
            gate,
            a_log,
            dt_bias,
            cu_seqlens,
            state_in,
            None,
            seed_checkpoints,
            tinv,
            series_items,
            series_count,
            scheduler_recompute,
            checkpoint_every_n,
            seed_every_n,
            recompute_words,
            stream,
        )
    if cutlass.const_expr(compact_qdo):
        bprop.prologue(
            io_dtype,
            b_t,
            bwd_orders,
            bwd_order_gen,
            expand_num,
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
            checkpoints,
            cu_seqlens_bprop,
            staging_bprop,
            work_count,
            work_items,
            scheduler_all_bprop,
            bprop_words,
            stream,
        )
    else:
        bprop.prologue(
            io_dtype,
            b_t,
            bwd_orders,
            bwd_order_gen,
            expand_num,
            q,
            k,
            v,
            do,
            dq,
            dk,
            dv,
            checkpoints,
            cu_seqlens_bprop,
            staging_bprop,
            work_count,
            work_items,
            scheduler_all_bprop,
            tinv,
            bprop_words,
            stream,
        )


@jit_cache
def _compile_warmup_backward(constexprs: tuple, cfg_args: tuple, specs: tuple, use_int64_offsets: bool):
    """Compile the warmup / uncut backward head over the upstream dynamic-layout tensor ABI.

    ``constexprs`` are the host's static arguments ahead of ``tinv_cfg`` (with the bprop module
    slot left as None), ``cfg_args`` rebuild the T-pass and recompute configs, and ``specs``
    describe every tensor argument in host order."""
    (io_dtype, state_dtype, num_sm, d_k, d_v, expand_num, tinv_pass, recompute, coarse, log_gate, safe_gate,
     use_beta_sigmoid, allow_neg_eigval, state_in) = cfg_args
    io = get_dtype(io_dtype)
    tinv_cfg = None
    if tinv_pass:
        tinv_cfg = gdn_tinv_f16.build_cfg(
            io,
            num_sm=num_sm,
            log_gate=log_gate,
            safe_gate=safe_gate,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            d_k=d_k,
            expand_num=expand_num,
        )
    recompute_cfg = None
    if recompute:
        recompute_cfg = gdn_recompute_f16.build_cfg(
            io,
            get_dtype(state_dtype),
            max_active_clusters=num_sm,
            use_initial_state=state_in,
            store_final_state=False,
            enable_checkpoints=True,
            seed_checkpoints=coarse,
            log_gate=log_gate,
            safe_gate=safe_gate,
            d_k=d_k,
            d_v=d_v,
            expand_num=expand_num,
        )
    head = list(constexprs)
    head[BPROP_SLOT] = gdn_bprop_f16
    head[IO_DTYPE_SLOT] = io
    head[N_HEADS_SLOT] = cutlass.Int32(0)
    scalars = [cutlass.Int32(0)] * 3 + [cutlass.Float32(0.0)] * 2 + [cutlass.Int32(0)] * 6
    flags = "_".join(str(int(v)) if isinstance(v, bool) else str(v) for v in constexprs if v is not None)
    return compile_tvm_ffi(
        warmup_backward_host,
        *head,
        tinv_cfg,
        recompute_cfg,
        *scalars,
        *(make_signature(spec, use_int64_offsets=use_int64_offsets) for spec in specs),
        name=f"gdn_cudnn_warmup_backward_{flags}_i64{int(use_int64_offsets)}".lower(),
        opt_level=2,
    )


# Positions in the host's leading static-argument block (see ``warmup_backward_host``).
N_HEADS_SLOT = 10
IO_DTYPE_SLOT = 12
BPROP_SLOT = 13


def build_warmup_backward(
    *,
    bprop_module,
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
    tinv,
    tinv_words,
    tinv_rows,
    tinv_row_count,
    checkpoints,
    seed_checkpoints,
    state_in,
    work_items,
    work_count,
    series_items,
    series_count,
    item_scratch,
    chunk_scratch,
    scheduler_all,
    scheduler_recompute,
    recompute_words,
    bprop_words,
    split,
    n_tiles,
    ideal_chunks,
    num_sm,
    b_t,
    expand_num,
    tinv_pass,
    recompute,
    recompute_orders,
    coarse,
    bwd_orders,
    compact_qdo,
    seed_span_tokens,
    seed_every_n_tokens,
    log_gate,
    safe_gate,
    use_beta_sigmoid,
    allow_neg_eigval,
    device,
    stream,
):
    """Compile (cached per static config) the head of the warmup or uncut backward over the buffers of one plan.  The
    placeholders repeat the marks of the standalone builds so every kernel compiles as it does there."""
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
        gate_lower_bound=None,
        expand_num=expand_num,
    )
    if bprop_module is not gdn_bprop_f16 or compact_qdo:
        raise ValueError("the GDN warmup backward supports only the gdn_bprop_f16 bprop without compact_qdo")
    recompute_order_gen = not (recompute_orders and split)
    bwd_order_gen = bwd_orders and not split
    cu_align = 8 if str(cu_seqlens.dtype).endswith("int64") else 4
    io_dtype = str(q.dtype).removeprefix("torch.")
    constexprs = (
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
        None,  # n_heads_out is a runtime scalar
        facts.num_sms,
        io_dtype,
        None,  # the bprop module
        bool(compact_qdo),
        bool(tinv_pass),
        bool(recompute),
        bool(recompute_orders),
        recompute_order_gen,
        bool(coarse),
        bool(bwd_orders),
        bwd_order_gen,
    )
    cfg_args = (
        io_dtype,
        str(state_in.dtype).removeprefix("torch.") if state_in is not None else "float32",
        int(num_sm),
        DK,
        DV,
        int(expand_num),
        bool(tinv_pass),
        bool(recompute),
        bool(coarse),
        bool(log_gate),
        bool(safe_gate),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        state_in is not None,
    )
    gate_table = gate if split else None
    staging = item_scratch if split else None
    specs = (
        signature_spec(q, assumed_align=16),
        signature_spec(k, assumed_align=16),
        signature_spec(v, assumed_align=16),
        signature_spec(do, assumed_align=16),
        signature_spec(dq, assumed_align=16),
        signature_spec(dk, assumed_align=16),
        signature_spec(dv, assumed_align=16),
        signature_spec(gate, assumed_align=16),
        signature_spec(gate_table, assumed_align=8 if facts.gate_elem_bytes == 2 else 4),
        signature_spec(beta, assumed_align=16),
        signature_spec(a_log, assumed_align=4),
        signature_spec(a_log, assumed_align=4),
        signature_spec(dt_bias, assumed_align=4),
        signature_spec(dt_bias, assumed_align=4, compact=True),
        signature_spec(cu_seqlens, assumed_align=cu_align),
        signature_spec(cu_seqlens, assumed_align=4),
        signature_spec(cu_seqlens, assumed_align=4),
        signature_spec(tinv, assumed_align=128) if tinv_pass else None,
        signature_spec(tinv_words, assumed_align=128) if tinv_pass else None,
        signature_spec(tinv_rows, assumed_align=16) if tinv_pass else None,
        signature_spec(tinv_row_count, assumed_align=4) if tinv_pass else None,
        signature_spec(checkpoints, assumed_align=16),
        signature_spec(seed_checkpoints, assumed_align=16) if coarse else None,
        signature_spec(state_in, assumed_align=16),
        signature_spec(work_items, assumed_align=16, compact=True),
        signature_spec(work_items, assumed_align=4, compact=True),
        signature_spec(work_count, assumed_align=4),
        signature_spec(work_count, assumed_align=4, compact=True),
        signature_spec(series_items, assumed_align=16, compact=True) if recompute else None,
        signature_spec(series_count, assumed_align=4) if recompute else None,
        signature_spec(staging, assumed_align=16, compact=True) if recompute and recompute_orders else None,
        signature_spec(staging, assumed_align=16, compact=True) if bwd_orders else None,
        signature_spec(staging, assumed_align=4, compact=True),
        signature_spec(chunk_scratch if split else None, assumed_align=4),
        signature_spec(scheduler_all, assumed_align=4),
        signature_spec(scheduler_all, assumed_align=4) if recompute and (recompute_orders or coarse) else None,
        signature_spec(scheduler_all, assumed_align=4) if bwd_orders else None,
        signature_spec(scheduler_recompute, assumed_align=4),
        signature_spec(recompute_words, assumed_align=128) if recompute else None,
        signature_spec(bprop_words, assumed_align=128),
    )
    use_int64_offsets = requires_int64_abi(
        q, k, v, do, dq, dk, dv, gate, beta, a_log, dt_bias, cu_seqlens, tinv, checkpoints, seed_checkpoints, state_in,
        recompute_words, bprop_words,
    )
    return _compile_warmup_backward(constexprs, cfg_args, specs, use_int64_offsets), facts


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
    tinv,
    tinv_words,
    tinv_rows,
    tinv_row_count,
    checkpoints,
    seed_checkpoints,
    state_in,
    work_items,
    work_count,
    series_items,
    series_count,
    item_scratch,
    chunk_scratch,
    scheduler_all,
    scheduler_recompute,
    recompute_words,
    bprop_words,
    b_t,
    tinv_pass,
    recompute,
    recompute_orders,
    coarse,
    bwd_orders,
    compact_qdo,
    seed_span_tokens,
    seed_every_n_tokens,
    stream,
) -> None:
    """Replay the head of the warmup or uncut backward: one crossing into the DSL.  The plan validated the contract at
    build, so nothing here raises."""
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
        q,
        k,
        v,
        do,
        dq,
        dk,
        dv,
        gate,
        gate if facts.split else None,
        beta,
        a_log if facts.safe_gate else None,
        a_log if facts.safe_gate else None,
        dt_bias if facts.safe_gate else None,
        dt_bias if facts.safe_gate else None,
        cu_seqlens,
        cu_seqlens,
        cu_seqlens,
        tinv if tinv_pass else None,
        tinv_words if tinv_pass else None,
        tinv_rows if tinv_pass else None,
        tinv_row_count if tinv_pass else None,
        checkpoints,
        seed_checkpoints if coarse else None,
        state_in,
        work_items,
        work_items,
        work_count,
        work_count,
        series_items if recompute else None,
        series_count if recompute else None,
        item_scratch if facts.split and recompute and recompute_orders else None,
        item_scratch if facts.split and bwd_orders else None,
        item_scratch if facts.split else None,
        chunk_scratch if facts.split else None,
        scheduler_all,
        scheduler_all if recompute and (recompute_orders or coarse) else None,
        scheduler_all if bwd_orders else None,
        scheduler_recompute,
        recompute_words if recompute else None,
        bprop_words,
    )
