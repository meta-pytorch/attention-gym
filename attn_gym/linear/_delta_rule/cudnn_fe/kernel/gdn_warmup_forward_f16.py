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
# attn_gym.linear._delta_rule.cudnn_fe; the launch compiles once per static configuration through
# jit_cache over fake TVM-FFI tensor signatures and runs on the environment stream.

"""One compiled launch for the GDN warmup, uncut and dv_split forwards: the split-K table (plan, scan and walk, warmup only), the
prefill prologue and the prefill issued from a single host, the way ``split_k.launch`` already sequences its three kernels.
Every kernel, its host and the tensor placeholder each host was compiled with are the standalone modules' own; this host
only sequences the launches, so the kernels' SASS is unchanged and the Python side crosses into the DSL once per call
instead of two or three times.  A buffer that two hosts read through different signature types is passed twice, once per
type: the table reads the gate at its element alignment, dt_bias, work_items, work_count and item_scratch as 4-byte
views, cu_seqlens at 4 bytes; the prologue and prefill read the same buffers at the standalone prefill wrapper's
alignments."""

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
    make_cu_seqlens_signature,
    make_strided_signature_tensor,
    make_workspace_signature,
)
from . import gdn_prefill_f16

OPT_LEVEL = 2


@cute.jit
def warmup_forward_host(
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
    order_gen: cutlass.Constexpr[bool],
    prefill_cfg: cutlass.Constexpr,
    tiles_per_head: cutlass.Constexpr[int],
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    gate_scale_log2: cutlass.Float32,
    n_scan_ctas: cutlass.Int32,
    n_scan_blocks: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    scale: cutlass.Float32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    gate_table: Optional[cute.Tensor],
    a_log: Optional[cute.Tensor],
    a_log_table: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    dt_bias_table: Optional[cute.Tensor],
    beta: Optional[cute.Tensor],
    o: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_seqlens_table: cute.Tensor,
    state_in: Optional[cute.Tensor],
    state_out: Optional[cute.Tensor],
    seed_indices: Optional[cute.Tensor],
    final_indices: Optional[cute.Tensor],
    has_initial_state: Optional[cute.Tensor],
    checkpoints: Optional[cute.Tensor],
    work_items: cute.Tensor,
    work_items_table: cute.Tensor,
    work_count: cute.Tensor,
    work_count_table: cute.Tensor,
    staging: Optional[cute.Tensor],
    item_scratch: Optional[cute.Tensor],
    chunk_scratch: Optional[cute.Tensor],
    scheduler: cute.Tensor,
    workspace: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
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
            scheduler,
            n_scan_ctas,
            n_scan_blocks,
            n_walk_ctas,
            stream,
        )
    gdn_prefill_f16.prologue(
        io_dtype,
        b_t,
        order_gen,
        expand_num,
        q,
        k,
        v,
        o,
        cu_seqlens,
        checkpoints,
        staging,
        work_count,
        work_items,
        scheduler,
        checkpoint_every_n,
        None,
        workspace,
        stream,
        tiles_per_head,
        seed_indices if cutlass.const_expr(has_initial_state is not None) else None,
        has_initial_state,
    )
    gdn_prefill_f16.host(
        prefill_cfg,
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        o,
        cu_seqlens,
        state_in,
        state_out,
        seed_indices,
        final_indices,
        has_initial_state,
        None,
        work_items,
        work_count,
        scheduler,
        checkpoint_every_n,
        scale,
        workspace,
        stream,
    )


@jit_cache
def _compile_warmup_forward(
    io_dtype,
    state_dtype,
    gate_dtype,
    a_log_dtype,
    bias_spec,
    beta_dtype,
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
    num_sms,
    use_initial_state,
    store_final_state,
    has_seed_indices,
    has_final_indices,
    has_initial_state,
    enable_checkpoints,
    use_beta_sigmoid,
    allow_neg_eigval,
    d_k,
    d_v,
    tiles_per_head,
):
    """Compile the warmup or uncut forward launch for one static configuration (dtypes, gate and
    state flags, split-table geometry, d_v split, device target); every extent is symbolic."""
    sym_int = cute.sym_int

    def tensor(dtype, rank, align):
        return make_strided_signature_tensor(
            dtype,
            tuple(sym_int() for _ in range(rank)),
            assumed_align=align,
            use_int64_offsets=True,
            stride_divisibility=1,
        )

    def work_items(align):
        return make_compact_signature_tensor(cutlass.Int32, (sym_int(), WORK_ITEM_FIELDS), assumed_align=align)

    prefill_cfg = gdn_prefill_f16.build_cfg(
        io_dtype,
        state_dtype,
        max_active_clusters=num_sms,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        log_gate=log_gate,
        safe_gate=safe_gate,
        beta_sigmoid=use_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        tinv_source="compute",
        d_k=d_k,
        d_v=d_v // tiles_per_head,
        expand_num=expand_num,
        tiles_per_head=tiles_per_head,
    )
    flags = (
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
        num_sms,
        use_initial_state,
        store_final_state,
        has_seed_indices,
        has_final_indices,
        has_initial_state,
        enable_checkpoints,
        use_beta_sigmoid,
        allow_neg_eigval,
        d_k,
        d_v,
        tiles_per_head,
    )
    dtypes = (io_dtype, state_dtype, gate_dtype, a_log_dtype, None if bias_spec is None else bias_spec[0], beta_dtype)
    name = "gdn_warmup_forward_" + "_".join(str(int(flag)) for flag in flags)
    name += "_" + "_".join("none" if dtype is None else dtype.__name__.lower() for dtype in dtypes)
    name += f"_biasrank{0 if bias_spec is None else bias_spec[1]}"
    gate_align = 8 if gate_dtype.width == 16 else 4
    return compile_tvm_ffi(
        warmup_forward_host,
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
        not split,
        prefill_cfg,
        tiles_per_head,
        *(cutlass.Int32(0) for _ in range(3)),
        cutlass.Float32(0),
        cutlass.Float32(0),
        *(cutlass.Int32(0) for _ in range(4)),
        cutlass.Float32(0),
        tensor(io_dtype, 3, 16),
        tensor(io_dtype, 3, 16),
        tensor(io_dtype, 3, 16),
        tensor(gate_dtype, 2, 16),
        tensor(gate_dtype, 3 if gate_channels else 2, gate_align) if split else None,
        tensor(a_log_dtype, 1, 4) if a_log_dtype is not None else None,
        tensor(a_log_dtype, 1, 4) if a_log_dtype is not None else None,
        tensor(bias_spec[0], bias_spec[1], 4) if bias_spec is not None else None,
        tensor(bias_spec[0], bias_spec[1], 4) if bias_spec is not None else None,
        tensor(beta_dtype, 2, 16) if beta_dtype is not None else None,
        tensor(io_dtype, 3, 16),
        make_cu_seqlens_signature(sym_int(), assumed_align=4),
        make_cu_seqlens_signature(sym_int(), assumed_align=4),
        tensor(state_dtype, 4, 16) if use_initial_state else None,
        tensor(state_dtype, 4, 16) if store_final_state else None,
        make_counter_signature(sym_int()) if has_seed_indices else None,
        make_counter_signature(sym_int()) if has_final_indices else None,
        make_compact_signature_tensor(cutlass.Uint8, (sym_int(),), assumed_align=1) if has_initial_state else None,
        tensor(io_dtype, 4, 16) if enable_checkpoints else None,
        work_items(16),
        work_items(4),
        make_counter_signature(sym_int()),
        make_counter_signature(sym_int()),
        work_items(16) if split else None,
        work_items(4) if split else None,
        tensor(cutlass.Float32, 2, 4) if split else None,
        make_counter_signature(sym_int()),
        make_workspace_signature(sym_int()),
        name=name,
        opt_level=OPT_LEVEL,
    )


def build_warmup_forward(
    *,
    q,
    k,
    v,
    gate,
    beta,
    a_log,
    dt_bias,
    o,
    cu_seqlens,
    state_in,
    state_out,
    seed_indices,
    final_indices,
    checkpoints,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler,
    workspace,
    split,
    n_tiles,
    ideal_chunks,
    num_sm,
    b_t,
    log_gate,
    safe_gate,
    use_beta_sigmoid,
    allow_neg_eigval,
    expand_num,
    checkpoint_every_n_tokens,
    scale,
    tiles_per_head=1,
    has_initial_state=None,
):
    """Return ``(compiled, facts)`` for the warmup or uncut forward over the buffers of one plan: the launch is compiled
    (and persisted) once per static configuration, so every shape-dependent value is a launch argument."""
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
        gate_lower_bound=None,
        expand_num=expand_num,
    )
    state_src = state_in if state_in is not None else state_out
    state_dtype = get_dtype(state_src.dtype) if state_src is not None else cutlass.Float32
    compiled = _compile_warmup_forward(
        get_dtype(q.dtype),
        state_dtype,
        get_dtype(gate.dtype),
        get_dtype(a_log.dtype) if a_log is not None else None,
        (get_dtype(dt_bias.dtype), dt_bias.ndim) if dt_bias is not None else None,
        get_dtype(beta.dtype) if beta is not None else None,
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
        facts.num_sms,
        state_in is not None,
        state_out is not None,
        seed_indices is not None,
        final_indices is not None,
        has_initial_state is not None,
        int(checkpoint_every_n_tokens) > 0,
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        int(DK),
        int(DV),
        int(tiles_per_head),
    )
    return compiled, facts


def run_warmup_forward(
    compiled,
    facts,
    *,
    q,
    k,
    v,
    gate,
    beta,
    a_log,
    dt_bias,
    o,
    cu_seqlens,
    state_in,
    state_out,
    seed_indices,
    final_indices,
    checkpoints,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler,
    workspace,
    checkpoint_every_n_tokens,
    scale,
    has_initial_state=None,
) -> None:
    """Replay the warmup or uncut forward on the current stream: one crossing into the DSL for the table, prologue and
    prefill launches.  The plan validated the contract at build, so nothing here raises."""
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
        int(checkpoint_every_n_tokens),
        float(scale),
        q,
        k,
        v,
        gate,
        gate if facts.split else None,
        a_log if facts.safe_gate else None,
        a_log if facts.safe_gate else None,
        dt_bias if facts.safe_gate else None,
        dt_bias if facts.safe_gate else None,
        beta,
        o,
        cu_seqlens,
        cu_seqlens,
        state_in,
        state_out,
        seed_indices,
        final_indices,
        has_initial_state,
        checkpoints if int(checkpoint_every_n_tokens) > 0 else None,
        work_items,
        work_items,
        work_count,
        work_count,
        item_scratch if facts.split else None,
        item_scratch if facts.split else None,
        chunk_scratch if facts.split else None,
        scheduler,
        workspace,
    )
