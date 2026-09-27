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

"""One compiled launch for the KDA warmup and uncut forwards: the split-K table (plan, scan and walk, warmup only), the
prefill prologue and the prefill issued from a single host.  Every kernel, its host and the tensor placeholder each host
was compiled with are the standalone modules' own; this host only sequences the launches, so the kernels' SASS is
unchanged and the Python side crosses into the DSL once per call instead of two or three times.  A buffer that two hosts
read through different placeholder types is passed twice, once per type: the table marks the gate at its element
alignment along its last mode, a_log fully dynamic, dt_bias, work_items, work_count and item_scratch as 4-byte compact
views, cu_seqlens at 4 bytes; the prologue and prefill mark the same buffers as the standalone prefill wrapper does."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from .._persist import persistent_compile
from ..common import split_k
from ..common.host import get_dtype
from . import kda_prefill_f16, kda_prep_f16, kda_prep_prefill_f16

warmup_forward_cache = {}


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
    beta: cute.Tensor,
    o: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_seqlens_table: cute.Tensor,
    state_in: Optional[cute.Tensor],
    state_out: Optional[cute.Tensor],
    seed_indices: Optional[cute.Tensor],
    final_indices: Optional[cute.Tensor],
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
    prep: cutlass.Constexpr[bool],
    prep_cfg: cutlass.Constexpr,
    prep_k_decay: Optional[cute.Tensor],
    prep_q_decay: Optional[cute.Tensor],
    prep_t: Optional[cute.Tensor],
    prep_a: Optional[cute.Tensor],
    prep_diag: Optional[cute.Tensor],
    prep_words: Optional[cute.Tensor],
    prep_rows: Optional[cute.Tensor],
    prep_row_count: Optional[cute.Tensor],
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
    if cutlass.const_expr(prep):
        kda_prep_prefill_f16.prologue(
            io_dtype,
            b_t,
            num_sms,
            order_gen,
            q,
            k,
            v,
            gate,
            o,
            checkpoints,
            cu_seqlens,
            staging,
            work_count,
            work_items,
            scheduler,
            workspace,
            checkpoint_every_n,
            stream,
            tiles_per_head,
            prep_k_decay,
            prep_q_decay,
            prep_t,
            prep_a,
            prep_diag,
            prep_cfg,
            prep_words,
            prep_rows,
            prep_row_count,
        )
        kda_prep_f16.host(
            prep_cfg,
            q,
            k,
            prep_words,
            gate,
            a_log,
            dt_bias,
            beta,
            cu_seqlens,
            prep_k_decay,
            prep_q_decay,
            prep_t,
            prep_a,
            prep_diag,
            prep_rows,
            prep_row_count,
            stream,
        )
    else:
        kda_prefill_f16.prologue(
            io_dtype,
            b_t,
            num_sms,
            order_gen,
            q,
            k,
            v,
            gate,
            o,
            checkpoints,
            cu_seqlens,
            staging,
            work_count,
            work_items,
            scheduler,
            workspace,
            checkpoint_every_n,
            stream,
            tiles_per_head,
        )
    prefill_host = kda_prep_prefill_f16.host if cutlass.const_expr(prep) else kda_prefill_f16.host
    prefill_host(
        prefill_cfg,
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        cu_seqlens,
        state_in,
        o,
        state_out,
        seed_indices,
        final_indices,
        work_items,
        work_count,
        scheduler,
        workspace,
        checkpoint_every_n,
        scale,
        stream,
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
    gate_lower_bound,
    use_qk_l2norm,
    use_beta_sigmoid,
    allow_neg_eigval,
    checkpoint_every_n_tokens,
    scale,
    device,
    stream,
    tiles_per_head=1,
    prep=False,
    prep_k_decay=None,
    prep_q_decay=None,
    prep_t=None,
    prep_a=None,
    prep_diag=None,
    prep_words=None,
    prep_rows=None,
    prep_row_count=None,
):
    """Compile (cached per static config: dtypes, heads, dims, gate flags and bound, the split-K geometry, the d_v split, state and
    checkpoint presence, device) the warmup or uncut forward launch over the buffers of one plan.  The placeholders repeat
    the marks of the standalone split-table and prefill builds so every kernel compiles as it does there."""
    HQ, DK = q.shape[1], q.shape[2]
    k.shape[1]
    HV, DV = v.shape[1], v.shape[2]
    max(HQ, HV)
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
    state_src = state_in if state_in is not None else state_out
    state_dtype = get_dtype(state_src.dtype) if state_src is not None else cutlass.Float32
    key = (
        str(q.dtype),
        str(gate.dtype),
        str(beta.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(cu_seqlens.dtype),
        str(state_src.dtype) if state_src is not None else "none",
        int(device),
        int(num_sm),
        DK,
        DV,
        int(b_t),
        bool(split),
        facts.scan_rows,
        bool(log_gate),
        bool(safe_gate),
        float(gate_lower_bound),
        bool(use_qk_l2norm),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        state_in is not None,
        state_out is not None,
        seed_indices is not None,
        final_indices is not None,
        int(checkpoint_every_n_tokens) > 0,
        int(tiles_per_head),
        bool(prep),
    )
    if key not in warmup_forward_cache:
        prefill_module = kda_prep_prefill_f16 if prep else kda_prefill_f16
        prefill_cfg = prefill_module.build_cfg(
            io_dtype,
            state_dtype,
            gate_dtype,
            use_initial_state=state_in is not None,
            store_final_state=state_out is not None,
            enable_checkpoints=int(checkpoint_every_n_tokens) > 0,
            l2norm=use_qk_l2norm,
            safe_gate=safe_gate,
            gate_scale_log2=float(gate_lower_bound) * kda_prefill_f16.LOG2_E,
            log_gate=log_gate,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            max_active_clusters=num_sm,
            d_k=DK,
            d_v=DV // tiles_per_head,
            tiles_per_head=tiles_per_head,
        )
        prep_cfg = None
        if prep:
            prep_cfg = kda_prep_f16.build_cfg(
                io_dtype,
                gate_dtype,
                num_sm=num_sm,
                l2norm=use_qk_l2norm,
                safe_gate=safe_gate,
                gate_scale_log2=float(gate_lower_bound) * kda_prefill_f16.LOG2_E,
                log_gate=log_gate,
                beta_sigmoid=use_beta_sigmoid,
                allow_neg_eigval=allow_neg_eigval,
                d_k=DK,
            )
        prep_placeholders = [None] * 8
        if prep:
            prep_placeholders = [from_dlpack(rec, assumed_align=128).mark_layout_dynamic(leading_dim=3) for rec in (prep_k_decay, prep_q_decay, prep_t)]
            prep_placeholders.append(from_dlpack(prep_a, assumed_align=16).mark_layout_dynamic(leading_dim=2))
            prep_placeholders.append(from_dlpack(prep_diag, assumed_align=16).mark_layout_dynamic(leading_dim=2))
            prep_placeholders.append(from_dlpack(prep_words, assumed_align=128).mark_layout_dynamic())
            prep_placeholders.append(from_dlpack(prep_rows, assumed_align=16).mark_layout_dynamic(leading_dim=1))
            prep_placeholders.append(from_dlpack(prep_row_count, assumed_align=4).mark_layout_dynamic())

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
        warmup_forward_cache[key] = persistent_compile(
            "kda_warmup_forward",
            key,
            cute.compile,
            warmup_forward_host,
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
            not split,
            prefill_cfg,
            int(tiles_per_head),
            cutlass.Int32(facts.n_tiles),
            cutlass.Int32(facts.ideal_chunks),
            cutlass.Int32(facts.batch_size),
            cutlass.Float32(facts.log2_threshold),
            cutlass.Float32(facts.gate_scale_log2),
            cutlass.Int32(facts.n_scan_ctas),
            cutlass.Int32(facts.n_scan_blocks),
            cutlass.Int32(facts.n_walk_ctas),
            cutlass.Int32(int(checkpoint_every_n_tokens)),
            float(scale),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            gate_table_placeholder,
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=16).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            dt_bias_table_placeholder,
            from_dlpack(beta, assumed_align=4).mark_layout_dynamic(leading_dim=1),
            from_dlpack(o, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(state_in, assumed_align=16).mark_layout_dynamic(leading_dim=3) if state_in is not None else None,
            from_dlpack(state_out, assumed_align=16).mark_layout_dynamic(leading_dim=3) if state_out is not None else None,
            from_dlpack(seed_indices, assumed_align=4).mark_layout_dynamic() if seed_indices is not None else None,
            from_dlpack(final_indices, assumed_align=4).mark_layout_dynamic() if final_indices is not None else None,
            from_dlpack(checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3) if int(checkpoint_every_n_tokens) > 0 else None,
            work_items_placeholder,
            work_items_table_placeholder,
            from_dlpack(work_count, assumed_align=4).mark_layout_dynamic(),
            work_count_table_placeholder,
            staging_placeholder,
            item_scratch_placeholder,
            chunk_scratch_placeholder,
            from_dlpack(scheduler, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(workspace, assumed_align=128).mark_layout_dynamic(),
            bool(prep),
            prep_cfg,
            *prep_placeholders,
            cuda.CUstream(int(stream)),
            options="--enable-tvm-ffi --opt-level 2",
        )
    return warmup_forward_cache[key], facts


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
    stream,
    prep_k_decay=None,
    prep_q_decay=None,
    prep_t=None,
    prep_a=None,
    prep_diag=None,
    prep_words=None,
    prep_rows=None,
    prep_row_count=None,
) -> None:
    """Replay the warmup or uncut forward: one crossing into the DSL for the table, prologue and prefill launches.  The
    plan validated the contract at build, so nothing here raises."""
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
        prep_k_decay,
        prep_q_decay,
        prep_t,
        prep_a,
        prep_diag,
        prep_words,
        prep_rows,
        prep_row_count,
        cuda.CUstream(int(stream)),
    )
