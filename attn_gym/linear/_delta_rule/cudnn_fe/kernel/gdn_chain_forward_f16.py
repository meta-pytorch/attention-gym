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

"""One compiled launch for the GDN chain forward: chain prologue, T pass, fused summary, fp32 state chain and prefill issued
from a single host, the way ``split_k.run_table`` launches plan, scan and walk.  Every kernel, its host and the tensor
placeholder each host was compiled with are the standalone modules' own; this host only sequences the five launches, so the
kernels' SASS is unchanged and the Python side crosses into the DSL once per call instead of five times.  A buffer that two
hosts read through different placeholder types (the summary's ``state_out`` is the state chain's ``H``) is passed twice, once
per type."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from .._compat import DeviceView

from ..common.host import get_dtype
from ..common.piece_chain import dtype_name, launch_state_chain
from . import gdn_chain_prologue_f16, gdn_prefill_f16, gdn_summary_f16, gdn_tinv_f16

chain_forward_cache = {}


@cute.jit
def chain_forward_host(
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    tinv_cfg: cutlass.Constexpr,
    summary_cfg: cutlass.Constexpr,
    prefill_cfg: cutlass.Constexpr,
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    chain_rows: cutlass.Constexpr[int],
    has_seed: cutlass.Constexpr[bool],
    pieces: cutlass.Int32,
    heads_out: cutlass.Int32,
    num_seqs: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    scale: cutlass.Float32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    beta: cute.Tensor,
    a_log: Optional[cute.Tensor],
    dt_bias: Optional[cute.Tensor],
    o: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: cute.Tensor,
    scheduler_all: cute.Tensor,
    scheduler_summary: cute.Tensor,
    scheduler_prefill: cute.Tensor,
    tinv_words: cute.Tensor,
    tinv_rows: cute.Tensor,
    tinv_row_count: cute.Tensor,
    summary_words: cute.Tensor,
    prefill_words: cute.Tensor,
    tinv: cute.Tensor,
    state_h_summary: cute.Tensor,
    state_m_summary: cute.Tensor,
    state_h_chain: cute.Tensor,
    state_m_chain: cute.Tensor,
    state_x_chain: cute.Tensor,
    seed: Optional[cute.Tensor],
    seed_indices: Optional[cute.Tensor],
    state_x_prefill: cute.Tensor,
    final_state: Optional[cute.Tensor],
    final_indices: Optional[cute.Tensor],
    checkpoints: Optional[cute.Tensor],
    stream: cuda.CUstream,
) -> None:
    gdn_chain_prologue_f16.chain_prologue(
        pieces,
        unit_chunks,
        b_t,
        expand_num,
        length_rule,
        heads_out,
        False,
        1,
        cutlass.Int32(0),
        checkpoint_every_n,
        cu_seqlens,
        cu_pieces,
        main_rows,
        summary_rows,
        main_count,
        summary_count,
        work_items,
        work_items_summary,
        scheduler_all,
        None,
        None,
        tinv_words,
        tinv_rows,
        tinv_row_count,
        summary_words,
        None,
        None,
        None,
        prefill_words,
        None,
        None,
        q,
        k,
        v,
        o,
        None,
        checkpoints,
        None,
        None,
        None,
        None,
        None,
        tinv,
        stream,
    )
    gdn_tinv_f16.host(tinv_cfg, False, k, tinv_words, gate, a_log, dt_bias, beta, cu_pieces, tinv, tinv_rows, tinv_row_count, stream)
    gdn_summary_f16.host(
        summary_cfg,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        cu_pieces,
        tinv,
        None,
        state_h_summary,
        state_m_summary,
        work_items_summary,
        summary_count,
        scheduler_summary,
        summary_words,
        stream,
    )
    launch_state_chain(
        heads_out,
        dim_v,
        dim_k,
        chain_rows,
        pieces,
        False,
        has_seed,
        False,
        False,
        num_seqs,
        state_h_chain,
        state_m_chain,
        state_x_chain,
        seed,
        None,
        None,
        main_rows,
        seed_indices,
        stream,
    )
    gdn_prefill_f16.host(
        prefill_cfg,
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        None,
        o,
        cu_pieces,
        state_x_prefill,
        final_state,
        None,
        final_indices,
        tinv,
        work_items,
        main_count,
        scheduler_prefill,
        checkpoint_every_n,
        scale,
        prefill_words,
        stream,
    )


def build_chain_forward(
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
    cu_pieces,
    main_rows,
    summary_rows,
    main_count,
    summary_count,
    work_items,
    work_items_summary,
    scheduler_all,
    scheduler_summary,
    scheduler_prefill,
    tinv_words,
    tinv_rows,
    tinv_row_count,
    summary_words,
    prefill_words,
    tinv,
    state_h,
    state_m,
    state_x,
    seed,
    seed_indices,
    final_state,
    final_indices,
    checkpoints,
    pieces,
    heads_out,
    num_seqs,
    unit_chunks,
    b_t,
    expand_num,
    length_rule,
    log_gate,
    safe_gate,
    use_beta_sigmoid,
    allow_neg_eigval,
    checkpoint_every_n_tokens,
    scale,
    chain_rows,
    device,
    num_sm,
    stream,
):
    """Compile (cached per static config: dtypes, heads, dims, gate flags, checkpoint and final-state presence, seed dtype,
    chain rows, device) the chain forward launch over the buffers of one plan; ``pieces``, ``heads_out`` and ``num_seqs`` are
    launch arguments.  The placeholders repeat the marks of the standalone modules' builds so every kernel compiles as it
    does there."""
    _HQ, DK = q.shape[1], q.shape[2]
    k.shape[1]
    _HV, DV = v.shape[1], v.shape[2]
    HO = gate.shape[1]
    if not safe_gate:
        a_log = None
        dt_bias = None
    io_dtype = get_dtype(q.dtype)
    state_dtype = get_dtype(state_x.dtype)
    has_seed = seed is not None
    seed_name = dtype_name(seed.dtype) if has_seed else "float32"
    key = (
        str(q.dtype),
        str(state_x.dtype),
        str(final_state.dtype) if final_state is not None else "none",
        str(cu_seqlens.dtype),
        str(gate.dtype),
        str(a_log.dtype) if a_log is not None else "none",
        str(dt_bias.dtype) if dt_bias is not None else "none",
        str(beta.dtype),
        seed_name,
        int(device),
        int(num_sm),
        DK,
        DV,
        int(unit_chunks),
        int(b_t),
        int(expand_num),
        bool(length_rule),
        bool(log_gate),
        bool(safe_gate),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        int(checkpoint_every_n_tokens) > 0,
        final_state is not None,
        has_seed,
        seed_indices is not None,
        final_indices is not None,
        int(chain_rows),
    )
    if key not in chain_forward_cache:
        tinv_cfg = gdn_tinv_f16.build_cfg(
            io_dtype,
            num_sm=num_sm,
            log_gate=log_gate,
            safe_gate=safe_gate,
            beta_sigmoid=use_beta_sigmoid,
            allow_neg_eigval=allow_neg_eigval,
            d_k=DK,
            expand_num=expand_num,
        )
        summary_cfg = gdn_summary_f16.build_cfg(
            io_dtype,
            get_dtype(state_h.dtype),
            max_active_clusters=num_sm,
            use_initial_state=False,
            log_gate=log_gate,
            safe_gate=safe_gate,
            d_k=DK,
            d_v=DV,
            expand_num=expand_num,
        )
        prefill_cfg = gdn_prefill_f16.build_cfg(
            io_dtype,
            state_dtype,
            max_active_clusters=num_sm,
            use_initial_state=True,
            store_final_state=final_state is not None,
            enable_checkpoints=int(checkpoint_every_n_tokens) > 0,
            log_gate=log_gate,
            safe_gate=safe_gate,
            tinv_source="gmem",
            d_k=DK,
            d_v=DV,
            expand_num=expand_num,
        )

        work_items_placeholder = from_dlpack(work_items, assumed_align=16)
        work_items_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        work_items_summary_placeholder = from_dlpack(work_items_summary, assumed_align=16)
        work_items_summary_placeholder.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        state_h_placeholder = from_dlpack(state_h, assumed_align=16)
        state_h_placeholder.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)
        state_m_placeholder = from_dlpack(state_m, assumed_align=16)
        state_m_placeholder.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)
        state_x_placeholder = from_dlpack(state_x, assumed_align=16)
        state_x_placeholder.mark_layout_dynamic().mark_compact_shape_dynamic(mode=3, stride_order=(0, 1, 2, 3), divisibility=DK)
        final_state_placeholder = None
        if final_state is not None:
            final_state_placeholder = from_dlpack(final_state, assumed_align=16).mark_layout_dynamic(leading_dim=3)
        chain_forward_cache[key] = cute.compile(
            chain_forward_host,
            int(unit_chunks),
            int(b_t),
            int(expand_num),
            bool(length_rule),
            tinv_cfg,
            summary_cfg,
            prefill_cfg,
            DV,
            DK,
            int(chain_rows),
            has_seed,
            cutlass.Int32(int(pieces)),
            cutlass.Int32(int(heads_out)),
            cutlass.Int32(int(num_seqs)),
            cutlass.Int32(int(checkpoint_every_n_tokens)),
            float(scale),
            from_dlpack(q, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(k, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(v, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(gate, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(beta, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(a_log, assumed_align=4).mark_layout_dynamic() if a_log is not None else None,
            from_dlpack(dt_bias, assumed_align=4).mark_layout_dynamic(leading_dim=len(dt_bias.shape) - 1) if dt_bias is not None else None,
            from_dlpack(o, assumed_align=16).mark_layout_dynamic(leading_dim=2),
            from_dlpack(cu_seqlens, assumed_align=8 if str(cu_seqlens.dtype).endswith("int64") else 4).mark_layout_dynamic(),
            from_dlpack(cu_pieces, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(main_rows, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(summary_rows, assumed_align=16).mark_layout_dynamic(),
            from_dlpack(main_count, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(summary_count, assumed_align=4).mark_layout_dynamic(),
            work_items_placeholder,
            work_items_summary_placeholder,
            from_dlpack(scheduler_all, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_summary, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(scheduler_prefill, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(tinv_words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(tinv_rows, assumed_align=16).mark_layout_dynamic(leading_dim=1),
            from_dlpack(tinv_row_count, assumed_align=4).mark_layout_dynamic(),
            from_dlpack(summary_words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(prefill_words, assumed_align=128).mark_layout_dynamic(),
            from_dlpack(tinv, assumed_align=128).mark_layout_dynamic(leading_dim=3),
            state_h_placeholder,
            state_m_placeholder,
            from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(DeviceView(256, (1, HO, DK, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(DeviceView(256, (1, HO, DV, DK), "float32", int(device)), assumed_align=16).mark_layout_dynamic(leading_dim=3),
            from_dlpack(DeviceView(256, (1, HO, DV, DK), seed_name, int(device)), assumed_align=4).mark_layout_dynamic(leading_dim=3) if has_seed else None,
            from_dlpack(seed_indices, assumed_align=4).mark_layout_dynamic() if seed_indices is not None else None,
            state_x_placeholder,
            final_state_placeholder,
            from_dlpack(final_indices, assumed_align=4).mark_layout_dynamic() if final_indices is not None else None,
            from_dlpack(checkpoints, assumed_align=16).mark_layout_dynamic(leading_dim=3) if checkpoints is not None else None,
            cuda.CUstream(int(stream)),
            options="--enable-tvm-ffi --opt-level 2",
        )
    return chain_forward_cache[key]


def run_chain_forward(
    compiled,
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
    cu_pieces,
    main_rows,
    summary_rows,
    main_count,
    summary_count,
    work_items,
    work_items_summary,
    scheduler_all,
    scheduler_summary,
    scheduler_prefill,
    tinv_words,
    tinv_rows,
    tinv_row_count,
    summary_words,
    prefill_words,
    tinv,
    state_h,
    state_m,
    state_x,
    seed,
    seed_indices,
    final_state,
    final_indices,
    checkpoints,
    pieces,
    heads_out,
    num_seqs,
    checkpoint_every_n_tokens,
    scale,
    stream,
) -> None:
    """Replay the chain forward: one crossing into the DSL for the five launches.  The plan validated the contract at build,
    so nothing here raises."""
    compiled(
        int(pieces),
        int(heads_out),
        int(num_seqs),
        int(checkpoint_every_n_tokens),
        float(scale),
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        dt_bias,
        o,
        cu_seqlens,
        cu_pieces,
        main_rows,
        summary_rows,
        main_count,
        summary_count,
        work_items,
        work_items_summary,
        scheduler_all,
        scheduler_summary,
        scheduler_prefill,
        tinv_words,
        tinv_rows,
        tinv_row_count,
        summary_words,
        prefill_words,
        tinv,
        state_h,
        state_m,
        state_h,
        state_m,
        state_x,
        seed,
        seed_indices,
        state_x,
        final_state,
        final_indices,
        checkpoints,
        cuda.CUstream(int(stream)),
    )
