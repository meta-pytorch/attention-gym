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
# attn_gym.linear._delta_rule.cudnn_fe. The host nests the summary / prefill Ops, compiles through a
# persisted jit_cache function over fake TVM-FFI signatures (int64 ABI variant when a tensor needs it),
# and launches with live tensors on the current Torch stream.

"""One compiled launch for the KDA chain forward: chain prologue, fused summary, fp32 state chain and prefill issued from a
single host, the way ``split_k.run_table`` launches plan, scan and walk.  Every kernel, its host and the tensor signature
each host was compiled with are the standalone modules' own; this host only sequences the four launches, so the kernels'
SASS is unchanged and the Python side crosses into the DSL once per call instead of four times.  A buffer that two hosts read
through different signature types is passed twice, once per type: H, M and X (the summary's and prefill's torch views, the
state chain's ``(1, HO, V, K)`` device views) and ``cu_pieces`` (the prologue marks it at its element alignment, the summary
and prefill at 8 bytes).  Compiled at ``--opt-level 2``, the level of every KDA module (the chain prologue and the state chain
take it standalone too, through ``opt_level``), so every nested kernel is the standalone one."""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from attn_gym._backends.cute import compile_tvm_ffi, jit_cache
from attn_gym._backends.cute.utils import requires_int64_abi

from ..common.host import get_dtype
from ..common.piece_chain import launch_state_chain
from ..common.tvm_ffi import WORK_ITEM_FIELDS, make_compact_signature_tensor, make_dynamic_signature_tensor
from . import kda_chain_prologue_f16, kda_prefill_f16, kda_summary_f16


@cute.jit
def chain_forward_host(
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    summary_op: cutlass.Constexpr,
    prefill_op: cutlass.Constexpr,
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
    cu_pieces_main: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: cute.Tensor,
    scheduler_all: cute.Tensor,
    scheduler_summary: cute.Tensor,
    scheduler_prefill: cute.Tensor,
    summary_words: cute.Tensor,
    prefill_words: cute.Tensor,
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
    kda_chain_prologue_f16.chain_prologue(
        pieces,
        unit_chunks,
        b_t,
        length_rule,
        heads_out,
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
        gate,
        o,
        None,
        checkpoints,
        None,
        None,
        None,
        None,
        stream,
    )
    summary_op(
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        cu_pieces_main,
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
    prefill_op(
        q,
        k,
        v,
        gate,
        a_log,
        dt_bias,
        beta,
        cu_pieces_main,
        state_x_prefill,
        o,
        final_state,
        None,
        final_indices,
        work_items,
        main_count,
        scheduler_prefill,
        prefill_words,
        checkpoint_every_n,
        scale,
        stream,
    )


def _dynamic(dtype, rank: int, align: int, use_int64_offsets: bool):
    """The legacy ``from_dlpack(t, assumed_align=align).mark_layout_dynamic()`` placeholder (last mode contiguous)."""
    return make_dynamic_signature_tensor(dtype, rank, assumed_align=align, use_int64_offsets=use_int64_offsets)


def _work_table(use_int64_offsets: bool):
    """The legacy work-item placeholder: compact ``[rows, WORK_ITEM_FIELDS]`` int32, only the row count dynamic."""
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    return make_compact_signature_tensor(cutlass.Int32, (sym_int(), WORK_ITEM_FIELDS), assumed_align=16)


@jit_cache
def _compile_chain_forward(
    unit_chunks: int,
    b_t: int,
    length_rule: bool,
    io_dtype,
    gate_dtype,
    beta_dtype,
    a_log_dtype,
    dt_bias_spec,
    cu_seqlens_dtype,
    state_dtype,
    final_state_dtype,
    checkpoint_dtype,
    seed_dtype,
    has_seed_indices: bool,
    has_final_indices: bool,
    d_k: int,
    d_v: int,
    chain_rows: int,
    log_gate: bool,
    safe_gate: bool,
    gate_scale_log2: float,
    use_qk_l2norm: bool,
    use_beta_sigmoid: bool,
    allow_neg_eigval: bool,
    num_sm: int,
    use_int64_offsets: bool,
):
    """Compile the chain forward host for one static config over fake tensors that repeat the standalone builds'
    placeholders.  Absent tensors have a None dtype; ``dt_bias_spec`` is ``(dtype, rank)`` or None."""
    i64 = use_int64_offsets
    flags = dict(
        l2norm=use_qk_l2norm,
        safe_gate=safe_gate,
        gate_scale_log2=gate_scale_log2,
        log_gate=log_gate,
        beta_sigmoid=use_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        max_active_clusters=num_sm,
        d_k=d_k,
        d_v=d_v,
    )
    summary_cfg = kda_summary_f16.build_cfg(io_dtype, gate_dtype, use_initial_state=False, **flags)
    prefill_cfg = kda_prefill_f16.build_cfg(
        io_dtype,
        state_dtype,
        gate_dtype,
        use_initial_state=True,
        store_final_state=final_state_dtype is not None,
        enable_checkpoints=checkpoint_dtype is not None,
        **flags,
    )
    static = (
        unit_chunks,
        b_t,
        length_rule,
        final_state_dtype is not None,
        checkpoint_dtype is not None,
        seed_dtype is not None,
        has_seed_indices,
        has_final_indices,
        d_k,
        d_v,
        chain_rows,
        log_gate,
        safe_gate,
        use_qk_l2norm,
        use_beta_sigmoid,
        allow_neg_eigval,
        num_sm,
        i64,
    )
    dtype_names = "_".join(
        "none" if dtype is None else dtype.__name__.lower()
        for dtype in (
            io_dtype,
            gate_dtype,
            beta_dtype,
            a_log_dtype,
            None if dt_bias_spec is None else dt_bias_spec[0],
            cu_seqlens_dtype,
            state_dtype,
            final_state_dtype,
            checkpoint_dtype,
            seed_dtype,
        )
    )
    dt_bias_rank = 0 if dt_bias_spec is None else dt_bias_spec[1]
    gate_tag = str(float(gate_scale_log2)).replace(".", "p").replace("-", "m").replace("+", "")
    name = (
        "kda_chain_forward_" + "_".join(str(int(flag)) for flag in static)
        + f"_{dtype_names}_biasrank{dt_bias_rank}_g{gate_tag}"
    )
    f32_state = lambda: _dynamic(cutlass.Float32, 4, 16, i64)  # noqa: E731
    counter = lambda: _dynamic(cutlass.Int32, 1, 4, i64)  # noqa: E731
    return compile_tvm_ffi(
        chain_forward_host,
        unit_chunks,
        b_t,
        length_rule,
        kda_summary_f16.KdaSummaryOp(summary_cfg, i64),
        kda_prefill_f16.KdaPrefillOp(prefill_cfg, i64),
        d_v,
        d_k,
        chain_rows,
        seed_dtype is not None,
        *(cutlass.Int32(0) for _ in range(4)),  # pieces, heads_out, num_seqs, checkpoint_every_n
        cutlass.Float32(0),  # scale
        _dynamic(io_dtype, 3, 16, i64),  # q
        _dynamic(io_dtype, 3, 16, i64),  # k
        _dynamic(io_dtype, 3, 16, i64),  # v
        _dynamic(gate_dtype, 3, 16, i64),  # gate
        _dynamic(beta_dtype, 2, 4, i64),  # beta
        _dynamic(a_log_dtype, 1, 4, i64) if a_log_dtype is not None else None,
        _dynamic(dt_bias_spec[0], dt_bias_spec[1], 16, i64) if dt_bias_spec is not None else None,
        _dynamic(io_dtype, 3, 16, i64),  # o
        _dynamic(cu_seqlens_dtype, 1, 8 if cu_seqlens_dtype is cutlass.Int64 else 4, i64),  # cu_seqlens
        _dynamic(cutlass.Int32, 1, 4, i64),  # cu_pieces (chain prologue)
        _dynamic(cutlass.Int32, 1, 8, i64),  # cu_pieces_main (summary, prefill)
        _dynamic(cutlass.Int32, 1, 16, i64),  # main_rows
        _dynamic(cutlass.Int32, 1, 16, i64),  # summary_rows
        counter(),  # main_count
        counter(),  # summary_count
        _work_table(i64),  # work_items
        _work_table(i64),  # work_items_summary
        counter(),  # scheduler_all
        counter(),  # scheduler_summary
        counter(),  # scheduler_prefill
        _dynamic(cutlass.Int64, 1, 128, i64),  # summary_words
        _dynamic(cutlass.Int64, 1, 128, i64),  # prefill_words
        f32_state(),  # state_h_summary
        f32_state(),  # state_m_summary
        f32_state(),  # state_h_chain
        f32_state(),  # state_m_chain
        f32_state(),  # state_x_chain
        _dynamic(seed_dtype, 4, 4, i64) if seed_dtype is not None else None,  # seed
        counter() if has_seed_indices else None,  # seed_indices
        _dynamic(state_dtype, 4, 16, i64),  # state_x_prefill
        _dynamic(final_state_dtype, 4, 16, i64) if final_state_dtype is not None else None,
        counter() if has_final_indices else None,  # final_indices
        _dynamic(checkpoint_dtype, 4, 16, i64) if checkpoint_dtype is not None else None,
        name=name,
        opt_level=2,
    )


def _dtype_or_none(tensor):
    return None if tensor is None else get_dtype(tensor.dtype)


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
    summary_words,
    prefill_words,
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
    length_rule,
    log_gate,
    safe_gate,
    gate_lower_bound,
    use_qk_l2norm,
    use_beta_sigmoid,
    allow_neg_eigval,
    checkpoint_every_n_tokens,
    scale,
    chain_rows,
    device,
    num_sm,
    stream,
):
    """Compile (persisted per static config: dtypes, dims, gate flags and bound, checkpoint and final-state presence,
    seed dtype, chain rows, SM count, the int64 ABI) the chain forward launch over the buffers of one plan; ``pieces``,
    ``heads_out`` and ``num_seqs`` are launch arguments.  The fake signatures repeat the marks of the standalone modules'
    builds so every kernel compiles as it does there.  ``device`` and ``stream`` are unused: the launch runs on the
    current Torch stream."""
    del pieces, heads_out, num_seqs, scale, device, stream
    if not safe_gate:
        a_log = None
        dt_bias = None
    if int(checkpoint_every_n_tokens) <= 0:
        checkpoints = None
    for name, tensor in (("state_h", state_h), ("state_m", state_m), ("state_x", state_x)):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {tensor.dtype}")
    for name, tensor in (("seed_indices", seed_indices), ("final_indices", final_indices)):
        if tensor is not None and tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be int32, got {tensor.dtype}")
    tensors = (
        q, k, v, gate, beta, a_log, dt_bias, o, cu_seqlens, cu_pieces, main_rows, summary_rows,
        main_count, summary_count, work_items, work_items_summary, scheduler_all, scheduler_summary,
        scheduler_prefill, summary_words, prefill_words, state_h, state_m, state_x, seed,
        seed_indices, final_state, final_indices, checkpoints,
    )
    return _compile_chain_forward(
        int(unit_chunks),
        int(b_t),
        bool(length_rule),
        get_dtype(q.dtype),
        get_dtype(gate.dtype),
        get_dtype(beta.dtype),
        _dtype_or_none(a_log),
        (get_dtype(dt_bias.dtype), dt_bias.ndim) if dt_bias is not None else None,
        cutlass.Int64 if cu_seqlens.dtype == torch.int64 else cutlass.Int32,
        get_dtype(state_x.dtype),
        _dtype_or_none(final_state),
        _dtype_or_none(checkpoints),
        _dtype_or_none(seed),
        seed_indices is not None,
        final_indices is not None,
        int(q.shape[2]),
        int(v.shape[2]),
        int(chain_rows),
        bool(log_gate),
        bool(safe_gate),
        float(gate_lower_bound) * kda_summary_f16.LOG2_E,
        bool(use_qk_l2norm),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        int(num_sm),
        requires_int64_abi(*tensors),
    )


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
    summary_words,
    prefill_words,
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
    """Replay the chain forward: one crossing into the DSL for the four launches, on the current Torch stream
    (``stream`` is unused).  The plan validated the contract at build, so nothing here raises."""
    del stream
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
        summary_words,
        prefill_words,
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
        checkpoints if int(checkpoint_every_n_tokens) > 0 else None,
    )
