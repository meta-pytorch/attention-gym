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

"""One compiled launch for the GDN chain forward: chain prologue, T pass, fused summary, fp32 state
chain and prefill issued from a single host, the way ``split_k.run_table`` launches plan, scan and
walk.  Every kernel, its host and the tensor placeholder each host was compiled with are the
standalone modules' own; this host only sequences the five launches, so the kernels' SASS is
unchanged and the Python side crosses into the DSL once per call instead of five times.  A buffer
that two hosts read through different signature types (the summary's ``state_out`` is the state
chain's ``H``) is passed twice, once per type."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor

from attn_gym._backends.cute import compile_tvm_ffi, jit_cache
from attn_gym._backends.cute.utils import requires_int64_abi

from ..common.host import get_dtype
from ..common.piece_chain import launch_state_chain
from ..common.tvm_ffi import (
    WORK_ITEM_FIELDS,
    make_compact_signature_tensor,
    make_counter_signature,
    make_cu_seqlens_signature,
    make_strided_signature_tensor,
    make_workspace_signature,
)
from . import gdn_chain_prologue_f16, gdn_prefill_f16, gdn_summary_f16, gdn_tinv_f16

OPT_LEVEL = 2


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
    a_log: cute.Tensor | None,
    dt_bias: cute.Tensor | None,
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
    seed: cute.Tensor | None,
    seed_indices: cute.Tensor | None,
    state_x_prefill: cute.Tensor,
    final_state: cute.Tensor | None,
    final_indices: cute.Tensor | None,
    checkpoints: cute.Tensor | None,
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
    gdn_tinv_f16.host(
        tinv_cfg,
        False,
        k,
        tinv_words,
        gate,
        a_log,
        dt_bias,
        beta,
        cu_pieces,
        tinv,
        tinv_rows,
        tinv_row_count,
        stream,
    )
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
        None,
        tinv,
        work_items,
        main_count,
        scheduler_prefill,
        checkpoint_every_n,
        scale,
        prefill_words,
        stream,
    )


@jit_cache
def _compile_chain_forward(
    io_dtype,
    state_dtype,
    final_dtype,
    gate_dtype,
    a_log_dtype,
    bias_spec,
    beta_dtype,
    seed_dtype,
    num_sm,
    d_k,
    d_v,
    unit_chunks,
    b_t,
    expand_num,
    length_rule,
    log_gate,
    safe_gate,
    use_beta_sigmoid,
    allow_neg_eigval,
    enable_checkpoints,
    has_seed_indices,
    has_final_indices,
    chain_rows,
    use_int64_offsets,
):
    """Compile the chain forward launch for one static configuration (dtypes, heads-independent
    gate and state flags, dims, chain rows, device target); every extent is symbolic.
    ``use_int64_offsets`` widens the extents for tensors past the int32 ABI (strides are always
    int64, as upstream traced them)."""
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int

    def tensor(dtype, rank, align):
        return make_strided_signature_tensor(
            dtype,
            tuple(sym_int() for _ in range(rank)),
            assumed_align=align,
            use_int64_offsets=True,
            stride_divisibility=1,
        )

    def work_items():
        return make_compact_signature_tensor(
            cutlass.Int32, (sym_int(), WORK_ITEM_FIELDS), assumed_align=16
        )

    def rows():
        return make_compact_signature_tensor(cutlass.Int32, (sym_int(),), assumed_align=16)

    def summary_state():
        # The summary's outputs and the prefill's seed read the compact [P, HO, D, d_k] chain
        # states whose inner extent is a multiple of d_k (16-byte vector accesses).
        return make_fake_compact_tensor(
            state_dtype,
            (sym_int(), sym_int(), sym_int(), sym_int(divisibility=d_k)),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )

    tinv_cfg = gdn_tinv_f16.build_cfg(
        io_dtype,
        num_sm=num_sm,
        log_gate=log_gate,
        safe_gate=safe_gate,
        beta_sigmoid=use_beta_sigmoid,
        allow_neg_eigval=allow_neg_eigval,
        d_k=d_k,
        expand_num=expand_num,
    )
    summary_cfg = gdn_summary_f16.build_cfg(
        io_dtype,
        state_dtype,
        max_active_clusters=num_sm,
        use_initial_state=False,
        log_gate=log_gate,
        safe_gate=safe_gate,
        d_k=d_k,
        d_v=d_v,
        expand_num=expand_num,
    )
    prefill_cfg = gdn_prefill_f16.build_cfg(
        io_dtype,
        state_dtype,
        max_active_clusters=num_sm,
        use_initial_state=True,
        store_final_state=final_dtype is not None,
        enable_checkpoints=enable_checkpoints,
        log_gate=log_gate,
        safe_gate=safe_gate,
        tinv_source="gmem",
        d_k=d_k,
        d_v=d_v,
        expand_num=expand_num,
    )
    has_seed = seed_dtype is not None
    flags = (
        num_sm,
        d_k,
        d_v,
        unit_chunks,
        b_t,
        expand_num,
        length_rule,
        log_gate,
        safe_gate,
        use_beta_sigmoid,
        allow_neg_eigval,
        enable_checkpoints,
        has_seed_indices,
        has_final_indices,
        chain_rows,
        use_int64_offsets,
    )
    dtypes = (
        io_dtype,
        state_dtype,
        final_dtype,
        gate_dtype,
        a_log_dtype,
        None if bias_spec is None else bias_spec[0],
        beta_dtype,
        seed_dtype,
    )
    name = "gdn_chain_forward_" + "_".join(str(int(flag)) for flag in flags)
    name += "_" + "_".join("none" if dtype is None else dtype.__name__.lower() for dtype in dtypes)
    name += f"_biasrank{0 if bias_spec is None else bias_spec[1]}"
    return compile_tvm_ffi(
        chain_forward_host,
        unit_chunks,
        b_t,
        expand_num,
        length_rule,
        tinv_cfg,
        summary_cfg,
        prefill_cfg,
        d_v,
        d_k,
        chain_rows,
        has_seed,
        *(cutlass.Int32(0) for _ in range(4)),
        cutlass.Float32(0),
        tensor(io_dtype, 3, 16),
        tensor(io_dtype, 3, 16),
        tensor(io_dtype, 3, 16),
        tensor(gate_dtype, 2, 16),
        tensor(beta_dtype, 2, 16),
        tensor(a_log_dtype, 1, 4) if a_log_dtype is not None else None,
        tensor(bias_spec[0], bias_spec[1], 4) if bias_spec is not None else None,
        tensor(io_dtype, 3, 16),
        make_cu_seqlens_signature(sym_int(), assumed_align=4),
        make_counter_signature(sym_int()),
        rows(),
        rows(),
        make_counter_signature(sym_int()),
        make_counter_signature(sym_int()),
        work_items(),
        work_items(),
        make_counter_signature(sym_int()),
        make_counter_signature(sym_int()),
        make_counter_signature(sym_int()),
        make_workspace_signature(sym_int()),
        tensor(cutlass.Int32, 2, 16),
        make_counter_signature(sym_int()),
        make_workspace_signature(sym_int()),
        make_workspace_signature(sym_int()),
        tensor(io_dtype, 4, 128),
        summary_state(),
        summary_state(),
        *(tensor(state_dtype, 4, 16) for _ in range(3)),
        tensor(seed_dtype, 4, 4) if has_seed else None,
        make_counter_signature(sym_int()) if has_seed_indices else None,
        summary_state(),
        tensor(final_dtype, 4, 16) if final_dtype is not None else None,
        make_counter_signature(sym_int()) if has_final_indices else None,
        tensor(io_dtype, 4, 16) if enable_checkpoints else None,
        name=name,
        opt_level=OPT_LEVEL,
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
    num_sm,
):
    """Return the compiled chain forward launch over the buffers of one plan: compiled (and
    persisted) once per static configuration (dtypes, dims, gate flags, checkpoint, seed and
    final-state presence, chain rows, device); ``pieces``, ``heads_out`` and ``num_seqs`` are
    launch arguments."""
    DK = q.shape[2]
    DV = v.shape[2]
    if not safe_gate:
        a_log = None
        dt_bias = None
    if state_h.dtype != state_x.dtype or state_h.dtype != state_m.dtype:
        raise TypeError("the chain states must share one dtype")
    # Every tensor the launch addresses; bounded int32 tables and counters excepted.
    use_int64_offsets = requires_int64_abi(
        q,
        k,
        v,
        gate,
        beta,
        a_log,
        dt_bias,
        o,
        tinv,
        state_h,
        state_m,
        state_x,
        seed,
        final_state,
        checkpoints,
    )
    return _compile_chain_forward(
        get_dtype(q.dtype),
        get_dtype(state_x.dtype),
        get_dtype(final_state.dtype) if final_state is not None else None,
        get_dtype(gate.dtype),
        get_dtype(a_log.dtype) if a_log is not None else None,
        (get_dtype(dt_bias.dtype), dt_bias.ndim) if dt_bias is not None else None,
        get_dtype(beta.dtype),
        get_dtype(seed.dtype) if seed is not None else None,
        int(num_sm),
        int(DK),
        int(DV),
        int(unit_chunks),
        int(b_t),
        int(expand_num),
        bool(length_rule),
        bool(log_gate),
        bool(safe_gate),
        bool(use_beta_sigmoid),
        bool(allow_neg_eigval),
        int(checkpoint_every_n_tokens) > 0,
        seed_indices is not None,
        final_indices is not None,
        int(chain_rows),
        use_int64_offsets,
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
) -> None:
    """Replay the chain forward on the current stream: one crossing into the DSL for the five
    launches.  The plan validated the contract at build, so nothing here raises."""
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
    )
