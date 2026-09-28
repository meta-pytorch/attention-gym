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
# attn_gym.linear._delta_rule.cudnn_fe; the host nests the chain prologue, KdaSummaryOp, and the
# recompute/bprop/bprop_summary hosts and compiles through a @jit_cache function over fake TVM-FFI
# signatures (int64 ABI from the kernel modules' selectors), launching on the current Torch stream
# (unused device/stream arguments dropped); upstream-only constexpr knobs pruned (safe_gate,
# A_log/dt_bias, beta sigmoid, allow_neg_eigval, Q/K L2 norm); Ruff formatting.

"""One compiled launch for the KDA chain backward: chain prologue, fused H and M summary and
forward state chain (M alone from the recompute when the forward's series is passed back), G
summary, reverse state chain, seeded series recompute and bprop issued from a single host.  Every
KDA module compiles at ``--opt-level 2``, the chain prologue and the state chain included
(``opt_level``), so the host does too and every nested kernel is the standalone one.  Every kernel,
its host and the tensor placeholder each host was compiled with are the standalone modules' own; a
buffer two hosts read through different placeholder types is passed twice, once per type: H, M, X,
G and dX (the summary, recompute and bprop hosts' torch views against the state chain's ``(1, HO,
V, K)`` device views), ``cu_pieces`` (the prologue marks it at its element alignment, every other
kernel at 8 bytes) and the gate when the bprop reads it as a linear alpha (a 4-byte view)."""

from functools import partial

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute

from attn_gym._backends.cute import compile_tvm_ffi, jit_cache

from ..common.host import get_dtype
from ..common.piece_chain import dtype_name, launch_state_chain
from ..common.tvm_ffi import (
    WORK_ITEM_FIELDS,
    make_compact_signature_tensor,
    make_dynamic_signature_tensor,
)
from . import (
    kda_bprop_f16,
    kda_bprop_summary_f16,
    kda_chain_prologue_f16,
    kda_recompute_f16,
    kda_summary_f16,
)


@cute.jit
def chain_backward_host(
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    fused_h_m: cutlass.Constexpr[bool],
    series: cutlass.Constexpr[bool],
    summary_cfg: cutlass.Constexpr,
    transition_cfg: cutlass.Constexpr,
    bwd_summary_cfg: cutlass.Constexpr,
    series_cfg: cutlass.Constexpr,
    bprop_cfg: cutlass.Constexpr,
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    chain_rows: cutlass.Constexpr[int],
    has_seed: cutlass.Constexpr[bool],
    has_dseed: cutlass.Constexpr[bool],
    pieces: cutlass.Int32,
    heads_out: cutlass.Int32,
    num_seqs: cutlass.Int32,
    series_span_chunks: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    seed_every_n: cutlass.Int32,
    scale: cutlass.Float32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    gate: cute.Tensor,
    gate_main: cute.Tensor | None,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
    cu_pieces_main: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: cute.Tensor,
    series_items: cute.Tensor | None,
    series_count: cute.Tensor | None,
    recompute_items: cute.Tensor | None,
    recompute_count: cute.Tensor | None,
    scheduler_all: cute.Tensor,
    scheduler_recompute: cute.Tensor,
    scheduler_m: cute.Tensor,
    scheduler_series: cute.Tensor,
    scheduler_summary: cute.Tensor,
    scheduler_bwd: cute.Tensor,
    summary_words: cute.Tensor | None,
    recompute_m_words: cute.Tensor,
    series_words: cute.Tensor | None,
    bprop_summary_words: cute.Tensor,
    bprop_words: cute.Tensor,
    checkpoints: cute.Tensor,
    seed_checkpoints: cute.Tensor | None,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dgate: cute.Tensor,
    dbeta: cute.Tensor,
    state_h_summary: cute.Tensor | None,
    state_m_main: cute.Tensor,
    state_h_chain: cute.Tensor | None,
    state_m_chain: cute.Tensor,
    state_x_chain: cute.Tensor | None,
    seed: cute.Tensor | None,
    state_x_series: cute.Tensor | None,
    state_g: cute.Tensor,
    state_g_chain: cute.Tensor,
    state_dx_end_chain: cute.Tensor,
    state_dx_end: cute.Tensor,
    dseed: cute.Tensor | None,
    dstate0: cute.Tensor | None,
    stream: cuda.CUstream,
) -> None:
    heads_out = cutlass.Int32(gate.shape[1])
    q_ratio = heads_out // cutlass.Int32(q.shape[1])
    k_ratio = heads_out // cutlass.Int32(k.shape[1])
    v_ratio = heads_out // cutlass.Int32(v.shape[1])
    kda_chain_prologue_f16.chain_prologue(
        pieces,
        unit_chunks,
        b_t,
        length_rule,
        heads_out,
        series_span_chunks,
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
        series_items,
        series_count,
        summary_words,
        None,
        recompute_m_words,
        series_words,
        None,
        bprop_summary_words,
        bprop_words,
        q,
        k,
        v,
        gate,
        None,
        do,
        checkpoints,
        dq,
        dk,
        dv,
        dgate,
        stream,
    )
    if cutlass.const_expr(fused_h_m):
        kda_summary_f16.KdaSummaryOp(summary_cfg)(
            k,
            v,
            gate,
            beta,
            cu_pieces_main,
            None,
            state_h_summary,
            state_m_main,
            work_items_summary,
            summary_count,
            scheduler_recompute,
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
            None,
            stream,
        )
    else:
        kda_recompute_f16.host(
            transition_cfg,
            k,
            k,
            gate,
            beta,
            cu_pieces_main,
            None,
            state_m_main,
            None,
            work_items_summary,
            summary_count,
            scheduler_m,
            recompute_m_words,
            cutlass.Int32(0),
            cutlass.Int32(0),
            stream,
        )
    kda_bprop_summary_f16.host(
        bwd_summary_cfg,
        q_ratio,
        k_ratio,
        beta,
        cu_pieces_main,
        state_g,
        None,
        work_items_summary,
        summary_count,
        scheduler_summary,
        bprop_summary_words,
        scale,
        stream,
    )
    launch_state_chain(
        heads_out,
        dim_v,
        dim_k,
        chain_rows,
        pieces,
        True,
        has_dseed,
        False,
        False,
        num_seqs,
        state_g_chain,
        state_m_chain,
        state_dx_end_chain,
        dseed,
        None,
        None,
        main_rows,
        None,
        stream,
    )
    if cutlass.const_expr(series):
        kda_recompute_f16.host(
            series_cfg,
            k,
            v,
            gate,
            beta,
            cu_pieces_main,
            state_x_series,
            None,
            seed_checkpoints,
            recompute_items,
            recompute_count,
            scheduler_series,
            series_words,
            checkpoint_every_n,
            seed_every_n,
            stream,
        )
    kda_bprop_f16.host(
        bprop_cfg,
        q_ratio,
        k_ratio,
        v_ratio,
        beta,
        gate_main,
        checkpoints,
        dgate,
        dbeta,
        cu_pieces_main,
        dstate0,
        state_dx_end,
        work_items,
        main_count,
        scheduler_bwd,
        bprop_words,
        scale,
        stream,
    )


@jit_cache
def _compile_chain_backward(
    io_dtype,
    gate_dtype,
    beta_dtype,
    cu_seqlens_dtype,
    state_m_dtype,
    state_x_dtype,
    seed_name: str,
    dseed_name: str,
    has_dstate0: bool,
    d_k: int,
    d_v: int,
    d_o: int,
    unit_chunks: int,
    b_t: int,
    length_rule: bool,
    fused_h_m: bool,
    series: bool,
    has_series_items: bool,
    coarse: bool,
    log_gate: bool,
    gate_scale_log2: float,
    has_seed: bool,
    has_dseed: bool,
    chain_rows: int,
    num_sm: int,
    use_int64_offsets: bool,
):
    """Compile the chain backward host over fake tensors that repeat the standalone kernels'
    dynamic-layout marks (int32 extents, int64 outer strides, last mode contiguous)."""
    sym_int = cute.sym_int

    def strided(dtype, rank, align):
        return make_dynamic_signature_tensor(
            dtype, rank, assumed_align=align, use_int64_offsets=use_int64_offsets
        )

    def table():
        return make_compact_signature_tensor(
            cutlass.Int32, (sym_int(), WORK_ITEM_FIELDS), assumed_align=16
        )

    gate_main = not log_gate
    flags = {
        "gate_scale_log2": gate_scale_log2,
        "log_gate": log_gate,
        "max_active_clusters": num_sm,
        "d_k": d_k,
    }
    summary_cfg = None
    transition_cfg = None
    if fused_h_m:
        summary_cfg = kda_summary_f16.build_cfg(
            io_dtype, gate_dtype, use_initial_state=False, d_v=d_v, **flags
        )
    else:
        transition_cfg = kda_recompute_f16.build_cfg(
            io_dtype,
            state_m_dtype,
            gate_dtype,
            use_initial_state=False,
            store_final_state=True,
            enable_checkpoints=False,
            seed_checkpoints=False,
            seed_identity=True,
            v_is_zero=True,
            d_v=d_k,
            **flags,
        )
    series_cfg = None
    if series:
        series_cfg = kda_recompute_f16.build_cfg(
            io_dtype,
            state_x_dtype if not coarse else cutlass.Float32,
            gate_dtype,
            use_initial_state=not coarse,
            store_final_state=False,
            enable_checkpoints=True,
            seed_checkpoints=coarse,
            d_v=d_v,
            **flags,
        )
    bwd_summary_cfg = kda_bprop_summary_f16.build_cfg(
        io_dtype, gate_dtype, use_dstate_in=False, d_v=d_o, **flags
    )
    bprop_cfg = kda_bprop_f16.build_cfg(
        io_dtype,
        gate_dtype,
        use_dstate_in=True,
        use_dstate0=has_dstate0,
        use_initial_state=True,
        d_v=d_v,
        **flags,
    )
    static = (
        has_dstate0,
        d_k,
        d_v,
        d_o,
        unit_chunks,
        b_t,
        length_rule,
        fused_h_m,
        series,
        has_series_items,
        coarse,
        log_gate,
        has_seed,
        has_dseed,
        chain_rows,
        num_sm,
        use_int64_offsets,
    )
    dtype_names = "_".join(
        "none" if dtype is None else dtype.__name__.lower()
        for dtype in (
            io_dtype,
            gate_dtype,
            beta_dtype,
            cu_seqlens_dtype,
            state_m_dtype,
            state_x_dtype,
        )
    )
    name = (
        "kda_chain_backward_"
        + "_".join(str(int(flag)) for flag in static)
        + f"_{dtype_names}_{seed_name}_{dseed_name}"
        f"_g{str(gate_scale_log2).replace('.', 'p').replace('-', 'm')}"
    )
    state = partial(strided, cutlass.Float32, 4)
    return compile_tvm_ffi(
        chain_backward_host,
        unit_chunks,
        b_t,
        length_rule,
        fused_h_m,
        series,
        summary_cfg,
        transition_cfg,
        bwd_summary_cfg,
        series_cfg,
        bprop_cfg,
        d_v,
        d_k,
        chain_rows,
        has_seed,
        has_dseed,
        *(cutlass.Int32(0) for _ in range(6)),
        cutlass.Float32(0),
        strided(io_dtype, 3, 16),  # q
        strided(io_dtype, 3, 16),  # k
        strided(io_dtype, 3, 16),  # v
        strided(io_dtype, 3, 16),  # do
        strided(gate_dtype, 3, 16),  # gate
        strided(gate_dtype, 3, 4) if gate_main else None,  # gate_main
        strided(beta_dtype, 2, 4),  # beta
        strided(cu_seqlens_dtype, 1, 8 if cu_seqlens_dtype is cutlass.Int64 else 4),  # cu_seqlens
        strided(cutlass.Int32, 1, 4),  # cu_pieces
        strided(cutlass.Int32, 1, 8),  # cu_pieces_main
        strided(cutlass.Int32, 1, 16),  # main_rows
        strided(cutlass.Int32, 1, 16),  # summary_rows
        strided(cutlass.Int32, 1, 4),  # main_count
        strided(cutlass.Int32, 1, 4),  # summary_count
        table(),  # work_items
        table(),  # work_items_summary
        table() if has_series_items else None,  # series_items
        strided(cutlass.Int32, 1, 4) if has_series_items else None,  # series_count
        table() if series else None,  # recompute_items
        strided(cutlass.Int32, 1, 4) if series else None,  # recompute_count
        *(strided(cutlass.Int32, 1, 4) for _ in range(6)),  # schedulers
        strided(cutlass.Int64, 1, 128) if fused_h_m else None,  # summary_words
        strided(cutlass.Int64, 1, 128),  # recompute_m_words
        strided(cutlass.Int64, 1, 128) if series else None,  # series_words
        strided(cutlass.Int64, 1, 128),  # bprop_summary_words
        strided(cutlass.Int64, 1, 128),  # bprop_words
        strided(io_dtype, 4, 16),  # checkpoints
        strided(io_dtype, 4, 16) if coarse else None,  # seed_checkpoints
        strided(io_dtype, 3, 16),  # dq
        strided(io_dtype, 3, 16),  # dk
        strided(io_dtype, 3, 16),  # dv
        strided(gate_dtype, 3, 16),  # dgate
        strided(beta_dtype, 2, 4),  # dbeta
        state(16) if fused_h_m else None,  # state_h_summary
        strided(state_m_dtype, 4, 16),  # state_m_main
        state(16) if fused_h_m else None,  # state_h_chain
        state(16),  # state_m_chain
        state(16) if fused_h_m else None,  # state_x_chain
        strided(get_dtype(seed_name), 4, 4) if has_seed else None,  # seed
        strided(state_x_dtype, 4, 16) if series and not coarse else None,  # state_x_series
        state(16),  # state_g
        state(16),  # state_g_chain
        state(16),  # state_dx_end_chain
        state(16),  # state_dx_end
        strided(get_dtype(dseed_name), 4, 4) if has_dseed else None,  # dseed
        state(16) if has_dstate0 else None,  # dstate0
        name=name,
        opt_level=2,
    )


def build_chain_backward(
    *,
    q,
    k,
    v,
    do,
    gate,
    beta,
    cu_seqlens,
    cu_pieces,
    main_rows,
    summary_rows,
    main_count,
    summary_count,
    work_items,
    work_items_summary,
    series_items,
    series_count,
    scheduler_all,
    scheduler_recompute,
    scheduler_m,
    scheduler_series,
    scheduler_summary,
    scheduler_bwd,
    summary_words,
    recompute_m_words,
    series_words,
    bprop_summary_words,
    bprop_words,
    checkpoints,
    seed_checkpoints,
    dq,
    dk,
    dv,
    dgate,
    dbeta,
    state_h,
    state_m,
    state_x,
    state_g,
    state_dx_end,
    seed,
    dseed,
    dstate0,
    pieces,
    heads_out,
    num_seqs,
    unit_chunks,
    b_t,
    length_rule,
    fused_h_m,
    series,
    coarse,
    series_span_tokens,
    seed_every_n_tokens,
    log_gate,
    gate_lower_bound,
    scale,
    chain_rows,
    num_sm,
):
    """Compile (persisted per static config) the chain backward launch over the buffers of one
    plan; ``pieces``, ``heads_out`` and ``num_seqs`` are launch arguments.  The fake signatures
    repeat the marks of the standalone modules' builds so every kernel compiles as it does there.
    """
    DK = q.shape[2]
    DV = v.shape[2]
    has_seed = seed is not None
    has_dseed = dseed is not None
    tensors = (
        q,
        k,
        v,
        do,
        gate,
        beta,
        cu_seqlens,
        cu_pieces,
        main_rows,
        summary_rows,
        main_count,
        summary_count,
        work_items,
        work_items_summary,
        series_items,
        series_count,
        scheduler_all,
        scheduler_recompute,
        scheduler_m,
        scheduler_series,
        scheduler_summary,
        scheduler_bwd,
        summary_words,
        recompute_m_words,
        series_words,
        bprop_summary_words,
        bprop_words,
        checkpoints,
        seed_checkpoints,
        dq,
        dk,
        dv,
        dgate,
        dbeta,
        state_h,
        state_m,
        state_x,
        state_g,
        state_dx_end,
        seed,
        dseed,
        dstate0,
    )
    # The bprop module owns the ABI selector for the bundled host (tests monkeypatch it).
    use_int64_offsets = kda_bprop_f16.requires_int64_abi(*(t for t in tensors if t is not None))
    return _compile_chain_backward(
        get_dtype(q.dtype),
        get_dtype(gate.dtype),
        get_dtype(beta.dtype),
        cutlass.Int64 if str(cu_seqlens.dtype).endswith("int64") else cutlass.Int32,
        get_dtype(state_m.dtype),
        get_dtype(state_x.dtype) if state_x is not None else cutlass.Float32,
        dtype_name(seed.dtype) if has_seed else "float32",
        dtype_name(dseed.dtype) if has_dseed else "float32",
        dstate0 is not None,
        int(DK),
        int(DV),
        int(do.shape[2]),
        int(unit_chunks),
        int(b_t),
        bool(length_rule),
        bool(fused_h_m),
        bool(series),
        series_items is not None,
        bool(coarse),
        bool(log_gate),
        float(gate_lower_bound) * kda_bprop_f16.LOG2_E,
        has_seed,
        has_dseed,
        int(chain_rows),
        int(num_sm),
        use_int64_offsets,
    )


def run_chain_backward(
    compiled,
    *,
    q,
    k,
    v,
    do,
    gate,
    beta,
    cu_seqlens,
    cu_pieces,
    main_rows,
    summary_rows,
    main_count,
    summary_count,
    work_items,
    work_items_summary,
    series_items,
    series_count,
    scheduler_all,
    scheduler_recompute,
    scheduler_m,
    scheduler_series,
    scheduler_summary,
    scheduler_bwd,
    summary_words,
    recompute_m_words,
    series_words,
    bprop_summary_words,
    bprop_words,
    checkpoints,
    seed_checkpoints,
    dq,
    dk,
    dv,
    dgate,
    dbeta,
    state_h,
    state_m,
    state_x,
    state_g,
    state_dx_end,
    seed,
    dseed,
    dstate0,
    pieces,
    heads_out,
    num_seqs,
    b_t,
    fused_h_m,
    series,
    coarse,
    series_span_tokens,
    seed_every_n_tokens,
    log_gate,
    scale,
) -> None:
    """Replay the chain backward: one crossing into the DSL for its seven launches.  The plan
    validated the contract at build, so nothing here raises."""
    compiled(
        int(pieces),
        int(heads_out),
        int(num_seqs),
        int(series_span_tokens) // int(b_t),
        int(b_t),
        int(seed_every_n_tokens),
        float(scale),
        q,
        k,
        v,
        do,
        gate,
        gate if not log_gate else None,
        beta,
        cu_seqlens,
        cu_pieces,
        cu_pieces,
        main_rows,
        summary_rows,
        main_count,
        summary_count,
        work_items,
        work_items_summary,
        series_items,
        series_count,
        (series_items if coarse else work_items) if series else None,
        (series_count if coarse else main_count) if series else None,
        scheduler_all,
        scheduler_recompute,
        scheduler_m,
        scheduler_series,
        scheduler_summary,
        scheduler_bwd,
        summary_words,
        recompute_m_words,
        series_words,
        bprop_summary_words,
        bprop_words,
        checkpoints,
        seed_checkpoints if coarse else None,
        dq,
        dk,
        dv,
        dgate,
        dbeta,
        state_h if fused_h_m else None,
        state_m,
        state_h if fused_h_m else None,
        state_m,
        state_x if fused_h_m else None,
        seed,
        state_x if series and not coarse else None,
        state_g,
        state_g,
        state_dx_end,
        state_dx_end,
        dseed,
        dstate0,
    )
