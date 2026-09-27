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
# attn_gym.linear._delta_rule.cudnn_fe; the head and tail launches compile through jit_cache on
# fake-tensor TVM-FFI signatures (legacy placeholder ABI) with an int64-shape variant, after
# launch-contract validation; the unused device/stream and bprop_module arguments and the GDP
# d_v=64 bprop fork dropped; the upstream-only GDP expand_num/summary_q_step,
# safe_gate/A_log/dt_bias, beta-sigmoid, negative-eigenvalue, fused-l2norm (inv_q/inv_k), and
# compact_qdo host paths removed at their pinned values; Ruff formatting.

"""Two compiled launches for the GDN chain backward.  The head (``--opt-level 2``, the option of
the T pass, the summary and the recompute) runs the chain prologue, the T pass, the fused H and M
summary and the forward state chain (M alone from the recompute when the forward's series is passed
back) and the seeded series recompute; the tail (``--opt-level 2``, the option of the bprop summary
and the bprop) runs the G summary, the reverse state chain and the bprop.  The series recompute
moves ahead of the G summary (it depends only on the forward chain); the reverse chain is the one
GDN kernel compiled at 2 while GDN's standalone level is 3 (same instruction count, one runtime
division per CTA folded differently).  Only gdn_bprop_f16 is vendored (upstream's GDP d_v = 64 fork
and its ``compact_qdo`` operands are not), so the hosts have no ``compact_qdo`` path.  Every
kernel, its host and the tensor placeholder each host was compiled with are the standalone modules'
own; a buffer two hosts read through different placeholder types is passed twice, once per type (H,
M and X: the summary's and recompute's mode-3 compact views against the state chain's ``(1, HO, V,
K)`` device views)."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute

from attn_gym._backends.cute import compile_tvm_ffi, jit_cache
from attn_gym._backends.cute.utils import requires_int64_abi

from ..common.host import get_dtype, validate_cuda_tensors
from ..common.launch import validate_seqlens, validate_tensor, validate_workspace
from ..common.piece_chain import CHAIN_WARPS, dtype_name, launch_state_chain
from ..common.tvm_ffi import WORK_ITEM_FIELDS, make_signature, signature_key, signature_spec
from . import (
    gdn_bprop_f16,
    gdn_bprop_summary_f16,
    gdn_chain_prologue_f16,
    gdn_recompute_f16,
    gdn_summary_f16,
    gdn_tinv_f16,
)


@cute.jit
def chain_backward_head_host(
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    fused_h_m: cutlass.Constexpr[bool],
    series: cutlass.Constexpr[bool],
    tinv_cfg: cutlass.Constexpr,
    summary_cfg: cutlass.Constexpr,
    transition_cfg: cutlass.Constexpr,
    series_cfg: cutlass.Constexpr,
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    chain_rows: cutlass.Constexpr[int],
    has_seed: cutlass.Constexpr[bool],
    pieces: cutlass.Int32,
    heads_out: cutlass.Int32,
    num_seqs: cutlass.Int32,
    series_span_chunks: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    seed_every_n: cutlass.Int32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    gate: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
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
    tinv_words: cute.Tensor,
    tinv_rows: cute.Tensor,
    tinv_row_count: cute.Tensor,
    summary_words: cute.Tensor | None,
    recompute_m_words: cute.Tensor,
    series_words: cute.Tensor | None,
    bprop_summary_words: cute.Tensor,
    bprop_words: cute.Tensor,
    checkpoints: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    summary_q: cute.Tensor,
    summary_do: cute.Tensor,
    tinv: cute.Tensor,
    state_h_summary: cute.Tensor | None,
    state_m_main: cute.Tensor,
    state_h_chain: cute.Tensor | None,
    state_m_chain: cute.Tensor | None,
    state_x_chain: cute.Tensor | None,
    seed: cute.Tensor | None,
    state_x_series: cute.Tensor | None,
    seed_checkpoints: cute.Tensor | None,
    stream: cuda.CUstream,
) -> None:
    gdn_chain_prologue_f16.chain_prologue(
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
        tinv_words,
        tinv_rows,
        tinv_row_count,
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
        None,
        do,
        checkpoints,
        dq,
        dk,
        dv,
        summary_q,
        summary_do,
        tinv,
        stream,
    )
    gdn_tinv_f16.host(
        tinv_cfg,
        False,
        k,
        tinv_words,
        gate,
        beta,
        cu_pieces,
        tinv,
        tinv_rows,
        tinv_row_count,
        stream,
    )
    if cutlass.const_expr(fused_h_m):
        gdn_summary_f16.host(
            summary_cfg,
            k,
            v,
            gate,
            cu_pieces,
            tinv,
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
        gdn_recompute_f16.host(
            transition_cfg,
            k,
            k,
            gate,
            cu_pieces,
            None,
            state_m_main,
            None,
            tinv,
            work_items_summary,
            summary_count,
            scheduler_m,
            cutlass.Int32(0),
            cutlass.Int32(0),
            recompute_m_words,
            stream,
        )
    if cutlass.const_expr(series):
        gdn_recompute_f16.host(
            series_cfg,
            k,
            v,
            gate,
            cu_pieces,
            state_x_series,
            None,
            seed_checkpoints,
            tinv,
            recompute_items,
            recompute_count,
            scheduler_series,
            checkpoint_every_n,
            seed_every_n,
            series_words,
            stream,
        )


@cute.jit
def chain_backward_tail_host(
    bprop: cutlass.Constexpr,
    summary_cfg: cutlass.Constexpr,
    bprop_cfg: cutlass.Constexpr,
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    chain_rows: cutlass.Constexpr[int],
    has_seed: cutlass.Constexpr[bool],
    pieces: cutlass.Int32,
    heads_out: cutlass.Int32,
    num_seqs: cutlass.Int32,
    scale: cutlass.Float32,
    summary_q: cute.Tensor,
    k: cute.Tensor,
    gate: cute.Tensor,
    summary_do: cute.Tensor,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    state_g: cute.Tensor,
    state_m_chain: cute.Tensor,
    state_dx_end: cute.Tensor,
    seed: cute.Tensor | None,
    q: cute.Tensor,
    v: cute.Tensor,
    beta: cute.Tensor,
    do: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dgate: cute.Tensor,
    dbeta: cute.Tensor,
    dstate0: cute.Tensor | None,
    tinv: cute.Tensor,
    work_items_summary: cute.Tensor,
    summary_count: cute.Tensor,
    scheduler_summary: cute.Tensor,
    bprop_summary_words: cute.Tensor,
    work_items: cute.Tensor,
    main_count: cute.Tensor,
    scheduler_bwd: cute.Tensor,
    bprop_words: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    gdn_bprop_summary_f16.host(
        summary_cfg,
        summary_q,
        k,
        gate,
        summary_do,
        cu_pieces,
        state_g,
        None,
        tinv,
        work_items_summary,
        summary_count,
        scheduler_summary,
        scale,
        bprop_summary_words,
        stream,
    )
    launch_state_chain(
        heads_out,
        dim_v,
        dim_k,
        chain_rows,
        pieces,
        True,
        has_seed,
        False,
        False,
        num_seqs,
        state_g,
        state_m_chain,
        state_dx_end,
        seed,
        None,
        None,
        main_rows,
        None,
        stream,
    )
    bprop.host(
        bprop_cfg,
        q,
        k,
        v,
        gate,
        beta,
        dgate,
        dbeta,
        do,
        dq,
        dk,
        dv,
        cu_pieces,
        dstate0,
        state_dx_end,
        tinv,
        work_items,
        main_count,
        scheduler_bwd,
        scale,
        bprop_words,
        stream,
    )


def _as_dtype(spec, dtype):
    """Retype a spec: the state chain was compiled against device views of a fixed dtype."""
    return None if spec is None else (spec[0], dtype, *spec[2:])


def _build_cfgs(cfg_args):
    (
        io_dtype,
        state_h_dtype,
        state_m_dtype,
        state_x_dtype,
        num_sm,
        d_k,
        d_v,
        fused_h_m,
        series,
        coarse,
        log_gate,
        use_dstate0,
    ) = cfg_args
    io = get_dtype(io_dtype)
    tinv_cfg = gdn_tinv_f16.build_cfg(
        io,
        num_sm=num_sm,
        log_gate=log_gate,
        d_k=d_k,
    )
    summary_cfg = None
    transition_cfg = None
    if fused_h_m:
        summary_cfg = gdn_summary_f16.build_cfg(
            io,
            get_dtype(state_h_dtype),
            max_active_clusters=num_sm,
            use_initial_state=False,
            log_gate=log_gate,
            d_k=d_k,
            d_v=d_v,
        )
    else:
        transition_cfg = gdn_recompute_f16.build_cfg(
            io,
            get_dtype(state_m_dtype),
            max_active_clusters=num_sm,
            use_initial_state=False,
            store_final_state=True,
            enable_checkpoints=False,
            seed_checkpoints=False,
            log_gate=log_gate,
            seed_identity=True,
            v_is_zero=True,
            d_k=d_k,
            d_v=d_k,
        )
    series_cfg = None
    if series:
        series_cfg = gdn_recompute_f16.build_cfg(
            io,
            get_dtype(state_x_dtype) if not coarse else cutlass.Float32,
            max_active_clusters=num_sm,
            use_initial_state=not coarse,
            store_final_state=False,
            enable_checkpoints=True,
            seed_checkpoints=coarse,
            log_gate=log_gate,
            d_k=d_k,
            d_v=d_v,
        )
    bwd_summary_cfg = gdn_bprop_summary_f16.build_cfg(
        io,
        max_active_clusters=num_sm,
        use_dstate_in=False,
        log_gate=log_gate,
        d_k=d_k,
        d_v=d_v,
    )
    bprop_cfg = gdn_bprop_f16.build_cfg(
        io,
        max_active_clusters=num_sm,
        use_initial_state=True,
        use_dstate_in=True,
        use_dstate0=use_dstate0,
        log_gate=log_gate,
        d_k=d_k,
        d_v=d_v,
        tinv_source="gmem",
    )
    return tinv_cfg, summary_cfg, transition_cfg, series_cfg, bwd_summary_cfg, bprop_cfg


def _name(prefix, constexprs, use_int64_offsets):
    flags = "_".join(str(int(value)) for value in constexprs)
    return f"{prefix}_{flags}_i64{int(use_int64_offsets)}"


@jit_cache
def _compile_chain_backward_head(
    constexprs: tuple, cfg_args: tuple, specs: tuple, use_int64_offsets: bool
):
    """Compile the chain-backward head over the upstream dynamic-layout tensor ABI."""
    tinv_cfg, summary_cfg, transition_cfg, series_cfg, _, _ = _build_cfgs(cfg_args)
    (
        unit_chunks,
        b_t,
        length_rule,
        fused_h_m,
        series,
        dim_v,
        dim_k,
        rows,
        has_seed,
    ) = constexprs
    return compile_tvm_ffi(
        chain_backward_head_host,
        unit_chunks,
        b_t,
        length_rule,
        fused_h_m,
        series,
        tinv_cfg,
        summary_cfg,
        transition_cfg,
        series_cfg,
        dim_v,
        dim_k,
        rows,
        has_seed,
        *(cutlass.Int32(0) for _ in range(6)),
        *(make_signature(spec, use_int64_offsets=use_int64_offsets) for spec in specs),
        name=_name("gdn_cudnn_chain_backward_head", constexprs, use_int64_offsets)
        + "_"
        + "_".join(cfg_args[:4]),
        opt_level=2,
    )


@jit_cache
def _compile_chain_backward_tail(
    constexprs: tuple, cfg_args: tuple, specs: tuple, use_int64_offsets: bool
):
    """Compile the chain-backward tail over the upstream dynamic-layout tensor ABI."""
    *_, bwd_summary_cfg, bprop_cfg = _build_cfgs(cfg_args)
    dim_v, dim_k, rows, has_dseed = constexprs
    return compile_tvm_ffi(
        chain_backward_tail_host,
        gdn_bprop_f16,
        bwd_summary_cfg,
        bprop_cfg,
        dim_v,
        dim_k,
        rows,
        has_dseed,
        *(cutlass.Int32(0) for _ in range(3)),
        cutlass.Float32(1.0),
        *(make_signature(spec, use_int64_offsets=use_int64_offsets) for spec in specs),
        name=_name("gdn_cudnn_chain_backward_tail", constexprs, use_int64_offsets)
        + "_"
        + "_".join(cfg_args[:4]),
        opt_level=2,
    )


def _validate_launch(
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
    tinv_words,
    tinv_rows,
    tinv_row_count,
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
    summary_q,
    summary_do,
    tinv,
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
    fused_h_m,
    series,
    coarse,
    chain_rows,
    num_sm,
):
    """Check the chain backward buffers of one plan: ``num_seqs * pieces`` piece series and piece
    tables."""
    if validate_seqlens(cu_seqlens) != num_seqs or gate.ndim != 2 or gate.shape[1] != heads_out:
        raise ValueError("num_seqs and heads_out must match cu_seqlens and gate")
    if min(pieces, unit_chunks, num_sm) < 1:
        raise ValueError("pieces, unit_chunks and num_sm must be positive")
    num_pieces = num_seqs * pieces
    gdn_bprop_f16.validate_bwd_bundle(
        q,
        k,
        v,
        do,
        dq,
        dk,
        dv,
        gate,
        beta,
        cu_seqlens,
        checkpoints,
        tinv,
        tinv_rows,
        tinv_row_count,
        bprop_words,
        num_pieces=num_pieces,
        b_t=b_t,
        dgate=dgate,
        dbeta=dbeta,
    )
    tables = {
        "cu_pieces": cu_pieces,
        "main_rows": main_rows,
        "summary_rows": summary_rows,
        "main_count": main_count,
        "summary_count": summary_count,
        "work_items": work_items,
        "work_items_summary": work_items_summary,
        "series_items": series_items,
        "series_count": series_count,
        "scheduler_all": scheduler_all,
        "scheduler_recompute": scheduler_recompute,
        "scheduler_m": scheduler_m,
        "scheduler_series": scheduler_series,
        "scheduler_summary": scheduler_summary,
        "scheduler_bwd": scheduler_bwd,
    }
    states = {
        "state_h": state_h,
        "state_m": state_m,
        "state_x": state_x,
        "state_g": state_g,
        "state_dx_end": state_dx_end,
    }
    words = {
        "tinv_words": tinv_words,
        "summary_words": summary_words,
        "recompute_m_words": recompute_m_words,
        "series_words": series_words,
        "bprop_summary_words": bprop_summary_words,
    }
    operands = {
        "summary_q": summary_q,
        "summary_do": summary_do,
        "seed_checkpoints": seed_checkpoints,
    }
    validate_cuda_tensors(
        q, **tables, **states, **words, **operands, seed=seed, dseed=dseed, dstate0=dstate0
    )
    dim_v, dim_k = v.shape[2], q.shape[2]
    if chain_rows < CHAIN_WARPS or dim_v % chain_rows or chain_rows % CHAIN_WARPS:
        raise ValueError("chain_rows must divide d_v and contain whole chain warp groups")
    for name, entries in (
        ("cu_pieces", num_pieces + 1),
        ("main_rows", num_seqs + 1),
        ("summary_rows", num_seqs + 1),
    ):
        validate_tensor(
            name, tables[name], (entries,), ("int32",), align=4 if name == "cu_pieces" else 16
        )
    counters = (
        "main_count",
        "summary_count",
        *(name for name in tables if name.startswith("scheduler")),
    )
    for name in (*counters, *(("series_count",) if series and coarse else ())):
        validate_tensor(name, tables[name], (None,), ("int32",), align=4, min_rows=1)
    for name in (
        "work_items",
        "work_items_summary",
        *(("series_items",) if series and coarse else ()),
    ):
        validate_tensor(
            name,
            tables[name],
            (None, WORK_ITEM_FIELDS),
            ("int32",),
            compact=True,
            min_rows=num_pieces * heads_out,
        )
    for name, tensor in states.items():
        if (
            name == "state_h"
            and not fused_h_m
            or name == "state_x"
            and not (fused_h_m or series and not coarse)
        ):
            continue
        width = dim_k if name == "state_m" else dim_v
        validate_tensor(
            name,
            tensor,
            (None, heads_out, width, dim_k),
            gdn_bprop_f16.STATE_DTYPES,
            min_rows=num_pieces,
        )
    for name, tensor in (("seed", seed), ("dseed", dseed), ("dstate0", dstate0)):
        if tensor is not None:
            validate_tensor(
                name,
                tensor,
                (None, heads_out, dim_v, dim_k),
                gdn_bprop_f16.STATE_DTYPES,
                align=16 if name == "dstate0" else 4,
                min_rows=num_seqs,
            )
    io = (str(q.dtype).removeprefix("torch."),)
    validate_tensor("summary_q", summary_q, (None, None, dim_k), io, tma=True)
    validate_tensor("summary_do", summary_do, (None, heads_out, dim_v), io, tma=True)
    if coarse:
        validate_tensor(
            "seed_checkpoints",
            seed_checkpoints,
            (None, heads_out, dim_v, dim_k),
            (*gdn_bprop_f16.STATE_DTYPES, *io),
        )
    for name, module, needed in (
        ("tinv_words", gdn_tinv_f16, True),
        ("summary_words", gdn_summary_f16, fused_h_m),
        ("recompute_m_words", gdn_recompute_f16, not fused_h_m),
        ("series_words", gdn_recompute_f16, series),
        ("bprop_summary_words", gdn_bprop_summary_f16, True),
    ):
        if needed:
            validate_workspace(name, words[name], module.TENSORMAP_DESC_ARRAYS, num_pieces)


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
    tinv_words,
    tinv_rows,
    tinv_row_count,
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
    summary_q,
    summary_do,
    tinv,
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
    scale,
    chain_rows,
    num_sm,
):
    """Compile (cached per static config) the head and tail launches of the chain backward over the
    buffers of one plan; ``pieces``, ``heads_out`` and ``num_seqs`` are launch arguments.  The
    placeholders repeat the marks of the standalone modules' builds so every kernel compiles as it
    does there."""
    _HQ, DK = q.shape[1], q.shape[2]
    k.shape[1]
    _HV, DV = v.shape[1], v.shape[2]
    # Attention Gym modification: validate the launch contract before selecting a compiled ABI.
    _validate_launch(
        q=q,
        k=k,
        v=v,
        do=do,
        gate=gate,
        beta=beta,
        cu_seqlens=cu_seqlens,
        cu_pieces=cu_pieces,
        main_rows=main_rows,
        summary_rows=summary_rows,
        main_count=main_count,
        summary_count=summary_count,
        work_items=work_items,
        work_items_summary=work_items_summary,
        series_items=series_items,
        series_count=series_count,
        scheduler_all=scheduler_all,
        scheduler_recompute=scheduler_recompute,
        scheduler_m=scheduler_m,
        scheduler_series=scheduler_series,
        scheduler_summary=scheduler_summary,
        scheduler_bwd=scheduler_bwd,
        tinv_words=tinv_words,
        tinv_rows=tinv_rows,
        tinv_row_count=tinv_row_count,
        summary_words=summary_words,
        recompute_m_words=recompute_m_words,
        series_words=series_words,
        bprop_summary_words=bprop_summary_words,
        bprop_words=bprop_words,
        checkpoints=checkpoints,
        seed_checkpoints=seed_checkpoints,
        dq=dq,
        dk=dk,
        dv=dv,
        dgate=dgate,
        dbeta=dbeta,
        summary_q=summary_q,
        summary_do=summary_do,
        tinv=tinv,
        state_h=state_h,
        state_m=state_m,
        state_x=state_x,
        state_g=state_g,
        state_dx_end=state_dx_end,
        seed=seed,
        dseed=dseed,
        dstate0=dstate0,
        pieces=pieces,
        heads_out=heads_out,
        num_seqs=num_seqs,
        unit_chunks=unit_chunks,
        b_t=b_t,
        fused_h_m=fused_h_m,
        series=series,
        coarse=coarse,
        chain_rows=chain_rows,
        num_sm=num_sm,
    )
    has_seed = seed is not None
    has_dseed = dseed is not None
    seed_name = dtype_name(seed.dtype) if has_seed else "float32"
    dseed_name = dtype_name(dseed.dtype) if has_dseed else "float32"
    cu_align = 8 if str(cu_seqlens.dtype).endswith("int64") else 4

    def name(tensor):
        return "none" if tensor is None else dtype_name(tensor.dtype)

    cfg_args = (
        name(q),
        name(state_h),
        name(state_m),
        name(state_x),
        int(num_sm),
        DK,
        DV,
        bool(fused_h_m),
        bool(series),
        bool(coarse),
        bool(log_gate),
        dstate0 is not None,
    )
    recompute_items = (series_items if coarse else work_items) if series else None
    recompute_count = (series_count if coarse else main_count) if series else None
    head_constexprs = (
        int(unit_chunks),
        int(b_t),
        bool(length_rule),
        bool(fused_h_m),
        bool(series),
        DV,
        DK,
        int(chain_rows),
        has_seed,
    )
    use_int64_offsets = requires_int64_abi(
        q,
        k,
        v,
        do,
        gate,
        beta,
        cu_seqlens,
        checkpoints,
        seed_checkpoints,
        dq,
        dk,
        dv,
        dgate,
        dbeta,
        summary_q,
        summary_do,
        tinv,
        state_h,
        state_m,
        state_x,
        state_g,
        state_dx_end,
        seed,
        dseed,
        dstate0,
    )
    tail_constexprs = (DV, DK, int(chain_rows), has_dseed)
    # Every tensor a signature spec reads, so the warm path keys both launches without the specs.
    tensor_key = signature_key(
        dynamic=(
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
            series_count,
            scheduler_all,
            scheduler_recompute,
            scheduler_m,
            scheduler_series,
            scheduler_summary,
            scheduler_bwd,
            tinv_words,
            tinv_rows,
            tinv_row_count,
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
            summary_q,
            summary_do,
            tinv,
            state_h,
            state_m,
            state_x,
            state_g,
            state_dx_end,
            seed,
            dseed,
            dstate0,
        ),
        compact=(work_items, work_items_summary, series_items),
    )
    static_key = (head_constexprs, tail_constexprs, cfg_args, use_int64_offsets, tensor_key)

    def compile_head():
        head_specs = (
            signature_spec(q, assumed_align=16),
            signature_spec(k, assumed_align=16),
            signature_spec(v, assumed_align=16),
            signature_spec(do, assumed_align=16),
            signature_spec(gate, assumed_align=16),
            signature_spec(beta, assumed_align=16),
            signature_spec(cu_seqlens, assumed_align=cu_align),
            signature_spec(cu_pieces, assumed_align=4),
            signature_spec(main_rows, assumed_align=16),
            signature_spec(summary_rows, assumed_align=16),
            signature_spec(main_count, assumed_align=4),
            signature_spec(summary_count, assumed_align=4),
            signature_spec(work_items, assumed_align=16, compact=True),
            signature_spec(work_items_summary, assumed_align=16, compact=True),
            signature_spec(series_items, assumed_align=16, compact=True),
            signature_spec(series_count, assumed_align=4),
            signature_spec(recompute_items, assumed_align=16, compact=True),
            signature_spec(recompute_count, assumed_align=4),
            signature_spec(scheduler_all, assumed_align=4),
            signature_spec(scheduler_recompute, assumed_align=4),
            signature_spec(scheduler_m, assumed_align=4),
            signature_spec(scheduler_series, assumed_align=4),
            signature_spec(tinv_words, assumed_align=128),
            signature_spec(tinv_rows, assumed_align=16),
            signature_spec(tinv_row_count, assumed_align=4),
            signature_spec(summary_words, assumed_align=128),
            signature_spec(recompute_m_words, assumed_align=128),
            signature_spec(series_words, assumed_align=128),
            signature_spec(bprop_summary_words, assumed_align=128),
            signature_spec(bprop_words, assumed_align=128),
            signature_spec(checkpoints, assumed_align=16),
            signature_spec(dq, assumed_align=16),
            signature_spec(dk, assumed_align=16),
            signature_spec(dv, assumed_align=16),
            signature_spec(summary_q, assumed_align=16),
            signature_spec(summary_do, assumed_align=16),
            signature_spec(tinv, assumed_align=128),
            signature_spec(state_h, assumed_align=16, mode3_divisibility=DK)
            if fused_h_m
            else None,
            signature_spec(state_m, assumed_align=16, mode3_divisibility=DK),
            _as_dtype(signature_spec(state_h, assumed_align=16), "float32") if fused_h_m else None,
            _as_dtype(signature_spec(state_m, assumed_align=16), "float32") if fused_h_m else None,
            _as_dtype(signature_spec(state_x, assumed_align=16), "float32") if fused_h_m else None,
            _as_dtype(signature_spec(seed, assumed_align=4), seed_name),
            signature_spec(state_x, assumed_align=16, mode3_divisibility=DK)
            if series and not coarse
            else None,
            signature_spec(seed_checkpoints, assumed_align=16) if coarse else None,
        )
        return _compile_chain_backward_head(
            head_constexprs, cfg_args, head_specs, use_int64_offsets
        )

    def compile_tail():
        tail_specs = (
            signature_spec(summary_q, assumed_align=16),
            signature_spec(k, assumed_align=16),
            signature_spec(gate, assumed_align=16),
            signature_spec(summary_do, assumed_align=16),
            signature_spec(cu_pieces, assumed_align=4),
            signature_spec(main_rows, assumed_align=16),
            signature_spec(state_g, assumed_align=16),
            _as_dtype(signature_spec(state_m, assumed_align=16), "float32"),
            signature_spec(state_dx_end, assumed_align=16),
            _as_dtype(signature_spec(dseed, assumed_align=4), dseed_name),
            signature_spec(q, assumed_align=16),
            signature_spec(v, assumed_align=16),
            signature_spec(beta, assumed_align=16),
            signature_spec(do, assumed_align=16),
            signature_spec(dq, assumed_align=16),
            signature_spec(dk, assumed_align=16),
            signature_spec(dv, assumed_align=16),
            signature_spec(dgate, assumed_align=16),
            signature_spec(dbeta, assumed_align=16),
            signature_spec(dstate0, assumed_align=16),
            signature_spec(tinv, assumed_align=128),
            signature_spec(work_items_summary, assumed_align=16, compact=True),
            signature_spec(summary_count, assumed_align=4),
            signature_spec(scheduler_summary, assumed_align=4),
            signature_spec(bprop_summary_words, assumed_align=128),
            signature_spec(work_items, assumed_align=16, compact=True),
            signature_spec(main_count, assumed_align=4),
            signature_spec(scheduler_bwd, assumed_align=4),
            signature_spec(bprop_words, assumed_align=128),
        )
        return _compile_chain_backward_tail(
            tail_constexprs, cfg_args, tail_specs, use_int64_offsets
        )

    head = _compile_chain_backward_head.by_static_key(static_key, compile_head)
    tail = _compile_chain_backward_tail.by_static_key(static_key, compile_tail)
    return head, tail


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
    tinv_words,
    tinv_rows,
    tinv_row_count,
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
    summary_q,
    summary_do,
    tinv,
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
    scale,
) -> None:
    """Replay the chain backward: two crossings into the DSL for its eight launches.  The plan
    validated the contract at build, so nothing here raises."""
    head, tail = compiled
    head(
        int(pieces),
        int(heads_out),
        int(num_seqs),
        int(series_span_tokens) // int(b_t),
        int(b_t),
        int(seed_every_n_tokens),
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
        (series_items if coarse else work_items) if series else None,
        (series_count if coarse else main_count) if series else None,
        scheduler_all,
        scheduler_recompute,
        scheduler_m,
        scheduler_series,
        tinv_words,
        tinv_rows,
        tinv_row_count,
        summary_words,
        recompute_m_words,
        series_words,
        bprop_summary_words,
        bprop_words,
        checkpoints,
        dq,
        dk,
        dv,
        summary_q,
        summary_do,
        tinv,
        state_h if fused_h_m else None,
        state_m,
        state_h if fused_h_m else None,
        state_m if fused_h_m else None,
        state_x if fused_h_m else None,
        seed,
        state_x if series and not coarse else None,
        seed_checkpoints if coarse else None,
    )
    tail(
        int(pieces),
        int(heads_out),
        int(num_seqs),
        float(scale),
        summary_q,
        k,
        gate,
        summary_do,
        cu_pieces,
        main_rows,
        state_g,
        state_m,
        state_dx_end,
        dseed,
        q,
        v,
        beta,
        do,
        dq,
        dk,
        dv,
        dgate,
        dbeta,
        dstate0,
        tinv,
        work_items_summary,
        summary_count,
        scheduler_summary,
        bprop_summary_words,
        work_items,
        main_count,
        scheduler_bwd,
        bprop_words,
    )
