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
# Modified by Attention Gym in 2026: derived from cudnn-frontend v1.30.0
# (python/cudnn/linear_attention/frost/kernel/gdn_chain_backward_f16.py); the head and tail
# launches compile through jit_cache on fake-tensor TVM-FFI signatures (legacy placeholder ABI)
# with an int64-shape variant, after launch-contract validation; the unused device/stream and
# bprop_module arguments and the GDP d_v=64 bprop fork dropped; the upstream-only GDP
# expand_num/summary_q_step, safe_gate/A_log/dt_bias, beta-sigmoid, negative-eigenvalue,
# fused-l2norm (inv_q/inv_k), and compact_qdo host paths removed at their pinned values; Ruff
# formatting. Every launch is a warpdrive (CuTeDSL 4.6.2) kernel and the chain prologue
# builds only tables (no per-piece descriptor arrays); the M-only transition recompute, the
# series-free and checkpoint-seeded coarse paths, the batch-invariant length rule and the piece
# cadence the drivers never compile are gone.

"""Two compiled launches for the GDN chain backward, both at ``--opt-level 2``. The head runs the
chain tables (``piece_chain``), the T pass, the fused H and M summary, the forward state chain
and the seeded series recompute (``gdn_recompute``); the tail runs the G summary
(``gdn_bprop_summary``), the reverse state chain and the bprop (``gdn_bprop``). The
series recompute moves ahead of the G summary (it depends only on the forward chain). Every kernel
addresses every piece through ``ragged`` descriptors (one per tensor; sequences ending in
the first box get per-item slots). A buffer two kernels read through
different placeholder types is passed twice, once per type (H, M and X: the summary's and
recompute's mode-3 compact views against the state chain's device views)."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute

from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache

from .. import (
    gdn_bprop,
    gdn_bprop_summary,
    gdn_recompute,
    gdn_summary,
    gdn_tinv,
    piece_chain,
)
from ..contract import (
    STATE_DTYPES,
    dtype_name,
    get_dtype,
    memoize_launch_check,
    validate_cuda_tensors,
    validate_gdn_bwd_bundle,
    validate_gdn_tinv,
    validate_seqlens,
    validate_tensor,
)
from ..work_tables import WORK_ITEM_FIELDS
from .tvm_ffi import make_signature, signature_key, signature_spec

B_T = gdn_recompute.B_T


@cute.jit
def chain_backward_head_host(
    tinv_cfg: cutlass.Constexpr,
    summary_cfg: cutlass.Constexpr,
    series_cfg: cutlass.Constexpr,
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    chain_rows: cutlass.Constexpr[int],
    pieces: cutlass.Int32,
    heads_out: cutlass.Int32,
    num_seqs: cutlass.Int32,
    k: cute.Tensor,
    v: cute.Tensor,
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
    scheduler_all: cute.Tensor,
    scheduler_series: cute.Tensor,
    tinv_rows: cute.Tensor,
    tinv_row_count: cute.Tensor,
    checkpoints: cute.Tensor,
    tinv: cute.Tensor,
    state_h_summary: cute.Tensor,
    state_m_main: cute.Tensor,
    state_h_chain: cute.Tensor,
    state_m_chain: cute.Tensor,
    state_x_chain: cute.Tensor,
    seed: cute.Tensor | None,
    state_x_series: cute.Tensor,
    tma_slots: cute.Tensor,
    dgate: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    piece_chain.launch_chain_tables(
        B_T,
        pieces,
        heads_out,
        cutlass.Int32(gate.shape[0]),
        cu_seqlens,
        cu_pieces,
        main_rows,
        summary_rows,
        main_count,
        summary_count,
        work_items,
        work_items_summary,
        scheduler_all,
        stream,
        rows=tinv_rows,
        row_count=tinv_row_count,
        rows_b_t=B_T,
        tail_rows=dgate,
    )
    tinv_op, tinv_sms = tinv_cfg
    tinv_op.launch(
        k,
        gate,
        beta,
        cu_pieces,
        tinv,
        tinv_rows,
        tinv_row_count,
        tma_slots,
        cutlass.Int32(tinv_sms),
        stream,
        emit_row_table=False,
    )
    summary_cfg.launch(
        k,
        v,
        gate,
        tinv,
        state_h_summary,
        state_m_main,
        work_items_summary,
        summary_count,
        tma_slots,
        stream,
    )
    piece_chain.launch_state_chain(
        dim_v,
        dim_k,
        chain_rows,
        False,
        heads_out,
        num_seqs,
        state_h_chain,
        state_m_chain,
        state_x_chain,
        seed,
        main_rows,
        None,
        stream,
    )
    series_op, series_sms = series_cfg
    series_op.launch(
        k,
        v,
        gate,
        tinv,
        work_items,
        main_count,
        scheduler_series,
        tma_slots,
        state_x_series,
        checkpoints,
        cutlass.Int32(series_sms),
        stream,
    )


@cute.jit
def chain_backward_tail_host(
    summary_cfg: cutlass.Constexpr,
    bprop_cfg: cutlass.Constexpr,
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    chain_rows: cutlass.Constexpr[int],
    pieces: cutlass.Int32,
    heads_out: cutlass.Int32,
    num_seqs: cutlass.Int32,
    scale: cutlass.Float32,
    summary_q: cute.Tensor,
    k: cute.Tensor,
    gate: cute.Tensor,
    summary_do: cute.Tensor,
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
    work_items: cute.Tensor,
    main_count: cute.Tensor,
    scheduler_bwd: cute.Tensor,
    checkpoints: cute.Tensor,
    tma_slots: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    summary_cfg.launch(
        summary_q,
        k,
        gate,
        summary_do,
        tinv,
        state_g,
        work_items_summary,
        summary_count,
        tma_slots,
        scale,
        stream,
    )
    piece_chain.launch_state_chain(
        dim_v,
        dim_k,
        chain_rows,
        True,
        heads_out,
        num_seqs,
        state_g,
        state_m_chain,
        state_dx_end,
        seed,
        main_rows,
        None,
        stream,
    )
    bprop_op, bprop_sms = bprop_cfg
    bprop_op.launch(
        q,
        k,
        v,
        gate,
        beta,
        do,
        checkpoints,
        dq,
        dk,
        dv,
        dgate,
        dbeta,
        state_dx_end,
        dstate0,
        work_items,
        main_count,
        scheduler_bwd,
        tma_slots,
        scale,
        cutlass.Int32(bprop_sms),
        stream,
    )


def _as_dtype(spec, dtype):
    """Retype a spec: the state chain was compiled against device views of a fixed dtype."""
    return None if spec is None else (spec[0], dtype, *spec[2:])


def _build_cfgs(cfg_args):
    # The state M/X dtypes key the compile (the traced tensors carry them).
    (
        io_dtype,
        state_h_dtype,
        _state_m_dtype,
        _state_x_dtype,
        num_sm,
        d_k,
        d_v,
        log_gate,
        dstate0,
    ) = cfg_args
    io = get_dtype(io_dtype)
    tinv_cfg = gdn_tinv.bundle_hook(io, d_k, num_sm, log_gate)
    summary_cfg = gdn_summary.bundle_hook(io, get_dtype(state_h_dtype), d_k, d_v, log_gate)
    series_cfg = gdn_recompute.bundle_hook(io, d_k, d_v, True, num_sm)
    bwd_summary_cfg = gdn_bprop_summary.bundle_hook(io, d_k, d_v, log_gate)
    bprop_cfg = gdn_bprop.bundle_hook(io, d_k, d_v, True, True, dstate0, log_gate, num_sm)
    return tinv_cfg, summary_cfg, series_cfg, bwd_summary_cfg, bprop_cfg


def _name(prefix, constexprs, use_int64_offsets):
    flags = "_".join(str(int(value)) for value in constexprs)
    return f"{prefix}_{flags}_i64{int(use_int64_offsets)}"


@jit_cache
def _compile_chain_backward_head(
    constexprs: tuple, cfg_args: tuple, specs: tuple, use_int64_offsets: bool
):
    """Compile the chain-backward head over the upstream dynamic-layout tensor ABI."""
    tinv_cfg, summary_cfg, series_cfg, _, _ = _build_cfgs(cfg_args)
    return compile_tvm_ffi(
        chain_backward_head_host,
        tinv_cfg,
        summary_cfg,
        series_cfg,
        *constexprs,
        *(cutlass.Int32(0) for _ in range(3)),
        *(make_signature(spec, use_int64_offsets=use_int64_offsets) for spec in specs),
        name=_name("gdn_wd_chain_backward_head", constexprs, use_int64_offsets)
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
    return compile_tvm_ffi(
        chain_backward_tail_host,
        bwd_summary_cfg,
        bprop_cfg,
        *constexprs,
        *(cutlass.Int32(0) for _ in range(3)),
        cutlass.Float32(1.0),
        *(make_signature(spec, use_int64_offsets=use_int64_offsets) for spec in specs),
        name=_name("gdn_wd_chain_backward_tail", constexprs, use_int64_offsets)
        + "_"
        + "_".join(cfg_args[:4]),
        opt_level=2,
    )


@memoize_launch_check
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
    scheduler_all,
    scheduler_series,
    scheduler_summary,
    scheduler_bwd,
    tinv_rows,
    tinv_row_count,
    checkpoints,
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
    chain_rows,
    num_sm,
):
    """Check the chain backward buffers of one plan: ``num_seqs * pieces`` piece series and piece
    tables."""
    if validate_seqlens(cu_seqlens) != num_seqs or gate.ndim != 2 or gate.shape[1] != heads_out:
        raise ValueError("num_seqs and heads_out must match cu_seqlens and gate")
    if min(pieces, num_sm) < 1:
        raise ValueError("pieces and num_sm must be positive")
    if b_t != B_T:
        raise ValueError(f"the GDN chain backward runs b_t={B_T} chunks, got {b_t}")
    num_pieces = num_seqs * pieces
    tokens, _heads_out, _num_seqs = validate_gdn_bwd_bundle(
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
        "scheduler_all": scheduler_all,
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
    operands = {
        "tinv": tinv,
        "tinv_rows": tinv_rows,
        "tinv_row_count": tinv_row_count,
        "summary_q": summary_q,
        "summary_do": summary_do,
    }
    validate_cuda_tensors(
        q, **tables, **states, **operands, seed=seed, dseed=dseed, dstate0=dstate0
    )
    dim_v, dim_k = v.shape[2], q.shape[2]
    warps = piece_chain.CHAIN_WARPS
    if chain_rows < warps or dim_v % chain_rows or chain_rows % warps:
        raise ValueError("chain_rows must divide d_v and contain whole chain warp groups")
    for name, entries in (
        ("cu_pieces", num_pieces + 1),
        ("main_rows", num_seqs + 1),
        ("summary_rows", num_seqs + 1),
    ):
        validate_tensor(
            name, tables[name], (entries,), ("int32",), align=4 if name == "cu_pieces" else 16
        )
    for name in ("main_count", "summary_count", *(n for n in tables if n.startswith("sched"))):
        validate_tensor(name, tables[name], (None,), ("int32",), align=4, min_rows=1)
    for name in ("work_items", "work_items_summary"):
        validate_tensor(
            name,
            tables[name],
            (None, WORK_ITEM_FIELDS),
            ("int32",),
            compact=True,
            min_rows=num_pieces * heads_out,
        )
    for name, tensor in states.items():
        width = dim_k if name == "state_m" else dim_v
        validate_tensor(
            name,
            tensor,
            (None, heads_out, width, dim_k),
            STATE_DTYPES,
            min_rows=num_pieces,
        )
    for name, tensor in (("seed", seed), ("dseed", dseed), ("dstate0", dstate0)):
        if tensor is not None:
            validate_tensor(
                name,
                tensor,
                (None, heads_out, dim_v, dim_k),
                STATE_DTYPES,
                align=16 if name == "dstate0" else 4,
                min_rows=num_seqs,
            )
    io = (str(q.dtype).removeprefix("torch."),)
    validate_tensor("summary_q", summary_q, (None, None, dim_k), io, tma=True)
    validate_tensor("summary_do", summary_do, (None, heads_out, dim_v), io, tma=True)
    validate_gdn_tinv(tinv, tinv_rows, tinv_row_count, tokens, num_pieces, heads_out, io[0])


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
    scheduler_all,
    scheduler_series,
    scheduler_summary,
    scheduler_bwd,
    tinv_rows,
    tinv_row_count,
    checkpoints,
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
    tma_slots,
    pieces,
    heads_out,
    num_seqs,
    b_t,
    log_gate,
    scale,
    chain_rows,
    num_sm,
):
    """Compile (cached per static config) the head and tail launches of the chain backward over the
    buffers of one plan; ``pieces``, ``heads_out`` and ``num_seqs`` are launch arguments."""
    DK, DV = q.shape[2], v.shape[2]
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
        scheduler_all=scheduler_all,
        scheduler_series=scheduler_series,
        scheduler_summary=scheduler_summary,
        scheduler_bwd=scheduler_bwd,
        tinv_rows=tinv_rows,
        tinv_row_count=tinv_row_count,
        checkpoints=checkpoints,
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
        b_t=b_t,
        chain_rows=chain_rows,
        num_sm=num_sm,
    )
    seed_name = dtype_name(seed.dtype) if seed is not None else "float32"
    dseed_name = dtype_name(dseed.dtype) if dseed is not None else "float32"
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
        bool(log_gate),
        dstate0 is not None,
    )
    chain_constexprs = (DV, DK, int(chain_rows))
    use_int64_offsets = requires_int64_abi(
        q,
        k,
        v,
        do,
        gate,
        beta,
        cu_seqlens,
        checkpoints,
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
            scheduler_all,
            scheduler_series,
            scheduler_bwd,
            tinv_rows,
            tinv_row_count,
            checkpoints,
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
        compact=(work_items, work_items_summary, tma_slots),
    )
    static_key = (chain_constexprs, cfg_args, use_int64_offsets, tensor_key)

    def compile_head():
        head_specs = (
            signature_spec(k, assumed_align=16),
            signature_spec(v, assumed_align=16),
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
            signature_spec(scheduler_all, assumed_align=4),
            signature_spec(scheduler_series, assumed_align=4),
            signature_spec(tinv_rows, assumed_align=16),
            signature_spec(tinv_row_count, assumed_align=4),
            signature_spec(checkpoints, assumed_align=16),
            signature_spec(tinv, assumed_align=128),
            signature_spec(state_h, assumed_align=16, mode3_divisibility=DK),
            signature_spec(state_m, assumed_align=16, mode3_divisibility=DK),
            _as_dtype(signature_spec(state_h, assumed_align=16), "float32"),
            _as_dtype(signature_spec(state_m, assumed_align=16), "float32"),
            _as_dtype(signature_spec(state_x, assumed_align=16), "float32"),
            _as_dtype(signature_spec(seed, assumed_align=4), seed_name),
            signature_spec(state_x, assumed_align=16, mode3_divisibility=DK),
            signature_spec(tma_slots, assumed_align=128, compact=True),
            signature_spec(dgate, assumed_align=16),
        )
        return _compile_chain_backward_head(
            chain_constexprs, cfg_args, head_specs, use_int64_offsets
        )

    def compile_tail():
        tail_specs = (
            signature_spec(summary_q, assumed_align=16),
            signature_spec(k, assumed_align=16),
            signature_spec(gate, assumed_align=16),
            signature_spec(summary_do, assumed_align=16),
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
            signature_spec(work_items, assumed_align=16, compact=True),
            signature_spec(main_count, assumed_align=4),
            signature_spec(scheduler_bwd, assumed_align=4),
            signature_spec(checkpoints, assumed_align=16),
            signature_spec(tma_slots, assumed_align=128, compact=True),
        )
        return _compile_chain_backward_tail(
            chain_constexprs, cfg_args, tail_specs, use_int64_offsets
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
    scheduler_all,
    scheduler_series,
    scheduler_bwd,
    tinv_rows,
    tinv_row_count,
    checkpoints,
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
    dseed,
    dstate0,
    seed,
    tma_slots,
    pieces,
    heads_out,
    num_seqs,
    scale,
    scheduler_summary=None,
    b_t=B_T,
) -> None:
    """Replay the chain backward: two crossings into the DSL for its eight launches. The plan
    validated the contract at build, so nothing here raises. The trailing keywords only shape the
    build; they are accepted so a caller replays with the build's buffers."""
    head, tail = compiled
    head(
        int(pieces),
        int(heads_out),
        int(num_seqs),
        k,
        v,
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
        scheduler_all,
        scheduler_series,
        tinv_rows,
        tinv_row_count,
        checkpoints,
        tinv,
        state_h,
        state_m,
        state_h,
        state_m,
        state_x,
        seed,
        state_x,
        tma_slots,
        dgate,
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
        work_items,
        main_count,
        scheduler_bwd,
        checkpoints,
        tma_slots,
    )
