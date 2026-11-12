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
# (python/cudnn/linear_attention/frost/kernel/kda_chain_forward_f16.py); the host nests the chain
# prologue and the summary/prefill Ops and compiles through a @jit_cache function over fake TVM-FFI
# signatures (+int64 variants), launching live tensors on the current Torch stream (unused
# device/stream arguments dropped); launch-contract validation of the plan buffers; upstream-only
# constexpr knobs pruned (safe_gate, A_log/dt_bias, beta sigmoid, allow_neg_eigval, Q/K L2 norm);
# Ruff formatting. Every launch is a warpdrive (CuTeDSL 4.6.2) kernel and the chain
# prologue builds only tables (no per-piece descriptor arrays); the batch-invariant length rule and
# piece cadence are gone.

"""One compiled launch for the KDA chain forward: chain tables, fused summary, fp32 state chain
and prefill issued from a single host, so the Python side crosses into the DSL once per call
instead of four times. Every kernel is a warpdrive kernel (``piece_chain``,
``kda_summary``, ``kda_prefill``) addressing every piece through ``ragged``
descriptors. A buffer that two kernels read through different signature types is passed twice,
once per type: H, M and X (the summary's and prefill's views and the state chain's device views).
Compiled at ``--opt-level 2``, the level of every KDA module."""

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute

from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache

from .. import kda_prefill, kda_summary, piece_chain
from ..contract import (
    get_dtype,
    memoize_launch_check,
    op_dtype_name,
    validate_cuda_tensors,
    validate_kda_forward_operands,
    validate_kda_forward_states,
    validate_seqlens,
    validate_tensor,
)
from ..kda_common import B_T
from ..work_tables import WORK_ITEM_FIELDS
from .tvm_ffi import (
    make_compact_signature_tensor,
    make_dynamic_signature_tensor,
)


@cute.jit
def chain_forward_host(
    summary_op: cutlass.Constexpr,
    prefill_op: cutlass.Constexpr,
    dim_v: cutlass.Constexpr[int],
    dim_k: cutlass.Constexpr[int],
    chain_rows: cutlass.Constexpr[int],
    pieces: cutlass.Int32,
    heads_out: cutlass.Int32,
    num_seqs: cutlass.Int32,
    scale: cutlass.Float32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    beta: cute.Tensor,
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
    state_h_summary: cute.Tensor,
    state_m_summary: cute.Tensor,
    state_h_chain: cute.Tensor,
    state_m_chain: cute.Tensor,
    state_x_chain: cute.Tensor,
    seed: cute.Tensor | None,
    seed_indices: cute.Tensor | None,
    state_x_prefill: cute.Tensor,
    final_state: cute.Tensor | None,
    tma_slots: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    piece_chain.launch_chain_tables(
        B_T,
        pieces,
        heads_out,
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
    )
    summary_op.launch(
        k,
        v,
        gate,
        beta,
        state_h_summary,
        state_m_summary,
        work_items_summary,
        summary_count,
        scheduler_summary,
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
        seed_indices,
        stream,
    )
    prefill_op.launch(
        q,
        k,
        v,
        gate,
        beta,
        state_x_prefill,
        o,
        final_state,
        None,
        None,
        work_items,
        main_count,
        scheduler_prefill,
        tma_slots,
        scale,
        stream,
    )


def _dynamic(dtype, rank: int, align: int, use_int64_offsets: bool):
    """The legacy ``from_dlpack(t, assumed_align=align).mark_layout_dynamic()`` placeholder (last
    mode contiguous)."""
    return make_dynamic_signature_tensor(
        dtype, rank, assumed_align=align, use_int64_offsets=use_int64_offsets
    )


def _work_table(use_int64_offsets: bool):
    """The legacy work-item placeholder: compact ``[rows, WORK_ITEM_FIELDS]`` int32, only the row
    count dynamic."""
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    return make_compact_signature_tensor(
        cutlass.Int32, (sym_int(), WORK_ITEM_FIELDS), assumed_align=16
    )


@jit_cache
def _compile_chain_forward(
    io_dtype,
    gate_dtype,
    beta_dtype,
    cu_seqlens_dtype,
    state_dtype,
    final_state_dtype,
    seed_dtype,
    has_seed_indices: bool,
    d_k: int,
    d_v: int,
    chain_rows: int,
    num_sm: int,
    use_int64_offsets: bool,
):
    """Compile the chain forward host for one static config over fake tensors that repeat the
    standalone builds' placeholders.  Absent tensors have a None dtype."""
    i64 = use_int64_offsets
    io, gate_name = op_dtype_name(io_dtype), op_dtype_name(gate_dtype)
    summary_op = kda_summary.KdaSummary(io, gate_name, d_k, d_v, num_sm)
    prefill_op = kda_prefill.KdaPrefill(
        io, gate_name, d_k, d_v, 1, True, final_state_dtype is not None, 0, num_sm
    )
    static = (
        final_state_dtype is not None,
        seed_dtype is not None,
        has_seed_indices,
        d_k,
        d_v,
        chain_rows,
        num_sm,
        i64,
    )
    dtype_names = "_".join(
        "none" if dtype is None else dtype.__name__.lower()
        for dtype in (
            io_dtype,
            gate_dtype,
            beta_dtype,
            cu_seqlens_dtype,
            state_dtype,
            final_state_dtype,
            seed_dtype,
        )
    )
    name = "kda_chain_forward_" + "_".join(str(int(flag)) for flag in static) + f"_{dtype_names}"

    def f32_state():
        return _dynamic(cutlass.Float32, 4, 16, i64)

    def counter():
        return _dynamic(cutlass.Int32, 1, 4, i64)

    return compile_tvm_ffi(
        chain_forward_host,
        summary_op,
        prefill_op,
        d_v,
        d_k,
        chain_rows,
        *(cutlass.Int32(0) for _ in range(3)),  # pieces, heads_out, num_seqs
        cutlass.Float32(0),  # scale
        _dynamic(io_dtype, 3, 16, i64),  # q
        _dynamic(io_dtype, 3, 16, i64),  # k
        _dynamic(io_dtype, 3, 16, i64),  # v
        _dynamic(gate_dtype, 3, 16, i64),  # gate
        _dynamic(beta_dtype, 2, 4, i64),  # beta
        _dynamic(io_dtype, 3, 16, i64),  # o
        _dynamic(
            cu_seqlens_dtype, 1, 8 if cu_seqlens_dtype is cutlass.Int64 else 4, i64
        ),  # cu_seqlens
        counter(),  # cu_pieces
        _dynamic(cutlass.Int32, 1, 16, i64),  # main_rows
        _dynamic(cutlass.Int32, 1, 16, i64),  # summary_rows
        counter(),  # main_count
        counter(),  # summary_count
        _work_table(i64),  # work_items
        _work_table(i64),  # work_items_summary
        counter(),  # scheduler_all
        counter(),  # scheduler_summary
        counter(),  # scheduler_prefill
        f32_state(),  # state_h_summary
        f32_state(),  # state_m_summary
        f32_state(),  # state_h_chain
        f32_state(),  # state_m_chain
        f32_state(),  # state_x_chain
        _dynamic(seed_dtype, 4, 4, i64) if seed_dtype is not None else None,  # seed
        counter() if has_seed_indices else None,  # seed_indices
        _dynamic(state_dtype, 4, 16, i64),  # state_x_prefill
        _dynamic(final_state_dtype, 4, 16, i64) if final_state_dtype is not None else None,
        make_compact_signature_tensor(
            cutlass.Int64, ((cute.sym_int64 if i64 else cute.sym_int)(), 16), assumed_align=128
        ),  # tma_slots
        name=name,
        opt_level=2,
    )


def _dtype_or_none(tensor):
    return None if tensor is None else get_dtype(tensor.dtype)


@memoize_launch_check
def _validate_launch(
    *,
    q,
    k,
    v,
    gate,
    beta,
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
    state_h,
    state_m,
    state_x,
    seed,
    seed_indices,
    final_state,
    pieces,
    heads_out,
    num_seqs,
    b_t,
    chain_rows,
    num_sm,
):
    """Check the chain forward buffers of one plan: ``num_seqs * pieces`` piece states and the
    piece tables."""
    if validate_seqlens(cu_seqlens) != num_seqs or gate.ndim != 3 or gate.shape[1] != heads_out:
        raise ValueError("num_seqs and heads_out must match cu_seqlens and gate")
    if min(pieces, num_sm) < 1:
        raise ValueError("pieces and num_sm must be positive")
    if b_t != B_T:
        raise ValueError(f"the KDA chain runs b_t={B_T} chunks, got {b_t}")
    validate_kda_forward_operands(q, k, v, gate, beta, o, cu_seqlens)
    validate_kda_forward_states(
        q,
        v,
        heads_out,
        num_seqs,
        seed=(seed, seed_indices is not None),
        final_state=(final_state, False),
    )
    num_pieces, dim_v, dim_k = num_seqs * pieces, v.shape[2], q.shape[2]
    tables = {
        "cu_pieces": cu_pieces,
        "main_rows": main_rows,
        "summary_rows": summary_rows,
        "main_count": main_count,
        "summary_count": summary_count,
        "work_items": work_items,
        "work_items_summary": work_items_summary,
        "scheduler_all": scheduler_all,
        "scheduler_summary": scheduler_summary,
        "scheduler_prefill": scheduler_prefill,
        "seed_indices": seed_indices,
    }
    states = {"state_h": state_h, "state_m": state_m, "state_x": state_x}
    validate_cuda_tensors(q, **tables, **states)
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
    for name in (
        "main_count",
        "summary_count",
        "scheduler_all",
        "scheduler_summary",
        "scheduler_prefill",
    ):
        validate_tensor(name, tables[name], (None,), ("int32",), align=4, min_rows=1)
    if seed_indices is not None:
        validate_tensor(
            "seed_indices", seed_indices, (num_seqs,), ("int32",), align=4, compact=True
        )
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
            name, tensor, (None, heads_out, width, dim_k), ("float32",), min_rows=num_pieces
        )


def build_chain_forward(
    *,
    q,
    k,
    v,
    gate,
    beta,
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
    state_h,
    state_m,
    state_x,
    seed,
    seed_indices,
    final_state,
    final_indices,
    checkpoints,
    tma_slots,
    pieces,
    heads_out,
    num_seqs,
    b_t,
    log_gate,
    checkpoint_every_n_tokens,
    scale,
    chain_rows,
    num_sm,
):
    """Compile (persisted per static config: dtypes, dims, final-state
    presence, seed dtype, chain rows, SM count, the int64 ABI) the chain forward launch over the
    buffers of one plan; ``pieces``, ``heads_out`` and ``num_seqs`` are launch arguments. The
    launch runs on the current Torch stream. The warpdrive KDA prefill takes no ``final_indices``
    and writes no checkpoints."""
    if not log_gate:
        raise ValueError("the warpdrive KDA chain forward takes natural-log gates")
    if final_indices is not None:
        raise ValueError("the warpdrive KDA prefill takes no final_indices")
    if int(checkpoint_every_n_tokens) > 0 or checkpoints is not None:
        raise ValueError("the warpdrive KDA prefill writes no checkpoints")
    # Attention Gym modification: validate the launch contract before selecting a compiled ABI.
    _validate_launch(
        q=q,
        k=k,
        v=v,
        gate=gate,
        beta=beta,
        o=o,
        cu_seqlens=cu_seqlens,
        cu_pieces=cu_pieces,
        main_rows=main_rows,
        summary_rows=summary_rows,
        main_count=main_count,
        summary_count=summary_count,
        work_items=work_items,
        work_items_summary=work_items_summary,
        scheduler_all=scheduler_all,
        scheduler_summary=scheduler_summary,
        scheduler_prefill=scheduler_prefill,
        state_h=state_h,
        state_m=state_m,
        state_x=state_x,
        seed=seed,
        seed_indices=seed_indices,
        final_state=final_state,
        pieces=pieces,
        heads_out=heads_out,
        num_seqs=num_seqs,
        b_t=b_t,
        chain_rows=chain_rows,
        num_sm=num_sm,
    )
    del pieces, heads_out, num_seqs, scale
    for name, tensor in (("state_h", state_h), ("state_m", state_m), ("state_x", state_x)):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32, got {tensor.dtype}")
    if seed_indices is not None and seed_indices.dtype != torch.int32:
        raise ValueError(f"seed_indices must be int32, got {seed_indices.dtype}")
    tensors = (
        q,
        k,
        v,
        gate,
        beta,
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
        state_h,
        state_m,
        state_x,
        seed,
        seed_indices,
        final_state,
    )
    return _compile_chain_forward.by_args(
        get_dtype(q.dtype),
        get_dtype(gate.dtype),
        get_dtype(beta.dtype),
        cutlass.Int64 if cu_seqlens.dtype == torch.int64 else cutlass.Int32,
        get_dtype(state_x.dtype),
        _dtype_or_none(final_state),
        _dtype_or_none(seed),
        seed_indices is not None,
        int(q.shape[2]),
        int(v.shape[2]),
        int(chain_rows),
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
    state_h,
    state_m,
    state_x,
    seed,
    seed_indices,
    final_state,
    tma_slots,
    pieces,
    heads_out,
    num_seqs,
    scale,
    final_indices=None,
    checkpoints=None,
    checkpoint_every_n_tokens=0,
) -> None:
    """Replay the chain forward: one crossing into the DSL for the four launches, on the current
    Torch stream.  The plan validated the contract at build, so nothing here raises. The trailing
    keywords only shape the build; they are accepted so a caller replays with the build's
    buffers."""
    compiled(
        int(pieces),
        int(heads_out),
        int(num_seqs),
        float(scale),
        q,
        k,
        v,
        gate,
        beta,
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
        state_h,
        state_m,
        state_h,
        state_m,
        state_x,
        seed,
        seed_indices,
        state_x,
        final_state,
        tma_slots,
    )
