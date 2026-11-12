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
# (python/cudnn/linear_attention/frost/kernel/kda_chain_backward_f16.py); the host compiles through
# a @jit_cache function over fake TVM-FFI signatures, launching on the current Torch stream;
# upstream-only constexpr knobs pruned (safe_gate, A_log/dt_bias, beta sigmoid, allow_neg_eigval,
# Q/K L2 norm); Ruff formatting. Every launch is a warpdrive (CuTeDSL 4.6.2) kernel
# addressing each piece through ragged descriptors: no per-piece descriptor arrays; the
# M-only transition recompute, the checkpoint-seeded coarse series, the length rule and the piece
# cadence the drivers never compile are gone.

"""One compiled launch for the KDA chain backward at ``--opt-level 2``, the level of every KDA
module: the chain tables and the forward state chain (``piece_chain``), the fused H and M
summary (``kda_summary``), the G summary (``kda_bprop_summary``), the reverse state
chain, the seeded series recompute (``kda_recompute``) and the bprop (``kda_bprop``)."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute

from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache

from .. import kda_bprop, kda_bprop_summary, kda_recompute, kda_summary, piece_chain
from ..contract import dtype_name, get_dtype, op_dtype_name
from ..work_tables import WORK_ITEM_FIELDS
from .tvm_ffi import (
    make_compact_signature_tensor,
    make_dynamic_signature_tensor,
)

B_T = kda_recompute.B_T


@cute.jit
def chain_backward_host(
    summary_op: cutlass.Constexpr,
    bwd_summary_op: cutlass.Constexpr,
    series_op: cutlass.Constexpr,
    bprop_op: cutlass.Constexpr,
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
    scheduler_all: cute.Tensor,
    scheduler_summary: cute.Tensor,
    scheduler_series: cute.Tensor,
    scheduler_bwd: cute.Tensor,
    checkpoints: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    dgate: cute.Tensor,
    dbeta: cute.Tensor,
    state_h: cute.Tensor,
    state_m: cute.Tensor,
    state_x: cute.Tensor,
    seed: cute.Tensor | None,
    state_g: cute.Tensor,
    state_dx_end: cute.Tensor,
    dseed: cute.Tensor | None,
    dstate0: cute.Tensor | None,
    tma_slots: cute.Tensor,
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
    )
    summary_op.launch(
        k,
        v,
        gate,
        beta,
        state_h,
        state_m,
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
        state_h,
        state_m,
        state_x,
        seed,
        main_rows,
        None,
        stream,
    )
    bwd_summary_op.launch(
        q,
        k,
        gate,
        do,
        beta,
        state_g,
        None,
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
        state_m,
        state_dx_end,
        dseed,
        main_rows,
        None,
        stream,
    )
    series_op.launch(
        k,
        v,
        gate,
        beta,
        state_x,
        checkpoints,
        work_items,
        main_count,
        scheduler_series,
        tma_slots,
        stream,
    )
    bprop_op.launch(
        q,
        k,
        v,
        do,
        gate,
        beta,
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
        stream,
    )


@jit_cache
def _compile_chain_backward(
    io_dtype,
    gate_dtype,
    beta_dtype,
    cu_seqlens_dtype,
    seed_name: str,
    dseed_name: str,
    has_dstate0: bool,
    d_k: int,
    d_v: int,
    d_o: int,
    has_seed: bool,
    has_dseed: bool,
    chain_rows: int,
    num_sm: int,
    use_int64_offsets: bool,
):
    """Compile the chain backward host over fake tensors with dynamic extents, int64 outer strides
    under ``use_int64_offsets`` and the last mode contiguous."""
    sym_int = cute.sym_int

    def strided(dtype, rank, align):
        return make_dynamic_signature_tensor(
            dtype, rank, assumed_align=align, use_int64_offsets=use_int64_offsets
        )

    def table():
        return make_compact_signature_tensor(
            cutlass.Int32, (sym_int(), WORK_ITEM_FIELDS), assumed_align=16
        )

    def state():
        return strided(cutlass.Float32, 4, 16)

    io, gate_name = op_dtype_name(io_dtype), op_dtype_name(gate_dtype)
    summary_op = kda_summary.KdaSummary(io, gate_name, d_k, d_v, num_sm)
    bwd_summary_op = kda_bprop_summary.bundle_hook(io_dtype, gate_dtype, d_k, d_o, True)
    series_op = kda_recompute.KdaRecompute(io, gate_name, d_k, d_v, True, num_sm)
    bprop_op = kda_bprop.KdaBprop(io, gate_name, d_k, d_v, True, True, has_dstate0, num_sm)
    static = (
        has_dstate0,
        d_k,
        d_v,
        d_o,
        has_seed,
        has_dseed,
        chain_rows,
        num_sm,
        use_int64_offsets,
    )
    dtype_names = "_".join(
        dtype.__name__.lower() for dtype in (io_dtype, gate_dtype, beta_dtype, cu_seqlens_dtype)
    )
    name = (
        "kda_chain_backward_"
        + "_".join(str(int(flag)) for flag in static)
        + f"_{dtype_names}_{seed_name}_{dseed_name}"
    )
    return compile_tvm_ffi(
        chain_backward_host,
        summary_op,
        bwd_summary_op,
        series_op,
        bprop_op,
        d_v,
        d_k,
        chain_rows,
        *(cutlass.Int32(0) for _ in range(3)),
        cutlass.Float32(0),
        strided(io_dtype, 3, 16),  # q
        strided(io_dtype, 3, 16),  # k
        strided(io_dtype, 3, 16),  # v
        strided(io_dtype, 3, 16),  # do
        strided(gate_dtype, 3, 16),  # gate
        strided(beta_dtype, 2, 4),  # beta
        strided(cu_seqlens_dtype, 1, 8 if cu_seqlens_dtype is cutlass.Int64 else 4),  # cu_seqlens
        strided(cutlass.Int32, 1, 4),  # cu_pieces
        strided(cutlass.Int32, 1, 16),  # main_rows
        strided(cutlass.Int32, 1, 16),  # summary_rows
        strided(cutlass.Int32, 1, 4),  # main_count
        strided(cutlass.Int32, 1, 4),  # summary_count
        table(),  # work_items
        table(),  # work_items_summary
        *(strided(cutlass.Int32, 1, 4) for _ in range(4)),  # schedulers
        strided(io_dtype, 4, 16),  # checkpoints
        strided(io_dtype, 3, 16),  # dq
        strided(io_dtype, 3, 16),  # dk
        strided(io_dtype, 3, 16),  # dv
        strided(gate_dtype, 3, 16),  # dgate
        strided(beta_dtype, 2, 4),  # dbeta
        state(),  # state_h
        state(),  # state_m
        state(),  # state_x
        strided(get_dtype(seed_name), 4, 4) if has_seed else None,  # seed
        state(),  # state_g
        state(),  # state_dx_end
        strided(get_dtype(dseed_name), 4, 4) if has_dseed else None,  # dseed
        state() if has_dstate0 else None,  # dstate0
        make_compact_signature_tensor(
            cutlass.Int64, (cute.sym_int(), 16), assumed_align=128
        ),  # tma_slots
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
    scheduler_all,
    scheduler_summary,
    scheduler_series,
    scheduler_bwd,
    checkpoints,
    dq,
    dk,
    dv,
    dgate,
    dbeta,
    state_h,
    state_m,
    state_x,
    seed,
    state_g,
    state_dx_end,
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
    """Compile (persisted per static config) the chain backward launch over the buffers of one
    plan; ``pieces``, ``heads_out`` and ``num_seqs`` are launch arguments."""
    if not log_gate or b_t != B_T:
        raise ValueError(
            "the warpdrive KDA chain backward serves 16-token chunks of natural-log gates"
        )
    if state_m.dtype != state_h.dtype:
        raise ValueError("the piece states are fp32")
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
        scheduler_all,
        scheduler_summary,
        scheduler_series,
        scheduler_bwd,
        checkpoints,
        dq,
        dk,
        dv,
        dgate,
        dbeta,
        state_h,
        state_m,
        state_x,
        seed,
        state_g,
        state_dx_end,
        dseed,
        dstate0,
    )
    return _compile_chain_backward.by_args(
        get_dtype(q.dtype),
        get_dtype(gate.dtype),
        get_dtype(beta.dtype),
        cutlass.Int64 if str(cu_seqlens.dtype).endswith("int64") else cutlass.Int32,
        dtype_name(seed.dtype) if seed is not None else "float32",
        dtype_name(dseed.dtype) if dseed is not None else "float32",
        dstate0 is not None,
        int(q.shape[2]),
        int(v.shape[2]),
        int(do.shape[2]),
        seed is not None,
        dseed is not None,
        int(chain_rows),
        int(num_sm),
        requires_int64_abi(*(t for t in tensors if t is not None)),
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
    scheduler_all,
    scheduler_summary,
    scheduler_series,
    scheduler_bwd,
    checkpoints,
    dq,
    dk,
    dv,
    dgate,
    dbeta,
    state_h,
    state_m,
    state_x,
    seed,
    state_g,
    state_dx_end,
    dseed,
    dstate0,
    tma_slots,
    pieces,
    heads_out,
    num_seqs,
    scale,
) -> None:
    """Replay the chain backward: one crossing into the DSL for its seven launches. The plan
    validated the contract at build, so nothing here raises."""
    compiled(
        int(pieces),
        int(heads_out),
        int(num_seqs),
        float(scale),
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
        scheduler_summary,
        scheduler_series,
        scheduler_bwd,
        checkpoints,
        dq,
        dk,
        dv,
        dgate,
        dbeta,
        state_h,
        state_m,
        state_x,
        seed,
        state_g,
        state_dx_end,
        dseed,
        dstate0,
        tma_slots,
    )
