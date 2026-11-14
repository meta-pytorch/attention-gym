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
# (python/cudnn/linear_attention/frost/kernel/kda_warmup_backward_f16.py); the host compiles
# through a @jit_cache function over fake TVM-FFI signatures, launching on the current Torch
# stream; upstream-only constexpr knobs pruned (safe_gate, A_log/dt_bias, beta sigmoid,
# allow_neg_eigval, Q/K L2 norm, expand_num, the log2_threshold override); Ruff formatting. Every
# launch is a warpdrive (CuTeDSL 4.6.2) kernel addressing each sequence through
# ragged descriptors: no per-batch descriptor workspaces, no checkpoint-seeded coarse series, no
# stage switches the drivers never vary.

"""One compiled launch for the KDA warmup and uncut backward at ``--opt-level 2``, the level of
every KDA module: the split-K table (warmup only), the work-table prologue (``work_tables``:
LPT order, zeroed scheduler counters), the checkpoint-series recompute (``kda_recompute``)
and the bprop (``kda_bprop``)."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute

from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache

from .. import kda_bprop, kda_recompute, work_tables
from ..contract import get_dtype, op_dtype_name
from ..work_tables import WORK_ITEM_FIELDS
from .tvm_ffi import (
    make_compact_signature_tensor,
    make_dynamic_signature_tensor,
)

B_T = kda_recompute.B_T


@cute.jit
def warmup_backward_host(
    split: cutlass.Constexpr,
    recompute_op: cutlass.Constexpr,
    bprop_op: cutlass.Constexpr,
    n_heads_out: cutlass.Int32,
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    n_scan_ctas: cutlass.Int32,
    n_scan_blocks: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    scale: cutlass.Float32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    do: cute.Tensor,
    dq: cute.Tensor,
    dk: cute.Tensor,
    dv: cute.Tensor,
    gate: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    checkpoints: cute.Tensor,
    state_in: cute.Tensor | None,
    dgate: cute.Tensor,
    dbeta: cute.Tensor,
    dstate0: cute.Tensor | None,
    dstate_in: cute.Tensor | None,
    work_items: cute.Tensor,
    work_count: cute.Tensor,
    item_scratch: cute.Tensor | None,
    chunk_scratch: cute.Tensor | None,
    scheduler_all: cute.Tensor,
    scheduler_recompute: cute.Tensor,
    scheduler_bwd: cute.Tensor,
    tma_slots: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    """``split`` is the split table's :class:`~work_tables.SplitStatic` (None: the uncut
    table); the eight scalars after the ops are its runtime values (zeros when uncut)."""
    heads_out = cutlass.Int32(gate.shape[1])
    if cutlass.const_expr(split is not None):
        work_tables.launch_split_table(
            split,
            n_heads_out,
            n_tiles,
            ideal_chunks,
            batch_size,
            log2_thresh,
            n_scan_ctas,
            n_scan_blocks,
            n_walk_ctas,
            gate,
            cu_seqlens,
            chunk_scratch,
            item_scratch,
            work_count,
            stream,
        )
    work_tables.launch_work_table(
        B_T,
        heads_out,
        cutlass.Int32(gate.shape[0]),
        cu_seqlens,
        item_scratch,
        work_count,
        work_items,
        scheduler_all,
        stream,
        tail_rows=dgate,
    )
    recompute_op.launch(
        k,
        v,
        gate,
        beta,
        state_in,
        checkpoints,
        work_items,
        work_count,
        scheduler_recompute,
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
        dstate_in,
        dstate0,
        work_items,
        work_count,
        scheduler_bwd,
        tma_slots,
        scale,
        stream,
    )


@jit_cache
def _compile_warmup_backward(
    split: work_tables.SplitStatic | None,
    num_sms: int,
    io_dtype,
    gate_dtype,
    beta_dtype,
    state_dtype,
    cu_seqlens_dtype,
    d_k: int,
    d_v: int,
    use_initial_state: bool,
    has_state_in: bool,
    has_dstate0: bool,
    has_dstate_in: bool,
    use_int64_offsets: bool,
):
    """Compile the warmup / uncut backward host over fake tensors with dynamic extents, int64
    outer strides under ``use_int64_offsets`` and the last mode contiguous."""
    sym_int = cute.sym_int

    def strided(dtype, rank, align):
        return make_dynamic_signature_tensor(
            dtype, rank, assumed_align=align, use_int64_offsets=use_int64_offsets
        )

    def table(align):
        return make_compact_signature_tensor(
            cutlass.Int32, (sym_int(), WORK_ITEM_FIELDS), assumed_align=align
        )

    io, gate_name = op_dtype_name(io_dtype), op_dtype_name(gate_dtype)
    recompute_op = kda_recompute.KdaRecompute(io, gate_name, d_k, d_v, has_state_in, num_sms)
    bprop_op = kda_bprop.KdaBprop(
        io, gate_name, d_k, d_v, use_initial_state, has_dstate_in, has_dstate0, num_sms
    )
    static = (
        num_sms,
        d_k,
        d_v,
        use_initial_state,
        has_state_in,
        has_dstate0,
        has_dstate_in,
        use_int64_offsets,
    )
    dtype_names = "_".join(
        "none" if dtype is None else dtype.__name__.lower()
        for dtype in (io_dtype, gate_dtype, beta_dtype, state_dtype, cu_seqlens_dtype)
    )
    split_tag = "none" if split is None else "_".join(str(v) for v in split)
    name = (
        "kda_warmup_backward_"
        + "_".join(str(int(flag)) for flag in static)
        + f"_{dtype_names}_split_{split_tag}"
    )
    return compile_tvm_ffi(
        warmup_backward_host,
        split,
        recompute_op,
        bprop_op,
        *(cutlass.Int32(0) for _ in range(4)),
        cutlass.Float32(0),
        *(cutlass.Int32(0) for _ in range(3)),
        cutlass.Float32(0),
        strided(io_dtype, 3, 16),  # q
        strided(io_dtype, 3, 16),  # k
        strided(io_dtype, 3, 16),  # v
        strided(io_dtype, 3, 16),  # do
        strided(io_dtype, 3, 16),  # dq
        strided(io_dtype, 3, 16),  # dk
        strided(io_dtype, 3, 16),  # dv
        strided(gate_dtype, 3, 16),  # gate
        strided(beta_dtype, 2, 4),  # beta
        strided(cu_seqlens_dtype, 1, 8 if cu_seqlens_dtype is cutlass.Int64 else 4),  # cu_seqlens
        strided(io_dtype, 4, 16),  # checkpoints
        strided(state_dtype, 4, 16) if has_state_in else None,  # state_in
        strided(gate_dtype, 3, 16),  # dgate
        strided(beta_dtype, 2, 4),  # dbeta
        strided(cutlass.Float32, 4, 16) if has_dstate0 else None,  # dstate0
        strided(cutlass.Float32, 4, 16) if has_dstate_in else None,  # dstate_in
        table(16),  # work_items
        strided(cutlass.Int32, 1, 4),  # work_count
        table(16) if split is not None else None,  # item_scratch
        strided(cutlass.Float32, 2, 4) if split is not None else None,  # chunk_scratch
        strided(cutlass.Int32, 1, 4),  # scheduler_all
        strided(cutlass.Int32, 1, 4),  # scheduler_recompute
        strided(cutlass.Int32, 1, 4),  # scheduler_bwd
        make_compact_signature_tensor(cutlass.Int64, (sym_int(), 16), assumed_align=128),
        name=name,
        opt_level=2,
    )


def build_warmup_backward(
    *,
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
    state_in,
    use_initial_state,
    dgate,
    dbeta,
    dstate0,
    dstate_in,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler_all,
    scheduler_recompute,
    scheduler_bwd,
    tma_slots,
    split,
    n_tiles,
    ideal_chunks,
    num_sm,
    b_t,
    log_gate,
    scale,
):
    """Compile (persisted per static config) the warmup or uncut backward launch over the buffers
    of one plan; return ``(compiled, facts)`` with ``facts`` the split table's
    :class:`~work_tables.SplitTableFacts` (None for the uncut table)."""
    if not log_gate or b_t != B_T:
        raise ValueError("the warpdrive KDA backward serves 16-token chunks of natural-log gates")
    facts = None
    if split:
        facts = work_tables.split_table_facts(
            gate, cu_seqlens, n_tiles=n_tiles, ideal_chunks=ideal_chunks, num_sms=num_sm, b_t=b_t
        )
    tensors = (
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
        state_in,
        dgate,
        dbeta,
        dstate0,
        dstate_in,
        work_items,
        work_count,
        item_scratch,
        chunk_scratch,
        scheduler_all,
        scheduler_recompute,
        scheduler_bwd,
    )
    compiled = _compile_warmup_backward.by_args(
        None if facts is None else facts.static,
        int(num_sm),
        get_dtype(q.dtype),
        get_dtype(gate.dtype),
        get_dtype(beta.dtype),
        get_dtype(state_in.dtype) if state_in is not None else cutlass.Float32,
        cutlass.Int64 if str(cu_seqlens.dtype).endswith("int64") else cutlass.Int32,
        int(q.shape[2]),
        int(v.shape[2]),
        bool(use_initial_state),
        state_in is not None,
        dstate0 is not None,
        dstate_in is not None,
        requires_int64_abi(*(t for t in tensors if t is not None)),
    )
    return compiled, facts


def run_warmup_backward(
    compiled,
    facts,
    *,
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
    state_in,
    dgate,
    dbeta,
    dstate0,
    dstate_in,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler_all,
    scheduler_recompute,
    scheduler_bwd,
    tma_slots,
    scale,
) -> None:
    """Replay the warmup or uncut backward: one crossing into the DSL on the current Torch stream.
    The plan validated the contract at build, so nothing here raises."""
    split = facts is not None
    compiled(
        *(facts.scalars() if split else (0, 0, 0, 0, 0.0, 0, 0, 0)),
        float(scale),
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
        state_in,
        dgate,
        dbeta,
        dstate0,
        dstate_in,
        work_items,
        work_count,
        item_scratch if split else None,
        chunk_scratch if split else None,
        scheduler_all,
        scheduler_recompute,
        scheduler_bwd,
        tma_slots,
    )
