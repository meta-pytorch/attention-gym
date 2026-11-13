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
# (python/cudnn/linear_attention/frost/kernel/kda_warmup_forward_f16.py); the host nests the prep,
# prefill, and prep-prefill Ops and compiles through a @jit_cache function over fake TVM-FFI
# signatures (+int64 variants), launching live tensors on the current Torch stream (unused
# device/stream arguments dropped); launch-contract validation of the plan buffers; optional
# paged-state routes through the prologue compaction and the prefill; upstream-only constexpr knobs
# pruned (safe_gate, A_log/dt_bias, beta sigmoid, allow_neg_eigval, Q/K L2 norm, expand_num, the
# log2_threshold override); Ruff formatting. The split table, the work-table prologue (with the
# prep's row table), the prep and the prefills run as CuTeDSL 4.6.2 kernels, so
# no descriptor workspace is built.

"""One compiled launch for the KDA warmup and uncut forwards: the split-K table (plan, scan and
walk, warmup only), the work-table prologue (with the prep's chunk row table), the shared prep and
the prefill issued from a single host, so the Python side crosses into the DSL once per call.
Every kernel is a warpdrive kernel (``work_tables``, ``kda_prep``, ``kda_prefill`` /
``kda_prep_prefill``) addressing every sequence through ``ragged`` descriptors."""

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute

from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache

from .. import kda_prefill, kda_prep, kda_prep_prefill, work_tables
from ..contract import (
    get_dtype,
    memoize_launch_check,
    op_dtype_name,
    validate_cuda_tensors,
    validate_kda_forward_operands,
    validate_kda_forward_states,
    validate_tensor,
    validate_work_table,
)
from ..kda_common import B_T
from ..work_tables import WORK_ITEM_FIELDS
from .tvm_ffi import (
    make_compact_signature_tensor,
    make_dynamic_signature_tensor,
    make_paged_route_signatures,
)


@cute.jit
def warmup_forward_host(
    split: cutlass.Constexpr,
    prefill_op: cutlass.Constexpr,
    prep_op: cutlass.Constexpr,
    tiles_per_head: cutlass.Constexpr[int],
    num_sms: cutlass.Constexpr[int],
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
    gate: cute.Tensor,
    beta: cute.Tensor,
    o: cute.Tensor,
    cu_seqlens: cute.Tensor,
    state_in: cute.Tensor | None,
    state_out: cute.Tensor | None,
    state_indices: cute.Tensor | None,
    has_initial_state: cute.Tensor | None,
    work_items: cute.Tensor,
    work_count: cute.Tensor,
    item_scratch: cute.Tensor | None,
    chunk_scratch: cute.Tensor | None,
    scheduler: cute.Tensor,
    prep_k_decay: cute.Tensor | None,
    prep_q_decay: cute.Tensor | None,
    prep_t: cute.Tensor | None,
    prep_a: cute.Tensor | None,
    prep_diag: cute.Tensor | None,
    prep_rows: cute.Tensor | None,
    prep_row_count: cute.Tensor | None,
    tma_slots: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    """``split`` is the split table's :class:`~work_tables.SplitStatic` (None: the uncut
    table); the eight scalars after ``num_sms`` are its runtime values (zeros when uncut).
    ``prep_op`` (None without the shared prep) runs the prep ahead of the prep-fed prefill."""
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
    # Up to one item per CTA runs unsorted (num_ctas): the prefills claim one item per CTA there.
    work_tables.launch_work_table(
        B_T,
        cutlass.Int32(gate.shape[1]) * cutlass.Int32(tiles_per_head),
        cutlass.Int32(gate.shape[0]),
        cu_seqlens,
        item_scratch,
        work_count,
        work_items,
        scheduler,
        stream,
        num_ctas=num_sms,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        rows=prep_rows,
        row_count=prep_row_count,
        rows_b_t=B_T if cutlass.const_expr(prep_op is not None) else 0,
    )
    if cutlass.const_expr(prep_op is not None):
        prep_op.launch(
            q,
            k,
            gate,
            beta,
            prep_k_decay,
            prep_q_decay,
            prep_t,
            prep_a,
            prep_diag,
            prep_rows,
            prep_row_count,
            tma_slots,
            stream,
        )
        prefill_op.launch(
            v,
            prep_k_decay,
            prep_q_decay,
            prep_t,
            prep_a,
            prep_diag,
            state_in,
            o,
            state_out,
            state_indices,
            has_initial_state,
            work_items,
            work_count,
            scheduler,
            tma_slots,
            scale,
            stream,
        )
    else:
        prefill_op.launch(
            q,
            k,
            v,
            gate,
            beta,
            state_in,
            o,
            state_out,
            state_indices,
            has_initial_state,
            work_items,
            work_count,
            scheduler,
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


def _compact(dtype, tail: tuple, align: int, use_int64_offsets: bool):
    """The legacy ``mark_compact_shape_dynamic(mode=0, ...)`` placeholder: row-major, only the
    leading mode dynamic."""
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    return make_compact_signature_tensor(dtype, (sym_int(), *tail), assumed_align=align)


@jit_cache
def _compile_warmup_forward(
    split: work_tables.SplitStatic | None,
    num_sms: int,
    io_dtype,
    gate_dtype,
    beta_dtype,
    cu_seqlens_dtype,
    state_in_dtype,
    state_out_dtype,
    d_k: int,
    d_v: int,
    tiles_per_head: int,
    prep: bool,
    use_int64_offsets: bool,
    paged_state: int = 0,
    fastmath: bool = True,
):
    """Compile the warmup / uncut forward host for one static config over fake tensors that repeat
    the standalone builds' placeholders; absent tensors have a None dtype. ``paged_state`` is 0
    without paged routing, 1 with per-sequence routes and 2 with routes plus the fresh-slot byte
    mask."""
    i64 = use_int64_offsets
    io, gate_name = op_dtype_name(io_dtype), op_dtype_name(gate_dtype)
    states = (state_in_dtype is not None, state_out_dtype is not None, paged_state, num_sms)
    prep_op = None
    prep_signatures = [None] * 7
    if prep:
        if d_v // tiles_per_head != kda_prep_prefill.D_V:
            raise ValueError("the KDA prep prefill runs 64-column value tiles")
        prefill_op = kda_prep_prefill.KdaPrepPrefill(io, d_k, tiles_per_head, *states)
        prep_op = kda_prep.KdaPrep(io, gate_name, d_k, num_sms, fastmath)
    else:
        prefill_op = kda_prefill.KdaPrefill(
            io, gate_name, d_k, d_v // tiles_per_head, tiles_per_head, *states, fastmath
        )
    if prep:
        prep_signatures = [
            _dynamic(io_dtype, 4, 128, i64),  # prep_k_decay
            _dynamic(io_dtype, 4, 128, i64),  # prep_q_decay
            _dynamic(io_dtype, 4, 128, i64),  # prep_t
            _dynamic(cutlass.Int32, 3, 16, i64),  # prep_a
            _dynamic(cutlass.Float32, 3, 16, i64),  # prep_diag
            _dynamic(cutlass.Int32, 2, 16, i64),  # prep_rows
            _dynamic(cutlass.Int32, 1, 4, i64),  # prep_row_count
        ]
    static = (
        num_sms,
        d_k,
        d_v,
        tiles_per_head,
        state_in_dtype is not None,
        state_out_dtype is not None,
        prep,
        i64,
        paged_state,
    )
    dtype_names = "_".join(
        "none" if dtype is None else dtype.__name__.lower()
        for dtype in (
            io_dtype,
            gate_dtype,
            beta_dtype,
            cu_seqlens_dtype,
            state_in_dtype,
            state_out_dtype,
        )
    )
    split_tag = "none" if split is None else "_".join(str(v) for v in split)
    name = (
        "kda_warmup_forward_"
        + "_".join(str(int(flag)) for flag in static)
        + f"_split_{split_tag}_{dtype_names}"
        + ("" if fastmath else "_nofm")
    )
    return compile_tvm_ffi(
        warmup_forward_host,
        split,
        prefill_op,
        prep_op,
        tiles_per_head,
        num_sms,
        *(cutlass.Int32(0) for _ in range(4)),  # n_heads_out, n_tiles, ideal_chunks, batch_size
        cutlass.Float32(0),  # log2_thresh
        *(cutlass.Int32(0) for _ in range(3)),  # n_scan_ctas, n_scan_blocks, n_walk_ctas
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
        _dynamic(state_in_dtype, 4, 16, i64) if state_in_dtype is not None else None,
        _dynamic(state_out_dtype, 4, 16, i64) if state_out_dtype is not None else None,
        *(
            make_paged_route_signatures(cute.sym_int(), has_initial_state=paged_state == 2)
            if paged_state
            else (None, None)
        ),
        _compact(cutlass.Int32, (WORK_ITEM_FIELDS,), 16, i64),  # work_items
        _dynamic(cutlass.Int32, 1, 4, i64),  # work_count
        _compact(cutlass.Int32, (WORK_ITEM_FIELDS,), 16, i64) if split else None,  # item_scratch
        _dynamic(cutlass.Float32, 2, 4, i64) if split else None,  # chunk_scratch
        _dynamic(cutlass.Int32, 1, 4, i64),  # scheduler
        *prep_signatures,
        _compact(cutlass.Int64, (16,), 128, i64),  # tma_slots
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
    state_in,
    state_out,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler,
    split,
    n_tiles,
    ideal_chunks,
    num_sm,
    b_t,
    tiles_per_head,
    prep,
    prep_k_decay,
    prep_q_decay,
    prep_t,
    prep_a,
    prep_diag,
    prep_rows,
    prep_row_count,
    state_indices,
):
    """Check the warmup / uncut forward buffers of one plan (``n_tiles`` = sequences x heads x
    value tiles)."""
    if b_t != B_T:
        raise ValueError(f"the KDA forward runs b_t={B_T} chunks, got {b_t}")
    tokens, heads_out, num_seqs = validate_kda_forward_operands(q, k, v, gate, beta, o, cu_seqlens)
    validate_kda_forward_states(
        q,
        v,
        heads_out,
        num_seqs,
        state_in=(state_in, state_indices is not None),
        state_out=(state_out, state_indices is not None),
    )
    validate_cuda_tensors(
        q,
        work_items=work_items,
        work_count=work_count,
        item_scratch=item_scratch,
        chunk_scratch=chunk_scratch,
        scheduler=scheduler,
        state_indices=state_indices,
    )
    dim_v = v.shape[2]
    if (
        num_sm < 1
        or tiles_per_head < 1
        or dim_v % tiles_per_head
        or (split and tiles_per_head != 1)
    ):
        raise ValueError(
            "num_sm must be positive and tiles_per_head must divide d_v (1 under the split table)"
        )
    if n_tiles != num_seqs * heads_out * tiles_per_head:
        raise ValueError("n_tiles must equal sequences x output heads x tiles_per_head")
    if state_indices is not None:
        validate_tensor(
            "state_indices", state_indices, (num_seqs,), ("int32",), align=4, compact=True
        )
    table_rows = n_tiles
    if split:
        if ideal_chunks:
            table_rows = work_tables.max_work_items(
                tokens, num_seqs, heads_out, ideal_chunks, b_t, num_sm
            )
        validate_tensor(
            "item_scratch",
            item_scratch,
            (None, WORK_ITEM_FIELDS),
            ("int32",),
            compact=True,
            min_rows=table_rows,
        )
        validate_tensor(
            "chunk_scratch",
            chunk_scratch,
            (None, heads_out),
            ("float32",),
            align=4,
            min_rows=work_tables.chunk_scratch_rows(tokens, num_seqs, b_t),
        )
    validate_work_table(work_items, work_count, scheduler, min_rows=table_rows)
    if prep:
        validate_cuda_tensors(
            q,
            prep_k_decay=prep_k_decay,
            prep_q_decay=prep_q_decay,
            prep_t=prep_t,
            prep_a=prep_a,
            prep_diag=prep_diag,
            prep_rows=prep_rows,
            prep_row_count=prep_row_count,
        )
        rows, dim_k, io = (
            tokens // b_t + num_seqs,
            q.shape[2],
            (str(q.dtype).removeprefix("torch."),),
        )
        for name, tensor in (
            ("prep_k_decay", prep_k_decay),
            ("prep_q_decay", prep_q_decay),
            ("prep_t", prep_t),
        ):
            validate_tensor(
                name, tensor, (None, heads_out, b_t, dim_k), io, tma=True, min_rows=rows
            )
        validate_tensor(
            "prep_a",
            prep_a,
            (None, heads_out, b_t * b_t // 2),
            ("int32",),
            tma=True,
            min_rows=rows,
        )
        validate_tensor(
            "prep_diag", prep_diag, (None, heads_out, dim_k), ("float32",), tma=True, min_rows=rows
        )
        validate_tensor("prep_rows", prep_rows, (None, 4), ("int32",), compact=True, min_rows=rows)
        validate_tensor("prep_row_count", prep_row_count, (None,), ("int32",), align=4, min_rows=1)


def build_warmup_forward(
    *,
    q,
    k,
    v,
    gate,
    beta,
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
    tma_slots,
    split,
    n_tiles,
    ideal_chunks,
    num_sm,
    b_t,
    log_gate,
    checkpoint_every_n_tokens,
    scale,
    tiles_per_head=1,
    prep=False,
    prep_k_decay=None,
    prep_q_decay=None,
    prep_t=None,
    prep_a=None,
    prep_diag=None,
    prep_rows=None,
    prep_row_count=None,
    state_indices=None,
    has_initial_state=None,
    fastmath=True,
):
    """Compile (persisted per static config: dtypes, dims, the split-K
    geometry, the d_v split, state presence, the int64 ABI) the warmup or uncut forward launch over
    the buffers of one plan; return ``(compiled, facts)`` with ``facts`` the split table's
    :class:`~work_tables.SplitTableFacts` (None for the uncut table). It launches on the
    current Torch stream. ``state_indices`` (int32 per sequence) and the optional uint8
    ``has_initial_state`` select paged state: ``state_in`` and ``state_out`` are then one pool
    routed per sequence (null, fresh and resumed slots). The warpdrive KDA prefills route state
    through ``state_indices`` only and write no checkpoints. ``fastmath=False`` selects the
    non-flushing gate exponentials."""
    if not log_gate:
        raise ValueError("the warpdrive KDA prefills take natural-log gates")
    if seed_indices is not None or final_indices is not None:
        raise ValueError("the warpdrive KDA prefills route state through state_indices only")
    if int(checkpoint_every_n_tokens) > 0:
        raise ValueError("the warpdrive KDA prefills write no checkpoints")
    # Attention Gym modification: validate the launch contract before selecting a compiled ABI.
    _validate_launch(
        q=q,
        k=k,
        v=v,
        gate=gate,
        beta=beta,
        o=o,
        cu_seqlens=cu_seqlens,
        state_in=state_in,
        state_out=state_out,
        work_items=work_items,
        work_count=work_count,
        item_scratch=item_scratch,
        chunk_scratch=chunk_scratch,
        scheduler=scheduler,
        split=split,
        n_tiles=n_tiles,
        ideal_chunks=ideal_chunks,
        num_sm=num_sm,
        b_t=b_t,
        tiles_per_head=tiles_per_head,
        prep=prep,
        prep_k_decay=prep_k_decay,
        prep_q_decay=prep_q_decay,
        prep_t=prep_t,
        prep_a=prep_a,
        prep_diag=prep_diag,
        prep_rows=prep_rows,
        prep_row_count=prep_row_count,
        state_indices=state_indices,
    )
    facts = None
    if split:
        facts = work_tables.split_table_facts(
            gate, cu_seqlens, n_tiles=n_tiles, ideal_chunks=ideal_chunks, num_sms=num_sm, b_t=b_t
        )
    tensors = (
        q,
        k,
        v,
        gate,
        beta,
        o,
        cu_seqlens,
        state_in,
        state_out,
        work_items,
        work_count,
        item_scratch,
        chunk_scratch,
        scheduler,
        prep_k_decay,
        prep_q_decay,
        prep_t,
        prep_a,
        prep_diag,
        prep_rows,
        prep_row_count,
        state_indices,
        has_initial_state,
    )
    if state_indices is not None:
        if state_in is None or state_in is not state_out:
            raise ValueError("paged state routes one aliased state pool")
        if has_initial_state is not None and (
            has_initial_state.dtype != torch.uint8 or not has_initial_state.is_contiguous()
        ):
            raise ValueError("has_initial_state must be a contiguous uint8 mask")
    elif has_initial_state is not None:
        raise ValueError("has_initial_state requires state_indices")
    paged_state = 0 if state_indices is None else 1 + (has_initial_state is not None)
    use_int64_offsets = requires_int64_abi(*tensors)
    compiled = _compile_warmup_forward.by_args(
        None if facts is None else facts.static,
        int(num_sm),
        get_dtype(q.dtype),
        get_dtype(gate.dtype),
        get_dtype(beta.dtype),
        cutlass.Int64 if cu_seqlens.dtype == torch.int64 else cutlass.Int32,
        _dtype_or_none(state_in),
        _dtype_or_none(state_out),
        int(q.shape[2]),
        int(v.shape[2]),
        int(tiles_per_head),
        bool(prep),
        use_int64_offsets,
        paged_state,
        bool(fastmath),
    )
    return compiled, facts


def run_warmup_forward(
    compiled,
    facts,
    *,
    q,
    k,
    v,
    gate,
    beta,
    o,
    cu_seqlens,
    state_in,
    state_out,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler,
    tma_slots,
    scale,
    prep_k_decay=None,
    prep_q_decay=None,
    prep_t=None,
    prep_a=None,
    prep_diag=None,
    prep_rows=None,
    prep_row_count=None,
    state_indices=None,
    has_initial_state=None,
    seed_indices=None,
    final_indices=None,
    checkpoints=None,
    checkpoint_every_n_tokens=0,
) -> None:
    """Replay the warmup or uncut forward: one crossing into the DSL for the table, prologue,
    prep and prefill launches, on the current Torch stream. The plan validated the contract at
    build, so nothing here raises. The trailing keywords only shape the build; they are accepted
    so a caller replays with the build's buffers."""
    split = facts is not None
    compiled(
        *(facts.scalars() if split else (0, 0, 0, 0, 0.0, 0, 0, 0)),
        float(scale),
        q,
        k,
        v,
        gate,
        beta,
        o,
        cu_seqlens,
        state_in,
        state_out,
        state_indices,
        has_initial_state,
        work_items,
        work_count,
        item_scratch if split else None,
        chunk_scratch if split else None,
        scheduler,
        prep_k_decay,
        prep_q_decay,
        prep_t,
        prep_a,
        prep_diag,
        prep_rows,
        prep_row_count,
        tma_slots,
    )
