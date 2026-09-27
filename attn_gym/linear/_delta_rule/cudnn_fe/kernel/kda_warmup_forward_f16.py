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
# attn_gym.linear._delta_rule.cudnn_fe. The host nests the prep / prefill Ops, compiles through a
# persisted jit_cache function over fake TVM-FFI signatures (int64 ABI variant when a tensor needs
# it), and launches with live tensors on the current Torch stream. Optional paged routes reach the
# prologue's compaction and the prefill. The always-pinned safe_gate, a_log, dt_bias, expand_num,
# beta-sigmoid, allow_neg_eigval and Q/K L2-norm knobs are removed.

"""One compiled launch for the KDA warmup and uncut forwards: the split-K table (plan, scan and
walk, warmup only), the prefill prologue and the prefill issued from a single host.  Every kernel,
its host and the tensor placeholder each host was compiled with are the standalone modules' own;
this host only sequences the launches, so the kernels' SASS is unchanged and the Python side
crosses into the DSL once per call instead of two or three times.  A buffer that two hosts read
through different signature types is passed twice, once per type: the table marks the gate at its
element alignment along its last mode, work_items, work_count and item_scratch as 4-byte compact
views, cu_seqlens at 4 bytes; the prologue and prefill mark the same
buffers as the standalone prefill wrapper does."""

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass import cute

from attn_gym._backends.cute import compile_tvm_ffi, jit_cache
from attn_gym._backends.cute.utils import requires_int64_abi

from ..common import split_k
from ..common.host import get_dtype, validate_cuda_tensors
from ..common.launch import validate_tensor, validate_workspace
from ..common.tvm_ffi import (
    WORK_ITEM_FIELDS,
    make_compact_signature_tensor,
    make_dynamic_signature_tensor,
    make_paged_route_signatures,
)
from . import kda_prefill_f16, kda_prep_f16, kda_prep_prefill_f16


@cute.jit
def warmup_forward_host(
    split: cutlass.Constexpr[bool],
    b_t: cutlass.Constexpr[int],
    scan_rows: cutlass.Constexpr[int],
    log_gate: cutlass.Constexpr[bool],
    gate_channels: cutlass.Constexpr[int],
    overhead_chunks: cutlass.Constexpr[int],
    warmup_cap: cutlass.Constexpr[int],
    full_scan: cutlass.Constexpr[bool],
    n_heads_out: cutlass.Int32,
    num_sms: cutlass.Constexpr[int],
    io_dtype: cutlass.Constexpr,
    order_gen: cutlass.Constexpr[bool],
    prefill_op: cutlass.Constexpr,
    tiles_per_head: cutlass.Constexpr[int],
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    gate_scale_log2: cutlass.Float32,
    n_scan_ctas: cutlass.Int32,
    n_scan_blocks: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    scale: cutlass.Float32,
    q: cute.Tensor,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    gate_table: cute.Tensor | None,
    beta: cute.Tensor,
    o: cute.Tensor,
    cu_seqlens: cute.Tensor,
    cu_seqlens_table: cute.Tensor,
    state_in: cute.Tensor | None,
    state_out: cute.Tensor | None,
    seed_indices: cute.Tensor | None,
    final_indices: cute.Tensor | None,
    state_indices: cute.Tensor | None,
    has_initial_state: cute.Tensor | None,
    checkpoints: cute.Tensor | None,
    work_items: cute.Tensor,
    work_items_table: cute.Tensor,
    work_count: cute.Tensor,
    work_count_table: cute.Tensor,
    staging: cute.Tensor | None,
    item_scratch: cute.Tensor | None,
    chunk_scratch: cute.Tensor | None,
    scheduler: cute.Tensor,
    workspace: cute.Tensor,
    prep: cutlass.Constexpr[bool],
    prep_op: cutlass.Constexpr,
    prep_k_decay: cute.Tensor | None,
    prep_q_decay: cute.Tensor | None,
    prep_t: cute.Tensor | None,
    prep_a: cute.Tensor | None,
    prep_diag: cute.Tensor | None,
    prep_words: cute.Tensor | None,
    prep_rows: cute.Tensor | None,
    prep_row_count: cute.Tensor | None,
    stream: cuda.CUstream,
) -> None:
    if cutlass.const_expr(split):
        split_k.launch(
            split,
            b_t,
            scan_rows,
            log_gate,
            gate_channels,
            overhead_chunks,
            warmup_cap,
            full_scan,
            n_heads_out,
            num_sms,
            n_tiles,
            ideal_chunks,
            batch_size,
            log2_thresh,
            gate_scale_log2,
            gate_table,
            cu_seqlens_table,
            chunk_scratch,
            item_scratch,
            work_items_table,
            work_count_table,
            scheduler,
            n_scan_ctas,
            n_scan_blocks,
            n_walk_ctas,
            stream,
        )
    if cutlass.const_expr(prep):
        kda_prep_prefill_f16.prologue(
            io_dtype,
            b_t,
            num_sms,
            order_gen,
            q,
            k,
            v,
            gate,
            o,
            checkpoints,
            cu_seqlens,
            staging,
            work_count,
            work_items,
            scheduler,
            workspace,
            checkpoint_every_n,
            stream,
            tiles_per_head,
            prep_k_decay,
            prep_q_decay,
            prep_t,
            prep_a,
            prep_diag,
            prep_op.cfg,
            prep_words,
            prep_rows,
            prep_row_count,
            state_indices,
            has_initial_state,
        )
        prep_op(
            q,
            k,
            prep_words,
            gate,
            beta,
            cu_seqlens,
            prep_k_decay,
            prep_q_decay,
            prep_t,
            prep_a,
            prep_diag,
            prep_rows,
            prep_row_count,
            stream,
        )
    else:
        kda_prefill_f16.prologue(
            io_dtype,
            b_t,
            num_sms,
            order_gen,
            q,
            k,
            v,
            gate,
            o,
            checkpoints,
            cu_seqlens,
            staging,
            work_count,
            work_items,
            scheduler,
            workspace,
            checkpoint_every_n,
            stream,
            tiles_per_head,
            state_indices,
            has_initial_state,
        )
    prefill_op(
        q,
        k,
        v,
        gate,
        beta,
        cu_seqlens,
        state_in,
        o,
        state_out,
        seed_indices,
        final_indices,
        state_indices,
        has_initial_state,
        work_items,
        work_count,
        scheduler,
        workspace,
        checkpoint_every_n,
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
    facts_static: tuple,
    io_dtype,
    gate_dtype,
    beta_dtype,
    cu_seqlens_dtype,
    state_in_dtype,
    state_out_dtype,
    checkpoint_dtype,
    d_k: int,
    d_v: int,
    tiles_per_head: int,
    has_seed_indices: bool,
    has_final_indices: bool,
    gate_scale_log2: float,
    prep: bool,
    use_int64_offsets: bool,
    paged_state: int = 0,
):
    """Compile the warmup / uncut forward host for one static config over fake tensors that repeat
    the standalone builds' placeholders.  ``facts_static`` is the split table's constexpr facts
    (split, b_t, scan_rows, log_gate, gate_channels, overhead_chunks, warmup_cap, full_scan,
    num_sms); absent tensors have a None dtype.  ``paged_state`` is 0 without paged routing, 1 with
    per-sequence routes and 2 with routes plus the fresh-slot byte mask."""
    (
        split,
        b_t,
        scan_rows,
        log_gate,
        gate_channels,
        overhead_chunks,
        warmup_cap,
        full_scan,
        num_sms,
    ) = facts_static
    i64 = use_int64_offsets
    state_dtype = cutlass.Float32
    if state_out_dtype is not None:
        state_dtype = state_out_dtype
    if state_in_dtype is not None:
        state_dtype = state_in_dtype
    flags = {"gate_scale_log2": gate_scale_log2, "log_gate": log_gate, "d_k": d_k}
    prefill_module = kda_prep_prefill_f16 if prep else kda_prefill_f16
    prefill_cfg = prefill_module.build_cfg(
        io_dtype,
        state_dtype,
        gate_dtype,
        use_initial_state=state_in_dtype is not None,
        store_final_state=state_out_dtype is not None,
        enable_checkpoints=checkpoint_dtype is not None,
        max_active_clusters=num_sms,
        d_v=d_v // tiles_per_head,
        tiles_per_head=tiles_per_head,
        paged_state=paged_state,
        **flags,
    )
    prefill_op_type = (
        kda_prep_prefill_f16.KdaPrepPrefillOp if prep else kda_prefill_f16.KdaPrefillOp
    )
    prefill_op = prefill_op_type(prefill_cfg, i64)
    prep_op = None
    prep_signatures = [None] * 8
    if prep:
        prep_op = kda_prep_f16.KdaPrepOp(
            kda_prep_f16.build_cfg(io_dtype, gate_dtype, num_sm=num_sms, **flags), i64
        )
        prep_signatures = [
            _dynamic(io_dtype, 4, 128, i64),  # prep_k_decay
            _dynamic(io_dtype, 4, 128, i64),  # prep_q_decay
            _dynamic(io_dtype, 4, 128, i64),  # prep_t
            _dynamic(cutlass.Int32, 3, 16, i64),  # prep_a
            _dynamic(cutlass.Float32, 3, 16, i64),  # prep_diag
            _dynamic(cutlass.Int64, 1, 128, i64),  # prep_words
            _dynamic(cutlass.Int32, 2, 16, i64),  # prep_rows
            _dynamic(cutlass.Int32, 1, 4, i64),  # prep_row_count
        ]
    static = (
        *facts_static,
        d_k,
        d_v,
        tiles_per_head,
        state_in_dtype is not None,
        state_out_dtype is not None,
        checkpoint_dtype is not None,
        has_seed_indices,
        has_final_indices,
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
            checkpoint_dtype,
        )
    )
    gate_tag = str(float(gate_scale_log2)).replace(".", "p").replace("-", "m").replace("+", "")
    name = (
        "kda_warmup_forward_"
        + "_".join(str(int(flag)) for flag in static)
        + f"_{dtype_names}_g{gate_tag}"
    )
    return compile_tvm_ffi(
        warmup_forward_host,
        split,
        b_t,
        scan_rows,
        log_gate,
        gate_channels,
        overhead_chunks,
        warmup_cap,
        full_scan,
        cutlass.Int32(0),  # n_heads_out
        num_sms,
        io_dtype,
        not split,  # order_gen
        prefill_op,
        tiles_per_head,
        *(cutlass.Int32(0) for _ in range(3)),  # n_tiles, ideal_chunks, batch_size
        cutlass.Float32(0),  # log2_thresh
        cutlass.Float32(0),  # gate_scale_log2
        *(
            cutlass.Int32(0) for _ in range(4)
        ),  # n_scan_ctas, n_scan_blocks, n_walk_ctas, checkpoint_every_n
        cutlass.Float32(0),  # scale
        _dynamic(io_dtype, 3, 16, i64),  # q
        _dynamic(io_dtype, 3, 16, i64),  # k
        _dynamic(io_dtype, 3, 16, i64),  # v
        _dynamic(gate_dtype, 3, 16, i64),  # gate
        _dynamic(gate_dtype, 3, 8 if gate_dtype.width == 16 else 4, i64)
        if split
        else None,  # gate_table
        _dynamic(beta_dtype, 2, 4, i64),  # beta
        _dynamic(io_dtype, 3, 16, i64),  # o
        _dynamic(
            cu_seqlens_dtype, 1, 8 if cu_seqlens_dtype is cutlass.Int64 else 4, i64
        ),  # cu_seqlens
        _dynamic(cu_seqlens_dtype, 1, 4, i64),  # cu_seqlens_table
        _dynamic(state_in_dtype, 4, 16, i64) if state_in_dtype is not None else None,
        _dynamic(state_out_dtype, 4, 16, i64) if state_out_dtype is not None else None,
        _dynamic(cutlass.Int32, 1, 4, i64) if has_seed_indices else None,
        _dynamic(cutlass.Int32, 1, 4, i64) if has_final_indices else None,
        *(
            make_paged_route_signatures(cute.sym_int(), has_initial_state=paged_state == 2)
            if paged_state
            else (None, None)
        ),
        _dynamic(checkpoint_dtype, 4, 16, i64) if checkpoint_dtype is not None else None,
        _compact(cutlass.Int32, (WORK_ITEM_FIELDS,), 16, i64),  # work_items
        _compact(cutlass.Int32, (WORK_ITEM_FIELDS,), 4, i64),  # work_items_table
        _dynamic(cutlass.Int32, 1, 4, i64),  # work_count
        _compact(cutlass.Int32, (), 4, i64),  # work_count_table
        _compact(cutlass.Int32, (WORK_ITEM_FIELDS,), 16, i64) if split else None,  # staging
        _compact(cutlass.Int32, (WORK_ITEM_FIELDS,), 4, i64) if split else None,  # item_scratch
        _dynamic(cutlass.Float32, 2, 4, i64) if split else None,  # chunk_scratch
        _dynamic(cutlass.Int32, 1, 4, i64),  # scheduler
        _dynamic(cutlass.Int64, 1, 128, i64),  # workspace
        prep,
        prep_op,
        *prep_signatures,
        name=name,
        opt_level=2,
    )


def _dtype_or_none(tensor):
    return None if tensor is None else get_dtype(tensor.dtype)


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
    seed_indices,
    final_indices,
    checkpoints,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler,
    workspace,
    split,
    n_tiles,
    ideal_chunks,
    num_sm,
    b_t,
    checkpoint_every_n_tokens,
    tiles_per_head,
    prep,
    prep_k_decay,
    prep_q_decay,
    prep_t,
    prep_a,
    prep_diag,
    prep_words,
    prep_rows,
    prep_row_count,
    state_indices,
):
    """Check the warmup / uncut forward buffers of one plan (``n_tiles`` = sequences x heads x
    value tiles)."""
    tokens, heads_out, num_seqs = kda_prefill_f16.validate_forward_operands(
        q, k, v, gate, beta, o, cu_seqlens, b_t=b_t
    )
    kda_prefill_f16.validate_forward_states(
        q,
        v,
        heads_out,
        num_seqs,
        state_in=(state_in, state_indices is not None or seed_indices is not None),
        state_out=(state_out, state_indices is not None or final_indices is not None),
    )
    kda_prefill_f16.validate_checkpoints(
        checkpoints, checkpoint_every_n_tokens, tokens, heads_out, num_seqs, q, v
    )
    validate_cuda_tensors(
        q,
        work_items=work_items,
        work_count=work_count,
        item_scratch=item_scratch,
        chunk_scratch=chunk_scratch,
        scheduler=scheduler,
        workspace=workspace,
        seed_indices=seed_indices,
        final_indices=final_indices,
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
    for name, table in (
        ("seed_indices", seed_indices),
        ("final_indices", final_indices),
        ("state_indices", state_indices),
    ):
        if table is not None:
            validate_tensor(name, table, (num_seqs,), ("int32",), align=4, compact=True)
    table_rows = n_tiles
    if split:
        if ideal_chunks:
            table_rows = split_k.max_work_items(
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
            min_rows=split_k.chunk_scratch_rows(tokens, num_seqs, b_t),
        )
    validate_tensor(
        "work_items",
        work_items,
        (None, WORK_ITEM_FIELDS),
        ("int32",),
        compact=True,
        min_rows=table_rows,
    )
    validate_tensor("work_count", work_count, (None,), ("int32",), align=4, min_rows=1)
    validate_tensor("scheduler", scheduler, (None,), ("int32",), align=4, min_rows=1)
    prefill = kda_prep_prefill_f16 if prep else kda_prefill_f16
    validate_workspace("workspace", workspace, prefill.TENSORMAP_DESC_ARRAYS, num_seqs)
    if prep:
        validate_cuda_tensors(
            q,
            prep_k_decay=prep_k_decay,
            prep_q_decay=prep_q_decay,
            prep_t=prep_t,
            prep_a=prep_a,
            prep_diag=prep_diag,
            prep_words=prep_words,
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
        validate_workspace("prep_words", prep_words, kda_prep_f16.TENSORMAP_DESC_ARRAYS, num_seqs)


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
    workspace,
    split,
    n_tiles,
    ideal_chunks,
    num_sm,
    b_t,
    log_gate,
    gate_lower_bound,
    checkpoint_every_n_tokens,
    scale,
    tiles_per_head=1,
    prep=False,
    prep_k_decay=None,
    prep_q_decay=None,
    prep_t=None,
    prep_a=None,
    prep_diag=None,
    prep_words=None,
    prep_rows=None,
    prep_row_count=None,
    state_indices=None,
    has_initial_state=None,
):
    """Compile (persisted per static config: dtypes, dims, gate flags and bound, the split-K
    geometry, the d_v split, state and checkpoint presence, the int64 ABI) the warmup or uncut
    forward launch over the buffers of one plan.  The fake signatures repeat the marks of the
    standalone split-table and prefill builds so every kernel compiles as it does there.
    It launches on the current Torch stream.  ``state_indices`` (int32 per sequence) and the
    optional uint8 ``has_initial_state`` select paged state: ``state_in`` and ``state_out`` are
    then one pool routed per sequence (null, fresh and resumed slots)."""
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
        seed_indices=seed_indices,
        final_indices=final_indices,
        checkpoints=checkpoints,
        work_items=work_items,
        work_count=work_count,
        item_scratch=item_scratch,
        chunk_scratch=chunk_scratch,
        scheduler=scheduler,
        workspace=workspace,
        split=split,
        n_tiles=n_tiles,
        ideal_chunks=ideal_chunks,
        num_sm=num_sm,
        b_t=b_t,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        tiles_per_head=tiles_per_head,
        prep=prep,
        prep_k_decay=prep_k_decay,
        prep_q_decay=prep_q_decay,
        prep_t=prep_t,
        prep_a=prep_a,
        prep_diag=prep_diag,
        prep_words=prep_words,
        prep_rows=prep_rows,
        prep_row_count=prep_row_count,
        state_indices=state_indices,
    )
    facts = split_k.split_table_facts(
        gate,
        cu_seqlens,
        split=split,
        n_tiles=n_tiles,
        ideal_chunks=ideal_chunks,
        num_sms=num_sm,
        b_t=b_t,
        log_gate=log_gate,
    )
    for name, tensor in (("seed_indices", seed_indices), ("final_indices", final_indices)):
        if tensor is not None and tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be int32, got {tensor.dtype}")
    if work_items.shape[1:] != (WORK_ITEM_FIELDS,) or (
        split and item_scratch.shape[1:] != (WORK_ITEM_FIELDS,)
    ):
        raise ValueError(f"work-item tables must be [rows, {WORK_ITEM_FIELDS}]")
    enable_checkpoints = int(checkpoint_every_n_tokens) > 0
    if not enable_checkpoints:
        checkpoints = None
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
        seed_indices,
        final_indices,
        checkpoints,
        work_items,
        work_count,
        item_scratch,
        chunk_scratch,
        scheduler,
        workspace,
        prep_k_decay,
        prep_q_decay,
        prep_t,
        prep_a,
        prep_diag,
        prep_words,
        prep_rows,
        prep_row_count,
        state_indices,
        has_initial_state,
    )
    if state_indices is not None:
        if (
            state_in is None
            or state_in is not state_out
            or seed_indices is not None
            or final_indices is not None
        ):
            raise ValueError(
                "paged state routes one aliased state pool without seed/final indices"
            )
        if state_indices.dtype != torch.int32 or not state_indices.is_contiguous():
            raise ValueError("state_indices must be contiguous int32")
        if has_initial_state is not None and (
            has_initial_state.dtype != torch.uint8 or not has_initial_state.is_contiguous()
        ):
            raise ValueError("has_initial_state must be a contiguous uint8 mask")
    elif has_initial_state is not None:
        raise ValueError("has_initial_state requires state_indices")
    paged_state = 0 if state_indices is None else 1 + (has_initial_state is not None)
    use_int64_offsets = requires_int64_abi(*tensors)
    compiled = _compile_warmup_forward(
        (
            bool(facts.split),
            int(facts.b_t),
            int(facts.scan_rows),
            bool(facts.log_gate),
            int(facts.gate_channels),
            int(facts.overhead_chunks),
            int(facts.warmup_cap),
            bool(facts.full_scan),
            int(facts.num_sms),
        ),
        get_dtype(q.dtype),
        get_dtype(gate.dtype),
        get_dtype(beta.dtype),
        cutlass.Int64 if cu_seqlens.dtype == torch.int64 else cutlass.Int32,
        _dtype_or_none(state_in),
        _dtype_or_none(state_out),
        _dtype_or_none(checkpoints),
        int(q.shape[2]),
        int(v.shape[2]),
        int(tiles_per_head),
        seed_indices is not None,
        final_indices is not None,
        float(gate_lower_bound) * kda_prefill_f16.LOG2_E,
        bool(prep),
        use_int64_offsets,
        paged_state,
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
    seed_indices,
    final_indices,
    checkpoints,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler,
    workspace,
    checkpoint_every_n_tokens,
    scale,
    prep_k_decay=None,
    prep_q_decay=None,
    prep_t=None,
    prep_a=None,
    prep_diag=None,
    prep_words=None,
    prep_rows=None,
    prep_row_count=None,
    state_indices=None,
    has_initial_state=None,
) -> None:
    """Replay the warmup or uncut forward: one crossing into the DSL for the table, prologue and
    prefill launches, on the current Torch stream.  The plan validated the contract at build, so
    nothing here raises."""
    compiled(
        facts.n_heads_out,
        facts.n_tiles,
        facts.ideal_chunks,
        facts.batch_size,
        facts.log2_threshold,
        facts.gate_scale_log2,
        facts.n_scan_ctas,
        facts.n_scan_blocks,
        facts.n_walk_ctas,
        int(checkpoint_every_n_tokens),
        float(scale),
        q,
        k,
        v,
        gate,
        gate if facts.split else None,
        beta,
        o,
        cu_seqlens,
        cu_seqlens,
        state_in,
        state_out,
        seed_indices,
        final_indices,
        state_indices,
        has_initial_state,
        checkpoints if int(checkpoint_every_n_tokens) > 0 else None,
        work_items,
        work_items,
        work_count,
        work_count,
        item_scratch if facts.split else None,
        item_scratch if facts.split else None,
        chunk_scratch if facts.split else None,
        scheduler,
        workspace,
        prep_k_decay,
        prep_q_decay,
        prep_t,
        prep_a,
        prep_diag,
        prep_words,
        prep_rows,
        prep_row_count,
    )
