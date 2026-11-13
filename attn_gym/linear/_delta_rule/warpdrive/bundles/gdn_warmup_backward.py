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
# (python/cudnn/linear_attention/frost/kernel/gdn_warmup_backward_f16.py); the launch compiles
# through jit_cache on fake-tensor TVM-FFI signatures (legacy placeholder ABI) with an int64-shape
# variant, after launch-contract validation; the unused device/stream and bprop_module arguments
# and the GDP d_v=64 bprop fork dropped; the upstream-only expand_num, safe_gate/A_log/dt_bias,
# beta-sigmoid, negative-eigenvalue, log2_threshold-override, and compact_qdo host paths removed at
# their pinned values; Ruff formatting. The head runs the CuTeDSL 4.6.2 T pass,
# split table, work-table prologue and recompute, so no per-batch descriptor workspace is built;
# the checkpoint-seeded coarse series, the bprop-side order and the stage switches the drivers
# never compile are gone.

"""One compiled launch for everything ahead of the bprop on the GDN warmup and uncut backward: the
T pass with its row table (``gdn_tinv``), the split-K table (warmup only), the work-table
prologue (``work_tables``: LPT order, zeroed scheduler counters of the recompute and the
bprop) and the checkpoint-series recompute (``gdn_recompute``), all at ``--opt-level 2``.
The bprop itself runs after it through ``chunk_gdn_bwd`` on the same work table. The warpdrive
kernels address every sequence through ``ragged`` descriptors: one per tensor, plus per-item slots
written in-kernel for sequences ending in the first box; no prologue builds descriptor arrays."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute

from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache

from .. import gdn_recompute, gdn_tinv, work_tables
from ..contract import (
    STATE_DTYPES,
    get_dtype,
    memoize_launch_check,
    validate_cuda_tensors,
    validate_gdn_bwd_bundle,
    validate_gdn_tinv,
    validate_seqlens,
    validate_tensor,
    validate_work_table,
)
from ..work_tables import WORK_ITEM_FIELDS
from .tvm_ffi import make_signature, signature_key, signature_spec


@cute.jit
def warmup_backward_host(
    split: cutlass.Constexpr,
    tinv_cfg: cutlass.Constexpr,
    recompute_cfg: cutlass.Constexpr,
    n_heads_out: cutlass.Int32,
    n_tiles: cutlass.Int32,
    ideal_chunks: cutlass.Int32,
    batch_size: cutlass.Int32,
    log2_thresh: cutlass.Float32,
    n_scan_ctas: cutlass.Int32,
    n_scan_blocks: cutlass.Int32,
    n_walk_ctas: cutlass.Int32,
    k: cute.Tensor,
    v: cute.Tensor,
    gate: cute.Tensor,
    beta: cute.Tensor,
    cu_seqlens: cute.Tensor,
    tinv: cute.Tensor,
    tinv_rows: cute.Tensor,
    tinv_row_count: cute.Tensor,
    checkpoints: cute.Tensor,
    state_in: cute.Tensor | None,
    work_items: cute.Tensor,
    work_count: cute.Tensor,
    item_scratch: cute.Tensor | None,
    chunk_scratch: cute.Tensor | None,
    scheduler_all: cute.Tensor,
    scheduler_recompute: cute.Tensor,
    tma_slots: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    """``split`` is the split table's :class:`~work_tables.SplitStatic` (None: the uncut
    table); the eight scalars after the configs are its runtime values (zeros when uncut).
    ``scheduler_all`` holds every consumer's claim counters (zeroed by the prologue),
    ``scheduler_recompute`` the recompute's."""
    tinv_op, tinv_sms = tinv_cfg
    tinv_op.launch(
        k,
        gate,
        beta,
        cu_seqlens,
        tinv,
        tinv_rows,
        tinv_row_count,
        tma_slots,
        cutlass.Int32(tinv_sms),
        stream,
        emit_row_table=True,
    )
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
        gdn_recompute.B_T,
        cutlass.Int32(gate.shape[1]),
        cu_seqlens,
        item_scratch,
        work_count,
        work_items,
        scheduler_all,
        stream,
    )
    recompute_op, recompute_sms = recompute_cfg
    recompute_op.launch(
        k,
        v,
        gate,
        tinv,
        work_items,
        work_count,
        scheduler_recompute,
        tma_slots,
        state_in,
        checkpoints,
        cutlass.Int32(recompute_sms),
        stream,
    )


@jit_cache
def _compile_warmup_backward(
    split: work_tables.SplitStatic | None, cfg_args: tuple, specs: tuple, use_int64_offsets: bool
):
    """Compile the warmup / uncut backward head over the upstream dynamic-layout tensor ABI.

    ``cfg_args`` rebuild the T-pass and recompute configs and ``specs`` describe every tensor
    argument in host order."""
    io_dtype, num_sm, d_k, d_v, log_gate, state_in = cfg_args
    io = get_dtype(io_dtype)
    tinv_cfg = gdn_tinv.bundle_hook(io, d_k, num_sm, log_gate)
    recompute_cfg = gdn_recompute.bundle_hook(io, d_k, d_v, state_in, num_sm)
    flags = "_".join(str(int(v)) if isinstance(v, bool) else str(v) for v in cfg_args)
    split_name = "none" if split is None else "_".join(str(v) for v in split)
    return compile_tvm_ffi(
        warmup_backward_host,
        split,
        tinv_cfg,
        recompute_cfg,
        *(cutlass.Int32(0) for _ in range(4)),
        cutlass.Float32(0.0),
        *(cutlass.Int32(0) for _ in range(3)),
        *(make_signature(spec, use_int64_offsets=use_int64_offsets) for spec in specs),
        name=f"gdn_wd_warmup_backward_{flags}_split_{split_name}_i64{int(use_int64_offsets)}".lower(),
        opt_level=2,
    )


@memoize_launch_check
def _validate_launch(
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
    tinv,
    tinv_rows,
    tinv_row_count,
    checkpoints,
    state_in,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler_all,
    scheduler_recompute,
    split,
    n_tiles,
    ideal_chunks,
    num_sm,
    b_t,
):
    """Check the warmup / uncut backward buffers of one plan (one checkpoint series per
    sequence)."""
    tokens, heads_out, num_seqs = validate_gdn_bwd_bundle(
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
        num_pieces=validate_seqlens(cu_seqlens),
        b_t=b_t,
    )
    validate_cuda_tensors(
        q,
        tinv=tinv,
        tinv_rows=tinv_rows,
        tinv_row_count=tinv_row_count,
        state_in=state_in,
        work_items=work_items,
        work_count=work_count,
        item_scratch=item_scratch,
        chunk_scratch=chunk_scratch,
        scheduler_all=scheduler_all,
        scheduler_recompute=scheduler_recompute,
    )
    if num_sm <= 0 or n_tiles != num_seqs * heads_out:
        raise ValueError(
            "num_sm must be positive and n_tiles must equal the number of sequences times output "
            "heads"
        )
    if b_t != gdn_recompute.B_T:
        raise ValueError(f"the warpdrive recompute serves {gdn_recompute.B_T}-token chunks")
    dim_v, dim_k = v.shape[2], q.shape[2]
    if state_in is not None:
        validate_tensor(
            "state_in",
            state_in,
            (None, heads_out, dim_v, dim_k),
            STATE_DTYPES,
            min_rows=num_seqs,
        )
    table_rows = n_tiles
    if split:
        scan_tokens = gate.shape[0]
        if ideal_chunks:
            table_rows = work_tables.max_work_items(
                scan_tokens, num_seqs, heads_out, ideal_chunks, b_t, num_sm
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
            min_rows=work_tables.chunk_scratch_rows(scan_tokens, num_seqs, b_t),
        )
    validate_work_table(work_items, work_count, scheduler_all, min_rows=table_rows)
    validate_tensor(
        "scheduler_recompute", scheduler_recompute, (None,), ("int32",), align=4, min_rows=1
    )
    validate_gdn_tinv(
        tinv,
        tinv_rows,
        tinv_row_count,
        tokens,
        num_seqs,
        heads_out,
        str(q.dtype).removeprefix("torch."),
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
    tinv,
    tinv_rows,
    tinv_row_count,
    checkpoints,
    state_in,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler_all,
    scheduler_recompute,
    tma_slots,
    split,
    n_tiles,
    ideal_chunks,
    num_sm,
    b_t,
    log_gate,
):
    """Compile (cached per static config) the head of the warmup or uncut backward over the buffers
    of one plan; return ``(compiled, facts)`` with ``facts`` the split table's
    :class:`~work_tables.SplitTableFacts` (None for the uncut table)."""
    # Attention Gym modification: validate the launch contract before selecting a compiled ABI.
    _validate_launch(
        q=q,
        k=k,
        v=v,
        do=do,
        dq=dq,
        dk=dk,
        dv=dv,
        gate=gate,
        beta=beta,
        cu_seqlens=cu_seqlens,
        tinv=tinv,
        tinv_rows=tinv_rows,
        tinv_row_count=tinv_row_count,
        checkpoints=checkpoints,
        state_in=state_in,
        work_items=work_items,
        work_count=work_count,
        item_scratch=item_scratch,
        chunk_scratch=chunk_scratch,
        scheduler_all=scheduler_all,
        scheduler_recompute=scheduler_recompute,
        split=split,
        n_tiles=n_tiles,
        ideal_chunks=ideal_chunks,
        num_sm=num_sm,
        b_t=b_t,
    )
    if not log_gate:
        raise ValueError("the warpdrive recompute takes natural-log gates (log_gate=True)")
    facts = None
    if split:
        facts = work_tables.split_table_facts(
            gate, cu_seqlens, n_tiles=n_tiles, ideal_chunks=ideal_chunks, num_sms=num_sm, b_t=b_t
        )
    cu_align = 8 if str(cu_seqlens.dtype).endswith("int64") else 4
    cfg_args = (
        str(q.dtype).removeprefix("torch."),
        int(num_sm),
        q.shape[2],
        v.shape[2],
        bool(log_gate),
        state_in is not None,
    )
    use_int64_offsets = requires_int64_abi(
        k, v, gate, beta, cu_seqlens, tinv, checkpoints, state_in
    )
    split_static = None if facts is None else facts.static
    # Every tensor a signature spec reads, so the warm path keys the launch without the specs.
    static_key = (
        split_static,
        cfg_args,
        use_int64_offsets,
        signature_key(
            dynamic=(
                k,
                v,
                gate,
                beta,
                cu_seqlens,
                tinv,
                tinv_row_count,
                checkpoints,
                state_in,
                work_count,
                chunk_scratch,
                scheduler_all,
                scheduler_recompute,
                tinv_rows,
            ),
            compact=(work_items, item_scratch, tma_slots),
        ),
    )

    def compile_launch():
        specs = (
            signature_spec(k, assumed_align=16),
            signature_spec(v, assumed_align=16),
            signature_spec(gate, assumed_align=16),
            signature_spec(beta, assumed_align=16),
            signature_spec(cu_seqlens, assumed_align=cu_align),
            signature_spec(tinv, assumed_align=128),
            signature_spec(tinv_rows, assumed_align=16),
            signature_spec(tinv_row_count, assumed_align=4),
            signature_spec(checkpoints, assumed_align=16),
            signature_spec(state_in, assumed_align=16),
            signature_spec(work_items, assumed_align=16, compact=True),
            signature_spec(work_count, assumed_align=4),
            signature_spec(item_scratch if split else None, assumed_align=16, compact=True),
            signature_spec(chunk_scratch if split else None, assumed_align=4),
            signature_spec(scheduler_all, assumed_align=4),
            signature_spec(scheduler_recompute, assumed_align=4),
            signature_spec(tma_slots, assumed_align=128, compact=True),
        )
        return _compile_warmup_backward(split_static, cfg_args, specs, use_int64_offsets)

    return _compile_warmup_backward.by_static_key(static_key, compile_launch), facts


def run_warmup_backward(
    compiled,
    facts,
    *,
    k,
    v,
    gate,
    beta,
    cu_seqlens,
    tinv,
    tinv_rows,
    tinv_row_count,
    checkpoints,
    state_in,
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler_all,
    scheduler_recompute,
    tma_slots,
    q=None,
    do=None,
    dq=None,
    dk=None,
    dv=None,
) -> None:
    """Replay the head of the warmup or uncut backward: one crossing into the DSL. The plan
    validated the contract at build, so nothing here raises. The bprop's own buffers (``q``,
    ``do`` and the gradients) are accepted so the caller replays with the build's buffers."""
    split = facts is not None
    compiled(
        *(facts.scalars() if split else (0, 0, 0, 0, 0.0, 0, 0, 0)),
        k,
        v,
        gate,
        beta,
        cu_seqlens,
        tinv,
        tinv_rows,
        tinv_row_count,
        checkpoints,
        state_in,
        work_items,
        work_count,
        item_scratch if split else None,
        chunk_scratch if split else None,
        scheduler_all,
        scheduler_recompute,
        tma_slots,
    )
