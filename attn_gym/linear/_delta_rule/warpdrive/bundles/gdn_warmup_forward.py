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
# (python/cudnn/linear_attention/frost/kernel/gdn_warmup_forward_f16.py); the launch compiles once
# per static configuration through jit_cache over fake TVM-FFI tensor signatures (independent
# state_in/state_out extents), with an int64-extent variant for oversized tensors, and runs on the
# environment stream; the paged-state prefill flag is threaded through; the upstream-only
# expand_num, safe_gate/A_log/dt_bias, beta-sigmoid, negative-eigenvalue, and
# log2_threshold-override host paths removed at their pinned values; Ruff formatting.

"""One compiled launch for the GDN warmup, uncut and dv_split forwards: the split-K table (plan,
scan and walk, warmup only), the work-table prologue and the prefill, issued from a single host so
the Python side crosses into the DSL once per call. All three are warpdrive kernels
(``work_tables`` and ``gdn_prefill``); they address every sequence through
``ragged`` descriptors, so the launch needs only the small early-end slot buffer."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute

from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache

from .. import gdn_prefill, work_tables
from ..contract import get_dtype
from ..work_tables import WORK_ITEM_FIELDS
from .tvm_ffi import (
    make_compact_signature_tensor,
    make_counter_signature,
    make_cu_seqlens_signature,
    make_strided_signature_tensor,
)

OPT_LEVEL = 2


@cute.jit
def warmup_forward_host(
    split: cutlass.Constexpr,
    prefill_cfg: cutlass.Constexpr,
    tiles_per_head: cutlass.Constexpr[int],
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
    beta: cute.Tensor | None,
    o: cute.Tensor,
    cu_seqlens: cute.Tensor,
    state_in: cute.Tensor | None,
    state_out: cute.Tensor | None,
    seed_indices: cute.Tensor | None,
    has_initial_state: cute.Tensor | None,
    work_items: cute.Tensor,
    work_count: cute.Tensor,
    item_scratch: cute.Tensor | None,
    chunk_scratch: cute.Tensor | None,
    scheduler: cute.Tensor,
    tma_slots: cute.Tensor,
    stream: cuda.CUstream,
) -> None:
    """``split`` is the split table's :class:`~work_tables.SplitStatic` (None: the uncut
    table); the eight scalars after it are its runtime values (zeros when uncut)."""
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
        gdn_prefill.B_T,
        cutlass.Int32(gate.shape[1]) * cutlass.Int32(tiles_per_head),
        cu_seqlens,
        item_scratch,
        work_count,
        work_items,
        scheduler,
        stream,
        state_indices=seed_indices if cutlass.const_expr(has_initial_state is not None) else None,
        has_initial_state=has_initial_state,
    )
    prefill_op, prefill_sms = prefill_cfg
    prefill_op.launch(
        q,
        k,
        v,
        gate,
        beta,
        o,
        state_in,
        state_out,
        seed_indices,
        has_initial_state,
        None,
        work_items,
        work_count,
        scheduler,
        tma_slots,
        scale,
        cutlass.Int32(prefill_sms),
        stream,
    )


@jit_cache
def _compile_warmup_forward(
    io_dtype,
    state_dtype,
    gate_dtype,
    beta_dtype,
    split,
    num_sms,
    use_initial_state,
    store_final_state,
    has_seed_indices,
    has_final_indices,
    paged_state,
    has_initial_state,
    enable_checkpoints,
    log_gate,
    d_k,
    d_v,
    tiles_per_head,
    use_int64_offsets,
):
    """Compile the warmup or uncut forward launch for one static configuration (dtypes, gate and
    state flags, split-table geometry, d_v split, device target); every extent is symbolic.
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

    prefill_cfg = gdn_prefill.bundle_hook(
        io_dtype,
        state_dtype,
        beta_dtype,
        num_sm=num_sms,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
        enable_checkpoints=enable_checkpoints,
        log_gate=log_gate,
        tinv_source="compute",
        d_k=d_k,
        d_v=d_v // tiles_per_head,
        tiles_per_head=tiles_per_head,
        paged_state=paged_state,
        has_final_indices=has_final_indices,
    )
    flags = (
        num_sms,
        use_initial_state,
        store_final_state,
        has_seed_indices,
        has_final_indices,
        paged_state,
        has_initial_state,
        enable_checkpoints,
        log_gate,
        d_k,
        d_v,
        tiles_per_head,
        use_int64_offsets,
    )
    dtypes = (io_dtype, state_dtype, gate_dtype, beta_dtype)
    name = "gdn_warmup_forward_" + "_".join(str(int(flag)) for flag in flags)
    name += "_split_" + ("none" if split is None else "_".join(str(v) for v in split))
    name += "_" + "_".join("none" if dtype is None else dtype.__name__.lower() for dtype in dtypes)
    return compile_tvm_ffi(
        warmup_forward_host,
        split,
        prefill_cfg,
        tiles_per_head,
        *(cutlass.Int32(0) for _ in range(4)),
        cutlass.Float32(0),
        *(cutlass.Int32(0) for _ in range(3)),
        cutlass.Float32(0),
        tensor(io_dtype, 3, 16),
        tensor(io_dtype, 3, 16),
        tensor(io_dtype, 3, 16),
        tensor(gate_dtype, 2, 16),
        tensor(beta_dtype, 2, 16),
        tensor(io_dtype, 3, 16),
        make_cu_seqlens_signature(sym_int(), assumed_align=4),
        tensor(state_dtype, 4, 16) if use_initial_state else None,
        tensor(state_dtype, 4, 16) if store_final_state else None,
        make_counter_signature(sym_int()) if has_seed_indices else None,
        make_compact_signature_tensor(cutlass.Uint8, (sym_int(),), assumed_align=1)
        if has_initial_state
        else None,
        work_items(),
        make_counter_signature(sym_int()),
        work_items() if split is not None else None,
        tensor(cutlass.Float32, 2, 4) if split is not None else None,
        make_counter_signature(sym_int()),
        make_compact_signature_tensor(cutlass.Int64, (sym_int(), 16), assumed_align=128),
        name=name,
        opt_level=OPT_LEVEL,
    )


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
    has_initial_state=None,
    paged_state=False,
):
    """Return ``(compiled, facts)`` for the warmup or uncut forward over the buffers of one plan:
    the launch is compiled (and persisted) once per static configuration, so every shape-dependent
    value is a launch argument. ``facts`` is the split table's
    :class:`~work_tables.SplitTableFacts`, None for the uncut table."""
    if b_t != gdn_prefill.B_T:
        raise ValueError(f"the GDN forward runs b_t={gdn_prefill.B_T} chunks, got {b_t}")
    facts = None
    if split:
        facts = work_tables.split_table_facts(
            gate, cu_seqlens, n_tiles=n_tiles, ideal_chunks=ideal_chunks, num_sms=num_sm, b_t=b_t
        )
    state_src = state_in if state_in is not None else state_out
    state_dtype = get_dtype(state_src.dtype) if state_src is not None else cutlass.Float32
    # Every tensor the launch addresses; bounded int32 routing arrays and counters excepted.
    use_int64_offsets = requires_int64_abi(
        q, k, v, gate, beta, o, state_in, state_out, checkpoints, chunk_scratch
    )
    compiled = _compile_warmup_forward.by_args(
        get_dtype(q.dtype),
        state_dtype,
        get_dtype(gate.dtype),
        get_dtype(beta.dtype),
        None if facts is None else facts.static,
        int(num_sm),
        state_in is not None,
        state_out is not None,
        seed_indices is not None,
        final_indices is not None,
        bool(paged_state),
        has_initial_state is not None,
        int(checkpoint_every_n_tokens) > 0,
        bool(log_gate),
        int(q.shape[2]),
        int(v.shape[2]),
        int(tiles_per_head),
        use_int64_offsets,
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
    work_items,
    work_count,
    item_scratch,
    chunk_scratch,
    scheduler,
    tma_slots,
    scale,
    has_initial_state=None,
    final_indices=None,
    checkpoints=None,
    checkpoint_every_n_tokens=0,
) -> None:
    """Replay the warmup or uncut forward on the current stream: one crossing into the DSL for the
    table, prologue and prefill launches.  The plan validated the contract at build, so nothing
    here raises. ``final_indices``, ``checkpoints`` and the cadence only shape the build; they are
    accepted so a caller replays with the build's buffers."""
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
        seed_indices,
        has_initial_state,
        work_items,
        work_count,
        item_scratch if split else None,
        chunk_scratch if split else None,
        scheduler,
        tma_slots,
    )
