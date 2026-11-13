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
# (python/cudnn/linear_attention/frost/kernel/gdn_chain_forward_f16.py); the launch compiles once
# per static configuration through jit_cache over fake TVM-FFI tensor signatures, with an
# int64-extent variant for oversized tensors, and runs on the environment stream; the upstream-only
# expand_num, safe_gate/A_log/dt_bias, beta-sigmoid, and negative-eigenvalue host paths removed at
# their pinned values; Ruff formatting. Every launch is a warpdrive (CuTeDSL 4.6.2) kernel
# and the chain prologue builds only tables (no per-piece descriptor arrays); the batch-invariant
# length rule and checkpoint-cadence piece units the drivers never compile are gone.

"""One compiled launch for the GDN chain forward: chain tables, T pass, fused summary, fp32 state
chain and prefill issued from a single host, so the Python side crosses into the DSL once per call
instead of five times. Every kernel is a warpdrive kernel (``piece_chain``, ``gdn_tinv``,
``gdn_summary``, ``gdn_prefill``) addressing every piece through ``ragged``
descriptors. A buffer that two kernels read through different signature types (the summary's
``state_out`` is the state chain's ``H``) is passed twice, once per type."""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.cute.runtime import make_fake_compact_tensor

from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache

from .. import gdn_prefill, gdn_summary, gdn_tinv, piece_chain
from ..contract import get_dtype
from ..work_tables import WORK_ITEM_FIELDS
from .tvm_ffi import (
    make_compact_signature_tensor,
    make_counter_signature,
    make_cu_seqlens_signature,
    make_strided_signature_tensor,
)

OPT_LEVEL = 2
B_T = gdn_prefill.B_T


@cute.jit
def chain_forward_host(
    tinv_cfg: cutlass.Constexpr,
    summary_cfg: cutlass.Constexpr,
    prefill_cfg: cutlass.Constexpr,
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
    scheduler_prefill: cute.Tensor,
    tinv_rows: cute.Tensor,
    tinv_row_count: cute.Tensor,
    tinv: cute.Tensor,
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
        rows=tinv_rows,
        row_count=tinv_row_count,
        rows_b_t=B_T,
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
        state_m_summary,
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
        seed_indices,
        stream,
    )
    prefill_op, prefill_sms = prefill_cfg
    prefill_op.launch(
        q,
        k,
        v,
        gate,
        None,
        o,
        state_x_prefill,
        final_state,
        None,
        None,
        tinv,
        work_items,
        main_count,
        scheduler_prefill,
        tma_slots,
        scale,
        cutlass.Int32(prefill_sms),
        stream,
    )


@jit_cache
def _compile_chain_forward(
    io_dtype,
    state_dtype,
    final_dtype,
    gate_dtype,
    beta_dtype,
    seed_dtype,
    num_sm,
    d_k,
    d_v,
    log_gate,
    enable_checkpoints,
    has_seed_indices,
    has_final_indices,
    chain_rows,
    use_int64_offsets,
):
    """Compile the chain forward launch for one static configuration (dtypes, heads-independent
    gate and state flags, dims, chain rows, device target); every extent is symbolic.
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

    def rows():
        return make_compact_signature_tensor(cutlass.Int32, (sym_int(),), assumed_align=16)

    def summary_state():
        # The summary's outputs and the prefill's seed read the compact [P, HO, D, d_k] chain
        # states whose inner extent is a multiple of d_k (16-byte vector accesses).
        return make_fake_compact_tensor(
            state_dtype,
            (sym_int(), sym_int(), sym_int(), sym_int(divisibility=d_k)),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )

    tinv_cfg = gdn_tinv.bundle_hook(io_dtype, d_k, num_sm, log_gate)
    summary_cfg = gdn_summary.bundle_hook(io_dtype, state_dtype, d_k, d_v, log_gate)
    prefill_cfg = gdn_prefill.bundle_hook(
        io_dtype,
        state_dtype,
        beta_dtype,
        num_sm=num_sm,
        use_initial_state=True,
        store_final_state=final_dtype is not None,
        enable_checkpoints=enable_checkpoints,
        log_gate=log_gate,
        tinv_source="gmem",
        d_k=d_k,
        d_v=d_v,
        has_final_indices=has_final_indices,
    )
    flags = (
        num_sm,
        d_k,
        d_v,
        log_gate,
        enable_checkpoints,
        has_seed_indices,
        has_final_indices,
        chain_rows,
        use_int64_offsets,
    )
    dtypes = (io_dtype, state_dtype, final_dtype, gate_dtype, beta_dtype, seed_dtype)
    name = "gdn_chain_forward_" + "_".join(str(int(flag)) for flag in flags)
    name += "_" + "_".join("none" if dtype is None else dtype.__name__.lower() for dtype in dtypes)
    return compile_tvm_ffi(
        chain_forward_host,
        tinv_cfg,
        summary_cfg,
        prefill_cfg,
        d_v,
        d_k,
        chain_rows,
        *(cutlass.Int32(0) for _ in range(3)),
        cutlass.Float32(0),
        tensor(io_dtype, 3, 16),
        tensor(io_dtype, 3, 16),
        tensor(io_dtype, 3, 16),
        tensor(gate_dtype, 2, 16),
        tensor(beta_dtype, 2, 16),
        tensor(io_dtype, 3, 16),
        make_cu_seqlens_signature(sym_int(), assumed_align=4),
        make_counter_signature(sym_int()),
        rows(),
        rows(),
        make_counter_signature(sym_int()),
        make_counter_signature(sym_int()),
        work_items(),
        work_items(),
        make_counter_signature(sym_int()),
        make_counter_signature(sym_int()),
        tensor(cutlass.Int32, 2, 16),
        make_counter_signature(sym_int()),
        tensor(io_dtype, 4, 128),
        summary_state(),
        summary_state(),
        *(tensor(state_dtype, 4, 16) for _ in range(3)),
        tensor(seed_dtype, 4, 4) if seed_dtype is not None else None,
        make_counter_signature(sym_int()) if has_seed_indices else None,
        summary_state(),
        tensor(final_dtype, 4, 16) if final_dtype is not None else None,
        make_compact_signature_tensor(cutlass.Int64, (sym_int(), 16), assumed_align=128),
        name=name,
        opt_level=OPT_LEVEL,
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
    tinv_rows,
    tinv_row_count,
    tinv,
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
    """Return the compiled chain forward launch over the buffers of one plan: compiled (and
    persisted) once per static configuration (dtypes, dims, gate flags, checkpoint, seed and
    final-state presence, chain rows, device); ``pieces``, ``heads_out`` and ``num_seqs`` are
    launch arguments."""
    if b_t != B_T:
        raise ValueError(f"the GDN chain runs b_t={B_T} chunks, got {b_t}")
    if state_h.dtype != state_x.dtype or state_h.dtype != state_m.dtype:
        raise TypeError("the chain states must share one dtype")
    # Every tensor the launch addresses; bounded int32 tables and counters excepted.
    use_int64_offsets = requires_int64_abi(
        q,
        k,
        v,
        gate,
        beta,
        o,
        tinv,
        state_h,
        state_m,
        state_x,
        seed,
        final_state,
        checkpoints,
    )
    return _compile_chain_forward.by_args(
        get_dtype(q.dtype),
        get_dtype(state_x.dtype),
        get_dtype(final_state.dtype) if final_state is not None else None,
        get_dtype(gate.dtype),
        get_dtype(beta.dtype),
        get_dtype(seed.dtype) if seed is not None else None,
        int(num_sm),
        int(q.shape[2]),
        int(v.shape[2]),
        bool(log_gate),
        int(checkpoint_every_n_tokens) > 0,
        seed_indices is not None,
        final_indices is not None,
        int(chain_rows),
        use_int64_offsets,
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
    scheduler_prefill,
    tinv_rows,
    tinv_row_count,
    tinv,
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
    scheduler_summary=None,
    final_indices=None,
    checkpoints=None,
    checkpoint_every_n_tokens=0,
) -> None:
    """Replay the chain forward on the current stream: one crossing into the DSL for the five
    launches.  The plan validated the contract at build, so nothing here raises. The trailing
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
        scheduler_prefill,
        tinv_rows,
        tinv_row_count,
        tinv,
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
