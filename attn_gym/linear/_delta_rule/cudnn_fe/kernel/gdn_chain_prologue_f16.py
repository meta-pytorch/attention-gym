# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe; the unused GDP d_v=64 bprop fork and the standalone
# run_chain_prologue host removed; the upstream-only GDP expand_num, compact_qdo, and
# summary_q_step paths removed at their pinned values; Ruff formatting.

"""
Chunked Gated Delta Net (GDN) piece-chain prologue for SM100 / SM103 / SM107 (Cutlass
primitives): the one launch that builds every table the chain's kernels read, so each consumer
keeps its body and skips its own prologue.

Phases (two blocks: both build the piece table, block 0 the work-item tables, block 1 the
descriptor arrays):
  piece table        : the flat piece slots of every sequence from cu_seqlens
  work-item tables   : the LPT order of the main and summary work-item tables, the scheduler rings, the
                       checkpoint-seeded series items where the backward reads a coarse series, the
                       T pass row table
  descriptor arrays  : the per-piece TMA descriptor arrays of every chain kernel, the T pass's K / tinv included

The bprop summary's q and dO arrays address the compact token timeline.

Warp assignments (descriptor phase, one warp per array):
  warp  0       : T pass K / tinv
  warps 1-2     : fused summary K / V
  warps 3-4     : recompute H K / V
  warps 5-6     : recompute M K / V (both from k)
  warps 7-9     : series recompute K / V / series out
  warps 10-14   : prefill Q / K / V / O / checkpoints out
  warps 15-17   : bprop summary Q / K / dO
  warps 18-25   : bprop Q / K / V / dO / series in / dQ / dK / dV
  warp  26      : T pass row table (one entry per valid chunk row)
  (every consumer's tinv array rides its K warp; the block has no warp to spare)
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.experimental.cuda.tensor_map as tma
import cutlass.experimental.primitives as nvvm
from cutlass import cute

from ..common.piece_chain import piece_table_body
from ..common.split_k import (
    ORDER_CAPACITY,
    ORDER_ELEMENTS,
    ORDER_THREADS,
    gen_interval_items,
    order_body,
)
from ..tile_dsl.barrier import launch_dependent_grids, wait_on_dependent_grids
from . import (
    gdn_bprop_f16,
    gdn_bprop_summary_f16,
    gdn_prefill_f16,
    gdn_recompute_f16,
    gdn_summary_f16,
    gdn_tinv_f16,
)

USE_PDL = True


@cute.kernel
def frost_gdn_chain_prologue(
    pieces: cutlass.Int32,
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    heads_out: cutlass.Int32,
    base_q: cutlass.GridConstant[tma.TensorMap],
    base_k: cutlass.GridConstant[tma.TensorMap],
    base_v: cutlass.GridConstant[tma.TensorMap],
    base_vk: cutlass.GridConstant[tma.TensorMap],
    base_o: cutlass.GridConstant[tma.TensorMap],
    base_do: cutlass.GridConstant[tma.TensorMap],
    base_checkpoint: cutlass.GridConstant[tma.TensorMap],
    base_dq: cutlass.GridConstant[tma.TensorMap],
    base_dk: cutlass.GridConstant[tma.TensorMap],
    base_dv: cutlass.GridConstant[tma.TensorMap],
    base_summary_q: cutlass.GridConstant[tma.TensorMap],
    base_summary_do: cutlass.GridConstant[tma.TensorMap],
    base_tinv: cutlass.GridConstant[tma.TensorMap],
    num_seqs: cutlass.Int32,
    series_span_chunks: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: cute.Tensor | None,
    scheduler: cute.Tensor,
    series_items: cute.Tensor | None,
    series_count: cute.Tensor | None,
    tinv_words: cute.Tensor | None,
    tinv_rows: cute.Tensor | None,
    tinv_row_count: cute.Tensor | None,
    summary_words: cute.Tensor | None,
    recompute_h_words: cute.Tensor | None,
    recompute_m_words: cute.Tensor | None,
    series_words: cute.Tensor | None,
    prefill_words: cute.Tensor | None,
    bprop_summary_words: cute.Tensor | None,
    bprop_words: cute.Tensor | None,
    q: cute.Tensor | None,
    k: cute.Tensor,
    v: cute.Tensor | None,
    o: cute.Tensor | None,
    do_: cute.Tensor | None,
    checkpoints: cute.Tensor | None,
    dq: cute.Tensor | None,
    dk: cute.Tensor | None,
    dv: cute.Tensor | None,
    summary_q: cute.Tensor | None,
    summary_do: cute.Tensor | None,
    tinv: cute.Tensor | None,
) -> None:
    if cutlass.const_expr(USE_PDL):
        wait_on_dependent_grids()
        launch_dependent_grids()
    tidx = cutlass.Int32(cute.arch.thread_idx()[0])
    widx = tidx // cutlass.Int32(32)
    bidx = cutlass.Int32(cute.arch.block_idx()[0])
    n_heads_out = cutlass.Int32(heads_out)

    # ---- piece table -----------------------------------------------------------------
    piece_table_body(
        ORDER_THREADS,
        pieces,
        unit_chunks,
        b_t,
        length_rule,
        heads_out,
        tidx,
        num_seqs,
        cu_seqlens,
        cu_pieces,
        main_rows,
        summary_rows,
    )
    nvvm.barrier_cta_sync()
    n_pieces = num_seqs * cutlass.Int32(pieces)

    if bidx == cutlass.Int32(0):
        # ---- work-item tables --------------------------------------------------------
        sKey = cutlass.Array(
            cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16
        )
        sIdx = cutlass.Array(
            cutlass.Int32, ORDER_CAPACITY, space=cutlass.AddressSpace.smem, alignment=16
        )
        sSpread = cutlass.Array(cutlass.Int32, 2, space=cutlass.AddressSpace.smem, alignment=8)
        order_body(
            False,
            b_t,
            ORDER_THREADS,
            ORDER_ELEMENTS,
            tidx,
            n_heads_out,
            n_heads_out * n_pieces,
            cu_pieces,
            None,
            main_count,
            work_items,
            scheduler,
            sKey,
            sIdx,
            sSpread,
            chain=True,
            mRowBase=main_rows,
        )
        if cutlass.const_expr(work_items_summary is not None):
            nvvm.barrier_cta_sync()
            order_body(
                False,
                b_t,
                ORDER_THREADS,
                ORDER_ELEMENTS,
                tidx,
                n_heads_out,
                n_heads_out * n_pieces,
                cu_pieces,
                None,
                summary_count,
                work_items_summary,
                None,
                sKey,
                sIdx,
                sSpread,
                chain=True,
                mRowBase=summary_rows,
                mSlotRows=main_rows,
            )
        if cutlass.const_expr(series_items is not None):
            gen_interval_items(
                b_t,
                ORDER_THREADS,
                tidx,
                n_heads_out,
                n_heads_out * n_pieces,
                series_span_chunks,
                cu_pieces,
                series_count,
                series_items,
                None,
            )
    else:
        # ---- descriptor arrays -------------------------------------------------------
        if cutlass.const_expr(tinv_words is not None):
            gdn_tinv_f16.build_descs_body(
                widx, base_k, base_tinv, tinv_words, cu_pieces, k, tinv, n_pieces, b_t
            )
        if cutlass.const_expr(tinv_rows is not None):
            if widx == 26:
                gdn_tinv_f16.emit_tinv_rows(
                    b_t, cu_pieces, tinv_rows, tinv_row_count, tidx % cutlass.Int32(32)
                )
        if cutlass.const_expr(summary_words is not None):
            gdn_summary_f16.build_descs_body(
                widx - cutlass.Int32(1),
                base_k,
                base_v,
                base_tinv,
                summary_words,
                cu_pieces,
                k,
                v,
                tinv,
                n_pieces,
                b_t,
            )
        if cutlass.const_expr(recompute_h_words is not None):
            gdn_recompute_f16.build_descs_body(
                widx - cutlass.Int32(3),
                base_k,
                base_v,
                base_k,
                base_tinv,
                recompute_h_words,
                cu_pieces,
                k,
                v,
                None,
                tinv,
                n_pieces,
                cutlass.Int32(0),
                b_t,
            )
        if cutlass.const_expr(recompute_m_words is not None):
            gdn_recompute_f16.build_descs_body(
                widx - cutlass.Int32(5),
                base_k,
                base_vk,
                base_k,
                base_tinv,
                recompute_m_words,
                cu_pieces,
                k,
                k,
                None,
                tinv,
                n_pieces,
                cutlass.Int32(0),
                b_t,
            )
        if cutlass.const_expr(series_words is not None):
            gdn_recompute_f16.build_descs_body(
                widx - cutlass.Int32(7),
                base_k,
                base_v,
                base_checkpoint,
                base_tinv,
                series_words,
                cu_pieces,
                k,
                v,
                checkpoints,
                tinv,
                n_pieces,
                checkpoint_every_n,
                b_t,
            )
        if cutlass.const_expr(prefill_words is not None):
            gdn_prefill_f16.build_descs_body(
                widx - cutlass.Int32(10),
                base_q,
                base_k,
                base_v,
                base_o,
                base_checkpoint,
                base_tinv,
                prefill_words,
                cu_pieces,
                q,
                k,
                v,
                o,
                checkpoints,
                tinv,
                n_pieces,
                checkpoint_every_n,
                b_t,
            )
        if cutlass.const_expr(bprop_summary_words is not None):
            gdn_bprop_summary_f16.build_descs_body(
                widx - cutlass.Int32(15),
                base_summary_q,
                base_k,
                base_summary_do,
                base_tinv,
                bprop_summary_words,
                cu_pieces,
                summary_q,
                k,
                summary_do,
                tinv,
                n_pieces,
                b_t,
            )
        if cutlass.const_expr(bprop_words is not None):
            gdn_bprop_f16.build_descs_body(
                widx - cutlass.Int32(18),
                base_q,
                base_k,
                base_v,
                base_do,
                base_checkpoint,
                base_dq,
                base_dk,
                base_dv,
                base_tinv,
                bprop_words,
                cu_pieces,
                q,
                k,
                v,
                do_,
                checkpoints,
                dq,
                dk,
                dv,
                None,
                n_pieces,
                checkpoint_every_n,
                b_t,
            )


@cute.jit
def chain_prologue(
    pieces: cutlass.Int32,
    unit_chunks: cutlass.Constexpr[int],
    b_t: cutlass.Constexpr[int],
    length_rule: cutlass.Constexpr[bool],
    heads_out: cutlass.Int32,
    series_span_chunks: cutlass.Int32,
    checkpoint_every_n: cutlass.Int32,
    cu_seqlens: cute.Tensor,
    cu_pieces: cute.Tensor,
    main_rows: cute.Tensor,
    summary_rows: cute.Tensor,
    main_count: cute.Tensor,
    summary_count: cute.Tensor,
    work_items: cute.Tensor,
    work_items_summary: cute.Tensor | None,
    scheduler: cute.Tensor,
    series_items: cute.Tensor | None,
    series_count: cute.Tensor | None,
    tinv_words: cute.Tensor | None,
    tinv_rows: cute.Tensor | None,
    tinv_row_count: cute.Tensor | None,
    summary_words: cute.Tensor | None,
    recompute_h_words: cute.Tensor | None,
    recompute_m_words: cute.Tensor | None,
    series_words: cute.Tensor | None,
    prefill_words: cute.Tensor | None,
    bprop_summary_words: cute.Tensor | None,
    bprop_words: cute.Tensor | None,
    q: cute.Tensor | None,
    k: cute.Tensor,
    v: cute.Tensor | None,
    o: cute.Tensor | None,
    do_: cute.Tensor | None,
    checkpoints: cute.Tensor | None,
    dq: cute.Tensor | None,
    dk: cute.Tensor | None,
    dv: cute.Tensor | None,
    summary_q: cute.Tensor | None,
    summary_do: cute.Tensor | None,
    tinv: cute.Tensor | None,
    stream: cuda.CUstream,
) -> None:
    swizzle_128b = tma.TensorMapSwizzle.s128b
    k_headed = cute.make_tensor(
        k.iterator,
        cute.make_layout(
            (k.shape[0], k.shape[1], k.shape[2]), stride=(k.stride[0], k.stride[1], 1)
        ),
    )
    base_k = tma.create_tensor_map_tiled_from_view(
        k_headed,
        box_dims=(b_t, 1, 128 // (k.element_type.width // 8)),
        stride_order=(2, 1, 0),
        swizzle=swizzle_128b,
    )
    vk_headed = cute.make_tensor(
        k.iterator,
        cute.make_layout(
            (k.shape[2], k.shape[1], k.shape[0]), stride=(1, k.stride[1], k.stride[0])
        ),
    )
    base_vk = tma.create_tensor_map_tiled_from_view(
        vk_headed,
        box_dims=(128 // (k.element_type.width // 8), 1, b_t),
        stride_order=(0, 1, 2),
        swizzle=swizzle_128b,
    )
    base_q = base_k
    base_v = base_k
    base_o = base_k
    base_do = base_k
    base_checkpoint = base_k
    base_dq = base_k
    base_dk = base_k
    base_dv = base_k
    base_summary_q = base_k
    base_summary_do = base_k
    base_tinv = base_k
    if cutlass.const_expr(q is not None):
        q_headed = cute.make_tensor(
            q.iterator,
            cute.make_layout(
                (q.shape[0], q.shape[1], q.shape[2]), stride=(q.stride[0], q.stride[1], 1)
            ),
        )
        base_q = tma.create_tensor_map_tiled_from_view(
            q_headed,
            box_dims=(b_t, 1, 128 // (q.element_type.width // 8)),
            stride_order=(2, 1, 0),
            swizzle=swizzle_128b,
        )
    if cutlass.const_expr(v is not None):
        v_headed = cute.make_tensor(
            v.iterator,
            cute.make_layout(
                (v.shape[2], v.shape[1], v.shape[0]), stride=(1, v.stride[1], v.stride[0])
            ),
        )
        base_v = tma.create_tensor_map_tiled_from_view(
            v_headed,
            box_dims=(128 // (v.element_type.width // 8), 1, b_t),
            stride_order=(0, 1, 2),
            swizzle=swizzle_128b,
        )
    if cutlass.const_expr(o is not None):
        o_headed = cute.make_tensor(
            o.iterator,
            cute.make_layout(
                (o.shape[2], o.shape[1], o.shape[0]), stride=(1, o.stride[1], o.stride[0])
            ),
        )
        base_o = tma.create_tensor_map_tiled_from_view(
            o_headed,
            box_dims=(128 // (o.element_type.width // 8), 1, b_t),
            stride_order=(0, 1, 2),
            swizzle=swizzle_128b,
        )
    if cutlass.const_expr(do_ is not None):
        do_headed = cute.make_tensor(
            do_.iterator,
            cute.make_layout(
                (do_.shape[2], do_.shape[1], do_.shape[0]),
                stride=(1, do_.stride[1], do_.stride[0]),
            ),
        )
        base_do = tma.create_tensor_map_tiled_from_view(
            do_headed,
            box_dims=(128 // (do_.element_type.width // 8), 1, b_t),
            stride_order=(0, 1, 2),
            swizzle=swizzle_128b,
        )
    if cutlass.const_expr(checkpoints is not None):
        checkpoint_view = cute.make_tensor(
            checkpoints.iterator,
            cute.make_layout(
                (
                    checkpoints.shape[3],
                    checkpoints.shape[2],
                    checkpoints.shape[0],
                    checkpoints.shape[1],
                ),
                stride=(
                    checkpoints.stride[3],
                    checkpoints.stride[2],
                    checkpoints.stride[0],
                    checkpoints.stride[1],
                ),
            ),
        )
        base_checkpoint = tma.create_tensor_map_tiled_from_view(
            checkpoint_view,
            box_dims=(128 // (checkpoints.element_type.width // 8), checkpoints.shape[2], 1, 1),
            stride_order=(0, 1, 2, 3),
            swizzle=swizzle_128b,
        )
    if cutlass.const_expr(dq is not None):
        dq_headed = cute.make_tensor(
            dq.iterator,
            cute.make_layout(
                (dq.shape[2], dq.shape[1], dq.shape[0]), stride=(1, dq.stride[1], dq.stride[0])
            ),
        )
        base_dq = tma.create_tensor_map_tiled_from_view(
            dq_headed,
            box_dims=(128 // (dq.element_type.width // 8), 1, b_t),
            stride_order=(0, 1, 2),
            swizzle=swizzle_128b,
        )
    if cutlass.const_expr(dk is not None):
        dk_headed = cute.make_tensor(
            dk.iterator,
            cute.make_layout(
                (dk.shape[2], dk.shape[1], dk.shape[0]), stride=(1, dk.stride[1], dk.stride[0])
            ),
        )
        base_dk = tma.create_tensor_map_tiled_from_view(
            dk_headed,
            box_dims=(128 // (dk.element_type.width // 8), 1, b_t),
            stride_order=(0, 1, 2),
            swizzle=swizzle_128b,
        )
    if cutlass.const_expr(dv is not None):
        dv_headed = cute.make_tensor(
            dv.iterator,
            cute.make_layout(
                (dv.shape[2], dv.shape[1], dv.shape[0]), stride=(1, dv.stride[1], dv.stride[0])
            ),
        )
        base_dv = tma.create_tensor_map_tiled_from_view(
            dv_headed,
            box_dims=(128 // (dv.element_type.width // 8), 1, b_t),
            stride_order=(0, 1, 2),
            swizzle=swizzle_128b,
        )
    if cutlass.const_expr(summary_q is not None):
        summary_q_headed = cute.make_tensor(
            summary_q.iterator,
            cute.make_layout(
                (summary_q.shape[0], summary_q.shape[1], summary_q.shape[2]),
                stride=(summary_q.stride[0], summary_q.stride[1], 1),
            ),
        )
        base_summary_q = tma.create_tensor_map_tiled_from_view(
            summary_q_headed,
            box_dims=(b_t, 1, 128 // (summary_q.element_type.width // 8)),
            stride_order=(2, 1, 0),
            swizzle=swizzle_128b,
        )
    if cutlass.const_expr(summary_do is not None):
        summary_do_headed = cute.make_tensor(
            summary_do.iterator,
            cute.make_layout(
                (summary_do.shape[2], summary_do.shape[1], summary_do.shape[0]),
                stride=(1, summary_do.stride[1], summary_do.stride[0]),
            ),
        )
        base_summary_do = tma.create_tensor_map_tiled_from_view(
            summary_do_headed,
            box_dims=(128 // (summary_do.element_type.width // 8), 1, b_t),
            stride_order=(0, 1, 2),
            swizzle=swizzle_128b,
        )
    if cutlass.const_expr(tinv is not None):
        tinv_tiles = cute.make_tensor(
            tinv.iterator,
            cute.make_layout(
                (tinv.shape[0], tinv.shape[1], tinv.shape[2], tinv.shape[3]),
                stride=(tinv.stride[0], tinv.stride[1], tinv.stride[2], 1),
            ),
        )
        base_tinv = tma.create_tensor_map_tiled_from_view(
            tinv_tiles, box_dims=(1, 1, b_t, b_t), stride_order=(3, 2, 1, 0), swizzle=swizzle_128b
        )
    frost_gdn_chain_prologue(
        pieces,
        unit_chunks,
        b_t,
        length_rule,
        heads_out,
        base_q,
        base_k,
        base_v,
        base_vk,
        base_o,
        base_do,
        base_checkpoint,
        base_dq,
        base_dk,
        base_dv,
        base_summary_q,
        base_summary_do,
        base_tinv,
        cutlass.Int32(cu_seqlens.shape[0] - 1),
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
        scheduler,
        series_items,
        series_count,
        tinv_words,
        tinv_rows,
        tinv_row_count,
        summary_words,
        recompute_h_words,
        recompute_m_words,
        series_words,
        prefill_words,
        bprop_summary_words,
        bprop_words,
        q,
        k,
        v,
        o,
        do_,
        checkpoints,
        dq,
        dk,
        dv,
        summary_q,
        summary_do,
        tinv,
    ).launch(grid=(2, 1, 1), block=(ORDER_THREADS, 1, 1), stream=stream, use_pdl=USE_PDL)


frost_gdn_chain_prologue.set_name_prefix("cudnn", remove_cutlass_symbol=False)
