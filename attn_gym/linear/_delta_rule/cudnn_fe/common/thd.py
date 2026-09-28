# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe.

"""Shared THD / varlen (packed ``[T,H,D]`` + ``cu_seqlens``) device helpers.

* :data:`TENSOR_MAP_QWORDS`, int64 words per 128-byte TMA descriptor.
* :func:`emit_seq_descs`, device helper (one electing thread) that builds
  a per-BATCH TMA-descriptor array in GMEM for varlen loads/stores over a
  packed ``[T,H,D]`` tensor whose head axis is a load coordinate.  Each op
  calls it from a single prologue-kernel launch (one
  warp per array).
* :func:`emit_checkpoint_seq_descs`, its per-chunk-checkpoint sibling; derives the
  per-sequence checkpoint offsets from the token ``cu_seqlens``.
"""

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass.base_dsl.typing import Pointer

TENSOR_MAP_QWORDS = 128 // 8
TENSOR_MAP_BIT21 = 1 << 21  # qword 1: encoder's "tensor >= 128 KiB" flag; tensormap.replace does not update it (issue #1013)


@cute.jit
def set_tensor_map_bit21(dptr, new_bytes: cutlass.Int64) -> None:
    """Recompute bit 21 from the patched extent (as cuTensorMapEncodeTiled would)."""
    w = (dptr + 1).load() & cutlass.Int64(~TENSOR_MAP_BIT21)
    (dptr + 1).store((w | cutlass.Int64(TENSOR_MAP_BIT21)) if new_bytes >= cutlass.Int64(128 << 10) else w)


@cute.jit
def emit_seq_descs(
    base_desc,
    desc_words,
    cu_seqlens,
    base_ptr,
    n_batch: cutlass.Int32,
    seq_ord: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int] = 1,
    lanes: cutlass.Constexpr[int] = 1,
) -> None:
    """Per-BATCH TMA-descriptor array for a VARLEN (THD) tensor whose base
    descriptor carries the head axis as a real dimension (``(d, head,
    token)``); the head index is a load COORDINATE, so only the sequence
    base and length are patched per slot.  GLOBAL_ADDRESS folds
    ``cu_seqlens[b] * base_ptr.stride[0]`` (Int64, the token-axis stride); GLOBAL_DIM[``seq_ord``] is
    capped to the per-sequence token count so tail loads zero-fill and tail
    stores clip in hardware.  ``expand_num`` scales every loaded ``cu``
    value onto GDP's sub-token timeline (1 = off).  GQA/GVA head grouping
    happens at the issue site (``head_idx // group`` with a static group),
    not here.  ``lanes == 1``: one electing thread emits every slot (the
    caller elects and release-fences GENERIC->TENSORMAP); ``lanes == 32``:
    the whole warp calls, lane ``l`` emits slots ``l, l + 32, ...`` and every
    lane release-fences its own."""
    desc_base = desc_words.iterator.raw_ptr()
    src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
    cu = cutlass.make_array_view(cu_seqlens)
    base = base_ptr.iterator.raw_ptr()
    first = cutlass.Int32(0)
    if cutlass.const_expr(lanes > 1):
        first = cutlass.Int32(cute.arch.thread_idx()[0]) % cutlass.Int32(lanes)
    for b in cutlass.range(first, n_batch, lanes, unroll=1):
        cu_b = cutlass.Int32(cu[b])
        s_b = cutlass.Int32(cu[b + cutlass.Int32(1)]) - cu_b
        if cutlass.const_expr(expand_num > 1):
            cu_b = cu_b * cutlass.Int32(expand_num)
            s_b = s_b * cutlass.Int32(expand_num)
        dptr = desc_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)
        for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
            (dptr + i).store((src_words + i).load())
        addr = base + cutlass.Int64(cu_b) * cutlass.Int64(base_ptr.stride[0])
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_ADDRESS,
            dptr,
            new_value=addr.toint(cutlass.Int64),
        )
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_DIM,
            dptr,
            new_value=s_b,
            ord=seq_ord,
        )
        set_tensor_map_bit21(dptr, cutlass.Int64(s_b) * cutlass.Int64(base_ptr.stride[0]) * cutlass.Int64(base_ptr.element_type.width // 8))


@cute.jit
def emit_tile_seq_descs(
    base_desc,
    desc_words,
    cu_seqlens,
    base_ptr,
    n_batch: cutlass.Int32,
    b_t: cutlass.Constexpr[int],
    seq_ord: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int] = 1,
    lanes: cutlass.Constexpr[int] = 1,
) -> None:
    """Per-BATCH descriptor array for the chunk-inverse tile buffer ``(row, head, b_t, b_t)`` of gdn_tinv_f16.  Sequence b
    owns the tile rows from ``cu[b] // b_t + b`` (one padding row per sequence keeps that base closed-form), so
    GLOBAL_ADDRESS advances by that row times ``base_ptr.stride[0]`` and GLOBAL_DIM[``seq_ord``] is capped to the
    sequence's chunk count.  ``expand_num`` scales the loaded ``cu`` values onto GDP's sub-token timeline (1 = off);
    ``lanes`` as in :func:`emit_seq_descs`."""
    desc_base = desc_words.iterator.raw_ptr()
    src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
    cu = cutlass.make_array_view(cu_seqlens)
    base = base_ptr.iterator.raw_ptr()
    first = cutlass.Int32(0)
    if cutlass.const_expr(lanes > 1):
        first = cutlass.Int32(cute.arch.thread_idx()[0]) % cutlass.Int32(lanes)
    for b in cutlass.range(first, n_batch, lanes, unroll=1):
        cu_b = cutlass.Int32(cu[b])
        s_b = cutlass.Int32(cu[b + cutlass.Int32(1)]) - cu_b
        if cutlass.const_expr(expand_num > 1):
            cu_b = cu_b * cutlass.Int32(expand_num)
            s_b = s_b * cutlass.Int32(expand_num)
        row_b = cu_b // cutlass.Int32(b_t) + b
        n_b = (s_b + cutlass.Int32(b_t - 1)) // cutlass.Int32(b_t)
        dptr = desc_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)
        for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
            (dptr + i).store((src_words + i).load())
        addr = base + cutlass.Int64(row_b) * cutlass.Int64(base_ptr.stride[0])
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_ADDRESS,
            dptr,
            new_value=addr.toint(cutlass.Int64),
        )
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_DIM,
            dptr,
            new_value=n_b,
            ord=seq_ord,
        )
        set_tensor_map_bit21(dptr, cutlass.Int64(n_b) * cutlass.Int64(base_ptr.stride[0]) * cutlass.Int64(base_ptr.element_type.width // 8))


@cute.jit
def emit_checkpoint_seq_descs(
    base_desc,
    desc_words,
    cu_seqlens,
    base_ptr,
    n_batch: cutlass.Int32,
    every_n: cutlass.Int32,
    seq_ord: cutlass.Constexpr[int],
    expand_num: cutlass.Constexpr[int] = 1,
    lanes: cutlass.Constexpr[int] = 1,
) -> None:
    """Per-BATCH descriptor array for the per-chunk checkpoint tensor with the head
    axis as a descriptor dimension (``(dv, dk, chunk, head)``).  Derives the
    per-sequence checkpoint offsets from the TOKEN ``cu_seqlens`` on the fly
    (``count_b = (batch_seqlen - 1) // every_n + 1``, running-prefix-summed) and
    caps GLOBAL_DIM[``seq_ord``] to ``count_b``; GLOBAL_ADDRESS advances by
    ``count_prefix * base_ptr.stride[0]`` (the checkpoint-row stride).  ``expand_num`` scales the
    loaded per-sequence token counts onto GDP's sub-token timeline (1 =
    off).  The head index is a load coordinate.  ``lanes`` as in
    :func:`emit_seq_descs`; a striped lane re-derives the prefix of its
    slot from the sequences before it."""
    desc_base = desc_words.iterator.raw_ptr()
    src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
    cu = cutlass.make_array_view(cu_seqlens)
    base = base_ptr.iterator.raw_ptr()
    first = cutlass.Int32(0)
    if cutlass.const_expr(lanes > 1):
        first = cutlass.Int32(cute.arch.thread_idx()[0]) % cutlass.Int32(lanes)
    run = cutlass.Int32(0)
    for b0 in cutlass.range(0, first, 1, unroll=1):
        if b0 < n_batch:
            s_tok0 = cutlass.Int32(cu[b0 + cutlass.Int32(1)]) - cutlass.Int32(cu[b0])
            if cutlass.const_expr(expand_num > 1):
                s_tok0 = s_tok0 * cutlass.Int32(expand_num)
            cnt0 = (s_tok0 - cutlass.Int32(1)) // every_n + cutlass.Int32(1)
            run = run + (cnt0 if s_tok0 > 0 else cutlass.Int32(0))
    for b in cutlass.range(first, n_batch, lanes, unroll=1):
        s_tok = cutlass.Int32(cu[b + cutlass.Int32(1)]) - cutlass.Int32(cu[b])
        if cutlass.const_expr(expand_num > 1):
            s_tok = s_tok * cutlass.Int32(expand_num)
        cnt = (s_tok - cutlass.Int32(1)) // every_n + cutlass.Int32(1)
        cnt = cnt if s_tok > 0 else cutlass.Int32(0)
        checkpoint_base = run
        run = run + cnt
        for bn in cutlass.range(b + cutlass.Int32(1), b + cutlass.Int32(lanes), 1, unroll=1):
            if bn < n_batch:
                s_tokn = cutlass.Int32(cu[bn + cutlass.Int32(1)]) - cutlass.Int32(cu[bn])
                if cutlass.const_expr(expand_num > 1):
                    s_tokn = s_tokn * cutlass.Int32(expand_num)
                cntn = (s_tokn - cutlass.Int32(1)) // every_n + cutlass.Int32(1)
                run = run + (cntn if s_tokn > 0 else cutlass.Int32(0))
        dptr = desc_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)
        for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
            (dptr + i).store((src_words + i).load())
        addr = base + cutlass.Int64(checkpoint_base) * cutlass.Int64(base_ptr.stride[0])
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_ADDRESS,
            dptr,
            new_value=addr.toint(cutlass.Int64),
        )
        nvvm.tensormap_replace(
            nvvm.TensormapField.GLOBAL_DIM,
            dptr,
            new_value=cnt,
            ord=seq_ord,
        )
        set_tensor_map_bit21(dptr, cutlass.Int64(cnt) * cutlass.Int64(base_ptr.stride[0]) * cutlass.Int64(base_ptr.element_type.width // 8))
