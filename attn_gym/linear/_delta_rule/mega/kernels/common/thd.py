# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared THD / varlen (packed ``[T,H,D]`` + ``cu_seqlens``) device helpers.

* :data:`TENSOR_MAP_QWORDS` — int64 words per 128-byte TMA descriptor.
* :func:`emit_seq_load_descs` — per-sequence input descriptors that keep the
  original packed allocation map at the physical token endpoint.
* :func:`emit_seq_descs` — device helper (one electing thread) that builds
  a per-BATCH TMA-descriptor array in GMEM for varlen loads/stores over a
  packed ``[T,H,D]`` tensor whose head axis is a load coordinate.  Each op
  calls it from a single ``prologue_kernel`` launch (one
  warp per array).
* :func:`emit_checkpoint_seq_descs` — its per-chunk-checkpoint sibling; derives the
  per-sequence checkpoint offsets from the token ``cu_seqlens`` in place of a
  caller-computed prefix array.
"""

import cutlass
import cutlass.cute as cute
import cutlass.experimental.primitives as nvvm
from cutlass.base_dsl.typing import Pointer

TENSOR_MAP_QWORDS = 128 // 8


@cute.jit
def emit_seq_load_descs(
    base_desc,
    desc_words,
    cu_seqlens,
    base_ptr,
    n_batch: cutlass.Int32,
    row_stride: cutlass.Int64,
    seq_ord: cutlass.Constexpr[int],
) -> None:
    """Build sequence-relative input maps except at the physical token endpoint.

    Interior maps retain per-sequence bounds so partial tail tiles zero-fill rather
    than reading the next sequence. Maps ending at the packed-token endpoint keep the
    original allocation base and use that endpoint as their bound, avoiding a
    relocated short map whose last TMA tile can overfetch beyond the allocation.
    """
    desc_base = desc_words.iterator.raw_ptr()
    src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
    cu = cutlass.make_array_view(cu_seqlens)
    base = base_ptr.iterator.raw_ptr()
    packed_tokens = cutlass.Int32(cu[n_batch])
    for b in cutlass.range(0, n_batch, 1, unroll=1):
        cu_b = cutlass.Int32(cu[b])
        cu_next = cutlass.Int32(cu[b + cutlass.Int32(1)])
        s_b = cu_next - cu_b
        dptr = desc_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)
        for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
            (dptr + i).store((src_words + i).load())
        if cu_next == packed_tokens:
            nvvm.tensormap_replace(
                nvvm.TensormapField.GLOBAL_DIM,
                dptr,
                new_value=packed_tokens,
                ord=seq_ord,
            )
        else:
            addr = base + cutlass.Int64(cu_b) * cutlass.Int64(row_stride)
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


@cute.jit
def emit_seq_descs(
    base_desc,
    desc_words,
    cu_seqlens,
    base_ptr,
    n_batch: cutlass.Int32,
    row_stride: cutlass.Int64,
    seq_ord: cutlass.Constexpr[int],
) -> None:
    """Per-BATCH TMA-descriptor array for a VARLEN (THD) tensor whose base
    descriptor carries the head axis as a real dimension (``(d, head,
    token)``); the head index is a load COORDINATE, so only the sequence
    base and length are patched per slot.  GLOBAL_ADDRESS folds
    ``cu_seqlens[b] * row_stride`` (Int64); GLOBAL_DIM[``seq_ord``] is
    capped to the per-sequence token count so tail loads zero-fill and tail
    stores clip in hardware.  GQA/GVA head grouping happens at the issue
    site (``head_idx // group`` with a static group), not here.  Runs on
    one electing thread; the calling warp elects and release-fences
    (GENERIC->TENSORMAP)."""
    desc_base = desc_words.iterator.raw_ptr()
    src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
    cu = cutlass.make_array_view(cu_seqlens)
    base = base_ptr.iterator.raw_ptr()
    for b in cutlass.range(0, n_batch, 1, unroll=1):
        cu_b = cutlass.Int32(cu[b])
        s_b = cutlass.Int32(cu[b + cutlass.Int32(1)]) - cu_b
        dptr = desc_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)
        for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
            (dptr + i).store((src_words + i).load())
        addr = base + cutlass.Int64(cu_b) * cutlass.Int64(row_stride)
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


@cute.jit
def emit_checkpoint_seq_descs(
    base_desc,
    desc_words,
    cu_seqlens,
    base_ptr,
    n_batch: cutlass.Int32,
    row_stride: cutlass.Int64,
    every_n: cutlass.Int32,
    seq_ord: cutlass.Constexpr[int],
) -> None:
    """Per-BATCH descriptor array for the per-chunk checkpoint tensor with the head
    axis as a descriptor dimension (``(dv, dk, chunk, head)``).  Derives the
    per-sequence checkpoint offsets from the TOKEN ``cu_seqlens`` on the fly
    (``count_b = (seqlen_b - 1) // every_n + 1``, running-prefix-summed) — an
    address fold no coordinate transform can express — and caps
    GLOBAL_DIM[``seq_ord``] to ``count_b``.  The head index is a load
    coordinate.  Runs on one electing thread; the calling warp elects and
    fences."""
    desc_base = desc_words.iterator.raw_ptr()
    src_words = Pointer(base_desc.get_ptr(), dtype=cutlass.Int64)
    cu = cutlass.make_array_view(cu_seqlens)
    base = base_ptr.iterator.raw_ptr()
    run = cutlass.Int32(0)
    for b in cutlass.range(0, n_batch, 1, unroll=1):
        s_tok = cutlass.Int32(cu[b + cutlass.Int32(1)]) - cutlass.Int32(cu[b])
        cnt = (s_tok - cutlass.Int32(1)) // every_n + cutlass.Int32(1)
        cnt = cnt if s_tok > 0 else cutlass.Int32(0)
        checkpoint_base = run
        run = run + cnt
        dptr = desc_base + b * cutlass.Int32(TENSOR_MAP_QWORDS)
        for i in cutlass.range_constexpr(TENSOR_MAP_QWORDS):
            (dptr + i).store((src_words + i).load())
        addr = base + cutlass.Int64(checkpoint_base) * cutlass.Int64(row_stride)
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
