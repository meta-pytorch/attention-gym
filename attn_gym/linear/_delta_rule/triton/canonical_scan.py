# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused per-document prefix of affine state maps for the deterministic CP recipe.

The maps are ``[N, H, V + K, K]`` packed ``[bias; transition]`` (see
``attn_gym.linear.context_parallel``). Tile ``i`` of a document enters with the bias of the
composition of all earlier tiles' maps. Composition right-multiplies by the next transition,
``(A0, B0) ∘ (A1, B1) = (A0 @ A1, B0 @ A1 + B1)``, so the running bias steps as
``B @ A_tile + B_tile`` and every row depends only on the same row before it: a program owns
``BM`` bias rows and walks its document's tiles serially with one ``[BM, K] @ [K, K]`` dot per
tile, storing the rows before each step. The running transition is never needed, there are no
intermediate buffers, and the whole scan is one launch.

The tree is the serial fold in three-pass TF32 (``tf32x3``, fp32-accurate), the same arithmetic
``compose_work_items`` uses. It is a fixed function of the document's tile count, so the result
does not depend on which rank produced a leaf, on the document's position in the packed stream,
or on the CP degree.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def canonical_scan_kernel(
    maps,
    doc_offsets,
    entries,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    REVERSE: tl.constexpr,
):
    """Entry states of one document, one head, ``BM`` rows of the ``[V, K]`` state.

    ``doc_offsets[d]:doc_offsets[d + 1]`` are the tile ids of document ``d``, contiguous in
    tile order; ``REVERSE`` walks them from the last tile (the exit-cotangent scan).
    """
    doc = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    row0 = tl.program_id(2) * BM
    rows = row0 + tl.arange(0, BM)
    cols = tl.arange(0, K)
    first = tl.load(doc_offsets + doc)
    stop = tl.load(doc_offsets + doc + 1)

    acc = tl.zeros([BM, K], dtype=tl.float32)  # entry state of the first tile
    for step in tl.range(0, stop - first, num_stages=2):
        if REVERSE:
            tile = stop - 1 - step
        else:
            tile = first + step
        base = maps + (tile.to(tl.int64) * H + head) * (V + K) * K
        tl.store(
            entries + (tile.to(tl.int64) * H + head) * V * K + rows[:, None] * K + cols[None, :],
            acc,
        )
        transition = tl.load(base + (V + cols[:, None]) * K + cols[None, :])  # [K, K]
        bias = tl.load(base + rows[:, None] * K + cols[None, :])
        acc = tl.dot(acc, transition, input_precision="tf32x3") + bias


def canonical_scan_entries(
    maps: torch.Tensor,
    doc_offsets: torch.Tensor,
    *,
    reverse: bool = False,
) -> torch.Tensor:
    """Entry state ``[N, H, V, K]`` of every tile from one fused per-document serial scan.

    ``maps`` are ``[N, H, V + K, K]`` FP32 in tile order (tiles of one document contiguous);
    ``doc_offsets`` is ``int32 [D + 1]`` on the device. ``reverse`` folds each document from its
    last tile, giving exit cotangents from reverse maps. One launch covers every document, head,
    and row tile.
    """
    count, heads, packed, key_dim = maps.shape
    value_dim = packed - key_dim
    entries = torch.empty(
        count, heads, value_dim, key_dim, dtype=torch.float32, device=maps.device
    )
    block_rows = 16
    documents = doc_offsets.shape[0] - 1
    canonical_scan_kernel[(documents, heads, value_dim // block_rows)](
        maps.contiguous(),
        doc_offsets,
        entries,
        H=heads,
        V=value_dim,
        K=key_dim,
        BM=block_rows,
        REVERSE=reverse,
        num_warps=8,
    )
    return entries


__all__ = ["canonical_scan_entries"]
