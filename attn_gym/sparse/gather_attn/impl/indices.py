"""Triton index construction for the FA4 gather-attention adapter."""

import torch
import triton
import triton.language as tl
from torch import Tensor

from attn_gym._backends.triton.utils import _document_ids, ptr_offset, requires_int64_offsets


@triton.jit
def _build_gather_indices_kernel(
    Indices,
    GatherIndices,
    CuQ,
    num_documents,
    T: tl.constexpr,
    K: tl.constexpr,
    STRIDES: tl.constexpr,
    WINDOW: tl.constexpr,
    LOCAL_KV_LEN: tl.constexpr,
    OUT_K: tl.constexpr,
    PACKED: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    if WIDE:
        row = row.to(tl.int64)
        tile = tile.to(tl.int64)
    batch = row // T
    query = row % T
    start = 0
    if PACKED:
        document = _document_ids(CuQ, query, num_documents, WIDE)
        offset = document.to(tl.int64) if WIDE else document
        start = tl.load(CuQ + offset)
    slot = tile * BLOCK + tl.arange(0, BLOCK)
    local_position = query - WINDOW + 1 + slot
    local_valid = (slot < WINDOW) & (local_position >= start)
    index = tl.load(
        Indices + ptr_offset((batch, query, slot - WINDOW), STRIDES),
        (slot >= WINDOW) & (slot < WINDOW + K),
        -1,
    )
    sparse_position = tl.where(index >= 0, index + LOCAL_KV_LEN, -1)
    combined = tl.where(slot < WINDOW, tl.where(local_valid, local_position, -1), sparse_position)
    tl.store(GatherIndices + row * OUT_K + slot, combined, slot < OUT_K)


def build_gather_indices(
    kv_indices: Tensor,
    cu_seqlens: Tensor | None,
    sliding_window_size: int,
    local_kv_len: int,
) -> Tensor:
    """Build FA4's existing padded indices directly, without expanded document metadata."""
    batch, tokens, topk = kv_indices.shape
    slots = triton.cdiv(max(sliding_window_size + topk, 1), 128) * 128
    indices = kv_indices.new_empty((batch, tokens, slots), dtype=torch.int32)
    if batch * tokens == 0:
        return indices
    with torch.cuda.device(kv_indices.device):
        _build_gather_indices_kernel[(batch * tokens, triton.cdiv(slots, 256))](
            kv_indices,
            indices,
            cu_seqlens,
            0 if cu_seqlens is None else cu_seqlens.numel() - 1,
            tokens,
            topk,
            kv_indices.stride(),
            sliding_window_size,
            local_kv_len,
            slots,
            cu_seqlens is not None,
            requires_int64_offsets(kv_indices, indices, cu_seqlens),
            256,
            num_warps=4,
        )
    return indices
