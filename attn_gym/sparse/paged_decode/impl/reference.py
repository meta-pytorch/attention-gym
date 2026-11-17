"""Eager PyTorch oracle that materializes each query's sparse and window entries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..api import PagedKV


def _rows(source: PagedKV, sequence: int, positions: Tensor) -> tuple[Tensor, Tensor]:
    """Map sequence-local entries to physical pages and slots; retain order and duplicates."""
    page_size = source.key_cache.shape[1]
    pages = source.block_table[sequence, positions // page_size].long()
    slots = positions % page_size
    return source.key_cache[pages, slots], source.value_cache[pages, slots]


def paged_sparse_decode(
    query: Tensor,
    kv: PagedKV,
    indices: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    topk_length: Tensor | None,
    window_kv: PagedKV | None,
    attention_sink: Tensor | None,
    *,
    index_block_size: int,
    window_size: int,
    scale: float,
) -> tuple[Tensor, Tensor]:
    """Evaluate the API's key sets in float32, or float64 for float64 inputs.

    Host reads and explicit gathers favor inspectable semantics over capture or performance.
    """
    compute_dtype = torch.promote_types(query.dtype, torch.float32)
    num_tokens, num_heads, _ = query.shape
    group = num_heads // kv.key_cache.shape[2]
    shared_window = window_kv is None
    window_source = kv if shared_window else window_kv
    out = query.new_zeros(num_tokens, num_heads, kv.value_cache.shape[-1])
    lse = torch.full(
        (num_tokens, num_heads), float("-inf"), dtype=torch.float32, device=query.device
    )
    sink = None if attention_sink is None else attention_sink.to(compute_dtype)
    sparse_capacity = kv.block_table.shape[1] * kv.key_cache.shape[1]
    window_capacity = window_source.block_table.shape[1] * window_source.key_cache.shape[1]
    block_offsets = torch.arange(index_block_size, device=query.device)
    offsets = cu_seqlens_q.tolist()
    lengths = None if topk_length is None else topk_length.tolist()

    for sequence, num_keys in enumerate(seqused_k.tolist()):
        start, end = offsets[sequence], offsets[sequence + 1]
        for row in range(start, end):
            position = num_keys - (end - row)
            if position < 0:
                # Queries before position zero must not attend to sparse entries either.
                window_start = window_end = width = 0
            else:
                width = indices.shape[1] if lengths is None else lengths[row]
                width = min(max(width, 0), indices.shape[1])
                window_start = max(position + 1 - window_size, 0)
                if shared_window:
                    window_start -= window_start % index_block_size
                window_end = min(position + 1, window_capacity)
            window_positions = torch.arange(window_start, window_end, device=query.device)
            blocks = indices[row, :width].long()
            blocks = blocks[blocks >= 0]
            entries = (blocks[:, None] * index_block_size + block_offsets).flatten()
            # A shared source excludes sparse entries at or after the window start to avoid
            # overlap. Duplicate selections before it retain their multiplicity.
            limit = min(window_start, sparse_capacity) if shared_window else sparse_capacity
            entries = entries[entries < limit]

            sparse_keys, sparse_values = _rows(kv, sequence, entries)
            window_keys, window_values = _rows(window_source, sequence, window_positions)
            keys = torch.cat((sparse_keys, window_keys)).to(compute_dtype)
            values = torch.cat((sparse_values, window_values)).to(compute_dtype)
            # Consecutive query heads in a group share the same KV head.
            keys = keys.repeat_interleave(group, dim=1)
            values = values.repeat_interleave(group, dim=1)
            logits = torch.einsum("hd,nhd->hn", query[row].to(compute_dtype), keys) * scale
            if sink is not None:
                logits = torch.cat((logits, sink[:, None]), dim=1)
            if logits.shape[1] == 0:
                continue
            row_lse = logits.logsumexp(dim=1)
            probabilities = torch.exp(logits - row_lse[:, None])[:, : keys.shape[0]]
            out[row] = torch.einsum("hn,nhd->hd", probabilities, values).to(out.dtype)
            lse[row] = row_lse
    return out, lse
