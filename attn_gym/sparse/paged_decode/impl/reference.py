"""Torch-only paged sparse decode reference implementation."""

from __future__ import annotations

import torch
from torch import Tensor


def _gather_rows(cache: Tensor, block_table_row: Tensor, positions: Tensor) -> Tensor:
    """Read ``cache`` rows ``[n, kv_heads, dim]`` at logical ``positions`` of one sequence."""
    page_size = cache.shape[1]
    pages = block_table_row[positions // page_size].long()
    return cache[pages, positions % page_size]


def paged_sparse_decode(
    query: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_table: Tensor,
    indices: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    topk_length: Tensor | None,
    window_key_cache: Tensor | None,
    window_value_cache: Tensor | None,
    window_block_table: Tensor | None,
    attention_sink: Tensor | None,
    *,
    index_block_size: int,
    window_size: int,
    scale: float,
) -> tuple[Tensor, Tensor]:
    """Build each query row's key set explicitly and attend in at least float32.

    Rows with no keys return zero output; their LSE is the sink logit, or ``-inf``.
    """
    compute_dtype = torch.promote_types(query.dtype, torch.float32)
    num_tokens, num_heads, _ = query.shape
    kv_heads = key_cache.shape[2]
    group = num_heads // kv_heads
    shared_window = window_key_cache is None
    if shared_window:
        window_key_cache, window_value_cache = key_cache, value_cache
        window_block_table = block_table
    out = query.new_zeros(num_tokens, num_heads, value_cache.shape[-1])
    lse = torch.full(
        (num_tokens, num_heads), float("-inf"), dtype=compute_dtype, device=query.device
    )
    sink = None if attention_sink is None else attention_sink.to(compute_dtype)
    sparse_capacity = block_table.shape[1] * key_cache.shape[1]
    window_capacity = window_block_table.shape[1] * window_key_cache.shape[1]
    block_offsets = torch.arange(index_block_size, device=query.device)
    offsets = cu_seqlens_q.tolist()
    lengths = None if topk_length is None else topk_length.tolist()

    for sequence, num_keys in enumerate(seqused_k.tolist()):
        start, end = offsets[sequence], offsets[sequence + 1]
        for row in range(start, end):
            position = num_keys - (end - row)
            if position < 0:
                # More queries than keys: the row has no keys, not even sparse ones.
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
            # A shared source attends each key once: the window owns positions from its start.
            limit = window_start if shared_window else sparse_capacity
            entries = entries[entries < min(limit, sparse_capacity)]

            keys, values = (
                torch.cat(
                    (
                        _gather_rows(cache, block_table[sequence], entries),
                        _gather_rows(window_cache, window_block_table[sequence], window_positions),
                    )
                ).to(compute_dtype)
                for cache, window_cache in (
                    (key_cache, window_key_cache),
                    (value_cache, window_value_cache),
                )
            )
            # [keys, kv_heads, dim] -> one key set per query head.
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
    return out, lse.float()
