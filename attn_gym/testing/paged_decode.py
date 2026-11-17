"""Reproducible paged-decode inputs shared by correctness tests and benchmarks."""

from __future__ import annotations

from functools import partial
from typing import Any

import torch

from attn_gym.sparse.paged_decode import PagedKV
from attn_gym.testing.kda import cumulative_sequence_offsets


def _paged_source(
    seqlens: list[int],
    page_size: int,
    kv_heads: int,
    head_dim: int,
    head_dim_v: int,
    latent_values: bool,
    dtype: torch.dtype,
    device: torch.device | str,
    generator: torch.Generator,
) -> PagedKV:
    """Use permuted physical pages so logical entry offsets cannot double as cache offsets.

    Every block-table row fits the longest sequence plus one spare page, so shorter sequences
    have more unused capacity. That padding must not affect visible-key counts.
    """
    pages_per_sequence = max(-(-length // page_size) for length in seqlens) + 1
    num_pages = len(seqlens) * pages_per_sequence + 3
    block_table = torch.randperm(num_pages, generator=generator)[
        : len(seqlens) * pages_per_sequence
    ]
    width = head_dim if latent_values else head_dim + head_dim_v
    # Views exercise non-contiguous outer strides; latent values also alias the key prefix.
    cache = torch.randn(num_pages, page_size, kv_heads, width, generator=generator).to(
        device, dtype
    )
    key_cache = cache[..., :head_dim]
    value_cache = cache[..., :head_dim_v] if latent_values else cache[..., head_dim:]
    return PagedKV(
        key_cache, value_cache, block_table.view(len(seqlens), -1).to(device, torch.int32)
    )


def make_paged_sparse_decode_inputs(
    seqlens: list[int],
    query_lens: list[int],
    *,
    heads: int = 24,
    kv_heads: int = 2,
    head_dim: int = 256,
    head_dim_v: int | None = None,
    page_size: int = 64,
    topk: int = 512,
    index_block_size: int = 4,
    window_size: int = 0,
    window_page_size: int | None = None,
    compress_ratio: int = 1,
    attention_sink: bool = False,
    latent_values: bool = False,
    topk_length: bool = True,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device | str = "cuda",
    seed: int = 0,
) -> dict[str, Any]:
    """Build deterministic, causal keyword arguments for ``paged_sparse_decode``.

    ``seqlens`` counts raw keys, including the queries. Selected block ids are distinct and
    refer only to complete blocks visible to the query. With ``compress_ratio > 1``, ``kv``
    represents one entry per complete group of raw tokens and the window uses a separate raw
    source with ``window_page_size``. Otherwise both key sets share one source.

    Caches contain random values, not actual compressed tokens. ``latent_values=True`` makes
    values a key-prefix view. With ``topk_length=False``, valid selections are scattered among
    negative padding instead of packed into a prefix, exercising the full-row read path.
    """
    generator = torch.Generator().manual_seed(seed)
    head_dim_v = head_dim_v or head_dim
    separate_window = compress_ratio > 1
    source = partial(
        _paged_source,
        kv_heads=kv_heads,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        latent_values=latent_values,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    kv = source([max(length // compress_ratio, 1) for length in seqlens], page_size)
    window_kv = source(seqlens, window_page_size or page_size) if separate_window else None

    rows = sum(query_lens)
    indices = torch.full((rows, topk), -1, dtype=torch.int32)
    lengths = torch.zeros(rows, dtype=torch.int32)
    row = 0
    for length, num_queries in zip(seqlens, query_lens):
        for i in range(num_queries):
            position = length - num_queries + i
            if separate_window:
                visible = (position + 1) // compress_ratio // index_block_size
            else:
                # Exclude the aligned window's entries so these mimic complete-block indexer ids.
                window_start = max(position + 1 - window_size, 0)
                visible = window_start // index_block_size
            count = min(max(visible, 0), topk)
            chosen = torch.randperm(max(visible, 1), generator=generator)[:count].to(torch.int32)
            columns = (
                torch.arange(count)
                if topk_length
                else torch.randperm(topk, generator=generator)[:count]
            )
            indices[row, columns] = chosen
            lengths[row] = count
            row += 1

    return {
        "query": torch.randn(rows, heads, head_dim, generator=generator).to(device, dtype),
        "kv": kv,
        "indices": indices.to(device),
        "cu_seqlens_q": cumulative_sequence_offsets(query_lens, device=device),
        "seqused_k": torch.tensor(seqlens, dtype=torch.int32, device=device),
        "max_seqlen_q": max(query_lens, default=1),
        "topk_length": lengths.to(device) if topk_length else None,
        "index_block_size": index_block_size,
        "window_size": window_size,
        "window_kv": window_kv,
        "attention_sink": (
            (torch.randn(heads, generator=generator) * 2).to(device) if attention_sink else None
        ),
    }
