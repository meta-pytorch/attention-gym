"""Input factories for paged sparse decode tests and benchmarks."""

from __future__ import annotations

from typing import Any

import torch

from attn_gym.sparse.paged_decode import PagedKV


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
    """Scatter each sequence over randomly permuted pages, with one spare page per row."""
    pages_per_sequence = max(-(-length // page_size) for length in seqlens) + 1
    num_pages = len(seqlens) * pages_per_sequence + 3
    block_table = torch.randperm(num_pages, generator=generator)[
        : len(seqlens) * pages_per_sequence
    ]
    width = head_dim if latent_values else head_dim + head_dim_v
    # One [pages, page, heads, K | V] allocation, the layout serving engines use.
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
    """Build keyword arguments for ``paged_sparse_decode``.

    Indices select random distinct complete blocks visible to each query. With
    ``compress_ratio == 1`` the sparse and window keys share one source (QSA); otherwise ``kv``
    holds one compressed entry per ``compress_ratio`` tokens and the window reads a separate raw
    cache paged by ``window_page_size`` (DeepSeek V4). ``topk_length=False`` scatters the
    selections across the row instead of compacting them in front.
    """
    generator = torch.Generator().manual_seed(seed)
    head_dim_v = head_dim_v or head_dim
    separate_window = compress_ratio > 1
    sparse_lengths = [length // compress_ratio for length in seqlens]
    kv = _paged_source(
        [max(length, 1) for length in sparse_lengths],
        page_size,
        kv_heads,
        head_dim,
        head_dim_v,
        latent_values,
        dtype,
        device,
        generator,
    )
    window_kv = None
    if separate_window:
        window_kv = _paged_source(
            seqlens,
            window_page_size or page_size,
            kv_heads,
            head_dim,
            head_dim_v,
            latent_values,
            dtype,
            device,
            generator,
        )

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
                # Only complete blocks that end before the window are selectable.
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
        "cu_seqlens_q": torch.tensor([0, *query_lens]).cumsum(0).to(device, torch.int32),
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
