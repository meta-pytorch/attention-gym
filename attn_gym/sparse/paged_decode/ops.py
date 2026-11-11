"""Private operators for the fused paged sparse decode.

Schemas and fakes register at import, so ``torch.compile`` traces each launch as one opaque op
and the Triton kernels see real tensors.
"""

from __future__ import annotations

import torch

_SCHEMA_ARGS = (
    "(Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, Tensor indices,"
    " Tensor cu_seqlens_q, Tensor seqused_k, Tensor? topk_length, Tensor? window_key_cache,"
    " Tensor? window_value_cache, Tensor? window_block_table, Tensor? attention_sink,"
    " SymInt max_seqlen_q, int index_block_size, int window_size, float scale)"
)
# Two fixed-arity ops instead of an optional LSE output.
torch.library.define("attn_gym::paged_sparse_decode", _SCHEMA_ARGS + " -> Tensor")
torch.library.define(
    "attn_gym::paged_sparse_decode_with_lse", _SCHEMA_ARGS + " -> (Tensor, Tensor)"
)


def _decode_cuda(*args, return_lse: bool):
    from .impl import triton

    *tensors, max_seqlen_q, index_block_size, window_size, scale = args
    return triton.paged_sparse_decode(
        *tensors,
        max_seqlen_q=max_seqlen_q,
        index_block_size=index_block_size,
        window_size=window_size,
        scale=scale,
        return_lse=return_lse,
    )


torch.library.impl(
    "attn_gym::paged_sparse_decode",
    "CUDA",
    lambda *args: _decode_cuda(*args, return_lse=False)[0],
)
torch.library.impl(
    "attn_gym::paged_sparse_decode_with_lse",
    "CUDA",
    lambda *args: _decode_cuda(*args, return_lse=True),
)


def _output_like(query: torch.Tensor, value_cache: torch.Tensor) -> torch.Tensor:
    return query.new_empty(query.shape[0], query.shape[1], value_cache.shape[-1])


@torch.library.register_fake("attn_gym::paged_sparse_decode")
def _decode_fake(query, key_cache, value_cache, *args) -> torch.Tensor:
    del key_cache, args
    return _output_like(query, value_cache)


@torch.library.register_fake("attn_gym::paged_sparse_decode_with_lse")
def _decode_with_lse_fake(
    query, key_cache, value_cache, *args
) -> tuple[torch.Tensor, torch.Tensor]:
    del key_cache, args
    return _output_like(query, value_cache), query.new_empty(query.shape[:2], dtype=torch.float32)
