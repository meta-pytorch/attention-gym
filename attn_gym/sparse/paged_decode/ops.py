"""Opaque CUDA operators keep Triton launch setup out of ``torch.compile`` tracing.

Schemas and fake implementations register at import time. Tracing uses output metadata only;
execution passes real tensors to the Triton launcher, which chooses the grid and scratch sizes.
"""

from __future__ import annotations

import torch

_SCHEMA_ARGS = (
    "(Tensor query, Tensor key_cache, Tensor value_cache, Tensor block_table, Tensor indices,"
    " Tensor cu_seqlens_q, Tensor seqused_k, Tensor? topk_length, Tensor? window_key_cache,"
    " Tensor? window_value_cache, Tensor? window_block_table, Tensor? attention_sink,"
    " SymInt max_seqlen_q, int index_block_size, int window_size, float scale)"
)
# Separate schemas let tracing know the output structure without an optional Tensor result.
torch.library.define("attn_gym::paged_sparse_decode", _SCHEMA_ARGS + " -> Tensor")
torch.library.define(
    "attn_gym::paged_sparse_decode_with_lse", _SCHEMA_ARGS + " -> (Tensor, Tensor)"
)


def _decode_cuda(*args, return_lse: bool):
    from .impl import triton

    return triton.paged_sparse_decode(*args, return_lse=return_lse)


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


@torch.library.register_fake("attn_gym::paged_sparse_decode")
def _decode_fake(query, key_cache, value_cache, *args) -> torch.Tensor:
    return query.new_empty(*query.shape[:2], value_cache.shape[-1])


@torch.library.register_fake("attn_gym::paged_sparse_decode_with_lse")
def _decode_with_lse_fake(query, *args) -> tuple[torch.Tensor, torch.Tensor]:
    return _decode_fake(query, *args), query.new_empty(query.shape[:2], dtype=torch.float32)


decode_op = torch.ops.attn_gym.paged_sparse_decode.default
decode_with_lse_op = torch.ops.attn_gym.paged_sparse_decode_with_lse.default
