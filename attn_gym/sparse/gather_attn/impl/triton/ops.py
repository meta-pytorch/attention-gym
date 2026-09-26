"""Opaque forward and backward operators for the Triton gather-attention backend."""

import torch
from torch import Tensor

from .backward import _build_index_query_map, _launch_backward
from .forward import _launch_forward

torch.library.define(
    "attn_gym::_gather_attn_fwd",
    "(Tensor query, Tensor sparse_kv, Tensor local_kv, Tensor kv_indices, "
    "Tensor attention_sink, Tensor? cu_seqlens, Tensor? cu_seqlens_k, "
    "int sliding_window_size, bool share_kv, float scale, bool needs_backward) "
    "-> (Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::_gather_attn_bwd",
    "(Tensor query, Tensor sparse_kv, Tensor local_kv, Tensor kv_indices, "
    "Tensor selected_queries, Tensor block_offsets, Tensor attention_sink, "
    "Tensor? cu_seqlens, Tensor? cu_seqlens_k, Tensor output, Tensor lse, "
    "Tensor grad_output, int sliding_window_size, bool share_kv, float scale) "
    "-> (Tensor, Tensor, Tensor, Tensor)",
)


def _gather_attn_fwd_cuda(
    query: Tensor,
    sparse_kv: Tensor,
    local_kv: Tensor,
    kv_indices: Tensor,
    attention_sink: Tensor,
    cu_seqlens: Tensor | None,
    cu_seqlens_k: Tensor | None,
    sliding_window_size: int,
    share_kv: bool,
    scale: float,
    needs_backward: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    with torch.cuda.device(query.device):
        query = query.contiguous()
        kv_indices = kv_indices.contiguous()
        attention_sink = attention_sink.contiguous()
        if share_kv:
            sparse_kv = sparse_kv.expand(-1, query.shape[1], -1, -1)
            local_kv = local_kv.expand(-1, query.shape[1], -1, -1)
        output, lse = _launch_forward(
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            attention_sink,
            cu_seqlens,
            cu_seqlens_k,
            sliding_window_size,
            scale,
        )
        if needs_backward:
            # Backward may enable deterministic mode after this forward has run.
            selected_queries, block_offsets = _build_index_query_map(
                kv_indices, sparse_kv.shape[2], cu_seqlens, cu_seqlens_k
            )
        else:
            selected_queries = kv_indices.new_empty((query.shape[0], 0), dtype=torch.int32)
            block_offsets = kv_indices.new_empty((query.shape[0], 0), dtype=torch.int32)
        return output, lse, selected_queries, block_offsets


def _gather_attn_bwd_cuda(
    query: Tensor,
    sparse_kv: Tensor,
    local_kv: Tensor,
    kv_indices: Tensor,
    selected_queries: Tensor,
    block_offsets: Tensor,
    attention_sink: Tensor,
    cu_seqlens: Tensor | None,
    cu_seqlens_k: Tensor | None,
    output: Tensor,
    lse: Tensor,
    grad_output: Tensor,
    sliding_window_size: int,
    share_kv: bool,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    with torch.cuda.device(query.device):
        query = query.contiguous()
        kv_indices = kv_indices.contiguous()
        attention_sink = attention_sink.contiguous()
        if share_kv:
            sparse_kv = sparse_kv.expand(-1, query.shape[1], -1, -1)
            local_kv = local_kv.expand(-1, query.shape[1], -1, -1)
        # The launcher reduces shared-KV gradients back to the original single head.
        # Deterministic mode is consulted here at execution time, not during tracing.
        return _launch_backward(
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            selected_queries,
            block_offsets,
            attention_sink,
            cu_seqlens,
            cu_seqlens_k,
            output,
            lse,
            grad_output,
            sliding_window_size,
            share_kv,
            scale,
        )


torch.library.impl("attn_gym::_gather_attn_fwd", "CUDA", _gather_attn_fwd_cuda)
torch.library.impl("attn_gym::_gather_attn_bwd", "CUDA", _gather_attn_bwd_cuda)


@torch.library.register_fake("attn_gym::_gather_attn_fwd")
def _gather_attn_fwd_fake(
    query: Tensor,
    sparse_kv: Tensor,
    local_kv: Tensor,
    kv_indices: Tensor,
    attention_sink: Tensor,
    cu_seqlens: Tensor | None,
    cu_seqlens_k: Tensor | None,
    sliding_window_size: int,
    share_kv: bool,
    scale: float,
    needs_backward: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batch, heads, seq_len, _ = query.shape
    selected_size = seq_len * kv_indices.shape[2] if needs_backward else 0
    offsets_size = sparse_kv.shape[2] + 1 if needs_backward else 0
    return (
        query.new_empty(query.shape),
        query.new_empty((batch, heads, seq_len), dtype=torch.float32),
        kv_indices.new_empty((batch, selected_size), dtype=torch.int32),
        kv_indices.new_empty((batch, offsets_size), dtype=torch.int32),
    )


@torch.library.register_fake("attn_gym::_gather_attn_bwd")
def _gather_attn_bwd_fake(
    query: Tensor,
    sparse_kv: Tensor,
    local_kv: Tensor,
    kv_indices: Tensor,
    selected_queries: Tensor,
    block_offsets: Tensor,
    attention_sink: Tensor,
    cu_seqlens: Tensor | None,
    cu_seqlens_k: Tensor | None,
    output: Tensor,
    lse: Tensor,
    grad_output: Tensor,
    sliding_window_size: int,
    share_kv: bool,
    scale: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        query.new_empty(query.shape),
        sparse_kv.new_empty(sparse_kv.shape),
        local_kv.new_empty(local_kv.shape),
        attention_sink.new_empty(attention_sink.shape),
    )


_gather_attn_fwd_op = torch.ops.attn_gym._gather_attn_fwd.default
_gather_attn_bwd_op = torch.ops.attn_gym._gather_attn_bwd.default
