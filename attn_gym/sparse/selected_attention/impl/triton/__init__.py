"""Triton backend for selected attention."""

import torch

from .backward import _build_index_query_map, _launch_backward
from .forward import _launch_forward


class _SelectedAttentionFunction(torch.autograd.Function):
    """Autograd wrapper around the Triton launchers."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        sparse_kv: torch.Tensor,
        local_kv: torch.Tensor,
        kv_indices: torch.Tensor,
        attention_sink: torch.Tensor,
        doc_ids: torch.Tensor | None,
        sliding_window_size: int,
        share_kv: bool,
    ) -> torch.Tensor:
        if share_kv:
            sparse_kv = sparse_kv.expand(-1, query.shape[1], -1, -1)
            local_kv = local_kv.expand(-1, query.shape[1], -1, -1)

        output, lse = _launch_forward(
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            attention_sink,
            doc_ids,
            sliding_window_size,
        )
        # Keep the output-owned fallback available if deterministic mode is enabled before backward.
        selected_queries, block_offsets = _build_index_query_map(kv_indices, sparse_kv.shape[2])
        ctx.save_for_backward(
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            selected_queries,
            block_offsets,
            attention_sink,
            output,
            lse,
        )
        ctx.doc_ids = doc_ids
        ctx.sliding_window_size = sliding_window_size
        ctx.share_kv = share_kv
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            selected_queries,
            block_offsets,
            attention_sink,
            output,
            lse,
        ) = ctx.saved_tensors
        grad_query, grad_sparse_kv, grad_local_kv, grad_sink = _launch_backward(
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            selected_queries,
            block_offsets,
            attention_sink,
            ctx.doc_ids,
            output,
            lse,
            grad_output,
            ctx.sliding_window_size,
            ctx.share_kv,
        )
        return grad_query, grad_sparse_kv, grad_local_kv, None, grad_sink, None, None, None


def selected_attention(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    sparse_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor,
    doc_ids: torch.Tensor | None,
    sliding_window_size: int,
    share_kv: bool,
) -> torch.Tensor:
    """Triton implementation of selected attention.

    Args:
        query: (batch, heads, seq_len, head_dim) — queries.
        local_kv: (batch, 1 or heads, seq_len, head_dim) — local sliding-window key-values.
        sparse_kv: (batch, 1 or heads, sparse_seq_len, head_dim) — candidate KV pool.
        kv_indices: (batch, seq_len, topk) — which sparse_kv positions each query attends to.
        attention_sink: (heads,) — learned per-head sink weight.
        doc_ids: (batch, seq_len) or None — document IDs for packing isolation.
        sliding_window_size: size of the causal sliding window.
        share_kv: if True, broadcast single-head KV and return single-head gradients.

    Returns:
        Attention output with same shape as query.
    """
    heads = query.shape[1]

    if query.device.type != "cuda":
        raise ValueError("The Triton selected attention backend requires CUDA tensors.")

    query = query.contiguous()
    kv_indices = kv_indices.contiguous()
    if doc_ids is not None:
        doc_ids = doc_ids.contiguous()

    requires_grad = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, local_kv, sparse_kv, attention_sink)
    )
    if requires_grad and query.shape[-1] == 512:
        raise NotImplementedError(
            "The Triton selected-attention backend currently supports head_dim=512 only for "
            "inference."
        )
    if requires_grad:
        return _SelectedAttentionFunction.apply(
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            attention_sink,
            doc_ids,
            sliding_window_size,
            share_kv,
        )

    if share_kv:
        local_kv = local_kv.expand(-1, heads, -1, -1)
        sparse_kv = sparse_kv.expand(-1, heads, -1, -1)
    return _launch_forward(
        query, sparse_kv, local_kv, kv_indices, attention_sink, doc_ids, sliding_window_size
    )[0]
