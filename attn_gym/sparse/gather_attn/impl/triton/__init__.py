"""Triton backend for gather attention."""

import torch

from .ops import _gather_attn_bwd_op, _gather_attn_fwd_op


class _GatherAttnFunction(torch.autograd.Function):
    """Autograd wrapper around the opaque Triton operators."""

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        sparse_kv: torch.Tensor,
        local_kv: torch.Tensor,
        kv_indices: torch.Tensor,
        attention_sink: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        cu_seqlens_k: torch.Tensor | None,
        sliding_window_size: int,
        share_kv: bool,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, lse, selected_queries, block_offsets = _gather_attn_fwd_op(
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            attention_sink,
            cu_seqlens,
            cu_seqlens_k,
            sliding_window_size,
            share_kv,
            scale,
            True,
        )
        # Save the function inputs, including offsets, for backward and version checks.
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
            cu_seqlens,
            cu_seqlens_k,
        )
        ctx.sliding_window_size = sliding_window_size
        ctx.share_kv = share_kv
        ctx.scale = scale
        ctx.mark_non_differentiable(lse)
        return output, lse

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: torch.Tensor, grad_lse: torch.Tensor):
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
            cu_seqlens,
            cu_seqlens_k,
        ) = ctx.saved_tensors
        grad_query, grad_sparse_kv, grad_local_kv, grad_sink = _gather_attn_bwd_op(
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
            ctx.sliding_window_size,
            ctx.share_kv,
            ctx.scale,
        )
        return (
            grad_query,
            grad_sparse_kv,
            grad_local_kv,
            None,
            grad_sink,
            None,
            None,
            None,
            None,
            None,
        )


def gather_attn(
    query: torch.Tensor,
    local_kv: torch.Tensor,
    sparse_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    attention_sink: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
    sliding_window_size: int,
    share_kv: bool,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton implementation of gather attention.

    Opaque operators keep launch selection and backward preparation out of Dynamo tracing.
    Sequence lengths and outer strides may vary with ``dynamic=True``;
    model dimensions and options can still specialize, as can Triton's normal alignment
    and equal-to-one argument properties. Packed offset contents remain device-side inputs.

    Args:
        query: (batch, heads, seq_len, head_dim) — queries.
        local_kv: (batch, 1 or heads, seq_len, head_dim) — local sliding-window key-values.
        sparse_kv: (batch, 1 or heads, sparse_seq_len, head_dim) — candidate KV pool.
        kv_indices: (batch, seq_len, topk) — global sparse_kv positions in dense mode,
            document-local positions with packed offsets.
        attention_sink: (heads,) — learned per-head sink weight.
        cu_seqlens: (num_documents + 1,) or None — cumulative packed query lengths.
        cu_seqlens_k: (num_documents + 1,) or None — cumulative packed candidate lengths.
        sliding_window_size: size of the causal sliding window.
        share_kv: if True, broadcast single-head KV and return single-head gradients.
        scale: Multiplier for query-key logits; does not scale sink logits.

    Returns:
        Tuple of (output, lse) where output has same shape as query and lse has
        shape (batch, heads, seq_len).
    """
    if query.device.type != "cuda":
        raise ValueError("The Triton gather attention backend requires CUDA tensors.")

    # Keep traceable normalization outside the opaque boundary so backward reuses each copy.
    query = query.contiguous()
    kv_indices = kv_indices.contiguous()
    attention_sink = attention_sink.contiguous()

    requires_grad = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, local_kv, sparse_kv, attention_sink)
    )
    if requires_grad:
        return _GatherAttnFunction.apply(
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            attention_sink,
            cu_seqlens,
            cu_seqlens_k,
            sliding_window_size,
            share_kv,
            scale,
        )

    output, lse, _, _ = _gather_attn_fwd_op(
        query,
        sparse_kv,
        local_kv,
        kv_indices,
        attention_sink,
        cu_seqlens,
        cu_seqlens_k,
        sliding_window_size,
        share_kv,
        scale,
        False,
    )
    return output, lse
