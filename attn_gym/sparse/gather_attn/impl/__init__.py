"""Autograd integration for graph-specialized gather-attention backends."""

import torch
from torch import Tensor

from .ops import _gather_attn_bwd_op, _gather_attn_fwd_op


class _GatherAttnFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        sparse_kv: Tensor,
        local_kv: Tensor | None,
        kv_indices: Tensor,
        attention_sink: Tensor | None,
        cu_seqlens: Tensor | None,
        cu_seqlens_k: Tensor | None,
        sliding_window_size: int,
        share_kv: bool,
        scale: float,
        backend: str,
        bwd_recompute_p: bool,
        query_rope: Tensor | None,
        sparse_rope: Tensor | None,
        local_rope: Tensor | None,
        sparse_v: Tensor | None,
        local_v: Tensor | None,
        autotune: bool,
    ) -> tuple[Tensor, Tensor]:
        output, lse, state = _gather_attn_fwd_op(
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
            backend,
            bwd_recompute_p,
            query_rope,
            sparse_rope,
            local_rope,
            sparse_v,
            local_v,
            autotune=autotune,
        )
        ctx.save_for_backward(
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            attention_sink,
            cu_seqlens,
            cu_seqlens_k,
            output,
            lse,
            query_rope,
            sparse_rope,
            local_rope,
            sparse_v,
            local_v,
            *state,
        )
        ctx.options = (sliding_window_size, share_kv, scale, backend, bwd_recompute_p)
        ctx.autotune = autotune
        ctx.mark_non_differentiable(lse)
        return output, lse

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output: Tensor, grad_lse: Tensor):
        (
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            attention_sink,
            cu_seqlens,
            cu_seqlens_k,
            output,
            lse,
            query_rope,
            sparse_rope,
            local_rope,
            sparse_v,
            local_v,
            *state,
        ) = ctx.saved_tensors
        dq, ds, dl, dqr, dsr, dlr, dsink, dsv, dlv = _gather_attn_bwd_op(
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            state,
            attention_sink,
            cu_seqlens,
            cu_seqlens_k,
            output,
            lse,
            grad_output,
            *ctx.options,
            query_rope,
            sparse_rope,
            local_rope,
            sparse_v,
            local_v,
            autotune=ctx.autotune,
        )
        return (
            *(dq, ds, dl, None, dsink, None, None, None, None, None, None, None),
            *(dqr, dsr, dlr, dsv, dlv),
            None,
        )


def gather_attn(
    query: Tensor,
    local_kv: Tensor | None,
    sparse_kv: Tensor,
    kv_indices: Tensor,
    attention_sink: Tensor | None,
    cu_seqlens: Tensor | None,
    cu_seqlens_k: Tensor | None,
    sliding_window_size: int,
    share_kv: bool,
    *,
    scale: float,
    backend: str,
    bwd_recompute_p: bool = True,
    autotune: bool,
    query_rope: Tensor | None = None,
    sparse_rope: Tensor | None = None,
    local_rope: Tensor | None = None,
    local_v: Tensor | None = None,
    sparse_v: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Flatten independent operands into one common opaque operator contract.

    ``local_v``/``sparse_v`` are separate values; with them, or with 1 < kv_heads < heads,
    the inputs run on the Triton grouped schedule. ``autotune`` lets Triton kernels benchmark
    candidate configurations; the CuTe backend has one schedule and ignores it.
    """
    if backend == "triton" and (sliding_window_size or kv_indices.shape[-1]):
        query = query.contiguous()
    kv_indices = kv_indices.contiguous()
    if attention_sink is not None:
        attention_sink = attention_sink.contiguous()
    tensors = (query, sparse_kv, local_kv, kv_indices, attention_sink, cu_seqlens, cu_seqlens_k)
    ropes = (query_rope, sparse_rope, local_rope)
    options = (sliding_window_size, share_kv, scale, backend, bwd_recompute_p)
    values = (sparse_v, local_v)
    requires_grad = torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in (*tensors, *ropes, *values)
    )
    if requires_grad:
        # Distinct views avoid duplicate HOP inputs while preserving leaf accumulation.
        inputs = tuple(None if tensor is None else tensor.view_as(tensor) for tensor in tensors)
        rope_inputs = tuple(None if tensor is None else tensor.view_as(tensor) for tensor in ropes)
        value_inputs = tuple(
            None if tensor is None else tensor.view_as(tensor) for tensor in values
        )
        return _GatherAttnFunction.apply(*inputs, *options, *rope_inputs, *value_inputs, autotune)
    output, lse, _ = _gather_attn_fwd_op(
        *tensors,
        sliding_window_size,
        share_kv,
        scale,
        False,
        backend,
        bwd_recompute_p,
        *ropes,
        *values,
        autotune=autotune,
    )
    return output, lse
