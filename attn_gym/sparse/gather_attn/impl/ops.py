"""Backend-neutral opaque operators and tensor tape for gather attention.

The graph specializes on a concrete backend, so each fake describes only that backend's
GPU tensor state. Returning the native buffers avoids packing/copying backward workspace;
no worst-case scratch reservation or CPU bookkeeping output is needed.
"""

import torch
from torch import Tensor

torch.library.define(
    "attn_gym::_gather_attn_fwd",
    "(Tensor query, Tensor sparse_kv, Tensor? local_kv, Tensor kv_indices, "
    "Tensor? attention_sink, Tensor? cu_seqlens, Tensor? cu_seqlens_k, "
    "int sliding_window_size, bool share_kv, float scale, bool needs_backward, "
    "str backend, bool bwd_recompute_p=True, Tensor? query_rope=None, "
    "Tensor? sparse_rope=None, Tensor? local_rope=None) -> (Tensor, Tensor, Tensor?[])",
)
torch.library.define(
    "attn_gym::_gather_attn_bwd",
    "(Tensor query, Tensor sparse_kv, Tensor? local_kv, Tensor kv_indices, "
    "Tensor?[] state, Tensor? attention_sink, "
    "Tensor? cu_seqlens, Tensor? cu_seqlens_k, Tensor output, Tensor lse, "
    "Tensor grad_output, int sliding_window_size, bool share_kv, float scale, "
    "str backend, bool bwd_recompute_p=True, Tensor? query_rope=None, "
    "Tensor? sparse_rope=None, Tensor? local_rope=None) "
    "-> (Tensor, Tensor, Tensor?, Tensor?, Tensor?, Tensor?, Tensor?)",
)


def _output_buffer(query: Tensor, backend: str) -> Tensor:
    batch, heads, tokens, dim = query.shape
    if backend == "cute":
        return query.new_empty((batch, tokens, heads, dim)).permute(0, 2, 1, 3).detach()
    return query.new_empty(query.shape)


def _triton_sink(query: Tensor, attention_sink: Tensor | None) -> Tensor:
    if attention_sink is None:
        return query.new_full((query.shape[1],), float("-inf"))
    return attention_sink.contiguous()


def _gather_attn_fwd_cuda(
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
    needs_backward: bool,
    backend: str,
    bwd_recompute_p: bool = True,
    query_rope: Tensor | None = None,
    sparse_rope: Tensor | None = None,
    local_rope: Tensor | None = None,
) -> tuple[Tensor, Tensor, list[Tensor | None]]:
    with torch.cuda.device(query.device):
        if backend not in ("cute", "triton"):
            raise ValueError(
                f"gather-attention operators require a concrete backend, got {backend!r}"
            )
        if sliding_window_size == 0 and kv_indices.shape[-1] == 0:
            output = _output_buffer(query, backend).zero_()
            if attention_sink is None:
                lse = query.new_full(query.shape[:3], float("-inf"), dtype=torch.float32)
            else:
                lse = (
                    attention_sink.float()[None, :, None]
                    .expand(query.shape[:3])
                    .clone(memory_format=torch.contiguous_format)
                )
            return output, lse, []
        state = ()
        if backend == "cute":
            from . import cute

            output, lse, state = cute._launch_forward(
                query,
                local_kv,
                sparse_kv,
                kv_indices,
                attention_sink,
                cu_seqlens,
                cu_seqlens_k,
                sliding_window_size,
                share_kv,
                scale,
                needs_backward,
                bwd_recompute_p,
                query_rope=query_rope,
                sparse_rope=sparse_rope,
                local_rope=local_rope,
            )
        else:
            from .triton.backward import _build_index_query_map
            from .triton.forward import _launch_forward

            query = query.contiguous()
            kv_indices = kv_indices.contiguous()
            sink = _triton_sink(query, attention_sink)
            if share_kv:
                sparse_kv = sparse_kv.expand(-1, query.shape[1], -1, -1)
                local_kv = (
                    None if local_kv is None else local_kv.expand(-1, query.shape[1], -1, -1)
                )
                sparse_rope = (
                    None if sparse_rope is None else sparse_rope.expand(-1, query.shape[1], -1, -1)
                )
                local_rope = (
                    None if local_rope is None else local_rope.expand(-1, query.shape[1], -1, -1)
                )
            output, lse = _launch_forward(
                query,
                sparse_kv,
                local_kv,
                kv_indices,
                sink,
                cu_seqlens,
                cu_seqlens_k,
                sliding_window_size,
                scale,
                query_rope=query_rope,
                sparse_rope=sparse_rope,
                local_rope=local_rope,
            )
            if needs_backward:
                # Only full-width backward uses the reverse map, including when determinism
                # is enabled after forward.
                state = (
                    _build_index_query_map(
                        kv_indices, sparse_kv.shape[2], cu_seqlens, cu_seqlens_k
                    )
                    if query_rope is None or query_rope.shape[-1] == 0
                    else (None, None)
                )
        return output, lse, list(state) if needs_backward else []


def _gather_attn_bwd_cuda(
    query: Tensor,
    sparse_kv: Tensor,
    local_kv: Tensor | None,
    kv_indices: Tensor,
    state: list[Tensor | None],
    attention_sink: Tensor | None,
    cu_seqlens: Tensor | None,
    cu_seqlens_k: Tensor | None,
    output: Tensor,
    lse: Tensor,
    grad_output: Tensor,
    sliding_window_size: int,
    share_kv: bool,
    scale: float,
    backend: str,
    bwd_recompute_p: bool = True,
    query_rope: Tensor | None = None,
    sparse_rope: Tensor | None = None,
    local_rope: Tensor | None = None,
) -> tuple[
    Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None
]:
    with torch.cuda.device(query.device):
        if sliding_window_size == 0 and kv_indices.shape[-1] == 0:
            return tuple(
                None if tensor is None else tensor.new_zeros(tensor.shape)
                for tensor in (
                    query,
                    sparse_kv,
                    local_kv,
                    query_rope,
                    sparse_rope,
                    local_rope,
                    attention_sink,
                )
            )
        # Autograd passes the forward's concrete choice, even if determinism changes.
        if backend == "cute":
            from . import cute

            return cute._launch_backward(
                query,
                local_kv,
                sparse_kv,
                attention_sink,
                cu_seqlens,
                cu_seqlens_k,
                output,
                lse,
                grad_output,
                state,
                scale,
                bwd_recompute_p,
                query_rope=query_rope,
                sparse_rope=sparse_rope,
                local_rope=local_rope,
            )
        from .triton.backward import _launch_backward

        selected_queries, block_offsets = state
        query = query.contiguous()
        kv_indices = kv_indices.contiguous()
        sink = _triton_sink(query, attention_sink)
        if share_kv:
            sparse_kv = sparse_kv.expand(-1, query.shape[1], -1, -1)
            local_kv = None if local_kv is None else local_kv.expand(-1, query.shape[1], -1, -1)
            sparse_rope = (
                None if sparse_rope is None else sparse_rope.expand(-1, query.shape[1], -1, -1)
            )
            local_rope = (
                None if local_rope is None else local_rope.expand(-1, query.shape[1], -1, -1)
            )
        dq, dsparse, dlocal, dqr, dsr, dlr, dsink = _launch_backward(
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            selected_queries,
            block_offsets,
            sink,
            cu_seqlens,
            cu_seqlens_k,
            output,
            lse,
            grad_output,
            sliding_window_size,
            share_kv,
            scale,
            query_rope=query_rope,
            sparse_rope=sparse_rope,
            local_rope=local_rope,
        )
        return dq, dsparse, dlocal, dqr, dsr, dlr, dsink if attention_sink is not None else None


torch.library.impl("attn_gym::_gather_attn_fwd", "CUDA", _gather_attn_fwd_cuda)
torch.library.impl("attn_gym::_gather_attn_bwd", "CUDA", _gather_attn_bwd_cuda)


@torch.library.register_fake("attn_gym::_gather_attn_fwd")
def _gather_attn_fwd_fake(
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
    needs_backward: bool,
    backend: str,
    bwd_recompute_p: bool = True,
    query_rope: Tensor | None = None,
    sparse_rope: Tensor | None = None,
    local_rope: Tensor | None = None,
) -> tuple[Tensor, Tensor, list[Tensor | None]]:
    if backend not in ("cute", "triton"):
        raise ValueError(f"gather-attention operators require a concrete backend, got {backend!r}")
    batch, heads, tokens, head_dim = query.shape
    sparse_tokens = sparse_kv.shape[2]
    output = _output_buffer(query, backend)
    state = []
    if needs_backward and (sliding_window_size != 0 or kv_indices.shape[-1] != 0):
        if backend == "triton":
            state = (
                [
                    kv_indices.new_empty((batch, tokens * kv_indices.shape[2]), dtype=torch.int32),
                    kv_indices.new_empty((batch, sparse_tokens + 1), dtype=torch.int32),
                ]
                if query_rope is None or query_rope.shape[-1] == 0
                else [None, None]
            )
        else:
            width = max((sliding_window_size + kv_indices.shape[2] + 127) // 128 * 128, 128)
            prefix = (batch, tokens) if cu_seqlens is None else (tokens,)
            residual = query.new_empty((batch, tokens, heads, head_dim))
            if cu_seqlens is None:
                # Match the native views, including singleton strides, without copying state.
                kv = query.new_empty(
                    (batch, 1, (0 if local_kv is None else tokens) + sparse_tokens, head_dim)
                ).permute(0, 2, 1, 3)
                cu_q = cu_kv = None
                if local_kv is None:
                    kv = None
            else:
                kv = query.new_empty(
                    ((0 if local_kv is None else tokens) + sparse_tokens, 1, head_dim)
                )
                cu_q = cu_seqlens.new_empty((cu_seqlens.numel() + 1,))
                cu_kv = cu_seqlens.new_empty((cu_seqlens.numel() + 1,))
                residual = residual.squeeze(0)
                if local_kv is None:
                    kv = None
            kv_rope = None
            if query_rope is not None and query_rope.shape[-1] != 0 and local_kv is not None:
                if cu_seqlens is None:
                    kv_rope = query.new_empty(
                        (batch, 1, tokens + sparse_tokens, query_rope.shape[-1])
                    ).permute(0, 2, 1, 3)
                else:
                    kv_rope = query.new_empty((tokens + sparse_tokens, 1, query_rope.shape[-1]))
            empty_pool = local_kv is None and sparse_tokens == 0
            state = [
                kv,
                kv_indices.new_empty((*prefix, width), dtype=torch.int32),
                cu_q,
                cu_kv,
                None
                if bwd_recompute_p or empty_pool
                else query.new_empty((*prefix, heads, width)),
                None
                if bwd_recompute_p or empty_pool
                else query.new_empty((*prefix, width // 128, heads), dtype=torch.float32),
                None if empty_pool else residual,
                kv_rope,
            ]
    return (
        output,
        query.new_empty((batch, heads, tokens), dtype=torch.float32),
        state,
    )


@torch.library.register_fake("attn_gym::_gather_attn_bwd")
def _gather_attn_bwd_fake(
    query: Tensor,
    sparse_kv: Tensor,
    local_kv: Tensor | None,
    kv_indices: Tensor,
    state: list[Tensor | None],
    attention_sink: Tensor | None,
    cu_seqlens: Tensor | None,
    cu_seqlens_k: Tensor | None,
    output: Tensor,
    lse: Tensor,
    grad_output: Tensor,
    sliding_window_size: int,
    share_kv: bool,
    scale: float,
    backend: str,
    bwd_recompute_p: bool = True,
    query_rope: Tensor | None = None,
    sparse_rope: Tensor | None = None,
    local_rope: Tensor | None = None,
) -> tuple[
    Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None, Tensor | None
]:
    return (
        query.new_empty(query.shape),
        sparse_kv.new_empty(sparse_kv.shape),
        None if local_kv is None else local_kv.new_empty(local_kv.shape),
        None if query_rope is None else query_rope.new_empty(query_rope.shape),
        None if sparse_rope is None else sparse_rope.new_empty(sparse_rope.shape),
        None if local_rope is None else local_rope.new_empty(local_rope.shape),
        None if attention_sink is None else attention_sink.new_empty(attention_sink.shape),
    )


_gather_attn_fwd_op = torch.ops.attn_gym._gather_attn_fwd.default
_gather_attn_bwd_op = torch.ops.attn_gym._gather_attn_bwd.default
