from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import torch
from torch import Tensor

from attn_gym.types import AuxRequest, Impl, resolve_impl

from . import ops  # noqa: F401  # registers the attn_gym::paged_sparse_decode ops
from .impl import reference


@dataclass(frozen=True, slots=True)
class PagedKV:
    """A paged key/value source addressed by logical position through a block table.

    Attributes:
        key_cache: Keys ``[num_pages, page_size, kv_heads, head_dim]``.
        value_cache: Values ``[num_pages, page_size, kv_heads, head_dim_v]``. Latent (MLA)
            caches whose values are a prefix of each key pass a view such as
            ``key_cache[..., :head_dim_v]``; nothing is copied.
        block_table: int32 ``[num_sequences, max_pages]``; row ``s`` lists sequence ``s``'s
            pages, so logical entry ``e`` lives at page ``block_table[s, e // page_size]``,
            slot ``e % page_size``.
    """

    key_cache: Tensor
    value_cache: Tensor
    block_table: Tensor


@dataclass(frozen=True, slots=True)
class PagedSparseDecodeAux:
    """Auxiliary outputs returned by paged_sparse_decode when requested.

    Attributes:
        lse: Natural-log log-sum-exp of the scaled logits, including the attention sink,
            float32 ``[tokens, heads]``, or None if not requested.
    """

    lse: Tensor | None = None


def _validate_source(name: str, kv: PagedKV, query: Tensor, num_sequences: int) -> None:
    """Check one paged source's tensor metadata against the query."""
    key_cache, value_cache, block_table = kv.key_cache, kv.value_cache, kv.block_table
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError(
            f"{name}.key_cache and {name}.value_cache must have shapes "
            "[num_pages, page_size, kv_heads, head_dim], got "
            f"{list(key_cache.shape)} and {list(value_cache.shape)}."
        )
    if value_cache.shape[:3] != key_cache.shape[:3]:
        raise ValueError(
            f"{name}.value_cache must match {name}.key_cache in pages, page size, and heads, "
            f"got {list(value_cache.shape)} and {list(key_cache.shape)}."
        )
    if key_cache.shape[3] != query.shape[2]:
        raise ValueError(
            f"{name}.key_cache head_dim must match query, got {key_cache.shape[3]} and "
            f"{query.shape[2]}."
        )
    if min(key_cache.shape[1:]) <= 0 or value_cache.shape[3] <= 0:
        raise ValueError(f"{name} caches must have positive page size, heads, and head dims.")
    if query.shape[1] % key_cache.shape[2] != 0:
        raise ValueError(
            f"query heads ({query.shape[1]}) must be a multiple of {name} KV heads "
            f"({key_cache.shape[2]})."
        )
    if block_table.ndim != 2 or block_table.shape[0] != num_sequences:
        raise ValueError(
            f"{name}.block_table must have shape [{num_sequences}, max_pages], got "
            f"{list(block_table.shape)}."
        )
    if block_table.dtype != torch.int32:
        raise TypeError(f"{name}.block_table must be int32, got {block_table.dtype}.")
    for field, tensor in (
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("block_table", block_table),
    ):
        if field != "block_table" and tensor.dtype != query.dtype:
            raise ValueError(
                f"{name}.{field} must have the same dtype as query, got {tensor.dtype} and "
                f"{query.dtype}."
            )
        if tensor.device != query.device:
            raise ValueError(
                f"{name}.{field} must be on the same device as query, got {tensor.device} and "
                f"{query.device}."
            )
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name}.{field} must have a contiguous last dimension.")
        if tensor.requires_grad:
            raise ValueError(
                f"{name}.{field} requires grad, but paged_sparse_decode is forward-only."
            )


def _validate_inputs(
    query: Tensor,
    kv: PagedKV,
    indices: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    topk_length: Tensor | None,
    window_kv: PagedKV | None,
    attention_sink: Tensor | None,
    max_seqlen_q: int,
    index_block_size: int,
    window_size: int,
) -> None:
    """Validate tensor metadata without reading values."""
    if type(max_seqlen_q) is not int or max_seqlen_q < 1:
        raise ValueError(f"max_seqlen_q must be a positive Python int, got {max_seqlen_q!r}.")
    if type(index_block_size) is not int or index_block_size < 1:
        raise ValueError(
            f"index_block_size must be a positive Python int, got {index_block_size!r}."
        )
    if type(window_size) is not int or window_size < 0:
        raise ValueError(f"window_size must be a non-negative Python int, got {window_size!r}.")
    if query.ndim != 3 or min(query.shape[1:]) <= 0:
        raise ValueError(
            f"query must have shape [tokens, heads, head_dim], got {list(query.shape)}."
        )
    if not query.is_floating_point():
        raise TypeError(f"query must have a floating-point dtype, got {query.dtype}.")
    if query.stride(-1) != 1:
        raise ValueError("query must have a contiguous last dimension.")
    if query.requires_grad:
        raise ValueError("query requires grad, but paged_sparse_decode is forward-only.")

    if seqused_k.ndim != 1:
        raise ValueError(
            f"seqused_k must have shape [num_sequences], got {list(seqused_k.shape)}."
        )
    num_sequences = seqused_k.shape[0]
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.shape[0] != num_sequences + 1:
        raise ValueError(
            "cu_seqlens_q must have shape [num_sequences + 1] and seqused_k [num_sequences], "
            f"got {list(cu_seqlens_q.shape)} and {list(seqused_k.shape)}."
        )
    _validate_source("kv", kv, query, num_sequences)
    if window_kv is not None:
        _validate_source("window_kv", window_kv, query, num_sequences)
        if (
            window_kv.key_cache.shape[2] != kv.key_cache.shape[2]
            or window_kv.value_cache.shape[3] != kv.value_cache.shape[3]
        ):
            raise ValueError("window_kv must have the same KV heads and value head_dim as kv.")

    if indices.ndim != 2 or indices.shape[0] != query.shape[0]:
        raise ValueError(
            f"indices must have shape [{query.shape[0]}, topk], got {list(indices.shape)}."
        )
    metadata = [("indices", indices), ("cu_seqlens_q", cu_seqlens_q), ("seqused_k", seqused_k)]
    if topk_length is not None:
        if topk_length.shape != (query.shape[0],):
            raise ValueError(
                f"topk_length must have shape [{query.shape[0]}], got {list(topk_length.shape)}."
            )
        metadata.append(("topk_length", topk_length))
    for name, tensor in metadata:
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32, got {tensor.dtype}.")
        if tensor.device != query.device:
            raise ValueError(
                f"{name} must be on the same device as query, got {tensor.device} and "
                f"{query.device}."
            )
        if name != "indices" and not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
    if indices.shape[1] > 0 and indices.stride(1) != 1:
        raise ValueError("indices must have a contiguous last dimension.")

    if attention_sink is not None:
        if attention_sink.shape != (query.shape[1],):
            raise ValueError(
                f"attention_sink must have shape [{query.shape[1]}], got "
                f"{list(attention_sink.shape)}."
            )
        if attention_sink.dtype not in (query.dtype, torch.float32):
            raise TypeError(
                f"attention_sink must use the query dtype or float32, got {attention_sink.dtype}."
            )
        if attention_sink.device != query.device:
            raise ValueError("attention_sink must be on the same device as query.")
        if not attention_sink.is_contiguous():
            raise ValueError("attention_sink must be contiguous.")


@overload
def paged_sparse_decode(
    query: Tensor,
    kv: PagedKV,
    indices: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    *,
    max_seqlen_q: int,
    topk_length: Tensor | None = None,
    index_block_size: int = 1,
    window_size: int = 0,
    window_kv: PagedKV | None = None,
    attention_sink: Tensor | None = None,
    scale: float | None = None,
    impl: Impl | str = Impl.FUSED,
    return_aux: None = ...,
) -> Tensor: ...


@overload
def paged_sparse_decode(
    query: Tensor,
    kv: PagedKV,
    indices: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    *,
    max_seqlen_q: int,
    topk_length: Tensor | None = None,
    index_block_size: int = 1,
    window_size: int = 0,
    window_kv: PagedKV | None = None,
    attention_sink: Tensor | None = None,
    scale: float | None = None,
    impl: Impl | str = Impl.FUSED,
    return_aux: AuxRequest,
) -> tuple[Tensor, PagedSparseDecodeAux]: ...


def paged_sparse_decode(
    query: Tensor,
    kv: PagedKV,
    indices: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    *,
    max_seqlen_q: int,
    topk_length: Tensor | None = None,
    index_block_size: int = 1,
    window_size: int = 0,
    window_kv: PagedKV | None = None,
    attention_sink: Tensor | None = None,
    scale: float | None = None,
    impl: Impl | str = Impl.FUSED,
    return_aux: AuxRequest | None = None,
) -> Tensor | tuple[Tensor, PagedSparseDecodeAux]:
    """Attend decode queries to selected entries of a paged cache plus a causal window.

    Each query row attends, in one softmax, to the union of two key sets:

    - **Sparse set:** each entry of ``indices[t, :topk_length[t]]`` is a logical block id
      ``b`` that selects the ``index_block_size`` consecutive entries
      ``b * index_block_size, ...`` of ``kv``. ``-1`` (any negative value) is padding.
      Repeated selections keep their multiplicity.
    - **Window set:** the positions ``window_start .. p`` of the window source, where ``p`` is
      the query's position and ``window_start = max(p + 1 - window_size, 0)``.

    The window source is ``window_kv`` when given (for example a separate sliding-window cache
    next to a compressed sparse cache), otherwise ``kv`` itself. With a shared source the
    window start is rounded down to a multiple of ``index_block_size`` and selected entries
    at or after it are skipped, so every key is attended once and the incomplete block holding
    the query, which an indexer over complete blocks cannot select, is always included.

    The queries of each sequence are its last tokens: query ``i`` of a sequence with ``n``
    queries and ``k`` keys sits at position ``k - n + i``. Rows past ``cu_seqlens_q[-1]``
    produce undefined output. A row with no keys returns zeros.

    Model mappings:

    - Qwen QSA: ``index_block_size=4`` with the indexer's compressed-block ids,
      ``window_size=0``, separate ``key_cache`` / ``value_cache``, grouped KV heads.
    - DeepSeek V3.2 DSA: ``index_block_size=1`` token ids into the latent cache, values as
      ``key_cache[..., :512]``.
    - DeepSeek V4 / V4.1: compressed-entry ids into ``kv``, the raw sliding-window cache as
      ``window_kv`` with ``window_size=128``, plus ``attention_sink``.

    Forward-only: inputs must not require grad, and the output carries no autograd history.

    Args:
        query: Queries ``[tokens, heads, head_dim]``.
        kv: Paged source that ``indices`` address. See :class:`PagedKV`.
        indices: int32 ``[tokens, topk]`` logical block ids with a contiguous last dimension.
            With a separate ``window_kv`` they are not bounded by the query position: causality
            of the sparse set is the caller's (indexer's) contract.
        cu_seqlens_q: Contiguous int32 ``[num_sequences + 1]`` packed query offsets.
        seqused_k: Contiguous int32 ``[num_sequences]`` key count per sequence in the window
            source, including the current queries; it fixes each query's position.
        max_seqlen_q: Host-side upper bound on any sequence's query count, such as
            ``1 + num_speculative_tokens``. Queries past it produce undefined output.
        topk_length: Optional contiguous int32 ``[tokens]``; only the first
            ``topk_length[t]`` indices of row ``t`` are read, which lets the kernel skip the
            padding of short rows. Defaults to all ``topk`` columns.
        index_block_size: Consecutive entries selected by one index.
        window_size: Number of trailing positions in the window set; 0 disables it, except
            for the incomplete block of a shared source described above.
        window_kv: Optional separate paged source for the window set, with the same KV heads
            and head dims as ``kv``.
        attention_sink: Optional per-head logit ``[heads]`` (query dtype or float32) that adds
            ``exp(sink)`` to the softmax denominator and contributes no value. It is not
            scaled.
        scale: Positive logit multiplier. Defaults to ``1 / sqrt(head_dim)``.
        impl: ``Impl.FUSED`` (default) runs the Triton kernel on CUDA. ``Impl.REFERENCE`` runs
            eager PyTorch on CPU or CUDA.
        return_aux: If None (default), return only the output. If an AuxRequest, return
            ``(output, PagedSparseDecodeAux)`` with the requested outputs.

    Returns:
        The output ``[tokens, heads, head_dim_v]`` in the query dtype, or a tuple of it and
        ``PagedSparseDecodeAux`` when ``return_aux`` is given.
    """
    selected_impl = resolve_impl(impl)
    _validate_inputs(
        query,
        kv,
        indices,
        cu_seqlens_q,
        seqused_k,
        topk_length,
        window_kv,
        attention_sink,
        max_seqlen_q,
        index_block_size,
        window_size,
    )
    if scale is None:
        scale = query.shape[-1] ** -0.5
    if not scale > 0:
        raise ValueError(f"scale must be greater than 0, got {scale}.")

    window = (
        (None, None, None)
        if window_kv is None
        else (
            window_kv.key_cache,
            window_kv.value_cache,
            window_kv.block_table,
        )
    )
    args = (
        query,
        kv.key_cache,
        kv.value_cache,
        kv.block_table,
        indices,
        cu_seqlens_q,
        seqused_k,
        topk_length,
        *window,
        attention_sink,
    )
    return_lse = return_aux is not None and return_aux.lse
    with torch.no_grad():
        if selected_impl is Impl.REFERENCE:
            out, lse = reference.paged_sparse_decode(
                *args, index_block_size=index_block_size, window_size=window_size, scale=scale
            )
        else:
            if not query.is_cuda:
                raise ValueError("The fused paged_sparse_decode requires CUDA tensors.")
            options = (max_seqlen_q, index_block_size, window_size, scale)
            if return_lse:
                out, lse = torch.ops.attn_gym.paged_sparse_decode_with_lse.default(*args, *options)
            else:
                out, lse = torch.ops.attn_gym.paged_sparse_decode.default(*args, *options), None

    if return_aux is None:
        return out
    return out, PagedSparseDecodeAux(lse=lse if return_lse else None)
