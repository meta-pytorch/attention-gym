from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict, overload

import torch
from torch import Tensor

from attn_gym.types import (
    AuxRequest,
    Impl,
    KernelOptionSpec,
    resolve_impl,
    validate_kernel_options,
)

from .impl import reference
from .ops import decode_op, decode_with_lse_op


class PagedSparseDecodeKernelOptions(TypedDict, total=False):
    batch_invariant: bool
    """Use fixed forward arithmetic shared with batch-invariant gather_attn. Defaults to False."""


KERNEL_OPTION_SPEC: KernelOptionSpec = {"batch_invariant": (bool, False)}


@dataclass(frozen=True, slots=True)
class PagedKV:
    """Key/value cache whose sequence-local entries map to physical pages and slots.

    Attributes:
        key_cache: Keys ``[num_pages, page_size, kv_heads, head_dim]``.
        value_cache: Values ``[num_pages, page_size, kv_heads, head_dim_v]``. Latent (MLA)
            caches whose values are a prefix of each key pass a view such as
            ``key_cache[..., :head_dim_v]``; nothing is copied.
        block_table: int32 ``[num_sequences, max_pages]`` physical page ids. Logical entry
            ``e`` of sequence ``s`` lives at page ``block_table[s, e // page_size]``,
            slot ``e % page_size``. These page ids are distinct from the block ids in
            ``paged_sparse_decode``'s ``indices``.
    """

    key_cache: Tensor
    value_cache: Tensor
    block_table: Tensor


@dataclass(frozen=True, slots=True)
class PagedSparseDecodeAux:
    """Optional softmax statistics from :func:`paged_sparse_decode`.

    Attributes:
        lse: Natural-log log-sum-exp of the scaled logits, including the attention sink,
            float32 ``[tokens, heads]``, or None if not requested.
    """

    lse: Tensor | None = None


def _tensors(kv: PagedKV | None) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    return (None,) * 3 if kv is None else (kv.key_cache, kv.value_cache, kv.block_table)


def _check_operand(name: str, tensor: Tensor, device: torch.device, field: str = "") -> None:
    # Validation runs on every decode call, so callers read query.device once and the full
    # name is formatted only on failure.
    if tensor.device != device:
        raise ValueError(
            f"{name}{field} must be on the same device as query, got {tensor.device} and {device}."
        )
    # A last dimension of size 0 or 1 is never strided over, so its stride is irrelevant.
    if tensor.stride(-1) != 1 and tensor.shape[-1] > 1:
        raise ValueError(f"{name}{field} must have a contiguous last dimension.")
    if tensor.requires_grad:
        raise ValueError(f"{name}{field} requires grad, but paged_sparse_decode is forward-only.")


def _validate_source(name: str, kv: PagedKV, query: Tensor, num_sequences: int) -> None:
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
    device = query.device
    for field, tensor in ((".key_cache", key_cache), (".value_cache", value_cache)):
        if tensor.dtype != query.dtype:
            raise ValueError(
                f"{name}{field} must have the same dtype as query, got {tensor.dtype} and "
                f"{query.dtype}."
            )
        _check_operand(name, tensor, device, field)
    _check_operand(name, block_table, device, ".block_table")


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
    """Validate metadata only: reading tensor contents would synchronize CUDA execution."""
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
    device = query.device
    _check_operand("query", query, device)

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
        _check_operand(name, tensor, device)

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
        _check_operand("attention_sink", attention_sink, device)


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
    kernel_options: PagedSparseDecodeKernelOptions | None = None,
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
    kernel_options: PagedSparseDecodeKernelOptions | None = None,
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
    kernel_options: PagedSparseDecodeKernelOptions | None = None,
    return_aux: AuxRequest | None = None,
) -> Tensor | tuple[Tensor, PagedSparseDecodeAux]:
    """Attend decode queries to selected entries of a paged cache plus a causal window.

    Each query row attends to two key sets in one softmax:

    - **Sparse set:** each block id ``b`` in ``indices[t, :topk_length[t]]`` selects the
      ``index_block_size`` consecutive logical entries starting at ``b * index_block_size``
      in ``kv``. Negative block ids are padding. Repeated selections retain their multiplicity.
    - **Window set:** positions ``window_start`` through ``p`` (inclusive) of the window
      source, where ``p`` is the query position and
      ``window_start = max(p + 1 - window_size, 0)`` before shared-source alignment.

    The window source is ``window_kv`` when supplied, otherwise ``kv``:

    - **Shared source:** round ``window_start`` down to a multiple of ``index_block_size``
      and discard sparse entries at or after that boundary. This removes sparse/window
      overlap, not duplicate sparse selections. The window includes the causal part of the
      query's incomplete block even when ``window_size=0``; an indexer selecting only complete
      blocks cannot supply that part.
    - **Separate sources:** do not align the window start or filter sparse entries by query
      position. The caller (usually the indexer) is responsible for sparse causality.
      Sparse and window entries contribute separately, even if the sources alias.

    A sequence's queries are its last tokens: query ``i`` (zero-based) of a sequence with
    ``n`` queries and ``k = seqused_k[s]`` keys has position ``p = k - n + i`` in the window
    source. Negative positions have no keys, including sparse keys. Rows at or beyond
    ``cu_seqlens_q[-1]`` produce undefined output. A row with no keys returns zeros; its LSE
    is the sink logit if supplied, otherwise ``-inf``.

    Forward-only: inputs must not require grad, and the output carries no autograd history.

    Args:
        query: Queries ``[tokens, heads, head_dim]``. Each KV head serves a consecutive group
            of ``heads // kv_heads`` query heads; ``heads`` must be divisible by ``kv_heads``.
        kv: Paged source that ``indices`` address. See :class:`PagedKV`.
        indices: int32 ``[tokens, topk]`` sequence-local block ids with a contiguous last
            dimension.
        cu_seqlens_q: Contiguous int32 ``[num_sequences + 1]`` packed query offsets.
        seqused_k: Contiguous int32 ``[num_sequences]`` key count per sequence within the
            window source, including the current queries; it determines their positions.
        max_seqlen_q: Host-side upper bound on any sequence's query count, such as
            ``1 + num_speculative_tokens``. Queries past it produce undefined output.
        topk_length: Optional contiguous int32 ``[tokens]`` prefix lengths, clamped to
            ``[0, topk]``. Only this prefix of each row is read, so short rows can skip trailing
            padding. Defaults to all ``topk`` columns.
        index_block_size: Number of consecutive logical entries selected by one block id.
        window_size: Requested number of trailing positions in the window set, before
            shared-source alignment. Zero disables the window except for the incomplete block
            of a shared source described above.
        window_kv: Optional separate paged source for the window set, with the same KV heads
            and head dims as ``kv``.
        attention_sink: Optional per-head logit ``[heads]`` (query dtype or float32) that adds
            ``exp(sink)`` to the softmax denominator and contributes no value. It is not
            scaled.
        scale: Positive logit multiplier. Defaults to ``1 / sqrt(head_dim)``.
        impl: ``Impl.FUSED`` (default) runs the Triton kernel on CUDA. ``Impl.REFERENCE`` runs
            eager PyTorch on CPU or CUDA.
        kernel_options: ``{"batch_invariant": True}`` uses fixed tiles without
            split reductions, independent of batch size and graph padding. It preserves
            output/LSE bits and matches batch-invariant ``gather_attn`` for the same
            ordered selected tokens followed by the window, Q/K/V, scale and sink.
            Head dimensions must be <= 512. Defaults to False. Reference validates and
            ignores this option. The guarantee assumes one GPU model and software stack.
        return_aux: If None (default), return only the output. If an AuxRequest, return
            ``(output, PagedSparseDecodeAux)`` with the requested outputs.

    Returns:
        The output ``[tokens, heads, head_dim_v]`` in the query dtype, or a tuple of it and
        ``PagedSparseDecodeAux`` when ``return_aux`` is given.
    """
    selected_impl = resolve_impl(impl)
    options = validate_kernel_options(
        kernel_options, KERNEL_OPTION_SPEC, op_name="paged_sparse_decode"
    )
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

    return_lse = return_aux is not None and return_aux.lse
    with torch.no_grad():
        if selected_impl is Impl.REFERENCE:
            out, lse = reference.paged_sparse_decode(
                query,
                kv,
                indices,
                cu_seqlens_q,
                seqused_k,
                topk_length,
                window_kv,
                attention_sink,
                index_block_size=index_block_size,
                window_size=window_size,
                scale=scale,
            )
        else:
            if not query.is_cuda:
                raise ValueError("The fused paged_sparse_decode requires CUDA tensors.")
            args = (
                query,
                *_tensors(kv),
                indices,
                cu_seqlens_q,
                seqused_k,
                topk_length,
                *_tensors(window_kv),
                attention_sink,
                max_seqlen_q,
                index_block_size,
                window_size,
                scale,
            )
            if return_lse:
                out, lse = decode_with_lse_op(*args, batch_invariant=options["batch_invariant"])
            else:
                out, lse = decode_op(*args, batch_invariant=options["batch_invariant"]), None

    if return_aux is None:
        return out
    return out, PagedSparseDecodeAux(lse=lse if return_lse else None)
