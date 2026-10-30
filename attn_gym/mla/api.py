from __future__ import annotations

from dataclasses import dataclass
from typing import overload

import torch
from torch import Tensor

from attn_gym.types import AuxRequest, Impl, resolve_impl


@dataclass(frozen=True, slots=True)
class PagedMLADecodeAux:
    """Auxiliary outputs returned by paged_mla_decode when requested.

    Attributes:
        lse: Natural-log log-sum-exp of the scaled logits, float32 ``[tokens, heads]``,
            or None if not requested.
    """

    lse: Tensor | None = None


def _validate_inputs(
    q_latent: Tensor,
    q_rope: Tensor,
    latent_cache: Tensor,
    rope_cache: Tensor,
    block_table: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    max_seqlen_q: int,
) -> None:
    """Validate tensor metadata without reading values."""
    if type(max_seqlen_q) is not int or max_seqlen_q < 1:
        raise ValueError(f"max_seqlen_q must be a positive Python int, got {max_seqlen_q!r}.")
    if q_latent.ndim != 3 or q_rope.ndim != 3 or q_latent.shape[:2] != q_rope.shape[:2]:
        raise ValueError(
            "q_latent and q_rope must have shapes [tokens, heads, latent_dim] and "
            f"[tokens, heads, rope_dim], got {list(q_latent.shape)} and {list(q_rope.shape)}."
        )
    if (
        latent_cache.ndim != 3
        or rope_cache.ndim != 3
        or latent_cache.shape[:2] != rope_cache.shape[:2]
        or latent_cache.shape[2] != q_latent.shape[2]
        or rope_cache.shape[2] != q_rope.shape[2]
    ):
        raise ValueError(
            "latent_cache and rope_cache must have shapes [num_blocks, block_size, latent_dim] "
            f"and [num_blocks, block_size, rope_dim], got {list(latent_cache.shape)} and "
            f"{list(rope_cache.shape)}."
        )
    if not q_latent.is_floating_point():
        raise TypeError(f"q_latent must have a floating-point dtype, got {q_latent.dtype}.")
    for name, tensor in (
        ("q_rope", q_rope),
        ("latent_cache", latent_cache),
        ("rope_cache", rope_cache),
    ):
        if tensor.dtype != q_latent.dtype:
            raise ValueError(
                f"{name} must have the same dtype as q_latent, got {tensor.dtype} and "
                f"{q_latent.dtype}."
            )

    for name, tensor in (
        ("q_latent", q_latent),
        ("q_rope", q_rope),
        ("latent_cache", latent_cache),
        ("rope_cache", rope_cache),
        ("block_table", block_table),
    ):
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have a contiguous last dimension.")

    for name, tensor in (
        ("q_latent", q_latent),
        ("q_rope", q_rope),
        ("latent_cache", latent_cache),
        ("rope_cache", rope_cache),
    ):
        if tensor.requires_grad:
            raise ValueError(
                f"{name} requires grad, but paged_mla_decode is forward-only; detach it first."
            )

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
    if block_table.ndim != 2 or block_table.shape[0] != num_sequences:
        raise ValueError(
            f"block_table must have shape [{num_sequences}, max_blocks_per_sequence], "
            f"got {list(block_table.shape)}."
        )
    for name, tensor in (
        ("block_table", block_table),
        ("cu_seqlens_q", cu_seqlens_q),
        ("seqused_k", seqused_k),
    ):
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must be int32, got {tensor.dtype}.")
    for name, tensor in (("cu_seqlens_q", cu_seqlens_q), ("seqused_k", seqused_k)):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")

    for name, tensor in (
        ("q_rope", q_rope),
        ("latent_cache", latent_cache),
        ("rope_cache", rope_cache),
        ("block_table", block_table),
        ("cu_seqlens_q", cu_seqlens_q),
        ("seqused_k", seqused_k),
    ):
        if tensor.device != q_latent.device:
            raise ValueError(
                f"{name} must be on the same device as q_latent, got {tensor.device} and "
                f"{q_latent.device}."
            )


@overload
def paged_mla_decode(
    q_latent: Tensor,
    q_rope: Tensor,
    latent_cache: Tensor,
    rope_cache: Tensor,
    block_table: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    *,
    max_seqlen_q: int,
    scale: float,
    impl: Impl | str = Impl.FUSED,
    return_aux: None = ...,
) -> Tensor: ...


@overload
def paged_mla_decode(
    q_latent: Tensor,
    q_rope: Tensor,
    latent_cache: Tensor,
    rope_cache: Tensor,
    block_table: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    *,
    max_seqlen_q: int,
    scale: float,
    impl: Impl | str = Impl.FUSED,
    return_aux: AuxRequest,
) -> tuple[Tensor, PagedMLADecodeAux]: ...


def paged_mla_decode(
    q_latent: Tensor,
    q_rope: Tensor,
    latent_cache: Tensor,
    rope_cache: Tensor,
    block_table: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    *,
    max_seqlen_q: int,
    scale: float,
    impl: Impl | str = Impl.FUSED,
    return_aux: AuxRequest | None = None,
) -> Tensor | tuple[Tensor, PagedMLADecodeAux]:
    """Attend packed queries to a paged, compressed MLA cache without expanding it.

    This is the absorbed ("latent space") form of multi-head latent attention used for
    decode. The cache stores one latent row and one shared rope key per token, for all
    heads. Each head attends with

        softmax(scale * (q_latent @ latent^T + q_rope @ rope^T)) @ latent,

    so the latent rows act as both keys and values. ``q_latent`` is the per-head nope query
    already multiplied by that head's key up-projection, and the returned output is still
    in latent space: the caller applies the value up-projection.

    Forward-only: inputs must not require grad, and the output carries no autograd history.

    The queries of each sequence are its last tokens: query ``i`` of a sequence with ``n``
    queries and ``k`` cached keys attends to keys ``0 .. k - n + i``. Multi-token rows
    (speculative decoding, short prefills) use the same rule.

    Args:
        q_latent: Absorbed nope queries ``[tokens, heads, latent_dim]``.
        q_rope: Rope queries ``[tokens, heads, rope_dim]``.
        latent_cache: Paged latent rows ``[num_blocks, block_size, latent_dim]``.
        rope_cache: Paged shared rope keys ``[num_blocks, block_size, rope_dim]``.
            Both caches may be strided views of one ``[..., latent_dim + rope_dim]`` cache;
            only their last dimension must be contiguous, so neither is copied.
        block_table: int32 ``[num_sequences, max_blocks_per_sequence]`` with a contiguous
            last dimension; row ``s`` lists the cache blocks of sequence ``s`` in order.
        cu_seqlens_q: Contiguous int32 ``[num_sequences + 1]`` packed query offsets.
        seqused_k: Contiguous int32 ``[num_sequences]`` cached key count per sequence,
            including the keys of the current queries.
        max_seqlen_q: Host-side upper bound on any sequence's query count.
        scale: Positive multiplier for the logits. There is no default: the right value
            is the model's ``1 / sqrt(qk_nope_head_dim + qk_rope_head_dim)`` (times any
            rope scaling correction), not ``1 / sqrt(latent_dim + rope_dim)``.
        impl: ``Impl.FUSED`` (default) runs FlashAttention-4's absorbed MLA kernel through a
            registered operator that works under ``torch.compile``; ``Impl.REFERENCE`` runs
            eager PyTorch on CPU or CUDA.
        return_aux: If None (default), return only the output tensor. If an AuxRequest
            instance, return ``(output, PagedMLADecodeAux)`` with the requested outputs.

    Returns:
        If return_aux is None: the output, shaped like ``q_latent``. Otherwise a tuple of
        the output and ``PagedMLADecodeAux``. Rows past ``cu_seqlens_q[-1]`` are undefined.
    """
    selected_impl = resolve_impl(impl)
    _validate_inputs(
        q_latent,
        q_rope,
        latent_cache,
        rope_cache,
        block_table,
        cu_seqlens_q,
        seqused_k,
        max_seqlen_q,
    )
    if not scale > 0:
        raise ValueError(f"scale must be greater than 0, got {scale}.")

    return_lse = return_aux is not None and return_aux.lse
    with torch.no_grad():
        if selected_impl is Impl.REFERENCE:
            from .impl import reference

            out, lse = reference.paged_mla_decode(
                q_latent,
                q_rope,
                latent_cache,
                rope_cache,
                block_table,
                cu_seqlens_q,
                seqused_k,
                scale=scale,
            )
        else:
            from .impl import cute

            out, lse = cute.paged_mla_decode(
                q_latent,
                q_rope,
                latent_cache,
                rope_cache,
                block_table,
                cu_seqlens_q,
                seqused_k,
                max_seqlen_q=max_seqlen_q,
                scale=scale,
            )

    if return_aux is None:
        return out

    return out, PagedMLADecodeAux(lse=lse if return_lse else None)
