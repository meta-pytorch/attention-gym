"""Paged indexer key updates: pool completed blocks into candidates, keep the tail."""

import torch
from torch import Tensor

from attn_gym.types import Impl, resolve_impl

from .ops import _paged_indexer_key_update_op


def paged_indexer_key_update(
    raw_keys: Tensor,
    key_cache: Tensor,
    raw_cache: Tensor,
    block_table: Tensor,
    raw_block_table: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    norm_weight: Tensor,
    rope_cache: Tensor,
    *,
    compress_ratio: int = 4,
    eps: float = 1e-6,
    max_seqlen_q: int | None = None,
    impl: Impl | str = Impl.FUSED,
) -> None:
    """Store each newly completed block's pooled key and the incomplete block's raw keys.

    This is the cache-maintenance step for a paged indexer that scores pooled,
    completed-block candidates during decode.
    For new query ``i`` of ``n`` in sequence ``s``, at position
    ``p = seqused_k[s] - n + i``:

    - If ``(p + 1) % compress_ratio == 0``, the block ending at ``p`` is complete.
      Its keys at positions ``p - compress_ratio + 1 .. p`` come from ``raw_keys`` when
      they belong to this call and from ``raw_cache`` otherwise. They are averaged in
      FP32 and rounded to the key dtype, normalized with an offset RMSNorm
      ``(1 + norm_weight) * x / sqrt(mean(x^2) + eps)`` computed in FP32, rounded,
      rotated by rotate-half RoPE on the leading ``rotary_dim`` features at position
      ``p - compress_ratio + 1`` with cos/sin and each product and sum rounded to the
      key dtype, then stored as candidate ``p // compress_ratio`` in ``key_cache``.
    - If ``p >= seqused_k[s] // compress_ratio * compress_ratio``, the key belongs to
      the sequence's final incomplete block and is stored at slot
      ``p % ring_capacity`` of the sequence's ``raw_cache`` page.

    Every pooled member is read before any ``raw_cache`` slot is overwritten. Capacity
    rows at or beyond ``cu_seqlens_q[-1]`` and rows that complete no block store
    nothing.

    Args:
        raw_keys: ``[tokens, head_dim]`` FP16/BF16 keys of this call's queries, before
            pooling. Arbitrary strides.
        key_cache: ``[pages, candidate_page_size, head_dim]`` pooled candidate pool in
            the key dtype, updated in place. Candidate ``b`` of sequence ``s`` is at
            ``key_cache[block_table[s, b // candidate_page_size],
            b % candidate_page_size]``. Arbitrary strides.
        raw_cache: ``[ring_pages, ring_capacity, head_dim]`` per-sequence rings of raw
            keys in the key dtype, updated in place, with ``ring_capacity >=
            compress_ratio``. Arbitrary strides.
        block_table: int32 ``[num_sequences, max_candidate_pages]`` candidate pages.
        raw_block_table: int32 ``[num_sequences, columns >= 1]``; column 0 holds each
            sequence's ring page.
        cu_seqlens_q: Contiguous int32 ``[num_sequences + 1]`` packed query offsets.
        seqused_k: Contiguous int32 ``[num_sequences]`` raw-token counts including this
            call's queries.
        norm_weight: ``[head_dim]`` RMSNorm offset weight.
        rope_cache: FP32 ``[positions, 2 * rotary_dim]`` rows ``[cos | sin]`` in the
            rotate-half layout (each half repeats the ``rotary_dim / 2`` frequencies).
            ``rotary_dim`` is even and at most ``head_dim``.
        compress_ratio: Raw tokens per completed block.
        eps: RMSNorm epsilon.
        max_seqlen_q: Optional host bound on any sequence's query count. When it is
            below ``ring_capacity - 2``, the fused path updates both caches in one
            launch; otherwise it uses two.
        impl: ``"reference"`` runs eager PyTorch; ``"fused"`` runs Triton on CUDA.

    Metadata values are caller invariants, never read on the host by the fused path,
    which supports CUDA Graph capture with changing metadata values. Its stores are
    independent of the batch composition and match the reference, whose only freedom
    is the FP32 summation order of the RMSNorm variance (equal bits in the tests).
    """
    selected_impl = resolve_impl(impl)
    operands = {
        "raw_keys": raw_keys,
        "key_cache": key_cache,
        "raw_cache": raw_cache,
        "block_table": block_table,
        "raw_block_table": raw_block_table,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_k": seqused_k,
        "norm_weight": norm_weight,
        "rope_cache": rope_cache,
    }
    for name, tensor in operands.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if tensor.device != raw_keys.device:
            raise ValueError(f"{name} must be on raw_keys.device.")
    if not isinstance(compress_ratio, int) or isinstance(compress_ratio, bool):
        raise TypeError("compress_ratio must be a Python int.")
    if compress_ratio < 1:
        raise ValueError("compress_ratio must be positive.")
    if max_seqlen_q is not None and (
        not isinstance(max_seqlen_q, int) or isinstance(max_seqlen_q, bool) or max_seqlen_q < 0
    ):
        raise ValueError("max_seqlen_q must be a nonnegative Python int or None.")
    if raw_keys.ndim != 2 or raw_keys.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("raw_keys must be FP16/BF16 with shape [tokens, head_dim].")
    head_dim = raw_keys.shape[1]
    for name, cache in (("key_cache", key_cache), ("raw_cache", raw_cache)):
        if cache.ndim != 3 or cache.shape[2] != head_dim or cache.dtype != raw_keys.dtype:
            raise ValueError(f"{name} must have shape [pages, slots, head_dim] in the key dtype.")
    if raw_cache.shape[1] < compress_ratio:
        raise ValueError("raw_cache must hold at least compress_ratio slots per page.")
    if norm_weight.shape != (head_dim,):
        raise ValueError("norm_weight must have shape [head_dim].")
    if (
        rope_cache.ndim != 2
        or rope_cache.dtype != torch.float32
        or rope_cache.shape[1] % 4
        or rope_cache.shape[1] // 2 > head_dim
    ):
        raise ValueError(
            "rope_cache must be FP32 [positions, 2 * rotary_dim] with even rotary_dim."
        )
    sequences = seqused_k.shape[0]
    if seqused_k.ndim != 1 or cu_seqlens_q.shape != (sequences + 1,):
        raise ValueError("cu_seqlens_q and seqused_k must be [S + 1] and [S].")
    if block_table.ndim != 2 or block_table.shape[0] != sequences:
        raise ValueError("block_table must have shape [num_sequences, max_candidate_pages].")
    if raw_block_table.ndim != 2 or raw_block_table.shape[0] != sequences:
        raise ValueError("raw_block_table must have shape [num_sequences, columns >= 1].")
    for name in ("block_table", "raw_block_table", "cu_seqlens_q", "seqused_k"):
        tensor = operands[name]
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must have dtype int32.")
        if name in ("cu_seqlens_q", "seqused_k") and not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
    if selected_impl == Impl.REFERENCE:
        from .impl.key_update_reference import launch

        launch(
            raw_keys,
            key_cache,
            raw_cache,
            block_table,
            raw_block_table,
            cu_seqlens_q,
            seqused_k,
            norm_weight,
            rope_cache,
            compress_ratio,
            eps,
        )
        return
    if not raw_keys.is_cuda:
        raise ValueError("the fused paged_indexer_key_update requires CUDA tensors.")
    _paged_indexer_key_update_op(
        raw_keys,
        key_cache,
        raw_cache,
        block_table,
        raw_block_table,
        cu_seqlens_q,
        seqused_k,
        norm_weight,
        rope_cache,
        compress_ratio,
        eps,
        max_seqlen_q,
    )
