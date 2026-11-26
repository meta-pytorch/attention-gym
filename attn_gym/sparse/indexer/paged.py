"""Causal selection from persistent, paged pools of completed indexer keys."""

from typing import Literal, TypedDict

import torch
from torch import Tensor

from attn_gym.types import Impl, KernelOptionSpec, resolve_impl, validate_kernel_options

from .ops import _paged_indexer_op
from .validation import validate_precision


class PagedIndexerKernelOptions(TypedDict, total=False):
    """Implementation choices for :func:`paged_lightning_indexer`."""

    backend: Literal["cute", "triton"]

    batch_invariant: bool
    """Use fixed FP16/BF16 scoring and deterministic selection shared with the whole-document API."""


PAGED_KERNEL_OPTION_SPEC: KernelOptionSpec = {
    "backend": (("cute", "triton"), None),
    "batch_invariant": (bool, False),
}


def paged_lightning_indexer(
    query: Tensor,
    key_cache: Tensor,
    weights: Tensor,
    block_table: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    *,
    topk: int,
    compress_ratio: int = 4,
    query_positions: Tensor | None = None,
    q_scale: Tensor | None = None,
    k_scale: Tensor | None = None,
    impl: Impl | str = Impl.FUSED,
    kernel_options: PagedIndexerKernelOptions | None = None,
) -> Tensor:
    """Select completed blocks from cached candidates for each new query.

    Scores are ``sum_h(weights[t, h] * relu(query[t, h] @ key)) / sqrt(H * D)``,
    as in :func:`lightning_indexer`. Returns contiguous int32 ``[tokens, topk]``
    **sequence-local block IDs**, with -1 padding after the valid IDs. Order and
    tie-breaking are unspecified. Deterministic algorithms mode makes selection
    repeatable, with ties resolved toward lower IDs. Selection is nondifferentiable.
    Every new query requires selection even when the indexer weights are frozen.

    Args:
        query: Floating-point ``[tokens, heads, head_dim]`` new queries.
        key_cache: ``[pages, candidate_page_size, head_dim]`` pooled keys, shared
            across heads, in the query dtype. Logical candidate ``b`` is at
            ``key_cache[block_table[s, b // candidate_page_size],
            b % candidate_page_size]``. Pages count *candidates*, not raw tokens.
        weights: ``[tokens, heads]`` per-query weights, in the query dtype (BF16 or
            FP32 for MXFP8). Negative weights are supported.
        block_table: int32 ``[num_sequences, max_candidate_pages]`` page mapping.
            Referenced pages must be valid; unused entries may contain -1.
        cu_seqlens_q: Contiguous int32 ``[num_sequences + 1]`` packed query
            offsets, starting at zero, nondecreasing, ending at or before tokens.
            Empty sequences are supported. Capacity-tail rows return only -1.
        seqused_k: Contiguous int32 ``[num_sequences]`` counts of **raw tokens**,
            including the current queries. Query i of n appended queries has
            zero-based sequence position ``seqused_k[s] - n + i``. Only the
            first ``(position + 1) // compress_ratio`` candidates are visible,
            capped by ``seqused_k[s] // compress_ratio``.
        topk: Nonnegative output width; may exceed the number of visible blocks.
        compress_ratio: Positive raw-token count per completed block.
        query_positions: Optional contiguous int32 ``[tokens]`` absolute,
            zero-based sequence positions, replacing the appended-query rule.
            Negative positions select nothing.
        q_scale: For prequantized E4M3FN queries, E8M0 scales of shape
            ``[tokens, heads, head_dim // 32]``. Each factor scales its 32-channel
            group inside the dot, before per-head ReLU. No implicit quantization.
        k_scale: Corresponding E8M0 scales ``[pages, candidate_page_size, head_dim // 32]``
            for the key cache, using the same physical pages and slots as its FP8 data.
        impl: ``"reference"`` runs eager PyTorch on CPU or CUDA; ``"fused"``
            runs on NVIDIA CUDA devices (SM80+), supporting FP16/BF16,
            heads <= 128 and head_dim <= 256. Arbitrary nonnegative strides
            are supported for FP16/BF16. The native MXFP8 scorer requires SM100/103,
            compact H4/D128 inputs and 64-candidate pages. Its selector uses FP32 scores.
        kernel_options: ``backend="cute"`` (default on SM100-family GPUs) uses the
            Warpdrive radix selector, with Warpdrive scoring for eligible decode
            inputs and Triton scoring otherwise. ``"triton"`` uses Triton for both
            stages (default on other GPUs). ``batch_invariant=True`` fixes the FP16/BF16
            Triton scoring tiles and uses deterministic radix selection, independently of
            global deterministic mode. This matches batch-invariant whole-document indexing
            for identical query, weight and transformed key values on one GPU/software stack.
            Fused MXFP8 scoring does not support this mode. Reference validates and ignores
            these options.

    Offset values, nonnegative sequence lengths, cache capacity for completed
    blocks, and referenced page IDs are caller invariants, never read on the host
    by the fused path. It supports ``torch.compile(fullgraph=True)`` and CUDA
    Graph replay with changing metadata values.

    The caller owns cache updates: apply the model's pooling and transforms, then
    store one finished key per completed block before selection. Never expose a
    partial block as complete. Selection neither changes nor repools this cache.

    Feed these IDs directly to ``paged_sparse_decode`` with
    ``index_block_size=compress_ratio`` for raw-token KV, or 1 for pooled KV.
    A shared raw-token KV source supplies the incomplete causal tail even with
    ``window_size=0``. With pooled sparse KV and raw lengths, pass a separate raw
    ``window_kv`` and request the desired tail via ``window_size``. The KV page
    table is independent of this candidate page table.
    ``paged_sparse_decode`` infers suffix positions from its own metadata; the
    caller must ensure they match the positions used here. ``query_positions``
    overrides this indexer only.
    """
    selected_impl = resolve_impl(impl)
    options = validate_kernel_options(
        kernel_options, PAGED_KERNEL_OPTION_SPEC, op_name="paged_lightning_indexer"
    )
    operands = {
        "query": query,
        "key_cache": key_cache,
        "weights": weights,
        "block_table": block_table,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_k": seqused_k,
    }
    if query_positions is not None:
        operands["query_positions"] = query_positions
    for name, tensor in operands.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor.")
    for name, value in (("topk", topk), ("compress_ratio", compress_ratio)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be a Python int.")
    if topk < 0 or compress_ratio < 1:
        raise ValueError("topk must be nonnegative and compress_ratio must be positive.")
    if query.ndim != 3 or min(query.shape[1:]) <= 0:
        raise ValueError("query must have shape [tokens, heads > 0, head_dim > 0].")
    tokens, heads, dim = query.shape
    if key_cache.ndim != 3 or key_cache.shape[1] < 1 or key_cache.shape[2] != dim:
        raise ValueError("key_cache must have shape [pages, candidate_page_size > 0, head_dim].")
    if weights.shape != (tokens, heads):
        raise ValueError("weights must have shape [tokens, heads].")
    if query.dtype == torch.float8_e4m3fn or q_scale is not None or k_scale is not None:
        validate_precision(query, key_cache, weights, q_scale, k_scale)
    elif not query.is_floating_point():
        raise TypeError("query must have a floating-point dtype.")
    elif key_cache.dtype != query.dtype or weights.dtype != query.dtype:
        raise TypeError("query, key_cache, and weights must have the same dtype.")
    if seqused_k.ndim != 1:
        raise ValueError("seqused_k must have shape [num_sequences].")
    sequences = seqused_k.shape[0]
    if block_table.ndim != 2 or block_table.shape[0] != sequences:
        raise ValueError("block_table must have shape [num_sequences, max_candidate_pages].")
    if cu_seqlens_q.shape != (sequences + 1,):
        raise ValueError("cu_seqlens_q must have shape [num_sequences + 1].")
    if query_positions is not None and query_positions.shape != (tokens,):
        raise ValueError("query_positions must have shape [tokens].")
    for name, tensor in operands.items():
        if tensor.device != query.device:
            raise ValueError(f"{name} must be on query.device.")
        if name not in ("query", "key_cache", "weights"):
            if tensor.dtype != torch.int32:
                raise TypeError(f"{name} must have dtype int32.")
            if name != "block_table" and not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous.")
    if selected_impl == Impl.REFERENCE:
        from .impl.paged_reference import launch

        return launch(
            query,
            key_cache,
            weights,
            block_table,
            cu_seqlens_q,
            seqused_k,
            topk,
            compress_ratio,
            query_positions,
            q_scale,
            k_scale,
        )
    if not query.is_cuda:
        raise ValueError("the fused paged_lightning_indexer requires CUDA tensors.")
    return _paged_indexer_op(
        query,
        key_cache,
        weights,
        block_table,
        cu_seqlens_q,
        seqused_k,
        topk,
        compress_ratio,
        query_positions,
        options["backend"] or "auto",
        q_scale,
        k_scale,
        batch_invariant=options["batch_invariant"],
    )
