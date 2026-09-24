"""Torch-only gather attention reference implementation."""

import torch
from torch import Tensor


def make_sliding_window_mask(
    query_length: int, window_size: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """
    Makes a mask for sliding window attention
    Args:
        query_length: length of query
        window_size: length of sliding window
        device: device to create tensors on
        dtype: dtype of the output mask
    Returns:
        A mask in shape (query_length, query_length), where valid attention positions are 0, invalid are -inf
    """
    query_positions = torch.arange(query_length, device=device)[:, None]
    key_positions = torch.arange(query_length, device=device)[None, :]
    valid = (key_positions <= query_positions) & (
        key_positions >= query_positions - window_size + 1
    )
    return torch.zeros(
        (query_length, query_length),
        device=device,
        dtype=dtype,
    ).masked_fill(~valid, float("-inf"))


def _softmax_with_sink(logits: Tensor, attention_sink: Tensor) -> tuple[Tensor, Tensor]:
    """Return real-key probabilities and sink-inclusive LSE, including empty rows."""
    sink = attention_sink.to(logits.dtype)[None, :, None, None]
    logits = torch.cat((logits, sink.expand(*logits.shape[:-1], 1)), dim=-1)
    empty = torch.isneginf(logits).all(dim=-1, keepdim=True)
    # Avoid the undefined derivative of logsumexp(all -inf), even with zero dLSE.
    safe_logits = torch.where(empty, 0, logits)
    probabilities = torch.where(empty, 0, torch.softmax(safe_logits, dim=-1)[..., :-1])
    lse = torch.where(empty.squeeze(-1), float("-inf"), torch.logsumexp(safe_logits, dim=-1))
    return probabilities, lse


def _packed_gather_attn(
    query: Tensor,
    local_kv: Tensor,
    sparse_kv: Tensor,
    kv_indices: Tensor,
    attention_sink: Tensor,
    cu_seqlens: Tensor,
    sliding_window_size: int,
    *,
    scale: float,
) -> tuple[Tensor, Tensor]:
    """Gather actual slots so masked KV (including NaNs) cannot contaminate other documents."""
    batch, heads, tokens, dim = query.shape
    positions = torch.arange(tokens, device=query.device)
    window = min(sliding_window_size, tokens)
    local_indices = positions[:, None] - window + 1 + torch.arange(window, device=query.device)
    local_valid = (local_indices >= 0).unsqueeze(0)
    documents = torch.searchsorted(cu_seqlens[1:], positions, right=True)
    starts = cu_seqlens[documents]
    local_valid = local_valid & (local_indices >= starts[:, None])
    local_indices = local_indices.clamp_min(0).expand(batch, -1, -1)
    indices = torch.cat((local_indices, (kv_indices + tokens).clamp_min(0)), dim=-1).long()
    valid = torch.cat((local_valid, kv_indices >= 0), dim=-1)[:, None, :, :, None]
    kv = torch.cat((local_kv, sparse_kv), dim=2)
    kv_heads = kv.shape[1]
    # Index the original pool rather than an expanded [B, H, T, pool, D] view:
    # gather's backward would otherwise allocate a gradient for that entire view.
    slots = kv[
        torch.arange(batch, device=query.device)[:, None, None, None],
        torch.arange(kv_heads, device=query.device)[None, :, None, None],
        indices[:, None],
    ]
    # Multiplication by a zero probability does not neutralize NaN values or gradients.
    slots = torch.where(valid, slots, 0)
    accumulation_dtype = torch.promote_types(query.dtype, torch.float32)
    slots_acc = slots.to(accumulation_dtype)
    # Share the gathered slots across query heads, including for the two matmuls.
    grouped_query = query.reshape(batch, kv_heads, heads // kv_heads, tokens, dim)
    grouped_query = grouped_query.permute(0, 1, 3, 2, 4).to(accumulation_dtype)
    logits = torch.matmul(grouped_query, slots_acc.transpose(-2, -1)) * scale
    logits = logits.permute(0, 1, 3, 2, 4).reshape(batch, heads, tokens, indices.shape[-1])
    logits = logits.masked_fill(~valid.squeeze(-1), float("-inf"))
    probabilities, lse = _softmax_with_sink(logits, attention_sink)
    probabilities = probabilities.to(query.dtype)
    grouped_probabilities = probabilities.reshape(
        batch, kv_heads, heads // kv_heads, tokens, indices.shape[-1]
    ).permute(0, 1, 3, 2, 4)
    output = torch.matmul(grouped_probabilities.to(accumulation_dtype), slots_acc)
    output = output.permute(0, 1, 3, 2, 4).reshape(batch, heads, tokens, dim)
    return output.to(query.dtype), lse


def gather_attn(
    query: Tensor,
    local_kv: Tensor,
    sparse_kv: Tensor,
    kv_indices: Tensor,
    attention_sink: Tensor,
    cu_seqlens: Tensor | None,
    sliding_window_size: int,
    share_kv: bool,
    *,
    scale: float,
) -> tuple[Tensor, Tensor]:
    """
    Performs gather attention as follows:
        if share_kv:
            expand local and sparse kv from (batch, 1, sequence_length, head_dim) to (batch, num_heads, sequence_length, head_dim)
        For each token, Q_i, in query:
            document = searchsorted(cu_seqlens[1:], i, right=True)
            first_token_of_document = cu_seqlens[document]
            farthest_past_token_index = max(i - sliding_window, first_token_of_document)
            KV = cat([local_kv[farthest_past_token_index: i + 1], sparse_kv[indices]])
            P = (Q @ KV.T) * scale
            P = softmax(cat([P, sink]))[:P.sequence_length]
            return P @ V

    Args:
        query: query, shaped like (batch_size, num_heads, sequence_length, head_dim)

        local_kv: Key and Value for the sliding window branch,
            represented as (batch_size, 1, sequence_length, head_dim) if share_kv is True.
            Otherwise represented as (batch_size, num_heads, sequence_length, head_dim).

        sparse_kv: KV candidate pool for the indexing branch, shape of (batch, 1, X, head_dim)
            if share_kv is True.
            Otherwise represented as (batch_size, num_heads, X, head_dim), where X may be zero.

        kv_indices: Which entries to select from sparse_kv.
            Shape of (batch, sequence_length, num_topk_blocks), integer tensor
            If less than num_topk_blocks should be indexed, pad the tensor with -1
            Duplicate indices will be upweighted based on the number of times they were duplicated


        attention_sink: tensor in shape of (num_heads, ), learnable per head weight that occupies denominator of softmax

        cu_seqlens: Packed query offsets, or None for ordinary batched inputs.
            They bound the local window. Sparse selections have already been translated
            to global pool positions and masked to their document by the public API.

        sliding_window_size: Integer, size of sliding window

        share_kv: bool, true iff all query heads attend to the same KV head
        scale: Multiplier for query-key logits; does not scale sink logits.
    Returns:
        Tuple of (output, lse) where output has shape (batch_size, num_heads, sequence_length,
        head_dim) and lse has shape (batch_size, num_heads, sequence_length).
    """
    device = query.device
    dtype = query.dtype
    accumulation_dtype = torch.promote_types(dtype, torch.float32)
    b, h, s, _head_dim = query.shape
    sparse_seq_len = sparse_kv.shape[2]
    if cu_seqlens is not None:
        return _packed_gather_attn(
            query,
            local_kv,
            sparse_kv,
            kv_indices,
            attention_sink,
            cu_seqlens,
            sliding_window_size,
            scale=scale,
        )
    if share_kv:
        local_kv = local_kv.expand(-1, h, -1, -1)
        sparse_kv = sparse_kv.expand(-1, h, -1, -1)
    counts = torch.zeros(b, s, sparse_seq_len, device=device, dtype=accumulation_dtype)
    if sparse_seq_len > 0:
        # We have s queries that each potentially attend to sparse_seq_len elements.
        # Indices of -1 are sentinel values meaning "no selection for this slot".
        # Repeated indices get extra weight (equivalent to multiple copies in the attention set).
        # This is specifically for edge case handling,
        # since most uses of this will have indices pass through torch.topk
        valid_mask = kv_indices >= 0
        safe_indices = kv_indices.clamp(min=0).long()
        # Count how many times each position is selected per query (ignoring sentinels)
        counts.scatter_add_(
            dim=-1,
            index=safe_indices,
            src=valid_mask.to(accumulation_dtype),
        )
    # Convert counts to additive log-mask: 0 selections → -inf, k selections → log(k).
    topk_mask = torch.where(counts > 0, torch.log(counts), float("-inf"))

    SWA_mask = make_sliding_window_mask(
        s, sliding_window_size, device, accumulation_dtype
    ).unsqueeze(0)
    SWA_mask = SWA_mask.expand(b, -1, -1)

    attention_kv = torch.cat([sparse_kv, local_kv], dim=-2)
    attention_mask = torch.cat([topk_mask, SWA_mask], dim=-1).unsqueeze(1)
    query_acc = query.to(accumulation_dtype)
    attention_kv_acc = attention_kv.to(accumulation_dtype)

    # Match the optimized backends' mixed-precision boundaries: accumulate QK and
    # softmax in FP32 for low-precision inputs, then quantize P for the PV dot.
    logits = (
        torch.matmul(query_acc, torch.permute(attention_kv_acc, (0, 1, 3, 2))) * scale
        + attention_mask
    )

    probs, lse = _softmax_with_sink(logits, attention_sink)
    probs = probs.to(dtype)

    # The low-precision P and KV operands accumulate in FP32 before the output
    # is stored in the input dtype, as in a tensor-core dot.
    output = torch.matmul(probs.to(accumulation_dtype), attention_kv_acc).to(dtype)
    return output, lse
