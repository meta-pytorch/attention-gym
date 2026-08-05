"""Torch-only compressed sparse attention reference implementation."""

import torch
from torch import Tensor


def make_sliding_window_mask(query_length: int, window_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
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


def sink_softmax(x: Tensor, sink: Tensor, dim: int) -> Tensor:
    """
    Applies a softmax with an attention sink.
    The sink contributes to the demoninator but not the numerator, so it is not returned
    Computes Y, where Y[a, b, c, d] = exp(x[a, b, c, d]) / (sum(exp(x[a, b, c, :]) + exp(sink[b])))
    Args:
        x: shape of (batch, num_heads, sequence, dim)
        sink: shape of (num_heads, )
        dim: dimension to apply softmax on
    Returns:
        Y, same shape as X
    """
    sink = sink[None, :, None, None]
    maximums = torch.max(x, dim=dim, keepdim=True).values
    maximums = torch.maximum(maximums, sink)
    x = x - maximums
    sink = sink - maximums
    x = torch.exp(x)
    return x / (torch.sum(x, dim, keepdim=True) + torch.exp(sink))


def make_packed_mask(
    doc_ids: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Creates an attention mask that prevents documents from attenting across boundaries

    Args:
        doc_ids: Has shape (batch, sequence) and integer dtype.
            Tokens in the same document have the same ID and are allowed to attention to each other.
            Tokens from documents recieve -inf for the attention mask, and tokens from the same document receive 0

    Returns:
        Additive attention mask of shape (batch, 1, sequence, sequence)
    """
    batch_size, seq_len = doc_ids.shape
    device = doc_ids.device

    # [B, S, 1] == [B, 1, S] -> [B, S_query, S_key]
    same_document = doc_ids[:, :, None] == doc_ids[:, None, :]

    mask = torch.full(
        (batch_size, seq_len, seq_len),
        float("-inf"),
        dtype=dtype,
        device=device,
    )
    mask.masked_fill_(same_document, 0.0)

    # Head-broadcasting dimension.
    return mask[:, None, :, :]


def selected_attention(
    query: Tensor,
    local_kv: Tensor,
    sparse_kv: Tensor,
    kv_indices: Tensor | None,
    attention_sink: Tensor,
    doc_ids: Tensor | None,
    sliding_window_size: int,
    share_kv: bool,
) -> Tensor:
    """
    Args:
        query: query, shaped like (batch_size, num_heads, sequence_length, head_dim)

        local_kv: Key and Value for the sliding window branch,
            represented as a shared tensor, (batch_size, 1, sequence_length, head_dim) if share_kv = False
            Otherwise represented as (batch_size, num_heads, sequence_length, head_dim)

        sparse_kv: KV candidate pool for the indexing branch, shape of (batch, 1, X, head_dim)
            if share_kv = False
            Otherwise represented as (batch_size, num_heads, X, head_dim), where X can be any nonzero integer

        kv_indices: Which entries to select from sparse_kv.
            Shape of (batch, sequence_length, num_topk_blocks), integer tensor
            If None, sparse_kv will be ignored
            If less than num_topk_blocks should be indexed, pad the tensor with -1
            Duplicate indices will be computed multiple times.


        attention_sink: tensor in shape of (num_heads, ), learnable per head weight that occupies denominator of softmax

        doc_ids: Integer tensor in shape of (batch_size, sequence_length) or None.
            Looks something like [0, 0, 0, 1, 1, 2, 2, 2, 2], where all tokens with the same id can causally attend to each other
            If doc_ids[i, j] = doc_ids[i, j-y], then query[i, j] can causally attend to local_kv[i, j-y]
            Should be monotonically increasing on the sequence axis
            If doc_ids is None, all tokens on the same sequence axis will be assumed to be in the same document.

        sliding_window_size: Integer, size of sliding window

        share_kv: bool, true iff all query heads attend to the same KV head
    Returns:
        Tensor in shape of (batch_size, num_heads, sequence_length, head_dim)
    """
    device = query.device
    dtype = query.dtype
    b, h, s, head_dim = query.shape
    sparse_seq_len = sparse_kv.shape[2]
    if share_kv:
        local_kv = local_kv.expand(-1, h, -1, -1)
        sparse_kv = sparse_kv.expand(-1, h, -1, -1)
    if kv_indices is not None:
        # We have s queries that each potentially attend to sparse_seq_len elements.
        # Indices of -1 are sentinel values meaning "no selection for this slot".
        # Repeated indices get extra weight (equivalent to multiple copies in the attention set).
        # This is specifically for edge case handling,
        # since most uses of this will have indices pass through torch.topk
        valid_mask = kv_indices >= 0
        safe_indices = kv_indices.clamp(min=0)
        # Count how many times each position is selected per query (ignoring sentinels)
        counts = torch.zeros(b, s, sparse_seq_len, device=device, dtype=dtype)
        counts.scatter_add_(
            dim=-1,
            index=safe_indices,
            src=valid_mask.to(dtype),
        )
        # Convert counts to additive log-mask: 0 selections → -inf, k selections → log(k)
        topk_mask = torch.where(
            counts > 0, torch.log(counts), torch.full_like(counts, float("-inf"))
        )

    SWA_mask = make_sliding_window_mask(s, sliding_window_size, device, dtype).unsqueeze(0)
    SWA_mask = SWA_mask.expand(b, -1, -1)

    if doc_ids is not None:
        packing_mask = make_packed_mask(doc_ids, dtype=dtype)
        # packing_mask is [B, 1, S, S], SWA_mask is [B, S, S]
        SWA_mask = SWA_mask + packing_mask.squeeze(1)

    if kv_indices is not None:
        attention_kv = torch.cat([sparse_kv, local_kv], dim=-2)
        attention_mask = torch.cat([topk_mask, SWA_mask], dim=-1).unsqueeze(1)
    else:
        attention_kv = local_kv
        attention_mask = SWA_mask
    scale = head_dim**0.5

    P = sink_softmax(
        torch.matmul(query, torch.permute(attention_kv, (0, 1, 3, 2))) / scale + attention_mask,
        attention_sink,
        dim=-1,
    )
    attn_output = P @ attention_kv
    return attn_output
