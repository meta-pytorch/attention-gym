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


def gather_attn(
    query: Tensor,
    local_kv: Tensor | None,
    sparse_kv: Tensor,
    kv_indices: Tensor,
    attention_sink: Tensor,
    cu_seqlens: Tensor | None,
    cu_seqlens_k: Tensor | None,
    sliding_window_size: int,
    share_kv: bool,
    *,
    scale: float,
    query_rope: Tensor | None = None,
    sparse_rope: Tensor | None = None,
    local_rope: Tensor | None = None,
    local_v: Tensor | None = None,
    sparse_v: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Gather actual slots, masking before dots so invalid NaNs cannot propagate.

    Query head h reads KV head h // (heads // kv_heads). ``local_v``/``sparse_v`` supply
    separate values for the same slots; without them the keys are also the values.
    """
    batch, heads, tokens, dim = query.shape
    positions = torch.arange(tokens, device=query.device)
    window = min(sliding_window_size, tokens)
    local_indices = positions[:, None] - window + 1 + torch.arange(window, device=query.device)
    local_valid = (local_indices >= 0).unsqueeze(0).expand(batch, -1, -1)
    local_indices = local_indices.expand(batch, -1, -1)
    offset = 0 if local_kv is None else tokens
    if cu_seqlens is None:
        sparse_valid = (kv_indices >= 0) & (kv_indices < sparse_kv.shape[2])
        sparse_indices = torch.where(sparse_valid, kv_indices, 0) + offset
    else:
        documents = torch.searchsorted(cu_seqlens[1:], positions, right=True)
        starts = cu_seqlens[documents]
        local_valid = local_valid & (local_indices >= starts[None, :, None])
        sparse_starts = cu_seqlens_k[documents][None, :, None]
        sparse_ends = cu_seqlens_k[(documents + 1).clamp(max=cu_seqlens_k.shape[0] - 1)][
            None, :, None
        ]
        sparse_valid = (kv_indices >= 0) & (kv_indices < sparse_ends - sparse_starts)
        sparse_indices = torch.where(sparse_valid, kv_indices, 0) + sparse_starts + offset
    sparse_indices = torch.where(sparse_valid, sparse_indices, 0)
    indices = torch.cat((local_indices, sparse_indices), dim=-1).long()
    valid = torch.cat((local_valid, sparse_valid), dim=-1)[:, None, :, :, None]
    kv_heads = sparse_kv.shape[1]
    accumulation_dtype = torch.promote_types(query.dtype, torch.float32)

    def gather_slots(local: Tensor | None, sparse: Tensor) -> Tensor:
        kv = sparse if local is None else torch.cat((local, sparse), dim=2)
        # A sentinel row keeps even a zero-capacity pool indexable without reading invalid KV.
        kv = torch.cat((kv, kv.new_zeros((batch, kv_heads, 1, kv.shape[-1]))), dim=2)
        safe_indices = torch.where(valid[:, 0, :, :, 0], indices, kv.shape[2] - 1)
        slots = kv[
            torch.arange(batch, device=query.device)[:, None, None, None],
            torch.arange(kv_heads, device=query.device)[None, :, None, None],
            safe_indices[:, None],
        ]
        return torch.where(valid, slots, 0).to(accumulation_dtype)

    def grouped_q(tensor: Tensor) -> Tensor:
        return (
            tensor.reshape(batch, kv_heads, heads // kv_heads, tokens, tensor.shape[-1])
            .permute(0, 1, 3, 2, 4)
            .to(accumulation_dtype)
        )

    slots = gather_slots(local_kv, sparse_kv)
    logits = torch.matmul(grouped_q(query), slots.transpose(-2, -1))
    if query_rope is not None:
        rope_slots = gather_slots(local_rope, sparse_rope)
        logits = logits + torch.matmul(grouped_q(query_rope), rope_slots.transpose(-2, -1))
    logits = (
        (logits * scale).permute(0, 1, 3, 2, 4).reshape(batch, heads, tokens, indices.shape[-1])
    )
    logits = logits.masked_fill(~valid.squeeze(-1), float("-inf"))
    probabilities, lse = _softmax_with_sink(logits, attention_sink)
    grouped_probabilities = (
        probabilities.to(query.dtype)
        .reshape(batch, kv_heads, heads // kv_heads, tokens, indices.shape[-1])
        .permute(0, 1, 3, 2, 4)
    )
    values = slots if sparse_v is None else gather_slots(local_v, sparse_v)
    output = torch.matmul(grouped_probabilities.to(accumulation_dtype), values)
    output = output.permute(0, 1, 3, 2, 4).reshape(batch, heads, tokens, dim)
    return output.to(query.dtype), lse
