"""Private PyTorch launcher for the public ``lightning_indexer`` API."""

import math

import torch
from torch import Tensor


def launch(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    topk: int,
    causal: bool,
    compress_ratio: int,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_k: Tensor | None = None,
) -> Tensor:
    """Multi-head weighted ReLU Top-K, reference implementation.

    Computes:
        Attention matrix between q and k, reduces across heads (sum of scaled relu),
        then takes topk q_i* k_j attention score positions for each q_i


    Args:
        q: [B, T, H, D]
        k: [B, S, D]
        weights: [B, T, H]
        topk: number of candidates to select per query
        causal: keep only the ``(t + 1) // compress_ratio`` leading candidates of query t
        compress_ratio: tokens summarized per candidate; ``S == T // compress_ratio``
            for nonpacked calls.
        cu_seqlens: Packed query offsets, [N + 1]; supplied together with cu_seqlens_k.
        cu_seqlens_k: Packed candidate offsets, [N + 1].

    Returns:
        [B, T, topk] INT32 tensor of selected candidate indices.
    """
    batch, queries, heads, head_dim = q.shape
    candidates = k.shape[1]
    if topk == 0 or candidates == 0:
        return torch.full((batch, queries, topk), -1, dtype=torch.int32, device=q.device)
    scale = 1.0 / math.sqrt(heads * head_dim)

    # Accumulate scoring in FP32 to match the CuTe backend's accumulation
    # contract, regardless of the input storage dtype (e.g. FP16/BF16 can
    # overflow the dot product or produce NaNs from mixed-sign weights before
    # scaling). FP64 inputs are left at FP64 rather than downcast.
    accum_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q = q.to(accum_dtype)
    k = k.to(accum_dtype)
    weights = weights.to(accum_dtype)

    # dots: [B, T, H, S]
    dots = torch.einsum("bthd,bsd->bths", q, k)
    # score: [B, T, S]
    scores = (torch.relu(dots) * weights.unsqueeze(-1)).sum(dim=2) * scale

    starts = torch.zeros((queries, 1), dtype=torch.int32, device=q.device)
    if cu_seqlens is not None:
        positions = torch.arange(queries, device=q.device, dtype=torch.int32)
        # Empty documents are skipped; capacity tails receive an empty interval.
        documents = torch.searchsorted(cu_seqlens[1:], positions, right=True, out_int32=True)
        starts = cu_seqlens_k.index_select(0, documents)
        ends = cu_seqlens_k.index_select(0, (documents + 1).clamp(max=cu_seqlens_k.shape[0] - 1))
        if causal:
            local_positions = positions - cu_seqlens.index_select(0, documents)
            ends = torch.minimum(ends, starts + (local_positions + 1) // compress_ratio)
        starts = starts[:, None]
        ends = ends[:, None]
    elif causal:
        ends = (torch.arange(1, queries + 1, device=q.device) // compress_ratio)[:, None]
    else:
        ends = torch.full_like(starts, candidates)
    key_positions = torch.arange(candidates, device=q.device)[None, :]
    valid = (key_positions >= starts) & (key_positions < ends)
    scores.masked_fill_(~valid, float("-inf"))

    indices = scores.topk(min(topk, candidates), dim=-1).indices.to(torch.int32)
    selected_valid = (indices >= starts) & (indices < ends)
    indices = torch.where(selected_valid, indices - starts, -1)
    return torch.nn.functional.pad(indices, (0, max(0, topk - candidates)), value=-1)
