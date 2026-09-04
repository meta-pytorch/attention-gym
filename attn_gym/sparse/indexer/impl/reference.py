"""Pure-PyTorch reference indexer implementation."""

import math

import torch
from torch import Tensor


def index(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    topk: int,
    causal: bool,
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
        causal: mask out candidates at positions beyond the query position

    Returns:
        [B, T, topk] INT32 tensor of selected candidate indices.
    """
    batch, queries, heads, head_dim = q.shape
    candidates = k.shape[1]
    if topk == 0:
        return torch.empty((batch, queries, 0), dtype=torch.int32, device=q.device)
    scale = 1.0 / math.sqrt(heads * head_dim)

    # dots: [B, T, H, S]
    dots = torch.einsum("bthd,bsd->bths", q, k)
    # score: [B, T, S]
    scores = (torch.relu(dots) * weights.unsqueeze(-1)).sum(dim=2) * scale

    if causal:
        query_positions = torch.arange(queries, device=q.device)[:, None]
        key_positions = torch.arange(candidates, device=q.device)[None, :]
        scores.masked_fill_(key_positions > query_positions, float("-inf"))

    indices = scores.topk(topk, dim=-1).indices.to(torch.int32)

    # Slots that selected a causally invalid candidate are replaced with -1.
    # This happens when topk exceeds the number of valid candidates for a row
    # (i.e. query position t < topk).
    if causal:
        row = torch.arange(queries, device=q.device).view(1, queries, 1)
        indices.masked_fill_(indices > row, -1)

    return indices
