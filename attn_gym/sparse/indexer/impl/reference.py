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

    Computes::

        dots[b, t, h, s] = q[b, t, h, :] · k[b, s, :]
        score[b, t, s]   = sum_h(w[b, t, h] * relu(dots[b, t, h, s]))
                           / sqrt(H * D)
        if causal: score[b, t, s] = -inf  where s > t
        output[b, t, :]  = topk(score[b, t, :]).indices

    Args:
        q: ``[B, T, H, D]``
        k: ``[B, S, D]``
        weights: ``[B, T, H]``
        topk: number of candidates to select per query
        causal: mask out candidates at positions beyond the query position

    Returns:
        ``[B, T, topk]`` INT32 tensor of selected candidate indices.
    """
    batch, queries, heads, head_dim = q.shape
    candidates = k.shape[1]
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
