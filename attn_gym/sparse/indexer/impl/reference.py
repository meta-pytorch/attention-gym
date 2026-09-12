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

    Returns:
        [B, T, topk] INT32 tensor of selected candidate indices.
    """
    batch, queries, heads, head_dim = q.shape
    candidates = k.shape[1]
    if topk == 0:
        return torch.empty((batch, queries, 0), dtype=torch.int32, device=q.device)
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

    if causal:
        visible = (torch.arange(1, queries + 1, device=q.device) // compress_ratio)[:, None]
        key_positions = torch.arange(candidates, device=q.device)[None, :]
        scores.masked_fill_(key_positions >= visible, float("-inf"))

    indices = scores.topk(topk, dim=-1).indices.to(torch.int32)

    # Slots that selected a causally invalid candidate are replaced with -1.
    # This happens when topk exceeds the number of visible candidates for a row.
    if causal:
        indices.masked_fill_(indices >= visible.view(1, queries, 1), -1)

    return indices
