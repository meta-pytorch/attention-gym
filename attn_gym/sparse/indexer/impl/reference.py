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
    q_scale: Tensor | None = None,
    k_scale: Tensor | None = None,
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
        q_scale: optional E8M0 per-32-element dequantization scales, [B, T, H, D/32].
        k_scale: optional E8M0 per-32-element dequantization scales, [B, S, D/32].

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
    if q_scale is not None:  # The API requires both scales or neither.
        q = (q.unflatten(-1, (head_dim // 32, 32)) * q_scale.float().unsqueeze(-1)).flatten(-2)
        k = (k.unflatten(-1, (head_dim // 32, 32)) * k_scale.float().unsqueeze(-1)).flatten(-2)

    indices = torch.empty((batch, queries, topk), dtype=torch.int32, device=q.device)
    # Query chunks preserve every candidate while bounding per-head intermediates.
    for start in range(0, queries, 128):
        end = min(start + 128, queries)
        # dots: [B, query_chunk, H, S]
        dots = torch.einsum("bthd,bsd->bths", q[:, start:end], k)
        # score: [B, query_chunk, S]
        scores = (torch.relu(dots) * weights[:, start:end].unsqueeze(-1)).sum(dim=2) * scale

        if causal:
            visible = (torch.arange(start + 1, end + 1, device=q.device) // compress_ratio)[
                :, None
            ]
            key_positions = torch.arange(candidates, device=q.device)[None, :]
            scores.masked_fill_(key_positions >= visible, float("-inf"))

        selected = scores.topk(topk, dim=-1).indices.to(torch.int32)

        # Slots that selected a causally invalid candidate are replaced with -1.
        # This happens when topk exceeds the number of visible candidates for a row.
        if causal:
            selected.masked_fill_(selected >= visible.view(1, end - start, 1), -1)
        indices[:, start:end] = selected
    return indices
