"""Private PyTorch launcher for the public ``lightning_indexer`` API."""

import math

import torch
from torch import Tensor


def score_rows(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    q_scale: Tensor | None = None,
    k_scale: Tensor | None = None,
) -> Tensor:
    """Score a tile of queries with Q/K/weights already in their accumulation dtype."""
    # Autocast would narrow the einsum result before the scales and head weights apply.
    with torch.autocast(q.device.type, enabled=False):
        if q_scale is None:
            dots = torch.einsum("bthd,bsd->bths", q, k)
        else:
            # Native MX scales a 32-element dot, not each FP8 operand separately. The
            # product of the E8M0 factors can itself exceed FP32 before scaling the dot.
            assert k_scale is not None
            dots = q.new_zeros((*q.shape[:-1], k.shape[1]))
            for group in range(q.shape[-1] // 32):
                channels = slice(32 * group, 32 * (group + 1))
                block = torch.einsum("bthd,bsd->bths", q[..., channels], k[..., channels])
                factor = (
                    q_scale[..., group, None].double() * k_scale[:, None, None, :, group].double()
                )
                dots = (block * factor + dots).float()
        return (dots.relu() * weights.unsqueeze(-1)).sum(2) * (
            1.0 / math.sqrt(q.shape[2] * q.shape[3])
        )


@torch.no_grad()
def launch(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    topk: int,
    causal: bool,
    compress_ratio: int,
    cu_seqlens: Tensor | None = None,
    cu_seqlens_k: Tensor | None = None,
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
            for nonpacked calls.
        cu_seqlens: Packed query offsets, [N + 1]; supplied together with cu_seqlens_k.
        cu_seqlens_k: Packed candidate offsets, [N + 1].
        q_scale: optional E8M0 per-32-element dequantization scales, [B, T, H, D/32].
        k_scale: optional E8M0 per-32-element dequantization scales, [B, S, D/32].

    Returns:
        [B, T, topk] INT32 tensor of selected candidate indices.
    """
    batch, queries = q.shape[:2]
    candidates = k.shape[1]
    if topk == 0 or candidates == 0:
        return torch.full((batch, queries, topk), -1, dtype=torch.int32, device=q.device)

    # Accumulate scoring in FP32 to match the CuTe backend's accumulation
    # contract, regardless of the input storage dtype (e.g. FP16/BF16 can
    # overflow the dot product or produce NaNs from mixed-sign weights before
    # scaling). FP64 inputs are left at FP64 rather than downcast.
    accum_dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q = q.to(accum_dtype)
    k = k.to(accum_dtype)
    weights = weights.to(accum_dtype)

    # Query chunks bound the [B, chunk, H, S] per-head intermediates; scores stay [B, T, S].
    scores = q.new_empty(batch, queries, candidates)
    for start in range(0, queries, 128):
        rows = slice(start, start + 128)
        row_scales = q_scale[:, rows] if q_scale is not None else None
        scores[:, rows] = score_rows(q[:, rows], k, weights[:, rows], row_scales, k_scale)

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
