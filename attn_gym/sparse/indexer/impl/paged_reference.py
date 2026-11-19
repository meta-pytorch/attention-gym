"""Independent eager oracle for paged completed-block selection."""

import math

import torch


def launch(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    weights: torch.Tensor,
    block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    topk: int,
    compress_ratio: int,
    query_positions: torch.Tensor | None,
) -> torch.Tensor:
    """Score visible candidates in logical block order, independently of physical pages."""
    output = torch.full((query.shape[0], topk), -1, device=query.device, dtype=torch.int32)
    if topk == 0:
        return output
    dtype = torch.float64 if query.dtype == torch.float64 else torch.float32
    page_size = key_cache.shape[1]
    for sequence, length in enumerate(seqused_k.tolist()):
        start, end = cu_seqlens_q[sequence : sequence + 2].tolist()
        candidates = length // compress_ratio
        if start == end or candidates == 0:
            continue
        logical = torch.arange(candidates, device=query.device)
        pages = block_table[sequence, logical // page_size].long()
        keys = key_cache[pages, logical % page_size].to(dtype)
        dots = torch.einsum("thd,sd->ths", query[start:end].to(dtype), keys)
        scores = (dots.relu() * weights[start:end].to(dtype).unsqueeze(-1)).sum(1)
        scores /= math.sqrt(query.shape[1] * query.shape[2])
        positions = (
            torch.arange(length - (end - start), length, device=query.device)
            if query_positions is None
            else query_positions[start:end].clamp_max(length - 1)
        )
        visible = ((positions + 1) // compress_ratio).clamp(min=0, max=candidates)
        scores.masked_fill_(logical[None, :] >= visible[:, None], -torch.inf)
        width = min(topk, candidates)
        if torch.are_deterministic_algorithms_enabled():
            ranked = scores.argsort(dim=-1, descending=True, stable=True)
            selected = ranked[:, :width].sort(dim=-1).values
        else:
            selected = scores.topk(width, dim=-1).indices
        output[start:end, : selected.shape[1]] = torch.where(
            selected < visible[:, None], selected, -1
        ).int()
    return output
