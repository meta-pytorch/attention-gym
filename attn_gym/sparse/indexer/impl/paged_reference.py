"""Independent eager oracle for paged completed-block selection."""

import torch

from .reference import score_and_select


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
    q_scale: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score visible candidates in logical block order, independently of physical pages."""
    output = torch.full((query.shape[0], topk), -1, device=query.device, dtype=torch.int32)
    if topk == 0:
        return output
    offsets = cu_seqlens_q.tolist()
    page_size = key_cache.shape[1]
    for sequence, length in enumerate(seqused_k.tolist()):
        start, end = offsets[sequence], offsets[sequence + 1]
        candidates = length // compress_ratio
        if start == end or candidates == 0:
            continue
        logical = torch.arange(candidates, device=query.device)
        pages = block_table[sequence, logical // page_size].long()
        physical = (pages, logical % page_size)
        positions = (
            torch.arange(length - (end - start), length, device=query.device)
            if query_positions is None
            else query_positions[start:end].clamp_max(length - 1)
        )
        visible = ((positions + 1) // compress_ratio).clamp(min=0, max=candidates)
        valid = logical[None, :] < visible[:, None]
        selected = score_and_select(
            query[start:end],
            key_cache[physical],
            weights[start:end],
            topk,
            valid,
            deterministic=torch.are_deterministic_algorithms_enabled(),
            q_scale=None if q_scale is None else q_scale[start:end],
            k_scale=None if k_scale is None else k_scale[physical],
        )
        output[start:end, : selected.shape[1]] = selected
    return output
