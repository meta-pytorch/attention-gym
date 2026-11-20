"""Independent eager oracle for paged completed-block selection."""

import math

import torch


def launch(
    query,
    key_cache,
    weights,
    block_table,
    cu_seqlens_q,
    seqused_k,
    topk,
    compress_ratio,
    query_positions,
):
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
        selected = scores.topk(min(topk, candidates), dim=-1).indices
        output[start:end, : selected.shape[1]] = torch.where(
            selected < visible[:, None], selected, -1
        ).int()
    return output
