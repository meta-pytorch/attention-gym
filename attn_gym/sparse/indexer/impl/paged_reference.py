"""Independent eager oracle for paged completed-block selection."""

import torch

from .reference import score_and_select


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
    offsets = cu_seqlens_q.tolist()
    page_size = key_cache.shape[1]
    for sequence, length in enumerate(seqused_k.tolist()):
        start, end = offsets[sequence], offsets[sequence + 1]
        candidates = length // compress_ratio
        if start == end or candidates == 0:
            continue
        logical = torch.arange(candidates, device=query.device)
        pages = block_table[sequence, logical // page_size].long()
        keys = key_cache[pages, logical % page_size]
        positions = (
            torch.arange(length - (end - start), length, device=query.device)
            if query_positions is None
            else query_positions[start:end].clamp_max(length - 1)
        )
        visible = ((positions + 1) // compress_ratio).clamp(min=0, max=candidates)
        valid = logical[None, :] < visible[:, None]
        selected = score_and_select(query[start:end], keys, weights[start:end], topk, valid)
        output[start:end, : selected.shape[1]] = selected
    return output
