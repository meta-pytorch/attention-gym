"""Independent eager oracle for paged indexer key updates."""

import torch


def _rms_offset_norm(x, weight, eps):
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return ((1.0 + weight.float()) * x32).to(x.dtype)


def _rotate(x, rope_cache, position):
    rotary = rope_cache.shape[-1] // 2
    cos = rope_cache[position, :rotary].to(x.dtype)
    sin = rope_cache[position, rotary:].to(x.dtype)
    head = x[..., :rotary]
    first, second = head.chunk(2, -1)
    rotated = head * cos + torch.cat((-second, first), -1) * sin
    return torch.cat((rotated, x[..., rotary:]), -1)


def launch(
    raw_keys,
    key_cache,
    raw_cache,
    block_table,
    raw_block_table,
    cu_seqlens_q,
    seqused_k,
    norm_weight,
    rope_cache,
    compress_ratio,
    eps,
):
    page_size = key_cache.shape[1]
    ring_capacity = raw_cache.shape[1]
    pooled_rows = []
    ring_rows = []
    for sequence, length in enumerate(seqused_k.tolist()):
        start, end = cu_seqlens_q[sequence : sequence + 2].tolist()
        if start == end:
            continue
        ring_page = int(raw_block_table[sequence, 0])
        first = length - (end - start)
        for row in range(start, end):
            position = first + row - start
            if (position + 1) % compress_ratio == 0:
                members = []
                for member in range(position - compress_ratio + 1, position + 1):
                    if member >= first:
                        members.append(raw_keys[start + member - first])
                    else:
                        members.append(raw_cache[ring_page, member % ring_capacity])
                pooled = torch.stack(members).float().mean(0).to(raw_keys.dtype)
                pooled = _rms_offset_norm(pooled, norm_weight, eps)
                pooled = _rotate(pooled, rope_cache, position - compress_ratio + 1)
                candidate = position // compress_ratio
                page = int(block_table[sequence, candidate // page_size])
                pooled_rows.append((page, candidate % page_size, pooled))
            if position >= length // compress_ratio * compress_ratio:
                ring_rows.append((ring_page, position % ring_capacity, raw_keys[row]))
    for page, slot, value in pooled_rows:
        key_cache[page, slot] = value
    for page, slot, value in ring_rows:
        raw_cache[page, slot] = value
