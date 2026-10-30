"""Torch-only paged MLA decode reference implementation."""

from __future__ import annotations

import torch
from torch import Tensor


def paged_mla_decode(
    q_latent: Tensor,
    q_rope: Tensor,
    latent_cache: Tensor,
    rope_cache: Tensor,
    block_table: Tensor,
    cu_seqlens_q: Tensor,
    seqused_k: Tensor,
    *,
    scale: float,
) -> tuple[Tensor, Tensor]:
    """Gather each sequence's cached rows and attend in at least float32.

    Rows whose causal window is empty return zero output and ``-inf`` LSE.
    """
    compute_dtype = torch.promote_types(q_latent.dtype, torch.float32)
    num_tokens, num_heads, _ = q_latent.shape
    out = q_latent.new_zeros(q_latent.shape)
    lse = torch.full(
        (num_tokens, num_heads), float("-inf"), dtype=compute_dtype, device=q_latent.device
    )
    block_size = latent_cache.shape[1]
    offsets = cu_seqlens_q.tolist()
    for sequence, num_keys in enumerate(seqused_k.tolist()):
        start, end = offsets[sequence], offsets[sequence + 1]
        if start == end or num_keys == 0:
            continue
        blocks = block_table[sequence, : -(-num_keys // block_size)].long()
        keys_latent = latent_cache[blocks].flatten(0, 1)[:num_keys].to(compute_dtype)
        keys_rope = rope_cache[blocks].flatten(0, 1)[:num_keys].to(compute_dtype)
        logits = (
            torch.einsum("qhl,kl->hqk", q_latent[start:end].to(compute_dtype), keys_latent)
            + torch.einsum("qhr,kr->hqk", q_rope[start:end].to(compute_dtype), keys_rope)
        ) * scale
        last_key = torch.arange(end - start, device=q_latent.device) + num_keys - (end - start)
        key_positions = torch.arange(num_keys, device=q_latent.device)
        logits = logits.masked_fill(key_positions > last_key[:, None], float("-inf"))
        empty = torch.isneginf(logits).all(dim=-1, keepdim=True)
        probabilities = torch.where(empty, 0, torch.softmax(torch.where(empty, 0, logits), -1))
        out[start:end] = torch.einsum("hqk,kl->qhl", probabilities, keys_latent).to(out.dtype)
        lse[start:end] = torch.where(empty.squeeze(-1), float("-inf"), logits.logsumexp(-1)).T
    return out, lse.float()
