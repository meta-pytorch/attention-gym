"""Adapt whole-document pools to the fixed paged scorer without copying keys."""

import torch

from .paged_triton import launch as paged_launch


def launch(q, k, weights, topk, causal, compress_ratio, cu_q, cu_k, *, backend):
    batch, tokens, heads, dim = q.shape
    if tokens == 0 or topk == 0:
        return q.new_empty((batch, tokens, topk), dtype=torch.int32)
    if cu_q is None:
        cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * tokens
        lengths = torch.full((batch,), tokens, device=q.device, dtype=torch.int32)
        table = torch.arange(batch, device=q.device, dtype=torch.int32)[:, None]
    else:
        lengths = cu_q[1:] - cu_q[:-1]
        table = torch.zeros((lengths.numel(), 1), device=q.device, dtype=torch.int32)
    # Noncausal selection exposes the complete pool to every query. Explicit positions
    # override only the causal visibility, while lengths still bound the candidate pool.
    positions = None
    if not causal:
        positions = torch.full((batch * tokens,), 2**31 - 1, device=q.device, dtype=torch.int32)
    result = paged_launch(
        q.reshape(batch * tokens, heads, dim),
        k,
        weights.reshape(batch * tokens, heads),
        table,
        cu_q,
        lengths,
        topk,
        compress_ratio,
        positions,
        backend,
        batch_invariant=True,
        key_offsets=cu_k,
        whole_pool=True,
    )
    return result.view(batch, tokens, topk)
