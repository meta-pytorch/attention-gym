# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One-launch fold of gathered affine state maps along per-row slot chains.

Maps are ``[bias; transition]`` packed ``[V + K, K]`` per head (``attn_gym.linear.context_parallel``):
applying one to a V-major state is ``state @ A + B``, and every state row depends only on the
same row before it. A program owns ``BM`` rows of one head for one output row of the chain
table and walks that row's slots serially, one ``[BM, K] @ [K, K]`` dot per slot, so the whole
table folds in a single launch instead of one batched matmul per chain step.

The chain table is the routing's ``[rows, L]`` index tensor (padded with an identity slot), so
the grid and the trip count are fixed by the layout bounds and the launch replays inside a
captured CUDA Graph while the indices change in place. Trailing padding slots are not applied,
so a padded chain is bitwise the same chain unpadded.

The dots run in IEEE FP32 (``input_precision="ieee"``): the chain is latency-bound on small
dots, where one FMA pass beats three tensor-core passes, and the result matches the batched
FP32 matmul fold to summation order (2.6e-6 vs 2.5e-6 relative to FP64 over a 63-step chain).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def fold_slots_kernel(
    maps,
    sources,
    states,
    neutral: tl.int32,
    H: tl.constexpr,
    V: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    L_POW2: tl.constexpr,
    BM: tl.constexpr,
):
    """Fold one chain row's slots onto ``BM`` rows of one head's zero state."""
    row = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    rows = tl.program_id(2) * BM + tl.arange(0, BM)
    cols = tl.arange(0, K)
    chain = tl.arange(0, L_POW2)
    slots = tl.load(sources + row * L + chain, mask=chain < L, other=neutral)
    count = tl.sum((slots != neutral).to(tl.int32))  # padding trails the real slots

    acc = tl.zeros([BM, K], dtype=tl.float32)
    for step in tl.range(0, count, num_stages=2):
        slot = tl.load(sources + row * L + step).to(tl.int64)
        base = maps + (slot * H + head) * (V + K) * K
        transition = tl.load(base + (V + cols[:, None]) * K + cols[None, :])  # [K, K]
        bias = tl.load(base + rows[:, None] * K + cols[None, :])
        acc = tl.dot(acc, transition, input_precision="ieee") + bias
    tl.store(states + (row * H + head) * V * K + rows[:, None] * K + cols[None, :], acc)


def fold_slots_fused(maps: torch.Tensor, sources: torch.Tensor) -> torch.Tensor:
    """Fold ``maps[sources[r]]`` in order from the zero state for every row ``r`` in one launch.

    ``maps`` is ``[slots, H, V + K, K]`` FP32; ``sources`` is ``int64 [rows, L]`` whose entries
    index ``maps`` or equal ``maps.shape[0]`` (the identity, skipped). Returns ``[rows, H, V, K]``.
    Arithmetic is IEEE FP32 and depends only on the slot chain, never on how many rows or slots
    share the launch.
    """
    slots, heads, packed, key_dim = maps.shape
    value_dim = packed - key_dim
    rows, chain = sources.shape
    states = maps.new_empty((rows, heads, value_dim, key_dim))
    if rows == 0 or chain == 0:
        return states.zero_()
    # GB200, H=64: 32-row blocks with 8 warps beat 16 (fewer transition reloads) and 64 (occupancy).
    block_rows = 32
    fold_slots_kernel[(rows, heads, value_dim // block_rows)](
        maps.contiguous(),
        sources.contiguous(),
        states,
        slots,
        H=heads,
        V=value_dim,
        K=key_dim,
        L=chain,
        L_POW2=triton.next_power_of_2(chain),
        BM=block_rows,
        num_warps=8,
    )
    return states


__all__ = ["fold_slots_fused"]
