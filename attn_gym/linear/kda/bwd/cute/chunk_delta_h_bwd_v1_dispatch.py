# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-ignore-all-errors

"""
Dispatch wrapper for BlackwellDeltaHBwd V1 — selects BV=16 or BV=32 SS-mode
based on the total V-tile CTA count relative to the device SM count.

Heuristic (empirically validated on GH200, SM=132):
    w_tiles_bv16 = ceil(V / 16) * H * B
    if w_tiles_bv16 > sm_count  →  BV=32  (fewer waves, more work per CTA)
    else                        →  BV=16  (fills SMs without wasting any)

Crossover point: w_tiles_bv16 = SM_count ≈ 132 on GH200.
  - H=16, B=1: w16=128 ≤ 132 → BV=16 (0.230ms)
  - H=24, B=1: w16=192 > 132 → BV=32 (0.270ms)
  - H=16, B=2: w16=256 > 132 → BV=32 (0.270ms)
"""

import torch

from attn_gym._backends.cute import ceildiv
from attn_gym.linear.kda.bwd.cute.chunk_delta_h_bwd_v1 import (
    blackwell_delta_h_bwd_dhu_v1,
)


def blackwell_delta_h_bwd_dhu_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float = 1.0,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    dv2_out: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    num_seqs: torch.Tensor | int | None = None,
    num_chunks: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    BlackwellDeltaHBwd backward dhu — auto-dispatches BV=16 or BV=32.

    Returns (dh, dh0, dv2).
    """
    batch, _tokens, heads, _head_dim = q.shape
    value_dim = do.shape[-1]
    logical_batch = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1

    w_tiles_bv16 = ceildiv(value_dim, 16) * heads * logical_batch
    sm_count = torch.cuda.get_device_properties(q.device).multi_processor_count

    bv = 32 if w_tiles_bv16 > sm_count else 16

    return blackwell_delta_h_bwd_dhu_v1(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        gk=gk,
        h0=h0,
        dht=dht,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        dv2_out=dv2_out,
        chunk_offsets=chunk_offsets,
        num_seqs=num_seqs,
        bv=bv,
        NT=num_chunks,
    )
