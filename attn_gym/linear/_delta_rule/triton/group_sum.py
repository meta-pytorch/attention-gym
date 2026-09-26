# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Deterministic reduction of per-value-head q/k gradients onto grouped q/k heads.

Grouped-head backward kernels write one dq/dk slice per value head ``[..., HK * G, D]``, where
the ``G`` consecutive value heads of a group share one q/k head. This kernel sums each group in
a fixed order with FP32 accumulation, optionally adding a same-shaped second gradient first, and
writes ``[..., HK, D]`` in the requested dtype in one pass. It uses no atomics, so results are
bitwise reproducible.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def group_sum_kernel(
    x,
    addend,
    out,
    rows,
    G: tl.constexpr,
    D: tl.constexpr,
    D_POW2: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    """Sum ``G`` consecutive ``[D]`` slices into one output row for ``BLOCK_R`` rows."""
    row = tl.program_id(0).to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R)
    col = tl.arange(0, D_POW2)
    mask = (row < rows)[:, None] & (col < D)[None, :]
    source = row[:, None] * (G * D) + col[None, :]
    acc = tl.zeros([BLOCK_R, D_POW2], dtype=tl.float32)
    for group in tl.static_range(G):
        term = tl.load(x + source + group * D, mask=mask).to(tl.float32)
        if HAS_ADDEND:
            term += tl.load(addend + source + group * D, mask=mask).to(tl.float32)
        acc += term
    tl.store(out + row[:, None] * D + col[None, :], acc.to(out.dtype.element_ty), mask=mask)


def group_sum(
    x: torch.Tensor,
    groups: int,
    *,
    out_dtype: torch.dtype,
    addend: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``(x + addend).unflatten(-2, (-1, groups)).sum(-2)`` in ``out_dtype``.

    ``x`` and the optional ``addend`` are contiguous ``[..., HK * groups, D]`` tensors. The sum
    runs in FP32 in a fixed group order, so it is deterministic.
    """
    if not x.is_contiguous() or (addend is not None and not addend.is_contiguous()):
        raise ValueError("group_sum requires contiguous inputs")
    if addend is not None and addend.shape != x.shape:
        raise ValueError("addend must match x")
    heads, dim = x.shape[-2:]
    if heads % groups:
        raise ValueError(f"heads={heads} must be divisible by groups={groups}")
    out = x.new_empty((*x.shape[:-2], heads // groups, dim), dtype=out_dtype)
    rows = out.numel() // dim
    if rows == 0:
        return out
    # GB200, [40960, 48, 128] -> [40960, 16, 128]: ~7 TB/s for BF16 and FP32-with-addend inputs.
    block_rows = 4
    group_sum_kernel[(triton.cdiv(rows, block_rows),)](
        x,
        x if addend is None else addend,
        out,
        rows,
        G=groups,
        D=dim,
        D_POW2=triton.next_power_of_2(dim),
        HAS_ADDEND=addend is not None,
        BLOCK_R=block_rows,
        num_warps=4,
    )
    return out


__all__ = ["group_sum"]
