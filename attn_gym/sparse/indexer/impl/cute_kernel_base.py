# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by Attention Gym in 2026: shared indexer kernel helpers.

"""Shared helpers for the SM100 indexer score and Top-K kernel classes."""

import cutlass
from cutlass import Int32, Int64, cute


class IndexerKernelBase:
    """Offset widening and causal visibility shared by every indexer kernel.

    Subclasses set ``use_int64_offsets`` and ``compress_ratio`` in ``__init__``.
    """

    use_int64_offsets: bool
    compress_ratio: int

    @cute.jit
    def upcast_offset(self, value):
        """Widen origins before address arithmetic in the wide specialization."""
        return Int64(value) if cutlass.const_expr(self.use_int64_offsets) else Int32(value)

    @cute.jit
    def visible_candidates(self, query):
        """Causal candidate count for query; the static ratio keeps r=1 division-free."""
        if cutlass.const_expr(self.compress_ratio == 1):
            return query + 1
        return (query + 1) // self.compress_ratio


class IndexerScoreKernelBase(IndexerKernelBase):
    """Static choices and artifact naming shared by the score kernels."""

    name_prefix: str
    heads: int
    head_dim: int
    causal: bool
    contiguous_weight_heads: bool

    def get_name(self) -> str:
        """Return a stable name for the shape, mask, weight layout, and offset width."""
        return (
            f"{self.name_prefix}_h{self.heads}_d{self.head_dim}_c{int(self.causal)}_"
            f"r{self.compress_ratio}_i64{int(self.use_int64_offsets)}_"
            f"wh{int(self.contiguous_weight_heads)}"
        )
