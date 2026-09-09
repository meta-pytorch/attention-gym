# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Context parallelism that never splits a document: no summaries, no collectives, exact bits.

NOTE [Document-aligned fragments]
If every fragment boundary is a document boundary, no rank ever needs another rank's state: each
document runs from the zero state to its end on one rank, exactly as an unsharded call would run
it. There is nothing to summarize, exchange, or compose, so the result is bitwise the same for
any CP degree and any document-aligned fragment table, including CP=1, and it costs less than the
standard recipe (which still computes and gathers summaries for such fragments).

What the recipe pins is the kernels, not the sharding: a document's bits must not depend on the
documents packed around it. The staged handles guarantee that (``test/test_canonical_tile_batching.py``
for the fused kernels; ``test/test_document_parallel.py`` for both) as long as autotuning is off and
Mega's forgetting-horizon split is not used. The public ``chunk_kda`` is not a substitute: it picks
kernels from the span's physical length. With the fused backend the recipe's output and gradients
are bitwise those of ``chunk_kda(autotune=False)``; with Mega the output is, and the gradients come
from Mega's own backward where the public op still runs the fused one.

Balance is the cost: cutting only at document boundaries can leave a rank a document's worth of
tokens out of balance, and a document longer than a rank can hold has no document-aligned cut at
all; that case needs the sequence split across ranks (``context_parallel_chunk``).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from attn_gym._backends.profiler import profiler_range
from attn_gym.linear.context_parallel import StagedOp


def check_document_aligned(
    cu_seqlens_global: Sequence[int], fragments: Sequence[Sequence[tuple[int, int]]]
) -> None:
    """Raise unless every fragment of every rank starts and stops on a document boundary."""
    boundaries = set(cu_seqlens_global)
    for rank, rank_fragments in enumerate(fragments):
        for start, stop in rank_fragments:
            if start not in boundaries or stop not in boundaries:
                raise ValueError(
                    f"rank {rank} fragment [{start}, {stop}) splits a document; document-parallel"
                    " context parallelism needs fragments cut on document boundaries"
                )


class _DocumentParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, gate, beta, cu_seqlens, stages: StagedOp):
        prepared = stages.prepare(q, k, v, gate, beta, cu_seqlens=cu_seqlens)
        with profiler_range("dp/run"):
            output, _ = prepared.run(None, output_final_state=False)
        ctx.save_for_backward(*prepared.saved)
        ctx.saved_type = type(prepared.saved)
        ctx.scale = prepared.scale
        ctx.stages = stages
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_output):
        saved = ctx.saved_type._make(ctx.saved_tensors)
        grads = ctx.stages.prepare_backward(saved, d_output, None, scale=ctx.scale)
        with profiler_range("dp/run"):
            dq, dk, dv, dgate, dbeta, _ = grads.run(None)
        return dq, dk, dv, dgate, dbeta, None, None


def document_parallel_chunk(
    stages: StagedOp,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Run a staged delta-rule op over this rank's whole documents (NOTE [Document-aligned fragments]).

    ``q``/``k``/``v``/``gate``/``beta`` are the rank's span and ``cu_seqlens`` its packed document
    boundaries (``ContextParallelPlan.routing(device).cu_seqlens``). No collective runs, so ranks
    need not call this the same number of times; a rank with no documents simply skips it.
    """
    return _DocumentParallel.apply(q, k, v, gate, beta, cu_seqlens, stages)


__all__ = ["check_document_aligned", "document_parallel_chunk"]
