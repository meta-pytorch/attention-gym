# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KDA bound to the delta-rule context-parallel recipe."""

from __future__ import annotations

from functools import partial

import torch
import torch.distributed as dist

from attn_gym.linear.context_parallel import (
    ContextParallelRouting,
    StagedOp,
    context_parallel_chunk,
)
from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward
from attn_gym.linear.types import KernelOptions


def context_parallel_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    routing: ContextParallelRouting,
    group: dist.ProcessGroup,
    scale: float | None = None,
    autotune: bool = True,
    fastmath: bool = False,
    kernel_options: KernelOptions | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``chunk_kda`` over this rank's span with state exchanged by all-gather.

    See ``attn_gym.linear.context_parallel.context_parallel_chunk`` for the argument contract;
    ``scale``, ``autotune``, and ``kernel_options`` follow ``chunk_kda``. With
    ``kernel_options={"backend": "mega"}`` the local pass runs on Mega and the fused factors are
    computed once over the span for the summaries; ``fastmath`` applies to the staged backward,
    which is the fused one for either backend.
    """
    stages = StagedOp(
        partial(chunk_kda_prepare, scale=scale, autotune=autotune, kernel_options=kernel_options),
        partial(chunk_kda_prepare_backward, autotune=autotune, fastmath=fastmath),
    )
    return context_parallel_chunk(stages, q, k, v, gate, beta, routing=routing, group=group)


__all__ = ["context_parallel_kda"]
