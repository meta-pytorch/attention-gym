# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""KDA bound to the delta-rule context-parallel recipe."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import torch
import torch.distributed as dist

from attn_gym.linear._delta_rule.validation import resolve_scale
from attn_gym.linear.context_parallel import (
    ContextParallelRouting,
    StagedOp,
    context_parallel_chunk,
)
from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward
from attn_gym.linear.kda.validation import resolve_kernel_options


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
    kernel_options: Mapping[str, object] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``chunk_kda`` over this rank's span with state exchanged by all-gather.

    See ``attn_gym.linear.context_parallel.context_parallel_chunk`` for the argument contract;
    ``scale``, ``autotune``, ``fastmath``, and ``kernel_options`` follow ``chunk_kda``. With
    ``kernel_options={"backend": "mega"}`` the local pass runs on Mega and the fused factors are
    computed once over the span for the summaries.
    """
    # The unsharded Mega op rejects fastmath; the staged backward alone cannot see the backend.
    if fastmath and resolve_kernel_options(kernel_options).backend == "mega":
        raise ValueError("fastmath is not supported by the Mega backend")
    scale = resolve_scale(scale, q.shape[-1])  # Both stages must see the same resolved value.
    stages = StagedOp(
        partial(chunk_kda_prepare, scale=scale, autotune=autotune, kernel_options=kernel_options),
        partial(chunk_kda_prepare_backward, scale=scale, autotune=autotune, fastmath=fastmath),
    )
    return context_parallel_chunk(stages, q, k, v, gate, beta, routing=routing, group=group)


__all__ = ["context_parallel_kda"]
