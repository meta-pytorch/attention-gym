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
from attn_gym.linear.context_parallel_deterministic import (
    CanonicalRouting,
    CanonicalTiling,
    context_parallel_chunk_deterministic,
)
from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward
from attn_gym.linear.kda.validation import resolve_kernel_options
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
    computed once over the span for the summaries; its backward is Mega's native stateful kernel,
    so ``fastmath`` applies only to the fused backend's backward.
    """
    stages = _kda_stages(scale, autotune, fastmath, kernel_options)
    return context_parallel_chunk(stages, q, k, v, gate, beta, routing=routing, group=group)


def context_parallel_kda_deterministic(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    tiling: CanonicalTiling,
    group: dist.ProcessGroup,
    routing: CanonicalRouting | None = None,
    scale: float | None = None,
    fastmath: bool = False,
    kernel_options: KernelOptions | None = None,
) -> torch.Tensor:
    """Run KDA over this rank's canonical tiles; results do not depend on the CP degree.

    See NOTE [Canonical Tiles] in ``attn_gym.linear.context_parallel_deterministic``. Autotuning
    is disabled because a tuned configuration is part of the arithmetic and would have to be
    identical on every rank; ``fastmath`` and ``kernel_options`` follow ``chunk_kda``.
    """
    stages = _kda_stages(scale, False, fastmath, kernel_options)
    return context_parallel_chunk_deterministic(
        stages, q, k, v, gate, beta, tiling=tiling, group=group, routing=routing
    )


def _kda_stages(scale, autotune, fastmath, kernel_options) -> StagedOp:
    return StagedOp(
        partial(chunk_kda_prepare, scale=scale, autotune=autotune, kernel_options=kernel_options),
        partial(
            chunk_kda_prepare_backward,
            autotune=autotune,
            fastmath=fastmath,
            schedule=resolve_kernel_options(kernel_options).schedule,
        ),
    )


__all__ = ["context_parallel_kda", "context_parallel_kda_deterministic"]
