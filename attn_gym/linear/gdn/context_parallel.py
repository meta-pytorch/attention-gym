# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GDN bound to the delta-rule context-parallel recipe."""

from __future__ import annotations

from functools import partial

import torch
import torch.distributed as dist

from attn_gym.linear.context_parallel import (
    ContextParallelRouting,
    StagedOp,
    context_parallel_chunk,
)
from attn_gym.linear.gdn.stages import chunk_gdn_prepare, chunk_gdn_prepare_backward
from attn_gym.types import TuningOptions


def context_parallel_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    routing: ContextParallelRouting,
    group: dist.ProcessGroup,
    scale: float | None = None,
    kernel_options: TuningOptions | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fused ``chunk_gdn`` over this rank's span with state exchanged by all-gather.

    See ``attn_gym.linear.context_parallel.context_parallel_chunk`` for the argument contract;
    ``scale`` and the ``"autotune"`` kernel option follow ``chunk_gdn``.
    """
    stages = StagedOp(
        partial(chunk_gdn_prepare, scale=scale, kernel_options=kernel_options),
        partial(chunk_gdn_prepare_backward, kernel_options=kernel_options),
    )
    return context_parallel_chunk(stages, q, k, v, gate, beta, routing=routing, group=group)


__all__ = ["context_parallel_gdn"]
