# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inference-only KDA wrappers for the shared fused recurrent delta-rule scan.

The vector-log2 specialization mirrors ``naive_recurrent_kda``; use ``chunk_kda`` for training.
"""

from __future__ import annotations

import torch

from attn_gym.linear._delta_rule.recurrent import launch_recurrent_delta_rule_fwd
from attn_gym.linear.kda.ops import recurrent_forward as forward
from attn_gym.linear.kda.ops import (
    recurrent_fwd_no_state_op as _recurrent_fwd_no_state_op,
)
from attn_gym.linear.kda.ops import recurrent_fwd_op as _recurrent_fwd_op
from attn_gym.linear.kda.ops import (
    recurrent_fwd_paged_op as _recurrent_fwd_paged_op,
)


def _launch_kda_recurrent_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    *,
    store_final_state: bool,
    state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch the vector-log2 specialization used by recurrent KDA."""
    return launch_recurrent_delta_rule_fwd(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale=q.shape[-1] ** -0.5,
        scalar_ln_gate=False,
        store_final_state=store_final_state,
        state_indices=state_indices,
    )


def _kda_recurrent_fwd_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, final_state = _launch_kda_recurrent_fwd(
        q, k, v, gate, beta, initial_state, cu_seqlens, store_final_state=True
    )
    assert final_state is not None
    return output, final_state


def _kda_recurrent_fwd_no_state_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    return _launch_kda_recurrent_fwd(
        q, k, v, gate, beta, initial_state, cu_seqlens, store_final_state=False
    )[0]


def _kda_recurrent_fwd_paged_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    return _launch_kda_recurrent_fwd(
        q,
        k,
        v,
        gate,
        beta,
        state_cache,
        cu_seqlens,
        store_final_state=True,
        state_indices=state_indices,
    )[0]


__all__ = [
    "_recurrent_fwd_no_state_op",
    "_recurrent_fwd_op",
    "_recurrent_fwd_paged_op",
    "forward",
]
