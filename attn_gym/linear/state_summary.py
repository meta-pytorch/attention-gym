# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Affine state-summary algebra shared by every delta-rule variant.

Each delta-rule token step is affine in the V-major recurrent state ``H: [V, K]`` (one per value
head ``HV``; GQA key heads are already expanded by the factor kernels)::

    H_t = H_{t-1} @ A_t + B_t      A_t = diag(exp g_t) (I - beta_t k_t k_t^T),  B_t = beta_t v_t k_t^T

so any token range collapses to one FP32 map ``H_out = H_in @ A + B``, packed here as
``[HV, V + K, K] = [bias; transition]``. Reverse summaries pack the cotangent map
``dH_in = dH_out @ R + C`` the same way. These helpers are pure PyTorch; the per-op ``stages``
modules produce the summaries and ``attn_gym.linear.context_parallel`` moves them between ranks.
"""

from __future__ import annotations

import torch


def neutral_summary(
    heads: int, value_dim: int, key_dim: int, *, device: torch.device | str
) -> torch.Tensor:
    """Return the identity map ``[0; I]``: ``merge_state(state, neutral) == state``.

    Collectives need every participant to contribute a same-shaped tensor even when nobody
    consumes its slot; the identity is the harmless filler.
    """
    summary = torch.zeros(heads, value_dim + key_dim, key_dim, dtype=torch.float32, device=device)
    summary[:, value_dim:, :] = torch.eye(key_dim, dtype=torch.float32, device=device)
    return summary


def merge_state(state: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
    """Apply a packed ``[bias; transition]`` summary to a V-major state: ``state @ A + B``."""
    value_dim = state.shape[-2]
    return state @ summary[..., value_dim:, :] + summary[..., :value_dim, :]


def compose_summaries(first: torch.Tensor, then: torch.Tensor) -> torch.Tensor:
    """Compose two summaries into the map that applies ``first`` and then ``then``.

    ``(A0, B0) ∘ (A1, B1) = (A0 @ A1, B0 @ A1 + B1)``. Folding predecessors from the zero state
    with ``merge_state`` gives the same entry state; composition is for scans that combine maps
    before any state is known.
    """
    value_dim = first.shape[-2] - first.shape[-1]
    bias, transition = first[..., :value_dim, :], first[..., value_dim:, :]
    return torch.cat((merge_state(bias, then), transition @ then[..., value_dim:, :]), dim=-2)


__all__ = ["compose_summaries", "merge_state", "neutral_summary"]
