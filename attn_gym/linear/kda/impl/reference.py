# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager FP32 reference execution behind the public KDA contract.

Wraps the naive oracles with the public packed semantics: FP32 compute
(autocast disabled), empty padding slots pass their state through, and output
rows past the terminal offset stay zero. Packed execution is eager-only because
it reads device offsets on the host before iterating over logical sequences.
"""

from __future__ import annotations

import torch

from attn_gym.linear._delta_rule.reference import packed_delta_rule_reference


def reference_kda(
    dense_op,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    scale: float,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run a dense reference op in FP32 under the public packed contract."""
    output_dtype = q.dtype
    q, k, v = (tensor.float() for tensor in (q, k, v))
    gate = gate.float()
    beta = beta.float()
    if q.shape[2] != v.shape[2]:
        # Grouped heads: expand each shared query/key head across its value-head group.
        # The gate already carries one decay per value head and passes through unexpanded.
        groups = v.shape[2] // q.shape[2]
        q, k = (tensor.repeat_interleave(groups, dim=2) for tensor in (q, k))
    if initial_state is not None:
        initial_state = initial_state.float()
    # ``.float()`` casts alone do not stop an active autocast region from
    # re-electing BF16/FP16 for the matmuls inside the oracles.
    with torch.autocast(device_type=q.device.type, enabled=False):
        if cu_seqlens is None:
            output, state = dense_op(
                q,
                k,
                v,
                gate,
                beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=output_final_state,
            )
        else:
            output, state = packed_delta_rule_reference(
                dense_op,
                q,
                k,
                v,
                gate,
                beta,
                initial_state,
                cu_seqlens,
                output_final_state,
                scale=scale,
            )
    return output.to(output_dtype), state


__all__ = ["reference_kda"]
