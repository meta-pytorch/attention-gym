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

from itertools import pairwise

import torch


def _packed_reference(
    dense_op,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate packed sequences independently through a dense reference op.

    The offsets are read on the host, so the reference path synchronizes; the
    fused implementations keep them device-resident.
    """
    heads, key_dim, value_dim = q.shape[2], q.shape[3], v.shape[-1]
    num_sequences = cu_seqlens.shape[0] - 1
    output = torch.zeros_like(v)
    final_state = None
    if output_final_state:
        final_state = (
            q.new_zeros(num_sequences, heads, key_dim, value_dim)
            if initial_state is None
            else initial_state.clone()
        )
    offsets = cu_seqlens.cpu().tolist()
    if (
        offsets[0] != 0
        or any(start > end for start, end in pairwise(offsets))
        or offsets[-1] > q.shape[1]
    ):
        raise ValueError(
            "cu_seqlens offsets must start at zero, be nondecreasing, and end within "
            "the physical token capacity"
        )
    for sequence, (start, end) in enumerate(pairwise(offsets)):
        if start == end:
            continue
        span = slice(start, end)
        span_output, span_state = dense_op(
            q[:, span],
            k[:, span],
            v[:, span],
            gate[:, span],
            beta[:, span],
            initial_state=None
            if initial_state is None
            else initial_state[sequence : sequence + 1],
            output_final_state=output_final_state,
        )
        output[:, span] = span_output
        if final_state is not None:
            final_state[sequence] = span_state[0]
    return output, final_state


def reference_kda(
    dense_op,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run a dense reference op in FP32 under the public packed contract."""
    output_dtype = q.dtype
    q, k, v = (tensor.float() for tensor in (q, k, v))
    gate = gate.float()
    beta = beta.float()
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
                initial_state=initial_state,
                output_final_state=output_final_state,
            )
        else:
            output, state = _packed_reference(
                dense_op, q, k, v, gate, beta, initial_state, cu_seqlens, output_final_state
            )
    return output.to(output_dtype), state


__all__ = ["reference_kda"]
