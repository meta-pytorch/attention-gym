"""Shared packed execution for eager delta-rule reference implementations."""

from __future__ import annotations

from collections.abc import Callable
from itertools import pairwise

import torch

DenseReference = Callable[..., tuple[torch.Tensor, torch.Tensor | None]]


def packed_delta_rule_reference(
    dense_op: DenseReference,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    output_final_state: bool,
    *,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Evaluate packed logical sequences independently through a dense reference op."""
    heads, key_dim, value_dim = v.shape[2], q.shape[3], v.shape[-1]
    num_sequences = cu_seqlens.shape[0] - 1
    output = torch.zeros_like(v)
    final_state = None
    if output_final_state:
        final_state = (
            q.new_zeros(num_sequences, heads, value_dim, key_dim)
            if initial_state is None
            else initial_state.clone()
        )

    offsets = cu_seqlens.cpu().tolist()
    if (
        offsets[0] != 0
        or any(begin > end for begin, end in pairwise(offsets))
        or offsets[-1] > q.shape[1]
    ):
        raise ValueError(
            "cu_seqlens offsets must start at zero, be nondecreasing, and end within "
            "the physical token capacity"
        )

    for sequence, (begin, end) in enumerate(pairwise(offsets)):
        if begin == end:
            continue
        span = slice(begin, end)
        span_output, span_state = dense_op(
            q[:, span],
            k[:, span],
            v[:, span],
            gate[:, span],
            beta[:, span],
            initial_state=None
            if initial_state is None
            else initial_state[sequence : sequence + 1],
            scale=scale,
            output_final_state=output_final_state,
        )
        output[:, span] = span_output
        if final_state is not None:
            assert span_state is not None
            final_state[sequence] = span_state[0]
    return output, final_state


__all__ = ["packed_delta_rule_reference"]
