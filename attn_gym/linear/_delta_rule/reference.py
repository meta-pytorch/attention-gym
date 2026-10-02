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


def reference_delta_rule(
    dense_op: DenseReference,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    compute_dtype: torch.dtype,
    scale: float,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run a dense reference op in ``compute_dtype`` under the public packed contract.

    Autocast is disabled, grouped query/key heads are expanded across their value-head
    group, empty padding slots pass their state through, and output rows past the
    terminal offset stay zero. Packed execution is eager-only because it reads device
    offsets on the host before iterating over logical sequences.
    """
    output_dtype = q.dtype
    q, k, v, gate, beta = (tensor.to(compute_dtype) for tensor in (q, k, v, gate, beta))
    if initial_state is not None:
        initial_state = initial_state.to(compute_dtype)
    if q.shape[2] != v.shape[2]:
        # Grouped heads: expand each shared query/key head across its value-head group.
        # The gate already carries one decay per value head and passes through unexpanded.
        groups = v.shape[2] // q.shape[2]
        q, k = (tensor.repeat_interleave(groups, dim=2) for tensor in (q, k))
    # Explicit casts alone do not stop an active autocast region from re-electing
    # BF16/FP16 for the matmuls inside the oracles.
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


__all__ = ["packed_delta_rule_reference", "reference_delta_rule"]
