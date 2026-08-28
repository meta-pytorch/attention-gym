"""Validation shared by public gated delta rule operations."""

from __future__ import annotations

import torch

from attn_gym.linear._delta_rule.validation import validate_delta_rule_inputs


def validate_gdn_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None = None,
) -> None:
    """Validate backend-independent gated delta rule tensor invariants."""
    validate_delta_rule_inputs(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        op_name="gated delta rule",
        gate_name="gate",
        vector_gate=False,
        allow_grouped_heads=True,
    )
    tensors = (q, k, v, gate, beta) + (() if initial_state is None else (initial_state,))
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise ValueError("all inputs must have floating-point dtypes")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v must have the same dtype")

    compute_dtype = torch.promote_types(q.dtype, torch.float32)
    if initial_state is not None and initial_state.dtype != compute_dtype:
        raise ValueError(
            f"initial_state must have dtype {compute_dtype} for {q.dtype} q, "
            f"got {initial_state.dtype}"
        )


__all__ = ["validate_gdn_inputs"]
