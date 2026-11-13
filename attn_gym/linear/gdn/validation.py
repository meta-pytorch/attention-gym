"""Validation shared by public gated delta rule operations."""

from __future__ import annotations

import torch

from attn_gym.linear._delta_rule.validation import validate_delta_rule_inputs


def cast_to_kernel_dtypes(
    gate: torch.Tensor, beta: torch.Tensor, initial_state: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Cast the gate, beta and entry state to the FP32 the GDN chunk kernels compute in.

    Q/K/V run as given. Each cast is an autograd op, so gradients come back in the caller's
    dtypes (a BF16 entry state gets a BF16 cotangent).
    """
    return gate.float(), beta.float(), None if initial_state is None else initial_state.float()


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
    if initial_state is not None and initial_state.dtype not in (compute_dtype, torch.bfloat16):
        raise ValueError(
            f"initial_state must have dtype {compute_dtype} or torch.bfloat16 for {q.dtype} q, "
            f"got {initial_state.dtype}"
        )


__all__ = ["cast_to_kernel_dtypes", "validate_gdn_inputs"]
