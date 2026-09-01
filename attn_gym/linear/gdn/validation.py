"""Validation shared by public gated delta rule operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import torch

from attn_gym.linear._delta_rule.validation import validate_delta_rule_inputs

_KERNEL_OPTION_NAMES = frozenset({"backend"})


def resolve_kernel_options(
    kernel_options: Mapping[str, object] | None,
) -> Literal["fused", "mega"]:
    """Validate chunk backend options while keeping the repo-local path as default."""
    if kernel_options is None:
        return "fused"
    unknown = kernel_options.keys() - _KERNEL_OPTION_NAMES
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unsupported chunk_gdn kernel options: {names}")
    backend = kernel_options.get("backend", "fused")
    if backend not in ("fused", "mega"):
        raise ValueError("kernel_options['backend'] must be 'fused' or 'mega'")
    return backend


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


__all__ = ["resolve_kernel_options", "validate_gdn_inputs"]
