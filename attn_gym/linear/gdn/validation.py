"""Validation shared by public gated delta rule operations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, NamedTuple

import torch

from attn_gym.linear._delta_rule.validation import validate_delta_rule_inputs


class ResolvedKernelOptions(NamedTuple):
    """Validated ``chunk_gdn`` backend selection and cuDNN split switches."""

    backend: Literal["fused", "cudnn"]
    split_backward: bool
    split_forward: bool


def resolve_kernel_options(
    kernel_options: Mapping[str, object] | None,
) -> ResolvedKernelOptions:
    """Validate chunk backend options while keeping the repo-local path as default."""
    if kernel_options is None:
        return ResolvedKernelOptions("fused", False, False)
    unknown = kernel_options.keys() - ResolvedKernelOptions._fields
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unsupported chunk_gdn kernel options: {names}")
    backend = kernel_options.get("backend", "fused")
    if backend not in ("fused", "cudnn"):
        raise ValueError("kernel_options['backend'] must be 'fused' or 'cudnn'")
    splits = []
    for name in ("split_backward", "split_forward"):
        value = kernel_options.get(name, False)
        if not isinstance(value, bool):
            raise TypeError(f"kernel_options['{name}'] must be a bool")
        if value and backend != "cudnn":
            raise ValueError(f"{name} requires kernel_options['backend']='cudnn'")
        splits.append(value)
    return ResolvedKernelOptions(backend, *splits)


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


__all__ = ["ResolvedKernelOptions", "resolve_kernel_options", "validate_gdn_inputs"]
