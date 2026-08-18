"""Validation shared by public gated delta rule operations."""

from __future__ import annotations

import torch


def validate_gdn_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> None:
    """Validate backend-independent gated delta rule tensor invariants."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError(
            "query, key, and value must have shape [batch, heads, sequence, dimension]"
        )
    if gate.ndim != 3 or beta.ndim != 3:
        raise ValueError("gate and beta must have shape [batch, heads, sequence]")
    if query.shape != key.shape:
        raise ValueError(
            f"query and key must have the same shape, got {query.shape} and {key.shape}"
        )

    batch, heads, sequence, key_dimension = query.shape
    if sequence == 0:
        raise ValueError("sequence length must be greater than zero")
    if value.shape[:3] != (batch, heads, sequence):
        raise ValueError("value must match query in its batch, head, and sequence dimensions")
    if gate.shape != (batch, heads, sequence) or beta.shape != (batch, heads, sequence):
        raise ValueError("gate and beta must match query's batch, head, and sequence dimensions")
    if initial_state is not None:
        expected_state_shape = (batch, heads, key_dimension, value.shape[-1])
        if initial_state.shape != expected_state_shape:
            raise ValueError(
                f"initial_state must have shape {expected_state_shape}, got {initial_state.shape}"
            )

    tensors = (query, key, value, gate, beta)
    if initial_state is not None:
        tensors += (initial_state,)
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise ValueError("all inputs must have floating-point dtypes")
    if any(tensor.device != query.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on the same device")
    if any(tensor.dtype != query.dtype for tensor in tensors[1:]):
        raise ValueError("all inputs must have the same dtype")


__all__ = ["validate_gdn_inputs"]
