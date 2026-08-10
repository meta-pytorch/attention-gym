"""Public API for the gated delta rule attention operation."""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch

from attn_gym.linear.gdn.impl.reference import forward

Mode = Literal["auto", "chunked", "recurrent"]
Backend = Literal["auto", "eager"]


class GatedDeltaRuleOutput(NamedTuple):
    """Output and optional recurrent state from :func:`gated_delta_rule`."""

    output: torch.Tensor
    final_state: torch.Tensor | None = None


def gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
    mode: Mode = "auto",
    backend: Backend = "auto",
    chunk_size: int = 64,
) -> GatedDeltaRuleOutput:
    """Apply gated delta rule attention.

    Inputs use the SDPA layout ``[batch, heads, sequence, dimension]``. The recurrent state has
    shape ``[batch, heads, key_dimension, value_dimension]``.

    Args:
        query: Query tensor with shape ``[B, H, T, K]``.
        key: Key tensor with shape ``[B, H, T, K]``.
        value: Value tensor with shape ``[B, H, T, V]``.
        gate: Per-token scalar log-decay with shape ``[B, H, T]``.
        beta: Per-token write gate with shape ``[B, H, T]``.
        scale: Query scale. Defaults to ``1 / sqrt(K)``.
        initial_state: Initial recurrent state with shape ``[B, H, K, V]``.
        return_final_state: Include the final recurrent state in the result.
        mode: Execution form. ``"auto"`` selects recurrent execution for one token and chunked
            execution otherwise.
        backend: Implementation backend. ``"auto"`` currently selects the eager reference.
        chunk_size: Number of tokens per chunk in chunked mode.

    Returns:
        The attention output and, when requested, the final recurrent state.
    """
    _validate_inputs(query, key, value, gate, beta, initial_state, chunk_size)

    if mode not in ("auto", "chunked", "recurrent"):
        raise ValueError(f"Unsupported mode {mode!r}; expected 'auto', 'chunked', or 'recurrent'")
    if backend not in ("auto", "eager"):
        raise ValueError(f"Unsupported backend {backend!r}; expected 'auto' or 'eager'")

    selected_mode = ("recurrent" if query.shape[2] == 1 else "chunked") if mode == "auto" else mode

    output, final_state = forward(
        query,
        key,
        value,
        gate,
        beta,
        scale=scale,
        initial_state=initial_state,
        return_final_state=return_final_state,
        mode=selected_mode,
        chunk_size=chunk_size,
    )
    return GatedDeltaRuleOutput(output=output, final_state=final_state)


def _validate_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    chunk_size: int,
) -> None:
    """Validate backend-independent gated delta rule input invariants."""
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
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be greater than zero, got {chunk_size}")
