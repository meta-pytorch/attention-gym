"""Public API for the gated delta rule attention operation."""

from __future__ import annotations

from typing import Literal, NamedTuple

import torch

from attn_gym.linear.gdn.impl.reference import forward
from attn_gym.linear.gdn.validation import validate_gated_delta_rule_inputs

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
    shape ``[batch, heads, key_dimension, value_dimension]``. For each token, the scalar natural-log
    gate first decays the previous state, then the beta-scaled delta rule updates it, and the query
    reads the updated state:

    ``decayed_state = exp(gate) * state``

    ``residual = beta * (value - key @ decayed_state)``

    ``state = decayed_state + outer(key, residual)``

    ``output = scale * query @ state``

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
    validate_gated_delta_rule_inputs(
        query,
        key,
        value,
        gate,
        beta,
        initial_state,
        mode=mode,
        backend=backend,
        chunk_size=chunk_size,
    )
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
