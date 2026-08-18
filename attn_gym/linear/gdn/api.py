"""Public gated delta rule operations."""

from __future__ import annotations

from typing import NamedTuple

import torch

from attn_gym.linear.gdn.impl.reference import chunk_forward, recurrent_forward
from attn_gym.linear.gdn.validation import validate_gdn_inputs
from attn_gym.linear.types import Impl, resolve_impl


class GatedDeltaRuleOutput(NamedTuple):
    """Output and optional recurrent state from a gated delta rule operation."""

    output: torch.Tensor
    final_state: torch.Tensor | None = None


def chunk_gdn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
    chunk_size: int = 64,
    impl: Impl | str = Impl.REFERENCE,
) -> GatedDeltaRuleOutput:
    """Apply chunk-parallel gated delta rule attention for training and prefill.

    Inputs use the SDPA layout ``[batch, heads, sequence, dimension]``. The scalar natural-log gate
    decays the previous state before each beta-scaled delta update, and the query reads the updated
    state. Chunking changes only the decomposition and floating-point order of that recurrence.

    Args:
        query: Query tensor with shape ``[B, H, T, K]``.
        key: Key tensor with shape ``[B, H, T, K]``.
        value: Value tensor with shape ``[B, H, T, V]``.
        gate: Per-token scalar natural-log decay with shape ``[B, H, T]``.
        beta: Per-token write gate with shape ``[B, H, T]``.
        scale: Query scale. Defaults to ``1 / sqrt(K)``.
        initial_state: Initial recurrent state with shape ``[B, H, K, V]``.
        return_final_state: Include the final recurrent state in the result.
        chunk_size: Number of tokens per chunk.
        impl: ``"reference"`` uses eager PyTorch. ``"fused"`` is reserved for the optimized
            backend and currently raises ``NotImplementedError``.

    Returns:
        The attention output and, when requested, the final recurrent state.
    """
    selected_impl = resolve_impl(impl)
    validate_gdn_inputs(query, key, value, gate, beta, initial_state)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be greater than zero, got {chunk_size}")
    if selected_impl is Impl.FUSED:
        raise NotImplementedError("chunk_gdn impl='fused' is not implemented yet")

    output, final_state = chunk_forward(
        query,
        key,
        value,
        gate,
        beta,
        scale=scale,
        initial_state=initial_state,
        return_final_state=return_final_state,
        chunk_size=chunk_size,
    )
    return GatedDeltaRuleOutput(output=output, final_state=final_state)


def recurrent_gdn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    return_final_state: bool = False,
    impl: Impl | str = Impl.REFERENCE,
) -> GatedDeltaRuleOutput:
    """Apply recurrent gated delta rule attention for decoding and inference prefill.

    The recurrence consumes tokens in order, carrying an explicit ``[B, H, K, V]`` state. Inputs
    and outputs use the SDPA layout ``[batch, heads, sequence, dimension]``.

    Args:
        query: Query tensor with shape ``[B, H, T, K]``.
        key: Key tensor with shape ``[B, H, T, K]``.
        value: Value tensor with shape ``[B, H, T, V]``.
        gate: Per-token scalar natural-log decay with shape ``[B, H, T]``.
        beta: Per-token write gate with shape ``[B, H, T]``.
        scale: Query scale. Defaults to ``1 / sqrt(K)``.
        initial_state: Initial recurrent state with shape ``[B, H, K, V]``.
        return_final_state: Include the final recurrent state in the result.
        impl: ``"reference"`` uses eager PyTorch. ``"fused"`` is reserved for the optimized
            backend and currently raises ``NotImplementedError``.

    Returns:
        The attention output and, when requested, the final recurrent state.
    """
    selected_impl = resolve_impl(impl)
    validate_gdn_inputs(query, key, value, gate, beta, initial_state)
    if selected_impl is Impl.FUSED:
        raise NotImplementedError("recurrent_gdn impl='fused' is not implemented yet")

    output, final_state = recurrent_forward(
        query,
        key,
        value,
        gate,
        beta,
        scale=scale,
        initial_state=initial_state,
        return_final_state=return_final_state,
    )
    return GatedDeltaRuleOutput(output=output, final_state=final_state)


__all__ = ["GatedDeltaRuleOutput", "chunk_gdn", "recurrent_gdn"]
