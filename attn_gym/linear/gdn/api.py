"""Public gated delta rule operations."""

from __future__ import annotations

import torch

from attn_gym.linear.gdn.impl.reference import (
    chunk_forward,
    recurrent_forward,
    reference_gdn,
)
from attn_gym.linear.gdn.validation import validate_gdn_inputs
from attn_gym.linear.types import Impl, resolve_impl


def chunk_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    output_final_state: bool = False,
    impl: Impl | str = Impl.REFERENCE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply chunk-parallel gated delta rule attention for training and prefill.

    Inputs use the token-major layout ``[batch, sequence, heads, dimension]``. The scalar
    natural-log gate decays the previous state before each beta-scaled delta update, and the query
    reads the updated state. Chunking changes only the decomposition and floating-point order of
    that recurrence. FP16 and BF16 inputs use FP32 recurrence math and state.

    Args:
        q: Queries shaped ``[B, T, H, K]``.
        k: Keys shaped like ``q`` and using the same dtype.
        v: Values shaped ``[B, T, H, V]`` and using the same dtype as ``q``.
        gate: Floating per-token scalar natural-log decay shaped ``[B, T, H]``.
        beta: Floating per-token write gate shaped ``[B, T, H]``.
        initial_state: Initial recurrent state shaped ``[B, H, K, V]`` in the recurrence compute
            dtype (FP32 for FP16/BF16 QKV).
        scale: Query scale. Defaults to ``1 / sqrt(K)``.
        output_final_state: Return the final recurrent state with the output.
        impl: ``"reference"`` uses eager PyTorch. ``"fused"`` is reserved for the optimized
            backend and currently raises ``NotImplementedError``.

    Returns:
        The output in ``q.dtype`` and either the final recurrent state or ``None``.
    """
    selected_impl = resolve_impl(impl)
    validate_gdn_inputs(q, k, v, gate, beta, initial_state)
    if selected_impl is Impl.FUSED:
        raise NotImplementedError("chunk_gdn impl='fused' is not implemented yet")

    return reference_gdn(
        chunk_forward,
        q,
        k,
        v,
        gate,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )


def recurrent_gdn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    output_final_state: bool = False,
    impl: Impl | str = Impl.REFERENCE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply recurrent gated delta rule attention for decoding and inference prefill.

    The recurrence consumes tokens in order, carrying an explicit ``[B, H, K, V]`` state. Inputs
    and outputs use the token-major layout ``[batch, sequence, heads, dimension]``. FP16 and BF16
    inputs use FP32 recurrence math and state.

    Args:
        q: Queries shaped ``[B, T, H, K]``.
        k: Keys shaped like ``q`` and using the same dtype.
        v: Values shaped ``[B, T, H, V]`` and using the same dtype as ``q``.
        gate: Floating per-token scalar natural-log decay shaped ``[B, T, H]``.
        beta: Floating per-token write gate shaped ``[B, T, H]``.
        initial_state: Initial recurrent state shaped ``[B, H, K, V]`` in the recurrence compute
            dtype (FP32 for FP16/BF16 QKV).
        scale: Query scale. Defaults to ``1 / sqrt(K)``.
        output_final_state: Return the final recurrent state with the output.
        impl: ``"reference"`` uses eager PyTorch. ``"fused"`` is reserved for the optimized
            backend and currently raises ``NotImplementedError``.

    Returns:
        The output in ``q.dtype`` and either the final recurrent state or ``None``.
    """
    selected_impl = resolve_impl(impl)
    validate_gdn_inputs(q, k, v, gate, beta, initial_state)
    if selected_impl is Impl.FUSED:
        raise NotImplementedError("recurrent_gdn impl='fused' is not implemented yet")

    return reference_gdn(
        recurrent_forward,
        q,
        k,
        v,
        gate,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )


__all__ = ["chunk_gdn", "recurrent_gdn"]
