"""Public gated delta rule operations."""

from __future__ import annotations

import torch

from attn_gym.linear.gdn.impl.reference import chunk_forward, recurrent_forward, reference_gdn
from attn_gym.linear.gdn.ops import recurrent_forward as fused_recurrent_forward
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
    cu_seqlens: torch.Tensor | None = None,
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
        initial_state: Initial recurrent state shaped ``[N, H, K, V]`` in the recurrence compute
            dtype, where ``N`` is the number of logical sequences.
        cu_seqlens: Optional packed offsets shaped ``[N + 1]`` for batch-one inputs. They start at
            zero, never decrease, and may end before ``T``; output beyond the terminal offset is
            unspecified.
        scale: Query scale. Defaults to ``1 / sqrt(K)``.
        output_final_state: Return the final recurrent state with the output.
        impl: ``"reference"`` uses eager PyTorch. ``"fused"`` is reserved for the optimized
            backend and currently raises ``NotImplementedError``.

    Returns:
        The output in ``q.dtype`` and either the final recurrent state or ``None``.
    """
    selected_impl = resolve_impl(impl)
    validate_gdn_inputs(q, k, v, gate, beta, initial_state, cu_seqlens)
    if selected_impl is Impl.FUSED:
        raise NotImplementedError("chunk_gdn impl='fused' is not implemented yet")

    return reference_gdn(
        chunk_forward,
        q,
        k,
        v,
        gate,
        beta,
        scale=q.shape[-1] ** -0.5 if scale is None else scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
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
    cu_seqlens: torch.Tensor | None = None,
    scale: float | None = None,
    output_final_state: bool = False,
    autotune: bool = True,
    impl: Impl | str = Impl.REFERENCE,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply recurrent gated delta rule attention for decoding and inference prefill.

    The recurrence consumes tokens in order, carrying an explicit ``[N, H, K, V]`` state. Inputs
    and outputs use the token-major layout ``[batch, sequence, heads, dimension]``. FP16 and BF16
    inputs use FP32 recurrence math and state.

    Args:
        q: Queries shaped ``[B, T, H, K]``.
        k: Keys shaped like ``q`` and using the same dtype.
        v: Values shaped ``[B, T, H, V]`` and using the same dtype as ``q``.
        gate: Floating per-token scalar natural-log decay shaped ``[B, T, H]``.
        beta: Floating per-token write gate shaped ``[B, T, H]``.
        initial_state: Initial recurrent state shaped ``[N, H, K, V]`` in the recurrence compute
            dtype, where ``N`` is the number of logical sequences.
        cu_seqlens: Optional packed offsets shaped ``[N + 1]`` for batch-one inputs. They start at
            zero, never decrease, and may end before ``T``; output beyond the terminal offset is
            unspecified.
        scale: Query scale. Defaults to ``1 / sqrt(K)``.
        output_final_state: Return the final recurrent state with the output.
        autotune: Benchmark candidate value-tile sizes for the fused implementation when true;
            winners are cached and reused.
        impl: ``"fused"`` uses the inference-only Triton scan; ``"reference"`` uses eager
            PyTorch with autograd support.

    Returns:
        The output in ``q.dtype`` and either the final recurrent state or ``None``.
    """
    selected_impl = resolve_impl(impl)
    validate_gdn_inputs(q, k, v, gate, beta, initial_state, cu_seqlens)
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if selected_impl is Impl.FUSED:
        return fused_recurrent_forward(
            q,
            k,
            v,
            gate,
            beta,
            initial_state,
            cu_seqlens=cu_seqlens,
            scale=scale,
            output_final_state=output_final_state,
            autotune=autotune,
        )

    return reference_gdn(
        recurrent_forward,
        q,
        k,
        v,
        gate,
        beta,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=output_final_state,
    )


__all__ = ["chunk_gdn", "recurrent_gdn"]
