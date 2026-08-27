"""Validation shared by public gated delta rule operations."""

from __future__ import annotations

import torch


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
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, HK, K], got {tuple(q.shape)}")
    batch, tokens, key_heads, key_dim = q.shape
    if batch == 0 or tokens == 0 or key_heads == 0 or key_dim == 0:
        raise ValueError(f"q must have nonempty dimensions, got {tuple(q.shape)}")
    if k.shape != q.shape:
        raise ValueError(f"k must have shape {tuple(q.shape)}, got {tuple(k.shape)}")
    if v.ndim != 4 or v.shape[:2] != (batch, tokens) or v.shape[-1] == 0:
        raise ValueError(f"v must have shape [{batch}, {tokens}, H, V], got {tuple(v.shape)}")
    heads = v.shape[2]
    if heads == 0 or heads % key_heads != 0:
        raise ValueError(
            f"v heads must be a positive multiple of q heads for grouped-head attention, "
            f"got {heads} value heads for {key_heads} query heads"
        )
    if gate.shape != (batch, tokens, heads):
        raise ValueError(f"gate must have shape {(batch, tokens, heads)}, got {tuple(gate.shape)}")
    if beta.shape != (batch, tokens, heads):
        raise ValueError(f"beta must have shape {(batch, tokens, heads)}, got {tuple(beta.shape)}")

    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("packed cu_seqlens require q to have batch size one")
        if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
            raise ValueError("cu_seqlens must have shape [num_sequences + 1]")
        if (
            cu_seqlens.dtype != torch.int32
            or not cu_seqlens.is_contiguous()
            or cu_seqlens.device != q.device
        ):
            raise ValueError("cu_seqlens must be contiguous int32 on q.device")

    state_batch = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    if initial_state is not None:
        expected_state_shape = (state_batch, heads, key_dim, v.shape[-1])
        if initial_state.shape != expected_state_shape:
            raise ValueError(
                f"initial_state must have shape {expected_state_shape}, got {initial_state.shape}"
            )

    tensors = (q, k, v, gate, beta) + (() if initial_state is None else (initial_state,))
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise ValueError("all inputs must have floating-point dtypes")
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on the same device")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v must have the same dtype")

    compute_dtype = torch.promote_types(q.dtype, torch.float32)
    if initial_state is not None and initial_state.dtype != compute_dtype:
        raise ValueError(
            f"initial_state must have dtype {compute_dtype} for {q.dtype} q, "
            f"got {initial_state.dtype}"
        )


__all__ = ["validate_gdn_inputs"]
