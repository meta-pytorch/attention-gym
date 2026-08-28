"""Structural validation shared by public delta-rule operations."""

from __future__ import annotations

import sys
from numbers import Real

import torch

SUPPORTED_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def validate_delta_rule_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    *,
    op_name: str,
    gate_name: str,
    vector_gate: bool,
    allow_grouped_heads: bool,
) -> None:
    """Validate structural invariants shared by scalar- and vector-gated delta rules."""
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, H, K], got {tuple(q.shape)}")
    batch, tokens, key_heads, key_dim = q.shape
    if batch == 0 or tokens == 0 or key_heads == 0 or key_dim == 0:
        raise ValueError(f"q must have nonempty dimensions, got {tuple(q.shape)}")
    if k.shape != q.shape:
        raise ValueError(f"k must have shape {tuple(q.shape)}, got {tuple(k.shape)}")
    if v.ndim != 4 or v.shape[:2] != (batch, tokens) or v.shape[-1] < 1:
        raise ValueError(f"v must have shape [{batch}, {tokens}, H, V], got {tuple(v.shape)}")

    heads = v.shape[2]
    grouped_heads = heads != key_heads
    if grouped_heads and not (allow_grouped_heads and heads != 0 and heads % key_heads == 0):
        relation = "be a positive multiple of" if allow_grouped_heads else "match"
        raise ValueError(
            f"v heads must {relation} q heads for {op_name}, "
            f"got {heads} value heads for {key_heads} query heads"
        )

    gate_shape = (batch, tokens, heads, key_dim) if vector_gate else (batch, tokens, heads)
    if gate.shape != gate_shape:
        raise ValueError(f"{gate_name} must have shape {gate_shape}, got {tuple(gate.shape)}")
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
    expected_state = (state_batch, heads, v.shape[-1], key_dim)
    if initial_state is not None and initial_state.shape != expected_state:
        raise ValueError(
            f"initial_state must have shape {expected_state}, got {tuple(initial_state.shape)}"
        )

    tensors = (q, k, v, gate, beta) + (() if initial_state is None else (initial_state,))
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on the same device")


def resolve_scale(scale: float | None, key_dim: int) -> float:
    """Resolve a query-scale override to a validated float, defaulting to ``1/sqrt(K)``."""
    if scale is None:
        return key_dim**-0.5
    if not isinstance(scale, Real) or isinstance(scale, bool):
        raise TypeError("scale must be a real scalar or None")
    scale = float(scale)
    if (
        scale != scale  # noqa: PLR0124 - compile-safe NaN check
        or scale <= 0
        or scale > sys.float_info.max
    ):
        raise ValueError(f"scale must be finite and positive, got {scale}")
    return scale


def validate_has_initial_state(
    has_initial_state: torch.Tensor | None,
    num_sequences: int,
    device: torch.device,
) -> None:
    """Validate the optional per-sequence fresh-slot mask shared by paged delta-rule ops."""
    if has_initial_state is None:
        return
    if (
        tuple(has_initial_state.shape) != (num_sequences,)
        or has_initial_state.dtype != torch.bool
        or not has_initial_state.is_contiguous()
        or has_initial_state.device != device
    ):
        raise ValueError(
            "has_initial_state must be a contiguous bool tensor with one entry "
            "per sequence on the inputs' device"
        )


def validate_decode_inputs(
    packed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    *,
    op_name: str,
) -> tuple[int, int, int, int]:
    """Validate the family-independent fused-decode contract.

    Family-specific checks (packed channel layout, gate/beta/dt_bias shapes) stay with the
    caller. Returns ``(batch, heads, value_dim, key_dim)`` from the pool and buffer.
    """
    if packed_qkv.ndim != 2 or packed_qkv.shape[0] < 1 or packed_qkv.stride(1) != 1:
        raise ValueError("packed_qkv must have shape [B, C] and be contiguous within each token")
    if state_cache.ndim != 4:
        raise ValueError("state_cache must have shape [num_slots, H, V, K]")
    num_slots, heads, value_dim, key_dim = state_cache.shape
    batch = packed_qkv.shape[0]
    if num_slots < 1 or heads < 1 or key_dim < 1 or value_dim < 1:
        raise ValueError(
            f"state_cache must have nonempty dimensions, got {tuple(state_cache.shape)}"
        )
    if A_log.shape != (heads,) or A_log.dtype != torch.float32 or not A_log.is_contiguous():
        raise ValueError(f"A_log must be contiguous float32 with shape ({heads},)")
    if state_cache.dtype != torch.float32:
        raise TypeError("state_cache must use float32")
    if state_cache.stride()[1:] != (value_dim * key_dim, key_dim, 1):
        raise TypeError("state_cache must be contiguous within each [H, V, K] slot")
    if state_cache.stride(0) < heads * key_dim * value_dim:
        raise ValueError("state_cache slots must not overlap")
    if (
        state_indices.shape != (batch,)
        or state_indices.dtype != torch.int32
        or not state_indices.is_contiguous()
    ):
        raise ValueError(f"state_indices must be contiguous int32 with shape ({batch},)")
    validate_has_initial_state(has_initial_state, batch, packed_qkv.device)
    if key_dim > 256:
        raise ValueError(f"{op_name} requires K in [1, 256], got {key_dim}")

    device = packed_qkv.device
    all_tensors = (raw_gate, raw_beta, A_log, dt_bias, state_cache, state_indices)
    if any(tensor.device != device for tensor in all_tensors):
        raise ValueError(f"all {op_name} inputs must be on the same device")
    activation_tensors = (packed_qkv, raw_gate, raw_beta)
    if any(tensor.dtype not in SUPPORTED_ACTIVATION_DTYPES for tensor in activation_tensors):
        supported = ", ".join(str(dtype) for dtype in SUPPORTED_ACTIVATION_DTYPES)
        raise TypeError(f"decode activation inputs must use one of {supported}")
    return batch, heads, value_dim, key_dim


def resolve_decode_out(
    packed_qkv: torch.Tensor,
    out: torch.Tensor | None,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    """Allocate the decode output or validate a caller-owned buffer."""
    if out is None:
        return packed_qkv.new_empty(expected_shape)
    if out.shape != expected_shape:
        raise ValueError(f"out must have shape {expected_shape}, got {tuple(out.shape)}")
    if out.dtype != packed_qkv.dtype:
        raise TypeError(f"out must use packed_qkv.dtype ({packed_qkv.dtype}), got {out.dtype}")
    if out.device != packed_qkv.device:
        raise ValueError("out must be on packed_qkv.device")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    return out


def validate_paged_state(
    q: torch.Tensor,
    v: torch.Tensor,
    state_cache: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None = None,
) -> None:
    """Validate the shared mutable ``[slots, H, V, K]`` state contract.

    The pool carries one state per value head, so grouped-head callers size ``H`` from ``v``.
    """
    expected_shape = (v.shape[2], v.shape[-1], q.shape[3])
    if state_cache.ndim != 4 or state_cache.shape[1:] != expected_shape:
        raise ValueError(
            "the paged state pool must have shape "
            f"[num_slots, {v.shape[2]}, {v.shape[-1]}, {q.shape[3]}], "
            f"got {tuple(state_cache.shape)}"
        )
    if state_cache.device != q.device:
        raise ValueError("the paged state pool must be on q.device")
    if state_cache.dtype != torch.float32:
        raise TypeError("the paged state pool must use float32")
    expected_inner_strides = (v.shape[-1] * q.shape[3], q.shape[3], 1)
    if state_cache.stride()[1:] != expected_inner_strides:
        raise TypeError("the paged state pool must be contiguous within each [H, V, K] slot")
    if state_cache.stride(0) < v.shape[2] * q.shape[3] * v.shape[-1]:
        raise ValueError("paged state pool slots must not overlap")

    num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    if (
        tuple(state_indices.shape) != (num_sequences,)
        or state_indices.dtype != torch.int32
        or not state_indices.is_contiguous()
        or state_indices.device != q.device
    ):
        raise ValueError(
            f"state_indices must be a contiguous int32 tensor of shape ({num_sequences},) "
            f"on q.device, got {tuple(state_indices.shape)} of {state_indices.dtype}"
        )
    validate_has_initial_state(has_initial_state, num_sequences, q.device)


__all__ = [
    "SUPPORTED_ACTIVATION_DTYPES",
    "resolve_decode_out",
    "resolve_scale",
    "validate_decode_inputs",
    "validate_delta_rule_inputs",
    "validate_has_initial_state",
    "validate_paged_state",
]
