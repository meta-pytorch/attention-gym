"""Validation shared by paged delta-rule operations."""

from __future__ import annotations

import torch


def validate_paged_state(
    q: torch.Tensor,
    v: torch.Tensor,
    state_cache: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None = None,
) -> None:
    """Validate the shared mutable ``[slots, H, V, K]`` state contract."""
    expected_shape = (q.shape[2], v.shape[-1], q.shape[3])
    if state_cache.ndim != 4 or state_cache.shape[1:] != expected_shape:
        raise ValueError(
            "the paged state pool must have shape "
            f"[num_slots, {q.shape[2]}, {v.shape[-1]}, {q.shape[3]}], "
            f"got {tuple(state_cache.shape)}"
        )
    if state_cache.device != q.device:
        raise ValueError("the paged state pool must be on q.device")
    if state_cache.dtype != torch.float32:
        raise TypeError("the paged state pool must use float32")
    expected_inner_strides = (v.shape[-1] * q.shape[3], q.shape[3], 1)
    if state_cache.stride()[1:] != expected_inner_strides:
        raise TypeError("the paged state pool must be contiguous within each [H, V, K] slot")
    if state_cache.stride(0) < q.shape[2] * q.shape[3] * v.shape[-1]:
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
    if has_initial_state is not None and (
        tuple(has_initial_state.shape) != (num_sequences,)
        or has_initial_state.dtype != torch.bool
        or not has_initial_state.is_contiguous()
        or has_initial_state.device != q.device
    ):
        raise ValueError(
            "has_initial_state must be a contiguous bool tensor with one entry "
            "per sequence on q.device"
        )


__all__ = ["validate_paged_state"]
