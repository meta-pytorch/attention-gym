"""Opt-in masking for fixed-capacity packed KDA tensors."""

from __future__ import annotations

import torch


def active_token_mask(x: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    """Construct a reusable predicate for the active packed token prefix.

    Construct the predicate inside CUDA Graph capture so replay recomputes it from
    the device-resident ``cu_seqlens[-1]`` endpoint.
    """
    if x.ndim < 2 or x.shape[0] != 1:
        raise ValueError(f"x must have packed shape [1, T, ...], got {tuple(x.shape)}")
    if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
        raise ValueError("cu_seqlens must have shape [num_sequences + 1]")
    if cu_seqlens.dtype != torch.int32:
        raise ValueError(f"cu_seqlens must have dtype torch.int32, got {cu_seqlens.dtype}")
    if cu_seqlens.device != x.device:
        raise ValueError(
            f"cu_seqlens must be on the same device as x, got {cu_seqlens.device} and {x.device}"
        )
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")

    token = torch.arange(x.shape[1], device=x.device, dtype=torch.int32)
    return token < cu_seqlens[-1]


def mask_inactive_tokens(
    x: torch.Tensor,
    active_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Zero inactive token values and derivatives using a reusable mask.

    Passing ``None`` returns ``x`` unchanged, which keeps the default dense and
    exact-packed paths free of masking operations.
    """
    if active_mask is None:
        return x
    if x.ndim < 2 or x.shape[0] != 1:
        raise ValueError(f"x must have packed shape [1, T, ...], got {tuple(x.shape)}")
    if active_mask.shape != (x.shape[1],):
        raise ValueError(
            f"active_mask must have shape [{x.shape[1]}], got {tuple(active_mask.shape)}"
        )
    mask = active_mask.reshape(1, -1, *((1,) * (x.ndim - 2)))
    return torch.where(mask, x, 0)


def mask_inactive_token_gradients(
    x: torch.Tensor,
    active_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Preserve values while blocking inactive automatic-differentiation paths.

    Place this barrier between a parameterized producer and a primitive that ignores
    inactive forward rows but leaves their input-gradient suffix undefined. The
    inactive branch is detached, so automatic-differentiation paths are zero there
    and subsequent derivatives inherit the same mask. Passing ``None`` returns ``x``
    unchanged.
    """
    if active_mask is None:
        return x
    if x.ndim < 2 or x.shape[0] != 1:
        raise ValueError(f"x must have packed shape [1, T, ...], got {tuple(x.shape)}")
    if active_mask.shape != (x.shape[1],):
        raise ValueError(
            f"active_mask must have shape [{x.shape[1]}], got {tuple(active_mask.shape)}"
        )
    mask = active_mask.reshape(1, -1, *((1,) * (x.ndim - 2)))
    return torch.where(mask, x, x.detach())


__all__ = [
    "active_token_mask",
    "mask_inactive_token_gradients",
    "mask_inactive_tokens",
]
