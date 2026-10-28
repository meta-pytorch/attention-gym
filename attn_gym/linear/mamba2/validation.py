"""Validation shared by public Mamba2 operations."""

from __future__ import annotations

import math

import torch

from attn_gym._cu_seqlens import validate_cu_seqlens


def validate_mamba2_inputs(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    *,
    dt_bias: torch.Tensor | None,
    dt_limit: tuple[float, float],
    cu_seqlens: torch.Tensor | None,
) -> None:
    """Validate backend-independent Mamba2 tensor invariants."""
    if x.ndim != 4 or 0 in x.shape:
        raise ValueError(f"x must have nonempty shape [batch, T, H, P], got {tuple(x.shape)}")
    batch, tokens, heads, head_dim = x.shape
    if dt.shape != (batch, tokens, heads):
        raise ValueError(f"dt must have shape {(batch, tokens, heads)}, got {tuple(dt.shape)}")
    if A.shape != (heads,):
        raise ValueError(f"A must have shape {(heads,)}, got {tuple(A.shape)}")
    if B.ndim != 4 or B.shape[:2] != (batch, tokens) or 0 in B.shape[2:]:
        raise ValueError(f"B must have shape [{batch}, {tokens}, G, N], got {tuple(B.shape)}")
    if C.shape != B.shape:
        raise ValueError(f"C must have shape {tuple(B.shape)}, got {tuple(C.shape)}")
    groups, state_dim = B.shape[2:]
    if heads % groups:
        raise ValueError(f"heads ({heads}) must be a multiple of the B/C groups ({groups})")
    if D is not None and D.shape not in ((heads,), (heads, head_dim)):
        raise ValueError(f"D must have shape {(heads,)} or {(heads, head_dim)}, got {D.shape}")
    if z is not None and z.shape != x.shape:
        raise ValueError(f"z must have shape {tuple(x.shape)}, got {tuple(z.shape)}")
    if dt_bias is not None and dt_bias.shape != (heads,):
        raise ValueError(f"dt_bias must have shape {(heads,)}, got {tuple(dt_bias.shape)}")
    low, high = dt_limit
    if not (0.0 <= low <= high) or math.isnan(high):
        raise ValueError(f"dt_limit must satisfy 0 <= low <= high, got {dt_limit}")

    if cu_seqlens is not None:
        validate_cu_seqlens(batch, x.device, cu_seqlens=cu_seqlens)
    states = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    expected_state = (states, heads, head_dim, state_dim)
    if initial_state is not None and initial_state.shape != expected_state:
        raise ValueError(
            f"initial_state must have shape {expected_state}, got {tuple(initial_state.shape)}"
        )

    activations = (x, B, C) + (() if z is None else (z,))
    if any(tensor.dtype != x.dtype for tensor in activations):
        raise ValueError("x, B, C, and z must have the same dtype")
    optional = (D, z, initial_state, dt_bias)
    tensors = (x, dt, A, B, C) + tuple(tensor for tensor in optional if tensor is not None)
    if not all(tensor.is_floating_point() for tensor in tensors):
        raise ValueError("all inputs must have floating-point dtypes")
    if any(tensor.device != x.device for tensor in tensors):
        raise ValueError("all inputs must be on the same device")


def validate_fused_inputs(x: torch.Tensor, B: torch.Tensor) -> None:
    """Validate the device, dtype, and state width the fused kernels support."""
    if not x.is_cuda:
        raise ValueError("impl='fused' requires CUDA tensors")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("impl='fused' requires float16 or bfloat16 x, B, C, and z")
    state_dim = B.shape[-1]
    if state_dim not in (16, 32, 64, 128, 256):
        raise ValueError(f"impl='fused' requires N in {{16, 32, 64, 128, 256}}, got {state_dim}")


def validate_recurrent_fused_inputs(x: torch.Tensor, B: torch.Tensor) -> None:
    """Validate the device, dtype, and state width of the fused token-by-token scan."""
    if not x.is_cuda:
        raise ValueError("impl='fused' requires CUDA tensors")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("the fused recurrent scan requires float16, bfloat16, or float32 x")
    if B.shape[-1] > 256:
        raise ValueError(f"the fused recurrent scan requires N <= 256, got {B.shape[-1]}")


def validate_decode_inputs(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    *,
    dt_bias: torch.Tensor | None,
    dt_limit: tuple[float, float],
) -> None:
    """Validate one-token decode inputs: ``x``/``z`` ``[batch, H, P]``, ``B``/``C`` ``[batch, G, N]``.

    Every input may be a strided view (for example a slice of a packed projection buffer) as
    long as its last axis is contiguous.
    """
    if x.ndim != 3:
        raise ValueError(f"x must have shape [batch, H, P], got {tuple(x.shape)}")
    if dt.ndim != 2 or B.ndim != 3:
        raise ValueError("dt must have shape [batch, H] and B, C shape [batch, G, N]")
    validate_mamba2_inputs(
        x.unsqueeze(1),
        dt.unsqueeze(1),
        A,
        B.unsqueeze(1),
        C.unsqueeze(1),
        D,
        None if z is None else z.unsqueeze(1),
        None,
        dt_bias=dt_bias,
        dt_limit=dt_limit,
        cu_seqlens=None,
    )
    tensors = (x, dt, B, C) + (() if z is None else (z,))
    if any(tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError("decode inputs must be contiguous along their last axis")
    validate_recurrent_fused_inputs(x, B)


__all__ = [
    "validate_decode_inputs",
    "validate_fused_inputs",
    "validate_mamba2_inputs",
    "validate_recurrent_fused_inputs",
]
