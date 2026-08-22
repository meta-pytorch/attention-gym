from __future__ import annotations
import torch
from torch import Tensor


def exclusive_output_mod(Y: Tensor, V: Tensor) -> Tensor:
    """Remove each attention output vector's projection onto its value vector."""
    v_sq_norm = (V * V).sum(dim=-1, keepdim=True)
    y_dot_v = (Y * V).sum(dim=-1, keepdim=True)
    scale = torch.where(
        v_sq_norm > 0,
        y_dot_v / v_sq_norm.clamp_min(torch.finfo(V.dtype).tiny),
        torch.zeros_like(y_dot_v),
    )
    Z = Y - scale * V

    if Z.is_floating_point():
        residual_norm = Z.norm(dim=-1, keepdim=True)
        reference_norm = torch.maximum(
            Y.norm(dim=-1, keepdim=True),
            V.norm(dim=-1, keepdim=True),
        )
        cleanup_tol = torch.finfo(Z.dtype).eps * Z.shape[-1] * reference_norm
        Z = torch.where(
            (v_sq_norm > 0) & (residual_norm <= cleanup_tol),
            torch.zeros_like(Z),
            Z,
        )
    return Z
