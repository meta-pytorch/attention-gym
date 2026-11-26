"""Storage-dtype and MXFP8 scale metadata checks shared by the API and backend launchers."""

import torch
from torch import Tensor


def validate_precision(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    q_scale: Tensor | None,
    k_scale: Tensor | None,
) -> None:
    """Check storage/scaling contracts without reading device-resident values."""
    if not q.is_floating_point():
        raise TypeError(f"q must have a floating-point dtype, got {q.dtype}.")
    if k.dtype != q.dtype:
        raise ValueError(f"k must have the same dtype as q, but got {k.dtype} and {q.dtype}.")
    if q.dtype != torch.float8_e4m3fn:
        if q_scale is not None or k_scale is not None:
            raise ValueError("q_scale and k_scale are only supported for MXFP8 E4M3FN Q/K.")
        if weights.dtype != q.dtype:
            raise ValueError(
                f"weights must have the same dtype as q, but got {weights.dtype} and {q.dtype}."
            )
        return

    if weights.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("MXFP8 Q/K require BF16 or FP32 weights.")
    if q.shape[-1] % 32:
        raise ValueError("MXFP8 requires a head dimension divisible by 32.")
    if q_scale is None or k_scale is None:
        raise ValueError("MXFP8 Q/K require both q_scale and k_scale.")
    groups = q.shape[-1] // 32
    for name, scale, shape in (
        ("q_scale", q_scale, (*q.shape[:-1], groups)),
        ("k_scale", k_scale, (*k.shape[:-1], groups)),
    ):
        if not isinstance(scale, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(scale).__name__}.")
        if scale.dtype != torch.float8_e8m0fnu:
            raise TypeError(f"{name} must have dtype float8_e8m0fnu, got {scale.dtype}.")
        if tuple(scale.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(scale.shape)}.")
        if scale.device != q.device:
            raise ValueError(f"{name} must be on the same device as q.")
