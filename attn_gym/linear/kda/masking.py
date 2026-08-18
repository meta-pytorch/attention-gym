"""Opt-in masking for fixed-capacity packed KDA tensors.

Masking is bandwidth-critical under CUDA Graph over-capture: it runs on the
physical capacity buffer every step. On the eligible eager CUDA path the row
kernel never reads padding rows (it write-only zeros them) and the gradient
barrier aliases ``x`` in the forward. Compiled graphs and unsupported layouts
keep the ``torch.where`` form so Inductor owns fusion.
"""

from __future__ import annotations

import torch


def _mask_inactive_rows(x: torch.Tensor, active_mask: torch.Tensor) -> torch.Tensor:
    """Use the optional Triton row kernel, falling back to ordinary ATen masking."""
    try:
        from attn_gym.linear.kda.fwd.triton.masking import mask_inactive_rows
    except ModuleNotFoundError as error:
        if error.name is None or error.name.split(".", 1)[0] != "triton":
            raise
        mask = active_mask.reshape(1, -1, *((1,) * (x.ndim - 2)))
        return torch.where(mask, x, 0)

    return mask_inactive_rows(x, active_mask)


class _MaskRows(torch.autograd.Function):
    """Row masking whose every derivative zeroes inactive tokens.

    ``preserve_values=True`` is the gradient barrier: a zero-copy identity in
    the forward (the result aliases ``x``) that still masks cotangents and
    tangents.
    """

    generate_vmap_rule = True

    @staticmethod
    def forward(x: torch.Tensor, active_mask: torch.Tensor, preserve_values: bool) -> torch.Tensor:
        return x if preserve_values else _mask_inactive_rows(x, active_mask)

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(inputs[1])
        ctx.save_for_forward(inputs[1])

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (active_mask,) = ctx.saved_tensors
        return _MaskRows.apply(grad_output.contiguous(), active_mask, False), None, None

    @staticmethod
    def jvp(ctx, tangent: torch.Tensor, _mask_tangent, _preserve_tangent):
        (active_mask,) = ctx.saved_for_forward
        return _MaskRows.apply(tangent.contiguous(), active_mask, False)


def _use_row_kernel(x: torch.Tensor, active_mask: torch.Tensor) -> bool:
    """Gate the bandwidth-optimized path to layouts the row kernel supports.

    Compiled graphs keep the where-form so Inductor owns fusion and traced
    autograd sees ordinary aten semantics; non-contiguous or CPU inputs fall
    back to the where-form as well.
    """
    return (
        not torch.compiler.is_compiling()
        and x.is_cuda
        and x.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and x.is_contiguous()
        and active_mask.is_contiguous()
    )


def _validate_packed_mask(x: torch.Tensor, active_mask: torch.Tensor) -> None:
    """Validate one packed ``[1, T, ...]`` tensor against its token mask."""
    if x.ndim < 2 or x.shape[0] != 1:
        raise ValueError(f"x must have packed shape [1, T, ...], got {tuple(x.shape)}")
    if active_mask.shape != (x.shape[1],):
        raise ValueError(
            f"active_mask must have shape [{x.shape[1]}], got {tuple(active_mask.shape)}"
        )
    if active_mask.dtype != torch.bool or active_mask.device != x.device:
        raise ValueError(
            f"active_mask must be torch.bool on {x.device}, "
            f"got {active_mask.dtype} on {active_mask.device}"
        )


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
    _validate_packed_mask(x, active_mask)
    if _use_row_kernel(x, active_mask):
        return _MaskRows.apply(x, active_mask, False)
    mask = active_mask.reshape(1, -1, *((1,) * (x.ndim - 2)))
    return torch.where(mask, x, 0)


def mask_inactive_token_gradients(
    x: torch.Tensor,
    active_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Preserve values while blocking inactive automatic-differentiation paths.

    Place this barrier between a parameterized producer and a primitive that ignores
    inactive forward rows but leaves their input-gradient suffix undefined. Forward
    values pass through unchanged while cotangents and tangents are row-masked, so
    automatic-differentiation paths are zero on inactive tokens and subsequent
    derivatives inherit the same mask. On the eligible eager CUDA path the result
    aliases ``x`` storage; compiled and fallback paths materialize ``torch.where``.
    Passing ``None`` returns ``x`` unchanged.
    """
    if active_mask is None:
        return x
    _validate_packed_mask(x, active_mask)
    if _use_row_kernel(x, active_mask):
        return _MaskRows.apply(x, active_mask, True)
    mask = active_mask.reshape(1, -1, *((1,) * (x.ndim - 2)))
    return torch.where(mask, x, x.detach())


__all__ = [
    "active_token_mask",
    "mask_inactive_token_gradients",
    "mask_inactive_tokens",
]
