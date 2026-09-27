# SPDX-License-Identifier: BSD-3-Clause

"""Torch launcher for the scalar-GDN cuDNN backward (vendored cudnn-frontend v1.30 kernels)."""

from __future__ import annotations

import torch

from attn_gym._backends.cute.utils import initialized_cuda_device
from attn_gym.linear._delta_rule.cudnn_fe.gdn import gdn_backward
from attn_gym.linear._delta_rule.validation import resolve_scale


def chunk_gdn_bwd_cudnn_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    d_final_state: torch.Tensor | None = None,
    *,
    scale: float | None = None,
    split: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Run checkpoint recompute followed by scalar-GDN backward.

    Long sequences on few (sequence, head) tiles run as an exact piece chain. ``split`` enables
    the approximate forgetting-horizon work table instead and requires a no-state call.
    """
    if split and (initial_state is not None or d_final_state is not None):
        raise ValueError("the split backward schedule requires a no-state call")
    scale = resolve_scale(scale, q.shape[-1])
    if d_output.dtype != q.dtype:
        raise TypeError(f"d_output must use q.dtype ({q.dtype}), got {d_output.dtype}")
    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    with initialized_cuda_device(q):
        key_heads, heads = q.shape[2], value.shape[2]
        if k.shape != q.shape or heads % key_heads:
            raise ValueError("k must match q and value heads must be divisible by query heads")
        gradients = gdn_backward(
            q[0],
            k[0],
            value[0],
            gate[0],
            beta[0],
            d_output[0],
            cu_seqlens,
            scale=scale,
            initial_state=initial_state,
            d_final_state=d_final_state,
            split=split,
        )
        return (*(gradient.unsqueeze(0) for gradient in gradients[:5]), gradients[5])


__all__ = ["chunk_gdn_bwd_cudnn_packed"]
