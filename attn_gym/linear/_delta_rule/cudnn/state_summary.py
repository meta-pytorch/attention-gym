# SPDX-License-Identifier: BSD-3-Clause

"""Validated cuDNN KDA affine summaries for whole packed sequences.

Forward maps pack [B; A] for ``exit = entry @ A + B``; reverse maps pack [C; R] for
``d_entry = d_exit @ R + C``. Native state/operand rounding makes these approximate affine
maps, not bitwise reconstruction of an ordinary unsharded cuDNN run. The v1.30 summary kernels
run every sequence uncut, so the arithmetic is independent of sequence ownership and packed
token count (:mod:`attn_gym.linear._delta_rule.cudnn_fe.summary`).
"""

from __future__ import annotations

import torch

from ..cudnn_fe import summary
from .forward import validate_available


def build_cudnn_state_summaries(
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    bounds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return FP32 ``[N, H, V + K, K]`` maps packed as ``[B; A]`` for subsequences.

    Inputs use staged ``[1, T, H, D]`` layout, FP32 gate/beta, and normalized keys. Optional
    contiguous int32 ``bounds[R, 2]`` on k.device select, reorder, or duplicate consecutive
    cu_seqlens pairs. Empty intervals yield ``[0; I]``; zero rows yield no maps. Bounds values
    stay on device and may change during CUDA Graph replay.
    """
    validate_available(k)
    return summary.build_state_summaries(k, value, gate, beta, cu_seqlens, bounds=bounds)


def build_cudnn_state_grad_summaries(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float,
    *,
    transpose_forward_transition: bool = False,
    bounds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return subsequence FP32 ``[N, H, V + K, K]`` maps packed as ``[C; R]``.

    C is the native entry cotangent with zero exit cotangent. R uses an identity exit
    cotangent and zero d_output. ``transpose_forward_transition=True`` instead uses A.T
    from the forward summary: algebraically the adjoint, but not native backward rounding.
    Inputs and bounds follow :func:`build_cudnn_state_summaries`.
    """
    validate_available(q)
    return summary.build_state_grad_summaries(
        q,
        k,
        value,
        gate,
        beta,
        d_output,
        cu_seqlens,
        scale,
        transpose_forward_transition=transpose_forward_transition,
        bounds=bounds,
    )
