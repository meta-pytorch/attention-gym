# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi-style per-token natural-log gate transform."""

from __future__ import annotations

import torch

from attn_gym.linear._delta_rule.gate import gate_transform
from attn_gym.linear.types import GateTransform, Impl


def bound_gate(
    raw_gate: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    lower_bound: float = -5.0,
    fastmath: bool = False,
    impl: Impl | str = Impl.FUSED,
) -> torch.Tensor:
    """Map projection outputs to bounded per-channel natural-log decays.

    Equivalent to ``gate_transform(..., kind="bounded")`` restricted to per-channel gates.
    The transform is ``lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))``.
    Arithmetic and output use FP32; a fused raw-gate gradient matches the input dtype.

    Args:
        raw_gate: Floating-point projection output shaped ``[B, T, H, D]``.
        A_log: FP32 per-head log scale shaped ``[H]``.
        dt_bias: FP32 per-channel bias shaped ``[H, D]``.
        lower_bound: Finite nonpositive gate floor.
        fastmath: Use approximate fused exponentials; rejected by the reference path.
        impl: ``"reference"`` uses ordinary PyTorch. ``"fused"`` uses private CuTeDSL
            kernels and requires CUDA capability 9.0 or newer, ``D=128``, and FP16, BF16,
            or FP32 logits.
    """
    if raw_gate.ndim != 4:
        raise ValueError(f"raw_gate must have shape [B, T, H, D], got {tuple(raw_gate.shape)}")
    return gate_transform(
        raw_gate,
        A_log,
        dt_bias,
        kind=GateTransform.BOUNDED,
        lower_bound=lower_bound,
        fastmath=fastmath,
        impl=impl,
    )
