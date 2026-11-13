# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared mathematical and structural constants for KDA implementations."""

from attn_gym.linear._delta_rule.constants import (
    DEFAULT_CHUNK_SIZE,
    LN2,
    LOG2_E,
)

# The causal intra-chunk reference spans 15 steps. Keep the rebase exponent below
# FP32's overflow boundary; equality can round to exp2(128) and produce non-finite values.
GATE_SPAN_STEPS = 15
FP32_EXPONENT_BUDGET = 128.0
GATE_LOWER_BOUND_SAFETY_MARGIN = 1e-3
MAX_GATE_LOWER_BOUND_MAGNITUDE = (
    FP32_EXPONENT_BUDGET / (GATE_SPAN_STEPS * LOG2_E) - GATE_LOWER_BOUND_SAFETY_MARGIN
)
# Per-token gate magnitude the warpdrive BF16 kernels (the SM100-family default) support: past a
# uniform -5.25 their gradient error grows quickly, and from -126 ln 2 per aligned 16-token sum
# (a uniform -5.45) results stop being finite. Empirical, unlike the fused bound above; see the
# KDA gate contract in docs/linear.md.
WARPDRIVE_MAX_GATE_LOWER_BOUND_MAGNITUDE = 5.25


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "GATE_LOWER_BOUND_SAFETY_MARGIN",
    "GATE_SPAN_STEPS",
    "LN2",
    "LOG2_E",
    "MAX_GATE_LOWER_BOUND_MAGNITUDE",
    "WARPDRIVE_MAX_GATE_LOWER_BOUND_MAGNITUDE",
]
