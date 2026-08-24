# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared mathematical and structural constants for KDA implementations."""

import math

LN2 = math.log(2.0)
LOG2_E = math.log2(math.e)
DEFAULT_CHUNK_SIZE = 64

# The causal intra-chunk reference spans 15 steps. Keep the rebase exponent below
# FP32's overflow boundary; equality can round to exp2(128) and produce non-finite values.
GATE_SPAN_STEPS = 15
FP32_EXPONENT_BUDGET = 128.0
GATE_LOWER_BOUND_SAFETY_MARGIN = 1e-3
MAX_GATE_LOWER_BOUND_MAGNITUDE = (
    FP32_EXPONENT_BUDGET / (GATE_SPAN_STEPS * LOG2_E) - GATE_LOWER_BOUND_SAFETY_MARGIN
)


__all__ = [
    "DEFAULT_CHUNK_SIZE",
    "GATE_LOWER_BOUND_SAFETY_MARGIN",
    "GATE_SPAN_STEPS",
    "LN2",
    "LOG2_E",
    "MAX_GATE_LOWER_BOUND_MAGNITUDE",
]
