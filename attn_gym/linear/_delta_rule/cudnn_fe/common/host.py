# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe.

"""Host-side helpers shared by the FROST LA kernel modules (engine-invoked)."""

import cutlass

from .thd import TENSOR_MAP_QWORDS


def get_dtype(dtype):
    """dtype string -> cutlass DSL type (bf16/fp16 io, fp32/bf16 states)."""
    name = str(dtype)
    if "bfloat16" in name:
        return cutlass.BFloat16
    if "float16" in name or "half" in name:
        return cutlass.Float16
    if "float32" in name:
        return cutlass.Float32
    raise ValueError(f"Unsupported dtype {dtype}, expected bfloat16, float16, or float32")


def tensormap_workspace_bytes(mod, B: int) -> int:
    """Runtime TMA-descriptor block for a kernel module, its per-batch arrays plus 128 alignment slack."""
    return TENSOR_MAP_QWORDS * 8 * mod.TENSORMAP_DESC_ARRAYS * B + 128
