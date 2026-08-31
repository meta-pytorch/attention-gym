# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by Attention Gym in 2026: dtype validation now uses exact supported names.

"""Host-side helpers shared by the FROST LA kernel modules (engine-invoked)."""

import cutlass

from .thd import TENSOR_MAP_QWORDS


_DTYPES = {
    "bfloat16": cutlass.BFloat16,
    "float16": cutlass.Float16,
    "half": cutlass.Float16,
    "float32": cutlass.Float32,
}


def get_dtype(dtype):
    """Map an exact Torch dtype name or supported alias to its CuTeDSL type."""
    name = str(dtype).removeprefix("torch.")
    try:
        return _DTYPES[name]
    except KeyError:
        raise ValueError(
            f"Unsupported dtype {dtype}, expected bfloat16, float16, half, or float32"
        ) from None


def tensormap_workspace_bytes(mod, B: int) -> int:
    """Runtime TMA-descriptor block for a kernel module: per-batch arrays +
    static slots + 128 alignment slack."""
    return TENSOR_MAP_QWORDS * 8 * (mod.TENSORMAP_DESC_ARRAYS * B + mod.TENSORMAP_STATIC_SLOTS) + 128
