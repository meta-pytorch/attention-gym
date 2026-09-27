# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by Attention Gym in 2026: vendored from cudnn-frontend v1.30.0; imports relocated into
# attn_gym.linear._delta_rule.cudnn_fe. get_dtype matches exact dtype names.

"""Host-side helpers shared by the vendored GDN and KDA kernel modules."""

import cutlass
import torch

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


def validate_cuda_tensors(reference: torch.Tensor, **tensors: torch.Tensor | None) -> None:
    """Reject cross-device or inactive-device launches before selecting a compiled ABI."""
    if not reference.is_cuda:
        raise ValueError("common cuDNN helpers require CUDA tensors")
    if torch.cuda.current_device() != reference.get_device():
        raise ValueError("the active CUDA device must match the input device")
    for name, tensor in tensors.items():
        if tensor is not None and tensor.device != reference.device:
            raise ValueError(f"{name} must be on {reference.device}")


def tensormap_workspace_bytes(mod, B: int) -> int:
    """Runtime TMA-descriptor block for a kernel module, its per-batch arrays plus 128 alignment
    slack."""
    return TENSOR_MAP_QWORDS * 8 * mod.TENSORMAP_DESC_ARRAYS * B + 128
