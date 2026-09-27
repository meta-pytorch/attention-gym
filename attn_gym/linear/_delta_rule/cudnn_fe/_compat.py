# SPDX-License-Identifier: BSD-3-Clause
"""Torch-backed replacements for the cudnn.frost host utilities the vendored kernels reach.

Surface required by the v1.30 kernel/common closure (see import_closure.py):
cudnn.frost.buffers.{DeviceView, probe}, cudnn.frost.device.{current_device, multiprocessor_count}.
"""

from __future__ import annotations

import torch

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
}


def DeviceView(ptr: int, shape, dtype: str, device_id: int) -> torch.Tensor:
    """Compile-time placeholder tensor (upstream wraps a raw pointer; only shape/dtype matter)."""
    del ptr
    return torch.empty(
        tuple(int(s) for s in shape), dtype=_DTYPES[dtype], device=f"cuda:{device_id}"
    )


def probe(buf: torch.Tensor):
    """(ptr, shape, strides_in_elements, dtype_name, device_id), matching cudnn.frost.buffers.probe."""
    name = {v: k for k, v in _DTYPES.items()}[buf.dtype]
    return buf.data_ptr(), tuple(buf.shape), tuple(buf.stride()), name, buf.device.index


def current_device() -> int:
    return torch.cuda.current_device()


def multiprocessor_count(device: int) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count
