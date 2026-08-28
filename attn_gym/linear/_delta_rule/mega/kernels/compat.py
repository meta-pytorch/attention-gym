# SPDX-License-Identifier: BSD-3-Clause

"""Torch host utilities for the experimental CuTeDSL 4.7 KDA backend."""

from __future__ import annotations

from functools import cache

import torch


def data_ptr(buffer: torch.Tensor) -> int:
    """Return a Torch tensor's CUDA address."""
    return buffer.data_ptr()


def current_device() -> int:
    """Return the active CUDA device ordinal."""
    return torch.cuda.current_device()


def tensor_device_index(tensor: torch.Tensor) -> int:
    """Return a CUDA tensor's explicit device ordinal."""
    return current_device() if tensor.device.index is None else tensor.device.index


@cache
def get_device_properties(device: int):
    """Return cached CUDA properties for a device ordinal."""
    return torch.cuda.get_device_properties(device)


def multiprocessor_count(device: int) -> int:
    """Return the cached number of SMs on a CUDA device."""
    return get_device_properties(device).multi_processor_count


def checkpoint_capacity_bound(tokens: int, num_sequences: int, interval: int) -> int:
    """Return the graph-safe checkpoint-row bound for packed sequences."""
    nonempty = min(tokens, num_sequences)
    return 0 if nonempty == 0 else nonempty + (tokens - nonempty) // interval


def validate_tma_tensor(name: str, tensor: torch.Tensor, *, alignment: int = 16) -> None:
    """Reject layouts that cannot be represented by the current int32 TMA ABI."""
    if tensor.data_ptr() % alignment:
        raise ValueError(f"{name} data pointer must be {alignment}-byte aligned")
    if tensor.stride(-1) != 1:
        raise ValueError(f"{name} innermost stride must be one")
    element_bytes = tensor.element_size()
    for stride in tensor.stride()[:-1]:
        if stride < 0 or stride > 2**31 - 1:
            raise ValueError(f"{name} outer strides must fit nonnegative int32")
        if stride * element_bytes % 16:
            raise ValueError(f"{name} outer byte strides must be multiples of 16")
