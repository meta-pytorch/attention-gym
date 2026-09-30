# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Validation of packed ``cu_seqlens`` offsets shared by the linear and sparse operations."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_cu_seqlens(batch: int, device: torch.device, **offsets: object) -> None:
    """Check batch-one packing and that each named offsets tensor is ``[N + 1]`` int32."""
    if batch != 1:
        raise ValueError("packed cu_seqlens require q to have batch size one")
    for name, tensor in offsets.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 1 or tensor.shape[0] < 2:
            raise ValueError(f"{name} must have shape [num_sequences + 1]")
        if tensor.dtype != torch.int32 or not tensor.is_contiguous() or tensor.device != device:
            raise ValueError(f"{name} must be contiguous int32 on q.device")


def validate_cu_seqlens_pair(
    cu_seqlens: Tensor | None,
    cu_seqlens_k: Tensor | None,
    *,
    batch: int,
    device: torch.device,
) -> None:
    """Validate optional query/key packed offsets that must be supplied together."""
    if (cu_seqlens is None) != (cu_seqlens_k is None):
        raise ValueError("cu_seqlens and cu_seqlens_k must be supplied together")
    if cu_seqlens is None:
        return
    validate_cu_seqlens(batch, device, cu_seqlens=cu_seqlens, cu_seqlens_k=cu_seqlens_k)
    if cu_seqlens.shape != cu_seqlens_k.shape:
        raise ValueError("cu_seqlens and cu_seqlens_k must describe the same number of sequences")
