# SPDX-License-Identifier: BSD-3-Clause

"""Operand normalization at the boundary of the registered warpdrive operators.

The warpdrive launchers reject operands their kernels cannot address. The registered GDN and KDA
operators pass every operand through these helpers (or ``normalize_tma_tensor`` for TMA
operands), which copy only an operand that violates that ABI, so aligned inputs stay zero-copy.
This module imports no CuTeDSL; the launchers stay lazily loaded.
"""

from __future__ import annotations

import torch
from torch import Tensor

from attn_gym._backends.cute import tensor_supports_contiguous_dim


def normalize_scalar_rows(tensor: Tensor) -> Tensor:
    """Copy unless the innermost mode is contiguous with element-aligned slice origins."""
    if tensor_supports_contiguous_dim(tensor, alignment_bytes=tensor.element_size()):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def normalize_cu_seqlens(cu_seqlens: Tensor) -> Tensor:
    """Copy packed offsets unless they are contiguous and 8-byte aligned."""
    if cu_seqlens.is_contiguous() and cu_seqlens.data_ptr() % 8 == 0:
        return cu_seqlens
    return cu_seqlens.clone(memory_format=torch.contiguous_format)


__all__ = ["normalize_cu_seqlens", "normalize_scalar_rows"]
