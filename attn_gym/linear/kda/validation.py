# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared validation for the public KDA delta-rule contract.

Both ops and both implementations share one tensor/packed contract, stated
here once; implementation-specific constraints stay with the implementation.
"""

from __future__ import annotations

import torch

SUPPORTED_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def validate_kda_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    *,
    op_name: str,
    gate_name: str,
    allow_grouped_heads: bool = False,
) -> None:
    """Validate the shared KDA operation contract before normalizing inputs.

    ``allow_grouped_heads`` admits value-head counts that are positive multiples of the
    q/k head count (the recurrent forms map heads in-kernel); the chunk pipeline requires
    equal head counts. This is the multi-value attention (MVA) pattern of "Transformers
    are SSMs" (arXiv:2405.21060, section 7.2): only q/k are shared across a group, while
    the gate and beta drive each value head's state update and keep one entry per value
    head.
    """
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, H, K], got {tuple(q.shape)}")
    batch, tokens, key_heads, key_dim = q.shape
    if batch == 0 or tokens == 0 or key_heads == 0 or key_dim == 0:
        raise ValueError(f"q must have nonempty dimensions, got {tuple(q.shape)}")
    if k.shape != q.shape:
        raise ValueError(f"k must have shape {tuple(q.shape)}, got {tuple(k.shape)}")
    if v.ndim != 4 or v.shape[:2] != (batch, tokens) or v.shape[-1] < 1:
        raise ValueError(f"v must have shape [{batch}, {tokens}, H, V], got {tuple(v.shape)}")
    heads = v.shape[2]
    if heads != key_heads and not (allow_grouped_heads and heads != 0 and heads % key_heads == 0):
        message = (
            f"v heads must be a positive multiple of q heads for {op_name}, "
            if allow_grouped_heads
            else f"v heads must match q heads for {op_name}, "
        )
        raise ValueError(message + f"got {heads} value heads for {key_heads} query heads")
    if gate.shape != (batch, tokens, heads, key_dim):
        raise ValueError(
            f"{gate_name} must have shape {(batch, tokens, heads, key_dim)}, "
            f"got {tuple(gate.shape)}"
        )
    if beta.shape != (batch, tokens, heads):
        raise ValueError(f"beta must have shape {(batch, tokens, heads)}, got {tuple(beta.shape)}")
    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("packed cu_seqlens require q to have batch size one")
        if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
            raise ValueError("cu_seqlens must have shape [num_sequences + 1]")
        if (
            cu_seqlens.dtype != torch.int32
            or not cu_seqlens.is_contiguous()
            or cu_seqlens.device != q.device
        ):
            raise ValueError("cu_seqlens must be contiguous int32 on q.device")
    state_batch = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    expected_state = (state_batch, heads, key_dim, v.shape[-1])
    if initial_state is not None and initial_state.shape != expected_state:
        raise ValueError(
            f"initial_state must have shape {expected_state}, got {tuple(initial_state.shape)}"
        )
    data_tensors = (q, k, v, gate, beta)
    if initial_state is not None:
        data_tensors += (initial_state,)
    if not all(tensor.device == q.device for tensor in data_tensors):
        raise ValueError(f"all {op_name} inputs must be on the same device")
    if any(tensor.dtype not in SUPPORTED_INPUT_DTYPES for tensor in data_tensors):
        supported = ", ".join(str(dtype) for dtype in SUPPORTED_INPUT_DTYPES)
        raise TypeError(f"{op_name} inputs must use one of {supported}")


__all__ = ["SUPPORTED_INPUT_DTYPES", "validate_kda_inputs"]
