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

from collections.abc import Mapping
from typing import Literal, NamedTuple

import torch

from attn_gym.linear._delta_rule.validation import validate_delta_rule_inputs

SUPPORTED_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_KERNEL_OPTION_NAMES = frozenset({"backend", "split_backward", "split_forward"})


class ResolvedKernelOptions(NamedTuple):
    """Validated ``chunk_kda`` backend selection and its Mega scheduling switches."""

    backend: Literal["fused", "mega"]
    split_backward: bool
    split_forward: bool


def resolve_kernel_options(
    kernel_options: Mapping[str, object] | None,
) -> ResolvedKernelOptions:
    """Validate chunk backend options and resolve their defaults."""
    if kernel_options is None:
        return ResolvedKernelOptions("fused", False, False)
    unknown = kernel_options.keys() - _KERNEL_OPTION_NAMES
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unsupported chunk_kda kernel options: {names}")
    backend = kernel_options.get("backend", "fused")
    if backend not in ("fused", "mega"):
        raise ValueError("kernel_options['backend'] must be 'fused' or 'mega'")
    splits = {}
    for name in ("split_backward", "split_forward"):
        value = kernel_options.get(name, False)
        if not isinstance(value, bool):
            raise TypeError(f"kernel_options['{name}'] must be a bool")
        if value and backend != "mega":
            raise ValueError(f"{name} requires kernel_options['backend']='mega'")
        splits[name] = value
    return ResolvedKernelOptions(backend, splits["split_backward"], splits["split_forward"])


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
    validate_delta_rule_inputs(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        op_name=op_name,
        gate_name=gate_name,
        vector_gate=True,
        allow_grouped_heads=allow_grouped_heads,
    )
    data_tensors = (q, k, v, gate, beta) + (() if initial_state is None else (initial_state,))
    if any(tensor.dtype not in SUPPORTED_INPUT_DTYPES for tensor in data_tensors):
        supported = ", ".join(str(dtype) for dtype in SUPPORTED_INPUT_DTYPES)
        raise TypeError(f"{op_name} inputs must use one of {supported}")


__all__ = ["SUPPORTED_INPUT_DTYPES", "resolve_kernel_options", "validate_kda_inputs"]
