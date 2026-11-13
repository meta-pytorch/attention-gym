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

from attn_gym.linear._delta_rule.chunk_schedule import ScheduleRequest
from attn_gym.linear._delta_rule.validation import (
    resolve_chunk_backend,
    resolve_chunk_kernel_options,
    validate_delta_rule_inputs,
)
from attn_gym.linear.types import KERNEL_OPTION_SPEC

SUPPORTED_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class ResolvedKernelOptions(NamedTuple):
    """Validated ``chunk_kda`` backend selection, warpdrive switches, tuning request, and ragged
    launch schedule."""

    backend: Literal["auto", "fused", "warpdrive"]
    split_backward: bool
    split_forward: bool
    autotune: bool
    schedule: ScheduleRequest = ScheduleRequest.AUTO


def resolve_kernel_options(
    kernel_options: Mapping[str, object] | None, *, op_name: str = "chunk_kda"
) -> ResolvedKernelOptions:
    """Validate chunk backend options and resolve their defaults."""
    options = resolve_chunk_kernel_options(kernel_options, KERNEL_OPTION_SPEC, op_name=op_name)
    # warpdrive accepts every schedule and ignores it: its grids already stride over active work.
    schedule = ScheduleRequest(options.pop("schedule"))
    return ResolvedKernelOptions(**options, schedule=schedule)


def kernel_dtype(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.dtype:
    """Return the Q/K/V dtype the KDA chunk kernels compute in.

    Q/K/V that share FP16 or BF16 run as given; FP32 or mixed Q/K/V run in BF16.
    """
    if q.dtype in (torch.float16, torch.bfloat16) and k.dtype == v.dtype == q.dtype:
        return q.dtype
    return torch.bfloat16


def cast_to_kernel_dtypes(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Cast chunk operands to the dtypes the KDA chunk kernels compute in.

    Q/K/V follow ``kernel_dtype``; gate and beta run in FP32. Callers return the output in the
    original ``q.dtype``. Each cast is an autograd op, so gradients come back in the caller's
    dtypes.
    """
    dtype = kernel_dtype(q, k, v)
    return (*(tensor.to(dtype) for tensor in (q, k, v)), gate.float(), beta.float())


def resolve_kda_backend(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    fused_only: bool = False,
) -> Literal["fused", "warpdrive"]:
    """Resolve the backend of one KDA chunk call (NOTE [Default chunk backend]).

    ``"auto"`` keeps FP16 kernels on the fused backend, because warpdrive FP16 does not hold
    model-range gates yet.
    """
    fused_only = fused_only or kernel_dtype(q, k, v) == torch.float16
    return resolve_chunk_backend(backend, q, fused_only=fused_only)


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


__all__ = [
    "SUPPORTED_INPUT_DTYPES",
    "cast_to_kernel_dtypes",
    "kernel_dtype",
    "resolve_kda_backend",
    "resolve_kernel_options",
    "validate_kda_inputs",
]
