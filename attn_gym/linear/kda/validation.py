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
    resolve_chunk_kernel_options,
    validate_delta_rule_inputs,
)
from attn_gym.linear.types import KERNEL_OPTION_SPEC

SUPPORTED_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


class ResolvedKernelOptions(NamedTuple):
    """Validated ``chunk_kda`` backend selection, warpdrive switches, tuning request, and ragged
    launch schedule."""

    backend: Literal["fused", "warpdrive"]
    split_backward: bool
    split_forward: bool
    autotune: bool
    schedule: ScheduleRequest = ScheduleRequest.AUTO


def resolve_kernel_options(
    kernel_options: Mapping[str, object] | None, *, op_name: str = "chunk_kda"
) -> ResolvedKernelOptions:
    """Validate chunk backend options and resolve their defaults."""
    options = resolve_chunk_kernel_options(kernel_options, KERNEL_OPTION_SPEC, op_name=op_name)
    schedule = ScheduleRequest(options.pop("schedule"))
    if schedule is not ScheduleRequest.AUTO and options["backend"] != "fused":
        raise ValueError("schedule requires kernel_options['backend']='fused'")
    return ResolvedKernelOptions(**options, schedule=schedule)


def cast_to_kernel_dtypes(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor, beta: torch.Tensor
) -> tuple[torch.Tensor, ...]:
    """Cast chunk operands to the dtypes the KDA chunk kernels compute in.

    Q/K/V that share FP16 or BF16 run as given; FP32 or mixed Q/K/V run in BF16. Gate and beta
    run in FP32. Callers return the output in the original ``q.dtype``. Each cast is an autograd
    op, so gradients come back in the caller's dtypes.
    """
    kernel_dtype = (
        q.dtype
        if q.dtype in (torch.float16, torch.bfloat16) and k.dtype == v.dtype == q.dtype
        else torch.bfloat16
    )
    return (
        *(tensor.to(kernel_dtype) for tensor in (q, k, v)),
        gate.float(),
        beta.float(),
    )


def validate_warpdrive_fastmath(fastmath: bool) -> None:
    """Reject a precision request that the native warpdrive KDA kernels cannot honor."""
    if not fastmath:
        raise ValueError(
            "warpdrive KDA cannot honor fastmath=False; "
            "use fastmath=True or the repo-local fused backend"
        )


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
    "resolve_kernel_options",
    "validate_kda_inputs",
    "validate_warpdrive_fastmath",
]
