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
from attn_gym.linear._delta_rule.constants import DEFAULT_CHUNK_SIZE as _CHUNK_SIZE
from attn_gym.linear._delta_rule.validation import (
    resolve_chunk_kernel_options,
    validate_delta_rule_inputs,
)
from attn_gym.linear.types import (
    KERNEL_OPTION_SPEC,
    STAGE_OPTION_SPEC,
    IntraChunkDtypes,
    ReplayState,
)

SUPPORTED_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# ``intra_chunk_dtypes`` when the option is omitted, and the one kernel set of FP16 Q/K/V.
DEFAULT_INTRA_CHUNK_DTYPES: IntraChunkDtypes = "tf32+bf16"
FP16_INTRA_CHUNK_DTYPES: IntraChunkDtypes = "tf32+fp16"


class ResolvedKernelOptions(NamedTuple):
    """Validated ``chunk_kda`` backend selection, warpdrive switches, tuning request, and ragged
    launch schedule."""

    backend: Literal["auto", "fused", "warpdrive"]
    split_backward: bool
    split_forward: bool
    autotune: bool
    schedule: ScheduleRequest
    # ``None`` when omitted; ``resolve_intra_chunk_dtypes`` picks the default by input dtype.
    intra_chunk_dtypes: IntraChunkDtypes | None
    batch_invariant: bool = False  # staged handles only


def resolve_kernel_options(
    kernel_options: Mapping[str, object] | None,
    *,
    op_name: str = "chunk_kda",
    staged: bool = False,
    tape_backend: str | None = None,
) -> ResolvedKernelOptions:
    """Validate chunk backend options and resolve their defaults; ``staged`` also accepts the
    staged handles' ``batch_invariant`` (:class:`attn_gym.linear.types.StageKernelOptions`).
    ``tape_backend`` resolves a staged backward's options against its saved tape's backend
    (:func:`attn_gym.linear._delta_rule.validation.resolve_chunk_kernel_options`)."""
    spec = STAGE_OPTION_SPEC if staged else KERNEL_OPTION_SPEC
    options = resolve_chunk_kernel_options(
        kernel_options, spec, op_name=op_name, tape_backend=tape_backend
    )
    # warpdrive accepts every schedule and ignores it: its grids already stride over active work.
    options["schedule"] = ScheduleRequest(options["schedule"])
    return ResolvedKernelOptions(**options)


def resolve_intra_chunk_dtypes(
    intra_chunk_dtypes: IntraChunkDtypes | None, compute_dtype: torch.dtype, *, op_name: str
) -> IntraChunkDtypes:
    """Choose intra-chunk operand dtypes from the Q/K/V kernel dtype, not the caller's dtype.

    Replay uses BF16 even for FP16 inputs. Validate before dispatch so invalid combinations
    fail on every backend, including during ``torch.compile`` tracing.
    """
    if compute_dtype == torch.float16:
        if intra_chunk_dtypes not in (None, FP16_INTRA_CHUNK_DTYPES):
            raise ValueError(
                f"{op_name}: FP16 q/k/v take only "
                f"kernel_options['intra_chunk_dtypes']={FP16_INTRA_CHUNK_DTYPES!r}, got "
                f"{intra_chunk_dtypes!r}: their decay products need TF32 (the in-chunk decay "
                "factors 2^-g overflow FP16) and their solve operands are FP16"
            )
        return FP16_INTRA_CHUNK_DTYPES
    if intra_chunk_dtypes == FP16_INTRA_CHUNK_DTYPES:
        raise ValueError(
            f"{op_name}: kernel_options['intra_chunk_dtypes']={FP16_INTRA_CHUNK_DTYPES!r} is the "
            "FP16 q/k/v kernel set; BF16 (or FP32 or mixed) q/k/v take 'tf32+bf16' (default), "
            "'tf32' or 'bf16'"
        )
    return DEFAULT_INTRA_CHUNK_DTYPES if intra_chunk_dtypes is None else intra_chunk_dtypes


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


def validate_replay_state(
    replay_state: ReplayState, q: torch.Tensor, v: torch.Tensor, num_slots: int
) -> None:
    """Check replay cache shapes, dtypes and disjoint slot strides without reading device data."""
    layout = ReplayState.slot_layout(q.shape[2], q.shape[3], value_dim=v.shape[3])
    for name, tensor, (slot_shape, expected_dtype) in zip(
        ReplayState._fields[:-1], replay_state[:-1], layout[:-1], strict=True
    ):
        expected_shape = (num_slots, *slot_shape)
        if tensor.shape != expected_shape:
            raise ValueError(
                f"replay_state {name} tensor must have shape {expected_shape}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != expected_dtype or tensor.device != q.device:
            raise TypeError(
                f"replay_state {name} tensor must use {expected_dtype} on the input device"
            )
        expected_stride = 1
        dense_inner = True
        for size, stride in reversed(tuple(zip(tensor.shape[2:], tensor.stride()[2:]))):
            dense_inner &= size == 1 or stride == expected_stride
            expected_stride *= size
        if not dense_inner or tensor.stride(1) != expected_stride:
            raise ValueError(f"replay_state {name} tensor must have dense per-token rows")
        if num_slots > 1 and tensor.stride(0) < _CHUNK_SIZE * expected_stride:
            raise ValueError(f"replay_state {name} tensor slots must not overlap")
    count_shape, count_dtype = layout[-1]
    if replay_state.count.shape not in ((num_slots,), (num_slots, *count_shape)):
        raise ValueError(
            f"replay_state count tensor must have shape ({num_slots},) or "
            f"({num_slots}, 1), got {tuple(replay_state.count.shape)}"
        )
    if replay_state.count.dtype != count_dtype or replay_state.count.device != q.device:
        raise TypeError("replay_state count tensor must be int32 on the input device")
    if num_slots > 1 and replay_state.count.stride(0) == 0:
        raise ValueError("replay_state count tensor slots must not overlap")
    # Invariant: callers initialize counts to zero, and replay kernels keep them in [0, 63].


__all__ = [
    "DEFAULT_INTRA_CHUNK_DTYPES",
    "FP16_INTRA_CHUNK_DTYPES",
    "SUPPORTED_INPUT_DTYPES",
    "cast_to_kernel_dtypes",
    "kernel_dtype",
    "resolve_intra_chunk_dtypes",
    "resolve_kernel_options",
    "validate_kda_inputs",
    "validate_replay_state",
]
