"""Shared deterministic scalar-GDN test inputs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F

from .kda import cumulative_sequence_offsets

GatePattern = Literal[
    "model_range",
    "mild",
    "softplus",
    "uniform_negative_twenty",
    "isolated_negative_twenty",
    "periodic_negative_twenty",
    "near_zero",
    "model_softplus",
]


def make_gdn_test_inputs(
    lengths: int | Sequence[int],
    *,
    batch: int = 1,
    key_heads: int = 1,
    value_heads: int = 2,
    gate_pattern: GatePattern = "model_range",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 31,
    value_scale: float = 0.125,
    state_scale: float = 0.01,
    sigmoid_beta: bool = False,
    requires_grad: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    """Create deterministic dense or packed grouped-head GDN operands and FP32 state.

    An integer ``lengths`` creates a dense ``[batch, lengths, ...]`` input and returns no
    sequence offsets. A sequence creates a batch-one packed input and returns its offsets.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if isinstance(lengths, int):
        tokens = lengths
        state_batch = batch
        cu_seqlens = None
    else:
        if batch != 1:
            raise ValueError("packed GDN test inputs require batch=1")
        tokens = sum(lengths)
        state_batch = len(lengths)
        cu_seqlens = cumulative_sequence_offsets(lengths)

    dim = 128
    q_shape = (batch, tokens, key_heads, dim)
    value_shape = (batch, tokens, value_heads, dim)
    gate_shape = value_shape[:-1]
    q = F.normalize(torch.randn(q_shape, device="cuda", generator=generator), dim=-1).to(dtype)
    k = F.normalize(torch.randn(q_shape, device="cuda", generator=generator), dim=-1).to(dtype)
    value = torch.randn(value_shape, device="cuda", dtype=dtype, generator=generator).mul_(
        value_scale
    )

    match gate_pattern:
        case "model_range":
            gate = (
                torch.empty(gate_shape, device="cuda")
                .uniform_(math.exp(-5.0), 1.0, generator=generator)
                .log_()
            )
        case "mild":
            gate = (
                torch.empty(gate_shape, device="cuda")
                .uniform_(0.5, 1.0, generator=generator)
                .log_()
            )
        case "softplus":
            gate = -F.softplus(torch.randn(gate_shape, device="cuda", generator=generator))
        case "uniform_negative_twenty":
            gate = torch.full(gate_shape, -20.0, device="cuda")
        case "isolated_negative_twenty":
            gate = (
                torch.empty(gate_shape, device="cuda")
                .uniform_(0.5, 1.0, generator=generator)
                .log_()
            )
            gate[:, tokens // 2] = -20.0
        case "periodic_negative_twenty":
            gate = torch.full(gate_shape, -0.1, device="cuda")
            gate[:, 7::16] = -20.0
        case "near_zero":
            gate = torch.empty(gate_shape, device="cuda").uniform_(-1e-4, 0.0, generator=generator)
        case "model_softplus":
            raw_gate = torch.randn(gate_shape, device="cuda", generator=generator)
            a_log = torch.linspace(-0.5, 0.5, value_heads, device="cuda")
            dt_bias = torch.linspace(-0.25, 0.25, value_heads, device="cuda")
            gate = -a_log.exp().view(1, 1, -1) * F.softplus(raw_gate + dt_bias.view(1, 1, -1))
        case _:
            raise ValueError(f"Unsupported gate pattern: {gate_pattern}")

    beta = (
        torch.randn(gate_shape, device="cuda", generator=generator).sigmoid_()
        if sigmoid_beta
        else torch.rand(gate_shape, device="cuda", generator=generator)
    )
    initial_state = torch.randn(
        state_batch,
        value_heads,
        dim,
        dim,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).mul_(state_scale)
    tensors = (q, k, value, gate, beta, initial_state)
    return (
        *(tensor.requires_grad_(requires_grad) for tensor in tensors),
        cu_seqlens,
    )


__all__ = ["make_gdn_test_inputs"]
