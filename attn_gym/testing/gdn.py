"""Shared deterministic scalar-GDN test inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F

from .kda import cumulative_sequence_offsets

GatePattern = Literal[
    "mild",
    "uniform_negative_twenty",
    "isolated_negative_twenty",
    "near_zero",
    "model_softplus",
]


def make_gdn_test_inputs(
    lengths: Sequence[int],
    *,
    key_heads: int = 1,
    value_heads: int = 2,
    gate_pattern: GatePattern = "mild",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 31,
    requires_grad: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Create deterministic packed grouped-head GDN operands and FP32 state."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tokens, dim = sum(lengths), 128
    q_shape = (1, tokens, key_heads, dim)
    value_shape = (1, tokens, value_heads, dim)
    gate_shape = value_shape[:-1]
    q = F.normalize(torch.randn(q_shape, device="cuda", generator=generator), dim=-1).to(dtype)
    k = F.normalize(torch.randn(q_shape, device="cuda", generator=generator), dim=-1).to(dtype)
    value = torch.randn(value_shape, device="cuda", dtype=dtype, generator=generator).div_(8)

    match gate_pattern:
        case "mild":
            gate = (
                torch.empty(gate_shape, device="cuda")
                .uniform_(0.5, 1.0, generator=generator)
                .log_()
            )
        case "uniform_negative_twenty":
            gate = torch.full(gate_shape, -20.0, device="cuda")
        case "isolated_negative_twenty":
            gate = (
                torch.empty(gate_shape, device="cuda")
                .uniform_(0.5, 1.0, generator=generator)
                .log_()
            )
            gate[:, tokens // 2] = -20.0
        case "near_zero":
            gate = torch.empty(gate_shape, device="cuda").uniform_(-1e-4, 0.0, generator=generator)
        case "model_softplus":
            raw_gate = torch.randn(gate_shape, device="cuda", generator=generator)
            a_log = torch.linspace(-0.5, 0.5, value_heads, device="cuda")
            dt_bias = torch.linspace(-0.25, 0.25, value_heads, device="cuda")
            gate = -a_log.exp().view(1, 1, -1) * F.softplus(raw_gate + dt_bias.view(1, 1, -1))
        case _:
            raise ValueError(f"Unsupported gate pattern: {gate_pattern}")

    beta = torch.rand(gate_shape, device="cuda", generator=generator)
    initial_state = torch.randn(
        len(lengths),
        value_heads,
        dim,
        dim,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    ).div_(100)
    tensors = (q, k, value, gate, beta, initial_state)
    return (
        *(tensor.requires_grad_(requires_grad) for tensor in tensors),
        cumulative_sequence_offsets(lengths),
    )


__all__ = ["make_gdn_test_inputs"]
