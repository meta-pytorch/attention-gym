"""Shared Mamba2 test helpers."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING

import torch

from attn_gym.linear import chunk_mamba2
from attn_gym.linear.mamba2.api import NO_DT_LIMIT, Core, reference_mamba2

if TYPE_CHECKING:
    import pytest


def make_mamba2_inputs(
    batch: int,
    tokens: int,
    *,
    heads: int = 4,
    groups: int = 2,
    head_dim: int = 64,
    state_dim: int = 64,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """CUDA activations in ``dtype`` with Mamba2-like step sizes and decays; parameters stay FP32.

    ``dt`` is raw (centered at -2, for ``dt_softplus=True`` with ``dt_bias``) and ``A`` lies in
    ``[-16, -1]``.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, device="cuda").to(dtype)

    return {
        "x": randn(batch, tokens, heads, head_dim, dtype=dtype),
        "dt": randn(batch, tokens, heads) - 2.0,
        "A": -torch.exp(torch.rand(heads, generator=generator, device="cuda") * 2.77),
        "B": randn(batch, tokens, groups, state_dim, dtype=dtype),
        "C": randn(batch, tokens, groups, state_dim, dtype=dtype),
        "D": randn(heads),
        "z": randn(batch, tokens, heads, head_dim, dtype=dtype),
        "dt_bias": randn(heads) * 0.5,
    }


def fp64_reference(
    inputs: dict[str, torch.Tensor], **kwargs
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``chunk_mamba2(impl="reference")`` on FP64 copies of the same rounded inputs."""
    upcast = {name: tensor.double() for name, tensor in inputs.items()}
    return chunk_mamba2(**upcast, impl="reference", **kwargs)


def force_segments(monkeypatch: pytest.MonkeyPatch, length: int) -> None:
    """Split every chunk scan into ``length``-token segments regardless of the program count."""

    def plan(num_programs, tokens, device):
        return math.ceil(tokens / length), length

    for module in ("chunk_fwd", "chunk_bwd"):
        monkeypatch.setattr(f"attn_gym.linear.mamba2.triton.{module}.segment_plan", plan)


def reference_with_token_decays(
    core: Core,
    leaves: dict[str, torch.Tensor],
    *,
    dt_softplus: bool = False,
    dt_limit: tuple[float, float] = NO_DT_LIMIT,
    cu_seqlens: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[tuple[torch.Tensor, torch.Tensor | None], torch.Tensor]:
    """Run ``reference_mamba2`` with one ``A`` leaf per token, returned with the outputs.

    ``core`` receives its sequence's ``[batch, T, H]`` slice of that leaf as ``A``. The leaf's
    gradient holds the per-token terms that ``dA`` sums, whose magnitudes bound a cancelling
    reduction's error.
    """
    A_tokens = leaves["A"].detach().expand(leaves["dt"].shape).clone().requires_grad_()
    offsets = [0, leaves["x"].shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    # Packed references call the core once per nonempty sequence, in order.
    spans = iter([(begin, end) for begin, end in pairwise(offsets) if begin < end])

    def token_decay_core(x, dt, A, B, C, **core_kwargs):
        begin, end = next(spans)
        return core(x, dt, A_tokens[:, begin:end], B, C, **core_kwargs)

    outputs = reference_mamba2(
        token_decay_core,
        *(leaves.get(name) for name in ("x", "dt", "A", "B", "C", "D", "z", "initial_state")),
        dt_bias=leaves.get("dt_bias"),
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        cu_seqlens=cu_seqlens,
        output_final_state=output_final_state,
    )
    return outputs, A_tokens
