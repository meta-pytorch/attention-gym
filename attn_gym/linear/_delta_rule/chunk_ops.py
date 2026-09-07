"""Torch-only contracts for shared delta-rule chunk routing and gate scans.

Register before graph capture; import Triton launchers only during CUDA dispatch.
The historical KDA operator names are retained so this ownership move does not
change graph targets or register a second operator boundary.
"""

from __future__ import annotations

import importlib

import torch

torch.library.define(
    "attn_gym::_kda_plain_gate_scan",
    "(Tensor values, Tensor? cu_seqlens, Tensor? chunk_offsets, bool reverse) -> Tensor",
)
torch.library.define(
    "attn_gym::kda_prepare_chunk_offsets",
    "(Tensor cu_seqlens, SymInt tokens, int chunk_size) -> Tensor",
)


def _plain_gate_backend():
    """Load the optional shared gate-scan launcher."""
    try:
        return importlib.import_module("attn_gym.linear._delta_rule.triton.plain_gate")
    except ImportError as error:
        raise ImportError(
            "fused delta-rule gate scans require CUDA with Triton support"
        ) from error


def _plain_gate_scan_cuda(*args):
    """Dispatch a gate scan without importing Triton during registration."""
    return _plain_gate_backend()._plain_gate_scan_cuda(*args)


def _prepare_chunk_offsets_cuda(*args):
    """Build packed offsets through the lazily loaded shared scheduler."""
    from attn_gym.linear._delta_rule.triton.chunk_scheduler import _prepare_ragged_chunk_offsets

    return _prepare_ragged_chunk_offsets(*args)


torch.library.impl("attn_gym::_kda_plain_gate_scan", "CUDA", _plain_gate_scan_cuda)
torch.library.impl(
    "attn_gym::kda_prepare_chunk_offsets",
    "CUDA",
    _prepare_chunk_offsets_cuda,
)


@torch.library.register_fake("attn_gym::_kda_plain_gate_scan")
def _plain_gate_scan_fake(
    values: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    reverse: bool,
) -> torch.Tensor:
    """Describe the compact internal gate scan output."""
    del cu_seqlens, chunk_offsets, reverse
    return torch.empty_like(values, memory_format=torch.contiguous_format)


@torch.library.register_fake("attn_gym::kda_prepare_chunk_offsets")
def _prepare_chunk_offsets_fake(
    cu_seqlens: torch.Tensor,
    tokens: int,
    chunk_size: int,
) -> torch.Tensor:
    """Preserve the sequence-boundary tensor metadata without reading values."""
    return torch.empty_like(cu_seqlens)


_plain_gate_scan_op = torch.ops.attn_gym._kda_plain_gate_scan.default
prepare_chunk_offsets_op = torch.ops.attn_gym.kda_prepare_chunk_offsets.default
