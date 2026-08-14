"""Shared KDA test helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def cumulative_sequence_offsets(
    lengths: Sequence[int],
    *,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Build an int32 packed-sequence boundary tensor from token lengths."""
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return torch.tensor(offsets, device=device, dtype=torch.int32)


__all__ = ["cumulative_sequence_offsets"]
