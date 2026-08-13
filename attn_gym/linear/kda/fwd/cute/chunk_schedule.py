"""Compile-time scheduling modes shared by CuTe KDA chunk kernels."""

from __future__ import annotations

from enum import Enum


class ChunkSchedule(Enum):
    """Logical chunk-routing policy selected when compiling a kernel."""

    DENSE = "dense"
    ALIGNED = "aligned"
    RAGGED = "ragged"


__all__ = ["ChunkSchedule"]
