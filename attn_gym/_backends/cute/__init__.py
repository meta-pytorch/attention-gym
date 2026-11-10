"""Infrastructure shared by CuTeDSL attention backends."""

from .tune import TunableKernel, benchmark_gpu, run_tunable, tune
from .utils import (
    TMA_ALIGNMENT_BYTES,
    get_device_properties,
    normalize_compact_tensor,
    normalize_tma_rows,
    normalize_tma_tensor,
    tensor_supports_contiguous_dim,
    tensor_supports_tma,
    tensor_supports_tma_rows,
)

__all__ = [
    "TMA_ALIGNMENT_BYTES",
    "TunableKernel",
    "benchmark_gpu",
    "get_device_properties",
    "normalize_compact_tensor",
    "normalize_tma_rows",
    "normalize_tma_tensor",
    "run_tunable",
    "tensor_supports_contiguous_dim",
    "tensor_supports_tma",
    "tensor_supports_tma_rows",
    "tune",
]
