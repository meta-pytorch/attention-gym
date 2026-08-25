"""Infrastructure shared by CuTeDSL attention backends."""

from ._key import function_cache_key
from .cache import jit_cache
from .tune import TunableKernel, benchmark_gpu, run_tunable, tune
from .utils import (
    TMA_ALIGNMENT_BYTES,
    compile_tvm_ffi,
    get_device_properties,
    make_fake_strided_tensor,
    tensor_supports_contiguous_dim,
    tensor_supports_tma,
)

__all__ = [
    "TMA_ALIGNMENT_BYTES",
    "TunableKernel",
    "benchmark_gpu",
    "compile_tvm_ffi",
    "function_cache_key",
    "get_device_properties",
    "jit_cache",
    "make_fake_strided_tensor",
    "run_tunable",
    "tensor_supports_contiguous_dim",
    "tensor_supports_tma",
    "tune",
]
