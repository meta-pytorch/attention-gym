"""Persistent compile cache and TVM-FFI compile helpers for CuTeDSL kernels.

Importing this package does not import cutlass, so torch-only code can use it.
"""

from ._key import canonical_key, function_cache_key, module_source_digest
from .compile import compile_many, precompile_many
from .jit import (
    CacheInfo,
    CacheLockError,
    cache_enabled,
    get_cache_path,
    jit_cache,
    set_cache_path,
)
from .target import (
    CompileTarget,
    detect_compile_target,
    get_compile_target,
    set_compile_target,
)
from .tvm_ffi import (
    compile_tvm_ffi,
    make_fake_compact_tensor,
    make_fake_strided_tensor,
)

__all__ = [
    "CacheInfo",
    "CacheLockError",
    "CompileTarget",
    "cache_enabled",
    "canonical_key",
    "compile_many",
    "compile_tvm_ffi",
    "detect_compile_target",
    "function_cache_key",
    "get_cache_path",
    "get_compile_target",
    "jit_cache",
    "make_fake_compact_tensor",
    "make_fake_strided_tensor",
    "module_source_digest",
    "precompile_many",
    "set_cache_path",
    "set_compile_target",
]
