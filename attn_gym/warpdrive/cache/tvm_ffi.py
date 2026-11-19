"""Compile CuTeDSL entry points against fake tensors with the TVM-FFI stream ABI."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

import torch

from .jit import EXPORT_FUNCTION_NAME, _load_compiled

_VALID_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


def _contains_torch_tensor(value: Any) -> bool:
    match value:
        case torch.Tensor():
            return True
        case tuple() | list():
            return any(_contains_torch_tensor(item) for item in value)
        case dict():
            return any(
                _contains_torch_tensor(key) or _contains_torch_tensor(item)
                for key, item in value.items()
            )
        case _:
            return False


def make_fake_strided_tensor(
    dtype: Any,
    shape: tuple[Any, ...],
    *,
    contiguous_dim: int | None = -1,
    stride_divisibility: int = 1,
    assumed_align: int | None = None,
    use_int64_strides: bool = True,
) -> Any:
    """Create a fake tensor with dynamic strides and an optional contiguous mode.

    ``contiguous_dim=None`` leaves every stride dynamic, for ordinary scalar loads.
    ``stride_divisibility`` is in elements. An omitted ``assumed_align`` is derived from it and
    the element width: the weakest alignment the dynamic strides promise.
    """
    if not shape:
        raise ValueError("make_fake_strided_tensor requires at least one dimension")
    if contiguous_dim is not None and not -len(shape) <= contiguous_dim < len(shape):
        raise ValueError(f"contiguous_dim is out of range for rank {len(shape)}")
    if not 1 <= stride_divisibility:
        raise ValueError("stride_divisibility must be positive")
    if assumed_align is not None and assumed_align < 1:
        raise ValueError("assumed_align must be positive")
    if contiguous_dim is not None:
        contiguous_dim %= len(shape)
    from cutlass import cute

    sym_int = cute.sym_int64 if use_int64_strides else cute.sym_int
    strides = tuple(
        1 if index == contiguous_dim else sym_int(divisibility=stride_divisibility)
        for index in range(len(shape))
    )
    if assumed_align is None:
        alignment_bits = stride_divisibility * dtype.width
        if alignment_bits % 8:
            raise ValueError("sub-byte fake tensors require an explicit assumed_align")
        assumed_align = max(1, alignment_bits // 8)
    return cute.runtime.make_fake_tensor(
        dtype,
        shape,
        stride=strides,
        assumed_align=assumed_align,
    )


def make_fake_compact_tensor(
    dtype: Any,
    shape: tuple[Any, ...],
    stride_order: tuple[int, ...] | None = None,
    assumed_align: int = 16,
) -> Any:
    """Create a fake compact tensor for a TVM-FFI compile signature.

    ``stride_order`` defaults to row-major (the last mode is contiguous).
    """
    from cutlass import cute

    if stride_order is None:
        stride_order = tuple(reversed(range(len(shape))))
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=stride_order, assumed_align=assumed_align
    )


class _CrossCompiledKernel:
    """Make an export-only build callable on the target GPU, including without a disk cache."""

    def __init__(self, compiled: Any) -> None:
        self.export_to_c = compiled.export_to_c
        with tempfile.TemporaryDirectory() as directory:
            obj = Path(directory) / "kernel.o"
            self.export_to_c(object_file_path=str(obj), function_name=EXPORT_FUNCTION_NAME)
            self._launch = _load_compiled(obj)

    def __call__(self, *args: Any) -> Any:
        return self._launch(*args)


def compile_tvm_ffi(
    entrypoint: Any,
    *compile_args: Any,
    name: str | None = None,
    opt_level: int | None = None,
    gpu_arch: str | None = None,
) -> Any:
    """Compile a fake-tensor signature with the TVM-FFI environment-stream ABI.

    ``compile_args`` are the callable's runtime signature without the stream; tensors must be
    fake CuTe tensors, not Torch tensors. This appends a fake environment stream, enables
    TVM-FFI and gives the outer artifact a stable name.

    Pass ``name=``, or expose ``get_name()`` on the entrypoint (class entrypoints do).
    ``opt_level`` is the CuTeDSL optimization level (``None`` is the compiler default). It
    changes the generated code, so a caller that needs a specific level must pass it. Every
    compile also passes ``compat.COMPILE_OPTIONS``, which pins C semantics for signed ``%``.
    ``gpu_arch`` overrides CuTeDSL's default target, which otherwise comes from ordinal 0.
    """
    if name is None:
        get_name = getattr(entrypoint, "get_name", None)
        if not callable(get_name):
            raise TypeError("compile_tvm_ffi() requires entrypoint.get_name() or name=")
        name = get_name()
    if not isinstance(name, str) or _VALID_NAME.fullmatch(name) is None:
        raise ValueError(
            "CuTeDSL compile names must start with a lowercase letter and contain "
            f"only lowercase letters, digits, and underscores; got {name!r}"
        )
    if any(_contains_torch_tensor(arg) for arg in compile_args):
        raise TypeError("compile_tvm_ffi() accepts fake CuTe tensors, not runtime Torch tensors")

    from cutlass import cute

    from attn_gym.warpdrive.compat import COMPILE_OPTIONS

    # A class entrypoint's jit wrapper is its `__call__`.
    jit_wrapper = entrypoint if hasattr(entrypoint, "set_name_prefix") else entrypoint.__call__
    jit_wrapper.set_name_prefix(name)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    options = (cute.EnableTVMFFI, *COMPILE_OPTIONS)
    if gpu_arch is not None:
        options += (cute.GPUArch(gpu_arch),)
    if opt_level is not None:
        options += (cute.OptLevel(opt_level),)
    compile_fn = cute.compile[options]
    from attn_gym.warpdrive.bounds import indexing
    from attn_gym.warpdrive.protocol import compile_scope

    # The opt-outs and index checks of what is traced belong to this compile; a kernel
    # entrypoint also scopes its own trace (__cute_trace_scope__).
    scope = getattr(entrypoint, "__cute_trace_scope__", None)
    with compile_scope(), indexing(False) if scope is None else scope():
        compiled = compile_fn(entrypoint, *compile_args, stream)
    # An explicit target incompatible with ordinal 0 gets an object but no execution engine.
    if gpu_arch is not None and compiled.engine is None:
        return _CrossCompiledKernel(compiled)
    return compiled


__all__ = ["compile_tvm_ffi", "make_fake_compact_tensor", "make_fake_strided_tensor"]
