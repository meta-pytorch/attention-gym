"""Compile CuTeDSL entry points against fake tensors with the TVM-FFI stream ABI."""

from __future__ import annotations

import re
from typing import Any

import torch

_VALID_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


def _contains_torch_tensor(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return True
    if isinstance(value, (tuple, list)):
        return any(_contains_torch_tensor(item) for item in value)
    if isinstance(value, dict):
        return any(
            _contains_torch_tensor(key) or _contains_torch_tensor(item)
            for key, item in value.items()
        )
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


def compile_tvm_ffi(
    entrypoint: Any,
    *compile_args: Any,
    name: str | None = None,
    opt_level: int | None = None,
) -> Any:
    """Compile a fake-tensor signature with the TVM-FFI environment-stream ABI.

    ``compile_args`` are the callable's runtime signature without the stream; tensors must be
    fake CuTe tensors, not Torch tensors. This appends a fake environment stream, enables
    TVM-FFI and gives the outer artifact a stable name.

    Pass ``name=``, or expose ``get_name()`` on the entrypoint (class entrypoints do).
    ``opt_level`` is the CuTeDSL optimization level (``None`` is the compiler default). It
    changes the generated code, so a caller that needs a specific level must pass it.
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

    # `set_name_prefix` needs CuTeDSL >= 4.5.0; 4.6.2 removed `_name_prefix=`. A class
    # entrypoint's jit wrapper is its `__call__`.
    jit_wrapper = entrypoint if hasattr(entrypoint, "set_name_prefix") else entrypoint.__call__
    jit_wrapper.set_name_prefix(name)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    if opt_level is None:
        compile_fn = cute.compile[cute.EnableTVMFFI]
    else:
        compile_fn = cute.compile[cute.EnableTVMFFI, cute.OptLevel(opt_level)]
    return compile_fn(
        entrypoint,
        *compile_args,
        stream,
    )


__all__ = ["compile_tvm_ffi", "make_fake_compact_tensor", "make_fake_strided_tensor"]
