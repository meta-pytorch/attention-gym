# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
#
# Portions of this file are derived from flash-linear-attention
# (https://github.com/fla-org/flash-linear-attention) and are licensed under
# the MIT license; for the full list of FLA contributors, visit
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
# The remaining portions are licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Consolidated utilities for the KDA kernels (forward + backward). Merges the
# former ``common/utils.py`` (Meta) and ``fla_ops/utils.py`` (FLA) into a single
# module. Targets NVIDIA GPUs only.

import contextlib
import functools
import inspect
import logging
import os
import sys
import warnings
from collections.abc import Callable
from functools import cache, lru_cache
from typing import Any

import torch
import triton
import triton.language as tl
from packaging import version
from triton.language.extra import libdevice

from attn_gym._backends.triton import utils as triton_utils
from attn_gym.linear.kda.constants import is_sm100_kda_capability

SUPPORTS_AUTOTUNE_CACHE = triton_utils.SUPPORTS_AUTOTUNE_CACHE
autotune_cache_kwargs = triton_utils.autotune_cache_kwargs

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment checks (OS / Triton / Python version)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def check_environments():
    """Warn if OS, Triton, or Python versions are below recommendations.

    Body only runs once due to lru_cache.
    """
    if sys.platform == "win32":
        try:
            from importlib.metadata import PackageNotFoundError, metadata

            metadata("triton-windows")
        except PackageNotFoundError:
            logger.warning(
                "Detected Windows operating system. Consider installing triton-windows "
                "(https://github.com/triton-lang/triton-windows) for better compatibility. "
                "Without it, some features may not work correctly.",
            )

    triton_version = version.parse(triton.__version__)
    required_triton_version = version.parse("3.3.0")
    if triton_version < required_triton_version:
        logger.warning(
            f"Current Triton version {triton_version} is below the recommended 3.3.0 version. "
            "Errors may occur and these issues will not be fixed. "
            "Please consider upgrading Triton.",
        )

    py_version = version.parse(f"{sys.version_info.major}.{sys.version_info.minor}")
    required_py_version = version.parse("3.11")
    if py_version < required_py_version:
        logger.warning(
            f"Current Python version {py_version} is below the recommended 3.11 version. "
            "It is recommended to upgrade to Python 3.11 or higher for the best experience.",
        )


check_environments()


# ---------------------------------------------------------------------------
# Device detection (NVIDIA only)
# ---------------------------------------------------------------------------
def _cpu_device_warning():
    warnings.warn(("Triton is not supported on current platform, roll back to CPU."), stacklevel=2)


@cache
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except Exception:
        _cpu_device_warning()
        return "cpu"


device = "cuda"
device_torch_lib = torch.cuda
device_name = "cuda"
device_platform = get_available_device()

IS_AMD = device_platform == "hip"
IS_NVIDIA = device_platform == "cuda"
IS_NVIDIA_BLACKWELL = IS_NVIDIA and torch.cuda.get_device_capability()[0] in (10, 12)
IS_TF32_SUPPORTED = IS_NVIDIA and torch.cuda.get_device_capability(0)[0] >= 8
IS_GATHER_SUPPORTED = hasattr(triton.language, "gather")


@lru_cache(maxsize=8)
def is_sm100_kda_target(device: torch.device) -> bool:
    """Return whether a physical device should use the SM100-specific KDA route."""
    properties = torch.cuda.get_device_properties(device)
    return is_sm100_kda_capability((properties.major, properties.minor))


if IS_NVIDIA and not IS_TF32_SUPPORTED:
    # Make old cards happy, since triton will use tf32 by default.
    os.environ["TRITON_F32_DEFAULT"] = "ieee"


if IS_NVIDIA_BLACKWELL:
    logger.info("Blackwell detected: registering default global_scratch allocator.")
triton_utils.configure_triton_allocator()


# ---------------------------------------------------------------------------
# Triton math ops
# ---------------------------------------------------------------------------
exp2 = tl.math.exp2


@triton.jit
def masked_exp2(value, mask, FASTMATH: tl.constexpr = True):
    """Mask exponent inputs and select approximate or precise exponentiation."""
    exponent = tl.where(mask, value, 0.0)
    result = exp2(exponent) if FASTMATH else libdevice.exp2(exponent)
    return tl.where(mask, result, 0.0)


@triton.jit
def exp(x):
    return tl.exp(x.to(tl.float32))


if not IS_GATHER_SUPPORTED:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        return None
else:
    gather = tl.gather  # type: ignore


# ---------------------------------------------------------------------------
# Device context
# ---------------------------------------------------------------------------
def custom_device_ctx(index: int):
    if index is None:
        return contextlib.nullcontext()
    try:
        return device_torch_lib.device(index)
    except (AttributeError, AssertionError, RuntimeError):
        return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# input_guard decorator
# ---------------------------------------------------------------------------
def _skip_contiguous(
    no_guard_contiguous: bool | list[str] | tuple[str, ...] | set[str],
    param_name: str,
    skip_params: set[str],
) -> bool:
    return no_guard_contiguous is True or param_name in skip_params


def _contiguous_if_needed(arg: Any, skip: bool) -> Any:
    if isinstance(arg, torch.Tensor) and not skip:
        return arg.contiguous()
    return arg


def input_guard(
    fn: Callable[..., torch.Tensor] | None = None,
    *,
    no_guard_contiguous: bool | list[str] | tuple[str, ...] | set[str] = False,
) -> (
    Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]]
    | Callable[..., torch.Tensor]
):
    """Ensure all input tensors are contiguous and set the device from input tensors.

    Args:
        no_guard_contiguous: If True, skip all contiguous checks. If a
            list/tuple/set of parameter names, skip contiguous checks for those.
    """

    def decorator(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        skip_params = (
            set(no_guard_contiguous)
            if isinstance(no_guard_contiguous, (list, tuple, set))
            else set()
        )

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            processed_args = []
            for i, arg in enumerate(args):
                param_name = param_names[i] if i < len(param_names) else f"__arg_{i}"
                processed_args.append(
                    _contiguous_if_needed(
                        arg, _skip_contiguous(no_guard_contiguous, param_name, skip_params)
                    )
                )

            processed_kwargs = {}
            for k, v in kwargs.items():
                processed_kwargs[k] = _contiguous_if_needed(
                    v, _skip_contiguous(no_guard_contiguous, k, skip_params)
                )

            tensor = None
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    tensor = arg
                    break
            if tensor is None:
                for value in kwargs.values():
                    if isinstance(value, torch.Tensor):
                        tensor = value
                        break

            if tensor is not None:
                ctx = custom_device_ctx(tensor.device.index)
            else:
                ctx = contextlib.nullcontext()

            with ctx:
                return fn(*processed_args, **processed_kwargs)

        return wrapper

    # Handle direct usage without parentheses: @input_guard
    if fn is not None:
        return decorator(fn)
    return decorator
