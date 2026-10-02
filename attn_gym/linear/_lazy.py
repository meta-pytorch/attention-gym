# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Lazy loaders for backend-backed exports and operator kernels (see Note: Lazy Imports)."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from types import ModuleType

import torch


def lazy_exports(
    module_name: str, exports: Mapping[str, str], *, requirement: str
) -> Callable[[str], object]:
    """Return a module ``__getattr__`` that imports each export's owning module on first use.

    ``exports`` maps a public name to the module that defines it. A missing optional dependency
    surfaces as ``ImportError("<name> requires the optional <requirement>: pip install ...")``.
    """

    def __getattr__(name: str) -> object:
        owner = exports.get(name)
        if owner is None:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
        try:
            module = importlib.import_module(owner)
        except ImportError as error:
            raise ImportError(
                f"{name} requires the optional {requirement}: pip install attn-gym[linear]"
            ) from error
        return getattr(module, name)

    return __getattr__


def register_lazy_cuda_impls(
    backend: Callable[[], ModuleType], kernels: Mapping[str, str]
) -> None:
    """Register ``attn_gym::<op>`` CUDA kernels that call ``backend().<attr>`` on dispatch.

    ``kernels`` maps an operator name to the backend attribute implementing it, so the optional
    backend is imported when an operator first executes rather than when it is registered.
    """

    def kernel(attr: str) -> Callable[..., object]:
        return lambda *args: getattr(backend(), attr)(*args)

    for op, attr in kernels.items():
        torch.library.impl(f"attn_gym::{op}", "CUDA", kernel(attr))
