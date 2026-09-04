# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module ``__getattr__`` factory for backend-backed exports (see Note: Lazy Imports)."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping


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
