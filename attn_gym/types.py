# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Types shared by attention operations."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypedDict


class Impl(str, Enum):
    """Select a fused or reference implementation without automatic fallback."""

    FUSED = "fused"
    REFERENCE = "reference"


def resolve_impl(impl: Impl | str) -> Impl:
    """Normalize an implementation selector and report the valid values."""
    try:
        return Impl(impl)
    except ValueError:
        valid = ", ".join(repr(member.value) for member in Impl)
        raise ValueError(f"unknown impl {impl!r}; expected one of {valid}") from None


class TuningOptions(TypedDict, total=False):
    """Kernel tuning control shared by public operations that tune."""

    autotune: bool
    """Benchmark candidate kernel configurations on first use and reuse the cached winner.
    ``False``, or ``torch.use_deterministic_algorithms(True)``, launches each kernel's pinned
    configuration instead. See "Autotuned kernels" in ``docs/determinism.md``."""


KernelOptionSpec = Mapping[str, tuple[type | tuple[str, ...], object]]
"""Map each ``kernel_options`` key to an ``(accepted, default)`` pair.

``accepted`` is ``bool``, a tuple of allowed strings, or a class checked with ``isinstance``.
"""

TUNING_OPTION_SPEC: KernelOptionSpec = {"autotune": (bool, True)}


def validate_kernel_options(
    kernel_options: Mapping[str, object] | None, spec: KernelOptionSpec, *, op_name: str
) -> dict[str, Any]:
    """Validate ``kernel_options`` and fill missing keys with defaults from ``spec``.

    Cross-option constraints remain the operation's responsibility.

    Raises:
        ValueError: An unknown key or a value outside its allowed literals.
        TypeError: A value does not match its required type; ``bool`` options reject integers.
    """
    options = {} if kernel_options is None else kernel_options
    unknown = options.keys() - spec.keys()
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"unsupported {op_name} kernel options: {names}")
    for name, value in options.items():
        accepted, _ = spec[name]
        if isinstance(accepted, tuple):
            if value not in accepted:
                *leading, last = map(repr, accepted)
                separator = ", or " if len(leading) > 1 else " or "
                raise ValueError(
                    f"kernel_options[{name!r}] must be {', '.join(leading)}{separator}{last}"
                )
        elif not (type(value) is bool if accepted is bool else isinstance(value, accepted)):
            raise TypeError(f"kernel_options[{name!r}] must be a {accepted.__name__}")
    return {name: options.get(name, default) for name, (_, default) in spec.items()}


@dataclass(frozen=True, slots=True)
class AuxRequest:
    """Specifies which auxiliary outputs an attention operation returns.

    Attributes:
        lse: If True, include the log-sum-exp tensor in the returned auxiliary data.
    """

    lse: bool = False
