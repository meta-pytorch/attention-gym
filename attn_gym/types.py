# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Types shared by attention operations."""

from dataclasses import dataclass
from enum import Enum


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


@dataclass(frozen=True, slots=True)
class AuxRequest:
    """Specifies which auxiliary outputs an attention operation returns.

    Attributes:
        lse: If True, include the log-sum-exp tensor in the returned auxiliary data.
    """

    lse: bool = False
