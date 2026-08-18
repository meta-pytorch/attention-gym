# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Types shared by public linear-attention operations."""

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


__all__ = ["Impl", "resolve_impl"]
