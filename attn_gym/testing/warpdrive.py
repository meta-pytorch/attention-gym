"""Shared host-only helpers for warpdrive tests."""

import re
from collections.abc import Callable

import pytest

from attn_gym.warpdrive import ProtocolError, WarpSpecialized

# A launch CUDA rejects (here: a grid past its limits). CuTeDSL words the error differently
# across versions and for fresh and cached kernels; every form names the CUDA error.
INVALID_LAUNCH = "cudaErrorInvalidValue"


def raises_protocol(match: str, fn: Callable[[], object]) -> None:
    with pytest.raises(Exception, match=match) as info:
        fn()
    assert isinstance(info.value, ProtocolError) or "ProtocolError" in repr(info.value)


def alias_section(op: WarpSpecialized) -> str:
    text = op.describe()
    return text[text.index("Alias diagnostic") :]


def status(op: WarpSpecialized, pair: str) -> str:
    match = re.search(rf"\n  {re.escape(pair)} \((?:tmem|smem)\)  ([a-z ]+)", alias_section(op))
    assert match, alias_section(op)
    return match.group(1).strip().split(" among")[0]
