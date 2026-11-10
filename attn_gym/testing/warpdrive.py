"""Shared host-only helpers for warpdrive tests."""

import re
from collections.abc import Callable

import pytest

from attn_gym.warpdrive import ProtocolError, WarpSpecialized

# Match a rejected CUDA launch, such as an oversized grid, in pytest.raises.
# CuTeDSL may report the name or, on a fresh process's first launch, code 1
# (cudaErrorInvalidValue). The wording also varies by version and cache state.
INVALID_LAUNCH = r"cudaErrorInvalidValue|CUDA Error Code: 1\b"


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
