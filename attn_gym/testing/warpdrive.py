"""Shared host-only helpers for warpdrive tests."""

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
    """Alias status for ``pair``, written as ``"Y ~ X"`` in ``describe()``."""
    aliases = op.checks().aliases
    names = tuple(pair.split(" ~ "))
    assert names in aliases, alias_section(op)
    return aliases[names]
