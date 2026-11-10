"""Resolve string role annotations in the defining module.

``from __future__ import annotations`` makes every annotation a string. Resolve it
with the role's globals, not the globals of the collected ``@cute.jit`` wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("cutlass.cute")

from cutlass import cute

from attn_gym.warpdrive import (
    Consumer,
    Instance,
    Producer,
    ProtocolError,
    Slots,
    Threads,
    WarpSpecialized,
    role,
)


@dataclass(frozen=True)
class FutureParams:
    steps: int


class FutureToy(WarpSpecialized):
    def __init__(self):
        self.ch = Slots(2, Threads, Threads)

    @role(warps=[range(1), range(1, 2)])
    @cute.jit
    def prod(self, stage: Instance, ch: Producer[Instance], params: FutureParams):
        return ch

    @role(warps=2)
    @cute.jit
    def cons(self, ch: Consumer, params: FutureParams):
        return ch


class FutureTypoToy(WarpSpecialized):
    def __init__(self):
        self.ch = Slots(2, Threads, Threads)

    @role(warps=0)
    @cute.jit
    def prod(self, ch: Producer, params: FutureParams):
        return ch

    @role(warps=1)
    @cute.jit
    def cons(self, ch: Consumer, params: FutreParams):  # noqa: F821 (the typo under test)
        return ch


def test_future_annotations_resolve_in_defining_module():
    """Layer annotations, subscripted ends, and a module-level context type all resolve."""
    text = FutureToy().describe()  # an unresolved annotation raises (see below)
    assert "params   FutureParams\n" in text, text
    assert "ch  Slots(2) prod[0]@0, prod[1]@1 -> cons\n" in text, text


def test_future_annotations_unknown_name_is_still_an_error():
    with pytest.raises(ProtocolError, match=r"'FutreParams', which does not resolve \(NameError"):
        FutureTypoToy().describe()
