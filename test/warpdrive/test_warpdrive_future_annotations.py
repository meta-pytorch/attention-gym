"""Role annotations resolve in the defining module under ``from __future__ import annotations``.

The future import turns every annotation into a string; the layer must evaluate it with the
globals of the role's own module, not those of the ``@cute.jit`` wrapper object that role
collection holds.
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
    toy = FutureToy()
    toy._collect()
    roles = {r.name: r for r in toy._ws_roles}
    assert roles["prod[1]"].params[0] == ("stage", Instance)
    assert roles["prod[1]"].params[1][1].side is Producer
    assert dict(roles["cons"].params)["ch"] is Consumer
    assert toy._ws_ctx_type is FutureParams
    assert "prod[1]" in toy.describe()


def test_future_annotations_unknown_name_is_still_an_error():
    with pytest.raises(ProtocolError, match=r"'FutreParams', which does not resolve \(NameError"):
        FutureTypoToy().describe()
