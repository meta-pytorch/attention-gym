"""Finite protocol graphs, independent of CuTe tracing and compilation."""

from types import SimpleNamespace

import pytest

from attn_gym.warpdrive import liveness
from attn_gym.warpdrive.alias import RoleEvents
from attn_gym.warpdrive.channels import Sharing, Threads, Tma, Umma
from attn_gym.warpdrive.liveness import Channel, Op, analyze, check
from attn_gym.warpdrive.protocol import Event


def channel(producers=("p",), consumers=("q",), **kwargs):
    return Channel(
        1, (tuple((r, None) for r in consumers), tuple((r, None) for r in producers)), **kwargs
    )


def op(role, kind, name="a", epoch=0, slot=None):
    return Op(role, kind, name, slot, epoch, f"{role}.py:{epoch + 10}")


def publish(role="p", name="a", epoch=0, slot=None):
    return [op(role, kind, name, epoch, slot) for kind in ("acquire", "commit")]


def consume(role="q", name="a", epoch=0, slot=None):
    return [op(role, kind, name, epoch, slot) for kind in ("wait", "release")]


def test_balanced_straight_line_is_verified():
    assert analyze({"a": channel()}, publish() + consume()).verdict == "verified"
    assert analyze({"unused": channel()}, []).verdict == "verified"


def test_cross_role_wait_cycle():
    ops = consume("p", "b") + publish() + consume() + publish("q", "b")
    result = analyze({"a": channel(), "b": channel(("q",), ("p",))}, ops)
    assert result.verdict == "error"
    assert "deadlock" in result.findings[0]
    assert "p.py:10" in result.findings[0] and "q.py:10" in result.findings[0]


@pytest.mark.parametrize("stages, verdict", [(1, "error"), (2, "verified")])
def test_ring_reuse_needs_release_before_acquire(stages, verdict):
    a = channel()
    a.stages = stages
    ops = publish() + publish(epoch=1) + publish("p", "b")
    ops += [op("q", "wait")] + consume("q", "b")
    ops += [op("q", "release")] + consume(epoch=1)
    result = analyze({"a": a, "b": channel()}, ops)
    assert result.verdict == verdict
    if verdict == "error":
        assert "deadlock" in result.findings[0]


def test_named_commit_wait_is_a_rendezvous():
    ops = publish() + [op("p", "rendezvous")] + publish("p", "b")
    ops += consume("q", "b") + consume()
    assert analyze({"a": channel(named=True), "b": channel()}, ops).verdict == "error"
    assert (
        analyze(
            {"a": channel(named=True)}, publish() + [op("p", "rendezvous")] + consume()
        ).verdict
        == "verified"
    )


@pytest.mark.parametrize("missing_side", ["producer", "consumer"])
def test_zero_op_bound_join_participant_is_not_ignored(missing_side):
    a = channel(
        ("p", "absent") if missing_side == "producer" else ("p",),
        ("q", "absent") if missing_side == "consumer" else ("q",),
    )
    result = analyze({"a": a}, publish() + consume())
    assert result.verdict == "error"
    assert "absent" in result.findings[0]


def test_named_acquire_needs_producer_peers_even_at_first_epoch():
    ops = publish("p1") + publish("p1", "b")
    ops += consume("p2", "b") + publish("p2") + consume()
    chans = {"a": channel(("p1", "p2"), named=True), "b": channel(("p1",), ("p2",))}
    assert analyze(chans, ops).verdict == "error"
    chans["a"].named = False
    assert analyze(chans, ops).verdict == "verified"


def test_named_empty_seeds_are_before_consumer_body():
    ops = publish() + publish("p", "b") + consume("q", "b") + consume()
    assert analyze({"a": channel(named=True), "b": channel()}, ops).verdict == "verified"


def test_named_consumer_peer_arrival_cycle():
    ops = publish() + consume("q1") + publish("q1", "b")
    ops += consume("q2", "b") + consume("q2")
    chans = {"a": channel(consumers=("q1", "q2"), named=True), "b": channel(("q1",), ("q2",))}
    assert analyze(chans, ops).verdict == "error"


def test_static_slot_owners_and_round_robin_epochs():
    slots = Channel(2, ((("q0", 0), ("q1", 1)), (("p", None),)), fixed=True)
    ops = publish(slot=0) + publish(slot=1) + consume("q0", slot=0) + consume("q1", slot=1)
    assert analyze({"a": slots}, ops).verdict == "verified"
    ring = channel(consumers=("q0", "q1"), turns=(2, 1))
    ring.stages = 2
    ops = publish() + publish(epoch=1) + consume("q0") + consume("q1", epoch=1)
    assert analyze({"a": ring}, ops).verdict == "verified"


@pytest.mark.parametrize("missing", ["acquire", "commit", "wait", "release"])
def test_missing_managed_operation_is_an_error(missing):
    result = analyze({"a": channel()}, [o for o in publish() + consume() if o.kind != missing])
    assert result.verdict == "error"
    assert "unbalanced epochs" in result.findings[0]


@pytest.mark.parametrize("budget", ["MAX_NODES", "MAX_EDGES"])
def test_both_graph_budgets_leave_unverified(monkeypatch, budget):
    monkeypatch.setattr(liveness, budget, 1)
    result = analyze({"a": channel()}, publish() + consume())
    assert result.verdict == "unverified"
    assert "budget exceeded" in result.detail


def test_unknown_epoch_correspondence_and_ops_are_unverified():
    a = channel(("p0", "p1"), ("q0", "q1"), turns=(2, 2))
    a.stages = 2
    assert (
        analyze({"a": a}, publish("p1", epoch=1) + consume("q1", epoch=1)).verdict == "unverified"
    )
    assert analyze({"a": a}, [op("p0", "raw")]).verdict == "unverified"
    kernel, roles = fixture_trace()
    kernel._ws_raw_bars = {"raw": object()}
    assert "raw barriers" in check(kernel, roles).detail


def test_internal_exception_never_fails_compilation(monkeypatch):
    def broken(*args):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(liveness, "_analyze", broken)
    assert "analysis failed" in analyze({}, []).detail
    monkeypatch.setattr(liveness, "_extract", broken)
    assert "analysis failed" in check(None, []).detail


def fixture_trace(events=(), **changes):
    role = SimpleNamespace(name="p", role=object())
    peer = SimpleNamespace(name="q", role=object())
    chan = SimpleNamespace(
        n=2,
        external=None,
        override=None,
        producer=Threads,
        turnstile=None,
        producers=[(role, None)],
        consumers=[(peer, None)],
        fixed=False,
        named=False,
        mechanism=lambda *args: Threads,
        sharing=lambda *args: Sharing.JOIN,
        turns=lambda *args: 1,
    )
    chan.__dict__.update(changes)
    kernel = SimpleNamespace(
        cluster=SimpleNamespace(size=1),
        _ws_chans={"a": chan},
        _ws_raw_bars={},
        _ws_raw_barrier_access="",
    )
    roles = [RoleEvents("p", (0,), False, list(events), None)]
    return kernel, roles


def event(kind="op", where=(), opted_out=False, **data):
    return Event(kind, data, "toy.py:42", where, (), opted_out)


def take(kind="acquire", slot=None, order=0, **kwargs):
    return event(chan="a", op=kind, slot=slot, order=order, count=1, **kwargs)


def close(producer=True, slot=None, order=0, **kwargs):
    return take("commit" if producer else "release", slot, order, **kwargs)


@pytest.mark.parametrize(
    "changes, events, why",
    [
        ({"external": "raw"}, [], "external"),
        ({"override": {"tx": 32}}, [], "overridden"),
        ({"producer": Tma(32)}, [], "TMA/async"),
        ({"turnstile": "turn"}, [], "unknown epoch"),
        ({"sharing": lambda *args: Sharing.EXCLUSIVE}, [], "unknown epoch"),
        ({"mechanism": lambda *args: Umma}, [], "TMA/async"),
        ({"named": True}, [take("tail")], "named or rotating tail"),
        ({}, [take("barrier pointer")], "barrier pointer"),
        ({}, [take(where=((0, "loop"),))], "runtime loop/branch"),
        ({}, [take(where=((0, "if"),))], "runtime loop/branch"),
        ({}, [take(where=((0, "other"),))], "unknown region"),
        ({}, [take(opted_out=True)], 'unchecked("liveness")'),
        ({}, [take("state")], "state"),
        ({}, [take("skip")], "skip"),
        ({}, [take(span=True)], "span"),
        ({}, [take(), take("retire")], "retire"),
        ({}, [event("barrier")], "standalone named barrier"),
        ({}, [event("barrier_native")], "standalone named barrier"),
        ({}, [take("turn")], "turn"),
    ],
)
def test_extraction_boundary_is_unverified(changes, events, why):
    result = check(*fixture_trace(events, **changes))
    assert result.verdict == "unverified"
    assert why in result.detail
    if events:
        assert "p at toy.py:42" in result.detail


def test_raw_role_cross_cta_and_raw_signal_are_unverified():
    kernel, roles = fixture_trace()
    roles[0].raw = True
    assert "raw role p" in check(kernel, roles).detail
    roles[0].raw = False
    kernel.cluster.size = 2
    assert "cross-CTA" in check(kernel, roles).detail
    assert "raw signal" in check(*fixture_trace([take(), take("commit(raw signal)")])).detail


def test_slots_tail_expands_only_acquired_slots_at_their_epoch():
    kernel, roles = fixture_trace([take(slot=0), close(slot=0), take("tail")], fixed=True)
    channels, ops = liveness._extract(kernel, roles)
    tails = [o for o in ops if o.kind == "tail"]
    assert [(o.slot, o.epoch) for o in tails] == [(0, 0)]
    ops += publish("p", "b")
    ops += [op("q", "wait", slot=0)] + consume("q", "b") + [op("q", "release", slot=0)]
    channels["b"] = channel()
    assert analyze(channels, ops).verdict == "error"
    assert analyze(channels, [o for o in ops if o.kind != "tail"]).verdict == "verified"
