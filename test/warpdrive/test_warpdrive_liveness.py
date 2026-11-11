"""Finite protocol graphs, independent of CuTe tracing and compilation."""

from collections import defaultdict
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
    monkeypatch.setattr(liveness, "_channels", broken)
    assert "analysis failed" in check(None, []).detail


def fixture_trace(events=(), peer_events=None, tma_verdict="verified", **changes):
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
        internal=False,
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
        _ws_tma={"a": SimpleNamespace(verdict=tma_verdict)},
    )
    roles = [RoleEvents("p", (0,), False, list(events), None)]
    if peer_events is not None:
        roles.append(RoleEvents("q", (1,), False, list(peer_events), None))
    return kernel, roles


def event(kind="op", where=(), opted_out=False, **data):
    return Event(kind, data, "toy.py:42", where, (), opted_out)


def take(kind="acquire", slot=None, order=0, **kwargs):
    return event(chan="a", op=kind, slot=slot, order=order, count=1, **kwargs)


def close(producer=True, slot=None, order=0, **kwargs):
    return take("commit" if producer else "release", slot, order, **kwargs)


def barrier(threads=32, native=False):
    return SimpleNamespace(threads=threads, _native_outside_roles=native)


LOOP = ((0, "loop"),)


def extracted(kernel, roles, trips=1):
    channels = liveness._channels(kernel)
    events = {r.name: liveness._events(r, channels, defaultdict(set)) for r in roles}
    return channels, liveness._ops(channels, events, trips)


@pytest.mark.parametrize(
    "changes, events, why",
    [
        ({"external": "raw"}, [], "external"),
        ({"override": {"tx": 32}}, [], "overridden"),
        ({"turnstile": "turn"}, [], "unknown epoch"),
        ({"sharing": lambda *args: Sharing.EXCLUSIVE}, [], "unknown epoch"),
        ({"named": True}, [take("tail")], "named or rotating tail"),
        ({}, [take("barrier pointer")], "barrier pointer"),
        ({"producer": object()}, [], "signal mechanism"),
        ({"internal": True}, [], "scheduler channel"),
        ({}, [take(where=((0, "if"),))], "runtime branch"),
        ({}, [take(where=((0, "loop"), (1, "if")))], "runtime branch"),
        ({"producer": Tma(32)}, [take(), close()], "committed without reading its barrier"),
        ({"producer": Tma(32)}, [take("barrier pointer")], "barrier pointer"),
        ({}, [take("tma pointer")], "tma pointer"),
        ({}, [event("barrier", where=((0, "if"),), barrier=barrier())], "runtime branch"),
        ({}, [take(where=((0, "other"),))], "unknown region"),
        ({}, [take(opted_out=True)], 'unchecked("liveness")'),
        ({}, [take("state")], "state"),
        ({}, [take("skip")], "skip"),
        ({}, [take(span=True)], "span"),
        ({}, [take(), take("retire")], "retire"),
        ({}, [event("barrier")], "named barrier shared between roles"),
        ({}, [event("barrier", barrier=barrier(64))], "named barrier shared between roles"),
        ({}, [event("barrier_arrive", barrier=barrier())], "named barrier shared between roles"),
        ({}, [event("barrier_native")], "named barrier shared between roles"),
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
    channels, ops = extracted(kernel, roles)
    tails = [o for o in ops if o.kind == "tail"]
    assert [(o.slot, o.epoch) for o in tails] == [(0, 0)]
    ops += publish("p", "b")
    ops += [op("q", "wait", slot=0)] + consume("q", "b") + [op("q", "release", slot=0)]
    channels["b"] = channel()
    assert analyze(channels, ops).verdict == "error"
    assert analyze(channels, [o for o in ops if o.kind != "tail"]).verdict == "verified"


def looped(p_events, q_events, **changes):
    return check(*fixture_trace(p_events, q_events, **changes))


TMA = {"producer": Tma(32), "mechanism": lambda *args: Umma}
DRAIN = [take("wait"), close(False)]


def test_a_tma_fill_is_placed_where_its_barrier_is_read():
    """The commit issues nothing: a tail before it still sees the stage filled and drained."""
    fill = [take(), take("tma pointer"), take("tail"), close()]
    result = looped(fill, DRAIN, **TMA)
    assert result.verdict == "no deadlock found"
    assert result.detail == "TMA copy at barrier read"
    # Without a copy to fill the stage the same tail waits on a release that never comes.
    late = [take(), take("tail"), take("tma pointer"), close()]
    result = looped(late, DRAIN, **TMA)
    assert result.verdict == "possible deadlock"  # where the copy really is issued is assumed


def test_tma_needs_its_copies_verified():
    fill = [take(), take("tma pointer"), close()]
    result = looped(fill, DRAIN, tma_verdict="unverified", **TMA)
    assert result.verdict == "unverified" and "TMA copies of a" in result.detail


def test_runtime_loops_are_repeated_and_report_what_was_assumed():
    fill = [take(where=LOOP), close(where=LOOP)]
    drain = [take("wait", where=LOOP), close(False, where=LOOP)]
    result = looped(fill, drain)
    assert result.verdict == "no deadlock found"
    assert result.detail == "1, 2 and 3 loop trips, matched counts"
    nested = ((0, "loop"), (1, "loop"))
    twice = [retag(ev, nested) for ev in drain]  # a loop inside the loop: counts differ
    assert looped(fill, twice).verdict == "unverified"


def retag(ev, where):
    return Event(ev.kind, ev.data, ev.site, where, ev.open, ev.opted_out)


def test_a_cycle_that_needs_one_trip_more_than_the_ring_is_deep_is_found():
    """p fills a (two stages) in a loop, then b; q takes b first. Two trips fit; three do not."""
    kernel, roles = fixture_trace(
        [take(where=LOOP), close(where=LOOP), on("b", take()), on("b", close())],
        [on("b", take("wait")), on("b", close(False)), *(retag(ev, LOOP) for ev in DRAIN)],
    )
    kernel._ws_chans["b"] = SimpleNamespace(**{**vars(kernel._ws_chans["a"]), "n": 1})
    result = check(kernel, roles)
    assert result.verdict == "possible deadlock"
    assert "every runtime loop running 3 times" in result.findings[0]


def on(chan, ev):
    return Event(ev.kind, {**ev.data, "chan": chan}, ev.site, ev.where, ev.open, ev.opted_out)


def test_loop_counts_that_do_not_line_up_are_unverified():
    fill = [take(where=LOOP), close(where=LOOP), take(order=1), close(order=1)]  # n + 1 fills
    drain = [take("wait", where=LOOP), close(False, where=LOOP)]  # n takes
    result = looped(fill, drain)
    assert result.verdict == "unverified"
    assert "loop counts on a do not line up" in result.detail


def test_a_named_barrier_only_one_role_syncs_on_is_ignored():
    sync = event("barrier", barrier=barrier())
    assert looped([sync, take(), close()], DRAIN).verdict == "verified"
    # In a loop the role's threads may not all reach it the same number of times.
    result = looped([retag(sync, LOOP), take(), close()], DRAIN)
    assert result.verdict == "no deadlock found"
    other = [event("barrier", barrier=sync.data["barrier"]), *DRAIN]
    assert "shared between roles" in looped([sync, take(), close()], other).detail
    native = event("barrier", barrier=barrier(native=True))
    assert "shared between roles" in looped([native, take(), close()], DRAIN).detail


def test_ignored_syncs_are_not_repeated(monkeypatch):
    """Only channel ops are unrolled, so the node budget bounds the work."""
    deep = tuple((key, "loop") for key in range(1, 7))
    fill = [take(where=LOOP), close(where=LOOP), event("barrier", where=deep, barrier=barrier())]
    seen = []
    unroll = liveness._unroll
    monkeypatch.setattr(
        liveness, "_unroll", lambda events, *a: seen.append(events) or unroll(events, *a)
    )
    assert looped(fill, [retag(ev, LOOP) for ev in DRAIN]).verdict == "no deadlock found"
    assert all(ev.kind == "op" for events in seen for ev in events)


def test_events_that_add_no_node_still_use_the_budget(monkeypatch):
    """A tail before any acquire waits nothing; repeating it must not run unbounded."""
    deep = tuple((key, "loop") for key in range(4))
    monkeypatch.setattr(liveness, "MAX_NODES", 100)
    result = looped([take("tail", where=deep)], [], n=16)
    assert result.verdict == "unverified" and "budget exceeded" in result.detail


def test_a_close_is_matched_to_its_own_handle_not_to_the_oldest():
    """p commits its second stage of a, waits for b, then commits the first; q needs the first
    before it sends b."""
    p = [
        take(),
        take(order=1),
        close(order=1),
        on("b", take("wait")),
        on("b", close(False)),
        close(),
    ]
    q = [take("wait"), on("b", take()), on("b", close()), close(False)]
    q += [take("wait", order=1), close(False, order=1)]
    kernel, roles = fixture_trace(p, q)
    b = SimpleNamespace(**{**vars(kernel._ws_chans["a"]), "n": 1})
    b.producers, b.consumers = kernel._ws_chans["a"].consumers, kernel._ws_chans["a"].producers
    kernel._ws_chans["b"] = b
    result = check(kernel, roles)
    assert result.verdict == "error" and "deadlock" in result.findings[0]


def test_only_a_proven_finding_raises_in_strict_mode():
    proven = liveness.Liveness("error", findings=["deadlock: a waits for b."])
    with pytest.raises(liveness.ProtocolError, match="Override with unchecked"):
        liveness.report(proven, "strict", "Toy")
    possible = liveness.Liveness("possible deadlock", findings=["deadlock: a waits for b."])
    with pytest.warns(UserWarning, match="deadlock"):
        liveness.report(possible, "strict", "Toy")
