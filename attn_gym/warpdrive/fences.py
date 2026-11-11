"""Fence rules for managed channel handoffs.

``RULES`` lists access pairs, their required fences and PTX reasons. A commit hands
storage to consumers; a release hands it to the next epoch's producer.

``derived_fences`` supplies two SMEM defaults: a release fence for thread consumers
before a TMA refill, and a commit fence for thread producers before an Umma read.
TMA fills imply SMEM even without ``payload=`` or ``buffer=``. External sides get no
default fence. Cross-CTA placement does not change these defaults.

``check`` checks explicit overrides against annotations recorded while channel handles
were open, including reuse of a ring's own stage. It uses ``alias.Access`` footprints
and reports through ``alias_check``. Explicit values replace defaults; the check never
changes emitted fences.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

from attn_gym.warpdrive import alias
from attn_gym.warpdrive.alias import PTX
from attn_gym.warpdrive.protocol import Event, ProtocolError

FENCE_KINDS = ("tmem_store", "tmem_load", "smem_async")
SIDE_FENCES = {"producer": "commit_fence", "consumer": "release_fence"}


@dataclass(frozen=True)
class Access:
    agent: str  # "threads" (generic proxy, tcgen05.ld/st), "tma" or "umma" (async)
    write: bool


@dataclass(frozen=True)
class Rule:
    """Require ``kind`` between ``mine`` and ``theirs`` accesses to ``storage``.

    ``visible=False`` leaves an absent fence unverified because a manual wait may be
    invisible to the trace.
    """

    storage: str  # "smem" | "tmem"
    mine: tuple[Access, ...]
    theirs: tuple[Access, ...]
    kind: str
    reason: str
    citation: str
    visible: bool = True


_READ, _WRITE = Access("threads", False), Access("threads", True)
_ASYNC_PROXY = f"PTX ISA 9.5 9.7.10.28.2 Async Proxy ({PTX}#async-proxy)"
_TCGEN05_SYNC = (
    f"PTX ISA 9.5 9.7.18.6.3 ({PTX}#tcgen05-memory-consistency-model-inter-thread-sync)"
)

RULES = (
    Rule(
        "smem",
        (_WRITE,),
        (Access("umma", False), Access("tma", False)),
        "smem_async",
        "MMA operands and TMA stores read SMEM through the async proxy, which sees generic "
        "stores only after the writer's own fence.proxy.async",
        f"{_ASYNC_PROXY}; 9.7.18.6.5 ({PTX}#tcgen05-memory-consistency-model-smem-access)",
    ),
    Rule(
        "smem",
        (_READ, _WRITE),
        (Access("tma", True),),
        "smem_async",
        "a TMA fill writes through the async proxy, ordered after a generic access only by "
        "fence.proxy.async in the accessing thread, not by the release and acquire",
        f"{_ASYNC_PROXY}; Memory Consistency Model, proxies and the causality axiom",
    ),
    Rule(
        "tmem",
        (_WRITE,),
        (
            Access("threads", False),
            Access("threads", True),
            Access("umma", False),
            Access("umma", True),
        ),
        "tmem_store",
        "tcgen05.wait::st establishes store completion before the handoff",
        _TCGEN05_SYNC,
    ),
    Rule(
        "tmem",
        (_READ,),
        (Access("threads", True), Access("umma", True)),
        "tmem_load",
        "tcgen05.wait::ld establishes load completion before an overwrite; an explicit wait "
        "in the role is not visible to this trace",
        _TCGEN05_SYNC,
        visible=False,
    ),
)


def default_access(mech, side: str) -> Access:
    """The mechanism's access to channel SMEM, not an MMA's TMEM accumulators."""
    # channels imports derived_fences, so mechanism types must be imported at call time.
    from attn_gym.warpdrive.channels import Tma, Umma

    if isinstance(mech, Tma):
        return Access("tma", True)
    if mech is Umma:
        return Access("umma", False)
    return _WRITE if side == "producer" else _READ


def matching_rules(storage: str, mine: Access, theirs: Access) -> list[Rule]:
    return [r for r in RULES if r.storage == storage and mine in r.mine and theirs in r.theirs]


def required(storage: str, mine, theirs) -> tuple[str, ...]:
    """The fence kinds, in ``FENCE_KINDS`` order, that ``mine`` accesses followed by
    ``theirs`` need on the signal between them."""
    kinds = {r.kind for m in mine for t in theirs for r in matching_rules(storage, m, t)}
    return tuple(k for k in FENCE_KINDS if k in kinds)


def derived_fences(chan) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(commit_fence, release_fence)`` derived from ``chan``'s mechanisms and storage."""
    producer = [default_access(chan.producer, "producer")]
    if producer[0].agent != "tma" and chan.payload is None and not chan.buffers:
        return (), ()
    consumers = [default_access(m, "consumer") for m in chan.mechanisms("consumer")]
    commit = () if chan.external == "producer" else required("smem", producer, consumers)
    release = () if chan.external == "consumer" else required("smem", consumers, producer)
    return commit, release


# Trace-time check of explicit overrides

COVERED, UNVERIFIED, MISSING = "covered", "unverified", "missing fence"
FenceStatus = Literal["covered", "unverified", "missing fence"]


@dataclass
class FenceResult:
    channel: str
    key: str  # "commit_fence" | "release_fence"
    fences: tuple[str, ...]
    missing: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)

    @property
    def status(self) -> FenceStatus:
        return MISSING if self.missing else UNVERIFIED if self.unverified else COVERED

    def describe(self) -> str:
        head = f"  {self.channel} {self.key} {'+'.join(self.fences) or 'none'}  {self.status}"
        lines = [head, *(f"      missing {m}" for m in dict.fromkeys(self.missing))]
        lines += [f"      unverified: {r}" for r in dict.fromkeys(self.unverified)]
        return "\n".join(lines)


@dataclass
class FenceReport:
    results: list[FenceResult]

    def describe(self) -> str:
        if not self.results:
            return ""
        head = "Fence check (explicit fences against annotated accesses)"
        return "\n".join([head, *(r.describe() for r in self.results)])

    @property
    def flagged(self) -> list[FenceResult]:
        return [r for r in self.results if r.missing]


def _seen(kernel, roles: list[alias.RoleEvents], chan, producer: bool) -> list[alias.Access]:
    """Stage views belong to their own handle; TMEM accesses belong to every open handle."""
    out = []
    for role in roles:
        for i, ev in enumerate(role.events):
            if ev.kind != "access" or ev.fence_opted_out:
                continue
            held = (ev.data["handle"],) if ev.data.get("handle") is not None else ev.open
            if any(h.chan is chan and h.producer == producer for h in held):
                out.append(alias.Access.from_event(kernel, role, i))
    return out


def _modes(access: alias.Access) -> list[Access]:
    return [
        Access(access.agent, w) for w, used in ((False, access.read), (True, access.write)) if used
    ]


def _event_fences(ev: Event) -> tuple[str, ...]:
    if ev.kind == "fence":
        return (ev.data["kind"],)
    if ev.kind == "close" and ev.data["signal"]:
        rec = ev.data["handle"]
        return rec.chan.fences("producer" if rec.producer else "consumer", rec.role)
    return ()


def _other_handoff(s: alias.Access, t: alias.Access | None, chan, kind: str) -> bool:
    if s.handle is not None or t is None:
        return False
    for h in s.handles:
        if h.chan is chan or not any(
            th.chan is h.chan and th.producer != h.producer for th in t.handles
        ):
            continue
        if any(
            ev.data.get("handle") is h and kind in _event_fences(ev)
            for ev in s.role.events[s.index + 1 :]
        ):
            return True
    return False


def _counterparts(s: alias.Access, chan, other: str, theirs, pairs, res: FenceResult):
    matches = []
    for ri, _ in chan.consumers if other == "consumer" else chan.producers:
        if ri.raw:
            continue
        default = default_access(chan.mechanism(other, ri.role), other)
        same_storage = [
            t
            for t in theirs
            if t.role.name == ri.name
            and (
                t.storage == s.storage
                or (s.storage[0], frozenset((s.storage[1], t.storage[1]))) in pairs
                and s.storage[0] == t.storage[0]
            )
        ]
        matches += [(m, t) for t in same_storage if alias._conflicts(s, t) for m in _modes(t)]
        implied = (
            s.storage[0] == "smem"
            and default.agent != "threads"
            and any(s.storage[1] == b.name for b in chan.buffers.values())
        )
        if implied:
            matches.append((default, None))
        elif not same_storage and not (s.storage[0] == "tmem" and default.agent == "tma"):
            res.unverified.append(f"{s.describe()}: no annotated {other} access by {ri.name}")
    return matches


def _check_handoff(s: alias.Access, rec, other: str, matches, res: FenceResult):
    fenced, uncertain = set(), set()
    where = s.role.events[s.index].where
    for ev in s.role.events[s.index + 1 :]:
        # Nested regions and another span's last-only signal need not execute here.
        covered = ev.data.get("handle") is rec or (
            where[: len(ev.where)] == ev.where and ev.data.get("once") != "last"
        )
        (fenced if covered else uncertain).update(_event_fences(ev))
        if ev.kind != "close" or ev.data["handle"] is not rec:
            continue
        break
    else:
        return
    if ev.data["retire"]:
        return
    managed = ev.data["signal"]
    for m in _modes(s):
        for t, access in matches:
            for rule in matching_rules(s.storage[0], m, t):
                if rule.kind in fenced:
                    continue
                text = (
                    f"{rule.kind}: {s.describe()}, then a {other} {t.agent} "
                    f"{'write' if t.write else 'read'} @ {ev.site}: {rule.reason}"
                )
                if rule.kind in uncertain:
                    text += "; conditional fence coverage is not established"
                alternative = _other_handoff(s, access, rec.chan, rule.kind)
                if alternative:
                    text += "; another open channel may order this access"
                known = rule.visible and managed and rule.kind not in uncertain and not alternative
                (res.missing if known else res.unverified).append(text)


def _check_side(kernel, chan, side: str, roles: list[alias.RoleEvents], pairs) -> FenceResult:
    key = SIDE_FENCES[side]
    producer = side == "producer"
    other = "consumer" if producer else "producer"
    res = FenceResult(chan.name, key, getattr(chan, key))
    if chan.external == other:
        res.unverified.append(f"external {other}: its accesses are not visible")
        return res
    signalling = {
        ri.name
        for ri, _ in (chan.producers if producer else chan.consumers)
        if default_access(chan.mechanism(side, ri.role), side).agent == "threads"
    }
    if not signalling:
        res.unverified.append("no signalling thread role")
    if chan.payload is not None:
        res.unverified.append("payload accesses are not recorded")
    mine = [s for s in _seen(kernel, roles, chan, producer) if s.role.name in signalling]
    theirs = _seen(kernel, roles, chan, not producer)
    for name in signalling:
        if not any(s.role.name == name for s in mine):
            res.unverified.append(f"no recorded access by {name} while holding {chan.name}")
    for s in mine:
        if s.agent != "threads" or s.role.raw:
            continue
        matches = _counterparts(s, chan, other, theirs, pairs, res)
        held = (s.handle,) if s.handle is not None else s.handles
        for rec in held:
            if rec.chan is not chan or rec.producer != producer:
                continue
            _check_handoff(s, rec, other, matches, res)
    involved = signalling | {ri.name for ri, _ in (chan.consumers if producer else chan.producers)}
    for role in roles:
        if role.name not in involved:
            continue
        if role.raw:
            res.unverified.append(f"raw role {role.name}: its accesses are not visible")
        for ev in role.events:
            if ev.kind == "raw" and any(h.chan is chan for h in ev.open):
                res.unverified.append(f"raw access by {role.name} @ {ev.site}")
    return res


def check(kernel, roles: list[alias.RoleEvents]) -> FenceReport:
    """Check explicit overrides without changing the fences the channel emits."""
    pairs = {(kind, frozenset(names)) for kind, names in alias.declared_pairs(kernel)}
    return FenceReport(
        [
            _check_side(kernel, chan, side, roles, pairs)
            for chan in kernel._ws_chans.values()
            if not chan.internal
            for side, key in SIDE_FENCES.items()
            if getattr(chan, f"{key}_overridden") and chan.external != side
        ]
    )


def report(rep: FenceReport, mode: str, kernel_name: str):
    """Warn about missing fences, or raise on them in strict mode."""
    flagged = rep.flagged
    if not flagged:
        return
    if mode == "strict":
        raise ProtocolError(f"warpdrive fence check (strict):\n{rep.describe()}")
    names = ", ".join(f"{r.channel} {r.key}" for r in flagged)
    warnings.warn(
        f"warpdrive: fence check flagged {names} in {kernel_name} (no error raised; see "
        f"describe()): missing {flagged[0].missing[0]}",
        stacklevel=3,
    )
