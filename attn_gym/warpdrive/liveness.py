"""Check managed channel protocols for deadlock without reading runtime control flow.

Trace extraction supplies Python records. The graph orders each role's operations, matches
channel epochs, and joins channel-bound named-barrier arrivals. A straight-line protocol is
verified or an error. Runtime loops are repeated a few fixed times, which proves nothing: no
cycle is "no deadlock found", a cycle is a warning. Runtime branches, raw ordering and
cross-CTA protocols are outside this model; unsupported protocols and analysis failures remain
unverified rather than failing compilation.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field

from attn_gym.warpdrive.protocol import ProtocolError

MODES = ("strict", "warn", "off")
MAX_NODES = 10_000
MAX_EDGES = 30_000
OVERRIDE = (
    'Override with unchecked("liveness", reason=...) around the ops, or liveness_check="warn" '
    'or "off" on the kernel.'
)


@dataclass
class Liveness:
    verdict: str
    detail: str = ""
    findings: list[str] = field(default_factory=list)

    def line(self) -> str:
        detail = "see the reported finding" if self.findings else self.detail
        return f"liveness: {self.verdict}" + (f" ({detail})" if detail else "")


@dataclass
class Channel:
    """Static participants, indexed by consumer (0) or producer (1); slots may be unbound."""

    stages: int
    bound: tuple[tuple[tuple[str, int | None], ...], ...]
    turns: tuple[int, int] = (1, 1)
    fixed: bool = False
    named: bool = False
    tma: bool = False

    def owners(self, producer: bool, slot: int | None, epoch: int) -> set[str]:
        side = self.bound[producer]
        if self.turns[producer] > 1:
            side = (side[epoch % self.turns[producer]],)
        return {role for role, bound in side if bound is None or bound == slot}


@dataclass
class Op:
    role: str
    kind: str
    channel: str
    slot: int | None
    epoch: int
    site: str

    def label(self) -> str:
        verb = "commit(wait=True)" if self.kind == "rendezvous" else f"{self.kind}()"
        return f"{self.role}: {self.channel}.{verb} of epoch {self.epoch} at {self.site}"


class Unsupported(Exception):
    pass


def report(result: Liveness | None, mode: str, kernel_name: str) -> None:
    if result is None or not result.findings or mode == "off":
        return
    message = f"{kernel_name}: {result.findings[0]} {OVERRIDE}"
    if mode == "strict" and result.verdict == "error":
        raise ProtocolError(message)
    warnings.warn(f"warpdrive: {message}", stacklevel=3)


def check(kernel, roles: list) -> Liveness:
    try:
        channels = _channels(kernel)
        barrier_users = defaultdict(set)
        for role in roles:
            for ev in role.events:
                if ev.kind.startswith("barrier"):
                    barrier_users[id(ev.data.get("barrier"))].add(role.name)
        events = {role.name: _events(role, channels, barrier_users) for role in roles}
        ops = [ev for role_events in events.values() for ev in role_events]
        looped = frozenset(ev.data.get("chan") for ev in ops if ev.where)
        deepest = max((channels[name].stages for name in looped if name), default=0)
        trips = sorted({1, 2, deepest + 1}) if looped else [1]
        # What the model takes on trust. With any of it a cycle is a warning, never a raise.
        assumed = []
        if looped:
            counts = f"{', '.join(map(str, trips[:-1]))} and {trips[-1]}"
            assumed.append(f"{counts} loop trips, matched counts")
        if any(channels[ev.data["chan"]].tma for ev in ops if ev.kind == "op"):
            assumed.append("TMA copy at barrier read")
        for count in trips:
            result = _analyze(channels, _ops(channels, events, count), looped)
            if result.findings and assumed:
                result.verdict = "possible deadlock"
                if looped:
                    result.findings[0] += f" Seen with every runtime loop running {count} times."
            if result.findings:
                return result
        if not assumed:
            return Liveness("verified")
        return Liveness("no deadlock found", "; ".join(assumed))
    except Exception as error:  # noqa: BLE001 - analysis must never fail compilation
        return _unverified(error)


def _unverified(error: Exception) -> Liveness:
    why = str(error) if isinstance(error, Unsupported) else f"analysis failed: {error!r}"
    return Liveness("unverified", why[:300])


def _channels(kernel) -> dict[str, Channel]:
    from attn_gym.warpdrive.channels import Sharing, Threads, Tma, Umma, Warps

    def modeled(mechanism) -> bool:
        # A TMA copy or a tcgen05 commit completes on its own once issued: nothing another role
        # does can hold it back. The commit or release issues a tcgen05 commit. A TMA commit
        # issues nothing, so the fill is placed where the role reads the stage's barrier for
        # its copy (the "tma pointer" op).
        return mechanism in (Threads, Warps, Umma) or isinstance(mechanism, Tma)

    if kernel.cluster.size != 1:
        raise Unsupported("cross-CTA placement")
    if kernel._ws_raw_bars or kernel._ws_raw_barrier_access:
        raise Unsupported(
            f"raw barriers: {kernel._ws_raw_barrier_access or list(kernel._ws_raw_bars)}"
        )
    channels = {}
    for name, ch in kernel._ws_chans.items():
        sides = (("consumer", ch.consumers), ("producer", ch.producers))
        if ch.external is not None:
            raise Unsupported(f"external side of {name}")
        if ch.override:
            raise Unsupported(f"overridden counts or masks on {name}")
        if ch.internal:
            raise Unsupported(f"scheduler channel {name} is not modeled")
        report = kernel._ws_tma.get(name)
        if isinstance(ch.producer, Tma) and (report is None or report.verdict != "verified"):
            raise Unsupported(f"TMA copies of {name} are not verified")
        if not modeled(ch.producer) or not all(
            modeled(ch.mechanism("consumer", ri.role)) for ri, _ in ch.consumers
        ):
            raise Unsupported(f"signal mechanism of {name} is not modeled")
        if ch.turnstile or any(
            ch.sharing(side) is Sharing.EXCLUSIVE and not ch.fixed for side, _ in sides
        ):
            raise Unsupported(f"unknown epoch correspondence on {name}")
        channels[name] = Channel(
            ch.n,
            tuple(tuple((ri.name, slot) for ri, slot in bound) for _, bound in sides),
            tuple(ch.turns(side) for side, _ in sides),
            ch.fixed,
            ch.named,
            isinstance(ch.producer, Tma),
        )
    return channels


KINDS = ("acquire", "wait", "commit", "release", "peek", "tail", "rendezvous", "fill")


def _events(role, channels: dict[str, Channel], barrier_users: dict) -> list:
    """The role's channel ops in trace order, or Unsupported for anything outside the model."""
    if role.raw:
        raise Unsupported(f"raw role {role.name}")
    events, copied = [], set()
    for ev in role.events:
        if ev.kind not in ("op", "barrier", "barrier_arrive", "barrier_native"):
            continue
        site = f"{role.name} at {ev.site}"
        if ev.opted_out:
            raise Unsupported(f'unchecked("liveness"): {site}')
        if any(region != "loop" for _, region in ev.where):
            raise Unsupported(f"runtime branch or unknown region: {site}")
        match ev.kind:
            case "barrier":
                barrier = ev.data.get("barrier")
                if (
                    barrier is None
                    or barrier.threads != 32 * len(role.warps)
                    or barrier._native_outside_roles
                    or barrier_users[id(barrier)] != {role.name}
                ):
                    raise Unsupported(f"named barrier shared between roles: {site}")
                # Every thread of this role and no one else: it waits on no other role. Kept
                # so a sync in a loop counts as looped.
            case "barrier_arrive" | "barrier_native":
                raise Unsupported(f"named barrier shared between roles: {site}")
            case "op":
                d = ev.data
                name, kind = d["chan"], d["op"]
                if name not in channels:
                    raise Unsupported(f"turnstile or unmanaged channel {name}: {site}")
                ch = channels[name]
                stage = (name, d["slot"], d["order"])
                match kind:
                    case "try_wait" | "try_acquire":
                        continue
                    case _ if d.get("span"):
                        raise Unsupported(f"{name}.{kind} or span: {site}")
                    case "tma pointer" if ch.tma:
                        copied.add(stage)
                    case "tail" if ch.named or ch.turns[True] > 1:
                        raise Unsupported(f"named or rotating tail: {site}")
                    case "commit" if ch.tma and stage not in copied:
                        raise Unsupported(
                            f"TMA stage committed without reading its barrier: {site}"
                        )
                    case (
                        "acquire" | "wait" | "commit" | "release" | "peek" | "tail" | "rendezvous"
                    ):
                        pass
                    case _:
                        raise Unsupported(f"{name}.{kind} or span: {site}")
        events.append(ev)
    return events


def _unroll(events: list, trips: int, depth: int = 0):
    """Yield the events with every runtime loop body repeated ``trips`` times."""
    i = 0
    while i < len(events):
        ev = events[i]
        if len(ev.where) <= depth:
            yield ev
            i += 1
            continue
        key, j = ev.where[depth][0], i
        inside = [len(e.where) > depth and e.where[depth][0] == key for e in events]
        while j < len(events) and inside[j]:
            j += 1
        if any(inside[j:]):
            raise Unsupported(f"loop body traced in two pieces at {ev.site}")
        for _ in range(trips):
            yield from _unroll(events[i:j], trips, depth + 1)
        i = j


def _ops(channels: dict[str, Channel], events: dict[str, list], trips: int) -> list[Op]:
    ops, visited = [], 0
    for role, role_events in events.items():
        next_epoch, taken = {}, {}
        # Named-barrier syncs are dropped before unrolling. An event may still add no node (a
        # tail before any acquire), so the budget counts events as well as nodes.
        for ev in _unroll([ev for ev in role_events if ev.kind == "op"], trips):
            visited += 1
            if max(visited, len(ops)) >= MAX_NODES:
                raise Unsupported(f"node budget exceeded ({MAX_NODES})")
            d = ev.data
            name, kind = d["chan"], d["op"]
            ch = channels[name]
            slot = d["slot"] if ch.fixed else None
            producer = kind in ("acquire", "commit", "tail", "rendezvous", "tma pointer")
            key = (name, slot, producer)
            if key not in next_epoch:
                names = [r for r, _ in ch.bound[producer]]
                next_epoch[key] = names.index(role) if ch.turns[producer] > 1 else 0
            epoch = next_epoch[key]
            match kind:
                case "tail":  # waits every stage the end still has out
                    for (n, s, p), end in list(next_epoch.items()):
                        if n == name and p:
                            for e in range(max(0, end - (1 if ch.fixed else ch.stages)), end):
                                ops.append(Op(role, kind, name, s, e, ev.site))
                    continue
                case "acquire" | "wait":
                    # A handle closes in the region that opened it, so within one trip.
                    taken[(*key, d["order"])] = epoch
                    next_epoch[key] += 1 if ch.fixed else ch.turns[producer]
                case "peek":
                    pass  # looks at the next epoch without taking it
                case "tma pointer":
                    kind, epoch = "fill", taken[(*key, d["order"])]
                case _:  # commit, release, rendezvous: the epoch its handle took
                    epoch = taken[(*key, d["order"])]
            ops.append(Op(role, kind, name, slot, epoch, ev.site))
    return ops


def analyze(
    channels: dict[str, Channel], ops: list[Op], looped: frozenset = frozenset()
) -> Liveness:
    """Check normalized operations; unsupported input and internal failures stay unverified.

    ``looped`` names the channels with ops in a runtime loop: their counts come from a chosen
    number of trips, so a mismatch there is unverified, not an imbalance."""
    try:
        return _analyze(channels, ops, looped)
    except Exception as error:  # noqa: BLE001 - analysis must never fail compilation
        return _unverified(error)


def _analyze(channels: dict[str, Channel], ops: list[Op], looped: frozenset) -> Liveness:
    preds, previous = [], {}
    groups = defaultdict(lambda: defaultdict(list))
    entries = {}
    edges = 0

    def node() -> int:
        if len(preds) >= MAX_NODES:
            raise Unsupported(f"node budget exceeded ({MAX_NODES})")
        preds.append([])
        return len(preds) - 1

    def edge(before: int, after: int) -> None:
        nonlocal edges
        if edges >= MAX_EDGES:
            raise Unsupported(f"edge budget exceeded ({MAX_EDGES})")
        preds[after].append(before)
        edges += 1

    def join(sources: list[int], targets: list[int]) -> None:
        if not sources or not targets:
            return
        # A shared completion node keeps fan-in/fan-out linear in the participant count.
        hub = node()
        for src in sources:
            edge(src, hub)
        for dst in targets:
            edge(hub, dst)

    for op in ops:
        i = node()
        entries[i] = previous.get(op.role)
        if op.role in previous:
            edge(previous[op.role], i)
        previous[op.role] = i
        if op.kind not in KINDS:
            raise Unsupported(f"unknown operation: {op.label()}")
        groups[(op.channel, op.slot, op.epoch)][op.kind].append(i)
    for (name, slot, epoch), kinds in list(groups.items()):
        ch = channels[name]
        lap = 1 if ch.fixed else ch.stages
        example = ops[next(iter(kinds.values()))[0]]
        if epoch > 0 and (name, slot, epoch - 1) not in groups:
            raise Unsupported(f"non-contiguous epochs: {example.label()}")
        for producer, take, signal in ((True, "acquire", "commit"), (False, "wait", "release")):
            expected = ch.owners(producer, slot, epoch)
            for kind in (take, signal):
                actual = [ops[i].role for i in kinds[kind]]
                if len(actual) == len(expected) and set(actual) == expected:
                    continue
                if name in looped:
                    raise Unsupported(
                        f"loop counts on {name} do not line up: epoch {epoch} has {kind} by "
                        f"{actual}, expected {sorted(expected)}; {example.label()}"
                    )
                return Liveness(
                    "error",
                    findings=[
                        (
                            f"unbalanced epochs: {name} slot {slot}, epoch {epoch}: {kind} by "
                            f"{actual}, expected {sorted(expected)}; {example.label()}."
                        )
                    ],
                )
        fills = kinds["fill"] if ch.tma else kinds["commit"]
        takes = kinds["wait"] + kinds["peek"] + kinds["rendezvous"]
        if ch.named:
            arrivals = [entries[i] for i in kinds["wait"] if entries[i] is not None]
            join(fills + arrivals, takes)
        else:
            join(fills, takes)
        reuse = groups.get((name, slot, epoch + lap), {}).get("acquire", [])
        join(kinds["release"], reuse + kinds["tail"])
        if ch.named:
            acquires = kinds["acquire"]
            arrivals = [entries[i] for i in acquires if entries[i] is not None]
            # Initial consumer EMPTY arrivals precede role bodies; epoch zero still needs every producer.
            join(arrivals, acquires)
    color = [0] * len(preds)
    for start in range(len(preds)):
        if color[start]:
            continue
        color[start] = 1
        path, stack = [start], [iter(preds[start])]
        while stack:
            before = next(stack[-1], None)
            if before is None:
                color[path.pop()] = 2
                stack.pop()
            elif color[before] == 1:
                cycle = [ops[i].label() for i in path[path.index(before) :] if i < len(ops)]
                return Liveness("error", findings=["deadlock: " + " waits for ".join(cycle) + "."])
            elif color[before] == 0:
                color[before] = 1
                path.append(before)
                stack.append(iter(preds[before]))
    return Liveness("verified")
