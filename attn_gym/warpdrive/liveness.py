"""Check finite, straight-line managed protocols without interpreting runtime control flow.

Trace extraction supplies Python records. The graph orders each role's operations, matches
channel epochs, and joins channel-bound named-barrier arrivals. Async completion, raw ordering,
and cross-CTA protocols are outside this model; unsupported protocols and analysis failures
remain unverified rather than failing compilation.
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
    if mode == "strict":
        raise ProtocolError(message)
    warnings.warn(f"warpdrive: {message}", stacklevel=3)


def check(kernel, roles: list) -> Liveness:
    try:
        channels, ops = _extract(kernel, roles)
        return analyze(channels, ops)
    except Exception as error:  # noqa: BLE001 - analysis must never fail compilation
        return _unverified(error)


def _unverified(error: Exception) -> Liveness:
    why = str(error) if isinstance(error, Unsupported) else f"analysis failed: {error!r}"
    return Liveness("unverified", why[:300])


def _extract(kernel, roles: list) -> tuple[dict[str, Channel], list[Op]]:
    from attn_gym.warpdrive.channels import Sharing, Threads, Warps

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
        if ch.producer not in (Threads, Warps) or any(
            ch.mechanism("consumer", ri.role) not in (Threads, Warps) for ri, _ in ch.consumers
        ):
            raise Unsupported(f"TMA/async completion on {name} is not modeled")
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
        )
    ops = []
    for role in roles:
        if role.raw:
            raise Unsupported(f"raw role {role.name}")
        next_epoch, taken = {}, {}
        for ev in role.events:
            if ev.kind not in ("op", "barrier", "barrier_arrive", "barrier_native"):
                continue
            site = f"{role.name} at {ev.site}"
            if ev.opted_out:
                raise Unsupported(f'unchecked("liveness"): {site}')
            if ev.where:
                raise Unsupported(f"runtime loop/branch or unknown region: {site}")
            d = ev.data
            if ev.kind.startswith("barrier"):
                raise Unsupported(f"standalone named barrier: {site}")
            name = d["chan"]
            if name not in channels:
                raise Unsupported(f"turnstile or unmanaged channel {name}: {site}")
            ch = channels[name]
            slot = d["slot"] if ch.fixed else None
            kind = d["op"]
            if kind in ("try_wait", "try_acquire"):
                continue
            if kind not in (
                "acquire",
                "wait",
                "commit",
                "release",
                "peek",
                "tail",
                "rendezvous",
            ) or d.get("span"):
                raise Unsupported(f"{name}.{kind} or span: {site}")
            producer = kind in ("acquire", "commit", "tail", "rendezvous")
            key = (name, slot, producer)
            if key not in next_epoch:
                names = [r for r, _ in ch.bound[producer]]
                next_epoch[key] = names.index(role.name) if ch.turns[producer] > 1 else 0
            epoch = next_epoch[key]
            if kind == "tail":
                if ch.named or ch.turns[True] > 1:
                    raise Unsupported(f"named or rotating tail: {site}")
                for (n, s, p), end in list(next_epoch.items()):
                    if n == name and p:
                        for e in range(max(0, end - (1 if ch.fixed else ch.stages)), end):
                            ops.append(Op(role.name, kind, name, s, e, ev.site))
                continue
            if kind in ("commit", "release", "rendezvous"):
                epoch = taken[(name, slot, producer, d["order"])]
            if kind in ("acquire", "wait"):
                taken[(name, slot, producer, d["order"])] = epoch
                next_epoch[key] += 1 if ch.fixed else ch.turns[producer]
            ops.append(Op(role.name, kind, name, slot, epoch, ev.site))
    return channels, ops


def analyze(channels: dict[str, Channel], ops: list[Op]) -> Liveness:
    """Check normalized operations; unsupported input and internal failures stay unverified."""
    try:
        return _analyze(channels, ops)
    except Exception as error:  # noqa: BLE001 - analysis must never fail compilation
        return _unverified(error)


def _analyze(channels: dict[str, Channel], ops: list[Op]) -> Liveness:
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
        if op.kind not in ("acquire", "wait", "commit", "release", "peek", "tail", "rendezvous"):
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
                if len(actual) != len(expected) or set(actual) != expected:
                    return Liveness(
                        "error",
                        findings=[
                            (
                                f"unbalanced epochs: {name} slot {slot}, epoch {epoch}: {kind} by "
                                f"{actual}, expected {sorted(expected)}; {example.label()}."
                            )
                        ],
                    )
        fills = kinds["commit"]
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
