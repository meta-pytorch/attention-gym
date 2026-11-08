"""Random valid warpdrive kernels, emitted as Python source.

Why source files rather than one constexpr-driven template class: a role's channel ends,
cursor, seat and TMEM view are bound by *parameter name and annotation* and its warps, register
budget and placement are ``@role(...)`` decorator arguments, all fixed when the class body
runs. The number of roles, their signatures, and which channels each takes are exactly what the
fuzzer varies, and CuTeDSL's preprocessor needs real source for every ``@cute.jit`` body. So
each ``KernelSpec`` is rendered to a module (deterministic name from the spec hash), imported
from a file, and compiled like a hand-written kernel.

Validity (the contract's "valid managed program", section 1) is by construction:

* Channels go forward in a random topological order of the roles (a DAG), every participant of
  a channel performs the same per-iteration handle program on it, and every handle a role opens
  in an iteration is closed in that iteration. A ring's per-iteration multiplicity never exceeds
  its depth. Induction on (iteration, topological position) then shows the abstract
  bounded-buffer program is live under any intra-iteration open/close interleaving: iteration
  ``i`` of a role waits only on upstream publications of iteration ``i`` and on downstream
  releases of iterations ``<= i - 1``.
* ``Threads`` sides are executed by every thread of every bound instance; ``Tma``/``Umma`` sides
  belong to one single-warp, single-instance role (leader-placed in a cluster of MMA pairs).
* Payloads only ride on CTA-local channels with ``Threads`` or ``Tma`` producers and ``Threads``
  consumers; ``Umma`` and cross-CTA channels are tokens (counts only). TMA stages read by
  threads release with ``release_fence="smem_async"``, declared or (``implicit_release_fence``)
  derived by the layer.
* In a ``Cluster`` roles run in every CTA, the pair leaders or rank 0; cross-CTA channels and
  multicast TMA stay within the layer's derivable placements (R11), so each CTA runs the same
  per-iteration program and the argument above holds across CTAs.
* With a ``WorkStream`` every role drains it (``while work.valid: ...; work.advance()``) and
  runs ``rounds(item)`` iterations per item, computed identically by every role.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import random
from dataclasses import asdict, dataclass, field

MAX_WARPS = 32
WARPGROUP = 4
REGS_PER_SM = 65536
MIN_REGS = 24
TMA_COLS = 32  # Int32 columns per TMA tile row: 128-byte rows
SMS = 148  # nominal SM count for generated caps; the runner never assumes it


Key = tuple[str, str, int]  # (channel, "p" | "c", ring handle k / slot j / BOUND)
BOUND = -1


@dataclass(frozen=True)
class Geometry:
    """Cluster geometry from the contract (R11), independent of ``warpdrive.cluster``: rank
    ``r = v + V * (mi + (m // V) * ni)`` with ``V = 2`` CTAs per MMA pair (1 without pairs)."""

    m: int
    n: int
    pair: bool

    @property
    def size(self) -> int:
        return self.m * self.n

    @property
    def v(self) -> int:
        return 2 if self.pair else 1

    def coords(self, rank: int) -> tuple[int, int, int]:
        pairs_m = self.m // self.v
        return rank % self.v, rank // self.v % pairs_m, rank // self.v // pairs_m

    def ctas(self, placement: str) -> tuple[int, ...]:
        if placement == "rank0":
            return (0,)
        if placement == "leader":
            return tuple(r for r in range(self.size) if r % self.v == 0)
        return tuple(range(self.size))

    def group(self, rank: int) -> tuple[int, ...]:
        return tuple(r for r in range(self.size) if r // self.v == rank // self.v)

    def reach(self, rank: int, modes: tuple[str, ...]) -> tuple[int, ...]:
        """``rank`` and every CTA differing from it only along one of ``modes``."""
        v, mi, ni = self.coords(rank)
        out = {rank}
        for r in range(self.size):
            rv, rm, rn = self.coords(r)
            if rv == v and (("m" in modes and rn == ni) or ("n" in modes and rm == mi)):
                out.add(r)
        return tuple(sorted(out))

    def emit(self) -> str | None:
        if self.size == 1:
            return None
        return f"Cluster({self.m}, {self.n}, pair={self.pair})"

    def key(self) -> str:
        return f"m{self.m}n{self.n}" + ("_pairs" if self.pair else "")


def parse_cluster(text: str) -> tuple[int, int, bool]:
    """``"2x2p"`` -> ``(2, 2, True)``; a trailing ``p`` declares MMA pairs."""
    pair = text.endswith("p")
    m, n = (int(x) for x in text.removesuffix("p").split("x"))
    return m, n, pair


@dataclass(frozen=True)
class RoleSpec:
    name: str
    warps: tuple[int, ...]  # first warp of each instance
    width: int  # warps per instance
    regs: int | None
    cta: str  # "all" | "leader" | "rank0"
    jitter: int  # spin scale in cycles (0: none)
    tmem: bool
    # Per-iteration handle program: ("o" | "c", channel, side, key) in execution order.
    events: tuple[tuple[str, str, str, int], ...] = ()
    with_handles: tuple[Key, ...] = ()  # handles emitted as nested ``with`` blocks
    # "": a generated managed role; "raw": role(raw=True) driving a Barriers ring;
    # "mma_load" / "mma" / "mma_epi": the TMA -> tcgen05 MMA -> TMEM epilogue subgraph.
    kind: str = ""

    @property
    def raw(self) -> bool:
        return self.kind == "raw"

    @property
    def instances(self) -> int:
        return len(self.warps)

    @property
    def threads(self) -> int:
        return 32 * self.width

    def warp_ranges(self) -> list[range]:
        return [range(w, w + self.width) for w in self.warps]


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    kind: str  # "ring" | "slots"
    n: int
    producer: str  # "threads" | "umma" | "tma"
    consumer: str  # "threads" | "umma"
    producers: tuple[tuple[str, bool], ...]  # (role, bound to Instance)
    consumers: tuple[tuple[str, bool], ...]
    mult: int = 1  # ring: handles per iteration per participant
    extra_mult: int = 0  # ring: handles in the final extra round
    extra_slots: tuple[int, ...] = ()  # slots: slots used in the final extra round
    payload: bool = False
    tma_rows: int = 0
    tma_pair: bool = False  # pair-scoped: both CTAs copy, CtaGroup.TWO, leader-only consumers
    # Tma(multicast=mode): each CTA copies its part of the tile into every CTA along the mode.
    multicast: str = ""
    # Tma(fills=...): the producer ("pair": on the pair leaders, "cluster": on rank 0) copies the
    # tile into every CTA it fills; fill_two: the bytes complete on the pair leaders (CtaGroup.TWO).
    fills: str = ""
    fill_two: bool = False
    commit_fence: str | None = None
    release_fence: str | None = None
    # Tma -> Threads: leave release_fence to the layer's derived default instead of declaring it.
    implicit_release_fence: bool = False
    remote_scope: str | None = None
    salt: int = 0
    # "raw" kind: a Barriers ring between raw roles; "auto" (Barriers(counts=...)) or "manual"
    # (counts=None, initialized in the kernel prologue).
    raw_init: str = ""
    mma: str = ""  # "ab" (Tma -> Umma A/B stages) | "acc" (Umma -> Threads accumulator)
    # Sides ("p", "c", "pc") that close with signal=False and arrive once per warp themselves,
    # under an override of that side's counts given as an int, a per-slot list or {rank: list}.
    manual: str = ""
    override_form: str = ""
    # Sides ("p", "c", "pc") of Threads mechanism declared ``Warps`` instead: every thread still
    # closes the handle, the layer arrives once per warp.
    warps: str = ""
    generic: bool = False  # declared as Channel(...) rather than Ring(...)
    # unchecked(...) site: "cross_region" (close inside a dynamic branch), "unclosed" (open
    # with the opt-out, signal through the raw barrier, never close) or "tma" (TMA barrier).
    unchecked: str = ""
    unchecked_side: str = ""
    # Payload channels: "buffer" (an SmemPlan buffer instead of payload=) or "overlap" (the
    # producer writes buffer Q, declared overlaps=P, the consumer reads P); TMA channels:
    # "buffer" (an SmemPlan buffer with a StageView of the TMA partition).
    storage: str = ""
    alias_opt_out: bool = False  # the consumer's read annotation sits in unchecked("alias")
    # How handles open: "" a plain acquire/wait; "try" probe each side first (try_acquire/
    # try_wait) and pass the token; "peek" the consumer previews the epoch before its wait.
    wait: str = ""

    @property
    def tma_tiles(self) -> int:
        return self.n + 1  # coprime with the stage count: stale stages read the wrong tile

    @property
    def tiles_checked(self) -> bool:
        """Threads consumers read and verify TMA tiles (an Umma consumer only releases)."""
        return self.producer == "tma" and self.consumer == "threads"

    @property
    def tma_bytes(self) -> int:
        return self.tma_rows * TMA_COLS * 4


@dataclass(frozen=True)
class StreamSpec:
    schedule: str  # "oneshot" | "static" | "atomic" | "clc"
    stages: int
    along: str
    swizzle: int
    reverse: tuple[str, ...]
    skip: tuple[int, int, int, int] | None  # skip item when (a m + b n + c l) % k == 0
    ready: bool
    extents: tuple[int, int, int]
    max_ctas: int
    scheduler_warp: int | None
    rounds: tuple[int, int, int, int, int]  # (base, spread, a, b, c)
    launch_1d: bool = False  # OneShot: a hand-built (total * cs, 1, 1) grid (R13)
    record: int = 0  # WorkStream(record=k): the timeline must match the item logs (R16)
    # "raster" (Raster / SkipMod); "table": a WorkMap without decodes_any_index reading a
    # permuted item table and counting decodes outside [0, total); "packed": PackedWork over a
    # table of the items ``skip`` keeps, ``pad`` spare rows, under PdlWait.
    mapping: str = "raster"
    pad: int = 0
    pdl: bool = False  # packed: a producer kernel writes the table, consumer launched with PDL
    order_seed: int = 0  # table/packed row order

    @property
    def raster_total(self) -> int:
        em, en, el = self.extents
        return em * en * el if min(self.extents) > 0 else 0  # R17: any extent <= 0 is empty

    @property
    def total(self) -> int:
        """Item indices ``[0, total)`` the stream visits (packed: the table capacity)."""
        if self.mapping == "packed":
            return len(packed_rows(self)) + self.pad
        return self.raster_total

    @property
    def plain_raster(self) -> bool:
        return (
            self.mapping == "raster"
            and self.along == "m"
            and self.swizzle == 1
            and not self.reverse
            and self.skip is None
        )

    @property
    def broadcast(self) -> bool:
        return self.schedule in BROADCAST

    @property
    def capped(self) -> bool:
        """Launch units capped by max_ctas (``uses_max_ctas``)."""
        return self.schedule in ("static", "atomic", "reverse", "rbroadcast")

    @property
    def tile_grid(self) -> bool:
        """The layer launches the tile space itself (R13: OneShot, plain Raster, no record)."""
        return self.schedule == "oneshot" and self.plain_raster and not self.record

    def item_rounds(self, m: int, n: int, li: int) -> int:
        base, spread, a, b, c = self.rounds
        return (m * a + n * b + li * c) % spread + base


BROADCAST = ("atomic", "clc", "rbroadcast")


@functools.cache
def item_order(s: StreamSpec) -> tuple[tuple[int, int, int], ...]:
    """Raster visiting order: ``l`` slowest; bands of ``swizzle`` tiles across the non-``along``
    axis, the banded axis fastest inside a band; reversed axes count down."""
    em, en, el = s.extents
    if min(s.extents) <= 0:
        return ()
    e_along, e_band = (em, en) if s.along == "m" else (en, em)
    out = []
    for li in range(el):
        for start in range(0, e_band, s.swizzle):
            band = range(start, min(start + s.swizzle, e_band))
            for a in range(e_along):
                for b in band:
                    m, n = (a, b) if s.along == "m" else (b, a)
                    if "m" in s.reverse:
                        m = em - 1 - m
                    if "n" in s.reverse:
                        n = en - 1 - n
                    out.append((m, n, el - 1 - li if "l" in s.reverse else li))
    return tuple(out)


def skipped(s: StreamSpec, item) -> bool:
    if s.skip is None:
        return False
    a, b, c, k = s.skip
    m, n, li = item
    return (m * a + n * b + li * c) % k == 0


@functools.cache
def table_rows(s: StreamSpec) -> tuple[tuple[int, int, int], ...]:
    """The "table" mapping's rows: the raster items in a seeded permutation."""
    rows = list(item_order(s))
    random.Random(s.order_seed).shuffle(rows)
    return tuple(rows)


@functools.cache
def packed_rows(s: StreamSpec) -> tuple[tuple[int, int, int], ...]:
    """The "packed" mapping's first ``length`` rows: the permuted items ``skip`` keeps."""
    return tuple(it for it in table_rows(s) if not skipped(s, it))


@dataclass(frozen=True)
class MmaSpec:
    """A 128 x n x k tcgen05 BF16 MMA tile per accumulator epoch (single CTA)."""

    n: int  # accumulator columns
    k: int  # K elements per A/B stage
    salt: int


@dataclass(frozen=True)
class KernelSpec:
    seed: int
    pair: bool
    roles: tuple[RoleSpec, ...]
    channels: tuple[ChannelSpec, ...]
    turn_role: str | None = None
    turn_enabled: bool = True
    stream: StreamSpec | None = None
    iters: int = 1  # iterations without a stream
    units: int = 1  # work units launched without a stream
    tmem_regions: tuple[tuple[str, int, int], ...] = ()
    smem_pad: int = 0  # Int32 words of extra SMEM (CLC: one CTA per SM)
    log_cap: int = 32
    item_cap: int = 64
    turn_cap: int = 64
    launches: int = 2
    trace: bool = False  # trace_protocol: must not change results (R16)
    # Further launches of the same compiled kernel with other runtime values:
    # ((m_tiles, n_tiles, l), max_ctas, iterations) each. Extents, caps and counts are runtime
    # arguments, so one binary must serve them all.
    variants: tuple[tuple[tuple[int, int, int], int, int], ...] = ()
    graph: bool = False  # also capture one launch in a CUDA graph and replay it twice
    # (m, n, pair) of a declared ``Cluster``; () is the legacy shape: ``CtaPair()`` when
    # ``pair``, else a single CTA. ``pair`` always says whether the cluster has MMA pairs.
    cluster: tuple = ()

    @property
    def shape(self) -> tuple[int, int, bool]:
        if self.cluster:
            return tuple(self.cluster)
        return (2, 1, True) if self.pair else (1, 1, False)

    @property
    def geom(self) -> Geometry:
        return Geometry(*self.shape)

    iket: int = 0  # 1: enable_iket(waits=False), 2: enable_iket() (compiled and run, no capture)
    alias_check: str = "warn"
    # Also launch the compiled kernel twice concurrently on two streams, each with its own
    # launch_args() workspace and logs (the documented per-stream workspace contract).
    concurrent: bool = False
    mma: MmaSpec | None = None

    @property
    def cs(self) -> int:
        return self.geom.size

    def role(self, name: str) -> RoleSpec:
        return next(r for r in self.roles if r.name == name)

    def ctas(self, rname: str) -> tuple[int, ...]:
        """Ranks role ``rname`` runs in."""
        return self.geom.ctas(self.role(rname).cta)

    def key(self) -> str:
        d = asdict(self)
        if not d["cluster"]:  # legacy specs keep their keys (kernel names, cached builds)
            del d["cluster"]
        for ch in d["channels"]:
            if not ch["multicast"]:
                del ch["multicast"]
        blob = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha1(blob).hexdigest()[:12]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(text: str) -> KernelSpec:
        return spec_from_dict(json.loads(text))


def _tuples(x):
    return tuple(_tuples(v) for v in x) if isinstance(x, list) else x


def spec_from_dict(d: dict) -> KernelSpec:
    roles = tuple(RoleSpec(**{k: _tuples(v) for k, v in r.items()}) for r in d["roles"])
    chans = tuple(ChannelSpec(**{k: _tuples(v) for k, v in c.items()}) for c in d["channels"])
    stream = d.get("stream")
    if stream is not None:
        stream = StreamSpec(**{k: _tuples(v) for k, v in stream.items()})
    mma = d.get("mma")
    if mma is not None:
        mma = MmaSpec(**mma)
    rest = {k: _tuples(v) for k, v in d.items() if k not in ("roles", "channels", "stream", "mma")}
    return KernelSpec(roles=roles, channels=chans, stream=stream, mma=mma, **rest)


# --------------------------------------------------------------------------- #
# Derived structure shared by the emitter and the oracle
# --------------------------------------------------------------------------- #


def is_local(spec: KernelSpec, ch: ChannelSpec) -> bool:
    """CTA-local per the contract (R11): every participant runs in the same CTAs."""
    return len({spec.ctas(r) for r, _ in (*ch.producers, *ch.consumers)}) == 1


def payload_offsets(spec: KernelSpec, ch: ChannelSpec) -> tuple[dict, int]:
    """``{role: (offset of each instance)}`` of the words each producer writes, and the width."""
    offsets, width = {}, 0
    for rname, bound in ch.producers:
        r = spec.role(rname)
        if bound:  # one instance per slot: they share one region
            offsets[rname] = (width,) * r.instances
            width += r.threads
        else:
            offsets[rname] = tuple(width + i * r.threads for i in range(r.instances))
            width += r.threads * r.instances
    return offsets, width


@dataclass
class Layout:
    """Per-CTA word offsets of every log the generated kernel writes (``clog[cta, off]``)."""

    per_cta: int = 0
    presence: int = 0
    role_it: dict = field(default_factory=dict)  # (role, instance) -> offset
    role_items: dict = field(default_factory=dict)  # (role, instance) -> [count, items, hashes]
    cons: dict = field(default_factory=dict)  # (chan, role, instance) -> [count, errs, log...]
    prod: dict = field(default_factory=dict)  # (chan, role, instance) -> count
    turn: int | None = None
    accumulators: list = field(default_factory=list)  # offsets initialized to 0 (else -1)
    instance_ids: dict = field(default_factory=dict)  # (role, instance) -> global ordinal

    @staticmethod
    @functools.cache
    def of(spec: KernelSpec) -> Layout:
        lay = Layout()
        off = MAX_WARPS
        lay.accumulators += list(range(MAX_WARPS))
        ordinal = 0
        for r in spec.roles:
            for i in range(r.instances):
                lay.instance_ids[(r.name, i)] = ordinal
                ordinal += 1
                lay.role_it[(r.name, i)] = off
                off += 1
                if spec.stream is not None:
                    lay.role_items[(r.name, i)] = off
                    off += 1 + 2 * spec.item_cap
        for ch in spec.channels:
            for rname, _ in ch.consumers:
                for i in range(spec.role(rname).instances):
                    lay.cons[(ch.name, rname, i)] = off
                    lay.accumulators.append(off + 1)
                    off += 2 + spec.log_cap
            for rname, _ in ch.producers:
                for i in range(spec.role(rname).instances):
                    lay.prod[(ch.name, rname, i)] = off
                    off += 1
        if spec.turn_role is not None:
            lay.turn = off
            lay.accumulators.append(off)
            off += 1 + spec.turn_cap
        lay.per_cta = off
        return lay


def presence_code(spec: KernelSpec, rname: str, inst: int) -> int:
    return 1 + Layout.of(spec).instance_ids[(rname, inst)]


# --------------------------------------------------------------------------- #
# Random generation
# --------------------------------------------------------------------------- #


def _place(rng: random.Random, roles: list[dict]) -> None:
    """Assign each role's instances to warps, in random order with random holes."""
    order = list(range(len(roles)))
    rng.shuffle(order)
    w = rng.choice([0, 0, 0, 1, 2])
    for i in order:
        r = roles[i]
        firsts = []
        for _ in range(r["instances"]):
            firsts.append(w)
            w += r["width"]
            if rng.random() < 0.15:
                w += 1
        r["warps"] = tuple(firsts)
        if rng.random() < 0.3:
            w += rng.randint(1, 3)


def _default_regs(num_warps: int) -> int:
    return min(REGS_PER_SM // (num_warps * 32) // 8 * 8, 256)


def _scheduler_warp(roles: list[dict], explicit: int | None) -> int:
    owned = {w for r in roles for f in r["warps"] for w in range(f, f + r["width"])}
    if explicit is not None:
        return explicit
    return next(w for w in range(max(owned) + 2) if w not in owned)


def registers_fit(roles: list[dict], sched: int | None) -> bool:
    """R9: warpgroup-uniform budgets (holes take their warpgroup's budget, idle groups 24,
    ``None`` the default) whose total fits the registers the launch starts with: the default
    budget of every padded warp (``setmaxnreg`` only moves registers between warps)."""
    budget = {}
    for r in roles:
        for f in r["warps"]:
            for w in range(f, f + r["width"]):
                budget[w] = r["regs"]
    if sched is not None:
        budget[sched] = _scheduler_regs(roles, sched)
    nw = -(-(max(budget) + 1) // WARPGROUP) * WARPGROUP
    default = _default_regs(nw)
    total = 0
    for g in range(nw // WARPGROUP):
        vals = {budget[w] for w in range(g * 4, g * 4 + 4) if w in budget}
        if len(vals) > 1:
            return False
        regs = vals.pop() if vals else MIN_REGS
        total += 4 * 32 * (default if regs is None else regs)
    return total <= default * nw * 32


def _assign_regs(rng: random.Random, roles: list[dict]) -> None:
    """Warpgroup-uniform budgets: roles sharing a warpgroup form one component."""
    parent = list(range(len(roles)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    groups = {}
    for i, r in enumerate(roles):
        for f in r["warps"]:
            for w in range(f, f + r["width"]):
                groups.setdefault(w // WARPGROUP, []).append(i)
    for members in groups.values():
        for j in members[1:]:
            parent[find(j)] = find(members[0])
    choice = {}
    for i in range(len(roles)):
        c = find(i)
        if c not in choice:
            choice[c] = None if rng.random() < 0.25 else rng.choice(range(24, 257, 8))
        roles[i]["regs"] = choice[c]


def _make_events(rng: random.Random, handles: list[Key], kinds: dict) -> list:
    """A random open/close interleaving: each handle opens before it closes; ring handles of one
    channel open in key order (acquire order assigns the epochs)."""
    opened, closed, events = set(), set(), []
    while len(closed) < len(handles):
        avail = []
        for h in handles:
            ch, side, k = h
            if h not in opened:
                prev = (ch, side, k - 1)
                if kinds[ch] != "ring" or k == 0 or prev in opened:
                    avail.append(("o", h))
            elif h not in closed:
                avail.append(("c", h))
        op, h = rng.choice(avail)
        (opened if op == "o" else closed).add(h)
        events.append((op, *h))
    return events


def _laminar(rng: random.Random, events: list) -> list[Key]:
    """A random subset of handles whose [open, close] intervals nest or are disjoint."""
    span = {}
    for pos, (op, *h) in enumerate(events):
        span.setdefault(tuple(h), [pos, pos])[1 if op == "c" else 0] = pos
    chosen = []
    keys = list(span)
    rng.shuffle(keys)
    for h in keys:
        a0, a1 = span[h]
        if rng.random() < 0.6 and all(
            a1 < b0 or b1 < a0 or (a0 < b0 and b1 < a1) or (b0 < a0 and a1 < b1)
            for b0, b1 in (span[c] for c in chosen)
        ):
            chosen.append(h)
    return sorted(chosen)


def _placement(rng: random.Random, pair: bool, cluster: tuple) -> str:
    if not cluster:  # legacy draws: seeds keep their specs
        return "leader" if pair and rng.random() < 0.35 else "all"
    if cluster[0] * cluster[1] == 1:
        return "all"
    x = rng.random()
    return "leader" if x < 0.3 else "rank0" if x < 0.45 else "all"


def _scheduler_regs(roles: list[dict], sched: int | None):
    """R9: a warp the scheduler takes in an owned warpgroup has that group's budget (None
    included); alone in a warpgroup it takes the smallest numeric budget."""
    if sched is None:
        return None
    grp = {
        r["regs"]
        for r in roles
        for f in r["warps"]
        for w in range(f, f + r["width"])
        if w // WARPGROUP == sched // WARPGROUP
    }
    nums = [r["regs"] for r in roles if r["regs"] is not None]
    return grp.pop() if len(grp) == 1 else (min(nums) if nums else None)


def _extend_stream(ext: random.Random, s: StreamSpec) -> StreamSpec:
    """Custom schedules (a local and a broadcast ``IndexSchedule``) and mappings without
    ``decodes_any_index`` (an item table, ``PackedWork`` under ``PdlWait``)."""
    if s.schedule in ("static", "atomic") and ext.random() < 0.3:
        s = dataclasses.replace(
            s, schedule={"static": "reverse", "atomic": "rbroadcast"}[s.schedule]
        )
    if s.extents[1] >= 65535:
        return s
    r = ext.random()
    if r < 0.15:
        s = dataclasses.replace(s, mapping="table", skip=None, order_seed=ext.randrange(1 << 16))
    elif r < 0.3:
        skip = s.skip or tuple(ext.randint(0, 3) for _ in range(3)) + (ext.randint(2, 4),)
        s = dataclasses.replace(
            s,
            mapping="packed",
            skip=skip,
            pad=ext.randint(1, 9),
            ready=True,
            pdl=ext.random() < 0.6,
            order_seed=ext.randrange(1 << 16),
        )
    return s


MMA_M = 128
MMA_MIN_REGS = 40
PRODUCER_THREADS = 128  # the packed table's PDL producer kernel
MMA_TILES_MAX_COLS = 512


def _owned(roles: list[dict]) -> set[int]:
    return {w for r in roles for f in r["warps"] for w in range(f, f + r["width"])}


def _extend_roles(ext, roles, channels, geom, sched, tmem_regions):
    """Append the MMA subgraph (single CTA) and a raw Barriers ring to ``roles``/``channels``
    in fresh warpgroups; give those groups budgets that fit. Returns ``(MmaSpec | None,
    tmem_regions, SMEM bytes)``."""
    top = max(_owned(roles) | ({sched} if sched is not None else set()))
    w = (top // WARPGROUP + 1) * WARPGROUP
    first_new = len(roles)
    mma, smem = None, 0
    used_cols = sum(c * k for _, c, k in tmem_regions)
    # F-C3: next to tcgen05 MMA code, ptxas cannot always allocate role budgets of 24 or 32
    # (C7600, a compile error); keep the MMA subgraph to kernels whose budgets are >= 40.
    low = any(r["regs"] is not None and r["regs"] < MMA_MIN_REGS for r in roles)
    if geom.size == 1 and not low and ext.random() < 0.25 and w + 6 <= MAX_WARPS:
        n, k = ext.choice([32, 64]), 64
        ab_n, acc_n, kb = ext.randint(1, 3), ext.randint(1, 2), ext.randint(1, 3)
        if used_cols + n * acc_n <= MMA_TILES_MAX_COLS:
            mma = MmaSpec(n=n, k=k, salt=ext.randrange(64))
            tmem_regions = (*tmem_regions, ("mma_acc", n, acc_n))
            base = {"instances": 1, "cta": "all", "jitter": 0, "tmem": False, "regs": None}
            roles += [
                {
                    **base,
                    "name": "m_epi",
                    "width": 4,
                    "warps": (w,),
                    "tmem": True,
                    "kind": "mma_epi",
                },
                {**base, "name": "m_load", "width": 1, "warps": (w + 4,), "kind": "mma_load"},
                {
                    **base,
                    "name": "m_mma",
                    "width": 1,
                    "warps": (w + 5,),
                    "tmem": True,
                    "kind": "mma",
                },
            ]
            w += 6
            chan = {
                "kind": "ring",
                "producers": (),
                "consumers": (),
                "extra_mult": 0,
                "payload": False,
                "salt": ext.randrange(4096),
            }
            channels += [
                {
                    **chan,
                    "name": "mab",
                    "n": ab_n,
                    "producer": "tma",
                    "consumer": "umma",
                    "tma_rows": 0,
                    "producers": (("m_load", False),),
                    "consumers": (("m_mma", False),),
                    "mult": kb,
                    "mma": "ab",
                },
                {
                    **chan,
                    "name": "macc",
                    "n": acc_n,
                    "producer": "umma",
                    "consumer": "threads",
                    "producers": (("m_mma", False),),
                    "consumers": (("m_epi", False),),
                    "mult": 1,
                    "mma": "acc",
                },
            ]
            roles[-2]["events"] = tuple(
                e for j in range(kb) for e in (("o", "mab", "p", j), ("c", "mab", "p", j))
            )
            roles[-1]["events"] = (
                ("o", "macc", "p", 0),
                *(e for j in range(kb) for e in (("o", "mab", "c", j), ("c", "mab", "c", j))),
                ("c", "macc", "p", 0),
            )
            roles[-3]["events"] = (("o", "macc", "c", 0), ("c", "macc", "c", 0))
            smem += ab_n * (MMA_M + n) * k * 2 + 2048
    if ext.random() < 0.25:
        wp, wc = ext.choice([1, 1, 2]), ext.choice([1, 1, 2])
        if w + wp + wc <= MAX_WARPS:
            cta = ext.choice(["all", "leader", "rank0"]) if geom.size > 1 else "all"
            jitter = any(r["jitter"] for r in roles)
            n = ext.randint(1, 4)
            for name, width in (("x_rawp", wp), ("x_rawc", wc)):
                side = "p" if name == "x_rawp" else "c"
                roles.append(
                    {
                        "name": name,
                        "width": width,
                        "instances": 1,
                        "cta": cta,
                        "regs": None,
                        "warps": (w,),
                        "jitter": ext.choice([0, 40, 400]) if jitter else 0,
                        "tmem": bool(tmem_regions) and side == "c" and ext.random() < 0.5,
                        "kind": "raw",
                        "events": (("o", "xraw", side, 0), ("c", "xraw", side, 0)),
                    }
                )
                w += width
            channels.append(
                {
                    "name": "xraw",
                    "kind": "raw",
                    "n": n,
                    "producer": "threads",
                    "consumer": "threads",
                    "producers": (("x_rawp", False),),
                    "consumers": (("x_rawc", False),),
                    "mult": 1,
                    "extra_mult": ext.randint(0, 1),
                    "payload": False,
                    "salt": ext.randrange(4096),
                    "raw_init": ext.choice(["auto", "manual"]),
                }
            )
            smem += 32 * wp * 4 * n + 16 * n + 16
    for r in roles[first_new:]:
        r.setdefault("with_handles", ())
    if not _budget_new_groups(ext, roles, first_new, sched):
        del roles[first_new:]
        channels[:] = [c for c in channels if c["kind"] != "raw" and not c.get("mma")]
        return None, tuple(t for t in tmem_regions if t[0] != "mma_acc"), 0
    return mma, tmem_regions, smem


def _budget_new_groups(ext, roles, first_new, sched) -> bool:
    """Warpgroup-uniform budgets for the appended roles' groups that keep R9's total (False:
    none fits next to the existing budgets)."""
    groups = sorted({w // WARPGROUP for r in roles[first_new:] for w in _owned([r])})
    for g in groups:
        members = [r for r in roles[first_new:] if any(w // WARPGROUP == g for w in _owned([r]))]
        heavy = any(r.get("kind") in ("mma", "mma_epi") for r in members)
        mma = any(r.get("kind", "").startswith("mma") for r in roles)
        floor = 64 if heavy else MMA_MIN_REGS if mma else 24
        options = [None, *range(floor, 257, 8)]
        ext.shuffle(options)
        for regs in options:
            for r in members:
                r["regs"] = regs
            if registers_fit(roles, sched):
                break
        else:
            return False
    return True


def _extend_channels(ext, channels, roles, geom):
    """Overrides with manual per-warp signals, unchecked(...) sites and SmemPlan storage on the
    generated channels."""
    by_name = {r["name"]: r for r in roles}
    for ch in channels:
        if ch["kind"] == "raw" or ch.get("mma"):
            continue
        parts = (*ch["producers"], *ch["consumers"])
        # Manual signals reach this CTA only: a multicast stage is released in every CTA it
        # was copied into (U7: raw signals must use the effective targets).
        local = len({geom.ctas(by_name[r]["cta"]) for r, _ in parts}) == 1
        local = local and not ch.get("multicast") and not ch.get("fills")
        if local and ext.random() < 0.2:
            sides = [
                x
                for x, mech in (("p", ch["producer"]), ("c", ch["consumer"]))
                if mech == "threads"
            ]
            if sides:
                ch["manual"] = "".join(x for x in sides if ext.random() < 0.7) or sides[0]
                ch["override_form"] = ext.choice(["int", "list", "rank"])
                ch["generic"] = ch["kind"] == "ring" and ext.random() < 0.5
        if not ch.get("manual") and ext.random() < 0.15:
            kinds = ["cross_region"]
            if local and ch["producer"] in ("threads", "tma"):
                kinds.append("unclosed")
            if ch["producer"] == "tma":
                kinds.append("tma")
            kind = ext.choice(kinds)
            side = ext.choice("pc") if kind == "cross_region" else "p"
            if kind == "unclosed" and ch["consumer"] == "threads" and ext.random() < 0.5:
                side = "c"
            ch["unchecked"], ch["unchecked_side"] = kind, side
        if ch["payload"] and ext.random() < 0.3:
            fan_in = len(ch["producers"]) > 1
            ch["storage"] = "buffer" if fan_in or ext.random() < 0.4 else "overlap"
            ch["alias_opt_out"] = ch["storage"] == "overlap" and ext.random() < 0.3
        elif ch["producer"] == "tma" and not ch.get("multicast") and ext.random() < 0.3:
            ch["storage"] = "buffer"


def _extend_kernel(ext, spec: KernelSpec) -> KernelSpec:
    s = spec.stream
    alias = ext.random()
    return dataclasses.replace(
        spec,
        iket=ext.choice([1, 2]) if ext.random() < 0.15 else 0,
        alias_check="strict" if alias < 0.25 else "off" if alias < 0.4 else "warn",
        concurrent=ext.random() < 0.15 and not spec.graph and (s is None or s.total < 4096),
    )


def generate(
    seed: int,
    *,
    pair: bool | None = None,
    stream: bool | None = None,
    empty: bool = False,
    cluster: tuple[int, int, bool] | None = None,
) -> KernelSpec:
    """A random valid kernel. ``pair``/``stream`` force those knobs; otherwise random.
    ``cluster=(m, n, pair)`` declares ``Cluster(m, n, pair=pair)`` (overriding ``pair``) and
    draws placements among ``all``, ``leader`` and ``rank0``; without it the kernel is a single
    CTA or a ``CtaPair``.
    ``empty``: the zero-work boundary corpus (a WorkStream whose space has a zero extent; the
    contract treats it separately from the nonempty corpus, G6)."""
    if empty:
        stream = True
    rng = random.Random(seed)
    # Knobs added after the first campaigns draw from their own stream, so a seed keeps the
    # kernel it had before them apart from the extensions (findings cite seeds).
    ext = random.Random(f"wdfuzz-ext-{seed}")
    fill_rng = random.Random(f"wdfuzz-fill-{seed}")  # Tma(fills=...), added after ext
    warps_rng = random.Random(f"wdfuzz-warps-{seed}")  # Warps sides, added after fills
    wait_rng = random.Random(f"wdfuzz-wait-{seed}")  # try tokens and previews, added after Warps
    fence_rng = random.Random(f"wdfuzz-fence-{seed}")  # derived release fences, added after waits
    cluster = tuple(cluster) if cluster is not None else ()
    if cluster:
        pair = cluster[2]
    else:
        pair = rng.random() < 0.3 if pair is None else pair
    geom = Geometry(*cluster) if cluster else Geometry(2 if pair else 1, 1, pair)
    use_stream = rng.random() < 0.6 if stream is None else stream
    nroles = rng.randint(2, 6)
    roles = []
    for i in range(nroles):
        width = rng.choice([1, 1, 1, 2, 4])
        roles.append(
            {
                "name": f"r{i}",
                "width": width,
                "instances": rng.choice([1, 1, 1, 2, 3]) if width <= 2 else 1,
                "cta": _placement(rng, pair, cluster),
                "jitter": rng.choice([0, 0, 0, 40, 400, 3000]),
                "tmem": False,
                "regs": None,
            }
        )
    single = [i for i, r in enumerate(roles) if r["width"] == 1 and r["instances"] == 1]

    channels = []
    nch = rng.randint(max(1, nroles - 2), nroles + 2)
    for c in range(nch):
        p, q = sorted(rng.sample(range(nroles), 2))  # forward edge in topological order
        P, Q = roles[p], roles[q]
        kind = "slots" if rng.random() < 0.4 else "ring"
        prod_mech = "threads"
        cons_mech = "threads"
        if p in single and rng.random() < 0.35:
            prod_mech = rng.choice(["umma", "tma", "tma"])
        if q in single and rng.random() < 0.2:
            cons_mech = "umma"
        producers = [(P["name"], False)]
        consumers = [(Q["name"], False)]
        # Fan-in (Threads producers only) and fan-out joins (Threads consumers only).
        # Every producer precedes every consumer in the topological order.
        last_prod = p
        if prod_mech == "threads" and rng.random() < 0.25:
            others = [i for i in range(q) if i != p]
            if others:
                extra = rng.choice(others)
                producers.append((roles[extra]["name"], False))
                last_prod = max(p, extra)
        if cons_mech == "threads" and rng.random() < 0.35:
            others = [i for i in range(last_prod + 1, nroles) if i != q]
            if others:
                consumers.append((roles[rng.choice(others)]["name"], False))
        n = rng.randint(1, 8) if kind == "ring" else rng.randint(1, 4)
        if kind == "slots" and rng.random() < 0.08:
            n = rng.randint(5, 31)  # phase bits up to bit 30 of the Int32 mask
        if kind == "slots":
            # Bind an instanced role to the slots (Producer[Instance] / Consumer[Instance]).
            if prod_mech == "threads" and P["instances"] > 1 and rng.random() < 0.6:
                n = P["instances"]
                producers[0] = (P["name"], True)
            if (
                cons_mech == "threads"
                and Q["instances"] > 1
                and rng.random() < 0.6
                and (not producers[0][1] or Q["instances"] == n)
            ):
                n = Q["instances"]
                consumers[0] = (Q["name"], True)
                if producers[0][1] and P["instances"] != n:
                    producers[0] = (P["name"], False)
        mult = rng.randint(1, min(n, 3)) if kind == "ring" else 1
        channels.append(
            {
                "name": f"c{c}",
                "kind": kind,
                "n": n,
                "producer": prod_mech,
                "consumer": cons_mech,
                "producers": tuple(producers),
                "consumers": tuple(consumers),
                "mult": mult,
                "extra_mult": rng.randint(0, mult) if kind == "ring" else 0,
                "extra_slots": tuple(sorted(rng.sample(range(n), rng.randint(0, n))))
                if kind == "slots"
                else (),
                "salt": rng.randint(0, 4095),
            }
        )

    # Cluster placement rules: with MMA pairs tcgen05.commit sides run in leaders; TMA is
    # CTA-local, or 2-CTA (producers in every CTA, consumers in exactly the leaders).
    def placed_ctas(side):
        return {geom.ctas(roles[int(r[1:])]["cta"]) for r, _ in side}

    everyone, leaders = geom.ctas("all"), geom.ctas("leader")
    if geom.size > 1:
        if pair:
            for ch in channels:
                for side in ("producers", "consumers"):
                    mech = ch["producer" if side == "producers" else "consumer"]
                    if mech == "umma":
                        for rname, _ in ch[side]:
                            if roles[int(rname[1:])]["cta"] == "all":
                                roles[int(rname[1:])]["cta"] = "leader"
            for ch in channels:  # promote some TMA consumers so the channel becomes pair-scoped
                if (
                    ch["producer"] == "tma"
                    and placed_ctas(ch["producers"]) == {everyone}
                    and rng.random() < 0.4
                ):
                    for rname, _ in ch["consumers"]:
                        roles[int(rname[1:])]["cta"] = "leader"
        # A rank-0 tcgen05.commit side can only serve rank 0's pair (without pairs, a
        # producer only rank 0): the other side's MMA pairs each need their own leader.
        # Promote until stable.
        sides = (("producers", "consumers"), ("consumers", "producers"))
        changed = True
        while changed:
            changed = False
            for ch in channels:
                for side, other in sides if pair else sides[:1]:
                    if ch["producer" if side == "producers" else "consumer"] != "umma":
                        continue
                    served = {geom.group(c)[0] for cs in placed_ctas(ch[other]) for c in cs}
                    for rname, _ in ch[side]:
                        r = roles[int(rname[1:])]
                        if r["cta"] == "rank0" and served - {0}:
                            r["cta"], changed = "leader", True
        for ch in channels:
            if ch["producer"] == "tma":
                prod, cons = placed_ctas(ch["producers"]), placed_ctas(ch["consumers"])
                if pair and prod == {everyone} and cons == {leaders} and rng.random() < 0.7:
                    ch["tma_pair"] = True  # each pair's bytes complete on its leader (R11)
                    continue
                # One CTA fills others (R11): the pair leaders fill their pair, rank 0 the
                # cluster; every CTA it fills has a consumer, or the consumers are the leaders.
                fills = "pair" if pair and prod == {leaders} else ""
                fills = "cluster" if not fills and prod == {(0,)} else fills
                waiting = set().union(*cons)
                fill_two = pair and waiting == set(leaders)
                fits = fills and (waiting == set(everyone) or fill_two)
                if fits and fill_rng.random() < (0.7 if len(prod | cons) > 1 else 0.3):
                    ch["fills"], ch["fill_two"] = fills, fill_two
                elif prod == {everyone} and waiting == set(everyone) and fill_rng.random() < 0.5:
                    pass  # each CTA loads its own stage; extra leader or rank0 consumers join
                elif len(prod | cons) > 1:
                    ch["producer"] = "threads"
    spec_roles = {r["name"]: r for r in roles}
    for ch in channels:
        local = len(placed_ctas((*ch["producers"], *ch["consumers"]))) == 1
        if ch["producer"] == "tma":
            ch["tma_rows"] = rng.choice([1, 2, 4, 8])
            ch["release_fence"] = "smem_async"
            ch["implicit_release_fence"] = ch["consumer"] == "threads" and fence_rng.random() < 0.5
            ch["payload"] = False
        else:
            ch["payload"] = local and ch["consumer"] == "threads" and ch["producer"] == "threads"
            ch["commit_fence"] = rng.choice([None, None, "smem_async"])
            ch["release_fence"] = rng.choice([None, None, "smem_async"])
        if not local and ch["producer"] == "threads" or not local and ch["consumer"] == "threads":
            ch["remote_scope"] = rng.choice([None, "cta", "cluster"])
        if ch["payload"]:
            width = sum(
                spec_roles[r]["width"] * 32 * (1 if b else spec_roles[r]["instances"])
                for r, b in ch["producers"]
            )
            if width > 512:
                ch["payload"] = False
    if cluster:  # multicast TMA along a mode with more than one CTA
        modes = [m for m in "mn" if len(geom.reach(0, (m,))) > 1]
        for ch in channels:
            side = (*ch["producers"], *ch["consumers"])
            if ch["producer"] != "tma" or not modes:
                continue
            # Every CTA a copy lands in needs a waiting consumer: producers and consumers in
            # every CTA (extra leader or rank0 consumers may join), 2-CTA channels, pair fills,
            # or one placement without rank0 roles.
            prod, cons = placed_ctas(ch["producers"]), placed_ctas(ch["consumers"])
            everywhere = prod == {everyone} and set().union(*cons) == set(everyone)
            single = len(placed_ctas(side)) == 1 and not any(
                roles[int(r[1:])]["cta"] == "rank0" for r, _ in side
            )
            pair_fill = ch.get("fills") == "pair"
            if not (everywhere or single or ch.get("tma_pair") or pair_fill):
                continue
            if rng.random() < 0.6:
                ch["multicast"] = rng.choice(modes)
                # Each sharing CTA copies a slice of whole rows.
                ch["tma_rows"] = max(ch["tma_rows"], len(geom.reach(0, (ch["multicast"],))))

    # TMEM: a plan and a few roles that take the view (alloc/free/teardown barrier counts).
    tmem_regions = ()
    if rng.random() < 0.3:
        regions, used = [], 0
        for k in range(rng.randint(1, 3)):
            cols, slots = rng.choice([32, 64, 128]), rng.randint(1, 2)
            if used + cols * slots <= 512:
                regions.append((f"t{k}", cols, slots))
                used += cols * slots
        tmem_regions = tuple(regions)
        users = rng.sample(range(nroles), rng.randint(1, nroles))
        for i in users:
            roles[i]["tmem"] = True
        # E39: a cluster allocates TMEM from a role that runs in every CTA.
        if geom.size > 1 and not any(roles[i]["cta"] == "all" for i in users):
            everywhere = [r for r in roles if r["cta"] == "all"]
            if everywhere:
                rng.choice(everywhere)["tmem"] = True
            else:
                tmem_regions = ()
                for r in roles:
                    r["tmem"] = False

    turn_candidates = [r["name"] for r in roles if r["instances"] >= 2]
    turn_role = rng.choice(turn_candidates) if turn_candidates and rng.random() < 0.5 else None

    _place(rng, roles)
    if max(f + r["width"] for r in roles for f in r["warps"]) > 24:
        return generate(
            seed + 1_000_003, pair=pair, stream=stream, empty=empty, cluster=cluster or None
        )

    stream_spec = None
    if use_stream:
        schedule = rng.choice(["oneshot", "static", "atomic", "clc"])
        stages = rng.randint(1, 4)
        along = rng.choice(["m", "m", "n"])
        swizzle = rng.choice([1, 1, 2, 3, 8])
        reverse = tuple(a for a in ("m", "n", "l") if rng.random() < 0.15)
        skip = None
        if rng.random() < 0.2:
            skip = (rng.randint(0, 3), rng.randint(0, 3), rng.randint(0, 3), rng.randint(1, 4))
        if schedule == "oneshot" and rng.random() < 0.5:
            along, swizzle, reverse, skip = "m", 1, (), None  # the tile-grid launch
        big = schedule == "oneshot" and along == "m" and swizzle == 1 and not reverse and not skip
        if big and rng.random() < 0.08:
            extents = (rng.randint(1, 2), 65535 + rng.randint(0, 2), 1)  # y-limit fallback edge
        else:
            extents = (rng.randint(1, 12), rng.randint(1, 12), rng.randint(1, 3))
            if rng.random() < 0.15:
                extents = (1, 1, 1)
        if empty:  # at least one extent is zero or negative (R17: both mean an empty space)
            zero_axis = rng.randrange(3)
            extents = tuple(
                (0 if rng.random() < 0.75 else -rng.randint(1, 3))
                if i == zero_axis or rng.random() < 0.2
                else e
                for i, e in enumerate(extents)
            )
        stream_spec = StreamSpec(
            schedule=schedule,
            stages=stages,
            along=along,
            swizzle=swizzle,
            reverse=reverse,
            skip=skip,
            ready=rng.random() < 0.2,
            extents=extents,
            max_ctas=rng.choice([1, 2, 3, 5, 8, 37, SMS]),
            scheduler_warp=None,
            rounds=(
                rng.randint(0, 2),
                rng.randint(1, 3),
                rng.randint(0, 3),
                rng.randint(0, 3),
                rng.randint(0, 3),
            ),
            launch_1d=schedule == "oneshot" and rng.random() < 0.3,
            record=rng.randint(1, 4) if extents[1] < 65535 and rng.random() < 0.15 else 0,
        )
        stream_spec = _extend_stream(ext, stream_spec)

    # Registers: random warpgroup-uniform budgets that fit, including the scheduler warp.
    sched = None
    if stream_spec is not None and stream_spec.broadcast:
        owned = sorted({w for r in roles for f in r["warps"] for w in range(f, f + r["width"])})
        holes = [w for w in range(max(owned) + 2) if w not in owned]
        explicit = rng.choice(holes) if rng.random() < 0.3 else None
        stream_spec = StreamSpec(**{**asdict(stream_spec), "scheduler_warp": explicit})
        sched = _scheduler_warp(roles, explicit)
    if rng.random() < 0.5:
        for _ in range(20):
            _assign_regs(rng, roles)
            if registers_fit(roles, sched):
                break
        else:
            for r in roles:
                r["regs"] = None

    # Per-role handle programs.
    kinds = {ch["name"]: ch["kind"] for ch in channels}
    for r in roles:
        handles = []
        for ch in channels:
            for side, lst in (("p", ch["producers"]), ("c", ch["consumers"])):
                for rname, bound in lst:
                    if rname != r["name"]:
                        continue
                    if ch["kind"] == "ring":
                        keys = range(ch["mult"])
                    else:
                        keys = [BOUND] if bound else rng.sample(range(ch["n"]), ch["n"])
                    handles += [(ch["name"], side, k) for k in keys]
        events = _make_events(rng, handles, kinds)
        r["events"] = tuple(tuple(e) for e in events)
        r["with_handles"] = tuple(_laminar(rng, events))

    # Raw roles over Barriers and the MMA subgraph sit in fresh warpgroups above every role and
    # the scheduler, with fixed sequential handle programs.
    mma, ext_smem = None, 0
    if stream_spec is None or stream_spec.extents[1] < 65535:
        mma, tmem_regions, ext_smem = _extend_roles(
            ext, roles, channels, geom, sched, tmem_regions
        )

    grid_rows = 1
    if stream_spec is None:
        units = rng.randint(1, 3)
        grid_rows = units
    else:
        units = 1
        s = stream_spec
        if s.capped:
            grid_rows = max(1, min(s.total, max(s.max_ctas // geom.size, 1)))
        else:
            grid_rows = max(s.total, 1)
    big_grid = grid_rows > 4096
    smem_pad = 0
    if stream_spec is not None and stream_spec.schedule == "clc" and rng.random() < 0.5:
        smem_pad = 40000  # ~160 KB: one CTA per SM, so try_cancel has CTAs to cancel
    used = ext_smem
    for ch in channels:  # barriers, layer payloads and TMA stages, 16-byte aligned
        used += 16 * ch["n"]
        if ch["payload"]:
            width = sum(
                spec_roles[r]["width"] * 32 * (1 if b else spec_roles[r]["instances"])
                for r, b in ch["producers"]
            )
            used += 4 * width * ch["n"] + 16
        if ch["producer"] == "tma":
            used += ch["tma_rows"] * TMA_COLS * 4 * ch["n"] + 128
    for ch in sorted(channels, key=lambda c: -c["n"]):
        if used <= 160_000:
            break
        if ch["payload"]:
            ch["payload"] = False
            used -= 4 * 512 * ch["n"]
    smem_pad = min(smem_pad, max(0, (200_000 - used) // 4))
    _extend_channels(ext, channels, roles, geom)
    for ch in channels:
        sides = [
            x
            for x, mech in (("p", ch["producer"]), ("c", ch["consumer"]))
            if mech == "threads"
            and x not in ch.get("manual", "")
            # an unclosed handle's raw signal arrives from every thread
            and not (ch.get("unchecked") == "unclosed" and ch.get("unchecked_side") == x)
        ]
        if ch["kind"] != "raw" and sides and warps_rng.random() < 0.1:
            ch["warps"] = "".join(x for x in sides if warps_rng.random() < 0.7) or sides[0]
        if ch["kind"] != "raw" and not ch.get("mma") and wait_rng.random() < 0.2:
            ch["wait"] = wait_rng.choice(["try", "peek"])
    spec = KernelSpec(
        seed=seed,
        pair=pair,
        cluster=cluster,
        roles=tuple(
            RoleSpec(
                name=r["name"],
                warps=r["warps"],
                width=r["width"],
                regs=r["regs"],
                cta=r["cta"],
                jitter=0 if big_grid else r["jitter"],
                tmem=bool(r["tmem"]),
                events=r["events"],
                with_handles=r["with_handles"],
                kind=r.get("kind", ""),
            )
            for r in roles
        ),
        channels=tuple(ChannelSpec(**ch) for ch in channels),
        turn_role=turn_role,
        turn_enabled=rng.random() < 0.85,
        stream=stream_spec,
        iters=rng.choice([0, 1, 2, 3, 5, 8, 17, 33]),
        units=units,
        tmem_regions=tmem_regions,
        smem_pad=smem_pad,
        log_cap=2 if big_grid else 32,
        item_cap=1 if big_grid else 64,
        turn_cap=2 if big_grid else 64,
        trace=rng.random() < 0.15,
        variants=_variants(rng, stream_spec, big_grid or any(r["jitter"] for r in roles)),
        graph=rng.random() < 0.2 and not (stream_spec is not None and stream_spec.record),
        mma=mma,
    )
    return _extend_kernel(ext, spec)


def _variants(rng: random.Random, s: StreamSpec | None, small_only: bool) -> tuple:
    """Runtime-shape variants: other extents (both sides of the tile-grid y/z limit when the
    base allows a large grid), caps and iteration counts."""
    out = []
    for _ in range(rng.choice([0, 0, 1, 2])):
        extents = (rng.randint(1, 16), rng.randint(1, 16), rng.randint(1, 3))
        if rng.random() < 0.1:
            extents = (1, 1, 1)
        if s is not None and s.tile_grid and not small_only and rng.random() < 0.15:
            extents = (1, 65535 + rng.randint(0, 1), 1)
        out.append((extents, rng.choice([1, 2, 3, 5, 8, 37, SMS]), rng.choice([0, 1, 2, 7, 33])))
    return tuple(out)


def variant_specs(spec: KernelSpec) -> list[KernelSpec]:
    """The spec itself, then one spec per runtime variant (same compiled kernel)."""
    out = [spec]
    for extents, max_ctas, iters in spec.variants:
        stream = spec.stream
        if stream is not None:
            stream = dataclasses.replace(stream, extents=extents, max_ctas=max_ctas)
        out.append(dataclasses.replace(spec, stream=stream, iters=iters, variants=()))
    return out


# --------------------------------------------------------------------------- #
# Source emission
# --------------------------------------------------------------------------- #


class _Src:
    def __init__(self):
        self.lines: list[str] = []
        self.depth = 0

    def __call__(self, line: str = ""):
        self.lines.append("    " * self.depth + line if line else "")

    def block(self, header: str):
        self(header)
        self.depth += 1

    def end(self, n: int = 1):
        self.depth -= n

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


_MECH = {"threads": "Threads", "umma": "Umma"}
_SCHEDULE = {
    "oneshot": "OneShot()",
    "static": "StaticPersistent()",
    "atomic": "AtomicPersistent(stages={})",
    "clc": "ClusterLaunchControl(stages={})",
    "reverse": "ReverseStatic()",
    "rbroadcast": "ReverseBroadcast(stages={})",
}
_FENCE = {
    "smem_async": "cute.arch.fence_view_async_shared()",
    "tmem_store": "cute.arch.fence_view_async_tmem_store()",
}
REASON = "fuzz: the generated program signals this edge itself"


def class_name(spec: KernelSpec) -> str:
    return f"FuzzKernel_{spec.key()}"


def kernel_name(spec: KernelSpec) -> str:
    return f"wdfuzz_{spec.key()}"


def _hvar(key) -> str:
    ch, side, k = key
    return f"h_{ch}_{side}_{'b' if k == BOUND else k}"


def warp_counts(spec: KernelSpec, ch: ChannelSpec, side: str) -> list[int]:
    """Per-slot arrivals when every warp of a side arrives once (an elected lane per warp)."""
    lst = ch.producers if side == "p" else ch.consumers
    return [
        sum(spec.role(r).width * (1 if b else spec.role(r).instances) for r, b in lst)
        for _ in range(ch.n)
    ]


def override_of(spec: KernelSpec, ch: ChannelSpec) -> dict:
    """The ``override=`` of a channel whose ``manual`` sides arrive once per warp."""
    out = {}
    for side in ch.manual:
        counts = warp_counts(spec, ch, side)
        key = "full_counts" if side == "p" else "empty_counts"
        if ch.override_form == "int" and len(set(counts)) == 1:
            out[key] = counts[0]
        elif ch.override_form == "rank":
            out[key] = {c: counts for c in range(spec.cs)}
        else:
            out[key] = counts
    return out


def uses_stream(spec: KernelSpec, r: RoleSpec) -> bool:
    """The role follows the work stream: managed roles, and raw roles under OneShot (through
    ``single_item()``). Other raw roles run ``iters`` iterations."""
    s = spec.stream
    return s is not None and (not r.raw or s.schedule == "oneshot")


def smem_buffers(spec: KernelSpec) -> list[tuple[str, str]]:
    """``(plan buffer, layout expression)`` of every SmemPlan buffer, in declaration order."""
    out = []
    for ch in spec.channels:
        if ch.storage and ch.producer == "tma":
            out.append(
                (
                    f"T_{ch.name}",
                    (
                        f"cute.make_ordered_layout(({ch.tma_rows}, {TMA_COLS}, "
                        f"{ch.n}), order=(1, 0, 2))"
                    ),
                )
            )
        elif ch.storage:
            _, width = payload_offsets(spec, ch)
            layout = f"cute.make_layout(({width}, {ch.n}))"
            out.append((f"P_{ch.name}", layout))
            if ch.storage == "overlap":
                out.append((f"Q_{ch.name}", layout))
    if spec.mma is not None:
        out += [("MA", "m_a_layout"), ("MB", "m_b_layout")]
    return out


def mma_channels(spec: KernelSpec) -> tuple[ChannelSpec, ChannelSpec]:
    chans = {c.mma: c for c in spec.channels if c.mma}
    return chans["ab"], chans["acc"]


def _role_params(spec: KernelSpec, r: RoleSpec) -> list[str]:
    params = ["self"]
    if r.raw:
        params.append("xraw: Barriers")
        if r.tmem:
            params.append("tmem: Tmem")
        params.extend(("me: Local", "ctx"))
        return params
    if r.instances > 1 or any(
        b for ch in spec.channels for rn, b in (*ch.producers, *ch.consumers) if rn == r.name
    ):
        params.append("stage: Instance")
    for ch in spec.channels:
        for side, lst in (("Producer", ch.producers), ("Consumer", ch.consumers)):
            for rname, bound in lst:
                if rname == r.name:
                    params.append(
                        f"{ch.name}: {side}[Instance]" if bound else f"{ch.name}: {side}"
                    )
    if spec.turn_role == r.name:
        params.append("ts: Turn")
    if spec.stream is not None:
        params.append("work: WorkCursor")
    if r.tmem:
        params.append("tmem: Tmem")
    params.extend(("me: Local", "ctx"))
    return params


def _by_stage(values: tuple) -> str:
    if len(set(values)) == 1:
        return str(values[0])
    return f"{tuple(values)}[stage]"


def _emit_body(src: _Src, spec: KernelSpec, r: RoleSpec, extra: bool):
    """One iteration of the role's handle program (``extra``: the final partial round)."""
    chans = {c.name: c for c in spec.channels}
    lay = Layout.of(spec)
    mma = spec.mma

    def present(key) -> str | None:
        """Python condition for the handle being used this round (None: always)."""
        ch, _, k = key
        c = chans[ch]
        if not extra:
            return None
        if c.kind != "slots":
            return "True" if k < c.extra_mult else "False"
        if k == BOUND:
            return f"stage in {tuple(c.extra_slots)!r}"
        return "True" if k in c.extra_slots else "False"

    def epoch(key) -> str:
        c = chans[key[0]]
        return f"(it * {c.mult} + {key[2]})" if c.kind != "slots" else "it"

    def slot(key) -> str:
        c = chans[key[0]]
        if c.kind != "slots":
            return "0"
        return "stage" if key[2] == BOUND else str(key[2])

    def unchecked_here(key, kind: str) -> bool:
        c = chans[key[0]]
        return c.unchecked == kind and c.unchecked_side == key[1]

    def words(key, write: bool) -> str:
        """The payload words of the handle's stage as a 1-D tensor."""
        c, h = chans[key[0]], _hvar(key)
        if c.kind == "raw":
            return f"ctx.rb_{c.name}[None, s_{c.name}_{key[1]}]"
        if c.storage == "overlap":
            if write:
                return f'{h}.view("Q", write=True)'
            if c.alias_opt_out:
                src('with unchecked("alias", reason="fuzz: ordered by the stage handoff"):')
                src(f'    v_{h} = {h}.view("P", read=True)')
                return f"v_{h}"
            return f'{h}.view("P", read=True)'
        if c.storage == "buffer":
            return f'{h}.view("P", {"write" if write else "read"}=True)'
        return f"{h}.data"

    def tile(key) -> str:
        """A TMA stage, ``(rows, cols)``, read by threads."""
        c, h = chans[key[0]], _hvar(key)
        if c.storage:
            return f'{h}.view("T", read=True)'
        return f"ctx.s_{c.name}[None, None, {h}.index]"

    def on_open(key):
        ch, side, _ = key
        c = chans[ch]
        h = _hvar(key)
        e = epoch(key)
        counter = f"k_{ch}_{side}"
        if c.kind == "raw":
            sv = f"s_{ch}_{side}"
            src(f"{sv} = {counter} % {c.n}")
            if side == "p":
                src(
                    f"cute.arch.mbarrier_wait(xraw + ({c.n} + {sv}), (({counter} // {c.n}) & 1) ^ 1)"
                )
            else:
                src(f"cute.arch.mbarrier_wait(xraw + {sv}, ({counter} // {c.n}) & 1)")
        if side == "p":
            src(f"D.jitter({e}, tid, {c.salt}, {r.jitter})")
            if c.mma == "acc":
                src("mma_t.set(tcgen05.Field.ACCUMULATE, False)")
            elif c.mma == "ab":
                for op in ("a", "b"):
                    src(
                        f"cute.copy(ctx.m_tma_{op}, ctx.m_t{op.upper()}g{op.upper()}[None, {e} % "
                        f'{c.tma_tiles}, 0], {h}.view("{op.upper()}", write=True, by="tma"), '
                        f"tma_bar_ptr={h}.barrier)"
                    )
            elif c.payload or c.kind == "raw":
                offs, _ = payload_offsets(spec, c)
                src(
                    f"D.write_words({words(key, True)}, {e}, tid, {slot(key)}, "
                    f"{_by_stage(offs[r.name])}, {c.salt})"
                )
            elif c.producer == "tma":
                mcast = f", mcast_mask=ctx.mc_{ch}" if c.multicast else ""
                bar = f"{h}.barrier"
                if unchecked_here(key, "tma"):
                    src('with unchecked("tma", reason="fuzz: opaque barrier access"):')
                    src(f"    bar_{h} = {h}.barrier")
                    bar = f"bar_{h}"
                dst = f"ctx.ts_{ch}[None, {h}.index]"
                if c.storage:
                    dst = f'{h}.view("T", write=True, by="tma")'
                # With fills, the same tile into each filled CTA (same SMEM offset).
                masks = [f", mcast_mask=ctx.fl_{ch}_{j}" for j in range(fill_count(spec, c))]
                for mask in masks or [mcast]:
                    src(
                        f"cute.copy(ctx.tma_{ch}, ctx.tg_{ch}[None, {e} % {c.tma_tiles}], {dst}, "
                        f"tma_bar_ptr={bar}{mask})"
                    )
        else:
            offs = tuple(lay.cons[(ch, r.name, i)] for i in range(r.instances))
            log = f"base + {_by_stage(offs)} + 2"
            if c.mma == "ab":
                acc = _hvar(("macc", "p", 0))
                src(f"for kb in cutlass.range_constexpr({mma.k // 16}):")
                src(
                    f"    cute.gemm(mma_t, tAcc[None, None, None, {acc}.index], "
                    f'{h}.view("A", None, None, kb, read=True, by="umma"), '
                    f'{h}.view("B", None, None, kb, read=True, by="umma"), '
                    f"tAcc[None, None, None, {acc}.index])"
                )
                src("    mma_t.set(tcgen05.Field.ACCUMULATE, True)")
            elif c.mma == "acc":
                ab = mma_channels(spec)[0]
                src(
                    "tDtC = t2r.partition_S(cute.zipped_divide(tAcc[None, None, None, "
                    f"{h}.index], ctx.m_epi_tiler))"
                )
                src(f"for i in cutlass.range_constexpr({mma.n // 32}):")
                src("    cute.copy(t2r, tDtC[None, None, i], rAcc)")
                src(
                    f"    err_{ch} = err_{ch} + D.mma_check(rAcc, tDcC[None, None, i], {e}, "
                    f"{ab.mult}, {ab.tma_tiles}, {mma.k}, {mma.salt})"
                )
            elif c.payload or c.kind == "raw":
                _, width = payload_offsets(spec, c)
                data = words(key, False)
                src(f"d_{h} = {data}")
                src(
                    f"err_{ch} = err_{ch} + D.check_words(d_{h}, {e}, tid, {slot(key)}, "
                    f"{width}, {r.threads}, {c.salt})"
                )
                src(
                    f"D.put_capped(ctx.clog, {log}, {counter}, {spec.log_cap}, d_{h}[0] >> 12, lead)"
                )
            elif c.tiles_checked:
                src(f"t_{h} = {tile(key)}")
                src(
                    f"err_{ch} = err_{ch} + D.check_tile(t_{h}, {e} % {c.tma_tiles}, tid, "
                    f"{c.tma_rows}, {TMA_COLS}, {r.threads}, {c.salt})"
                )
                src(
                    f"D.put_capped(ctx.clog, {log}, {counter}, {spec.log_cap}, t_{h}[0, 0] >> 12, "
                    "lead)"
                )
        src(f"{counter} = {counter} + 1")

    def on_close(key):
        ch, side, _ = key
        c = chans[ch]
        if side == "c":
            src(f"D.jitter({epoch(key)}, tid, {c.salt + 1}, {r.jitter})")
            if c.payload or c.kind == "raw":
                _, width = payload_offsets(spec, c)
                src(
                    f"err_{ch} = err_{ch} + D.check_words({words(key, False)}, {epoch(key)}, tid, "
                    f"{slot(key)}, {width}, {r.threads}, {c.salt})"
                )
            elif c.tiles_checked:
                src(
                    f"err_{ch} = err_{ch} + D.check_tile({tile(key)}, {epoch(key)} % "
                    f"{c.tma_tiles}, tid, {c.tma_rows}, {TMA_COLS}, {r.threads}, {c.salt})"
                )

    def open_args(key) -> str:
        """The acquire/wait arguments, after emitting the channel's probe or preview."""
        ch, side, k = key
        c = chans[ch]
        arg = "" if c.kind == "ring" else ("stage" if k == BOUND else str(k))
        if c.wait == "try":
            probe = "try_acquire" if side == "p" else "try_wait"
            src(f"tok_{_hvar(key)} = {ch}.{probe}({arg})")
            return f"{arg}, token=tok_{_hvar(key)}" if arg else f"token=tok_{_hvar(key)}"
        if c.wait == "peek" and side == "c":
            src(f"{ch}.peek({arg})")
        return arg

    def open_stmt(key):
        ch, side, _k = key
        c = chans[ch]
        if c.kind == "raw":
            return
        op = "acquire" if side == "p" else "wait"
        arg = open_args(key)
        if unchecked_here(key, "unclosed"):
            src('with unchecked("unclosed", reason="fuzz: signalled through the raw barrier"):')
            src(f"    {_hvar(key)} = {ch}.{op}({arg})")
        else:
            src(f"{_hvar(key)} = {ch}.{op}({arg})")

    def close_stmt(key):
        ch, side, _ = key
        c = chans[ch]
        h = _hvar(key)
        fence = c.commit_fence if side == "p" else c.release_fence
        bar = f"{h}.full_barrier" if side == "p" else f"{h}.empty_barrier"
        close = "commit" if side == "p" else "release"
        if c.kind == "raw":
            target = f"xraw + s_{ch}_p" if side == "p" else f"xraw + ({c.n} + s_{ch}_c)"
            src(f"cute.arch.mbarrier_arrive({target})")
        elif side in c.manual:
            if fence:
                src(_FENCE[fence])
            src(f"D.warp_arrive({bar})")
            src(f"{h}.{close}(signal=False)")
        elif unchecked_here(key, "unclosed"):
            if c.producer == "tma" and side == "p":
                return  # the TMA transaction completes the stage; no close, no arrival
            if fence:
                src(_FENCE[fence])
            src(f"cute.arch.mbarrier_arrive({bar})")
        elif unchecked_here(key, "cross_region"):
            src("if tid >= 0:")
            src('    with unchecked("cross_region", reason="fuzz: closed in a uniform branch"):')
            src(f"        {h}.{close}()")
        else:
            src(f"{h}.{close}()")

    def with_ok(key) -> bool:
        c = chans[key[0]]
        return not c.manual and c.unchecked not in ("unclosed", "cross_region")

    withs = {k for k in r.with_handles if with_ok(k)} if not extra else set()
    stack = []
    emitted = False
    for op, ch, side, k in r.events:
        key = (ch, side, k)
        cond = present(key)
        if cond == "False":
            continue
        guarded = cond not in (None, "True")
        if guarded:
            src.block(f"if cutlass.const_expr({cond}):")
        if op == "o":
            if key in withs:
                verb = "produce" if side == "p" else "consume"
                src.block(f"with {ch}.{verb}({open_args(key)}) as {_hvar(key)}:")
                stack.append(key)
                on_open(key)
            else:
                open_stmt(key)
                on_open(key)
        else:
            on_close(key)
            if key in withs:
                assert stack and stack[-1] == key, (key, stack)
                stack.pop()
                src.end()
            else:
                close_stmt(key)
        if guarded:
            src.end()
        emitted = True
    if not emitted:
        src("pass")


def _emit_role_setup(src: _Src, spec: KernelSpec, r: RoleSpec):
    """Per-role values the MMA subgraph and storage-bound channels need before the loop."""
    if r.kind == "mma_load":
        src("mab = mab.views(A=ctx.m_tAsA, B=ctx.m_tBsB)")
    elif r.kind == "mma":
        src("mab = mab.views(A=ctx.m_tCrA, B=ctx.m_tCrB)")
        src("mma_t = ctx.m_tiled")
        src('tAcc = cute.make_tensor(tmem.ptr("mma_acc"), ctx.m_acc_layout)')
    elif r.kind == "mma_epi":
        src(
            "t2r = tcgen05.make_tmem_copy(cute.make_copy_atom(tcgen05.Ld32x32bOp("
            "tcgen05.Repetition.x32), cutlass.Float32), cute.zipped_divide(ctx.m_acc_fake, "
            "ctx.m_epi_tiler)[None, 0]).get_slice(tid)"
        )
        src('tAcc = cute.make_tensor(tmem.ptr("mma_acc"), ctx.m_acc_layout)')
        src("tDcC = t2r.partition_D(cute.zipped_divide(ctx.m_tCcC, ctx.m_epi_tiler))")
        src("rAcc = cute.make_rmem_tensor(tDcC[None, None, 0].shape, cutlass.Float32)")
    for ch in spec.channels:
        if ch.storage and ch.producer == "tma" and any(rn == r.name for rn, _ in ch.producers):
            src(f"{ch.name} = {ch.name}.views(T=StageView(ctx.ts_{ch.name}, axis=1))")


def _emit_role(src: _Src, spec: KernelSpec, r: RoleSpec):
    lay = Layout.of(spec)
    regs = "" if r.regs is None else f", regs={r.regs}"
    cta = "" if r.cta == "all" else f', cta="{r.cta}"'
    raw = ", raw=True" if r.raw else ""
    warps = f"[{', '.join(f'range({w}, {w + r.width})' for w in r.warps)}]"
    src(f"@role(warps={warps}{regs}{cta}{raw})")
    src("@cute.jit")
    src.block(f"def {r.name}({', '.join(_role_params(spec, r))}):")
    src("tid = me.tid")
    src("lead = tid == 0")
    src("rank = self.cluster.rank()")
    src(f"base = D.cta_id() * {lay.per_cta}")
    codes = tuple(presence_code(spec, r.name, i) for i in range(r.instances))
    src(f"D.mark_warp(ctx.clog, base + {lay.presence}, {_by_stage(codes)})")
    _emit_role_setup(src, spec, r)
    src("it = Int32(0)")
    counters = []
    for ch in spec.channels:
        for side, lst in (("p", ch.producers), ("c", ch.consumers)):
            if any(rn == r.name for rn, _ in lst):
                counters.append((ch.name, side))
                src(f"k_{ch.name}_{side} = Int32(0)")
                if side == "c":
                    src(f"err_{ch.name} = Int32(0)")
    ordinal = _by_stage(tuple(lay.instance_ids[(r.name, i)] for i in range(r.instances)))

    def iteration():
        _emit_body(src, spec, r, extra=False)
        if spec.turn_role == r.name:
            src.block("with ts.turn():")
            if spec.turn_enabled:
                src(f"D.turn_log(ctx.clog, base + {lay.turn}, stage, {spec.turn_cap}, lead)")
            else:
                src("pass")
            src.end()
        src("it = it + 1")

    s = spec.stream
    if not uses_stream(spec, r):
        src.block("for _ in cutlass.range(ctx.iters, unroll=1):")
        iteration()
        src.end()
    elif r.raw:  # OneShot: this unit's one item, as a managed cursor would start
        items = _by_stage(tuple(lay.role_items[(r.name, i)] for i in range(r.instances)))
        base_r, spread, a, b, c = s.rounds
        src("valid, (m, n, l) = self.work.single_item()")
        src("rounds = Int32(0)")
        src.block("if valid:")
        src(f"rounds = (Int32(m) * {a} + Int32(n) * {b} + Int32(l) * {c}) % {spread} + {base_r}")
        src.end()
        src(f"D.log_raw_item(ctx.clog, base + {items}, {spec.item_cap}, valid, (m, n, l), lead)")
        src.block("for _ in cutlass.range(rounds, unroll=1):")
        iteration()
        src.end()
    else:
        items = _by_stage(tuple(lay.role_items[(r.name, i)] for i in range(r.instances)))
        base_r, spread, a, b, c = s.rounds
        src("nitems = Int32(0)")
        src.block("while work.valid:")
        src("m, n, l = work.item")
        src(f"hit = ({ordinal} * {spec.cs} + rank) * ctx.total + work.index")
        src(
            f"D.log_item(ctx.clog, ctx.hits, base + {items} + 1, nitems, {spec.item_cap}, "
            "work.index, (m, n, l), hit, lead)"
        )
        src("nitems = nitems + 1")
        src(f"rounds = (Int32(m) * {a} + Int32(n) * {b} + Int32(l) * {c}) % {spread} + {base_r}")
        src.block("for _ in cutlass.range(rounds, unroll=1):")
        iteration()
        src.end()
        src("work.advance()")
        src.end()
        src(f"D.put(ctx.clog, base + {items}, nitems, lead)")
    _emit_body(src, spec, r, extra=True)
    it_off = _by_stage(tuple(lay.role_it[(r.name, i)] for i in range(r.instances)))
    src(f"D.put(ctx.clog, base + {it_off}, it, lead)")
    for ch, side in counters:
        table = lay.cons if side == "c" else lay.prod
        off = _by_stage(tuple(table[(ch, r.name, i)] for i in range(r.instances)))
        src(f"D.put(ctx.clog, base + {off}, k_{ch}_{side}, lead)")
        if side == "c":
            src(f"D.add(ctx.clog, base + {off} + 1, err_{ch})")
    returned = [] if r.raw else [ch for ch, _ in counters]
    if spec.turn_role == r.name:
        returned.append("ts")
    if s is not None and not r.raw:
        returned.append("work")
    if returned:
        src(f"return {', '.join(returned)}")
    src.end()
    src()


def call_args(spec: KernelSpec) -> list[tuple[str, str]]:
    """``(name, kind)`` of the entry point's tensor arguments after ``max_ctas`` (before the
    stream). Kinds: ``tma_src``, ``mma_a``, ``mma_b``, ``rows`` (int32 (rows, 3)), ``vec``
    (int32 vector), ``int`` (Int32)."""
    out = [
        (f"src_{ch.name}", "tma_src")
        for ch in spec.channels
        if ch.producer == "tma" and not ch.mma
    ]
    if spec.mma is not None:
        out += [("mma_a", "mma_a"), ("mma_b", "mma_b")]
    s = spec.stream
    if s is not None and s.mapping == "table":
        out += [("table", "rows"), ("bad", "vec")]
    elif s is not None and s.mapping == "packed":
        out += [("table", "rows"), ("length", "vec")]
        if s.pdl:
            out += [("stage", "rows"), ("plen", "int")]
    return out


_PREAMBLE_SCHEDULES = '''
@cute.jit
def _countdown(index, problem):
    """``index`` while it is a valid index, else the terminal index ``total``."""
    return cutlass.select_(index >= 0, index, problem.total)


@dataclass(frozen=True)
class ReverseStatic(IndexSchedule):
    """Static persistent from the back: unit ``b`` runs ``total-1-b, total-1-b-U, ...``."""

    def first_index(self, problem):
        return _countdown(problem.total - 1 - problem.first_unit(), problem)

    def next_index(self, index, problem):
        return _countdown(index - problem.units(), problem)


@dataclass(frozen=True)
class ReverseBroadcast(IndexSchedule):
    """The same sequence, claimed on the scheduler warp and broadcast."""

    stages: int = 2
    broadcast = True

    def first_index(self, problem):
        return _countdown(problem.total - 1 - problem.first_unit(), problem)

    def claim_next(self, fetch, problem, index):
        return _countdown(index - problem.units(), problem)

'''

_PREAMBLE_TABLE = '''
@dataclass(frozen=True)
class TableMap(WorkMap):
    """Rows of an int32 ``(max(total, 1), 3)`` table; no ``decodes_any_index``, so a decode
    outside ``[0, total)`` (R14) counts in ``bad[0]``."""

    words = 3

    def bind(self, space):
        table, bad, em, en, el = space
        nonempty = (em > 0) & (en > 0) & (el > 0)
        return table, bad, cutlass.select_(nonempty, em * en * el, Int32(0))

    def count64(self, space):
        return Int64(space[2])

    def decode(self, index, space, state):
        table, bad, total = space
        return True, D.table_row(table, bad, index, total), state

    def get_key(self):
        return "fuzztable"

'''


def emit(spec: KernelSpec) -> str:
    """Python source of ``spec``'s kernel class."""
    src = _Src()
    src(f'"""Generated by test/warpdrive_fuzz/gen.py: seed {spec.seed}, key {spec.key()}."""')
    src()
    src("from dataclasses import dataclass")
    src("from types import SimpleNamespace")
    src()
    src("import cutlass")
    src("import cutlass.utils.blackwell_helpers as sm100_utils")
    src("from cuda.bindings import driver as cuda")
    src("from cutlass import Int32, Int64, cute")
    src("from cutlass.cute.nvgpu import cpasync, tcgen05")
    src()
    src("from warpdrive_fuzz import runtime_device as D")
    src("from attn_gym.warpdrive import (")
    for name in (
        "AtomicPersistent",
        "Barriers",
        "Channel",
        "Cluster",
        "ClusterLaunchControl",
        "Consumer",
        "CtaPair",
        "IndexSchedule",
        "Instance",
        "Local",
        "OneShot",
        "PdlWait",
        "Producer",
        "Raster",
        "Ring",
        "Slots",
        "SmemPlan",
        "StageView",
        "StaticPersistent",
        "Threads",
        "Tma",
        "Tmem",
        "TmemPlan",
        "Turn",
        "Turnstile",
        "Umma",
        "Warps",
        "WarpSpecialized",
        "WorkCursor",
        "WorkMap",
        "WorkStream",
        "role",
        "unchecked",
    ):
        src(f"    {name},")
    src(")")
    src("from attn_gym.warpdrive.examples.packed_work import PackedWork")
    src()
    s = spec.stream
    if s is not None and s.mapping == "raster" and s.skip is not None:
        a, b, c, k = s.skip
        src()
        src("@dataclass(frozen=True)")
        src.block("class SkipMod(Raster):")
        src("skips = True")
        src()
        src.block("def decode(self, index, space, state):")
        src("_, (m, n, l), state = super().decode(index, space, state)")
        src(f"return (m * {a} + n * {b} + l * {c}) % {k} != 0, (m, n, l), state")
        src.end()
        src()
        src.block("def get_key(self):")
        src(f'return "skipmod{a}{b}{c}{k}_" + super().get_key()')
        src.end(2)
        src()
    if s is not None and s.schedule in ("reverse", "rbroadcast"):
        src.lines += _PREAMBLE_SCHEDULES.splitlines()
    if s is not None and s.mapping == "table":
        src.lines += _PREAMBLE_TABLE.splitlines()
    src()
    src.block(f"class {class_name(spec)}(WarpSpecialized):")
    src.block("def __init__(self):")
    if spec.cluster:
        if spec.geom.emit():
            src(f"self.cluster = {spec.geom.emit()}")
    elif spec.pair:
        src("self.cluster = CtaPair()")
    plan = smem_buffers(spec)
    if plan:
        src("self.smem_plan = SmemPlan()")
        for name, _ in plan:
            if name in ("MA", "MB"):
                src(
                    f'{name.lower()} = self.smem_plan.buffer("{name}", cutlass.BFloat16, align=1024)'
                )
            elif name.startswith("Q_"):
                root = "P_" + name[2:]
                src(
                    f'{name.lower()} = self.smem_plan.buffer("{name}", Int32, '
                    f"overlaps={root.lower()})"
                )
            elif name.startswith("T_"):
                src(f'{name.lower()} = self.smem_plan.buffer("{name}", Int32, align=128)')
            else:
                src(f'{name.lower()} = self.smem_plan.buffer("{name}", Int32)')
    for ch in spec.channels:
        if ch.kind == "raw":
            p_threads = spec.role(ch.producers[0][0]).threads
            c_threads = spec.role(ch.consumers[0][0]).threads
            counts = (
                f", counts={[p_threads] * ch.n + [c_threads] * ch.n}"
                if ch.raw_init == "auto"
                else ""
            )
            src(f"self.{ch.name} = Barriers({2 * ch.n}{counts})")
            continue
        cls = "Channel" if ch.generic else "Ring" if ch.kind == "ring" else "Slots"
        if ch.mma == "ab":
            nbytes = (MMA_M + spec.mma.n) * spec.mma.k * 2
            prod = f"Tma({nbytes})"
        else:
            mcast = f', multicast="{ch.multicast}"' if ch.multicast else ""
            mcast += f', fills="{ch.fills}"' if ch.fills else ""
            prod = f"Tma({ch.tma_bytes}{mcast})" if ch.producer == "tma" else _MECH[ch.producer]
        prod = "Warps" if "p" in ch.warps else prod
        args = [str(ch.n), prod, "Warps" if "c" in ch.warps else _MECH[ch.consumer]]
        if ch.mma == "ab":
            args.append('buffer={"A": ma, "B": mb}')
        elif ch.storage and ch.producer == "tma":
            args.append(f'buffer={{"T": t_{ch.name}}}')
        elif ch.storage == "overlap":
            args.append(f'buffer={{"P": p_{ch.name}, "Q": q_{ch.name}}}')
        elif ch.storage:
            args.append(f'buffer={{"P": p_{ch.name}}}')
        elif ch.payload:
            _, width = payload_offsets(spec, ch)
            args.append(f"payload=(Int32, ({width},))")
        if ch.commit_fence:
            args.append(f'commit_fence="{ch.commit_fence}"')
        if ch.release_fence and not ch.implicit_release_fence:
            args.append(f'release_fence="{ch.release_fence}"')
        if ch.remote_scope:
            args.append(f'remote_scope="{ch.remote_scope}"')
        if ch.manual:
            args.append(f"override={override_of(spec, ch)!r}")
        src(f"self.{ch.name} = {cls}({', '.join(args)})")
    if spec.turn_role is not None:
        src(f"self.ts = Turnstile(enabled={spec.turn_enabled})")
    if spec.tmem_regions:
        tplan = "TmemPlan()" + "".join(
            f'.region("{n}", {c}, {k})' for n, c, k in spec.tmem_regions
        )
        src(f"self.tmem_plan = {tplan}")
    if s is not None:
        if s.mapping == "table":
            mapping = "TableMap()"
        elif s.mapping == "packed":
            mapping = "PackedWork(words=3)"
        else:
            mapping = "SkipMod" if s.skip is not None else "Raster"
            mapping += f'(along="{s.along}", swizzle={s.swizzle}, reverse={tuple(s.reverse)!r})'
        schedule = _SCHEDULE[s.schedule].format(s.stages)
        extra = f", scheduler_warp={s.scheduler_warp}" if s.scheduler_warp is not None else ""
        ready = ", ready=PdlWait()" if s.ready else ""
        record = f", record={s.record}" if s.record else ""
        src(
            f"self.work = WorkStream(mapping={mapping}, schedule={schedule}{ready}{extra}{record})"
        )
    src.end()
    src()
    src.block("def get_name(self):")
    src(f'return "{kernel_name(spec)}"')
    src.end()
    src()
    for r in spec.roles:
        _emit_role(src, spec, r)

    tma = [ch for ch in spec.channels if ch.producer == "tma" and not ch.mma]
    kargs = ["self", "clog: cute.Tensor", "hits: cute.Tensor", "iters: Int32"]
    for ch in tma:
        kargs += [f"tma_{ch.name}", f"tg_{ch.name}"]
    if spec.mma is not None:
        kargs += ["m_tiled", "m_tma_a", "m_mA", "m_tma_b", "m_mB", "m_a_layout", "m_b_layout"]
    if s is not None:
        kargs.append("bound")
    src("@cute.kernel")
    src.block(f"def kernel({', '.join(kargs)}):")
    layouts = ", ".join(f'"{name}": {layout}' for name, layout in plan)
    begin = ["work=bound"] if s is not None else []
    if plan:
        begin.append(f"layouts={{{layouts}}}")
    src(f"smem = self.begin_roles({', '.join(begin)})")
    ctx = ["clog=clog", "hits=hits", "iters=iters"]
    ctx.append("total=bound.total" if s is not None else "total=Int32(0)")
    for ch in spec.channels:
        if ch.kind == "raw":
            p_threads = spec.role(ch.producers[0][0]).threads
            c_threads = spec.role(ch.consumers[0][0]).threads
            src(
                f"rb_{ch.name} = smem.allocate_tensor(Int32, cute.make_layout(({p_threads}, {ch.n})), 16)"
            )
            ctx.append(f"rb_{ch.name}=rb_{ch.name}")
            if ch.raw_init == "manual":  # Barriers(counts=None): before run_roles
                src(f'xbars = self.barrier_ptr("{ch.name}")')
                src.block("if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == 0:")
                src.block("with cute.arch.elect_one():")
                src.block(f"for i in cutlass.range_constexpr({ch.n}):")
                src(f"cute.arch.mbarrier_init(xbars + i, {p_threads})")
                src(f"cute.arch.mbarrier_init(xbars + ({ch.n} + i), {c_threads})")
                src.end(3)
    for ch in tma:
        c = ch.name
        if ch.storage:
            src(f's_{c} = self.buffer("T_{c}")')
        else:
            src(
                f"s_{c} = smem.allocate_tensor(Int32, cute.make_ordered_layout(({ch.tma_rows}, "
                f"{TMA_COLS}, {ch.n}), order=(1, 0, 2)), 128)"
            )
        coord, layout = "0", "cute.make_layout(1)"
        if ch.multicast and not ch.fills:
            src(f'mc_{c} = self.cluster.multicast("{ch.multicast}")')
            coord, layout = f"mc_{c}.coord", f"mc_{c}.layout"
            ctx.append(f"mc_{c}=mc_{c}.mask")
        if ch.fills:  # the filler's copy masks, one per CTA it fills
            first = "0" if ch.fills == "cluster" else f"fr_{c} - fr_{c} % 2"
            if ch.fills == "pair":
                src(f"fr_{c} = self.cluster.rank()")
            mode = f', "{ch.multicast}"' if ch.multicast else ""
            for j in range(fill_count(spec, ch)):
                src(f"fl_{c}_{j} = self.cluster.fill({first} + {j}{mode})")
                ctx.append(f"fl_{c}_{j}=fl_{c}_{j}.mask")
            if ch.multicast:  # both CTAs of a pair sit at one position along each mode
                coord, layout = f"fl_{c}_0.coord", f"fl_{c}_0.layout"
        src(
            f"ts_{c}, tgp_{c} = cpasync.tma_partition(tma_{c}, {coord}, {layout}, "
            f"cute.group_modes(s_{c}, 0, 2), cute.group_modes(tg_{c}, 0, 2))"
        )
        ctx += [f"tma_{c}=tma_{c}", f"tg_{c}=tgp_{c}", f"ts_{c}=ts_{c}", f"s_{c}=s_{c}"]
    if spec.mma is not None:
        m = spec.mma
        tile = f"({MMA_M}, {m.n}, {m.k})"
        acc_n = mma_channels(spec)[1].n
        src('m_sA, m_sB = self.buffer("MA"), self.buffer("MB")')
        src("m_thr = m_tiled.get_slice(0)")
        src(f"m_gA = cute.local_tile(m_mA, {tile}, (None, None, None), proj=(1, None, 1))")
        src(f"m_gB = cute.local_tile(m_mB, {tile}, (None, None, None), proj=(None, 1, 1))")
        for op, part in (("A", "partition_A"), ("B", "partition_B")):
            src(
                f"m_t{op}s{op}, m_t{op}g{op} = cpasync.tma_partition(m_tma_{op.lower()}, 0, "
                f"cute.make_layout(1), cute.group_modes(m_s{op}, 0, 3), "
                f"cute.group_modes(m_thr.{part}(m_g{op}), 0, 3))"
            )
        src(f"m_acc_shape = m_tiled.partition_shape_C(({MMA_M}, {m.n}))")
        src("m_acc_fake = m_tiled.make_fragment_C(m_acc_shape)")
        src(f"m_acc_staged = m_tiled.make_fragment_C(cute.append(m_acc_shape, {acc_n}))")
        src(f"m_tCcC = m_thr.partition_C(cute.make_identity_tensor(({MMA_M}, {m.n})))")
        ctx += [
            "m_tiled=m_tiled",
            "m_tma_a=m_tma_a",
            "m_tma_b=m_tma_b",
            "m_tAgA=m_tAgA",
            "m_tBgB=m_tBgB",
            "m_tAsA=StageView(m_tAsA, axis=1)",
            "m_tBsB=StageView(m_tBsB, axis=1)",
            "m_tCrA=StageView(m_tiled.make_fragment_A(m_sA), axis=3)",
            "m_tCrB=StageView(m_tiled.make_fragment_B(m_sB), axis=3)",
            "m_acc_fake=m_acc_fake",
            "m_acc_layout=m_acc_staged.layout",
            (
                f"m_epi_tiler=((cute.size(m_acc_fake, mode=[0, 0]), "
                f"cute.size(m_acc_fake, mode=[0, 1]) // {m.n // 32}),)"
            ),
            "m_tCcC=m_tCcC",
        ]
    if spec.smem_pad:
        src(f"smem.allocate_array(Int32, {spec.smem_pad})")
    src(f"self.run_roles(SimpleNamespace({', '.join(ctx)}))")
    src.end()
    src()
    if s is not None and s.pdl:
        src("@cute.kernel")
        src.block(
            "def produce(self, table: cute.Tensor, length: cute.Tensor, stage: cute.Tensor, "
            "plen: Int32):"
        )
        src("D.produce_rows(table, length, stage, plen)")
        src.end()
        src()
    cargs = [
        "self",
        "clog: cute.Tensor",
        "hits: cute.Tensor",
        "ws: cute.Tensor",
        "iters: Int32",
        "em: Int32",
        "en: Int32",
        "el: Int32",
        "max_ctas: Int32",
    ]
    cargs += [
        f"{name}: {'Int32' if kind == 'int' else 'cute.Tensor'}" for name, kind in call_args(spec)
    ]
    cargs.append("stream: cuda.CUstream")
    src("@cute.jit")
    src.block(f"def __call__({', '.join(cargs)}):")
    launch_args = ["clog", "hits", "iters"]
    for ch in tma:
        c = ch.name
        src(
            f"layout_{c} = cute.make_ordered_layout(({ch.tma_rows}, {TMA_COLS}, {ch.n}), "
            "order=(1, 0, 2))"
        )
        group = "TWO" if ch.tma_pair or ch.fill_two else "ONE"
        op, sharing = "CopyBulkTensorTileG2SOp", ""
        if ch.multicast:
            op = "CopyBulkTensorTileG2SMulticastOp"
            sharing = f", num_multicast={len(spec.geom.reach(0, (ch.multicast,)))}"
        if ch.fills:  # copies into other CTAs take a mask
            op = "CopyBulkTensorTileG2SMulticastOp"
        src(
            f"tma_{c}, tgt_{c} = cpasync.make_tiled_tma_atom(cpasync.{op}("
            f"tcgen05.CtaGroup.{group}), src_{c}, cute.select(layout_{c}, mode=[0, 1]), "
            f"({ch.tma_rows}, {TMA_COLS}){sharing})"
        )
        src(f"tiles_{c} = cute.local_tile(tgt_{c}, ({ch.tma_rows}, {TMA_COLS}), (None, 0))")
        launch_args += [f"tma_{c}", f"tiles_{c}"]
    if spec.mma is not None:
        m = spec.mma
        tile = f"({MMA_M}, {m.n}, {m.k})"
        ab_n = mma_channels(spec)[0].n
        src(
            "m_tiled = cute.make_tiled_mma(tcgen05.MmaF16BF16Op(cutlass.BFloat16, "
            f"cutlass.Float32, ({MMA_M}, {m.n}, 16), tcgen05.CtaGroup.ONE, "
            "tcgen05.OperandSource.SMEM, cute.nvgpu.OperandMajorMode.K, "
            "cute.nvgpu.OperandMajorMode.K))"
        )
        src(
            f"m_a_layout = sm100_utils.make_smem_layout_a(m_tiled, {tile}, cutlass.BFloat16, {ab_n})"
        )
        src(
            f"m_b_layout = sm100_utils.make_smem_layout_b(m_tiled, {tile}, cutlass.BFloat16, {ab_n})"
        )
        src("m_tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)")
        for op in ("a", "b"):
            src(
                f"m_tma_{op}, m_m{op.upper()} = cute.nvgpu.make_tiled_tma_atom_{op.upper()}("
                f"m_tma_op, mma_{op}, cute.select(m_{op}_layout, mode=[0, 1, 2]), {tile}, m_tiled)"
            )
        launch_args += [
            "m_tiled",
            "m_tma_a",
            "m_mA",
            "m_tma_b",
            "m_mB",
            "m_a_layout",
            "m_b_layout",
        ]
    pdl = ""
    if s is not None:
        if s.mapping == "table":
            space = "(table, bad, em, en, el)"
        elif s.mapping == "packed":
            space = "(table, length)"
        else:
            space = "(em, en, el)"
        if s.pdl:
            src(f'self.produce.set_name_prefix("{kernel_name(spec)}_produce")')
            src(
                "self.produce(table, length, stage, plen).launch(grid=(1, 1, 1), "
                f"block=({PRODUCER_THREADS}, 1, 1), stream=stream)"
            )
            pdl = ", use_pdl=True"
        src(f"bound = self.work.bind({space}, ws, max_ctas)")
        launch_args.append("bound")
        cfg = "**self.launch_config(bound)"
        if s.launch_1d:
            cfg = f"**self.launch_config(grid=(cutlass.max(bound.total, 1) * {spec.cs}, 1, 1))"
    else:
        cfg = f"**self.launch_config(grid=({spec.units * spec.cs}, 1, 1))"
    src(f"self.kernel({', '.join(launch_args)}).launch({cfg}, stream=stream{pdl})")
    src.end(2)
    return src.text()


def fill_count(spec: KernelSpec, ch: ChannelSpec) -> int:
    """CTAs a ``Tma(fills=...)`` producer copies into (0 without fills)."""
    if not ch.fills:
        return 0
    return spec.geom.size if ch.fills == "cluster" else 2


def features(spec: KernelSpec) -> set[str]:
    """Knob classes a spec exercises (for campaign coverage summaries)."""
    out = {f"roles{len(spec.roles)}"}
    out.add(f"cluster_{spec.geom.key()}" if spec.cluster else "pair" if spec.pair else "single")
    s = spec.stream
    out.add(f"schedule_{s.schedule}" if s else "no_stream")
    if s is not None:
        if s.skip:
            out.add("skip")
        if s.ready:
            out.add("ready")
        if s.total > 65535 or max(s.extents[1:]) >= 65535:
            out.add("grid_yz_edge")
        if s.scheduler_warp is not None:
            out.add("explicit_scheduler_warp")
        if s.schedule in ("static", "atomic") and s.max_ctas < SMS:
            out.add("capped_grid")
    if spec.smem_pad:
        out.add("one_cta_per_sm")
    if spec.trace:
        out.add("trace_protocol")
    if spec.variants:
        out.add("runtime_shape_variants")
    if spec.graph:
        out.add("cuda_graph_replay")
    if s is not None and s.launch_1d:
        out.add("oneshot_1d_launch")
    if s is not None and s.record:
        out.add("record")
    if s is not None and s.mapping != "raster":
        out.add(f"mapping_{s.mapping}")
    if s is not None and s.pdl:
        out.add("pdl_producer")
    if spec.iket:
        out.add("iket" if spec.iket == 2 else "iket_nowait")
    if spec.alias_check != "warn":
        out.add(f"alias_{spec.alias_check}")
    if spec.concurrent:
        out.add("concurrent_streams")
    if spec.mma is not None:
        out.add("tcgen05_mma")
    for r in spec.roles:
        if r.raw:
            out.add("raw_role_tmem" if r.tmem else "raw_role")
            if s is not None and s.schedule == "oneshot":
                out.add("raw_single_item")
    if spec.turn_role:
        out.add("turnstile" if spec.turn_enabled else "turnstile_disabled")
    if spec.tmem_regions:
        out.add("tmem")
    if any(r.regs is not None for r in spec.roles):
        out.add("regs")
    owned = {w for r in spec.roles for rng in r.warp_ranges() for w in rng}
    if len(owned) != max(owned) + 1:
        out.add("warp_holes")
    if any(r.instances > 1 for r in spec.roles):
        out.add("instances")
    if any(r.cta == "leader" for r in spec.roles):
        out.add("leader_roles")
    if any(r.cta == "rank0" for r in spec.roles):
        out.add("rank0_roles")
    if any(r.with_handles for r in spec.roles):
        out.add("with_blocks")
    for ch in spec.channels:
        out.add(f"{ch.kind}_{ch.producer}_{ch.consumer}")
        out.add(f"depth{ch.n}")
        if ch.mult > 1:
            out.add("overlapping_ring_handles")
        if len(ch.producers) > 1:
            out.add("fan_in")
        if len(ch.consumers) > 1:
            out.add("join")
        if any(b for _, b in (*ch.producers, *ch.consumers)):
            out.add("bound_slots")
        if spec.cs > 1 and not is_local(spec, ch):
            out.add("pair_scoped")
        if ch.payload:
            out.add("payload")
        if ch.tma_pair:
            out.add("tma_pair")
        if ch.fills:
            out.add(f"tma_fills_{ch.fills}" + ("_two" if ch.fill_two else ""))
            if ch.multicast:
                out.add("tma_fills_multicast")
        if ch.multicast:
            out.add("tma_multicast")
        if ch.kind == "slots" and ch.n > 4:
            out.add("slots_many")
        if ch.kind == "raw":
            out.add(f"barriers_{ch.raw_init}")
        if ch.manual:
            out.add("override_signal_false")
            out.add(f"override_{ch.override_form}_{ch.manual}")
        if ch.warps:
            out.add(f"warps_{ch.warps}")
            if any(max(warp_counts(spec, ch, side)) > 1 for side in ch.warps):
                out.add("warps_join")
        if ch.generic:
            out.add("channel_class")
        if ch.unchecked:
            out.add(f"unchecked_{ch.unchecked}")
        if ch.storage:
            out.add(f"storage_{ch.storage}_{'tma' if ch.producer == 'tma' else 'threads'}")
        if ch.alias_opt_out:
            out.add("unchecked_alias")
        if ch.wait:
            out.add(f"wait_{ch.wait}")
    return out
