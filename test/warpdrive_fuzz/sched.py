"""Host-side scheduler and derived-count properties, checked against independent models.

* Raster decoding equals an explicit enumeration of the documented visiting order (not only a
  bijection), over random extents, orders and the 1-D fallback boundary.
* Launch grids (``BoundWork.grid``, ``tile_grid_dims``) equal the R12/R13 formulas.
* Every channel scope of a generated kernel (per-rank, per-slot full/empty counts, target CTAs,
  commit masks, armed TMA bytes, tails), the scheduler's warp and work-ring counts, the
  warpgroup register budgets and padding, and the barrier initialization counts equal a
  recomputation from the declared graph per R4, R9, R11 and R12. The recomputation reads only
  the spec, never ``slot_counts``/``derive_scope``.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from warpdrive_fuzz import gen, oracle

# --------------------------------------------------------------------------- #
# Raster and launch grids
# --------------------------------------------------------------------------- #


def random_stream(rng: random.Random) -> gen.StreamSpec:
    em, en, el = rng.randint(1, 40), rng.randint(1, 40), rng.randint(1, 4)
    if rng.random() < 0.1:
        em, en, el = rng.choice([(1, 1, 1), (1, 9, 1), (9, 1, 1), (1, 1, 5)])
    return gen.StreamSpec(
        schedule=rng.choice(["oneshot", "static", "atomic", "clc"]),
        stages=rng.randint(1, 4),
        along=rng.choice("mn"),
        swizzle=rng.choice([1, 2, 3, 4, 5, 8, 13, 64]),
        reverse=tuple(a for a in "mnl" if rng.random() < 0.3),
        skip=None,
        ready=False,
        extents=(em, en, el),
        max_ctas=rng.choice([1, 2, 3, 7, 148, 1000]),
        scheduler_warp=None,
        rounds=(1, 1, 0, 0, 0),
    )


def raster_of(s: gen.StreamSpec):
    from attn_gym.warpdrive import Raster

    return Raster(along=s.along, swizzle=s.swizzle, reverse=s.reverse)


def check_raster(s: gen.StreamSpec) -> list[str]:
    order = raster_of(s)
    want = oracle.raster_order(s)
    got = [tuple(order.decode(i, s.extents, None)[1]) for i in range(s.total)]
    if got != want:
        bad = next(i for i, (a, b) in enumerate(zip(got, want, strict=True)) if a != b)
        return [
            f"{order.describe()} over {s.extents}: index {bad} -> {got[bad]}, want {want[bad]}"
        ]
    return []


def check_grid(s: gen.StreamSpec, cluster: int | tuple[int, int, bool]) -> list[str]:
    """``cluster``: 1 (a single CTA), 2 (``CtaPair``) or a ``Cluster`` shape ``(m, n, pair)``."""
    from attn_gym.warpdrive import (
        AtomicPersistent,
        Cluster,
        ClusterLaunchControl,
        CtaPair,
        OneShot,
        SingleCta,
        StaticPersistent,
        WorkStream,
    )
    from attn_gym.warpdrive.scheduling import BoundWork

    schedule = {
        "oneshot": OneShot(),
        "static": StaticPersistent(),
        "atomic": AtomicPersistent(s.stages),
        "clc": ClusterLaunchControl(s.stages),
    }[s.schedule]
    stream = WorkStream(mapping=raster_of(s), schedule=schedule)
    if isinstance(cluster, tuple):
        declared, cs = Cluster(*cluster), cluster[0] * cluster[1]
    else:
        declared, cs = (CtaPair() if cluster == 2 else SingleCta()), cluster
    stream.owner = SimpleNamespace(cluster=declared)
    got = tuple(BoundWork(stream, stream.mapping.bind(s.extents), None, s.max_ctas).grid())
    want = oracle.expected_grid(SimpleNamespace(cs=cs, stream=s, units=1))
    return (
        []
        if got == want
        else [f"{s.schedule} {s.extents} cs={cs} cap={s.max_ctas}: grid {got}, want {want}"]
    )


def check_tile_grid_dims(space: tuple[int, int, int], cs: int) -> list[str]:
    from attn_gym.warpdrive.scheduling import tile_grid_dims

    em, en, el = space
    if min(space) <= 0:  # R17: an empty space launches one work unit
        want = (cs, 1, 1)
        got = tuple(tile_grid_dims(space, cs))
        return [] if got == want else [f"tile_grid_dims({space}, {cs}) = {got}, want {want}"]
    fits = en <= oracle.GRID_YZ_LIMIT and el <= oracle.GRID_YZ_LIMIT
    want = (max(em, 1) * cs, max(en, 1), max(el, 1)) if fits else (max(em * en * el, 1) * cs, 1, 1)
    got = tuple(tile_grid_dims(space, cs))
    return [] if got == want else [f"tile_grid_dims({space}, {cs}) = {got}, want {want}"]


def grid_boundary_spaces() -> list[tuple[int, int, int]]:
    lim = oracle.GRID_YZ_LIMIT
    return [
        (1, lim, 1),
        (1, lim + 1, 1),
        (3, 1, lim),
        (3, 1, lim + 1),
        (2, lim, lim),
        (7, lim + 1, lim + 1),
        (1, 1, 1),
        (0, 0, 1),
        (5, 0, 3),
    ]


# --------------------------------------------------------------------------- #
# Derived scopes of generated kernels
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Inst:
    """One bound role instance: a channel side's contributor."""

    name: str
    threads: int
    cta: str
    slot: int | None


@dataclass
class WantScope:
    pair: bool
    full_ctas: tuple
    empty_ctas: tuple
    full_counts: dict
    empty_counts: dict
    tx: int | None
    tail: bool
    full_mask: int | dict
    empty_mask: int | dict


def _mask(targets: dict) -> int | dict:
    """``tcgen05.commit`` masks per signalling rank, one int when every rank agrees."""
    masks = {r: sum(1 << t for t in ts) for r, ts in targets.items()}
    return masks.popitem()[1] if len(set(masks.values())) == 1 else masks


def arrivals(mech: str, threads: int) -> int:
    """Arrivals of one signalling instance: its threads, its warps, or one elected signal."""
    return {"threads": threads, "warps": threads // 32}.get(mech, 1)


def want_scope(
    n: int,
    prod_mech: str,
    cons_mech: str,
    producers: list[Inst],
    consumers: list[Inst],
    geom: gen.Geometry,
    tma_bytes: int | None,
    *,
    internal: bool = False,
    multicast: tuple[str, ...] = (),
    fills: str = "",
) -> WantScope:
    """R4 (CTA-local) and R11 (cross-CTA) counts, from the participants and the cluster."""
    ranks = tuple(range(geom.size))
    insts = (*producers, *consumers)
    where = {i.name: geom.ctas(i.cta) for i in insts}
    # The CTAs each rank's copies land in: its own (or, with fills, every CTA it fills) and
    # their multicast lines.
    filled = {r: geom.group(r) if fills == "pair" else ranks if fills else (r,) for r in ranks}
    reach = {r: set().union(*(geom.reach(d, multicast) for d in filled[r])) for r in ranks}
    copies_cross = any(len(cs) > 1 for cs in reach.values())
    same = len(set(where.values())) == 1

    def per_slot(insts, j):
        return [i for i in insts if i.slot in (None, j)]

    if same and not copies_cross:
        full, empty = (
            [sum(arrivals(mech, i.threads) for i in per_slot(side, j)) for j in range(n)]
            for mech, side in ((prod_mech, producers), (cons_mech, consumers))
        )

        def local_mask(mech, side):  # a CTA's own commit: bit ``rank`` of each signalling CTA
            if mech != "umma":
                return 1
            return _mask({c: (c,) for i in side for c in where[i.name]})

        return WantScope(
            False,
            (),
            (),
            dict.fromkeys(ranks, full),
            dict.fromkeys(ranks, empty),
            tma_bytes,
            False,
            local_mask(prod_mech, producers),
            local_mask(cons_mech, consumers),
        )
    full_ctas = tuple(sorted({c for i in consumers for c in where[i.name]}))
    empty_ctas = tuple(sorted({c for i in producers for c in where[i.name]}))
    # The CTAs one signal may reach: itself when only multicast crosses CTAs, its MMA pair when
    # every participant runs somewhere in each pair, else the whole cluster.
    if same:
        unit = {r: (r,) for r in ranks}
    elif all(set(cs) & set(geom.group(r)) for cs in where.values() for r in ranks):
        unit = {r: geom.group(r) for r in ranks}
    else:
        unit = dict.fromkeys(ranks, ranks)
    full_targets = {p: tuple(c for c in full_ctas if c in unit[p]) for p in empty_ctas}
    empty_targets = {
        c: tuple(p for p in empty_ctas if set(reach[p]) & set(unit[c])) for c in full_ctas
    }
    tx = tma_bytes
    leaders = geom.ctas("leader")
    if tma_bytes is not None and not internal:
        # Copies landing in CTA d complete on d's barrier, or with leader-only consumers on its
        # pair leader's (2-CTA copies). That barrier is armed once, for every CTA completing
        # there, by its own CTA or (fills) by the CTA filling it. A consumer releases each
        # producer whose copies land in a CTA completing on its barrier or in its unit.
        two = not set().union(*(reach[p] for p in empty_ctas)) <= set(full_ctas)
        assert not two or (geom.pair and set(full_ctas) <= set(leaders)), "generator bug"
        home = {d: geom.group(d)[0] if two else d for d in ranks}
        tx = tma_bytes * (2 if two else 1)
        if fills:
            full_targets = {p: tuple(sorted({home[d] for d in filled[p]})) for p in empty_ctas}
        else:
            bars = {home[d] for p in empty_ctas for d in reach[p]}
            full_targets = {p: (p,) for p in empty_ctas if p in bars}
        empty_targets = {
            c: tuple(
                p
                for p in empty_ctas
                if reach[p] & ({d for d in ranks if home[d] == c} | set(unit[c]))
            )
            for c in full_ctas
        }

    def counts(mech, side, targets):
        out = {}
        for c in ranks:
            row = []
            for j in range(n):
                senders = [(i, s) for i in per_slot(side, j) for s in where[i.name]]
                senders = [(i, s) for i, s in senders if c in targets.get(s, ())]
                # Threads: every thread of every signalling (instance, CTA); Warps: every warp;
                # a TMA arm or a multicast tcgen05.commit: one arrival per signalling CTA.
                if mech in ("threads", "warps"):
                    row.append(sum(arrivals(mech, i.threads) for i, _ in senders))
                else:
                    row.append(len({s for _, s in senders}))
            out[c] = row if any(row) else [1] * n
        return out

    return WantScope(
        True,
        full_ctas,
        empty_ctas,
        counts(prod_mech, producers, full_targets),
        counts(cons_mech, consumers, empty_targets),
        tx,
        cons_mech == "umma" and any(p != c for c, ps in empty_targets.items() for p in ps),
        _mask(full_targets),
        _mask(empty_targets),
    )


def _insts(spec: gen.KernelSpec, side: list[tuple[str, bool]]) -> list[Inst]:
    out = []
    for rname, bound in side:
        r = spec.role(rname)
        for i in range(r.instances):
            out.append(Inst(f"{rname}[{i}]", r.threads, r.cta, i if bound else None))
    return out


def scheduler_warp(spec: gen.KernelSpec) -> int:
    s = spec.stream
    if s.scheduler_warp is not None:
        return s.scheduler_warp
    owned = {w for r in spec.roles for rng in r.warp_ranges() for w in rng}
    return next(w for w in range(max(owned) + 2) if w not in owned)


def expected_scopes(spec: gen.KernelSpec) -> dict[str, WantScope]:
    geom = spec.geom
    cs = geom.size
    out = {}
    for ch in spec.channels:
        if ch.kind == "raw":  # Barriers: no derived scope (checked with the barrier plan)
            continue
        tma_bytes = ch.tma_bytes
        if ch.mma == "ab":
            tma_bytes = (gen.MMA_M + spec.mma.n) * spec.mma.k * 2
        out[ch.name] = want_scope(
            ch.n,
            "warps" if "p" in ch.warps else ch.producer,
            "warps" if "c" in ch.warps else ch.consumer,
            _insts(spec, list(ch.producers)),
            _insts(spec, list(ch.consumers)),
            geom,
            tma_bytes if ch.producer == "tma" else None,
            multicast=tuple(ch.multicast),
            fills=ch.fills,
        )
    s = spec.stream
    if s is not None and s.broadcast:
        # One scheduler per cluster: rank 0 (the leader in a CTA pair) publishes to every CTA.
        place = "all" if cs == 1 else "leader" if geom.ctas("leader") == (0,) else "rank0"
        sched = Inst("work.scheduler", 32, place, None)
        users = [
            Inst(f"{r.name}[{i}]", r.threads, r.cta, None)
            for r in spec.roles
            for i in range(r.instances)
            if not r.raw
        ]
        out["work_ring"] = want_scope(
            s.stages,
            "tma" if cs > 1 else "threads",
            "threads",
            [sched],
            [*users, sched],
            geom,
            16 if cs > 1 else None,
            internal=True,
        )
        if s.schedule == "clc":
            out["work_clc"] = want_scope(1, "tma", "threads", [sched], [sched], geom, 16)
    return out


def expected_regs(spec: gen.KernelSpec) -> tuple[int, list, dict] | None:
    """(padded warp count, [(hole range, budget)], {role instance: budget}) per R9, or None when
    no role sets a budget."""
    budget = {}
    names = {}
    for r in spec.roles:
        for i, rng in enumerate(r.warp_ranges()):
            for w in rng:
                budget[w] = r.regs
            names[f"{r.name}[{i}]" if r.instances > 1 else r.name] = r.regs
    s = spec.stream
    if not any(r.regs is not None for r in spec.roles):
        return None
    if s is not None and s.broadcast:
        w = scheduler_warp(spec)
        group = {budget[x] for x in range(w // 4 * 4, w // 4 * 4 + 4) if x in budget}
        numeric = [r.regs for r in spec.roles if r.regs is not None]
        regs = group.pop() if len(group) == 1 else min(numeric)
        budget[w] = regs
        names["work.scheduler"] = regs
    nw = -(-(max(budget) + 1) // 4) * 4
    holes = []
    for g in range(nw // 4):
        vals = {budget[w] for w in range(4 * g, 4 * g + 4) if w in budget}
        regs = vals.pop() if vals else gen.MIN_REGS
        run = []
        for w in range(4 * g, 4 * g + 4):
            if w in budget:
                if run:
                    holes.append((range(run[0], run[-1] + 1), regs))
                run = []
            else:
                run.append(w)
        if run:
            holes.append((range(run[0], run[-1] + 1), regs))
    return nw, holes, names


_ROW = re.compile(r"^  (\d+)(?:-(\d+))?\s+(\S+)\s+(\d+)?")


def described_roles(text: str) -> list[tuple[range, str, int | None]]:
    rows = []
    body = text.split("\nRoles\n", 1)[1].split("\n\n", 1)[0].splitlines()[1:]
    for line in body:
        m = _ROW.match(line)
        if m:
            a, b, name, regs = m.groups()
            rows.append((range(int(a), int(b or a) + 1), name, int(regs) if regs else None))
    return rows


@dataclass
class ScopeReport:
    failures: list[str] = field(default_factory=list)
    channels: int = 0


def check_declared(spec: gen.KernelSpec, workdir: Path) -> ScopeReport:
    """Compare a generated kernel's derived scopes, scheduler, budgets and barrier init counts
    with the independent recomputation (host only: ``describe()``)."""
    from warpdrive_fuzz import runtime

    rep = ScopeReport()
    op = runtime.load_kernel(spec, workdir)
    text = op.describe()
    chans = op._ws_chans
    for name, want in expected_scopes(spec).items():
        rep.channels += 1
        got = chans[name].derived_scope
        # Overrides replace exactly the overridden fields of the effective scope (U7).
        spec_ch = next((c for c in spec.channels if c.name == name), None)
        if spec_ch is not None and spec_ch.manual:
            effective = chans[name].scope
            for key, value in gen.override_of(spec, spec_ch).items():
                field_counts = getattr(effective, key)
                want_counts = (
                    value
                    if isinstance(value, dict)
                    else {
                        c: [value] * spec_ch.n if isinstance(value, int) else value
                        for c in range(spec.cs)
                    }
                )
                if {c: list(v) for c, v in field_counts.items()} != want_counts:
                    rep.failures.append(
                        f"{name}.{key}: effective {field_counts}, override {value}"
                    )
        for f in (
            "pair",
            "full_ctas",
            "empty_ctas",
            "full_counts",
            "empty_counts",
            "tx",
            "tail",
            "full_mask",
            "empty_mask",
        ):
            if getattr(got, f) != getattr(want, f):
                rep.failures.append(
                    f"{name}.{f}: layer {getattr(got, f)!r}, model {getattr(want, f)!r}"
                )
    s = spec.stream
    if s is not None and s.broadcast:
        warp = scheduler_warp(spec)
        if f"scheduler warp {warp} broadcasts" not in text:
            rep.failures.append(f"scheduler not at warp {warp}:\n{text}")
    want = expected_regs(spec)
    if want is not None:
        nw, holes, regs = want
        rows = described_roles(text)
        if op.num_warps != nw:
            rep.failures.append(f"num_warps {op.num_warps}, model {nw}")
        got_holes = [(rng, b) for rng, name, b in rows if name == "(padding)"]
        if got_holes != holes:
            rep.failures.append(f"padding {got_holes}, model {holes}")
        got_regs = {name: b for _, name, b in rows if name != "(padding)"}
        for name, b in regs.items():
            default = gen._default_regs(nw)
            if got_regs.get(name) not in (b, default if b is None else b):
                rep.failures.append(f"{name} budget {got_regs.get(name)}, model {b}")
    # R1: barriers are initialized with the effective counts (full then empty, per stage).
    plan, _ = op._barrier_plan()
    raw = {c.name: c for c in spec.channels if c.kind == "raw"}
    for name, _, counts in plan:
        if name in chans:
            sc = chans[name].scope
            for c in range(spec.cs):
                if counts[c] != sc.full_counts[c] + sc.empty_counts[c]:
                    rep.failures.append(f"{name} rank {c} init counts {counts[c]}")
        elif name in raw:  # Barriers(counts=...) initialized as declared; None: by the kernel
            ch = raw[name]
            p_threads = spec.role(ch.producers[0][0]).threads
            c_threads = spec.role(ch.consumers[0][0]).threads
            want = [p_threads] * ch.n + [c_threads] * ch.n if ch.raw_init == "auto" else None
            for c in range(spec.cs):
                if want is None and any(x is not None for x in counts[c]):
                    rep.failures.append(f"{name} rank {c}: counts=None initialized {counts[c]}")
                elif want is not None and list(counts[c]) != want:
                    rep.failures.append(f"{name} rank {c} init counts {counts[c]}, want {want}")
    return rep
