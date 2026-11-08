"""Host model of a generated kernel and the checks against what the device logged.

Everything here is derived from the topic contracts (CASES.md maps R1-R14) and the spec, not
from warpdrive's helpers: the launch grid (R12/R13), each work unit's item sequence (R12), the
Raster order (an explicit enumeration, independent of ``Raster.decode``'s closed form), and each
consumer's expected epoch stream (R2/R3/R4) from the role's own handle program.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

import numpy as np

from warpdrive_fuzz.gen import (
    BOUND,
    MAX_WARPS,
    KernelSpec,
    Layout,
    StreamSpec,
    item_order,
    packed_rows,
    presence_code,
    skipped,
    table_rows,
    uses_stream,
)

GRID_YZ_LIMIT = 65535


def raster_order(s: StreamSpec) -> list[tuple[int, int, int]]:
    """Every (m, n, l) in visiting order (``gen.item_order``: an explicit enumeration of the
    documented order, independent of ``Raster.decode``)."""
    return list(item_order(s))


def garbage_row(s: StreamSpec) -> tuple[int, int, int]:
    """Spare packed-table rows: outside the raster space, so a decoded one is visibly wrong."""
    return (max(s.extents[0], 0) + 7, 0, 0)


@functools.cache
def space(s: StreamSpec) -> tuple[tuple, tuple]:
    """``(item per index, kept per index)`` over ``[0, s.total)`` for the stream's mapping."""
    if s.mapping == "table":
        items = table_rows(s)
        return items, (True,) * len(items)
    if s.mapping == "packed":
        rows = packed_rows(s)
        items = rows + (garbage_row(s),) * s.pad
        return items, tuple(i < len(rows) for i in range(len(items)))
    items = item_order(s)
    return items, tuple(not skipped(s, it) for it in items)


def expected_grid(spec: KernelSpec) -> tuple[int, int, int]:
    cs = spec.cs
    s = spec.stream
    if s is None:
        return (spec.units * cs, 1, 1)
    em, en, el = s.extents
    if s.total == 0 and not s.launch_1d:  # R17: an empty space launches one work unit
        return (cs, 1, 1)
    if s.launch_1d:
        return (max(s.total, 1) * cs, 1, 1)
    if s.tile_grid:
        if en <= GRID_YZ_LIMIT and el <= GRID_YZ_LIMIT:
            return (max(em, 1) * cs, max(en, 1), max(el, 1))
        return (max(s.total, 1) * cs, 1, 1)
    if s.capped:
        cap = max(s.max_ctas // cs, 1) if cs > 1 else s.max_ctas
        return (max(1, min(s.total, cap)) * cs, 1, 1)
    return (max(s.total, 1) * cs, 1, 1)


def item_hash(m: int, n: int, li: int) -> int:
    return (m * 1009 + n * 917 + li * 101) & 0x3FFFFFFF


@dataclass
class Cta:
    b: int  # linear CTA id
    rank: int
    unit: int


def ctas(spec: KernelSpec) -> list[Cta]:
    gx, gy, gz = expected_grid(spec)
    cs = spec.cs
    out = []
    s = spec.stream
    for bz in range(gz):
        for by in range(gy):
            for bx in range(gx):
                b = bx + gx * (by + gy * bz)
                unit = bx // cs
                if s is not None and s.tile_grid and gy * gz > 1:
                    em, en, _ = s.extents
                    unit = unit + em * (by + en * bz)
                out.append(Cta(b, bx % cs, unit))
    return out


def static_items(spec: KernelSpec, unit: int, units: int) -> list[int] | None:
    """The unit's kept indices for schedules with a fixed per-unit sequence (OneShot, static,
    and the two reverse-static custom schedules); None for claimed schedules."""
    s = spec.stream
    _, kept = space(s)
    if s.schedule == "oneshot":
        seq = [unit] if unit < s.total else []
    elif s.schedule == "static":
        seq = list(range(unit, s.total, units))
    elif s.schedule in ("reverse", "rbroadcast"):
        seq = list(range(s.total - 1 - unit, -1, -units))
    else:
        return None
    return [i for i in seq if kept[i]]


@dataclass
class Expected:
    count: int
    values: list[int] | None  # observed tags in program order (None: a token channel)


def handle_stream(spec: KernelSpec, rname: str, inst: int, its: int) -> tuple[dict, dict]:
    """``({chan: Expected}, {chan: producer opens})`` for one role instance running ``its``
    iterations and the extra round."""
    r = spec.role(rname)
    chans = {c.name: c for c in spec.channels}
    cons: dict[str, list] = {}
    prods: dict[str, int] = {}

    def run(i: int, extra: bool):
        for op, ch, side, k in r.events:
            if op != "o":
                continue
            c = chans[ch]
            if extra:
                if c.kind != "slots" and k >= c.extra_mult:
                    continue
                slot = inst if k == BOUND else k
                if c.kind == "slots" and slot not in c.extra_slots:
                    continue
            e = i * c.mult + k if c.kind != "slots" else i
            if side == "p":
                prods[ch] = prods.get(ch, 0) + 1
            else:
                value = e % c.tma_tiles if c.producer == "tma" else e
                cons.setdefault(ch, []).append(value)

    for i in range(its):
        run(i, False)
    run(its, True)
    out = {}
    for ch in {c.name for c in spec.channels if any(rn == rname for rn, _ in c.consumers)}:
        c = chans[ch]
        vals = cons.get(ch, [])
        tagged = c.payload or c.tiles_checked or c.kind == "raw"
        out[ch] = Expected(len(vals), vals if tagged else None)
    for ch in {c.name for c in spec.channels if any(rn == rname for rn, _ in c.producers)}:
        prods.setdefault(ch, 0)
    return out, prods


@dataclass
class Launch:
    clog: np.ndarray  # (rows, per_cta) int32, rows >= expected CTA count
    hits: np.ndarray  # flat int32
    ws: np.ndarray
    timeline: dict | None = None  # WorkStream.timeline(): {role: [[(index, t), ...] per CTA]}
    bad: int = 0  # decodes outside [0, total) of a mapping without decodes_any_index (R14)


@dataclass
class Report:
    failures: list[str] = field(default_factory=list)
    checked: int = 0

    def fail(self, msg: str):
        if len(self.failures) < 50:
            self.failures.append(msg)

    def expect(self, cond: bool, msg: str):
        self.checked += 1
        if not cond:
            self.fail(msg)


def initial_clog(spec: KernelSpec, rows: int) -> np.ndarray:
    lay = Layout.of(spec)
    clog = np.full((rows, lay.per_cta), -1, dtype=np.int32)
    clog[:, lay.accumulators] = 0
    return clog


def placed(spec: KernelSpec, rname: str, rank: int) -> bool:
    return rank in spec.ctas(rname)


FULL_CHECK_CTAS = 4096


def _sample(cta_list: list[Cta]) -> list[Cta]:
    """Every CTA of an ordinary grid; for the y/z-limit fallback grids (>65535 CTAs) a fixed
    sample of whole units plus both ends (item coverage is still checked on every item)."""
    if len(cta_list) <= FULL_CHECK_CTAS:
        return cta_list
    units = sorted({c.unit for c in cta_list})
    keep = set(units[:64] + units[-64:] + units[:: max(1, len(units) // 512)])
    return [c for c in cta_list if c.unit in keep]


def check_launch(spec: KernelSpec, run: Launch, tag: str) -> Report:
    rep = Report()
    lay = Layout.of(spec)
    cs = spec.cs
    cta_list = ctas(spec)
    ncta = len(cta_list)
    init = initial_clog(spec, run.clog.shape[0])
    rep.expect(
        np.array_equal(run.clog[ncta:], init[ncta:]),
        f"{tag}: CTAs beyond the expected grid {expected_grid(spec)} wrote logs",
    )
    s = spec.stream
    units = max(c.unit for c in cta_list) + 1
    order, kept = space(s) if s is not None else ((), ())
    unit_items: dict[int, list[int] | None] = {}
    absent = {}
    active_ranks = {
        rank for rank in range(cs) if any(placed(spec, r.name, rank) for r in spec.roles)
    }
    for cta in _sample(cta_list):
        row = run.clog[cta.b]
        if cta.rank not in active_ranks:
            rep.expect(
                np.array_equal(row, init[cta.b]), f"{tag}: cta {cta.b} has no roles but wrote logs"
            )
            continue
        if s is not None and s.schedule == "clc" and np.array_equal(row, init[cta.b]):
            # try_cancel took this cluster's launch: it never ran, another unit ran its item.
            absent.setdefault(cta.unit, set()).add(cta.rank)
            continue
        # R1 placement: exactly the declared warps of placed instances executed role code.
        want = np.zeros(MAX_WARPS, dtype=np.int32)
        for r in spec.roles:
            if not placed(spec, r.name, cta.rank):
                continue
            for i, rng in enumerate(r.warp_ranges()):
                want[list(rng)] = presence_code(spec, r.name, i)
        got = row[lay.presence : lay.presence + MAX_WARPS]
        rep.expect(
            np.array_equal(got, want),
            f"{tag}: cta {cta.b} warp presence {got.tolist()} != {want.tolist()}",
        )
        # Work items (R12): the unit's sequence, the same in every role and CTA of the unit.
        its = spec.iters
        if s is not None:
            seqs, raw_seqs = [], []
            for r in spec.roles:
                if not placed(spec, r.name, cta.rank) or not uses_stream(spec, r):
                    continue
                if r.raw:  # single_item(): a count and the item's hash, no index
                    off = lay.role_items[(r.name, 0)]
                    k = int(row[off])
                    raw_seqs.append(
                        (
                            r.name,
                            k,
                            row[off + 1 + spec.item_cap : off + 2 + spec.item_cap][:k].tolist(),
                        )
                    )
                    continue
                for i in range(r.instances):
                    off = lay.role_items[(r.name, i)]
                    k = int(row[off])
                    seqs.append(
                        (
                            f"{r.name}[{i}]",
                            k,
                            row[off + 1 : off + 1 + min(k, spec.item_cap)].tolist(),
                            row[
                                off + 1 + spec.item_cap : off
                                + 1
                                + spec.item_cap
                                + min(k, spec.item_cap)
                            ].tolist(),
                        )
                    )
            if len({(k, tuple(v)) for _, k, v, _ in seqs}) > 1:
                rep.fail(f"{tag}: cta {cta.b} roles saw different item sequences: {seqs[:4]}")
            expect = static_items(spec, cta.unit, units)  # None for claimed schedules
            if cta.unit in unit_items and unit_items[cta.unit] is not None and seqs:
                prev = unit_items[cta.unit]
                rep.expect(
                    seqs[0][2] == prev[: spec.item_cap] or len(prev) > spec.item_cap,
                    f"{tag}: cta {cta.b} (rank {cta.rank}) item sequence differs from its cluster",
                )
            if expect is not None:
                unit_items[cta.unit] = expect
                for name, k, idx, hashes in seqs:
                    rep.expect(
                        k == len(expect) and idx == expect[: spec.item_cap],
                        f"{tag}: cta {cta.b} {name} items {k}:{idx[:8]} != {len(expect)}:"
                        f"{expect[:8]}",
                    )
                for name, k, hashes in raw_seqs:  # R20: what a managed cursor starts with
                    want_h = [item_hash(*order[i]) for i in expect]
                    rep.expect(
                        k == len(expect) and hashes == want_h[:1],
                        f"{tag}: cta {cta.b} raw {name} single_item {k}:{hashes} != "
                        f"{len(expect)}:{want_h[:1]}",
                    )
            elif seqs:
                name, k, idx, _ = seqs[0]
                unit_items.setdefault(cta.unit, idx if k <= spec.item_cap else None)
                expect = idx if k <= spec.item_cap else None
            for name, k, idx, hashes in seqs:
                rep.expect(
                    hashes == [item_hash(*order[i]) for i in idx if 0 <= i < len(order)]
                    and all(0 <= i < len(order) for i in idx),
                    f"{tag}: cta {cta.b} {name} work words do not decode its indices {idx[:8]}",
                )
            if run.timeline is not None and s.record:
                # R16: the recorded log holds each role's (and the scheduler's) first items.
                for name, per_cta in run.timeline.items():
                    role_name = name.split("[")[0]
                    runs_here = (
                        cta.rank == 0  # one scheduler per cluster, in rank 0
                        if name == "work.scheduler"
                        else placed(spec, role_name, cta.rank)
                    )
                    if not runs_here:
                        want_rec = []
                    else:  # None: more items than the log holds, nothing to compare
                        want_rec = None if expect is None else expect[: s.record]
                    got_rec = [i for i, _ in per_cta[cta.b]] if cta.b < len(per_cta) else []
                    rep.expect(
                        want_rec is None or got_rec == want_rec,
                        f"{tag}: cta {cta.b} timeline[{name}] {got_rec} != {want_rec}",
                    )
            if expect is not None:
                its = sum(s.item_rounds(*order[i]) for i in expect)
            else:  # more items than the log holds: every role must still agree on the count
                its = None
        # Channels (R2-R4): each consumer saw every epoch once, in order, intact.
        reported = set()
        for r in spec.roles:
            if not placed(spec, r.name, cta.rank):
                continue
            for i in range(r.instances):
                got_it = int(row[lay.role_it[(r.name, i)]])
                follows = uses_stream(spec, r)
                if follows:
                    reported.add(got_it)
                role_its = spec.iters if not follows else got_it if its is None else its
                rep.expect(
                    got_it == role_its,
                    f"{tag}: cta {cta.b} {r.name}[{i}] ran {got_it} iterations, model {role_its}",
                )
                cons, prods = handle_stream(spec, r.name, i, role_its)
                for ch, exp in cons.items():
                    off = lay.cons[(ch, r.name, i)]
                    count, errs = int(row[off]), int(row[off + 1])
                    rep.expect(
                        count == exp.count,
                        f"{tag}: cta {cta.b} {r.name}[{i}] consumed {count} of {ch}, model "
                        f"{exp.count}",
                    )
                    rep.expect(
                        errs == 0,
                        f"{tag}: cta {cta.b} {r.name}[{i}] read {errs} corrupt/stale words of {ch}",
                    )
                    if exp.values is not None:
                        n = min(exp.count, spec.log_cap)
                        seen = row[off + 2 : off + 2 + n].tolist()
                        rep.expect(
                            seen == exp.values[:n],
                            f"{tag}: cta {cta.b} {r.name}[{i}] {ch} epochs {seen[:12]} != "
                            f"{exp.values[:12]}",
                        )
                for ch, n in prods.items():
                    got = int(row[lay.prod[(ch, r.name, i)]])
                    rep.expect(
                        got == n,
                        f"{tag}: cta {cta.b} {r.name}[{i}] produced {got} of {ch}, model {n}",
                    )
        if its is None:
            rep.expect(len(reported) <= 1, f"{tag}: cta {cta.b} roles disagree on iterations")
        # R10 turnstile: seats strictly cyclic.
        if spec.turn_role is not None and placed(spec, spec.turn_role, cta.rank):
            tr = spec.role(spec.turn_role)
            cnt = int(row[lay.turn])
            if spec.turn_enabled:
                role_its = its if its is not None else int(row[lay.role_it[(tr.name, 0)]])
                n = min(cnt, spec.turn_cap)
                want = [k % tr.instances for k in range(n)]
                rep.expect(
                    cnt == role_its * tr.instances
                    and row[lay.turn + 1 : lay.turn + 1 + n].tolist() == want,
                    f"{tag}: cta {cta.b} turnstile log {cnt}:"
                    f"{row[lay.turn + 1 : lay.turn + 1 + min(n, 12)].tolist()}",
                )
    for unit, ranks in absent.items():
        rep.expect(ranks >= active_ranks, f"{tag}: CLC unit {unit} ran in only some of its CTAs")
    # R12 coverage: each kept index once per participating (role instance, rank).
    if s is not None:
        kept = np.array(kept, dtype=np.int32)
        total = s.total
        for r in spec.roles:
            for i in range(r.instances):
                ordinal = lay.instance_ids[(r.name, i)]
                for rank in range(cs):
                    seg = run.hits[
                        (ordinal * cs + rank) * total : (ordinal * cs + rank + 1) * total
                    ]
                    logs = placed(spec, r.name, rank) and not r.raw
                    want = kept if logs else np.zeros_like(kept)
                    if not np.array_equal(seg, want):
                        bad = np.nonzero(seg != want)[0][:8].tolist()
                        rep.fail(
                            f"{tag}: {r.name}[{i}] rank {rank} item hits wrong at {bad}: "
                            f"{seg[bad].tolist()} (want {want[bad].tolist()})"
                        )
        if s.schedule == "atomic":
            rep.expect(
                run.ws[:2].tolist() == [0, 0],
                f"{tag}: atomic workspace not reset: {run.ws[:2].tolist()}",
            )
        rep.expect(
            run.bad == 0,
            f"{tag}: the table mapping (no decodes_any_index) was decoded {run.bad} times "
            "outside [0, total)",
        )
    return rep


def deterministic(spec: KernelSpec) -> bool:
    return spec.stream is None or spec.stream.schedule in (
        "oneshot",
        "static",
        "reverse",
        "rbroadcast",
    )


def check(spec: KernelSpec, runs: list[Launch]) -> Report:
    rep = Report()
    for k, run in enumerate(runs):
        sub = check_launch(spec, run, f"launch {k}")
        rep.failures += sub.failures
        rep.checked += sub.checked
    if deterministic(spec):
        for k, run in enumerate(runs[1:], 1):
            rep.expect(
                np.array_equal(run.clog, runs[0].clog) and np.array_equal(run.hits, runs[0].hits),
                f"launch {k} logs differ bitwise from launch 0",
            )
    return rep
