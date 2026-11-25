"""Render a self-contained HTML graph for ``describe(format="html")``.

Uses ``describe_graph``'s collected model: role cards show warps, registers and CTA
placement; channel chips show stages and signal mechanisms. Wire colours match each
side's mechanism, and double wires cross CTAs. Badges show scope, fences, overrides,
sharing and compile results. Non-handoff synchronization uses dotted hub spokes; raw
code uses dashed nodes. Below the graph, the inspector explains the selected role or
channel (what completes FULL and EMPTY, who waits for what), a table lists which role
fills and frees each channel, and a panel lists streams and aliases.

Layout runs here: cycle breaking, longest-path ranks, virtual nodes for long edges,
barycenter ordering and constrained vertical placement. The page needs no layout library
or network. Every relation the page states is computed here and shipped as JSON; its
script only renders and handles interaction.
"""

from __future__ import annotations

import html
import inspect
import itertools
import statistics
from dataclasses import dataclass, field

from attn_gym.warpdrive.channels import SIDES, Channel, Sharing, Tma, Umma, Warps
from attn_gym.warpdrive.describe import _overrides, channel_barriers, is_pair, rank_label
from attn_gym.warpdrive.describe_graph import (
    Graph,
    _warps,
    build_graph,
    consumer_mechanism,
    mechanism,
    wire_mechanism,
)
from attn_gym.warpdrive.html_page import asset, page

_HUB_KINDS = ("named", "turnstile", "barriers")

COL_GAP = 48  # between rank columns: wires curve only here
ROW_GAP = 22  # between boxes of one column
MARGIN = 24
LANE_GAP = 18  # between the graph's lowest box and each feedback lane under it


def describe_html(kernel) -> str:
    return to_html(build_graph(kernel), kernel)


def _tw(text: str, size: float) -> float:
    """Estimate monospace text width in pixels at ``size`` (0.61 em per character)."""
    return len(text) * size * 0.61


def _esc(text) -> str:
    return html.escape(str(text), quote=True)


# Drawn items


@dataclass
class _Box:
    """A positioned node, channel chip or virtual node on a long wire."""

    id: str
    kind: str  # a NodeKind, "chip", or "dummy"
    w: float
    h: float
    svg: str = ""  # drawn at the origin; translated to (x - w/2, y - h/2)
    details: dict = field(default_factory=dict)
    rank: int = 0
    x: float = 0.0  # centre
    y: float = 0.0
    chan: Channel | None = None  # a chip's channel (the first, when it groups several alike)


@dataclass
class _Wire:
    src: str
    dst: str
    kind: str  # "into" (producer -> chip), "flow", "stream", "spoke"
    color: str
    pair: bool = False
    label: str = ""
    chip: str = ""  # the chip whose details the wire shows
    feedback: bool = False  # runs back to an earlier column, in a lane under the graph


def to_html(graph: Graph, kernel) -> str:
    boxes, wires, panel = _items(graph, kernel)
    links = [w for w in wires if w.kind != "spoke"]
    spokes = [w for w in wires if w.kind == "spoke"]
    routes, feedback = _layout(
        boxes, [(w.src, w.dst) for w in links], [(w.src, w.dst) for w in spokes]
    )
    for i in feedback:
        links[i].feedback = True
    svg = _svg(boxes, links, routes, spokes)
    _handoffs(boxes, links)
    details = {b.id: b.details for b in boxes.values() if b.details}
    matrix = _matrix(boxes)
    body = f"""<div class="legend">{_LEGEND}</div>
<div class="figure"><div class="graph-tools">
<span class="figlabel">Ctrl/⌘ + scroll to zoom · drag to pan</span>
<button type="button" id="zoom-out" aria-label="Zoom out">−</button>
<button type="button" id="zoom-in" aria-label="Zoom in">+</button>
<button type="button" id="zoom-fit">Fit</button></div>{svg}</div>
<section id="inspector" aria-label="Pipeline inspector">
<p class="figlabel">Read a handoff</p>
<p id="inspector-intro">Select a role or channel.</p>
<div id="inspection" aria-live="polite"></div>
</section>
{matrix}
<div class="notes">{panel}</div>
<div id="tip" role="status" hidden></div>
<footer>Drawn by <code>describe(format="html")</code> from the kernel's declarations
(uncompiled values unless a compile recorded TMA checks, epochs and barrier users).</footer>"""
    return page(
        graph.title.split(":")[0],
        graph.title.split(": ", 1)[-1],
        body,
        asset("pipeline.js"),
        details,
        asset("pipeline.css"),
    )


def _items(graph: Graph, kernel):
    roles = {r.name: r for r in kernel._ws_roles}
    chans = kernel._ws_chans
    edged = {e.src for e in graph.edges} | {e.dst for e in graph.edges}
    boxes: dict[str, _Box] = {}
    panel = []
    streams = []
    for n in graph.nodes:
        match n.kind:
            case "storage" | "named_list":  # drawn in the side panel instead of the graph
                title = "storage aliases" if n.kind == "storage" else "named barriers"
                panel.append(_section(title, n.lines[1:]))
            case "stream":
                streams.append(n.lines)
                if n.id in edged:
                    boxes[n.id] = _stream_box(n)
            case "role":
                boxes[n.id] = _role_box(n, roles[n.ref], kernel)
            case "channel":
                boxes[n.id] = _chip_box(n.id, [n.ref], n.lines, kernel)
            case kind if kind in _HUB_KINDS and n.id not in edged:  # no known users to join
                panel.append(_section(n.ref, n.lines[1:]))
            case kind if kind in _HUB_KINDS:
                boxes[n.id] = _hub_box(n)
            case _:  # raw code, kernel code
                boxes[n.id] = _external_box(n)
    if streams:
        panel.insert(
            0, _section("work streams", [" · ".join(s[:2]) + f" ({s[2]})" for s in streams])
        )
    wires = []
    chip_of = {n.id for n in graph.nodes if n.kind == "channel"}  # junctions
    for i, e in enumerate(graph.edges):
        match e.kind:
            case "spoke":
                wires.append(_Wire(e.src, e.dst, "spoke", "sync", label=" ".join(e.lines)))
            case "stream":
                wires.append(_Wire(e.src, e.dst, "stream", "stream"))
            case "into" | "flow" if e.src in chip_of or e.dst in chip_of:
                junction = e.dst if e.kind == "into" else e.src
                color = wire_mechanism(graph, e, chans)
                label = " ".join(e.lines)
                wires.append(_Wire(e.src, e.dst, e.kind, color, e.cross_cta, label, junction))
            case _:  # a 1:1 channel edge: a chip between its ends
                chip = f"chip_{i}"
                boxes[chip] = _chip_box(chip, list(e.channels), e.lines, kernel)
                ch = chans[e.channels[0]]
                prod = mechanism(ch.producer, ch.named)
                cons = mechanism(consumer_mechanism(ch, graph, e.dst), ch.named)
                wires.append(_Wire(e.src, chip, "into", prod, e.cross_cta, chip=chip))
                wires.append(_Wire(chip, e.dst, "flow", cons, e.cross_cta, chip=chip))
    return boxes, wires, "\n".join(panel)


def _section(title: str, lines) -> str:
    items = "".join(f"<li>{_esc(line)}</li>" for line in lines)
    return f'<section><p class="figlabel">{_esc(title)}</p><ul>{items}</ul></section>'


def _text(x, y, text, cls, anchor="start") -> str:
    a = "" if anchor == "start" else f' text-anchor="{anchor}"'
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}"{a}>{_esc(text)}</text>'


def _pill(x, y, text, cls) -> tuple[str, float]:
    """A badge at ``(x, y)`` (top left) and its width."""
    w = _tw(text, 8.5) + 8
    return (
        (
            f'<g class="badge {cls}"><rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="12" '
            f'rx="2"/>{_text(x + 4, y + 9, text, "badge-t")}</g>'
        ),
        w,
    )


def _role_box(n, r, kernel) -> _Box:
    cluster = kernel.cluster
    facts = _warps(r.warps) + ("" if r.regs is None else f" · {r.regs} regs")
    strip_w = kernel.num_warps * 8.5 + (kernel.num_warps - 1) // 4 * 3
    ranks = cluster.ctas(r.cta)
    placement, where = _placement(cluster, r.cta)
    badges = []
    if cluster.size > 1:
        badges.append((placement, "place"))
    if r.raw:
        badges.append(("raw", "warn"))
    badge_w = sum(_tw(t, 8.5) + 12 for t, _ in badges)
    w = max(_tw(r.name, 13) + badge_w + 24, _tw(facts, 10.5) + 20, strip_w + 20, 132)
    h = 60
    parts = []
    if len(ranks) > 1:  # a stacked card: the role runs on more than one CTA
        parts.append(f'<rect class="card back" x="4" y="-4" width="{w:.1f}" height="{h}" rx="3"/>')
    dash = " raw" if r.raw else ""
    parts.append(f'<rect class="card{dash}" width="{w:.1f}" height="{h}" rx="2"/>')
    parts.append(f'<rect class="role-bar" width="4" height="{h}" rx="1"/>')  # roles execute
    parts.append(_text(10, 19, r.name, "role-name"))
    parts.append(_text(10, 34, facts, "role-facts"))
    x = w - 8
    for text, cls in reversed(badges):
        x -= _tw(text, 8.5) + 8
        parts.append(_pill(x, 8, text, cls)[0])
        x -= 4
    x = 10.0
    for warp in range(kernel.num_warps):
        if warp and warp % 4 == 0:
            x += 3  # warpgroup boundary
        cls = "warp on" if warp in r.warps else "warp"
        parts.append(f'<rect class="{cls}" x="{x:.1f}" y="43" width="7" height="8" rx="1"/>')
        x += 8.5
    lines = [f"{_warps(r.warps)} of the CTA's {kernel.num_warps}", *n.lines[1:]]
    size = f"{len(r.warps)} warp{'s' * (len(r.warps) > 1)} · {r.num_threads} threads"
    details = {
        "title": r.name,
        "lines": lines,
        "warps": list(r.warps),
        "num_warps": kernel.num_warps,
        "size": size + ("" if r.regs is None else f" · {r.regs} regs"),
        "handoffs": [],  # filled by _handoffs
    }
    if cluster.size > 1:
        lines.append(f"cta {r.cta}: {where}")
        details["placement"] = placement
        details["where"] = where[0].upper() + where[1:] + "."
        if cluster.size <= 4:  # few enough thread blocks to draw each one's warps
            pair = is_pair(cluster)
            details["ctas"] = [
                {"label": rank_label(k, pair) + " CTA" * pair, "here": k in ranks}
                for k in range(cluster.size)
            ]
    # Only the role body's own docstring says what it does; the declarations cannot.
    doc = inspect.getdoc(inspect.unwrap(r.fn))
    if doc:
        details["doc"] = "\n\n".join(" ".join(p.split()) for p in doc.split("\n\n"))
    return _Box(n.id, "role", w, h, "".join(parts), details)


def _placement(cluster, cta: str) -> tuple[str, str]:
    """Badge and sentence for a ``cta=`` placement, from the ranks it resolves to."""
    if len(cluster.ctas(cta)) == cluster.size:  # includes cta="leader" without pairs
        return f"all {cluster.size} CTAs", "runs on every CTA of the cluster"
    if cta == "leader":
        return "leader CTA", "runs on the leader CTA of each MMA pair"
    return "rank 0", "runs on cluster rank 0 only"


def _chip_badges(names: list[str], kernel, ch) -> list[tuple[str, str]]:
    cluster = kernel.cluster
    badges = []
    if ch.scope.pair:
        badges.append(("pair" if is_pair(cluster) else "x-cta", "pair"))
    if isinstance(ch.producer, Tma) and ch.producer.multicast:
        badges.append(("mcast", "pair"))
    if ch.exclusive:
        badges.append(("excl", "plain"))
    turns = max(ch.turns(side) for side in SIDES)
    if turns > 1:
        badges.append((f"rr\u00d7{turns}", "plain"))
    if ch.commit_fence or ch.release_fence:
        badges.append(("fence", "plain"))
    overrides = _overrides(ch, cluster)
    if overrides:
        differs = any("differs" in o for o in overrides)
        badges.append(("override", "warn" if differs else "plain"))
    if ch.external is not None:
        badges.append(("raw " + ch.external, "warn"))
    reports = [kernel._ws_tma[n] for n in names if n in kernel._ws_tma]
    if reports:
        unverified = any(r.unverified for r in reports)
        badges.append(("tma ?" if unverified else "tma \u2713", "warn" if unverified else "ok"))
    if any(n in kernel._ws_ownership for n in names):
        badges.append(("epochs", "plain"))
    return badges


def _chip_box(cid: str, names: list[str], lines, kernel) -> _Box:
    """A channel is a handoff between roles: a capsule holding its slots, not another card."""
    ch = kernel._ws_chans[names[0]]
    prod = mechanism(ch.producer, ch.named)
    label = ", ".join(names)
    kind = ("slots" if ch.fixed else "ring") + (" each" if len(names) > 1 else "")
    ticks_w = ch.n * 9
    badges = _chip_badges(names, kernel, ch)
    badges_w = sum(_tw(t, 8.5) + 11 for t, _ in badges)
    pad = 7  # keeps the text clear of the rounded ends
    w = max(_tw(label, 11) + 16, ticks_w + _tw(kind, 9.5) + 22, badges_w + 12, 56) + 2 * pad
    h = 34 + (15 if badges else 0)
    parts = [
        f'<rect class="chip-body" width="{w:.1f}" height="{h}" rx="{min(h / 2, 15):.0f}"/>',
        f'<g transform="translate({pad},0)">',
        _text(8, 16, label, "chip-name"),
    ]
    for s in range(ch.n):
        parts.append(f'<rect class="tick {prod}" x="{8 + 9 * s}" y="21" width="7" height="7"/>')
    parts.append(_text(8 + ticks_w + 4, 28, kind, "chip-kind"))
    x = 8.0
    for text, cls in badges:
        pill, bw = _pill(x, 33, text, cls)
        parts.append(pill)
        x += bw + 3
    parts.append("</g>")
    counts = dict(channel_barriers(ch, kernel.cluster))
    handshake = {"slots": ch.n, "kind": kind, "writers": [], "readers": []}  # ends: _handoffs
    for side, word in (("producer", "full"), ("consumer", "empty")):
        handshake[word] = _side(ch, side, counts["named"] if ch.named else counts[word])
    details = {
        "title": label,
        "lines": list(lines),
        "guide": _chip_guide(ch, kernel.cluster, len(names) > 1),
        "handshake": handshake,
    }
    return _Box(cid, "chip", w, h, "".join(parts), details, chan=ch)


def _chip_guide(ch, cluster, grouped: bool) -> dict[str, str]:
    """The inspector's sentences: what the channel is and what completes each barrier."""
    stage = f"{ch.n} independently addressed slots" if ch.fixed else f"{ch.n}-stage ring"
    guide = {"Channel": stage + " per channel" * grouped}
    for side, heading in (("producer", "FULL when"), ("consumer", "EMPTY when")):
        if ch.external == side:
            guide[heading] = "raw code signals it."
            continue
        if side == "consumer" and ch.per_role and not ch.counts_overridden(side):
            said = [
                f"{role}: {_signal_description(ch, side, m)}" for role, m in ch.consumer.items()
            ]
            guide[heading] = " ".join(said) + " Every role must signal."
        else:
            guide[heading] = _signal_description(ch, side, ch.mechanisms(side)[0])
        if any(slot is not None for _, slot in ch.bound(side)):
            guide[heading] += " An instance bound to one slot signals only that slot."
    if ch.commit_fence:
        guide["Before commit"] = " ".join(_FENCE_REASON[f] for f in ch.commit_fence)
    if ch.release_fence:
        guide["Before release"] = " ".join(_FENCE_REASON[f] for f in ch.release_fence)
        if ch.per_role and Umma in ch.consumer.values():
            guide["Before release"] += " Thread roles only: UMMA readers emit no fence."
    if any("differs" in entry for entry in _overrides(ch, cluster)):
        guide["Review the override"] = "Non-default ordering or counts. Check the role body."
    return guide


_FENCE_REASON = {
    "tmem_load": "Finish TMEM reads before a later writer can overwrite those columns.",
    "tmem_store": "Finish TMEM stores before the next user accesses those columns.",
    "smem_async": (
        "Order shared-memory accesses across the generic and async proxies before signalling."
    ),
}


def _side(ch, side: str, count: str) -> dict[str, str]:
    """How ``side`` completes its barrier: a short label, the colour key and the counts."""
    if ch.external == side:
        return {"by": "raw code", "color": "raw", "count": count}
    mechs = ch.mechanisms(side)
    if side == "consumer" and ch.per_role and not ch.counts_overridden(side):
        named = [f"{role}: {_signal_name(ch, side, m)}" for role, m in ch.consumer.items()]
        by = " · ".join(named)
    else:
        by = _signal_name(ch, side, mechs[0])
    colors = {mechanism(m, ch.named) for m in mechs}
    return {"by": by, "color": colors.pop() if len(colors) == 1 else "mixed", "count": count}


def _joins(ch, side: str) -> bool:
    return ch.sharing(side) is Sharing.JOIN


def _signal_name(ch, side: str, mech) -> str:
    if ch.counts_overridden(side):
        return "author-set count"
    if ch.named:
        return "both sides arrive"
    if isinstance(mech, Tma):
        return "the scheduler, no TMA copy" if ch.internal else f"TMA {mech.tx_bytes:,} B per CTA"
    name = "UMMA commit" if mech is Umma else "one lane per warp" if mech is Warps else "threads"
    return name + ("" if _joins(ch, side) else " of the epoch's owner")


def _signal_description(ch, side: str, mech) -> str:
    """Finish the sentence "FULL when ..." (producer) or "EMPTY when ..." (consumer)."""
    doing = "writing" if side == "producer" else "reading"
    if ch.counts_overridden(side):
        return (
            "the author-set arrival count is reached. The role bodies must supply it; "
            "the declared mechanism does not say who arrives."
        )
    if ch.named:
        writers, readers = (
            f"the {verb} threads" if _joins(ch, s) else f"the owning instance's {verb} threads"
            for s, verb in (("producer", "writing"), ("consumer", "reading"))
        )
        said = f"{writers} and {readers} have all arrived at the slot's named barrier."
        return said + "".join(_sharing_note(ch, s) for s in ("producer", "consumer"))
    if isinstance(mech, Tma):
        if ch.internal:
            return "the layer's scheduler completes the barrier. No TMA copy lands here."
        tx = ch.scope.tx
        total = f"; {tx:,} B expected at the barrier" if tx and tx != mech.tx_bytes else ""
        return f"the TMA copies land ({mech.tx_bytes:,} B per CTA{total})."
    join = _joins(ch, side)
    if mech is Umma:
        said = f"the MMAs {doing} it finish. One elected lane issues tcgen05.commit."
    elif mech is Warps:
        done = f"every {doing} warp is" if join else f"the owning instance's {doing} warps are"
        said = f"{done} done. One elected lane per warp arrives."
    elif join:
        said = f"every {doing} thread arrives."
    else:
        said = f"the owning instance's {doing} threads arrive."
    return said + _sharing_note(ch, side)


def _sharing_note(ch, side: str) -> str:
    match ch.sharing(side):
        case Sharing.ROUND_ROBIN:
            return f" {ch.turns(side)} {side} instances take turns: one owns each epoch."
        case Sharing.EXCLUSIVE:
            return " One consumer owns each epoch; the others skip it."
    return ""


def _handoffs(boxes: dict[str, _Box], links: list[_Wire]) -> None:
    """Record each channel's ends once: the chip's writers and readers, and each role's rows.

    The inspector, the role drill-down and the roles-by-channels table all read these.
    """
    for w in links:
        if not w.chip or w.kind not in ("into", "flow"):
            continue
        chip = boxes[w.chip]
        handshake = chip.details["handshake"]
        fills = w.kind == "into"
        end = boxes[w.src if fills else w.dst]
        names = handshake["writers" if fills else "readers"]
        if end.details["title"] not in names:
            names.append(end.details["title"])
        if end.kind != "role":
            continue
        side, mine, theirs = (
            ("producer", "full", "empty") if fills else ("consumer", "empty", "full")
        )
        ch = chip.chan
        bound = [(ri.role, slot) for ri, slot in ch.bound(side) if ri.name == end.details["title"]]
        row = {
            "chip": w.chip,
            "verb": "fills" if fills else "frees",
            "slots": [slot for _, slot in bound if slot is not None],  # empty: every slot
            "waits": {"word": theirs.upper(), **handshake[theirs]},
            "signals": {
                "word": mine.upper(),
                "by": _signal_name(ch, side, ch.mechanism(side, bound[0][0])),
                "color": w.color,
                "count": handshake[mine]["count"],
            },
        }
        if row not in end.details["handoffs"]:
            end.details["handoffs"].append(row)


def _matrix(boxes: dict[str, _Box]) -> str:
    """Roles by channels: which role fills each channel and which frees it.

    Rows follow warp order; columns follow the graph left to right. Cells pick the channel.
    """
    roles = sorted(
        (b for b in boxes.values() if b.kind == "role"), key=lambda b: b.details["warps"][0]
    )
    chips = sorted((b for b in boxes.values() if b.kind == "chip"), key=lambda b: (b.x, b.y))
    if not roles or not chips:
        return ""
    head = "".join(
        f'<th scope="col"><button type="button" data-pick="{c.id}">{_esc(c.details["title"])}'
        f'<span class="slots">{"<i></i>" * c.details["handshake"]["slots"]}</span>'
        f"<small>{_esc(c.details['handshake']['kind'])}</small></button></th>"
        for c in chips
    )
    body = []
    for r in roles:
        d = r.details
        strip = "".join(
            f'<i class="{"on" if w in d["warps"] else ""}{" g" if w and w % 4 == 0 else ""}"></i>'
            for w in range(d["num_warps"])
        )
        cells = "".join(
            "<td>"
            + "".join(
                f'<button type="button" class="use {row["verb"]} {row["signals"]["color"]}" '
                f'data-pick="{c.id}">{row["verb"]}</button>'
                for row in d["handoffs"]
                if row["chip"] == c.id
            )
            + "</td>"
            for c in chips
        )
        span = f"w{d['warps'][0]}" + (f"-{d['warps'][-1]}" if len(d["warps"]) > 1 else "")
        facts = " · ".join(filter(None, [span, d.get("placement")]))
        body.append(
            f'<tr><th scope="row"><button type="button" data-pick="{r.id}">'
            f'{_esc(d["title"])}</button><span class="strip">{strip}</span>'
            f"<small>{_esc(facts)}</small></th>{cells}</tr>"
        )
    return (
        '<section id="matrix"><p class="figlabel">Roles × channels · who fills, who frees</p>'
        f'<div class="matrix-scroll"><table><thead><tr><td></td>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div></section>"
    )


def _hub_box(n) -> _Box:
    w = max(_tw(n.ref, 10.5) + 8, 30)
    cx = w / 2
    match n.kind:
        case "named":
            pts = " ".join(
                f"{cx + 11 * dx:.1f},{11 + 10 * dy:.1f}"
                for dx, dy in ((-1, 0), (-0.5, -1), (0.5, -1), (1, 0), (0.5, 1), (-0.5, 1))
            )
            shape = f'<polygon class="hub" points="{pts}"/>'
        case "turnstile":
            shape = (
                f'<circle class="hub" cx="{cx:.1f}" cy="11" r="10"/>'
                f'<path class="hub-mark" d="M{cx - 5:.1f},11 a5,5 0 1,1 5,5"/>'
            )
        case _:
            shape = (
                f'<path class="hub" d="M{cx - 10:.1f},4 v14 a10,4 0 0,0 20,0 v-14"/>'
                f'<ellipse class="hub" cx="{cx:.1f}" cy="4" rx="10" ry="4"/>'
            )
    svg = shape + _text(cx, 35, n.ref, "hub-name", "middle")
    return _Box(n.id, n.kind, w, 38, svg, {"title": n.ref, "lines": list(n.lines[1:])})


def _external_box(n) -> _Box:
    w = max(_tw(line, 10.5) for line in n.lines) + 18
    h = 14 + 13 * len(n.lines)
    parts = [f'<rect class="external" width="{w:.1f}" height="{h}" rx="{h / 2:.1f}"/>']
    parts += [_text(w / 2, 17 + 13 * i, line, "ext-t", "middle") for i, line in enumerate(n.lines)]
    return _Box(n.id, "external", w, h, "".join(parts), {"title": n.lines[0], "lines": n.lines})


def _stream_box(n) -> _Box:
    name, how = n.lines[0], n.lines[1]
    w = max(_tw(name, 11.5), _tw(how, 9.5)) + 22
    parts = [
        f'<rect class="stream" width="{w:.1f}" height="38" rx="3"/>',
        '<rect class="stream-bar" width="4" height="38" rx="1"/>',
        _text(12, 16, name, "stream-name"),
        _text(12, 30, how, "stream-t"),
    ]
    return _Box(n.id, "stream", w, 38, "".join(parts), {"title": name, "lines": n.lines[1:]})


# Layered layout


def _layout(boxes: dict[str, _Box], links, spokes):
    """Set box rank/x/y; return a port-to-port route for each link and the feedback links.

    Long links pass through virtual nodes. Undirected spokes only pull hubs toward users.
    A link back to a node more than one column earlier (a feedback handoff) leaves the
    columns alone: it runs through the gaps beside them and a lane under the graph. The
    returned set holds the indices of those links.
    """
    ids = list(boxes)
    succ: dict[str, list[str]] = {n: [] for n in ids}
    indeg = dict.fromkeys(ids, 0)
    for a, b in links:
        succ[a].append(b)
        indeg[b] += 1

    # Reverse DFS back edges for ranking; routes later restore the original direction.
    state: dict[str, int] = {}
    back: set[tuple[str, str]] = set()
    found: list[str] = []

    def visit(n: str):
        state[n] = 1
        found.append(n)
        for m in succ[n]:
            if state.get(m) == 1:
                back.add((n, m))
            elif m not in state:
                visit(m)
        state[n] = 2

    linked = {n for link in links for n in link}
    for n in [n for n in ids if indeg[n] == 0] + ids:
        if n in linked and n not in state:
            visit(n)
    dag = [(b, a) if (a, b) in back else (a, b) for a, b in links]

    # Longest-path ranks keep edges forward. Move sources toward their nearest successor
    # to avoid empty columns.
    preds: dict[str, list[str]] = {n: [] for n in ids}
    dsucc: dict[str, list[str]] = {n: [] for n in ids}
    for a, b in dag:
        preds[b].append(a)
        dsucc[a].append(b)
    rank: dict[str, int] = {}

    def rank_of(n: str) -> int:
        if n not in rank:
            rank[n] = max((rank_of(p) + 1 for p in preds[n]), default=0)
        return rank[n]

    for n in linked:
        rank_of(n)
    for n in sorted(linked, key=lambda n: -rank[n]):
        if not preds[n] and dsucc[n]:
            rank[n] = min(rank[m] for m in dsucc[n]) - 1
    users: dict[str, list[str]] = {n: [] for n in ids}
    for a, b in spokes:
        users[a].append(b)
        users[b].append(a)
    for n in ids:
        if n not in linked:  # hubs sit at their users' median rank
            near = [rank[u] for u in users[n] if u in rank]
            rank[n] = round(statistics.median(near)) if near else 0
    low = min(rank.values(), default=0)
    for n in ids:
        boxes[n].rank = rank[n] - low

    # Virtual nodes let long links participate in each crossed rank's ordering.
    first = {n: float(i) for i, n in enumerate(found)}  # initial order: discovery
    chains = []
    for i, (a, b) in enumerate(dag):
        if links[i] in back and boxes[b].rank - boxes[a].rank > 1:
            chains.append(None)  # feedback: routed under the graph by _routes
            continue
        chain = [a]
        for r in range(boxes[a].rank + 1, boxes[b].rank):
            d = f"_d{i}_{r}"
            boxes[d] = _Box(d, "dummy", 0, 6, rank=r)
            first[d] = first[a] + 0.5
            chain.append(d)
        chains.append([*chain, b])
    up: dict[str, list[str]] = {n: [] for n in boxes}
    down: dict[str, list[str]] = {n: [] for n in boxes}
    for chain in filter(None, chains):
        for a, b in itertools.pairwise(chain):
            down[a].append(b)
            up[b].append(a)

    layers: list[list[str]] = [[] for _ in range(max(b.rank for b in boxes.values()) + 1)]
    for n in sorted(boxes, key=lambda n: first.get(n, len(found))):
        layers[boxes[n].rank].append(n)
    order = _order(layers, up, down, users)
    for layer in order:  # a hub sits between its users, so spokes stay short
        for hub in [n for n in layer if boxes[n].kind in _HUB_KINDS]:
            layer.remove(hub)
            at = {n: i for i, n in enumerate(layer)}
            near = sorted(at[u] for u in users[hub] if u in at)
            if near:
                layer.insert(near[(len(near) - 1) // 2] + 1, hub)
            else:
                layer.append(hub)

    x = MARGIN
    col_w = []
    for layer in order:
        width = max((boxes[n].w for n in layer), default=0) or 24
        for n in layer:
            boxes[n].x = x + width / 2
        col_w.append(width)
        x += width + COL_GAP
    for layer in order:
        _place(layer, boxes, [0.0] * len(layer))
    for sweep in range(40):
        for layer in order if sweep % 2 == 0 else order[::-1]:
            want = []
            for n in layer:
                near = [boxes[m].y for m in up[n] + down[n]]
                near += [boxes[m].y for m in users[n]] if boxes[n].kind in _HUB_KINDS else []
                want.append(sum(near) / len(near) if near else boxes[n].y)
            _place(layer, boxes, want)
    top = min(boxes[n].y - boxes[n].h / 2 for n in boxes)
    for b in boxes.values():
        b.y += MARGIN + 6 - top
    return _routes(boxes, links, chains, col_w), {i for i, c in enumerate(chains) if c is None}


def _order(layers, up, down, users) -> list[list[str]]:
    """Use barycenter sweeps, keeping the ordering with the fewest crossings."""

    def pos(order):
        return {n: (i + 0.5) / len(layer) for layer in order for i, n in enumerate(layer)}

    def crossings(order) -> int:
        p = pos(order)
        count = 0
        for layer in order[:-1]:
            segs = [(p[a], p[b]) for a in layer for b in down[a]]
            count += sum(
                1
                for i, s in enumerate(segs)
                for t in segs[i + 1 :]
                if (s[0] - t[0]) * (s[1] - t[1]) < 0
            )
        return count

    best, best_count = [list(layer) for layer in layers], crossings(layers)
    order = [list(layer) for layer in layers]
    for sweep in range(16):
        p = pos(order)
        nbrs = up if sweep % 2 == 0 else down
        for layer in order if sweep % 2 == 0 else order[::-1]:
            keys = {}
            for n in layer:
                near = [p[m] for m in nbrs[n] + users.get(n, [])]
                keys[n] = sum(near) / len(near) if near else p[n]
            layer.sort(key=keys.__getitem__)
            p = pos(order)
        count = crossings(order)
        if count < best_count:
            best, best_count = [list(layer) for layer in order], count
    return best


def _place(layer: list[str], boxes, want: list[float]) -> None:
    """Place centres closest to ``want`` in least squares, preserving order and gaps.

    Use isotonic regression after subtracting each box's minimum offset from the first.
    """
    offsets = [0.0]
    for a, b in itertools.pairwise(layer):
        gap = ROW_GAP if "dummy" not in (boxes[a].kind, boxes[b].kind) else 8
        offsets.append(offsets[-1] + (boxes[a].h + boxes[b].h) / 2 + gap)
    blocks: list[list[float]] = []  # [sum, count]
    for w, o in zip(want, offsets):
        blocks.append([w - o, 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    values = [s / c for s, c in blocks for _ in range(int(c))]
    for n, v, o in zip(layer, values, offsets):
        boxes[n].y = v + o


def _routes(boxes, links, chains, col_w):
    """Route links in their original direction with ports spread along box sides.

    Wires cross columns straight and curve only in the gaps.
    """
    out_ports: dict[str, list[tuple[float, int]]] = {}
    in_ports: dict[str, list[tuple[float, int]]] = {}
    for i, chain in enumerate(chains):
        if chain is None:
            continue
        out_ports.setdefault(chain[0], []).append((boxes[chain[1]].y, i))
        in_ports.setdefault(chain[-1], []).append((boxes[chain[-2]].y, i))
    start, end = {}, {}
    for ports, at in ((out_ports, start), (in_ports, end)):
        for n, items in ports.items():
            items.sort()
            span = boxes[n].h * 0.5
            for k, (_, i) in enumerate(items):
                frac = 0.5 if len(items) == 1 else k / (len(items) - 1)
                at[i] = boxes[n].y + (frac - 0.5) * span
    routes = []
    floor = max(b.y + b.h / 2 for b in boxes.values())
    lanes = 0
    for i, ((a, b), chain) in enumerate(zip(links, chains)):
        if chain is None:
            # Out of the source's right side, down the gap beside its column, along a lane
            # under every box, up the gap before the target's column, into its left side.
            src, dst = boxes[a], boxes[b]
            spread = 5 * lanes  # keeps several feedback wires apart
            down = src.x + col_w[src.rank] / 2 + 8 + spread
            rise = max(4, dst.x - col_w[dst.rank] / 2 - 8 - spread)
            leave = src.y + src.h * (0.2 + 0.1 * (lanes % 3))
            enter = dst.y + dst.h * (0.2 + 0.1 * (lanes % 3))
            lanes += 1
            lane = floor + LANE_GAP * lanes
            routes.append(
                [
                    (src.x + src.w / 2, leave),
                    (down, leave),
                    (down, lane),
                    (rise, lane),
                    (rise, enter),
                    (dst.x - dst.w / 2, enter),
                ]
            )
            continue
        pts = []
        for k, n in enumerate(chain):
            box = boxes[n]
            half = col_w[box.rank] / 2
            if k == 0:
                pts += [(box.x + box.w / 2, start[i]), (box.x + half, start[i])]
            elif k == len(chain) - 1:
                pts += [(box.x - half, end[i]), (box.x - box.w / 2, end[i])]
            else:
                pts += [(box.x - half, box.y), (box.x + half, box.y)]
        pts = [p for k, p in enumerate(pts) if k == 0 or p != pts[k - 1]]
        routes.append(pts if (chain[0], chain[-1]) == (a, b) else pts[::-1])
    return routes


# SVG


def _path(pts) -> str:
    d = [f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"]
    for (x0, y0), (x1, y1) in itertools.pairwise(pts):
        if abs(y1 - y0) < 0.5 or abs(x1 - x0) < 0.5:
            d.append(f"L{x1:.1f},{y1:.1f}")
        else:
            mx = (x1 - x0) / 2
            d.append(f"C{x0 + mx:.1f},{y0:.1f} {x1 - mx:.1f},{y1:.1f} {x1:.1f},{y1:.1f}")
    return " ".join(d)


def _elbow_path(pts, r: float = 12) -> str:
    """Axis-aligned segments joined by rounded corners."""
    d = [f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"]
    for (x0, y0), (x1, y1), (x2, y2) in zip(pts, pts[1:], pts[2:]):
        ax, ay = (x1 - x0, y1 - y0), (x2 - x1, y2 - y1)
        la, lb = abs(ax[0]) + abs(ax[1]), abs(ay[0]) + abs(ay[1])
        k = min(r, la / 2, lb / 2)
        d.append(f"L{x1 - ax[0] / la * k:.1f},{y1 - ax[1] / la * k:.1f}")
        d.append(f"Q{x1:.1f},{y1:.1f} {x1 + ay[0] / lb * k:.1f},{y1 + ay[1] / lb * k:.1f}")
    d.append(f"L{pts[-1][0]:.1f},{pts[-1][1]:.1f}")
    return " ".join(d)


def _clip(box: _Box, toward: tuple[float, float]) -> tuple[float, float]:
    """Where the segment from ``box``'s centre toward a point leaves the box."""
    dx, dy = toward[0] - box.x, toward[1] - box.y
    if dx == dy == 0:
        return box.x, box.y
    t = min(
        (box.w / 2 + 3) / abs(dx) if dx else float("inf"),
        (box.h / 2 + 3) / abs(dy) if dy else float("inf"),
    )
    return box.x + dx * t, box.y + dy * t


def _svg(boxes, links, routes, spokes) -> str:
    reach = max((x for pts in routes for x, _ in pts), default=0)
    width = max(reach, max(b.x + b.w / 2 for b in boxes.values())) + MARGIN + 8
    lowest = max((y for pts in routes for _, y in pts), default=0)
    height = max(lowest, max(b.y + b.h / 2 for b in boxes.values())) + MARGIN
    out = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" id="graph" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" role="img">'
        ),
        "<defs>",
        *(
            f'<marker id="arrow-{c}" class="arrow {c}" viewBox="0 0 8 8" refX="7" refY="4" '
            'markerWidth="12" markerHeight="12" markerUnits="userSpaceOnUse" '
            'orient="auto-start-reverse">'
            '<path d="M0,0 L8,4 L0,8 z"/></marker>'
            for c in ("tma", "umma", "threads", "named", "stream")
        ),
        "</defs>",
    ]
    for w in spokes:
        a, b = boxes[w.src], boxes[w.dst]
        (x0, y0), (x1, y1) = _clip(a, (b.x, b.y)), _clip(b, (a.x, a.y))
        out.append(
            f'<g class="wire spoke" data-ends="{w.src} {w.dst}" data-ref="{w.dst}">'
            f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}"/>'
            + _text((x0 + x1) / 2 + 3, (y0 + y1) / 2 - 3, w.label, "wire-label")
            + "</g>"
        )
    for w, pts in zip(links, routes):
        d = _elbow_path(pts) if w.feedback else _path(pts)
        # A chip sits right after its writer, so only a feedback "into" wire needs a direction.
        headless = w.kind == "into" and not w.feedback
        arrow = "" if headless else f' marker-end="url(#arrow-{w.color})"'
        cls = f"wire {w.kind} {w.color}" + " pair" * w.pair + " feedback" * w.feedback
        ends = f"{w.src} {w.dst}"
        layers = [f'<path class="line" d="{d}"{arrow}/>']
        if w.pair:  # a double wire: the gap is drawn over the thick line
            layers.append(f'<path class="gap" d="{d}"/>')
        label = ""
        if w.feedback:
            (x0, lane), (x1, _) = pts[2:4]
            label += _text((x0 + x1) / 2, lane - 5, "feedback", "wire-label", "middle")
        if w.label:
            lx, ly = pts[-1] if w.kind == "flow" else pts[0]
            anchor = "end" if w.kind == "flow" else "start"
            label = _text(
                lx + (-8 if w.kind == "flow" else 8), ly - 4, w.label, "wire-label", anchor
            )
        ref = f' data-ref="{w.chip}"' if w.chip else ""
        out.append(f'<g class="{cls}" data-ends="{ends}"{ref}>{"".join(layers)}{label}</g>')
    for b in boxes.values():
        if b.kind == "dummy":
            continue
        tip = _esc(b.details.get("title", ""))
        out.append(
            f'<g class="node kind-{b.kind}" id="{b.id}" data-ref="{b.id}" tabindex="0" '
            f'transform="translate({b.x - b.w / 2:.1f},{b.y - b.h / 2:.1f})">'
            f"<title>{tip}</title>{b.svg}</g>"
        )
    out.append("</svg>")
    return "\n".join(out)


def _legend_wire(cls: str, label: str, pair: bool = False) -> str:
    gap = '<path class="gap" d="M2,7 H34"/>' if pair else ""
    return (
        f'<span><svg width="38" height="14"><g class="wire flow {cls}{" pair" * pair}">'
        f'<path class="line" d="M2,7 H34"/>{gap}</g></svg>{label}</span>'
    )


_LEGEND = "".join(
    [
        (
            '<span><svg width="28" height="14"><rect class="card" x="1" y="1" width="26" '
            'height="12" rx="2"/><rect class="role-bar" x="1" y="1" width="3" height="12"/>'
            "</svg>role: warps that run</span>"
        ),
        (
            '<span><svg width="36" height="14"><rect class="chip-body" x="1" y="1" width="34" '
            'height="12" rx="6"/><rect class="tick tma" x="14" y="3.5" width="7" height="7"/>'
            "</svg>channel: a handoff and its stages</span>"
        ),
        _legend_wire("tma", "TMA"),
        _legend_wire("umma", "UMMA (tcgen05)"),
        _legend_wire("threads", "threads / warps"),
        _legend_wire("named", "named barrier"),
        _legend_wire("threads", "cross-CTA (pair)", pair=True),
        _legend_wire("threads feedback", "feedback"),
        (
            '<span><svg width="30" height="14"><rect class="tick tma" x="2" y="3" width="7" '
            'height="7"/><rect class="tick tma" x="11" y="3" width="7" height="7"/><rect '
            'class="tick tma" x="20" y="3" width="7" height="7"/></svg>stages</span>'
        ),
        (
            '<span><svg width="20" height="14"><rect class="warp on" x="2" y="3" width="7" '
            'height="8"/><rect class="warp" x="11" y="3" width="7" height="8"/></svg>'
            "warps of the CTA</span>"
        ),
        (
            '<span><svg width="22" height="14"><line class="spoke-key" x1="2" y1="7" x2="20" '
            'y2="7"/></svg>barrier / turnstile user</span>'
        ),
    ]
)
