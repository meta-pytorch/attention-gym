"""Graph view of a ``WarpSpecialized`` kernel: ``describe(format="mermaid" | "dot")``.

The graph shows what ``describe()`` prints, from the same collected state: role instances are
nodes, and each channel is an edge from its producer to its consumer (through a junction node
when either side has several roles or a slot binding), labelled with its stages, mechanisms,
arrival counts, scope and storage. Cross-CTA channels are drawn thick. Raw code driving an
external side, the work streams and ``kernel code`` native callers are nodes of their own.
Synchronization that is not a handoff (named barriers, turnstiles, ``Barriers``) is a hub joined
to its users by undirected dotted edges, and one note lists the SMEM/TMEM aliases. Compile-only
facts (TMA check status, epoch ownership, named-barrier users) appear once a compile has
recorded them. ``describe_html`` draws the same model as a self-contained page.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from attn_gym.warpdrive.channels import SIDES, Barriers, Tma, Umma
from attn_gym.warpdrive.describe import (
    _overrides,
    channel_barriers,
    channel_kind,
    channel_scope,
    channel_signal,
    channel_storage,
)
from attn_gym.warpdrive.scheduling import scheduler_role

FORMATS = ("mermaid", "dot")

# channel: a junction; named_list: the named barriers before a compile names their users.
NodeKind = Literal[
    "role",
    "channel",
    "stream",
    "external",
    "named",
    "turnstile",
    "named_list",
    "barriers",
    "storage",
]
# into: a producer joining a channel junction (no arrowhead); flow: into a consumer;
# stream: a work stream to its users; spoke: a hub's user (undirected).
EdgeKind = Literal["into", "flow", "stream", "spoke"]


@dataclass(frozen=True)
class Node:
    id: str
    lines: tuple[str, ...]
    kind: NodeKind
    ref: str = ""  # the role, channel, stream or hub it draws (empty for notes and raw code)


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    lines: tuple[str, ...] = ()
    kind: EdgeKind = "flow"
    cross_cta: bool = False
    channels: tuple[str, ...] = ()  # the channels a handoff edge carries


@dataclass
class Graph:
    title: str
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)

    def add(self, kind: NodeKind, key: str, lines, ref: str = "") -> str:
        """Add a node; its id is ``kind_key`` made identifier-safe and unique."""
        base = f"{kind}_{re.sub(r'[^0-9A-Za-z]+', '_', key).strip('_')}"
        taken = {n.id for n in self.nodes}
        nid, k = base, 1
        while nid in taken:
            nid, k = f"{base}_{k}", k + 1
        self.nodes.append(Node(nid, tuple(lines), kind, ref))
        return nid


def describe_graph(kernel, fmt: str) -> str:
    """``kernel.describe(format=fmt)`` for the graph formats: ``FORMATS`` or ``"html"``."""
    if fmt == "html":
        # describe_html builds on this module's graph model.
        from attn_gym.warpdrive.describe_html import describe_html

        return describe_html(kernel)
    if fmt not in FORMATS:
        raise ValueError(
            f"describe format must be 'text', 'html' or one of {FORMATS}, got {fmt!r}"
        )
    graph = build_graph(kernel)
    return to_mermaid(graph, kernel._ws_chans) if fmt == "mermaid" else to_dot(graph)


def mechanism(mech, named: bool = False) -> str:
    """The colour key of a signal mechanism: ``tma``, ``umma``, ``threads`` (threads or warps)
    or ``named`` (a named-barrier channel)."""
    if named:
        return "named"
    if isinstance(mech, Tma):
        return "tma"
    return "umma" if mech is Umma else "threads"


def wire_mechanism(graph: Graph, edge: Edge, chans: dict) -> str:
    """The mechanism a handoff edge carries: its consumers' from a junction to a consumer,
    else its producers' (``""`` for stream and hub edges)."""
    if not edge.channels:
        return ""
    ch = chans[edge.channels[0]]
    from_junction = any(n.id == edge.src and n.kind == "channel" for n in graph.nodes)
    return mechanism(ch.consumer if from_junction else ch.producer, ch.named)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def build_graph(kernel) -> Graph:
    cluster = kernel.cluster
    graph = Graph(
        f"{type(kernel).__name__}: {kernel.num_warps} warps, {cluster.describe()}",
    )
    roles = {
        r.name: graph.add("role", r.name, _role_lines(r, cluster), r.name)
        for r in kernel._ws_roles
    }
    _streams(kernel, graph, roles)
    _channels(kernel, graph, roles)
    _hubs(kernel, graph, roles)
    aliases = _aliases(kernel)
    if aliases:
        graph.add("storage", "aliases", ["storage aliases", *aliases])
    return graph


def _warps(warps: range) -> str:
    return f"warp {warps.start}" if len(warps) == 1 else f"warps {warps.start}-{warps.stop - 1}"


def _role_lines(r, cluster) -> list[str]:
    facts = [_warps(r.warps)]
    if r.regs is not None:
        facts.append(f"{r.regs} regs")
    if cluster.size > 1:
        facts.append(f"cta {r.cta}")
    if r.raw:
        facts.append("raw")
    return [r.name, " · ".join(facts)]


def _streams(kernel, graph: Graph, roles: dict) -> None:
    for name, st in kernel._ws_streams.items():
        sched = next(
            (
                r.name
                for r in kernel._ws_roles
                if r.fn is scheduler_role and r.role == f"{name}.scheduler"
            ),
            None,
        )
        users = [r for r in st.roles if r != sched]
        # A broadcast reaches its users over the work ring; a stream every role decodes
        # needs no edges.
        if sched is not None:
            how, targets = f"claimed by {sched}, broadcast over {name}_ring", [sched]
        elif set(users) == set(roles):
            how, targets = "decoded locally in every role", []
        else:
            how, targets = "decoded locally", users
        sid = graph.add("stream", name, [f"stream {name}", st.describe(), how], name)
        graph.edges += [Edge(sid, roles[r], kind="stream") for r in targets]


def _channels(kernel, graph: Graph, roles: dict) -> None:
    """One edge per channel with one producer and one consumer, else a junction node joining
    them. Parallel channels that differ only in name and storage share one edge."""
    parallel = {}  # (producer, consumer, cross-CTA, kind, facts) -> [(name, storage)]
    for name, ch in kernel._ws_chans.items():
        facts = _channel_facts(kernel, name, ch)
        storage = ", ".join(channel_storage(ch))
        producers, consumers = (_ends(graph, roles, name, ch, side) for side in SIDES)
        cross = ch.scope.pair
        if len(producers) == len(consumers) == 1 and producers[0][1] is consumers[0][1] is None:
            key = (producers[0][0], consumers[0][0], cross, channel_kind(ch), facts)
            parallel.setdefault(key, []).append((name, storage))
            continue
        lines = [f"{name} · {channel_kind(ch)}", *facts, *([storage] if storage else [])]
        junction = graph.add("channel", name, lines, name)
        on = (name,)
        for nid, slot in producers:
            graph.edges.append(Edge(nid, junction, _slot(slot), "into", cross, on))
        for nid, slot in consumers:
            graph.edges.append(Edge(junction, nid, _slot(slot), "flow", cross, on))
    for (src, dst, cross, kind, facts), chans in parallel.items():
        names = ", ".join(n for n, _ in chans)
        stores = {s for _, s in chans}
        if stores == {""}:
            storage = []
        elif len(stores) == 1:
            storage = [chans[0][1]]
        else:
            storage = [" · ".join(f"{n}: {s}" for n, s in chans if s)]
        head = f"{names} · {kind}" + (" each" if len(chans) > 1 else "")
        on = tuple(n for n, _ in chans)
        graph.edges.append(Edge(src, dst, (head, *facts, *storage), "flow", cross, on))


def _ends(graph: Graph, roles: dict, name: str, ch, side: str) -> list[tuple[str, int | None]]:
    """``(node, bound slot)`` of each end on ``side``: its roles, or a raw-code node."""
    if ch.external == side:
        return [(graph.add("external", f"{name}_{side}", ["raw code", f"{name} {side}"]), None)]
    bound = ch.producers if side == "producer" else ch.consumers
    return [(roles[r.name], slot) for r, slot in bound]


def _channel_facts(kernel, name: str, ch) -> tuple[str, ...]:
    """A channel's label lines after its name and before its storage."""
    cluster = kernel.cluster
    lines = [channel_signal(ch)]
    lines += [f"{key} {value}" for key, value in channel_barriers(ch, cluster)]
    if ch.exclusive:
        lines.append("exclusive consumers")
    for side in SIDES:
        if ch.turns(side) > 1:
            lines.append(f"{side}s round-robin x{ch.turns(side)}")
    if ch.external is not None:
        lines.append(f"external {ch.external} (counts unverified)")
    scope = channel_scope(ch, cluster)
    if scope:
        lines.append(", ".join(scope))
    lines += [f"override {o}" for o in _overrides(ch, cluster)]
    report = kernel._ws_tma.get(name)
    if report is not None:
        lines.append(f"tma {_tma_verdict(report)}")
    verdict = kernel._ws_ownership.get(name)
    if verdict is not None:
        lines.append(f"epochs {verdict}")
    return tuple(lines)


def _tma_verdict(report) -> str:
    """The TMA check's outcome without its copy sites (and reasons, which ``describe()``
    lists)."""
    if report.unverified:
        return "unverified"
    if report.copies:
        return "verified" + ("; some barriers opted out" if report.opted_out else "")
    return report.status()  # opted out, or no copies: nothing to drop


def _slot(slot: int | None) -> tuple[str, ...]:
    return () if slot is None else (f"slot {slot}",)


def _hubs(kernel, graph: Graph, roles: dict) -> None:
    """Named barriers, turnstiles and ``Barriers``, each joined to the roles that use it."""
    traced = getattr(kernel, "_ws_traced", False)
    outside = None  # the "kernel code" node, added for the first native caller outside roles
    named = kernel._ws_named_bars
    if named and not traced:  # no users to join yet: list them in one node
        lines = [
            f"{name} id {nb._allocated_id()} · {nb.threads} threads" for name, nb in named.items()
        ]
        graph.add(
            "named_list", "barriers", ["named barriers", *lines, "users known after a compile"]
        )
        named = {}
    for name, nb in named.items():
        lines = [name, f"named barrier id {nb._allocated_id()} · {nb.threads} threads"]
        users = kernel._ws_named_users.get(name, {})
        if not any(users.get(op) for op in ("arrive", "sync", "native")):
            lines.append("unused")
        if users.get("unverified"):
            lines.append("participation unverified")
        hub = graph.add("named", name, lines, name)
        for op, label in (("arrive", "arrive"), ("sync", "sync"), ("native", "native id")):
            for user in users.get(op, ()):
                if user not in roles and outside is None:
                    outside = graph.add("external", "kernel code", ["kernel code"])
                end = roles.get(user, outside)
                graph.edges.append(Edge(end, hub, (label,), kind="spoke"))
    for name, ts in kernel._ws_turns.items():
        hub = graph.add(
            "turnstile", name, [name, "turnstile" + ("" if ts.enabled else " (disabled)")], name
        )
        graph.edges += [
            Edge(roles[r.name], hub, (f"turn {i}",), kind="spoke")
            for i, r in enumerate(ts.parties)
        ]
    for name, b in kernel._ws_raw_bars.items():
        init = "user-initialized" if b.counts is None else f"layer-initialized {b.counts}"
        users = [r.name for r in kernel._ws_roles if (name, Barriers) in r.params]
        lines = [name, f"Barriers: {b.n} mbarriers, {init}", "user-driven (unchecked)"]
        if not users:
            lines.append("roles reach it through barrier_ptr")
        hub = graph.add("barriers", name, lines, name)
        graph.edges += [Edge(roles[r], hub, ("Barriers",), kind="spoke") for r in users]


def _aliases(kernel) -> list[str]:
    """SMEM buffers that overlap another or that several channels bind, and TMEM regions
    placed inside another, each with the channels binding the buffers."""
    lines = []
    plan = kernel.smem_plan
    if plan is not None:
        chans = {
            name: [
                n
                for n, ch in kernel._ws_chans.items()
                if any(b is buf for b in ch.buffers.values())
            ]
            for name, buf in plan.buffers.items()
        }

        def bound(name: str) -> str:
            return f" ({', '.join(chans[name])})" if chans[name] else ""

        for name, buf in plan.buffers.items():
            if buf.overlaps is not None:
                root = buf.overlaps.name
                lines.append(f"smem {name}{bound(name)} overlaps {root}{bound(root)}")
            elif len(chans[name]) > 1:
                lines.append(f"smem {name} shared by {', '.join(chans[name])}")
    if kernel.tmem_plan is not None:
        lines += [
            f"tmem {r.name} inside {r.alias_of}"
            for r in kernel.tmem_plan.regions.values()
            if r.alias_of is not None
        ]
    return lines


# --------------------------------------------------------------------------- #
# Mermaid
# --------------------------------------------------------------------------- #

_MERMAID_SHAPES = {
    "role": ('["', '"]'),
    "channel": ('("', '")'),
    "stream": ('[["', '"]]'),
    "external": ('(["', '"])'),
    "named": ('{{"', '"}}'),
    "turnstile": ('{{"', '"}}'),
    "named_list": ('["', '"]'),
    "barriers": ('[("', '")]'),
    "storage": ('["', '"]'),
}
_MERMAID_CLASSES = {
    "role": "fill:#fbf9f4,stroke:#1a2e22,color:#1a2e22",
    "channel": "fill:#eae6de,stroke:#6a7a6e,color:#1a2e22",
    "stream": "fill:#eef2ec,stroke:#5f7f67,color:#1a2e22",
    "external": "fill:#fbf9f4,stroke:#6a7a6e,stroke-dasharray:4 3,color:#1a2e22",
    "sync": "fill:#f6efe3,stroke:#a26f22,color:#1a2e22",
    "storage": "fill:#fbf9f4,stroke:#b9b4aa,stroke-dasharray:2 2,color:#6a7a6e",
}
# Wire colours by signal mechanism (describe_html's palette); thick wires cross CTAs.
_MERMAID_WIRES = {
    "tma": "#3f7d5c",
    "umma": "#3b689c",
    "threads": "#8a5a9e",
    "named": "#a26f22",
    "": "#6a7a6e",
}
_MERMAID_INIT = (
    '%%{init: {"theme": "base", "flowchart": {"wrappingWidth": 400, "curve": "basis"}, '
    '"themeVariables": {"fontSize": "13px", "lineColor": "#6a7a6e", '
    '"edgeLabelBackground": "#fbf9f4", '
    '"primaryColor": "#fbf9f4", "primaryBorderColor": "#1a2e22", '
    '"primaryTextColor": "#1a2e22"}}}%%'
)
# (edge kind, crosses CTAs) -> link: thick links cross CTAs.
_MERMAID_LINKS = {
    ("into", False): "---",
    ("into", True): "===",
    ("flow", False): "-->",
    ("flow", True): "==>",
    ("stream", False): "-.->",
    ("spoke", False): "-.-",
}
_SYNC_KINDS = ("named", "turnstile", "named_list", "barriers")


def _mermaid_text(lines) -> str:
    escaped = (
        line.replace("#", "#35;").replace('"', "#quot;").replace("<", "#lt;").replace(">", "#gt;")
        for line in lines
    )
    return "<br>".join(escaped)


def to_mermaid(graph: Graph, chans: dict) -> str:
    # Labels are explicit lines: widen Mermaid's 200px wrapping so they keep them.
    out = [
        _MERMAID_INIT,
        f"%% {graph.title}",
        "flowchart LR",
    ]
    for n in graph.nodes:
        left, right = _MERMAID_SHAPES[n.kind]
        out.append(f"  {n.id}{left}{_mermaid_text(n.lines)}{right}")
    for e in graph.edges:
        label = f'|"{_mermaid_text(e.lines)}"|' if e.lines else ""
        out.append(f"  {e.src} {_MERMAID_LINKS[e.kind, e.cross_cta]}{label} {e.dst}")
    by_class = {}
    for n in graph.nodes:
        by_class.setdefault("sync" if n.kind in _SYNC_KINDS else n.kind, []).append(n.id)
    for cls, ids in by_class.items():
        out.append(f"  classDef {cls} {_MERMAID_CLASSES[cls]}")
        out.append(f"  class {','.join(ids)} {cls}")
    by_wire = {}  # (colour, thick) -> link indices
    for i, e in enumerate(graph.edges):
        by_wire.setdefault((wire_mechanism(graph, e, chans), e.cross_cta), []).append(str(i))
    for (mech, thick), ids in by_wire.items():
        width = "3.5px" if thick else "1.6px"
        out.append(
            f"  linkStyle {','.join(ids)} stroke:{_MERMAID_WIRES[mech]},stroke-width:{width}"
        )
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Graphviz DOT
# --------------------------------------------------------------------------- #

_DOT_NODES = {
    "role": 'shape=box, style=filled, fillcolor="#e8f0fe", color="#3b6db3"',
    "channel": 'shape=box, style="rounded,filled", fillcolor="#f4f4f4", color="#777777"',
    "stream": 'shape=component, style=filled, fillcolor="#e6f4ea", color="#3a7d44"',
    "external": 'shape=box, style="rounded,dashed", color="#888888"',
    "named": 'shape=hexagon, style=filled, fillcolor="#fdf2e0", color="#b26a00"',
    "turnstile": 'shape=hexagon, style=filled, fillcolor="#fdf2e0", color="#b26a00"',
    "named_list": 'shape=box, style=filled, fillcolor="#fdf2e0", color="#b26a00"',
    "barriers": 'shape=cylinder, style=filled, fillcolor="#fdf2e0", color="#b26a00"',
    "storage": 'shape=note, style=dashed, color="#aaaaaa"',
}
_DOT_EDGES = {
    "into": ["dir=none"],
    "flow": [],
    "stream": ["style=dashed"],
    "spoke": ["dir=none", "style=dotted"],
}


def _dot_text(lines) -> str:
    return "\\n".join(line.replace("\\", "\\\\").replace('"', '\\"') for line in lines)


def to_dot(graph: Graph) -> str:
    out = [
        "digraph warpdrive {",
        f'  label="{_dot_text([graph.title])}"; labelloc=t; rankdir=LR;',
        "  node [fontname=Helvetica, fontsize=10];",
        "  edge [fontname=Helvetica, fontsize=9];",
    ]
    for n in graph.nodes:
        out.append(f'  {n.id} [label="{_dot_text(n.lines)}", {_DOT_NODES[n.kind]}];')
    for e in graph.edges:
        attrs = list(_DOT_EDGES[e.kind])
        if e.lines:
            attrs.append(f'label="{_dot_text(e.lines)}"')
        if e.cross_cta:
            attrs.append("penwidth=2.5")
        out.append(f"  {e.src} -> {e.dst}" + (f" [{', '.join(attrs)}];" if attrs else ";"))
    out.append("}")
    return "\n".join(out)
