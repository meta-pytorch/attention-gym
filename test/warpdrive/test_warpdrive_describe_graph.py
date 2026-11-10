"""Test role and channel graphs from ``describe``.

Toys check edge direction, pair scope, role instances, junctions, raw-code sides, and
hubs. Every example must produce a deterministic, well-formed graph.
"""

import importlib
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from xml.etree import ElementTree

import pytest

pytest.importorskip("cutlass.cute")

from cutlass import cute

from attn_gym.warpdrive import (
    Barriers,
    Cluster,
    ClusterLaunchControl,
    Consumer,
    CtaPair,
    Instance,
    NamedBarrier,
    Producer,
    Ring,
    Threads,
    Tma,
    Turn,
    Turnstile,
    Umma,
    WarpSpecialized,
    role,
)
from attn_gym.warpdrive.examples import (
    gemm,
    grouped_gemm,
    persistent_gemm,
    qwen_megakernel,
    raw_interop,
)

# Node: `  id<open>"line<br>line"<close>`; edge: `  src <link>|"label"| dst`.
NODE = re.compile(r'^  (\w+)(?:\[\[|\[\(|\(\[|\{\{|\[|\()"(.*)"(?:\]\]|\)\]|\]\)|\}\}|\]|\))$')
EDGE = re.compile(r'^  (\w+) (-->|==>|---|===|-\.->|-\.-)(?:\|"(.*)"\|)? (\w+)$')
ENTITIES = {"#quot;": '"', "#lt;": "<", "#gt;": ">", "#35;": "#"}


def unescape(text: str) -> list[str]:
    for entity, char in ENTITIES.items():
        text = text.replace(entity, char)
    return text.split("<br>")


class Mermaid:
    """The nodes (``{id: label lines}``) and edges (``(src, link, label lines, dst)``) of a
    ``describe(format="mermaid")`` chart."""

    def __init__(self, text: str):
        self.nodes, self.edges = {}, []
        for line in text.splitlines():
            if m := NODE.match(line):
                assert m[1] not in self.nodes, f"node {m[1]} declared twice"
                self.nodes[m[1]] = unescape(m[2])
            elif m := EDGE.match(line):
                self.edges.append((m[1], m[2], unescape(m[3]) if m[3] else [], m[4]))

    def node(self, *lines: str) -> str:
        """The one node whose label starts with ``lines``."""
        (nid,) = (n for n, label in self.nodes.items() if label[: len(lines)] == list(lines))
        return nid

    def label(self, first_line: str) -> list[str]:
        """A channel's label lines: its junction node's, or its edge's."""
        for lines in [*self.nodes.values(), *(e[2] for e in self.edges)]:
            if lines and lines[0].startswith(first_line):
                return lines
        raise AssertionError(f"no label starting {first_line!r}")

    def links(self, src: str, dst: str) -> list[tuple[str, list[str]]]:
        """``(link, label)`` of every edge from node ``src`` to node ``dst``."""
        return [(link, lab) for s, link, lab, d in self.edges if (s, d) == (src, dst)]


class PairPipeline(WarpSpecialized):
    """load (both CTAs) --data (TMA, pair)--> mma (leader) --acc (UMMA, pair)--> sink[0],
    sink[1] (both CTAs); load --note (CTA-local)--> log."""

    def __init__(self):
        self.cluster = CtaPair()
        self.data = Ring(2, Tma(4096), Umma)
        self.acc = Ring(1, Umma, Threads)
        self.note = Ring(1)

    @role(warps=0)
    @cute.jit
    def load(self, data: Producer, note: Producer):
        pass

    @role(warps=1, cta="leader")
    @cute.jit
    def mma(self, data: Consumer, acc: Producer):
        pass

    @role(warps=[range(2, 3), range(3, 4)])
    @cute.jit
    def sink(self, acc: Consumer):
        pass

    @role(warps=4)
    @cute.jit
    def log(self, note: Consumer):
        pass


class SyncToy(WarpSpecialized):
    """worker[0], worker[1] take turns on ``order`` and fill ``out``, which raw code consumes;
    ``poll`` consumes ``inbox``, which raw code fills, and takes the ``flags`` mbarriers;
    ``ready`` is a named barrier."""

    def __init__(self):
        self.out = Ring(2, external="consumer", override={"empty_counts": 1})
        self.inbox = Ring(1, external="producer", override={"full_counts": 1})
        self.order = Turnstile()
        self.flags = Barriers(2, counts=1)
        self.ready = NamedBarrier(threads=64)

    @role(warps=[range(1), range(1, 2)])
    @cute.jit
    def worker(self, stage: Instance, out: Producer, order: Turn):
        pass

    @role(warps=2)
    @cute.jit
    def poll(self, inbox: Consumer, flags: Barriers):
        pass


def test_pair_channels_point_from_producer_to_consumers():
    """Draw 1:1 channels as edges and multi-consumer channels as junctions.

    Producers join junctions; every consumer instance leaves them. Cross-CTA edges are
    thick and CTA-local edges thin. Roles show warps and CTA placement. Edge colors follow
    the producer mechanism, or the consumer mechanism for edges leaving a junction.
    """
    text = PairPipeline().describe(format="mermaid")
    g = Mermaid(text)
    styles = {
        int(i): style
        for ids, style in re.findall(r"^  linkStyle ([\d,]+) (.*)$", text, re.MULTILINE)
        for i in ids.split(",")
    }

    def style(src: str, dst: str) -> str:
        (i,) = (i for i, (s, _, _, d) in enumerate(g.edges) if (s, d) == (src, dst))
        return styles[i]

    load, mma, log = (g.node(n) for n in ("load", "mma", "log"))
    sinks = [g.node(f"sink[{i}]") for i in range(2)]
    assert g.nodes[mma] == ["mma", "warp 1 · cta leader"]
    assert g.nodes[sinks[1]] == ["sink[1]", "warp 3 · cta all"]

    ((link, label),) = g.links(load, mma)
    assert link == "==>" and label[:2] == ["data · Ring(2)", "Tma(4096 B) -> Umma"]
    assert "pair-scoped, release_mask 0b11, tail" in label
    assert g.links(mma, load) == []
    assert style(load, mma) == "stroke:#3f7d5c,stroke-width:3.5px"  # TMA, pair

    acc = g.node("acc · Ring(1)")
    assert [link for link, _ in g.links(mma, acc)] == ["==="]  # the producer joins, no arrow
    assert all([link for link, _ in g.links(acc, s)] == ["==>"] for s in sinks)
    assert "pair-scoped, commit_mask 0b11, remote_scope cta" in g.nodes[acc]
    assert style(mma, acc) == "stroke:#3b689c,stroke-width:3.5px"  # UMMA producer joins
    assert style(acc, sinks[0]) == "stroke:#8a5a9e,stroke-width:3.5px"  # to thread consumers

    ((link, label),) = g.links(load, log)
    assert link == "-->" and "CTA-local" in label
    assert style(load, log) == "stroke:#8a5a9e,stroke-width:1.6px"  # CTA-local threads


def test_hubs_join_their_users_and_raw_code_is_a_node():
    """Raw code on an external side is its own node at that end; a turnstile and ``Barriers``
    are hubs joined to their users; before a compile the named barriers are only listed."""
    g = Mermaid(SyncToy().describe(format="mermaid"))
    workers = [g.node(f"worker[{i}]") for i in range(2)]
    poll = g.node("poll")
    out = g.node("out · Ring(2)")
    raw_out = g.node("raw code", "out consumer")
    assert [lab for _, lab in g.links(out, raw_out)] == [[]]
    assert all(g.links(w, out) for w in workers) and g.links(raw_out, out) == []
    assert "external consumer (counts unverified)" in g.nodes[out]
    raw_in = g.node("raw code", "inbox producer")
    ((link, label),) = g.links(raw_in, poll)
    assert link == "-->" and "external producer (counts unverified)" in label
    assert g.links(poll, raw_in) == []

    order = g.node("order")
    assert [g.links(w, order) for w in workers] == [[("-.-", ["turn 0"])], [("-.-", ["turn 1"])]]
    flags = g.node("flags")
    assert g.links(poll, flags) == [("-.-", ["Barriers"])]
    assert "Barriers: 2 mbarriers, layer-initialized [1, 1]" in g.nodes[flags]
    assert g.nodes[g.node("named barriers")][1:] == [
        "ready id 3 · 64 threads",
        "users known after a compile",
    ]


def test_multicast_without_pairs_is_cross_cta():
    """In a 1 x 2 cluster the multicast A/B ring spans both CTAs (thick), while the
    accumulator handoff stays in each CTA (thin)."""
    g = Mermaid(persistent_gemm.PersistentGemm(cluster=Cluster(1, 2)).describe(format="mermaid"))
    load, mma, epilogue = (g.node(n) for n in ("load", "mma", "epilogue"))
    ((link, label),) = g.links(load, mma)
    assert link == "==>" and "cross-CTA, multicast n, release_mask 0b11, tail" in label
    ((link, label),) = g.links(mma, epilogue)
    assert link == "-->" and "CTA-local" in label


def test_broadcast_stream_reaches_roles_over_its_ring():
    """The scheduler claims work from its stream and broadcasts it to every role (itself too)
    over the work ring, whose response slot loops back on the scheduler."""
    op = persistent_gemm.PersistentGemm(schedule=ClusterLaunchControl(), cluster=CtaPair())
    g = Mermaid(op.describe(format="mermaid"))
    sched = g.node("work.scheduler")
    assert g.links(g.node("stream work"), sched) == [("-.->", [])]
    ring = g.node("work_ring · Ring(2)")
    receivers = {g.nodes[d][0] for s, _, _, d in g.edges if s == ring}
    assert receivers == {"load", "mma", "epilogue", "work.scheduler"}
    ((_, label),) = g.links(sched, sched)
    assert label[0] == "work_clc · Slots(1)"


def test_parallel_channels_share_an_edge_only_when_alike():
    """FA4's backward: k, kt, v and do differ only in their buffers, so they share one edge
    from load to mma, listing each buffer; q and qt carry fewer bytes and share another."""
    g = Mermaid(
        optional_example("fa4_bwd", "FlashAttentionBackward2Cta").describe(format="mermaid")
    )
    labels = [lab for _, lab in g.links(g.node("load"), g.node("mma"))]
    assert [lab[0] for lab in labels] == ["k, kt, v, do · Ring(1) each", "q, qt · Ring(1) each"]
    assert labels[0][-1] == "k: buffer K · kt: buffer Kt · v: buffer V · do: buffer dO, dOt"
    ds = g.node("ds · Ring(1)")  # the dS exchange: compute and relay both join it
    assert {g.nodes[s][0] for s, _, _, d in g.edges if d == ds} == {"compute", "relay"}
    assert "smem dKepi overlaps K (k)" in g.nodes[g.node("storage aliases")]


def test_compile_adds_named_barrier_users(requires_blackwell):
    """After a compile each named barrier is a hub joined to its traced users."""
    op = raw_interop.RawInterop()
    raw_interop.compile_raw_interop(op)
    g = Mermaid(op.describe(format="mermaid"))
    done = g.node("done")
    assert g.links(g.node("consumer"), done) == [("-.-", ["arrive"])]
    assert g.links(g.node("watcher"), done) == [("-.-", ["native id"])]
    assert "participation unverified" in g.nodes[done]
    assert g.links(g.node("consumer"), g.node("read")) == [("-.-", ["sync"])]


def test_compile_adds_tma_status(requires_blackwell):
    op = gemm.Gemm()
    g = Mermaid(op.describe(format="mermaid"))
    assert not any(line.startswith("tma") for line in g.label("ab · Ring"))
    gemm.compile_gemm(op)
    g = Mermaid(op.describe(format="mermaid"))
    assert "tma verified" in g.label("ab · Ring")


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="describe format"):
        gemm.Gemm().describe(format="svg")


# describe(format="html")


class Page(HTMLParser):
    """A ``describe(format="html")`` page: its drawn nodes (``{id: (kind, x, y, w, h)}``,
    box from the translate and first rect), wires (``(classes, ends)``), details JSON, and
    any reference to an outside resource."""

    def __init__(self, text: str):
        super().__init__()
        self.nodes, self.wires, self.external = {}, [], []
        self._node, self._data = None, None
        self.feed(text)
        self.svg = ElementTree.fromstring(text[text.index("<svg") : text.index("</svg>") + 6])

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        self.external += [v for k, v in attrs if k in ("src", "href") and v]
        if tag == "link" or (tag == "script" and "src" in a):
            self.external.append(tag)
        classes = a.get("class", "").split()
        if tag == "g" and "node" in classes:
            x, y = map(
                float, re.match(r"translate\(([-\d.]+),([-\d.]+)\)", a["transform"]).groups()
            )
            self._node = a["id"]
            self.nodes[a["id"]] = [classes[1].removeprefix("kind-"), x, y, None, None]
        elif (
            tag == "rect"
            and self._node
            and self.nodes[self._node][3] is None
            and "back" not in classes
        ):
            self.nodes[self._node][3:] = float(a["width"]), float(a["height"])
        elif tag == "g" and "data-ends" in a:  # a drawn wire (not the legend's samples)
            self.wires.append((set(classes), a["data-ends"].split()))
        elif tag == "script" and a.get("id") == "data":
            self._data = []

    def handle_endtag(self, tag):
        if tag == "g" and self._node:
            self._node = None
        if tag == "script" and self._data is not None:
            self.details = json.loads("".join(self._data))
            self._data = None

    def handle_data(self, data):
        if self._data is not None:
            self._data.append(data)

    def by_title(self, title: str) -> str:
        (nid,) = (n for n, d in self.details.items() if d["title"] == title)
        return nid

    def ends(self, a: str, b: str) -> list[set[str]]:
        return [classes for classes, ends in self.wires if ends == [a, b]]


def test_html_draws_channels_as_chips_between_their_roles():
    """A 1:1 channel is a chip the producer wires into and that wires to the consumer, in its
    mechanisms' colours; a pair-scoped channel's wires are double; a junction chip wires to
    every consumer instance."""
    g = Page(PairPipeline().describe(format="html"))
    load, mma, log = (g.by_title(n) for n in ("load", "mma", "log"))
    data = g.by_title("data")
    assert g.ends(load, data) == [{"wire", "into", "tma", "pair"}]
    assert g.ends(data, mma) == [{"wire", "flow", "umma", "pair"}]
    note = g.by_title("note")
    assert g.ends(load, note) == [{"wire", "into", "threads"}]
    assert g.ends(note, log) == [{"wire", "flow", "threads"}]
    acc = g.by_title("acc")
    assert all(g.ends(acc, g.by_title(f"sink[{i}]")) for i in range(2))
    assert "cta leader: runs on the leader CTA of each MMA pair" in g.details[mma]["lines"]


def test_html_hubs_and_raw_code():
    """Hubs join their users; raw code is a node at its channel's external end."""
    g = Page(SyncToy().describe(format="html"))
    order = g.by_title("order")
    assert g.nodes[order][0] == "turnstile"
    assert all(g.ends(g.by_title(f"worker[{i}]"), order) for i in range(2))
    assert g.ends(g.by_title("poll"), g.by_title("flags"))
    raw = {g.details[n]["lines"][1]: n for n, node in g.nodes.items() if node[0] == "external"}
    assert g.ends(g.by_title("out"), raw["out consumer"])
    assert g.ends(raw["inbox producer"], g.by_title("inbox"))


def optional_example(module: str, name: str, *args):
    pytest.importorskip("flash_attn.cute")
    if module in ("fa4_fwd", "fa4_decode_swapab"):
        pytest.importorskip("quack")
    example = importlib.import_module(f"attn_gym.warpdrive.examples.{module}")
    return getattr(example, name)(*args)


EXAMPLES = {
    "gemm": gemm.Gemm,
    "persistent_gemm": persistent_gemm.PersistentGemm,
    "persistent_gemm_pair_clc": lambda: persistent_gemm.PersistentGemm(
        schedule=ClusterLaunchControl(), cluster=CtaPair()
    ),
    "grouped_gemm": grouped_gemm.GroupedGemm,
    "fa4_fwd": lambda: optional_example("fa4_fwd", "FlashAttentionForward"),
    "fa4_bwd": lambda: optional_example("fa4_bwd", "FlashAttentionBackward"),
    "fa4_bwd_2cta": lambda: optional_example("fa4_bwd", "FlashAttentionBackward2Cta"),
    "decode": lambda: optional_example("fa4_decode_swapab", "SwapABDecode", 32),
    "decode_shared_kv": lambda: optional_example("fa4_decode_swapab", "SharedKVDecode", 8),
    "qwen_megakernel": qwen_megakernel.QwenDecodeMegakernel,
    "raw_interop": raw_interop.RawInterop,
}


@pytest.mark.parametrize("make", EXAMPLES.values(), ids=EXAMPLES.keys())
def test_every_example_draws_well_formed_graphs(make):
    """Every format must draw the complete graph deterministically.

    Mermaid has one node per role instance, labels every channel, and joins only declared
    nodes. DOT parses with Graphviz when installed. HTML is self-contained, has valid SVG,
    draws every role and exactly one chip per channel, joins only drawn nodes, and keeps
    cards and chips from overlapping.
    """
    op, twin = make(), make()
    for fmt in ("mermaid", "dot", "html"):
        assert twin.describe(format=fmt) == op.describe(format=fmt)
    text = op.describe(format="mermaid")
    g = Mermaid(text)
    body = [
        line
        for line in text.splitlines()[3:]
        if not line.lstrip().startswith(("class", "linkStyle"))
    ]
    assert len(body) == len(g.nodes) + len(g.edges), "a line is neither a node nor an edge"
    assert all(s in g.nodes and d in g.nodes for s, _, _, d in g.edges)
    styled = [
        int(i)
        for i in re.findall(r"^  linkStyle ([\d,]+) ", text, re.MULTILINE)
        for i in i.split(",")
    ]
    assert sorted(styled) == list(range(len(g.edges))), "each link styled once (or Mermaid fails)"
    assert {r.name for r in op._ws_roles} <= {lines[0] for lines in g.nodes.values()}
    heads = [lab[0] for lab in [*g.nodes.values(), *(e[2] for e in g.edges)] if lab]
    names = {n.strip() for head in heads for n in head.split(" · ")[0].split(",")}
    assert set(op._ws_chans) <= names
    if shutil.which("dot"):
        dot = op.describe(format="dot")
        subprocess.run(["dot", "-Tsvg"], input=dot, text=True, capture_output=True, check=True)
    g = Page(op.describe(format="html"))
    assert g.external == []
    titles = {d["title"] for d in g.details.values()}
    assert {r.name for r in op._ws_roles} <= titles
    chips = [g.details[n]["title"] for n, node in g.nodes.items() if node[0] == "chip"]
    names = [name for title in chips for name in title.split(", ")]
    assert sorted(names) == sorted(op._ws_chans)
    assert all(set(ends) <= set(g.nodes) for _, ends in g.wires)
    boxes = [n[1:] for n in g.nodes.values() if n[0] in ("role", "chip")]
    for i, (x0, y0, w0, h0) in enumerate(boxes):
        for x1, y1, w1, h1 in boxes[i + 1 :]:
            assert x0 + w0 <= x1 or x1 + w1 <= x0 or y0 + h0 <= y1 or y1 + h1 <= y0
