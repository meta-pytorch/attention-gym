"""Host-only tests for declarative CuTeDSL pipeline analysis and Perfetto export."""

from __future__ import annotations

import importlib
import inspect
import itertools
import json
import pkgutil
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass")  # attn_gym.warpdrive.profiler imports CuTeDSL
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace, TrackEvent

import attn_gym.warpdrive.profiler.pipeline as pipeline_package
import attn_gym.warpdrive.profiler.pipeline.annotations as pipeline_annotations
from attn_gym.testing.perfetto_trace import read_slices
from attn_gym.warpdrive.profiler import parse_iket
from attn_gym.warpdrive.profiler.pipeline import (
    Dependency,
    MeasuredCapture,
    MeasuredRegion,
    PipelineAnnotations,
    PipelinePlan,
    Region,
    Resource,
    Role,
    analyze_pipeline,
    extract_plan,
    join,
    report,
    write_pipeline_perfetto,
)
from attn_gym.warpdrive.profiler.pipeline.analysis import ring_occupancy_samples

ANNOTATED_SOURCE = """
PIPELINE = PipelineAnnotations("demo", iteration_name="chunk")
PIPELINE.resource(
    "x", label="X ring", depth=1
)
PIPELINE.resource(
    "state", label="State", depth=1, kind="state"
)

class Kernel:
    @PIPELINE.role("load", label="Load", warp_start=0, warp_end=1)
    def run_load(self):
        PIPELINE.region(
            "load.x", label="Load X", weight=1, produces=("x",)
        )
        PIPELINE.iteration_end()

    @PIPELINE.role("compute", label="Compute", warp_start=1, warp_end=2)
    def run_compute(self):
        PIPELINE.region(
            "compute.x", label="Consume X", weight=3,
            consumes=("x", "state"), produces=("state@1",), releases=("x",)
        )
        PIPELINE.iteration_end()
"""


def toy_plan() -> PipelinePlan:
    """Build a depth-one, two-role pipeline with a known cycle ratio."""
    plan = PipelinePlan("toy", iteration_name="chunk")
    plan.add_role(Role("load", "Load", 0, 1))
    plan.add_role(Role("compute", "Compute", 1, 2))
    plan.add_resource(Resource("x", "X ring", 1))
    plan.add_region(Region("load.x", "load", "Load X", 0, 1))
    plan.add_region(Region("compute.x", "compute", "Consume X", 0, 3))
    plan.add_dependency(Dependency("load.x", "compute.x", "x"))
    plan.add_dependency(Dependency("compute.x", "load.x", "x", distance=1, kind="reuse"))
    return plan


def logical_capture(timeline, *, tick_ns: int = 100) -> MeasuredCapture:
    regions = tuple(
        MeasuredRegion(
            item.region.name,
            item.region.role,
            item.iteration,
            item.start * tick_ns,
            item.end * tick_ns,
            float((item.end - item.start) * tick_ns),
            1,
        )
        for item in timeline.regions
    )
    return MeasuredCapture(
        (0, 0, 0),
        0,
        timeline.logical_duration * tick_ns,
        regions,
    )


def one_role_plan(name: str) -> PipelinePlan:
    """Regions ``a`` then ``b`` on one single-warp role."""
    plan = PipelinePlan(name, iteration_name="tile")
    plan.add_role(Role("role", "Role", 0, 1))
    plan.add_region(Region("a", "role", "A", 0, 1))
    plan.add_region(Region("b", "role", "B", 1, 1))
    return plan


def capture(duration_ns: int, regions) -> MeasuredCapture:
    """CTA 0 from ``(name, iteration, start, end)`` regions, each region on its own role."""
    return MeasuredCapture(
        (0, 0, 0),
        0,
        duration_ns,
        tuple(MeasuredRegion(n, n, i, s, e, float(e - s), 1) for n, i, s, e in regions),
    )


def test_annotations_extract_data_state_and_reuse(tmp_path: Path) -> None:
    source = tmp_path / "annotated.py"
    source.write_text(ANNOTATED_SOURCE)

    plan = extract_plan(source)

    assert list(plan.roles) == ["load", "compute"]
    assert {(edge.resource, edge.kind, edge.distance) for edge in plan.dependencies} == {
        ("x", "data", 0),
        ("state", "state", 1),
        ("x", "reuse", 1),
    }
    assert plan.schedule(iterations=3).logical_duration > 0


def test_runtime_annotations_are_noops_until_iket_is_enabled() -> None:
    annotations = PipelineAnnotations("test", iteration_name="tile")

    @annotations.role("load", label="Load", warp_start=0, warp_end=1)
    def body(value: int) -> int:
        annotations.region("work", label="Work", weight=1)
        return value + 1

    assert body(2) == 3
    assert annotations.resource("x", label="X", depth=1) is None


def test_scheduler_rejects_same_iteration_cycle() -> None:
    plan = PipelinePlan("cycle", iteration_name="tile")
    plan.add_role(Role("a", "A", 0, 1))
    plan.add_role(Role("b", "B", 1, 2))
    plan.add_resource(Resource("x", "X", 1))
    plan.add_region(Region("a.work", "a", "A", 0, 1))
    plan.add_region(Region("b.work", "b", "B", 0, 1))
    plan.add_dependency(Dependency("a.work", "b.work", "x"))
    plan.add_dependency(Dependency("b.work", "a.work", "x"))

    with pytest.raises(ValueError, match="cycle"):
        plan.schedule(iterations=1)


def test_semantic_edges_survive_role_edge_deduplication() -> None:
    plan = one_role_plan("same-role")
    plan.add_resource(Resource("x", "X", 1))
    plan.add_resource(Resource("y", "Y", 1))
    plan.add_dependency(Dependency("a", "b", "x"))
    plan.add_dependency(Dependency("a", "b", "y"))

    timeline = plan.schedule(iterations=1)

    assert [item.dependency.resource for item in timeline.dependencies] == ["x", "y"]


def write_iket(path: Path, ranges: list[tuple[str, int, int, int]]) -> Path:
    """Write an IKET JSON capture of CTA (0, 0, 0) from ``(name, warp, start, end)`` ranges."""
    names = sorted({name for name, *_ in ranges})
    warps = sorted({warp for _, warp, *_ in ranges})
    path.write_text(
        json.dumps(
            {
                "stringTable": names,
                "locationTable": [{"ctaId": [0, 0, 0], "warpId": warp} for warp in warps],
                "launches": [
                    {
                        "kernelName": "toy",
                        "ranges": [
                            {
                                "startTs": start,
                                "endTs": end,
                                "rangeNameIdx": names.index(name),
                                "warpLocIdxs": [warps.index(warp)],
                            }
                            for name, warp, start, end in ranges
                        ],
                    }
                ],
            }
        )
    )
    return path


def logical_iket(path: Path, timeline) -> Path:
    """Encode ``timeline`` at 10 ns per tick: warp 0 loads, warp 1 computes."""
    return write_iket(
        path,
        [
            (
                item.region.name,
                0 if item.region.role == "load" else 1,
                1_000 + item.start * 10,
                1_000 + item.end * 10,
            )
            for item in timeline.regions
        ],
    )


def test_iket_json_join(tmp_path: Path) -> None:
    timeline = toy_plan().schedule(iterations=2)

    measured = join(parse_iket(logical_iket(tmp_path / "iket.json", timeline)), timeline)

    assert len(measured.regions) == len(timeline.regions)
    assert measured.duration_ns == timeline.logical_duration * 10


def two_warp_plan() -> PipelinePlan:
    """One region on a two-warp role."""
    plan = PipelinePlan("two-warp", iteration_name="tile")
    plan.add_role(Role("role", "Role", 0, 2))
    plan.add_region(Region("work", "role", "Work", 0, 1))
    return plan


@pytest.mark.parametrize(
    "iterations, occurrences, match",
    [
        # A persistent CTA that ran three items does not silently analyze the first two.
        (2, {0: (0, 1, 2), 1: (0, 1, 2)}, r"warp 0 recorded 'work' 3 times.*iterations=3"),
        # Warp 1 missing its middle range would shift its last one onto iteration 1, while
        # warp 0 alone satisfies every (region, iteration).
        (3, {0: (0, 1, 2), 1: (0, 2)}, r"warp 1 recorded 'work' 2 times"),
    ],
    ids=["extra_occurrence", "warp_missing_an_occurrence"],
)
def test_iket_join_rejects_mismatched_occurrences(tmp_path: Path, iterations, occurrences, match):
    timeline = two_warp_plan().schedule(iterations=iterations)
    ranges = [
        ("work", warp, 100 * i, 100 * i + 50) for warp, items in occurrences.items() for i in items
    ]
    with pytest.raises(ValueError, match=match):
        join(parse_iket(write_iket(tmp_path / "iket.json", ranges)), timeline)


def test_analysis_and_perfetto_direct_flows(tmp_path: Path) -> None:
    timeline = toy_plan().schedule(iterations=4)
    measured = logical_capture(timeline)
    analysis = analyze_pipeline(timeline, measured)

    assert analysis.logical_critical_path.makespan == timeline.logical_duration
    assert analysis.logical_critical_cycle.initiation_interval == 4
    assert analysis.logical_critical_cycle.resource_initiation_interval == 3
    ring = next(item for item in analysis.rings if item.resource == "x")
    assert ring.peak_occupancy <= ring.depth
    assert ring.final_occupancy == 0

    output = write_pipeline_perfetto(
        tmp_path / "pipeline.pftrace",
        timeline,
        measured,
        analysis=analysis,
    )
    slices = read_slices(output)  # also checks that every begin has its end
    outgoing = [flow_id for item in slices for flow_id in item.flows_out]
    incoming = [flow_id for item in slices for flow_id in item.flows_in]
    assert len(slices) == len(timeline.regions)
    assert sorted(outgoing) == sorted(incoming)
    assert len(outgoing) == len(timeline.dependencies)
    assert any(item.flows_out and item.flows_in for item in slices)
    for track in {(item.process, item.track) for item in slices}:  # no slice nests in another
        lane = [item for item in slices if (item.process, item.track) == track]
        assert all(left.end_ns <= right.start_ns for left, right in itertools.pairwise(lane))
    trace = Trace()
    trace.ParseFromString(output.read_bytes())
    assert any(
        packet.HasField("track_event") and packet.track_event.type == TrackEvent.TYPE_COUNTER
        for packet in trace.packet
    )


def test_steady_state_analysis_does_not_depend_on_the_window() -> None:
    """A reuse distance at least the window length still bounds the ring recurrence."""
    results = []
    for iterations in (1, 2, 4):
        timeline = toy_plan().schedule(iterations=iterations)
        analysis = analyze_pipeline(timeline, logical_capture(timeline))
        cycle = analysis.logical_critical_cycle
        results.append((cycle.initiation_interval, [ring.resource for ring in analysis.rings]))
    assert results == [(4, ["x"])] * 3


def fanout_plan(first: str) -> PipelinePlan:
    """``load`` fills one slot of ``x`` that ``fast`` and ``slow`` both read and release."""
    plan = PipelinePlan("fanout", iteration_name="tile")
    for warp, name in enumerate(("load", "fast", "slow")):
        plan.add_role(Role(name, name, warp, warp + 1))
        plan.add_region(Region(name, name, name, 0, 1))
    plan.add_resource(Resource("x", "X", 1))
    consumers = ("fast", "slow") if first == "fast" else ("slow", "fast")
    for consumer in consumers:
        plan.add_dependency(Dependency("load", consumer, "x"))
    for consumer in consumers:
        plan.add_dependency(Dependency(consumer, "load", "x", distance=1, kind="reuse"))
    return plan


@pytest.mark.parametrize("first", ["fast", "slow"])
def test_ring_slot_is_free_after_its_last_release(first: str) -> None:
    """The slot drains when the slower reader releases it, regardless of dependency order."""
    timeline = fanout_plan(first).schedule(iterations=2)
    regions = [
        (name, i, 600 * i + start, 600 * i + end)
        for i in range(2)
        for name, start, end in (("load", 0, 100), ("fast", 100, 300), ("slow", 100, 600))
    ]
    measured = capture(1200, regions)
    analysis = analyze_pipeline(timeline, measured)
    assert [(r.resource, r.peak_occupancy, r.final_occupancy) for r in analysis.rings] == [
        ("x", 1, 0)
    ]
    assert ring_occupancy_samples(timeline, measured) == {
        "x": ((0, 0), (100, 1), (600, 0), (700, 1), (1200, 0))
    }
    assert [d.resource for d in analysis.depth_counterfactuals] == ["x"]


def test_ring_slot_with_two_producers_fills_once_both_finish() -> None:
    """Two producer roles fill one slot of ``x`` together: it holds one item, not two."""
    plan = PipelinePlan("fanin", iteration_name="tile")
    for warp, name in enumerate(("a", "b", "use")):
        plan.add_role(Role(name, name, warp, warp + 1))
        plan.add_region(Region(name, name, name, 0, 1))
    plan.add_resource(Resource("x", "X", 1))
    for producer in ("a", "b"):
        plan.add_dependency(Dependency(producer, "use", "x"))
        plan.add_dependency(Dependency("use", producer, "x", distance=1, kind="reuse"))
    timeline = plan.schedule(iterations=2)
    regions = [
        (name, i, 300 * i + start, 300 * i + end)
        for i in range(2)
        for name, start, end in (("a", 0, 100), ("b", 0, 150), ("use", 150, 300))
    ]
    measured = capture(600, regions)
    assert ring_occupancy_samples(timeline, measured) == {
        "x": ((0, 0), (150, 1), (300, 0), (450, 1), (600, 0))
    }
    (ring,) = analyze_pipeline(timeline, measured).rings
    assert (ring.peak_occupancy, ring.final_occupancy) == (1, 0)


def test_perfetto_splits_crossing_role_envelopes_and_normalizes_suffix(
    tmp_path: Path,
) -> None:
    timeline = one_role_plan("crossing").schedule(iterations=1)
    measured = MeasuredCapture(
        (0, 0, 0),
        0,
        15,
        (
            MeasuredRegion("a", "role", 0, 0, 10, 10.0, 1),
            MeasuredRegion("b", "role", 0, 5, 15, 10.0, 1),
        ),
    )

    output = write_pipeline_perfetto(tmp_path / "crossing.json", timeline, measured)
    trace = Trace()
    trace.ParseFromString(output.read_bytes())
    descriptors = {
        packet.track_descriptor.uuid: packet.track_descriptor
        for packet in trace.packet
        if packet.HasField("track_descriptor")
    }
    begins = [
        packet.track_event
        for packet in trace.packet
        if packet.HasField("track_event")
        and packet.track_event.type == TrackEvent.TYPE_SLICE_BEGIN
    ]

    assert output.suffix == ".pftrace"
    assert len({event.track_uuid for event in begins}) == 2
    assert all(descriptors[event.track_uuid].sibling_merge_behavior for event in begins)
    assert any(descriptor.HasField("process") for descriptor in descriptors.values())


def test_perfetto_zero_duration_slice_begins_before_it_ends(tmp_path: Path) -> None:
    """An empty region at the time the next one starts on its lane opens and closes before it."""
    timeline = one_role_plan("empty").schedule(iterations=1)
    regions = (
        MeasuredRegion("a", "role", 0, 5, 5, 0.0, 1),
        MeasuredRegion("b", "role", 0, 5, 9, 4.0, 1),
    )
    measured = MeasuredCapture((0, 0, 0), 5, 4, regions)

    output = write_pipeline_perfetto(tmp_path / "empty.pftrace", timeline, measured)

    # read_slices rejects an end with no open slice on its track.
    assert [(s.name, s.start_ns, s.end_ns) for s in read_slices(output)] == [
        ("B · tile 0", 0, 4),
        ("A · tile 0", 0, 0),
    ]


INDEXED_SOURCE = """
PIPELINE = PipelineAnnotations("indexed", iteration_name="chunk")
PIPELINE.resource("x_0", label="X hi", depth=1)
PIPELINE.resource("x_1", label="X lo", depth=1)
PIPELINE.resource("acc", label="Acc", depth=1)

class Kernel:
    @PIPELINE.role("state", label="State", warp_start=0, warp_end=4)
    def run_state(self):
        PIPELINE.region(
            "state.split", label="Split", weight=2,
            produces=("x_0", "x_1"),
        )
        PIPELINE.region(
            "state.wait", label="Wait", weight=1,
            consumes=("acc",), releases=("acc",),
        )
        PIPELINE.iteration_end()

    @PIPELINE.role("mma", label="MMA", warp_start=4, warp_end=5)
    def run_mma(self):
        for split in cutlass.range_constexpr(2):
            PIPELINE.region(
                "mma.pass", label="Pass", weight=2,
                consumes=("x_{index}",), releases=("x_{index}",), produces=("acc",),
                count=2, index=split,
            )
        PIPELINE.iteration_end()
"""


def test_indexed_regions_expand_with_per_copy_tokens(tmp_path: Path) -> None:
    source = tmp_path / "indexed.py"
    source.write_text(INDEXED_SOURCE)

    plan = extract_plan(source)

    assert [region.name for region in plan.regions.values() if region.role == "mma"] == [
        "mma.pass[0]",
        "mma.pass[1]",
    ]
    edges = {(edge.source, edge.target, edge.resource, edge.kind) for edge in plan.dependencies}
    assert ("state.split", "mma.pass[0]", "x_0", "data") in edges
    assert ("state.split", "mma.pass[1]", "x_1", "data") in edges
    # Plain produces attach to the last copy only.
    assert ("mma.pass[1]", "state.wait", "acc", "data") in edges
    assert not any(
        edge.source == "mma.pass[0]" and edge.resource == "acc" for edge in plan.dependencies
    )
    assert plan.schedule(iterations=2).logical_duration > 0


@pytest.mark.parametrize(
    "source, old, new, match",
    [
        (
            ANNOTATED_SOURCE,
            'label="Load", warp_start',
            'label="Load", color="#00f", warp_start',
            r"Unknown role\(\) keyword 'color'",
        ),
        (INDEXED_SOURCE, "count=2, index=split,", "count=2,", "count>1 and index"),
    ],
    ids=["unknown_keyword", "count_without_index"],
)
def test_extractor_rejects_what_the_markers_reject(tmp_path: Path, source, old, new, match):
    path = tmp_path / "annotated.py"
    path.write_text(source.replace(old, new))
    with pytest.raises(ValueError, match=match):
        extract_plan(path)


def test_runtime_indexed_region_names_and_rejects_dynamic_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotations = PipelineAnnotations("test", iteration_name="tile")
    annotations.enable_iket()
    pushed: list[str] = []
    fake_iket = SimpleNamespace(range_push=pushed.append, range_pop=lambda: None)
    monkeypatch.setattr(pipeline_annotations, "iket", fake_iket)

    @annotations.role("mma", label="MMA", warp_start=0, warp_end=1)
    def body(index) -> None:
        annotations.region("pass", label="Pass", weight=1, count=2, index=index)
        annotations.iteration_end()

    body(1)
    assert pushed == ["pass[1]"]
    with pytest.raises(TypeError, match="compile-time int"):
        body(SimpleNamespace())
    with pytest.raises(ValueError, match="outside count"):
        body(2)


def test_report_lists_regions_and_scheduler_findings() -> None:
    timeline = toy_plan().schedule(iterations=4)
    measured = logical_capture(timeline)
    analysis = analyze_pipeline(timeline, measured)

    text = report(timeline, measured, analysis, unprofiled_iteration_ns=100.0)

    assert "per iteration ~" in text and "instrumentation +" in text
    assert "load     load.x" in text and "compute  compute.x" in text
    assert "iteration 2 timeline" in text
    assert "RecMII" in text and "critical cycle: " in text


def test_pipeline_report_is_the_function_after_importing_every_submodule() -> None:
    """Submodule imports must not shadow package exports, regardless of import order.

    Python binds a submodule's name on its package ahead of lazy exports, even if the
    export was already resolved.
    """
    for info in pkgutil.iter_modules(pipeline_package.__path__):
        importlib.import_module(f"{pipeline_package.__name__}.{info.name}")
    assert callable(pipeline_package.report), pipeline_package.report
    shadowed = [
        n for n in pipeline_package.__all__ if inspect.ismodule(getattr(pipeline_package, n))
    ]
    assert shadowed == []


def test_capture_cli_reanalyzes_an_existing_trace(tmp_path: Path, capsys) -> None:
    from attn_gym.warpdrive.profiler.pipeline.capture import main

    source = tmp_path / "annotated.py"
    source.write_text(ANNOTATED_SOURCE)
    timeline = extract_plan(source).schedule(iterations=3)
    trace = logical_iket(tmp_path / "iket.trace.json", timeline)

    rc = main(
        [
            "--tag",
            "demo.v1",
            "--annotated",
            str(source),
            "--iterations",
            "3",
            "--out",
            str(tmp_path / "traces"),
            "--from-trace",
            str(trace),
        ]
    )

    assert rc == 0
    # A dotted tag keeps its dot: demo.v1.report.txt, not demo.report.txt.
    assert (tmp_path / "traces" / "demo.v1.enriched.pftrace").exists()
    assert "RecMII" in (tmp_path / "traces" / "demo.v1.report.txt").read_text()
    assert "iteration 1 timeline" in capsys.readouterr().out
