"""Host-only tests of the profiler's range model: decoding, statistics, joins and traces."""

from pathlib import Path

import pytest

pytest.importorskip("cutlass")  # attn_gym.profiler imports CuTeDSL

import torch

from attn_gym._backends.cute._key import module_closure
from attn_gym.profiler import (
    Range,
    TagTable,
    allocate_profile_buffer,
    decode_ranges,
    overlap_between,
    summarize_by_tag,
    write_ranges,
)
from attn_gym.profiler.host import check_counts
from attn_gym.profiler.ops import PAYLOAD_FLAG
from attn_gym.profiler.pipeline import (
    Dependency,
    PipelinePlan,
    Region,
    Resource,
    Role,
    analyze_pipeline,
    join,
)
from attn_gym.testing.perfetto_trace import read_slices


def span(name: str, start_ns: int, dur_ns: int, cta: int = 0, **fields) -> Range:
    return Range(name, start_ns, start_ns + dur_ns, (cta,), **fields)


def test_summarize_by_tag_reports_duration_statistics():
    ranges = [
        span("load", 0, 10),
        span("load", 20, 20),
        span("load", 50, 30),
        span("store", 90, 5),
    ]

    summaries = summarize_by_tag(ranges)

    assert summaries["load"].count == 3
    assert summaries["load"].total_ns == 60
    assert summaries["load"].mean_ns == 20
    assert summaries["load"].min_ns == 10
    assert summaries["load"].p50_ns == 20
    assert summaries["load"].p95_ns == 29
    assert summaries["load"].max_ns == 30
    assert summaries["store"].p50_ns == 5


def test_overlap_between_merges_intervals_and_isolates_ctas():
    ranges = [
        span("load", 0, 10),
        span("load", 20, 10),
        span("compute", 5, 10),
        span("compute", 25, 10),
        span("compute", 0, 100, cta=1),
    ]

    overlap = overlap_between(ranges, "load", "compute")

    assert overlap.overlap_ns == 10
    assert overlap.left_total_ns == 20
    assert overlap.right_total_ns == 120
    assert overlap.left_fraction == 0.5
    assert overlap.right_fraction == 10 / 120

    global_overlap = overlap_between(ranges, "load", "compute", same_cta=False)
    assert global_overlap.overlap_ns == 20
    assert global_overlap.left_fraction == 1.0


def test_check_counts_reports_too_few_and_too_many():
    ranges = [span("load", 0, 10), span("load", 20, 10), span("load", 0, 10, cta=1)]

    with pytest.raises(RuntimeError, match="Expected 2 events.*unit 1: 1"):
        check_counts(ranges, num_units=2, expected_per_unit=2)
    with pytest.raises(RuntimeError, match="Expected 1 events.*unit 0: 2"):
        check_counts(ranges, num_units=2, expected_per_unit=1)


def test_decode_full_records_with_payloads():
    """A tag word with PAYLOAD_FLAG carries a signed payload in its high half; one without
    decodes with payload None."""
    prof = allocate_profile_buffer(2, 2, device="cpu")
    buf = prof.tensor.view(2, prof.slice_size)
    buf[0, 1:5] = torch.tensor([100, 5, 1, 3])  # plain record: tag 1, warp 3
    word = 2 | PAYLOAD_FLAG | (-7 << 32)
    buf[1, 5:9] = torch.tensor([200, 6, word, 4])  # slot 1 of unit 1: tag 2, payload -7
    assert decode_ranges(prof, TagTable(["a", "plain", "item"])) == [
        Range("plain", 100, 105, (0,), warp=3, slot=0),
        Range("item", 200, 206, (1,), warp=4, slot=1, payload=-7),
    ]


def compact_record(ts_lo: int, dur_ns: int, tag: int) -> int:
    """Pack one compact record: ``tag(8) | dur_ns(24) | ts_lo32(32)``, as a signed int64."""
    word = ts_lo | dur_ns << 32 | tag << 56
    return word - (1 << 64) if word >= 1 << 63 else word


def compact_buffer(units: list[list[int]], max_events: int):
    """A CPU compact profile buffer: each unit's anchor followed by its records."""
    prof = allocate_profile_buffer(max_events, len(units), device="cpu", compact=True)
    grid = prof.tensor.view(len(units), prof.slice_size)
    for unit, words in enumerate(units):
        grid[unit, : len(words)] = torch.tensor(words)
    return prof


def test_compact_decode_reconstructs_starts_around_the_anchor():
    """Another warp can read its start before warp 0 writes the anchor: a start just before the
    anchor decodes there, and one past a 2^32 ns boundary after it decodes after it."""
    anchor = (8 << 32) + 1_000
    wrapped_anchor = (8 << 32) - 50  # unit 1's events start past the 2^32 ns boundary
    prof = compact_buffer(
        [
            [anchor, compact_record(900, 20, 0), compact_record(1_500, 5, 1)],
            [wrapped_anchor, compact_record(30, 10, 1)],
        ],
        max_events=2,
    )

    assert decode_ranges(prof, TagTable(["early", "late"])) == [
        Range("early", anchor - 100, anchor - 80, (0,), slot=0),
        Range("late", anchor + 500, anchor + 505, (0,), slot=1),
        Range("late", wrapped_anchor + 80, wrapped_anchor + 90, (1,), slot=0),
    ]


def test_compact_decode_rejects_records_without_an_anchor():
    prof = compact_buffer([[1 << 40, compact_record(1, 2, 0)], [0, compact_record(1, 2, 0)]], 1)
    with pytest.raises(RuntimeError, match=r"units \[1\] hold 1 record\(s\) but no anchor"):
        decode_ranges(prof, TagTable(["t"]))


def test_write_ranges_lays_out_processes_tracks_and_flows(tmp_path):
    """One process per CTA in CTA order and one track per warp; crossing ranges on one warp
    split into an overlap lane, nested ones stay on it, and a flow joins its two slices."""
    producer = span("row", 0, 10, cta=0, warp=1, payload=7)
    consumer = span("row", 20, 10, cta=1, warp=1, payload=8)
    ranges = [
        consumer,
        producer,
        span("inner", 2, 3, cta=0, warp=1),  # nested in the producer
        span("crossing", 5, 10, cta=0, warp=1),  # crosses the producer's end
        span("empty", 10, 0, cta=0, warp=1),  # empty, where the producer ends
        span("other", 1, 1, cta=0, warp=0),
    ]

    path = write_ranges(
        tmp_path / "ranges.json",
        ranges,
        name=lambda r: f"{r.name} {r.payload}" if r.payload is not None else r.name,
        flows=[(producer, consumer)],
    )

    assert path.suffix == ".pftrace"
    slices = {item.name: item for item in read_slices(path)}
    assert {name: (s.process, s.track) for name, s in slices.items()} == {
        "row 7": ("CTA 0", "warp 1"),
        "row 8": ("CTA 1", "warp 1"),
        "inner": ("CTA 0", "warp 1"),
        "crossing": ("CTA 0", "warp 1 overlap"),
        "empty": ("CTA 0", "warp 1"),
        "other": ("CTA 0", "warp 0"),
    }
    assert (slices["row 7"].start_ns, slices["row 8"].end_ns) == (0, 30)
    assert slices["row 7"].args["payload"] == 7
    assert slices["row 7"].flows_out == slices["row 8"].flows_in != ()


def warp_plan(iteration_name: str = "item") -> PipelinePlan:
    """A producer warp and a consumer warp handing items over a two-slot ring."""
    plan = PipelinePlan("ring", iteration_name=iteration_name)
    plan.add_role(Role("prod", "Producer", 2, 3))
    plan.add_role(Role("cons", "Consumer", 0, 2))
    plan.add_resource(Resource("ring", "ring", 2))
    plan.add_region(Region("prod", "prod", "prod", 0, 1))
    plan.add_region(Region("cons", "cons", "cons", 0, 1))
    plan.add_dependency(Dependency("prod", "cons", "ring"))
    plan.add_dependency(Dependency("cons", "prod", "ring", distance=2, kind="reuse"))
    return plan


def test_join_uses_the_waits_traced_inside_a_region():
    """Globaltimer-style ranges (one recording warp per role): the consumer's nested
    ``ring.wait`` is its measured wait, other nested ranges are not waits, and a role without
    traced waits keeps an inferred one."""
    ranges = [
        span("prod", 0, 100, warp=2),
        span("prod", 100, 100, warp=2),
        span("cons", 50, 150, warp=0),
        span("ring.wait", 50, 60, warp=0),
        span("store", 120, 10, warp=0),  # a user region, not a wait
        span("cons", 200, 150, warp=0),
        span("ring.wait", 200, 10, warp=0),
        span("ring.wait", 220, 5, warp=0),
        span("cons", 0, 10, cta=1, warp=0),  # another CTA
    ]
    timeline = warp_plan().schedule(iterations=2)

    measured = join(ranges, timeline)

    assert measured.cta == (0,)
    waits = {(r.name, r.iteration): r.wait_ns for r in measured.regions}
    assert waits == {("prod", 0): None, ("prod", 1): None, ("cons", 0): 60, ("cons", 1): 15}
    blockers = {(b.name, b.iteration): b for b in analyze_pipeline(timeline, measured).blockers}
    consumer = blockers[("cons", 0)]
    assert (consumer.wait_source, consumer.wait_inside_ns, consumer.busy_ns) == (
        "measured",
        60,
        90,
    )
    assert blockers[("prod", 1)].wait_source == "inferred"


def test_join_treats_compact_ranges_as_one_recorder_per_role():
    ranges = [
        span(name, 100 * i + offset, 40)
        for i in range(3)
        for name, offset in [("prod", 0), ("cons", 50)]
    ]
    measured = join(ranges, warp_plan().schedule(iterations=3))
    assert [(r.name, r.iteration, r.sample_count) for r in measured.regions][:2] == [
        ("prod", 0, 1),
        ("cons", 0, 1),
    ]
    with pytest.raises(ValueError, match="role 'cons' recorded 'cons' 3 times"):
        join(ranges, warp_plan().schedule(iterations=2))


PROFILER = Path(__file__).resolve().parents[2] / "attn_gym" / "profiler"


@pytest.mark.parametrize(
    "kernel_module, recorder",
    [
        ("warpdrive/tracing.py", "ops.py"),
        ("profiler/pipeline/annotations.py", "pipeline/annotations.py"),
    ],
)
def test_recording_kernels_key_only_the_profilers_device_code(kernel_module, recorder):
    """A kernel's compile key hashes the modules it imports: a kernel that records ranges must
    reach only the profiler's device code, not its decoding, analysis, or plan model."""
    closure = module_closure(PROFILER.parent / kernel_module)
    assert {
        path.relative_to(PROFILER).as_posix() for path in closure if path.is_relative_to(PROFILER)
    } == {recorder}


@pytest.mark.parametrize(
    "statement, keyed",
    [
        (
            "from attn_gym.profiler import compact_profile_region",
            {"__init__.py", "ops.py"},
        ),
        (
            "from attn_gym.profiler.pipeline import PipelineAnnotations",
            {"pipeline/__init__.py", "pipeline/annotations.py"},
        ),
    ],
)
def test_package_imports_of_device_code_key_no_host_code(tmp_path, statement, keyed):
    """Importing a recorder through its package keys the package's ``__init__`` and the device
    module, not the host modules the package exports lazily."""
    kernel = tmp_path / "kernel.py"
    kernel.write_text(statement + "\n")
    closure = module_closure(kernel)
    assert {
        path.relative_to(PROFILER).as_posix() for path in closure if path.is_relative_to(PROFILER)
    } == keyed
