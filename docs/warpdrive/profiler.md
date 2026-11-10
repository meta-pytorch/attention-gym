# Profiler

[Back to warpdrive](index.md)

`attn_gym.warpdrive.profiler` records code regions **inside** a CuTeDSL kernel, decodes them
into ranges, analyzes them against a pipeline plan, and exports native
[Perfetto](https://ui.perfetto.dev/) traces. A warpdrive kernel gets its ranges from
[`enable_trace`](tracing.md#tracing-backends) without hand-placed regions; this page covers the
recorders and analysis underneath, which any CuTeDSL kernel can use directly.

> Inspired by [gau-nernst's intra-kernel profiling for AMD MI300X](https://gau-nernst.github.io/amd-a2a/#intra-kernel-profiling)

## Quick start

```python
import cutlass
from cutlass import Int32, cute

from attn_gym.warpdrive.profiler import (
    compact_anchor_init,
    compact_profile_region,
    profile_session,
    summarize_by_tag,
)

TAG_COMPUTE = 0  # the position of "compute" in tag_names


@cute.kernel
def my_kernel(output, prof_buf, max_events):
    bidx, _, _ = cute.arch.block_idx()
    compact_anchor_init(prof_buf, bidx, max_events)
    for i in cutlass.range(4):
        with compact_profile_region(prof_buf, max_events, Int32(TAG_COMPUTE), bidx, Int32(i)):
            compute_something()


with profile_session(
    max_events_per_unit=4,
    num_units=(num_blocks, "Block"),
    tag_names=["compute"],
    trace_path="trace.pftrace",
    compact=True,
    expected_events_per_unit=4,
) as session:
    # `compiled` is the compiled entry point that launches my_kernel.
    compiled(output, session.prof.tensor, session.prof.max_events_per_unit)

print(summarize_by_tag(session.ranges)["compute"])
```

A tag is its name's position in `tag_names`; `session.tags.id("compute")` returns it for a
kernel that takes the tag as an argument. `session.ranges` is set after the block exits.
[`warp_specialized_static.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/profiler/warp_specialized_static.py)
is a complete script of this shape.

## One model, two recorders

Every measurement is a frozen `Range(name, start_ns, end_ns, cta, warp, slot, payload)`:

| Recorder | Produces ranges with | Decoded by |
|---|---|---|
| Compact records (`attn_gym.warpdrive.profiler`, `%globaltimer`) | `cta=(unit,)`, `slot`; no warp or payload | `decode_ranges` / `profile_session(compact=True)` |
| Full records (warpdrive's `"globaltimer"` backend) | `cta=(unit,)`, `slot`, the recording `warp`, an optional `payload` | `decode_ranges` / `warpdrive.trace_session` |
| IKET (`cutlass.cute.experimental.iket`, `run-iket`) | `cta=(x, y, z)`, `warp`, an optional `payload` | `parse_iket(path)` |

The same ranges feed the statistics below, `write_ranges`, and the pipeline join. The
backends do not have equal overheads, clocks or sampling: IKET records every warp of a role,
the globaltimer backend one warp per role instance.

## Recording compact ranges

`compact_profile_region(buf, max_events, tag, unit_id, event_idx, target_warp=None,
enabled=True, record=True, smem=False)` records one region in static slot `event_idx`: lane 0
of `target_warp` (default warp 0) reads the start timer on entry and stores the record on
exit. `enabled` is a compile-time switch (a disabled region emits no code); `record` is a
runtime predicate selecting the units that record (sparse CTA sampling). For deeply nested
call sites, pair `read_globaltimer_lo32()` with `compact_event_stop` / `compact_event_stop_smem`
directly; they guard only on lane 0, so the caller elects the warp.

Contracts:

- **Anchor.** Each unit's slot 0 holds a 64-bit timer anchor written by warp 0
  (`compact_anchor_init` for global memory, `compact_prepare_smem` for shared memory). The
  decoder rebuilds each start as the anchor plus the signed difference of their low 32 bits, so
  an event may start up to ~2.1 s before or after the anchor; other warps need not wait for it.
  Records in a unit without an anchor raise (the anchor and the events used different `record`
  predicates, say).
- **Slots.** Out-of-range slots (`event_idx < 0` or `>= max_events_per_unit`) are dropped on the
  device rather than written over a neighboring unit or shared memory. Use
  `expected_events_per_unit` to catch them on the host.
- **Limits.** Durations wrap modulo 2^24 ns (~16.7 ms); tags are 8 bits.

### Global or shared memory

| Records | Per-event device cost | Storage |
|---|---|---|
| Compact gmem (`compact_event_stop`, `smem=False`) | 2x `CS2R.32` + 1x `STG.E.EF.64` + address math + slot bounds check | 8 B per event, global |
| Compact smem (`compact_event_stop_smem`, `smem=True`) | 2x `CS2R.32` + 1x `STS.64` + bit-pack + slot bounds check | 8 B per event in shared memory, flushed at exit |

The smem path needs a CTA-local `Int64` tensor of `1 + max_events_per_unit` slots
(`SmemAllocator.allocate_array` or a `cute.struct.MemRange`): `compact_prepare_smem(smem, max,
record)` clears and anchors it, `compact_flush_smem_to_gmem(smem, gmem, unit, max, record)`
copies it out at kernel exit. The smem path pays its global stores in that final flush, so each
event perturbs the region it measures less. Pick smem when recorded durations drive
optimization decisions, gmem when plumbing shared memory is not worth it. To measure the cost
on your kernel, compile it with `enabled=False` and `True`, as
`examples/profiler/measure_profiler_overhead.py` does.

### Full records

`event_stop(buf, unit_id, event_idx, start_ns, tag, warp, max_events, payload=None)` (with
`read_globaltimer()` for the start) stores a 32-byte record: 64-bit start and duration, the tag
word, and the warp. It is unguarded: the caller elects the recording thread and keeps the slot
in range. The globaltimer backend of warpdrive records these, since its ranges need a warp, a
payload (a work item's index) and durations past 2^24 ns.

## Statistics and traces

```python
from attn_gym.warpdrive.profiler import overlap_between, summarize_by_tag, write_ranges

summaries = summarize_by_tag(session.ranges)  # count, total, mean, min, p50, p95, max per name
overlap = overlap_between(session.ranges, "producer_load", "consumer_compute_store")
```

`overlap_between` merges intervals and, by default, measures overlap within each CTA so
unrelated CTAs cannot inflate it.

`profile_session(trace_path=...)` writes one process per unit and one track per warp (per range
name for compact records). For another layout, call `write_ranges` on the ranges; each option
is a function of a range:

```python
write_ranges(
    "trace.pftrace",
    session.ranges,
    process=lambda r: f"CTA {r.cta[0]}",
    thread=lambda r: "producer" if r.name.startswith("producer") else "consumer",
    name=lambda r: f"{r.name} {r.slot}",
    args=lambda r: {"k_tile": r.slot // 4},
    flows=[(producer_range, consumer_range)],  # arrows between two ranges
)
```

Ranges that cross on one track move to a sibling "overlap" track that Perfetto merges under the
track's name; nested ranges stay nested. `perfetto.TraceBuilder` is the builder underneath, also
used by the pipeline trace; the `perfetto` package is imported only when a trace is written.

## Worked example

[`examples/profiler/cute_profiler_buffering.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/profiler/cute_profiler_buffering.py)
(`python examples/profiler/cute_profiler_buffering.py --trace-dir DIR`) compares two otherwise
identical warp-specialized kernels: one shared-memory stage forces
`load i -> wait i -> compute/store i -> release i -> load i+1`; two stages let the producer load
tile `i + 1` while the consumer works on tile `i`. The example records compact shared-memory
ranges for one sampled CTA, joins them to the kernel's pipeline plan (stage reuse is a
`num_stages`-tile dependency), and writes one trace per variant with the plan's dependency
arrows and the stage ring's occupancy. It checks correctness and times the unprofiled
specializations separately.

What to look for in the two traces: with two stages the producer's load overlaps consumer
work, and the producer's wait for a released stage (inferred from the consumer's release, as
compact records trace no wait ranges) shrinks toward zero.

## Pipeline analysis

`attn_gym.warpdrive.profiler.pipeline` models a warp-specialized kernel as a
`PipelinePlan`: roles (warp ranges), resources, ordered regions per role, and data, state and
reuse dependencies with iteration distances. `plan.schedule(iterations)` unrolls it into a
`Timeline`; `join(ranges, timeline, cta=None)` assigns one CTA's measured ranges to its
`(region, iteration)`s; `analyze_pipeline` and `write_pipeline_perfetto` analyze and draw the
result.

```python
from attn_gym.warpdrive.profiler import parse_iket
from attn_gym.warpdrive.profiler.pipeline import (
    analyze_pipeline,
    extract_plan,
    join,
    report,
    write_pipeline_perfetto,
)

plan = extract_plan("annotated_kernel.py")  # or plan_from_warpdrive(op), or PipelinePlan(...)
timeline = plan.schedule(iterations=4)
measured = join(parse_iket("iket.trace.json"), timeline, cta=(0, 0, 0))
analysis = analyze_pipeline(timeline, measured)
write_pipeline_perfetto("pipeline.pftrace", timeline, measured, analysis=analysis)
print(report(timeline, measured, analysis))
```

The join is strict: a warp's n-th range of a region is iteration n, so every recording warp
must record each region of its role exactly `iterations` times. A persistent CTA that ran more
items than the timeline raises with the count to schedule; a warp missing a range raises rather
than shifting its later ranges onto the wrong iterations. Ranges without a warp (compact
records) count as one recorder per role. `cta` defaults to the lowest CTA that recorded a plan
region.

**Waits.** A range named `<x>.<op>` with `op` one of `acquire`, `wait`, `peek`, `tail`, `turn`,
`sync` (warpdrive's wait ranges, `enable_trace(waits=True)`), nested in a region's range on the
same warp, is a measured wait: the region's `wait_ns` is the largest per-warp sum of them, and
the blocker analysis uses it (`wait_source="measured"`). Without wait ranges, a region's wait
is inferred from the latest predecessor's end (`wait_source="inferred"`).

The analysis reports finite critical paths and slack, latest-arriving blockers, wait/busy
decomposition, ring occupancy and run-ahead (a ring slot is free once its last reader releases
it), depth+1 counterfactuals, and the steady-state maximum cycle ratio (RecMII versus per-role
ResMII, in the plan's logical weight units). Recurrences and rings come from the plan's symbolic
dependencies, so they do not depend on the scheduled window. The depth+1 counterfactual replaces
a ring's reuse distance with `depth + 1`, which matches one use per iteration only.

`report(timeline, measured, analysis, range_overhead_ns=None)` renders per-role region medians
over the interior iterations, one iteration's timeline in ns, and the scheduler findings; pass
`summary.IKET_RANGE_OVERHEAD_NS` (~50 ns per IKET push/pop on GB200) to flag regions dominated
by instrumentation.

### Annotated kernels

`PipelineAnnotations` markers in a CuTeDSL kernel emit IKET ranges when `enable_iket()` was
called and nothing otherwise; `extract_plan(path)` reads the same literal declarations into a
plan. A kernel imports only `pipeline/annotations.py`, so the extractor and the plan model stay
out of its compile key.

```python
PIPELINE = PipelineAnnotations("my_kernel", iteration_name="tile")
PIPELINE.resource("operand", label="Operand ring", depth=2)

# In the CuTeDSL implementation:
# @PIPELINE.role("load", label="Load", warp_start=0, warp_end=1)
# PIPELINE.region("load.operand", label="Load", weight=1, produces=("operand",))
# PIPELINE.region(..., consumes=("operand",), releases=("operand",))
# PIPELINE.iteration_end()
```

For a region inside a `cutlass.range_constexpr` loop, pass the compile-time loop variable and
the trip count so each copy gets its own range:

```python
for split in cutlass.range_constexpr(2):
    PIPELINE.region(
        "mma.pass", label="kg@X pass", weight=2,
        consumes=("x_{index}",), releases=("x_{index}",), produces=("acc",),
        count=2, index=split,
    )
```

This expands to `mma.pass[0]` and `mma.pass[1]`. Tokens containing `{index}` are substituted
per copy; plain `consumes` attach to the first copy and plain `produces`/`releases` to the
last, so the group acts as one region externally.

### Capture CLI

```
python -m attn_gym.warpdrive.profiler.pipeline.capture \
    --tag 02_split_stages --annotated rev_annotated.py --iterations 8 \
    --unprofiled-iteration-ns 950 -- python profile_kernel.py --iket
```

It runs the command under `run-iket` with `CUTE_DSL_NO_CACHE=1` (the command must launch the
annotated kernel exactly once), then writes `traces/<tag>.raw_iket.{trace.json,pftrace}`,
`traces/<tag>.enriched.pftrace` and `traces/<tag>.report.txt`, keeping the tag whole.
`--from-trace <trace.json>` re-analyzes an existing capture without a GPU.

## warpdrive kernels

warpdrive kernels need no hand-placed regions: `kernel.enable_trace("globaltimer")` or
`enable_trace("iket")` makes the layer emit ranges named after its roles, channel waits and
`trace_region`s, and `plan_from_warpdrive(op)` exports the matching plan, so either backend's
ranges join it with their waits measured. See
[Tracing backends](tracing.md#tracing-backends).

## Buffer layouts

Each unit owns a slice of `1 + R * max_events_per_unit` int64 words:

```
compact (R = 1):  slice[0]     = 64-bit globaltimer anchor
                  slice[1 + i] = tag(8) | dur_ns(24) | ts_lo32(32)
full    (R = 4):  slice[0]     = unused
                  slice[1 + 4*i + 0] = start_ns
                  slice[1 + 4*i + 1] = dur_ns
                  slice[1 + 4*i + 2] = tag | PAYLOAD_FLAG | payload << 32  (payload optional)
                  slice[1 + 4*i + 3] = warp
```

Importing `attn_gym.warpdrive.profiler` loads the device recorders
(`attn_gym/warpdrive/profiler/ops.py`) eagerly, and they import CuTeDSL, so host-only analysis
still needs `cutlass` installed. Host code (decoding, statistics, trace writers) loads on first
use.

## Examples

Scripts under `examples/profiler/`, each runnable as `python examples/profiler/<name>.py`:

- [`cute_profiler_buffering.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/profiler/cute_profiler_buffering.py):
  the worked example above, compact smem ranges joined to a pipeline plan.
- [`measure_profiler_overhead.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/profiler/measure_profiler_overhead.py):
  the same kernel compiled with `enabled=False` and `True`.
- [`warp_specialized_static.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/profiler/warp_specialized_static.py):
  one region per warp role, compact gmem records in static slots.

The profiler's users elsewhere in this repository:

- [`attn_gym/warpdrive/tracing.py`](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/tracing.py) and
  [`trace_session.py`](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/trace_session.py): warpdrive's globaltimer
  backend records full records with `event_stop` and decodes them through `profile_session`;
  [`pipeline_plan.py`](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/pipeline_plan.py) exports a kernel's plan.
- [`attn_gym/warpdrive/examples/qwen_megakernel_trace.py`](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/qwen_megakernel_trace.py):
  decodes a megakernel step's ranges per instruction-table row and writes a custom trace layout
  with cross-CTA flows.
- [`attn_gym/warpdrive/profiler/pipeline/capture.py`](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/profiler/pipeline/capture.py): the
  capture CLI above, from `run-iket` to the enriched trace and report.
- [`test/profiler/`](https://github.com/meta-pytorch/attention-gym/blob/main/test/profiler): compact gmem/smem recording, statistics and the
  pipeline analysis on small kernels.
