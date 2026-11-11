# Profiler reference

[Back to the profiler guide](profiler.md)

Use these APIs when you need manual instrumentation, custom trace views or pipeline analysis
beyond the built-in warpdrive capture. For a first trace, start with the guide above.

## Manual recording

For standalone CuTeDSL kernels, mark regions with `compact_profile_region`. In this sketch,
each block is a recording **unit**, each loop iteration gets an **event slot**, and tag `0`
names the region `compute`. Replace `compute_something()` with your work; the launch grid is
one-dimensional.

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

<a id="one-model-two-recorders"></a>

## Recording formats

Every decoded measurement is an immutable
`Range(name, start_ns, end_ns, cta, warp, slot, payload)`:

| Recorder | Produces ranges with | Decoded by |
|---|---|---|
| Compact records (`attn_gym.warpdrive.profiler`, `%globaltimer`) | `cta=(unit,)`, `slot`; no warp or payload | `decode_ranges` / `profile_session(compact=True)` |
| Full records (warpdrive's `"globaltimer"` backend) | `cta=(unit,)`, `slot`, the recording `warp`, an optional `payload` | `decode_ranges` / `warpdrive.trace_session` |
| IKET (`cutlass.cute.experimental.iket`, `run-iket`) | `cta=(x, y, z)`, `warp`, an optional `payload` | `parse_iket(path)` |

These ranges work with the [statistics functions](profiler.md#statistics-and-traces),
`write_ranges`, and pipeline analysis. The backends differ in clock, overhead and sampling:
warpdrive records every warp with IKET, but one warp per role instance with globaltimer.

## Recording compact ranges

`compact_profile_region` times its body using lane 0 of `target_warp` (default warp 0). It
does not add a barrier or wait for asynchronous work to finish. Give each event its own
`event_idx`; reusing a slot overwrites the earlier record.

- `enabled=False`: omit profiling code at compile time; the body still runs.
- `record`: choose which units record at runtime.
- `smem=True`: store in shared memory rather than global memory.

For deeply nested code, pair `read_globaltimer_lo32()` with `compact_event_stop` or
`compact_event_stop_smem`. These stop helpers select lane 0 only; you must select the warp.

### Buffer limits

- **Anchor.** Each unit's slot 0 holds a 64-bit timer anchor written by warp 0
  (`compact_anchor_init` at kernel entry outside warp guards, or `compact_prepare_smem`). The
  decoder rebuilds each start as the anchor plus the signed difference of their low 32 bits, so
  an event may start up to ~2.1 s before or after the anchor; other warps need not wait for it.
  Records in a unit without an anchor raise (the anchor and the events used different `record`
  predicates, say).
- **Slots.** Out-of-range slots (`event_idx < 0` or `>= max_events_per_unit`) are dropped on the
  device rather than written over a neighboring unit or shared memory. Use
  `expected_events_per_unit` to catch missing records. It checks every allocated unit.
- **Sampling.** Map sampled CTAs to dense `unit_id`s in `[0, num_units)`; unit IDs are not
  bounds-checked. Use the same sampling guard for anchors, events and flushes. If you allocate
  only one unit for CTA 7, that CTA must record as unit 0, not unit 7.
- **Limits.** Durations wrap modulo 2^24 ns (~16.7 ms); tags are 8 bits.

### Global or shared memory

| Records | Per-event device cost | Storage |
|---|---|---|
| Compact gmem (`compact_event_stop`, `smem=False`) | 2x `CS2R.32` + 1x `STG.E.EF.64` + address math + slot bounds check | 8 B per event, global |
| Compact smem (`compact_event_stop_smem`, `smem=True`) | 2x `CS2R.32` + 1x `STS.64` + bit-pack + slot bounds check | 8 B per event in shared memory, flushed at exit |

The smem path needs a CTA-local `Int64` tensor of `1 + max_events_per_unit` slots
(`SmemAllocator.allocate_array` or a `cute.struct.MemRange`): `compact_prepare_smem(smem, max,
record)` clears and anchors it, `compact_flush_smem_to_gmem(smem, gmem, unit, max, record)`
copies it out at kernel exit. Both are CTA-wide operations with barriers; all threads must
participate and agree on `record`. Global stores happen in the final flush, outside the timed
regions. Start with gmem for simpler setup; compare `enabled=False` and `True` before drawing
performance conclusions (see `examples/profiler/measure_profiler_overhead.py`).

### Full records

`event_stop(buf, unit_id, event_idx, start_ns, tag, warp, max_events, payload=None)` (with
`read_globaltimer()` for the start) stores a 32-byte record: 64-bit start and duration, the tag
word, and the warp. It is unguarded: the caller elects the recording thread and keeps the slot
in range. The globaltimer backend of warpdrive records these, since its ranges need a warp, a
payload (a work item's index) and durations past 2^24 ns.


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

## Customize the trace view

`profile_session(trace_path=...)` writes one process per unit and one track per warp, or per
region name for compact records. `write_ranges` controls grouping and labels. Its callbacks
receive a `Range`; `flows` adds arrows between selected ranges:

```python
from attn_gym.warpdrive.profiler import write_ranges

write_ranges(
    "custom.pftrace",
    ranges,
    process=lambda r: f"CTA {r.cta}",
    thread=lambda r: f"warp {r.warp}",
    name=lambda r: r.name,
    args=lambda r: {"work_item": r.payload},
    flows=[(producer_range, consumer_range)],
)
```

Here, `producer_range` and `consumer_range` are ranges selected from the capture. Nested ranges
stay nested; crossing ranges move to sibling overlap tracks. The writer and pipeline view use
`attn_gym.warpdrive.profiler.perfetto.TraceBuilder`, which imports `perfetto` only when writing.

## Pipeline matching and analysis

- Each recording warp must record every region of its role exactly `iterations` times.
  The n-th occurrence is iteration n. Missing or extra ranges raise instead of shifting later
  records; the error reports the count to schedule. Compact ranges count as one recorder per
  role. With no `cta`, the join selects the lowest CTA that recorded a plan region.
- Wait ranges end in `.acquire`, `.wait`, `.peek`, `.tail`, `.turn` or `.sync` and nest inside
  a region on the same warp. With these, `wait_ns` is the largest per-warp sum of nested waits
  (`wait_source="measured"`). Otherwise, waiting is estimated from the latest predecessor's
  end (`wait_source="inferred"`).
- Analysis includes critical paths and slack, latest-arriving blockers, wait/busy time, buffer
  occupancy and producer run-ahead. A ring slot becomes free after its last reader releases it.
- RecMII (maximum cycle ratio) and per-role ResMII describe steady-state limits in the plan's
  logical weight units, not nanoseconds. Recurrences and rings come from symbolic dependencies,
  independent of the displayed iteration count.
- The depth+1 counterfactual sets reuse distance to `depth + 1`. It represents an extra stage
  only for one use per iteration, and is a prediction to test, not a measured speedup.
- `report` shows interior-iteration region medians and one iteration's timeline in ns. To
  flag short regions, pass `range_overhead_ns=`, the recorder's cost per range measured on your
  hardware and toolchain.
