# Profiler

[Back to warpdrive](index.md)

Use the profiler to see where time goes **inside** a CuTeDSL kernel. For warpdrive kernels,
start with IKET: enable tracing, launch once, and open the result in
[Perfetto](https://ui.perfetto.dev/). Warpdrive adds the role ranges for you.

> Inspired by [gau-nernst's intra-kernel profiling for AMD MI300X](https://gau-nernst.github.io/amd-a2a/#intra-kernel-profiling)

## Quick start

This example traces the built-in GEMM on a Blackwell GPU. Save it as `profile_gemm.py`:

```python
import torch

from attn_gym.warpdrive.examples.gemm import Gemm, compile_gemm

op = Gemm().enable_trace("iket", waits=False)
compiled = compile_gemm(op)

a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
b = torch.randn_like(a)
out = torch.empty_like(a)
compiled(a, b, out)
```

Run it with NVIDIA's `run-iket` tool. It captures the instrumentation and writes a Perfetto
trace by default; running the Python script alone does not write a trace.

```bash
CUTE_DSL_NO_CACHE=1 run-iket -o traces/gemm --clobber profile -- python profile_gemm.py
```

Open the generated `.pftrace` file in Perfetto. Each role has named ranges showing when it
ran. Start with `waits=False` to limit recording overhead; use `waits=True` when you need to
see channel and synchronization waits. Enable tracing **before compiling**.

Keep the script to one launch of the traced kernel; `parse_iket` rejects multi-launch captures.
The command bypasses the disk compile cache. Use a fresh process so an in-memory build cannot
skip tracing, and a different output directory if you want to keep earlier captures.

## Statistics and traces

Add `--postprocess all` after `profile` in the capture command to also generate JSON.
Then read the `.trace.json` file in Python:

```python
from attn_gym.warpdrive.profiler import parse_iket, summarize_by_tag
from attn_gym.warpdrive.trace_session import role_summary

ranges = parse_iket("path/to/kernel.trace.json")
print(role_summary(ranges))
summaries = summarize_by_tag(ranges)
```

Replace the path with the JSON file under `traces/gemm`. `role_summary` shows item counts,
p50/p90 durations and recorded wait shares per role. Without wait tracing, a zero wait share
means waits were not recorded, not that the kernel never waited. `summarize_by_tag` reports
count, total, mean, min, p50, p95 and max for each range name.

For producer/consumer captures, `overlap_between(ranges, producer_name, consumer_name)`
measures overlap within each CTA by default. It merges intervals so unrelated CTAs do not
inflate the result.

Use unprofiled runs to compare kernel speed. Instrumentation changes the timing it measures.

## warpdrive kernels

IKET needs no extra kernel arguments. The alternative `globaltimer` backend requires a kernel
argument passed to `run_roles(..., trace_buffer=...)`; capture it with `trace_session`.
It records one warp per role instance, rather than every warp as IKET does. See
[Tracing backends](tracing.md#tracing-backends) for setup and custom `trace_region`s.

`plan_from_warpdrive(op)` exports the matching pipeline plan for either backend. Waits are
measured when captured with `waits=True`.

## Pipeline analysis

A trace shows when work ran. A `PipelinePlan` describes what it depends on: role order, data
and state handoffs, and buffer reuse across iterations. Matching the two helps explain waits
and whether another buffer stage might help.

For an already-compiled `op` and its decoded `ranges`:

```python
from attn_gym.warpdrive import plan_from_warpdrive
from attn_gym.warpdrive.profiler.pipeline import (
    analyze_pipeline,
    join,
    report,
    write_pipeline_perfetto,
)

plan = plan_from_warpdrive(op)
timeline = plan.schedule(iterations=1)
measured = join(ranges, timeline, cta=(0, 0, 0))
analysis = analyze_pipeline(timeline, measured)
write_pipeline_perfetto("pipeline.pftrace", timeline, measured, analysis=analysis)
print(report(timeline, measured, analysis))
```

The GEMM above has one work item per CTA. For a persistent kernel, set `iterations` to the
number of items that the selected CTA processed. `join` matches recordings by region and item.
The report shows region durations and scheduling findings; the Perfetto export adds dependency
arrows and buffer occupancy.

For exact matching rules, measured versus inferred waits, and scheduling limits, see the
[pipeline reference](profiler-reference.md#pipeline-matching-and-analysis).

## Worked example

[`cute_profiler_buffering.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/profiler/cute_profiler_buffering.py)
compares one versus two shared-memory stages. Run it with `--trace-dir DIR`. One stage forces
`load i -> wait i -> compute/store i -> release i -> load i+1`; two allow loading tile `i + 1`
while the consumer works on tile `i`.

The script checks correctness, times unprofiled builds, and records compact shared-memory
ranges for one CTA. Its traces show dependency arrows and ring occupancy. Look for load/compute
overlap and less producer waiting. The producer wait is inferred from consumer release, not
directly recorded; stage reuse is a dependency `num_stages` tiles apart.

<a id="one-model-two-recorders"></a>
<a id="recording-compact-ranges"></a>
<a id="global-or-shared-memory"></a>
<a id="full-records"></a>
<a id="buffer-layouts"></a>
<a id="annotated-kernels"></a>
<a id="capture-cli"></a>

## Manual instrumentation

For standalone CuTeDSL kernels, use `compact_profile_region` to time selected code regions.
The [reference](profiler-reference.md#manual-recording) covers buffer setup, compact/full
formats, custom Perfetto views, and IKET annotations with the capture CLI.

## Examples

Run these as `python examples/profiler/<name>.py`:

- `warp_specialized_static.py`: manual compact recording, one region per warp role.
- `cute_profiler_buffering.py`: one versus two stages, with pipeline analysis.
- `measure_profiler_overhead.py`: the same kernel with profiling disabled and enabled.

Implementation references: `attn_gym/warpdrive/{tracing,trace_session,pipeline_plan}.py` for
warpdrive integration; `profiler/pipeline/capture.py` for the capture CLI; and
`test/profiler/` for recorder and analysis tests. The
[`Qwen trace example`](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/qwen_megakernel_trace.py)
shows instruction-row grouping and cross-CTA flows.
