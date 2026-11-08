"""Draw a measured pipeline capture with its plan's dependencies and scheduling analysis."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from attn_gym.profiler.perfetto import TraceBuilder, stable_id
from attn_gym.profiler.pipeline.analysis import (
    PipelineAnalysis,
    ring_occupancy_samples,
)
from attn_gym.profiler.pipeline.measure import MeasuredCapture
from attn_gym.profiler.pipeline.plan import Timeline


def write_pipeline_perfetto(
    path: str | Path,
    timeline: Timeline,
    measured: MeasuredCapture,
    *,
    analysis: PipelineAnalysis | None = None,
) -> Path:
    """Write one CTA's role tracks, dependency flows and (with ``analysis``) scheduling
    findings and ring occupancy counters; return the trace's path."""
    builder = TraceBuilder()
    process = builder.process(
        ("pipeline", timeline.plan_name, measured.cta),
        f"{timeline.plan_name} · CTA {measured.cta}",
    )
    role_tracks = {
        role.name: builder.track(
            process, role.name, f"{role.label} · warps {role.warp_start}:{role.warp_end}"
        )
        for role in timeline.roles
    }

    if analysis is not None:
        track = builder.track(process, "analysis", "Scheduling analysis")
        builder.instant(
            track, 0, "Critical path / cycle / depth summary", analysis.perfetto_summary()
        )
        depths = {resource.name: resource.depth for resource in timeline.resources}
        for resource, samples in ring_occupancy_samples(timeline, measured).items():
            track = builder.track(
                process,
                ("ring", resource),
                f"ring {resource} occupancy / depth {depths[resource]}",
                counter=True,
            )
            for timestamp_ns, occupancy in samples:
                builder.counter(track, timestamp_ns, occupancy)

    measured_regions = {(item.name, item.iteration): item for item in measured.regions}
    outgoing: dict[tuple[str, int], list[int]] = defaultdict(list)
    incoming: dict[tuple[str, int], list[int]] = defaultdict(list)
    edge_metadata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for edge_index, edge in enumerate(timeline.dependencies):
        source = edge.dependency.source, edge.source_iteration
        target = edge.dependency.target, edge.target_iteration
        if source not in measured_regions or target not in measured_regions:
            continue
        flow_id = stable_id("pipeline-flow", timeline.plan_name, edge_index, source, target)
        outgoing[source].append(flow_id)
        incoming[target].append(flow_id)
        description = (
            f"{edge.dependency.kind}:{edge.dependency.resource}:"
            f"{source[0]}[{source[1]}]->{target[0]}[{target[1]}]"
        )
        edge_metadata[source].append(f"unblocks {description}")
        edge_metadata[target].append(f"depends_on {description}")

    stage_analysis = analysis.perfetto_annotations() if analysis is not None else {}
    planned = {item.key: item for item in timeline.regions}
    roles = {role.name: role for role in timeline.roles}
    for region in measured.regions:
        key = region.name, region.iteration
        plan_region = planned[key]
        role = roles[region.role]
        args: dict[str, object] = {
            "region_id": region.name,
            "iteration": region.iteration,
            "role": region.role,
            "warp_start": role.warp_start,
            "warp_end": role.warp_end,
            "start_ns": region.start_ns - measured.origin_ns,
            "duration_ns": region.end_ns - region.start_ns,
            "median_duration_ns": region.median_duration_ns,
            "warp_samples": region.sample_count,
            "logical_start": plan_region.start,
            "logical_end": plan_region.end,
            "logical_weight": plan_region.region.weight,
        }
        if region.wait_ns is not None:
            args["measured_wait_ns"] = region.wait_ns
        if plan_region.region.source is not None:
            source = plan_region.region.source
            args["source"] = f"{source.path}:{source.line}"
        for index, description in enumerate(edge_metadata.get(key, ())):
            args[f"dependency[{index}]"] = description
        args.update(stage_analysis.get(key, {}))
        builder.slice(
            role_tracks[region.role],
            region.start_ns - measured.origin_ns,
            region.end_ns - measured.origin_ns,
            f"{plan_region.region.label} · {timeline.iteration_name} {region.iteration}",
            args=args,
            flows_out=outgoing.get(key, ()),
            flows_in=incoming.get(key, ()),
        )
    return builder.write(path)
