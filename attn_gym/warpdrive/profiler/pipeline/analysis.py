"""Critical-path, blocker, ring, and depth analysis for unrolled pipeline schedules."""

from __future__ import annotations

import itertools
import statistics
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

from attn_gym.warpdrive.profiler.pipeline.measure import MeasuredCapture
from attn_gym.warpdrive.profiler.pipeline.plan import Dependency, Timeline

NodeKey = tuple[str, int]


@dataclass(frozen=True)
class ConcreteEdge:
    """One unrolled edge; ``kind`` is ``"role"`` for program order, else the dependency's kind."""

    source: NodeKey
    target: NodeKey
    kind: str
    resource: str


@dataclass(frozen=True)
class CriticalPath:
    """Critical-path method result over the unrolled, acyclic graph."""

    makespan: float
    earliest_start: dict[NodeKey, float]
    latest_start: dict[NodeKey, float]
    slack: dict[NodeKey, float]
    path: tuple[NodeKey, ...]


@dataclass(frozen=True)
class StageBlocker:
    """One measured region split into wait and busy time, with its latest-arriving predecessor.

    ``wait_source`` is ``"measured"`` when ``wait_inside_ns`` sums the region's traced wait
    ranges, ``"inferred"`` when it is the overlap with the latest predecessor's end.
    ``blocker_robustness_ns`` is how much earlier the second-latest predecessor arrived.
    """

    name: str
    iteration: int
    start_ns: int
    end_ns: int
    duration_ns: int
    ready_ns: int
    wait_inside_ns: int
    ready_delay_ns: int
    busy_ns: int
    blocker: NodeKey | None
    blocker_kind: str | None
    blocker_resource: str | None
    blocker_robustness_ns: int | None
    wait_source: str
    slack_ns: float
    critical: bool


@dataclass(frozen=True)
class RingAnalysis:
    """Measured occupancy, run-ahead, and reuse backpressure for one ring."""

    resource: str
    depth: int
    peak_occupancy: int
    final_occupancy: int
    mean_occupancy: float
    full_fraction: float
    max_run_ahead: int
    reuse_wait_ns: int


@dataclass(frozen=True)
class DepthCounterfactual:
    """First-order makespan estimate after increasing one ring depth."""

    resource: str
    baseline_depth: int
    candidate_depth: int
    baseline_makespan: float
    candidate_makespan: float
    saved: float


@dataclass(frozen=True)
class CriticalCycle:
    """Maximum cycle ratio of the one-iteration modulo schedule.

    ``initiation_interval`` (RecMII) is the largest cycle weight / iteration distance;
    ``resource_initiation_interval`` (ResMII) is the busiest role's total weight.
    """

    initiation_interval: float
    resource_initiation_interval: float
    nodes: tuple[str, ...]
    edge_kinds: tuple[str, ...]
    total_weight: float
    total_distance: int
    cycles_checked: int


@dataclass(frozen=True)
class PipelineAnalysis:
    """Combined finite and steady-state scheduling analysis."""

    logical_critical_path: CriticalPath
    measured_critical_path: CriticalPath
    blockers: tuple[StageBlocker, ...]
    rings: tuple[RingAnalysis, ...]
    depth_counterfactuals: tuple[DepthCounterfactual, ...]
    logical_critical_cycle: CriticalCycle

    def perfetto_annotations(self) -> dict[NodeKey, dict[str, object]]:
        """Per-region slice arguments for Perfetto, keyed by ``(name, iteration)``."""
        cycle_nodes = set(self.logical_critical_cycle.nodes)
        annotations = {}
        for stage in self.blockers:
            key = stage.name, stage.iteration
            annotations[key] = {
                "ready_ns": stage.ready_ns,
                "wait_inside_ns": stage.wait_inside_ns,
                "ready_delay_ns": stage.ready_delay_ns,
                "busy_ns": stage.busy_ns,
                "wait_source": stage.wait_source,
                "blocker": format_node(stage.blocker),
                "blocker_kind": stage.blocker_kind or "none",
                "blocker_resource": stage.blocker_resource or "none",
                "blocker_robustness_ns": stage.blocker_robustness_ns or 0,
                "critical_path": stage.critical,
                "slack_ns": stage.slack_ns,
                "critical_cycle_member": stage.name in cycle_nodes,
            }
        return annotations

    def perfetto_summary(self) -> dict[str, object]:
        """Trace-level scheduling findings, flattened to scalar fields."""
        summary: dict[str, object] = {
            "logical_makespan": self.logical_critical_path.makespan,
            "measured_modeled_makespan_ns": self.measured_critical_path.makespan,
            "recurrence_mii": self.logical_critical_cycle.initiation_interval,
            "resource_mii": self.logical_critical_cycle.resource_initiation_interval,
            "critical_cycle": " -> ".join(self.logical_critical_cycle.nodes),
            "critical_path": " -> ".join(
                format_node(node) for node in self.measured_critical_path.path
            ),
        }
        for ring in self.rings:
            summary[f"ring.{ring.resource}.peak"] = ring.peak_occupancy
            summary[f"ring.{ring.resource}.depth"] = ring.depth
            summary[f"ring.{ring.resource}.reuse_wait_ns"] = ring.reuse_wait_ns
        for candidate in self.depth_counterfactuals:
            summary[f"depth_plus_one.{candidate.resource}.saved_ns"] = candidate.saved
        return summary


def analyze_pipeline(
    timeline: Timeline,
    measured: MeasuredCapture,
    *,
    timer_epsilon_ns: int = 32,
    cycle_cap: int = 10_000,
) -> PipelineAnalysis:
    """Analyze the unrolled logical plan and one wait-inclusive measured capture.

    A region's wait is its measured ``wait_ns`` when the capture traced waits, else inferred
    from the latest predecessor's end; an inferred wait within ``timer_epsilon_ns`` counts as
    none.

    ``cycle_cap`` bounds the simple-cycle enumeration behind the critical cycle. Densely
    annotated plans (tens of regions per role) can need a few million.
    """
    edges = concrete_edges(timeline)
    logical_weights = {item.key: float(item.region.weight) for item in timeline.regions}
    logical_critical_path = critical_path(tuple(logical_weights), edges, logical_weights)
    blockers, measured_weights = measured_blockers(
        timeline,
        measured,
        edges,
        timer_epsilon_ns=timer_epsilon_ns,
    )
    measured_critical_path = critical_path(tuple(measured_weights), edges, measured_weights)
    blockers = tuple(
        replace(
            item,
            slack_ns=measured_critical_path.slack[(item.name, item.iteration)],
            critical=measured_critical_path.slack[(item.name, item.iteration)] == 0,
        )
        for item in blockers
    )
    return PipelineAnalysis(
        logical_critical_path=logical_critical_path,
        measured_critical_path=measured_critical_path,
        blockers=blockers,
        rings=ring_analysis(timeline, measured, blockers),
        depth_counterfactuals=depth_counterfactuals(timeline, measured_weights),
        logical_critical_cycle=critical_cycle(timeline, logical_weights, cycle_cap=cycle_cap),
    )


def concrete_edges(
    timeline: Timeline,
    *,
    depth_overrides: Mapping[str, int] | None = None,
) -> tuple[ConcreteEdge, ...]:
    """Unroll role program order and plan dependencies over the timeline's iterations.

    ``depth_overrides`` replaces a resource's ring depth, the distance of its reuse edges.
    """
    depth_overrides = depth_overrides or {}
    edges: list[ConcreteEdge] = []
    regions_by_role: dict[str, list[object]] = defaultdict(list)
    for item in timeline.regions:
        if item.iteration == 0:
            regions_by_role[item.region.role].append(item.region)
    for role, regions in regions_by_role.items():
        regions.sort(key=lambda region: region.order)
        for iteration in range(timeline.iterations):
            for left, right in itertools.pairwise(regions):
                edges.append(
                    ConcreteEdge(
                        (left.name, iteration),
                        (right.name, iteration),
                        "role",
                        role,
                    )
                )
            if iteration + 1 < timeline.iterations:
                edges.append(
                    ConcreteEdge(
                        (regions[-1].name, iteration),
                        (regions[0].name, iteration + 1),
                        "role",
                        role,
                    )
                )

    for dependency in timeline.plan_dependencies:
        distance = (
            depth_overrides.get(dependency.resource, dependency.distance)
            if dependency.kind == "reuse"
            else dependency.distance
        )
        for source_iteration in range(timeline.iterations):
            target_iteration = source_iteration + distance
            if target_iteration >= timeline.iterations:
                continue
            edges.append(
                ConcreteEdge(
                    (dependency.source, source_iteration),
                    (dependency.target, target_iteration),
                    dependency.kind,
                    dependency.resource,
                )
            )
    return tuple(dict.fromkeys(edges))


def critical_path(
    nodes: tuple[NodeKey, ...],
    edges: tuple[ConcreteEdge, ...],
    weights: Mapping[NodeKey, float],
) -> CriticalPath:
    """Compute earliest/latest times, slack, and one deterministic critical path."""
    predecessors: dict[NodeKey, list[ConcreteEdge]] = defaultdict(list)
    successors: dict[NodeKey, list[ConcreteEdge]] = defaultdict(list)
    indegree: dict[NodeKey, int] = dict.fromkeys(nodes, 0)
    for edge in edges:
        predecessors[edge.target].append(edge)
        successors[edge.source].append(edge)
        indegree[edge.target] += 1
    queue = deque(node for node in nodes if indegree[node] == 0)
    order: list[NodeKey] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for edge in successors[node]:
            indegree[edge.target] -= 1
            if indegree[edge.target] == 0:
                queue.append(edge.target)
    if len(order) != len(nodes):
        raise ValueError("Critical-path graph is cyclic after finite unrolling")

    earliest: dict[NodeKey, float] = {}
    binding: dict[NodeKey, ConcreteEdge] = {}
    for node in order:
        candidates = [
            (earliest[edge.source] + weights[edge.source], edge) for edge in predecessors[node]
        ]
        if candidates:
            end, edge = max(
                candidates,
                key=lambda item: (item[0], item[1].kind == "role", item[1].source),
            )
            earliest[node] = end
            binding[node] = edge
        else:
            earliest[node] = 0.0
    makespan = max(earliest[node] + weights[node] for node in nodes)

    latest_start: dict[NodeKey, float] = {}
    for node in reversed(order):
        finish = min((latest_start[edge.target] for edge in successors[node]), default=makespan)
        latest_start[node] = finish - weights[node]
    slack = {node: latest_start[node] - earliest[node] for node in nodes}

    sink = max(nodes, key=lambda node: (earliest[node] + weights[node], node))
    path = [sink]
    while path[-1] in binding:
        path.append(binding[path[-1]].source)
    path.reverse()
    return CriticalPath(makespan, earliest, latest_start, slack, tuple(path))


def measured_blockers(
    timeline: Timeline,
    measured: MeasuredCapture,
    edges: tuple[ConcreteEdge, ...],
    *,
    timer_epsilon_ns: int,
) -> tuple[tuple[StageBlocker, ...], dict[NodeKey, float]]:
    """Split each region into wait and busy time; blame a wait on the latest predecessor.

    Also returns each region's weight for the measured critical path: busy time plus the
    delay between becoming ready and starting.
    """
    measured_by_key = {(item.name, item.iteration): item for item in measured.regions}
    predecessors: dict[NodeKey, list[ConcreteEdge]] = defaultdict(list)
    for edge in edges:
        predecessors[edge.target].append(edge)
    blockers: list[StageBlocker] = []
    weights: dict[NodeKey, float] = {}
    for key, item in measured_by_key.items():
        start_ns = item.start_ns - measured.origin_ns
        end_ns = item.end_ns - measured.origin_ns
        arrivals = sorted(
            [
                (
                    measured_by_key[edge.source].end_ns - measured.origin_ns,
                    edge,
                )
                for edge in predecessors[key]
            ],
            key=lambda item: (item[0], item[1].kind, item[1].source),
        )
        ready_ns, binding = arrivals[-1] if arrivals else (0, None)
        robustness = ready_ns - arrivals[-2][0] if len(arrivals) > 1 else None
        ready_delay_ns = max(start_ns - ready_ns, 0)
        if item.wait_ns is None:
            wait_inside_ns = min(max(ready_ns - start_ns, 0), end_ns - start_ns)
            if wait_inside_ns <= timer_epsilon_ns:
                wait_inside_ns = 0
            busy_ns = max(end_ns - max(start_ns, ready_ns), 0)
        else:
            wait_inside_ns = min(item.wait_ns, end_ns - start_ns)
            busy_ns = end_ns - start_ns - wait_inside_ns
        weights[key] = float(busy_ns + ready_delay_ns)
        blocker = binding if wait_inside_ns else None
        blockers.append(
            StageBlocker(
                name=key[0],
                iteration=key[1],
                start_ns=start_ns,
                end_ns=end_ns,
                duration_ns=end_ns - start_ns,
                ready_ns=ready_ns,
                wait_inside_ns=wait_inside_ns,
                ready_delay_ns=ready_delay_ns,
                busy_ns=busy_ns,
                blocker=blocker.source if blocker else None,
                blocker_kind=blocker.kind if blocker else None,
                blocker_resource=blocker.resource if blocker else None,
                blocker_robustness_ns=robustness,
                wait_source="inferred" if item.wait_ns is None else "measured",
                slack_ns=0.0,
                critical=False,
            )
        )
    blockers.sort(key=lambda item: (item.start_ns, item.name, item.iteration))
    return tuple(blockers), weights


def reuse_rings(timeline: Timeline) -> dict[str, tuple[Dependency, ...]]:
    """Each ring resource's reuse dependencies, one per releasing reader."""
    rings: dict[str, list[Dependency]] = defaultdict(list)
    for dependency in timeline.plan_dependencies:
        if dependency.kind == "reuse":
            rings[dependency.resource].append(dependency)
    return {resource: tuple(dependencies) for resource, dependencies in rings.items()}


def ring_lifetimes(
    timeline: Timeline,
    measured: MeasuredCapture,
    dependencies: tuple[Dependency, ...],
) -> tuple[list[int], list[int]]:
    """Origin-relative fill and release times of one ring's slots, one of each per iteration.

    A slot is full once the last of its producers finishes and free again once the last of its
    readers releases it.
    """
    measured_by_key = {(item.name, item.iteration): item for item in measured.regions}
    producers = tuple(dict.fromkeys(dependency.target for dependency in dependencies))
    readers = tuple(dict.fromkeys(dependency.source for dependency in dependencies))

    def last_end(regions: tuple[str, ...], iteration: int) -> list[int]:
        ends = [
            measured_by_key[(name, iteration)].end_ns
            for name in regions
            if (name, iteration) in measured_by_key
        ]
        return [max(ends) - measured.origin_ns] if ends else []

    fills: list[int] = []
    releases: list[int] = []
    for iteration in range(timeline.iterations):
        fills += last_end(producers, iteration)
        releases += last_end(readers, iteration)
    return fills, releases


def ring_occupancy_samples(
    timeline: Timeline,
    measured: MeasuredCapture,
) -> dict[str, tuple[tuple[int, int], ...]]:
    """Origin-relative occupancy samples of every ring, starting from ``(0, 0)``."""
    samples = {}
    for resource, dependencies in reuse_rings(timeline).items():
        fills, releases = ring_lifetimes(timeline, measured, dependencies)
        deltas: dict[int, int] = defaultdict(int)
        for timestamp in fills:
            deltas[timestamp] += 1
        for timestamp in releases:
            deltas[timestamp] -= 1
        occupancy = 0
        resource_samples = [(0, 0)]
        for timestamp, delta in sorted(deltas.items()):
            occupancy += delta
            resource_samples.append((timestamp, occupancy))
        samples[resource] = tuple(resource_samples)
    return samples


def ring_analysis(
    timeline: Timeline,
    measured: MeasuredCapture,
    blockers: Iterable[StageBlocker],
) -> tuple[RingAnalysis, ...]:
    """Measure ring occupancy, run-ahead, and reuse-attributed wait."""
    resources = {resource.name: resource for resource in timeline.resources}
    results = []
    for resource_name, dependencies in reuse_rings(timeline).items():
        resource = resources[resource_name]
        fills, releases = ring_lifetimes(timeline, measured, dependencies)
        events = sorted([(timestamp, 1) for timestamp in fills] + [(t, -1) for t in releases])
        occupancy = 0
        peak = 0
        area = 0
        full_time = 0
        previous_time = events[0][0]
        for timestamp, delta in events:
            duration = timestamp - previous_time
            area += occupancy * duration
            if occupancy == resource.depth:
                full_time += duration
            occupancy += delta
            peak = max(peak, occupancy)
            previous_time = timestamp
        span = max(events[-1][0] - events[0][0], 1)
        max_run_ahead = 0
        for fill_index, timestamp in enumerate(sorted(fills), start=1):
            released = sum(release <= timestamp for release in releases)
            max_run_ahead = max(max_run_ahead, fill_index - released)
        reuse_wait_ns = sum(
            stage.wait_inside_ns
            for stage in blockers
            if stage.blocker_kind == "reuse" and stage.blocker_resource == resource_name
        )
        results.append(
            RingAnalysis(
                resource=resource_name,
                depth=resource.depth,
                peak_occupancy=peak,
                final_occupancy=occupancy,
                mean_occupancy=area / span,
                full_fraction=full_time / span,
                max_run_ahead=max_run_ahead,
                reuse_wait_ns=reuse_wait_ns,
            )
        )
    return tuple(results)


def depth_counterfactuals(
    timeline: Timeline,
    weights: Mapping[NodeKey, float],
) -> tuple[DepthCounterfactual, ...]:
    """Estimate depth+1 benefit by relaxing one reuse distance at a time."""
    nodes = tuple(weights)
    baseline = critical_path(nodes, concrete_edges(timeline), weights).makespan
    resources = {resource.name: resource for resource in timeline.resources}
    results = []
    for resource_name in reuse_rings(timeline):
        resource = resources[resource_name]
        candidate = critical_path(
            nodes,
            concrete_edges(
                timeline,
                depth_overrides={resource_name: resource.depth + 1},
            ),
            weights,
        ).makespan
        results.append(
            DepthCounterfactual(
                resource_name,
                resource.depth,
                resource.depth + 1,
                baseline,
                candidate,
                baseline - candidate,
            )
        )
    return tuple(results)


def critical_cycle(
    timeline: Timeline,
    logical_weights: Mapping[NodeKey, float],
    *,
    cycle_cap: int = 10_000,
) -> CriticalCycle:
    """Return the simple cycle of the one-iteration graph with the largest weight/distance.

    Region weights are medians over iterations. Raises ValueError past ``cycle_cap`` cycles or
    when no cycle has a positive distance.
    """
    weights_by_region: dict[str, list[float]] = defaultdict(list)
    for (name, _), weight in logical_weights.items():
        weights_by_region[name].append(weight)
    region_weights = {
        name: statistics.median(weights) for name, weights in weights_by_region.items()
    }
    symbolic: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
    roles: dict[str, list[object]] = defaultdict(list)
    for item in timeline.regions:
        if item.iteration == 0:
            roles[item.region.role].append(item.region)
    for role, regions in roles.items():
        regions.sort(key=lambda region: region.order)
        for left, right in itertools.pairwise(regions):
            symbolic[left.name].append((right.name, "role", 0))
        symbolic[regions[-1].name].append((regions[0].name, f"role:{role}", 1))
    for dependency in timeline.plan_dependencies:
        symbolic[dependency.source].append(
            (dependency.target, f"{dependency.kind}:{dependency.resource}", dependency.distance)
        )

    cycles: list[tuple[tuple[str, ...], tuple[str, ...], float, int]] = []
    for start in sorted(region_weights):
        stack: list[tuple[str, list[str], list[str], int]] = [(start, [start], [], 0)]
        while stack:
            node, path, kinds, distance = stack.pop()
            for target, kind, edge_distance in symbolic[node]:
                if target == start:
                    total_distance = distance + edge_distance
                    if total_distance:
                        total_weight = sum(region_weights[item] for item in path)
                        cycles.append(
                            (tuple(path), tuple(kinds + [kind]), total_weight, total_distance)
                        )
                        if len(cycles) > cycle_cap:
                            raise ValueError(f"Cycle enumeration exceeded cap {cycle_cap}")
                elif target not in path and target >= start:
                    stack.append(
                        (target, path + [target], kinds + [kind], distance + edge_distance)
                    )
    if not cycles:
        raise ValueError("Modulo schedule contains no positive-distance cycle")
    nodes, kinds, total_weight, total_distance = max(
        cycles,
        key=lambda cycle: (cycle[2] / cycle[3], cycle[0]),
    )
    role_busy = max(
        sum(region_weights[region.name] for region in regions) for regions in roles.values()
    )
    return CriticalCycle(
        initiation_interval=total_weight / total_distance,
        resource_initiation_interval=role_busy,
        nodes=nodes,
        edge_kinds=kinds,
        total_weight=total_weight,
        total_distance=total_distance,
        cycles_checked=len(cycles),
    )


def format_node(node: NodeKey | None) -> str:
    """Format a node as ``name[iteration]``, or ``none``."""
    return "none" if node is None else f"{node[0]}[{node[1]}]"
