"""Host-only model for source-linked, iteration-aware GPU pipeline timelines.

The prototype models a warp-specialized kernel as ordered regions on role lanes.
Dependencies may remain within one logical iteration or cross iterations for
recurrent state and circular-buffer reuse. Static durations are dimensionless
logical weights; measured timestamps can replace them later.
"""

from __future__ import annotations

import itertools
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Literal

DependencyKind = Literal["data", "state", "reuse"]


@dataclass(frozen=True)
class SourceLocation:
    """A source span associated with an annotated role or region."""

    path: str
    line: int
    end_line: int | None = None


@dataclass(frozen=True)
class Role:
    """One visible timeline lane owned by a contiguous warp range."""

    name: str
    label: str
    warp_start: int
    warp_end: int
    source: SourceLocation | None = None


@dataclass(frozen=True)
class Resource:
    """A logical value, state, or circular pipeline buffer."""

    name: str
    label: str
    depth: int
    kind: str = "buffer"


@dataclass(frozen=True)
class Region:
    """One ordered span of work repeated on a role for each iteration."""

    name: str
    role: str
    label: str
    order: int
    weight: int
    source: SourceLocation | None = None


@dataclass(frozen=True)
class Dependency:
    """A producer-to-consumer constraint with an affine iteration distance."""

    source: str
    target: str
    resource: str
    distance: int = 0
    kind: DependencyKind = "data"
    label: str | None = None


@dataclass(frozen=True)
class ScheduledRegion:
    """One unrolled region placed on the logical timeline."""

    region: Region
    iteration: int
    start: int
    end: int

    @property
    def key(self) -> tuple[str, int]:
        return self.region.name, self.iteration


@dataclass(frozen=True)
class ScheduledDependency:
    """One concrete dependency between unrolled scheduled regions."""

    dependency: Dependency
    source_iteration: int
    target_iteration: int


@dataclass(frozen=True)
class Timeline:
    """Scheduled regions and dependencies for a finite iteration window.

    ``dependencies`` are the window's concrete edges; ``plan_dependencies`` the plan's
    symbolic ones, including those whose distance reaches past the window, which steady-state
    analysis (recurrences, rings) needs.
    """

    plan_name: str
    iteration_name: str
    iterations: int
    roles: tuple[Role, ...]
    resources: tuple[Resource, ...]
    regions: tuple[ScheduledRegion, ...]
    dependencies: tuple[ScheduledDependency, ...]
    plan_dependencies: tuple[Dependency, ...]
    logical_duration: int


class PipelinePlan:
    """Build and validate a symbolic modulo-scheduling plan."""

    def __init__(self, name: str, *, iteration_name: str) -> None:
        self.name = name
        self.iteration_name = iteration_name
        self.roles: dict[str, Role] = {}
        self.resources: dict[str, Resource] = {}
        self.regions: dict[str, Region] = {}
        self.dependencies: list[Dependency] = []

    def add_role(self, role: Role) -> None:
        """Register a unique warp-role lane."""
        if role.name in self.roles:
            raise ValueError(f"Duplicate role {role.name!r}")
        if role.warp_start < 0 or role.warp_end <= role.warp_start:
            raise ValueError(f"Invalid warp range for role {role.name!r}")
        self.roles[role.name] = role

    def add_resource(self, resource: Resource) -> None:
        """Register a unique logical resource."""
        if resource.name in self.resources:
            raise ValueError(f"Duplicate resource {resource.name!r}")
        if resource.depth < 1:
            raise ValueError(f"Resource {resource.name!r} must have depth >= 1")
        self.resources[resource.name] = resource

    def add_region(self, region: Region) -> None:
        """Register a unique region on an existing role."""
        if region.name in self.regions:
            raise ValueError(f"Duplicate region {region.name!r}")
        if region.role not in self.roles:
            raise ValueError(f"Unknown role {region.role!r} for region {region.name!r}")
        if region.weight < 1:
            raise ValueError(f"Region {region.name!r} must have weight >= 1")
        self.regions[region.name] = region

    def add_dependency(self, dependency: Dependency) -> None:
        """Register one data, state, or reuse constraint."""
        if dependency.source not in self.regions:
            raise ValueError(f"Unknown dependency source {dependency.source!r}")
        if dependency.target not in self.regions:
            raise ValueError(f"Unknown dependency target {dependency.target!r}")
        if dependency.resource not in self.resources:
            raise ValueError(f"Unknown dependency resource {dependency.resource!r}")
        if dependency.distance < 0:
            raise ValueError("Dependency distance must be non-negative")
        self.dependencies.append(dependency)

    def schedule(self, iterations: int) -> Timeline:
        """Unroll ``iterations`` and place every region at its earliest legal tick."""
        if iterations < 1:
            raise ValueError("iterations must be positive")

        nodes = [
            (region_name, iteration)
            for iteration in range(iterations)
            for region_name in self.regions
        ]
        successors: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)
        indegree: dict[tuple[str, int], int] = dict.fromkeys(nodes, 0)
        concrete_dependencies: list[ScheduledDependency] = []
        seen_edges: set[tuple[tuple[str, int], tuple[str, int]]] = set()
        seen_dependencies: set[tuple[object, ...]] = set()

        def add_edge(
            source: tuple[str, int],
            target: tuple[str, int],
            dependency: Dependency | None,
        ) -> None:
            if dependency is not None:
                dependency_key = (
                    source,
                    target,
                    dependency.resource,
                    dependency.kind,
                    dependency.label,
                )
                if dependency_key not in seen_dependencies:
                    seen_dependencies.add(dependency_key)
                    concrete_dependencies.append(
                        ScheduledDependency(dependency, source[1], target[1])
                    )

            edge = source, target
            if edge in seen_edges:
                return
            seen_edges.add(edge)
            successors[source].append(target)
            indegree[target] += 1

        regions_by_role: dict[str, list[Region]] = defaultdict(list)
        for region in self.regions.values():
            regions_by_role[region.role].append(region)
        for role_regions in regions_by_role.values():
            role_regions.sort(key=lambda region: region.order)
            orders = [region.order for region in role_regions]
            if len(orders) != len(set(orders)):
                raise ValueError("Region order must be unique within each role")
            for iteration in range(iterations):
                for left, right in itertools.pairwise(role_regions):
                    add_edge((left.name, iteration), (right.name, iteration), None)
                if iteration + 1 < iterations:
                    add_edge(
                        (role_regions[-1].name, iteration),
                        (role_regions[0].name, iteration + 1),
                        None,
                    )

        for dependency in self.dependencies:
            for source_iteration in range(iterations):
                target_iteration = source_iteration + dependency.distance
                if target_iteration >= iterations:
                    continue
                add_edge(
                    (dependency.source, source_iteration),
                    (dependency.target, target_iteration),
                    dependency,
                )

        queue = deque(node for node in nodes if indegree[node] == 0)
        starts: dict[tuple[str, int], int] = dict.fromkeys(nodes, 0)
        visited = 0
        while queue:
            source = queue.popleft()
            visited += 1
            source_end = starts[source] + self.regions[source[0]].weight
            for target in successors[source]:
                starts[target] = max(starts[target], source_end)
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)

        if visited != len(nodes):
            blocked = sorted(node for node, degree in indegree.items() if degree)
            raise ValueError(f"Dependency graph contains a same-window cycle: {blocked}")

        scheduled = tuple(
            sorted(
                (
                    ScheduledRegion(
                        region=self.regions[name],
                        iteration=iteration,
                        start=starts[(name, iteration)],
                        end=starts[(name, iteration)] + self.regions[name].weight,
                    )
                    for name, iteration in nodes
                ),
                key=lambda item: (item.start, item.region.role, item.region.order, item.iteration),
            )
        )
        return Timeline(
            plan_name=self.name,
            iteration_name=self.iteration_name,
            iterations=iterations,
            roles=tuple(self.roles.values()),
            resources=tuple(self.resources.values()),
            regions=scheduled,
            dependencies=tuple(concrete_dependencies),
            plan_dependencies=tuple(self.dependencies),
            logical_duration=max(item.end for item in scheduled),
        )
