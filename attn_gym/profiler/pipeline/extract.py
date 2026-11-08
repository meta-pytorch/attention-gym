"""Static extractor: lower literal ``PipelineAnnotations`` declarations to a ``PipelinePlan``."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path

from attn_gym.profiler.pipeline.annotations import (
    PipelineAnnotations,
    indexed_region_name,
)
from attn_gym.profiler.pipeline.plan import (
    Dependency,
    PipelinePlan,
    Region,
    Resource,
    Role,
    SourceLocation,
)


@dataclass(frozen=True)
class ParsedRegion:
    """One inline region marker plus its token declarations."""

    region: Region
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    releases: tuple[str, ...]


def extract_plan(path: str | Path) -> PipelinePlan:
    """Extract a validated pipeline plan from inline source annotations."""
    return _AnnotatedPlanExtractor(path).extract()


class _AnnotatedPlanExtractor:
    """Lower literal ``PipelineAnnotations`` calls from Python AST to a plan."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.text = self.path.read_text()
        self.tree = ast.parse(self.text, filename=str(self.path))
        self.pipeline_name, self.iteration_name, self.annotation_name = self.find_pipeline()

    def extract(self) -> PipelinePlan:
        """Build a validated plan and derive token and ring-reuse dependencies."""
        plan = PipelinePlan(self.pipeline_name, iteration_name=self.iteration_name)
        for resource in self.find_resources():
            plan.add_resource(resource)

        parsed_roles = self.find_roles()
        for _, role in parsed_roles:
            plan.add_role(role)

        parsed_regions: list[ParsedRegion] = []
        for function, role in parsed_roles:
            role_regions = self.find_regions(function, role)
            parsed_regions.extend(role_regions)
            for parsed_region in role_regions:
                plan.add_region(parsed_region.region)

        producers: dict[str, tuple[str, int]] = {}
        releases: dict[str, str] = {}
        for parsed in parsed_regions:
            for token in parsed.produces:
                resource, offset = parse_token(token)
                if resource in producers:
                    raise ValueError(
                        f"Resource {resource!r} has multiple producers: "
                        f"{producers[resource][0]!r} and {parsed.region.name!r}"
                    )
                producers[resource] = parsed.region.name, offset
            for resource in parsed.releases:
                if resource in releases:
                    raise ValueError(
                        f"Resource {resource!r} has multiple release regions: "
                        f"{releases[resource]!r} and {parsed.region.name!r}"
                    )
                releases[resource] = parsed.region.name

        for parsed in parsed_regions:
            for token in parsed.consumes:
                resource_name, consumer_offset = parse_token(token)
                if resource_name not in producers:
                    raise ValueError(
                        f"Region {parsed.region.name!r} consumes {resource_name!r} "
                        "without an annotated producer"
                    )
                producer_name, producer_offset = producers[resource_name]
                distance = producer_offset - consumer_offset
                if distance < 0:
                    raise ValueError(
                        f"Dependency {producer_name!r} -> {parsed.region.name!r} "
                        f"requires negative distance {distance}"
                    )
                resource = plan.resources[resource_name]
                plan.add_dependency(
                    Dependency(
                        producer_name,
                        parsed.region.name,
                        resource_name,
                        distance=distance,
                        kind="state" if resource.kind == "state" else "data",
                        label=(
                            f"{resource.label}[c+{producer_offset}]"
                            if distance
                            else resource.label
                        ),
                    )
                )

        for resource_name, release_region in releases.items():
            if resource_name not in producers:
                raise ValueError(f"Released resource {resource_name!r} has no producer")
            resource = plan.resources[resource_name]
            if resource.kind == "state":
                raise ValueError(f"State resource {resource_name!r} cannot use ring release")
            producer_region, _ = producers[resource_name]
            plan.add_dependency(
                Dependency(
                    release_region,
                    producer_region,
                    resource_name,
                    distance=resource.depth,
                    kind="reuse",
                    label=f"reuse {resource.label}",
                )
            )
        return plan

    def find_pipeline(self) -> tuple[str, str, str]:
        """Find the unique module-level ``PipelineAnnotations`` assignment."""
        matches: list[tuple[str, ast.Call]] = []
        for node in self.tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            if not isinstance(node.targets[0], ast.Name) or not isinstance(node.value, ast.Call):
                continue
            if call_name(node.value.func) == "PipelineAnnotations":
                matches.append((node.targets[0].id, node.value))
        if len(matches) != 1:
            raise ValueError(f"Expected one PipelineAnnotations assignment, found {len(matches)}")
        annotation_name, call = matches[0]
        name = positional_literal(call, 0, str)
        iteration_name = keyword_literal(call, "iteration_name", str)
        return name, iteration_name, annotation_name

    def find_resources(self) -> list[Resource]:
        """Read module-level resource declarations in source order."""
        resources: list[Resource] = []
        for node in self.tree.body:
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            call = node.value
            if not is_annotation_call(call, self.annotation_name, "resource"):
                continue
            check_keywords(call, "resource")
            resources.append(
                Resource(
                    name=positional_literal(call, 0, str),
                    label=keyword_literal(call, "label", str),
                    depth=keyword_literal(call, "depth", int),
                    kind=keyword_literal(call, "kind", str, default="buffer"),
                )
            )
        if not resources:
            raise ValueError("Annotated pipeline declares no resources")
        return resources

    def find_roles(
        self,
    ) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, Role]]:
        """Read role decorators from nested functions in source order."""
        roles: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, Role]] = []
        functions = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for function in sorted(functions, key=lambda node: node.lineno):
            decorators = [
                decorator
                for decorator in function.decorator_list
                if isinstance(decorator, ast.Call)
                and is_annotation_call(decorator, self.annotation_name, "role")
            ]
            if not decorators:
                continue
            if len(decorators) != 1:
                raise ValueError(f"Function {function.name!r} has multiple role annotations")
            call = decorators[0]
            check_keywords(call, "role")
            roles.append(
                (
                    function,
                    Role(
                        name=positional_literal(call, 0, str),
                        label=keyword_literal(call, "label", str),
                        warp_start=keyword_literal(call, "warp_start", int),
                        warp_end=keyword_literal(call, "warp_end", int),
                        source=SourceLocation(
                            str(self.path), function.lineno, function.end_lineno
                        ),
                    ),
                )
            )
        if not roles:
            raise ValueError("Annotated pipeline declares no roles")
        return roles

    def find_regions(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        role: Role,
    ) -> list[ParsedRegion]:
        """Read region marker calls within one role function."""
        calls = [
            node
            for node in descendants_without_nested_functions(function)
            if isinstance(node, ast.Call)
            and is_annotation_call(node, self.annotation_name, "region")
        ]
        calls.sort(key=lambda node: node.lineno)
        regions: list[ParsedRegion] = []
        for position, call in enumerate(calls):
            end_line = (
                calls[position + 1].lineno - 1
                if position + 1 < len(calls)
                else function.end_lineno
            )
            source = SourceLocation(str(self.path), call.lineno, end_line)
            regions.extend(self.expand_region(call, role, source, len(regions)))
        if not regions:
            raise ValueError(f"Role function {function.name!r} declares no regions")
        return regions

    def expand_region(
        self,
        call: ast.Call,
        role: Role,
        source: SourceLocation,
        first_order: int,
    ) -> list[ParsedRegion]:
        """Lower one region call into ``count`` regions with per-copy tokens."""
        check_keywords(call, "region")
        name = positional_literal(call, 0, str)
        count = keyword_literal(call, "count", int, default=1)
        if count < 1:
            raise ValueError(f"Region {name!r} at line {call.lineno} has count {count} < 1")
        has_index = any(item.arg == "index" for item in call.keywords)
        if (count > 1) != has_index:
            raise ValueError(
                f"Region {name!r} at line {call.lineno} must pass both count>1 and index"
            )
        label = keyword_literal(call, "label", str)
        weight = keyword_literal(call, "weight", int)
        consumes = keyword_string_tuple(call, "consumes")
        produces = keyword_string_tuple(call, "produces")
        releases = keyword_string_tuple(call, "releases")
        if count == 1:
            region = Region(name, role.name, label, first_order, weight, source)
            return [ParsedRegion(region, consumes, produces, releases)]

        def copy_tokens(tokens: tuple[str, ...], index: int, plain_at: int) -> tuple[str, ...]:
            return tuple(
                token.replace("{index}", str(index))
                for token in tokens
                if "{index}" in token or index == plain_at
            )

        return [
            ParsedRegion(
                Region(
                    indexed_region_name(name, index),
                    role.name,
                    f"{label} [{index}]",
                    first_order + index,
                    weight,
                    source,
                ),
                consumes=copy_tokens(consumes, index, 0),
                produces=copy_tokens(produces, index, count - 1),
                releases=copy_tokens(releases, index, count - 1),
            )
            for index in range(count)
        ]


def descendants_without_nested_functions(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[ast.AST]:
    """Return descendants of ``function`` without entering nested callables."""
    descendants: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
            ):
                continue
            descendants.append(child)
            visit(child)

    visit(function)
    return descendants


def parse_token(token: str) -> tuple[str, int]:
    """Parse ``resource`` or ``resource@offset`` token syntax."""
    resource, separator, raw_offset = token.partition("@")
    if not resource:
        raise ValueError(f"Invalid empty resource token {token!r}")
    offset = int(raw_offset) if separator else 0
    return resource, offset


def is_annotation_call(call: ast.Call, object_name: str, method: str) -> bool:
    """Return whether ``call`` targets ``object_name.method``."""
    return (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == object_name
        and call.func.attr == method
    )


def call_name(node: ast.expr) -> str | None:
    """Return the terminal name of a call target."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def positional_literal(call: ast.Call, index: int, expected_type: type):
    """Read and type-check one literal positional argument."""
    if index >= len(call.args):
        raise ValueError(f"Missing positional argument {index} at line {call.lineno}")
    value = ast.literal_eval(call.args[index])
    if type(value) is not expected_type:
        raise ValueError(
            f"Expected {expected_type.__name__} at line {call.lineno}, got {type(value).__name__}"
        )
    return value


def check_keywords(call: ast.Call, method: str) -> None:
    """Reject keywords that ``PipelineAnnotations.<method>`` would reject when the kernel runs."""
    parameters = inspect.signature(getattr(PipelineAnnotations, method)).parameters
    unknown = [item.arg for item in call.keywords if item.arg not in parameters]
    if unknown:
        raise ValueError(f"Unknown {method}() keyword {unknown[0]!r} at line {call.lineno}")


def keyword_literal(call: ast.Call, name: str, expected_type: type, *, default=None):
    """Read and type-check one literal keyword argument."""
    keyword = next((item for item in call.keywords if item.arg == name), None)
    if keyword is None:
        if default is not None:
            return default
        raise ValueError(f"Missing keyword {name!r} at line {call.lineno}")
    value = ast.literal_eval(keyword.value)
    if type(value) is not expected_type:
        raise ValueError(
            f"Expected {expected_type.__name__} for {name!r} at line {call.lineno}, "
            f"got {type(value).__name__}"
        )
    return value


def keyword_string_tuple(call: ast.Call, name: str) -> tuple[str, ...]:
    """Read an optional tuple/list of string tokens."""
    keyword = next((item for item in call.keywords if item.arg == name), None)
    if keyword is None:
        return ()
    value = ast.literal_eval(keyword.value)
    if not isinstance(value, (tuple, list)) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{name!r} at line {call.lineno} must be a tuple/list of strings")
    return tuple(value)
