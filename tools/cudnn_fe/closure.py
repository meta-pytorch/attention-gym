"""Static import closure of the cudnn-frontend GDN + KDA kernels at one upstream revision.

Everything is read from git objects (``git show <rev>:<path>``), so no checkout of the tag is
needed. The roots are the upstream kernel and common modules that the Attention Gym drivers
(``cudnn_fe/{gdn,kda,summary,plan}.py``) import; the walk follows every ``import`` statement,
module level or nested, plus literal ``importlib.import_module``/``__import__`` calls, and
records whether each edge is nested so prune cuts stay visible. A dynamic import whose module is
not a string literal fails the walk rather than silently dropping a dependency.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

PACKAGE = "attn_gym.linear._delta_rule.cudnn_fe"
PACKAGE_RELPATH = PACKAGE.replace(".", "/")  # git path of the package in an Attention Gym checkout
DRIVERS = ("gdn.py", "kda.py", "summary.py", "plan.py")

# Vendored subpackage -> (upstream module prefix, upstream directory).
SOURCES = {
    "tile_dsl": ("cudnn.frost.tile_dsl", "python/cudnn/frost/tile_dsl"),
    "common": (
        "cudnn.linear_attention.frost.common",
        "python/cudnn/linear_attention/frost/common",
    ),
    "kernel": (
        "cudnn.linear_attention.frost.kernel",
        "python/cudnn/linear_attention/frost/kernel",
    ),
}
# Upstream host utilities the kernels reach; relocated to the package's Torch-backed ``_compat``.
COMPAT_MODULES = ("cudnn.frost.buffers", "cudnn.frost.device")
# The documented prune cut: the GDN chain prologue's compact_qdo (GDP d_v=64) backward fork.
PRUNE_MODULES = ("kernel/gdp_bprop_v64_f16.py", "kernel/gdp_bprop_v64_config.py")
PRUNE_ENTRY = "kernel/gdn_chain_prologue_f16.py"
# The upstream engines are the closure the verbatim v1.30 drop (PR A) vendored. They also reach common/expand.py,
# but only on the GDP path of gdn_engine, so it was never vendored.
ENGINE_ROOTS = (
    "cudnn.linear_attention.frost.gdn_engine",
    "cudnn.linear_attention.frost.kda_engine",
)
ENGINE_SKIP = ("common/expand.py",)


@cache
def package_dir() -> Path:
    """Directory of the installed ``cudnn_fe`` package (editable checkout or wheel)."""
    spec = importlib.util.find_spec(PACKAGE)
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(f"cannot locate {PACKAGE}; install attention-gym first")
    return Path(next(iter(spec.submodule_search_locations)))


@cache
def repo_root() -> Path:
    """Git checkout containing the package (needed only to read past revisions)."""
    out = subprocess.run(
        ["git", "-C", str(package_dir()), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode:
        raise SystemExit(f"{package_dir()} is not in a git checkout: {out.stderr.strip()}")
    return Path(out.stdout.strip())


class Upstream:
    """Read-only view of one revision of a cudnn-frontend clone."""

    def __init__(self, clone: Path, rev: str):
        self.clone = Path(clone)
        self.rev = rev
        out = self._git("ls-tree", "-r", "--name-only", rev, "--", "python/cudnn")
        self.files = frozenset(out.splitlines())

    def _git(self, *args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.clone), *args], capture_output=True, text=True, check=True
        ).stdout

    def show_bytes(self, path: str) -> bytes:
        return _show(str(self.clone), self.rev, path)

    def show(self, path: str) -> str:
        return self.show_bytes(path).decode()

    def module_file(self, module: str) -> str | None:
        """Upstream path of a module file; package ``__init__`` modules are not followed."""
        path = "python/" + module.replace(".", "/") + ".py"
        return path if path in self.files else None


@cache
def _show(clone: str, rev: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "-C", clone, "show", f"{rev}:{path}"], capture_output=True, check=True
    ).stdout


def vendored_path(module: str) -> str | None:
    """Package-relative path (``kernel/x.py``) of an upstream module, or None if not vendored."""
    for package, (prefix, _) in SOURCES.items():
        if module.startswith(prefix + "."):
            rest = module[len(prefix) + 1 :]
            if "." not in rest:
                return f"{package}/{rest}.py"
    return None


def upstream_path(rel: str) -> str:
    package, name = rel.split("/")
    return f"{SOURCES[package][1]}/{name}"


def upstream_module(rel: str) -> str:
    package, name = rel.split("/")
    return f"{SOURCES[package][0]}.{name.removesuffix('.py')}"


@dataclass(frozen=True)
class Edge:
    target: str
    nested: bool
    lineno: int


DYNAMIC_IMPORTS = {"importlib.import_module", "import_module", "__import__"}


def _dynamic_import(call: ast.Call, module: str, package: str) -> str:
    """Absolute target of a literal ``importlib.import_module``/``__import__`` call."""
    where = f"{module}:{call.lineno}"
    target = call.args[0] if call.args else None
    if not (isinstance(target, ast.Constant) and isinstance(target.value, str)):
        raise ValueError(f"{where}: dynamic import of a non-literal module; vendor it by hand")  # noqa: TRY004
    name = target.value
    if _dotted(call.func) == "__import__":
        level = call.args[4] if len(call.args) > 4 else None
        level = next((kw.value for kw in call.keywords if kw.arg == "level"), level)
        if level is not None and not (isinstance(level, ast.Constant) and level.value == 0):
            raise ValueError(f"{where}: relative __import__ is not supported")
        return name
    if not name.startswith("."):
        return name
    anchor = call.args[1] if len(call.args) > 1 else None
    anchor = next((kw.value for kw in call.keywords if kw.arg == "package"), anchor)
    if isinstance(anchor, ast.Name) and anchor.id == "__package__":
        base = package
    elif isinstance(anchor, ast.Constant) and isinstance(anchor.value, str):
        base = anchor.value
    else:
        raise ValueError(f"{where}: relative import_module needs a literal or __package__ anchor")
    level = len(name) - len(name.lstrip("."))
    parts = base.split(".")
    return ".".join(parts[: len(parts) - (level - 1)] + [name[level:]])


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def imports_of(source: str, module: str, exists) -> list[Edge]:
    """Absolute ``cudnn.*`` targets imported anywhere in ``source`` (nested edges flagged)."""
    package = module.rsplit(".", 1)[0]
    edges: list[Edge] = []

    def visit(node: ast.AST, nested: bool) -> None:
        for child in ast.iter_child_nodes(node):
            inner = nested or isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.If, ast.Try)
            )
            if isinstance(child, ast.Call) and _dotted(child.func) in DYNAMIC_IMPORTS:
                target = _dynamic_import(child, module, package)
                edges.append(Edge(target, nested, child.lineno))
            if isinstance(child, ast.Import):
                edges.extend(Edge(a.name, nested, child.lineno) for a in child.names)
            elif isinstance(child, ast.ImportFrom):
                if child.level:
                    parts = package.split(".")
                    base_parts = parts[: len(parts) - (child.level - 1)]
                    base = ".".join(base_parts + ([child.module] if child.module else []))
                else:
                    base = child.module or ""
                edges.append(Edge(base, nested, child.lineno))
                # `from pkg import name` may name a submodule.
                edges.extend(
                    Edge(f"{base}.{a.name}", nested, child.lineno)
                    for a in child.names
                    if exists(f"{base}.{a.name}")
                )
            visit(child, inner)

    visit(ast.parse(source), False)
    return [e for e in edges if e.target == "cudnn" or e.target.startswith("cudnn.")]


def driver_roots(package: Path | None = None) -> list[str]:
    """Upstream kernel/common/tile_dsl modules imported by the Attention Gym drivers."""
    package = package or package_dir()
    roots: set[str] = set()
    for name in DRIVERS:
        path = package / name
        if not path.exists():
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
                continue
            head, _, rest = node.module.partition(".")
            if head not in SOURCES:
                continue
            if rest:
                roots.add(upstream_module(f"{head}/{rest}.py"))
            else:
                roots.update(upstream_module(f"{head}/{a.name}.py") for a in node.names)
    return sorted(roots)


@dataclass
class Closure:
    files: dict[str, list[Edge]] = field(default_factory=dict)  # vendored rel path -> edges
    importers: dict[str, list[tuple[str, Edge]]] = field(default_factory=dict)
    compat: set[str] = field(default_factory=set)  # names imported from COMPAT_MODULES
    not_upstream: list[str] = field(default_factory=list)  # AG-authored roots
    external: set[str] = field(default_factory=set)  # unvendorable cudnn modules reached
    cut: list[Edge] = field(default_factory=list)  # PRUNE_ENTRY -> PRUNE_MODULES edges not taken

    def nested_only(self, rel: str) -> bool:
        edges = [e for _, e in self.importers.get(rel, [])]
        return bool(edges) and all(e.nested for e in edges)


def compute(
    upstream: Upstream, roots: list[str], prune: bool = False, skip: tuple[str, ...] = ()
) -> Closure:
    """Walk imports from upstream ``roots`` modules and collect the vendorable files they reach.

    Roots outside the vendored directories (the engines) are parsed but not vendored. ``prune``
    drops only the documented edge (PRUNE_ENTRY importing a PRUNE_MODULES file) and fails if a GDP
    module is still reachable some other way; ``skip`` lists package-relative files never followed.
    """
    result = Closure()
    stop = set(skip)

    def exists(module: str) -> bool:
        return upstream.module_file(module) is not None

    seen: set[str] = set()
    stack = []
    for module in roots:
        if exists(module):
            stack.append(module)
        else:
            result.not_upstream.append(module)
    while stack:
        module = stack.pop()
        if module in seen:
            continue
        seen.add(module)
        source = upstream.show(upstream.module_file(module))
        edges = imports_of(source, module, exists)
        rel = vendored_path(module)
        if rel is not None:
            result.files[rel] = edges
        for edge in edges:
            target = vendored_path(edge.target)
            if target is not None and exists(edge.target):
                if prune and rel == PRUNE_ENTRY and target in PRUNE_MODULES:
                    result.cut.append(edge)
                    continue
                if rel is not None:
                    result.importers.setdefault(target, []).append((rel, edge))
                if target not in stop:
                    stack.append(edge.target)
            elif edge.target in COMPAT_MODULES and rel is not None:
                result.compat.update(_imported_names(source, edge))
            elif exists(edge.target) and rel is not None:
                result.external.add(edge.target)
    if prune and (survivors := sorted(set(PRUNE_MODULES) & set(result.files))):
        via = sorted({imp for m in survivors for imp, _ in result.importers.get(m, [])})
        raise ValueError(f"GDP modules {survivors} stay reachable after the prune cut via {via}")
    return result


def _imported_names(source: str, edge: Edge) -> set[str]:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.lineno == edge.lineno:
            return {f"{edge.target}.{a.name}" for a in node.names}
    return set()
