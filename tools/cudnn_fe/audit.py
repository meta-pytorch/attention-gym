"""Report upstream cudnn-frontend patterns that remain in the vendored ``cudnn_fe`` package.

The vendored kernels follow the Attention Gym house style (``cute.arch`` wrappers, register
``cute.make_rmem_tensor``, SMEM data in a ``SharedStorage`` struct, ``swizzle_box_offset_*``,
``@jit_cache`` fake-signature compiles, no upstream-only knobs). A fresh upstream drop uses the
upstream forms instead. This checker walks the Python AST (comments and docstrings are ignored)
and prints every remaining upstream form as ``path:line: rule [key] message``.

Names are resolved through the module's imports, so aliased imports (``from cutlass import
Array as A``, ``import cutlass.cute as c``, directly imported NVVM functions, fully qualified
chains) report like the canonical spelling.

Each finding has a *key* (usually the assigned variable or the enclosing function) so that
documented, SASS-driven exceptions can be listed in ``audit_allowlist.txt`` without line numbers:

    <rule> <path regex relative to cudnn_fe> <key regex> -- <reason>   (both regexes full-match)

Usage:
    python -m tools.cudnn_fe.audit                      # installed package + allowlist
    python -m tools.cudnn_fe.audit --rev 88eb5ce --no-allowlist --counts   # verbatim drop
    python -m tools.cudnn_fe.audit --files kernel/gdn_prefill_f16.py      # subset of --root
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tarfile
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from tools.cudnn_fe import closure

DEFAULT_ALLOWLIST = Path(__file__).resolve().with_name("audit_allowlist.txt")

RULES = {
    "raw-nvvm-wrapper": "raw nvvm call where the house style uses a SASS-identical cute.arch "
    "wrapper (mbarrier arrive/expect_tx/init, tcgen05.commit, bulk-group commit/wait, packed "
    "fp32 add)",
    "inline-ptx-packed-f32": "hand-written f32x2 inline PTX (cute.arch.*_packed_f32x2 exists)",
    "rmem-array": "register cutlass.Array; use cute.make_rmem_tensor",
    "smem-data-array": "SMEM data buffer allocated as cutlass.Array instead of a SharedStorage "
    "field (barrier/scheduler/TMEM-slot/sort-scratch storage is exempt)",
    "inline-swizzle-box": "inline segment-major swizzle arithmetic; use "
    "swizzle_box_offset_{128b,32b}",
    "compile-outside-jit-cache": "cute.compile / from_dlpack outside a module-level @jit_cache "
    "compile function",
    "upstream-ref": "reference to upstream-only cudnn.frost modules, engines, GDP or GDN2",
    "pruned-knob": "upstream-only constexpr knob that the vendored kernels pin and prune",
    "tma-helper-args": "acquire=/cta_group=/multicast/L2-hint argument to a TMA helper",
    "sched-arrival-literal": "literal scheduler mbarrier arrival count; derive it from the "
    "warp-role map",
}

NVVM_MODULES = {"cutlass.experimental.primitives", "cutlass._mlir.dialects.nvvm"}
WRAPPED_NVVM = {
    "mbarrier_arrive": "cute.arch.mbarrier_arrive",
    "mbarrier_arrive_expect_tx": "cute.arch.mbarrier_arrive_and_expect_tx",
    "mbarrier_init": "cute.arch.mbarrier_init",
    "tcgen05_commit": "cute.nvgpu.tcgen05.commit",
    "cp_async_bulk_commit_group": "cute.arch.cp_async_bulk_commit_group",
    "cp_async_bulk_wait_group": "cute.arch.cp_async_bulk_wait_group",
    "add_packed_f32x2": "cute.arch.add_packed_f32x2 (via tile_dsl fadd2)",
    "mul_packed_f32x2": "cute.arch.mul_packed_f32x2",
    "fma_packed_f32x2": "cute.arch.fma_packed_f32x2",
}
PACKED_PTX = re.compile(r"\b(mul|add|fma\.rn|fma)\.f32x2\b")
# SMEM cutlass.Array control storage (besides Int64 barriers) exempt from smem-data-array: the
# scheduler slots, TMEM base holders, token-slot and sort key/index/spread/cut scratch the v1.30
# kernels allocate. Any other name is reported; allowlist it with evidence.
CONTROL_SMEM_NAMES = frozenset(
    {
        "sCut",
        "sIdx",
        "sKey",
        "sScheduler",
        "sSpread",
        "sTmem_base",
        "tmem_base_holder",
        "tmem_base_slot",
        "tok_slot_raw",
    }
)
# Unimported conventional module names (fixtures and snippets): how the kernels bind them.
CONVENTIONAL = {"nvvm": "cutlass.experimental.primitives", "cute": "cutlass.cute"}
SWIZZLE_BYTES = {"swizzle_xor_128b": 128, "swizzle_xor_32b": 32}
BOX_HELPER = {
    "swizzle_xor_128b": "swizzle_box_offset_128b",
    "swizzle_xor_32b": "swizzle_box_offset_32b",
}
PRUNED_KNOBS = (
    "expand_num",
    "safe_gate",
    "beta_sigmoid",
    "allow_neg_eigval",
    "fused_l2norm",
    "compact_qdo",
    "own_prologue",
)
PRUNED_KNOB_RE = re.compile("|".join(PRUNED_KNOBS), re.IGNORECASE)
UPSTREAM_IDENT_RE = re.compile(r"(?i)(?:^|_)(gdp|gdn2|engines?)(?:_|$|\d)")
UPSTREAM_STRING_RE = re.compile(r"cudnn\.frost|linear_attention\.frost|cudnn\.linear_attention")
TMA_HELPERS = {"tma_load_tile", "tma_store_tile"}
TMA_DROPPED_KWARGS = {"acquire", "cta_group", "mcast_mask", "multicast_mask", "l2_cache_hint"}
MBARRIER_INIT_COUNT_POS = 2  # MBarrier(base_ptr, stages, init_count, ...)


@dataclass
class Finding:
    rule: str
    path: str  # posix path relative to the audited package root
    line: int
    key: str
    message: str
    node: ast.AST | None = field(default=None, repr=False, compare=False)
    detail: object = field(default=None, repr=False, compare=False)
    func: ast.AST | None = field(default=None, repr=False, compare=False)

    def format(self) -> str:
        return f"{self.path}:{self.line}: {self.rule} [{self.key}] {self.message}"


@dataclass
class AllowEntry:
    rule: str
    path_re: re.Pattern[str]
    key_re: re.Pattern[str]
    reason: str
    lineno: int
    hits: int = 0

    def matches(self, finding: Finding) -> bool:
        return (
            finding.rule == self.rule
            and self.path_re.fullmatch(finding.path) is not None
            and self.key_re.fullmatch(finding.key) is not None
        )


def dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for Name/Attribute chains, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def call_name(call: ast.Call) -> str | None:
    """Last component of the called name (``f`` for ``f(...)`` and ``m.f(...)``)."""
    func = call.func
    if isinstance(func, ast.Subscript):  # cute.compile[...](...)
        func = func.value
    name = dotted(func)
    return None if name is None else name.rsplit(".", 1)[-1]


def int_const(node: ast.AST | None) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    return None


def same(a: ast.AST, b: ast.AST) -> bool:
    return ast.dump(a) == ast.dump(b)


def keyword(call: ast.Call, name: str) -> ast.keyword | None:
    return next((kw for kw in call.keywords if kw.arg == name), None)


def flatten_add(node: ast.AST) -> list[ast.AST]:
    """Terms of a left-associated ``a + b + c`` chain."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return flatten_add(node.left) + [node.right]
    return [node]


def _mult_by(node: ast.AST, factor: int) -> ast.AST | None:
    """``x`` for ``x * factor`` / ``factor * x``."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult)):
        return None
    if int_const(node.right) == factor:
        return node.left
    if int_const(node.left) == factor:
        return node.right
    return None


def _floordiv_by(node: ast.AST, divisor: int) -> ast.AST | None:
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.FloorDiv)
        and int_const(node.right) == divisor
    ):
        return node.left
    return None


@dataclass
class BoxMatch:
    """A segment-major swizzle chain ``[extra +] seg * (rows * W) + row * W + swz(row, col)``."""

    chain: ast.AST
    helper: str
    elem_bytes: int
    width: int
    extra: list[ast.AST]  # terms kept in front of the helper call
    rows: ast.AST  # box_rows
    row: ast.AST
    col: ast.AST | None  # logical column when the chain is exact inline form
    seg_name: str | None  # segment variable for the ``seg = col // W`` form
    col_name: str | None  # in-box column variable for that form
    exact_row: bool
    seg_assign: ast.Assign | None = None  # ``seg = col // W`` (segment-variable form)
    col_assign: ast.Assign | None = None  # ``seg_col = col - seg * W`` or ``col % W``
    swz: ast.Call | None = None  # the swizzle_xor_* call

    @property
    def exact(self) -> bool:
        """The chain equals ``swizzle_box_offset_*`` for any column value."""
        return self.exact_row and self.col is not None


def match_swizzle_box(
    expr: ast.AST, name_of: Callable[[ast.Call], str | None] = call_name
) -> BoxMatch | None:
    """``name_of`` maps a call to its canonical function name (import aliases resolved)."""
    terms = flatten_add(expr)
    if len(terms) < 3:
        return None
    swz = next(
        (
            t
            for t in terms
            if isinstance(t, ast.Call) and name_of(t) in SWIZZLE_BYTES and len(t.args) == 2
        ),
        None,
    )
    if swz is None:
        return None
    eb_kw = keyword(swz, "elem_bytes")
    elem_bytes = 2 if eb_kw is None else int_const(eb_kw.value)
    if elem_bytes not in (1, 2, 4) or len(swz.keywords) > (eb_kw is not None):
        return None
    swz_name = name_of(swz)
    width = SWIZZLE_BYTES[swz_name] // elem_bytes
    row, col_arg = swz.args
    row_term = seg_term = rows = None
    seg_col = seg_name = None
    for t in terms:
        if t is swz:
            continue
        inner = _mult_by(t, width)
        if inner is not None and row_term is None and not isinstance(inner, ast.BinOp):
            row_term = t
            continue
        if isinstance(t, ast.BinOp) and isinstance(t.op, ast.Mult) and seg_term is None:
            for seg, stride in ((t.left, t.right), (t.right, t.left)):
                r = _mult_by(stride, width)
                if r is None:
                    continue
                c = _floordiv_by(seg, width)
                if c is not None or isinstance(seg, ast.Name):
                    seg_term, rows = t, r
                    seg_col = c
                    seg_name = seg.id if isinstance(seg, ast.Name) else None
                    break
    if row_term is None or seg_term is None:
        return None
    row_factor = _mult_by(row_term, width)
    exact_row = same(row_factor, row)
    col = col_name = None
    if seg_col is not None:
        mod = col_arg
        if (
            isinstance(mod, ast.BinOp)
            and isinstance(mod.op, ast.Mod)
            and int_const(mod.right) == width
            and same(mod.left, seg_col)
        ):
            col = seg_col
    elif isinstance(col_arg, ast.Name):
        col_name = col_arg.id
    extra = [t for t in terms if t is not swz and t is not row_term and t is not seg_term]
    return BoxMatch(
        chain=expr,
        helper=BOX_HELPER[swz_name],
        elem_bytes=elem_bytes,
        width=width,
        extra=extra,
        rows=rows,
        row=row,
        col=col,
        seg_name=seg_name,
        col_name=col_name,
        exact_row=exact_row,
        swz=swz,
    )


class _Auditor(ast.NodeVisitor):
    def __init__(self, source: str, relpath: str):
        self.source = source
        self.relpath = relpath
        self.tree = ast.parse(source)
        self.findings: list[Finding] = []
        self.parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                self.parents[child] = parent
        self.aliases = self._aliases()
        self.smem_aliases = {
            t.id
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Assign)
            for t in node.targets
            if isinstance(t, ast.Name) and _is_space(node.value, "smem")
        }
        self.func_stack: list[ast.AST] = []
        self.docstrings = {
            id(node.value)
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        }

    def _aliases(self) -> dict[str, str]:
        """Local name -> imported dotted name (relative imports keep their leading dots)."""
        aliases = {}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ImportFrom):
                base = "." * node.level + (node.module or "")
                sep = "." if node.module else ""
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f"{base}{sep}{alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        aliases[alias.asname] = alias.name
                    else:
                        head = alias.name.split(".")[0]
                        aliases[head] = head
        return aliases

    def qual(self, node: ast.AST) -> str | None:
        """Canonical dotted name of a Name/Attribute chain, resolving import aliases."""
        name = dotted(node)
        if name is None:
            return None
        head, _, rest = name.partition(".")
        base = self.aliases.get(head, CONVENTIONAL.get(head, head))
        return f"{base}.{rest}" if rest else base

    def called(self, call: ast.Call) -> str | None:
        """Canonical last component of the called function (``cute.compile[...]`` unwrapped)."""
        func = call.func.value if isinstance(call.func, ast.Subscript) else call.func
        name = self.qual(func)
        return None if name is None else name.rsplit(".", 1)[-1]

    # -- helpers -------------------------------------------------------------------------------
    def add(self, rule: str, node: ast.AST, key: str, message: str, detail=None) -> None:
        func = self.func_stack[-1] if self.func_stack else None
        self.findings.append(
            Finding(rule, self.relpath, node.lineno, key, message, node, detail, func)
        )

    def func_name(self) -> str:
        names = [f.name for f in self.func_stack if hasattr(f, "name")]
        return ".".join(names) or "<module>"

    def binding_key(self, node: ast.AST) -> str:
        """Assigned name or enclosing keyword for ``node``, else the enclosing function."""
        child, parent = node, self.parents.get(node)
        while parent is not None and not isinstance(parent, ast.stmt):
            if isinstance(parent, ast.keyword) and parent.arg:
                return parent.arg
            child, parent = parent, self.parents.get(parent)
        if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
            if parent.value is child or child in ast.walk(parent.value):
                text = dotted(targets[0])
                if text is None and isinstance(targets[0], ast.Tuple):
                    text = ",".join(dotted(e) or "?" for e in targets[0].elts)
                if text:
                    return text
        return self.func_name()

    def in_jit_cache(self) -> bool:
        if not self.func_stack:
            return False
        outer = self.func_stack[0]
        return isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            (self.qual(d.func if isinstance(d, ast.Call) else d) or "").rsplit(".", 1)[-1]
            == "jit_cache"
            for d in outer.decorator_list
        )

    # -- traversal ----------------------------------------------------------------------------
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_identifier(node, node.name)
        extra = [a for a in (node.args.vararg, node.args.kwarg) if a is not None]
        for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs, *extra]:
            self._check_identifier(arg, arg.arg, key=f"{node.name}({arg.arg})")
            if node.name in TMA_HELPERS and arg.arg in TMA_DROPPED_KWARGS:
                self.add(
                    "tma-helper-args",
                    arg,
                    f"{node.name}.{arg.arg}",
                    f"{node.name} still declares {arg.arg}=",
                )
        self.func_stack.append(node)
        self.generic_visit(node)
        self.func_stack.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._check_identifier(node, node.name)
        self.func_stack.append(node)
        self.generic_visit(node)
        self.func_stack.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.func_stack.append(node)
        self.generic_visit(node)
        self.func_stack.pop()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module(node, alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self._check_module(node, "." * node.level + node.module)
        for alias in node.names:
            self._check_identifier(node, alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        self._check_identifier(node, node.id)
        if isinstance(node.ctx, ast.Load):
            self._check_nvvm(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._check_identifier(node, node.attr)
        self._check_nvvm(node)
        self.generic_visit(node)

    def _check_nvvm(self, node: ast.Name | ast.Attribute) -> None:
        """A wrapped NVVM function, however it was imported; ``detail`` is its NVVM name."""
        module, _, attr = (self.qual(node) or "").rpartition(".")
        if module not in NVVM_MODULES or attr not in WRAPPED_NVVM:
            return
        parent = self.parents.get(node)
        called = isinstance(parent, ast.Call) and parent.func is node
        self.add(
            "raw-nvvm-wrapper",
            parent if called else node,
            f"{attr}@{self.func_name()}",
            f"{dotted(node)} -> {WRAPPED_NVVM[attr]}",
            detail=attr,
        )

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg:
            self._check_identifier(node.value, node.arg, key=node.arg, lineno_node=node.value)
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, str) or id(node) in self.docstrings:
            return
        match = UPSTREAM_STRING_RE.search(node.value)
        if match:
            self.add("upstream-ref", node, match.group(0), f"string names {match.group(0)}")
        match = PRUNED_KNOB_RE.search(node.value)
        if match and "_" in node.value and " " not in node.value:
            self.add("pruned-knob", node, match.group(0).lower(), f"string {node.value!r}")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        parent = self.parents.get(node)
        is_chain_top = not (
            isinstance(parent, ast.BinOp)
            and isinstance(parent.op, ast.Add)
            and parent.left is node
        )
        if is_chain_top and isinstance(node.op, ast.Add):
            match = match_swizzle_box(node, self.called)
            if match is not None and match.seg_name and match.col_name:
                self._resolve_segment_vars(match)
            if match is not None and match.exact:
                self.add(
                    "inline-swizzle-box",
                    node,
                    self.binding_key(node),
                    f"segment-major swizzle arithmetic; use {match.helper}",
                    detail=match,
                )
        self.generic_visit(node)

    def _resolve_segment_vars(self, match: BoxMatch) -> None:
        """Accept ``seg = c // W; col = c - seg * W`` (or ``c % W``) defined before the chain."""
        scope = self.func_stack[-1] if self.func_stack else self.tree
        line = match.chain.lineno

        def last_assign(name: str) -> ast.Assign | None:
            candidates = [
                n
                for n in ast.walk(scope)
                if isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == name
                and n.lineno < line
            ]
            return max(candidates, key=lambda n: n.lineno, default=None)

        seg_assign = last_assign(match.seg_name)
        col_assign = last_assign(match.col_name)
        if seg_assign is None or col_assign is None:
            return
        col = _floordiv_by(seg_assign.value, match.width)
        value = col_assign.value
        if col is None or not isinstance(value, ast.BinOp):
            return
        if isinstance(value.op, ast.Mod):
            ok = int_const(value.right) == match.width and same(value.left, col)
        elif isinstance(value.op, ast.Sub):
            seg_times_w = _mult_by(value.right, match.width)
            ok = (
                same(value.left, col)
                and isinstance(seg_times_w, ast.Name)
                and seg_times_w.id == match.seg_name
            )
        else:
            ok = False
        if ok:
            match.col, match.seg_assign, match.col_assign = col, seg_assign, col_assign

    def visit_Call(self, node: ast.Call) -> None:
        name = self.called(node)
        func = self.qual(node.func.value if isinstance(node.func, ast.Subscript) else node.func)
        if name == "inline_ptx":
            text = "".join(
                c.value
                for arg in [*node.args, *(kw.value for kw in node.keywords)]
                for c in ast.walk(arg)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)
            )
            if PACKED_PTX.search(text):
                self.add("inline-ptx-packed-f32", node, self.func_name(), "f32x2 op as inline PTX")
        if func == "cutlass.Array" or (func or "").endswith(".cutlass.Array"):
            self._check_array(node)
        if (func == "cutlass.cute.compile" or name == "from_dlpack") and not self.in_jit_cache():
            self.add(
                "compile-outside-jit-cache",
                node,
                f"{name}@{self.func_name()}",
                f"{func or name} outside a module-level @jit_cache function",
            )
        if name in TMA_HELPERS:
            for kw in node.keywords:
                if kw.arg in TMA_DROPPED_KWARGS:
                    self.add(
                        "tma-helper-args",
                        kw.value,
                        f"{name}.{kw.arg}",
                        f"{name}(..., {kw.arg}=...)",
                        detail=(node, kw),
                    )
        if name == "MBarrier":
            count = keyword(node, "init_count")
            count = count.value if count is not None else None
            if count is None and len(node.args) > MBARRIER_INIT_COUNT_POS:
                count = node.args[MBARRIER_INIT_COUNT_POS]
            key = self.binding_key(node)
            if count is not None and "sched" in key.lower():
                value = int_const(count)
                if value is not None and value != 1:
                    self.add("sched-arrival-literal", count, key, f"init_count={value} literal")
        self.generic_visit(node)

    def _check_array(self, node: ast.Call) -> None:
        space = keyword(node, "space")
        is_smem = space is not None and (
            _is_space(space.value, "smem")
            or (isinstance(space.value, ast.Name) and space.value.id in self.smem_aliases)
        )
        key = self.binding_key(node)
        if not is_smem:
            if space is None or _is_space(space.value, "rmem"):
                self.add("rmem-array", node, key, "register cutlass.Array", detail="rmem")
            return
        dtype = dotted(node.args[0]) if node.args else None
        if dtype in ("cutlass.Int64", "Int64") or key in CONTROL_SMEM_NAMES:
            return
        self.add("smem-data-array", node, key, "SMEM cutlass.Array data buffer")

    def _check_module(self, node: ast.AST, module: str) -> None:
        if module == "cudnn" or module.startswith("cudnn."):
            self.add("upstream-ref", node, module, f"imports upstream module {module}")
        elif any(UPSTREAM_IDENT_RE.search(part) for part in module.split(".")):
            self.add("upstream-ref", node, module, f"imports upstream-only module {module}")

    def _check_identifier(
        self, node: ast.AST, ident: str, key: str | None = None, lineno_node=None
    ) -> None:
        target = lineno_node or node
        if not hasattr(target, "lineno"):
            return
        match = PRUNED_KNOB_RE.search(ident)
        if match:
            self.add("pruned-knob", target, key or ident, f"identifier {ident}")
        match = UPSTREAM_IDENT_RE.search(ident)
        if match:
            self.add("upstream-ref", target, key or ident, f"identifier {ident}")


def _is_space(node: ast.AST, space: str) -> bool:
    name = dotted(node)
    return name is not None and name.endswith(f"AddressSpace.{space}")


def analyze(source: str, relpath: str) -> _Auditor:
    """Parse and audit one module; the result keeps the tree and parent map for rewriting."""
    auditor = _Auditor(source, relpath)
    auditor.visit(auditor.tree)
    auditor.findings = _dedupe(auditor.findings)
    return auditor


def _dedupe(findings: list[Finding]) -> list[Finding]:
    # An identifier repeated on one line (``cfg.safe_gate`` as Name and keyword) reports once.
    seen, unique = set(), []
    for f in findings:
        ident = (f.rule, f.line, f.key, f.message)
        if ident not in seen:
            seen.add(ident)
            unique.append(f)
    return sorted(unique, key=lambda f: (f.line, f.rule))


def audit_source(source: str, relpath: str) -> list[Finding]:
    return analyze(source, relpath).findings


def default_root() -> Path:
    """The installed ``cudnn_fe`` package."""
    return closure.package_dir()


def iter_sources(root: Path, files: Iterable[str] | None = None) -> Iterator[Path]:
    """Python sources under ``root``, or only ``files`` (paths relative to ``root``)."""
    if files is not None:
        for rel in files:
            path = root / rel
            if not path.is_file():
                raise FileNotFoundError(f"{rel} is not a file under {root}")
            yield path
        return
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path


def audit_tree(root: Path, files: Iterable[str] | None = None) -> list[Finding]:
    findings = []
    for path in iter_sources(root, files):
        findings += audit_source(path.read_text(), path.relative_to(root).as_posix())
    return findings


def load_allowlist(path: Path) -> list[AllowEntry]:
    entries = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        spec, sep, reason = line.partition(" -- ")
        fields = spec.split()
        if not sep or not reason.strip() or len(fields) != 3:
            raise ValueError(f"{path}:{lineno}: expected '<rule> <path-re> <key-re> -- <reason>'")
        rule, path_re, key = fields
        if rule not in RULES:
            raise ValueError(f"{path}:{lineno}: unknown rule {rule!r}")
        if re.search(r"(?<!\\)\.[*+]", key):
            raise ValueError(f"{path}:{lineno}: list the evidenced keys instead of {key!r}")
        entries.append(
            AllowEntry(rule, re.compile(path_re), re.compile(key), reason.strip(), lineno)
        )
    return entries


def apply_allowlist(
    findings: Iterable[Finding], entries: list[AllowEntry]
) -> tuple[list[Finding], list[Finding]]:
    """Split findings into (remaining, allowed); counts hits on each entry."""
    remaining, allowed = [], []
    for finding in findings:
        entry = next((e for e in entries if e.matches(finding)), None)
        if entry is None:
            remaining.append(finding)
        else:
            entry.hits += 1
            allowed.append(finding)
    return remaining, allowed


def extract_rev(rev: str, dest: Path) -> Path:
    """Export the package at ``rev`` of the checkout into ``dest`` and return its root."""
    blob = subprocess.run(
        ["git", "-C", str(closure.repo_root()), "archive", "--format=tar", rev]
        + [closure.PACKAGE_RELPATH],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(blob)) as tar:
        tar.extractall(dest, filter="data")
    return dest / closure.PACKAGE_RELPATH


def rule_counts(findings: Iterable[Finding]) -> Counter[str]:
    counts = Counter(f.rule for f in findings)
    return Counter({rule: counts[rule] for rule in RULES})


def format_counts(counts: Counter[str]) -> str:
    width = max(map(len, RULES))
    lines = [f"  {rule:<{width}}  {counts[rule]:>5}" for rule in RULES]
    lines.append(f"  {'total':<{width}}  {sum(counts.values()):>5}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, help="cudnn_fe package dir (default: installed)")
    parser.add_argument("--rev", help="audit the package at this git revision instead of --root")
    parser.add_argument(
        "--files", nargs="+", metavar="REL", help="audit only these paths relative to the root"
    )
    parser.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--no-allowlist", action="store_true")
    parser.add_argument("--counts", action="store_true", help="print per-rule counts only")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail on allowlist entries that match nothing (whole-tree runs only)",
    )
    args = parser.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        root = extract_rev(args.rev, Path(tmp)) if args.rev else args.root or default_root()
        findings = audit_tree(root, args.files)
    entries = [] if args.no_allowlist else load_allowlist(args.allowlist)
    remaining, allowed = apply_allowlist(findings, entries)

    if not args.counts:
        for finding in remaining:
            print(finding.format())
    label = args.rev or str(root)
    print(f"audit {label}: {len(remaining)} findings, {len(allowed)} allowlisted")
    print(format_counts(rule_counts(remaining)))
    unused = [e for e in entries if e.hits == 0] if args.files is None else []
    for entry in unused:
        print(f"warning: {args.allowlist.name}:{entry.lineno}: allowlist entry matched nothing")
    if not remaining:
        print("CLEAN")
    return int(bool(remaining) or (args.strict and bool(unused)))


if __name__ == "__main__":
    sys.exit(main())
