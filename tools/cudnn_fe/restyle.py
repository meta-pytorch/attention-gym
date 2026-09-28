"""Apply the mechanical Attention Gym restyle rewrites to a vendored cudnn-frontend drop.

Only rewrites that were SASS-identical in the v1.30 port and can be proven from the AST:

- ``tma-helper-args``: drop an ``acquire=False``, ``cta_group=1``, ``mcast_mask=None``,
  ``multicast_mask=None`` or ``l2_cache_hint=None`` argument at a ``tma_load_tile`` /
  ``tma_store_tile`` call, and only once the target tree's ``tile_dsl/tma.py`` helper no longer
  declares that parameter. The restyled helpers never fence, use one CTA and no multicast or L2
  hint, so those are the only values that keep the call's behavior; any other value (including
  ``acquire=True``) or an unrestyled helper (whose ``acquire`` default is True) stays manual.
- ``raw-nvvm-wrapper``: ``nvvm.mbarrier_arrive/_expect_tx/_init`` -> ``cute.arch`` on
  ``.data_ptr()``; ``nvvm.tcgen05_commit(mb, group=CTA_1)`` -> ``tcgen05.commit``; bulk-group
  commit/wait -> ``cute.arch``; the v1.30 packed-add body of ``fadd2`` (matched statement for
  statement) -> ``cute.arch.add_packed_f32x2``; direct kernel ``nvvm.add_packed_f32x2`` pairs ->
  ``fadd2`` when the tree's ``fadd2`` is one of those two packed-add bodies.
- ``rmem-array``: register ``cutlass.Array(T, N, ...)`` -> ``cute.make_rmem_tensor((N,), T)``
  and full-range slices ``arr[0:N]`` -> ``arr.load()``.
- ``inline-swizzle-box``: exact segment-major swizzle arithmetic -> ``swizzle_box_offset_*``,
  when the tree's ``tile_dsl/swizzle.py`` defines that helper in the house form and, for the
  ``seg = c // W; col = c % W`` form, both bindings sit in the chain's block with only simple
  statements between them and no writes to ``seg``, ``col`` or ``c``.

A rewrite is emitted only if the helper it calls exists in the target tree and the file already
imports that helper module; otherwise it stays pending. Everything else the audit reports
(SharedStorage placement, ``smem_data_ptr``/``SmemTile``, Op classes and ``@jit_cache`` compiles,
knob and dead-code pruning, value-range-dependent swizzle rewrites) is a judgment call and stays
manual; ``--check`` lists it. Entries in ``audit_allowlist.txt`` are skipped unless
``--no-allowlist`` is given. The rewrite is idempotent: a second run finds nothing to change.

Usage:
    python -m tools.cudnn_fe.restyle --check                 # installed package, exit 1 if dirty
    python -m tools.cudnn_fe.restyle --write --root <drop>/attn_gym/linear/_delta_rule/cudnn_fe
    python -m tools.cudnn_fe.restyle --check --files kernel/gdn_prefill_f16.py   # subset of --root
"""

from __future__ import annotations

import argparse
import ast
import difflib
import io
import re
import sys
import tokenize
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from tools.cudnn_fe import audit

MECHANICAL_RULES = ("tma-helper-args", "raw-nvvm-wrapper", "rmem-array", "inline-swizzle-box")
MANUAL_STEPS = (
    (
        "SMEM data buffers -> host-defined @cute.struct SharedStorage + SmemAllocator, keeping "
        "the v1.30 order of barrier/scheduler arrays (placement is SASS/perf-sensitive)"
    ),
    (
        "raw SMEM pointers -> smem_data_ptr(...) and SmemTile stage slices on tensor bases (only "
        "valid once the buffer is a struct tensor)"
    ),
    (
        "value-range swizzle rewrites (row * W + swizzle_xor(row, col) with col < W known) and "
        "row-XOR-segment gate layouts"
    ),
    (
        "frozen cfg + <K>Op classes + module-level @jit_cache compiles over fake signatures "
        "(replaces cute.compile/from_dlpack)"
    ),
    "census-driven prune of unused tile_dsl helpers, upstream-only knobs, GDP/engine paths",
    (
        "restyle the tile_dsl helpers the pending rewrites need (tma acquire/cta_group/multicast "
        "parameters, swizzle_box_offset_*), then rerun the codemod"
    ),
    "scheduler arrival counts derived from the warp-role map; host launch-contract validation",
    "per-file 'Modified by Attention Gym' headers, NOTICE/README refresh",
)
HELPER_MODULES = ("swizzle", "pointwise", "tma")
# Argument values the restyled TMA helpers hard-code (no fence, one CTA, no multicast/L2 hint).
TMA_HOUSE_VALUES = {
    "acquire": False,
    "cta_group": 1,
    "mcast_mask": None,
    "multicast_mask": None,
    "l2_cache_hint": None,
}
FADD2_PARAMS = ["a_lo", "a_hi", "b_lo", "b_hi"]
FADD2_HOUSE = "return cute.arch.add_packed_f32x2((a_lo, a_hi), (b_lo, b_hi))\n"
FADD2_V130 = """\
vec_a = cutlass.Vector.from_elements((a_lo, a_hi), cutlass.Float32)
vec_b = cutlass.Vector.from_elements((b_lo, b_hi), cutlass.Float32)
res = cutlass.Vector(_packed_f32x2(nvvm_ops.add_packed_f32x2, vec_a.ir_value(), vec_b.ir_value()), dtype=cutlass.Float32)
return cutlass.Float32(res[0]), cutlass.Float32(res[1])
"""
PACKED_F32X2_V130 = """\
def _packed_f32x2(op, vec_a, vec_b):
    if _PACKED_RES_FIRST:
        return op(vec_a.type, vec_a, vec_b, rnd=nvvm_ops.FPRoundingMode.RN)
    return op(vec_a, vec_b, rnd=nvvm_ops.FPRoundingMode.RN)
"""
SWIZZLE_BOX_HOUSE = """\
def {helper}(row, col, *, box_rows: cutlass.Constexpr[int], elem_bytes: cutlass.Constexpr[int] = 2):
    box_cols = cutlass.const_expr({nbytes} // elem_bytes)
    box = col // box_cols
    col_in_box = col - box * box_cols
    return box * box_rows * box_cols + row * box_cols + {xor}(row, col_in_box, elem_bytes=elem_bytes)
"""
SIMPLE_STMTS = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Pass)


def _body(func: ast.FunctionDef) -> list[ast.stmt]:
    return func.body[1:] if func.body and audit_is_docstring(func.body[0]) else func.body


def _dump(stmts) -> str:
    return "".join(ast.dump(stmt) for stmt in stmts)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    return next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)


def _same_function(func: ast.FunctionDef | None, template: str) -> bool:
    """Same arguments and statements as ``template`` (decorators and docstring ignored)."""
    ref = ast.parse(template).body[0]
    return (
        func is not None
        and ast.dump(func.args) == ast.dump(ref.args)
        and _dump(_body(func)) == _dump(ref.body)
    )


def fadd2_form(info) -> str | None:
    """``"house"``/``"v130"`` if the module's ``fadd2`` is exactly a packed lane-wise fp32 add."""
    func = _function(info.tree, "fadd2")
    if func is None or ast.dump(func.args) != ast.dump(
        ast.parse(f"def f({', '.join(FADD2_PARAMS)}): pass").body[0].args
    ):
        return None
    body = _dump(_body(func))
    if body == _dump(ast.parse(FADD2_HOUSE).body) and info.qual(_name("cute")) == "cutlass.cute":
        return "house"
    if (
        body == _dump(ast.parse(FADD2_V130).body)
        and info.qual(_name("nvvm_ops")) == "cutlass._mlir.dialects.nvvm"
        and _same_function(_function(info.tree, "_packed_f32x2"), PACKED_F32X2_V130)
    ):
        return "v130"
    return None


def box_helper_ok(info, helper: str) -> bool:
    """The swizzle module defines ``helper`` exactly as the house segment-major form."""
    nbytes = 128 if helper.endswith("128b") else 32
    template = SWIZZLE_BOX_HOUSE.format(helper=helper, nbytes=nbytes, xor=f"swizzle_xor_{nbytes}b")
    return _same_function(_function(info.tree, helper), template)


def _name(ident: str) -> ast.Name:
    return ast.Name(ident, ast.Load())


def tile_helpers(root: Path) -> dict:
    """Analyzed ``tile_dsl`` helper modules of the target tree, keyed by module name."""
    helpers = {}
    for name in HELPER_MODULES:
        path = root / "tile_dsl" / f"{name}.py"
        if path.is_file():
            helpers[name] = audit.analyze(path.read_text(), f"tile_dsl/{name}.py")
    return helpers


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    text: str
    rule: str


class Source:
    """Character offsets for AST (line, UTF-8 byte column) positions."""

    def __init__(self, text: str):
        self.text = text
        self.lines = text.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))

    def offset(self, lineno: int, col: int) -> int:
        line = self.lines[lineno - 1] if lineno <= len(self.lines) else ""
        return self.starts[lineno - 1] + len(line.encode()[:col].decode())

    def span(self, node: ast.AST) -> tuple[int, int]:
        return (
            self.offset(node.lineno, node.col_offset),
            self.offset(node.end_lineno, node.end_col_offset),
        )

    def seg(self, node: ast.AST) -> str:
        start, end = self.span(node)
        return self.text[start:end]

    def line_span(self, node: ast.AST) -> tuple[int, int] | None:
        """Whole physical lines of a statement, if nothing else shares them."""
        start, end = self.span(node)
        line_start = self.starts[node.lineno - 1]
        line_end = self.starts[node.end_lineno]
        if self.text[line_start:start].strip() or self.text[end:line_end].strip():
            return None
        return line_start, line_end


def call_arg_texts(src: Source, call: ast.Call) -> list[str]:
    """Raw text of each top-level argument, preserving redundant parentheses."""
    text = src.seg(call)
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    depth, args, current, started = 0, [], [], False
    base_line = tokens[0].start[0] if tokens else 1
    lines = text.splitlines(keepends=True)

    def pos(tok_pos):
        row, col = tok_pos
        return sum(len(line) for line in lines[: row - base_line]) + col

    arg_start = None
    for tok in tokens:
        if tok.type == tokenize.OP and tok.string in "([{":
            depth += 1
            if depth == 1 and not started and tok.string == "(":
                started = True
                arg_start = pos(tok.end)
                continue
        elif tok.type == tokenize.OP and tok.string in ")]}":
            depth -= 1
            if depth == 0 and started:
                current = text[arg_start : pos(tok.start)].strip()
                if current:
                    args.append(current)
                break
        elif tok.type == tokenize.OP and tok.string == "," and depth == 1 and started:
            args.append(text[arg_start : pos(tok.start)].strip())
            arg_start = pos(tok.end)
    return args


class _Rewriter:
    def __init__(self, text: str, relpath: str, entries: list[audit.AllowEntry], helpers: dict):
        self.src = Source(text)
        self.relpath = relpath
        self.info = audit.analyze(text, relpath)
        self.entries = entries
        self.helpers = helpers
        self.has_cute = self.info.qual(_name("cute")) == "cutlass.cute"
        self.edits: list[Edit] = []
        self.imports: dict[str, set[str]] = defaultdict(set)  # module suffix -> names
        self.new_import_lines: set[str] = set()
        self.manual: list[audit.Finding] = []
        self.applied: Counter[str] = Counter()
        self.bound = self._bound_names()

    def _bound_names(self) -> set[str]:
        names = set()
        for node in ast.walk(self.info.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    names.add((alias.asname or alias.name).split(".")[0])
        return names

    def parent(self, node: ast.AST) -> ast.AST | None:
        return self.info.parents.get(node)

    def stmt_of(self, node: ast.AST) -> ast.stmt | None:
        while node is not None and not isinstance(node, ast.stmt):
            node = self.parent(node)
        return node

    def run(self) -> list[Edit]:
        findings, _ = audit.apply_allowlist(self.info.findings, self.entries)
        swizzle_chains = {id(f.detail.chain) for f in findings if f.rule == "inline-swizzle-box"}
        for finding in findings:
            if finding.rule not in MECHANICAL_RULES:
                self.manual.append(finding)
                continue
            handler = getattr(self, "fix_" + finding.rule.replace("-", "_"))
            edits = handler(finding, swizzle_chains)
            if edits is None:
                self.manual.append(finding)
            else:
                self.edits += edits
                self.applied[finding.rule] += 1
        self.edits += self._import_edits()
        return self.edits

    # -- rules ---------------------------------------------------------------------------------
    def fix_tma_helper_args(self, finding, _chains):
        if not isinstance(finding.detail, tuple):
            return None  # a helper definition: prune by hand
        call, kw = finding.detail
        name = self.info.called(call)
        value = kw.value
        if not (
            isinstance(value, ast.Constant)
            and type(value.value) is type(TMA_HOUSE_VALUES[kw.arg])
            and value.value == TMA_HOUSE_VALUES[kw.arg]
        ):
            return None
        tma = self._helper_module("tma")
        func = None if tma is None else _function(tma.tree, name)
        if func is None or kw.arg in {
            a.arg for a in ast.walk(func.args) if isinstance(a, ast.arg)
        }:
            return None  # the helper still takes the argument: dropping it changes behavior
        if not self._resolves_to(call.func, "tma", name):
            return None
        start, end = self.src.span(kw)
        text = self.src.text
        before = text[:start].rstrip()
        if before.endswith(","):
            return [Edit(len(before) - 1, end, "", finding.rule)]
        after = re.match(r"\s*,\s*", text[end:])
        return [Edit(start, end + (after.end() if after else 0), "", finding.rule)]

    def fix_raw_nvvm_wrapper(self, finding, _chains):
        node = finding.node
        if not isinstance(node, ast.Call):
            return self._fix_fadd2_body(finding)
        attr = finding.detail
        args = call_arg_texts(self.src, node)
        pos, kws = node.args, node.keywords
        if attr == "add_packed_f32x2":
            return self._fix_add_pair(finding)
        if attr in ("mbarrier_arrive",) and len(pos) == 1 and not kws:
            new = f"cute.arch.mbarrier_arrive({args[0]}.data_ptr())"
        elif attr == "mbarrier_arrive_expect_tx" and len(pos) == 2 and not kws:
            new = f"cute.arch.mbarrier_arrive_and_expect_tx({args[0]}.data_ptr(), {args[1]})"
        elif attr == "mbarrier_init" and len(pos) == 2 and not kws:
            new = f"cute.arch.mbarrier_init({args[0]}.data_ptr(), {args[1]})"
        elif attr == "tcgen05_commit" and len(pos) == 1 and len(kws) == 1:
            group = audit.dotted(kws[0].value) or ""
            if kws[0].arg != "group" or not group.endswith("CTA_1"):
                return None
            new = f"tcgen05.commit({args[0]}.data_ptr())"
            if "tcgen05" not in self.bound:
                self.new_import_lines.add("from cutlass.cute.nvgpu import tcgen05")
        elif attr in ("cp_async_bulk_commit_group", "cp_async_bulk_wait_group"):
            new = f"cute.arch.{attr}({', '.join(args)})"
        else:
            return None
        if not self.has_cute:
            return None
        start, end = self.src.span(node)
        return [Edit(start, end, new, finding.rule)]

    def _fix_fadd2_body(self, finding):
        """The v1.30 ``fadd2`` body, matched exactly, -> the ``cute.arch`` packed add."""
        func = finding.func
        if (
            func is None
            or func not in self.info.tree.body
            or getattr(func, "name", None) != "fadd2"
            or not self.has_cute
            or fadd2_form(self.info) != "v130"
        ):
            return None
        body = _body(func)
        start = self.src.starts[body[0].lineno - 1]
        end = self.src.starts[body[-1].end_lineno]
        indent = re.match(r"\s*", self.src.lines[body[0].lineno - 1]).group(0)
        return [Edit(start, end, indent + FADD2_HOUSE, finding.rule)]

    def _fix_add_pair(self, finding):
        """``v = nvvm.add_packed_f32x2(Vec((a, b)), Vec((c, d)), ...); x, y = F32(v[0]), F32(v[1])``."""
        call = finding.node
        stmt = self.stmt_of(call)
        if not (
            isinstance(stmt, ast.Assign)
            and stmt.value is call
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and len(call.args) == 2
        ):
            return None
        pairs = []
        for arg in call.args:
            if not (
                isinstance(arg, ast.Call)
                and audit.call_name(arg) == "from_elements"
                and arg.args
                and isinstance(arg.args[0], ast.Tuple)
                and len(arg.args[0].elts) == 2
            ):
                return None
            pairs.append([self.src.seg(e) for e in arg.args[0].elts])
        kws = {kw.arg: kw.value for kw in call.keywords}
        if set(kws) - {"ftz", "rnd"} or (
            "ftz" in kws
            and not (isinstance(kws["ftz"], ast.Constant) and kws["ftz"].value is False)
        ):
            return None
        if "rnd" in kws and not (
            isinstance(kws["rnd"], ast.Constant) and kws["rnd"].value == "rn"
        ):
            return None
        body = self._body_of(stmt)
        if body is None or body.index(stmt) + 1 >= len(body):
            return None
        nxt = body[body.index(stmt) + 1]
        vec = stmt.targets[0].id
        expected = [f"cutlass.Float32({vec}[0])", f"cutlass.Float32({vec}[1])"]
        if not (
            isinstance(nxt, ast.Assign)
            and isinstance(nxt.value, ast.Tuple)
            and [ast.unparse(e) for e in nxt.value.elts] == expected
            and isinstance(nxt.targets[0], ast.Tuple)
        ):
            return None
        if self._loads_after(vec, finding.func, nxt, exclude=nxt):
            return None
        pointwise = self._helper_module("pointwise")
        if pointwise is None or fadd2_form(pointwise) is None:
            return None
        if not self._need("pointwise", "fadd2"):
            return None
        targets = self.src.seg(nxt.targets[0])
        args = ", ".join(pairs[0] + pairs[1])
        start = self.src.offset(stmt.lineno, stmt.col_offset)
        _, end = self.src.span(nxt)
        return [Edit(start, end, f"{targets} = fadd2({args})", finding.rule)]

    def fix_rmem_array(self, finding, _chains):
        call = finding.node
        stmt = self.stmt_of(call)
        kws = {kw.arg for kw in call.keywords}
        if (
            isinstance(stmt, ast.Expr)
            and stmt.value is call
            and len(call.args) == 2
            and not kws - {"alignment", "space"}
            and self.has_cute
        ):  # an unused (bare) register array
            dtype, size = call_arg_texts(self.src, call)[:2]
            start, end = self.src.span(call)
            return [Edit(start, end, f"cute.make_rmem_tensor(({size},), {dtype})", finding.rule)]
        if (
            not isinstance(stmt, ast.Assign)
            or stmt.value is not call
            or len(stmt.targets) != 1
            or not isinstance(stmt.targets[0], ast.Name)
            or len(call.args) != 2
            or kws - {"alignment", "space"}
            or not self.has_cute
        ):
            return None
        name = stmt.targets[0].id
        dtype, size = call_arg_texts(self.src, call)[:2]
        edits = []
        for use in self._uses_in_lifetime(name, finding.func, stmt):
            parent = self.parent(use)
            if isinstance(parent, ast.Attribute):
                load = self._full_pointer_load(parent, call.args[1])
                if load is None:
                    return None  # other pointer use: not a mechanical rewrite
                start, end = self.src.span(load)
                edits.append(Edit(start, end, f"{name}.load()", finding.rule))
                continue
            if isinstance(parent, ast.Subscript) and parent.value is use:
                sl = parent.slice
                if isinstance(sl, ast.Slice):
                    full = (
                        sl.step is None
                        and (sl.lower is None or audit.int_const(sl.lower) == 0)
                        and sl.upper is not None
                        and _same_value(sl.upper, call.args[1])
                    )
                    if not full:
                        return None
                    start, end = self.src.span(parent)
                    edits.append(Edit(start, end, f"{name}.load()", finding.rule))
        start, end = self.src.span(call)
        edits.append(Edit(start, end, f"cute.make_rmem_tensor(({size},), {dtype})", finding.rule))
        return edits

    def _full_pointer_load(self, attr: ast.Attribute, size: ast.AST) -> ast.Call | None:
        """``arr.data_ptr().load(count=N[, alignment=a])`` reading the whole array."""
        ptr_call = self.parent(attr)
        load_attr = self.parent(ptr_call)
        load = self.parent(load_attr)
        if not (
            attr.attr == "data_ptr"
            and isinstance(ptr_call, ast.Call)
            and not ptr_call.args
            and not ptr_call.keywords
            and isinstance(load_attr, ast.Attribute)
            and load_attr.attr == "load"
            and isinstance(load, ast.Call)
            and load.func is load_attr
            and not load.args
        ):
            return None
        kws = {kw.arg: kw.value for kw in load.keywords}
        if set(kws) - {"count", "alignment"} or "count" not in kws:
            return None
        return load if _same_value(kws["count"], size) else None

    def fix_inline_swizzle_box(self, finding, chains):
        match: audit.BoxMatch = finding.detail
        swizzle = self._helper_module("swizzle")
        if (
            swizzle is None
            or not box_helper_ok(swizzle, match.helper)
            or not self._resolves_to(match.swz.func, "swizzle", self.info.called(match.swz))
            or (match.seg_assign is not None and not self._straight_line(match))
            or not self._need("swizzle", match.helper)
        ):
            return None
        src = self.src
        col_text = src.seg(match.col)
        call = f"{match.helper}({src.seg(match.row)}, {col_text}, box_rows={src.seg(match.rows)}"
        call += ")" if match.elem_bytes == 2 else f", elem_bytes={match.elem_bytes})"
        expr = " + ".join([src.seg(t) for t in match.extra] + [call])
        stmt = self.stmt_of(match.chain)
        if isinstance(stmt, ast.Assign) and stmt.value is match.chain and len(stmt.targets) == 1:
            start = src.offset(stmt.targets[0].end_lineno, stmt.targets[0].end_col_offset)
            _, end = src.span(stmt)
            edits = [Edit(start, end, f" = {expr}", finding.rule)]
        else:
            start, end = src.span(match.chain)
            edits = [Edit(start, end, expr, finding.rule)]
        # Delete the in-box column first; its own load of ``seg`` then no longer counts.
        deleted = None
        for assign in (match.col_assign, match.seg_assign):
            if assign is None:
                continue
            var = assign.targets[0].id
            span = src.line_span(assign)
            if span and not self._loads_after(
                var, finding.func, assign, exclude=deleted, skip_ids=chains
            ):
                edits.append(Edit(*span, "", finding.rule))
                deleted = assign
        return edits

    # -- helpers -------------------------------------------------------------------------------
    def _helper_module(self, suffix: str):
        """The target tree's ``tile_dsl/<suffix>.py`` (this file's own text when it is that)."""
        if self.relpath == f"tile_dsl/{suffix}.py":
            return self.info
        return self.helpers.get(suffix)

    def _resolves_to(self, func: ast.AST, suffix: str, name: str | None) -> bool:
        """``func`` names ``tile_dsl/<suffix>.py``'s ``name`` through this module's imports."""
        if self.relpath == f"tile_dsl/{suffix}.py":
            return isinstance(func, ast.Name) and func.id == name
        qual = self.info.qual(func) or ""
        return qual.startswith(".") and qual.endswith(f"{suffix}.{name}")

    def _import_node(self, suffix: str) -> ast.ImportFrom | None:
        return next(
            (
                n
                for n in self.info.tree.body
                if isinstance(n, ast.ImportFrom)
                and n.level
                and n.module
                and n.module.rsplit(".", 1)[-1] == suffix
            ),
            None,
        )

    def _need(self, suffix: str, name: str) -> bool:
        """Make ``tile_dsl/<suffix>.py``'s ``name`` callable here; False if that is manual."""
        if self.relpath == f"tile_dsl/{suffix}.py":
            return self._module_defines(name)
        if name in self.info.aliases:
            return self._resolves_to(_name(name), suffix, name)
        if name in self.bound or self._module_defines(name):
            return False
        node = self._import_node(suffix)
        if node is None or any(a.asname for a in node.names):
            return False
        self.imports[suffix].add(name)
        return True

    def _straight_line(self, match: audit.BoxMatch) -> bool:
        """``seg``/``col`` bindings reach the chain unchanged through straight-line code."""
        stmt = self.stmt_of(match.chain)
        body = self._body_of(stmt)
        seg, col = match.seg_assign, match.col_assign
        if body is None or not all(any(s is a for s in body) for a in (seg, col)):
            return False
        i_seg, i_col, i_chain = (
            next(i for i, s in enumerate(body) if s is a) for a in (seg, col, stmt)
        )
        if not i_seg < i_col < i_chain:
            return False
        allowed = (ast.Name, ast.Attribute, ast.Constant, ast.BinOp, ast.UnaryOp)
        col_nodes = list(ast.walk(match.col))
        if not all(
            isinstance(n, (*allowed, ast.operator, ast.unaryop, ast.expr_context))
            for n in col_nodes
        ):
            return False
        watched = {match.seg_name, match.col_name}
        watched |= {audit.dotted(n) for n in col_nodes if isinstance(n, (ast.Name, ast.Attribute))}
        has_attr = any(isinstance(n, ast.Attribute) for n in col_nodes)
        if not isinstance(stmt, (*SIMPLE_STMTS, ast.Return)):
            return False
        for s in body[i_seg + 1 : i_chain]:
            if not isinstance(s, SIMPLE_STMTS) or (
                has_attr and any(isinstance(n, ast.Call) for n in ast.walk(s))
            ):
                return False
        # Every write after ``seg = ...`` up to the chain's evaluation, except the ``col`` binding
        # itself (the chain statement's own target is written after its value is computed).
        between = [s for s in body[i_seg + 1 : i_chain] if s is not col]
        for s in [*between, stmt.value]:
            for n in ast.walk(s):
                if isinstance(getattr(n, "ctx", None), ast.Store) and audit.dotted(n) in watched:
                    return False
        return True

    def _module_defines(self, name: str) -> bool:
        return any(isinstance(n, ast.FunctionDef) and n.name == name for n in self.info.tree.body)

    def _body_of(self, stmt: ast.stmt) -> list[ast.stmt] | None:
        parent = self.parent(stmt)
        for field in ("body", "orelse", "finalbody"):
            body = getattr(parent, field, None)
            if isinstance(body, list) and stmt in body:
                return body
        return None

    def _name_nodes(self, name: str, scope: ast.AST | None) -> list[ast.Name]:
        scope = scope or self.info.tree
        return [n for n in ast.walk(scope) if isinstance(n, ast.Name) and n.id == name]

    def _next_store_line(self, name: str, scope, after: ast.stmt) -> int:
        stores = [
            n.lineno
            for n in self._name_nodes(name, scope)
            if isinstance(n.ctx, ast.Store) and n.lineno > after.end_lineno
        ]
        return min(stores, default=sys.maxsize)

    def _uses_in_lifetime(self, name: str, scope, stmt: ast.stmt) -> list[ast.Name]:
        """Loads of ``name`` after ``stmt`` and before its next assignment (line order)."""
        limit = self._next_store_line(name, scope, stmt)
        return [
            n
            for n in self._name_nodes(name, scope)
            if isinstance(n.ctx, ast.Load) and stmt.end_lineno < n.lineno <= limit
        ]

    def _loads_after(self, name, scope, stmt, exclude=None, skip_ids=frozenset()) -> bool:
        skipped = {
            id(n)
            for f in self.info.findings
            if f.rule == "inline-swizzle-box" and id(f.detail.chain) in skip_ids
            for n in ast.walk(f.detail.chain)
        }
        excluded = {id(n) for n in ast.walk(exclude)} if exclude is not None else set()
        return any(
            id(n) not in skipped and id(n) not in excluded
            for n in self._uses_in_lifetime(name, scope, stmt)
        )

    def _import_edits(self) -> list[Edit]:
        edits = []
        tree = self.info.tree
        for suffix, names in self.imports.items():
            node = next(
                (
                    n
                    for n in tree.body
                    if isinstance(n, ast.ImportFrom)
                    and n.level
                    and n.module
                    and n.module.rsplit(".", 1)[-1] == suffix
                ),
                None,
            )
            have = [a.name for a in node.names]
            missing = sorted(set(names) - set(have))
            if not missing:
                continue
            module = "." * node.level + node.module
            all_names = sorted(set(have) | set(missing))
            text = f"from {module} import {', '.join(all_names)}"
            if len(text) > 99 or node.end_lineno != node.lineno:
                text = (
                    f"from {module} import (\n" + "".join(f"    {n},\n" for n in all_names) + ")"
                )
            start, end = self.src.span(node)
            edits.append(Edit(start, end, text, "import"))
        if self.new_import_lines:
            anchor = next(
                (
                    n
                    for n in tree.body
                    if isinstance(n, ast.ImportFrom)
                    and n.module == "cutlass.experimental"
                    and any(a.name == "primitives" for a in n.names)
                ),
                None,
            )
            anchor = (
                anchor or [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))][-1]
            )
            at = self.src.starts[anchor.lineno - 1]
            text = "".join(line + "\n" for line in sorted(self.new_import_lines))
            edits.append(Edit(at, at, text, "import"))
        return edits


def audit_is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _same_value(a: ast.AST, b: ast.AST) -> bool:
    if audit.same(a, b):
        return True
    try:
        return ast.literal_eval(a) == ast.literal_eval(b)
    except ValueError:
        return False


def apply_edits(text: str, edits: list[Edit]) -> tuple[str, int]:
    """Apply non-overlapping edits; returns the new text and the number of edits skipped."""
    out, pos, skipped = [], 0, 0
    for edit in sorted(edits, key=lambda e: (e.start, e.end)):
        if edit.start < pos:
            skipped += 1
            continue
        out += [text[pos : edit.start], edit.text]
        pos = edit.end
    out.append(text[pos:])
    return "".join(out), skipped


@dataclass
class FileResult:
    relpath: str
    original: str
    text: str
    applied: Counter
    manual: list


def restyle_source(
    text: str,
    relpath: str,
    entries: list[audit.AllowEntry] | None = None,
    helpers: dict | None = None,
    max_passes: int = 4,
) -> FileResult:
    """Rewrite one module to a fixed point.

    ``helpers`` are the target tree's analyzed ``tile_dsl`` modules (``tile_helpers(root)``);
    rewrites that call a helper missing from them stay pending.
    """
    entries = entries or []
    helpers = helpers or {}
    original, applied = text, Counter()
    for _ in range(max_passes):
        rewriter = _Rewriter(text, relpath, entries, helpers)
        edits = rewriter.run()
        if not any(e.rule != "import" for e in edits):
            break
        new_text, skipped = apply_edits(text, edits)
        try:
            ast.parse(new_text)
        except SyntaxError as exc:
            raise RuntimeError(f"{relpath}: restyle produced invalid Python: {exc}") from exc
        applied += rewriter.applied
        text = new_text
        if not skipped:
            break
    final = _Rewriter(text, relpath, entries, helpers)
    final.run()
    for entry in entries:  # hit counts belong to the final audit pass only
        entry.hits = 0
    return FileResult(relpath, original, text, applied, final.manual)


def restyle_tree(root: Path, entries, files=None, write=False) -> list[FileResult]:
    """Restyle ``root`` (or only ``files`` under it) against the tree's own tile_dsl helpers."""
    helpers = tile_helpers(root)
    results = []
    for path in audit.iter_sources(root, files):
        relpath = path.relative_to(root).as_posix()
        result = restyle_source(path.read_text(), relpath, entries, helpers)
        if write and result.text != result.original:
            path.write_text(result.text)
        results.append(result)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="report pending rewrites; exit 1")
    mode.add_argument("--write", action="store_true", help="rewrite files in place")
    parser.add_argument("--root", type=Path, help="cudnn_fe package dir (default: installed)")
    parser.add_argument(
        "--files", nargs="+", metavar="REL", help="restyle only these paths relative to the root"
    )
    parser.add_argument("--allowlist", type=Path, default=audit.DEFAULT_ALLOWLIST)
    parser.add_argument("--no-allowlist", action="store_true")
    parser.add_argument("--diff", action="store_true", help="with --check, print a unified diff")
    args = parser.parse_args(argv)

    entries = [] if args.no_allowlist else audit.load_allowlist(args.allowlist)
    root = args.root or audit.default_root()
    results = restyle_tree(root, entries, args.files, write=args.write)
    changed = [r for r in results if r.text != r.original]
    applied = sum((r.applied for r in results), Counter())
    for result in changed:
        print(
            f"{'rewrote' if args.write else 'would rewrite'} {result.relpath}: "
            + ", ".join(f"{rule} x{n}" for rule, n in sorted(result.applied.items()))
        )
        if args.diff:
            sys.stdout.writelines(
                difflib.unified_diff(
                    result.original.splitlines(keepends=True),
                    result.text.splitlines(keepends=True),
                    f"a/{result.relpath}",
                    f"b/{result.relpath}",
                )
            )
    manual = [f for r in results for f in r.manual]
    print(
        f"\nmechanical rewrites: {sum(applied.values())} "
        + " ".join(f"{rule}={applied[rule]}" for rule in MECHANICAL_RULES)
    )
    print(f"left for manual restyle ({len(manual)} audit findings):")
    print(audit.format_counts(audit.rule_counts(manual)))
    for step in MANUAL_STEPS:
        print(f"  manual: {step}")
    return int(bool(changed) and args.check)


if __name__ == "__main__":
    sys.exit(main())
