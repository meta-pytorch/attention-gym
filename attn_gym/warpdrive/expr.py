"""Read loop bounds and branch conditions of a traced role as small symbolic expressions.

The liveness check unrolls each role's channel ops for chosen loop counts, so it needs to know
how often each loop runs and which branch arm is taken. This module reads that from the MLIR a
role body traced into, and only as far as it can: anything it does not recognise is None
(unknown), and an unknown never becomes a finding, only an ``unverified`` verdict.

Expressions are nested tuples, comparable and hashable:

* ``("c", int)``: a constant.
* ``("iv", key)``: the induction variable of the loop whose body is region ``key``.
* ``("item", stream, field)``: the current work item's index or a decoded word.
* ``("tid", name)`` / ``("rank", name)`` / ``("uni", name)``: thread index, cluster rank, and
  other special registers that are the same for every thread of a CTA.
* ``(op, *args)``: integer arithmetic with MLIR's semantics (``div``/``rem`` truncate,
  ``floordiv`` floors, a ``u`` suffix is unsigned), ``("cmp", predicate, a, b)``
  (``arith.cmpi`` predicates), ``("select", c, a, b)`` (also an ``scf.if`` that yields a
  value).
* ``("arg", n)``, ``("op", signature, *args)``, ``("load", type, *args)``: a kernel argument, a
  pure op the evaluator does not model (CuTe layout queries, ...), and a load from memory. They
  are leaves: equal expressions read the same value in every role, which is what relates two
  roles' loops. A load is assumed to read an input no role writes during the launch.

Loop-carried values the loop passes through unchanged resolve to their initial value; others
are unknown.
"""

from __future__ import annotations

import re

from cutlass._mlir import ir

ARITH = {
    "arith.addi": "add",
    "arith.subi": "sub",
    "arith.muli": "mul",
    "arith.divsi": "div",
    "arith.divui": "divu",
    "arith.floordivsi": "floordiv",
    "arith.ceildivsi": "ceildiv",
    "arith.ceildivui": "ceildivu",
    "arith.remsi": "rem",
    "arith.remui": "remu",
    "arith.minsi": "min",
    "arith.minui": "minu",
    "arith.maxsi": "max",
    "arith.maxui": "maxu",
    "arith.andi": "and",
    "arith.ori": "or",
    "arith.xori": "xor",
    "arith.select": "select",
}
# CuTe integer-tuple arithmetic on one-element tuples, and the casts that wrap scalars in them.
CUTE_ARITH = {
    "cute.tuple_add": "add",
    "cute.tuple_sub": "sub",
    "cute.tuple_mul": "mul",
    "cute.tuple_div": "div",
    "cute.ceil_div": "ceildiv",
}
# Integer casts, read as ``("trunc", ("c", bits), x)`` to a narrower type or as a sign or zero
# extension (``"sext"``/``"zext"``) from ``bits``. True if the cast extends with the sign.
CASTS = {
    "arith.index_cast": True,
    "arith.extsi": True,
    "arith.trunci": True,
    "arith.index_castui": False,
    "arith.extui": False,
}
PASS = ("builtin.unrealized_conversion_cast",)
CUTE_PASS = ("cute.get_scalars", "cute.get_leaves", "cute.make_int_tuple", "cute.to_int_tuple")
# Op names that touch memory or synchronize: their results are not functions of their operands.
IMPURE = (
    "load",
    "store",
    "copy",
    "alloc",
    "atomic",
    "mma",
    "arrive",
    "wait",
    "barrier",
    "tma",
    "gemm",
    "fence",
    "elect",
    "shfl",
    "vote",
    "clock",
)
KERNELS = ("cuda.kernel", "gpu.func", "func.func")
STATIC_INT = re.compile(r'!cute\.int_tuple<"(-?\d+)">')
MAX_DEPTH = 400


def _bits(t) -> int | None:
    t = str(t)
    if t == "index":
        return 64
    return int(t[1:]) if re.fullmatch(r"i\d+", t) else None


def _op(owner):
    return owner.operation if hasattr(owner, "operation") else owner


def _value(v):
    """The ``ir.Value`` behind a DSL value."""
    if isinstance(v, ir.Value):
        return v
    inner = getattr(v, "value", None)
    return inner if isinstance(inner, ir.Value) else v.__extract_mlir_values__()[0]


def _same(a, b) -> bool:
    # CuTeDSL overloads == on values to build comparisons; compare the SSA values themselves.
    try:
        return ir.Value.__eq__(_value(a), _value(b)) is True
    except Exception:  # noqa: BLE001 - anything unreadable is simply not the same value
        return False


def _terminator(block):
    ops = list(block.operations)
    return _op(ops[-1]) if ops else None


def _passes(value, target, depth: int = 0) -> bool:
    """Whether ``value`` is ``target``, or an ``scf.if`` result both arms of which yield it."""
    if _same(value, target):
        return True
    owner = _value(value).owner
    if depth > 20 or isinstance(owner, ir.Block) or _op(owner).name != "scf.if":
        return False
    n = ir.OpResult(_value(value)).result_number
    arms = [r.blocks[0] for r in _op(owner).regions if len(r.blocks)]
    return len(arms) == 2 and all(
        _passes(list(_terminator(b).operands)[n], target, depth + 1) for b in arms
    )


def _loop_init(loop, block, n: int):
    """The initial value of block argument ``n`` of a loop body the loop never changes."""
    if loop.name == "scf.for":
        k = n - 1
        if _passes(list(_terminator(block).operands)[k], block.arguments[n]):
            return list(loop.operands)[3 + k]
        return None
    if loop.name == "scf.while":
        before, after = (r.blocks[0] for r in loop.regions)
        if block == before:
            k = n
        else:  # an "after" argument is a value scf.condition forwards from "before"
            forwarded = list(_terminator(before).operands)[n + 1]
            k = next((i for i, a in enumerate(before.arguments) if _same(a, forwarded)), None)
            if k is None:
                return None
        passed = list(_terminator(after).operands)[k]
        if _passes(passed, before.arguments[k]) or (
            block == after and _passes(passed, block.arguments[n])
        ):
            return list(loop.operands)[k]
    return None


class Reader:
    """Read values of one role body. ``ivs(block)`` names a loop body's region key (None if it
    holds no traced event); ``items`` lists ``(value, expr)`` of the work cursor's carried
    item values."""

    def __init__(self, ivs, items):
        self._ivs, self._memo = ivs, list(items)
        self.why = ""  # what the last unknown value depends on

    def read(self, value):
        """The expression of ``value``, or None (``why`` says what it depends on). Never
        raises."""
        self.why = ""
        try:
            return self._read(_value(value), 0)
        except Exception as e:  # noqa: BLE001 - an unreadable value is unknown, not an error
            self.why = f"{type(e).__name__}: {e}"
            return None

    def _read(self, v, depth: int):
        if depth > MAX_DEPTH:
            self.why = "a deep expression"
            return None
        for seen, expr in self._memo:
            if _same(seen, v):
                return expr
        expr = self._build(v, depth)
        self._memo.append((v, expr))
        return expr

    def _build(self, v, depth: int):
        static = STATIC_INT.fullmatch(str(v.type))
        if static:  # a CuTe integer whose value is in its type
            return ("c", int(static.group(1)))
        owner = v.owner
        if isinstance(owner, ir.Block):
            parent = _op(owner.owner)
            n = ir.BlockArgument(v).arg_number
            if parent.name in KERNELS:
                return ("arg", n)
            if parent.name == "scf.for" and n == 0:
                key = self._ivs(owner)
                if key is None:
                    self.why = "the index of a loop without channel ops"
                return None if key is None else ("iv", key)
            init = _loop_init(parent, owner, n)
            if init is None:
                self.why = f"a value the {parent.name} loop carries"
                return None
            return self._read(_value(init), depth + 1)
        op = _op(owner)
        name = op.name
        if name == "arith.constant":
            return ("c", int(op.attributes["value"].value))
        args = [self._read(_value(o), depth + 1) for o in op.operands]
        if any(a is None for a in args):
            return None  # ``why`` names the operand's cause
        scalar = str(v.type) in ("i1", "i32", "i64", "index") or (
            str(v.type).startswith("!cute.int_tuple") and "," not in str(v.type)
        )
        if name in ARITH:
            return (ARITH[name], *args)
        if name == "arith.cmpi":
            return ("cmp", int(op.attributes["predicate"].value), *args)
        if name in PASS or (name in CUTE_PASS and scalar and ir.OpResult(v).result_number == 0):
            if len(args) == 1:
                return args[0]
        if name in CASTS:
            src, dst = _bits(op.operands[0].type), _bits(v.type)
            if src is None or dst is None:
                self.why = f"{name} from {op.operands[0].type} to {v.type}"
                return None
            if src == dst:
                return args[0]
            kind = "trunc" if dst < src else ("sext" if CASTS[name] else "zext")
            return (kind, ("c", min(src, dst)), *args)
        if name in CUTE_ARITH and scalar and len(args) == 2:
            return (CUTE_ARITH[name], *args)
        if name == "scf.if":
            n = ir.OpResult(v).result_number
            arms = [
                self._read(_value(list(_terminator(r.blocks[0]).operands)[n]), depth + 1)
                for r in op.regions
            ]
            if len(arms) != 2 or None in arms:
                self.why = self.why or "an scf.if result"
                return None
            return arms[0] if arms[0] == arms[1] else ("select", args[0], *arms)
        if "sreg" in name:
            kind = "tid" if ("tid" in name or "laneid" in name) else "uni"
            kind = "rank" if "cluster" in name and "rank" in name else kind
            return (kind, name)
        if name == "cute.memref.load":
            return ("load", str(v.type), *args)
        if any(word in name for word in IMPURE) or not name.startswith(
            ("arith.", "math.", "index.", "cute.")
        ):
            self.why = name
            return None
        attrs = ",".join(sorted(f"{k}={op.attributes[k]}" for k in op.attributes))
        return ("op", f"{name}#{ir.OpResult(v).result_number}:{v.type}[{attrs}]", *args)
