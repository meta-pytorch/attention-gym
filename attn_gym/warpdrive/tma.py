"""Trace-time check of TMA loads against the channel they complete on.

A TMA load names its barrier through ``cute.copy(atom, src, dst, tma_bar_ptr=ptr)``, a bulk
(non-tensor) ``cp.async.bulk`` load through ``mbar_ptr=ptr``. Whether the bytes land where the
layer armed them depends on the atom: a ``CtaGroup.TWO`` tensor load completes on the leader
CTA's barrier from either CTA of a pair, a ``CtaGroup.ONE`` tensor load and a bulk load on the
issuing CTA's own barrier. A mismatch with the channel's derived scope, or a ``Tma(tx_bytes)``
that differs from the bytes actually copied, leaves a barrier that never completes (a hang) or
completes early.

The layer binds copies to channels by usage, with no change to user code. Every full-barrier
pointer of a ``Tma`` channel it hands out (``handle.barrier``, ``end.full_barrier(i)``) is recorded
with its SSA value. When the role returns, the check follows that value's uses in the traced IR to
``cute_nvgpu.atom.set_value<tma_bar>`` ops, reads the atom's type (load kind, ``copy_bits`` per
box) and each ``cute.copy`` destination layout, and compares:

* the atom's completion scope (its CTA group) with the one the channel's scope requires;
* per acquired stage, the bytes of every copy issued in the stage's region with the bytes the
  layer arms (``Scope.tx``, per producing CTA). A tensor copy counts as issued once per stage
  when it sits directly in the region that acquired the stage or in an ``elect_one()`` block
  there; a bulk copy does not elect a lane itself, so it counts only inside ``elect_one()``
  (directly in the region, every active lane would issue it).

The check fails open: anything it cannot recognize (a CuTeDSL IR change, an unknown atom kind, a
copy inside a dynamic loop or branch, a pointer the layer did not hand out) marks the channel
*unverified* in ``describe()`` and emits one warning per kernel. It raises only on a positively
identified mismatch. Values are inspected without the DSL's value casters, which emit ops when
invoked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from attn_gym.warpdrive.cluster import LEADER
from attn_gym.warpdrive.protocol import ProtocolError

SET_VALUE = "cute_nvgpu.atom.set_value"
MAKE_EXEC = "cute_nvgpu.atom.make_exec_tma"
MAKE_ATOM = "cute.make_atom"
BULK_G2S = "!cute_nvgpu.atom.bulk_copy_g2s<"
COPY = "cute.copy"
ELECT = "nvvm.elect.sync"
# TiledTmaLoadEnum spelling -> (CtaGroup, multicast)
KINDS = {
    "sm_90": ("ONE", False),
    "sm_100_2sm": ("TWO", False),
    "sm_90_multicast": ("ONE", True),
    "sm_100_2sm_multicast": ("TWO", True),
}


# Multicast masks handed out by ``Cluster.multicast``/``Cluster.fill`` in the kernel being
# traced: {hash of the mask's ir.Value, or ("const", bits) for a mask the same in every CTA:
# (mode, CTAs reached, the CTAs ``Cluster.fill`` built it for)}. Tracing is single-threaded;
# cleared at ``begin_roles``.
_MASKS: dict = {}


def record_multicast_mask(mask, mode: str, ctas: int, target=None) -> None:
    """Record a mask; ``target``: the rank (a Python int or a device value) of the CTA a
    ``Cluster.fill`` copy is for. Fills of different CTAs can share a constant mask (two CTAs on
    one multicast line), so a mask keeps every target it was built for."""
    _, _, targets = _MASKS.setdefault(_mask_key(mask.ir_value()), (mode, ctas, set()))
    if target is not None:
        targets.add(target if isinstance(target, int) else ("rank", hash(target.ir_value())))


def _mask_key(value) -> int | tuple[str, int | None]:
    owner = value.owner
    key = hash(value)
    if getattr(owner, "name", None) == "arith.constant":
        # A constant mask is rematerialized at each use, so it is known by its bits.
        key = ("const", _constant_bits(owner))
    return key


def _constant_bits(op) -> int | None:
    match = re.match(r"(-?\d+)", str(op.attributes["value"]))
    return int(match.group(1)) & 0xFFFF if match else None


def clear_multicast_masks() -> None:
    _MASKS.clear()


@dataclass
class TmaUse:
    """A full-barrier pointer of a TMA channel handed to user code."""

    chan: object
    value: object  # the pointer's ir.Value
    site: str
    handle: object | None  # the HandleRecord of the acquired stage; None for end.full_barrier
    opted_out: bool = False  # taken inside unchecked("tma"): its copies are not checked
    # (copies, set_value ops, problem), computed when the stage's handle closes: its region is
    # live then, while a region the tracer discards later must not be walked.
    result: tuple | None = None


@dataclass
class TmaReport:
    """What the check established for one channel (shown by ``describe()``)."""

    copies: set = field(default_factory=set)  # sites of verified copies
    groups: set = field(default_factory=set)
    stage_bytes: set = field(default_factory=set)
    unverified: list = field(default_factory=list)
    opted_out: bool = False

    def status(self, fmt_sites=None) -> str:
        """``verified ...``, ``unverified (reasons)``, or ``opted out``; ``fmt_sites`` joins the
        copy sites (default: comma-separated)."""
        if self.unverified:
            return f"unverified ({'; '.join(dict.fromkeys(self.unverified))})"
        if not self.copies:
            return "opted out (unchecked)" if self.opted_out else "no TMA copies traced"
        groups = "/".join(_atom_label(g) for g in sorted(self.groups))
        nbytes = ", ".join(str(b) for b in sorted(self.stage_bytes)) or "not counted"
        sites = (fmt_sites or (lambda s: ", ".join(sorted(s))))(self.copies)
        partial = "; some barriers opted out" if self.opted_out else ""
        return f"verified {groups}, {nbytes} B/stage per CTA @ {sites}{partial}"


@dataclass
class _Copy:
    site: str
    group: str  # the CTA group whose barrier the bytes complete on ("ONE" for a bulk copy)
    nbytes: int | None  # None: not countable (dynamic layout or unparsed type)
    in_stage_region: bool  # issued in the region that acquired the stage (runs once per stage)
    bulk: bool = False  # a cp.async.bulk (non-tensor) load
    unelected: bool = False  # a bulk copy outside elect_one(): issued by every active lane
    # The cluster mode of a multicast copy's mask ("fill": a single-CTA Cluster.fill mask, "?": a
    # mask the layer did not hand out); None when not multicast.
    multicast: str | None = None
    # The copy's key in _MASKS; None when not multicast (it lands in the issuing CTA).
    mask_key: object = None


@dataclass
class _Stage:
    """Every barrier use of one acquired stage; bytes are compared only when complete."""

    use: TmaUse
    copies: list = field(default_factory=list)
    incomplete: list = field(default_factory=list)  # why the byte total may be partial
    opted_out: bool = False  # a barrier of this stage was taken inside unchecked("tma")


def required_group(chan, cluster) -> str | None:
    """The CTA group a TMA load into ``chan`` must use (None: either works)."""
    if not cluster.pair:
        return "ONE"
    if chan.derived_scope.tx_ctas > 1:
        return "TWO"  # both CTAs' bytes complete on the leader's barrier
    leaders = set(cluster.ctas(LEADER))
    if all(set(cluster.ctas(ri.cta)) <= leaders for ri, _ in [*chan.producers, *chan.consumers]):
        return None  # leader-only on both sides: a 2-CTA load also lands on the leader
    return "ONE"


def _fill_count(chan) -> int:
    return len(chan.cluster.filled(0, chan.producer.fills))


def stage_closed(trace, handle) -> None:
    """Analyze the barrier uses of one TMA stage when its handle closes."""
    for use in trace.tma_bars:
        if use.handle is handle and use.result is None:
            use.result = _copies(use)


def check_role(trace, cluster, reports: dict) -> None:
    """Analyze every TMA barrier use recorded while tracing one role.

    The atom's CTA group is checked for every copy positively traced to a channel barrier.
    A stage's byte total is compared only when every barrier use of that stage was traced and
    the role issued no TMA copy the check could not attribute; otherwise the channel is marked
    unverified.
    """
    stages: dict[int, _Stage] = {}
    partial_chans: dict[str, str] = {}  # channel -> why its per-stage totals may be partial
    opted_out_chans: set[str] = set()  # end.full_barrier() taken inside unchecked("tma")
    reached = []
    for use in trace.tma_bars:
        report = reports.setdefault(use.chan.name, TmaReport())
        stage = None
        if use.handle is not None:
            stage = stages.setdefault(id(use.handle), _Stage(use))
        copies, ops, problem = use.result or _copies(use)
        reached.extend(ops)
        if use.opted_out:
            report.opted_out = True
            if stage is not None:
                stage.opted_out = True
            else:
                opted_out_chans.add(use.chan.name)
            continue
        if problem is None and not copies:
            problem = f"barrier taken at {use.site} reaches no recognized TMA copy"
        if problem is not None:
            report.unverified.append(problem)
            if stage is not None:
                stage.incomplete.append(problem)
            else:
                partial_chans.setdefault(use.chan.name, problem)
            continue
        need = required_group(use.chan, cluster)
        tma = use.chan.producer
        # Masks a channel admits: its multicast modes, and with fills the single-CTA fill mask.
        modes = (*tma.multicast, "fill") if tma.fills else tma.multicast
        for cp in copies:
            if need is not None and cp.group != need:
                raise ProtocolError(_group_message(trace.role_name, use, cp, need, cluster))
            if cp.multicast is not None and (not modes or cp.multicast not in (*modes, "?")):
                raise ProtocolError(_multicast_message(trace.role_name, use, cp))
            if cp.multicast == "?":
                what = "Cluster.fill" if tma.fills else "Cluster.multicast"
                report.unverified.append(
                    f"multicast copy at {cp.site} takes a mask not from {what}"
                )
                if tma.fills and stage is not None:
                    # Its CTAs are unknown, so the bytes it delivers are too.
                    stage.incomplete.append(f"multicast copy at {cp.site} has an unknown mask")
            report.groups.add("bulk" if cp.bulk else cp.group)
            report.copies.add(cp.site)
        if stage is not None:
            stage.copies.extend(copies)
        else:  # end.full_barrier(i): stage unknown, so totals of every stage may be partial
            partial_chans.setdefault(use.chan.name, f"end.full_barrier() copy at {use.site}")
    untracked = _untracked_tma_copies(trace.root, reached)
    for stage in stages.values():
        use, rec = stage.use, stage.use.handle
        report = reports[use.chan.name]
        why = list(stage.incomplete)
        if use.chan.name in partial_chans:
            why.append(partial_chans[use.chan.name])
        if untracked:
            why.append(untracked)
        for cp in stage.copies:
            if cp.unelected:
                why.append(
                    f"the bulk copy at {cp.site} sits outside elect_one(), so every lane issues it"
                )
        if any(cp.nbytes is None or not cp.in_stage_region for cp in stage.copies):
            why.append("a copy sits in a nested loop/branch or has a dynamic layout")
        filled = _fill_count(use.chan)
        if filled > 1 and _fill_targets(stage.copies) < filled:
            # Copies that land twice in one CTA can total the right bytes yet starve another.
            why.append(f"its copies take fewer distinct masks than the {filled} CTAs it fills")
        if why:
            report.unverified.append(f"bytes of {rec.what} at {rec.site}: {why[0]}")
        if why or stage.opted_out or use.chan.name in opted_out_chans:
            continue
        got = sum(cp.nbytes for cp in stage.copies)
        # One producing CTA's share of the armed bytes, for each CTA it fills.
        want = use.chan.scope.tx // use.chan.derived_scope.tx_ctas * filled
        if got != want:
            raise ProtocolError(_bytes_message(trace.role_name, use, stage.copies, got, want))
        report.stage_bytes.add(got // filled)  # per CTA, like Tma.tx_bytes


def _copies(use: TmaUse) -> tuple[list[_Copy], list, str | None]:
    """The TMA copies whose barrier is ``use.value``, the ``set_value`` ops that name it, or a
    reason they cannot be analyzed."""
    copies, ops = [], []
    try:
        for operand in use.value.uses:
            op = operand.owner.operation
            if not _is_tma_bar(op):
                continue
            ops.append(op)
            kind = _kind(op)
            if kind is None:
                return [], ops, f"unrecognized TMA atom {str(op.results[0].type)[:60]!r}"
            group, multicast, bulk = kind
            copy_bits = re.search(r"copy_bits = (\d+)", str(op.results[0].type))
            for copy_op in _copy_ops(op):
                nbytes = _copy_bytes(copy_op, copy_bits, bulk)
                region = use.handle.block if use.handle is not None else None
                elected = _elect_parent(copy_op)
                direct = region is not None and copy_op.block == region
                in_stage = (region is not None and elected is not None and elected == region) or (
                    direct and not bulk
                )
                unelected = bulk and direct
                mode, key = _mask_mode(copy_op) if multicast else (None, None)
                copies.append(_Copy(use.site, group, nbytes, in_stage, bulk, unelected, mode, key))
    except (AttributeError, KeyError, IndexError, ValueError, RuntimeError) as e:
        return [], ops, f"cannot inspect the IR at {use.site}: {type(e).__name__}: {e}"
    return copies, ops, None


def _fill_targets(copies) -> int:
    """How many CTAs a stage's copies can be for, at most: the CTAs ``Cluster.fill`` built their
    masks for (each mask used at most once per such CTA), plus one per other mask and one for
    unmasked copies (the issuing CTA)."""
    uses = {}
    for cp in copies:
        uses[cp.mask_key] = uses.get(cp.mask_key, 0) + 1
    filled, uses_of_filled, other = set(), 0, 0
    for key, n in uses.items():
        targets = _MASKS.get(key, (None, 0, set()))[2] if key is not None else ()
        if targets:
            filled |= targets
            uses_of_filled += min(n, len(targets))
        else:
            other += 1
    return min(uses_of_filled, len(filled)) + other


def _is_tma_bar(op) -> bool:
    return op.name == SET_VALUE and "<tma_bar>" in str(op.attributes["field"])


def _untracked_tma_copies(root, reached: list) -> str | None:
    """A reason when the role issues TMA copies whose barrier the layer did not hand out
    (their bytes may complete on a layer channel), else None. Walks op names and attributes
    only."""
    try:
        found = []

        def walk(block):
            for op in block.operations:
                op = op.operation
                if _is_tma_bar(op):
                    found.append(op)
                for region in op.regions:
                    for b in region.blocks:
                        walk(b)

        walk(root)
        extra = sum(1 for op in found if not any(op == r for r in reached))
    except (AttributeError, KeyError, ValueError, RuntimeError) as e:
        return f"cannot scan the role's TMA copies: {type(e).__name__}"
    return f"{extra} TMA copy(ies) use a barrier the layer did not hand out" if extra else None


def _kind(set_value) -> tuple[str, bool, bool] | None:
    """(CTA group, multicast, bulk) of the atom a ``set_value`` chain starts from: a tensor TMA
    exec atom, or a bulk ``cp.async.bulk`` G2S atom, whose bytes complete on the barrier at the
    given offset in the issuing CTA (in every CTA of its mask when multicast), like a
    ``CtaGroup.ONE`` tensor load."""
    op = set_value
    while op.name == SET_VALUE:  # earlier fields set on the same atom
        op = op.operands[0].owner.operation  # atom-typed values have no DSL caster
    if op.name == MAKE_ATOM and str(op.results[0].type).startswith(BULK_G2S):
        return "ONE", "mcast = true" in str(op.results[0].type), True
    if op.name != MAKE_EXEC:
        return None
    kind = re.match(r"!cute_nvgpu\.atom\.non_exec_tiled_tma_load<(\w+),", str(op.operands[0].type))
    return (*KINDS[kind.group(1)], False) if kind and kind.group(1) in KINDS else None


def _atom_label(group: str) -> str:
    return "cp.async.bulk" if group == "bulk" else f"CtaGroup.{group}"


def _elect_parent(copy_op):
    """The enclosing block when a copy sits directly in an ``elect_one()`` region (one lane per
    warp issues it once per execution of that block), else None."""
    owner = copy_op.block.owner.operation
    if owner.name == "scf.if" and owner.operands[0].owner.operation.name == ELECT:
        return owner.block
    return None


def _mask_mode(copy_op) -> tuple[str, object]:
    """``(mode, key)`` of the multicast mask set on a copy's atom: its cluster mode, or "?" when
    the layer did not hand it out, and its key in ``_MASKS``."""
    op = copy_op.operands[0].owner.operation
    while op.name == SET_VALUE:
        if "<mcast_mask>" in str(op.attributes["field"]):
            key = _mask_key(op.operands[1])
            return _MASKS.get(key, ("?",))[0], key
        op = op.operands[0].owner.operation
    return "?", None


def _copy_ops(set_value) -> list:
    """``cute.copy`` ops consuming this exec atom, through any later ``set_value`` fields."""
    out, frontier = [], [set_value]
    while frontier:
        op = frontier.pop()
        for operand in op.results[0].uses:
            user = operand.owner.operation
            if user.name == COPY:
                out.append(user)
            elif user.name == SET_VALUE:
                frontier.append(user)
    return out


def _copy_bytes(copy_op, copy_bits, bulk: bool) -> int | None:
    """Bytes one ``cute.copy`` delivers: every element of its static SMEM destination, which
    must be a whole number of ``copy_bits`` boxes (a bulk copy without ``copy_bits`` moves the
    destination in one instruction). None when the type is not recognized."""
    if copy_bits is None and not bulk:
        return None
    text = str(copy_op)
    dst = re.search(r'!cute\.memref<([a-z]+)(\d+)\w*, smem[^"]*"([^"]*)">\) -> \(\)\s*$', text)
    if "operandSegmentSizes = array<i32: 1, 1, 1, 0>" not in text or dst is None:
        return None
    size = _shape_size(dst.group(3).split(":")[0])
    if size is None:
        return None
    bits = int(dst.group(2)) * size
    box = int(copy_bits.group(1)) if copy_bits is not None else 8
    return bits // 8 if bits % box == 0 and bits % 8 == 0 else None


def _shape_size(text: str) -> int | None:
    """Product of extents in a CuTe shape string; None if dynamic or unrecognized."""
    if "?" in text:
        return None
    tokens = re.findall(r"\(|\)|\d+", text)
    pos = 0

    def parse():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok != "(":
            return int(tok)
        size = 1
        while tokens[pos] != ")":
            size *= parse()
        pos += 1
        return size

    try:
        return parse()
    except (IndexError, ValueError):
        return None


def _group_message(role: str, use: TmaUse, cp: _Copy, need: str, cluster) -> str:
    chan = use.chan
    if cp.bulk:
        what = "a cp.async.bulk copy, which completes on the issuing CTA's barrier"
    else:
        what = f"a CtaGroup.{cp.group} atom"
    head = (
        f"{role}: the TMA copy at {cp.site} completes on channel {chan.name}'s full barrier with "
        f"{what}, but {chan.name} needs CtaGroup.{need}: "
    )
    if cluster.size == 1:
        why = "a 2-CTA load in a single-CTA kernel completes on a peer CTA's barrier"
    elif need == "TWO":
        why = (
            f"{chan.name} is pair-scoped, so its full barrier lives in the leader, which arms "
            f"{chan.scope.tx} bytes for both CTAs' copies; a copy that completes in the peer "
            "completes on the peer's own barrier, which nobody waits on, and the leader hangs"
        )
    else:
        why = (
            f"{chan.name} is CTA-local (each CTA arms its own barrier with {chan.scope.tx} bytes); "
            "a CtaGroup.TWO copy from the peer completes on the leader's barrier instead"
        )
    if cp.bulk:
        fix = "use a tensor TMA atom built with self.cluster.cta_group"
    elif need == "TWO" or cluster.size == 1:
        fix = "build the atom's op with self.cluster.cta_group"
    else:
        fix = "build the atom's op with CtaGroup.ONE"
    return f"{head}{why}; {fix}"


def _multicast_message(role: str, use: TmaUse, cp: _Copy) -> str:
    chan = use.chan
    modes = chan.producer.multicast
    if cp.multicast == "fill":
        return (
            f"{role}: the TMA copy at {cp.site} into channel {chan.name} takes a Cluster.fill "
            "mask, but its Tma declares no fills: the CTA the copy lands in would not release "
            "this CTA's stage, and nobody arms its barrier; declare Tma(..., fills=...)"
        )
    along = "" if cp.multicast == "?" else f" along {cp.multicast}"
    declared = f"declares multicast={modes}" if modes else "declares no multicast"
    return (
        f"{role}: the TMA copy at {cp.site} is multicast{along} into channel {chan.name}, but "
        f"its Tma {declared}: consumers in the CTAs the copy reaches would not release this "
        f"CTA's stage, so it could be overwritten while they read it; declare "
        f"Tma(..., multicast=...) with the copy's mode"
    )


def _bytes_message(role: str, use: TmaUse, copies: list, got: int, want: int) -> str:
    chan, rec = use.chan, use.handle
    sites = ", ".join(dict.fromkeys(cp.site for cp in copies))
    effect = "never completes (hang)" if got < want else "completes before every copy lands"
    filled = _fill_count(chan)
    per_cta = " per CTA" if chan.scope.pair else ""
    if filled > 1:
        per_cta = f" for the {filled} CTAs it fills"
    return (
        f"{role}: {rec.what} opened at {rec.site}: the TMA copies at {sites} deliver {got} "
        f"bytes{per_cta}, but the layer arms {want}{per_cta} (channel {chan.name}, "
        f"Tma({chan.producer.tx_bytes})), so the stage's full barrier {effect}; declare "
        f"Tma({got // filled}) or fix the copies"
    )
