"""Classify the kernels of two snapshots as identical, same-histogram, noise or a real change.

Per kernel (paired by label and stem), from strongest to weakest:

- **identical**: the normalized instruction text (addresses and label numbers stripped) is
  equal, and so are the resources. Two runs of one tree produce this;
- **same-histogram**: the text differs (scheduling, register allocation) but the full-mnemonic
  histogram, the shared-memory layout, the mbarrier offsets and the resources are equal;
- **noise**: additionally the histogram moved at most ``NOISE_BUDGET`` instructions, all in
  ``NOISE_OPS`` (address arithmetic, moves, padding);
- **real**: anything else. Resources ``REG``/``STACK``/``DSMEM``/``SHARED``/``LOCAL``, an
  unparsable (``?``) resource on either side, the shared-memory layout (multiset of
  ``(mnemonic, immediate)`` over ``LDS``/``STS``/``LDSM``/``STSM``/``SYNCS*``), the mbarrier
  init offsets, any delta of a ``REAL_OPS`` count (TMA, convergence barriers, mbarrier waits,
  sleeps, packed fp32 math), and any other histogram delta outside the noise band.

The layout check catches a SMEM tile or barrier relocation that leaves the histogram intact
(the GDN prefill barrier move that cost ~3 %, the kda_recompute tile move of the KDA restyle).
Keying the histogram on the full mnemonic makes width (``LDS.128`` vs ``LDS.64``), rounding
(``FFMA.FTZ``) and signedness (``IMAD.WIDE.U32``) changes visible. Labels or kernels present on
one side only are real changes.
"""

from __future__ import annotations

import enum
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .sass import UNKNOWN, KernelRecord, base_opcode, format_mbarriers, labels, read_label

NOISE_OPS = frozenset({"NOP", "UMOV", "UIADD3", "IMAD", "LOP3", "CS2R", "ULOP3"})
NOISE_BUDGET = 8
REAL_OPS = frozenset(
    {"UTMALDG", "UTMASTG", "BSSY", "SYNCS", "NANOSLEEP", "FMUL2", "FADD2", "FFMA2"}
)
STRICT_RESOURCES = ("REG", "STACK", "DSMEM", "SHARED", "LOCAL")


class Verdict(enum.IntEnum):
    IDENTICAL = 0
    SAME_HISTOGRAM = 1
    NOISE = 2
    REAL = 3

    @property
    def marker(self) -> str:
        return {
            Verdict.IDENTICAL: "=",
            Verdict.SAME_HISTOGRAM: "-",
            Verdict.NOISE: "~",
            Verdict.REAL: "!",
        }[self]

    @property
    def label(self) -> str:
        return self.name.lower().replace("_", "-")


@dataclass
class KernelDiff:
    stem: str
    verdict: Verdict
    reasons: list[str] = field(default_factory=list)


@dataclass
class LabelDiff:
    label: str
    verdict: Verdict
    kernels: list[KernelDiff] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)


@dataclass
class Report:
    labels: list[LabelDiff]
    case_problems: list[str]

    def count(self, verdict: Verdict) -> int:
        return sum(1 for label in self.labels if label.verdict is verdict)

    def failed(self, strict: bool) -> bool:
        worst = Verdict.NOISE if strict else Verdict.REAL
        return bool(self.case_problems) or any(label.verdict >= worst for label in self.labels)


def span(offsets: list[int]) -> str:
    """``n@0x24000..0x243e0`` for a barrier block, the full list when short."""
    if len(offsets) <= 4:
        return format_mbarriers(offsets)
    return f"{len(offsets)}@{offsets[0]:#x}..{offsets[-1]:#x}"


def opcode_delta(before: Counter, after: Counter) -> Counter:
    return Counter(
        {op: after[op] - before[op] for op in set(before) | set(after) if after[op] != before[op]}
    )


def layout_summary(before: Counter, after: Counter, limit: int = 6) -> str:
    """``N entries: LDS.128 0x1ec00 -12 0x2ec00 +12, ...`` grouped per mnemonic."""
    delta = opcode_delta(before, after)
    per_op: dict[str, list[str]] = {}
    for (op, offset), count in sorted(delta.items()):
        per_op.setdefault(op, []).append(f"{offset:#x} {count:+d}")
    parts = [f"{op} {' '.join(moves)}" for op, moves in per_op.items()]
    shown = ", ".join(parts[:limit]) + (", ..." if len(parts) > limit else "")
    return f"smem layout {sum(abs(c) for c in delta.values())} entries: {shown}"


def classify_kernel(before: KernelRecord, after: KernelRecord) -> KernelDiff:
    reasons = []
    for name in STRICT_RESOURCES:
        old, new = before.resources.get(name, UNKNOWN), after.resources.get(name, UNKNOWN)
        if old != new:
            reasons.append(f"{name} {old} -> {new}")
        elif old == UNKNOWN:
            reasons.append(f"{name} unknown on both sides")
    if before.mbarriers != after.mbarriers:
        reasons.append(f"mbarrier offsets {span(before.mbarriers)} -> {span(after.mbarriers)}")
    if before.layout != after.layout:
        reasons.append(layout_summary(before.layout, after.layout))
    real = bool(reasons)
    delta = opcode_delta(before.ops, after.ops)
    if delta:
        summary = " ".join(f"{op} {count:+d}" for op, count in sorted(delta.items()))
        total = f"{before.instructions} -> {after.instructions} instrs"
        bases = {base_opcode(op) for op in delta}
        real_ops = sorted(bases & REAL_OPS)
        moved = sum(abs(count) for count in delta.values())
        if real_ops:
            reasons.append(f"{total}: {summary} (real opcodes {', '.join(real_ops)})")
            real = True
        elif bases <= NOISE_OPS and moved <= NOISE_BUDGET:
            reasons.append(f"noise {total}: {summary}")
        else:
            reasons.append(f"{total}: {summary}")
            real = True
    if real:
        return KernelDiff(after.stem, Verdict.REAL, reasons)
    if reasons:
        return KernelDiff(after.stem, Verdict.NOISE, reasons)
    if before.digest != after.digest:
        return KernelDiff(after.stem, Verdict.SAME_HISTOGRAM, ["same histogram, text differs"])
    return KernelDiff(after.stem, Verdict.IDENTICAL)


def classify_label(
    label: str, before: dict[str, KernelRecord], after: dict[str, KernelRecord]
) -> LabelDiff:
    kernels = []
    for stem in sorted(set(before) | set(after)):
        if stem not in before:
            kernels.append(KernelDiff(stem, Verdict.REAL, ["kernel only in b"]))
        elif stem not in after:
            kernels.append(KernelDiff(stem, Verdict.REAL, ["kernel only in a"]))
        else:
            kernels.append(classify_kernel(before[stem], after[stem]))
    verdict = max((kernel.verdict for kernel in kernels), default=Verdict.IDENTICAL)
    return LabelDiff(label, verdict, kernels)


def case_statuses(directory: Path) -> dict[str, str]:
    path = directory / "CASES.txt"
    if not path.exists():
        return {}
    statuses = {}
    for line in path.read_text().splitlines():
        status, case, *_ = line.split()
        statuses[case] = status
    return statuses


def compare(a: Path, b: Path) -> Report:
    before_labels, after_labels = set(labels(a)), set(labels(b))
    diffs = []
    for label in sorted(before_labels | after_labels):
        if label not in after_labels:
            diffs.append(LabelDiff(label, Verdict.REAL, reasons=[f"only in {a}"]))
        elif label not in before_labels:
            diffs.append(LabelDiff(label, Verdict.REAL, reasons=[f"only in {b}"]))
        else:
            diffs.append(classify_label(label, read_label(a, label), read_label(b, label)))
    problems = []
    before_cases, after_cases = case_statuses(a), case_statuses(b)
    for case in sorted(set(before_cases) | set(after_cases)):
        old, new = before_cases.get(case, "absent"), after_cases.get(case, "absent")
        if old != new or old != "ok":
            problems.append(f"case {case}: {old} -> {new}")
    return Report(diffs, problems)


def render(report: Report, a: Path, b: Path, *, quiet: bool) -> str:
    lines = []
    for label in report.labels:
        if quiet and label.verdict <= Verdict.SAME_HISTOGRAM:
            continue
        lines.append(
            f"{label.verdict.marker} {label.label}" + "".join(f": {r}" for r in label.reasons)
        )
        for kernel in label.kernels:
            if kernel.verdict is Verdict.IDENTICAL:
                continue
            lines.append(f"    {kernel.verdict.marker} {kernel.stem}")
            lines.extend(f"        {reason}" for reason in kernel.reasons)
    lines.extend(report.case_problems)
    counts = ", ".join(f"{report.count(verdict)} {verdict.label}" for verdict in Verdict)
    lines.append(f"summary: {counts} ({a} vs {b})")
    return "\n".join(lines)
