"""GPU-free parsing of cubins, nvdisasm text, cuobjdump resource usage and host IR, and the
per-label snapshot files.

A label is one ``cute.compile`` artifact (``<case>__<compiled fn>_<k>``); it holds one kernel
per ``.text`` section. Per label the snapshot stores ``.sass`` (nvdisasm, the source of truth
for everything but resources), ``.res`` (``<kernel> REG=.. STACK=.. SHARED=.. LOCAL=.. DSMEM=..
MBAR=0x..,..``), ``.fns`` (``<kernel> <symbol>``) and two derived views for reading and
grepping: ``.ops`` (``<kernel> <OPCODE.MODS> <count>``) and ``.layout`` (``<kernel> <OPCODE.MODS>
<smem immediate> <count>``). Kernels are keyed by a short stem derived from the symbol so two
trees whose mangled names differ still pair up.

Per kernel the parser keeps the opcode histogram keyed by the full mnemonic (``LDS.128``,
``SYNCS.PHASECHK.TRANS64.TRYWAIT``), the shared-memory *layout*: the multiset of ``(mnemonic,
immediate offset)`` of every ``LDS``/``STS``/``LDSM``/``STSM``/``SYNCS*`` instruction, the sorted
``SYNCS.EXCH`` (mbarrier init) offsets, and a digest of the normalized instruction text
(addresses and label numbers stripped) that tells byte-equal code from a re-scheduled one.
"""

from __future__ import annotations

import hashlib
import re
import struct
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

RESOURCE_FIELDS = ("REG", "STACK", "SHARED", "LOCAL", "DSMEM")
UNKNOWN = "?"
LAYOUT_OPS = frozenset({"LDS", "STS", "LDSM", "STSM", "SYNCS"})

_ELF_MAGIC = b"\x7fELF\x02\x01"
_EM_CUDA = 190
_SECTION = re.compile(r"^\.text\.(\S+):")
_INSTRUCTION = re.compile(
    r"^\s+/\*[0-9a-f]+\*/\s+((?:@!?U?P[0-9T]\s+)?)([A-Z][A-Z0-9_]*)((?:\.[A-Z0-9_]+)*)\s*(.*)"
)
_LABEL = re.compile(r"^\.L_x_\d+:")
_LABEL_REF = re.compile(r"\.L_x_\d+")
_ADDRESS = re.compile(r"\[([^\]]*)\]")
_RESOURCE = re.compile(r"Function ([^:\s]+):\s+REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)")
# The stem ends at the first mangled argument: a type/config name (``_GdnCfg``), a struct or
# tensor repr (``_cutlasscutecorestructobjectat``, ``_tensorptr``), ``__`` or a trailing index.
_STEM = re.compile(r"frost_([a-z0-9_]+?)(?:_[A-Z]|__|_cutlass|_tensor|_\d+$|$)")
_LAUNCH_CONFIG = r"!llvm\.struct<\(array<3 x i32>, array<3 x i32>, i64, ptr, ptr, i32\)>"
_CONSTANT = re.compile(r"^\s*(%\w+) = llvm\.mlir\.constant\((-?\d+) : i64\)", re.MULTILINE)
_SMEM_SLOT = re.compile(
    r"^\s*(%\w+) = llvm\.getelementptr %\w+\[0, 2\] : \(!llvm\.ptr\) -> !llvm\.ptr, "
    + _LAUNCH_CONFIG,
    re.MULTILINE,
)
_STORE_I64 = re.compile(r"^\s*llvm\.store (%\w+), (%\w+) : i64, !llvm\.ptr", re.MULTILINE)
_KERNEL_ADDRESS = re.compile(r"llvm\.mlir\.addressof @kernels_(\w+)")
_HOST_FUNCTION = re.compile(
    r"^  llvm\.func (?:\w+ )*@\w+\([^\n]*\{\n.*?^  }", re.MULTILINE | re.DOTALL
)


@dataclass
class KernelCode:
    """Everything the parser reads from one ``.text`` section."""

    ops: Counter = field(default_factory=Counter)
    layout: Counter = field(default_factory=Counter)
    mbarriers: list[int] = field(default_factory=list)
    digest: str = ""


@dataclass
class KernelRecord:
    """One kernel of one label: parsed code plus the launch resources."""

    stem: str
    symbol: str
    ops: Counter = field(default_factory=Counter)
    resources: dict[str, str] = field(default_factory=dict)
    mbarriers: list[int] = field(default_factory=list)
    layout: Counter = field(default_factory=Counter)
    digest: str = ""

    @property
    def instructions(self) -> int:
        return sum(self.ops.values())


def base_opcode(mnemonic: str) -> str:
    return mnemonic.partition(".")[0]


def carve_cubins(blob: bytes):
    """Yield every embedded cubin (64-bit little-endian ELF, ``e_machine == EM_CUDA``) of a host
    object, sized from its section and program header tables."""
    for match in re.finditer(re.escape(_ELF_MAGIC), blob):
        start = match.start()
        if len(blob) < start + 64:
            continue
        (machine,) = struct.unpack_from("<H", blob, start + 18)
        if machine != _EM_CUDA:
            continue
        phoff, shoff = struct.unpack_from("<QQ", blob, start + 32)
        _, phentsize, phnum, shentsize, shnum = struct.unpack_from("<HHHHH", blob, start + 52)
        end = max(shoff + shentsize * shnum, phoff + phentsize * phnum)
        yield blob[start : start + end]


def kernel_stem(symbol: str) -> str:
    """Short kernel key: the ``frost_<name>`` stem, else the compile-name prefix before
    ``_kernel_cutlass``, else the leading 40 characters."""
    match = _STEM.search(symbol)
    if match:
        return match.group(1)
    prefix, sep, _ = symbol.partition("_kernel_cutlass")
    return prefix if sep else symbol[:40]


def address_offset(operands: str) -> int | None:
    """Immediate byte offset of the first ``[...]`` shared-memory operand (0 without one)."""
    match = _ADDRESS.search(operands)
    if match is None:
        return None
    immediates = re.findall(r"(?<![A-Za-z])(-?0x[0-9a-fA-F]+)", match.group(1))
    return int(immediates[-1], 16) if immediates else 0


def parse_sass(text: str) -> dict[str, KernelCode]:
    """Per ``.text`` section: full-mnemonic histogram, shared-memory layout, sorted mbarrier
    init offsets and the normalized-text digest."""
    kernels: dict[str, KernelCode] = {}
    hashers: dict[str, hashlib._Hash] = {}
    current = hasher = None
    for line in text.splitlines():
        section = _SECTION.match(line)
        if section:
            current = kernels.setdefault(section.group(1), KernelCode())
            hasher = hashers.setdefault(section.group(1), hashlib.sha256())
            continue
        if current is None:
            continue
        if _LABEL.match(line):
            hasher.update(b".L:\n")
            continue
        instruction = _INSTRUCTION.match(line)
        if instruction is None:
            continue
        predicate, mnemonic, modifiers, operands = instruction.groups()
        full = mnemonic + modifiers
        hasher.update(_LABEL_REF.sub(".L", f"{predicate}{full} {operands}").encode() + b"\n")
        current.ops[full] += 1
        if mnemonic in LAYOUT_OPS:
            offset = address_offset(operands)
            if offset is not None:
                current.layout[(full, offset)] += 1
                if full.startswith("SYNCS.EXCH"):
                    current.mbarriers.append(offset)
    for symbol, code in kernels.items():
        code.mbarriers.sort()
        code.digest = hashers[symbol].hexdigest()[:16]
    return kernels


def parse_resource_usage(text: str) -> dict[str, dict[str, str]]:
    """``cuobjdump --dump-resource-usage`` output -> ``{symbol: {REG, STACK, SHARED, LOCAL}}``."""
    return {
        symbol: dict(zip(RESOURCE_FIELDS[:4], values))
        for symbol, *values in _RESOURCE.findall(text)
    }


def parse_dynamic_smem(ir_text: str) -> dict[str, str]:
    """Dynamic shared-memory bytes each kernel is launched with, from the host LLVM IR.

    The host stores an ``i64`` constant into slot 2 (``sharedMemBytes``) of the
    ``cudaLaunchConfig_t`` it passes to ``_cudaLaunchKernelEx``, then takes the
    ``@kernels_<symbol>`` address of the kernel it launches; each store pairs with the next
    address taken. Non-constant sizes read ``?``.
    """
    sizes: dict[str, str] = {}
    for function in _HOST_FUNCTION.findall(ir_text):
        constants = dict(_CONSTANT.findall(function))
        slots = {match.group(1) for match in _SMEM_SLOT.finditer(function)}
        stores = [
            (match.start(), match.group(1))
            for match in _STORE_I64.finditer(function)
            if match.group(2) in slots
        ]
        kernels = [(match.start(), match.group(1)) for match in _KERNEL_ADDRESS.finditer(function)]
        for position, value in stores:
            following = [symbol for at, symbol in kernels if at > position]
            if following:
                sizes.setdefault(following[0], constants.get(value, UNKNOWN))
    return sizes


def build_records(
    sass_text: str, resource_text: str, dynamic_smem: dict[str, str] | None = None
) -> list[KernelRecord]:
    """Combine the parsed views of one cubin into records sorted by stem; repeated stems get
    ``#2``, ``#3``... in symbol order."""
    parsed = parse_sass(sass_text)
    usage = parse_resource_usage(resource_text)
    dynamic_smem = dynamic_smem or {}
    symbols = list(parsed) + [symbol for symbol in usage if symbol not in parsed]
    seen: Counter = Counter()
    records = []
    for symbol in symbols:
        stem = kernel_stem(symbol)
        seen[stem] += 1
        if seen[stem] > 1:
            stem = f"{stem}#{seen[stem]}"
        code = parsed.get(symbol, KernelCode())
        resources = dict(usage.get(symbol, {}))
        resources["DSMEM"] = dynamic_smem.get(symbol, UNKNOWN)
        records.append(
            KernelRecord(
                stem, symbol, code.ops, resources, code.mbarriers, code.layout, code.digest
            )
        )
    return sorted(records, key=lambda record: record.stem)


def format_mbarriers(offsets: list[int]) -> str:
    return ",".join(hex(offset) for offset in offsets) if offsets else "-"


def write_label(out: Path, label: str, records: list[KernelRecord], sass_text: str) -> None:
    base = out / label
    base.with_suffix(".sass").write_text(sass_text)
    base.with_suffix(".fns").write_text(
        "".join(f"{record.stem} {record.symbol}\n" for record in records)
    )
    base.with_suffix(".ops").write_text(
        "".join(
            f"{record.stem} {op} {count}\n"
            for record in records
            for op, count in sorted(record.ops.items())
        )
    )
    base.with_suffix(".layout").write_text(
        "".join(
            f"{record.stem} {op} {offset:#x} {count}\n"
            for record in records
            for (op, offset), count in sorted(record.layout.items())
        )
    )
    base.with_suffix(".res").write_text(
        "".join(
            f"{record.stem} "
            + " ".join(f"{name}={record.resources.get(name, UNKNOWN)}" for name in RESOURCE_FIELDS)
            + f" MBAR={format_mbarriers(record.mbarriers)}\n"
            for record in records
        )
    )


def read_resources(path: Path) -> dict[str, dict[str, str]]:
    """``.res`` -> ``{stem: {REG, STACK, SHARED, LOCAL, DSMEM}}`` (``MBAR`` is re-derived)."""
    resources: dict[str, dict[str, str]] = {}
    for line in path.read_text().splitlines():
        stem, *tokens = line.split()
        fields = dict(token.partition("=")[::2] for token in tokens)
        fields.pop("MBAR", None)
        resources[stem] = fields
    return resources


def read_label(directory: Path, label: str) -> dict[str, KernelRecord]:
    """Load one label's records (keyed by stem): code re-parsed from ``.sass`` so older
    snapshots classify under the current rules, resources from ``.res`` joined through the
    ``.fns`` symbols so a stem-rule change does not orphan them."""
    base = directory / label
    records = {
        record.stem: record for record in build_records(base.with_suffix(".sass").read_text(), "")
    }
    by_symbol = {record.symbol: record for record in records.values()}
    symbols = dict(
        line.partition(" ")[::2] for line in base.with_suffix(".fns").read_text().splitlines()
    )
    for stem, fields in read_resources(base.with_suffix(".res")).items():
        record = by_symbol.get(symbols.get(stem, stem))
        if record is None:
            record = records.setdefault(stem, KernelRecord(stem, symbols.get(stem, stem)))
        record.resources = {name: fields.get(name, UNKNOWN) for name in RESOURCE_FIELDS}
    return records


def labels(directory: Path) -> list[str]:
    return sorted(path.stem for path in directory.glob("*.res"))
