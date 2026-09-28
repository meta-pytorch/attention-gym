"""Hook ``cute.compile`` and write every compiled cubin into a snapshot directory.

The hook wraps ``cutlass.cute.compile`` (both ``cute.compile(...)`` and
``cute.compile[options](...)``), which ``attn_gym._backends.cute.utils.compile_tvm_ffi`` resolves
at call time, so every ``jit_cache`` compile of the drivers passes through it. Each artifact is
re-exported as a host object, its cubins carved out and disassembled (``nvdisasm``,
``cuobjdump --dump-resource-usage``), and the launch's dynamic SMEM size read from the host IR.
Compile caches must be fresh (see ``cli.py``) or nothing compiles.

The export mirrors production: ``cute.compile`` lowers the device code to a cubin embedded in
``ir_module`` (the CuTeDSL ``OptLevel`` applies there), and both ``jit_cache`` (``export_to_c``)
and the in-process ``BinaryExecutionEngine`` then emit the host object with
``export_module_to_bytes(..., opt_level=3)``; ``opt_level`` is the host LLVM level and does not
touch the cubin. ``cli.py`` verifies this per run by carving the cubins out of the ``.o`` files
``jit_cache`` published and matching them byte for byte against the snapshot.
"""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from cutlass import cute
from cutlass._mlir._mlir_libs._cutlass_ir import _aot_support

from .sass import build_records, carve_cubins, parse_dynamic_smem, write_label


def compiled_name(fn) -> str:
    fn = getattr(fn, "func", fn)  # functools.partial
    for attr in ("__qualname__", "__name__"):
        name = getattr(fn, attr, None)
        if isinstance(name, str):
            return re.sub(r"[^A-Za-z0-9_]+", "_", name.replace("<locals>.", "")).strip("_")
    return type(fn).__name__


@dataclass
class Capture:
    """Snapshot writer; ``case`` names the current driver case for label attribution."""

    out: Path
    case: str = "_nocase"
    artifacts: list[tuple[str, str, int]] = field(default_factory=list)
    counts: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))
    log: list[str] = field(default_factory=list)

    def install(self) -> None:
        cute.compile = _HookedCompile(cute.compile, self)

    def record(self, fn, compiled) -> None:
        name = compiled_name(fn)
        index = self.counts[(self.case, name)]
        self.counts[(self.case, name)] += 1
        label = f"{self.case}__{name}_{index}"
        blob = bytes(
            _aot_support.export_module_to_bytes(
                compiled.ir_module, format="o", opt_level=3, enable_pic=True
            )
        )
        dynamic_smem = parse_dynamic_smem(str(compiled.ir_module))
        cubins = list(carve_cubins(blob))
        for part, cubin in enumerate(cubins):
            self.write(label if len(cubins) == 1 else f"{label}_c{part}", cubin, dynamic_smem)
        if not cubins:
            self.say(f"  [sass] {label}: no embedded cubin")
        self.artifacts.append((self.case, label, len(cubins)))

    def write(self, label: str, cubin: bytes, dynamic_smem: dict[str, str]) -> None:
        path = self.out / f"{label}.cubin"
        path.write_bytes(cubin)
        sass_text = _run("nvdisasm", "-c", str(path))
        resources = _run("cuobjdump", "--dump-resource-usage", str(path))
        records = build_records(sass_text, resources, dynamic_smem)
        write_label(self.out, label, records, sass_text)
        kernels = " ".join(
            f"{r.stem}:{r.instructions}/{r.resources.get('REG')}r/{r.resources.get('DSMEM')}s"
            for r in records
        )
        self.say(f"  [sass] {label:60s} {kernels}")

    def say(self, line: str) -> None:
        self.log.append(line)
        print(line, flush=True)


class _HookedCompile:
    def __init__(self, inner, capture: Capture):
        self._inner = inner
        self._capture = capture

    def __getitem__(self, options):
        return _HookedCompile(self._inner[options], self._capture)

    def __call__(self, fn, *args, **kwargs):
        compiled = self._inner(fn, *args, **kwargs)
        self._capture.record(fn, compiled)
        return compiled

    def __getattr__(self, attr):
        return getattr(self._inner, attr)


def _run(*command: str) -> str:
    return subprocess.run(command, capture_output=True, text=True, check=True).stdout
