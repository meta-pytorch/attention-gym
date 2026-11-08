"""Compile and run generated kernels; isolate each case in a child process.

A case is one ``KernelSpec``: its source is written to ``workdir/<kernel>.py`` and imported from
there (CuTeDSL needs real source), compiled once, launched ``spec.launches`` times, and checked
by ``oracle.check``. ``run_isolated`` runs a case in a fresh interpreter under a process-level
timeout, so a hang (the R15 progress oracle) or a device fault is a recorded outcome instead of
a stuck or poisoned test process.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# The directory holding this package, so a child interpreter imports ``warpdrive_fuzz``.
TEST_ROOT = REPO / "test"

# Outcome classes; everything but PASS is a candidate finding for a valid generated program.
PASS, FAIL, HANG, CRASH, REJECT, ERROR = "pass", "fail", "hang", "crash", "reject", "error"
TOOL = "tool"  # a wrapper (compute-sanitizer --error-exitcode) reported a finding


# Confirmed findings (test/warpdrive_fuzz/FINDINGS.md, runtime section): a non-passing case
# whose status and detail match is reported under the finding's id instead of as new.
KNOWN = {
    "F-A1": (REJECT, re.compile(r"mixed register budgets \{[^}]*work\.scheduler")),
}


def known_finding(res: CaseResult) -> str | None:
    for fid, (status, pattern) in KNOWN.items():
        if res.status == status and pattern.search(res.detail + "\n".join(res.failures)):
            return fid
    return None


@dataclass
class CaseResult:
    seed: int
    key: str
    status: str
    failures: list[str] = field(default_factory=list)
    detail: str = ""
    compile_s: float = 0.0
    run_s: float = 0.0
    source: str = ""


def write_case(spec, workdir: Path) -> Path:
    from warpdrive_fuzz import gen

    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / f"{gen.kernel_name(spec)}.py"
    text = gen.emit(spec)
    if not path.exists() or path.read_text() != text:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(text)
        tmp.replace(path)
    (workdir / f"{gen.kernel_name(spec)}.json").write_text(spec.to_json())
    return path


def load_kernel(spec, workdir: Path):
    from warpdrive_fuzz import gen

    path = write_case(spec, workdir)
    name = f"wdfuzz_gen_{spec.key()}"
    module = sys.modules.get(name)
    if module is None:
        mod_spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(mod_spec)
        sys.modules[name] = module
        mod_spec.loader.exec_module(module)
    op = getattr(module, gen.class_name(spec))()
    op.trace_protocol = spec.trace  # before the first collection (R16)
    op.alias_check = spec.alias_check
    if spec.iket:
        op.enable_iket(waits=spec.iket == 2)
    return op


def compile_kernel(spec, op):
    import cutlass
    from cutlass import Int32, cute

    from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
    from warpdrive_fuzz import gen

    i32 = cutlass.Int32
    chans = {f"src_{ch.name}": ch for ch in spec.channels}
    srcs = []
    for name, kind in gen.call_args(spec):
        if kind == "tma_src":
            ch = chans[name]
            srcs.append(make_fake_compact_tensor(i32, (ch.tma_tiles * ch.tma_rows, gen.TMA_COLS)))
        elif kind in ("mma_a", "mma_b"):
            rows = gen.MMA_M if kind == "mma_a" else spec.mma.n
            tiles = gen.mma_channels(spec)[0].tma_tiles
            srcs.append(make_fake_compact_tensor(cutlass.BFloat16, (tiles * rows, spec.mma.k)))
        elif kind == "rows":
            srcs.append(make_fake_compact_tensor(i32, (cute.sym_int(), 3)))
        elif kind == "vec":
            srcs.append(make_fake_compact_tensor(i32, (cute.sym_int(),), assumed_align=4))
        else:
            srcs.append(Int32(1))
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(i32, (cute.sym_int(),), assumed_align=4),
        make_fake_compact_tensor(i32, (cute.sym_int(),), assumed_align=4),
        make_fake_compact_tensor(i32, (cute.sym_int(),), assumed_align=4),
        *([Int32(1)] * 5),
        *srcs,
        name=gen.kernel_name(spec),
    )


def tma_source(ch, device):
    import torch

    from warpdrive_fuzz import gen, runtime_device

    tiles, rows, cols = ch.tma_tiles, ch.tma_rows, gen.TMA_COLS
    t = torch.arange(tiles).view(-1, 1, 1)
    r = torch.arange(rows).view(1, -1, 1)
    c = torch.arange(cols).view(1, 1, -1)
    vals = runtime_device.tma_value(t, r, c, cols, ch.salt)
    return vals.reshape(tiles * rows, cols).to(torch.int32).to(device).contiguous()


def mma_operand(spec, kind: str, device):
    """The bf16 A (``(tiles * 128, k)``) or B (``(tiles * n, k)``) source of the MMA subgraph:
    small integers, so every accumulator is exact in fp32."""
    import torch

    from warpdrive_fuzz import gen, runtime_device

    tiles = gen.mma_channels(spec)[0].tma_tiles
    rows = gen.MMA_M if kind == "mma_a" else spec.mma.n
    fn = runtime_device.mma_a if kind == "mma_a" else runtime_device.mma_b
    t = torch.arange(tiles).view(-1, 1, 1)
    r = torch.arange(rows).view(1, -1, 1)
    k = torch.arange(spec.mma.k).view(1, 1, -1)
    vals = fn(t, r, k, spec.mma.salt).reshape(tiles * rows, spec.mma.k)
    return vals.to(torch.bfloat16).to(device).contiguous()


def static_args(spec, device) -> dict:
    """Entry-point tensors that do not depend on the runtime variant."""
    from warpdrive_fuzz import gen

    chans = {f"src_{ch.name}": ch for ch in spec.channels}
    out = {}
    for name, kind in gen.call_args(spec):
        if kind == "tma_src":
            out[name] = tma_source(chans[name], device)
        elif kind in ("mma_a", "mma_b"):
            out[name] = mma_operand(spec, kind, device)
    return out


class MappingArgs:
    """The table/packed mapping's tensors for one launch of a variant. ``reset()`` restores
    the pre-launch state (stream-ordered). With PDL the table and length start wrong (every
    row an out-of-space item, the length the full capacity) and the producer kernel fixes them,
    so a consumer that decoded before ``griddepcontrol.wait`` returned is caught."""

    def __init__(self, s, device):
        import torch

        from warpdrive_fuzz import gen, oracle

        self.s, self.tensors, self.initial = s, {}, {}
        if s is None or s.mapping == "raster":
            return
        items, _ = oracle.space(s)  # the table mapping's empty space keeps one unread row
        rows = torch.tensor(items or [oracle.garbage_row(s)], dtype=torch.int32).view(-1, 3)
        if s.mapping == "table":
            init = {"table": rows, "bad": torch.zeros(1, dtype=torch.int32)}
        elif not s.pdl:
            init = {"table": rows, "length": torch.tensor([len(gen.packed_rows(s))])}
        else:
            init = {
                "table": torch.full_like(rows, 99),
                "length": torch.tensor([rows.shape[0]]),
                "stage": rows,
            }
        self.tensors = {k: v.to(device, torch.int32) for k, v in init.items()}
        self.initial = {k: v.clone() for k, v in self.tensors.items()}

    def reset(self):
        for k, v in self.tensors.items():
            v.copy_(self.initial[k])

    def args(self) -> dict:
        from warpdrive_fuzz import gen

        out = dict(self.tensors)
        if self.s is not None and self.s.pdl:
            out["plen"] = len(gen.packed_rows(self.s))
        return out

    def bad(self) -> int:
        return int(self.tensors["bad"][0]) if "bad" in self.tensors else 0


def run_case(spec, workdir: Path) -> CaseResult:
    """Compile once, then launch the spec and each runtime-shape variant with the same binary,
    checking every launch (a hang hangs the caller)."""
    import torch

    from attn_gym.warpdrive import ProtocolError
    from warpdrive_fuzz import gen

    result = CaseResult(spec.seed, spec.key(), PASS)
    device = torch.device("cuda")
    t0 = time.perf_counter()
    try:
        op = load_kernel(spec, workdir)
        fn = compile_kernel(spec, op)
    except Exception as exc:  # noqa: BLE001 — a valid program the layer refused is an outcome
        protocol = isinstance(exc, (ProtocolError, ValueError)) or "ProtocolError" in repr(exc)
        result.status = REJECT if protocol else ERROR
        result.detail = "".join(traceback.format_exception(exc))[-4000:]
        return result
    result.compile_s = time.perf_counter() - t0
    _phase("compiled")
    srcs = static_args(spec, device)
    failures, checked = [], 0
    t1 = time.perf_counter()
    for v, view in enumerate(gen.variant_specs(spec)):
        rep = _launch_and_check(view, op, fn, device, srcs, f"variant {v}" if v else "")
        failures += rep.failures
        checked += rep.checked
    result.run_s = time.perf_counter() - t1
    failures += describe_checks(spec, op.describe())
    if failures:
        result.status = FAIL
        result.failures = failures[:50]
    result.detail = f"{checked} checks"
    return result


def describe_checks(spec, text: str) -> list[str]:
    """What ``describe()`` must report after the compile (R16, U8, the alias diagnostic)."""
    out = []
    if spec.trace and "Traced protocol" not in text:
        out.append("trace_protocol=True compiled, but describe() lists no traced protocol")
    kinds = {c.unchecked for c in spec.channels if c.unchecked}
    kinds |= {"alias"} if any(c.alias_opt_out for c in spec.channels) else set()
    for kind in sorted(kinds):
        lines = [ln for ln in text.splitlines() if ln.strip().startswith(f"unchecked {kind} @")]
        if not lines:
            out.append(f"describe() does not list the unchecked({kind!r}) opt-out")
        elif (kind != "alias" or spec.alias_check != "off") and all(
            "stale opt-out" in ln for ln in lines
        ):
            out.append(f"unchecked({kind!r}) let nothing through: {lines[0].strip()}")
    overlaps = [c for c in spec.channels if c.storage == "overlap"]
    if overlaps and spec.alias_check != "off":
        if "Alias diagnostic" not in text:
            out.append("alias_check on with declared overlaps, but no alias diagnostic report")
        else:
            report = text.split("Alias diagnostic", 1)[1].split("\n\n", 1)[0].splitlines()[1:]
            hazards = [ln.strip() for ln in report if "potential hazard" in ln]
            if hazards:
                out.append(f"alias diagnostic: potential hazard on an ordered overlap: {hazards}")
    return out


class _LaunchState:
    """One stream's buffers for repeated launches of the compiled kernel."""

    def __init__(self, spec, op, device, rows: int, hits_len: int):
        import torch

        from warpdrive_fuzz import oracle

        s = spec.stream
        self.hits = torch.zeros(hits_len + 64, dtype=torch.int32, device=device)
        if s is not None:
            self.ws, _ = op.work.launch_args(device)
        else:
            self.ws = torch.zeros(4, dtype=torch.int32, device=device)
        self.init = torch.from_numpy(oracle.initial_clog(spec, rows).reshape(-1)).to(device)
        self.clog = torch.empty_like(self.init)
        self.mapping = MappingArgs(s, device)


def _launch_and_check(spec, op, fn, device, srcs, tag: str):
    """Launch ``spec`` (its runtime extents, cap and iterations) ``spec.launches`` times, plus
    two CUDA-graph replays when ``spec.graph`` and two concurrent launches on two streams when
    ``spec.concurrent``; check each against the model."""
    import numpy as np
    import torch

    from warpdrive_fuzz import gen, oracle

    lay = gen.Layout.of(spec)
    ncta = len(oracle.ctas(spec))
    rows = ncta + (8 if ncta > 4096 else ncta + 8)  # slack rows catch an oversized grid
    s = spec.stream
    total = s.total if s is not None else 0
    n_inst = sum(r.instances for r in spec.roles)
    hits_len = max(n_inst * spec.cs * total, 1)
    extents, max_ctas = (s.extents, s.max_ctas) if s is not None else ((1, 1, 1), 1)
    gx = oracle.expected_grid(spec)[0]
    prefix = f"{tag} " if tag else ""
    st = _LaunchState(spec, op, device, rows, hits_len)

    def collect(st):
        return oracle.Launch(
            st.clog.cpu().numpy().reshape(rows, lay.per_cta).copy(),
            st.hits.cpu().numpy().copy(),
            st.ws.cpu().numpy().copy(),
            op.work.timeline(st.ws, gx) if s is not None and s.record else None,
            st.mapping.bad(),
        )

    def launch(st):
        st.clog.copy_(st.init)
        st.hits.zero_()
        st.mapping.reset()
        extra = {**srcs, **st.mapping.args()}
        args = [extra[name] for name, _ in gen.call_args(spec)]
        fn(st.clog, st.hits, st.ws, spec.iters, *extents, max_ctas, *args)

    runs, states = [], [st]
    for k in range(spec.launches):
        if s is not None and s.record:  # refills the log with -1; [claims, finished] stay
            st.ws, _ = op.work.launch_args(device, max_ctas=gx)
        torch.cuda.synchronize()
        _phase(f"launch {prefix}{k}")
        launch(st)
        torch.cuda.synchronize()
        _phase(f"launched {prefix}{k}")
        runs.append(collect(st))
    if spec.graph:  # R14: ordered graph replays reuse the self-resetting workspace
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(st)
        for k in range(2):
            _phase(f"launch {prefix}graph {k}")
            graph.replay()
            torch.cuda.synchronize()
            _phase(f"launched {prefix}graph {k}")
            runs.append(collect(st))
    if spec.concurrent:
        # Two launches in flight at once on two streams. launch_args() caches one workspace
        # per (device, stream), so each gets its own: the documented way to run concurrently.
        # Schedules that never touch a workspace share one placeholder per device.
        streams = [torch.cuda.Stream(device) for _ in range(2)]
        pair = []
        for strm in streams:
            with torch.cuda.stream(strm):
                pair.append(_LaunchState(spec, op, device, rows, hits_len))
                if s is not None and s.record:
                    pair[-1].ws, _ = op.work.launch_args(device, max_ctas=gx)
        shared = s is not None and pair[0].ws.data_ptr() == pair[1].ws.data_ptr()
        if shared and (op.work.schedule.uses_workspace or s.record):
            raise AssertionError("launch_args() returned one workspace for two streams")
        torch.cuda.synchronize()
        _phase(f"launch {prefix}concurrent")
        for strm, cst in zip(streams, pair, strict=True):
            with torch.cuda.stream(strm):
                launch(cst)
        torch.cuda.synchronize()
        _phase(f"launched {prefix}concurrent")
        runs += [collect(cst) for cst in pair]
        states += pair
    _phase(f"checking {prefix}")
    rep = oracle.check(spec, runs)
    for state in states:
        if np.any(state.hits.cpu().numpy()[hits_len:] != 0):
            rep.fail("hits written past the item range")
    if tag:
        rep.failures = [f"{tag} {extents}/{max_ctas}/{spec.iters}: {f}" for f in rep.failures]
    return rep


PHASE = "@@phase "
PHASE_COMPILED = (PHASE + "compiled").encode()
COMPILE_BUDGET_S = 900.0  # compile/startup allowance before the hang timer starts


def _phase(name: str):
    """Progress marker on stdout: a timeout after ``launch k`` and before ``launched k`` is a
    device hang; any other timeout is the harness being slow."""
    print(PHASE + name, flush=True)


def last_phase(output: str) -> str:
    phases = [line[len(PHASE) :] for line in output.splitlines() if line.startswith(PHASE)]
    return phases[-1] if phases else "start"


def run_isolated(
    spec,
    workdir: Path,
    timeout: float = 120.0,
    env: dict | None = None,
    wrapper: list[str] | None = None,
) -> CaseResult:
    """Run one case in a child interpreter; ``timeout`` seconds without exit is a hang."""
    from warpdrive_fuzz import gen

    write_case(spec, workdir)
    spec_path = workdir / f"{gen.kernel_name(spec)}.json"
    out_path = workdir / f"{gen.kernel_name(spec)}.result.{os.getpid()}.json"
    out_path.unlink(missing_ok=True)
    cmd = [
        *(wrapper or []),
        sys.executable,
        "-m",
        "warpdrive_fuzz.runtime",
        str(spec_path),
        str(out_path),
        str(workdir),
    ]
    child_env = {**os.environ, **(env or {})}
    child_env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(TEST_ROOT), str(REPO), child_env.get("PYTHONPATH", "")) if p
    )
    proc = subprocess.Popen(
        cmd,
        cwd=REPO,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    # The hang timer starts once the child reports "compiled": a slow compile (file cache off,
    # loaded host) is not a hang, and must not turn a detected mutant into a harness error.
    chunks: list[bytes] = []
    reader = threading.Thread(target=lambda: chunks.extend(iter(proc.stdout.readline, b"")))
    reader.start()
    started, compiled_at = time.monotonic(), None
    while proc.poll() is None:
        now = time.monotonic()
        if compiled_at is None and PHASE_COMPILED in b"".join(chunks):
            compiled_at = now
        deadline = (compiled_at + timeout) if compiled_at else (started + COMPILE_BUDGET_S)
        if now > deadline:
            break
        time.sleep(0.05)
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        reader.join()
        text = b"".join(chunks).decode(errors="replace")
        phase = last_phase(text)
        status = HANG if phase.startswith("launch ") else ERROR
        return CaseResult(
            spec.seed,
            spec.key(),
            status,
            detail=f"no exit in {timeout}s during {phase!r}\n{text[-2000:]}",
            source=str(workdir / f"{gen.kernel_name(spec)}.py"),
        )
    reader.join()
    text = b"".join(chunks).decode(errors="replace")
    if out_path.exists():
        data = json.loads(out_path.read_text())
        out_path.unlink()
        res = CaseResult(**data)
        if wrapper:  # keep the tool's verdict lines (sanitizer summaries, CUTracer status)
            summary = [
                ln for ln in text.splitlines() if "SUMMARY" in ln or "cutracer" in ln.lower()
            ]
            res.detail += "\n" + "\n".join(summary[-20:])
        if res.status == PASS and wrapper and proc.returncode != 0:
            res.status = TOOL
            res.detail = f"wrapper exit {proc.returncode}\n{text[-4000:]}"
        elif res.status != PASS and not res.detail:
            res.detail = text[-2000:]
        return res
    return CaseResult(
        spec.seed,
        spec.key(),
        CRASH,
        detail=f"exit {proc.returncode}\n{text[-3000:]}",
        source=str(workdir / f"{gen.kernel_name(spec)}.py"),
    )


def _child(spec_path: str, out_path: str, workdir: str) -> int:
    from warpdrive_fuzz import gen

    spec = gen.KernelSpec.from_json(Path(spec_path).read_text())
    mutant = os.environ.get("WDFUZZ_MUTANT")
    if mutant:
        from warpdrive_fuzz import runtime_mutants

        runtime_mutants.apply(mutant)
    try:
        res = run_case(spec, Path(workdir))
    except Exception as exc:  # noqa: BLE001 — device faults and harness errors are outcomes
        res = CaseResult(
            spec.seed, spec.key(), ERROR, detail="".join(traceback.format_exception(exc))[-4000:]
        )
    res.source = str(Path(workdir) / f"{gen.kernel_name(spec)}.py")
    Path(out_path).write_text(json.dumps(asdict(res)))
    return 0


if __name__ == "__main__":
    args = sys.argv[1:4]
    sys.argv = sys.argv[:1]  # CuTeDSL re-parses sys.argv at import
    sys.exit(_child(*args))
