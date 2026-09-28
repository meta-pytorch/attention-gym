"""A/B benchmark gate for the vendored cuDNN GDN/KDA kernels.

Two source trees (``--new``, ``--base``) are activated in child processes that share this
interpreter, so torch / triton / cutlass versions are identical on both sides. Each suite runs
``--rounds`` process rounds with alternating tree order; every child times the whole op under
fixed-pointer CUDA-graph replay (3 samples x 20 replays per event pair, in-process median of the
per-replay time), and the reported cell is the median of the process-round medians. Round 0 also
records the kernel launch set, saves outputs (and input gradients) for a cross-tree correctness
check, and the parent compares them (bitwise or relative L2 per case) and deletes the dumps
unless ``--keep-dumps``. ``report`` renders ``report.md`` with new/base ratios and a threshold
verdict; a changed launch set or a failed correctness row fails the gate as well.

Suites: ``gdn`` (``chunk_gdn``), ``kda`` (``chunk_kda``), ``summary`` (KDA context-parallel
state summaries), ``paged`` (``paged_chunk_{gdn,kda}`` forward with resumed routes).

Reserve the GPU around the run (``gpu-run --timeout 900 auto -- ...``); the gate never claims
one itself. Examples::

    python -m tools.cudnn_fe.bench run --suite gdn kda --new /path/to/candidate \\
        --base main=/path/to/main --out gate_out
    python -m tools.cudnn_fe.bench run --suite paged --small --new . --base . --out /tmp/self
    python -m tools.cudnn_fe.bench report --out gate_out --threshold 0.02
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .._common import (
    REPO_ROOT,
    D,
    Workload,
    activate_tree,
    check_tree,
    chunk_op,
    git_head,
    op_inputs,
    paged_chunk_op,
    parse_tree,
    run_logged,
    timestamp,
)

SUITES = ("gdn", "kda", "summary", "paged")
SAMPLES, REPLAYS, WARMUP = 3, 20, 5
CONTRACT = (
    f"fixed-pointer CUDA graph replay, factory DVFS, median of {SAMPLES} samples x {REPLAYS} "
    "replays per event pair"
)


@dataclass(frozen=True)
class Case:
    suite: str
    name: str
    work: Workload
    check: str  # "bitwise" or "rel"
    small: bool = False


CASES = (
    # GDN, Qwen3.5-27B shapes (HK=16, H=48); NEW vs the previous vendored generation is bitwise.
    Case("gdn", "prefill_2048", Workload("gdn", (2048,), 16, 48), "bitwise", small=True),
    Case("gdn", "prefill_8x2048", Workload("gdn", (2048,) * 8, 16, 48), "bitwise"),
    Case("gdn", "train_10x4096", Workload("gdn", (4096,) * 10, 16, 48), "bitwise"),
    Case("gdn", "train_1x40960", Workload("gdn", (40960,), 16, 48), "bitwise"),
    # KDA migration smoke grid: small prep, medium low occupancy, long chain, packed tails.
    Case("kda", "T2048_B1_H8", Workload("kda", (2048,), 8, 8), "rel", small=True),
    Case("kda", "T8192_B1_H48", Workload("kda", (8192,), 48, 48), "rel"),
    Case("kda", "T32768_B1_H48", Workload("kda", (32768,), 48, 48), "rel"),
    Case("kda", "T2305_B4_H8", Workload("kda", (257, 0, 511, 1537), 8, 8), "rel"),
    # KDA CP affine summaries: forward [B;A] and reverse [C;R] with explicit bounds.
    Case("summary", "T4096_B2_H8", Workload("kda", (2048,) * 2, 8, 8), "rel", small=True),
    Case("summary", "T16384_B4_H16", Workload("kda", (4096,) * 4, 16, 16), "rel"),
    Case("summary", "T16384_B1_H16", Workload("kda", (16384,), 16, 16), "rel"),
    Case("summary", "T16384_B8_H32", Workload("kda", (2048,) * 8, 32, 32), "rel"),
    # Paged forward, every route resumed; GDN uses HK query/key heads, KDA has HK == H.
    Case("paged", "gdn_paged_4x512", Workload("gdn", (512,) * 4, 16, 48), "bitwise", small=True),
    Case("paged", "gdn_paged_8x2048", Workload("gdn", (2048,) * 8, 16, 48), "bitwise"),
    Case("paged", "gdn_paged_32x512", Workload("gdn", (512,) * 32, 16, 48), "bitwise"),
    Case("paged", "kda_paged_4x512", Workload("kda", (512,) * 4, 8, 8), "rel", small=True),
    Case("paged", "kda_paged_8x2048", Workload("kda", (2048,) * 8, 8, 8), "rel"),
    Case("paged", "kda_paged_4x4096", Workload("kda", (4096,) * 4, 48, 48), "rel"),
)


def select(suite: str, names: list[str], small: bool) -> list[Case]:
    cases = [c for c in CASES if c.suite == suite]
    if names:
        cases = [c for c in cases if c.name in names]
    elif small:
        cases = [c for c in cases if c.small]
    if not cases:
        raise SystemExit(f"no {suite} case matches {names or 'small preset'}")
    return cases


# ----------------------------------------------------------------------------- child (timing)


def time_graph(fn, samples=SAMPLES, replays=REPLAYS, warmup=WARMUP) -> list[float]:
    """Per sample, the mean replay time in µs of ``replays`` back-to-back replays of a
    fixed-pointer CUDA graph of ``fn`` between one event pair (host launch gaps excluded)."""
    import torch

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    times = []
    for _ in range(samples):
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / replays)
    return times


def kernel_set(fn) -> dict[str, float]:
    """Full kernel name -> summed device time (µs) of one eager run of ``fn``."""
    import torch

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    kernels: dict[str, float] = {}
    for event in prof.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernels[event.name] = kernels.get(event.name, 0.0) + event.device_time
    return kernels


def op_phases(case: Case):
    """chunk_gdn / chunk_kda: forward, backward, forward_backward on autograd leaves."""
    import torch

    ops = op_inputs(case.work, seed=130)
    inputs = tuple(x.requires_grad_() for x in ops.leaves)
    torch.manual_seed(130)  # op_inputs draws from its own generator; seed the cotangent too
    do = torch.randn_like(ops.v)
    op = chunk_op(case.work.kind)

    def forward():
        return op(*inputs, cu_seqlens=ops.cu_seqlens, kernel_options={"backend": "cudnn"})[0]

    output = forward()

    def backward():
        return torch.autograd.grad(output, inputs, do, retain_graph=True)

    def forward_backward():
        return torch.autograd.grad(forward(), inputs, do)

    grads = backward()
    dump = {"o": output.detach(), **dict(zip(("dq", "dk", "dv", "dg", "db"), grads))}
    phases = {"forward": forward, "backward": backward, "forward_backward": forward_backward}
    return phases, dump


def summary_phases(case: Case):
    import torch

    from attn_gym.linear._delta_rule.cudnn.state_summary import (
        build_cudnn_state_grad_summaries,
        build_cudnn_state_summaries,
    )

    ops = op_inputs(case.work, seed=130)
    torch.manual_seed(130)
    do = torch.randn_like(ops.v)
    cu = ops.cu_seqlens
    bounds = torch.stack((cu[:-1], cu[1:]), dim=1).contiguous()

    def forward():
        return build_cudnn_state_summaries(ops.k, ops.v, ops.g, ops.beta, cu, bounds=bounds)

    def reverse():
        return build_cudnn_state_grad_summaries(
            *ops.leaves, do, cu, D**-0.5, transpose_forward_transition=True, bounds=bounds
        )

    dump = {"forward": forward(), "reverse": reverse()}
    return {"forward": forward, "reverse": reverse}, dump


def paged_phases(case: Case):
    import torch

    ops = op_inputs(case.work, seed=130)
    n, h = case.work.sequences, case.work.value_heads
    op = paged_chunk_op(case.work.kind)
    torch.manual_seed(130)
    pool = torch.randn(n + 2, h, D, D, device="cuda") * 0.1
    indices = torch.arange(1, n + 1, device="cuda", dtype=torch.int32)
    seeds = torch.ones(n, device="cuda", dtype=torch.bool)
    snapshot = pool.clone()

    def forward():
        with torch.no_grad():
            return op(
                *ops.leaves,
                pool,
                indices,
                cu_seqlens=ops.cu_seqlens,
                has_initial_state=seeds,
                kernel_options={"backend": "cudnn"},
            )

    out = forward()
    torch.cuda.synchronize()
    dump = {"output": out, "pool": pool.clone()}
    # Replay advances the pool in place; gates are negative and beta in (0, 1), so it stays
    # bounded. Start timing from the same state on both trees.
    pool.copy_(snapshot)
    return {"forward": forward}, dump


PHASES = {"gdn": op_phases, "kda": op_phases, "summary": summary_phases, "paged": paged_phases}

TELEMETRY_FIELDS = ("uuid", "clocks.sm", "clocks.max.sm", "power.draw", "temperature.gpu")


def parse_telemetry(text: str, uuid: str) -> dict[str, str]:
    """The ``nvidia-smi --query-gpu`` CSV row of the device with ``uuid`` (case-insensitive, with
    or without the ``GPU-`` prefix) as a field dict; empty when absent."""
    wanted = uuid.lower().removeprefix("gpu-")
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if (
            len(fields) == len(TELEMETRY_FIELDS)
            and fields[0].lower().removeprefix("gpu-") == wanted
        ):
            return dict(zip(TELEMETRY_FIELDS, fields))
    return {}


def telemetry(uuid: str) -> dict[str, str]:
    """Clock, power and temperature of the GPU torch runs on (matched by UUID, not index:
    ``CUDA_VISIBLE_DEVICES`` renumbers devices, ``nvidia-smi`` does not)."""
    text = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(TELEMETRY_FIELDS)}", "--format=csv,noheader"],
        check=False,
        text=True,
        capture_output=True,
    ).stdout
    return parse_telemetry(text, uuid)


def device_uuid() -> str:
    import torch

    return str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)


def child(args) -> None:
    activate_tree(args.tree)
    import torch

    source = check_tree(args.tree)
    torch.autograd.graph.set_override_stale_capture_stream(True)
    uuid = device_uuid()
    meta = {
        "label": args.label,
        "round": args.round,
        "source": str(source),
        "gpu": torch.cuda.get_device_name(),
        "uuid": uuid,
        "versions": {
            n: importlib.metadata.version(n) for n in ("torch", "triton", "nvidia-cutlass-dsl")
        },
        "contract": CONTRACT,
    }
    print(json.dumps(meta), flush=True)
    rows, dumps = [], {}
    for name in args.cases:
        case = next(c for c in CASES if c.suite == args.suite and c.name == name)
        phases, dump = PHASES[case.suite](case)
        torch.cuda.synchronize()
        for tensor in dump.values():
            assert torch.isfinite(tensor).all(), f"{case.name}: non-finite output"
        if args.dump:
            dumps[case.name] = {k: v.detach().cpu() for k, v in dump.items()}
        del dump
        row = {"case": case.name, "label": args.label, "round": args.round}
        if args.round == 0:
            last = list(phases)[-1]
            row["kernels"] = kernel_set(phases[last])
        for phase, fn in phases.items():
            row[f"{phase}_us"] = time_graph(fn)
        row["telemetry"] = telemetry(uuid)
        print(json.dumps(row), flush=True)
        rows.append(row)
        Path(args.output).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
        del phases
        torch.cuda.empty_cache()
    if args.dump:
        torch.save(dumps, args.dump)


# ---------------------------------------------------------------------------- parent (driver)


def run_child(args, trees, suite, label, turn, cases, dump: Path | None) -> Path:
    out = Path(args.out)
    log, output = out / f"{suite}_{label}_{turn}.log", out / f"{suite}_{label}_{turn}.json"
    tree = trees[label]
    cmd = [
        sys.executable,
        "-m",
        "tools.cudnn_fe.bench",
        "child",
        "--tree",
        str(tree),
        "--suite",
        suite,
        "--label",
        label,
        "--round",
        str(turn),
        "--output",
        str(output),
        "--cases",
        *cases,
    ]
    if dump is not None:
        cmd += ["--dump", str(dump)]
    print(f"[{timestamp()}] {suite} round {turn} {label} ({tree})", flush=True)
    rc, text = run_logged(
        cmd,
        log,
        timeout=args.child_timeout,
        env=dict(os.environ, PYTHONUNBUFFERED="1"),
        cwd=REPO_ROOT,
    )
    if rc != 0 or not output.exists():
        print(text[-3000:])
        raise SystemExit(f"{suite} child for {label} round {turn} failed (rc={rc}); see {log}")
    return output


def compare_dumps(new: dict, base: dict, cases: list[Case], rel_tol: float) -> list[dict]:
    import torch

    rows = []
    for case in cases:
        for name, x in base[case.name].items():
            y = new[case.name][name]
            x32, y32 = x.float(), y.float()
            rel = ((y32 - x32).norm() / x32.norm().clamp_min(1e-30)).item()
            bitwise = bool(torch.equal(x, y))
            ok = bitwise if case.check == "bitwise" else rel <= rel_tol
            rows.append(
                {
                    "case": case.name,
                    "tensor": name,
                    "check": case.check,
                    "bitwise": bitwise,
                    "rel_l2": rel,
                    "max_abs": (y32 - x32).abs().max().item(),
                    "ok": ok,
                }
            )
    return rows


def run(args) -> None:
    import torch

    new_label, new_tree = parse_tree(args.new, "new")
    base_label, base_tree = parse_tree(args.base, "base")
    if new_label == base_label:
        base_label = f"{base_label}_base"
    trees = {new_label: new_tree, base_label: base_tree}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "new": {"label": new_label, "tree": str(new_tree), "commit": git_head(new_tree)},
        "base": {"label": base_label, "tree": str(base_tree), "commit": git_head(base_tree)},
        "suites": {},
        "rounds": args.rounds,
        "threshold": args.threshold,
        "rel_tol": args.rel_tol,
        "contract": CONTRACT,
        "started": timestamp(),
    }
    for suite in args.suite:
        cases = select(suite, args.cases, args.small)
        names = [c.name for c in cases]
        config["suites"][suite] = names
        (out / "config.json").write_text(json.dumps(config, indent=1))
        for turn in range(args.rounds):
            order = list(trees) if turn % 2 == 0 else list(reversed(trees))
            for label in order:
                dump = out / f"{suite}_dump_{label}.pt" if turn == 0 else None
                run_child(args, trees, suite, label, turn, names, dump)
            if turn == 0:
                dumps = {label: torch.load(out / f"{suite}_dump_{label}.pt") for label in trees}
                rows = compare_dumps(dumps[new_label], dumps[base_label], cases, args.rel_tol)
                (out / f"{suite}_compare.json").write_text(json.dumps(rows, indent=1))
                bad = [r for r in rows if not r["ok"]]
                print(f"{suite} correctness: {len(rows) - len(bad)}/{len(rows)} ok", flush=True)
                if not args.keep_dumps:
                    for label in trees:
                        (out / f"{suite}_dump_{label}.pt").unlink()
    config["finished"] = timestamp()
    (out / "config.json").write_text(json.dumps(config, indent=1))
    text, failures = report(out, args.threshold)
    print(text)
    if failures:
        raise SystemExit(1)


# ---------------------------------------------------------------------------------- report


def load_rows(out: Path, suite: str, label: str) -> tuple[dict, dict, list[str]]:
    """case -> phase -> [round medians]; case -> round-0 kernel set; SM clocks per row."""
    cells: dict[str, dict[str, list[float]]] = {}
    kernels: dict[str, dict] = {}
    clocks: list[str] = []
    for path in sorted(out.glob(f"{suite}_{label}_[0-9]*.json")):
        for row in json.loads(path.read_text())["rows"]:
            for key, value in row.items():
                if key.endswith("_us"):
                    cells.setdefault(row["case"], {}).setdefault(key[:-3], []).append(
                        statistics.median(value)
                    )
            if "kernels" in row:
                kernels[row["case"]] = row["kernels"]
            fields = row.get("telemetry") or {}
            if fields:
                clocks.append(f"{fields['clocks.sm']}/{fields['clocks.max.sm']}")
    return cells, kernels, clocks


def launch_set_changes(new_kernels: dict, base_kernels: dict) -> dict[str, list[str]]:
    """case -> ``+name``/``-name`` lines for kernels launched on only one tree."""
    changes = {}
    for case in sorted(set(new_kernels) & set(base_kernels)):
        added = sorted(set(new_kernels[case]) - set(base_kernels[case]))
        removed = sorted(set(base_kernels[case]) - set(new_kernels[case]))
        if added or removed:
            changes[case] = [f"+{n}" for n in added] + [f"-{n}" for n in removed]
    return changes


def report(out: Path, threshold: float | None = None) -> tuple[str, list[str]]:
    """Render ``report.md`` from an ``--out`` directory -> ``(markdown, failures)``."""
    config = json.loads((out / "config.json").read_text())
    new, base = config["new"], config["base"]
    threshold = threshold if threshold is not None else config["threshold"]
    failures: list[str] = []
    lines = [
        "# cuDNN kernel benchmark gate",
        "",
        f"- new: `{new['label']}` = `{new['tree']}` @ {new['commit']}",
        f"- base: `{base['label']}` = `{base['tree']}` @ {base['commit']}",
        (
            f"- {config['rounds']} interleaved process rounds; cell = median of process-round "
            f"medians (µs per replay, lower is better); {config.get('contract', CONTRACT)}; "
            f"regression threshold {threshold:.1%}; relL2 tolerance {config['rel_tol']:g}"
        ),
        f"- started {config['started']}, finished {config.get('finished', 'n/a')}",
        "",
    ]
    for suite, names in config["suites"].items():
        new_cells, new_kernels, new_clocks = load_rows(out, suite, new["label"])
        base_cells, base_kernels, base_clocks = load_rows(out, suite, base["label"])
        lines += [
            f"## {suite}",
            "",
            (
                f"| case | phase | {base['label']} µs (rounds) | {new['label']} µs (rounds) | "
                f"{new['label']}/{base['label']} |"
            ),
            "|---|---|---|---|---|",
        ]
        for name in names:
            for phase, new_vals in new_cells.get(name, {}).items():
                base_vals = base_cells[name][phase]
                ratio = statistics.median(new_vals) / statistics.median(base_vals)
                flag = ""
                if ratio > 1 + threshold:
                    flag = " ⚠"
                    failures.append(f"{suite}/{name}/{phase} ratio {ratio:.3f}")

                def fmt(xs):
                    return f"{statistics.median(xs):.1f} ({', '.join(f'{x:.1f}' for x in xs)})"

                lines.append(
                    f"| {name} | {phase} | {fmt(base_vals)} | {fmt(new_vals)} | "
                    f"{ratio:.3f}{flag} |"
                )
        clocks = sorted(set(new_clocks + base_clocks))
        lines += ["", f"SM clock / max of the timed GPU per row: {', '.join(clocks) or 'n/a'}"]
        changes = launch_set_changes(new_kernels, base_kernels)
        if changes:
            lines.append("Launch set (round 0 kernel names) differs:")
            for case, entries in changes.items():
                lines.append(f"- {case}: " + ", ".join(f"`{e}`" for e in entries))
                failures.append(f"{suite}/{case} launch set changed")
        else:
            lines.append("Launch set (round 0 kernel names): identical on both trees")
        compare = out / f"{suite}_compare.json"
        if compare.exists():
            rows = json.loads(compare.read_text())
            lines += [
                "",
                "| case | tensor | check | relL2 | max abs | bitwise | ok |",
                "|---|---|---|---|---|---|---|",
            ]
            for r in rows:
                lines.append(
                    f"| {r['case']} | {r['tensor']} | {r['check']} | {r['rel_l2']:.2e} | "
                    f"{r['max_abs']:.2e} | {'yes' if r['bitwise'] else 'no'} | "
                    f"{'yes' if r['ok'] else 'NO'} |"
                )
                if not r["ok"]:
                    failures.append(f"{suite}/{r['case']}/{r['tensor']} {r['check']} check")
        lines.append("")
    verdict = "PASS" if not failures else "FAIL: " + "; ".join(failures)
    lines += ["## Verdict", "", verdict, ""]
    text = "\n".join(lines)
    (out / "report.md").write_text(text)
    return text, failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m tools.cudnn_fe.bench",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="run suites on both trees, then report")
    p.add_argument("--suite", nargs="+", choices=SUITES, required=True)
    p.add_argument("--new", required=True, help="[LABEL=]PATH of the candidate tree")
    p.add_argument("--base", required=True, help="[LABEL=]PATH of the baseline tree")
    p.add_argument("--cases", nargs="*", default=[], help="case names (default: full grid)")
    p.add_argument("--small", action="store_true", help="one small case per suite")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", required=True, help="results directory")
    p.add_argument("--child-timeout", type=int, default=1500)
    p.add_argument("--threshold", type=float, default=0.02, help="new/base regression threshold")
    p.add_argument("--rel-tol", type=float, default=1e-2, help="relL2 tolerance for rel checks")
    p.add_argument("--keep-dumps", action="store_true", help="keep the round-0 .pt dumps")
    r = sub.add_parser("report", help="re-render report.md from an existing --out directory")
    r.add_argument("--out", required=True)
    r.add_argument("--threshold", type=float)
    c = sub.add_parser("child")
    c.add_argument("--tree", type=Path, required=True)
    c.add_argument("--suite", choices=SUITES, required=True)
    c.add_argument("--label", required=True)
    c.add_argument("--round", type=int, required=True)
    c.add_argument("--output", required=True)
    c.add_argument("--cases", nargs="+", required=True)
    c.add_argument("--dump")
    args = ap.parse_args(argv)
    if args.command == "run":
        run(args)
    elif args.command == "child":
        child(args)
    else:
        text, failures = report(Path(args.out), args.threshold)
        print(text)
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
