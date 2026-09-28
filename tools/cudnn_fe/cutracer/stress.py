"""CUTracer random-delay race stress (and bounded deadlock detection) for the cuDNN kernels.

Records an unperturbed oracle reference under the CUTracer launch logger, derives one kernel
filter per vendored kernel family from the launch log (or takes ``--filter``), then runs
``--patterns`` fresh random-delay patterns per ``--delays`` entry for every filter. Each attempt
re-runs the oracle ``compare`` under ``cutracer trace -a random_delay`` and must stay bitwise
identical *and* must have perturbed something: an attempt whose filter matched no launched
kernel, whose delay dump enabled no site, or whose replay CUTracer could not apply fails. The
delay pattern is dumped for deterministic ``--replay``. Per-attempt verdicts, the enabled delay
sites by SASS kind and the kernel hashes land in ``<out>/results.{md,json}``.

Reserve the GPU around the whole run (``gpu-run --timeout 900 auto -- ...``) or pass
``--gpu-run`` to claim one per attempt without waiting (a busy GPU fails the attempt).

Examples::

    python -m tools.cudnn_fe.cutracer.stress --family gdn --ref-dir /tmp/refs --out /tmp/stress
    python -m tools.cudnn_fe.cutracer.stress --family kda --filter frost_kda_prep_KdaPrepCfg \\
        --delays 5000 --patterns 1 ...
    python -m tools.cudnn_fe.cutracer.stress --family gdn --filter frost_gdn_prefill_GdnPrefillCfg \\
        --mode deadlock --case uncut_packed ...
    python -m tools.cudnn_fe.cutracer.stress --family gdn --filter F \\
        --replay /tmp/stress/gdn/F/d5000_a1.delay.json ...
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .._common import GPU_BUSY, REPO_ROOT, describe_exit, run_logged

DEFAULT_CUTRACER = Path("~/.venvs/cutracer/bin/cutracer").expanduser()
LAUNCH = re.compile(r"LAUNCH - Kernel pc \S+ - Kernel name (\S+) - kernel hash (0x[0-9a-f]+)")
VERDICT = re.compile(
    r"all \d+ tensors bitwise identical to reference|MISMATCH.*|missing reference.*"
)
# CUTracer's deadlock detector: a transient hang report and the sustained-hang SIGTERM.
DEADLOCK = re.compile(r"Possible kernel hang|Deadlock sustained")
# CUTracer replay could not map the dumped pattern onto the traced kernel and ran unperturbed.
REPLAY_MISS = re.compile(r"No config found for kernel|NOT FOUND in config")
# Kernel-name fragments that identify a vendored cuDNN kernel and, in order, the suffix rules
# that turn a launch name into a stable substring filter.
FILTER_RULES = (
    re.compile(r"(frost_\w+?_prologue)"),
    re.compile(r"(frost_\w+?_[A-Z]\w*?Cfg)"),
    re.compile(r"(frost_split_k)"),
    re.compile(r"(frost_state_chain)"),
)


def derive_filter(kernel_name: str) -> str | None:
    for rule in FILTER_RULES:
        m = rule.search(kernel_name)
        if m:
            return m[1]
    return None


def parse_launches(text: str) -> dict[str, set[str]]:
    """kernel name -> hashes, restricted to the vendored cuDNN kernels."""
    kernels: dict[str, set[str]] = collections.defaultdict(set)
    for name, digest in LAUNCH.findall(text):
        if derive_filter(name):
            kernels[name].add(digest)
    return dict(kernels)


def opcode_kind(sass: str) -> str:
    """Base mnemonic of one SASS line, skipping a ``@P0``/``@!UP1`` predicate."""
    tokens = sass.split()
    while tokens and tokens[0].startswith("@"):
        tokens.pop(0)
    return tokens[0].partition(".")[0].rstrip(";") if tokens else "?"


def site_summary(dump_path: Path) -> dict[str, dict]:
    """Per kernel: number of enabled delay sites and their SASS opcode kinds."""
    try:
        dump = json.loads(dump_path.read_text())
    except (OSError, ValueError):
        return {}
    summary = {}
    for kernel in dump.get("kernels", {}).values():
        on = [p for p in kernel["instrumentation_points"].values() if p["on"]]
        kinds = collections.Counter(opcode_kind(p["sass"]) for p in on)
        summary[kernel["kernel_name"]] = {
            "hash": kernel.get("kernel_checksum"),
            "enabled": len(on),
            "kinds": dict(kinds.most_common()),
        }
    return summary


def instrumented(sites: dict[str, dict]) -> bool:
    return any(site["enabled"] > 0 for site in sites.values())


def judge(
    mode: str,
    rc: int,
    text: str,
    *,
    sites: dict[str, dict],
    replay: bool,
    matched: bool | None,
) -> tuple[str, bool]:
    """Verdict line and pass/fail of one attempt.

    ``matched`` is whether the filter matched a launched vendored kernel (``None`` when no
    launch log exists, e.g. ``--skip-record --filter``). Random mode requires enabled delay
    sites; replay additionally fails when CUTracer reported an unmatched pattern; deadlock mode
    fails on any hang report.
    """
    verdicts = VERDICT.findall(text)
    verdict = verdicts[-1] if verdicts else f"no verdict ({describe_exit(rc)})"
    ok = rc == 0 and verdict.startswith("all ")
    if matched is False:
        return f"filter matched no launched kernel :: {verdict}", False
    if mode == "random":
        if replay and REPLAY_MISS.search(text):
            return (
                f"replay pattern not applied (CUTracer: no config for kernel) :: {verdict}",
                False,
            )
        if not instrumented(sites):
            return f"no kernel instrumented (no enabled delay site) :: {verdict}", False
    else:
        reports = [line.strip() for line in text.splitlines() if DEADLOCK.search(line)]
        if reports:
            return f"DEADLOCK REPORT: {reports[0][:120]}", False
    return verdict, ok


@dataclass
class Attempt:
    family: str
    filter: str
    delay_ns: int | None
    attempt: int
    mode: str
    rc: int
    verdict: str
    seconds: float
    log: str
    ok: bool
    dump: str | None = None
    sites: dict = field(default_factory=dict)


class Runner:
    def __init__(self, args):
        self.args = args
        self.env = {k: v for k, v in os.environ.items() if k != "CUDA_INJECTION64_PATH"}
        self.env["PYTHONUNBUFFERED"] = "1"

    def oracle(self, mode: str, family: str) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "tools.cudnn_fe.cutracer.oracle",
            mode,
            "--ref-dir",
            str(self.args.ref_dir),
            "--family",
            family,
        ]
        for case in self.args.case:
            cmd += ["--case", case]
        return cmd

    def run(self, cmd: list[str], log: Path) -> tuple[int, str, float]:
        start = time.time()
        rc, text = run_logged(
            cmd,
            log,
            timeout=self.args.timeout,
            env=self.env,
            cwd=REPO_ROOT,
            gpu_run=self.args.gpu_run,
        )
        return rc, text, time.time() - start

    def launch_log(self, family: str, out: Path) -> dict[str, set[str]]:
        """Record the reference (or just re-run the oracle) under the bare launch logger."""
        mode = "compare" if self.args.skip_record else "record"
        log = out / f"launch_{family}.log"
        cmd = [
            str(self.args.cutracer),
            "trace",
            "-o",
            str(out / f"launch_{family}"),
            "--",
            *self.oracle(mode, family),
        ]
        print(f"[{family}] {mode} + launch log -> {log}", flush=True)
        rc, text, seconds = self.run(cmd, log)
        if rc != 0 or (mode == "record" and "recorded" not in text):
            print(text[-3000:])
            raise SystemExit(f"{mode} under the launch logger failed ({describe_exit(rc)})")
        print(f"[{family}] {mode} ok in {seconds:.0f}s", flush=True)
        return parse_launches(text)

    def attempt(
        self,
        family: str,
        kfilter: str,
        delay: int | None,
        index: int,
        out: Path,
        matched: bool | None,
        load: Path | None = None,
    ) -> Attempt:
        args = self.args
        tag = f"d{delay}_a{index}" if args.mode == "random" else "deadlock"
        if load is not None:
            tag = f"replay_{load.stem.removesuffix('.delay')}"
        log, trace = out / f"{tag}.log", out / f"trace_{tag}"
        dump = out / f"{tag}.delay.json" if args.mode == "random" and load is None else None
        cmd = [
            str(args.cutracer),
            "trace",
            "--no-data-timeout-s",
            "0",
            "-k",
            kfilter,
            "-o",
            str(trace),
        ]
        if args.mode == "random":
            cmd += ["-a", "random_delay"]
            if load is not None:
                delay = json.loads(load.read_text())["delay_ns"]
                cmd += ["--delay-ns", str(delay), "--delay-load-path", str(load)]
            else:
                cmd += ["--delay-ns", str(delay), "--delay-dump-path", str(dump)]
        else:
            cmd += [
                "-a",
                "deadlock_detection",
                "--trace-format",
                "zstd",
                "--trace-size-limit-mb",
                str(args.trace_size_limit_mb),
            ]
        cmd += ["--", *self.oracle("compare", family)]
        rc, text, seconds = self.run(cmd, log)
        pattern = dump or load
        sites = site_summary(pattern) if pattern else {}
        verdict, ok = judge(
            args.mode, rc, text, sites=sites, replay=load is not None, matched=matched
        )
        result = Attempt(
            family,
            kfilter,
            delay,
            index,
            args.mode,
            rc,
            verdict,
            seconds,
            str(log),
            ok,
            str(pattern) if pattern else None,
            sites,
        )
        if not args.keep_traces:
            shutil.rmtree(trace, ignore_errors=True)
        enabled = " | ".join(f"{k[:40]}:{v['enabled']}" for k, v in sites.items())
        print(
            f"[{family}:{kfilter}] {tag} rc={rc} {seconds:.0f}s :: {verdict} :: {enabled}",
            flush=True,
        )
        return result


def write_results(
    out: Path, args, kernels: dict[str, dict[str, set[str]]], attempts: list[Attempt]
) -> bool:
    ok = bool(attempts) and all(a.ok for a in attempts)
    payload = {
        "argv": sys.argv[1:],
        "mode": args.mode,
        "delays_ns": args.delays,
        "patterns": args.patterns,
        "kernels": {fam: {k: sorted(v) for k, v in ks.items()} for fam, ks in kernels.items()},
        "attempts": [asdict(a) for a in attempts],
        "pass": ok,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    passed = sum(a.ok for a in attempts)
    lines = [
        f"# CUTracer {args.mode} stress ({'PASS' if ok else 'FAIL'})",
        "",
        (
            f"{passed}/{len(attempts)} attempts bitwise identical with delay sites enabled; "
            f"delays {args.delays} ns, {args.patterns} pattern(s) per delay."
        ),
        "",
    ]
    lines += [
        "| family | filter | delay ns | attempt | rc | s | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in attempts:
        lines.append(
            f"| {a.family} | `{a.filter}` | {a.delay_ns or '-'} | {a.attempt} | {a.rc} | "
            f"{a.seconds:.0f} | {a.verdict} |"
        )
    sites: dict[tuple, dict] = {}
    for a in attempts:
        for kernel, s in a.sites.items():
            entry = sites.setdefault(
                (a.family, a.filter, kernel[:48]),
                {"n": 0, "enabled": [], "kinds": collections.Counter()},
            )
            entry["n"] += 1
            entry["enabled"].append(s["enabled"])
            entry["kinds"].update(s["kinds"])
    if sites:
        lines += [
            "",
            "## Enabled delay sites (per dump, SASS kinds summed over dumps)",
            "",
            "| family | filter | kernel | dumps | enabled sites / dump | kinds |",
            "|---|---|---|---|---|---|",
        ]
        for (fam, flt, kernel), e in sites.items():
            kinds = " ".join(f"{k}={v}" for k, v in e["kinds"].most_common())
            lines.append(
                f"| {fam} | `{flt}` | {kernel} | {e['n']} | "
                f"{min(e['enabled'])}–{max(e['enabled'])} | {kinds} |"
            )
    lines += [
        "",
        "## Kernel hashes from the launch log",
        "",
        "| family | filter | kernel | hashes |",
        "|---|---|---|---|",
    ]
    for fam, ks in kernels.items():
        for name, hashes in sorted(ks.items()):
            lines.append(
                f"| {fam} | `{derive_filter(name)}` | {name[:60]} | {', '.join(sorted(hashes))} |"
            )
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print(f"\nresults -> {out / 'results.md'} ({'PASS' if ok else 'FAIL'})")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m tools.cudnn_fe.cutracer.stress",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--family", default="all", choices=("gdn", "kda", "all"))
    ap.add_argument(
        "--filter",
        action="append",
        default=[],
        help="kernel-name substring; default: every filter found in the launch log",
    )
    ap.add_argument("--case", action="append", default=[], help="restrict the oracle cases")
    ap.add_argument(
        "--delays",
        type=int,
        nargs="+",
        default=[5000, 20000, 100000],
        help="max delay per site in ns (random mode)",
    )
    ap.add_argument("--patterns", type=int, default=3, help="fresh random patterns per delay")
    ap.add_argument("--mode", choices=("random", "deadlock"), default="random")
    ap.add_argument(
        "--trace-size-limit-mb",
        type=int,
        default=2048,
        help="deadlock mode: stop tracing past this size, kernels keep running",
    )
    ap.add_argument("--replay", type=Path, help="replay one dumped delay pattern (one --filter)")
    ap.add_argument("--ref-dir", type=Path, required=True, help="oracle reference directory")
    ap.add_argument("--out", type=Path, required=True, help="logs, dumps and results")
    ap.add_argument("--cutracer", type=Path, default=DEFAULT_CUTRACER)
    ap.add_argument("--timeout", type=int, default=1800, help="per-attempt process timeout (s)")
    ap.add_argument(
        "--gpu-run",
        action="store_true",
        help="claim a GPU per attempt with `gpu-run --no-wait auto` (fails fast when busy)",
    )
    ap.add_argument("--skip-record", action="store_true", help="reuse references in --ref-dir")
    ap.add_argument("--keep-traces", action="store_true", help="keep CUTracer trace directories")
    ap.add_argument("--dry-run", action="store_true", help="record + discover filters, then stop")
    args = ap.parse_args(argv)
    if not args.cutracer.exists():
        ap.error(f"cutracer CLI not found at {args.cutracer} (see README: CUTracer setup)")
    if args.replay and len(args.filter) != 1:
        ap.error("--replay needs exactly one --filter")
    args.out, args.ref_dir = args.out.resolve(), args.ref_dir.resolve()
    if args.replay:
        args.replay = args.replay.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    runner = Runner(args)
    families = ["gdn", "kda"] if args.family == "all" else [args.family]
    kernels: dict[str, dict[str, set[str]]] = {}
    attempts: list[Attempt] = []
    for family in families:
        need_launch = not args.filter or not args.skip_record
        kernels[family] = runner.launch_log(family, args.out) if need_launch else {}
        filters = args.filter or sorted({derive_filter(k) for k in kernels[family]})
        if not filters:
            raise SystemExit(f"[{family}] no vendored cuDNN kernel in the launch log")
        print(f"[{family}] filters: {' '.join(filters)}", flush=True)
        if args.dry_run:
            continue
        for kfilter in filters:
            matched = any(kfilter in name for name in kernels[family]) if need_launch else None
            out = args.out / family / re.sub(r"[^A-Za-z0-9]", "_", kfilter)
            out.mkdir(parents=True, exist_ok=True)
            if args.replay:
                plan = [(None, 1, args.replay)]
            elif args.mode == "deadlock":
                plan = [(None, 1, None)]
            else:
                plan = [(d, i, None) for d in args.delays for i in range(1, args.patterns + 1)]
            for delay, index, load in plan:
                attempts.append(runner.attempt(family, kfilter, delay, index, out, matched, load))
                if attempts[-1].rc == GPU_BUSY:
                    print("every GPU is reserved; stopping the ladder", flush=True)
                    write_results(args.out, args, kernels, attempts)
                    return GPU_BUSY
    return 0 if write_results(args.out, args, kernels, attempts) else 1


if __name__ == "__main__":
    sys.exit(main())
