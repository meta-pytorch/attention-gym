"""Capture one IKET run of an annotated kernel and keep every artifact under a tag.

    python -m attn_gym.profiler.pipeline.capture \\
        --tag 02_split_stages --annotated rev_annotated.py --iterations 8 \\
        -- python profile_kernel.py --iket

The command must launch the annotated kernel exactly once with IKET enabled. The
run executes under ``run-iket`` with ``CUTE_DSL_NO_CACHE=1``; the raw IKET JSON
and pftrace, the enriched pipeline pftrace, and the text report are copied to
``<out>/<tag>.*`` (``<tag>`` kept whole, dots included) so successive experiments stay side
by side. Pass ``--from-trace <trace.json>`` to re-analyze an existing capture without a GPU.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from attn_gym.profiler.iket import parse_iket
from attn_gym.profiler.pipeline import (
    analyze_pipeline,
    extract_plan,
    join,
    write_pipeline_perfetto,
)
from attn_gym.profiler.pipeline.summary import IKET_RANGE_OVERHEAD_NS, report


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tag", required=True, help="artifact prefix under --out")
    parser.add_argument("--annotated", required=True, type=Path, help="annotated kernel source")
    parser.add_argument("--iterations", required=True, type=int, help="profiled loop iterations")
    parser.add_argument("--out", type=Path, default=Path("traces"))
    parser.add_argument("--cta", default="0,0,0", help="CTA to join, e.g. 0,0,0")
    parser.add_argument(
        "--unprofiled-iteration-ns", type=float, default=None, help="host-timed ns per iteration"
    )
    parser.add_argument("--keep-run-dir", action="store_true", help="keep the run-iket directory")
    parser.add_argument(
        "--from-trace", type=Path, default=None, help="re-analyze this IKET trace.json instead"
    )
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="-- <command that launches once>"
    )
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command and args.from_trace is None:
        parser.error("missing command after --")
    return args


def run_iket_once(command: list[str], run_dir: Path) -> Path:
    """Run ``command`` under ``run-iket`` with the JIT cache disabled; return its trace."""
    run_iket = shutil.which("run-iket")
    if run_iket is None:
        raise SystemExit("run-iket not found on PATH; install a CuTeDSL release with IKET")
    shutil.rmtree(run_dir, ignore_errors=True)
    completed = subprocess.run(
        [run_iket, "--output-dir", str(run_dir), "--clobber", "profile", "--postprocess", "all"]
        + ["--", *command],
        env=dict(os.environ, CUTE_DSL_NO_CACHE="1"),
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"run-iket exited with {completed.returncode}")
    traces = sorted(run_dir.glob("*.trace.json"))
    if len(traces) != 1:
        raise SystemExit(f"expected one IKET trace in {run_dir}, found {len(traces)}")
    return traces[0]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    plan = extract_plan(args.annotated)
    timeline = plan.schedule(iterations=args.iterations)
    cta = tuple(int(part) for part in args.cta.split(","))

    args.out.mkdir(parents=True, exist_ok=True)

    def artifact(suffix: str) -> Path:
        return args.out / f"{args.tag}{suffix}"

    run_dir = args.out / f"iket_{args.tag}"
    if args.from_trace is not None:
        trace = args.from_trace
    else:
        trace = run_iket_once(args.command, run_dir)
        shutil.copy(trace, artifact(".raw_iket.trace.json"))
        for raw in run_dir.glob("*.pftrace"):
            shutil.copy(raw, artifact(".raw_iket.pftrace"))

    measured = join(parse_iket(trace), timeline, cta=cta)
    analysis = analyze_pipeline(timeline, measured)
    write_pipeline_perfetto(artifact(".enriched.pftrace"), timeline, measured, analysis=analysis)
    text = report(
        timeline,
        measured,
        analysis,
        unprofiled_iteration_ns=args.unprofiled_iteration_ns,
        range_overhead_ns=IKET_RANGE_OVERHEAD_NS,
    )
    artifact(".report.txt").write_text(text + "\n")
    print(text)
    if not args.keep_run_dir and args.from_trace is None:
        shutil.rmtree(run_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
