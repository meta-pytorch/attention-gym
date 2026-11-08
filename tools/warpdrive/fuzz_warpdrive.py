"""Long-running warpdrive fuzz campaigns.

Run under a GPU reservation for the runtime fuzzer::

    gpu-run --timeout 60 auto -- python tools/warpdrive/fuzz_warpdrive.py runtime --seeds 0:500 \
        --jobs 6 --out agent_space/warpdrive/fuzz/campaign

Each runtime case runs in its own interpreter with ``--timeout`` seconds before it counts as a
hang. Results stream to ``<out>/results.jsonl``; failing cases keep their spec JSON and source in
``<out>/cases``. ``static`` and ``mutate`` run the host/compile-only corpora (no GPU needed
with ``CUTE_DSL_ARCH=sm_100a``)::

    python tools/warpdrive/fuzz_warpdrive.py static --seed 0 --budget 1000

``--cluster 2x2p`` (or ``2x2``, ``4x1p``, ...) generates kernels on a larger ``Cluster`` with
``all``/``leader``/``rank0`` placements instead of single CTAs and ``CtaPair``s.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "test"))  # the warpdrive_fuzz package


def _seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if ":" in part:
            a, b = part.split(":")
            out += range(int(a), int(b))
        else:
            out.append(int(part))
    return out


def runtime_campaign(args) -> int:
    from warpdrive_fuzz import gen, runtime

    out = Path(args.out)
    cases = out / "cases"
    out.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    failed = False
    deadline_reached = False
    start = time.time()

    def one(seed: int):
        force = {}
        if args.pair is not None:
            force["pair"] = args.pair
        if args.stream is not None:
            force["stream"] = args.stream
        if args.empty:
            force["empty"] = True
        if args.cluster:
            force["cluster"] = gen.parse_cluster(args.cluster)
        spec = gen.generate(seed, **force)
        # "{seed}" in the wrapper names per-case tool outputs (CUTracer delay dumps).
        wrapper = args.wrapper.replace("{seed}", str(seed)).split() if args.wrapper else None
        return spec, runtime.run_isolated(spec, cases, timeout=args.timeout, wrapper=wrapper)

    seeds = _seeds(args.seeds)
    with (out / "results.jsonl").open("a") as log, ThreadPoolExecutor(args.jobs) as pool:
        futures = {pool.submit(one, seed): seed for seed in seeds}
        for fut in as_completed(futures):
            if fut.cancelled():
                continue  # recorded when cancelled at the campaign deadline
            spec, res = fut.result()
            failed |= res.status != runtime.PASS
            known = runtime.known_finding(res)
            label = f"known:{known}" if known else res.status
            counts[label] = counts.get(label, 0) + 1
            row = asdict(res)
            row["pair"], row["schedule"] = spec.pair, spec.stream.schedule if spec.stream else None
            row["cluster"] = spec.cluster
            row["known"], row["features"] = known, sorted(gen.features(spec))
            log.write(json.dumps(row) + "\n")
            log.flush()
            if res.status != runtime.PASS:
                print(
                    f"seed {res.seed} {res.key} {res.status}: "
                    f"{(res.failures or [res.detail])[0][:300]}",
                    flush=True,
                )
            elif not args.quiet:
                print(
                    f"seed {res.seed} {res.key} pass ({res.compile_s:.1f}s compile, "
                    f"{res.run_s:.2f}s run)",
                    flush=True,
                )
            if not deadline_reached and args.hours and time.time() - start > args.hours * 3600:
                deadline_reached = True
                for pending, seed in futures.items():
                    if pending.cancel():
                        counts["cancelled"] = counts.get("cancelled", 0) + 1
                        log.write(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "status": "cancelled",
                                    "reason": "campaign deadline",
                                }
                            )
                            + "\n"
                        )
                log.flush()
                # Running jobs cannot be cancelled. Keep draining their verdicts: a late
                # failure must still fail the campaign, even if it matches a known finding.
    print(json.dumps({"elapsed_s": round(time.time() - start), **counts}))
    return int(failed)


def peer_campaign(args) -> int:
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")
    if args.command == "static":
        from warpdrive_fuzz.static import run_static_fuzz as fn
    else:
        from warpdrive_fuzz.mutate import run_mutation_fuzz as fn
    report = fn(args.seed, args.budget)
    print(json.dumps(asdict(report), indent=2, default=str))
    return 1 if report.findings else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    rt = sub.add_parser("runtime", help="random valid kernels on the GPU")
    rt.add_argument("--seeds", default="0:100", help="'a:b' ranges and/or comma-separated seeds")
    rt.add_argument("--jobs", type=int, default=6, help="cases compiled/run concurrently")
    rt.add_argument("--timeout", type=float, default=120.0, help="seconds before a case hangs")
    rt.add_argument("--out", default="agent_space/warpdrive/fuzz/campaign")
    rt.add_argument(
        "--hours",
        type=float,
        default=0.0,
        help="cancel queued jobs after this long; collect running jobs (0: all)",
    )
    rt.add_argument("--pair", type=lambda s: s == "1", default=None, help="force CtaPair (1/0)")
    rt.add_argument(
        "--cluster",
        default="",
        help="declare Cluster(m, n): 'MxN', with a trailing 'p' for MMA pairs (e.g. 2x2p)",
    )
    rt.add_argument("--stream", type=lambda s: s == "1", default=None, help="force a WorkStream")
    rt.add_argument("--empty", action="store_true", help="zero-work boundary corpus (G6)")
    rt.add_argument(
        "--wrapper",
        default="",
        help="command prefix, e.g. 'compute-sanitizer --tool racecheck'; {seed} expands",
    )
    rt.add_argument("--quiet", action="store_true", help="print only non-passing cases")
    for name in ("static", "mutate"):
        p = sub.add_parser(name, help=f"{name} corpus (compile-only)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--budget", type=int, default=1000)
    args = parser.parse_args()
    sys.argv = sys.argv[:1]  # CuTeDSL re-parses sys.argv at import
    if args.command == "runtime":
        return runtime_campaign(args)
    return peer_campaign(args)


if __name__ == "__main__":
    sys.exit(main())
