"""Run host-only declaration or compile-only mutation campaigns; never launch kernels.

    python tools/warpdrive/fuzz_warpdrive.py static --seed 0 --budget 10000
    python tools/warpdrive/fuzz_warpdrive.py mutate --seed 0 --budget 1000

CUTE_DSL_ARCH defaults to sm_100a. Print a JSON report and exit nonzero on any finding.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "test"))  # the warpdrive_fuzz package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("static", "mutate"):
        p = sub.add_parser(name, help=f"{name} corpus (compile-only)")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--budget", type=int, default=1000)
    args = parser.parse_args()
    sys.argv = sys.argv[:1]  # CuTeDSL re-parses sys.argv at import
    os.environ.setdefault("CUTE_DSL_ARCH", "sm_100a")
    if args.command == "static":
        from warpdrive_fuzz.static import run_static_fuzz as run
    else:
        from warpdrive_fuzz.mutate import run_mutation_fuzz as run
    report = run(args.seed, args.budget)
    print(json.dumps(asdict(report), indent=2, default=str))
    return 1 if report.findings else 0


if __name__ == "__main__":
    sys.exit(main())
