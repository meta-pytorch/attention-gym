"""Tabulate KDA context-parallel scaling runs.

Reads the ``kda_cp_scaling_*.json`` files written by
``examples/kda_context_parallel.py --benchmark-steps N`` (one per world size and
mode) and prints a table of step time, throughput, and per-rank memory against
the CP world size, plus ideal-scaling references.

    python benchmarks/kda_cp_scaling_report.py ~/.mast_play/results/*/data/kda_cp_scaling_*.json
    python benchmarks/kda_cp_scaling_report.py results/*.json --csv scaling.csv
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Annotated

import typer


def main(
    files: Annotated[list[Path], typer.Argument(help="kda_cp_scaling_*.json files.")],
    csv_path: Annotated[Path | None, typer.Option("--csv", help="Also write rows as CSV.")] = None,
) -> None:
    rows = [json.loads(path.read_text()) for path in files]
    rows.sort(key=lambda row: (row["tokens"], row["mode"], row["world_size"]))
    header = (
        f"{'tokens':>9} {'mode':<10} {'W':>3} {'tok/rank':>9} {'step ms':>15} {'Mtok/s':>8} "
        f"{'speedup':>8} {'ideal':>6} {'alloc GiB':>10} {'step GiB':>9} {'reserved GiB':>13}"
    )
    print(header)
    print("-" * len(header))
    for (tokens, mode), group in _grouped(rows):
        base = group[0]
        for row in group:
            speedup = base["step_ms_mean"] / row["step_ms_mean"]
            ideal = row["world_size"] / base["world_size"]
            print(
                f"{tokens:>9} {mode:<10} {row['world_size']:>3} {tokens // row['world_size']:>9} "
                f"{row['step_ms_mean']:>9.2f}±{row['step_ms_std']:<5.2f} {row['tokens_per_s'] / 1e6:>8.2f} "
                f"{speedup:>7.2f}x {ideal:>5.0f}x {row['peak_gib_max']:>10.2f} "
                f"{row['step_peak_gib_max']:>9.2f} {row['reserved_gib_max']:>13.2f}"
            )
    if csv_path:
        fields = [
            "tokens",
            "mode",
            "world_size",
            "hidden_size",
            "heads",
            "kda_backend",
            "device_name",
            "step_ms_mean",
            "step_ms_std",
            "step_ms_min",
            "tokens_per_s",
            "peak_gib_max",
            "step_peak_gib_max",
            "resident_gib_max",
            "reserved_gib_max",
        ]
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {csv_path}")


def _grouped(rows: list[dict]) -> list[tuple[tuple[int, str], list[dict]]]:
    groups: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["tokens"], row["mode"]), []).append(row)
    return sorted(groups.items())


if __name__ == "__main__":
    typer.run(main)
