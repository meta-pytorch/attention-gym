"""Tabulate KDA context-parallel scaling runs.

Reads the ``kda_cp_scaling_*.json`` files written by
``examples/delta_rule_context_parallel.py --benchmark-steps N`` (one per world size and
mode) and prints a table of step time, throughput, and per-rank memory against
the CP world size, plus ideal-scaling references.

    python benchmarks/kda_cp_scaling_report.py ~/.mast_play/results/*/data/kda_cp_scaling_*.json
    python benchmarks/kda_cp_scaling_report.py results/*.json --csv scaling.csv
    python benchmarks/kda_cp_scaling_report.py results/*.json --plot scaling.png
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
    plot_path: Annotated[
        Path | None,
        typer.Option(
            "--plot", help="Write step time, throughput, and memory vs world size (matplotlib)."
        ),
    ] = None,
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
    if plot_path:
        _plot(rows, plot_path)
        print(f"wrote {plot_path}")


def _plot(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    styles = {"eager": "o-", "cuda_graph": "s--"}
    for (tokens, mode), group in _grouped(rows):
        worlds = [row["world_size"] for row in group]
        label = f"{mode} · {tokens / 2**20:g}M tokens"
        axes[0].plot(
            worlds, [row["step_ms_mean"] for row in group], styles.get(mode, "o-"), label=label
        )
        axes[1].plot(
            worlds,
            [row["tokens_per_s"] / 1e6 for row in group],
            styles.get(mode, "o-"),
            label=label,
        )
        memory_key = "reserved_gib_max" if mode == "cuda_graph" else "peak_gib_max"
        axes[2].plot(
            worlds,
            [row[memory_key] for row in group],
            styles.get(mode, "o-"),
            label=f"{label} ({'reserved' if mode == 'cuda_graph' else 'peak allocated'})",
        )
        if mode == "eager":
            base = group[0]
            axes[0].plot(
                worlds,
                [base["step_ms_mean"] * base["world_size"] / world for world in worlds],
                ":",
                color="gray",
                label="ideal (1/W)" if tokens == rows[0]["tokens"] else None,
            )
            axes[2].plot(
                worlds,
                [base["peak_gib_max"] * base["world_size"] / world for world in worlds],
                ":",
                color="gray",
                label="ideal (1/W)" if tokens == rows[0]["tokens"] else None,
            )
    for axis, title, ylabel in (
        (axes[0], "Step time (fwd+bwd, slowest rank)", "ms"),
        (axes[1], "Throughput", "Mtok/s"),
        (axes[2], "Per-rank GPU memory", "GiB"),
    ):
        axis.set_title(title)
        axis.set_xlabel("CP world size W (GPUs)")
        axis.set_ylabel(ylabel)
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted({row["world_size"] for row in rows}))
        axis.set_xticklabels([str(world) for world in sorted({row["world_size"] for row in rows})])
        axis.grid(True, which="both", alpha=0.3)
    axes[0].set_yscale("log", base=2)
    axes[2].set_yscale("log", base=2)
    axes[0].legend(fontsize=8)
    axes[2].legend(fontsize=7)
    first = rows[0]
    fig.suptitle(
        f"KDA context parallel: hidden {first['hidden_size']}, {first['heads']}×{first['head_dim']} heads, "
        f"{first['device_name']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def _grouped(rows: list[dict]) -> list[tuple[tuple[int, str], list[dict]]]:
    groups: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["tokens"], row["mode"]), []).append(row)
    return sorted(groups.items())


if __name__ == "__main__":
    typer.run(main)
