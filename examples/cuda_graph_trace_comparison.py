"""Generate the tutorial's aligned raw-versus-postprocessed CUDA Graph trace."""

import multiprocessing
from pathlib import Path

import torch
from transformer_nuggets.utils.merge_traces import merge_traces

from examples.cuda_graphs import hello_world_training_loop

ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / "agent_space/cuda_graph_training_loop_comparison"
OUTPUT = ROOT / "docs/assets/traces/hello_world_training_loop_comparison.pftrace"


def _capture_trace(
    annotated: bool,
    fix_overlapping_events: bool,
    trace_path: Path,
) -> None:
    """Capture one regular-profiler comparison arm in an isolated process."""
    torch.manual_seed(0)
    hello_world_training_loop(
        enable_graph_annotations=annotated,
        trace_path=trace_path,
        trace_format="chrome_json",
        fix_overlapping_events=fix_overlapping_events,
    )


def main() -> None:
    """Capture stock and annotated arms, then write one native Perfetto trace."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    before_trace = WORK_DIR / "before-raw.json"
    after_trace = WORK_DIR / "after-postprocessed.json.gz"

    context = multiprocessing.get_context("spawn")
    captures = (
        (False, False, before_trace),
        (True, True, after_trace),
    )
    for annotated, fix_overlapping_events, trace_path in captures:
        process = context.Process(
            target=_capture_trace,
            args=(annotated, fix_overlapping_events, trace_path),
        )
        process.start()
        process.join()
        if process.exitcode:
            raise RuntimeError(f"trace capture failed with exit code {process.exitcode}")

    merge_traces(
        [str(before_trace), str(after_trace)],
        str(OUTPUT),
        labels=[
            "1 · BEFORE · stock CUDA Graph profiler",
            "2 · AFTER · graph annotations",
        ],
        align_timestamps=True,
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
