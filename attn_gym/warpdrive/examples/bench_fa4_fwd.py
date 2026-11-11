"""Compare warpdrive and FA4 forward with interleaved, fixed-buffer CUDA graphs.

Run under gpu-run. Compilation, allocation, reference checks and capture are untimed.
Each graph contains 100 forwards; times are microseconds per forward. Keep per-round
samples: shared-GPU interference can affect even the reported minimum round medians.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from flash_attn.cute import flash_attn_func, flash_fwd_sm100

from attn_gym.warpdrive import ClusterLaunchControl, CtaPair, StaticPersistent
from attn_gym.warpdrive.examples.bench_support import (
    benchmark_cuda_graph_stats,
    provenance,
    settle,
)
from attn_gym.warpdrive.examples.fa4_fwd import (
    LOG2_E,
    FlashAttentionForward,
    compile_fa4_fwd,
)

SHAPES = [(4, 16, 4096), (2, 16, 8192), (8, 16, 1024)]  # batch, heads, query/key length
FORWARDS_PER_GRAPH = 100


def run_shape(b: int, h: int, s: int, variants: list[str], samples: int, profile: bool) -> dict:
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, s, h, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    ref, ref_lse = flash_attn_func(q, k, v, causal=False, return_lse=True)
    calls = {}
    for label in variants:
        if label == "fa4":
            calls[label] = lambda: flash_attn_func(q, k, v, causal=False)
            continue
        op = FlashAttentionForward(
            cluster=CtaPair(),
            schedule=StaticPersistent() if label == "static" else ClusterLaunchControl(),
            split_p=True,
            use_ldred_rowmax=label != "baseline",
        )
        fn = compile_fa4_fwd(op)
        out = torch.empty_like(q)
        lse = torch.empty(b, h, s, device=q.device)
        workspace, sms = op.work.launch_args(q.device)

        def call(fn=fn, out=out, lse=lse, workspace=workspace, sms=sms):
            fn(q, k, v, out, lse, 128**-0.5 * LOG2_E, workspace, sms)

        call()
        torch.cuda.synchronize()
        error_o = (out.float() - ref.float()).abs().max().item()
        error_lse = (lse - ref_lse).abs().max().item()
        # Same BF16 P/output-rounding allowance used by the example's tests.
        tolerance = 2**-7 * ref.float().abs().max().item() + 4e-3
        if (
            not torch.isfinite(out).all()
            or not torch.isfinite(lse).all()
            or error_o > tolerance
            or error_lse > 1e-3
        ):
            raise AssertionError((label, error_o, error_lse, tolerance))
        print(label, op.get_name(), "max errors O/LSE", error_o, error_lse, flush=True)
        calls[label] = call

    for call in calls.values():
        for _ in range(5):
            call()
    torch.cuda.synchronize()
    if profile:
        from cuda.bindings import runtime

        runtime.cudaProfilerStart()
        for call in calls.values():
            call()
            torch.cuda.synchronize()
        runtime.cudaProfilerStop()
        return {"shape": [b, h, s], "profiled_variants": variants}

    graphs = {}
    for label, call in calls.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(FORWARDS_PER_GRAPH):
                call()
        graphs[label] = graph
        graph.replay()
    torch.cuda.synchronize()
    settle(graphs.values(), seconds=3.0)
    medians = {label: [] for label in variants}
    raw_samples = {label: [] for label in variants}
    # Alternate order to reduce timing bias; all variants share inputs and device.
    for round_idx in range(7):
        order = variants if round_idx % 2 == 0 else list(reversed(variants))
        for label in order:
            stats = benchmark_cuda_graph_stats(graphs[label], num_iters=samples, warmup_iters=3)
            median = stats.median_us / FORWARDS_PER_GRAPH
            medians[label].append(median)
            raw_samples[label].append([x / FORWARDS_PER_GRAPH for x in stats.samples_us])
            print((b, h, s), round_idx, label, median, "us", flush=True)
    minima = {label: min(values) for label, values in medians.items()}
    print("minimum round medians (us)", minima, flush=True)
    return {
        "shape": [b, h, s],
        "medians_us": medians,
        "samples_us": raw_samples,
        "min_median_us": minima,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, choices=range(len(SHAPES)))
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["baseline", "rowmax", "static", "fa4"],
        default=["baseline", "rowmax", "static", "fa4"],
    )
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument(
        "--profile", action="store_true", help="one warm launch per variant for NCU"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or len(args.variants) != len(set(args.variants)):
        parser.error("samples must be positive and variants must be unique")
    # Keep application flags out of CuTeDSL's diagnostic parser.
    sys.argv = sys.argv[:1]
    metadata = provenance(
        {
            "warpdrive": Path(__file__).with_name("fa4_fwd.py"),
            "benchmark": Path(__file__),
            "flash_attn": Path(flash_fwd_sm100.__file__),
        },
        {
            "shape": args.shape,
            "variants": args.variants,
            "samples": args.samples,
            "profile": args.profile,
            "rounds": 7,
            "forwards_per_graph": FORWARDS_PER_GRAPH,
            "warmup_graphs_per_round": 3,
            "settle_seconds": 3,
        },
    )
    print("metadata", metadata, flush=True)
    shapes = SHAPES if args.shape is None else [SHAPES[args.shape]]
    results = [run_shape(*shape, args.variants, args.samples, args.profile) for shape in shapes]
    if args.output is not None:
        args.output.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2))


if __name__ == "__main__":
    main()
