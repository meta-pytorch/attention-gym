"""Compare ``persistent_gemm`` configurations with ``torch.mm`` using CUDA graphs.

Run under gpu-run. Compilation, allocation, reference checks and capture are untimed.
Each graph holds ``CALLS_PER_GRAPH`` calls with fixed, reused buffers. Times are microseconds
per call. Alternate variant order across rounds to reduce timing bias; report the minimum
round median. Shared-GPU interference can still affect it.

A variant label is ``mm`` (cuBLAS through ``torch.mm``) or ``single``/``pair`` (one CTA or a
CTA pair) followed by options: ``_static``/``_clc``/``_atomic`` (schedule, default static;
``_clc3`` sets the broadcast ring depth), ``_n<tile_n>`` (default ``default_tile_n``),
``_acc<stages>`` (default 2) and ``_late`` (release the accumulator after the last subtile's
TMA store is issued, rather than after its last TMEM load).
"""

import argparse
import json
import sys
from pathlib import Path

import torch

from attn_gym.warpdrive import (
    AtomicPersistent,
    ClusterLaunchControl,
    CtaPair,
    StaticPersistent,
)
from attn_gym.warpdrive.examples.bench_support import benchmark_cuda_graph_stats, settle
from attn_gym.warpdrive.examples.persistent_gemm import (
    PersistentGemm,
    compile_persistent_gemm,
)

CALLS_PER_GRAPH = 20
# (M, N, K): three short-K shapes whose epilogue is comparable to the mainloop, two long-K.
SHAPES = [
    (1024, 12800, 1152),
    (2048, 8192, 1024),
    (4096, 4096, 512),
    (4096, 4096, 4096),
    (3000, 5000, 4096),
]
SCHEDULES = {
    "static": StaticPersistent,
    "clc": ClusterLaunchControl,
    "atomic": AtomicPersistent,
}


def parse_variant(label: str) -> PersistentGemm:
    cluster, *options = label.split("_")
    if cluster not in ("single", "pair"):
        raise ValueError(f"variant {label!r} must start with mm, single or pair")
    config = {"cluster": CtaPair() if cluster == "pair" else None}
    for option in options:
        name = option.rstrip("0123456789")
        if name in SCHEDULES:
            stages = option[len(name) :]
            config["schedule"] = SCHEDULES[name](**({"stages": int(stages)} if stages else {}))
        elif option.startswith("n"):
            config["tile_n"] = int(option[1:])
        elif option.startswith("acc"):
            config["acc_stages"] = int(option[3:])
        elif option == "late":
            config["early_release"] = False
        else:
            raise ValueError(f"variant {label!r}: unknown option {option!r}")
    return PersistentGemm(**config)


def variants_for(a, b, labels: list[str]) -> dict:
    calls = {}
    for label in labels:
        if label == "mm":
            bt = b.T
            calls[label] = lambda bt=bt: torch.mm(a, bt)
            continue
        op = parse_variant(label)
        fn = compile_persistent_gemm(op)
        out = torch.empty(a.shape[0], b.shape[0], device=a.device, dtype=a.dtype)
        workspace, sms = op.work.launch_args(a.device)

        def call(fn=fn, out=out, workspace=workspace, sms=sms):
            fn(a, b, out, workspace, sms)
            return out

        calls[label] = call
    return calls


def run_shape(m, n, k, labels, samples, rounds) -> dict:
    torch.manual_seed(0)
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    ref = torch.nn.functional.linear(a.float(), b.float())
    calls = variants_for(a, b, labels)
    for label, call in calls.items():
        out = call()
        torch.cuda.synchronize()
        err = (out.float() - ref).abs().max().item()
        if not torch.isfinite(out).all() or err > 2**-6 * ref.abs().max().item() + 1e-2:
            raise AssertionError((label, err))
    graphs = {}
    for label, call in calls.items():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(CALLS_PER_GRAPH):
                call()
        graphs[label] = graph
        graph.replay()
    torch.cuda.synchronize()
    settle(graphs.values(), seconds=2.0)
    medians = {label: [] for label in calls}
    for round_idx in range(rounds):
        order = list(calls) if round_idx % 2 == 0 else list(reversed(calls))
        for label in order:
            stats = benchmark_cuda_graph_stats(
                graphs[label], num_iters=samples, warmup_iters=2, n_resamples=50
            )
            medians[label].append(stats.median_us / CALLS_PER_GRAPH)
    flops = 2 * m * n * k
    best = {label: min(v) for label, v in medians.items()}
    tflops = {label: flops / us / 1e6 for label, us in best.items()}
    print(f"M={m} N={n} K={k}", flush=True)
    for label, us in best.items():
        print(f"  {label:<32} {us:7.1f} us  {tflops[label]:5.0f} TFLOP/s", flush=True)
    return {
        "shape": [m, n, k],
        "medians_us": medians,
        "min_median_us": best,
        "tflops": tflops,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=int, nargs="*", choices=range(len(SHAPES)))
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["mm", "single", "single_late", "pair", "pair_late", "pair_n256", "pair_clc"],
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sys.argv = sys.argv[:1]  # Keep application flags out of CuTeDSL's diagnostic parser.
    shapes = SHAPES if not args.shape else [SHAPES[i] for i in args.shape]
    results = [run_shape(*shape, args.variants, args.samples, args.rounds) for shape in shapes]
    if args.output is not None:
        args.output.write_text(
            json.dumps({"gpu": torch.cuda.get_device_name(), "results": results}, indent=2)
        )


if __name__ == "__main__":
    main()
