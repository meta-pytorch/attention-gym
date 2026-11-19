"""Complete decode indexer timings, stage profiles, and partial roofline bounds.

Example (advertised B200 ceilings, not measured physical traffic):
    python benchmarks/sparse/paged_indexer_benchmark.py \
        --profile-rep 20 --peak-memory-gbps 8000 --peak-compute-tflops 2250 --reference

Requires transformer_nuggets with roofline.py and benchmark_cuda_function_stats.
The workload has one new query per sequence and a full, shuffled candidate pool.
"""

import argparse
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path

import torch

from attn_gym.sparse import paged_lightning_indexer
from attn_gym.testing.indexer import (
    assert_paged_indexer_selection,
    make_paged_indexer_test_inputs,
)


def stage_work(
    batch, candidates, heads, dim, topk, page_size, *, capacity=None, zero_weights=False
):
    """Include score intermediates and selection I/O; mark integer work unknown.

    Radix selection uses integer instructions, shared histograms, atomics, and
    barriers. Tensor-core TFLOP/s cannot bound these operations. Its minimum I/O
    is still useful, but neither this stage nor the complete operation has a
    modeled compute ceiling. Logical traffic does not predict DRAM/L2 traffic.
    """
    from transformer_nuggets.roofline import RooflineWork

    if topk == 0:
        return {}
    if candidates <= topk or zero_weights:
        work = {
            "selection": RooflineWork(
                None,
                4 * (2 * batch + 1),
                4 * batch * topk,
                "direct visible IDs; integer work unmodeled",
                confidence="unknown",
            )
        }
        if (capacity or candidates) > topk:
            work["scoring"] = RooflineWork(
                0,
                4 * (2 * batch + 1) + (2 * batch * heads if candidates > topk else 0),
                8 * batch,
                "visibility/zero-weight shortcut; no query or key reads",
                confidence="medium",
            )
        return work
    # FP16/BF16 Q/K/weights. Read each key once per sequence, shared across heads.
    # Page table, query offsets, raw lengths, and the returned INT32 IDs.
    pairs = batch * candidates
    reads = 2 * batch * (heads * dim + candidates * dim + heads)
    reads += 4 * (batch * ((candidates + page_size - 1) // page_size) + 2 * batch + 1)
    return {
        "scoring": RooflineWork(
            logical_flops=pairs * heads * (2 * dim + 3),
            read_bytes=reads,
            write_bytes=4 * pairs,
            model_kind="paged weighted-ReLU scoring including FP32 score writes",
            confidence="medium",
        ),
        "selection": RooflineWork(
            logical_flops=None,
            read_bytes=4 * pairs,
            write_bytes=4 * batch * topk,
            model_kind="Top-K minimum score reads and ID writes; integer work unmodeled",
            confidence="unknown",
        ),
    }


def capture_graph(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def profile_stages(fn, iterations, trace_path=None):
    """Time kernels in the complete graph, preserving scoring-to-selection reuse.

    These profiler means are diagnostics, separate from unprofiled CUDA-event
    medians. Do not add them to the event latency or interpret the difference
    between separately measured totals as a precise launch overhead.
    """
    graph = capture_graph(fn)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as profiler:
        for _ in range(iterations):
            graph.replay()
        torch.cuda.synchronize()
    if trace_path:
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))
    return summarize_stage_events(profiler.events(), iterations)


def summarize_stage_events(events, iterations):
    """Attribute every GPU event, keeping unknown kernels visible for coverage checks."""
    stages = {}
    for event in events:
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        if event.name.startswith(("_score_tma", "_score")):
            stage = "scoring"
        elif event.name.startswith(("_select", "indexer_topk_")):
            stage = "selection"
        else:
            stage = "other"
        entry = stages.setdefault(stage, {"total_us": 0.0, "count": 0, "kernels": set()})
        entry["total_us"] += event.device_time_total
        entry["count"] += 1
        entry["kernels"].add(event.name)
    for stage in stages.values():
        if stage["count"] % iterations:
            raise RuntimeError("Incomplete profiler kernel coverage across graph replays.")
        stage["profiled_mean_us"] = stage.pop("total_us") / iterations
        stage["launches_per_call"] = stage.pop("count") // iterations
        stage["kernels"] = sorted(stage["kernels"])
    return stages


def partial_bounds(work, stages, spec):
    """Sum serial stage bounds without claiming a full-operation efficiency."""
    from transformer_nuggets.roofline import _runtime_metrics

    if set(work) != set(stages):
        raise RuntimeError(f"Kernel/model coverage mismatch: {set(stages)} vs {set(work)}")
    bounds = {}
    for name, model in work.items():
        profile = stages[name]
        metrics = _runtime_metrics(
            model,
            profile["profiled_mean_us"],
            spec,
            launch_count=profile["launches_per_call"],
        )
        # Unknown integer/atomic/barrier work must not become a zero-FLOP
        # complete roofline or an advertised achieved-roofline percentage.
        metrics.pop("achieved_roofline_percent", None)
        metrics.pop("achieved_memory_roofline_percent", None)
        metrics["partial_floor_us"] = metrics.pop("effective_floor_us")
        bounds[name] = metrics
    return {
        "stages": bounds,
        "partial_serial_floor_us": sum(stage["partial_floor_us"] for stage in bounds.values()),
        "full_roofline_percent": None,
        "missing_model": "selection integer instructions, shared atomics, and synchronization",
    }


def dense_selection(args, topk, candidates):
    """Capture-safe baseline that gathers the whole pool, scores, then torch.topk."""
    q, cache, weights, table, _, _ = args
    keys = cache[table[:, : candidates // cache.shape[1]].long()].flatten(1, 2)
    dots = torch.matmul(q.float(), keys.float().transpose(-1, -2))
    scores = (dots.relu() * weights.float().unsqueeze(-1)).sum(1)
    output = scores.topk(min(topk, candidates), dim=-1).indices.int()
    if candidates < topk:
        output = torch.nn.functional.pad(output, (0, topk - candidates), value=-1)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 16, 64])
    parser.add_argument("--candidates", type=int, nargs="+", default=[1024, 8192, 32768])
    parser.add_argument("--capacity", type=int, help="Reserved candidates per sequence (>= live)")
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--backend", choices=["auto", "triton", "cute"], default="auto")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument(
        "--weights", choices=["mixed", "positive", "negative", "zero"], default="mixed"
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--peak-memory-gbps", type=float)
    parser.add_argument("--peak-compute-tflops", type=float)
    parser.add_argument("--launch-latency-us", type=float, default=1.0)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profile-rep", type=int, default=0, help="Optional stage profiling iterations"
    )
    parser.add_argument("--trace", type=Path, help="Raw traces, suffixed by batch/candidate count")
    parser.add_argument(
        "--ncu",
        action="store_true",
        help="Warm and profile one full graph using cudaProfilerStart/Stop; choose one case",
    )
    config = parser.parse_args()
    if config.rep <= 0 or config.profile_rep < 0 or config.warmup <= 0:
        parser.error("rep/warmup must be positive and profile-rep nonnegative")
    if config.profile_rep and (
        config.peak_memory_gbps is None or config.peak_compute_tflops is None
    ):
        parser.error("stage profiling requires peak-memory-gbps and peak-compute-tflops")
    if config.trace and not config.profile_rep:
        parser.error("--trace requires --profile-rep")
    if config.ncu and (len(config.batch) != 1 or len(config.candidates) != 1):
        parser.error("--ncu requires exactly one batch and candidate count")
    try:
        from transformer_nuggets.roofline import RooflineSpec
        from transformer_nuggets.utils.benchmark import benchmark_cuda_function_stats
    except ImportError:
        raise SystemExit(
            "Install Transformer Nuggets with roofline.py and benchmark_cuda_function_stats, "
            "or add its checkout to PYTHONPATH."
        ) from None
    device_name = torch.cuda.get_device_name()
    spec = (
        RooflineSpec(
            device_name,
            config.peak_compute_tflops,
            config.peak_memory_gbps,
            config.launch_latency_us,
        )
        if config.profile_rep
        else None
    )
    torch.use_deterministic_algorithms(config.deterministic)
    print(
        f"{device_name}: warm fixed-pointer CUDA Graph replay; {config.dtype}; one query/sequence"
    )
    print("Full-operation roofline efficiency: unknown (integer/atomic Top-K ceiling unmodeled).")
    print("Stage bounds include score writes/reads; kernel profiles include complete Top-K work.")
    print("batch candidates capacity implementation median_us scoring_us topk_us topk_kernel_%")
    results = []
    for batch in config.batch:
        for candidates in config.candidates:
            capacity = config.capacity or candidates
            if candidates <= 0 or candidates % config.page_size:
                parser.error("candidates must be a positive multiple of page-size")
            if capacity < candidates or capacity % config.page_size:
                parser.error("capacity must be a multiple of page-size and at least candidates")
            args = list(
                make_paged_indexer_test_inputs(
                    (1,) * batch,
                    (4 * candidates,) * batch,
                    heads=config.heads,
                    head_dim=config.head_dim,
                    page_size=config.page_size,
                    dtype=getattr(torch, config.dtype),
                    seed=config.seed,
                )
            )
            if config.weights in ("positive", "negative"):
                args[2].abs_()
                if config.weights == "negative":
                    args[2].neg_()
            elif config.weights == "zero":
                args[2].zero_()
            if capacity != candidates:
                table = args[3].new_full((batch, capacity // config.page_size), -1)
                table[:, : args[3].shape[1]] = args[3]
                args[3] = table
            options = None if config.backend == "auto" else {"backend": config.backend}

            fused = partial(
                paged_lightning_indexer, *args, topk=config.topk, kernel_options=options
            )

            if config.ncu:
                graph = capture_graph(fused)
                torch.cuda.cudart().cudaProfilerStart()
                graph.replay()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                return
            assert_paged_indexer_selection(args, fused(), config.topk)
            routes = [("fused/" + config.backend, fused)]
            if config.reference:
                routes.append(
                    ("gather+torch", partial(dense_selection, args, config.topk, candidates))
                )
            for name, fn in routes:
                stats = benchmark_cuda_function_stats(
                    fn,
                    USE_CUDA_GRAPHS=True,
                    NUM_ITERS=config.rep,
                    CUDAGRAPH_WARMUP_ITERS=config.warmup,
                )
                result = {
                    "batch": batch,
                    "candidates": candidates,
                    "capacity": capacity,
                    "implementation": name,
                    "timing": asdict(stats),
                }
                score_us = topk_us = share = 0.0
                if name.startswith("fused/") and config.profile_rep:
                    trace = config.trace
                    if trace:
                        trace = trace.with_name(
                            f"{trace.stem}.b{batch}.s{candidates}.c{capacity}{trace.suffix}"
                        )
                    stages = profile_stages(fn, config.profile_rep, trace)
                    work = stage_work(
                        batch,
                        candidates,
                        config.heads,
                        config.head_dim,
                        config.topk,
                        config.page_size,
                        capacity=capacity,
                        zero_weights=config.weights == "zero",
                    )
                    result["profile"] = stages
                    result["bounds"] = partial_bounds(work, stages, spec)
                    score_us = stages.get("scoring", {}).get("profiled_mean_us", 0)
                    topk_us = stages.get("selection", {}).get("profiled_mean_us", 0)
                    share = 100 * topk_us / max(score_us + topk_us, 1e-12)
                    result["topk_share_of_profiled_kernel_time_percent"] = share
                results.append(result)
                detail = (
                    f"{score_us:10.3f} {topk_us:8.3f} {share:13.1f}"
                    if "profile" in result
                    else "         -        -             -"
                )
                print(
                    f"{batch:5} {candidates:10} {capacity:8} {name:14} "
                    f"{stats.median_us:9.3f} {detail}",
                    flush=True,
                )
    if config.output:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(
            json.dumps(
                {
                    "device": asdict(spec) if spec else {"name": device_name},
                    "config": vars(config),
                    "torch_version": torch.__version__,
                    "results": results,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
