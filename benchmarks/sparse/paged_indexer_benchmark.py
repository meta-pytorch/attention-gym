"""Complete decode indexer timings, stage profiles, and logical HBM/launch estimates.

Example (advertised B200 ceilings, not measured physical traffic):
    python benchmarks/sparse/paged_indexer_benchmark.py \
        --profile-rep 20 --peak-memory-gbps 8000 --peak-compute-tflops 2250 --reference

Install the same Transformer Nuggets revision as the whole-document indexer benchmark:
    uv pip install "git+https://github.com/drisspg/transformer_nuggets.git@020d19f700a35bd9be7417a4a3111156204bd23b"
Stage profiling also requires cuda.bindings for captured graph inspection.
The workload has one new query per sequence and a full, shuffled candidate pool.
"""

import argparse
import json
from collections import Counter
from collections.abc import Iterable
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
    batch, candidates, heads, dim, topk, page_size, *, capacity, cute_selection, zero_weights=False
):
    """Count unique logical payloads, not cache-line transactions or repeated radix reads."""
    from transformer_nuggets.roofline import RooflineWork

    if topk == 0:
        return {}
    # Each sequence's raw length and query end offset (cu_seqlens_q[1:]).
    visibility_bytes = 8 * batch
    pairs = batch * candidates if candidates > topk and not zero_weights else 0
    score_bytes = 4 * pairs
    has_bounds = capacity > topk
    # CuTe reads [start, end); Triton only needs end. Without scoring, the
    # Triton selector derives visibility directly from lengths and offsets.
    selection_metadata = (8 if cute_selection else 4) * batch if has_bounds else visibility_bytes
    work = {
        "selection": RooflineWork(
            None,
            selection_metadata + score_bytes,
            4 * batch * topk,
            "Top-K logical inputs and output IDs; integer work unmodeled",
            confidence="unknown",
        )
    }
    if has_bounds:
        reads = visibility_bytes
        if candidates > topk:
            reads += 2 * batch * heads  # FP16/BF16 weights, even for the zero-weight shortcut
        if pairs:
            reads += 2 * batch * (heads * dim + candidates * dim)
            reads += 4 * batch * ((candidates + page_size - 1) // page_size)
        work["scoring"] = RooflineWork(
            pairs * heads * (2 * dim + 3),
            reads,
            8 * batch + score_bytes,
            "visibility bounds and, for scored rows, paged weighted-ReLU scores",
            confidence="medium",
        )
    return work


def capture_graph(fn, warmup: int, *, keep_graph: bool = False) -> torch.cuda.CUDAGraph:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=keep_graph)
    with torch.cuda.graph(graph):
        fn()
    return graph


def captured_kernel_names(graph: torch.cuda.CUDAGraph) -> list[str]:
    """Inventory this kernel-only graph independently of Kineto's event stream."""
    from cuda.bindings import driver

    def checked(result):
        error, *values = result
        if error != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"Cannot inspect the paged indexer's kernel-only graph: {error}")
        return values

    raw = graph.raw_cuda_graph()
    _, count = checked(driver.cuGraphGetNodes(raw))
    nodes, _ = checked(driver.cuGraphGetNodes(raw, count))
    names = []
    for node in nodes:
        (params,) = checked(driver.cuGraphKernelNodeGetParams(node))
        (symbol,) = checked(driver.cuFuncGetName(params.func))
        names.append(torch._C._demangle(symbol.decode()))
    return names


def profile_stages(graph, iterations, trace_path=None):
    """Time kernels in the complete graph, preserving scoring-to-selection reuse.

    These profiler means are diagnostics, separate from unprofiled CUDA-event
    medians. Do not add them to the event latency or interpret the difference
    between separately measured totals as a precise launch overhead.
    """
    kernel_names = captured_kernel_names(graph)
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
    return summarize_stage_events(profiler.events(), iterations, kernel_names)


def summarize_stage_events(events, iterations: int, kernel_names: Iterable[str]):
    """Require the profiler to account for the launch count of each captured kernel."""
    expected = Counter({name: n * iterations for name, n in Counter(kernel_names).items()})
    observed = Counter()
    stages = {}
    for event in events:
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        observed[event.name] += 1
        if event.name.startswith("_score"):
            stage = "scoring"
        elif event.name.startswith(("_select", "indexer_topk_")):
            stage = "selection"
        elif "at::native::FillFunctor<" in event.name:
            stage = "initialization"
        else:
            raise RuntimeError(f"Unrecognized paged indexer kernel: {event.name}")
        entry = stages.setdefault(stage, {"total_us": 0.0, "count": 0, "kernels": set()})
        entry["total_us"] += event.device_time_total
        entry["count"] += 1
        entry["kernels"].add(event.name)
    if observed != expected:
        raise RuntimeError(
            f"Profiler/graph kernel count mismatch: missing={expected - observed}, "
            f"unexpected={observed - expected}"
        )
    for stage in stages.values():
        stage["profiled_mean_us"] = stage.pop("total_us") / iterations
        stage["launches_per_call"] = stage.pop("count") // iterations
        stage["kernels"] = sorted(stage["kernels"])
    return stages


def model_scenario(work, stages, spec):
    """Estimate serial modeled work if all logical bytes cross HBM at the supplied peak."""
    from transformer_nuggets.roofline import _runtime_metrics

    if set(work) != stages.keys() - {"initialization"}:
        raise RuntimeError(f"Kernel/model coverage mismatch: {set(stages)} vs {set(work)}")
    estimates = {}
    for name, model in work.items():
        profile = stages[name]
        metrics = _runtime_metrics(
            model,
            profile["profiled_mean_us"],
            spec,
            launch_count=profile["launches_per_call"],
        )
        estimates[name] = {
            **{
                k: v
                for k, v in metrics.items()
                if k.startswith(("logical_", "model_", "achieved_logical_"))
            },
            "all_logical_bytes_through_hbm_us": metrics["memory_floor_us"],
            "useful_compute_at_peak_us": metrics["compute_floor_us"],
            "assumed_launch_cost_us": metrics["launch_floor_us"],
            "stage_estimate_us": metrics["effective_floor_us"],
        }
    return {
        "stages": estimates,
        "modeled_serial_estimate_us": sum(s["stage_estimate_us"] for s in estimates.values()),
        "unmodeled_profiled_stages": sorted(stages.keys() - work.keys()),
        "full_roofline_percent": None,
        "missing_model": "selection integer instructions, shared atomics, and synchronization",
    }


def dense_selection(args, topk, candidates):
    """Gather and score the live pool; match the requested tie and output ordering."""
    q, cache, weights, table, _, _ = args
    keys = cache[table[:, : candidates // cache.shape[1]].long()].flatten(1, 2)
    dots = torch.matmul(q.float(), keys.float().transpose(-1, -2))
    scores = (dots.relu() * weights.float().unsqueeze(-1)).sum(1)
    scores *= (q.shape[1] * q.shape[2]) ** -0.5
    width = min(topk, candidates)
    if torch.are_deterministic_algorithms_enabled():
        ranked = scores.argsort(dim=-1, descending=True, stable=True)
        output = ranked[:, :width].sort(dim=-1).values.int()
    else:
        output = scores.topk(width, dim=-1).indices.int()
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
        from transformer_nuggets.utils.benchmark import benchmark_cuda_graph_stats
    except ImportError:
        raise SystemExit(
            "Install the compatible Transformer Nuggets revision shown in --help."
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
    print(
        "Model scenario: all logical bytes through HBM, assumed per-node launch cost; "
        "not a lower bound on warm replay."
    )
    print(
        "batch candidates capacity implementation median_us init_us scoring_us topk_us topk_kernel_%"
    )
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

            assert_paged_indexer_selection(args, fused(), config.topk)
            if config.ncu:
                graph = capture_graph(fused, config.warmup)
                for _ in range(config.warmup):
                    graph.replay()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStart()
                graph.replay()
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
                return
            routes = [("fused/" + config.backend, fused)]
            if config.reference:
                dense = partial(dense_selection, args, config.topk, candidates)
                assert_paged_indexer_selection(args, dense(), config.topk)
                routes.append(("gather+torch", dense))
            for name, fn in routes:
                profile = fn is fused and config.profile_rep > 0
                graph = capture_graph(fn, config.warmup, keep_graph=profile)
                stats = benchmark_cuda_graph_stats(
                    graph, num_iters=config.rep, warmup_iters=config.warmup
                )
                result = {
                    "batch": batch,
                    "candidates": candidates,
                    "capacity": capacity,
                    "implementation": name,
                    "timing": asdict(stats),
                }
                if profile:
                    trace = config.trace
                    if trace:
                        trace = trace.with_name(
                            f"{trace.stem}.b{batch}.s{candidates}.c{capacity}{trace.suffix}"
                        )
                    stages = profile_stages(graph, config.profile_rep, trace)
                    work = stage_work(
                        batch,
                        candidates,
                        config.heads,
                        config.head_dim,
                        config.topk,
                        config.page_size,
                        capacity=capacity,
                        zero_weights=config.weights == "zero",
                        cute_selection=any(
                            k.startswith("indexer_topk_")
                            for k in stages.get("selection", {}).get("kernels", ())
                        ),
                    )
                    result["profile"] = stages
                    result["model_scenario"] = model_scenario(work, stages, spec)
                    init_us = stages.get("initialization", {}).get("profiled_mean_us", 0)
                    score_us = stages.get("scoring", {}).get("profiled_mean_us", 0)
                    topk_us = stages.get("selection", {}).get("profiled_mean_us", 0)
                    kernel_us = sum(s["profiled_mean_us"] for s in stages.values())
                    share = 100 * topk_us / kernel_us if kernel_us else 0.0
                    result["topk_share_of_profiled_kernel_time_percent"] = share
                    detail = f"{init_us:7.3f} {score_us:10.3f} {topk_us:8.3f} {share:13.1f}"
                else:
                    detail = "      -          -        -             -"
                results.append(result)
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
                    "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
                    "results": results,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
