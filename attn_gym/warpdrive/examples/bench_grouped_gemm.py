"""Compare grouped GEMM backends with CUDA graphs on skewed MoE routing.

Baselines: ``torch._grouped_mm``, ``F.grouped_mm``, per-group ``torch.mm``, Inductor's Triton
grouped GEMM template and FlexGEMM's QUACK backend.

Run under gpu-run. Compilation, Inductor autotuning, allocation, reference checks and capture
are untimed. Each graph holds ``CALLS_PER_GRAPH`` calls; times are microseconds per call.
Alternate variant order across rounds and report the minimum round median. Shared-GPU
interference can still affect it. TFLOP/s uses ``2 * sum(size_g) * N * K``.

Print launched kernel names to identify the compiled route: FlexGEMM's TRITON backend without
max-autotune uses extern ``_grouped_mm``; NVGEMM has no grouped GEMM choice.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch._inductor.config as inductor_config
import torch.nn.functional as F
from torch._higher_order_ops.flex_gemm import flex_gemm
from torch.profiler import ProfilerActivity, profile

from attn_gym.warpdrive import (
    AtomicPersistent,
    ClusterLaunchControl,
    CtaPair,
    StaticPersistent,
)
from attn_gym.warpdrive.examples.bench_support import benchmark_cuda_graph_stats, settle
from attn_gym.warpdrive.examples.grouped_gemm import (
    GroupedGemm,
    compile_grouped_gemm,
    grouped_gemm_reference,
)

CALLS_PER_GRAPH = 20
# (experts, total tokens, N, K)
SHAPES = [
    (8, 4096, 2048, 7168),
    (8, 16384, 7168, 2048),
    (64, 8192, 4096, 4096),
    (64, 32768, 2048, 7168),
    (128, 16384, 7168, 2048),
    (128, 32768, 4096, 4096),
]


def skewed_sizes(groups: int, total: int, seed: int = 0) -> list[int]:
    """Generate skewed expert token counts summing to ``total``, including empty experts.

    Counts are not rounded to tile multiples.
    """
    gen = torch.Generator().manual_seed(seed)
    weights = torch.rand(groups, generator=gen) ** 3  # cube a uniform: a long tail
    weights[torch.randperm(groups, generator=gen)[: groups // 8]] = 0  # idle experts
    sizes = (weights / weights.sum() * total).floor().to(torch.int64)
    sizes[sizes.argmax()] += total - int(sizes.sum())
    return sizes.tolist()


def _identity(acc):
    return acc


def _flex_grouped_mm(a, bt, offs, options):
    return flex_gemm(
        F.grouped_mm, (a, bt), _identity, gemm_kwargs={"offs": offs}, kernel_options=options
    )


def _grouped_mm(a, bt, offs):
    return F.grouped_mm(a, bt, offs=offs)


def compile_flex_quack(a, bt, offsets, tuned: bool):
    """Compile and warm up FlexGEMM's QUACK backend with an identity epilogue.

    ``tuned`` benchmarks the config shortlist. PyTorch accepts this option as
    ``autotune`` or ``tuned``, depending on version.
    """
    compiled = torch.compile(_flex_grouped_mm, backend="inductor", fullgraph=True, dynamic=False)
    options = {"backend": "QUACK"}
    if not tuned:
        compiled(a, bt, offsets, options)
        return lambda: compiled(a, bt, offsets, options)
    for key in ("autotune", "tuned"):
        tuned_options = {**options, key: True}
        try:
            compiled(a, bt, offsets, tuned_options)
        except Exception as exc:  # the other spelling is rejected while lowering
            if "unsupported FlexGEMM kernel options" not in str(exc):
                raise
            continue
        return lambda: compiled(a, bt, offsets, tuned_options)
    raise RuntimeError("FlexGEMM accepts neither autotune=True nor tuned=True")


def compile_triton_template(a, bt, offsets):
    """Compile and warm up Inductor's persistent TMA Triton grouped GEMM template.

    Max-autotune uses Triton as its only GEMM backend.
    """
    compiled = torch.compile(_grouped_mm, backend="inductor", fullgraph=True, dynamic=False)
    with inductor_config.patch(max_autotune=True, max_autotune_gemm_backends="TRITON"):
        compiled(a, bt, offsets)
    return lambda: compiled(a, bt, offsets)


def kernels_per_call(call) -> list[str]:
    """Return launched kernel names to check which compiled route ran."""
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        call()
        torch.cuda.synchronize()
    return [e.key for e in prof.key_averages() if e.device_time_total > 0]


def variants_for(a, b, offsets, labels: list[str]) -> dict:
    calls = {}
    bt = b.transpose(1, 2)  # (G, K, N) with K contiguous: the layout every baseline takes
    for label in labels:
        if label == "grouped_mm":
            calls[label] = lambda: torch._grouped_mm(a, bt, offs=offsets)
        elif label == "f_grouped_mm":
            calls[label] = lambda: F.grouped_mm(a, bt, offs=offsets)
        elif label == "triton_tem":
            calls[label] = compile_triton_template(a, bt, offsets)
        elif label in ("flex_quack", "flex_quack_tuned"):
            calls[label] = compile_flex_quack(a, bt, offsets, tuned=label.endswith("tuned"))
        elif label == "mm_loop":
            ends = offsets.tolist()
            bounds = [(g, s, e) for g, (s, e) in enumerate(zip([0, *ends[:-1]], ends)) if e > s]
            out = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)

            def loop(bounds=bounds, out=out):
                for g, s, e in bounds:
                    torch.mm(a[s:e], b[g].T, out=out[s:e])
                return out

            calls[label] = loop
        else:
            cluster, *options = label.split("_")
            schedules = {
                "atomic": AtomicPersistent(),
                "clc": ClusterLaunchControl(),
                "static": StaticPersistent(),
            }
            op = GroupedGemm(
                cluster=CtaPair() if cluster == "pair" else None,
                schedule=next((schedules[o] for o in options if o in schedules), None),
                tile_n=next((int(o[1:]) for o in options if o.startswith("n")), 256),
            )
            fn = compile_grouped_gemm(op)
            out = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=a.dtype)
            workspace, sms = op.work.launch_args(a.device)

            def call(fn=fn, out=out, workspace=workspace, sms=sms):
                fn(a, b, out, offsets, workspace, sms)
                return out

            calls[label] = call
    return calls


def run_shape(groups, tokens, n, k, labels, samples, rounds) -> dict:
    torch._dynamo.reset()  # each shape recompiles the static-shape variants; stay under the limit
    torch.manual_seed(0)
    sizes = skewed_sizes(groups, tokens)
    offsets = torch.tensor(sizes, device="cuda").cumsum(0).to(torch.int32)
    a = torch.randn(tokens, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(groups, n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    ref = grouped_gemm_reference(a, b, offsets)
    calls = variants_for(a, b, offsets, labels)
    kernels = {}
    for label, call in calls.items():
        out = call()
        torch.cuda.synchronize()
        err = (out.float() - ref).abs().max().item()
        if not torch.isfinite(out).all() or err > 2**-6 * ref.abs().max().item() + 1e-2:
            raise AssertionError((label, err))
        print(f"{label}: max abs err {err:.3g}", flush=True)
        if label.startswith(("flex_", "triton_")):
            kernels[label] = kernels_per_call(call)
            print(f"{label}: kernels {[k[:48] for k in kernels[label]]}", flush=True)
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
            stats = benchmark_cuda_graph_stats(graphs[label], num_iters=samples, warmup_iters=2)
            medians[label].append(stats.median_us / CALLS_PER_GRAPH)
    flops = 2 * sum(sizes) * n * k
    best = {label: min(v) for label, v in medians.items()}
    tflops = {label: flops / us / 1e6 for label, us in best.items()}
    shape = f"G={groups} M={tokens} N={n} K={k}"
    print(
        shape,
        "  ".join(
            f"{label} {us:.1f} us ({tflops[label]:.0f} TFLOP/s)" for label, us in best.items()
        ),
        flush=True,
    )
    return {
        "shape": [groups, tokens, n, k],
        "sizes": sizes,
        "kernels": kernels,
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
        default=["grouped_mm", "mm_loop", "triton_tem", "flex_quack", "single", "pair"],
        help="grouped_mm (torch._grouped_mm), f_grouped_mm (F.grouped_mm), mm_loop, triton_tem "
        "(Inductor's Triton grouped GEMM template), flex_quack / flex_quack_tuned (FlexGEMM, "
        "QUACK backend), or single/pair with optional _atomic/_clc/_static and _n<tile_n>",
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
