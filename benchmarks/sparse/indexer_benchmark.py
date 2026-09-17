"""Benchmark the indexer's Top-K selection across backends and shapes.

Usage:
    python benchmarks/sparse/indexer_benchmark.py
    python benchmarks/sparse/indexer_benchmark.py --impl reference fused
    python benchmarks/sparse/indexer_benchmark.py --batch 4 --sequence-length 2048
    python benchmarks/sparse/indexer_benchmark.py --dtype mxfp8 --backend cute triton --selector default gvr2
"""

import argparse
import statistics
from collections.abc import Callable
from functools import partial

import torch

from attn_gym.sparse.indexer import lightning_indexer

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "mxfp8": torch.float8_e4m3fn,
}


def benchmark_graph(fn: Callable[[], object], *, warmup: int, samples: int) -> list[float]:
    """Time captured device work with event nodes inside the graph, in microseconds.

    Host submission and synchronization are outside the interval. GPU graph
    scheduling and every captured operation remain included.
    """
    if warmup < 0 or samples < 1:
        raise ValueError("warmup must be nonnegative and samples must be positive")
    for _ in range(warmup):
        _output = fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        start.record()
        # Retain allocating callables' output through all replays.
        _output = fn()
        end.record()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    durations = []
    for _ in range(samples):
        graph.replay()
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end) * 1000)
    return durations


def useful_flops(args: argparse.Namespace) -> int:
    """Compute forward-pass FLOPs for the indexer's scoring step.

    Count 2*D FLOPs for each valid query/candidate/head dot product. ReLU,
    head reduction, and selection remain in the timed call but are not added
    to this useful-work numerator.
    """
    tokens = args.sequence_length
    candidates = tokens // args.compress_ratio
    if args.causal:
        pairs = args.compress_ratio * candidates * (candidates - 1) // 2
        pairs += candidates * (tokens % args.compress_ratio + 1)
    else:
        pairs = tokens * candidates
    return 2 * args.batch * args.heads * args.head_dim * pairs


def quantize_mxfp8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare E4M3 data and group-32 E8M0 scales outside the timed region."""
    blocks = tensor.float().unflatten(-1, (tensor.shape[-1] // 32, 32))
    maximum = blocks.abs().amax(-1)
    maximum = torch.where(maximum > 0, maximum, 1.0)
    exponent = torch.ceil(torch.log2(maximum / 448.0)).clamp(-127, 127)
    scale = torch.exp2(exponent).to(torch.float8_e8m0fnu)
    data = (blocks / scale.float().unsqueeze(-1)).flatten(-2).to(torch.float8_e4m3fn)
    return data, scale


def make_inputs(args: argparse.Namespace):
    """Create one shared set of inputs for every measured implementation."""
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "mxfp8" else DTYPES[args.dtype]
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    q = randn(args.batch, args.sequence_length, args.heads, args.head_dim)
    k = randn(args.batch, args.sequence_length // args.compress_ratio, args.head_dim)
    weights = randn(args.batch, args.sequence_length, args.heads)
    q_scale = k_scale = None
    if args.dtype == "mxfp8":
        q, q_scale = quantize_mxfp8(q)
        k, k_scale = quantize_mxfp8(k)
    return q, k, weights, q_scale, k_scale


def parse_args() -> argparse.Namespace:
    """Parse shapes, implementations, and graph-replay iteration counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--compress-ratio", type=int, default=1)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--impl", nargs="+", default=["fused"], choices=["reference", "fused"])
    parser.add_argument(
        "--backend",
        nargs="+",
        choices=["cute", "triton"],
        default=[None],
        help="Override fused backend selection; omit to select by device",
    )
    parser.add_argument(
        "--selector", nargs="+", choices=["auto", "default", "gvr2"], default=["auto"]
    )
    parser.add_argument(
        "--warmup", type=int, default=25, help="Warmup iterations before/after capture"
    )
    parser.add_argument("--rep", type=int, default=100, help="Number of timed graph replays")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.compress_ratio < 1:
        parser.error("--compress-ratio must be positive")
    if args.dtype == "mxfp8" and args.head_dim % 32:
        parser.error("MXFP8 requires --head-dim divisible by 32")
    return args


def main() -> None:
    """Measure public forward selection with setup excluded from graph replay."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")

    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"shape: B={args.batch} H={args.heads} S={args.sequence_length} D={args.head_dim}")
    print(
        f"sparsity: topk={args.topk} causal={args.causal} dtype={args.dtype} "
        f"compress_ratio={args.compress_ratio} candidates={args.sequence_length // args.compress_ratio}"
    )

    fwd_flops = useful_flops(args)

    print("contract: warm fixed-pointer CUDA graph, timing events inside capture; forward only")
    q, k, weights, q_scale, k_scale = make_inputs(args)
    if args.dtype == "mxfp8":
        print("MXFP8 group-32 quantization and scale preparation excluded from timing")
    for impl in args.impl:
        for backend in args.backend if impl == "fused" else [None]:
            for selector in args.selector if impl == "fused" else ["default"]:
                options = {"selector": selector} if impl == "fused" else None
                if backend:
                    options["backend"] = backend
                fwd = partial(
                    lightning_indexer,
                    q,
                    k,
                    weights,
                    args.topk,
                    causal=args.causal,
                    compress_ratio=args.compress_ratio,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    impl=impl,
                    kernel_options=options,
                )
                samples = benchmark_graph(fwd, warmup=args.warmup, samples=args.rep)
                fwd_ms = statistics.median(samples) / 1000
                fwd_tflops = fwd_flops / (fwd_ms * 1e9)
                route = f"{impl}/{backend or 'auto'}/{selector}" if impl == "fused" else impl
                print(f"[{route}] forward: {fwd_ms:.3f} ms  ({fwd_tflops:.2f} useful TFLOP/s)")


if __name__ == "__main__":
    main()
