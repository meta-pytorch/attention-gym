"""Benchmark the indexer's Top-K selection across backends and shapes.

Requires Transformer Nuggets with CUDA graph sample statistics:
    uv pip install "git+https://github.com/drisspg/transformer_nuggets.git@b8ae46be93f7c9d2133c025a7f15310484df8685"

Usage:
    python benchmarks/sparse/indexer_benchmark.py
    python benchmarks/sparse/indexer_benchmark.py --impl reference fused
    python benchmarks/sparse/indexer_benchmark.py --batch 4 --sequence-length 2048
"""

import argparse
from functools import partial

import torch

from attn_gym.sparse.indexer import lightning_indexer

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def useful_flops(args: argparse.Namespace) -> int:
    """Compute forward-pass FLOPs for the indexer's scoring step.

    Each query scores every candidate across every head via a dot product
    (q . k), which is one matmul-like reduction of 2*D FLOPs per (query,
    candidate, head) triple. The subsequent ReLU, per-head weighted sum, and
    Top-K selection are all O(1) per element (no reduction over D), so they
    are treated as negligible next to the dot-product FLOPs.
    """
    b = args.batch
    s = args.sequence_length
    h = args.heads
    d = args.head_dim
    if not args.causal:
        return b * h * s * s * d * 2
    else:
        return b * h * s * (s + 1) * d


def make_inputs(args: argparse.Namespace):
    """Create one shared set of inputs for every measured implementation."""
    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def randn(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator)

    q = randn(args.batch, args.sequence_length, args.heads, args.head_dim)
    k = randn(args.batch, args.sequence_length, args.head_dim)
    weights = randn(args.batch, args.sequence_length, args.heads)

    return q, k, weights


def parse_args() -> argparse.Namespace:
    """Parse shapes, implementations, and graph-replay iteration counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
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
        "--warmup", type=int, default=25, help="Warmup iterations before/after capture"
    )
    parser.add_argument("--rep", type=int, default=100, help="Number of timed graph replays")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    """Measure public forward selection with setup excluded from graph replay."""
    args = parse_args()
    try:
        from transformer_nuggets.utils.benchmark import benchmark_cuda_function_stats
    except ImportError:
        raise SystemExit(
            "This benchmark requires Transformer Nuggets with benchmark_cuda_function_stats. "
            "Install the compatible revision using the uv pip install command in --help."
        ) from None
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")

    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"shape: B={args.batch} H={args.heads} S={args.sequence_length} D={args.head_dim}")
    print(f"sparsity: topk={args.topk} causal={args.causal} dtype={args.dtype}")

    fwd_flops = useful_flops(args)

    print(
        "contract: warm fixed-pointer CUDA graph replay; forward only (indices have no backward)"
    )
    q, k, weights = make_inputs(args)
    for impl in args.impl:
        for backend in args.backend if impl == "fused" else [None]:
            fwd = partial(
                lightning_indexer,
                q,
                k,
                weights,
                args.topk,
                causal=args.causal,
                impl=impl,
                kernel_options={"backend": backend} if backend else None,
            )
            stats = benchmark_cuda_function_stats(
                fwd,
                USE_CUDA_GRAPHS=True,
                NUM_ITERS=args.rep,
                CUDAGRAPH_WARMUP_ITERS=args.warmup,
            )
            fwd_ms = stats.median_us / 1000
            fwd_tflops = fwd_flops / (fwd_ms * 1e9)
            route = f"{impl}/{backend}" if backend else impl
            print(f"[{route}] forward: {fwd_ms:.3f} ms  ({fwd_tflops:.2f} useful TFLOP/s)")


if __name__ == "__main__":
    main()
