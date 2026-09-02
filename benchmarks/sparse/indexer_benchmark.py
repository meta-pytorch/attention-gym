"""Benchmark the indexer's Top-K selection across backends and shapes.

Usage:
    python benchmarks/sparse/indexer_benchmark.py
    python benchmarks/sparse/indexer_benchmark.py --backend eager cute
    python benchmarks/sparse/indexer_benchmark.py --batch 4 --sequence-length 2048
"""

import argparse

import torch
import triton

from attn_gym.sparse.indexer import index

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
    return b * h * s * s * d * 2


def make_inputs(args: argparse.Namespace):
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=128)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument(
        "--backend", nargs="+", default=["eager"], choices=["eager", "triton", "cute"]
    )
    parser.add_argument("--warmup", type=int, default=200, help="Warmup duration in ms")
    parser.add_argument("--rep", type=int, default=1000, help="Measurement duration in ms")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "cute" in args.backend:
        assert args.heads % 2 == 0, "cute backend requires an even number of heads"
        assert args.head_dim % 16 == 0, "cute backend requires head_dim divisible by 16"
        assert args.dtype in ("float16", "bfloat16"), "cute backend requires fp16 or bf16"
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")

    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(
        f"shape: B={args.batch} H={args.heads} S={args.sequence_length} D={args.head_dim}"
    )
    print(f"sparsity: topk={args.topk} causal={args.causal} dtype={args.dtype}")

    fwd_flops = useful_flops(args)

    for backend in args.backend:
        q, k, weights = make_inputs(args)

        def fwd(_q=q, _k=k, _weights=weights, _backend=backend):
            return index(_q, _k, _weights, args.topk, causal=args.causal, backend=_backend)

        fwd()
        fwd_ms = triton.testing.do_bench(
            fwd, warmup=args.warmup, rep=args.rep, return_mode="median"
        )
        fwd_tflops = fwd_flops / (fwd_ms * 1e9)

        print(f"[{backend}] forward: {fwd_ms:.3f} ms  ({fwd_tflops:.2f} TFLOP/s)")


if __name__ == "__main__":
    main()
