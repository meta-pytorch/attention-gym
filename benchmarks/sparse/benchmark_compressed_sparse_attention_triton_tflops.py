"""Benchmark Triton CSA useful sparse throughput without running the eager reference."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch
import triton

from attn_gym.sparse import compressed_sparse_attention
from benchmarks.sparse.benchmark_compressed_sparse_attention_triton import (
    DTYPES,
    make_inputs,
    useful_matmul_flops,
)


def benchmark_triton_flops(
    operation: Callable[[], object],
    args: argparse.Namespace,
) -> tuple[float, float]:
    """Return median Triton latency in milliseconds and useful sparse TFLOP/s."""
    triton_ms = triton.testing.do_bench(
        operation,
        warmup=args.warmup,
        rep=args.rep,
        return_mode="median",
    )
    indexer_flops, attention_flops = useful_matmul_flops(args)
    useful_tflops = (indexer_flops + attention_flops) / (triton_ms * 1e9)
    return triton_ms, useful_tflops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--index-heads", type=int, default=4)
    parser.add_argument("--index-dim", type=int, default=64)
    parser.add_argument("--compression-rate", type=int, default=4)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--rope-dims", type=int, default=64)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--share-kv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=200, help="Warmup duration in ms")
    parser.add_argument("--rep", type=int, default=1000, help="Measurement duration in ms")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    inputs = make_inputs(args)
    optimized = lambda: compressed_sparse_attention(*inputs, backend="triton")

    with torch.inference_mode():
        optimized()
        torch.cuda.synchronize()
        triton_ms, useful_tflops = benchmark_triton_flops(optimized, args)

    indexer_flops, attention_flops = useful_matmul_flops(args)
    total_flops = indexer_flops + attention_flops
    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(
        f"shape: B={args.batch} H={args.heads} S={args.sequence_length} "
        f"D={args.head_dim} HI={args.index_heads} DI={args.index_dim} dtype={args.dtype}"
    )
    print(
        f"sparsity: compression={args.compression_rate} topk={args.topk} "
        f"window={args.window} share_kv={args.share_kv}"
    )
    print(
        f"useful FLOPs: indexer={indexer_flops / 1e9:.6f} GF, "
        f"selected QK+PV={attention_flops / 1e9:.6f} GF, "
        f"total={total_flops / 1e9:.6f} GF"
    )
    print(f"Triton end-to-end: {triton_ms:.4f} ms")
    print(f"Triton useful sparse throughput: {useful_tflops:.2f} TFLOP/s")


if __name__ == "__main__":
    main()
