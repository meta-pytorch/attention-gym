"""Benchmark the full SM100 CuTe CSA backend against the eager reference."""

from __future__ import annotations

import argparse
import torch
import triton

from attn_gym.sparse.compressed_sparse_attention.api import compressed_sparse_attention

if __package__:
    from .benchmark_compressed_sparse_attention_triton import make_inputs
else:
    from benchmark_compressed_sparse_attention_triton import make_inputs


ERROR_ATOL = 3e-2
ERROR_RTOL = 1e-2


def useful_flops(args: argparse.Namespace) -> tuple[int, int, int, int]:
    """Return local pairs, compressed pairs, indexer FLOPs, and selected-attention FLOPs."""
    selectable_blocks = args.sequence_length // args.compression_rate
    effective_topk = min(args.topk, selectable_blocks)
    local_pairs = sum(
        min(args.window, query_position + 1)
        for query_position in range(args.sequence_length)
    )
    compressed_pairs = sum(
        min(args.topk, (query_position + 1) // args.compression_rate)
        for query_position in range(args.sequence_length)
    )
    completed_pairs = sum(
        (query_position + 1) // args.compression_rate
        for query_position in range(args.sequence_length)
    )
    indexer_flops = 0
    if 0 < effective_topk < selectable_blocks:
        indexer_flops = (
            2
            * args.batch
            * args.index_heads
            * completed_pairs
            * args.index_dim
        )
    attention_flops = (
        4
        * args.batch
        * args.heads
        * args.head_dim
        * (local_pairs + compressed_pairs)
    )
    return local_pairs, compressed_pairs, indexer_flops, attention_flops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--index-heads", type=int, default=64)
    parser.add_argument("--index-dim", type=int, default=64)
    parser.add_argument("--compression-rate", type=int, default=4)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--rope-dims", type=int, default=64)
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--share-kv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=200, help="Warmup duration in ms")
    parser.add_argument("--rep", type=int, default=1000, help="Measurement duration in ms")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("This benchmark targets SM100 exclusively.")

    inputs = make_inputs(args)
    eager = lambda: compressed_sparse_attention(*inputs, backend="eager")
    optimized = lambda: compressed_sparse_attention(*inputs, backend="cute")

    with torch.inference_mode():
        expected = eager()
        actual = optimized()
        torch.cuda.synchronize()
        error = (actual.float() - expected.float()).abs()
        max_abs_error = error.max().item()
        mean_abs_error = error.mean().item()
        if not torch.allclose(actual, expected, atol=ERROR_ATOL, rtol=ERROR_RTOL):
            raise AssertionError(
                f"CuTe output is not allclose to the reference with "
                f"atol={ERROR_ATOL:g} and rtol={ERROR_RTOL:g}; "
                f"max absolute error is {max_abs_error:.6g}."
            )
        eager_ms = triton.testing.do_bench(
            eager, warmup=args.warmup, rep=args.rep, return_mode="median"
        )
        cute_ms = triton.testing.do_bench(
            optimized, warmup=args.warmup, rep=args.rep, return_mode="median"
        )

    local_pairs, compressed_pairs, indexer_flops, attention_flops = useful_flops(args)
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
        f"useful selected pairs per batch: local={local_pairs:,} "
        f"compressed={compressed_pairs:,} total={local_pairs + compressed_pairs:,}"
    )
    print(
        f"useful FLOPs: indexer={indexer_flops / 1e9:.6f} GF, "
        f"selected QK+PV={attention_flops / 1e9:.6f} GF, total={total_flops / 1e9:.6f} GF"
    )
    print(f"error: max_abs={max_abs_error:.7g} mean_abs={mean_abs_error:.7g}")
    print(f"eager end-to-end: {eager_ms:.4f} ms")
    print(f"CuTe end-to-end:  {cute_ms:.4f} ms")
    print(f"speedup: {eager_ms / cute_ms:.2f}x")
    print(f"CuTe useful sparse throughput: {total_flops / (cute_ms * 1e9):.2f} TFLOP/s")


if __name__ == "__main__":
    main()
