"""Benchmark forward and backward throughput of the SM100 CuTe CSA backend.

The backward implementation checkpoints the attention intermediates, so a timed
backward call includes a forward recomputation.  This script reports both the
algorithmic gradient FLOPs and the effective useful FLOPs including that recompute.
"""

from __future__ import annotations

import argparse
import torch
import triton

from attn_gym.sparse.compressed_sparse_attention.api import compressed_sparse_attention

if __package__:
    from .benchmark_compressed_sparse_attention_triton import make_inputs
else:
    from benchmark_compressed_sparse_attention_triton import make_inputs


# These are the tensor arguments for which the CuTe backward returns gradients.
DIFFERENTIABLE_INPUTS = (0, 2, 3, 4, 5, 6, 7, 8, 16, 18, 19)


def useful_flops(args: argparse.Namespace) -> tuple[int, int, int, int]:
    """Return indexer, attention, gradient-only, and timed-backward useful FLOPs.

    Matmul FLOPs count one multiply-add as two operations.  Elementwise operations,
    normalization, RoPE, top-k, and softmax are intentionally omitted, matching the
    usual useful sparse-attention throughput convention.
    """
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
    indexer_forward = 0
    if 0 < effective_topk < selectable_blocks:
        indexer_forward = (
            2
            * args.batch
            * args.index_heads
            * completed_pairs
            * args.index_dim
        )
    attention_forward = (
        4
        * args.batch
        * args.heads
        * args.head_dim
        * (local_pairs + compressed_pairs)
    )

    # Each forward QK or PV matmul has two corresponding backward matmuls.
    gradient_only = 2 * attention_forward
    # The custom backward recomputes the complete checkpointed forward first.
    timed_backward = indexer_forward + attention_forward + gradient_only
    return indexer_forward, attention_forward, gradient_only, timed_backward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=512)
    parser.add_argument("--index-heads", type=int, default=4)
    parser.add_argument("--index-dim", type=int, default=64)
    parser.add_argument("--compression-rate", type=int, default=4)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--rope-dims", type=int, default=64)
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--share-kv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup", type=int, default=100, help="Warmup duration in ms")
    parser.add_argument("--rep", type=int, default=500, help="Measurement duration in ms")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("This benchmark targets SM100 exclusively.")

    raw_inputs = make_inputs(args)
    inputs = tuple(
        value.detach().requires_grad_(index in DIFFERENTIABLE_INPUTS)
        if isinstance(value, torch.Tensor)
        else value
        for index, value in enumerate(raw_inputs)
    )
    gradient_targets = tuple(inputs[index] for index in DIFFERENTIABLE_INPUTS)

    def forward() -> torch.Tensor:
        return compressed_sparse_attention(*inputs, backend="cute")

    # Compile and validate that the result participates in autograd before timing.
    output = forward()
    if output.grad_fn is None:
        raise RuntimeError("The CuTe output has no autograd graph.")
    grad_output = torch.randn_like(output)

    def backward() -> tuple[torch.Tensor | None, ...]:
        return torch.autograd.grad(
            output,
            gradient_targets,
            grad_outputs=grad_output,
            retain_graph=True,
            allow_unused=True,
        )

    backward()
    torch.cuda.synchronize()

    forward_ms = triton.testing.do_bench(
        forward, warmup=args.warmup, rep=args.rep, return_mode="median"
    )
    backward_ms = triton.testing.do_bench(
        backward, warmup=args.warmup, rep=args.rep, return_mode="median"
    )

    indexer_flops, attention_flops, gradient_flops, timed_backward_flops = useful_flops(args)
    forward_flops = indexer_flops + attention_flops

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
        f"forward useful FLOPs: {forward_flops / 1e9:.3f} GF "
        f"(indexer={indexer_flops / 1e9:.3f}, attention={attention_flops / 1e9:.3f})"
    )
    print(f"forward: {forward_ms:.4f} ms, {forward_flops / (forward_ms * 1e9):.2f} TFLOP/s")
    print(
        f"backward gradient FLOPs: {gradient_flops / 1e9:.3f} GF; "
        f"timed useful FLOPs with recompute: {timed_backward_flops / 1e9:.3f} GF"
    )
    print(
        f"backward: {backward_ms:.4f} ms, "
        f"{gradient_flops / (backward_ms * 1e9):.2f} gradient TFLOP/s, "
        f"{timed_backward_flops / (backward_ms * 1e9):.2f} effective TFLOP/s "
        "(including checkpoint recompute)"
    )


if __name__ == "__main__":
    main()
