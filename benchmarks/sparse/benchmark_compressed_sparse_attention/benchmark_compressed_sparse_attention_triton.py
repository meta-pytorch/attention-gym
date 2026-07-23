"""Benchmark eager and Triton compressed sparse attention backends."""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
import triton

from attn_gym.sparse.compressed_sparse_attention.api import compressed_sparse_attention
from attn_gym.sparse.compressed_sparse_attention.triton import (
    _launch_selected_attention,
    _prepare_attention_inputs,
)


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

ERROR_ATOL = 1e-2
ERROR_RTOL = 1e-2
DIFFERENTIABLE_INPUTS = (0, 2, 3, 4, 5, 6, 7, 8, 16, 18, 19)


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor | int | bool, ...]:
    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    generator = torch.Generator(device=device).manual_seed(args.seed)

    def randn(*shape: int, scale: float = 0.2) -> torch.Tensor:
        return torch.randn(*shape, device=device, dtype=dtype, generator=generator) * scale

    def query(*shape: int) -> torch.Tensor:
        return F.normalize(randn(*shape), dim=-1)

    kv_heads = 1 if args.share_kv else args.heads
    index_kv_heads = 1 if args.share_kv else args.index_heads
    return (
        query(args.batch, args.heads, args.sequence_length, args.head_dim),
        query(args.batch, args.index_heads, args.sequence_length, args.index_dim),
        randn(args.batch, kv_heads, args.sequence_length, args.head_dim),
        randn(args.batch, kv_heads, args.sequence_length, args.head_dim),
        randn(args.batch, kv_heads, args.sequence_length, args.head_dim),
        randn(args.batch, kv_heads, args.sequence_length, args.head_dim),
        randn(args.batch, kv_heads, args.sequence_length, args.head_dim),
        randn(args.compression_rate, args.head_dim),
        randn(args.compression_rate, args.head_dim),
        randn(args.batch, args.sequence_length, args.index_heads),
        randn(args.batch, index_kv_heads, args.sequence_length, args.index_dim),
        randn(args.batch, index_kv_heads, args.sequence_length, args.index_dim),
        randn(args.batch, index_kv_heads, args.sequence_length, args.index_dim),
        randn(args.batch, index_kv_heads, args.sequence_length, args.index_dim),
        randn(args.compression_rate, args.index_dim),
        randn(args.compression_rate, args.index_dim),
        1.0 + randn(args.head_dim, scale=0.05),
        1.0 + randn(args.index_dim, scale=0.05),
        1.0 + randn(args.head_dim, scale=0.05),
        randn(args.heads),
        args.compression_rate,
        args.topk,
        args.window,
        args.rope_dims,
        args.share_kv,
    )


def useful_matmul_flops(args: argparse.Namespace) -> tuple[int, int]:
    """Return indexer and selected-attention QK/PV FLOPs."""
    sequence_length = args.sequence_length
    num_blocks = math.ceil(sequence_length / args.compression_rate)
    selected_blocks = min(args.topk, num_blocks)
    selected_pairs = sum(
        min(selected_blocks, (query_position + 1) // args.compression_rate)
        + min(args.window, query_position + 1)
        for query_position in range(sequence_length)
    )
    indexer = 2 * args.batch * args.index_heads * sequence_length * num_blocks * args.index_dim
    attention = 4 * args.batch * args.heads * args.head_dim * selected_pairs
    return indexer, attention


def differentiable_copy(inputs):
    return tuple(
        value.detach().clone().requires_grad_(index in DIFFERENTIABLE_INPUTS)
        if isinstance(value, torch.Tensor)
        else value
        for index, value in enumerate(inputs)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=16)
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
    parser.add_argument("--warmup", type=int, default=100, help="Warmup duration in milliseconds")
    parser.add_argument(
        "--rep", type=int, default=500, help="Measurement duration in milliseconds"
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")

    inputs = make_inputs(args)
    eager_inputs = differentiable_copy(inputs)
    triton_inputs = differentiable_copy(inputs)
    eager_targets = tuple(eager_inputs[index] for index in DIFFERENTIABLE_INPUTS)
    triton_targets = tuple(triton_inputs[index] for index in DIFFERENTIABLE_INPUTS)
    eager = lambda: compressed_sparse_attention(*eager_inputs, backend="eager")
    optimized = lambda: compressed_sparse_attention(*triton_inputs, backend="triton")

    expected = eager()
    actual = optimized()
    query, compressed_kv, local_kv, topk_blocks = _prepare_attention_inputs(*triton_inputs)
    with torch.inference_mode():
        selected_attention = lambda: _launch_selected_attention(
            query,
            compressed_kv,
            local_kv,
            topk_blocks,
            triton_inputs[19],
            args.compression_rate,
            args.window,
        )
    torch.cuda.synchronize()
    max_abs_error = (expected.float() - actual.float()).abs().max().item()
    if not torch.allclose(actual, expected, atol=ERROR_ATOL, rtol=ERROR_RTOL):
        raise AssertionError(
            f"Triton output is not allclose to the reference with "
            f"atol={ERROR_ATOL:g} and rtol={ERROR_RTOL:g}; "
            f"max absolute error is {max_abs_error:.6g}."
        )

    generator = torch.Generator(device=actual.device).manual_seed(args.seed + 1)
    grad_output = (
        torch.randn(
            actual.shape,
            device=actual.device,
            dtype=actual.dtype,
            generator=generator,
        )
        * 0.01
    )

    def eager_backward():
        return torch.autograd.grad(
            expected,
            eager_targets,
            grad_outputs=grad_output,
            retain_graph=True,
        )

    def triton_backward():
        return torch.autograd.grad(
            actual,
            triton_targets,
            grad_outputs=grad_output,
            retain_graph=True,
        )

    expected_gradients = eager_backward()
    actual_gradients = triton_backward()
    max_gradient_error = 0.0
    for input_index, actual_gradient, expected_gradient in zip(
        DIFFERENTIABLE_INPUTS,
        actual_gradients,
        expected_gradients,
    ):
        gradient_error = (actual_gradient.float() - expected_gradient.float()).abs().max().item()
        max_gradient_error = max(max_gradient_error, gradient_error)
        if not torch.allclose(
            actual_gradient,
            expected_gradient,
            atol=ERROR_ATOL,
            rtol=ERROR_RTOL,
        ):
            raise AssertionError(
                f"Triton gradient for input {input_index} is not allclose with "
                f"atol={ERROR_ATOL:g} and rtol={ERROR_RTOL:g}; "
                f"max absolute error is {gradient_error:.6g}."
            )

    eager_ms = triton.testing.do_bench(
        eager, warmup=args.warmup, rep=args.rep, return_mode="median"
    )
    triton_ms = triton.testing.do_bench(
        optimized, warmup=args.warmup, rep=args.rep, return_mode="median"
    )
    kernel_ms = triton.testing.do_bench(
        selected_attention, warmup=args.warmup, rep=args.rep, return_mode="median"
    )
    eager_backward_ms = triton.testing.do_bench(
        eager_backward, warmup=args.warmup, rep=args.rep, return_mode="median"
    )
    triton_backward_ms = triton.testing.do_bench(
        triton_backward, warmup=args.warmup, rep=args.rep, return_mode="median"
    )

    indexer_flops, attention_flops = useful_matmul_flops(args)
    attention_tflops = attention_flops / (kernel_ms * 1e9)
    end_to_end_tflops = (indexer_flops + attention_flops) / (triton_ms * 1e9)

    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(
        f"shape: B={args.batch} H={args.heads} S={args.sequence_length} "
        f"D={args.head_dim} HI={args.index_heads} DI={args.index_dim}"
    )
    print(
        f"sparsity: compression={args.compression_rate} topk={args.topk} "
        f"window={args.window} share_kv={args.share_kv} dtype={args.dtype}"
    )
    print(f"max absolute error: {max_abs_error:.6g}")
    print(f"maximum gradient absolute error: {max_gradient_error:.6g}")
    print(f"eager:  {eager_ms:.3f} ms")
    print(f"triton: {triton_ms:.3f} ms")
    print(f"speedup: {eager_ms / triton_ms:.2f}x")
    print(f"eager backward:  {eager_backward_ms:.3f} ms")
    print(f"triton backward: {triton_backward_ms:.3f} ms")
    print(f"backward speedup: {eager_backward_ms / triton_backward_ms:.2f}x")
    print(f"selected-attention Triton kernel: {kernel_ms:.3f} ms")
    print(f"selected-attention Triton kernel useful TFLOP/s: {attention_tflops:.3f}")
    print(f"end-to-end modeled matmul TFLOP/s: {end_to_end_tflops:.3f}")


if __name__ == "__main__":
    main()
