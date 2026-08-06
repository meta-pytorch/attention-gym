"""Benchmark selected_attention forward and backward across backends and shapes.

Usage:
    python benchmarks/sparse/selected_attention_perf.py
    python benchmarks/sparse/selected_attention_perf.py --backend triton eager --calculate-bwd
    python benchmarks/sparse/selected_attention_perf.py --batch 4 --sequence-length 4096
"""

import argparse

import torch
import triton

from attn_gym.sparse.selected_attention import selected_attention

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def useful_flops(args: argparse.Namespace) -> int:
    """Compute forward-pass matmul FLOPs for selected attention.

    Each query attends to num_topk sparse entries and up to sliding_window_size local entries.
    Two matmuls per query position (QK^T and PV), each is 2*N*D FLOPs.
    Some assumptions:
        No indices are -1 (standard for most implementations)
        Total number of documents << sequence length
        Dim >> 1, so softmax flops are negligible
    """
    s = args.sequence_length
    d = args.head_dim
    window = min(args.window, s)
    effective_kv_len = args.topk + window
    flops_per_head = s * effective_kv_len * d * 2 * 2
    return args.batch * args.heads * flops_per_head


def make_inputs(args: argparse.Namespace, requires_grad: bool = False):
    device = torch.device("cuda")
    dtype = DTYPES[args.dtype]
    generator = torch.Generator(device=device).manual_seed(args.seed)
    kv_heads = 1 if args.share_kv else args.heads

    def randn(*shape):
        return torch.randn(
            *shape, device=device, dtype=dtype, generator=generator, requires_grad=requires_grad
        )

    query = randn(args.batch, args.heads, args.sequence_length, args.head_dim)
    local_kv = randn(args.batch, kv_heads, args.sequence_length, args.head_dim)
    sparse_kv = randn(args.batch, kv_heads, args.sparse_seq_len, args.head_dim)
    attention_sink = randn(args.heads)

    scores = torch.randn(
        args.batch, args.sequence_length, args.sparse_seq_len, device=device, generator=generator
    )
    _, kv_indices = torch.topk(scores, k=min(args.topk, args.sparse_seq_len), dim=-1)

    return query, local_kv, sparse_kv, kv_indices, attention_sink


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--sparse-seq-len", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--window", type=int, default=512)
    parser.add_argument("--dtype", choices=DTYPES, default="bfloat16")
    parser.add_argument("--share-kv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backend", nargs="+", default=["triton"], choices=["eager", "triton"])
    parser.add_argument("--calculate-bwd", action="store_true")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup duration in ms")
    parser.add_argument("--rep", type=int, default=500, help="Measurement duration in ms")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")

    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(
        f"shape: B={args.batch} H={args.heads} S={args.sequence_length} "
        f"D={args.head_dim} sparse_S={args.sparse_seq_len}"
    )
    print(
        f"sparsity: topk={args.topk} window={args.window} "
        f"share_kv={args.share_kv} dtype={args.dtype}"
    )

    fwd_flops = useful_flops(args)

    for backend in args.backend:
        requires_grad = args.calculate_bwd
        query, local_kv, sparse_kv, kv_indices, attention_sink = make_inputs(
            args, requires_grad=requires_grad
        )

        def fwd():
            return selected_attention(
                query,
                local_kv,
                sparse_kv,
                kv_indices,
                attention_sink,
                None,
                args.window,
                backend=backend,
            )

        out = fwd()
        fwd_ms = triton.testing.do_bench(
            fwd, warmup=args.warmup, rep=args.rep, return_mode="median"
        )
        fwd_tflops = fwd_flops / (fwd_ms * 1e9)

        print(f"[{backend}] forward: {fwd_ms:.3f} ms  ({fwd_tflops:.2f} TFLOP/s)")

        if args.calculate_bwd:
            grad_output = torch.randn_like(out)

            def bwd():
                return torch.autograd.grad(
                    out,
                    (query, local_kv, sparse_kv, attention_sink),
                    grad_outputs=grad_output,
                    retain_graph=True,
                )

            bwd()  # warmup autograd graph
            bwd_ms = triton.testing.do_bench(
                bwd, warmup=args.warmup, rep=args.rep, return_mode="median"
            )
            # We approximate backwards useful tflops as 2x forward
            # We need dP/dO, dV/dO, dQ/dS, and dK/dS, which is 4 matmuls
            # compared to the 2 in the forward pass

            bwd_tflops = fwd_flops * 2 / (bwd_ms * 1e9)
            print(f"[{backend}] backward: {bwd_ms:.3f} ms  ({bwd_tflops:.2f} TFLOP/s)")

    if len(args.backend) == 2:
        # Run both and report speedup
        timings = {}
        for backend in args.backend:
            query, local_kv, sparse_kv, kv_indices, attention_sink = make_inputs(args)

            def fwd(b=backend):
                return selected_attention(
                    query,
                    local_kv,
                    sparse_kv,
                    kv_indices,
                    attention_sink,
                    None,
                    args.window,
                    backend=b,
                )

            fwd()
            timings[backend] = triton.testing.do_bench(
                fwd, warmup=args.warmup, rep=args.rep, return_mode="median"
            )
        baseline = max(timings.values())
        fastest = min(timings.values())
        print(f"\nspeedup: {baseline / fastest:.2f}x")


if __name__ == "__main__":
    main()
