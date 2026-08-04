"""Benchmark selected_attention CuTe (SM100) backend — forward, backward, and combined TFLOP/s.

All measurements use torch.compile(fullgraph=True).
"""

from __future__ import annotations

import argparse

import torch
import triton

from attn_gym.sparse.selected_attention import selected_attention


def useful_flops(batch: int, heads: int, seq_len: int, head_dim: int,
                 window: int, topk: int) -> int:
    """Attention FLOPs (QK + PV, counting multiply-add as 2 ops)."""
    local_pairs = sum(min(window, q + 1) for q in range(seq_len))
    compressed_pairs = seq_len * topk
    return 4 * batch * heads * head_dim * (local_pairs + compressed_pairs)


def make_inputs(args, requires_grad=False):
    device = torch.device("cuda")
    dtype = torch.bfloat16
    gen = torch.Generator(device=device).manual_seed(args.seed)

    Q = torch.randn(args.batch, args.heads, args.sequence_length, args.head_dim,
                    device=device, dtype=dtype, generator=gen, requires_grad=requires_grad)
    KV = torch.randn(args.batch, 1, args.sequence_length, args.head_dim,
                     device=device, dtype=dtype, generator=gen, requires_grad=requires_grad)
    index_kv = torch.randn(args.batch, 1, args.index_seq_len, args.head_dim,
                           device=device, dtype=dtype, generator=gen, requires_grad=requires_grad)
    scores = torch.randn(args.batch, args.sequence_length, args.index_seq_len,
                         device=device, generator=gen)
    _, indices = torch.topk(scores, k=args.topk, dim=-1)
    sink = torch.zeros(args.heads, device=device, dtype=dtype)
    return Q, KV, index_kv, indices, sink


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--heads", type=int, default=128)
    p.add_argument("--sequence-length", type=int, default=4096)
    p.add_argument("--head-dim", type=int, default=512)
    p.add_argument("--index-seq-len", type=int, default=1024)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--window", type=int, default=512)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--rep", type=int, default=500)
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"shape: B={args.batch} H={args.heads} S={args.sequence_length} D={args.head_dim}")
    print(f"sparsity: topk={args.topk} window={args.window} index_seq_len={args.index_seq_len}")
    print()

    fwd_flops = useful_flops(args.batch, args.heads, args.sequence_length,
                             args.head_dim, args.window, args.topk)
    bwd_flops = 2.5 * fwd_flops
    fwd_bwd_flops = fwd_flops + bwd_flops
    print(f"useful attention FLOPs (fwd): {fwd_flops / 1e12:.4f} TF")
    print(f"useful attention FLOPs (bwd): {bwd_flops / 1e12:.4f} TF")
    print()

    # ---- Compiled forward (inference) ----
    Q_inf, KV_inf, ikv_inf, indices_inf, sink_inf = make_inputs(args, requires_grad=False)

    @torch.compile(fullgraph=True)
    def fwd_compiled(Q, KV, ikv, indices, sink):
        return selected_attention(Q, KV, ikv, indices, sink, None,
                                  args.window, True, backend="cute")

    with torch.inference_mode():
        fwd_compiled(Q_inf, KV_inf, ikv_inf, indices_inf, sink_inf)
        torch.cuda.synchronize()

        fwd_ms = triton.testing.do_bench(
            lambda: fwd_compiled(Q_inf, KV_inf, ikv_inf, indices_inf, sink_inf),
            warmup=args.warmup, rep=args.rep, return_mode="median",
        )

    # ---- Compiled backward only ----
    Q, KV, index_kv, indices, sink = make_inputs(args, requires_grad=True)
    grad_out = torch.randn(args.batch, args.heads, args.sequence_length, args.head_dim,
                           device=Q.device, dtype=Q.dtype)

    @torch.compile(fullgraph=True)
    def fwd_train(Q, KV, ikv, indices, sink):
        return selected_attention(Q, KV, ikv, indices, sink, None,
                                  args.window, True, backend="cute")

    # Warmup the compiled training forward so backward graph is ready
    out = fwd_train(Q, KV, index_kv, indices, sink)
    out.backward(grad_out)
    Q.grad = KV.grad = index_kv.grad = None
    torch.cuda.synchronize()

    # Benchmark backward only: run forward, then time backward
    def run_bwd():
        out = fwd_train(Q, KV, index_kv, indices, sink)
        out.backward(grad_out)
        Q.grad = KV.grad = index_kv.grad = None

    # Time forward alone in training mode (with grad)
    fwd_train_ms = triton.testing.do_bench(
        lambda: fwd_train(Q, KV, index_kv, indices, sink),
        warmup=args.warmup, rep=args.rep, return_mode="median",
    )

    # Time forward + backward together
    fwd_bwd_ms = triton.testing.do_bench(
        run_bwd, warmup=args.warmup, rep=args.rep, return_mode="median",
    )

    bwd_ms = fwd_bwd_ms - fwd_train_ms

    # ---- Results ----
    print("=" * 72)
    print(f"{'Measurement':<35} {'Time (ms)':>12} {'TFLOP/s':>12}")
    print("-" * 72)
    print(f"{'Forward (inference, compiled)':<35} {fwd_ms:>12.4f} {fwd_flops / (fwd_ms * 1e9):>12.2f}")
    print(f"{'Forward (training, compiled)':<35} {fwd_train_ms:>12.4f} {fwd_flops / (fwd_train_ms * 1e9):>12.2f}")
    print(f"{'Backward (compiled)':<35} {bwd_ms:>12.4f} {bwd_flops / (bwd_ms * 1e9):>12.2f}")
    print(f"{'Forward + Backward (compiled)':<35} {fwd_bwd_ms:>12.4f} {fwd_bwd_flops / (fwd_bwd_ms * 1e9):>12.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
