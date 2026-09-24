"""Benchmark gather_attn forward and backward across backends and shapes.

Usage:
    python benchmarks/sparse/gather_attn_benchmark.py
    python benchmarks/sparse/gather_attn_benchmark.py --impl fused --backend triton
    python benchmarks/sparse/gather_attn_benchmark.py --impl reference --batch 4
"""

from functools import partial
from typing import Annotated

import torch
import triton
import typer

from attn_gym.sparse.gather_attn import Impl, gather_attn

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def main(
    batch: int = 8,
    heads: int = 32,
    sequence_length: int = 4096,
    head_dim: int = 128,
    sparse_seq_len: int = 1024,
    topk: int = 16,
    window: int = 512,
    dtype: Annotated[str, typer.Option(help="One of: " + ", ".join(DTYPES))] = "bfloat16",
    share_kv: bool = True,
    impl: Impl = Impl.FUSED,
    backend: Annotated[
        list[str] | None,
        typer.Option(help="Fused override: triton or cute. Repeat to compare; omit for auto."),
    ] = None,
    calculate_bwd: bool = True,
    warmup: Annotated[int, typer.Option(help="Warmup duration in ms")] = 100,
    rep: Annotated[int, typer.Option(help="Measurement duration in ms")] = 500,
    seed: int = 123,
) -> None:
    if dtype not in DTYPES:
        raise typer.BadParameter("Choose " + ", ".join(DTYPES), param_hint="--dtype")
    if backend and (
        impl is Impl.REFERENCE or any(name not in ("triton", "cute") for name in backend)
    ):
        raise typer.BadParameter("Use triton or cute with --impl fused", param_hint="--backend")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU.")

    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"shape: B={batch} H={heads} S={sequence_length} D={head_dim} sparse_S={sparse_seq_len}")
    print(f"sparsity: topk={topk} window={window} share_kv={share_kv} dtype={dtype}")

    def make_inputs(requires_grad: bool):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        kv_heads = 1 if share_kv else heads

        def randn(*shape):
            return torch.randn(
                *shape,
                device="cuda",
                dtype=DTYPES[dtype],
                generator=generator,
                requires_grad=requires_grad,
            )

        query = randn(batch, heads, sequence_length, head_dim)
        local_kv = randn(batch, kv_heads, sequence_length, head_dim)
        sparse_kv = randn(batch, kv_heads, sparse_seq_len, head_dim)
        scores = torch.randn(
            batch, sequence_length, sparse_seq_len, device="cuda", generator=generator
        )
        kv_indices = scores.topk(min(topk, sparse_seq_len), dim=-1).indices
        return query, local_kv, sparse_kv, kv_indices

    # Estimated QK/PV FLOPs, ignoring softmax and the shorter initial causal windows.
    # Assumes every selected index is valid and no document boundaries shorten the window.
    fwd_flops = (
        batch * heads * sequence_length * (topk + min(window, sequence_length)) * head_dim * 4
    )
    for name in backend or [None]:
        label = f"fused/{name or 'auto'}" if impl is Impl.FUSED else "reference"
        attention = partial(
            gather_attn,
            sliding_window_size=window,
            impl=impl,
            kernel_options={"backend": name} if name is not None else None,
        )
        # Pure forward compute without autograd graph construction.
        fwd = partial(attention, *make_inputs(requires_grad=False))
        fwd()
        fwd_ms = triton.testing.do_bench(fwd, warmup=warmup, rep=rep, return_mode="median")
        print(f"[{label}] forward: {fwd_ms:.3f} ms  ({fwd_flops / (fwd_ms * 1e9):.2f} TFLOP/s)")

        if calculate_bwd:
            inputs = make_inputs(requires_grad=True)
            out = attention(*inputs)
            grad_output = torch.randn_like(out)
            bwd = partial(
                torch.autograd.grad,
                out,
                inputs[:3],
                grad_output,
                retain_graph=True,
                allow_unused=True,
            )
            bwd()
            bwd_ms = triton.testing.do_bench(bwd, warmup=warmup, rep=rep, return_mode="median")
            # Backward's four matmuls give approximately twice the forward FLOPs.
            bwd_tflops = fwd_flops * 2 / (bwd_ms * 1e9)
            print(f"[{label}] backward: {bwd_ms:.3f} ms  ({bwd_tflops:.2f} TFLOP/s)")


if __name__ == "__main__":
    typer.run(main)
