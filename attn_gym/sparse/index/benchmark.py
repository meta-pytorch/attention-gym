"""GB200 benchmark for the fused DSA/CSA indexer kernel.

The two materialized baselines deliberately share the same tensor-core score
construction and differ only in the final selector:

* PyTorch matmul + ``torch.topk``
* PyTorch matmul + cuDNN Frontend ``IndexerTopK``

The third baseline is cuDNN's SM100 combined ``IndexerForwardTopK`` operation,
which owns tensor-core scoring, per-head ReLU, weighted head reduction, and
Top-K without a dense score tensor. The fused CuTeDSL path likewise materializes
neither the ``[B,T,H,S]`` logits nor the ``[B,T,S]`` score matrix.

Reported TFLOP/s is conventional useful matmul throughput:
``2 * B * T * H * S * D / time``. Top-K and epilogue operations are excluded.
"""

import argparse
import gc
import math
from collections.abc import Callable

import index as index_kernel
import torch
import triton.testing
from cudnn import DSA
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

_BENCH_WARMUP = 100
_BENCH_REP = 100
_CUDNN_RATIO = 4


def _scores(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    batch, queries, heads, head_dim = q.shape
    candidates = k.shape[1]
    dots = torch.bmm(
        q.reshape(batch, queries * heads, head_dim),
        k.transpose(1, 2),
        out_dtype=torch.float32,
    ).reshape(batch, queries, heads, candidates)
    return (
        (dots.relu_() * weights.float().unsqueeze(-1))
        .sum(2)
        .mul_(1.0 / math.sqrt(head_dim * heads))
    )


def _time_ms(fn: Callable[[], torch.Tensor], warmup: int, rep: int) -> float:
    """Return the median latency using Triton's CUDA-event benchmark harness.

    ``triton.testing.do_bench`` interprets ``warmup`` and ``rep`` as
    millisecond budgets and derives iteration counts from a latency estimate.
    """
    return float(
        triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="median",
        )
    )


def _matmul_flops(
    batch: int,
    queries: int,
    heads: int,
    head_dim: int,
    candidates: int,
) -> int:
    """Useful tensor-core work, counting each fused multiply-add as two FLOPs."""
    return 2 * batch * queries * heads * head_dim * candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--queries", type=int, default=1)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--candidates", type=int, default=65536)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=_BENCH_WARMUP)
    parser.add_argument("--rep", type=int, default=_BENCH_REP)
    args = parser.parse_args()

    if args.head_dim <= 0 or args.head_dim % index_kernel._HEAD_DIM_GRANULARITY:
        raise ValueError(
            f"head_dim must be positive and divisible by {index_kernel._HEAD_DIM_GRANULARITY}"
        )
    if args.heads <= 0 or args.heads % 2:
        raise ValueError("heads must be a positive multiple of 2")
    if not 0 < args.topk <= min(args.candidates, 2048):
        raise ValueError("benchmark requires 0 < topk <= min(S, 2048)")
    run_cudnn_combined = args.head_dim == 128 and args.heads in (32, 64)

    torch.manual_seed(2026)
    device = "cuda"
    dtype = torch.bfloat16
    shape = (args.batch, args.queries, args.heads, args.head_dim)
    q = torch.randn(*shape, device=device, dtype=dtype)
    k = torch.randn(
        args.batch,
        args.candidates,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    weights = torch.randn(
        args.batch,
        args.queries,
        args.heads,
        device=device,
        dtype=dtype,
    )
    seq_lens = torch.full(
        (args.batch * args.queries,),
        args.candidates,
        device=device,
        dtype=torch.int32,
    )

    if run_cudnn_combined:
        cudnn_ratio = max(_CUDNN_RATIO, args.queries)
        if cudnn_ratio * args.candidates - 1 > torch.iinfo(torch.int32).max:
            raise ValueError("cuDNN q_causal_offsets would exceed INT32_MAX")

        # The cuDNN combined kernel accepts BSHD or THD. Zero-copy THD views plus
        # caller-owned scratch/output buffers avoid allocation and offset-building
        # inside the timed call. Place local q[0] at the end of the uncompressed
        # timeline so every row can select from the same full S-candidate prefix.
        q_thd = q.view(args.batch * args.queries, args.heads, args.head_dim)
        k_thd = k.view(args.batch * args.candidates, 1, args.head_dim)
        weights_thd = weights.view(args.batch * args.queries, args.heads)
        cu_seqlens_q = torch.arange(
            0,
            (args.batch + 1) * args.queries,
            args.queries,
            device=device,
            dtype=torch.int32,
        )
        cu_seqlens_k = torch.arange(
            0,
            (args.batch + 1) * args.candidates,
            args.candidates,
            device=device,
            dtype=torch.int32,
        )
        q_causal_offsets = torch.full(
            (args.batch,),
            cudnn_ratio * args.candidates - 1,
            device=device,
            dtype=torch.int32,
        )
        cand_batch_offsets, cand_floats = DSA.compress_topk_cand_buffer_size_thd(
            cu_seqlens_q,
            cu_seqlens_k,
            ratio=cudnn_ratio,
            q_causal_offsets=q_causal_offsets,
        )
        cudnn_combined_cand = torch.empty(cand_floats, device=device, dtype=torch.float32)
        cudnn_combined_indices = torch.empty(
            (args.batch * args.queries, args.topk),
            device=device,
            dtype=torch.int32,
        )
        cudnn_combined_logits = torch.empty(
            (args.batch * args.queries, args.topk),
            device=device,
            dtype=torch.float32,
        )

    # Compile exactly once for this benchmark shape. Calling the @cute.jit
    # launcher directly would retrace/compile in every timed invocation and
    # measure hundreds of milliseconds of Python/JIT dispatch instead of the
    # GPU kernel. This is deliberately local, not a cache or compile-key layer.
    fused_output = torch.empty(
        (args.batch, args.queries, args.topk),
        device=device,
        dtype=torch.int32,
    )
    num_splits = math.ceil(args.candidates / index_kernel._CANDIDATES_PER_CTA)
    merge_levels = (num_splits - 1).bit_length()
    partial_shape = (args.batch, args.queries, num_splits, args.topk)
    partial_keys_a = torch.empty(partial_shape, device=device, dtype=torch.int64)
    partial_keys_b = torch.empty_like(partial_keys_a)
    q_cute = from_dlpack(q, assumed_align=index_kernel._ALIGNMENT)
    k_cute = from_dlpack(k, assumed_align=index_kernel._ALIGNMENT)
    weights_cute = from_dlpack(weights, assumed_align=index_kernel._ALIGNMENT)
    partial_keys_a_cute = from_dlpack(partial_keys_a, assumed_align=index_kernel._ALIGNMENT)
    partial_keys_b_cute = from_dlpack(partial_keys_b, assumed_align=index_kernel._ALIGNMENT)
    output_cute = from_dlpack(fused_output, assumed_align=index_kernel._ALIGNMENT)
    score_scale = 1.0 / math.sqrt(args.heads * args.head_dim)
    search_levels = (args.topk - 1).bit_length()
    compiled_fused = cute.compile(
        index_kernel._launch,
        q_cute,
        k_cute,
        weights_cute,
        partial_keys_a_cute,
        partial_keys_b_cute,
        output_cute,
        score_scale,
        args.topk,
        search_levels,
        num_splits,
        merge_levels,
    )

    def fused() -> torch.Tensor:
        compiled_fused(
            q_cute,
            k_cute,
            weights_cute,
            partial_keys_a_cute,
            partial_keys_b_cute,
            output_cute,
            score_scale,
        )
        return fused_output

    def torch_matmul_topk() -> torch.Tensor:
        return _scores(q, k, weights).topk(args.topk, dim=-1, sorted=False).indices

    def cudnn_matmul_topk() -> torch.Tensor:
        scores = _scores(q, k, weights)
        return DSA.indexer_top_k_wrapper(
            scores.reshape(args.batch * args.queries, args.candidates),
            seq_lens,
            top_k=args.topk,
            next_n=1,
            return_val=False,
        )["indices"]

    def cudnn_combined_indexer_topk() -> torch.Tensor:
        result = DSA.indexer_forward_top_k_wrapper(
            q_thd,
            k_thd,
            weights_thd,
            top_k=args.topk,
            ratio=cudnn_ratio,
            sm_scale=score_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=args.queries,
            max_seqlen_k=args.candidates,
            q_causal_offsets=q_causal_offsets,
            return_softmax=False,
            topk_indices_global=False,
            cand_buffer=cudnn_combined_cand,
            out_indices=cudnn_combined_indices,
            out_logits=cudnn_combined_logits,
            cand_batch_offsets=cand_batch_offsets,
            deterministic=False,
        )
        return result["indices"].view(args.batch, args.queries, args.topk)

    # Compile/warm each path, then check selected sets before timing.
    actual = fused()
    expected = torch_matmul_topk().to(torch.int32)
    torch.testing.assert_close(
        actual.sort(-1).values,
        expected.sort(-1).values,
        rtol=0,
        atol=0,
    )
    cudnn_expected = cudnn_matmul_topk().reshape_as(actual).to(torch.int32)
    torch.testing.assert_close(
        actual.sort(-1).values,
        cudnn_expected.sort(-1).values,
        rtol=0,
        atol=0,
    )
    if run_cudnn_combined:
        cudnn_combined_expected = cudnn_combined_indexer_topk().to(torch.int32)
        torch.testing.assert_close(
            actual.sort(-1).values,
            cudnn_combined_expected.sort(-1).values,
            rtol=0,
            atol=0,
        )
    torch.cuda.synchronize()

    cases = [
        ("CuTeDSL fused indexer", fused),
        ("PyTorch matmul + torch.topk", torch_matmul_topk),
        ("PyTorch matmul + cuDNN TopK", cudnn_matmul_topk),
    ]
    if run_cudnn_combined:
        cases.append(("cuDNN combined indexer + TopK", cudnn_combined_indexer_topk))

    timings = []
    for label, fn in cases:
        gc.collect()
        torch.cuda.empty_cache()
        timings.append((label, _time_ms(fn, args.warmup, args.rep)))

    print(
        f"B={args.batch} T={args.queries} H={args.heads} D={args.head_dim} "
        f"S={args.candidates} K={args.topk} dtype={dtype} "
        f"do_bench(warmup={args.warmup}, rep={args.rep})"
    )
    print("Useful QK TFLOP/s counts 2*B*T*H*S*D; epilogue and Top-K are excluded")
    matmul_flops = _matmul_flops(
        args.batch,
        args.queries,
        args.heads,
        args.head_dim,
        args.candidates,
    )
    baseline = timings[1][1]
    for label, milliseconds in timings:
        tflops = matmul_flops / (milliseconds * 1e9)
        print(
            f"{label:34s} {milliseconds:9.3f} ms  "
            f"{tflops:9.2f} TFLOP/s  {baseline / milliseconds:6.2f}x"
        )
    if not run_cudnn_combined:
        print(
            "cuDNN combined indexer + TopK     unsupported "
            "(official contract requires D=128 and H in {32, 64})"
        )


if __name__ == "__main__":
    main()
