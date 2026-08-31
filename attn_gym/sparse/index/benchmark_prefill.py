"""Benchmark the one-CTA-per-query CuTeDSL prefill indexer.

The workload is fixed to the DSA indexer contract used by this directory:
BF16, B=2, H=128, D=128, and Top-K=128, with equal query/key sequence
lengths.  The cuDNN reference is the exact H=128 construction supported by
its dense indexer API: two prepacked H=64 forward calls, an FP32 score add,
and one standalone cuDNN Top-K call.

``triton.testing.do_bench`` interprets ``warmup`` and ``rep`` as millisecond
budgets.  Useful TFLOP/s counts only valid QK multiply-adds; head reduction,
ReLU, masking, and Top-K are excluded.
"""

import argparse
import gc
import json
import math
from collections.abc import Callable

import prefill
import torch
import triton.testing
from cudnn import DSA
from cutlass import cute
from cutlass.cute.runtime import from_dlpack

_BATCH = 2
_HEADS = 128
_HEAD_DIM = 128
_TOPK = 128
_HALF_HEADS = 64
_DTYPE = torch.bfloat16
_DEFAULT_SEQUENCE = 1024
_DEFAULT_WARMUP = 200
_DEFAULT_REP = 1000


def _scores(q: torch.Tensor, k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Materialize the exact FP32 DSA/CSA indexer score matrix."""
    batch, sequence, heads, head_dim = q.shape
    dots = torch.bmm(
        q.reshape(batch, sequence * heads, head_dim),
        k.transpose(1, 2),
        out_dtype=torch.float32,
    ).reshape(batch, sequence, heads, sequence)
    return (
        (dots.relu_() * weights.float().unsqueeze(-1))
        .sum(dim=2)
        .mul_(1.0 / math.sqrt(heads * head_dim))
    )


def _time_ms(fn: Callable[[], torch.Tensor], warmup: int, rep: int) -> float:
    return float(
        triton.testing.do_bench(
            fn,
            warmup=warmup,
            rep=rep,
            return_mode="median",
        )
    )


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _incremental_peak_bytes(fn: Callable[[], torch.Tensor]) -> int:
    """Measure temporary allocated memory above the already-owned buffers."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    return max(0, peak - baseline)


def _useful_flops(sequence: int, causal: bool) -> int:
    candidate_pairs = sequence * (sequence + 1) // 2 if causal else sequence * sequence
    return 2 * _BATCH * _HEADS * _HEAD_DIM * candidate_pairs


def _validate_indices(
    name: str,
    actual: torch.Tensor,
    reference_indices: torch.Tensor,
    reference_values: torch.Tensor,
    reference_scores: torch.Tensor,
    causal: bool,
) -> dict[str, int | float | str]:
    """Require exact sets, except for genuinely tied numerical boundaries."""
    if actual.shape != reference_indices.shape:
        raise AssertionError(
            f"{name}: expected shape {tuple(reference_indices.shape)}, got {tuple(actual.shape)}"
        )
    actual = actual.to(torch.int64)
    sequence = actual.shape[1]
    row = torch.arange(sequence, device=actual.device).view(1, sequence, 1)
    valid = actual >= 0
    if torch.any(actual < -1) or torch.any(actual >= sequence):
        raise AssertionError(f"{name}: indices must be -1 or in [0, {sequence})")
    if causal and torch.any(valid & (actual > row)):
        raise AssertionError(f"{name}: selected a key above the causal diagonal")

    expected_valid = (
        torch.minimum(row + 1, torch.full_like(row, _TOPK)).expand(_BATCH, -1, -1)
        if causal
        else torch.full((_BATCH, sequence, 1), _TOPK, device=actual.device)
    )
    if not torch.equal(valid.sum(-1, keepdim=True), expected_valid):
        raise AssertionError(f"{name}: incorrect number of valid entries in one or more rows")

    # Duplicated -1 padding is intentional; valid selected indices must be unique.
    sorted_actual = actual.sort(-1).values
    duplicate = (sorted_actual[..., 1:] == sorted_actual[..., :-1]) & (sorted_actual[..., 1:] >= 0)
    if torch.any(duplicate):
        raise AssertionError(f"{name}: duplicate valid index in one or more rows")

    sorted_reference = reference_indices.to(torch.int64).sort(-1).values
    exact_rows = (sorted_actual == sorted_reference).all(-1)
    mismatch_rows = int((~exact_rows).sum().item())
    if mismatch_rows == 0:
        return {
            "status": "exact",
            "mismatched_rows": 0,
            "max_boundary_gap": 0.0,
        }

    # Rows with fewer than K causal candidates have no Top-K boundary: every
    # valid key must be present, so a set mismatch there is always an error.
    full_rows = expected_valid.squeeze(-1) == _TOPK
    if torch.any((~exact_rows) & (~full_rows)):
        raise AssertionError(f"{name}: mismatch in a causal row with fewer than K valid keys")

    safe_indices = actual.clamp_min(0)
    selected_values = reference_scores.gather(-1, safe_indices)
    selected_values = selected_values.masked_fill(~valid, float("inf"))
    min_selected = selected_values.min(-1).values
    kth_reference = reference_values[..., -1]
    boundary_gap = (kth_reference - min_selected).clamp_min(0)
    mismatch_gap = boundary_gap.masked_select(~exact_rows)
    max_gap = float(mismatch_gap.max().item())
    scale = torch.maximum(
        kth_reference.abs(),
        torch.ones_like(kth_reference),
    )
    tolerance = 5.0e-4 * scale
    if torch.any((boundary_gap > tolerance) & (~exact_rows)):
        raise AssertionError(
            f"{name}: selected-set mismatch is not confined to the numerical "
            f"Top-K boundary (max score gap {max_gap:.6g})"
        )
    return {
        "status": "boundary_tolerance",
        "mismatched_rows": mismatch_rows,
        "max_boundary_gap": max_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", type=int, default=_DEFAULT_SEQUENCE)
    parser.add_argument("--causal", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup", type=int, default=_DEFAULT_WARMUP)
    parser.add_argument("--rep", type=int, default=_DEFAULT_REP)
    parser.add_argument("--device", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()

    if args.sequence < _TOPK or args.sequence > prefill._MAX_SEQUENCE:
        raise ValueError(
            f"sequence must be in [{_TOPK}, {prefill._MAX_SEQUENCE}], got {args.sequence}"
        )
    if args.warmup <= 0 or args.rep <= 0:
        raise ValueError("warmup and rep must both be positive")

    torch.cuda.set_device(args.device)
    torch.manual_seed(2026)
    device = torch.device("cuda", args.device)
    sequence = args.sequence
    score_scale = 1.0 / math.sqrt(_HEADS * _HEAD_DIM)

    q = torch.randn(
        _BATCH,
        sequence,
        _HEADS,
        _HEAD_DIM,
        device=device,
        dtype=_DTYPE,
    )
    k = torch.randn(_BATCH, sequence, _HEAD_DIM, device=device, dtype=_DTYPE)
    weights = torch.randn(
        _BATCH,
        sequence,
        _HEADS,
        device=device,
        dtype=_DTYPE,
    )
    common_input_bytes = _tensor_bytes(q) + _tensor_bytes(k) + _tensor_bytes(weights)

    # Compile this shape/config exactly once. The result owns no hidden output
    # or scratch allocation; each invocation writes the caller-owned INT32 buffer.
    cute_output = torch.empty(
        (_BATCH, sequence, _TOPK),
        device=device,
        dtype=torch.int32,
    )
    q_cute = from_dlpack(q, assumed_align=prefill._ALIGNMENT)
    k_cute = from_dlpack(k, assumed_align=prefill._ALIGNMENT)
    weights_cute = from_dlpack(weights, assumed_align=prefill._ALIGNMENT)
    output_cute = from_dlpack(cute_output, assumed_align=prefill._ALIGNMENT)
    compiled_cute = cute.compile(
        prefill._launch,
        q_cute,
        k_cute,
        weights_cute,
        output_cute,
        score_scale,
        _TOPK,
        args.causal,
    )

    def cute_prefill() -> torch.Tensor:
        compiled_cute(q_cute, k_cute, weights_cute, output_cute, score_scale)
        return cute_output

    # The materialized PyTorch baseline computes all heads in one BF16 x BF16
    # tensor-core BMM, performs the head epilogue in FP32, then selects Top-K.
    causal_mask = None
    causal_row = None
    if args.causal:
        causal_mask = torch.ones(
            (sequence, sequence),
            dtype=torch.bool,
            device=device,
        ).triu_(diagonal=1)
        causal_row = torch.arange(sequence, device=device).view(1, sequence, 1)

    def torch_prefill() -> torch.Tensor:
        scores = _scores(q, k, weights)
        if causal_mask is not None:
            scores.masked_fill_(causal_mask, float("-inf"))
        indices = scores.topk(_TOPK, dim=-1, sorted=False).indices
        if causal_row is not None:
            indices.masked_fill_(indices > causal_row, -1)
        return indices.to(torch.int32)

    # cuDNN's dense BF16 indexer accepts 32 or 64 query heads. Prepack two
    # contiguous H=64 views once, run each with the full H=128 variance scale,
    # add their FP32 scores, and invoke cuDNN's standalone selector once.
    q_lo = q[:, :, :_HALF_HEADS, :].contiguous()
    q_hi = q[:, :, _HALF_HEADS:, :].contiguous()
    weights_lo = weights[:, :, :_HALF_HEADS].contiguous()
    weights_hi = weights[:, :, _HALF_HEADS:].contiguous()
    k_bshd = k.unsqueeze(2)
    cudnn_score_lo = torch.empty(
        (_BATCH, sequence, sequence),
        dtype=torch.float32,
        device=device,
    )
    cudnn_score_hi = torch.empty_like(cudnn_score_lo)
    q_causal_offsets = None
    if not args.causal:
        q_causal_offsets = torch.full(
            (_BATCH,),
            sequence - 1,
            dtype=torch.int32,
            device=device,
        )
    if args.causal:
        seq_lens = torch.arange(1, sequence + 1, dtype=torch.int32, device=device).repeat(_BATCH)
    else:
        seq_lens = torch.full(
            (_BATCH * sequence,),
            sequence,
            dtype=torch.int32,
            device=device,
        )

    def cudnn_prefill() -> torch.Tensor:
        score_lo = DSA.indexer_forward_wrapper(
            q_lo,
            k_bshd,
            weights_lo,
            ratio=1,
            qhead_per_kv_head=_HALF_HEADS,
            sm_scale=score_scale,
            out=cudnn_score_lo,
            q_causal_offsets=q_causal_offsets,
        )["scores"]
        score_hi = DSA.indexer_forward_wrapper(
            q_hi,
            k_bshd,
            weights_hi,
            ratio=1,
            qhead_per_kv_head=_HALF_HEADS,
            sm_scale=score_scale,
            out=cudnn_score_hi,
            q_causal_offsets=q_causal_offsets,
        )["scores"]
        score_lo.add_(score_hi)
        indices = DSA.indexer_top_k_wrapper(
            score_lo.view(_BATCH * sequence, sequence),
            seq_lens,
            top_k=_TOPK,
            next_n=1,
            return_val=False,
        )["indices"]
        return indices.view(_BATCH, sequence, _TOPK)

    # Compile/warm all paths once, then validate sets against one retained
    # PyTorch score matrix. Numerical-boundary tolerance is reported explicitly.
    actual_cute = cute_prefill().clone()
    actual_cudnn = cudnn_prefill().clone()
    reference_scores = _scores(q, k, weights)
    if causal_mask is not None:
        reference_scores.masked_fill_(causal_mask, float("-inf"))
    reference_values, reference_indices = reference_scores.topk(
        _TOPK,
        dim=-1,
        sorted=True,
    )
    if causal_row is not None:
        reference_indices.masked_fill_(reference_indices > causal_row, -1)
    validation = {
        "cutedsl_prefill": _validate_indices(
            "CuTeDSL prefill",
            actual_cute,
            reference_indices,
            reference_values,
            reference_scores,
            args.causal,
        ),
        "cudnn_dense_h64x2_topk": _validate_indices(
            "cuDNN dense H64x2 + TopK",
            actual_cudnn,
            reference_indices,
            reference_values,
            reference_scores,
            args.causal,
        ),
    }
    del actual_cute, actual_cudnn, reference_indices, reference_values, reference_scores
    gc.collect()
    torch.cuda.empty_cache()

    persistent_bytes = {
        "cutedsl_prefill": _tensor_bytes(cute_output),
        "torch_matmul_topk": (
            (_tensor_bytes(causal_mask) + _tensor_bytes(causal_row))
            if causal_mask is not None and causal_row is not None
            else 0
        ),
        "cudnn_dense_h64x2_topk": (
            _tensor_bytes(q_lo)
            + _tensor_bytes(q_hi)
            + _tensor_bytes(weights_lo)
            + _tensor_bytes(weights_hi)
            + _tensor_bytes(cudnn_score_lo)
            + _tensor_bytes(cudnn_score_hi)
            + (_tensor_bytes(q_causal_offsets) if q_causal_offsets is not None else 0)
            + _tensor_bytes(seq_lens)
        ),
    }
    functions = {
        "cutedsl_prefill": cute_prefill,
        "torch_matmul_topk": torch_prefill,
        "cudnn_dense_h64x2_topk": cudnn_prefill,
    }

    peaks = {}
    timings = {}
    for name, fn in functions.items():
        extra_peak = _incremental_peak_bytes(fn)
        peaks[name] = common_input_bytes + persistent_bytes[name] + extra_peak
        timings[name] = _time_ms(fn, args.warmup, args.rep)

    useful_flops = _useful_flops(sequence, args.causal)
    torch_ms = timings["torch_matmul_topk"]
    results = {}
    for name in functions:
        milliseconds = timings[name]
        results[name] = {
            "latency_ms": milliseconds,
            "useful_tflops": useful_flops / (milliseconds * 1.0e9),
            "speedup_vs_torch": torch_ms / milliseconds,
            "peak_memory_bytes": peaks[name],
            "peak_memory_gib": peaks[name] / (1 << 30),
        }

    report = {
        "config": {
            "batch": _BATCH,
            "sequence": sequence,
            "query_sequence": sequence,
            "key_sequence": sequence,
            "heads": _HEADS,
            "head_dim": _HEAD_DIM,
            "topk": _TOPK,
            "dtype": str(_DTYPE).removeprefix("torch."),
            "causal": args.causal,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(device),
            "warmup_ms": args.warmup,
            "rep_ms": args.rep,
        },
        "useful_flops": useful_flops,
        "validation": validation,
        "memory_definition": (
            "logical input bytes + path-owned persistent buffers + measured peak temporary "
            "torch allocations; device shared/TMEM is excluded"
        ),
        "results": results,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
