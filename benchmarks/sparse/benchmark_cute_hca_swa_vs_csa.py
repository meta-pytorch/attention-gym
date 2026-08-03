"""Compare CuTe HCA and SWA wall-clock time with same-head-count CuTe CSA.

This is deliberately a wall-clock benchmark.  It does not calculate modeled
FLOPs or use throughput to adjust the comparison:

* Every operator keeps the sparse defaults from its corresponding benchmark
  profile.
* Only the attention head count is aligned between a candidate and CSA.
* HCA uses ``R=128, W=128``; SWA uses ``W=128``.
* CSA keeps ``R=4, K=64, W=512, HI=4, DI=64``.
* Forward and activation-checkpointed backward are timed independently.
* The reported ratio is exactly ``candidate_ms / csa_ms``.

The default suite uses the shared canonical throughput shape
``B=8, H=64, S=4096, D=512`` and requires both candidate phases to take at
most 80% of CSA's wall clock.
Coverage/scaling cases are available as diagnostics and are never performance
gates.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import triton

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


Variant = Literal["hca", "swa"]
Category = Literal["default", "scaling", "padded-head", "degenerate", "boundary"]

HEAD_DIM = 512
# Keep the CSA forward/backward benchmark's indexer defaults.
INDEX_HEADS = 4
INDEX_DIM = 64
CSA_COMPRESSION_RATE = 4
CSA_TOPK = 64
CSA_WINDOW = 512
DEFAULT_MAX_RATIO = 0.8
TIMED_PHASES = ("forward", "backward")
BACKWARD_INCLUDES_CHECKPOINT_RECOMPUTE = True
WALL_CLOCK_ONLY = True
PERFORMANCE_CASES = frozenset({"hca_canonical_h64", "swa_canonical_h64"})

HCA_DIFFERENTIABLE_INPUTS = tuple(range(8))
SWA_DIFFERENTIABLE_INPUTS = tuple(range(4))
CSA_DIFFERENTIABLE_INPUTS = (0, 2, 3, 4, 5, 6, 7, 8, 16, 18, 19)


@dataclass(frozen=True)
class ShapeCase:
    """One same-shape candidate/CSA wall-clock comparison."""

    name: str
    variant: Variant
    category: Category
    batch: int
    heads: int
    sequence: int
    window: int
    compression_rate: int = 128
    rope_dims: int = 64

    @property
    def padded_heads(self) -> int:
        return math.ceil(self.heads / 64) * 64

    @property
    def is_performance_gate(self) -> bool:
        return self.name in PERFORMANCE_CASES


# The long cases exercise useful steady-state performance.  Small cases cover
# padding and branch boundaries without turning the support matrix into a huge
# Cartesian product.
SHAPE_CASES = (
    ShapeCase("hca_canonical_h64", "hca", "default", 8, 64, 4096, 128),
    ShapeCase("hca_fast_h128", "hca", "scaling", 1, 128, 4096, 128),
    ShapeCase("hca_fast_h192", "hca", "scaling", 1, 192, 1024, 128),
    ShapeCase("hca_pad_h1", "hca", "padded-head", 1, 1, 129, 128),
    ShapeCase("hca_pad_h63", "hca", "padded-head", 1, 63, 129, 128),
    ShapeCase("hca_pad_h65", "hca", "padded-head", 1, 65, 129, 128),
    ShapeCase("hca_pad_h127", "hca", "padded-head", 1, 127, 129, 128),
    ShapeCase("hca_pad_h129", "hca", "padded-head", 1, 129, 129, 128),
    ShapeCase("hca_sink_only", "hca", "degenerate", 1, 64, 127, 0),
    ShapeCase("hca_local_only", "hca", "degenerate", 1, 64, 127, 127),
    ShapeCase("hca_compressed_only", "hca", "degenerate", 1, 64, 256, 0),
    ShapeCase("hca_exact_block", "hca", "boundary", 1, 64, 128, 128),
    ShapeCase("hca_partial_block", "hca", "boundary", 1, 64, 129, 129),
    ShapeCase("hca_long_odd", "hca", "boundary", 1, 64, 4097, 128),
    ShapeCase("swa_canonical_h64", "swa", "default", 8, 64, 4096, 128),
    ShapeCase("swa_fast_h128", "swa", "scaling", 1, 128, 4096, 128),
    ShapeCase("swa_fast_h192", "swa", "scaling", 1, 192, 1024, 128),
    ShapeCase("swa_pad_h1", "swa", "padded-head", 1, 1, 129, 128),
    ShapeCase("swa_pad_h63", "swa", "padded-head", 1, 63, 129, 128),
    ShapeCase("swa_pad_h65", "swa", "padded-head", 1, 65, 129, 128),
    ShapeCase("swa_pad_h127", "swa", "padded-head", 1, 127, 129, 128),
    ShapeCase("swa_pad_h129", "swa", "padded-head", 1, 129, 129, 128),
    ShapeCase("swa_sink_only", "swa", "degenerate", 1, 64, 127, 0),
    ShapeCase("swa_single_token", "swa", "boundary", 1, 64, 1, 1),
    ShapeCase("swa_window_one", "swa", "boundary", 1, 64, 129, 1),
    ShapeCase("swa_exact_tile", "swa", "boundary", 1, 64, 128, 128),
    ShapeCase("swa_partial_tile", "swa", "boundary", 1, 64, 129, 128),
    ShapeCase("swa_oversized_window", "swa", "boundary", 1, 64, 129, 1024),
    ShapeCase("swa_long_odd", "swa", "boundary", 1, 64, 4097, 128),
)


@dataclass(frozen=True)
class Timing:
    forward_ms: float
    backward_ms: float


@dataclass(frozen=True)
class Comparison:
    case: ShapeCase
    candidate: Timing
    csa: Timing

    @property
    def forward_ratio(self) -> float:
        return raw_ratio(self.candidate.forward_ms, self.csa.forward_ms)

    @property
    def backward_ratio(self) -> float:
        return raw_ratio(self.candidate.backward_ms, self.csa.backward_ms)


def raw_ratio(candidate_ms: float, csa_ms: float) -> float:
    """Return the unadjusted candidate/CSA wall-clock ratio."""
    return candidate_ms / csa_ms


def comparison_order(case_index: int) -> tuple[str, str]:
    """Alternate launch order so neither implementation always runs second."""
    if case_index % 2:
        return ("candidate", "csa")
    return ("csa", "candidate")


def csa_sparse_configuration(case: ShapeCase) -> tuple[int, int, int]:
    """Return CSA's own default ``(compression_rate, topk, window)``."""
    del case
    return CSA_COMPRESSION_RATE, CSA_TOPK, CSA_WINDOW


def candidate_sparse_configuration(case: ShapeCase) -> tuple[str, str, int]:
    """Return display values for the candidate's own sparse defaults."""
    if case.variant == "hca":
        return str(case.compression_rate), "all", case.window
    return "-", "-", case.window


def selected_cases(
    *,
    variant: str = "all",
    category: str = "default",
    names: tuple[str, ...] = (),
) -> tuple[ShapeCase, ...]:
    """Filter the matrix; explicit case names override category filtering."""
    unknown = set(names) - {case.name for case in SHAPE_CASES}
    if unknown:
        raise ValueError(f"Unknown benchmark case(s): {', '.join(sorted(unknown))}.")
    return tuple(
        case
        for case in SHAPE_CASES
        if (variant == "all" or case.variant == variant)
        and (names or category == "all" or case.category == category)
        and (not names or case.name in names)
    )


def format_case_table(cases: tuple[ShapeCase, ...]) -> str:
    """Format the support/performance matrix without requiring CUDA."""
    header = (
        "case                         kind category       B    H H_pad      S"
        " candidate(R,K,W)     csa(R,K,W) rope gate"
    )
    rows = [header, "-" * len(header)]
    for case in cases:
        candidate_rate, candidate_topk, candidate_window = candidate_sparse_configuration(case)
        csa_rate, csa_topk, csa_window = csa_sparse_configuration(case)
        rows.append(
            f"{case.name:<28} {case.variant:<4} {case.category:<13} "
            f"{case.batch:>2} {case.heads:>4} {case.padded_heads:>5} "
            f"{case.sequence:>6} "
            f"({candidate_rate:>3},{candidate_topk:>3},{candidate_window:>3}) "
            f"({csa_rate:>3},{csa_topk:>3},{csa_window:>3}) "
            f"{case.rope_dims:>4} {'yes' if case.is_performance_gate else 'no':>4}"
        )
    return "\n".join(rows)


def _make_random_factory(seed: int) -> Callable[..., torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape: int, scale: float = 0.2) -> torch.Tensor:
        return (
            torch.randn(
                *shape,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            * scale
        )

    return randn


def _make_candidate_inputs(
    case: ShapeCase,
    seed: int,
) -> tuple[tuple[torch.Tensor | int | bool, ...], tuple[int, ...]]:
    randn = _make_random_factory(seed)
    query = F.normalize(
        randn(case.batch, case.heads, case.sequence, HEAD_DIM),
        dim=-1,
    )
    kv = randn(case.batch, 1, case.sequence, HEAD_DIM)
    norm_weight = 1.0 + randn(HEAD_DIM, scale=0.05)
    sink = randn(case.heads)
    if case.variant == "hca":
        raw_inputs: tuple[torch.Tensor | int | bool, ...] = (
            query,
            kv,
            randn(case.batch, 1, case.sequence, HEAD_DIM),
            randn(case.batch, 1, case.sequence, HEAD_DIM),
            randn(case.compression_rate, HEAD_DIM),
            norm_weight,
            1.0 + randn(HEAD_DIM, scale=0.05),
            sink,
            case.compression_rate,
            case.window,
            case.rope_dims,
            True,
        )
        differentiable = HCA_DIFFERENTIABLE_INPUTS
    else:
        raw_inputs = (
            query,
            kv,
            norm_weight,
            sink,
            case.window,
            case.rope_dims,
            True,
        )
        differentiable = SWA_DIFFERENTIABLE_INPUTS
    return _make_differentiable(raw_inputs, differentiable), differentiable


def _make_csa_inputs(
    case: ShapeCase,
    seed: int,
) -> tuple[tuple[torch.Tensor | int | bool, ...], tuple[int, ...]]:
    randn = _make_random_factory(seed)
    compression_rate, topk, window = csa_sparse_configuration(case)
    raw_inputs: tuple[torch.Tensor | int | bool, ...] = (
        F.normalize(
            randn(case.batch, case.heads, case.sequence, HEAD_DIM),
            dim=-1,
        ),
        F.normalize(
            randn(case.batch, INDEX_HEADS, case.sequence, INDEX_DIM),
            dim=-1,
        ),
        randn(case.batch, 1, case.sequence, HEAD_DIM),
        randn(case.batch, 1, case.sequence, HEAD_DIM),
        randn(case.batch, 1, case.sequence, HEAD_DIM),
        randn(case.batch, 1, case.sequence, HEAD_DIM),
        randn(case.batch, 1, case.sequence, HEAD_DIM),
        randn(compression_rate, HEAD_DIM),
        randn(compression_rate, HEAD_DIM),
        randn(case.batch, case.sequence, INDEX_HEADS),
        randn(case.batch, 1, case.sequence, INDEX_DIM),
        randn(case.batch, 1, case.sequence, INDEX_DIM),
        randn(case.batch, 1, case.sequence, INDEX_DIM),
        randn(case.batch, 1, case.sequence, INDEX_DIM),
        randn(compression_rate, INDEX_DIM),
        randn(compression_rate, INDEX_DIM),
        1.0 + randn(HEAD_DIM, scale=0.05),
        1.0 + randn(INDEX_DIM, scale=0.05),
        1.0 + randn(HEAD_DIM, scale=0.05),
        randn(case.heads),
        compression_rate,
        topk,
        window,
        case.rope_dims,
        True,
    )
    return (
        _make_differentiable(raw_inputs, CSA_DIFFERENTIABLE_INPUTS),
        CSA_DIFFERENTIABLE_INPUTS,
    )


def _make_differentiable(
    inputs: tuple[torch.Tensor | int | bool, ...],
    differentiable: tuple[int, ...],
) -> tuple[torch.Tensor | int | bool, ...]:
    return tuple(
        value.detach().contiguous().requires_grad_(index in differentiable)
        if isinstance(value, torch.Tensor)
        else value
        for index, value in enumerate(inputs)
    )


def _forward_callable(
    operator: Literal["hca", "swa", "csa"],
    inputs: tuple[torch.Tensor | int | bool, ...],
) -> Callable[[], torch.Tensor]:
    if operator == "hca":
        from attn_gym.sparse.heavily_compressed_attention.api import (
            heavily_compressed_attention,
        )

        return lambda: heavily_compressed_attention(*inputs, backend="cute")
    if operator == "swa":
        from attn_gym.sparse.sliding_window_attention.api import (
            sliding_window_attention,
        )

        return lambda: sliding_window_attention(*inputs, backend="cute")
    from attn_gym.sparse.compressed_sparse_attention.api import (
        compressed_sparse_attention,
    )

    return lambda: compressed_sparse_attention(*inputs, backend="cute")


def _timed(
    function: Callable[[], object],
    *,
    warmup: int,
    rep: int,
) -> float:
    # Keep synchronization explicit around Triton's already synchronized timer.
    torch.cuda.synchronize()
    milliseconds = triton.testing.do_bench(
        function,
        warmup=warmup,
        rep=rep,
        return_mode="median",
    )
    torch.cuda.synchronize()
    return float(milliseconds)


def _measure_operator(
    case: ShapeCase,
    operator: Literal["hca", "swa", "csa"],
    *,
    seed: int,
    warmup: int,
    rep: int,
) -> Timing:
    if operator == "csa":
        inputs, differentiable = _make_csa_inputs(case, seed)
    else:
        inputs, differentiable = _make_candidate_inputs(case, seed)
    targets = tuple(inputs[index] for index in differentiable)
    forward = _forward_callable(operator, inputs)

    # Compile both paths before either timed region.
    forward()
    output = forward()
    if output.grad_fn is None:
        raise RuntimeError(f"{operator.upper()} output has no autograd graph.")
    generator = torch.Generator(device=output.device).manual_seed(seed + 10_000)
    grad_output = torch.randn(
        output.shape,
        device=output.device,
        dtype=output.dtype,
        generator=generator,
    )

    def backward() -> tuple[torch.Tensor | None, ...]:
        return torch.autograd.grad(
            output,
            targets,
            grad_outputs=grad_output,
            retain_graph=True,
            allow_unused=True,
        )

    backward()
    torch.cuda.synchronize()
    forward_ms = _timed(forward, warmup=warmup, rep=rep)
    backward_ms = _timed(backward, warmup=warmup, rep=rep)
    return Timing(forward_ms, backward_ms)


def benchmark_case(
    case: ShapeCase,
    *,
    seed: int,
    warmup: int,
    rep: int,
    csa_first: bool,
) -> Comparison:
    measurements = ("csa", case.variant) if csa_first else (case.variant, "csa")
    timings = {}
    for operator in measurements:
        timings[operator] = _measure_operator(
            case,
            operator,
            seed=seed,
            warmup=warmup,
            rep=rep,
        )
        gc.collect()
        torch.cuda.empty_cache()
    csa = timings["csa"]
    candidate = timings[case.variant]
    return Comparison(case, candidate, csa)


def format_results(comparisons: list[Comparison], max_ratio: float) -> str:
    header = (
        "case                         phase       candidate_ms       csa_ms"
        "    candidate/csa status"
    )
    rows = [header, "-" * len(header)]
    for comparison in comparisons:
        measurements = (
            (
                "forward",
                comparison.candidate.forward_ms,
                comparison.csa.forward_ms,
                comparison.forward_ratio,
            ),
            (
                "backward",
                comparison.candidate.backward_ms,
                comparison.csa.backward_ms,
                comparison.backward_ratio,
            ),
        )
        for phase, candidate_ms, csa_ms, ratio in measurements:
            if not comparison.case.is_performance_gate:
                status = "INFO"
            else:
                status = "PASS" if ratio <= max_ratio else "FAIL"
            rows.append(
                f"{comparison.case.name:<28} {phase:<8} "
                f"{candidate_ms:>16.6f} {csa_ms:>12.6f} "
                f"{ratio:>16.8f} {status}"
            )
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("all", "hca", "swa"), default="all")
    parser.add_argument(
        "--category",
        choices=(
            "all",
            "default",
            "scaling",
            "padded-head",
            "degenerate",
            "boundary",
        ),
        default="default",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        choices=tuple(case.name for case in SHAPE_CASES),
        help="Run only this case; may be specified more than once.",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print the shape/comparator matrix and exit without CUDA.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup duration in ms")
    parser.add_argument("--rep", type=int, default=500, help="Measurement duration in ms")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=DEFAULT_MAX_RATIO,
        help="Maximum raw candidate_ms / csa_ms ratio for default gates.",
    )
    parser.add_argument(
        "--enforce",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if either raw wall-clock ratio exceeds --max-ratio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = selected_cases(
        variant=args.variant,
        category=args.category,
        names=tuple(args.case),
    )
    if args.list_cases:
        print(format_case_table(cases))
        return
    if not cases:
        raise ValueError("The benchmark filters selected no cases.")
    if args.max_ratio <= 0 or not math.isfinite(args.max_ratio):
        raise ValueError("--max-ratio must be finite and positive.")
    if args.warmup <= 0 or args.rep <= 0:
        raise ValueError("--warmup and --rep must be positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("This benchmark targets SM100 exclusively.")

    print(f"device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("comparison matrix:")
    print(format_case_table(cases))
    comparisons = []
    for case_index, case in enumerate(cases):
        order = comparison_order(case_index)
        print(
            f"benchmarking {case.name} ({case_index + 1}/{len(cases)}; "
            f"order={order[0]}->{order[1]})...",
            flush=True,
        )
        comparisons.append(
            benchmark_case(
                case,
                seed=args.seed + case_index,
                warmup=args.warmup,
                rep=args.rep,
                csa_first=order[0] == "csa",
            )
        )

    print("\nraw synchronized wall-clock results:")
    print(format_results(comparisons, args.max_ratio))
    failures = [
        (comparison.case.name, phase, ratio)
        for comparison in comparisons
        if comparison.case.is_performance_gate
        for phase, ratio in (
            ("forward", comparison.forward_ratio),
            ("backward", comparison.backward_ratio),
        )
        if ratio > args.max_ratio
    ]
    if args.enforce and failures:
        details = ", ".join(f"{name}/{phase}={ratio:.8f}" for name, phase, ratio in failures)
        raise AssertionError(
            "Candidate CuTe wall time exceeded same-shape CSA wall time "
            f"(maximum raw ratio {args.max_ratio:g}): {details}."
        )


if __name__ == "__main__":
    main()
