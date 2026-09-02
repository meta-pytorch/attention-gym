"""Public Mega chunk-delta performance and memory matrix for GDN and KDA."""

from __future__ import annotations

import argparse
import json
import shlex
import statistics
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from transformer_nuggets.utils.benchmark import benchmark_cuda_function_stats
from transformer_nuggets.utils.tracing import cuda_kernel_profiler

from attn_gym.linear import chunk_gdn, chunk_kda

Operation = Literal["gdn", "kda"]
OPERATIONS: tuple[Operation, ...] = ("gdn", "kda")


@dataclass(frozen=True)
class Case:
    name: str
    lengths: tuple[int, ...]
    key_heads: int
    value_heads: int
    gdn_budget_us: float | None
    kda_budget_us: float | None
    operations: tuple[Operation, ...] = OPERATIONS

    def budget_for(self, operation: Operation) -> float | None:
        """Return the B200 BF16 train-step budget for one operation."""
        return self.gdn_budget_us if operation == "gdn" else self.kda_budget_us


CASES = (
    Case("smoke", (128,), 1, 1, 80.0, 300.0),
    Case("dense_t1024_h16", (1024,), 16, 16, 275.0, 250.0),
    Case("dense_t1024_h64", (1024,), 64, 64, 330.0, 520.0),
    Case("dense_t4096_h16", (4096,), 16, 16, 1050.0, 780.0),
    Case("dense_t4096_h64", (4096,), 64, 64, 1150.0, 1850.0),
    Case("dense_t8192_h64", (8192,), 64, 64, 2250.0, 3450.0),
    Case("dense_t32768_h16", (32768,), 16, 16, None, 5500.0, ("kda",)),
    Case("dense_t32768_h64", (32768,), 64, 64, 9000.0, 14000.0),
    Case("packed_balanced", (4096, 4096), 64, 64, 1250.0, 1950.0),
    Case("packed_imbalanced", (65, 1024, 4096, 0, 127), 64, 64, 1250.0, 1950.0),
    Case("grouped_t4096", (4096,), 16, 64, 1350.0, None, ("gdn",)),
    Case("packed_many_short", (64,) * 64, 64, 64, None, 1000.0, ("kda",)),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the public Mega GDN and KDA chunk training paths."
    )
    parser.add_argument(
        "--op",
        action="append",
        dest="operations",
        choices=OPERATIONS,
        help="operation to run; may be repeated and defaults to both",
    )
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        choices=[case.name for case in CASES],
        help="case to run; may be repeated",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16"),
        default="bfloat16",
        help="Q/K/V and output-gradient dtype; budgets apply only to BF16",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def make_inputs(
    operation: Operation,
    case: Case,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor | None]:
    """Create one deterministic BF16 public chunk-delta workload."""
    torch.manual_seed(401)
    tokens, dim = sum(case.lengths), 128
    q_shape = (1, tokens, case.key_heads, dim)
    value_shape = (1, tokens, case.value_heads, dim)
    q = F.normalize(torch.randn(q_shape, device="cuda"), dim=-1).to(dtype)
    k = F.normalize(torch.randn_like(q.float()), dim=-1).to(dtype)
    value = torch.randn(value_shape, device="cuda", dtype=dtype)
    if operation == "gdn":
        gate = -F.softplus(torch.randn(value_shape[:-1], device="cuda"))
    else:
        gate = torch.empty(value_shape, device="cuda").uniform_(0.5, 1.0).log_()
    beta = torch.randn(value_shape[:-1], device="cuda").sigmoid_()
    offsets = [0]
    for length in case.lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = (
        None if len(case.lengths) == 1 else torch.tensor(offsets, dtype=torch.int32, device="cuda")
    )
    return (q, k, value, gate, beta), cu_seqlens


def operation_forward(
    operation: Operation,
    inputs: tuple[torch.Tensor, ...],
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Run one public Mega chunk implementation without final-state output."""
    q, k, value, gate, beta = inputs
    if operation == "gdn":
        return chunk_gdn(
            q,
            k,
            value,
            gate,
            beta,
            cu_seqlens=cu_seqlens,
            impl="fused",
        )[0]

    return chunk_kda(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens=cu_seqlens,
        autotune=False,
        impl="fused",
        kernel_options={"backend": "mega"},
    )[0]


def make_callables(
    operation: Operation,
    inputs: tuple[torch.Tensor, ...],
    cu_seqlens: torch.Tensor | None,
    d_output: torch.Tensor,
) -> dict[str, Callable[[], object]]:
    """Build fixed-pointer forward, isolated-backward, and fresh train-step callables."""
    leaves = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    output = operation_forward(operation, leaves, cu_seqlens)

    def forward() -> torch.Tensor:
        return operation_forward(operation, inputs, cu_seqlens)

    def backward() -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(output, leaves, d_output, retain_graph=True)

    def train_step() -> tuple[torch.Tensor, ...]:
        train_output = operation_forward(operation, leaves, cu_seqlens)
        return torch.autograd.grad(train_output, leaves, d_output)

    return {"forward": forward, "backward": backward, "train_step": train_step}


def benchmark_phase(
    function: Callable[[], object],
    rounds: int,
    iterations: int,
    warmups: int,
) -> dict[str, object]:
    """Collect and summarize CUDA Graph replay timings for one phase."""
    samples = []
    for round_index in range(rounds):
        stats = benchmark_cuda_function_stats(
            function,
            NUM_ITERS=iterations,
            CUDAGRAPH_WARMUP_ITERS=warmups,
            USE_CUDA_GRAPHS=True,
            N_RESAMPLES=500,
            SEED=round_index,
        )
        samples.append(asdict(stats))
    return {
        "rounds": samples,
        "median_of_round_medians_us": statistics.median(result["median_us"] for result in samples),
    }


def measure_incremental_peak_bytes(function: Callable[[], object], warmups: int) -> int:
    """Measure peak allocated bytes above the materialized fixed workload."""
    for _ in range(warmups):
        function()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    function()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() - baseline


def confirm_kernel_route(function: Callable[[], object], operation: Operation) -> list[str]:
    """Record and validate the selected Mega kernel family outside the timed region."""
    pattern = "Gdn" if operation == "gdn" else "Kda"
    with cuda_kernel_profiler(pattern, record_name=f"{operation}_route") as result:
        function()
    if not result["found"]:
        raise RuntimeError(
            f"{operation} did not execute a kernel containing {pattern!r}; "
            f"observed {result['kernel_names']}"
        )
    return list(result["kernel_names"])


def benchmark_case(
    operation: Operation,
    case: Case,
    rounds: int,
    iterations: int,
    warmups: int,
    dtype: torch.dtype,
    enforce_budgets: bool,
) -> dict[str, object]:
    """Measure one public Mega operation under the frozen chunk workload."""
    inputs, cu_seqlens = make_inputs(operation, case, dtype)
    d_output = torch.randn_like(inputs[2])
    callables = make_callables(operation, inputs, cu_seqlens, d_output)
    for function in callables.values():
        function()
    torch.cuda.synchronize()

    route = confirm_kernel_route(callables["forward"], operation)
    peak_bytes = measure_incremental_peak_bytes(callables["train_step"], warmups)
    phases = {
        phase: benchmark_phase(function, rounds, iterations, warmups)
        for phase, function in callables.items()
    }
    budget_us = case.budget_for(operation)
    train_us = phases["train_step"]["median_of_round_medians_us"]
    return {
        "operation": operation,
        "case": asdict(case),
        "phases": phases,
        "incremental_peak_bytes": peak_bytes,
        "kernel_names": route,
        "budget_us": budget_us,
        "budget_enforced": enforce_budgets and budget_us is not None,
        "budget_passed": (
            None if not enforce_budgets or budget_us is None else train_us <= budget_us
        ),
    }


def git_revision() -> str:
    """Describe the repository revision without requiring GitPython."""
    revision = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    return f"{revision}+working-tree" if dirty else revision


def main() -> None:
    args = parse_args()
    operations = OPERATIONS if args.operations is None else tuple(args.operations)
    selected_cases = [case for case in CASES if args.cases is None or case.name in args.cases]
    workloads = [
        (operation, case)
        for operation in operations
        for case in selected_cases
        if operation in case.operations
    ]
    if not workloads:
        raise ValueError("the selected operations and cases have no supported combinations")
    gpu = torch.cuda.get_device_name()
    dtype = getattr(torch, args.dtype)
    enforce_budgets = "B200" in gpu and dtype == torch.bfloat16
    torch.autograd.graph.set_override_stale_capture_stream(True)
    try:
        payload = {
            "date": datetime.now(UTC).date().isoformat(),
            "revision": git_revision(),
            "command": shlex.join(sys.argv),
            "gpu": gpu,
            "torch": torch.__version__,
            "dtype": str(dtype),
            "budgets_enforced": enforce_budgets,
            "contract": (
                "public API; fixed pointers; warm cache; CUDA Graph replay; unlocked clocks with "
                "no explicit stabilization; separate forward, isolated backward, and fresh "
                "train-step timing; compilation, warmup, route profiling, and allocation setup "
                "excluded"
            ),
            "results": [
                benchmark_case(
                    operation,
                    case,
                    args.rounds,
                    args.iterations,
                    args.warmups,
                    dtype,
                    enforce_budgets,
                )
                for operation, case in workloads
            ],
        }
    finally:
        torch.autograd.graph.set_override_stale_capture_stream(False)
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if any(result["budget_passed"] is False for result in payload["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
