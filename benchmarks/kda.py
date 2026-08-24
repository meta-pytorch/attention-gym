"""Correctness-preserving KDA performance matrix for reference, fused, and Mega."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class Case:
    name: str
    lengths: tuple[int, ...]
    heads: int
    mega_bf16_budget_us: float
    reference: bool = False


CASES = (
    Case("smoke", (128,), 1, 300.0, True),
    Case("primary", (8192,), 64, 3450.0),
    Case("dense_t1024_h16", (1024,), 16, 250.0),
    Case("dense_t1024_h64", (1024,), 64, 520.0),
    Case("dense_t4096_h16", (4096,), 16, 780.0),
    Case("dense_t4096_h64", (4096,), 64, 1850.0),
    Case("dense_t32768_h16", (32768,), 16, 5500.0),
    Case("dense_t32768_h64", (32768,), 64, 14000.0),
    Case("packed_balanced", (4096, 4096), 64, 1950.0),
    Case("packed_imbalanced", (65, 1024, 4096, 0, 127), 64, 1950.0),
    Case("packed_many_short", (64,) * 64, 64, 1000.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark KDA training implementations against fixed correctness-preserving cases."
    )
    parser.add_argument(
        "--impl",
        nargs="+",
        choices=("reference", "fused", "mega"),
        default=("fused", "mega"),
        help="implementations to benchmark",
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
        help="Q/K/V and output-gradient dtype",
    )
    parser.add_argument("--rounds", type=int, default=3, help="timing rounds")
    parser.add_argument("--iterations", type=int, default=20, help="graph replays per round")
    parser.add_argument("--warmups", type=int, default=3, help="warmup iterations")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser.parse_args()


def make_inputs(case: Case, dtype: torch.dtype = torch.bfloat16):
    torch.manual_seed(0)
    tokens, dim = sum(case.lengths), 128
    shape = (1, tokens, case.heads, dim)
    values = (
        torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1).to(dtype),
        torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1).to(dtype),
        torch.randn(shape, device="cuda", dtype=dtype),
        torch.empty(shape, device="cuda").uniform_(0.5, 1.0).log(),
        torch.sigmoid(torch.randn(1, tokens, case.heads, device="cuda")),
    )
    inputs = tuple(value.requires_grad_() for value in values)
    offsets = [0]
    for length in case.lengths:
        offsets.append(offsets[-1] + length)
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    return inputs, cu_seqlens, torch.randn_like(inputs[2])


def forward_for(implementation: str, inputs, cu_seqlens):
    from attn_gym.linear import chunk_kda

    q, k, value, raw_gate, beta = inputs
    packed = cu_seqlens.shape[0] > 2
    output, _ = chunk_kda(
        q,
        k,
        value,
        raw_gate,
        beta,
        cu_seqlens=cu_seqlens if packed else None,
        autotune=False,
        impl="fused" if implementation == "mega" else implementation,
        kernel_options={"backend": "mega"} if implementation == "mega" else None,
    )
    return output


def benchmark(
    case: Case,
    implementation: str,
    dtype: torch.dtype,
    rounds: int,
    iterations: int,
    warmups: int,
):
    if implementation == "reference" and not case.reference:
        return {"status": "skipped", "reason": "reference is restricted to smoke cases"}
    inputs, cu_seqlens, d_output = make_inputs(case, dtype)

    def step():
        output = forward_for(implementation, inputs, cu_seqlens)
        return torch.autograd.grad(output, inputs, d_output)

    for _ in range(warmups):
        step()
    torch.cuda.synchronize()

    if implementation == "reference":
        graph = None
    else:
        graph = torch.cuda.CUDAGraph()
        torch.autograd.graph.set_override_stale_capture_stream(True)
        try:
            with torch.cuda.graph(graph):
                captured = step()
        finally:
            torch.autograd.graph.set_override_stale_capture_stream(False)
        del captured
        for _ in range(warmups):
            graph.replay()
        torch.cuda.synchronize()

    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            step() if graph is None else graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    median = statistics.median(samples)
    budget_us = (
        case.mega_bf16_budget_us if implementation == "mega" and dtype == torch.bfloat16 else None
    )
    passed = budget_us is None or median <= budget_us
    return {
        "status": "passed" if passed else "failed",
        "samples_us": samples,
        "median_us": median,
        "budget_us": budget_us,
    }


def main() -> None:
    args = parse_args()
    selected = [case for case in CASES if args.cases is None or case.name in args.cases]
    dtype = getattr(torch, args.dtype)
    results = []
    failed = False
    for case in selected:
        for implementation in args.impl:
            result = benchmark(
                case,
                implementation,
                dtype,
                args.rounds,
                args.iterations,
                args.warmups,
            )
            failed |= result["status"] == "failed"
            row = {"case": asdict(case), "impl": implementation, **result}
            results.append(row)
            print(json.dumps(row), flush=True)
    payload = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "dtype": str(dtype),
        "results": results,
    }
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
