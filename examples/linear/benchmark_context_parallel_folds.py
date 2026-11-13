"""Compare root-only and independent folds through public KDA/GDN context parallelism.

Run with two reserved NVIDIA GPUs, for example:

    torchrun --standalone --nproc-per-node=2 examples/linear/benchmark_context_parallel_folds.py --variant kda --tokens 32768 --fragments 512 --output results.json

The independent fold is a benchmark baseline, not a production scheduling option.
Both sides use the same inputs and routing. Forward and backward are captured and
measured separately, including NCCL but excluding setup, projections and short convolution.
Use --slots to measure a small live layout under a larger fixed routing capacity.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import statistics
import subprocess
from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Annotated, NamedTuple, cast
from unittest.mock import patch

import torch
import torch.distributed as dist
import typer

import attn_gym._backends.triton.utils as triton_utils
import attn_gym.linear._delta_rule.triton.fold_slots as fold_impl
import attn_gym.linear.context_parallel as cp_impl
import attn_gym.testing.delta_rule as timing
import attn_gym.testing.gdn as gdn_inputs
import attn_gym.testing.kda as kda_inputs
from attn_gym.linear.gdn import context_parallel_gdn
from attn_gym.linear.kda import context_parallel_kda


class Variant(str, Enum):
    KDA = "kda"
    GDN = "gdn"


class CapturedPhase(NamedTuple):
    graph: torch.cuda.CUDAGraph
    outputs: tuple[torch.Tensor, ...]
    fold_calls: int


@dataclass
class FoldCounter:
    fold: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    calls: int = 0

    def __call__(
        self, gathered: torch.Tensor, sources: torch.Tensor, links: torch.Tensor
    ) -> torch.Tensor:
        self.calls += 1
        return self.fold(gathered, sources, links)


def independent_fold(
    gathered: torch.Tensor, sources: torch.Tensor, links: torch.Tensor
) -> torch.Tensor:
    return cp_impl.fold_slots(gathered, sources)


@contextmanager
def capture_phases(
    forward: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    inputs: tuple[torch.Tensor, ...],
    cotangents: tuple[torch.Tensor, torch.Tensor],
    stream: torch.cuda.Stream,
    warmup: int,
    fold_calls: Callable[[], int],
) -> Iterator[dict[str, CapturedPhase]]:
    """Keep captured tensors alive and release NCCL graphs before their process group."""
    backward = partial(
        torch.autograd.grad, inputs=inputs, grad_outputs=cotangents, retain_graph=True
    )
    graphs = []
    try:
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                backward(forward())
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        forward_graph = torch.cuda.CUDAGraph()
        graphs.append(forward_graph)
        before_forward = fold_calls()
        with torch.cuda.graph(forward_graph, stream=stream):
            outputs = forward()
        forward_calls = fold_calls() - before_forward
        forward_graph.replay()
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            for _ in range(warmup):
                backward(outputs)
        torch.cuda.synchronize()
        backward_graph = torch.cuda.CUDAGraph()
        graphs.append(backward_graph)
        before_backward = fold_calls()
        with torch.cuda.graph(backward_graph, stream=stream):
            gradients = backward(outputs)
        backward_calls = fold_calls() - before_backward
        backward_graph.replay()
        torch.cuda.synchronize()
        yield {
            "forward": CapturedPhase(forward_graph, outputs, forward_calls),
            "backward": CapturedPhase(backward_graph, gradients, backward_calls),
        }
    finally:
        for graph in graphs:
            graph.reset()


def assert_exact_parity(
    captured: dict[str, dict[str, CapturedPhase]], device: torch.device
) -> None:
    mismatches = []
    for phase in ("forward", "backward"):
        for label in ("independent", "root"):
            if captured[label][phase].fold_calls == 0:
                mismatches.append(f"{label} {phase} fold was not captured")
        independent = captured["independent"][phase].outputs
        root = captured["root"][phase].outputs
        for index, (expected, actual) in enumerate(zip(independent, root, strict=True)):
            if not torch.isfinite(actual).all() or not torch.equal(expected, actual):
                mismatches.append(f"{phase}[{index}]")
    passed = torch.tensor(int(not mismatches), device=device)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    # Every rank completes the collective before reporting a mismatch.
    assert passed.item(), f"rank {dist.get_rank()}: mismatches {mismatches} (or a peer failed)"


def summarize(ranks: Sequence[list[dict[str, object]]], rounds: int) -> dict[str, float]:
    result = {}
    for label in ("independent", "root"):
        for phase in ("forward", "backward"):
            medians = []
            for round_index in range(rounds):
                samples = [
                    cast(
                        list[float],
                        next(
                            row["step_ms"]
                            for row in rows
                            if (row["round"], row["label"], row["phase"])
                            == (round_index, label, phase)
                        ),
                    )
                    for rows in ranks
                ]
                medians.append(
                    statistics.median(max(times) for times in zip(*samples, strict=True))
                )
            result[f"{label}_{phase}_ms"] = statistics.median(medians)
    return result


def source_hashes() -> dict[str, str]:
    files = [Path(__file__)] + [
        Path(module.__file__)
        for module in (cp_impl, fold_impl, triton_utils, timing, kda_inputs, gdn_inputs)
    ]
    return {str(item): hashlib.sha256(item.read_bytes()).hexdigest() for item in files}


def gpu_telemetry(device: torch.device) -> tuple[str, str]:
    gpu_uuid = f"GPU-{torch.cuda.get_device_properties(device).uuid}"
    readings = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpu_uuid}",
            "--query-gpu=uuid,clocks.sm,power.draw,temperature.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    )
    return gpu_uuid, readings


def repository_state() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    command = ["git", "-C", str(root)]
    commit = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output([*command, "diff", "--binary", "HEAD", "--"])
    untracked = subprocess.check_output(
        [*command, "ls-files", "--others", "--exclude-standard", "-z"], text=True
    )
    return {
        "commit": commit,
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in untracked.split("\0")
            if name and (root / name).is_file()
        },
    }


def main(
    output: Annotated[Path, typer.Option(help="Rank-zero JSON report, including raw samples.")],
    variant: Variant = Variant.KDA,
    tokens: Annotated[int, typer.Option(min=1, help="Global tokens in one document.")] = 512,
    fragments: Annotated[int, typer.Option(min=2, help="Global equal-length fragments.")] = 16,
    heads: Annotated[int, typer.Option(min=1)] = 2,
    slots: Annotated[
        int | None, typer.Option(min=1, help="Per-rank routing capacity; defaults to exact.")
    ] = None,
    rounds: Annotated[int, typer.Option(min=1, help="Odd number of interleaved A/B rounds.")] = 5,
    steps: Annotated[int, typer.Option(min=1, help="Timed replays per round and phase.")] = 20,
    warmup: Annotated[int, typer.Option(min=1)] = 3,
) -> None:
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise typer.BadParameter("launch with torchrun --standalone --nproc-per-node=2")
    if tokens % fragments or fragments % 2:
        raise typer.BadParameter(
            "tokens must be divisible by fragments, and fragments must be even"
        )
    if rounds % 2 == 0:
        raise typer.BadParameter("rounds must be odd")
    if slots is not None and slots < fragments // 2:
        raise typer.BadParameter("slots must fit every rank's fragments")
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    sources = source_hashes()
    repository = repository_state()
    dist.init_process_group("nccl", device_id=device)
    try:
        width = tokens // fragments
        owned = [[(i * width, (i + 1) * width) for i in range(r, fragments, 2)] for r in range(2)]
        plan = cp_impl.ContextParallelPlan.from_fragments((0, tokens), owned, rank)
        routing = plan.routing(device, slots=slots)
        ids = plan.global_token_ids(device)
        if variant is Variant.KDA:
            global_inputs = kda_inputs.make_kda_test_inputs(
                tokens, heads=heads, seed=42, normalize_qk=True
            )
            operation = context_parallel_kda
        else:
            global_inputs = gdn_inputs.make_gdn_test_inputs(
                tokens, key_heads=heads, value_heads=heads, seed=42
            )[:5]
            operation = context_parallel_gdn
        inputs = tuple(t[:, ids].contiguous().requires_grad_() for t in global_inputs)
        generator = torch.Generator(device=device).manual_seed(43 + rank)
        d_output = torch.randn(inputs[2].shape, device=device, generator=generator).to(
            inputs[2].dtype
        )
        value_heads, value_dim = inputs[2].shape[-2:]
        key_dim = inputs[1].shape[-1]
        state_shape = (value_heads, value_dim, key_dim)
        d_exit = torch.zeros(len(plan.subsequences), *state_shape, device=device)
        for row in plan.terminal:
            d_exit[row] = torch.randn(*state_shape, device=device, generator=generator)
        forward = partial(
            operation,
            *inputs,
            routing=routing,
            group=dist.group.WORLD,
            kernel_options={"autotune": False},
        )
        captured, records = {}, []
        stream = torch.cuda.Stream()
        routed_fold = cp_impl._fold_routed_slots
        with ExitStack() as captures:
            for label, fold in (("independent", independent_fold), ("root", routed_fold)):
                counter = FoldCounter(fold)
                with patch.object(cp_impl, "_fold_routed_slots", counter):
                    captured[label] = captures.enter_context(
                        capture_phases(
                            forward,
                            inputs,
                            (d_output, d_exit),
                            stream,
                            warmup,
                            lambda counter=counter: counter.calls,
                        )
                    )
            assert_exact_parity(captured, device)
            if rank == 0:
                print("Capture and exact parity passed; measuring", flush=True)
            for round_index in range(rounds):
                labels = (
                    ("independent", "root") if round_index % 2 == 0 else ("root", "independent")
                )
                for phase in ("forward", "backward"):
                    for label in labels:
                        gpu_uuid, clocks = gpu_telemetry(device)
                        measurement = timing.measure_training_step(
                            captured[label][phase].graph.replay,
                            device,
                            steps=steps,
                            warmup_steps=warmup,
                            local_tokens=len(ids),
                        )
                        records.append(
                            {
                                "round": round_index,
                                "phase": phase,
                                "label": label,
                                "gpu_uuid": gpu_uuid,
                                "telemetry": clocks,
                                **measurement,
                            }
                        )
        gathered = [None, None]
        dist.all_gather_object(gathered, records)
        if rank == 0:
            assert source_hashes() == sources, "source files changed during the benchmark"
            assert repository_state() == repository, "repository changed during the benchmark"
            report = {
                "variant": variant.value,
                "tokens": tokens,
                "global_fragments": fragments,
                "local_fragments": fragments // 2,
                "slots": routing.slots,
                "heads": value_heads,
                "key": key_dim,
                "value": value_dim,
                "gpu": torch.cuda.get_device_name(),
                "devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "versions": {
                    p: importlib.metadata.version(p)
                    for p in ("torch", "triton", "nvidia-cutlass-dsl")
                },
                "rounds": rounds,
                "steps": steps,
                "warmup": warmup,
                "timing": "separate_forward_backward_cuda_graphs",
                "buffer_reuse": "fixed",
                "clock_policy": "default_dvfs",
                "input_seed": 42,
                "cotangent_seed_base": 43,
                "input_dtypes": [str(t.dtype) for t in inputs],
                "kernel_options": {"autotune": False},
                "repository": repository,
                "captured_fold_calls": {
                    label: {phase: value.fold_calls for phase, value in phases.items()}
                    for label, phases in captured.items()
                },
                "source_sha256": sources,
                "exact_parity": True,
                "summary": summarize(gathered, rounds),
                "ranks": gathered,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report["summary"]), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    typer.run(main)
