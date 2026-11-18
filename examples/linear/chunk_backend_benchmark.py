"""Time chunk GDN/KDA prefill and training under the old and new default backends.

The old default is ``kernel_options={"backend": "fused"}`` (Triton, autotuned); the new default
omits ``kernel_options``, so ``"auto"`` picks warpdrive on SM100-family GPUs. Each case is one
captured CUDA Graph: prefill is a forward under ``torch.no_grad()``, training a forward plus
``torch.autograd.grad`` of every input. Rounds interleave all cases and the table reports the
median replay time in microseconds. Run on one GPU with::

    python examples/linear/chunk_backend_benchmark.py --rounds 5

Q/K are L2-normalized, gates uniform in ``[-5, 0]`` (per channel for KDA, per head for GDN),
betas sigmoid of a normal. ``states`` cases pass an initial state and return the final state.
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch
import torch.nn.functional as F

from attn_gym.linear import chunk_gdn, chunk_kda

BACKENDS = {"fused": {"backend": "fused"}, "auto": None}
# (batch, tokens per sequence, heads); head dimension 128 throughout.
SHAPES = ((1, 8192, 16), (8, 2048, 16), (1, 2048, 8))


def make_inputs(family: str, batch: int, tokens: int, heads: int, states: bool):
    """Build BF16 Q/K/V, FP32 gate and beta, and an optional FP32 initial state as leaves."""
    shape = (batch, tokens, heads, 128)
    q, k = (F.normalize(torch.randn(shape, device="cuda"), dim=-1) for _ in range(2))
    v = torch.randn(shape, device="cuda")
    gate = torch.empty(shape if family == "kda" else shape[:3], device="cuda").uniform_(-5, 0)
    beta = torch.randn(shape[:3], device="cuda").sigmoid()
    inputs = [q.bfloat16(), k.bfloat16(), v.bfloat16(), gate, beta]
    if states:
        inputs.append(torch.randn(batch, heads, 128, 128, device="cuda") * 0.01)
    return [tensor.requires_grad_() for tensor in inputs]


def make_step(family: str, inputs, backend: str, training: bool) -> Callable[[], object]:
    """Return one prefill or training step on fixed inputs."""
    op = chunk_kda if family == "kda" else chunk_gdn
    states = len(inputs) == 6

    def step():
        with torch.set_grad_enabled(training):
            output, final_state = op(
                *inputs, output_final_state=states, kernel_options=BACKENDS[backend]
            )
            if not training:
                return output, final_state
            outputs = (output, final_state) if states else (output,)
            return torch.autograd.grad(outputs, inputs, [torch.ones_like(t) for t in outputs])

    return step


def capture(step: Callable[[], object]) -> torch.cuda.CUDAGraph:
    """Warm up (tuning Triton) on a side stream, then capture one step.

    The graph does not own the step's inputs; the caller keeps them alive while it replays.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    return graph


def time_replay(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    """Return the mean replay time in microseconds."""
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1e3 / replays


def describe_environment() -> str:
    """Name the source revision, GPU and toolchain the timings belong to."""
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "--short=8", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    try:
        cutedsl = version("nvidia-cutlass-dsl")
    except PackageNotFoundError:
        cutedsl = "not installed"
    return (
        f"revision {revision or 'unknown'}, {torch.cuda.get_device_name()}, "
        f"torch {torch.__version__}, CuTeDSL {cutedsl}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--families", nargs="+", default=["gdn", "kda"])
    args = parser.parse_args()

    torch.manual_seed(0)
    graphs, live_inputs = {}, []
    for family in args.families:
        for shape in SHAPES:
            for states in (False, True):
                inputs = make_inputs(family, *shape, states)
                live_inputs.append(inputs)
                for phase in ("prefill", "training"):
                    for backend in BACKENDS:
                        step = make_step(family, inputs, backend, phase == "training")
                        key = (family, shape, states, phase, backend)
                        graphs[key] = capture(step)
    times = {key: [] for key in graphs}
    for round_index in range(args.rounds):
        # Alternate the order so neither backend always runs first.
        keys = list(graphs) if round_index % 2 == 0 else list(reversed(graphs))
        for key in keys:
            times[key].append(time_replay(graphs[key], args.replays))

    print(describe_environment())
    print(f"CUDA Graph replay, median of {args.rounds} interleaved rounds, µs")
    print("| Op | Shape (B × T, H) | States | Phase | fused (µs) | auto (µs) | fused / auto |")
    print("| --- | --- | --- | --- | ---: | ---: | ---: |")
    for family, (batch, tokens, heads), states, phase, backend in graphs:
        if backend != "fused":
            continue
        fused, auto = (
            statistics.median(times[family, (batch, tokens, heads), states, phase, name])
            for name in BACKENDS
        )
        print(
            f"| {family.upper()} | {batch} × {tokens}, {heads} | {'yes' if states else 'no'} "
            f"| {phase} | {fused:.0f} | {auto:.0f} | {fused / auto:.2f} |"
        )


if __name__ == "__main__":
    main()
