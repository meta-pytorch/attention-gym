"""Time warpdrive GDN/KDA chained and unchained plans on few-tile shapes.

The evidence behind NOTE [Chain plan cost] (attn_gym/linear/_delta_rule/warpdrive/drivers/
plan.py). For each ``NxTxH`` case (N sequences of T tokens, H heads, BF16, ``--dim``, no state) it
CUDA-graph-times the forward under ``no_grad`` and the backward alone (forward outside the graph)
with the fused backend, the automatic warpdrive plan, and pinned plans: unchained (the forward's
d_v split when it fits) and the chain at the geometry's piece count and at half of it. Rows are
JSON lines of median microseconds per plan.

    python benchmarks/warpdrive_chain_plans.py out.jsonl --case 1x2048x8 --case 4x1024x16
    python benchmarks/warpdrive_chain_plans.py out.jsonl --family gdn --dim 64 --case 1x2048x32
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F
from transformer_nuggets.utils.benchmark import benchmark_cuda_graph_stats

from attn_gym.linear import chunk_gdn, chunk_kda
from attn_gym.linear._delta_rule.warpdrive.drivers import gdn, kda
from attn_gym.linear._delta_rule.warpdrive.piece_chain import choose_pieces

DRIVERS = {"gdn": gdn, "kda": kda}
OPS = {"gdn": chunk_gdn, "kda": chunk_kda}


@contextmanager
def pinned_pieces(driver, forward: int | None, backward: int | None) -> Iterator[None]:
    """Pin the chain piece counts the plans choose (``None`` keeps the automatic plan); an
    unchained forward keeps its automatic d_v split."""
    builds = driver.ForwardPlan.build, driver.BackwardPlan.build
    if forward is not None:
        sms = torch.cuda.get_device_properties().multi_processor_count

        def forward_plan(cls, tokens, num_seqs, heads, dim_v, device):
            auto = builds[0](tokens, num_seqs, heads, dim_v, device)
            dv = 2 if not forward and dim_v == 128 and num_seqs * heads * 2 <= sms else 1
            return replace(auto, pieces=forward, tiles_per_head=dv)

        driver.ForwardPlan.build = classmethod(forward_plan)
    if backward is not None:
        driver.BackwardPlan.build = classmethod(
            lambda cls, *args: replace(builds[1](*args), pieces=backward)
        )
    try:
        yield
    finally:
        driver.ForwardPlan.build, driver.BackwardPlan.build = builds


def capture(fn: Callable[[], object]) -> torch.cuda.CUDAGraph:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph


def interleaved_medians(graphs: dict[str, torch.cuda.CUDAGraph], rounds: int) -> dict:
    """Median microseconds per graph over ``rounds`` alternating-order rounds."""
    times = {name: [] for name in graphs}
    for index in range(rounds):
        for name in graphs if index % 2 == 0 else reversed(graphs):
            stats = benchmark_cuda_graph_stats(
                graphs[name], num_iters=20, warmup_iters=10, n_resamples=10
            )
            times[name].append(statistics.median(stats.samples_us))
    return {name: statistics.median(values) for name, values in times.items()}


def inputs_for(family: str, num_seqs: int, tokens: int, heads: int, dim: int):
    generator = torch.Generator(device="cuda").manual_seed(123)
    total = num_seqs * tokens

    def uniform(*sizes):
        return 2 * torch.rand(sizes, device="cuda", generator=generator) - 1

    q, k = (F.normalize(uniform(1, total, heads, dim), dim=-1).bfloat16() for _ in range(2))
    value = uniform(1, total, heads, dim).bfloat16()
    gate_shape = (1, total, heads) if family == "gdn" else (1, total, heads, dim)
    gate = -torch.rand(gate_shape, device="cuda", generator=generator) * 0.3
    beta = torch.rand((1, total, heads), device="cuda", generator=generator)
    leaves = tuple(t.requires_grad_() for t in (q, k, value, gate, beta))
    offsets = torch.arange(num_seqs + 1, device="cuda", dtype=torch.int32) * tokens
    return leaves, offsets, uniform(1, total, heads, dim).bfloat16()


def measure(family: str, case: str, dim: int, rounds: int) -> dict:
    num_seqs, tokens, heads = (int(x) for x in case.split("x"))
    driver, op = DRIVERS[family], OPS[family]
    leaves, offsets, d_output = inputs_for(family, num_seqs, tokens, heads, dim)
    sms = torch.cuda.get_device_properties().multi_processor_count

    def forward(backend: str) -> torch.Tensor:
        options = {"backend": backend} | ({"autotune": False} if backend == "fused" else {})
        return op(*leaves, cu_seqlens=offsets, kernel_options=options)[0]

    row = {"family": family, "case": case, "dim": dim, "num_sm": sms}
    for direction, reverse in (("fwd", False), ("bwd", True)):
        geo = choose_pieces(
            num_seqs=num_seqs, heads_out=heads, num_sm=sms, total_tokens=num_seqs * tokens,
            b_t=driver.B_T, reverse=reverse,
        )  # fmt: skip
        pieces = sorted({0, geo, geo // 2} - {1})
        plans = {"fused": None, "auto": None, **{f"pieces{p}": p for p in pieces}}
        graphs, keep = {}, []
        for name, count in plans.items():
            backend = "fused" if name == "fused" else "warpdrive"
            pins = (count, None) if direction == "fwd" else (None, count)
            with pinned_pieces(driver, *pins):
                if direction == "fwd":
                    with torch.no_grad():
                        graphs[name] = capture(lambda b=backend: forward(b))
                else:
                    output = forward(backend)
                    keep.append(output)  # replays read the forward's saved tensors
                    graphs[name] = capture(
                        lambda o=output: torch.autograd.grad(
                            o, leaves, d_output, retain_graph=True
                        )
                    )
        row[f"{direction}_geo_pieces"] = geo
        row[direction] = interleaved_medians(graphs, rounds)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path, help="JSON-lines file to append rows to")
    parser.add_argument("--family", action="append", choices=DRIVERS, dest="families")
    parser.add_argument("--case", action="append", dest="cases", help="NxTxH, may be repeated")
    parser.add_argument("--dim", type=int, choices=(64, 128), default=128, help="K = V head dim")
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    cases = args.cases or [
        f"1x{t}x{h}" for h, t in itertools.product((2, 8, 32), (1024, 2048, 8192))
    ]
    # Replays run backward graphs captured on a side stream after an eager forward.
    torch.autograd.graph.set_override_stale_capture_stream(True)
    with args.output.open("a") as file, torch.autograd.set_multithreading_enabled(False):
        for family, case in itertools.product(args.families or list(DRIVERS), cases):
            row = measure(family, case, args.dim, args.rounds)
            file.write(json.dumps(row) + "\n")
            file.flush()
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
