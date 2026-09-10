"""Overhead of CP-degree-invariant KDA against the standard CP recipe and the unsharded op.

    torchrun --standalone --nproc-per-node=2 benchmarks/kda_cp_deterministic.py --heads 64 --tokens 8192

Every rank times its own fwd and bwd with CUDA events; the reported time per iteration is the
slowest rank, which is what gates a training step. Rounds interleave the variants (A,B,C / C,B,A)
so clock drift hits them evenly. Rank 0 prints a table and writes JSON to ``--out``.

Variants:
  unsharded    ``chunk_kda`` over the whole packed stream on every rank (CP=1 reference cost).
  cp           ``context_parallel_kda`` with contiguous fragments cut at tile boundaries.
  det-<T>      ``context_parallel_kda_deterministic`` with canonical tile size T.

The document layout is a fixed Zipf-like ragged set scaled to ``--tokens``; the standard CP
fragments are cut on the deterministic tile grid so both recipes move the same tokens.
"""

from __future__ import annotations

import json
import os
import statistics
from collections.abc import Callable
from functools import partial
from itertools import accumulate, pairwise
from pathlib import Path
from typing import Annotated

import torch
import torch.distributed as dist
import typer

from attn_gym.linear.context_parallel import ContextParallelPlan
from attn_gym.linear.context_parallel_deterministic import CanonicalTiling, tile_stream
from attn_gym.linear.kda import chunk_kda, context_parallel_kda
from attn_gym.linear.kda.context_parallel import context_parallel_kda_deterministic
from attn_gym.testing.kda import make_kda_test_inputs

ZIPF_FRACTIONS = (0.5, 0.25, 0.125, 0.0625)


def document_lengths(tokens: int, grid: int) -> tuple[int, ...]:
    """Zipf-like documents on a ``grid`` so every recipe can share fragment boundaries.

    Fractions are floored to the grid and the last document absorbs rounding. For small
    workloads, shrink the document grid (not the canonical tiles) to keep all five nonempty.
    Document boundaries are always valid tile boundaries, even for documents shorter than a tile.
    """
    grid = min(grid, max(64, tokens // 16 // 64 * 64))
    lengths = [int(tokens * f) // grid * grid for f in ZIPF_FRACTIONS]
    remainder = tokens - sum(lengths)
    if min(lengths) < grid or remainder < grid:
        raise ValueError(f"tokens={tokens} is too small for grid={grid}")
    return (*lengths, remainder)


def balanced_cut(cu: tuple[int, ...], tile: int, world: int) -> list[list[tuple[int, int]]]:
    """Contiguous fragments with boundaries on the canonical tile grid, balanced by tokens."""
    tiles = tile_stream(cu, tile)
    stops = [t.stop for t in tiles]
    fragments, start = [], 0
    for rank in range(world):
        target = cu[-1] * (rank + 1) // world
        stop = cu[-1] if rank == world - 1 else min(stops, key=lambda s: abs(s - target))
        fragments.append([(start, stop)] if stop > start else [])
        start = stop
    return fragments


class PhaseClock:
    """CUDA-event intervals named by their closing mark; read only after synchronization."""

    def __init__(self) -> None:
        self.marks: list[tuple[str, torch.cuda.Event]] = []

    def mark(self, name: str) -> None:
        """Record a phase boundary on the current stream."""
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self.marks.append((name, event))

    def elapsed(self) -> dict[str, float]:
        """Return local phase durations in milliseconds."""
        return {
            name: prev.elapsed_time(event) for (_, prev), (name, event) in pairwise(self.marks)
        }


def timed(
    step: Callable[[PhaseClock], object],
    *,
    iters: int,
    device: torch.device,
    distributed: bool = True,
) -> tuple[list[dict[str, float]], list[float], list[float]]:
    """Local phase samples and slowest-rank fwd/bwd ms, with a barrier before each step.

    Unlike single-device graph timing, these eager measurements include launch gaps and
    collective waits. ``distributed=False`` measures the single-GPU split benchmark.
    """
    phases, fwd, bwd = [], [], []
    for _ in range(iters):
        if distributed:
            dist.barrier()
        clock = PhaseClock()
        step(clock)
        torch.cuda.synchronize(device)
        local = clock.elapsed()
        phases.append(local)
        totals = torch.tensor(
            [
                sum(v for k, v in local.items() if k.startswith(prefix))
                for prefix in ("fwd", "bwd")
            ],
            device=device,
        )
        if distributed:
            dist.all_reduce(totals, op=dist.ReduceOp.MAX)
        f, b = totals.tolist()
        fwd.append(f)
        bwd.append(b)
    return phases, fwd, bwd


def clocked(fn: Callable[[], object], clock: PhaseClock, *, phase: str) -> None:
    """Time a public operation as a single phase."""
    clock.mark("start")
    fn()
    clock.mark(phase)


def forward(
    op: Callable[..., torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]],
    inputs: tuple[torch.Tensor, ...],
    grad: torch.Tensor,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
    """Run a KDA public API with fresh autograd leaves; ignore optional final states."""
    leaves = tuple(v.detach().requires_grad_() for v in inputs)
    result = op(*leaves)
    return (result[0] if isinstance(result, tuple) else result), leaves, grad


def main(
    tokens: Annotated[int, typer.Option(help="Total packed tokens.")] = 8192,
    heads: Annotated[int, typer.Option()] = 64,
    backend: Annotated[str, typer.Option(help="fused or mega staged backend.")] = "fused",
    tiles: Annotated[str, typer.Option(help="Comma-separated canonical tile sizes.")] = "64,256",
    rounds: Annotated[int, typer.Option()] = 5,
    iters: Annotated[int, typer.Option(help="Timed iterations per round per variant.")] = 10,
    out_path: Annotated[Path | None, typer.Option("--out")] = None,
) -> None:
    """Compare unsharded, standard CP, and canonical CP forward/backward costs."""
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    tile_sizes = [int(t) for t in tiles.split(",")]
    grid = max(tile_sizes)
    lengths = document_lengths(tokens, grid)
    cu = (0, *accumulate(lengths))
    options = {"backend": backend} if backend != "fused" else None
    torch.manual_seed(0)
    inputs = make_kda_test_inputs(
        tokens,
        heads=heads,
        seed=0,
        normalize_qk=True,
        sigmoid_beta=True,
        gate_scale=0.02,
        log_uniform_gate=False,
    )
    grad = torch.randn_like(inputs[2])
    cu_dev = torch.tensor(cu, dtype=torch.int32, device=device)
    fragments = balanced_cut(cu, grid, world)

    variants = {
        "unsharded": partial(
            forward,
            partial(chunk_kda, cu_seqlens=cu_dev, kernel_options=options, autotune=False),
            inputs,
            grad,
        )
    }

    plan = ContextParallelPlan.from_fragments(cu, fragments, rank)
    routing = plan.routing(device)
    cp_ids = plan.global_token_ids(device)
    cp_local = tuple(v[:, cp_ids].contiguous() for v in inputs)
    cp_grad = grad[:, cp_ids].contiguous()

    variants["cp"] = partial(
        forward,
        partial(
            context_parallel_kda,
            routing=routing,
            group=dist.group.WORLD,
            kernel_options=options,
            autotune=False,
        ),
        cp_local,
        cp_grad,
    )

    for tile in tile_sizes:
        tiling = CanonicalTiling.from_fragments(cu, fragments, rank, tile_size=tile)
        det_routing = tiling.routing(device)
        ids = tiling.global_token_ids(device)
        local = tuple(v[:, ids].contiguous() for v in inputs)
        local_grad = grad[:, ids].contiguous()

        variants[f"det-{tile}"] = partial(
            forward,
            partial(
                context_parallel_kda_deterministic,
                tiling=tiling,
                routing=det_routing,
                group=dist.group.WORLD,
                kernel_options=options,
            ),
            local,
            local_grad,
        )

    # Warm up (compiles) and keep one retained graph per variant for backward timing.
    retained = {}
    for name, fwd in variants.items():
        for _ in range(2):
            out, leaves, g = fwd()
            torch.autograd.grad(out, leaves, g)
        retained[name] = fwd()
    torch.cuda.synchronize(device)

    fwd_ms: dict[str, list[float]] = {name: [] for name in variants}
    bwd_ms: dict[str, list[float]] = {name: [] for name in variants}
    names = list(variants)
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for name in order:
            _, f, _ = timed(
                partial(clocked, variants[name], phase="fwd/run"), iters=iters, device=device
            )
            fwd_ms[name] += f
            out, leaves, g = retained[name]
            _, _, b = timed(
                partial(
                    clocked,
                    partial(torch.autograd.grad, out, leaves, g, retain_graph=True),
                    phase="bwd/run",
                ),
                iters=iters,
                device=device,
            )
            bwd_ms[name] += b

    if rank == 0:
        rows = []
        for name in names:
            rows.append(
                {
                    "variant": name,
                    "fwd_ms": statistics.median(fwd_ms[name]),
                    "bwd_ms": statistics.median(bwd_ms[name]),
                    "fwd_p05": sorted(fwd_ms[name])[len(fwd_ms[name]) // 20],
                    "fwd_p95": sorted(fwd_ms[name])[-1 - len(fwd_ms[name]) // 20],
                    "bwd_p05": sorted(bwd_ms[name])[len(bwd_ms[name]) // 20],
                    "bwd_p95": sorted(bwd_ms[name])[-1 - len(bwd_ms[name]) // 20],
                }
            )
        base = next(r for r in rows if r["variant"] == "cp")
        print(
            f"tokens={tokens} heads={heads} backend={backend} world={world} lengths={lengths}\n"
            f"fragments={fragments}\n"
            f"{'variant':<12}{'fwd ms':>10}{'bwd ms':>10}{'fwd/cp':>9}{'bwd/cp':>9}"
        )
        for row in rows:
            print(
                f"{row['variant']:<12}{row['fwd_ms']:>10.3f}{row['bwd_ms']:>10.3f}"
                f"{row['fwd_ms'] / base['fwd_ms']:>9.2f}{row['bwd_ms'] / base['bwd_ms']:>9.2f}"
            )
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(
                    {
                        "tokens": tokens,
                        "heads": heads,
                        "backend": backend,
                        "world": world,
                        "lengths": lengths,
                        "fragments": fragments,
                        "rounds": rounds,
                        "iters": iters,
                        "device": torch.cuda.get_device_name(device),
                        "torch": torch.__version__,
                        "rows": rows,
                        "samples": {"fwd_ms": fwd_ms, "bwd_ms": bwd_ms},
                    },
                    indent=2,
                )
                + "\n"
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    typer.run(main)
