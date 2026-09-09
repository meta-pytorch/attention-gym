"""Fragment-plan study for CP-degree-invariant KDA: ownership, tile size, and Mega splitting.

    torchrun --standalone --nproc-per-node=2 benchmarks/kda_cp_fragment_study.py ownership
    torchrun --standalone --nproc-per-node=2 benchmarks/kda_cp_fragment_study.py tile-sweep
    torchrun --standalone --nproc-per-node=2 benchmarks/kda_cp_fragment_study.py crossover
    python benchmarks/kda_cp_fragment_study.py split      # single process

Canonical variants mirror ``_CanonicalContextParallel`` using staged handles and CUDA events
between phases, including the single-tile-document skip. Each is checked bitwise against the
public autograd API before timing. Standard CP and unsharded baselines use that API directly;
``@module`` rows expose the canonical autograd overhead. Rounds alternate variant order; tables
report slowest-rank forward/backward medians and local phase medians, in milliseconds.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from collections.abc import Callable, Sequence
from functools import partial
from itertools import accumulate, pairwise
from pathlib import Path
from typing import Annotated

import torch
import torch.distributed as dist
import typer
from kda_cp_deterministic import (
    PhaseClock,
    balanced_cut,
    clocked,
    document_lengths,
    forward,
    timed,
)

from attn_gym.linear._delta_rule.triton.canonical_scan import canonical_scan_entries, fold_ranges
from attn_gym.linear.context_parallel import ContextParallelPlan, StagedOp
from attn_gym.linear.context_parallel_deterministic import (
    CanonicalRouting,
    CanonicalTiling,
    all_gather_block_maps,
    tile_stream,
)
from attn_gym.linear.kda import chunk_kda, context_parallel_kda
from attn_gym.linear.kda.context_parallel import _kda_stages, context_parallel_kda_deterministic
from attn_gym.testing.kda import make_kda_test_inputs

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

Fragments = list[list[tuple[int, int]]]
GRAD_NAMES = ("out", "dq", "dk", "dv", "dgate", "dbeta")
FWD_PHASES = ("fwd/prepare", "fwd/summary", "fwd/fold", "fwd/gather", "fwd/scan", "fwd/run")
BWD_PHASES = ("bwd/prepare", "bwd/summary", "bwd/fold", "bwd/gather", "bwd/scan", "bwd/run")


def zipf_many_lengths(tokens: int) -> tuple[int, ...]:
    """One long document plus six tiers of ``2**j`` documents of ``tokens / (8 * 2**j)`` each.

    32k gives 8192 + 2x2048 + 4x1024 + 8x512 + 16x256 + 32x128 + 64x64: 127 documents, most
    shorter than any practical tile.
    """
    if tokens <= 0 or tokens % 8192:
        raise ValueError(
            "zipf-many needs a positive multiple of 8192 tokens (shortest tier is 16)"
        )
    lengths = [tokens // 4]
    for j in range(1, 7):
        lengths += [tokens // (8 * 2**j)] * 2**j
    assert sum(lengths) == tokens
    return tuple(lengths)


def ownership_lengths(tokens: int) -> tuple[int, ...]:
    """Nine ragged documents whose whole-document LPT packing differs from a contiguous cut."""
    if tokens % 32768:
        raise ValueError("ownership set needs a multiple of 32768 tokens")
    unit = tokens // 32
    return tuple(unit * n for n in (12, 6, 4, 3, 2, 2, 1, 1, 1))


def cyclic_fragments(cu: Sequence[int], tile: int, world: int) -> Fragments:
    """Assign successive tiles round-robin."""
    tiles = tile_stream(cu, tile)
    return [
        [(t.start, t.stop) for i, t in enumerate(tiles) if i % world == r] for r in range(world)
    ]


def halves_fragments(cu: Sequence[int], tile: int, world: int) -> Fragments:
    """Every document cut into ``world`` tile-aligned pieces; odd leftovers alternate ranks."""
    out: Fragments = [[] for _ in range(world)]
    for doc, (start, stop) in enumerate(pairwise(cu)):
        n = math.ceil((stop - start) / tile)
        base, extra = divmod(n, world)
        counts = [base + (1 if (r + doc) % world < extra else 0) for r in range(world)]
        edges = [start + c * tile for c in accumulate(counts, initial=0)]
        for r, (a, b) in enumerate(pairwise(edges)):
            if b > a:
                out[r].append((a, min(b, stop)))
    return out


def lpt_document_fragments(cu: Sequence[int], world: int) -> Fragments:
    """Whole documents, longest first, each onto the least-loaded rank."""
    loads = [0] * world
    out: Fragments = [[] for _ in range(world)]
    for start, stop in sorted(pairwise(cu), key=lambda d: d[0] - d[1]):
        r = loads.index(min(loads))
        out[r].append((start, stop))
        loads[r] += stop - start
    return [sorted(f) for f in out]


def skew_fragments(cu: Sequence[int], tile: int, world: int, share: float) -> Fragments:
    """Rank 0 owns a contiguous ``share`` of the tiles; the others split the rest evenly."""
    tiles = tile_stream(cu, tile)
    first = round(len(tiles) * share)
    cuts = [
        0,
        first,
        *(first + (len(tiles) - first) * (r + 1) // (world - 1) for r in range(world - 1)),
    ]
    return [[(tiles[a].start, tiles[b - 1].stop)] if b > a else [] for a, b in pairwise(cuts)]


def layout_fragments(name: str, cu: Sequence[int], tile: int, world: int) -> Fragments:
    """Build the selected ownership table without changing the canonical tile grid."""
    match name:
        case "contiguous":
            return balanced_cut(tuple(cu), tile, world)
        case "halves":
            return halves_fragments(cu, tile, world)
        case "cyclic":
            return cyclic_fragments(cu, tile, world)
        case "lpt-docs":
            return lpt_document_fragments(cu, world)
        case "all-on-0":
            return [[(0, cu[-1])]] + [[] for _ in range(world - 1)]
        case _ if name.startswith("skew-"):
            return skew_fragments(cu, tile, world, int(name[5:]) / 100)
        case _:
            raise ValueError(f"unknown layout {name}")


def bits_differ(actual: torch.Tensor, expected: torch.Tensor) -> int:
    """Count unequal storage representations, including signed zeros."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, (
        actual.shape,
        expected.shape,
    )
    view = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    return int((actual.contiguous().view(view) != expected.contiguous().view(view)).sum())


def canonical_states(
    summary_maps: torch.Tensor,
    tiling: CanonicalTiling,
    routing: CanonicalRouting,
    like: torch.Tensor,
    group: dist.ProcessGroup,
    clock: PhaseClock,
    *,
    reverse: bool,
) -> torch.Tensor:
    """``canonical_entries`` with a clock mark after the block fold, the gather, and the scan.

    Same statements as the module function, in the same order; the marks are the only addition
    (checked bitwise against the module's autograd path in ``Study.add_canonical``).
    """
    prefix = "bwd" if reverse else "fwd"
    if summary_maps.shape[0] == 0:
        clock.mark(f"{prefix}/fold")
        all_gather_block_maps(summary_maps, tiling, routing, like, group)
        clock.mark(f"{prefix}/gather")
        states = like.new_zeros(
            (len(tiling.mine), like.shape[1], like.shape[2] - like.shape[3], like.shape[3])
        )
        clock.mark(f"{prefix}/scan")
        return states
    if tiling.scan_block == 1:
        clock.mark(f"{prefix}/fold")
        gathered = all_gather_block_maps(summary_maps, tiling, routing, like, group)
        clock.mark(f"{prefix}/gather")
        entries = canonical_scan_entries(
            gathered, routing.block_document_offsets, reverse=reverse
        )[routing.my_blocks]
    else:
        block_maps = fold_ranges(summary_maps, routing.my_block_offsets, reverse=reverse)
        clock.mark(f"{prefix}/fold")
        gathered = all_gather_block_maps(block_maps, tiling, routing, like, group)
        clock.mark(f"{prefix}/gather")
        block_entries = canonical_scan_entries(
            gathered, routing.block_document_offsets, reverse=reverse
        )
        entries = canonical_scan_entries(
            summary_maps,
            routing.my_block_offsets,
            reverse=reverse,
            initial=block_entries[routing.my_blocks],
        )
    if routing.summarized_positions is not None:
        full = entries.new_zeros((len(tiling.mine), *entries.shape[1:]))
        full[routing.summarized_positions] = entries
        entries = full
    clock.mark(f"{prefix}/scan")
    return entries


def canonical_step(
    stages: StagedOp,
    local: tuple[torch.Tensor, ...],
    grad: torch.Tensor,
    tiling: CanonicalTiling,
    routing: CanonicalRouting,
    group: dist.ProcessGroup,
    clock: PhaseClock,
) -> tuple[torch.Tensor, ...]:
    """``context_parallel_chunk_deterministic`` fwd+bwd by hand, one clock mark per phase.

    Keeps the module's arithmetic and collective order; omits autograd tape bookkeeping.
    """
    q, k, v, gate, beta = local
    heads, dim, value_dim = v.shape[2], q.shape[3], v.shape[3]
    n = len(tiling.mine)
    like = q.new_empty((0, heads, value_dim + dim, dim), dtype=torch.float32)
    summarized = routing.summary_bounds.shape[0]
    prepared = backward = None
    clock.mark("start")
    if n:
        prepared = stages.prepare(q, k, v, gate, beta, cu_seqlens=routing.cu_seqlens)
    clock.mark("fwd/prepare")
    maps = like
    if n and summarized:
        maps = prepared.state_summaries(routing.summary_bounds, deterministic_work=True)
    clock.mark("fwd/summary")
    entry = canonical_states(maps, tiling, routing, like, group, clock, reverse=False)
    output = prepared.run(entry, output_final_state=False)[0] if n else v[:, :0]
    clock.mark("fwd/run")

    if n:
        backward = stages.prepare_backward(prepared.saved, grad, entry, scale=prepared.scale)
    clock.mark("bwd/prepare")
    rmaps = like
    if n and summarized:
        rmaps = backward.state_grad_summaries(routing.summary_bounds, deterministic_work=True)
    clock.mark("bwd/summary")
    exit_cotangent = canonical_states(rmaps, tiling, routing, like, group, clock, reverse=True)
    grads = backward.run(exit_cotangent)[:5] if n else tuple(t[:, :0] for t in local)
    clock.mark("bwd/run")
    return (output, *grads)


def autograd_step(
    fwd: Callable[[], tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]],
    clock: PhaseClock,
) -> tuple[torch.Tensor, ...]:
    """Public-API fwd+bwd: two intervals, ``fwd/run`` and ``bwd/run``."""
    clock.mark("start")
    out, leaves, grad = fwd()
    clock.mark("fwd/run")
    grads = torch.autograd.grad(out, leaves, grad)
    clock.mark("bwd/run")
    return (out.detach(), *grads)


class Study:
    """Inputs, group, and the variant table of one torchrun session."""

    def __init__(self, tokens: int, heads: int, backend: str, lengths: Sequence[int]) -> None:
        """Initialize a two-or-more-rank study with identical seeded global inputs."""
        self.rank, self.world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(self.device)
        dist.init_process_group("nccl", device_id=self.device)
        self.group = dist.group.WORLD
        self.tokens, self.heads, self.backend = tokens, heads, backend
        self.options = {"backend": backend} if backend != "fused" else None
        self.stages = _kda_stages(None, False, False, self.options)
        self.lengths = tuple(lengths)
        self.cu = (0, *accumulate(self.lengths))
        torch.manual_seed(0)
        self.inputs = make_kda_test_inputs(
            tokens,
            heads=heads,
            seed=0,
            normalize_qk=True,
            sigmoid_beta=True,
            gate_scale=0.02,
            log_uniform_gate=False,
        )
        self.grad = torch.randn_like(self.inputs[2])
        self.variants: dict[str, Callable[[PhaseClock], tuple[torch.Tensor, ...]]] = {}
        self.checks: dict[str, dict[str, int]] = {}

    def log(self, *args) -> None:
        """Print layout information once across ranks."""
        if self.rank == 0:
            print(*args, flush=True)

    def gather_local(self, ids: torch.Tensor) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Copy inputs and output cotangents into the rank-local span order."""
        return tuple(v[:, ids].contiguous() for v in self.inputs), self.grad[:, ids].contiguous()

    def add_unsharded(self) -> None:
        """Register the CP=1 reference cost on every rank."""
        op = partial(
            chunk_kda,
            cu_seqlens=torch.tensor(self.cu, dtype=torch.int32, device=self.device),
            kernel_options=self.options,
            autotune=False,
        )
        self.variants["unsharded"] = partial(
            autograd_step, partial(forward, op, self.inputs, self.grad)
        )

    def add_standard(self, name: str, fragments: Fragments) -> None:
        """Register the public standard recipe, which does not support empty spans."""
        if any(not f for f in fragments):
            return
        plan = ContextParallelPlan.from_fragments(self.cu, fragments, self.rank)
        local, grad = self.gather_local(plan.global_token_ids(self.device))
        op = partial(
            context_parallel_kda,
            routing=plan.routing(self.device),
            group=self.group,
            kernel_options=self.options,
            autotune=False,
        )
        self.variants[name] = partial(autograd_step, partial(forward, op, local, grad))

    def add_canonical(
        self,
        name: str,
        fragments: Fragments,
        tile: int,
        *,
        scan_block: int = 1,
        module: bool = False,
    ) -> None:
        """Register the hand-run canonical recipe (and, with ``module``, the autograd path)."""
        tiling = CanonicalTiling.from_fragments(
            self.cu, fragments, self.rank, tile_size=tile, scan_block=scan_block
        )
        routing = tiling.routing(self.device)
        ids = tiling.global_token_ids(self.device)
        local, grad = self.gather_local(ids)
        self.variants[name] = partial(
            canonical_step, self.stages, local, grad, tiling, routing, self.group
        )

        op = partial(
            context_parallel_kda_deterministic,
            tiling=tiling,
            routing=routing,
            group=self.group,
            kernel_options=self.options,
        )
        fwd = partial(forward, op, local, grad)
        expected = autograd_step(fwd, PhaseClock())
        actual = self.variants[name](PhaseClock())
        self.checks[name] = {
            k: bits_differ(a, e) for k, a, e in zip(GRAD_NAMES, actual, expected, strict=True)
        }
        if module:
            self.variants[f"{name}@module"] = partial(autograd_step, fwd)

    def run(self, rounds: int, iters: int, out_path: Path | None, header: str) -> None:
        """Validate all ranks, warm up, then time interleaved variants and emit JSON."""
        checks = [None] * self.world
        dist.all_gather_object(checks, self.checks)
        merged_checks = {
            n: {k: sum(c[n][k] for c in checks) for k in GRAD_NAMES} for n in self.checks
        }
        assert not any(v for check in merged_checks.values() for v in check.values()), (
            merged_checks
        )
        names = list(self.variants)
        for name in names:  # warm-up / compile
            for _ in range(2):
                self.variants[name](PhaseClock())
        torch.cuda.synchronize(self.device)
        phases = {n: [] for n in names}
        fwd = {n: [] for n in names}
        bwd = {n: [] for n in names}
        for r in range(rounds):
            for name in names if r % 2 == 0 else names[::-1]:
                p, f, b = timed(self.variants[name], iters=iters, device=self.device)
                phases[name] += p
                fwd[name] += f
                bwd[name] += b
        # Every rank's phase medians to rank 0.
        local_medians = {
            n: {k: statistics.median(p[k] for p in phases[n]) for k in phases[n][0]} for n in names
        }
        all_medians = [None] * self.world
        dist.all_gather_object(all_medians, local_medians)
        rows = []
        if self.rank == 0:
            for name in names:
                rows.append(
                    {
                        "variant": name,
                        "fwd_ms": statistics.median(fwd[name]),
                        "bwd_ms": statistics.median(bwd[name]),
                        "fwd_p95": sorted(fwd[name])[-1 - len(fwd[name]) // 20],
                        "bwd_p95": sorted(bwd[name])[-1 - len(bwd[name]) // 20],
                        "ranks": [m[name] for m in all_medians],
                        "check_bits_differ": merged_checks.get(name),
                    }
                )
            self.print_table(rows, header)
            if out_path:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(
                        {
                            "header": header,
                            "tokens": self.tokens,
                            "heads": self.heads,
                            "backend": self.backend,
                            "world": self.world,
                            "lengths": self.lengths,
                            "rounds": rounds,
                            "iters": iters,
                            "device": torch.cuda.get_device_name(self.device),
                            "torch": torch.__version__,
                            "rows": rows,
                            "samples": {"fwd_ms": fwd, "bwd_ms": bwd},
                        },
                        indent=1,
                    )
                    + "\n"
                )
        dist.barrier()

    def print_table(self, rows: list[dict], header: str) -> None:
        """Report slowest-rank totals and each rank's local phase medians (ms)."""
        print(f"\n{header}\ntokens={self.tokens} heads={self.heads} backend={self.backend}")
        print(f"{'variant':<22}{'fwd ms':>9}{'bwd ms':>9}{'fwd p95':>9}{'bwd p95':>9}  checks")
        for row in rows:
            chk = row["check_bits_differ"]
            chk_s = "-" if chk is None else ("bitwise" if not any(chk.values()) else str(chk))
            print(
                f"{row['variant']:<22}{row['fwd_ms']:>9.3f}{row['bwd_ms']:>9.3f}"
                f"{row['fwd_p95']:>9.3f}{row['bwd_p95']:>9.3f}  {chk_s}"
            )
        print("\nper-rank phase medians (ms):")
        head = f"{'variant':<22}{'rank':>5}" + "".join(
            f"{p.split('/')[1]:>9}" for p in FWD_PHASES + BWD_PHASES
        )
        print(head + f"{'fwd':>9}{'bwd':>9}")
        for row in rows:
            for r, m in enumerate(row["ranks"]):
                cells = "".join(f"{m.get(p, 0.0):>9.3f}" for p in FWD_PHASES + BWD_PHASES)
                f = sum(v for k, v in m.items() if k.startswith("fwd"))
                b = sum(v for k, v in m.items() if k.startswith("bwd"))
                print(f"{row['variant']:<22}{r:>5}{cells}{f:>9.3f}{b:>9.3f}")


OUT_DIR = Path("agent_space/friendly_fragments")


@app.command()
def ownership(
    tokens: Annotated[int, typer.Option()] = 32768,
    heads: Annotated[int, typer.Option()] = 64,
    backend: Annotated[str, typer.Option()] = "mega",
    tile: Annotated[int, typer.Option()] = 1024,
    layouts: Annotated[str, typer.Option()] = (
        "contiguous,halves,cyclic,lpt-docs,skew-62,skew-75,all-on-0"
    ),
    scan_blocks: Annotated[str, typer.Option(help="Tiles per exchanged block.")] = "1,4",
    rounds: Annotated[int, typer.Option()] = 5,
    iters: Annotated[int, typer.Option()] = 10,
    out_path: Annotated[Path | None, typer.Option("--out")] = None,
) -> None:
    """Q1: does ownership shape matter at a fixed tile size? Standard vs canonical per layout."""
    study = Study(tokens, heads, backend, ownership_lengths(tokens))
    study.add_unsharded()
    for name in layouts.split(","):
        fragments = layout_fragments(name, study.cu, tile, study.world)
        study.log(f"{name}: {fragments}")
        study.add_standard(f"cp/{name}", fragments)
        for block in (int(b) for b in scan_blocks.split(",")):
            suffix = f"/b{block}" if block > 1 else ""
            try:
                study.add_canonical(
                    f"det/{name}{suffix}", fragments, tile, scan_block=block, module=block == 1
                )
            except ValueError as error:  # fragments cut a scan block
                study.log(f"det/{name}{suffix}: skipped ({error})")
    study.run(
        rounds,
        iters,
        out_path or OUT_DIR / f"ownership_{backend}_{tokens}_t{tile}.json",
        f"Q1 ownership, tile={tile}, docs={study.lengths}",
    )
    dist.destroy_process_group()


@app.command()
def tile_sweep(
    tokens: Annotated[int, typer.Option()] = 32768,
    heads: Annotated[int, typer.Option()] = 64,
    backend: Annotated[str, typer.Option()] = "mega",
    tiles: Annotated[str, typer.Option()] = "256,512,1024,2048,4096,8192",
    docs: Annotated[str, typer.Option(help="zipf-many or zipf5")] = "zipf-many",
    layout: Annotated[str, typer.Option()] = "contiguous",
    rounds: Annotated[int, typer.Option()] = 5,
    iters: Annotated[int, typer.Option()] = 10,
    out_path: Annotated[Path | None, typer.Option("--out")] = None,
) -> None:
    """Q2: tile size against a Zipf-like document mix (many single-tile documents)."""
    tile_sizes = [int(t) for t in tiles.split(",")]
    lengths = (
        zipf_many_lengths(tokens)
        if docs == "zipf-many"
        else document_lengths(tokens, max(tile_sizes))
    )
    study = Study(tokens, heads, backend, lengths)
    grid = max(tile_sizes)
    fragments = layout_fragments(layout, study.cu, grid, study.world)
    study.log(f"fragments={fragments}")
    study.add_unsharded()
    study.add_standard("cp", fragments)
    for tile in tile_sizes:
        tiles_all = tile_stream(study.cu, tile)
        single = sum(length <= tile for length in lengths)
        study.log(f"tile {tile}: {len(tiles_all)} tiles, {single} in single-tile documents")
        study.add_canonical(f"det-{tile}", fragments, tile, module=True)
    study.run(
        rounds,
        iters,
        out_path or OUT_DIR / f"tile_sweep_{backend}_{docs}_{tokens}.json",
        f"Q2 tile sweep, docs={docs}, layout={layout}",
    )
    dist.destroy_process_group()


@app.command()
def crossover(
    tokens: Annotated[int, typer.Option()] = 32768,
    heads: Annotated[int, typer.Option()] = 64,
    backend: Annotated[str, typer.Option()] = "mega",
    doc_len: Annotated[int, typer.Option(help="Every document has this length.")] = 512,
    tile: Annotated[int, typer.Option()] = 1024,
    rounds: Annotated[int, typer.Option()] = 5,
    iters: Annotated[int, typer.Option()] = 10,
    out_path: Annotated[Path | None, typer.Option("--out")] = None,
) -> None:
    """Q3: many equal short documents, none crossing ranks: when is canonical CP cheaper?"""
    if tokens % doc_len or doc_len > tile:
        raise typer.BadParameter("doc_len must divide tokens and not exceed tile")
    study = Study(tokens, heads, backend, [doc_len] * (tokens // doc_len))
    fragments = balanced_cut(study.cu, tile, study.world)
    study.log(f"fragments={fragments}")
    study.add_unsharded()
    study.add_standard("cp", fragments)
    study.add_canonical(f"det-{tile}", fragments, tile, module=True)
    study.run(
        rounds,
        iters,
        out_path or OUT_DIR / f"crossover_{backend}_{tokens}_d{doc_len}_t{tile}.json",
        f"Q3 crossover, {tokens // doc_len} documents of {doc_len}, tile={tile}",
    )
    dist.destroy_process_group()


@app.command()
def split(
    tokens: Annotated[int, typer.Option()] = 32768,
    heads: Annotated[str, typer.Option()] = "64,16,8",
    gates: Annotated[str, typer.Option(help="mild (bench gate) and/or strong.")] = "mild,strong",
    tile: Annotated[int, typer.Option(help="Packing whose split table is also counted.")] = 1024,
    rounds: Annotated[int, typer.Option()] = 5,
    iters: Annotated[int, typer.Option()] = 20,
    out_path: Annotated[Path | None, typer.Option("--out")] = None,
) -> None:
    """Q4: Mega unsplit vs forgetting-horizon split forward on one dense stream (single process)."""
    from attn_gym.linear._delta_rule.mega.schedule import prepare_mega_schedule

    device = torch.device("cuda", torch.cuda.current_device())
    results = []
    for h in (int(x) for x in heads.split(",")):
        for gate_name in gates.split(","):
            strong = gate_name == "strong"
            inputs = make_kda_test_inputs(
                tokens,
                heads=h,
                seed=0,
                normalize_qk=True,
                sigmoid_beta=True,
                gate_scale=5.0 if strong else 0.02,
                log_uniform_gate=strong,
            )
            variants = {
                "unsplit": {"backend": "mega"},
                "split": {"backend": "mega", "split_forward": True},
            }
            outs = {}
            with torch.no_grad():
                for name, opts in variants.items():
                    for _ in range(3):
                        outs[name] = chunk_kda(*inputs, kernel_options=opts, autotune=False)[0]
                torch.cuda.synchronize()
                samples = {n: [] for n in variants}
                for r in range(rounds):
                    order = list(variants) if r % 2 == 0 else list(variants)[::-1]
                    for name in order:
                        op = partial(
                            chunk_kda, *inputs, kernel_options=variants[name], autotune=False
                        )
                        _, fwd, _ = timed(
                            partial(clocked, op, phase="fwd/run"),
                            iters=iters,
                            device=device,
                            distributed=False,
                        )
                        samples[name] += fwd
            gate, cu_dense = inputs[3], torch.tensor([0, tokens], dtype=torch.int32, device=device)
            cu_tiles = torch.arange(0, tokens + 1, tile, dtype=torch.int32, device=device)
            stream = torch.cuda.current_stream().cuda_stream
            items = {}
            for label, cu in (("dense", cu_dense), (f"tiles-{tile}", cu_tiles)):
                sched = prepare_mega_schedule(
                    gate, cu, tile_tokens=16, counter_count=2, split=True, stream=stream
                )
                torch.cuda.synchronize()
                items[label] = (int(sched.work_count.item()), (cu.numel() - 1) * h)
            row = {
                "heads": h,
                "gate": gate_name,
                "unsplit_ms": statistics.median(samples["unsplit"]),
                "split_ms": statistics.median(samples["split"]),
                "rel_l2": (
                    (outs["split"].float() - outs["unsplit"].float()).norm()
                    / outs["unsplit"].float().norm().clamp_min(1e-30)
                ).item(),
                "max_abs": (outs["split"].float() - outs["unsplit"].float()).abs().max().item(),
                "bits_differ": bits_differ(outs["split"], outs["unsplit"]),
                "work_items": items,
            }
            results.append(row)
            print(
                f"H={h:<3} gate={gate_name:<6} unsplit {row['unsplit_ms']:.3f} ms  "
                f"split {row['split_ms']:.3f} ms  speedup {row['unsplit_ms'] / row['split_ms']:.2f}x"
                f"  rel-L2 {row['rel_l2']:.2e} max|d| {row['max_abs']:.2e}"
                f"  bits differ {row['bits_differ']}  items {items}",
                flush=True,
            )
    path = out_path or OUT_DIR / f"split_{tokens}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tokens": tokens,
                "tile": tile,
                "rounds": rounds,
                "iters": iters,
                "device": torch.cuda.get_device_name(device),
                "torch": torch.__version__,
                "rows": results,
            },
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    app()
