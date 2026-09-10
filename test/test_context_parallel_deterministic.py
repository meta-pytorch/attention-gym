"""CP-degree-invariant KDA: canonical tiles give bitwise-equal results for any rank ownership.

The reference for every case is a fresh single-process canonical run (CP=1) over the same tiles.
Two-rank cases use NCCL when two GPUs exist, otherwise Gloo between two processes on one GPU (the
transport carries FP32 maps bit-for-bit either way). Ownership tables cover contiguous, cyclic,
uneven, and empty-rank layouts on ragged packed documents.
"""

from __future__ import annotations

import math
import os
import socket
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from itertools import accumulate, pairwise
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("cutlass")

import attn_gym.linear.context_parallel_deterministic as det
from attn_gym.linear._delta_rule.triton.canonical_scan import canonical_scan_entries
from attn_gym.linear.context_parallel import compose_summaries
from attn_gym.linear.context_parallel_deterministic import (
    CanonicalTiling,
    canonical_entries,
    tile_stream,
)
from attn_gym.linear.kda.context_parallel import _kda_stages, context_parallel_kda_deterministic
from attn_gym.linear.types import KernelOptions
from attn_gym.testing.kda import kda_reference, make_kda_test_inputs

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.xdist_group("two-gpu"),
]

HEADS = 4
RAGGED = (17, 65, 129, 257, 513)
ELEPHANT = (800, 31, 33, 31, 33, 31, 33)
NAMES = ("output", "dq", "dk", "dv", "dgate", "dbeta")


# ---------------------------------------------------------------- planner (CPU)


def test_tiles_align_to_document_starts_and_keep_tails():
    cu = (0, *accumulate(RAGGED))
    tiles = tile_stream(cu, 64)
    assert [t.length for t in tiles if t.document == 0] == [17]
    assert [t.length for t in tiles if t.document == 1] == [64, 1]
    assert [t.length for t in tiles if t.document == 4] == [64] * 8 + [1]
    assert all((t.start - cu[t.document]) % 64 == 0 for t in tiles)
    assert tiles[-1].stop == cu[-1]


def test_fragments_must_fall_on_tile_boundaries():
    cu = (0, *accumulate(RAGGED))
    CanonicalTiling.from_fragments(
        cu, [[(0, 17), (82, 211)], [(17, 82), (211, 981)]], 0, tile_size=64
    )
    with pytest.raises(ValueError, match="tile boundaries"):
        CanonicalTiling.from_fragments(cu, [[(0, 100)], [(100, 981)]], 0, tile_size=64)
    with pytest.raises(ValueError, match="multiple of 64"):
        CanonicalTiling.from_fragments(cu, [[(0, 981)], []], 0, tile_size=48)
    with pytest.raises(ValueError, match="exactly once"):
        CanonicalTiling.from_fragments(cu, [[(0, 17)], [(82, 981)]], 0, tile_size=64)
    with pytest.raises(ValueError, match="reversed"):
        CanonicalTiling.from_fragments(cu, [[(17, 0)], [(17, 981)]], 0, tile_size=64)
    with pytest.raises(ValueError, match="outside"):
        CanonicalTiling.from_fragments(cu, [[(0, 981)], []], 2, tile_size=64)
    tiling = CanonicalTiling.from_fragments(cu, [[(0, 981)]], 0, tile_size=64)
    # The 17-token document fits one tile and is not summarized.
    single = {i for i, t in enumerate(tiling.tiles) if t.document == 0}
    assert set(tiling.summarized) == set(range(len(tiling.tiles))) - single


def test_non_contiguous_ownership_keeps_span_order_consistent():
    """A rank owning tiles from two places lists them in fragment order; ids follow that order."""
    cu = (0, *accumulate(RAGGED))
    fragments = [[(0, 17), (211, 981)], [(17, 211)]]
    tiling = CanonicalTiling.from_fragments(cu, fragments, 0, tile_size=64)
    ids = tiling.global_token_ids("cpu").tolist()
    assert ids == list(range(17)) + list(range(211, 981))
    offsets = tiling.span_offsets()
    assert offsets[0] == 0 and offsets[-1] == len(ids)
    assert [tiling.tiles[i].length for i in tiling.mine] == [b - a for a, b in pairwise(offsets)]


def _serial_entries(
    leaves: torch.Tensor, offsets: Sequence[int], *, reverse: bool
) -> torch.Tensor:
    """Exclusive serial prefix bias, restarting at each document in either direction."""
    value_dim = leaves.shape[2] - leaves.shape[3]
    entries = torch.zeros(
        leaves.shape[0], leaves.shape[1], value_dim, leaves.shape[3], device=leaves.device
    )
    for start, stop in pairwise(offsets):
        indices = range(stop - 1, start - 1, -1) if reverse else range(start, stop)
        prefix = leaves[indices[0]]
        for i in indices[1:]:
            entries[i] = prefix[:, :value_dim, :]
            prefix = compose_summaries(prefix, leaves[i])
    return entries


def _leaves(n: int, seed: int) -> torch.Tensor:
    """Random affine maps with near-identity transitions to keep long prefixes nontrivial."""
    torch.manual_seed(seed)
    leaves = torch.randn(n, 2, 256, 128, device="cuda")
    leaves[:, :, 128:, :] = torch.eye(128, device="cuda") + 0.01 * torch.randn(
        n, 2, 128, 128, device="cuda"
    )
    return leaves


def _single_rank_tiling(lengths: Sequence[int], tile_size: int = 64) -> CanonicalTiling:
    """Tile packed document lengths with every token owned by one rank."""
    cu = (0, *accumulate(lengths))
    return CanonicalTiling.from_fragments(cu, [[(0, cu[-1])]], 0, tile_size=tile_size)


@pytest.mark.parametrize("tile_counts", [(9,), (3, 1, 3), (1, 1, 1, 1), (1,)])
def test_fused_scan_matches_serial_composition_per_document(tile_counts):
    """The fused scan restarts at documents and equals serial composition to fp32 accuracy."""
    leaves = _leaves(sum(tile_counts), 0)
    offsets = [0, *accumulate(tile_counts)]
    device_offsets = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    for reverse in (False, True):
        torch.testing.assert_close(
            canonical_scan_entries(leaves, device_offsets, reverse=reverse),
            _serial_entries(leaves, offsets, reverse=reverse),
            atol=1e-4,
            rtol=1e-5,
        )


def test_entries_of_a_document_are_independent_of_its_neighbours():
    """A document's entry states are bitwise the same wherever it sits in the packed stream."""
    document = _leaves(5, 1)
    reference: dict[bool, torch.Tensor] = {}
    for before, after in ((0, 0), (1, 0), (3, 2), (7, 5)):
        others = _leaves(before + after, 2 + before)
        leaves = torch.cat((others[:before], document, others[before:]))
        tiling = _single_rank_tiling([64 * count for count in (before, 5, after) if count])
        summarized = torch.tensor(tiling.summarized, device="cuda")
        for reverse in (False, True):
            entries = canonical_entries(
                leaves[summarized],
                tiling,
                tiling.routing("cuda"),
                leaves[:1],
                None,
                reverse=reverse,
            )
            entries = entries[before : before + 5]
            reference.setdefault(reverse, entries)
            assert torch.equal(entries, reference[reverse]), (before, after, reverse)


# ---------------------------------------------------------------- transports


def _cpu_all_gather_into_tensor(
    gathered: torch.Tensor, packet: torch.Tensor, group: dist.ProcessGroup | None = None
) -> None:
    """Gloo path: gather CPU copies, then place them into the CUDA output buffer."""
    staging = torch.empty(gathered.shape, dtype=gathered.dtype)
    dist.all_gather_into_tensor(staging, packet.cpu(), group=group)
    gathered.copy_(staging)


@contextmanager
def _process_group(cp_rank: int, world: int, port: int) -> Iterator[torch.device]:
    """NCCL on one GPU per rank, or Gloo with the module's collective staged through the CPU."""
    shared = torch.cuda.device_count() < world
    device = torch.device("cuda", 0 if shared else cp_rank)
    torch.cuda.set_device(device)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group(
        "gloo" if shared else "nccl",
        rank=cp_rank,
        world_size=world,
        device_id=None if shared else device,
    )
    try:
        if shared:
            with patch.object(det, "_all_gather_into_tensor", _cpu_all_gather_into_tensor):
                yield device
        else:
            yield device
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------- canonical CP1 reference


def _inputs(
    lengths: Sequence[int], gate: str, seed: int
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Seeded KDA inputs and output cotangent on the current rank device."""
    torch.manual_seed(seed)
    values = make_kda_test_inputs(
        sum(lengths),
        heads=HEADS,
        seed=seed,
        normalize_qk=True,
        sigmoid_beta=True,
        gate_scale=math.log(2) if gate == "strong" else 0.02,
        log_uniform_gate=gate == "strong",
        gate_value=0.0 if gate == "zero" else None,
    )
    return values, torch.randn_like(values[2])


@torch.no_grad()
def _canonical_cp1(
    inputs: tuple[torch.Tensor, ...],
    grad: torch.Tensor,
    lengths: Sequence[int],
    tile_size: int,
    kernel_options: KernelOptions | None,
) -> tuple[torch.Tensor, ...]:
    """Independent per-tile staged calls with the production scan: the bitwise CP contract."""
    tiling = _single_rank_tiling(lengths, tile_size)
    tiles = tiling.tiles
    stages = _kda_stages(None, False, False, kernel_options)
    device = inputs[0].device

    bounds = [torch.tensor([[0, t.length]], dtype=torch.int32, device=device) for t in tiles]

    prepared = [
        stages.prepare(*(v[:, t.start : t.stop] for v in inputs), cu_seqlens=None) for t in tiles
    ]
    # Single-tile documents are not summarized (NOTE [Single-Tile Documents]).
    summarized = tiling.summarized
    maps = torch.cat(
        [prepared[i].state_summaries(bounds[i], deterministic_work=True) for i in summarized]
    )
    routing = tiling.routing(device)
    entries = canonical_entries(maps, tiling, routing, maps[:1], None, reverse=False)
    outputs, backwards = [], []
    for i, (p, t) in enumerate(zip(prepared, tiles, strict=True)):
        outputs.append(p.run(entries[i : i + 1], output_final_state=False)[0])
        backwards.append(
            stages.prepare_backward(
                p.saved, grad[:, t.start : t.stop], entries[i : i + 1], scale=p.scale
            )
        )
    rmaps = torch.cat(
        [backwards[i].state_grad_summaries(bounds[i], deterministic_work=True) for i in summarized]
    )
    exits = canonical_entries(rmaps, tiling, routing, maps[:1], None, reverse=True)
    grads = [b.run(exits[i : i + 1])[:5] for i, b in enumerate(backwards)]
    return (
        torch.cat(outputs, dim=1),
        *(torch.cat([g[c] for g in grads], dim=1) for c in range(5)),
    )


def _bit_mismatches(actual: torch.Tensor, expected: torch.Tensor) -> int:
    """Count unequal storage elements, including signed zeros; reject non-finite values."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    view = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    return int((actual.contiguous().view(view) != expected.contiguous().view(view)).sum())


# ---------------------------------------------------------------- two-rank cases

Case = tuple[Sequence[int], int, str, str]  # lengths, tile_size, ownership, gate

TWO_RANK_CASES: dict[str, Case] = {
    f"{layout}-{gate}": (lengths, tile_size, ownership, gate)
    for layout, lengths, tile_size, ownership in (
        ("ragged-contig", RAGGED, 64, "contiguous"),
        ("ragged-cyclic", RAGGED, 64, "cyclic"),
        ("ragged-empty", RAGGED, 128, "empty_rank"),
        ("elephant-cyclic", ELEPHANT, 64, "cyclic"),
    )
    for gate in ("strong", "mild", "zero")
}


def _case_mismatches(
    cp_rank: int, world: int, device: torch.device, case: Case, backend: str
) -> dict[str, int]:
    """Bit mismatches of this rank's output and five gradients against per-tile CP=1 calls."""
    lengths, tile_size, ownership, gate = case
    kernel_options = {"backend": backend} if backend != "fused" else None
    inputs, grad = _inputs(lengths, gate, 97)
    cu = (0, *accumulate(lengths))
    expected = _canonical_cp1(inputs, grad, lengths, tile_size, kernel_options)
    if ownership == "cyclic":
        # Whole tiles alternate between the ranks, including single-tile documents.
        tiles = tile_stream(cu, tile_size)
        fragments = [[(t.start, t.stop) for t in tiles[r::world]] for r in range(world)]
    else:
        fragments = {
            "contiguous": [[(0, cu[len(cu) // 2])], [(cu[len(cu) // 2], cu[-1])]],
            "empty_rank": [[], [(0, cu[-1])]],
            "single": [[(0, cu[-1])]],
        }[ownership]
    tiling = CanonicalTiling.from_fragments(cu, fragments, cp_rank, tile_size=tile_size)
    ids = tiling.global_token_ids(device)
    local = tuple(v[:, ids].contiguous().requires_grad_() for v in inputs)
    output = context_parallel_kda_deterministic(
        *local, tiling=tiling, group=dist.group.WORLD, kernel_options=kernel_options
    )
    # Every rank runs the backward: its all-gather is a collective even for an empty rank.
    grads = torch.autograd.grad(output, local, grad[:, ids])
    return {
        name: _bit_mismatches(a, e[:, ids])
        for name, a, e in zip(NAMES, (output, *grads), expected, strict=True)
    }


def _rank_main(cp_rank: int, world: int, port: int, cases: dict[str, Case], backend: str) -> None:
    """Run every case in one process group (both ranks in the same order); report all failures."""
    with _process_group(cp_rank, world, port) as device:
        failures = {}
        for case_id, case in cases.items():
            mismatches = _case_mismatches(cp_rank, world, device, case, backend)
            if any(mismatches.values()):
                failures[case_id] = mismatches
        assert not failures, failures


@pytest.fixture(params=["fused", "mega"])
def backend(request: pytest.FixtureRequest) -> str:
    """Skip Mega before spawning when its DSL or GPU architecture is unavailable."""
    if request.param == "mega":
        pytest.importorskip("cutlass.experimental")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            pytest.skip("Mega needs SM100/103")
    return request.param


def test_two_ranks_match_canonical_cp1_bitwise(backend):
    """One spawn per backend covers every layout and gate; a failure names each failing case."""
    _spawn(2, (TWO_RANK_CASES, backend))


def _spawn(nprocs: int, args: tuple) -> None:
    """Surface failed ranks promptly and bound the run to 600 seconds, including compilation."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = mp.spawn(_rank_main, args=(nprocs, port, *args), nprocs=nprocs, join=False)
    deadline = time.monotonic() + 600
    try:
        # join() reports failed ranks immediately, but may reap only one successful rank at a time.
        while not context.join(timeout=max(0.0, deadline - time.monotonic())):
            assert time.monotonic() < deadline, "spawned cases did not finish within 600 s"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()


@pytest.mark.parametrize("option", ["split_forward", "split_backward"])
def test_split_schedules_are_rejected_before_any_collective(option):
    """Reject Mega splits before collecting: they would change the canonical FP32 summaries."""
    tiling = _single_rank_tiling(RAGGED)
    inputs, _ = _inputs(RAGGED, "mild", 3)
    with pytest.raises(ValueError, match="split_forward/split_backward"):
        context_parallel_kda_deterministic(
            *inputs,
            tiling=tiling,
            group=None,
            kernel_options={"backend": "mega", option: True},
        )


def test_single_rank_public_api_matches_canonical_cp1_bitwise(backend):
    """CP=1 through the public entry point is the same contract the two-rank cases reproduce."""
    _spawn(1, ({"ragged-single-strong": (RAGGED, 64, "single", "strong")}, backend))


# ---------------------------------------------------------------- accuracy vs FP64


def test_canonical_cp1_matches_fp64_reference_within_bf16_budget():
    """The new contract is checked against an independent FP64 recurrence, not only self-consistency."""
    lengths = (17, 65, 129)
    inputs, grad = _inputs(lengths, "mild", 41)
    inputs = tuple(v[:, :, :1].contiguous() for v in inputs)
    grad = grad[:, :, :1].contiguous()
    actual = _canonical_cp1(inputs, grad, lengths, 64, None)
    cu = torch.tensor((0, *accumulate(lengths)), dtype=torch.int32, device="cuda")
    high = tuple(v.detach().cpu().double().requires_grad_() for v in inputs)
    out_high, _ = kda_reference(*high, cu_seqlens=cu.cpu(), output_final_state=False)
    grads_high = torch.autograd.grad(out_high, high, grad.cpu().double())
    low = tuple(v.detach().cpu().float().requires_grad_() for v in inputs)
    out_low, _ = kda_reference(*low, cu_seqlens=cu.cpu(), output_final_state=False)
    grads_low = torch.autograd.grad(out_low, low, grad.cpu().float())
    for name, a, h, lo in zip(
        NAMES, actual, (out_high, *grads_high), (out_low, *grads_low), strict=True
    ):
        err = (a.cpu().double() - h.double()).abs().max().item()
        eager = (lo.double() - h.double()).abs().max().item()
        budget = 2 * (eager + torch.finfo(torch.bfloat16).eps * h.abs().max().item())
        assert err <= budget, f"{name}: {err} > {budget}"
