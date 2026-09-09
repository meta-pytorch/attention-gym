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
from itertools import accumulate, pairwise
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("cutlass")

import attn_gym.linear.context_parallel_deterministic as det
from attn_gym.linear._delta_rule.triton.canonical_scan import canonical_scan_entries
from attn_gym.linear.context_parallel_deterministic import (
    CanonicalTiling,
    canonical_entries,
    tile_stream,
)
from attn_gym.linear.kda.context_parallel import _kda_stages, context_parallel_kda_deterministic
from attn_gym.testing.kda import kda_reference, make_kda_test_inputs

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.xdist_group("two-gpu"),
]

HEADS, DIM = 4, 128
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
    tiling = CanonicalTiling.from_fragments(cu, [[(0, 981)]], 0, tile_size=64, scan_block=2)
    # The 17-token document fits one tile and is not summarized; blocks never span documents.
    single = {i for i, t in enumerate(tiling.tiles) if t.document == 0}
    assert set(tiling.summarized) == set(range(len(tiling.tiles))) - single
    assert all(len({tiling.tiles[i].document for i in block}) == 1 for block in tiling.blocks)
    assert [i for block in tiling.blocks for i in block] == list(tiling.summarized)
    # Document 4 (tokens 468..981) has tiles at 468, 532, ...; a 2-tile block boundary is 596.
    CanonicalTiling.from_fragments(cu, [[(0, 596)], [(596, 981)]], 0, tile_size=64, scan_block=2)
    with pytest.raises(ValueError, match="one scan block"):
        CanonicalTiling.from_fragments(
            cu, [[(0, 532)], [(532, 981)]], 0, tile_size=64, scan_block=2
        )


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


def _serial_entries(leaves: torch.Tensor, documents: list[int]) -> torch.Tensor:
    """Exclusive prefix bias per tile by serial composition, restarting at each document."""
    from attn_gym.linear.context_parallel import compose_summaries

    value_dim = leaves.shape[2] - leaves.shape[3]
    entries = torch.zeros(
        leaves.shape[0], leaves.shape[1], value_dim, leaves.shape[3], device=leaves.device
    )
    prefix = None
    for i, doc in enumerate(documents):
        if i == 0 or doc != documents[i - 1]:
            prefix = leaves[i]
        else:
            entries[i] = prefix[:, :value_dim, :]
            prefix = compose_summaries(prefix, leaves[i])
    return entries


def _leaves(n: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    leaves = torch.randn(n, 2, 256, 128, device="cuda")
    leaves[:, :, 128:, :] = torch.eye(128, device="cuda") + 0.01 * torch.randn(
        n, 2, 128, 128, device="cuda"
    )
    return leaves


def _doc_offsets(documents: list[int]) -> torch.Tensor:
    offsets = [0] + [i for i in range(1, len(documents)) if documents[i] != documents[i - 1]]
    return torch.tensor([*offsets, len(documents)], dtype=torch.int32, device="cuda")


def _single_rank_tiling(documents: list[int], scan_block: int) -> CanonicalTiling:
    """One rank owning every tile of documents with the given tile counts (64-token tiles)."""
    lengths = []
    for doc in dict.fromkeys(documents):
        lengths.append(64 * documents.count(doc))
    cu = (0, *accumulate(lengths))
    return CanonicalTiling.from_fragments(
        cu, [[(0, cu[-1])]], 0, tile_size=64, scan_block=scan_block
    )


@pytest.mark.parametrize("documents", [[0] * 9, [0, 0, 0, 1, 2, 2, 2], [0, 1, 2, 3], [0] * 1])
def test_fused_scan_matches_serial_composition_per_document(documents):
    """The fused scan restarts at documents and equals serial composition to fp32 accuracy."""
    leaves = _leaves(len(documents), 0)
    offsets = _doc_offsets(documents)
    torch.testing.assert_close(
        canonical_scan_entries(leaves, offsets),
        _serial_entries(leaves, documents),
        atol=1e-4,
        rtol=1e-5,
    )
    expected = _serial_entries(leaves.flip(0), documents[::-1]).flip(0)
    torch.testing.assert_close(
        canonical_scan_entries(leaves, offsets, reverse=True), expected, atol=1e-4, rtol=1e-5
    )


@pytest.mark.parametrize("scan_block", [1, 2, 4])
def test_blocked_entries_match_serial_composition(scan_block):
    """The three-level block tree equals serial composition to fp32 accuracy, both directions."""
    documents = [0] * 7 + [1] * 3 + [2] * 5
    leaves = _leaves(len(documents), 5)
    tiling = _single_rank_tiling(documents, scan_block)
    like = leaves[:1]
    for reverse in (False, True):
        got = canonical_entries(
            leaves, tiling, tiling.routing("cuda"), like, None, reverse=reverse
        )
        if reverse:
            expected = _serial_entries(leaves.flip(0), documents[::-1]).flip(0)
        else:
            expected = _serial_entries(leaves, documents)
        torch.testing.assert_close(got, expected, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("scan_block", [1, 4])
def test_entries_of_a_document_are_independent_of_its_neighbours(scan_block):
    """A document's entry states are bitwise the same wherever it sits in the packed stream."""
    document = _leaves(5, 1)
    reference: dict[bool, torch.Tensor] = {}
    for before, after in ((0, 0), (1, 0), (3, 2), (7, 5)):
        others = _leaves(before + after, 2 + before)
        leaves = torch.cat((others[:before], document, others[before:]))
        documents = [0] * before + [1] * 5 + [2] * after
        tiling = _single_rank_tiling(documents, scan_block)
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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


_REAL_ALL_GATHER = dist.all_gather_into_tensor


def _cpu_all_gather_into_tensor(gathered, packet, group=None):
    """Gloo path: gather CPU copies, then place them into the CUDA output buffer."""
    staging = torch.empty(gathered.shape, dtype=gathered.dtype)
    _REAL_ALL_GATHER(staging, packet.cpu(), group=group)
    gathered.copy_(staging)


def _init(cp_rank: int, world: int, port: int) -> torch.device:
    shared = torch.cuda.device_count() == 1
    device = torch.device("cuda", 0 if shared else cp_rank)
    torch.cuda.set_device(device)
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group(
        "gloo" if shared else "nccl",
        rank=cp_rank,
        world_size=world,
        device_id=None if shared else device,
    )
    if shared:
        patcher = patch.object(det.dist, "all_gather_into_tensor", _cpu_all_gather_into_tensor)
        patcher.start()
    return device


# ---------------------------------------------------------------- canonical CP1 reference


def _inputs(lengths, gate: str, seed: int, device):
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
    values = tuple(v.to(device) for v in values)
    return values, torch.randn_like(values[2])


@torch.no_grad()
def _canonical_cp1(inputs, grad, lengths, tile_size: int, scan_block: int, kernel_options):
    """Single-process canonical run over the same tiles: the contract every ownership reproduces.

    Every tile is prepared by its own staged call (the strictest form of the leaf contract); the
    batched production path must match this bitwise. The scan tree is the production one.
    """
    cu = (0, *accumulate(lengths))
    tiles = tile_stream(cu, tile_size)
    tiling = CanonicalTiling.from_fragments(
        cu, [[(0, cu[-1])]], 0, tile_size=tile_size, scan_block=scan_block
    )
    stages = _kda_stages(None, False, False, kernel_options)
    device = inputs[0].device

    def bounds(t):
        return torch.tensor([[0, t.length]], dtype=torch.int32, device=device)

    prepared = [
        stages.prepare(*(v[:, t.start : t.stop] for v in inputs), cu_seqlens=None) for t in tiles
    ]
    # Single-tile documents are not summarized (NOTE [Single-Tile Documents]).
    summarized = tiling.summarized
    maps = torch.cat(
        [
            prepared[i].state_summaries(bounds(tiles[i]), deterministic_work=True)
            for i in summarized
        ]
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
        [
            backwards[i].state_grad_summaries(bounds(tiles[i]), deterministic_work=True)
            for i in summarized
        ]
    )
    exits = canonical_entries(rmaps, tiling, routing, maps[:1], None, reverse=True)
    grads = [b.run(exits[i : i + 1])[:5] for i, b in enumerate(backwards)]
    return (
        torch.cat(outputs, dim=1),
        *(torch.cat([g[c] for g in grads], dim=1) for c in range(5)),
    )


def _bits_equal(actual: torch.Tensor, expected: torch.Tensor) -> int:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    view = torch.int16 if actual.dtype == torch.bfloat16 else torch.int32
    return int((actual.contiguous().view(view) != expected.contiguous().view(view)).sum())


# ---------------------------------------------------------------- two-rank cases

OWNERSHIPS = {
    "contiguous": lambda cu, n: [[(0, cu[len(cu) // 2])], [(cu[len(cu) // 2], cu[-1])]],
    "empty_rank": lambda cu, n: [[], [(0, cu[-1])]],
}


def _rank_main(cp_rank, world, port, lengths, tile_size, scan_block, ownership, gate, backend):
    device = _init(cp_rank, world, port)
    try:
        kernel_options = {"backend": backend} if backend != "fused" else None
        inputs, grad = _inputs(lengths, gate, 97, device)
        cu = (0, *accumulate(lengths))
        expected = _canonical_cp1(inputs, grad, lengths, tile_size, scan_block, kernel_options)
        if ownership == "cyclic":
            # Alternate whole blocks between the ranks (tiles when scan_block == 1); tiles of
            # single-tile documents form no block and alternate on their own.
            tiling = CanonicalTiling.from_fragments(
                cu, [[(0, cu[-1])]], 0, tile_size=tile_size, scan_block=scan_block
            )
            units = sorted(
                [
                    *tiling.blocks,
                    *((i,) for i in range(len(tiling.tiles)) if i not in tiling.summarized),
                ]
            )
            tiles = tiling.tiles
            fragments = [
                [
                    (tiles[unit[0]].start, tiles[unit[-1]].stop)
                    for u, unit in enumerate(units)
                    if u % 2 == r
                ]
                for r in range(2)
            ]
        else:
            fragments = OWNERSHIPS[ownership](cu, len(lengths))
        tiling = CanonicalTiling.from_fragments(
            cu, fragments, cp_rank, tile_size=tile_size, scan_block=scan_block
        )
        ids = tiling.global_token_ids(device)
        local = tuple(v[:, ids].contiguous().requires_grad_() for v in inputs)
        output = context_parallel_kda_deterministic(
            *local, tiling=tiling, group=dist.group.WORLD, kernel_options=kernel_options
        )
        # Every rank runs the backward: its all-gather is a collective even for an empty rank.
        grads = torch.autograd.grad(output, local, grad[:, ids])
        mismatches = {
            name: _bits_equal(a, e[:, ids])
            for name, a, e in zip(NAMES, (output, *grads), expected, strict=True)
        }
        assert all(v == 0 for v in mismatches.values()), mismatches
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", ["fused", "mega"])
@pytest.mark.parametrize("gate", ["strong", "mild", "zero"])
@pytest.mark.parametrize(
    "lengths,tile_size,scan_block,ownership",
    [
        (RAGGED, 64, 1, "contiguous"),
        (RAGGED, 64, 1, "cyclic"),
        (RAGGED, 128, 1, "empty_rank"),
        (ELEPHANT, 64, 1, "cyclic"),
        (RAGGED, 64, 4, "cyclic"),
        (ELEPHANT, 64, 4, "contiguous"),
    ],
    ids=[
        "ragged-contig",
        "ragged-cyclic",
        "ragged-empty",
        "elephant-cyclic",
        "ragged-cyclic-block4",
        "elephant-contig-block4",
    ],
)
def test_two_ranks_match_canonical_cp1_bitwise(
    lengths, tile_size, scan_block, ownership, gate, backend
):
    if backend == "mega":
        pytest.importorskip("cutlass.experimental")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            pytest.skip("Mega needs SM100/103")
    # A rank that raises leaves its peer blocked in a collective; poll the context so the
    # failure surfaces instead of the case hanging until the outer timeout.
    context = mp.spawn(
        _rank_main,
        args=(2, _free_port(), lengths, tile_size, scan_block, ownership, gate, backend),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 300
    try:
        # join(timeout) returns after each process exits (raising if it failed); loop until all did.
        while not context.join(timeout=max(0.0, deadline - time.monotonic())):
            assert time.monotonic() < deadline, "two-rank case did not finish within 300 s"
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()


# ---------------------------------------------------------------- accuracy vs FP64


def test_canonical_cp1_matches_fp64_reference_within_bf16_budget():
    """The new contract is checked against an independent FP64 recurrence, not only self-consistency."""
    lengths = (17, 65, 129)
    inputs, grad = _inputs(lengths, "mild", 41, torch.device("cuda"))
    inputs = tuple(v[:, :, :1].contiguous() for v in inputs)
    grad = grad[:, :, :1].contiguous()
    actual = _canonical_cp1(inputs, grad, lengths, 64, 2, None)
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
