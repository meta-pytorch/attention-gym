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
from itertools import accumulate, pairwise
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("cutlass")

import attn_gym.linear.context_parallel_deterministic as det
from attn_gym.linear.context_parallel_deterministic import (
    CanonicalTiling,
    Tile,
    boundary_states,
    canonical_scan,
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
    entries = torch.zeros(leaves.shape[0], leaves.shape[1], value_dim, leaves.shape[3])
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
    leaves = torch.randn(n, 2, 8, 4)
    leaves[:, :, 4:, :] = torch.eye(4) + 0.01 * torch.randn(n, 2, 4, 4)
    return leaves


@pytest.mark.parametrize("documents", [[0] * 9, [0, 0, 0, 1, 2, 2, 2], [0, 1, 2, 3], [0] * 1])
def test_scan_matches_serial_composition_per_document(documents):
    """The Blelloch scan restarts at documents and equals serial composition (fp32 rounding)."""
    leaves = _leaves(len(documents), 0)
    tiles = [Tile(d, 0, 0) for d in documents]
    torch.testing.assert_close(
        boundary_states(leaves, tiles, reverse=False), _serial_entries(leaves, documents)
    )
    reversed_docs = documents[::-1]
    expected = _serial_entries(leaves.flip(0), reversed_docs).flip(0)
    torch.testing.assert_close(boundary_states(leaves, tiles, reverse=True), expected)


def test_scan_of_a_document_is_independent_of_its_neighbours():
    """A document's entry states are bitwise the same wherever it sits in the packed stream."""
    document = _leaves(5, 1)
    reference = None
    for before, after in ((0, 0), (1, 0), (3, 2), (7, 5)):
        others = _leaves(before + after, 2 + before)
        leaves = torch.cat((others[:before], document, others[before:]))
        tiles = [Tile(0, 0, 0)] * before + [Tile(1, 0, 0)] * 5 + [Tile(2, 0, 0)] * after
        for reverse in (False, True):
            entries = boundary_states(leaves, tiles, reverse=reverse)[before : before + 5]
            key = (reverse,)
            if reference is None:
                reference = {}
            if key not in reference:
                reference[key] = entries
            assert torch.equal(entries, reference[key]), (before, after, reverse)


def test_scan_prefix_ignores_right_padding():
    """Blelloch reads only positions <= j, so padding a row on the right never changes bits."""
    row = _leaves(6, 3)
    identity = torch.zeros(1, 2, 8, 4)
    identity[:, :, 4:, :] = torch.eye(4)
    base = canonical_scan(row[None])[0]
    for pad in (1, 2, 5, 10):
        padded = torch.cat((row, identity.expand(pad, -1, -1, -1)))
        assert torch.equal(canonical_scan(padded[None])[0, :6], base)


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
def _canonical_cp1(inputs, grad, lengths, tile_size: int, kernel_options):
    """Single-process canonical run over the same tiles: the contract every ownership reproduces.

    Every tile is prepared by its own staged call (the strictest form of the leaf contract); the
    batched production path must match this bitwise.
    """
    cu = (0, *accumulate(lengths))
    tiles = tile_stream(cu, tile_size)
    stages = _kda_stages(None, False, False, kernel_options)
    device = inputs[0].device

    def bounds(t):
        return torch.tensor([[0, t.length]], dtype=torch.int32, device=device)

    prepared = [
        stages.prepare(*(v[:, t.start : t.stop] for v in inputs), cu_seqlens=None) for t in tiles
    ]
    maps = torch.cat(
        [
            p.state_summaries(bounds(t), deterministic_work=True)
            for p, t in zip(prepared, tiles, strict=True)
        ]
    )
    entries = boundary_states(maps, tiles, reverse=False)
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
            b.state_grad_summaries(bounds(t), deterministic_work=True)
            for b, t in zip(backwards, tiles, strict=True)
        ]
    )
    exits = boundary_states(rmaps, tiles, reverse=True)
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


def _rank_main(cp_rank, world, port, lengths, tile_size, ownership, gate, backend):
    device = _init(cp_rank, world, port)
    try:
        kernel_options = {"backend": backend} if backend != "fused" else None
        inputs, grad = _inputs(lengths, gate, 97, device)
        cu = (0, *accumulate(lengths))
        expected = _canonical_cp1(inputs, grad, lengths, tile_size, kernel_options)
        if ownership == "cyclic":
            tiles = tile_stream(cu, tile_size)
            fragments = [
                [(t.start, t.stop) for i, t in enumerate(tiles) if i % 2 == r] for r in range(2)
            ]
        else:
            fragments = OWNERSHIPS[ownership](cu, len(lengths))
        tiling = CanonicalTiling.from_fragments(cu, fragments, cp_rank, tile_size=tile_size)
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
    "lengths,tile_size,ownership",
    [
        (RAGGED, 64, "contiguous"),
        (RAGGED, 64, "cyclic"),
        (RAGGED, 128, "empty_rank"),
        (ELEPHANT, 64, "cyclic"),
    ],
    ids=["ragged-contig", "ragged-cyclic", "ragged-empty", "elephant-cyclic"],
)
def test_two_ranks_match_canonical_cp1_bitwise(lengths, tile_size, ownership, gate, backend):
    if backend == "mega":
        pytest.importorskip("cutlass.experimental")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            pytest.skip("Mega needs SM100/103")
    mp.spawn(
        _rank_main,
        args=(2, _free_port(), lengths, tile_size, ownership, gate, backend),
        nprocs=2,
        join=True,
    )


# ---------------------------------------------------------------- accuracy vs FP64


def test_canonical_cp1_matches_fp64_reference_within_bf16_budget():
    """The new contract is checked against an independent FP64 recurrence, not only self-consistency."""
    lengths = (17, 65, 129)
    inputs, grad = _inputs(lengths, "mild", 41, torch.device("cuda"))
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
