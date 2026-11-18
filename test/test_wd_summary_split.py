"""warpdrive CP summaries split into pieces: accuracy against FP64, selection, replay, plans."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F
import triton

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear._delta_rule.warpdrive.drivers import plan
from attn_gym.linear._delta_rule.warpdrive.drivers import summary as native_summary
from attn_gym.linear.gdn.stages import chunk_gdn_prepare, chunk_gdn_prepare_backward
from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward
from attn_gym.testing.kda import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="native warpdrive summaries require an SM100-family GPU",
)

NUM_SM = 152
SCALE = 128**-0.5
# One long subsequence, a shorter one, a short one and an empty one: the long ones split.
LENGTHS = [37, 0, 4000, 1500]


class Variant(NamedTuple):
    forward: object
    reverse: object
    channel_gate: bool  # KDA decays per key channel, GDN per head
    chunk: int
    cost: plan.SummaryCost  # both directions run the forward's plan


KDA = Variant(
    partial(native_summary.build_state_summaries, intra_chunk_dtypes="tf32+fp16"),
    partial(native_summary.build_state_grad_summaries, intra_chunk_dtypes="tf32+fp16"),
    True,
    16,
    native_summary.KDA_SUMMARY_COST,
)
GDN = Variant(
    native_summary.build_gdn_state_summaries,
    native_summary.build_gdn_state_grad_summaries,
    False,
    64,
    native_summary.GDN_SUMMARY_COST,
)
VARIANTS = [pytest.param(KDA, id="kda"), pytest.param(GDN, id="gdn")]


def _inputs(variant: Variant, lengths: list[int], heads: int = 2, seed: int = 0):
    """Staged operands with gates mild enough to keep every transition above rounding."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(lengths)
    shape = (1, tokens, heads, 128)

    def normal(*size):
        return torch.randn(size, device="cuda", generator=generator)

    q, k = (F.normalize(normal(*shape), dim=-1).bfloat16() for _ in range(2))
    v, d_output = (normal(*shape).bfloat16() for _ in range(2))
    gate_shape = shape if variant.channel_gate else shape[:3]
    gate = -0.004 * torch.rand(gate_shape, device="cuda", generator=generator)
    beta = torch.rand(shape[:3], device="cuda", generator=generator)
    return (q, k, v, gate, beta, d_output), cumulative_sequence_offsets(lengths)


def _build(variant: Variant, direction: str, operands, cu, **kwargs) -> torch.Tensor:
    q, k, v, gate, beta, d_output = operands
    if direction == "forward":
        return variant.forward(k, v, gate, beta, cu, **kwargs)
    return variant.reverse(q, k, v, gate, beta, d_output, cu, SCALE, **kwargs)


def _splits(variant: Variant, lengths: list[int], heads: int = 2) -> bool:
    span = plan._summary_span(
        sum(lengths), len(lengths), heads, NUM_SM, variant.chunk, variant.cost
    )
    return span > 0


def _fp64_maps(variant: Variant, operands, cu: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Per-sequence FP64 ``[B; A]`` and ``[C; A.T]`` from the token recurrence ``S <- S D_t (I -
    beta k k^T) + beta v k^T`` (V-major state, ``o_t = scale * S_t q_t``)."""
    q, k, v, gate, beta, d_output = (t[0].double() for t in operands)
    heads, dim = q.shape[1], q.shape[2]
    forward, reverse = [], []
    for start, stop in zip(cu[:-1].tolist(), cu[1:].tolist(), strict=True):
        entry = torch.zeros(heads, dim, dim, dtype=torch.float64, device="cuda")
        entry = entry.requires_grad_()
        state, transition = entry, torch.eye(dim, dtype=torch.float64, device="cuda")
        transition = transition.expand(heads, dim, dim)
        loss = 0.0
        for t in range(start, stop):
            decay = gate[t].exp()
            decay = decay[:, None, :] if variant.channel_gate else decay[:, None, None]
            kt, bt = k[t], beta[t][:, None, None]
            state = state * decay
            state = state + bt * (v[t][:, :, None] - state @ kt[:, :, None]) * kt[:, None, :]
            transition = transition * decay
            transition = transition - bt * (transition @ kt[:, :, None]) * kt[:, None, :]
            loss = loss + SCALE * ((state @ q[t][:, :, None])[..., 0] * d_output[t]).sum()
        (bias,) = torch.autograd.grad(loss, entry) if stop > start else (torch.zeros_like(entry),)
        forward.append(torch.cat((state.detach(), transition), dim=1))
        reverse.append(torch.cat((bias, transition.transpose(-1, -2)), dim=1))
    return torch.stack(forward), torch.stack(reverse)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected).norm() / expected.norm()).item()


def _uncut(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(native_summary, "summary_span", lambda *args, **kwargs: 0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_split_summaries_are_as_accurate_as_uncut(variant: Variant, monkeypatch) -> None:
    """Splitting reorders the scan, not its precision: every split map part tracks FP64 as
    closely as the uncut scan does, and the empty sequence keeps the exact identity."""
    operands, cu = _inputs(variant, LENGTHS)
    references = dict(zip(("forward", "reverse"), _fp64_maps(variant, operands, cu), strict=True))
    split = {}
    for direction in references:
        assert _splits(variant, LENGTHS)
        split[direction] = _build(variant, direction, operands, cu)
    _uncut(monkeypatch)
    for direction, expected in references.items():
        uncut = _build(variant, direction, operands, cu)
        assert torch.isfinite(split[direction]).all()
        for index in (0, 2, 3):
            for part, rows in (("bias", slice(0, 128)), ("transition", slice(128, None))):
                rows = (index, slice(None), rows)
                split_error = _relative_l2(split[direction][rows], expected[rows])
                uncut_error = _relative_l2(uncut[rows], expected[rows])
                assert split_error <= 1.25 * uncut_error + 1e-6, (
                    direction,
                    index,
                    part,
                    split_error,
                    uncut_error,
                )
                # Pointwise too, so an error local to a piece boundary cannot hide in the norm.
                split_worst = (split[direction][rows].double() - expected[rows]).abs().max()
                uncut_worst = (uncut[rows].double() - expected[rows]).abs().max()
                assert split_worst <= 2 * uncut_worst + 1e-6, (direction, index, part)
        identity = torch.cat((torch.zeros(128, 128), torch.eye(128))).cuda().expand(2, -1, -1)
        assert torch.equal(split[direction][1], identity)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize(
    ("lengths", "heads"),
    [([256], 16), ([8192], 4), ([2048], 8), (LENGTHS, 2)],
    ids=["256x16", "8192x4", "2048x8", "ragged"],
)
def test_reverse_transition_is_the_forward_transition_transposed(
    variant: Variant, lengths: list[int], heads: int
) -> None:
    """The reverse maps' R is the forward maps' A transposed bit for bit, the exact adjoint,
    uncut (KDA 256x16) or split, also where a transposed composition of the same pieces rounds
    differently (GDN 2048x8)."""
    operands, cu = _inputs(variant, lengths, heads=heads)
    forward = _build(variant, "forward", operands, cu)
    reverse = _build(variant, "reverse", operands, cu)
    transposed = forward[:, :, 128:].transpose(-1, -2)
    assert torch.equal(reverse[:, :, 128:].view(torch.int32), transposed.view(torch.int32))


@pytest.mark.parametrize("variant", VARIANTS)
def test_split_summaries_ignore_the_default_dtype(variant: Variant) -> None:
    """The split maps and their scratch name FP32: a float64 default dtype changes no bit."""
    operands, cu = _inputs(variant, LENGTHS)
    directions = ("forward", "reverse")
    expected = [_build(variant, direction, operands, cu) for direction in directions]
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        actual = [_build(variant, direction, operands, cu) for direction in directions]
    finally:
        torch.set_default_dtype(default)
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got, want)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_split_bounds_select_rows_bitwise(variant: Variant, direction: str) -> None:
    """The piece span follows the call's shapes, not the bounds, so selecting, reordering or
    repeating subsequences returns the full call's rows bit for bit, a call repeats bitwise,
    and bounds may change between CUDA Graph replays."""
    assert _splits(variant, LENGTHS)
    operands, cu = _inputs(variant, LENGTHS)
    build = partial(_build, variant, direction, operands, cu)
    full = build()
    assert torch.equal(build().view(torch.int32), full.view(torch.int32))
    starts = cu.tolist()
    rows = {index: [starts[index], starts[index + 1]] for index in range(len(LENGTHS))}
    cases = ([3, 1, 2, 3], [2, 0, 1, 1], [0, 3, 2, 0])
    bounds = torch.tensor([rows[i] for i in cases[0]], dtype=torch.int32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = build(bounds=bounds)
    for indices in cases:
        bounds.copy_(torch.tensor([rows[i] for i in indices], dtype=torch.int32))
        graph.replay()
        assert torch.equal(captured.view(torch.int32), full[indices].view(torch.int32))
    # A nonempty row that is not a subsequence is poisoned with NaN rather than silently mapped.
    unmatched = [rows[2], [starts[2] + 64, starts[3]]]
    poisoned = build(bounds=torch.tensor(unmatched, dtype=torch.int32, device="cuda"))
    assert torch.equal(poisoned[0], full[2]) and poisoned[1].isnan().all()


def _staged_summaries(variant: Variant, operands, cu, bounds, options) -> tuple[torch.Tensor, ...]:
    """Forward and reverse summaries of the staged handles, the context-parallel entry."""
    q, k, v, gate, beta, d_output = operands
    if variant.channel_gate:
        prepare, prepare_backward = chunk_kda_prepare, chunk_kda_prepare_backward
    else:
        prepare, prepare_backward = chunk_gdn_prepare, chunk_gdn_prepare_backward
    prepared = prepare(q, k, v, gate, beta, cu_seqlens=cu, kernel_options=options)
    backward = prepare_backward(
        prepared.saved, d_output, None, scale=prepared.scale, kernel_options=options
    )
    return prepared.state_summaries(bounds), backward.state_grad_summaries(bounds)


@pytest.mark.parametrize("variant", VARIANTS)
def test_batch_invariant_summaries_ignore_the_pack(variant: Variant) -> None:
    """One document alone, then first and last among forty neighbors: alone its summaries split,
    packed they do not. ``batch_invariant`` scans it uncut either way, so its forward and
    reverse maps keep their bits; by default they change with the pack."""
    doc_lengths, neighbor_lengths = [2048], [512] * 40
    assert _splits(variant, doc_lengths) and not _splits(variant, doc_lengths + neighbor_lengths)
    doc, _ = _inputs(variant, doc_lengths, seed=1)
    neighbors, _ = _inputs(variant, neighbor_lengths, seed=2)
    packs = {
        "alone": (doc_lengths, (doc,), 0),
        "first": (doc_lengths + neighbor_lengths, (doc, neighbors), 0),
        "last": (neighbor_lengths + doc_lengths, (neighbors, doc), sum(neighbor_lengths)),
    }
    results = {True: {}, False: {}}
    for name, (lengths, parts, start) in packs.items():
        operands = [torch.cat(tensors, dim=1) for tensors in zip(*parts, strict=True)]
        cu = cumulative_sequence_offsets(lengths)
        bounds = torch.tensor([[start, start + 2048]], dtype=torch.int32, device="cuda")
        for invariant, by_pack in results.items():
            options = {"backend": "warpdrive", "batch_invariant": invariant}
            by_pack[name] = _staged_summaries(variant, operands, cu, bounds, options)
    for name in ("first", "last"):
        for got, want in zip(results[True][name], results[True]["alone"], strict=True):
            assert torch.equal(got.view(torch.int32), want.view(torch.int32)), name
        assert not torch.equal(results[False][name][0], results[False]["alone"][0]), name


def _poison_free_memory() -> None:
    """Leave a NaN-filled block in the caching allocator, so uninitialized scratch is NaN."""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.full((64 << 20,), float("nan"), device="cuda")
    torch.cuda.synchronize()


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_split_unselected_sequences_keep_the_identity(
    variant: Variant, direction: str, monkeypatch
) -> None:
    """A sequence no bound selects still owns a piece; its sequence map is ``[0; I]`` rather
    than a composition of scratch no kernel wrote (NaN here)."""
    assert _splits(variant, LENGTHS)
    sequence_maps = []
    gather = native_summary._summary_rows

    def spy(maps, cu_seqlens, bounds):
        sequence_maps.append(maps.clone())
        return gather(maps, cu_seqlens, bounds)

    monkeypatch.setattr(native_summary, "_summary_rows", spy)
    operands, cu = _inputs(variant, LENGTHS)
    starts = cu.tolist()
    bounds = torch.tensor([starts[2:4]], dtype=torch.int32, device="cuda")
    _poison_free_memory()
    selected = _build(variant, direction, operands, cu, bounds=bounds)
    assert torch.isfinite(selected).all()
    identity = torch.cat((torch.zeros(128, 128), torch.eye(128))).cuda().expand(2, -1, -1)
    for index in (0, 1, 3):
        assert torch.equal(sequence_maps[0][index], identity), index


def test_split_plans_follow_the_span_shape() -> None:
    """Few long subsequences on few heads split into pieces that fit one wave; short spans and
    spans whose subsequences already fill the SMs stay uncut."""
    span = partial(plan._summary_span, num_sm=NUM_SM)
    for variant in (KDA, GDN):
        every = partial(span, b_t=variant.chunk, cost=variant.cost)
        long_span = every(8192, 1, 4)
        assert long_span > 0
        assert 8192 // variant.chunk // long_span * 4 <= NUM_SM
        assert every(32 * 4096, 32, 8) == 0  # 256 subsequence-heads fill the SMs
        assert every(129, 1, 1) == 0
    # The test spans of the bitwise native-summary tests stay uncut.
    assert span(211, 4, 1, b_t=16, cost=KDA.cost) == 0


def test_split_planner_compiles_once_per_piece_bucket(triton_compiles) -> None:
    """Token, sequence, span and bound-row counts (1, multiples of 16, others) are runtime
    values of the planner: it compiles once per power-of-two bucket of a sequence's pieces."""
    cases = [[4000], [3000, 1000], [2900, 0, 1100], [2000, 16, 1984], [1700, 300, 1, 2047]]
    buckets = set()
    for lengths in cases:
        tokens = sum(lengths)
        span = plan._summary_span(tokens, len(lengths), 1, NUM_SM, 16, KDA.cost)
        assert span
        buckets.add(triton.next_power_of_2(-(-tokens // 16 // span)))
        operands, cu = _inputs(KDA, lengths, heads=1)
        for rows in (1, 16):
            bounds = cu[:2].expand(rows, 2).contiguous()
            _build(KDA, "forward", operands, cu, bounds=bounds)
        _build(KDA, "forward", operands, cu)
    assert triton_compiles["_plan_summary_pieces"] == len(buckets) * 2, triton_compiles
