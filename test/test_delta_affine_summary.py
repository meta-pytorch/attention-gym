"""Tests for the architecture-routed delta-rule affine-summary recurrence."""

from __future__ import annotations

from functools import partial, reduce
from itertools import pairwise
from unittest.mock import Mock

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass")

import attn_gym.linear._delta_rule.cute.affine_summary_fwd as native_fwd
import attn_gym.linear._delta_rule.cute.affine_summary_rev as native_rev
import attn_gym.linear._delta_rule.triton.affine_summary_fwd as portable_fwd
import attn_gym.linear._delta_rule.triton.affine_summary_rev as portable_rev
from attn_gym.linear._delta_rule.cute import build_state_grad_summaries, build_state_summaries
from attn_gym.linear._delta_rule.triton.work_items import compose_work_items, plan_work_items
from attn_gym.linear.kda.constants import is_sm100_kda_capability
from attn_gym.testing.kda import assert_relative_rms_within


def whole_stream(like: torch.Tensor) -> torch.Tensor:
    """``bounds`` covering all ``T`` tokens of a ``[1, T, ...]`` tensor; ``arange`` keeps it capturable."""
    return (torch.arange(2, dtype=torch.int32, device=like.device) * like.shape[1]).view(1, 2)


def build_state_summary(kg, w, u, cumulative_gate):
    """The single whole-stream summary: ``R = 1`` of ``build_state_summaries``."""
    return build_state_summaries(kg, w, u, cumulative_gate, whole_stream(kg))[0]


def build_state_grad_summary(qg, kg, w, dout, aqk, cumulative_gate, scale):
    """The single whole-stream reverse summary: ``R = 1`` of ``build_state_grad_summaries``."""
    bounds = whole_stream(qg)
    return build_state_grad_summaries(qg, kg, w, dout, aqk, cumulative_gate, scale, bounds)[0]


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the affine summaries require CUDA capability 8.0 or newer",
)

VALUE_DIM = 128


@pytest.fixture(params=["routed", "portable"])
def summary_backend(request, monkeypatch):
    """Run the routed backend, or force the public wrappers onto the portable Triton path."""
    if request.param == "portable":
        if not is_sm100_kda_capability(torch.cuda.get_device_capability()):
            pytest.skip("the routed backend is already the portable Triton path")
        for module in (native_fwd, native_rev):
            monkeypatch.setattr(module, "is_sm100_kda_capability", lambda _capability: False)


def compose_summaries(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Compose V-first packed summaries so ``state @ R1 + b1`` feeds ``@ R2 + b2``."""
    bias_1, transition_1 = first[:, :VALUE_DIM], first[:, VALUE_DIM:]
    bias_2, transition_2 = second[:, :VALUE_DIM], second[:, VALUE_DIM:]
    return torch.cat((bias_1 @ transition_2 + bias_2, transition_1 @ transition_2), dim=1)


def shard_summaries(summary, tensors, boundaries: tuple[int, ...]) -> list[torch.Tensor]:
    """Summarize each token shard delimited by ``boundaries``, in token order."""
    endpoints = (0, *boundaries, tensors[0].shape[1])
    return [
        summary(*(tensor[:, start:stop] for tensor in tensors))
        for start, stop in pairwise(endpoints)
    ]


def make_summary_inputs(
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
    tokens: int = 64,
    heads: int = 1,
    divisor: float = 32,
) -> tuple[torch.Tensor, ...]:
    """Create one nontrivial deterministic input set for both affine summaries."""
    torch.manual_seed(seed)
    shape = (1, tokens, heads, 128)
    qg = torch.randn(shape, dtype=dtype, device="cuda") / divisor
    kg = torch.randn_like(qg) / divisor
    w = torch.randn_like(qg) / divisor
    u = torch.randn_like(qg) / divisor
    dout = torch.randn_like(qg) / divisor
    aqk = torch.randn(1, tokens, heads, 64, dtype=dtype, device="cuda") / divisor
    cumulative_gate = -torch.rand(shape, dtype=torch.float32, device="cuda") / 8
    return qg, kg, w, u, dout, aqk, cumulative_gate


def int64_stride_copy(tensor: torch.Tensor) -> torch.Tensor:
    """Copy a B=1 tensor into a contiguous view whose unreachable batch stride needs int64."""
    result = torch.empty_strided(
        tensor.shape,
        (2**31, *tensor.stride()[1:]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    result.copy_(tensor)
    assert result.is_contiguous()
    return result


def misaligned_copy(tensor: torch.Tensor) -> torch.Tensor:
    """Copy into contiguous storage beginning one element past an aligned allocation."""
    storage = torch.empty(tensor.numel() + 1, dtype=tensor.dtype, device=tensor.device)
    result = storage[1:].view(tensor.shape)
    result.copy_(tensor)
    assert result.is_contiguous() and result.data_ptr() % 16 != 0
    return result


def state_summary_reference(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    cumulative_gate: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the augmented affine recurrence in eager FP32."""
    _, tokens, heads, key_dim = kg.shape
    value_dim = u.shape[-1]
    pad = (-tokens) % 64
    if pad:
        padding = (0, 0, 0, 0, 0, pad)
        kg, w, u = (F.pad(tensor, padding) for tensor in (kg, w, u))
        cumulative_gate = torch.cat(
            (cumulative_gate, cumulative_gate[:, -1:].expand(-1, pad, -1, -1)),
            dim=1,
        )
        tokens += pad
    identity = torch.eye(key_dim, dtype=torch.float32, device=kg.device)
    state = torch.cat(
        (
            torch.zeros(heads, key_dim, value_dim, dtype=torch.float32, device=kg.device),
            identity.expand(heads, key_dim, key_dim),
        ),
        dim=-1,
    )
    for start in range(0, tokens, 64):
        stop = start + 64
        chunk_w = w[0, start:stop].transpose(0, 1).float()
        chunk_kg = kg[0, start:stop].transpose(0, 1).float()
        tmp = chunk_w @ state
        tmp[..., :value_dim] = u[0, start:stop].transpose(0, 1).float() - tmp[..., :value_dim]
        tmp[..., value_dim:].neg_()
        decay = cumulative_gate[0, stop - 1].exp2().unsqueeze(-1)
        state = state * decay + chunk_kg.transpose(-2, -1) @ tmp
    return state.transpose(-2, -1).contiguous()


def state_grad_summary_reference(
    qg: torch.Tensor,
    kg: torch.Tensor,
    w: torch.Tensor,
    dout: torch.Tensor,
    aqk: torch.Tensor,
    cumulative_gate: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Evaluate the augmented reverse-affine recurrence in eager FP32."""
    _, tokens, heads, key_dim = qg.shape
    value_dim = dout.shape[-1]
    pad = (-tokens) % 64
    if pad:
        padding = (0, 0, 0, 0, 0, pad)
        qg, kg, w, dout, aqk = (F.pad(tensor, padding) for tensor in (qg, kg, w, dout, aqk))
        cumulative_gate = torch.cat(
            (cumulative_gate, cumulative_gate[:, -1:].expand(-1, pad, -1, -1)),
            dim=1,
        )
        tokens += pad
    identity = torch.eye(key_dim, dtype=torch.float32, device=qg.device)
    state = torch.cat(
        (
            torch.zeros(heads, key_dim, value_dim, dtype=torch.float32, device=qg.device),
            identity.expand(heads, key_dim, key_dim),
        ),
        dim=-1,
    )
    for start in range(tokens - 64, -1, -64):
        stop = start + 64
        chunk_kg = kg[0, start:stop].transpose(0, 1).float()
        chunk_w = w[0, start:stop].transpose(0, 1).float()
        chunk_qg = qg[0, start:stop].transpose(0, 1).float()
        chunk_aqk = aqk[0, start:stop].transpose(0, 1).float()
        dout_augmented = torch.zeros(
            heads, 64, value_dim + key_dim, dtype=torch.float32, device=qg.device
        )
        dout_augmented[..., :value_dim] = dout[0, start:stop].transpose(0, 1).float()
        corrected = chunk_kg @ state + chunk_aqk.transpose(-2, -1) @ dout_augmented
        decay = cumulative_gate[0, stop - 1].exp2().unsqueeze(-1)
        state = (
            state * decay
            + scale * chunk_qg.transpose(-2, -1) @ dout_augmented
            - chunk_w.transpose(-2, -1) @ corrected
        )
    return state.transpose(-2, -1).contiguous()


def test_affine_summaries_support_int64_tensor_strides():
    """Compile wide-address kernels for realistic long-sequence batch strides."""
    qg, kg, w, u, dout, aqk, cumulative_gate = make_summary_inputs(19)

    expected_forward = build_state_summary(kg, w, u, cumulative_gate)
    expected_reverse = build_state_grad_summary(
        qg,
        kg,
        w,
        dout,
        aqk,
        cumulative_gate,
        128**-0.5,
    )
    actual_forward = build_state_summary(
        *(int64_stride_copy(tensor) for tensor in (kg, w, u, cumulative_gate))
    )
    actual_reverse = build_state_grad_summary(
        *(int64_stride_copy(tensor) for tensor in (qg, kg, w, dout, aqk, cumulative_gate)),
        128**-0.5,
    )

    torch.testing.assert_close(actual_forward, expected_forward, rtol=0, atol=0)
    torch.testing.assert_close(actual_reverse, expected_reverse, rtol=0, atol=0)


@pytest.mark.parametrize("budget", [1, 3])
def test_portable_affine_summaries_support_forced_int64_offsets(budget, monkeypatch):
    """Keep the portable single-item and split wide-address specializations executable."""
    plan_budget = Mock(return_value=budget)
    for module in (native_fwd, native_rev):
        monkeypatch.setattr(module, "is_sm100_kda_capability", lambda _capability: False)
        monkeypatch.setattr(module, "plan_work_budget", plan_budget)
    qg, kg, w, u, dout, aqk, cumulative_gate = make_summary_inputs(20, tokens=640)

    expected_forward = build_state_summary(kg, w, u, cumulative_gate)
    expected_reverse = build_state_grad_summary(
        qg,
        kg,
        w,
        dout,
        aqk,
        cumulative_gate,
        128**-0.5,
    )
    assert plan_budget.call_count == 2
    plan_budget.reset_mock()
    wide_offsets = Mock(return_value=True)
    monkeypatch.setattr(portable_fwd, "requires_int64_offsets", wide_offsets)
    monkeypatch.setattr(portable_rev, "requires_int64_offsets", wide_offsets)

    actual_forward = build_state_summary(kg, w, u, cumulative_gate)
    actual_reverse = build_state_grad_summary(
        qg,
        kg,
        w,
        dout,
        aqk,
        cumulative_gate,
        128**-0.5,
    )

    torch.testing.assert_close(actual_forward, expected_forward, rtol=0, atol=0)
    torch.testing.assert_close(actual_reverse, expected_reverse, rtol=0, atol=0)
    assert plan_budget.call_count == 2
    assert wide_offsets.call_count == 2


def test_affine_summaries_accept_misaligned_contiguous_storage():
    """Normalize ordinary storage-offset views before constructing TMA descriptors."""
    qg, kg, w, u, dout, aqk, cumulative_gate = make_summary_inputs(21)

    expected_forward = build_state_summary(kg, w, u, cumulative_gate)
    expected_reverse = build_state_grad_summary(
        qg,
        kg,
        w,
        dout,
        aqk,
        cumulative_gate,
        128**-0.5,
    )
    actual_forward = build_state_summary(
        *(misaligned_copy(tensor) for tensor in (kg, w, u, cumulative_gate))
    )
    actual_reverse = build_state_grad_summary(
        *(misaligned_copy(tensor) for tensor in (qg, kg, w, dout, aqk, cumulative_gate)),
        128**-0.5,
    )

    torch.testing.assert_close(actual_forward, expected_forward, rtol=0, atol=0)
    torch.testing.assert_close(actual_reverse, expected_reverse, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "tokens,heads",
    [(64, 1), (65, 2), (256, 3), (64, 8), (64, 12), (512, 2), (2048, 1)],
)
def test_affine_summary_matches_fp32_reference(dtype, tokens, heads):
    torch.manual_seed(23)
    shape = (1, tokens, heads, 128)
    kg = torch.randn(shape, dtype=dtype, device="cuda") / 16
    w = torch.randn_like(kg) / 16
    u = torch.randn_like(kg) / 16
    cumulative_gate = -torch.rand(shape, dtype=torch.float32, device="cuda") / 8

    actual = build_state_summary(kg, w, u, cumulative_gate)
    expected = state_summary_reference(kg, w, u, cumulative_gate)

    assert actual.shape == (heads, 256, 128)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "tokens,heads",
    [(64, 1), (65, 2), (256, 3), (64, 8), (64, 12), (512, 2), (2048, 1)],
)
def test_build_state_grad_summary_matches_fp32_reference(dtype, tokens, heads):
    torch.manual_seed(27)
    shape = (1, tokens, heads, 128)
    qg = torch.randn(shape, dtype=dtype, device="cuda") / 32
    kg = torch.randn_like(qg) / 32
    w = torch.randn_like(qg) / 32
    dout = torch.randn_like(qg) / 32
    aqk = torch.randn(1, tokens, heads, 64, dtype=dtype, device="cuda") / 32
    cumulative_gate = -torch.rand(shape, dtype=torch.float32, device="cuda") / 8
    scale = 128**-0.5

    actual = build_state_grad_summary(qg, kg, w, dout, aqk, cumulative_gate, scale)
    expected = state_grad_summary_reference(qg, kg, w, dout, aqk, cumulative_gate, scale)

    assert actual.shape == (heads, 256, 128)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_affine_summary_selector_variants_and_persistent_work():
    torch.manual_seed(28)
    shape = (1, 64, 32, 128)
    kg = torch.randn(shape, dtype=torch.bfloat16, device="cuda") / 32
    w = torch.randn_like(kg) / 32
    u = torch.randn_like(kg) / 32
    qg = torch.randn_like(kg) / 32
    dout = torch.randn_like(kg) / 32
    aqk = torch.randn(1, 64, 32, 64, dtype=torch.bfloat16, device="cuda") / 32
    cumulative_gate = -torch.rand(shape, dtype=torch.float32, device="cuda") / 8

    torch.testing.assert_close(
        build_state_summary(kg, w, u, cumulative_gate),
        state_summary_reference(kg, w, u, cumulative_gate),
        atol=2e-4,
        rtol=2e-4,
    )
    torch.testing.assert_close(
        build_state_grad_summary(qg, kg, w, dout, aqk, cumulative_gate, 128**-0.5),
        state_grad_summary_reference(qg, kg, w, dout, aqk, cumulative_gate, 128**-0.5),
        atol=2e-4,
        rtol=2e-4,
    )


@pytest.mark.usefixtures("summary_backend")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_reverse_summary_cancels_exactly_like_forward(dtype):
    """Reverse must split its FP32 state hi/lo like forward so exact cancellations survive.

    Chunk 0 zeroes transition entry (0, 0); chunk 1 then multiplies by ``-eps``, where
    ``1 + eps`` is FP32-exact but rounds to ``1`` in ``dtype``. Forward returns an exact
    zero; a reverse scan that narrows its state once returns ``eps`` instead.
    """
    eps = 2.0**-9 if dtype is torch.bfloat16 else 2.0**-12
    shape = (1, 128, 1, 128)
    zeros = torch.zeros(shape, dtype=dtype, device="cuda")
    kg = zeros.clone()
    w = zeros.clone()
    kg[0, 0, 0, 0] = 1
    w[0, 0, 0, 0] = 1
    kg[0, 64, 0, 0] = 1
    w[0, 64, 0, 0] = -eps
    aqk = torch.zeros(1, 128, 1, 64, dtype=dtype, device="cuda")
    cumulative_gate = torch.zeros(shape, dtype=torch.float32, device="cuda")

    expected = torch.zeros(1, 256, 128, dtype=torch.float32, device="cuda")
    expected[0, VALUE_DIM:] = torch.eye(128, device="cuda")
    expected[0, VALUE_DIM, 0] = 0

    torch.testing.assert_close(
        build_state_summary(kg, w, zeros, cumulative_gate), expected, rtol=0, atol=0
    )
    torch.testing.assert_close(
        build_state_grad_summary(zeros, kg, w, zeros, aqk, cumulative_gate, 1.0),
        expected,
        rtol=0,
        atol=0,
    )


@pytest.mark.usefixtures("summary_backend")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens,heads", [(256, 2), (2048, 1), (512, 16)])
def test_reverse_transition_is_adjoint_of_forward_transition(dtype, tokens, heads):
    """Bound ``<state @ R, cotangent> - <state, cotangent @ R_rev>`` for every pair.

    The operator norm of ``R - R_rev^T`` bounds that defect relative to the Frobenius
    norms of any state/cotangent pair. A reverse scan that narrows its state once lands
    at 5e-4 or worse on these shapes; the hi/lo scan stays below 1e-4.
    """
    qg, kg, w, u, dout, aqk, cumulative_gate = make_summary_inputs(
        37, dtype, tokens, heads, divisor=16
    )

    transition = build_state_summary(kg, w, u, cumulative_gate)[:, VALUE_DIM:]
    reverse_transition = build_state_grad_summary(
        qg, kg, w, dout, aqk, cumulative_gate, 128**-0.5
    )[:, VALUE_DIM:]

    adjoint_defect = torch.linalg.matrix_norm(
        transition - reverse_transition.transpose(-2, -1), ord=2
    )
    assert (adjoint_defect <= 2e-4).all(), adjoint_defect


@pytest.mark.parametrize(
    ("max_chunks", "work_tiles", "sm_count", "expected"),
    [
        pytest.param(512, 64, 148, 2, id="two-items"),
        pytest.param(512, 128, 148, 1, id="one-wave-already"),
        pytest.param(512, 256, 148, 1, id="oversubscribed"),
        pytest.param(512, 8, 148, 18, id="sm-bound"),
        pytest.param(3, 8, 148, 1, id="short-chain-stays-whole"),
        pytest.param(31, 8, 148, 1, id="just-below-two-items"),
        pytest.param(32, 8, 148, 2, id="first-split"),
    ],
)
def test_plan_work_budget_fills_one_wave_without_tiny_items(
    max_chunks, work_tiles, sm_count, expected
):
    budget = native_fwd.plan_work_budget(max_chunks, work_tiles, sm_count)
    assert budget == expected
    # A scan level costs more than several chunks, so no split may produce short items.
    assert budget == 1 or max_chunks // budget >= native_fwd.MIN_CHUNKS_PER_ITEM


WORK_LAYOUTS = [
    pytest.param([[0, 300], [300, 300], [300, 428], [428, 500], [500, 640]], 4, id="mixed"),
    pytest.param([[0, 32768]] + [[32768, 32768]] * 7, 18, id="one-long-seven-empty"),
    pytest.param([[0, 640]], 1, id="single-range-single-item"),
    pytest.param([[0, 64], [64, 128]], 18, id="more-budget-than-chunks"),
    pytest.param([[0, 100], [100, 30000], [30000, 32768]], 18, id="uneven"),
]


@pytest.mark.parametrize(("layout", "budget"), WORK_LAYOUTS)
def test_plan_work_items_tiles_every_range_in_order(layout, budget):
    """Items cover each range's chunks exactly once, contiguously and in chunk order."""
    bounds = torch.tensor(layout, dtype=torch.int32, device="cuda")
    work, range_ids = (t.cpu() for t in plan_work_items(bounds, budget))
    ranges = len(layout)
    assert work.shape == (budget + ranges, 4) and range_ids.shape == (budget + ranges,)
    assert torch.equal(work[range_ids == ranges], torch.zeros_like(work[range_ids == ranges]))
    total = sum(-(-(stop - start) // 64) for start, stop in layout)
    for range_id, (start, stop) in enumerate(layout):
        rows = work[range_ids == range_id]
        chunks = -(-(stop - start) // 64)
        positions = (range_ids == range_id).nonzero().flatten().tolist()
        assert (
            positions == list(range(positions[0], positions[0] + len(positions)))
            if positions
            else chunks == 0
        )
        assert (rows[:, 0] == start).all() and (rows[:, 3] == stop - start).all()
        begins, ends = rows[:, 1].tolist(), rows[:, 2].tolist()
        assert begins == [0, *ends[:-1]][: len(ends)] and (ends[-1] if ends else 0) == chunks
        # The flat split hands each range a share proportional to its length.
        assert len(rows) >= min(chunks, max(1, budget * chunks // total - 1))


@pytest.mark.parametrize(("layout", "budget"), WORK_LAYOUTS)
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_compose_work_items_matches_sequential_composition(layout, budget, reverse):
    """The segmented scan equals composing each range's items in (reverse) chunk order."""
    bounds = torch.tensor(layout, dtype=torch.int32, device="cuda")
    work, range_ids = plan_work_items(bounds, budget)
    ranges = len(layout)
    generator = torch.Generator(device="cuda").manual_seed(5)
    partials = torch.randn(work.shape[0], 2, 256, 128, device="cuda", generator=generator) / 16
    partials[:, :, VALUE_DIM:] += torch.eye(128, device="cuda")
    identity = torch.cat(
        (torch.zeros(2, 128, 128), torch.eye(128).expand(2, -1, -1)), dim=1
    ).cuda()

    composed = compose_work_items(partials, range_ids, ranges, reverse=reverse)
    for range_id in range(ranges):
        items = [partials[i].double() for i in range(work.shape[0]) if range_ids[i] == range_id]
        if reverse:
            items.reverse()
        expected = reduce(compose_summaries, items, identity.double()).float()
        # IEEE fp32 dots over up to ``budget`` chained maps: ~1e-5, far inside the bf16 summary
        # error and two decades inside what TF32 operands would produce.
        torch.testing.assert_close(composed[range_id], expected, atol=1e-4, rtol=1e-4)
        assert_relative_rms_within(
            composed[range_id],
            expected,
            f"range {range_id} compose",
            max_eps=64,
            source_dtype=torch.float32,
        )


def test_compose_work_items_ignores_the_float32_matmul_mode():
    """The fold must not pick up TF32 from ``torch.set_float32_matmul_precision("high")``."""
    generator = torch.Generator(device="cuda").manual_seed(6)
    partials = torch.randn(4, 2, 256, 128, device="cuda", generator=generator) / 16
    partials[:, :, VALUE_DIM:] += torch.eye(128, device="cuda")
    range_ids = torch.zeros(4, dtype=torch.int64, device="cuda")
    exact = reduce(compose_summaries, partials.double().unbind(0)).float()
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        folded = compose_work_items(partials, range_ids, 1, reverse=False)[0]
    finally:
        torch.set_float32_matmul_precision(precision)
    # TF32 rounds operands to 10 mantissa bits (~1e-3 relative); IEEE fp32 dots stay at ~1e-5.
    torch.testing.assert_close(folded, exact, atol=1e-4, rtol=1e-4)
    assert_relative_rms_within(
        folded, exact, "compose under matmul mode", max_eps=64, source_dtype=torch.float32
    )


@pytest.mark.usefixtures("summary_backend")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_split_summaries_match_single_item_scans(dtype, monkeypatch):
    """Splitting the chunk chain across idle SMs must only reorder FP32 composition."""
    qg, kg, w, u, dout, aqk, cumulative_gate = make_summary_inputs(
        43, dtype, tokens=2048, heads=1, divisor=16
    )
    reverse_summary = partial(build_state_grad_summary, scale=128**-0.5)
    # Exercise splitting even on small devices where the production budget is one.
    plan_budget = Mock(return_value=2)
    monkeypatch.setattr(native_fwd, "plan_work_budget", plan_budget)
    monkeypatch.setattr(native_rev, "plan_work_budget", plan_budget)
    split = (
        build_state_summary(kg, w, u, cumulative_gate),
        reverse_summary(qg, kg, w, dout, aqk, cumulative_gate),
    )
    assert plan_budget.call_count == 2

    plan_budget.reset_mock()
    plan_budget.return_value = 1
    unsplit = (
        build_state_summary(kg, w, u, cumulative_gate),
        reverse_summary(qg, kg, w, dout, aqk, cumulative_gate),
    )
    assert plan_budget.call_count == 2
    for name, actual, expected in zip(("forward", "reverse"), split, unsplit, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        assert_relative_rms_within(
            actual, expected, f"split {name} summary", max_eps=64, source_dtype=torch.float32
        )


@pytest.mark.usefixtures("summary_backend")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("boundaries", [(256,), (64, 320), (128, 256, 384)])
def test_affine_summaries_are_invariant_to_shard_boundaries(dtype, boundaries):
    """Composing per-shard summaries must match the unsharded scan in both directions."""
    qg, kg, w, u, dout, aqk, cumulative_gate = make_summary_inputs(
        41, dtype, tokens=512, heads=2, divisor=16
    )
    reverse_summary = partial(build_state_grad_summary, scale=128**-0.5)

    forward_shards = shard_summaries(build_state_summary, (kg, w, u, cumulative_gate), boundaries)
    torch.testing.assert_close(
        reduce(compose_summaries, forward_shards),
        build_state_summary(kg, w, u, cumulative_gate),
        atol=2e-5,
        rtol=2e-5,
    )
    # The cotangent flows through the last shard first.
    reverse_shards = shard_summaries(
        reverse_summary, (qg, kg, w, dout, aqk, cumulative_gate), boundaries
    )
    torch.testing.assert_close(
        reduce(compose_summaries, reversed(reverse_shards)),
        reverse_summary(qg, kg, w, dout, aqk, cumulative_gate),
        atol=2e-5,
        rtol=2e-5,
    )


def test_affine_summaries_reject_torch_export_instead_of_emitting_empty_outputs():
    """Fail capture explicitly until the launchers become registered graph operators."""

    class ForwardSummary(torch.nn.Module):
        def forward(self, kg, w, u, cumulative_gate):
            return build_state_summary(kg, w, u, cumulative_gate)

    class ReverseSummary(torch.nn.Module):
        def forward(self, qg, kg, w, dout, aqk, cumulative_gate):
            return build_state_grad_summary(
                qg,
                kg,
                w,
                dout,
                aqk,
                cumulative_gate,
                128**-0.5,
            )

    shape = (1, 64, 1, 128)
    x = torch.zeros(shape, dtype=torch.bfloat16, device="cuda")
    gate = torch.zeros(shape, dtype=torch.float32, device="cuda")
    aqk = torch.zeros(1, 64, 1, 64, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(TypeError, match="does not support torch.export"):
        torch.export.export(ForwardSummary(), (x, x, x, gate), strict=False)
    with pytest.raises(TypeError, match="does not support torch.export"):
        torch.export.export(ReverseSummary(), (x, x, x, x, aqk, gate), strict=False)


def test_build_state_summary_cuda_graph_replay():
    torch.manual_seed(29)
    shape = (1, 128, 2, 128)
    kg = torch.randn(shape, dtype=torch.bfloat16, device="cuda") / 16
    w = torch.randn_like(kg) / 16
    u = torch.randn_like(kg) / 16
    cumulative_gate = -torch.rand(shape, dtype=torch.float32, device="cuda") / 8

    build_state_summary(kg, w, u, cumulative_gate)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = build_state_summary(kg, w, u, cumulative_gate)

    u.mul_(0.5)
    expected = state_summary_reference(kg, w, u, cumulative_gate)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_build_state_grad_summary_cuda_graph_replay():
    torch.manual_seed(31)
    shape = (1, 128, 2, 128)
    qg = torch.randn(shape, dtype=torch.bfloat16, device="cuda") / 32
    kg = torch.randn_like(qg) / 32
    w = torch.randn_like(qg) / 32
    dout = torch.randn_like(qg) / 32
    aqk = torch.randn(1, 128, 2, 64, dtype=torch.bfloat16, device="cuda") / 32
    cumulative_gate = -torch.rand(shape, dtype=torch.float32, device="cuda") / 8
    scale = 128**-0.5

    build_state_grad_summary(qg, kg, w, dout, aqk, cumulative_gate, scale)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = build_state_grad_summary(qg, kg, w, dout, aqk, cumulative_gate, scale)

    dout.mul_(0.5)
    expected = state_grad_summary_reference(qg, kg, w, dout, aqk, cumulative_gate, scale)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


RANGE_BOUNDS = [
    (0, 300),  # partial tail chunk, five chunks
    (300, 300),  # empty: the identity
    (300, 428),  # two whole chunks starting mid-stream
    (428, 500),  # one chunk plus an 8-token tail, ending inside the stream
    (500, 640),  # to the end of the stream
]


def reference_summaries(tensors, start: int, stop: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager FP32 forward and reverse maps of one token range, computed on the slice."""
    qg, kg, w, u, dout, aqk, cumulative_gate = (t[:, start:stop] for t in tensors)
    return (
        state_summary_reference(kg, w, u, cumulative_gate),
        state_grad_summary_reference(qg, kg, w, dout, aqk, cumulative_gate, 128**-0.5),
    )


@pytest.mark.usefixtures("summary_backend")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_range_summaries_match_per_range_launches(dtype):
    """Device ``bounds`` reproduce slicing each range on the host, tails and empties included."""
    tensors = make_summary_inputs(41, dtype, tokens=640, heads=2)
    qg, kg, w, u, dout, aqk, cumulative_gate = tensors
    bounds = torch.tensor(RANGE_BOUNDS, dtype=torch.int32, device="cuda")
    scale = 128**-0.5

    forward = build_state_summaries(kg, w, u, cumulative_gate, bounds)
    reverse = build_state_grad_summaries(qg, kg, w, dout, aqk, cumulative_gate, scale, bounds)
    assert forward.shape == reverse.shape == (len(RANGE_BOUNDS), 2, 256, 128)

    identity = torch.cat((torch.zeros(2, 128, 128), torch.eye(128).expand(2, -1, -1)), dim=1)
    for index, (start, stop) in enumerate(RANGE_BOUNDS):
        if start == stop:
            torch.testing.assert_close(forward[index].cpu(), identity, atol=0, rtol=0)
            torch.testing.assert_close(reverse[index].cpu(), identity, atol=0, rtol=0)
            continue
        sliced = [t[:, start:stop] for t in tensors]
        # Both launches pick one segment at these lengths, so the results are bitwise equal.
        expected_forward = build_state_summary(sliced[1], sliced[2], sliced[3], sliced[6])
        expected_reverse = build_state_grad_summary(
            sliced[0], sliced[1], sliced[2], sliced[4], sliced[5], sliced[6], scale
        )
        torch.testing.assert_close(forward[index], expected_forward, atol=0, rtol=0)
        torch.testing.assert_close(reverse[index], expected_reverse, atol=0, rtol=0)
        # ...and exact against the eager FP32 recurrence on the slice, in both directions.
        for name, actual, expected in zip(
            ("forward", "reverse"),
            (forward[index], reverse[index]),
            reference_summaries(tensors, start, stop),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            assert_relative_rms_within(
                actual,
                expected,
                f"{name} [{start}, {stop})",
                max_eps=64,
                source_dtype=torch.float32,
            )


@pytest.mark.usefixtures("summary_backend")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_range_tails_ignore_the_tokens_that_follow(dtype):
    """A partial last chunk over-reads the next tokens; scaled-up neighbours must not leak in.

    The single-range launch sees TMA zero fill past its slice instead of those tokens, so the two
    must agree bitwise; the neighbours are finite, which the kernels require (``0 * inf`` is NaN).
    """
    tensors = make_summary_inputs(41, dtype, tokens=640, heads=2)
    qg, kg, w, u, dout, aqk, cumulative_gate = tensors
    for tensor in (qg, kg, w, u, dout, aqk):
        tensor[:, 300:364] *= 64  # never inside a range below
    bounds = torch.tensor([[0, 300], [172, 300]], dtype=torch.int32, device="cuda")
    scale = 128**-0.5

    forward = build_state_summaries(kg, w, u, cumulative_gate, bounds)
    reverse = build_state_grad_summaries(qg, kg, w, dout, aqk, cumulative_gate, scale, bounds)
    for index, (start, stop) in enumerate(bounds.tolist()):
        sliced = [t[:, start:stop] for t in tensors]
        expected_forward = build_state_summary(sliced[1], sliced[2], sliced[3], sliced[6])
        expected_reverse = build_state_grad_summary(
            sliced[0], sliced[1], sliced[2], sliced[4], sliced[5], sliced[6], scale
        )
        torch.testing.assert_close(forward[index], expected_forward, atol=0, rtol=0)
        torch.testing.assert_close(reverse[index], expected_reverse, atol=0, rtol=0)
        for name, actual, expected in zip(
            ("forward", "reverse"),
            (forward[index], reverse[index]),
            reference_summaries(tensors, start, stop),
            strict=True,
        ):
            torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
            assert_relative_rms_within(
                actual,
                expected,
                f"{name} [{start}, {stop})",
                max_eps=64,
                source_dtype=torch.float32,
            )


@pytest.mark.usefixtures("summary_backend")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_range_summaries_survive_a_tiny_work_budget(dtype, monkeypatch):
    """Three items over seven ranges: cuts inside ranges and one-chunk ranges compose."""
    tensors = make_summary_inputs(47, dtype, tokens=640, heads=2)
    qg, kg, w, u, dout, aqk, cumulative_gate = tensors
    plan_budget = Mock(return_value=3)
    monkeypatch.setattr(native_fwd, "plan_work_budget", plan_budget)
    monkeypatch.setattr(native_rev, "plan_work_budget", plan_budget)
    ranges = [*RANGE_BOUNDS, (500, 501), (0, 65)]
    bounds = torch.tensor(ranges, dtype=torch.int32, device="cuda")

    forward = build_state_summaries(kg, w, u, cumulative_gate, bounds)
    reverse = build_state_grad_summaries(qg, kg, w, dout, aqk, cumulative_gate, 128**-0.5, bounds)
    assert plan_budget.call_count == 2
    for index, (start, stop) in enumerate(ranges):
        for name, actual, expected in zip(
            ("forward", "reverse"),
            (forward[index], reverse[index]),
            reference_summaries(tensors, start, stop),
            strict=True,
        ):
            tolerance = 0 if start == stop else 2e-4
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            assert_relative_rms_within(
                actual,
                expected,
                f"{name} [{start}, {stop})",
                max_eps=64,
                source_dtype=torch.float32,
            )


@pytest.mark.usefixtures("summary_backend")
@pytest.mark.parametrize("budget", [1, 3])
def test_range_summaries_replay_across_layouts_in_one_cuda_graph(budget, monkeypatch):
    """One captured graph serves changed bounds, including all-empty padded work tables."""
    tensors = make_summary_inputs(43, tokens=640, heads=2)
    qg, kg, w, u, dout, aqk, cumulative_gate = tensors
    plan_budget = Mock(return_value=budget)
    monkeypatch.setattr(native_fwd, "plan_work_budget", plan_budget)
    monkeypatch.setattr(native_rev, "plan_work_budget", plan_budget)
    scale = 128**-0.5
    bounds = torch.tensor([[0, 256], [256, 640], [640, 640]], dtype=torch.int32, device="cuda")

    def launch():
        return (
            build_state_summaries(kg, w, u, cumulative_gate, bounds),
            build_state_grad_summaries(qg, kg, w, dout, aqk, cumulative_gate, scale, bounds),
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = launch()

    assert plan_budget.call_count == 4
    for layout in (
        [[0, 256], [256, 640], [640, 640]],
        [[0, 100], [100, 300], [300, 640]],
        [[0, 0], [300, 300], [640, 640]],
        [[0, 65], [65, 65], [65, 640]],
    ):
        bounds.copy_(torch.tensor(layout, dtype=torch.int32, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        for actual, expected in zip(captured, launch(), strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for index, (start, stop) in enumerate(layout):
            for name, actual, expected in zip(
                ("forward", "reverse"),
                (captured[0][index], captured[1][index]),
                reference_summaries(tensors, start, stop),
                strict=True,
            ):
                tolerance = 0 if start == stop else 2e-4
                torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                assert_relative_rms_within(
                    actual,
                    expected,
                    f"replayed {name} [{start}, {stop})",
                    max_eps=64,
                    source_dtype=torch.float32,
                )
