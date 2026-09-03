"""Tests for the architecture-routed delta-rule affine-summary recurrence."""

from __future__ import annotations

from functools import partial, reduce
from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass")

import attn_gym.linear._delta_rule.cute.affine_summary_fwd as native_fwd
import attn_gym.linear._delta_rule.cute.affine_summary_rev as native_rev
import attn_gym.linear._delta_rule.triton.affine_summary_fwd as portable_fwd
import attn_gym.linear._delta_rule.triton.affine_summary_rev as portable_rev
from attn_gym.linear._delta_rule.cute import build_state_grad_summary, build_state_summary
from attn_gym.linear.kda.constants import is_sm100_kda_capability
from attn_gym.testing.kda import assert_relative_rms_within

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


def test_portable_affine_summaries_support_forced_int64_offsets(monkeypatch):
    """Keep the portable launchers' wide-address specialization executable."""
    capability = torch.cuda.get_device_capability()
    if is_sm100_kda_capability(capability):
        pytest.skip("SM100 uses the native CuTeDSL affine summaries")

    qg, kg, w, u, dout, aqk, cumulative_gate = make_summary_inputs(20)

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
    monkeypatch.setattr(portable_fwd, "requires_int64_offsets", lambda *_tensors: True)
    monkeypatch.setattr(portable_rev, "requires_int64_offsets", lambda *_tensors: True)

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
    ("num_chunks", "work_tiles", "sm_count", "expected"),
    [
        pytest.param(512, 64, 148, (2, 256), id="two-segments"),
        pytest.param(512, 128, 148, (1, 512), id="one-wave-already"),
        pytest.param(512, 256, 148, (1, 512), id="oversubscribed"),
        pytest.param(512, 8, 148, (18, 29), id="sm-bound"),
        pytest.param(3, 8, 148, (1, 3), id="short-chain-stays-whole"),
        pytest.param(31, 8, 148, (1, 31), id="just-below-two-segments"),
        pytest.param(32, 8, 148, (2, 16), id="first-split"),
        pytest.param(70, 8, 32, (4, 18), id="ragged-tail"),
        pytest.param(65, 8, 32, (4, 17), id="rounding-drops-a-segment"),
    ],
)
def test_plan_segments_fills_one_wave_without_empty_or_tiny_segments(
    num_chunks, work_tiles, sm_count, expected
):
    segments, chunks_per_segment = native_fwd.plan_segments(num_chunks, work_tiles, sm_count)
    assert (segments, chunks_per_segment) == expected
    assert (segments - 1) * chunks_per_segment < num_chunks <= segments * chunks_per_segment
    # A fold level costs more than several chunks, so no split may produce short segments.
    assert segments == 1 or chunks_per_segment >= native_fwd.MIN_CHUNKS_PER_SEGMENT


@pytest.mark.parametrize("segments", [1, 3, 5, 8])
def test_compose_segment_summaries_matches_sequential_composition(segments):
    """The pairwise fold, including its odd tail, equals left-to-right composition."""
    generator = torch.Generator().manual_seed(5)
    summaries = torch.randn(segments, 2, 256, 128, generator=generator) / 16
    summaries[:, :, VALUE_DIM:] += torch.eye(128)
    expected = reduce(compose_summaries, summaries.unbind(0))
    torch.testing.assert_close(
        native_fwd.compose_segment_summaries(summaries), expected, atol=1e-5, rtol=1e-5
    )


def test_compose_segment_summaries_ignores_the_float32_matmul_mode():
    """The fold must not pick up TF32 from ``torch.set_float32_matmul_precision("high")``."""
    generator = torch.Generator(device="cuda").manual_seed(6)
    summaries = torch.randn(4, 2, 256, 128, device="cuda", generator=generator) / 16
    summaries[:, :, VALUE_DIM:] += torch.eye(128, device="cuda")
    exact = reduce(compose_summaries, summaries.double().unbind(0)).float()
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        folded = native_fwd.compose_segment_summaries(summaries)
    finally:
        torch.set_float32_matmul_precision(precision)
    # TF32 rounds operands to 10 mantissa bits (~1e-3 relative); an FP64 fold stays within a few
    # FP32 ulps of the exact composition.
    torch.testing.assert_close(folded, exact, atol=1e-6, rtol=1e-6)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_sm100_kda_capability(torch.cuda.get_device_capability()),
    reason="segmented launches are specific to the native SM100 kernels",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_segmented_summaries_match_single_segment_scans(dtype, monkeypatch):
    """Splitting the chunk chain across idle SMs must only reorder FP32 composition."""
    qg, kg, w, u, dout, aqk, cumulative_gate = make_summary_inputs(
        43, dtype, tokens=2048, heads=1, divisor=16
    )
    reverse_summary = partial(build_state_grad_summary, scale=128**-0.5)
    segmented = (
        build_state_summary(kg, w, u, cumulative_gate),
        reverse_summary(qg, kg, w, dout, aqk, cumulative_gate),
    )

    plans = []
    plan_segments = native_fwd.plan_segments

    def single_segment(num_chunks, work_tiles, sm_count):
        plans.append(plan_segments(num_chunks, work_tiles, sm_count))
        return 1, num_chunks

    monkeypatch.setattr(native_fwd, "plan_segments", single_segment)
    monkeypatch.setattr(native_rev, "plan_segments", single_segment)
    unsegmented = (
        build_state_summary(kg, w, u, cumulative_gate),
        reverse_summary(qg, kg, w, dout, aqk, cumulative_gate),
    )
    assert all(segments > 1 for segments, _ in plans), plans
    for name, actual, expected in zip(("forward", "reverse"), segmented, unsegmented, strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        assert_relative_rms_within(
            actual, expected, f"segmented {name} summary", max_eps=64, source_dtype=torch.float32
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
    with pytest.raises(TypeError, match="build_state_summary does not support torch.export"):
        torch.export.export(ForwardSummary(), (x, x, x, gate), strict=False)
    with pytest.raises(TypeError, match="build_state_grad_summary does not support torch.export"):
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
