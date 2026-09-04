"""Staged KDA/GDN primitives and the context-parallel routing simulated in one process."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear.context_parallel import (
    ContextParallelPlan,
    compose_entry_states,
    compose_exit_cotangents,
    grad_summary_slots,
    summary_slots,
)
from attn_gym.linear.gdn import chunk_gdn
from attn_gym.linear.gdn.stages import chunk_gdn_prepare, chunk_gdn_prepare_backward
from attn_gym.linear.kda import chunk_kda, context_parallel_kda
from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward
from attn_gym.linear.state_summary import compose_summaries, merge_state, neutral_summary
from attn_gym.testing.gdn import make_gdn_test_inputs
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    assert_relative_rms_within,
    make_kda_test_inputs,
)

HEAD_DIM = 128

requires_kda_target = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8),
    reason="fused delta-rule kernels require CUDA capability 8.0+",
)


Inputs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class Op(NamedTuple):
    """One delta-rule variant: its public op, staged entry points, and shared input factory."""

    chunk: Callable[..., tuple[torch.Tensor, torch.Tensor | None]]
    reference: Callable[..., tuple[torch.Tensor, torch.Tensor | None]]
    prepare: Callable[..., object]
    prepare_backward: Callable[..., object]
    factory: Callable[..., Inputs]  # (tokens, *, key_heads, value_heads, seed, dtype)
    key_heads: int
    value_heads: int

    def make_inputs(self, tokens: int, seed: int, dtype: torch.dtype = torch.bfloat16) -> Inputs:
        return self.factory(
            tokens, key_heads=self.key_heads, value_heads=self.value_heads, seed=seed, dtype=dtype
        )


def relative_rms_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Relative RMS error against an FP32 oracle."""
    return ((actual.float() - reference).square().mean() / reference.square().mean()).sqrt().item()


def kda_inputs(tokens: int, *, key_heads: int, value_heads: int, seed: int, dtype) -> Inputs:
    assert key_heads == value_heads, "KDA has no grouped key heads"
    return make_kda_test_inputs(
        tokens, heads=value_heads, seed=seed, dtype=dtype, normalize_qk=True
    )


def gdn_inputs(tokens: int, *, key_heads: int, value_heads: int, seed: int, dtype) -> Inputs:
    return make_gdn_test_inputs(
        tokens, key_heads=key_heads, value_heads=value_heads, seed=seed, dtype=dtype
    )[:5]


KDA = Op(
    chunk_kda,
    partial(chunk_kda, impl="reference"),
    chunk_kda_prepare,
    chunk_kda_prepare_backward,
    kda_inputs,
    key_heads=2,
    value_heads=2,
)
GDN = Op(
    partial(chunk_gdn, impl="fused"),
    partial(chunk_gdn, impl="reference"),
    chunk_gdn_prepare,
    chunk_gdn_prepare_backward,
    gdn_inputs,
    key_heads=2,
    value_heads=2,
)
MEGA = {"backend": "mega"}
requires_mega = pytest.mark.skipif(
    importlib.util.find_spec("cutlass.experimental") is None
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Mega requires CuTeDSL>=4.7 on SM100/SM103",
)
KDA_MEGA = KDA._replace(
    chunk=partial(chunk_kda, kernel_options=MEGA),
    prepare=partial(chunk_kda_prepare, kernel_options=MEGA),
)
op_param = pytest.mark.parametrize(
    "op",
    [
        pytest.param(KDA, id="kda"),
        pytest.param(KDA_MEGA, id="kda-mega", marks=requires_mega),
        pytest.param(GDN, id="gdn"),
        pytest.param(GDN._replace(key_heads=1), id="gdn-gqa"),
    ],
)


def test_summary_algebra_composes_like_sequential_merges():
    generator = torch.Generator().manual_seed(0)
    first = torch.randn(2, 6, 4, generator=generator)
    then = torch.randn(2, 6, 4, generator=generator)
    state = torch.randn(2, 2, 4, generator=generator)

    sequential = merge_state(merge_state(state, first), then)
    composed = compose_summaries(first, then)
    torch.testing.assert_close(merge_state(state, composed), sequential)
    # Spelled out: (A0, B0) then (A1, B1) is (A0 @ A1, B0 @ A1 + B1), packed [bias; transition].
    bias, transition = first[:, :2], first[:, 2:]
    torch.testing.assert_close(
        composed, torch.cat((bias @ then[:, 2:] + then[:, :2], transition @ then[:, 2:]), dim=1)
    )

    identity = neutral_summary(2, 2, 4, device="cpu")
    torch.testing.assert_close(merge_state(state, identity), state)
    torch.testing.assert_close(compose_summaries(identity, first), first)
    torch.testing.assert_close(compose_summaries(first, identity), first)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 is a CUDA matmul mode")
def test_summary_algebra_ignores_the_float32_matmul_mode():
    """``set_float32_matmul_precision("high")`` must not round state routing to TF32."""
    generator = torch.Generator(device="cuda").manual_seed(7)
    first = torch.randn(2, 2 * HEAD_DIM, HEAD_DIM, device="cuda", generator=generator) / 16
    then = torch.randn(2, 2 * HEAD_DIM, HEAD_DIM, device="cuda", generator=generator) / 16
    state = torch.randn(2, HEAD_DIM, HEAD_DIM, device="cuda", generator=generator)
    exact_state = (state.double() @ then[:, HEAD_DIM:].double() + then[:, :HEAD_DIM]).float()
    exact_composed = torch.cat(
        (
            first[:, :HEAD_DIM].double() @ then[:, HEAD_DIM:].double() + then[:, :HEAD_DIM],
            first[:, HEAD_DIM:].double() @ then[:, HEAD_DIM:].double(),
        ),
        dim=1,
    ).float()

    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        merged = merge_state(state, then)
        composed = compose_summaries(first, then)
    finally:
        torch.set_float32_matmul_precision(precision)
    # TF32 keeps 10 mantissa bits (~1e-3 relative); an FP64 product stays within FP32 rounding.
    torch.testing.assert_close(merged, exact_state, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(composed, exact_composed, atol=1e-6, rtol=1e-6)


@requires_kda_target
@op_param
def test_prepare_run_matches_chunk_op_with_packed_initial_state(op):
    """``prepare`` + ``run`` is the public op's own kernel sequence, so results are bitwise equal."""
    q, k, v, gate, beta = op.make_inputs(320, seed=1)
    cu_seqlens = torch.tensor([0, 100, 320], dtype=torch.int32, device="cuda")
    initial_state = torch.randn(2, op.value_heads, HEAD_DIM, HEAD_DIM, device="cuda")

    expected, expected_state = op.chunk(
        q, k, v, gate, beta, initial_state, cu_seqlens=cu_seqlens, output_final_state=True
    )
    prepared = op.prepare(q, k, v, gate, beta, cu_seqlens=cu_seqlens)
    output, final_state = prepared.run(initial_state, output_final_state=True)

    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(final_state, expected_state, atol=0, rtol=0)

    # A head slice of a larger state buffer has a unit key stride but is not contiguous.
    wide = torch.randn(2, 2 * op.value_heads, HEAD_DIM, HEAD_DIM, device="cuda")
    sliced = wide[:, ::2]
    assert not sliced.is_contiguous()
    expected, expected_state = op.chunk(
        q, k, v, gate, beta, sliced.contiguous(), cu_seqlens=cu_seqlens, output_final_state=True
    )
    output, final_state = prepared.run(sliced, output_final_state=True)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    torch.testing.assert_close(final_state, expected_state, atol=0, rtol=0)


@requires_kda_target
@op_param
@pytest.mark.parametrize("tokens", [128, 70], ids=["complete-chunks", "partial-tail"])
def test_prepare_run_matches_chunk_op_unpacked_with_strided_qkv(op, tokens):
    """Unpacked inputs and last-dim-strided Q/K/V follow the same normalization as the op."""
    q, k, v, gate, beta = op.make_inputs(tokens, seed=6)
    q, k, v = (torch.stack((tensor, tensor), dim=-1)[..., 0] for tensor in (q, k, v))
    assert q.stride(-1) == 2

    # The fused ops accept either layout and the stages normalize strided inputs themselves; the
    # public Mega op rejects strides, so every reference gets compact copies.
    expected, _ = op.chunk(q.contiguous(), k.contiguous(), v.contiguous(), gate, beta)
    output, final_state = op.prepare(q, k, v, gate, beta).run()

    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert final_state is None


@requires_kda_target
@op_param
def test_state_summary_matches_zero_and_identity_probes(op):
    """``final(H) = H @ A + B`` for every H, so zero and identity probes recover ``[B; A]``."""
    q, k, v, gate, beta = op.make_inputs(200, seed=2)
    # Model-range gates forget everything over 72 tokens; keep the transition above rounding.
    gate = gate / 50
    cu_seqlens = torch.tensor([0, 72, 200], dtype=torch.int32, device="cuda")
    prepared = op.prepare(q, k, v, gate, beta, cu_seqlens=cu_seqlens)

    zero = torch.zeros(2, op.value_heads, HEAD_DIM, HEAD_DIM, device="cuda")
    identity = torch.eye(HEAD_DIM, device="cuda").expand_as(zero).contiguous()

    def probe(run, *inputs):
        _, bias = run(*inputs, zero, cu_seqlens=cu_seqlens, output_final_state=True)
        _, shifted = run(*inputs, identity, cu_seqlens=cu_seqlens, output_final_state=True)
        return bias, shifted - bias

    fused = probe(op.chunk, q, k, v, gate, beta)
    oracle = probe(op.reference, q.float(), k.float(), v.float(), gate, beta)
    for index, (start, stop) in enumerate(((0, 72), (72, 200))):
        summary = prepared.state_summary(start, stop)
        assert summary.shape == (op.value_heads, 2 * HEAD_DIM, HEAD_DIM)
        assert_summary_parts_match(summary, fused, oracle, index)


def assert_summary_parts_match(summary, fused, oracle, index: int) -> None:
    """Check ``[bias; transition]`` against probes of the fused op and the FP32 oracle."""
    for name, part, low, high in zip(
        ("bias", "transition"), summary.split(HEAD_DIM, dim=1), fused, oracle, strict=True
    ):
        assert_matches_low_precision_reference(part, high[index], low[index], name)
        assert_relative_rms_within(part, high[index], name, max_eps=1.0)


@requires_kda_target
@op_param
@pytest.mark.parametrize("scale", [None, 0.5], ids=["default-scale", "custom-scale"])
def test_prepare_backward_run_matches_chunk_op_gradients(op, scale):
    """The staged backward is the op's own kernel sequence, so all six gradients are bitwise."""
    inputs = op.make_inputs(320, seed=7)
    cu_seqlens = torch.tensor([0, 100, 320], dtype=torch.int32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(8)
    initial_state = torch.randn(
        2, op.value_heads, HEAD_DIM, HEAD_DIM, device="cuda", generator=generator
    )
    d_final_state = torch.randn(initial_state.shape, device="cuda", generator=generator)

    leaves = tuple(tensor.detach().clone().requires_grad_() for tensor in (*inputs, initial_state))
    output, final_state = op.chunk(
        *leaves, cu_seqlens=cu_seqlens, output_final_state=True, scale=scale
    )
    d_output = torch.randn(output.shape, device="cuda", generator=generator).to(output.dtype)
    expected = torch.autograd.grad((output, final_state), leaves, (d_output, d_final_state))

    prepared = op.prepare(*inputs, cu_seqlens=cu_seqlens, scale=scale)
    prepared.run(initial_state, output_final_state=True)
    grads = op.prepare_backward(prepared.saved, d_output, initial_state, scale=prepared.scale)
    actual = grads.run(d_final_state)

    assert len(actual) == 6
    for name, got, want in zip(
        ("dq", "dk", "dv", "dgate", "dbeta", "d_initial_state"), actual, expected, strict=True
    ):
        torch.testing.assert_close(
            got, want, atol=0, rtol=0, msg=lambda m, name=name: f"{name}: {m}"
        )


@requires_kda_target
@op_param
def test_state_grad_summary_matches_zero_and_identity_probes(op):
    """``d_entry = d_exit @ R + C``, so zero and identity exit cotangents recover ``[C; R]``."""
    q, k, v, gate, beta = op.make_inputs(200, seed=9)
    gate = gate / 50  # As in the forward probe: keep the transition above rounding.
    cu_seqlens = torch.tensor([0, 72, 200], dtype=torch.int32, device="cuda")
    zero = torch.zeros(2, op.value_heads, HEAD_DIM, HEAD_DIM, device="cuda")
    identity = torch.eye(HEAD_DIM, device="cuda").expand_as(zero).contiguous()
    d_output = torch.randn(
        v.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(10)
    )

    def probe(run, *inputs):
        def d_entry_state(d_exit_state):
            entry = zero.clone().requires_grad_()
            output, final_state = run(
                *inputs, entry, cu_seqlens=cu_seqlens, output_final_state=True
            )
            (grad,) = torch.autograd.grad(
                (output, final_state), entry, (d_output.to(output.dtype), d_exit_state)
            )
            return grad

        bias = d_entry_state(zero)
        return bias, d_entry_state(identity) - bias

    fused = probe(op.chunk, q, k, v, gate, beta)
    oracle = probe(op.reference, q.float(), k.float(), v.float(), gate, beta)
    prepared = op.prepare(q, k, v, gate, beta, cu_seqlens=cu_seqlens)
    prepared.run(zero, output_final_state=True)
    grads = op.prepare_backward(prepared.saved, d_output.to(v.dtype), zero, scale=prepared.scale)
    for index, (start, stop) in enumerate(((0, 72), (72, 200))):
        summary = grads.state_grad_summary(start, stop)
        assert summary.shape == (op.value_heads, 2 * HEAD_DIM, HEAD_DIM)
        assert_summary_parts_match(summary, fused, oracle, index)


@requires_kda_target
@op_param
def test_summaries_are_exact_over_whole_chunks_of_one_subsequence(op):
    """NOTE [Summary ranges are whole chunks of one subsequence]: interior chunk runs compose."""
    q, k, v, gate, beta = op.make_inputs(520, seed=11)
    gate = gate / 50
    # Subsequences [0, 72) and [72, 520); the second has chunk boundaries at 72 + 64 * i.
    cu_seqlens = torch.tensor([0, 72, 520], dtype=torch.int32, device="cuda")
    d_output = torch.randn(
        v.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(12)
    )
    zero = torch.zeros(2, op.value_heads, HEAD_DIM, HEAD_DIM, device="cuda")
    prepared = op.prepare(q, k, v, gate, beta, cu_seqlens=cu_seqlens)
    prepared.run(zero, output_final_state=True)
    grads = op.prepare_backward(prepared.saved, d_output.to(v.dtype), zero, scale=prepared.scale)

    one = torch.zeros(1, op.value_heads, HEAD_DIM, HEAD_DIM, device="cuda")
    identity = torch.eye(HEAD_DIM, device="cuda").expand_as(one).contiguous()

    def oracle(start, stop):
        """Run the FP32 reference on the token range as a sequence of its own and probe it."""
        sliced = [tensor[:, start:stop] for tensor in (q, k, v, gate, beta)]
        inputs = [tensor.float() for tensor in sliced[:3]] + sliced[3:]

        def d_entry_state(d_exit_state):
            entry = one.clone().requires_grad_()
            output, final_state = op.reference(*inputs, entry, output_final_state=True)
            (grad,) = torch.autograd.grad(
                (output, final_state), entry, (d_output[:, start:stop], d_exit_state)
            )
            return grad

        _, bias = op.reference(*inputs, one, output_final_state=True)
        _, shifted = op.reference(*inputs, identity, output_final_state=True)
        reverse_bias = d_entry_state(one)
        forward = torch.cat((bias, shifted - bias), dim=-2)[0]
        reverse = torch.cat((reverse_bias, d_entry_state(identity) - reverse_bias), dim=-2)[0]
        return forward, reverse

    # (start, stop) pairs on the second subsequence's chunk grid: interior runs, a single chunk,
    # a run ending at the subsequence end, and the first chunk alone.
    for start, stop in ((136, 264), (200, 456), (136, 200), (392, 520), (72, 136)):
        forward, reverse = oracle(start, stop)
        for name, actual, expected in (
            ("forward", prepared.state_summary(start, stop), forward),
            ("reverse", grads.state_grad_summary(start, stop), reverse),
        ):
            assert_relative_rms_within(actual, expected, f"{name} [{start}, {stop})", max_eps=1.0)


@requires_kda_target
@op_param
def test_state_summaries_match_per_range_summaries(op):
    """One device-``bounds`` launch reproduces ``state_summary`` per range, forward and reverse."""
    q, k, v, gate, beta = op.make_inputs(520, seed=13)
    gate = gate / 50
    cu_seqlens = torch.tensor([0, 72, 520], dtype=torch.int32, device="cuda")
    d_output = torch.randn(v.shape, device="cuda").to(v.dtype)
    zero = torch.zeros(2, op.value_heads, HEAD_DIM, HEAD_DIM, device="cuda")
    prepared = op.prepare(q, k, v, gate, beta, cu_seqlens=cu_seqlens)
    prepared.run(zero, output_final_state=True)
    grads = op.prepare_backward(prepared.saved, d_output, zero, scale=prepared.scale)

    # Both subsequences, an interior chunk run, an empty range, and a chunk-aligned prefix.
    ranges = [(0, 72), (72, 520), (136, 264), (300, 300), (72, 200)]
    bounds = torch.tensor(ranges, dtype=torch.int32, device="cuda")
    forward = prepared.state_summaries(bounds)
    reverse = grads.state_grad_summaries(bounds)
    assert forward.shape == reverse.shape == (len(ranges), op.value_heads, 2 * HEAD_DIM, HEAD_DIM)
    identity = neutral_summary(op.value_heads, HEAD_DIM, HEAD_DIM, device="cuda")
    for index, (start, stop) in enumerate(ranges):
        if start == stop:
            torch.testing.assert_close(forward[index], identity, atol=0, rtol=0)
            torch.testing.assert_close(reverse[index], identity, atol=0, rtol=0)
            continue
        for name, actual, expected in (
            ("forward", forward[index], prepared.state_summary(start, stop)),
            ("reverse", reverse[index], grads.state_grad_summary(start, stop)),
        ):
            # Segmentation differs between the two launches, so agreement is FP32 rounding.
            torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
            assert_relative_rms_within(
                actual,
                expected,
                f"{name} [{start}, {stop})",
                max_eps=64,
                source_dtype=torch.float32,
            )


@requires_kda_target
def test_state_summary_rejects_ranges_outside_the_stream():
    q, k, v, gate, beta = KDA.make_inputs(128, seed=3)
    prepared = chunk_kda_prepare(q, k, v, gate, beta)
    with pytest.raises(ValueError, match="summary range"):
        prepared.state_summary(64, 200)
    with pytest.raises(ValueError, match="summary range"):
        prepared.state_summary(64, 64)


@requires_kda_target
@requires_mega
@pytest.mark.parametrize("bounds", [(0, 72), (72, 200)], ids=["partial-tail", "aligned-offset"])
def test_mega_state_summary_replays_under_cuda_graph(bounds):
    """The lazily factored fragment builds its boundaries on device, so capture never syncs."""
    q, k, v, gate, beta = KDA.make_inputs(200, seed=7)
    cu_seqlens = torch.tensor([0, 72, 200], dtype=torch.int32, device="cuda")
    prepared = chunk_kda_prepare(q, k, v, gate, beta, cu_seqlens=cu_seqlens, kernel_options=MEGA)
    expected = prepared.state_summary(*bounds)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepared.state_summary(*bounds)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = prepared.state_summary(*bounds)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)
    # Replay must recompute from the captured inputs rather than replay stale results.
    prepared.saved.v.mul_(0.5)
    fresh = prepared.state_summary(*bounds)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(captured, expected)
    torch.testing.assert_close(captured, fresh, atol=0, rtol=0)


@requires_kda_target
@requires_mega
def test_mega_prepare_rejects_split_schedules():
    q, k, v, gate, beta = KDA.make_inputs(128, seed=8)
    for option in ("split_backward", "split_forward"):
        with pytest.raises(ValueError, match="split schedules"):
            chunk_kda_prepare(
                q, k, v, gate, beta, kernel_options={"backend": "mega", option: True}
            )


@requires_kda_target
def test_mega_context_parallel_rejects_fastmath_like_the_unsharded_op():
    q, k, v, gate, beta = KDA.make_inputs(128, seed=8)
    plan = ContextParallelPlan.from_fragments((0, 128), [[(0, 128)]], cp_rank=0)
    with pytest.raises(ValueError, match="fastmath is not supported by the Mega backend"):
        context_parallel_kda(
            q,
            k,
            v,
            gate,
            beta,
            routing=plan.routing("cuda"),
            group=None,
            fastmath=True,
            kernel_options={"backend": "mega"},
        )


class RankResult(NamedTuple):
    plan: ContextParallelPlan
    token_ids: torch.Tensor
    output: torch.Tensor
    final_state: torch.Tensor
    grads: tuple[torch.Tensor, ...]  # dq, dk, dv, dgate, dbeta


def simulate_context_parallel(
    op: Op, cu_seqlens_global, fragments, inputs, d_output, d_final_state
) -> list[RankResult]:
    """Run the recipe for every rank in one process, replacing each all-gather with a stack."""
    device = inputs[0].device
    ranks = range(len(fragments))
    plans = [
        ContextParallelPlan.from_fragments(cu_seqlens_global, fragments, rank) for rank in ranks
    ]
    token_ids = [plan.global_token_ids(device) for plan in plans]
    routings = [plan.routing(device) for plan in plans]
    prepared = [
        op.prepare(
            *(tensor[:, token_ids[r]] for tensor in inputs), cu_seqlens=routings[r].cu_seqlens
        )
        for r in ranks
    ]

    gathered = torch.stack([summary_slots(prepared[r], routings[r]) for r in ranks])
    initial_states = [compose_entry_states(gathered, routings[r]) for r in ranks]
    forward = [prepared[r].run(initial_states[r], output_final_state=True) for r in ranks]

    # The loss's state cotangent reaches only the subsequences that end their sequence.
    d_exit_states = [
        torch.stack(
            [
                d_final_state[subsequence.sequence]
                if index in plan.terminal
                else torch.zeros_like(d_final_state[0])
                for index, subsequence in enumerate(plan.subsequences)
            ]
        )
        for plan in plans
    ]
    grads = [
        op.prepare_backward(
            prepared[r].saved,
            d_output[:, token_ids[r]],
            initial_states[r],
            scale=prepared[r].scale,
        )
        for r in ranks
    ]
    gathered = torch.stack(
        [grad_summary_slots(grads[r], d_exit_states[r], routings[r]) for r in ranks]
    )
    input_grads = [
        grads[r].run(compose_exit_cotangents(gathered, d_exit_states[r], routings[r]))[:5]
        for r in ranks
    ]
    return [RankResult(plans[r], token_ids[r], *forward[r], input_grads[r]) for r in ranks]


# Sequences [0, 40), [40, 232), [232, 384); each entry lists one rank's fragments.
CU_SEQLENS = (0, 40, 232, 384)
LAYOUTS = [
    pytest.param([[(0, 192)], [(192, 384)]], id="contiguous-2"),
    pytest.param([[(0, 96), (288, 384)], [(96, 192), (192, 288)]], id="zigzag-2"),
    pytest.param(
        [[(0, 64), (320, 384)], [(64, 128), (256, 320)], [(128, 192), (192, 256)]], id="zigzag-3"
    ),
    # Uneven ownership: rank 0 holds one short piece, rank 1 the rest in reversed order.
    pytest.param([[(96, 160)], [(160, 384), (0, 96)]], id="uneven-reordered"),
]


@requires_kda_target
@op_param
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_simulated_context_parallel_matches_unsharded_op(op, layout, dtype, request):
    if op is KDA_MEGA and dtype is torch.float16:
        # The unsharded Mega op itself returns non-finite FP16 outputs for model-range gates:
        # exp(-cumulative gate) over a 16-token chunk exceeds the FP16 range once gates fall below
        # -ln 2. Sharding neither causes nor hides that, so pin it here until the kernel is fixed.
        request.applymarker(pytest.mark.xfail(strict=True, reason="Mega FP16 gate overflow"))
    q, k, v, gate, beta = op.make_inputs(CU_SEQLENS[-1], seed=4, dtype=dtype)
    inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in (q, k, v, gate, beta))
    offsets = torch.tensor(CU_SEQLENS, dtype=torch.int32, device="cuda")

    output, final_state = op.chunk(*inputs, cu_seqlens=offsets, output_final_state=True)
    generator = torch.Generator(device="cuda").manual_seed(5)
    d_output = torch.randn(output.shape, device="cuda", generator=generator).to(output.dtype)
    d_final_state = torch.randn(final_state.shape, device="cuda", generator=generator)
    expected_grads = torch.autograd.grad(
        (output, final_state), inputs, grad_outputs=(d_output, d_final_state)
    )

    # Sharding must not cost accuracy: measure both paths against the FP32 eager oracle.
    f32 = tuple(tensor.detach().float().requires_grad_() for tensor in (q, k, v, gate, beta))
    ref_output, ref_final = op.reference(*f32, cu_seqlens=offsets, output_final_state=True)
    ref_grads = torch.autograd.grad(
        (ref_output, ref_final), f32, grad_outputs=(d_output.float(), d_final_state)
    )

    results = simulate_context_parallel(
        op, CU_SEQLENS, layout, (q, k, v, gate, beta), d_output, d_final_state
    )
    # Sharding changes accumulation order, so compare both paths against the FP32 oracle: the
    # pointwise budget is derived from the unsharded op's own error, and a systematic precision
    # loss in the summary path would show up as a multiple of its relative RMS error.
    for result in results:
        ids = result.token_ids
        budget = partial(assert_matches_low_precision_reference, source_dtype=dtype)
        budget(result.output, ref_output[:, ids], output[:, ids], "output")
        for index in result.plan.terminal:
            sequence = result.plan.subsequences[index].sequence
            budget(result.final_state[index], ref_final[sequence], final_state[sequence], "state")
        names = ("dq", "dk", "dv", "dgate", "dbeta")
        for name, actual, expected, reference in zip(
            names, result.grads, expected_grads, ref_grads, strict=True
        ):
            budget(actual, reference[:, ids], expected[:, ids], name)
            sharded = relative_rms_error(actual, reference[:, ids])
            unsharded = relative_rms_error(expected[:, ids], reference[:, ids])
            assert sharded <= 1.5 * unsharded + 1e-6, (name, sharded, unsharded)
