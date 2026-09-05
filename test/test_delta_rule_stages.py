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
from attn_gym.linear.kda import chunk_kda
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


def assert_sharded_matches(
    name: str, actual: torch.Tensor, reference: torch.Tensor, unsharded: torch.Tensor, *, dtype
) -> None:
    """Sharding must not cost accuracy.

    Pointwise within the unsharded op's own budget against the FP32 oracle, and in aggregate
    within 1.5x of the unsharded op's relative RMS error: sharding only reorders accumulation, so
    a systematic precision loss in the summary path would show up as a multiple.
    """
    assert_matches_low_precision_reference(actual, reference, unsharded, name, source_dtype=dtype)
    sharded = relative_rms_error(actual, reference)
    baseline = relative_rms_error(unsharded, reference)
    assert sharded <= 1.5 * baseline + 1e-6, (name, sharded, baseline)


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
    summaries = prepared.state_summaries(bounds_of((0, 72), (72, 200)))
    assert summaries.shape == (2, op.value_heads, 2 * HEAD_DIM, HEAD_DIM)
    for index in range(2):
        assert_summary_parts_match(summaries[index], fused, oracle, index)


def bounds_of(*ranges: tuple[int, int]) -> torch.Tensor:
    """``int32 [R, 2]`` device tensor of span ranges for ``state_summaries``."""
    return torch.tensor(ranges, dtype=torch.int32, device="cuda")


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
@pytest.mark.parametrize(
    "loss", ["both", "output-only", "final-state-only"], ids=lambda loss: f"loss-{loss}"
)
@pytest.mark.parametrize("state", ["contiguous", "strided-key", "none"])
@pytest.mark.parametrize("packing", ["packed", "dense"])
def test_prepare_backward_run_matches_chunk_op_gradients(op, scale, loss, state, packing):
    """The staged backward is the op's own kernel sequence, so all six gradients are bitwise.

    Covers the cotangents the staged API makes optional (``d_output=None``, no final-state loss),
    the entry-state layouts ``run`` accepts (none, contiguous, a key-strided view), and both chunk
    schedules: packed ``cu_seqlens`` and a dense span of complete chunks (``metadata is None``).
    """
    inputs = op.make_inputs(320, seed=7)
    if packing == "packed":
        cu_seqlens = torch.tensor([0, 100, 320], dtype=torch.int32, device="cuda")
        sequences = 2
    else:
        cu_seqlens = None
        sequences = 1
    generator = torch.Generator(device="cuda").manual_seed(8)
    wide = torch.randn(
        sequences, op.value_heads, HEAD_DIM, 2 * HEAD_DIM, device="cuda", generator=generator
    )
    initial_state = {
        "contiguous": wide[..., :HEAD_DIM].contiguous(),
        "strided-key": wide[..., ::2],
        "none": None,
    }[state]
    d_final_state = torch.randn(
        sequences, op.value_heads, HEAD_DIM, HEAD_DIM, device="cuda", generator=generator
    )

    leaves = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs) + (
        () if initial_state is None else (initial_state.detach().clone().requires_grad_(),)
    )
    output, final_state = op.chunk(
        *leaves, cu_seqlens=cu_seqlens, output_final_state=True, scale=scale
    )
    d_output = torch.randn(output.shape, device="cuda", generator=generator).to(output.dtype)
    if loss == "output-only":
        expected = torch.autograd.grad(output, leaves, d_output)
        d_final_state = torch.zeros_like(d_final_state)
    elif loss == "final-state-only":
        expected = torch.autograd.grad(final_state, leaves, d_final_state)
        d_output = None
    else:
        expected = torch.autograd.grad((output, final_state), leaves, (d_output, d_final_state))

    prepared = op.prepare(*inputs, cu_seqlens=cu_seqlens, scale=scale)
    prepared.run(initial_state, output_final_state=True)
    grads = op.prepare_backward(prepared.saved, d_output, initial_state, scale=prepared.scale)
    grads.state_grad_summaries(
        bounds_of((0, 64))
    )  # Exchange order: summary first, run reuses Aqk.
    actual = grads.run(d_final_state)

    assert len(actual) == 6
    names = ("dq", "dk", "dv", "dgate", "dbeta", "d_initial_state")
    if initial_state is None:
        assert actual[5] is None
        actual, names = actual[:5], names[:5]
    for name, got, want in zip(names, actual, expected, strict=True):
        torch.testing.assert_close(
            got, want, atol=0, rtol=0, msg=lambda m, name=name: f"{name}: {m}"
        )


@requires_kda_target
@pytest.mark.parametrize("packing", ["dense", "packed"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("with_state", [False, True], ids=["no-state", "with-state"])
def test_backward_from_a_tape_without_factors_matches_saved_factors(packing, dtype, with_state):
    """Recomputed intra factors must give bit-identical reverse summaries, gradients, and states."""
    inputs = KDA.make_inputs(128, seed=31, dtype=dtype)
    cu_seqlens = (
        torch.tensor([0, 65, 65, 128], dtype=torch.int32, device="cuda")
        if packing == "packed"
        else None
    )
    sequences = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else 1
    generator = torch.Generator(device="cuda").manual_seed(32)
    state = torch.randn(
        sequences, KDA.value_heads, HEAD_DIM, HEAD_DIM, device="cuda", generator=generator
    )
    d_output = torch.randn(inputs[2].shape, device="cuda", dtype=dtype, generator=generator)
    bounds = (
        bounds_of((0, 65), (65, 65), (65, 128)) if packing == "packed" else bounds_of((0, 128))
    )

    prepared = chunk_kda_prepare(*inputs, cu_seqlens=cu_seqlens, scale=0.25, autotune=False)
    # A Mega forward saves no factors; its backward must match one that did, bit for bit.
    tapes = (prepared.saved, prepared.saved._replace(aqk=None, akk=None))
    results = []
    for saved in tapes:
        backward = chunk_kda_prepare_backward(
            saved, d_output, state if with_state else None, scale=prepared.scale, autotune=False
        )
        results.append((backward.state_grad_summaries(bounds), *backward.run(state)))
    for expected, actual in zip(*results, strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


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
    summaries = grads.state_grad_summaries(bounds_of((0, 72), (72, 200)))
    assert summaries.shape == (2, op.value_heads, 2 * HEAD_DIM, HEAD_DIM)
    for index in range(2):
        assert_summary_parts_match(summaries[index], fused, oracle, index)


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

    def standalone(start, stop):
        """The staged summaries of the token range run as a stream of its own."""
        sliced = [tensor[:, start:stop] for tensor in (q, k, v, gate, beta)]
        own = op.prepare(*sliced)
        own.run(one, output_final_state=True)
        own_grads = op.prepare_backward(
            own.saved, d_output[:, start:stop].to(v.dtype), one, scale=own.scale
        )
        whole = bounds_of((0, stop - start))
        return own.state_summaries(whole)[0], own_grads.state_grad_summaries(whole)[0]

    # Rows on the second subsequence's chunk grid: interior runs, a single chunk, a run ending at
    # the subsequence end, the first chunk alone, and an empty range. One launch summarizes them
    # all; each interior summary must match the same range summarized on its own, pointwise
    # within that run's own error against the FP32 oracle and in aggregate.
    ranges = ((136, 264), (200, 456), (136, 200), (392, 520), (72, 136), (300, 300))
    forward = prepared.state_summaries(bounds_of(*ranges))
    reverse = grads.state_grad_summaries(bounds_of(*ranges))
    assert forward.shape == reverse.shape == (len(ranges), op.value_heads, 2 * HEAD_DIM, HEAD_DIM)
    empty = neutral_summary(op.value_heads, HEAD_DIM, HEAD_DIM, device="cuda")
    torch.testing.assert_close(forward[-1], empty, atol=0, rtol=0)
    torch.testing.assert_close(reverse[-1], empty, atol=0, rtol=0)
    for index, (start, stop) in enumerate(ranges[:-1]):
        oracles = oracle(start, stop)
        for name, actual, expected, low in zip(
            ("forward", "reverse"),
            (forward[index], reverse[index]),
            oracles,
            standalone(start, stop),
            strict=True,
        ):
            label = f"{name} [{start}, {stop})"
            assert_matches_low_precision_reference(actual, expected, low, label)
            assert_relative_rms_within(actual, expected, label, max_eps=1.0)


@requires_kda_target
@requires_mega
@pytest.mark.parametrize("bounds", [(0, 72), (72, 200)], ids=["partial-tail", "aligned-offset"])
def test_mega_state_summaries_replay_under_cuda_graph(bounds):
    """The lazily factored span reads its ranges on the device, so capture never syncs."""
    q, k, v, gate, beta = KDA.make_inputs(200, seed=7)
    cu_seqlens = torch.tensor([0, 72, 200], dtype=torch.int32, device="cuda")
    prepared = chunk_kda_prepare(q, k, v, gate, beta, cu_seqlens=cu_seqlens, kernel_options=MEGA)
    bounds = bounds_of(bounds)
    expected = prepared.state_summaries(bounds)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepared.state_summaries(bounds)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = prepared.state_summaries(bounds)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)
    # Replay must recompute from the captured inputs rather than replay stale results.
    prepared.saved.v.mul_(0.5)
    fresh = prepared.state_summaries(bounds)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(captured, expected)
    torch.testing.assert_close(captured, fresh, atol=0, rtol=0)


@requires_kda_target
def test_mega_prepare_rejects_split_schedules():
    """The option check precedes the availability check, so this needs no Mega install."""
    q, k, v, gate, beta = KDA.make_inputs(128, seed=8)
    for option in ("split_backward", "split_forward"):
        with pytest.raises(ValueError, match="split schedules"):
            chunk_kda_prepare(q, k, v, gate, beta, kernel_options={**MEGA, option: True})


class RankResult(NamedTuple):
    plan: ContextParallelPlan
    token_ids: torch.Tensor
    output: torch.Tensor
    final_state: torch.Tensor
    grads: tuple[torch.Tensor, ...]  # dq, dk, dv, dgate, dbeta


def simulate_context_parallel(
    op: Op, cu_seqlens_global, fragments, inputs, d_output, d_final_state
) -> list[RankResult]:
    """Run every rank's forward and backward in one process; a stack stands in for all-gather."""
    device = inputs[0].device
    ranks = range(len(fragments))
    plans = [
        ContextParallelPlan.from_fragments(cu_seqlens_global, fragments, cp_rank)
        for cp_rank in ranks
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
        request.applymarker(
            pytest.mark.xfail(strict=True, raises=AssertionError, reason="Mega FP16 gate overflow")
        )
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
    for result in results:
        ids = result.token_ids
        check = partial(assert_sharded_matches, dtype=dtype)
        check("output", result.output, ref_output[:, ids], output[:, ids])
        for index in result.plan.terminal:
            sequence = result.plan.subsequences[index].sequence
            check("state", result.final_state[index], ref_final[sequence], final_state[sequence])
        names = ("dq", "dk", "dv", "dgate", "dbeta")
        for name, actual, expected, reference in zip(
            names, result.grads, expected_grads, ref_grads, strict=True
        ):
            check(name, actual, reference[:, ids], expected[:, ids])
