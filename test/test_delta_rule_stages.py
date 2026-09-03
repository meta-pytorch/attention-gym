"""Staged delta-rule primitives: ``prepare`` / ``run`` / state summaries against the public ops."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear.kda import chunk_kda
from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward
from attn_gym.linear.state_summary import compose_summaries, merge_state, neutral_summary
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


def kda_inputs(tokens: int, *, key_heads: int, value_heads: int, seed: int, dtype) -> Inputs:
    assert key_heads == value_heads, "KDA has no grouped key heads"
    return make_kda_test_inputs(
        tokens, heads=value_heads, seed=seed, dtype=dtype, normalize_qk=True
    )


KDA = Op(
    chunk_kda,
    partial(chunk_kda, impl="reference"),
    chunk_kda_prepare,
    chunk_kda_prepare_backward,
    kda_inputs,
    key_heads=2,
    value_heads=2,
)
op_param = pytest.mark.parametrize("op", [pytest.param(KDA, id="kda")])


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
    # reference gets compact copies so the same call also serves ops that reject strides.
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
@pytest.mark.parametrize(
    "loss", ["both", "output-only", "final-state-only"], ids=lambda loss: f"loss-{loss}"
)
@pytest.mark.parametrize("state", ["contiguous", "strided-key", "none"])
def test_prepare_backward_run_matches_chunk_op_gradients(op, scale, loss, state):
    """The staged backward is the op's own kernel sequence, so all six gradients are bitwise.

    Covers the cotangents the staged API makes optional (``d_output=None``, no final-state loss)
    and the entry-state layouts ``run`` accepts: none, contiguous, and a key-strided view.
    """
    inputs = op.make_inputs(320, seed=7)
    cu_seqlens = torch.tensor([0, 100, 320], dtype=torch.int32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(8)
    wide = torch.randn(
        2, op.value_heads, HEAD_DIM, 2 * HEAD_DIM, device="cuda", generator=generator
    )
    initial_state = {
        "contiguous": wide[..., :HEAD_DIM].contiguous(),
        "strided-key": wide[..., ::2],
        "none": None,
    }[state]
    d_final_state = torch.randn(
        2, op.value_heads, HEAD_DIM, HEAD_DIM, device="cuda", generator=generator
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

    def standalone(start, stop):
        """The staged summaries of the token range run as a stream of its own."""
        sliced = [tensor[:, start:stop] for tensor in (q, k, v, gate, beta)]
        own = op.prepare(*sliced)
        own.run(one, output_final_state=True)
        own_grads = op.prepare_backward(
            own.saved, d_output[:, start:stop].to(v.dtype), one, scale=own.scale
        )
        return own.state_summary(0, stop - start), own_grads.state_grad_summary(0, stop - start)

    # (start, stop) pairs on the second subsequence's chunk grid: interior runs, a single chunk,
    # a run ending at the subsequence end, and the first chunk alone. The interior summary must
    # match the same range summarized on its own, pointwise within that run's own error against
    # the FP32 oracle and in aggregate.
    for start, stop in ((136, 264), (200, 456), (136, 200), (392, 520), (72, 136)):
        oracles = oracle(start, stop)
        for name, actual, expected, low in zip(
            ("forward", "reverse"),
            (prepared.state_summary(start, stop), grads.state_grad_summary(start, stop)),
            oracles,
            standalone(start, stop),
            strict=True,
        ):
            label = f"{name} [{start}, {stop})"
            assert_matches_low_precision_reference(actual, expected, low, label)
            assert_relative_rms_within(actual, expected, label, max_eps=1.0)


@requires_kda_target
def test_state_summary_rejects_ranges_outside_the_stream():
    q, k, v, gate, beta = KDA.make_inputs(128, seed=3)
    prepared = chunk_kda_prepare(q, k, v, gate, beta)
    with pytest.raises(ValueError, match="summary range"):
        prepared.state_summary(64, 200)
    with pytest.raises(ValueError, match="summary range"):
        prepared.state_summary(64, 64)
