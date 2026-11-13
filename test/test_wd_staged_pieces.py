"""Staged warpdrive passes over the split summaries' pieces (NOTE [Staged piece plans]): accuracy
against FP64, independence from call order and bounds, and inactive gate rows."""

from __future__ import annotations

import itertools
from functools import partial
from typing import NamedTuple

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear._delta_rule.warpdrive.drivers import summary as native_summary
from attn_gym.linear.gdn.stages import chunk_gdn_prepare, chunk_gdn_prepare_backward
from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward
from attn_gym.testing.kda import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="warpdrive requires an SM100-family GPU",
)

HEADS = 2
SCALE = 128**-0.5
# A short, an empty and two long sequences (the long ones split), then inactive tokens.
LENGTHS = [37, 0, 4000, 1500]
INACTIVE = 13
NAMES = ("output", "final_state", "dq", "dk", "dv", "dgate", "dbeta", "d_initial_state")
STATES = ("final_state", "d_initial_state")


class Variant(NamedTuple):
    prepare: object
    prepare_backward: object
    channel_gate: bool  # KDA decays per key channel, GDN per head
    key_heads: int


WARPDRIVE = {"backend": "warpdrive"}
KDA = Variant(
    partial(chunk_kda_prepare, kernel_options=WARPDRIVE), chunk_kda_prepare_backward, True, HEADS
)
GDN = Variant(
    partial(chunk_gdn_prepare, kernel_options=WARPDRIVE), chunk_gdn_prepare_backward, False, HEADS
)
VARIANTS = [
    pytest.param(KDA, id="kda"),
    pytest.param(GDN, id="gdn"),
    pytest.param(GDN._replace(key_heads=1), id="gdn-gqa"),
]


def _operands(variant: Variant, seed: int = 0, dtype: torch.dtype = torch.bfloat16):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tokens = sum(LENGTHS) + INACTIVE
    shape = (1, tokens, HEADS, 128)

    def normal(*size):
        return torch.randn(size, device="cuda", generator=generator)

    q, k = (F.normalize(normal(1, tokens, variant.key_heads, 128), dim=-1).to(dtype) for _ in "qk")
    v, d_output = (normal(*shape).to(dtype) for _ in range(2))
    # Mild gates keep every piece's transition above rounding over thousands of tokens.
    gate_shape = shape if variant.channel_gate else shape[:3]
    gate = -0.004 * torch.rand(gate_shape, device="cuda", generator=generator)
    beta = torch.rand(shape[:3], device="cuda", generator=generator)
    state_shape = (len(LENGTHS), HEADS, 128, 128)
    initial_state, d_final_state = 0.1 * normal(*state_shape), normal(*state_shape)
    return (q, k, v, gate, beta), d_output, initial_state, d_final_state


def _staged(variant, inputs, d_output, initial_state, d_final_state, *, split, bounds=None):
    cu_seqlens = cumulative_sequence_offsets(LENGTHS)
    prepared = variant.prepare(*inputs, cu_seqlens=cu_seqlens)
    grads = variant.prepare_backward(prepared.saved, d_output, initial_state, scale=SCALE)
    assert (prepared.pieces is not None, grads.pieces is not None) == (split, split)
    results = (
        *prepared.run(initial_state, output_final_state=True),
        *grads.run(d_final_state),
    )
    if bounds is None:
        return results
    return results, prepared.state_summaries(bounds), grads.state_grad_summaries(bounds)


def _fp64(variant, inputs, d_output, initial_state, d_final_state):
    """Output, final state and gradients of the token recurrence ``S <- S D_t (I - beta k k^T) +
    beta v k^T`` (V-major state, ``o_t = scale * S_t q_t``) in FP64."""
    leaves = [t.double().requires_grad_() for t in (*inputs, initial_state)]
    q, k, v, gate, beta = (t[0] for t in leaves[:5])
    entry = leaves[5]
    groups = HEADS // variant.key_heads
    q, k = (t.repeat_interleave(groups, dim=1) for t in (q, k))
    do = d_output[0].double()
    outputs, finals = torch.zeros_like(do), []
    cu = cumulative_sequence_offsets(LENGTHS).tolist()
    for index, (start, stop) in enumerate(itertools.pairwise(cu)):
        state = entry[index]
        rows = []
        for t in range(start, stop):
            kt, bt, decay = k[t], beta[t][:, None, None], gate[t].exp()
            state = state * (decay[:, None, :] if variant.channel_gate else decay[:, None, None])
            state = state + bt * (v[t][:, :, None] - state @ kt[:, :, None]) * kt[:, None, :]
            rows.append(SCALE * (state @ q[t][:, :, None])[..., 0])
        if rows:
            tokens = torch.arange(start, stop, device="cuda")
            outputs = outputs.index_put((tokens,), torch.stack(rows))
        finals.append(state)
    final = torch.stack(finals)
    loss = (outputs * do).sum() + (final * d_final_state.double()).sum()
    grads = torch.autograd.grad(loss, leaves)
    return (outputs[None].detach(), final.detach(), *grads)


def _assert_same_results(actual, expected) -> None:
    """Bitwise equality of staged results where they are defined: token rows past the endpoint
    are unspecified."""
    for name, got, want in zip(NAMES, actual, expected, strict=True):
        if name not in STATES:  # token rows: the active ones
            got, want = got[:, : sum(LENGTHS)], want[:, : sum(LENGTHS)]
        torch.testing.assert_close(got, want, atol=0, rtol=0, msg=name)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual.double() - expected).norm() / expected.norm()).item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_staged_pieces_are_as_accurate_as_the_unsplit_run(variant, dtype, monkeypatch):
    """Against FP64, the staged plans stay within 1.25x of the chunk op's own error on every
    output (2x pointwise), and the gate cotangent is zero on inactive tokens.

    Half an FP16 eps of slack absorbs rounding-level noise: across seeds the FP16 GQA dgate
    ratio spans 0.97-1.38 at about 1e-3 relative L2, one element dominating its worst case."""
    operands = _operands(variant, dtype=dtype)
    staged = _staged(variant, *operands, split=True)
    expected = _fp64(variant, *operands)
    with monkeypatch.context() as patch:
        patch.setattr(native_summary, "summary_span", lambda *args, **kwargs: 0)
        unsplit = _staged(variant, *operands, split=False)
    active, slack = sum(LENGTHS), 0.5 * torch.finfo(torch.float16).eps
    for name, got, reference, want in zip(NAMES, staged, unsplit, expected, strict=True):
        if name not in STATES:  # token rows: the active ones
            got, reference, want = got[:, :active], reference[:, :active], want[:, :active]
        error, baseline = _relative_l2(got, want), _relative_l2(reference, want)
        assert error <= 1.25 * baseline + slack, f"{name}: {error:.3e} vs unsplit {baseline:.3e}"
        # Pointwise too, so an error local to a piece boundary cannot hide in the norm.
        worst, worst_baseline = ((t.double() - want).abs().max() for t in (got, reference))
        budget = 2 * worst_baseline + slack * want.abs().max()
        assert worst <= budget, f"{name}: {worst:.3e} vs unsplit {worst_baseline:.3e}"
    assert not staged[5][:, active:].any()


@pytest.mark.parametrize("variant", VARIANTS)
def test_staged_pieces_ignore_call_order_and_bounds(variant):
    """Summaries before or after ``run``, any bounds: the same bits, and every row is the
    split summary of its sequence."""
    inputs, d_output, initial_state, d_final_state = _operands(variant, seed=1)
    cu_seqlens = cumulative_sequence_offsets(LENGTHS)
    every = torch.stack((cu_seqlens[:-1], cu_seqlens[1:]), dim=1).contiguous()
    results, summaries, grad_summaries = _staged(
        variant, inputs, d_output, initial_state, d_final_state, split=True, bounds=every
    )
    prepared = variant.prepare(*inputs, cu_seqlens=cu_seqlens)
    grads = variant.prepare_backward(prepared.saved, d_output, initial_state, scale=SCALE)
    selected = every[[3, 2]].contiguous()
    early = (prepared.state_summaries(selected), grads.state_grad_summaries(selected))
    late = (*prepared.run(initial_state, output_final_state=True), *grads.run(d_final_state))
    _assert_same_results(late, results)
    for got, want in zip(early, (summaries[[3, 2]], grad_summaries[[3, 2]]), strict=True):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    saved = prepared.saved
    q, k, v, gate, beta = (saved.q, saved.k, saved.v, saved.gate, saved.beta)
    if variant.channel_gate:
        forward, reverse = (
            native_summary.build_state_summaries,
            native_summary.build_state_grad_summaries,
        )
        expected = (
            forward(k, v, gate, beta, cu_seqlens, bounds=every),
            reverse(q, k, v, gate, beta, grads.d_output, cu_seqlens, SCALE, bounds=every),
        )
    else:
        forward = native_summary.build_gdn_state_summaries
        reverse = native_summary.build_gdn_state_grad_summaries
        expected = (
            forward(k, v, gate, beta, cu_seqlens, bounds=every),
            reverse(q, k, v, gate, beta, grads.d_output, cu_seqlens, SCALE, bounds=every),
        )
    for got, want in zip((summaries, grad_summaries), expected, strict=True):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_staged_pieces_replay_new_inputs_when_the_graph_captures_prepare(variant):
    """A split handle's maps come from its inputs at ``prepare`` (NOTE [Staged piece plans]), so
    a graph that captures ``prepare`` and ``prepare_backward`` with the summaries and ``run``
    replays new input values bit for bit."""
    inputs, d_output, initial_state, d_final_state = _operands(variant, seed=3)
    cu_seqlens = cumulative_sequence_offsets(LENGTHS)
    every = torch.stack((cu_seqlens[:-1], cu_seqlens[1:]), dim=1).contiguous()

    def step():
        prepared = variant.prepare(*inputs, cu_seqlens=cu_seqlens)
        grads = variant.prepare_backward(prepared.saved, d_output, initial_state, scale=SCALE)
        assert prepared.pieces is not None and grads.pieces is not None
        results = (
            *prepared.run(initial_state, output_final_state=True),
            *grads.run(d_final_state),
        )
        return results, prepared.state_summaries(every), grads.state_grad_summaries(every)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    inputs[2].mul_(0.5)
    d_output.mul_(0.5)
    expected = step()
    graph.replay()
    torch.cuda.synchronize()
    _assert_same_results(captured[0], expected[0])
    for got, want in zip(captured[1:], expected[1:], strict=True):
        torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_staged_pieces_ignore_the_default_dtype(variant):
    """The pieces' maps, states and scratch name FP32: a float64 default dtype changes no bit."""
    operands = _operands(variant, seed=4)
    expected = _staged(variant, *operands, split=True)
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        actual = _staged(variant, *operands, split=True)
    finally:
        torch.set_default_dtype(default)
    _assert_same_results(actual, expected)


@pytest.mark.parametrize("variant", VARIANTS)
def test_staged_pieces_without_states_match_zero_states(variant):
    """No entry state, exit cotangent or final state gives the bits of explicit zeros."""
    inputs, d_output, initial_state, d_final_state = _operands(variant, seed=2)
    cu_seqlens = cumulative_sequence_offsets(LENGTHS)
    zero = torch.zeros_like(initial_state)
    prepared = variant.prepare(*inputs, cu_seqlens=cu_seqlens)
    output, final_state = prepared.run(None)
    zero_output, _ = prepared.run(zero, output_final_state=True)
    assert final_state is None
    grads = variant.prepare_backward(prepared.saved, d_output, None, scale=SCALE)
    zero_grads = variant.prepare_backward(prepared.saved, d_output, zero, scale=SCALE)
    actual = (output, *grads.run(None))
    expected = (zero_output, *zero_grads.run(torch.zeros_like(d_final_state)))
    assert actual[-1] is None and expected[-1] is not None
    active = sum(LENGTHS)
    for got, want in zip(actual[:-1], expected[:-1], strict=True):
        torch.testing.assert_close(got[:, :active], want[:, :active], atol=0, rtol=0)


def test_staged_pieces_compile_once_across_sequence_counts(triton_compiles):
    """The entry-state and tail kernels compile once across sequence counts and lengths
    (AGENTS.md, kernel specialization); the planner's buckets are test_wd_summary_split.py's."""
    inputs, d_output, _, _ = _operands(KDA)
    for lengths in ([37, 0, 4000, 1500], [3000, 2000, 17], [5000], [1300] * 4):
        cu_seqlens = cumulative_sequence_offsets(lengths)
        state = torch.zeros(len(lengths), HEADS, 128, 128, device="cuda")
        prepared = chunk_kda_prepare(*inputs, cu_seqlens=cu_seqlens, kernel_options=WARPDRIVE)
        assert prepared.pieces is not None
        prepared.run(state, output_final_state=True)
        grads = chunk_kda_prepare_backward(prepared.saved, d_output, state, scale=SCALE)
        grads.run(state)
    assert triton_compiles["_apply_maps"] == 1 and triton_compiles["_zero_tail"] == 1
