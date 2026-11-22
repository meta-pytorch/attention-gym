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
from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.linear._delta_rule.warpdrive import (
    gdn_prefill,
    gdn_recompute,
    kda_prefill,
    kda_recompute,
)
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
        forward = native_summary.build_state_summaries
        reverse = native_summary.build_state_grad_summaries
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


def test_staged_pieces_compile_once_across_sequence_and_head_counts(triton_compiles):
    """The entry-state, adjoint and tail kernels compile once per direction across sequence
    counts, lengths and head counts (AGENTS.md, kernel specialization); the planner's buckets are
    test_wd_summary_split.py's."""
    inputs, d_output, _, _ = _operands(KDA)
    cases = [([37, 0, 4000, 1500], 2), ([3000, 2000, 17], 1), ([5000], 2), ([1300] * 4, 1)]
    for lengths, heads in cases:
        operands = [t[:, :, :heads] for t in (*inputs, d_output)]
        cu_seqlens = cumulative_sequence_offsets(lengths)
        state = torch.zeros(len(lengths), heads, 128, 128, device="cuda")
        prepared = chunk_kda_prepare(
            *operands[:5], cu_seqlens=cu_seqlens, kernel_options=WARPDRIVE
        )
        assert prepared.pieces is not None
        prepared.run(state, output_final_state=True)
        grads = chunk_kda_prepare_backward(prepared.saved, operands[5], state, scale=SCALE)
        grads.run(state)
    # One _apply_maps binary for the entry states (forward and recompute), one for the exit
    # cotangents with the sequences' entry cotangents.
    assert triton_compiles["_apply_maps"] == 2
    for kernel in ("_prefix_adjoint_products", "_sum_piece_cotangents", "_zero_tail"):
        assert triton_compiles[kernel] == 1, kernel


def _spy_entry_states(monkeypatch) -> dict[str, list[torch.Tensor]]:
    """Record the piece entry states the staged forward prefill and the backward checkpoint
    recompute start from."""
    seen: dict[str, list[torch.Tensor]] = {"forward": [], "backward": []}

    def spy_compiled(module, name, direction, index):
        compile_fn = getattr(module, name)

        def compile_spy(*args, **kwargs):
            fn = compile_fn(*args, **kwargs)

            def call(*launch):
                seen[direction].append(launch[index].clone())
                return fn(*launch)

            return call

        monkeypatch.setattr(module, name, compile_spy)

    spy_compiled(kda_prefill, "compile_kda_prefill", "forward", 5)
    spy_compiled(kda_recompute, "compile_kda_recompute", "backward", 4)
    spy_compiled(gdn_prefill, "compile_gdn_prefill", "forward", 6)
    recompute = gdn_recompute.gdn_recompute

    def gdn_spy(*args, state_in, **kwargs):
        seen["backward"].append(state_in.clone())
        return recompute(*args, state_in=state_in, **kwargs)

    monkeypatch.setattr(gdn_recompute, "gdn_recompute", gdn_spy)
    return seen


@pytest.mark.parametrize("seeded", [True, False], ids=["seeded", "unseeded"])
@pytest.mark.parametrize(
    ("variant", "fastmath"),
    [
        pytest.param(KDA, True, id="kda"),
        pytest.param(KDA, False, id="kda-exact-exp"),
        pytest.param(GDN, None, id="gdn"),
    ],
)
def test_backward_recomputes_from_the_forward_entry_states(variant, fastmath, seeded, monkeypatch):
    """The checkpoint recompute starts every piece from the bits the forward prefill started it
    from, and a sequence's first piece starts from the supplied state itself."""
    inputs, d_output, initial_state, _ = _operands(variant, seed=5)
    initial_state = initial_state if seeded else None
    options = {} if fastmath is None else {"fastmath": fastmath}
    cu_seqlens = cumulative_sequence_offsets(LENGTHS)
    seen = _spy_entry_states(monkeypatch)
    prepared = variant.prepare(*inputs, cu_seqlens=cu_seqlens, **options)
    prepared.run(initial_state, output_final_state=True)
    grads = variant.prepare_backward(
        prepared.saved, d_output, initial_state, scale=SCALE, **options
    )
    assert prepared.pieces is not None and grads.pieces is not None
    main_rows = prepared.pieces.launch.main_rows.tolist()
    pieces = main_rows[-1] // HEADS
    (forward,), (backward,) = seen["forward"], seen["backward"]
    assert torch.equal(forward[:pieces].view(torch.int32), backward[:pieces].view(torch.int32))
    if seeded:
        first = [row // HEADS for row in main_rows[:-1]]
        assert torch.equal(forward[first].view(torch.int32), initial_state.view(torch.int32))


@pytest.mark.parametrize("variant", [pytest.param(KDA, id="kda"), pytest.param(GDN, id="gdn")])
def test_staged_seed_cotangent_is_the_adjoint_of_the_forward_prefix(variant):
    """The forward starts piece j from ``seed @ A_<j + B_<j``, so with the loss confined to the
    last piece the seed cotangent is that piece's entry cotangent through the forward's own
    ``A_<last`` transposed. It is also the sequence's gathered reverse summary bias, bit for
    bit, and a zero exit cotangent changes no bit of it."""
    lengths = [4000]
    inputs, d_output, initial_state, _ = _operands(variant, seed=6)
    tokens = sum(lengths)
    inputs = tuple(t[:, :tokens] for t in inputs)
    d_output, initial_state = d_output[:, :tokens].clone(), initial_state[:1]
    cu_seqlens = cumulative_sequence_offsets(lengths)
    prepared = variant.prepare(*inputs, cu_seqlens=cu_seqlens)
    assert prepared.pieces is not None
    last = prepared.pieces.launch.main_rows.tolist()[1] // HEADS - 1
    start = prepared.pieces.launch.cu_pieces.tolist()[last]
    assert last > 0
    d_output[:, :start] = 0
    grads = variant.prepare_backward(prepared.saved, d_output, initial_state, scale=SCALE)
    d_initial_state = grads.run(None)[-1]
    saved = prepared.saved
    reverse = (
        native_summary.build_state_grad_summaries
        if variant.channel_gate
        else native_summary.build_gdn_state_grad_summaries
    )
    alone = torch.tensor([0, start, tokens], dtype=torch.int32, device="cuda")
    entry_cotangent = reverse(
        saved.q, saved.k, saved.v, saved.gate, saved.beta, grads.d_output, alone, SCALE,
        batch_invariant=True,
    )[1, :, :128]  # fmt: skip
    prefix = prepared.pieces.prefix[last, :, 128:]
    expected = entry_cotangent.double() @ prefix.double().transpose(-1, -2)
    assert _relative_l2(d_initial_state[0], expected) < 1e-5
    every = torch.stack((cu_seqlens[:-1], cu_seqlens[1:]), dim=1).contiguous()
    bias = grads.state_grad_summaries(every)[:, :, :128]
    zero_exit = grads.run(torch.zeros_like(initial_state))[-1]
    for got in (d_initial_state, zero_exit):
        assert torch.equal(got.view(torch.int32), bias.view(torch.int32))


@pytest.mark.parametrize("variant", VARIANTS)
def test_empty_sequences_pass_their_state_through(variant):
    """An empty sequence returns its entry state as its final state and its exit cotangent as
    its entry cotangent, bit for bit, subnormal values included."""
    inputs, d_output, initial_state, d_final_state = _operands(variant, seed=7)
    empty = LENGTHS.index(0)
    initial_state[empty, :, 0, :4] = 1e-40
    d_final_state[empty, :, 0, :4] = -1e-40
    _, final_state, *grads = _staged(
        variant, inputs, d_output, initial_state, d_final_state, split=True
    )
    assert torch.equal(
        final_state[empty].view(torch.int32), initial_state[empty].view(torch.int32)
    )
    assert torch.equal(grads[-1][empty].view(torch.int32), d_final_state[empty].view(torch.int32))


def _padded(tensor: torch.Tensor) -> torch.Tensor:
    """``tensor`` ``[1, T, H, D]`` in a 16-byte aligned token stride whose offsets need 64
    bits."""
    tokens, row = tensor.shape[1], tensor[0, 0].numel()
    stride = (-(-(2**31) // (tokens - 1)) + row + 63) // 64 * 64
    base = torch.empty(tokens, stride, dtype=tensor.dtype, device=tensor.device)
    base[:, :row] = tensor[0].flatten(1)
    return base[:, :row].unflatten(1, tensor.shape[2:]).unsqueeze(0)


@pytest.mark.parametrize("variant", [pytest.param(KDA, id="kda"), pytest.param(GDN, id="gdn")])
def test_backward_runs_the_forward_plan(variant):
    """The backward runs the plan its forward ran, whatever its own options or operands: a
    query whose offsets need 64 bits keeps both directions unsplit, a batch-invariant forward
    keeps a default backward unsplit (the gradients of an invariant backward), and an invariant
    backward of a split forward is rejected."""
    inputs, d_output, initial_state, d_final_state = _operands(variant, seed=8)
    cu_seqlens = cumulative_sequence_offsets(LENGTHS)
    padded = variant.prepare(_padded(inputs[0]), *inputs[1:], cu_seqlens=cu_seqlens)
    assert requires_int64_abi(padded.saved.q)
    grads = variant.prepare_backward(padded.saved, d_output, initial_state, scale=SCALE)
    assert padded.pieces is None and grads.pieces is None
    invariant = {**WARPDRIVE, "batch_invariant": True}
    prepared = variant.prepare(*inputs, cu_seqlens=cu_seqlens, kernel_options=invariant)
    default = variant.prepare_backward(prepared.saved, d_output, initial_state, scale=SCALE)
    pinned = variant.prepare_backward(
        prepared.saved, d_output, initial_state, scale=SCALE, kernel_options=invariant
    )
    assert default.pieces is None
    *got, got_state = default.run(d_final_state)
    *want, want_state = pinned.run(d_final_state)
    active = sum(LENGTHS)
    for actual, expected in zip(got, want, strict=True):
        torch.testing.assert_close(actual[:, :active], expected[:, :active], atol=0, rtol=0)
    torch.testing.assert_close(got_state, want_state, atol=0, rtol=0)
    split = variant.prepare(*inputs, cu_seqlens=cu_seqlens)
    with pytest.raises(ValueError, match="batch_invariant"):
        variant.prepare_backward(
            split.saved, d_output, initial_state, scale=SCALE, kernel_options=invariant
        )


@pytest.mark.parametrize("invariant", [False, True], ids=["split", "unsplit"])
@pytest.mark.parametrize("variant", [pytest.param(KDA, id="kda"), pytest.param(GDN, id="gdn")])
def test_backward_rejects_bad_cotangents_and_scales(variant, invariant):
    """Both routes reject a ``d_output`` of the wrong shape and a scale that is not finite and
    positive before launching anything."""
    inputs, d_output, initial_state, _ = _operands(variant, seed=9)
    options = {**WARPDRIVE, "batch_invariant": invariant}
    prepared = variant.prepare(
        *inputs, cu_seqlens=cumulative_sequence_offsets(LENGTHS), kernel_options=options
    )
    assert (prepared.pieces is None) == invariant
    backward = partial(variant.prepare_backward, prepared.saved, kernel_options=options)
    for bad in (torch.cat((d_output, d_output)), torch.cat((d_output, d_output), dim=2)):
        with pytest.raises(ValueError, match="d_output"):
            backward(bad, initial_state, scale=SCALE)
    for scale in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="scale"):
            backward(d_output, initial_state, scale=scale)
