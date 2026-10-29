"""Numerical contracts of the fused Mamba2 SSD scan.

The fused path carries the recurrent state and its cotangent in FP32 with FP32 log2 decays,
stages chunk-entry states and the state-cotangent tape in the activation dtype as MMA operands,
keeps the pre-gate output in FP32, and reduces ``dD`` from the FP32 gated cotangent
``dy * silu(z)``, which the scan kernels read in the activation dtype. The oracle is the
FP64 token-by-token recurrence on the same rounded inputs; the FP32 eager recurrence sets the
pointwise reference-error envelope. Each result pairs that pointwise check with an aggregate
relative-RMS bound in BF16 epsilon, and the regressions below pin cases where a narrowing or an
unstable form once changed a representable result. For nonnegative step sizes the chunked
forward and the state and state-cotangent scans apply decays to FP32 values (carries,
entry-state reads, writes) without early underflow; the chunk backward's in-chunk gradient
terms and the token-by-token step form their decays alone (strict xfails below).
"""

from __future__ import annotations

import math
from itertools import pairwise

import pytest
import torch

from attn_gym.linear import (
    chunk_mamba2,
    paged_chunk_mamba2,
    recurrent_mamba2,
    recurrent_mamba2_decode,
)
from attn_gym.linear.mamba2.api import NO_DT_LIMIT, reference_mamba2
from attn_gym.linear.mamba2.reference import expand_groups
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    assert_relative_rms_within,
    cumulative_sequence_offsets,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# On B300, healthy model-range results measure at most 0.39 BF16 eps of relative RMS error
# (ddt_bias at T=4096; dx, dB, dC, and d_initial_state reach 0.37 and the output 0.30). This
# limit keeps about 2x headroom.
RMS_EPS_LIMIT = 0.75
# dA, dD, and ddt_bias sum per-token terms that cancel, so their error is bounded relative to
# the summed term magnitudes. On B300 the largest ratio is 0.082 BF16 eps (dA, long memory);
# ddt_bias reaches 0.018 and dD stays at FP32 rounding.
REDUCTION_EPS_LIMIT = 0.2
REDUCTION_NAMES = ("A", "D", "dt_bias")
FP32_CARRY_RTOL = 1e-5
# A steep FP32 log2 decay sum carries about 1e-5 relative error into the decayed values.
DECAY_SUM_RTOL = 1e-4
# One rounding to BF16 (unit roundoff 2^-8).
BF16_ROUNDING_RTOL = 2**-7
# With A = -1, one 64-token chunk of these step sizes decays by 2^-155, below the FP32 range.
STEEP_DT = 155 * math.log(2) / 64
RECURRENCE_NAMES = ("output", "final_state", "x", "dt", "A", "B", "C", "initial_state")
GRADIENT_NAMES = ("x", "dt", "A", "B", "C", "D", "z", "dt_bias", "initial_state")
# Documented limitation: only the contract assertions may fail, and they must keep failing.
FP16_STATE_RANGE = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="FP16 execution stages chunk-entry states as FP16 MMA operands; 65536 overflows",
)
FP16_COTANGENT_RANGE = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="FP16 execution stages state cotangents as an FP16 tape; 2^-25 flushes to zero",
)
FP16_GATED_COTANGENT_RANGE = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the scan kernels read dy * silu(z) in FP16; 7e4 overflows before dx",
)
CHUNK_BWD_PRODUCT_RANGE = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the chunk backward contracts states and writes with dS before decaying; 2^168 overflows",
)
CHUNK_BWD_DECAY_RANGE = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the chunk backward forms in-chunk decays alone; one below 2^-126 flushes to zero",
)
RECURRENT_DECAY_RANGE = pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="the token-by-token step forms exp2(dt * A) alone; below 2^-126 it flushes to zero",
)


def model_inputs(batch, tokens, *, heads=4, groups=2, head_dim=64, state_dim=64, seed=0):
    """BF16 activations with Mamba2-like step sizes and decays; parameters stay FP32."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=generator, device="cuda").to(dtype)

    return {
        "x": randn(batch, tokens, heads, head_dim, dtype=torch.bfloat16),
        "dt": randn(batch, tokens, heads) - 2.0,
        "A": -torch.exp(torch.rand(heads, generator=generator, device="cuda") * 2.77),
        "B": randn(batch, tokens, groups, state_dim, dtype=torch.bfloat16),
        "C": randn(batch, tokens, groups, state_dim, dtype=torch.bfloat16),
        "D": randn(heads),
        "z": randn(batch, tokens, heads, head_dim, dtype=torch.bfloat16),
        "dt_bias": randn(heads) * 0.5,
    }


def evaluate(inputs, cotangents, precision, **kwargs):
    """Output, final state, and every input gradient of one implementation.

    ``precision=None`` runs the fused kernels on the inputs as given; a dtype runs the eager
    recurrence on the same values upcast to it.
    """
    leaves = {
        name: (tensor if precision is None else tensor.to(precision)).detach().requires_grad_()
        for name, tensor in inputs.items()
    }
    if precision is None:
        output, state = chunk_mamba2(**leaves, impl="fused", output_final_state=True, **kwargs)
    else:
        output, state = recurrent_mamba2(
            **leaves, impl="reference", output_final_state=True, **kwargs
        )
    targets = [output, state][: len(cotangents)]
    gradients = torch.autograd.grad(
        targets, list(leaves.values()), [c.to(t.dtype) for c, t in zip(cotangents, targets)]
    )
    return {"output": output, "final_state": state} | dict(zip(leaves, gradients))


def decay_terms(inputs, cotangents, **kwargs):
    """Per-head sums of the absolute per-token ``dA`` terms, from the FP64 recurrence."""
    leaves = {name: tensor.double() for name, tensor in inputs.items()}
    A_tokens = leaves["A"].expand(inputs["dt"].shape).clone().requires_grad_()
    cu_seqlens = kwargs.get("cu_seqlens")
    offsets = [0, inputs["x"].shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    # Packed references call the core once per nonempty sequence, in order.
    spans = iter([(begin, end) for begin, end in pairwise(offsets) if begin < end])

    def core(x, dt, A, B, C, *, initial_state, output_final_state):
        begin, end = next(spans)
        A = A_tokens[:, begin:end]
        B, C = expand_groups(B, x.shape[2]), expand_groups(C, x.shape[2])
        state = initial_state
        if state is None:
            state = x.new_zeros(*x.shape[::2], x.shape[-1], B.shape[-1])
        outputs = []
        for token in range(x.shape[1]):
            decay = torch.exp(dt[:, token] * A[:, token])[..., None, None]
            write = (x[:, token] * dt[:, token, :, None])[..., None] * B[:, token, :, None]
            state = decay * state + write
            outputs.append(torch.einsum("bhpn,bhn->bhp", state, C[:, token]))
        return torch.stack(outputs, dim=1), state

    names = ("x", "dt", "A", "B", "C", "D", "z", "initial_state")
    output, state = reference_mamba2(
        core,
        *(leaves.get(name) for name in names),
        dt_bias=leaves.get("dt_bias"),
        dt_softplus=kwargs.get("dt_softplus", False),
        dt_limit=kwargs.get("dt_limit", NO_DT_LIMIT),
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    targets = [output, state][: len(cotangents)]
    (gradient,) = torch.autograd.grad(targets, A_tokens, [c.double() for c in cotangents])
    return gradient.abs().sum((0, 1))


def assert_reduction(actual, high, scale, name):
    """Bound a cancelling reduction's error by its summed term magnitudes."""
    assert torch.isfinite(actual).all(), f"{name}: kernel output contains non-finite values"
    ratio = ((actual.double() - high).abs() / scale).max().item() / torch.finfo(torch.bfloat16).eps
    assert ratio <= REDUCTION_EPS_LIMIT, (
        f"{name}: error is {ratio:.3f} BF16 eps of the summed term magnitudes"
    )


def assert_value(actual, high, low, name, *, source_dtype=torch.bfloat16):
    """Pointwise reference-error envelope plus aggregate relative RMS in source epsilon."""
    assert_matches_low_precision_reference(actual, high, low, name, source_dtype=source_dtype)
    assert_relative_rms_within(
        actual, high, name, max_eps=RMS_EPS_LIMIT, source_dtype=source_dtype
    )


def assert_matches_recurrence(actual, high, names=RECURRENCE_NAMES):
    """Results of a no-gate case, by default all of them, against the FP64 recurrence."""
    for name in names:
        value = actual[name]
        rtol = BF16_ROUNDING_RTOL if value.dtype == torch.bfloat16 else DECAY_SUM_RTOL
        torch.testing.assert_close(
            value.double(), high[name], rtol=rtol, atol=0, msg=lambda m, name=name: f"{name}: {m}"
        )


def force_segments(monkeypatch, length):
    """Split every chunk scan into ``length``-token segments regardless of the program count."""

    def plan(num_programs, tokens, device):
        return math.ceil(tokens / length), length

    for module in ("chunk_fwd", "chunk_bwd"):
        monkeypatch.setattr(f"attn_gym.linear.mamba2.triton.{module}.segment_plan", plan)


@pytest.mark.parametrize("case", ["dense", "packed", "segmented", "long_memory"])
def test_model_range_contract(case, monkeypatch):
    """Output, final state, and all nine gradients against the FP64 recurrence."""
    kwargs = {"dt_softplus": True}
    if case == "dense":
        inputs = model_inputs(2, 300)
        states = 2
    elif case == "long_memory":
        # Weak decay keeps 2048 tokens of history in the state across four scan segments.
        inputs = model_inputs(1, 2048)
        inputs["A"] = -torch.linspace(0.005, 0.05, 4, device="cuda")
        states = 1
        force_segments(monkeypatch, 512)
    else:
        lengths = [200, 0, 1, 130, 64]
        inputs = model_inputs(1, sum(lengths) + 3)
        states = len(lengths)
        kwargs["cu_seqlens"] = cumulative_sequence_offsets(lengths)
        if case == "segmented":
            force_segments(monkeypatch, 128)
    _, tokens, heads, head_dim = inputs["x"].shape
    # Output past the terminal offset is unspecified.
    cu_seqlens = kwargs.get("cu_seqlens")
    end = tokens if cu_seqlens is None else int(cu_seqlens[-1])
    generator = torch.Generator(device="cuda").manual_seed(1)
    inputs["initial_state"] = torch.randn(
        states, heads, head_dim, inputs["B"].shape[-1], generator=generator, device="cuda"
    )
    d_output = torch.randn(inputs["x"].shape, generator=generator, device="cuda")
    d_output[:, end:] = 0
    cotangents = [d_output.bfloat16(), torch.randn_like(inputs["initial_state"])]

    actual = evaluate(inputs, cotangents, None, **kwargs)
    high = evaluate(inputs, cotangents, torch.float64, **kwargs)
    low = evaluate(inputs, cotangents, torch.float32, **kwargs)
    for name in ("output", "final_state", *GRADIENT_NAMES):
        value, expected, reference = actual[name], high[name], low[name]
        if name == "output":
            value, expected, reference = (t[:, :end] for t in (value, expected, reference))
        if name in inputs:
            assert value.dtype == inputs[name].dtype, name
        if name not in REDUCTION_NAMES:
            assert_value(value, expected, reference, name)
    z = inputs["z"].double()
    d_skip = cotangents[0].double() * z * torch.sigmoid(z) * inputs["x"].double()
    scales = {
        "A": decay_terms(inputs, cotangents, **kwargs),
        "D": d_skip.abs().sum((0, 1, 3)),
        "dt_bias": high["dt"].abs().sum((0, 1)),
    }
    for name, scale in scales.items():
        assert_reduction(actual[name], high[name], scale, name)


def constant_inputs(dtype, tokens=1, *, x=1.0, dt=0.0, A=-1.0, B=0.0, C=0.0):
    """Constant one-head inputs with 16 channels.

    The default zero ``B`` and ``C`` make the scan contribute nothing, isolating the ``D`` skip
    and ``silu(z)`` gate.
    """

    def full(*shape, value, dtype=torch.float32):
        return torch.full(shape, value, device="cuda", dtype=dtype)

    return {
        "x": full(1, tokens, 1, 16, value=x, dtype=dtype),
        "dt": full(1, tokens, 1, value=dt),
        "A": full(1, value=A),
        "B": full(1, tokens, 1, 16, value=B, dtype=dtype),
        "C": full(1, tokens, 1, 16, value=C, dtype=dtype),
    }


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize(
    ("dtype", "skip", "gate"),
    [
        # Rounding the pre-gate output 1.003 to BF16 first would give 0.73046875.
        pytest.param(torch.bfloat16, 1.003, 1.0, id="bf16_single_rounding"),
        # An FP16 pre-gate output of 1e5 would overflow; the gated output 502.5 is finite.
        pytest.param(torch.float16, 1e5, 0.01, id="fp16_pre_gate_range"),
    ],
)
def test_gate_reads_the_fp32_pre_gate_output(dtype, skip, gate, training):
    """The gated output is rounded once from the FP32 ``Yc + D x``, as from the accumulator."""
    inputs = constant_inputs(dtype)
    inputs["D"] = torch.full((1,), skip, device="cuda")
    inputs["z"] = torch.full_like(inputs["x"], gate)
    leaves = {name: tensor.requires_grad_(training) for name, tensor in inputs.items()}
    with torch.set_grad_enabled(training):
        output, _ = chunk_mamba2(**leaves, impl="fused")
    gate64 = inputs["z"].flatten()[0].double()  # the gate as rounded to the activation dtype
    expected = (skip * gate64 * torch.sigmoid(gate64)).to(dtype)
    assert torch.equal(output, torch.full_like(output, expected.item()))


def test_gated_cotangent_stays_fp32_for_skip_gradients():
    """``dD`` and ``dz`` read ``dy * silu(z)`` and the pre-gate output in FP32.

    A BF16 copy of ``dy * silu(z)`` would put up to 2^-8 relative error into every ``dD`` term.
    """
    generator = torch.Generator(device="cuda").manual_seed(0)
    inputs = constant_inputs(torch.bfloat16, 128)
    shape = inputs["x"].shape
    inputs["x"] = torch.randn(shape, generator=generator, device="cuda").bfloat16()
    inputs["D"] = torch.full((1,), 0.5, device="cuda")
    inputs["z"] = torch.randn(shape, generator=generator, device="cuda").bfloat16()
    d_output = torch.randn(shape, generator=generator, device="cuda").bfloat16()
    actual = evaluate(inputs, [d_output], None)
    high = evaluate(inputs, [d_output], torch.float64)
    torch.testing.assert_close(actual["D"].double(), high["D"], rtol=1e-5, atol=0)
    # dz is rounded once to BF16 (unit roundoff 2^-8). dx reads dy * silu(z) as the BF16
    # operand of its products and skip term, as mamba_ssm does, and is rounded again on output.
    torch.testing.assert_close(actual["z"].double(), high["z"], rtol=2**-7, atol=0)
    torch.testing.assert_close(actual["x"].double(), high["x"], rtol=2**-6, atol=0)


def fp16_gated_cotangent_case():
    """FP16 inputs whose ``dy * silu(z)`` (about 7e4) exceeds the FP16 range."""
    inputs = constant_inputs(torch.float16, x=1e-4)
    inputs["D"] = torch.full((1,), 0.5, device="cuda")
    inputs["z"] = torch.full_like(inputs["x"], 2.0)
    d_output = torch.full_like(inputs["x"], 40000.0)
    return evaluate(inputs, [d_output], None), evaluate(inputs, [d_output], torch.float64)


def test_fp16_gated_cotangent_beyond_fp16_range_keeps_skip_and_gate_gradients():
    """``dy * silu(z)`` of 7e4 exceeds FP16, yet ``dD`` and ``dz`` stay finite and accurate.

    ``dx`` reads that cotangent in FP16 (see the next test).
    """
    actual, high = fp16_gated_cotangent_case()
    torch.testing.assert_close(actual["D"].double(), high["D"], rtol=1e-5, atol=0)
    torch.testing.assert_close(actual["z"].double(), high["z"], rtol=2**-10, atol=0)


@FP16_GATED_COTANGENT_RANGE
def test_fp16_gated_cotangent_beyond_fp16_range_limits_dx():
    """Documented limit: the scan kernels, including ``dx``'s skip term, read FP16 ``dY``."""
    actual, high = fp16_gated_cotangent_case()
    torch.testing.assert_close(actual["x"].double(), high["x"], rtol=2**-10, atol=0)


@pytest.mark.parametrize("raw", [-15.0, -17.0, -20.0])
def test_softplus_keeps_small_step_sizes(raw):
    """``log(1 + exp(dt))`` rounds these step sizes to zero below -16; ``log1p`` keeps them."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    inputs = constant_inputs(torch.bfloat16, 64, dt=raw)
    shape = inputs["x"].shape
    for name in ("x", "B", "C"):
        inputs[name] = torch.randn(shape, generator=generator, device="cuda").bfloat16()
    cotangents = [torch.randn(shape, generator=generator, device="cuda").bfloat16()]
    actual = evaluate(inputs, cotangents, None, dt_softplus=True)
    high = evaluate(inputs, cotangents, torch.float64, dt_softplus=True)
    low = evaluate(inputs, cotangents, torch.float32, dt_softplus=True)
    for name in ("output", "final_state", "x", "dt", "A", "B", "C"):
        assert_value(actual[name], high[name], low[name], name)


def carry_case(tokens, state, cotangent):
    """Zero writes and outputs, a uniform initial state, and a final-state cotangent only."""
    inputs = constant_inputs(torch.bfloat16, tokens, x=0.0)
    inputs["initial_state"] = torch.full((1, 1, 16, 16), state, device="cuda")
    cotangents = [
        torch.zeros_like(inputs["x"]),
        torch.full((1, 1, 16, 16), cotangent, device="cuda"),
    ]
    return inputs, cotangents


def assert_carry_matches_recurrence(inputs, cotangents):
    actual = evaluate(inputs, cotangents, None)
    high = evaluate(inputs, cotangents, torch.float64)
    # The FP32 final state holds the oracle's value rounded to FP32, zero when it underflows.
    high["final_state"] = high["final_state"].float().double()
    assert_matches_recurrence(actual, high)


@pytest.mark.parametrize("decay_chunk", [3, 12], ids=["forward_combine", "backward_combine"])
def test_segment_summaries_scale_a_large_carry_without_underflow(decay_chunk, monkeypatch):
    """A segment's total decay 2^-155 underflows FP32; the decayed 2^79 state does not.

    The steep chunk sits in the first segment, whose decay the forward combine applies to the
    state, or in the second, whose decay the backward combine applies to the state cotangent.
    """
    force_segments(monkeypatch, 512)
    inputs, cotangents = carry_case(1024, 2.0**79, 2.0**40)
    inputs["dt"][:, decay_chunk * 64 : (decay_chunk + 1) * 64] = STEEP_DT
    assert_carry_matches_recurrence(inputs, cotangents)


@pytest.mark.parametrize(
    ("state", "cotangent"),
    [(2.0**79, 2.0**40), (2.0**-80, 2.0**120)],
    ids=["large_state", "large_cotangent"],
)
def test_chunk_carry_scales_without_underflow(state, cotangent):
    """One 64-token chunk decaying by 2^-155 keeps every representable carry and gradient.

    The backward splits the decay of the contracted ``<S_in, dS_out>``: decaying the 2^-80
    state before the product would underflow, and ``exp2`` of the decay alone flushes both cases.
    """
    inputs, cotangents = carry_case(64, state, cotangent)
    inputs["dt"].fill_(STEEP_DT)
    assert_carry_matches_recurrence(inputs, cotangents)


@CHUNK_BWD_PRODUCT_RANGE
@pytest.mark.parametrize("term", ["state", "write"])
def test_undecayed_backward_products_limit_huge_magnitudes(term):
    """Documented limit: ``<S_in, dS_out>`` and the write contraction overflow before decaying.

    Each product is about 2^168 before a decay that brings ``dA`` back into FP32 range.
    """
    if term == "state":
        inputs, cotangents = carry_case(64, 2.0**80, 2.0**80)
        inputs["dt"].fill_(STEEP_DT)
    else:
        inputs, cotangents = carry_case(2, 0.0, 2.0**80)
        inputs["x"][:, 0] = 2.0**80
        inputs["B"].fill_(1.0)
        inputs["dt"] = torch.tensor([1.0, 100 * math.log(2)], device="cuda").view(1, 2, 1)
    actual = evaluate(inputs, cotangents, None)
    assert_matches_recurrence(actual, evaluate(inputs, cotangents, torch.float64), ("dt", "A"))


def entry_read_case():
    """A 2^60 entry state decayed by up to 2^-155 within its chunk, read by every output.

    The last output is 2^-91, and a 2^60 cotangent on it gives ``d initial_state`` of 2^-95.
    """
    inputs = constant_inputs(torch.bfloat16, 64, x=0.0, dt=STEEP_DT, C=1.0)
    inputs["initial_state"] = torch.full((1, 1, 16, 16), 2.0**60, device="cuda")
    d_output = torch.zeros_like(inputs["x"])
    d_output[:, -1] = 2.0**60
    actual = evaluate(inputs, [d_output], None)
    return actual, evaluate(inputs, [d_output], torch.float64)


def test_entry_state_reads_keep_a_representable_product():
    """The output and the state-cotangent scan decay the entry state without early underflow.

    A decay ``exp2(g)`` formed alone flushes the last output and ``d initial_state`` to zero.
    """
    actual, high = entry_read_case()
    assert_matches_recurrence(actual, high, ("output", "final_state", "x", "B", "initial_state"))

    # Paged prefill computes the output inside the scan.
    inputs = constant_inputs(torch.bfloat16, 64, x=0.0, dt=STEEP_DT, C=1.0)
    pool = torch.zeros(2, 1, 16, 16, device="cuda")
    pool[1] = 2.0**60
    state_indices = torch.ones(1, device="cuda", dtype=torch.int32)
    with torch.no_grad():
        output = paged_chunk_mamba2(**inputs, state_cache=pool, state_indices=state_indices)
    torch.testing.assert_close(output.double(), high["output"], rtol=BF16_ROUNDING_RTOL, atol=0)


@CHUNK_BWD_DECAY_RANGE
def test_entry_state_reads_limit_in_chunk_gradients():
    """Documented limit: the chunk backward's ``exp(g)`` on ``dC`` and ``dA`` underflows alone."""
    actual, high = entry_read_case()
    assert_matches_recurrence(actual, high, ("dt", "A", "C"))


def state_write_case():
    """A 2^50 write decayed by 2^-155 within its chunk, reaching the state as 2^-105.

    The scaled write ``x dt exp(g_L - g)`` is a normal BF16 MMA operand, so this is not the
    operand staging limit.
    """
    inputs = constant_inputs(torch.bfloat16, 2, x=0.0, B=1.0)
    inputs["x"][:, 0] = 2.0**50
    inputs["dt"] = torch.tensor([1.0, 155 * math.log(2)], device="cuda").view(1, 2, 1)
    inputs["initial_state"] = torch.zeros(1, 1, 16, 16, device="cuda")
    cotangents = [
        torch.zeros_like(inputs["x"]),
        torch.full((1, 1, 16, 16), 2.0**60, device="cuda"),
    ]
    actual = evaluate(inputs, cotangents, None)
    return actual, evaluate(inputs, cotangents, torch.float64)


def test_state_writes_keep_a_representable_product():
    """The scan decays the write before staging it, keeping the 2^-105 final state.

    A write scale ``dt * exp2(g_L - g)`` formed alone flushes it to zero.
    """
    actual, high = state_write_case()
    assert_matches_recurrence(actual, high, ("output", "final_state", "C", "initial_state"))


@CHUNK_BWD_DECAY_RANGE
def test_state_writes_limit_in_chunk_gradients():
    """Documented limit: the chunk backward's write scale underflows alone in dx, dB, and dA."""
    actual, high = state_write_case()
    assert_matches_recurrence(actual, high, ("x", "B", "dt", "A"))


@RECURRENT_DECAY_RANGE
def test_recurrent_step_limits_a_steep_single_step_decay():
    """Documented limit: one token decaying by 2^-155 flushes a 2^60 state, not 2^-95.

    A single step reaches this only with ``dt * |A|`` above about 87; splitting the decay as the
    chunk carries do costs about 4% of decode on B300.
    """
    inputs = constant_inputs(torch.bfloat16, 1, x=0.0, dt=155 * math.log(2), C=1.0)
    initial_state = torch.full((1, 1, 16, 16), 2.0**60, device="cuda")
    expected, expected_state = recurrent_mamba2(
        **{name: tensor.double() for name, tensor in inputs.items()},
        initial_state=initial_state.double(),
        impl="reference",
        output_final_state=True,
    )
    pool = torch.zeros(2, 1, 16, 16, device="cuda")
    pool[1] = initial_state[0]
    with torch.no_grad():
        output, state = recurrent_mamba2(
            **inputs, initial_state=initial_state, impl="fused", output_final_state=True
        )
        decoded = recurrent_mamba2_decode(
            *(inputs[name][:, 0] for name in ("x", "dt")),
            inputs["A"],
            *(inputs[name][:, 0] for name in ("B", "C")),
            pool,
            torch.ones(1, device="cuda", dtype=torch.int32),
        )
    for value, high in ((output, expected), (decoded, expected[:, 0])):
        torch.testing.assert_close(value.double(), high, rtol=BF16_ROUNDING_RTOL, atol=0)
    for value in (state, pool[1:]):
        torch.testing.assert_close(value.double(), expected_state, rtol=DECAY_SUM_RTOL, atol=0)


def test_decay_gradient_is_free_of_diagonal_cancellation():
    """A 1e8 cotangent on the first token must not swamp the second token's dA term.

    The row-minus-column form of the log-decay cotangent rounded the diagonal 1e8 terms away
    unevenly and returned 11.77 for this dA; the exact value is 5.886.
    """
    inputs = constant_inputs(torch.bfloat16, 2, dt=1.0, B=1.0, C=1.0)
    d_output = torch.ones_like(inputs["x"])
    d_output[:, 0] = 1e8
    actual = evaluate(inputs, [d_output], None)
    high = evaluate(inputs, [d_output], torch.float64)
    torch.testing.assert_close(actual["A"].double(), high["A"], rtol=1e-5, atol=0)
    torch.testing.assert_close(actual["dt"].double(), high["dt"], rtol=1e-5, atol=0)


def test_state_write_decay_gradient_is_free_of_cancellation():
    """A 1e8 write must not swamp the decay gradient of the write before it.

    Forming the write term as ``-w_j`` plus ``sum(w)`` at the last token cancelled the 1e8 write
    in FP32 and reversed this dA (-94.18 instead of +94.18).
    """
    inputs = constant_inputs(torch.bfloat16, 2, dt=1.0, B=1.0)
    inputs["x"][:, 1] = 1e8  # 100139008 in BF16
    inputs["initial_state"] = torch.zeros(1, 1, 16, 16, device="cuda")
    cotangents = [torch.zeros_like(inputs["x"]), torch.ones(1, 1, 16, 16, device="cuda")]
    actual = evaluate(inputs, cotangents, None)
    high = evaluate(inputs, cotangents, torch.float64)
    for name in ("A", "dt", "initial_state"):
        torch.testing.assert_close(actual[name].double(), high[name], rtol=1e-5, atol=0)
    # dx rounds once to BF16 (unit roundoff 2^-8); dB also reads the BF16-staged scaled write.
    torch.testing.assert_close(actual["x"].double(), high["x"], rtol=2**-7, atol=0)
    torch.testing.assert_close(actual["B"].double(), high["B"], rtol=2**-6, atol=0)


@pytest.mark.parametrize("limit", [(0.25, 0.5), (0.25, 0.25)], ids=["interval", "equal"])
def test_clamp_passes_no_gradient_at_its_bounds(limit):
    """Like ``Tensor.clamp``, step sizes exactly at ``dt_limit`` get zero ``ddt``."""
    bounds = torch.tensor([0.25, 0.25, 0.5, 0.5], device="cuda")
    # The adjacent FP32 values on each side of both bounds.
    neighbors = torch.nextafter(
        bounds, bounds + torch.tensor([-1.0, 1.0, -1.0, 1.0], device="cuda")
    )
    values = torch.cat([torch.tensor([0.1, 0.25, 0.3, 0.5, 0.7], device="cuda"), neighbors])
    inputs = constant_inputs(torch.bfloat16, values.numel(), A=0.0, B=1.0, C=1.0)
    inputs["dt"] = values.view(1, -1, 1).clone()
    d_output = torch.ones_like(inputs["x"])
    actual = evaluate(inputs, [d_output], None, dt_limit=limit)
    expected = evaluate(inputs, [d_output], torch.float64, dt_limit=limit)
    torch.testing.assert_close(actual["dt"].double(), expected["dt"], rtol=1e-5, atol=0)
    inside = (values > limit[0]) & (values < limit[1])
    assert actual["dt"][0, ~inside].eq(0).all()


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float16, id="fp16", marks=FP16_STATE_RANGE),
    ],
)
def test_large_state_decays_in_fp32(dtype):
    """An FP32 state element above the FP16 range must decay to its FP32 value without NaNs."""
    inputs = constant_inputs(dtype, 65, x=0.0, dt=4.0 / 65)
    initial_state = torch.zeros(1, 1, 16, 16, device="cuda")
    initial_state[0, 0, 0, 0] = 65536.0
    with torch.no_grad():
        output, state = chunk_mamba2(
            **inputs, initial_state=initial_state, impl="fused", output_final_state=True
        )
        _, expected = recurrent_mamba2(
            **inputs, initial_state=initial_state, impl="reference", output_final_state=True
        )
    assert torch.equal(output, torch.zeros_like(output))
    torch.testing.assert_close(state, expected, rtol=FP32_CARRY_RTOL, atol=0)


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(torch.bfloat16, id="bf16"),
        pytest.param(torch.float16, id="fp16", marks=FP16_COTANGENT_RANGE),
    ],
)
def test_tiny_final_state_cotangent_reaches_decay_gradient(dtype):
    """A final-state cotangent below the FP16 range must still produce ``dA`` and ``ddt``."""
    inputs = constant_inputs(dtype, 64, x=0.0, dt=0.01)
    inputs["initial_state"] = torch.ones(1, 1, 16, 16, device="cuda")
    cotangents = [
        torch.zeros_like(inputs["x"]),
        torch.full((1, 1, 16, 16), 2.0**-25, device="cuda"),
    ]
    actual = evaluate(inputs, cotangents, None)
    high = evaluate(inputs, cotangents, torch.float64)
    assert (high["A"] != 0).all()
    # dA reads the activation-dtype state checkpoint, so it carries source-dtype rounding.
    for name in ("A", "dt"):
        assert_matches_low_precision_reference(
            actual[name], high[name], high[name], name, source_dtype=dtype
        )
    torch.testing.assert_close(
        actual["initial_state"].double(), high["initial_state"], rtol=FP32_CARRY_RTOL, atol=0
    )
