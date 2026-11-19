"""FP16 warpdrive KDA forward and backward across the model gate range, against FP64 and the
fused FP16 backend.

FP16 cannot hold the in-chunk decay factors 2^g and 2^-g (their ratio reaches 2^121 at the
gate bound), so the FP16 kernels form the in-chunk products from TF32 decay operands. The last
tests pin the documented backward limits on hand-set cases, including the dq headroom that BF16
shares.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import chunk_kda
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    assert_relative_rms_within,
    assert_rms_matches_low_precision_reference,
    cumulative_sequence_offsets,
    kda_reference,
    make_kda_test_inputs,
)

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive KDA path requires an SM100-family GPU",
)

LENGTHS = [200, 65, 1, 127]
HEADS = 2
D = 128
# Uniform per-token gates up to the gradient guideline, and Kimi-style bounded sigmoid gates.
GATES = [-0.1, -0.5, -2.0, -5.25, "sigmoid"]
FUSED = {"backend": "fused", "autotune": False}


@pytest.fixture(params=["uncut", "dv_split", "chain"])
def plan(request, monkeypatch) -> str:
    """Pin the forward and backward plans: the prefill, the shared-prep d_v split (prep and
    prep-prefill) or a four-piece chain (summary and prefill)."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda

    pieces, tiles = {"uncut": (0, 1), "dv_split": (0, 2), "chain": (4, 1)}[request.param]
    forward, backward = kda.ForwardPlan.build, kda.BackwardPlan.build
    monkeypatch.setattr(
        kda.ForwardPlan,
        "build",
        lambda *args: replace(forward(*args), pieces=pieces, tiles_per_head=tiles),
    )
    monkeypatch.setattr(
        kda.BackwardPlan, "build", lambda *args: replace(backward(*args), pieces=pieces)
    )
    return request.param


def _inputs(gate: float | str, stateful: bool, seed: int = 0) -> tuple[torch.Tensor, ...]:
    gates = {"sigmoid_gate_bound": 5.45} if gate == "sigmoid" else {"gate_value": gate}
    q, k, v, g, beta = make_kda_test_inputs(
        sum(LENGTHS),
        heads=HEADS,
        seed=seed,
        dtype=torch.half,
        normalize_qk=True,
        sigmoid_beta=True,
        **gates,
    )
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    shape = (len(LENGTHS), HEADS, D, D)
    state = torch.randn(shape, device="cuda", generator=generator) / 8 if stateful else None
    return q, k, v, g, beta, state


@pytest.mark.parametrize("stateful", [False, True], ids=["no-state", "state"])
@pytest.mark.parametrize("gate", GATES, ids=str)
def test_fp16_forward_matches_fp64_across_the_gate_range(plan, gate, stateful) -> None:
    q, k, v, g, beta, state = _inputs(gate, stateful)
    cu_seqlens = cumulative_sequence_offsets(LENGTHS)

    def run(**options):
        return chunk_kda(
            q,
            k,
            v,
            g,
            beta,
            initial_state=None if state is None else state.clone(),
            cu_seqlens=cu_seqlens,
            output_final_state=stateful,
            **options,
        )

    with torch.no_grad():
        output, final_state = run(kernel_options={"backend": "warpdrive"})
        fused_output, fused_final = run(kernel_options=FUSED)
        expected, expected_final = kda_reference(
            *(t.double() for t in (q, k, v, g, beta)),
            initial_state=None if state is None else state.double(),
            cu_seqlens=cu_seqlens,
            output_final_state=stateful,
        )
    assert_relative_rms_within(output, expected, "output", max_eps=1.0, source_dtype=torch.half)
    assert_matches_low_precision_reference(
        output, expected, fused_output, "output", source_dtype=torch.half
    )
    if stateful:
        for check in (
            assert_matches_low_precision_reference,
            assert_rms_matches_low_precision_reference,
        ):
            check(final_state, expected_final, fused_final, "final state", source_dtype=torch.half)


# The d_v split only tiles the forward, so the backward runs the uncut and chain plans.
@pytest.mark.parametrize("plan", ["uncut", "chain"], indirect=True)
@pytest.mark.parametrize("states", ["no-state", "exit", "entry+exit"])
@pytest.mark.parametrize("gate", GATES, ids=str)
def test_fp16_backward_matches_fp64_across_the_gate_range(plan, gate, states) -> None:
    """The local backward, the exit-cotangent backward, and the native stateful backward with
    its d_initial_state, all through ``chunk_kda``."""
    q, k, v, g, beta, state = _inputs(gate, stateful=states == "entry+exit")
    cu_seqlens = cumulative_sequence_offsets(LENGTHS)
    generator = torch.Generator(device="cuda").manual_seed(1)
    d_output = torch.randn(v.shape, device="cuda", generator=generator).half()
    final_shape = (len(LENGTHS), HEADS, D, D)
    d_final_state = torch.randn(final_shape, device="cuda", generator=generator) / 8
    output_final_state = states != "no-state"

    def grads(op, *inputs, **options):
        leaves = [t.detach().clone().requires_grad_() for t in inputs]
        if state is not None:
            options["initial_state"] = leaves[5]
        output, final_state = op(
            *leaves[:5], cu_seqlens=cu_seqlens, output_final_state=output_final_state, **options
        )
        if not output_final_state:
            return torch.autograd.grad(output, leaves, d_output.to(output.dtype))
        cotangents = (d_output.to(output.dtype), d_final_state.to(final_state.dtype))
        return torch.autograd.grad((output, final_state), leaves, cotangents)

    inputs = (q, k, v, g, beta) + (() if state is None else (state,))
    expected = grads(kda_reference, *(t.double() for t in inputs))
    actual = grads(chunk_kda, *inputs, kernel_options={"backend": "warpdrive"})
    fused = grads(chunk_kda, *inputs, kernel_options=FUSED)
    names = ("dq", "dk", "dv", "dgate", "dbeta", "d_initial_state")[: len(inputs)]
    for name, got, low, high in zip(names, actual, fused, expected, strict=True):
        assert_relative_rms_within(got, high, name, max_eps=2.0, source_dtype=torch.half)
        assert_matches_low_precision_reference(got, high, low, name, source_dtype=torch.half)


# ---- documented backward limits (the KDA gate contract in docs/linear.md) -------------------


def _one_head(tokens: int) -> dict[str, torch.Tensor]:
    """Zeroed fp32 inputs and cotangents of one sequence with one head, for hand-set cases."""
    shape = (1, tokens, 1, D)
    names = ("q", "k", "v", "gate", "d_output")
    case = {name: torch.zeros(shape, device="cuda") for name in names}
    case["beta"] = torch.zeros(shape[:3], device="cuda")
    case["d_final_state"] = torch.zeros(1, 1, D, D, device="cuda")
    return case


def _one_head_grads(case: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """``dq, dk, dv, dgate, dbeta`` of ``case`` from warpdrive with ``dtype`` Q/K/V, and from
    the FP64 reference under ``dtype=torch.float64``."""
    wide = torch.float64 if dtype is torch.float64 else torch.float32
    leaves = [case[name].to(dtype, copy=True).requires_grad_() for name in ("q", "k", "v")]
    leaves += [case[name].to(wide, copy=True).requires_grad_() for name in ("gate", "beta")]
    op, options = chunk_kda, {"kernel_options": {"backend": "warpdrive"}}
    if dtype is torch.float64:
        op, options = kda_reference, {}
    output, final_state = op(
        *leaves,
        cu_seqlens=cumulative_sequence_offsets([case["q"].shape[1]]),
        scale=1.0,
        output_final_state=True,
        **options,
    )
    cotangents = (case["d_output"].to(output.dtype), case["d_final_state"].to(final_state.dtype))
    grads = torch.autograd.grad((output, final_state), leaves, cotangents)
    return dict(zip(("dq", "dk", "dv", "dgate", "dbeta"), grads, strict=True))


def _assert_matches_fp64(
    case: dict[str, torch.Tensor], dtype: torch.dtype, max_eps: float
) -> None:
    """Every gradient of ``case`` from warpdrive matches the FP64 reference."""
    actual = _one_head_grads(case, dtype)
    expected = _one_head_grads(case, torch.float64)
    for name, got in actual.items():
        assert_relative_rms_within(got, expected[name], name, max_eps=max_eps, source_dtype=dtype)


def test_fp16_backward_is_accurate_at_the_largest_fp16_state_cotangent() -> None:
    """FP16 stages the FP32 state cotangent itself as an FP16 operand; at 65504, the largest
    FP16 value, every gradient stays accurate. One token with gate ``-ln 2``, key ``0.5 e0`` and
    an exit cotangent of 65504 on that key. docs/linear.md records what happens past it."""
    case = _one_head(1)
    case["gate"].fill_(-math.log(2.0))
    case["beta"].fill_(0.5)
    case["k"][0, 0, 0, 0] = 0.5
    case["d_final_state"][0, 0, 0, 0] = 65504.0
    _assert_matches_fp64(case, torch.half, max_eps=1.0)


def test_fp16_backward_is_accurate_at_the_smallest_fp16_k_restore() -> None:
    """FP16 stages ``K_restore = k 2^(g_last - g)``: key ``2^-11 e0`` followed by 13 tokens that
    halve that channel leaves the smallest FP16 subnormal ``2^-24``, and the exit cotangent's
    term ``4096 * 2^-24`` must still reach every gradient. docs/linear.md records the flush one
    halving later."""
    decays = 13
    case = _one_head(decays + 1)
    case["k"][0, 0, 0, 0] = 2.0**-11
    case["k"][0, 1:, 0, 1] = 1.0
    case["beta"].fill_(1.0)
    case["gate"][0, 1:, 0, 0] = -math.log(2.0)
    case["d_final_state"][0, 0, :, 0] = 4096.0
    _assert_matches_fp64(case, torch.half, max_eps=1.0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def test_backward_dq_is_accurate_below_the_documented_headroom(dtype) -> None:
    """The dq accumulator holds ``k 2^-g dA``, ``dA[i, j] = dO_i . U_j``, before the ``2^g`` scale
    cancels it. At a uniform ``-5.25`` gate ``2^-g`` reaches ``2^114`` at token 14: with a unit
    key component and a unit value there and ``dO = 128`` at token 15 (``dA = 2^14``), every
    gradient stays accurate. docs/linear.md records the overflow at ``dA = 2^15``."""
    case = _one_head(16)
    case["gate"].fill_(-5.25)
    for name in ("k", "q"):
        case[name][0, :, 0, 1] = 1.0
    for name, token in (("k", 14), ("q", 15)):
        case[name][0, token, 0, :2] = torch.tensor([1.0, 0.0])
    case["beta"].fill_(1.0)
    case["v"][0, 14] = 1.0
    case["d_output"][0, 15] = 128.0
    _assert_matches_fp64(case, dtype, max_eps=2.0)
