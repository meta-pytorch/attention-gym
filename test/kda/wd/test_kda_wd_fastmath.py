"""warpdrive KDA with ``fastmath=False``: non-flushing gate exponentials in every kernel."""

from __future__ import annotations

import importlib
import math
import re
import warnings
from dataclasses import replace

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import chunk_kda
from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs
from attn_gym.warpdrive.ptx_check import capture_ptx, read_ptx

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive KDA path requires an SM100-family GPU",
)

WARPDRIVE = {"backend": "warpdrive"}
D = 128
# One 16-token chunk whose channel-0 decay is 2^-127: subnormal, so the flushing exp2 of
# fastmath=True returns zero there, and its reciprocal is +inf (K_inv = 0 * inf = NaN).
SUBNORMAL_LOG2_DECAY = -127


@pytest.fixture(params=["uncut", "dv_split", "chain"])
def plan(request, monkeypatch) -> str:
    """Pin the forward and backward plans: one item per sequence and head, the shared-prep d_v
    split (prep and prep-prefill kernels), or a four-piece chain (summary, bprop summary)."""
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


def _decay_probe(tokens: int = 64) -> tuple[torch.Tensor, ...]:
    """Zero Q/K/V, so the final state is the entry state decayed per key channel: channel 0 by
    2^-127 within the first chunk, channel 1 by 2^-60, the others not at all."""
    zeros = torch.zeros(1, tokens, 1, D, device="cuda", dtype=torch.bfloat16)
    gate = torch.zeros(zeros.shape, device="cuda")
    gate[0, :16, 0, 0] = SUBNORMAL_LOG2_DECAY / 16 * math.log(2)
    gate[0, :16, 0, 1] = -60 / 16 * math.log(2)
    beta = torch.zeros(zeros.shape[:3], device="cuda")
    state = torch.randn(
        1, 1, D, D, device="cuda", generator=torch.Generator("cuda").manual_seed(3)
    )
    return zeros, gate, beta, state


def _decayed(state: torch.Tensor) -> torch.Tensor:
    decay = torch.ones(D, dtype=torch.float64, device=state.device)
    decay[0], decay[1] = 2.0**SUBNORMAL_LOG2_DECAY, 2.0**-60
    return (state.double() * decay).float()


def test_fastmath_false_keeps_a_subnormal_state_decay(plan: str) -> None:
    """Forward, backward and no-state backward: a 2^-127 chunk decay survives as a subnormal.

    The flushing exp2 zeroes it, so fastmath=True turns the zero key into 0 * inf = NaN.
    """
    zeros, gate, beta, state = _decay_probe()
    leaves = [t.clone().requires_grad_() for t in (zeros, zeros, zeros, gate, beta, state)]
    output, final_state = chunk_kda(
        *leaves[:5],
        initial_state=leaves[5],
        cu_seqlens=cumulative_sequence_offsets([zeros.shape[1]]),
        output_final_state=True,
        fastmath=False,
        kernel_options=WARPDRIVE,
    )
    assert torch.equal(output, torch.zeros_like(output))
    expected = _decayed(state)
    torch.testing.assert_close(final_state, expected, rtol=1e-5, atol=0)
    assert final_state[..., 0].abs().min() > 0
    d_final_state = torch.randn_like(final_state)
    grads = torch.autograd.grad(final_state, leaves, d_final_state)
    torch.testing.assert_close(grads[5], _decayed(d_final_state), rtol=1e-5, atol=0)
    for grad in (*grads[:3], grads[4]):
        assert torch.equal(grad, torch.zeros_like(grad))
    # final = state * exp(sum_t gate_t) per key channel: every token shares one gate gradient.
    # Its 128-term sum carries 16-bit rounding (measured up to 9e-4 of the channel's absolute
    # sum); a flushed channel would be off by about 128^-0.5.
    terms = (d_final_state.double() * expected.double())[0, 0]
    d_gate = grads[3][0, :, 0].double()
    error = (d_gate - terms.sum(0)).abs().amax(0) / terms.abs().sum(0)
    assert error.max() < 1e-2, error.max()

    leaves = [t.clone().requires_grad_() for t in (zeros, zeros, zeros, gate, beta)]
    output, _ = chunk_kda(*leaves, fastmath=False, kernel_options=WARPDRIVE)
    grads = torch.autograd.grad(output, leaves, torch.ones_like(output))
    for tensor in (output, *grads):
        assert torch.equal(tensor, torch.zeros_like(tensor))


def test_fastmath_false_summaries_keep_a_subnormal_decay() -> None:
    """The context-parallel summaries: A = diag(decay), B = C = 0 and R = A^T."""
    zeros, gate, beta, _ = _decay_probe()
    bounds = torch.tensor([[0, zeros.shape[1]]], dtype=torch.int32, device="cuda")
    prepared = chunk_kda_prepare(
        zeros, zeros, zeros, gate, beta, fastmath=False, kernel_options=WARPDRIVE
    )
    identity = torch.eye(D, device="cuda").expand(1, 1, D, D)
    expected = torch.cat((torch.zeros_like(identity), _decayed(identity)), dim=2)
    torch.testing.assert_close(prepared.state_summaries(bounds), expected, rtol=1e-5, atol=0)
    backward = chunk_kda_prepare_backward(
        prepared.saved,
        torch.ones_like(zeros),
        None,
        scale=prepared.scale,
        fastmath=False,
        kernel_options=WARPDRIVE,
    )
    torch.testing.assert_close(backward.state_grad_summaries(bounds), expected, rtol=1e-5, atol=0)


def test_fastmath_false_matches_fastmath_true_bitwise_in_the_gate_range(plan: str) -> None:
    """Gates in [-5, 0] keep every decay normal, where both exponentials agree bit for bit."""
    lengths = [200, 65, 1, 127]
    q, k, value, gate, beta = make_kda_test_inputs(
        sum(lengths), heads=2, seed=5, normalize_qk=True, sigmoid_beta=True
    )
    generator = torch.Generator(device="cuda").manual_seed(6)
    state = torch.randn(len(lengths), 2, D, D, device="cuda", generator=generator) / 8
    d_output = torch.randn(value.shape, device="cuda", generator=generator).to(value.dtype)
    d_final_state = torch.randn_like(state)
    results = []
    for fastmath in (True, False):
        leaves = [t.clone().requires_grad_() for t in (q, k, value, gate, beta, state)]
        output, final_state = chunk_kda(
            *leaves[:5],
            initial_state=leaves[5],
            cu_seqlens=cumulative_sequence_offsets(lengths),
            output_final_state=True,
            fastmath=fastmath,
            kernel_options=WARPDRIVE,
        )
        grads = torch.autograd.grad((output, final_state), leaves, (d_output, d_final_state))
        leaves = [t.clone().requires_grad_() for t in (q, k, value, gate, beta)]
        local, _ = chunk_kda(
            *leaves,
            cu_seqlens=cumulative_sequence_offsets(lengths),
            fastmath=fastmath,
            kernel_options=WARPDRIVE,
        )
        local_grads = torch.autograd.grad(local, leaves, d_output)
        results.append((output, final_state, *grads, local, *local_grads))
    for index, (fast, exact) in enumerate(zip(*results, strict=True)):
        torch.testing.assert_close(exact, fast, rtol=0, atol=0, msg=f"result {index}")


PACKAGE = "attn_gym.linear._delta_rule.warpdrive"
# (module, class, constructor arguments, compile function, its arguments after the op)
GATE_KERNELS = [
    (
        "kda_prefill",
        "KdaPrefill",
        ("bf16", "fp32", 128, 128, 1, True, True, 0, 152),
        ("fp32", "fp32"),
    ),
    ("kda_prep", "KdaPrep", ("bf16", "fp32", 128, 152), ("fp32",)),
    ("kda_summary", "KdaSummary", ("bf16", "fp32", 128, 128, 152), ("fp32",)),
    ("kda_recompute", "KdaRecompute", ("bf16", "fp32", 128, 128, True, 152), ("fp32", "fp32")),
    ("kda_bprop", "KdaBprop", ("bf16", "fp32", 128, 128, True, True, True, 152), ("fp32",)),
    ("kda_bprop_summary", "KdaBpropSummary", ("bf16", 128, 128, True), ()),
]


@pytest.mark.parametrize("case", GATE_KERNELS, ids=lambda case: case[0])
def test_fastmath_false_compiles_no_flushing_math(requires_blackwell, tmp_path, case) -> None:
    """Every kernel that exponentiates gates emits non-FTZ ``ex2``/``rcp`` and no FTZ op."""
    module, cls, args, compile_args = case
    mod = importlib.import_module(f"{PACKAGE}.{module}")
    with capture_ptx(tmp_path) as out, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(mod, f"compile_{module}").__wrapped__(
            getattr(mod, cls)(*args, fastmath=False), *compile_args
        )
    (path,) = out.glob("*.ptx")
    ptx = read_ptx(path)
    assert re.search(r"\bex2\.approx\.f32\b", ptx) and re.search(r"\brcp\.approx\.f32\b", ptx)
    assert ".ftz" not in ptx
