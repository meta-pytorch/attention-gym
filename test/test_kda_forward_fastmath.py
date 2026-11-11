"""Behavioral regressions for KDA's fastmath policy."""

import importlib

import pytest
import torch

from attn_gym.linear import chunk_kda
from attn_gym.linear.kda.constants import MAX_GATE_LOWER_BOUND_MAGNITUDE, is_sm100_kda_capability
from attn_gym.testing.kda import cumulative_sequence_offsets, kda_reference, make_kda_test_inputs

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the fused KDA core requires CUDA capability 8.0 or newer",
)
requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_sm100_kda_capability(torch.cuda.get_device_capability()),
    reason="the BF16 KDA intra engine requires SM100/SM103",
)


def test_cudnn_precision_request_is_rejected_before_cuda_dispatch(monkeypatch):
    cudnn = importlib.import_module("attn_gym.linear.kda.impl.cudnn")

    def reached_native(_q):
        raise RuntimeError("reached native cuDNN dispatch")

    monkeypatch.setattr(cudnn, "validate_cudnn_available", reached_native)
    q = torch.zeros(1, 64, 1, 128, dtype=torch.bfloat16)
    beta = torch.ones(1, 64, 1)
    gate = torch.zeros_like(q, dtype=torch.float32)
    with pytest.raises(ValueError, match="cuDNN KDA cannot honor fastmath=False"):
        chunk_kda(q, q, q, gate, beta, kernel_options={"backend": "cudnn"}, fastmath=False)
    for kwargs in ({}, {"fastmath": True}):
        with pytest.raises(RuntimeError, match="reached native cuDNN dispatch"):
            chunk_kda(q, q, q, gate, beta, kernel_options={"backend": "cudnn"}, **kwargs)


@requires_blackwell
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_public_fastmath_preserves_decay_across_strip_boundary(compiled):
    """Both fastmath modes keep a decayed term that crosses a 16-token strip boundary.

    The intra engine splits each pair's decay into two strip-rebased factors. Rebasing at the
    strip's first row put this pair's row factor at exp(-88), which the flush-to-zero ``ex2``
    of ``fastmath=True`` zeroed; the engine now shifts the rebase by 2^64, so both factors stay
    normal and the engine has no fastmath switch.
    """
    pytest.importorskip("cutlass")
    q = torch.zeros(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16)
    q[..., 0] = 1
    k = q.clone()
    v = torch.ones_like(q, requires_grad=True)
    gate = torch.full(q.shape, -5.5, device="cuda")
    beta = torch.zeros(q.shape[:-1], device="cuda")
    beta[:, 15] = 1
    beta.requires_grad_()
    operation = torch.compile(chunk_kda, fullgraph=True) if compiled else chunk_kda

    # Token 15 writes state; token 16 reads exp(-5.5).
    golden = gate[0, 16, 0, 0].double().exp().expand(128).to(q.dtype)
    for fastmath in (False, True, None):
        options = {} if fastmath is None else {"fastmath": fastmath}
        output, _ = operation(
            q, k, v, gate, beta, scale=1.0, kernel_options={"autotune": False}, **options
        )
        torch.testing.assert_close(output[0, 16, 0], golden, rtol=0, atol=0)
        dv, dbeta = torch.autograd.grad(output[0, 16, 0, 0], (v, beta))
        torch.testing.assert_close(dv[0, 15, 0, 0], golden[0], rtol=0, atol=0)
        torch.testing.assert_close(dbeta[0, 15, 0], golden[0].to(beta.dtype), rtol=0, atol=0)


@requires_blackwell
@pytest.mark.parametrize("gate_value", [-5.9, -MAX_GATE_LOWER_BOUND_MAGNITUDE])
def test_default_fastmath_gate_gradient_at_gate_bound(gate_value):
    """The default fastmath keeps the gate gradient accurate at constant gates near the bound.

    The backward intra kernel used to derive each row's in-subchunk factor from its
    earlier-key factor, which the flush-to-zero exp2 zeroed for a 16-row subchunk's last row,
    dropping that row's in-subchunk terms (rel-L2 0.8-3.2 across seeds).
    """
    pytest.importorskip("cutlass")
    inputs = make_kda_test_inputs(128, normalize_qk=True, gate_value=gate_value, seed=3)
    generator = torch.Generator(device="cuda").manual_seed(4)
    d_output = torch.randn(inputs[2].shape, device="cuda", generator=generator).to(inputs[2].dtype)
    leaves = [value.detach().clone().requires_grad_() for value in inputs]
    output, _ = chunk_kda(*leaves, kernel_options={"autotune": False})
    (d_gate,) = torch.autograd.grad(output, leaves[3], d_output)
    reference = [value.detach().double().requires_grad_() for value in inputs]
    expected, _ = kda_reference(*reference, output_final_state=False)
    (expected_d_gate,) = torch.autograd.grad(expected, reference[3], d_output.double())
    error = (d_gate.double() - expected_d_gate).norm() / expected_d_gate.norm()
    assert error < 1e-2


@requires_cuda
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "ragged"])
def test_backward_factor_recomputation_preserves_fastmath(packed):
    pytest.importorskip("cutlass")
    stages = importlib.import_module("attn_gym.linear.kda.stages")
    torch.manual_seed(17)
    inputs = make_kda_test_inputs(128, normalize_qk=True, gate_value=-5.5)
    offsets = cumulative_sequence_offsets([65, 63]) if packed else None
    state = torch.randn(2 if packed else 1, 1, 128, 128, device="cuda") / 32
    prepared = stages.chunk_kda_prepare(
        *inputs, cu_seqlens=offsets, fastmath=False, kernel_options={"autotune": False}
    )
    d_output = torch.randn_like(inputs[2])
    d_state = torch.randn_like(state)

    def backward(saved):
        handle = stages.chunk_kda_prepare_backward(
            saved,
            d_output,
            state,
            scale=prepared.scale,
            fastmath=False,
            kernel_options={"autotune": False},
        )
        return handle.run(d_state)

    saved = backward(prepared.saved)
    recomputed = backward(prepared.saved._replace(aqk=None, akk=None))
    for actual, expected in zip(recomputed, saved, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_state_summary_fastmath_preserves_subnormal_decay(reverse):
    """Exercise the compiler's FTZ policy on an isolated FP32 decay."""
    pytest.importorskip("cutlass")
    summaries = importlib.import_module("attn_gym.linear._delta_rule.cute")
    zeros = torch.zeros(1, 64, 1, 128, dtype=torch.bfloat16, device="cuda")
    cumulative_gate = torch.zeros_like(zeros, dtype=torch.float32)
    steps = torch.arange(1, 65, device="cuda", dtype=torch.float32)
    cumulative_gate[0, :, 0, 0] = steps * (-130.0 / 64)
    cumulative_gate[0, :, 0, 1] = steps * (-0.5 / 64)
    bounds = torch.tensor([[0, 64]], dtype=torch.int32, device="cuda")
    if reverse:
        aqk = torch.zeros(1, 64, 1, 64, dtype=zeros.dtype, device=zeros.device)
        operation = summaries.build_state_grad_summaries
        args = (zeros, zeros, zeros, zeros, aqk, cumulative_gate, 1.0, bounds)
    else:
        operation = summaries.build_state_summaries
        args = (zeros, zeros, zeros, cumulative_gate, bounds)
    results = {flag: operation(*args, fastmath=flag) for flag in (False, True)}
    expected = torch.zeros_like(results[False])
    expected[0, 0, 128:].diagonal().copy_(cumulative_gate[0, -1, 0].double().exp2().float())
    torch.testing.assert_close(results[False], expected, rtol=2e-6, atol=0)
    assert results[False][0, 0, 128, 0].item() == 2.0**-130
    expected[0, 0, 128, 0] = 0
    torch.testing.assert_close(results[True], expected, rtol=2e-6, atol=0)
