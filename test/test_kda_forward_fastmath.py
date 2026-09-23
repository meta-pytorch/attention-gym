"""Behavioral regressions for KDA's fastmath policy."""

import importlib

import pytest
import torch

from attn_gym.linear import chunk_kda
from attn_gym.linear.kda.constants import is_sm100_kda_capability
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

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

    # Token 15 writes state; token 16 reads exp(-5.5). Rebasing splits this into
    # exp(-88) * exp(82.5), so flushing the subnormal loses a visible contribution.
    golden = gate[0, 16, 0, 0].double().exp().expand(128).to(q.dtype)
    for fastmath in (False, True, False):
        output, _ = operation(q, k, v, gate, beta, scale=1.0, autotune=False, fastmath=fastmath)
        expected = torch.zeros_like(golden) if fastmath else golden
        torch.testing.assert_close(output[0, 16, 0], expected, rtol=0, atol=0)
        dv, dbeta = torch.autograd.grad(output[0, 16, 0, 0], (v, beta))
        torch.testing.assert_close(dv[0, 15, 0, 0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(dbeta[0, 15, 0], expected[0].to(beta.dtype), rtol=0, atol=0)

    default, _ = operation(q, k, v, gate, beta, scale=1.0, autotune=False)
    assert torch.count_nonzero(default[0, 16, 0]) == 0


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
        *inputs, cu_seqlens=offsets, fastmath=False, autotune=False
    )
    d_output = torch.randn_like(inputs[2])
    d_state = torch.randn_like(state)

    def backward(saved):
        handle = stages.chunk_kda_prepare_backward(
            saved,
            d_output,
            state,
            scale=prepared.scale,
            schedule=prepared.schedule,
            fastmath=False,
            autotune=False,
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
