"""Backend shape reuse, independent numerics, and replay for training preprocessing."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from attn_gym.linear import gate_transform
from attn_gym.linear._delta_rule.gate import _softplus_uses_cute
from attn_gym.linear._delta_rule.triton import softplus_gate as softplus_backend
from attn_gym.linear.kda.bwd.triton.l2norm_bwd import l2norm_bwd_kernel
from attn_gym.linear.kda.fwd.triton.l2norm_fwd import l2norm, l2norm_fwd_kernel
from attn_gym.testing.kda import assert_matches_low_precision_reference, assert_relative_rms_within

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def fixed_l2norm_schedule(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Separate backend reuse from the unchanged, exact-NB autotuning policy."""
    torch.manual_seed(123)
    if request.node.get_closest_marker("full_autotune") is not None:
        return
    config = next(
        config
        for config in l2norm_bwd_kernel.configs
        if config.kwargs["BT"] == 16 and config.num_warps == 4
    )
    monkeypatch.setattr(l2norm_bwd_kernel, "configs", [config])
    monkeypatch.setattr(l2norm_bwd_kernel, "cache", {})


def cache_size(kernel) -> int:
    return len(kernel.device_caches[torch.cuda.current_device()][0])


def make_view(shape: tuple[int, ...], dtype: torch.dtype, layout: str) -> torch.Tensor:
    if layout == "qkv":
        batch, tokens, heads, dim = shape
        storage = torch.full((batch, tokens, 3, heads, dim), torch.nan, device="cuda", dtype=dtype)
        result = storage[:, :, 1]
    elif layout in ("strided", "strided3"):
        step = 2 if layout == "strided" else 3
        storage = torch.full(
            (*shape[:-1], shape[-1] * step), torch.nan, device="cuda", dtype=dtype
        )
        result = storage[..., ::step]
    elif layout == "transposed":
        # Head pitch includes T, and B/T cannot be flattened in physical order.
        order = (0, 2, 1, *range(3, len(shape)))
        storage = torch.empty(tuple(shape[i] for i in order), device="cuda", dtype=dtype)
        result = storage.permute(order)
    elif layout == "misaligned":
        storage = torch.full(
            (torch.Size(shape).numel() + 1,), torch.nan, device="cuda", dtype=dtype
        )
        result = storage[1:].view(shape)
    else:
        result = torch.empty(shape, device="cuda", dtype=dtype)
    return result.uniform_(-1, 1)


def run_operation(
    kind: str, inputs: tuple[torch.Tensor, ...], metadata: torch.Tensor | None = None
) -> torch.Tensor:
    if kind == "l2norm":
        return l2norm(inputs[0], cu_seqlens=metadata)
    assert not _softplus_uses_cute(inputs[0])
    return gate_transform(*inputs, kind="softplus")


def reference(
    kind: str,
    inputs: tuple[torch.Tensor, ...],
    cotangent: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    values = tuple(x.detach().to(dtype).requires_grad_() for x in inputs)
    if kind == "l2norm":
        x = values[0]
        output = x * torch.rsqrt(x.square().sum(-1, keepdim=True) + 1e-6)
    else:
        raw, amplitude, bias = values
        output = -amplitude.exp().view(1, 1, -1) * F.softplus(raw + bias)
    return output, *torch.autograd.grad(output, values, cotangent.to(dtype))


def check_numerics(
    kind: str,
    inputs: tuple[torch.Tensor, ...],
    output: torch.Tensor,
    grads: tuple[torch.Tensor, ...],
    cotangent: torch.Tensor,
    active: int | None = None,
) -> None:
    """Pair shared pointwise and RMS budgets; detailed operator contracts live in existing tests."""
    if active is not None:
        inputs = (inputs[0][:, :active],)
        output, grads, cotangent = (
            output[:, :active],
            (grads[0][:, :active],),
            cotangent[:, :active],
        )
    if output.numel() == 0:
        return
    golden = reference(kind, inputs, cotangent, torch.float64)
    eager = reference(kind, inputs, cotangent, torch.float32)
    names = ("output", "dx") if kind == "l2norm" else ("output", "dx", "d_A_log", "d_dt_bias")
    if kind == "l2norm":
        tolerances = [(2e-5, 2e-6) if inputs[0].dtype == torch.float32 else (2e-2, 2e-3)] * 2
    else:
        tolerances = [(2e-6, 2e-6), (1e-2, 8e-3), (3e-4, 3e-4), (3e-4, 3e-4)]
    for name, actual, high, low, (rtol, atol) in zip(
        names, (output, *grads), golden, eager, tolerances, strict=True
    ):
        torch.testing.assert_close(actual.float(), low.float(), rtol=rtol, atol=atol)
        assert_matches_low_precision_reference(
            actual, high, low.to(actual.dtype), name, source_dtype=inputs[0].dtype
        )
        parameter = name in ("d_A_log", "d_dt_bias")
        # Parameter gradients reduce B*T rows; use the existing softplus RMS policy.
        max_eps = 4 * math.sqrt(cotangent.shape[0] * cotangent.shape[1]) if parameter else 4
        assert_relative_rms_within(actual, high, name, max_eps=max_eps, source_dtype=actual.dtype)


def make_inputs(
    kind: str, batch: int, tokens: int, dtype: torch.dtype, layout: str
) -> tuple[torch.Tensor, ...]:
    shape = (batch, tokens, 8, 128) if kind == "l2norm" else (batch, tokens, 32)
    raw = make_view(shape, dtype, layout).requires_grad_()
    if kind == "l2norm":
        return (raw,)
    return (
        raw,
        make_view((32,), torch.float32, "strided").requires_grad_(),
        make_view((32,), torch.float32, "strided").requires_grad_(),
    )


@pytest.mark.parametrize("kind", ["l2norm", "softplus"])
@pytest.mark.parametrize("layout", ["compact", "strided", "strided3", "transposed", "misaligned"])
def test_pointwise_reuses_backend_cache(kind, layout):
    kernels = (
        (l2norm_fwd_kernel, l2norm_bwd_kernel.fn)
        if kind == "l2norm"
        else (softplus_backend.softplus_gate_fwd_kernel, softplus_backend.softplus_gate_bwd_kernel)
    )
    for kernel in kernels:
        kernel.device_caches.clear()
    counts = []
    # Keep the generic head-pitch alignment class fixed as well as the tile schedule.
    shapes = [(2, 65), (3, 97), (2, 129)]
    alignment_transition = kind == "softplus" and layout == "transposed"
    if alignment_transition:
        shapes += [(2, 128), (2, 144)]
    for batch, tokens in shapes:
        inputs = make_inputs(kind, batch, tokens, torch.bfloat16, layout)
        output = run_operation(kind, inputs)
        cotangent = make_view(tuple(output.shape), output.dtype, layout)
        grads = torch.autograd.grad(output, inputs, cotangent)
        check_numerics(kind, inputs, output, grads, cotangent)
        counts.append(tuple(cache_size(kernel) for kernel in kernels))
    assert counts[:3] == [(1, 1)] * 3, counts
    if alignment_transition:
        assert counts[3] == counts[4], counts
        assert all(1 <= count <= 2 for pair in counts for count in pair), counts
    if kind == "softplus":
        expected_stride = {"compact": 1, "misaligned": 1, "strided": 2}.get(layout, 0)
        assert softplus_backend._softplus_element_stride(inputs[0]) == expected_stride
        assert (
            softplus_backend._softplus_element_stride(inputs[0].unsqueeze(-1)) == expected_stride
        )


@pytest.mark.full_autotune
def test_l2norm_retunes_without_recompiling(monkeypatch):
    """A new NB retunes the same two binaries, not a new binary per row count."""
    configs = [
        config
        for config in l2norm_bwd_kernel.configs
        if config.kwargs["BT"] in (16, 32) and config.num_warps == 4
    ]
    monkeypatch.setattr(l2norm_bwd_kernel, "configs", configs)
    monkeypatch.setattr(l2norm_bwd_kernel, "cache", {})
    monkeypatch.setattr(l2norm_bwd_kernel, "cache_results", False)
    trials = []

    def one_trial(call, quantiles):
        call()
        trials.append(1)
        return [1.0] * len(quantiles)

    monkeypatch.setattr(l2norm_bwd_kernel, "do_bench", one_trial)
    l2norm_bwd_kernel.fn.device_caches.clear()
    counts = []
    for tokens in (65, 97):
        inputs = make_inputs("l2norm", 2, tokens, torch.bfloat16, "compact")
        output = run_operation("l2norm", inputs)
        torch.autograd.grad(output, inputs, torch.randn_like(output))
        counts.append(cache_size(l2norm_bwd_kernel.fn))
    assert len(trials) == 4
    assert len(l2norm_bwd_kernel.cache) == 2
    assert counts == [2, 2], counts


@pytest.mark.parametrize("layout", ["compact", "qkv", "transposed"])
def test_packed_l2norm_reuses_backend_cache(layout):
    kernels = (l2norm_fwd_kernel, l2norm_bwd_kernel.fn)
    for kernel in kernels:
        kernel.device_caches.clear()
    counts = []
    # N changes independently, across a power-of-two boundary; then T changes.
    for tokens, sequences in ((65, 3), (65, 4), (65, 5), (97, 5)):
        inputs = make_inputs("l2norm", 1, tokens, torch.bfloat16, layout)
        active = tokens - 7
        metadata = torch.linspace(0, active, sequences + 1, device="cuda").to(torch.int32)
        metadata[1] = 0  # Empty sequence, no host reads of metadata in the operation.
        cotangent = make_view(tuple(inputs[0].shape), inputs[0].dtype, layout)
        with torch.no_grad():
            inputs[0][:, active:] = torch.nan
            cotangent[:, active:] = torch.nan
        output = run_operation("l2norm", inputs, metadata)
        grads = torch.autograd.grad(output, inputs, cotangent)
        check_numerics("l2norm", inputs, output, grads, cotangent, active)
        counts.append(tuple(cache_size(kernel) for kernel in kernels))
    assert counts == [(1, 1)] * 4, counts


@pytest.mark.parametrize("kind", ["l2norm", "softplus"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_pointwise_fullgraph_dynamic(kind, dtype, fresh_compile_cache):
    compiled = torch.compile(
        lambda *inputs: run_operation(kind, inputs), fullgraph=True, dynamic=True
    )
    for batch, tokens in ((2, 65), (3, 97)):
        inputs = make_inputs(kind, batch, tokens, dtype, "compact")
        output = compiled(*inputs)
        cotangent = torch.randn_like(output)
        grads = torch.autograd.grad(output, inputs, cotangent)
        check_numerics(kind, inputs, output, grads, cotangent)


@pytest.mark.parametrize("kind", ["l2norm", "softplus"])
def test_pointwise_graph_replay_changed_values(kind):
    inputs = make_inputs(kind, 1, 97, torch.bfloat16, "strided")
    metadata = (
        torch.tensor([0, 0, 35, 65], device="cuda", dtype=torch.int32)
        if kind == "l2norm"
        else None
    )
    cotangent = torch.randn_like(
        inputs[0], dtype=inputs[0].dtype if kind == "l2norm" else torch.float32
    )
    for _ in range(3):
        output = run_operation(kind, inputs, metadata)
        torch.autograd.grad(output, inputs, cotangent)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    previous_override = torch._C._override_stale_capture_stream()
    torch.autograd.graph.set_override_stale_capture_stream(True)
    try:
        with torch.cuda.graph(graph):
            output = run_operation(kind, inputs, metadata)
            grads = torch.autograd.grad(output, inputs, cotangent)
    finally:
        torch.autograd.graph.set_override_stale_capture_stream(previous_override)
    for active in (81, 0, 47):
        with torch.no_grad():
            for value in inputs:
                value.uniform_(-1, 1)
            cotangent.normal_()
            if metadata is not None:
                metadata.copy_(
                    torch.tensor([0, 0, active // 2, active], device="cuda", dtype=torch.int32)
                )
                inputs[0][:, active:] = torch.nan
                cotangent[:, active:] = torch.nan
        graph.replay()
        torch.cuda.synchronize()
        check_numerics(
            kind, inputs, output, grads, cotangent, active if metadata is not None else None
        )
