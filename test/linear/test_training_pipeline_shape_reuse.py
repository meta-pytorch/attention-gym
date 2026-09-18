"""Composed training must keep backend reuse and CUDA Graph replay, not just leaf shapes."""

from __future__ import annotations

import sys
from collections.abc import Callable
from functools import partial
from types import FunctionType

import pytest
import torch

pytest.importorskip("cutlass")
triton = pytest.importorskip("triton")

from triton.runtime.autotuner import Autotuner, Heuristics
from triton.runtime.jit import JITFunction

from attn_gym.linear import causal_conv1d, chunk_gdn, chunk_kda, gate_transform, l2norm
from attn_gym.linear.kda.bwd.triton.l2norm_bwd import l2norm_bwd_kernel
from attn_gym.linear.kda.masking import active_token_mask, mask_inactive_tokens

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="composed training gate requires Hopper or newer",
)


def _inputs(variant: str, tokens: int, sequences: int) -> tuple[torch.Tensor, ...]:
    heads, dim = 2, 128
    gate_shape = (1, tokens, heads, dim) if variant == "kda" else (1, tokens, heads)
    tensors = (
        torch.randn(1, tokens, 3 * heads * dim, device="cuda", dtype=torch.bfloat16),
        torch.randn(3 * heads * dim, 4, device="cuda", dtype=torch.bfloat16) * 0.1,
        torch.randn(gate_shape, device="cuda", dtype=torch.bfloat16),
        torch.full((heads,), -3.0, device="cuda"),
        torch.zeros(gate_shape[2:], device="cuda"),
        torch.randn(1, tokens, heads, device="cuda"),
        torch.randn(sequences, heads, dim, dim, device="cuda") * 0.01,
    )
    return tuple(tensor.requires_grad_() for tensor in tensors)


def _forward(
    qkv: torch.Tensor,
    weight: torch.Tensor,
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    bias: torch.Tensor,
    raw_beta: torch.Tensor,
    initial: torch.Tensor,
    offsets: torch.Tensor,
    *,
    variant: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    active = active_token_mask(qkv, offsets)
    convolved = causal_conv1d(
        mask_inactive_tokens(qkv, active), weight, cu_seqlens=offsets, activation="silu"
    )
    convolved = mask_inactive_tokens(convolved, active)
    # Q/K are the same interleaved views produced by a packed QKV projection.
    q, k, v = convolved.view(1, qkv.shape[1], 3, 2, 128).unbind(2)
    q, k = l2norm(q, cu_seqlens=offsets), l2norm(k, cu_seqlens=offsets)
    gate = gate_transform(
        mask_inactive_tokens(raw_gate, active),
        a_log,
        bias,
        kind="bounded" if variant == "kda" else "softplus",
        lower_bound=-1.0 if variant == "kda" else None,
        impl="fused",
    )
    beta = mask_inactive_tokens(raw_beta.sigmoid(), active)
    kwargs = {"autotune": False} if variant == "kda" else {}
    output, final_state = (chunk_kda if variant == "kda" else chunk_gdn)(
        q,
        k,
        v,
        mask_inactive_tokens(gate, active),
        beta,
        initial,
        cu_seqlens=offsets,
        output_final_state=True,
        **kwargs,
    )
    return mask_inactive_tokens(output, active), final_state


def _step(
    forward: Callable[..., tuple[torch.Tensor, torch.Tensor | None]],
    inputs: tuple[torch.Tensor, ...],
    offsets: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    output, state = forward(*inputs, offsets)
    assert state is not None
    gradients = torch.autograd.grad(
        (output, state), inputs, grad_outputs=(torch.ones_like(output), torch.ones_like(state))
    )
    return (output, state, *gradients)


def _backends() -> dict[str, JITFunction | FunctionType]:
    """Find loaded Triton kernels and CuTe jit_cache factories, including lazy imports."""
    kernels = {}
    for name, module in tuple(sys.modules.items()):
        if module is None or not name.startswith("attn_gym.linear."):
            continue
        for kernel in tuple(vars(module).values()):
            while isinstance(kernel, (Autotuner, Heuristics)):
                kernel = kernel.fn
            if isinstance(kernel, JITFunction):
                kernels[f"{kernel.fn.__module__}.{kernel.fn.__qualname__}"] = kernel
            elif isinstance(kernel, FunctionType) and "cache_namespace" in kernel.__dict__:
                kernels[f"{kernel.__module__}.{kernel.__qualname__}"] = kernel
    return kernels


def _backend_entries() -> dict[str, int]:
    """Count loaded specializations, not whether they were compiled or loaded from disk."""
    entries = {}
    for name, kernel in _backends().items():
        if isinstance(kernel, JITFunction):
            cache = kernel.device_caches.get(torch.cuda.current_device())
            count = 0 if cache is None else len(cache[0])
        else:
            count = kernel.cache_info().currsize
        if count:
            entries[name] = count
    return entries


@pytest.mark.parametrize("variant", ["kda", "gdn"])
def test_training_pipeline_reuses_backend_specializations(variant: str, monkeypatch):
    """Changing T/N in one metadata/config class must not compile new training leaves."""
    # Different cached tuning winners may load different pre-existing tile variants.
    # Freeze that choice here; the graph tests below exercise the default tuner.
    monkeypatch.setattr(l2norm_bwd_kernel, "configs", [triton.Config({"BT": 16}, num_warps=4)])
    monkeypatch.setattr(l2norm_bwd_kernel, "cache", {})
    torch.cuda.synchronize()
    for kernel in _backends().values():
        if isinstance(kernel, JITFunction):
            kernel.device_caches.clear()
        else:
            kernel.cache_clear()
    torch.manual_seed(501)
    forward = partial(_forward, variant=variant)
    expected_entries = None
    for tokens, boundaries in (
        (128, (0, 42, 83, 127)),
        (192, (0, 0, 96, 190)),
        (192, (0, 0, 64, 128, 190)),
    ):
        offsets = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
        inputs = _inputs(variant, tokens, len(boundaries) - 1)
        results = _step(forward, inputs, offsets)
        torch.cuda.synchronize()
        assert all(torch.isfinite(result).all() for result in results)
        # Physical padding cannot contribute to the projected inputs' gradient.
        padding_gradient = results[2][:, boundaries[-1] :]
        torch.testing.assert_close(
            padding_gradient, torch.zeros_like(padding_gradient), rtol=0, atol=0
        )
        entries = _backend_entries()
        if expected_entries is None:
            expected_entries = entries
        else:
            assert entries == expected_entries


@pytest.mark.parametrize("variant", ["kda", "gdn"])
def test_training_pipeline_dynamic_fullgraph(variant: str, fresh_compile_cache):
    """The composed public forward and its backward accept varying packed capacities."""
    torch.manual_seed(502)
    eager = partial(_forward, variant=variant)
    compiled = torch.compile(eager, fullgraph=True, dynamic=True)
    with torch._dynamo.config.patch(error_on_recompile=True):
        for tokens, boundaries in ((128, (0, 42, 83, 127)), (192, (0, 0, 64, 128, 190))):
            offsets = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
            inputs = _inputs(variant, tokens, len(boundaries) - 1)
            # T=128 must not duck-shape with the fixed 128-wide state dimensions.
            for index in (0, 2, 5):
                torch._dynamo.mark_dynamic(inputs[index], 1)
            torch._dynamo.mark_dynamic(inputs[-1], 0)
            torch._dynamo.mark_dynamic(offsets, 0)
            expected = _step(eager, inputs, offsets)
            actual = _step(compiled, inputs, offsets)
            for result, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(result, reference, rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("variant", ["kda", "gdn"])
def test_training_pipeline_cuda_graph_replays_inputs_and_boundaries(variant: str):
    """Replay a complete forward/backward step with fresh inputs and empty/padded requests."""
    torch.manual_seed(503)
    inputs = _inputs(variant, 192, 4)
    offsets = torch.tensor((0, 48, 96, 144, 192), device="cuda", dtype=torch.int32)
    forward = partial(_forward, variant=variant)
    for _ in range(3):
        _step(forward, inputs, offsets)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _step(forward, inputs, offsets)
    for boundaries in ((0, 64, 64, 128, 192), (0, 0, 65, 65, 160)):
        with torch.no_grad():
            inputs[0].normal_()
            inputs[1].mul_(0.95)
            inputs[2].add_(0.1)
            inputs[-1].mul_(0.9)
            offsets.copy_(torch.tensor(boundaries, device="cuda", dtype=torch.int32))
        graph.replay()
        torch.cuda.synchronize()
        actual = tuple(tensor.clone() for tensor in captured)
        expected = _step(forward, inputs, offsets)
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
