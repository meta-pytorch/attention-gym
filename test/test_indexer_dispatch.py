"""Public selection policy and nondifferentiable output contracts."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch._dynamo.testing import CompileCounterWithBackend

from attn_gym.sparse.indexer import lightning_indexer, ops
from attn_gym.testing.indexer import assert_indexer_selection, make_indexer_test_inputs


def require_backend(backend: str | None = None) -> str:
    """Gate only the dependencies and hardware used by this test's selected route."""
    if torch.version.hip or not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    capability = torch.cuda.get_device_capability()
    selected = backend or ("cute" if capability == (10, 0) else "triton")
    if selected == "cute":
        if capability != (10, 0):
            pytest.skip("SM100 required for CuTe")
        pytest.importorskip("cutlass.cute")
    else:
        if capability[0] < 9:
            pytest.skip("Hopper or newer required for Triton")
        pytest.importorskip("triton.tools.tensor_descriptor")
    return selected


@pytest.mark.parametrize(
    "capability,backend,expected",
    [
        ((9, 0), "auto", "triton"),
        ((10, 0), "auto", "cute"),
        ((10, 3), "auto", "triton"),
        ((12, 0), "auto", "triton"),
        ((10, 0), "triton", "triton"),
        ((9, 0), "cute", "cute"),
    ],
)
@pytest.mark.parametrize("fails", [False, True])
def test_backend_dispatch_uses_input_device(monkeypatch, capability, backend, expected, fails):
    """Dispatch once on the input device and propagate launch errors without retrying."""
    q = torch.empty(1, 2, 2, 16)
    k = torch.empty(1, 2, 16)
    weights = torch.empty(1, 2, 2)
    failure = RuntimeError("selected backend failed")
    launch = Mock(
        return_value=torch.empty(1, 2, 1, dtype=torch.int32),
        side_effect=failure if fails else None,
    )
    other = Mock(side_effect=AssertionError("the nonselected backend ran"))
    device_capability = Mock(return_value=capability)
    monkeypatch.setattr(torch.cuda, "get_device_capability", device_capability)
    for name in ("cute", "triton"):
        monkeypatch.setitem(
            sys.modules,
            f"attn_gym.sparse.indexer.impl.{name}",
            SimpleNamespace(launch=launch if name == expected else other),
        )
    if fails:
        with pytest.raises(RuntimeError) as exc:
            ops._indexer_cuda(q, k, weights, 1, True, backend)
        assert exc.value is failure
    else:
        assert ops._indexer_cuda(q, k, weights, 1, True, backend) is launch.return_value
    launch.assert_called_once_with(q, k, weights, 1, True)
    other.assert_not_called()
    if backend == "auto":
        device_capability.assert_called_once_with(q.device)
    else:
        device_capability.assert_not_called()


@pytest.mark.parametrize(
    "options",
    [
        {"backend": "auto"},
        {"backend": "missing"},
        {"other": "triton"},
        {"backend": "triton", "other": "x"},
        {"backend": None},
    ],
)
def test_invalid_backend_options(options):
    """Reject unknown options before launching a backend."""
    q, k, weights = torch.zeros(1, 2, 2, 16), torch.zeros(1, 2, 16), torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="unsupported lightning_indexer kernel options"):
        lightning_indexer(q, k, weights, 1, kernel_options=options)
    with pytest.raises(ValueError, match="kernel_options are not supported"):
        lightning_indexer(q, k, weights, 1, impl="reference", kernel_options=options)


def test_fused_cpu_rejected():
    """CPU callers receive a clear error rather than a dispatcher implementation dump."""
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        lightning_indexer(torch.zeros(1, 2, 2, 16), torch.zeros(1, 2, 16), torch.zeros(1, 2, 2), 1)


@pytest.mark.parametrize("impl", ["reference", "fused"])
def test_selection_does_not_train_scoring_weights(impl):
    """Gathered values train, but no gradient flows through integer selection."""
    if impl == "fused":
        require_backend()
    device = "cuda" if impl == "fused" else "cpu"
    q = torch.randn(1, 16, 2, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 16, 16, device=device, dtype=torch.bfloat16, requires_grad=True)
    weights = torch.randn(1, 16, 2, device=device, dtype=torch.bfloat16, requires_grad=True)
    indices = lightning_indexer(q, k, weights, 4, impl=impl)
    assert indices.dtype == torch.int32
    assert not indices.requires_grad and indices.grad_fn is None
    values = torch.randn(1, 16, device=device, requires_grad=True)
    selected = values[:, None, :].expand(-1, 16, -1).gather(-1, indices.long())
    gradients = torch.autograd.grad(selected.sum(), (values, q, k, weights), allow_unused=True)
    assert gradients[0] is not None
    assert gradients[0].sum() == indices.numel()
    assert gradients[1:] == (None, None, None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_auto_fullgraph_uses_device_backend():
    """One dynamic public graph matches the explicitly selected device backend."""
    expected_backend = require_backend()
    counter = CompileCounterWithBackend("inductor")
    compiled = torch.compile(lightning_indexer, fullgraph=True, dynamic=True, backend=counter)
    # Distinct initial dimensions avoid Dynamo's incidental batch == heads duck-shape guard.
    for batch, tokens in ((3, 17), (4, 33), (5, 65)):
        q = torch.randn(batch, tokens, 2, 16, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(batch, tokens, 16, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(batch, tokens, 2, device="cuda", dtype=torch.bfloat16)
        expected = lightning_indexer(
            q, k, weights, 4, causal=True, kernel_options={"backend": expected_backend}
        )
        actual = compiled(q, k, weights, 4, causal=True)
        assert_indexer_selection(expected, q, k, weights, 4, True)
        assert_indexer_selection(actual, q, k, weights, 4, True)
    assert counter.frame_count == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "backend,dtype,tokens,heads,dim,topk",
    [
        (None, torch.bfloat16, 65, 2, 16, 16),
        ("cute", torch.bfloat16, 65, 2, 16, 16),
        ("triton", torch.bfloat16, 65, 2, 16, 16),
        ("triton", torch.float16, 129, 3, 48, 37),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_backend_cuda_graph_replay(backend, dtype, tokens, heads, dim, topk, causal):
    """All public fused routes replay after static inputs change."""
    require_backend(backend)
    inputs = make_indexer_test_inputs(tokens, heads, dim, dtype)
    options = {"backend": backend} if backend else None

    def run():
        return lightning_indexer(*inputs, topk, causal=causal, kernel_options=options)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = run()
    for _ in range(3):
        for tensor in inputs:
            tensor.normal_()
        graph.replay()
        assert_indexer_selection(actual, *inputs, topk, causal)
        assert_indexer_selection(run(), *inputs, topk, causal)
