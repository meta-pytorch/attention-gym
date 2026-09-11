"""Public selection policy and nondifferentiable output contracts."""

import sys
from types import SimpleNamespace

import pytest
import torch

from attn_gym.sparse.indexer import lightning_indexer, ops


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
def test_backend_dispatch_uses_input_device(monkeypatch, capability, backend, expected):
    """Route by the input device, allowing explicit overrides without retrying."""
    q = torch.empty(1, 2, 2, 16)
    k = torch.empty(1, 2, 16)
    weights = torch.empty(1, 2, 2)
    calls = []

    def device_capability(device):
        assert device == q.device
        return capability

    def launch(*args):
        calls.append(args)
        return torch.empty(1, 2, 1, dtype=torch.int32)

    def reject(*args):
        pytest.fail("the nonselected backend ran")

    monkeypatch.setattr(torch.cuda, "get_device_capability", device_capability)
    for name in ("cute", "triton"):
        monkeypatch.setitem(
            sys.modules,
            f"attn_gym.sparse.indexer.impl.{name}",
            SimpleNamespace(launch=launch if name == expected else reject),
        )
    actual = ops._indexer_cuda(q, k, weights, 1, True, backend)
    assert actual.shape == (1, 2, 1)
    assert len(calls) == 1
    assert all(a is b for a, b in zip(calls[0][:3], (q, k, weights)))
    assert calls[0][3:] == (1, True)


@pytest.mark.parametrize("options", [{"backend": "missing"}, {"other": "triton"}])
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
    if impl == "fused" and (
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9
    ):
        pytest.skip("fused indexer requires Hopper or newer")
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
    capability = torch.cuda.get_device_capability()
    if capability[0] < 9:
        pytest.skip("fused indexer requires Hopper or newer")
    expected_backend = "cute" if capability == (10, 0) else "triton"
    compiled = torch.compile(lightning_indexer, fullgraph=True, dynamic=True)
    for tokens in (17, 33):
        q = torch.randn(1, tokens, 2, 16, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, tokens, 16, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(1, tokens, 2, device="cuda", dtype=torch.bfloat16)
        expected = lightning_indexer(
            q, k, weights, 4, causal=True, kernel_options={"backend": expected_backend}
        )
        actual = compiled(q, k, weights, 4, causal=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
