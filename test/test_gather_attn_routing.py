"""CPU-only dispatch coverage using CUDA fake tensors, without launching kernels."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from attn_gym.sparse.gather_attn import Impl, gather_attn
from attn_gym.sparse.gather_attn import impl as implementations
from attn_gym.sparse.gather_attn.api import _select_backend
from attn_gym.sparse.gather_attn.impl import cute


@pytest.fixture(autouse=True)
def clear_fa4_probe():
    probe = cute._fa4_available
    probe.cache_clear()
    yield
    probe.cache_clear()


@pytest.fixture
def triton_backend(monkeypatch):
    backend = SimpleNamespace(gather_attn=Mock())
    monkeypatch.setitem(sys.modules, "attn_gym.sparse.gather_attn.impl.triton", backend)
    monkeypatch.setattr(implementations, "triton", backend, raising=False)
    return backend


@pytest.fixture
def cuda_inputs():
    with FakeTensorMode():
        yield {
            "query": torch.empty(1, 128, 8, 512, device="cuda", dtype=torch.bfloat16),
            "local_kv": torch.empty(1, 1, 8, 512, device="cuda", dtype=torch.bfloat16),
            "sparse_kv": torch.empty(1, 1, 4, 512, device="cuda", dtype=torch.bfloat16),
            "kv_indices": torch.empty(1, 8, 2, device="cuda", dtype=torch.int32),
            "attention_sink": torch.empty(128, device="cuda", dtype=torch.float32),
        }


@pytest.fixture(params=[(10, 0), (10, 3)], ids=["sm100", "sm103"])
def fa4_available(monkeypatch, request):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: request.param)
    available = Mock(return_value=True)
    monkeypatch.setattr(cute, "_fa4_available", available)
    return available


@pytest.mark.parametrize("heads", [1, 24, 64, 96, 127, 128])
def test_default_prefers_cute(cuda_inputs, fa4_available, monkeypatch, heads):
    cuda_inputs["query"] = cuda_inputs["query"][:, :heads]
    cuda_inputs["attention_sink"] = cuda_inputs["attention_sink"][:heads]
    query = cuda_inputs["query"]
    launch = Mock(return_value=(query, None))
    monkeypatch.setattr(cute, "gather_attn", launch)
    assert gather_attn(**cuda_inputs) is query
    launch.assert_called_once()
    fa4_available.assert_called_once_with(True, padded_heads=heads != 128)


@pytest.mark.parametrize(
    "heads,head_dim,dtype,kv_heads,capability",
    [
        (129, 512, torch.bfloat16, 1, (10, 0)),
        (128, 256, torch.bfloat16, 1, (10, 0)),
        (128, 512, torch.float16, 1, (10, 0)),
        (128, 512, torch.float32, 1, (10, 0)),
        (128, 512, torch.bfloat16, 128, (10, 0)),
        (128, 512, torch.bfloat16, 1, (9, 0)),
        (128, 512, torch.bfloat16, 1, (12, 0)),
    ],
)
def test_unsupported_metadata_uses_triton(
    monkeypatch, fa4_available, heads, head_dim, dtype, kv_heads, capability
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    with FakeTensorMode():
        query = torch.empty(1, heads, 8, head_dim, device="cuda", dtype=dtype)
    assert _select_backend(query, None, kv_heads == 1, num_keys=6) == "triton"
    fa4_available.assert_not_called()


@pytest.mark.parametrize("constraint", ["compiling", "deterministic", "missing_fa4", "empty_keys"])
def test_auto_fallback_calls_triton(
    cuda_inputs, fa4_available, triton_backend, monkeypatch, constraint
):
    if constraint == "compiling":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif constraint == "deterministic":
        monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    elif constraint == "missing_fa4":
        fa4_available.return_value = False
    else:
        cuda_inputs["kv_indices"] = cuda_inputs["kv_indices"][..., :0]
        cuda_inputs["sliding_window_size"] = 0
    query = cuda_inputs["query"]
    launch = Mock(return_value=(query, None))
    triton_backend.gather_attn = launch
    assert gather_attn(**cuda_inputs) is query
    launch.assert_called_once()
    if constraint != "missing_fa4":
        fa4_available.assert_not_called()


@pytest.mark.parametrize(
    "impl,kernel_options", [(Impl.FUSED, {"backend": "triton"}), (Impl.REFERENCE, None)]
)
def test_explicit_implementation_is_honored(
    cuda_inputs, fa4_available, triton_backend, monkeypatch, impl, kernel_options
):
    from attn_gym.sparse.gather_attn.impl import reference

    module = reference if impl is Impl.REFERENCE else triton_backend
    query = cuda_inputs["query"]
    launch = Mock(return_value=(query, None))
    monkeypatch.setattr(module, "gather_attn", launch)
    assert gather_attn(**cuda_inputs, impl=impl, kernel_options=kernel_options) is query
    launch.assert_called_once()
    fa4_available.assert_not_called()


@pytest.mark.parametrize("constraint", ["compiling", "deterministic"])
def test_explicit_cute_bypasses_auto_policy(cuda_inputs, fa4_available, monkeypatch, constraint):
    if constraint == "compiling":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    else:
        monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    query = cuda_inputs["query"]
    launch = Mock(return_value=(query, None))
    monkeypatch.setattr(cute, "gather_attn", launch)
    assert gather_attn(**cuda_inputs, kernel_options={"backend": "cute"}) is query
    launch.assert_called_once()
    fa4_available.assert_not_called()


def test_selected_backend_failure_is_not_retried(
    cuda_inputs, fa4_available, triton_backend, monkeypatch
):
    fallback = triton_backend.gather_attn
    monkeypatch.setattr(cute, "gather_attn", Mock(side_effect=RuntimeError("kernel failed")))
    with pytest.raises(RuntimeError, match="kernel failed"):
        gather_attn(**cuda_inputs)
    fallback.assert_not_called()


def test_explicit_cute_does_not_fall_back(cuda_inputs, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    with pytest.raises(ValueError, match="requires SM100"):
        gather_attn(**cuda_inputs, kernel_options={"backend": "cute"})


def test_invalid_inputs_are_not_hidden(cuda_inputs, fa4_available):
    with pytest.raises(ValueError, match="sliding_window_size must be non-negative"):
        gather_attn(**cuda_inputs, sliding_window_size=-1)
    fa4_available.assert_not_called()


def test_missing_fa4_import(monkeypatch):
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", None)
    assert not cute._fa4_available(with_sink=False)


@pytest.mark.parametrize("supports_sink", [False, True])
def test_fa4_sink_feature_probe(monkeypatch, supports_sink):
    def current_kernel(learnable_sink):
        pass

    def old_kernel():
        pass

    monkeypatch.setitem(
        sys.modules, "flash_attn.cute.interface", SimpleNamespace(flash_attn_func=Mock())
    )
    monkeypatch.setitem(
        sys.modules,
        "flash_attn.cute.flash_fwd_mla_sm100",
        SimpleNamespace(
            FlashAttentionMLAForwardSm100=SimpleNamespace(
                __call__=current_kernel if supports_sink else old_kernel
            )
        ),
    )
    assert cute._fa4_available(with_sink=True) is supports_sink


def test_missing_sink_feature_uses_triton(cuda_inputs, fa4_available):
    fa4_available.side_effect = lambda with_sink, **kwargs: not with_sink
    assert (
        _select_backend(
            cuda_inputs["query"],
            cuda_inputs["attention_sink"],
            True,
            num_keys=6,
        )
        == "triton"
    )
    assert _select_backend(cuda_inputs["query"], None, True, num_keys=6) == "cute"


@pytest.mark.parametrize("heads,expected", [(1, "triton"), (64, "triton"), (128, "cute")])
def test_old_fa4_head_limit(cuda_inputs, fa4_available, heads, expected):
    fa4_available.side_effect = lambda with_sink, *, padded_heads: not padded_heads
    query = cuda_inputs["query"][:, :heads]
    assert _select_backend(query, None, True, num_keys=6) == expected


def test_singleton_strided_sink_uses_triton(cuda_inputs, fa4_available):
    query = cuda_inputs["query"][:, :1]
    sink = cuda_inputs["attention_sink"][::2][:1]
    assert sink.is_contiguous() and sink.stride(0) == 2
    assert _select_backend(query, sink, True, num_keys=6) == "triton"
    fa4_available.assert_not_called()


def test_probe_old_fa4_head_limit(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "flash_attn.cute.interface", SimpleNamespace(flash_attn_func=Mock())
    )
    monkeypatch.setitem(sys.modules, "flash_attn.cute.pack_gqa", None)
    assert cute._fa4_available(False)
    assert not cute._fa4_available(False, padded_heads=True)
