"""CPU-only dispatch coverage using CUDA fake tensors, without launching kernels."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from attn_gym.sparse.gather_attn import Impl, RopePair, api, gather_attn
from attn_gym.sparse.gather_attn.api import _select_backend
from attn_gym.sparse.gather_attn.impl import cute


@pytest.fixture(autouse=True)
def clear_fa4_probe():
    probe = cute._fa4_available
    probe.cache_clear()
    yield
    probe.cache_clear()


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


@pytest.mark.parametrize("rope_dim", [0, 64], ids=["bare", "paired"])
@pytest.mark.parametrize("heads", [1, 24, 64, 96, 127, 128])
def test_default_prefers_cute(cuda_inputs, fa4_available, heads, rope_dim):
    query = cuda_inputs["query"][:, :heads]
    sink = cuda_inputs["attention_sink"][:heads]
    rope = query[..., :rope_dim] if rope_dim else None
    assert _select_backend(query, sink, True, num_keys=6, query_rope=rope) == "cute"
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


@pytest.mark.parametrize("rope_dim", [0, 64], ids=["bare", "paired"])
@pytest.mark.parametrize("constraint", ["deterministic", "missing_fa4", "empty_keys"])
def test_auto_fallback_selects_triton(
    cuda_inputs, fa4_available, monkeypatch, constraint, rope_dim
):
    if constraint == "deterministic":
        monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    elif constraint == "missing_fa4":
        fa4_available.return_value = False
    num_keys = 0 if constraint == "empty_keys" else 6
    query = cuda_inputs["query"]
    rope = query[..., :rope_dim] if rope_dim else None
    assert (
        _select_backend(
            query, cuda_inputs["attention_sink"], True, num_keys=num_keys, query_rope=rope
        )
        == "triton"
    )
    if constraint != "missing_fa4":
        fa4_available.assert_not_called()


@pytest.mark.parametrize("rope_dim", [0, 64], ids=["bare", "paired"])
@pytest.mark.parametrize("backend", [None, "triton", "cute"])
def test_fused_fake_outputs_have_no_cpu_bookkeeping(cuda_inputs, fa4_available, backend, rope_dim):
    """Common output metadata is CUDA-only, so Inductor can capture the opaque boundary."""
    options = None if backend is None else {"backend": backend}
    inputs = cuda_inputs.copy()
    if rope_dim:
        for name in ("query", "local_kv", "sparse_kv"):
            tensor = inputs[name]
            inputs[name] = RopePair(tensor, torch.empty_like(tensor[..., :rope_dim]))
    kwargs = {"scale": 0.0625} if rope_dim else {}
    result = gather_attn(**inputs, **kwargs, kernel_options=options)
    assert result.shape == cuda_inputs["query"].shape
    assert result.device == cuda_inputs["query"].device


@pytest.mark.parametrize("backend", [None, "cute", "triton"])
def test_backend_request_is_forwarded(cuda_inputs, monkeypatch, backend):
    query = cuda_inputs["query"]
    launch = Mock(return_value=(query, None))
    select = Mock(return_value="cute")
    monkeypatch.setattr(api, "_select_backend", select)
    monkeypatch.setattr(api, "fused_gather_attn", launch)
    options = None if backend is None else {"backend": backend}
    assert gather_attn(**cuda_inputs, kernel_options=options) is query
    assert launch.call_args.kwargs["backend"] == (backend or "cute")
    assert select.call_count == (1 if backend is None else 0)


@pytest.mark.parametrize("fallback", ["missing-fa4", "deterministic"])
@pytest.mark.parametrize("recompute", [True, False])
def test_auto_fallback_reserves_only_triton_tape(monkeypatch, fallback, recompute):
    """A fallback must not retain the GB-sized FA4 tape it will never consume."""
    monkeypatch.setattr(cute, "_environment_supported", lambda *args: False)
    if fallback == "deterministic":
        monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    saved = []

    def save(tensor):
        saved.append(tensor)
        return tensor

    with FakeTensorMode(), torch.autograd.graph.saved_tensors_hooks(save, lambda x: x):
        q = torch.empty(1, 128, 8192, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        kv = torch.empty(1, 1, 8192, 512, device="cuda", dtype=torch.bfloat16)
        indices = torch.empty(1, 8192, 32, device="cuda", dtype=torch.int32)
        gather_attn(
            q,
            kv,
            kv,
            indices,
            sliding_window_size=512,
            kernel_options={"bwd_recompute_p": recompute},
        )
    maps = [t for t in saved if t.dtype == torch.int32 and t.ndim == 2]
    assert len(maps) == 2
    assert all(t.device.type == "cuda" for t in saved)
    assert sum(t.numel() * t.element_size() for t in maps) == (8192 * 32 + 8192 + 1) * 4
    # Q, local KV, sparse KV, output only: no additional BF16 FA4 KV/residual/P buffers.
    assert sum(t.dtype == torch.bfloat16 for t in saved) == 4


def test_reference_implementation_is_honored(cuda_inputs, fa4_available, monkeypatch):
    from attn_gym.sparse.gather_attn.impl import reference

    query = cuda_inputs["query"]
    launch = Mock(return_value=(query, None))
    monkeypatch.setattr(reference, "gather_attn", launch)
    assert gather_attn(**cuda_inputs, impl=Impl.REFERENCE) is query
    launch.assert_called_once()
    fa4_available.assert_not_called()


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
    monkeypatch.setitem(sys.modules, "flash_attn.cute.config", None)
    assert cute._fa4_available(False)
    assert not cute._fa4_available(False, padded_heads=True)


def test_reference_does_not_require_kernel_dependencies():
    """Importing the shared schemas must not import Triton or FA4 for CPU users."""
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import torch
sys.modules['triton'] = None
sys.modules['flash_attn'] = None
from attn_gym.sparse import gather_attn
q = torch.ones(1, 1, 1, 2)
indices = torch.zeros(1, 1, 1, dtype=torch.int32)
actual = gather_attn(q, q, q, indices, sliding_window_size=0, impl='reference')
torch.testing.assert_close(actual, q)
""",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
