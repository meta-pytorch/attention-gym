import importlib
import sys

import pytest
import torch


api = importlib.import_module("attn_gym.sparse.compressed_sparse_attention.api")
csa_package = importlib.import_module("attn_gym.sparse.compressed_sparse_attention")
sparse_package = importlib.import_module("attn_gym.sparse")


def make_arguments(*, share_kv=False):
    batch, heads, sequence_length, head_dim = 2, 3, 5, 8
    index_heads, index_dim, compression_rate = 2, 4, 2
    kv_heads = 1 if share_kv else heads
    index_kv_heads = 1 if share_kv else index_heads
    return (
        torch.randn(batch, heads, sequence_length, head_dim),
        torch.randn(batch, index_heads, sequence_length, index_dim),
        torch.randn(batch, kv_heads, sequence_length, head_dim),
        torch.randn(batch, kv_heads, sequence_length, head_dim),
        torch.randn(batch, kv_heads, sequence_length, head_dim),
        torch.randn(batch, kv_heads, sequence_length, head_dim),
        torch.randn(batch, kv_heads, sequence_length, head_dim),
        torch.randn(compression_rate, head_dim),
        torch.randn(compression_rate, head_dim),
        torch.randn(batch, sequence_length, index_heads),
        torch.randn(batch, index_kv_heads, sequence_length, index_dim),
        torch.randn(batch, index_kv_heads, sequence_length, index_dim),
        torch.randn(batch, index_kv_heads, sequence_length, index_dim),
        torch.randn(batch, index_kv_heads, sequence_length, index_dim),
        torch.randn(compression_rate, index_dim),
        torch.randn(compression_rate, index_dim),
        torch.randn(head_dim),
        torch.randn(index_dim),
        torch.randn(head_dim),
        torch.randn(heads),
        compression_rate,
        3,
        5,
        4,
        share_kv,
    )


def fail_loader():
    raise AssertionError("unexpected backend loader call")


def reset_cute_backend(monkeypatch):
    monkeypatch.setattr(api, "_cute_implementation", None)
    monkeypatch.setattr(api, "_cute_initialization_error", None)


def test_compressed_sparse_attention_is_publicly_exported():
    assert csa_package.compressed_sparse_attention is api.compressed_sparse_attention
    assert sparse_package.compressed_sparse_attention is api.compressed_sparse_attention
    assert "compressed_sparse_attention" in csa_package.__all__
    assert "compressed_sparse_attention" in sparse_package.__all__


@pytest.mark.parametrize("backend_kwargs", [{}, {"backend": "eager"}], ids=["default", "explicit"])
def test_eager_dispatch(monkeypatch, backend_kwargs):
    arguments = make_arguments()
    expected = object()
    calls = []

    def implementation(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(api, "_load_eager_implementation", lambda: implementation)
    monkeypatch.setattr(api, "_load_triton_implementation", fail_loader)
    monkeypatch.setattr(api, "_load_cute_implementation", fail_loader)

    result = api.compressed_sparse_attention(*arguments, **backend_kwargs)

    assert result is expected
    assert calls == [arguments]


def test_triton_dispatch(monkeypatch):
    arguments = make_arguments()
    expected = object()
    calls = []

    def implementation(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)
    monkeypatch.setattr(api, "_load_triton_implementation", lambda: implementation)
    monkeypatch.setattr(api, "_load_cute_implementation", fail_loader)

    result = api.compressed_sparse_attention(*arguments, backend="triton")

    assert result is expected
    assert calls == [arguments]


def test_cute_dispatch(monkeypatch):
    arguments = make_arguments()
    expected = object()
    calls = []

    def implementation(*args):
        calls.append(args)
        return expected

    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)
    monkeypatch.setattr(api, "_load_triton_implementation", fail_loader)
    monkeypatch.setattr(api, "_load_cute_implementation", fail_loader)
    monkeypatch.setattr(api, "_cute_implementation", implementation)

    result = api.compressed_sparse_attention(*arguments, backend="cute")

    assert result is expected
    assert calls == [arguments]


def test_cute_dispatch_reaches_registered_op_without_loading_during_trace(monkeypatch):
    if api._cute_implementation is None:
        pytest.skip(f"CuTe backend is unavailable: {api._cute_initialization_error}")

    captured_graphs = []

    def capture_backend(graph_module, _example_inputs):
        captured_graphs.append(graph_module)
        return lambda *args: (torch.empty_like(args[0]),)

    def run(*args):
        return api.compressed_sparse_attention(*args, backend="cute")

    monkeypatch.setattr(api, "_load_cute_implementation", fail_loader)
    monkeypatch.setattr(api, "_validate_cute_dependencies", fail_loader)
    torch._dynamo.reset()
    compiled = torch.compile(run, backend=capture_backend, fullgraph=True)
    result = compiled(*make_arguments(share_kv=True))

    assert result.shape == (2, 3, 5, 8)
    assert result.dtype == torch.float32
    assert len(captured_graphs) == 1
    assert any(
        node.target is torch.ops.attention_gym._cute_compressed_sparse_attention_forward.default
        for node in captured_graphs[0].graph.nodes
    )


def test_unavailable_triton_backend_has_clear_error(monkeypatch):
    module_name = "attn_gym.sparse.compressed_sparse_attention.triton"
    monkeypatch.setitem(sys.modules, module_name, None)

    with pytest.raises(RuntimeError, match="(?i)triton"):
        api.compressed_sparse_attention(*make_arguments(), backend="triton")


def test_missing_external_triton_dependency_has_clear_error(monkeypatch):
    module_name = "attn_gym.sparse.compressed_sparse_attention.triton"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setitem(sys.modules, "triton", None)

    with pytest.raises(RuntimeError, match="(?i)install triton"):
        api.compressed_sparse_attention(*make_arguments(), backend="triton")


def test_unavailable_cute_backend_has_clear_error(monkeypatch):
    reset_cute_backend(monkeypatch)
    module_name = "attn_gym.sparse.compressed_sparse_attention.cute"
    monkeypatch.setitem(sys.modules, module_name, None)
    api._initialize_cute_backend()

    with pytest.raises(RuntimeError, match="(?i)cute"):
        api.compressed_sparse_attention(*make_arguments(), backend="cute")


def test_missing_flash_attention_dependency_has_clear_error(monkeypatch):
    reset_cute_backend(monkeypatch)
    module_name = "attn_gym.sparse.compressed_sparse_attention.cute"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setitem(sys.modules, "flash_attn", None)
    api._initialize_cute_backend()

    with pytest.raises(RuntimeError, match="(?i)flash-attn-4"):
        api.compressed_sparse_attention(*make_arguments(), backend="cute")


def test_incompatible_cute_dependency_version_has_clear_error(monkeypatch):
    versions = {
        distribution: expected
        for distribution, _module, expected in api._CUTE_RUNTIME_DEPENDENCIES
    }
    versions["nvidia-cutlass-dsl"] = "0.0.0"
    monkeypatch.setattr(api.importlib, "import_module", lambda _module: object())
    monkeypatch.setattr(api.metadata, "version", versions.__getitem__)

    with pytest.raises(
        RuntimeError,
        match=r"nvidia-cutlass-dsl==4\.5\.2 is required; found 0\.0\.0",
    ):
        api._validate_cute_dependencies()


def test_cute_backend_initialization_runs_once(monkeypatch):
    reset_cute_backend(monkeypatch)
    implementation = object()
    load_calls = 0

    def load():
        nonlocal load_calls
        load_calls += 1
        return implementation

    monkeypatch.setattr(api, "_load_cute_implementation", load)
    api._initialize_cute_backend()
    api._initialize_cute_backend()

    assert api._cute_implementation is implementation
    assert load_calls == 1


def test_cute_initialization_failure_is_deferred_until_backend_use(monkeypatch):
    reset_cute_backend(monkeypatch)
    error = ValueError("broken optional backend")

    def load():
        raise error

    monkeypatch.setattr(api, "_load_cute_implementation", load)
    api._initialize_cute_backend()

    assert api._cute_initialization_error is error
    with pytest.raises(RuntimeError, match="broken optional backend"):
        api.compressed_sparse_attention(*make_arguments(), backend="cute")


def test_invalid_backend_is_rejected_without_loading_an_implementation(monkeypatch):
    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)
    monkeypatch.setattr(api, "_load_triton_implementation", fail_loader)
    monkeypatch.setattr(api, "_cute_implementation", fail_loader)

    with pytest.raises(ValueError, match="(?i)backend"):
        api.compressed_sparse_attention(*make_arguments(), backend="cuda")


@pytest.mark.parametrize("backend", ["eager", "triton", "cute"])
def test_shared_shape_validation_happens_before_backend_loading(monkeypatch, backend):
    arguments = list(make_arguments())
    arguments[9] = torch.randn(2, 5, 1)
    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)
    monkeypatch.setattr(api, "_load_triton_implementation", fail_loader)
    monkeypatch.setattr(api, "_cute_implementation", fail_loader)

    with pytest.raises(ValueError, match="W_I must have shape"):
        api.compressed_sparse_attention(*arguments, backend=backend)


def test_shared_scalar_validation_rejects_bool_as_an_integer(monkeypatch):
    arguments = list(make_arguments())
    arguments[20] = True
    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)

    with pytest.raises(TypeError, match="compression_rate must be a Python int"):
        api.compressed_sparse_attention(*arguments)


def test_shared_dtype_validation_happens_before_backend_loading(monkeypatch):
    arguments = list(make_arguments())
    arguments[17] = arguments[17].double()
    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)

    with pytest.raises(TypeError, match="compressed_indices_norm_weight must have dtype"):
        api.compressed_sparse_attention(*arguments)


def test_shared_kv_contract_accepts_shared_and_expanded_physical_heads(monkeypatch):
    arguments = list(make_arguments(share_kv=True))
    arguments[4] = arguments[4].expand(-1, 3, -1, -1)
    arguments[12] = arguments[12].expand(-1, 2, -1, -1)
    expected = object()
    monkeypatch.setattr(api, "_load_eager_implementation", lambda: lambda *args: expected)

    assert api.compressed_sparse_attention(*arguments) is expected
