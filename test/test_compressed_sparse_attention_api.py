import importlib
import sys

import pytest
import torch


api = importlib.import_module("attn_gym.sparse.compressed_sparse_attention.api")
csa_package = importlib.import_module("attn_gym.sparse.compressed_sparse_attention")
sparse_package = importlib.import_module("attn_gym.sparse")


def make_arguments():
    batch = 1
    heads = 2
    index_heads = 2
    sequence_length = 4
    head_dim = 4
    index_dim = 4
    compression_rate = 2
    dtype = torch.float32

    def tensor(*shape):
        return torch.empty(shape, dtype=dtype)

    return (
        tensor(batch, heads, sequence_length, head_dim),  # Q
        tensor(batch, index_heads, sequence_length, index_dim),  # Q_I
        tensor(batch, heads, sequence_length, head_dim),  # KV
        tensor(batch, heads, sequence_length, head_dim),  # C_a
        tensor(batch, heads, sequence_length, head_dim),  # C_b
        tensor(batch, heads, sequence_length, head_dim),  # Z_a
        tensor(batch, heads, sequence_length, head_dim),  # Z_b
        tensor(compression_rate, head_dim),  # B_a
        tensor(compression_rate, head_dim),  # B_b
        tensor(batch, sequence_length, index_heads),  # W_I
        tensor(batch, index_heads, sequence_length, index_dim),  # K_Ia
        tensor(batch, index_heads, sequence_length, index_dim),  # K_Ib
        tensor(batch, index_heads, sequence_length, index_dim),  # Z_Ia
        tensor(batch, index_heads, sequence_length, index_dim),  # Z_Ib
        tensor(compression_rate, index_dim),  # B_Ia
        tensor(compression_rate, index_dim),  # B_Ib
        tensor(head_dim),  # KV_norm_weight
        tensor(index_dim),  # compressed_indices_norm_weight
        tensor(head_dim),  # compressed_kv_norm_weight
        tensor(heads),  # attention_sink
        compression_rate,
        1,  # num_topk_blocks
        2,  # sliding_window_size
        4,  # rope_dims
        False,  # share_kv
    )


def fail_loader():
    raise AssertionError("unexpected backend loader call")


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
    monkeypatch.setattr(api, "_load_cute_implementation", lambda: implementation)

    result = api.compressed_sparse_attention(*arguments, backend="cute")

    assert result is expected
    assert calls == [arguments]


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
    module_name = "attn_gym.sparse.compressed_sparse_attention.cute"
    monkeypatch.setitem(sys.modules, module_name, None)

    with pytest.raises(RuntimeError, match="(?i)cute"):
        api.compressed_sparse_attention(*make_arguments(), backend="cute")


def test_missing_flash_attention_dependency_has_clear_error(monkeypatch):
    module_name = "attn_gym.sparse.compressed_sparse_attention.cute"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setitem(sys.modules, "flash_attn", None)

    with pytest.raises(RuntimeError, match="(?i)flash-attn-4"):
        api.compressed_sparse_attention(*make_arguments(), backend="cute")


def test_invalid_backend_is_rejected_without_loading_an_implementation(monkeypatch):
    monkeypatch.setattr(api, "_load_eager_implementation", fail_loader)
    monkeypatch.setattr(api, "_load_triton_implementation", fail_loader)
    monkeypatch.setattr(api, "_load_cute_implementation", fail_loader)

    with pytest.raises(ValueError, match="(?i)backend"):
        api.compressed_sparse_attention(*make_arguments(), backend="cuda")
