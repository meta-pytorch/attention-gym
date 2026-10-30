"""Dynamic graph and kernel-reuse contracts for Triton gather attention."""

from collections import Counter

import pytest
import torch

from attn_gym.sparse.gather_attn import gather_attn


def _cuda():
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("NVIDIA CUDA required")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Ampere or newer required")


def _inputs(
    tokens, candidates, *, heads=2, dim=16, share_kv=True, dtype=torch.float32, inner_stride=1
):
    torch.manual_seed(817)
    kv_heads = 1 if share_kv else heads
    q = torch.randn(1, tokens, heads, dim, device="cuda", dtype=dtype).transpose(1, 2) * 0.1
    local = torch.randn(
        1, kv_heads, tokens * 2, dim * inner_stride, device="cuda", dtype=dtype
    ).mul_(0.1)[:, :, ::2, ::inner_stride]
    sparse = torch.randn(
        1, kv_heads, candidates * 2, dim * inner_stride, device="cuda", dtype=dtype
    ).mul_(0.1)[:, :, ::2, ::inner_stride]
    indices = torch.tensor([0, 1, -1], device="cuda", dtype=torch.int32).repeat(1, tokens, 1)
    sink = torch.linspace(-0.4, 0.6, heads * 2, device="cuda", dtype=torch.float32)[::2]
    cu_q = torch.tensor([0, 5, tokens], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 4, candidates], device="cuda", dtype=torch.int32)
    return q, local, sparse, indices, sink, cu_q, cu_k


def _forward_and_grads(tensors, indices, cu_q, cu_k, **kwargs):
    q, local, sparse, sink = tensors
    out = gather_attn(
        q,
        local,
        sparse,
        indices,
        sink,
        cu_seqlens=cu_q,
        cu_seqlens_k=cu_k,
        sliding_window_size=7,
        **kwargs,
    )
    return out, *torch.autograd.grad(out.sum(), tensors)


@pytest.mark.parametrize("share_kv", [False, True])
def test_triton_operator_contracts(share_kv, gather_attn_single_config):
    """Both opaque operators preserve fake metadata, aliasing and compiled dispatch."""
    _cuda()
    from attn_gym.sparse.gather_attn.impl.triton.ops import (
        _gather_attn_bwd_op,
        _gather_attn_fwd_op,
    )

    q, local, sparse, indices, sink, cu_q, cu_k = _inputs(20, 22, share_kv=share_kv)
    indices = indices.repeat_interleave(2, dim=-1)[..., ::2]
    args = (q, sparse, local, indices, sink, cu_q, cu_k, 7, share_kv, 0.25)
    for needs_backward in (False, True):
        torch.library.opcheck(_gather_attn_fwd_op, (*args, needs_backward))
    out, lse, queries, offsets = _gather_attn_fwd_op(*args, True)
    torch.library.opcheck(
        _gather_attn_bwd_op,
        (
            q,
            sparse,
            local,
            indices,
            queries,
            offsets,
            sink,
            cu_q,
            cu_k,
            out,
            lse,
            torch.randn_like(out),
            7,
            share_kv,
            0.25,
        ),
    )


def test_wide_addresses_keep_tma_coordinates_int32(monkeypatch, gather_attn_single_config):
    """Wide byte addressing must not widen the TMA descriptor's logical coordinates."""
    _cuda()
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("TMA requires Hopper or newer")
    from attn_gym.sparse.gather_attn.impl.triton import backward, forward

    q, local, sparse, indices, sink, cu_q, cu_k = _inputs(20, 22, dim=32, share_kv=False)
    tensors = tuple(t.requires_grad_() for t in (q, local, sparse, sink))
    inputs = (tensors, indices, cu_q, cu_k)

    expected = _forward_and_grads(*inputs, kernel_options={"backend": "triton"})
    monkeypatch.setattr(forward, "requires_int64_offsets", lambda *args: True)
    monkeypatch.setattr(backward, "requires_int64_offsets", lambda *args: True)
    actual = _forward_and_grads(*inputs, kernel_options={"backend": "triton"})
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)


def test_runtime_kv_addresses_beyond_int32(gather_attn_single_config):
    """Real wide KV batch strides must address distinct batches in forward and backward."""
    _cuda()
    if torch.cuda.mem_get_info()[0] < 24 * 2**30:
        pytest.skip("wide-address test needs two 8 GiB allocations plus headroom")
    q = torch.randn(2, 2, 19, 32, device="cuda") * 0.1
    local = torch.randn_like(q) * 0.1
    sparse = torch.randn(2, 2, 13, 32, device="cuda") * 0.1
    sink = torch.randn(2, device="cuda") * 0.1
    indices = torch.tensor([0, 1, -1], device="cuda", dtype=torch.int32).repeat(2, 19, 1)

    def widen(tensor):
        strides = (2**31 + 32, tensor.shape[2] * 32, 32, 1)
        wide = torch.empty_strided(tensor.shape, strides, device="cuda", dtype=tensor.dtype)
        return wide.copy_(tensor)

    compact = (q, local, sparse, sink)
    wide = (q.clone(), widen(local), widen(sparse), sink.clone())

    def run(tensors):
        q, local, sparse, sink = (tensor.requires_grad_() for tensor in tensors)
        out = gather_attn(
            q,
            local,
            sparse,
            indices,
            sink,
            sliding_window_size=7,
            kernel_options={"backend": "triton"},
        )
        return out, *torch.autograd.grad(out.sum(), tensors)

    expected = run(compact)
    actual = run(wide)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference)


@pytest.mark.parametrize(
    "heads,dim,inner_stride,dtype,share_kv,deterministic",
    [
        (2, 16, 2, torch.float32, False, False),
        (2, 32, 1, torch.float32, False, False),
        (16, 32, 1, torch.bfloat16, True, False),
        (16, 32, 1, torch.bfloat16, True, True),
    ],
    ids=["generic", "tma", "shared-atomic", "shared-deterministic"],
)
def test_sequence_lengths_reuse_triton_kernels(
    heads,
    dim,
    inner_stride,
    dtype,
    share_kv,
    deterministic,
    triton_compiles,
    gather_attn_single_config,
):
    """New document and sparse-KV counts, and T within a T % 16 class, reuse compiled kernels."""
    _cuda()
    from test_gather_attn_triton import assert_matches_low_precision_eager

    def attention_compiles() -> Counter[str]:
        return Counter(
            {
                name: count
                for name, count in triton_compiles.items()
                if name.startswith(("_index_query_map_keys", "_gather_attn"))
            }
        )

    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(deterministic)
    try:
        # Tokens, candidates and documents span Triton's multiple-of-16 and other classes; the
        # single-document pack comes last to isolate the kept ``== 1`` class.
        for tokens, candidates, documents in ((32, 16, 2), (40, 20, 16), (48, 36, 3), (32, 16, 1)):
            q, local, sparse, indices, sink, cu_q, cu_k = _inputs(
                tokens,
                candidates,
                heads=heads,
                dim=dim,
                dtype=dtype,
                share_kv=share_kv,
                inner_stride=inner_stride,
            )
            # Unselected slots keep the index row stride a multiple of 16 as T changes.
            indices = torch.nn.functional.pad(indices, (0, 13), value=-1)
            if documents == 1:
                multi_document = attention_compiles()
                cu_q, cu_k = cu_q[[0, -1]], cu_k[[0, -1]]
            else:
                # Leading empty documents change the document count without changing attention.
                cu_q = torch.cat((cu_q.new_zeros(documents - 2), cu_q))
                cu_k = torch.cat((cu_k.new_zeros(documents - 2), cu_k))
            tensors = tuple(t.requires_grad_() for t in (q, local, sparse, sink))
            actual = _forward_and_grads(
                tensors, indices, cu_q, cu_k, kernel_options={"backend": "triton"}
            )
            low, high = (
                _forward_and_grads(
                    tuple(cast(t.detach()).requires_grad_() for t in tensors),
                    indices,
                    cu_q,
                    cu_k,
                    impl="reference",
                )
                for cast in (torch.Tensor.clone, torch.Tensor.double)
            )
            forward_reductions = (dim, 10, 10)
            head_reduction = heads if share_kv else 1
            reductions = (
                forward_reductions,
                forward_reductions + (dim, 10),
                forward_reductions + (dim, 7, head_reduction),
                forward_reductions + (dim, tokens, head_reduction),
                forward_reductions + (dim, tokens),
            )
            for i, (value, low_ref, high_ref) in enumerate(zip(actual, low, high)):
                assert_matches_low_precision_eager(
                    value,
                    low_ref,
                    high_ref,
                    reductions[i],
                    quantized_intermediates=2 if i == 4 else 0,
                    intermediate_dtype=dtype,
                )
    finally:
        torch.use_deterministic_algorithms(previous)
    # Lse row strides are tuple elements, which Triton always specializes, so attention kernels
    # keep one binary per T % 16 class; the reverse-map kernel takes no T-sized strides.
    attention = {
        name: count for name, count in multi_document.items() if name.startswith("_gather_attn")
    }
    assert multi_document["_index_query_map_keys"] == 1
    assert attention and set(attention.values()) == {2}, multi_document
    # One document adds at most its ``== 1`` binary per kernel, which the key kernel keeps.
    single_document = attention_compiles() - multi_document
    assert single_document["_index_query_map_keys"] == 1
    assert set(single_document.values()) == {1}, single_document


def test_batch_sizes_reuse_shared_schedule(
    monkeypatch, tmp_path, triton_compiles, gather_attn_single_config
):
    """New batch sizes launch the shared-KV kernels and tuned configs of an earlier one."""
    _cuda()
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("The shared-KV schedule requires Blackwell")
    import importlib

    tune_module = importlib.import_module("attn_gym._backends.cute.tune")
    # Start without in-process or on-disk winners so every tuning decision is observed.
    monkeypatch.setenv("ATTN_GYM_CUTE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(tune_module, "_WINNERS", {})
    monkeypatch.setattr(tune_module, "_WINNERS_FAST", {})
    tuned = Counter()
    tune = tune_module.tune

    def count_tuning(configs, compile_fn, *args, **kwargs):
        tuned[compile_fn().__name__] += 1
        return tune(configs, compile_fn, *args, **kwargs)

    monkeypatch.setattr(tune_module, "tune", count_tuning)
    heads, tokens, dim = 16, 32, 32
    for batch in (1, 2, 3):
        q = torch.randn(batch, heads, tokens, dim, device="cuda", dtype=torch.bfloat16)
        local = torch.randn(batch, 1, tokens, dim, device="cuda", dtype=torch.bfloat16)
        sparse = torch.randn(batch, 1, 16, dim, device="cuda", dtype=torch.bfloat16)
        sink = torch.zeros(heads, device="cuda")
        indices = torch.tensor([0, 1, -1], device="cuda", dtype=torch.int32)
        tensors = tuple(t.requires_grad_() for t in (q, local, sparse, sink))
        out = gather_attn(
            *tensors[:3],
            indices.repeat(batch, tokens, 1),
            tensors[3],
            sliding_window_size=7,
            kernel_options={"backend": "triton"},
        )
        torch.autograd.grad(out.sum(), tensors)
    assert "_gather_attn_fwd_shared" in tuned
    assert {name: count for name, count in tuned.items() if count > 1} == {}
    assert {name: count for name, count in triton_compiles.items() if count > 1} == {}
