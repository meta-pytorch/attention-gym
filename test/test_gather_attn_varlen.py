"""Packed gather attention uses document-local indices and matches independent documents."""

import pytest
import torch

from attn_gym.sparse.gather_attn import AuxRequest, gather_attn


def _device(backend):
    if backend == "reference":
        return "cpu"
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if backend == "cute":
        pytest.importorskip("flash_attn.cute.interface")
        if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
            pytest.skip("FA4 gather requires SM100/SM103")
    return "cuda"


def _backend_kwargs(backend):
    return (
        {"impl": "reference"}
        if backend == "reference"
        else {"kernel_options": {"backend": backend}}
    )


def _inputs(device="cpu", dtype=torch.float64, *, share_kv=True, head_dim=16, capacity=0):
    cu_q = torch.tensor([0, 0, 3, 12, 12, 17, 17], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 0, 0, 2, 2, 3, 3], device=device, dtype=torch.int32)
    heads = 16 if device == "cuda" else 2
    kv_heads = 1 if share_kv else heads
    tensors = [
        torch.randn(1, heads, 17 + capacity, head_dim, device=device, dtype=dtype) * 0.2,
        torch.randn(1, kv_heads, 17 + capacity, head_dim, device=device, dtype=dtype) * 0.2,
        torch.randn(1, kv_heads, 3 + capacity, head_dim, device=device, dtype=dtype) * 0.2,
        torch.linspace(
            -0.7, 0.9, heads, device=device, dtype=torch.float32 if device == "cuda" else dtype
        ),
    ]
    for tensor in tensors:
        tensor.requires_grad_()
    # Include a duplicate, a sentinel, and an OOB index that overflows when a base is added.
    indices = torch.tensor([0, 0, 1, -1, 99, 2**31 - 1], device=device, dtype=torch.int32)
    indices = indices.expand(1, 17 + capacity, -1).contiguous()
    return tensors, indices, cu_q, cu_k


def _per_document(tensors, indices, cu_q, cu_k, window):
    query, local, sparse, sink = tensors
    outputs, lses = [], []
    # The test oracle knows the CPU offsets; production must not inspect device values.
    q_offsets, k_offsets = cu_q.cpu().tolist(), cu_k.cpu().tolist()
    for q_start, q_end, k_start, k_end in zip(
        q_offsets[:-1], q_offsets[1:], k_offsets[:-1], k_offsets[1:]
    ):
        if q_start == q_end:
            continue
        result, aux = gather_attn(
            query[:, :, q_start:q_end],
            local[:, :, q_start:q_end],
            sparse[:, :, k_start:k_end],
            indices[:, q_start:q_end],
            sink,
            sliding_window_size=window,
            impl="reference",
            return_aux=AuxRequest(lse=True),
        )
        outputs.append(result)
        lses.append(aux.lse)
    return torch.cat(outputs, dim=2), torch.cat(lses, dim=2)


@pytest.mark.parametrize("share_kv", [True, False])
@pytest.mark.parametrize(
    "dtype,slots,window",
    [(torch.float64, 6, 7), (torch.float32, 0, 3), (torch.float32, 1, 3), (torch.float32, 3, 3)],
)
def test_packed_reference_matches_document_outputs_and_gradients(share_kv, dtype, slots, window):
    tensors, indices, cu_q, cu_k = _inputs(dtype=dtype, share_kv=share_kv, capacity=3)
    indices = indices[..., :slots]
    tolerance = 1e-12 if dtype == torch.float64 else 1e-5
    query, local, sparse, sink = tensors
    expected, expected_lse = _per_document(tensors, indices, cu_q, cu_k, window=window)
    actual, aux = gather_attn(
        query,
        local,
        sparse,
        indices,
        sink,
        sliding_window_size=window,
        cu_seqlens=cu_q,
        cu_seqlens_k=cu_k,
        impl="reference",
        return_aux=AuxRequest(lse=True),
    )
    torch.testing.assert_close(actual[:, :, :17], expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(aux.lse[:, :, :17], expected_lse, atol=tolerance, rtol=tolerance)
    grad = torch.randn_like(expected)
    expected_grads = torch.autograd.grad(expected, tensors, grad)
    actual_grads = torch.autograd.grad(actual[:, :, :17], tensors, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA memory accounting required")
def test_shared_reference_keeps_kv_slots_shared_in_backward():
    """Reference workspace scales with KV heads, not query heads times the gathered slots."""
    heads, tokens, dim, slots, window = 128, 128, 64, 32, 16
    q = torch.randn(1, heads, tokens, dim, device="cuda", dtype=torch.float64, requires_grad=True)
    local = torch.randn(1, 1, tokens, dim, device="cuda", dtype=q.dtype, requires_grad=True)
    sparse = torch.randn_like(local, requires_grad=True)
    indices = torch.arange(slots, device="cuda", dtype=torch.int32).expand(1, tokens, -1)
    offsets = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    grad = torch.randn_like(q)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = gather_attn(
        q,
        local,
        sparse,
        indices,
        cu_seqlens=offsets,
        cu_seqlens_k=offsets,
        sliding_window_size=window,
        impl="reference",
    )
    gradients = torch.autograd.grad(out, (q, local, sparse), grad)
    peak = torch.cuda.max_memory_allocated() - before
    # Allow eight copies of input/slot/score-sized intermediates across forward/backward.
    # Head-replicated slots or gradients of a token-expanded full pool exceed this bound.
    elements = q.numel() + local.numel() + sparse.numel()
    elements += tokens * (slots + window) * (dim + heads)
    assert peak < 8 * elements * q.element_size(), f"reference workspace grew to {peak} bytes"
    assert all(torch.isfinite(tensor).all() for tensor in (out, *gradients))


@pytest.mark.parametrize("backend", ["reference", "triton", "cute"])
@pytest.mark.parametrize("with_sink", [False, True])
@pytest.mark.parametrize("slots", [0, 5])
def test_empty_sparse_pool_and_attention_have_zero_output_gradients(backend, with_sink, slots):
    device = _device(backend)
    dtype = torch.float64 if backend == "reference" else torch.bfloat16
    heads, dim = (16, 512) if backend == "cute" else (2, 16)
    q = torch.randn(1, heads, 5, dim, device=device, dtype=dtype, requires_grad=True)
    local = torch.randn(1, 1, 5, dim, device=device, dtype=dtype, requires_grad=True)
    sparse = torch.empty(1, 1, 0, dim, device=device, dtype=dtype, requires_grad=True)
    sink = torch.randn(heads, device=device, dtype=torch.float32, requires_grad=True)
    cu_q = torch.tensor([0, 0, 3, 5, 5], dtype=torch.int32, device=device)
    cu_k = torch.zeros_like(cu_q)
    indices = torch.full((1, 5, slots), -1, device=device, dtype=torch.int32)
    out, aux = gather_attn(
        q,
        local,
        sparse,
        indices,
        sink if with_sink else None,
        cu_seqlens=cu_q,
        cu_seqlens_k=cu_k,
        sliding_window_size=0,
        return_aux=AuxRequest(lse=True),
        **_backend_kwargs(backend),
    )
    assert torch.equal(out, torch.zeros_like(out))
    expected_lse = (
        sink[None, :, None].expand_as(aux.lse)
        if with_sink
        else torch.full_like(aux.lse, -torch.inf)
    )
    torch.testing.assert_close(aux.lse, expected_lse.to(aux.lse.dtype))
    inputs = [q, local, sparse] + ([sink] if with_sink else [])
    for gradient in torch.autograd.grad(out.sum(), inputs):
        assert torch.equal(gradient, torch.zeros_like(gradient))


def test_packed_reference_fullgraph_and_dynamic_offsets():
    tensors, indices, cu_q, cu_k = _inputs(capacity=3)
    q, local, sparse, sink = tensors
    compiled = torch.compile(gather_attn, backend="eager", fullgraph=True, dynamic=True)
    for offsets in ([0, 0, 3, 12, 12, 17, 17], [0, 2, 3, 10, 10, 16, 17]):
        cu_q.copy_(torch.tensor(offsets, dtype=torch.int32))
        expected, _ = _per_document(tensors, indices, cu_q, cu_k, window=7)
        actual = compiled(
            q,
            local,
            sparse,
            indices,
            sink,
            sliding_window_size=7,
            cu_seqlens=cu_q,
            cu_seqlens_k=cu_k,
            impl="reference",
        )
        torch.testing.assert_close(actual[:, :, :17], expected)


@pytest.mark.parametrize("backend", ["reference", "triton", "cute"])
@pytest.mark.parametrize("poison_start,checked_tokens", [(2, 12), (3, 17)])
def test_unused_sparse_storage_cannot_poison_another_document(
    backend, poison_start, checked_tokens
):
    device = _device(backend)
    dtype = torch.float64 if backend == "reference" else torch.bfloat16
    tensors, indices, cu_q, cu_k = _inputs(
        device, dtype, head_dim=512 if backend == "cute" else 16, capacity=3
    )
    q, local, sparse, sink = tensors
    kwargs = dict(
        sliding_window_size=7,
        cu_seqlens=cu_q,
        cu_seqlens_k=cu_k,
        **_backend_kwargs(backend),
    )
    with torch.no_grad():
        expected = gather_attn(q, local, sparse, indices, sink, **kwargs)
        # Check another document's pool and inactive capacity separately. Local OOB index 1
        # in the last document would address the poisoned capacity if only global bounds applied.
        sparse[:, :, poison_start:] = torch.nan
        actual = gather_attn(q, local, sparse, indices, sink, **kwargs)
    torch.testing.assert_close(actual[:, :, :checked_tokens], expected[:, :, :checked_tokens])


@pytest.mark.parametrize("backend,share_kv", [("triton", True), ("triton", False), ("cute", True)])
def test_packed_gpu_matches_per_document_forward_backward(
    backend, share_kv, gather_attn_single_config
):
    _device(backend)
    from test_gather_attn_triton import assert_matches_low_precision_eager

    dim = 512 if backend == "cute" else 64
    tensors, indices, cu_q, cu_k = _inputs("cuda", torch.bfloat16, share_kv=share_kv, head_dim=dim)
    high_inputs = [tensor.detach().double().requires_grad_() for tensor in tensors]
    low_inputs = [tensor.detach().clone().requires_grad_() for tensor in tensors]
    high, high_lse = _per_document(high_inputs, indices, cu_q, cu_k, window=7)
    low, _ = _per_document(low_inputs, indices, cu_q, cu_k, window=7)
    q, local, sparse, sink = tensors
    actual, aux = gather_attn(
        q,
        local,
        sparse,
        indices,
        sink,
        sliding_window_size=7,
        cu_seqlens=cu_q,
        cu_seqlens_k=cu_k,
        kernel_options={"backend": backend},
        return_aux=AuxRequest(lse=True),
    )
    forward_reductions = (dim, indices.shape[-1] + 7, indices.shape[-1] + 7)
    assert_matches_low_precision_eager(actual, low, high, forward_reductions)
    # LSE is FP32: allow dot-product accumulation and exp/log rounding, not BF16 output error.
    fp32_eps = torch.finfo(torch.float32).eps
    torch.testing.assert_close(
        aux.lse.double(), high_lse, atol=8 * dim * fp32_eps, rtol=8 * fp32_eps
    )
    grad = torch.randn_like(actual) * 0.2
    high_grads = torch.autograd.grad(high, high_inputs, grad.double())
    low_grads = torch.autograd.grad(low, low_inputs, grad)
    actual_grads = torch.autograd.grad(actual, tensors, grad)
    # Use the same per-input reduction model as the D512 training tests. Only dSink needs
    # an extra allowance for the low-precision forward output and upstream gradient product.
    head_reduction = q.shape[1] if share_kv else 1
    reductions = (
        forward_reductions + (dim, indices.shape[-1] + 7),
        forward_reductions + (dim, 7, head_reduction),
        forward_reductions + (dim, q.shape[2], head_reduction),
        forward_reductions + (dim, q.shape[2]),
    )
    for i, (actual_grad, low_grad, high_grad) in enumerate(
        zip(actual_grads, low_grads, high_grads)
    ):
        assert_matches_low_precision_eager(
            actual_grad,
            low_grad,
            high_grad,
            reductions[i],
            quantized_intermediates=2 if i == 3 else 0,
            intermediate_dtype=q.dtype,
        )


def test_packed_triton_fullgraph_and_graph_replay(gather_attn_single_config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    tensors, indices, cu_q, cu_k = _inputs("cuda", torch.bfloat16, capacity=3)
    tensors = [tensor.detach() for tensor in tensors]
    q, local, sparse, sink = tensors

    def run():
        return gather_attn(
            q,
            local,
            sparse,
            indices,
            sink,
            sliding_window_size=7,
            cu_seqlens=cu_q,
            cu_seqlens_k=cu_k,
            kernel_options={"backend": "triton"},
        )

    compiled = torch.compile(run, fullgraph=True)
    torch.testing.assert_close(compiled(), run())
    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    cu_q.copy_(torch.tensor([0, 2, 3, 10, 10, 16, 17], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.testing.assert_close(actual, run())


def test_packed_triton_fullgraph_backward(gather_attn_single_config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    tensors, indices, cu_q, cu_k = _inputs("cuda", torch.bfloat16, capacity=3)
    q, local, sparse, sink = tensors
    compiled = torch.compile(gather_attn, fullgraph=True)
    kwargs = {
        "sliding_window_size": 7,
        "cu_seqlens": cu_q,
        "cu_seqlens_k": cu_k,
        "kernel_options": {"backend": "triton"},
    }
    expected = gather_attn(q, local, sparse, indices, sink, **kwargs)
    actual = compiled(q, local, sparse, indices, sink, **kwargs)
    torch.testing.assert_close(actual, expected)
    grad = torch.randn_like(actual[:, :, :17])
    expected_grads = torch.autograd.grad(expected[:, :, :17], tensors, grad)
    actual_grads = torch.autograd.grad(actual[:, :, :17], tensors, grad)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_packed_backward_rejects_changed_query_boundaries(gather_attn_single_config):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    tensors, indices, cu_q, cu_k = _inputs("cuda", torch.bfloat16, capacity=3)
    q, local, sparse, sink = tensors
    output = gather_attn(
        q,
        local,
        sparse,
        indices,
        sink,
        sliding_window_size=7,
        cu_seqlens=cu_q,
        cu_seqlens_k=cu_k,
        kernel_options={"backend": "triton"},
    )
    cu_q[2] = 2
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()
