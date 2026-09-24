"""The FA4 adapter's KV permutation preserves values, gradients, and capacity isolation."""

import pytest
import torch

pack_kv = pytest.importorskip("attn_gym.sparse.gather_attn.impl.packed_kv").pack_kv
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference_pack(local, sparse, q_offsets, k_offsets):
    parts = []
    for qs, qe, ks, ke in zip(q_offsets, q_offsets[1:], k_offsets, k_offsets[1:]):
        parts.extend((local[0, 0, qs:qe], sparse[0, 0, ks:ke]))
    parts.extend((local[0, 0, q_offsets[-1] :], sparse[0, 0, k_offsets[-1] :]))
    return torch.cat(parts).unsqueeze(1)


@pytest.mark.parametrize(
    "tokens,candidates,q_offsets,k_offsets,dim,strided",
    [
        (7, 5, [0, 0, 2, 2, 5], [0, 0, 1, 1, 3], 16, True),
        (4, 2, [0, 1, 4], [0, 1, 2], 512, False),
        (0, 0, [0, 0, 0, 0], [0, 0, 0, 0], 8, False),
        (3, 0, [0, 0, 2], [0, 0, 0], 8, True),
        (3, 2, [0, 0, 0], [0, 0, 0], 8, False),
    ],
)
def test_packed_kv_permutation_and_inverse(tokens, candidates, q_offsets, k_offsets, dim, strided):
    torch.manual_seed(21)

    def make(length):
        if strided:
            tensor = torch.randn(1, 1, length * 2, dim * 2, device="cuda")[:, :, ::2, ::2]
        else:
            tensor = torch.randn(1, 1, length, dim, device="cuda")
        return tensor.requires_grad_()

    local, sparse = make(tokens), make(candidates)
    cu_q = torch.tensor(q_offsets, device="cuda", dtype=torch.int32)
    cu_k = torch.tensor(k_offsets, device="cuda", dtype=torch.int32)
    actual, extended_q, combined_k = pack_kv(local, sparse, cu_q, cu_k)
    expected = _reference_pack(local, sparse, q_offsets, k_offsets)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert actual.is_contiguous()
    assert extended_q.tolist() == [*q_offsets, tokens]
    assert combined_k.tolist() == [
        *(q + k for q, k in zip(q_offsets, k_offsets)),
        tokens + candidates,
    ]
    grad = torch.randn_like(actual)
    expected_grads = torch.autograd.grad(expected, (local, sparse), grad)
    actual_grads = torch.autograd.grad(actual, (local, sparse), grad)
    for result, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)


def test_packed_kv_replay_changes_document_boundaries():
    torch.manual_seed(22)
    local = torch.randn(1, 1, 7, 16, device="cuda", requires_grad=True)
    sparse = torch.randn(1, 1, 5, 16, device="cuda", requires_grad=True)
    cu_q = torch.tensor([0, 0, 2, 5], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 0, 1, 3], dtype=torch.int32, device="cuda")
    grad = torch.randn(12, 1, 16, device="cuda")

    def run():
        packed, q, k = pack_kv(local, sparse, cu_q, cu_k)
        return packed, q, k, *torch.autograd.grad(packed, (local, sparse), grad)

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    previous_override = torch._C._override_stale_capture_stream()
    torch.autograd.graph.set_override_stale_capture_stream(True)
    try:
        with torch.cuda.graph(graph):
            actual = run()
    finally:
        torch.autograd.graph.set_override_stale_capture_stream(previous_override)
    for q_offsets, k_offsets in (([0, 1, 1, 6], [0, 1, 2, 4]), ([0, 0, 0, 0], [0, 0, 0, 0])):
        cu_q.copy_(torch.tensor(q_offsets, dtype=torch.int32, device="cuda"))
        cu_k.copy_(torch.tensor(k_offsets, dtype=torch.int32, device="cuda"))
        graph.replay()
        expected = _reference_pack(local, sparse, q_offsets, k_offsets)
        expected_grads = torch.autograd.grad(expected, (local, sparse), grad)
        torch.testing.assert_close(actual[0], expected, atol=0, rtol=0)
        assert actual[1].tolist() == [*q_offsets, 7]
        assert actual[2].tolist() == [*(q + k for q, k in zip(q_offsets, k_offsets)), 12]
        for result, reference in zip(actual[3:], expected_grads):
            torch.testing.assert_close(result, reference, atol=0, rtol=0)


def test_packed_kv_active_addresses_beyond_int32():
    if torch.cuda.mem_get_info()[0] < 8 * 2**30:
        pytest.skip("wide permutation test needs 4 GiB plus headroom")
    stride = 2**31 + 1
    local = torch.empty_strided(
        (1, 1, 2, 8), (0, 0, stride, 1), dtype=torch.bfloat16, device="cuda"
    )
    values = torch.arange(16, dtype=torch.bfloat16, device="cuda").reshape(1, 1, 2, 8)
    local.copy_(values)
    local.requires_grad_()
    sparse = torch.empty(1, 1, 0, 8, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    cu_q = torch.tensor([0, 1, 2], device="cuda", dtype=torch.int32)
    cu_k = torch.zeros_like(cu_q)
    packed, _, _ = pack_kv(local, sparse, cu_q, cu_k)
    torch.testing.assert_close(packed, values.reshape(2, 1, 8), atol=0, rtol=0)
    # Reuse the same wide allocation as an independently strided incoming gradient.
    grad = local.detach().as_strided((2, 1, 8), (stride, 8, 1))
    grad_local, grad_sparse = torch.autograd.grad(packed, (local, sparse), grad)
    torch.testing.assert_close(grad_local, values, atol=0, rtol=0)
    assert grad_sparse.shape == sparse.shape


def test_packed_kv_fullgraph_forward_backward():
    local = torch.randn(1, 1, 7, 16, device="cuda", requires_grad=True)
    sparse = torch.randn(1, 1, 5, 16, device="cuda", requires_grad=True)
    q_offsets, k_offsets = [0, 0, 2, 5], [0, 0, 1, 3]
    cu_q = torch.tensor(q_offsets, device="cuda", dtype=torch.int32)
    cu_k = torch.tensor(k_offsets, device="cuda", dtype=torch.int32)
    compiled = torch.compile(pack_kv, fullgraph=True)
    actual, _, _ = compiled(local, sparse, cu_q, cu_k)
    expected = _reference_pack(local, sparse, q_offsets, k_offsets)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, (local, sparse), grad)
    expected_grads = torch.autograd.grad(expected, (local, sparse), grad)
    for result, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result, reference, atol=0, rtol=0)
