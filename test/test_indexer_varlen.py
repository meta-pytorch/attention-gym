"""Packed selection uses document-local coordinates before ranking candidates."""

import pytest
import torch

from attn_gym.sparse.gather_attn import gather_attn
from attn_gym.sparse.indexer import lightning_indexer
from attn_gym.sparse.indexer.ops import _indexer_op


def _device(backend):
    if backend == "reference":
        return "cpu"
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("NVIDIA CUDA required")
    capability = torch.cuda.get_device_capability()
    if backend == "cute" and capability not in ((10, 0), (10, 3)):
        pytest.skip("CuTe indexer requires SM100/SM103")
    if backend == "triton" and capability[0] < 9:
        pytest.skip("Triton indexer requires Hopper or newer")
    return "cuda"


def _inputs(lengths, ratio, device, *, heads=2, dim=16, capacity=0):
    q_offsets = [0]
    k_offsets = [0]
    for length in lengths:
        q_offsets.append(q_offsets[-1] + length)
        k_offsets.append(k_offsets[-1] + length // ratio)
    tokens, candidates = q_offsets[-1] + capacity, k_offsets[-1]
    # FP16 keeps adjacent integer candidates distinct in the larger slab-boundary case.
    dtype = (
        torch.float64
        if device == "cpu"
        else (torch.float16 if candidates > 256 else torch.bfloat16)
    )
    q = torch.rand(1, tokens, heads, dim, device=device, dtype=dtype) + 0.5
    # Every candidate has a distinct score. Earlier documents' candidates outrank later
    # ones for positive weights, so filtering after global top-k cannot pass this oracle.
    k = torch.arange(candidates, 0, -1, device=device, dtype=dtype)
    k = k.view(1, candidates, 1).expand(1, candidates, dim).contiguous()
    weights = torch.ones(1, tokens, heads, device=device, dtype=dtype)
    weights[:, 1::2] = -1
    cu_q = torch.tensor(q_offsets, dtype=torch.int32, device=device)
    cu_k = torch.tensor(k_offsets, dtype=torch.int32, device=device)
    return q, k, weights, cu_q, cu_k


def _call(backend, inputs, topk, ratio, causal=True):
    q, k, weights, cu_q, cu_k = inputs
    return lightning_indexer(
        q,
        k,
        weights,
        topk,
        causal=causal,
        compress_ratio=ratio,
        cu_seqlens=cu_q,
        cu_seqlens_k=cu_k,
        impl="reference" if backend == "reference" else "fused",
        kernel_options=None if backend == "reference" else {"backend": backend},
    )


def _per_document(inputs, topk, ratio, causal):
    q, k, weights, cu_q, cu_k = inputs
    expected = torch.full((1, q.shape[1], topk), -1, dtype=torch.int32, device=q.device)
    q_offsets, k_offsets = cu_q.tolist(), cu_k.tolist()
    for qs, qe, ks, ke in zip(q_offsets, q_offsets[1:], k_offsets, k_offsets[1:]):
        if qs == qe:
            continue
        expected[:, qs:qe] = lightning_indexer(
            q[:, qs:qe],
            k[:, ks:ke],
            weights[:, qs:qe],
            topk,
            causal=causal,
            compress_ratio=ratio,
            impl="reference",
        )
    return expected


@pytest.mark.parametrize("backend", ["reference", "triton", "cute"])
@pytest.mark.parametrize(
    "lengths,ratio,topk,causal,heads,dim",
    [
        ([0, 3, 9, 0, 7, 6, 0], 4, 3, True, 2, 16),
        ([0, 3, 9, 0, 7, 6, 0], 4, 3, True, 32, 128),
        ([1, 4, 0, 7], 1, 3, False, 2, 16),
        ([0, 1, 3, 0], 4, 5, True, 2, 16),
        ([0, 0], 4, 3, True, 2, 16),
        ([3, 9], 4, 0, True, 2, 16),
        ([3, 9], 4, 7, True, 2, 16),
        ([7, 1027, 9], 4, 3, True, 2, 16),
    ],
    ids=[
        "ragged-generic",
        "ragged-paired",
        "noncausal",
        "empty-pool",
        "empty-docs",
        "zero-topk",
        "topk-exceeds-pool",
        "score-slab-boundary",
    ],
)
def test_packed_selection_matches_document_loop(backend, lengths, ratio, topk, causal, heads, dim):
    inputs = _inputs(lengths, ratio, _device(backend), heads=heads, dim=dim, capacity=3)
    actual = _call(backend, inputs, topk, ratio, causal)
    expected = _per_document(inputs, topk, ratio, causal)
    assert actual.dtype == torch.int32 and actual.is_contiguous()
    torch.testing.assert_close(actual.sort().values, expected.sort().values)
    # Padding follows valid selections, regardless of backend-specific winner ordering.
    counts = (expected >= 0).sum(-1, keepdim=True)
    padding = torch.arange(topk, device=actual.device) >= counts
    assert torch.all(actual[padding] == -1)


@pytest.mark.parametrize("backend", ["reference", "triton", "cute"])
def test_dense_empty_pool_and_fixed_topk(backend):
    device = _device(backend)
    q, k, weights, _, _ = _inputs([3], 4, device)
    out = lightning_indexer(
        q,
        k,
        weights,
        5,
        causal=True,
        compress_ratio=4,
        impl="reference" if backend == "reference" else "fused",
        kernel_options=None if backend == "reference" else {"backend": backend},
    )
    assert out.shape == (1, 3, 5)
    assert torch.all(out == -1)


@pytest.mark.parametrize("backend", ["reference", "triton", "cute"])
def test_unused_candidate_capacity_does_not_affect_selection(backend):
    inputs = list(_inputs([3, 9, 7], 4, _device(backend), capacity=3))
    k = inputs[1]
    inputs[1] = torch.cat((k, torch.full_like(k[:, :2], torch.nan)), dim=1)
    actual = _call(backend, inputs, 3, 4)
    expected = _per_document(inputs, 3, 4, True)
    torch.testing.assert_close(actual.sort().values, expected.sort().values)


def test_cute_rejects_candidate_capacity_beyond_workspace_budget():
    inputs = list(_inputs([4], 4, _device("cute")))
    # Broadcast storage keeps this a cheap metadata test; two FP32 score rows would
    # exceed the documented 32 MiB workspace even though only one candidate is active.
    inputs[1] = inputs[1].expand(1, 2**22 + 1, 16)
    with pytest.raises(ValueError, match="candidate capacity.*32 MiB"):
        _call("cute", inputs, 1, 4)


@pytest.mark.parametrize("operation", ["indexer", "gather"])
@pytest.mark.parametrize(
    "case,message",
    [
        ("missing-q", "supplied together"),
        ("missing-k", "supplied together"),
        ("batch", "batch size one"),
        ("dtype", "contiguous int32"),
        ("strided", "contiguous int32"),
        ("rank", "shape"),
        ("count", "same number"),
        ("device", "contiguous int32"),
        ("not-tensor", "torch.Tensor"),
    ],
)
def test_packed_metadata_validation(operation, case, message):
    q, k, weights, cu_q, cu_k = _inputs([3, 9], 4, "cpu")
    match case:
        case "missing-q":
            cu_q = None
        case "missing-k":
            cu_k = None
        case "batch":
            q, k, weights = (x.expand(2, *x.shape[1:]) for x in (q, k, weights))
        case "dtype":
            cu_k = cu_k.long()
        case "strided":
            cu_q = torch.tensor([0, 0, 3, 0, 12, 0], dtype=torch.int32)[::2]
        case "rank":
            cu_q = cu_q.unsqueeze(0)
        case "count":
            cu_k = cu_k[:2]
        case "device":
            cu_k = cu_k.to("meta")
        case "not-tensor":
            cu_k = [0, 0, 2]
    with pytest.raises((TypeError, ValueError), match=message):
        if operation == "indexer":
            lightning_indexer(
                q,
                k,
                weights,
                2,
                causal=True,
                compress_ratio=4,
                cu_seqlens=cu_q,
                cu_seqlens_k=cu_k,
                impl="reference",
            )
        else:
            query = q.transpose(1, 2)
            gather_attn(
                query,
                query[:, :1],
                k.unsqueeze(1),
                torch.zeros(q.shape[0], q.shape[1], 2, dtype=torch.int32),
                sliding_window_size=3,
                cu_seqlens=cu_q,
                cu_seqlens_k=cu_k,
                impl="reference",
            )


@pytest.mark.parametrize("backend", ["reference", "triton", "cute"])
def test_packed_fullgraph_dynamic_metadata(backend):
    device = _device(backend)
    compiled = torch.compile(
        _call,
        fullgraph=True,
        dynamic=True,
        backend="eager" if backend == "reference" else "inductor",
    )
    for lengths in ([3, 9, 7], [7, 5, 7], [3, 17, 7]):
        inputs = _inputs(lengths, 4, device, capacity=3)
        actual = compiled(backend, inputs, 3, 4)
        expected = _per_document(inputs, 3, 4, True)
        torch.testing.assert_close(actual.sort().values, expected.sort().values)


@pytest.mark.parametrize("backend", ["triton", "cute"])
@pytest.mark.parametrize(
    "lengths,topk,heads,dim,replays",
    [
        (
            [3, 9, 7],
            3,
            32,
            128,
            (([0, 7, 12, 19], [0, 1, 2, 3]), ([0, 0, 11, 16], [0, 0, 2, 3])),
        ),
        (
            [257, 511, 1027],
            8,
            2,
            16,
            (([0, 511, 768, 1795], [0, 127, 191, 447]), ([0, 0, 1023, 1792], [0, 0, 255, 447])),
        ),
        (
            [257, 511, 1027],
            8,
            32,
            128,
            (([0, 511, 768, 1795], [0, 127, 191, 447]), ([0, 0, 1023, 1792], [0, 0, 255, 447])),
        ),
    ],
    ids=["short-rows", "selective-generic", "selective-paired"],
)
def test_packed_cuda_graph_replays_changed_offsets(backend, lengths, topk, heads, dim, replays):
    # Selective rows must read freshly scored intervals, not only take the all-candidates shortcut.
    inputs = _inputs(lengths, 4, _device(backend), heads=heads, dim=dim, capacity=3)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _call(backend, inputs, topk, 4)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = _call(backend, inputs, topk, 4)
    q, _, _, cu_q, cu_k = inputs
    for offsets_q, offsets_k in replays:
        cu_q.copy_(torch.tensor(offsets_q, dtype=torch.int32, device=q.device))
        cu_k.copy_(torch.tensor(offsets_k, dtype=torch.int32, device=q.device))
        graph.replay()
        expected = _per_document(inputs, topk, 4, True)
        torch.testing.assert_close(actual.sort().values, expected.sort().values)


@pytest.mark.parametrize("backend", ["triton", "cute"])
def test_packed_opcheck(backend):
    q, k, weights, cu_q, cu_k = _inputs([4, 8], 4, _device(backend))
    # Selecting all candidates gives a stable order even in nondeterministic CuTe mode.
    torch.library.opcheck(_indexer_op, (q, k, weights, 3, True, 4, backend, cu_q, cu_k))


@pytest.mark.parametrize("backend", ["triton", "cute"])
def test_packed_int64_addressing(backend, monkeypatch):
    inputs = _inputs([3, 17, 9], 4, _device(backend), capacity=3)
    expected = _call(backend, inputs, 3, 4)
    if backend == "cute":
        from attn_gym.sparse.indexer.impl import cute

        monkeypatch.setattr(cute, "requires_int64_abi", lambda *args: True)
    else:
        from attn_gym.sparse.indexer.impl import triton

        monkeypatch.setattr(triton, "requires_int64_offsets", lambda *args: True)
    actual = _call(backend, inputs, 3, 4)
    torch.testing.assert_close(actual.sort().values, expected.sort().values)


@pytest.mark.parametrize("backend", ["reference", "triton", "cute"])
def test_indexer_indices_compose_with_packed_gather(backend):
    inputs = _inputs([3, 9, 7], 4, _device(backend))
    index_q, index_k, _, cu_q, cu_k = inputs
    indices = _call(backend, inputs, 3, 4)
    q = torch.randn(1, 2, index_q.shape[1], 8, device=index_q.device, dtype=torch.float64)
    local_kv = torch.randn_like(q[:, :1])
    sparse_kv = torch.randn(1, 1, index_k.shape[1], 8, device=q.device, dtype=q.dtype)
    actual = gather_attn(
        q,
        local_kv,
        sparse_kv,
        indices,
        sliding_window_size=3,
        cu_seqlens=cu_q,
        cu_seqlens_k=cu_k,
        impl="reference",
    )
    expected = torch.empty_like(actual)
    local_indices = _per_document(inputs, 3, 4, True)
    q_offsets, k_offsets = cu_q.tolist(), cu_k.tolist()
    for qs, qe, ks, ke in zip(q_offsets, q_offsets[1:], k_offsets, k_offsets[1:]):
        expected[:, :, qs:qe] = gather_attn(
            q[:, :, qs:qe],
            local_kv[:, :, qs:qe],
            sparse_kv[:, :, ks:ke],
            local_indices[:, qs:qe],
            sliding_window_size=3,
            impl="reference",
        )
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
