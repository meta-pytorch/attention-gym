"""Exact packed candidate intervals, independent of scoring kernels."""

import pytest
import torch


def _cuda():
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("NVIDIA CUDA required")
    return "cuda"


@pytest.mark.parametrize("causal,ratio", [(False, 1), (True, 1), (True, 4)])
def test_candidate_bounds_match_document_intervals(causal, ratio):
    device = _cuda()
    from attn_gym.sparse.indexer.impl.triton import prepare_candidate_bounds

    # Exercise different binary-search depths, repeated offsets, partial vector tiles,
    # and inactive query/candidate capacity without deriving expected IDs via search.
    for lengths in ([0], [319], [0, 0], [0, 3, 0, 9, 257, 0, 7], [i % 7 for i in range(33)]):
        q_offsets, k_offsets = [0], [0]
        for length in lengths:
            q_offsets.append(q_offsets[-1] + length)
            k_offsets.append(k_offsets[-1] + length // ratio)
        tokens = q_offsets[-1] + 3
        cu_q = torch.tensor(q_offsets, device=device, dtype=torch.int32)
        cu_k = torch.tensor(k_offsets, device=device, dtype=torch.int32)
        expected = torch.full((tokens, 2), k_offsets[-1], dtype=torch.int32)
        for qs, qe, ks, ke in zip(q_offsets, q_offsets[1:], k_offsets, k_offsets[1:]):
            for query in range(qs, qe):
                expected[query] = torch.tensor(
                    [ks, min(ke, ks + (query - qs + 1) // ratio) if causal else ke]
                )
        actual = prepare_candidate_bounds(cu_q, cu_k, tokens, causal, ratio)
        assert actual.is_contiguous()
        torch.testing.assert_close(actual.cpu(), expected)


def test_candidate_bounds_cuda_graph_replays_changed_offsets():
    device = _cuda()
    from attn_gym.sparse.indexer.impl.triton import prepare_candidate_bounds

    cu_q = torch.tensor([0, 3, 12, 17], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 0, 2, 3], device=device, dtype=torch.int32)
    for _ in range(3):
        prepare_candidate_bounds(cu_q, cu_k, 19, True, 4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = prepare_candidate_bounds(cu_q, cu_k, 19, True, 4)
    cu_q.copy_(torch.tensor([0, 0, 7, 16], device=device, dtype=torch.int32))
    cu_k.copy_(torch.tensor([0, 0, 1, 3], device=device, dtype=torch.int32))
    graph.replay()
    expected = torch.tensor(
        [[0, 0]] * 3 + [[0, 1]] * 4 + [[1, 1]] * 3 + [[1, 2]] * 4 + [[1, 3]] * 2 + [[3, 3]] * 3,
        device=device,
        dtype=torch.int32,
    )
    torch.testing.assert_close(actual, expected)


def test_candidate_bounds_forced_int64_offsets(monkeypatch):
    device = _cuda()
    from attn_gym.sparse.indexer.impl import triton as indexer_triton

    singleton = torch.empty_strided((1, 19, 7), (2**40, 7, 1), device="meta")
    wide = torch.empty_strided((1, 2, 7), (0, 2**31, 1), device="meta")
    assert not indexer_triton.requires_int64_offsets(singleton)
    assert indexer_triton.requires_int64_offsets(wide)
    cu_q = torch.tensor([0, 3, 12, 17], device=device, dtype=torch.int32)
    cu_k = torch.tensor([0, 0, 2, 3], device=device, dtype=torch.int32)
    expected = indexer_triton.prepare_candidate_bounds(cu_q, cu_k, 19, True, 4)
    monkeypatch.setattr(indexer_triton, "requires_int64_offsets", lambda *tensors: True)
    actual = indexer_triton.prepare_candidate_bounds(cu_q, cu_k, 19, True, 4)
    torch.testing.assert_close(actual, expected)
