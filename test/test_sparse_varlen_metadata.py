"""Integer contracts for packed candidate bounds and native FA4 index construction."""

import pytest
import torch


def _cuda():
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("NVIDIA CUDA required")
    return "cuda"


def _indices(batch, tokens, slots, dtype, layout="contiguous", device="cuda"):
    values = torch.tensor([-1, 0, 1, 2, 7, torch.iinfo(dtype).max], dtype=dtype, device=device)
    indices = values.repeat((slots + 5) // 6)[:slots].expand(batch, tokens, slots).clone()
    if layout == "transposed":
        indices = indices.transpose(1, 2).contiguous().transpose(1, 2)
    elif layout == "strided":
        storage = torch.empty(batch * 2, tokens * 2, slots * 2, dtype=dtype, device=device)
        storage[::2, ::2, ::2] = indices
        indices = storage[::2, ::2, ::2]
    return indices


@pytest.mark.parametrize(
    "dtype,batch,slots,candidates,q_offsets,k_offsets,layout,window",
    [
        (torch.int64, 2, 257, 9, None, None, "strided", 31),
        (torch.int32, 1, 7, 9, None, None, "transposed", 31),
        (torch.int32, 1, 7, 0, None, None, "contiguous", 31),
        (torch.int32, 2, 0, 9, None, None, "contiguous", 31),
        (torch.int64, 2, 0, 9, None, None, "strided", 0),
        (torch.int64, 1, 257, 9, [0, 0, 3, 12, 12, 17, 17], [0, 0, 0, 2, 2, 3, 3], "strided", 31),
        (torch.int32, 1, 7, 9, [0, 0, 3, 12, 12, 17, 17], [0, 0, 0, 2, 2, 3, 3], "transposed", 1),
        (torch.int32, 1, 0, 9, [0, 0, 3, 12, 12, 17, 17], [0, 0, 0, 2, 2, 3, 3], "contiguous", 31),
        (torch.int32, 1, 7, 0, [0, 0, 0], [0, 0, 0], "contiguous", 31),
        (torch.int32, 1, 0, 0, [0, 0, 0], [0, 0, 0], "strided", 128),
        (torch.int64, 1, 7, 9, [0, 17], [0, 3], "contiguous", 0),
    ],
    ids=[
        "dense-strided-multitile",
        "dense-transposed",
        "dense-empty-pool",
        "dense-local-only",
        "dense-empty-attention",
        "packed-strided-multitile",
        "packed-transposed-window-one",
        "packed-local-only",
        "packed-empty-docs",
        "packed-empty-docs-wide-window",
        "packed-single-doc-sparse-only",
    ],
)
def test_fa4_indices_match_local_index_contract(
    dtype, batch, slots, candidates, q_offsets, k_offsets, layout, window
):
    tokens = 19
    indices = _indices(batch, tokens, slots, dtype, layout, _cuda())
    cu_q = None if q_offsets is None else torch.tensor(q_offsets, device="cuda", dtype=torch.int32)
    cu_k = None if k_offsets is None else torch.tensor(k_offsets, device="cuda", dtype=torch.int32)
    from attn_gym.sparse.gather_attn.impl.indices import build_gather_indices

    actual = build_gather_indices(indices, cu_q, cu_k, window, tokens, candidates)
    expected = _fa4_indices_oracle(indices, candidates, q_offsets, k_offsets, window)
    assert actual.is_contiguous()
    torch.testing.assert_close(actual.cpu(), expected)


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


def test_fa4_index_builder_fullgraph_dynamic():
    from attn_gym.sparse.gather_attn.impl.indices import build_gather_indices

    device = _cuda()
    compiled = torch.compile(build_gather_indices, fullgraph=True, dynamic=True)
    for tokens, candidates, offsets in ((19, 7, [0, 3, 17]), (23, 9, [0, 0, 17])):
        indices = _indices(1, tokens, 7, torch.int64, "strided", device)
        cu_q = torch.tensor(offsets, device=device, dtype=torch.int32)
        cu_k = torch.tensor([0, 0, 3], device=device, dtype=torch.int32)
        actual = compiled(indices, cu_q, cu_k, 31, tokens, candidates)
        expected = _fa4_indices_oracle(indices, candidates, offsets, [0, 0, 3], 31)
        torch.testing.assert_close(actual.cpu(), expected)


def test_preparation_cuda_graph_replays_changed_offsets_and_indices():
    device = _cuda()
    from attn_gym.sparse.gather_attn.impl.indices import build_gather_indices
    from attn_gym.sparse.indexer.impl.triton import prepare_candidate_bounds

    indices = _indices(1, 19, 7, torch.int64, "strided", device)
    cu_q = torch.tensor([0, 3, 12, 17], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 0, 2, 3], device="cuda", dtype=torch.int32)

    def run():
        return (
            build_gather_indices(indices, cu_q, cu_k, 31, 19, 9),
            prepare_candidate_bounds(cu_q, cu_k, 19, True, 4),
        )

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    cu_q.copy_(torch.tensor([0, 0, 7, 16], device="cuda", dtype=torch.int32))
    cu_k.copy_(torch.tensor([0, 0, 1, 3], device="cuda", dtype=torch.int32))
    indices[..., 0] = 1
    graph.replay()
    expected = _fa4_indices_oracle(indices, 9, [0, 0, 7, 16], [0, 0, 1, 3], 31)
    torch.testing.assert_close(actual[0].cpu(), expected)
    expected = torch.tensor(
        [[0, 0]] * 3 + [[0, 1]] * 4 + [[1, 1]] * 3 + [[1, 2]] * 4 + [[1, 3]] * 2 + [[3, 3]] * 3,
        device=device,
        dtype=torch.int32,
    )
    torch.testing.assert_close(actual[1], expected)


def test_preparation_forced_int64_offsets(monkeypatch):
    device = _cuda()
    from attn_gym.sparse.gather_attn.impl import indices as gather_indices
    from attn_gym.sparse.indexer.impl import triton as indexer_triton

    singleton = torch.empty_strided((1, 19, 7), (2**40, 7, 1), device="meta")
    wide = torch.empty_strided((1, 2, 7), (0, 2**31, 1), device="meta")
    assert not indexer_triton.requires_int64_offsets(singleton)
    assert indexer_triton.requires_int64_offsets(wide)
    indices = _indices(1, 19, 257, torch.int64, "strided", device)
    cu_q = torch.tensor([0, 3, 12, 17], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 0, 2, 3], device="cuda", dtype=torch.int32)
    assert not indexer_triton.requires_int64_offsets(indices, cu_q, cu_k)
    expected_bounds = indexer_triton.prepare_candidate_bounds(cu_q, cu_k, 19, True, 4)
    expected_fa4 = gather_indices.build_gather_indices(indices, cu_q, cu_k, 31, 19, 9)
    monkeypatch.setattr(indexer_triton, "requires_int64_offsets", lambda *tensors: True)
    monkeypatch.setattr(gather_indices, "requires_int64_offsets", lambda *tensors: True)
    actual_bounds = indexer_triton.prepare_candidate_bounds(cu_q, cu_k, 19, True, 4)
    torch.testing.assert_close(actual_bounds, expected_bounds)
    actual_fa4 = gather_indices.build_gather_indices(indices, cu_q, cu_k, 31, 19, 9)
    torch.testing.assert_close(actual_fa4, expected_fa4)


def test_fa4_indices_active_offset_beyond_int32():
    from attn_gym.sparse.gather_attn.impl.indices import build_gather_indices

    device = _cuda()
    if torch.cuda.mem_get_info()[0] < 12 * 2**30:
        pytest.skip("wide active-offset test needs an 8 GiB strided allocation plus headroom")
    indices = torch.empty_strided((1, 2, 3), (0, 2**31 + 1, 1), dtype=torch.int32, device=device)
    compact = torch.tensor([[[0, -1, 1], [2, 0, 2**31 - 1]]], device=device, dtype=torch.int32)
    indices.copy_(compact)
    cu = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    actual = build_gather_indices(indices, cu, cu, 1, 2, 2)
    expected = build_gather_indices(compact, cu, cu, 1, 2, 2)
    torch.testing.assert_close(actual, expected)


def _fa4_indices_oracle(indices, candidates, q_offsets, k_offsets, window):
    indices = indices.cpu()
    batch, tokens, topk = indices.shape
    slots = max(128, ((window + topk + 127) // 128) * 128)
    expected = torch.full((batch, tokens, slots), -1, dtype=torch.int32)
    if q_offsets is None:
        q_offsets, k_offsets = [0, tokens], [0, candidates]
    else:
        # The inactive query tail has its own local window but no sparse candidates.
        q_offsets, k_offsets = [*q_offsets, tokens], [*k_offsets, k_offsets[-1]]
    for qs, qe, ks, ke in zip(q_offsets, q_offsets[1:], k_offsets, k_offsets[1:]):
        for row in range(qs, qe):
            for slot in range(window):
                key = row - window + 1 + slot
                if key >= qs:
                    expected[:, row, slot] = key
        local = indices[:, qs:qe]
        valid = (local >= 0) & (local < ke - ks)
        safe = torch.where(valid, local, 0) + ks + tokens
        expected[:, qs:qe, window : window + topk] = torch.where(valid, safe, -1)
    return expected
