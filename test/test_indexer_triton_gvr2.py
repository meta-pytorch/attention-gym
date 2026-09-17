"""Exact self-sampling/radix refinement and bounded scorer integration in Triton."""

import pytest
import torch

from attn_gym.sparse.indexer import lightning_indexer
from attn_gym.testing.indexer import (
    assert_indexer_selection,
    assert_indexer_topk_values,
    make_indexer_test_inputs,
)


@pytest.fixture(autouse=True)
def require_hopper():
    if (
        torch.version.hip
        or not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] < 9
    ):
        pytest.skip("Hopper or newer required")
    pytest.importorskip("triton")


@pytest.mark.parametrize(
    "pattern",
    [
        "random",
        "equal",
        "signed_zero",
        "clustered",
        "extreme",
        "sample_low",
        "sample_high",
        "subnormal",
    ],
)
@pytest.mark.parametrize("topk", [1, 513, 2047])
def test_selector_exact_adversarial(pattern, topk):
    from attn_gym.sparse.indexer.impl.triton_gvr2 import launch_topk

    torch.manual_seed(77)
    scores = torch.randn(2, 2, 2049, device="cuda")
    sample = torch.arange(256, device="cuda") * 2049 // 256
    match pattern:
        case "equal":
            scores.fill_(-3)
        case "signed_zero":
            scores.zero_()
            scores[..., ::2] = -0.0
        case "clustered":
            scores.round_()
        case "extreme":
            scores = scores.sign() * torch.finfo(torch.float32).max
        case "sample_low":
            scores[..., sample] = -100
        case "sample_high":
            scores[..., sample] = 100
        case "subnormal":
            # Distinct finite subnormal bit patterns must not flush to zero in comparisons.
            bits = torch.arange(1, 2050, device="cuda", dtype=torch.int32)
            scores.copy_(bits.view(torch.float32))
            scores[..., ::2].neg_()
    output = torch.empty(1, 4, topk, device="cuda", dtype=torch.int32)
    launch_topk(scores, output, 0, False, 1, False)
    assert_indexer_topk_values(output[0], scores.view(4, -1))


@pytest.mark.parametrize("causal,ratio", [(False, 1), (True, 4)])
def test_selector_slab_tail_padding_int64(causal, ratio):
    from attn_gym.sparse.indexer.impl.triton_gvr2 import launch_topk

    tokens, candidates, topk, start, pairs = 9, 9 // ratio, 2, 3, 7
    # Slab spans the end of batch 0 and all of batch 1, including odd-T invalid rows.
    scores = torch.randn(pairs, 2, candidates, device="cuda")
    output = torch.full((2, tokens, topk), -99, device="cuda", dtype=torch.int32)
    wide = torch.empty_like(output).fill_(-99)
    for destination, use_wide in ((output, False), (wide, True)):
        launch_topk(scores, destination, start, causal, ratio, use_wide)
    torch.testing.assert_close(output, wide, rtol=0, atol=0)
    for local in range(pairs * 2):
        pair, half = divmod(local, 2)
        batch, query_pair = divmod(start + pair, (tokens + 1) // 2)
        query = query_pair * 2 + half
        if query >= tokens:
            continue
        visible = (query + 1) // ratio if causal else candidates
        selected = output[batch, query]
        valid = selected >= 0
        assert valid.sum() == min(topk, visible)
        assert (selected[~valid] == -1).all()
        if valid.any():
            actual = scores.view(-1, candidates)[local, selected[valid].long()].sort().values
            expected = scores.view(-1, candidates)[local, :visible].topk(min(topk, visible)).values
            assert torch.equal(actual, expected.sort().values)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("tokens,ratio,causal,topk", [(137, 4, True, 17), (65, 1, False, 65)])
def test_gvr2_complete_indexer(dtype, tokens, ratio, causal, topk):
    from attn_gym.sparse.indexer.impl.triton_gvr2 import launch

    q, k, w = make_indexer_test_inputs(tokens, 5, 48, dtype, compress_ratio=ratio)
    actual = launch(q, k, w, topk, causal, ratio)
    assert_indexer_selection(actual, q, k, w, topk, causal, ratio)


@pytest.mark.parametrize(
    "causal,ratio,topk,dtype", [(False, 1, 9, torch.float16), (True, 4, 7, torch.bfloat16)]
)
def test_gvr2_all_fit_skips_scoring(causal, ratio, topk, dtype, triton_compiles):
    tokens = 9
    inputs = make_indexer_test_inputs(tokens, 5, 48, dtype, compress_ratio=ratio)
    q, k, w = (x.transpose(1, -1).contiguous().transpose(1, -1) for x in inputs)
    actual = lightning_indexer(
        q,
        k,
        w,
        topk,
        causal=causal,
        compress_ratio=ratio,
        kernel_options={"backend": "triton", "selector": "gvr2"},
    )
    ranks = torch.arange(topk, dtype=torch.int32, device=q.device).expand_as(actual)
    visible = (
        ((torch.arange(tokens, device=q.device) + 1) // ratio)[:, None] if causal else k.shape[1]
    )
    assert torch.equal(actual, torch.where(ranks < visible, ranks, -1))
    assert triton_compiles["_score_kernel"] == 0
    assert triton_compiles["_gvr2_topk_kernel"] == 0


def test_gvr2_strided_graph_replay_and_wide(monkeypatch):
    from attn_gym.sparse.indexer.impl import triton_gvr2 as impl

    q, k, w = make_indexer_test_inputs(133, 5, 48, torch.bfloat16, compress_ratio=4)
    q = q.transpose(-1, -2).contiguous().transpose(-1, -2)
    k = k.transpose(-1, -2).contiguous().transpose(-1, -2)
    w = w.transpose(-1, -2).contiguous().transpose(-1, -2)
    expected = impl.launch(q, k, w, 17, True, 4)
    monkeypatch.setattr(impl, "requires_int64_offsets", lambda *tensors: True)
    actual = impl.launch(q, k, w, 17, True, 4)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = impl.launch(q, k, w, 17, True, 4)
    for _ in range(3):
        w.neg_()
        graph.replay()
        assert_indexer_selection(actual, q, k, w, 17, True, 4)


@pytest.mark.parametrize("heads,dim", [(1, 8), (256, 256)])
def test_gvr2_legacy_shape_boundaries(heads, dim):
    from attn_gym.sparse.indexer.impl.triton_gvr2 import launch

    inputs = make_indexer_test_inputs(17, heads, dim, torch.bfloat16, batch=1)
    # Unreachable singleton strides must not enter int32 constexpr pointer arithmetic.
    q, k, w = (
        tensor.as_strided(tensor.shape, (2**31 + 1, *tensor.stride()[1:])) for tensor in inputs
    )
    actual = launch(q, k, w, 9, False, 1)
    assert_indexer_selection(actual, q, k, w, 9, False)


@pytest.mark.parametrize("strided", [False, True])
def test_gvr2_reuses_score_and_selector_across_lengths(triton_compiles, strided):
    from attn_gym.sparse.indexer.impl.triton_gvr2 import launch

    # Vary T/S, batch strides, and the offset of the trailing score slab.
    for tokens in (1, 16, 17, 1025):
        q, k, w = make_indexer_test_inputs(tokens + int(strided), 5, 48, torch.bfloat16)
        if strided:
            # The feature strides depend on T/S, including a non-unit stride at T=1.
            q, k, w = (
                x.transpose(1, -1).contiguous().transpose(1, -1)[:, :tokens] for x in (q, k, w)
            )
        actual = launch(q, k, w, 1, True, 1)
        assert_indexer_selection(actual, q, k, w, 1, True)
    assert triton_compiles["_score_kernel"] == 1
    assert triton_compiles["_gvr2_topk_kernel"] == 1


def test_gvr2_zero_topk():
    from attn_gym.sparse.indexer.impl.triton_gvr2 import launch

    inputs = make_indexer_test_inputs(9, 2, 8, torch.bfloat16, batch=1)
    assert launch(*inputs, 0, True, 1).shape == (1, 9, 0)
