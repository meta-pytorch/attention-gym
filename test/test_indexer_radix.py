"""Reference and slab-boundary checks for the bounded-workspace CuTe indexer."""

import os

import pytest
import torch

from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.sparse.indexer import lightning_indexer
from attn_gym.sparse.indexer.impl import cute as impl
from attn_gym.testing.indexer import assert_indexer_selection, make_indexer_test_inputs

SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 0)


@pytest.mark.parametrize("batch", [1, 3, 64])
@pytest.mark.parametrize("tokens", [1, 65, 1023, 1024, 1025, 4096, 65537, 2**20])
def test_score_workspace_bound(batch, tokens):
    """Score scratch has both an absolute cap and a sequence-independent row cap."""
    pairs = impl.score_workspace_pairs(batch, tokens)
    assert 1 <= pairs <= min(512, batch * ((tokens + 1) // 2))
    assert pairs * 2 * tokens * 4 <= 32 * 1024 * 1024


@pytest.fixture
def radix_impl():
    """Load the optional backend only on compatible hardware."""
    if not SM100:
        pytest.skip("SM100 required for the CuTe radix port")
    pytest.importorskip("cutlass.cute")
    return impl


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("heads", [32, 64])
@pytest.mark.parametrize("tokens,topk", [(1, 1), (65, 0), (129, 37), (257, 128)])
def test_radix_selection(radix_impl, dtype, causal, heads, tokens, topk):
    """Both precisions and masks preserve indices and FP64 selection-regret bounds."""
    q, k, weights = make_indexer_test_inputs(tokens, heads, 128, dtype)
    actual = lightning_indexer(
        q, k, weights, topk, causal=causal, kernel_options={"backend": "cute"}
    )
    assert_indexer_selection(actual, q, k, weights, topk, causal)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("heads,head_dim", [(2, 16), (66, 48), (128, 128), (130, 96), (258, 384)])
def test_radix_generic_scores(radix_impl, dtype, causal, heads, head_dim):
    """Tiled score generation covers head/reduction tails without shape-sized SMEM."""
    inputs = make_indexer_test_inputs(65, heads, head_dim, dtype)
    actual = lightning_indexer(*inputs, 16, causal=causal, kernel_options={"backend": "cute"})
    assert_indexer_selection(actual, *inputs, 16, causal)


@pytest.mark.parametrize("causal", [False, True])
def test_radix_reuses_slab_across_batches(radix_impl, monkeypatch, causal):
    """An odd tail in one batch never aliases the next batch or a previous slab."""
    monkeypatch.setattr(radix_impl, "_MAX_SCORE_PAIRS", 3)
    inputs = make_indexer_test_inputs(17, 32, 128, torch.bfloat16, batch=3)
    actual = lightning_indexer(*inputs, 7, causal=causal, kernel_options={"backend": "cute"})
    assert_indexer_selection(actual, *inputs, 7, causal)


@pytest.mark.parametrize("distribution", ["random", "ties", "clustered", "signed_zero"])
@pytest.mark.parametrize("topk", [1, 37, 512])
def test_radix_threshold_and_overflow(radix_impl, distribution, topk):
    """Radix refinement handles negative scores, exact ties and shrink-buffer overflow."""
    import cutlass

    from attn_gym._backends.cute.target import (
        detect_compile_target,
        get_compile_target,
        set_compile_target,
    )

    tokens = 4097
    generator = torch.Generator(device="cuda").manual_seed(32)
    scores = torch.randn(2, 2, tokens, device="cuda", generator=generator)
    match distribution:
        case "ties":
            scores.fill_(-1)
        case "clustered":
            # All 4097 candidates land in bin0, overflowing the 2048-entry shrink slab;
            # tiny spacings also force refinement through the low ten key bits.
            scores = 1 + scores.abs() * 1e-5
        case "signed_zero":
            scores = torch.copysign(torch.zeros_like(scores), scores)
    output = torch.empty((1, tokens, topk), device="cuda", dtype=torch.int32)
    previous = get_compile_target()
    try:
        set_compile_target(detect_compile_target(torch.cuda.current_device()))
        kernel = radix_impl._compile_topk(topk, False, False)
    finally:
        set_compile_target(previous)
    kernel(scores, output, cutlass.Int32(0))
    indices = output[0, :4].long()
    assert ((indices >= 0) & (indices < tokens)).all()
    ordered = indices.sort(-1).values
    assert not (ordered[:, 1:] == ordered[:, :-1]).any()
    rows = scores.view(4, tokens)
    actual = rows.gather(1, indices).sort(-1).values
    expected = rows.topk(topk, sorted=False).values.sort(-1).values
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("heads,head_dim", [(64, 128), (66, 48)])
def test_radix_forced_int64(radix_impl, monkeypatch, heads, head_dim):
    """The wide ABI agrees with the ordinary address path on real data."""
    inputs = make_indexer_test_inputs(65, heads, head_dim, torch.float16)
    assert not requires_int64_abi(*inputs)
    expected = radix_impl.launch(*inputs, 16, True)
    monkeypatch.setattr(radix_impl, "requires_int64_abi", lambda *args: True)
    actual = radix_impl.launch(*inputs, 16, True)
    assert_indexer_selection(actual, *inputs, 16, True)
    torch.testing.assert_close(actual.sort(-1).values, expected.sort(-1).values)


@pytest.mark.parametrize(
    "tokens,heads,dim,unused_stride",
    [
        (65, 32, 128, 2**31 + 16),
        (65, 32, 128, 1),
        (1, 32, 128, 2**31 + 3),
        (65, 32, 128, 1 << 40),
        (65, 2, 16, 1),
        (65, 2, 16, 1 << 40),
    ],
)
def test_radix_singleton_stride(radix_impl, tokens, heads, dim, unused_stride):
    """Contiguous singleton modes may carry unaligned or unrepresentable unused strides."""
    inputs = make_indexer_test_inputs(tokens, heads, dim, torch.bfloat16, batch=1)
    views = tuple(
        x.as_strided(
            x.shape,
            (unused_stride, unused_stride if tokens == 1 else x.stride(1), *x.stride()[2:]),
        )
        for x in inputs
    )
    assert all(x.is_contiguous() for x in views)
    assert requires_int64_abi(*views) == (unused_stride > 2**31 - 1)
    topk = min(16, tokens)
    actual = radix_impl.launch(*views, topk, False)
    assert_indexer_selection(actual, *inputs, topk, False)


@pytest.mark.skipif(
    os.environ.get("ATTN_GYM_RUN_LARGE_INDEXER_TESTS") != "1",
    reason="set ATTN_GYM_RUN_LARGE_INDEXER_TESTS=1 for the 4 GiB active-offset test",
)
def test_radix_active_int64_offset(radix_impl):
    """Read meaningful Q values beyond signed-int32 element offsets, not just wide strides."""
    if torch.cuda.mem_get_info()[0] < 8 * 2**30:
        pytest.skip("8 GiB free device memory required")
    batch, tokens, heads, dim = 65, 2, 1024, 16384
    q = torch.zeros((batch, tokens, heads, dim), device="cuda", dtype=torch.float16)
    k = torch.zeros((batch, tokens, dim), device="cuda", dtype=q.dtype)
    weights = torch.zeros((batch, tokens, heads), device="cuda", dtype=q.dtype)
    signal = torch.where(torch.arange(batch * tokens, device="cuda") % 3 == 0, -1, 1)
    q[:, :, -1, -1] = signal.reshape(batch, tokens)
    k[:, 0, -1], k[:, 1, -1] = -1, 1
    weights[:, :, -1] = 1
    assert q.numel() > 2**31 and requires_int64_abi(q, k, weights)
    actual = lightning_indexer(q, k, weights, 1, kernel_options={"backend": "cute"})
    expected = (signal > 0).to(torch.int32).reshape(batch, tokens, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_radix_graph_replay(radix_impl, monkeypatch):
    """Captured slab reuse reads updated operands rather than stale score storage."""
    monkeypatch.setattr(radix_impl, "_MAX_SCORE_PAIRS", 3)
    inputs = make_indexer_test_inputs(17, 64, 128, torch.bfloat16, batch=2)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        radix_impl.launch(*inputs, 7, True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = radix_impl.launch(*inputs, 7, True)
    for seed in (81, 82):
        updated = make_indexer_test_inputs(17, 64, 128, torch.bfloat16, batch=2, seed=seed)
        for target, source in zip(inputs, updated):
            target.copy_(source)
        graph.replay()
        assert_indexer_selection(actual, *inputs, 7, True)
