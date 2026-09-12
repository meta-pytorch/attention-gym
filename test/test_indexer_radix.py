"""Reference and slab-boundary checks for the bounded-workspace CuTe indexer."""

import os
from unittest.mock import Mock

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
    pairs = impl.score_workspace_pairs(batch, tokens, tokens)
    assert 1 <= pairs <= min(512, batch * ((tokens + 1) // 2))
    assert pairs * 2 * tokens * 4 <= 32 * 1024 * 1024
    # Compressed candidates shrink each row, so more pairs fit in the same bytes.
    assert impl.score_workspace_pairs(batch, tokens, max(1, tokens // 4)) >= pairs


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
@pytest.mark.parametrize("heads,dim", [(32, 128), (66, 48)])
def test_radix_reuses_slab_across_batches(radix_impl, monkeypatch, causal, heads, dim):
    """Odd batch tails and a partial final slab never alias other score rows."""
    monkeypatch.setattr(radix_impl, "_MAX_SCORE_PAIRS", 4)
    inputs = make_indexer_test_inputs(17, heads, dim, torch.bfloat16, batch=3)
    actual = lightning_indexer(*inputs, 7, causal=causal, kernel_options={"backend": "cute"})
    assert_indexer_selection(actual, *inputs, 7, causal)


@pytest.mark.parametrize("batch,tokens,capacity", [(1, 1025, 512), (3, 17, 4)])
def test_radix_launches_only_active_pairs(radix_impl, monkeypatch, batch, tokens, capacity):
    """Both launch grids use the final slab's live extent, not its allocation capacity."""
    monkeypatch.setattr(radix_impl, "_MAX_SCORE_PAIRS", capacity)
    score_kernel, topk_kernel = Mock(), Mock()
    monkeypatch.setattr(radix_impl, "_compile_scores", lambda *args: score_kernel)
    monkeypatch.setattr(radix_impl, "_compile_topk", lambda *args: topk_kernel)
    q, k, weights = make_indexer_test_inputs(tokens, 32, 128, torch.bfloat16, batch=batch)
    inputs = (
        q.transpose(1, 2).contiguous().transpose(1, 2),
        k.transpose(0, 1).contiguous().transpose(0, 1),
        weights.transpose(1, 2).contiguous().transpose(1, 2),
    )
    lightning_indexer(*inputs, 7, causal=True, kernel_options={"backend": "cute"})
    total_pairs = batch * ((tokens + 1) // 2)
    expected = [min(capacity, total_pairs - start) for start in range(0, total_pairs, capacity)]
    assert [call.args[3].shape[0] for call in score_kernel.call_args_list] == expected
    assert [call.args[0].shape[0] for call in topk_kernel.call_args_list] == expected
    for call in score_kernel.call_args_list:
        for actual, original in zip(call.args[:3], inputs):
            assert actual.data_ptr() == original.data_ptr()
            assert actual.stride() == original.stride()
    assert all(
        score.args[3] is topk.args[0]
        for score, topk in zip(score_kernel.call_args_list, topk_kernel.call_args_list)
    )


@pytest.mark.parametrize("use_int64_offsets", [False, True])
@pytest.mark.parametrize("contiguous_weight_heads", [False, True])
def test_radix_fake_tensor_abi(monkeypatch, use_int64_offsets, contiguous_weight_heads):
    """Only Q/K's D stride and an available unit weight-head stride are static."""
    pytest.importorskip("cutlass.cute")
    monkeypatch.setattr(impl, "compile_tvm_ffi", lambda operation, *args: args)
    q, k, weights, scores, *_ = impl._compile_scores.__wrapped__(
        torch.bfloat16, 32, 128, True, 1, use_int64_offsets, contiguous_weight_heads
    )
    topk_scores, output, _ = impl._compile_topk.__wrapped__(37, True, 1, use_int64_offsets)
    width = 64 if use_int64_offsets else 32
    assert q._assumed_align == k._assumed_align == 16
    assert weights._assumed_align == 2
    for tensor in (q, k):
        assert tensor.stride[-1] == 1
        assert all(
            stride.width == width and stride.divisibility == 8 for stride in tensor.stride[:-1]
        )
    if contiguous_weight_heads:
        assert weights.stride[-1] == 1
    dynamic_weight_strides = weights.stride[:-1] if contiguous_weight_heads else weights.stride
    assert all(
        stride.width == width and stride.divisibility == 1 for stride in dynamic_weight_strides
    )
    for tensor in (scores, topk_scores, output):
        assert tensor.stride[-1] == 1
        assert tensor.stride[0].width == width


@pytest.mark.parametrize("operand", [0, 1, 2], ids=["q", "k", "weights"])
@pytest.mark.parametrize("heads,dim", [(32, 128), (66, 48)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_radix_input_alignment(radix_impl, operand, heads, dim, dtype):
    """Only TMA operands require 16-byte bases; contiguous weights may start at any element."""
    inputs = list(make_indexer_test_inputs(65, heads, dim, dtype))
    tensor = inputs[operand]
    shifted = torch.empty(tensor.numel() + 1, device=tensor.device, dtype=dtype)[1:]
    inputs[operand] = shifted.view_as(tensor).copy_(tensor)
    assert inputs[operand].is_contiguous() and inputs[operand].data_ptr() % 16 == 2
    if operand < 2:
        with pytest.raises(ValueError, match="16-byte aligned"):
            lightning_indexer(*inputs, 16, causal=True, kernel_options={"backend": "cute"})
    else:
        actual = lightning_indexer(*inputs, 16, causal=True, kernel_options={"backend": "cute"})
        assert_indexer_selection(actual, *inputs, 16, True)


@pytest.mark.parametrize("layout", ["permuted", "padded", "broadcast"])
@pytest.mark.parametrize("heads,dim", [(32, 128), (64, 128), (66, 96)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_radix_strided_inputs(radix_impl, monkeypatch, layout, heads, dim, dtype, causal):
    """Packed and generic scorers preserve independent outer strides without input copies."""
    batch, tokens = 3, 257
    monkeypatch.setattr(radix_impl, "_MAX_SCORE_PAIRS", 5)
    q, k, weights = make_indexer_test_inputs(tokens, heads, dim, dtype, batch=batch)
    match layout:
        case "permuted":
            q = q.permute(2, 0, 1, 3).contiguous().permute(1, 2, 0, 3)
            k = k.transpose(0, 1).contiguous().transpose(0, 1)
            weights = weights.permute(2, 1, 0).contiguous().permute(2, 1, 0)
        case "padded":
            q = torch.full(
                (batch, tokens + 1, heads + 2, dim + 16), torch.nan, device="cuda", dtype=dtype
            )[:, :tokens, :heads, :dim].copy_(q)
            k = torch.full((batch, tokens + 1, dim + 16), torch.nan, device="cuda", dtype=dtype)[
                :, :tokens, :dim
            ].copy_(k)
            weights = torch.full(
                (batch, tokens + 1, heads + 1), torch.nan, device="cuda", dtype=dtype
            )[:, :tokens, :heads].copy_(weights)
        case "broadcast":
            q = q[:, :, :1, :].expand_as(q)
            k = k[:1].expand_as(k)
            weights = weights[:, :1, :1].expand_as(weights)
    assert not q.is_contiguous() and not k.is_contiguous() and not weights.is_contiguous()
    actual = lightning_indexer(
        q, k, weights, 37, causal=causal, kernel_options={"backend": "cute"}
    )
    assert_indexer_selection(actual, q, k, weights, 37, causal)


@pytest.mark.parametrize("operand", [0, 1], ids=["q", "k"])
@pytest.mark.parametrize("layout", ["strided_reduction", "unaligned_rows"])
def test_radix_rejects_non_tma_layout(radix_impl, operand, layout):
    """Reject only the last-stride/alignment requirements needed by TMA."""
    inputs = list(make_indexer_test_inputs(65, 32, 128, torch.bfloat16))
    tensor = inputs[operand]
    if layout == "strided_reduction":
        inputs[operand] = tensor.transpose(-1, -2).contiguous().transpose(-1, -2)
    else:
        inputs[operand] = torch.empty(
            (*tensor.shape[:-1], tensor.shape[-1] + 1), device="cuda", dtype=tensor.dtype
        )[..., :-1].copy_(tensor)
    with pytest.raises(ValueError, match="unit last strides and 16-byte aligned"):
        lightning_indexer(*inputs, 16, kernel_options={"backend": "cute"})


@pytest.mark.parametrize("distribution", ["random", "ties", "clustered", "signed_zero"])
@pytest.mark.parametrize("topk", [1, 37, 512, 1024, 4096])
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
        kernel = radix_impl._compile_topk(topk, False, 1, False)
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


@pytest.mark.parametrize("compress_ratio", [1, 4])
@pytest.mark.parametrize("heads,head_dim", [(64, 128), (66, 48)])
def test_radix_forced_int64(radix_impl, monkeypatch, heads, head_dim, compress_ratio):
    """The wide ABI agrees with the ordinary address path on real data."""
    inputs = make_indexer_test_inputs(
        65, heads, head_dim, torch.float16, compress_ratio=compress_ratio
    )
    assert not requires_int64_abi(*inputs)
    expected = radix_impl.launch(*inputs, 16, True, compress_ratio)
    monkeypatch.setattr(radix_impl, "requires_int64_abi", lambda *args: True)
    actual = radix_impl.launch(*inputs, 16, True, compress_ratio)
    assert_indexer_selection(actual, *inputs, 16, True, compress_ratio)
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


@pytest.mark.parametrize("strided", [False, True])
def test_radix_graph_replay(radix_impl, monkeypatch, strided):
    """Captured partial-slab reuse reads updated operands, including independent outer strides."""
    monkeypatch.setattr(radix_impl, "_MAX_SCORE_PAIRS", 4)
    inputs = make_indexer_test_inputs(17, 64, 128, torch.bfloat16, batch=2)
    if strided:
        q, k, weights = inputs
        inputs = (
            q.transpose(1, 2).contiguous().transpose(1, 2),
            k.transpose(0, 1).contiguous().transpose(0, 1),
            weights.transpose(1, 2).contiguous().transpose(1, 2),
        )
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
