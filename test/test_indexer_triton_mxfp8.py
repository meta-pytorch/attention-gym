"""Native block-scaled Triton scoring and streaming selection, independent of CuTe."""

import math

import pytest
import torch

from attn_gym.sparse.indexer import lightning_indexer
from attn_gym.testing.indexer import assert_indexer_selection, make_indexer_mxfp8_test_inputs


@pytest.fixture(autouse=True)
def require_native_mxfp8():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("Native MXFP8 requires SM100 or SM103")
    pytest.importorskip("triton")


@pytest.mark.parametrize("heads,weights_dtype", [(32, torch.float32), (64, torch.bfloat16)])
@pytest.mark.parametrize("tokens,ratio,causal,topk", [(137, 4, True, 17), (65, 1, False, 33)])
def test_mxfp8_selection(heads, weights_dtype, tokens, ratio, causal, topk):
    q, k, w, qs, ks = make_indexer_mxfp8_test_inputs(
        tokens, heads, 128, weights_dtype, compress_ratio=ratio
    )
    actual = lightning_indexer(
        q,
        k,
        w,
        topk,
        causal=causal,
        compress_ratio=ratio,
        q_scale=qs,
        k_scale=ks,
        kernel_options={"backend": "triton"},
    )
    assert_indexer_selection(actual, q, k, w, topk, causal, ratio, q_scale=qs, k_scale=ks)


def test_mxfp8_strides_misalignment_and_int64(monkeypatch):
    from attn_gym.sparse.indexer.impl import triton as impl

    inputs = make_indexer_mxfp8_test_inputs(133, 32, 128, torch.float32, compress_ratio=4)
    strided = []
    for tensor in inputs:
        # All dimensions may be strided; same-size E8M0 byte views preserve group stride 2.
        storage = torch.empty(
            (*tensor.shape[:-1], tensor.shape[-1] * 2 + 1), device="cuda", dtype=tensor.dtype
        )
        strided.append(storage[..., 1::2].copy_(tensor))
    q, k, w, qs, ks = strided
    actual = impl.launch(q, k, w, 17, True, 4, qs, ks)
    assert_indexer_selection(actual, q, k, w, 17, True, 4, q_scale=qs, k_scale=ks)
    monkeypatch.setattr(impl, "requires_int64_offsets", lambda *tensors: True)
    wide = impl.launch(q, k, w, 17, True, 4, qs, ks)
    torch.testing.assert_close(actual, wide, rtol=0, atol=0)


@pytest.mark.parametrize("heads", [32, 64])
def test_mxfp8_score_4k_native_codegen(heads):
    from attn_gym.sparse.indexer.impl.triton import launch_mxfp8_scores

    q, k, w, qs, ks = make_indexer_mxfp8_test_inputs(
        4096, heads, 128, torch.float32, batch=1, compress_ratio=4
    )
    # Last query pairs exercise a full 1K candidate row, without a quadratic FP64 oracle.
    scores = torch.full((2, 2, 1024), float("nan"), device="cuda")
    kernel = launch_mxfp8_scores(q, k, w, qs, ks, scores, 2046, True, 4, False)
    assert "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.block32" in kernel.asm["ptx"]
    assert "ttng.tc_gen5_mma_scaled" in kernel.asm["ttgir"]
    for row in range(4):
        query = 4092 + row
        end = (query + 1) // 4
        q64 = q[0, query].double() * qs[0, query].double().repeat_interleave(32, -1)
        k64 = k[0, :end].double() * ks[0, :end].double().repeat_interleave(32, -1)
        w64 = w[0, query].double()
        expected = ((q64 @ k64.T).relu() * w64[:, None]).sum(0) / math.sqrt(heads * 128)
        magnitude = ((q64.abs() @ k64.abs().T) * w64.abs()[:, None]).sum(0)
        magnitude /= math.sqrt(heads * 128)
        error = (scores.view(4, -1)[row, :end].double() - expected).abs()
        eps = torch.finfo(torch.float32).eps
        assert torch.all(error <= (128 + heads + 4) * eps * magnitude)
        assert (
            error.square().mean().sqrt()
            <= 4 * (math.sqrt(128) + math.sqrt(heads)) * eps * magnitude.square().mean().sqrt()
        )
        assert torch.isnan(scores.view(4, -1)[row, end:]).all()


@pytest.mark.parametrize("topk", [0, 33])
def test_mxfp8_zero_weights_graph(topk):
    from attn_gym.sparse.indexer.impl.triton import launch

    q, k, w, qs, ks = make_indexer_mxfp8_test_inputs(
        133, 64, 128, torch.bfloat16, batch=1, compress_ratio=4
    )
    w.zero_()
    launch(q, k, w, topk, True, 4, qs, ks)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = launch(q, k, w, topk, True, 4, qs, ks)
    for _ in range(3):
        qs.view(torch.uint8).add_(1)
        graph.replay()
        assert_indexer_selection(actual, q, k, w, topk, True, 4, q_scale=qs, k_scale=ks)


def test_mxfp8_wide_exponents_and_nonmutation():
    from attn_gym.sparse.indexer.impl.triton import launch

    q, k, w, qs, ks = make_indexer_mxfp8_test_inputs(65, 32, 128, torch.float32, batch=1)
    for scales, width in ((qs, 41), (ks, 61)):
        scales.view(torch.uint8).copy_(
            (
                torch.arange(scales.numel(), device="cuda").reshape(scales.shape) % width
                + 127
                - width // 2
            ).to(torch.uint8)
        )
    snapshots = [x.view(torch.uint8).clone() for x in (q, k, w, qs, ks)]
    actual = launch(q, k, w, 17, False, 1, qs, ks)
    assert_indexer_selection(actual, q, k, w, 17, False, q_scale=qs, k_scale=ks)
    for tensor, snapshot in zip((q, k, w, qs, ks), snapshots, strict=True):
        assert torch.equal(tensor.view(torch.uint8), snapshot)
