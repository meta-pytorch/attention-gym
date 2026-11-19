"""Decode semantics, cached history, registration, and dynamic metadata reuse."""

import pytest
import torch

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.sparse import PagedKV, paged_lightning_indexer, paged_sparse_decode
from attn_gym.sparse.indexer.ops import _paged_indexer_op
from attn_gym.testing.indexer import (
    assert_paged_indexer_selection,
    make_paged_indexer_test_inputs,
)

CUDA = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
pytestmark = pytest.mark.skipif(not CUDA, reason="SM80+ required")


@pytest.fixture(params=["triton", "cute"])
def backend(request):
    if request.param == "cute":
        if not cuda_runs_sm100_kernels():
            pytest.skip("SM100-family GPU required")
        pytest.importorskip("cutlass.cute")
    return request.param


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ratio,topk", [(1, 0), (1, 7), (4, 17), (7, 1024)])
def test_paged_selection(dtype, ratio, topk):
    args = make_paged_indexer_test_inputs(
        compress_ratio=ratio,
        page_size=13,
        dtype=dtype,
        device="cuda",
        tail=2,
    )
    actual = paged_lightning_indexer(*args, topk=topk, compress_ratio=ratio)
    assert_paged_indexer_selection(args, actual, topk, ratio)


@pytest.mark.parametrize("heads,dim", [(1, 7), (4, 128), (17, 48)])
@pytest.mark.parametrize("wide", [False, True])
def test_positions_layouts_and_wide_offsets(monkeypatch, heads, dim, wide):
    from attn_gym.sparse.indexer.impl import paged_triton

    if wide:
        monkeypatch.setattr(paged_triton, "requires_int64_offsets", lambda *args: True)
    args = list(make_paged_indexer_test_inputs(heads=heads, head_dim=dim))
    for i in (0, 1, 2, 3):
        args[i] = args[i].transpose(0, -1).contiguous().transpose(0, -1)
    positions = torch.tensor([2**31 - 1, -1, 3, 11, 23, 511], device="cuda", dtype=torch.int32)
    actual = paged_lightning_indexer(*args, topk=3, query_positions=positions)
    assert_paged_indexer_selection(args, actual, 3, positions=positions)


@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("topk,candidates", [(7, 16), (512, 8192)])
def test_opcheck_and_public_compile(backend, permuted, topk, candidates, deterministic_algorithms):
    compiled = torch.compile(paged_lightning_indexer, fullgraph=True, dynamic=True)
    kwargs = {"topk": topk, "kernel_options": {"backend": backend}}
    for step, length in enumerate((candidates, candidates + 1, candidates * 4)):
        args = list(
            make_paged_indexer_test_inputs(
                (step + 1, step + 2), (4 * length + 3, 4 * length + 25), heads=3, head_dim=16
            )
        )
        if permuted:
            for i in range(4):
                args[i] = args[i].transpose(0, -1).contiguous().transpose(0, -1)
        for tensor in args[:3]:
            tensor.requires_grad_(True)
        if step == 0:
            torch.library.opcheck(_paged_indexer_op, (*args, topk, 4, None, backend))
        actual = compiled(*args, **kwargs)
        assert_paged_indexer_selection(args, actual, topk)
        torch.testing.assert_close(actual, paged_lightning_indexer(*args, **kwargs))


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("page_size", [13, 64])
def test_graph_replay_observes_metadata(
    backend, wide, page_size, monkeypatch, deterministic_algorithms
):
    from attn_gym.sparse.indexer.impl import paged_triton

    if wide:
        monkeypatch.setattr(paged_triton, "requires_int64_offsets", lambda *args: True)
    args = list(
        make_paged_indexer_test_inputs((2, 1), (4 * 33024, 4 * 32768), tail=1, page_size=page_size)
    )
    if page_size == 13:
        args[0] = args[0].transpose(1, 2).contiguous().transpose(1, 2)
    positions = torch.tensor(
        [4 * 32768 - 1, 4 * 33024 - 1, 17, -1], device="cuda", dtype=torch.int32
    )

    def call():
        return paged_lightning_indexer(
            *args, topk=512, query_positions=positions, kernel_options={"backend": backend}
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    for visible in (100, 8193, 32768, 33024):
        args[-1][0] = 4 * visible
        positions[0] = 4 * visible - 1
        args[2][1].zero_()
        args[3][:, :2] = args[3][:, :2].flip(1)
        graph.replay()
        assert_paged_indexer_selection(args, output, 512, positions=positions)
        torch.testing.assert_close(output, call())


def test_raw_kv_decode_integration():
    # Complete candidate blocks select raw-token blocks; decode includes the
    # incomplete causal tail. No index conversion or candidate page IDs escape.
    args = make_paged_indexer_test_inputs((1, 2), (71, 42), head_dim=16)
    indices = paged_lightning_indexer(*args, topk=7)
    query = args[0]
    k = torch.randn(12, 16, 1, 16, device="cuda", dtype=query.dtype)
    v = torch.randn_like(k)
    table = torch.randperm(12, device="cuda").int().reshape(2, 6)
    kv = PagedKV(k, v, table)
    kwargs = {"max_seqlen_q": 2, "index_block_size": 4}
    actual = paged_sparse_decode(query, kv, indices, args[4], args[5], **kwargs)
    reference = paged_sparse_decode(
        query, kv, indices, args[4], args[5], impl="reference", **kwargs
    )
    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)


@pytest.mark.parametrize("permuted", [False, True])
def test_shape_specialization_reuse(backend, permuted, triton_compiles):
    from attn_gym.sparse.indexer.impl.cute import _compile_topk

    if backend == "cute":
        _compile_topk.cache_clear()
    # Cross ==1, multiple-of-16 and odd classes for batch, queries, and page
    # table capacity. Keep the portable selector's clamped BLOCK bucket fixed.
    for batch, candidates in [(1, 4160), (16, 4224), (17, 4288)]:
        args = list(make_paged_indexer_test_inputs((1,) * batch, (4 * candidates,) * batch))
        if permuted:
            for i in range(4):
                args[i] = args[i].transpose(0, -1).contiguous().transpose(0, -1)
        output = paged_lightning_indexer(*args, topk=512, kernel_options={"backend": backend})
        assert_paged_indexer_selection(args, output, 512)
    if not permuted and torch.cuda.get_device_capability()[0] >= 9:
        # One low-fill and one medium-fill TMA schedule; runtime sizes within
        # each schedule, including divisible/odd classes, do not specialize.
        assert triton_compiles["_score_tma"] == 2
    else:
        assert triton_compiles["_score"] == 1
    if backend == "triton":
        assert triton_compiles["_select"] == 1
    else:
        info = _compile_topk.cache_info()
        assert info.currsize == 1 and info.hits + info.misses == 3


@pytest.mark.parametrize("topk", [7, 512])
def test_zero_weights_and_pool_growth(backend, topk):
    capacity = 4 * 16389 + 1
    args = make_paged_indexer_test_inputs((1,), (capacity,))
    args[2].zero_()
    ids = torch.arange(topk, device="cuda", dtype=torch.int32)[None]
    for length in (3, 4, 5, 7, 8, 17, 4096, capacity):
        args[-1].fill_(length)
        selected = paged_lightning_indexer(*args, topk=topk, kernel_options={"backend": backend})
        torch.testing.assert_close(selected, torch.where(ids < length // 4, ids, -1))


def test_zero_sequences_and_empty_gpu():
    args = list(make_paged_indexer_test_inputs((0,), (0,), tail=2))
    args[3] = args[3][:0]
    args[4] = args[4][:1]
    args[5] = args[5][:0]
    actual = paged_lightning_indexer(*args, topk=3)
    assert actual.shape == (2, 3) and (actual == -1).all()


@pytest.mark.parametrize("page_size", [32, 96, 128])
@pytest.mark.parametrize("misaligned", [False, True])
def test_page_tiles_and_alignment(page_size, misaligned):
    args = list(make_paged_indexer_test_inputs(page_size=page_size))
    if misaligned:
        for i in (0, 1):
            copy = args[i].new_empty(args[i].numel() + 1)[1:].view(args[i].shape)
            copy.copy_(args[i])
            args[i] = copy
    actual = paged_lightning_indexer(*args, topk=7)
    assert_paged_indexer_selection(args, actual, 7)


def test_workspace_reused_across_slabs():
    # Large reserved capacity, few live candidates. A pair is exactly the
    # 32 MiB score budget; the third query must reuse that scratch correctly.
    args = list(make_paged_indexer_test_inputs((1, 1, 1), (65, 129, 257)))
    table = torch.full((3, 2**22 // 64), -1, device="cuda", dtype=torch.int32)
    table[:, : args[3].shape[1]] = args[3]
    args[3] = table
    actual = paged_lightning_indexer(*args, topk=7)
    assert_paged_indexer_selection(args, actual, 7)


@pytest.mark.parametrize("page_size", [13, 64])
def test_active_cache_address_above_int32(page_size):
    from attn_gym._backends.triton.utils import requires_int64_offsets

    args = list(
        make_paged_indexer_test_inputs(
            (1, 1),
            (4 * page_size,) * 2,
            page_size=page_size,
            head_dim=16,
        )
    )
    stride = 2**31 + 64
    size = stride + page_size * 16
    if torch.cuda.mem_get_info()[0] < size * 2 + 2**30:
        pytest.skip("requires 4 GiB free for an active wide cache-page offset")
    storage = args[1].new_empty(size)
    wide_cache = storage.as_strided((2, page_size, 16), (stride, 16, 1))
    wide_cache.copy_(args[1])
    assert requires_int64_offsets(wide_cache)
    args[1] = wide_cache
    actual = paged_lightning_indexer(*args, topk=7)
    assert_paged_indexer_selection(args, actual, 7)
