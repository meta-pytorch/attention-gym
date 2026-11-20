"""Key-set semantics, fused numerical accuracy, and kernel reuse for paged sparse decode."""

import dataclasses

import pytest
import torch

from attn_gym.sparse import AuxRequest, PagedKV, paged_sparse_decode
from attn_gym.sparse.paged_decode.ops import decode_op, decode_with_lse_op
from attn_gym.testing.kda import assert_relative_rms_within
from attn_gym.testing.paged_decode import make_paged_sparse_decode_inputs

CUDA = torch.cuda.is_available() and not torch.version.hip
requires_cuda = pytest.mark.skipif(not CUDA, reason="NVIDIA CUDA required")


def _fp64_reference(inputs):
    """Use float64 as the oracle for both fused and low-precision reference error."""

    def widen(value):
        if isinstance(value, PagedKV):
            return PagedKV(value.key_cache.double(), value.value_cache.double(), value.block_table)
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.double()
        return value

    widened = {name: widen(value) for name, value in inputs.items()}
    return paged_sparse_decode(**widened, impl="reference", return_aux=AuxRequest(lse=True))


def _assert_matches_reference(inputs, **kwargs):
    out, aux = paged_sparse_decode(**inputs, return_aux=AuxRequest(lse=True), **kwargs)
    expected, expected_aux = _fp64_reference(inputs)
    low, _ = paged_sparse_decode(**inputs, impl="reference", return_aux=AuxRequest(lse=True))
    assert out.shape == expected.shape and out.dtype == inputs["query"].dtype
    # Allow twice the sum of the reference error and a BF16-sized rounding allowance.
    error = (out.double() - expected).abs().max().item()
    budget = 2 * (
        (low.double() - expected).abs().max().item() + 2**-8 * expected.abs().max().item()
    )
    assert error <= budget, f"max error {error:.3e} exceeds {budget:.3e}"
    assert_relative_rms_within(out, expected, "output", max_eps=2.0)
    lse, expected_lse = aux.lse, expected_aux.lse
    assert torch.equal(torch.isneginf(lse), torch.isneginf(expected_lse))
    finite = torch.isfinite(expected_lse)
    torch.testing.assert_close(lse[finite], expected_lse[finite].float(), atol=1e-4, rtol=1e-5)


def test_reference_matches_explicit_key_sets():
    """Shared sources remove sparse/window overlap, not repeated sparse selections."""
    torch.manual_seed(0)
    page_size, block, heads, dim = 2, 2, 2, 4
    keys = torch.randn(6, page_size, 1, dim, dtype=torch.float64)
    values = torch.randn(6, page_size, 1, dim, dtype=torch.float64)
    block_table = torch.tensor([[4, 1, 5, 0, 3]], dtype=torch.int32)
    query = torch.randn(1, heads, dim, dtype=torch.float64)
    sink = torch.tensor([0.3, -1.0], dtype=torch.float64)
    # At position 8, a size-2 window aligns from 7 down to 6 and includes entries 6..8.
    # Block id 3 overlaps that window and must be skipped; block id 0 must contribute twice.
    indices = torch.tensor([[1, 3, 0, -1, 0]], dtype=torch.int32)
    out, aux = paged_sparse_decode(
        query,
        PagedKV(keys, values, block_table),
        indices,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([9], dtype=torch.int32),
        max_seqlen_q=1,
        index_block_size=block,
        window_size=2,
        attention_sink=sink,
        scale=0.5,
        impl="reference",
        return_aux=AuxRequest(lse=True),
    )
    positions = torch.tensor([2, 3, 0, 1, 0, 1, 6, 7, 8])
    pages = block_table[0, positions // page_size].long()
    key_rows = keys[pages, positions % page_size, 0]
    value_rows = values[pages, positions % page_size, 0]
    logits = torch.cat((query[0] @ key_rows.T * 0.5, sink[:, None]), dim=1)
    expected_lse = logits.logsumexp(dim=1)
    expected = torch.softmax(logits, dim=1)[:, :-1] @ value_rows
    torch.testing.assert_close(out[0], expected)
    torch.testing.assert_close(aux.lse[0], expected_lse.float())


def test_reference_separate_window_source():
    """Separate sources retain overlapping logical positions and use independent page tables."""
    torch.manual_seed(1)
    sparse = PagedKV(
        *(torch.randn(3, 4, 1, 8, dtype=torch.float64) for _ in range(2)),
        torch.tensor([[2, 0]], dtype=torch.int32),
    )
    window = PagedKV(
        *(torch.randn(5, 2, 1, 8, dtype=torch.float64) for _ in range(2)),
        torch.tensor([[4, 3, 1, 0]], dtype=torch.int32),
    )
    query = torch.randn(1, 2, 8, dtype=torch.float64)
    out = paged_sparse_decode(
        query,
        sparse,
        torch.tensor([[5, 1]], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([7], dtype=torch.int32),
        max_seqlen_q=1,
        window_size=3,
        window_kv=window,
        impl="reference",
    )
    sparse_rows = [(sparse.block_table[0, e // 4], e % 4) for e in (5, 1)]
    window_rows = [(window.block_table[0, p // 2], p % 2) for p in (4, 5, 6)]
    keys = torch.stack(
        [sparse.key_cache[p, s, 0] for p, s in sparse_rows]
        + [window.key_cache[p, s, 0] for p, s in window_rows]
    )
    values = torch.stack(
        [sparse.value_cache[p, s, 0] for p, s in sparse_rows]
        + [window.value_cache[p, s, 0] for p, s in window_rows]
    )
    expected = torch.softmax(query[0] @ keys.T / 8**0.5, dim=1) @ values
    torch.testing.assert_close(out[0], expected)


SCENARIOS = {
    "qsa_tp1": {"seqlens": [5000, 37, 4102, 9], "query_lens": [1, 1, 1, 1]},
    "qsa_tp4": {"seqlens": [3001, 100], "query_lens": [1, 1], "heads": 6, "kv_heads": 1},
    "qsa_mtp": {"seqlens": [3000, 901, 10], "query_lens": [3, 3, 3]},
    "qsa_scattered_padding": {"seqlens": [3000, 100], "query_lens": [1, 2], "topk_length": False},
    "qsa_page_not_block_multiple": {"seqlens": [700, 64], "query_lens": [1, 1], "page_size": 6},
    "shared_window_sink": {
        "seqlens": [3000, 50],
        "query_lens": [2, 1],
        "window_size": 100,
        "attention_sink": True,
    },
    "dsa_token_topk": {
        "seqlens": [3000, 50],
        "query_lens": [1, 1],
        "index_block_size": 1,
        "topk": 256,
        "heads": 16,
        "kv_heads": 1,
        "head_dim": 128,
    },
    "dsv4_compressed_plus_window": {
        "seqlens": [3000, 50, 700],
        "query_lens": [1, 1, 2],
        "index_block_size": 1,
        "topk": 128,
        "compress_ratio": 4,
        "window_size": 128,
        "window_page_size": 32,
        "attention_sink": True,
        "heads": 16,
        "kv_heads": 1,
        "head_dim": 128,
        "latent_values": True,
    },
    "dsv4_tp8_head_dim_512": {
        "seqlens": [5000, 300, 64000],
        "query_lens": [1, 1, 2],
        "index_block_size": 1,
        "topk": 512,
        "compress_ratio": 4,
        "window_size": 128,
        "attention_sink": True,
        "heads": 8,
        "kv_heads": 1,
        "head_dim": 512,
        "latent_values": True,
    },
    "latent_value_prefix": {
        "seqlens": [1000, 50],
        "query_lens": [1, 1],
        "index_block_size": 1,
        "topk": 128,
        "heads": 16,
        "kv_heads": 1,
        "head_dim": 128,
        "head_dim_v": 64,
        "latent_values": True,
    },
    "fp16": {"seqlens": [2000, 30], "query_lens": [1, 1], "dtype": torch.float16},
    "more_queries_than_keys": {
        "seqlens": [2, 1000],
        "query_lens": [3, 1],
        "index_block_size": 1,
        "topk": 64,
        "attention_sink": True,
    },
    # Small grids exercise one-tile and looped splits, empty rows, and row-major K loads.
    "small_batch_qsa_mtp": {"seqlens": [3000], "query_lens": [3], "heads": 6, "kv_heads": 1},
    "small_batch_partial_block": {
        "seqlens": [3003],
        "query_lens": [1],
        "heads": 6,
        "kv_heads": 1,
    },
    "small_batch_empty_row": {
        "seqlens": [0],
        "query_lens": [1],
        "heads": 6,
        "kv_heads": 1,
        "attention_sink": True,
    },
    "small_batch_large_budget": {
        "seqlens": [10003],
        "query_lens": [1],
        "heads": 6,
        "kv_heads": 1,
        "topk": 2048,
    },
    "small_batch_looped_budget": {
        "seqlens": [40003],
        "query_lens": [1],
        "heads": 6,
        "kv_heads": 1,
        "head_dim": 32,
        "topk": 8192,
    },
    "small_batch_shared_window_sink": {
        "seqlens": [3000],
        "query_lens": [1],
        "window_size": 100,
        "attention_sink": True,
    },
    "small_batch_dsv4_head_dim_512": {
        "seqlens": [64000],
        "query_lens": [1],
        "index_block_size": 1,
        "topk": 512,
        "compress_ratio": 4,
        "window_size": 128,
        "attention_sink": True,
        "heads": 8,
        "kv_heads": 1,
        "head_dim": 512,
        "latent_values": True,
    },
    "wide_fp16": {
        "seqlens": [600, 200],
        "query_lens": [2, 2],
        "heads": 8,
        "kv_heads": 1,
        "head_dim": 512,
        "topk": 128,
        "dtype": torch.float16,
    },
    "large_head_group": {
        "seqlens": [600, 200],
        "query_lens": [2, 2],
        "heads": 48,
        "kv_heads": 1,
        "topk": 128,
    },
}


@requires_cuda
@pytest.mark.parametrize("name", SCENARIOS)
def test_fused_matches_reference(name):
    _assert_matches_reference(make_paged_sparse_decode_inputs(**SCENARIOS[name]))


@requires_cuda
@pytest.mark.parametrize("splits", [1, 2, 7, 64])
def test_split_counts_agree(splits, monkeypatch):
    """Changing the split count preserves attention; empty splits contribute nothing."""
    from attn_gym.sparse.paged_decode.impl import triton

    inputs = make_paged_sparse_decode_inputs(
        [0, 3000, 300, 0, 100, 0], [0, 1, 17, 0, 1, 0], attention_sink=True
    )
    monkeypatch.setattr(triton, "_choose_splits", lambda *args: splits)
    _assert_matches_reference(inputs)


@requires_cuda
@pytest.mark.parametrize("max_seqlen_q", [33, 32768])
def test_mixed_queries_launch_only_packed_rows(max_seqlen_q, monkeypatch):
    """Long queries must not multiply decode work or overflow CUDA's second grid axis."""
    from attn_gym.sparse.paged_decode.impl import triton

    inputs = make_paged_sparse_decode_inputs(
        [0, 100, 100, 0, 100, 0], [0, 1, 33, 0, 1, 0], head_dim=32, topk=16
    )
    inputs["max_seqlen_q"] = max_seqlen_q
    kernel = triton._paged_sparse_decode_kernel

    class CheckedGrid:
        def __getitem__(self, grid):
            assert grid[0] * grid[1] <= (
                inputs["query"].shape[0] * inputs["kv"].key_cache.shape[2]
            )
            return kernel[grid]

    monkeypatch.setattr(triton, "_paged_sparse_decode_kernel", CheckedGrid())
    _assert_matches_reference(inputs)


@requires_cuda
def test_empty_batches_and_selections():
    inputs = make_paged_sparse_decode_inputs([100], [1])
    empty = {
        **inputs,
        "query": inputs["query"][:0],
        "indices": inputs["indices"][:0],
        "cu_seqlens_q": torch.zeros(2, dtype=torch.int32, device="cuda"),
        "topk_length": inputs["topk_length"][:0],
    }
    assert paged_sparse_decode(**empty).shape == (0, 24, 256)
    # At 100 keys the final 4-entry block is complete, so no window entries fill the empty set.
    no_topk = {**inputs, "indices": inputs["indices"][:, :0], "topk_length": None}
    _assert_matches_reference(no_topk)


@requires_cuda
@pytest.mark.parametrize("small", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
def test_cuda_graph_replay_tracks_new_inputs(small, mixed):
    kwargs = {"heads": 6, "kv_heads": 1} if small else {}
    query_lens = [0, 1, 7, 0, 1, 0] if mixed else [1, 1, 1]
    fresh_lens = [1, 0, 0, 6, 0, 0] if mixed else query_lens
    seqlens = [3003, 900, 64, 128, 32, 0] if mixed else [3003, 900, 64]
    inputs = make_paged_sparse_decode_inputs(seqlens, query_lens, seed=3, **kwargs)
    fresh = make_paged_sparse_decode_inputs(seqlens, fresh_lens, seed=4, **kwargs)
    paged_sparse_decode(**inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = paged_sparse_decode(**inputs)
    for name in ("query", "indices", "topk_length"):
        inputs[name][: fresh[name].shape[0]].copy_(fresh[name])
    inputs["cu_seqlens_q"].copy_(fresh["cu_seqlens_q"])
    inputs["kv"].key_cache.copy_(fresh["kv"].key_cache)
    inputs["kv"].value_cache.copy_(fresh["kv"].value_cache)
    graph.replay()
    expected = paged_sparse_decode(**inputs, impl="reference")
    num_live = sum(fresh_lens)
    torch.testing.assert_close(out[:num_live], expected[:num_live], atol=2e-2, rtol=2e-2)
    assert_relative_rms_within(
        out[:num_live], expected[:num_live].double(), "graph replay", max_eps=2.0
    )


@requires_cuda
def test_mixed_query_shapes_reuse_kernels(triton_compiles):
    """Packed row lookup must not specialize on sequence count or maximum query length."""
    compiled = torch.compile(paged_sparse_decode, fullgraph=True, dynamic=True)
    for num_sequences, query_len in ((1, 17), (16, 2), (17, 16), (33, 3)):
        query_lens = [1] * (num_sequences - 1) + [query_len]
        inputs = make_paged_sparse_decode_inputs(
            [100] * num_sequences, query_lens, heads=2, kv_heads=1, head_dim=32, topk=16
        )
        actual = compiled(**inputs)
        expected = paged_sparse_decode(**inputs, impl="reference")
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        assert_relative_rms_within(actual, expected.double(), "dynamic query lengths", max_eps=2.0)
    assert triton_compiles["_paged_sparse_decode_kernel"] == 1


@requires_cuda
@pytest.mark.parametrize("topk", [512, 1024, 2048])
def test_compiled_calls_reuse_kernels(triton_compiles, topk):
    """Changing runtime sizes must reuse producer kernels within each tile configuration."""
    compiled = torch.compile(paged_sparse_decode, fullgraph=True, dynamic=True)
    # Cross both tile-configuration boundaries and Triton's usual integer specialization classes.
    for rows in (1, 2, 3, 4, 16, 17, 48, 5):
        inputs = make_paged_sparse_decode_inputs(
            [2100 + 7 * i for i in range(rows)], [1] * rows, heads=6, kv_heads=1, topk=topk
        )
        out = compiled(**inputs)
        expected = paged_sparse_decode(**inputs, impl="reference")
        assert_relative_rms_within(out, expected.double(), f"rows={rows}", max_eps=2.0)
    # For these large budgets, one/two pairs use 8 warps and three pairs use 4; larger grids
    # share a 2-warp kernel. The 512-block budget needs only two configurations.
    assert triton_compiles["_paged_sparse_decode_kernel"] == (2 if topk == 512 else 3)
    # With these fixed head dimensions, merge variants depend on split buckets, not exact counts.
    assert triton_compiles["_merge_splits_kernel"] <= 4


@requires_cuda
@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("topk_length", [False, True])
@pytest.mark.parametrize("separate_window", [False, True])
def test_capacity_changes_reuse_kernels(triton_compiles, topk_length, separate_window, rows):
    """Changing sparse/table capacities must not specialize the fixed producer tile shape."""
    compiled = torch.compile(paged_sparse_decode, fullgraph=True, dynamic=True)
    # Cross sparse-tile counts and table widths without changing the 16-entry (rows=1) or
    # 64-entry (rows=4) producer tile. Split-count buckets may still change.
    for topk, context in ((1, 1), (16, 960), (17, 1024), (65, 1088), (66, 2048), (68, 2112)):
        inputs = make_paged_sparse_decode_inputs(
            [context] * rows,
            [1] * rows,
            heads=2,
            kv_heads=1,
            head_dim=32,
            topk=topk,
            topk_length=topk_length,
            index_block_size=1,
            window_size=8,
            compress_ratio=4 if separate_window else 1,
        )
        # Remove spare pages to exercise Triton's usual size-one specialization boundary.
        if context == 1:
            for name in ("kv", "window_kv"):
                source = inputs[name]
                if source is not None:
                    inputs[name] = dataclasses.replace(
                        source, block_table=source.block_table[:, :1].contiguous()
                    )
        expected = paged_sparse_decode(**inputs, impl="reference")
        actual = compiled(**inputs)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        assert_relative_rms_within(actual, expected.double(), "dynamic capacity", max_eps=2.0)
    assert triton_compiles["_paged_sparse_decode_kernel"] == 1
    assert triton_compiles["_merge_splits_kernel"] <= (3 if rows == 1 else 2)


def test_int64_offset_inventory():
    pytest.importorskip("triton")
    from attn_gym.sparse.paged_decode.impl import triton

    names = (
        "Q",
        "K",
        "V",
        "BlockTable",
        "Indices",
        "Out",
        "PartialOut",
        "PartialStats",
        "WK",
        "WV",
        "WBlockTable",
    )
    small = torch.empty(16, device="meta")
    wide = torch.empty_strided((3, 16), (2**30, 1), device="meta")
    singleton = torch.empty_strided((1, 16), (2**32, 1), device="meta")
    args = {
        **dict.fromkeys(names, small),
        "SHARED_WINDOW": False,
        "WINDOW": 8,
        "topk": 16,
        "INDEX_BLOCK": 4,
        "table_width": 1,
        "PAGE_SIZE": 64,
        "wtable_width": 1,
        "WINDOW_PAGE_SIZE": 64,
    }
    assert not triton._requires_int64_decode_offsets(args)
    for name in names:
        assert triton._requires_int64_decode_offsets({**args, name: wide}), name
        # A singleton axis never contributes its stride to a reachable relative offset.
        assert not triton._requires_int64_decode_offsets({**args, name: singleton}), name
    assert not triton._requires_int64_decode_offsets({**args, "WK": wide, "WINDOW": 0})
    for name in ("topk", "table_width", "wtable_width"):
        assert triton._requires_int64_decode_offsets({**args, name: 2**30}), name
    assert triton._requires_int64_decode_offsets(
        {**args, "table_width": 2**31 - 1, "PAGE_SIZE": 1, "INDEX_BLOCK": 3}
    )


@requires_cuda
@pytest.mark.parametrize("name", ["qsa_tp1", "small_batch_qsa_mtp", "dsv4_compressed_plus_window"])
def test_forced_int64_matches_default(name, monkeypatch):
    from attn_gym.sparse.paged_decode.impl import triton

    inputs = make_paged_sparse_decode_inputs(**SCENARIOS[name])
    expected, expected_aux = paged_sparse_decode(**inputs, return_aux=AuxRequest(lse=True))
    calls = []

    def force(*tensors):
        calls.append(True)
        return True

    monkeypatch.setattr(triton, "requires_int64_offsets", force)
    actual, aux = paged_sparse_decode(**inputs, return_aux=AuxRequest(lse=True))
    assert calls
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(aux.lse, expected_aux.lse, atol=0, rtol=0)


@requires_cuda
def test_cache_offsets_beyond_int32():
    """Keep correct and wrapped addresses allocated, with a sentinel to expose int32 overflow."""
    if torch.cuda.mem_get_info()[0] < 10 * 2**30:
        pytest.skip("wide-stride regression needs 8 GiB of free device storage")
    origin, stride = 2**31, 2**30
    storage = torch.empty(origin + 2 * stride + 16, device="cuda", dtype=torch.bfloat16)
    storage[:16].fill_(42)
    cache = storage.as_strided((1, 3, 1, 16), (3 * stride, stride, 16, 1), origin)
    cache.copy_(torch.ones_like(cache))
    inputs = make_paged_sparse_decode_inputs(
        [3], [1], heads=1, kv_heads=1, head_dim=16, topk=1, index_block_size=1
    )
    inputs["kv"] = PagedKV(cache, cache, torch.zeros((1, 1), device="cuda", dtype=torch.int32))
    inputs["indices"].fill_(2)
    expected = paged_sparse_decode(**inputs, impl="reference")
    actual = paged_sparse_decode(**inputs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@requires_cuda
def test_shared_window_start_near_int32_max():
    """A valid int32 window start can overflow the block-bound ceiling division's addition."""
    page_size, index_block_size, head_dim = 64, 3, 16
    # 2**31 - 2 is divisible by 3: alignment leaves it unchanged, but cdiv's addition of
    # INDEX_BLOCK - 1 exceeds INT32_MAX before dividing by 3.
    seqused = 2**31 - 2
    torch.manual_seed(0)
    key = torch.randn(1, page_size, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    query = torch.randn(1, 1, head_dim, device="cuda", dtype=torch.bfloat16)
    # Alias every logical page to physical page 0 to test large positions with a one-page cache.
    pages = -(-seqused // page_size)
    table = torch.zeros((1, pages), device="cuda", dtype=torch.int32)
    inputs = {
        "query": query,
        "kv": PagedKV(key, value, table),
        "indices": torch.zeros((1, 1), device="cuda", dtype=torch.int32),
        "cu_seqlens_q": torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        "seqused_k": torch.tensor([seqused], device="cuda", dtype=torch.int32),
        "max_seqlen_q": 1,
        "index_block_size": index_block_size,
    }
    # The aligned, size-zero window contributes nothing, so block id 0 supplies all three keys.
    expected = paged_sparse_decode(**inputs, impl="reference")
    assert expected.abs().amax() > 0
    torch.testing.assert_close(paged_sparse_decode(**inputs), expected, atol=0, rtol=0)


@requires_cuda
@pytest.mark.parametrize("separate_window", [False, True])
def test_oversized_block_ids_are_ignored(separate_window):
    inputs = make_paged_sparse_decode_inputs(
        [32],
        [1],
        heads=1,
        kv_heads=1,
        head_dim=16,
        topk=1,
        compress_ratio=4 if separate_window else 1,
    )
    inputs["indices"].fill_(2**30)
    inputs["topk_length"].fill_(1)
    actual, aux = paged_sparse_decode(**inputs, return_aux=AuxRequest(lse=True))
    expected, expected_aux = paged_sparse_decode(
        **inputs, impl="reference", return_aux=AuxRequest(lse=True)
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(aux.lse, expected_aux.lse, atol=0, rtol=0)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("permuted", [False, True])
@pytest.mark.parametrize("return_lse", [False, True])
def test_opcheck(dtype, permuted, return_lse):
    inputs = make_paged_sparse_decode_inputs(
        [40, 23], [1, 2], heads=6, kv_heads=1, head_dim=32, topk=16, dtype=dtype
    )
    query = inputs["query"]
    kv = inputs["kv"]
    key, value = kv.key_cache.contiguous(), kv.value_cache.contiguous()
    if permuted:
        query, key, value = (
            tensor.transpose(0, 1).contiguous().transpose(0, 1) for tensor in (query, key, value)
        )
    torch.library.opcheck(
        decode_with_lse_op if return_lse else decode_op,
        (
            query,
            key,
            value,
            kv.block_table,
            inputs["indices"],
            inputs["cu_seqlens_q"],
            inputs["seqused_k"],
            inputs["topk_length"],
            None,
            None,
            None,
            None,
            inputs["max_seqlen_q"],
            inputs["index_block_size"],
            inputs["window_size"],
            query.shape[-1] ** -0.5,
        ),
    )


@pytest.mark.parametrize(
    ("update", "error", "match"),
    [
        (lambda i: {"indices": i["indices"].long()}, TypeError, "indices must be int32"),
        (lambda i: {"query": i["query"][..., :-1]}, ValueError, "head_dim must match"),
        (lambda i: {"cu_seqlens_q": i["cu_seqlens_q"][:-1]}, ValueError, "cu_seqlens_q"),
        (lambda i: {"index_block_size": 0}, ValueError, "index_block_size"),
        (lambda i: {"window_size": -1}, ValueError, "window_size"),
        (lambda i: {"attention_sink": torch.zeros(3)}, ValueError, "attention_sink"),
        (lambda i: {"topk_length": i["topk_length"][:1]}, ValueError, "topk_length"),
        (lambda i: {"query": i["query"].requires_grad_()}, ValueError, "forward-only"),
        (
            lambda i: {"attention_sink": torch.zeros(i["query"].shape[1], requires_grad=True)},
            ValueError,
            "forward-only",
        ),
        (
            lambda i: {"kv": dataclasses.replace(i["kv"], block_table=i["kv"].block_table[:1])},
            ValueError,
            "block_table",
        ),
    ],
)
def test_rejects_invalid_inputs(update, error, match):
    inputs = make_paged_sparse_decode_inputs([40, 9], [1, 2], device="cpu", dtype=torch.float32)
    with pytest.raises(error, match=match):
        paged_sparse_decode(**{**inputs, **update(inputs)}, impl="reference")


@requires_cuda
def test_fused_rejects_unsupported_head_dims():
    inputs = make_paged_sparse_decode_inputs([100], [1], heads=2, kv_heads=1, head_dim=96)
    with pytest.raises(ValueError, match="power of two"):
        paged_sparse_decode(**inputs)
