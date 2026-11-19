"""Exact per-sequence sparse forwards, including training/paged numerical parity."""

import pytest
import torch

from attn_gym.sparse import (
    AuxRequest,
    PagedKV,
    gather_attn,
    lightning_indexer,
    paged_lightning_indexer,
    paged_sparse_decode,
)
from attn_gym.testing import cumulative_sequence_offsets
from attn_gym.testing.gather_attn import make_gather_attn_test_inputs, make_inputs
from attn_gym.testing.indexer import assert_paged_indexer_selection, make_paged_indexer_test_inputs
from attn_gym.testing.paged_decode import make_paged_sparse_decode_inputs

OPTIONS = {"batch_invariant": True}
CUDA = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
gpu = pytest.mark.skipif(not CUDA, reason="NVIDIA SM80+ required")
blackwell = pytest.mark.skipif(
    not CUDA or torch.cuda.get_device_capability()[0] != 10, reason="grouped backward needs SM100"
)


@gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("separate", [False, True])
def test_paged_batch_bits_and_reference(dtype, separate, no_benchmarking):
    inputs = make_paged_sparse_decode_inputs(
        [3003],
        [1],
        dtype=dtype,
        compress_ratio=4 if separate else 1,
        window_size=17 if separate else 0,
        topk_length=False,
        attention_sink=True,
    )
    expected, expected_aux = paged_sparse_decode(
        **inputs, kernel_options=OPTIONS, return_aux=AuxRequest(lse=True)
    )
    ref = paged_sparse_decode(**inputs, impl="reference")
    torch.testing.assert_close(expected, ref, atol=2e-3, rtol=2e-2)
    relative_rms = (
        expected.float() - ref.float()
    ).square().mean().sqrt() / ref.float().square().mean().sqrt()
    assert relative_rms < 0.01
    for batch in (2, 16, 17, 64):
        kwargs = dict(inputs)
        for name in ("query", "indices"):
            kwargs[name] = inputs[name].repeat(batch, *([1] * (inputs[name].ndim - 1)))
        kwargs["cu_seqlens_q"] = cumulative_sequence_offsets([1] * batch, device="cuda")
        kwargs["seqused_k"] = inputs["seqused_k"].repeat(batch)
        for name in ("kv", "window_kv"):
            source = inputs[name]
            if source is not None:
                kwargs[name] = PagedKV(
                    source.key_cache, source.value_cache, source.block_table.repeat(batch, 1)
                )
        actual, aux = paged_sparse_decode(
            **kwargs, kernel_options=OPTIONS, return_aux=AuxRequest(lse=True)
        )
        assert torch.equal(actual, expected.expand_as(actual))
        assert torch.equal(aux.lse, expected_aux.lse.expand_as(aux.lse))


@gpu
def test_paged_query_chunking_bits():
    inputs = make_paged_sparse_decode_inputs([71], [3], head_dim=32, topk=7)
    expected, ea = paged_sparse_decode(
        **inputs, kernel_options=OPTIONS, return_aux=AuxRequest(lse=True)
    )
    for row in range(3):
        single = {
            **inputs,
            **{name: inputs[name][row : row + 1] for name in ("query", "indices", "topk_length")},
            "cu_seqlens_q": cumulative_sequence_offsets([1], device="cuda"),
            "seqused_k": inputs["seqused_k"] - (2 - row),
            "max_seqlen_q": 1,
        }
        actual, aa = paged_sparse_decode(
            **single, kernel_options=OPTIONS, return_aux=AuxRequest(lse=True)
        )
        assert torch.equal(actual, expected[row : row + 1])
        assert torch.equal(aa.lse, ea.lse[row : row + 1])


@gpu
@pytest.mark.parametrize("window", [0, 11])
def test_gather_packing_padding_and_strides(window, no_benchmarking):
    inputs = make_gather_attn_test_inputs(
        batch=1,
        heads=24,
        kv_heads=2,
        head_dim=256,
        tokens=37,
        sparse_tokens=37,
        topk=39,
        separate_values=True,
        dtype=torch.bfloat16,
        device="cuda",
    )
    indices = torch.arange(39, device="cuda")[None].expand(37, -1).clone()
    indices[indices > torch.arange(37, device="cuda")[:, None]] = -1
    inputs["kv_indices"] = indices[None]
    expected, ea = gather_attn(
        **inputs,
        sliding_window_size=window,
        kernel_options=OPTIONS,
        return_aux=AuxRequest(lse=True),
    )
    for prefix in (1, 3, 16):
        padded = {}
        for name, tensor in inputs.items():
            if tensor is None or tensor.ndim != 4:
                padded[name] = tensor
            else:
                padded[name] = torch.cat((torch.zeros_like(tensor[:, :, :prefix]), tensor), 2)
        padded["kv_indices"] = torch.cat(
            (
                torch.full((prefix, 39), -1, device="cuda", dtype=indices.dtype),
                torch.where(indices >= 0, indices + prefix, -1),
            )
        )[None]
        # Global indices reproduce TorchTitan's QSA call. A local window needs document bounds.
        if window:
            padded["kv_indices"][:, prefix:] = indices
            offsets = cumulative_sequence_offsets([prefix, 37], device="cuda")
            padded.update(cu_seqlens=offsets, cu_seqlens_k=offsets)
        actual, aa = gather_attn(
            **padded,
            sliding_window_size=window,
            kernel_options=OPTIONS,
            return_aux=AuxRequest(lse=True),
        )
        assert torch.equal(actual[:, :, prefix:], expected)
        assert torch.equal(aa.lse[:, :, prefix:], ea.lse)


@blackwell
@pytest.mark.parametrize("tail", [0, 1, 3])
def test_gather_matches_paged_bits(tail):
    length = 3000 + tail
    inputs = make_paged_sparse_decode_inputs([length], [1], attention_sink=True, topk_length=False)
    kv = inputs["kv"]
    positions = torch.arange(length, device="cuda")
    pages = kv.block_table[0, positions // kv.key_cache.shape[1]].long()
    key = kv.key_cache[pages, positions % kv.key_cache.shape[1]]
    value = kv.value_cache[pages, positions % kv.value_cache.shape[1]]
    blocks = inputs["indices"][0]
    blocks = blocks[blocks >= 0]
    selected = (blocks[:, None] * 4 + torch.arange(4, device="cuda")).flatten()
    selected = torch.cat((selected, torch.arange(length // 4 * 4, length, device="cuda")))
    actual, aa = gather_attn(
        inputs["query"].transpose(0, 1)[None],
        None,
        key.transpose(0, 1)[None],
        selected[None, None],
        inputs["attention_sink"],
        sliding_window_size=0,
        sparse_v=value.transpose(0, 1)[None],
        kernel_options=OPTIONS,
        return_aux=AuxRequest(lse=True),
    )
    expected, ea = paged_sparse_decode(
        **inputs, kernel_options=OPTIONS, return_aux=AuxRequest(lse=True)
    )
    assert torch.equal(actual[0, :, 0], expected[0])
    assert torch.equal(aa.lse[0, :, 0], ea.lse[0])


@gpu
@pytest.mark.parametrize("backend", ["triton", "cute"])
@pytest.mark.parametrize("zero_weights", [False, True])
def test_indexer_batch_and_paged_whole_bits(backend, zero_weights):
    if backend == "cute" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("CuTe selector requires SM100")
    args = make_paged_indexer_test_inputs(
        (1,), (4096,), heads=4, head_dim=128, page_size=64, tail=0
    )
    q, cache, weights, table, _cu_q, used = args
    if zero_weights:
        weights.zero_()
    options = {**OPTIONS, "backend": backend}
    expected = paged_lightning_indexer(*args, topk=512, kernel_options=options)
    assert_paged_indexer_selection(args, expected, 512)
    for batch in (16, 33, 64):
        actual = paged_lightning_indexer(
            q.repeat(batch, 1, 1),
            cache,
            weights.repeat(batch, 1),
            table.repeat(batch, 1),
            cumulative_sequence_offsets([1] * batch, device="cuda"),
            used.repeat(batch),
            topk=512,
            kernel_options=options,
        )
        assert torch.equal(actual, expected.expand_as(actual))
    ids = torch.arange(1024, device="cuda")
    keys = cache[table[0, ids // 64].long(), ids % 64]
    whole = lightning_indexer(
        q[None].repeat(1, 4096, 1, 1),
        keys[None],
        weights[None].repeat(1, 4096, 1),
        512,
        compress_ratio=4,
        causal=True,
        cu_seqlens=torch.tensor([0, 4096], device="cuda", dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 1024], device="cuda", dtype=torch.int32),
        kernel_options=options,
    )
    assert torch.equal(whole[0, -1], expected[0])


@gpu
def test_indexer_packed_document_offsets():
    # The same query/key values retain the same local IDs when the candidate pool moves.
    q = torch.randn(1, 132, 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 33, 128, device="cuda", dtype=q.dtype)
    w = torch.ones(q.shape[:-1], device="cuda", dtype=q.dtype)
    expected = lightning_indexer(q, k, w, 7, compress_ratio=4, causal=True, kernel_options=OPTIONS)
    prefix = 12
    actual = lightning_indexer(
        torch.cat((q[:, :prefix], q), 1),
        torch.cat((k[:, :3], k), 1),
        torch.cat((w[:, :prefix], w), 1),
        7,
        compress_ratio=4,
        causal=True,
        cu_seqlens=cumulative_sequence_offsets([prefix, 132], device="cuda"),
        cu_seqlens_k=cumulative_sequence_offsets([3, 33], device="cuda"),
        kernel_options=OPTIONS,
    )
    assert torch.equal(actual[:, prefix:], expected)


@blackwell
def test_gather_backward_and_fullgraph(no_benchmarking):
    inputs = make_gather_attn_test_inputs(
        batch=1,
        heads=6,
        kv_heads=2,
        head_dim=32,
        tokens=17,
        sparse_tokens=11,
        topk=7,
        separate_values=True,
        dtype=torch.bfloat16,
        device="cuda",
    )
    inputs.pop("attention_sink", None)
    leaves = [
        x.requires_grad_() for x in inputs.values() if x is not None and x.is_floating_point()
    ]
    # Backward tuning follows autotune as usual; pin it so no_benchmarking can hold.
    actual = torch.compile(gather_attn, fullgraph=True)(
        **inputs, sliding_window_size=3, kernel_options={**OPTIONS, "autotune": False}
    )
    expected = gather_attn(**inputs, sliding_window_size=3, impl="reference")
    grad = torch.randn_like(actual)
    actual_grads = torch.autograd.grad(actual, leaves, grad)
    expected_grads = torch.autograd.grad(expected, leaves, grad)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e, rtol=0.06, atol=0.06)
        assert (a.float() - e.float()).norm() / e.float().norm().clamp_min(1e-6) < 0.03


@gpu
@pytest.mark.parametrize("permuted", [False, True])
def test_gather_operator_contract(permuted, no_benchmarking):
    from attn_gym.sparse.gather_attn.impl.ops import _gather_attn_bwd_op, _gather_attn_fwd_op

    inputs = make_gather_attn_test_inputs(
        heads=2,
        kv_heads=1,
        head_dim=32,
        tokens=5,
        sparse_tokens=7,
        topk=3,
        separate_values=False,
        dtype=torch.bfloat16,
        device="cuda",
    )
    if permuted:
        inputs = {
            name: x.transpose(0, -1).contiguous().transpose(0, -1) if x.ndim > 1 else x
            for name, x in inputs.items()
        }
    args = (
        inputs["query"],
        inputs["sparse_kv"],
        inputs["local_kv"],
        inputs["kv_indices"],
        inputs["attention_sink"],
        None,
        None,
        3,
        True,
        32**-0.5,
    )
    options = {**OPTIONS, "autotune": True}
    torch.library.opcheck(_gather_attn_fwd_op, (*args, True, "triton"), options)
    out, lse, state = _gather_attn_fwd_op(*args, True, "triton", **options)
    torch.library.opcheck(
        _gather_attn_bwd_op,
        (*args[:4], state, *args[4:7], out, lse, torch.randn_like(out), *args[7:], "triton"),
        # The fixed forward above never tunes; backward tunes unless pinned.
        {"autotune": False},
    )


@gpu
def test_paged_compile_graph_and_kernel_reuse(triton_compiles):
    fn = torch.compile(paged_sparse_decode, fullgraph=True, dynamic=True)
    for batch in (1, 16, 17):
        inputs = make_paged_sparse_decode_inputs([71] * batch, [1] * batch, topk=7)
        actual = fn(**inputs, kernel_options=OPTIONS)
        expected = paged_sparse_decode(**inputs, kernel_options=OPTIONS)
        assert torch.equal(actual, expected)
    # At most one kernel per address width, independent of singleton/divisibility classes.
    assert triton_compiles["_attention"] == 1
    assert triton_compiles["_paged_rows"] == 1
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        paged_sparse_decode(**inputs, kernel_options=OPTIONS)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = paged_sparse_decode(**inputs, kernel_options=OPTIONS)
    inputs["seqused_k"].sub_(3)
    inputs["kv"].block_table.copy_(inputs["kv"].block_table.flip(0))
    graph.replay()
    assert torch.equal(output, paged_sparse_decode(**inputs, kernel_options=OPTIONS))


@gpu
def test_indexer_compile_graph_and_kernel_reuse(triton_compiles):
    fn = torch.compile(paged_lightning_indexer, fullgraph=True, dynamic=True)
    options = {**OPTIONS, "backend": "triton"}
    for batch in (1, 16, 17):
        args = make_paged_indexer_test_inputs((1,) * batch, (4096,) * batch)
        actual = fn(*args, topk=7, kernel_options=options)
        assert torch.equal(actual, paged_lightning_indexer(*args, topk=7, kernel_options=options))
    assert triton_compiles["_score"] == 1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = paged_lightning_indexer(*args, topk=7, kernel_options=options)
    args[-1].sub_(1024)
    graph.replay()
    assert torch.equal(output, paged_lightning_indexer(*args, topk=7, kernel_options=options))


@gpu
@pytest.mark.parametrize("packed", [False, True])
def test_whole_indexer_pool_length_reuses_kernel(packed, triton_compiles):
    q = torch.randn(1, 140, 4, 32, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 35, 32, device="cuda", dtype=q.dtype)
    weights = torch.randn(q.shape[:-1], device="cuda", dtype=q.dtype)
    baseline = None
    for tokens in (132, 136, 140):
        offsets = {}
        if packed:
            offsets = {
                "cu_seqlens": cumulative_sequence_offsets([tokens], device="cuda"),
                "cu_seqlens_k": cumulative_sequence_offsets([tokens // 4], device="cuda"),
            }
        actual = lightning_indexer(
            q[:, :tokens],
            k[:, : tokens // 4],
            weights[:, :tokens],
            7,
            causal=True,
            compress_ratio=4,
            **offsets,
            kernel_options={**OPTIONS, "backend": "triton"},
        )
        if baseline is None:
            baseline = actual
        else:
            assert torch.equal(actual[:, :132], baseline)
    assert triton_compiles["_score"] == 1


@gpu
@pytest.mark.parametrize("permuted", [False, True])
def test_paged_opcheck_and_wide_equivalence(permuted, monkeypatch):
    from attn_gym._backends.triton import utils
    from attn_gym.sparse.paged_decode.ops import decode_with_lse_op

    inputs = make_paged_sparse_decode_inputs([37, 25], [3, 1], head_dim=32, topk=7)
    kv = inputs["kv"]
    if permuted:
        kv = PagedKV(
            kv.key_cache.movedim(1, 0).contiguous().movedim(0, 1),
            kv.value_cache.movedim(1, 0).contiguous().movedim(0, 1),
            kv.block_table.T.contiguous().T,
        )
        inputs["kv"] = kv
    args = (
        inputs["query"],
        kv.key_cache,
        kv.value_cache,
        kv.block_table,
        inputs["indices"],
        inputs["cu_seqlens_q"],
        inputs["seqused_k"],
        inputs["topk_length"],
        None,
        None,
        None,
        None,
        3,
        4,
        0,
        32**-0.5,
    )
    torch.library.opcheck(decode_with_lse_op, args, {"batch_invariant": True})
    # The raw operator handles arbitrary table strides; the public API requires unit inner stride.
    inputs["kv"] = PagedKV(kv.key_cache, kv.value_cache, kv.block_table.contiguous())
    expected, ea = paged_sparse_decode(
        **inputs, kernel_options=OPTIONS, return_aux=AuxRequest(lse=True)
    )
    monkeypatch.setattr(utils, "requires_int64_offsets", lambda *args: True)
    actual, aa = paged_sparse_decode(
        **inputs, kernel_options=OPTIONS, return_aux=AuxRequest(lse=True)
    )
    assert torch.equal(actual, expected)
    assert torch.equal(aa.lse, ea.lse)


@gpu
@pytest.mark.parametrize("permuted", [False, True])
def test_indexer_opcheck_and_noncausal(permuted):
    from attn_gym.sparse.indexer.ops import _indexer_op, _paged_indexer_op

    args = list(make_paged_indexer_test_inputs(heads=4, head_dim=16))
    if permuted:
        for i in range(4):
            args[i] = args[i].transpose(0, -1).contiguous().transpose(0, -1)
    torch.library.opcheck(
        _paged_indexer_op, (*args, 7, 4, None, "triton"), {"batch_invariant": True}
    )
    q = torch.randn(2, 17, 4, 16, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 17, 16, device="cuda", dtype=q.dtype)
    w = torch.ones(q.shape[:-1], device="cuda", dtype=q.dtype)
    torch.library.opcheck(_indexer_op, (q, k, w, 7, False, 1, "triton"), {"batch_invariant": True})
    options = {**OPTIONS, "backend": "triton"}
    actual = lightning_indexer(q, k, w, 7, kernel_options=options)
    for batch in range(2):
        expected = lightning_indexer(
            q[batch : batch + 1],
            k[batch : batch + 1],
            w[batch : batch + 1],
            7,
            kernel_options=options,
        )
        assert torch.equal(actual[batch : batch + 1], expected)


@gpu
@pytest.mark.parametrize("sink", [False, True])
def test_empty_duplicate_and_padded_selections(sink):
    inputs = make_paged_sparse_decode_inputs(
        [100, 0], [1, 2], heads=6, kv_heads=2, head_dim=32, topk=3, attention_sink=sink
    )
    inputs["indices"][0] = torch.tensor([0, 0, -1], device="cuda", dtype=torch.int32)
    inputs["topk_length"] = None
    expected, ea = paged_sparse_decode(
        **inputs, kernel_options=OPTIONS, return_aux=AuxRequest(lse=True)
    )
    ref, ra = paged_sparse_decode(**inputs, impl="reference", return_aux=AuxRequest(lse=True))
    torch.testing.assert_close(expected, ref, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(ea.lse, ra.lse, atol=2e-5, rtol=2e-5)
    inputs["indices"] = torch.nn.functional.pad(inputs["indices"], (5, 73), value=-1)
    actual, aa = paged_sparse_decode(
        **inputs, kernel_options=OPTIONS, return_aux=AuxRequest(lse=True)
    )
    assert torch.equal(actual, expected)
    assert torch.equal(aa.lse, ea.lse)
    inputs["indices"] = inputs["indices"][:, :0]
    actual, aa = paged_sparse_decode(
        **inputs, kernel_options=OPTIONS, return_aux=AuxRequest(lse=True)
    )
    assert torch.count_nonzero(actual) == 0
    if sink:
        assert torch.equal(aa.lse, inputs["attention_sink"].expand_as(aa.lse))
    else:
        assert torch.isneginf(aa.lse).all()


@gpu
@pytest.mark.parametrize("window", [0, 7])
def test_gather_rope_and_selection_padding(window, no_benchmarking):
    inputs = make_inputs(
        heads=17, latent_dim=96, rope_dim=32, seq_len=19, topk=9, local=bool(window)
    ).kwargs()
    kwargs = {"sliding_window_size": window, "scale": 0.0625, "return_aux": AuxRequest(lse=True)}
    expected, ea = gather_attn(**inputs, **kwargs, kernel_options=OPTIONS)
    ref, ra = gather_attn(**inputs, **kwargs, impl="reference")
    torch.testing.assert_close(expected, ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(ea.lse, ra.lse, atol=2e-5, rtol=2e-5)
    inputs["kv_indices"] = torch.nn.functional.pad(inputs["kv_indices"], (3, 61), value=-1)
    actual, aa = gather_attn(**inputs, **kwargs, kernel_options=OPTIONS)
    assert torch.equal(actual, expected)
    assert torch.equal(aa.lse, ea.lse)


@gpu
@pytest.mark.parametrize("dtype,dim", [(torch.float32, 256), (torch.float32, 512)])
@pytest.mark.parametrize("small_head_tile", [False, True])
def test_large_fp32_gather(dtype, dim, small_head_tile, monkeypatch):
    if small_head_tile:
        from attn_gym.sparse import _batch_invariant

        monkeypatch.setattr(_batch_invariant, "_head_tile", lambda *args: 4)
    inputs = make_gather_attn_test_inputs(
        heads=7,
        kv_heads=1,
        head_dim=dim,
        tokens=3,
        sparse_tokens=3,
        topk=2,
        separate_values=False,
        dtype=dtype,
        device="cuda",
    )
    actual = gather_attn(**inputs, sliding_window_size=0, kernel_options=OPTIONS)
    expected = gather_attn(**inputs, sliding_window_size=0, impl="reference")
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@gpu
@pytest.mark.parametrize(
    "capability,budget,dtype,dim,rope",
    [
        ((8, 0), 163840, torch.bfloat16, 512, 128),
        ((8, 6), 101376, torch.bfloat16, 512, 128),
        ((8, 6), 101376, torch.float32, 512, 0),
        ((8, 6), 101376, torch.float32, 384, 0),
        ((9, 0), 232448, torch.float32, 512, 0),
        ((10, 0), 232448, torch.float32, 512, 0),
        ((12, 0), 101376, torch.float32, 512, 0),
    ],
)
def test_attention_tiles_fit_target(capability, budget, dtype, dim, rope):
    from attn_gym.testing.triton_budget import compile_only_for_target

    inputs = make_inputs(heads=17, latent_dim=dim, rope_dim=rope, seq_len=19, topk=9, dtype=dtype)
    with compile_only_for_target(capability, budget) as launches:
        gather_attn(**inputs.kwargs(), sliding_window_size=7, scale=0.0625, kernel_options=OPTIONS)
    assert "_attention" in {launch.kernel for launch in launches}
    assert all(launch.fits(budget, 512) for launch in launches)


@gpu
def test_rejects_unsupported_fp64_and_rope():
    inputs = make_gather_attn_test_inputs(
        heads=2,
        kv_heads=1,
        head_dim=16,
        tokens=3,
        sparse_tokens=3,
        topk=1,
        separate_values=False,
        dtype=torch.float64,
        device="cuda",
    )
    with pytest.raises(ValueError, match="FP16, BF16, and FP32"):
        gather_attn(**inputs, sliding_window_size=0, kernel_options=OPTIONS)
    paired = make_inputs(heads=8, latent_dim=64, rope_dim=64, dtype=torch.float16, local=False)
    with pytest.raises(ValueError, match="shared-KV schedule"):
        gather_attn(**paired.kwargs(), sliding_window_size=0, scale=0.0625, kernel_options=OPTIONS)


@pytest.mark.parametrize("value", [1, "yes", None])
def test_batch_invariant_validation(value):
    q = torch.randn(1, 2, 4, 16)
    options = {"batch_invariant": value}
    with pytest.raises(TypeError, match="batch_invariant"):
        gather_attn(
            q,
            q,
            q,
            torch.zeros(1, 4, 1, dtype=torch.int32),
            impl="reference",
            kernel_options=options,
        )
    args = make_paged_sparse_decode_inputs([8], [1], device="cpu")
    with pytest.raises(TypeError, match="batch_invariant"):
        paged_sparse_decode(**args, impl="reference", kernel_options=options)
    args = make_paged_indexer_test_inputs(device="cpu")
    with pytest.raises(TypeError, match="batch_invariant"):
        paged_lightning_indexer(*args, topk=1, impl="reference", kernel_options=options)
    with pytest.raises(TypeError, match="batch_invariant"):
        lightning_indexer(q, q[:, :, 0], q[..., 0], 1, impl="reference", kernel_options=options)
