"""Triton grouped schedule: separate values and grouped KV heads on Blackwell."""

import contextlib

import pytest
import torch

from attn_gym.sparse.gather_attn import AuxRequest, Impl, gather_attn
from attn_gym.testing import cumulative_sequence_offsets
from attn_gym.testing.gather_attn import (
    assert_matches_low_precision_eager,
    assert_sink_gradient_fp32,
    make_gather_attn_test_inputs,
)

triton = pytest.importorskip("triton")
from attn_gym.sparse.gather_attn.impl.triton import backward, forward, shared_backward

# The grouped schedule runs only on datacenter Blackwell (SM100/SM103).
BLACKWELL_AVAILABLE = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10

pytestmark = [
    pytest.mark.skipif(not BLACKWELL_AVAILABLE, reason="Blackwell GPU required"),
    pytest.mark.usefixtures("gather_attn_single_config"),
]


def _differentiable_leaves(inputs):
    """Unique floating-point inputs, made to require grad; QSA passes one tensor as both local
    and sparse keys."""
    leaves = {}
    for name, tensor in inputs.items():
        if tensor is not None and tensor.is_floating_point():
            leaves.setdefault(id(tensor), (name, tensor.requires_grad_()))
    return list(leaves.values())


def _run(inputs, leaves, grad_output, **options):
    output, aux = gather_attn(**inputs, **options, return_aux=AuxRequest(lse=True))
    gradients = torch.autograd.grad(output, [tensor for _, tensor in leaves], grad_output)
    return output, aux.lse, gradients


@contextlib.contextmanager
def _deterministic_algorithms(enabled):
    """Set deterministic mode, which also NaN-fills fresh buffers, then restore the caller's."""
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)


def _assert_matches_reference(inputs, mode, *, grad_output=None, **options):
    """Bound Triton output, LSE, and every input gradient by bf16 reference error vs FP64.

    Triton runs in real deterministic mode for ``deterministic``, and a second run must be
    bitwise equal in both modes.
    """
    leaves = _differentiable_leaves(inputs)
    high_precision_inputs = {
        name: tensor.detach().double()
        if tensor is not None and tensor.is_floating_point()
        else tensor
        for name, tensor in inputs.items()
    }
    if inputs["local_kv"] is inputs["sparse_kv"]:
        high_precision_inputs["sparse_kv"] = high_precision_inputs["local_kv"]
    if inputs.get("local_v") is not None and inputs["local_v"] is inputs["sparse_v"]:
        high_precision_inputs["sparse_v"] = high_precision_inputs["local_v"]
    high_precision_leaves = _differentiable_leaves(high_precision_inputs)
    if grad_output is None:
        grad_output = torch.randn_like(inputs["query"])

    high_precision = _run(
        high_precision_inputs,
        high_precision_leaves,
        grad_output.double(),
        **options,
        impl=Impl.REFERENCE,
    )
    expected = _run(inputs, leaves, grad_output, **options, impl=Impl.REFERENCE)
    triton_options = {**options, "kernel_options": {"backend": "triton"}}
    with _deterministic_algorithms(mode == "deterministic"):
        actual = _run(inputs, leaves, grad_output, **triton_options)
        repeat = _run(inputs, leaves, grad_output, **triton_options)[2]

    heads, tokens, head_dim = inputs["query"].shape[1:]
    keys_per_query = inputs["kv_indices"].shape[-1] + options["sliding_window_size"]
    forward_reductions = (head_dim, keys_per_query)
    for index in range(2):
        # Without a sink, a query with no keys has an LSE of -inf on every path.
        finite = high_precision[index].isfinite()
        assert torch.equal(actual[index].isfinite(), finite)
        assert_matches_low_precision_eager(
            actual[index][finite],
            expected[index][finite],
            high_precision[index][finite],
            forward_reductions,
        )
    # KV gradients also sum over every query head and position that read the row.
    gradient_reductions = (head_dim, keys_per_query, heads * tokens)
    for (name, tensor), actual_grad, expected_grad, high_precision_grad in zip(
        leaves, actual[2], expected[2], high_precision[2], strict=True
    ):
        assert actual_grad.shape == expected_grad.shape, name
        if name == "attention_sink" and tensor.dtype != inputs["query"].dtype:
            # The bf16 output's rounding dominates an FP32 sink's gradient, so check it
            # against the kernel's own output instead of the generic bound.
            assert_sink_gradient_fp32(
                tensor.detach(), actual[1], actual[0], grad_output, actual_grad
            )
            continue
        # dsink comes from rowsum(dO * O) over the saved, rounded output.
        assert_matches_low_precision_eager(
            actual_grad,
            expected_grad,
            high_precision_grad,
            gradient_reductions,
            quantized_intermediates=2 if name == "attention_sink" else 0,
        )
    for first, second in zip(actual[2], repeat, strict=True):
        assert torch.equal(first, second)


@pytest.mark.usefixtures("nan_filled_empty")
@pytest.mark.parametrize("mode", ["default", "deterministic"])
@pytest.mark.parametrize(
    "heads,kv_heads,head_dim,separate_values,documents,window,topk",
    [
        (32, 4, 128, False, None, 19, 37),
        (24, 2, 256, True, ([11, 9, 13], [23, 21, 35]), 19, 37),
        (16, 1, 64, True, ([11, 9, 13], [23, 21, 35]), 19, 37),
        (4, 4, 64, True, None, 7, 5),
        (24, 2, 256, True, None, 0, 2051),
        (24, 2, 128, True, ([0, 11, 0, 20], [70, 0, 100, 33]), 0, 37),
        (48, 1, 256, True, ([11, 9, 13], [23, 21, 35]), 0, 37),
        (96, 2, 64, False, None, 19, 37),
        (8, 2, 96, True, ([11, 9, 13], [23, 21, 35]), 19, 37),
    ],
    ids=[
        "gqa-shared-values",
        "gqa-group12-d256-packed",
        "mqa-separate-values-packed",
        "mha-separate-values",
        "long-topk-sparse-only",
        "ragged-documents",
        "group48-d256-packed",
        "group48-d64-window",
        "gqa-d96-padded-head-tile",
    ],
)
def test_grouped_matches_reference(
    heads, kv_heads, head_dim, separate_values, documents, window, topk, mode
):
    """Forward and every gradient match for each query-head group, including masked head tiles.

    ``documents`` packs (query, key) document lengths into one batch row; the ragged case has
    empty documents, key documents spanning several dK/dV tiles with untouched tiles, and
    query positions past the last document. Groups of 48 span several head blocks per KV
    head in the forward (D=256) or the backward (D=64). Fresh buffers start as NaN.
    """
    inputs = make_gather_attn_test_inputs(
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        tokens=40,
        sparse_tokens=sum(documents[1]) if documents else 79,
        topk=topk,
        separate_values=separate_values,
        dtype=torch.bfloat16,
        batch=1 if documents else 2,
    )
    options = {"sliding_window_size": window}
    if documents:
        options["cu_seqlens"] = cumulative_sequence_offsets(documents[0])
        options["cu_seqlens_k"] = cumulative_sequence_offsets(documents[1])
    _assert_matches_reference(inputs, mode, **options)


def _qsa_indices(document_lengths, budget_blocks, generator):
    """QSA selections: up to ``budget_blocks`` distinct random complete causal 4-token blocks
    plus the query's partial tail block, as document-local indices padded with -1."""
    block = 4
    width = budget_blocks * block + block - 1
    rows = []
    for length in document_lengths:
        local = torch.arange(length)
        complete = (local + 1) // block
        scores = torch.rand(length, length // block, generator=generator)
        scores[torch.arange(length // block)[None, :] >= complete[:, None]] = -1
        chosen_scores, chosen = scores.topk(min(budget_blocks, length // block), dim=-1)
        tokens = (chosen[..., None] * block + torch.arange(block)).flatten(1)
        tokens[(chosen_scores < 0).repeat_interleave(block, dim=1)] = -1
        tail = complete[:, None] * block + torch.arange(block - 1)
        row = torch.full((length, width), -1, dtype=torch.int64)
        row[:, : tokens.shape[1]] = tokens
        row[:, budget_blocks * block :] = torch.where(tail <= local[:, None], tail, -1)
        rows.append(row)
    return torch.cat(rows)[None]


@pytest.mark.usefixtures("nan_filled_empty")
@pytest.mark.parametrize("mode", ["default", "deterministic"])
@pytest.mark.parametrize("heads,kv_heads", [(24, 2), (12, 1)], ids=["24-2", "12-1"])
def test_qsa_block_selections_match_reference(heads, kv_heads, mode):
    """QSA's regime at reduced size: D=256, one pool for local and sparse keys and values, no
    sink, window 0, and distinct 4-token blocks (top-k 131) over 1024- and 512-token documents,
    so the key-major map inverts thousands of distinct keys.
    """
    document_lengths = [1024, 512]
    inputs = make_gather_attn_test_inputs(
        heads=heads,
        kv_heads=kv_heads,
        head_dim=256,
        tokens=sum(document_lengths),
        sparse_tokens=sum(document_lengths),
        topk=1,
        separate_values=True,
        dtype=torch.bfloat16,
    )
    inputs["sparse_kv"], inputs["sparse_v"] = inputs["local_kv"], inputs["local_v"]
    inputs["attention_sink"] = None
    inputs["kv_indices"] = _qsa_indices(document_lengths, 32, torch.Generator().manual_seed(0))
    inputs["kv_indices"] = inputs["kv_indices"].cuda()
    offsets = cumulative_sequence_offsets(document_lengths)
    options = {"sliding_window_size": 0, "cu_seqlens": offsets, "cu_seqlens_k": offsets}
    _assert_matches_reference(inputs, mode, **options)


def _misaligned(tensor):
    """A contiguous copy whose data starts one element past a 16-byte boundary."""
    storage = torch.empty(tensor.numel() + 1, device=tensor.device, dtype=tensor.dtype)
    return storage[1:].view_as(tensor).copy_(tensor)


@pytest.mark.usefixtures("nan_filled_empty")
@pytest.mark.parametrize("mode", ["default", "deterministic"])
@pytest.mark.parametrize(
    "variant",
    [
        "no-sink",
        "fp32-sink",
        "aliased-pools",
        "int32-indices",
        "bshd-strides",
        "misaligned",
        "int64-offsets",
        "fused-kv-buffer",
        "mixed-layouts",
        "empty-pool",
        "no-selections",
    ],
)
def test_grouped_input_variants_match_reference(monkeypatch, variant, mode):
    """Grouped inputs QSA-style callers pass, on the gather kernels with a local window.

    ``aliased-pools`` passes one tensor as both the local and sparse keys (and values), as
    QSA does; ``misaligned`` starts the query and output gradient off a 16-byte boundary,
    which TMA gathers cannot read in place; ``int64-offsets`` forces 64-bit addressing.
    ``fused-kv-buffer`` slices keys and values from one [..., 2 * D] buffer and
    ``mixed-layouts`` keeps keys BHSD with values BSHD, so values' strides differ from the
    keys'.
    """
    from attn_gym.sparse.gather_attn.impl.triton import primitives

    documents = ([11, 9, 13], [23, 21, 35])
    inputs = make_gather_attn_test_inputs(
        heads=24,
        kv_heads=2,
        head_dim=256,
        tokens=40,
        sparse_tokens=sum(documents[1]),
        topk=37,
        separate_values=True,
        dtype=torch.bfloat16,
    )
    options = {
        "sliding_window_size": 19,
        "cu_seqlens": cumulative_sequence_offsets(documents[0]),
        "cu_seqlens_k": cumulative_sequence_offsets(documents[1]),
    }
    grad_output = torch.randn_like(inputs["query"])
    if variant == "no-sink":
        inputs["attention_sink"] = None
    elif variant == "fp32-sink":
        inputs["attention_sink"] = inputs["attention_sink"].float()
    elif variant == "aliased-pools":
        inputs["sparse_kv"], inputs["sparse_v"] = inputs["local_kv"], inputs["local_v"]
        options["cu_seqlens_k"] = options["cu_seqlens"]
    elif variant == "int32-indices":
        inputs["kv_indices"] = inputs["kv_indices"].int()
    elif variant == "bshd-strides":
        for name in ("query", "local_kv", "sparse_kv", "local_v", "sparse_v"):
            inputs[name] = inputs[name].transpose(1, 2).contiguous().transpose(1, 2)
    elif variant == "misaligned":
        inputs["query"], grad_output = _misaligned(inputs["query"]), _misaligned(grad_output)
    elif variant == "int64-offsets":
        for module in (forward, backward, primitives):
            monkeypatch.setattr(module, "requires_int64_offsets", lambda *tensors: True)
    elif variant == "fused-kv-buffer":
        for keys, values in (("local_kv", "local_v"), ("sparse_kv", "sparse_v")):
            fused = torch.cat((inputs[keys], inputs[values]), dim=-1)
            head_dim = inputs[keys].shape[-1]
            inputs[keys], inputs[values] = fused[..., :head_dim], fused[..., head_dim:]
    elif variant == "mixed-layouts":
        for name in ("local_v", "sparse_v"):
            inputs[name] = inputs[name].transpose(1, 2).contiguous().transpose(1, 2)
    elif variant == "empty-pool":
        for name in ("sparse_kv", "sparse_v"):
            inputs[name] = inputs[name][:, :, :0]
        options["cu_seqlens_k"] = torch.zeros_like(options["cu_seqlens_k"])
    else:
        inputs["kv_indices"] = inputs["kv_indices"][..., :0]
    _assert_matches_reference(inputs, mode, grad_output=grad_output, **options)


def _valid_candidates(*tuners_and_args):
    """The largest number of valid candidates among (tuner, launch arguments) pairs."""
    return max(
        len(tuner._prune(tuner._configs, args) if tuner._prune else tuner._configs)
        for tuner, args in tuners_and_args
    )


_GATHER_ARGS = {"TOPK": 64, "D": 128, "WINDOW": 19}
_TUNER_CASES = {
    "gather": _valid_candidates(
        (forward._FWD_GROUPED, _GATHER_ARGS),
        (shared_backward.BWD_DQ_GROUPED, _GATHER_ARGS),
    ),
}


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(case, marks=pytest.mark.tuner_config(index), id=f"{case}-{index}")
        for case, count in _TUNER_CASES.items()
        for index in range(count)
    ],
)
@pytest.mark.usefixtures("nan_filled_empty")
def test_every_tuner_config_matches_reference(case):
    """Every grouped tuner candidate is correct, not only the first: the ``tuner_config``
    marker pins each tuner to that index of its valid candidates (see
    ``gather_attn_single_config``), and each case runs as many indices as its largest valid
    candidate list, so shorter lists repeat but none is skipped.

    ``gather`` (D=128, a window, top-k 64) covers the gather forward, dQ, and dK/dV
    candidates.
    """
    inputs = make_gather_attn_test_inputs(
        heads=32,
        kv_heads=4,
        head_dim=128,
        tokens=40,
        sparse_tokens=79,
        topk=64,
        separate_values=True,
        dtype=torch.bfloat16,
        batch=2,
    )
    options = {"sliding_window_size": 19}
    # The candidate counts above come from these inputs' launch arguments.
    counted = _GATHER_ARGS
    assert inputs["kv_indices"].shape[-1] == counted["TOPK"]
    assert inputs["query"].shape[-1] == counted["D"]
    assert options["sliding_window_size"] == counted["WINDOW"]
    _assert_matches_reference(inputs, "default", **options)


@pytest.mark.parametrize("mode", ["default", "deterministic"])
@torch.no_grad()
def test_grouped_cuda_graph_replays_new_inputs(mode):
    """The forward captures into a CUDA graph, so it reads nothing back to the host (a host
    read fails capture), and a replay on new values and selections matches eager bitwise.

    Only the forward is captured here. Shifting every query's selections to its neighbour,
    plus a repeat in the first query, changes every query's compacted rows and counts.
    """
    inputs = make_gather_attn_test_inputs(
        heads=24,
        kv_heads=2,
        head_dim=128,
        tokens=40,
        sparse_tokens=79,
        topk=37,
        separate_values=True,
        dtype=torch.bfloat16,
        batch=2,
    )
    options = {"sliding_window_size": 19}
    options["kernel_options"] = {"backend": "triton"}

    def step():
        output, aux = gather_attn(**inputs, **options, return_aux=AuxRequest(lse=True))
        return output, aux.lse

    with _deterministic_algorithms(mode == "deterministic"):
        # Warm up (compile and tune) on a side stream, as CUDA graph capture requires.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            step()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = step()
        for tensor in inputs.values():
            if tensor is not None and tensor.is_floating_point():
                tensor.copy_(torch.randn_like(tensor))
        replay_indices = inputs["kv_indices"].roll(1, dims=1)
        replay_indices[:, 0, 1] = replay_indices[:, 0, 0]
        inputs["kv_indices"].copy_(replay_indices)
        graph.replay()
        expected = step()
    torch.cuda.synchronize()
    for actual, reference in zip(captured, expected, strict=True):
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("heads,kv_heads", [(16, 1), (4, 4), (24, 2)], ids=["mqa", "mha", "gqa"])
@pytest.mark.usefixtures("nan_filled_empty")
def test_keys_passed_as_values_run_as_shared_values(monkeypatch, heads, kv_heads):
    """``local_v=local_kv, sparse_v=sparse_kv`` reaches the backend as shared values (no
    separate value tensors), so MQA and MHA keep their shared-value kernels, and its output,
    LSE, and every gradient equal omitting the values bitwise."""
    from attn_gym.sparse.gather_attn import api

    backend_values = []
    launch = api.fused_gather_attn

    def recording_launch(*args, **kwargs):
        backend_values.append((kwargs["local_v"], kwargs["sparse_v"]))
        return launch(*args, **kwargs)

    monkeypatch.setattr(api, "fused_gather_attn", recording_launch)
    inputs = make_gather_attn_test_inputs(
        heads=heads,
        kv_heads=kv_heads,
        head_dim=64,
        tokens=40,
        sparse_tokens=79,
        topk=37,
        separate_values=False,
        dtype=torch.bfloat16,
    )
    leaves = _differentiable_leaves(inputs)
    grad_output = torch.randn_like(inputs["query"])
    options = {"sliding_window_size": 19, "kernel_options": {"backend": "triton"}}
    with _deterministic_algorithms(True):
        expected = _run(inputs, leaves, grad_output, **options)
        aliased = {**inputs, "local_v": inputs["local_kv"], "sparse_v": inputs["sparse_kv"]}
        actual = _run(aliased, leaves, grad_output, **options)
    assert backend_values == [(None, None), (None, None)]
    for value, reference in zip(
        (actual[0], actual[1], *actual[2]), (expected[0], expected[1], *expected[2]), strict=True
    ):
        assert torch.equal(value, reference)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("dense", [False, True], ids=["sparse", "dense"])
@pytest.mark.parametrize("wide_keys", [False, True], ids=["int32-keys", "int64-keys"])
def test_tile_query_map_counts_every_valid_selection(packed, dtype, dense, wide_keys):
    """Each valid selection appears in exactly one pair bit, repeats in separate pairs, after
    resolving document-local selections to global rows.

    Invalid selections (negative, past the pool, or outside the query's document, including
    for query positions past the last document) never appear. Dense selections make pairs
    of up to ``tile`` entries that straddle the map kernels' blocks. ``wide_keys`` moves some
    selections near the end of a 2**27-key pool, so their encoded sort keys need int64.
    """
    from attn_gym.sparse.gather_attn.impl.triton.key_major_backward import (
        build_tile_query_map,
    )
    from attn_gym.sparse.gather_attn.impl.triton.primitives import resolve_selected_rows

    batch, tokens, topk, tile = 2, 29, 160 if dense else 23, 32
    sparse_tokens = 2**27 if wide_keys else 150
    generator = torch.Generator(device="cuda").manual_seed(0)
    indices = torch.randint(
        -2, 90, (batch, tokens, topk), device="cuda", dtype=dtype, generator=generator
    )
    if dense:
        indices[..., 10:] = torch.arange(150, device="cuda", dtype=dtype)
    indices[:, ::2, 1] = indices[:, ::2, 0]
    indices[:, ::3, 2] = indices[:, ::3, 0]
    indices[:, 5, -1] = torch.iinfo(dtype).max
    if wide_keys:
        far = indices[..., ::4]
        indices[..., ::4] = torch.where(far >= 0, far + sparse_tokens - 160, far)
    indices = indices.transpose(1, 2).contiguous().transpose(1, 2)
    q_offsets, k_offsets = [0, 7, 7, 20, 25], [0, 40, 41, 41, sparse_tokens if wide_keys else 120]
    cu_q = torch.tensor(q_offsets, device="cuda", dtype=torch.int32) if packed else None
    cu_k = torch.tensor(k_offsets, device="cuda", dtype=torch.int32) if packed else None

    selected_rows, selected_counts = resolve_selected_rows(indices, sparse_tokens, cu_q, cu_k)
    pair_queries, pair_masks, tile_offsets = build_tile_query_map(
        selected_rows, selected_counts, sparse_tokens, tile
    )

    intervals = (
        list(zip(q_offsets, q_offsets[1:], k_offsets, k_offsets[1:]))
        if packed
        else [(0, tokens, 0, sparse_tokens)]
    )
    pair_queries, pair_masks = pair_queries.cpu(), pair_masks.cpu()
    tile_offsets, indices = tile_offsets.cpu(), indices.cpu()
    selected_rows, selected_counts = selected_rows.cpu(), selected_counts.cpu()
    assert tile_offsets.shape == (batch, triton.cdiv(sparse_tokens, tile) + 1)
    for b in range(batch):
        expected_pairs = []
        for q_start, q_end, k_start, k_end in intervals:
            for query in range(q_start, q_end):
                expected = [
                    k_start + index
                    for index in indices[b, query].tolist()
                    if 0 <= index < k_end - k_start
                ]
                # Each query's valid selections are compacted, in order, to its row's front.
                assert selected_counts[b, query].item() == len(expected)
                assert selected_rows[b, query, : len(expected)].tolist() == expected
                expected_pairs += [(key, query) for key in expected]
        actual = []
        previous_query = -1
        for tile_index in (
            (tile_offsets[b, 1:] > tile_offsets[b, :-1]).nonzero().flatten().tolist()
        ):
            pairs = range(tile_offsets[b, tile_index], tile_offsets[b, tile_index + 1])
            for pair in pairs:
                query, bits = pair_queries[b, pair].item(), pair_masks[b, pair].item()
                assert bits != 0
                assert query >= previous_query or pair == pairs.start
                previous_query = query
                actual += [
                    (tile_index * tile + bit, query) for bit in range(tile) if bits >> bit & 1
                ]
        assert sorted(actual) == sorted(expected_pairs)


@pytest.mark.parametrize(
    "separate_values", [False, True], ids=["shared-values", "separate-values"]
)
def test_unselected_keys_get_zero_gradients(separate_values):
    """Keys no query selects keep zero dK/dV even when they share a key tile with selected
    keys and hold NaN, as inactive pool capacity may. A nonfinite delta (rowsum(dO * O)) in one
    document also stays out of another document's keys in the same tile."""
    generator = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    heads, kv_heads, head_dim, tokens, pool_rows = 24, 2, 64, 64, 96
    query = randn(1, heads, tokens, head_dim)
    pools = [randn(1, kv_heads, pool_rows, head_dim) for _ in range(1 + separate_values)]
    # Documents of 36 and 28 queries over 36 and 30 keys: rows 66 onward are inactive capacity.
    # Row 33, which document 0 selects, makes that document's outputs and deltas NaN.
    for pool in pools:
        pool[:, :, 66:] = float("nan")
        pool[:, :, 33] = float("nan")
    kv_indices = torch.full((1, tokens, 2), -1, device="cuda")
    kv_indices[0, :, 0] = 0
    # Row 33 (document 0) and row 36 (document 1) share the key tile of rows 32-63; row 65
    # (document 1's local 29) shares its tile with the capacity.
    kv_indices[0, :36, 1] = 33
    kv_indices[0, 36:, 1] = 29
    leaves = [tensor.requires_grad_() for tensor in (query, *pools)]
    grad_output = randn(*query.shape)
    output = gather_attn(
        query,
        None,
        pools[0],
        kv_indices,
        sliding_window_size=0,
        sparse_v=pools[-1] if separate_values else None,
        cu_seqlens=cumulative_sequence_offsets([36, 28]),
        cu_seqlens_k=cumulative_sequence_offsets([36, 30]),
        kernel_options={"backend": "triton"},
    )
    pool_gradients = torch.autograd.grad(output, leaves, grad_output)[1:]
    unselected = torch.ones(pool_rows, dtype=torch.bool, device="cuda")
    unselected[[0, 33, 36, 65]] = False
    for gradient in pool_gradients:
        assert torch.equal(
            gradient[:, :, unselected], torch.zeros_like(gradient[:, :, unselected])
        )
        # Only document 1's queries, whose deltas are finite, select rows 36 and 65.
        assert gradient[:, :, [36, 65]].isfinite().all()


def test_resolver_reuses_one_binary_across_shapes(triton_compiles):
    """Sequence lengths, pool lengths, document counts, batch strides, and top-k widths reuse
    one resolver binary across Triton's == 1 and multiple-of-16 classes.

    Top-k widths stay off both classes because the query stride, kept specialized, equals the
    top-k width of contiguous indices.
    """
    from attn_gym.sparse.gather_attn.impl.triton.primitives import resolve_selected_rows

    for batch, documents, topk in (
        (1, [1], 23),
        (2, [8, 8], 37),
        (1, [3] * 16, 131),
        (3, [5, 0, 24], 2051),
    ):
        tokens = sum(documents)
        indices = torch.randint(
            -1, tokens, (batch, tokens, topk), device="cuda", dtype=torch.int32
        )
        offsets = cumulative_sequence_offsets(documents)
        resolve_selected_rows(indices, tokens, offsets, offsets)
    assert triton_compiles["_resolve_selected_rows"] == 1


@pytest.mark.parametrize(
    "dtype,heads,head_dim,window,expected",
    [
        (torch.bfloat16, 64, 512, 128, "shared"),
        (torch.bfloat16, 4, 64, 7, "tma"),
        (torch.float32, 4, 64, 7, "tma"),
    ],
    ids=["shared-mqa", "mha-bf16", "mha-fp32"],
)
def test_existing_inputs_keep_their_schedule(
    monkeypatch, dtype, heads, head_dim, window, expected
):
    """Shared-value MQA and MHA inputs keep their previous forward schedule, and neither
    grouped launcher runs in their forward and backward."""
    from attn_gym.sparse.gather_attn.impl.triton import backward, forward

    def grouped_launch(*args, **kwargs):
        raise AssertionError("a shared-value input took the grouped schedule")

    monkeypatch.setattr(forward, "_launch_grouped_forward", grouped_launch)
    monkeypatch.setattr(backward, "_launch_grouped_backward", grouped_launch)
    schedules = []
    schedule = forward.forward_schedule

    def recording_schedule(*args, **kwargs):
        schedules.append(schedule(*args, **kwargs))
        return schedules[-1]

    monkeypatch.setattr(forward, "forward_schedule", recording_schedule)
    kv_heads = 1 if heads == 64 else heads
    inputs = make_gather_attn_test_inputs(
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        tokens=8,
        sparse_tokens=8,
        topk=2,
        separate_values=False,
        dtype=dtype,
    )
    leaves = _differentiable_leaves(inputs)
    _run(
        inputs,
        leaves,
        torch.randn_like(inputs["query"]),
        sliding_window_size=window,
        kernel_options={"backend": "triton"},
    )
    assert schedules == [expected]


@pytest.mark.parametrize(
    "dtype,head_dim,window",
    [(torch.float32, 256, 0), (torch.bfloat16, 512, 0), (torch.bfloat16, 256, 4096)],
    ids=["fp32", "d512", "wide-window"],
)
def test_ineligible_grouped_inputs_raise(dtype, head_dim, window):
    """Inputs the grouped schedule cannot run raise instead of ignoring values or groups."""
    inputs = make_gather_attn_test_inputs(
        heads=24,
        kv_heads=2,
        head_dim=head_dim,
        tokens=8,
        sparse_tokens=8,
        topk=2,
        separate_values=True,
        dtype=dtype,
    )
    with pytest.raises(ValueError, match="supports separate values and grouped KV heads only"):
        gather_attn(**inputs, sliding_window_size=window, kernel_options={"backend": "triton"})


@pytest.mark.usefixtures("fresh_compile_cache")
@pytest.mark.parametrize("case", ["separate-values", "grouped-shared-values", "one-pool"])
def test_grouped_fullgraph(case):
    """Grouped inputs trace through fullgraph compile, forward and backward, and match eager
    bitwise, including the LSE: separate values, grouped heads with shared values (no dV),
    and QSA's packed call with one pool for local and sparse keys (and values) and one offsets
    tensor for both partitions."""
    if case == "one-pool":
        inputs = make_gather_attn_test_inputs(
            heads=24,
            kv_heads=2,
            head_dim=64,
            tokens=16,
            sparse_tokens=16,
            topk=4,
            separate_values=True,
            dtype=torch.bfloat16,
        )
        inputs["sparse_kv"], inputs["sparse_v"] = inputs["local_kv"], inputs["local_v"]
        # QSA packs documents and passes one offsets tensor for queries and keys.
        offsets = cumulative_sequence_offsets([9, 7])
        options = {"sliding_window_size": 3, "cu_seqlens": offsets, "cu_seqlens_k": offsets}
    else:
        inputs = make_gather_attn_test_inputs(
            heads=24,
            kv_heads=2,
            head_dim=64,
            tokens=16,
            sparse_tokens=16,
            topk=4,
            separate_values=case == "separate-values",
            dtype=torch.bfloat16,
        )
        options = {"sliding_window_size": 3}
    options["kernel_options"] = {"backend": "triton"}
    leaves = _differentiable_leaves(inputs)
    grad_output = torch.randn_like(inputs["query"])
    with _deterministic_algorithms(True):
        expected = _run(inputs, leaves, grad_output, **options)
        compiled = torch.compile(gather_attn, fullgraph=True)
        output, aux = compiled(**inputs, **options, return_aux=AuxRequest(lse=True))
        gradients = torch.autograd.grad(output, [tensor for _, tensor in leaves], grad_output)
    for actual, reference in zip(
        (output, aux.lse, *gradients), (expected[0], expected[1], *expected[2]), strict=True
    ):
        assert torch.equal(actual, reference)


@pytest.mark.usefixtures("fresh_compile_cache")
def test_grouped_dynamic_shapes_compile_once():
    """A dynamic-shape compile serves grouped, separate-value inputs of three sequence lengths
    with one graph, and each matches eager bitwise."""
    torch._dynamo.reset()
    compiled = torch.compile(gather_attn, dynamic=True, fullgraph=True)
    frames = []
    for tokens in (40, 56, 72):
        inputs = make_gather_attn_test_inputs(
            heads=24,
            kv_heads=2,
            head_dim=64,
            tokens=tokens,
            sparse_tokens=tokens,
            topk=8,
            separate_values=True,
            dtype=torch.bfloat16,
        )
        leaves = _differentiable_leaves(inputs)
        grad_output = torch.randn_like(inputs["query"])
        options = {"sliding_window_size": 5, "kernel_options": {"backend": "triton"}}
        with _deterministic_algorithms(True):
            expected = _run(inputs, leaves, grad_output, **options)
            output, aux = compiled(**inputs, **options, return_aux=AuxRequest(lse=True))
            gradients = torch.autograd.grad(output, [tensor for _, tensor in leaves], grad_output)
        frames.append(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        for actual, reference in zip(
            (output, aux.lse, *gradients), (expected[0], expected[1], *expected[2]), strict=True
        ):
            assert torch.equal(actual, reference)
    assert frames[0] == frames[-1], frames


@pytest.mark.parametrize("mode", ["default", "deterministic"])
def test_grouped_matches_on_a_side_stream(mode):
    """Forward and backward on a busy non-default stream equal the default-stream run bitwise.

    A sleep queued first keeps the side stream's kernels pending while the host queues the
    rest of the step, so every kernel and buffer must follow the current stream. The warm-up
    selects nothing to gather, so buffers it recycles hold stale values for the timed runs.
    """
    documents = ([11, 9, 13], [23, 21, 35])
    inputs = make_gather_attn_test_inputs(
        heads=24,
        kv_heads=2,
        head_dim=256,
        tokens=sum(documents[0]),
        sparse_tokens=sum(documents[1]),
        topk=16,
        separate_values=True,
        dtype=torch.bfloat16,
    )
    options = {
        "sliding_window_size": 0,
        "cu_seqlens": cumulative_sequence_offsets(documents[0]),
        "cu_seqlens_k": cumulative_sequence_offsets(documents[1]),
        "kernel_options": {"backend": "triton"},
    }
    leaves = _differentiable_leaves(inputs)
    grad_output = torch.randn_like(inputs["query"])
    with _deterministic_algorithms(mode == "deterministic"):
        # Compile and tune first, so the timed runs only launch.
        selections = inputs["kv_indices"]
        inputs["kv_indices"] = torch.full_like(selections, -1)
        _run(inputs, leaves, grad_output, **options)
        inputs["kv_indices"] = selections
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            torch.cuda._sleep(10**8)
            actual = _run(inputs, leaves, grad_output, **options)
        torch.cuda.current_stream().wait_stream(stream)
        expected = _run(inputs, leaves, grad_output, **options)
    torch.cuda.synchronize()
    for value, reference in zip(
        (actual[0], actual[1], *actual[2]), (expected[0], expected[1], *expected[2]), strict=True
    ):
        assert torch.equal(value, reference)


@pytest.mark.parametrize(
    "case", ["gqa-shared-values", "gqa-separate-values", "mqa-separate-values"]
)
def test_grouped_operator_contracts(case):
    """The opaque operators' schema, fake implementations, autograd registration, and compiled
    dispatch hold for grouped heads and separate values, including shared values' absent dV."""
    from attn_gym.sparse.gather_attn.impl.ops import _gather_attn_bwd_op, _gather_attn_fwd_op

    inputs = make_gather_attn_test_inputs(
        heads=8,
        kv_heads=1 if case.startswith("mqa") else 2,
        head_dim=64,
        tokens=20,
        sparse_tokens=22,
        topk=5,
        separate_values=case.endswith("separate-values"),
        dtype=torch.bfloat16,
    )
    offsets, window = (None, None), 7
    query, sparse_kv, local_kv = inputs["query"], inputs["sparse_kv"], inputs["local_kv"]
    kv_indices, sink = inputs["kv_indices"], inputs["attention_sink"]
    values = (inputs.get("sparse_v"), inputs.get("local_v"))
    common = (*offsets, window, sparse_kv.shape[1] == 1, 0.25)
    trailing = ("triton", True, None, None, None, *values)
    for needs_backward in (False, True):
        torch.library.opcheck(
            _gather_attn_fwd_op,
            (query, sparse_kv, local_kv, kv_indices, sink, *common, needs_backward, *trailing),
        )
    output, lse, state = _gather_attn_fwd_op(
        query, sparse_kv, local_kv, kv_indices, sink, *common, True, *trailing
    )
    torch.library.opcheck(
        _gather_attn_bwd_op,
        (
            query,
            sparse_kv,
            local_kv,
            kv_indices,
            state,
            sink,
            *common[:2],
            output,
            lse,
            torch.randn_like(output),
            *common[2:],
            *trailing,
        ),
    )
