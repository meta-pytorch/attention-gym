"""Triton grouped schedule: separate values and grouped KV heads on Blackwell."""

import contextlib

import pytest
import torch
import triton

from attn_gym.sparse.gather_attn import AuxRequest, Impl, gather_attn
from attn_gym.sparse.gather_attn.impl.triton import forward
from attn_gym.testing import cumulative_sequence_offsets
from attn_gym.testing.gather_attn import (
    assert_matches_low_precision_eager,
    make_gather_attn_test_inputs,
)

# The grouped schedule runs only on datacenter Blackwell (SM100/SM103).
BLACKWELL_AVAILABLE = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10

pytestmark = [
    pytest.mark.skipif(not BLACKWELL_AVAILABLE, reason="Blackwell GPU required"),
    pytest.mark.usefixtures("gather_attn_single_config"),
]


def _forward(inputs, **options):
    output, aux = gather_attn(**inputs, **options, return_aux=AuxRequest(lse=True))
    return output, aux.lse


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


@torch.no_grad()
def _assert_matches_reference(inputs, mode, **options):
    """Bound Triton output and LSE by bf16 reference error vs FP64.

    Triton runs in real deterministic mode for ``deterministic``, and a second run must be
    bitwise equal in both modes.
    """
    high_precision_inputs = {
        name: tensor.double() if tensor is not None and tensor.is_floating_point() else tensor
        for name, tensor in inputs.items()
    }
    high_precision = _forward(high_precision_inputs, **options, impl=Impl.REFERENCE)
    expected = _forward(inputs, **options, impl=Impl.REFERENCE)
    triton_options = {**options, "kernel_options": {"backend": "triton"}}
    with _deterministic_algorithms(mode == "deterministic"):
        actual = _forward(inputs, **triton_options)
        repeat = _forward(inputs, **triton_options)

    head_dim = inputs["query"].shape[-1]
    keys_per_query = inputs["kv_indices"].shape[-1] + options["sliding_window_size"]
    for index in range(2):
        # Without a sink, a query with no keys has an LSE of -inf on every path.
        finite = high_precision[index].isfinite()
        assert torch.equal(actual[index].isfinite(), finite)
        assert_matches_low_precision_eager(
            actual[index][finite],
            expected[index][finite],
            high_precision[index][finite],
            (head_dim, keys_per_query),
        )
        assert torch.equal(actual[index], repeat[index])


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
    """The forward matches for each query-head group, including masked head tiles.

    ``documents`` packs (query, key) document lengths into one batch row; the ragged case has
    empty documents and query positions past the last document. Groups of 48 span several
    head blocks per KV head (D=256 and D=64). Fresh buffers start as NaN.
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


def _dense_tile_inputs(heads, kv_heads, head_dim, separate_values, packed, all_dense, spans):
    """Distinct document-local selections, -1 padded, so tiles can take the dense path.

    Unless ``all_dense``, a repeat and tiles straddling documents keep some tiles on the
    gather kernels, with empty and partly padded queries inside dense tiles. ``long`` spans
    (top-k 48 over keys of up to 170) cover several K/V tiles and bitmap words per dense tile.
    """
    topk = 48 if spans == "long" else 16
    tile_queries = 128 // (heads // kv_heads)
    # With all_dense, documents of whole tiles keep every tile inside one document.
    if packed:
        query_lengths = [tile_queries * 2, tile_queries] if all_dense else [11, 9, 13]
        short_keys, long_keys = (
            ([23, 21], [150, 170]) if all_dense else ([23, 21, 35], [150, 130, 170])
        )
    else:
        query_lengths = [tile_queries * 3] if all_dense else [33]
        short_keys, long_keys = [40], [170]
    key_lengths = long_keys if spans == "long" else short_keys
    batch = 1 if packed else 2
    query_documents = [d for d, length in enumerate(query_lengths) for _ in range(length)]
    inputs = make_gather_attn_test_inputs(
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        tokens=len(query_documents),
        sparse_tokens=sum(key_lengths),
        topk=topk,
        separate_values=separate_values,
        dtype=torch.bfloat16,
        batch=batch,
    )
    generator = torch.Generator().manual_seed(0)
    indices = torch.full((batch, len(query_documents), topk), -1, dtype=torch.int64)
    for b in range(batch):
        for position, document in enumerate(query_documents):
            chosen = torch.randperm(key_lengths[document], generator=generator)[:topk]
            indices[b, position, : chosen.numel()] = chosen
    indices[:, 1::4, -3:] = -1
    if not all_dense:
        indices[:, 25] = -1
        # A repeat makes its query, and so its tile, gather: the last batch row for MHA,
        # whose 128-query tiles span the whole sequence.
        indices[-1, 12, 1] = indices[-1, 12, 0]
    inputs["kv_indices"] = indices.cuda()
    options = {"sliding_window_size": 0}
    if packed:
        options["cu_seqlens"] = cumulative_sequence_offsets(query_lengths)
        options["cu_seqlens_k"] = cumulative_sequence_offsets(key_lengths)
    return inputs, options


def _spy_dense_launches(monkeypatch):
    """Count launches of the dense-tile forward kernel."""
    launches = []
    tuner = forward._FWD_DENSE

    class CountingLaunch:
        def __getitem__(self, grid):
            launches.append(grid)
            return tuner[grid]

    monkeypatch.setattr(forward, "_FWD_DENSE", CountingLaunch())
    return launches


def _planned_tile_spans(inputs, options):
    """The dense-tile plan for window-0 inputs: ``tile_spans`` [B, tiles, 2], ``hi`` -1 for
    tiles left to the gather kernels."""
    from attn_gym.sparse.gather_attn.impl.triton.primitives import (
        dense_tile_shape,
        plan_dense_tiles,
        resolve_selected_rows,
    )

    heads, kv_heads = inputs["query"].shape[1], inputs["sparse_kv"].shape[1]
    tile_queries, mask_words = dense_tile_shape(heads, kv_heads, inputs["kv_indices"].shape[-1], 0)
    _, counts, _, selection_spans = resolve_selected_rows(
        inputs["kv_indices"],
        inputs["sparse_kv"].shape[2],
        options.get("cu_seqlens"),
        options.get("cu_seqlens_k"),
        mask_words=mask_words,
    )
    return plan_dense_tiles(counts, selection_spans, options.get("cu_seqlens"), tile_queries)


@pytest.mark.usefixtures("nan_filled_empty")
@pytest.mark.parametrize("mode", ["default", "deterministic"])
@pytest.mark.parametrize(
    "heads,kv_heads,head_dim,separate_values,packed",
    [
        (24, 2, 256, True, True),
        (32, 4, 128, False, False),
        (16, 1, 64, True, False),
        (4, 4, 64, True, False),
        (6, 2, 192, True, False),
        (48, 1, 64, True, False),
    ],
    ids=[
        "gqa-group12-d256-packed",
        "gqa-shared-values",
        "mqa-separate-values",
        "mha",
        "gqa-group3-d192",
        "mqa-group48",
    ],
)
@pytest.mark.parametrize("all_dense", [False, True], ids=["mixed", "all-dense"])
@pytest.mark.parametrize("spans", ["short", "long"])
def test_dense_tiles_match_reference(
    monkeypatch, heads, kv_heads, head_dim, separate_values, packed, mode, all_dense, spans
):
    """Dense tiles walk their span under selection bitmaps; gather tiles keep the rest.

    With ``all_dense`` every tile is dense, so the gather kernel skips every query. Groups 3
    and 48 leave part of each 128-row tile empty; D=192 pads the head tile. The dense kernel
    must run.
    """
    launches = _spy_dense_launches(monkeypatch)
    inputs, options = _dense_tile_inputs(
        heads, kv_heads, head_dim, separate_values, packed, all_dense, spans
    )
    tile_spans = _planned_tile_spans(inputs, options)
    dense = tile_spans[..., 1] >= 0
    assert dense.all() if all_dense else dense.any() and not dense.all()
    if spans == "long":
        # Some dense tile walks several 64-key tiles and reads past its first two words.
        assert (tile_spans[..., 1][dense] >= 64).any()
    _assert_matches_reference(inputs, mode, **options)
    assert launches


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
def test_qsa_block_selections_match_reference(monkeypatch, heads, kv_heads, mode):
    """QSA's regime at reduced size: D=256, one pool for local and sparse keys and values, no
    sink, window 0, and distinct 4-token blocks (top-k 131) over 1024- and 512-token documents.

    Early queries' tiles go dense over 20-word bitmaps; later queries select past the bitmap
    and gather.
    """
    launches = _spy_dense_launches(monkeypatch)
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
    dense = _planned_tile_spans(inputs, options)[..., 1] >= 0
    assert dense.any() and not dense.all()
    _assert_matches_reference(inputs, mode, **options)
    assert launches


@pytest.mark.usefixtures("nan_filled_empty")
@pytest.mark.parametrize("mode", ["default", "deterministic"])
@pytest.mark.parametrize(
    "base,variant",
    [
        (base, variant)
        for base in ("gather", "dense")
        for variant in (
            "no-sink",
            "fp32-sink",
            "aliased-pools",
            "int32-indices",
            "bshd-strides",
            "int64-offsets",
            "fused-kv-buffer",
            "mixed-layouts",
        )
    ]
    + [("gather", "empty-pool"), ("gather", "no-selections")],
)
def test_grouped_input_variants_match_reference(monkeypatch, base, variant, mode):
    """Grouped inputs QSA-style callers pass, on the gather kernel (with a local window) and
    on mixed dense and gather tiles.

    ``aliased-pools`` passes one tensor as both the local and sparse keys (and values), as
    QSA does; ``int64-offsets`` forces 64-bit addressing. ``fused-kv-buffer`` slices keys
    and values from one [..., 2 * D] buffer and ``mixed-layouts`` keeps keys BHSD with values
    BSHD, so values' strides differ from the keys'. Every ``dense`` variant still plans dense
    tiles next to gather tiles.
    """
    from attn_gym.sparse.gather_attn.impl.triton import primitives

    launches = _spy_dense_launches(monkeypatch)
    if base == "dense":
        inputs, options = _dense_tile_inputs(24, 2, 256, True, True, False, "short")
    else:
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
    elif variant == "int64-offsets":
        for module in (forward, primitives):
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
    if base == "dense":
        dense = _planned_tile_spans(inputs, options)[..., 1] >= 0
        assert dense.any() and not dense.all()
    _assert_matches_reference(inputs, mode, **options)
    assert bool(launches) == (base == "dense")


def _valid_candidates(*tuners_and_args):
    """The largest number of valid candidates among (tuner, launch arguments) pairs."""
    return max(
        len(tuner._prune(tuner._configs, args) if tuner._prune else tuner._configs)
        for tuner, args in tuners_and_args
    )


_GATHER_ARGS = {"TOPK": 64, "D": 128, "WINDOW": 19}
_DENSE_ARGS = {"TOPK": 48, "D": 256, "WINDOW": 0}
_TUNER_CASES = {
    "gather": _valid_candidates((forward._FWD_GROUPED, _GATHER_ARGS)),
    "dense": _valid_candidates(
        (forward._FWD_DENSE, _DENSE_ARGS),
        (forward._FWD_GROUPED, _DENSE_ARGS),
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

    ``gather`` (D=128, a window, top-k 64) covers the gather forward candidates, and
    ``dense`` (D=256) the dense-tile kernel's and the D=256 gather candidates.
    """
    if case == "dense":
        inputs, options = _dense_tile_inputs(24, 2, 256, True, True, False, "long")
    else:
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
    counted = _DENSE_ARGS if case == "dense" else _GATHER_ARGS
    assert inputs["kv_indices"].shape[-1] == counted["TOPK"]
    assert inputs["query"].shape[-1] == counted["D"]
    assert options["sliding_window_size"] == counted["WINDOW"]
    _assert_matches_reference(inputs, "default", **options)


@pytest.mark.parametrize("mode", ["default", "deterministic"])
@pytest.mark.parametrize("base", ["dense", "window"])
@torch.no_grad()
def test_grouped_cuda_graph_replays_new_inputs(base, mode):
    """The forward captures into a CUDA graph, so it reads nothing back to the host (a host
    read fails capture), and a replay on new values and selections matches eager bitwise.

    Shifting every query's selections to its neighbour, plus a repeat in the first query,
    changes which tiles are dense, so nothing sized at capture may depend on them.
    """
    if base == "dense":
        inputs, options = _dense_tile_inputs(24, 2, 256, True, True, False, "short")
    else:
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
    with _deterministic_algorithms(mode == "deterministic"):
        # Warm up (compile and tune) on a side stream, as CUDA graph capture requires.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _forward(inputs, **options)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _forward(inputs, **options)
        for tensor in inputs.values():
            if tensor.is_floating_point():
                tensor.copy_(torch.randn_like(tensor))
        replay_indices = inputs["kv_indices"].roll(1, dims=1)
        replay_indices[:, 0, 1] = replay_indices[:, 0, 0]
        if base == "dense":
            captured_plan = _planned_tile_spans(inputs, options)[..., 1] >= 0
        inputs["kv_indices"].copy_(replay_indices)
        if base == "dense":
            assert not torch.equal(
                captured_plan, _planned_tile_spans(inputs, options)[..., 1] >= 0
            )
        graph.replay()
        expected = _forward(inputs, **options)
    torch.cuda.synchronize()
    for actual, reference in zip(captured, expected, strict=True):
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("heads,kv_heads", [(16, 1), (4, 4), (24, 2)], ids=["mqa", "mha", "gqa"])
@pytest.mark.usefixtures("nan_filled_empty")
@torch.no_grad()
def test_keys_passed_as_values_run_as_shared_values(monkeypatch, heads, kv_heads):
    """``local_v=local_kv, sparse_v=sparse_kv`` reaches the backend as shared values (no
    separate value tensors), so MQA and MHA keep their shared-value kernels, and its output
    and LSE equal omitting the values bitwise."""
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
    options = {"sliding_window_size": 19, "kernel_options": {"backend": "triton"}}
    expected = _forward(inputs, **options)
    aliased = {**inputs, "local_v": inputs["local_kv"], "sparse_v": inputs["sparse_kv"]}
    actual = _forward(aliased, **options)
    assert backend_values == [(None, None), (None, None)]
    for value, reference in zip(actual, expected, strict=True):
        assert torch.equal(value, reference)


@pytest.mark.parametrize("packed", [False, True])
def test_dense_tile_plan_matches_recount(packed):
    """Selection bitmaps, spans, the bitmap-exact flag, and the dense-tile plan match a Python
    recount of their definitions."""
    from attn_gym.sparse.gather_attn.impl.triton.primitives import (
        DENSE_SPAN_RATIO,
        dense_tile_shape,
        plan_dense_tiles,
        resolve_selected_rows,
    )

    inputs, options = _dense_tile_inputs(24, 2, 256, True, packed, False, "long")
    kv_indices = inputs["kv_indices"]
    batch, tokens, topk = kv_indices.shape
    pool = inputs["sparse_kv"].shape[2]
    cu_seqlens, cu_seqlens_k = options.get("cu_seqlens"), options.get("cu_seqlens_k")
    tile_queries, mask_words = dense_tile_shape(24, 2, topk, 0)
    _, counts, masks, spans = resolve_selected_rows(
        kv_indices, pool, cu_seqlens, cu_seqlens_k, mask_words=mask_words
    )
    tile_spans = plan_dense_tiles(counts, spans, cu_seqlens, tile_queries)

    bits = 32 * mask_words
    query_offsets = cu_seqlens.tolist() if packed else [0, tokens]
    key_offsets = cu_seqlens_k.tolist() if packed else [0, pool]
    num_tiles = triton.cdiv(tokens, tile_queries)
    masks, spans, tile_spans = masks.cpu(), spans.cpu(), tile_spans.cpu()
    for b in range(batch):
        query_spans = []
        for query in range(tokens):
            document = sum(offset <= query for offset in query_offsets[1:])
            key_count = key_offsets[document + 1] - key_offsets[document]
            local = [i for i in kv_indices[b, query].tolist() if 0 <= i < key_count]
            words = [0] * mask_words
            for index in local:
                if index < bits:
                    words[index // 32] |= 1 << index % 32
            assert [word & 0xFFFFFFFF for word in masks[b, query].tolist()] == words
            exact = len(set(local)) == len(local) and all(index < bits for index in local)
            lo, hi = min([bits, *local]), max(local, default=-1)
            assert spans[b, query].tolist() == [lo, hi if exact else -2]
            query_spans.append((document, len(local), lo, hi if exact else -2))
        for tile in range(num_tiles):
            members = query_spans[tile * tile_queries : (tile + 1) * tile_queries]
            selected = sum(count for _, count, _, _ in members)
            span_lo = min((lo for _, count, lo, _ in members if count), default=2**30)
            span_hi = max((hi for _, count, _, hi in members if count), default=-1)
            dense = (
                all(hi >= -1 for _, _, _, hi in members)
                and selected > 0
                and (span_hi - span_lo + 1) * len(members) <= DENSE_SPAN_RATIO * selected
                and (not packed or members[0][0] == members[-1][0])
            )
            assert tile_spans[b, tile].tolist() == [span_lo, span_hi if dense else -1]
    assert 0 < (tile_spans[..., 1] >= 0).sum() < tile_spans[..., 1].numel()


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_resolved_rows_match_recount(packed, dtype):
    """Each query's valid selections are compacted, in order and with repeats, to its row's
    front as global pool rows.

    Invalid selections (negative, past the pool, or outside the query's document, including
    for query positions past the last document) never appear.
    """
    from attn_gym.sparse.gather_attn.impl.triton.primitives import resolve_selected_rows

    batch, tokens, topk, sparse_tokens = 2, 29, 23, 150
    generator = torch.Generator(device="cuda").manual_seed(0)
    indices = torch.randint(
        -2, 90, (batch, tokens, topk), device="cuda", dtype=dtype, generator=generator
    )
    indices[:, ::2, 1] = indices[:, ::2, 0]
    indices[:, 5, -1] = torch.iinfo(dtype).max
    indices = indices.transpose(1, 2).contiguous().transpose(1, 2)
    q_offsets, k_offsets = [0, 7, 7, 20, 25], [0, 40, 41, 41, 120]
    cu_q = torch.tensor(q_offsets, device="cuda", dtype=torch.int32) if packed else None
    cu_k = torch.tensor(k_offsets, device="cuda", dtype=torch.int32) if packed else None

    selected_rows, selected_counts, _, _ = resolve_selected_rows(
        indices, sparse_tokens, cu_q, cu_k
    )

    intervals = (
        list(zip(q_offsets, q_offsets[1:], k_offsets, k_offsets[1:]))
        if packed
        else [(0, tokens, 0, sparse_tokens)]
    )
    indices, selected_rows, selected_counts = (
        indices.cpu(),
        selected_rows.cpu(),
        selected_counts.cpu(),
    )
    covered = set()
    for b in range(batch):
        for q_start, q_end, k_start, k_end in intervals:
            for query in range(q_start, q_end):
                expected = [
                    k_start + index
                    for index in indices[b, query].tolist()
                    if 0 <= index < k_end - k_start
                ]
                assert selected_counts[b, query].item() == len(expected)
                assert selected_rows[b, query, : len(expected)].tolist() == expected
                covered.add(query)
        # Queries past the last document select nothing.
        for query in set(range(tokens)) - covered:
            assert selected_counts[b, query].item() == 0


@pytest.mark.parametrize(
    "dtype,heads,head_dim,window,expected",
    [
        (torch.bfloat16, 64, 512, 128, "shared"),
        (torch.bfloat16, 4, 64, 7, "tma"),
        (torch.float32, 4, 64, 7, "tma"),
    ],
    ids=["shared-mqa", "mha-bf16", "mha-fp32"],
)
@torch.no_grad()
def test_existing_inputs_keep_their_schedule(
    monkeypatch, dtype, heads, head_dim, window, expected
):
    """Shared-value MQA and MHA inputs keep their previous forward schedule, and the grouped
    launcher never runs for them."""

    def grouped_launch(*args, **kwargs):
        raise AssertionError("a shared-value input took the grouped schedule")

    monkeypatch.setattr(forward, "_launch_grouped_forward", grouped_launch)
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
    _forward(inputs, sliding_window_size=window, kernel_options={"backend": "triton"})
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
@pytest.mark.parametrize(
    "case", ["separate-values", "grouped-shared-values", "dense-tiles", "one-pool"]
)
@torch.no_grad()
def test_grouped_fullgraph(case):
    """Grouped inputs trace through fullgraph compile and match eager bitwise, including the
    LSE: separate values, grouped heads, dense tiles, and QSA's packed call with one pool for
    local and sparse keys (and values) and one offsets tensor for both partitions."""
    if case == "dense-tiles":
        inputs, options = _dense_tile_inputs(24, 2, 256, True, True, False, "short")
    elif case == "one-pool":
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
    expected = _forward(inputs, **options)
    compiled = torch.compile(gather_attn, fullgraph=True)
    output, aux = compiled(**inputs, **options, return_aux=AuxRequest(lse=True))
    for actual, reference in zip((output, aux.lse), expected, strict=True):
        assert torch.equal(actual, reference)


@pytest.mark.usefixtures("fresh_compile_cache")
@torch.no_grad()
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
        options = {"sliding_window_size": 5, "kernel_options": {"backend": "triton"}}
        expected = _forward(inputs, **options)
        output, aux = compiled(**inputs, **options, return_aux=AuxRequest(lse=True))
        frames.append(torch._dynamo.utils.counters["stats"]["unique_graphs"])
        for actual, reference in zip((output, aux.lse), expected, strict=True):
            assert torch.equal(actual, reference)
    assert frames[0] == frames[-1], frames


@pytest.mark.parametrize(
    "case", ["gqa-shared-values", "gqa-separate-values", "mqa-separate-values", "packed-dense"]
)
def test_grouped_operator_contracts(case):
    """The forward operator's schema, fake implementation, and compiled dispatch hold for
    grouped heads and separate values, and for packed window-0 inputs with dense tiles."""
    from attn_gym.sparse.gather_attn.impl.ops import _gather_attn_fwd_op

    if case == "packed-dense":
        inputs, options = _dense_tile_inputs(24, 2, 256, True, True, False, "short")
        offsets = (options["cu_seqlens"], options["cu_seqlens_k"])
        window = 0
    else:
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
    torch.library.opcheck(
        _gather_attn_fwd_op,
        (query, sparse_kv, local_kv, kv_indices, sink, *common, False, "triton", True)
        + (None, None, None, *values),
    )
