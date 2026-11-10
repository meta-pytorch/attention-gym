"""Triton grouped schedule: separate values and grouped KV heads on Blackwell."""

import contextlib

import pytest
import torch

from attn_gym.sparse.gather_attn import AuxRequest, Impl, gather_attn
from attn_gym.testing import cumulative_sequence_offsets
from attn_gym.testing.gather_attn import (
    assert_matches_low_precision_eager,
    make_gather_attn_test_inputs,
)

pytest.importorskip("triton")
from attn_gym.sparse.gather_attn.impl.triton import forward

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
    sink, window 0, and distinct 4-token blocks (top-k 131) over 1024- and 512-token documents.
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
        "int64-offsets",
        "fused-kv-buffer",
        "mixed-layouts",
        "empty-pool",
        "no-selections",
    ],
)
def test_grouped_input_variants_match_reference(monkeypatch, variant, mode):
    """Grouped inputs QSA-style callers pass, on the gather kernel with a local window.

    ``aliased-pools`` passes one tensor as both the local and sparse keys (and values), as
    QSA does; ``int64-offsets`` forces 64-bit addressing. ``fused-kv-buffer`` slices keys
    and values from one [..., 2 * D] buffer and ``mixed-layouts`` keeps keys BHSD with values
    BSHD, so values' strides differ from the keys'.
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
    _assert_matches_reference(inputs, mode, **options)


def _valid_candidates(*tuners_and_args):
    """The largest number of valid candidates among (tuner, launch arguments) pairs."""
    return max(
        len(tuner._prune(tuner._configs, args) if tuner._prune else tuner._configs)
        for tuner, args in tuners_and_args
    )


_GATHER_ARGS = {"TOPK": 64, "D": 128, "WINDOW": 19}
_TUNER_CASES = {"gather": _valid_candidates((forward._FWD_GROUPED, _GATHER_ARGS))}


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

    ``gather`` (D=128, a window, top-k 64) covers the gather forward candidates.
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

    Shifting every query's selections to its neighbour, plus a repeat in the first query,
    changes every query's compacted rows and counts.
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
        inputs["kv_indices"].copy_(replay_indices)
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

    selected_rows, selected_counts = resolve_selected_rows(indices, sparse_tokens, cu_q, cu_k)

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
@pytest.mark.parametrize("case", ["separate-values", "grouped-shared-values", "one-pool"])
@torch.no_grad()
def test_grouped_fullgraph(case):
    """Grouped inputs trace through fullgraph compile and match eager bitwise, including the
    LSE: separate values, grouped heads, and QSA's packed call with one pool for local and
    sparse keys (and values) and one offsets tensor for both partitions."""
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
    "case", ["gqa-shared-values", "gqa-separate-values", "mqa-separate-values"]
)
def test_grouped_operator_contracts(case):
    """The forward operator's schema, fake implementation, and compiled dispatch hold for
    grouped heads and separate values."""
    from attn_gym.sparse.gather_attn.impl.ops import _gather_attn_fwd_op

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
