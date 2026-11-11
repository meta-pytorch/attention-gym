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


def _force_mode(monkeypatch, mode):
    """Pick the sparse dK/dV kernel: query-major atomics, key-major, or deterministic mode.

    The density heuristic prefers atomics in deterministic mode too, so only the
    deterministic override keeps that mode key-major. Each mode fails the test if the other
    kernel runs: the atomic launch, or the key-major map build.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError(f"{mode} mode ran the other sparse dK/dV kernel")

    class ForbiddenLaunch:
        def __getitem__(self, grid):
            return forbidden

    monkeypatch.setattr(backward, "_prefer_atomic_dsparse_kv", lambda *args: mode != "key-major")
    if mode == "atomic":
        monkeypatch.setattr(backward, "build_tile_query_map", forbidden)
    else:
        monkeypatch.setattr(backward, "BWD_DSPARSE_KV_SHARED_ATOMIC", ForbiddenLaunch())


def _assert_matches_reference(inputs, mode, *, grad_output=None, **options):
    """Bound Triton output, LSE, and every input gradient by bf16 reference error vs FP64.

    Triton runs in real deterministic mode for ``deterministic``. Without atomics (every mode
    but ``atomic``) a second run must be bitwise equal.
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
    if mode != "atomic":
        for first, second in zip(actual[2], repeat, strict=True):
            assert torch.equal(first, second)


@pytest.mark.usefixtures("nan_filled_empty")
@pytest.mark.parametrize("mode", ["atomic", "key-major", "deterministic"])
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
    monkeypatch, heads, kv_heads, head_dim, separate_values, documents, window, topk, mode
):
    """Forward and every gradient match for each query-head group, including masked head tiles.

    ``documents`` packs (query, key) document lengths into one batch row; the ragged case has
    empty documents, key documents spanning several dK/dV tiles with untouched tiles, and
    query positions past the last document. Groups of 48 span several head blocks per KV
    head in the forward (D=256) or the backward (D=64). Fresh buffers start as NaN.
    """
    _force_mode(monkeypatch, mode)
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
    """Count launches of the three dense-tile kernels: forward, dQ, and dK/dV."""
    launches = {}

    class CountingLaunch:
        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __getitem__(self, grid):
            launches[self.name] += 1
            return self.kernel[grid]

    for module, name in (
        (forward, "_FWD_DENSE"),
        (backward, "BWD_DQ_DENSE"),
        (backward, "BWD_DSPARSE_KV_DENSE"),
    ):
        launches[name] = 0
        monkeypatch.setattr(module, name, CountingLaunch(name, getattr(module, name)))
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
    return plan_dense_tiles(
        counts, selection_spans, options.get("cu_seqlens"), tile_queries, document_bounds=False
    )[0]


@pytest.mark.usefixtures("nan_filled_empty")
@pytest.mark.parametrize("mode", ["atomic", "key-major", "deterministic"])
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

    With ``all_dense`` every tile is dense, so the key-major map holds only unused slots. Groups
    3 and 48 leave part of each 128-row tile empty; D=192 pads the head tile. Every dense kernel
    must run.
    """
    _force_mode(monkeypatch, mode)
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
    assert all(launches.values()), launches


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
@pytest.mark.parametrize("mode", ["atomic", "key-major", "deterministic"])
@pytest.mark.parametrize("heads,kv_heads", [(24, 2), (12, 1)], ids=["24-2", "12-1"])
def test_qsa_block_selections_match_reference(monkeypatch, heads, kv_heads, mode):
    """QSA's regime at reduced size: D=256, one pool for local and sparse keys and values, no
    sink, window 0, and distinct 4-token blocks (top-k 131) over 1024- and 512-token documents.

    Early queries' tiles go dense over 20-word bitmaps; later queries select past the bitmap
    and gather, so the key-major map inverts thousands of distinct keys.
    """
    _force_mode(monkeypatch, mode)
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
    assert all(launches.values()), launches


def _misaligned(tensor):
    """A contiguous copy whose data starts one element past a 16-byte boundary."""
    storage = torch.empty(tensor.numel() + 1, device=tensor.device, dtype=tensor.dtype)
    return storage[1:].view_as(tensor).copy_(tensor)


@pytest.mark.usefixtures("nan_filled_empty")
@pytest.mark.parametrize("mode", ["atomic", "key-major", "deterministic"])
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
            "misaligned",
            "int64-offsets",
            "fused-kv-buffer",
            "mixed-layouts",
        )
    ]
    + [("gather", "empty-pool"), ("gather", "no-selections")],
)
def test_grouped_input_variants_match_reference(monkeypatch, base, variant, mode):
    """Grouped inputs QSA-style callers pass, on the gather kernels (with a local window) and
    on mixed dense and gather tiles.

    ``aliased-pools`` passes one tensor as both the local and sparse keys (and values), as
    QSA does; ``misaligned`` starts the query and output gradient off a 16-byte boundary,
    which TMA gathers cannot read in place; ``int64-offsets`` forces 64-bit addressing.
    ``fused-kv-buffer`` slices keys and values from one [..., 2 * D] buffer and
    ``mixed-layouts`` keeps keys BHSD with values BSHD, so values' strides differ from the
    keys'. Every ``dense`` variant still plans dense tiles next to gather tiles.
    """
    from attn_gym.sparse.gather_attn.impl.triton import primitives

    _force_mode(monkeypatch, mode)
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
    if base == "dense":
        dense = _planned_tile_spans(inputs, options)[..., 1] >= 0
        assert dense.any() and not dense.all()
    _assert_matches_reference(inputs, mode, grad_output=grad_output, **options)
    assert all(launches.values()) == (base == "dense"), launches


def _valid_candidates(*tuners_and_args):
    """The largest number of valid candidates among (tuner, launch arguments) pairs."""
    return max(
        len(tuner._prune(tuner._configs, args) if tuner._prune else tuner._configs)
        for tuner, args in tuners_and_args
    )


_GATHER_ARGS = {"TOPK": 64, "D": 128, "WINDOW": 19}
_DENSE_ARGS = {
    "TOPK": 48,
    "D": 256,
    "WINDOW": 0,
    "BLOCK_D": 256,
    "query_ptr": torch.empty(0, dtype=torch.bfloat16),
}
_TUNER_CASES = {
    "gather": _valid_candidates(
        (forward._FWD_GROUPED, _GATHER_ARGS),
        (shared_backward.BWD_DQ_GROUPED, _GATHER_ARGS),
    ),
    "atomic": _valid_candidates((backward.BWD_DSPARSE_KV_SHARED_ATOMIC, _GATHER_ARGS)),
    "dense": _valid_candidates(
        (forward._FWD_DENSE, _DENSE_ARGS),
        (forward._FWD_GROUPED, _DENSE_ARGS),
        (backward.BWD_DQ_DENSE, _DENSE_ARGS),
        (shared_backward.BWD_DQ_GROUPED, _DENSE_ARGS),
        (backward.BWD_DSPARSE_KV_DENSE, _DENSE_ARGS),
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
def test_every_tuner_config_matches_reference(monkeypatch, case):
    """Every grouped tuner candidate is correct, not only the first: the ``tuner_config``
    marker pins each tuner to that index of its valid candidates (see
    ``gather_attn_single_config``), and each case runs as many indices as its largest valid
    candidate list, so shorter lists repeat but none is skipped.

    ``gather`` (D=128, a window, top-k 64) covers the gather forward and dQ candidates,
    ``atomic`` the atomic dK/dV candidates, and ``dense`` (D=256) the dense-tile kernels' and
    the D=256 gather candidates.
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
    mode = "atomic" if case == "atomic" else "key-major"
    _force_mode(monkeypatch, mode)
    _assert_matches_reference(inputs, mode, **options)


@pytest.mark.parametrize("mode", ["atomic", "key-major", "deterministic"])
@pytest.mark.parametrize("base", ["dense", "all-dense", "window"])
def test_grouped_cuda_graph_replays_new_inputs(monkeypatch, base, mode):
    """Forward and backward capture into one CUDA graph, so they read nothing back to the
    host (a host read fails capture), and a replay on new values and selections matches eager.

    Shifting every query's selections to its neighbour, plus a repeat in the first query,
    changes which tiles are dense and how many selections the key-major map holds, so nothing
    sized at capture may depend on them. ``all-dense`` captures with no gather selections and
    replays with some. Nothing may compile or tune during capture: eager and captured launches
    share every kernel and specialization. ``window`` adds the local dK/dV kernel, and its
    4736-slot map takes ATen's large sort. Only atomic sparse dK/dV may differ, by FP32
    summation order.
    """
    _force_mode(monkeypatch, mode)
    key_major_launches = []
    key_major_launch = backward._gather_attn_bwd_dsparse_kv_grouped

    class CountingLaunch:
        def __getitem__(self, grid):
            key_major_launches.append(grid)
            return key_major_launch[grid]

    monkeypatch.setattr(backward, "_gather_attn_bwd_dsparse_kv_grouped", CountingLaunch())
    if base != "window":
        inputs, options = _dense_tile_inputs(24, 2, 256, True, True, base == "all-dense", "short")
    else:
        inputs = make_gather_attn_test_inputs(
            heads=24,
            kv_heads=2,
            head_dim=128,
            tokens=128,
            sparse_tokens=79,
            topk=37,
            separate_values=True,
            dtype=torch.bfloat16,
            batch=2,
        )
        options = {"sliding_window_size": 19}
    options["kernel_options"] = {"backend": "triton"}
    leaves = _differentiable_leaves(inputs)
    grad_output = torch.randn_like(inputs["query"])
    with _deterministic_algorithms(mode == "deterministic"):
        # Warm up (compile and tune) on a side stream, as CUDA graph capture requires.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _run(inputs, leaves, grad_output, **options)
        torch.cuda.current_stream().wait_stream(stream)
        warm_up_launches = len(key_major_launches)
        captured_compiles = []

        def record_compile(*args, **kwargs):
            captured_compiles.append(kwargs.get("repr"))
            return False

        compile_hook = triton.knobs.runtime.jit_cache_hook
        monkeypatch.setattr(triton.knobs.runtime, "jit_cache_hook", record_compile)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = _run(inputs, leaves, grad_output, **options)
        monkeypatch.setattr(triton.knobs.runtime, "jit_cache_hook", compile_hook)
        assert len(key_major_launches) == 2 * warm_up_launches
        assert not captured_compiles
        with torch.no_grad():
            for _, tensor in leaves:
                tensor.copy_(torch.randn_like(tensor))
            grad_output.copy_(torch.randn_like(grad_output))
            replay_indices = inputs["kv_indices"].roll(1, dims=1)
            replay_indices[:, 0, 1] = replay_indices[:, 0, 0]
            if base != "window":
                captured_plan = _planned_tile_spans(inputs, options)[..., 1] >= 0
            inputs["kv_indices"].copy_(replay_indices)
        if base != "window":
            assert not torch.equal(
                captured_plan, _planned_tile_spans(inputs, options)[..., 1] >= 0
            )
        graph.replay()
        expected = _run(inputs, leaves, grad_output, **options)
    torch.cuda.synchronize()
    names = ["output", "lse", *(name for name, _ in leaves)]
    for name, actual, reference in zip(
        names,
        (captured[0], captured[1], *captured[2]),
        (expected[0], expected[1], *expected[2]),
        strict=True,
    ):
        if mode == "atomic" and name in ("sparse_kv", "sparse_v"):
            # One bf16 rounding of a reordered FP32 sum.
            torch.testing.assert_close(actual, reference, atol=1e-5, rtol=2**-7)
        else:
            assert torch.equal(actual, reference), name


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
def test_dense_tile_plan_matches_recount(packed):
    """Selection bitmaps, spans, the bitmap-exact flag, the dense-tile plan, and each
    document's dense-tile bounds match a Python recount of their definitions."""
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
    tile_spans, bounds = plan_dense_tiles(
        counts, spans, cu_seqlens, tile_queries, document_bounds=True
    )

    bits = 32 * mask_words
    query_offsets = cu_seqlens.tolist() if packed else [0, tokens]
    key_offsets = cu_seqlens_k.tolist() if packed else [0, pool]
    num_tiles = triton.cdiv(tokens, tile_queries)
    expected_bounds = [
        [num_tiles, -1, -1] for _ in range(len(query_offsets) - 1 if packed else batch)
    ]
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
            if dense:
                document_bounds = expected_bounds[members[0][0] if packed else b]
                document_bounds[0] = min(document_bounds[0], tile)
                document_bounds[1] = max(document_bounds[1], tile)
                document_bounds[2] = max(document_bounds[2], span_hi)
    assert bounds.tolist() == expected_bounds
    assert 0 < (tile_spans[..., 1] >= 0).sum() < tile_spans[..., 1].numel()


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

    selected_rows, selected_counts, _, _ = resolve_selected_rows(
        indices, sparse_tokens, cu_q, cu_k
    )
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
@pytest.mark.parametrize("path", ["gather", "dense"])
@pytest.mark.parametrize("mode", ["key-major", "atomic"])
def test_unselected_keys_get_zero_gradients(monkeypatch, separate_values, path, mode):
    """Keys no query selects keep zero dK/dV even when they share a key tile with selected
    keys and hold NaN, as inactive pool capacity may. A nonfinite delta (rowsum(dO * O)) in one
    document also stays out of keys it did not select, including another document's keys in
    the same tile.

    ``gather`` turns dense tiles off; ``dense`` plans dense tiles in both documents, whose
    spans cover the unselected keys between their selections."""
    _force_mode(monkeypatch, mode)
    if path == "gather":
        for module in (forward, backward):
            monkeypatch.setattr(module, "dense_tile_shape", lambda *args: None)
    generator = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    heads, kv_heads, head_dim, tokens, pool_rows = 24, 2, 64, 64, 96
    pools = [randn(1, kv_heads, pool_rows, head_dim) for _ in range(1 + separate_values)]
    # Documents of 36 and 28 queries over 36 and 30 keys: rows 66 onward are inactive capacity.
    # Row 33, which document 0 selects, makes that document's outputs and deltas NaN.
    for pool in pools:
        pool[:, :, 66:] = float("nan")
        pool[:, :, 33] = float("nan")
    # Every fourth key plus the document's last: rows 0, 4, ..., 32, 33 for document 0 and
    # 36, 40, ..., 64, 65 for document 1. Rows 32-63 hold both documents' keys, and rows 64
    # and 65 share their tile with the capacity.
    kv_indices = torch.full((1, tokens, 16), -1, device="cuda")
    unselected = torch.ones(pool_rows, dtype=torch.bool, device="cuda")
    for start, stop, last in ((0, 36, 33), (36, 64, 29)):
        local = list(range(0, last, 4)) + [last]
        kv_indices[0, start:stop, : len(local)] = torch.tensor(local, device="cuda")
        unselected[[start + index for index in local]] = False
    inputs = {
        "query": randn(1, heads, tokens, head_dim),
        "local_kv": None,
        "sparse_kv": pools[0],
        "kv_indices": kv_indices,
        "sparse_v": pools[-1] if separate_values else None,
    }
    options = {
        "sliding_window_size": 0,
        "cu_seqlens": cumulative_sequence_offsets([36, 28]),
        "cu_seqlens_k": cumulative_sequence_offsets([36, 30]),
        "kernel_options": {"backend": "triton"},
    }
    if path == "dense":
        dense = _planned_tile_spans(inputs, options)[0, :, 1] >= 0
        # Tiles of 10 queries: 0-2 lie in document 0 and 4-5 in document 1.
        assert dense[[0, 1, 2, 4, 5]].all(), dense
    leaves = [tensor.requires_grad_() for tensor in (inputs["query"], *pools)]
    output = gather_attn(**inputs, **options)
    pool_gradients = torch.autograd.grad(output, leaves, randn(*output.shape))[1:]
    for gradient in pool_gradients:
        assert torch.equal(
            gradient[:, :, unselected], torch.zeros_like(gradient[:, :, unselected])
        )
        # Only document 1's queries, whose deltas are finite, select rows 36 onward.
        assert gradient[:, :, 36:66].isfinite().all()


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


def test_dense_tile_kernels_reuse_one_binary_across_shapes(triton_compiles):
    """Sequence lengths, tile counts, document counts, and pool lengths reuse one binary of
    each dense-tile kernel across Triton's == 1 and multiple-of-16 classes.

    Heads, head dim, and top-k stay fixed, so the tile shape and kernel config do too: groups
    of 12 make 10-query tiles. The planner builds one binary for the forward and one with
    the backward's document bounds.
    """
    # (S, tiles, documents, pool): (1, 1, 1, 1), (16, 2, 2, 16), (160, 16, 16, 208),
    # (37, 4, 3, 49), (320, 32, 10, 350).
    for documents, key_lengths in (
        ([1], [1]),
        ([8, 8], [11, 5]),
        ([10] * 16, [13] * 16),
        ([5, 0, 32], [7, 2, 40]),
        ([32] * 10, [35] * 10),
    ):
        inputs = make_gather_attn_test_inputs(
            heads=24,
            kv_heads=2,
            head_dim=128,
            tokens=sum(documents),
            sparse_tokens=sum(key_lengths),
            topk=16,
            separate_values=True,
            dtype=torch.bfloat16,
        )
        leaves = [
            tensor.requires_grad_() for tensor in inputs.values() if tensor.is_floating_point()
        ]
        output = gather_attn(
            **inputs,
            sliding_window_size=0,
            cu_seqlens=cumulative_sequence_offsets(documents),
            cu_seqlens_k=cumulative_sequence_offsets(key_lengths),
            kernel_options={"backend": "triton"},
        )
        torch.autograd.grad(output, leaves, torch.randn_like(output))
    assert triton_compiles["_plan_dense_tiles"] == 2
    for kernel in (
        "_gather_attn_fwd_dense",
        "_gather_attn_bwd_dq_dense",
        "_gather_attn_bwd_dsparse_kv_dense",
    ):
        assert triton_compiles[kernel] == 1, kernel


def test_atomic_dsparse_kv_reuses_one_binary_across_shapes(monkeypatch, triton_compiles):
    """Sequence lengths, document counts, pool lengths, and their strides reuse one binary of
    the atomic sparse dK/dV kernel across Triton's == 1 and multiple-of-16 classes.

    Top-k 37 keeps the indices' batch stride, top-k times the sequence length, off both
    classes unless the sequence length moves it there.
    """
    _force_mode(monkeypatch, "atomic")
    # (S, documents, pool): (1, 1, 1), (16, 2, 16), (160, 16, 208), (37, 3, 49), (320, 10, 350).
    for documents, key_lengths in (
        ([1], [1]),
        ([8, 8], [11, 5]),
        ([10] * 16, [13] * 16),
        ([5, 0, 32], [7, 2, 40]),
        ([32] * 10, [35] * 10),
    ):
        inputs = make_gather_attn_test_inputs(
            heads=16,
            kv_heads=2,
            head_dim=64,
            tokens=sum(documents),
            sparse_tokens=sum(key_lengths),
            topk=37,
            separate_values=True,
            dtype=torch.bfloat16,
        )
        leaves = [
            tensor.requires_grad_() for tensor in inputs.values() if tensor.is_floating_point()
        ]
        output = gather_attn(
            **inputs,
            sliding_window_size=0,
            cu_seqlens=cumulative_sequence_offsets(documents),
            cu_seqlens_k=cumulative_sequence_offsets(key_lengths),
            kernel_options={"backend": "triton"},
        )
        torch.autograd.grad(output, leaves, torch.randn_like(output))
    assert triton_compiles["_gather_attn_bwd_dsparse_kv_shared_atomic"] == 1


@pytest.mark.xfail(
    raises=AssertionError,
    strict=True,
    reason="dense tiles multiply unselected rows by zero probabilities and dS",
)
@pytest.mark.parametrize("poisoned", ["values", "keys"])
def test_dense_tiles_ignore_unselected_nan_rows(poisoned):
    """A NaN in a pool row no query selects leaves the output and dQ finite.

    Every query selects only key 1, so the one dense tile's span is [1, 1], but its first
    K/V tile starts at key 0: 0 * NaN poisons P @ V with a NaN value and dS @ K with a NaN key.
    """
    query = torch.randn(1, 24, 10, 64, device="cuda", dtype=torch.bfloat16)
    keys = torch.randn(1, 2, 2, 64, device="cuda", dtype=torch.bfloat16)
    values = torch.randn_like(keys)
    (values if poisoned == "values" else keys)[:, :, 0] = torch.nan
    indices = torch.ones(1, 10, 1, device="cuda", dtype=torch.int64)
    output = gather_attn(
        query.requires_grad_(),
        None,
        keys,
        indices,
        sparse_v=values,
        sliding_window_size=0,
        kernel_options={"backend": "triton"},
    )
    (grad_query,) = torch.autograd.grad(output, query, torch.randn_like(output))
    assert output.isfinite().all()
    assert grad_query.isfinite().all()


@pytest.mark.parametrize(
    "group,head_dim,v_is_k,topk,keys_per_document,atomic",
    [
        (12, 256, False, 2051, 8192, False),
        (12, 256, False, 2051, 32768, False),
        (1, 128, False, 128, 8192, False),
        (8, 128, False, 512, 8192, False),
        (16, 64, False, 512, 8192, True),
        (8, 128, True, 128, 8192, True),
        (8, 64, False, 256, 8192, False),
        (8, 128, False, 128, 8192, False),
    ],
    ids=[
        "qsa-8k",
        "qsa-32k",
        "mha",
        "gqa-dense",
        "mqa-d64",
        "gqa-sparse-shared-values",
        "gqa-d64-separate-values",
        "gqa-d128-separate-values",
    ],
)
def test_sparse_kv_backward_choice(group, head_dim, v_is_k, topk, keys_per_document, atomic):
    """QSA keeps the key-major dK/dV; sparse selections over small heads use atomics."""
    from attn_gym.sparse.gather_attn.impl.triton import backward

    choice = backward._prefer_atomic_dsparse_kv(
        group, head_dim, v_is_k, topk, keys_per_document, 32
    )
    assert choice == atomic


def test_atomic_dsparse_kv_addresses_expanded_pool_gradients(monkeypatch):
    """A pool expanded across KV heads addresses little storage, but its contiguous FP32
    gradient does not: the last head's rows lie past signed int32 offsets."""
    if torch.cuda.mem_get_info()[0] < 24 * 2**30:
        pytest.skip("needs an 8 GiB FP32 pool gradient plus headroom")
    _force_mode(monkeypatch, "atomic")
    pool = 2**23 + 1
    generator = torch.Generator(device="cuda").manual_seed(0)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    query, local_kv, last_key = randn(1, 32, 3, 64), randn(1, 4, 3, 64), randn(1, 1, 1, 64)
    grad_output = randn(1, 32, 3, 64)

    def run(sparse_rows, last_row):
        leaves = [query.clone().requires_grad_(), local_kv.clone().requires_grad_()]
        sparse_kv = torch.zeros(1, 1, sparse_rows, 64, device="cuda", dtype=torch.bfloat16)
        sparse_kv[:, :, -1:] = last_key
        leaves.append(sparse_kv.requires_grad_())
        indices = torch.full((1, 3, 1), last_row, device="cuda", dtype=torch.int32)
        output = gather_attn(
            leaves[0],
            leaves[1],
            sparse_kv.expand(-1, 4, -1, -1),
            indices,
            sliding_window_size=1,
            kernel_options={"backend": "triton"},
        )
        return output, *torch.autograd.grad(output, leaves, grad_output)

    expected = run(1, 0)
    actual = run(pool, pool - 1)
    for value, reference in zip(actual[:3], expected[:3], strict=True):
        torch.testing.assert_close(value, reference)
    torch.testing.assert_close(actual[3][:, :, -1:], expected[3])
    assert not actual[3][:, :, :-1].any()


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
@pytest.mark.parametrize(
    "case", ["separate-values", "grouped-shared-values", "dense-tiles", "one-pool"]
)
def test_grouped_fullgraph(case):
    """Grouped inputs trace through fullgraph compile, forward and backward, and match eager
    bitwise, including the LSE: separate values, grouped heads whose dV outputs stay empty,
    dense tiles, and QSA's packed call with one pool for local and sparse keys (and values)
    and one offsets tensor for both partitions."""
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


@pytest.mark.parametrize("mode", ["key-major", "deterministic"])
def test_grouped_matches_on_a_side_stream(monkeypatch, mode):
    """Forward and backward on a busy non-default stream equal the default-stream run bitwise.

    A sleep queued first keeps the side stream's kernels pending while the host queues the
    rest of the step, so every kernel and buffer must follow the current stream. The warm-up
    selects nothing to gather, so buffers it recycles hold stale values for the timed runs.
    """
    _force_mode(monkeypatch, mode)
    inputs, options = _dense_tile_inputs(24, 2, 256, True, True, False, "short")
    options["kernel_options"] = {"backend": "triton"}
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
    "case", ["gqa-shared-values", "gqa-separate-values", "mqa-separate-values", "packed-dense"]
)
def test_grouped_operator_contracts(case):
    """The opaque operators' schema, fake implementations, autograd registration, and compiled
    dispatch hold for grouped heads and separate values, including shared values' absent dV,
    and for packed window-0 inputs with dense tiles."""
    from attn_gym.sparse.gather_attn.impl.ops import _gather_attn_bwd_op, _gather_attn_fwd_op

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
