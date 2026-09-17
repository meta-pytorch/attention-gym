"""Direct selector tests; public registration/dispatch is covered by indexer integration tests."""

import pytest
import torch

pytest.importorskip("cutlass.cute", reason="CuTeDSL 4.5+ is required for named TVM-FFI kernels")
import cutlass

from attn_gym.sparse.indexer.impl.cute_topk_gvr2 import (
    _CROSS_MAX,
    _RANK_MAX,
    _STAGE_MAX,
    IndexerGVR2TopKKernel,
)
from attn_gym.testing.indexer import (
    assert_indexer_topk_values as _assert_rows,
)
from attn_gym.testing.indexer import (
    compile_indexer_topk as _compile_kernel,
)
from attn_gym.testing.indexer import (
    indexer_gvr2_sample_positions as _sample_positions,
)

CUTE_SUPPORTED = torch.cuda.is_available() and torch.cuda.get_device_capability() in (
    (10, 0),
    (10, 3),
)
pytestmark = pytest.mark.skipif(not CUTE_SUPPORTED, reason="SM100 or SM103 required")


@pytest.mark.parametrize(
    "distribution", ["random", "negative", "zero", "equal", "clustered", "extreme", "signed_zero"]
)
@pytest.mark.parametrize("topk", [1, 37, 512, 1024, 2047, 2048, 4096])
def test_gvr2_exact_values(distribution, topk):
    """Sampling, tie emission, degenerate samples and the arbitrary-K radix path are exact.

    S=4097 also misaligns odd rows, exercising the scalar head/tail around float4 loads.
    """
    generator = torch.Generator(device="cuda").manual_seed(32)
    scores = torch.randn(2, 2, 4097, device="cuda", generator=generator)
    match distribution:
        case "negative":
            scores = -scores.abs()
        case "zero":
            scores.zero_()
        case "equal":
            scores.fill_(-1)
        case "clustered":
            scores = 1 + scores.abs() * 1e-5
        case "extreme":
            values = torch.tensor(
                [
                    -torch.finfo(torch.float32).max,
                    -1e30,
                    -1e-30,
                    -1e-45,
                    0,
                    1e-45,
                    1e-30,
                    1e30,
                    torch.finfo(torch.float32).max,
                ],
                device="cuda",
            )
            scores.copy_(
                values[torch.arange(scores.numel(), device="cuda") % len(values)].view_as(scores)
            )
        case "signed_zero":
            scores = torch.copysign(torch.zeros_like(scores), scores)
    output = torch.full((1, 4, topk), -99, device="cuda", dtype=torch.int32)
    old_output = torch.empty_like(output)
    _compile_kernel(topk)(scores, output, cutlass.Int32(0))
    _compile_kernel(topk, radix=True)(scores, old_output, cutlass.Int32(0))
    _assert_rows(output[0], scores.view(4, -1))
    _assert_rows(old_output[0], scores.view(4, -1))


@pytest.mark.parametrize(
    "scenario",
    ["degenerate", "infinite_width", "overshoot", "rung", "overflow", "ties", "narrowing"],
)
@pytest.mark.parametrize("candidates,topk", [(16384, 512), (32768, 1024)])
@pytest.mark.parametrize("rows", [2, 300])
def test_gvr2_constructed_paths(scenario, candidates, topk, rows):
    """Force verify/refine/fallback routes by controlling sampled values.

    Rows 2/300 exercise the 512/256-thread variants.
    """
    threads = 256 if rows > IndexerGVR2TopKKernel.wide_row_limit else 512
    sampled = _sample_positions(candidates, topk, threads)
    generator = torch.Generator(device="cuda").manual_seed(5)
    scores = torch.randn(1, 2, candidates, device="cuda", generator=generator) * 0.1
    match scenario:
        case "degenerate":
            scores[..., sampled] = 0.25
        case "infinite_width":
            scores[..., sampled[::2]] = torch.finfo(torch.float32).max
            scores[..., sampled[1::2]] = -torch.finfo(torch.float32).max
        case "overshoot":
            # Half the sample is the row maximum: the guess and the deeper rung both
            # resolve to the top sample bin, which holds fewer than K values.
            scores[..., sampled[::2]] = 10
            scores[..., sampled[1::2]] = 0
            assert sampled.numel() // 2 < topk
        case "rung":
            # Fewer maxima than the guess target but more than the rung target: the
            # first pass counts < K, the rung at twice the depth admits the whole row.
            scores.uniform_(-5, 5, generator=generator)
            scores[..., sampled] = 0
            scores[..., sampled[:40]] = 10
            assert 40 < topk
        case "overflow":
            # Sample spread over [-10, 10]; a dense block just under its top quantiles
            # keeps the affine bins narrow while well over _STAGE_MAX values survive.
            spread = torch.linspace(-10, 10, sampled.numel(), device="cuda")
            scores[..., sampled] = spread
            free = torch.ones(candidates, dtype=torch.bool, device="cuda")
            free[sampled] = False
            free_idx = free.nonzero().flatten()
            count = _STAGE_MAX + 1024
            dense = torch.linspace(9.2, 10, count, device="cuda")
            scores[..., free_idx[:count]] = dense
            scores[..., free_idx[count:]] = -100
        case "ties":
            free = torch.ones(candidates, dtype=torch.bool, device="cuda")
            free[sampled] = False
            free_idx = free.nonzero().flatten()
            scores[..., free_idx[: 3 * _CROSS_MAX]] = 1.0
            scores[..., sampled[::4]] = 1.0
        case "narrowing":
            # 800 distinct values one ulp apart share one wide bin (the sample spans
            # [-100, 1]) with the sampled anchors; K-212 clear winners above them leave
            # a partial need in that bin.
            count = 800
            free = torch.ones(candidates, dtype=torch.bool, device="cuda")
            free[sampled] = False
            free_idx = free.nonzero().flatten()
            ladder = 1 + torch.arange(1, count + 1, device="cuda") * 2.0**-23
            assert ladder.unique().numel() == count
            scores[..., free_idx[:count]] = ladder
            scores[..., free_idx[count : count + topk - 212]] = 5.0
            scores[..., sampled[::8]] = 1.0
            scores[..., sampled[1]] = -100
            assert count > _RANK_MAX
    scores = scores.expand(rows // 2, 2, candidates).contiguous()
    output = torch.full((1, rows, topk), -99, device="cuda", dtype=torch.int32)
    _compile_kernel(topk)(scores, output, cutlass.Int32(0))
    _assert_rows(output[0], scores.view(rows, candidates))


@pytest.mark.parametrize("count,topk", [(288, 37), (289, 37), (1024, 512), (1025, 512)])
def test_gvr2_crossing_size_boundaries(count, topk):
    """Crossing bins at the direct-rank and refinement-scratch limits stay exact.

    ``count`` one-ulp-spaced values share one bin below a few sampled clear winners
    (fewer than the guess target, at least the K-th-value target so the window covers
    them), so the crossing bin holds exactly ``count`` candidates: 288 ranks directly,
    289 narrows, 1024 narrows, 1025 falls back to pool radix over the staged survivors.
    """
    candidates = 16384
    anchors = 10 if topk == 37 else 25
    sampled = _sample_positions(candidates, topk)
    generator = torch.Generator(device="cuda").manual_seed(11)
    scores = torch.empty(1, 2, candidates, device="cuda").uniform_(-50, 0, generator=generator)
    free = torch.ones(candidates, dtype=torch.bool, device="cuda")
    free[sampled] = False
    free_idx = free.nonzero().flatten()
    scores[..., free_idx[:count]] = 0.5 + torch.arange(1, count + 1, device="cuda") * 2.0**-24
    scores[..., sampled[:anchors]] = 2.0
    output = torch.full((1, 2, topk), -99, device="cuda", dtype=torch.int32)
    _compile_kernel(topk)(scores, output, cutlass.Int32(0))
    _assert_rows(output[0], scores.view(2, candidates))


def test_gvr2_narrowing_interval_bounds():
    """Narrow a ladder of distinct large positive values after saturated affine binning.

    This narrow key span does not overflow an endpoint. The mixed-sign endpoint
    regression is covered in test_indexer_gvr2_boundaries.py.
    """
    candidates, topk, count = 16384, 37, 400
    sampled = _sample_positions(candidates, topk)
    generator = torch.Generator(device="cuda").manual_seed(13)
    scores = torch.randn(1, 2, candidates, device="cuda", generator=generator) * 0.1
    free = torch.ones(candidates, dtype=torch.bool, device="cuda")
    free[sampled] = False
    ladder = 3e38 - torch.arange(count, device="cuda", dtype=torch.float64) * 2.0**104
    assert ladder.float().unique().numel() == count
    scores[..., free.nonzero().flatten()[:count]] = ladder.float()
    output = torch.full((1, 2, topk), -99, device="cuda", dtype=torch.int32)
    _compile_kernel(topk)(scores, output, cutlass.Int32(0))
    _assert_rows(output[0], scores.view(2, candidates))


def test_gvr2_collapsed_window():
    """Review construction: unsampled +FLT_MAX outliers overflow the window span.

    The sample sits near -FLT_MAX, so the classification span saturates; the
    outliers must still classify above every other survivor.
    """
    candidates, topk = 1024, 37
    sampled = _sample_positions(candidates, topk)
    assert sampled.numel() == 512
    largest = torch.finfo(torch.float32).max
    scores = torch.full((1, 2, candidates), -largest, device="cuda")
    scores[..., sampled] = -1.7e38
    scores[..., sampled[:20]] = -5e37
    free = torch.ones(candidates, dtype=torch.bool, device="cuda")
    free[sampled] = False
    scores[..., free.nonzero().flatten()[:40]] = largest
    output = torch.full((1, 2, topk), -99, device="cuda", dtype=torch.int32)
    _compile_kernel(topk)(scores, output, cutlass.Int32(0))
    _assert_rows(output[0], scores.view(2, candidates))


@pytest.mark.parametrize("topk", [37, 512])
def test_gvr2_whole_crossing_bin(topk):
    """Exactly K values above the rest emits the crossing bin whole without refinement."""
    candidates = 16384
    generator = torch.Generator(device="cuda").manual_seed(9)
    scores = torch.randn(1, 2, candidates, device="cuda", generator=generator) * 0.01
    winners = torch.randperm(candidates, device="cuda", generator=generator)[:topk]
    scores[..., winners] = 5 + torch.rand(topk, device="cuda", generator=generator) * 1e-3
    output = torch.full((1, 2, topk), -99, device="cuda", dtype=torch.int32)
    _compile_kernel(topk)(scores, output, cutlass.Int32(0))
    _assert_rows(output[0], scores.view(2, candidates))
    assert (output[0].sort(-1).values == winners.sort().values.int()).all()


@pytest.mark.parametrize("candidates", [7, 33, 257, 4100, 4103])
def test_gvr2_short_and_unaligned_rows(candidates):
    """Rows below the float4 body, with scalar heads/tails, and arbitrary K stay exact."""
    scores = torch.randn((2, 2, candidates), device="cuda")
    for topk in (1, candidates // 2, candidates - 1):
        output = torch.full((1, 4, topk), -99, device="cuda", dtype=torch.int32)
        _compile_kernel(topk)(scores, output, cutlass.Int32(0))
        _assert_rows(output[0], scores.view(4, candidates))


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("tokens,ratio,topk", [(1, 1, 1), (17, 4, 2), (65, 1, 37), (9, 4, 2)])
def test_gvr2_partial_batch_slabs(wide, causal, tokens, ratio, topk):
    """Odd batch pairs, empty prefixes, NaN-poisoned padding and a partial final slab."""
    batch, capacity = 3, 4
    candidates = tokens // ratio
    generator = torch.Generator(device="cuda").manual_seed(17)
    rows = torch.randn(batch, tokens, candidates, device="cuda", generator=generator)
    pairs_per_batch = (tokens + 1) // 2
    padded = torch.full((batch, pairs_per_batch * 2, candidates), torch.nan, device="cuda")
    padded[:, :tokens] = rows
    if causal:
        visible = (torch.arange(tokens, device="cuda") + 1) // ratio
        invalid = torch.arange(candidates, device="cuda")[None, :] >= visible[:, None]
        padded[:, :tokens].masked_fill_(invalid, torch.nan)
    slabs = padded.view(-1, 2, candidates)
    output = torch.full((batch, tokens, topk), -99, device="cuda", dtype=torch.int32)
    kernel = _compile_kernel(topk, causal, ratio, wide)
    integer = cutlass.Int64 if wide else cutlass.Int32
    for start in range(0, len(slabs), capacity):
        kernel(slabs[start : start + capacity], output, integer(start))
    for query in range(tokens):
        visible = (query + 1) // ratio if causal else candidates
        count = min(topk, visible)
        indices = output[:, query]
        assert ((indices == -1).sum(-1) == topk - count).all()
        valid_indices = indices[indices >= 0].view(batch, count)
        _assert_rows(valid_indices, rows[:, query, :visible])


@pytest.mark.parametrize("candidates,slab_rows,topk", [(16384, 512, 512), (32768, 256, 1024)])
@pytest.mark.parametrize("wide", [False, True])
def test_gvr2_representative_causal_slab(candidates, slab_rows, topk, wide):
    """Late compressed slabs exercise full rows, output offsets and both CTA-size variants.

    512 rows exceed ``wide_row_limit`` and run the 256-thread kernel; 256 rows run the
    512-thread kernel.
    """
    assert (slab_rows > IndexerGVR2TopKKernel.wide_row_limit) == (slab_rows == 512)
    tokens, ratio = candidates * 4 + 1, 4
    start_query = tokens - slab_rows - 1
    generator = torch.Generator(device="cuda").manual_seed(31)
    scores = torch.randn(slab_rows // 2, 2, candidates, device="cuda", generator=generator)
    visible = (torch.arange(start_query, start_query + slab_rows, device="cuda") + 1) // ratio
    valid = torch.arange(candidates, device="cuda")[None, :] < visible[:, None]
    scores.view(slab_rows, candidates).masked_fill_(~valid, torch.nan)
    output = torch.full((1, tokens, topk), -99, device="cuda", dtype=torch.int32)
    integer = cutlass.Int64 if wide else cutlass.Int32
    _compile_kernel(topk, True, ratio, wide)(scores, output, integer(start_query // 2))
    selected = output[0, start_query : start_query + slab_rows]
    assert ((selected >= 0) & (selected < visible[:, None])).all()
    reference = scores.view(slab_rows, candidates).masked_fill(~valid, -torch.inf)
    _assert_rows(selected, reference)
    assert (output[:, :start_query] == -99).all()
    assert (output[:, start_query + slab_rows :] == -99).all()


@pytest.mark.parametrize("wide", [False, True])
def test_gvr2_graph_replay_changed_inputs(wide):
    """One capture reuses the same pointers across fallback, ties, overshoot and normal rows."""
    candidates, topk = 16384, 512
    scores = torch.randn((2, 2, candidates), device="cuda")
    output = torch.empty((1, 4, topk), device="cuda", dtype=torch.int32)
    kernel = _compile_kernel(topk, wide=wide)
    integer = cutlass.Int64 if wide else cutlass.Int32
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        kernel(scores, output, integer(0))
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel(scores, output, integer(0))
    sampled = _sample_positions(candidates, topk)
    for update in range(5):
        generator = torch.Generator(device="cuda").manual_seed(83 + update)
        scores.normal_(generator=generator)
        if update == 1:
            scores.fill_(-1)
        elif update == 2:
            scores[..., sampled] = 100
        elif update == 3:
            scores[..., sampled] = 0
        for _ in range(3):
            output.fill_(-99)
            graph.replay()
            _assert_rows(output[0], scores.view(4, candidates))


@pytest.mark.parametrize("wide", [False, True])
def test_gvr2_large_row_address(wide):
    """Row and float4 addressing stay exact when an index product would overflow int32."""
    candidates = (2**31 - 1) // 511 + 1
    assert 2 * candidates < 2**31 - 1
    scores = torch.zeros((1, 2, candidates), device="cuda")
    scores[..., -1] = 1
    output = torch.empty((1, 2, 1), device="cuda", dtype=torch.int32)
    integer = cutlass.Int64 if wide else cutlass.Int32
    _compile_kernel(1, wide=wide)(scores, output, integer(0))
    assert (output == candidates - 1).all()
