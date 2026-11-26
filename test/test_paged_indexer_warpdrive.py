"""Public paged scorer dispatch and replay on the SM100 family."""

import pytest
import torch

from attn_gym.sparse import paged_lightning_indexer
from attn_gym.testing.indexer import assert_paged_indexer_selection, make_paged_indexer_test_inputs


@pytest.fixture
def score_cache(requires_blackwell):
    pytest.importorskip("cutlass.cute")
    from attn_gym.sparse.indexer.impl.paged_score_warpdrive import compile_paged_score

    compile_paged_score.cache_clear()
    return compile_paged_score


def make_inputs(dtype=torch.bfloat16, heads=4, dim=128, *, reserved=8256):
    # Sequence 0 has one live slot in page 14 and a -1 entry for page 15.
    args = make_paged_indexer_test_inputs(
        (2, 0, 1),
        (4 * 897, 4 * 1023 + 1, 4 * 17 + 1),
        tail=1,
        dtype=dtype,
        heads=heads,
        head_dim=dim,
    )
    weights = args[2]
    weights[:, 0] = 1
    weights[:, 1] = -2
    # Four query rows × 8256 slots is one candidate page beyond the routing threshold.
    table = args[3].new_full((args[3].shape[0], reserved // 64), -1)
    table[:, : args[3].shape[1]] = args[3]
    return (*args[:3], table, *args[4:])


@pytest.mark.parametrize("explicit", [False, True], ids=["suffix", "positions"])
def test_score_replay_after_repacking_queries(explicit, score_cache, triton_compiles):
    args = make_inputs(dtype=torch.float16 if explicit else torch.bfloat16)
    _, _, _, table, cu, lengths = args
    positions = cu.new_tensor([2**31 - 1, -1, 65, 0]) if explicit else None
    options = {"backend": "cute"} if explicit else None

    def select():
        return paged_lightning_indexer(
            *args, topk=512, query_positions=positions, kernel_options=options
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        warm = select()
    torch.cuda.current_stream().wait_stream(stream)
    assert_paged_indexer_selection(args, warm, 512, positions=positions)
    assert score_cache.cache_info().currsize > 0
    assert triton_compiles["paged_score_tma_kernel"] == triton_compiles["paged_score_kernel"] == 0

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = select()

    # Row 0 moves to sequence 1; row 2 leaves the live query prefix. The replay
    # also sees new physical page IDs and a different number of completed blocks.
    cu.copy_(cu.new_tensor([0, 0, 1, 2]))
    # 833 visible candidates span 13 full pages and one slot of the 14th.
    lengths[1] = 4 * 833 + 1
    table[1, :14] = table[1, :14].flip(0)
    table[1, 14:] = -1
    if explicit:
        positions[:2] = cu.new_tensor([4 * 587 - 1, 2**31 - 1])
    graph.replay()
    assert_paged_indexer_selection(args, selected, 512, positions=positions)


def test_compiled_score_finds_last_sequence_after_empty_sequences(score_cache):
    # For B=34 the first search probes odd sequence IDs. The last live sequence is
    # just beyond the final probe (32), so its length and end must be loaded separately.
    empty = (0,) * 32
    args = make_paged_indexer_test_inputs((1, *empty, 2), (15, *empty, 4 * 8193), tail=1)
    select = torch.compile(paged_lightning_indexer, fullgraph=True, dynamic=True)
    output = select(*args, topk=512)
    assert_paged_indexer_selection(args, output, 512)
    assert score_cache.cache_info().currsize == 1


def test_checked_score_slab_with_partial_last_page(requires_blackwell):
    """Exercise zero, one and 18 K-ring epochs; untouched score slots must remain NaN."""
    pytest.importorskip("cutlass.cute")
    from attn_gym.sparse.indexer.impl.paged_score_warpdrive import get_paged_score_runner
    from attn_gym.warpdrive.cache.target import compile_target_scope, detect_compile_target

    # More rows than resident CTAs forces each active CTA to process its row's entire tile
    # stream, including a second batch of shuffled page IDs after the first 16 tiles.
    q, keys, w, table, cu, lengths = make_paged_indexer_test_inputs(
        (1, 1, 1), (1, 33, 2241), compress_ratio=1, tail=1021
    )
    q.zero_()
    q[:, 0, 0] = 1
    w.fill_(-1)
    capacity = (table.shape[1] + 1) * 64
    table = torch.nn.functional.pad(table, (0, 1), value=-1)
    scores = torch.full((512, 2, capacity), torch.nan, device=q.device)
    bounds = cu.new_full((len(q), 2), -1)
    with compile_target_scope(detect_compile_target(q.device.index)):
        run = get_paged_score_runner(
            q, keys, w, table, cu, lengths, None, scores, bounds, 32, 1, checked=True
        )
    run(0, len(q))
    expected_bounds = torch.zeros_like(bounds)
    expected_bounds[:3, 1] = lengths
    torch.testing.assert_close(bounds, expected_bounds)
    expected = torch.full_like(scores, torch.nan).view(-1, capacity)
    for row, count in [(1, 33), (2, 2241)]:
        ids = torch.arange(count, device=q.device)
        logical_keys = keys[table[row, ids // 64].long(), ids % 64, 0].float()
        expected[row, :count] = -logical_keys.relu() * (4 * 128) ** -0.5
    torch.testing.assert_close(scores.view_as(expected), expected, equal_nan=True)


def test_compression_ratio_above_int32_yields_no_blocks(requires_blackwell):
    args = make_paged_indexer_test_inputs((1,), (1024,), compress_ratio=1)
    output = paged_lightning_indexer(*args, topk=512, compress_ratio=2**32 + 1)
    torch.testing.assert_close(output, torch.full_like(output, -1))


@pytest.mark.parametrize(
    "route",
    ["explicit-triton", "other-shape", "misaligned-query", "small-grid", "wide-table-stride"],
)
def test_triton_scoring_remains_available(route, score_cache, triton_compiles):
    heads, dim = (3, 16) if route == "other-shape" else (4, 128)
    args = list(
        make_inputs(heads=heads, dim=dim, reserved=8192 if route == "small-grid" else 8256)
    )
    if route == "misaligned-query":
        q = args[0]
        args[0] = q.new_empty(q.numel() + 1)[1:].view_as(q).copy_(q)
    elif route == "wide-table-stride":
        table = args[3][:1]
        args[3] = table.as_strided(table.shape, (2**31, 1))
        args[4], args[5] = args[4][:2], args[5][:1]
    backend = "triton" if route == "explicit-triton" else "cute"
    actual = paged_lightning_indexer(*args, topk=512, kernel_options={"backend": backend})
    assert_paged_indexer_selection(args, actual, 512)
    assert score_cache.cache_info().currsize == 0
    assert triton_compiles["paged_score_tma_kernel"] + triton_compiles["paged_score_kernel"] > 0
