"""Paged indexer key updates: pooled candidates, raw rings, races, and capture."""

import pytest
import torch

from attn_gym.sparse import paged_indexer_key_update

CUDA = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
pytestmark = pytest.mark.skipif(not CUDA, reason="SM80+ required")


def _rope_cache(positions, rotary, theta=1e7):
    inv = 1.0 / theta ** (torch.arange(0, rotary, 2, dtype=torch.float32) / rotary)
    angles = torch.outer(torch.arange(positions, dtype=torch.float32), inv)
    angles = torch.cat([angles, angles], -1)
    return torch.cat([angles.cos(), angles.sin()], -1).cuda()


def _inputs(specs, *, page=8, ring=16, dim=128, rotary=64, padding=3, seed=0, dtype=None):
    dtype = dtype or torch.bfloat16
    generator = torch.Generator(device="cuda").manual_seed(seed)
    lengths = [length for length, _ in specs]
    queries = [count for _, count in specs]
    sequences = len(specs)
    cu = torch.tensor([0, *torch.tensor(queries).cumsum(0).tolist()], device="cuda")
    tokens = int(cu[-1]) + padding
    pages_per_sequence = -(-(max(lengths) // 4 + 1) // page)
    num_pages = sequences * pages_per_sequence + 2
    block_table = torch.randperm(num_pages, device="cuda", generator=generator)
    block_table = block_table[: sequences * pages_per_sequence].view(sequences, -1).int()
    raw_block_table = torch.randperm(sequences + 3, device="cuda", generator=generator)
    raw_block_table = raw_block_table[:sequences].view(sequences, 1).int()
    return [
        torch.randn(tokens, 2 * dim, device="cuda", dtype=dtype, generator=generator)[:, ::2],
        torch.randn(num_pages, page, dim, device="cuda", dtype=dtype, generator=generator),
        torch.randn(sequences + 3, ring, dim, device="cuda", dtype=dtype, generator=generator),
        block_table,
        raw_block_table,
        cu.int(),
        torch.tensor(lengths, device="cuda", dtype=torch.int32),
        (0.1 * torch.randn(dim, device="cuda", generator=generator)).to(dtype),
        _rope_cache(max(lengths) + 8, rotary),
    ]


def _run(args, **kwargs):
    caches = [args[1].clone(), args[2].clone()]
    paged_indexer_key_update(args[0], *caches, *args[3:], **kwargs)
    return caches


def _assert_matches_reference(args, **kwargs):
    expected = _run(args, impl="reference")
    actual = _run(args, **kwargs)
    torch.cuda.synchronize()
    for name, got, want, before in zip(("key_cache", "raw_cache"), actual, expected, args[1:3]):
        assert torch.equal(got, want), f"{name} differs from the reference"
    return expected


DECODE = [(4097, 1), (230, 1), (8, 1), (3, 1), (17, 1), (4, 1)]
MIXED = [(4097, 1), (210, 210), (1203, 400), (16, 16), (40, 33), (8, 1)]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "specs,max_seqlen_q",
    [(DECODE, 1), (DECODE, None), (MIXED, None), (MIXED, 400), ([(5, 5), (12, 9)], 9)],
)
def test_matches_reference(dtype, specs, max_seqlen_q):
    args = _inputs(specs, dtype=dtype)
    expected = _assert_matches_reference(args, max_seqlen_q=max_seqlen_q)
    assert not torch.equal(expected[0], args[1]) or all(
        (length - count + offset + 1) % 4 for length, count in specs for offset in range(count)
    )


def test_chunked_prefill_reads_the_ring():
    args = _inputs([(23, 6), (30, 2), (102, 1)], ring=16)
    _assert_matches_reference(args, max_seqlen_q=6)


def test_tail_only_rows_store_no_candidate():
    args = _inputs([(6, 1), (10, 2), (1, 1)])
    key_cache, raw_cache = _run(args)
    torch.cuda.synchronize()
    assert torch.equal(key_cache, args[1])
    assert not torch.equal(raw_cache, args[2])


def test_padding_rows_store_nothing():
    args = _inputs([(8, 1)], padding=40)
    args[0][1:] = torch.inf
    _assert_matches_reference(args, max_seqlen_q=1)


@pytest.mark.parametrize("count", [14, 15, 31, 64])
def test_ring_aliasing_reads_before_writes(count):
    args = _inputs([(count + 3, count), (count + 1, count)], ring=16)
    _assert_matches_reference(args)
    _assert_matches_reference(args, max_seqlen_q=count)


def test_candidate_page_boundaries():
    args = _inputs([(64, 64), (37, 30)], page=3)
    _assert_matches_reference(args, max_seqlen_q=64)


def test_cuda_graph_replay_with_changing_metadata():
    batches = [
        _inputs([(4097, 1), (8, 1), (300, 1), (16, 1)], seed=1),
        _inputs([(4097, 1), (12, 1), (299, 1), (3, 1)], seed=2),
    ]
    static = [tensor.clone() for tensor in batches[0]]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        paged_indexer_key_update(*static, max_seqlen_q=1)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        paged_indexer_key_update(*static, max_seqlen_q=1)
    for args in batches:
        expected = _run(args, impl="reference")
        for target, source in zip(static, args):
            target.copy_(source)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(static[1], expected[0])
        assert torch.equal(static[2], expected[1])


def test_compile_fullgraph():
    args = _inputs(MIXED)
    expected = _run(args, impl="reference")
    compiled = torch.compile(paged_indexer_key_update, fullgraph=True, dynamic=True)
    caches = [args[1].clone(), args[2].clone()]
    compiled(args[0], *caches, *args[3:])
    torch.cuda.synchronize()
    assert torch.equal(caches[0], expected[0])
    assert torch.equal(caches[1], expected[1])


def test_rejects_bad_inputs():
    args = _inputs(DECODE)
    with pytest.raises(ValueError, match="rope_cache"):
        paged_indexer_key_update(*args[:8], args[8][:, :-2])
    with pytest.raises(TypeError, match="int32"):
        paged_indexer_key_update(*args[:3], args[3].long(), *args[4:])
    with pytest.raises(ValueError, match="raw_cache"):
        paged_indexer_key_update(args[0], args[1], args[2][:, :2], *args[3:])
