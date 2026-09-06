"""CPU contracts for the examples' Zipf sampling and user-editable rank mappings."""

from itertools import pairwise

import pytest
import torch

pytest.importorskip("cutlass", reason="the example modules import their optional fused kernels")
pytest.importorskip("typer")

from examples.delta_rule_context_parallel import partition_fragments
from examples.delta_rule_training import packed_sequence_metadata


@pytest.mark.parametrize("seed", [0, 7, 123])
@pytest.mark.parametrize("num_sequences,max_tokens", [(1, 1), (4, 63), (32, 257)])
def test_packed_sequence_metadata_matches_original_sampler(seed, num_sequences, max_tokens):
    """Moving the sampler must preserve samples and consumption of the global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        weights = torch.arange(1, max_tokens + 1, dtype=torch.float64).reciprocal()
        expected = tuple(
            torch.multinomial(weights, num_sequences, replacement=True).add(1).tolist()
        )
        expected_next = torch.rand(8)

        torch.manual_seed(seed)
        lengths, offsets = packed_sequence_metadata(num_sequences, max_tokens)
        actual_next = torch.rand(8)

    assert lengths == expected
    assert offsets == tuple(sum(expected[:index]) for index in range(num_sequences + 1))
    torch.testing.assert_close(actual_next, expected_next, rtol=0, atol=0)


@pytest.mark.parametrize("max_tokens", [1, 2, 31])
def test_packed_sequence_metadata_bounds_and_offsets(max_tokens):
    """Lengths stay in the truncated support and offsets describe every sampled token."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        lengths, offsets = packed_sequence_metadata(64, max_tokens)
    assert isinstance(lengths, tuple) and isinstance(offsets, tuple)
    assert len(lengths) == 64
    assert len(offsets) == 65
    assert all(isinstance(length, int) and 1 <= length <= max_tokens for length in lengths)
    assert offsets[0] == 0
    assert offsets[-1] == sum(lengths)
    assert tuple(end - start for start, end in pairwise(offsets)) == lengths


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_packed_sequence_metadata_rejects_empty_support(max_tokens):
    with pytest.raises(ValueError, match="at least one token"):
        packed_sequence_metadata(4, max_tokens)


@pytest.mark.parametrize("partition", ["contiguous", "zigzag"])
@pytest.mark.parametrize("world_size", [1, 2, 3, 7])
def test_partition_fragments_preserves_divisible_boundaries(partition, world_size):
    """Divisible totals retain the original CP example's exact block ownership."""
    blocks = world_size if partition == "contiguous" else 2 * world_size
    size = 11
    expected = []
    for rank in range(world_size):
        ranges = [(rank * size, (rank + 1) * size)]
        if partition == "zigzag":
            ranges.append(((blocks - 1 - rank) * size, (blocks - rank) * size))
        expected.append(ranges)
    assert partition_fragments(blocks * size, world_size, partition) == expected


@pytest.mark.parametrize(
    "partition,expected",
    [
        ("contiguous", [[(0, 4)], [(4, 8)], [(8, 13)]]),
        ("zigzag", [[(0, 2), (10, 13)], [(2, 4), (8, 10)], [(4, 6), (6, 8)]]),
    ],
)
def test_partition_fragments_odd_total_boundaries(partition, expected):
    assert partition_fragments(13, 3, partition) == expected


@pytest.mark.parametrize("partition", ["contiguous", "zigzag"])
@pytest.mark.parametrize("world_size", [1, 2, 3, 7])
@pytest.mark.parametrize("extra_tokens", [0, 1, 5, 16])
def test_partition_fragments_tiles_all_tokens_in_local_order(partition, world_size, extra_tokens):
    """Near-balanced blocks tile exactly, including minimal and non-divisible totals."""
    blocks = world_size if partition == "contiguous" else 2 * world_size
    tokens = blocks + extra_tokens
    fragments = partition_fragments(tokens, world_size, partition)
    assert len(fragments) == world_size
    assert all(len(ranges) == blocks // world_size for ranges in fragments)
    all_tokens = []
    rank_sizes = []
    block_sizes = []
    for ranges in fragments:
        local_tokens = [token for start, end in ranges for token in range(start, end)]
        assert all(0 <= start < end <= tokens for start, end in ranges)
        assert all(left < right for left, right in pairwise(local_tokens))
        all_tokens.extend(local_tokens)
        rank_sizes.append(len(local_tokens))
        block_sizes.extend(end - start for start, end in ranges)
    assert sorted(all_tokens) == list(range(tokens))
    assert max(block_sizes) - min(block_sizes) <= 1
    assert max(rank_sizes) - min(rank_sizes) <= (1 if partition == "contiguous" else 2)


def test_partition_fragments_defaults_to_contiguous():
    assert partition_fragments(13, 3) == [[(0, 4)], [(4, 8)], [(8, 13)]]


@pytest.mark.parametrize(
    "tokens,world_size,partition,message",
    [
        (13, 0, "contiguous", "world_size must be positive"),
        (13, -1, "zigzag", "world_size must be positive"),
        (13, 3, "unknown", "partition must be"),
        (0, 1, "contiguous", "nonempty blocks"),
        (-1, 1, "zigzag", "nonempty blocks"),
        (2, 3, "contiguous", "nonempty blocks"),
        (5, 3, "zigzag", "nonempty blocks"),
    ],
)
def test_partition_fragments_rejects_invalid_inputs(tokens, world_size, partition, message):
    with pytest.raises(ValueError, match=message):
        partition_fragments(tokens, world_size, partition)
