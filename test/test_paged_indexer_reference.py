"""Cached-pool selection and the public decode contract on CPU."""

import pytest
import torch

from attn_gym.sparse import paged_lightning_indexer
from attn_gym.testing.indexer import (
    assert_paged_indexer_selection,
    make_paged_indexer_test_inputs,
)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ratio,topk", [(1, 0), (1, 7), (4, 17), (7, 1024)])
def test_paged_selection(dtype, ratio, topk):
    args = make_paged_indexer_test_inputs(
        compress_ratio=ratio,
        page_size=13,
        dtype=dtype,
        device="cpu",
        tail=2,
    )
    actual = paged_lightning_indexer(*args, topk=topk, compress_ratio=ratio, impl="reference")
    assert_paged_indexer_selection(args, actual, topk, ratio)


@pytest.mark.parametrize("permuted", [False, True])
def test_positions_layouts_and_nondifferentiability(permuted):
    args = list(make_paged_indexer_test_inputs(device="cpu"))
    if permuted:
        for i in (0, 1, 2, 3):
            args[i] = args[i].transpose(0, -1).contiguous().transpose(0, -1)
    for tensor in args[:3]:
        tensor.requires_grad_(True)
    positions = torch.tensor([2**31 - 1, -1, 3, 11, 23, 511], dtype=torch.int32)
    actual = paged_lightning_indexer(*args, topk=3, query_positions=positions, impl="reference")
    assert_paged_indexer_selection(args, actual, 3, positions=positions)


def test_deterministic_selection_prefers_lower_ties_and_orders_ids(deterministic_algorithms):
    args = make_paged_indexer_test_inputs(
        (1,), (8,), heads=1, head_dim=1, page_size=8, compress_ratio=1, device="cpu"
    )
    query, cache, weights = args[:3]
    query.fill_(1)
    weights.fill_(1)
    cache.zero_()
    cache[0, 7] = 1
    selected = paged_lightning_indexer(*args, topk=3, compress_ratio=1, impl="reference")
    assert selected.tolist() == [[0, 1, 7]]


def test_completed_block_boundaries():
    args = make_paged_indexer_test_inputs((3,), (10,), device="cpu")
    for length, expected_counts in [(9, [1, 2, 2]), (10, [2, 2, 2])]:
        args[-1].fill_(length)
        selected = paged_lightning_indexer(*args, topk=4, impl="reference")
        assert (selected >= 0).sum(1).tolist() == expected_counts
        assert_paged_indexer_selection(args, selected, 4)


def test_validation_and_empty_cpu():
    args = make_paged_indexer_test_inputs((0,), (0,), device="cpu")
    assert paged_lightning_indexer(*args, topk=4, impl="reference").shape == (0, 4)
    for kwargs, error in [
        ({"topk": True}, TypeError),
        ({"topk": -1}, ValueError),
        ({"topk": 1, "compress_ratio": 0}, ValueError),
    ]:
        with pytest.raises(error):
            paged_lightning_indexer(*args, impl="reference", **kwargs)


def test_metadata_layout_and_operand_types():
    args = list(make_paged_indexer_test_inputs(device="cpu"))
    cu = args[4]
    args[4] = cu.to(torch.int64)
    with pytest.raises(TypeError, match="cu_seqlens_q must have dtype int32"):
        paged_lightning_indexer(*args, topk=3, impl="reference")
    args[4] = torch.stack((cu, cu), dim=1)[:, 0]
    with pytest.raises(ValueError, match="cu_seqlens_q must be contiguous"):
        paged_lightning_indexer(*args, topk=3, impl="reference")
    args[4] = cu
    args[1] = args[1].float()
    with pytest.raises(TypeError, match="same dtype"):
        paged_lightning_indexer(*args, topk=3, impl="reference")


def test_zero_sequences_and_capacity_tail():
    args = list(make_paged_indexer_test_inputs((0,), (0,), tail=2, device="cpu"))
    args[3] = args[3][:0]
    args[4] = args[4][:1]
    args[5] = args[5][:0]
    actual = paged_lightning_indexer(*args, topk=3, impl="reference")
    assert actual.shape == (2, 3) and (actual == -1).all()
