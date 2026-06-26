import math

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from attn_gym.masks import (
    build_tree_ancestor_matrix,
    generate_jetspec_training_mask_mod,
    generate_jetspec_tree_causal_mask_mod,
)

CUDA_DEVICE = pytest.param(
    "cuda",
    marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
)
DEVICES = ["cpu", CUDA_DEVICE]
PARENT_INDICES = [-1, 0, 1, 1, 3, 3, 0]
EXPECTED_ANCESTOR = torch.tensor(
    [
        [True, False, False, False, False, False, False],
        [True, True, False, False, False, False, False],
        [True, True, True, False, False, False, False],
        [True, True, False, True, False, False, False],
        [True, True, False, True, True, False, False],
        [True, True, False, True, False, True, False],
        [True, False, False, False, False, False, True],
    ],
    dtype=torch.bool,
)
EXPECTED_DEPTHS = torch.tensor([0, 1, 2, 2, 3, 3, 1], dtype=torch.long)


def _validate_parent_indices(parent_indices):
    """Keep parent-order assertions in tests rather than the construction path."""
    assert parent_indices
    for node, parent in enumerate(parent_indices[1:], start=1):
        assert 0 <= parent < node


def _materialize_mask(mask_mod, num_queries, num_keys, device):
    """Materialize a rectangular mask for semantic checks."""
    q_idx = torch.arange(num_queries, device=device, dtype=torch.int32).view(num_queries, 1)
    kv_idx = torch.arange(num_keys, device=device, dtype=torch.int32).view(1, num_keys)
    return mask_mod(0, 0, q_idx, kv_idx)


def _dense_attention(query, key, value, mask):
    """Use the dense reference path for rectangular tree verification."""
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    return torch.softmax(torch.where(mask, scores, float("-inf")), dim=-1) @ value


def _expected_tree_mask(prefix_length, ancestor_matrix):
    """Build the JetSpec tree-causal matrix independently from the mask_mod closure."""
    return torch.cat(
        [
            torch.ones(
                ancestor_matrix.shape[0],
                prefix_length,
                dtype=torch.bool,
                device=ancestor_matrix.device,
            ),
            ancestor_matrix,
        ],
        dim=1,
    )


def _expected_training_mask(prefix_length, block_size, num_blocks, device):
    """Build the paper's multi-block training mask from the figure caption."""
    num_queries = num_blocks * block_size
    q_idx = torch.arange(num_queries, device=device).view(num_queries, 1)
    kv_idx = torch.arange(prefix_length + num_queries, device=device).view(1, -1)
    q_block = q_idx // block_size
    q_block_offset = q_idx % block_size
    kv_block_idx = kv_idx - prefix_length
    return (kv_idx < prefix_length) | (
        (q_block == kv_block_idx // block_size)
        & (kv_block_idx % block_size <= q_block_offset)
        & (kv_idx >= prefix_length)
    )


@pytest.mark.parametrize("device", DEVICES)
def test_build_tree_ancestor_matrix_matches_parent_paths(device):
    _validate_parent_indices(PARENT_INDICES)

    ancestor_matrix = build_tree_ancestor_matrix(PARENT_INDICES, device=device)

    assert torch.equal(ancestor_matrix, EXPECTED_ANCESTOR.to(device))
    assert torch.equal(
        ancestor_matrix.sum(dim=-1).to(dtype=torch.long) - 1,
        EXPECTED_DEPTHS.to(device),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_jetspec_tree_causal_mask_allows_prefix_and_tree_ancestors(device):
    prefix_length = 8
    ancestor_matrix = build_tree_ancestor_matrix(PARENT_INDICES, device=device)
    mask_mod = generate_jetspec_tree_causal_mask_mod(prefix_length, ancestor_matrix)

    assert torch.equal(
        _materialize_mask(
            mask_mod,
            ancestor_matrix.shape[0],
            prefix_length + ancestor_matrix.shape[0],
            device,
        ),
        _expected_tree_mask(prefix_length, ancestor_matrix),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prefix_length, block_size, num_blocks", [(5, 4, 3), (4096, 16, 3)])
def test_jetspec_training_mask_matches_paper_multi_block_pattern(
    device,
    prefix_length,
    block_size,
    num_blocks,
):
    mask_mod = generate_jetspec_training_mask_mod(prefix_length, block_size)

    assert torch.equal(
        _materialize_mask(
            mask_mod,
            num_blocks * block_size,
            prefix_length + num_blocks * block_size,
            device,
        ),
        _expected_training_mask(prefix_length, block_size, num_blocks, device),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_jetspec_tree_causal_mask_flex_attention_matches_dense_reference_cuda():
    torch.manual_seed(0)
    device = "cuda"
    prefix_length = 8
    ancestor_matrix = build_tree_ancestor_matrix(PARENT_INDICES, device=device)
    mask_mod = generate_jetspec_tree_causal_mask_mod(prefix_length, ancestor_matrix)
    num_tree_nodes = ancestor_matrix.shape[0]
    kv_seq_len = prefix_length + num_tree_nodes

    query = torch.randn(1, 2, num_tree_nodes, 16, device=device)
    key = torch.randn(1, 2, kv_seq_len, 16, device=device)
    value = torch.randn(1, 2, kv_seq_len, 16, device=device)
    block_mask = create_block_mask(mask_mod, 1, 2, num_tree_nodes, kv_seq_len, device=device)
    dense_mask = _materialize_mask(mask_mod, num_tree_nodes, kv_seq_len, device)

    torch.testing.assert_close(
        torch.compile(flex_attention)(query, key, value, block_mask=block_mask),
        _dense_attention(query, key, value, dense_mask.view(1, 1, num_tree_nodes, kv_seq_len)),
        atol=1e-4,
        rtol=1e-4,
    )


def test_jetspec_training_mask_rejects_negative_prefix_length():
    with pytest.raises(AssertionError, match="non-negative"):
        generate_jetspec_training_mask_mod(-1, 16)


def test_jetspec_training_mask_rejects_non_positive_block_size():
    with pytest.raises(AssertionError, match="positive"):
        generate_jetspec_training_mask_mod(4096, 0)
