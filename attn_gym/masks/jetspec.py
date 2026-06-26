"""Generates JetSpec attention masks from the paper.

Official paper: https://arxiv.org/abs/2606.18394
Official code: https://github.com/hao-ai-lab/JetSpec
"""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature


def build_tree_ancestor_matrix(
    parent_indices: Sequence[int],
    device: str | torch.device = "cuda",
) -> Tensor:
    """Build the ancestor matrix for the paper's tree-causal mask.

    Args:
        parent_indices: Parent index for each flattened tree node in parent-before-child order.
            The root entry is ignored and is conventionally ``-1``.
        device: Output device.

    Returns:
        A boolean tensor where ``ancestor[i, j]`` is true when node ``j`` is on node
        ``i``'s root-to-self path.
    """
    ancestor_matrix = torch.eye(len(parent_indices), dtype=torch.bool)
    for node, parent in enumerate(parent_indices[1:], start=1):
        ancestor_matrix[node] |= ancestor_matrix[parent]
    return ancestor_matrix.to(device=device)


def generate_jetspec_tree_causal_mask_mod(
    prefix_length: int,
    ancestor_matrix: Tensor,
) -> _mask_mod_signature:
    """Generate the paper's tree-causal mask for parallel tree drafting/verification.

    Args:
        prefix_length: Number of prefix keys before the flattened tree keys.
        ancestor_matrix: Boolean ``(num_tree_nodes, num_tree_nodes)`` tensor. Entry
            ``[i, j]`` is true when tree node ``j`` is an ancestor of query node ``i``,
            including self.

    Returns:
        A ``mask_mod`` for ``Q_LEN=num_tree_nodes`` and
        ``KV_LEN=prefix_length + num_tree_nodes``.
    """
    assert prefix_length >= 0, "prefix_length must be non-negative"
    assert ancestor_matrix.ndim == 2, "ancestor_matrix must be 2D"
    assert ancestor_matrix.shape[0] == ancestor_matrix.shape[1], "ancestor_matrix must be square"
    num_tree_nodes = ancestor_matrix.shape[0]
    assert num_tree_nodes > 0, "ancestor_matrix must describe at least one tree node"
    ancestor_matrix = ancestor_matrix.to(dtype=torch.bool)
    tree_stop = prefix_length + num_tree_nodes

    def jetspec_tree_causal_mask_mod(b, h, q_idx, kv_idx):
        tree_idx = (kv_idx - prefix_length).clamp(min=0, max=num_tree_nodes - 1)
        tree_visible = (kv_idx >= prefix_length) & (kv_idx < tree_stop)
        return (kv_idx < prefix_length) | (tree_visible & ancestor_matrix[q_idx, tree_idx])

    jetspec_tree_causal_mask_mod.__name__ = (
        f"jetspec_tree_causal_mask_p{prefix_length}_n{num_tree_nodes}"
    )
    return jetspec_tree_causal_mask_mod


def generate_jetspec_training_mask_mod(
    prefix_length: int,
    block_size: int,
) -> _mask_mod_signature:
    """Generate the paper's multi-block causal draft-head training mask.

    Query rows are sampled-block tokens. KV columns are ``prefix`` followed by one or more
    sampled blocks, each laid out as ``[anchor, future_1, ..., future_N]``. Each query can
    attend to the full verified prefix and to the anchor plus earlier positions in its own
    block, but not to future positions or other sampled blocks.

    Args:
        prefix_length: Number of verified prefix keys before sampled training blocks.
        block_size: Number of tokens in each sampled block, including the anchor.

    Returns:
        A ``mask_mod`` for ``Q_LEN=num_blocks * block_size`` and
        ``KV_LEN=prefix_length + num_blocks * block_size``.
    """
    assert prefix_length >= 0, "prefix_length must be non-negative"
    assert block_size > 0, "block_size must be positive"

    def jetspec_training_mask_mod(b, h, q_idx, kv_idx):
        q_block = q_idx // block_size
        q_block_offset = q_idx % block_size
        kv_block_idx = kv_idx - prefix_length
        kv_block = kv_block_idx // block_size
        kv_block_offset = kv_block_idx % block_size
        same_block_causal = (
            (kv_idx >= prefix_length) & (q_block == kv_block) & (kv_block_offset <= q_block_offset)
        )
        return (kv_idx < prefix_length) | same_block_causal

    jetspec_training_mask_mod.__name__ = (
        f"jetspec_training_mask_p{prefix_length}_block{block_size}"
    )
    return jetspec_training_mask_mod


def main(device: str = "cuda"):
    """Visualize the two JetSpec masks defined in the paper.

    The tree demo follows Figure 3's order: ``return, a, -, +, B, b, sum``.

    Args:
        device: Device to use for computation. Defaults to ``"cuda"``.
    """
    from attn_gym import visualize_attention_scores

    B, H, HEAD_DIM = 1, 1, 8

    parent_indices = [-1, 0, 1, 1, 3, 3, 0]
    tree_prefix_length = 0
    ancestor_matrix = build_tree_ancestor_matrix(parent_indices, device=device)
    tree_mask_mod = generate_jetspec_tree_causal_mask_mod(tree_prefix_length, ancestor_matrix)
    tree_nodes = ancestor_matrix.shape[0]
    visualize_attention_scores(
        torch.ones(B, H, tree_nodes, HEAD_DIM, device=device),
        torch.ones(B, H, tree_prefix_length + tree_nodes, HEAD_DIM, device=device),
        mask_mod=tree_mask_mod,
        device=device,
        name=tree_mask_mod.__name__,
    )

    training_prefix_length = 5
    block_size = 4
    num_blocks = 3
    training_mask_mod = generate_jetspec_training_mask_mod(training_prefix_length, block_size)
    training_queries = num_blocks * block_size
    training_keys = training_prefix_length + training_queries
    visualize_attention_scores(
        torch.ones(B, H, training_queries, HEAD_DIM, device=device),
        torch.ones(B, H, training_keys, HEAD_DIM, device=device),
        mask_mod=training_mask_mod,
        device=device,
        name=training_mask_mod.__name__,
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
