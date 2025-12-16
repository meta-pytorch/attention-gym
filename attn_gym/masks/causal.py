"""Standard Causal Attention Masking."""

import torch
from torch.nn.attention.flex_attention import BlockMask
from attn_gym.utils import cdiv


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def create_causal_block_mask_fast(
    batch_size: int | None,
    num_heads: int | None,
    q_seq_len: int,
    kv_seq_len: int,
    device: torch.device,
    block_size: int = 128,
    separate_full_blocks: bool = True,
) -> BlockMask:
    """Create a causal block mask efficiently without materializing the full mask.

    This function generates the block mask data structure directly for causal attention,
    avoiding the need to create and process a full dense mask. This is much more efficient
    for long sequences.

    Args:
        q_seq_len: Query sequence length
        kv_seq_len: Key/value sequence length
        device: Device to create tensors on
        batch_size: Batch size (defaults to 1 if None)
        num_heads: Number of attention heads (defaults to 1 if None)
        block_size: Block size for the block mask (both Q and KV use same size)
        separate_full_blocks: Whether to separate full blocks from partial blocks

    Returns:
        BlockMask: Block mask object for causal attention
    """
    if batch_size is None:
        batch_size = 1
    if num_heads is None:
        num_heads = 1
    if isinstance(block_size, tuple):
        q_block_size, kv_block_size = block_size
    else:
        q_block_size = kv_block_size = block_size

    num_q_blocks = cdiv(q_seq_len, q_block_size)
    num_kv_blocks = cdiv(kv_seq_len, kv_block_size)

    q_block_indices = torch.arange(num_q_blocks, device=device, dtype=torch.int32)
    kv_block_indices = torch.arange(num_kv_blocks, device=device, dtype=torch.int32)

    num_full_per_row = torch.clamp(q_block_indices, max=num_kv_blocks)
    has_partial = q_block_indices < num_kv_blocks

    if separate_full_blocks:
        min_q_indices = q_block_indices * q_block_size
        max_kv_indices = torch.clamp((q_block_indices + 1) * kv_block_size - 1, max=kv_seq_len - 1)
        is_diagonal_full = has_partial & (min_q_indices >= max_kv_indices)

        full_counts = torch.where(is_diagonal_full, num_full_per_row + 1, num_full_per_row)
        partial_counts = torch.where(is_diagonal_full, 0, has_partial.int())

        full_kv_num_blocks = (
            full_counts.view(1, 1, num_q_blocks)
            .expand(batch_size, num_heads, num_q_blocks)
            .contiguous()
        )
        kv_num_blocks = (
            partial_counts.view(1, 1, num_q_blocks)
            .expand(batch_size, num_heads, num_q_blocks)
            .contiguous()
        )

        full_mask = kv_block_indices.unsqueeze(0) < full_counts.unsqueeze(1)
        full_kv_indices = (
            torch.where(
                full_mask, kv_block_indices.unsqueeze(0), torch.zeros_like(kv_block_indices)
            )
            .view(1, 1, num_q_blocks, num_kv_blocks)
            .expand(batch_size, num_heads, num_q_blocks, num_kv_blocks)
            .contiguous()
        )

        partial_indices = torch.where(
            partial_counts > 0, q_block_indices, torch.zeros_like(q_block_indices)
        )
        kv_indices = torch.zeros(
            (batch_size, num_heads, num_q_blocks, num_kv_blocks), dtype=torch.int32, device=device
        )
        kv_indices[:, :, :, 0] = partial_indices.view(1, 1, num_q_blocks).expand(
            batch_size, num_heads, num_q_blocks
        )
    else:
        full_kv_num_blocks = None
        full_kv_indices = None

        total_counts = num_full_per_row + has_partial.int()
        kv_num_blocks = (
            total_counts.view(1, 1, num_q_blocks)
            .expand(batch_size, num_heads, num_q_blocks)
            .contiguous()
        )

        mask = kv_block_indices.unsqueeze(0) < total_counts.unsqueeze(1)
        kv_indices = (
            torch.where(mask, kv_block_indices.unsqueeze(0), torch.zeros_like(kv_block_indices))
            .view(1, 1, num_q_blocks, num_kv_blocks)
            .expand(batch_size, num_heads, num_q_blocks, num_kv_blocks)
            .contiguous()
        )

    return BlockMask.from_kv_blocks(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        full_kv_num_blocks=full_kv_num_blocks,
        full_kv_indices=full_kv_indices,
        BLOCK_SIZE=(q_block_size, kv_block_size),
        mask_mod=causal_mask,
        seq_lengths=(q_seq_len, kv_seq_len),
    )


def main(device: str = "cpu"):
    """Visualize the attention scores of causal masking.

    Args:
        device (str): Device to use for computation. Defaults
    """
    import torch
    from attn_gym import visualize_attention_scores

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 12, 8

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    visualize_attention_scores(query, key, mask_mod=causal_mask, device=device, name="causal_mask")


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
