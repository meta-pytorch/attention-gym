"""Generates the Block Diffusion hybrid attention mask.

Based on the Block Diffusion (BD3-LM) paper: https://arxiv.org/abs/2503.09573
"""

import torch
from torch.nn.attention.flex_attention import _mask_mod_signature


def generate_block_diffusion_mask(seq_len: int, block_size: int) -> _mask_mod_signature:
    """Generates the Block Diffusion training mask from BD3-LM section 3.1.

    Training runs attention over a length ``2 * seq_len`` sequence: the noised
    tokens ``x_t`` occupy positions ``[0, seq_len)`` and the clean tokens ``x_0``
    occupy positions ``[seq_len, 2 * seq_len)``. The mask is the union of three
    pieces:

    - Block Diagonal: noised tokens attend bidirectionally within their own block.
    - Offset Block Causal: noised tokens attend to clean tokens in strictly
      previous blocks.
    - Block Causal: clean tokens attend to clean tokens in their own and previous
      blocks.

    Clean tokens never attend to noised tokens.

    Args:
        seq_len: Length of the clean sequence ``x_0``. The returned mask_mod is
            defined over indices in ``[0, 2 * seq_len)``.
        block_size: Diffusion block size ``L'``; the sequence is partitioned into
            contiguous blocks of this many tokens.
    """

    def block_diffusion_mask(b, h, q_idx, kv_idx):
        q_noised = q_idx < seq_len
        kv_noised = kv_idx < seq_len
        q_block = (q_idx % seq_len) // block_size
        kv_block = (kv_idx % seq_len) // block_size

        block_diagonal = (q_block == kv_block) & (q_noised == kv_noised)
        offset_block_causal = (q_block > kv_block) & q_noised & ~kv_noised
        block_causal = (q_block >= kv_block) & ~q_noised & ~kv_noised
        return block_diagonal | offset_block_causal | block_causal

    block_diffusion_mask.__name__ = f"block_diffusion_{seq_len}_{block_size}"
    return block_diffusion_mask


def main(device: str = "cpu"):
    """Visualize the attention scores of the block diffusion mask mod.

    Args:
        device (str): Device to use for computation.
    """
    from attn_gym import visualize_attention_scores

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 12, 8

    def make_tensor():
        return torch.ones(B, H, 2 * SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    block_diffusion_mask = generate_block_diffusion_mask(SEQ_LEN, block_size=4)
    visualize_attention_scores(
        query,
        key,
        mask_mod=block_diffusion_mask,
        device=device,
        name="block_diffusion_mask",
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
