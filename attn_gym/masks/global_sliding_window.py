"""Generates a Longformer-style global + sliding window attention mask.

Based on the Longformer paper: https://arxiv.org/abs/2004.05150
"""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature, or_masks


def generate_global_sliding_window(window_size: int, is_global: Tensor) -> _mask_mod_signature:
    """Generates a Longformer-style global + sliding window attention mask.

    Args:
        window_size: The symmetric sliding window radius; position i attends to
            positions j with abs(i - j) <= window_size.
        is_global: Boolean tensor of shape [SEQ_LEN] marking global tokens
            (e.g. CLS or task tokens). Global tokens attend to all positions and
            all positions attend to them.

    Note:
        Following the Longformer paper, attention is bidirectional: local
        attention is a window centered on each position, and global attention is
        symmetric. Compose with a causal mask via `and_masks` for decoder use.
    """

    def sliding_window(b, h, q_idx, kv_idx):
        return (q_idx - kv_idx).abs() <= window_size

    def global_attention(b, h, q_idx, kv_idx):
        return is_global[q_idx] | is_global[kv_idx]

    global_sliding_window_mask = or_masks(sliding_window, global_attention)
    global_sliding_window_mask.__name__ = f"global_sliding_window_{window_size}"
    return global_sliding_window_mask


def main(device: str = "cpu"):
    """Visualize the attention scores of the global + sliding window mask mod.

    Args:
        device (str): Device to use for computation.
    """
    from attn_gym import visualize_attention_scores

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 16, 8

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    is_global = torch.zeros(SEQ_LEN, dtype=torch.bool, device=device)
    is_global[0] = True
    is_global[8] = True

    global_sliding_window_mask = generate_global_sliding_window(3, is_global)
    visualize_attention_scores(
        query,
        key,
        mask_mod=global_sliding_window_mask,
        device=device,
        name="global_sliding_window_mask",
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
