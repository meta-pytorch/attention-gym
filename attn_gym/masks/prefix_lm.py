"""Generates a prefix LM causal attention mask"""

import torch
from torch.nn.attention.flex_attention import _mask_mod_signature, or_masks

from attn_gym.masks import causal_mask


def generate_prefix_lm_mask(prefix_length: int) -> _mask_mod_signature:
    """Generates a prefix LM causal attention mask.

    Args:
        prefix_length: The length of the prefix.

    Note:
        This mask allows full attention within the prefix (first PREFIX_LENGTH tokens)
        and causal attention for the rest of the sequence.
    """

    def prefix_mask(b, h, q_idx, kv_idx):
        return kv_idx < prefix_length

    prefix_lm_causal_mask = or_masks(prefix_mask, causal_mask)
    prefix_lm_causal_mask.__name__ = f"prefix_lm_causal_mask_{prefix_length}"
    return prefix_lm_causal_mask


def main(device: str = "cpu"):
    """Visualize the attention scores of prefix LM causal mask mod.

    Args:
        device (str): Device to use for computation. Defaults to "cpu".
    """
    from attn_gym import visualize_attention_scores

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 12, 8

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()
    prefix_lm_causal_mask = generate_prefix_lm_mask(prefix_length=4)

    visualize_attention_scores(
        query,
        key,
        mask_mod=prefix_lm_causal_mask,
        device=device,
        name="prefix_lm_causal_mask_length_4",
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .[viz]")

    CLI(main)
