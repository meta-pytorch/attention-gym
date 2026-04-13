"""Generates a sliding window attention mask"""

import torch
from torch.nn.attention.flex_attention import _mask_mod_signature, and_masks, or_masks
from attn_gym.masks import causal_mask

def generate_sink_mask(window_size: int, sink_size: int = 4) -> _mask_mod_signature:
    """Generates a sliding window with sink attention mask.
    
    Args:
        window_size: The size of the sliding window.
        sink_size: The number of initial tokens that are always visible (sink tokens). Defaults to 4.

    Note:
        We assume that the window size represents the lookback size and we mask out all future tokens
        similar to causal masking, but additionally all tokens can attend to the first `sink_size` tokens.
    """

    def sink_mask(b, h, q_idx, kv_idx):
        # The sink tokens: the first `sink_size` tokens are always visible
        return kv_idx < sink_size 

    def sliding_window(b, h, q_idx, kv_idx):
        # The sliding window constraint: within the window
        return q_idx - kv_idx <= window_size

    # Combine: (sliding window OR sink) AND causal
    combined_mask = and_masks(
        or_masks(sliding_window, sink_mask), 
        causal_mask
    )
    
    combined_mask.__name__ = f"sink_window_{window_size}_sink_{sink_size}"
    return combined_mask


def main(device: str = "cpu", mask_type: str = "sink", window_size: int = 3, sink_size: int = 4):
    """Visualize the attention scores of sink mask.

    Args:
        device: Device to use for computation. Defaults to "cpu".
        mask_type: Type of mask to use (only "sink" is supported). Defaults to "sink".
        window_size: The size of the sliding window. Defaults to 3.
        sink_size: The number of initial tokens that are always visible (sink tokens). Defaults to 4.
    """
    from attn_gym import visualize_attention_scores

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 12, 8

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    if mask_type != "sink":
        raise ValueError("This module only supports 'sink' mask type")
        
    mask_mod = generate_sink_mask(window_size, sink_size)

    visualize_attention_scores(
        query, key, mask_mod=mask_mod, device=device, name=mask_mod.__name__
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
