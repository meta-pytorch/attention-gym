"""Batches input tokens into groups. Attention is only allowed within the same group."""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _mask_mod_signature, noop_mask
from attn_gym.masks import causal_mask


def batchify_mask_mod(mask_mod: _mask_mod_signature, batchify_size: int) -> _mask_mod_signature:
    """Given arbirary mask_mod, batchify it to only allow attention within the same batch.

    Args:
        mask_mod: The mask mod to apply within each batch.
        batchify_size: The number of tokens in each batch.
    """

    def batched_mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor):
        # Get the batch index of the query and key
        q_batch = q_idx // batchify_size
        kv_batch = kv_idx // batchify_size

        # Only allow attention within the same batch
        same_batch = q_batch == kv_batch

        # Apply the original mask mod
        inner_mask = mask_mod(b, h, q_idx % batchify_size, kv_idx % batchify_size)

        return same_batch & inner_mask

    batched_mask_mod.__name__ = f"batched_mask_mod_{mask_mod.__name__}_batch_size_{batchify_size}"
    return batched_mask_mod


def main(device: str = "cpu", causal: bool = False):
    """Visualize the attention scores of document causal mask mod.

    Args:
        device (str): Device to use for computation. Defaults to "cpu".
    """
    from attn_gym import visualize_attention_scores
    import random

    random.seed(0)

    seq_len, batchify_size = 12, 4
    B, H, SEQ_LEN, HEAD_DIM = 1, 1, seq_len, 8

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()
    if causal:
        base_mask_mod = causal_mask
    else:
        base_mask_mod = noop_mask

    batched_mask_mod = batchify_mask_mod(base_mask_mod, batchify_size)

    visualize_attention_scores(
        query,
        key,
        mask_mod=batched_mask_mod,
        device=device,
        name="batchify mask_mod",
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .[viz]")

    CLI(main)
