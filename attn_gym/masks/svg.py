"""
Generates a spatial attention mask and temporal attention mask, following the paper `Sparse VideoGen`.

Official Code: https://github.com/svg-project/Sparse-VideoGen
Official Paper: https://arxiv.org/abs/2502.01776
"""

from math import floor

import torch
from torch.nn.attention.flex_attention import _mask_mod_signature


def generate_spatial_head_mask_mod(
    prompt_length: int = 226,
    num_frames: int = 13,
    token_per_frame: int = 1350,
    width: int = 2,
    attn_sink: bool = False,
    round_width: int = 128,
) -> _mask_mod_signature:
    """
    Generates a spatial head mask as specified in SVG. The same mask can also be used for
    temporal attention mask after applying the layout transformation by setting `attn_sink` to False.

    Args:
        prompt_length: The length of the prompt.
        num_frames: The number of frames in the video.
        token_per_frame: The number of tokens per frame.
        width: The width of the spatial head mask, determine the number of frames that can be attended.
        attn_sink: Whether to use the attention sink for the first column.
        round_width: The number to round to for better hardware utilization, usually set to 128.
    """

    def round_to_multiple(idx):
        return floor(idx / round_width) * round_width

    def spatial_mask_mod(b, h, q_idx, kv_idx):
        first_row_mask = q_idx < prompt_length
        if attn_sink:
            first_column_mask = kv_idx < (prompt_length + token_per_frame)
        else:
            first_column_mask = kv_idx < prompt_length

        mask_width = round_to_multiple(width * token_per_frame)
        spatial_head_mask = torch.abs(q_idx - kv_idx) < mask_width
        return first_column_mask | first_row_mask | spatial_head_mask

    return spatial_mask_mod


def generate_temporal_head_mask_mod(
    prompt_length: int = 226,
    num_frames: int = 13,
    token_per_frame: int = 1350,
    width: int = 2,
) -> _mask_mod_signature:
    """
    Generates a temporal head mask as specified in SVG.

    Args:
        prompt_length: The length of the prompt.
        num_frames: The number of frames in the video.
        token_per_frame: The number of tokens per frame.
        width: The width of the temporal head mask, determine the number of frames that can be attended.
    """

    def get_token_id_in_frame(idx, prompt_length):
        return (idx - prompt_length) % token_per_frame

    def temporal_mask_mod(b, h, q_idx, kv_idx):
        first_row_mask = q_idx < prompt_length
        first_column_mask = kv_idx < prompt_length

        mask_width = (width * token_per_frame) // num_frames
        q_token_id = get_token_id_in_frame(q_idx, prompt_length)
        kv_token_id = get_token_id_in_frame(kv_idx, prompt_length)
        temporal_head_mask = torch.abs(q_token_id - kv_token_id) < mask_width
        return first_column_mask | first_row_mask | temporal_head_mask

    return temporal_mask_mod


def main(device: str = "cpu", causal: bool = True):
    """Visualize the attention scores of spatial head mask mod and temporal head mask mod.

    For reference on how to use a the sparse attention mask mod, checkout:
        1. https://github.com/svg-project/Sparse-VideoGen/blob/d0b2dfea4fd1e0069f0e6d0a42292649550e21af/svg/models/wan/utils.py#L23
        2. https://github.com/svg-project/Sparse-VideoGen/blob/d0b2dfea4fd1e0069f0e6d0a42292649550e21af/svg/models/wan/attention.py#L162

    Args:
        device (str): Device to use for computation. Defaults to "cpu".
    """
    from attn_gym import visualize_attention_scores
    import random

    random.seed(0)

    # Basic parameters for the input tensor
    PROMPT_LENGTH, NUM_FRAMES, TOKEN_PER_FRAME = 2, 10, 14
    B, H, SEQ_LEN, HEAD_DIM = 1, 1, PROMPT_LENGTH + NUM_FRAMES * TOKEN_PER_FRAME, 8

    # Parameters for the spatial head mask
    WIDTH = 2
    ROUND_WIDTH = 4  # Set to 4 for better visualization
    ATTN_SINK = True

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    spatial_head_mask = generate_spatial_head_mask_mod(
        prompt_length=PROMPT_LENGTH,
        num_frames=NUM_FRAMES,
        token_per_frame=TOKEN_PER_FRAME,
        width=WIDTH,
        attn_sink=ATTN_SINK,
        round_width=ROUND_WIDTH,
    )

    visualize_attention_scores(
        query,
        key,
        mask_mod=spatial_head_mask,
        device=device,
        name="svg_spatial_head_mask",
    )

    temporal_head_mask = generate_temporal_head_mask_mod(
        prompt_length=PROMPT_LENGTH,
        num_frames=NUM_FRAMES,
        token_per_frame=TOKEN_PER_FRAME,
        width=WIDTH,
    )

    visualize_attention_scores(
        query,
        key,
        mask_mod=temporal_head_mask,
        device=device,
        name="svg_temporal_head_mask",
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .[viz]")

    CLI(main)
