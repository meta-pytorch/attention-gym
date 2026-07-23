"""Generates a STA mask"""

import torch
from torch import IntTensor, BoolTensor
from torch.nn.attention.flex_attention import _mask_mod_signature
from typing import Tuple


def generate_sta_mask_mod_2d(
    canvas_hw: Tuple[int, int],
    kernel_hw: Tuple[int, int],
    tile_hw: Tuple[int, int],
    text_seq_len: int = 0,
) -> _mask_mod_signature:
    """Generates a 2D STA mask with a given kernel size.

    Args:
        canvas_hw (Tuple[int, int]): The shape of the canvas (height, width).
        kernel_hw (Tuple[int, int]): The shape of the kernel (height, width).
        tile_hw (Tuple[int, int]): The shape of the tile (height, width).
        text_seq_len (int): The length of the text sequence for masking.
    """
    canvas_h, canvas_w = canvas_hw
    kernel_h, kernel_w = kernel_hw
    tile_h, tile_w = tile_hw
    tile_numel = tile_h * tile_w
    assert canvas_h % tile_h == 0, (
        f"Canvas height {canvas_h} is not divisible by tile height {tile_h}"
    )
    assert canvas_w % tile_w == 0, (
        f"Canvas width {canvas_w} is not divisible by tile width {tile_w}"
    )
    assert kernel_h % tile_h == 0, (
        f"Kernel height {kernel_h} is not divisible by tile height {tile_h}"
    )
    assert kernel_w % tile_w == 0, (
        f"Kernel width {kernel_w} is not divisible by tile width {tile_w}"
    )
    canvas_tile_h, canvas_tile_w = canvas_h // tile_h, canvas_w // tile_w
    kernel_tile_h, kernel_tile_w = kernel_h // tile_h, kernel_w // tile_w
    vision_seq_len = canvas_h * canvas_w

    def get_h_w_idx_tiled(idx: IntTensor) -> Tuple[IntTensor, IntTensor]:
        tile_id = idx // tile_numel
        tile_h_idx = tile_id // canvas_tile_w
        tile_w_idx = tile_id % canvas_tile_w
        return tile_h_idx, tile_w_idx

    def get_border(kernel_size: IntTensor) -> Tuple[IntTensor, IntTensor]:
        left_border = kernel_size // 2
        right_border = kernel_size // 2 + (kernel_size % 2 - 1)
        return left_border, right_border

    def sta_mask_mod_2d(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        q_tile_h, q_tile_w = get_h_w_idx_tiled(q_idx)
        kv_tile_h, kv_tile_w = get_h_w_idx_tiled(kv_idx)
        left_border_h, right_border_h = get_border(kernel_tile_h)
        left_border_w, right_border_w = get_border(kernel_tile_w)
        kernel_center_h = q_tile_h.clamp(left_border_h, (canvas_tile_h - 1) - right_border_h)
        kernel_center_w = q_tile_w.clamp(left_border_w, (canvas_tile_w - 1) - right_border_w)
        h_mask = (kv_tile_h >= kernel_center_h - left_border_h) & (
            kv_tile_h <= kernel_center_h + right_border_h
        )
        w_mask = (kv_tile_w >= kernel_center_w - left_border_w) & (
            kv_tile_w <= kernel_center_w + right_border_w
        )
        vision_mask = (q_idx < vision_seq_len) & (kv_idx < vision_seq_len)
        vision_to_text_mask = (
            (q_idx < vision_seq_len)
            & (kv_idx >= vision_seq_len)
            & (kv_idx < vision_seq_len + text_seq_len)
        )
        text_to_all_mask = (q_idx >= vision_seq_len) & (kv_idx < vision_seq_len + text_seq_len)
        return (vision_mask & h_mask & w_mask) | vision_to_text_mask | text_to_all_mask

    sta_mask_mod_2d.__name__ = (
        f"sta_2d_c{canvas_h}x{canvas_w}_k{kernel_h}x{kernel_w}_t{tile_h}x{tile_w}"
    )
    return sta_mask_mod_2d


def generate_sta_mask_mod_3d(
    canvas_twh: Tuple[int, int, int],
    kernel_twh: Tuple[int, int, int],
    tile_twh: Tuple[int, int, int],
    text_seq_len: int = 0,
) -> _mask_mod_signature:
    """Generates a 3D STA mask with a given kernel size.

    Args:
        canvas_twh (Tuple[int, int, int]): The shape of the canvas (time, height, width).
        kernel_twh (Tuple[int, int, int]): The shape of the kernel (time, height, width).
        tile_twh (Tuple[int, int, int]): The shape of the tile (time, height, width).
        text_seq_len (int): The length of the text sequence for masking.
    """
    canvas_t, canvas_h, canvas_w = canvas_twh
    kernel_t, kernel_h, kernel_w = kernel_twh
    tile_t, tile_h, tile_w = tile_twh
    tile_numel = tile_t * tile_h * tile_w
    assert canvas_t % tile_t == 0, f"Canvas time {canvas_t} is not divisible by tile time {tile_t}"
    assert canvas_h % tile_h == 0, (
        f"Canvas height {canvas_h} is not divisible by tile height {tile_h}"
    )
    assert canvas_w % tile_w == 0, (
        f"Canvas width {canvas_w} is not divisible by tile width {tile_w}"
    )
    assert kernel_t % tile_t == 0, f"Kernel time {kernel_t} is not divisible by tile time {tile_t}"
    assert kernel_h % tile_h == 0, (
        f"Kernel height {kernel_h} is not divisible by tile height {tile_h}"
    )
    assert kernel_w % tile_w == 0, (
        f"Kernel width {kernel_w} is not divisible by tile width {tile_w}"
    )
    canvas_tile_t, canvas_tile_h, canvas_tile_w = (
        canvas_t // tile_t,
        canvas_h // tile_h,
        canvas_w // tile_w,
    )
    kernel_tile_t, kernel_tile_h, kernel_tile_w = (
        kernel_t // tile_t,
        kernel_h // tile_h,
        kernel_w // tile_w,
    )
    vision_seq_len = canvas_t * canvas_h * canvas_w

    def get_t_h_w_idx_tiled(idx: IntTensor) -> Tuple[IntTensor, IntTensor, IntTensor]:
        tile_id = idx // tile_numel
        tile_t_idx = tile_id // (canvas_tile_h * canvas_tile_w)
        tile_h_idx = (tile_id % (canvas_tile_h * canvas_tile_w)) // canvas_tile_w
        tile_w_idx = tile_id % canvas_tile_w
        return tile_t_idx, tile_h_idx, tile_w_idx

    def get_border(kernel_size: IntTensor) -> Tuple[IntTensor, IntTensor]:
        left_border = kernel_size // 2
        right_border = kernel_size // 2 + (kernel_size % 2 - 1)
        return left_border, right_border

    def sta_mask_mod_3d(
        b: IntTensor,
        h: IntTensor,
        q_idx: IntTensor,
        kv_idx: IntTensor,
    ) -> BoolTensor:
        q_tile_t, q_tile_h, q_tile_w = get_t_h_w_idx_tiled(q_idx)
        kv_tile_t, kv_tile_h, kv_tile_w = get_t_h_w_idx_tiled(kv_idx)
        left_border_t, right_border_t = get_border(kernel_tile_t)
        left_border_h, right_border_h = get_border(kernel_tile_h)
        left_border_w, right_border_w = get_border(kernel_tile_w)
        kernel_center_t = q_tile_t.clamp(left_border_t, (canvas_tile_t - 1) - right_border_t)
        kernel_center_h = q_tile_h.clamp(left_border_h, (canvas_tile_h - 1) - right_border_h)
        kernel_center_w = q_tile_w.clamp(left_border_w, (canvas_tile_w - 1) - right_border_w)
        t_mask = (kv_tile_t >= kernel_center_t - left_border_t) & (
            kv_tile_t <= kernel_center_t + right_border_t
        )
        h_mask = (kv_tile_h >= kernel_center_h - left_border_h) & (
            kv_tile_h <= kernel_center_h + right_border_h
        )
        w_mask = (kv_tile_w >= kernel_center_w - left_border_w) & (
            kv_tile_w <= kernel_center_w + right_border_w
        )
        vision_mask = (q_idx < vision_seq_len) & (kv_idx < vision_seq_len)
        vision_to_text_mask = (
            (q_idx < vision_seq_len)
            & (kv_idx >= vision_seq_len)
            & (kv_idx < vision_seq_len + text_seq_len)
        )
        text_to_all_mask = (q_idx >= vision_seq_len) & (kv_idx < vision_seq_len + text_seq_len)
        return (vision_mask & t_mask & w_mask & h_mask) | vision_to_text_mask | text_to_all_mask

    sta_mask_mod_3d.__name__ = f"sta_3d_c{canvas_t}x{canvas_h}x{canvas_w}_k{kernel_t}x{kernel_h}x{kernel_w}_t{tile_t}x{tile_h}x{tile_w}"
    return sta_mask_mod_3d


def main(device: str = "cpu"):
    """Visualize the attention scores of STA mask mod.
    Original repo: https://github.com/hao-ai-lab/FastVideo
    See blog: https://hao-ai-lab.github.io/blogs/sta/
    For reference on how to use a Sliding Tile Attention (STA) module, checkout:
        1, https://github.com/hao-ai-lab/FastVideo/blob/6ef8fcb61d5046d22b51a6ef5ef312731cef503d/fastvideo/v1/attention/backends/sliding_tile_attn.py#L105
        2, https://github.com/fla-org/fla-zoo/blob/main/flazoo/models/attentions.py#L702

    Note that this version alters some of the original code for better readability and include a 2d use case.
    Args:
        device (str): Device to use for computation. Defaults
    """
    from attn_gym import visualize_attention_scores

    B, H, CANVAS_TIME, CANVAS_HEIGHT, CANVAS_WIDTH, HEAD_DIM = 1, 1, 24, 24, 24, 8
    KERNEL_T, KERNEL_H, KERNEL_W = 12, 12, 12
    TILE_T, TILE_H, TILE_W = 4, 4, 4

    def make_tensor():
        return torch.ones(B, H, CANVAS_HEIGHT, CANVAS_WIDTH, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    sta_mask_2d = generate_sta_mask_mod_2d(
        canvas_hw=(CANVAS_HEIGHT, CANVAS_WIDTH),
        kernel_hw=(KERNEL_H, KERNEL_W),
        tile_hw=(TILE_H, TILE_W),
    )

    visualize_attention_scores(
        query.flatten(start_dim=2, end_dim=3),
        key.flatten(start_dim=2, end_dim=3),
        mask_mod=sta_mask_2d,
        device=device,
        name=sta_mask_2d.__name__,
    )

    def make_3d_tensor():
        return torch.ones(B, H, CANVAS_TIME, CANVAS_HEIGHT, CANVAS_WIDTH, HEAD_DIM, device=device)

    query_3d, key_3d = make_3d_tensor(), make_3d_tensor()

    sta_mask_3d = generate_sta_mask_mod_3d(
        canvas_twh=(CANVAS_TIME, CANVAS_HEIGHT, CANVAS_WIDTH),
        kernel_twh=(KERNEL_T, KERNEL_H, KERNEL_W),
        tile_twh=(TILE_T, TILE_H, TILE_W),
    )

    visualize_attention_scores(
        query_3d.flatten(start_dim=2, end_dim=4),
        key_3d.flatten(start_dim=2, end_dim=4),
        mask_mod=sta_mask_3d,
        device=device,
        name=sta_mask_3d.__name__,
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
