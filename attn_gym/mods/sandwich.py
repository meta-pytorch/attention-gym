"""Implementation of the Sandwich score mod from the paper Dissecting Transformer Length
Extrapolation via the Lens of Receptive Field Analysis: https://arxiv.org/abs/2212.10356"""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _score_mod_signature


def generate_sandwich_bias(
    H: int, max_seq_len: int, d_bar: int = 128, device: str | torch.device = "cpu"
) -> _score_mod_signature:
    """Returns a Sandwich bias score_mod.

    Sandwich keeps only the position-position inner product of sinusoidal embeddings:
    bias[h, m, n] = sum_i cos((m - n) * w_i) / ratio_h with w_i = 1/10000^(2i/d_bar)
    and ALiBi-style per-head compression ratios ratio_h = 8(h+1)/H. The bias is
    precomputed as a 1-D relative-distance table, padded to a multiple of 8 for
    vector-load-friendly indexing, and the per-head ratio is applied via a small
    lookup table (uniform loads let the FLASH backend vectorize the score_mod).

    Args:
        H: number of heads.
        max_seq_len: maximum sequence length the bias will be used with.
        d_bar: Sandwich shape hyperparameter (paper uses 128).
        device: device for the precomputed bias table.

    Returns:
        sandwich_bias: sandwich bias score_mod
    """
    freqs = 1.0 / 10000 ** (
        2 * torch.arange(d_bar // 2, device=device, dtype=torch.float32) / d_bar
    )
    padded_len = -(-(2 * max_seq_len - 1) // 8) * 8
    distances = torch.arange(padded_len, device=device, dtype=torch.float32) - (max_seq_len - 1)
    rel_bias = torch.cos(distances.abs()[:, None] * freqs).sum(dim=-1)
    offset = max_seq_len - 1
    head_scale = H / (8.0 * torch.arange(1, H + 1, device=device, dtype=torch.float32))

    def sandwich_mod(score: Tensor, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor):
        return score + rel_bias[q_idx - kv_idx + offset] * head_scale[h]

    return sandwich_mod


def main(device: str = "cpu", causal: bool = True):
    """Visualize the attention scores of the sandwich bias score mod.

    Args:
        device (str): Device to use for computation.
        causal (bool): Whether to combine with a causal mask.
    """
    from attn_gym import visualize_attention_scores

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 12, 8

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    sandwich_score_mod = generate_sandwich_bias(H, SEQ_LEN, device=device)

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    visualize_attention_scores(
        query,
        key,
        score_mod=sandwich_score_mod,
        mask_mod=causal_mask if causal else None,
        device=device,
        name=f"sandwich_score_mod_{'causal' if causal else 'non-causal'}",
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
