"""Implementation of an ALIBI score mod from the paper Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation: https://arxiv.org/abs/2108.12409"""

import torch
from torch.nn.attention.flex_attention import _score_mod_signature


def generate_alibi_bias(H: int) -> _score_mod_signature:
    """Returns an alibi bias score_mod given the number of heads H

    Args:
        H: number of heads

    Returns:
        alibi_bias: alibi bias score_mod
    """

    def alibi_mod(score, b, h, q_idx, kv_idx):
        scale = torch.exp2(-((h + 1) * 8.0 / H))
        bias = (kv_idx - q_idx) * scale
        return score + bias

    return alibi_mod


def main(device: str = "cpu", causal: bool = True):
    """Visualize the attention scores alibi bias score mod.

    Args:
        device (str): Device to use for computation. Defaults
    """
    import torch

    from attn_gym import visualize_attention_scores

    B, H, SEQ_LEN, HEAD_DIM = 1, 1, 12, 8

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device=device)

    query, key = make_tensor(), make_tensor()

    alibi_score_mod = generate_alibi_bias(H)

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    visualize_attention_scores(
        query,
        key,
        score_mod=alibi_score_mod,
        mask_mod=causal_mask if causal else None,
        device=device,
        name=f"alibi_score_mod_{'causal' if causal else 'non-causal'}",
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
