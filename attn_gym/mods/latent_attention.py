"""Multi-head Latent Attention RoPE score modification from DeepSeek-V2."""

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import _score_mod_signature


def generate_mla_rope_score_mod(
    query_rope: Tensor,
    key_rope: Tensor,
    num_heads: int,
    scale: float = 1.0,
) -> _score_mod_signature:
    """Return an MLA RoPE score modification function for FlexAttention.

    Args:
        query_rope: Query positional embeddings with shape ``(B, H, T, D)``.
        key_rope: Key positional embeddings with shape ``(B, H_kv, T, D)``.
        num_heads: Number of query heads.
        scale: Scaling factor for the positional embedding contribution.

    Returns:
        Score modification function for FlexAttention.
    """

    def mla_rope_score_mod(
        score: Tensor, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor
    ) -> Tensor:
        return score + (
            scale * torch.dot(query_rope[b, h, q_idx], key_rope[b, h // num_heads, kv_idx])
        )

    mla_rope_score_mod.__name__ = f"mla_rope_score_mod_scale_{scale}"
    return mla_rope_score_mod


def main(device: str = "cuda"):
    """Visualize the attention scores with MLA RoPE modification.

    Args:
        device: Device to use for computation.
    """
    from attn_gym import visualize_attention_scores

    B, H, SEQ_LEN, LATENT_HEAD_DIM = 1, 128, 8, 512
    ROPE_HEAD_DIM = 64

    query = torch.rand(B, H, SEQ_LEN, LATENT_HEAD_DIM, device=device)
    key = torch.rand(B, 1, SEQ_LEN, LATENT_HEAD_DIM, device=device)

    query_pe = torch.rand(B, H, SEQ_LEN, ROPE_HEAD_DIM, device=device)
    key_pe = torch.rand(B, 1, SEQ_LEN, ROPE_HEAD_DIM, device=device)

    mla_rope_score_mod = generate_mla_rope_score_mod(
        query_rope=query_pe, key_rope=key_pe, num_heads=H
    )

    visualize_attention_scores(
        query, key, score_mod=mla_rope_score_mod, device=device, name="mla_rope_score_mod"
    )


if __name__ == "__main__":
    try:
        from jsonargparse import CLI
    except ImportError:
        raise ImportError("Be sure to run: pip install -e .'[viz]'")
    CLI(main)
