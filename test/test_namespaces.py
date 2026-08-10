import attn_gym
import attn_gym.linear
import attn_gym.sparse
from attn_gym.masks import (
    batchify_mask_mod,
    generate_spatial_head_mask_mod,
    generate_temporal_head_mask_mod,
    generate_vision_cross_attention_mask_mod,
)
from attn_gym.mods import generate_mla_rope_score_mod


def test_attention_namespaces_are_exported():
    assert attn_gym.linear.__name__ == "attn_gym.linear"
    assert attn_gym.sparse.__name__ == "attn_gym.sparse"
    assert "linear" in attn_gym.__all__
    assert "sparse" in attn_gym.__all__
    assert "paged_attention" not in attn_gym.__all__


def test_documented_mask_and_score_mods_are_exported():
    assert all(
        callable(function)
        for function in (
            batchify_mask_mod,
            generate_spatial_head_mask_mod,
            generate_temporal_head_mask_mod,
            generate_vision_cross_attention_mask_mod,
            generate_mla_rope_score_mod,
        )
    )
