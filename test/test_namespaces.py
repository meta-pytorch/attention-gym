import attn_gym
import attn_gym.linear
import attn_gym.sparse


def test_attention_namespaces_are_exported():
    assert attn_gym.linear.__name__ == "attn_gym.linear"
    assert attn_gym.sparse.__name__ == "attn_gym.sparse"
    assert "linear" in attn_gym.__all__
    assert "sparse" in attn_gym.__all__
    assert "paged_attention" not in attn_gym.__all__
