# Score Mods

Score mods transform attention scores before softmax. Each function returns a `score_mod` — a callable with signature `(score, b, h, q_idx, kv_idx) -> score` — that can be passed directly to [`flex_attention`](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.flex_attention). See [Concepts](concepts.md) for the difference between `score_mod` and `mask_mod`.

```python
from torch.nn.attention.flex_attention import flex_attention

out = flex_attention(query, key, value, score_mod=my_score_mod)
```

## ALiBi

Attention with Linear Biases — adds a linear position-dependent bias to attention scores, removing the need for positional embeddings.

```python
from attn_gym.mods import generate_alibi_bias

alibi = generate_alibi_bias(8)
out = flex_attention(query, key, value, score_mod=alibi)
```

::: attn_gym.mods.alibi.generate_alibi_bias

## Soft-Capping

Tanh soft-capping of attention scores, as used in Gemma-2 and Grok-1.

```python
from attn_gym.mods import generate_tanh_softcap

softcap = generate_tanh_softcap(soft_cap=50.0)
out = flex_attention(query, key, value, score_mod=softcap)
```

::: attn_gym.mods.softcapping.generate_tanh_softcap

## Activation Score Mod

Wraps an activation function to operate in log-space on attention scores.

::: attn_gym.mods.activation.generate_activation_score_mod

::: attn_gym.mods.activation.undo_softmax

## MLA RoPE Score Mod

RoPE-based score modification for Multi-Head Latent Attention (DeepSeek-V2).

!!! warning "Not recommended for production use"
    This implementation is a demonstration of what FlexAttention can express, not a performant implementation. It works correctly but is not optimized — use it as a reference for understanding the API, not as a drop-in for real workloads.

::: attn_gym.mods.latent_attention.generate_mla_rope_score_mod
