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

## Sandwich

Sandwich relative positional bias ([paper](https://arxiv.org/abs/2212.10356)) — the position-position inner product of sinusoidal embeddings with ALiBi-style per-head compression ratios, enabling length extrapolation without learned parameters.

```python
from attn_gym.mods import generate_sandwich_bias

sandwich = generate_sandwich_bias(H=num_heads, max_seq_len=S, device=device)
out = flex_attention(query, key, value, score_mod=sandwich)
```

::: attn_gym.mods.sandwich.generate_sandwich_bias

## Graphormer

Graphormer attention biases ([paper](https://arxiv.org/abs/2106.05234)) — two learnable terms added to attention scores between graph node pairs. Gradients flow back into the captured bias tables through flex_attention's backward, so both train end to end.

**Spatial encoding** — a learnable per-head bias indexed by the shortest-path distance between node pairs:

```python
import torch
from attn_gym.mods import generate_graphormer_spatial_bias, shortest_path_distances

distances = shortest_path_distances(adjacency, max_distance=5)  # (B, N, N)
spatial_bias = torch.nn.Parameter(torch.zeros(num_heads, 5 + 2, device=device))
graphormer = generate_graphormer_spatial_bias(spatial_bias, distances)
out = flex_attention(query, key, value, score_mod=graphormer)
```

**Edge encoding** — averages a learnable scalar per (head, path position, edge type) over the edges along each pair's shortest path:

```python
from attn_gym.mods import generate_graphormer_edge_bias, shortest_path_edge_types

path_types, path_lengths = shortest_path_edge_types(adjacency, edge_types, max_path_len=4)
edge_bias = torch.nn.Parameter(torch.zeros(num_heads, 4, num_edge_types, device=device))
graphormer_edge = generate_graphormer_edge_bias(edge_bias, path_types, path_lengths)
out = flex_attention(query, key, value, score_mod=graphormer_edge)
```

::: attn_gym.mods.graphormer.generate_graphormer_spatial_bias

::: attn_gym.mods.graphormer.generate_graphormer_edge_bias

::: attn_gym.mods.graphormer.shortest_path_distances

::: attn_gym.mods.graphormer.shortest_path_edge_types

## Activation Score Mod

Wraps an activation function to operate in log-space on attention scores.

::: attn_gym.mods.activation.generate_activation_score_mod

::: attn_gym.mods.activation.undo_softmax

## MLA RoPE Score Mod

RoPE-based score modification for Multi-Head Latent Attention (DeepSeek-V2).

!!! warning "Not recommended for production use"
    This implementation is a demonstration of what FlexAttention can express, not a performant implementation. It works correctly but is not optimized — use it as a reference for understanding the API, not as a drop-in for real workloads.

::: attn_gym.mods.latent_attention.generate_mla_rope_score_mod
