# Masks

Mask functions define which query-key pairs can attend to each other. Each function returns a `mask_mod` — a callable with signature `(b, h, q_idx, kv_idx) -> bool` — that can be passed to [`create_block_mask`](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.create_block_mask) to produce a [`BlockMask`](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask). See [Concepts](concepts.md) for details on how masks and block sparsity work together.

## Causal

Standard lower-triangular causal mask. Each position attends only to itself and earlier positions.

```python
from attn_gym.masks import causal_mask

block_mask = create_block_mask(causal_mask, B, H, S, S, device=device)
```

::: attn_gym.masks.causal.causal_mask

::: attn_gym.masks.causal.create_causal_block_mask_fast

## Sliding Window

Each position attends to a fixed-size window of preceding tokens (combined with causal masking).

```python
from attn_gym.masks import generate_sliding_window

mask_mod = generate_sliding_window(window_size=1024)
block_mask = create_block_mask(mask_mod, B, H, S, S, device=device)
```

::: attn_gym.masks.sliding_window.generate_sliding_window

## Dilated Sliding Window

Sliding window with dilation — attends to every `dilation`-th token within the window.

::: attn_gym.masks.dilated_sliding_window.generate_dilated_sliding_window

## Global + Sliding Window

Longformer-style attention ([paper](https://arxiv.org/abs/2004.05150)): a bidirectional sliding window plus designated global tokens (e.g. CLS) that attend to and are attended by every position.

```python
import torch
from attn_gym.masks import generate_global_sliding_window

is_global = torch.zeros(S, dtype=torch.bool, device=device)
is_global[0] = True  # CLS token
mask_mod = generate_global_sliding_window(window_size=512, is_global=is_global)
block_mask = create_block_mask(mask_mod, B, H, S, S, device=device)
```

::: attn_gym.masks.global_sliding_window.generate_global_sliding_window

## Prefix LM

Bidirectional attention over a prefix, causal attention over the rest.

```python
from attn_gym.masks import generate_prefix_lm_mask

mask_mod = generate_prefix_lm_mask(prefix_length=512)
```

::: attn_gym.masks.prefix_lm.generate_prefix_lm_mask

## JetSpec Tree Attention

Tree-verification attention for [JetSpec](https://arxiv.org/abs/2606.18394): each tree
query attends to cached prefix keys and to flattened tree keys only when they are ancestors
of that query node, including self. The example below uses the Figure 3 candidate-tree
order: `return, a, -, +, B, b, sum`.

```python
from attn_gym.masks import build_tree_ancestor_matrix, generate_jetspec_tree_causal_mask_mod

parent_indices = [-1, 0, 1, 1, 3, 3, 0]
ancestor = build_tree_ancestor_matrix(parent_indices, device="cuda")
mask_mod = generate_jetspec_tree_causal_mask_mod(prefix_length=1024, ancestor_matrix=ancestor)
block_mask = create_block_mask(
    mask_mod,
    B,
    H,
    ancestor.shape[0],
    1024 + ancestor.shape[0],
    device="cuda",
)
```

::: attn_gym.masks.jetspec.build_tree_ancestor_matrix

::: attn_gym.masks.jetspec.generate_jetspec_tree_causal_mask_mod

JetSpec's draft-head training path uses a different multi-block causal mask: sampled-block
queries attend to all verified prefix keys and causally within their own sampled block.

```python
from attn_gym.masks import generate_jetspec_training_mask_mod

prefix_length = 4096
block_size = 16
num_blocks = 3
mask_mod = generate_jetspec_training_mask_mod(prefix_length, block_size)
block_mask = create_block_mask(
    mask_mod,
    B,
    H,
    num_blocks * block_size,
    prefix_length + num_blocks * block_size,
    device="cuda",
)
```

::: attn_gym.masks.jetspec.generate_jetspec_training_mask_mod

## Document Mask

For packed sequences: restrict attention to within document boundaries by wrapping a base
mask with document offsets.

```python
from attn_gym.masks import causal_mask, generate_doc_mask_mod
from attn_gym.masks.document_mask import length_to_offsets

lengths = [3, 2, 5]
offsets = length_to_offsets(lengths, device="cuda")
mask_mod = generate_doc_mask_mod(causal_mask, offsets)
```

::: attn_gym.masks.document_mask.generate_doc_mask_mod

For causal packed sequences where every batch element shares the same document layout, use the
causal-only offset form. It is equivalent to wrapping `causal_mask`, but exposes each query row as
one contiguous KV interval.

```python
from attn_gym.masks import generate_packed_causal_doc_mask_mod
from attn_gym.masks.document_mask import length_to_offsets

lengths = [3, 2, 5]
offsets = length_to_offsets(lengths, device="cuda")
mask_mod = generate_packed_causal_doc_mask_mod(offsets)
```

::: attn_gym.masks.document_mask.generate_packed_causal_doc_mask_mod

## Neighborhood Attention (NATTEN)

Multi-dimensional neighborhood attention patterns.

::: attn_gym.masks.natten.generate_natten

::: attn_gym.masks.natten.generate_tiled_natten

::: attn_gym.masks.natten.generate_morton_natten

## STA (Sparse Temporal Attention)

::: attn_gym.masks.sta.generate_sta_mask_mod_2d

::: attn_gym.masks.sta.generate_sta_mask_mod_3d

## VSA (Video Sparse Attention)

VSA's fine sparse pass can be represented by precomputing each query tile's top-k KV
tiles and turning those tile ids into a `BlockMask`. Use `create_vsa_block_mask` for
the direct Triton construction path; deriving the same mask with generic
`create_block_mask` is intended only for visualization or small correctness checks.
For current Flex FLASH all-full tile paths, use `create_vsa_flash_block_mask`. The
coarse top-k selection and FastVideo-style additive coarse/fine output combine run
outside the FlexAttention kernel.

```python
from attn_gym.masks import compute_vsa_coarse_attention, create_vsa_block_mask

coarse = compute_vsa_coarse_attention(q, k, v, tile_numel=64, top_k=78)
block_mask = create_vsa_block_mask(
    coarse.topk_indices,
    tile_numel=64,
    num_kv_tiles=k.shape[-2] // 64,
)
out = flex_attention(q, k, v, block_mask=block_mask)
```

::: attn_gym.masks.vsa.compute_vsa_coarse_attention

::: attn_gym.masks.vsa.create_vsa_block_mask

::: attn_gym.masks.vsa.create_vsa_flash_block_mask

::: attn_gym.masks.vsa.generate_vsa_mask_mod

## Batchify

Groups tokens into batches where attention is only allowed within the same group.

::: attn_gym.masks.batchify.batchify_mask_mod

## Flamingo Cross-Attention

Cross-attention mask for Flamingo-style vision-language models.

::: attn_gym.masks.flamingo.generate_vision_cross_attention_mask_mod

## Sparse VideoGen

Spatial and temporal attention masks following the [Sparse VideoGen](https://arxiv.org/abs/2502.01776) paper.

::: attn_gym.masks.svg.generate_spatial_head_mask_mod

::: attn_gym.masks.svg.generate_temporal_head_mask_mod
