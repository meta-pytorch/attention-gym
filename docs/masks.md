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

## Prefix LM

Bidirectional attention over a prefix, causal attention over the rest.

```python
from attn_gym.masks import generate_prefix_lm_mask

mask_mod = generate_prefix_lm_mask(prefix_length=512)
```

::: attn_gym.masks.prefix_lm.generate_prefix_lm_mask

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

## Neighborhood Attention (NATTEN)

Multi-dimensional neighborhood attention patterns.

::: attn_gym.masks.natten.generate_natten

::: attn_gym.masks.natten.generate_tiled_natten

::: attn_gym.masks.natten.generate_morton_natten

## STA (Sparse Temporal Attention)

::: attn_gym.masks.sta.generate_sta_mask_mod_2d

::: attn_gym.masks.sta.generate_sta_mask_mod_3d

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
