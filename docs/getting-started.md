# Getting Started

## Prerequisites

- Python 3.9+
- PyTorch 2.5+ (for FlexAttention support)

## Installation

```bash
git clone https://github.com/meta-pytorch/attention-gym.git
cd attention-gym
pip install -e .
```

For visualization support:

```bash
pip install -e ".[viz]"
```

## Core Concepts

FlexAttention has two main extension points:

1. **`mask_mod`** — a function `(b, h, q_idx, kv_idx) -> bool` that defines which positions can attend to which. Used to create a `BlockMask`.
2. **`score_mod`** — a function `(score, b, h, q_idx, kv_idx) -> score` that transforms attention scores before softmax.

Attention Gym provides generators for both.

## Your First Mask

```python
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from attn_gym.masks import causal_mask

B, H, S, D = 1, 8, 2048, 64
device = "cuda"

block_mask = create_block_mask(causal_mask, B, H, S, S, device=device)

query = torch.randn(B, H, S, D, device=device)
key = torch.randn(B, H, S, D, device=device)
value = torch.randn(B, H, S, D, device=device)

out = flex_attention(query, key, value, block_mask=block_mask)
```

## Combining Masks and Score Mods

```python
from attn_gym.masks import generate_sliding_window
from attn_gym.mods import generate_alibi_bias

sliding_window = generate_sliding_window(window_size=512)
block_mask = create_block_mask(sliding_window, B, H, S, S, device=device)

alibi = generate_alibi_bias(H)
out = flex_attention(query, key, value, block_mask=block_mask, score_mod=alibi)
```

## Visualizing Masks

Every mask file can be run standalone to produce a visualization:

```bash
pip install -e ".[viz]"
python attn_gym/masks/causal.py
python attn_gym/masks/sliding_window.py
python attn_gym/masks/document_mask.py
```

You can also visualize programmatically:

```python
from attn_gym import visualize_attention_scores
from attn_gym.masks import causal_mask

B, H, SEQ_LEN, HEAD_DIM = 1, 1, 12, 8
query = torch.ones(B, H, SEQ_LEN, HEAD_DIM)
key = torch.ones(B, H, SEQ_LEN, HEAD_DIM)

visualize_attention_scores(query, key, mask_mod=causal_mask, name="causal")
```
