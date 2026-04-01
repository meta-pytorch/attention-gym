# Attention Gym

![Attention Gym](assets/hero.png)

Attention Gym is a collection of tools and examples for working with PyTorch's [FlexAttention](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html) API.

It provides ready-to-use **mask functions** and **score mods** that you can compose, visualize, and use directly in your models.

## What's Inside

- **[Concepts](concepts.md)** — how `mask_mod`, `score_mod`, and `BlockMask` work together
- **[Masks](masks.md)** — `mask_mod` functions for causal, sliding window, document-level, neighborhood attention, and more
- **[Score Mods](mods.md)** — `score_mod` functions like ALiBi, soft-capping, and latent attention
- **[Examples](examples.md)** — benchmarks, MLA, paged attention, determinism testing, and advanced patterns
- **[Utilities](utilities.md)** — visualization, benchmarking, and profiling helpers

## Quick Example

```python
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from attn_gym.masks import generate_sliding_window

B, H, S, D = 1, 8, 4096, 64
device = "cuda"

query = torch.randn(B, H, S, D, device=device)
key = torch.randn(B, H, S, D, device=device)
value = torch.randn(B, H, S, D, device=device)

sliding_window = generate_sliding_window(window_size=1024)
block_mask = create_block_mask(sliding_window, B, H, S, S, device=device)
out = flex_attention(query, key, value, block_mask=block_mask)
```

## Installation

```bash
git clone https://github.com/meta-pytorch/attention-gym.git
cd attention-gym
pip install -e .
```

Requires PyTorch 2.5+. See the [PyTorch FlexAttention docs](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html) for API details.
