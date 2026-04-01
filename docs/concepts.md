# Concepts

This page covers the core concepts behind FlexAttention. For the official PyTorch documentation, see the [FlexAttention API reference](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html).

## score_mod vs mask_mod

FlexAttention has two extension points, and understanding when to use each is key to getting the best performance.

### mask_mod

A `mask_mod` defines **which positions attend to which**, based only on positional information. It has the signature:

```python
def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
    # Returns a boolean: True = attend, False = mask out
    return q_idx >= kv_idx  # causal mask
```

Mask mods are used to create a [`BlockMask`](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask) via [`create_block_mask`](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.create_block_mask), which enables **block sparsity** — entire blocks of the attention matrix that are fully masked get skipped entirely, saving both compute and memory.

### score_mod

A `score_mod` transforms the **actual attention score values** before softmax. It has the signature:

```python
def score_mod(score: Tensor, b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
    # Modify and return the score
    bias = alibi_slope[h] * (kv_idx - q_idx)
    return score + bias
```

Score mods are passed directly to [`flex_attention`](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.flex_attention).

### When to use which?

If your modification **doesn't depend on the score value** and just decides "attend or don't attend" based on position, use a `mask_mod`. Any `mask_mod` *could* be written as a `score_mod` (by returning `-inf` for masked positions), but you'd lose the block sparsity optimization — potentially doing 2x the work for causal attention.

You can combine both: pass a `block_mask` and a `score_mod` to `flex_attention`, and the score mod is only applied to unmasked positions.

```python
block_mask = create_block_mask(causal_mask, B, H, S, S, device=device)
out = flex_attention(q, k, v, block_mask=block_mask, score_mod=alibi_bias)
```

### Composing masks

Mask mods compose naturally with boolean operators:

```python
from torch.nn.attention.flex_attention import and_masks, or_masks

# Sliding window + causal
sliding_causal = and_masks(causal_mask, sliding_window_mask)

# Or combine inline
def my_mask(b, h, q_idx, kv_idx):
    return (q_idx >= kv_idx) & (q_idx - kv_idx <= window_size)
```

## BlockMask Internals

A [`BlockMask`](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html#torch.nn.attention.flex_attention.BlockMask) is the block-sparse representation of your attention pattern. Understanding its structure helps with debugging and advanced usage.

The attention matrix is divided into blocks of size `BLOCK_SIZE` (default 128x128). Each block is classified as:

- **Full** (`██`) — all positions attend, no per-element masking needed
- **Partial** (`░░`) — some positions attend, the `mask_mod` is evaluated per-element
- **Empty** (absent) — no positions attend, the block is skipped entirely

### Key attributes

| Attribute | Shape | Description |
|---|---|---|
| `kv_num_blocks` | `(B, H, num_q_blocks)` | How many KV blocks each query block attends to |
| `kv_indices` | `(B, H, num_q_blocks, max_kv_blocks)` | Which KV block indices each query block attends to |
| `full_kv_num_blocks` | `(B, H, num_q_blocks)` | How many of those are fully unmasked (optional, for perf) |
| `full_kv_indices` | `(B, H, num_q_blocks, max_kv_blocks)` | Indices of the fully unmasked KV blocks |
| `q_num_blocks` | `(B, H, num_kv_blocks)` | Transposed view — how many query blocks attend to each KV block (auto-generated, used in backward) |
| `q_indices` | `(B, H, num_kv_blocks, max_q_blocks)` | Transposed indices (auto-generated, used in backward) |
| `BLOCK_SIZE` | `(int, int)` | Block size for query and KV dimensions |
| `mask_mod` | callable | The original mask function, evaluated for partial blocks |

### Inspecting a BlockMask

```python
from torch.nn.attention.flex_attention import create_block_mask
from attn_gym.masks import causal_mask

block_mask = create_block_mask(causal_mask, B=1, H=1, Q_LEN=2048, KV_LEN=2048, device="cuda")

print(block_mask)            # ASCII visualization of the sparsity pattern
print(block_mask.sparsity()) # percentage of blocks that are empty
```

The `print` output shows a grid where `██` = full block, `░░` = partial block, and blank = empty (skipped). For causal attention this is the lower triangle.

### Changing BlockMask doesn't require recompilation

An important property: **only changes to `score_mod` or `mask_mod` functions trigger recompilation**. You can freely swap `BlockMask` objects (e.g., when document boundaries change between batches) without recompilation. This makes patterns like document masking efficient in practice.

## Debugging score_mods

FlexAttention compiles your `score_mod` via `torch.compile` by default, which means you can't use `print()` or `breakpoint()` inside it. The debug flag disables compilation for inspection:

```python
import torch.nn.attention.flex_attention as fa

fa._FLEX_ATTENTION_DISABLE_COMPILE_DEBUG = True
```

When debugging, use `torch._C._functorch.get_unwrapped` to inspect tensor values inside the vmap:

```python
unwrap = torch._C._functorch.get_unwrapped

def my_score_mod(score, b, h, q_idx, kv_idx):
    distance = q_idx - kv_idx
    print(f"distance: {unwrap(distance)}")  # indices need 1 unwrap
    return score + distance
```

The `score` tensor needs 4 layers of unwrapping (one per vmap dimension: batch, head, query, key). Index tensors need only 1.

For a full interactive walkthrough of this debugging workflow, run the marimo notebook:

```bash
pip install -e ".[viz]"
marimo run examples/debug_score_mod.py
```

## Interactive Notebook

The `examples/flex_attn.ipynb` Jupyter notebook provides a comprehensive interactive walkthrough covering:

- Basic usage and score modification vs masking
- Causal, sliding window, prefix LM, document masking
- NATTEN and tiled NATTEN with correctness verification
- ALiBi bias (closure-based and functional)
- Tanh soft-capping with custom approximate tanh operator
- Nested jagged tensors for variable-length sequences
- Flamingo cross-attention

```bash
jupyter notebook examples/flex_attn.ipynb
```

## Further Reading

- [FlexAttention API reference](https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html) — official PyTorch docs
- [FlexAttention blog post](https://pytorch.org/blog/flexattention/) — introduction and motivation
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original transformer paper
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) — the block-sparse attention algorithm that FlexAttention builds on
