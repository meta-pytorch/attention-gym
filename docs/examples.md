# Examples

End-to-end examples demonstrating FlexAttention patterns, from basic usage to advanced techniques.

## Interactive Notebooks

### FlexAttention Walkthrough

[`examples/flex_attn.ipynb`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/flex_attn.ipynb) — Comprehensive Jupyter notebook covering the full FlexAttention API: basic usage, score_mod vs mask_mod, causal masking, sliding window, prefix LM, document masking, NATTEN (with tiled layout), ALiBi, tanh soft-capping, nested jagged tensors, and Flamingo cross-attention. Each section includes performance benchmarks against `F.scaled_dot_product_attention`.

```bash
jupyter notebook examples/flex_attn.ipynb
```

### Debugging Score Mods

[`examples/debug_score_mod.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/debug_score_mod.py) — Interactive [marimo](https://marimo.io) notebook that walks through debugging a broken `score_mod` step-by-step. Demonstrates the `_FLEX_ATTENTION_DISABLE_COMPILE_DEBUG` flag and how to use `get_unwrapped` to inspect tensor values inside vmap. See [Concepts - Debugging](concepts.md#debugging-score_mods) for more details.

```bash
pip install -e ".[viz]"
marimo run examples/debug_score_mod.py
```

## Benchmarking

[`examples/benchmark.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/benchmark.py) — Compare FlexAttention against causal FA2 and `F.sdpa` with a dense mask across multiple attention patterns.

```bash
# Run all benchmarks
python examples/benchmark.py

# Run specific patterns
python examples/benchmark.py --examples causal sliding_window document
```

Available benchmark patterns: `causal`, `alibi`, `sliding_window`, `prefix_lm`, `document`, `softcap`, `softcap_approx`.

Each benchmark reports forward and backward time (ms) and TFLOPS for three implementations:

| Implementation | Description |
|---|---|
| causal FA2 | `F.scaled_dot_product_attention(is_causal=True)` |
| F.sdpa + mask | `F.scaled_dot_product_attention(attn_mask=...)` |
| flexattention | `flex_attention` with `BlockMask` |

The key takeaway: FlexAttention with block sparsity can match or beat dense SDPA while supporting arbitrary attention patterns.

## Flash Backend Comparison

[`examples/flex_flash_attention.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/flex_flash_attention.py) — Compare the Flash (CuTeDSL-based) and Triton backends for FlexAttention.

```bash
python examples/flex_flash_attention.py                    # Run all
python examples/flex_flash_attention.py --mode benchmark   # Just performance
python examples/flex_flash_attention.py --mode compare     # Just numerical accuracy
```

Reports both numerical accuracy (max absolute error vs FP32 reference) and performance (ms, TFLOPS) for forward and backward passes. Requires PyTorch >= 2.10 and flash-attn with CuTeDSL support.

## Multi-Head Latent Attention (MLA)

[`examples/mla.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/mla.py) — DeepSeek-V2 style MLA with weight absorption, showing how FlexAttention can express the absorbed RoPE score modification.

```bash
python examples/mla.py --mode acc     # Accuracy test (vanilla vs absorbed vs flex)
python examples/mla.py --mode perf    # Performance comparison
```

!!! warning "Not recommended for production use"
    This is a demonstration of what FlexAttention can express. The flex-based MLA path is not optimized for production workloads.

## Determinism Testing

[`examples/flex_determinism.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/flex_determinism.py) — Test bitwise determinism of FlexAttention across multiple compilation settings and tensor shapes.

```bash
python examples/flex_determinism.py
```

Tests forward and backward determinism across configurations (eager, inductor default, forced reduction filtering) with shapes covering standard attention, decode, GQA, and long-context scenarios.

## Ring Attention

[`examples/ring_attention.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/ring_attention.py) — Build a custom distributed ring-attention op by directly invoking FlexAttention's forward and backward primitives.

Implements a generic single-node ring-attention example launched with `torchrun`. Each rank owns a contiguous sequence shard of `q/k/v`, rotates `k/v` with point-to-point communication, merges the local `(out, lse)` online, then routes `dk/dv` contributions back to the owning rank by circulating gradient accumulators with each shard in backward. The script validates local slices and gathered outputs and gradients against a single-process causal reference.

```bash
torchrun --standalone --nproc_per_node=4 examples/ring_attention.py --seq-len 131072
```

## Kernel Tuning

### Autotune Replay

[`examples/flex_autotune_replay.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/flex_autotune_replay.py) — Capture kernel tuning decisions from `max-autotune` and replay them deterministically. Useful for locking in performance-critical configurations.

### Grid Sweep

[`examples/flex_grid_sweep.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/flex_grid_sweep.py) — Exhaustive manual sweep over kernel configurations (block sizes, stages, warps) for a given attention pattern and problem size. Edit `MASK_MOD` and `SCORE_MOD` at the top of the file to test your specific pattern.

```bash
python examples/flex_grid_sweep.py --B 4 --H 32 --S 8192 --D 128
```

## Paged Attention

[`attn_gym/paged_attention/`](https://github.com/meta-pytorch/attention-gym/tree/main/attn_gym/paged_attention) — Efficient inference with FlexAttention for batches with variable-length KV caches. KV tensors are split into fixed-size pages and stored compactly instead of padding to the maximum length.

```python
import torch
from attn_gym.paged_attention.paged_attention import PagedAttention

paged_attn = PagedAttention(
    n_pages=256,
    page_size=128,
    max_batch_size=32,
    device="cuda",
)

# Reserve capacity for a batch element
paged_attn.reserve(batch_idx=torch.tensor(0), seq_len=torch.tensor(512))

# Assign KV values into the paged cache
paged_attn.assign(batch_idx, input_pos, k_val, v_val, k_cache, v_cache)

# Convert a logical block mask to physical page layout
physical_block_mask = paged_attn.convert_logical_block_mask(block_mask)
```

Key methods:

- **`reserve(batch_idx, seq_len)`** — ensure capacity for at least `seq_len` tokens
- **`erase(batch_idx)`** — free all pages for a batch element
- **`assign(...)`** — write KV values into the paged cache
- **`convert_logical_block_mask(block_mask)`** — remap logical block indices to physical page indices
- **`get_mask_mod(mask_mod)`** / **`get_score_mod(score_mod)`** — wrap mods to operate in physical page space

See [`paged_attention.py`](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/paged_attention/paged_attention.py) for full implementation.

## ROWS_GUARANTEED_SAFE Demo

[`examples/rows_guaranteed_safe_demo.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/rows_guaranteed_safe_demo.py) — Demonstrates the subtle `ROWS_GUARANTEED_SAFE` contract in BlockMask metadata. Shows how block-level metadata can declare a block "safe" even when individual rows within it have no surviving attention entries, leading to NaN under `torch.compile` but not in eager mode.

```bash
python examples/rows_guaranteed_safe_demo.py
```

## AOTInductor Integration

### Block Mask Export

[`examples/aoti_create_block_mask.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/aoti_create_block_mask.py) — Export `create_block_mask` as an AOTInductor package with dynamic batch/head dimensions. Useful for deploying block mask creation outside of Python.

```bash
python examples/aoti_create_block_mask.py
```

### Full FlexAttention Export

[`examples/aoti_flex_attention.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/aoti_flex_attention.py) — Export the full `flex_attention` forward (with LSE) as an AOTInductor artifact that can be serialized, saved, and reloaded.

```bash
python examples/aoti_flex_attention.py --mode aoti    # AOTInductor backend
python examples/aoti_flex_attention.py --mode python  # Python inductor backend
```
