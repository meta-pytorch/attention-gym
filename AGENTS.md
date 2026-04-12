# Attention Gym

A collection of examples and tools for PyTorch's `flex_attention` API (`torch.nn.attention.flex_attention`).

## What This Repo Provides

**Mask functions** (`attn_gym/masks/`) — ready-to-use `mask_mod` functions that return `BlockMask` objects:
- `causal` — standard causal (triangular) mask
- `sliding_window` — local sliding window attention
- `dilated_sliding_window` — dilated sliding window patterns
- `prefix_lm` — prefix LM mask (bidirectional prefix + causal suffix)
- `document_mask` — document-level masking for packed sequences
- `natten` — neighborhood attention (multi-dimensional)
- `flamingo` — cross-attention mask for Flamingo-style models
- `batchify` — group tokens into batches with intra-group attention only
- `sta` — STA (sparse temporal attention) mask
- `svg` — Sparse VideoGen spatial/temporal attention masks

**Score mods** (`attn_gym/mods/`) — `score_mod` functions that transform attention scores:
- `alibi` — ALiBi positional bias
- `softcapping` — Gemma-2 style soft-capping
- `latent_attention` — latent/compressed attention

**Paged attention** (`attn_gym/paged_attention/`) — `PagedAttention` class for efficient inference with variable-length KV caches using fixed-size page blocks.

**Utilities** (`attn_gym/utils.py`) — helpers for visualizing and debugging `score_mod`/`mask_mod` functions.

**Examples** (`examples/`) — end-to-end usage and benchmarks:
- `benchmark.py` — performance comparison of mask implementations
- `mla.py` — Multi-Head Latent Attention (DeepSeek-style)
- `flex_attn.ipynb` — interactive notebook walkthrough
- `debug_score_mod.py` — marimo app for interactive score_mod debugging
- `flex_determinism.py` — determinism testing for flex_attention

## Quick Start

```bash
pip install -e ".[dev]"
```

Most files are runnable standalone to see visualizations:
```bash
python attn_gym/masks/document_mask.py
python attn_gym/masks/causal.py
```

## Development

```bash
pytest                          # run tests
ruff check && ruff format       # lint + format
prek                            # full pre-commit suite
```

### Docs

```bash
pip install -e ".[docs]"
mkdocs serve                    # local preview at localhost:8000
mkdocs build                    # static site in site/
```

Line length: 99 chars. Python target: 3.10+. Formatter/linter: ruff.

## Agent Scratch Space

If you need scratch space for intermediate files, drafts, or temporary artifacts, use the `agent_space/` directory. This directory is gitignored and will not be checked in.
