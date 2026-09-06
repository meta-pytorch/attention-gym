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
- `graphormer` — Graphormer spatial + edge encodings (learnable shortest-path biases)
- `latent_attention` — latent/compressed attention

**Paged attention example** (`examples/paged_attention/`) — `PagedAttention` example for efficient inference with variable-length KV caches using fixed-size page blocks.

**Utilities** (`attn_gym/utils.py`) — helpers for visualizing and debugging `score_mod`/`mask_mod` functions.

**Examples** (`examples/`) — end-to-end usage and benchmarks:
- `benchmark.py` — performance comparison of mask implementations
- `mla.py` — Multi-Head Latent Attention (DeepSeek-style)
- `delta_rule_training.py` — trainable single-device KDA/GDN showcase with reference or fused backends
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

Every worktree needs its own `.venv` with an editable install of *this* checkout (see the
`worktree-env-setup` skill; `uv` hard-links wheels, so it takes seconds). Never symlink or
reuse a sibling worktree's `.venv`: its editable install resolves `attn_gym` to the other
checkout, so scripts and pytest silently run the wrong sources. `test/conftest.py` refuses to
start when that happens.

```bash
pytest -n 6                     # run tests in parallel (strongly preferred)
pytest -n 6 test/test_kda.py    # one file, same parallelism
pytest test/test_kda.py::test_x # single test; -n adds only overhead here
ruff check && ruff format       # lint + format
prek                            # full pre-commit suite
```

Use `pytest -n 6` (pytest-xdist, already in `[tests]`) for anything wider than a single
test. Much of the suite is CuTeDSL and `torch.compile` work that is CPU-bound during
compilation, so a serial run leaves the machine idle and takes minutes where a parallel one
takes tens of seconds. The workers share one GPU, so raise the count only if the GPU has
headroom, and drop back to `-n 0` when a failure needs a clean serial repro or readable
output.

### Docs

```bash
pip install -e ".[docs]"
mkdocs serve                    # local preview at localhost:8000
mkdocs build                    # static site in site/
```

Line length: 99 chars. Python target: 3.10+. Formatter/linter: ruff.

### Stacked PRs

Do not use `ghstack` in this repository. For commit stacks, use `stack-pr` or GitHub's
native stacked-PR support through `gh stack`.

## Project-local Agent Skills

Repository-specific workflows live under `.agents/skills/`. Load the matching `SKILL.md`
before changing the covered subsystem:

- [`worktree-env-setup`](.agents/skills/worktree-env-setup/SKILL.md) —
  isolated per-worktree `.venv` with nightly PyTorch via the CI-mirroring uv flow.
- [`validating-pytorch-custom-ops`](.agents/skills/validating-pytorch-custom-ops/SKILL.md) —
  registration, fake implementations, autograd, `opcheck`, and `torch.compile` validation.
- [`cutedsl-tunable-kernel-template`](.agents/skills/cutedsl-tunable-kernel-template/SKILL.md) —
  typed config generation, cached TVM-FFI compilation, parallel compile, and sequential tuning.

## Agent Scratch Space

If you need scratch space for intermediate files, drafts, or temporary artifacts, use the `agent_space/` directory. This directory is gitignored and will not be checked in.
