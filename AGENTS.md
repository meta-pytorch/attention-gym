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

**Paged attention example** (`examples/flex_attention/paged_attention.py`) — `PagedAttention` example for efficient inference with variable-length KV caches using fixed-size page blocks.

**Utilities** (`attn_gym/utils.py`) — helpers for visualizing and debugging `score_mod`/`mask_mod` functions.

**Examples** (`examples/`) — end-to-end usage and benchmarks, at most one directory level deep:
- `flex_attention/` — masks/score mods, notebooks, MLA, paged/ring attention, and runtime integration
- `linear/` — KDA/GDN training, context parallelism, and decoding
- `sparse/` — compressed sparse attention, VSA, and FastWan integration

Keep variant-specific names in filenames rather than adding nested example directories.

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
output. On a 144-core GB300 host the warm full suite takes 112s at `-n 6`, 78s at `-n 12`,
and plateaus near 75s beyond that; use `-n 12` there for full runs.

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

## Adding CuTeDSL Kernels

Compile every CuTeDSL kernel through a module-level function decorated with `@jit_cache`
(`attn_gym/_backends/cute/cache.py`); see the `cutedsl-tunable-kernel-template` skill for the
full adapter. A compiled kernel is stored on disk and reused until one of its key inputs changes:

- the compile function's arguments (static, pickleable values) and the compile target;
- the Python, cutlass, tvm_ffi, torch, and CUDA versions and the codegen environment variables
  listed in `attn_gym/_backends/cute/_key.py` (`CUTE_DSL_ARCH` reaches it through the target);
- the source of the compile function's module and of every `attn_gym` module it imports,
  directly or transitively (`_key.module_closure`). Unrelated modules do not invalidate it.

Imports are found statically, so keep the code a kernel traces reachable by import:

- Import kernel helpers, constants, and ops with ordinary `import`/`from` statements (local and
  relative imports are fine). A module named as a string literal, as in
  `importlib.import_module("attn_gym.x.y")`, also counts; computed module names do not.
- `import pkg.sub.module` does not add `pkg/sub/__init__.py`; keep package `__init__` files to
  plain re-exports rather than state a kernel depends on.
- Pass `extra_sources=(...)` to `jit_cache` only for inputs import analysis cannot see: files read
  while tracing, or code outside `attn_gym` other than the versioned dependencies above. A class
  passed as a compile argument is keyed by its module and name only, so if its module is not
  imported by the compile module, list that module in `extra_sources`.
- Do not add `extra_sources` for modules the compile module already imports; they are hashed.

When testing specialization reuse, assert on `compile_fn.cache_info()` (distinct keys in
`currsize`, launches in `hits + misses`) instead of disabling the disk cache with
`CUTE_DSL_NO_CACHE`, which forces cold compiles.

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
