---
name: worktree-env-setup
description: Sets up an isolated per-worktree Python environment for attention-gym development using nightly PyTorch and the CI-mirroring uv flow. Use when creating a new git worktree or when a worktree lacks a local .venv.
---

# Worktree Environment Setup

Each attention-gym worktree gets its own `.venv` so editable installs, concurrent
agents, and test runs never cross-import another checkout. Never reuse a shared
env's editable install across worktrees.

## Setup

From the worktree root (mirrors `.github/workflows/test.yml`):

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu132
uv pip install --prerelease allow -e '.[tests,linear,dev]'
```

Notes:

- uv hard-links wheels from its cache, so after the first nightly download this
  takes seconds and costs almost no extra disk per worktree.
- Activate `.venv` before installing so an already-active foreign environment is not modified.
- `--prerelease allow` is required for the `flash-attn-4` beta in `[tests]`.
- Do not use `uv sync`/`uv.lock`: nightly torch churns daily and CI uses the
  imperative `uv pip` flow above, not a lockfile.
- Drop `[linear]` if CuTeDSL/TVM-FFI kernels are not needed (CPU-only work).
- `[tests]` currently brings FlashAttention's CuTeDSL 4.6 pin and cannot be combined with the
  CuTeDSL 4.7+ `[mega]` extra. For Mega worktrees, install `-e '.[mega,dev]' pytest pytest-xdist`
  instead; Mega tests import-skip optional FlashAttention coverage.

## Running commands

Prefer the worktree's own interpreter — either activate `.venv` first, or use
`uv run --no-sync pytest test` (matches CI exactly). Never invoke a Python from
another worktree or a shared `~/.venvs/*` env for attn_gym imports.

## Verifying isolation

```bash
python -c "import attn_gym; print(attn_gym.__file__)"
```

The printed path must be inside the current worktree. If it points at another
checkout, the editable install is wrong — rerun the `-e '.[tests,linear,dev]'`
install from this worktree root.
