# Attention Gym

A collection of examples and tools for PyTorch's `flex_attention` API (`torch.nn.attention.flex_attention`).

## What This Repo Provides

- **Masks** (`attn_gym/masks/`) — `mask_mod` functions and `BlockMask` builders; see
  [`docs/masks.md`](docs/masks.md) for the full list.
- **Score mods** (`attn_gym/mods/`) — `score_mod` functions; see [`docs/mods.md`](docs/mods.md).
- **Linear attention** (`attn_gym/linear/`) — KDA and GDN chunked/recurrent operators and short
  convolution, with reference, Triton, CuTeDSL, and vendored cuDNN backends; see
  [`docs/linear.md`](docs/linear.md).
- **Sparse attention** (`attn_gym/sparse/`) — `lightning_indexer` Top-K selection and
  `gather_attn`; see [`docs/sparse.md`](docs/sparse.md).
- **Distributed** (`attn_gym/distributed/`) — copy-engine all-gather and point-to-point channels over
  symmetric memory, used by the context-parallel attention example.
- **Utilities** (`attn_gym/utils.py`) — helpers for visualizing and debugging
  `score_mod`/`mask_mod` functions.

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

## Kernel Specialization for Dynamic Shapes

Training and serving here are varlen: total tokens, document count, batch, and chunk capacity
change nearly every step, and each new compile or autotune sweep stalls that step (measured from 0.1 s
to over 10 s per occurrence). Keep kernels as dynamic as possible without giving up measured steady-state
performance:

- Pass sizes that only feed bounds, masks, grids, offsets, or loop trip counts (`T`, `S`, batch,
  document count, capacity, strides that scale with them) as runtime values. In Triton also list
  them in `do_not_specialize`: a plain int still compiles separate variants for `== 1` and for
  multiples of 16. In CuTeDSL keep them `cute.sym_int` in fake signatures and out of `jit_cache`
  arguments.
- Specialize what changes generated code and measurably pays: head/feature dims, tiles, dtypes,
  mode flags, int32 versus int64 offsets, and strides fixed by the model. Element-count bounds on
  vectorized masked loads may keep Triton's default divisibility hint (dropping it measured
  1.26-1.52x slower on paged replay).
- Never key a compile or tuning cache on a value the kernel does not read, or on a flag whose code
  is behind another disabled `const_expr` branch.
- Tuning keys bucket what moves the winner (for example a power of two of grid fill or work
  waves), not exact `T` or `N`. Clamp or bucket size-derived constexprs such as
  `next_power_of_2(N)` arange widths so the variant count stays small.
- Prove reuse with a test that varies shapes across the specialization classes (`1`, multiples
  of 16, other values, bucket edges) and asserts compile or tuning counts
  (`triton.knobs.runtime.jit_post_compile_hook`, `compile_fn.cache_info()`, the tunable winner
  cache). A/B warm performance before keeping or removing a specialization.

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
- An argument object can name its own key with `__cute_cache_key__()` (classes passed as values
  key by module and name instead). To key a class defined outside the compile module's imports,
  include `module_source_digest(cls.__module__)` in that hook rather than listing the module in
  `extra_sources`; warpdrive kernels do this.
- Host-only code that tracing never runs (report renderers, trace decoders) stays out of the key
  when imported at call time by relative name, `importlib.import_module(".mod", __package__)`:
  the scan does not follow relative string names. Never import code that tracing runs this way.

A kernel loaded from disk was never traced. An argument that defines `__cute_trace_record__()`
(a picklable record of what tracing left on it) and `__cute_restore_trace__(record)` gets the
record of its build back on every memory or disk hit (`by_static_key` hits excepted); the
record is pickled beside the object file as `<key>.trace`. `jit_cache(uncached=predicate)`
compiles the calls the predicate selects without either cache, for when the trace itself is
the point (warpdrive's `trace_protocol`).

When testing specialization reuse, assert on `compile_fn.cache_info()` (distinct keys in
`currsize`, launches in `hits + misses`) instead of disabling the disk cache with
`CUTE_DSL_NO_CACHE`, which forces cold compiles.

## warpdrive

`attn_gym/warpdrive/` is the warp-specialization layer for CuTeDSL kernels (roles, channels,
work streams); `attn_gym/profiler/` is the intra-kernel profiler its trace backends feed. Load
the [`authoring-warpdrive`](.agents/skills/authoring-warpdrive/SKILL.md) skill when authoring or
porting a kernel. The authoring guide, safety contract and topic guides live under
[`docs/warpdrive/`](docs/warpdrive/index.md) and the profiler under [`docs/profiler.md`](docs/profiler.md);
they must match the checkout being used. Fuzz report labels live in
`test/warpdrive_fuzz/CASES.md`, not a public error-code API.

Warpdrive kernels compile through `jit_cache` like every other CuTeDSL kernel, with two extra
rules:

- Pass a kernel's static choices to its constructor: the key is the constructor call, and a
  public attribute assigned afterwards (other than `cluster`, work streams, and the IKET,
  tracing and alias settings) raises.
- A cached kernel restores its trace results (`describe()`, TMA/alias reports, warnings) without
  tracing. Only a test that asserts on tracing itself (IKET ranges, monkeypatched trace hooks)
  takes the `traced_compiles` fixture (`test/conftest.py`) and calls `compile_fn.cache_clear()`.

### warpdrive Testing Tiers

Pick the smallest tier that covers the change; run bigger tiers only when asked or before
landing core synchronization/scheduling changes.

- Every change (~45 s cold, ~22 s with a warm compile cache on a GB300):
  `pytest -n 12 test/warpdrive test/profiler`. Fuzz tests are skipped by default (`addopts`
  deselects the `fuzz` marker).
- Quick fuzz (~2-3 min, on request or before landing layer changes):
  `pytest -n 12 -m fuzz test/warpdrive test/warpdrive_fuzz`.
- Long campaigns, sanitizer and CUTracer runs, SASS/perf A/Bs: only when the user asks.
  Use `tools/warpdrive/fuzz_warpdrive.py {runtime,static,mutate}` under `gpu-run`.
  `CUTE_DSL_ARCH=sm_100a python tools/warpdrive/warpdrive_sass.py dump OUTDIR` records the
  seven standard configs; `compare BEFORE AFTER` gates on PTX and compiler identity and prints
  SASS diffs (`--strict-sass` also requires identical SASS). ptxas can assign registers
  differently from identical PTX (seen on the small decode configs), so a SASS-only diff with
  identical PTX is not a code change. `tools/warpdrive/warpdrive_ptx_audit.py DIR` screens a
  dump for the saved-thread-index ptxas hazard without a GPU.

## cuDNN Vendor Maintenance

`tools/cudnn_fe/` contains the GDN/KDA vendor/import tools, mechanical restyle and audit,
SASS comparison, CUTracer stress, benchmark gates, and the machine-readable fix ledger.
The CPU audit tests (`test/test_cudnn_fe_tools_*.py`) guard edits to the vendored kernels;
run them through this checkout's editable install, including from outside the repository root.
Read `attn_gym/linear/_delta_rule/cudnn_fe/MAINTENANCE.md` for the upgrade procedure and
`WORKLOG.md` beside it for evidence and replay notes. `tools/cudnn_fe/fixes.toml` is the ledger
source; regenerate its documentation index with
`python -m tools.cudnn_fe.verify_fixes --write-ledger`.

## Project-local Agent Skills

Repository-specific workflows live under `.agents/skills/`. Load the matching `SKILL.md`
before changing the covered subsystem:

- [`worktree-env-setup`](.agents/skills/worktree-env-setup/SKILL.md) —
  isolated per-worktree `.venv` with nightly PyTorch via the CI-mirroring uv flow.
- [`validating-pytorch-custom-ops`](.agents/skills/validating-pytorch-custom-ops/SKILL.md) —
  registration, fake implementations, autograd, `opcheck`, and `torch.compile` validation.
- [`cutedsl-tunable-kernel-template`](.agents/skills/cutedsl-tunable-kernel-template/SKILL.md) —
  typed config generation, cached TVM-FFI compilation, parallel compile, and sequential tuning.
- [`authoring-warpdrive`](.agents/skills/authoring-warpdrive/SKILL.md) —
  warp-specialized CuTeDSL kernels from roles, channels and work streams, or ports to that API.

## Agent Scratch Space

If you need scratch space for intermediate files, drafts, or temporary artifacts, use the `agent_space/` directory. This directory is gitignored and will not be checked in.
