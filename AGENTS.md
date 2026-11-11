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
- `profiler/` — intra-kernel profiler recordings on small CuTeDSL kernels

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

## Public Operation Options

Keep the math in keyword arguments and the kernel in `kernel_options`:

- An option that changes what is computed (`scale`, `fastmath`, gate transforms, masks) is a
  keyword argument, and the reference implementation honors or documents it.
- An option that only chooses how a fused kernel runs (`backend`, `schedule`, `autotune`,
  `bwd_recompute_p`, `batch_invariant`) is a key of the operation's typed `kernel_options`
  dict (`attn_gym/types.py`, `attn_gym/linear/types.py`). New kernel knobs are added there, not
  as keyword arguments. Every implementation validates the options: unknown keys and wrongly
  typed values raise. `impl="reference"` then ignores them, so one dict serves both.
- One option has one spelling across operations. Breaking a public signature to restore that is
  acceptable; say so in the PR.

## Tuned Kernels and Reproducibility

Candidates of one kernel usually differ in floating-point order (tile sizes, `num_warps`), so a
timing-picked winner changes output bits between processes, ranks, and cold winner caches
(issue #777). Every operation that tunes must also run pinned;
[`docs/developers/tuned-kernels.md`](docs/developers/tuned-kernels.md) records why:

- Tune Triton kernels with `TritonTuner` (`attn_gym/_backends/triton/tune.py`), never
  `triton.autotune`. List the heuristic default first: pinning selects the first candidate
  left by `prune` that the device can run. The tuning `key` must determine what `prune` keeps.
  Tune CuTeDSL kernels with `run_tunable`; pinning selects `default_config`.
- Launch `(tuner if autotune else tuner.pinned)[grid](...)`. The recurrent scan instead keeps
  its host shape heuristic for untuned launches. Neither untuned path benchmarks or reads the
  winner cache.
- Tuners do not read deterministic mode. Operations resolve `kernel_options["autotune"]` with
  `should_autotune` (`attn_gym/_backends/autotune.py`), then pass the bool through every launch
  and custom-op schema. Deterministic algorithms disable tuning. An autograd backward, and the `run`
  of a staged handle, resolve the stored request again because eager code can change the mode;
  `gather_attn` resolves inside its CUDA ops.
- `autotune` defaults to `True` except in `paged_mla_decode`. `{"autotune": False}` or
  deterministic mode pins every tuned kernel in forward and backward. State-mutating recurrences
  cannot be benchmarked safely and stay pinned regardless of the option; their docstrings state
  this.
- Migrating a kernel between tuners, or changing its candidates, key, or prune, must leave the
  pinned result bit-identical (compare against the base commit) and needs a warm before/after
  latency table for the tuned path.
- Test both ways to pin with the `no_benchmarking` fixture (`test/conftest.py`): forward and
  backward must not benchmark under `{"autotune": False}` or deterministic mode with default
  options.
  Compare bits where schedules match; `test/test_gather_attn_autotune.py` compares forward
  outputs and LSE because deterministic mode changes its backward schedules.

## Adding CuTeDSL Kernels

Compile every CuTeDSL kernel through a module-level function decorated with `@jit_cache` from
`attn_gym.warpdrive.cache`; see [`docs/warpdrive/cache.md`](docs/warpdrive/cache.md) for the
full contract and the `cutedsl-tunable-kernel-template` skill for the tuning adapter. Entries
live under `WARPDRIVE_CACHE_DIR` (default `$XDG_CACHE_HOME/warpdrive`). A disk entry is reused
until the compile arguments, compile target, toolchain versions, codegen environment variables
(`attn_gym/warpdrive/cache/_key.py`), or the source of the compile module and every `attn_gym`
module it imports transitively change. Imports are found statically, so:

- Reach traced code with ordinary `import`/`from` statements (or a string-literal
  `importlib.import_module("attn_gym.x.y")`); computed module names are invisible.
- `import pkg.sub.module` does not hash `pkg/sub/__init__.py`; keep `__init__` files to
  re-exports.
- Use `extra_sources=(...)` only for files read while tracing or code outside `attn_gym`; never
  for modules already imported. For a class defined outside the compile module's imports, include
  `module_source_digest(cls.__module__)` in its `__cute_cache_key__()`.
- Host-only code tracing never runs may be imported at call time by relative name
  (`importlib.import_module(".mod", __package__)`) to stay out of the key; never traced code.

A kernel loaded from disk was never traced: arguments restore what tracing recorded through
`__cute_trace_record__`/`__cute_restore_trace__`, and `jit_cache(uncached=predicate)` compiles
selected calls without either cache when the trace itself is needed. To test specialization
reuse, assert on `compile_fn.cache_info()` (distinct keys in `currsize`, launches in
`hits + misses`) instead of setting `CUTE_DSL_NO_CACHE`, which forces cold compiles.

## warpdrive

`attn_gym/warpdrive/` helps write warp-specialized CuTeDSL kernels (roles, channels, work
streams) and is the CuTeDSL authoring base of the repository: `cache/` (compile cache),
`compat.py` (CuTeDSL version shims) and `profiler/` (intra-kernel profiler). Nothing under it
imports the rest of `attn_gym` (`test/warpdrive/test_warpdrive_boundary.py`). Load the
[`authoring-warpdrive`](.agents/skills/authoring-warpdrive/SKILL.md) skill when authoring or
porting a kernel. Docs live under [`docs/warpdrive/`](docs/warpdrive/index.md) and must match
the checkout being used. Fuzz report labels name cases in `test/warpdrive_fuzz/{static,mutate}.py`;
they are not a public error-code API.

Warpdrive kernels compile through `jit_cache` with two extra rules:

- Pass a kernel's static choices to its constructor: the key is the constructor call, and a
  public attribute assigned afterwards (other than `cluster`, work streams, and the IKET,
  tracing and alias settings) raises.
- A cached kernel restores its trace results (`describe()`, TMA/alias reports, warnings) without
  tracing. Only a test that asserts on tracing itself (IKET ranges, monkeypatched trace hooks,
  the PTX of a compile) bypasses the cache: it takes the `traced_compiles` fixture
  (`test/conftest.py`) and calls `compile_fn.cache_clear()`, or calls
  `compile_fn.__wrapped__(...)`, which compiles that one call without either cache.

### warpdrive Testing Tiers

Pick the smallest tier that covers the change; run bigger tiers only when asked or before
landing core synchronization/scheduling changes.

- Every change (~45 s cold, ~22 s with a warm compile cache on a GB300):
  `pytest -n 12 test/warpdrive test/profiler test/warpdrive_fuzz`. Tests marked `fuzz` are
  skipped by default (`addopts` deselects the marker); the unmarked regressions in
  `test/warpdrive_fuzz/test_static.py` are compile-only and run here.
- Quick fuzz (~2-3 min, on request or before landing layer changes):
  `pytest -n 12 -m fuzz test/warpdrive test/warpdrive_fuzz`. It includes the mutant check:
  each synthetic layer bug in `test/warpdrive_fuzz/runtime_mutants.py` must fail or hang its
  listed test (four hang by design, 90 s each).
- Long campaigns, sanitizer and CUTracer runs, SASS/perf A/Bs: only when the user asks.
  Use `tools/warpdrive/fuzz_warpdrive.py {static,mutate}` (compile-only).
  `CUTE_DSL_ARCH=sm_100a python tools/warpdrive/warpdrive_sass.py dump OUTDIR` records the
  seven standard configs; `compare BEFORE AFTER` gates on PTX and compiler identity and prints
  SASS diffs (`--strict-sass` also requires identical SASS). ptxas can assign registers
  differently from identical PTX, so a SASS-only diff with identical PTX is not a code change.

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
