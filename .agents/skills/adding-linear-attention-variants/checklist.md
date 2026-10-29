# Linear-Attention Variant Parity Checklist

R marks surfaces GDN and KDA both provide (required for parity); O marks optional or
variant-specific ones. Precedents are the `origin/main` PRs that introduced each surface (a PR
title may name a different main change); read them with `git log --grep '(#NNN)'`.

## Public API (`attn_gym/linear/<v>/api.py`, `__init__.py`)

- R `chunk_<v>(..., initial_state, *, cu_seqlens, output_final_state, impl=Impl.FUSED,
  kernel_options)` returning `(output, final_state | None)`. GDN #234, #347; KDA #238, #260.
- R `recurrent_<v>` with `state_indices`/`has_initial_state` paged pools; the fused path is
  inference-only (`_delta_rule.validation.require_inference_only`). GDN #380, #381; KDA #317.
- R `paged_chunk_<v>` advancing `[num_slots, H, V, K]` pools in place. GDN #431; KDA #354, #349.
- R `recurrent_<v>_decode`: one token per sequence, preprocessing fused, caller-owned `out`
  (`resolve_decode_out`). GDN #391, #392; KDA #351.
- R `Impl`/`resolve_impl` from `attn_gym/linear/types.py` (#346, #347, #552).
- R `kernel_options` (`KernelOptions`, per-variant `resolve_kernel_options`). KDA #401; GDN #425;
  split options #600.
- O `ReplayState` batch-invariant replay decode (KDA #556); lazy backend exports through
  `attn_gym/linear/_lazy.py` (#471).

## Validation

- R a per-variant `validation.py` (GDN #341; shared delta-rule checks #400).
- R paged-state validation through `_delta_rule/paged_state.PagedState` (#440).

## References and precision

- R eager chunked and recurrent references, packed via per-sequence loops (#234, #238, #400).
- R the precision contract in code, docs, and tests: see [numerics.md](numerics.md) (#348, #446,
  #450, #464).

## Custom operators (`<v>/ops.py`)

- R `torch.library.define` schemas `attn_gym::<v>_*`, CUDA impls, fakes matching real
  allocations (#626, #627), an autograd wrapper, and mutation annotations for paged ops.
- R opcheck plus `torch.compile(fullgraph=True)` of the public functions (#380, #381, #300, #317).
- R dynamic-shape and kernel-reuse tests (#331, #565-#569;
  `test/linear/test_gdn_symbolic_schema.py`).

## Kernels

- R a Triton path tuned through `TritonTuner`/`run_tunable`, never `triton.autotune` (#601, #625).
  GDN and KDA use `<v>/fwd/triton/`, `<v>/bwd/triton/`, and `<v>/impl/`; a smaller variant may keep
  one `<v>/triton/` package.
- O CuTeDSL kernels compiled through module-level `@jit_cache` functions (`AGENTS.md`; #585).
- O a cuDNN backend in `<v>/impl/cudnn.py` and `cudnn_ops.py` (#369, #428), vendored through
  `tools/cudnn_fe` with a fix-ledger entry and guard test per upstream bug (#606-#612).

## Exports and tests

- R a `<V>_OPS` list in `attn_gym/linear/__init__.py` (#316) and export assertions in
  `test/test_namespaces.py` (#222).
- R reference and API tests that run on CPU, fused-path tests with opcheck and compile, decode
  tests, and a chunk-path test with `torch.cuda.graph` capture (#380, #392, #425). GDN keeps them
  in `test/linear/test_gdn*.py`; KDA's older suite lives in `test/test_kda*.py`.
- R a numerics contract file (pattern: `test/linear/test_delta_rule_state_precision.py`,
  `test/kda/cudnn/test_delta_rule_numerics.py`).
- O shared input builders in `attn_gym/testing/<v>.py` (#303, #427); add one as soon as a
  second test file needs the same factory (`test/AGENTS.md`).

## Docs, examples, benchmarks

- R a `docs/linear.md` section: recurrence and layouts, "Supported capabilities", a precision
  note, a snippet, and `:::` entries for every public function (#227, #253, #342).
- R a variant in `examples/linear/delta_rule_training.py`, tested by
  `test/test_delta_rule_training_example.py` (#499, #505); examples stay one directory deep with
  the variant in the filename. Its overfit run (`--backend=fused` and `--backend=reference`) with
  loss curves on identical weights, data, and seeds is the end-to-end evidence for the PR.
- R an entry in `benchmarks/chunk_delta.py` `OPERATIONS` with budgets (#429).
- R CUDA Graph and ragged capture support (`docs/guides/ragged-cuda-graphs.md`, #372).
- O context parallelism: `<v>/stages.py`, `<v>/context_parallel.py`, state summaries (#470,
  #471, #473, #415, #528).

## Gap table template

Before calling a variant complete, fill in a table of every R item above as have / partial /
missing with the file or test that proves it, and list the deliberate gaps in the PR body.
