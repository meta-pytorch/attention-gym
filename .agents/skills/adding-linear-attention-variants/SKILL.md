---
name: adding-linear-attention-variants
description: Adds a new attn_gym.linear attention variant, or a new public operation or backend for one (fused training, paged prefill, decode, cuDNN), at GDN/KDA parity with a pinned numerical contract. Use when starting a variant such as GDN, KDA, or Mamba2, or when extending one with a new execution path.
---

# Adding a Linear-Attention Variant

Use this for a new `attn_gym/linear/<variant>/` package or a new public operation or backend on
one. Do not use it for FlexAttention masks or score mods, for tuning an existing CuTeDSL kernel
(`cutedsl-tunable-kernel-template`), or for upgrading the vendored cuDNN kernels
(`attn_gym/linear/_delta_rule/cudnn_fe/MAINTENANCE.md`). Registering an operator always also
follows `validating-pytorch-custom-ops`.

Before writing code, read `DESIGN.md` ("Linear attention API shape" and "Testing requirements")
and the `docs/linear.md` section of the variant you mirror.

## Ground rules

Each rule encodes a correction from the repository history; the cited PRs hold the precedent.

- **One public function per execution form:** `chunk_<v>` (training, prefill), `recurrent_<v>`,
  `paged_chunk_<v>`, `recurrent_<v>_decode`. `impl` selects fused or reference and never falls back
  silently; `kernel_options={"backend": ...}` (`attn_gym.linear.types.KernelOptions`) chooses
  among optimized backends, and unsupported options raise (#318, #379, #401).
- **Layouts:** token-major `[B, T, H, D]`; packed batch-one inputs with device `cu_seqlens`;
  persistent state value-major `[N, H, V, K]` (#402); paged pools `[num_slots, ...]` routed by
  `state_indices` (non-positive indices are null routes: zero output, untouched slot) and
  `has_initial_state` (`_delta_rule/paged_state.PagedState`, #440).
- **Reference first:** an eager chunked and a token-by-token recurrent reference that compute in
  `promote_types(x, float32)`, so FP64 inputs keep FP64 math and state (#348). Optimized kernels
  never import the reference.
- **Numerical contract before kernels:** read [numerics.md](numerics.md) before writing the
  reference, any fused kernel, or any tolerance.
- **Operators:** torch-only `ops.py`, `define`/`impl` plus fakes that match the real allocations
  (shape, dtype, and strides; #626, #627), an `autograd.Function` outside the raw ops, `opcheck`
  with dense-permuted inputs, and `torch.compile(fullgraph=True)` through the public function.
- **Share before copying:** `_delta_rule` (paged state, decode output and inference-only checks),
  `attn_gym/_backends` (`TritonTuner`/`run_tunable`, `jit_cache`), and `attn_gym/testing`
  helpers. Optional backends load lazily.

## Landing order

Prefer one PR per step, in this order (use `stack-pr` or `gh stack`, not ghstack); GDN and KDA
landed each step on its own, which kept every PR reviewable.

1. Reference, public API, validation, CPU tests, and the `docs/linear.md` section.
2. Fused training path: forward, backward, custom ops, opcheck and compile, numerics contract.
3. Inference: fused recurrent scan, paged prefill, one-token decode.
4. Optional backends: CuTeDSL kernels, cuDNN (vendored through `tools/cudnn_fe`).
5. Integration: `examples/linear/`, `benchmarks/chunk_delta.py`, CUDA Graph capture, context
   parallelism (stages and state summaries). Add the variant to the training example and run its
   overfit loop (`examples/linear/delta_rule_training.py` fits a fixed target): a loss that stalls,
   diverges between backends, or turns NaN is the cheapest end-to-end bug detector after the
   kernel tests. Put loss curves for the fused path, the eager reference, and any upstream
   implementation the variant claims parity with (same weights, data, and seeds) in the PR, with
   the commands to reproduce them.

Read [checklist.md](checklist.md) when planning a variant or reviewing a variant PR: it lists every
parity surface with its GDN/KDA precedent. A PR that must carry several steps lists the parity
gaps it leaves in its description.

## Performance claims

Several items here and in [numerics.md](numerics.md) come from the Mamba2 integration that lands
with this skill.

- Benchmark the model's real input path (raw step sizes plus bias with an in-kernel softplus) as
  well as pre-transformed inputs. The Mamba2 Triton path took 1.17-1.25x as long on the model
  path as on pre-transformed inputs until the transform was computed once per call instead of in
  every kernel; afterwards it took 0.88-0.95x as long.
- Use one harness for every baseline (`triton.testing.do_bench` medians, forward and
  forward+backward) and run it twice; upstream autotuned baselines can swing by 60% between runs.
- Report shapes `(B, T, H, G, ...)`, dtype, GPU, and commit, label ratio direction, and profile per
  kernel before optimizing. Check shapes with `B * H` below about twice the SM count: serial scans
  are latency-bound there.

## Kernel pitfalls hit in this repository

- CUDA grid axes 1 and 2 stop at 65535 programs: put `(sequence, head)` programs on axis 0.
- Widen slot and offset arithmetic to int64 before multiplying by strides.
- Config prune rules must leave a candidate for small dimensions (a `BP` prune rejected every
  tile for `P < 32`).
- Autotuning a kernel that advances a paged pool in place advances it once per candidate: pass
  `TritonTuner(reset=...)` that snapshots only the routed slots, never the whole pool.
- Ops that write `out` or a state cache must reject aliasing with `torch._C._overlaps`, as
  `_delta_rule/decode.py` does.
- Triton elementwise epilogues on `tl.dot` accumulators (an output gate, a skip term) can cost
  more than a separate elementwise kernel; reload a tensor for its accumulator-layout use rather
  than reusing the MMA operand copy. Keep `numel` specialized in elementwise kernels;
  `do_not_specialize` on it cost 2.5x through lost vectorization.
- Check `n_regs`/`n_spills` of every candidate config (compiled-kernel metadata) before
  restructuring a slow kernel; spills, not FLOPs, explained the slow Mamba2 chunk backward.
- After changing a fake's dtype or layout without a schema change, use a fresh
  `TORCHINDUCTOR_CACHE_DIR`; a stale FX graph raises "expected dtype ... but got ...".
- Warm tuning caches skip tuning and its reset hooks; test tuned paths with the
  `isolated_winners` fixture pattern from `test/test_triton_tunable.py`.

## Completion

A variant change is done when:

- `pytest -n 6` passes for the variant's CPU and GPU tests, including the numerics file;
- every new numerical regression has been shown to fail on the previous commit;
- `mkdocs build` succeeds, and `docs/linear.md` and the `AGENTS.md` overview list the new
  operations;
- a fresh-context review of the full diff has run, plus a numerics-focused review by a separate
  reviewer briefed with [numerics.md](numerics.md) for any kernel or tolerance change.
