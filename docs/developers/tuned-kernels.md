# Tuned kernels: design notes

This page records why tuned kernels are built the way they are. The rules themselves live in
`AGENTS.md` ("Public Operation Options" and "Tuned Kernels and Reproducibility"); the user-facing
contract is in [Determinism](../determinism.md#autotuned-kernels).

## The problem

[Issue #777](https://github.com/meta-pytorch/attention-gym/issues/777): torchtitan's DeepSeek-V4
training was not reproducible under `torch.use_deterministic_algorithms(True)`. Four cold-cache
runs of one 4-GPU job gave four different step-1 losses.

Each kernel candidate was repeatable. The winner was not: it is picked by timing, and candidates
differ in floating-point order. Forcing one candidate at a time on a GB200 showed how widespread
this was:

| Operation | Tuners with changed bits | Largest observed difference |
|---|---|---|
| `gather_attn` (Triton) | All eight reached | Output: 7.8e-3 (BF16); gradients: 3.1e-2 |
| `chunk_gdn` portable backward, K=V=64 | Three of four | Gradients: ~2e-3 |
| `chunk_gdn` portable backward, K=V=128 | None | — |
| `chunk_gdn` forward | None | — |

Tile sizes changed bits as well as `num_warps`, so restricting the search space to bit-identical
candidates would have given up most of the tuning.

## Decisions

1. **Tuners carry no policy.** `TritonTuner` and `run_tunable` tune when asked and otherwise
   launch a pinned configuration. They never read `torch.are_deterministic_algorithms_enabled()`.
   The alternative, having the tuner skip benchmarking in deterministic mode, hides a public
   behavior inside infrastructure and would not have covered kernels using `triton.autotune`.
2. **One policy function.** `should_autotune(requested)` is false in deterministic mode. The
   operation resolves it and passes a bool down. An autograd backward resolves it again, because
   eager code can enable deterministic mode after the forward.
3. **Triton pinning selects the first usable candidate.** `tuner.pinned` selects the first
   candidate left by `prune` that the device can run. CuTeDSL uses `default_config`; the
   recurrent scan uses a host shape heuristic when untuned. Skipping unusable candidates matters:
   on 99 KiB shared-memory devices some candidates do not fit, and tuning already skipped them.
   The choice is remembered per tuning key, so the key must determine what `prune` keeps
   (`_FWD_SHARED` gained the device key tile for this reason).
4. **One tuner type.** Every tuned Triton kernel uses `TritonTuner`. `triton.autotune` kernels
   needed a second pinning path that launched the first listed config with no pruning and no
   fallback. Migrating them removed that path. Pinned results stayed bit-identical and warm tuned
   latency was unchanged within noise; two keys that held exact sizes are now bucketed.
5. **`autotune` is a kernel option.** It was a keyword argument on the linear operations and a
   `kernel_options` key on `gather_attn`. It now has one spelling: a key of the typed
   `kernel_options` dict. This broke the keyword on purpose. `fastmath` stays a keyword argument
   because it changes the result, not only the kernel that computes it.

## What pinning does and does not give

- For otherwise deterministic kernels, bits repeat across processes and winner-cache states on
  one GPU model and software stack. The pinned configuration follows input shapes and the device,
  so it can differ across GPU architectures.
- Pinning does not remove other nondeterminism. `gather_attn`'s default backward uses atomics;
  only deterministic mode switches to the schedules without them.
- State-mutating recurrences (paged scans, the `paged_chunk_gdn` recurrence) are never tuned.
- `paged_mla_decode` keeps its earlier default of not tuning; when asked to tune, each bucket
  must run once before CUDA graph capture.

## Things that were tried and dropped

- **Tuning only `num_stages`** for the two kernels in the issue: the broader sweep found
  candidates that changed bits in all eight gather tuners reached and in three of four
  portable GDN backward kernels at K=V=64.
- **Changing the tuner's warm-up or timing policy.** Over 30 cold sweeps of a roughly 7 µs
  kernel, the existing policy picked the fastest candidate 21 times; an untimed first launch
  picked it 27 times, and `triton.testing.do_bench` picked it 23 times. Median sweep time was
  42 ms for the existing policy and 776 ms for `do_bench`. The policy was left unchanged.

## Checks to repeat when touching a tuned kernel

- Pinned bits: `{"autotune": False}` output and gradients equal the base commit's.
- No benchmarking when pinned: the `no_benchmarking` fixture in `test/conftest.py`.
- Winner reuse across shapes: count `tune` calls under `isolated_winners`.
- Warm tuned latency before and after, with the winners picked on each side.
