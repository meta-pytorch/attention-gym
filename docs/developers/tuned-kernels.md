# Tuned kernels: design notes

This page records why tuned kernels are built the way they are. The rules themselves live in
`AGENTS.md` ("Public Operation Options" and "Tuned Kernels and Reproducibility"); the user-facing
contract is in [Determinism](../determinism.md#autotuned-kernels).

## The problem

[Issue #777](https://github.com/meta-pytorch/attention-gym/issues/777): torchtitan's DeepSeek-V4
training was not reproducible under `torch.use_deterministic_algorithms(True)`. Four cold-cache
runs of one 4-GPU job gave four different step-1 losses.

Each kernel candidate was repeatable. The winner was not: it is picked by timing, and candidates
differ in floating-point order. Forcing one candidate at a time showed how widespread this was
(GB200, BF16, revision `ff06a922`, deterministic mode, each alternate candidate compared with the
first):

| Operation | Tuners with changed bits | Largest observed difference |
|---|---|---|
| `gather_attn` (Triton) | All eight reached | Output: 7.8e-3 (BF16); gradients: 3.1e-2 |
| `chunk_gdn` portable backward, K=V=64 | Three of four | Gradients: ~2e-3 |
| `chunk_gdn` forward, K=V=64 and 128 | None | — |

K=V=128 takes the CuTeDSL backward on that GPU, so the portable backward was not exercised there.

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
   fallback. Migrating them removed that path. Pinned results stayed bit-identical, and the same
   configuration runs at the same speed (see [Migration measurements](#migration-measurements));
   two keys that held exact sizes are now bucketed.
5. **`autotune` is a kernel option.** It was a keyword argument on the linear operations and a
   `kernel_options` key on `gather_attn`. It now has one spelling: a key of the typed
   `kernel_options` dict. This broke the keyword on purpose. `fastmath` stays a keyword argument
   because it changes the result, not only the kernel that computes it.

6. **Split-phase handles resolve the policy at each phase.** A staged `prepare` and its `run`,
   like an autograd forward and backward, can straddle a change of deterministic mode in eager
   code. Each launch boundary resolves the stored request again.

## What pinning does and does not give

- For otherwise deterministic kernels, bits repeat across processes and winner-cache states on
  one GPU model and software stack. The pinned configuration follows input shapes and the device,
  so it can differ across GPU architectures.
- Pinning does not remove other nondeterminism. `gather_attn`'s default backward uses atomics;
  only deterministic mode switches to the schedules without them.
- State-mutating recurrences (paged scans, the `paged_chunk_gdn` recurrence) are never tuned.
- The reference implementation validates `kernel_options` and ignores them, so one dict can be
  passed to either implementation.
- `paged_mla_decode` keeps its earlier default of not tuning; when asked to tune, each bucket
  must run once before CUDA graph capture.

## Things that were tried and dropped

- **Tuning only `num_stages`** for the two kernels in the issue: the broader sweep found
  candidates that changed bits in all eight gather tuners reached and in three of four
  portable GDN backward kernels at K=V=64.
- **Changing the tuner's warm-up or timing policy.** Over 30 cold sweeps of a roughly 7 µs
  kernel (the pointer output kernel, T=256 H=2, GB200), the existing policy picked the fastest candidate 21 times; an untimed first launch
  picked it 27 times, and `triton.testing.do_bench` picked it 23 times. Median sweep time was
  42 ms for the existing policy and 776 ms for `do_bench`. The policy was left unchanged.

## Migration measurements

Moving the `triton.autotune` kernels to `TritonTuner`, before (`a6a5a8e4`) against after, on
a GB200 with BF16 inputs, tuning enabled, each run in its own process after a fresh
tuning sweep. Lower is better. The portable GDN backward and the pointer output kernel do not
run on this GPU by default and were forced the way their tests force them.

| Workload | Before (µs) | After (µs) | After/before | Winner |
|---|---|---|---|---|
| GDN portable backward, dense T=16384 H=16 K=128 (host) | 2071 | 2072 | 1.00 | same |
| GDN portable backward, dense T=4096 H=8 K=128 (host) | 881 | 806 | 0.92 | same |
| GDN portable backward, packed 8×1k H=8 K=128 (host) | 941 | 950 | 1.01 | same |
| GDN portable backward, packed 32×250 K=64 (host) | 1013 | 1046 | 1.03 | same |
| `l2norm` forward and backward, [1, 16384, 8, 128] | 12.3 | 12.3 | 1.00 | ties |
| `recompute_w_u`, dense T=4096 H=8 | 29.3 | 29.3 | 1.00 | same |
| `recompute_w_u`, dense T=256 H=2 | 9.9 | 11.7 | 1.17 | 8 warps before; 4 warps in 2 of 3 runs after |
| `recompute_w_u`, persistent T=4096 H=8 | 31.4 | 31.5 | 1.00 | same |
| Output kernel, pointer path, T=256 H=2 | 7.3 | 7.5 | 1.02 | differs between sweeps |
| `chunk_kda` forward and backward, T=4096 H=4 | 117.9 | 118.1 | 1.00 | same |
| `recurrent_gdn` prefill, T=4096 H=16 | 2414.8 | 2414.4 | 1.00 | same |
| `recurrent_kda` varlen, 16 sequences, T=2048 | 200.7 | 200.8 | 1.00 | same |
| `recurrent_gdn` decode, 64 sequences, T=1 | 24.22 | 24.30 | 1.00 | same |

Rows marked "host" are synchronized wall-clock medians of five runs, which vary by about 10%
between runs on the smaller shapes; the others are CUDA kernel time, medians of three runs.
With matching winners, no latency change was observed beyond run-to-run variation. The one 1.17 row is a different pick between
two candidates the tuner times within its resolution (15.4 µs each including launch).

A winner-cache miss became cheaper, because `run_tunable` times fewer launches than
`triton.testing.do_bench`: `l2norm` backward 3.1–3.3 s to 0.17–0.27 s, `chunk_gdn` at K=64
1.5–1.7 s to 0.21–0.48 s, the recurrent scan 1.8 s to 0.47 s.

The measurement scripts were scratch files and are not checked in.

- **Bucketing the recurrent scan's sequence count.** Its key holds the exact count, so each new
  batch size tunes once. Sharing one winner across a power-of-two bucket was measured on a
  GB200 (H=16, K=V=128, BF16): no change where the winner agrees, but 1.12–1.34x slower between
  9 and 16 sequences at 256 and 2048 tokens, where the preferred value tile
  changes. The key was left exact.

## Checks to repeat when touching a tuned kernel

- Pinned bits: `{"autotune": False}` output and gradients equal the base commit's.
- No benchmarking when pinned: the `no_benchmarking` fixture in `test/conftest.py`.
- Winner reuse across shapes: count `tune` calls under `isolated_winners`.
- Warm tuned latency before and after, with the winners picked on each side.
