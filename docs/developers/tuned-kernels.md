# Tuned kernels: design notes

Authoring rules live in `AGENTS.md` ("Public Operation Options" and "Tuned Kernels and
Reproducibility"); the user contract is in [Determinism](../determinism.md#autotuned-kernels).

## The problem

[Issue #777](https://github.com/meta-pytorch/attention-gym/issues/777): four cold-cache runs of
one 4-GPU torchtitan DeepSeek-V4 training job gave four different step-1 losses under
`torch.use_deterministic_algorithms(True)`.

Each candidate was repeatable, but timing picks among different floating-point orders, changing
bits across processes, ranks, and cold winner caches. Forcing each candidate showed the scope
(GB200, BF16, revision `ff06a922`, deterministic mode, each alternate compared with the first):

| Operation | Tuners with changed bits | Largest observed difference |
|---|---|---|
| `gather_attn` (Triton) | All eight reached | Output: 7.8e-3 (BF16); gradients: 3.1e-2 |
| `chunk_gdn` portable backward, K=V=64 | Three of four | Gradients: ~2e-3 |
| `chunk_gdn` forward, K=V=64 and 128 | None | — |

K=V=128 uses CuTeDSL backward on that GPU, so portable backward was not exercised there.
Both tile sizes and `num_warps` changed bits; restricting candidates to bit-identical ones
would sacrifice most tuning.

## Decisions

- **Policy belongs to operations, not tuners.** `should_autotune(requested)` disables tuning in
  deterministic mode; operations pass the resolved bool to `TritonTuner` and `run_tunable`.
  Reading `torch.are_deterministic_algorithms_enabled()` in tuners would hide public behavior
  in infrastructure and would not have covered `triton.autotune` kernels. Autograd backward
  and staged handles' `run` resolve the stored request again: eager code can change the mode
  after forward or `prepare`.
- **Pin the first usable Triton candidate.** A `tuner[grid](..., autotune=False)` launch follows
  `prune` and skips candidates the device cannot run, as tuning already did: some do not fit
  devices with 99 KiB shared memory. The choice is cached per device and tuning key, so the key
  must determine pruning; `_FWD_SHARED` gained the device key tile for this reason. CuTeDSL pins
  `default_config`.
- **Express a shape heuristic as candidate order.** The recurrent scan's `prune` lists the
  configuration its shape heuristic picks first, so untuned and paged launches go through the
  tuner like every other pinned launch, bit-identical to the separate launch they replaced.
  When that configuration uses one warp it caps registers at 128, the only capped candidate:
  tuning now compares it with the uncapped tiles (GDN prefill T=4096 H=16: 959 → 935 µs),
  whereas capping every one-warp tile slowed KDA prefill (8×512, H=16) from 478 to 500 µs.
  A pinned launch costs about 6 µs more host time than the direct launch did: 55 → 61 µs per
  call for a single-token scan over 64 sequences (GB200, BF16, K=V=128). Kernel time is
  unchanged, so only host-bound scans notice.
- **Use one Triton tuner.** Migrating all `triton.autotune` kernels to `TritonTuner` removed a
  second pinning path that used the first listed config without pruning or fallback. Pinned
  results stayed bit-identical; matching configurations had unchanged speed
  ([measurements](#migration-measurements)). Two exact-size keys became bucketed.
- **Make `autotune` a kernel option.** Unifying linear operations' keyword argument with
  `gather_attn`'s typed `kernel_options` key deliberately broke the keyword. `fastmath` remains
  a keyword because it changes the result, not just the kernel.

## What pinning does and does not give

- Otherwise deterministic kernels repeat bits across processes and winner-cache states on one
  GPU model and software stack. Pinned configurations depend on shapes and device and may
  differ across GPU architectures.
- Pinning does not remove other nondeterminism: `gather_attn`'s default backward uses atomics;
  only deterministic mode selects schedules without them.
- State-mutating recurrences (paged scans, the `paged_chunk_gdn` recurrence) never tune.
- Reference implementations validate and ignore `kernel_options`, allowing the same dict for
  either implementation.
- `paged_mla_decode` retains its default-off tuning. When tuning, run each bucket once before
  CUDA graph capture.

## Things that were tried and dropped

- **Tuning only `num_stages`** for the issue's two kernels: the broader sweep changed bits in
  all eight gather tuners reached and three of four portable GDN backward kernels at K=V=64.
- **Changing warm-up or timing policy.** Over 30 cold sweeps of the roughly 7 µs pointer output
  kernel (T=256 H=2, GB200), fastest-candidate picks were 21 with the existing policy, 27 with
  an untimed first launch, and 23 with `triton.testing.do_bench`. Median sweep time was 42 ms
  with the existing policy versus 776 ms with `do_bench`. The policy stayed unchanged.
- **Bucketing the recurrent scan's sequence count.** Its exact-count key tunes once per new
  batch size. Sharing a winner across a power-of-two bucket (GB200, H=16, K=V=128, BF16) was
  unchanged where winners agreed, but 1.12–1.34x slower between 9 and 16 sequences at 256 and
  2048 tokens, where the preferred value tile changes. The key stayed exact.

## Migration measurements

`triton.autotune` → `TritonTuner`, before (`a6a5a8e4`) versus after: GB200, BF16, tuning enabled,
separate processes after fresh tuning sweeps. Lower is better. Portable GDN backward and the
pointer output kernel were forced as in their tests; neither runs by default on this GPU.

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

"Host" rows are synchronized wall-clock medians of five runs; smaller shapes vary about 10%
between runs. Other rows are CUDA kernel time, medians of three runs. Matching winners showed
no change beyond run-to-run variation. The 1.17 row picked differently between two candidates
timed within the tuner's resolution (15.4 µs each including launch).

Winner-cache misses became cheaper because `run_tunable` times fewer launches than
`triton.testing.do_bench`:

| Workload | Before | After |
|---|---|---|
| `l2norm` backward | 3.1–3.3 s | 0.17–0.27 s |
| `chunk_gdn`, K=64 | 1.5–1.7 s | 0.21–0.48 s |
| Recurrent scan | 1.8 s | 0.47 s |

Measurement scripts were scratch files, not checked in.

## Checks to repeat when touching a tuned kernel

Follow `AGENTS.md` for base-commit pinned-bit comparisons, `no_benchmarking`
(`test/conftest.py`), and warm before/after tuned latency; record each side's winners.
Check winner reuse across shapes by counting `tune` calls under `isolated_winners`.
