# Examples

[Back to warpdrive](index.md)

Start with GEMM for the basic handoff; use the other examples for the feature you need.
These are teaching kernels with the input limits below, not drop-in replacements for every
PyTorch or FA4 feature.

| Learn | Source | Guide |
|---|---|---|
| Three roles, two rings, one CTA per tile; `WorkStream(schedule=OneShot())` | [gemm.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/gemm.py) | [GEMM](#gemm) |
| Four schedules, double-buffered accumulators, TMA stores, pairs, multicast clusters, `tile_n` | [persistent_gemm.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/persistent_gemm.py) | [GEMM](#gemm), [tile boundaries](#tile-boundaries) |
| Device offsets, stateful `WorkMap`, one descriptor per operand, ragged stores | [grouped_gemm.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/grouped_gemm.py) | [Grouped GEMM](#grouped-gemm) |
| Two softmax instances, correction, slots, fan-out/join, turnstiles, S/P alias, split-P, persistence and pairs | [fa4_fwd.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/fa4_fwd.py) | [Forward](#attention-forward) |
| Overlapping handles, `Warps` arrivals, TMEM/SMEM aliases, a 2-CTA pair with a DSMEM exchange and relay warp | [fa4_bwd.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/fa4_bwd.py) | [Backward](#attention-backward) |
| Device-built work table: host capacity, device length | [packed_work.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/packed_work.py) | [Scheduling](scheduling.md) |
| Split-KV items, per-sequence counts, round-robin softmax, named mailboxes, SMEM P | [fa4_decode_swapab.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/fa4_decode_swapab.py) | [Decode](#single-token-decode), [shared KV](#shared-kv-ring) |
| Whole-model persistent decode: host instruction table, static schedule, cooperative launch, epoch-tagged words between CTAs | [qwen_megakernel.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/qwen_megakernel.py) | [Qwen3 megakernel](#qwen3-06b-decode-megakernel) |
| Managed producer, raw `external="consumer"` with overridden count, native wait on a `NamedBarrier.id` | [raw_interop.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/raw_interop.py) | [Dropping down](dropping-down.md) |
| Handoffs/fixed costs against raw twins | [bench_overhead.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/bench_overhead.py) | [Fixed costs](#warpdrive-fixed-costs) |
| Schedules, clusters, widths and accumulator handoff against `torch.mm` | [bench_persistent_gemm.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/bench_persistent_gemm.py) | [Tile boundaries](#tile-boundaries) |

**Measurements are historical reference, not new evidence for this checkout.** Expand the
experiment sections for shapes, versions, methods, failed attempts and reproduction commands.
Compare columns from the same run; different replay/cache/clock contracts are not one dataset.

## GEMM

```python
import torch
from attn_gym.warpdrive.examples.gemm import gemm

a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
b = torch.randn_like(a)
d = gemm(a, b)  # A @ B.T
```

To use a persistent two-CTA kernel:

```python
from attn_gym.warpdrive import ClusterLaunchControl, CtaPair
from attn_gym.warpdrive.examples.persistent_gemm import persistent_gemm

d = persistent_gemm(a, b, schedule=ClusterLaunchControl(), cluster=CtaPair())
```

```text
load A/B -- ab ring --> MMA -- acc ring --> epilogue writes A @ B.T
```

Both wrappers require:

- BF16 `a: (M, K)`, `b: (N, K)` on one CUDA device; contiguous and 16-byte aligned.
- K and N multiples of 8.
- Optional `out`: BF16 `(M, N)`, same device, contiguous, 16-byte aligned, disjoint from
  both inputs. Empty outputs return early; K=0 writes zeros.

`PersistentGemm` defaults to two accumulator stages: the next mainloop can use one while
the epilogue drains the other. All stage choices must fit SMEM and TMEM.

For two pairs sharing A along N, pass `cluster=Cluster(2, 2, pair=True)`; see
[clusters](clusters.md#larger-clusters). At `tile_n=224`, this matches the cuBLAS 256x224 2x2
configuration observed for 3000x5000x4096.

<details>
<summary>Tile-width experiments and default choices (GB300, November 2026)</summary>


`tile_n` defaults to 224 for `CtaPair()` and 256 otherwise. The width that wins is mostly
wave quantization: how the tile count divides into the resident CTAs (152 CTAs, 76 pairs or 36
2x2 clusters on a GB300) and how full the last wave is, against the better per-FLOP efficiency
of the full 256-wide tile once there are many waves. So no single width wins everywhere; the
defaults are a compromise over the shapes below. CUDA-graph replay, GB300, CuTeDSL 4.8.0,
November 2026, `StaticPersistent`, min of five interleaved rounds' medians of 60 replays
(microseconds, lower is better):

| M x N x K | `torch.mm` | 1 CTA 224 | 1 CTA 256 | pair 192 | pair 224 | pair 256 | 2x2 192 | 2x2 224 | 2x2 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 x 4096 x 4096 | 75.2 | 83.2 | 87.9 | 77.9 | 72.4 | 81.2 | 77.5 | 87.1 | 79.4 |
| 3000 x 5000 x 4096 | 72.9 | 83.4 | 85.6 | 77.3 | 72.8 | 81.1 | 77.4 | 71.8 | 79.6 |
| 2048 x 8192 x 4096 | 79.2 | 83.3 | 86.0 | 77.9 | 72.9 | 81.2 | 78.8 | 87.5 | 81.2 |
| 8192 x 2048 x 4096 | 76.0 | 100.8 | 88.7 | 80.1 | 89.4 | 81.2 | 90.0 | 86.2 | 79.5 |
| 4096 x 4096 x 14336 | 264.2 | 292.9 | 306.9 | 282.3 | 269.0 | 290.2 | 277.5 | 309.4 | 277.3 |
| 4096 x 14336 x 4096 | 244.7 | 282.3 | 265.9 | 258.3 | 262.3 | 253.6 | 263.8 | 264.2 | 260.8 |
| 8192 x 11008 x 4096 | 393.1 | 454.3 | 442.4 | 416.3 | 441.3 | 410.5 | 438.0 | 448.6 | 410.5 |
| 6000 x 3000 x 2048 | 43.1 | 50.7 | 47.5 | 48.4 | 47.8 | 44.4 | 48.9 | 47.3 | 44.7 |
| 1234 x 4096 x 4096 | 30.1 | 38.7 | 42.7 | 34.3 | 39.3 | 42.8 | 34.9 | 39.9 | 43.2 |
| 1024 x 1024 x 8192 | 18.4 | 38.0 | 42.0 | 33.9 | 38.0 | 42.0 | 34.0 | 38.1 | 42.2 |

Two sweeps (3 and 5 rounds, also testing 128 and `ClusterLaunchControl`) supported these
choices:

- **Pair: 224.** Versus 256, geomean speedup was 1–3% in each of four sweep/schedule
  combinations (2.2% pooled), and repeatably 7–11% on 4096-class shapes. Width 192 gained
  0.5–3.5% (1.7% pooled), mostly on noisy large shapes.
- **Single CTA: 256.** Widths 224 and 256 tied within 0.6%.
- **2x2: 256.** Width 224 was the worst default: its 448-wide cluster tile gives N=4096 a
  fifth wave. Width 192 gained only 1–3%, within noise.
- **128:** won only the 32-tile 1024 x 1024 x 8192 case, by 24% over 192.

Most table shapes agreed within 5% between sweeps (8% for the two smallest), with the same
width ranking under CLC. Shapes taking at least 400 us—8192 x 11008 plus the omitted 8192^3
and 16384 x 4096 x 14336—moved up to 10% between sweeps and 19% between rounds on the shared
GPU; single-digit differences there are inconclusive.

A tile-count rule picked one-/two-wave winners but missed many-wave winners by 10–15%.
There is no shape-based selector; set `tile_n` for the shapes you measure.

The harness is a CUDA-graph A/B: one graph per configuration, captured once, replayed in
round-robin rounds (`torch.mm` first, then the widths in order) with default clocks and warm
L2, reporting each configuration's best round median. Rerun it at your shapes before trusting
a difference under 5%.

</details>

### Tile boundaries

Two accumulator buffers let MMA and epilogue overlap. `Umma` publishes asynchronously;
the epilogue can return TMEM as soon as its final read completes:

```text
early_release=True:   last tcgen05.ld -> wait::ld -> release TMEM -> SMEM/TMA store
 early_release=False: last tcgen05.ld -> wait::ld -> SMEM/TMA store -> release TMEM
```

`early_release=True` is the default. It changes **when TMEM is reusable**, not when the
output store finishes. Historically it helped mainly for short-K pairs, where epilogue time
is large relative to mainloop time.

<details>
<summary>Accumulator overlap and work-ring scope measurements</summary>

In the measured SASS, early release moved the arrive up to the last `LDTM`. Single-CTA
epilogues, including `cp.async.bulk.wait_group` pacing, were shorter than their mainloops
on every shape below.

`bench_persistent_gemm.py --variants mm single single_late single_clc single_acc1 pair
pair_late pair_clc pair_clc_late pair_atomic pair_acc1 pair_n256 pair_n256_late pair_n256_clc`,
GB300, CuTeDSL 4.8.0, November 2026, min of five interleaved rounds' medians of 10 replays of
20 calls, microseconds (`clc` is `ClusterLaunchControl()`, `atomic` `AtomicPersistent()`,
`late` is `early_release=False`, `acc1` is `acc_stages=1`; the pair is `tile_n=224` unless
`256`). The shared GPU ran about 10% slower than for the table above (`torch.mm` 82.4 vs 75.2
at 4096^3), so compare within a table:

| M x N x K | `torch.mm` | 1 CTA | 1 CTA late | 1 CTA clc | 1 CTA acc1 | pair | pair late | pair clc | pair clc late | pair atomic | pair acc1 | pair 256 | pair 256 late | pair 256 clc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 x 12800 x 1152 | 18.6 | 21.5 | 21.5 | 21.6 | 22.5 | 23.3 | 23.5 | 23.2 | 23.1 | 23.1 | 26.2 | 21.0 | 20.9 | 21.2 |
| 2048 x 8192 x 1024 | 20.2 | 25.8 | 25.2 | 25.0 | 27.9 | 22.3 | 22.5 | 22.3 | 22.7 | 22.9 | 25.7 | 24.9 | 24.9 | 24.7 |
| 4096 x 4096 x 512 | 11.8 | 15.7 | 15.7 | 15.4 | 16.7 | 13.0 | 13.3 | 13.0 | 13.3 | 13.0 | 15.8 | 14.0 | 14.4 | 14.0 |
| 4096 x 4096 x 4096 | 82.4 | 94.3 | 94.6 | 96.9 | 96.7 | 81.2 | 81.5 | 86.2 | 86.3 | 82.1 | 85.6 | 90.8 | 90.4 | 90.3 |
| 3000 x 5000 x 4096 | 78.5 | 90.3 | 90.2 | 93.5 | 91.2 | 77.9 | 77.8 | 81.9 | 79.1 | 79.1 | 82.1 | 88.0 | 86.9 | 85.8 |

The second accumulator stage is worth 1-22% (most at 4096 x 4096 x 512 on a pair, where the
epilogue is as long as the 8-k-tile mainloop; least at K = 4096 on one CTA). Early release is worth 0-3% on pairs at short K
(repeatable at 4096 x 4096 x 512 in two sweeps for both pair widths) and within noise elsewhere,
including by NCU duration at 1024 x 12800 x 1152 (`--clock-control none`, five launches each:
single 22.0 vs 21.6, pair 256 20.8 vs 21.1, pair 224 22.8 vs 23.1, pair clc 24.4 vs 24.2 early
vs late, against cuBLAS's 128x256 2-CTA kernel at 19.4). The short-K shapes trail `torch.mm`
by 10-25% for wave quantization (1024 x 12800: 400 single tiles or 200 pair tiles over 148 SMs
are 2.7 waves) and the store-bound 4096 x 4096 x 512 (32 MB out), not for tile-boundary gaps.

A separate work-ring scope fix removed a short-K scheduling gap:

1. **Before:** CLC/atomic pairs were 8–20% slower than static. First three shapes:
   24.2–24.4 vs 22.4–22.6, 23.1 vs 21.2, and 15.2–15.4 vs 12.7–12.8 us. Deeper rings did
   not help.
2. **Attribution:** `%globaltimer` probes at 4096 x 4096 x 512 found 0.7–1.2 us per item
   inside each role's ring **release**, although the slot was already full. Cluster-scope
   arrival emitted `MEMBAR.ALL.GPU` per receiving thread. Scheduler claim cost was only
   0.26–0.29 us CLC / 0.32 us atomic, plus 0.32 us decode and `st.async` publication.
3. **Fix:** payload readiness comes from completed bytes; releases publish nothing. Deriving
   `remote_scope="cta"` ([scheduling](scheduling.md)) reduced probed releases to 0.03–0.13 us
   and matched static's MMA item period.
4. **Remaining:** CLC tied static at short K but trailed by 0–6% at K=4096 (largest: 86.2 vs
   81.2 us; earlier: 82.8 vs 81.8). Unexplained: role starts were within 0.1 us of static,
   and the 0.6 us scheduler chain ran one work slot ahead, off the critical path.

</details>

## Grouped GEMM

```python
from attn_gym.warpdrive.examples.grouped_gemm import grouped_gemm

a = torch.randn(4096, 7168, device="cuda", dtype=torch.bfloat16)  # the experts' rows, concatenated
b = torch.randn(8, 2048, 7168, device="cuda", dtype=torch.bfloat16)  # one (N, K) weight per expert
offsets = torch.tensor([700, 700, 1500, 2100, 2100, 3000, 4000, 4096], device="cuda", dtype=torch.int32)
out = grouped_gemm(a, b, offsets)  # out[s_g:e_g] = a[s_g:e_g] @ b[g].T
```

`offsets` contains **end rows**, not sizes (`torch._grouped_mm` calls it `offs`):

```text
offsets = [700, 700, 1500, ...]
group 0: rows [0, 700)      group 1: empty      group 2: rows [700, 1500)
```

Contract:

- `a: (total_M, K)`, `b: (G, N, K)`, BF16; GEMM's device, contiguity, alignment and
  K/N-multiple-of-8 limits apply.
- `offsets: (G,)`, contiguous int32 on the same device; nonnegative, nondecreasing,
  at most `total_M`, and unchanged during execution. Values are read only on-device.
- `total_M + (G + 1) * 256` must fit int32. `out` must not overlap `a`, `b` or `offsets`.
- Empty/arbitrary-size groups are allowed; rows past `offsets[-1]` stay untouched. Invalid
  offset values are clamped to the previous end and A's end: wrong rows, but in bounds.

One compiled kernel handles dynamic `G`, `total_M`, `N`, `K` and routing. Options are
`schedule` (default `StaticPersistent()`), `cluster` (`SingleCta()` or `CtaPair()`),
`max_ctas`, `tile_n`, `ab_stages`, `acc_stages`, `c_stages`, `epi_tile_n` and `roles`.

**How work is found:** the host reserves an upper bound of
`ceil(total_M / tile_m) + G` m-tiles, since each ragged group adds at most one. `GroupedTiles`
numbers real tiles back to back, dropping only indices after the last tile. Its
[decode state](scheduling.md#decode-state) carries the prefix-sum walk between items,
testing eight groups per round trip instead of restarting.

**How stores stop at a group boundary:** A/out each use one descriptor plus
`cute.domain_offset`; B is `(N, K, G)`, with group as a TMA coordinate. The output's ragged
view has shape `(2^30, N, 2^31 - 1)`: its row mode ends at TMA's clipping boundary, and its
third coordinate selects the group's end row. A partial group tile is clipped like a tensor edge.

<details>
<summary>Grouped scheduling and backend comparisons (historical)</summary>

Exact tile numbering fixed an earlier imbalance. Binary search over a padded space left
one dropped slot per group, scattered through static units' strided sequences: some units
got 40 items instead of 36 at G=128, and static was 10–27% slower than broadcast schedules.
Caching only the search group did not help; the next index often lay several groups ahead.
With exact numbering, static tied atomic/CLC within 2% without a scheduler warp or workspace,
hence the default.

CUDA-graph replay on a shared GB300 (`bench_grouped_gemm.py`, 20 calls per graph, min of
five interleaved rounds' medians, PyTorch 2.15 nightly cu132), skewed routing with 1/8 of the
experts idle; microseconds per call and TFLOP/s on the real rows. `grouped_mm` is
`torch._grouped_mm` (`F.grouped_mm` is the same aten op and measured within 1%), `mm loop` a
`torch.mm` per group, `Triton` Inductor's persistent TMA grouped GEMM template
(`max_autotune` with Triton as the only GEMM backend), `QUACK` FlexGEMM's QUACK backend with
an identity epilogue (its untuned pick, a 128 x 128 tile on a 2-CTA MMA with a dynamic
persistent scheduler) and `QUACK tuned` the same with Inductor's `tuned=True` shortlist;
`pair` is `CtaPair()`:

| G, tokens, N x K | `grouped_mm` | `mm loop` | `Triton` | `QUACK` | `QUACK tuned` | 1 CTA static | 1 CTA atomic | 1 CTA CLC | pair static |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8, 4096, 2048 x 7168 | 115 (1049) | 121 | 102 | 106 (1135) | 122 | 95 (1268) | 94 | 92 | 116 |
| 8, 16384, 7168 x 2048 | 318 (1511) | 319 | 361 | 382 (1260) | 323 | 337 (1429) | 335 | 338 | 319 |
| 64, 8192, 4096 x 4096 | 397 (693) | 533 | 382 | 370 (742) | 381 | 378 (726) | 373 | 378 | 396 |
| 64, 32768, 2048 x 7168 | 788 (1220) | 936 | 808 | 890 (1081) | 837 | 767 (1255) | 787 | 783 | 810 |
| 128, 16384, 7168 x 2048 | 700 (687) | 860 | 661 | 681 (706) | 657 | 639 (753) | 628 | 624 | 688 |
| 128, 32768, 4096 x 4096 | 1065 (1032) | 1252 | 1119 | 1207 (911) | 1136 | 1091 (1007) | 1104 | 1102 | 1131 |

Small skewed groups waste rows in 128-row tiles for every kernel (1.5x the real rows at
G=64 with 8192 tokens and G=128 with 16384, 1.13-1.25x at 32768); a pair's 256-row tiles
waste about twice that, so `CtaPair()` leads only on the G=8 shape with N=7168.

Interpretation of this run:

- Untuned QUACK led only at G=64 / 8192 tokens, by 1–2%: its 64 rows per CTA waste fewer
  rows. It trailed one-CTA static by 7–16% on the other five shapes.
- Tuning helped G=8 / N=7168: a 256 x 256 two-CTA MMA tile in a 2x2 cluster multicasts A
  along N and tied `pair`. It hurt the smallest case: Inductor chose its winner using
  evenly spaced offsets, not this skewed routing.
- FlexGEMM TRITON without `max_autotune` used extern `_grouped_mm`; NVGEMM had no grouped
  choice (`NoValidChoicesError`).
- QUACK also numbers tiles exactly using a cached warp prefix sum over offsets. Its
  swizzled bands aim for L2 reuse; these timings did not isolate a gain from that raster.

</details>

## Attention forward

```python
from attn_gym.warpdrive.examples.fa4_fwd import fa4_fwd

out, lse = fa4_fwd(q, k, v)
```

This teaching kernel computes **unmasked attention** using `flash_attn.cute` and `quack`
helpers. It does not implement the full FA4 API.

| Input/output | Contract |
|---|---|
| Q/K/V | BF16 `(batch, seqlen, heads, head_dim)`, contiguous, 16-byte aligned, one CUDA device |
| Dimensions | Head dimension 64 or 128; matching batch/head counts; K and V shapes match |
| Sequence lengths | Positive key length divisible by `tile_n` (128 or 64); ragged Q allowed |
| O / LSE | O has Q's shape; LSE is fp32 `(batch, heads, seqlen_q)`, natural-log units |
| Not supported | Varlen or GQA |

The tiling needs `2 * (tile_n + head_dim) <= 512` TMEM columns. Wrapper checks are not a
complete validator of rank, head dimension or device requirements; enforce the full contract
in another wrapper.

Choose [schedule and mapping](scheduling.md) with `schedule=`/`mapping=` (`Raster` or FA4
mappings), and [role placement/register budgets](roles.md#choose-placement-in-the-constructor)
with `roles={...}`. `tile_n=64` halves the K/V tile; `rescale_threshold` controls FA4's
lazy rescale; head dimension follows the tensors. See [configuration](configuration.md)
for all knobs and fixed choices.

K/V share channel-bound SMEM. The constructor-only `overlap_o_q=True` also reuses Q's bytes
for epilogue SMEM; `fa4_fwd` does not expose it. This requires a non-persistent schedule
(`OneShot()` among built-ins). Persistence would need an epilogue-to-load ordering edge:
**sharing an address does not create a handoff**.

### Hardware row max on SM103

```python
from attn_gym.warpdrive import ClusterLaunchControl, CtaPair

out, lse = fa4_fwd(
    q,
    k,
    v,
    cluster=CtaPair(),
    schedule=ClusterLaunchControl(),
    split_p=True,
    use_ldred_rowmax=True,
)
```

`use_ldred_rowmax=True` combines score loading and maximum reduction with `tcgen05.ld.red`:

```text
128 scores -> four 32-column hardware maxima -> existing software row-max updater
```

- SM103 only; an SM100 target raises `ValueError`. Default is false, retaining the software path.
- Rescale threshold, exponentials, BF16 P conversion and row sums are unchanged, including
  fast-exp2's flush-to-zero rescale behavior.
- Tests compare O/LSE bitwise: ragged Q, nonuniform scales, repeated launches, persistent
  wraps, and subnormal rescale amplified by large finite V. SASS and wrapper kernel-name
  checks verify that the hardware path runs.

<details>
<summary>Hardware row-max, scheduling and rejected tuning experiments</summary>


Run the fixed-buffer, interleaved CUDA-graph comparison against FA4's default:

```bash
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=5s 60s \
  .venv/bin/python -m attn_gym.warpdrive.examples.bench_fa4_fwd \
  --shape 0 --output agent_space/fa4_gap.json
```

Repeat with `--shape 1` and `--shape 2`. Shapes are `(batch, heads, query/key length)`:
`(4,16,4096)`, `(2,16,8192)` and `(8,16,1024)`, all BF16, head dimension 128, non-causal.
Each graph has 100 forwards. The script settles DVFS for three seconds, alternates variant
order over seven rounds, and saves all 15 replay samples per round. Numbers below are the
minimum round median, in microseconds per forward; setup/capture are excluded. FA4 timing
uses `flash_attn.cute.flash_attn_func(..., causal=False)` with its defaults; the example
also writes LSE. Inputs use seed 0, normal-distributed `torch.randn` values rounded to BF16.
Buffers are reused, not rotated for cold HBM timing.

GB300, CuTeDSL 4.8.0, PyTorch `2.15.0.dev20260922+cu132`, Triton `3.8.0+gitc01b6774`,
FA4 `00807a02`, warpdrive base `e4f3ede` (November 2026), normal DVFS without clock locking.
GPUs were shared; sub-percent differences are not stable wins.

| Forward | 4x16x4096 | 2x16x8192 | 8x16x1024 |
|---|---:|---:|---:|
| CLC split-P, software max | 342.56 | 645.15 | 57.19 |
| CLC split-P, hardware max | 336.67 | 632.54 | 56.70 |
| Static persistent split-P, hardware max | 337.49 | 634.53 | 54.28 |
| FA4 default | 336.79 | 634.74 | 55.21 |

| Change | Time delta (lower is better) | Kept? |
|---|---|---|
| Hardware max, same CLC/register split | -1.72%, -1.95%, -0.85% in shape order above | Opt-in |
| Static instead of CLC, with hardware max | +0.24%, +0.31%, -4.27% (repeat: -2.24% small shape) | Existing schedule option |
| Match FA4 registers: 176/80/80 instead of 192/80/48 | No consistent gain; +0.56% at 8x16x1024 | No default change |
| exp2 emulation, frequency 16 or 8 | +2.85% / +3.86% at 8x16x1024 (exploratory) | Removed |
| Registers 200/64/48 or 184/96/48 | +0.26% / +0.31% at 8x16x1024 (exploratory) | No default change |

Hardware max leaves CLC **-0.04%, -0.35%, +2.69%** relative to FA4. The small-shape gap is
mostly scheduling: this FA4 revision defaults to **static persistence**, not CLC. Matching
that schedule leaves **+0.21%, -0.03%, -1.69%** in this run, but the small-shape static minimum
was a transient low round. A separate repeat was **+0.37%** (55.34 vs 55.14 us), while CLC
was +2.67%. Static reaches approximate parity, not a repeatable advantage. These results do
not justify a shape-based schedule selector.

Matched NCU 2025.4.1 kernel-replay reports (`--cache-control none`, 16 passes) explain the
row-max change: at 4x16x4096, dynamic warp instructions fell from 128.43M to 112.41M,
with 128 launch registers/thread and 232,448 bytes dynamic SMEM unchanged. SASS replaced
264 `FMNMX` instructions with 10 and adds `LDTM.STAT.x32.MAX.F32`; the register split and
local accesses are unchanged. NCU duration fell 428.06 -> 416.80 us; it is a different
replay/cache/clock contract from graph timing, not the timing authority (default NCU clock
control, average GPC clocks about 1.20 GHz and memory clocks 4.00 GHz). DRAM traffic stays
near 201 MB read and 73 MB written: hardware max reduces instructions, not input traffic.

At 8x16x1024, CLC's grid has 512 CTAs while static and FA4 have 152, with capacity for
76 resident clusters of two CTAs. Static avoids the work-broadcast/CLC machinery:
NCU duration 79.62 -> 76.74 us, eligible warps per scheduler 0.370 -> 0.386
(FA4: 76.74 us, 0.426). DRAM read/write traffic stays near 100/39 MB and resources are
unchanged, so this is not a traffic or residency improvement. The same mbarrier stats
channel remains in the faster static example. Named-barrier stats and correction-warp rescheduling were not changed; these
measurements do not identify either as the remaining bottleneck.

For NCU, use `--profile` with `--profile-from-start off`: the script warms all variants,
then profiles one complete forward per variant between CUDA profiler start/stop. All
seven standard example configurations kept identical normalized SASS versus the then-current
`origin/main` on SM100 and SM103; only the opt-in hardware-max variant intentionally changed it.
This is a historical check, not a comparison against today's remote branch.

</details>

## Attention backward

```python
from attn_gym.warpdrive import CtaPair
from attn_gym.warpdrive.examples.fa4_bwd import fa4_bwd

dq, dk, dv = fa4_bwd(q, k, v, out, dout, lse, causal=False)  # FA4's 1-CTA kernel
dq, dk, dv = fa4_bwd(q, k, v, out, dout, lse, causal=False, cluster=CtaPair())  # FA4's default
```

This ports both head-dimension-128 main kernels of FA4's `flash_bwd_sm100.py`: its 1-CTA
kernel (`FlashAttentionBackward`, which FA4 selects with `disable_2cta`) and its default
pipelined 2-CTA kernel (`FlashAttentionBackward2Cta`, `cluster=CtaPair()`).

```text
FA4 preprocess -> warpdrive main kernel -> FA4 dQ postprocess
  dPsum = rowsum(dO * O)
  LSE_log2 = LSE * log2(e)
  zero fp32 dQ accumulator
```

Contract:

- BF16 `(batch, seqlen, heads, 128)` inputs, contiguous, 16-byte aligned, one CUDA device;
  matching batch/head counts, no GQA. Forward LSE: fp32 `(batch, heads, seqlen_q)`.
- Arbitrary sequence lengths: TMA zero-fills partial loads, keys past `seqlen_k` are masked,
  and dK/dV stores are clipped. `causal=True` is bottom-right aligned.
- No varlen, paging, block sparsity, score mods, softcap or deterministic mode.

**1-CTA.** Each `OneShot()` item is a 128-key tile. `Raster()` over `(n_blocks, heads, batch)`
matches FA4's `SingleTileScheduler` order. Four roles use 14 warps: reduce 0–3, two compute
warpgroups 4–11 sharing each tile, MMA 12, load 13. Eleven channels (seven with `Warps`
sides) pipeline five MMAs per tile through overlapping handles, three TMEM aliases and two
SMEM epilogue overlaps.

**2-CTA.** A pair owns a 256-key item and the leader issues `CtaGroup.TWO` MMAs; a fifth role,
the relay warp 14, joins them. S^T, dP^T, dV and dK keep their rows in their CTA, so the compute
warps and epilogue are the 1-CTA code (shared helpers). The other operands split differently
across the pair: Qt, dOt and Kt are separate TMA loads, and dQ = dS K needs each CTA's dS for
the other's query rows:

```text
compute (CTA r): dS^T for its 128 keys x 128 queries
   own query half  -> own sdS, key block r
   peer's half     -> sdSx --cp.async.bulk.shared::cluster--> peer's sdS, key block r
                                     completes on peer's xchg barrier (armed by thread 0)
relay (CTA r):   wait xchg ----------------------------------------\
compute (both):  sdS written and S_{t+1} read --------------------- ds full (leader, 18 arrivals)
                                                                    -> leader issues dQ_t
```

Thirteen channels (those with an MMA side pair-scoped), one raw barrier, and TMEM S / dV / dP /
dK with P over S, dS over dP and this CTA's 64 dQ columns over S's tail. Both
[class docstrings](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/fa4_bwd.py) map channels, TMEM and MMA order.

### Correctness

On every shape below and in the tests, dK and dV are bit for bit FA4's results from the same
kernel configuration (1-CTA against `disable_2cta`, the pair against FA4's default), non-causal
and causal, including ragged lengths. dQ differs only by the order in which CTAs' TMA
reduce-adds land in the fp32 accumulator (max |diff| up to 9.8e-4 against values of order 1;
FA4 is not bitwise reproducible there either, and some runs are bitwise equal). All three are
checked against an fp32 autograd reference, within FA4's own error (identical maxima for the
pair at 2x16x1024 and 4x16x3000x2900, both modes). dK/dV are deterministic across launches;
dQ is not, as in FA4.

### Timings

<details>
<summary>Both ports against both FA4 kernels, and what closed or did not close the 2-CTA gap</summary>

Fixed-buffer CUDA-graph replays of one whole backward call (preprocess, main kernel,
postprocess) per variant, seven interleaved rounds of 30 replays after three seconds of
warmup; CUPTI gives the main kernel. Minimum round medians in microseconds, lower is better.
GB300, CuTeDSL 4.8.0, PyTorch `2.15.0.dev20260922+cu132`, FA4 `00807a02`, normal DVFS,
shared GPUs, November 2026; `fa4_bwd.py` SHA-256 prefix `fa7ad2de9b7dd9f4` (SASS identical to
`de18eb84fb39c0c7`, which renames two locals):

```bash
gpu-run --timeout 600 auto -- timeout --signal=TERM --kill-after=10s 1500s \
  .venv/bin/python -m attn_gym.warpdrive.examples.bench_fa4_bwd \
  --output agent_space/bwd.json  # add --causal for the causal rows
```

Main kernel (CUPTI). *2-CTA* is `cluster=CtaPair()`, *1-CTA* the default; the last column is
the whole graph:

| batch x heads x seqlen_q x seqlen_k | 2-CTA port | FA4 default (2-CTA) | 1-CTA port | FA4 1-CTA | 2-CTA port / FA4 2-CTA | 2-CTA / 1-CTA port | 1-CTA port / FA4 1-CTA | graph 2-CTA port / FA4 2-CTA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2x16x1024x1024 | 39.7 | 40.8 | 41.8 | 42.0 | 0.973 | 0.951 | 0.995 | 0.991 |
| 4x16x4096x4096 | 868.0 | 895.4 | 976.7 | 973.1 | 0.969 | 0.889 | 1.004 | 0.971 |
| 1x16x8192x8192 | 825.9 | 849.9 | 965.0 | 958.7 | 0.972 | 0.856 | 1.007 | 0.972 |
| 4x16x3000x2900 | 514.3 | 521.8 | 533.2 | 530.9 | 0.986 | 0.965 | 1.004 | 0.989 |
| causal 2x16x1024x1024 | 35.4 | 36.1 | 36.0 | 36.0 | 0.982 | 0.984 | 1.000 | 0.992 |
| causal 4x16x4096x4096 | 503.0 | 510.7 | 551.1 | 551.3 | 0.985 | 0.913 | 1.000 | 0.986 |
| causal 1x16x8192x8192 | 470.4 | 479.1 | 539.0 | 536.2 | 0.982 | 0.873 | 1.005 | 0.983 |
| causal 4x16x3000x2900 | 306.2 | 311.1 | 330.5 | 330.2 | 0.984 | 0.926 | 1.001 | 0.989 |

The 2-CTA port is 1.4-3.1% faster than FA4's 2-CTA kernel on every shape (0.8-2.9% for the
whole graph) and 1.6-14% faster than the 1-CTA port; the 1-CTA port is within -0.5% to +0.7%
of its FA4 kernel. The non-causal rows are a second run: in the first, FA4's 2-CTA kernel at
4x16x4096 (935.9) and our 1-CTA kernel at 1x16x8192 (1033.3) were outliers on the shared GPU.
Before the compute-loop changes below (SHA-256 prefix `21e79205ba496356`), the 2-CTA port
trailed FA4's 2-CTA kernel by 1.1-4.1% non-causal and 5.2-6.4% causal.

What was measured on the 1-CTA port (historical, at its SHA-256 prefix `ceb4035159cd5e55`,
against FA4's 1-CTA kernel), 4x16x4096 unless noted (NCU 2025.4.1 kernel replay,
`--cache-control none`, and interleaved in-process CUPTI A/Bs):

| Change | Effect | Kept? |
|---|---|---|
| One elected arrival per warp (`Warps`) instead of 256 per-thread arrivals per release | NCU cycles 1.889M -> 1.862M (FA4 1.822M); stall long-scoreboard 6.6 -> 5.8 warps per issue | Yes |
| dV/dK published right after their last MMA, as FA4 does, instead of after the item's last dQ MMA | -1.7% at 2x16x1024, -2.8% at 4x16x4096 (minimum round medians) | Yes |
| S_0 waits only for K; V before the first dP (FA4 arms V with dO_0) | -0.9% at 2x16x1024 | Yes |
| FA4's `gemm_ptx_w_idx` for the SMEM/TMEM MMAs instead of `cute.gemm` | Within noise | Yes (FA4's code) |
| No peek wait before S_{t+1} (timing-only, incorrect) | Within noise | Peek kept |
| One release fence for dPsum and dS instead of two | Within noise | No |
| FA4's two extra compute-wide barriers before the S and dS signals | +0.6% | No |
| Dropping the second dS-over-dP barrier | +6% | No |
| In-graph workspace zero fill (`launch_args` before the placeholder fix) | +4 us whole-graph at 2x16x1024 | Fixed in the layer |

With `Warps` the port executes 1.6% more warp instructions than FA4 (285M vs 280M: the
`bar.warp.sync` and elect per signal; with per-thread arrivals it executed 3% fewer), with
the same opcode mix otherwise, 128 launch registers and 231 vs 232 KB of SMEM. The remaining
excess is in mbarrier wait stalls (about +4,000 of 66,700 PC samples, at the barrier-pointer
lines both kernels' waits are attributed to), not in issued work; which wait carries it is not
attributed. Swapping `Threads` for `Warps` in the forward, decode and persistent GEMM examples
moved them by -0.8% to +2.3% (outputs bitwise equal), so `Threads` stays their default.

What was measured on the 2-CTA port, interleaved in-process CUDA-graph A/Bs of the main kernel
(4x16x4096, 1x16x8192 and 4x16x3000x2900, both modes unless noted) and NCU 2025.4.1 kernel
replay (`--cache-control none`, base clocks) of 1x16x8192 and 4x16x3000x2900:

| Change or observation | Effect | Kept? |
|---|---|---|
| ds commit deferred with `commit(signal=False)` and a raw arrival after S_{t+1}'s load (FA4's loop shape) instead of a rotated compute loop that closes the handle in its own iteration | Causal -1.5% to -2.4%; non-causal within ±1% | Yes |
| Register split 128/136/112 (reduce/compute/rest) instead of FA4's 136/136/104 | Within noise (removes 8 of the MMA warp's 10 spill instructions) | No |
| Causal compute 144 registers (reduce 120), which removes the compute warps' causal spills | +0.8% with the deferred commit | No |
| No `release_fence` on dpsum: make_ds fences and syncs every compute thread after its last dPsum read (the only layer-emitted sync op FA4 does not have, from the sync-skeleton diff below) | -0.3% to -1% on 5 of 6 shapes | Yes |
| dQ MMA as `gemm_ptx_w_idx` or one unrolled `cute.gemm` instead of FA4's two-group `gemm_w_idx` | 214 / 108 local-memory instructions instead of 10 (compile only) | No |
| One-tile items (64x16x128x256), the layer's fixed per-CTA costs | +3.5% vs FA4 2-CTA (0.27 us per wave of 74 pairs) | — |
| Register split 104/152/104 with reduce (warpgroup 0) and load/MMA (warpgroup 3) in one `setmaxnreg` dispatch branch, as the layer merged equal budgets before, vs the same split in two branches (4x16x4096 and 1x16x8192, GB300, 15 shuffled rounds) | Merged is +1.2% to +1.6% causal (ptxas spills seven of the MMA warp's hoisted descriptors inside its loop), +0.6% to +0.8% non-causal (within noise); 96/152/112 is another 2% faster causal. 104/152/112 and 112/152/104 do not fit the 65536-register pool | The layer never merges non-adjacent warpgroups now |

The compute-loop changes (make_ds; A/Bs against FA4 at 4x16x4096 and 1x16x8192, both modes,
each change on top of the ones kept before it):

| Change | Effect | Kept? |
|---|---|---|
| The peer's query half of dS stored straight to sdSx, which the ds acquire already guards, instead of held in registers until after the stage loop (FA4's form, which its serial schedule needs: there sdSx aliases the dQ staging buffer) | -1.3% to -5%; the 2-CTA compute loop's static SASS from 561 to 541 instructions (619 to 587 causal; FA4's causal loop has 583), causal spills gone | Yes |
| dPsum read before each stage's dP TMEM load and barrier instead of after | -1.2% to -2.3% | Yes |
| ds acquired once both dS stages are in TMEM, then sdS / sdSx written, instead of at stage 0 before the first SMEM store (FA4's place, where the pair's compute warps waited for the previous dQ MMA: 881 NCU samples at causal 1x16x8192 to FA4's 3, whose compute warps reach it later after waiting longer for S, 1790 samples to our 989) | -2% to -3.6%; static loop 508 instructions (553 causal) | Yes |
| ds acquired before the dP load | +2% to +4% | No |
| dP released before that acquire | +0.4% to +2.6% | No |
| One P-over-S barrier after both stages' exp2 | Non-causal within noise, causal +1.4% to +1.8% | No |
| Stage 0's LSE read before the S wait | +2.8% to +3.5% (1x16x8192 non-causal within noise) | No |
| Register split 96/152/112 instead of 136/136/104 (the best split on the earlier loop) | Within ±0.4% (11 rounds) | No |

The kept changes also run in the 1-CTA kernel (shared code): within noise to 1.2% faster.

Before those changes the gap to FA4 was spread over the tile loop rather than fixed costs (it
grew with the tiles per item: 1x16x8192 trailed most), and it was not synchronization:

- **Sync skeleton.** For each frequency class of instructions (a role's loop body) the
  ordered fences, barriers, mbarrier waits/arrivals, commits, TMEM loads/stores and MMAs of
  both kernels, aligned (a script over NCU's per-instruction source export). The compute,
  dQ reduce and MMA loops issue the same synchronization in the same order; FA4's compute loop
  has one more compute-wide barrier per tile. The only extra op in ours was the dPsum release
  fence (removed above).
- **Code generation.** The compute loop executed 3% more instructions per tile (464 vs 451
  non-causal; extra register moves and re-read thread indices, and two spilled values when
  causal) and spent ~6% more stall samples, half of them waiting at the fence and barrier
  after the sdSx stores. The per-section NCU comparison then showed the same arithmetic in
  both loops: the extra instructions were the sdSx staging moves and spills, and the largest
  stall difference was the ds acquire, which the changes above remove or overlap. ptxas also
  hoists five of the MMA warp's SMEM descriptors out of the loop and spills them (FA4 spills
  none); removing that spill with a larger MMA budget did not change the time.
- Each CTA keeps all 16 warps until the exit cluster barrier, and pair-scoped producers drain
  before it (FA4's warps exit as they finish). This shows as +10% NCU stall samples, all idle
  warps at the barrier, but the gap does not track it (one-tile items are +3.5% vs FA4).

**Without a reference.** On our 2-CTA kernel alone, before the compute-loop changes (1x16x8192
non-causal unless noted):

- Roofline: 10 b h s_q s_k d FLOPs in 894 us is 1.54 PFLOP/s (FA4 1.60); NCU reports the tensor
  pipe 66% busy (FA4 73%, the same 1.57M UTCHMMA instructions) and SMs active 95%.
- IKET with waits (1x4x4096, one wave): the MMA warp spends 64% of its item time waiting,
  24% on `ds.wait` and 17% on `dp.acquire` (dS ready / stored over dP), only 5% on input loads;
  the compute warps wait 27-29% (12% at `compute_sync`, 3% each on s, dp, lse, dPsum and ds);
  the load warp waits 92% (on stage reuse).

Both views pointed at the compute warps as the critical role (the MMA waited on them, not on
data), which is where the FA4 diff found the excess. Neither view alone separated "compute
does too much work" from "compute is scheduled badly"; the instruction count per tile and the
sync skeleton made that distinction. Only the reference turned a 66% busy tensor pipe into a
known 3.8% (1x16x8192) shortfall.

</details>

### What it needed from the layer

- `Warps` channel sides, added for this port (above).
- `fence_async_shared()` had never run under tracing: the event's `kind` field collided with
  the recorder's parameter. Fixed.
- `launch_args()` zeroed a per-stream workspace even for `OneShot`, so graphs captured on a new
  stream replayed a memset. Fixed: one untouched placeholder per device.
- [Previews](channels.md#previews) separate **reading the next Q** from **owning its epoch**:
  S_{t+1} needs Q_{t+1} while dK_t still holds Q_t. The port uses `q.peek()` and
  `preview.view("Q", read=True, by="umma")`; the next `q.wait()` owns that epoch (the peeled
  tile's wait for the last preview). FA4 instead rotates `handle_Q = handle_Q_next`; carrying
  that handle here would retain the pre-loop trace record and look like a double close.
  The former raw implementation used `q.full_barrier(q.index)`, `q.phase()` and `params.sQ`.
  Both causal modes kept identical SASS and alias reports after switching to previews;
  the historical timings therefore still applied. The 2-CTA kernel previews dO the same way
  (dP_{t+1} reads dOt before dV_{t+1}'s `do.wait()`).
- The 2-CTA kernel needed no layer change: pair-scoped channels derive the leader-armed TMA
  bytes, multicast commits and joined releases; a second producer role joins a channel (the
  relay on ds, 18 arrivals); `Barriers` carries the exchange.

Dropped down, each documented at its site:

- The last tile's S acquire (and, in the pair, dP acquire) has no successor to publish and is
  retired. That, the dK/dV epilogue tiles over Q and dO (over K and V in the pair), and
  `Q_STAGE = 2` (one Q stage deadlocks the 1-CTA schedule: S_{t+1} waits for a load that needs
  Q_t's release) are protocol facts the layer cannot check; both kernels stay `OneShot`, as FA4's.
- dQ reduce and dK/dV epilogues order their SMEM staging with declared `NamedBarrier`s (allocated
  IDs 3-6, the hand-picked IDs they replaced) and `bulk_commit_group`/`bulk_wait_group`, as in
  the persistent GEMM epilogue.
- The pair's dS exchange: compute thread 0 arms the peer's `xchg` mbarrier
  (`mbarrier.arrive.expect_tx` on the peer) and issues `cp.async.bulk.shared::cluster` from
  sdSx into the peer's sdS; the relay waits on its own `xchg` with a hand-kept phase. Reuse of
  sdS/sdSx and of the barrier follows from the ds channel: a CTA rewrites them only after its
  ds acquire, which waits for the dQ MMA that waited for both relays.
- The pair's ds commit: `dsh.commit(signal=False)`, then `signal_ds` (warp sync, one elected
  remote arrival on the leader's full barrier) after the next S load.

Layer gaps found by the 2-CTA port, with proposals:

| Gap | Workaround here | Proposal |
|---|---|---|
| No channel one CTA fills for its pair peer (a peer exchange). `Tma(fills="pair")` only lets a leader fill both CTAs, and TMA producers must be single-warp roles. | Raw `Barriers(1, counts=1)`, remote arm + `cp.async.bulk.shared::cluster` from compute thread 0, relay warp waiting on it | `Tma(bytes, fills="peer")`: producers in every CTA, each arming its peer's full barrier and filling its stage (with `Cluster.fill(peer)` masks), consumers in every CTA; let a multi-warp producer issue it from one elected thread. The relay then becomes an ordinary consumer that produces on ds. |
| A handle must close in the dynamic region that opened it, so a commit deferred past the loop's back edge (FA4's "dS_t is published once S_{t+1} is read") needs either a rotated loop, which carries P_t in registers across the back edge (1.5-2.4% slower causal), or a raw signal. | `commit(signal=False)` + raw `signal_ds` | A checked carry for open handles, like try tokens: `pending = h.defer()` returns a loop-carried value whose `commit()` must happen before the end's next acquire, with the last one closed after the loop. |
| Raw signals need hand-written targets: the remote arrival on the leader takes a rank and a local pointer. | `mbarrier_arrive(ds.full_barrier(0), peer_cta_rank_in_cluster=0)` | `h.signal()` on a handle closed with `signal=False` (or `end.signal(index)`), emitting the managed signal (count, targets, fences) where the caller places it. |
| No fence kind for TMEM loads: a release meaning "I have read this TMEM" needs `tcgen05.wait::ld` written by hand. | `fence_view_async_tmem_load()` before `signal_ds` | `release_fence="tmem_load"` / `commit_fence="tmem_load"`. |
| `commit_fence` applies to every producer of a joined side; the relay would run the compute warps' fences. | ds declares no fence; compute fences before the exchange | Per-role fences on a joined side. |
| The alias diagnostic does not model CTA pairs: every alias is `unverified`. | — | Already listed under [Still missing](#still-missing). |
| Producers of pair-scoped channels tail and every CTA joins an exit cluster barrier, so idle warps stay resident; FA4's exit as they finish. | — | Measure an opt-out for `OneShot` kernels whose final waits already cover the peer's accesses (no regression found attributable to it here: one-tile items are +3.5% vs FA4). |

Diagnostics that stay incomplete:

- P~S, dS~dP, dQ~dP and dQ~dS are `unverified`: unknown epoch correspondence, not reported
  hazards. Channel round trips order them, but the checker needs matching per-item counts.
  MMA traces two iterations **plus a peeled tile**; compute/reduce trace two iterations.
  The report names those channels/counts without a warning.
- Modeling this needs loop bounds and peeled iterations—not artificial counts that could
  hide a missing publication/release. Compute's exact TMEM column annotations do allow its
  cross-warpgroup ordering to be checked.
- In the pair every alias is `unverified` (pairs are not modeled); every TMA load is verified.

### Size

Code lines: tokens other than comments, docstrings and blank lines, per line (Python's
`tokenize` and `ast`; it gives 948 for the 1-CTA port before the 2-CTA kernel, where the
historical count was 924):

| Source | Code lines |
|---|---:|
| Both ports, including wrapper/compile helper/checks | 1,299 |
| `FlashAttentionBackward2Cta` alone (load and MMA schedules, relay warp) | 200 |
| FA4 `flash_bwd_sm100.py` (all paths) | 4,448 |

FA4's remaining paths include varlen, GQA, block-sparse, score-mod, deterministic, folded,
serial 2-CTA and M=64 variants. Shared pipeline/mask/copy helpers are excluded; this is not an
equal-feature size comparison.

## Single-token decode

```python
from attn_gym.warpdrive.examples.fa4_decode_swapab import fa4_decode

out, lse = fa4_decode(q, k, v, num_splits=8, seqused_k=lengths, splits_per_seq=counts)
```

This ports FA4's `flash_fwd_decode_sm100.py`. Swapping the MMA operands puts 128 KV tokens
along M and one KV head's query heads along N:

```text
S^T = K Q^T -> softmax -> P^T -> O^T = V^T P^T
```

| Input | Contract |
|---|---|
| Q | BF16, contiguous `(batch, 1, heads, d)` |
| K / V | BF16, contiguous `(batch, kv_len, kv_heads, d or d_v)` |
| Heads | Query heads divisible by KV heads, at most 32 per KV head |
| Dimensions | `d` divisible by K step (default 64); `d_v` divisible by 128. Defaults 128/128; 256 works for either |
| `seqused_k` | int32 per-sequence lengths; rows past the length must be finite—FA4's bounded V descriptor is not ported |
| `splits_per_seq` | int32 per-sequence counts, at most `num_splits`; caps splits as FA4's prepare kernel does |
| Not supported | Paging, varlen Q, masks, score mods |

`decode_partials` returns fp32 O `(splits, batch, 1, heads, d_v)` and LSE
`(splits, batch, heads, 1)` in FA4's combine layout. Splits past a sequence's count are
unwritten; kept splits with no KV blocks write O=0, LSE=-inf. `fa4_decode` uses FA4's combine
kernel and returns BF16 O `(batch, 1, heads, d_v)` and fp32 LSE `(batch, heads, 1)`.

The roles keep FA4's placement: QK and PV MMA warps, K/V-load plus split-LSE warp,
Q-load/O-store warp, two alternating softmax warpgroups, and correction warpgroup.

- `SplitKV` supplies `(split, kv_head, batch)` items. Per-sequence counts are read after
  `PdlWait()`; indices past the count are dropped.
- `Sharing.ROUND_ROBIN` on S, P and row-max/row-sum mailboxes gives softmax[i] blocks
  i, i+2, … with one warpgroup's counts. **OneShot only:** after an odd block count, another
  item would start on softmax[1]'s epochs.
- Thread mailboxes use NamedBarrier-backed channels and a turnstile for the shared row-max
  exchange. Three one-shot `NamedBarrier` flags finish split statistics.
- P occupies a 4-stage SMEM ring as PV's B operand, with no TMEM alias.

`SwapABDecode` configures `tile_n` (default 256), `tile_k` (64), S/P depths (4/4), head
dimensions and role registers; wrappers forward these keywords. See
[configuration](configuration.md) for valid ranges. Measurements use FA4-matching defaults.

Default K/V storage is **split rings**, 7/5 stages for 8-head tiles. `shared_kv=True` selects
[FA4's shared ring](#shared-kv-ring), with two MMA cursors and two-block K lookahead. Other
port differences: the first PV overwrites its O slot instead of zero-filling TMEM first;
empty splits use the normal tail with zero sums.

Tests/recorded comparisons match FA4 partials and combined outputs bitwise on the shapes
below, using FA4's actual split counts and table captured at its combine call.

<details>
<summary>Initial decode comparison and split-ring try-token experiments</summary>

The initial timings below are CUPTI medians of CUDA-graph replays, in microseconds; the better
of two interleaved rounds of 200 replays. Graph times also include FA4's prepare kernel (per-sequence
rows) and the shared combine. Buffers are reused, so short shapes run from L2. These initial
numbers used the earlier benchmark; the command below runs the updated method described
under decode setup costs.

```bash
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=5s 900s \
  .venv/bin/python -m attn_gym.warpdrive.examples.bench_fa4_decode
```

GB300, CuTeDSL 4.8.0, PyTorch `2.15.0.dev20260922+cu132`, FA4 `00807a02`, November 2026,
shared GPUs.

| batch x kv_len, heads/kv_heads | splits | FA4 decode | port | port / FA4 | graph FA4 / port |
|---|---|---:|---:|---:|---:|
| 1x8192, 8/1 | 16 | 4.22 | 4.70 | 1.114 | 16.40 / 16.69 |
| 1x32768, 32/1 | 32 | 7.04 | 7.90 | 1.123 | 19.39 / 19.18 |
| 4x16384, 32/8 | 4 | 39.81 | 41.18 | 1.035 | 49.02 / 49.09 |
| 8x32768, 32/8 | 2 | 151.10 | 149.70 | 0.991 | 159.41 / 157.63 |
| 8x65536 ragged, 8/1 | per-seq <= 32 | 26.24 | 28.18 | 1.074 | 36.80 / 36.80 |
| 16x32768 ragged, 64/8 | per-seq <= 10 | 180.38 | 180.37 | 1.000 | 190.30 / 190.11 |
| 2x131072 ragged, 16/1 | per-seq <= 32 | 9.28 | 9.47 | 1.021 | 21.41 / 21.38 |

In that initial run, long splits tied and short ones were 2-12% (0.2-1.9 us) slower. Under NCU
(1x8192, cold caches) the port issued 5% more instructions and spent more stall cycles on long
and short scoreboards. Its other resources matched FA4's: 68 registers, no spills, 512 threads.
Changes measured and ruled out as the cause, each within noise: allocating TMEM from warp 1 as
FA4 does (later measured at -0.4 to -0.7% and made the default; see
[fixed costs](#warpdrive-fixed-costs)), aligned instead of unaligned named barriers, K/V stage splits from 4:7 to 9:3
(FA4's shared ring has no direct equivalent), decoding the item and reading `seqused_k` once in
the prologue, skipping all setup for dropped splits (slower: it serializes on the PDL wait), and
prefetching the TMA descriptors (kept, as FA4 does).

[Try tokens](channels.md#try-tokens) were measured on the split rings as FA4 uses them, in
`fa4_decode_swapab.py` as of `5c8db04` (SHA-256 prefix `5aac3058290b9a8f`) behind a temporary
constructor option, not kept:

- PV probed V before waiting for P/acquiring O, consumed the token at its first V wait,
  then reprobed after each release.
- QK likewise probed K before acquiring S.
- An "all" variant also probed P, O and S, as FA4 does.

Every variant kept partials bitwise equal to FA4's. Decode-kernel CUPTI medians of graph-replayed `decode_partials` (nine or eleven
rotating rounds of 200 replays after three seconds of warmup, three processes on one GB300,
CuTeDSL 4.8.0, PyTorch `2.15.0.dev20260922+cu132`, FA4 `00807a02`, November 5, 2026), in
microseconds, median over the processes; lower is better:

| batch x kv_len, heads/kv_heads | no tokens | K/V tokens | delta |
|---|---:|---:|---:|
| 1x8192, 8/1 | 4.289 | 4.352 | +1.5% |
| 4x16384, 32/8 | 40.256 | 40.352 | +0.2% |
| 8x32768, 32/8 | 148.368 | 148.448 | +0.1% |
| 8x65536 ragged, 8/1 | 28.065 | 28.064 | 0.0% |
| 16x32768 ragged, 64/8 | 179.810 | 179.794 | 0.0% |
| 1x32768, 32/1 | 7.360 | 7.328 | -0.4% |
| 2x131072 ragged, 16/1 | 8.928 | 8.928 | 0.0% |

Split-ring tokens were not consistently neutral-or-better, so blocking waits stayed:

- 1x8192: K/V tokens were 4.352 us in all three processes; baseline medians were 4.288–4.416.
- 4x16384 and 8x32768: 0.01–0.3% slower in every process. Other shapes moved at most one
  32 ns CUPTI tick, in either direction.
- "All" variant, two processes: within 0.4%, except 1x8192 (+1.5% and -3.6%).
- Ragged split rings gained nothing, unlike the shared prototype's 0.06 us precheck gain.

Consumer tokens did not cross loop iterations. Carrying a producer token across blocks,
as FA4's loader does, was not tried.

</details>

### Decode setup costs

<details>
<summary>Fast-divmod change, setup timeline and rejected alternatives</summary>


`SplitKV.bind()` builds shape-dependent `FastDivmodDivisor` values (`FastDivmodDivisorV2` from
CuTeDSL 4.7, via `_compat`) once on the host.
Roles use them instead of repeating general integer division, as in FA4's scheduler. Batch,
heads and splits remain dynamic; per-sequence counts are still read after `PdlWait()`.
Partials remained bitwise equal to FA4.

Matched before/after, GB300, same versions as above and Triton `3.8.0`, November 4, 2026.
The benchmark source SHA-256 prefix was `78d54f3cf2c58af0`, against the mapping at `c205c64`
and fast-divmod source `18088cc627d69597`. These source hashes were recorded before the
jit_cache migration; the three decode builds' normalized SASS stayed identical after it.
One GPU reservation covered three process pairs in A/B, B/A, A/B order, each with three
alternating FA4/port rounds of 200 replays after three seconds of warmup.
Fixed buffers, seed 0, normal-distributed BF16 inputs, normal DVFS without clock locking.
The table gives CUPTI decode-kernel medians across the process medians, in microseconds;
lower is better. Prepare and combine are excluded from these kernel times; whole-graph
samples are recorded too, but were noisy. No synchronization or layer code changed.

| batch x kv_len, heads/kv_heads | before | fast divmod | delta |
|---|---:|---:|---:|
| 1x8192, 8/1 | 4.863 | 4.704 | -3.27% |
| 1x32768, 32/1 | 8.001 | 7.872 | -1.61% |
| 4x16384, 32/8 | 41.055 | 40.433 | -1.52% |
| 8x32768, 32/8 | 149.984 | 149.090 | -0.60% |
| 8x65536 ragged, 8/1 | 28.223 | 28.065 | -0.56% |
| 16x32768 ragged, 64/8 | 180.673 | 180.193 | -0.27% |
| 2x131072 ragged, 16/1 | 9.600 | 9.279 | -3.34% |

This comparison was run in transformer_nuggets, where warpdrive lived before moving here. To
repeat it there, archive the `transformer_nuggets` package at `c205c64` into scratch space, copy
that repository's later `bench_fa4_decode.py` (which imports `transformer_nuggets`) into the
archive, and run the same benchmark with `python -P` and `PYTHONPATH` pointing at each source
tree, interleaved under one `gpu-run` reservation. `--output` records the actual kernel source
hashes, so the archive (which has no Git checkout) is not mislabeled as the current commit;
check them before comparing. This repository's copy imports `attn_gym.warpdrive` and so always
measures the current kernel.

Matched NCU 2025.4.1 at 1x8192 (`--cache-control none --clock-control none`, kernel replay,
12 passes) counts 121,600 -> 111,184 executed warp instructions (-8.57%). FA4 counts 117,200:
removing the extra instructions alone does not close the gap. SASS for the static 8-head
mapping falls from 2,992 to 2,664 instructions: two mapping divides in each of seven role
instances lose 14 `MUFU.RCP`, 14 `F2I` and 28 `IABS` sites. Both port builds still use
68 registers and 216,096 bytes dynamic SMEM, at 0.11 launch waves; no spills. NCU duration is
7.296 -> 6.912 us, under its separate replay contract. The fixed buffers fit in L2, and this
report's DRAM reads are nearly zero. Source-level PC sampling has too few samples to assign
hundreds of nanoseconds to one particular load or wait.

Sparse IKET (all 16 CTAs and 16 warps, entry marker plus four pairs per warp) localized
setup/body delay. Times are ns from each CTA's first marker, **instrumented boundaries,
not production latency**:

| Boundary | FA4 | Port |
|---|---:|---:|
| Post-prologue CTA sync | 672 | 480 |
| Dispatched body start | 672 | 736–928 |
| Critical O-store finish | 3,200 | 3,904 |
| Median CTA lifetime, body markers | 3,936 | 4,512 |
| Median CTA lifetime, trampoline-only `--warp-trace` | 3,904 | 4,512 |

Outputs matched bitwise. The port decodes items, retrieves TMEM pointers and sets up roles
**after** sync; correction/statistics also finish later. This points to post-sync setup and
body/tail delay, not slower barrier initialization/TMEM allocation. It is not a complete
critical-cycle model.

| Suspected cause | Evidence | Change tested | Time delta / decision |
|---|---|---|---|
| Repeated mapping divides | Source/SASS and -8.57% executed instructions | Host-built fast divmod | Kept; -1.6 to -3.3% on the three small cases above |
| TMEM base round trip | FA4 allocates all 512 columns and uses constant addresses | Full allocation and constant TMEM views | -0.5% at 1x8192; noise, removed |
| Q load after the CTA sync | FA4 starts Q TMA before synchronization | Prologue-owned raw Q barrier and early TMA | +5.9% at 1x8192; removed |
| Long-hint mbarrier waits | Port/FA4 execute 3,335/1,315 `NANOSLEEP.SYNCS` instructions in the first matched report | Try-wait first / hintless retry in channel waits | +0.4% / -1.1% at 1x8192; no general win, removed |
| O-staging handoff | Port uses an mbarrier ring; FA4 a one-shot named barrier | Raw one-shot named O-done barrier | -0.8% / -1.7% at 1x8192 / 1x32768; not kept |

The wait prototypes changed all seven standard configurations' SASS. At FA4 2-CTA CLC
split-P, try-first adds 144 static instructions; hintless retry removes 40 and removes 70
suspend sites. Neither is a general improvement: two interleaved runs made this FA4 case
0.7-2.3% slower. CLC persistent-GEMM results changed direction between runs; the static pair
was faster in one run but was not repeated. No layer patch was retained. The normal
seven-config SASS gate is unchanged. The raw O-done exchange's less-than-2% gain did not
justify a manually reserved barrier ID and giving up the managed epilogue handoff.

Run the benchmark with `--output` to save versions, source hashes, kernel names and every
replay sample. It now uses three seconds of warmup and seven alternating rounds, reporting
the median of round medians. In the matched A/B above, the updated port still trails FA4
by 7.4-9.3% on 1x8192, 1x32768 and 8x65536 ragged, and by 1.7% at 4x16384; long splits
approximately tie. [Where the remaining gap was](#where-the-remaining-gap-was) locates that
delay and narrows it.

```bash
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=5s 900s \
  .venv/bin/python -m attn_gym.warpdrive.examples.bench_fa4_decode \
  --output agent_space/decode.json
```

For matched NCU source attribution, add `--profile --shapes 0` and run with
`CUTE_DSL_LINEINFO=1 ncu --profile-from-start off --cache-control none --import-source yes`.
The script compiles and warms both paths before profiling one complete call each, including
prepare/combine where present; filter the reports to `attn_kernel|warpdrive_fa4_decode`.

Original code size (no blank/comment/docstring lines): FA4 had 1,496 lines, 1,347 excluding
its interface adapter, with paging/masks/score mods/varlen Q. The initial port had 993,
including wrappers and mapping. These are not equal feature sets.

</details>

### Where the remaining gap was

<details>
<summary>Head versus correction-tail attribution and the tail reorder</summary>


Lane-0 `clock64` stamps in every warp located a roughly 970-cycle Q-store finish gap
(1x8192, 8/1, 16 splits; instrumented copies still bitwise equal):

- **Head, about 550 cycles:** port reads `seqused_k` after CTA sync, then issues Q TMA;
  FA4 used a static length and issued Q before sync.
- **Correction tail, about 450 cycles:** port waited for both final O slots before the
  row-sum/1/L stats round trip; FA4 did the round trip first.

Stamps perturbed the port by +384 ns versus FA4's +128 ns. They locate phases, not
production latency; use the uninstrumented kernel times below.

The correction now runs the round trip first, and one CTA allocates TMEM off the
barrier-init warp (see [fixed costs](#warpdrive-fixed-costs)). Decode-kernel CUPTI medians,
in microseconds, from a scratch harness that graph-replays each decode kernel alone (fixed
buffers, seed 0, 200 replays per round, nine rounds in rotating order after three seconds of
warmup, normal DVFS) and checks partials bitwise against FA4's. GB300, November 4, 2026,
CuTeDSL 4.8.0, PyTorch `2.15.0.dev20260922+cu132`, FA4 `00807a02`; "before" is the port at
`47a1183` (the allocator change), "now" at `e4faad3`, both in one process:

| batch x kv_len, heads/kv_heads | FA4 | port before | port now | now / FA4 |
|---|---:|---:|---:|---:|
| 1x8192, 8/1 | 4.224 | 4.672 | 4.447 | 1.053 |
| 1x32768, 32/1 | 7.072 | 7.936 | 7.361 | 1.041 |
| 4x16384, 32/8 | 39.87 | 40.61 | 40.23 | 1.009 |
| 8x65536 ragged, 8/1 | 26.27 | 28.32 | 28.03 | 1.067 |
| 2x131072 ragged, 16/1 | 9.184 | 9.216 | 8.928 | 0.972 |

Only columns of one run compare: on another GPU the same 1x8192 build measured 4.768 us.
Tried and not kept, with the tail order in place: a static
length without `seqused_k` (no load before the first TMA, as FA4 does) moved 1x8192 by one
CUPTI tick (-31 ns) and 1x32768 by +31 ns; computing the split range in the kernel prologue
added nothing on top. Before the allocator change, both had been worth about 2%. Try-wait-
first channel waits, patched in for every wait of the port, changed nothing.

</details>

### Per-instruction attribution of the ragged gap

<details>
<summary>Why shared KV helped one ragged case, and why it is not the default</summary>

The recorded `8x65536`, `8/1` gap mainly came from K/V stage ownership and progress, not
barrier initialization or correction arithmetic. A raw prototype using FA4's shared ring
and two skipping MMA cursors closed most of it but regressed other shapes. The later
[managed shared-KV option](#shared-kv-ring) replaces that prototype's raw synchronization;
split rings remain the default.

Matched November 4, 2026 captures used GB300, CuTeDSL 4.8.0, PyTorch
`2.15.0.dev20260922+cu132`, Triton 3.8.0, FA4 `00807a02`, and the port source SHA-256 prefix
`2a592081cfa1db52`. The committed benchmark source prefix was `949e2f06ee0a44cd` (before the
stateful-WorkMap migration; the seven standard configurations' normalized SASS stayed identical
through that migration). NCU 2025.4.1 collected `SourceCounters`, `WarpStateStats`,
`InstructionStats` and `LaunchStats`, with source import, kernel replay, no cache flushing and
no clock locking. Both paths used the same inputs and FA4 split table.

```bash
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=5s 900s \
  env CUTE_DSL_LINEINFO=1 ATTN_GYM_CUTE_CACHE_DIR=$PWD/agent_space/decode-cache \
  ncu --profile-from-start off --replay-mode kernel --cache-control none --clock-control none \
  --section SourceCounters --section WarpStateStats --section InstructionStats \
  --section LaunchStats --warp-sampling-interval 0 --warp-sampling-max-passes 10 \
  --import-source yes -k 'regex:attn_kernel|warpdrive_fa4_decode' -o agent_space/decode-source \
  .venv/bin/python -m attn_gym.warpdrive.examples.bench_fa4_decode \
  --profile --profile-reps 5 --shapes 3 --output agent_space/decode-profile.json
ncu -i agent_space/decode-source.ncu-rep --page source --print-source sass --csv
```

`--profile-reps` alternates FA4/port and port/FA4 pairs after three seconds of warmup;
`--output` records source hashes, split counts and launch order, not NCU timings. Read the
report's effective sampling interval: requesting interval 0 did not always produce 32-cycle
sampling. The ragged repeat used two aggregated passes and 128 or 256 cycles. The following
means are normalized to 256-cycle, two-pass equivalent samples, not nanoseconds. Static PCs
include out-of-line wait stubs assigned to the phase they return to. FA4 shares its softmax
body between warpgroups; the port's counts combine its two copies.

| Phase | static PCs FA4 / port | equivalent samples FA4 / port | dominant reason |
|---|---:|---:|---|
| Entry, mapping and role setup | 370 / 619 | 1830 / 1980 | memory dependencies, instruction fetch, barriers |
| KV load | 586 / 435 | 806 / 914 | long scoreboard |
| PV MMA, including its waits | 290 / 367 | 831 / 1260 | long scoreboard, barriers |
| Softmax S wait | 10 / 10 | 5344 / 5494 | long scoreboard |
| Softmax P acquire | 3 / 6 | 14 / 83 | long scoreboard |
| Correction loop | 165 / 157 | 2720 / 2862 | barriers, long scoreboard |
| Correction final O wait/load | 37 / 93 | 41 / 106 | long scoreboard |
| Epilogue wait/store | 58 / 63 | 875 / 884 | FA4 named barrier; port mbarrier |

Attribution limits:

- Source view omitted 1381/1630 equivalent samples; full accounting retained that residual.
- Source-instrumented instructions and hardware counts came from different passes.
- Neither sample totals nor parked epilogue warps establish an additive latency model or
  a new critical path.

Native `enable_iket(waits=True)` captures, joined by logical CTA, warp and channel epoch, show
the dependency behind the V stalls. All 76 kept CTAs have 16 blocks; both kernels' wait
cardinalities match. In a warmed representative capture, the port's mid-loop loader V acquires
sum to about 0.5 us per block, versus FA4's 0.16–0.19 us. The port's PV warp waits about
0.7–0.9 us on V; FA4 waits about 0.45–0.59 us at QKIssued first, then roughly 0.16 us on V.
Those waits partly relocate the same dependency, so their widths cannot be subtracted as
extra work. The port's last V-ready observation is about 0.8 us later and its correction and
O-store finishes about 0.9 us later, relative to each CTA's first warp entry.

Stage lifetimes explain the dependency. In the warmed capture, split-ring V acquisition
started before the prior PV reader was ready for 22.5% of reused stages, versus 4.3% in FA4.
FA4's V loads mostly reuse K stages already released by QK:

```text
split rings: PV -> V-stage reuse -> loader -> subsequent K/QK -> V/PV
shared ring: QK releases K capacity -> V can reuse it
             QKIssued/PVIssued rendezvous bounds cursor skew
```

The median V read-ready to EMPTY-reply interval was about 0.2 us in both. Shared capacity
and progress ordering changed the path, not the mbarrier implementation.

IKET perturbed the widths: seven alternating warmed eager captures (excluding correctness
launches) gave minimal/native-waits warp-lifetime envelopes of 28.096/28.992 us port versus
26.368/27.904 us FA4. Do not pool these with stabilized CUPTI graph times.

`plan_from_warpdrive` joins seven role ranges but promotes within-item stats/epilogue
feedback to the next work item. Its RecMII (recurrence-constrained initiation interval) is
**not a per-block model** of OneShot decode. The causal evidence here is per-warp wait
tracks joined to explicit stage epochs.

Two ablations rule out tempting alternatives. Moving the PDL wait and validity test before
setup really reduces TMEM allocations from 256 to 76 and hardware warp instructions from
2.403M to 2.117M, but changes CUPTI time by only -32 ns. Keeping the identical 76 partials
while capping the allocated splits from 32 to 16 removes 128 dropped CTAs for -97 ns. A
uniform-count control with no dropped CTAs still has a 5.1% port gap. Increasing K/V from
7/5 to 8/4 regresses the ragged case by 2.1%; blocking on V before P/O changes nothing.

The shared-ring prototype keeps the math and partials bitwise. It uses one TMA producer and
two exclusive UMMA consumers: each cursor skips the other's epochs, with FA4's two-block
K lookahead. The full/empty counts are one, not a join of both readers. A raw-done split-ring
control isolates the three one-ID OneShot stats flags needed to free named-barrier IDs.
The table is a single five-way isolated-decode comparison: nine rotating rounds of 200
fixed-buffer graph replays after three seconds of warmup, CUPTI medians in microseconds.
Shared ring depth equals the original K+V depth (12 for the 8-head tiles). No row may be
compared with a different run's table above.

| batch x kv_len, heads/kv_heads | FA4 | split default | raw-done split | shared, no KV precheck | shared, KV precheck |
|---|---:|---:|---:|---:|---:|
| 1x8192, 8/1 | 4.544 | 4.512 | 4.512 | 4.672 | 4.640 |
| 4x16384, 32/8 | 40.257 | 40.368 | 40.352 | 40.737 | 40.736 |
| 8x32768, 32/8 | 152.559 | 148.530 | 148.625 | 150.498 | 150.337 |
| 8x65536 ragged, 8/1 | 26.849 | 28.033 | 27.936 | 27.136 | 27.073 |
| 16x32768 ragged, 64/8 | 180.065 | 179.969 | 179.986 | 180.529 | 180.354 |
| 1x32768, 32/1 | 7.424 | 7.520 | 7.488 | 7.456 | 7.328 |
| 2x131072 ragged, 16/1 | 9.312 | 9.056 | 9.058 | 9.568 | 9.504 |

For the ragged case, the one-shot flags account for 0.097 us, shared capacity/rendezvous
without KV prechecks for another 0.800 us, and prechecks for 0.063 us. Together they close
81% of this run's 1.184 us gap, leaving 0.224 us unexplained. A matched source capture reduces
the four V-wait branch samples from 592 to 252 and correction O-wait branch samples from
1264 to 1054 (FA4: 1039), despite increasing hardware instructions by 2.9%. NCU duration
sometimes disagrees with the CUPTI direction; it did not select the winner.

The unchanged committed benchmark confirms the direction through the usual wrappers (FA4:
prepare/decode/combine; port: decode/combine), still timing only the decode kernel with CUPTI.
Three process pairs in A/B, B/A, A/B order on one GPU,
each with seven alternating FA4/port rounds of 200 replays, give ragged split/shared medians
27.487/26.592 us (-3.26%; FA4 26.239 us). That closes about 72% of the 1.248 us gap under this
contract. The 32-head small case improves from 7.616 to 7.360 us, while other cases regress
by up to 5.4%. All seven shapes pass the partial-bitwise assertions in every process. These
wrapper measurements and the isolated-kernel table are different contracts, not one pooled
sample set.

The example now declares those three flags as one-shot `NamedBarrier`s (correction arrives,
the load warp syncs, or the reverse for `stats_ready`), cutting its named-barrier IDs from 12
to 9 of 13. With the committed benchmark in A/B/A/B processes on one GB300 (seven rounds of
200 replays each), the port's decode kernel moved by -0.43% to +0.04% across the seven shapes
(mean of two processes per side; ragged 8x65536: 27.392 -> 27.361 us). Partials stayed
bitwise equal to FA4's in every process.

The missing layer capability was a shared TMA ring with several independent, exclusive MMA
cursors, explicit skipped epochs and a bootstrap/skew contract preventing parity aliasing.
`Slots` requires Python-int slot IDs and only advances a reader's visited slots; it does not
supply that dynamic ownership protocol. [Exclusive consumers](channels.md#exclusive-consumers)
now provide the ring and the skips; the skew contract stays the author's, as the rendezvous
recipe there shows. The original investigation retained its raw prototype, source phase
tables, matched Perfetto tracks, failed ablations and scripts in git-ignored
`agent_space/decode_sass/`; those local artifacts are not distributed with the example.

</details>

### Decode racecheck

The audited reports below are **false positives on row-sum publication**, not four distinct
races. This is specific to those access pairs/toolchain; new reports need a fresh audit.

<details>
<summary>Reported accesses, ordering proof, historical controls and reproduction test</summary>

Rechecked with source-line info on GB300,
compute-sanitizer 2025.4.1 (CUDA 13.1), driver 580.126.20, CuTeDSL 4.8.0 and torch
2.15.0.dev20260922+cu132, using the example at `0535f67`:

- Dense: `(batch, KV length, heads, KV heads, splits) = (2, 1024, 8, 1, 1)`.
- Per-sequence: `(3, 2048, 8, 1, 4)`, lengths `[1, 700, 2048]`, split counts `[1, 2, 4]`.
- Two launches of each shape. Each launch produces one RAW analysis group, so the summary
  is `4 hazards displayed (4 errors, 0 warnings)` for either build.

All four groups map to the same two source accesses in `fa4_decode_swapab.py`: correction
line 960 writes `sRowSums[warp, h]` from lane 0 of warps 12-15; `split_stats` line 610 reads
`sums[0..3, lane]` from load warp 2, lanes 0-7. The four byte hazards per float are not four
independent bugs. In this build, eight scalar stores feed one vectorized `LDS.128` per head:

| build / shape (both launches) | read PC | write PCs (all line 960) |
|---|---|---|
| split / dense | `0x4080` | `0x8d20, 8d40, 8d70, 8d90, 8db0, 8de0, 8df0, 8e00` |
| split / per-sequence | `0x43a0` | `0x9450, 9470, 94a0, 94c0, 94e0, 9510, 9520, 9530` |
| shared / dense | `0x45b0` | `0x9250, 9270, 92a0, 92c0, 92e0, 9310, 9320, 9330` |
| shared / per-sequence | `0x49d0` | `0x9a80, 9aa0, 9ad0, 9af0, 9b10, 9b40, 9b50, 9b60` |

These are kernel-relative byte offsets, not stable API identifiers. With `--racecheck-report
hazard`, the dense run shows writers 384/416/448/480 (lane 0 of each correction warp) and
readers 64-71; the addresses are the four float words per head of `sRowSums`.

The generated PTX and line-attributed SASS preserve this sequence:

```text
correction (128 threads)                 load / split_stats (32 threads)
st.shared sRowSums[warp, h]              barrier.sync   sum_done_id, 160
barrier.arrive sum_done_id, 160          ld.shared.v4.b32 sRowSums[:, lane]
```

The ID is 4 for split and 6 for shared K/V. All correction threads arrive (only the stores
are lane-0-predicated), all load threads sync before the head predicate, and the allocation
is never overwritten after these stores. The [one-shot done-flag recipe](channels.md#one-shot-done-flag)
cites the PTX memory-model guarantee: arrival publishes prior ordinary SMEM writes at barrier
completion, and sync prevents subsequent reads before completion. No extra SMEM fence or
publisher-side wait is required. Each CTA runs at most one item, so the publisher cannot lap
the waiter. This conclusion follows from the access/participation audit and emitted code,
not from numerical tests or a clean synccheck alone. Racecheck does not credit this ordering
in these kernels; its internal reason is not established.

Historical example sources compiled against the same current layer reproduce the pair:

| example source | protocol | store / read lines | four-launch racecheck summary |
|---|---|---|---|
| `fc45c88^` (`cf02a26`) | already one-shot `NamedBarrier` | 897 / 547 | 4 errors, 0 warnings |
| `4b6bda4^` (`f3ce28f`), the actual pre-port version | managed named channels | 906 / 547 | 4 errors, 0 warnings |

The latter used `with sum_done.produce()` and `sum_done.wait()` around exactly these stores
and reads; its extra empty-barrier handshake did not remove the reports. `fc45c88` added
width/schedule guards, not the NamedBarrier port (`4b6bda4`).

FA4's own swap-AB decode at `00807a02` was also checked with the same Q/K/V shapes, lengths
and requested split counts. Its per-sequence split table comes from FA4's prepare path, not
the port's fixed `[1, 2, 4]` table. It reports the same row-sum handoff: correction
`flash_fwd_decode_sm100.py:1502` writes `sRowSumWarp`, and `store_split_stats:1597-1598`
reads the four warp sums after `bar_sum_done.arrive_and_wait()`. Four scalar read PCs yield
four RAW groups per launch, for 16 errors over four launches. There are also four separate
warnings on Q-mbarrier initialization (line 834) versus arrive/expect-tx (line 836), not on
the row-sum data. No FA4 changes or blanket suppression of those warnings were made.

`test/warpdrive/test_warpdrive_decode.py::test_decode_stats_handoff` isolates one dense CTA and
uses a CPU fp32 reference, so unrelated SDPA reference kernels cannot contaminate racecheck.
Its docstring gives the reservation, timeout, `CUTE_DSL_LINEINFO=1` and sanitizer invocation
for each backend and the expected nonzero result: one analysis error for split/shared,
four errors plus one warning for FA4. All five four-launch runs also passed their CPU
reference assertions; split/shared/FA4 pass synccheck on the isolated test. Hazard counts
and grouping depend on launch count and codegen; new access pairs need a fresh audit.

</details>

### Shared KV ring

```python
out, lse = fa4_decode(q, k, v, num_splits, seqused_k=lengths, splits_per_seq=counts, shared_kv=True)
```

`shared_kv=True` chooses `SharedKVDecode`: one ring of `sum(kv_stages)` stages containing
**either K or V^T** (their buffers overlap), not two readers joining on every stage.

```text
one TMA producer -> mixed K / V^T epochs
                   QK owns K epochs; skips V
                   PV owns V epochs; skips K
```

The declaration uses `consumer_sharing=Sharing.EXCLUSIVE` with QK/PV consumer roles.
FA4's QKIssued/PVIssued `NamedBarrier`s bound cursor skew; follow the
[rendezvous recipe](channels.md#rendezvous-recipe-fa4s-qkissuedpvissued).

Both MMA warps probe the next stage after releasing, using try tokens. The loader probes
after each commit within a block's K or V group; its token cannot cross those dynamic
regions, unlike FA4's loader token. Both variants preserve bitwise partials in the recorded
comparisons. Split rings stay default: shared capacity helps some shapes and hurts others.

<details>
<summary>Shared versus split rings, try-token comparison and benchmark command</summary>


Decode-kernel CUPTI medians in microseconds of the committed benchmark
(`bench_fa4_decode.py`, FA4 prepare/decode/combine, port decode/combine; only the decode
kernel timed), `--shared-kv` against the default in three process pairs (A/B, B/A, A/B)
under one reservation, seven alternating FA4/port rounds of 200 replays after three seconds
of warmup, median of the three process medians. GB300, CuTeDSL 4.8.0, PyTorch
`2.15.0.dev20260922+cu132`, Triton 3.8.0, FA4 `00807a02`, November 5, 2026, normal DVFS;
lower is better:

| batch x kv_len, heads/kv_heads | FA4 | split (default) | shared_kv | shared vs split |
|---|---:|---:|---:|---:|
| 1x8192, 8/1 | 4.320 | 4.256 | 4.288 | +0.75% |
| 4x16384, 32/8 | 39.872 | 40.224 | 40.257 | +0.08% |
| 8x32768, 32/8 | 152.097 | 148.162 | 150.432 | +1.53% |
| 8x65536 ragged, 8/1 | 26.945 | 28.162 | 27.104 | -3.76% |
| 16x32768 ragged, 64/8 | 179.906 | 179.842 | 180.019 | +0.10% |
| 1x32768, 32/1 | 7.088 | 7.168 | 6.976 | -2.68% |
| 2x131072 ragged, 16/1 | 9.281 | 8.736 | 9.248 | +5.86% |

The ragged case gained 1.06 us, leaving 0.16 us to FA4. The raw prototype's wrapper
confirmation gained 0.895 us (27.487 -> 26.592). Each variant's three process medians agreed
within 0.16 us, but absolute levels moved across reservations: an earlier run on the same
GPU gave 27.36 split, 26.37–26.40 shared, and 26.18 us FA4. Compare columns of one run only.

Use these results as candidates to remeasure, not a selector: shared helped the 8x65536
ragged single-KV-head and 1x32768 many-query-head cases. It hurt 2x131072 ragged by 5.9% and
8x32768 full by 1.5%; "ragged" alone does not predict a win.

The try tokens are worth keeping on the shared ring, unlike on the split rings: removing them
(blocking waits throughout), interleaved in three process pairs on shapes 1x8192, 8x65536
ragged, 1x32768 and 2x131072 ragged, measured 4.352/26.432/7.040/9.248 us against
4.288/26.431/6.944/9.248 us with them.

```bash
gpu-run --timeout 1800 auto -- timeout --signal=TERM --kill-after=5s 900s \
  .venv/bin/python -m attn_gym.warpdrive.examples.bench_fa4_decode --shared-kv \
  --output agent_space/decode_shared.json
```

</details>

## Qwen3-0.6B decode megakernel

```python
from attn_gym.warpdrive.examples.qwen_megakernel import QwenMegakernel
from attn_gym.warpdrive.examples.qwen_megakernel_ref import (
    load_qwen3_weights,
    make_decode_state,
)

weights = load_qwen3_weights()  # Qwen/Qwen3-0.6B from the HF hub, or a local checkpoint path
caches, token, pos = make_decode_state(weights, ctx_len=1024, max_len=1088)  # random prompt
runner = QwenMegakernel(weights, max_len=1088)  # one compile serves every max_len
logits = runner(weights, token, pos, *caches)  # decode_step_ref's contract: writes K/V at pos
```

One persistent launch runs a whole batch-one decode step. The host lowers the step into a
table of 20.3k instructions and assigns them to CTAs ahead of time:

```text
for layer in 28 layers:
    QKV     128 items   32 rows of [Wq; Wk; Wv] x RMSNorm(h)            -> qkv
    ATTN    8 x 16      kv head x KV split: q/k norm, RoPE, flash decode  -> attention partials
    OPROJ   32 x 4      32 rows x 512-column slice of Wo (split K)        -> o partials
    GATEUP  192         16 gate + 16 up rows x RMSNorm(h + o)             -> SiLU(g) * u
    DOWN    32 x 4      32 rows x 768-column slice of Wd (split K)        -> down partials;
                        the last split also sums them into the residual
LMHEAD  594         8 x 32 rows of the tied embedding x final RMSNorm    -> logits
```

| Input | Contract |
|---|---|
| Model | Qwen3-0.6B shapes, bf16 weights, batch 1, one token per launch |
| Caches | Contiguous bf16 `(28, 8, max_len, 128)`, `max_len` a multiple of 64, `0 < pos < max_len`; only row `pos` is written |
| GPU | SM100 family; one CTA per SM, cooperative launch |
| Not supported | Batch > 1, prefill, paging, other model shapes |

- **Schedule.** `InstructionTable` is a `WorkMap` over the int32 table. `list_schedule`
  assigns rows to CTAs with a row-latency model and `assign_static` lays the table out
  `[slot][cta]` for `StaticPersistent`; `schedule="atomic"` claims rows in dependency order with
  `AtomicPersistent` instead. Each CTA's rows stay in dependency order, so with every CTA
  resident (the reason for `cooperative=True`) the earliest unfinished row can always run.
- **Roles.** Eight compute warps and one TMA load warp. The load warp streams 16 KB weight
  and KV stages through a 12-stage `Ring(Tma, Warps)` and never waits on activations, so it
  prefetches the next rows' weights while compute polls for inputs. The GEMV is packed FFMA2
  on CUDA cores; weights and KV carry an L2 evict-first hint.
- **Dependencies between CTAs** are raw code: every activation is a 64-bit word
  `(fp32 value, launch epoch)` published with one relaxed store, and consumers spin on relaxed
  (mostly 16- and 32-byte vector) loads until each word carries the current epoch (NCCL's LL protocol). The
  last CTA to finish advances the epoch. Acquire/release counters with separate waiter and
  signal warps were slower (686 vs 546 us at the time). Split-K partials are summed by their
  consumers, except down's: its last split, which finishes last because its GATEUP rows run in
  a second wave, fetches the other splits' partials for its 32 columns and publishes the
  finished residual, so QKV and the LM head fetch one word per element instead of five.
- **Numerics** follow HF's bf16 rounding boundaries with fp32 accumulation; every per-step
  intermediate has its own buffer, so a launch has no write-after-read hazards.

Median us per step (lower is better) on a GB300, 2026-11-06 (torch 2.15.0.dev20260922+cu132,
CuTeDSL 4.8.0, CUDA-graph replay). The baselines and the read roofline come from one run of
`bench_qwen_megakernel`; the roofline is the bench's logical bytes (weights once, KV read and
new-row write, logits) at its measured 6.14 TB/s read bandwidth. The megakernel columns use
the bench's `time_step` in interleaved rounds (`before` is the kernel as first landed):

| ctx | eager + graph | torch.compile + graph | megakernel before | megakernel | `splits=19` | read roofline | roofline / megakernel |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 4448 | 1365 | 474 | 453 | 495 | 213 | 47% |
| 4096 | 5326 | 1417 | 545 | 528 | 550 | 271 | 51% |
| 8192 | 6712 | 1488 | 645 | 629 | 611 | 347 | 55% (57% with 19 splits) |

`splits=19` gives 8 x 19 attention items, one per SM of the GB300. It measured faster at ctx
6208-8192 (from 6208 it cuts the most KV blocks one split reads from 7 to 6) and slower at
1024-6144. The
split count is static and the position is read on the device, so a runner can't switch it
per step; pick it for the contexts the runner decodes.

**What limits it** is latency between dependent phases, not bandwidth. A step has about 140
dependent phases, and a bare flag handoff between SMs measured 0.5 us one way with relaxed
accesses (0.75–0.9 us with acquire/release). At ctx 1024 a layer takes about 14.1 us against
about 5 us of bytes; IKET shows compute almost never waits for weights (0.05 us per stage).
A row's fetch of words that are already published costs about 0.47 us plus 15 ns per KB
(one-instruction-kind runs: 0.56 us for down's 6 KB, 1.51 us for o_proj's 68 KB), so
the poll cost is the L2 round trip plus bytes, not cache misses. A 16 KB GEMV stage costs
about 0.18 us of compute-warp time even with the data resident. Per layer at ctx 1024 the
phase spans (last producer done to last row done) are QKV 1.9, attention 3.8, o_proj 2.2,
GATEUP 3.8 and down 2.2 us; GATEUP's 192 rows on 152 SMs run a second wave, and at ctx 8192
attention grows to 9.8 us per layer. The LM head runs at bandwidth.

<details>
<summary>Run, bench and trace; how it got here</summary>

Checkpoint tests skip unless Qwen3-0.6B is already in the local HF cache; they never download it
(`hf download Qwen/Qwen3-0.6B` fetches it):

```bash
gpu-run --timeout 600 auto -- timeout --signal=TERM --kill-after=5s 600s \
  .venv/bin/python -m pytest test/warpdrive/test_warpdrive_qwen_megakernel.py \
  test/warpdrive/test_warpdrive_qwen_megakernel_ref.py
gpu-run --timeout 600 auto -- timeout --signal=TERM --kill-after=5s 2400s \
  .venv/bin/python -m attn_gym.warpdrive.examples.bench_qwen_megakernel \
  --output agent_space/qwen_megakernel.json
```

`QwenMegakernel(..., trace="globaltimer")` compiles the layer's
[globaltimer ranges](tracing.md#globaltimer): every CTA records its compute and load work items
(payload: the table row) and the compute role's `trace_region`s, also with the row as payload:
`poll` (waiting for the row's inputs; it ends when they are ready), `stream` (a GEMV tile's
weight stages), `attn.blocks` (an attention row's KV blocks) and `reduce.poll` (the last down
split fetching the other partials). Earlier regions (`prologue`, `publish`, `attn.norm`,
`attn.merge`) were dropped to keep the recording cost low: the gaps between the remaining
ranges show them. `qwen_megakernel_trace` decodes one launch:

```python
from attn_gym.warpdrive.examples import qwen_megakernel_trace as mkt

runner = QwenMegakernel(weights, max_len=1088, trace="globaltimer")
runner(weights, token, pos, *caches)  # compile and warm up
with mkt.trace_session(runner, trace_path="step.pftrace") as session:
    runner.launch(*caches)
mkt.phase_spans(session.ranges, runner.table)  # {"QKV": 1.9, "ATTN": 3.8, ...} us per layer
mkt.input_waits(session.ranges, runner.table)  # median row start -> inputs ready, us
```

The trace has one process per SM and one track per role; work items are named after their
instruction (`QKV`, `load QKV`, ...) with `layer`, `a` and `b` in their args, and an arrow runs
from each input group's last-finishing producer row to the row waiting for it, across SMs
(`write_step_trace`, the rows matched by their payloads). `bench_qwen_megakernel --trace-dir DIR` writes one trace per context and
reports both views. `trace="iket"` emits the same ranges for `run-iket` (as in
[Tracing](tracing.md#iket); the payload is each range's `payloadVal`), and `waits=True` adds the
layer's wait ranges with either backend. Disabled, both emit no instructions.

The globaltimer ranges replaced hand-written per-row stamps (one store per stamp). In one
reservation at ctx 1024 (GB300, 15 interleaved launches each), the per-layer phase spans agreed
to within 0.1 us except GATEUP (old stamps 3.84, globaltimer 3.98 us) and the LM head (46.3,
47.4 us); a launch took 472.2 us untraced, 483.1 us with the stamps and 485.7 us with
globaltimer ranges. With IKET, roles mode measured 480 us against 470–480 us unprofiled and
waits mode 508 us at the time. Traces at ctx 1024:
[IKET roles](https://fburl.com/perfetto/3h1t248o),
[IKET waits](https://fburl.com/perfetto/ak5nt8jt),
[globaltimer, one process per SM](https://fburl.com/perfetto/pqda0s42).

At ctx 1024 the step went 1050 us (atomic claiming, counters) -> 668 (host static schedule)
-> 640 (row-independent loads before waiting) -> 546 (tagged activations) -> 526 (FFMA2 GEMV)
-> 515 (three-warp q/k norm) -> 490 (L2 evict-first) -> 470–480 (12 stages, 32-byte polls)
-> 460 (GEMV k-loop unrolled: the rolled loop kept a 32-byte stack frame) -> 453 (the last
down split publishes the residual). An L2 prefetch of the LM head, poll backoff,
probe-then-fetch, and 32 KV splits were no faster and were dropped.

Ideas borrowed from Helion's megakernels (one layer per launch, many one-warp programs,
acquire/release counters) and other latency ideas, each tried alone against the then-current
kernel. Deltas are us per step at ctx 1024 / 4096 / 8192 from interleaved rounds with an
identical-code control row (identical builds differed by up to 3.6 us at ctx 1024 and up to
9 us in noisier runs):

| idea | result | delta | why |
|---|---|---:|---|
| GEMV k-loop `unroll_full` | kept | -12 / -13 / -20 | no stack frame; a 4-stage GEMV 0.86 -> 0.73 us |
| last down split publishes the residual (Helion's last-arrival reduce, counter-free) | kept | -11 / -5 / -6 | QKV span 2.6 -> 1.9 us; down span 1.9 -> 2.2 us |
| `splits=19` | explicit option | +42 / +22 / -17; -10 at 6208 | fills the SMs; not a default |
| RMSNorm scale after the projection, `(W(x g)) r` | rejected | +4 / 0 / 0 | residual chunks become ready within 0.04-0.2 us of each other, so there is nothing to overlap; logit error vs fp32 improved (geometric-mean rms 0.89x the HF-bf16 reference's, against 1.02x before) |
| release by input chunk | already so | | o_proj waits on its head pair, down on its GATEUP group; GATEUP has no chunk stagger |
| last-arrival reduce for o_proj: designated split / acq_rel counter | rejected | +37..47 / +35..46 | no o_proj split reliably finishes last; the counter round trip plus reading the others costs more than GATEUP saves |
| GATEUP in one wave: 2-tile items / 16-row half tiles | rejected | +16 / +3 / -12; +18 / +17 / +16 | the second wave already shares the prologue; doubled items make every down split late; half-tile stages cost 0.31 us |
| L2 hints: evict_last tagged words, 256 B prefetch size or L1::no_allocate on polls, load-warp prefetch.global.L2 of next inputs, KV evict_normal | rejected | within noise to +26 | fetch times don't move: polls pay the L2 round trip, not misses; KV is read once per step |
| o_proj combine with 32-byte loads | rejected | +77 | spilled (24-byte stack) |
| attention partials packed two per word (24-bit floats, 16-bit tag) | rejected | -3..-6 / -4 / mixed | o_proj fetch 1.51 -> 1.29 us; about 1%, within the control spread |
| q/k norm replicated per warp; 19 balanced splits; fewer splits at short ctx | rejected | +5..+70 | spills, or fewer rows streaming |

Attention splits beyond the context running first (Helion) has no counterpart: the splits
here cover only `[0, pos)`, and the load warp already streams KV before q is ready.

**Sharing a GPU.** The static schedule needs every CTA resident; the cooperative launch
guarantees it, and a CUDA-graph capture keeps the attribute (the kernel node reports
`cudaLaunchAttributeCooperative = 1`). Time-sliced with another process looping large
matmuls, 4000 eager launches and 4000 graph replays ran 2.1-2.2x slower with bitwise-equal
logits and no stall. Steps did run 4x slower, or stalled for minutes until a timeout, while
another user's job shared the GPU; a capture of that state is still missing. `schedule="atomic"`
does not need co-residency: rows are claimed in dependency order by running CTAs, so the
earliest unfinished claimed row always has its producers done.

</details>

### Layer gaps found

Writing this kernel exposed these warpdrive gaps, most useful fix first:

1. **No channel between CTAs.** Both the counter and tagged-word versions are raw code; a
   tagged-slot or global-flag channel kind (any CTA produces, consumers spin on an epoch) is
   the missing primitive.
   `cute.arch.store(buf.iterator + i, pack_tagged(value, epoch), sem="relaxed", scope="gpu")`
2. **A `Ring(Tma, Warps)` read with generic loads needed an explicit proxy fence** (fixed).
   Without it TMA can refill a stage still being read (PTX 9.5 §8.9.5, §8.10.6); every test
   passed without it. The layer now derives `release_fence="smem_async"` for Tma-filled
   channels released by `Threads`/`Warps` (`release_fence=()` opts out); see channels.md.
   `Ring(stages, Tma(STAGE_BYTES), Warps, buffer=...)`
3. **`Tma(tx_bytes)` fixes a ring's bytes per stage**, so every instruction loads exactly 16 KB
   per stage; wider GATEUP items (which would avoid its second wave) need a per-handle
   transaction count. `Ring(12, Tma(32 * 256 * 2), Warps, buffer={"W": w, "KV": kv})`
4. **Argument tensors materialized lazily inside a role** fail with "operand does not
   dominate this use" at an unrelated line; so do dicts of tensors in params and Python loops
   in the kernel prologue. Materialize before `run_roles` and keep params flat.
   `tensors = [cute.make_tensor(t.iterator, t.layout) for t in tensors]`
5. **The launch config does not guarantee co-residency.** A static schedule over spinning CTAs needs
   every CTA resident; `launch_args()`'s SM-count cap is only a capacity estimate.
   `.launch(**self.launch_config(bound), stream=stream, cooperative=True)`
6. **SMEM over-subscription fails only at launch**: `stages=13` compiles, then the launch fails
   with `cudaErrorInvalidValue`; `describe()` cannot report bytes before compiling.
   `QwenDecodeMegakernel(stages=13)`
7. **Bit-level helpers need private MLIR imports.** fp32/int32 bitcasts, i64 packing and vector
   relaxed loads have no `cute.arch` wrapper (nor do nanosleep and `.L2::cache_hint` loads and
   stores, which needed inline PTX to try evict_last on tagged words). Each strong load is one
   L2 request per lane, so vector loads matter for polling.
   `from cutlass._mlir_helpers.arith import bitcast`
8. **Trace ranges are named after the role, not the instruction** (partly fixed). A range's
   payload now carries the table row (`trace_region(..., payload=row)`; globaltimer work items
   carry their index), and the globaltimer backend records once per role rather than once per
   warp; matching rows to instructions is still host work, and `trace_region` is only a
   context manager.
   `with self.trace_region("poll", payload=row): fetch(segments, vals, epoch)`
9. **`plan_from_warpdrive` models only the CTA-local ring**: analyzed against the IKET
   capture, it reports compute as critical and the load warp as blocked on reuse, but the
   real critical path crosses CTAs and is invisible to the plan. `plan_from_warpdrive(op)  # one ring (w), two roles; no cross-CTA edges`
10. **Register budgets don't help a 9-warp CTA.** Three warps share one SM sub-partition, so
    ptxas caps every thread at 168 registers. `RoleConfig` budgets (compute 208 or 224, load
    warpgroup 88 or 56) left the kernel at 168 with a 16-32 byte stack instead of none, and a
    small change that pushes the compute role past 168 spills in the hot loops (+50 to +77 us
    per step). Check `cuobjdump --dump-resource-usage` before timing.
    `roles={"compute": RoleConfig(range(8), regs=224), "load": RoleConfig(8, regs=56)}`
11. **Two API traps.** A one-entry buffer dict makes `h.data` the bare tensor, and a rank-1
    SMEM layout with an int shape crashes allocation.
    `h.view("W")  # not h.data["W"]`; `cute.make_layout((3072,))  # not make_layout(3072)`

## warpdrive fixed costs

<details>
<summary>Handoff twins and decode-layout setup costs (historical GB300 measurements)</summary>


[bench_overhead.py](https://github.com/meta-pytorch/attention-gym/blob/main/attn_gym/warpdrive/examples/bench_overhead.py) times the layer against hand-written
`cute.arch` twins (GB300, same versions and date as above, seven interleaved rounds):

```bash
gpu-run --timeout 600 auto -- timeout --signal=TERM --kill-after=5s 900s \
  .venv/bin/python -m attn_gym.warpdrive.examples.bench_overhead \
  --output agent_space/overhead.json
```

Per one-way handoff, **SM cycles, lower is better**:

| Mechanism | warpdrive | Raw `cute.arch.mbarrier_wait` | Raw hintless `try_wait` first |
|---|---:|---:|---:|
| Threads | 170 | 166 | 107 |
| Named | 53 | 53 | 53 |
| UMMA commit | 164 | 164 | 105 |
| TMA | 317 | 317 | 257 |
| Round robin | 170 | 172 | 119 |

Streams through 1-/4-stage rings also matched. These twins found no material channel
cost; they do not establish zero overhead for arbitrary kernels. Try-wait-first saved
about 60 cycles with an already parked waiter (`mbarrier_wait` lowers to `TRYWAIT`, then
`NANOSLEEP.SYNCS` with a 10 ms hint), but did not help decode/GEMM/FA4; see
[decode setup costs](#decode-setup-costs).

The fixed-cost suite runs the decode example's 16-warp layout, 48 mbarriers, TMEM and a
OneShot stream with empty roles. On 148 CTAs, before the allocator change, it took 2.14 us against 1.95 us for a twin
that allocates TMEM from warp 0 and 1.82 us for one that allocates from warp 1, as FA4 does.
Teardown (role end to kernel end) costs about 70 cycles in both. The CTA sync to the first
role instruction took about 600 cycles longer than the warp-1 twin, for three reasons:

1. Warp 0 initialized every barrier and then allocated TMEM. One CTA now allocates from the
   lowest other warp of a TMEM role: role start 1260 -> 925 cycles, 1.89-1.92 us.
2. LLVM folds the role dispatch's compares on the warp index into a switch, which ptxas
   lowers to an indexed constant load and an indirect branch (`BRX`), here worth about 116
   cycles. It appears with three or more consecutive single-warp roles, as in the decode,
   not in the other examples. Opaque per-test copies of the warp index removed it but made
   the decode 2% and single-CTA GEMM/FA4 about 1% slower; not kept.
3. Each role decodes its first item behind an `index < total` guard after the sync, so the
   guard's constant load and then the mapping's divisor loads are serialized there; the
   twin's unguarded decode is hoisted above the barrier. A mapping that sets
   `decodes_any_index` drops the guard (role start about 925 -> 858 cycles). Decoding the first
   item once in `begin_roles` instead lengthened warp 0's path to the sync and cost 80 ns.

</details>

## Historical timings

The [original README at 49b9713](https://github.com/drisspg/transformer_nuggets/blob/49b9713/transformer_nuggets/cute/warp_roles/README.md)
records the November 2026 GB300 GEMM/FA4 experiments. Those were BF16 CUDA-graph replay medians
with CuTeDSL 4.8.0, a PyTorch 2.15 nightly cu132 build and FA4 at `849b5daa`.

They are measurements of that revision, not of the storage-binding or work-API changes.
Keep the original shapes, schedule and reference implementation with the numbers when comparing;
a FA4 default schedule and a matched single-CTA schedule answer different questions.

## Still missing

Non-bulk `cp.async` producer mechanisms are not supported (`cp.async.bulk` loads can use
`Tma`; see [TMA checks](tracing.md#tma-checks)). Both decode variants remain OneShot-only. Using CUTLASS pipeline objects where their semantics match remains a possible
implementation cleanup, not a reason to change the role bodies.

The alias diagnostic still needs IR-inferred accesses, partial K-block footprints, cluster
analysis and completion-aware epoch checks. Its current limits are in
[storage](storage.md#read-the-result).

For the normal and quick-fuzz commands, see [Tests](index.md#tests). Long campaigns and
what their oracles cover are in the [fuzz harness guide](https://github.com/meta-pytorch/attention-gym/blob/main/test/warpdrive_fuzz/README.md).
