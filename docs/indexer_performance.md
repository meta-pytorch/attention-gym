# Lightning indexer measurements

These measurements are for the indexer, not attention or model latency. The current
warpdrive timings and historical sweep below used a GB300 (SM103) and CUDA-event timing
nodes inside graph capture. For the API contract, see [sparse.md](sparse.md).

## Warpdrive selectors on GB300

Warm forward latency in microseconds, including BF16 scoring and selection: B=1, H=64,
D=128, causal, identical inputs, median of three interleaved rounds of 20 graph replays.
PyTorch 2.16.0.dev20261125+cu134, CuTeDSL 4.8.1. Use the benchmark command below with
the sequence length, compression ratio and top-k from each row.

| Query tokens | Compression ratio | Top-K | Radix (µs) | GVR2 (µs) |
|---:|---:|---:|---:|---:|
| 2,048 | 1 | 64 | 69.90 | 64.61 |
| 8,192 | 4 | 512 | 271.30 | 259.23 |
| 16,384 | 1 | 512 | 2,222.90 | 1,968.38 |
| 8,192 | 1 | 2,048 | 605.18 | 675.84 |

## Warpdrive MXFP8 on GB300

For the same causal shapes and timing protocol, the table below compares complete CuTe
forward calls with `selector="auto"`. BF16 receives the exact dequantization of the MXFP8
Q/K (seed 77, nonuniform scales); weights are BF16 in both cases. Quantization and
compilation are outside the timed graph.

| Query tokens | Compression ratio | Top-K | BF16 (µs) | MXFP8 (µs) |
|---:|---:|---:|---:|---:|
| 2,048 | 1 | 64 | 64.61 | 60.90 |
| 8,192 | 4 | 512 | 259.62 | 231.15 |
| 16,384 | 1 | 512 | 1,952.67 | 1,618.66 |
| 8,192 | 1 | 2,048 | 595.57 | 504.82 |

The native MXFP8 scorer also produced bit-identical FP32 score slabs to `aaa1f9b9` on
four matched workloads (H=32/64, T=4K–128K). This checks the change in execution on
warpdrive; it does not imply that quantizing arbitrary BF16 inputs preserves their Top-K.

**Empty-tile lifetime check.** On SM103, `drisspg/CUTracer` at `911dba6` traced H=32
and H=64 with `T=9`, `D=128`, and causal compression ratio 4. The first query pair has
no candidate tiles; the second pair has one. With the unconditional Q-scale copy,
`races-run` reported the copy racing TMEM deallocation on both zero-tile launches and
no races on either one-tile launch. With the copy skipped for empty intervals, all four
traces were race-free with full TMEM coverage. Both versions passed the numerical
checks, including masked-score sentinels; output comparison alone did not detect the race.

## `auto` threshold sweep

On SM100/SM103 for dense nondeterministic CuTe calls, `auto` resolves to
`gvr2 if topk < 2048 else default`. The pre-warpdrive sweep used CuTe, BF16, B=1,
H=64, D=128, random normal inputs, three interleaved rounds of 20 replays with identical
inputs. Ratios are radix time / GVR2 time, so above 1 means GVR2 is faster.

| | K ≤ 1024 | K ≥ 2048 |
|---|---|---|
| Selection kernel alone | 1.1–1.4× (S=2–4K) → 1.6–2.1× (8–16K) → 2.6–4.3× (32K) → 4.7–9× (64K) | 0.55–1.0× |
| End to end, `compress_ratio=4` | 1.04–1.10× | 0.94–0.96× |
| End to end, `compress_ratio=1` | 1.07× (8K) → 1.16× (32K) → 1.26–1.37× (64K) | 0.90–1.02× |

The boundary is K, not S. At `topk >= 2048` GVR2 takes its in-kernel radix fallback
(`NATIVE_TOPK_LIMIT` in `cute_topk_gvr2.py`) and can only tie or lose, so `auto` launches
radix directly. Both pre-port CuTe selectors passed compute-sanitizer and CUTracer checks on SM103.

On the warpdrive port, `checked=True` builds on GB300 matched the Top-K oracle across empty
and short rows, a ragged tail, both native GVR2 CTA widths and the K=2048 fallback. In
focused CTA traces with `drisspg/CUTracer` at `911dba6`, radix had full race-check coverage.
Native GVR2 had no reported race, but the checker could not bound the effect of its L2
prefetches (`CCTL.E.PF2`); shared-memory race coverage for that path is partial.
Heatmaps and raw samples are attached to
[PR #559](https://github.com/meta-pytorch/attention-gym/pull/559).

Reproduce a cell:

```bash
uv run benchmarks/sparse/indexer_benchmark.py --batch 1 --heads 64 --head-dim 128 --sequence-length 65536 --compress-ratio 4 --topk 512 --dtype bfloat16 --backend cute --selector default gvr2 --warmup 5 --rep 30
```

## Historical external comparisons: setup

The pre-warpdrive DeepGEMM and FlashInfer comparisons below share this setup:

- PyTorch `2.15.0.dev20260911+cu132`, CuTeDSL `4.7.1`.
- DeepSeek V4-shaped workloads: `B=1, H=64, D=128`, causal compression ratio 4,
  `T=65536/131072`, `S=T/4`, Flash `K=512`, Pro `K=1024`.
- Seed 77, `make_indexer_mxfp8_test_inputs`: E4M3 data, nonuniform E8M0 block-32
  scales, signed BF16 weights. BF16 receives exact dequantizations of the same
  values. These are synthetic inputs, not model-quality evaluations.
- **External CUDA timing-event nodes are inside graph capture**, around the
  measured work. CPU submission and synchronization are outside the interval;
  GPU graph scheduling and every captured operation remain included. Five eager
  and five replay warmups, three interleaved rounds with the middle order reversed,
  30 synchronized samples per round. Tables use the median of three round medians.
- Quantization, allocation, compilation and capture are excluded. All in-kernel
  scale loads/packing, score writes and index writes are included. No buffer rotation
  or clock-policy changes. Recorded clocks were **1260 MHz SM / 3996 MHz memory**.
- Useful causal dot work is `2048*T*(T-2)` FLOPs, excluding selection/reduction,
  projections, compression, attention and backward.
- Both kernels passed compute-sanitizer memcheck/racecheck/synccheck and CUTracer
  delay/deadlock checks in these runs.

Raw samples, source hashes, routes, numerical checks and per-round telemetry are attached
to [PR #560](https://github.com/meta-pytorch/attention-gym/pull/560). Each comparison uses
the same inputs, process and device. The MXFP8 and GVR2 experiments use distinct timing
boundaries described below; do not subtract their measurements to infer individual kernel
costs. The comparator harnesses are not in this repository; the CLI below reproduces the
Attention Gym variants only.

## Historical MXFP8 scorer versus DeepGEMM

The comparator is DeepGEMM
[`78b6900`](https://github.com/deepseek-ai/DeepGEMM/tree/78b69000794d0937b47ae3387eff7663410264d1),
using its **actual native MXFP8** path, not its per-vector mode. Four E8M0 bytes
are viewed as each INT32 scale word without allocation. No block scale is folded
into weights. DeepGEMM receives normalized FP32 weights because its FP32 logits
require that dtype; local normalization follows reduction. Their FP32 rounding
orders differ, and both are checked against independent FP64 scoring.

Both local columns and the DeepGEMM comparison use the CuTe radix selector
(`selector="default"`). **Milliseconds, lower is better:**

| T | Boundary | Attention Gym MXFP8 | DeepGEMM MXFP8 |
|---:|---|---:|---:|
| 65536 | Scoring | 8.462 | 7.002 |
| 65536 | K512 complete | 11.785 | 10.400 |
| 65536 | K1024 complete | 11.957 | 10.569 |
| 131072 | Scoring | 31.114 | 27.118 |
| 131072 | K512 complete | 44.098 | 39.863 |
| 131072 | K1024 complete | 45.147 | 40.877 |

Attention Gym is **15–21% slower for scoring and 10–13% slower complete** than DeepGEMM.

### Design and remaining differences

- Two query-specific epilogue warpgroups cache weights independently. Each holds
  one query's logits rather than both queries' working sets.
- Independent Q/K loaders, four K shared-memory stages and three accumulator slots
  overlap loading, native block-scaled MMA and FP32 reduction. Each accumulator
  owns a distinct K-scale TMEM slot; immutable Q scales are shared across tiles.
- Both epilogue groups release their slot after TMEM load completion, **before**
  reduction. Packed FP32 FMA pairs preserve the original four reduction chains.
- Generated code uses **141 registers/thread** with no stack or LDL/STL.
  Register-count hints (`maxnreg`) caused spills here and are intentionally absent.

DeepGEMM still differs in candidate chunk size, persistent scheduling, TMA scale
staging and resource allocation; the residual bottleneck has not been isolated.
Attention Gym accepts arbitrary scale strides, so it cannot assume DeepGEMM's packed
TMA scale layout.

### Controlled safe-ReLU comparison

DeepGEMM's `(x + abs(x))/2` ReLU overflows for finite dots near `2**127`; Attention Gym
uses `max(x, 0)`. Changing only that formulation in DeepGEMM costs it 6.8–6.9%, which
accounts for 34–47% of the local gap. **Scoring milliseconds, lower is better;** compare
within this table only:

| T | Original DeepGEMM | Safe-ReLU DeepGEMM | Attention Gym |
|---:|---:|---:|---:|
| 65536 | 7.075 | 7.554 | 8.480 |
| 131072 | 27.109 | 28.983 | 31.106 |

## Historical GVR2 selector versus FlashInfer

The comparator is FlashInfer
[`c11c109`](https://github.com/flashinfer-ai/flashinfer/tree/c11c1090172f578bad37b8bca2b40e4161d72144),
`top_k_varlen(backend="gvr_2")`, `pre_idx=None`. Every measured upstream shape
selects its **main family**, not its register or cluster kernels.

These measurements use the production score and selection kernels in the complete
slab loop, with **shared preallocated score/output buffers** for all selectors. Every
measured configuration was checked for exact selected-score multisets against its
FP32 slab.

### Complete score + selection

**Milliseconds, lower is better:**

| T | K | Precision | Radix (`default`) | Local GVR2 | FlashInfer GVR2 |
|---:|---:|---|---:|---:|---:|
| 65536 | 512 | BF16 | 13.218 | 11.912 | 11.855 |
| 65536 | 512 | MXFP8 | 11.844 | 10.555 | 10.504 |
| 65536 | 1024 | BF16 | 13.399 | 12.253 | 12.153 |
| 65536 | 1024 | MXFP8 | 12.006 | 10.909 | 10.782 |
| 131072 | 512 | BF16 | 49.287 | 51.259 | 52.265 |
| 131072 | 512 | MXFP8 | 44.359 | 39.394 | 39.392 |
| 131072 | 1024 | BF16 | 50.267 | 49.382 | 52.424 |
| 131072 | 1024 | MXFP8 | 45.373 | 38.965 | 39.280 |

GVR2 reduces complete MXFP8 latency **9–14% versus radix** in these cases. At 64K,
local complete latency is within about 1.2% of FlashInfer. At 128K, the BF16 local
GVR2 round medians vary **2.8%/5.6%** (K512/K1024, max-minus-min divided by median);
the K1024 rounds straddle the radix result. The BF16 128K differences are within
round-to-round variation. The broader K/S sweep behind the `auto` rule is in the
[`auto` threshold sweep](#auto-threshold-sweep) above.

### Late-slab selector only

**Microseconds, lower is better.** T64K uses 512 rows/N16384; T128K uses 256 rows/N32768.

| T | K | Precision | Radix (`default`) | Local GVR2 | FlashInfer GVR2 |
|---:|---:|---|---:|---:|---:|
| 65536 | 512 | BF16 | 42.752 | 25.632 | 23.984 |
| 65536 | 512 | MXFP8 | 43.136 | 25.376 | 24.000 |
| 65536 | 1024 | BF16 | 46.640 | 29.104 | 26.560 |
| 65536 | 1024 | MXFP8 | 46.448 | 28.960 | 26.544 |
| 131072 | 512 | BF16 | 44.912 | 23.744 | 23.808 |
| 131072 | 512 | MXFP8 | 45.024 | 23.456 | 23.824 |
| 131072 | 1024 | BF16 | 49.824 | 24.768 | 25.552 |
| 131072 | 1024 | MXFP8 | 49.408 | 24.848 | 25.264 |

The selector is **6–10% slower than FlashInfer at 64K**, with point estimates
within 3.1% at 128K. Late-slab time cannot be multiplied by slab count to predict
the complete causal pipeline, whose early rows follow different paths.

## Memory

Resident Q/K/scales/weights occupy 1,086,324,736 bytes (BF16) versus 564,199,424 bytes
(MXFP8) at 64K, doubled at 128K: about 48.1% input-storage savings. Score scratch remains
at most 32 MiB. Returned indices occupy 128/256/512 MiB depending on T/K. DeepGEMM uses an
additional 32 MiB score buffer in the comparison.

## Reproducing individual variants

The public benchmark records timing events inside graph capture:

```bash
uv run benchmarks/sparse/indexer_benchmark.py --batch 1 --heads 64 --head-dim 128 --sequence-length 65536 --compress-ratio 4 --topk 512 --dtype mxfp8 --backend cute triton --selector default gvr2 --warmup 5 --rep 30
```

Reserve an otherwise unused GPU. For matched-precision comparisons, explicitly
dequantize the generated MXFP8 inputs for BF16; separate CLI calls with different
dtypes do not guarantee identical values. Match capture/timing boundaries and
interleave repeated rounds before interpreting small performance differences.

## Shape-polymorphic Triton MXFP8 scorer

The Triton MXFP8 scorer and streaming selector now reuse their binaries across sequence
lengths, including transposed inputs. On three contiguous cases, the dynamic kernels
matched the original shape-specialized kernels (`aaa1f9b9`) within 1% in warm graph replay
and returned bit-identical indices. These measurements do not compare Triton to CuTe.
For the explicit Triton GVR2 route with BF16 on the same three shapes, making row lengths
dynamic cost 1–15% against the shape-specialized `077fa395` (original → current: 31.7 →
36.2, 121.9 → 140.2, and 209.9 → 212.2 µs). The tradeoff is one compile across changing
lengths instead of one per length; Triton does not auto-select GVR2 for BF16.

| Batch | Query tokens | Heads | Compression ratio | Top-K | Original (µs) | Dynamic (µs) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 512 | 32 | 1 | 32 | 36.45 | 36.29 |
| 2 | 2048 | 64 | 4 | 128 | 128.51 | 128.22 |
| 1 | 4097 | 64 | 4 | 256 | 259.26 | 257.41 |

Time includes all scoring and streaming selection kernels, with causal masking and `D=128`.
Both versions used the same fixed pointers (`make_indexer_mxfp8_test_inputs`, seed 77,
signed BF16 weights). Times are the median of five alternating-order round medians
(25 timed replays per round, five warmups before and after capture). GB300, default clock
policy, observed 2070 MHz SM / 3996 MHz memory; PyTorch `2.16.0.dev20261125+cu134`, Triton
`3.9.0`. Compilation, capture, input preparation and host submission are excluded.
