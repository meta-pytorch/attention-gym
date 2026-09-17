# MXFP8 and GVR2 indexer measurements

These measure **indexer scoring plus selection** on the CuTe kernels, not attention or
model latency. Triton routes were not swept.

## Measurement contract

- NVIDIA GB300 (SM103), PyTorch `2.15.0.dev20260911+cu132`, CuTeDSL `4.7.1`.
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

Raw samples, source hashes, routes, numerical checks and per-round telemetry are attached
to [PR #560](https://github.com/meta-pytorch/attention-gym/pull/560). Each comparison uses
the same inputs, process and device. The MXFP8 and GVR2 experiments use distinct timing
boundaries described below; do not subtract their measurements to infer individual kernel
costs. The upstream comparator harnesses are not shipped; the public CLI below reproduces
individual variants only.

## MXFP8 scorer versus DeepGEMM

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
Attention Gym retains arbitrary logical scale strides rather than narrowing the
interface to force a particular TMA layout.

### Controlled safe-ReLU comparison

DeepGEMM's `(x + abs(x))/2` ReLU overflows for finite dots near `2**127`; Attention Gym
uses `max(x, 0)`. Changing only that formulation in DeepGEMM costs it 6.8–6.9%, which
accounts for 34–47% of the local gap. **Scoring milliseconds, lower is better;** compare
within this table only:

| T | Original DeepGEMM | Safe-ReLU DeepGEMM | Attention Gym |
|---:|---:|---:|---:|
| 65536 | 7.075 | 7.554 | 8.480 |
| 131072 | 27.109 | 28.983 | 31.106 |

## GVR2 selector versus FlashInfer

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

| T | K | Precision | Default radix | Local GVR2 | FlashInfer GVR2 |
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
the K1024 rounds straddle the radix result. Do not claim a consistent BF16 128K win,
or infer universal parity from close point estimates. The broader K/S sweep behind the
`auto` rule is in [indexer_gvr2_performance.md](indexer_gvr2_performance.md).

### Late-slab selector only

**Microseconds, lower is better.** T64K uses 512 rows/N16384; T128K uses 256 rows/N32768.

| T | K | Precision | Default radix | Local GVR2 | FlashInfer GVR2 |
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

## Memory and validation

Resident Q/K/scales/weights occupy 1,086,324,736 bytes (BF16) versus 564,199,424 bytes
(MXFP8) at 64K, doubled at 128K: about 48.1% input-storage savings. Score scratch remains
at most 32 MiB. Returned indices occupy 128/256/512 MiB depending on T/K. DeepGEMM uses an
additional 32 MiB score buffer in the comparison.

Hardware runs were on SM103; SM100 was compile-checked only. Both kernels passed
compute-sanitizer memcheck/racecheck/synccheck and CUTracer delay/deadlock checks.

## Reproducing individual variants

The public benchmark records timing events inside graph capture:

```bash
uv run benchmarks/sparse/indexer_benchmark.py --batch 1 --heads 64 --head-dim 128 --sequence-length 65536 --compress-ratio 4 --topk 512 --dtype mxfp8 --backend cute triton --selector default gvr2 --warmup 5 --rep 30
```

Reserve an otherwise unused GPU. For matched-precision comparisons, explicitly
dequantize the generated MXFP8 inputs for BF16; separate CLI calls with different
dtypes do not guarantee identical values. Match capture/timing boundaries and
interleave repeated rounds before interpreting small performance differences.
