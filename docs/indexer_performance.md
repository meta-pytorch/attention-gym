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

On four matched workloads (H=32/64, T=4K–128K), the native MXFP8 scorer produced
bit-identical FP32 score slabs to the pre-warpdrive MXFP8 scorer. This does not imply that
quantizing arbitrary BF16 inputs preserves their Top-K.

**Empty-tile lifetime check.** On SM103, `drisspg/CUTracer` at `911dba6` traced H=32
and H=64 with `T=9`, `D=128`, and causal compression ratio 4. The first query pair has
no candidate tiles; the second pair has one. With the unconditional Q-scale copy,
`races-run` reported the copy racing TMEM deallocation on both zero-tile launches and
no races on either one-tile launch. With the copy skipped for empty intervals, all four
traces were race-free with full TMEM coverage. Both versions passed the numerical
checks, including masked-score sentinels; output comparison alone did not detect the race.

Time one MXFP8 configuration (the CLI quantizes inputs before timing):

```bash
uv run benchmarks/sparse/indexer_benchmark.py --batch 1 --heads 64 --sequence-length 8192 --compress-ratio 4 --topk 512 --dtype mxfp8 --backend cute --selector auto --warmup 4 --rep 20
```

For the comparison with BF16, use the same dequantized values in both runs; two independent
CLI invocations do not guarantee that. The [archived MXFP8 benchmark data](https://gist.github.com/drisspg/ac430e5772b0a48d6e7ff789ef98a611)
cover the earlier pre-warpdrive implementation.

## `auto` threshold sweep

In the original pre-warpdrive sweep on CuTe, `auto` resolves to `gvr2 if topk < 2048 else default`. Sweep: CuTe, BF16, B=1,
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
