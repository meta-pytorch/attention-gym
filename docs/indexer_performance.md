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
