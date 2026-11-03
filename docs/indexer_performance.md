# Lightning indexer measurements

These measure **indexer scoring plus selection** on the CuTe kernels, not attention or
model latency. Triton routes were not swept. All runs are on an NVIDIA GB300 (SM103) with
CUDA-event timing nodes inside graph capture; SM100 was compile-checked only. The API
contract is in [sparse.md](sparse.md).

## `auto` threshold sweep

On CuTe, `auto` resolves to `gvr2 if topk < 2048 else default`. Sweep: CuTe, BF16, B=1,
H=64, D=128, random normal inputs, three interleaved rounds of 20 replays with identical
inputs. Ratios are radix time / GVR2 time, so above 1 means GVR2 is faster.

| | K ≤ 1024 | K ≥ 2048 |
|---|---|---|
| Selection kernel alone | 1.1–1.4× (S=2–4K) → 1.6–2.1× (8–16K) → 2.6–4.3× (32K) → 4.7–9× (64K) | 0.55–1.0× |
| End to end, `compress_ratio=4` | 1.04–1.10× | 0.94–0.96× |
| End to end, `compress_ratio=1` | 1.07× (8K) → 1.16× (32K) → 1.26–1.37× (64K) | 0.90–1.02× |

The boundary is K, not S. At `topk >= 2048` GVR2 takes its in-kernel radix fallback
(`NATIVE_TOPK_LIMIT` in `cute_topk_gvr2.py`) and can only tie or lose, so `auto` launches
radix directly. Both CuTe selectors passed compute-sanitizer and CUTracer checks on SM103.
Heatmaps and raw samples are attached to
[PR #559](https://github.com/meta-pytorch/attention-gym/pull/559).

Reproduce a cell:

```bash
uv run benchmarks/sparse/indexer_benchmark.py --batch 1 --heads 64 --head-dim 128 --sequence-length 65536 --compress-ratio 4 --topk 512 --dtype bfloat16 --backend cute --selector default gvr2 --warmup 5 --rep 30
```
