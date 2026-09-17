# Sparse attention

## Lightning indexer

`lightning_indexer` selects a shared pool of key positions using multi-head weighted
ReLU scores. It returns indices, not attention outputs or differentiable scores:

```python
from attn_gym.sparse import lightning_indexer

indices = lightning_indexer(q, k, weights, topk=128, causal=True)
# q: [B, T, H, D], k: [B, T, D], weights: [B, T, H]
# indices: [B, T, 128], int32

# DeepSeek compressed sparse attention: one candidate per 4 consecutive tokens.
indices = lightning_indexer(q, k_compressed, weights, topk=128, causal=True, compress_ratio=4)
# k_compressed: [B, T // 4, D]; query t may select candidates s < (t + 1) // 4
```

The score for a query/candidate pair is
`sum_h(weights[h] * relu(dot(q[h], k))) / sqrt(H * D)`.
Weights may be negative. `k` holds `S = T // compress_ratio` candidates, each summarizing
`compress_ratio` consecutive tokens (a trailing partial window forms no candidate). With
causal selection, query `t` considers only candidates `s < (t + 1) // compress_ratio`, those
whose tokens all lie at positions `<= t` (candidates `0..t` with the default ratio of 1);
rows with fewer than `topk` candidates contain `-1` padding. `compress_ratio != 1` requires
`causal=True`. Mask padding before gathering: a raw PyTorch index of `-1` selects the last
position rather than an invalid position. `topk=0` returns an empty last dimension.
Output order and tie-breaking are unspecified, including between repeated calls.

### Implementations and device dispatch

- `impl="reference"` evaluates PyTorch scoring and Top-K on CPU or CUDA. FP16/BF16/FP32
  inputs accumulate in FP32; FP64 inputs retain FP64. It materializes the score intermediates
  and is intended for correctness checks and small inputs.
- `impl="fused"` (the default) selects **CuTe on SM100/SM103**, or **Triton on other NVIDIA GPUs
  with compute capability 9.0 or newer**, including Hopper. Both optimized implementations
  keep their selection state on chip. CuTe separates score generation from Top-K selection and
  reuses a per-call FP32 score slab capped at **32 MiB and 1024 query rows**, independent of
  batch size. Large inputs are processed in slabs rather than an unbounded quadratic score
  allocation. This scratch is additional to the returned indices. Triton's default
  FP16/BF16 route needs no global score scratch; its GVR2 route uses a separate
  scorer and the same 32 MiB / 1024-row bound.
- `kernel_options={"backend": "cute"}` or `{"backend": "triton"}` overrides that choice.
  Omit options for automatic selection. Options are rejected for `impl="reference"`.
  Unsupported shapes, missing dependencies, and launch errors propagate;
  there is no retry with another backend.

| FP16/BF16 restriction | CuTe | Default Triton selector |
|---|---|---|
| GPU | SM100 or SM103 | SM90 or newer |
| Input dtype | FP16 or BF16, shared by all inputs | FP16 or BF16, shared by all inputs |
| Heads `H` | Positive and even | `1..256` |
| Head dimension `D` | Positive, divisible by 16 | `8..256`, divisible by 8 |
| Sequence length `T` | `1..2**20` | `1..2**20` |
| `topk` | `0..S` | `0..S`; per-tile cost grows with `topk` |
| Layout | Q/K last stride 1, bases and non-singleton outer strides 16-byte aligned; weights may be strided | Q/K last stride 1, bases and outer strides 16-byte aligned; weights may be strided |

CuTe accepts independently permuted or padded outer dimensions and broadcast inputs without
materializing contiguous copies. Weights need only element alignment, not TMA alignment.
Triton GVR2 uses a separate scorer accepting arbitrary nonnegative input strides; the
Triton Q/K alignment restrictions in the table apply only to its default selector.

CuTe additionally requires the optional `linear` dependencies. It supports SM100 and SM103
(including GB300), not every Blackwell variant; other supported devices use Triton by default.

### Selector: GVR2 and `auto`

`kernel_options["selector"]` chooses the exact Top-K algorithm. Omitting it (or passing
`"auto"`) lets CuTe use GVR2 when `topk < 2048` and its radix selector otherwise, and keeps
Triton's streaming Top-K. `"default"` forces the backend-native selector (CuTe radix, Triton
streaming); `"gvr2"` forces GVR2 on either backend. GVR2 guesses a cutoff from a row sample,
verifies it over the full row, and refines exactly with radix passes when the guess misses,
so the result is the exact Top-K of the scorer's finite FP32 values.

```python
indices = lightning_indexer(
    q,
    k,
    weights,
    topk=512,
    causal=True,
    compress_ratio=4,
    kernel_options={"backend": "cute", "selector": "gvr2"},
)
```

The `auto` boundary comes from the [selector sweep](indexer_gvr2_performance.md): at
`topk >= 2048` GVR2 falls back to radix in-kernel and cannot win. On Triton, `"gvr2"` also
switches to a separate tiled scorer, so scores may round differently from the default path.

The benchmark CLI accepts both backend and selector lists. For example:

```bash
uv run benchmarks/sparse/indexer_benchmark.py --batch 1 --heads 64 --head-dim 128 --sequence-length 65536 --compress-ratio 4 --topk 512 --dtype bfloat16 --backend cute triton --selector default gvr2 --warmup 5 --rep 30
```

Timing includes scoring, in-kernel scale loading/packing, and selection, but excludes caller
compilation, allocations, and graph capture. Force `"selector": "default"` or `"gvr2"` to
compare both exact paths against the `auto` choice on your own shapes.

### Compilation and training scope

Both fused backends support the **public API** under `torch.compile(fullgraph=True)`,
including changing batch/sequence sizes with `dynamic=True`. Backend selection and TMA
setup happen inside a private operator on real CUDA tensors, rather than during tracing.
Static-shape CUDA Graph capture/replay is supported after warming up compilation:

```python
import torch

compiled_indexer = torch.compile(lightning_indexer, fullgraph=True, dynamic=True)
indices = compiled_indexer(q, k, weights, 128, causal=True)
```

**Selection is nondifferentiable.** Inputs may require gradients, but the integer result
has no gradient function. Selected attention can train its own Q/K/V computation with
these indices held fixed. It does **not** propagate gradients through the selection step
into the indexer's queries, keys, or scoring weights. This API supplies no surrogate gradient
or indexer-training loss.

Selection with NaN/Inf scores is unspecified and may differ across backends.

::: attn_gym.sparse.indexer.lightning_indexer
