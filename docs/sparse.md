# Sparse attention

## Lightning indexer

`lightning_indexer` selects a shared pool of key positions using multi-head weighted
ReLU scores. It returns indices, not attention outputs or differentiable scores:

```python
from attn_gym.sparse import lightning_indexer

indices = lightning_indexer(q, k, weights, topk=128, causal=True)
# q: [B, T, H, D], k: [B, T, D], weights: [B, T, H]
# indices: [B, T, 128], int32
```

The score for a query/candidate pair is
`sum_h(weights[h] * relu(dot(q[h], k))) / sqrt(H * D)`.
Weights may be negative. Query and candidate lengths must match. With causal selection,
query `t` considers only candidates `0..t`; rows with fewer than `topk` candidates contain
`-1` padding. Mask padding before gathering: a raw PyTorch index of `-1` selects the last
position rather than an invalid position. `topk=0` returns an empty last dimension.
Output order and tie-breaking are unspecified, including between repeated calls.

### Implementations and device dispatch

- `impl="reference"` evaluates PyTorch scoring and Top-K on CPU or CUDA. FP16/BF16/FP32
  inputs accumulate in FP32; FP64 inputs retain FP64. It materializes the score intermediates
  and is intended for correctness checks and small inputs.
- `impl="fused"` (the default) selects **CuTe on SM100**, or **Triton on other NVIDIA GPUs
  with compute capability 9.0 or newer**, including Hopper. Both optimized implementations
  keep their selection state on chip. CuTe separates score generation from radix Top-K and
  reuses a per-call FP32 score slab capped at **32 MiB and 1024 query rows**, independent of
  batch size. Large inputs are processed in slabs rather than an unbounded quadratic score
  allocation. This scratch is additional to the returned indices; Triton needs no global
  score scratch.
- `kernel_options={"backend": "cute"}` or `{"backend": "triton"}` overrides that choice.
  Omit options for automatic selection. Options are rejected for `impl="reference"`.
  Unsupported shapes, missing dependencies, and launch errors propagate;
  there is no retry with another backend.

| Restriction | CuTe | Triton |
|---|---|---|
| GPU | SM100 | SM90 or newer |
| Input dtype | FP16 or BF16, shared by all inputs | FP16 or BF16, shared by all inputs |
| Heads `H` | Positive and even | `1..256` |
| Head dimension `D` | Positive, divisible by 16 | `8..256`, divisible by 8 |
| Sequence length `T` | `1..2**20` | `1..2**20` |
| `topk` | `0..min(T, 512)` | `0..min(T, 512)` |
| Layout | All inputs contiguous, with 16-byte-aligned bases | Q/K last stride 1, bases and outer strides 16-byte aligned; weights may be strided |

CuTe additionally requires the optional `linear` dependencies. Its support is specifically
SM100, not every Blackwell variant; other supported devices use Triton by default.

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
into the indexer's queries, keys, or scoring weights. Train those scoring parameters with
a separate objective; this API supplies no surrogate gradient or indexer-training loss.

Selection with NaN/Inf scores is unspecified and may differ across backends.

::: attn_gym.sparse.indexer.lightning_indexer
