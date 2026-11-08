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
Weights may be negative. Without packed offsets, `k` holds `S = T // compress_ratio`
candidates, each summarizing `compress_ratio` consecutive tokens (a trailing partial window
forms no candidate). For packed documents, pass paired `cu_seqlens` and `cu_seqlens_k`:
compression and causal positions restart per document, and output indices are document-local.
With causal selection, query `t` considers only candidates `s < (t + 1) // compress_ratio`, those
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
| GPU | SM100 or SM103 | SM90 or newer |
| Input dtype | FP16 or BF16, shared by all inputs | FP16 or BF16, shared by all inputs |
| Heads `H` | Positive and even | `1..256` |
| Head dimension `D` | Positive, divisible by 16 | `8..256`, divisible by 8 |
| Sequence length `T` | `1..2**20` | `1..2**20` |
| `topk` | Non-negative; missing candidates pad with `-1` | Non-negative; missing candidates pad with `-1`; per-tile cost grows with `topk` |
| Layout | Q/K last stride 1, bases and non-singleton outer strides 16-byte aligned; weights may be strided | Q/K last stride 1, bases and outer strides 16-byte aligned; weights may be strided |

For nonzero `topk`, CuTe limits candidate capacity `S` to `2**22` so at least one query
pair's FP32 scores fit its 32 MiB workspace. This includes inactive packed capacity.

CuTe accepts independently permuted or padded outer dimensions and broadcast inputs without
materializing contiguous copies. Weights need only element alignment, not TMA alignment.

CuTe additionally requires the optional `linear` dependencies. It supports SM100 and SM103
(including GB300), not every Blackwell variant; other supported devices use Triton by default.

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
has no gradient function. Gather attention can train its own Q/K/V computation with
these indices held fixed. It does **not** propagate gradients through the selection step
into the indexer's queries, keys, or scoring weights. This API supplies no surrogate gradient
or indexer-training loss.

Selection with NaN/Inf scores is unspecified and may differ across backends.

::: attn_gym.sparse.indexer.lightning_indexer

## Paged sparse decode

`paged_sparse_decode` is the serving-side counterpart: decode queries attend to indexer-selected
entries of a paged KV cache plus an optional causal window, in one softmax with an optional
attention sink. Indices stay **logical** (positions within each request, `-1` padded); the
kernel translates them through the block table itself, so no expanded token list or physical
slot list is built per step.

```python
from attn_gym.sparse import PagedKV, paged_sparse_decode

kv = PagedKV(key_cache, value_cache, block_table)
# key_cache/value_cache: [num_pages, page_size, kv_heads, head_dim]; block_table: int32
out = paged_sparse_decode(
    query,              # [tokens, heads, head_dim]
    kv,
    indices,            # int32 [tokens, topk] logical block ids
    cu_seqlens_q,       # int32 [num_sequences + 1]
    seqused_k,          # int32 [num_sequences], keys including the current queries
    max_seqlen_q=1,
    topk_length=topk_length,  # optional int32 [tokens]: valid prefix of each row
    index_block_size=4,
)
```

Each index selects `index_block_size` consecutive entries. The window holds the last
`window_size` positions of `window_kv` (a separate cache) or, by default, of `kv` itself. With
a shared source the window start rounds down to a block boundary and selections inside the
window are skipped, so each key is attended once and the incomplete block holding the query
(which an indexer over complete blocks cannot select) is always included.

| Model | Mapping |
|---|---|
| Qwen QSA | compressed-block ids, `index_block_size=4`, `window_size=0`, separate K/V, GQA |
| DeepSeek V3.2 DSA | token ids, `index_block_size=1`, values as `key_cache[..., :512]` |
| DeepSeek V4 / V4.1 | compressed-entry ids into `kv`, raw cache as `window_kv`, `window_size=128`, `attention_sink` |

### Implementations

- `impl="reference"` builds each row's key set explicitly on CPU or CUDA; it is the
  correctness oracle and supports every shape.
- `impl="fused"` (default) runs a Triton split-K kernel and an LSE merge on CUDA, under
  `torch.compile(fullgraph=True, dynamic=True)` and CUDA graph capture. Batch, sequence count,
  and split count are runtime values, so changing batches do not recompile. The split count
  comes from a wave-occupancy model of the GPU's SMs. It needs fp16/bf16 inputs, power-of-two
  head dims of at least 16, and at most 16K elements of `group x head_dim` per KV head (for
  example 16 x 256 or 32 x 512). Quantized caches (FP8/FP4 DeepSeek formats) are not supported
  yet; DeepSeek V3.2's 576-wide latent needs the reference path.

On B200 with Qwen3.8-Flash-Next QSA shapes (24/2 heads per rank at TP1, 6/1 at TP4, head dim
256, 512 blocks of 4, page 64), the fused kernel matches vLLM's native QSA kernel at batch 1
and is 5-25% faster from batch 4 to 128 (up to 1.5x with 2048-block budgets), before counting
the separate index-expansion kernel vLLM runs first. See
`benchmarks/sparse/paged_sparse_decode_benchmark.py --vllm-root <checkout>`.

::: attn_gym.sparse.paged_decode.paged_sparse_decode
