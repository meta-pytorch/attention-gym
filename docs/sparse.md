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
  keep their selection state on chip. CuTe generates scores with
  [warpdrive](warpdrive/index.md) TMA/MMA/epilogue kernels, separately from radix Top-K, and
  reuses a per-call FP32 score slab capped at **32 MiB and 1024 query rows**, independent of
  batch size. Large inputs are processed in slabs rather than an unbounded quadratic score
  allocation. This scratch is additional to the returned indices; Triton needs no global
  score scratch.
- `kernel_options={"backend": "cute"}` or `{"backend": "triton"}` overrides that choice.
  Omit options for automatic selection. `impl="reference"` validates options and ignores them.
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


## Gather attention and sparse MLA

`gather_attn` takes a query, a local KV pool, a sparse KV pool, and selected sparse indices.
Each KV tensor serves as both key and value. The local window and sparse selections share one
softmax. Packed selections are document-local; invalid indices are ignored and duplicates retain
their multiplicity. The caller is responsible for sparse causality.

Inputs can be either:

- a **Tensor**, used directly for the query or KV operand; or
- a **two-tensor tuple** `(latent, rope)`, with an optional readability helper
  `RopePair(latent=..., rope=...)`.

The second component is already rotated by the model. It adds a score term, not a value term.

For query position `t`, let `M_t` be the **multiset union** of:

- local positions `max(document_start, t - sliding_window_size + 1) <= j <= t`, including
  the current position; and
- valid sparse selections from every slot of `kv_indices[..., t, :]`.

Each occurrence `i` supplies a key and latent value from its own pool. Duplicate sparse slots,
and local/sparse occurrences referring to the same tensor row, contribute separately:

```text
z_i    = scale * (query.latent · kv_i.latent + query.rope · kv_i.rope)
Z      = sum_{i in M_t} exp(z_i) + exp(s_h)
output = sum_{i in M_t} exp(z_i) * kv_i.latent / Z
LSE    = log(Z)
```

Bare tensors omit the rotary dot product and use the whole KV tensor as both key and value.
The optional per-head sink `s_h = attention_sink[h]` is **unscaled**: it adds only to the
softmax denominator and LSE, never to the value numerator. Without a sink, `s_h = -inf`.
A row with no real keys returns zero output and its sink logit as LSE, or `-inf` without a
sink. In particular, when `Z = 0`, output is defined as zero rather than dividing by zero;
that row's output-loss gradient contributions to all inputs are zero.

The same query pair attends to both pools; each pool provides its own rotary keys. The latent
KV gradient includes both its key and value contributions. The rotary components have separate
gradients. Tuple construction stores references only; it does not concatenate or copy tensors.
The output width is the latent width—there is no `value_dim` argument.

### DeepSeek-V4: bare tensors

DeepSeek-V4 applies RoPE within its shared 512-wide representation before calling attention,
and inverse RoPE to the corresponding output channels afterward. It uses bare tensors:

```python
from attn_gym.sparse import gather_attn

out = gather_attn(
    q,                   # [B, H, T, 512]
    local_kv,            # [B, 1, T, 512]
    compressed_kv,       # [B, 1, S, 512]
    indices,             # [B, T, topk]
    attention_sink=sink,
    sliding_window_size=128,
    scale=model_scale,
    cu_seqlens=cu_q,
    cu_seqlens_k=cu_k,
)
```

### GLM absorbed MLA: separate latent and rotary components

The non-rotary query must already be multiplied by the key up-projection. The shared latent
remains both a key component and the value; the extra rotary keys affect only the scores.

```python
import torch
from attn_gym.sparse import RopePair, gather_attn


def sparse_mla(q_latent, q_rope, kv_latent, k_rope, indices, cu_q, cu_k):
    return gather_attn(
        query=RopePair(q_latent, q_rope),    # [1, H, T, 512], [1, H, T, 64]
        local_kv=None,
        sparse_kv=(kv_latent, k_rope),      # [1, 1, S, 512], [1, 1, S, 64]
        kv_indices=indices,
        sliding_window_size=0,
        scale=1 / 16,  # GLM's original expanded QK width is 192 + 64.
        cu_seqlens=cu_q,
        cu_seqlens_k=cu_k,
    )


compiled_mla = torch.compile(sparse_mla, fullgraph=True, dynamic=True)
# out_latent = compiled_mla(q_latent, q_rope, kv_latent, k_rope, indices, cu_q, cu_k)
# out_latent: [1, H, T, 512]; the model then applies its value up-projection.
```

Paired inputs require an explicit `scale`, **including zero-width rotary pairs**: absorbing
the key projection changes the tensor width, not the model's attention scale. Bare tensors
retain their default inverse-square-root head-dimension scale. `local_kv=None` requires
`sliding_window_size=0`; it avoids packing an unused local pool. Each supplied pool must match
the query's bare/paired form, even when the local window is zero.

### Backends and compilation

Automatic fused selection prefers eligible CuTe/FA4 calls on SM100/SM103, otherwise it
selects Triton. This is **not a general paired-input fallback**: nonempty rotary components
run only on the restricted Triton shared-KV schedule below. In particular, nonempty paired
FP16 calls are not supported by either fused backend; use the reference implementation for
those inputs.

Here `H` is the query head count, `D` the latent width, `R` the rotary width, and `W` the local
window size. All query/KV components must share a dtype and device; the optional sink can
also use FP32. Local and sparse pools must have the same KV head count: one shared head or
`H` per-head operands.

| Input form | CuTe (FA4 sparse MLA) | Triton |
|---|---|---|
| Bare tensors | SM100/SM103, BF16, one shared KV head, `1 <= H <= 128`, `D = 512`; no additional window-size bound in the adapter | Generic CUDA path for FP16/BF16/FP32, shared or per-head KV; no fixed `H`, `D`, or `W` eligibility bounds in the generic launcher. A specialized BF16 shared-KV schedule uses the same GPU, `H`, `D`, `W`, and latent-stride constraints as nonempty pairs below, without rotary requirements. |
| Nonempty `(latent, rope)` pairs (`R > 0`) | Same CuTe constraints as bare, with `R = 64` | Compute capability major >= 8 (Ampere or newer, including Hopper and SM120), BF16, one shared KV head, `16 <= H <= 128`, `D` in `{16, 32, ..., 128, 512}`, `R` in `{16, 32, 64, 128}`, `0 <= W <= 2048`; shared-schedule stride requirements below must hold. No generic paired path. |
| Zero-width pairs (`R = 0`) | Same eligibility as bare tensors | Same eligibility as bare tensors; rotary operands are omitted from the kernels and their gradients have empty feature dimensions. |

Zero-width pairs retain paired-form validation and require explicit `scale`; they do not gain
the bare-tensor default scale. Reference attention runs on CPU or CUDA and accepts either
shared or per-head KV for every form, without the fused backends' fixed feature/head bounds.
The generic Triton launcher does not define a universal maximum latent width: compilation and
device resource limits still apply, so its lack of a fixed bound is not a promise that every
positive width can launch.

**Layouts and dependency constraints:**

- The Triton shared schedule requires query latent feature stride 1, sparse latent KV feature
  stride 1, and, when `W > 0`, local latent KV feature stride 1. KV head strides must be zero;
  the public API expands shape-one shared KV to zero-head-stride views. Rotary KV must also
  have zero head stride, but need not have unit feature stride. Required query layouts are
  normalized by the launch path. Without a window, local operands are not read and do not
  restrict this schedule. These rules also govern the specialized bare-tensor schedule, except
  for the rotary requirements.
- Generic Triton uses stride-aware KV loads when a tensor-descriptor layout is unavailable;
  the shared schedule's unit-KV-feature-stride restriction does not constrain that bare or
  zero-width fallback.
- CuTe's eligibility predicate does not impose query/KV stride bounds. FA4 applies its own
  layout normalization and may copy incompatible layouts. A singleton sink whose stride is
  not 1 is excluded from automatic CuTe selection, even if PyTorch regards it as contiguous.
  FA4 must provide sparse-MLA sink support when a sink is supplied, and head-padding support
  when `H < 128` (FA4 4.0.0b32+); default probability recomputation requires 4.0.0b33+.

Pool rows that no query selects must be finite on Triton's grouped schedule (separate values or
`1 < kv_heads < H`). With `W = 0`, tiles of nearby queries whose selections fall in a short
span read every key and value row in that span, selected or not, and give unselected rows zero
probability. Zero times NaN or Inf is still NaN, so a non-finite unselected row near selected
ones can make those queries' outputs and gradients NaN. Zero-fill fixed-capacity pool padding
rather than leaving it NaN.

With `local_kv=None`, Attention Gym passes the latent and rotary sparse pools as separate
operands without concatenating components or packing the pools. Eligible native layouts avoid
pool copies; FA4 may normalize other layouts. Supplying a local pool packs both pools into
FA4's candidate order. When `W = 0` and the sparse slot count is zero, fused attention instead
uses the common trivial-empty path: zero output, sink-inclusive LSE, no attention kernel
launch, and deterministic zero output-loss gradients, independent of backend kernel eligibility.

Both forward and backward support `torch.compile(fullgraph=True)`; static-shape CUDA Graph
training replay is supported, including Inductor's `mode="reduce-overhead"`. Tensor metadata
and determinism are guarded; device and installed FA4 capabilities are constant for a compiled
graph. Changing the installed dependency requires retracing or restarting.

Use `kernel_options={"backend": "cute"}` or `{"backend": "triton"}` to require a backend;
launch errors propagate without retrying another backend. The CuTe backend defaults to
`bwd_recompute_p=True`, recomputing probabilities in backward; set it to `False` to save them.
LSE can be requested with `return_aux=AuxRequest(lse=True)` and is nondifferentiable on fused
backends. Selection itself remains nondifferentiable.

FA4 sparse backward is not deterministic. Automatic selection uses Triton when deterministic
algorithms are enabled. The Triton kernels autotune by default, and their candidates differ in
floating-point order, so the winner picked by timing decides the output bits. Deterministic
algorithms, or `kernel_options={"autotune": False}`, launch each kernel's pinned configuration
instead (see [Determinism](determinism.md#autotuned-kernels)). Triton backward with nonempty
rotary components has no strict deterministic implementation; it raises in strict mode and
warns and runs in warn-only mode.
PyTorch does not allow changing
deterministic mode between a compiled forward and its backward.

::: attn_gym.sparse.gather_attn.gather_attn
