# Linear Attention

Attention Gym provides functional linear-attention operators with eager reference implementations.

This page is the API and implementation reference. Start with the operation you need:

- **Train or prefill:** [Gated Delta Rule](#gated-delta-rule) or
  [Kimi Delta Attention](#kimi-delta-attention), using the `chunk_*` operations.
- **Carry state between calls or decode:** the `recurrent_*` operations and the paged/decode
  variants in those sections.
- **Split a packed stream across ranks:** [Context Parallelism](#context-parallelism).
- **Understand changing sequence lengths under capture:** the
  [Ragged CUDA Graphs guide](guides/ragged-cuda-graphs.md).

## Chunk backend selection

`impl="fused"` selects optimized execution, not a particular backend. For chunk KDA and GDN,
including paged, staged and context-parallel calls, choose the kernels with
`kernel_options["backend"]`:

- `"auto"` (default) uses warpdrive when CuTeDSL is installed and supports the GPU: SM100, SM103,
  or SM107 with CuTeDSL 4.8 or newer. Otherwise it uses Triton.
- `"fused"` selects the Triton kernels (CUDA capability 8.0 or newer). They do not need CuTeDSL.
- `"warpdrive"` selects the CuTeDSL kernels. Install the `[linear]` extra; unsupported calls raise
  rather than switching to Triton. `"cudnn"`, the backend's former name, is an alias.

FP16 KDA follows the same rule as BF16. Paged calls that write
[state checkpoints](#paged-state-checkpoints-gdn-and-kda) are the exception: `"auto"` uses
Triton, and an explicit `"warpdrive"` raises because warpdrive cannot write checkpoints.

Selection does not depend on batch size or sequence lengths. A staged backward uses the backend
saved by its forward. Select `"warpdrive"` explicitly on forward calls to use `split_forward`,
`split_backward` or [`batch_invariant`](#batch-invariance).

Switching from Triton to warpdrive changes more than speed:

- **Numerics.** KDA uses 16-token chunks on warpdrive and 64-token chunks on Triton. GDN uses
  64-token chunks on both, but the kernels round differently. Do not expect bitwise equality
  across backends. On captured BF16 KDA inputs, default warpdrive had lower output and gradient
  errors against FP64 than Triton did; [Operand precision](#operand-precision) compares all
  three BF16 choices. Pass `{"backend": "fused"}` to keep using Triton.
- **KDA gate range.** The recommended training floor for both FP16 and BF16 is about `-5.25` on
  warpdrive and `-5.914` on Triton. The default
  `bound_gate` floor of `-5` fits both ([KDA gate contract](#kimi-delta-attention)).
- **Determinism.** warpdrive neither autotunes nor needs a tuning warm-up, and repeats its bits
  run to run, but its plan follows the packed shape and the SM count, so a sequence's bits can
  depend on the rest of the batch unless `batch_invariant` pins it
  ([Batch invariance](#batch-invariance), [Determinism](determinism.md#warpdrive-chunk-kernels)).
  Triton needs no batch-invariance flag for document-aligned context parallelism
  ([Document-aligned fragments](#document-aligned-fragments-cp-invariant-bits)).
- **Speed.** On GB300, the default was faster than Triton on every measured BF16 GDN and KDA
  prefill and training shape below; small single-sequence calls gain least. For BF16 KDA operand
  choices, see [Operand precision](#operand-precision). Across ten FP16 KDA shapes, the geometric
  mean speedups over the faster of pinned and autotuned Triton were 2.6× forward and 1.5×
  forward + backward. At 1 × 2048 tokens and eight heads, forward + backward was 2–4% slower.
  These timings predate the FP16 backward scale-placement change, which made FP16 forward +
  backward 1–6% slower on the three shapes in [Operand precision](#operand-precision). Other
  shapes and GPU models can give different results.

**Default versus Triton.** GB300, PR #907 on main 278c4f60, PyTorch `2.16.0.dev20261113+cu134`,
CuTeDSL 4.8.1, BF16 Q/K/V, head width 128. Each cell is warm CUDA Graph replay in µs, Triton
(`"fused"`, autotuned, the previous default) / default (`"auto"`, warpdrive here), with the
ratio in parentheses; medians of five interleaved rounds.
Prefill is a forward without gradients; training is a forward plus the gradients of every input.
**States** passes an initial state and returns the final one. Reproduce with
`python examples/linear/chunk_backend_benchmark.py --rounds 5 --replays 50`.

| Op | B × T, H | States | Prefill (µs) | Training (µs) |
| --- | --- | --- | ---: | ---: |
| GDN | 1 × 8192, 16 | no | 283 / 126 (2.24×) | 1173 / 420 (2.79×) |
| GDN | 1 × 8192, 16 | yes | 280 / 126 (2.22×) | 1173 / 425 (2.76×) |
| GDN | 8 × 2048, 16 | no | 385 / 68 (5.68×) | 1620 / 374 (4.33×) |
| GDN | 8 × 2048, 16 | yes | 390 / 76 (5.13×) | 1629 / 395 (4.12×) |
| GDN | 1 × 2048, 8 | no | 72 / 55 (1.30×) | 262 / 164 (1.60×) |
| GDN | 1 × 2048, 8 | yes | 72 / 61 (1.16×) | 260 / 174 (1.50×) |
| KDA | 1 × 8192, 16 | no | 387 / 208 (1.86×) | 1681 / 783 (2.15×) |
| KDA | 1 × 8192, 16 | yes | 389 / 210 (1.85×) | 1685 / 790 (2.13×) |
| KDA | 8 × 2048, 16 | no | 636 / 146 (4.37×) | 2371 / 886 (2.67×) |
| KDA | 8 × 2048, 16 | yes | 643 / 152 (4.24×) | 2384 / 898 (2.65×) |
| KDA | 1 × 2048, 8 | no | 77 / 70 (1.10×) | 317 / 221 (1.44×) |
| KDA | 1 × 2048, 8 | yes | 78 / 74 (1.06×) | 320 / 227 (1.41×) |

### Batch invariance

Set `kernel_options={"backend": "warpdrive", "batch_invariant": True}` if a sequence must give
the same results alone or packed with other sequences. Keep its tokens, entry state, incoming
gradients, math options, GPU model and software fixed. Its neighbors, their lengths and order,
and the sequence's offset in the pack may change. This covers:

- `chunk_kda` and `chunk_gdn`: output, final state and all gradients.
- Paged KDA (including replay) and GDN: output and updated state slots.
- Staged handles: `run` and state summaries in both directions. In context parallelism this
  covers unchanged rank-local fragments, not a different split of a document across ranks.

By default, warpdrive uses the packed shape and SM count to choose how to parallelize a scan.
It may split a long sequence into pieces, compute each piece's state transition, then compose
those transitions. KDA may instead share a preparation pass across two value tiles. Both can
change rounding. With `batch_invariant` the forward and backward each scan the sequence without
pieces, KDA skips shared prep, and staged summaries stay uncut. The forward may still split the
value dimension: the two tiles update independent state rows, so their width does not change
rounding. A staged backward inherits the forward's setting even if backward omits the flag or
passes `False`. Passing `True` after an unpinned forward raises.

The flag uses existing kernels. It leaves the default plan unchanged when neither direction
uses a piece chain, KDA does not share prep, and staged summaries do not split. A few long
sequences with few heads can be slower because they lose parallelism across pieces. The table
shows that trade-off in warm CUDA Graph replay on GB300 (BF16, median of three interleaved
rounds). The KDA measurements predate the matched TF32 backward kernels.

| Shape | Forward off → on (µs) | Forward + backward off → on (µs) |
| --- | ---: | ---: |
| KDA 1 x 32768, 4 heads | 332 → 1938 (5.8x) | 1187 → 9759 (8.2x) |
| KDA 1 x 32768, 16 heads | 592 → 1943 (3.3x) | 2274 → 9800 (4.3x) |
| KDA 64 x 512, 16 heads | 246 → 246 (1.0x) | 1278 → 1278 (1.0x) |
| GDN 1 x 32768, 4 heads | 152 → 707 (4.6x) | 574 → 4059 (7.1x) |
| GDN 1 x 32768, 16 heads | 270 → 712 (2.6x) | 1099 → 4080 (3.7x) |
| GDN 64 x 512, 16 heads | 123 → 123 (1.0x) | 684 → 685 (1.0x) |

Select `"warpdrive"` explicitly. A forward call cannot request `batch_invariant` with `"auto"`,
`"fused"`, `split_forward` or `split_backward`. The split schedules use the shape of the pack.

## Gated Delta Rule

`chunk_gdn` and `recurrent_gdn` use the token-major layout
`[batch, sequence, heads, dimension]` and return an output/state tuple. For each token, the scalar
natural-log gate decays the previous state before the delta update, and the query reads the updated
state:

```text
decayed_state = exp(gate) * state
residual = beta * (value - decayed_state @ key)
state = decayed_state + outer(residual, key)
output = scale * state @ query
```

Public and persistent state uses axis order `[N, H, V, K]`; paged pools replace `N` with the
slot count. `chunk_gdn` uses a chunk-parallel decomposition for training and prefill.
`recurrent_gdn` consumes
tokens in order for decoding, inference prefill, and state-carrying correctness checks. Like
`chunk_kda`, `chunk_gdn` defaults to the optimized chunk kernels (`impl="fused"`): warpdrive on
SM100-family GPUs and the Triton fused pipeline elsewhere
([Chunk backend selection](#chunk-backend-selection)); `impl="reference"` selects eager PyTorch.
With `kernel_options={"backend": "warpdrive"}` the warpdrive CuTeDSL backend accepts the same
approximate `split_forward` and `split_backward` forgetting-horizon options as warpdrive KDA (see
[Kimi Delta Attention](#kimi-delta-attention)), with the same no-state restriction. They help only
when few sequences leave the GPU idle and every head forgets quickly; heads that never forget stay
uncut. `recurrent_gdn(..., impl="fused")` selects the inference-only Triton scan. Pass
`kernel_options={"batch_invariant": True}` to `recurrent_gdn` or `recurrent_kda` to make a
sequence's bits independent of the rest of the batch
(see [Determinism](determinism.md#kda-determinism)); for the chunk ops, select warpdrive
explicitly ([Batch invariance](#batch-invariance)). These kernel controls live in
`kernel_options`, including `{"autotune": False}`, which launches each tuned kernel's pinned
configuration (see [Autotuned kernels](determinism.md#autotuned-kernels)); options that change the
math, such as `fastmath`, are keyword arguments.

```python
from attn_gym.linear import chunk_gdn

output, final_state = chunk_gdn(query, key, value, gate, beta, output_final_state=True)
```

### Supported capabilities

- Fixed-length inputs in `[batch, sequence, heads, dimension]` layout and packed batch-one inputs
  with `cu_seqlens`.
- Separate recurrent and chunked operations with explicit initial and final state.
- CPU and CUDA execution through eager PyTorch operations.
- A Triton fused chunk pipeline on CUDA capability 8.0+ with dense and packed inputs, grouped
  Q/K heads, tails, empty sequences, initial/final state, strict `torch.compile`, and CUDA Graph
  support. On CUDA capability 8.0+, `paged_chunk_gdn` advances selected
  `[num_slots, H, V, K]` cache rows in place for inference prefill without caller-side
  gather/scatter copies.
- A warpdrive chunk implementation, the default on SM100-family GPUs, with the same public
  training/state contract and direct paged-cache prefill.
- An inference-only fused recurrent implementation on CUDA, including mutable paged state caches
  shaped `[num_slots, H, V, K]` selected by `state_indices`.
- Autograd for the reference and fused chunk implementations, including initial-state gradients.
- Q/K/V share one dtype. FP16 and BF16 inputs use FP32 recurrence math and state while returning
  output in the Q dtype.
- Gate and beta may use independent floating dtypes and are converted to the recurrence compute
  dtype. A provided initial state uses FP32 for low-precision QKV.
- FP64 reference inputs retain FP64 recurrence math and state.

!!! note "What the fused paths keep in FP32"

    Every optimized GDN and KDA path (Triton fused and warpdrive) carries the recurrent state
    from `initial_state` to `final_state`, and the state cotangent from `d_final_state` to
    `d_initial_state`, in FP32 accumulators with FP32 decay. Internal chunk-entry states and
    state-cotangent tapes are staged in the Q/K/V dtype because they are tensor-core operands.
    Caller-owned `StateCheckpoints` instead store post-update checkpoints directly from the FP32
    accumulator. For the internal tapes, BF16 shares the FP32
    exponent range, so this only costs mantissa precision. FP16 also limits the range: state
    values that round beyond the FP16 finite range (about 65504) become non-finite when a chunk
    reads them, and cotangent values in the FP16 subnormal range (below `2^-14`) lose precision
    or round to zero before they reach `dgate`, `dk`, and `dv`. Keep FP16 states and state
    cotangents within the FP16 normal range, for example through loss scaling, or use BF16 when
    state magnitudes can leave it. Use L2-normalized Q/K with FP16, as the training example does:
    unnormalized Q/K or unusually large V can likewise push attention, solve, and value
    intermediates outside the FP16 range between FP32-accumulating GEMMs. BF16 KDA on warpdrive
    does not have this FP16 range limit (see [Operand precision](#operand-precision)).

The Triton fused chunk backend requires CUDA capability 8.0+, matching FP16 or BF16 Q/K/V,
and equal `K = V` of 64 or 128, but has no cuDNN runtime dependency. Kernels contract raw QK/KK
before applying masked nonpositive causal decay differences; the scalar natural-log gate has the
numerical-range contract below. The public
chunk size is BT64; internal 16x16 blocks are only the hierarchical triangular-solve representation.
Layouts requiring int64 tensor offsets are rejected until every fused kernel has a
wide-address path. Forward, backward, the context-parallel summaries and the gate transform run
Triton kernels on every supported architecture, so the fused backend needs no CuTeDSL.

The warpdrive chunk backend, the default on SM100-family GPUs, requires the optional `linear`
dependencies, an SM100-family GPU, FP16/BF16 Q/K/V, and `K = V` of 64 or 128.
Like the fused backend it runs gate, beta, and entry states in FP32 (returning their gradients in
the caller's dtypes), packs dense batches into one sequence per row, and copies operands whose
layout its kernels cannot address. It uses the same scalar natural-log gate contract and exact
execution by default.

#### Scalar gate numerical range

Chunk GDN gates must be finite and nonpositive. Both chunk backends accumulate the gates in FP32
(log2 units) within each 64-token chunk, then subtract prefix sums to obtain causal decay
exponents. The numerical contract requires finite prefix sums and enough FP32 precision to retain
these decay differences. Mixing extremely negative isolated gates with ordinary-sized gates can
lose subsequent small increments; it is not an unrestricted "no lower bound" contract.

For example, the following maximum output errors relative to an independent FP64 token recurrence
were measured on GB300 (SM103), PyTorch `2.16.0.dev20261112+cu134`, and CuTeDSL 4.8.1, with BF16
Q/K/V, FP32 gates, two heads, `K = V = 128`, packed lengths `[200, 65, 0, 127]`, and seed 21.
One gate every 37 tokens has the listed value; other gates
are in `[-1, 0]`. Error is `max(abs(output - reference)) / max(abs(reference))`:

| Isolated gate | Triton fused error | warpdrive error |
| --- | --- | --- |
| `-1e4` | 0.6% | 0.6% |
| `-1e5` | 1.5% | 1.6% |
| `-1e6` | 7.2% | 13.5% |
| `-1e7` | 159% | 373% |

Avoid isolated gates of magnitude `1e5` or larger among ordinary-sized gates. Spikes down to `-1e4`
and uniform gates down to `-1e30` were accurate in this workload; uniform huge gates do not lose
small increments in the same way. Uniform gates near `-1e38` can overflow the FP32 chunk prefix
and produce NaNs. These are empirical examples, not a universal safe lower bound: the distribution
within each chunk matters. Neither backend clamps gates or validates their values. The supported
isolated and uniform cases are covered by `test/linear/test_gdn_gate_range.py`.

The fused recurrent implementation requires CUDA with Triton, Q/K/V in FP16, BF16, or FP32, and
`K <= 256`. Packed offsets must begin at zero, be nondecreasing, and end within the physical token
capacity. Output rows beyond the terminal offset are inactive capacity and unspecified. Unsupported
implementation choices fail rather than falling back to the reference.

### Migration from the prototype API

- `naive_recurrent_gated_delta_rule(...)` and `gated_delta_rule(..., mode="recurrent")` become
  `recurrent_gdn(...)`.
- `naive_chunk_gated_delta_rule(...)` and `gated_delta_rule(..., mode="chunked")` become
  `chunk_gdn(...)`.
- Inputs use `[batch, sequence, heads, dimension]`, matching the KDA operations; keyword callers
  use `q`, `k`, and `v`.
- The public chunk size is fixed at 64, matching KDA's fused decomposition.
- `initial_state` may be positional, and `output_final_state` controls the optional state output.
- Both operations return `(output, final_state)`, matching the KDA operations.

`recurrent_gdn_decode` is the serving-specific one-token path, mirroring
`recurrent_kda_decode`: it consumes the packed post-convolution QKV buffer plus raw gate and
beta projections, computes the gate transform (`-exp(A_log) * softplus(raw + dt_bias)`), the
beta sigmoid, and the query/key L2 normalization in-kernel, supports grouped q/k heads, and
advances the paged FP32 state pool in place through `state_indices`, so no separate
elementwise kernels run per decode step.

::: attn_gym.linear.chunk_gdn

::: attn_gym.linear.paged_chunk_gdn

::: attn_gym.linear.recurrent_gdn

::: attn_gym.linear.recurrent_gdn_decode

### Paged state checkpoints (GDN and KDA)

`paged_chunk_gdn`, `paged_chunk_kda` (with or without `replay_state`), and paged `recurrent_gdn`
(with `state_indices`, including `kernel_options={"batch_invariant": True}`) accept
`state_checkpoints=StateCheckpoints(...)`. While advancing the running state cache, they write
selected post-update states directly into caller-chosen slots of a state pool. These checkpoints
let a serving engine publish prefix-cache entries without a follow-up state copy. Checkpointing
is inference-only, runs on the repo-local fused backend, and leaves return types unchanged.
The default `backend="auto"` runs checkpointed chunk calls on the fused backend even on
SM100-family GPUs; an explicit `"warpdrive"` raises.

For each sequence, pass the number of **new tokens in this call** before the selected prefix
and the pool slot that should receive its state. A non-positive destination skips the sequence.
The scheduler can choose a different prefix on the next call.

```python
from attn_gym.linear import StateCheckpoints, paged_chunk_gdn

# Packed sequence lengths are 600 and 300. Save sequence 0 after 512 new tokens
# into free slot 7 of the running cache; sequence 1 publishes nothing.
checkpoints = StateCheckpoints(
    state_pool=state_cache,
    checkpoint_positions=torch.tensor([512, 0], device="cuda", dtype=torch.int32),
    checkpoint_slots=torch.tensor([7, -1], device="cuda", dtype=torch.int32),
)
output = paged_chunk_gdn(
    q, k, v, gate, beta, state_cache, state_indices,
    cu_seqlens=cu_seqlens, state_checkpoints=checkpoints,
)
```

A positive destination requests a valid checkpoint. Without replay its position must be an
exact positive multiple of 64 within the sequence. With Triton debugging (`TRITON_DEBUG=1`),
an invalid request can trigger a device assert. In normal execution it leaves the destination
untouched: do **not** publish that slot as a cache hit. Use a non-positive destination to skip
intentionally. Both vectors are contiguous int32 `[N]` tensors on the input device; a captured
CUDA graph can read new positions and slots on each replay.

The pool has shape `[slots, H, V, K]`, contiguous within each slot, with optional padding
between slots. An FP32 pool preserves the recurrence accumulator; a BF16 pool rounds it.
The pool may be the running state cache but must not alias other inputs. The caller must
supply in-bounds destination slots, unique among writers and not routed by the call if sharing
the cache. Keep the position and slot vectors disjoint from storage the call mutates, including
replay counts. The kernel reads these vectors while updating the caches.

**KDA replay.** Positions still count only new tokens. The recurrent state, however, lives at
the last completed 64-token boundary, counting any tokens already buffered by a resumed slot.
For example, with 19 buffered tokens, position 45 is exactly a boundary. Position 50 selects
the same boundary state and needs five additional replay tokens to resume. A position before
the first completed boundary or outside the sequence publishes nothing.

Checkpointing writes the **boundary state only**. To make the selected slots resumable, call
`copy_replay_tails` **before** packed `paged_chunk_kda` on the same CUDA stream. Pass both
operations the same routing and requested positions, with destination slots distinct from all
active sources. The helper reads each source's pre-call count (zero for a fresh slot), copies
the trailing Q, K, V, gate and beta inputs and sets the destination count. At an exact boundary
it only resets the count to zero. Publish the slots after both operations; neither operation
alone produces a complete replay checkpoint. Resume with `has_initial_state=True` (or omit it).
See the
[prefix-cache example](https://github.com/meta-pytorch/attention-gym/blob/main/examples/linear/kda_prefix_cache_checkpoints.py)
for a bitwise comparison against a full prefill. It pins `{"backend": "fused"}` on every
call: on SM100-family GPUs the default runs calls without checkpoints on warpdrive, whose
rounding differs from the fused calls that wrote the checkpoints.

::: attn_gym.linear.StateCheckpoints

::: attn_gym.linear.copy_replay_tails

## Gate Transforms

`gate_transform` turns a raw gate projection into the FP32 natural-log decay consumed by the
chunked and recurrent operations. The transform kind and the gate shape are independent:

| `kind`       | formula                                              | range              |
| ------------ | ---------------------------------------------------- | ------------------ |
| `"bounded"`  | `lower_bound * sigmoid(exp(A_log) * (raw + dt_bias))` | `(lower_bound, 0)` |
| `"softplus"` | `-exp(A_log) * softplus(raw + dt_bias)`              | `(-inf, 0)`        |

Either kind applies to per-head gates (`raw [B, T, H]`, `dt_bias [H]`, the GDN convention) or
per-channel gates (`raw [B, T, H, D]`, `dt_bias [H, D]`, the KDA convention). `kind` is
required: a default could silently select the wrong recurrence convention. `bound_gate` is the
KDA-specific spelling of `kind="bounded"` for per-channel gates with `lower_bound=-5.0`.
Both producers default to `fastmath=True`; see [Fast math and precision](#fast-math-and-precision).

`impl="fused"` uses portable Triton kernels for both kinds and both gate shapes on every CUDA
architecture. On GB300 they took 1.1-1.2x the forward time and 1.1-1.25x the backward time of
the removed per-channel `D=128` CuTeDSL gate kernels (BF16, 16-134M elements). The backward
reduces the `A_log` and `dt_bias` gradients in FP32 from per-block partials, so they are
deterministic. The Triton bounded backward evaluates `sigmoid'(x)` as `e / (1 + e)^2` with
`e = exp(-|x|)`, so saturated logits keep gradients that `s * (1 - s)` would cancel to zero. The
decode kernels (`recurrent_kda_decode`, `recurrent_gdn_decode`) evaluate the same transforms
in-kernel from the raw projections and do not call this operation. Unbounded softplus gates
exceed the finite range some fused chunk-KDA schedules assume; keep per-channel softplus
gates for backends that document support for it.

```python
from attn_gym.linear import chunk_gdn, gate_transform

gate = gate_transform(raw_gate, A_log, dt_bias, kind="softplus")
output, final_state = chunk_gdn(q, k, v, gate, beta)
```

::: attn_gym.linear.gate_transform

::: attn_gym.linear.GateTransform

## Stateful Short Convolution

`causal_conv1d` is the differentiable dense or packed operation. It accepts compact
per-sequence history and can return a newly allocated final history. Serving callers can keep
history in a shared `[num_slots, W - 1, C]` pool instead: `paged_causal_conv1d` reads selected
slots during multi-token prefill and advances them in place, while `causal_conv1d_decode` does
the same for one token per sequence. Both are inference-only and support padding between slots.
Positive `state_indices` must be unique and within the pool; non-positive routes return zero without
touching it. `has_initial_state=False` gives paged prefill a zero history before it overwrites a
fresh slot.

`causal_conv1d` and `causal_conv1d_decode` never benchmark; `kernel_options` can override their
default schedules. To tune `causal_conv1d` explicitly, call `tune_causal_conv1d` and pass its
result as `kernel_options=tuned.kernel_options()`.

::: attn_gym.linear.causal_conv1d

::: attn_gym.linear.paged_causal_conv1d

::: attn_gym.linear.causal_conv1d_decode

::: attn_gym.linear.short_conv.ShortConvKernelOptions

::: attn_gym.linear.short_conv.ShortConvDecodeKernelOptions

::: attn_gym.linear.short_conv.ShortConvConfig

::: attn_gym.linear.short_conv.tune_causal_conv1d

## Kimi Delta Attention

The KDA references use token-major tensors: query, key, and per-channel gate are
`[batch, sequence, heads, key_dimension]`; value is
`[batch, sequence, heads, value_dimension]`; beta is
`[batch, sequence, heads]`. Both recurrent and chunked forms support ordinary
PyTorch autograd and an optional recurrent state shaped `[N, H, V, K]`. Paged prefill and decode
use the same axis order with the leading dimension interpreted as cache slots.

`examples/linear/delta_rule_training.py` builds these operations into a small trainable
`[B, T, hidden_size] -> [B, T, hidden_size]` attention module (`--variant=kda`; the same module
runs the GDN recipe with `--variant=gdn`, swapping only the gate and the core). To mirror the
main Kimi block structure, projected Q/K/V pass through a causal depthwise SiLU
convolution, the forget and output gates use two-stage factorized projections,
and the per-head output uses learned RMS normalization before sigmoid gating.
`--backend=reference` uses the PyTorch reference throughout. The optimized `chunk_kda`
core used by `--backend=fused` supports Ampere or newer and owns the gate scan behind the same
public boundary, as does its separate `bound_gate` producer. Implementations may inline the
scan, but cumulative gates and chunk boundaries are not caller-visible representations.

The example explicitly runs projections in BF16 while retaining FP32 parameters and gate
math; no ambient autocast context is required. Distributed mixed-precision policies must
preserve `A_log` and `dt_bias` as FP32 when parameters are materialized, rather than only
casting activations inside `forward`. A module-wide BF16 FSDP policy violates that
contract. A correctness-first integration can keep the KDA unit under an FP32 policy while
its projections explicitly compute in BF16; isolating only the strict-FP32 decay state is
a future bandwidth optimization.

The optimized `chunk_kda` core requires Ampere or newer and `head_dim=128`. Homogeneous FP16 or
BF16 Q/K/V keep their dtype; FP32 or mixed inputs run in BF16. Warpdrive uses 16-token chunks
([Chunk backend selection](#chunk-backend-selection)). The Triton backend uses 64-token chunks
and takes its direct dense path when `B=1` and `T` is a multiple of 64. It converts other dense
inputs to equal-length packed sequences; pass `cu_seqlens` for explicitly packed `[1, T, H, D]`
inputs. Both forms preserve sequence boundaries, including empty sequences and partial final
chunks, through forward, backward and recurrent states. For fixed-capacity execution, the
terminal offset may be smaller than physical `T`; primitive forward
values outside `[0, cu_seqlens[-1])` are unspecified. `paged_chunk_kda` uses the same
Ampere-or-newer forward route while updating selected FP32 state-cache slots in place. Both chunk
backends return zero cotangents for inactive gate rows (the fused internal reverse scan writes
them; warpdrive zeroes them in its backward prologue). This does not sanitize arbitrary
parameterized gate producers: callers still need the masking rules below because ``0 * NaN`` can
poison their reductions.

!!! warning "FP16 intermediate range"

    Use L2-normalized Q/K with FP16, as the training example does, and keep states and state
    cotangents within the FP16 normal range. See "What the fused paths keep in FP32" in the
    Gated Delta Rule section for the boundary between the FP32 carry and the Q/K/V-dtype staging
    that this shares with GDN. BF16 on warpdrive keeps BF16 range under every
    `intra_chunk_dtypes` value ([Operand precision](#operand-precision)).

A captured graph with sequence capacity `N` keeps `cu_seqlens.shape == (N + 1,)`. If a
replay has `M <= N` nonempty sequences and `L <= T` active tokens, repeat the terminal
endpoint through the unused tail:

```text
[0, sequence_start_1, ..., L, L, ..., L]
```

The repeated ranges are ordinary empty sequences. Stateful APIs therefore retain `N`
state rows even when only `M` sequences are nonempty. Both `L` and `M` may change on
replay, but physical token capacity `T` and metadata capacity `N` may not.

Scheduling for this over-capture regime is automatic and selected independently for
each eligible chunk-parallel ragged kernel: a kernel switches to a bounded persistent
worker grid only when its capacity task count exceeds a few waves of that grid.
Persistent workers stride over the active chunk count built on device from
`cu_seqlens`, so chunk-compute launch overhead tracks active work rather than captured
capacity. Capacity-sized initialization and reduction work may remain. Exact or mildly
padded shapes keep capacity-sized STATIC grids whose padding CTAs return immediately; there is no
user-facing scheduling knob.

Callers that use dynamic active lengths within fixed-capacity tensors must opt in to
masking. Ragged primitives read only `[0, cu_seqlens[-1])` from token-shaped inputs,
including output cotangents, and leave the suffix of token-shaped outputs and input
gradients unspecified. Four edge rules make those primitives safe to compose:

- **Caller buffer → ordinary operation:** value-mask the buffer. A zero cotangent does
  not neutralize a NaN activation in a weight reduction: `0 * NaN` is still NaN.
- **Parameterized producer → ragged primitive:** add a gradient barrier so the
  primitive's unspecified input-gradient suffix cannot enter the producer's reduction.
- **Ragged primitive → ordinary operation:** value-mask the primitive output before the
  ordinary operation saves it for backward.
- **Ragged primitive → ragged primitive:** do nothing; neither primitive reads the
  inactive suffix.

Construct one device-resident predicate inside the captured graph and reuse it at every
boundary. This keeps all masks consistent and lets replay recompute them when
`cu_seqlens[-1]` changes without a host read or recapture.

```python
from attn_gym.linear.kda import (
    active_token_mask,
    mask_inactive_token_gradients,
    mask_inactive_tokens,
)

active_mask = active_token_mask(hidden, cu_seqlens)
hidden = mask_inactive_tokens(hidden, active_mask)  # Caller → ordinary projection.
projected = input_projection(hidden)
projected = mask_inactive_token_gradients(projected, active_mask)
stage = ragged_primitive_one(projected, cu_seqlens=cu_seqlens)
output = ragged_primitive_two(stage, cu_seqlens=cu_seqlens)  # No mask between primitives.
output = mask_inactive_tokens(output, active_mask)
output = output_projection(output)
output = mask_inactive_token_gradients(output, active_mask)  # Model boundary.
```

`mask_inactive_token_gradients(x, active_mask)` preserves forward values while zeroing
inactive automatic-differentiation paths. For contiguous packed CUDA tensors in eager
mode its autograd path aliases `x` in the forward and row-masks tangents and
cotangents; compiled graphs and unsupported layouts keep the
`torch.where(mask, x, x.detach())` form (which materializes an elementwise result) so
Inductor can own fusion. Automatic-differentiation paths are zero on inactive rows,
and subsequent derivatives inherit the same mask. Recurrent
and convolution states have one row per logical sequence rather than one row per physical
token, so token masks must not be applied to them.

::: attn_gym.linear.kda.active_token_mask

::: attn_gym.linear.kda.mask_inactive_tokens

::: attn_gym.linear.kda.mask_inactive_token_gradients

`DeltaRuleAttention.forward` passes explicit offsets to its short convolution and `chunk_kda`
(or `chunk_gdn`);
the selected implementation owns its sequence-local scan and scheduling. Set
`mask_inactive_capacity=True` only when the packed tensor reserves physical rows beyond
`cu_seqlens[-1]`; dense and exact-packed
callers leave it disabled and pay no masking cost. The optimized boundaries are
first-order and do not support higher-order autograd. Run
`python examples/linear/delta_rule_training.py --backend=fused --packed --batch-size=4 --tokens=256`
to sample token-level lengths from a truncated Zipf distribution, pack them exactly
into one physical batch, print their `cu_seqlens`, and pass those offsets through the
complete training step. The complete composed core forward and backward use private custom
operators with fake-tensor registrations and first-order autograd wrappers, so fused
`chunk_kda` supports strict `torch.compile(fullgraph=True)` and CUDA Graph capture for
fixed physical token capacity and sequence count. Packed reference execution is
eager-only: it reads `cu_seqlens` on the host to run each logical sequence
independently. Boundary values and the active token count may change on replay;
changing the physical token capacity or sequence count
requires recompilation or recapture. Pass `--compile` to compile the complete example as
one full graph. This keeps the custom KDA core behind its registered operator boundary
while allowing
Inductor to fuse the surrounding PyTorch normalization and remaining pointwise work.
The bounded gate itself uses private forward and backward operators. It can be
combined with `--profile`; compilation warmups run before the
trace starts. Profiling requires transformer-nuggets and writes a native Perfetto `.pftrace`
through the shared `attn_gym.testing.profile_trace` helper. Like FLA's default training path, its backward recomputes the W/U,
gated Q/K, recurrent-state, and
corrected-value intermediates instead of retaining them across the forward/backward
boundary.

Graph-safe active-token replay does not by itself make complete model time proportional
to `L`. The ragged short convolution, gate scan, and KDA core avoid reading inactive
token values, but the example's projections, output normalization, output gate, and
output projection still process physical capacity `T`. An end-to-end integration needs
active-prefix-aware surrounding operations to turn smaller `L` into a comparable step-time
reduction. Million-token training also requires model-level activation checkpointing and
context parallelism; this single-device example implements neither distributed policy.

The module can sit behind a transformer layer's attention slot while state is
threaded explicitly:

```python
from examples.linear.delta_rule_training import DeltaRuleAttention

attention = DeltaRuleAttention(hidden_size=512, num_heads=4, head_dim=128).cuda()
first = attention(hidden_states[:, :128], return_final_state=True)
second = attention(
    hidden_states[:, 128:],
    first.final_state,
    initial_conv_state=first.final_conv_state,
    return_final_state=True,
)
```

The example intentionally is not checkpoint-compatible with Kimi K3. A model
adapter must still provide exact checkpoint parameter names and initialization,
the model's packed-sequence metadata, cache layout, and distributed execution policy. The
short convolution, factorized gates, and learned gated RMS normalization match
the production structure but remain ordinary PyTorch teaching implementations.

The two public KDA cores share one gate contract: `gate` is the per-token
natural-log decay before any prefix sum. `chunk_kda` owns the natural-log-to-log2
conversion and the cumulative sum within each backend's chunk; `recurrent_kda` performs only the
conversion because recurrence consumes one token decay at a time. This keeps chunking out
of model code and lets callers switch execution modes without changing gate
representation. Custom producers should return finite, nonpositive values. The Triton fused
backend requires about `[-5.914, 0]` (`MAX_GATE_LOWER_BOUND_MAGNITUDE`). Warpdrive's recommended
training floor is about `-5.25` (`-WARPDRIVE_MAX_GATE_LOWER_BOUND_MAGNITUDE`). For finite
gate factors with
`fastmath=True`, each aligned 16-token sum in each channel must exceed `-126 * ln(2)` (about
`-5.45` per token for uniform gates). With `fastmath=False` the limit is about `-128 * ln(2)`.
Staying above that threshold does not guarantee accurate gradients. About `-5.25` is an empirical
training guideline for a uniform gate, not a safe cutoff for every input. Decayed Q/K operands
can round to zero. In backward, an inverse-decayed key times a score gradient can overflow before
the opposing decay cancels it. Large key components, large output gradients (including
loss scaling), or concentrating the decay on one token all reduce the margin. For example, at a
uniform `-5.25`, `2^-g` reaches about `2^114` after 15 tokens. With a unit key and a score
gradient of `2^15`, their product already exceeds the FP32 range.

FP16 cannot hold the inverse decay after all 16 gates at `-5.25` (about `2^121`). Warpdrive
FP16 therefore uses TF32 decay operands and forms the state-gradient products from the unscaled
cotangent and
`K_restore`, whose factors stay at or below one; it shares the gate range of the BF16 contract
above. In a benchmark predating the matched FP16 backward (normalized inputs, one 512-token
sequence, two heads, plus ragged sequences), FP16 relative RMS errors from a uniform
`-0.1` to `-5.25` and for Kimi-style gates were 0.05% on the output, 0.03-0.06% on dq, dk, dv and
dbeta and 0.05-0.14% on dgate, against 0.08%, 0.03-0.09% and 0.11-0.20% for fused FP16; the
forward is non-finite past the same `-126 * ln(2)` bound. The matched backward changed FP16
gradient errors on captured training inputs by under 1%. Finite does not mean accurate near the
bound, in either dtype: a decayed operand too small for its TF32 or BF16 operand rounds to zero
before the inverse decay cancels it. With sixteen `-5.45` gates, `k = e0` and a query component
of `2^-13` on that channel (`scale=1`, only the last token writing a unit value), the output was 0
instead of `2^-13`, with or without `fastmath`. The products with the recurrent state also still
read FP16 `Q_decay` and `K_decay`, which flush below the FP16 range: with five `-3.5` gates, an
entry-state value of 4096, a unit query selecting it and `beta = 0`, warpdrive and fused FP16
both returned 0 where the exact output is the FP16-normal `1.03e-4` (BF16: `1.03e-4`). The FP16
backward also stages the FP32 state cotangent `dS` itself as an FP16 operand. Each element must
round within 65504 (an exit cotangent of 65520 made dk, dv, dgate and dbeta non-finite where 65504
was accurate), but that per-element limit is not a safe loss-scaling bound: products built from
`dS` are also cast to FP16 before later factors shrink them. `dU = dS K_restore` is cast before
`beta` scales it, so one token with zero gate, a normalized key of 64 components `1/8`, `beta =
1/8` and a `dS` row of 8192 on those components gave dv = inf where the exact value is 8192.
`K_restore = k 2^(g_last - g)` falls below the smallest FP16 subnormal, `2^-24`, once
`g_last - g < -24 - log2|k|`, and then rounds up to `2^-24` or down to zero even when its product
with `dS` is FP16-normal, so the gradients lose or misstate that term: a key `2^-11 e0` followed
by fourteen `-ln 2` gates on its channel gave dv = 0 instead of `2^-13` (thirteen gates:
accurate). This rounding is shared with the forward, whose state update reads the same FP16
`K_restore`: with a value of 4096 at that first token, the FP16 final state was 0 instead of
`2^-13`. Widening `dS` alone therefore cannot recover the term; a TF32 or power-of-two-scaled
fix has to keep the key's range before that cast, in the forward and the backward, and the full
backward kernel has no tensor or shared memory left for it without a redesign. These limits are not checked at runtime. The training
example uses the Kimi-style FP32 transform
`lower_bound * sigmoid(exp(A_log) * (raw_gate.float() + dt_bias))`, but that model policy is not
part of the public KDA API.

Both cores select their implementation with `impl`: `"fused"` runs the optimized kernels
and enforces their constraints (the chunked core requires `head_dim=128` and Ampere or newer,
preserves homogeneous FP16/BF16 Q/K/V, normalizes FP32 or mixed inputs to BF16, and chunks at
64 tokens on the Triton fused kernels and 16 on warpdrive; the fused recurrent scan is
inference-only), while `"reference"`
runs the eager FP32 oracle behind the identical packed contract on any hardware and head
dimension, and stays differentiable. There is no automatic fallback between implementations, and
the chunk-versus-recurrent switch is caller policy (on B200 the scan wins below roughly 32 tokens
per sequence).

`recurrent_kda_decode` is the serving-specific one-token path. It consumes
channel-major post-convolution QKV (`[Q for all heads | K for all heads | V for all
heads]`), raw gate and beta projections, and a paged state cache. Q/K normalization,
gate activation, beta sigmoid, recurrence, output, and state-cache update run in one
Triton kernel. Callers may provide a stable output buffer for allocation-free CUDA
Graph replay.

Token-at-a-time decode groups floating-point additions differently than 64-row `chunk_kda`
training, so decode drifts from training numerics.
[`examples/linear/kda_zero_drift_decode.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/linear/kda_zero_drift_decode.py)
replays the current partial chunk with `chunk_kda` from the last FP32 boundary state, which
makes decode match training bit for bit when both calls use the same kernels and plan. The
example disables Triton autotuning. Warpdrive plans from the packed shape and SM count, so even
on the same GPU two shapes can choose different plans
([Determinism](determinism.md#warpdrive-chunk-kernels)).

For replay-backed `paged_chunk_kda`, a serving engine can plan its pool before allocating
device memory. `ReplayState.slot_layout(num_heads, head_dim)` returns the per-slot shape and
dtype of each cache in constructor order. To initialize empty slots, only the counts need to be zero:

```python
import torch
from attn_gym.linear import ReplayState

layout = ReplayState.slot_layout(local_num_heads, head_dim)
replay_state = ReplayState(
    *(torch.empty((num_slots, *shape), dtype=dtype, device=device) for shape, dtype in layout)
)
replay_state.count.zero_()
```

The current fused KDA kernel requires key and value dimensions of 128. A pool may pad the
stride between slots, and the count tensor may be shaped `[num_slots, 1]` or `[num_slots]`.

Replay runs on either chunk backend. Triton kernels maintain the replay caches for both, and the
prefix and tail chunk passes run on the backend that `kernel_options["backend"]` selects. The
result reproduces that backend's `chunk_kda` over the whole stream bit for bit when the inputs
are equal and both calls run the same plan:

- **Inputs:** replay casts Q/K/V to BF16 and stores boundary states in the pool's dtype. The
  guarantee therefore holds for BF16 Q/K/V and an FP32 pool. FP16 inputs match a BF16-cast
  `chunk_kda` instead (with `intra_chunk_dtypes` resolved as for BF16 inputs), and a BF16 pool
  rounds each completed boundary state before the next call resumes from it.
- **Kernel configurations:** the fused backend autotunes some kernels per token-count bucket and
  packed mode (the output kernel picks `BK` 32 or 64), so a 64-token replay window and a
  full-stream call can reduce in a different order. Bitwise agreement needs
  `kernel_options={"autotune": False}`, or the same tuning winners, on both calls.
- **Plans:** apart from those configurations the fused backend has a single plan. warpdrive's plan
  depends on how many sequences are packed and how long they are: with few sequences times heads per
  GPU it splits the value dimension, and a long full-stream `chunk_kda` can pick the piece-chain
  plan, while paged replay always runs the unchained plan. Either choice rounds differently, so a
  replay step can differ at rounding level from a `chunk_kda` call that packs a different number of
  sequences, or from one over a stream long enough to chain, even with the same request and head
  count ([Determinism](determinism.md#warpdrive-chunk-kernels)).

To select warpdrive explicitly, pass `kernel_options={"backend": "warpdrive"}` to `chunk_kda`
([Chunk backend selection](#chunk-backend-selection)). As on Triton, homogeneous FP16 or BF16
Q/K/V keep their dtype; FP32 or mixed Q/K/V run in BF16. Gate, beta and the entry state run in
FP32. The output keeps `q.dtype`; gradients return in their inputs' dtypes. Warpdrive uses
warpdrive CuTeDSL kernels (CuTeDSL 4.6.2 or newer, the `[linear]` extra) and has no cuDNN Frontend
runtime dependency. Like the other
implementations, it returns `(output, final_state)`; request the latter with
`output_final_state=True`. Dense calls, including `B > 1`, run as one packed sequence per batch
row. Operands with TMA-compatible innermost modes and aligned outer strides (beta: an
element-aligned contiguous inner mode) are passed without copies; others are copied first, as in
the fused backend. By default the forward uses the unsplit recurrence:
one persistent work item per sequence and head, which leaves the GPU underused for a few
long sequences at low head counts. The backward, with or without states, runs warpdrive's own
BT16 kernels and is exact by default. Callers that guarantee normalized keys, post-sigmoid beta,
and nonpositive decay increments may set `split_backward` and/or `split_forward` to `True` in a
warpdrive `kernel_options` mapping to opt into approximate forgetting-horizon splitting: a sequence is cut
only where its cumulative decay has crossed the kernel's threshold, and each cut item rebuilds its
entry state from zero over a short warmup window. Gates that never forget produce no cuts and
reproduce the unsplit result exactly; when cuts are accepted the result stays within the
low-precision error budget while exposing several times the parallel work. The threshold is a
margin of bits past the output dtype's half-ulp, not an underflow; see the module docstring and
`DEFAULT_LOG2_THRESHOLD` in `attn_gym/linear/_delta_rule/warpdrive/work_tables.py`. `split_forward`
affects only the forward recurrence: unless `split_backward` is also enabled, backward computes the
exact unsplit recurrence's gradient rather than the derivative of the split forward. The
implementation chooses the split count from the input geometry; no public split-size knob is
exposed. Split schedules currently require a no-state call, so context parallelism never uses them.
Warpdrive `paged_chunk_kda` updates the selected cache slots directly. It does not support
forgetting-horizon splits.

`kernel_options={"schedule": "persistent"}` is for CUDA graphs whose replays can carry far fewer
tokens than the capacity they were captured for. The default static grid launches one CTA per
capacity chunk, so a replay at 1/16 of capacity still pays for the empty CTAs; the persistent grid
strides over only the active work. Below roughly a quarter of capacity active, persistent wins
(measured 0.1-0.7x the static time on GB200); at full capacity it loses (1.1-2.4x). Outputs are
bitwise identical. For packed inputs it requires the TMA output path (Hopper or newer). The
warpdrive backend accepts every `schedule` value and ignores it: its persistent grids follow the SM
count and always stride over the active work only, so it needs no option for over-captured graphs.

### Operand precision

`kernel_options["intra_chunk_dtypes"]` controls warpdrive chunk KDA, including paged and staged
calls and context parallelism. It is not a GDN option. Within each 16-token KDA chunk, the first
dtype is for products of decayed keys and queries; the second is for the triangular solve and
its contribution to the output. A single dtype sets both. All products accumulate in FP32.

The table describes the standard recurrence, not approximate forgetting-horizon splits.
**Operands** lists decay / solve / inter-chunk state matmuls in forward and backward; B = BF16,
T = TF32, F = FP16. **Match** asks whether backward reconstructs the forward's rounded operands,
not whether gradients are exact in real arithmetic. **Causal** means
that at fixed shape and plan, later tokens cannot change earlier output bits through rounding.
This assumes finite intermediates; even a masked product of zero and NaN can produce NaN.
**FP64 error vs fused** ranks each choice against Triton by its error from the FP64 recurrence,
for the output and all five input gradients on the BF16 capture below. **Speed** ranks forward +
backward on the three BF16 shapes. Dashes mean native FP16 is outside this BF16 comparison.

| Choice | Operands | Match | Causal | FP64 error vs fused | Speed | Use when |
| --- | --- | --- | --- | --- | --- | --- |
| `"bf16"` | B/B/B | Yes¹ | Yes | Higher | Fastest | Minimize BF16 training time |
| `"tf32+bf16"` (BF16 default) | T/B/B | Yes¹ | Yes | Lower | Second | Balance accuracy and speed |
| `"tf32"` | T/T/B | Yes¹ | Yes | Lowest | Third | Minimize numerical error |
| `"tf32+fp16"` (FP16 default) | T/F/F | Yes¹ | Yes | — | — | Native FP16 |
| Triton fused | Separate kernels | No² | Yes | Baseline | Fourth | Use the portable backend |

¹ On the per-tile path all four warpdrive choices reconstruct the forward's rounded operands,
native FP16 included, although its error and speed fall outside the BF16 comparison.
Backward keeps the output scale outside the rounded query and score tiles and multiplies the
unscaled state cotangent by the same rounded `K_restore` used in the forward state update.
The shared-prep forward is an exception for every choice: it folds the inverse and beta into
records for both the output and state updates. Backward recomputes the per-tile formula and can
round differently.
Forward and backward may also choose their piece-chain plans independently.

² Triton uses 64-token chunks and its own factorization. It builds diagonal inverse blocks in
FP32 and stores the assembled inverse and attention scores in the Q/K/V kernel dtype. Its
backward can split decay factors around a different reference row than its forward. See
[Fast math and precision](#fast-math-and-precision).

Native FP16 uses TF32 decay products because inverse decay can overflow FP16. Its solve operands
are FP16 without per-tile range scaling. Optimized `chunk_kda` and `paged_chunk_kda` cast mixed
or FP32 Q/K/V to BF16. Paged replay uses BF16 even for FP16 inputs. In both cases, use a BF16
choice. The staged `chunk_kda_prepare` requires matching FP16 or BF16 Q/K/V; pass the same
precision to its backward preparation.
A precision choice incompatible with the kernel dtype raises
`ValueError` before dispatch. Triton and `impl="reference"` validate but ignore the option.
For bitwise comparisons across calls, see [Determinism](determinism.md#warpdrive-chunk-kernels).

**Measurements.** GB300, November 2026, PyTorch `2.16.0.dev20261113+cu134`, CuTeDSL 4.8.1,
BF16 Q/K/V, head width 128. Errors are relative RMS (error norm / reference norm) against an
FP64 recurrence, in units of 1e-3. The inputs are 4096 tokens × 8 heads from step 600 of
`examples/linear/delta_rule_training.py`. Lower is better; rounding the FP64 output to BF16
alone gives about 1.66 in these units on the full capture.

Each latency cell gives **forward / forward + backward**, in µs of warm CUDA Graph replay
(median of three interleaved rounds; normalized Q/K, gates sampled uniformly in [-5, 0],
Triton autotuning enabled). The columns are A: 1 × 8192 tokens, 16 heads; B: 8 × 2048, 16 heads;
C: 1 × 2048, 8 heads. The `"bf16"` row was timed separately on the same GPU model. The FP16-input
path is not part of this BF16 comparison.

| Choice | Output | dq | dk | dv | dgate | dbeta | A (µs) | B (µs) | C (µs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `"bf16"` | 3.78 | 3.27 | 3.38 | 3.24 | 4.81 | 2.87 | 193/716 | 121/769 | 68/212 |
| `"tf32+bf16"` | 3.45 | 2.90 | 2.99 | 2.92 | 2.72 | 2.46 | 209/774 | 146/868 | 70/220 |
| `"tf32"` | 1.82 | 1.83 | 1.94 | 1.78 | 1.62 | 0.73 | 230/860 | 164/980 | 72/229 |
| Triton fused | 3.60 | 2.98 | 3.03 | 2.99 | 2.87 | 2.51 | 388/1659 | 637/2355 | 77/315 |

### Fast math and precision

`fastmath` controls gating exponentials in KDA's forward, backward, recomputation, and CP summaries.

| Backend | `True` (default) | `False` |
|---|---|---|
| Triton fused KDA | Fast, approximate exponentials | Non-fast exponentials that preserve FP32 subnormal results |
| `impl="reference"` | Normal PyTorch math | Normal PyTorch math |
| warpdrive KDA | Flush-to-zero approximate exponentials and reciprocals | Non-flushing approximate exponentials and reciprocals that preserve FP32 subnormal results |

The fused forward intra-chunk diagonal kernel rebases each 16-row subchunk at its first row, so
every output depends only on earlier gates. At steep gates the last rows' factors then fall below
`2^-126`: `fastmath=True` flushes them, losing those rows' within-subchunk terms at constant
gates below about `-5.82` (inside the `-5.914` bound), while `fastmath=False` keeps them as
subnormals. The fused backward intra-chunk kernel splits its decays so that a factor flushes only
when the decay it belongs to is below `2^-126`, so `fastmath=True` gradients stay accurate up to
the gate bound; such a decay is still lost with `fastmath=True` even when a large K entry would
make its term normal. Its in-subchunk factors reach `2^68.3` there, so
at the gate bound the backward needs Q entries and `beta`-scaled K entries below `2^59.7`.

warpdrive's forward and backward (`chunk_kda`, and the staged handles' `run` when unsplit) and its
unsplit CP summaries give bitwise-identical results in both modes while every in-chunk decay stays
at or above `2^-126`, as the [KDA gate contract](#kimi-delta-attention) implies. Split CP
summaries, and the staged passes over their pieces, do not: they compose their pieces in 3xTF32
with `True` and on the FP32 FMA walk with `False`, which round differently (by up to about `6e-7`
of the largest map entry in our tests) even when no decay flushes. Below `2^-126`, `False` keeps
the gate exponentials and their reciprocals from flushing, so a decayed state or summary entry
stays an FP32 subnormal instead of zero. The exception is the 16-row tensor-core walk of the piece
chain (used when sequences x heads x 16 exceeds the SM count): its TF32 operands flush subnormal
carried states to zero. The reciprocal of the in-chunk decay stays finite while every
aligned 16-token per-channel sum exceeds about `-128 * ln(2)` (a uniform gate of at least `-5.54`)
instead of `-126 * ln(2)`. Results stay finite there only while each key entry times that
reciprocal fits FP32, so `|k|` must stay below 2 at the limit; L2-normalized keys satisfy this.
Preserving subnormals does not restore contributions already lost when staging matmul operands.
The training guideline of about `-5.25` therefore applies with either setting.

Use the same value for staged forward and backward preparation. Choose it before warmup and
CUDA Graph capture. FP16/BF16 rounding and the documented gate-range limits apply in both modes.

`bound_gate` and `gate_transform` have separate flags, also defaulting to `True`.
For non-fast math in both gate production and attention:

```python
from attn_gym.linear import chunk_kda
from attn_gym.linear.kda import bound_gate

gate = bound_gate(raw_gate, A_log, dt_bias, fastmath=False)
output, final_state = chunk_kda(q, k, v, gate, beta, fastmath=False, output_final_state=True)
```

The training and CP examples use `--fastmath` by default; `--no-fastmath` selects `False`.
Paged chunk and replay execution use `True`.

::: attn_gym.linear.chunk_kda

::: attn_gym.linear.paged_chunk_kda

::: attn_gym.linear.recurrent_kda

::: attn_gym.linear.recurrent_kda_decode

::: attn_gym.linear.Impl

## Context Parallelism

Every delta-rule token step is affine in the V-major recurrent state, so any token range
collapses to one FP32 map `H_out = H_in @ A + B`, packed as `[HV, V + K, K] = [bias; transition]`
(one map per value head; GQA key heads are expanded by the factor kernels).
Context parallelism is therefore a prefix scan over these summaries, and the same machinery serves
KDA and GDN (on the fused backend, GDN's scalar per-head gate broadcasts onto the per-channel
summary kernels; warpdrive has native GDN summary kernels). The
public surface has three tiers so that new partitionings or communication topologies never add
options to an op.

### Terminology and index spaces

Anything a plan is built from is **global**; anything that touches a tensor on a rank is
**local**. `attn_gym.linear.context_parallel` carries the canonical `NOTE [Terminology]`.

| term | meaning | space |
|---|---|---|
| global stream | the whole packed token stream, `cu_seqlens_global` | global |
| sequence | one document in the global stream | global |
| fragment | one contiguous global token range a rank owns; may cover several sequences. A rank owns a list of them: one for a contiguous shard, two for zig-zag | global |
| subsequence | the tokens of one fragment that belong to one sequence; a fragment covering several sequences has several subsequences, and each becomes one `cu_seqlens` segment of the owner's span | global |
| span | the concatenation of a rank's fragments, in the order it listed them: the packed `q`/`k`/`v`/`output` tensors on that rank, with `cu_seqlens` marking its subsequence boundaries | local |
| chunk | the `attn_gym.linear.kda.stages.CHUNK_SIZE` (64) token block the fused kernels work in (WY factors and one state step per block); every subsequence is chunked on its own, starting at its first token, so its last chunk may be partial and no fragment cut needs to be chunk-aligned | local |
| summary | the `[bias; transition]` map of a token range of the span | local range |
| slot | `gathered[cp_rank][f]`: one per fragment. Forward, the summary of fragment `f`'s last subsequence; backward, the reverse map of its first; the identity when nobody needs it | (cp_rank, fragment) |
| predecessors / successors | the slots holding a fragment's sequence's earlier / later pieces, in token order | (cp_rank, fragment) |
| terminal | subsequences that end their sequence; true final states live there | local index |

Fragment cut points are arbitrary global tokens: chunking restarts at every subsequence, so they
need no alignment to sequences or chunks. With `cu_seqlens_global = (0, 40, 232, 384)` and the
zig-zag table `fragments_global = [[(0, 96), (288, 384)], [(96, 192), (192, 288)]]`:

```text
global tokens   0     40             96       192   232      288      384
sequences       |- s0 -|------------ s1 --------------|-------- s2 --------|
rank 0           [====== frag A ======]                        [= frag B =]
rank 1                                 [ frag C ][= frag D ==]

rank 0 span:  A∩s0 | A∩s1 | B∩s2     local cu_seqlens (0, 40, 96, 192)
rank 1 span:  C∩s1 | D∩s1 | D∩s2     local cu_seqlens (0, 96, 136, 192)
slot order:   s1 = A∩s1 -> C∩s1 -> D∩s1        s2 = D∩s2 -> B∩s2
```

`state_summaries(bounds)` takes an `int32 [R, 2]` device tensor of **local** span ranges and is
exact over whole chunks of one subsequence: `start = sub_start + 64·i`, `stop = sub_start + 64·j`
or the subsequence end, never crossing a `cu_seqlens` boundary; `start == stop` is the identity.
The recipe always passes whole subsequences, `(cu_seqlens[i], cu_seqlens[i + 1])`.

### Local computation: prepare, summarize, run

**Primitives** (`attn_gym.linear.kda.chunk_kda_prepare` / `chunk_kda_prepare_backward`, and
`attn_gym.linear.gdn.chunk_gdn_prepare` / `chunk_gdn_prepare_backward`) split the fused core
around the communication point without exposing its WY factors:

```python
from attn_gym.linear.kda import chunk_kda_prepare, chunk_kda_prepare_backward

prepared = chunk_kda_prepare(q, k, v, gate, beta, cu_seqlens=cu_seqlens)
summaries = prepared.state_summaries(bounds)  # [bias; transition] per range
# ...exchange summaries and compose each subsequence's entry state...
output, final_state = prepared.run(initial_state, output_final_state=True)

grads = chunk_kda_prepare_backward(prepared.saved, d_output, initial_state, scale=prepared.scale)
grad_summaries = grads.state_grad_summaries(bounds)  # reverse maps [bias; transition]
# ...exchange and compose each subsequence's exit cotangent...
dq, dk, dv, dgate, dbeta, _ = grads.run(d_final_state)
```

Each `bounds` row names one local `cu_seqlens` segment (or a whole-chunk run inside one), and
one call returns every row's map; the context-parallel routing supplies the rows. Both backends
split long ranges into pieces when the span's shape leaves SMs idle, scan the pieces in parallel
and compose their maps on the device (the Triton summaries through work items within a static
budget, warpdrive's as described under [warpdrive local execution](#warpdrive-local-execution)).
The split depends on the span length, sequence and head counts, selected tile size and SM count;
unsplit, whole ranges scan in one kernel without a planner or compose launch. Planning reads only
device values, preserving CUDA Graph replay across different `bounds` values of the same shape.
`attn_gym.linear.context_parallel` holds the pure-PyTorch algebra on summaries (`merge_state`,
`compose_summaries`, `neutral_summary`).

### Ownership: from global fragments to local routing

**Ownership plans** (`attn_gym.linear.context_parallel.ContextParallelPlan.from_fragments`) take
every rank's fragments. You choose them in plain Python; the plan cuts each fragment at sequence
boundaries into `Subsequence`s, one span `cu_seqlens` segment each, and derives the routing:
which fragments need a forward or reverse summary (one gather slot per fragment), which slots to
fold for each entry state or exit cotangent, and which subsequences end their sequence and
therefore hold true final states. Two pieces of one document on the same rank are simply two span
segments whose entry states the routing supplies, so contiguous shards, zig-zag load balancing, and
document-aligned partitions differ only in the fragment lists. The fragments must tile
`[0, total_tokens)` exactly once. `plan.routing(device)` materializes the same routing as
fixed-shape device tensors; `summary_slots` / `grad_summary_slots` fill a rank's `[slots, ...]`
buffer from a prepared handle and a routing, and `compose_entry_states` /
`compose_exit_cotangents` fold the gathered `[world, slots, ...]` buffers into each subsequence's
entry state or exit cotangent; the collective in between is the caller's.
[`examples/linear/delta_rule_context_parallel.py`](https://github.com/meta-pytorch/attention-gym/blob/main/examples/linear/delta_rule_context_parallel.py)
builds contiguous and zig-zag fragment lists in a dozen lines.

Routed folds use one **root-fold policy**, without a workload-dependent switch. Nonempty chains
for the same document on a rank share a root: its kernel program keeps the accumulator between
linked checkpoints rather than restarting each prefix from zero. Backward does the same in reverse
order. The `predecessor_links` and `successor_links` rows contain `(length, next_row, is_root)`;
empty and padded rows are singleton roots that produce zero. This preserves the sequential IEEE
FP32 arithmetic and requires no additional communication. It removes repeated prefix work but can
cost some parallelism for short chains.

### Communication: the all-gather recipe

**Reference recipe** (`attn_gym.linear.context_parallel.context_parallel_chunk`, bound as
`attn_gym.linear.kda.context_parallel_kda` and `attn_gym.linear.gdn.context_parallel_gdn`) moves
summaries with one padded all-gather per direction and owns the autograd function; it is generic
over the delta-rule variant through a `StagedOp` pair of the two `prepare` entry points, but not
over the communication topology. `context_parallel_conv_history` builds the short convolution's
`initial_state` from the same routing (`plan.routing(device, conv_history=W - 1)`). For a
point-to-point pipeline, a recursive-doubling scan over `compose_summaries`, DTensor, or
communication overlap, compose the staged primitives and the routing helpers (`summary_slots`,
`compose_entry_states`, ...) around your own collective; the primitives and plans are unchanged.
Sharding does not cost accuracy: against an FP32 reference, sharded gradients match the unsharded
fused op's error to within noise for both ops. When a rank's span is one whole subsequence (a
routing built with `max_subsequences=1`, the usual case for one long document over contiguous
shards), the recipe passes `cu_seqlens=None` to the stages, so a chunk-aligned span runs the same
dense kernels as an unpacked `chunk_kda` call; the choice follows the routing's shape, so it is
fixed per captured graph.

```python
from attn_gym.linear.context_parallel import ContextParallelPlan
from attn_gym.linear.kda import context_parallel_kda

# Two ranks, one fragment each: the global stream split down the middle.
total_tokens = cu_seqlens_global[-1]
half = total_tokens // 2
plan = ContextParallelPlan.from_fragments(
    cu_seqlens_global, [[(0, half)], [(half, total_tokens)]], cp_rank=dist.get_rank(group)
)
token_ids = plan.global_token_ids(device)  # gather global tensors into this rank's span
routing = plan.routing(device)  # the layout as fixed-shape device tensors
output, final_state = context_parallel_kda(q, k, v, gate, beta, routing=routing, group=group)
true_final_states = final_state[routing.terminal]  # boolean indexing syncs; see below
```

### CUDA Graph replay: fixed storage, changing layout

The recipe starts each sequence from zero, always returns every subsequence's exit state, and does
not accept `initial_state` or `output_final_state=False`. Every index it uses comes from the
routing tensors, so a CUDA Graph captured around it replays for any document layout and fragment
table within the routing caps: capture with `plan.routing(device, slots=S, max_subsequences=N)`
(at most `S` fragments per rank, `N` segments per span, the same span length), then copy each new
layout's routing tensors into the captured ones (`test/test_context_parallel_distributed.py`
checks small and highly fragmented layout changes). Eager callers pass no caps.

The host plan is not rerun by graph replay. Before each replay, build the new plan outside
capture, arrange the new inputs in its span order, and copy its routing values into the existing
device tensors. Do not replace those tensors: the captured kernels use their original addresses.
The process group, per-rank span length, routing caps, and convolution history width remain fixed.

The recipe returns `N` exit-state rows; only those where `routing.terminal` is set are true final
states. Boolean indexing produces a variable-size result and synchronizes, so keep all `N` rows
inside capture and use a fixed-shape device mask or a per-row cotangent buffer. The replay test
updates that cotangent buffer before each replay, setting nonterminal rows to zero. Host-side
`plan.terminal` indices are useful outside capture, or when terminal rows never change; capturing
an index selection from them does **not** make it follow later layouts. Update routing only
between complete forward/backward steps, not while backward still needs the previous layout.

### Document-aligned fragments: CP-invariant bits

If every fragment boundary is a document boundary, no rank needs another rank's state: each
document runs from the zero state to its end on the rank that owns it, exactly as the unsharded
op runs it. The recipe above then composes nothing (every entry state is zero), and its output,
gradients, and final states are **bitwise identical for any CP degree and any document-aligned
fragment table, including CP=1**; for KDA they are also bitwise the public `chunk_kda`. Nothing
else is required: no flag, no tiling, no change to the kernels. `document_aligned_fragments` in
`examples/linear/delta_rule_context_parallel.py` (`--partition documents`) assigns whole documents
to ranks, longest first onto the least-loaded rank, so ranks stay within one document's length of
each other. What it cannot do is split a document that does not fit on one rank; that is what
the summaries above are for.

This guarantee applies to Triton. Pass `kernel_options={"backend": "fused"}` to select it.
Warpdrive's default plan depends on packed shape and SM count, so changing which documents
share a rank can change a document's results. For example, KDA's forward takes the d_v-split prep
path while `num_seqs * heads` stays below about half the SM count, and a long document can leave
the forward piece chain for it when packed with a single other one. On a 152-SM GB300 with two
heads, packing a document with 40 others instead of at most one moved about 62% of its output
elements by one bf16 ulp (relative L2 4.5e-3, within the 1% that the plan tests allow). In those
public calls without states, its gradients stayed bitwise equal, including through the backward
piece chain; that observation does not cover the stateful staged gradients of a full context
parallel step. The public warpdrive `chunk_gdn`
forward stayed bitwise across the d_v split and the forward piece chain. Its backward piece chain
did not: taken for a few long documents, it changed dk, dv, dgate and dbeta at rounding level
(relative L2 at most 2.3e-4), while dq stayed bitwise. GDN's warpdrive recipe runs these kernels,
so the same holds for it; `test/test_context_parallel_distributed.py` checks the document-aligned
bits for short documents, which take neither chain.

### warpdrive local execution

On SM100-family GPUs the default backend (or an explicit
`kernel_options={"backend": "warpdrive"}`) runs each rank's local passes with warpdrive's
kernels, for KDA (`chunk_kda_prepare`, `context_parallel_kda`) and GDN (`chunk_gdn_prepare`,
`context_parallel_gdn`). The recipe is the same as for the fused backend: summarize every local
range as an affine map of its entry state, exchange and compose the maps, then run the local pass
from the composed state.

*Forward.* `prepare` stores only the inputs; warpdrive keeps its WY factors on chip and never
writes them out. `state_summaries` produces each range's `[B; A]` map (`S_exit = S_entry @ A + B`)
with a native summary kernel (`kda_summary`, or `gdn_summary` after one `gdn_tinv` pass for the
chunk inverses) that carries two chains over the range: one from a zero entry state, whose final
state is `B`, and one from the identity with the value term disabled, whose final state is `A`.
Which ranges are summarized and how their rows are ordered is decided on the device, so the call
is CUDA Graph replayable and an empty range yields the identity map. Every range must be a
subsequence (one rank's piece of a document, one segment of the local `cu_seqlens`), which is
all the CP recipes ever request; a nonempty range that is not one is filled with NaN. `run` is
warpdrive's forward from the composed entry state.

*Backward.* In a CP step `run` is warpdrive's own stateful backward
(`kda_chunk_cudnn_packed_bwd_with_state`, `gdn_chunk_cudnn_packed_bwd_with_state`): the state pass from the saved entry state writes
checkpoints, then the backward consumes them together with the exit cotangent, returning the
token gradients and the entry-state cotangent. `state_grad_summaries` produces each
subsequence's `[C; R]` map: `C` is the entry-state cotangent of the range's outputs under a zero
exit cotangent (`kda_bprop_summary`, `gdn_bprop_summary`), and `R` is the forward transition
transposed. No fused factors are computed anywhere on the warpdrive path. The summary kernels scan
a range's 16-token (KDA) or 64-token (GDN) chunks serially, one CTA per (range, head), so a few
long subsequences on few heads would leave most SMs idle. When the span's shape does that
(NOTE [Summary split cost] in `warpdrive/drivers/plan.py`), a device planner cuts each selected
subsequence into pieces of one span length, the kernels summarize every (piece, head) in
parallel, and the piece state chain composes each subsequence's pieces in order in 3xTF32, or in
FP32 with `fastmath=False`, whose subnormal decays 3xTF32 would round. Both directions use the
forward's piece length, and the reverse map is the exact adjoint of the composed forward map: `R`
is its `A` transposed bit for bit, and `C` sums every piece's cotangent through the transposed
transitions of the pieces before it, as the forward composed them (3xTF32 products added in piece
order; NOTE [Reverse maps are the forward's adjoint] in `warpdrive/drivers/summary.py`), so the
gradients are those of the function the forward ran. On a GB300 this takes the KDA forward summary
of a single 8192-token subsequence at four heads from about 550 µs to about 85 µs (CUDA-graph
time). The forward and reverse passes behind `state_grad_summaries` overlap on a side stream.

*Staged passes over the pieces.* When the summaries split, the local pass reuses their pieces (NOTE
[Staged piece plans] in `warpdrive/drivers/staged.py`). `prepare` plans the pieces and keeps the
plan on its tape, summarizes every piece of every sequence once and composes them, keeping each
piece's entry map; then `state_summaries` only gathers the composed maps, and `run` turns the entry
state into every piece's entry state with one 3xTF32 product (a sequence's first piece takes the
entry state itself) and runs the prefill over all pieces at once. `prepare_backward` runs the
tape's plan whatever its own options or operands: it recomposes the forward's entry maps bit for
bit, recomputes the checkpoints from the entry states the forward's prefill started from, and
builds the reverse maps, so `run` applies the exit cotangent and runs the backward over the pieces.
Its entry-state cotangent is `d_final_state @ R + C` with the sequence's gathered `[C; R]`, the
adjoint of the forward's entry-state application; an empty sequence passes its state and
cotangent through bit for bit. The maps are built once from the inputs at `prepare`
(`prepare_backward`), so a split handle's inputs must not change before its last use, and to replay
its summaries or `run` over new input values from a CUDA graph, capture `prepare`
(`prepare_backward`) with them, as `context_parallel_kda`/`context_parallel_gdn` do. Against
#926, whose split handles reran the public op in `run`, on a GB300 (CUDA-graph time of prepare,
summaries and run; medians of three interleaved rounds), one 2048-token sequence at eight heads
takes a KDA CP step's forward from about 129 µs to about 100 µs and its backward from about 198 µs
to about 179 µs, and GDN's from about 103 µs to about 80 µs and about 158 µs to about 143 µs; one
8192-token sequence at four heads takes KDA's from about 219 µs to about 149 µs and about 357 µs to
about 263 µs.

*Numerics.* KDA's warpdrive kernels chunk at 16 tokens: a document cut on 16-token boundaries
whose fragments exchange warpdrive's own state and cotangent reproduces the unsharded warpdrive
gradients bit for bit, but the composed entry states do not, so KDA under CP is its own numerical
baseline, neither the fused CP baseline nor unsharded warpdrive. A split summary's bits depend on
the span's token, sequence and head counts and the SM count (they fix the piece length), not on
the bounds or on other sequences' contents; `kernel_options={"backend": "warpdrive",
"batch_invariant": True}` on the staged handles scans every subsequence uncut instead, so its
summaries and `run` keep their bits when their neighbors change
([Batch invariance](#batch-invariance)). Compared with FP64, split maps are as accurate as the
unsplit scan,
whose 16-bit state operands already round the maps by about 4e-3 (bias) to 2e-2 (transition)
relative L2 over thousands of tokens, so split and unsplit maps differ at that level
(`test/test_wd_summary_split.py`). The same goes for the staged passes over the pieces: their
outputs, final states and gradients follow the piece length and are not the bits of the public
op on the same stream, but they are as close to FP64 as the public op's and depend neither on
the bounds nor on whether `run` comes before the summaries (`test/test_wd_staged_pieces.py`).
GDN's warpdrive kernels chunk at 64 tokens like the fused ones; sharded GDN matches the FP32
reference as closely as unsharded warpdrive and fused CP do (`test/test_delta_rule_stages.py`).
The staged handles and this recipe are eager-only; `torch.compile` is not supported through
them.

::: attn_gym.linear.context_parallel.context_parallel_chunk

::: attn_gym.linear.kda.context_parallel.context_parallel_kda

::: attn_gym.linear.gdn.context_parallel.context_parallel_gdn

::: attn_gym.linear.context_parallel.ContextParallelPlan

::: attn_gym.linear.context_parallel.Subsequence

::: attn_gym.linear.context_parallel.ContextParallelRouting

::: attn_gym.linear.kda.stages.chunk_kda_prepare

::: attn_gym.linear.kda.stages.chunk_kda_prepare_backward

::: attn_gym.linear.gdn.stages.chunk_gdn_prepare

::: attn_gym.linear.gdn.stages.chunk_gdn_prepare_backward
