# Linear Attention

Attention Gym provides functional linear-attention operators with eager reference implementations.

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

Public and persistent state uses V-major storage shaped `[N, H, V, K]`; paged pools replace `N`
with the slot count. `chunk_gdn` uses a chunk-parallel decomposition for training and prefill.
`recurrent_gdn` consumes
tokens in order for decoding, inference prefill, and state-carrying correctness checks. The caller
chooses the execution form explicitly. `impl="reference"` selects eager PyTorch;
`recurrent_gdn(..., impl="fused")` selects the inference-only Triton scan.

```python
from attn_gym.linear import chunk_gdn

output, final_state = chunk_gdn(
    query,
    key,
    value,
    gate,
    beta,
    impl="reference",
    output_final_state=True,
)
```

### Supported capabilities

- Fixed-length inputs in `[batch, sequence, heads, dimension]` layout and packed batch-one inputs
  with `cu_seqlens`.
- Separate recurrent and chunked operations with explicit initial and final state.
- CPU and CUDA execution through eager PyTorch operations.
- An inference-only fused recurrent implementation on CUDA, including mutable paged state caches
  shaped `[num_slots, H, V, K]` selected by `state_indices`.
- Autograd for reference inputs and initial state.
- Q/K/V share one dtype. FP16 and BF16 inputs use FP32 recurrence math and state while returning
  output in the Q dtype.
- Gate and beta may use independent floating dtypes and are converted to the recurrence compute
  dtype. A provided initial state uses FP32 for low-precision QKV.
- FP64 reference inputs retain FP64 recurrence math and state.

The fused recurrent implementation requires CUDA with Triton, Q/K/V in FP16, BF16, or FP32, and
`K <= 256`. Packed offsets must begin at zero, be nondecreasing, and end within the physical token
capacity. Output rows beyond the terminal offset are inactive capacity and unspecified. The fused
chunk implementation is not implemented yet; unsupported implementation choices fail rather than
falling back to the reference.

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

::: attn_gym.linear.recurrent_gdn

::: attn_gym.linear.recurrent_gdn_decode

## Kimi Delta Attention

The KDA references use token-major tensors: query, key, and per-channel gate are
`[batch, sequence, heads, key_dimension]`; value is
`[batch, sequence, heads, value_dimension]`; beta is
`[batch, sequence, heads]`. Both recurrent and chunked forms support ordinary
PyTorch autograd and an optional V-major recurrent state shaped `[N, H, V, K]`. Paged prefill and
decode use the same layout with the leading dimension interpreted as cache slots.

`examples/kda_training.py` builds these operations into a small trainable
`[B, T, hidden_size] -> [B, T, hidden_size]` attention module. To mirror the
main Kimi block structure, projected Q/K/V pass through a causal depthwise SiLU
convolution, the forget and output gates use two-stage factorized projections,
and the per-head output uses learned RMS normalization before sigmoid gating.
`--backend=reference` uses the PyTorch reference throughout. On Blackwell,
`--backend=fused` uses the same public boundary: the model produces per-token natural-log decay
and `chunk_kda` owns the BT64 scan. Implementations may inline that scan, but cumulative gates and
chunk boundaries are not caller-visible representations.

The example explicitly runs projections in BF16 while retaining FP32 parameters and gate
math; no ambient autocast context is required. Distributed mixed-precision policies must
preserve `A_log` and `dt_bias` as FP32 when parameters are materialized, rather than only
casting activations inside `forward`. A module-wide BF16 FSDP policy violates that
contract. A correctness-first integration can keep the KDA unit under an FP32 policy while
its projections explicitly compute in BF16; isolating only the strict-FP32 decay state is
a future bandwidth optimization.
The optimized core requires Blackwell and `head_dim=128`; its public boundary accepts
FP16, BF16, or FP32 inputs. Homogeneous FP16 and BF16 Q/K/V stay in their input dtype, while
FP32 or mixed-dtype inputs retain the existing BF16 normalization. The core chunks internally at
64 tokens. Complete `B=1` inputs whose length is a multiple of the chunk size run on the
direct dense route; other dense `[B, T, H, D]` inputs are lowered internally to
equal-length packed sequences, while `chunk_kda(..., cu_seqlens=offsets)` accepts
explicitly packed `[1, T, H, D]` inputs. All forms carry sequence boundaries through the
forward,
backward, and recurrent states; logical sequences may have tails or be empty. For
fixed-capacity execution, the terminal offset may be smaller than physical `T`; primitive
forward values outside `[0, cu_seqlens[-1])` are unspecified. The internal reverse scan
returns zero cotangents for inactive gate rows. This does not sanitize arbitrary
parameterized gate producers: callers still need the masking rules below because
``0 * NaN`` can poison their reductions.

!!! warning "FP16 intermediate range"

    Use L2-normalized Q/K with FP16, as the training example does. Unnormalized Q/K can easily
    produce attention or solve factors outside the FP16 range. Unusually large V or initial-state
    carries can likewise overflow the FP16 chunk-state and value intermediates. The GEMMs
    accumulate in FP32, but their results are converted back to FP16 when used as inputs to the
    next GEMM; an overflow at that conversion cannot be recovered by the next FP32 accumulator.
    BF16 uses the same storage size with a much larger exponent range, so it is substantially less
    likely to hit these issues, although sufficiently large values can overflow any finite dtype.

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

`KDAAttention.forward` passes explicit offsets to its short convolution and `chunk_kda`;
the selected implementation owns its sequence-local scan and scheduling. Set
`mask_inactive_capacity=True` only when the packed tensor reserves physical rows beyond
`cu_seqlens[-1]`; dense and exact-packed
callers leave it disabled and pay no masking cost. The optimized boundaries are
first-order and do not support higher-order autograd. Run
`python examples/kda_training.py --backend=fused --packed --batch-size=4 --tokens=256`
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
The bounded gate itself uses private CuTeDSL forward and backward operators. It can be
combined with `--profile`; compilation warmups run before the
trace starts. Like FLA's default training path, its backward recomputes the W/U,
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
from examples.kda_training import KDAAttention

attention = KDAAttention(hidden_size=512, num_heads=4, head_dim=128).cuda()
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
conversion and its sequence-local BT64 cumulative sum; `recurrent_kda` performs only the
conversion because recurrence consumes one token decay at a time. This keeps chunking out
of model code and lets callers switch execution modes without changing gate
representation. Custom producers should return finite, nonpositive values; the fused chunk
backend additionally requires approximately `[-5.914, 0]`; this tensor-value range is
not checked at runtime. The training example uses the Kimi-style FP32 transform
`lower_bound * sigmoid(exp(A_log) * (raw_gate.float() + dt_bias))`, but that model policy is not
part of the public KDA API.

Both cores select their implementation with `impl`: `"fused"` runs the optimized kernels
and enforces their constraints (the chunked core requires `head_dim=128` and Blackwell,
preserves homogeneous FP16/BF16 Q/K/V, normalizes FP32 or mixed inputs to BF16, and chunks at
64 tokens; the fused recurrent scan is inference-only), while `"reference"`
runs the eager FP32 oracle behind the identical packed contract on any hardware and head
dimension, and stays differentiable. There is no automatic fallback between the two, and
the chunk-versus-recurrent switch is caller policy (on B200 the scan wins below roughly
32 tokens per sequence).

`recurrent_kda_decode` is the serving-specific one-token path. It consumes
channel-major post-convolution QKV (`[Q for all heads | K for all heads | V for all
heads]`), raw gate and beta projections, and a paged state cache. Q/K normalization,
gate activation, beta sigmoid, recurrence, output, and state-cache update run in one
Triton kernel. Callers may provide a stable output buffer for allocation-free CUDA
Graph replay.

::: attn_gym.linear.chunk_kda

::: attn_gym.linear.recurrent_kda

::: attn_gym.linear.recurrent_kda_decode

::: attn_gym.linear.Impl
