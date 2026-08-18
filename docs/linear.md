# Linear Attention

Attention Gym provides functional linear-attention operators with eager reference implementations.

## Gated Delta Rule

`chunk_gdn` and `recurrent_gdn` use the SDPA layout
`[batch, heads, sequence, dimension]` and return a structured result. For each token, the scalar
natural-log gate decays the previous state before the delta update, and the query reads the updated
state:

```text
decayed_state = exp(gate) * state
residual = beta * (value - key @ decayed_state)
state = decayed_state + outer(key, residual)
output = scale * query @ state
```

`chunk_gdn` uses a chunk-parallel decomposition for training and prefill. `recurrent_gdn` consumes
tokens in order for decoding, inference prefill, and state-carrying correctness checks. The caller
chooses the execution form explicitly; `impl="reference"` selects the eager PyTorch implementation.

```python
from attn_gym.linear import chunk_gdn

result = chunk_gdn(
    query,
    key,
    value,
    gate,
    beta,
    impl="reference",
    return_final_state=True,
)

output = result.output
final_state = result.final_state
```

### Supported capabilities

- Fixed-length inputs in `[batch, heads, sequence, dimension]` layout.
- Separate recurrent and chunked operations with explicit initial and final state.
- CPU and CUDA execution through eager PyTorch operations.
- Autograd for inputs and initial state.
- Q/K/V share one dtype. FP16 and BF16 inputs use FP32 recurrence math and state while returning
  output in the Q dtype.
- Gate and beta may use independent floating dtypes and are converted to the recurrence compute
  dtype. A provided initial state uses FP32 for low-precision QKV.
- FP64 reference inputs retain FP64 recurrence math and state.

Packed variable-length inputs and fused implementations are not implemented yet. Explicit
`impl="fused"` calls fail rather than falling back to the reference.

### Migration from the prototype API

- `naive_recurrent_gated_delta_rule(...)` and `gated_delta_rule(..., mode="recurrent")` become
  `recurrent_gdn(...)`.
- `naive_chunk_gated_delta_rule(...)` and `gated_delta_rule(..., mode="chunked")` become
  `chunk_gdn(...)`.
- Inputs use `[batch, heads, sequence, dimension]`, matching SDPA and FlexAttention.
- `output_final_state` is named `return_final_state`.
- Both operations return `GatedDeltaRuleOutput`; access tensors through `.output` and
  `.final_state`.

::: attn_gym.linear.chunk_gdn

::: attn_gym.linear.recurrent_gdn

::: attn_gym.linear.GatedDeltaRuleOutput

## Kimi Delta Attention

The KDA references use token-major tensors: query, key, and per-channel gate are
`[batch, sequence, heads, key_dimension]`; value is
`[batch, sequence, heads, value_dimension]`; beta is
`[batch, sequence, heads]`. Both recurrent and chunked forms support ordinary
PyTorch autograd and an optional recurrent state.

`examples/kda_training.py` builds these operations into a small trainable
`[B, T, hidden_size] -> [B, T, hidden_size]` attention module. To mirror the
main Kimi block structure, projected Q/K/V pass through a causal depthwise SiLU
convolution, the forget and output gates use two-stage factorized projections,
and the per-head output uses learned RMS normalization before sigmoid gating.
`--backend=reference` uses the PyTorch reference throughout. On Blackwell,
`--backend=fused` selects the composed CuTeDSL/Triton forward and first-order
backward, including optimized Q/K normalization, bounded-gate activation and
prefix sum, intra-chunk solves, inter-chunk state recurrence, output
composition, and bounded-gate backward.
The kernel boundaries are labeled explicitly in profiler traces. Dense gate backward
consumes the complete `[B, T, H, D]` gate tensor in one CuTe launch and emits
per-batch, per-chunk partials for the shared parameter reduction. Packed gate backward
instead uses one graph-safe ragged Triton launch that fuses the reverse scan, bounded-gate
derivative, and FP32 parameter-gradient partials. The example explicitly runs projections
in BF16 while retaining FP32 parameters and gate reductions; no ambient autocast context
is required. Distributed mixed-precision policies must preserve `A_log` and `dt_bias` as
FP32 when parameters are materialized, rather than only casting activations inside
`forward`. A module-wide BF16 FSDP policy violates the fused gate ABI. A correctness-first
integration can keep the KDA unit under an FP32 policy while its projections explicitly
compute in BF16; isolating only the strict-FP32 decay state is a future bandwidth
optimization.
The CuTe dense gate backward requires a head dimension divisible by 32 in `[32, 1024]`;
gate forward is Triton for both dense and ragged inputs.
The optimized core requires Blackwell, BF16 kernel inputs, `head_dim=128`, and 64-token
chunks. Complete `B=1` inputs whose length is a multiple of the chunk size run on the
direct dense route; other dense `[B, T, H, D]` inputs are lowered internally to
equal-length packed sequences, while `chunk_kda(..., cu_seqlens=offsets)` accepts
explicitly packed `[1, T, H, D]` inputs. All forms carry sequence boundaries through the
forward,
backward, and recurrent states; logical sequences may have tails or be empty. For
fixed-capacity execution, the terminal offset may be smaller than physical `T`; primitive
values outside `[0, cu_seqlens[-1])` are unspecified. Primitives deliberately do not mask
this inactive suffix automatically; caller-owned masking keeps that cost out of each
primitive hot path.

A captured graph with sequence capacity `N` keeps `cu_seqlens.shape == (N + 1,)`. If a
replay has `M <= N` nonempty sequences and `L <= T` active tokens, repeat the terminal
endpoint through the unused tail:

```text
[0, sequence_start_1, ..., L, L, ..., L]
```

The repeated ranges are ordinary empty sequences. Stateful APIs therefore retain `N`
state rows even when only `M` sequences are nonempty. Both `L` and `M` may change on
replay, but physical token capacity `T` and metadata capacity `N` may not.

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

`mask_inactive_token_gradients(x, active_mask)` broadcasts the predicate as
`[1, T, 1, ...]` and evaluates `torch.where(mask, x, x.detach())`. It preserves forward
values but materializes an elementwise result; its purpose is masked automatic-
differentiation semantics, not a free forward identity. Automatic-differentiation paths
are zero on inactive rows, and subsequent derivatives inherit the same mask. Recurrent
and convolution states have one row per logical sequence rather than one row per physical
token, so token masks must not be applied to them.

::: attn_gym.linear.kda.active_token_mask

::: attn_gym.linear.kda.mask_inactive_tokens

::: attn_gym.linear.kda.mask_inactive_token_gradients

`KDAAttention.forward` passes explicit offsets to its short convolution and prepares one
private ragged chunk schedule shared by the bounded-gate prefix sum and KDA core. Set
`mask_inactive_capacity=True` only when the packed tensor reserves physical rows beyond
`cu_seqlens[-1]`; dense and exact-packed
callers leave it disabled and pay no masking cost. The optimized boundaries are
first-order and do not support higher-order autograd. Run
`python examples/kda_training.py --backend=fused --packed --batch-size=4 --tokens=256`
to sample token-level lengths from a truncated Zipf distribution, pack them exactly
into one physical batch, print their `cu_seqlens`, and pass those offsets through the
complete training step. Add `--padded` when chunk-aligned samples are needed. The
complete composed core forward and backward use private custom
operators with fake-tensor registrations and first-order autograd wrappers, so fused
`chunk_kda` supports strict `torch.compile(fullgraph=True)` and CUDA Graph capture for
fixed physical token capacity and sequence count. Packed reference execution is
eager-only: it reads `cu_seqlens` on the host to run each logical sequence
independently. Boundary values and the active token count may change on replay;
changing the physical token capacity or sequence count
requires recompilation or recapture. Pass `--compile` to compile the complete example as
one full graph. This keeps the custom KDA core behind its registered operator boundary
while allowing
Inductor to fuse the surrounding PyTorch normalization, gating, and pointwise
work. It can be combined with `--profile`; compilation warmups run before the
trace starts. Like FLA's default training path, its backward recomputes the W/U,
gated Q/K, recurrent-state, and
corrected-value intermediates instead of retaining them across the forward/backward
boundary.

Graph-safe active-token replay does not by itself make complete model time proportional
to `L`. The ragged short convolution, gate scan, and KDA core can skip inactive token
work, but the example's projections, output normalization, output gate, and output
projection still process physical capacity `T`. An end-to-end integration needs
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

The two public KDA cores are `chunk_kda` (training and prefill; consumes the
chunk-local inclusive cumulative log2 gate from `bounded_gate_cumsum(chunk_size=64)`
without a second cumulative sum) and `recurrent_kda` (decode and inference prefill;
consumes the per-token log2 gate from `bounded_gate_cumsum(chunk_size=1)`). Both
select their implementation with `impl`: `"fused"` runs the optimized kernels and
enforces their constraints (the chunked core needs BF16 operands, `head_dim=128`,
`chunk_size=64`, and Blackwell; the fused scan is inference-only), while
`"reference"` runs the eager FP32 oracle behind the identical packed contract on
any hardware and head dimension, and stays differentiable. There is no automatic
fallback between the two, and the chunk-versus-recurrent switch is caller policy
(on B200 the scan wins below roughly 32 tokens per sequence).

The serving limitations listed under `recurrent_kda` below are deliberate and
the contract is otherwise stable to build against; CUDA-graph capture amortizes
the multi-launch decode step.

::: attn_gym.linear.chunk_kda

::: attn_gym.linear.recurrent_kda

::: attn_gym.linear.Impl
