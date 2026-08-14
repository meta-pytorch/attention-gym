# Linear Attention

Attention Gym provides functional linear-attention operators with eager reference implementations.

## Gated Delta Rule

`gated_delta_rule` uses the SDPA layout `[batch, heads, sequence, dimension]` and returns a structured result. Its eager backend supports recurrent execution for decoding and chunked execution for training or prefill.

```python
from attn_gym.linear import gated_delta_rule

result = gated_delta_rule(
    query,
    key,
    value,
    gate,
    beta,
    mode="chunked",
    backend="eager",
    return_final_state=True,
)

output = result.output
final_state = result.final_state
```

`mode="auto"` selects recurrent execution for a one-token sequence and chunked execution otherwise. `backend="auto"` currently selects the eager reference implementation.

### Supported capabilities

- Fixed-length inputs in `[batch, heads, sequence, dimension]` layout.
- Recurrent and chunked execution, including initial and final recurrent state.
- CPU and CUDA execution through eager PyTorch operations.
- Autograd for inputs and initial state.

Packed variable-length inputs and optimized backends are not implemented yet. Requesting an unsupported backend fails explicitly rather than changing execution semantics.

### Migration from the prototype API

The earlier prototype functions were replaced by the variant-level API:

- `naive_recurrent_gated_delta_rule(...)` becomes `gated_delta_rule(..., mode="recurrent")`.
- `naive_chunk_gated_delta_rule(...)` becomes `gated_delta_rule(..., mode="chunked")`.
- Inputs now use `[batch, heads, sequence, dimension]`, matching SDPA and FlexAttention.
- `output_final_state` was renamed to `return_final_state`.
- The return value is always `GatedDeltaRuleOutput`; access tensors through `.output` and `.final_state`.
- The unimplemented fused recurrent stub was removed. Optimized backends will be selected through `backend=` when they are added.

::: attn_gym.linear.gated_delta_rule

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
instead uses a graph-safe ragged FP32 reverse cumsum followed by the pointwise Triton
derivative and FP32 parameter reductions. The example explicitly runs projections in BF16
while retaining FP32 parameters and gate reductions; no ambient autocast context is
required.
The CuTe gate path requires a head dimension divisible by 32 in `[32, 1024]`.
The optimized core requires Blackwell, BF16 kernel inputs, `head_dim=128`, and 64-token
chunks. Dense `[B, T, H, D]` inputs are lowered internally to equal-length packed
sequences, while `chunk_kda(..., cu_seqlens=offsets)` accepts explicitly packed
`[1, T, H, D]` inputs. Both forms carry sequence boundaries through the forward,
backward, and recurrent states; logical sequences may have tails or be empty.
`KDAAttention.forward` threads explicit offsets through its short convolution,
bounded-gate prefix sum, and KDA core. The optimized boundaries are first-order and do
not support higher-order autograd. Run
`python examples/kda_training.py --backend=fused --packed --batch-size=4 --tokens=256`
to sample token-level lengths from a truncated Zipf distribution, pack them into one
physical batch, print their `cu_seqlens`, and pass those offsets through the complete
training step. Add `--padded` when chunk-aligned samples are needed. The complete
composed core forward and backward use private custom
operators with fake-tensor registrations and first-order autograd wrappers, so the public
`chunk_kda` operation supports strict `torch.compile(fullgraph=True)` and CUDA Graph
capture for fixed token and sequence counts; changing either shape requires recompilation
or recapture. Pass `--compile` to the training example to compile the complete module as
one full graph. This keeps
the custom KDA core behind its registered operator boundary while allowing
Inductor to fuse the surrounding PyTorch normalization, gating, and pointwise
work. It can be combined with `--profile`; compilation warmups run before the
trace starts. Like FLA's default training path, its backward recomputes the W/U,
gated Q/K, recurrent-state, and
corrected-value intermediates instead of retaining them across the forward/backward
boundary. Million-token training still requires model-level activation
checkpointing and context parallelism; this single-device example implements
neither distributed policy.

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

For integrations that fuse gate activation with its forward chunk prefix sum,
`naive_chunk_kda_from_cumulative` exposes the matching reference boundary. Its
cumulative log2 gate is inclusive and resets at the same `chunk_size` passed to
KDA; the function does not perform a second cumulative sum. The composed `chunk_kda` backend consumes this same representation. That
trainable operator remains intentionally narrow: physical batch size one, complete
64-token chunks within every fixed or packed sequence, BF16 kernel operands,
`head_dim=128`, `chunk_size=64`, and Blackwell.

::: attn_gym.linear.naive_chunk_kda

::: attn_gym.linear.naive_chunk_kda_from_cumulative

::: attn_gym.linear.naive_recurrent_kda
