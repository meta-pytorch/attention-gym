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
`[B, T, hidden_size] -> [B, T, hidden_size]` attention module. Its default path
uses PyTorch autograd. On CUDA capability 9.0 or newer,
`--gate-backward=cute` demonstrates using the fused CuTeDSL bounded-gate and
reverse-cumsum backward leaf while keeping the rest of KDA as an inspectable
reference implementation. The low-level CuTe leaf consumes the complete
`[B, T, H, D]` gate tensor in one launch and emits per-batch, per-chunk
partials for the shared parameter reduction. The example explicitly runs
projections in BF16 while retaining FP32 parameters and gate reductions; no
ambient autocast context is required.
The CuTe path also requires a head dimension divisible by 32 in
`[32, 1024]`. It is a first-order backward leaf and does not support
higher-order autograd.

The module can sit behind a transformer layer's attention slot while state is
threaded explicitly:

```python
from examples.kda_training import KDAAttention

attention = KDAAttention(hidden_size=512, num_heads=4, head_dim=128).cuda()
result = attention(hidden_states, initial_state, return_final_state=True)
hidden_states, recurrent_state = result
```

The example intentionally is not checkpoint-compatible with Kimi K3. A model
adapter must still provide K3's short convolution, factorized projections,
exact gated RMSNorm, checkpoint layout, and distributed execution policy.

For integrations that fuse gate activation with its forward chunk prefix sum,
`naive_chunk_kda_from_cumulative` exposes the matching reference boundary. Its
cumulative log2 gate is inclusive and resets at the same `chunk_size` passed to
KDA; the function does not perform a second cumulative sum. The existing
`chunk_kda_fwd_intra` kernel consumes this same representation, but currently
returns pipeline intermediates and is restricted to its production
specialization. It is not yet composed with the state/output kernels and full
backward into a public trainable operator.

::: attn_gym.linear.naive_chunk_kda

::: attn_gym.linear.naive_chunk_kda_from_cumulative

::: attn_gym.linear.naive_recurrent_kda
