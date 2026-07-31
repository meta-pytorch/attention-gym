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
