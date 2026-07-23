# Linear Attention

Attention Gym includes reference implementations for experimenting with linear-attention recurrences.

## Gated Delta Rule

`naive_recurrent_gated_delta_rule` is a direct token-by-token reference. `naive_chunk_gated_delta_rule` computes the same recurrence in chunks and supports sequence lengths that are not divisible by the chunk size. Both functions optionally accept an initial state and return the final state.

```python
from attn_gym.linear import naive_chunk_gated_delta_rule

output, final_state = naive_chunk_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    chunk_size=64,
    output_final_state=True,
)
```

::: attn_gym.linear.naive_recurrent_gated_delta_rule

::: attn_gym.linear.naive_chunk_gated_delta_rule

!!! warning "Reference implementation"
    `fused_recurrent_gated_delta_rule` is currently an explicit stub. It raises `NotImplementedError` until the optimized decode kernel is implemented.
