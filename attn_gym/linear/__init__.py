"""Linear attention primitives."""

from attn_gym.linear.gdn import (
    fused_recurrent_gated_delta_rule,
    naive_chunk_gated_delta_rule,
    naive_recurrent_gated_delta_rule,
)

__all__ = [
    "fused_recurrent_gated_delta_rule",
    "naive_chunk_gated_delta_rule",
    "naive_recurrent_gated_delta_rule",
]
