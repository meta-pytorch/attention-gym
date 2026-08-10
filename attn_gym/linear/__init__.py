"""Linear attention operations."""

from attn_gym.linear.gdn import GatedDeltaRuleOutput, gated_delta_rule
from attn_gym.linear.kda import (
    naive_chunk_kda,
    naive_recurrent_kda,
)

__all__ = [
    "GatedDeltaRuleOutput",
    "gated_delta_rule",
    "naive_chunk_kda",
    "naive_recurrent_kda",
]
