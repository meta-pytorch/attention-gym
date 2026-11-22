"""Shared test utilities for Attention Gym implementations."""

from .gdn import make_gdn_test_inputs
from .kda import (
    CHECKPOINT_SENTINEL,
    assert_checkpoints_into_cache_match,
    cumulative_sequence_offsets,
    sentinel_state_checkpoints,
    strided_state_pool,
)
from .profiling import (
    TraceFormat,
    annotate_kernels,
    kernel_stage,
    profile_trace,
    record_distributed_profile,
    record_function,
)

__all__ = [
    "CHECKPOINT_SENTINEL",
    "TraceFormat",
    "annotate_kernels",
    "assert_checkpoints_into_cache_match",
    "cumulative_sequence_offsets",
    "kernel_stage",
    "make_gdn_test_inputs",
    "profile_trace",
    "record_distributed_profile",
    "record_function",
    "sentinel_state_checkpoints",
    "strided_state_pool",
]
