"""Shared test utilities for Attention Gym implementations."""

from .gdn import make_gdn_test_inputs
from .kda import cumulative_sequence_offsets, strided_state_pool
from .profiling import (
    TraceFormat,
    annotate_kernels,
    kernel_stage,
    profile_trace,
    record_distributed_profile,
    record_function,
)

__all__ = [
    "TraceFormat",
    "annotate_kernels",
    "cumulative_sequence_offsets",
    "kernel_stage",
    "make_gdn_test_inputs",
    "profile_trace",
    "record_distributed_profile",
    "record_function",
    "strided_state_pool",
]
