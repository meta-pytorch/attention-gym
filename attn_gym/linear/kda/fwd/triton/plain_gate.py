"""Compatibility imports for the shared delta-rule plain_gate."""

# ruff: noqa: F401 -- compatibility re-exports

from attn_gym.linear._delta_rule.triton.plain_gate import (
    DEFAULT_CHUNK_SIZE,
    LOG2_E,
    __all__,
    _plain_gate_scan_cuda,
    _plain_gate_scan_dense_kernel,
    _plain_gate_scan_ragged_kernel,
    chunk_capacity,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)
