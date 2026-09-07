"""Compatibility imports for the shared delta-rule chunk_schedule."""

# ruff: noqa: F401 -- compatibility re-exports

from attn_gym.linear._delta_rule.chunk_schedule import (
    RaggedChunkMetadata,
    ResolvedSchedule,
    ScheduleKind,
    ScheduleRequest,
    __all__,
    chunk_capacity,
    prepare_chunk_offsets_op,
    prepare_ragged_chunk_metadata,
    validate_schedule_request,
)
