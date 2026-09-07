"""Compatibility imports for the shared delta-rule chunk_scheduler."""

# ruff: noqa: F401 -- compatibility re-exports

from attn_gym.linear._delta_rule.triton.chunk_scheduler import (
    PERSISTENT_AUTO_WAVES,
    PERSISTENT_CTAS_PER_SM,
    ChunkWork,
    GridScheduler,
    RaggedChunkMetadata,
    ResolvedSchedule,
    ScheduleKind,
    ScheduleRequest,
    __all__,
    _decode_ragged_chunk_work_kernel,
    _multiprocessor_count,
    _multiprocessor_count_for_index,
    _prepare_ragged_chunk_offsets,
    _prepare_ragged_chunk_offsets_kernel,
    chunk_capacity,
    chunk_work_oracle,
    decode_ragged_chunk_work,
    decode_ragged_task,
    load_ragged_chunk_count,
    load_ragged_chunk_work,
    load_ragged_sequence_extent,
    load_ragged_sequence_work,
    load_ragged_task_count,
    prepare_ragged_chunk_metadata,
    validate_schedule_request,
)
