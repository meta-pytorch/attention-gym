"""Schedule policy for the heavy ragged KDA kernels (output composition, W/U recompute).

These tests supply the SM count explicitly so they run without CUDA. Rationale:
``GridScheduler.resolve_flat``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from attn_gym.linear._delta_rule.triton import chunk_scheduler
from attn_gym.linear._delta_rule.triton.chunk_scheduler import (
    GridScheduler,
    RaggedChunkMetadata,
    ScheduleKind,
    ScheduleRequest,
    chunk_capacity,
)

SM100_SM_COUNT = 152
CHUNK_SIZE = 64
# Output composition flattens (head, 64-wide value tile) pairs; recompute flattens heads.
MANY_SEQUENCE_SHAPES = {
    "uniform_b8_h96_t8192_output": (96 * 2, [8192] * 8),
    "zipf_s8_h96_t11920_recompute": (96, [6148, 497, 46, 3832, 272, 1106, 3, 16]),
}


def resolve_on_sm100(monkeypatch, subtasks, lengths, request, *, auto_persistent):
    """Resolve the flat plan for one packed shape as an SM100 device would."""
    monkeypatch.setattr(chunk_scheduler, "_multiprocessor_count", lambda device: SM100_SM_COUNT)
    metadata = RaggedChunkMetadata(
        None, None, chunk_capacity(sum(lengths), len(lengths), CHUNK_SIZE), CHUNK_SIZE
    )
    return GridScheduler(metadata).resolve_flat(
        request, subtasks, "cuda", auto_persistent=auto_persistent
    )


@pytest.mark.parametrize("shape", sorted(MANY_SEQUENCE_SHAPES))
def test_heavy_ragged_kernels_stay_static_under_auto(monkeypatch, shape):
    subtasks, lengths = MANY_SEQUENCE_SHAPES[shape]
    generic = resolve_on_sm100(
        monkeypatch, subtasks, lengths, ScheduleRequest.AUTO, auto_persistent=True
    )
    # The generic wave rule alone would pick the slower persistent variant here.
    assert generic.kind is ScheduleKind.PERSISTENT
    static = resolve_on_sm100(
        monkeypatch, subtasks, lengths, ScheduleRequest.AUTO, auto_persistent=False
    )
    assert static == replace(generic, kind=ScheduleKind.STATIC)
    explicit = resolve_on_sm100(
        monkeypatch, subtasks, lengths, ScheduleRequest.PERSISTENT, auto_persistent=False
    )
    assert explicit.kind is ScheduleKind.PERSISTENT
