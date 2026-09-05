"""CPU/meta checks for restoring the packed schedule from saved device tensors."""

from functools import partial

import pytest
import torch

from attn_gym.linear.kda import chunk_schedule
from attn_gym.linear.kda.chunk_schedule import RaggedChunkMetadata


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize(
    ("boundaries", "offsets", "tokens", "chunk_size", "capacity"),
    [
        pytest.param([0, 128], [0, 2], 128, 64, 2, id="complete-chunks"),
        pytest.param([0, 65, 65, 128], [0, 2, 2, 3], 256, 64, 6, id="empty-and-slack"),
        pytest.param([0, 0, 0], [0, 0, 0], 256, 64, 5, id="all-empty-with-capacity"),
        pytest.param([0, 0, 0], [0, 0, 0], 0, 64, 0, id="zero-capacity"),
        pytest.param([0, 5, 5, 12], [0, 2, 2, 4], 12, 4, 5, id="different-chunk-size"),
    ],
)
def test_restore_metadata_preserves_offsets_and_physical_capacity(
    monkeypatch, device, boundaries, offsets, tokens, chunk_size, capacity
):
    """Restoration preserves tensor identity and needs neither data reads nor a scheduler launch."""
    monkeypatch.setattr(
        chunk_schedule,
        "prepare_chunk_offsets_op",
        partial(pytest.fail, "restoring saved offsets must not run the scheduler"),
    )
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=device)
    chunk_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)

    metadata = RaggedChunkMetadata.from_offsets(cu_seqlens, chunk_offsets, tokens, chunk_size)

    assert metadata.cu_seqlens is cu_seqlens
    assert metadata.chunk_offsets is chunk_offsets
    assert metadata.capacity == capacity
    assert metadata.chunk_size == chunk_size


@pytest.mark.parametrize(
    ("tokens", "sequences", "chunk_size", "message"),
    [(-1, 1, 64, "tokens"), (128, 0, 64, "num_sequences"), (128, 1, 0, "chunk_size")],
)
def test_restore_metadata_preserves_capacity_validation(tokens, sequences, chunk_size, message):
    offsets = torch.empty(sequences + 1, dtype=torch.int32, device="meta")
    with pytest.raises(ValueError, match=message):
        RaggedChunkMetadata.from_offsets(offsets, offsets, tokens, chunk_size)
