"""Packed correctness tests for the KDA dAqk Triton stage."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_daqk import chunk_kda_bwd_daqk
from attn_gym.linear.kda.chunk_scheduler import ScheduleRequest, prepare_ragged_chunk_metadata
from attn_gym.testing import cumulative_sequence_offsets
from attn_gym.testing.kda import bwd_daqk_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def make_inputs(
    lengths: list[int],
    heads: int = 2,
    *,
    misaligned: bool = False,
    physical_tokens: int | None = None,
):
    """Build packed BF16 operands and their shared metadata."""
    torch.manual_seed(29)
    tokens = sum(lengths) if physical_tokens is None else physical_tokens
    shape = (1, tokens, heads, 128)
    if misaligned:
        value = torch.randn(tokens * heads * 128 + 1, device="cuda", dtype=torch.bfloat16)[
            1:
        ].view(shape)
        d_output = torch.randn(tokens * heads * 128 + 1, device="cuda", dtype=torch.bfloat16)[
            1:
        ].view(shape)
    else:
        value = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        d_output = torch.randn_like(value)
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), tokens, 64)
    return value, d_output, metadata


@pytest.mark.parametrize(
    "lengths",
    ([65, 0, 63], [1, 63, 64, 65, 127, 128, 129, 0]),
)
@pytest.mark.parametrize("misaligned", [False, True], ids=["tma", "pointer"])
def test_ragged_daqk_matches_reference(lengths, misaligned):
    value, d_output, metadata = make_inputs(lengths, misaligned=misaligned)
    scale = 128**-0.5

    actual = chunk_kda_bwd_daqk(
        value,
        d_output,
        scale,
        metadata=metadata,
        schedule=ScheduleRequest.STATIC,
    )
    expected = bwd_daqk_reference(value, d_output, lengths, scale)
    torch.testing.assert_close(actual, expected, rtol=4e-3, atol=4e-3)


def test_ragged_daqk_persistent_matches_static_over_capacity():
    lengths = [65, 0, 63]
    active_tokens = sum(lengths)
    value, d_output, metadata = make_inputs(
        lengths,
        heads=2,
        physical_tokens=16 * active_tokens,
    )
    assert metadata.capacity >= 8 * metadata.chunk_offsets[-1].item()
    kwargs = {"metadata": metadata, "scale": 128**-0.5}

    static = chunk_kda_bwd_daqk(value, d_output, schedule=ScheduleRequest.STATIC, **kwargs)
    persistent = chunk_kda_bwd_daqk(value, d_output, schedule=ScheduleRequest.PERSISTENT, **kwargs)
    torch.testing.assert_close(
        persistent[:, :active_tokens], static[:, :active_tokens], rtol=0, atol=0
    )


def test_ragged_daqk_forced_int64_matches_default(monkeypatch):
    import attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_daqk as daqk_module

    lengths = [65, 0, 63]
    value, d_output, metadata = make_inputs(lengths)
    expected = chunk_kda_bwd_daqk(value, d_output, 128**-0.5, metadata=metadata)
    calls = 0

    def force_int64(*_tensors):
        nonlocal calls
        calls += 1
        return True

    monkeypatch.setattr(daqk_module, "requires_int64_offsets", force_int64)
    actual = chunk_kda_bwd_daqk(value, d_output, 128**-0.5, metadata=metadata)
    assert calls > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ragged_daqk_handles_all_empty_sequences():
    value, d_output, metadata = make_inputs([0, 0])
    actual = chunk_kda_bwd_daqk(value, d_output, 128**-0.5, metadata=metadata)
    assert actual.shape == (1, 0, 2, 64)


def test_ragged_daqk_rejects_mismatched_chunk_size():
    value, d_output, metadata = make_inputs([128])
    with pytest.raises(ValueError, match="metadata chunk size"):
        chunk_kda_bwd_daqk(
            value,
            d_output,
            128**-0.5,
            chunk_size=32,
            metadata=metadata,
        )
