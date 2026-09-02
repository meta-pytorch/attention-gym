"""Validate active writes in the portable packed KDA backward stages."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_delta_h_triton import (
    chunk_kda_bwd_delta_h_triton,
)
from attn_gym.linear.kda.bwd.triton.chunk_kda_bwd_wy_triton import chunk_kda_bwd_wy_triton
from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.testing import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the portable KDA backward requires CUDA capability 8.0 or newer",
)


@contextmanager
def _poison_uninitialized_memory():
    """Fill fresh empty tensors with NaNs so active read-before-write becomes visible."""
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        yield
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous_fill
        torch.use_deterministic_algorithms(
            previous_deterministic,
            warn_only=previous_warn_only,
        )


def _bf16(shape: tuple[int, ...]) -> torch.Tensor:
    """Create one nontrivial low-precision CUDA tensor."""
    return torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8


def test_portable_ragged_backward_fully_overwrites_active_outputs():
    """Keep active delta-H and WY results independent of fresh output storage."""
    torch.manual_seed(37)
    tokens, heads = 128, 1
    lengths = [17, 15, 33, 0]
    active_tokens = sum(lengths)
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), tokens, 64)
    active_chunks = sum((length + 63) // 64 for length in lengths)
    shape = (1, tokens, heads, 128)

    q = _bf16(shape)
    k = _bf16(shape)
    v = _bf16(shape)
    v_new = _bf16(shape)
    w = _bf16(shape)
    d_output = _bf16(shape)
    incoming_dv = _bf16(shape)
    gate = -torch.rand(shape, device="cuda")
    beta = torch.rand(1, tokens, heads, device="cuda")
    aqk = _bf16((1, tokens, heads, 64))
    inverse = _bf16((1, tokens, heads, 64))
    h = _bf16((1, metadata.capacity, heads, 128, 128))
    incoming_dh = _bf16((1, metadata.capacity, heads, 128, 128))

    for tensor in (q, k, v, v_new, w, d_output, incoming_dv, gate, aqk, inverse):
        tensor[:, active_tokens:].fill_(float("nan"))
    beta[:, active_tokens:].fill_(float("nan"))
    h[:, active_chunks:].fill_(float("nan"))
    incoming_dh[:, active_chunks:].fill_(float("nan"))

    delta_args = (q, k, w, d_output, aqk)
    delta_kwargs = {
        "gk": gate,
        "initial_state": None,
        "d_final_state": None,
        "scale": 128**-0.5,
        "metadata": metadata,
    }
    expected_dh, expected_initial, expected_dv = chunk_kda_bwd_delta_h_triton(
        *delta_args, **delta_kwargs
    )
    with _poison_uninitialized_memory():
        actual_dh, actual_initial, actual_dv = chunk_kda_bwd_delta_h_triton(
            *delta_args, **delta_kwargs
        )
    assert expected_initial is actual_initial is None
    torch.testing.assert_close(
        actual_dh[:, :active_chunks], expected_dh[:, :active_chunks], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_dv[:, :active_tokens], expected_dv[:, :active_tokens], rtol=0, atol=0
    )

    wy_args = (
        q,
        k,
        v,
        v_new,
        gate,
        beta,
        inverse,
        h,
        d_output,
        incoming_dh,
        incoming_dv,
        metadata,
    )
    expected = chunk_kda_bwd_wy_triton(*wy_args, scale=128**-0.5)
    with _poison_uninitialized_memory():
        actual = chunk_kda_bwd_wy_triton(*wy_args, scale=128**-0.5)
    for actual_output, expected_output in zip(actual, expected, strict=True):
        active_actual = actual_output[:, :active_tokens]
        active_expected = expected_output[:, :active_tokens]
        assert torch.isfinite(active_actual).all()
        torch.testing.assert_close(active_actual, active_expected, rtol=0, atol=0)
