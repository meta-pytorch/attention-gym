"""Focused ragged scheduling tests for fused KDA WY/dQKG backward."""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_wy_dqkg_fused import (
    chunk_kda_bwd_wy_dqkg,
)
from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    bwd_wy_dqkg_reference,
    cumulative_sequence_offsets,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the fused CuTe WY backward requires CUDA capability 10.0 or 10.3",
)


class StageInputs(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    v_new: torch.Tensor
    g: torch.Tensor
    beta: torch.Tensor
    A: torch.Tensor
    h: torch.Tensor
    do: torch.Tensor
    dh: torch.Tensor
    dv: torch.Tensor


def _inputs(tokens: int, capacity: int) -> StageInputs:
    torch.manual_seed(37)
    shape = (1, tokens, 1, 128)

    def bf16(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8

    return StageInputs(
        q=bf16(shape),
        k=bf16(shape),
        v=bf16(shape),
        v_new=bf16(shape),
        g=-torch.rand(shape, device="cuda"),
        beta=torch.rand(1, tokens, 1, device="cuda"),
        A=bf16((1, tokens, 1, 64)),
        h=bf16((1, capacity, 1, 128, 128)),
        do=bf16(shape),
        dh=bf16((1, capacity, 1, 128, 128)),
        dv=bf16(shape),
    )


def _run(inputs: StageInputs, metadata):
    return chunk_kda_bwd_wy_dqkg(*inputs, metadata)


def _sequence_local_reference(inputs: StageInputs, lengths: list[int]):
    output_parts: list[list[torch.Tensor]] = [[] for _ in range(6)]
    token_start = 0
    chunk_start = 0
    for length in lengths:
        chunks = (length + 63) // 64
        if length:
            token_slice = slice(token_start, token_start + length)
            chunk_slice = slice(chunk_start, chunk_start + chunks)
            local_inputs = StageInputs(
                q=inputs.q[:, token_slice].clone(),
                k=inputs.k[:, token_slice].clone(),
                v=inputs.v[:, token_slice].clone(),
                v_new=inputs.v_new[:, token_slice].clone(),
                g=inputs.g[:, token_slice].clone(),
                beta=inputs.beta[:, token_slice].clone(),
                A=inputs.A[:, token_slice].clone(),
                h=inputs.h[:, chunk_slice].clone(),
                do=inputs.do[:, token_slice].clone(),
                dh=inputs.dh[:, chunk_slice].clone(),
                dv=inputs.dv[:, token_slice].clone(),
            )
            metadata = prepare_ragged_chunk_metadata(
                cumulative_sequence_offsets([length]), length, 64
            )
            for parts, output in zip(output_parts, _run(local_inputs, metadata), strict=True):
                parts.append(output)
        token_start += length
        chunk_start += chunks
    return tuple(torch.cat(parts, dim=1) for parts in output_parts)


def test_ragged_wy_matches_independent_numerical_reference():
    lengths = [65, 0, 63]
    cu_seqlens = cumulative_sequence_offsets(lengths)
    chunk_routes = torch.tensor(
        [
            (sequence, chunk)
            for sequence, length in enumerate(lengths)
            for chunk in range((length + 63) // 64)
        ],
        device="cuda",
        dtype=torch.int64,
    )
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, sum(lengths), 64)
    inputs = _inputs(sum(lengths), metadata.capacity)
    actual = _run(inputs, metadata)
    reference_args = inputs[3:]
    scale = 128**-0.5
    golden = bwd_wy_dqkg_reference(
        inputs.q.double(),
        inputs.k.double(),
        inputs.v.double(),
        *(tensor.double() for tensor in reference_args),
        scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_routes,
    )
    reference = bwd_wy_dqkg_reference(
        inputs.q,
        inputs.k,
        inputs.v,
        *reference_args,
        scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_routes,
    )
    # The mathematical reference returns db before dg; the kernel returns dg before db.
    actual = (*actual[:3], actual[4], actual[3], actual[5])
    names = ("dq", "dk", "dv", "db", "dg", "dA")
    for name, output, golden_output, reference_output in zip(
        names, actual, golden, reference, strict=True
    ):
        assert_matches_low_precision_reference(output, golden_output, reference_output, name)


def test_ragged_wy_rejects_mismatched_chunk_size():
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets([64]), 64, 32)
    inputs = _inputs(64, metadata.capacity)

    with pytest.raises(ValueError, match="metadata chunk size"):
        _run(inputs, metadata)


def test_ragged_wy_matches_sequence_local_tails():
    lengths = [65, 0, 63]
    metadata = prepare_ragged_chunk_metadata(
        cumulative_sequence_offsets(lengths), sum(lengths), 64
    )
    inputs = _inputs(sum(lengths), metadata.capacity)

    actual = _run(inputs, metadata)
    expected = _sequence_local_reference(inputs, lengths)

    assert metadata.capacity == 4
    assert metadata.chunk_offsets.tolist() == [0, 2, 2, 3]
    for packed, local in zip(actual, expected, strict=True):
        torch.testing.assert_close(packed, local, rtol=0, atol=0)


def test_ragged_wy_ignores_poisoned_inactive_capacity():
    lengths = [64, 64]
    metadata = prepare_ragged_chunk_metadata(
        cumulative_sequence_offsets(lengths), sum(lengths), 64
    )
    inputs = _inputs(sum(lengths), metadata.capacity)
    poisoned_inputs = inputs._replace(h=inputs.h.clone(), dh=inputs.dh.clone())
    poisoned_inputs.h[:, 2] = torch.nan
    poisoned_inputs.dh[:, 2] = torch.nan

    expected = _run(inputs, metadata)
    actual = _run(poisoned_inputs, metadata)

    assert metadata.capacity == 3
    assert metadata.chunk_offsets.tolist() == [0, 1, 2]
    for poisoned, unpoisoned in zip(actual, expected, strict=True):
        assert torch.isfinite(poisoned).all()
        torch.testing.assert_close(poisoned, unpoisoned, rtol=0, atol=0)


@pytest.mark.parametrize("lengths", [[0], [0, 0, 0]])
def test_ragged_wy_accepts_all_empty_sequences(lengths):
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), 0, 64)
    outputs = _run(_inputs(0, metadata.capacity), metadata)

    for output in outputs:
        assert output.numel() == 0


def test_ragged_wy_masks_partial_chunk_columns():
    lengths = [61]
    metadata = prepare_ragged_chunk_metadata(
        cumulative_sequence_offsets(lengths), sum(lengths), 64
    )
    inputs = _inputs(sum(lengths), metadata.capacity)
    neutral_A = inputs.A.clone()
    neutral_A[..., 61:] = 0
    poisoned_A = neutral_A.clone()
    poisoned_A[..., 61:] = torch.nan

    expected = _run(inputs._replace(A=neutral_A), metadata)
    actual = _run(inputs._replace(A=poisoned_A), metadata)

    for masked, neutral in zip(actual, expected, strict=True):
        assert torch.isfinite(masked).all()
        torch.testing.assert_close(masked, neutral, rtol=0, atol=0)


def test_ragged_wy_cuda_graph_replay():
    tokens = 128
    cu_seqlens = cumulative_sequence_offsets([64, 64])
    inputs = _inputs(tokens, capacity=3)

    def operation(*args):
        *stage_tensors, offsets = args
        metadata = prepare_ragged_chunk_metadata(offsets, tokens, 64)
        return chunk_kda_bwd_wy_dqkg(*stage_tensors, metadata)

    operation(*inputs, cu_seqlens)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = operation(*inputs, cu_seqlens)

    cu_seqlens.copy_(cumulative_sequence_offsets([65, 63]))
    graph.replay()
    torch.cuda.synchronize()

    replay_expected = operation(*inputs, cu_seqlens)
    for captured, eager in zip(replayed, replay_expected, strict=True):
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
