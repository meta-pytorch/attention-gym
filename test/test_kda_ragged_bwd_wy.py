"""Focused ragged scheduling tests for fused KDA WY/dQKG backward."""

from __future__ import annotations

from typing import NamedTuple

import pytest
import torch

from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_wy_dqkg_fused import (
    _chunk_kda_bwd_wy_dqkg_custom_op,
    chunk_kda_bwd_wy_dqkg,
)
from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.utils import ChunkMetadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the fused CuTe WY backward requires CUDA capability 10.0 or newer",
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


def _offsets(lengths: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )


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
            metadata = prepare_ragged_chunk_metadata(_offsets([length]), length, 64)
            for parts, output in zip(output_parts, _run(local_inputs, metadata), strict=True):
                parts.append(output)
        token_start += length
        chunk_start += chunks
    return tuple(torch.cat(parts, dim=1) for parts in output_parts)


def test_ragged_wy_matches_sequence_local_tails():
    lengths = [65, 0, 63]
    metadata = prepare_ragged_chunk_metadata(_offsets(lengths), sum(lengths), 64)
    inputs = _inputs(sum(lengths), metadata.capacity)

    actual = _run(inputs, metadata)
    expected = _sequence_local_reference(inputs, lengths)

    assert metadata.capacity == 4
    assert metadata.chunk_offsets.tolist() == [0, 2, 2, 3]
    for packed, local in zip(actual, expected, strict=True):
        torch.testing.assert_close(packed, local, rtol=0, atol=0)


def test_ragged_wy_preserves_legacy_map_and_ignores_inactive_capacity():
    lengths = [64, 64]
    cu_seqlens = _offsets(lengths)
    ragged = prepare_ragged_chunk_metadata(cu_seqlens, sum(lengths), 64)
    inputs = _inputs(sum(lengths), ragged.capacity)
    inputs.h[:, 2] = torch.nan
    inputs.dh[:, 2] = torch.nan
    legacy = ChunkMetadata(
        cu_seqlens,
        torch.tensor([[0, 0], [1, 0]], device="cuda", dtype=torch.int32),
        torch.tensor(2, device="cuda", dtype=torch.int32),
    )

    actual = _run(inputs, ragged)
    expected = _run(inputs._replace(h=inputs.h[:, :2], dh=inputs.dh[:, :2]), legacy)

    assert ragged.capacity == 3
    assert ragged.chunk_offsets.tolist() == [0, 1, 2]
    for scheduled, mapped in zip(actual, expected, strict=True):
        assert torch.isfinite(scheduled).all()
        torch.testing.assert_close(scheduled, mapped, rtol=0, atol=0)


def test_ragged_wy_fullgraph_and_cuda_graph_replay():
    tokens = 128
    cu_seqlens = _offsets([64, 64])
    inputs = _inputs(tokens, capacity=3)

    def operation(*args):
        *stage_tensors, offsets = args
        metadata = prepare_ragged_chunk_metadata(offsets, tokens, 64)
        return chunk_kda_bwd_wy_dqkg(*stage_tensors, metadata)

    metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)
    torch.library.opcheck(
        _chunk_kda_bwd_wy_dqkg_custom_op,
        (
            *inputs,
            metadata.cu_seqlens,
            metadata.chunk_offsets,
            64,
            False,
            1,
            True,
        ),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
    )

    expected = operation(*inputs, cu_seqlens)
    compiled = torch.compile(operation, fullgraph=True)
    actual = compiled(*inputs, cu_seqlens)
    for compiled_output, eager_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(compiled_output, eager_output, rtol=0, atol=0)

    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = operation(*inputs, cu_seqlens)

    cu_seqlens.copy_(_offsets([65, 63]))
    graph.replay()
    torch.cuda.synchronize()

    replay_expected = operation(*inputs, cu_seqlens)
    for captured, eager in zip(replayed, replay_expected, strict=True):
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
