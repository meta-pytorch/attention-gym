"""Composition tests for the scheduler-routed KDA forward stages."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra import chunk_kda_fwd_intra

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTe KDA forward stages require CUDA capability 10.0 or newer",
)


def _metadata(lengths: list[int]):
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    return prepare_ragged_chunk_metadata(offsets, sum(lengths), 64)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_ragged_forward_stages_match_independent_sequences(dtype: torch.dtype):
    torch.manual_seed(13)
    lengths = [65, 0, 63]
    tokens = sum(lengths)
    shape = (1, tokens, 1, 128)
    q = torch.randn(shape, device="cuda", dtype=dtype) / 8
    k = torch.randn_like(q) / 8
    v = torch.randn_like(q) / 8
    gk = -torch.rand(shape, device="cuda")
    beta = torch.rand(1, tokens, 1, device="cuda")
    scale = 128**-0.5

    actual = chunk_kda_fwd_intra(q, k, v, gk, beta, scale, _metadata(lengths))
    expected_parts: list[list[torch.Tensor]] = [[] for _ in actual]
    begin = 0
    for length in lengths:
        if length == 0:
            continue
        end = begin + length
        sequence_outputs = chunk_kda_fwd_intra(
            q[:, begin:end],
            k[:, begin:end],
            v[:, begin:end],
            gk[:, begin:end],
            beta[:, begin:end].clone(),
            scale,
            _metadata([length]),
        )
        for parts, output in zip(expected_parts, sequence_outputs):
            parts.append(output)
        begin = end

    for packed, parts in zip(actual, expected_parts):
        expected = torch.cat(parts, dim=1)
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_ragged_forward_stages_replay_aligned_to_ragged(dtype: torch.dtype):
    torch.manual_seed(17)
    tokens = 128
    shape = (1, tokens, 1, 128)
    q = torch.randn(shape, device="cuda", dtype=dtype) / 8
    k = torch.randn_like(q) / 8
    v = torch.randn_like(q) / 8
    gk = -torch.rand(shape, device="cuda")
    beta = torch.rand(1, tokens, 1, device="cuda")
    scale = 128**-0.5
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)

    warm_metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)
    chunk_kda_fwd_intra(q, k, v, gk, beta, scale, warm_metadata)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)
        actual = chunk_kda_fwd_intra(q, k, v, gk, beta, scale, metadata)

    cu_seqlens.copy_(torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    expected = chunk_kda_fwd_intra(q, k, v, gk, beta, scale, _metadata([65, 63]))
    for captured, eager in zip(actual, expected):
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)
