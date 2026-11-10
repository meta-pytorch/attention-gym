"""Temporary bitwise gate for the backward-intra work-stream port."""

from functools import wraps

import pytest
import torch

from attn_gym.linear.kda.bwd.cute import chunk_kda_bwd_intra as port
from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_intra_legacy import (
    legacy_compile_chunk_kda_bwd_intra,
)
from test.test_kda_ragged_bwd_intra import _inputs, _metadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="requires CUDA capability >= 8.0",
)


@wraps(legacy_compile_chunk_kda_bwd_intra)
def legacy_compile(*args, **kwargs):
    compiled = legacy_compile_chunk_kda_bwd_intra(*args, **kwargs)

    def launch(*inputs):
        *tensors, capacity, grid = inputs
        return compiled(*tensors, min(capacity, grid))

    return launch


def assert_bitwise(actual, expected):
    for a, b in zip(actual, expected, strict=True):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("heads", [16, 32, 64])
@pytest.mark.parametrize(
    "tokens,lengths,cap",
    [
        (64, None, 1),
        (320, None, 2),
        (130, [0, 1, 64, 0, 65], 16),
        (577, [0, 193, 0, 384], 3),
    ],
    ids=["dense-one", "dense-remainder", "ragged-empty-overcapture", "ragged-remainder"],
)
def test_bwd_intra_legacy_parity(monkeypatch, dtype, heads, tokens, lengths, cap):
    inputs = tuple(t.repeat_interleave(heads, dim=2) for t in _inputs(tokens, dtype))
    metadata = _metadata(lengths) if lengths is not None else None
    config = port.ChunkKdaBwdIntraConfig(cap)
    actual = port.chunk_kda_bwd_intra(*inputs, metadata, config=config)
    monkeypatch.setattr(port.ChunkKdaBwdIntraTunable, "compile", staticmethod(legacy_compile))
    expected = port.chunk_kda_bwd_intra(*inputs, metadata, config=config)
    assert legacy_compile_chunk_kda_bwd_intra.cache_info().currsize > 0
    assert_bitwise(actual, expected)


def test_bwd_intra_legacy_graph_replay(monkeypatch):
    inputs = _inputs(320)
    metadata = _metadata([64, 0, 64, 0, 192])
    config = port.ChunkKdaBwdIntraConfig(2)
    original_compile = port.ChunkKdaBwdIntraTunable.compile
    graphs, outputs = [], []
    for compiler in (original_compile, legacy_compile):
        monkeypatch.setattr(port.ChunkKdaBwdIntraTunable, "compile", staticmethod(compiler))
        port.chunk_kda_bwd_intra(*inputs, metadata, config=config)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = port.chunk_kda_bwd_intra(*inputs, metadata, config=config)
        graph.replay()
        graphs.append(graph)
        outputs.append(output)
    for lengths in ([65, 0, 63, 0, 192], [0, 0, 0, 0, 320], [0, 0, 0, 0, 128]):
        fresh = _metadata(lengths)
        metadata.cu_seqlens.copy_(fresh.cu_seqlens)
        metadata.chunk_offsets.copy_(fresh.chunk_offsets)
        assert metadata.capacity > int(metadata.chunk_offsets[-1])
        for graph in graphs:
            graph.replay()
        assert_bitwise(*outputs)
