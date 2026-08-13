"""Integration tests for ragged scheduling in the KDA output-composition stage."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o import chunk_gla_fwd_o_gk

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the KDA output kernel requires CUDA",
)


def _reference(
    q: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    A: torch.Tensor,
    h: torch.Tensor,
    lengths: list[int],
    scale: float,
) -> torch.Tensor:
    output = torch.empty_like(v)
    token_start = 0
    global_chunk = 0
    for length in lengths:
        sequence_end = token_start + length
        for chunk_start in range(token_start, sequence_end, 64):
            valid = min(64, sequence_end - chunk_start)
            q_tile = q[0, chunk_start : chunk_start + valid].float()
            v_tile = v[0, chunk_start : chunk_start + valid].float()
            g_tile = g[0, chunk_start : chunk_start + valid].float()
            A_tile = A[0, chunk_start : chunk_start + valid, :, :valid].float()
            h_tile = h[0, global_chunk].float()
            inter = torch.einsum("thk,hkv->thv", q_tile * torch.exp2(g_tile), h_tile)
            intra = torch.einsum("hts,shv->thv", A_tile.permute(1, 0, 2).tril(), v_tile)
            output[0, chunk_start : chunk_start + valid] = (inter * scale + intra).to(output.dtype)
            global_chunk += 1
        token_start = sequence_end
    return output


@pytest.mark.parametrize("lengths", [[65, 63], [1, 64, 0, 65]])
def test_ragged_output_composition_matches_reference(lengths):
    torch.manual_seed(0)
    tokens = sum(lengths)
    heads, key_dim, value_dim = 2, 32, 32
    device = "cuda"
    dtype = torch.bfloat16
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device=device,
        dtype=torch.int32,
    )
    metadata = prepare_ragged_chunk_metadata(offsets, tokens, 64)

    q = torch.randn(1, tokens, heads, key_dim, device=device, dtype=dtype) / 8
    v = torch.randn(1, tokens, heads, value_dim, device=device, dtype=dtype) / 8
    g = -torch.rand(1, tokens, heads, key_dim, device=device, dtype=torch.float32)
    A = torch.randn(1, tokens, heads, 64, device=device, dtype=dtype) / 8
    h = (
        torch.randn(
            1,
            metadata.capacity,
            heads,
            key_dim,
            value_dim,
            device=device,
            dtype=dtype,
        )
        / 8
    )
    scale = 0.125

    expected = _reference(q, v, g, A, h, lengths, scale)
    actual = chunk_gla_fwd_o_gk(q, v, g, A, h, scale, metadata=metadata)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_ragged_output_routes_full_chunks_through_tma_and_masks_tails():
    torch.manual_seed(2)
    lengths = [65, 63]
    tokens = sum(lengths)
    heads = 1
    offsets = torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(offsets, tokens, 64)
    q = torch.randn(1, tokens, heads, 128, device="cuda", dtype=torch.bfloat16) / 8
    v = torch.randn_like(q) / 8
    g = -torch.rand(1, tokens, heads, 128, device="cuda")
    A = torch.randn(1, tokens, heads, 64, device="cuda", dtype=torch.bfloat16) / 8
    h = torch.randn(1, 3, heads, 128, 128, device="cuda", dtype=torch.bfloat16) / 8
    scale = 0.125

    expected = _reference(q, v, g, A, h, lengths, scale)
    actual = chunk_gla_fwd_o_gk(q, v, g, A, h, scale, metadata=metadata)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


def test_ragged_output_replays_aligned_to_ragged():
    torch.manual_seed(1)
    tokens = 128
    heads, key_dim, value_dim = 1, 128, 128
    dtype = torch.bfloat16
    cu_seqlens = torch.tensor([0, 64, 128], device="cuda", dtype=torch.int32)
    q = torch.randn(1, tokens, heads, key_dim, device="cuda", dtype=dtype) / 8
    v = torch.randn(1, tokens, heads, value_dim, device="cuda", dtype=dtype) / 8
    g = -torch.rand(1, tokens, heads, key_dim, device="cuda")
    A = torch.randn(1, tokens, heads, 64, device="cuda", dtype=dtype) / 8
    # Static capacity is three: aligned replay uses two CTAs; ragged replay uses three.
    h = torch.randn(1, 3, heads, key_dim, value_dim, device="cuda", dtype=dtype) / 8
    scale = 0.125

    warm_metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)
    chunk_gla_fwd_o_gk(q, v, g, A, h, scale, metadata=warm_metadata)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)
        actual = chunk_gla_fwd_o_gk(q, v, g, A, h, scale, metadata=metadata)

    cu_seqlens.copy_(torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference(q, v, g, A, h, [65, 63], scale)
    assert metadata.chunk_offsets.tolist() == [0, 2, 3]
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
