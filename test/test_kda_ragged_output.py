"""Integration tests for ragged scheduling in the KDA output-composition stage."""

from __future__ import annotations

import pytest
import torch

import attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o as output_module
from attn_gym.linear.kda.chunk_scheduler import ScheduleRequest, prepare_ragged_chunk_metadata

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
    actual = output_module.chunk_gla_fwd_o_gk(q, v, g, A, h, scale, metadata=metadata)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_ragged_output_routes_full_chunks_through_tma_and_masks_tails(monkeypatch):
    torch.manual_seed(2)
    launch_count = 0
    tma_kernel = output_module.chunk_gla_fwd_kernel_o_ragged_tma

    class RecordingKernel:
        def __getitem__(self, grid):
            launch = tma_kernel[grid]

            def record_launch(*args, **kwargs):
                nonlocal launch_count
                launch_count += 1
                return launch(*args, **kwargs)

            return record_launch

    monkeypatch.setattr(output_module, "chunk_gla_fwd_kernel_o_ragged_tma", RecordingKernel())
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
    expected_tma = output_module._can_use_tensor_descriptors(q, v, g, h, torch.empty_like(v), A)
    actual = output_module.chunk_gla_fwd_o_gk(q, v, g, A, h, scale, metadata=metadata)
    assert launch_count == int(expected_tma)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    "schedule",
    [
        ScheduleRequest.STATIC,
        pytest.param(
            ScheduleRequest.PERSISTENT,
            marks=pytest.mark.skipif(
                torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9,
                reason="persistent scheduling requires the TMA path",
            ),
        ),
    ],
)
def test_ragged_output_replays_aligned_to_ragged(schedule):
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
    output_module.chunk_gla_fwd_o_gk(
        q, v, g, A, h, scale, metadata=warm_metadata, schedule=schedule
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        metadata = prepare_ragged_chunk_metadata(cu_seqlens, tokens, 64)
        actual = output_module.chunk_gla_fwd_o_gk(
            q, v, g, A, h, scale, metadata=metadata, schedule=schedule
        )

    cu_seqlens.copy_(torch.tensor([0, 65, 128], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference(q, v, g, A, h, [65, 63], scale)
    assert metadata.chunk_offsets.tolist() == [0, 2, 3]
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 9,
    reason="persistent scheduling requires the TMA path",
)
@pytest.mark.parametrize("lengths", [[65, 63], [1, 64, 0, 65]])
def test_persistent_ragged_output_matches_static_over_capacity(lengths):
    torch.manual_seed(3)
    tokens = sum(lengths)
    capacity_tokens = tokens * 8
    heads, key_dim, value_dim = 2, 128, 128
    dtype = torch.bfloat16
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )
    metadata = prepare_ragged_chunk_metadata(offsets, capacity_tokens, 64)

    q = torch.randn(1, capacity_tokens, heads, key_dim, device="cuda", dtype=dtype) / 8
    v = torch.randn(1, capacity_tokens, heads, value_dim, device="cuda", dtype=dtype) / 8
    g = -torch.rand(1, capacity_tokens, heads, key_dim, device="cuda")
    A = torch.randn(1, capacity_tokens, heads, 64, device="cuda", dtype=dtype) / 8
    h = (
        torch.randn(1, metadata.capacity, heads, key_dim, value_dim, device="cuda", dtype=dtype)
        / 8
    )
    scale = 0.125

    expected = _reference(q, v, g, A, h, lengths, scale)
    static = output_module.chunk_gla_fwd_o_gk(
        q, v, g, A, h, scale, metadata=metadata, schedule=ScheduleRequest.STATIC
    )
    persistent = output_module.chunk_gla_fwd_o_gk(
        q, v, g, A, h, scale, metadata=metadata, schedule=ScheduleRequest.PERSISTENT
    )
    torch.testing.assert_close(persistent[:, :tokens], expected[:, :tokens], atol=2e-2, rtol=2e-2)
    assert torch.equal(persistent[:, :tokens], static[:, :tokens])


def test_persistent_is_noop_for_dense_and_raises_off_the_packed_tma_path():
    torch.manual_seed(4)
    tokens, heads, dim = 128, 2, 32
    q = torch.randn(1, tokens, heads, dim, device="cuda", dtype=torch.bfloat16) / 8
    v = torch.randn_like(q) / 8
    g = -torch.rand(1, tokens, heads, dim, device="cuda")
    A = torch.randn(1, tokens, heads, 64, device="cuda", dtype=torch.bfloat16) / 8
    h = torch.randn(1, 2, heads, dim, dim, device="cuda", dtype=torch.bfloat16) / 8

    # Dense launch grids are already exact, so the request is trivially satisfied.
    dense = output_module.chunk_gla_fwd_o_gk(q, v, g, A, h, 0.125)
    dense_persistent = output_module.chunk_gla_fwd_o_gk(
        q, v, g, A, h, 0.125, schedule=ScheduleRequest.PERSISTENT
    )
    assert torch.equal(dense, dense_persistent)

    # Packed inputs off the TMA path have no persistent kernel to honor the request.
    offsets = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(offsets, tokens, 64)
    with pytest.raises(ValueError, match="persistent scheduling"):
        output_module.chunk_gla_fwd_o_gk(
            q,
            v,
            g,
            A,
            h,
            0.125,
            metadata=metadata,
            schedule=ScheduleRequest.PERSISTENT,
        )


def test_ragged_output_zero_capacity_skips_launch():
    heads = 1
    for schedule in (ScheduleRequest.STATIC, ScheduleRequest.PERSISTENT):
        q = torch.empty(1, 0, heads, 128, device="cuda", dtype=torch.bfloat16)
        v = torch.empty_like(q)
        g = torch.empty(1, 0, heads, 128, device="cuda")
        A = torch.empty(1, 0, heads, 64, device="cuda", dtype=torch.bfloat16)
        h = torch.empty(1, 0, heads, 128, 128, device="cuda", dtype=torch.bfloat16)
        metadata = prepare_ragged_chunk_metadata(
            torch.tensor([0, 0], device="cuda", dtype=torch.int32), 0, 64
        )
        assert metadata.capacity == 0
        output = output_module.chunk_gla_fwd_o_gk(
            q, v, g, A, h, 0.125, metadata=metadata, schedule=schedule
        )
        assert output.shape == v.shape


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="persistent scheduling requires the TMA path",
)
def test_persistent_ragged_output_strides_multiple_tasks_per_worker(monkeypatch):
    """Force fewer workers than tasks so the stride loop iterates repeatedly."""
    from attn_gym.linear.kda import chunk_scheduler

    monkeypatch.setattr(chunk_scheduler.GridScheduler, "num_workers", lambda self, s, d: 3)
    torch.manual_seed(7)
    tokens, heads = 256, 2
    q = torch.randn(1, tokens, heads, 128, device="cuda", dtype=torch.bfloat16) / 8
    v = torch.randn_like(q) / 8
    g = -torch.rand(1, tokens, heads, 128, device="cuda")
    A = torch.randn(1, tokens, heads, 64, device="cuda", dtype=torch.bfloat16) / 8
    metadata = prepare_ragged_chunk_metadata(
        torch.tensor([0, 129, 256], device="cuda", dtype=torch.int32), tokens, 64
    )
    h = torch.randn(1, metadata.capacity, heads, 128, 128, device="cuda", dtype=torch.bfloat16) / 8

    static = output_module.chunk_gla_fwd_o_gk(
        q, v, g, A, h, 0.125, metadata=metadata, schedule=ScheduleRequest.STATIC
    )
    persistent = output_module.chunk_gla_fwd_o_gk(
        q,
        v,
        g,
        A,
        h,
        0.125,
        metadata=metadata,
        schedule=ScheduleRequest.PERSISTENT,
    )
    assert torch.equal(persistent, static)
