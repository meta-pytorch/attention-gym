"""Paged prefill through the KDA Mega backend."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip(
    "cutlass.experimental",
    reason="the CuTeDSL 4.7 KDA path requires nvidia-cutlass-dsl>=4.7",
)

from attn_gym.linear import chunk_kda, paged_chunk_kda
from attn_gym.linear.kda.impl.mega_ops import chunk_mega_packed_fwd_paged_op
from attn_gym.testing import cumulative_sequence_offsets, strided_state_pool
from attn_gym.testing.kda import make_kda_test_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTeDSL 4.7 KDA path requires SM100 or SM103",
)

_MEGA = {"backend": "mega"}


def test_mega_paged_matches_gather_scatter() -> None:
    """Resumed, fresh, empty, and null routes match ordinary Mega execution bitwise."""
    q, k, value, gate, beta = make_kda_test_inputs(256, heads=2, seed=9)
    cu_seqlens = cumulative_sequence_offsets((65, 0, 0, 0, 127, 64))
    state_indices = torch.tensor([4, 2, 5, 0, 0, 1], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([True, False, True, True, True, False], device="cuda")
    for tensor in (q, k, value, gate, beta):
        tensor[:, 65:192] = torch.nan

    storage, pool = strided_state_pool(6, 2, 128, 128, prefix=0, suffix=32)
    expected_storage = storage.clone()
    expected_pool = expected_storage[:, : pool[0].numel()].view_as(pool)
    initial_state = torch.zeros(6, 2, 128, 128, device="cuda")
    initial_state[0] = expected_pool[4]
    initial_state[2] = expected_pool[5]
    expected_output, final_state = chunk_kda(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        kernel_options=_MEGA,
    )
    assert final_state is not None
    expected_output = expected_output.clone()
    expected_output[:, 65:192] = 0
    expected_pool[4] = final_state[0]
    expected_pool[2] = final_state[1]
    expected_pool[5] = final_state[2]
    expected_pool[1] = final_state[5]

    with torch.no_grad():
        output = paged_chunk_kda(
            q,
            k,
            value,
            gate,
            beta,
            pool,
            state_indices,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
            kernel_options=_MEGA,
        )

    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(storage, expected_storage, rtol=0, atol=0)


def test_mega_paged_dense_batch_resumes_every_slot() -> None:
    """Dense batches use the same packed routing contract without a seed mask."""
    q, k, value, gate, beta = make_kda_test_inputs(64, batch=3, heads=2, seed=17)
    state_indices = torch.tensor([3, 1, 2], device="cuda", dtype=torch.int32)
    pool = torch.randn(4, 2, 128, 128, device="cuda")
    expected_pool = pool.clone()
    packed = tuple(
        tensor.reshape(1, -1, *tensor.shape[2:]) for tensor in (q, k, value, gate, beta)
    )
    cu_seqlens = torch.arange(4, device="cuda", dtype=torch.int32) * 64
    expected_output, final_state = chunk_kda(
        *packed,
        pool[state_indices.long()].clone(),
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        kernel_options=_MEGA,
    )
    assert final_state is not None
    expected_pool[state_indices.long()] = final_state

    with torch.no_grad():
        output = paged_chunk_kda(
            q, k, value, gate, beta, pool, state_indices, kernel_options=_MEGA
        )

    torch.testing.assert_close(output, expected_output.view_as(output), rtol=0, atol=0)
    torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0)


def test_mega_paged_raw_operator_registration() -> None:
    q, k, value, gate, beta = make_kda_test_inputs(96, heads=2, seed=21)
    cu_seqlens = cumulative_sequence_offsets((64, 32))
    pool = torch.zeros(3, 2, 128, 128, device="cuda")
    state_indices = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    torch.library.opcheck(
        chunk_mega_packed_fwd_paged_op,
        (
            q,
            k,
            value,
            gate,
            beta,
            pool,
            state_indices,
            None,
            cu_seqlens,
            128**-0.5,
        ),
    )


def test_mega_paged_fullgraph_and_cuda_graph_replay() -> None:
    q, k, value, gate, beta = make_kda_test_inputs(128, heads=2, seed=27)
    cu_seqlens = cumulative_sequence_offsets((65, 63))
    state_indices = torch.tensor([2, 1], device="cuda", dtype=torch.int32)
    has_initial_state = torch.tensor([True, False], device="cuda")
    seed_pool = torch.randn(3, 2, 128, 128, device="cuda") * 0.01

    @torch.compile(fullgraph=True)
    def compiled(q, k, value, gate, beta, pool, state_indices, has_initial_state, cu_seqlens):
        return paged_chunk_kda(
            q,
            k,
            value,
            gate,
            beta,
            pool,
            state_indices,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
            kernel_options=_MEGA,
        )

    with torch.no_grad():
        eager_pool = seed_pool.clone()
        expected = paged_chunk_kda(
            q,
            k,
            value,
            gate,
            beta,
            eager_pool,
            state_indices,
            cu_seqlens=cu_seqlens,
            has_initial_state=has_initial_state,
            kernel_options=_MEGA,
        )
        graph_pool = seed_pool.clone()
        actual = compiled(
            q, k, value, gate, beta, graph_pool, state_indices, has_initial_state, cu_seqlens
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(graph_pool, eager_pool, rtol=0, atol=0)

        graph_pool.copy_(seed_pool)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = compiled(
                q, k, value, gate, beta, graph_pool, state_indices, has_initial_state, cu_seqlens
            )
        graph_pool.copy_(seed_pool)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected, rtol=0, atol=0)
        torch.testing.assert_close(graph_pool, eager_pool, rtol=0, atol=0)
