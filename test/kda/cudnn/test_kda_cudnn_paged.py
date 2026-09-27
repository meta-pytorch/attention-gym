"""Paged prefill through the KDA cuDNN backend."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip(
    "cutlass.experimental",
    reason="the CuTeDSL 4.7 KDA path requires nvidia-cutlass-dsl>=4.7",
)

from attn_gym.linear import chunk_kda, paged_chunk_kda
from attn_gym.linear.kda.impl.cudnn_ops import chunk_cudnn_packed_fwd_paged_op
from attn_gym.testing import cumulative_sequence_offsets, strided_state_pool
from attn_gym.testing.kda import make_kda_test_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTeDSL 4.7 KDA path requires SM100 or SM103",
)

_CUDNN = {"backend": "cudnn"}


def test_cudnn_paged_matches_gather_scatter() -> None:
    """Resumed, fresh, empty, and null routes match ordinary cuDNN execution bitwise."""
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
        kernel_options=_CUDNN,
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
            kernel_options=_CUDNN,
        )

    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(storage, expected_storage, rtol=0, atol=0)


def test_cudnn_paged_dense_batch_resumes_every_slot() -> None:
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
        kernel_options=_CUDNN,
    )
    assert final_state is not None
    expected_pool[state_indices.long()] = final_state

    with torch.no_grad():
        output = paged_chunk_kda(
            q, k, value, gate, beta, pool, state_indices, kernel_options=_CUDNN
        )

    torch.testing.assert_close(output, expected_output.view_as(output), rtol=0, atol=0)
    torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0)


def test_cudnn_paged_raw_operator_registration() -> None:
    q, k, value, gate, beta = make_kda_test_inputs(96, heads=2, seed=21)
    cu_seqlens = cumulative_sequence_offsets((64, 32))
    pool = torch.zeros(3, 2, 128, 128, device="cuda")
    state_indices = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    torch.library.opcheck(
        chunk_cudnn_packed_fwd_paged_op,
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


def test_cudnn_paged_fullgraph_and_cuda_graph_replay() -> None:
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
            kernel_options=_CUDNN,
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
            kernel_options=_CUDNN,
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


def _dense_reference(q, k, value, gate, beta, cu_seqlens, pool, routes, fresh):
    """Ordinary cuDNN execution over the gathered seeds, scattered back like the paged contract."""
    active = [route > 0 for route in routes]
    initial_state = torch.zeros(len(routes), *pool.shape[1:], device="cuda")
    for b, route in enumerate(routes):
        if active[b] and not fresh[b]:
            initial_state[b] = pool[route]
    output, final_state = chunk_kda(
        q,
        k,
        value,
        gate,
        beta,
        initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        kernel_options=_CUDNN,
    )
    assert final_state is not None
    expected_pool = pool.clone()
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    output = output.clone()
    for b, route in enumerate(routes):
        start, end = int(cu_seqlens[b]), int(cu_seqlens[b + 1])
        if not active[b]:
            output[:, start:end] = 0
        elif lengths[b] > 0 or fresh[b]:
            expected_pool[route] = final_state[b]
    return output, expected_pool


@pytest.mark.parametrize(
    "extra_empty",
    (0, 40),
    ids=("value-split-plan", "uncut-plan"),
)
def test_cudnn_paged_negative_and_zero_routes_never_touch_the_pool(extra_empty: int) -> None:
    """Negative and zero routes emit zeros and never read or write any slot, in both plans.

    Slot 0 and the unused slot hold NaN: any read would poison an output or state, any write
    would clear the NaN. Forty trailing null empties push ``sequences * heads`` past the
    value-split threshold so the uncut plan runs too.
    """
    lengths = (33, 0, 70, 0, 17, 0) + (0,) * extra_empty
    routes = [3, 4, -2, 2, 0, 1] + [0] * extra_empty
    fresh = [False, True, False, False, False, True] + [False] * extra_empty
    q, k, value, gate, beta = make_kda_test_inputs(sum(lengths), heads=2, seed=31)
    cu_seqlens = cumulative_sequence_offsets(lengths)
    q[:, 33:103] = torch.nan  # the negative route's tokens
    pool = torch.randn(6, 2, 128, 128, device="cuda") * 0.1
    pool[0] = torch.nan
    pool[5] = torch.nan
    expected_output, expected_pool = _dense_reference(
        q, k, value, gate, beta, cu_seqlens, pool, routes, fresh
    )
    has_initial_state = torch.tensor([not f for f in fresh], device="cuda")
    state_indices = torch.tensor(routes, device="cuda", dtype=torch.int32)

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
            kernel_options=_CUDNN,
        )

    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0, equal_nan=True)


def test_cudnn_paged_empty_routes_past_compaction_capacity() -> None:
    """Fresh empties clear and resumed empties persist on the uncompacted fallback table."""
    num_sequences = 4200  # more sequences than the order pass can compact
    lengths = [64, 48] + [0] * (num_sequences - 2)
    routes = [0] * num_sequences
    fresh = [False] * num_sequences
    routes[:2] = [1, 2]
    routes[2100], routes[-1] = 3, 4
    fresh[-1] = True
    q, k, value, gate, beta = make_kda_test_inputs(sum(lengths), heads=1, seed=37)
    cu_seqlens = cumulative_sequence_offsets(lengths)
    pool = torch.randn(5, 1, 128, 128, device="cuda")
    pool[0] = torch.nan
    expected_output, expected_pool = _dense_reference(
        q, k, value, gate, beta, cu_seqlens, pool, routes, fresh
    )
    assert expected_pool[4].abs().sum() == 0 and expected_pool[3].equal(pool[3])

    with torch.no_grad():
        output = paged_chunk_kda(
            q,
            k,
            value,
            gate,
            beta,
            pool,
            torch.tensor(routes, device="cuda", dtype=torch.int32),
            cu_seqlens=cu_seqlens,
            has_initial_state=torch.tensor([not f for f in fresh], device="cuda"),
            kernel_options=_CUDNN,
        )

    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
    torch.testing.assert_close(pool, expected_pool, rtol=0, atol=0, equal_nan=True)


def test_cudnn_paged_runs_the_native_driver(monkeypatch) -> None:
    """Paged calls launch the v1.30 driver in place and reject final-state and split requests."""
    from attn_gym.linear._delta_rule.cudnn import forward
    from attn_gym.linear._delta_rule.cudnn_fe import kda

    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs.get("state_indices"))
        return kda.kda_forward(*args, **kwargs)

    monkeypatch.setattr(forward, "kda_forward", spy)
    q, k, value, gate, beta = make_kda_test_inputs(96, heads=2, seed=41)
    cu_seqlens = cumulative_sequence_offsets((64, 32))
    pool = torch.zeros(3, 2, 128, 128, device="cuda")
    state_indices = torch.tensor([2, 1], device="cuda", dtype=torch.int32)
    with torch.no_grad():
        paged_chunk_kda(
            q,
            k,
            value,
            gate,
            beta,
            pool,
            state_indices,
            cu_seqlens=cu_seqlens,
            kernel_options=_CUDNN,
        )
    assert len(calls) == 1 and calls[0] is state_indices
    assert pool[1:].abs().sum() > 0 and pool[0].abs().sum() == 0

    packed = (q[0], k[0], value[0], gate[0], beta[0], cu_seqlens)
    for options in ({"output_final_state": True}, {"split": True}):
        with pytest.raises(ValueError, match="paged KDA"):
            kda.kda_forward(
                *packed, scale=1.0, initial_state=pool, state_indices=state_indices, **options
            )
