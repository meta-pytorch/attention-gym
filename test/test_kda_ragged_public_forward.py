"""End-to-end forward tests for the public ragged CuTe KDA API."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd import chunk_kda

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTe KDA core requires CUDA capability 10.0 or newer",
)


def _inputs(tokens: int):
    torch.manual_seed(23)
    shape = (1, tokens, 1, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8
    k = torch.randn_like(q) / 8
    v = torch.randn_like(q) / 8
    gate = -torch.rand(shape, device="cuda")
    beta = torch.rand(1, tokens, 1, device="cuda")
    return q, k, v, gate, beta


def _offsets(lengths: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )


def test_public_ragged_forward_matches_independent_sequences_and_fullgraph():
    lengths = [65, 0, 63]
    inputs = _inputs(sum(lengths))
    initial_state = torch.randn(3, 1, 128, 128, device="cuda") / 8
    offsets = _offsets(lengths)

    actual, actual_state = chunk_kda(
        *inputs,
        initial_state,
        cu_seqlens=offsets,
        output_final_state=True,
    )

    expected_outputs = []
    expected_states = []
    begin = 0
    for sequence, length in enumerate(lengths):
        if length == 0:
            expected_states.append(initial_state[sequence])
            continue
        end = begin + length
        sequence_inputs = tuple(tensor[:, begin:end].clone() for tensor in inputs)
        output, state = chunk_kda(
            *sequence_inputs,
            initial_state[sequence : sequence + 1],
            cu_seqlens=_offsets([length]),
            output_final_state=True,
        )
        expected_outputs.append(output)
        expected_states.append(state[0])
        begin = end

    torch.testing.assert_close(actual, torch.cat(expected_outputs, dim=1), rtol=0, atol=0)
    torch.testing.assert_close(actual_state, torch.stack(expected_states), rtol=0, atol=0)

    compiled = torch.compile(chunk_kda, fullgraph=True)
    compiled_output, compiled_state = compiled(
        *inputs,
        initial_state,
        cu_seqlens=offsets,
        output_final_state=True,
    )
    torch.testing.assert_close(compiled_output, actual, rtol=0, atol=0)
    torch.testing.assert_close(compiled_state, actual_state, rtol=0, atol=0)


def test_dense_tail_batch_matches_explicit_packed_lowering():
    torch.manual_seed(31)
    shape = (2, 65, 1, 128)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8
    k = torch.randn_like(q) / 8
    v = torch.randn_like(q) / 8
    gate = -torch.rand(shape, device="cuda")
    beta = torch.rand(2, 65, 1, device="cuda")
    initial_state = torch.randn(2, 1, 128, 128, device="cuda") / 8

    dense, dense_state = chunk_kda(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        output_final_state=True,
    )
    packed_shape = (1, 130, 1, 128)
    packed, packed_state = chunk_kda(
        q.reshape(packed_shape),
        k.reshape(packed_shape),
        v.reshape(packed_shape),
        gate.reshape(packed_shape),
        beta.reshape(1, 130, 1),
        initial_state,
        cu_seqlens=_offsets([65, 65]),
        output_final_state=True,
    )

    torch.testing.assert_close(dense, packed.reshape(shape), rtol=0, atol=0)
    torch.testing.assert_close(dense_state, packed_state, rtol=0, atol=0)


def test_public_forward_replays_aligned_to_ragged():
    inputs = _inputs(128)
    initial_state = torch.randn(2, 1, 128, 128, device="cuda") / 8
    offsets = _offsets([64, 64])
    chunk_kda(
        *inputs,
        initial_state,
        cu_seqlens=offsets,
        output_final_state=True,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, actual_state = chunk_kda(
            *inputs,
            initial_state,
            cu_seqlens=offsets,
            output_final_state=True,
        )

    offsets.copy_(_offsets([65, 63]))
    graph.replay()
    torch.cuda.synchronize()

    expected, expected_state = chunk_kda(
        *inputs,
        initial_state,
        cu_seqlens=_offsets([65, 63]),
        output_final_state=True,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
