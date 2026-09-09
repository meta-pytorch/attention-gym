"""Native Mega affine probes define an ownership-invariant canonical CP baseline."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

pytest.importorskip("cutlass.experimental", reason="native Mega summaries require CuTeDSL 4.7")

from attn_gym.linear._delta_rule.mega.kernels import kda_prefill_f16
from attn_gym.linear._delta_rule.mega.kernels.common.host import tensormap_workspace_bytes
from attn_gym.linear._delta_rule.mega.schedule import prepare_mega_schedule
from attn_gym.linear._delta_rule.mega.state_summary import build_mega_state_summaries
from attn_gym.linear.context_parallel import merge_state
from attn_gym.linear.kda.stages import chunk_kda_prepare
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    clone_kda_inputs,
    cumulative_sequence_offsets,
    kda_reference,
    make_kda_test_inputs,
)
from attn_gym.utils import ceildiv

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="native Mega summaries require SM100 or SM103",
)


def _native_no_state_final(
    inputs: tuple[torch.Tensor, ...], cu_seqlens: torch.Tensor
) -> torch.Tensor:
    """Run the output-producing native no-state specialization as an independent B oracle."""
    q, k, value, gate, beta = inputs
    sequences = cu_seqlens.numel() - 1
    state = torch.zeros(sequences, q.shape[2], 128, 128, device=q.device)
    schedule = prepare_mega_schedule(
        gate,
        cu_seqlens,
        tile_tokens=16,
        counter_count=2,
        split=False,
        stream=torch.cuda.current_stream().cuda_stream,
    )
    workspace = torch.empty(
        ceildiv(tensormap_workspace_bytes(kda_prefill_f16, sequences), 8),
        device=q.device,
        dtype=torch.int64,
    )
    kda_prefill_f16.chunk_kda_sm100(
        q[0],
        k[0],
        value[0],
        gate[0],
        beta[0],
        torch.empty_like(value[0]),
        cu_seqlens,
        None,
        state,
        128**-0.5,
        work_items=schedule.work_items,
        work_count=schedule.work_count,
        sched_ctr=schedule.counters,
        tensormap_workspace=workspace,
    )
    return state


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_bias_matches_no_state_final_bitwise(dtype: torch.dtype) -> None:
    lengths = [17, 0, 65, 129]
    inputs = make_kda_test_inputs(sum(lengths), dtype=dtype, gate_scale=0.02, normalize_qk=True)
    cu = cumulative_sequence_offsets(lengths)
    maps = build_mega_state_summaries(*inputs[1:], cu)
    expected = _native_no_state_final(inputs, cu)
    assert torch.equal(maps[:, :, :128].contiguous().view(torch.int32), expected.view(torch.int32))
    torch.testing.assert_close(maps[1, 0, 128:], torch.eye(128, device="cuda"), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("gate_value,beta_value", [(0.0, None), (-0.002, 0.0), (-0.02, 1.0)])
def test_native_transition_matches_basis_reference(
    dtype: torch.dtype, gate_value: float, beta_value: float | None
) -> None:
    inputs = list(make_kda_test_inputs(17, dtype=dtype, gate_value=gate_value, normalize_qk=True))
    if beta_value is not None:
        inputs[-1].fill_(beta_value)
    cu = cumulative_sequence_offsets([17])
    actual = build_mega_state_summaries(*inputs[1:], cu)[:, :, 128:]
    inputs[2] = torch.zeros_like(inputs[2])
    references = []
    for precision in (torch.float64, torch.float32):
        # Every V row carries a different basis vector e_i: the final rows are A, not A.T.
        basis = torch.eye(128, device="cuda", dtype=precision)[None, None]
        _, final = kda_reference(*clone_kda_inputs(inputs, dtype=precision), basis)
        assert final is not None
        references.append(final)
    assert_matches_low_precision_reference(actual, *references, "transition A", source_dtype=dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_two_tile_composition_matches_reference(dtype: torch.dtype, monkeypatch) -> None:
    inputs = make_kda_test_inputs(128, dtype=dtype, gate_scale=0.02, normalize_qk=True)
    cu = cumulative_sequence_offsets([64, 64])
    prepared = chunk_kda_prepare(
        *inputs, cu_seqlens=cu, autotune=False, kernel_options={"backend": "mega"}
    )
    monkeypatch.setattr(
        "attn_gym.linear.kda.stages._prepare_chunk_kda_fwd",
        lambda *args, **kwargs: pytest.fail("native summaries must not prepare fused factors"),
    )
    bounds = torch.tensor([[0, 64], [64, 128]], device="cuda", dtype=torch.int32)
    maps = prepared.state_summaries(bounds, deterministic_work=True)
    entry0 = torch.zeros_like(maps[0, :, :128])
    entry1 = merge_state(entry0, maps[0])
    output, final = prepared.run(torch.stack((entry0, entry1)), output_final_state=True)
    assert final is not None
    composed_final = merge_state(entry1, maps[1])
    high_output, high_final = kda_reference(*clone_kda_inputs(inputs, dtype=torch.float64))
    low_output, low_final = kda_reference(*clone_kda_inputs(inputs, dtype=torch.float32))
    assert high_final is not None and low_final is not None
    assert_matches_low_precision_reference(
        output, high_output, low_output, "two-tile output", source_dtype=dtype
    )
    for name, actual in (("native final", final[-1:]), ("composed final", composed_final[None])):
        assert_matches_low_precision_reference(
            actual, high_final, low_final, name, source_dtype=dtype
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_summary_ownership_is_bitwise_invariant(dtype: torch.dtype) -> None:
    lengths = [17, 0, 65, 129]
    inputs = make_kda_test_inputs(sum(lengths), dtype=dtype, gate_scale=0.02, normalize_qk=True)
    cu = cumulative_sequence_offsets(lengths)
    packed = build_mega_state_summaries(*inputs[1:], cu)
    for index, (start, stop) in enumerate(pairwise(cu.tolist())):
        alone = build_mega_state_summaries(
            *(tensor[:, start:stop] for tensor in inputs[1:]),
            cumulative_sequence_offsets([stop - start]),
        )
        assert torch.equal(packed[index : index + 1].view(torch.int32), alone.view(torch.int32))


def test_native_summary_persistent_waves_are_bitwise_invariant() -> None:
    """Exercise more than four nonempty work items per physical CTA, including empty tiles."""
    lengths = [17, 0, 65, 129]
    repeats = 4 * torch.cuda.get_device_properties(0).multi_processor_count // 3 + 1
    inputs = make_kda_test_inputs(sum(lengths), gate_scale=0.02, normalize_qk=True)[1:]
    expected = build_mega_state_summaries(*inputs, cumulative_sequence_offsets(lengths))
    repeated = tuple(tensor.repeat(1, repeats, *([1] * (tensor.ndim - 2))) for tensor in inputs)
    cu = cumulative_sequence_offsets(lengths * repeats)
    for _ in range(2):
        actual = build_mega_state_summaries(*repeated, cu).reshape(repeats, *expected.shape)
        assert torch.equal(
            actual.view(torch.int32), expected[None].expand_as(actual).view(torch.int32)
        )


def test_native_summary_cuda_graph_replay() -> None:
    inputs = make_kda_test_inputs(81, gate_scale=0.02, normalize_qk=True)
    cu = cumulative_sequence_offsets([17, 0, 64])
    expected = build_mega_state_summaries(*inputs[1:], cu)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = build_mega_state_summaries(*inputs[1:], cu)
    graph.replay()
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
