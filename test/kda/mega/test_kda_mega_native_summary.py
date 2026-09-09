"""Native Mega affine probes define an ownership-invariant canonical CP baseline."""

from __future__ import annotations

from itertools import pairwise

import pytest
import torch

pytest.importorskip("cutlass.experimental", reason="native Mega summaries require CuTeDSL 4.7")

from attn_gym.linear._delta_rule.mega.kernels import kda_prefill_f16
from attn_gym.linear._delta_rule.mega.kernels.common.host import tensormap_workspace_bytes
from attn_gym.linear._delta_rule.mega.schedule import prepare_mega_schedule
from attn_gym.linear._delta_rule.mega.state_summary import (
    build_mega_state_grad_summaries,
    build_mega_state_summaries,
)
from attn_gym.linear.context_parallel import merge_state
from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_selected_forward_bounds(dtype: torch.dtype, monkeypatch) -> None:
    """Standard CP selects whole subsequences, reordered/duplicated or empty, on device."""
    inputs = make_kda_test_inputs(211, dtype=dtype, gate_scale=0.02, normalize_qk=True)
    cu = cumulative_sequence_offsets([17, 0, 65, 129])
    prepared = chunk_kda_prepare(
        *inputs, cu_seqlens=cu, autotune=False, kernel_options={"backend": "mega"}
    )
    monkeypatch.setattr(
        "attn_gym.linear.kda.stages._prepare_chunk_kda_fwd",
        lambda *args, **kwargs: pytest.fail("whole-sequence summaries must stay native"),
    )
    all_maps = build_mega_state_summaries(*inputs[1:], cu)
    bounds = torch.tensor(
        [[82, 211], [0, 0], [17, 82], [82, 211]], device="cuda", dtype=torch.int32
    )
    expected = all_maps[[3, 1, 2, 3]]
    actual = prepared.state_summaries(bounds, whole_sequences=True)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = prepared.state_summaries(bounds, whole_sequences=True)
    graph.replay()
    assert torch.equal(captured.view(torch.int32), expected.view(torch.int32))
    bounds.copy_(torch.tensor([[0, 17], [17, 17], [82, 211], [0, 0]], device="cuda"))
    graph.replay()
    assert torch.equal(captured.view(torch.int32), all_maps[[0, 1, 3, 1]].view(torch.int32))
    bounds.zero_()
    graph.replay()
    assert torch.equal(captured, all_maps[1:2].expand_as(captured))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("transpose_forward_transition", [False, True])
def test_native_reverse_bias_and_ownership(
    dtype: torch.dtype, transpose_forward_transition: bool, monkeypatch
) -> None:
    """C is exactly native dH0 and neither reverse probe depends on tile ownership/state."""
    lengths = [17, 0, 65, 129]
    inputs = make_kda_test_inputs(sum(lengths), dtype=dtype, gate_scale=0.02, normalize_qk=True)
    cu = cumulative_sequence_offsets(lengths)
    d_output = torch.randn_like(inputs[2])
    prepared = chunk_kda_prepare(
        *inputs, cu_seqlens=cu, autotune=False, kernel_options={"backend": "mega"}
    )
    initial = torch.randn(4, 1, 128, 128, device="cuda")
    backward = chunk_kda_prepare_backward(
        prepared.saved, d_output, initial, scale=prepared.scale, autotune=False
    )
    maps = build_mega_state_grad_summaries(
        *inputs,
        d_output,
        cu,
        prepared.scale,
        transpose_forward_transition=transpose_forward_transition,
    )
    expected = backward.run()[-1]
    assert torch.equal(maps[:, :, :128], expected)
    monkeypatch.setattr(
        "attn_gym.linear.kda.stages._prepare_chunk_kda_bwd",
        lambda *args, **kwargs: pytest.fail("canonical reverse summaries must stay native"),
    )
    if transpose_forward_transition:
        bounds = torch.stack((cu[:-1], cu[1:]), dim=1)
        assert torch.equal(backward.state_grad_summaries(bounds, deterministic_work=True), maps)
    else:
        zero_backward = chunk_kda_prepare_backward(
            prepared.saved,
            torch.zeros_like(d_output),
            initial,
            scale=prepared.scale,
            autotune=False,
        )
        identity = torch.eye(128, device="cuda").expand_as(initial).contiguous()
        assert torch.equal(maps[:, :, 128:], zero_backward.run(identity)[-1])
    for index, (start, stop) in enumerate(pairwise(cu.tolist())):
        alone = build_mega_state_grad_summaries(
            *(tensor[:, start:stop] for tensor in inputs),
            d_output[:, start:stop],
            cumulative_sequence_offsets([stop - start]),
            prepared.scale,
            transpose_forward_transition=transpose_forward_transition,
        )
        assert torch.equal(maps[index : index + 1].view(torch.int32), alone.view(torch.int32))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("gate_value", [0.0, -0.002, -0.02])
def test_native_reverse_matches_fp64(dtype: torch.dtype, gate_value: float) -> None:
    """Both reverse recipes and fused maps are measured against an independent FP64 oracle."""
    inputs = make_kda_test_inputs(129, dtype=dtype, gate_value=gate_value, normalize_qk=True)
    cu = cumulative_sequence_offsets([129])
    do = torch.randn_like(inputs[2])
    scale = 0.25
    maps = [
        build_mega_state_grad_summaries(
            *inputs, do, cu, scale, transpose_forward_transition=transpose
        )
        for transpose in (False, True)
    ]
    prepared = chunk_kda_prepare(*inputs, cu_seqlens=cu, scale=scale, autotune=False)
    backward = chunk_kda_prepare_backward(prepared.saved, do, None, scale=scale, autotune=False)
    maps.append(
        backward.state_grad_summaries(torch.tensor([[0, 129]], device="cuda", dtype=torch.int32))
    )
    references = []
    for precision in (torch.float64, torch.float32):
        state = torch.zeros(1, 1, 128, 128, device="cuda", dtype=precision, requires_grad=True)
        output, _ = kda_reference(*clone_kda_inputs(inputs, dtype=precision), state, scale=scale)
        (bias,) = torch.autograd.grad(output, state, do.to(precision))
        basis_inputs = list(clone_kda_inputs(inputs, dtype=precision))
        basis_inputs[2] = torch.zeros_like(basis_inputs[2])
        _, transition = kda_reference(
            *basis_inputs, torch.eye(128, device="cuda", dtype=precision)[None, None], scale=scale
        )
        references.append(torch.cat((bias, transition.transpose(-1, -2)), dim=-2))
    for label, actual in zip(("native bprop", "native C + A.T", "fused"), maps, strict=True):
        for offset, part in ((0, "C"), (128, "R")):
            high = references[0][:, :, offset : offset + 128]
            low = references[1][:, :, offset : offset + 128]
            got = actual[:, :, offset : offset + 128]
            if part == "R" and label != "fused":
                # Native R rounds the recurrent state/decay every BT16, not just at the
                # output. A uniform gate can bias those roundings in the same direction;
                # allow one source epsilon per chunk, rather than the single-round budget.
                allowance = ceildiv(inputs[0].shape[1], 16) * torch.finfo(dtype).eps
                budget = (low.double() - high).abs().max() + allowance * high.abs().max()
                assert torch.isfinite(got).all()
                assert (got.double() - high).abs().max() <= budget
            else:
                assert_matches_low_precision_reference(
                    got, high, low, f"{label} {part}", source_dtype=dtype
                )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_two_tile_backward_matches_fp64(dtype: torch.dtype) -> None:
    """Native forward/reverse maps hand off both states while all five gradients track FP64."""
    inputs = make_kda_test_inputs(129, dtype=dtype, gate_scale=0.02, normalize_qk=True)
    cu = cumulative_sequence_offsets([64, 65])
    bounds = torch.tensor([[0, 64], [64, 129]], device="cuda", dtype=torch.int32)
    prepared = chunk_kda_prepare(
        *inputs, cu_seqlens=cu, autotune=False, kernel_options={"backend": "mega"}
    )
    maps = prepared.state_summaries(bounds, deterministic_work=True)
    entry0 = torch.zeros_like(maps[0, :, :128])
    entries = torch.stack((entry0, merge_state(entry0, maps[0])))
    output, final = prepared.run(entries, output_final_state=True)
    do = torch.randn_like(output)
    exit1 = torch.randn_like(entry0) * 0.1
    backward = chunk_kda_prepare_backward(
        prepared.saved, do, entries, scale=prepared.scale, autotune=False
    )
    reverse = backward.state_grad_summaries(bounds, deterministic_work=True)
    exits = torch.stack((merge_state(exit1, reverse[1]), exit1))
    gradients = backward.run(exits)[:5]
    references = []
    for precision in (torch.float64, torch.float32):
        leaves = tuple(t.requires_grad_() for t in clone_kda_inputs(inputs, dtype=precision))
        ref_output, ref_final = kda_reference(*leaves, scale=prepared.scale)
        ref_gradients = torch.autograd.grad(
            (ref_output, ref_final), leaves, (do.to(precision), exit1[None].to(precision))
        )
        references.append((ref_output, ref_final, *ref_gradients))
    for name, actual, high, low in zip(
        ("output", "final", "dq", "dk", "dv", "dgate", "dbeta"),
        (output, final[-1:], *gradients),
        *references,
        strict=True,
    ):
        assert_matches_low_precision_reference(actual, high, low, name, source_dtype=dtype)


def test_native_reverse_cuda_graph_replay() -> None:
    inputs = make_kda_test_inputs(81, gate_scale=0.02, normalize_qk=True)
    cu = cumulative_sequence_offsets([17, 0, 64])
    do = torch.randn_like(inputs[2])
    expected = build_mega_state_grad_summaries(
        *inputs, do, cu, 128**-0.5, transpose_forward_transition=True
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = build_mega_state_grad_summaries(
            *inputs, do, cu, 128**-0.5, transpose_forward_transition=True
        )
    do.mul_(0.5)
    graph.replay()
    fresh = build_mega_state_grad_summaries(
        *inputs, do, cu, 128**-0.5, transpose_forward_transition=True
    )
    assert not torch.equal(actual, expected)
    assert torch.equal(actual.view(torch.int32), fresh.view(torch.int32))
