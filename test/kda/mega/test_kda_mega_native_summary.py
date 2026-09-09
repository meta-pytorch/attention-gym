"""Native Mega affine probes define a standard Mega CP baseline."""

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

DTYPES = [torch.bfloat16, torch.float16]
# Ragged packed span with an empty sequence; cu_seqlens = 0, 17, 17, 82, 211.
LENGTHS = [17, 0, 65, 129]
SCALE = 128**-0.5
MEGA = {"backend": "mega"}


def _packed_inputs(
    lengths: list[int],
    dtype: torch.dtype = torch.bfloat16,
    *,
    heads: int = 1,
    gate_value: float | None = None,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Normalized-QK inputs with mild gates, plus their packed boundaries."""
    inputs = make_kda_test_inputs(
        sum(lengths),
        heads=heads,
        dtype=dtype,
        gate_scale=0.02,
        gate_value=gate_value,
        normalize_qk=True,
    )
    return inputs, cumulative_sequence_offsets(lengths)


def _bounds(rows: list[list[int]]) -> torch.Tensor:
    """Device ``int32 [R, 2]`` token bounds, one whole (or empty) sequence per row."""
    return torch.tensor(rows, device="cuda", dtype=torch.int32)


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Bit-exact FP32 comparison; unlike ``torch.equal`` it distinguishes signed zeros."""
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def _forbid_fused_factors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if a staged handle materializes fused WY factors on a native-summary path."""
    for name in ("_prepare_chunk_kda_fwd", "_prepare_chunk_kda_bwd"):
        monkeypatch.setattr(
            f"attn_gym.linear.kda.stages.{name}",
            lambda *args, **kwargs: pytest.fail("native summaries must not prepare fused factors"),
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
        SCALE,
        work_items=schedule.work_items,
        work_count=schedule.work_count,
        sched_ctr=schedule.counters,
        tensormap_workspace=workspace,
    )
    return state


@pytest.mark.parametrize("dtype", DTYPES)
def test_native_bias_matches_no_state_final_bitwise(dtype: torch.dtype) -> None:
    inputs, cu = _packed_inputs(LENGTHS, dtype)
    maps = build_mega_state_summaries(*inputs[1:], cu)
    _assert_bitwise_equal(maps[:, :, :128].contiguous(), _native_no_state_final(inputs, cu))
    _assert_bitwise_equal(maps[1, 0, 128:], torch.eye(128, device="cuda"))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("gate_value,beta_value", [(0.0, None), (-0.002, 0.0), (-0.02, 1.0)])
def test_native_transition_matches_basis_reference(
    dtype: torch.dtype, gate_value: float, beta_value: float | None
) -> None:
    inputs, cu = _packed_inputs([17], dtype, gate_value=gate_value)
    inputs = list(inputs)
    if beta_value is not None:
        inputs[-1].fill_(beta_value)
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


@pytest.mark.parametrize("dtype", DTYPES)
def test_native_two_tile_handoff_matches_fp64(dtype: torch.dtype, monkeypatch) -> None:
    """Native forward maps hand off states; outputs and finals track FP64."""
    inputs, cu = _packed_inputs([64, 65], dtype)
    prepared = chunk_kda_prepare(*inputs, cu_seqlens=cu, autotune=False, kernel_options=MEGA)
    _forbid_fused_factors(monkeypatch)
    bounds = torch.stack((cu[:-1], cu[1:]), dim=1)
    maps = prepared.state_summaries(bounds)
    entry0 = torch.zeros_like(maps[0, :, :128])
    entries = torch.stack((entry0, merge_state(entry0, maps[0])))
    output, final = prepared.run(entries, output_final_state=True)
    assert final is not None
    composed_final = merge_state(entries[1], maps[1])
    references = []
    for precision in (torch.float64, torch.float32):
        leaves = tuple(t.requires_grad_() for t in clone_kda_inputs(inputs, dtype=precision))
        ref_output, ref_final = kda_reference(*leaves, scale=prepared.scale)
        references.append((ref_output, ref_final, ref_final))
    for name, actual, high, low in zip(
        ("output", "native final", "composed final"),
        (output, final[-1:], composed_final[None]),
        *references,
        strict=True,
    ):
        assert_matches_low_precision_reference(actual, high, low, name, source_dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_native_summary_ownership_is_bitwise_invariant(dtype: torch.dtype) -> None:
    inputs, cu = _packed_inputs(LENGTHS, dtype)
    packed = build_mega_state_summaries(*inputs[1:], cu)
    for index, (start, stop) in enumerate(pairwise(cu.tolist())):
        alone = build_mega_state_summaries(
            *(tensor[:, start:stop] for tensor in inputs[1:]),
            cumulative_sequence_offsets([stop - start]),
        )
        _assert_bitwise_equal(packed[index : index + 1], alone)


def test_native_summary_persistent_waves_are_bitwise_invariant() -> None:
    """Exercise more than four nonempty work items per physical CTA, including empty tiles."""
    repeats = 4 * torch.cuda.get_device_properties(0).multi_processor_count // 3 + 1
    inputs, cu = _packed_inputs(LENGTHS)
    expected = build_mega_state_summaries(*inputs[1:], cu)
    repeated = tuple(t.repeat(1, repeats, *([1] * (t.ndim - 2))) for t in inputs[1:])
    cu = cumulative_sequence_offsets(LENGTHS * repeats)
    for _ in range(2):
        actual = build_mega_state_summaries(*repeated, cu).reshape(repeats, *expected.shape)
        _assert_bitwise_equal(actual, expected[None].expand_as(actual))


def test_native_summary_cuda_graph_replay() -> None:
    inputs, cu = _packed_inputs([17, 0, 64])
    expected = build_mega_state_summaries(*inputs[1:], cu)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = build_mega_state_summaries(*inputs[1:], cu)
    graph.replay()
    _assert_bitwise_equal(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", [1, 64])
def test_native_selected_bounds(dtype: torch.dtype, heads: int, monkeypatch) -> None:
    """Selecting subsequences (reordered, duplicated, or empty) on device preserves the bits.

    The staged handles stay native, and bounds may change between CUDA Graph replays.
    """
    inputs, cu = _packed_inputs(LENGTHS, dtype, heads=heads)
    prepared = chunk_kda_prepare(*inputs, cu_seqlens=cu, autotune=False, kernel_options=MEGA)
    _forbid_fused_factors(monkeypatch)
    full = build_mega_state_summaries(*inputs[1:], cu)
    selected = prepared.state_summaries
    cases = (
        ([[82, 211], [17, 17], [17, 82], [82, 211]], [3, 1, 2, 3]),
        ([[0, 17], [17, 17], [82, 211], [0, 0]], [0, 1, 3, 1]),
        ([[17, 17]] * 4, [1] * 4),
        ([[17, 82], [82, 211], [0, 17], [17, 82]], [2, 3, 0, 2]),
    )
    bounds = _bounds(cases[0][0])
    _assert_bitwise_equal(selected(bounds=bounds), full[cases[0][1]])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = selected(bounds=bounds)
    for rows, indices in cases:
        bounds.copy_(_bounds(rows))
        graph.replay()
        _assert_bitwise_equal(captured, full[indices])
    assert selected(bounds=bounds[:0]).shape == (0, heads, 256, 128)
    # A nonempty row that is not a subsequence is poisoned with NaN rather than silently mapped.
    poisoned = selected(bounds=_bounds([[17, 82], [20, 82], [0, 17]]))
    assert not poisoned[[0, 2]].isnan().any() and poisoned[1].isnan().all()
