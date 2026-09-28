"""Native cuDNN affine probes define a standard cuDNN CP baseline."""

from __future__ import annotations

from functools import partial
from itertools import pairwise

import pytest
import torch

pytest.importorskip("cutlass.experimental", reason="native cuDNN summaries require CuTeDSL 4.7")

from attn_gym.linear._delta_rule.cudnn.state_summary import (
    build_cudnn_state_grad_summaries,
    build_cudnn_state_summaries,
)
from attn_gym.linear._delta_rule.cudnn_fe import kda as cudnn_fe_kda
from attn_gym.linear._delta_rule.cudnn_fe import summary as native_summary
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
    reason="native cuDNN summaries require SM100 or SM103",
)

DTYPES = [torch.bfloat16, torch.float16]
# Ragged packed span with an empty sequence; cu_seqlens = 0, 17, 17, 82, 211.
LENGTHS = [17, 0, 65, 129]
SCALE = 128**-0.5
CUDNN = {"backend": "cudnn"}


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
    """Run the output-producing native no-state kernel as an independent B oracle.

    ``split=True`` pins the uncut BT16 prefill plan (the split-K table never cuts these short
    sequences); the automatic plan may pick the d_v split, whose rounding differs.
    """
    _, final = cudnn_fe_kda.kda_forward(
        *(t[0] for t in inputs), cu_seqlens, scale=SCALE, output_final_state=True, split=True
    )
    assert final is not None
    return final


def _forbid_driver_plans(monkeypatch: pytest.MonkeyPatch) -> None:
    """Summaries launch the summary kernels directly, never a forward/backward driver plan."""
    for name in ("kda_forward", "kda_backward"):
        monkeypatch.setattr(
            cudnn_fe_kda,
            name,
            lambda *args, **kwargs: pytest.fail("native summaries must not select a driver plan"),
        )


def _record_work_tables(monkeypatch: pytest.MonkeyPatch) -> list[torch.Tensor]:
    """Capture the selected work table handed to every native summary launch."""
    tables: list[torch.Tensor] = []
    for name in ("_forward_summary", "_reverse_summary"):
        original = getattr(native_summary, name)

        def checked(*args, original=original, **kwargs):
            tables.append(args[-1].work_items.clone())
            return original(*args, **kwargs)

        monkeypatch.setattr(native_summary, name, checked)
    return tables


@pytest.mark.parametrize("dtype", DTYPES)
def test_native_bias_matches_no_state_final_bitwise(dtype: torch.dtype) -> None:
    inputs, cu = _packed_inputs(LENGTHS, dtype)
    maps = build_cudnn_state_summaries(*inputs[1:], cu)
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
    actual = build_cudnn_state_summaries(*inputs[1:], cu)[:, :, 128:]
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
    """Native forward/reverse maps hand off both states; outputs, finals, and grads track FP64."""
    inputs, cu = _packed_inputs([64, 65], dtype)
    prepared = chunk_kda_prepare(*inputs, cu_seqlens=cu, autotune=False, kernel_options=CUDNN)
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
    packed = build_cudnn_state_summaries(*inputs[1:], cu)
    for index, (start, stop) in enumerate(pairwise(cu.tolist())):
        alone = build_cudnn_state_summaries(
            *(tensor[:, start:stop] for tensor in inputs[1:]),
            cumulative_sequence_offsets([stop - start]),
        )
        _assert_bitwise_equal(packed[index : index + 1], alone)


def test_native_summary_persistent_waves_are_bitwise_invariant() -> None:
    """Exercise more than four nonempty work items per physical CTA, including empty tiles."""
    repeats = 4 * torch.cuda.get_device_properties(0).multi_processor_count // 3 + 1
    inputs, cu = _packed_inputs(LENGTHS)
    expected = build_cudnn_state_summaries(*inputs[1:], cu)
    repeated = tuple(t.repeat(1, repeats, *([1] * (t.ndim - 2))) for t in inputs[1:])
    cu = cumulative_sequence_offsets(LENGTHS * repeats)
    for _ in range(2):
        actual = build_cudnn_state_summaries(*repeated, cu).reshape(repeats, *expected.shape)
        _assert_bitwise_equal(actual, expected[None].expand_as(actual))


def test_native_summary_cuda_graph_replay() -> None:
    inputs, cu = _packed_inputs([17, 0, 64])
    expected = build_cudnn_state_summaries(*inputs[1:], cu)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = build_cudnn_state_summaries(*inputs[1:], cu)
    graph.replay()
    _assert_bitwise_equal(actual, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads", [1, 64])
@pytest.mark.parametrize("probe", ["forward", "reverse", "reverse_bprop"])
def test_native_selected_bounds(dtype: torch.dtype, heads: int, probe: str, monkeypatch) -> None:
    """Selecting subsequences (reordered, duplicated, or empty) on device preserves the bits.

    The staged handles stay native, and bounds may change between CUDA Graph replays.
    """
    inputs, cu = _packed_inputs(LENGTHS, dtype, heads=heads)
    do = torch.randn_like(inputs[2])
    prepared = chunk_kda_prepare(*inputs, cu_seqlens=cu, autotune=False, kernel_options=CUDNN)
    backward = chunk_kda_prepare_backward(
        prepared.saved, do, None, scale=prepared.scale, autotune=False
    )
    _forbid_fused_factors(monkeypatch)
    grad_summaries = partial(build_cudnn_state_grad_summaries, *inputs, do, cu, prepared.scale)
    match probe:
        case "forward":
            full = build_cudnn_state_summaries(*inputs[1:], cu)
            selected = prepared.state_summaries
        case "reverse":
            full = grad_summaries(transpose_forward_transition=True)
            selected = backward.state_grad_summaries
        case "reverse_bprop":
            full = grad_summaries()
            selected = grad_summaries
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


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("transpose_forward_transition", [False, True])
def test_native_reverse_bias_and_ownership(
    dtype: torch.dtype, transpose_forward_transition: bool, monkeypatch
) -> None:
    """C is exactly native dH0 and neither reverse probe depends on tile ownership/state."""
    inputs, cu = _packed_inputs(LENGTHS, dtype)
    d_output = torch.randn_like(inputs[2])
    prepared = chunk_kda_prepare(*inputs, cu_seqlens=cu, autotune=False, kernel_options=CUDNN)
    initial = torch.randn(4, 1, 128, 128, device="cuda")
    backward = chunk_kda_prepare_backward(
        prepared.saved, d_output, initial, scale=prepared.scale, autotune=False
    )
    _forbid_fused_factors(monkeypatch)
    grad_summaries = partial(
        build_cudnn_state_grad_summaries,
        transpose_forward_transition=transpose_forward_transition,
    )
    maps = grad_summaries(*inputs, d_output, cu, prepared.scale)
    assert torch.equal(maps[:, :, :128], backward.run()[-1])
    if transpose_forward_transition:
        bounds = torch.stack((cu[:-1], cu[1:]), dim=1)
        assert torch.equal(backward.state_grad_summaries(bounds), maps)
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
        alone = grad_summaries(
            *(tensor[:, start:stop] for tensor in (*inputs, d_output)),
            cumulative_sequence_offsets([stop - start]),
            prepared.scale,
        )
        _assert_bitwise_equal(maps[index : index + 1], alone)


def test_native_reverse_selection_skips_unrequested_work(monkeypatch) -> None:
    """The native kernels consume the selected work table, not a prologue-generated full one."""
    inputs, cu = _packed_inputs(LENGTHS)
    do = torch.randn_like(inputs[2])
    bounds = _bounds([[82, 211], [17, 17], [82, 211]])
    tables = _record_work_tables(monkeypatch)
    _forbid_driver_plans(monkeypatch)
    grad_summaries = partial(
        build_cudnn_state_grad_summaries, *inputs, do, cu, SCALE, transpose_forward_transition=True
    )
    grad_summaries(bounds=bounds)
    assert len(tables) == 2
    # [seq, head, write_start, write_end, compute_start, compute_end, start, stop, dsts...]
    expected = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 17, 17, 1, 1],
            [2, 0, 0, 0, 0, 0, 17, 17, 2, 2],
            [3, 0, 0, 9, 0, 9, 82, 211, 3, 3],
        ],
        device="cuda",
        dtype=torch.int32,
    )
    for table in tables:
        assert torch.equal(table, expected)
    tables.clear()
    assert grad_summaries(bounds=bounds[:0]).shape == (0, 1, 256, 128)
    assert not tables


def test_native_summary_launches_no_driver_plans(monkeypatch) -> None:
    """Every validated entry point launches only the v1.30 summary kernels."""
    _forbid_driver_plans(monkeypatch)
    tables = _record_work_tables(monkeypatch)
    inputs, cu = _packed_inputs(LENGTHS)
    do = torch.randn_like(inputs[2])
    build_cudnn_state_summaries(*inputs[1:], cu)
    build_cudnn_state_grad_summaries(*inputs, do, cu, SCALE, transpose_forward_transition=True)
    build_cudnn_state_grad_summaries(*inputs, do, cu, SCALE)
    assert len(tables) == 5


def test_native_summary_all_empty_or_unmatched_bounds() -> None:
    """Rows never touching a launch still follow the identity/NaN contract, also under replay."""
    inputs, cu = _packed_inputs(LENGTHS)
    do = torch.randn_like(inputs[2])
    identity = torch.cat((torch.zeros(128, 128), torch.eye(128))).cuda()
    bounds = _bounds([[0, 0], [17, 17], [211, 211]])
    graphs = []
    for build in (
        partial(build_cudnn_state_summaries, *inputs[1:], cu),
        partial(build_cudnn_state_grad_summaries, *inputs, do, cu, SCALE),
    ):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = build(bounds=bounds)
        graph.replay()
        _assert_bitwise_equal(captured, identity.expand_as(captured))
        bounds.copy_(_bounds([[0, 16], [18, 82], [83, 211]]))
        graph.replay()
        assert captured.isnan().all()
        bounds.copy_(_bounds([[0, 0], [17, 17], [211, 211]]))
        graphs.append((graph, captured))
    for graph, captured in graphs:
        graph.replay()
        _assert_bitwise_equal(captured, identity.expand_as(captured))


@pytest.mark.parametrize("dtype", DTYPES)
def test_native_summary_k64_v64_composes(dtype: torch.dtype) -> None:
    """``[B; A]`` and ``[C; A.T]`` track FP64 for the 64-wide state the kernels also support."""
    inputs, cu = _packed_inputs([17, 65], dtype)
    q, k, value, gate = (t[..., :64].contiguous() for t in inputs[:4])
    beta = inputs[4]
    do = torch.randn_like(value)
    forward = build_cudnn_state_summaries(k, value, gate, beta, cu)
    reverse = build_cudnn_state_grad_summaries(
        q, k, value, gate, beta, do, cu, 0.125, transpose_forward_transition=True
    )
    assert forward.shape == reverse.shape == (2, 1, 128, 64)
    assert torch.equal(forward[:, :, 64:].transpose(-1, -2), reverse[:, :, 64:])
    references = []
    for precision in (torch.float64, torch.float32):
        leaves = clone_kda_inputs((q, k, value, gate, beta), dtype=precision)
        rows = []
        for start, stop in pairwise(cu.tolist()):
            sliced = tuple(t[:, start:stop] for t in leaves)
            state = torch.zeros(1, 1, 64, 64, device="cuda", dtype=precision, requires_grad=True)
            output, final = kda_reference(*sliced, state, scale=0.125)
            (bias,) = torch.autograd.grad(output, state, do[:, start:stop].to(precision))
            basis = list(sliced)
            basis[2] = torch.zeros_like(basis[2])
            _, transition = kda_reference(
                *basis, torch.eye(64, device="cuda", dtype=precision)[None, None], scale=0.125
            )
            rows.append(torch.cat((final.detach(), transition, bias), dim=-2))
        references.append(torch.cat(rows))
    actual = torch.cat((forward, reverse[:, :, :64]), dim=-2)
    assert_matches_low_precision_reference(actual, *references, "K=V=64 maps", source_dtype=dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("gate_value", [0.0, -0.002, -0.02])
def test_native_reverse_matches_fp64(dtype: torch.dtype, gate_value: float) -> None:
    """Both reverse recipes and fused maps are measured against an independent FP64 oracle."""
    inputs, cu = _packed_inputs([129], dtype, gate_value=gate_value)
    do = torch.randn_like(inputs[2])
    scale = 0.25
    maps = [
        build_cudnn_state_grad_summaries(
            *inputs, do, cu, scale, transpose_forward_transition=transpose
        )
        for transpose in (False, True)
    ]
    prepared = chunk_kda_prepare(*inputs, cu_seqlens=cu, scale=scale, autotune=False)
    backward = chunk_kda_prepare_backward(prepared.saved, do, None, scale=scale, autotune=False)
    maps.append(backward.state_grad_summaries(_bounds([[0, 129]])))
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
    high, low = references
    for label, actual in zip(("native bprop", "native C + A.T", "fused"), maps, strict=True):
        for part, rows in (("C", slice(0, 128)), ("R", slice(128, None))):
            if part == "R" and label != "fused":
                # Native R rounds the recurrent state/decay every BT16, not just at the
                # output. A uniform gate can bias those roundings in the same direction;
                # allow one source epsilon per chunk, rather than the single-round budget.
                allowance = ceildiv(inputs[0].shape[1], 16) * torch.finfo(dtype).eps
                error = (low[:, :, rows].double() - high[:, :, rows]).abs().max()
                budget = error + allowance * high[:, :, rows].abs().max()
                assert torch.isfinite(actual[:, :, rows]).all()
                assert (actual[:, :, rows].double() - high[:, :, rows]).abs().max() <= budget
            else:
                assert_matches_low_precision_reference(
                    actual[:, :, rows],
                    high[:, :, rows],
                    low[:, :, rows],
                    f"{label} {part}",
                    source_dtype=dtype,
                )


def test_native_reverse_cuda_graph_replay() -> None:
    inputs, cu = _packed_inputs([17, 0, 64])
    do = torch.randn_like(inputs[2])
    grad_summaries = partial(
        build_cudnn_state_grad_summaries, *inputs, do, cu, SCALE, transpose_forward_transition=True
    )
    expected = grad_summaries()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = grad_summaries()
    do.mul_(0.5)
    graph.replay()
    assert not torch.equal(actual, expected)
    _assert_bitwise_equal(actual, grad_summaries())
