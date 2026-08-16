"""Integration tests for the fused KDA training example."""

from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F
from torch._inductor import config as inductor_config

pytest.importorskip("cutlass")
pytest.importorskip("typer")

from examples import kda_training
from examples.kda_training import KDAAttention, packed_sequence_metadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the fused KDA example requires CUDA capability 10.0 or newer",
)


@pytest.mark.parametrize(
    ("width", "tokens", "with_initial_state"),
    [(5, 2, True), (5, 2, False)],
)
def test_kda_example_fused_short_conv_supports_generic_width_and_state(
    width: int,
    tokens: int,
    with_initial_state: bool,
):
    """Keep the example on CuTeDSL when adapting optional convolution history."""
    torch.manual_seed(3)
    model = KDAAttention(
        hidden_size=128,
        num_heads=1,
        head_dim=128,
        short_conv_kernel_size=width,
        backend="fused",
        device="cuda",
    ).to(torch.bfloat16)
    channels = model.qkv_conv1d.in_channels
    qkv = torch.randn(2, tokens, channels, device="cuda", dtype=torch.bfloat16)
    qkv_actual = qkv.clone().requires_grad_()
    qkv_expected = qkv.clone().requires_grad_()
    state = torch.randn(
        2,
        width - 1,
        channels,
        device="cuda",
        dtype=torch.bfloat16,
    )
    state_actual = state.clone().requires_grad_() if with_initial_state else None
    state_expected = state.clone().requires_grad_() if with_initial_state else None
    weight_actual = model.qkv_conv1d.weight
    weight_expected = weight_actual.detach().clone().requires_grad_()

    actual, actual_final = model.short_convolution(
        qkv_actual,
        state_actual,
        return_final_state=True,
    )
    reference_state = (
        qkv_expected.new_zeros(2, width - 1, channels)
        if state_expected is None
        else state_expected
    )
    reference_input = torch.cat((reference_state, qkv_expected), dim=1)
    expected = F.silu(
        F.conv1d(
            reference_input.transpose(1, 2),
            weight_expected,
            groups=channels,
        ).transpose(1, 2)
    )
    expected_final = reference_input[:, tokens:].clone()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_final, expected_final)

    grad_output = torch.randn_like(actual)
    grad_final = torch.randn_like(actual_final)
    actual_inputs = (qkv_actual, weight_actual)
    expected_inputs = (qkv_expected, weight_expected)
    if with_initial_state:
        actual_inputs += (state_actual,)
        expected_inputs += (state_expected,)
    actual_gradients = torch.autograd.grad(
        (actual, actual_final),
        actual_inputs,
        (grad_output, grad_final),
    )
    expected_gradients = torch.autograd.grad(
        (expected, expected_final),
        expected_inputs,
        (grad_output, grad_final),
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients,
        expected_gradients,
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-2, atol=3e-1)


def test_kda_example_builds_packed_sequence_metadata():
    """Keep token-level and optionally aligned Zipf samples bounded and cumulative."""
    torch.manual_seed(0)
    lengths, offsets = packed_sequence_metadata(4, 63, 64)
    assert len(lengths) == 4
    assert all(0 < length <= 63 and length % 64 != 0 for length in lengths)
    assert offsets[0] == 0
    assert tuple(end - start for start, end in pairwise(offsets)) == lengths

    padded_lengths, padded_offsets = packed_sequence_metadata(4, 256, 64, padded=True)
    assert all(length % 64 == 0 and 0 < length <= 256 for length in padded_lengths)
    assert tuple(end - start for start, end in pairwise(padded_offsets)) == padded_lengths

    with pytest.raises(ValueError, match="divisible"):
        packed_sequence_metadata(2, 96, 64, padded=True)
    with pytest.raises(ValueError, match="at least one"):
        packed_sequence_metadata(3, 0, 64)


def test_kda_example_packed_matches_sequence_for_loop(monkeypatch):
    """Match independent execution while sharing one packed schedule across fused stages."""
    scheduled_metadata = []
    consumed_metadata = []
    prepare_metadata = kda_training.prepare_ragged_chunk_metadata
    gate = kda_training._bounded_gate_cumsum
    core = kda_training._chunk_kda

    def record_prepare(*args):
        scheduled_metadata.append(prepare_metadata(*args))
        return scheduled_metadata[-1]

    def record_gate(*args, **kwargs):
        if kwargs.get("metadata") is not None:
            consumed_metadata.append(kwargs["metadata"])
        return gate(*args, **kwargs)

    def record_core(*args, **kwargs):
        if kwargs.get("metadata") is not None:
            consumed_metadata.append(kwargs["metadata"])
        return core(*args, **kwargs)

    monkeypatch.setattr(kda_training, "prepare_ragged_chunk_metadata", record_prepare)
    monkeypatch.setattr(kda_training, "_bounded_gate_cumsum", record_gate)
    monkeypatch.setattr(kda_training, "_chunk_kda", record_core)
    torch.manual_seed(19)
    model = KDAAttention(
        hidden_size=32,
        num_heads=1,
        head_dim=128,
        backend="fused",
        device="cuda",
    )
    offsets = (0, 65, 65, 128)
    packed_hidden = torch.randn(1, offsets[-1], 32, device="cuda", requires_grad=True)
    loop_hidden = packed_hidden.detach().clone().requires_grad_()
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)

    actual = model(packed_hidden, cu_seqlens=cu_seqlens).hidden_states
    loop_outputs = [
        model(loop_hidden[:, start:end]).hidden_states
        for start, end in pairwise(offsets)
        if start < end
    ]
    expected = torch.cat(loop_outputs, dim=1)
    torch.testing.assert_close(actual, expected)

    cotangent = torch.randn_like(actual)
    actual_gradient = torch.autograd.grad(actual, packed_hidden, cotangent)[0]
    expected_gradient = torch.autograd.grad(expected, loop_hidden, cotangent)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=3e-2, atol=3e-2)

    assert len(scheduled_metadata) == 1
    assert scheduled_metadata[0].cu_seqlens is cu_seqlens
    assert len(consumed_metadata) == 2
    assert all(metadata is scheduled_metadata[0] for metadata in consumed_metadata)


@pytest.mark.parametrize(
    ("captured_offsets", "replayed_offsets"),
    [
        pytest.param([0, 64, 128], [0, 33, 96], id="fewer-tokens"),
        pytest.param([0, 48, 96, 128], [0, 33, 96, 96], id="fewer-tokens-and-sequences"),
    ],
)
def test_kda_example_masked_capacity_matches_active_run_under_graph_replay(
    captured_offsets,
    replayed_offsets,
):
    """Replay a shorter active length over a stale suffix and match an exactly sized run.

    The second case drops a whole sequence at replay time by repeating the active endpoint
    in the cu_seqlens tail (capture N sequences, replay M<N).
    """
    torch.manual_seed(31)
    capacity, hidden_size = captured_offsets[-1], 32
    model = KDAAttention(
        hidden_size=hidden_size,
        num_heads=1,
        head_dim=128,
        backend="fused",
        device="cuda",
        mask_inactive_capacity=True,
    )
    names, parameters = zip(*model.named_parameters(), strict=True)
    hidden = torch.randn(1, capacity, hidden_size, device="cuda", requires_grad=True)
    cotangent = torch.randn(1, capacity, hidden_size, device="cuda")
    cu_seqlens = torch.tensor(captured_offsets, device="cuda", dtype=torch.int32)

    def run(states, offsets, grad_output):
        output = model(states, cu_seqlens=offsets).hidden_states
        return output, torch.autograd.grad(output, (states, *parameters), grad_output)

    run(hidden, cu_seqlens, cotangent)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output, captured_gradients = run(hidden, cu_seqlens, cotangent)

    active_tokens = replayed_offsets[-1]
    # Drop repeated tail endpoints to build the exactly sized comparison offsets.
    trimmed_offsets = list(replayed_offsets)
    while len(trimmed_offsets) > 2 and trimmed_offsets[-2] == trimmed_offsets[-1]:
        trimmed_offsets.pop()
    active_offsets = torch.tensor(trimmed_offsets, device="cuda", dtype=torch.int32)
    with torch.no_grad():
        cu_seqlens.copy_(torch.tensor(replayed_offsets, device="cuda", dtype=torch.int32))
        hidden[:, active_tokens:].fill_(float("nan"))
        cotangent[:, active_tokens:].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()

    expected_output, expected_gradients = run(
        hidden[:, :active_tokens].detach().clone().requires_grad_(),
        active_offsets,
        cotangent[:, :active_tokens],
    )
    torch.testing.assert_close(
        captured_output[:, :active_tokens], expected_output, rtol=3e-2, atol=3e-2
    )
    assert not captured_output[:, active_tokens:].any()
    torch.testing.assert_close(
        captured_gradients[0][:, :active_tokens], expected_gradients[0], rtol=3e-2, atol=3e-2
    )
    assert not captured_gradients[0][:, active_tokens:].any()
    for name, actual_gradient, expected_gradient in zip(
        names,
        captured_gradients[1:],
        expected_gradients[1:],
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=3e-2,
            atol=3e-2,
            msg=lambda message, parameter=name: f"{parameter}: {message}",
        )


def test_kda_example_masked_capacity_supports_strict_fullgraph_autograd():
    """Compile the reusable gradient barriers through the complete example."""
    torch.manual_seed(32)
    capacity, active_tokens, hidden_size = 128, 96, 32
    model = KDAAttention(
        hidden_size=hidden_size,
        num_heads=1,
        head_dim=128,
        backend="fused",
        device="cuda",
        mask_inactive_capacity=True,
    )
    names, parameters = zip(*model.named_parameters(), strict=True)
    hidden = torch.randn(1, capacity, hidden_size, device="cuda")
    hidden[:, active_tokens:] = float("nan")
    hidden.requires_grad_()
    cotangent = torch.randn_like(hidden)
    cotangent[:, active_tokens:].fill_(float("nan"))
    offsets = torch.tensor([0, 33, active_tokens], device="cuda", dtype=torch.int32)

    expected_hidden = hidden[:, :active_tokens].detach().clone().requires_grad_()
    expected_output = model(expected_hidden, cu_seqlens=offsets).hidden_states
    expected_gradients = torch.autograd.grad(
        expected_output,
        (expected_hidden, *parameters),
        cotangent[:, :active_tokens],
    )

    with inductor_config.patch("triton.cudagraph_or_error", True):
        compiled = torch.compile(model, fullgraph=True, mode="reduce-overhead")
        actual_output = compiled(hidden, cu_seqlens=offsets).hidden_states
        actual_gradients = torch.autograd.grad(
            actual_output,
            (hidden, *parameters),
            cotangent,
        )

    torch.testing.assert_close(
        actual_output[:, :active_tokens], expected_output, rtol=3e-2, atol=3e-2
    )
    assert not actual_output[:, active_tokens:].any()
    torch.testing.assert_close(
        actual_gradients[0][:, :active_tokens],
        expected_gradients[0],
        rtol=3e-2,
        atol=3e-2,
    )
    assert not actual_gradients[0][:, active_tokens:].any()
    for name, actual_gradient, expected_gradient in zip(
        names,
        actual_gradients[1:],
        expected_gradients[1:],
        strict=True,
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            rtol=3e-2,
            atol=3e-2,
            msg=lambda message, parameter=name: f"{parameter}: {message}",
        )
