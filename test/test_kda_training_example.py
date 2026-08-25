"""Integration tests for the fused KDA training example."""

from contextlib import contextmanager
from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F
from torch._inductor import config as inductor_config

pytest.importorskip("cutlass")
pytest.importorskip("typer")

from attn_gym.linear.kda.constants import MAX_GATE_LOWER_BOUND_MAGNITUDE
from examples import kda_training
from examples.kda_training import KDAAttention, packed_sequence_metadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the fused KDA example requires CUDA capability 10.0 or newer",
)


def test_kda_example_validates_public_gate_configuration():
    with pytest.raises(ValueError, match="lower_bound"):
        KDAAttention(
            hidden_size=128,
            num_heads=1,
            head_dim=128,
            lower_bound=-(MAX_GATE_LOWER_BOUND_MAGNITUDE + 1e-3),
            backend="fused",
            device="cuda",
        )
    KDAAttention(
        hidden_size=128,
        num_heads=1,
        head_dim=128,
        lower_bound=-(MAX_GATE_LOWER_BOUND_MAGNITUDE + 1e-3),
        backend="reference",
        device="cuda",
    )
    with pytest.raises(ValueError, match="finite and nonpositive"):
        KDAAttention(
            hidden_size=128,
            num_heads=1,
            head_dim=128,
            lower_bound=1.0,
            backend="reference",
            device="cuda",
        )
    with pytest.raises(ValueError, match="fastmath applies only"):
        KDAAttention(
            hidden_size=128,
            num_heads=1,
            head_dim=128,
            fastmath=True,
            backend="reference",
            device="cuda",
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
    """Keep token-level Zipf samples bounded and cumulative."""
    torch.manual_seed(0)
    lengths, offsets = packed_sequence_metadata(4, 63)
    assert len(lengths) == 4
    assert all(0 < length <= 63 for length in lengths)
    assert offsets[0] == 0
    assert tuple(end - start for start, end in pairwise(offsets)) == lengths

    with pytest.raises(ValueError, match="at least one"):
        packed_sequence_metadata(3, 0)


def test_kda_example_marks_cuda_graph_kernel_stages(monkeypatch):
    labels = []

    @contextmanager
    def record_mark_kernels(annotation, *, backward=True):
        assert backward
        labels.append(annotation)
        yield

    monkeypatch.setattr(kda_training, "mark_kernels", record_mark_kernels)
    model = KDAAttention(
        hidden_size=32,
        num_heads=1,
        head_dim=128,
        backend="fused",
        device="cuda",
        enable_graph_annotations=True,
    )
    hidden = torch.randn(1, 64, 32, device="cuda")
    offsets = torch.tensor([0, 33, 64], device="cuda", dtype=torch.int32)

    model(hidden, cu_seqlens=offsets)

    assert labels == [
        "kda/qkv_projection",
        "kda/short_convolution",
        "kda/qk_normalization",
        "kda/gate_projections",
        "kda/gate_activation",
        "kda/core/fused",
        "kda/output_normalization",
        "kda/output_gate",
        "kda/output_projection",
    ]


def test_kda_example_packed_matches_sequence_for_loop(monkeypatch):
    """Match one packed execution against independent sequence calls."""
    fused_calls = {"short_conv": 0, "l2norm": 0, "gate": 0, "core": 0}
    short_conv = kda_training.causal_conv1d
    fused_l2norm = kda_training.l2norm
    gate = kda_training.bound_gate
    core = kda_training.chunk_kda

    def record_short_conv(*args, **kwargs):
        fused_calls["short_conv"] += 1
        return short_conv(*args, **kwargs)

    def record_l2norm(*args, **kwargs):
        fused_calls["l2norm"] += 1
        return fused_l2norm(*args, **kwargs)

    def record_gate(*args, **kwargs):
        fused_calls["gate"] += 1
        return gate(*args, **kwargs)

    def record_core(*args, **kwargs):
        fused_calls["core"] += 1
        return core(*args, **kwargs)

    monkeypatch.setattr(kda_training, "causal_conv1d", record_short_conv)
    monkeypatch.setattr(kda_training, "l2norm", record_l2norm)
    monkeypatch.setattr(kda_training, "bound_gate", record_gate)
    monkeypatch.setattr(kda_training, "chunk_kda", record_core)
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
    assert fused_calls == {"short_conv": 1, "l2norm": 2, "gate": 1, "core": 1}

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


@pytest.mark.parametrize(
    ("captured_offsets", "replayed_offsets"),
    [
        pytest.param([0, 64, 128], [0, 33, 96], id="fewer-tokens"),
        pytest.param([0, 48, 96, 128], [0, 33, 96, 96], id="fewer-tokens-and-sequences"),
        pytest.param(
            [0, 64, 128, 128, 128],
            [0, 33, 64, 96, 96],
            id="more-sequences-within-capacity",
        ),
    ],
)
def test_kda_example_masked_capacity_matches_active_run_under_graph_replay(
    captured_offsets,
    replayed_offsets,
):
    """Replay changing L and M over stale capacity and match an exactly sized run."""
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

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        run(hidden, cu_seqlens, cotangent)
    capture_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
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

    with torch.cuda.stream(capture_stream):
        expected_output, expected_gradients = run(
            hidden[:, :active_tokens].detach().clone().requires_grad_(),
            active_offsets,
            cotangent[:, :active_tokens],
        )
    capture_stream.synchronize()
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
