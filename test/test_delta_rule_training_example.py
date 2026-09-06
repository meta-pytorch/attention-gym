"""Integration tests for the fused delta-rule (KDA/GDN) training example."""

from contextlib import contextmanager
from functools import partial
from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F
from torch._inductor import config as inductor_config

pytest.importorskip("cutlass")
pytest.importorskip("typer")
pytest.importorskip("torch.cuda.graph_annotations", reason="the example requires a newer torch")

from attn_gym.linear.context_parallel import ContextParallelRouting
from attn_gym.linear.kda.constants import MAX_GATE_LOWER_BOUND_MAGNITUDE
from attn_gym.testing.kda import assert_relative_rms_within
from examples import delta_rule_training
from examples.delta_rule_context_parallel import ContextParallelDeltaRuleAttention
from examples.delta_rule_training import DeltaRuleAttention, packed_sequence_metadata

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0),
    reason="the fused delta-rule example requires CUDA capability 9.0 or newer",
)


def test_example_validates_public_gate_configuration():
    make = partial(DeltaRuleAttention, hidden_size=128, num_heads=1, head_dim=128, device="cuda")
    too_low = -(MAX_GATE_LOWER_BOUND_MAGNITUDE + 1e-3)
    with pytest.raises(ValueError, match="lower_bound"):
        make(lower_bound=too_low, backend="fused")
    make(lower_bound=too_low, backend="reference")
    # GDN never reads lower_bound, so the KDA-only bound must not constrain it.
    make(variant="gdn", lower_bound=too_low, backend="fused")
    with pytest.raises(ValueError, match="finite and nonpositive"):
        make(lower_bound=1.0, backend="reference")
    with pytest.raises(ValueError, match="fastmath applies only"):
        make(fastmath=True, backend="reference")
    with pytest.raises(ValueError, match="fastmath applies only"):
        make(variant="gdn", fastmath=True, backend="fused")
    with pytest.raises(ValueError, match="variant must be"):
        make(variant="mamba")


@pytest.mark.parametrize("variant", ["kda", "gdn"])
def test_example_fused_backend_matches_reference_module(variant):
    """Same weights, dense input: the fused bf16 module must track the fp32 reference recipe."""
    torch.manual_seed(5)
    options = {"hidden_size": 32, "num_heads": 2, "head_dim": 128, "variant": variant}
    reference = DeltaRuleAttention(**options, backend="reference", device="cuda")
    with torch.no_grad():  # zero-initialized gate parameters would make the gate a constant
        reference.A_log.uniform_(-1.0, 1.0)
        reference.dt_bias.uniform_(-1.0, 1.0)
    fused = DeltaRuleAttention(**options, backend="fused", device="cuda")
    fused.load_state_dict(reference.state_dict())
    hidden = torch.randn(2, 64, 32, device="cuda")
    cotangent = torch.randn_like(hidden)

    def run(model):
        leaf = hidden.clone().requires_grad_()
        result = model(leaf, return_final_state=True)
        grads = torch.autograd.grad(result.hidden_states, (leaf, *model.parameters()), cotangent)
        return [
            ("output", result.hidden_states),
            ("final_state", result.final_state),
            ("input gradient", grads[0]),
            *zip((name for name, _ in model.named_parameters()), grads[1:], strict=True),
        ]

    # The reference runs fp32 end to end; the fused module rounds every stage input to bf16,
    # so pointwise error scales with the tensor's magnitude (observed <= 4% of max, ~1 eps RMS).
    for (name, actual), (_, expected) in zip(run(fused), run(reference), strict=True):
        budget = 5e-2 * expected.abs().max().item()
        torch.testing.assert_close(actual.float(), expected, rtol=0, atol=budget, msg=name)
        assert_relative_rms_within(actual, expected, name, max_eps=8)


def test_context_parallel_module_overrides_only_the_stateful_stages():
    """CP reuses the base pipeline: only entry validation and the two stateful stages differ."""
    overridden = {
        name
        for name, value in vars(ContextParallelDeltaRuleAttention).items()
        if callable(value) and name in vars(DeltaRuleAttention)
    }
    assert overridden == {"__init__", "forward", "short_convolution", "delta_rule_core"}
    assert not any(
        isinstance(value, ContextParallelRouting)
        for value in vars(ContextParallelDeltaRuleAttention).values()
    )


@pytest.mark.parametrize(
    ("width", "tokens", "with_initial_state"),
    [(5, 2, True), (5, 2, False)],
)
def test_example_fused_short_conv_supports_generic_width_and_state(
    width: int,
    tokens: int,
    with_initial_state: bool,
):
    """Keep the example on CuTeDSL when adapting optional convolution history."""
    torch.manual_seed(3)
    model = DeltaRuleAttention(
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


def test_example_builds_packed_sequence_metadata():
    """Keep token-level Zipf samples bounded and cumulative."""
    torch.manual_seed(0)
    lengths, offsets = packed_sequence_metadata(4, 63)
    assert len(lengths) == 4
    assert all(0 < length <= 63 for length in lengths)
    assert offsets[0] == 0
    assert tuple(end - start for start, end in pairwise(offsets)) == lengths

    with pytest.raises(ValueError, match="at least one"):
        packed_sequence_metadata(3, 0)


@pytest.mark.parametrize("variant", ["kda", "gdn"])
def test_example_marks_cuda_graph_kernel_stages(monkeypatch, variant):
    labels = []

    @contextmanager
    def record_mark_kernels(annotation, *, backward=True):
        assert backward
        labels.append(annotation)
        yield

    monkeypatch.setattr(delta_rule_training, "mark_kernels", record_mark_kernels)
    model = DeltaRuleAttention(
        hidden_size=32,
        num_heads=1,
        head_dim=128,
        variant=variant,
        backend="fused",
        device="cuda",
        enable_graph_annotations=True,
    )
    hidden = torch.randn(1, 64, 32, device="cuda")
    offsets = torch.tensor([0, 33, 64], device="cuda", dtype=torch.int32)

    model(hidden, cu_seqlens=offsets)

    assert labels == [
        f"{variant}/qkv_projection",
        f"{variant}/short_convolution",
        f"{variant}/qk_normalization",
        f"{variant}/gate_projections",
        f"{variant}/gate_activation",
        f"{variant}/core/fused",
        f"{variant}/output_normalization",
        f"{variant}/output_gate",
        f"{variant}/output_projection",
    ]


@pytest.mark.parametrize("variant", ["kda", "gdn"])
@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, torch.float16])
def test_example_packed_matches_sequence_for_loop(monkeypatch, compute_dtype, variant):
    """Match packed BF16/FP16 training against independent sequence calls."""
    fused_calls = {"short_conv": 0, "l2norm": 0, "gate": 0, "core": 0}
    core_name = "chunk_kda" if variant == "kda" else "chunk_gdn"
    short_conv = delta_rule_training.causal_conv1d
    fused_l2norm = delta_rule_training.l2norm
    gate = delta_rule_training.bound_gate
    core = getattr(delta_rule_training, core_name)

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

    monkeypatch.setattr(delta_rule_training, "causal_conv1d", record_short_conv)
    monkeypatch.setattr(delta_rule_training, "l2norm", record_l2norm)
    monkeypatch.setattr(delta_rule_training, "bound_gate", record_gate)
    monkeypatch.setattr(delta_rule_training, core_name, record_core)
    torch.manual_seed(19)
    model = DeltaRuleAttention(
        hidden_size=32,
        num_heads=1,
        head_dim=128,
        variant=variant,
        backend="fused",
        compute_dtype=compute_dtype,
        device="cuda",
    )
    offsets = (0, 65, 65, 128)
    packed_hidden = torch.randn(1, offsets[-1], 32, device="cuda", requires_grad=True)
    loop_hidden = packed_hidden.detach().clone().requires_grad_()
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)

    actual = model(packed_hidden, cu_seqlens=cu_seqlens).hidden_states
    # GDN must never route through KDA's bounded gate.
    fused_gate_calls = 1 if variant == "kda" else 0
    assert fused_calls == {"short_conv": 1, "l2norm": 2, "gate": fused_gate_calls, "core": 1}

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


def _assert_masked_run_matches(names, actual, expected, active_tokens):
    """Compare a fixed-capacity run's output, input and parameter gradients against an exact run.

    ``actual`` and ``expected`` are ``(output, input_gradient, *parameter_gradients)``; the
    actual tensors span the capacity and must be zero past ``active_tokens``.
    """
    for label, value, target in zip(("output", "input gradient"), actual[:2], expected[:2]):
        torch.testing.assert_close(
            value[:, :active_tokens], target, rtol=3e-2, atol=3e-2, msg=label
        )
        assert not value[:, active_tokens:].any(), label
    for name, value, target in zip(names, actual[2:], expected[2:], strict=True):
        torch.testing.assert_close(value, target, rtol=3e-2, atol=3e-2, msg=name)


@pytest.mark.parametrize(
    ("variant", "captured_offsets", "replayed_offsets"),
    [
        pytest.param("kda", [0, 64, 128], [0, 33, 96], id="kda-fewer-tokens"),
        pytest.param(
            "kda", [0, 48, 96, 128], [0, 33, 96, 96], id="kda-fewer-tokens-and-sequences"
        ),
        pytest.param(
            "kda",
            [0, 64, 128, 128, 128],
            [0, 33, 64, 96, 96],
            id="kda-more-sequences-within-capacity",
        ),
        # The masking is variant-agnostic; one GDN case covers its 3-D raw-gate barrier.
        pytest.param(
            "gdn", [0, 48, 96, 128], [0, 33, 96, 96], id="gdn-fewer-tokens-and-sequences"
        ),
    ],
)
def test_example_masked_capacity_matches_active_run_under_graph_replay(
    variant,
    captured_offsets,
    replayed_offsets,
):
    """Replay changing L and M over stale capacity and match an exactly sized run."""
    torch.manual_seed(31)
    capacity, hidden_size = captured_offsets[-1], 32
    model = DeltaRuleAttention(
        hidden_size=hidden_size,
        num_heads=1,
        head_dim=128,
        variant=variant,
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
    _assert_masked_run_matches(
        names,
        (captured_output, *captured_gradients),
        (expected_output, *expected_gradients),
        active_tokens,
    )


@pytest.mark.parametrize("variant", ["kda", "gdn"])
def test_example_masked_capacity_supports_strict_fullgraph_autograd(variant):
    """Compile the reusable gradient barriers through the complete example."""
    torch.manual_seed(32)
    capacity, active_tokens, hidden_size = 128, 96, 32
    model = DeltaRuleAttention(
        hidden_size=hidden_size,
        num_heads=1,
        head_dim=128,
        variant=variant,
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

    _assert_masked_run_matches(
        names,
        (actual_output, *actual_gradients),
        (expected_output, *expected_gradients),
        active_tokens,
    )
