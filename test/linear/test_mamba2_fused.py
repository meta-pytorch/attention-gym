import math
from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear import chunk_mamba2
from attn_gym.linear.mamba2.api import NO_DT_LIMIT, reference_mamba2
from attn_gym.linear.mamba2.ops import chunk_bwd_op, chunk_fwd_op, chunk_fwd_with_state_op
from attn_gym.linear.mamba2.reference import chunk_forward
from attn_gym.testing.kda import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def make_inputs(batch, sequence, heads, groups, head_dim, state_dim, *, dtype, seed=0):
    """BF16/FP16 activations with Mamba2-like dt and A statistics."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.float32):
        return torch.randn(*shape, generator=generator, device="cuda").to(dtype)

    return {
        "x": randn(batch, sequence, heads, head_dim, dtype=dtype),
        "dt": randn(batch, sequence, heads) - 2.0,
        "A": -torch.exp(torch.rand(heads, generator=generator, device="cuda") * 2.77),
        "B": randn(batch, sequence, groups, state_dim, dtype=dtype),
        "C": randn(batch, sequence, groups, state_dim, dtype=dtype),
        "D": randn(heads),
        "z": randn(batch, sequence, heads, head_dim, dtype=dtype),
        "dt_bias": randn(heads) * 0.5,
    }


def assert_close_to_reference(actual, expected, tolerance):
    """Relative L2 and max-abs budgets scaled by the reference magnitude."""
    error = (actual.double() - expected.double()).norm() / expected.double().norm()
    assert error < tolerance, f"relative L2 error {error:.3e} >= {tolerance:.1e}"
    scale = expected.double().abs().max()
    max_error = (actual.double() - expected.double()).abs().max() / scale
    assert max_error < 10 * tolerance, f"normalized max error {max_error:.3e}"


def reference(inputs, **kwargs):
    """FP64 reference on the same rounded activations."""
    upcast = {name: tensor.double() for name, tensor in inputs.items()}
    return chunk_mamba2(**upcast, impl="reference", **kwargs)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    ("batch", "sequence", "heads", "groups", "head_dim", "state_dim"),
    [
        (2, 64, 4, 1, 64, 128),
        (1, 200, 8, 2, 64, 128),
        (3, 77, 6, 3, 32, 64),
        (1, 129, 4, 4, 80, 16),
        (2, 90, 4, 2, 16, 32),
    ],
)
def test_fused_matches_reference(dtype, batch, sequence, heads, groups, head_dim, state_dim):
    inputs = make_inputs(batch, sequence, heads, groups, head_dim, state_dim, dtype=dtype)
    initial_state = torch.randn(batch, heads, head_dim, state_dim, device="cuda")
    kwargs = {"dt_softplus": True, "output_final_state": True, "initial_state": initial_state}
    with torch.no_grad():
        output, state = chunk_mamba2(**inputs, impl="fused", **kwargs)
    expected, expected_state = reference(inputs, **kwargs)
    assert output.dtype == dtype and state.dtype == torch.float32
    assert_close_to_reference(output, expected, 1e-2)
    assert_close_to_reference(state, expected_state, 1e-2)


@pytest.mark.parametrize("optional", ["none", "per_channel_D", "dt_limit"])
def test_fused_optional_terms(optional):
    inputs = make_inputs(2, 150, 4, 2, 64, 64, dtype=torch.bfloat16)
    kwargs = {"dt_softplus": True}
    match optional:
        case "none":
            inputs.pop("D"), inputs.pop("z"), inputs.pop("dt_bias")
            inputs["dt"] = F.softplus(inputs["dt"])
            kwargs = {}
        case "per_channel_D":
            inputs["D"] = torch.randn(4, 64, device="cuda")
        case "dt_limit":
            kwargs["dt_limit"] = (0.01, 0.08)
    with torch.no_grad():
        output, _ = chunk_mamba2(**inputs, impl="fused", **kwargs)
    assert_close_to_reference(output, reference(inputs, **kwargs)[0], 1e-2)


def test_fused_packed_sequences():
    lengths = [100, 0, 1, 64, 300]
    capacity = sum(lengths) + 5
    inputs = make_inputs(1, capacity, 8, 2, 64, 128, dtype=torch.bfloat16)
    cu_seqlens = cumulative_sequence_offsets(lengths)
    states = torch.randn(len(lengths), 8, 64, 128, device="cuda")
    kwargs = {
        "dt_softplus": True,
        "cu_seqlens": cu_seqlens,
        "initial_state": states,
        "output_final_state": True,
    }
    with torch.no_grad():
        output, state = chunk_mamba2(**inputs, impl="fused", **kwargs)
    expected, expected_state = reference(inputs, **kwargs)
    active = int(cu_seqlens[-1])
    assert_close_to_reference(output[:, :active], expected[:, :active], 1e-2)
    assert_close_to_reference(state, expected_state, 1e-2)
    torch.testing.assert_close(state[1], states[1])


# BF16 rounds each matmul operand with unit roundoff 2**-8 (RMS ~2.3e-3 relative), and every
# gradient composes at most four such roundings with FP32 accumulation, so ~4e-3 relative L2 is
# expected; observed errors are 1e-3 to 3.5e-3. The budget leaves about 3x headroom.
GRADIENT_TOLERANCE = 1e-2


def reference_gradients(inputs, cotangents, kwargs):
    """FP64 reference gradients, plus the per-token ``dA`` terms.

    The reference core reads one ``A`` leaf per token, so ``dA`` comes with the magnitudes of the
    terms it sums.
    """
    leaves = {name: tensor.detach().double().requires_grad_() for name, tensor in inputs.items()}
    A_tokens = leaves["A"].detach().expand(inputs["dt"].shape).clone().requires_grad_()
    cu_seqlens = kwargs.get("cu_seqlens")
    offsets = [0, inputs["x"].shape[1]] if cu_seqlens is None else cu_seqlens.tolist()
    # Packed references call the core once per nonempty sequence, in order.
    spans = iter([(begin, end) for begin, end in pairwise(offsets) if begin < end])

    def core(x, dt, A, B, C, **core_kwargs):
        begin, end = next(spans)
        return chunk_forward(x, dt, A_tokens[:, begin:end], B, C, **core_kwargs)

    outputs = reference_mamba2(
        core,
        *(leaves.get(name) for name in ("x", "dt", "A", "B", "C", "D", "z", "initial_state")),
        dt_bias=leaves.get("dt_bias"),
        dt_softplus=kwargs.get("dt_softplus", False),
        dt_limit=kwargs.get("dt_limit", NO_DT_LIMIT),
        cu_seqlens=cu_seqlens,
        output_final_state=kwargs.get("output_final_state", False),
    )
    names = [name for name in leaves if name != "A"]
    gradients = torch.autograd.grad(
        outputs[: len(cotangents)],
        [leaves[name] for name in names] + [A_tokens],
        [cotangent.double() for cotangent in cotangents],
    )
    return dict(zip(names, gradients[:-1])) | {"A": gradients[-1].sum((0, 1))}, gradients[-1]


def assert_gradients_match_reference(inputs, tolerance=GRADIENT_TOLERANCE, **kwargs):
    """Backpropagate random output and final-state cotangents through both implementations.

    ``dA`` and ``ddt_bias`` sum per-token terms that cancel, so their error is measured against
    the summed term magnitudes instead of the result.
    """
    leaves = {name: tensor.detach().clone().requires_grad_() for name, tensor in inputs.items()}
    output, state = chunk_mamba2(**leaves, impl="fused", **kwargs)
    generator = torch.Generator(device="cuda").manual_seed(1)
    cotangents = [torch.randn(output.shape, generator=generator, device="cuda").to(output.dtype)]
    if (cu_seqlens := kwargs.get("cu_seqlens")) is not None:
        # Output past the terminal offset is unspecified, so no loss may depend on it.
        cotangents[0][:, cu_seqlens[-1] :] = 0
    if state is not None:
        cotangents.append(torch.randn(state.shape, generator=generator, device="cuda"))
    actual = torch.autograd.grad(
        [output, state][: len(cotangents)], list(leaves.values()), cotangents
    )
    actual = dict(zip(leaves, actual))
    expected, dA_terms = reference_gradients(inputs, cotangents, kwargs)
    term_scales = {"A": dA_terms.abs().sum((0, 1)), "dt_bias": expected["dt"].abs().sum((0, 1))}

    failures = []
    for name, actual_grad in actual.items():
        assert actual_grad.dtype == inputs[name].dtype, name
        try:
            if name in term_scales:
                error = (actual_grad.double() - expected[name]).abs() / term_scales[name]
                assert error.max() < tolerance, f"term-relative error {error.max():.3e}"
            else:
                assert_close_to_reference(actual_grad, expected[name], tolerance)
        except AssertionError as error:
            failures.append(f"d{name}: {error}")
    assert not failures, "; ".join(failures)
    return actual


def random_state(*shape):
    return torch.randn(
        *shape, generator=torch.Generator(device="cuda").manual_seed(2), device="cuda"
    )


@pytest.mark.parametrize(
    ("batch", "sequence", "heads", "groups", "head_dim", "state_dim", "dtype"),
    [
        (2, 1000, 8, 2, 64, 128, torch.bfloat16),
        (2, 200, 4, 1, 64, 64, torch.float16),
        (1, 129, 8, 4, 80, 128, torch.bfloat16),
        (3, 77, 6, 6, 32, 16, torch.bfloat16),
        (1, 100, 2, 1, 64, 256, torch.bfloat16),
        (2, 90, 4, 2, 16, 32, torch.bfloat16),
    ],
)
def test_fused_gradients_match_reference(
    batch, sequence, heads, groups, head_dim, state_dim, dtype
):
    inputs = make_inputs(batch, sequence, heads, groups, head_dim, state_dim, dtype=dtype)
    inputs["initial_state"] = random_state(batch, heads, head_dim, state_dim)
    assert_gradients_match_reference(inputs, dt_softplus=True, output_final_state=True)


@pytest.mark.parametrize("optional", ["none", "per_channel_D", "dt_limit", "no_final_state"])
def test_fused_gradient_optional_terms(optional):
    inputs = make_inputs(2, 150, 4, 2, 64, 64, dtype=torch.bfloat16)
    inputs["initial_state"] = random_state(2, 4, 64, 64)
    kwargs = {"dt_softplus": True, "output_final_state": True}
    match optional:
        case "none":
            for name in ("D", "z", "dt_bias", "initial_state"):
                inputs.pop(name)
            inputs["dt"] = F.softplus(inputs["dt"])
            kwargs = {}
        case "per_channel_D":
            inputs["D"] = random_state(4, 64)
        case "dt_limit":
            # Clamps ~19% of tokens at the lower bound and ~47% at the upper bound, which zeroes
            # their dt gradients.
            kwargs["dt_limit"] = (0.06, 0.15)
        case "no_final_state":
            kwargs["output_final_state"] = False
    assert_gradients_match_reference(inputs, **kwargs)


def test_fused_packed_gradients():
    lengths = [100, 0, 1, 64, 300, 0]
    capacity = sum(lengths) + 5
    inputs = make_inputs(1, capacity, 8, 4, 80, 64, dtype=torch.bfloat16)
    inputs["initial_state"] = random_state(len(lengths), 8, 80, 64)
    cu_seqlens = cumulative_sequence_offsets(lengths)
    gradients = assert_gradients_match_reference(
        inputs, dt_softplus=True, cu_seqlens=cu_seqlens, output_final_state=True
    )
    # Tokens past the terminal offset belong to no sequence.
    for gradient in (gradients["x"], gradients["dt"], gradients["B"], gradients["z"]):
        assert not gradient[:, sum(lengths) :].any()


SEGMENT_LENGTH = 256


@pytest.mark.parametrize("packed", [False, True])
def test_fused_segmented_scans_match_reference(packed, monkeypatch):
    """Force 256-token scan segments, whose summaries feed later segments in both directions."""

    def plan(num_programs, tokens, device):
        return math.ceil(tokens / SEGMENT_LENGTH), SEGMENT_LENGTH

    for module in ("chunk_fwd", "chunk_bwd"):
        monkeypatch.setattr(f"attn_gym.linear.mamba2.triton.{module}.segment_plan", plan)
    # Packed lengths cover a partial last segment, an exact multiple, a tiny, and an empty one.
    lengths = [700, 0, 2 * SEGMENT_LENGTH, 1, 300] if packed else None
    batch, tokens = (1, sum(lengths) + 7) if packed else (2, 900)
    states = len(lengths) if packed else batch
    inputs = make_inputs(batch, tokens, 4, 2, 64, 64, dtype=torch.bfloat16)
    inputs["initial_state"] = random_state(states, 4, 64, 64)
    kwargs = {"dt_softplus": True, "output_final_state": True}
    if packed:
        kwargs["cu_seqlens"] = cumulative_sequence_offsets(lengths)
    active = sum(lengths) if packed else tokens
    expected, expected_state = reference(inputs, **kwargs)
    # Inference gates in place; training keeps the pre-gate output and gates into y.
    for grad in (False, True):
        leaves = {name: tensor.detach().requires_grad_(grad) for name, tensor in inputs.items()}
        with torch.set_grad_enabled(grad):
            output, state = chunk_mamba2(**leaves, impl="fused", **kwargs)
        assert_close_to_reference(output[:, :active], expected[:, :active], 1e-2)
        assert_close_to_reference(state, expected_state, 1e-2)
    assert_gradients_match_reference(inputs, **kwargs)


@pytest.mark.parametrize("packed", [False, True])
def test_fused_operator_registration(packed):
    inputs = make_inputs(1, 100, 4, 2, 80, 32, dtype=torch.bfloat16)
    # A token-innermost layout checks that the fakes match the kernels' contiguous outputs.
    inputs["x"] = inputs["x"].movedim(1, -1).contiguous().movedim(-1, 1)
    inputs["D"] = torch.randn(80, 4, device="cuda").t()
    # AOT dispatch compares every output element, so packed offsets cover all tokens.
    offsets = [0, 37, 37, 100] if packed else [0, 100]
    initial_state = torch.randn(len(offsets) - 1, 4, 80, 32, device="cuda")
    cu_seqlens = torch.tensor(offsets, dtype=torch.int32, device="cuda") if packed else None
    args = (
        *(inputs[name] for name in ("x", "dt", "A", "B", "C", "D", "z")),
        initial_state,
        inputs["dt_bias"],
        cu_seqlens,
    )
    transform = (True, 0.0, math.inf)
    test_utils = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
    for op in (chunk_fwd_op, chunk_fwd_with_state_op):
        for store_pre_gate in (False, True):
            torch.library.opcheck(op, (*args, *transform, store_pre_gate), test_utils=test_utils)
    pre_gate = chunk_fwd_op(*args, *transform, True)[1]
    d_output = torch.randn_like(inputs["x"])
    d_final_state = torch.randn_like(initial_state)
    torch.library.opcheck(
        chunk_bwd_op,
        (*args, pre_gate, d_output, d_final_state, *transform),
        test_utils=test_utils,
    )


def test_fused_compiles_fullgraph():
    inputs = make_inputs(2, 150, 4, 2, 64, 64, dtype=torch.bfloat16)
    inputs["initial_state"] = torch.randn(2, 4, 64, 64, device="cuda")
    eager = {name: tensor.detach().clone().requires_grad_() for name, tensor in inputs.items()}
    compiled = {name: tensor.detach().clone().requires_grad_() for name, tensor in inputs.items()}
    kwargs = {"dt_softplus": True, "output_final_state": True, "impl": "fused"}

    def loss(output, state):
        return output.float().square().sum() + state.sin().sum()

    loss(*chunk_mamba2(**eager, **kwargs)).backward()
    loss(*torch.compile(chunk_mamba2, fullgraph=True)(**compiled, **kwargs)).backward()
    for name in inputs:
        torch.testing.assert_close(compiled[name].grad, eager[name].grad, rtol=0, atol=0)
