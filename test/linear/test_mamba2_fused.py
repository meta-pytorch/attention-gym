import math

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounterWithBackend

from attn_gym.linear import chunk_mamba2
from attn_gym.linear.mamba2.ops import chunk_bwd_op, chunk_fwd_op, chunk_fwd_with_state_op
from attn_gym.linear.mamba2.reference import chunk_forward
from attn_gym.testing.kda import cumulative_sequence_offsets
from attn_gym.testing.mamba2 import (
    fp64_reference,
    make_mamba2_inputs,
    reference_with_token_decays,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def assert_close_to_reference(actual, expected, tolerance):
    """Relative L2 and max-abs budgets scaled by the reference magnitude."""
    error = (actual.double() - expected.double()).norm() / expected.double().norm()
    assert error < tolerance, f"relative L2 error {error:.3e} >= {tolerance:.1e}"
    scale = expected.double().abs().max()
    max_error = (actual.double() - expected.double()).abs().max() / scale
    assert max_error < 10 * tolerance, f"normalized max error {max_error:.3e}"


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
    inputs = make_mamba2_inputs(
        batch,
        sequence,
        heads=heads,
        groups=groups,
        head_dim=head_dim,
        state_dim=state_dim,
        dtype=dtype,
    )
    initial_state = torch.randn(batch, heads, head_dim, state_dim, device="cuda")
    kwargs = {"dt_softplus": True, "output_final_state": True, "initial_state": initial_state}
    with torch.no_grad():
        output, state = chunk_mamba2(**inputs, impl="fused", **kwargs)
    expected, expected_state = fp64_reference(inputs, **kwargs)
    assert output.dtype == dtype and state.dtype == torch.float32
    assert_close_to_reference(output, expected, 1e-2)
    assert_close_to_reference(state, expected_state, 1e-2)


@pytest.mark.parametrize("optional", ["none", "per_channel_D", "dt_limit"])
def test_fused_optional_terms(optional):
    inputs = make_mamba2_inputs(2, 150)
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
    assert_close_to_reference(output, fp64_reference(inputs, **kwargs)[0], 1e-2)


def test_fused_packed_sequences():
    lengths = [100, 0, 1, 64, 300]
    capacity = sum(lengths) + 5
    inputs = make_mamba2_inputs(1, capacity, heads=8, state_dim=128)
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
    expected, expected_state = fp64_reference(inputs, **kwargs)
    active = int(cu_seqlens[-1])
    assert_close_to_reference(output[:, :active], expected[:, :active], 1e-2)
    assert_close_to_reference(state, expected_state, 1e-2)
    torch.testing.assert_close(state[1], states[1])


# BF16 rounds each matmul operand with unit roundoff 2**-8 (RMS ~2.3e-3 relative), and every
# gradient composes at most four such roundings with FP32 accumulation, so ~4e-3 relative L2 is
# expected; observed errors are 1e-3 to 3.5e-3. The budget leaves about 3x headroom.
GRADIENT_TOLERANCE = 1e-2


def reference_gradients(inputs, cotangents, kwargs):
    """FP64 reference gradients, plus the per-token ``dA`` terms."""
    leaves = {name: tensor.detach().double().requires_grad_() for name, tensor in inputs.items()}
    outputs, A_tokens = reference_with_token_decays(chunk_forward, leaves, **kwargs)
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
    inputs = make_mamba2_inputs(
        batch,
        sequence,
        heads=heads,
        groups=groups,
        head_dim=head_dim,
        state_dim=state_dim,
        dtype=dtype,
    )
    inputs["initial_state"] = random_state(batch, heads, head_dim, state_dim)
    assert_gradients_match_reference(inputs, dt_softplus=True, output_final_state=True)


@pytest.mark.parametrize("optional", ["none", "per_channel_D", "dt_limit", "no_final_state"])
def test_fused_gradient_optional_terms(optional):
    inputs = make_mamba2_inputs(2, 150)
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
    inputs = make_mamba2_inputs(1, capacity, heads=8, groups=4, head_dim=80)
    inputs["initial_state"] = random_state(len(lengths), 8, 80, 64)
    cu_seqlens = cumulative_sequence_offsets(lengths)
    gradients = assert_gradients_match_reference(
        inputs, dt_softplus=True, cu_seqlens=cu_seqlens, output_final_state=True
    )
    # Tokens past the terminal offset belong to no sequence.
    for gradient in (gradients["x"], gradients["dt"], gradients["B"], gradients["z"]):
        assert not gradient[:, sum(lengths) :].any()


@pytest.mark.parametrize("packed", [False, True])
def test_fused_operator_registration(packed):
    inputs = make_mamba2_inputs(1, 100, head_dim=80, state_dim=32)
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
        torch.library.opcheck(op, (*args, *transform), test_utils=test_utils)
    d_output = torch.randn_like(inputs["x"])
    d_final_state = torch.randn_like(initial_state)
    torch.library.opcheck(
        chunk_bwd_op, (*args, d_output, d_final_state, *transform), test_utils=test_utils
    )


def test_fused_compiles_fullgraph():
    inputs = make_mamba2_inputs(2, 150)
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


def packed_training_step(tokens, num_sequences, heads=16, groups=1, initial_state=False):
    """Gated packed forward and backward over ``num_sequences`` near-equal sequences."""
    lengths = [tokens // num_sequences] * num_sequences
    lengths[0] += tokens % num_sequences
    inputs = make_mamba2_inputs(1, tokens, heads=heads, groups=groups)
    if initial_state:
        inputs["initial_state"] = torch.randn(num_sequences, heads, 64, 64, device="cuda")
    leaves = {name: tensor.requires_grad_() for name, tensor in inputs.items()}
    cu_seqlens = cumulative_sequence_offsets(lengths)
    output, _ = chunk_mamba2(**leaves, dt_softplus=True, cu_seqlens=cu_seqlens)
    output.float().square().sum().backward()
    torch.cuda.synchronize()


# 3584 and 6528 tokens of 8 sequences fill 64 and 110 chunk slots, one power-of-two bucket.
SLOT_BUCKET_TOKENS = (3584, 5000, 6528)


def test_fused_training_reuses_kernels_within_a_slot_bucket(
    mamba2_first_candidate, triton_compiles
):
    """Capacities in one power-of-two bucket of chunk slots launch one chunk-program layout."""
    packed_training_step(SLOT_BUCKET_TOKENS[0], 8)
    first = dict(triton_compiles)
    for tokens in SLOT_BUCKET_TOKENS[1:]:
        packed_training_step(tokens, 8)
    assert triton_compiles == first


def test_fused_training_reuses_tuning_within_a_slot_bucket(isolated_winners):
    """New capacities in one bucket of chunk slots, and sequence counts in one bucket of
    ``(sequence, head)`` programs, re-tune nothing."""
    packed_training_step(SLOT_BUCKET_TOKENS[0], 8)
    tuned = len(isolated_winners._WINNERS)
    for tokens in SLOT_BUCKET_TOKENS[1:]:
        packed_training_step(tokens, 8)
    # 5-7 sequences of 4096 tokens fill 65-70 chunk slots and 80-112 programs with 16 heads.
    for num_sequences in (5, 6, 7):
        packed_training_step(4096, num_sequences)
    assert len(isolated_winners._WINNERS) == tuned


def test_fused_dense_training_reuses_kernels_across_shapes(
    mamba2_first_candidate, triton_compiles
):
    """Batch and token counts in every integer class launch the dense kernels compiled first."""
    # One head per group keeps one head per chunk program on any GPU.
    for step, (batch, tokens) in enumerate(((1, 1), (2, 16), (3, 37), (4, 48), (2, 100))):
        inputs = make_mamba2_inputs(batch, tokens, heads=16, groups=16)
        leaves = {name: tensor.requires_grad_() for name, tensor in inputs.items()}
        output, _ = chunk_mamba2(**leaves, dt_softplus=True)
        output.float().square().sum().backward()
        torch.cuda.synchronize()
        if step == 0:
            first = dict(triton_compiles)
    assert first and triton_compiles == first


def test_fused_rejects_more_heads_than_the_grid_holds():
    inputs = make_mamba2_inputs(1, 1, heads=65536, groups=65536, head_dim=16, state_dim=16)
    with pytest.raises(ValueError, match="at most 65535 heads"):
        chunk_mamba2(**inputs)


def test_fused_compiles_once_across_lengths(fresh_compile_cache):
    """``dynamic=True`` traces dense and packed training once for every later length."""
    # Lengths and sequence counts avoid the model sizes, which duck sizing would tie them to.
    shapes = ((100, [60, 40]), (37, [10, 20, 0, 7]), (257, [100, 0, 50, 7, 90, 10]))
    for packed in (False, True):
        counter = CompileCounterWithBackend("aot_eager")
        compiled = torch.compile(chunk_mamba2, backend=counter, dynamic=True, fullgraph=True)
        for step, (tokens, lengths) in enumerate(shapes):
            inputs = make_mamba2_inputs(1 if packed else 2, tokens)
            leaves = {name: tensor.requires_grad_() for name, tensor in inputs.items()}
            cu_seqlens = cumulative_sequence_offsets(lengths) if packed else None
            output, _ = compiled(**leaves, dt_softplus=True, cu_seqlens=cu_seqlens)
            output.float().square().sum().backward()
            if step == 0:
                traced = counter.frame_count
        assert counter.frame_count == traced
