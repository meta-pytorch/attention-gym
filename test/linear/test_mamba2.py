from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear import Impl
from attn_gym.linear import chunk_mamba2 as _chunk_mamba2
from attn_gym.linear import recurrent_mamba2 as _recurrent_mamba2
from attn_gym.testing.kda import cumulative_sequence_offsets


def chunk_mamba2(*args, **kwargs):
    kwargs.setdefault("impl", Impl.REFERENCE)
    return _chunk_mamba2(*args, **kwargs)


def recurrent_mamba2(*args, **kwargs):
    kwargs.setdefault("impl", Impl.REFERENCE)
    return _recurrent_mamba2(*args, **kwargs)


REFERENCE_OPS = [recurrent_mamba2, chunk_mamba2]


def make_inputs(sequence, *, batch=2, heads=4, groups=2, head_dim=6, state_dim=5, seed=0):
    """Mamba2-like inputs: positive dt, negative A, and per-group B/C."""
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=generator, dtype=torch.float64)

    return {
        "x": randn(batch, sequence, heads, head_dim),
        "dt": F.softplus(randn(batch, sequence, heads) - 1.0),
        "A": -torch.exp(randn(heads)),
        "B": randn(batch, sequence, groups, state_dim),
        "C": randn(batch, sequence, groups, state_dim),
        "D": randn(heads),
        "z": randn(batch, sequence, heads, head_dim),
        "initial_state": randn(batch, heads, head_dim, state_dim),
    }


def masked_attention_oracle(x, dt, A, B, C, D, z, initial_state):
    """The quadratic SSD dual: causal attention with decay mask, plus the decayed entry state."""
    heads = x.shape[2]
    B, C = (tensor.repeat_interleave(heads // tensor.shape[2], dim=2) for tensor in (B, C))
    log_decay = (dt * A).cumsum(1)
    causal = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool).tril()
    segment = log_decay[:, :, None] - log_decay[:, None, :]
    mask = torch.where(causal[None, :, :, None], segment, -torch.inf).exp()
    scores = torch.einsum("bihn,bjhn->bijh", C, B) * mask
    y = torch.einsum("bijh,bjhp->bihp", scores, x * dt[..., None])
    y = y + torch.einsum("bihn,bhpn->bihp", C * log_decay.exp()[..., None], initial_state)
    return (y + x * (D if D.ndim == 2 else D[:, None])) * F.silu(z)


@pytest.mark.parametrize("op", REFERENCE_OPS)
def test_matches_masked_attention_dual(op):
    inputs = make_inputs(37)
    output, _ = op(**inputs)
    torch.testing.assert_close(output, masked_attention_oracle(**inputs))


@pytest.mark.parametrize("sequence", [1, 63, 64, 65, 200])
def test_chunk_matches_recurrent_with_state(sequence):
    inputs = make_inputs(sequence)
    expected = recurrent_mamba2(**inputs, output_final_state=True)
    actual = chunk_mamba2(**inputs, output_final_state=True)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("op", REFERENCE_OPS)
def test_state_carry_matches_single_call(op):
    inputs = make_inputs(150)
    del inputs["D"], inputs["z"]
    full_output, full_state = op(**inputs, output_final_state=True)
    cut = 77
    first = {name: inputs[name][:, :cut] for name in ("x", "dt", "B", "C")}
    second = {name: inputs[name][:, cut:] for name in ("x", "dt", "B", "C")}
    first_output, state = op(
        **first, A=inputs["A"], initial_state=inputs["initial_state"], output_final_state=True
    )
    second_output, state = op(
        **second, A=inputs["A"], initial_state=state, output_final_state=True
    )
    torch.testing.assert_close(torch.cat([first_output, second_output], dim=1), full_output)
    torch.testing.assert_close(state, full_state)


@pytest.mark.parametrize("op", REFERENCE_OPS)
def test_packed_sequences_match_independent_calls(op):
    lengths = [5, 0, 70, 1]
    capacity = sum(lengths) + 3
    inputs = make_inputs(capacity, batch=1)
    states = torch.randn(len(lengths), *inputs["initial_state"].shape[1:], dtype=torch.float64)
    cu_seqlens = cumulative_sequence_offsets(lengths, device="cpu")

    output, final_state = op(
        **(inputs | {"initial_state": states}), cu_seqlens=cu_seqlens, output_final_state=True
    )

    for sequence, (begin, end) in enumerate(pairwise(cu_seqlens.tolist())):
        span = {name: inputs[name][:, begin:end] for name in ("x", "dt", "B", "C", "z")}
        if begin == end:
            torch.testing.assert_close(final_state[sequence], states[sequence])
            continue
        expected, expected_state = op(
            **span,
            A=inputs["A"],
            D=inputs["D"],
            initial_state=states[sequence : sequence + 1],
            output_final_state=True,
        )
        torch.testing.assert_close(output[:, begin:end], expected)
        torch.testing.assert_close(final_state[sequence], expected_state[0])


@pytest.mark.parametrize("op", REFERENCE_OPS)
def test_dt_transform_and_per_channel_skip(op):
    inputs = make_inputs(20)
    raw_dt, dt_bias = inputs["dt"] - 2.0, torch.linspace(-1.0, 1.0, inputs["A"].shape[0])
    limit = (0.05, 0.6)
    inputs["D"] = torch.randn(*inputs["x"].shape[2:], dtype=torch.float64)
    transformed = F.softplus(raw_dt + dt_bias).clamp(*limit)
    expected = masked_attention_oracle(**(inputs | {"dt": transformed}))
    output, _ = op(**(inputs | {"dt": raw_dt}), dt_bias=dt_bias, dt_softplus=True, dt_limit=limit)
    torch.testing.assert_close(output, expected)


def test_chunk_gradients_match_recurrent():
    inputs = {name: tensor.requires_grad_() for name, tensor in make_inputs(70, batch=1).items()}
    gradients = []
    for op in REFERENCE_OPS:
        output, state = op(**inputs, output_final_state=True)
        loss = output.square().sum() + state.sin().sum()
        gradients.append(torch.autograd.grad(loss, list(inputs.values())))
    for expected, actual in zip(*gradients):
        torch.testing.assert_close(actual, expected)


def test_low_precision_inputs_use_fp32_state():
    inputs = make_inputs(80)
    low = {
        name: tensor.to(torch.bfloat16) if name in ("x", "B", "C", "z") else tensor.float()
        for name, tensor in inputs.items()
    }
    output, state = chunk_mamba2(**low, output_final_state=True)
    assert output.dtype == torch.bfloat16
    assert state.dtype == torch.float32
    expected, _ = chunk_mamba2(
        **{name: tensor.double() for name, tensor in low.items()}, output_final_state=True
    )
    torch.testing.assert_close(output.double(), expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"A": torch.ones(3)}, "A must have shape"),
        ({"B": torch.ones(2, 5, 3, 5), "C": torch.ones(2, 5, 3, 5)}, "multiple of the B/C groups"),
        ({"C": torch.ones(2, 5, 2, 4)}, "C must have shape"),
        ({"D": torch.ones(4, 7)}, "D must have shape"),
        ({"z": torch.ones(2, 5, 4, 7)}, "z must have shape"),
        ({"initial_state": torch.ones(2, 4, 5, 6)}, "initial_state must have shape"),
        ({"dt_limit": (0.5, 0.1)}, "dt_limit"),
    ],
)
def test_validation_errors(override, message):
    inputs = make_inputs(5) | {
        name: value.double() if isinstance(value, torch.Tensor) else value
        for name, value in override.items()
    }
    with pytest.raises(ValueError, match=message):
        chunk_mamba2(**inputs)


def test_activation_dtypes_must_match():
    inputs = make_inputs(5)
    with pytest.raises(ValueError, match="same dtype"):
        chunk_mamba2(**(inputs | {"x": inputs["x"].float()}))
