"""Seeded random samples of the Mamba2 numerical and state-routing contracts.

Each seed draws a shape, dtype, packing, option set, and one property: training against the FP64
recurrence (every gradient, deterministic replay), packed against per-sequence calls, paged
prefill routing for both prefill ops, prefill plus decode against one recurrent scan, or rejection
of invalid inputs. A campaign can import ``generate_case`` and ``run_case`` to run more seeds.
"""

import random
from dataclasses import dataclass
from itertools import pairwise

import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear import (
    chunk_mamba2,
    paged_chunk_mamba2,
    recurrent_mamba2,
    recurrent_mamba2_decode,
)
from attn_gym.testing.kda import cumulative_sequence_offsets
from attn_gym.testing.mamba2 import force_segments, make_mamba2_inputs

from .test_mamba2_numerics import (
    REDUCTION_NAMES,
    assert_reduction,
    assert_value,
    decay_terms,
    evaluate,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
TOKEN_INPUTS = ("x", "dt", "B", "C", "z")
PROPERTIES = ("training", "packing", "paged_chunk", "paged_recurrent", "decode", "invalid")


@dataclass(frozen=True)
class FuzzCase:
    seed: int
    property: str
    batch: int
    tokens: int
    heads: int
    groups: int
    head_dim: int
    state_dim: int
    fp16: bool
    lengths: tuple[int, ...] | None
    skip: str
    gate: bool
    bias: bool
    softplus: bool
    limit: tuple[float, float]
    initial: bool
    final: bool
    segment: int | None


def generate_case(seed: int, *, extended: bool = False) -> FuzzCase:
    """Use a local RNG so xdist order cannot change shapes, options, or input values."""
    rng = random.Random(seed)
    property_name = PROPERTIES[seed % len(PROPERTIES)]
    heads = rng.choice((1, 2, 4))
    groups = rng.choice(sorted({1, heads, *([2] if heads % 2 == 0 else [])}))
    tokens = rng.choice((1, 2, 63, 64, 65, 127, 128, 129))
    if extended and seed % 13 == 0:
        tokens = rng.choice((255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 1499))
    lengths = None
    if property_name in ("packing", "paged_chunk", "paged_recurrent") or (
        property_name == "training" and seed % 12 == 6
    ):
        lengths = (rng.choice((0, 1, 64)), tokens, 0, 1, rng.choice((0, 64, 128)))
        tokens = sum(lengths) + rng.randint(1, 7)
    if property_name in ("decode", "invalid"):
        tokens = rng.choice((1, 2, 7, 17))
    return FuzzCase(
        seed,
        property_name,
        1 if lengths else rng.randint(1, 4),
        tokens,
        heads,
        groups,
        rng.choice((16, 32, 64, 128)),
        rng.choice((16, 32, 64, 128, 256)),
        rng.choice((False, True)),
        lengths,
        rng.choice(("none", "head", "channel")),
        rng.choice((False, True)),
        rng.choice((False, True)),
        rng.choice((False, True)),
        rng.choice(((0.0, float("inf")), (0.01, 0.2), (0.06, 0.15))),
        rng.choice((False, True)),
        rng.choice((False, True)),
        rng.choice((None, 64, 128)),
    )


def case_inputs(case: FuzzCase) -> tuple[dict[str, torch.Tensor], dict]:
    inputs = make_mamba2_inputs(
        case.batch,
        case.tokens,
        heads=case.heads,
        groups=case.groups,
        head_dim=case.head_dim,
        state_dim=case.state_dim,
        seed=case.seed,
    )
    generator = torch.Generator(device="cuda").manual_seed(case.seed + 10000)
    dtype = torch.float16 if case.fp16 else torch.bfloat16
    if case.fp16:
        for name in ("x", "B", "C", "z"):
            inputs[name] = torch.randn(inputs[name].shape, device="cuda", generator=generator).to(
                dtype
            )
    if not case.softplus:
        # Keep effective steps nonnegative without depending on the optional clamp.
        inputs["dt"] = F.softplus(inputs["dt"])
        inputs["dt_bias"] = inputs["dt_bias"].abs() * 0.1
    if case.skip == "none":
        inputs.pop("D")
    elif case.skip == "channel":
        inputs["D"] = torch.randn(case.heads, case.head_dim, device="cuda", generator=generator)
    if not case.gate:
        inputs.pop("z")
    if not case.bias:
        inputs.pop("dt_bias")
    states = len(case.lengths) if case.lengths else case.batch
    if case.initial:
        inputs["initial_state"] = torch.randn(
            states,
            case.heads,
            case.head_dim,
            case.state_dim,
            device="cuda",
            generator=generator,
        )
    kwargs = {"dt_softplus": case.softplus, "dt_limit": case.limit}
    if case.lengths:
        kwargs["cu_seqlens"] = cumulative_sequence_offsets(case.lengths)
        for name in TOKEN_INPUTS:
            if name in inputs:
                inputs[name][:, sum(case.lengths) :] = float("nan")
    return inputs, kwargs


def active_inputs(inputs: dict[str, torch.Tensor], end: int) -> dict[str, torch.Tensor]:
    """Zero unowned capacity, which the skip-term magnitudes below would otherwise read."""
    clean = {name: value.clone() for name, value in inputs.items()}
    for name in TOKEN_INPUTS:
        if name in clean:
            clean[name][:, end:] = 0
    return clean


def assert_replay(actual: torch.Tensor, replay: torch.Tensor, name: str) -> None:
    assert torch.equal(actual, replay), f"{name}: repeated calls differ"


# Gradients whose kernels read the staged operands below. dz and dD read FP32 values, and the
# initial-state gradient carries the final-state cotangent in FP32.
STAGED_NAMES = ("x", "dt", "A", "B", "C", "dt_bias")
# The model-range error limits do not cover sequences this short, where staging can reach 1.3
# BF16 eps of a reduction's term magnitudes.
STAGED_MAX_LENGTH = 2


def staged_oracle(clean, cotangents, kwargs):
    """FP64 and FP32 recurrences on the operands the kernels stage in the activation dtype.

    The scans read ``dy * silu(z)``, the chunk-entry states, and the state-cotangent tape in the
    activation dtype (docs "Mamba2 precision"). A sequence of at most ``STAGED_MAX_LENGTH``
    tokens is one chunk, whose entry state is ``initial_state`` and whose state cotangent is the
    final one, so rounding those two and ``dy * silu(z)`` reproduces the three roundings. Other
    MMA operand roundings stay within the unchanged value limits.
    """
    dtype = clean["x"].dtype
    inputs, rounded = dict(clean), list(cotangents)
    if "initial_state" in inputs:
        inputs["initial_state"] = inputs["initial_state"].to(dtype).float()
    if len(rounded) > 1:
        rounded[1] = rounded[1].to(dtype).float()
    if "z" in inputs:
        rounded[0] = (rounded[0].float() * F.silu(inputs.pop("z").float())).to(dtype)
    high = evaluate(inputs, rounded, torch.float64, **kwargs)
    low = evaluate(inputs, rounded, torch.float32, **kwargs)
    scales = {"A": decay_terms(inputs, rounded, **kwargs), "dt_bias": high["dt"].abs().sum((0, 1))}
    return high, low, scales


def assert_gradient(value, expected, reference, scale, name, source_dtype):
    if name in REDUCTION_NAMES:
        # An all-clamped step transform has exactly zero bias gradient.
        if not scale.any():
            assert torch.equal(value.double(), expected), name
        else:
            assert_reduction(value, expected, scale.clamp_min(1e-300), name)
    else:
        assert_value(value, expected, reference, name, source_dtype=source_dtype)


def check_training(case: FuzzCase, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, kwargs = case_inputs(case)
    if case.segment:
        force_segments(monkeypatch, case.segment)
    end = sum(case.lengths) if case.lengths else case.tokens
    clean = active_inputs(inputs, end)
    generator = torch.Generator(device="cuda").manual_seed(case.seed + 20000)
    cotangents = [
        torch.randn(inputs["x"].shape, device="cuda", generator=generator).to(inputs["x"])
    ]
    cotangents[0][:, end:] = 0
    states = len(case.lengths) if case.lengths else case.batch
    if case.final:
        cotangents.append(
            torch.randn(
                states,
                case.heads,
                case.head_dim,
                case.state_dim,
                device="cuda",
                generator=generator,
            )
        )
    actual = evaluate(inputs, cotangents, None, **kwargs)
    replay = evaluate(inputs, cotangents, None, **kwargs)
    high = evaluate(clean, cotangents, torch.float64, **kwargs)
    low = evaluate(clean, cotangents, torch.float32, **kwargs)
    scales = {"A": decay_terms(clean, cotangents, **kwargs)}
    if "dt_bias" in inputs:
        scales["dt_bias"] = high["dt"].abs().sum((0, 1))
    if "D" in inputs:
        skip_terms = cotangents[0].double() * clean["x"].double()
        if "z" in clean:
            skip_terms *= F.silu(clean["z"].double())
        axes = (0, 1, 3) if case.skip == "head" else (0, 1)
        scales["D"] = skip_terms.abs().sum(axes)
    failures, staged = [], None
    source_dtype = inputs["x"].dtype
    short = max(case.lengths or [case.tokens]) <= STAGED_MAX_LENGTH
    for name, value in actual.items():
        expected, reference, repeated = high[name], low[name], replay[name]
        if name == "output":
            value, expected, reference, repeated = (
                tensor[:, :end] for tensor in (value, expected, reference, repeated)
            )
        if not torch.equal(value, repeated):
            failures.append(f"{name}: repeated calls differ")
        try:
            assert_gradient(value, expected, reference, scales.get(name), name, source_dtype)
        except AssertionError as error:
            if not short or name not in STAGED_NAMES:
                failures.append(str(error))
                continue
            # The only permitted deviation from the exact oracle is the documented staging.
            staged = staged or staged_oracle(clean, cotangents, kwargs)
            try:
                assert_gradient(
                    value,
                    staged[0][name],
                    staged[1][name],
                    staged[2].get(name),
                    name,
                    source_dtype,
                )
            except AssertionError as staged_error:
                failures.append(f"{error}\nagainst staged operands: {staged_error}")
    assert not failures, "\n".join(failures)
    if not case.final:
        leaves = {name: value.detach().requires_grad_() for name, value in inputs.items()}
        output, state = chunk_mamba2(**leaves, output_final_state=False, **kwargs)
        assert state is None
        gradients = torch.autograd.grad(output, list(leaves.values()), cotangents[0])
        for name, gradient in zip(leaves, gradients):
            assert_replay(gradient, actual[name], f"{name}: without final state")
    with torch.no_grad():
        output, state = chunk_mamba2(**inputs, output_final_state=case.final, **kwargs)
    assert (state is not None) == case.final
    assert_replay(output[:, :end], actual["output"][:, :end], "training/inference output")


def sliced_inputs(
    inputs: dict[str, torch.Tensor], begin: int, end: int
) -> dict[str, torch.Tensor]:
    return {
        name: value[:, begin:end] if name in TOKEN_INPUTS else value
        for name, value in inputs.items()
        if name != "initial_state"
    }


@torch.no_grad()
def check_packing(case: FuzzCase, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs, kwargs = case_inputs(case)
    if case.segment:
        force_segments(monkeypatch, case.segment)
    for op in (chunk_mamba2, recurrent_mamba2):
        output, state = op(**inputs, output_final_state=True, **kwargs)
        replay, replay_state = op(**inputs, output_final_state=True, **kwargs)
        assert_replay(
            output[:, : sum(case.lengths)], replay[:, : sum(case.lengths)], "packed output"
        )
        assert_replay(state, replay_state, "packed state")
        offsets = kwargs["cu_seqlens"].tolist()
        for index, (begin, end) in enumerate(pairwise(offsets)):
            initial = inputs.get("initial_state")
            initial = None if initial is None else initial[index : index + 1]
            if begin == end:
                expected = torch.zeros_like(state[index]) if initial is None else initial[0]
                torch.testing.assert_close(state[index], expected, rtol=0, atol=0)
                continue
            expected, final = op(
                **sliced_inputs(inputs, begin, end),
                initial_state=initial,
                output_final_state=True,
                **{k: v for k, v in kwargs.items() if k != "cu_seqlens"},
            )
            # FP32 accumulation noise can straddle an output-rounding midpoint. Allow
            # one output ulp, while the unrounded FP32 carry must still match tightly.
            torch.testing.assert_close(
                output[:, begin:end], expected, rtol=torch.finfo(output.dtype).eps, atol=1e-5
            )
            torch.testing.assert_close(state[index], final[0], rtol=1e-5, atol=1e-5)


@torch.no_grad()
def check_paged(case: FuzzCase) -> None:
    inputs, kwargs = case_inputs(case)
    inputs.pop("initial_state", None)
    generator = torch.Generator(device="cuda").manual_seed(case.seed + 30000)
    routes = [0, 3, 2, -1, 5]
    indices = torch.tensor(routes, device="cuda", dtype=torch.int32)
    resume = torch.tensor([True, case.initial, False, False, True], device="cuda")
    pool = torch.randn(
        7, case.heads, case.head_dim, case.state_dim, device="cuda", generator=generator
    )
    op = chunk_mamba2 if case.property == "paged_chunk" else recurrent_mamba2
    for dtype in (torch.float32, torch.bfloat16):
        before = pool.to(dtype)
        before[0] = float("nan")
        for i, route in enumerate(routes):
            if route > 0 and not resume[i]:
                before[route] = float("nan")
        initial = torch.stack(
            [
                before[route].float() if route > 0 and resume[i] else torch.zeros_like(pool[0])
                for i, route in enumerate(routes)
            ]
        )
        expected, final = op(**inputs, initial_state=initial, output_final_state=True, **kwargs)
        results = []
        for _ in range(2):
            cache = before.clone()
            if case.property == "paged_chunk":
                output = paged_chunk_mamba2(
                    **inputs,
                    state_cache=cache,
                    state_indices=indices,
                    has_initial_state=resume,
                    **kwargs,
                )
            else:
                output, _ = recurrent_mamba2(
                    **inputs,
                    initial_state=cache,
                    state_indices=indices,
                    has_initial_state=resume,
                    **kwargs,
                )
            for i, (begin, end) in enumerate(pairwise(kwargs["cu_seqlens"].tolist())):
                if routes[i] <= 0:
                    assert output[:, begin:end].eq(0).all()
                else:
                    if begin < end:
                        assert_value(
                            output[:, begin:end],
                            expected[:, begin:end].double(),
                            expected[:, begin:end].float(),
                            "paged output",
                            source_dtype=inputs["x"].dtype,
                        )
                    torch.testing.assert_close(
                        cache[routes[i]], final[i].to(dtype), rtol=0, atol=0
                    )
            torch.testing.assert_close(
                cache[[0, 1, 4, 6]], before[[0, 1, 4, 6]], rtol=0, atol=0, equal_nan=True
            )
            results.append((output[:, : sum(case.lengths)], cache[1:]))
        for first, second in zip(*results):
            assert_replay(first, second, "paged replay")


@torch.no_grad()
def check_decode(case: FuzzCase) -> None:
    inputs, kwargs = case_inputs(case)
    reference_inputs = {name: value.double() for name, value in inputs.items()}
    high, high_state = recurrent_mamba2(
        **reference_inputs, impl="reference", output_final_state=True, **kwargs
    )
    low, low_state = recurrent_mamba2(
        **inputs, impl="reference", output_final_state=True, **kwargs
    )
    expected, final = recurrent_mamba2(**inputs, output_final_state=True, **kwargs)
    for name, value, oracle, baseline in (
        ("recurrent output", expected, high, low),
        ("recurrent state", final, high_state, low_state),
    ):
        assert_value(value, oracle, baseline, name, source_dtype=inputs["x"].dtype)
    indices = torch.arange(1, case.batch + 1, device="cuda", dtype=torch.int32)
    replay = {}
    for cut in (0, max(1, case.tokens // 2)) * 2:
        pool = torch.zeros(
            case.batch + 1, case.heads, case.head_dim, case.state_dim, device="cuda"
        )
        if "initial_state" in inputs:
            pool[1:] = inputs["initial_state"]
        outputs = []
        if cut:
            outputs.append(
                paged_chunk_mamba2(
                    **sliced_inputs(inputs, 0, cut),
                    state_cache=pool,
                    state_indices=indices,
                    **kwargs,
                )
            )
        for token in range(cut, case.tokens):
            step = {
                name: value[:, token] if name in TOKEN_INPUTS else value
                for name, value in inputs.items()
                if name != "initial_state"
            }
            outputs.append(
                recurrent_mamba2_decode(
                    **step, state_cache=pool, state_indices=indices, **kwargs
                ).unsqueeze(1)
            )
        output = torch.cat(outputs, dim=1)
        assert_value(output, high, low, "prefill/decode", source_dtype=inputs["x"].dtype)
        assert_value(
            pool[1:],
            high_state,
            low_state,
            "prefill/decode state",
            source_dtype=inputs["x"].dtype,
        )
        if cut == 0:
            # A one-token launch can fuse differently from a loop iteration. Compare
            # FP32 states before output rounding, not low-precision bit patterns.
            torch.testing.assert_close(pool[1:], final, rtol=1e-5, atol=1e-5)
        if cut in replay:
            assert_replay(output, replay[cut][0], "decode output")
            assert_replay(pool, replay[cut][1], "decode pool")
        replay[cut] = (output, pool)


def check_invalid(case: FuzzCase) -> None:
    inputs, kwargs = case_inputs(case)
    for op in (chunk_mamba2, recurrent_mamba2):
        with pytest.raises(ValueError, match="dt must have shape"):
            op(**(inputs | {"dt": inputs["dt"][..., :0]}), **kwargs)
        with pytest.raises(ValueError, match="same dtype"):
            op(**(inputs | {"B": inputs["B"].float()}), **kwargs)
        with pytest.raises(ValueError, match="dt_limit"):
            op(**inputs, **(kwargs | {"dt_limit": (0.2, 0.1)}))
    step = {
        name: value[:, 0] if name in TOKEN_INPUTS else value
        for name, value in inputs.items()
        if name != "initial_state"
    }
    pool = torch.zeros(case.batch + 1, case.heads, case.head_dim, case.state_dim, device="cuda")
    indices = torch.arange(1, case.batch + 1, device="cuda", dtype=torch.int32)
    step["x"] = step["x"].contiguous()
    with torch.no_grad(), pytest.raises(ValueError, match="out must not alias"):
        recurrent_mamba2_decode(
            **step, state_cache=pool, state_indices=indices, out=step["x"], **kwargs
        )
    with torch.no_grad(), pytest.raises(ValueError, match="state_indices"):
        recurrent_mamba2_decode(**step, state_cache=pool, state_indices=indices.long(), **kwargs)
    with torch.no_grad(), pytest.raises(TypeError, match="float32 or bfloat16"):
        recurrent_mamba2_decode(**step, state_cache=pool.half(), state_indices=indices, **kwargs)
    aliased_pool = pool.bfloat16()
    aliased = {
        name: value.bfloat16() if name in ("x", "B", "C", "z") else value
        for name, value in inputs.items()
        if name != "initial_state"
    }
    aliased["x"] = aliased_pool.flatten()[: inputs["x"].numel()].view_as(inputs["x"])
    with torch.no_grad(), pytest.raises(ValueError, match="must not alias"):
        paged_chunk_mamba2(**aliased, state_cache=aliased_pool, state_indices=indices, **kwargs)
    with torch.no_grad(), pytest.raises(ValueError, match="must not alias"):
        recurrent_mamba2(**aliased, initial_state=aliased_pool, state_indices=indices, **kwargs)


def run_case(case: FuzzCase, monkeypatch: pytest.MonkeyPatch) -> None:
    match case.property:
        case "training":
            check_training(case, monkeypatch)
        case "packing":
            check_packing(case, monkeypatch)
        case "paged_chunk" | "paged_recurrent":
            check_paged(case)
        case "decode":
            check_decode(case)
        case "invalid":
            check_invalid(case)


# Seeds after the first 36 failed a 1000-seed campaign before their checks allowed the documented
# staging (training) and one output-dtype rounding step (packing).
SEEDS = (*range(36), 36, 229, 504, 673, 720, 804, 864, 960)


@pytest.mark.parametrize("seed", SEEDS, ids=lambda seed: f"{PROPERTIES[seed % 6]}-{seed}")
def test_seeded_mamba2_contract(seed, monkeypatch):
    run_case(generate_case(seed), monkeypatch)
