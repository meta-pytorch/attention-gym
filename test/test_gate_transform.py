"""Shared gate transform semantics: kind x shape references and the fused gate kernels."""

from __future__ import annotations

import importlib.util
import math

import pytest
import torch
import torch.nn.functional as F

from attn_gym.linear import GateTransform, chunk_gdn, gate_transform
from attn_gym.linear._delta_rule.gate import (
    _gate_transform_bwd_op,
    _gate_transform_fwd_op,
    _gate_uses_cute,
)
from attn_gym.linear._delta_rule.routing import LEGACY_CUTE_ENV, uses_legacy_cute
from attn_gym.linear._delta_rule.triton import gate_transform as triton_backend
from attn_gym.linear.kda import bound_gate
from attn_gym.testing.gdn import make_gdn_test_inputs
from attn_gym.testing.kda import assert_relative_rms_within, clone_kda_inputs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="gate ops require CUDA")

SCALAR = "scalar"
VECTOR = "vector"
# Per-channel D=128 gates route to the legacy CuTeDSL kernels on SM100-family GPUs; every other
# configuration uses Triton.
VECTOR128 = "vector128"
FUSED_SHAPES = [SCALAR, VECTOR, VECTOR128]
HEAD_DIMS = {SCALAR: None, VECTOR: 40, VECTOR128: 128}
KINDS = {"bounded": -3.25, "softplus": None}
HAS_CUTLASS = importlib.util.find_spec("cutlass") is not None
# The CuTeDSL gate kernels still compile on sm90+, so their parity tests run there.
CUTE_CAPABLE = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9 and HAS_CUTLASS
)
requires_cute = pytest.mark.skipif(
    not CUTE_CAPABLE, reason="CuTeDSL gate kernels require CuTeDSL and CUDA capability >= 9.0"
)


def uses_cute(shape: str) -> bool:
    """Expected backend routing for a fused gate of this shape."""
    return shape == VECTOR128 and HAS_CUTLASS and uses_legacy_cute(torch.device("cuda"))


@pytest.fixture(params=["routed", "triton"])
def route(request, monkeypatch) -> str:
    """Run the routed backend, or force the legacy CuTeDSL route onto Triton."""
    if request.param == "triton":
        if not uses_cute(VECTOR128):
            pytest.skip("the routed backend is already the Triton kernel")
        monkeypatch.setenv(LEGACY_CUTE_ENV, "0")
    return request.param


def make_gate_inputs(
    shape: str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    batch: int = 2,
    tokens: int = 65,
    heads: int = 3,
    head_dim: int = 40,
    requires_grad: bool = False,
    seed: int = 23,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(raw_gate, A_log, dt_bias)`` for a per-head or per-channel gate."""
    torch.manual_seed(seed)
    head_dim = HEAD_DIMS[shape] or head_dim
    trailing = (heads,) if shape == SCALAR else (heads, head_dim)
    return (
        torch.randn(batch, tokens, *trailing, device="cuda", dtype=dtype).requires_grad_(
            requires_grad
        ),
        torch.randn(heads, device="cuda").requires_grad_(requires_grad),
        torch.randn(*trailing, device="cuda").requires_grad_(requires_grad),
    )


def expected_gate(raw_gate, A_log, dt_bias, kind, lower_bound=None):
    """Independent formula in FP64."""
    s = raw_gate.double() + dt_bias.double()
    amplitude = A_log.double().exp().view(1, 1, -1, *([1] * (raw_gate.ndim - 3)))
    if kind == "bounded":
        return lower_bound * torch.sigmoid(amplitude * s)
    return -amplitude * F.softplus(s)


@pytest.mark.parametrize("shape", [SCALAR, VECTOR])
@pytest.mark.parametrize("kind", ["bounded", "softplus"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_reference_matches_formula(shape: str, kind: str, dtype: torch.dtype):
    """Pin both formulas on both shapes; output is FP32 regardless of input precision."""
    inputs = make_gate_inputs(shape, dtype=dtype)
    lower_bound = -3.25 if kind == "bounded" else None
    actual = gate_transform(*inputs, kind=kind, lower_bound=lower_bound, impl="reference")
    assert actual.dtype == torch.float32
    assert actual.shape == inputs[0].shape
    expected = expected_gate(*inputs, kind, lower_bound)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-6, atol=1e-6)
    if kind == "softplus":
        assert (actual <= 0).all()
    else:
        assert ((actual >= lower_bound) & (actual <= 0)).all()


def test_softplus_nonfast_uses_accurate_amplitude():
    """Resolve a subnormal exp(A_log) accurately before finite amplification."""
    raw = torch.full((1, 1, 1), 2.0**100, device="cuda")
    log_scale = torch.tensor([-90.0], device="cuda", requires_grad=True)
    bias = torch.zeros(1, device="cuda")
    assert not _gate_uses_cute(raw)
    expected = expected_gate(raw, log_scale, bias, "softplus")
    accurate = gate_transform(raw, log_scale, bias, kind="softplus", fastmath=False)
    # Allow two FP32 subnormal quanta before multiplication by the large finite factor.
    atol = 2 * 2.0**-149 * 2.0**100
    torch.testing.assert_close(accurate.double(), expected, rtol=0, atol=atol)
    (accurate_grad,) = torch.autograd.grad(accurate.sum(), log_scale)
    torch.testing.assert_close(accurate_grad.double(), expected.reshape(1), rtol=0, atol=atol)


def test_kind_is_required_and_validated():
    inputs = make_gate_inputs(SCALAR)
    with pytest.raises(TypeError):
        gate_transform(*inputs)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="gate transform kind"):
        gate_transform(*inputs, kind="sigmoid")
    assert gate_transform(*inputs, kind=GateTransform.SOFTPLUS, impl="reference").shape


def test_lower_bound_rules():
    inputs = make_gate_inputs(SCALAR)
    with pytest.raises(ValueError, match="lower_bound applies only"):
        gate_transform(*inputs, kind="softplus", lower_bound=-5.0, impl="reference")
    with pytest.raises(TypeError, match="requires a real lower_bound"):
        gate_transform(*inputs, kind="bounded", impl="reference")
    for invalid in (1.0, float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="lower_bound"):
            gate_transform(*inputs, kind="bounded", lower_bound=invalid, impl="reference")
    with pytest.raises(TypeError, match="lower_bound"):
        gate_transform(*inputs, kind="bounded", lower_bound=True, impl="reference")


def test_shape_and_dtype_validation():
    raw_gate, A_log, dt_bias = make_gate_inputs(VECTOR)
    with pytest.raises(ValueError, match=r"raw_gate must have shape"):
        gate_transform(raw_gate[0, 0], A_log, dt_bias, kind="softplus")
    with pytest.raises(ValueError, match="A_log"):
        gate_transform(raw_gate, A_log.half(), dt_bias, kind="softplus")
    with pytest.raises(ValueError, match="dt_bias"):
        gate_transform(raw_gate, A_log, dt_bias[:, 0], kind="softplus")
    with pytest.raises(ValueError, match="dt_bias"):
        gate_transform(raw_gate[..., 0], A_log, dt_bias, kind="softplus")
    with pytest.raises(TypeError, match="floating-point"):
        gate_transform(raw_gate.int(), A_log, dt_bias, kind="softplus")
    with pytest.raises(ValueError, match="same device"):
        gate_transform(raw_gate, A_log.cpu(), dt_bias, kind="softplus")
    with pytest.raises(TypeError, match="fastmath must be bool"):
        gate_transform(raw_gate, A_log, dt_bias, kind="softplus", impl="reference", fastmath=1)


def test_fused_rejections():
    raw_gate, A_log, dt_bias = make_gate_inputs(SCALAR)
    with pytest.raises(ValueError, match="nonzero"):
        gate_transform(raw_gate[:, :0], A_log, dt_bias, kind="softplus")
    with pytest.raises(ValueError, match="CUDA FP16, BF16, or FP32"):
        gate_transform(raw_gate.double(), A_log, dt_bias, kind="softplus")
    with pytest.raises(ValueError, match="CUDA FP16, BF16, or FP32"):
        gate_transform(raw_gate.cpu(), A_log.cpu(), dt_bias.cpu(), kind="softplus")
    with pytest.raises(TypeError, match="fastmath"):
        gate_transform(raw_gate, A_log, dt_bias, kind="softplus", fastmath=1)


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("shape", FUSED_SHAPES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("fastmath", [False, True])
def test_fused_gate_matches_reference(
    shape: str, dtype: torch.dtype, fastmath: bool, kind: str, route: str
):
    """Forward and all three gradients on partial row and channel blocks, both backends."""
    lower_bound = KINDS[kind]
    expected_inputs = make_gate_inputs(shape, dtype=dtype, requires_grad=True)
    assert _gate_uses_cute(expected_inputs[0]) == uses_cute(shape)
    actual_inputs = clone_kda_inputs(expected_inputs)
    expected = gate_transform(
        *expected_inputs, kind=kind, lower_bound=lower_bound, impl="reference"
    )
    actual = gate_transform(*actual_inputs, kind=kind, lower_bound=lower_bound, fastmath=fastmath)
    assert actual.dtype == torch.float32
    forward_tolerance = 1e-5 if fastmath else 2e-6
    torch.testing.assert_close(actual, expected, rtol=forward_tolerance, atol=forward_tolerance)
    assert_relative_rms_within(actual, expected, "gate", max_eps=1.0, source_dtype=torch.float32)

    cotangent = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, expected_inputs, cotangent)
    actual_gradients = torch.autograd.grad(actual, actual_inputs, cotangent)
    assert actual_gradients[0].dtype == dtype
    assert actual_gradients[1].shape == expected_inputs[1].shape
    assert actual_gradients[2].shape == expected_inputs[2].shape
    # d_raw_gate rounds to the input dtype. The parameter gradients reduce over B*T (and D)
    # in FP32 with a different summation order than eager, so their aggregate budget scales
    # with sqrt(reduction size).
    batch, tokens = expected.shape[:2]
    reduction_sizes = (1, expected[0, 0, 0].numel() * batch * tokens, batch * tokens)
    tolerances = (
        {"rtol": 1e-2, "atol": 8e-3},
        {"rtol": 3e-4, "atol": 3e-4},
        {"rtol": 3e-4, "atol": 3e-4},
    )
    names = ("d_raw_gate", "d_A_log", "d_dt_bias")
    for name, actual_gradient, expected_gradient, tolerance, reduction_size in zip(
        names, actual_gradients, expected_gradients, tolerances, reduction_sizes, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, **tolerance)
        assert_relative_rms_within(
            actual_gradient.float(),
            expected_gradient.float(),
            name,
            max_eps=4.0 * math.sqrt(reduction_size),
            source_dtype=dtype if name == "d_raw_gate" else torch.float32,
        )


@pytest.mark.parametrize("shape", [SCALAR, VECTOR128])
@pytest.mark.parametrize("fastmath", [False, True])
@pytest.mark.parametrize(
    ("logits", "a_log"),
    [
        pytest.param((3e38, -20.0, 80.0), 0.0, id="huge-logits"),
        pytest.param((-20.0, -12.0, -8.0), 20.0, id="large-amplitude-small-tail"),
    ],
)
def test_fused_softplus_extreme_logits(
    shape: str, fastmath: bool, logits: tuple[float, ...], a_log: float
):
    """Extreme logits match the reference, with gradients, on both backends.

    Regressions: ``(z + |z|) / 2`` overflows for ``z > FLT_MAX / 2``; ``1 - 1/(1+e)`` cancels the
    ``sigmoid(z)`` factor to zero for ``z <= -17``; fastmath ``log2(1 + e)`` rounds the softplus
    tail to zero, which ``exp(A_log)`` amplifies (``raw=-20, A_log=20`` is a gate of -1, not 0).
    """
    raw_gate, A_log, dt_bias = make_gate_inputs(shape, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        for offset, logit in enumerate(logits):
            raw_gate[:, offset::3] = logit
        dt_bias.zero_()
        A_log.fill_(a_log)
    actual = gate_transform(raw_gate, A_log, dt_bias, kind="softplus", fastmath=fastmath)
    expected = gate_transform(raw_gate, A_log, dt_bias, kind="softplus", impl="reference")
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    cotangent = torch.ones_like(expected)
    expected_gradients = torch.autograd.grad(expected, (raw_gate, A_log, dt_bias), cotangent)
    actual_gradients = torch.autograd.grad(actual, (raw_gate, A_log, dt_bias), cotangent)
    assert (expected_gradients[0][:, 1::3] != 0).all()
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("shape", [SCALAR, VECTOR])
@pytest.mark.parametrize("fastmath", [False, True])
def test_fused_bounded_keeps_saturated_gradients(shape: str, fastmath: bool):
    """Saturated logits keep their tiny but representable gradients.

    ``s * (1 - s)`` cancels to zero once ``sigmoid(x)`` rounds to one (``x >~ 17`` in FP32 and
    ``x >~ 37`` in FP64), so the oracle is the closed form ``e / (1 + e)^2``, ``e = exp(-|x|)``.
    """
    raw_gate, A_log, dt_bias = make_gate_inputs(shape, dtype=torch.float32, requires_grad=True)
    with torch.no_grad():
        for offset, logit in enumerate((-60.0, -18.0, 0.5, 18.0, 60.0)):
            raw_gate[:, offset::5] = logit
        dt_bias.zero_()
        A_log.zero_()
    lower_bound = -5.0
    actual = gate_transform(
        raw_gate, A_log, dt_bias, kind="bounded", lower_bound=lower_bound, fastmath=fastmath
    )
    logits = raw_gate.detach().double()
    torch.testing.assert_close(
        actual.double(), lower_bound * torch.sigmoid(logits), rtol=1e-6, atol=1e-6
    )
    decay = torch.exp(-logits.abs())
    exact_d_raw = lower_bound * decay / (1 + decay) ** 2
    exact_d_A_log = (exact_d_raw * logits).sum((0, 1, 3)[: logits.ndim - 1])
    d_raw, d_A_log = torch.autograd.grad(actual, (raw_gate, A_log), torch.ones_like(actual))
    torch.testing.assert_close(d_raw.double(), exact_d_raw, rtol=1e-5, atol=0)
    torch.testing.assert_close(d_A_log.double(), exact_d_A_log, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("fastmath", [False, True])
@pytest.mark.parametrize(
    ("kind", "lower_bound", "raw", "a_log", "bias"),
    [
        pytest.param("bounded", -5.0, 0.0, 88.0, 0.0, id="bounded-amplitude-near-max"),
        pytest.param("bounded", -5.0, 1e4, 80.0, 0.0, id="bounded-overflowing-logit"),
        pytest.param("softplus", None, -10020.0, 80.0, 1e4, id="softplus-overflowing-padding"),
    ],
)
def test_fused_gate_gradients_stay_finite_at_extreme_amplitudes(
    kind: str, lower_bound: float | None, raw: float, a_log: float, bias: float, fastmath: bool
):
    """All three gradients match the FP32 reference when only an intermediate would overflow.

    Regressions: the bounded backward formed ``lower_bound * exp(A_log)`` before the sigmoid
    slope, which overflows for ``exp(88)`` although the gradient is finite; and the padded rows
    of a partial row block evaluate ``s = dt_bias``, where ``exp(A_log) * softplus(s)``
    overflowed, so their zero cotangent gave ``inf * 0`` and poisoned ``d_A_log``. An
    overflowing bounded logit has a zero slope that ``d_A_log`` must not multiply by the logit.
    """
    raw_gate = torch.full((1, 1, 1), raw, device="cuda", requires_grad=True)
    A_log = torch.full((1,), a_log, device="cuda", requires_grad=True)
    dt_bias = torch.full((1,), bias, device="cuda", requires_grad=True)
    inputs = (raw_gate, A_log, dt_bias)
    options = {"kind": kind, "lower_bound": lower_bound}
    actual = gate_transform(*inputs, **options, fastmath=fastmath)
    expected = gate_transform(*inputs, **options, impl="reference")
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=0)
    cotangent = torch.ones_like(expected)
    expected_gradients = torch.autograd.grad(expected, inputs, cotangent)
    actual_gradients = torch.autograd.grad(actual, inputs, cotangent)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert torch.isfinite(expected_gradient).all()
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-4, atol=0)


@requires_cute
@pytest.mark.parametrize(
    ("batch", "tokens", "heads"),
    [
        pytest.param(1, 1, 1, id="minimum"),
        pytest.param(1, 32, 9, id="exact-token-tile-ragged-head-group"),
        pytest.param(65536, 1, 1, id="flattened-batch-grid"),
    ],
)
def test_cute_softplus_boundary_shapes(batch: int, tokens: int, heads: int):
    """The CuTeDSL softplus specialization on the forward head-group and TMA tile boundaries."""
    if not uses_cute(VECTOR128):
        pytest.skip("the legacy CuTeDSL gate route is not selected on this device")
    inputs = make_gate_inputs(
        VECTOR128, batch=batch, tokens=tokens, heads=heads, requires_grad=True
    )
    actual_inputs = clone_kda_inputs(inputs)
    expected = gate_transform(*inputs, kind="softplus", impl="reference")
    actual = gate_transform(*actual_inputs, kind="softplus")
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    cotangent = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, inputs, cotangent)
    actual_gradients = torch.autograd.grad(actual, actual_inputs, cotangent)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        # d_raw_gate rounds to bf16 (one ulp at |g|~4 is 0.03); parameter gradients are FP32.
        torch.testing.assert_close(
            actual_gradient.float(), expected_gradient.float(), rtol=1e-2, atol=1e-2
        )


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("shape", FUSED_SHAPES)
def test_fused_gate_supports_strided_inputs(shape: str, kind: str, route: str):
    """Strided raw gates and cotangents are handled (in place by Triton, normalized by CuTeDSL)."""
    lower_bound = KINDS[kind]
    raw_gate, A_log, dt_bias = make_gate_inputs(shape, heads=4, head_dim=48, requires_grad=True)
    # Slice heads and (for vector gates) channels out of a wider projection buffer; the
    # views stay differentiable so both paths see identical strided operands.
    views = [raw_gate[:, :, 1:3], A_log[1:3], dt_bias[1:3]]
    if shape == VECTOR:
        views[0] = views[0][..., ::2]
        views[2] = views[2][:, ::2]
    assert not views[0].is_contiguous()
    assert _gate_uses_cute(views[0]) == uses_cute(shape)
    expected = gate_transform(*views, kind=kind, lower_bound=lower_bound, impl="reference")
    actual = gate_transform(*views, kind=kind, lower_bound=lower_bound)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    cotangent = torch.randn(*expected.shape, 2, device="cuda")[..., 0]
    assert not cotangent.is_contiguous()
    expected_gradients = torch.autograd.grad(expected, views, cotangent)
    actual_gradients = torch.autograd.grad(actual, views, cotangent)
    assert actual_gradients[0].is_contiguous()
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient.float(), expected_gradient.float(), rtol=3e-3, atol=8e-3
        )


@pytest.mark.parametrize("shape", FUSED_SHAPES)
def test_operator_registration(shape: str):
    """Validate the shared operators' schema and fakes, including a permuted ``dt_bias``."""
    utilities = ("test_schema", "test_faketensor", "test_aot_dispatch_dynamic")
    raw_gate, A_log, dt_bias = make_gate_inputs(shape, tokens=33)
    d_gate = torch.randn(raw_gate.shape, device="cuda")
    kinds = [("softplus", 0.0), ("bounded", -5.0)]
    if dt_bias.ndim == 2:
        dt_bias = (
            dt_bias.t().contiguous().t()
        )  # Dense but permuted: the fake must still be compact.
    for kind, lower_bound in kinds:
        torch.library.opcheck(
            _gate_transform_fwd_op,
            (raw_gate, A_log, dt_bias, kind, lower_bound, False),
            test_utils=utilities,
        )
        torch.library.opcheck(
            _gate_transform_bwd_op,
            (raw_gate, A_log, dt_bias, d_gate, kind, lower_bound, False),
            test_utils=utilities,
        )


@requires_cute
@pytest.mark.parametrize("transform", list(GateTransform))
@pytest.mark.parametrize("fastmath", [False, True])
def test_gate_backends_agree_on_d128(fastmath: bool, transform: GateTransform):
    """The CuTeDSL and Triton gate launchers implement one contract on the shared shape."""
    from attn_gym.linear.kda.bwd.cute import gate_bwd as cute_bwd
    from attn_gym.linear.kda.fwd.cute import gate_fwd as cute_fwd

    raw_gate, A_log, dt_bias = make_gate_inputs(VECTOR128, tokens=97)
    d_gate = torch.randn(raw_gate.shape, device="cuda")
    lower_bound = -5.0 if transform is GateTransform.BOUNDED else 0.0
    args = (raw_gate, A_log, dt_bias, lower_bound, fastmath, transform)
    cute_gate = cute_fwd._gate_transform_fwd_cuda(*args)
    triton_gate = triton_backend._gate_transform_fwd_cuda(*args)
    torch.testing.assert_close(cute_gate, triton_gate, rtol=1e-5, atol=1e-5)
    args = (raw_gate, A_log, dt_bias, d_gate, lower_bound, fastmath, transform)
    cute_grads = cute_bwd._gate_transform_bwd_cuda(*args)
    triton_grads = triton_backend._gate_transform_bwd_cuda(*args)
    for name, cute_grad, triton_grad in zip(
        ("d_raw_gate", "d_A_log", "d_dt_bias"), cute_grads, triton_grads, strict=True
    ):
        torch.testing.assert_close(
            cute_grad.float(), triton_grad.float(), rtol=3e-3, atol=3e-3, msg=name
        )


@requires_cute
def test_cute_gate_compiles_two_grid_variants():
    """Batch, tokens and tile counts are runtime values: one kernel serves every shape whose
    batch fits ``gridDim.z``, and one more (the batch folded into ``gridDim.x``) the batches past 65535."""
    from attn_gym.linear.kda.bwd.cute import gate_bwd as cute_bwd
    from attn_gym.linear.kda.fwd.cute import gate_fwd as cute_fwd

    compile_fns = (cute_fwd._compile_gate_transform_fwd, cute_bwd._compile_gate_transform_bwd)
    lower_bound = -4.6875  # this test's own compile key, so every variant below is a new entry
    before = [fn.cache_info().currsize for fn in compile_fns]

    def run(batch: int, tokens: int):
        raw_gate, A_log, dt_bias = make_gate_inputs(VECTOR128, batch=batch, tokens=tokens, heads=8)
        d_gate = torch.randn(raw_gate.shape, device="cuda")
        reference_inputs = [t.detach().requires_grad_() for t in (raw_gate, A_log, dt_bias)]
        reference = gate_transform(
            *reference_inputs, kind="bounded", lower_bound=lower_bound, impl="reference"
        )
        gate = cute_fwd._gate_transform_fwd_cuda(
            raw_gate, A_log, dt_bias, lower_bound, False, GateTransform.BOUNDED
        )
        torch.testing.assert_close(gate, reference, rtol=2e-5, atol=2e-5)
        grads = cute_bwd._gate_transform_bwd_cuda(
            raw_gate, A_log, dt_bias, d_gate, lower_bound, False, GateTransform.BOUNDED
        )
        expected_grads = torch.autograd.grad(reference, reference_inputs, d_gate)
        # d_raw_gate rounds to bf16; the parameter gradients sum B * T rows in FP32.
        rows = math.sqrt(batch * tokens)
        tolerances = ((1e-2, 8e-3), (3e-3, 5e-4 * rows), (3e-3, 5e-5 * rows))
        for grad, expected, (rtol, atol) in zip(grads, expected_grads, tolerances, strict=True):
            torch.testing.assert_close(grad.float(), expected.float(), rtol=rtol, atol=atol)
        return [fn.cache_info().currsize - old for fn, old in zip(compile_fns, before)]

    # Specialization classes of a plain int (1, multiples of 16, others), tile and stage edges.
    for batch, tokens in [(1, 1), (16, 32), (3, 97), (17, 16), (2, 4096), (65535, 1)]:
        assert run(batch, tokens) == [1, 1], (batch, tokens)
    for batch, tokens in [(65536, 1), (65537, 3)]:
        assert run(batch, tokens) == [2, 2], (batch, tokens)
    assert run(2, 65) == [2, 2]


@requires_cute
@pytest.mark.parametrize(("batch", "tokens"), [(0, 65), (2, 0)])
def test_cute_gate_launchers_skip_an_empty_space(batch: int, tokens: int):
    """The kernels assume every CTA has an item, so the launchers must not launch an empty
    space (the public ``gate_transform`` rejects one earlier): empty outputs, zero gradients."""
    from attn_gym.linear.kda.bwd.cute import gate_bwd as cute_bwd
    from attn_gym.linear.kda.fwd.cute import gate_fwd as cute_fwd

    raw_gate, A_log, dt_bias = make_gate_inputs(VECTOR128, batch=batch, tokens=tokens)
    softplus = GateTransform.SOFTPLUS
    gate = cute_fwd._gate_transform_fwd_cuda(raw_gate, A_log, dt_bias, 0.0, False, softplus)
    assert gate.shape == raw_gate.shape
    d_raw_gate, dA_log, d_dt_bias = cute_bwd._gate_transform_bwd_cuda(
        raw_gate, A_log, dt_bias, gate, 0.0, False, softplus
    )
    torch.cuda.synchronize()
    assert d_raw_gate.shape == raw_gate.shape
    assert dA_log.shape == A_log.shape and d_dt_bias.shape == dt_bias.shape
    assert not dA_log.any() and not d_dt_bias.any()


@pytest.mark.parametrize("kind", sorted(KINDS))
@pytest.mark.parametrize("shape", [SCALAR, VECTOR128])
def test_fused_gate_fullgraph_dynamic_tokens(shape: str, kind: str):
    """Reuse one fullgraph callable across batch and token sizes with backward, both backends."""
    lower_bound = KINDS[kind]
    torch.compiler.reset()
    with torch._dynamo.config.patch(error_on_recompile=True):
        compiled = torch.compile(gate_transform, fullgraph=True, dynamic=True)
        for batch, tokens in ((2, 65), (3, 97)):
            expected_inputs = make_gate_inputs(
                shape, batch=batch, tokens=tokens, requires_grad=True, seed=tokens
            )
            actual_inputs = clone_kda_inputs(expected_inputs)
            expected = gate_transform(
                *expected_inputs, kind=kind, lower_bound=lower_bound, impl="reference"
            )
            actual = compiled(*actual_inputs, kind=kind, lower_bound=lower_bound)
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
            cotangent = torch.randn_like(expected)
            expected_gradients = torch.autograd.grad(expected, expected_inputs, cotangent)
            actual_gradients = torch.autograd.grad(actual, actual_inputs, cotangent)
            for actual_gradient, expected_gradient in zip(
                actual_gradients, expected_gradients, strict=True
            ):
                torch.testing.assert_close(
                    actual_gradient.float(), expected_gradient.float(), rtol=3e-3, atol=8e-3
                )


def test_bound_gate_is_bounded_gate_transform():
    """``bound_gate`` keeps its per-channel contract and equals ``kind="bounded"``."""
    inputs = make_gate_inputs(VECTOR128)
    for impl in ("reference", "fused"):
        expected = gate_transform(*inputs, kind="bounded", lower_bound=-3.25, impl=impl)
        actual = bound_gate(*inputs, lower_bound=-3.25, impl=impl)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    scalar = make_gate_inputs(SCALAR)
    with pytest.raises(ValueError, match=r"\[B, T, H, D\]"):
        bound_gate(*scalar, impl="reference")
    assert gate_transform(*scalar, kind="bounded", lower_bound=-5.0, impl="reference").shape


def test_chunk_gdn_accepts_fused_softplus_gate():
    """A fused gate feeds ``chunk_gdn`` identically to the hand-built eager gate, both ways."""
    q, k, v, _, beta, initial_state, _ = make_gdn_test_inputs(
        96, batch=2, value_heads=4, dtype=torch.bfloat16
    )
    raw_gate, A_log, dt_bias = make_gate_inputs(
        SCALAR, dtype=torch.bfloat16, batch=2, tokens=96, heads=4, requires_grad=True
    )
    eager_leaves = clone_kda_inputs((raw_gate, A_log, dt_bias))
    eager_gate = -eager_leaves[1].exp().view(1, 1, -1) * F.softplus(
        eager_leaves[0].float() + eager_leaves[2].view(1, 1, -1)
    )
    fused_gate = gate_transform(raw_gate, A_log, dt_bias, kind="softplus")
    expected, _ = chunk_gdn(q, k, v, eager_gate, beta, initial_state)
    actual, _ = chunk_gdn(q, k, v, fused_gate, beta, initial_state)
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    cotangent = torch.randn_like(expected)
    expected_gradients = torch.autograd.grad(expected, eager_leaves, cotangent)
    actual_gradients = torch.autograd.grad(actual, (raw_gate, A_log, dt_bias), cotangent)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient.float(), expected_gradient.float(), rtol=2e-3, atol=2e-3
        )


def test_triton_bounded_gate_reuses_kernels_across_bounds_and_lengths(triton_compiles):
    """``lower_bound`` and the token count are runtime values of the Triton bounded kernels."""
    for tokens, lower_bound in ((65, -5.0), (97, -3.25), (128, -1.0)):
        inputs = make_gate_inputs(VECTOR, tokens=tokens, requires_grad=True)
        gate = gate_transform(*inputs, kind="bounded", lower_bound=lower_bound)
        torch.autograd.grad(gate.sum(), inputs)
    assert triton_compiles["gate_transform_fwd_kernel"] == 1, triton_compiles
    assert triton_compiles["gate_transform_bwd_kernel"] == 1, triton_compiles
