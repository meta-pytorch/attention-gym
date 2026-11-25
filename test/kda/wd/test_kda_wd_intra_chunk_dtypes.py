"""Precision selection, numerical accuracy and bitwise causality for warpdrive KDA."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import ReplayState, chunk_kda, paged_chunk_kda
from attn_gym.linear.kda.impl.warpdrive_ops import (
    chunk_warpdrive_packed_bwd_with_state_op,
    chunk_warpdrive_packed_fwd_with_state_op,
)
from attn_gym.linear.kda.naive import naive_recurrent_kda
from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward
from attn_gym.linear.kda.validation import resolve_kernel_options
from attn_gym.testing.kda import cumulative_sequence_offsets, kda_reference, relative_rms_error

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive KDA path requires an SM100-family GPU",
)

VALUES = ("tf32+bf16", "tf32", "bf16")  # BF16 inputs
TF32 = "tf32"  # the widest operands
FP16_VALUE = "tf32+fp16"  # FP16 inputs' one kernel set
WARPDRIVE = {"backend": "warpdrive"}
# Output relative RMS error limits against the independent FP64 recurrence on _inputs.
BUDGETS = {"tf32": 2.5e-3, "tf32+bf16": 4.5e-3, "bf16": 5.5e-3}
# One shape per forward plan: (tokens, heads, PREP_TILE_FRACTION override); "split" forces the
# per-tile kernels at the prep plan's d_v split.
PLANS = {
    "chain": (16384, 32, None),
    "prep": (2048, 8, None),
    "uncut": (4096, 80, None),
    "split": (2048, 8, 0.0),
}
PLAN_PARAMS = [pytest.param(*shape, id=name) for name, shape in PLANS.items()]


@pytest.fixture
def plan_fraction(monkeypatch):
    """Apply a plan's PREP_TILE_FRACTION override (None keeps the default)."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda

    def apply(fraction):
        if fraction is not None:
            monkeypatch.setattr(kda, "PREP_TILE_FRACTION", fraction)

    return apply


def _inputs(tokens, heads, dtype=torch.bfloat16, v_scale=1.0, seed=0, batch=1):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def normal(*shape):
        return torch.randn(*shape, device="cuda", generator=gen)

    shape = (batch, tokens, heads, 128)
    q, k = (F.normalize(normal(*shape), dim=-1).to(dtype) for _ in range(2))
    v = (normal(*shape) * v_scale).to(dtype)
    gate = -5 * torch.rand(*shape, device="cuda", generator=gen)
    beta = torch.rand(*shape[:3], device="cuda", generator=gen)
    return q, k, v, gate, beta


@torch.no_grad()
def _reference(q, k, v, gate, beta, state=None):
    expected, _ = naive_recurrent_kda(
        *(t.double() for t in (q, k, v)),
        gate.double() / math.log(2),
        beta.double(),
        initial_state=None if state is None else state.double(),
    )
    return expected


def _warpdrive(inputs, value, **kwargs):
    with torch.no_grad():
        options = {**WARPDRIVE, "intra_chunk_dtypes": value}
        return chunk_kda(*inputs, kernel_options=options, **kwargs)


def test_intra_chunk_dtypes_validation() -> None:
    assert resolve_kernel_options(None).intra_chunk_dtypes is None  # resolved by input dtype
    for value in VALUES:
        assert resolve_kernel_options({"intra_chunk_dtypes": value}).intra_chunk_dtypes == value
    with pytest.raises(ValueError, match="intra_chunk_dtypes"):
        resolve_kernel_options({"intra_chunk_dtypes": "fp16"})
    for removed in ("bf16+fp16",):
        with pytest.raises(ValueError, match="intra_chunk_dtypes"):
            resolve_kernel_options({"intra_chunk_dtypes": removed})


def test_bf16_inputs_reject_the_fp16_kernel_set() -> None:
    """Only native FP16 kernels accept FP16 solve operands."""
    inputs = _inputs(64, 2)
    with pytest.raises(ValueError, match="FP16 q/k/v kernel set"):
        chunk_kda(*inputs, kernel_options={**WARPDRIVE, "intra_chunk_dtypes": FP16_VALUE})


@pytest.mark.parametrize(
    "dtype, expected",
    [(torch.bfloat16, "tf32+bf16"), (torch.float16, FP16_VALUE)],
    ids=["bf16", "fp16"],
)
def test_omitted_option_runs_the_default_of_the_input_dtype(dtype, expected) -> None:
    """Omitting the option matches the dtype-specific default bit for bit, including gradients
    and the staged API."""
    inputs = _inputs(1000, 4, dtype)
    cu = cumulative_sequence_offsets([1000])
    state = torch.randn(1, 4, 128, 128, device="cuda") * 0.1
    d_output = torch.randn_like(inputs[2])

    def run(options):
        leaves = [t.clone().requires_grad_() for t in inputs]
        output, _ = chunk_kda(*leaves, kernel_options=options)
        grads = torch.autograd.grad((output * d_output).sum(), leaves)
        prepared = chunk_kda_prepare(*inputs, cu_seqlens=cu, kernel_options=options)
        backward = chunk_kda_prepare_backward(
            prepared.saved, d_output, state, scale=prepared.scale, kernel_options=options
        )
        return output, *grads, *prepared.run(state, output_final_state=True), *backward.run()

    omitted = run(WARPDRIVE)
    explicit = run({**WARPDRIVE, "intra_chunk_dtypes": expected})
    for index, (got, want) in enumerate(zip(omitted, explicit, strict=True)):
        assert torch.equal(got, want), index


@pytest.mark.parametrize("backend", ["fused", "warpdrive"])
def test_fp16_inputs_reject_other_values_before_any_launch(backend) -> None:
    """Reject a BF16-only precision before dispatch, including during full-graph tracing."""
    q, k, v, gate, beta = _inputs(64, 2, torch.float16)
    pool = torch.zeros(2, 2, 128, 128, device="cuda")
    slots = torch.ones(1, dtype=torch.int32, device="cuda")
    cu = cumulative_sequence_offsets([64])
    message = r"FP16 q/k/v take only .*'tf32\+fp16'"
    for value in VALUES:
        options = {"backend": backend, "intra_chunk_dtypes": value}
        with pytest.raises(ValueError, match=message):
            chunk_kda(q, k, v, gate, beta, kernel_options=options)
        with pytest.raises(ValueError, match=message), torch.no_grad():
            paged_chunk_kda(
                q, k, v, gate, beta, pool, slots, cu_seqlens=cu, kernel_options=options
            )
        with pytest.raises(ValueError, match=message):
            chunk_kda_prepare(q, k, v, gate, beta, cu_seqlens=cu, kernel_options=options)
    compiled = torch.compile(chunk_kda, fullgraph=True)
    # Full-graph tracing reports the ValueError it observed.
    with pytest.raises(torch._dynamo.exc.Unsupported, match=message):
        compiled(
            q, k, v, gate, beta, kernel_options={"backend": backend, "intra_chunk_dtypes": "bf16"}
        )


def _replay(q, k, v, gate, beta, options):
    """One replay-backed ``paged_chunk_kda`` call on fresh slot 1 of a zero FP32 pool."""
    heads = q.shape[2]
    replay = ReplayState(
        *(
            torch.zeros((2, *shape), device="cuda", dtype=dtype)
            for shape, dtype in ReplayState.slot_layout(heads, 128)
        )
    )
    pool = torch.zeros(2, heads, 128, 128, device="cuda")
    slots = torch.ones(1, dtype=torch.int32, device="cuda")
    with torch.no_grad():
        return paged_chunk_kda(
            q, k, v, gate, beta, pool, slots, kernel_options=options, replay_state=replay
        )


def test_fp16_replay_resolves_the_option_as_its_bf16_cast() -> None:
    """Replay must match a BF16-cast chunk call. A tiny beta would round to zero in FP16,
    so the nonzero output distinguishes which solve actually ran."""
    shape = (1, 1, 1, 128)
    q = torch.zeros(shape, device="cuda")
    q[..., 0] = 1
    v = torch.zeros(shape, device="cuda")
    v[..., 0] = 2.0**15
    gate = torch.zeros(shape, device="cuda")
    beta = torch.full(shape[:3], 2.0**-26, device="cuda")
    zero = torch.zeros(1, 1, 128, 128, device="cuda")
    for value in (None, *VALUES):
        options = WARPDRIVE if value is None else {**WARPDRIVE, "intra_chunk_dtypes": value}
        output = _replay(q.half(), q.half(), v.half(), gate, beta, options)
        cast = (t.bfloat16() for t in (q, q, v))
        with torch.no_grad():
            expected, _ = chunk_kda(*cast, gate, beta, initial_state=zero, kernel_options=options)
        assert torch.equal(output, expected.half()), value
        assert output[0, 0, 0, 0] != 0


PRE_OPTION_OPS = (
    "fwd",
    "fwd_with_initial_state",
    "fwd_with_state",
    "local_bwd",
    "bwd_with_state",
    "bwd_with_exit_cotangent",
)


def _pre_option_args(name, tokens, heads, dtype):
    """Build the positional call saved by old graphs, without an operand precision argument."""
    x = _inputs(tokens, heads, dtype)
    d_output = torch.randn_like(x[2])
    state, d_state = (torch.randn(1, heads, 128, 128, device="cuda") * 0.1 for _ in range(2))
    cu, scale = cumulative_sequence_offsets([tokens]), 128**-0.5
    return {
        "fwd": (*x, cu, False, scale, True),
        "fwd_with_initial_state": (*x, state, cu, scale, True),
        "fwd_with_state": (*x, state, cu, scale, True),
        "local_bwd": (*x, d_output, cu, False, scale, True),
        "bwd_with_state": (*x, d_output, cu, state, d_state, scale, True),
        "bwd_with_exit_cotangent": (*x, d_output, cu, d_state, scale, True),
    }[name]


@pytest.mark.parametrize("name", PRE_OPTION_OPS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def test_pre_option_op_calls_run_the_pre_option_kernels(name, dtype) -> None:
    """An old positional call, with or without fastmath, still selects the legacy precision."""
    op = getattr(torch.ops.attn_gym, f"kda_chunk_cudnn_packed_{name}").default
    args = _pre_option_args(name, 1000, 4, dtype)
    kernels = "bf16" if dtype == torch.bfloat16 else FP16_VALUE
    expected = op(*args, intra_chunk_dtypes=kernels)
    for old_graph in (op(*args), op(*args[:-1])):
        torch.testing.assert_close(old_graph, expected, rtol=0, atol=0)
    torch.library.opcheck(op, args, test_utils=("test_schema", "test_faketensor"))


def test_fused_backend_ignores_intra_chunk_dtypes() -> None:
    inputs = _inputs(64, 2)
    outputs = [
        chunk_kda(*inputs, kernel_options={"backend": "fused", "intra_chunk_dtypes": value})[0]
        for value in VALUES
    ]
    for output in outputs[1:]:
        assert torch.equal(output, outputs[0])


def test_each_value_compiles_its_own_kernels_once_across_shapes() -> None:
    """Changing the precision compiles a new variant; changing token count does not."""
    from attn_gym.linear._delta_rule.warpdrive.bundles import kda_warmup_forward

    compile_fn = kda_warmup_forward._compile_warmup_forward
    compile_fn.cache_clear()
    for value in VALUES:
        for tokens, heads in ((16, 80), (1000, 80), (2048, 80)):  # uncut: B * H fills the GPU
            inputs = _inputs(tokens, heads)
            chunk_kda(*inputs, kernel_options={**WARPDRIVE, "intra_chunk_dtypes": value})
    assert compile_fn.cache_info().currsize == len(VALUES), compile_fn.cache_info()


def test_chain_bundles_compile_one_kernel_set_per_value() -> None:
    """Both chain bundles compile a separate variant for each operand precision."""
    from attn_gym.linear._delta_rule.warpdrive.bundles import kda_chain_backward, kda_chain_forward

    compile_fns = (
        kda_chain_forward._compile_chain_forward,
        kda_chain_backward._compile_chain_backward,
    )
    for compile_fn in compile_fns:
        compile_fn.cache_clear()
    for value in VALUES:
        leaves = [t.requires_grad_() for t in _inputs(4096, 8)]  # the chain plan
        output, _ = chunk_kda(*leaves, kernel_options={**WARPDRIVE, "intra_chunk_dtypes": value})
        torch.autograd.grad(output.float().square().sum(), leaves)
    for compile_fn in compile_fns:
        assert compile_fn.cache_info().currsize == len(VALUES), compile_fn.cache_info()


@pytest.mark.parametrize("tokens, heads, fraction", PLAN_PARAMS)
def test_values_rank_by_operand_precision(tokens, heads, fraction, plan_fraction) -> None:
    """Each value meets its output budget, and widening operands lowers the error: "tf32" beats
    "tf32+bf16", which beats "bf16" (on each forward plan)."""
    plan_fraction(fraction)
    inputs = _inputs(tokens, heads)
    expected = _reference(*inputs)
    errors = {
        value: relative_rms_error(_warpdrive(inputs, value)[0], expected) for value in VALUES
    }
    for value, error in errors.items():
        assert error < BUDGETS[value], errors
    assert errors[TF32] < errors["tf32+bf16"] < errors["bf16"], errors


V_SCALES = {
    "1e5": 1e5,  # beyond FP16
    "1": 1.0,
    "1e-6": 1e-6,  # FP16 subnormal
    "mixed": torch.tensor([1e5] * 32 + [1e-6] * 32 + [1.0] * 64),
}


@pytest.mark.parametrize("v_scale", V_SCALES.values(), ids=V_SCALES.keys())
@pytest.mark.parametrize("tokens, heads, fraction", PLAN_PARAMS)
def test_tf32_solve_operands_follow_the_value_range(
    tokens, heads, fraction, v_scale, plan_fraction
) -> None:
    """All channel groups meet the same relative error budget, even when one group would
    overflow FP16 and another is at FP16 subnormal scale. Gradients must stay finite."""
    plan_fraction(fraction)
    if isinstance(v_scale, torch.Tensor):
        v_scale = v_scale.cuda()
    q, k, v, gate, beta = _inputs(tokens, heads, v_scale=v_scale)
    state_scale = v_scale[:, None] if isinstance(v_scale, torch.Tensor) else v_scale
    state = (torch.randn(1, heads, 128, 128, device="cuda") * state_scale * 0.1).requires_grad_()
    leaves = [t.requires_grad_() for t in (q, k, v, gate, beta)]
    options = {**WARPDRIVE, "intra_chunk_dtypes": TF32}
    output, final = chunk_kda(*leaves, state, output_final_state=True, kernel_options=options)
    assert torch.isfinite(output).all() and torch.isfinite(final).all()
    expected = _reference(q, k, v, gate, beta, state)
    for group in range(0, 128, 32):
        channels = slice(group, group + 32)
        error = relative_rms_error(output[..., channels], expected[..., channels])
        assert error < BUDGETS[TF32], (group, error)
    grads = torch.autograd.grad((output.float() / v_scale).square().sum(), [*leaves, state])
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("poison", [math.nan, math.inf], ids=["nan", "inf"])
@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("tokens, heads, fraction", PLAN_PARAMS)
def test_non_finite_values_stay_in_their_channel(
    tokens, heads, fraction, value, poison, plan_fraction
) -> None:
    """A non-finite value can poison its channel, but not any other channel. Within that
    channel, even a masked product (zero times infinity or NaN) can yield NaN."""
    plan_fraction(fraction)
    inputs = _inputs(tokens, heads)
    clean = _warpdrive(inputs, value)[0]
    q, k, v, gate, beta = inputs
    head, channel, token = heads - 1, 5, 100
    v = v.clone()
    v[0, token, head, channel] = poison
    poisoned = _warpdrive((q, k, v, gate, beta), value)[0]
    assert not torch.isfinite(poisoned[0, token:, head, channel]).any()
    others = torch.ones_like(clean, dtype=torch.bool)
    others[0, :, head, channel] = False
    assert torch.equal(poisoned[others], clean[others])


def _perturb_after(inputs, cuts, gen):
    """Change inputs at or after each head's cut; leave its prefix untouched."""
    q, k, v, gate, beta = (t.clone() for t in inputs)
    fresh = _inputs(q.shape[1], q.shape[2], seed=int(torch.randint(1 << 30, (1,), generator=gen)))
    for head, cut in enumerate(cuts.tolist()):
        later = (0, slice(cut, None), head)
        kind = head % 5
        if kind == 0:
            v[later] = 2.0**40
        elif kind == 1:
            v[later] = 2.0**-20
        elif kind == 2:
            beta[later] = 2.0**-30
        elif kind == 3:
            v[0, cut, head, 3] = 2.0**17
        else:
            for tensor, new in zip((q, k, v, gate, beta), fresh):
                tensor[later] = new[later]
    return q, k, v, gate, beta


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("tokens, heads, fraction", PLAN_PARAMS)
def test_outputs_depend_on_no_later_token(tokens, heads, fraction, value, plan_fraction) -> None:
    """Later finite values must not change earlier output bits, even within the same chunk."""
    plan_fraction(fraction)
    gen = torch.Generator().manual_seed(0)
    for v_scale in (1.0, 1e5):
        base = _inputs(tokens, heads, v_scale=v_scale)
        cuts = 16 * torch.randint(tokens // 16, (heads,), generator=gen)
        cuts += torch.randint(1, 16, (heads,), generator=gen)
        expected = _warpdrive(base, value)[0]
        output = _warpdrive(_perturb_after(base, cuts, gen), value)[0]
        for head, cut in enumerate(cuts.tolist()):
            assert torch.equal(output[0, :cut, head], expected[0, :cut, head]), (v_scale, head)


@pytest.mark.parametrize("plan", ["prep", "per_tile"])
def test_a_later_huge_value_keeps_an_earlier_output(plan, monkeypatch) -> None:
    """With orthogonal keys, even a huge write to the same channel at token 1 must leave
    token 0's output exactly one."""
    if plan == "per_tile":
        from attn_gym.linear._delta_rule.warpdrive.drivers import kda

        monkeypatch.setattr(kda, "PREP_TILE_FRACTION", 0.0)
    q = torch.zeros(1, 2, 1, 128, device="cuda")
    q[0, 0, 0, 0] = q[0, 1, 0, 1] = 1
    gate = torch.zeros_like(q)
    beta = torch.ones(1, 2, 1, device="cuda")
    for later in (0.0, 2.0**15, 2.0**40, 2.0**-20):
        v = torch.zeros_like(q)
        v[0, 0, 0, 0], v[0, 1, 0, 0] = 1, later
        inputs = (q.bfloat16(), q.bfloat16(), v.bfloat16(), gate, beta)
        for value in VALUES:
            output, _ = _warpdrive(inputs, value, scale=1.0)
            assert output[0, 0, 0, 0].item() == 1.0, (later, value)
            assert output[0, 1, 0, 0].item() == v[0, 1, 0, 0].bfloat16().item(), (later, value)


@pytest.mark.parametrize("value", VALUES)
def test_batch_invariant_bits_follow_the_document_not_the_pack(value) -> None:
    """Adding a neighbor changes the forward's value tile width. With a pinned plan, that
    must leave the document's output unchanged."""
    heads = 64
    q = torch.zeros(1, 1, heads, 128, device="cuda")
    q[..., 0] = 1
    v = torch.zeros_like(q)
    v[..., 0], v[..., 16] = 1, 2.0**40
    q, v = q.bfloat16(), v.bfloat16()
    gate = torch.zeros(1, 1, heads, 128, device="cuda")
    beta = torch.ones(1, 1, heads, device="cuda")
    options = {**WARPDRIVE, "batch_invariant": True, "intra_chunk_dtypes": value}
    with torch.no_grad():
        alone, _ = chunk_kda(
            q, q, v, gate, beta, scale=1.0, kernel_options=options,
            cu_seqlens=cumulative_sequence_offsets([1]),
        )  # fmt: skip
        two = (torch.cat([t, t], 1) for t in (q, q, v, gate, beta))
        packed, _ = chunk_kda(
            *two, scale=1.0, kernel_options=options, cu_seqlens=cumulative_sequence_offsets([1, 1])
        )
    assert alone[0, 0, 0, 0].item() == 1.0
    assert torch.equal(alone[0, 0], packed[0, 0])


@pytest.mark.parametrize("batch_invariant", [False, True], ids=["default", "invariant"])
@pytest.mark.parametrize("value", VALUES)
def test_backward_differentiates_a_split_forward(value, batch_invariant) -> None:
    """Token 64 reads the unit state written by token 0 beside a much larger value channel.
    The split forward and full-width backward must both preserve that unit contribution."""
    tokens = 128
    q = torch.zeros(1, tokens, 1, 128, device="cuda")
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    q[0, 0, 0, 0] = k[0, 0, 0, 0] = 1
    v[0, 0, 0, 0], v[0, 0, 0, 16] = 1, 2.0**40
    q[0, 64, 0, 0] = 1
    gate = torch.zeros_like(q)
    beta = torch.zeros(1, tokens, 1, device="cuda")
    beta[0, 0] = 1
    d_output = torch.zeros(1, tokens, 1, 128, device="cuda", dtype=torch.bfloat16)
    d_output[0, 64, 0, 0] = 1
    options = {**WARPDRIVE, "intra_chunk_dtypes": value, "batch_invariant": batch_invariant}
    leaves = [t.bfloat16().requires_grad_() for t in (q, k, v)]
    output, _ = chunk_kda(
        *leaves, gate, beta, scale=1.0, kernel_options=options,
        cu_seqlens=cumulative_sequence_offsets([tokens]),
    )  # fmt: skip
    (dq,) = torch.autograd.grad(output, leaves[0], d_output)
    assert output[0, 64, 0, 0].item() == 1.0
    assert dq[0, 64, 0, 0].item() == 1.0


@pytest.mark.parametrize("tokens, heads, fraction", PLAN_PARAMS)
def test_tf32_solve_operands_keep_large_q_dot_k_finite(
    tokens, heads, fraction, plan_fraction
) -> None:
    """An unscaled diagonal attention score above the FP16 maximum stays finite in both modes."""
    plan_fraction(fraction)
    _, k, v, gate, beta = _inputs(tokens, heads)
    q = (k.float() * 1e5).bfloat16()  # q . k = 1e5 on the diagonal
    for value in (TF32, "tf32+bf16"):
        output = _warpdrive((q, k, v, gate, beta), value)[0]
        assert torch.isfinite(output).all(), value


def _differ(a, b):
    return not torch.equal(a, b)


def test_paged_and_replay_calls_take_the_option() -> None:
    """Paged and replay-backed KDA match an ordinary chunk call at the same precision."""
    cu = cumulative_sequence_offsets((65, 0, 127, 64))
    q, k, v, gate, beta = _inputs(256, 2)
    pool = torch.randn(4, 2, 128, 128, device="cuda") * 0.1
    slots = torch.tensor([1, 3, 2, 0], dtype=torch.int32, device="cuda")
    results = {}
    for value in ("bf16", TF32):
        expected, final = _warpdrive(
            (q, k, v, gate, beta), value, initial_state=pool[[1, 3, 2, 0]], cu_seqlens=cu,
            output_final_state=True,
        )  # fmt: skip
        advanced = pool.clone()
        with torch.no_grad():
            output = paged_chunk_kda(
                q, k, v, gate, beta, advanced, slots, cu_seqlens=cu,
                kernel_options={**WARPDRIVE, "intra_chunk_dtypes": value},
            )  # fmt: skip
        assert torch.equal(output[:, :192], expected[:, :192]), value  # slot 0 is null
        assert torch.equal(advanced[[1, 3, 2]], final[:3]), value
        results[value] = output
    assert _differ(*results.values())

    # Replay: a 100-token prefill, then one token per call, matches one chunk_kda call.
    q, k, v, gate, beta = _inputs(130, 2, batch=2, seed=1)
    slots = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
    results = {}
    for value in ("bf16", TF32):
        options = {**WARPDRIVE, "intra_chunk_dtypes": value}
        expected = _warpdrive((q, k, v, gate, beta), value)[0]
        state_cache = torch.zeros(3, 2, 128, 128, device="cuda")
        replay = ReplayState(
            *(
                torch.zeros((3, *shape), device="cuda", dtype=dtype)
                for shape, dtype in ReplayState.slot_layout(2, 128)
            )
        )
        outputs = []
        with torch.no_grad():
            for start, stop in ((0, 100), *((t, t + 1) for t in range(100, 130))):
                tokens = (t[:, start:stop] for t in (q, k, v, gate, beta))
                outputs.append(
                    paged_chunk_kda(
                        *tokens, state_cache, slots, kernel_options=options, replay_state=replay
                    )
                )
        output = torch.cat(outputs, dim=1)
        assert torch.equal(output, expected), value
        results[value] = output
    assert _differ(*results.values())


# Short sequences summarize whole; the long ones split into pieces (NOTE [Staged piece plans]).
STAGED_LENGTHS = {"whole": [37, 0, 400, 150], "pieces": [37, 0, 4000, 1500]}


@pytest.mark.parametrize("lengths", STAGED_LENGTHS.values(), ids=STAGED_LENGTHS.keys())
def test_staged_calls_take_the_option(lengths) -> None:
    """Compare staged forward, backward and summary results against FP64, for whole sequences
    and for sequences that split into pieces."""
    heads, scale = 2, 128**-0.5
    cu = cumulative_sequence_offsets(lengths)
    q, k, v, gate, beta = _inputs(sum(lengths), heads)
    gate = gate * 0.001  # transitions stay above rounding over thousands of tokens
    d_output = torch.randn_like(v)
    initial = torch.randn(len(lengths), heads, 128, 128, device="cuda") * 0.1
    d_final = torch.randn_like(initial)
    bounds = torch.stack((cu[:-1], cu[1:]), dim=1).contiguous()

    leaves = [t.double().requires_grad_() for t in (q, k, v, gate, beta, initial)]
    output, final = kda_reference(*leaves[:5], leaves[5], cu_seqlens=cu, scale=scale)
    grads = torch.autograd.grad(
        (output * d_output.double()).sum() + (final * d_final.double()).sum(), leaves
    )
    zero = torch.zeros_like(leaves[5], requires_grad=True)
    zero_output, zero_final = kda_reference(*leaves[:5], zero, cu_seqlens=cu, scale=scale)
    (entry_cotangent,) = torch.autograd.grad((zero_output * d_output.double()).sum(), zero)
    references = {
        "output": output,
        "final_state": final,
        "B": zero_final,
        **dict(zip(("dq", "dk", "dv", "dgate", "dbeta", "d_initial_state"), grads)),
        "C": entry_cotangent,
    }

    results = {}
    for value in ("bf16", TF32):
        options = {**WARPDRIVE, "intra_chunk_dtypes": value}
        prepared = chunk_kda_prepare(q, k, v, gate, beta, cu_seqlens=cu, kernel_options=options)
        backward = chunk_kda_prepare_backward(
            prepared.saved, d_output, initial, scale=prepared.scale, kernel_options=options
        )
        assert (prepared.pieces is None) == (lengths is STAGED_LENGTHS["whole"])
        actual = dict(
            zip(("output", "final_state"), prepared.run(initial, output_final_state=True))
        )
        actual["B"] = prepared.state_summaries(bounds)[:, :, :128]
        actual.update(
            zip(("dq", "dk", "dv", "dgate", "dbeta", "d_initial_state"), backward.run(d_final))
        )
        actual["C"] = backward.state_grad_summaries(bounds)[:, :, :128]
        for name, tensor in actual.items():
            error = relative_rms_error(tensor, references[name])
            assert error < 6e-3, (value, name, error)
        results[value] = actual
    # The FP64 checks cover every result; these comparisons also detect an ignored option.
    for name in ("output", "final_state", "B", "dq", "dk", "dgate", "dbeta"):
        assert _differ(results["bf16"][name], results[TF32][name]), name


@pytest.mark.parametrize("value", VALUES)
def test_backward_operand_precision_matches_fp64(value) -> None:
    inputs = _inputs(513, 4, seed=3)
    generator = torch.Generator(device="cuda").manual_seed(19)
    d_output = torch.randn(
        inputs[2].shape, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    leaves = tuple(t.double().requires_grad_() for t in inputs)
    expected, _ = naive_recurrent_kda(*leaves[:3], leaves[3] / math.log(2), leaves[4])
    reference = torch.autograd.grad(expected, leaves, d_output.double())

    def gradients(options):
        leaves = tuple(t.detach().requires_grad_() for t in inputs)
        output, _ = chunk_kda(*leaves, kernel_options=options)
        return torch.autograd.grad(output, leaves, d_output)

    actual = gradients({**WARPDRIVE, "intra_chunk_dtypes": value})
    fused = gradients({"backend": "fused", "autotune": False})
    names = ("dq", "dk", "dv", "dgate", "dbeta")
    errors = {
        name: (relative_rms_error(result, ref), relative_rms_error(baseline, ref))
        for name, result, baseline, ref in zip(names, actual, fused, reference, strict=True)
    }
    if value == "bf16":
        # BF16 decay operands have a separate error budget, especially for dgate.
        assert all(error <= 8.5e-3 for error, _ in errors.values()), errors
    else:
        assert all(error <= baseline + 4e-4 for error, baseline in errors.values()), errors


@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("tokens, cut", [(17, 16), (1024, 512)], ids=["ragged", "aligned"])
def test_backward_recompute_repeats_the_forward_states(value, tokens, cut, monkeypatch) -> None:
    """On uncut plans, passing the forward state across a chunk boundary must produce the
    same input and entry-state gradients as recomputing that state in one backward call."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda

    forward, backward = kda.ForwardPlan.build, kda.BackwardPlan.build
    monkeypatch.setattr(
        kda.ForwardPlan, "build", lambda *a: replace(forward(*a), pieces=0, tiles_per_head=1)
    )
    monkeypatch.setattr(kda.BackwardPlan, "build", lambda *a: replace(backward(*a), pieces=0))
    heads, scale = 4, 128**-0.5
    q, k, v, gate, beta = _inputs(tokens, heads)
    d_output = torch.randn_like(v)
    state = torch.randn(1, heads, 128, 128, device="cuda") * 0.1
    d_state = torch.randn_like(state)
    whole = cumulative_sequence_offsets([tokens])
    left = cumulative_sequence_offsets([cut])
    right = cumulative_sequence_offsets([tokens - cut])
    operands = (q, k, v, gate, beta, d_output)
    expected = chunk_warpdrive_packed_bwd_with_state_op(
        *operands, whole, state, d_state, scale, intra_chunk_dtypes=value
    )
    head = tuple(t[:, :cut].contiguous() for t in operands)
    tail = tuple(t[:, cut:].contiguous() for t in operands)
    _, handoff = chunk_warpdrive_packed_fwd_with_state_op(
        *head[:5], state, left, scale, intra_chunk_dtypes=value
    )
    tail_grads = chunk_warpdrive_packed_bwd_with_state_op(
        *tail, right, handoff, d_state, scale, intra_chunk_dtypes=value
    )
    head_grads = chunk_warpdrive_packed_bwd_with_state_op(
        *head, left, state, tail_grads[5], scale, intra_chunk_dtypes=value
    )
    for index in range(5):
        joined = torch.cat((head_grads[index], tail_grads[index]), 1)
        assert torch.equal(joined, expected[index]), index
    assert torch.equal(head_grads[5], expected[5])


SCALE_CASES = pytest.mark.parametrize(
    "dtype, value",
    [(torch.bfloat16, "bf16"), (torch.bfloat16, "tf32+bf16"), (torch.float16, FP16_VALUE)],
    ids=["bf16", "tf32+bf16", "fp16"],
)


@SCALE_CASES
def test_backward_applies_scale_outside_the_forward_tiles(dtype, value) -> None:
    """The forward rounds ``Q 2^g`` and A unscaled and applies the 2^-3.5 scale to the output,
    so the backward scales the cotangent: one token with q = k = e0, beta 1 has dv =
    io(scale dO) (not ``dO`` times the rounded ``scale``), and its entry-state cotangent ``scale dO
    q - dY k`` cancels exactly, from the bprop kernel and from the reverse summary."""
    scale = 128**-0.5
    q = torch.zeros(1, 1, 1, 128, device="cuda", dtype=dtype)
    q[..., 0] = 1
    v = torch.randn(q.shape, device="cuda").to(dtype)
    gate = torch.zeros(q.shape, device="cuda")
    beta = torch.ones(1, 1, 1, device="cuda")
    d_output = (torch.arange(1, 129, device="cuda") * 1.37).to(dtype).view(q.shape)
    options = {**WARPDRIVE, "intra_chunk_dtypes": value}
    initial = torch.zeros(1, 1, 128, 128, device="cuda", requires_grad=True)
    leaves = [t.clone().requires_grad_() for t in (q, q, v)]
    output, _ = chunk_kda(*leaves, gate, beta, initial_state=initial, kernel_options=options)
    dv, d_initial = torch.autograd.grad(output, (leaves[2], initial), d_output)
    expected = (d_output.float() * scale).to(dtype)
    assert torch.equal(dv, expected), (dv - expected).abs().max()
    assert not torch.equal(expected, (d_output.float() * torch.tensor(scale).to(dtype)).to(dtype))
    assert torch.count_nonzero(d_initial) == 0, d_initial.abs().max()

    cu = cumulative_sequence_offsets([1])
    prepared = chunk_kda_prepare(q, q, v, gate, beta, cu_seqlens=cu, kernel_options=options)
    backward = chunk_kda_prepare_backward(
        prepared.saved, d_output, initial.detach(), scale=prepared.scale, kernel_options=options
    )
    bounds = torch.stack((cu[:-1], cu[1:]), dim=1).contiguous()
    summary = backward.state_grad_summaries(bounds)[:, :, :128]
    assert torch.count_nonzero(summary) == 0, summary.abs().max()


@SCALE_CASES
def test_key_gradient_scales_the_score_cotangent(dtype, value) -> None:
    """dK's score term pairs the forward's unscaled ``Q 2^g`` with the scaled score cotangent:
    two tokens per head (q1 = k0 = e0, k1 = e1, q0 = 0, v0 = x e0, dO1 = y e0) give
    dk0[0] = io(scale x y) (TF32-rounded first for TF32 decay), not io(x y) times
    io(scale). Except under "tf32+bf16", dQ's score term reads the same rounded tile: dq1[0]
    equals dk0[0]."""
    heads, scale = 64, 128**-0.5
    shape = (1, 2, heads, 128)
    q = torch.zeros(shape, device="cuda", dtype=dtype)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    d_output = torch.zeros_like(q)
    q[0, 1, :, 0] = k[0, 0, :, 0] = k[0, 1, :, 1] = 1
    x = (1 + torch.arange(heads, device="cuda") / 8).to(dtype)
    y = torch.tensor(1.37, device="cuda").to(dtype)
    v[0, 0, :, 0], d_output[0, 1, :, 0] = x, y
    gate = torch.zeros(shape, device="cuda")
    beta = torch.ones(shape[:3], device="cuda")
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    output, _ = chunk_kda(
        *leaves, gate, beta, kernel_options={**WARPDRIVE, "intra_chunk_dtypes": value}
    )
    dq, dk = torch.autograd.grad(output, leaves[:2], d_output)
    product = x.float() * y.float()
    operand = product * scale
    if value != "bf16":  # a TF32 score-cotangent tile: half an ulp, then the MMA truncates
        operand = ((operand.view(torch.int32) + 0x1000) & ~0x1FFF).view(torch.float32)
    expected = operand.to(dtype)
    assert torch.equal(dk[0, 0, :, 0], expected), dk[0, 0, :, 0] - expected
    rounded = (product.to(dtype).float() * torch.tensor(scale).to(dtype).float()).to(dtype)
    assert not torch.equal(expected, rounded)
    if value != "tf32+bf16":
        assert torch.equal(dq[0, 1, :, 0], expected), dq[0, 1, :, 0] - expected


@SCALE_CASES
def test_state_cotangent_meets_the_forward_restore_key(dtype, value) -> None:
    """The forward updates the state with ``K_restore = io(K 2^(g_last - g))``, so the
    backward pairs that tile with the unscaled state cotangent: with k0 = e0, a 2^g = 0.7 step on
    token 1 and q16 = e0 reading the state (scale 1), dv0 = io(dO16 io(0.7)), not io(dO16 0.7)
    from io(2^g_last dS) times K 2^-g. The reverse summary meets the same tiles, so its entry
    cotangent equals the bprop kernel's."""
    shape = (1, 17, 1, 128)
    q = torch.zeros(shape, device="cuda", dtype=dtype)
    k = torch.zeros_like(q)
    q[0, 16, 0, 0] = k[0, 0, 0, 0] = 1
    v = torch.randn(shape, device="cuda").to(dtype)
    gate = torch.zeros(shape, device="cuda")
    gate[0, 1, 0, 0] = math.log(0.7)
    beta = torch.ones(shape[:3], device="cuda")
    d_output = torch.zeros_like(q)
    d_output[0, 16, 0] = (torch.arange(1, 129, device="cuda") * 1.37).to(dtype)
    options = {**WARPDRIVE, "intra_chunk_dtypes": value}
    initial = torch.zeros(1, 1, 128, 128, device="cuda", requires_grad=True)
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    output, _ = chunk_kda(
        *leaves, gate, beta, scale=1.0, initial_state=initial, kernel_options=options
    )
    dv, d_initial = torch.autograd.grad(output, (leaves[2], initial), d_output)
    restore = torch.tensor(0.7).to(dtype).float()
    expected = (d_output[0, 16, 0].float() * restore).to(dtype)
    assert torch.equal(dv[0, 0, 0], expected), dv[0, 0, 0] - expected
    assert not torch.equal(expected, (d_output[0, 16, 0].float() * 0.7).to(dtype))

    cu = cumulative_sequence_offsets([17])
    prepared = chunk_kda_prepare(
        q, k, v, gate, beta, scale=1.0, cu_seqlens=cu, kernel_options=options
    )
    backward = chunk_kda_prepare_backward(
        prepared.saved, d_output, initial.detach(), scale=prepared.scale, kernel_options=options
    )
    bounds = torch.stack((cu[:-1], cu[1:]), dim=1).contiguous()
    summary = backward.state_grad_summaries(bounds)[:, :, :128]
    assert torch.equal(summary, d_initial), (summary - d_initial).abs().max()


def test_fp16_key_gradient_keeps_the_cancelling_score_terms_finite() -> None:
    """``dA`` and ``beta dM`` of an FP16 call can exceed FP16's range and still cancel in dK:
    two tokens with k0 = e0 / 4, q1 = k1 = e1, v0 = 512 e0 and dO1 = 256 e0 (scale and beta 1)
    give both 131072 at (1, 0). The FP64 recurrence has dk0 = 0 and dk1 = -32768 e0, which an
    FP16 cast of ``beta dM`` before its TF32 multiply turns into infinities."""
    shape = (1, 2, 1, 128)
    q, k, v, d_output = (torch.zeros(shape, device="cuda", dtype=torch.float16) for _ in range(4))
    k[0, 0, 0, 0] = 0.25
    q[0, 1, 0, 1] = k[0, 1, 0, 1] = 1
    v[0, 0, 0, 0], d_output[0, 1, 0, 0] = 512, 256
    gate = torch.zeros(shape, device="cuda")
    beta = torch.ones(shape[:3], device="cuda")
    leaves = [t.clone().requires_grad_() for t in (q, k, v)]
    output, _ = chunk_kda(*leaves, gate, beta, scale=1.0, kernel_options=WARPDRIVE)
    dk = torch.autograd.grad(output, leaves[1], d_output)[0]
    expected = torch.zeros_like(dk)
    expected[0, 1, 0, 0] = -32768
    assert torch.equal(dk, expected), dk[0, :, 0, :2]


def test_fp16_query_state_term_keeps_a_scaled_cotangent_finite() -> None:
    """With ``scale`` above 1, ``scale dO`` can exceed FP16's range although the entry-state
    gradient does not: q = e0 / 2, scale 2 and dO = 32768 e0 give d_initial[0, 0] = 32768.
    The Q state term moves the scale's power of two onto Q_decay, so one token with
    q = k = e0, beta 1 and scale 3 still cancels its entry-state cotangent exactly, from the bprop
    kernel and from the reverse summary."""
    shape = (1, 1, 1, 128)
    gate = torch.zeros(shape, device="cuda")
    cu = cumulative_sequence_offsets([1])
    bounds = torch.stack((cu[:-1], cu[1:]), dim=1).contiguous()
    unit = torch.zeros(shape, device="cuda", dtype=torch.float16)
    unit[..., 0] = 1
    large = unit * 32768
    ramp = (torch.arange(1, 129, device="cuda") * 1.37).half().view(shape)
    cases = [  # q, k, beta, scale, dO, expected d_initial[0, 0, 0]
        (unit / 2, torch.zeros_like(unit), 0.0, 2.0, large, large[0, 0, 0].float()),
        (unit, unit, 1.0, 3.0, ramp, torch.zeros(128, device="cuda")),
    ]
    for q, k, beta_value, scale, d_output, expected in cases:
        beta = torch.full(shape[:3], beta_value, device="cuda")
        v = torch.randn(shape, device="cuda").half()
        initial = torch.zeros(1, 1, 128, 128, device="cuda", requires_grad=True)
        output, _ = chunk_kda(q, k, v, gate, beta, initial, scale=scale, kernel_options=WARPDRIVE)
        (d_initial,) = torch.autograd.grad(output, initial, d_output)
        assert torch.equal(d_initial[0, 0, :, 0], expected), d_initial[0, 0, :, 0]
        assert torch.count_nonzero(d_initial[0, 0, :, 1:]) == 0

        prepared = chunk_kda_prepare(
            q, k, v, gate, beta, scale=scale, cu_seqlens=cu, kernel_options=WARPDRIVE
        )
        backward = chunk_kda_prepare_backward(
            prepared.saved, d_output, initial.detach(), scale=scale, kernel_options=WARPDRIVE
        )
        summary = backward.state_grad_summaries(bounds)[:, :, :128]
        assert torch.equal(summary, d_initial), (summary - d_initial).abs().max()
