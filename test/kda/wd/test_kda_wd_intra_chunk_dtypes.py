"""``kernel_options["intra_chunk_dtypes"]``: validation, defaults, routing, compile keys,
accuracy and range, non-finite values, recompute match."""

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

VALUES = ("tf32+fp16", "tf32+bf16", "bf16+fp16", "bf16")
FP16_SOLVE = VALUES[0]  # the widest operands, FP16 inputs' one kernel set
WARPDRIVE = {"backend": "warpdrive"}
# Output relative RMS budgets against FP64 on ``_inputs`` (measured: 1.74e-3, 3.27e-3, 2.89e-3,
# 4.00e-3 at 8 x 2048 H32; the BF16 rounding of the exact output alone is 1.66e-3).
BUDGETS = {"tf32+fp16": 2.5e-3, "tf32+bf16": 4.5e-3, "bf16+fp16": 4e-3, "bf16": 5.5e-3}
# One shape per forward plan: (tokens, heads).
PLANS = {"chain": (16384, 32), "prep": (2048, 8), "uncut": (4096, 80)}
PLAN_PARAMS = [pytest.param(*shape, id=name) for name, shape in PLANS.items()]


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


@pytest.mark.parametrize(
    "dtype, expected",
    [(torch.bfloat16, "tf32+bf16"), (torch.float16, FP16_SOLVE)],
    ids=["bf16", "fp16"],
)
def test_omitted_option_runs_the_default_of_the_input_dtype(dtype, expected) -> None:
    """Without the option, BF16 inputs run "tf32+bf16" and FP16 inputs their one kernel set: the
    forward, gradients and staged forward and backward equal the explicit value's bitwise."""
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
    """FP16 Q/K/V accept only their one value on every backend and entry point, so a call's
    validity never depends on the backend or GPU; under ``torch.compile`` the error comes from
    tracing, since the ops themselves accept every value."""
    q, k, v, gate, beta = _inputs(64, 2, torch.float16)
    pool = torch.zeros(2, 2, 128, 128, device="cuda")
    slots = torch.ones(1, dtype=torch.int32, device="cuda")
    cu = cumulative_sequence_offsets([64])
    message = r"FP16 q/k/v take only .*'tf32\+fp16'"
    for value in VALUES[1:]:
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


PRE_OPTION_OPS = (
    "fwd",
    "fwd_with_initial_state",
    "fwd_with_state",
    "local_bwd",
    "bwd_with_state",
    "bwd_with_exit_cotangent",
)


def _pre_option_args(name, tokens, heads, dtype):
    """The op's arguments as graphs serialized before the option pass them: ``fastmath``
    positionally right after ``scale``, and no ``intra_chunk_dtypes``."""
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
    """Pre-option calls, with ``fastmath`` or without it, bind as before: the schema default
    "bf16" runs the BF16 kernels of that time, and FP16 operands keep their one kernel set
    (bitwise)."""
    op = getattr(torch.ops.attn_gym, f"kda_chunk_cudnn_packed_{name}").default
    args = _pre_option_args(name, 1000, 4, dtype)
    kernels = "bf16" if dtype == torch.bfloat16 else FP16_SOLVE
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
    """The option is a compile key (one entry per value) that token and head counts reuse."""
    from attn_gym.linear._delta_rule.warpdrive.bundles import kda_warmup_forward

    compile_fn = kda_warmup_forward._compile_warmup_forward
    compile_fn.cache_clear()
    for value in VALUES:
        for tokens, heads in ((16, 80), (1000, 80), (2048, 80)):  # uncut: B * H fills the GPU
            inputs = _inputs(tokens, heads)
            chunk_kda(*inputs, kernel_options={**WARPDRIVE, "intra_chunk_dtypes": value})
    assert compile_fn.cache_info().currsize == len(VALUES), compile_fn.cache_info()


def test_chain_bundles_compile_one_kernel_set_per_value() -> None:
    """The chain plan's forward and backward bundles (with the backward's checkpoint recompute)
    key on the option."""
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


@pytest.mark.parametrize("tokens, heads", PLAN_PARAMS)
def test_values_rank_by_operand_precision(tokens, heads) -> None:
    """Each value meets its output budget, and widening either operand class lowers the error:
    "tf32+fp16" beats both one-sided values, which beat "bf16" (on each forward plan)."""
    inputs = _inputs(tokens, heads)
    expected = _reference(*inputs)
    errors = {
        value: relative_rms_error(_warpdrive(inputs, value)[0], expected) for value in VALUES
    }
    for value, error in errors.items():
        assert error < BUDGETS[value], errors
    for one_sided in ("tf32+bf16", "bf16+fp16"):
        assert errors[FP16_SOLVE] < errors[one_sided] < errors["bf16"], errors


V_SCALES = {
    "1e5": 1e5,  # overflows FP16
    "1": 1.0,
    "1e-6": 1e-6,  # FP16 subnormal
    # Value channels are TMEM lanes, 32 per warp: warps pick different exponents.
    "mixed": torch.tensor([1e5] * 32 + [1e-6] * 32 + [1.0] * 64),
}


@pytest.mark.parametrize("v_scale", V_SCALES.values(), ids=V_SCALES.keys())
@pytest.mark.parametrize("tokens, heads", PLAN_PARAMS)
def test_fp16_solve_operands_follow_the_value_range(tokens, heads, v_scale) -> None:
    """The FP16 solve operands carry exact per-warp power-of-two scales: values far outside the
    FP16 range (1e5 overflows, 1e-6 is subnormal), or both in one call, keep the accuracy of unit
    values in every 32-channel group, and their gradients stay finite."""
    if isinstance(v_scale, torch.Tensor):
        v_scale = v_scale.cuda()
    q, k, v, gate, beta = _inputs(tokens, heads, v_scale=v_scale)
    state_scale = v_scale[:, None] if isinstance(v_scale, torch.Tensor) else v_scale
    state = (torch.randn(1, heads, 128, 128, device="cuda") * state_scale * 0.1).requires_grad_()
    leaves = [t.requires_grad_() for t in (q, k, v, gate, beta)]
    options = {**WARPDRIVE, "intra_chunk_dtypes": FP16_SOLVE}
    output, final = chunk_kda(*leaves, state, output_final_state=True, kernel_options=options)
    assert torch.isfinite(output).all() and torch.isfinite(final).all()
    expected = _reference(q, k, v, gate, beta, state)
    for group in range(0, 128, 32):
        channels = slice(group, group + 32)
        error = relative_rms_error(output[..., channels], expected[..., channels])
        assert error < BUDGETS[FP16_SOLVE], (group, error)
    grads = torch.autograd.grad((output.float() / v_scale).square().sum(), [*leaves, state])
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("poison", [math.nan, math.inf], ids=["nan", "inf"])
@pytest.mark.parametrize("value", VALUES)
@pytest.mark.parametrize("tokens, heads", PLAN_PARAMS)
def test_non_finite_values_stay_in_their_channel(tokens, heads, value, poison) -> None:
    """A NaN or inf in one value channel propagates through that channel only, as in BF16: every
    other output channel is bitwise the clean run's. (Earlier tokens of its 16-token chunk may
    turn NaN too: the masked A U product multiplies the NaN by zero.)"""
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


def test_fp16_solve_scale_is_shared_by_a_warps_value_channels() -> None:
    """The 32 value channels a warp packs share one FP16 scale: a one-token output of 1e-10 beside
    1e5 flushes, while the same value in another warp, or under BF16 solve operands, survives."""
    q = torch.zeros(1, 1, 2, 128, dtype=torch.bfloat16, device="cuda")
    q[..., 0] = 1
    v = torch.zeros_like(q)
    v[..., 0], v[..., 1], v[..., 32] = 1e5, 1e-10, 1e-10
    gate = torch.zeros_like(q, dtype=torch.float32)
    beta = torch.ones(1, 1, 2, device="cuda")
    tiny = v[0, 0, 0, 1].item()
    for value, kept in (("tf32+fp16", 0.0), ("bf16+fp16", 0.0), ("tf32+bf16", tiny)):
        output, _ = _warpdrive((q, q, v, gate, beta), value, scale=1.0)
        assert output[0, 0, 0, 1].item() == kept, value
        assert output[0, 0, 0, 32].item() == tiny, value


@pytest.mark.parametrize("tokens, heads", PLAN_PARAMS)
def test_fp16_solve_operands_bound_q_dot_k(tokens, heads) -> None:
    """A and T^-1 are plain FP16 under FP16 solve operands: the forward stays finite while
    |q . k| < 65504 and overflows past it; BF16 solve operands keep BF16 range."""
    _, k, v, gate, beta = _inputs(tokens, heads)
    for magnitude, finite in ((3e4, True), (1e5, False)):
        q = (k.float() * magnitude).bfloat16()  # q . k = magnitude on the diagonal
        output = _warpdrive((q, k, v, gate, beta), FP16_SOLVE)[0]
        assert torch.isfinite(output).all().item() is finite, magnitude
    output = _warpdrive(((k.float() * 1e5).bfloat16(), k, v, gate, beta), "tf32+bf16")[0]
    assert torch.isfinite(output).all()


def _differ(a, b):
    return not torch.equal(a, b)


def test_paged_and_replay_calls_take_the_option() -> None:
    """Paged and replay-backed calls run the value's kernels: their bits equal ``chunk_kda``'s
    under the same value (whose accuracy the ranking test covers) and differ between values."""
    cu = cumulative_sequence_offsets((65, 0, 127, 64))
    q, k, v, gate, beta = _inputs(256, 2)
    pool = torch.randn(4, 2, 128, 128, device="cuda") * 0.1
    slots = torch.tensor([1, 3, 2, 0], dtype=torch.int32, device="cuda")
    results = {}
    for value in ("bf16", FP16_SOLVE):
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
    for value in ("bf16", FP16_SOLVE):
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
    """``chunk_kda_prepare`` (``run``, ``state_summaries``) and ``chunk_kda_prepare_backward``
    (``run``, ``state_grad_summaries``) run the value's kernels, whole or in pieces: the results
    that read forward states differ between values, and all stay within their FP64 budgets."""
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
    for value in ("bf16", FP16_SOLVE):
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
            assert error < 6e-3, (value, name, error)  # measured at most 4.6e-3
        results[value] = actual
    # Whole sequences' dv, d_initial_state and C read no forward state (the pieces' do).
    for name in ("output", "final_state", "B", "dq", "dk", "dgate", "dbeta"):
        assert _differ(results["bf16"][name], results[FP16_SOLVE][name]), name


@pytest.mark.parametrize("value", VALUES)
def test_backward_recompute_repeats_the_forward_states(value, monkeypatch) -> None:
    """The checkpoint recompute takes the forward's operand dtypes, so a state handed over
    between two calls equals the one the backward of a single call recomputes: the joined
    gradients are bitwise the single call's (uncut plans)."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import kda

    forward, backward = kda.ForwardPlan.build, kda.BackwardPlan.build
    monkeypatch.setattr(
        kda.ForwardPlan, "build", lambda *a: replace(forward(*a), pieces=0, tiles_per_head=1)
    )
    monkeypatch.setattr(kda.BackwardPlan, "build", lambda *a: replace(backward(*a), pieces=0))
    tokens, cut, heads, scale = 1024, 512, 4, 128**-0.5
    q, k, v, gate, beta = _inputs(tokens, heads)
    d_output = torch.randn_like(v)
    state = torch.randn(1, heads, 128, 128, device="cuda") * 0.1
    d_state = torch.randn_like(state)
    whole = cumulative_sequence_offsets([tokens])
    half = cumulative_sequence_offsets([cut])
    operands = (q, k, v, gate, beta, d_output)
    expected = chunk_warpdrive_packed_bwd_with_state_op(
        *operands, whole, state, d_state, scale, intra_chunk_dtypes=value
    )
    head = tuple(t[:, :cut].contiguous() for t in operands)
    tail = tuple(t[:, cut:].contiguous() for t in operands)
    _, handoff = chunk_warpdrive_packed_fwd_with_state_op(
        *head[:5], state, half, scale, intra_chunk_dtypes=value
    )
    tail_grads = chunk_warpdrive_packed_bwd_with_state_op(
        *tail, half, handoff, d_state, scale, intra_chunk_dtypes=value
    )
    head_grads = chunk_warpdrive_packed_bwd_with_state_op(
        *head, half, state, tail_grads[5], scale, intra_chunk_dtypes=value
    )
    for index in range(5):
        joined = torch.cat((head_grads[index], tail_grads[index]), 1)
        assert torch.equal(joined, expected[index]), index
    assert torch.equal(head_grads[5], expected[5])
