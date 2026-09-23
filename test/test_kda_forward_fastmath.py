"""Forward exponential policy, specialization isolation, and recomputation regressions."""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from attn_gym.linear import chunk_kda, gate_transform
from attn_gym.linear.kda import bound_gate, ops
from attn_gym.linear.kda.constants import LOG2_E, is_sm100_kda_capability
from attn_gym.linear.kda.impl import fused
from attn_gym.linear.kda.naive import chunk_cumsum_ref
from attn_gym.testing.kda import (
    assert_matches_low_precision_reference,
    assert_rms_matches_low_precision_reference,
    clone_kda_inputs,
    cumulative_sequence_offsets,
    kda_reference,
    make_kda_test_inputs,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the fused KDA core requires CUDA capability 8.0 or newer",
)
requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_sm100_kda_capability(torch.cuda.get_device_capability()),
    reason="the BF16 KDA intra engine requires SM100/SM103",
)


def test_public_math_defaults_allow_approximations(monkeypatch):
    api = importlib.import_module("attn_gym.linear.kda.api")
    received = []

    def forward(q, *_args, **kwargs):
        received.append(kwargs["fastmath"])
        return q, None

    monkeypatch.setattr(api, "_fused_chunk_forward", forward)
    q = torch.zeros(1, 4, 1, 128)
    beta = torch.ones(1, 4, 1)
    chunk_kda(q, q, q, -q, beta)
    chunk_kda(q, q, q, -q, beta, fastmath=False)
    assert received == [True, False]
    for operation in (chunk_kda, gate_transform, bound_gate, fused.chunk_forward):
        assert inspect.signature(operation).parameters["fastmath"].default is True


def test_reference_accepts_fastmath_without_changing_outputs_or_gradients():
    torch.manual_seed(12)
    shape = (1, 4, 1, 4)
    base = (
        torch.randn(shape),
        torch.randn(shape) / 4,
        torch.randn(shape),
        -torch.rand(shape),
        torch.rand(shape[:-1]),
    )
    expected = None
    for kwargs in ({"fastmath": False}, {"fastmath": True}, {}):
        inputs = tuple(x.clone().requires_grad_() for x in base)
        output, state = chunk_kda(*inputs, impl="reference", output_final_state=True, **kwargs)
        assert state is not None
        gradients = torch.autograd.grad(output.sum() + state.sum(), inputs)
        values = (output, state, *gradients)
        if expected is not None:
            for actual, reference in zip(values, expected, strict=True):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        expected = values


@pytest.mark.parametrize("kind", ["bounded", "softplus"])
def test_reference_gate_permission_preserves_outputs_and_gradients(kind):
    torch.manual_seed(23)
    base = (torch.randn(1, 4, 1, 128), torch.randn(1), torch.randn(1, 128))
    operation = bound_gate if kind == "bounded" else gate_transform
    options = {} if kind == "bounded" else {"kind": kind}
    expected = None
    for kwargs in ({"fastmath": False}, {"fastmath": True}, {}):
        inputs = tuple(x.clone().requires_grad_() for x in base)
        output = operation(*inputs, impl="reference", **options, **kwargs)
        values = (output, *torch.autograd.grad(output.sum(), inputs))
        if expected is not None:
            for actual, reference in zip(values, expected, strict=True):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        expected = values


def test_cudnn_precision_request_is_rejected_before_cuda_dispatch(monkeypatch):
    cudnn = importlib.import_module("attn_gym.linear.kda.impl.cudnn")

    def reached_native(_q):
        raise RuntimeError("reached native cuDNN dispatch")

    monkeypatch.setattr(cudnn, "validate_cudnn_available", reached_native)
    q = torch.zeros(1, 64, 1, 128, dtype=torch.bfloat16)
    beta = torch.ones(1, 64, 1)
    gate = torch.zeros_like(q, dtype=torch.float32)
    with pytest.raises(ValueError, match="cuDNN KDA cannot honor fastmath=False"):
        chunk_kda(q, q, q, gate, beta, kernel_options={"backend": "cudnn"}, fastmath=False)
    for kwargs in ({}, {"fastmath": True}):
        with pytest.raises(RuntimeError, match="reached native cuDNN dispatch"):
            chunk_kda(q, q, q, gate, beta, kernel_options={"backend": "cudnn"}, **kwargs)


def test_kernel_exponentials_use_an_explicit_policy():
    """Catch a fixed math call before paying for CuTe/Triton compilation."""
    root = Path(__file__).resolve().parents[1] / "attn_gym/linear"
    kernels = (
        "kda/fwd/cute/chunk_kda_fwd_intra_engine.py",
        "kda/fwd/cute/chunk_kda_k3b_offdiag_cutedsl.py",
        "kda/bwd/cute/chunk_kda_bwd_intra.py",
        "kda/bwd/cute/chunk_delta_h_bwd.py",
        "kda/bwd/cute/chunk_kda_bwd_wy_dqkg_fused.py",
        "kda/fwd/triton/chunk_kda_fwd_intra_sub_chunk_forloop.py",
        "kda/fwd/triton/chunk_kda_fwd_k3_triton.py",
        "kda/fwd/triton/recompute_w_u.py",
        "kda/fwd/triton/chunk_delta_h.py",
        "kda/fwd/triton/chunk_gla_fwd_o.py",
        "kda/bwd/triton/chunk_kda_bwd_delta_h_triton.py",
        "_delta_rule/cute/affine_summary_fwd.py",
        "_delta_rule/cute/affine_summary_rev.py",
        "_delta_rule/triton/affine_summary_fwd.py",
        "_delta_rule/triton/affine_summary_rev.py",
    )
    for kernel in kernels:
        tree = ast.parse((root / kernel).read_text())
        policy_positions = {"exp2": 1, "exp": 1, "masked_exp2": 2}
        policy_positions.update(
            (node.name, [arg.arg for arg in node.args.args].index("FASTMATH"))
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and any(arg.arg == "FASTMATH" for arg in node.args.args)
        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            location = f"{kernel}:{node.lineno}"
            if isinstance(node.func, ast.Attribute) and node.func.attr in ("exp2", "exp"):
                policy = next((kw.value for kw in node.keywords if kw.arg == "fastmath"), None)
            elif isinstance(node.func, ast.Name) and node.func.id in policy_positions:
                position = policy_positions[node.func.id]
                policy = (
                    node.args[position]
                    if len(node.args) > position
                    else next((kw.value for kw in node.keywords if kw.arg == "FASTMATH"), None)
                )
            else:
                continue
            assert policy is not None and not isinstance(policy, ast.Constant), location


def test_triton_launches_control_libdevice_ftz():
    """libdevice.exp2 still flushes unless the launch also disables NVVM reflect FTZ."""
    root = Path(__file__).resolve().parents[1] / "attn_gym/linear"
    files = (
        *root.glob("kda/fwd/triton/*.py"),
        *root.glob("kda/bwd/triton/*.py"),
        *root.glob("_delta_rule/triton/affine_summary*.py"),
        root / "_delta_rule/triton/softplus_gate.py",
    )
    checked = 0
    for file in files:
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Call):
                options = {kw.arg: kw.value for kw in node.keywords}
            elif isinstance(node, ast.Dict):
                options = {
                    key.value: value
                    for key, value in zip(node.keys, node.values, strict=True)
                    if isinstance(key, ast.Constant)
                }
            else:
                continue
            if "FASTMATH" not in options:
                continue
            location = f"{file.name}:{node.lineno}"
            assert "enable_reflect_ftz" in options, location
            assert ast.dump(options["FASTMATH"]) == ast.dump(options["enable_reflect_ftz"]), (
                location
            )
            checked += 1
    assert checked > 0


@pytest.mark.parametrize("packed", [False, True], ids=["dense", "ragged"])
@pytest.mark.parametrize("with_state", [False, True], ids=["output", "output-state"])
def test_autograd_forward_passes_fastmath(monkeypatch, packed, with_state):
    """Exercise all four dispatch branches without importing CUDA kernel dependencies."""
    q = torch.ones(1, 64, 1, 128, requires_grad=True)
    beta = torch.ones(1, 64, 1)
    offsets = torch.tensor([0, 64], dtype=torch.int32) if packed else None
    chunks = torch.tensor([0, 1], dtype=torch.int32) if packed else None
    received = []
    op_name = "chunk_fwd" + ("_ragged" if packed else "")
    op_name += "_with_state_op" if with_state else "_op"

    def forward_op(*args, **kwargs):
        received.append(kwargs.get("fastmath", args[-1]))
        factors = q.new_empty(1, 64, 1, 64)
        result = (q.clone(),)
        if with_state:
            result += (q.new_zeros(1, 1, 128, 128),)
        return (*result, factors, factors.clone())

    monkeypatch.setattr(fused, op_name, forward_op)
    monkeypatch.setattr(fused, "_plain_gate_scan_op", lambda value, *_args: value)
    for fastmath in (False, True, False):
        fused._ChunkKDA.apply(
            q, q, q, q, beta, None, offsets, chunks, 1.0, with_state, fastmath, False, "auto"
        )
    assert received == [False, True, False]

    # The real schemas/fakes must accept the flag, and old positional callers retain False.
    args = (q, q, q, q, beta, None)
    if packed:
        args += (offsets, chunks)
    args += (1.0, False, "auto")
    with FakeTensorMode(allow_non_fake_inputs=True):
        for kwargs in ({}, {"fastmath": False}, {"fastmath": True}):
            outputs = getattr(ops, op_name)(*args, **kwargs)
            assert outputs[0].shape == q.shape
            assert len(outputs) == (4 if with_state else 3)


@requires_blackwell
def test_public_fastmath_preserves_decay_across_strip_boundary():
    """Recover a visible output with unit-norm Q/K and bounded V, not huge operands."""
    pytest.importorskip("cutlass")
    q = torch.zeros(1, 64, 1, 128, device="cuda", dtype=torch.bfloat16)
    q[..., 0] = 1
    k = q.clone()
    v = torch.ones_like(q, requires_grad=True)
    gate = torch.full(q.shape, -5.5, device="cuda")
    beta = torch.zeros(q.shape[:-1], device="cuda")
    beta[:, 15] = 1
    beta.requires_grad_()

    # Only token 15 writes state. Token 16 must return exp(-5.5) in every V channel.
    # The strip-zero rebase splits this into exp(-88) * exp(82.5): the former
    # is FP32-subnormal even though the resulting output is approximately 0.004.
    expected = gate[0, 16, 0, 0].double().exp().expand(128).to(q.dtype)
    nonfast, _ = chunk_kda(q, k, v, gate, beta, scale=1.0, autotune=False, fastmath=False)
    fast, _ = chunk_kda(q, k, v, gate, beta, scale=1.0, autotune=False, fastmath=True)
    default, _ = chunk_kda(q, k, v, gate, beta, scale=1.0, autotune=False)
    torch.testing.assert_close(default, fast, rtol=0, atol=0)
    torch.testing.assert_close(nonfast[0, 16, 0], expected, rtol=0, atol=0)
    assert torch.count_nonzero(fast[0, 16, 0]) == 0

    # The saved Aqk factor also controls gradients to the original write.
    nonfast_dv, nonfast_dbeta = torch.autograd.grad(nonfast[0, 16, 0, 0], (v, beta))
    fast_dv, fast_dbeta = torch.autograd.grad(fast[0, 16, 0, 0], (v, beta))
    torch.testing.assert_close(nonfast_dv[0, 15, 0, 0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(nonfast_dbeta[0, 15, 0], expected[0].to(beta.dtype), rtol=0, atol=0)
    assert fast_dv[0, 15, 0, 0] == 0
    assert fast_dbeta[0, 15, 0] == 0


@requires_blackwell
def test_intra_engine_subnormal_factor_and_cache_isolation():
    """Rescue exp2(-130) before the BF16 cast; do not assume MMA preserves subnormals."""
    pytest.importorskip("cutlass")
    engine = importlib.import_module("attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra_engine")
    q = torch.zeros(1, 64, 1, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.zeros_like(q)
    # Within the first BC16 strip, the right factor is finite exp2(+100),
    # while the left exp2(-130) is FP32-subnormal.
    rows = torch.arange(64, dtype=torch.float32, device="cuda")
    gate_rows = torch.where(rows <= 15, -rows * (100.0 / 15.0), -100.0 - (rows - 15) * 6)
    gate_rows.clamp_(min=-130)
    g = gate_rows[None, :, None, None].expand_as(q).contiguous()
    beta = torch.zeros(1, 64, 1, device="cuda")
    # Q=2^32 rescues the subnormal before conversion; Q=1 also survives the
    # BF16/tensor-core stages. Q=2^-4 instead rounds 2^-134 to BF16 zero even
    # with non-fast exp2: disabling fastmath is not a blanket underflow guarantee.
    for query_exponent in (32, 0, -4):
        q[0, 20, 0, 0] = 2.0**query_exponent
        k[0, 15, 0, 0] = 2.0**-query_exponent
        golden = (
            (
                q[0, 20, 0].double()
                * k[0, 15, 0].double()
                * torch.exp2(g[0, 20, 0].double() - g[0, 15, 0].double())
            )
            .sum()
            .item()
        )
        assert golden == 2.0**-30
        results = []
        for fastmath in (False, True, False, True):
            aqk, offdiag, diagonal = engine.kda_intra_engine_fwd(
                q, k, g, beta, 1.0, None, fastmath=fastmath
            )
            assert all(torch.isfinite(value).all() for value in (aqk, offdiag, diagonal))
            results.append(aqk[0, 20, 0, 15].item())
        # Alternating both ways in one process catches omitted cache-key dimensions.
        expected = golden if query_exponent >= 0 else 0.0
        assert results == [expected, 0.0, expected, 0.0], f"Q=2^{query_exponent}"
    accurate = engine._compile_intra_engine_fwd(1, 128, False, False, False)
    approximate = engine._compile_intra_engine_fwd(1, 128, False, False, True)
    assert accurate is not approximate
    assert accurate is engine._compile_intra_engine_fwd(1, 128, False, False, False)
    assert engine.KdaIntraFwdEngine(1, fastmath=False).get_name() != (
        engine.KdaIntraFwdEngine(1, fastmath=True).get_name()
    )


def test_context_parallel_stages_preserve_fastmath(monkeypatch):
    """Bind the same policy through both staged wrappers without executing GPU kernels."""
    pytest.importorskip("cutlass")
    cp = importlib.import_module("attn_gym.linear.kda.context_parallel")
    stages = importlib.import_module("attn_gym.linear.kda.stages")
    for operation in (
        cp.context_parallel_kda,
        stages.chunk_kda_prepare,
        stages.chunk_kda_prepare_backward,
    ):
        assert inspect.signature(operation).parameters["fastmath"].default is True
    q = torch.ones(1, 64, 1, 128, dtype=torch.bfloat16)
    gate = torch.zeros_like(q, dtype=torch.float32)
    beta = torch.zeros(1, 64, 1)
    factors = q.new_empty(1, 64, 1, 64)
    received = []

    def prepare_forward(*args, **kwargs):
        received.append(("forward", kwargs["fastmath"]))
        return stages.ChunkKDAFactors(q, q, q, factors, factors)

    def prepare_backward(*args, **kwargs):
        received.append(("backward", kwargs["fastmath"]))
        return stages.ChunkKDABwdPrepared(factors, factors, q, q, q, q, q, factors)

    monkeypatch.setattr(stages, "_validate_fused_constraints", lambda *_args: None)
    monkeypatch.setattr(stages, "_plain_gate_scan_op", lambda value, *_args: value)
    monkeypatch.setattr(stages, "_prepare_chunk_kda_fwd", prepare_forward)
    monkeypatch.setattr(stages, "_prepare_chunk_kda_bwd", prepare_backward)

    def record(stage, result):
        def capture(*args, **kwargs):
            received.append((stage, kwargs["fastmath"]))
            return result

        return capture

    monkeypatch.setattr(stages, "build_state_summaries", record("forward-summary", factors))
    monkeypatch.setattr(stages, "_finish_chunk_kda_fwd", record("forward-run", (q, None)))
    monkeypatch.setattr(stages, "build_state_grad_summaries", record("backward-summary", factors))
    monkeypatch.setattr(
        stages, "_finish_chunk_kda_bwd", record("backward-run", (q, q, q, gate, beta, None))
    )
    bounds = torch.tensor([[0, 64]], dtype=torch.int32)
    for fastmath in (False, True):
        bound = cp._kda_stages(None, False, fastmath, None)
        prepared = bound.prepare(q, q, q, gate, beta)
        assert prepared.fastmath is fastmath
        prepared.state_summaries(bounds)
        prepared.run()
        backward = bound.prepare_backward(prepared.saved, q, None, scale=prepared.scale)
        assert backward.fastmath is fastmath
        backward.state_grad_summaries(bounds)
        backward.run()
    assert received == [
        (stage, fastmath)
        for fastmath in (False, True)
        for stage in (
            "forward",
            "forward-summary",
            "forward-run",
            "backward",
            "backward-summary",
            "backward-run",
        )
    ]


def _training_inputs(packed: bool, strong_gate: bool = False, dtype=torch.bfloat16):
    inputs = make_kda_test_inputs(
        128 if packed else 64,
        normalize_qk=True,
        gate_scale=0.5,
        gate_value=-5.0 if strong_gate else None,
        dtype=dtype,
        requires_grad=True,
    )
    state = (torch.randn(2 if packed else 1, 1, 128, 128, device="cuda") / 32).requires_grad_()
    offsets = cumulative_sequence_offsets([65, 63]) if packed else None
    return (*inputs, state), offsets


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("packed", [False, True], ids=["dense-ordinary", "ragged-strong"])
def test_public_fastmath_training_matches_reference(monkeypatch, packed, dtype):
    """Check both policies against FP64, including public compiled forward/backward."""
    pytest.importorskip("cutlass")
    received = []
    modules = {
        "fwd.cute.chunk_kda_fwd_intra": (
            "kda_intra_engine_fwd",
            "chunk_kda_fwd_intra_diagonal",
            "chunk_kda_fwd_k3b_triton",
            "chunk_kda_fwd_inter_solve_cute",
            "chunk_kda_fwd_inter_solve_ragged_cute",
            "recompute_w_u_fwd_triton",
        ),
        "fwd.cute.chunk_kda_fwd": ("chunk_gated_delta_rule_fwd_h", "chunk_gla_fwd_o_gk"),
        "bwd.cute.chunk_kda_bwd": (
            "recompute_w_u_fwd_triton",
            "chunk_gated_delta_rule_fwd_h",
            "chunk_kda_bwd_delta_h_triton",
            "blackwell_delta_h_bwd_dhu_dv_fused_dispatch",
            "chunk_kda_bwd_wy_triton",
            "chunk_kda_bwd_wy_dqkg",
            "chunk_kda_bwd_intra",
        ),
    }
    for module_name, entrypoints in modules.items():
        module = importlib.import_module(f"attn_gym.linear.kda.{module_name}")
        for entrypoint in entrypoints:
            original = getattr(module, entrypoint)

            def capture(*args, _original=original, _name=entrypoint, **kwargs):
                received.append((_name, kwargs["fastmath"]))
                return _original(*args, **kwargs)

            monkeypatch.setattr(module, entrypoint, capture)
    expected_stages = {
        "recompute_w_u_fwd_triton",
        "chunk_gated_delta_rule_fwd_h",
        "chunk_gla_fwd_o_gk",
        "chunk_kda_bwd_intra",
    }
    blackwell = is_sm100_kda_capability(torch.cuda.get_device_capability())
    if blackwell:
        expected_stages.update(
            ("blackwell_delta_h_bwd_dhu_dv_fused_dispatch", "chunk_kda_bwd_wy_dqkg")
        )
        expected_stages.add(
            "kda_intra_engine_fwd"
            if dtype == torch.bfloat16
            else "chunk_kda_fwd_inter_solve_ragged_cute"
            if packed
            else "chunk_kda_fwd_inter_solve_cute"
        )
    else:
        expected_stages.update(
            (
                "chunk_kda_fwd_intra_diagonal",
                "chunk_kda_fwd_k3b_triton",
                "chunk_kda_bwd_delta_h_triton",
                "chunk_kda_bwd_wy_triton",
            )
        )
    torch.manual_seed(17)
    inputs, offsets = _training_inputs(packed, strong_gate=packed, dtype=dtype)
    reference_inputs = clone_kda_inputs(inputs)
    high_inputs = clone_kda_inputs(inputs, dtype=torch.float64)
    reference_output, reference_state = chunk_kda(
        *reference_inputs, cu_seqlens=offsets, output_final_state=True, impl="reference"
    )
    high_output, high_state = kda_reference(*high_inputs, cu_seqlens=offsets)
    assert reference_state is not None and high_state is not None
    d_output = torch.randn_like(reference_output)
    d_state = torch.randn_like(reference_state)
    reference_gradients = torch.autograd.grad(
        (reference_output, reference_state), reference_inputs, (d_output, d_state)
    )
    high_gradients = torch.autograd.grad(
        (high_output, high_state), high_inputs, (d_output.double(), d_state.double())
    )

    def operation(*args, fastmath):
        return chunk_kda(
            *args,
            cu_seqlens=offsets,
            output_final_state=True,
            autotune=False,
            fastmath=fastmath,
        )

    # The same compiled callable must specialize the flag instead of freezing its first value.
    compiled = torch.compile(operation, fullgraph=True)
    for fastmath in (False, True):
        actual_inputs = clone_kda_inputs(inputs)
        received.clear()
        output, state = operation(*actual_inputs, fastmath=fastmath)
        assert state is not None
        gradients = torch.autograd.grad((output, state), actual_inputs, (d_output, d_state))
        assert expected_stages <= {name for name, _ in received}
        assert all(value is fastmath for _, value in received)
        actual_values = (output, state, *gradients)
        reference_values = (reference_output, reference_state, *reference_gradients)
        high_values = (high_output, high_state, *high_gradients)
        names = ("output", "state", "dq", "dk", "dv", "dgate", "dbeta", "dinitial_state")
        # Check finiteness for every result before numerical assertions can stop at an outlier.
        assert all(torch.isfinite(value).all() for value in actual_values)
        for name, actual, high, reference in zip(
            names, actual_values, high_values, reference_values, strict=True
        ):
            label = f"fastmath={fastmath} {name}"
            assert_matches_low_precision_reference(actual, high, reference, label)
            assert_rms_matches_low_precision_reference(actual, high, reference, label)

        compiled_inputs = clone_kda_inputs(inputs)
        received.clear()
        compiled_output, compiled_state = compiled(*compiled_inputs, fastmath=fastmath)
        assert compiled_state is not None
        compiled_gradients = torch.autograd.grad(
            (compiled_output, compiled_state), compiled_inputs, (d_output, d_state)
        )
        assert expected_stages <= {name for name, _ in received}
        assert all(value is fastmath for _, value in received)
        for actual, expected in zip(
            (compiled_output, compiled_state, *compiled_gradients), actual_values, strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@requires_cuda
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "ragged"])
@pytest.mark.parametrize("with_state", [False, True], ids=["output", "output-state"])
def test_forward_fastmath_opcheck(packed, with_state):
    """Keep both flag values in the real/fake/AOT contracts of all forward schemas."""
    pytest.importorskip("cutlass")
    inputs, offsets = _training_inputs(packed)
    q, k, v, gate, beta, state = (value.detach() for value in inputs)
    cumulative_gate = chunk_cumsum_ref(gate, 64, scale=LOG2_E, cu_seqlens=offsets)
    args = (q, k, v, cumulative_gate, beta, state if with_state else None)
    if packed:
        scheduler = importlib.import_module("attn_gym.linear._delta_rule.triton.chunk_scheduler")
        metadata = scheduler.prepare_ragged_chunk_metadata(offsets, q.shape[1], 64)
        args += (offsets, metadata.chunk_offsets)
        op = ops.chunk_fwd_ragged_with_state_op if with_state else ops.chunk_fwd_ragged_op
    else:
        op = ops.chunk_fwd_with_state_op if with_state else ops.chunk_fwd_op
    args += (128**-0.5, False, "auto")
    for fastmath in (False, True):
        torch.library.opcheck(op, args, {"fastmath": fastmath}, rtol=2e-2, atol=2e-3)


@requires_cuda
@pytest.mark.parametrize("packed", [False, True], ids=["dense", "ragged"])
def test_backward_factor_recomputation_preserves_fastmath(monkeypatch, packed):
    """Rebuilding the saved factor tape must use the same exponential specialization."""
    pytest.importorskip("cutlass")
    backward = importlib.import_module("attn_gym.linear.kda.bwd.cute.chunk_kda_bwd")
    prepare_factors = backward.chunk_kda_fwd_factors
    received = []

    def capture_fastmath(*args, **kwargs):
        received.append(kwargs["fastmath"])
        return prepare_factors(*args, **kwargs)

    monkeypatch.setattr(backward, "chunk_kda_fwd_factors", capture_fastmath)
    inputs, offsets = _training_inputs(packed)
    q, k, v, gate, beta, state = (value.detach() for value in inputs)
    cumulative_gate = chunk_cumsum_ref(gate, 64, scale=LOG2_E, cu_seqlens=offsets)
    args = (q, k, v, cumulative_gate, beta)
    forward_args = (*args, state)
    chunks = None
    if packed:
        scheduler = importlib.import_module("attn_gym.linear._delta_rule.triton.chunk_scheduler")
        chunks = scheduler.prepare_ragged_chunk_metadata(offsets, q.shape[1], 64).chunk_offsets
        forward_args += (offsets, chunks)
        forward_op = ops.chunk_fwd_ragged_with_state_op
    else:
        forward_op = ops.chunk_fwd_with_state_op
    for fastmath in (False, True):
        output, final_state, aqk, akk = forward_op(
            *forward_args, 128**-0.5, False, "auto", fastmath=fastmath
        )
        backward_args = (
            offsets,
            chunks,
            torch.randn_like(output),
            torch.randn_like(final_state),
            state,
            128**-0.5,
            fastmath,
            False,
            "auto",
        )
        saved = ops.chunk_bwd_with_state_grad_op(*args, aqk, akk, *backward_args)
        recomputed = ops.chunk_bwd_recompute_factors_with_state_grad_op(*args, *backward_args)
        for actual, expected in zip(recomputed, saved, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert received == [False, True]


@requires_cuda
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_state_summary_fastmath_preserves_subnormal_decay(reverse):
    """A zero-update chunk isolates the summary's FP32 diagonal decay and legacy default."""
    pytest.importorskip("cutlass")
    summaries = importlib.import_module("attn_gym.linear._delta_rule.cute")
    zeros = torch.zeros(1, 64, 1, 128, dtype=torch.bfloat16, device="cuda")
    cumulative_gate = torch.zeros_like(zeros, dtype=torch.float32)
    steps = torch.arange(1, 65, device="cuda", dtype=torch.float32)
    cumulative_gate[0, :, 0, 0] = steps * (-130.0 / 64)
    cumulative_gate[0, :, 0, 1] = steps * (-0.5 / 64)
    bounds = torch.tensor([[0, 64]], dtype=torch.int32, device="cuda")
    if reverse:
        aqk = torch.zeros(1, 64, 1, 64, dtype=zeros.dtype, device=zeros.device)
        operation = summaries.build_state_grad_summaries
        args = (zeros, zeros, zeros, zeros, aqk, cumulative_gate, 1.0, bounds)
    else:
        operation = summaries.build_state_summaries
        args = (zeros, zeros, zeros, cumulative_gate, bounds)
    results = {flag: operation(*args, fastmath=flag) for flag in (False, True)}
    expected = torch.zeros_like(results[False])
    expected[0, 0, 128:].diagonal().copy_(cumulative_gate[0, -1, 0].double().exp2().float())
    torch.testing.assert_close(results[False], expected, rtol=2e-6, atol=0)
    assert results[False][0, 0, 128, 0].item() == 2.0**-130
    expected[0, 0, 128, 0] = 0
    torch.testing.assert_close(results[True], expected, rtol=2e-6, atol=0)
    legacy_fastmath = not (reverse and is_sm100_kda_capability(torch.cuda.get_device_capability()))
    torch.testing.assert_close(operation(*args), results[legacy_fastmath], rtol=0, atol=0)


@requires_blackwell
@torch.no_grad()
def test_cudnn_ordinary_staged_and_cp_share_math_policy():
    pytest.importorskip("cutlass")
    stages = importlib.import_module("attn_gym.linear.kda.stages")
    cp = importlib.import_module("attn_gym.linear.kda.context_parallel")
    inputs = make_kda_test_inputs(64, normalize_qk=True, gate_scale=0.5)
    options = {"backend": "cudnn"}
    default, _ = chunk_kda(*inputs, kernel_options=options, autotune=False)
    allowed, _ = chunk_kda(*inputs, kernel_options=options, fastmath=True, autotune=False)
    torch.testing.assert_close(default, allowed, rtol=0, atol=0)
    with pytest.raises(ValueError, match="cannot honor fastmath=False"):
        chunk_kda(*inputs, kernel_options=options, fastmath=False)
    with pytest.raises(ValueError, match="cannot honor fastmath=False"):
        stages.chunk_kda_prepare(*inputs, kernel_options=options, fastmath=False)
    with pytest.raises(ValueError, match="cannot honor fastmath=False"):
        cp._kda_stages(None, False, False, options).prepare(*inputs)
    assert isinstance(
        cp._kda_stages(None, False, True, options).prepare(*inputs), stages.ChunkKDACudnnPrepared
    )
    for kwargs in ({}, {"fastmath": True}):
        prepared = stages.chunk_kda_prepare(*inputs, kernel_options=options, **kwargs)
        output, state = prepared.run(output_final_state=True)
        assert torch.isfinite(output).all() and torch.isfinite(state).all()
        backward = stages.chunk_kda_prepare_backward(
            prepared.saved, torch.ones_like(output), None, scale=prepared.scale, **kwargs
        )
        assert all(torch.isfinite(x).all() for x in backward.run() if x is not None)
        with pytest.raises(ValueError, match="cannot honor fastmath=False"):
            stages.chunk_kda_prepare_backward(
                prepared.saved, torch.ones_like(output), None, scale=prepared.scale, fastmath=False
            )
