# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Default chunk-backend selection for GDN and KDA (docs/linear.md, "Chunk backend selection").

The dispatch tests replace both backends of each public op with probes and simulate the device
capability, so they check the selection rule without launching kernels. The tests at the end
run real kernels on an SM100-family GPU and check that the default reproduces the selected
backend's bits, and that warpdrive repeats its bits run to run.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from attn_gym._backends import arch
from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import chunk_gdn, chunk_kda, paged_chunk_gdn, paged_chunk_kda
from attn_gym.linear._delta_rule import validation
from attn_gym.linear.gdn import api as gdn_api
from attn_gym.linear.kda import api as kda_api
from attn_gym.linear.types import ReplayState
from attn_gym.testing import cumulative_sequence_offsets, strided_state_pool

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="chunk ops require CUDA")

# The SM100 family: SM100, SM103 and SM107 (simulated with a CuTeDSL that targets it).
BLACKWELL = ((10, 0), (10, 3), (10, 7))
ON_SM100 = cuda_runs_sm100_kernels()
OTHER_GPUS = ((8, 0), (9, 0), (12, 0))
PROBED = {
    kda_api: (
        "_warpdrive_chunk_forward",
        "_fused_chunk_forward",
        "_warpdrive_paged_chunk_forward",
        "_fused_paged_chunk_forward",
    ),
    gdn_api: (
        "warpdrive_chunk_forward",
        "fused_chunk_forward",
        "warpdrive_paged_chunk_forward",
        "fused_paged_chunk_forward",
    ),
}


@pytest.fixture
def dispatched(monkeypatch):
    """Record which backend each public call selects instead of running it."""
    calls: list[str] = []
    for module, names in PROBED.items():
        for name in names:
            backend = "warpdrive" if "warpdrive" in name else "fused"

            def probe(*args, backend=backend, **kwargs):
                calls.append(backend)
                return None, None

            monkeypatch.setattr(module, name, probe)
    return calls


def _simulate_capability(
    monkeypatch, capability: tuple[int, int], *, cutedsl_targets_rubin: bool = True
) -> None:
    """Make the backend choice see ``capability``, and on SM107 a CuTeDSL that can or cannot
    target it. Patching ``torch.cuda`` itself would leak into Dynamo's trace rules, which are
    built once per process."""
    monkeypatch.setattr(arch, "cutedsl_targets_rubin", lambda: cutedsl_targets_rubin)
    selected = arch.runs_sm100_kernels(capability)
    monkeypatch.setattr(validation, "runs_sm100_kernels", lambda _capability: selected)


def _inputs(*, vector_gate, dtype=torch.bfloat16, tokens=32, heads=2, dim=128):
    """Normalized Q/K, values, gates in ``[-1, 0]`` and betas in ``[0, 1]``."""
    shape = (1, tokens, heads, dim)
    q, k = (F.normalize(torch.randn(shape, device="cuda"), dim=-1).to(dtype) for _ in range(2))
    v = torch.randn(shape, device="cuda", dtype=dtype)
    gate = -torch.rand(shape if vector_gate else shape[:3], device="cuda")
    beta = torch.rand(shape[:3], device="cuda")
    return q, k, v, gate, beta


def _kda_inputs(dtype=torch.bfloat16, **kwargs):
    return _inputs(vector_gate=True, dtype=dtype, **kwargs)


def _gdn_inputs(dtype=torch.bfloat16, **kwargs):
    return _inputs(vector_gate=False, dtype=dtype, **kwargs)


def _pool(prefix: int = 0):
    """Return a three-slot FP32 pool (with its storage) and the route of one sequence to slot 1.

    The slot stride stays 16-byte aligned; ``prefix=1`` misaligns the base by 4 bytes.
    """
    storage, pool = strided_state_pool(3, 2, 128, 128, prefix=prefix, suffix=4 - prefix)
    return storage, pool, torch.tensor([1], device="cuda", dtype=torch.int32)


def _run_kda(**kwargs):
    q, k, v, gate, beta = _kda_inputs(kwargs.pop("dtype", torch.bfloat16))
    chunk_kda(q, k, v, gate, beta, **kwargs)


def _run_gdn(**kwargs):
    q, k, v, gate, beta = _gdn_inputs(dim=kwargs.pop("dim", 128))
    chunk_gdn(q, k, v, gate, beta, **kwargs)


def _run_paged_kda(prefix=0, replay=False, dtype=torch.bfloat16, kernel_options=None):
    _storage, pool, indices = _pool(prefix)
    replay_state = None
    if replay:
        layout = ReplayState.slot_layout(2, 128)
        replay_state = ReplayState(
            *(torch.zeros((3, *shape), dtype=dt, device="cuda") for shape, dt in layout)
        )
    with torch.no_grad():
        paged_chunk_kda(
            *_kda_inputs(dtype),
            pool,
            indices,
            replay_state=replay_state,
            kernel_options=kernel_options,
        )


def _run_paged_gdn(prefix=0):
    _storage, pool, indices = _pool(prefix)
    with torch.no_grad():
        paged_chunk_gdn(*_gdn_inputs(), pool, indices)


SERVED = {
    "chunk_kda bf16": lambda: _run_kda(),
    "chunk_kda fp16": lambda: _run_kda(dtype=torch.float16),
    "chunk_kda fp32 (runs in bf16)": lambda: _run_kda(dtype=torch.float32),
    "chunk_kda with states": lambda: _run_kda(
        initial_state=torch.zeros(1, 2, 128, 128, device="cuda"), output_final_state=True
    ),
    "chunk_kda schedule=persistent": lambda: _run_kda(kernel_options={"schedule": "persistent"}),
    "chunk_kda fastmath=False": lambda: _run_kda(fastmath=False),
    "chunk_gdn d128": lambda: _run_gdn(),
    "chunk_gdn d64": lambda: _run_gdn(dim=64),
    "paged_chunk_kda aligned pool": lambda: _run_paged_kda(),
    "paged_chunk_kda fp16": lambda: _run_paged_kda(dtype=torch.float16),
    "paged_chunk_gdn aligned pool": lambda: _run_paged_gdn(),
    "paged_chunk_kda replay_state": lambda: _run_paged_kda(replay=True),
    "paged_chunk_kda misaligned pool": lambda: _run_paged_kda(prefix=1),
    "paged_chunk_gdn misaligned pool": lambda: _run_paged_gdn(prefix=1),
}


@pytest.mark.parametrize("capability", BLACKWELL)
@pytest.mark.parametrize("case", SERVED)
def test_default_runs_warpdrive_on_sm100_family(monkeypatch, dispatched, capability, case):
    _simulate_capability(monkeypatch, capability)
    monkeypatch.setattr(validation, "_CUTEDSL_INSTALLED", True)
    SERVED[case]()
    assert dispatched == ["warpdrive"]


@pytest.mark.parametrize("capability", OTHER_GPUS)
@pytest.mark.parametrize("case", SERVED)
def test_default_runs_fused_off_sm100_family(monkeypatch, dispatched, capability, case):
    _simulate_capability(monkeypatch, capability)
    SERVED[case]()
    assert dispatched == ["fused"]


@pytest.mark.parametrize("case", SERVED)
def test_default_runs_fused_without_cutedsl(monkeypatch, dispatched, case):
    _simulate_capability(monkeypatch, (10, 0))
    monkeypatch.setattr(validation, "_CUTEDSL_INSTALLED", False)
    SERVED[case]()
    assert dispatched == ["fused"]


@pytest.mark.parametrize("case", SERVED)
def test_default_runs_fused_on_rubin_without_a_cutedsl_that_targets_it(
    monkeypatch, dispatched, case
):
    """SM107 needs CuTeDSL 4.8+ (``runs_sm100_kernels``); with an older one it stays on fused."""
    _simulate_capability(monkeypatch, (10, 7), cutedsl_targets_rubin=False)
    monkeypatch.setattr(validation, "_CUTEDSL_INSTALLED", True)
    SERVED[case]()
    assert dispatched == ["fused"]


@pytest.mark.parametrize("capability", (*BLACKWELL, (9, 0)))
def test_explicit_backends_are_never_rerouted(monkeypatch, dispatched, capability):
    """Explicit choices hold on every device and route."""
    _simulate_capability(monkeypatch, capability)
    fused, warpdrive = {"backend": "fused"}, {"backend": "warpdrive"}
    _run_kda(kernel_options=fused)
    _run_gdn(kernel_options=fused)
    _run_paged_kda(prefix=1, kernel_options=warpdrive)
    _run_gdn(kernel_options=warpdrive)
    _run_paged_kda(replay=True, kernel_options=fused)
    _run_paged_kda(replay=True, kernel_options=warpdrive)
    assert dispatched == ["fused", "fused", "warpdrive", "warpdrive", "fused", "warpdrive"]


needs_warpdrive = pytest.mark.skipif(
    not ON_SM100 or not validation._CUTEDSL_INSTALLED,
    reason="the warpdrive backend requires an SM100-family GPU and CuTeDSL",
)


def _train(op, inputs, initial_state, **kwargs):
    """Run one forward and backward; return the output, final state, and every gradient."""
    leaves = [tensor.detach().clone().requires_grad_() for tensor in (*inputs, initial_state)]
    output, final_state = op(*leaves[:5], leaves[5], output_final_state=True, **kwargs)
    cotangents = (torch.ones_like(output), torch.ones_like(final_state))
    grads = torch.autograd.grad((output, final_state), leaves, cotangents)
    return output, final_state, *grads


def _assert_bitwise(actual, expected):
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got, want)


def _two_sequences():
    return torch.tensor([0, 20, 32], device="cuda", dtype=torch.int32)


@needs_warpdrive
@pytest.mark.parametrize(
    "name", ["kda bf16", "kda fp16", "kda fastmath=False", "gdn bf16", "gdn fp16"]
)
def test_default_reproduces_the_selected_backend(name):
    """The default call's outputs and gradients are bitwise those of warpdrive, which it
    selects."""
    family, variant = name.split(" ")
    dtype = torch.float16 if variant == "fp16" else torch.bfloat16
    kwargs = {"fastmath": False} if variant == "fastmath=False" else {}
    op, inputs = (
        (chunk_kda, _kda_inputs(dtype)) if family == "kda" else (chunk_gdn, _gdn_inputs(dtype))
    )
    initial_state = torch.randn(2, 2, 128, 128, device="cuda") * 0.01
    default, selected = (
        _train(
            op,
            inputs,
            initial_state,
            cu_seqlens=_two_sequences(),
            kernel_options=options,
            **kwargs,
        )
        for options in ({"autotune": False}, {"backend": "warpdrive", "autotune": False})
    )
    _assert_bitwise(default, selected)


@needs_warpdrive
@pytest.mark.parametrize("name", ["kda bf16", "kda fp16", "kda fastmath=False", "gdn bf16"])
def test_default_stages_select_like_the_public_op(name):
    """The staged entry points behind context parallelism resolve the default as the ops do."""
    from attn_gym.linear.gdn.stages import ChunkGDNWarpdrivePrepared, chunk_gdn_prepare
    from attn_gym.linear.kda.stages import ChunkKDAWarpdrivePrepared, chunk_kda_prepare

    family, variant = name.split(" ")
    dtype = torch.float16 if variant == "fp16" else torch.bfloat16
    if family == "kda":
        prepared = chunk_kda_prepare(
            *_kda_inputs(dtype),
            cu_seqlens=_two_sequences(),
            fastmath=variant != "fastmath=False",
        )
        assert isinstance(prepared, ChunkKDAWarpdrivePrepared)
    else:
        prepared = chunk_gdn_prepare(*_gdn_inputs(dtype), cu_seqlens=_two_sequences())
        assert isinstance(prepared, ChunkGDNWarpdrivePrepared)


def _relative_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.double() - expected.double()).square().mean().sqrt()
    return (error / expected.double().square().mean().sqrt()).item()


@needs_warpdrive
@pytest.mark.parametrize("family", ["kda", "gdn"])
@pytest.mark.parametrize("route", ["aligned", "misaligned", "replay"])
def test_default_paged_routes_run_warpdrive(family, route):
    """Every paged route, misaligned pools and KDA replay included, reproduces explicit warpdrive
    bit for bit and the fused backend within low-precision rounding."""
    if family == "gdn" and route == "replay":
        pytest.skip("replay_state is KDA-only")
    inputs = _kda_inputs() if family == "kda" else _gdn_inputs()
    op = paged_chunk_kda if family == "kda" else paged_chunk_gdn
    results = {}
    for backend in ("auto", "warpdrive", "fused"):
        torch.manual_seed(1)
        _storage, pool, indices = _pool(prefix=int(route == "misaligned"))
        pool.mul_(0.01)
        extra = {}
        if route == "replay":
            layout = ReplayState.slot_layout(2, 128)
            extra["replay_state"] = ReplayState(
                *(torch.zeros((3, *shape), dtype=dt, device="cuda") for shape, dt in layout)
            )
        with torch.no_grad():
            output = op(
                *inputs,
                pool,
                indices,
                kernel_options={"backend": backend, "autotune": False},
                **extra,
            )
        results[backend] = (output, pool.clone(), *extra.get("replay_state", ()))
    _assert_bitwise(results["auto"], results["warpdrive"])
    output, pool, *replay = results["warpdrive"]
    fused_output, fused_pool, *fused_replay = results["fused"]
    assert _relative_rms(output, fused_output) < 1e-2
    assert _relative_rms(pool, fused_pool) < 1e-2
    _assert_bitwise(replay, fused_replay)  # the token caches hold inputs, not results


@needs_warpdrive
@pytest.mark.parametrize("chained", [False, True], ids=["default_plan", "chained_plan"])
@pytest.mark.parametrize("family", ["kda", "gdn"])
def test_default_warpdrive_repeats_its_bits(monkeypatch, family, chained):
    """Repeated default calls reproduce outputs, final states, and gradients bitwise, including
    through the piece chains whose work claims use atomics (docs/determinism.md)."""
    from attn_gym.linear._delta_rule.warpdrive.drivers import gdn, kda

    if chained:
        for driver in (gdn, kda):
            for costs in (driver.CHAIN_COST_FWD, driver.CHAIN_COST_BWD):
                for dim, cost in costs.items():
                    monkeypatch.setitem(costs, dim, cost._replace(overhead_tokens=0))
    lengths = (3000, 0, 2000)
    cu_seqlens = cumulative_sequence_offsets(lengths)
    torch.manual_seed(2)
    op = chunk_kda if family == "kda" else chunk_gdn
    inputs = _inputs(vector_gate=family == "kda", tokens=sum(lengths), heads=1)
    initial_state = torch.randn(3, 1, 128, 128, device="cuda") * 0.01
    first, second = (_train(op, inputs, initial_state, cu_seqlens=cu_seqlens) for _ in range(2))
    _assert_bitwise(first, second)


@needs_warpdrive
def test_compiled_default_runs_warpdrive():
    """``torch.compile`` resolves the default at trace time to the same warpdrive call."""
    inputs = _kda_inputs()
    cu_seqlens = _two_sequences()
    compiled = torch.compile(chunk_kda, fullgraph=True)
    actual, _ = compiled(*inputs, cu_seqlens=cu_seqlens)
    expected, _ = chunk_kda(
        *inputs, cu_seqlens=cu_seqlens, kernel_options={"backend": "warpdrive"}
    )
    assert torch.equal(actual, expected)
