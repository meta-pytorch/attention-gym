# Copyright (c) 2026 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Backend selection for public KDA and GDN calls. Dispatch probes exercise the device and
call-specific rules without launching kernels; GPU tests compare real outputs and gradients
with the explicitly selected backend.
"""

from __future__ import annotations

import functools

import pytest
import torch

from attn_gym._backends import arch
from attn_gym._backends.arch import cuda_runs_sm100_kernels
from attn_gym.linear import chunk_gdn, chunk_kda, paged_chunk_gdn, paged_chunk_kda
from attn_gym.linear._delta_rule import validation
from attn_gym.linear.gdn import api as gdn_api
from attn_gym.linear.kda import api as kda_api
from attn_gym.linear.types import ReplayState
from attn_gym.testing import (
    cumulative_sequence_offsets,
    make_gdn_test_inputs,
    sentinel_state_checkpoints,
    strided_state_pool,
)
from attn_gym.testing.kda import make_kda_test_inputs

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
    """Simulate GPU and compiler support at the selector. Patching torch.cuda directly would
    also change the capability seen by Dynamo's process-wide trace rules."""
    monkeypatch.setattr(arch, "cutedsl_targets_rubin", lambda: cutedsl_targets_rubin)
    selected = arch.runs_sm100_kernels(capability)
    monkeypatch.setattr(validation, "cuda_runs_sm100_kernels", lambda _device: selected)


def _kda_inputs(dtype=torch.bfloat16, *, tokens=32, heads=2):
    """Normalized Q/K, gates in ``[-1, 0]`` and betas in ``[0, 1]``."""
    return make_kda_test_inputs(
        tokens, heads=heads, dtype=dtype, normalize_qk=True, gate_scale=1.0, log_uniform_gate=False
    )


def _gdn_inputs(dtype=torch.bfloat16, *, tokens=32, heads=2, dim=128):
    """Normalized Q/K, gates in ``[-5, 0]`` and betas in ``[0, 1]``."""
    q, k, v, gate, beta, _state, _offsets = make_gdn_test_inputs(
        tokens, key_heads=heads, value_heads=heads, key_dim=dim, dtype=dtype, value_scale=1.0
    )
    return q, k, v, gate, beta


def _inputs(family, **kwargs):
    return _kda_inputs(**kwargs) if family == "kda" else _gdn_inputs(**kwargs)


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


def _checkpoints(requested: bool):
    """Request a checkpoint at position zero to exercise dispatch without writing a slot."""
    return sentinel_state_checkpoints([0], (2, 128, 128)) if requested else None


def _run_paged_kda(
    prefix=0, replay=False, dtype=torch.bfloat16, kernel_options=None, checkpoints=False
):
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
            state_checkpoints=_checkpoints(checkpoints),
        )


def _run_paged_gdn(prefix=0, kernel_options=None, checkpoints=False):
    _storage, pool, indices = _pool(prefix)
    with torch.no_grad():
        paged_chunk_gdn(
            *_gdn_inputs(),
            pool,
            indices,
            kernel_options=kernel_options,
            state_checkpoints=_checkpoints(checkpoints),
        )


SERVED = {
    "chunk_kda bf16": lambda: _run_kda(),
    "chunk_kda fp32 (runs in bf16)": lambda: _run_kda(dtype=torch.float32),
    "chunk_kda with states": lambda: _run_kda(
        initial_state=torch.zeros(1, 2, 128, 128, device="cuda"), output_final_state=True
    ),
    "chunk_kda schedule=persistent": lambda: _run_kda(kernel_options={"schedule": "persistent"}),
    "chunk_kda fastmath=False": lambda: _run_kda(fastmath=False),
    "chunk_gdn d128": lambda: _run_gdn(),
    "chunk_gdn d64": lambda: _run_gdn(dim=64),
    "paged_chunk_kda aligned pool": lambda: _run_paged_kda(),
    "paged_chunk_gdn aligned pool": lambda: _run_paged_gdn(),
    "paged_chunk_kda replay_state": lambda: _run_paged_kda(replay=True),
    "paged_chunk_kda misaligned pool": lambda: _run_paged_kda(prefix=1),
    "paged_chunk_gdn misaligned pool": lambda: _run_paged_gdn(prefix=1),
}
ROUTED_TO_FUSED = {
    "chunk_kda fp16": lambda: _run_kda(dtype=torch.float16),
    "paged_chunk_kda fp16": lambda: _run_paged_kda(dtype=torch.float16),
    "paged_chunk_kda state_checkpoints": lambda: _run_paged_kda(checkpoints=True),
    "paged_chunk_kda replay_state state_checkpoints": lambda: _run_paged_kda(
        replay=True, checkpoints=True
    ),
    "paged_chunk_gdn state_checkpoints": lambda: _run_paged_gdn(checkpoints=True),
}


@pytest.mark.parametrize("capability", BLACKWELL)
@pytest.mark.parametrize("case", SERVED)
def test_default_runs_warpdrive_on_sm100_family(monkeypatch, dispatched, capability, case):
    _simulate_capability(monkeypatch, capability)
    monkeypatch.setattr(validation, "_CUTEDSL_INSTALLED", True)
    SERVED[case]()
    assert dispatched == ["warpdrive"]


@pytest.mark.parametrize("capability", BLACKWELL)
@pytest.mark.parametrize("case", ROUTED_TO_FUSED)
def test_default_routes_unserved_calls_to_fused(monkeypatch, dispatched, capability, case):
    _simulate_capability(monkeypatch, capability)
    ROUTED_TO_FUSED[case]()
    assert dispatched == ["fused"]


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
    """Explicit choices hold on every device and for calls that ``"auto"`` would route;
    explicit warpdrive with checkpoints raises rather than running fused."""
    _simulate_capability(monkeypatch, capability)
    fused, warpdrive = {"backend": "fused"}, {"backend": "warpdrive"}
    _run_kda(kernel_options=fused)
    _run_gdn(kernel_options=fused)
    _run_kda(dtype=torch.float16, kernel_options=warpdrive)
    _run_gdn(kernel_options=warpdrive)
    _run_paged_kda(replay=True, kernel_options=fused)
    _run_paged_kda(replay=True, kernel_options=warpdrive)
    for run in (_run_paged_kda, _run_paged_gdn):
        with pytest.raises(ValueError, match="state_checkpoints requires the fused backend"):
            run(kernel_options=warpdrive, checkpoints=True)
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
    ("name", "expected"),
    [
        ("kda bf16", "warpdrive"),
        ("kda fp16", "fused"),
        ("kda fastmath=False", "warpdrive"),
        ("gdn bf16", "warpdrive"),
        ("gdn fp16", "warpdrive"),
    ],
)
def test_default_reproduces_the_selected_backend(name, expected):
    """The default call's outputs and gradients are bitwise those of the backend it selects."""
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
        for options in ({"autotune": False}, {"backend": expected, "autotune": False})
    )
    _assert_bitwise(default, selected)


@needs_warpdrive
@pytest.mark.parametrize(
    ("name", "warpdrive"),
    [("kda bf16", True), ("kda fp16", False), ("kda fastmath=False", True), ("gdn bf16", True)],
)
def test_default_stages_select_like_the_public_op(name, warpdrive):
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
        assert isinstance(prepared, ChunkKDAWarpdrivePrepared) is warpdrive
    else:
        prepared = chunk_gdn_prepare(*_gdn_inputs(dtype), cu_seqlens=_two_sequences())
        assert isinstance(prepared, ChunkGDNWarpdrivePrepared) is warpdrive


def _relative_rms(actual: torch.Tensor, expected: torch.Tensor) -> float:
    error = (actual.double() - expected.double()).square().mean().sqrt()
    return (error / expected.double().square().mean().sqrt()).item()


def _run_paged(family, backend, *, compiled=False, route="aligned", checkpoint_at=None):
    """Run one paged call from a seeded pool; return the output and every buffer it mutates.

    ``backend="auto"`` omits the backend. ``compiled`` runs the call under
    ``torch.compile(fullgraph=True)``.
    """
    inputs = _inputs(family, tokens=32 if checkpoint_at is None else 128)
    torch.manual_seed(1)
    _storage, pool, indices = _pool(prefix=int(route == "misaligned"))
    pool.mul_(0.01)
    extra = {}
    if route == "replay":
        layout = ReplayState.slot_layout(2, 128)
        extra["replay_state"] = ReplayState(
            *(torch.zeros((3, *shape), dtype=dt, device="cuda") for shape, dt in layout)
        )
    if checkpoint_at is not None:
        extra["state_checkpoints"] = sentinel_state_checkpoints([checkpoint_at], (2, 128, 128))
    options = {"autotune": False} if backend == "auto" else {"backend": backend, "autotune": False}
    op = functools.partial(
        paged_chunk_kda if family == "kda" else paged_chunk_gdn, kernel_options=options
    )
    if compiled:
        op = torch.compile(op, fullgraph=True)
    with torch.no_grad():
        output = op(*inputs, pool, indices, **extra)
    mutated = [pool.clone(), *extra.get("replay_state", ())]
    if checkpoint_at is not None:
        mutated.append(extra["state_checkpoints"].state_pool)
    return output, *mutated


@needs_warpdrive
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("family", ["kda", "gdn"])
@pytest.mark.parametrize("route", ["aligned", "misaligned", "replay"])
def test_default_paged_routes_run_warpdrive(family, route, compiled):
    """Default paged outputs and state, eager or compiled, match warpdrive bit for bit, and
    Triton within tolerance."""
    if family == "gdn" and route == "replay":
        pytest.skip("replay_state is KDA-only")
    default = _run_paged(family, "auto", compiled=compiled, route=route)
    _assert_bitwise(default, _run_paged(family, "warpdrive", route=route))
    output, pool, *replay = default
    fused_output, fused_pool, *fused_replay = _run_paged(family, "fused", route=route)
    assert _relative_rms(output, fused_output) < 1e-2
    assert _relative_rms(pool, fused_pool) < 1e-2
    _assert_bitwise(replay, fused_replay)  # the token caches hold inputs, not results


@needs_warpdrive
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
@pytest.mark.parametrize("family", ["kda", "gdn"])
def test_default_checkpointed_paged_call_runs_fused(family, compiled):
    """A paged call that writes ``state_checkpoints``, which warpdrive kernels cannot produce,
    reproduces explicit fused bit for bit, eager or compiled, state cache and checkpoints
    included."""
    default = _run_paged(family, "auto", compiled=compiled, checkpoint_at=64)
    _assert_bitwise(default, _run_paged(family, "fused", checkpoint_at=64))


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
    inputs = _inputs(family, tokens=sum(lengths), heads=1)
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
