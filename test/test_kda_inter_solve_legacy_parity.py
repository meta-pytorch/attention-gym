"""Bitwise parity of the warpdrive-scheduled KDA K3b/K4b kernels with the previous kernels.

The previous kernels live in ``chunk_kda_inter_solve_legacy`` only for this test and the A/B
timing; both are removed together.
"""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear._delta_rule.chunk_schedule import ScheduleKind, ScheduleRequest
from attn_gym.linear._delta_rule.triton import chunk_scheduler
from attn_gym.linear._delta_rule.triton.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.fwd.cute import chunk_kda_fwd_inter_solve as inter_solve
from attn_gym.linear.kda.fwd.cute import chunk_kda_inter_solve_legacy as legacy
from attn_gym.testing import cumulative_sequence_offsets

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTe K3b/K4b kernels require an SM100 or SM103 GPU",
)

HEADS = 4


def _legacy_compiler(compile_legacy):
    """Stand in for ``_compile_k3b``: the same compile arguments, and a callable that takes the
    port's launch arguments, whose last one (the chunk-worker cap) the previous kernel took at
    compile time."""

    def compile_fn(*args, **kwargs):
        schedule_kind = args[7]

        def launch(*call_args):
            *call_args, workers = call_args
            persistent = schedule_kind is ScheduleKind.PERSISTENT
            fn = compile_legacy(*args, workers if persistent else 0, **kwargs)
            return fn(*call_args)

        return launch

    return compile_fn


@pytest.fixture
def run_both(monkeypatch):
    """``run(fn)`` returns ``fn()`` with the port and with the previous kernels (K4b keeps the
    previous compile signature)."""

    def run(fn):
        new = fn()
        with monkeypatch.context() as m:
            m.setattr(inter_solve, "_compile_k3b", _legacy_compiler(legacy._compile_legacy_k3b))
            m.setattr(inter_solve, "_compile_k4b", legacy._compile_legacy_k4b)
            old = fn()
        return old, new

    return run


def _inputs(tokens, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    shape = (1, tokens, HEADS, 128)
    q = torch.randn(shape, device="cuda", dtype=dtype, generator=g) / 8
    k = torch.randn(shape, device="cuda", dtype=dtype, generator=g) / 8
    gk = -torch.rand(shape, device="cuda", generator=g).cumsum(1) / 64
    beta = torch.rand(1, tokens, HEADS, device="cuda", generator=g)
    akkd = torch.randn(1, tokens, HEADS, 16, device="cuda", generator=g) / 16
    return q, k, gk, beta, akkd


def _bits(t):
    return t.view(torch.int32) if t.element_size() == 4 else t.view(torch.int16)


@pytest.mark.parametrize(
    "lengths, capacity_extra",
    [
        ([65, 63], 0),
        ([0, 1, 64, 0, 65], 0),
        ([65, 63, 130, 70, 7, 300], 0),
        ([64] * 9, 5),  # graph overcapture: capacity past the active chunks
    ],
)
@pytest.mark.parametrize("request_kind", ["STATIC", "PERSISTENT"])
@pytest.mark.parametrize("workers", [2, 3, 148])
def test_ragged_inter_solve_matches_previous_kernels(
    run_both, monkeypatch, lengths, capacity_extra, request_kind, workers
):
    """Every output is bitwise equal on the rows both define: all of them for the exact grid,
    the active ``AkkOD`` rows for the persistent one (its capacity slack is never written)."""
    if request_kind == "STATIC" and workers != 148:
        pytest.skip("the exact grid has no worker count")
    monkeypatch.setattr(
        chunk_scheduler.GridScheduler, "num_chunk_workers", lambda self, device: workers
    )
    tokens = sum(lengths)
    q, k, gk, beta, akkd = _inputs(tokens)
    offsets = cumulative_sequence_offsets(lengths)
    metadata = prepare_ragged_chunk_metadata(offsets, tokens + 64 * capacity_extra, 64)
    request = ScheduleRequest[request_kind]
    aqk0 = torch.zeros(1, tokens, HEADS, 64, device="cuda", dtype=q.dtype)

    def forward():
        aqk, akk_od = inter_solve.chunk_kda_fwd_k3b_ragged_cute(
            q, k, gk, beta, aqk0, 128**-0.5, metadata, schedule=request
        )
        akk = inter_solve.chunk_kda_fwd_k4b_ragged_cute(akk_od, akkd, metadata, schedule=request)
        return aqk, akk_od, akk

    (old_aqk, old_od, old_akk), (new_aqk, new_od, new_akk) = run_both(forward)
    rows = (
        old_od.shape[0]
        if request is ScheduleRequest.STATIC
        else metadata.chunk_offsets[-1].item() * 6
    )
    assert torch.equal(_bits(old_aqk), _bits(new_aqk))
    assert torch.equal(_bits(old_od[:rows]), _bits(new_od[:rows]))
    assert torch.equal(_bits(old_akk), _bits(new_akk))


@pytest.mark.parametrize("tokens", [64, 64 * 37])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dense_inter_solve_matches_previous_kernels(run_both, tokens, dtype):
    q, k, gk, beta, akkd = _inputs(tokens, dtype, seed=3)

    def forward():
        # The diagonal Aqk blocks come from the intra stage: start from zeros.
        aqk = torch.zeros(1, tokens, HEADS, 64, device="cuda", dtype=dtype)
        return inter_solve.chunk_kda_fwd_inter_solve_cute(q, k, gk, beta, akkd, 128**-0.5, Aqk=aqk)

    (old_aqk, old_akk), (new_aqk, new_akk) = run_both(forward)
    assert torch.equal(_bits(old_aqk), _bits(new_aqk))
    assert torch.equal(_bits(old_akk), _bits(new_akk))
