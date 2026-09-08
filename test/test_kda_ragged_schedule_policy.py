"""Schedule policy and public ``schedule`` option for the heavy ragged KDA kernels.

The policy tests supply the SM count explicitly so they run without CUDA; only the
end-to-end kernel spy needs a GPU. Rationale: ``GridScheduler.resolve_flat``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

import attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o as output_module
import attn_gym.linear.kda.fwd.triton.recompute_w_u as recompute_module
from attn_gym.linear import chunk_kda
from attn_gym.linear._delta_rule.triton import chunk_scheduler
from attn_gym.linear._delta_rule.triton.chunk_scheduler import (
    GridScheduler,
    RaggedChunkMetadata,
    ScheduleKind,
    ScheduleRequest,
    chunk_capacity,
)
from attn_gym.linear.kda.validation import resolve_kernel_options
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

SM100_SM_COUNT = 152
CHUNK_SIZE = 64
# Output composition flattens (head, 64-wide value tile) pairs; recompute flattens heads.
MANY_SEQUENCE_SHAPES = {
    "uniform_b8_h96_t8192_output": (96 * 2, [8192] * 8),
    "zipf_s8_h96_t11920_recompute": (96, [6148, 497, 46, 3832, 272, 1106, 3, 16]),
}


def resolve_on_sm100(monkeypatch, subtasks, lengths, request, *, auto_persistent):
    """Resolve the flat plan for one packed shape as an SM100 device would."""
    monkeypatch.setattr(chunk_scheduler, "_multiprocessor_count", lambda device: SM100_SM_COUNT)
    metadata = RaggedChunkMetadata(
        None, None, chunk_capacity(sum(lengths), len(lengths), CHUNK_SIZE), CHUNK_SIZE
    )
    return GridScheduler(metadata).resolve_flat(
        request, subtasks, "cuda", auto_persistent=auto_persistent
    )


@pytest.mark.parametrize("shape", sorted(MANY_SEQUENCE_SHAPES))
def test_heavy_ragged_kernels_stay_static_under_auto(monkeypatch, shape):
    subtasks, lengths = MANY_SEQUENCE_SHAPES[shape]
    generic = resolve_on_sm100(
        monkeypatch, subtasks, lengths, ScheduleRequest.AUTO, auto_persistent=True
    )
    # The generic wave rule alone would pick the slower persistent variant here.
    assert generic.kind is ScheduleKind.PERSISTENT
    static = resolve_on_sm100(
        monkeypatch, subtasks, lengths, ScheduleRequest.AUTO, auto_persistent=False
    )
    assert static == replace(generic, kind=ScheduleKind.STATIC)
    explicit = resolve_on_sm100(
        monkeypatch, subtasks, lengths, ScheduleRequest.PERSISTENT, auto_persistent=False
    )
    assert explicit.kind is ScheduleKind.PERSISTENT


@pytest.mark.parametrize(
    ("kernel_options", "expected_schedule"),
    [
        (None, ScheduleRequest.AUTO),
        ({"schedule": "static"}, ScheduleRequest.STATIC),
        ({"backend": "fused", "schedule": "persistent"}, ScheduleRequest.PERSISTENT),
        ({"backend": "mega", "schedule": "auto"}, ScheduleRequest.AUTO),
    ],
)
def test_kernel_options_resolve_schedule(kernel_options, expected_schedule):
    assert resolve_kernel_options(kernel_options).schedule is expected_schedule


@pytest.mark.parametrize(
    ("kernel_options", "match"),
    [
        ({"schedule": "eager"}, "must be 'auto', 'static', or 'persistent'"),
        (
            {"backend": "mega", "schedule": "static"},
            "requires kernel_options\\['backend'\\]='fused'",
        ),
    ],
)
def test_kernel_options_reject_invalid_schedule(kernel_options, match):
    with pytest.raises(ValueError, match=match):
        resolve_kernel_options(kernel_options)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9,
    reason="the persistent ragged kernels require the TMA path",
)
def test_public_chunk_kda_schedule_option_selects_kernels(monkeypatch):
    """``kernel_options['schedule']`` reaches both heavy kernels in forward and backward."""
    pytest.importorskip("cutlass", reason="the fused chunk_kda core is CuTeDSL-backed")
    launches: list[str] = []

    class RecordingKernel:
        def __init__(self, name, kernel):
            self.name, self.kernel = name, kernel

        def __getitem__(self, grid):
            launch = self.kernel[grid]

            def record(*args, **kwargs):
                launches.append(self.name)
                return launch(*args, **kwargs)

            return record

    persistent_kernels = {
        output_module: "chunk_gla_fwd_kernel_o_ragged_tma_persistent",
        recompute_module: "recompute_w_u_fwd_kernel_persistent",
    }
    for module, name in persistent_kernels.items():
        monkeypatch.setattr(module, name, RecordingKernel(name, getattr(module, name)))

    q, k, v, gate, beta = make_kda_test_inputs(
        2048, heads=2, normalize_qk=True, requires_grad=True
    )
    cu_seqlens = cumulative_sequence_offsets([700, 1348])
    results = {}
    for schedule in ("auto", "static", "persistent"):
        launches.clear()
        output = chunk_kda(
            q, k, v, gate, beta, cu_seqlens=cu_seqlens, kernel_options={"schedule": schedule}
        )[0]
        forward_launches = list(launches)
        grads = torch.autograd.grad(output.float().sum(), (q, k, v, gate, beta))
        backward_launches = launches[len(forward_launches) :]
        results[schedule] = (output, *grads)
        if schedule == "persistent":
            assert set(forward_launches) == set(persistent_kernels.values())
            assert "recompute_w_u_fwd_kernel_persistent" in backward_launches
        else:
            assert not launches
    for schedule in ("static", "persistent"):
        for actual, expected in zip(results[schedule], results["auto"], strict=True):
            assert torch.equal(actual, expected)
