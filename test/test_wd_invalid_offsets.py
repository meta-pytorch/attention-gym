"""Every warpdrive GDN/KDA schedule rejects invalid packed cu_seqlens values on the device.

The first warpdrive kernel of each call checks the offsets (``work_tables.check_packed_offsets``):
the work table (uncut and d_v split), the split planner, the chain prologue, or the GDN T pass's
row table in the backward. The native KDA state summaries check in their Triton work selector.
Each case runs in a fresh subprocess because the device assert poisons
the CUDA context. ``test_wd_paged_offsets.py`` covers the paged entries and
``test_gdn_wd_training.py`` the public GDN state modes.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from attn_gym._backends.arch import cuda_runs_sm100_kernels

pytest.importorskip("cutlass.cute", reason="the warpdrive backend requires CuTeDSL")

pytestmark = pytest.mark.skipif(
    not cuda_runs_sm100_kernels(),
    reason="the warpdrive GDN/KDA kernels require an SM100-family GPU",
)

# Runs the raw registered forward or backward op once with valid offsets, then with ``offsets``.
_OP_CODE = """
import torch
import torch.nn.functional as F
import attn_gym.linear.gdn.impl.warpdrive_ops
import attn_gym.linear.kda.impl.warpdrive_ops

family, direction, split, T, H = {family!r}, {direction!r}, {split!r}, {tokens!r}, {heads!r}
valid, offsets = {valid!r}, {offsets!r}
q = F.normalize(torch.randn(1, T, H, 128, device="cuda"), dim=-1).bfloat16()
k = F.normalize(torch.randn(1, T, H, 128, device="cuda"), dim=-1).bfloat16()
v = torch.randn(1, T, H, 128, device="cuda").bfloat16()
gate = -torch.rand((1, T, H, 128) if family == "kda" else (1, T, H), device="cuda") * 0.1
beta = torch.rand(1, T, H, device="cuda")
cu = torch.tensor(valid, dtype=torch.int32, device="cuda")
ops = torch.ops.attn_gym
if direction == "forward":
    op = ops.kda_chunk_cudnn_packed_fwd if family == "kda" else ops.gdn_chunk_cudnn_packed_fwd
    args = (q, k, v, gate, beta, cu, split, 128**-0.5)
else:
    op = ops.kda_chunk_cudnn_packed_local_bwd if family == "kda" else ops.gdn_chunk_cudnn_packed_bwd
    args = (q, k, v, gate, beta, torch.randn_like(v), cu, split, 128**-0.5)
op(*args)
torch.cuda.synchronize()
print("valid offsets completed", flush=True)
cu.copy_(torch.tensor(offsets, dtype=torch.int32, device="cuda"))
op(*args)
torch.cuda.synchronize()
"""

# The public chunk_kda entry in each state mode; mirrors the GDN test in test_gdn_wd_training.py.
_PUBLIC_KDA_CODE = """
import torch
import torch.nn.functional as F
from attn_gym.linear import chunk_kda

shape = (1, 8, 2, 128)
q = F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
k = F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
v = torch.randn(shape, device="cuda").bfloat16()
gate = -torch.rand(shape, device="cuda")
beta = torch.rand(shape[:3], device="cuda")
cu = torch.tensor({valid!r}, dtype=torch.int32, device="cuda")
state = torch.zeros(({sequences}, 2, 128, 128), device="cuda")
call = torch.compile(chunk_kda, fullgraph=True) if {compiled!r} else chunk_kda
kwargs = dict(
    initial_state=state if {mode!r} != "no_state" else None,
    output_final_state={mode!r} == "with_state",
    cu_seqlens=cu,
    kernel_options={{"backend": "warpdrive"}},
)
call(q, k, v, gate, beta, **kwargs)
torch.cuda.synchronize()
print("valid offsets completed", flush=True)
cu.copy_(torch.tensor({offsets!r}, dtype=torch.int32, device="cuda"))
call(q, k, v, gate, beta, **kwargs)
torch.cuda.synchronize()
"""

# A staged warpdrive KDA handle reused after its device offsets change: the forward summaries,
# the backward handle's reverse summaries, or the direct identity-probe reverse launch.
_SUMMARY_CODE = """
import torch
import torch.nn.functional as F
from attn_gym.linear._delta_rule.warpdrive.launchers.state_summary import (
    build_warpdrive_state_grad_summaries,
)
from attn_gym.linear.kda.stages import chunk_kda_prepare, chunk_kda_prepare_backward

route, valid, offsets = {route!r}, {valid!r}, {offsets!r}
shape = (1, valid[-1], 2, 128)
q = F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
k = F.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()
v = torch.randn(shape, device="cuda").bfloat16()
gate = -torch.rand(shape, device="cuda") * 0.1
beta = torch.rand(shape[:3], device="cuda")
cu = torch.tensor(valid, dtype=torch.int32, device="cuda")
prepared = chunk_kda_prepare(
    q, k, v, gate, beta, cu_seqlens=cu, kernel_options={{"backend": "warpdrive"}}
)
saved = prepared.saved
backward = chunk_kda_prepare_backward(
    saved, torch.randn_like(v), None, scale=prepared.scale,
    kernel_options={{"backend": "warpdrive"}},
)
bounds = torch.stack((saved.cu_seqlens[:-1], saved.cu_seqlens[1:]), dim=1).contiguous()


def run():
    if route == "forward":
        return prepared.state_summaries(bounds)
    if route == "reverse":
        return backward.state_grad_summaries(bounds)
    return build_warpdrive_state_grad_summaries(
        saved.q, saved.k, saved.v, saved.gate, saved.beta, backward.d_output, saved.cu_seqlens,
        prepared.scale, transpose_forward_transition=False, bounds=bounds,
    )


run()
torch.cuda.synchronize()
print("valid offsets completed", flush=True)
saved.cu_seqlens.copy_(torch.tensor(offsets, dtype=torch.int32, device="cuda"))
bounds.copy_(torch.stack((saved.cu_seqlens[:-1], saved.cu_seqlens[1:]), dim=1))
run()
torch.cuda.synchronize()
"""


def _assert_rejected(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
        env={
            **{k: v for k, v in os.environ.items() if "COREDUMP" not in k},
            "CUDA_LAUNCH_BLOCKING": "1",
        },
    )
    assert "valid offsets completed" in result.stdout, result.stdout + result.stderr
    assert result.returncode != 0, "invalid packed offsets reached the warpdrive kernels"
    assert "invalid packed cu_seqlens" in result.stderr, result.stdout + result.stderr


def _chain_shape(family: str, direction: str) -> tuple[int, int]:
    """One sequence of ``(tokens, heads)`` that the driver's plan runs as a piece chain."""
    if family == "kda":
        from attn_gym.linear._delta_rule.warpdrive.drivers import kda as driver
    else:
        from attn_gym.linear._delta_rule.warpdrive.drivers import gdn as driver
    tokens, heads = 4096, 2
    if direction == "forward":
        assert driver.ForwardPlan.build(tokens, 1, heads, 128, "cuda").chain
    else:
        assert driver.BackwardPlan.build(tokens, 1, heads, 128, "cuda").chain
    return tokens, heads


@pytest.mark.parametrize("family", ["gdn", "kda"])
@pytest.mark.parametrize(
    ("direction", "schedule", "invalid"),
    [
        ("forward", "split", "decreasing"),
        ("forward", "chain", "nonzero-start"),
        ("backward", "uncut", "past-end"),
        ("backward", "split", "nonzero-start"),
        ("backward", "chain", "int32-max"),
    ],
)
def test_warpdrive_schedules_reject_invalid_packed_offsets(family, direction, schedule, invalid):
    """The split planner, chain prologue, GDN row table and KDA work table each check first."""
    tokens, heads = _chain_shape(family, direction) if schedule == "chain" else (256, 2)
    sequences = 1 if schedule == "chain" else 2
    valid = [tokens * i // sequences for i in range(sequences + 1)]
    offsets = list(valid)
    match invalid:
        case "nonzero-start":
            offsets[0] = 1
        case "decreasing":
            offsets[1] = offsets[2] + 1
        case "past-end":
            offsets[-1] = tokens + 1
        case "int32-max":
            offsets[-1] = 2**31 - 1
    code = _OP_CODE.format(
        family=family,
        direction=direction,
        split=schedule == "split",
        tokens=tokens,
        heads=heads,
        valid=valid,
        offsets=offsets,
    )
    _assert_rejected(code)


@pytest.mark.parametrize(
    ("mode", "compiled", "offsets"),
    [
        pytest.param("no_state", False, [1, 3], id="no_state-eager-nonzero-start"),
        pytest.param("initial_only", True, [0, 4, 3], id="initial_only-compiled-decreasing"),
        pytest.param("with_state", False, [0, 9], id="with_state-eager-past-end"),
    ],
)
def test_public_kda_warpdrive_rejects_invalid_packed_offset_values(mode, compiled, offsets):
    """The public KDA entry builds no Triton chunk table in its forward, so the warpdrive work
    table's check is the only one."""
    code = _PUBLIC_KDA_CODE.format(
        valid=[0] + [3] * (len(offsets) - 1),
        sequences=len(offsets) - 1,
        compiled=compiled,
        mode=mode,
        offsets=offsets,
    )
    _assert_rejected(code)


@pytest.mark.parametrize(
    ("route", "offsets"),
    [
        pytest.param("forward", [1, 16, 32], id="forward-nonzero-start"),
        pytest.param("forward", [0, 33, 32], id="forward-decreasing"),
        pytest.param("reverse", [0, 16, 33], id="reverse-past-end"),
        pytest.param("reverse_probe", [1, 16, 32], id="reverse_probe-nonzero-start"),
        pytest.param("reverse_probe", [0, 16, 2**31 - 1], id="reverse_probe-int32-max"),
    ],
)
def test_kda_warpdrive_summaries_reject_invalid_packed_offsets(route, offsets):
    """A prepared handle's summaries recheck offsets changed on the device after preparation."""
    _assert_rejected(_SUMMARY_CODE.format(route=route, valid=[0, 16, 32], offsets=offsets))
