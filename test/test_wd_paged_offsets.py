"""Paged warpdrive GDN/KDA reject invalid packed cu_seqlens values like the other entries."""

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

# Each case is a fresh subprocess: a device assert poisons the CUDA context.
_CODE = """
import torch
import torch.nn.functional as F
from attn_gym.linear import paged_chunk_gdn, paged_chunk_kda

family, offsets = {family!r}, {offsets!r}
T, H = 200, 2
q = F.normalize(torch.randn(1, T, H, 128, device="cuda"), dim=-1).bfloat16()
k = F.normalize(torch.randn(1, T, H, 128, device="cuda"), dim=-1).bfloat16()
v = torch.randn(1, T, H, 128, device="cuda").bfloat16()
gate = -torch.rand((1, T, H, 128) if family == "kda" else (1, T, H), device="cuda")
beta = torch.rand(1, T, H, device="cuda")
pool = torch.zeros(4, H, 128, 128, device="cuda")
slots = torch.tensor([1, 2], dtype=torch.int32, device="cuda")
cu = torch.tensor([0, 100, 200], dtype=torch.int32, device="cuda")
call = paged_chunk_kda if family == "kda" else paged_chunk_gdn
options = {{"backend": "warpdrive"}}
with torch.no_grad():
    call(q, k, v, gate, beta, pool, slots, cu_seqlens=cu, kernel_options=options)
    torch.cuda.synchronize()
    print("valid offsets completed", flush=True)
    cu.copy_(torch.tensor(offsets, dtype=torch.int32, device="cuda"))
    call(q, k, v, gate, beta, pool, slots, cu_seqlens=cu, kernel_options=options)
    torch.cuda.synchronize()
"""


@pytest.mark.parametrize("family", ["gdn", "kda"])
@pytest.mark.parametrize(
    "offsets",
    [[1, 50, 100], [0, 100, 50], [0, 100, 260], [0, 100, 2**31 - 1]],
    ids=["nonzero-start", "decreasing", "past-end", "int32-max"],
)
def test_paged_warpdrive_rejects_invalid_packed_offsets(family, offsets):
    """Without the check these read and write past the tensors or fault (illegal address)."""
    result = subprocess.run(
        [sys.executable, "-c", _CODE.format(family=family, offsets=offsets)],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
        env={
            **{k: v for k, v in os.environ.items() if "COREDUMP" not in k},
            "CUDA_LAUNCH_BLOCKING": "1",
        },
    )
    assert "valid offsets completed" in result.stdout, result.stdout + result.stderr
    assert result.returncode != 0, "invalid packed offsets reached the warpdrive kernels"
    assert "invalid packed cu_seqlens" in result.stderr, result.stdout + result.stderr
