# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SM100 intra-chunk engine vs the shipped kernels, including gate saturation.

Mirrors the agent_space verification harnesses in committed form: the fwd
engine must reproduce the shipped forloop + K3b + K4b trio outputs, the bwd
engine must reproduce the shipped chunk_kda_bwd_intra, and both must stay
finite and shipped-matching at the bounded-gate hard rate (-5 nats/token =
-7.21 log2 units/token), where a naive full-chunk gate rebase overflows.
"""

import math
from itertools import accumulate

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="KDA intra engine requires SM100",
)

D, BT = 128, 64
SATURATED_LOG2_PER_TOKEN = 5.0 * math.log2(math.e)


def _inputs(lengths, heads, seed=2):
    from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata

    total = sum(lengths)
    cu = torch.tensor([0, *accumulate(lengths)], dtype=torch.int32, device="cuda")
    metadata = prepare_ragged_chunk_metadata(cu, total, BT)
    torch.manual_seed(seed)
    q = torch.randn(1, total, heads, D, device="cuda", dtype=torch.bfloat16) / 8
    k = torch.randn_like(q) / 8
    g = -torch.rand(1, total, heads, D, device="cuda") / 4
    beta = torch.sigmoid(torch.randn(1, total, heads, device="cuda"))
    return metadata, q, k, g, beta


def _saturate(g, lengths, rate=SATURATED_LOG2_PER_TOKEN):
    """Chunk-local cumulative gate at the hard bound (g restarts per chunk,
    chunks are counted within each sequence)."""
    idx = torch.cat([torch.arange(n, device=g.device) % BT for n in lengths if n > 0])
    return torch.full_like(g, -rate) * (idx + 1).view(1, -1, 1, 1)


def _max_scaled(a, b):
    diff = (a.float() - b.float()).abs()
    return (diff / b.float().abs().clamp_min(1.0)).max().item()


@pytest.mark.parametrize("saturated", [False, True], ids=["mild", "gate-bound"])
@pytest.mark.parametrize(
    "lengths,heads", [([64, 128], 2), ([65, 0, 63, 200], 2)], ids=["aligned", "ragged"]
)
def test_fwd_engine_matches_shipped_trio(lengths, heads, saturated):
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
        _chunk_kda_fwd_k3b_ragged_impl,
        _chunk_kda_fwd_k4b_ragged_impl,
    )
    from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_intra_sub_chunk_forloop import (
        chunk_kda_fwd_intra_diagonal,
    )
    from attn_gym.linear.kda.intra.engine import kda_intra_engine_fwd

    metadata, q, k, g, beta = _inputs(lengths, heads)
    if saturated:
        g = _saturate(g, lengths)
    scale = D**-0.5

    eaqk, eakkod, eakkd = kda_intra_engine_fwd(q, k, g, beta, scale, metadata)
    eakk = _chunk_kda_fwd_k4b_ragged_impl(eakkod, eakkd, metadata)

    saqk, sakkd = chunk_kda_fwd_intra_diagonal(q, k, g, beta, scale, metadata)
    saqk, sakkod = _chunk_kda_fwd_k3b_ragged_impl(q, k, g, beta, saqk, scale, metadata)
    sakk = _chunk_kda_fwd_k4b_ragged_impl(sakkod, sakkd, metadata)

    for name, got, ref, tol in (
        ("Aqk", eaqk, saqk, 2e-3),
        ("AkkOD", eakkod, sakkod, 5e-3),
        ("Akkd", eakkd, sakkd, 5e-3),
        ("Akk", eakk, sakk, 1e-2),
    ):
        assert torch.isfinite(got.float()).all(), f"{name} produced nonfinite values"
        assert _max_scaled(got, ref) < tol, f"{name} diverged from the shipped trio"


@pytest.mark.parametrize("saturated", [False, True], ids=["mild", "gate-bound"])
@pytest.mark.parametrize(
    "lengths,heads", [([64, 128], 2), ([300, 1, 64, 129], 4)], ids=["aligned", "ragged"]
)
def test_bwd_engine_matches_shipped(lengths, heads, saturated):
    from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd_intra import chunk_kda_bwd_intra
    from attn_gym.linear.kda.intra.engine import kda_intra_engine_bwd

    metadata, q, k, g, beta = _inputs(lengths, heads)
    if saturated:
        g = _saturate(g, lengths)
    total = q.shape[1]
    torch.manual_seed(3)
    dAqk = torch.randn(1, total, heads, BT, device="cuda") / 16
    dAkk = torch.randn_like(dAqk) / 16
    dq = torch.randn(1, total, heads, D, device="cuda") / 16
    dk = torch.randn_like(dq) / 16
    db = torch.randn(1, total, heads, device="cuda") / 16
    dg = torch.randn(1, total, heads, D, device="cuda") / 16
    args = (q, k, g, beta, dAqk, dAkk, dq, dk, db, dg)

    engine_out = kda_intra_engine_bwd(*args, metadata)
    shipped_out = chunk_kda_bwd_intra(*args, metadata=metadata)
    for name, got, ref in zip(("dq2", "dk2", "dg2", "db2"), engine_out, shipped_out, strict=False):
        assert torch.isfinite(got.float()).all(), f"{name} produced nonfinite values"
        assert _max_scaled(got, ref) < 5e-3, f"{name} diverged from the shipped kernel"
