# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""SM100 forward intra-chunk engine vs the shipped kernels at mild and saturated gates.

The engine must stay finite and match the shipped forloop + K3b + K4b trio at the bounded-gate hard
rate (-5 nats/token = -7.21 log2 units/token), where a naive full-chunk gate rebase overflows.
"""

import math

import pytest
import torch

from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="KDA intra engine requires SM100 or SM103",
)

D, BT = 128, 64
SATURATED_LOG2_PER_TOKEN = 5.0 * math.log2(math.e)


def _inputs(lengths, heads, seed=2):
    total = sum(lengths)
    metadata = prepare_ragged_chunk_metadata(cumulative_sequence_offsets(lengths), total, BT)
    q, k, _value, gate, beta = make_kda_test_inputs(
        total,
        heads=heads,
        seed=seed,
        gate_scale=0.25,
    )
    return metadata, q, k, gate, beta


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
        chunk_kda_fwd_k3b_ragged_cute,
        chunk_kda_fwd_k4b_ragged_cute,
    )
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra_engine import kda_intra_engine_fwd
    from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_intra_sub_chunk_forloop import (
        chunk_kda_fwd_intra_diagonal,
    )

    metadata, q, k, g, beta = _inputs(lengths, heads)
    if saturated:
        g = _saturate(g, lengths)
    scale = D**-0.5

    eaqk, eakkod, eakkd = kda_intra_engine_fwd(q, k, g, beta, scale, metadata)
    eakk = chunk_kda_fwd_k4b_ragged_cute(eakkod, eakkd, metadata)

    saqk, sakkd = chunk_kda_fwd_intra_diagonal(q, k, g, beta, scale, metadata)
    saqk, sakkod = chunk_kda_fwd_k3b_ragged_cute(q, k, g, beta, saqk, scale, metadata)
    sakk = chunk_kda_fwd_k4b_ragged_cute(sakkod, sakkd, metadata)

    for name, got, ref, tol in (
        ("Aqk", eaqk, saqk, 2e-3),
        ("AkkOD", eakkod, sakkod, 5e-3),
        ("Akkd", eakkd, sakkd, 5e-3),
        ("Akk", eakk, sakk, 1e-2),
    ):
        assert torch.isfinite(got.float()).all(), f"{name} produced nonfinite values"
        assert _max_scaled(got, ref) < tol, f"{name} diverged from the shipped trio"


@pytest.mark.parametrize("saturated", [False, True], ids=["mild", "gate-bound"])
def test_fwd_engine_dense_matches_shipped(saturated):
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
        chunk_kda_fwd_inter_solve_cute,
        chunk_kda_fwd_k4b_dense_cute,
    )
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra_engine import kda_intra_engine_fwd
    from attn_gym.linear.kda.fwd.triton.chunk_kda_fwd_intra_sub_chunk_forloop import (
        chunk_kda_fwd_intra_diagonal,
    )

    _, q, k, g, beta = _inputs([128], 2)
    if saturated:
        g = _saturate(g, [128])
    scale = D**-0.5

    eaqk, eakkod, eakkd = kda_intra_engine_fwd(q, k, g, beta, scale, None)
    eakk = chunk_kda_fwd_k4b_dense_cute(eakkod, eakkd)
    saqk, sakkd = chunk_kda_fwd_intra_diagonal(q, k, g, beta, scale, None)
    saqk, sakk = chunk_kda_fwd_inter_solve_cute(q, k, g, beta, sakkd, scale, Aqk=saqk)

    for name, got, ref, tol in (
        ("Aqk", eaqk, saqk, 2e-3),
        ("Akkd", eakkd, sakkd, 5e-3),
        ("Akk", eakk, sakk, 1e-2),
    ):
        assert torch.isfinite(got.float()).all(), f"{name} produced nonfinite values"
        assert _max_scaled(got, ref) < tol, f"{name} diverged from the shipped dense path"


def test_fwd_engine_normalizes_misaligned_gate_and_beta():
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra_engine import kda_intra_engine_fwd

    lengths = [65, 0, 63]
    metadata, q, k, gate, beta = _inputs(lengths, heads=2)
    gate_storage = torch.empty(gate.numel() + 1, device="cuda", dtype=gate.dtype)
    misaligned_gate = gate_storage[1:].view_as(gate)
    misaligned_gate.copy_(gate)
    beta_storage = torch.empty(beta.numel() + 1, device="cuda", dtype=beta.dtype)
    misaligned_beta = beta_storage[1:].view_as(beta)
    misaligned_beta.copy_(beta)
    assert misaligned_gate.data_ptr() % 128
    assert misaligned_beta.data_ptr() % 8

    expected = kda_intra_engine_fwd(q, k, gate, beta, D**-0.5, metadata)
    actual = kda_intra_engine_fwd(
        q,
        k,
        misaligned_gate,
        misaligned_beta,
        D**-0.5,
        metadata,
    )
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True], ids=["dense", "all-empty-packed"])
def test_fwd_engine_preserves_zero_capacity_shapes(packed):
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_inter_solve import (
        chunk_kda_fwd_k4b_dense_cute,
        chunk_kda_fwd_k4b_ragged_cute,
    )
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra_engine import kda_intra_engine_fwd

    tokens = 0
    q, k, _value, gate, beta = make_kda_test_inputs(tokens, heads=2, seed=5)
    metadata = (
        prepare_ragged_chunk_metadata(cumulative_sequence_offsets([0, 0]), tokens, BT)
        if packed
        else None
    )
    aqk, akkod, akkd = kda_intra_engine_fwd(q, k, gate, beta, D**-0.5, metadata)
    akk = (
        chunk_kda_fwd_k4b_ragged_cute(akkod, akkd, metadata)
        if metadata is not None
        else chunk_kda_fwd_k4b_dense_cute(akkod, akkd)
    )

    assert akkod.shape == (0, 2 * 256)
    assert aqk.shape == akk.shape == (1, tokens, 2, BT)
    assert akkd.shape == (1, tokens, 2, 16)
