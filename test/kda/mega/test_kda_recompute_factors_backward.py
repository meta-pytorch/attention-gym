"""Tests for the minimal-factor bridge into the existing KDA backward."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.naive import chunk_cumsum_ref
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the fused KDA backward requires SM100 or SM103",
)

D = 128


def inputs(lengths: list[int], heads: int = 2):
    """Create dense or packed KDA core inputs and optional routing metadata."""
    from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata

    total = sum(lengths)
    q, k, v, increments, beta = make_kda_test_inputs(
        total,
        heads=heads,
        seed=0,
        gate_scale=1 / 16,
    )
    cu_seqlens = cumulative_sequence_offsets(lengths)
    gate = chunk_cumsum_ref(
        increments,
        64,
        cu_seqlens=cu_seqlens if len(lengths) > 1 else None,
    ).contiguous()
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, total, 64) if len(lengths) > 1 else None
    return q, k, v, gate, beta.contiguous(), metadata


@pytest.mark.parametrize("lengths", [[128], [65, 0, 63]], ids=["dense", "packed"])
def test_recomputed_factors_backward_matches_saved_factors(lengths):
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra import chunk_kda_fwd_factors
    from attn_gym.linear.kda.ops import chunk_bwd_op, chunk_bwd_recompute_factors_op

    q, k, v, gate, beta, metadata = inputs(lengths)
    Aqk, Akk = chunk_kda_fwd_factors(q, k, gate, beta, D**-0.5, metadata)
    d_output = torch.randn_like(v)
    cu_seqlens = None if metadata is None else metadata.cu_seqlens
    chunk_offsets = None if metadata is None else metadata.chunk_offsets
    common = (
        q,
        k,
        v,
        gate,
        beta,
        cu_seqlens,
        chunk_offsets,
        d_output,
        None,
        None,
        D**-0.5,
        False,
        False,
    )
    expected = chunk_bwd_op(*common[:5], Aqk, Akk, *common[5:])
    actual = chunk_bwd_recompute_factors_op(*common)
    for got, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_ragged_backward_ignores_akk_capacity_slack():
    from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra import chunk_kda_fwd_factors
    from attn_gym.linear.kda.ops import chunk_bwd_op

    lengths = [65, 63]
    active_tokens, physical_tokens = sum(lengths), 256
    q, k, v, increments, beta = make_kda_test_inputs(
        physical_tokens,
        heads=2,
        seed=7,
        gate_scale=1 / 16,
    )
    cu_seqlens = cumulative_sequence_offsets(lengths)
    gate = torch.zeros_like(increments)
    gate[:, :active_tokens] = chunk_cumsum_ref(
        increments[:, :active_tokens],
        64,
        cu_seqlens=cu_seqlens,
    )
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, physical_tokens, 64)
    aqk, akk = chunk_kda_fwd_factors(q, k, gate, beta, D**-0.5, metadata)
    clean_akk = akk.clone()
    clean_akk[:, active_tokens:] = 0
    poisoned_akk = akk.clone()
    poisoned_akk[:, active_tokens:] = float("nan")
    common = (
        q,
        k,
        v,
        gate,
        beta,
        aqk,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        torch.randn_like(v),
        None,
        None,
        D**-0.5,
        False,
        False,
    )
    expected = chunk_bwd_op(*common[:6], clean_akk, *common[6:])
    actual = chunk_bwd_op(*common[:6], poisoned_akk, *common[6:])
    for got, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(got[:, :active_tokens], ref[:, :active_tokens], rtol=0, atol=0)


def test_recomputed_factors_backward_with_state_matches_saved_factors():
    from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra import chunk_kda_fwd_factors
    from attn_gym.linear.kda.ops import (
        chunk_bwd_recompute_factors_with_state_grad_op,
        chunk_bwd_with_state_grad_op,
    )

    q, k, v, gate, beta, metadata = inputs([65, 0, 63])
    assert metadata is not None
    aqk, akk = chunk_kda_fwd_factors(q, k, gate, beta, D**-0.5, metadata)
    initial_state = torch.randn(3, q.shape[2], D, D, device="cuda") / 8
    d_final_state = torch.randn_like(initial_state)
    common = (
        q,
        k,
        v,
        gate,
        beta,
        metadata.cu_seqlens,
        metadata.chunk_offsets,
        torch.randn_like(v),
        d_final_state,
        initial_state,
        D**-0.5,
        False,
        False,
    )
    expected = chunk_bwd_with_state_grad_op(*common[:5], aqk, akk, *common[5:])
    actual = chunk_bwd_recompute_factors_with_state_grad_op(*common)
    for got, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_recomputed_factors_backward_op_registration():
    from attn_gym.linear.kda.ops import (
        chunk_bwd_recompute_factors_op,
        chunk_bwd_recompute_factors_with_state_grad_op,
    )

    q, k, v, gate, beta, _ = inputs([64], heads=1)
    d_output = torch.randn_like(v)
    torch.library.opcheck(
        chunk_bwd_recompute_factors_op,
        (q, k, v, gate, beta, None, None, d_output, None, None, D**-0.5, False, False),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )
    initial_state = torch.randn(1, 1, D, D, device="cuda") / 8
    torch.library.opcheck(
        chunk_bwd_recompute_factors_with_state_grad_op,
        (
            q,
            k,
            v,
            gate,
            beta,
            None,
            None,
            d_output,
            torch.randn_like(initial_state),
            initial_state,
            D**-0.5,
            False,
            False,
        ),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )
