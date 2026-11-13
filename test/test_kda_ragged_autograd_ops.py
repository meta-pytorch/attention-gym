"""Custom-op registration contracts for composed ragged KDA."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear._delta_rule.triton.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.constants import LOG2_E
from attn_gym.linear.kda.naive import chunk_cumsum_ref
from attn_gym.linear.kda.ops import (
    chunk_bwd_op,
    chunk_bwd_with_state_grad_op,
    chunk_fwd_ragged_op,
    chunk_fwd_ragged_with_state_op,
)
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="the fused KDA core requires CUDA capability 8.0 or newer",
)


_DEFAULT_SCALE = 128**-0.5


def test_ragged_custom_op_registrations():
    q, k, v, gate, beta = make_kda_test_inputs(128, requires_grad=True)
    initial_state = (torch.randn(2, 1, 128, 128, device="cuda") / 8).requires_grad_()
    cu_seqlens = cumulative_sequence_offsets([65, 63])
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, q.shape[1], 64)
    cumulative_gate = chunk_cumsum_ref(gate, 64, scale=LOG2_E, cu_seqlens=cu_seqlens)
    inputs = (q, k, v, cumulative_gate, beta)
    forward_args = (
        *(value.detach() for value in inputs),
        initial_state.detach(),
        cu_seqlens,
        metadata.chunk_offsets,
        _DEFAULT_SCALE,
        True,
        "auto",
    )
    torch.library.opcheck(
        chunk_fwd_ragged_with_state_op,
        forward_args,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )

    torch.library.opcheck(
        chunk_fwd_ragged_op,
        forward_args,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )

    with torch.no_grad():
        output, state, Aqk, Akk = chunk_fwd_ragged_with_state_op(*forward_args)
    torch.library.opcheck(
        chunk_bwd_with_state_grad_op,
        (
            *(value.detach() for value in inputs),
            Aqk,
            Akk,
            cu_seqlens,
            metadata.chunk_offsets,
            torch.randn_like(output),
            torch.randn_like(state),
            initial_state.detach(),
            _DEFAULT_SCALE,
            False,
            True,
            "auto",
        ),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )

    no_state_args = (
        *(value.detach() for value in inputs),
        None,
        cu_seqlens,
        metadata.chunk_offsets,
        _DEFAULT_SCALE,
        True,
        "auto",
    )
    with torch.no_grad():
        output, Aqk, Akk = chunk_fwd_ragged_op(*no_state_args)
    torch.library.opcheck(
        chunk_bwd_op,
        (
            *(value.detach() for value in inputs),
            Aqk,
            Akk,
            cu_seqlens,
            metadata.chunk_offsets,
            torch.randn_like(output),
            None,
            None,
            _DEFAULT_SCALE,
            False,
            True,
            "auto",
        ),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )


def test_ragged_backward_ignores_akk_capacity_slack():
    from attn_gym.linear.kda.fwd.factors import chunk_kda_fwd_factors

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
    aqk, akk = chunk_kda_fwd_factors(q, k, gate, beta, _DEFAULT_SCALE, metadata)
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
        _DEFAULT_SCALE,
        False,
        False,
        "auto",
    )
    expected = chunk_bwd_op(*common[:6], clean_akk, *common[6:])
    actual = chunk_bwd_op(*common[:6], poisoned_akk, *common[6:])
    for got, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(got[:, :active_tokens], ref[:, :active_tokens], rtol=0, atol=0)
