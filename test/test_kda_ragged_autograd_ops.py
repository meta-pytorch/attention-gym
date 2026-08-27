"""Custom-op registration contracts for composed ragged KDA."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("cutlass")

from attn_gym.linear.kda.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.constants import LOG2_E
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd import (
    _chunk_kda_bwd_op,
    _chunk_kda_bwd_with_state_grad_op,
    _chunk_kda_fwd_ragged_op,
    _chunk_kda_fwd_ragged_with_state_op,
)
from attn_gym.linear.kda.naive import chunk_cumsum_ref
from attn_gym.testing.kda import cumulative_sequence_offsets, make_kda_test_inputs

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTe KDA core requires CUDA capability 10.0 or 10.3",
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
    )
    torch.library.opcheck(
        _chunk_kda_fwd_ragged_with_state_op,
        forward_args,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )

    torch.library.opcheck(
        _chunk_kda_fwd_ragged_op,
        forward_args,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )

    with torch.no_grad():
        output, state, Aqk, Akk = _chunk_kda_fwd_ragged_with_state_op(*forward_args)
    torch.library.opcheck(
        _chunk_kda_bwd_with_state_grad_op,
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
    )
    with torch.no_grad():
        output, Aqk, Akk = _chunk_kda_fwd_ragged_op(*no_state_args)
    torch.library.opcheck(
        _chunk_kda_bwd_op,
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
        ),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )
