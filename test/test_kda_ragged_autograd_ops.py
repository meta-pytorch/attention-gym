"""Custom-op contract tests for composed ragged KDA autograd."""

from __future__ import annotations

import pytest
import torch

from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd import (
    _chunk_kda_bwd_custom_op,
    _chunk_kda_fwd_ragged_custom_op,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0),
    reason="the CuTe KDA core requires CUDA capability 10.0 or newer",
)


def _offsets(lengths: list[int]) -> torch.Tensor:
    return torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device="cuda",
        dtype=torch.int32,
    )


def _inputs(tokens: int):
    torch.manual_seed(41)
    shape = (1, tokens, 1, 128)
    values = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8,
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8,
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) / 8,
        -torch.rand(shape, device="cuda"),
        torch.rand(1, tokens, 1, device="cuda"),
    ]
    return tuple(value.requires_grad_() for value in values)


def test_ragged_custom_op_registrations():
    inputs = _inputs(128)
    initial_state = (torch.randn(2, 1, 128, 128, device="cuda") / 8).requires_grad_()
    cu_seqlens = _offsets([65, 63])
    forward_args = (
        *inputs,
        initial_state,
        cu_seqlens,
        True,
        False,
        False,
    )
    torch.library.opcheck(
        _chunk_kda_fwd_ragged_custom_op,
        forward_args,
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )

    with torch.no_grad():
        output, state, Aqk, Akk, chunk_offsets = _chunk_kda_fwd_ragged_custom_op(*forward_args)
    torch.library.opcheck(
        _chunk_kda_bwd_custom_op,
        (
            *(value.detach() for value in inputs),
            Aqk,
            Akk,
            cu_seqlens,
            chunk_offsets,
            chunk_offsets.new_empty(()),
            torch.randn_like(output),
            torch.randn_like(state),
            initial_state.detach(),
            True,
            False,
            False,
        ),
        test_utils=("test_schema", "test_faketensor", "test_aot_dispatch_dynamic"),
        rtol=2e-2,
        atol=2e-3,
    )
