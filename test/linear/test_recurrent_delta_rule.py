# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness tests for the shared recurrent delta-rule scan."""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from attn_gym.linear import recurrent_gdn
from attn_gym.linear._delta_rule.recurrent import GateKind, launch_recurrent_delta_rule_fwd
from attn_gym.testing.kda import assert_matches_low_precision_reference

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused recurrent scan requires CUDA"
)


def _make_inputs(
    *,
    tokens: int,
    dtype: torch.dtype,
    initial_state: bool,
    key_dim: int = 64,
    value_dim: int = 48,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Create stable scalar-gated inputs in the public GDN layout."""
    torch.manual_seed(0)
    batch, heads = 2, 2
    query = F.normalize(torch.randn(batch, tokens, heads, key_dim, device="cuda"), dim=-1)
    key = F.normalize(torch.randn_like(query), dim=-1)
    value = torch.randn(batch, tokens, heads, value_dim, device="cuda")
    gate = F.logsigmoid(torch.randn(batch, tokens, heads, device="cuda"))
    beta = torch.sigmoid(torch.randn(batch, tokens, heads, device="cuda"))
    state = torch.randn(batch, heads, value_dim, key_dim, device="cuda") if initial_state else None
    return query.to(dtype), key.to(dtype), value.to(dtype), gate, beta, state


def _launch_scalar_gdn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    scale: float,
    store_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch the scalar-gate specialization with public GDN inputs."""
    output, final_state = launch_recurrent_delta_rule_fwd(
        query,
        key,
        value,
        gate,
        beta,
        initial_state,
        None,
        scale=scale,
        gate_kind=GateKind.SCALAR,
        store_final_state=store_final_state,
    )
    return output, final_state


@pytest.mark.parametrize(
    "dtype,tokens,use_initial_state,return_final_state,scale,key_dim,value_dim",
    [
        (torch.float32, 37, True, True, 0.37, 64, 48),
        (torch.float32, 3, True, True, None, 40, 37),
        (torch.float32, 5, False, False, None, 64, 48),
        (torch.bfloat16, 37, True, True, None, 64, 48),
    ],
)
def test_scalar_gate_matches_recurrent_gdn(
    dtype, tokens, use_initial_state, return_final_state, scale, key_dim, value_dim
):
    query, key, value, gate, beta, state = _make_inputs(
        tokens=tokens,
        dtype=dtype,
        initial_state=use_initial_state,
        key_dim=key_dim,
        value_dim=value_dim,
    )
    kernel_scale = query.shape[-1] ** -0.5 if scale is None else scale
    with torch.no_grad():
        expected = recurrent_gdn(
            query,
            key,
            value,
            gate,
            beta,
            state,
            scale=scale,
            output_final_state=return_final_state,
            impl="reference",
        )
        output, final_state = _launch_scalar_gdn(
            query,
            key,
            value,
            gate,
            beta,
            state,
            scale=kernel_scale,
            store_final_state=return_final_state,
        )

    assert output.dtype == dtype
    if return_final_state:
        assert final_state is not None and final_state.dtype == torch.float32
    else:
        assert final_state is None and expected[1] is None
    if dtype == torch.bfloat16:
        high_precision = recurrent_gdn(
            query.double(),
            key.double(),
            value.double(),
            gate.double(),
            beta.double(),
            scale=scale,
            initial_state=None if state is None else state.double(),
            output_final_state=return_final_state,
            impl="reference",
        )
        assert_matches_low_precision_reference(output, high_precision[0], expected[0], "output")
        if return_final_state:
            assert_matches_low_precision_reference(
                final_state, high_precision[1], expected[1], "final_state"
            )
    else:
        torch.testing.assert_close(output, expected[0], rtol=1e-5, atol=1e-5)
        if return_final_state:
            torch.testing.assert_close(final_state, expected[1], rtol=1e-5, atol=1e-5)


def test_scalar_and_vector_gate_specializations_agree():
    query, key, value, gate, beta, state = _make_inputs(
        tokens=37, dtype=torch.float32, initial_state=True
    )
    scale = query.shape[-1] ** -0.5
    with torch.no_grad():
        scalar_output, scalar_state = _launch_scalar_gdn(
            query, key, value, gate, beta, state, scale=scale, store_final_state=True
        )
        vector_gate = gate.unsqueeze(-1).expand(*gate.shape, query.shape[-1])
        vector_output, vector_state = launch_recurrent_delta_rule_fwd(
            query,
            key,
            value,
            vector_gate.contiguous(),
            beta,
            state,
            None,
            scale=scale,
            gate_kind=GateKind.VECTOR,
            store_final_state=True,
        )

    torch.testing.assert_close(scalar_output, vector_output, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(scalar_state, vector_state, rtol=1e-5, atol=1e-5)
