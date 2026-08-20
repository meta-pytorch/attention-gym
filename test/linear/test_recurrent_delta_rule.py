# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Correctness tests for the shared recurrent delta-rule scan."""

import math

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from attn_gym.linear import recurrent_gdn
from attn_gym.linear._delta_rule.recurrent import launch_recurrent_delta_rule_fwd

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused recurrent scan requires CUDA"
)


def _make_inputs(
    *,
    tokens: int,
    dtype: torch.dtype,
    initial_state: bool,
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
    batch, heads, key_dim, value_dim = 2, 2, 64, 48
    query = F.normalize(torch.randn(batch, heads, tokens, key_dim, device="cuda"), dim=-1)
    key = F.normalize(torch.randn_like(query), dim=-1)
    value = torch.randn(batch, heads, tokens, value_dim, device="cuda")
    gate = F.logsigmoid(torch.randn(batch, heads, tokens, device="cuda"))
    beta = torch.sigmoid(torch.randn(batch, heads, tokens, device="cuda"))
    state = torch.randn(batch, heads, key_dim, value_dim, device="cuda") if initial_state else None
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
    """Adapt public head-major GDN inputs to the private token-major scan."""
    output, final_state = launch_recurrent_delta_rule_fwd(
        query.transpose(1, 2).contiguous(),
        key.transpose(1, 2).contiguous(),
        value.transpose(1, 2).contiguous(),
        gate.transpose(1, 2).contiguous(),
        beta.transpose(1, 2).contiguous(),
        initial_state,
        None,
        scale=scale,
        scalar_ln_gate=True,
        store_final_state=store_final_state,
    )
    return output.transpose(1, 2), final_state


@pytest.mark.parametrize(
    "dtype,tokens,use_initial_state,return_final_state,scale",
    [
        (torch.float32, 37, True, True, 0.37),
        (torch.bfloat16, 1, False, False, None),
    ],
)
def test_scalar_gate_matches_recurrent_gdn(
    dtype, tokens, use_initial_state, return_final_state, scale
):
    query, key, value, gate, beta, state = _make_inputs(
        tokens=tokens, dtype=dtype, initial_state=use_initial_state
    )
    kernel_scale = query.shape[-1] ** -0.5 if scale is None else scale
    with torch.no_grad():
        expected = recurrent_gdn(
            query,
            key,
            value,
            gate,
            beta,
            scale=scale,
            initial_state=state,
            return_final_state=return_final_state,
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

    tolerance = 1e-5 if dtype == torch.float32 else 2e-2
    assert output.dtype == dtype
    torch.testing.assert_close(
        output.float(), expected.output.float(), rtol=tolerance, atol=tolerance
    )
    if return_final_state:
        assert final_state is not None and final_state.dtype == torch.float32
        torch.testing.assert_close(
            final_state, expected.final_state, rtol=tolerance, atol=tolerance
        )
    else:
        assert final_state is None and expected.final_state is None


def test_scalar_and_vector_gate_specializations_agree():
    query, key, value, gate, beta, state = _make_inputs(
        tokens=37, dtype=torch.float32, initial_state=True
    )
    scale = query.shape[-1] ** -0.5
    with torch.no_grad():
        scalar_output, scalar_state = _launch_scalar_gdn(
            query, key, value, gate, beta, state, scale=scale, store_final_state=True
        )
        token_gate = gate.transpose(1, 2).contiguous()
        vector_gate = token_gate.unsqueeze(-1).expand(*token_gate.shape, query.shape[-1])
        vector_output, vector_state = launch_recurrent_delta_rule_fwd(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            (vector_gate * math.log2(math.e)).contiguous(),
            beta.transpose(1, 2).contiguous(),
            state,
            None,
            scale=scale,
            scalar_ln_gate=False,
            store_final_state=True,
        )

    torch.testing.assert_close(scalar_output, vector_output.transpose(1, 2), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(scalar_state, vector_state, rtol=1e-5, atol=1e-5)
