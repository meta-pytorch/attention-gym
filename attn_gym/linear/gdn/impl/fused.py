"""Fused gated delta rule implementations."""

from __future__ import annotations

import torch

from attn_gym.linear._delta_rule.decode import GateTransform, launch_recurrent_delta_rule_decode
from attn_gym.linear._delta_rule.recurrent import GateKind, launch_recurrent_delta_rule_fwd


def _launch_recurrent(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    scale: float,
    autotune: bool,
    *,
    output_final_state: bool,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch the scalar-gate specialization of the shared log2-space scan."""
    return launch_recurrent_delta_rule_fwd(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale=scale,
        gate_kind=GateKind.SCALAR,
        store_final_state=output_final_state,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        autotune=autotune,
    )


def _gdn_recurrent_fwd_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    scale: float,
    autotune: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, final_state = _launch_recurrent(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale,
        autotune,
        output_final_state=True,
    )
    assert final_state is not None
    return output, final_state


def _gdn_recurrent_fwd_no_state_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    scale: float,
    autotune: bool,
) -> torch.Tensor:
    return _launch_recurrent(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        scale,
        autotune,
        output_final_state=False,
    )[0]


def _gdn_recurrent_fwd_paged_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Advance selected cache slots with the shared scalar-gate scan."""
    return _launch_recurrent(
        q,
        k,
        v,
        gate,
        beta,
        state_cache,
        cu_seqlens,
        scale,
        False,
        output_final_state=True,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
    )[0]


def _gdn_recurrent_decode_cuda(
    packed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    output: torch.Tensor,
    scale: float,
) -> None:
    """Advance one token per sequence with GDN preprocessing fused into the step."""
    heads, value_dim, key_dim = state_cache.shape[1:]
    key_heads = (packed_qkv.shape[1] - heads * value_dim) // (2 * key_dim)
    launch_recurrent_delta_rule_decode(
        packed_qkv,
        # The shared boundary takes token-major gates; drop the vLLM-style unit token dim.
        raw_gate[0],
        raw_beta[0],
        A_log,
        dt_bias,
        state_cache,
        state_indices,
        output,
        gate_kind=GateKind.SCALAR,
        gate_transform=GateTransform.SOFTPLUS,
        key_heads=key_heads,
        lower_bound=0.0,
        scale=scale,
        has_initial_state=has_initial_state,
        op_name="recurrent_gdn_decode",
    )


__all__ = []
