"""Fused gated delta rule implementations."""

from __future__ import annotations

import torch

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
    qk_l2norm: bool = False,
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
        qk_l2norm=qk_l2norm,
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
    qk_l2norm: bool,
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
        qk_l2norm=qk_l2norm,
    )[0]


__all__ = []
