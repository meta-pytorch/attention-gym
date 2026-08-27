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


__all__ = []
