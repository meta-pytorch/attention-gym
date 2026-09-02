# SPDX-License-Identifier: BSD-3-Clause

"""Torch launcher for scalar-GDN checkpoint recompute and Mega backward."""

from __future__ import annotations

import torch

from attn_gym.linear._delta_rule.mega.kernels.common.host import tensormap_workspace_bytes
from attn_gym.linear._delta_rule.mega.kernels.compat import (
    checkpoint_capacity_bound,
    initialized_cuda_device,
)
from attn_gym.linear._delta_rule.validation import resolve_scale
from attn_gym.utils import ceildiv

from .kernels import gdn_bprop_f16, gdn_recompute_f16
from .schedule import prepare_mega_schedule

_KERNEL_CHUNK_SIZE = 64
_SCHEDULER_COUNTERS = 4
_WORKSPACE_WORD_BYTES = 8


def chunk_gdn_bwd_mega_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    d_final_state: torch.Tensor | None = None,
    *,
    scale: float | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Run exact BT64 checkpoint recompute followed by scalar-GDN backward."""
    scale = resolve_scale(scale, q.shape[-1])
    if d_output.dtype != q.dtype:
        raise TypeError(f"d_output must use q.dtype ({q.dtype}), got {d_output.dtype}")
    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    with initialized_cuda_device(q):
        _, tokens, key_heads, key_dim = q.shape
        heads = value.shape[2]
        if k.shape != q.shape or heads % key_heads:
            raise ValueError("k must match q and value heads must be divisible by query heads")
        groups = heads // key_heads
        if groups > 1:
            # Temporary bridge until the kernel supports grouped-head reduction natively.
            q_kernel = q.repeat_interleave(groups, dim=2).contiguous()
            k_kernel = k.repeat_interleave(groups, dim=2).contiguous()
        else:
            q_kernel, k_kernel = q, k
        num_sequences = cu_seqlens.shape[0] - 1
        stream = torch.cuda.current_stream(q.device).cuda_stream
        schedule = prepare_mega_schedule(
            gate,
            cu_seqlens,
            tile_tokens=_KERNEL_CHUNK_SIZE,
            counter_count=_SCHEDULER_COUNTERS,
            split=False,
            stream=stream,
        )
        checkpoints = torch.empty(
            max(checkpoint_capacity_bound(tokens, num_sequences, _KERNEL_CHUNK_SIZE), 1),
            heads,
            value.shape[-1],
            key_dim,
            dtype=q.dtype,
            device=q.device,
        )
        recompute_workspace = torch.empty(
            ceildiv(
                tensormap_workspace_bytes(gdn_recompute_f16, num_sequences),
                _WORKSPACE_WORD_BYTES,
            ),
            dtype=torch.int64,
            device=q.device,
        )
        backward_workspace = torch.empty(
            ceildiv(
                tensormap_workspace_bytes(gdn_bprop_f16, num_sequences),
                _WORKSPACE_WORD_BYTES,
            ),
            dtype=torch.int64,
            device=q.device,
        )
        d_initial_state = torch.empty_like(initial_state) if initial_state is not None else None
        dq = torch.empty_like(q_kernel[0])
        dk = torch.empty_like(k_kernel[0])
        dv = torch.empty_like(value[0])
        dgate = torch.empty_like(gate[0])
        dbeta = torch.empty_like(beta[0])

        gdn_recompute_f16.chunk_gdn_recompute_sm100(
            k_kernel[0],
            value[0],
            gate[0],
            beta[0],
            cu_seqlens,
            initial_state,
            None,
            checkpoint_every_n_tokens=_KERNEL_CHUNK_SIZE,
            output_state_checkpoints=checkpoints,
            work_items=schedule.work_items,
            work_count=schedule.work_count,
            sched_ctr=schedule.counters[:2],
            sched_all=schedule.counters,
            order_in_prologue=True,
            log_gate=True,
            tensormap_workspace=recompute_workspace,
        )
        backward_scheduler = (
            schedule.counters[2:] if num_sequences * heads <= schedule.num_sms else None
        )
        gdn_bprop_f16.chunk_gdn_bwd_sm100(
            q_kernel[0],
            k_kernel[0],
            value[0],
            gate[0],
            beta[0],
            d_output[0],
            checkpoints,
            dq,
            dk,
            dv,
            dgate,
            dbeta,
            cu_seqlens,
            scale,
            use_initial_state=initial_state is not None,
            d_initial_state=d_initial_state,
            d_final_state=d_final_state,
            work_items=schedule.work_items,
            work_count=schedule.work_count,
            sched_ctr=backward_scheduler,
            tensormap_workspace=backward_workspace,
        )
        if groups > 1:
            reduced_dq = dq.reshape(tokens, key_heads, groups, key_dim).sum(2)
            reduced_dk = dk.reshape(tokens, key_heads, groups, key_dim).sum(2)
            dq = torch.empty_like(q[0]).copy_(reduced_dq)
            dk = torch.empty_like(k[0]).copy_(reduced_dk)
        return (
            dq.unsqueeze(0),
            dk.unsqueeze(0),
            dv.unsqueeze(0),
            dgate.unsqueeze(0),
            dbeta.unsqueeze(0),
            d_initial_state,
        )


__all__ = ["chunk_gdn_bwd_mega_packed"]
