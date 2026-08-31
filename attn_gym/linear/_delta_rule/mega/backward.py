# SPDX-License-Identifier: BSD-3-Clause

"""Torch launcher for the CuTeDSL 4.7 Mega no-state long-context backward."""

from __future__ import annotations

import torch

from attn_gym._backends.cute import tensor_supports_contiguous_dim, tensor_supports_tma
from attn_gym.linear._delta_rule.mega.kernels.common.host import tensormap_workspace_bytes
from attn_gym.linear._delta_rule.mega.kernels.compat import initialized_cuda_device
from attn_gym.linear._delta_rule.validation import resolve_scale
from attn_gym.utils import ceildiv

from .kernels import kda_bprop_f16, kda_recompute_f16
from .schedule import prepare_mega_schedule

_SUPPORTED_IO_DTYPES = (torch.float16, torch.bfloat16)
_KERNEL_CHUNK_SIZE = 16
_SCHEDULER_COUNTERS = 4
_WORKSPACE_WORD_BYTES = 8


def chunk_delta_rule_bwd_mega_packed(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    d_output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float | None = None,
    split: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run checkpoint recompute followed by exact or forgetting-horizon backward."""
    scale = resolve_scale(scale, q.shape[-1])
    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    with initialized_cuda_device(q):
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
            raise ValueError("q must have shape [1, T, H, 128]")
        if any(tensor.shape != q.shape for tensor in (k, value, gate, d_output)):
            raise ValueError("k, value, gate, and d_output must match q")
        if beta.shape != q.shape[:3]:
            raise ValueError("beta must have shape [1, T, H]")
        inputs = (q, k, value, gate, beta, d_output)
        if any(tensor.device != q.device for tensor in inputs):
            raise ValueError("all inputs must be on q.device")
        if any(not tensor_supports_tma(tensor) for tensor in (q, k, value, gate, d_output)):
            raise TypeError("q, k, value, gate, and d_output require a TMA-compatible inner mode")
        if not tensor_supports_contiguous_dim(beta, alignment_bytes=4):
            raise TypeError("beta requires a contiguous, element-aligned inner mode")
        if q.dtype not in _SUPPORTED_IO_DTYPES or any(
            tensor.dtype != q.dtype for tensor in (k, value, d_output)
        ):
            raise TypeError("q, k, value, and d_output must share dtype float16 or bfloat16")
        if gate.dtype != torch.float32 or beta.dtype != torch.float32:
            raise TypeError("gate and beta must be float32")

        _, tokens, heads, dim = q.shape
        if (
            cu_seqlens.ndim != 1
            or cu_seqlens.shape[0] < 2
            or cu_seqlens.dtype != torch.int32
            or not cu_seqlens.is_contiguous()
            or cu_seqlens.device != q.device
            or cu_seqlens.data_ptr() % 8
        ):
            raise TypeError("cu_seqlens must be aligned contiguous int32 on q.device")
        num_sequences = cu_seqlens.shape[0] - 1
        stream = torch.cuda.current_stream(q.device).cuda_stream
        schedule = prepare_mega_schedule(
            gate,
            cu_seqlens,
            tile_tokens=_KERNEL_CHUNK_SIZE,
            counter_count=_SCHEDULER_COUNTERS,
            split=split,
            stream=stream,
        )
        checkpoints = torch.empty(
            tokens // _KERNEL_CHUNK_SIZE + num_sequences,
            heads,
            dim,
            dim,
            dtype=q.dtype,
            device=q.device,
        )
        recompute_workspace = torch.empty(
            ceildiv(
                tensormap_workspace_bytes(kda_recompute_f16, num_sequences),
                _WORKSPACE_WORD_BYTES,
            ),
            dtype=torch.int64,
            device=q.device,
        )
        backward_workspace = torch.empty(
            ceildiv(
                tensormap_workspace_bytes(kda_bprop_f16, num_sequences),
                _WORKSPACE_WORD_BYTES,
            ),
            dtype=torch.int64,
            device=q.device,
        )
        dq = torch.empty_like(q[0])
        dk = torch.empty_like(k[0])
        dv = torch.empty_like(value[0])
        dgate = torch.empty_like(gate[0])
        dbeta = torch.empty_like(beta[0])

        kda_recompute_f16.chunk_kda_recompute_sm100(
            k[0],
            value[0],
            gate[0],
            beta[0],
            cu_seqlens,
            None,
            None,
            checkpoint_every_n_tokens=16,
            output_state_checkpoints=checkpoints,
            work_items=schedule.work_items,
            work_count=schedule.work_count,
            sched_ctr=schedule.counters[:2],
            sched_all=schedule.counters,
            work_item_scratch=schedule.item_scratch,
            order_in_prologue=True,
            tensormap_workspace=recompute_workspace,
        )
        bwd_scheduler = (
            schedule.counters[2:] if num_sequences * heads <= schedule.num_sms else None
        )
        kda_bprop_f16.chunk_kda_bwd_sm100(
            q[0],
            k[0],
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
            use_initial_state=False,
            d_initial_state=None,
            d_final_state=None,
            work_items=schedule.work_items,
            work_count=schedule.work_count,
            sched_ctr=bwd_scheduler,
            order_in_prologue=False,
            tensormap_workspace=backward_workspace,
        )
        return (
            dq.unsqueeze(0),
            dk.unsqueeze(0),
            dv.unsqueeze(0),
            dgate.unsqueeze(0),
            dbeta.unsqueeze(0),
        )


__all__ = ["chunk_delta_rule_bwd_mega_packed"]
