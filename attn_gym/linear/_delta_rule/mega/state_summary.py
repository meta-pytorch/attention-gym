# SPDX-License-Identifier: BSD-3-Clause

"""Native BT16 affine probes for canonical Mega context-parallel tiles.

B is the no-state final state; A is the identity-seeded, zero-value final state. Native
state/operand rounding makes this an approximate affine map, not bitwise reconstruction of
an ordinary unsharded Mega run. Fixed whole-sequence work items define the canonical baseline.
"""

from __future__ import annotations

import torch

from attn_gym._backends.cute.utils import initialized_cuda_device
from attn_gym.utils import ceildiv

from .forward import validate_available
from .kernels import kda_recompute_f16
from .kernels.common.host import tensormap_workspace_bytes
from .schedule import prepare_mega_schedule


def build_mega_state_summaries(
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Return FP32 ``[N, H, 256, 128]`` maps packed as ``[B; A]`` for whole sequences.

    Inputs use the staged ``[1, T, H, 128]`` layout, with FP32 gate/beta and already
    normalized keys. Each packed sequence is one canonical tile. Both passes are unsplit,
    independent of packed token count or ownership; empty sequences return ``[0; I]``.
    The transition pass synthesizes identity and zero values on chip, without reading V.
    """
    validate_available(k)
    if k.ndim != 4 or k.shape[0] != 1 or k.shape[-1] != 128:
        raise ValueError("k must have shape [1, T, H, 128]")
    if value.shape != k.shape or gate.shape != k.shape or beta.shape != k.shape[:3]:
        raise ValueError("value, gate, and beta must match k's token and head extents")
    if k.dtype not in (torch.float16, torch.bfloat16) or value.dtype != k.dtype:
        raise TypeError("k and value must share dtype float16 or bfloat16")
    if gate.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("gate and beta must be float32")
    if any(t.device != k.device for t in (value, gate, beta, cu_seqlens)):
        raise ValueError("all inputs must be on k.device")

    with initialized_cuda_device(k):
        num_sequences, heads = cu_seqlens.shape[0] - 1, k.shape[2]
        summaries = torch.zeros(
            num_sequences, heads, 256, 128, dtype=torch.float32, device=k.device
        )
        summaries[:, :, 128:, :].diagonal(dim1=-2, dim2=-1).fill_(1)
        if k.shape[1] == 0:
            return summaries
        schedule = prepare_mega_schedule(
            gate,
            cu_seqlens,
            tile_tokens=16,
            counter_count=2,
            split=False,
            stream=torch.cuda.current_stream(k.device).cuda_stream,
        )
        workspace = torch.empty(
            ceildiv(tensormap_workspace_bytes(kda_recompute_f16, num_sequences), 8),
            dtype=torch.int64,
            device=k.device,
        )
        for transition_only in (False, True):
            kda_recompute_f16.chunk_kda_recompute_sm100(
                k[0],
                None if transition_only else value[0],
                gate[0],
                beta[0],
                cu_seqlens,
                initial_state=None,
                output_state=summaries[:, :, 128:, :]
                if transition_only
                else summaries[:, :, :128, :],
                work_items=schedule.work_items,
                work_count=schedule.work_count,
                sched_ctr=schedule.counters,
                sched_all=schedule.counters,
                order_in_prologue=True,
                tensormap_workspace=workspace,
                transition_only=transition_only,
            )
        return summaries
