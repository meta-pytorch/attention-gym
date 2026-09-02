# SPDX-License-Identifier: BSD-3-Clause

"""Torch launcher for the private CuTeDSL 4.7 scalar-GDN forward kernel."""

from __future__ import annotations

import torch

from attn_gym._backends.cute import tensor_supports_contiguous_dim, tensor_supports_tma
from attn_gym.linear._delta_rule.paged_state import PagedState
from attn_gym.linear._delta_rule.validation import resolve_scale
from attn_gym.utils import ceildiv

from .kernels import gdn_prefill_f16 as kernel
from .kernels.common.host import tensormap_workspace_bytes
from .kernels.compat import get_device_properties, tensor_device_index
from .schedule import prepare_mega_schedule

_SUPPORTED_IO_DTYPES = (torch.float16, torch.bfloat16)
_WORKSPACE_WORD_BYTES = 8


def validate_available(q: torch.Tensor) -> None:
    """Validate the scalar-GDN device contract without launching work."""
    if not q.is_cuda:
        raise ValueError("the Mega GDN backend requires CUDA tensors")
    properties = get_device_properties(tensor_device_index(q))
    if (properties.major, properties.minor) not in ((10, 0), (10, 3)):
        raise ValueError("the CuTeDSL 4.7 GDN backend requires SM100 or SM103")


def run_forward_on_current_device(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state: torch.Tensor | PagedState | None,
    *,
    scale: float | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Validate and launch one exact unsplit packed scalar-GDN forward.

    A ``PagedState`` is advanced in place and produces no separate final state.
    """
    paged_state = state if isinstance(state, PagedState) else None
    initial_state = paged_state.cache if paged_state is not None else state
    scale = resolve_scale(scale, q.shape[-1])
    validate_available(q)
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
        raise ValueError("q must have shape [1, T, HK, 128]")
    if k.shape != q.shape:
        raise ValueError("k must match q")
    _, tokens, key_heads, key_dim = q.shape
    if value.ndim != 4 or value.shape[:2] != (1, tokens) or value.shape[-1] != 128:
        raise ValueError("value must have shape [1, T, H, 128]")
    heads = value.shape[2]
    if heads % key_heads:
        raise ValueError("value heads must be a positive multiple of query/key heads")
    if gate.shape != (1, tokens, heads) or beta.shape != gate.shape:
        raise ValueError("gate and beta must have shape [1, T, H]")
    if q.dtype not in _SUPPORTED_IO_DTYPES or k.dtype != q.dtype or value.dtype != q.dtype:
        raise TypeError("q, k, and value must share dtype float16 or bfloat16")
    if gate.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("gate and beta must be float32")

    inputs = (q, k, value, gate, beta)
    if any(not tensor.is_cuda for tensor in inputs):
        raise ValueError("all inputs must be CUDA tensors")
    if any(tensor.device != q.device for tensor in inputs[1:]):
        raise ValueError("all inputs must be on q.device")
    if any(not tensor_supports_tma(tensor) for tensor in (q, k, value)):
        raise TypeError("q, k, and value require a TMA-compatible inner mode")
    if any(
        not tensor_supports_contiguous_dim(tensor, alignment_bytes=4) for tensor in (gate, beta)
    ):
        raise TypeError("gate and beta require a contiguous, element-aligned head mode")
    if (
        cu_seqlens.ndim != 1
        or cu_seqlens.shape[0] < 2
        or cu_seqlens.dtype != torch.int32
        or not cu_seqlens.is_contiguous()
        or not cu_seqlens.is_cuda
        or cu_seqlens.device != q.device
        or cu_seqlens.data_ptr() % 8
    ):
        raise TypeError("cu_seqlens must be aligned contiguous int32 on q.device")

    num_sequences = cu_seqlens.shape[0] - 1
    if paged_state is not None:
        if output_final_state:
            raise ValueError("paged state advances the pool in place; drop output_final_state")
    elif initial_state is not None:
        expected_state = (num_sequences, heads, 128, key_dim)
        if initial_state.shape != expected_state or initial_state.dtype != torch.float32:
            raise TypeError(f"initial_state must be float32 with shape {expected_state}")
        if not initial_state.is_cuda or initial_state.device != q.device:
            raise ValueError("initial_state must be on q.device")
        if not tensor_supports_tma(initial_state):
            raise TypeError("initial_state requires a TMA-compatible inner mode")

    if output_final_state and initial_state is None:
        initial_state = torch.zeros(
            num_sequences,
            heads,
            128,
            key_dim,
            dtype=torch.float32,
            device=q.device,
        )
    final_state = initial_state.clone() if output_final_state else None
    if paged_state is not None:
        final_state = paged_state.cache
    output = torch.empty_like(value)
    stream = torch.cuda.current_stream(q.device).cuda_stream
    schedule = prepare_mega_schedule(
        gate,
        cu_seqlens,
        tile_tokens=kernel.CFG.B_T,
        counter_count=2,
        split=False,
        stream=stream,
    )
    tensormap_workspace = torch.empty(
        ceildiv(tensormap_workspace_bytes(kernel, num_sequences), _WORKSPACE_WORD_BYTES),
        dtype=torch.int64,
        device=q.device,
    )
    kernel.chunk_gdn_sm100(
        q[0],
        k[0],
        value[0],
        gate[0],
        beta[0],
        output[0],
        cu_seqlens,
        initial_state,
        final_state,
        scale,
        work_items=schedule.work_items,
        work_count=schedule.work_count,
        sched_ctr=schedule.counters,
        log_gate=True,
        tensormap_workspace=tensormap_workspace,
        state_indices=None if paged_state is None else paged_state.indices,
        has_initial_state=None if paged_state is None else paged_state.byte_mask,
    )
    return output, (None if paged_state is not None else final_state)


def run_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state: torch.Tensor | PagedState | None,
    *,
    scale: float | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Launch under the input tensor's CUDA device guard."""
    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    with torch.cuda.device(q.device):
        return run_forward_on_current_device(
            q,
            k,
            value,
            gate,
            beta,
            cu_seqlens,
            state,
            scale=scale,
            output_final_state=output_final_state,
        )


__all__ = ["run_forward", "validate_available"]
