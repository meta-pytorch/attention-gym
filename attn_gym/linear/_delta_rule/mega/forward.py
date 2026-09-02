# SPDX-License-Identifier: BSD-3-Clause

"""Shared Torch launcher for the CuTeDSL 4.7 Mega forward kernel."""

from __future__ import annotations

import torch

from attn_gym._backends.cute import tensor_supports_contiguous_dim, tensor_supports_tma
from attn_gym.linear._delta_rule.paged_state import PagedState
from attn_gym.linear._delta_rule.validation import resolve_scale
from attn_gym.utils import ceildiv

from .kernels import kda_prefill_f16 as kernel
from .kernels.common.host import tensormap_workspace_bytes
from .kernels.compat import get_device_properties, tensor_device_index
from .schedule import prepare_mega_schedule

_SUPPORTED_IO_DTYPES = (torch.float16, torch.bfloat16)
_WORKSPACE_WORD_BYTES = 8


def validate_available(q: torch.Tensor) -> None:
    """Validate the device contract without launching asynchronous work."""
    if not q.is_cuda:
        raise ValueError("the Mega KDA backend requires CUDA tensors")
    properties = get_device_properties(tensor_device_index(q))
    if (properties.major, properties.minor) not in ((10, 0), (10, 3)):
        raise ValueError("the CuTeDSL 4.7 KDA backend requires SM100 or SM103")


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
    """Validate and launch one unsplit packed forward invocation."""
    paged_state = state if isinstance(state, PagedState) else None
    initial_state = paged_state.cache if paged_state is not None else state
    scale = resolve_scale(scale, q.shape[-1])
    validate_available(q)
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
        raise ValueError("q must have shape [1, T, H, 128]")
    if k.shape != q.shape or gate.shape != q.shape:
        raise ValueError("k and gate must match q")
    heads = q.shape[2]
    if value.shape != q.shape:
        raise ValueError("value must match q")
    if beta.shape != q.shape[:3]:
        raise ValueError("beta must have shape [1, T, H]")
    if q.dtype not in _SUPPORTED_IO_DTYPES or k.dtype != q.dtype or value.dtype != q.dtype:
        raise TypeError("q, k, and value must share dtype float16 or bfloat16")
    if gate.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("gate and beta must be float32")
    inputs = (q, k, value, gate, beta)
    if any(not tensor.is_cuda for tensor in inputs):
        raise ValueError("all inputs must be CUDA tensors")
    if any(tensor.device != q.device for tensor in inputs[1:]):
        raise ValueError("all inputs must be on q.device")
    if any(not tensor_supports_tma(tensor) for tensor in (q, k, value, gate)):
        raise TypeError("q, k, value, and gate require a TMA-compatible inner mode")
    if not tensor_supports_contiguous_dim(beta, alignment_bytes=4):
        raise TypeError("beta requires a contiguous, element-aligned inner mode")
    if (
        cu_seqlens.ndim != 1
        or cu_seqlens.shape[0] < 2
        or cu_seqlens.dtype != torch.int32
        or not cu_seqlens.is_contiguous()
    ):
        raise TypeError("cu_seqlens must be a contiguous int32 vector")
    if not cu_seqlens.is_cuda or cu_seqlens.device != q.device:
        raise ValueError("cu_seqlens must be on the input CUDA device")

    num_sequences = cu_seqlens.shape[0] - 1
    if paged_state is not None:
        if output_final_state:
            raise ValueError("paged state advances the pool in place; drop output_final_state")
    elif initial_state is not None:
        expected_state = (num_sequences, heads, 128, 128)
        if initial_state.shape != expected_state or initial_state.dtype != torch.float32:
            raise TypeError(f"initial_state must be float32 with shape {expected_state}")
        if not initial_state.is_cuda or initial_state.device != q.device:
            raise ValueError("initial_state must be on the input CUDA device")
        if not tensor_supports_tma(initial_state):
            raise TypeError("initial_state requires a TMA-compatible inner mode")
    if output_final_state and initial_state is None:
        raise ValueError("output_final_state requires an initial_state buffer")

    # Empty sequences emit no token work. Cloning only when requested preserves their state.
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
    kernel.chunk_kda_sm100(
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
        work_item_scratch=schedule.item_scratch,
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


def chunk_delta_rule_fwd_mega_unsplit(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Run an unsplit packed forward without recurrent state."""
    output, _ = run_forward(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens,
        None,
        scale=scale,
        output_final_state=False,
    )
    return output


def chunk_delta_rule_fwd_mega_unsplit_with_initial_state(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Run an unsplit packed forward from an initial state without storing the final state."""
    output, _ = run_forward(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens,
        initial_state,
        scale=scale,
        output_final_state=False,
    )
    return output


def chunk_delta_rule_fwd_mega_unsplit_with_state(
    q: torch.Tensor,
    k: torch.Tensor,
    value: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run an unsplit packed forward and return the final `[V, K]` state."""
    output, final_state = run_forward(
        q,
        k,
        value,
        gate,
        beta,
        cu_seqlens,
        initial_state,
        scale=scale,
        output_final_state=True,
    )
    assert final_state is not None
    return output, final_state


__all__ = [
    "chunk_delta_rule_fwd_mega_unsplit",
    "chunk_delta_rule_fwd_mega_unsplit_with_initial_state",
    "chunk_delta_rule_fwd_mega_unsplit_with_state",
]
