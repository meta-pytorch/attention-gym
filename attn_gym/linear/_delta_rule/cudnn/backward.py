# SPDX-License-Identifier: BSD-3-Clause

"""Validated v1.30 KDA backward adapter, with optional state and exit cotangent."""

from __future__ import annotations

import torch

from attn_gym._backends.cute import tensor_supports_contiguous_dim, tensor_supports_tma
from attn_gym._backends.cute.utils import initialized_cuda_device
from attn_gym.linear._delta_rule.validation import resolve_scale

from ..cudnn_fe.kda import kda_backward

_SUPPORTED_IO_DTYPES = (torch.float16, torch.bfloat16)


def chunk_delta_rule_bwd_cudnn_packed(
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
    initial_state: torch.Tensor | None = None,
    d_final_state: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
]:
    """Run checkpoint recompute followed by exact or forgetting-horizon backward.

    ``initial_state`` and ``d_final_state`` are optional FP32 ``[N, H, V, K]`` entry states
    and exit cotangents per packed sequence. ``None`` skips state loads, with token gradients
    bitwise equal to an explicit zero tensor. Returns ``(dq, dk, dv, dgate, dbeta,
    d_initial_state)``; the last is ``None`` unless ``initial_state`` was given.
    The approximate forgetting-horizon ``split`` schedule requires a no-state call.
    """
    scale = resolve_scale(scale, q.shape[-1])
    if not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    if split and (initial_state is not None or d_final_state is not None):
        raise ValueError("the split backward schedule requires a no-state call")
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

        _, _, heads, dim = q.shape
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
        state_shape = (num_sequences, heads, value.shape[-1], dim)
        for name, state in (("initial_state", initial_state), ("d_final_state", d_final_state)):
            if state is None:
                continue
            if state.shape != state_shape or state.dtype != torch.float32:
                raise TypeError(f"{name} must be float32 with shape {state_shape}")
            if state.device != q.device:
                raise ValueError(f"{name} must be on q.device")
            if not tensor_supports_tma(state):
                raise TypeError(f"{name} requires a TMA-compatible inner mode")
        *gradients, d_initial_state = kda_backward(
            q[0],
            k[0],
            value[0],
            gate[0],
            beta[0],
            d_output[0],
            cu_seqlens,
            scale=scale,
            split=split,
            initial_state=initial_state,
            d_final_state=d_final_state,
        )
        return (*(grad.unsqueeze(0) for grad in gradients), d_initial_state)


__all__ = ["chunk_delta_rule_bwd_cudnn_packed"]
