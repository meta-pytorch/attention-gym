"""Registered operators for fused gated delta rule implementations."""

from __future__ import annotations

import importlib

import torch

_RECURRENT_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor? initial_state, "
    "Tensor? cu_seqlens, float scale, bool autotune)"
)
torch.library.define("attn_gym::gdn_recurrent_fwd", _RECURRENT_ARGS + " -> (Tensor, Tensor)")
torch.library.define("attn_gym::gdn_recurrent_fwd_no_state", _RECURRENT_ARGS + " -> Tensor")


def _recurrent_backend():
    try:
        return importlib.import_module("attn_gym.linear.gdn.impl.fused")
    except ImportError as error:
        raise ImportError(
            "recurrent_gdn(impl='fused') requires CUDA with Triton support"
        ) from error


def _recurrent_fwd_cuda(*args):
    return _recurrent_backend()._gdn_recurrent_fwd_cuda(*args)


def _recurrent_fwd_no_state_cuda(*args):
    return _recurrent_backend()._gdn_recurrent_fwd_no_state_cuda(*args)


torch.library.impl("attn_gym::gdn_recurrent_fwd", "CUDA", _recurrent_fwd_cuda)
torch.library.impl(
    "attn_gym::gdn_recurrent_fwd_no_state",
    "CUDA",
    _recurrent_fwd_no_state_cuda,
)


@torch.library.register_fake("attn_gym::gdn_recurrent_fwd")
def _recurrent_fwd_fake(
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
    del k, gate, beta, initial_state, scale, autotune
    num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    final_state = q.new_empty(
        num_sequences, q.shape[2], q.shape[3], v.shape[-1], dtype=torch.float32
    )
    return torch.empty_like(v, dtype=q.dtype), final_state


@torch.library.register_fake("attn_gym::gdn_recurrent_fwd_no_state")
def _recurrent_fwd_no_state_fake(
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
    del k, gate, beta, initial_state, cu_seqlens, scale, autotune
    return torch.empty_like(v, dtype=q.dtype)


recurrent_fwd_op = torch.ops.attn_gym.gdn_recurrent_fwd.default
recurrent_fwd_no_state_op = torch.ops.attn_gym.gdn_recurrent_fwd_no_state.default


def recurrent_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    cu_seqlens: torch.Tensor | None,
    scale: float,
    output_final_state: bool,
    autotune: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Normalize inputs and invoke the fused recurrent GDN operator."""
    if q.shape[-1] > 256:
        raise ValueError(f"recurrent_gdn requires K in [1, 256], got {q.shape[-1]}")
    if not q.is_cuda:
        raise ValueError("recurrent_gdn(impl='fused') requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("recurrent_gdn(impl='fused') requires float16, bfloat16, or float32 QKV")
    tensors = (q, k, v, gate, beta) + (() if initial_state is None else (initial_state,))
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in tensors):
        raise RuntimeError(
            "recurrent_gdn(impl='fused') is inference-only and has no backward; "
            "call under torch.no_grad() or torch.inference_mode()"
        )

    q, k, v = (tensor.contiguous() for tensor in (q, k, v))
    gate, beta = (tensor.float().contiguous() for tensor in (gate, beta))
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    args = (q, k, v, gate, beta, initial_state, cu_seqlens, scale, autotune)
    if output_final_state:
        return recurrent_fwd_op(*args)
    return recurrent_fwd_no_state_op(*args), None


__all__ = ["recurrent_forward", "recurrent_fwd_no_state_op", "recurrent_fwd_op"]
