"""Structural validation shared by public delta-rule operations."""

from __future__ import annotations

import functools
import sys
import warnings
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

import torch

from attn_gym._cu_seqlens import validate_cu_seqlens
from attn_gym.linear._delta_rule.paged_state import PagedState, validate_has_initial_state
from attn_gym.linear.types import StateCheckpoints
from attn_gym.types import KernelOptionSpec, validate_kernel_options

SUPPORTED_ACTIVATION_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def validate_delta_rule_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    *,
    op_name: str,
    gate_name: str,
    vector_gate: bool,
    allow_grouped_heads: bool,
) -> None:
    """Validate structural invariants shared by scalar- and vector-gated delta rules."""
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, H, K], got {tuple(q.shape)}")
    batch, tokens, key_heads, key_dim = q.shape
    if batch == 0 or tokens == 0 or key_heads == 0 or key_dim == 0:
        raise ValueError(f"q must have nonempty dimensions, got {tuple(q.shape)}")
    if k.shape != q.shape:
        raise ValueError(f"k must have shape {tuple(q.shape)}, got {tuple(k.shape)}")
    if v.ndim != 4 or v.shape[:2] != (batch, tokens) or v.shape[-1] < 1:
        raise ValueError(f"v must have shape [{batch}, {tokens}, H, V], got {tuple(v.shape)}")

    heads = v.shape[2]
    grouped_heads = heads != key_heads
    if grouped_heads and not (allow_grouped_heads and heads != 0 and heads % key_heads == 0):
        relation = "be a positive multiple of" if allow_grouped_heads else "match"
        raise ValueError(
            f"v heads must {relation} q heads for {op_name}, "
            f"got {heads} value heads for {key_heads} query heads"
        )

    gate_shape = (batch, tokens, heads, key_dim) if vector_gate else (batch, tokens, heads)
    if gate.shape != gate_shape:
        raise ValueError(f"{gate_name} must have shape {gate_shape}, got {tuple(gate.shape)}")
    if beta.shape != (batch, tokens, heads):
        raise ValueError(f"beta must have shape {(batch, tokens, heads)}, got {tuple(beta.shape)}")

    if cu_seqlens is not None:
        validate_cu_seqlens(batch, q.device, cu_seqlens=cu_seqlens)

    state_batch = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    expected_state = (state_batch, heads, v.shape[-1], key_dim)
    if initial_state is not None and initial_state.shape != expected_state:
        raise ValueError(
            f"initial_state must have shape {expected_state}, got {tuple(initial_state.shape)}"
        )

    tensors = (q, k, v, gate, beta) + (() if initial_state is None else (initial_state,))
    if any(tensor.device != q.device for tensor in tensors[1:]):
        raise ValueError("all inputs must be on the same device")


def require_inference_only(tensors: Iterable[torch.Tensor | None], message: str) -> None:
    """Raise ``RuntimeError(message)`` if autograd would track any given tensor."""
    if torch.is_grad_enabled() and any(
        tensor is not None and tensor.requires_grad for tensor in tensors
    ):
        raise RuntimeError(message)


def validate_packed_offsets_tensor(cu_seqlens: torch.Tensor, device) -> None:
    """Check that packed offsets are an int32 ``[N + 1]`` vector on ``device``. Their values (a
    zero start, nondecreasing, an end within the tokens) are checked on the device by the first
    kernel that reads them (the warpdrive prologues, ``invalid packed cu_seqlens``)."""
    if (
        cu_seqlens.ndim != 1
        or cu_seqlens.shape[0] < 2
        or cu_seqlens.dtype != torch.int32
        or not cu_seqlens.is_cuda
        or cu_seqlens.device != device
    ):
        raise TypeError("cu_seqlens must be a contiguous int32 vector on q.device")


@functools.cache
def warn_cudnn_backend_renamed() -> None:
    """Warn once per process that ``backend="cudnn"`` is the deprecated name of ``"warpdrive"``."""
    warnings.warn(
        "kernel_options['backend']='cudnn' is deprecated and runs the same kernels as "
        "'warpdrive'; pass 'backend': 'warpdrive'",
        FutureWarning,
        stacklevel=4,
    )


def resolve_chunk_kernel_options(
    kernel_options: Mapping[str, object] | None, spec: KernelOptionSpec, *, op_name: str
) -> dict[str, Any]:
    """Resolve chunk options using ``SPLIT_OPTION_SPEC`` or an extension of it.

    Split schedules require ``backend="warpdrive"``. The backend's former name, ``"cudnn"``, still
    selects it and warns once per process, outside ``torch.compile`` tracing (Dynamo cannot trace
    ``warnings.warn`` in a full graph).
    """
    if kernel_options is not None and kernel_options.get("backend") == "cudnn":
        if not torch.compiler.is_compiling():
            warn_cudnn_backend_renamed()
        kernel_options = {**kernel_options, "backend": "warpdrive"}
    options = validate_kernel_options(kernel_options, spec, op_name=op_name)
    for name in ("split_backward", "split_forward"):
        if options[name] and options["backend"] != "warpdrive":
            raise ValueError(f"{name} requires kernel_options['backend']='warpdrive'")
    return options


def resolve_scale(scale: float | None, key_dim: int) -> float:
    """Resolve a query-scale override to a validated float, defaulting to ``1/sqrt(K)``."""
    if scale is None:
        return key_dim**-0.5
    if not isinstance(scale, Real) or isinstance(scale, bool):
        raise TypeError("scale must be a real scalar or None")
    scale = float(scale)
    if (
        scale != scale  # noqa: PLR0124 - compile-safe NaN check
        or scale <= 0
        or scale > sys.float_info.max
    ):
        raise ValueError(f"scale must be finite and positive, got {scale}")
    return scale


def validate_decode_inputs(
    packed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    *,
    op_name: str,
) -> tuple[int, int, int, int]:
    """Validate the family-independent fused-decode contract.

    Family-specific checks (packed channel layout, gate/beta/dt_bias shapes) stay with the
    caller. Returns ``(batch, heads, value_dim, key_dim)`` from the pool and buffer.
    """
    if packed_qkv.ndim != 2 or packed_qkv.shape[0] < 1 or packed_qkv.stride(1) != 1:
        raise ValueError("packed_qkv must have shape [B, C] and be contiguous within each token")
    if state_cache.ndim != 4:
        raise ValueError("state_cache must have shape [num_slots, H, V, K]")
    num_slots, heads, value_dim, key_dim = state_cache.shape
    batch = packed_qkv.shape[0]
    if num_slots < 1 or heads < 1 or key_dim < 1 or value_dim < 1:
        raise ValueError(
            f"state_cache must have nonempty dimensions, got {tuple(state_cache.shape)}"
        )
    if A_log.shape != (heads,) or A_log.dtype != torch.float32 or not A_log.is_contiguous():
        raise ValueError(f"A_log must be contiguous float32 with shape ({heads},)")
    if state_cache.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("state_cache must use float32 or bfloat16")
    if state_cache.stride()[1:] != (value_dim * key_dim, key_dim, 1):
        raise TypeError("state_cache must be contiguous within each [H, V, K] slot")
    if state_cache.stride(0) < heads * key_dim * value_dim:
        raise ValueError("state_cache slots must not overlap")
    if (
        state_indices.shape != (batch,)
        or state_indices.dtype != torch.int32
        or not state_indices.is_contiguous()
    ):
        raise ValueError(f"state_indices must be contiguous int32 with shape ({batch},)")
    validate_has_initial_state(has_initial_state, batch, packed_qkv.device)
    if key_dim > 256:
        raise ValueError(f"{op_name} requires K in [1, 256], got {key_dim}")

    device = packed_qkv.device
    all_tensors = (raw_gate, raw_beta, A_log, dt_bias, state_cache, state_indices)
    if any(tensor.device != device for tensor in all_tensors):
        raise ValueError(f"all {op_name} inputs must be on the same device")
    activation_tensors = (packed_qkv, raw_gate, raw_beta)
    if any(tensor.dtype not in SUPPORTED_ACTIVATION_DTYPES for tensor in activation_tensors):
        supported = ", ".join(str(dtype) for dtype in SUPPORTED_ACTIVATION_DTYPES)
        raise TypeError(f"decode activation inputs must use one of {supported}")
    return batch, heads, value_dim, key_dim


def resolve_decode_out(
    packed_qkv: torch.Tensor,
    out: torch.Tensor | None,
    expected_shape: tuple[int, ...],
) -> torch.Tensor:
    """Allocate the decode output or validate a caller-owned buffer."""
    if out is None:
        return packed_qkv.new_empty(expected_shape)
    if out.shape != expected_shape:
        raise ValueError(f"out must have shape {expected_shape}, got {tuple(out.shape)}")
    if out.dtype != packed_qkv.dtype:
        raise TypeError(f"out must use packed_qkv.dtype ({packed_qkv.dtype}), got {out.dtype}")
    if out.device != packed_qkv.device:
        raise ValueError("out must be on packed_qkv.device")
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    return out


def validate_paged_state(
    q: torch.Tensor,
    v: torch.Tensor,
    state_cache: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None = None,
) -> PagedState:
    """Validate and package the shared mutable ``[slots, H, V, K]`` state contract.

    The pool carries one state per value head, so grouped-head callers size ``H`` from ``v``.
    """
    num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    return PagedState.validate(
        state_cache,
        state_indices,
        has_initial_state,
        num_sequences=num_sequences,
        heads=v.shape[2],
        value_dim=v.shape[-1],
        key_dim=q.shape[3],
        device=q.device,
    )


def validate_state_checkpoints(
    checkpoints: StateCheckpoints | None,
    q: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> None:
    """Check snapshot metadata without reading device-resident offsets."""
    if checkpoints is None:
        return
    if not isinstance(checkpoints, StateCheckpoints):
        raise TypeError("state_checkpoints must be a StateCheckpoints instance")
    states, offsets, interval = checkpoints
    if type(interval) is not int:
        raise TypeError("state_checkpoints.every_n_tokens must be an int")
    if interval <= 0 or interval % 64:
        raise ValueError("state_checkpoints.every_n_tokens must be a positive multiple of 64")
    if states.ndim != 4 or states.shape[1:] != (v.shape[2], v.shape[-1], q.shape[-1]):
        raise ValueError("state_checkpoints.states must have shape [capacity, H, V, K]")
    if states.dtype != torch.float32:
        raise TypeError("state_checkpoints.states must use float32")
    if not states.is_contiguous():
        raise ValueError("state_checkpoints.states must be contiguous")
    num_sequences = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    if offsets.shape != (num_sequences + 1,):
        raise ValueError("state_checkpoints.cu_checkpoint_offsets must have shape [N + 1]")
    if offsets.dtype != torch.int32:
        raise TypeError("state_checkpoints.cu_checkpoint_offsets must use int32")
    if not offsets.is_contiguous():
        raise ValueError("state_checkpoints.cu_checkpoint_offsets must be contiguous")
    if states.device != q.device or offsets.device != q.device:
        raise ValueError("state_checkpoints tensors must be on the input device")
    require_inference_only(
        (states,),
        "state_checkpoints is inference-only; call under torch.no_grad() or torch.inference_mode()",
    )


__all__ = [
    "SUPPORTED_ACTIVATION_DTYPES",
    "require_inference_only",
    "resolve_chunk_kernel_options",
    "resolve_decode_out",
    "resolve_scale",
    "validate_decode_inputs",
    "validate_delta_rule_inputs",
    "validate_has_initial_state",
    "validate_packed_offsets_tensor",
    "validate_paged_state",
    "validate_state_checkpoints",
]
