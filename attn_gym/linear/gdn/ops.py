"""Registered operators for fused gated delta rule implementations."""

from __future__ import annotations

import importlib

import torch

from attn_gym._backends.cute import get_device_properties
from attn_gym.linear.kda.chunk_schedule import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.ops import _plain_gate_scan_op

_CHUNK_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor cumulative_gate, Tensor beta, "
    "Tensor? initial_state, float scale)"
)
torch.library.define("attn_gym::gdn_chunk_fwd", _CHUNK_ARGS + " -> (Tensor, Tensor)")
torch.library.define(
    "attn_gym::gdn_chunk_fwd_with_state",
    _CHUNK_ARGS + " -> (Tensor, Tensor, Tensor)",
)
_CHUNK_BWD_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor cumulative_gate, Tensor beta, Tensor inverse, "
    "Tensor d_output, Tensor? d_final_state, Tensor? initial_state, Tensor? cu_seqlens, "
    "Tensor? chunk_offsets, float scale)"
)
torch.library.define(
    "attn_gym::gdn_chunk_bwd",
    _CHUNK_BWD_ARGS + " -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::gdn_chunk_bwd_with_state_grad",
    _CHUNK_BWD_ARGS + " -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)
_CHUNK_PACKED_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor cumulative_gate, Tensor beta, "
    "Tensor? initial_state, Tensor cu_seqlens, Tensor chunk_offsets, int capacity, float scale)"
)
torch.library.define(
    "attn_gym::gdn_chunk_fwd_packed",
    _CHUNK_PACKED_ARGS + " -> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::gdn_chunk_fwd_packed_with_state",
    _CHUNK_PACKED_ARGS + " -> (Tensor, Tensor, Tensor)",
)
_RECURRENT_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor? initial_state, "
    "Tensor? cu_seqlens, float scale, bool autotune)"
)
torch.library.define("attn_gym::gdn_recurrent_fwd", _RECURRENT_ARGS + " -> (Tensor, Tensor)")
torch.library.define("attn_gym::gdn_recurrent_fwd_no_state", _RECURRENT_ARGS + " -> Tensor")
torch.library.define(
    "attn_gym::gdn_recurrent_fwd_paged",
    "(Tensor q, Tensor k, Tensor v, Tensor gate, Tensor beta, Tensor(a!) state_cache, "
    "Tensor state_indices, Tensor? has_initial_state, Tensor? cu_seqlens, float scale) -> Tensor",
)
torch.library.define(
    "attn_gym::gdn_recurrent_decode",
    "(Tensor packed_qkv, Tensor raw_gate, Tensor raw_beta, Tensor A_log, Tensor dt_bias,"
    " Tensor(a!) state_cache, Tensor state_indices, Tensor? has_initial_state, Tensor(b!) out,"
    " float scale) -> ()",
)


def _chunk_backend():
    try:
        return importlib.import_module("attn_gym.linear.gdn.impl.chunk")
    except ImportError as error:
        raise ImportError("chunk_gdn(impl='fused') requires CUDA with Triton support") from error


def _chunk_fwd_cuda(*args):
    return _chunk_backend()._gdn_chunk_fwd_cuda(*args)


def _chunk_fwd_with_state_cuda(*args):
    return _chunk_backend()._gdn_chunk_fwd_with_state_cuda(*args)


def _chunk_bwd_cuda(*args):
    return _chunk_backend()._gdn_chunk_bwd_cuda(*args)


def _chunk_bwd_with_state_grad_cuda(*args):
    return _chunk_backend()._gdn_chunk_bwd_with_state_grad_cuda(*args)


def _chunk_fwd_packed_cuda(*args):
    return _chunk_backend()._gdn_chunk_fwd_packed_cuda(*args)


def _chunk_fwd_packed_with_state_cuda(*args):
    return _chunk_backend()._gdn_chunk_fwd_packed_with_state_cuda(*args)


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


def _recurrent_fwd_paged_cuda(*args):
    return _recurrent_backend()._gdn_recurrent_fwd_paged_cuda(*args)


def _recurrent_decode_cuda(*args):
    return _recurrent_backend()._gdn_recurrent_decode_cuda(*args)


torch.library.impl("attn_gym::gdn_chunk_fwd", "CUDA", _chunk_fwd_cuda)
torch.library.impl(
    "attn_gym::gdn_chunk_fwd_with_state",
    "CUDA",
    _chunk_fwd_with_state_cuda,
)
torch.library.impl("attn_gym::gdn_chunk_bwd", "CUDA", _chunk_bwd_cuda)
torch.library.impl(
    "attn_gym::gdn_chunk_bwd_with_state_grad",
    "CUDA",
    _chunk_bwd_with_state_grad_cuda,
)
torch.library.impl("attn_gym::gdn_chunk_fwd_packed", "CUDA", _chunk_fwd_packed_cuda)
torch.library.impl(
    "attn_gym::gdn_chunk_fwd_packed_with_state",
    "CUDA",
    _chunk_fwd_packed_with_state_cuda,
)
torch.library.impl("attn_gym::gdn_recurrent_fwd", "CUDA", _recurrent_fwd_cuda)
torch.library.impl(
    "attn_gym::gdn_recurrent_fwd_no_state",
    "CUDA",
    _recurrent_fwd_no_state_cuda,
)
torch.library.impl(
    "attn_gym::gdn_recurrent_fwd_paged",
    "CUDA",
    _recurrent_fwd_paged_cuda,
)
torch.library.impl(
    "attn_gym::gdn_recurrent_decode",
    "CUDA",
    _recurrent_decode_cuda,
)


@torch.library.register_fake("attn_gym::gdn_chunk_fwd")
def _chunk_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del k, cumulative_gate, beta, initial_state, scale
    inverse = q.new_empty(q.shape[0], q.shape[1], v.shape[2], 64)
    return torch.empty_like(v, dtype=q.dtype), inverse


@torch.library.register_fake("attn_gym::gdn_chunk_fwd_with_state")
def _chunk_fwd_with_state_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output, inverse = _chunk_fwd_fake(q, k, v, cumulative_gate, beta, initial_state, scale)
    final_state = q.new_empty(
        q.shape[0], v.shape[2], v.shape[-1], q.shape[-1], dtype=torch.float32
    )
    return output, final_state, inverse


@torch.library.register_fake("attn_gym::gdn_chunk_fwd_packed")
def _chunk_fwd_packed_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    capacity: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    del k, cumulative_gate, beta, initial_state, cu_seqlens, chunk_offsets, capacity, scale
    inverse = q.new_empty(q.shape[0], q.shape[1], v.shape[2], 64)
    return torch.empty_like(v, dtype=q.dtype), inverse


@torch.library.register_fake("attn_gym::gdn_chunk_fwd_packed_with_state")
def _chunk_fwd_packed_with_state_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    capacity: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    output, inverse = _chunk_fwd_packed_fake(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        cu_seqlens,
        chunk_offsets,
        capacity,
        scale,
    )
    final_state = q.new_empty(
        cu_seqlens.shape[0] - 1,
        v.shape[2],
        v.shape[-1],
        q.shape[-1],
        dtype=torch.float32,
    )
    return output, final_state, inverse


@torch.library.register_fake("attn_gym::gdn_chunk_bwd")
def _chunk_bwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    inverse: torch.Tensor,
    d_output: torch.Tensor,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, ...]:
    del inverse, d_output, d_final_state, initial_state, cu_seqlens, chunk_offsets, scale
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(cumulative_gate),
        torch.empty_like(beta),
    )


@torch.library.register_fake("attn_gym::gdn_chunk_bwd_with_state_grad")
def _chunk_bwd_with_state_grad_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    inverse: torch.Tensor,
    d_output: torch.Tensor,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None,
    scale: float,
) -> tuple[torch.Tensor, ...]:
    outputs = _chunk_bwd_fake(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        inverse,
        d_output,
        d_final_state,
        initial_state,
        cu_seqlens,
        chunk_offsets,
        scale,
    )
    state_batch = q.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    d_initial_state = q.new_empty(
        state_batch, v.shape[2], v.shape[-1], q.shape[-1], dtype=torch.float32
    )
    return *outputs, d_initial_state


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
    # The state carries one [V, K] slab per value head; grouped callers have v.shape[2] > HK.
    final_state = q.new_empty(
        num_sequences, v.shape[2], v.shape[-1], q.shape[3], dtype=torch.float32
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


@torch.library.register_fake("attn_gym::gdn_recurrent_fwd_paged")
def _recurrent_fwd_paged_fake(
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
) -> torch.Tensor:
    del k, gate, beta, state_cache, state_indices, has_initial_state, cu_seqlens, scale
    return torch.empty_like(v, dtype=q.dtype)


@torch.library.register_fake("attn_gym::gdn_recurrent_decode")
def _recurrent_decode_fake(
    packed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    out: torch.Tensor,
    scale: float,
) -> None:
    """The decode op returns nothing and mutates preallocated buffers; no metadata to fake."""


chunk_fwd_op = torch.ops.attn_gym.gdn_chunk_fwd.default
chunk_fwd_with_state_op = torch.ops.attn_gym.gdn_chunk_fwd_with_state.default
chunk_bwd_op = torch.ops.attn_gym.gdn_chunk_bwd.default
chunk_bwd_with_state_grad_op = torch.ops.attn_gym.gdn_chunk_bwd_with_state_grad.default
chunk_fwd_packed_op = torch.ops.attn_gym.gdn_chunk_fwd_packed.default
chunk_fwd_packed_with_state_op = torch.ops.attn_gym.gdn_chunk_fwd_packed_with_state.default
recurrent_fwd_op = torch.ops.attn_gym.gdn_recurrent_fwd.default
recurrent_fwd_no_state_op = torch.ops.attn_gym.gdn_recurrent_fwd_no_state.default
recurrent_fwd_paged_op = torch.ops.attn_gym.gdn_recurrent_fwd_paged.default
recurrent_decode_op = torch.ops.attn_gym.gdn_recurrent_decode.default


class _ChunkGDN(torch.autograd.Function):
    """Attach first-order autograd to dense or fixed-capacity packed chunk operators."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        chunk_offsets,
        capacity,
        scale,
        output_final_state,
    ):
        cumulative_gate = _plain_gate_scan_op(
            gate.unsqueeze(-1), cu_seqlens, chunk_offsets, False
        ).squeeze(-1)
        if cu_seqlens is None:
            args = (q, k, v, cumulative_gate, beta, initial_state, scale)
            if output_final_state:
                output, final_state, inverse = chunk_fwd_with_state_op(*args)
            else:
                output, inverse = chunk_fwd_op(*args)
        else:
            args = (
                q,
                k,
                v,
                cumulative_gate,
                beta,
                initial_state,
                cu_seqlens,
                chunk_offsets,
                capacity,
                scale,
            )
            if output_final_state:
                output, final_state, inverse = chunk_fwd_packed_with_state_op(*args)
            else:
                output, inverse = chunk_fwd_packed_op(*args)
        ctx.save_for_backward(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            inverse,
            initial_state,
            cu_seqlens,
            chunk_offsets,
        )
        ctx.scale = scale
        ctx.set_materialize_grads(False)
        if output_final_state:
            return output, final_state
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_output, d_final_state=None):
        (
            q,
            k,
            v,
            cumulative_gate,
            beta,
            inverse,
            initial_state,
            cu_seqlens,
            chunk_offsets,
        ) = ctx.saved_tensors
        if d_output is None:
            d_output = torch.zeros_like(v)
        if torch.compiler.is_compiling():
            args = (
                q,
                k,
                v,
                cumulative_gate,
                beta,
                inverse,
                d_output,
                d_final_state,
                initial_state,
                cu_seqlens,
                chunk_offsets,
                ctx.scale,
            )
            if initial_state is not None:
                dq, dk, dv, d_gate, db, d_initial_state = chunk_bwd_with_state_grad_op(*args)
            else:
                dq, dk, dv, d_gate, db = chunk_bwd_op(*args)
                d_initial_state = None
        else:
            backend = _chunk_backend()
            metadata = backend.resolve_backward_metadata(q, cu_seqlens, chunk_offsets)
            dq, dk, dv, d_gate, db, d_initial_state = backend.chunk_gdn_bwd(
                q,
                k,
                v,
                cumulative_gate,
                beta,
                inverse,
                d_output,
                d_final_state,
                initial_state,
                metadata,
                ctx.scale,
            )
        return dq, dk, dv, d_gate, db, d_initial_state, None, None, None, None, None


def chunk_forward(
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
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Route dense inputs and invoke the fused scalar chunk forward operator."""
    if not q.is_cuda:
        raise ValueError("chunk_gdn(impl='fused') requires CUDA tensors")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("chunk_gdn(impl='fused') requires matching float16 or bfloat16 QKV")
    if q.shape[-1] != 128 or v.shape[-1] != 128:
        raise ValueError("chunk_gdn(impl='fused') requires K=V=128")
    # The dense recurrence uses Hopper TensorDescriptors/TMA; Blackwell additionally selects
    # the CuTe backward, while Hopper uses the portable Triton backward.
    if not torch.compiler.is_compiling() and get_device_properties(q.device).major < 9:
        raise ValueError("chunk_gdn(impl='fused') requires CUDA capability 9.0 or newer")
    output_shape = v.shape
    batch, tokens = q.shape[:2]
    if cu_seqlens is not None and batch != 1:
        raise ValueError("explicit packed chunk_gdn requires batch size one")
    # The dense kernels specialize for complete BT64 chunks. Flatten other dense inputs and
    # synthesize one packed segment per batch row so tails stay masked and never cross batches.
    if cu_seqlens is None and (batch != 1 or tokens % 64):
        q = q.reshape(1, batch * tokens, q.shape[2], q.shape[3])
        k = k.reshape(1, batch * tokens, k.shape[2], k.shape[3])
        v = v.reshape(1, batch * tokens, v.shape[2], v.shape[3])
        gate = gate.reshape(1, batch * tokens, gate.shape[2])
        beta = beta.reshape(1, batch * tokens, beta.shape[2])
        cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * tokens

    metadata = (
        prepare_ragged_chunk_metadata(cu_seqlens, q.shape[1], 64)
        if cu_seqlens is not None
        else None
    )
    gate = gate.float()
    beta = beta.float()
    if initial_state is not None:
        initial_state = initial_state.float()
    chunk_offsets = None if metadata is None else metadata.chunk_offsets
    capacity = 0 if metadata is None else metadata.capacity
    result = _ChunkGDN.apply(
        q,
        k,
        v,
        gate,
        beta,
        initial_state,
        cu_seqlens,
        chunk_offsets,
        capacity,
        scale,
        output_final_state,
    )
    if output_final_state:
        output, final_state = result
        return output.reshape(output_shape), final_state
    return result.reshape(output_shape), None


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
    state_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
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
    if state_indices is not None:
        assert initial_state is not None
        return recurrent_fwd_paged_op(
            q,
            k,
            v,
            gate,
            beta,
            initial_state,
            state_indices,
            has_initial_state,
            cu_seqlens,
            scale,
        ), None
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    args = (q, k, v, gate, beta, initial_state, cu_seqlens, scale, autotune)
    if output_final_state:
        return recurrent_fwd_op(*args)
    return recurrent_fwd_no_state_op(*args), None


def recurrent_decode_forward(
    packed_qkv: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    out: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Invoke the lazily loaded fused decode implementation."""
    if not packed_qkv.is_cuda:
        raise ValueError("recurrent_gdn_decode requires CUDA tensors")
    data_tensors = (packed_qkv, raw_gate, raw_beta, A_log, dt_bias, state_cache, out)
    if torch.is_grad_enabled() and any(tensor.requires_grad for tensor in data_tensors):
        raise RuntimeError(
            "recurrent_gdn_decode is inference-only and has no backward; "
            "call under torch.no_grad() / torch.inference_mode()"
        )
    recurrent_decode_op(
        packed_qkv,
        raw_gate,
        raw_beta,
        A_log,
        dt_bias,
        state_cache,
        state_indices,
        has_initial_state,
        out,
        scale,
    )
    return out


__all__ = [
    "chunk_bwd_op",
    "chunk_bwd_with_state_grad_op",
    "chunk_forward",
    "chunk_fwd_op",
    "chunk_fwd_packed_op",
    "chunk_fwd_packed_with_state_op",
    "chunk_fwd_with_state_op",
    "recurrent_decode_forward",
    "recurrent_decode_op",
    "recurrent_forward",
    "recurrent_fwd_no_state_op",
    "recurrent_fwd_op",
    "recurrent_fwd_paged_op",
]
