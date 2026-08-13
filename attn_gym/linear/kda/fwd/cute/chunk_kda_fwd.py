"""Composed fixed-length and packed Blackwell KDA core forward."""

from __future__ import annotations

from contextlib import nullcontext
from enum import Enum
from typing import NamedTuple

import torch

from attn_gym._backends.cute import get_device_properties, tensor_supports_tma
from attn_gym.linear.kda.chunk_scheduler import (
    RaggedChunkMetadata,
    prepare_ragged_chunk_metadata,
)
from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra import chunk_kda_fwd_intra
from attn_gym.linear.kda.fwd.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o import chunk_gla_fwd_o_gk
from attn_gym.linear.kda.utils import ChunkMetadata, prepare_complete_chunk_metadata

_SUPPORTED_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# TODO: Revisit model-approved chunk sizes: this is a major performance lever,
# but it changes the KDA decomposition and rounding order, so it can affect numerics.
_CHUNK_SIZE = 64
_HEAD_DIM = 128


class SequenceMode(Enum):
    """Provenance of the logical sequences entering the packed KDA core."""

    DENSE = "dense"
    SHAPE_PACKED = "shape_packed"
    PACKED = "packed"


class SequenceMetadata(NamedTuple):
    """Static lowering result passed to the selected registered KDA operator."""

    mode: SequenceMode
    cu_seqlens: torch.Tensor | None
    packed_shape: tuple[int, int, int, int]
    output_shape: tuple[int, int, int, int]


def _validate_chunk_kda_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> None:
    """Validate the exported operation's contract before normalizing inputs."""
    if q.ndim != 4:
        raise ValueError(f"q must have shape [B, T, H, K], got {tuple(q.shape)}")
    batch, tokens, heads, head_dim = q.shape
    if tokens == 0 or heads == 0:
        raise ValueError(f"q must have nonempty token and head dimensions, got {tuple(q.shape)}")
    if k.shape != q.shape:
        raise ValueError(f"k must have shape {tuple(q.shape)}, got {tuple(k.shape)}")
    if v.shape != (batch, tokens, heads, _HEAD_DIM):
        raise ValueError(
            f"v must have shape {(batch, tokens, heads, _HEAD_DIM)}, got {tuple(v.shape)}"
        )
    if cumulative_gate.shape != q.shape:
        raise ValueError(
            f"cumulative_gate must have shape {tuple(q.shape)}, got {tuple(cumulative_gate.shape)}"
        )
    if beta.shape != (batch, tokens, heads):
        raise ValueError(f"beta must have shape {(batch, tokens, heads)}, got {tuple(beta.shape)}")
    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("packed cu_seqlens require q to have batch size one")
        if cu_seqlens.ndim != 1 or cu_seqlens.shape[0] < 2:
            raise ValueError("cu_seqlens must have shape [num_sequences + 1]")
        if cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_contiguous():
            raise ValueError("cu_seqlens must be contiguous CUDA int32")
    state_batch = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    expected_state = (state_batch, heads, head_dim, v.shape[-1])
    if initial_state is not None and initial_state.shape != expected_state:
        raise ValueError(
            f"initial_state must have shape {expected_state}, got {tuple(initial_state.shape)}"
        )
    data_tensors = (q, k, v, cumulative_gate, beta)
    if initial_state is not None:
        data_tensors += (initial_state,)
    tensors = data_tensors if cu_seqlens is None else (*data_tensors, cu_seqlens)
    if not all(tensor.is_cuda and tensor.device == q.device for tensor in tensors):
        raise ValueError("all chunk_kda inputs must be CUDA tensors on the same device")
    if any(tensor.dtype not in _SUPPORTED_INPUT_DTYPES for tensor in data_tensors):
        supported = ", ".join(str(dtype) for dtype in _SUPPORTED_INPUT_DTYPES)
        raise TypeError(f"chunk_kda inputs must use one of {supported}")
    if head_dim != _HEAD_DIM:
        raise ValueError("the CuTe KDA core requires K=V=128")
    if tokens % _CHUNK_SIZE:
        raise ValueError("the CuTe KDA core requires complete 64-token chunks")
    if not torch.compiler.is_compiling() and get_device_properties(q.device).major < 10:
        raise ValueError("the CuTe KDA core requires CUDA capability 10.0 or newer")


def _has_supported_qkv_layout(tensor: torch.Tensor) -> bool:
    """Return whether QKV rows satisfy the vectorized kernel input contract."""
    return tensor.stride(-2) == tensor.shape[-1] and tensor_supports_tma(tensor)


def _normalize_qkv_layout(tensor: torch.Tensor) -> torch.Tensor:
    """Copy only layouts unsupported by one or more composed KDA stages."""
    if _has_supported_qkv_layout(tensor):
        return tensor
    return tensor.clone(memory_format=torch.contiguous_format)


def _validate_private_abi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> None:
    contiguous_tensors = (cumulative_gate, beta)
    if initial_state is not None:
        contiguous_tensors += (initial_state,)
    if (q.dtype, k.dtype, v.dtype) != (torch.bfloat16,) * 3:
        raise TypeError("the private chunk_kda ABI requires bfloat16 q, k, and v")
    if cumulative_gate.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("the private chunk_kda ABI requires float32 cumulative_gate and beta")
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError("the private chunk_kda ABI requires a float32 initial_state")
    if not all(tensor.is_contiguous() for tensor in contiguous_tensors):
        raise ValueError("the private chunk_kda ABI requires contiguous gate, beta, and state")
    if not all(_has_supported_qkv_layout(tensor) for tensor in (q, k, v)):
        raise ValueError(
            "the private chunk_kda ABI requires QKV to have contiguous heads and "
            "16-byte-aligned token rows"
        )
    if get_device_properties(q.device).major < 10:
        raise ValueError("the CuTe KDA core requires CUDA capability 10.0 or newer")


def _prepare_sequence_metadata(
    q: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> SequenceMetadata:
    """Distinguish direct, shape-packed, and caller-packed sequence layouts."""
    batch, tokens, heads, head_dim = q.shape
    output_shape = (batch, tokens, heads, head_dim)
    if cu_seqlens is not None:
        return SequenceMetadata(SequenceMode.PACKED, cu_seqlens, output_shape, output_shape)
    if batch == 1 and tokens % _CHUNK_SIZE == 0:
        return SequenceMetadata(SequenceMode.DENSE, None, output_shape, output_shape)

    packed_shape = (1, batch * tokens, heads, head_dim)
    generated_cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=q.device) * tokens
    return SequenceMetadata(
        SequenceMode.SHAPE_PACKED, generated_cu_seqlens, packed_shape, output_shape
    )


def _prepare_complete_chunk_metadata(
    q: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> ChunkMetadata:
    """Build dense or shape-aligned routing for the fixed-schedule operator."""
    tokens = q.shape[1]
    chunks = tokens // _CHUNK_SIZE
    if cu_seqlens is not None:
        chunk_indices, num_chunks = prepare_complete_chunk_metadata(
            cu_seqlens, tokens, _CHUNK_SIZE
        )
        return ChunkMetadata(cu_seqlens, chunk_indices, num_chunks)

    num_chunks = torch.full((), chunks, dtype=torch.int32, device=q.device)
    fixed_cu_seqlens = torch.arange(2, dtype=torch.int32, device=q.device) * tokens
    chunk_indices = torch.stack(
        (
            torch.zeros(chunks, dtype=torch.int32, device=q.device),
            torch.arange(chunks, dtype=torch.int32, device=q.device),
        ),
        dim=1,
    )
    return ChunkMetadata(fixed_cu_seqlens, chunk_indices, num_chunks)


def _chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    metadata: ChunkMetadata | RaggedChunkMetadata,
    *,
    output_final_state: bool,
    profile_ranges: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Run the optimized KDA core using an already selected chunk schedule."""
    scale = _HEAD_DIM**-0.5

    def record(name: str):
        return torch.profiler.record_function(name) if profile_ranges else nullcontext()

    with record("kda/fused/chunk_kda_fwd_intra"):
        w, u, kg, Aqk, Akk = chunk_kda_fwd_intra(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            scale,
            metadata,
            chunk_size=_CHUNK_SIZE,
            profile_ranges=profile_ranges,
        )
    # Dense B=1 stages derive their direct chunk routing from the input shape.
    boundary_metadata = (
        metadata
        if isinstance(metadata, RaggedChunkMetadata) or metadata.has_multiple_sequences
        else None
    )
    with record("kda/triton/inter_chunk_state"):
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            kg,
            w,
            u,
            cumulative_gate,
            initial_state,
            chunk_size=_CHUNK_SIZE,
            output_final_state=output_final_state,
            metadata=boundary_metadata,
        )
    with record("kda/triton/output_composition"):
        output = chunk_gla_fwd_o_gk(
            q,
            v_new,
            cumulative_gate,
            Aqk,
            h,
            scale,
            chunk_size=_CHUNK_SIZE,
            metadata=boundary_metadata,
        )
    return output, final_state, Aqk, Akk


# Fixed-arity schema pair instead of an optional final-state output. Tapes and packing
# metadata are op outputs only so the autograd.Function can save them.
_FWD_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor cumulative_gate, Tensor beta, "
    "Tensor? initial_state, Tensor? cu_seqlens, bool profile_ranges)"
)
torch.library.define(
    "attn_gym::kda_chunk_fwd",
    f"{_FWD_ARGS} -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_fwd_with_state",
    f"{_FWD_ARGS} -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)


def _chunk_kda_fwd_shared(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    output_final_state: bool,
    profile_ranges: bool,
):
    """Keep the complete composed forward behind one compiler-opaque boundary."""
    q, k, v = (_normalize_qkv_layout(tensor) for tensor in (q, k, v))
    _validate_private_abi(q, k, v, cumulative_gate, beta, initial_state)
    metadata = _prepare_complete_chunk_metadata(q, cu_seqlens)
    output, final_state, Aqk, Akk = _chunk_kda_fwd(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        metadata,
        output_final_state=output_final_state,
        profile_ranges=profile_ranges,
    )
    backward_cu_seqlens = (
        metadata.cu_seqlens if cu_seqlens is None else metadata.cu_seqlens.new_empty(0)
    )
    return (
        output,
        final_state,
        Aqk,
        Akk,
        backward_cu_seqlens,
        metadata.chunk_indices,
        metadata.num_chunks,
    )


def _chunk_kda_fwd_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    profile_ranges: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    output, _final_state, Aqk, Akk, cu_seqlens, chunk_indices, num_chunks = _chunk_kda_fwd_shared(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        cu_seqlens,
        False,
        profile_ranges,
    )
    return output, Aqk, Akk, cu_seqlens, chunk_indices, num_chunks


def _chunk_kda_fwd_with_state_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    profile_ranges: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    output, final_state, Aqk, Akk, cu_seqlens, chunk_indices, num_chunks = _chunk_kda_fwd_shared(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        cu_seqlens,
        True,
        profile_ranges,
    )
    return output, final_state, Aqk, Akk, cu_seqlens, chunk_indices, num_chunks


torch.library.impl("attn_gym::kda_chunk_fwd", "CUDA", _chunk_kda_fwd_cuda)
torch.library.impl("attn_gym::kda_chunk_fwd_with_state", "CUDA", _chunk_kda_fwd_with_state_cuda)


def _fwd_fake_common(q, v, cu_seqlens):
    """Describe the composed forward outputs shared by both schemas."""
    batch, tokens, heads, _head_dim = q.shape
    tape_shape = (batch, tokens, heads, _CHUNK_SIZE)
    chunks = tokens // _CHUNK_SIZE
    routing = q.new_empty((chunks, 2), dtype=torch.int32)
    return (
        v.new_empty(v.shape),
        q.new_empty(tape_shape),
        q.new_empty(tape_shape),
        q.new_empty((0 if cu_seqlens is not None else 2,), dtype=torch.int32),
        routing,
        q.new_empty((), dtype=torch.int32),
    )


@torch.library.register_fake("attn_gym::kda_chunk_fwd")
def _chunk_kda_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    profile_ranges: bool,
):
    del k, cumulative_gate, beta, initial_state, profile_ranges
    return _fwd_fake_common(q, v, cu_seqlens)


@torch.library.register_fake("attn_gym::kda_chunk_fwd_with_state")
def _chunk_kda_fwd_with_state_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    profile_ranges: bool,
):
    del k, cumulative_gate, beta, initial_state, profile_ranges
    batch, _tokens, heads, head_dim = q.shape
    state_batch = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    output, Aqk, Akk, cu, chunk_indices, num_chunks = _fwd_fake_common(q, v, cu_seqlens)
    state = q.new_empty((state_batch, heads, head_dim, v.shape[-1]), dtype=torch.float32)
    return output, state, Aqk, Akk, cu, chunk_indices, num_chunks


_RAGGED_FWD_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor cumulative_gate, Tensor beta, "
    "Tensor? initial_state, Tensor cu_seqlens, bool output_final_state, "
    "bool profile_ranges)"
)
torch.library.define(
    "attn_gym::kda_chunk_fwd_ragged",
    f"{_RAGGED_FWD_ARGS} -> (Tensor, Tensor)",
)


def _chunk_kda_fwd_ragged_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    output_final_state: bool,
    profile_ranges: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference-only ragged forward with a stable two-output schema."""
    q, k, v = (_normalize_qkv_layout(tensor) for tensor in (q, k, v))
    _validate_private_abi(q, k, v, cumulative_gate, beta, initial_state)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, q.shape[1], _CHUNK_SIZE)
    output, final_state, *_ = _chunk_kda_fwd(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        metadata,
        output_final_state=output_final_state,
        profile_ranges=profile_ranges,
    )
    state = final_state if final_state is not None else q.new_empty((0,), dtype=torch.float32)
    return output, state


torch.library.impl("attn_gym::kda_chunk_fwd_ragged", "CUDA", _chunk_kda_fwd_ragged_cuda)


@torch.library.register_fake("attn_gym::kda_chunk_fwd_ragged")
def _chunk_kda_fwd_ragged_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor,
    output_final_state: bool,
    profile_ranges: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe inference-only ragged forward outputs."""
    del k, cumulative_gate, beta, initial_state, profile_ranges
    state = (
        q.new_empty(
            (cu_seqlens.shape[0] - 1, q.shape[2], q.shape[3], v.shape[-1]),
            dtype=torch.float32,
        )
        if output_final_state
        else q.new_empty((0,), dtype=torch.float32)
    )
    return v.new_empty(v.shape), state


# Fixed-arity schema pair instead of an optional initial-state-gradient output.
_BWD_ARGS = (
    "(Tensor q, Tensor k, Tensor v, Tensor cumulative_gate, Tensor beta, Tensor Aqk, "
    "Tensor Akk, Tensor cu_seqlens, Tensor chunk_indices, Tensor num_chunks, "
    "Tensor? d_output, Tensor? d_final_state, {initial_state}, bool fastmath, "
    "bool profile_ranges)"
)
torch.library.define(
    "attn_gym::kda_chunk_bwd",
    _BWD_ARGS.format(initial_state="Tensor? initial_state")
    + " -> (Tensor, Tensor, Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::kda_chunk_bwd_with_state_grad",
    _BWD_ARGS.format(initial_state="Tensor initial_state")
    + " -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)


def _chunk_kda_bwd_shared(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    num_chunks: torch.Tensor,
    d_output: torch.Tensor | None,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    fastmath: bool,
    profile_ranges: bool,
):
    """Keep the complete first-order composed backward opaque to AOTAutograd."""
    from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd import chunk_kda_bwd

    q, k, v = (_normalize_qkv_layout(tensor) for tensor in (q, k, v))
    _validate_private_abi(q, k, v, cumulative_gate, beta, initial_state)
    if d_output is None:
        d_output = v.new_zeros(v.shape)
    return chunk_kda_bwd(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        Aqk,
        Akk,
        d_output.contiguous(),
        None if d_final_state is None else d_final_state.float().contiguous(),
        initial_state,
        ChunkMetadata(cu_seqlens, chunk_indices, num_chunks),
        fastmath=fastmath,
        profile_ranges=profile_ranges,
    )


def _chunk_kda_bwd_cuda(*args) -> tuple[torch.Tensor, ...]:
    dq, dk, dv, dg, db, _d_initial_state = _chunk_kda_bwd_shared(*args)
    return dq, dk, dv, dg, db


def _chunk_kda_bwd_with_state_grad_cuda(*args) -> tuple[torch.Tensor, ...]:
    return _chunk_kda_bwd_shared(*args)


torch.library.impl("attn_gym::kda_chunk_bwd", "CUDA", _chunk_kda_bwd_cuda)
torch.library.impl(
    "attn_gym::kda_chunk_bwd_with_state_grad", "CUDA", _chunk_kda_bwd_with_state_grad_cuda
)


def _bwd_fake_common(q, k, v, cumulative_gate, beta):
    """Describe backward output metadata without invoking a launcher."""
    return (
        q.new_empty(q.shape),
        k.new_empty(k.shape),
        v.new_empty(v.shape),
        cumulative_gate.new_empty(cumulative_gate.shape),
        beta.new_empty(beta.shape),
    )


@torch.library.register_fake("attn_gym::kda_chunk_bwd")
def _chunk_kda_bwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    *args,
):
    del args
    return _bwd_fake_common(q, k, v, cumulative_gate, beta)


@torch.library.register_fake("attn_gym::kda_chunk_bwd_with_state_grad")
def _chunk_kda_bwd_with_state_grad_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    num_chunks: torch.Tensor,
    d_output: torch.Tensor | None,
    d_final_state: torch.Tensor | None,
    initial_state: torch.Tensor,
    fastmath: bool,
    profile_ranges: bool,
):
    del Aqk, Akk, cu_seqlens, chunk_indices, num_chunks, d_output, d_final_state
    del fastmath, profile_ranges
    return (*_bwd_fake_common(q, k, v, cumulative_gate, beta), torch.empty_like(initial_state))


_chunk_kda_fwd_op = torch.ops.attn_gym.kda_chunk_fwd.default
_chunk_kda_fwd_with_state_op = torch.ops.attn_gym.kda_chunk_fwd_with_state.default
_chunk_kda_bwd_op = torch.ops.attn_gym.kda_chunk_bwd.default
_chunk_kda_bwd_with_state_grad_op = torch.ops.attn_gym.kda_chunk_bwd_with_state_grad.default


class _ChunkKDA(torch.autograd.Function):
    """First-order autograd wrapper around the opaque composed forward and backward ops.

    The forward returns ``output`` alone or ``(output, final_state)``; the autograd tapes
    and packing metadata are saved as intermediates rather than exposed as outputs.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        cu_seqlens,
        output_final_state,
        fastmath,
        profile_ranges,
    ):
        if output_final_state:
            output, state, Aqk, Akk, fwd_cu_seqlens, chunk_indices, num_chunks = (
                _chunk_kda_fwd_with_state_op(
                    q, k, v, cumulative_gate, beta, initial_state, cu_seqlens, profile_ranges
                )
            )
        else:
            output, Aqk, Akk, fwd_cu_seqlens, chunk_indices, num_chunks = _chunk_kda_fwd_op(
                q, k, v, cumulative_gate, beta, initial_state, cu_seqlens, profile_ranges
            )
        backward_cu_seqlens = fwd_cu_seqlens if cu_seqlens is None else cu_seqlens
        ctx.save_for_backward(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            Aqk,
            Akk,
            initial_state,
            backward_cu_seqlens,
            chunk_indices,
            num_chunks,
        )
        ctx.has_initial_state = initial_state is not None
        ctx.fastmath = fastmath
        ctx.profile_ranges = profile_ranges
        ctx.set_materialize_grads(False)
        if output_final_state:
            return output, state
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
            Aqk,
            Akk,
            initial_state,
            cu_seqlens,
            chunk_indices,
            num_chunks,
        ) = ctx.saved_tensors
        args = (
            q,
            k,
            v,
            cumulative_gate,
            beta,
            Aqk,
            Akk,
            cu_seqlens,
            chunk_indices,
            num_chunks,
            d_output,
            d_final_state,
            initial_state,
            ctx.fastmath,
            ctx.profile_ranges,
        )
        if ctx.has_initial_state:
            dq, dk, dv, dg, db, d_initial_state = _chunk_kda_bwd_with_state_grad_op(*args)
        else:
            dq, dk, dv, dg, db = _chunk_kda_bwd_op(*args)
            d_initial_state = None
        return (
            dq,
            dk,
            dv,
            dg,
            db,
            d_initial_state,
            None,
            None,
            None,
            None,
        )


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    cu_seqlens: torch.Tensor | None = None,
    output_final_state: bool = False,
    fastmath: bool = False,
    profile_ranges: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply the graph-capturable, first-order Blackwell KDA core.

    ``cu_seqlens`` selects packed ``[1, T, H, D]`` execution. Dense batches are
    lowered to the same packed representation with equal-length sequences. Every
    sequence must contain complete 64-token chunks. The optimized core computes Q/K/V in
    BF16 and gates, beta, and recurrent states in FP32. FP16 or FP32 Q/K/V inputs are cast
    to BF16 for the core, and the output is cast back to ``q.dtype``. Recurrent states remain
    FP32 and have one leading entry per logical sequence.
    """
    _validate_chunk_kda_inputs(q, k, v, cumulative_gate, beta, initial_state, cu_seqlens)
    output_dtype = q.dtype
    sequence_metadata = _prepare_sequence_metadata(q, cu_seqlens)
    q, k, v = (tensor.to(torch.bfloat16) for tensor in (q, k, v))
    cumulative_gate = cumulative_gate.float().contiguous()
    beta = beta.float().contiguous()
    if sequence_metadata.mode is SequenceMode.SHAPE_PACKED:
        q, k, v, cumulative_gate = (
            tensor.reshape(sequence_metadata.packed_shape) for tensor in (q, k, v, cumulative_gate)
        )
        beta = beta.reshape(sequence_metadata.packed_shape[:3])
    if initial_state is not None:
        initial_state = initial_state.float().contiguous()
    cu_seqlens = sequence_metadata.cu_seqlens
    if output_final_state:
        output, state = _ChunkKDA.apply(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            initial_state,
            cu_seqlens,
            True,
            fastmath,
            profile_ranges,
        )
    else:
        output = _ChunkKDA.apply(
            q,
            k,
            v,
            cumulative_gate,
            beta,
            initial_state,
            cu_seqlens,
            False,
            fastmath,
            profile_ranges,
        )
        state = None
    return output.reshape(sequence_metadata.output_shape).to(output_dtype), state


__all__ = ["chunk_kda"]
