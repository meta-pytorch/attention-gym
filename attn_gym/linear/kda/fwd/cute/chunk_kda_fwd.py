"""Composed fixed-length Blackwell KDA core forward."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from attn_gym.linear.kda.fwd.cute.chunk_kda_fwd_intra import chunk_kda_fwd_intra
from attn_gym.linear.kda.fwd.triton.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from attn_gym.linear.kda.fwd.triton.chunk_gla_fwd_o import chunk_gla_fwd_o_gk

_SUPPORTED_INPUT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
# TODO: Revisit model-approved chunk sizes: this is a major performance lever,
# but it changes the KDA decomposition and rounding order, so it can affect numerics.
_CHUNK_SIZE = 64
_HEAD_DIM = 128


def _validate_chunk_kda_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
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
    expected_state = (batch, heads, head_dim, v.shape[-1])
    if initial_state is not None and initial_state.shape != expected_state:
        raise ValueError(
            f"initial_state must have shape {expected_state}, got {tuple(initial_state.shape)}"
        )
    tensors = (q, k, v, cumulative_gate, beta)
    if initial_state is not None:
        tensors += (initial_state,)
    if not all(tensor.is_cuda and tensor.device == q.device for tensor in tensors):
        raise ValueError("all chunk_kda inputs must be CUDA tensors on the same device")
    if any(tensor.dtype not in _SUPPORTED_INPUT_DTYPES for tensor in tensors):
        supported = ", ".join(str(dtype) for dtype in _SUPPORTED_INPUT_DTYPES)
        raise TypeError(f"chunk_kda inputs must use one of {supported}")
    if batch != 1 or head_dim != _HEAD_DIM:
        raise ValueError("the CuTe KDA core requires B=1 and K=V=128")
    if tokens % _CHUNK_SIZE:
        raise ValueError("the CuTe KDA core requires complete 64-token chunks")
    if not torch.compiler.is_compiling() and torch.cuda.get_device_capability(q.device) < (10, 0):
        raise ValueError("the CuTe KDA core requires CUDA capability 10.0 or newer")


def _validate_private_abi(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> None:
    tensors = (q, k, v, cumulative_gate, beta)
    if initial_state is not None:
        tensors += (initial_state,)
    if (q.dtype, k.dtype, v.dtype) != (torch.bfloat16,) * 3:
        raise TypeError("the private chunk_kda ABI requires bfloat16 q, k, and v")
    if cumulative_gate.dtype != torch.float32 or beta.dtype != torch.float32:
        raise TypeError("the private chunk_kda ABI requires float32 cumulative_gate and beta")
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise TypeError("the private chunk_kda ABI requires a float32 initial_state")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("the private chunk_kda ABI requires contiguous inputs")
    if torch.cuda.get_device_capability(q.device) < (10, 0):
        raise ValueError("the CuTe KDA core requires CUDA capability 10.0 or newer")


def _fixed_chunk_metadata(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the fixed-length metadata required by legacy recompute kernels."""
    tokens = q.shape[1]
    chunks = tokens // _CHUNK_SIZE
    cu_seqlens = torch.arange(2, dtype=torch.int32, device=q.device) * tokens
    chunk_indices = torch.stack(
        (
            torch.zeros(chunks, dtype=torch.int32, device=q.device),
            torch.arange(chunks, dtype=torch.int32, device=q.device),
        ),
        dim=1,
    )
    return cu_seqlens, chunk_indices, torch.full((), chunks, dtype=torch.int32, device=q.device)


def _chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    output_final_state: bool,
    profile_ranges: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Run the optimized fixed-length KDA core and return its minimal backward tape."""
    cu_seqlens, chunk_indices, num_chunks = _fixed_chunk_metadata(q)
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
            cu_seqlens,
            chunk_indices,
            num_chunks,
            chunk_size=_CHUNK_SIZE,
            profile_ranges=profile_ranges,
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
        )
    return output, final_state, Aqk, Akk, cu_seqlens, chunk_indices, num_chunks


@torch.library.custom_op("attn_gym::kda_chunk_fwd", mutates_args=())
def _chunk_kda_fwd_custom_op(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    fastmath: bool,
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
    """Keep the complete composed forward behind one compiler-opaque boundary."""
    del fastmath  # This static option configures the registered backward.
    _validate_chunk_kda_inputs(q, k, v, cumulative_gate, beta, initial_state)
    _validate_private_abi(q, k, v, cumulative_gate, beta, initial_state)
    output, final_state, Aqk, Akk, cu_seqlens, chunk_indices, num_chunks = _chunk_kda_fwd(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        output_final_state=output_final_state,
        profile_ranges=profile_ranges,
    )
    state = final_state if final_state is not None else q.new_empty((0,), dtype=torch.float32)
    return output, state, Aqk, Akk, cu_seqlens, chunk_indices, num_chunks


@_chunk_kda_fwd_custom_op.register_fake
def _chunk_kda_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    fastmath: bool,
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
    """Describe the composed forward outputs without invoking a launcher."""
    del k, cumulative_gate, beta, initial_state, fastmath, profile_ranges
    batch, tokens, heads, head_dim = q.shape
    state = (
        q.new_empty((batch, heads, head_dim, v.shape[-1]), dtype=torch.float32)
        if output_final_state
        else q.new_empty((0,), dtype=torch.float32)
    )
    tape_shape = (batch, tokens, heads, _CHUNK_SIZE)
    chunks = tokens // _CHUNK_SIZE
    return (
        torch.empty_like(v),
        state,
        q.new_empty(tape_shape),
        q.new_empty(tape_shape),
        q.new_empty((2,), dtype=torch.int32),
        q.new_empty((chunks, 2), dtype=torch.int32),
        q.new_empty((), dtype=torch.int32),
    )


@torch.library.custom_op("attn_gym::kda_chunk_bwd", mutates_args=())
def _chunk_kda_bwd_custom_op(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the complete first-order composed backward opaque to AOTAutograd."""
    from attn_gym.linear.kda.bwd.cute.chunk_kda_bwd import chunk_kda_bwd

    if d_output is None:
        d_output = torch.zeros_like(v)
    dq, dk, dv, dg, db, d_initial_state = chunk_kda_bwd(
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
        cu_seqlens,
        chunk_indices,
        num_chunks,
        fastmath=fastmath,
        profile_ranges=profile_ranges,
    )
    initial_state_gradient = (
        d_initial_state if d_initial_state is not None else q.new_empty((0,), dtype=torch.float32)
    )
    return dq, dk, dv, dg, db, initial_state_gradient


@_chunk_kda_bwd_custom_op.register_fake
def _chunk_kda_bwd_fake(
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe backward output metadata without invoking a launcher."""
    del (
        Aqk,
        Akk,
        cu_seqlens,
        chunk_indices,
        num_chunks,
        d_output,
        d_final_state,
        fastmath,
        profile_ranges,
    )
    d_initial_state = (
        torch.empty_like(initial_state)
        if initial_state is not None
        else q.new_empty((0,), dtype=torch.float32)
    )
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(cumulative_gate),
        torch.empty_like(beta),
        d_initial_state,
    )


def _setup_chunk_kda_context(ctx, inputs, output) -> None:
    (
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        output_final_state,
        fastmath,
        profile_ranges,
    ) = inputs
    _output, state, Aqk, Akk, cu_seqlens, chunk_indices, num_chunks = output
    ctx.save_for_backward(
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
    )
    ctx.has_initial_state = initial_state is not None
    ctx.fastmath = fastmath
    ctx.profile_ranges = profile_ranges
    ctx.set_materialize_grads(False)
    ctx.mark_non_differentiable(Aqk, Akk)
    if not output_final_state:
        ctx.mark_non_differentiable(state)


@torch.autograd.function.once_differentiable
def _chunk_kda_backward(
    ctx,
    d_output,
    d_final_state,
    _dAqk,
    _dAkk,
    _d_cu_seqlens,
    _d_chunk_indices,
    _d_num_chunks,
):
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
    dq, dk, dv, dg, db, d_initial_state = _chunk_kda_bwd_custom_op(
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
    return (
        dq,
        dk,
        dv,
        dg,
        db,
        d_initial_state if ctx.has_initial_state else None,
        None,
        None,
        None,
    )


_chunk_kda_fwd_custom_op.register_autograd(
    _chunk_kda_backward,
    setup_context=_setup_chunk_kda_context,
)


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cumulative_gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    *,
    output_final_state: bool = False,
    fastmath: bool = False,
    profile_ranges: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply the graph-capturable, first-order, fixed-length Blackwell KDA core.

    The output uses ``q.dtype``. The recurrent state remains FP32 so streamed
    execution does not repeatedly quantize the accumulated state.
    """
    _validate_chunk_kda_inputs(q, k, v, cumulative_gate, beta, initial_state)
    output_dtype = q.dtype
    q, k, v = (tensor.to(torch.bfloat16).contiguous() for tensor in (q, k, v))
    cumulative_gate = cumulative_gate.float().contiguous()
    beta = beta.float().contiguous()
    if initial_state is not None:
        initial_state = initial_state.float().contiguous()
    output, state, _Aqk, _Akk, _cu_seqlens, _chunk_indices, _num_chunks = _chunk_kda_fwd_custom_op(
        q,
        k,
        v,
        cumulative_gate,
        beta,
        initial_state,
        output_final_state,
        fastmath,
        profile_ranges,
    )
    return output.to(output_dtype), state if output_final_state else None


__all__ = ["chunk_kda"]
