"""Registered operators for the fused Mamba2 SSD scans; autograd for the chunked form."""

from __future__ import annotations

import importlib
from types import ModuleType

import torch

_INPUT_ARGS = (
    "Tensor x, Tensor dt, Tensor A, Tensor B, Tensor C, Tensor? D, Tensor? z, "
    "Tensor? initial_state, Tensor? dt_bias, Tensor? cu_seqlens"
)
_TRANSFORM_ARGS = "bool dt_softplus, float dt_min, float dt_max"
# The last output is the gate-backward residual (pre-gate output); empty unless stored.
torch.library.define(
    "attn_gym::mamba2_chunk_fwd",
    f"({_INPUT_ARGS}, {_TRANSFORM_ARGS}, bool store_pre_gate) -> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::mamba2_chunk_fwd_with_state",
    f"({_INPUT_ARGS}, {_TRANSFORM_ARGS}, bool store_pre_gate) -> (Tensor, Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::mamba2_recurrent_fwd",
    f"({_INPUT_ARGS}, {_TRANSFORM_ARGS}) -> (Tensor, Tensor)",
)
torch.library.define(
    "attn_gym::mamba2_recurrent_fwd_no_state", f"({_INPUT_ARGS}, {_TRANSFORM_ARGS}) -> Tensor"
)
_PAGED_ARGS = (
    "Tensor x, Tensor dt, Tensor A, Tensor B, Tensor C, Tensor? D, Tensor? z, "
    "Tensor(a!) state_cache, Tensor state_indices, Tensor? has_initial_state, Tensor? dt_bias"
)
torch.library.define(
    "attn_gym::mamba2_recurrent_fwd_paged",
    f"({_PAGED_ARGS}, Tensor? cu_seqlens, {_TRANSFORM_ARGS}) -> Tensor",
)
torch.library.define(
    "attn_gym::mamba2_chunk_fwd_paged",
    f"({_PAGED_ARGS}, Tensor? cu_seqlens, {_TRANSFORM_ARGS}) -> Tensor",
)
torch.library.define(
    "attn_gym::mamba2_decode",
    f"({_PAGED_ARGS}, {_TRANSFORM_ARGS}, Tensor(b!) out) -> ()",
)
# Gradients of absent optional inputs are empty tensors, keeping the output arity fixed.
torch.library.define(
    "attn_gym::mamba2_chunk_bwd",
    f"({_INPUT_ARGS}, Tensor? pre_gate, Tensor d_output, Tensor? d_final_state, "
    f"{_TRANSFORM_ARGS}) -> "
    "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)",
)


def _triton_module(name: str) -> ModuleType:
    # Note: Lazy Imports (see attn_gym/linear/__init__.py)
    try:
        return importlib.import_module(f"attn_gym.linear.mamba2.triton.{name}")
    except ImportError as error:
        raise ImportError("the fused Mamba2 kernels require CUDA with Triton support") from error


def _chunk_fwd_outputs(*args, output_final_state: bool) -> tuple[torch.Tensor, ...]:
    *inputs, store_pre_gate = args
    output, final_state, pre_gate = _triton_module("chunk_fwd").mamba2_chunk_fwd(
        *inputs, output_final_state=output_final_state, store_pre_gate=store_pre_gate
    )
    pre_gate = output.new_empty(0) if pre_gate is None else pre_gate
    return (output, pre_gate) if final_state is None else (output, final_state, pre_gate)


def _chunk_fwd_cuda(*args) -> tuple[torch.Tensor, torch.Tensor]:
    return _chunk_fwd_outputs(*args, output_final_state=False)


def _chunk_fwd_with_state_cuda(*args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _chunk_fwd_outputs(*args, output_final_state=True)


def _chunk_bwd_cuda(*args) -> tuple[torch.Tensor, ...]:
    return _triton_module("chunk_bwd").mamba2_chunk_bwd(*args)


def _recurrent_fwd_cuda(*args) -> tuple[torch.Tensor, torch.Tensor]:
    return _triton_module("recurrent").mamba2_recurrent_fwd(*args, output_final_state=True)


def _recurrent_fwd_no_state_cuda(*args) -> torch.Tensor:
    return _triton_module("recurrent").mamba2_recurrent_fwd(*args, output_final_state=False)[0]


def _reject_pool_aliasing(state_cache: torch.Tensor, *inputs: torch.Tensor | None) -> None:
    # The opaque kernels must honor the schema: only state_cache is written.
    if any(tensor is not None and torch._C._overlaps(state_cache, tensor) for tensor in inputs):
        raise ValueError("state_cache must not alias any read-only Mamba2 input")


def _recurrent_fwd_paged_cuda(*args) -> torch.Tensor:
    _reject_pool_aliasing(args[7], *args[:7], *args[8:12])
    return _triton_module("recurrent").mamba2_recurrent_fwd_paged(*args)


def _chunk_fwd_paged_cuda(*args) -> torch.Tensor:
    _reject_pool_aliasing(args[7], *args[:7], *args[8:12])
    return _triton_module("chunk_fwd").mamba2_paged_chunk_fwd(*args)


def _decode_cuda(
    x,
    dt,
    A,
    B,
    C,
    D,
    z,
    state_cache,
    state_indices,
    has_initial_state,
    dt_bias,
    dt_softplus,
    dt_min,
    dt_max,
    out,
) -> None:
    # The opaque kernel must honor the schema: only state_cache and out are written.
    read_only = tuple(
        tensor
        for tensor in (x, dt, A, B, C, D, z, state_indices, has_initial_state, dt_bias)
        if tensor is not None
    )
    if any(torch._C._overlaps(out, tensor) for tensor in (*read_only, state_cache)):
        raise ValueError("out must not alias any recurrent_mamba2_decode input")
    if any(torch._C._overlaps(state_cache, tensor) for tensor in read_only):
        raise ValueError("state_cache must not alias recurrent_mamba2_decode read-only inputs")
    # One token per sequence: a dense batch of length-one sequences.
    _triton_module("recurrent").mamba2_recurrent_fwd_paged(
        x.unsqueeze(1),
        dt.unsqueeze(1),
        A,
        B.unsqueeze(1),
        C.unsqueeze(1),
        D,
        None if z is None else z.unsqueeze(1),
        state_cache,
        state_indices,
        has_initial_state,
        dt_bias,
        None,
        dt_softplus,
        dt_min,
        dt_max,
        y=out.unsqueeze(1),
    )


torch.library.impl("attn_gym::mamba2_chunk_fwd", "CUDA", _chunk_fwd_cuda)
torch.library.impl("attn_gym::mamba2_chunk_fwd_with_state", "CUDA", _chunk_fwd_with_state_cuda)
torch.library.impl("attn_gym::mamba2_chunk_bwd", "CUDA", _chunk_bwd_cuda)
torch.library.impl("attn_gym::mamba2_recurrent_fwd", "CUDA", _recurrent_fwd_cuda)
torch.library.impl("attn_gym::mamba2_recurrent_fwd_no_state", "CUDA", _recurrent_fwd_no_state_cuda)
torch.library.impl("attn_gym::mamba2_recurrent_fwd_paged", "CUDA", _recurrent_fwd_paged_cuda)
torch.library.impl("attn_gym::mamba2_chunk_fwd_paged", "CUDA", _chunk_fwd_paged_cuda)
torch.library.impl("attn_gym::mamba2_decode", "CUDA", _decode_cuda)


def _final_state_fake(
    x: torch.Tensor, B: torch.Tensor, cu_seqlens: torch.Tensor | None
) -> torch.Tensor:
    num_states = x.shape[0] if cu_seqlens is None else cu_seqlens.shape[0] - 1
    return x.new_empty(num_states, *x.shape[2:], B.shape[-1], dtype=torch.float32)


@torch.library.register_fake("attn_gym::mamba2_recurrent_fwd")
def _recurrent_fwd_fake(x, dt, A, B, C, D, z, initial_state, dt_bias, cu_seqlens, *transform):
    return x.new_empty(x.shape), _final_state_fake(x, B, cu_seqlens)


@torch.library.register_fake("attn_gym::mamba2_recurrent_fwd_no_state")
@torch.library.register_fake("attn_gym::mamba2_recurrent_fwd_paged")
@torch.library.register_fake("attn_gym::mamba2_chunk_fwd_paged")
def _output_fake(x, *args):
    return x.new_empty(x.shape)


@torch.library.register_fake("attn_gym::mamba2_decode")
def _decode_fake(*args) -> None:
    return None


@torch.library.register_fake("attn_gym::mamba2_chunk_fwd")
def _chunk_fwd_fake(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    dt_softplus: bool,
    dt_min: float,
    dt_max: float,
    store_pre_gate: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return x.new_empty(x.shape), _pre_gate_fake(x, z, store_pre_gate)


def _pre_gate_fake(x: torch.Tensor, z: torch.Tensor | None, store: bool) -> torch.Tensor:
    if store and z is not None:
        return x.new_empty(x.shape, dtype=torch.float32)
    return x.new_empty(0)


@torch.library.register_fake("attn_gym::mamba2_chunk_fwd_with_state")
def _chunk_fwd_with_state_fake(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    dt_softplus: bool,
    dt_min: float,
    dt_max: float,
    store_pre_gate: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    final_state = _final_state_fake(x, B, cu_seqlens)
    return x.new_empty(x.shape), final_state, _pre_gate_fake(x, z, store_pre_gate)


@torch.library.register_fake("attn_gym::mamba2_chunk_bwd")
def _chunk_bwd_fake(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    pre_gate: torch.Tensor | None,
    d_output: torch.Tensor,
    d_final_state: torch.Tensor | None,
    dt_softplus: bool,
    dt_min: float,
    dt_max: float,
) -> tuple[torch.Tensor, ...]:
    empty = x.new_empty(0)
    return (
        *(tensor.new_empty(tensor.shape) for tensor in (x, dt, A, B, C)),
        *(
            empty if tensor is None else tensor.new_empty(tensor.shape)
            for tensor in (D, z, dt_bias, initial_state)
        ),
    )


chunk_fwd_op = torch.ops.attn_gym.mamba2_chunk_fwd.default
chunk_fwd_with_state_op = torch.ops.attn_gym.mamba2_chunk_fwd_with_state.default
chunk_bwd_op = torch.ops.attn_gym.mamba2_chunk_bwd.default


class ChunkMamba2(torch.autograd.Function):
    """First-order autograd over the fused forward and backward operators."""

    @staticmethod
    def forward(
        ctx,
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        initial_state,
        dt_bias,
        cu_seqlens,
        dt_softplus,
        dt_min,
        dt_max,
        output_final_state,
        store_pre_gate,
    ):
        args = (x, dt, A, B, C, D, z, initial_state, dt_bias, cu_seqlens)
        transform = (dt_softplus, dt_min, dt_max)
        ctx.transform = transform
        ctx.set_materialize_grads(False)
        if output_final_state:
            output, final_state, pre_gate = chunk_fwd_with_state_op(
                *args, *transform, store_pre_gate
            )
            outputs = (output, final_state)
        else:
            output, pre_gate = chunk_fwd_op(*args, *transform, store_pre_gate)
            outputs = output
        ctx.save_for_backward(*args, pre_gate if store_pre_gate else None)
        return outputs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, d_output, d_final_state=None):
        *args, pre_gate = ctx.saved_tensors
        x, D, z, initial_state, dt_bias = args[0], *args[5:9]
        if d_output is None:
            d_output = torch.zeros_like(x)
        dx, ddt, dA, dB, dC, dD, dz, ddt_bias, d_initial_state = chunk_bwd_op(
            *args, pre_gate, d_output, d_final_state, *ctx.transform
        )
        return (
            dx,
            ddt,
            dA,
            dB,
            dC,
            None if D is None else dD,
            None if z is None else dz,
            None if initial_state is None else d_initial_state,
            None if dt_bias is None else ddt_bias,
            *(None,) * 6,
        )


def chunk_forward(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    *,
    dt_bias: torch.Tensor | None,
    dt_softplus: bool,
    dt_limit: tuple[float, float],
    cu_seqlens: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the differentiable fused chunk scan on validated inputs."""
    dt_min, dt_max = dt_limit
    inputs = (x, dt, A, B, C, D, z, initial_state, dt_bias)
    # The gate backward needs the pre-gate output; store it only when a backward can follow.
    store_pre_gate = (
        z is not None
        and torch.is_grad_enabled()
        and any(tensor is not None and tensor.requires_grad for tensor in inputs)
    )
    result = ChunkMamba2.apply(
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        initial_state,
        dt_bias,
        cu_seqlens,
        dt_softplus,
        float(dt_min),
        float(dt_max),
        output_final_state,
        store_pre_gate,
    )
    return result if output_final_state else (result, None)


recurrent_fwd_op = torch.ops.attn_gym.mamba2_recurrent_fwd.default
recurrent_fwd_no_state_op = torch.ops.attn_gym.mamba2_recurrent_fwd_no_state.default
recurrent_fwd_paged_op = torch.ops.attn_gym.mamba2_recurrent_fwd_paged.default
chunk_fwd_paged_op = torch.ops.attn_gym.mamba2_chunk_fwd_paged.default
decode_op = torch.ops.attn_gym.mamba2_decode.default


__all__ = [
    "chunk_bwd_op",
    "chunk_forward",
    "chunk_fwd_op",
    "chunk_fwd_paged_op",
    "chunk_fwd_with_state_op",
    "decode_op",
    "recurrent_fwd_no_state_op",
    "recurrent_fwd_op",
    "recurrent_fwd_paged_op",
]
