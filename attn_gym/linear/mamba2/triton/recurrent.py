"""Triton token-by-token Mamba2 scan for inference prefill and decode.

One program owns one ``(sequence, head, P tile)`` and keeps its FP32 ``[BP, N]`` state in
registers while it consumes the sequence one token at a time:

    state = exp(dt * A) * state + dt * outer(x, B)
    y     = (state @ C + D * x) * silu(z)

The state comes from an explicit initial state, a zero state, or a paged pool selected by
``state_indices`` (see :mod:`attn_gym.linear._delta_rule.triton.paged_state`). Inputs are
addressed through ``(token, head)`` strides, so one-token decode reads strided views of a packed
projection buffer without copies. Forward-only; training uses the chunked forms.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import LOG2_E, configure_triton_allocator
from attn_gym.linear._delta_rule.triton.paged_state import resolve_paged_state
from attn_gym.linear.mamba2.triton.chunk_fwd import has_dt_limit, transform_dt

configure_triton_allocator()


@triton.jit(do_not_specialize=["T"])
def mamba2_recurrent_kernel(
    x,
    dt,
    A,
    dt_bias,
    B,
    C,
    D,
    z,
    y,
    state_in,
    state_out,
    state_indices,
    has_initial_state,
    cu_seqlens,
    T,
    x_stride_t,
    x_stride_h,
    dt_stride_t,
    b_stride_t,
    b_stride_g,
    c_stride_t,
    c_stride_g,
    z_stride_t,
    z_stride_h,
    y_stride_t,
    y_stride_h,
    state_stride_slot,
    H: tl.constexpr,
    G: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    BP: tl.constexpr,
    BN: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    DT_MIN: tl.constexpr,
    DT_MAX: tl.constexpr,
    HAS_DT_LIMIT: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    D_PER_CHANNEL: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    LOAD_STATE: tl.constexpr,
    STORE_STATE: tl.constexpr,
    USE_STATE_INDICES: tl.constexpr,
    USE_HAS_INITIAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    sequence_head, p_tile = tl.program_id(0), tl.program_id(1)
    sequence, head = sequence_head // H, sequence_head % H
    group = head // (H // G)
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + sequence).to(tl.int64)
        length = (tl.load(cu_seqlens + sequence + 1) - tl.load(cu_seqlens + sequence)).to(tl.int32)
    else:
        bos = sequence.to(tl.int64) * T
        length = T

    offs_p = p_tile * BP + tl.arange(0, BP)
    offs_n = tl.arange(0, BN)
    mask_p = offs_p < P
    mask_n = offs_n < N
    mask_pn = mask_p[:, None] & mask_n[None, :]

    slot, active, load_initial = resolve_paged_state(
        sequence, state_indices, has_initial_state, USE_STATE_INDICES, USE_HAS_INITIAL_STATE
    )
    if USE_STATE_INDICES and not active:
        # Null routes produce zero output and never touch the pool.
        for t in range(length):
            token = bos + t
            tl.store(y + token * y_stride_t + head * y_stride_h + offs_p, 0.0, mask=mask_p)
        return

    state_ptrs = (
        slot.to(tl.int64) * state_stride_slot
        + head * (P * N)
        + offs_p[:, None] * N
        + offs_n[None, :]
    )
    if LOAD_STATE:
        state = tl.load(state_in + state_ptrs, mask=mask_pn & load_initial, other=0.0)
        state = state.to(tl.float32)
    else:
        state = tl.zeros([BP, BN], dtype=tl.float32)

    decay_rate = tl.load(A + head).to(tl.float32) * LOG2_E
    bias = tl.load(dt_bias + head).to(tl.float32) if HAS_DT_BIAS else 0.0
    if HAS_D:
        if D_PER_CHANNEL:
            skip = tl.load(D + head * P + offs_p, mask=mask_p, other=0.0).to(tl.float32)
        else:
            skip = tl.load(D + head).to(tl.float32)

    for t in range(length):
        token = bos + t
        x_t = tl.load(x + token * x_stride_t + head * x_stride_h + offs_p, mask=mask_p, other=0.0)
        x_t = x_t.to(tl.float32)
        step = tl.load(dt + token * dt_stride_t + head).to(tl.float32)
        step = transform_dt(step, bias, DT_SOFTPLUS, DT_MIN, DT_MAX, HAS_DT_LIMIT)
        b = token * b_stride_t + group * b_stride_g + offs_n
        b_t = tl.load(B + b, mask=mask_n, other=0.0).to(tl.float32)
        c = token * c_stride_t + group * c_stride_g + offs_n
        c_t = tl.load(C + c, mask=mask_n, other=0.0).to(tl.float32)
        state = state * tl.math.exp2(step * decay_rate) + (x_t * step)[:, None] * b_t[None, :]
        y_t = tl.sum(state * c_t[None, :], axis=1)
        if HAS_D:
            y_t += x_t * skip
        if HAS_Z:
            gate = tl.load(z + token * z_stride_t + head * z_stride_h + offs_p, mask=mask_p)
            gate = gate.to(tl.float32)
            y_t *= gate * tl.sigmoid(gate)
        tl.store(
            y + token * y_stride_t + head * y_stride_h + offs_p,
            y_t.to(y.dtype.element_ty),
            mask=mask_p,
        )

    if STORE_STATE:
        tl.store(state_out + state_ptrs, state.to(state_out.dtype.element_ty), mask=mask_pn)


def _launch_config(head_dim: int, state_dim: int, programs: int) -> tuple[int, int]:
    """P tile and warp count: one-token decode wants many small programs."""
    block_n = triton.next_power_of_2(state_dim)
    block_p = min(triton.next_power_of_2(head_dim), 16 if programs < 1024 else 32)
    num_warps = 4 if block_p * block_n >= 2048 else 2
    return block_p, num_warps


def _mergeable(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """A view whose batch and token axes merge into one token axis, copying only if needed."""
    if tensor is None:
        return None
    batch, tokens = tensor.shape[:2]
    if tensor.stride(-1) != 1 or (
        batch > 1 and tokens > 1 and tensor.stride(0) != tokens * tensor.stride(1)
    ):
        return tensor.contiguous()
    return tensor


def launch_recurrent(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    y: torch.Tensor,
    *,
    dt_bias: torch.Tensor | None,
    dt_softplus: bool,
    dt_min: float,
    dt_max: float,
    cu_seqlens: torch.Tensor | None,
    state_in: torch.Tensor | None,
    state_out: torch.Tensor | None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
) -> None:
    """Launch the scan over ``[batch, T, H, P]``-shaped views with flattened token strides.

    Token-major inputs keep their head and feature strides when their batch and token axes
    merge into one token axis (dense batches are contiguous across them, and decode passes
    ``T == 1``) and their last axis is contiguous; other inputs are copied.
    """
    batch, tokens, heads, head_dim = x.shape
    groups, state_dim = B.shape[2:]
    num_sequences = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    state = state_in if state_in is not None else state_out
    block_p, num_warps = _launch_config(head_dim, state_dim, num_sequences * heads)
    x, dt, B, C, z = (_mergeable(tensor) for tensor in (x, dt, B, C, z))
    A, D, dt_bias = (None if tensor is None else tensor.contiguous() for tensor in (A, D, dt_bias))

    def token_stride(tensor: torch.Tensor) -> int:
        return tensor.stride(1) if tokens > 1 else tensor.stride(0)

    grid = (num_sequences * heads, triton.cdiv(head_dim, block_p))
    mamba2_recurrent_kernel[grid](
        x,
        dt,
        A,
        dt_bias,
        B,
        C,
        D,
        z,
        y,
        state_in,
        state_out,
        state_indices,
        has_initial_state,
        cu_seqlens,
        tokens,
        token_stride(x),
        x.stride(2),
        token_stride(dt),
        token_stride(B),
        B.stride(2),
        token_stride(C),
        C.stride(2),
        0 if z is None else token_stride(z),
        0 if z is None else z.stride(2),
        token_stride(y),
        y.stride(2),
        0 if state is None else state.stride(0),
        H=heads,
        G=groups,
        P=head_dim,
        N=state_dim,
        BP=block_p,
        BN=triton.next_power_of_2(state_dim),
        DT_SOFTPLUS=dt_softplus,
        DT_MIN=float(dt_min),
        DT_MAX=float(dt_max),
        HAS_DT_LIMIT=has_dt_limit(dt_min, dt_max),
        HAS_DT_BIAS=dt_bias is not None,
        D_PER_CHANNEL=D is not None and D.ndim == 2,
        HAS_D=D is not None,
        HAS_Z=z is not None,
        LOAD_STATE=state_in is not None,
        STORE_STATE=state_out is not None,
        USE_STATE_INDICES=state_indices is not None,
        USE_HAS_INITIAL_STATE=has_initial_state is not None,
        IS_VARLEN=cu_seqlens is not None,
        num_warps=num_warps,
    )


def mamba2_recurrent_fwd(
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
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Scan ``[batch, T, H, P]`` inputs; return the output and optionally the FP32 final state."""
    batch, _, heads, head_dim = x.shape
    num_sequences = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    y = x.new_empty(x.shape)
    final_state = (
        x.new_empty(num_sequences, heads, head_dim, B.shape[-1], dtype=torch.float32)
        if output_final_state
        else None
    )
    launch_recurrent(
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        y,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_min=dt_min,
        dt_max=dt_max,
        cu_seqlens=cu_seqlens,
        state_in=None if initial_state is None else initial_state.contiguous(),
        state_out=final_state,
    )
    return y, final_state


def mamba2_recurrent_fwd_paged(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    dt_softplus: bool,
    dt_min: float,
    dt_max: float,
    y: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scan inputs while advancing the selected ``state_cache`` slots in place."""
    y = x.new_empty(x.shape) if y is None else y
    launch_recurrent(
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        y,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_min=dt_min,
        dt_max=dt_max,
        cu_seqlens=cu_seqlens,
        state_in=state_cache,
        state_out=state_cache,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
    )
    return y


__all__ = [
    "launch_recurrent",
    "mamba2_recurrent_fwd",
    "mamba2_recurrent_fwd_paged",
    "mamba2_recurrent_kernel",
]
