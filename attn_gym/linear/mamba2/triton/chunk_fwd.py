"""Single-pass Triton forward for the chunked Mamba2 SSD scan.

One program owns one ``(sequence, head, P tile)`` and keeps its FP32 ``[BP, N]`` state in
registers while it walks the sequence in ``BT`` token chunks. Each chunk computes the masked
``C B^T`` intra-chunk product, reads the entry state, applies the ``D`` skip and ``silu(z)`` gate,
and advances the state, so no per-chunk state or score tensor reaches global memory.

The backward reruns the same kernel with ``STORE_OUTPUT=False`` and ``STORE_CHUNK_STATES=True``
to write every chunk's entry state, rounded to the input dtype exactly as the output MMA reads
it, into its chunk slot (see :func:`chunk_slot_base`).
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.tune import TritonTuner
from attn_gym._backends.triton.utils import LOG2_E
from attn_gym.linear._delta_rule.triton.paged_state import resolve_paged_state

CHUNK_SIZE = 64


@triton.jit
def transform_dt(
    dt_raw,
    bias,
    DT_SOFTPLUS: tl.constexpr,
    DT_MIN: tl.constexpr,
    DT_MAX: tl.constexpr,
    HAS_DT_LIMIT: tl.constexpr,
):
    """``clamp(softplus(dt_raw + bias), DT_MIN, DT_MAX)`` with each stage optional."""
    dt = dt_raw + bias
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)
    if HAS_DT_LIMIT:
        dt = tl.minimum(tl.maximum(dt, DT_MIN), DT_MAX)
    return dt


@triton.jit
def chunk_slot_base(cu_seqlens, sequence, bos, T, BT: tl.constexpr, IS_VARLEN: tl.constexpr):
    """First chunk slot of ``sequence``; its chunk ``k`` lives in slot ``base + k``.

    Dense sequences own ``cdiv(T, BT)`` slots each. Packed sequence ``s`` starts at
    ``cu_seqlens[s] // BT + s``: the slots stay disjoint because ``cdiv(length, BT)`` never
    exceeds ``(bos + length) // BT - bos // BT + 1``, and all fit in ``T // BT + num_sequences``.
    """
    if IS_VARLEN:
        return bos // BT + sequence
    return sequence.to(tl.int64) * tl.cdiv(T, BT)


@triton.jit(do_not_specialize=["T"])
def mamba2_chunk_fwd_kernel(
    x,
    dt,
    A,
    dt_bias,
    B,
    C,
    D,
    z,
    y,
    initial_state,
    final_state,
    chunk_states,
    state_indices,
    has_initial_state,
    cu_seqlens,
    T,
    state_stride_slot,
    H: tl.constexpr,
    G: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    BT: tl.constexpr,
    BP: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    DT_MIN: tl.constexpr,
    DT_MAX: tl.constexpr,
    HAS_DT_LIMIT: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    D_PER_CHANNEL: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    STORE_OUTPUT: tl.constexpr,
    STORE_CHUNK_STATES: tl.constexpr,
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

    offs_t = tl.arange(0, BT)
    offs_p = p_tile * BP + tl.arange(0, BP)
    offs_n = tl.arange(0, N)
    mask_p = offs_p < P
    causal = offs_t[:, None] >= offs_t[None, :]

    decay_rate = tl.load(A + head).to(tl.float32) * LOG2_E
    bias = tl.load(dt_bias + head).to(tl.float32) if HAS_DT_BIAS else 0.0
    if HAS_D:
        if D_PER_CHANNEL:
            skip = tl.load(D + head * P + offs_p, mask=mask_p, other=0.0).to(tl.float32)[None, :]
        else:
            skip = tl.load(D + head).to(tl.float32)

    # A paged pool passes the same tensor as initial and final state; null routes write zero
    # output and never touch it.
    state_slot, active, load_initial = resolve_paged_state(
        sequence, state_indices, has_initial_state, USE_STATE_INDICES, USE_HAS_INITIAL_STATE
    )
    if USE_STATE_INDICES and not active:
        if STORE_OUTPUT:
            for chunk_start in range(0, length, BT):
                token0 = bos + chunk_start
                mask_t = chunk_start + offs_t < length
                xp = token0 * (H * P) + head * P + (offs_t[:, None] * (H * P) + offs_p[None, :])
                tl.store(y + xp, 0.0, mask=mask_t[:, None] & mask_p[None, :])
        return
    state_offset = state_slot.to(tl.int64) * state_stride_slot + head * (P * N)
    state_ptrs = state_offset + offs_p[:, None] * N + offs_n[None, :]
    if HAS_INITIAL_STATE:
        state = tl.load(initial_state + state_ptrs, mask=mask_p[:, None] & load_initial, other=0.0)
        state = state.to(tl.float32)
    else:
        state = tl.zeros([BP, N], dtype=tl.float32)
    if STORE_CHUNK_STATES:
        slot_base = chunk_slot_base(cu_seqlens, sequence, bos, T, BT, IS_VARLEN)
        chunk_state_ptrs = head * P * N + offs_p[:, None] * N + offs_n[None, :]

    for chunk_start in range(0, length, BT):
        # Chunk bases are int64; in-chunk offsets stay int32.
        token0 = bos + chunk_start
        mask_t = chunk_start + offs_t < length
        xp = token0 * (H * P) + head * P + (offs_t[:, None] * (H * P) + offs_p[None, :])
        mask_xp = mask_t[:, None] & mask_p[None, :]
        x_chunk = tl.load(x + xp, mask=mask_xp, other=0.0)
        bc = token0 * (G * N) + group * N + (offs_t[:, None] * (G * N) + offs_n[None, :])
        b_chunk = tl.load(B + bc, mask=mask_t[:, None], other=0.0)
        dt_chunk = tl.load(dt + token0 * H + head + offs_t * H, mask=mask_t, other=0.0)
        dt_chunk = dt_chunk.to(tl.float32)
        dt_chunk = transform_dt(dt_chunk, bias, DT_SOFTPLUS, DT_MIN, DT_MAX, HAS_DT_LIMIT)
        dt_chunk = tl.where(mask_t, dt_chunk, 0.0)
        # Cumulative log2 decay; inactive tail tokens have dt = 0 and keep it constant.
        g = tl.cumsum(dt_chunk * decay_rate, axis=0)
        g_last = tl.sum(tl.where(offs_t == BT - 1, g, 0.0), axis=0)

        if STORE_CHUNK_STATES:
            chunk_slot = slot_base + chunk_start // BT
            tl.store(
                chunk_states + chunk_slot * (H * P * N) + chunk_state_ptrs,
                state.to(chunk_states.dtype.element_ty),
                mask=mask_p[:, None],
            )
        if STORE_OUTPUT:
            c_chunk = tl.load(C + bc, mask=mask_t[:, None], other=0.0)
            scores = tl.dot(c_chunk, tl.trans(b_chunk))
            decay = tl.math.exp2(tl.where(causal, g[:, None] - g[None, :], float("-inf")))
            weights = (scores * decay * dt_chunk[None, :]).to(x_chunk.dtype)
            output = tl.dot(weights, x_chunk)
            entry = tl.dot(c_chunk, tl.trans(state).to(c_chunk.dtype))
            output += tl.math.exp2(g)[:, None] * entry
            if HAS_D:
                output += x_chunk.to(tl.float32) * skip
            if HAS_Z:
                gate = tl.load(z + xp, mask=mask_xp, other=0.0).to(tl.float32)
                output *= gate * tl.sigmoid(gate)
            tl.store(y + xp, output.to(y.dtype.element_ty), mask=mask_xp)

        write_scale = dt_chunk * tl.math.exp2(g_last - g)
        scaled_x = (x_chunk.to(tl.float32) * write_scale[:, None]).to(x_chunk.dtype)
        state = state * tl.math.exp2(g_last) + tl.dot(tl.trans(scaled_x), b_chunk)

    if STORE_FINAL_STATE:
        tl.store(
            final_state + state_ptrs,
            state.to(final_state.dtype.element_ty),
            mask=mask_p[:, None],
        )


def prune_p_tiles(configs, args):
    """Drop P tiles wider than the head dimension; partial tiles are masked."""
    widest = max(triton.next_power_of_2(args["P"]), 32)
    return [config for config in configs if config.kwargs["BP"] <= widest]


def restore_paged_state(_grid, args):
    """Snapshot the routed pool slots, which candidates advance in place, between trials."""
    pool = args["final_state"]
    if pool is None or args["initial_state"] is not pool:
        return None
    # Null routes read slot 0 but never write it, so restoring it is harmless.
    slots = args["state_indices"].clamp_min(0).long()
    snapshot = pool.index_select(0, slots)
    return lambda: pool.index_copy_(0, slots, snapshot)


_mamba2_chunk_fwd = TritonTuner(
    mamba2_chunk_fwd_kernel,
    [
        triton.Config({"BP": bp}, num_warps=warps, num_stages=stages)
        for bp in (32, 64)
        for warps in (4, 8)
        # Triton 3.9 miscompiles the output at BP=64 with num_stages >= 2 (relative error ~1 on
        # B300 at every tested shape, with a correct final state), so BP=64 stays unpipelined.
        for stages in ((1, 2, 3) if bp == 32 else (1,))
    ],
    key=lambda a: (
        a["H"],
        a["G"],
        a["P"],
        a["N"],
        a["x"].dtype,
        a["IS_VARLEN"],
        a["HAS_Z"],
        a["HAS_INITIAL_STATE"],
        a["STORE_FINAL_STATE"],
        a["STORE_OUTPUT"],
        # Few (sequence, head) programs favor splitting P; bucket the program count.
        triton.next_power_of_2(
            a["H"] * (a["cu_seqlens"].shape[0] - 1 if a["IS_VARLEN"] else a["x"].shape[0])
        ),
    ),
    prune=prune_p_tiles,
    reset=restore_paged_state,
)


def num_chunk_slots(x: torch.Tensor, cu_seqlens: torch.Tensor | None) -> int:
    """Chunk-slot capacity of :func:`chunk_slot_base`, known without reading offsets."""
    batch, tokens = x.shape[:2]
    if cu_seqlens is None:
        return batch * triton.cdiv(tokens, CHUNK_SIZE)
    return tokens // CHUNK_SIZE + cu_seqlens.shape[0] - 1


def has_dt_limit(dt_min: float, dt_max: float) -> bool:
    return (dt_min, dt_max) != (0.0, math.inf)


def launch_chunk_fwd(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor | None,
    D: torch.Tensor | None,
    z: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    dt_softplus: bool,
    dt_min: float,
    dt_max: float,
    *,
    y: torch.Tensor | None = None,
    final_state: torch.Tensor | None = None,
    chunk_states: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
) -> None:
    """Run the scan on contiguous token-major inputs, writing each given output buffer.

    With ``state_indices``, ``initial_state`` and ``final_state`` are the same paged pool.
    """
    batch, tokens, heads, head_dim = x.shape
    groups, state_dim = B.shape[2:]
    num_states = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1

    def grid(meta):
        return (num_states * heads, triton.cdiv(head_dim, meta["BP"]))

    _mamba2_chunk_fwd[grid](
        x=x,
        dt=dt,
        A=A,
        dt_bias=dt_bias,
        B=B,
        C=C,
        D=D,
        z=z,
        y=y,
        initial_state=initial_state,
        final_state=final_state,
        chunk_states=chunk_states,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        cu_seqlens=cu_seqlens,
        T=tokens,
        state_stride_slot=(
            initial_state.stride(0) if initial_state is not None else heads * head_dim * state_dim
        ),
        H=heads,
        G=groups,
        P=head_dim,
        N=state_dim,
        BT=CHUNK_SIZE,
        DT_SOFTPLUS=dt_softplus,
        DT_MIN=float(dt_min),
        DT_MAX=float(dt_max),
        HAS_DT_LIMIT=has_dt_limit(dt_min, dt_max),
        HAS_DT_BIAS=dt_bias is not None,
        D_PER_CHANNEL=D is not None and D.ndim == 2,
        HAS_D=D is not None,
        HAS_Z=z is not None,
        HAS_INITIAL_STATE=initial_state is not None,
        STORE_FINAL_STATE=final_state is not None,
        STORE_OUTPUT=y is not None,
        STORE_CHUNK_STATES=chunk_states is not None,
        USE_STATE_INDICES=state_indices is not None,
        USE_HAS_INITIAL_STATE=has_initial_state is not None,
        IS_VARLEN=cu_seqlens is not None,
    )


def mamba2_chunk_fwd(
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
    """Return the output and, if requested, the FP32 final state."""
    batch, _, heads, head_dim = x.shape
    state_dim = B.shape[-1]
    num_states = batch if cu_seqlens is None else cu_seqlens.shape[0] - 1
    x, dt, A, B, C = (tensor.contiguous() for tensor in (x, dt, A, B, C))
    D, z, initial_state, dt_bias = (
        None if tensor is None else tensor.contiguous()
        for tensor in (D, z, initial_state, dt_bias)
    )
    y = x.new_empty(x.shape)
    final_state = (
        x.new_empty(num_states, heads, head_dim, state_dim, dtype=torch.float32)
        if output_final_state
        else None
    )
    launch_chunk_fwd(
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
        y=y,
        final_state=final_state,
    )
    return y, final_state


def mamba2_paged_chunk_fwd(
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
) -> torch.Tensor:
    """Return the output and advance the selected ``state_cache`` slots in place."""
    x, dt, A, B, C = (tensor.contiguous() for tensor in (x, dt, A, B, C))
    D, z, dt_bias = (None if tensor is None else tensor.contiguous() for tensor in (D, z, dt_bias))
    y = x.new_empty(x.shape)
    launch_chunk_fwd(
        x,
        dt,
        A,
        B,
        C,
        D,
        z,
        state_cache,
        dt_bias,
        cu_seqlens,
        dt_softplus,
        dt_min,
        dt_max,
        y=y,
        final_state=state_cache,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
    )
    return y


def mamba2_chunk_states(
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    initial_state: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    dt_softplus: bool,
    dt_min: float,
    dt_max: float,
) -> torch.Tensor:
    """Recompute every chunk's entry state, ``[num_chunk_slots, H, P, N]`` in ``x.dtype``.

    Slots of packed capacity that no chunk owns stay uninitialized.
    """
    heads, head_dim = x.shape[2:]
    chunk_states = x.new_empty(num_chunk_slots(x, cu_seqlens), heads, head_dim, B.shape[-1])
    launch_chunk_fwd(
        x,
        dt,
        A,
        B,
        None,
        None,
        None,
        initial_state,
        dt_bias,
        cu_seqlens,
        dt_softplus,
        dt_min,
        dt_max,
        chunk_states=chunk_states,
    )
    return chunk_states


__all__ = [
    "CHUNK_SIZE",
    "chunk_slot_base",
    "has_dt_limit",
    "mamba2_chunk_fwd",
    "mamba2_chunk_states",
    "mamba2_paged_chunk_fwd",
    "num_chunk_slots",
    "prune_p_tiles",
    "transform_dt",
]
