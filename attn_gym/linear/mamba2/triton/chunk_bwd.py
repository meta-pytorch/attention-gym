"""Triton backward for the chunked Mamba2 SSD scan.

Per ``(sequence, head)`` and ``BT`` token chunk, with transformed step sizes ``v``, in-chunk
cumulative log decay ``g`` (last index ``L``), ``E_ij = exp(g_i - g_j) [i >= j]``, entry state
``S_in`` and exit-state cotangent ``dS_out``, the forward is

.. code-block:: text

    W     = (C B^T) * E * v[None, :]          Yc = W X + exp(g) * (C S_in^T)
    S_out = exp(g_L) S_in + (c * X)^T B,      c_j = v_j exp(g_L - g_j)
    y     = (Yc + D X) * silu(z)

The backward runs three kernels:

1. :func:`mamba2_chunk_states` reruns the forward scan to store each chunk's ``S_in``.
2. ``mamba2_state_bwd_kernel`` scans chunks in reverse per ``(sequence, head, P tile)``,
   storing each chunk's ``dS_out`` in the input dtype and carrying
   ``dS_in = exp(g_L) dS_out + (exp(g) * dYc)^T C`` into the previous chunk; the first
   chunk's ``dS_in`` is the initial-state gradient.
3. ``mamba2_chunk_bwd_kernel`` is chunk-parallel. One program owns a chunk of one B/C group and
   loops over some of its heads, accumulating that group's ``dB``/``dC`` in registers:

.. code-block:: text

    dYc = dy * silu(z)                     dz = dy * (Yc + D X) * silu'(z)
    dX  = W^T dYc + c * (B dS_out^T) + D dYc
    K   = (dYc X^T) * E                    dG = K * v[None, :]        (cotangent of C B^T)
    dC  = dG B + exp(g) * (dYc S_in)       dB = dG^T C + c * (X dS_out)
    Q   = dG * (C B^T)                     (cotangent of g_i - g_j inside E)
    dv  = colsum(K * (C B^T)) + exp(g_L - g) * rowdot(X, B dS_out^T)
    dg  = rowsum(Q) - colsum(Q) + rowdot(Yc_inter, dYc) - c * rowdot(X, B dS_out^T)
          + [i == L] (sum(c * rowdot(X, B dS_out^T)) + exp(g_L) <S_in, dS_out>)
    da  = reverse_cumsum(dg),  dv += A da,  dA = sum(v da)

Matmul operands are rounded to the input dtype as in the forward; every accumulation and
reduction is FP32. ``dA``, ``dD``, and ``ddt_bias`` are per-chunk partial sums reduced afterwards
with ``torch.sum``, and heads split across programs write separate ``dB``/``dC`` partials, so the
result is deterministic.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.tune import TritonTuner
from attn_gym._backends.triton.utils import LOG2_E
from attn_gym.linear.mamba2.triton.chunk_fwd import (
    CHUNK_SIZE,
    chunk_slot_base,
    has_dt_limit,
    mamba2_chunk_states,
    num_chunk_slots,
    prune_p_tiles,
    transform_dt,
)

# Group-head splits are chosen to reach this many chunk-gradient programs per SM.
PROGRAMS_PER_SM = 4


@triton.jit
def transform_dt_grad(
    dt_raw,
    bias,
    DT_SOFTPLUS: tl.constexpr,
    DT_MIN: tl.constexpr,
    DT_MAX: tl.constexpr,
    HAS_DT_LIMIT: tl.constexpr,
):
    """Derivative of :func:`transform_dt` with respect to ``dt_raw``."""
    u = dt_raw + bias
    grad = tl.full(u.shape, 1.0, tl.float32)
    if DT_SOFTPLUS:
        grad = tl.where(u <= 20.0, tl.sigmoid(u), 1.0)
        u = tl.where(u <= 20.0, tl.log(1.0 + tl.exp(u)), u)
    if HAS_DT_LIMIT:
        grad = tl.where((u >= DT_MIN) & (u <= DT_MAX), grad, 0.0)
    return grad


@triton.jit(do_not_specialize=["T"])
def mamba2_state_bwd_kernel(
    dt,
    A,
    dt_bias,
    C,
    z,
    dy,
    d_final_state,
    d_initial_state,
    chunk_state_grads,
    cu_seqlens,
    T,
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
    HAS_Z: tl.constexpr,
    HAS_FINAL_STATE_GRAD: tl.constexpr,
    STORE_INITIAL_STATE_GRAD: tl.constexpr,
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
    slot_base = chunk_slot_base(cu_seqlens, sequence, bos, T, BT, IS_VARLEN)

    offs_t = tl.arange(0, BT)
    offs_p = p_tile * BP + tl.arange(0, BP)
    offs_n = tl.arange(0, N)
    mask_p = offs_p < P

    decay_rate = tl.load(A + head).to(tl.float32) * LOG2_E
    bias = tl.load(dt_bias + head).to(tl.float32) if HAS_DT_BIAS else 0.0

    state_ptrs = (sequence.to(tl.int64) * H + head) * P * N + offs_p[:, None] * N + offs_n[None, :]
    if HAS_FINAL_STATE_GRAD:
        d_state = tl.load(d_final_state + state_ptrs, mask=mask_p[:, None], other=0.0)
        d_state = d_state.to(tl.float32)
    else:
        d_state = tl.zeros([BP, N], dtype=tl.float32)
    chunk_state_ptrs = head * P * N + offs_p[:, None] * N + offs_n[None, :]

    num_chunks = tl.cdiv(length, BT)
    for step in range(num_chunks):
        chunk = num_chunks - 1 - step
        tl.store(
            chunk_state_grads + (slot_base + chunk) * (H * P * N) + chunk_state_ptrs,
            d_state.to(chunk_state_grads.dtype.element_ty),
            mask=mask_p[:, None],
        )
        chunk_start = chunk * BT
        token0 = bos + chunk_start
        mask_t = chunk_start + offs_t < length
        dt_chunk = tl.load(dt + token0 * H + head + offs_t * H, mask=mask_t, other=0.0)
        dt_chunk = transform_dt(
            dt_chunk.to(tl.float32), bias, DT_SOFTPLUS, DT_MIN, DT_MAX, HAS_DT_LIMIT
        )
        dt_chunk = tl.where(mask_t, dt_chunk, 0.0)
        g = tl.cumsum(dt_chunk * decay_rate, axis=0)
        g_last = tl.sum(tl.where(offs_t == BT - 1, g, 0.0), axis=0)

        xp = token0 * (H * P) + head * P + (offs_t[:, None] * (H * P) + offs_p[None, :])
        mask_xp = mask_t[:, None] & mask_p[None, :]
        d_out = tl.load(dy + xp, mask=mask_xp, other=0.0).to(tl.float32)
        if HAS_Z:
            gate = tl.load(z + xp, mask=mask_xp, other=0.0).to(tl.float32)
            d_out *= gate * tl.sigmoid(gate)
        bc = token0 * (G * N) + group * N + (offs_t[:, None] * (G * N) + offs_n[None, :])
        c_chunk = tl.load(C + bc, mask=mask_t[:, None], other=0.0)
        scaled_d_out = (d_out * tl.math.exp2(g)[:, None]).to(c_chunk.dtype)
        d_state = d_state * tl.math.exp2(g_last) + tl.dot(tl.trans(scaled_d_out), c_chunk)

    if STORE_INITIAL_STATE_GRAD:
        tl.store(d_initial_state + state_ptrs, d_state, mask=mask_p[:, None])


@triton.jit(do_not_specialize=["T", "num_sequences"])
def mamba2_chunk_bwd_kernel(
    x,
    dt,
    A,
    dt_bias,
    B,
    C,
    D,
    z,
    dy,
    chunk_states,
    chunk_state_grads,
    dx,
    ddt,
    dz,
    dB_partial,
    dC_partial,
    dA_partial,
    dD_partial,
    ddt_bias_partial,
    cu_seqlens,
    T,
    num_sequences,
    H: tl.constexpr,
    G: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    BT: tl.constexpr,
    BP: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    DT_MIN: tl.constexpr,
    DT_MAX: tl.constexpr,
    HAS_DT_LIMIT: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    D_PER_CHANNEL: tl.constexpr,
    HAS_D: tl.constexpr,
    HAS_Z: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    slot, group_split = tl.program_id(0), tl.program_id(1)
    SPLITS: tl.constexpr = H // G // HEADS_PER_PROGRAM
    group, split = group_split // SPLITS, group_split % SPLITS
    if IS_VARLEN:
        # The owner of a slot is the last sequence whose first slot does not exceed it.
        low = 0
        high = num_sequences - 1
        while low < high:
            middle = (low + high + 1) // 2
            owns = tl.load(cu_seqlens + middle) // BT + middle <= slot
            low = tl.where(owns, middle, low)
            high = tl.where(owns, high, middle - 1)
        sequence = low
        start = tl.load(cu_seqlens + sequence)
        length = tl.load(cu_seqlens + sequence + 1) - start
        chunk = slot - (start // BT + sequence)
        bos = start.to(tl.int64)
    else:
        num_chunks = tl.cdiv(T, BT)
        sequence, chunk = slot // num_chunks, slot % num_chunks
        bos = sequence.to(tl.int64) * T
        length = T
    chunk_start = chunk * BT
    # Packed capacity has unowned slots; their partial sums stay zero.
    if chunk_start >= length:
        return

    token0 = bos + chunk_start
    offs_t = tl.arange(0, BT)
    offs_n = tl.arange(0, N)
    mask_t = chunk_start + offs_t < length
    causal = offs_t[:, None] >= offs_t[None, :]
    is_last = offs_t == BT - 1

    bc = token0 * (G * N) + group * N + (offs_t[:, None] * (G * N) + offs_n[None, :])
    b_chunk = tl.load(B + bc, mask=mask_t[:, None], other=0.0)
    c_chunk = tl.load(C + bc, mask=mask_t[:, None], other=0.0)
    scores = tl.dot(c_chunk, tl.trans(b_chunk))
    d_b = tl.zeros([BT, N], dtype=tl.float32)
    d_c = tl.zeros([BT, N], dtype=tl.float32)
    slot64 = slot.to(tl.int64)

    for head_index in range(HEADS_PER_PROGRAM):
        head = group * (H // G) + split * HEADS_PER_PROGRAM + head_index
        decay_rate = tl.load(A + head).to(tl.float32)
        bias = tl.load(dt_bias + head).to(tl.float32) if HAS_DT_BIAS else 0.0
        dt_raw = tl.load(dt + token0 * H + head + offs_t * H, mask=mask_t, other=0.0)
        dt_raw = dt_raw.to(tl.float32)
        dt_chunk = transform_dt(dt_raw, bias, DT_SOFTPLUS, DT_MIN, DT_MAX, HAS_DT_LIMIT)
        dt_chunk = tl.where(mask_t, dt_chunk, 0.0)
        g = tl.cumsum(dt_chunk * (decay_rate * LOG2_E), axis=0)
        g_last = tl.sum(tl.where(is_last, g, 0.0), axis=0)
        decay = tl.math.exp2(tl.where(causal, g[:, None] - g[None, :], float("-inf")))
        weights = (scores * decay * dt_chunk[None, :]).to(c_chunk.dtype)
        exp_g = tl.math.exp2(g)
        tail_decay = tl.math.exp2(g_last - g)
        write_scale = dt_chunk * tail_decay

        d_weights = tl.zeros([BT, BT], dtype=tl.float32)
        inter_dot = tl.zeros([BT], dtype=tl.float32)
        write_dot = tl.zeros([BT], dtype=tl.float32)
        state_dot = tl.zeros([N], dtype=tl.float32)
        d_skip = 0.0
        state_base = (slot64 * H + head) * (P * N)
        for p_start in range(0, P, BP):
            offs_p = p_start + tl.arange(0, BP)
            mask_p = offs_p < P
            xp = token0 * (H * P) + head * P + (offs_t[:, None] * (H * P) + offs_p[None, :])
            mask_xp = mask_t[:, None] & mask_p[None, :]
            x_chunk = tl.load(x + xp, mask=mask_xp, other=0.0)
            x_float = x_chunk.to(tl.float32)
            d_out = tl.load(dy + xp, mask=mask_xp, other=0.0).to(tl.float32)
            state_ptrs = state_base + offs_p[:, None] * N + offs_n[None, :]
            state = tl.load(chunk_states + state_ptrs, mask=mask_p[:, None], other=0.0)
            d_state = tl.load(chunk_state_grads + state_ptrs, mask=mask_p[:, None], other=0.0)
            state_dot += tl.sum(state.to(tl.float32) * d_state.to(tl.float32), axis=0)
            state, d_state = state.to(x_chunk.dtype), d_state.to(x_chunk.dtype)
            inter = exp_g[:, None] * tl.dot(c_chunk, tl.trans(state))
            if HAS_D:
                if D_PER_CHANNEL:
                    skip = tl.load(D + head * P + offs_p, mask=mask_p, other=0.0)
                    skip = skip.to(tl.float32)[None, :]
                else:
                    skip = tl.load(D + head).to(tl.float32)
            if HAS_Z:
                output = tl.dot(weights, x_chunk) + inter
                if HAS_D:
                    output += x_float * skip
                gate = tl.load(z + xp, mask=mask_xp, other=0.0).to(tl.float32)
                sigmoid = tl.sigmoid(gate)
                d_gate = d_out * output * sigmoid * (1.0 + gate * (1.0 - sigmoid))
                tl.store(dz + xp, d_gate.to(dz.dtype.element_ty), mask=mask_xp)
                d_out *= gate * sigmoid
            if HAS_D:
                if D_PER_CHANNEL:
                    tl.store(
                        dD_partial + (slot64 * H + head) * P + offs_p,
                        tl.sum(d_out * x_float, axis=0),
                        mask=mask_p,
                    )
                else:
                    d_skip += tl.sum(tl.sum(d_out * x_float, axis=1), axis=0)
            inter_dot += tl.sum(inter * d_out, axis=1)
            b_d_state = tl.dot(b_chunk, tl.trans(d_state))
            write_dot += tl.sum(x_float * b_d_state, axis=1)

            d_out_lp = d_out.to(x_chunk.dtype)
            d_x = tl.dot(tl.trans(weights), d_out_lp) + write_scale[:, None] * b_d_state
            if HAS_D:
                d_x += skip * d_out
            tl.store(dx + xp, d_x.to(dx.dtype.element_ty), mask=mask_xp)
            d_weights += tl.dot(d_out_lp, tl.trans(x_chunk))
            scaled_d_out = (d_out * exp_g[:, None]).to(x_chunk.dtype)
            d_c += tl.dot(scaled_d_out, state)
            scaled_x = (x_float * write_scale[:, None]).to(x_chunk.dtype)
            d_b += tl.dot(scaled_x, d_state)

        # ``decay`` is zero above the diagonal, which applies the causal mask to dW.
        d_masked = d_weights * decay
        d_scores = d_masked * dt_chunk[None, :]
        d_scores_lp = d_scores.to(c_chunk.dtype)
        d_c += tl.dot(d_scores_lp, b_chunk)
        d_b += tl.dot(tl.trans(d_scores_lp), c_chunk)

        d_segment = d_scores * scores
        write_term = write_dot * write_scale
        d_v = tl.sum(d_masked * scores, axis=0) + write_dot * tail_decay
        d_g = tl.sum(d_segment, axis=1) - tl.sum(d_segment, axis=0) + inter_dot - write_term
        last_term = tl.sum(write_term, axis=0) + tl.math.exp2(g_last) * tl.sum(state_dot, axis=0)
        d_g += tl.where(is_last, last_term, 0.0)
        d_a = tl.cumsum(d_g, axis=0, reverse=True)
        d_v += d_a * decay_rate
        d_dt = d_v * transform_dt_grad(dt_raw, bias, DT_SOFTPLUS, DT_MIN, DT_MAX, HAS_DT_LIMIT)
        d_dt = tl.where(mask_t, d_dt, 0.0)
        tl.store(ddt + token0 * H + head + offs_t * H, d_dt, mask=mask_t)

        partial = slot64 * H + head
        tl.store(dA_partial + partial, tl.sum(d_a * dt_chunk, axis=0))
        if HAS_DT_BIAS:
            tl.store(ddt_bias_partial + partial, tl.sum(d_dt, axis=0))
        if HAS_D and not D_PER_CHANNEL:
            tl.store(dD_partial + partial, d_skip)

    tokens = T if IS_VARLEN else num_sequences * T
    partial_bc = split.to(tl.int64) * tokens * (G * N)
    tl.store(dB_partial + partial_bc + bc, d_b, mask=mask_t[:, None])
    tl.store(dC_partial + partial_bc + bc, d_c, mask=mask_t[:, None])


_mamba2_state_bwd = TritonTuner(
    mamba2_state_bwd_kernel,
    [
        triton.Config({"BP": bp}, num_warps=warps, num_stages=stages)
        for bp in (32, 64)
        for warps in (4, 8)
        for stages in (1, 2, 3)
    ],
    key=lambda a: (
        a["H"],
        a["G"],
        a["P"],
        a["N"],
        a["C"].dtype,
        a["IS_VARLEN"],
        a["HAS_Z"],
        triton.next_power_of_2(
            a["H"] * (a["cu_seqlens"].shape[0] - 1 if a["IS_VARLEN"] else a["dy"].shape[0])
        ),
    ),
    prune=prune_p_tiles,
)

_mamba2_chunk_bwd = TritonTuner(
    mamba2_chunk_bwd_kernel,
    [
        triton.Config({"BP": bp}, num_warps=warps, num_stages=stages)
        for bp in (32, 64)
        for warps in (4, 8)
        for stages in (1, 2)
    ],
    key=lambda a: (
        a["H"],
        a["G"],
        a["P"],
        a["N"],
        a["x"].dtype,
        a["HEADS_PER_PROGRAM"],
        a["IS_VARLEN"],
        a["HAS_Z"],
        a["HAS_D"],
        triton.next_power_of_2(a["dA_partial"].shape[0]),
    ),
    prune=prune_p_tiles,
)


def heads_per_program(x: torch.Tensor, groups: int, num_slots: int) -> int:
    """Largest divisor of the group size that still launches enough chunk programs."""
    heads_per_group = x.shape[2] // groups
    target = PROGRAMS_PER_SM * torch.cuda.get_device_properties(x.device).multi_processor_count
    for splits in range(1, heads_per_group + 1):
        if heads_per_group % splits == 0 and num_slots * groups * splits >= target:
            return heads_per_group // splits
    return 1


def mamba2_chunk_bwd(
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
    d_output: torch.Tensor,
    d_final_state: torch.Tensor | None,
    dt_softplus: bool,
    dt_min: float,
    dt_max: float,
) -> tuple[torch.Tensor, ...]:
    """Return ``(dx, ddt, dA, dB, dC, dD, dz, ddt_bias, d_initial_state)``.

    Gradients of absent optional inputs are empty tensors. Tokens past the terminal packed
    offset get zero gradients.
    """
    batch, tokens, heads, head_dim = x.shape
    groups, state_dim = B.shape[2:]
    is_varlen = cu_seqlens is not None
    num_states = cu_seqlens.shape[0] - 1 if is_varlen else batch
    x, dt, A, B, C, d_output = (tensor.contiguous() for tensor in (x, dt, A, B, C, d_output))
    D, z, initial_state, dt_bias, d_final_state = (
        None if tensor is None else tensor.contiguous()
        for tensor in (D, z, initial_state, dt_bias, d_final_state)
    )
    num_slots = num_chunk_slots(x, cu_seqlens)
    transform = {
        "DT_SOFTPLUS": dt_softplus,
        "DT_MIN": float(dt_min),
        "DT_MAX": float(dt_max),
        "HAS_DT_LIMIT": has_dt_limit(dt_min, dt_max),
        "HAS_DT_BIAS": dt_bias is not None,
    }
    shapes = {"H": heads, "G": groups, "P": head_dim, "N": state_dim, "BT": CHUNK_SIZE}

    chunk_states = mamba2_chunk_states(
        x, dt, A, B, initial_state, dt_bias, cu_seqlens, dt_softplus, dt_min, dt_max
    )
    chunk_state_grads = torch.empty_like(chunk_states)
    d_initial_state = (
        x.new_empty(num_states, heads, head_dim, state_dim, dtype=torch.float32)
        if initial_state is not None
        else None
    )
    _mamba2_state_bwd[lambda meta: (num_states * heads, triton.cdiv(head_dim, meta["BP"]))](
        dt=dt,
        A=A,
        dt_bias=dt_bias,
        C=C,
        z=z,
        dy=d_output,
        d_final_state=d_final_state,
        d_initial_state=d_initial_state,
        chunk_state_grads=chunk_state_grads,
        cu_seqlens=cu_seqlens,
        T=tokens,
        HAS_Z=z is not None,
        HAS_FINAL_STATE_GRAD=d_final_state is not None,
        STORE_INITIAL_STATE_GRAD=d_initial_state is not None,
        IS_VARLEN=is_varlen,
        **shapes,
        **transform,
    )

    # Packed tokens past the terminal offset belong to no chunk and keep zero gradients.
    new_buffer = torch.zeros if is_varlen else torch.empty
    dx = new_buffer(x.shape, dtype=x.dtype, device=x.device)
    dz = None if z is None else new_buffer(x.shape, dtype=x.dtype, device=x.device)
    ddt = new_buffer(dt.shape, dtype=torch.float32, device=x.device)
    group_heads = heads_per_program(x, groups, num_slots)
    splits = heads // groups // group_heads
    dB_partial = new_buffer(splits, *B.shape, dtype=torch.float32, device=x.device)
    dC_partial = new_buffer(splits, *B.shape, dtype=torch.float32, device=x.device)
    dA_partial = x.new_zeros(num_slots, heads, dtype=torch.float32)
    dD_partial = None if D is None else x.new_zeros(num_slots, *D.shape, dtype=torch.float32)
    ddt_bias_partial = None if dt_bias is None else torch.zeros_like(dA_partial)
    _mamba2_chunk_bwd[(num_slots, groups * splits)](
        x=x,
        dt=dt,
        A=A,
        dt_bias=dt_bias,
        B=B,
        C=C,
        D=D,
        z=z,
        dy=d_output,
        chunk_states=chunk_states,
        chunk_state_grads=chunk_state_grads,
        dx=dx,
        ddt=ddt,
        dz=dz,
        dB_partial=dB_partial,
        dC_partial=dC_partial,
        dA_partial=dA_partial,
        dD_partial=dD_partial,
        ddt_bias_partial=ddt_bias_partial,
        cu_seqlens=cu_seqlens,
        T=tokens,
        num_sequences=num_states,
        HEADS_PER_PROGRAM=group_heads,
        D_PER_CHANNEL=D is not None and D.ndim == 2,
        HAS_D=D is not None,
        HAS_Z=z is not None,
        IS_VARLEN=is_varlen,
        **shapes,
        **transform,
    )

    empty = x.new_empty(0)
    return (
        dx,
        ddt.to(dt.dtype),
        dA_partial.sum(0).to(A.dtype),
        dB_partial.sum(0).to(B.dtype),
        dC_partial.sum(0).to(C.dtype),
        empty if D is None else dD_partial.sum(0).to(D.dtype),
        empty if dz is None else dz,
        empty if dt_bias is None else ddt_bias_partial.sum(0).to(dt_bias.dtype),
        empty if initial_state is None else d_initial_state.to(initial_state.dtype),
    )


__all__ = ["mamba2_chunk_bwd"]
