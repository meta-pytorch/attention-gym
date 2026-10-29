"""Triton backward for the chunked Mamba2 SSD scan.

Per ``(sequence, head)`` and ``BT`` token chunk, with transformed step sizes ``v``, in-chunk
cumulative log decay ``g`` (last index ``L``), ``E_ij = exp(g_i - g_j) [i >= j]``, entry state
``S_in`` and exit-state cotangent ``dS_out``, the forward is

.. code-block:: text

    W     = (C B^T) * E * v[None, :]          Yc = W X + exp(g) * (C S_in^T)
    S_out = exp(g_L) S_in + (c * X)^T B,      c_j = v_j exp(g_L - g_j)
    y     = (Yc + D X) * silu(z)

The backward transforms ``dt`` once (``mamba2_dt_fwd_kernel``), runs four kernels on the FP32
transformed step sizes ``v``, and maps ``dv`` back through the transform
(``mamba2_dt_bwd_kernel``):

1. ``mamba2_gate_bwd_kernel`` reads the forward's stored pre-gate output ``Yc + D X`` and forms
   ``dz = dy * (Yc + D X) * silu'(z)`` and the core cotangent ``dY = dy * silu(z)``
   elementwise, so the scan kernels below never see the gate. It reduces ``dD`` from the FP32
   ``dY`` and stores ``dY`` in the input dtype, the dtype in which the ``dx``, ``dB``, and
   ``dC`` products (and ``dx``'s skip term, as in ``mamba_ssm``) consume it.
2. :func:`mamba2_chunk_states` reruns the forward scan to store each chunk's ``S_in``.
3. ``mamba2_state_bwd_kernel`` scans chunks in reverse per ``(sequence, head, P tile)``,
   storing each chunk's ``dS_out`` in the input dtype and carrying
   ``dS_in = exp(g_L) dS_out + (exp(g) * dYc)^T C`` into the previous chunk; the first
   chunk's ``dS_in`` is the initial-state gradient. Like the forward scan, it splits sequences
   into segments when few programs run, with a ``SUMMARY`` launch for every segment but the
   first.
4. ``mamba2_chunk_bwd_kernel`` is chunk-parallel. One program owns a chunk of one B/C group and
   loops over some of its heads, accumulating that group's ``dB``/``dC`` in registers:

.. code-block:: text

    dX  = W^T dYc + c * (B dS_out^T) + D dYc
    K   = (dYc X^T) * E                    dG = K * v[None, :]        (cotangent of C B^T)
    dC  = dG B + exp(g) * (dYc S_in)       dB = dG^T C + c * (X dS_out)
    Q   = dG * (C B^T)                     (cotangent of g_i - g_j inside E)
    dv  = colsum(K * (C B^T)) + exp(g_L - g) * rowdot(X, B dS_out^T)
    w   = c * rowdot(X, B dS_out^T)        (write weights c_j = v_j exp(g_L - g_j))
    dg  = rowdot(Yc_inter, dYc) + [i == L] exp(g_L) <S_in, dS_out>
    da  = reverse_cumsum(dg) + sum_{i >= k > j} Q_ij + sum_{j < k} w_j
    dv += A da,  dA = sum(v da)

``da`` is the cotangent of the in-chunk log-decay increments ``a_k = v_k A``. The pairwise
``Q`` term and the write term sum only the contributions that cross each increment; the
algebraically equal ``rowsum(Q) - colsum(Q)`` and ``-w + [i == L] sum(w)`` forms cancel large
terms in FP32 and lose small ones.

Matmul operands are rounded to the input dtype as in the forward; every accumulation and
reduction is FP32. ``ddt_bias`` is the sum of ``mamba2_dt_bwd_kernel``'s output, ``dA`` and
``dD`` are per-chunk partial sums reduced afterwards with ``torch.sum``, and heads split across
programs write separate ``dB``/``dC`` partials, so the result is deterministic.

The ``dS`` scan decays through :func:`scale_by_exp2`, as does the ``<S_in, dS_out>`` term after
its contraction. ``mamba2_chunk_bwd_kernel`` forms its other in-chunk decay factors
(``exp(g)``, ``c_j``, ``E``) on their own, so a gradient contribution through a factor below
``2^-126`` flushes to zero, and it contracts ``S_in`` and the writes with ``dS_out`` before
decaying them, so a product beyond FP32 range overflows. Splitting the factors as the carries do
costs about 9% of this kernel on B300, whose tuned configuration already spills registers;
decaying the writes before their contraction costs more.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.tune import TritonTuner
from attn_gym._backends.triton.utils import LOG2_E
from attn_gym.linear.mamba2.triton.chunk_fwd import (
    CHUNK_SIZE,
    DT_BLOCK,
    chunk_slot_base,
    decay_by_exp2,
    has_dt_limit,
    has_dt_transform,
    heads_per_program,
    locate_chunk,
    mamba2_chunk_states,
    num_chunk_slots,
    prune_p_tiles,
    scale_by_exp2,
    segment_plan,
    segment_ptrs,
    transform_dt,
    transformed_dt,
)

GATE_ROWS = 32


@triton.jit(do_not_specialize=["tokens", "num_sequences"])
def mamba2_gate_bwd_kernel(
    dy,
    z,
    pre_gate,
    x,
    d_core,
    dz,
    dD_partial,
    cu_seqlens,
    tokens,
    num_sequences,
    H: tl.constexpr,
    P: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_P: tl.constexpr,
    HAS_D: tl.constexpr,
    D_PER_CHANNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """One ``[BLOCK_T tokens, P]`` tile of one head; ``dD`` partials are per token block."""
    token_block, head = tl.program_id(0), tl.program_id(1)
    rows = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_p = tl.arange(0, BLOCK_P)
    mask_p = offs_p < P
    mask = (rows < tokens)[:, None] & mask_p[None, :]
    offs = rows.to(tl.int64)[:, None] * (H * P) + head * P + offs_p[None, :]
    d_out = tl.load(dy + offs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(z + offs, mask=mask, other=0.0).to(tl.float32)
    output = tl.load(pre_gate + offs, mask=mask, other=0.0).to(tl.float32)
    sigmoid = tl.sigmoid(gate)
    d_gate = d_out * output * sigmoid * (1.0 + gate * (1.0 - sigmoid))
    d_out *= gate * sigmoid
    if IS_VARLEN:
        # Tokens past the terminal offset hold unwritten forward output and get zero gradient.
        valid = rows < tl.load(cu_seqlens + num_sequences)
        d_gate = tl.where(valid[:, None], d_gate, 0.0)
        d_out = tl.where(valid[:, None], d_out, 0.0)
    tl.store(dz + offs, d_gate.to(dz.dtype.element_ty), mask=mask)
    tl.store(d_core + offs, d_out.to(d_core.dtype.element_ty), mask=mask)
    if HAS_D:
        # Masked like d_out: capacity past the terminal offset may hold any x, even NaN.
        if IS_VARLEN:
            mask &= valid[:, None]
        skip = d_out * tl.load(x + offs, mask=mask, other=0.0).to(tl.float32)
        partial = token_block.to(tl.int64) * H + head
        if D_PER_CHANNEL:
            tl.store(dD_partial + partial * P + offs_p, tl.sum(skip, axis=0), mask=mask_p)
        else:
            tl.store(dD_partial + partial, tl.sum(tl.sum(skip, axis=1), axis=0))


@triton.jit
def transform_dt_grad(
    dt_raw,
    bias,
    DT_SOFTPLUS: tl.constexpr,
    DT_MIN: tl.constexpr,
    DT_MAX: tl.constexpr,
    HAS_DT_LIMIT: tl.constexpr,
):
    """Derivative of :func:`transform_dt` with respect to ``dt_raw``.

    Like ``Tensor.clamp``, the clamp passes no gradient at its bounds.
    """
    u = dt_raw + bias
    grad = tl.full(u.shape, 1.0, tl.float32)
    if DT_SOFTPLUS:
        grad = tl.where(u > 20.0, 1.0, tl.sigmoid(u))
    if HAS_DT_LIMIT:
        u = transform_dt(dt_raw, bias, DT_SOFTPLUS, DT_MIN, DT_MAX, False)
        grad = tl.where((u > DT_MIN) & (u < DT_MAX), grad, 0.0)
    return grad


@triton.jit
def mamba2_dt_bwd_kernel(
    ddt,
    dt,
    dt_bias,
    cu_seqlens,
    numel,
    num_sequences,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    DT_SOFTPLUS: tl.constexpr,
    DT_MIN: tl.constexpr,
    DT_MAX: tl.constexpr,
    HAS_DT_LIMIT: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Map ``dv`` (cotangent of the transformed step size) to ``d dt_raw`` in place."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    if IS_VARLEN:
        # Capacity past the terminal offset may hold any dt, even NaN; its gradient stays zero.
        mask &= offs < tl.load(cu_seqlens + num_sequences).to(tl.int64) * H
    raw = tl.load(dt + offs, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(dt_bias + offs % H, mask=mask, other=0.0).to(tl.float32) if HAS_DT_BIAS else 0.0
    grad = transform_dt_grad(raw, bias, DT_SOFTPLUS, DT_MIN, DT_MAX, HAS_DT_LIMIT)
    tl.store(ddt + offs, tl.load(ddt + offs, mask=mask, other=0.0) * grad, mask=mask)


@triton.jit(do_not_specialize=["T", "segment_length"])
def mamba2_state_bwd_kernel(
    dt,
    A,
    C,
    dy,
    d_final_state,
    d_initial_state,
    chunk_state_grads,
    segment_states,
    segment_decays,
    cu_seqlens,
    T,
    segment_length,
    H: tl.constexpr,
    G: tl.constexpr,
    P: tl.constexpr,
    N: tl.constexpr,
    BT: tl.constexpr,
    BP: tl.constexpr,
    HAS_FINAL_STATE_GRAD: tl.constexpr,
    STORE_INITIAL_STATE_GRAD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SEGMENTED: tl.constexpr,
    SUMMARY: tl.constexpr,
):
    sequence_head, p_tile, segment = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    sequence, head = sequence_head // H, sequence_head % H
    group = head // (H // G)
    if IS_VARLEN:
        bos = tl.load(cu_seqlens + sequence).to(tl.int64)
        length = (tl.load(cu_seqlens + sequence + 1) - tl.load(cu_seqlens + sequence)).to(tl.int32)
    else:
        bos = sequence.to(tl.int64) * T
        length = T
    segment_start = segment * segment_length
    if segment > 0 and segment_start >= length:
        return
    # The first segment's summary is never read.
    if SUMMARY and segment == 0:
        return
    segment_end = tl.minimum(segment_start + segment_length, length)
    slot_base = chunk_slot_base(cu_seqlens, sequence, bos, T, BT, IS_VARLEN)

    offs_t = tl.arange(0, BT)
    offs_p = p_tile * BP + tl.arange(0, BP)
    offs_n = tl.arange(0, N)
    mask_p = offs_p < P

    decay_rate = tl.load(A + head).to(tl.float32) * LOG2_E

    state_ptrs = (sequence.to(tl.int64) * H + head) * P * N + offs_p[:, None] * N + offs_n[None, :]
    if HAS_FINAL_STATE_GRAD and not SUMMARY:
        d_state = tl.load(d_final_state + state_ptrs, mask=mask_p[:, None], other=0.0)
        d_state = d_state.to(tl.float32)
    else:
        d_state = tl.zeros([BP, N], dtype=tl.float32)
    if SEGMENTED and not SUMMARY:
        last_segment = tl.cdiv(length, segment_length) - 1
        for step in range(last_segment - segment):
            index, ptrs = segment_ptrs(
                sequence, head, last_segment - step, offs_p, offs_n, H, P, N
            )
            local = tl.load(segment_states + ptrs, mask=mask_p[:, None], other=0.0)
            d_state = scale_by_exp2(d_state, tl.load(segment_decays + index)) + local
    chunk_state_ptrs = head * P * N + offs_p[:, None] * N + offs_n[None, :]
    total_decay = 0.0

    first_chunk = segment_start // BT
    end_chunk = tl.cdiv(segment_end, BT)
    for step in range(end_chunk - first_chunk):
        chunk = end_chunk - 1 - step
        if not SUMMARY:
            tl.store(
                chunk_state_grads + (slot_base + chunk) * (H * P * N) + chunk_state_ptrs,
                d_state.to(chunk_state_grads.dtype.element_ty),
                mask=mask_p[:, None],
            )
        chunk_start = chunk * BT
        token0 = bos + chunk_start
        mask_t = chunk_start + offs_t < length
        dt_chunk = tl.load(dt + token0 * H + head + offs_t * H, mask=mask_t, other=0.0)
        dt_chunk = dt_chunk.to(tl.float32)
        g = tl.cumsum(dt_chunk * decay_rate, axis=0)
        g_last = tl.sum(tl.where(offs_t == BT - 1, g, 0.0), axis=0)
        total_decay += g_last

        xp = token0 * (H * P) + head * P + (offs_t[:, None] * (H * P) + offs_p[None, :])
        mask_xp = mask_t[:, None] & mask_p[None, :]
        d_out = tl.load(dy + xp, mask=mask_xp, other=0.0).to(tl.float32)
        bc = token0 * (G * N) + group * N + (offs_t[:, None] * (G * N) + offs_n[None, :])
        c_chunk = tl.load(C + bc, mask=mask_t[:, None], other=0.0)
        steep = g_last < -126.0
        scaled_d_out = decay_by_exp2(d_out, g[:, None], steep).to(c_chunk.dtype)
        d_state = scale_by_exp2(d_state, g_last) + tl.dot(tl.trans(scaled_d_out), c_chunk)

    if SUMMARY:
        index, ptrs = segment_ptrs(sequence, head, segment, offs_p, offs_n, H, P, N)
        tl.store(segment_states + ptrs, d_state, mask=mask_p[:, None])
        if p_tile == 0:
            tl.store(segment_decays + index, total_decay)
    if STORE_INITIAL_STATE_GRAD and segment == 0:
        tl.store(d_initial_state + state_ptrs, d_state, mask=mask_p[:, None])


@triton.jit(do_not_specialize=["T", "num_sequences"])
def mamba2_chunk_bwd_kernel(
    x,
    dt,
    A,
    B,
    C,
    D,
    dy,
    chunk_states,
    chunk_state_grads,
    dx,
    ddt,
    dB_partial,
    dC_partial,
    dA_partial,
    dD_partial,
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
    D_PER_CHANNEL: tl.constexpr,
    HAS_D: tl.constexpr,
    REDUCE_D_GRAD: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    slot, group_split = tl.program_id(0), tl.program_id(1)
    SPLITS: tl.constexpr = H // G // HEADS_PER_PROGRAM
    group, split = group_split // SPLITS, group_split % SPLITS
    bos, length, chunk = locate_chunk(slot, cu_seqlens, num_sequences, T, BT, IS_VARLEN)
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
        dt_chunk = tl.load(dt + token0 * H + head + offs_t * H, mask=mask_t, other=0.0)
        dt_chunk = dt_chunk.to(tl.float32)
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
                    if REDUCE_D_GRAD:
                        tl.store(
                            dD_partial + (slot64 * H + head) * P + offs_p,
                            tl.sum(d_out * x_float, axis=0),
                            mask=mask_p,
                        )
                else:
                    skip = tl.load(D + head).to(tl.float32)
                    if REDUCE_D_GRAD:
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

        # The cotangent of ``g_i - g_j`` reaches the increments ``a_k`` with ``j < k <= i``.
        # Summing exactly those pairs keeps the diagonal (whose decay has no derivative) and the
        # cancelling row and column sums of ``g``'s cotangent out of the FP32 reductions.
        strict = offs_t[:, None] > offs_t[None, :]
        crossing = tl.cumsum(tl.where(strict, d_scores * scores, 0.0), axis=0, reverse=True)
        d_a = tl.sum(tl.where(strict, crossing, 0.0), axis=1)
        # The write ``w_j`` decays by ``exp(g_L - g_j)``, which depends on the increments
        # after ``j``: increment ``k`` collects the writes before it.
        write_term = write_dot * write_scale
        d_a += tl.sum(tl.where(strict, write_term[None, :], 0.0), axis=1)
        d_v = tl.sum(d_masked * scores, axis=0) + write_dot * tail_decay
        state_term = scale_by_exp2(tl.sum(state_dot, axis=0), g_last)
        d_g = inter_dot + tl.where(is_last, state_term, 0.0)
        d_a += tl.cumsum(d_g, axis=0, reverse=True)
        d_v += d_a * decay_rate
        tl.store(ddt + token0 * H + head + offs_t * H, tl.where(mask_t, d_v, 0.0), mask=mask_t)

        partial = slot64 * H + head
        tl.store(dA_partial + partial, tl.sum(d_a * dt_chunk, axis=0))
        if REDUCE_D_GRAD and not D_PER_CHANNEL:
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
        a["SUMMARY"],
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
        a["HAS_D"],
        a["REDUCE_D_GRAD"],
        triton.next_power_of_2(a["dA_partial"].shape[0]),
    ),
    prune=prune_p_tiles,
)


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
    pre_gate: torch.Tensor | None,
    d_output: torch.Tensor,
    d_final_state: torch.Tensor | None,
    dt_softplus: bool,
    dt_min: float,
    dt_max: float,
) -> tuple[torch.Tensor, ...]:
    """Return ``(dx, ddt, dA, dB, dC, dD, dz, ddt_bias, d_initial_state)``.

    ``pre_gate`` is the forward's pre-gate output ``Yc + D X``, required with ``z``.
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
    dz = dD = None
    if z is not None:
        # The gate backward leaves a gate-free core cotangent for the scan kernels and reduces
        # dD from its FP32 value.
        if pre_gate is None or pre_gate.shape != x.shape:
            raise ValueError("mamba2_chunk_bwd requires the pre-gate output with z")
        d_core, dz = torch.empty_like(d_output), torch.empty_like(z)
        token_blocks = triton.cdiv(batch * tokens, GATE_ROWS)
        gate_dD = (
            None
            if D is None
            else x.new_empty(token_blocks, heads, *D.shape[1:], dtype=torch.float32)
        )
        mamba2_gate_bwd_kernel[(token_blocks, heads)](
            d_output,
            z,
            pre_gate.contiguous(),
            x,
            d_core,
            dz,
            gate_dD,
            cu_seqlens,
            batch * tokens,
            num_states,
            H=heads,
            P=head_dim,
            BLOCK_T=GATE_ROWS,
            BLOCK_P=triton.next_power_of_2(head_dim),
            HAS_D=D is not None,
            D_PER_CHANNEL=D is not None and D.ndim == 2,
            IS_VARLEN=is_varlen,
        )
        d_output = d_core
        dD = None if gate_dD is None else gate_dD.sum(0)
    num_slots = num_chunk_slots(x, cu_seqlens)
    transform = {
        "DT_SOFTPLUS": dt_softplus,
        "DT_MIN": float(dt_min),
        "DT_MAX": float(dt_max),
        "HAS_DT_LIMIT": has_dt_limit(dt_min, dt_max),
        "HAS_DT_BIAS": dt_bias is not None,
    }
    shapes = {"H": heads, "G": groups, "P": head_dim, "N": state_dim, "BT": CHUNK_SIZE}

    step = transformed_dt(dt, dt_bias, dt_softplus, dt_min, dt_max)
    chunk_states = mamba2_chunk_states(x, step, A, B, initial_state, cu_seqlens)
    chunk_state_grads = torch.empty_like(chunk_states)
    d_initial_state = (
        x.new_empty(num_states, heads, head_dim, state_dim, dtype=torch.float32)
        if initial_state is not None
        else None
    )
    num_segments, segment_length = segment_plan(num_states * heads, tokens, x.device)
    segment_states = segment_decays = None
    if num_segments > 1:
        summaries = num_states * heads * num_segments
        segment_states = x.new_empty(summaries, head_dim, state_dim, dtype=torch.float32)
        segment_decays = x.new_empty(summaries, dtype=torch.float32)
    state_bwd_args = dict(
        dt=step,
        A=A,
        C=C,
        dy=d_output,
        segment_states=segment_states,
        segment_decays=segment_decays,
        cu_seqlens=cu_seqlens,
        T=tokens,
        segment_length=segment_length,
        IS_VARLEN=is_varlen,
        SEGMENTED=num_segments > 1,
        **shapes,
    )

    def state_grid(meta):
        return (num_states * heads, triton.cdiv(head_dim, meta["BP"]), num_segments)

    if num_segments > 1:
        _mamba2_state_bwd[state_grid](
            **state_bwd_args,
            d_final_state=None,
            d_initial_state=None,
            chunk_state_grads=None,
            HAS_FINAL_STATE_GRAD=False,
            STORE_INITIAL_STATE_GRAD=False,
            SUMMARY=True,
        )
    _mamba2_state_bwd[state_grid](
        **state_bwd_args,
        d_final_state=d_final_state,
        d_initial_state=d_initial_state,
        chunk_state_grads=chunk_state_grads,
        HAS_FINAL_STATE_GRAD=d_final_state is not None,
        STORE_INITIAL_STATE_GRAD=d_initial_state is not None,
        SUMMARY=False,
    )

    # Packed tokens past the terminal offset belong to no chunk and keep zero gradients.
    new_buffer = torch.zeros if is_varlen else torch.empty
    dx = new_buffer(x.shape, dtype=x.dtype, device=x.device)
    ddt = new_buffer(dt.shape, dtype=torch.float32, device=x.device)
    group_heads = heads_per_program(x, groups, num_slots)
    splits = heads // groups // group_heads
    dB_partial = new_buffer(splits, *B.shape, dtype=torch.float32, device=x.device)
    dC_partial = new_buffer(splits, *B.shape, dtype=torch.float32, device=x.device)
    dA_partial = x.new_zeros(num_slots, heads, dtype=torch.float32)
    reduce_d_grad = D is not None and dD is None
    dD_partial = x.new_zeros(num_slots, *D.shape, dtype=torch.float32) if reduce_d_grad else None
    _mamba2_chunk_bwd[(num_slots, groups * splits)](
        x=x,
        dt=step,
        A=A,
        B=B,
        C=C,
        D=D,
        dy=d_output,
        chunk_states=chunk_states,
        chunk_state_grads=chunk_state_grads,
        dx=dx,
        ddt=ddt,
        dB_partial=dB_partial,
        dC_partial=dC_partial,
        dA_partial=dA_partial,
        dD_partial=dD_partial,
        cu_seqlens=cu_seqlens,
        T=tokens,
        num_sequences=num_states,
        HEADS_PER_PROGRAM=group_heads,
        D_PER_CHANNEL=D is not None and D.ndim == 2,
        HAS_D=D is not None,
        REDUCE_D_GRAD=reduce_d_grad,
        IS_VARLEN=is_varlen,
        **shapes,
    )
    if reduce_d_grad:
        dD = dD_partial.sum(0)
    if has_dt_transform(dt_bias, dt_softplus, dt_min, dt_max):
        mamba2_dt_bwd_kernel[(triton.cdiv(dt.numel(), DT_BLOCK),)](
            ddt,
            dt,
            dt_bias,
            cu_seqlens,
            dt.numel(),
            num_states,
            H=heads,
            BLOCK=DT_BLOCK,
            IS_VARLEN=is_varlen,
            **transform,
        )

    empty = x.new_empty(0)
    return (
        dx,
        ddt.to(dt.dtype),
        dA_partial.sum(0).to(A.dtype),
        dB_partial.sum(0).to(B.dtype),
        dC_partial.sum(0).to(C.dtype),
        empty if D is None else dD.to(D.dtype),
        empty if dz is None else dz,
        empty if dt_bias is None else ddt.sum((0, 1)).to(dt_bias.dtype),
        empty if initial_state is None else d_initial_state.to(initial_state.dtype),
    )


__all__ = ["mamba2_chunk_bwd"]
