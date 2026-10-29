"""Triton forward for the chunked Mamba2 SSD scan.

``mamba2_chunk_fwd_kernel`` is the scan: one program owns one ``(sequence, head, P tile)`` and
keeps its FP32 ``[BP, N]`` state in registers while it walks the sequence in ``BT`` token chunks.
It stores every chunk's entry state, rounded to the input dtype exactly as the output MMA reads
it, into its chunk slot (see :func:`chunk_slot_base`), and the final state.
``mamba2_chunk_out_kernel`` then computes every chunk's pre-gate output ``Yc + D X`` in
parallel from those entry states: the masked ``C B^T`` intra-chunk product (shared by a group's
heads), the entry-state read, and the ``D`` skip. ``mamba2_gate_fwd_kernel`` applies the
``silu(z)`` gate elementwise; in the accumulator layout the gate's loads cost more than this
extra pass. The pre-gate output stays FP32, so the gated output is rounded once, as when the
gate reads the accumulator, and FP16 pre-gate values beyond the FP16 range cannot overflow a
finite gated output. Training keeps it so the backward can form ``dz`` and ``dy * silu(z)``
elementwise. The backward reruns the scan to recover the entry states.

The entry states are a transient ``[num_chunk_slots, H, P, N]`` tensor in the input dtype.

Paged prefill keeps the single-pass form (``STORE_OUTPUT``), which computes each chunk's output
inside the walk and so never materializes entry states.

With few ``(sequence, head)`` programs the walk is latency-bound, so :func:`segment_plan` splits
each sequence into segments that run in parallel. A ``SUMMARY`` launch scans every segment but
the last from a zero state and stores its end state and total log2 decay; the main launch then
enters segment ``s`` with ``decay_{s-1} * (... (decay_0 * S_0 + local_0) ...) + local_{s-1}``.
Every segment re-combines the summaries before it, so the segment count is capped.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from attn_gym._backends.triton.tune import TritonTuner
from attn_gym._backends.triton.utils import LOG2_E
from attn_gym.linear._delta_rule.triton.paged_state import resolve_paged_state

CHUNK_SIZE = 64
# Chunk-parallel kernels split a group's heads until they launch this many programs per SM.
PROGRAMS_PER_SM = 4
# Segments are added until the scan launches this many (sequence, head) programs per SM.
SEGMENT_PROGRAMS_PER_SM = 4
MIN_SEGMENT_CHUNKS = 8
MAX_SEGMENTS = 16


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
        # ``log1p`` keeps small step sizes that ``log(1 + exp(dt))`` rounds to zero below -16.
        dt = tl.where(dt > 20.0, dt, libdevice.log1p(libdevice.exp(dt)))
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


@triton.jit
def scale_by_exp2(value, log2_scale):
    """``value * 2**log2_scale`` for a decay (``log2_scale <= 0``) without early underflow.

    A decay can underflow FP32 on its own while the decayed value stays representable. For
    nonpositive scales, up to two split-off factors of at least ``2**-126`` reach every nonzero
    FP32 product of a finite value; a scale at or above ``-126`` computes exactly what one
    ``exp2`` and multiply would. Growth factors (positive scales) get no such protection.
    """
    first = tl.maximum(log2_scale, -126.0)
    rest = log2_scale - first
    second = tl.maximum(rest, -126.0)
    return value * tl.math.exp2(first) * tl.math.exp2(second) * tl.math.exp2(rest - second)


@triton.jit
def decay_by_exp2(value, log2_scale, steep):
    """:func:`scale_by_exp2`, splitting the decay only when the program-uniform ``steep`` is set.

    Set ``steep`` when some scale may fall below ``-126``; at or above it one ``exp2`` and
    multiply give the same result, so ordinary chunks skip the split factors' multiplies. With
    nonnegative step sizes a chunk's cumulative log2 decay is monotone, so ``g_last < -126``
    finds every steep chunk; a negative step can hide one, which then decays as one ``exp2``.
    """
    if steep:
        result = scale_by_exp2(value, log2_scale)
    else:
        result = value * tl.math.exp2(log2_scale)
    return result


@triton.jit
def segment_ptrs(sequence, head, segment, offs_p, offs_n, H: tl.constexpr, P, N: tl.constexpr):
    """Index of one segment summary and its ``[BP, N]`` state pointers."""
    index = (sequence.to(tl.int64) * H + head) * tl.num_programs(2) + segment
    return index, index * (P * N) + offs_p[:, None] * N + offs_n[None, :]


@triton.jit(do_not_specialize=["T", "segment_length"])
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
    segment_states,
    segment_decays,
    state_indices,
    has_initial_state,
    cu_seqlens,
    T,
    state_stride_slot,
    segment_length,
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
    segment_end = tl.minimum(segment_start + segment_length, length)
    if SUMMARY:
        # The last segment's summary is never read.
        if segment_end >= length:
            return
    elif segment > 0 and segment_start >= length:
        return

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
    if HAS_INITIAL_STATE and not SUMMARY:
        state = tl.load(initial_state + state_ptrs, mask=mask_p[:, None] & load_initial, other=0.0)
        state = state.to(tl.float32)
    else:
        state = tl.zeros([BP, N], dtype=tl.float32)
    if SEGMENTED and not SUMMARY:
        for previous in range(segment):
            index, ptrs = segment_ptrs(sequence, head, previous, offs_p, offs_n, H, P, N)
            local = tl.load(segment_states + ptrs, mask=mask_p[:, None], other=0.0)
            state = scale_by_exp2(state, tl.load(segment_decays + index)) + local
    total_decay = 0.0
    if STORE_CHUNK_STATES:
        slot_base = chunk_slot_base(cu_seqlens, sequence, bos, T, BT, IS_VARLEN)
        chunk_state_ptrs = head * P * N + offs_p[:, None] * N + offs_n[None, :]

    for chunk_start in range(segment_start, segment_end, BT):
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
        total_decay += g_last
        steep = g_last < -126.0

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
            output += decay_by_exp2(entry, g[:, None], steep)
            if HAS_D:
                output += x_chunk.to(tl.float32) * skip
            if HAS_Z:
                gate = tl.load(z + xp, mask=mask_xp, other=0.0).to(tl.float32)
                output *= gate * tl.sigmoid(gate)
            tl.store(y + xp, output.to(y.dtype.element_ty), mask=mask_xp)

        if steep:
            # Decay the write itself: ``dt * exp2(g_L - g)`` alone can underflow (``x * dt``
            # must fit FP32).
            scaled_x = x_chunk.to(tl.float32) * dt_chunk[:, None]
            scaled_x = scale_by_exp2(scaled_x, (g_last - g)[:, None])
        else:
            write_scale = dt_chunk * tl.math.exp2(g_last - g)
            scaled_x = x_chunk.to(tl.float32) * write_scale[:, None]
        scaled_x = scaled_x.to(x_chunk.dtype)
        state = scale_by_exp2(state, g_last) + tl.dot(tl.trans(scaled_x), b_chunk)

    if SUMMARY:
        index, ptrs = segment_ptrs(sequence, head, segment, offs_p, offs_n, H, P, N)
        tl.store(segment_states + ptrs, state, mask=mask_p[:, None])
        if p_tile == 0:
            tl.store(segment_decays + index, total_decay)
    if STORE_FINAL_STATE and segment_end >= length:
        tl.store(
            final_state + state_ptrs,
            state.to(final_state.dtype.element_ty),
            mask=mask_p[:, None],
        )


@triton.jit
def locate_chunk(slot, cu_seqlens, num_sequences, T, BT: tl.constexpr, IS_VARLEN: tl.constexpr):
    """Return ``(bos, length, chunk)`` of the chunk stored in ``slot``."""
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
    return bos, length, chunk


@triton.jit(do_not_specialize=["T", "num_sequences"])
def mamba2_chunk_out_kernel(
    x,
    dt,
    A,
    dt_bias,
    B,
    C,
    D,
    y,
    chunk_states,
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
    IS_VARLEN: tl.constexpr,
):
    """Chunk-parallel pre-gate output from stored entry states, per chunk of one group."""
    slot, group_split = tl.program_id(0), tl.program_id(1)
    SPLITS: tl.constexpr = H // G // HEADS_PER_PROGRAM
    group, split = group_split // SPLITS, group_split % SPLITS
    bos, length, chunk = locate_chunk(slot, cu_seqlens, num_sequences, T, BT, IS_VARLEN)
    chunk_start = chunk * BT
    if chunk_start >= length:
        return

    token0 = bos + chunk_start
    offs_t = tl.arange(0, BT)
    offs_n = tl.arange(0, N)
    mask_t = chunk_start + offs_t < length
    causal = offs_t[:, None] >= offs_t[None, :]
    bc = token0 * (G * N) + group * N + (offs_t[:, None] * (G * N) + offs_n[None, :])
    b_chunk = tl.load(B + bc, mask=mask_t[:, None], other=0.0)
    c_chunk = tl.load(C + bc, mask=mask_t[:, None], other=0.0)
    # The group's heads share C B^T.
    scores = tl.dot(c_chunk, tl.trans(b_chunk))
    slot64 = slot.to(tl.int64)

    for head_index in range(HEADS_PER_PROGRAM):
        head = group * (H // G) + split * HEADS_PER_PROGRAM + head_index
        decay_rate = tl.load(A + head).to(tl.float32) * LOG2_E
        bias = tl.load(dt_bias + head).to(tl.float32) if HAS_DT_BIAS else 0.0
        dt_chunk = tl.load(dt + token0 * H + head + offs_t * H, mask=mask_t, other=0.0)
        dt_chunk = transform_dt(
            dt_chunk.to(tl.float32), bias, DT_SOFTPLUS, DT_MIN, DT_MAX, HAS_DT_LIMIT
        )
        dt_chunk = tl.where(mask_t, dt_chunk, 0.0)
        g = tl.cumsum(dt_chunk * decay_rate, axis=0)
        decay = tl.math.exp2(tl.where(causal, g[:, None] - g[None, :], float("-inf")))
        weights = (scores * decay * dt_chunk[None, :]).to(c_chunk.dtype)
        steep = tl.min(g, axis=0) < -126.0
        state_base = (slot64 * H + head) * (P * N)
        for p_start in range(0, P, BP):
            offs_p = p_start + tl.arange(0, BP)
            mask_p = offs_p < P
            xp = token0 * (H * P) + head * P + (offs_t[:, None] * (H * P) + offs_p[None, :])
            mask_xp = mask_t[:, None] & mask_p[None, :]
            x_chunk = tl.load(x + xp, mask=mask_xp, other=0.0)
            state_ptrs = state_base + offs_p[:, None] * N + offs_n[None, :]
            state = tl.load(chunk_states + state_ptrs, mask=mask_p[:, None], other=0.0)
            output = tl.dot(weights, x_chunk)
            output += decay_by_exp2(tl.dot(c_chunk, tl.trans(state)), g[:, None], steep)
            if HAS_D:
                # A second load lets the residual take the accumulator layout directly.
                residual = tl.load(x + xp, mask=mask_xp, other=0.0).to(tl.float32)
                if D_PER_CHANNEL:
                    skip = tl.load(D + head * P + offs_p, mask=mask_p, other=0.0)
                    output += residual * skip.to(tl.float32)[None, :]
                else:
                    output += residual * tl.load(D + head).to(tl.float32)
            tl.store(y + xp, output.to(y.dtype.element_ty), mask=mask_xp)


GATE_BLOCK = 2048


# ``numel`` stays specialized: its divisibility by 16 lets the masked accesses vectorize.
@triton.jit
def mamba2_gate_fwd_kernel(pre_gate, z, y, numel, BLOCK: tl.constexpr):
    """``y = pre_gate * silu(z)`` elementwise."""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    output = tl.load(pre_gate + offs, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(z + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(y + offs, (output * gate * tl.sigmoid(gate)).to(y.dtype.element_ty), mask=mask)


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
        a["SUMMARY"],
        # Few (sequence, head) programs favor splitting P; bucket the program count.
        triton.next_power_of_2(
            a["H"] * (a["cu_seqlens"].shape[0] - 1 if a["IS_VARLEN"] else a["x"].shape[0])
        ),
    ),
    prune=prune_p_tiles,
    reset=restore_paged_state,
)


_mamba2_chunk_out = TritonTuner(
    mamba2_chunk_out_kernel,
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
        triton.next_power_of_2(a["chunk_states"].shape[0]),
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


def segment_plan(num_programs: int, tokens: int, device: torch.device) -> tuple[int, int]:
    """Return ``(num_segments, segment_length)`` for a chunk scan of ``num_programs`` walks.

    ``tokens`` bounds every sequence's length, so ``num_segments`` segments of
    ``segment_length`` tokens (a multiple of the chunk size) cover each sequence. Packed inputs
    plan from their capacity, so short packed sequences leave later segments idle.
    """
    target = (
        SEGMENT_PROGRAMS_PER_SM * torch.cuda.get_device_properties(device).multi_processor_count
    )
    most = min(max(tokens // (MIN_SEGMENT_CHUNKS * CHUNK_SIZE), 1), MAX_SEGMENTS)
    wanted = min(triton.cdiv(target, num_programs), most)
    if wanted <= 1:
        return 1, max(tokens, 1)
    length = triton.cdiv(triton.cdiv(tokens, wanted), CHUNK_SIZE) * CHUNK_SIZE
    return triton.cdiv(tokens, length), length


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
    # A paged pool is read and written in place, so its walks stay whole.
    num_segments, segment_length = (
        (1, max(tokens, 1))
        if state_indices is not None
        else segment_plan(num_states * heads, tokens, x.device)
    )
    segment_states = segment_decays = None
    if num_segments > 1:
        summaries = num_states * heads * num_segments
        segment_states = x.new_empty(summaries, head_dim, state_dim, dtype=torch.float32)
        segment_decays = x.new_empty(summaries, dtype=torch.float32)

    def grid(meta):
        return (num_states * heads, triton.cdiv(head_dim, meta["BP"]), num_segments)

    common = {
        "x": x,
        "dt": dt,
        "A": A,
        "dt_bias": dt_bias,
        "B": B,
        "C": C,
        "D": D,
        "z": z,
        "initial_state": initial_state,
        "segment_states": segment_states,
        "segment_decays": segment_decays,
        "state_indices": state_indices,
        "has_initial_state": has_initial_state,
        "cu_seqlens": cu_seqlens,
        "T": tokens,
        "state_stride_slot": (
            initial_state.stride(0) if initial_state is not None else heads * head_dim * state_dim
        ),
        "segment_length": segment_length,
        "H": heads,
        "G": groups,
        "P": head_dim,
        "N": state_dim,
        "BT": CHUNK_SIZE,
        "DT_SOFTPLUS": dt_softplus,
        "DT_MIN": float(dt_min),
        "DT_MAX": float(dt_max),
        "HAS_DT_LIMIT": has_dt_limit(dt_min, dt_max),
        "HAS_DT_BIAS": dt_bias is not None,
        "D_PER_CHANNEL": D is not None and D.ndim == 2,
        "HAS_D": D is not None,
        "HAS_Z": z is not None,
        "HAS_INITIAL_STATE": initial_state is not None,
        "USE_STATE_INDICES": state_indices is not None,
        "USE_HAS_INITIAL_STATE": has_initial_state is not None,
        "IS_VARLEN": cu_seqlens is not None,
        "SEGMENTED": num_segments > 1,
    }
    if num_segments > 1:
        _mamba2_chunk_fwd[grid](
            **common,
            y=None,
            final_state=None,
            chunk_states=None,
            STORE_FINAL_STATE=False,
            STORE_OUTPUT=False,
            STORE_CHUNK_STATES=False,
            SUMMARY=True,
        )
    _mamba2_chunk_fwd[grid](
        **common,
        y=y,
        final_state=final_state,
        chunk_states=chunk_states,
        STORE_FINAL_STATE=final_state is not None,
        STORE_OUTPUT=y is not None,
        STORE_CHUNK_STATES=chunk_states is not None,
        SUMMARY=False,
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
    store_pre_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Return the output, the FP32 final state if requested, and the pre-gate output if stored.

    The FP32 pre-gate output ``Yc + D X`` is the gate-backward residual; it is returned only
    with a gate ``z`` and ``store_pre_gate``.
    """
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
    pre_gate = None if z is None else x.new_empty(x.shape, dtype=torch.float32)
    chunk_states = mamba2_chunk_states(
        x,
        dt,
        A,
        B,
        initial_state,
        dt_bias,
        cu_seqlens,
        dt_softplus,
        dt_min,
        dt_max,
        final_state=final_state,
    )
    groups = B.shape[2]
    num_slots = chunk_states.shape[0]
    group_heads = heads_per_program(x, groups, num_slots)
    splits = heads // groups // group_heads
    # With a gate the output kernel writes the FP32 pre-gate output for the gate kernel.
    _mamba2_chunk_out[(num_slots, groups * splits)](
        x=x,
        dt=dt,
        A=A,
        dt_bias=dt_bias,
        B=B,
        C=C,
        D=D,
        y=y if pre_gate is None else pre_gate,
        chunk_states=chunk_states,
        cu_seqlens=cu_seqlens,
        T=x.shape[1],
        num_sequences=num_states,
        H=heads,
        G=groups,
        P=head_dim,
        N=state_dim,
        BT=CHUNK_SIZE,
        HEADS_PER_PROGRAM=group_heads,
        DT_SOFTPLUS=dt_softplus,
        DT_MIN=float(dt_min),
        DT_MAX=float(dt_max),
        HAS_DT_LIMIT=has_dt_limit(dt_min, dt_max),
        HAS_DT_BIAS=dt_bias is not None,
        D_PER_CHANNEL=D is not None and D.ndim == 2,
        HAS_D=D is not None,
        IS_VARLEN=cu_seqlens is not None,
    )
    if z is not None:
        mamba2_gate_fwd_kernel[(triton.cdiv(y.numel(), GATE_BLOCK),)](
            y if pre_gate is None else pre_gate, z, y, y.numel(), BLOCK=GATE_BLOCK
        )
    return y, final_state, pre_gate if store_pre_gate else None


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
    final_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute every chunk's entry state, ``[num_chunk_slots, H, P, N]`` in ``x.dtype``.

    Slots of packed capacity that no chunk owns stay uninitialized. ``final_state``, if given,
    receives each sequence's FP32 final state.
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
        final_state=final_state,
        chunk_states=chunk_states,
    )
    return chunk_states


__all__ = [
    "CHUNK_SIZE",
    "GATE_BLOCK",
    "chunk_slot_base",
    "decay_by_exp2",
    "has_dt_limit",
    "heads_per_program",
    "locate_chunk",
    "mamba2_chunk_fwd",
    "mamba2_chunk_states",
    "mamba2_paged_chunk_fwd",
    "num_chunk_slots",
    "prune_p_tiles",
    "scale_by_exp2",
    "segment_plan",
    "segment_ptrs",
    "transform_dt",
]
