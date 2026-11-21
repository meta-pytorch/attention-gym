# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/kernel/kda_*_f16.py
# pointer loads and barrier wrappers), rewritten by Attention Gym in 2026 for the warpdrive KDA
# forward ports.

"""Helpers shared by the warpdrive KDA kernels: raw TMEM-handoff barrier ops, swizzled SMEM
offsets, the upstream pointer loads and stores, the dtype maps and the work-item row layout."""

import cutlass
import torch
from cutlass import Float32, Int32, Int64, cute
from cutlass.cute.arch import inline_ptx
from cutlass.cute.nvgpu import tcgen05

from attn_gym.warpdrive.thread_sync import tcgen05_after_thread_sync, tcgen05_before_thread_sync

from .ptx import (
    elect_sync,
    f16x2_to_f32,
    fmul2,
    fp32_to_fp16,
    ldmatrix,
    mma_step,
    mma_step_tf32,
    st_shared_v4,
    tf32_round,
    tmem_st,
)
from .ragged import RaggedTma, ragged_load

B_T = 16
A_WORDS = B_T * B_T // 2  # int32 words of one A record (a 32-byte swizzled 16 x 16 b16 image)
CUTE_DTYPES = {"fp16": cutlass.Float16, "bf16": cutlass.BFloat16, "fp32": cutlass.Float32}
TORCH_NAMES = {torch.float16: "fp16", torch.bfloat16: "bf16", torch.float32: "fp32"}
# Work-item row (bounds in chunks): batch, head x value tile, write_start, write_end,
# compute_start, compute_end, batch_start, batch_end (tokens), final_dst, dstate_dst.
WORK_ITEM_FIELDS = 10
FINAL_DST = 8


@cute.jit
def load_rows(
    load: RaggedTma,
    dtype: cutlass.Constexpr,
    addr,
    layout,
    rows,
    head,
    col_box,
    cols: cutlass.Constexpr[int],
    bar,
    desc,
):
    """TMA column box ``col_box`` (``cols`` wide) of the tokens ``rows = (token, valid)`` of head
    ``head`` into the 16-row box of ``layout`` at SMEM byte address ``addr``; rows past
    ``valid`` load as zeros, as from upstream's per-sequence descriptors. ``desc``: the item's
    ``ragged.sequence_descriptors`` entry for ``load``."""
    dst = cute.make_tensor(
        cute.make_ptr(dtype, addr, cute.AddressSpace.smem, assumed_align=1024), layout
    )
    token, valid = rows
    ragged_load(load, dst, token, valid, head, col_box, bar, B_T, cols, desc)


@cute.jit
def spin_wait(bar, phase, fence: cutlass.Constexpr[bool] = True):
    """Hint-less ``mbarrier.try_wait.parity`` loop (the upstream kernels' ``spin=True`` wait):
    ptxas keeps it a ``SYNCS.PHASECHK`` retry branch, without the ``NANOSLEEP`` of the
    suspend-hint form that ``cute.arch.mbarrier_wait`` (and so a channel wait) emits. Then
    ``tcgen05.fence::after_thread_sync``, which orders the caller's later tcgen05 operations
    after the wait (PTX ISA 9.7.18.6.3), as a managed wait in a TMEM role does; ``fence=False``
    omits it where a later wait's fence precedes the caller's next tcgen05 operation
    (``AFTER_FENCE_COVERED``)."""
    inline_ptx(
        "{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [$0], $1;\n\t"
        "@P1 bra.uni DONE;\n\tbra.uni LAB_WAIT;\n\tDONE:\n\t}",
        read_only_args=[Int32(bar.toint()), Int32(phase)],
    )
    if cutlass.const_expr(fence):
        tcgen05_after_thread_sync()


@cute.jit
def warp_arrive(bar, fence: cutlass.Constexpr[bool] = True):
    """One arrival per warp after all its lanes (the ``Warps`` signal): every lane fences its
    tcgen05 operations (``tcgen05.fence::before_thread_sync``; ``fence=False`` omits it where an
    earlier fence already follows them, ``BEFORE_FENCE_COVERED``), then the warp sync orders
    every lane's prior accesses and completion waits before the elected lane's arrive."""
    if cutlass.const_expr(fence):
        tcgen05_before_thread_sync()
    cute.arch.sync_warp()
    if elect_sync():
        cute.arch.mbarrier_arrive(bar)


@cute.jit
def umma_commit(bar, elect):
    """``tcgen05.commit`` of the issuing lane's MMAs to ``bar``."""
    if elect:
        cute.nvgpu.tcgen05.commit(bar)


@cute.jit
def swz128(row, col, elem_bytes: cutlass.Constexpr[int] = 2):
    """Element column of ``(row, col)`` in a 128-byte swizzled row."""
    return col ^ ((row & 7) * (16 // elem_bytes))


@cute.jit
def box128(row, col):
    """16-bit ``(row, col)`` in segment-major 128-byte swizzled ``[16 x 64]`` boxes."""
    box = col // 64
    return box * (B_T * 64) + row * 64 + swz128(row, col - box * 64)


@cute.jit
def swz32(row, col):
    """16-bit column of ``(row, col)`` in a 32-byte swizzled 16 x 16 tile."""
    return ((col // 8) ^ ((row >> 2) & 1)) * 8 + col % 8


@cute.jit
def opaque_i32(value):
    """Identity ``mov.b32`` that keeps a per-lane offset in its own register."""
    return inline_ptx("mov.b32 $0, $1;", write_only_types=[Int32], read_only_args=[Int32(value)])


@cute.jit
def lds_f32(addr):
    return inline_ptx(
        "ld.shared.f32 $0, [$1];", write_only_types=[Float32], read_only_args=[Int32(addr)]
    )


@cute.jit
def lds_16_f32(addr, dtype: cutlass.Constexpr):
    """One 16-bit element widened to fp32 (exact)."""
    if cutlass.const_expr(dtype is cutlass.BFloat16):
        return inline_ptx(
            "{ .reg .b16 h; ld.shared.b16 h, [$1]; mov.b32 $0, {0, h}; }",
            write_only_types=[Float32],
            read_only_args=[Int32(addr)],
        )
    return inline_ptx(
        "{ .reg .b16 h; ld.shared.b16 h, [$1]; cvt.f32.f16 $0, h; }",
        write_only_types=[Float32],
        read_only_args=[Int32(addr)],
    )


@cute.jit
def lds_v2_f32(addr):
    return list(
        inline_ptx(
            "ld.shared.v2.f32 {$0, $1}, [$2];",
            write_only_types=[Float32] * 2,
            read_only_args=[Int32(addr)],
        )
    )


@cute.jit
def lds_v4_f32(addr):
    return list(
        inline_ptx(
            "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
            write_only_types=[Float32] * 4,
            read_only_args=[Int32(addr)],
        )
    )


@cute.jit
def lds_v4(addr):
    return list(
        inline_ptx(
            "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
            write_only_types=[Int32] * 4,
            read_only_args=[Int32(addr)],
        )
    )


@cute.jit
def sts_f32(addr, value):
    inline_ptx("st.shared.f32 [$0], $1;", read_only_args=[Int32(addr), value])


@cute.jit
def ldg_v4(addr):
    return list(
        inline_ptx(
            "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
            write_only_types=[Int32] * 4,
            read_only_args=[Int64(addr)],
        )
    )


@cute.jit
def stg_v4(addr, words):
    inline_ptx("st.global.v4.b32 [$0], {$1, $2, $3, $4};", read_only_args=[Int64(addr), *words])


@cute.jit
def as_f32(word):
    return inline_ptx("mov.b32 $0, $1;", write_only_types=[Float32], read_only_args=[word])


@cute.jit
def as_i32(value):
    return inline_ptx("mov.b32 $0, $1;", write_only_types=[Int32], read_only_args=[value])


def state_align_suffix(state_align: int) -> str:
    """Kernel-name tag of a state row alignment; empty for the default 16 bytes."""
    return "" if state_align == 16 else f"_a{state_align}"


# NOTE [Unaligned state rows]
# A paged pool carved from packed byte pages may place a state row at any element offset, so
# its 16-byte blocks start at a per-row runtime misalignment ``m = addr & 15``. Each thread owns
# one row, so element loads and stores would split every warp access into 32 scattered words
# (measured up to 2.6x the aligned prefill time on state-bound GB300 calls). Instead the row
# keeps its 16-byte vector accesses on the enclosing aligned blocks and realigns in registers:
# words move by ``m // 4`` through two levels of selects and, for an odd BF16 offset, by one
# more half word through a funnel shift (measured 1.0-1.2x the aligned time). Loads read only
# blocks that hold a byte of the row, so they never leave its allocation. Stores write the
# interior blocks whole and the two partial end blocks element by element, so no byte outside
# the row is written.


def _shift_word(words, j: int, odd, high):
    """``words[j + shift]`` for a runtime ``shift`` in ``[0, 3]`` given as its two bits."""
    low_pair = cutlass.select_(odd, words[j + 1], words[j])
    high_pair = cutlass.select_(odd, words[j + 3], words[j + 2])
    return cutlass.select_(high, high_pair, low_pair)


@cute.jit
def shift_words(words, shift, count: cutlass.Constexpr[int]):
    """``words[j + shift]`` for ``j < count``, with ``shift`` a runtime value in ``[0, 3]``."""
    odd, high = (shift & 1) != 0, (shift & 2) != 0
    # A comprehension, not range_constexpr: half-row loads shift 65 words, past its unroll warning.
    return [_shift_word(words, j, odd, high) for j in range(count)]


@cute.jit
def funnel_half(low, high):
    """The 32-bit word that starts half way into ``low`` (``(high:low) >> 16``)."""
    return inline_ptx(
        "shf.r.clamp.b32 $0, $1, $2, 16;", write_only_types=[Int32], read_only_args=[low, high]
    )


@cute.jit
def realign_words(words, misalignment, count: cutlass.Constexpr[int]):
    """``count`` words of the byte stream that starts ``misalignment`` (runtime, a multiple of
    two below 16) bytes into ``words``; needs ``count + 4`` words."""
    shifted = shift_words(words, misalignment >> 2, count + 1)
    half = ((misalignment >> 1) & 1) != 0
    return [
        cutlass.select_(half, funnel_half(shifted[j], shifted[j + 1]), shifted[j])
        for j in range(count)
    ]


@cute.jit
def load_state(tile, align: cutlass.Constexpr[int] = 16):
    """The fp32 or bf16 state values of the global ``tile`` (a multiple of 16 bytes) as fp32.
    Below 16-byte ``align``, the row is realigned in registers (NOTE [Unaligned state rows])."""
    dtype = tile.element_type
    blocks = cute.size(tile) * (dtype.width // 8) // 16
    addr = tile.iterator.toint()
    words = []
    if cutlass.const_expr(align < 16):
        misalignment = addr & 15
        base = addr - misalignment
        for g in cutlass.range_constexpr(blocks):
            words += ldg_v4(base + g * 16)
        # The values spill into one more block unless the tile is aligned; reread the last one
        # then, so the load stays on blocks of the row.
        words += ldg_v4(
            base + blocks * 16 - cutlass.select_(misalignment == 0, Int64(16), Int64(0))
        )
        words = realign_words(words, misalignment, 4 * blocks)
    else:
        for g in cutlass.range_constexpr(blocks):
            words += ldg_v4(addr + g * 16)
    vals = []
    for w in words:
        if cutlass.const_expr(dtype is cutlass.Float32):
            vals.append(as_f32(w))
        else:
            lo, hi = f16x2_to_f32(w, dtype=dtype)
            vals += [lo, hi]
    return vals


@cute.jit
def store_unaligned_state_32(tile, vals):
    """Store 32 fp32 ``vals`` into the element-aligned global ``tile`` (NOTE [Unaligned state
    rows])."""
    dtype = tile.element_type
    element_bytes = dtype.width // 8
    if cutlass.const_expr(dtype is Float32):
        words = [as_i32(v) for v in vals]
    else:
        words = [fp32_to_fp16(vals[2 * j], vals[2 * j + 1], dtype=dtype) for j in range(16)]
    blocks = len(words) // 4
    addr = tile.iterator.toint()
    misalignment = addr & 15
    base = addr - misalignment
    if misalignment == 0:
        for g in cutlass.range_constexpr(blocks):
            stg_v4(base + g * 16, words[4 * g : 4 * g + 4])
    else:
        # A head of 16 - misalignment bytes up to the first 16-byte boundary, blocks - 1 full
        # blocks from base + 16 holding the values from that byte on, and a tail of
        # misalignment bytes.
        interior = realign_words(words, 16 - misalignment, len(words) - 4)
        for g in cutlass.range_constexpr(blocks - 1):
            stg_v4(base + (g + 1) * 16, interior[4 * g : 4 * g + 4])
        head_elems = (16 - misalignment) // element_bytes
        tail_elems = misalignment // element_bytes
        for e in cutlass.range_constexpr(16 // element_bytes - 1):
            if e < head_elems:
                tile[e] = vals[e].to(dtype)
            if e < tail_elems:
                tile[31 - e] = vals[31 - e].to(dtype)


@cute.jit
def store_state_32(tile, vals, align: cutlass.Constexpr[int] = 16):
    """Store 32 fp32 ``vals`` into the global ``tile`` as its dtype (fp32, or bf16 rounded to
    nearest even). Below 16-byte ``align``, the row is realigned in registers (NOTE [Unaligned
    state rows])."""
    if cutlass.const_expr(align < 16):
        store_unaligned_state_32(tile, vals)
    else:
        dtype = tile.element_type
        addr = tile.iterator.toint()
        if cutlass.const_expr(dtype is Float32):
            for g in cutlass.range_constexpr(8):
                stg_v4(addr + g * 16, [as_i32(vals[g * 4 + t]) for t in range(4)])
        else:
            for g in cutlass.range_constexpr(4):
                words = [
                    fp32_to_fp16(vals[g * 8 + 2 * t], vals[g * 8 + 2 * t + 1], dtype=dtype)
                    for t in range(4)
                ]
                stg_v4(addr + g * 16, words)


@cute.jit
def beta_residual(
    v_word, beta0, beta1, ks0, ks1, dtype: cutlass.Constexpr, with_state: cutlass.Constexpr
):
    """``beta * (v - ks)`` for one packed V pair, in fp32."""
    v0, v1 = f16x2_to_f32(v_word, dtype=dtype)
    if cutlass.const_expr(with_state):
        v0, v1 = cute.arch.add_packed_f32x2((v0, v1), (-ks0, -ks1))
    return fmul2(beta0, beta1, v0, v1)


@cute.jit
def mask_pair(word, token0, valid):
    """Zero the halves of a packed (token0, token0 + 1) pair at or past ``valid`` tokens."""
    word = word if token0 + 1 < valid else word & Int32(0x0000FFFF)
    return word if token0 < valid else Int32(0)


@cute.jit
def resolve_paged_state(batch, state_indices, has_initial_state):
    """``(slot, active, load_initial, clear_empty)``: non-positive routes are null; the optional
    byte mask marks resumed (nonzero) versus fresh slots."""
    slot = batch
    active = cutlass.Boolean(True)
    load_initial = cutlass.Boolean(True)
    clear_empty = cutlass.Boolean(False)
    if cutlass.const_expr(state_indices is not None):
        route = state_indices[batch]
        active = route > 0
        slot = route if active else Int32(0)
        load_initial = active
        if cutlass.const_expr(has_initial_state is not None):
            load_initial = active and has_initial_state[batch] != 0
            clear_empty = active and has_initial_state[batch] == 0
    return slot, active, load_initial, clear_empty


@cute.jit
def raw_beta_wait(
    raw, beta, n, stages: cutlass.Constexpr[int], fence: cutlass.Constexpr[bool] = True
):
    """Waits on chunk ``n``'s stages of two ``stages``-deep rings that a compute group consumes
    once per chunk, in order, driven by one shared counter through the end accessors, then
    ``tcgen05.fence::after_thread_sync`` as after a managed wait in a TMEM role (``fence`` as
    for ``spin_wait``). Returns the stage."""
    stage = n & (stages - 1)
    phase = (n // stages) & 1
    cute.arch.mbarrier_wait(raw.full_barrier(stage), phase)
    cute.arch.mbarrier_wait(beta.full_barrier(stage), phase)
    if cutlass.const_expr(fence):
        tcgen05_after_thread_sync()
    return stage


@cute.jit
def raw_beta_release(raw, beta, stage):
    """Release ``stage`` of both rings as their managed ``Warps`` consumer would: every lane runs
    the raw ring's derived ``smem_async`` release fence (generic reads before the TMA refill),
    the warp syncs, one lane arrives on each empty barrier."""
    cute.arch.fence_view_async_shared()
    cute.arch.sync_warp()
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(raw.empty_barrier(stage))
        cute.arch.mbarrier_arrive(beta.empty_barrier(stage))


@cute.jit
def ldg_l2_f32(addr, dtype: cutlass.Constexpr):
    """One fp32 or 16-bit element at global ``addr`` widened to fp32 (exact), loaded with
    ``.cg`` (cached in L2 only): a per-chunk scalar load that allocated L1 lines next to the
    TMA ring made the KDA summary bimodal (6-18% slower) depending on the sequence offsets."""
    if cutlass.const_expr(dtype is Float32):
        return inline_ptx(
            "ld.global.cg.f32 $0, [$1];", write_only_types=[Float32], read_only_args=[Int64(addr)]
        )
    if cutlass.const_expr(dtype is cutlass.BFloat16):
        return inline_ptx(
            "{ .reg .b16 h; ld.global.cg.b16 h, [$1]; mov.b32 $0, {0, h}; }",
            write_only_types=[Float32],
            read_only_args=[Int64(addr)],
        )
    return inline_ptx(
        "{ .reg .b16 h; ld.global.cg.b16 h, [$1]; cvt.f32.f16 $0, h; }",
        write_only_types=[Float32],
        read_only_args=[Int64(addr)],
    )


def gate_tile(d_k: int):
    """Layout of the in-place exp2(g) gate exchange in fp32 elements, indexed by ``(row,
    channel)`` after ``gate_offset``'s segment fold: one ``[16, 32]`` tile of 128-byte rows per
    32-channel segment, whose 16-byte chunks ``Swizzle<3, 2, 3>`` XORs with ``row & 7``."""
    rows = cute.make_layout((B_T, (4, 8, d_k // 32)), stride=(32, (1, 4, B_T * 32)))
    return cute.make_composed_layout(cute.make_swizzle(3, 2, 3), 0, rows)


@cute.jit
def gate_offset(row, dim, d_k: cutlass.Constexpr[int]):
    """Byte offset of exp2(g) ``(row, dim)`` in the gate exchange, where chunk ``c`` of segment
    ``s`` sits at ``c ^ (row ^ s) & 7`` so a row's eight-lane reads span two segments without
    bank conflicts. A CuTe swizzle XORs one bit field, so the segment's part is folded into the
    column before ``gate_tile`` applies the row's."""
    return 4 * cute.crd2idx((row, dim ^ ((dim >> 5) << 2)), gate_tile(d_k))


# ---- Operand precision options -----------------------------------------------------------
# NOTE [KDA operand precision]
# The public ``intra_chunk_dtypes`` option ("<decay products>+<solve operands>", one name for
# both) selects the operand dtypes of the intra-chunk matmuls, which the forward kernels (and the
# backward's checkpoint recompute) take as two constructor arguments, each a dtype name:
# ``decay_dtype`` ("tf32" or the io dtype) for the operands of K_decay K_inv^T (L) and
# Q_decay K_inv^T (A), and ``solve_dtype`` ("fp16" or the io dtype) for T^-1 (and its 16 x 16
# inverse), Y, U and A in U = T^-1 Y and O += A U. The inter-chunk state matmuls (K_decay S,
# Q_decay S, K_restore^T U) always take the io dtype. ``operand_dtypes`` maps the option to the
# two choices; it is the only place that does. fp16 io always takes tf32 tiles (2^-g overflows
# fp16) and its solve operands are fp16 already, so it has one kernel set, the unchanged fp16
# kernels, which ``operand_dtypes`` returns for every value: the public entry points
# (``resolve_intra_chunk_dtypes`` in ``attn_gym.linear.kda.validation``) select "tf32+fp16"
# for fp16 when the option is omitted and reject any other value, and the op schemas' default
# "bf16" (the pre-option kernels of graphs serialized before the option) then means the same
# kernels for both io dtypes.
#
# fp16 solve operands under bf16 io keep the state path (the S operand, K_restore and the U of
# the state update) in bf16 and guard the fp16 range of Y and U with exact power-of-two scales
# (``store_solve_rows``): a warp packs its rows of Y unscaled when every finite value is below
# 2^15 and every 4-value row segment a lane holds is zero or reaches the fp16 normal range (a
# subnormal's rounding error is then at most 2^-11 of the largest value packed beside it);
# otherwise it repacks them times the one power of two that brings its largest finite magnitude
# into [2^14, 2^15). NaN and inf pack as themselves, so they stay in their value channel as in
# bf16. The scale is per warp, so its 32 value channels share it: a channel far below the warp's
# largest finite value rounds in fp16 relative to that value (it can flush to zero), and since the
# maximum skips non-finite values, a NaN or inf can change how the other channels round.
# The check is lane-local plus two warp reductions issued after the unscaled stores, so a repack
# costs one more store; it fires on none of the captured training chunks for Y and U and on
# decayed rows of the summaries' transition chain. U' = T^-1 Y' comes back scaled by the same
# factor (unscaled, exactly, for the bf16 state-update copy), and U' is packed the same way for
# A U. A U'' accumulates apart from Q_decay S, and the O drain adds it back times the inverse
# factors (``pow2_pair``), so Y and U round only in their fp16 packs and overflow only where fp32
# would. A, T^-1 and the prep path's ``a`` record (A T^-1 diag(beta)) are plain fp16, unscaled:
# |A| <= |q| |k| before ``scale`` and, for L2-normalized k and beta in [0, 1], the entries of
# T^-1 stay at or below 2^14, so they are finite for L2-normalized Q/K but overflow once |q . k|
# reaches the fp16 maximum 65504; "tf32+bf16" and "bf16" have no such bound.

# ``intra_chunk_dtypes`` value -> (decay_dtype, solve_dtype) under bf16 io.
INTRA_CHUNK_DTYPES = {
    "tf32+fp16": ("tf32", "fp16"),
    "tf32+bf16": ("tf32", "bf16"),
    "bf16+fp16": ("bf16", "fp16"),
    "bf16": ("bf16", "bf16"),
}
# The one value of fp16 io.
FP16_INTRA_CHUNK_DTYPES = "tf32+fp16"
# Biased fp32 exponent a repack gives the warp's largest magnitude (2^14, the top fp16 binade).
INTRA_TOP_EXPONENT = 127 + 14


def operand_dtypes(io: str, intra_chunk_dtypes: str) -> tuple[str, str]:
    """Kernel dtype names ``(decay_dtype, solve_dtype)`` of the ``intra_chunk_dtypes`` option
    for ``io`` operands; fp16 io has the one pair ``("tf32", "fp16")`` for every value.

    Raises:
        ValueError: an unknown value.
    """
    if intra_chunk_dtypes not in INTRA_CHUNK_DTYPES:
        raise ValueError(
            f"intra_chunk_dtypes must be one of {tuple(INTRA_CHUNK_DTYPES)}, "
            f"got {intra_chunk_dtypes!r}"
        )
    if io == "fp16":
        return INTRA_CHUNK_DTYPES[FP16_INTRA_CHUNK_DTYPES]
    return INTRA_CHUNK_DTYPES[intra_chunk_dtypes]


def check_operand_dtypes(io: str, decay_dtype: str, solve_dtype: str) -> None:
    """Validate a kernel's resolved operand dtype names (``operand_dtypes``)."""
    if decay_dtype not in ("tf32", io) or (io == "fp16" and decay_dtype != "tf32"):
        raise ValueError(f"decay_dtype must be tf32 (or bf16 for bf16 io), got {decay_dtype}")
    if solve_dtype not in ("fp16", io):
        raise ValueError(f"solve_dtype must be fp16 or the io dtype, got {solve_dtype}")


def solve_suffix(io: str, solve_dtype: str) -> str:
    """Kernel-name tag of fp16 solve operands under bf16 io; empty for fp16 io."""
    return "_icfp16" if io != "fp16" and solve_dtype == "fp16" else ""


def precision_suffix(io: str, decay_dtype: str, solve_dtype: str) -> str:
    """Kernel-name tag of non-io operand dtypes under bf16 io; empty for fp16 io."""
    decay = "_kktf32" if io != "fp16" and decay_dtype == "tf32" else ""
    return decay + solve_suffix(io, solve_dtype)


@cute.jit
def pow2(exponent):
    """``2^exponent`` for an Int32 ``exponent`` in [-126, 127]."""
    return as_f32((exponent + 127) << 23)


@cute.jit
def pow2_pair(exponent):
    """Two factors whose product is ``2^-exponent`` for ``exponent`` in [-252, 252], each a
    normal fp32 power of two, so multiplying by both in turn is exact."""
    half = exponent >> 1
    return pow2(-half), pow2(half - exponent)


# Magnitude keys order fp32 values by magnitude and leave out NaN and inf at no extra cost: the
# key of bits ``b`` is ``2 b + 2^31 + 2^24`` mod 2^32 as an int32. Doubling drops the sign and
# puts a finite value's biased exponent e in the top byte as e + 1, which the 2^31 shifts into
# signed order (the top byte reads e - 127), while e = 255 (NaN, inf) wraps below the key of zero.
def magnitude_key_of(bits: int) -> int:
    """Host-side ``magnitude_key`` of the non-negative fp32 bit pattern ``bits``."""
    key = (2 * bits + (1 << 31) + (1 << 24)) % (1 << 32)
    return key - (1 << 32) if key >= 1 << 31 else key


# Keys of 0, 2^15 (above every unscaled solve operand, below the FP16 maximum 65504) and 2^-14
# (the FP16 normal minimum).
ZERO_KEY = magnitude_key_of(0)
FP16_SAFE_KEY = magnitude_key_of(0x47000000)
FP16_NORMAL_KEY = magnitude_key_of(0x38800000)


@cute.jit
def magnitude_key(value):
    """The magnitude key of fp32 ``value`` (``magnitude_key_of``)."""
    return inline_ptx(
        f"mad.lo.s32 $0, $1, 2, {ZERO_KEY};",
        write_only_types=[Int32],
        read_only_args=[as_i32(value)],
    )


@cute.jit
def warp_max(value):
    """``redux.sync.max.s32`` of ``value`` over the warp."""
    return inline_ptx(
        "redux.sync.max.s32 $0, $1, 0xffffffff;", write_only_types=[Int32], read_only_args=[value]
    )


def row_halves(two: bool) -> tuple[int, ...]:
    """TMEM address offsets of a fragment's 16-lane halves: ``(0, 16 << 16)`` when ``two``."""
    return (0, 16 << 16) if two else (0,)


@cute.jit
def pack_fragment(vals, dtype: cutlass.Constexpr, factor=None):
    """Pack the 4 pairs of one fp32 ``16x256b`` x2 fragment (``tmem_ld``'s layout: pairs 0 and
    2 in row ``lane // 4``, 1 and 3 in row ``lane // 4 + 8``), times ``factor`` when given, into
    4 ``dtype`` words."""
    words = []
    for r in cutlass.range_constexpr(4):
        p0, p1 = vals[2 * r], vals[2 * r + 1]
        if cutlass.const_expr(factor is not None):
            p0, p1 = fmul2(p0, p1, factor, factor)
        words.append(fp32_to_fp16(p0, p1, dtype=dtype))
    return words


@cute.jit
def store_solve_rows(rows, dtype: cutlass.Constexpr, scaled: cutlass.Constexpr):
    """Store the fp32 fragments of ``rows`` (``(TMEM address, 16x256b x2 fragment)`` pairs,
    ``16x128b`` stores) as ``dtype`` and return the power-of-two exponent ``d`` the warp scaled
    them by (NOTE [KDA operand precision]). Without ``scaled`` they are stored as they are and
    ``d`` is 0; with it, ``d`` is 0 (unscaled) when every finite value of the warp is below
    2^15 and the finite values of every 4-value row segment a lane holds are zero or reach the
    fp16 normal range, else the one exponent in [-126, 126] that brings the warp's largest
    finite magnitude into [2^14, 2^15), and the fragments are stored again scaled. NaN and inf
    store as themselves (``magnitude_key`` leaves them out). The unscaled stores issue before the
    two warp reductions, so their latency overlaps the stores; the outcome is uniform per warp,
    whose rows no other warp packs or drains. The caller completes the stores
    (``tcgen05.wait::st``)."""
    lane_max = Int32(ZERO_KEY)
    flagged = Int32(0)
    for addr, vals in rows:
        tmem_st("16x128b", addr, pack_fragment(vals, dtype))
        if cutlass.const_expr(scaled):
            for sub in cutlass.range_constexpr(2):
                seg = Int32(ZERO_KEY)
                for r in cutlass.range_constexpr(sub, 4, 2):
                    for j in cutlass.range_constexpr(2):
                        seg = cutlass.max(seg, magnitude_key(vals[2 * r + j]))
                lane_max = cutlass.max(lane_max, seg)
                subnormal = (seg > ZERO_KEY) & (seg < FP16_NORMAL_KEY)
                flagged = cutlass.max(
                    flagged, Int32(cutlass.select_(subnormal, Int32(1), Int32(0)))
                )
    exponent = Int32(0)
    if cutlass.const_expr(scaled):
        largest = warp_max(lane_max)
        if largest >= FP16_SAFE_KEY or warp_max(flagged) > 0:  # warp-uniform, rare
            # The key's top byte reads the biased exponent minus 127.
            exponent = cutlass.max(
                cutlass.min(Int32(INTRA_TOP_EXPONENT - 127) - (largest >> 24), Int32(126)),
                Int32(-126),
            )
            factor = pow2(exponent)
            cute.arch.fence_view_async_tmem_store()  # the unscaled stores land first
            for addr, vals in rows:
                tmem_st("16x128b", addr, pack_fragment(vals, dtype, factor))
    return exponent


@cute.jit
def store_unscaled_rows(rows, exponent, dtype: cutlass.Constexpr):
    """Store the fp32 fragments of ``rows`` (as for ``store_solve_rows``) times 2^-``exponent``
    (exact) as ``dtype``."""
    factor = pow2(-exponent)
    for addr, vals in rows:
        tmem_st("16x128b", addr, pack_fragment(vals, dtype, factor))


# ---- TF32 decay operands -----------------------------------------------------------------
# NOTE [TF32 decay operands]
# With g the per-channel prefix log2 decay of a 16-token chunk, the in-chunk products are
# A[i, j] = (Q[i] * 2^g[i]) . (K[j] * 2^-g[j]) for j <= i, and KK[i, j], which uses K[i] in place
# of Q[i]. On the diagonal the decay cancels exactly in the math, but staging the operands in
# fp16 can turn one into zero and the other into infinity before they multiply: their ratio spans
# the whole 16-token gate sum (about 2^121 at the gate bound). That is why the fp16 kernels (and
# the bf16 ones with "tf32" decay products, NOTE [KDA operand precision]) form these operands
# (Q_decay or K_decay, and K_inv) in fp32 and round them to tf32 for m16n8k8 tf32 mma.sync (and,
# in the backward, tcgen05 ``.kind::tf32``): they need the exponent range, not accumulator
# precision.
#
# The operands are fp32 ``[16, d_k]`` tiles in one of two layouts (``tf32_tile``), which both the
# row stores (``sts_tf32x8``) and the fragment loads (``tf32_fragment_offsets``) index by
# (row, column). Rows are 16-byte chunks of four fp32 columns.
# - mma.sync only (``umma=False``): rows of ``4 d_k`` bytes; logical chunk
#   ``c = (c4, c3 c2 c1, c0)`` sits at physical chunk ``(c4, c0, c3 c2 c1 ^ row & 7)``, so the
#   eight rows each ``ldmatrix`` matrix reads, and the eight chunks ``2 l + c0`` that eight lanes
#   of a row store, land in distinct 16-byte bank groups.
# - also a tcgen05 MN-major operand (``umma=True``): CuTe's ``MN_SW128_32B`` atom
#   (SWIZZLE_128B_BASE32B) tiled rows first, so 2048-byte boxes of 32 columns hold the 16 rows of
#   128 bytes, the 32-byte unit XORed with ``row & 3`` (2-way conflicts on those reads and stores).


def tf32_tile(d_k: int, umma: bool = False):
    """Layout of a tf32 decay tile (NOTE [TF32 decay operands]). mma.sync: in fp32 elements,
    indexed by ``(row, column)``, the column permutation, then ``Swizzle<3, 2, log2(d_k) - 2>``
    XORs the chunk bits ``c3 c2 c1`` with ``row & 7``. ``umma``: in bytes, indexed by
    ``(column, row)``, the ``MN_SW128_32B`` atom tiled to ``[max(d_k, 32), 16]`` rows first (a
    16-column tile uses half of one box); the atom's swizzle acts on byte offsets."""
    if umma:
        atom = tcgen05.make_smem_layout_atom(tcgen05.SmemLayoutAtomKind.MN_SW128_32B, Float32)
        tiled = cute.tile_to_shape(atom, (max(d_k, 32), B_T), (1, 0))
        to_bytes = cute.make_layout(cute.cosize(tiled.outer), stride=4)
        return cute.make_composed_layout(atom.inner, 0, cute.composition(to_bytes, tiled.outer))
    perm = cute.make_layout((B_T, (4, 2, 8, d_k // 64)), stride=(d_k, (1, 32, 4, 64)))
    return cute.make_composed_layout(cute.make_swizzle(3, 2, d_k.bit_length() - 3), 0, perm)


@cute.jit
def tf32_offset(row, dim, d_k: cutlass.Constexpr[int], umma: cutlass.Constexpr[bool] = False):
    """Byte offset of fp32 ``(row, dim)`` in a tf32 decay tile."""
    if cutlass.const_expr(umma):
        return cute.crd2idx((dim, row), tf32_tile(d_k, True))
    return 4 * cute.crd2idx((row, dim), tf32_tile(d_k))


@cute.jit
def sts_tf32x8(
    base, row, dim, values, d_k: cutlass.Constexpr[int], umma: cutlass.Constexpr = False
):
    """Round eight fp32 ``values`` of row ``row``, columns ``[dim, dim + 8)``, to tf32
    (``tf32_round``) and store them into the tile at ``base``."""
    words = [tf32_round(v) for v in values]
    for half in cutlass.range_constexpr(2):
        st_shared_v4(
            base + tf32_offset(row, dim + 4 * half, d_k, umma), words[4 * half : 4 * half + 4]
        )


def _tf32_period(umma: bool) -> tuple[int, int]:
    """K steps after which a tile's row offsets repeat, and the bytes the next period adds."""
    return (4, 2048) if umma else (8, 256)


def tf32_fragment_offsets(lane, d_k: int, *, b_operand: bool, umma: bool = False):
    """This lane's ``ldmatrix.x4`` row offsets of the first period of k-steps (see
    ``mma_tf32_16x16``). With ``lane = r + 8 u + 16 v``, k-step ``s`` reads
    ``(r + 8 u, 8 s + 4 v)`` for the A fragment of rows ``[0, 16)`` and, swapping ``u`` and
    ``v``, ``(r + 8 v, 8 s + 4 u)`` for the B fragment of tokens ``[0, 16)``. Within a period
    the swizzle only permutes a row's units, so step ``s`` is step 0 XOR the row-0 offset of
    column ``8 s``. Evaluating ``tf32_tile`` at every step instead lengthened the roles' setup
    code and measured about 9% slower FP16 prefill."""
    r, u, v = lane % 8, (lane // 8) % 2, lane // 16
    if b_operand:
        u, v = v, u
    period, _ = _tf32_period(umma)
    base = tf32_offset(r + 8 * u, 4 * v, d_k, umma)
    return [opaque_i32(base ^ tf32_offset(0, 8 * ks, d_k, umma)) for ks in range(period)]


@cute.jit
def mma_tf32_16x16(
    acc, s_a, s_b, a_offs, b_offs, d_k: cutlass.Constexpr[int], umma: cutlass.Constexpr = False
):
    """``acc[16, 16] += A B^T`` over ``d_k`` columns of two tf32 decay tiles (A rows at ``s_a``,
    B rows at ``s_b``), in the m16n8k16 accumulator layout."""
    period, box = _tf32_period(umma)
    for ks in cutlass.range_constexpr(d_k // 8):
        step = (ks // period) * box
        af = ldmatrix(s_a + a_offs[ks % period] + step, 4)
        bf = ldmatrix(s_b + b_offs[ks % period] + step, 4)
        mma_step_tf32(acc, af, bf, M=16, N=16)


@cute.jit
def mma_decay_16x16(
    acc,
    s_a,
    s_kinv,
    stage,
    a_offs,
    b_offs,
    d_k: cutlass.Constexpr[int],
    dtype: cutlass.Constexpr,
    umma: cutlass.Constexpr = False,
):
    """``acc[16, 16] += A K_inv^T`` over ``d_k`` columns of one decay stage, ``A`` the decayed
    rows (K_decay for KK, Q_decay for A) at ``s_a + stage`` and K_inv at ``s_kinv + stage``:
    tf32 tiles (``umma`` selecting their layout) when ``dtype`` is Float32, else ``dtype`` tiles
    of 64-column 128-byte-swizzled boxes. ``a_offs`` and ``b_offs`` come from
    ``decay_fragment_offsets`` (the tcgen05 layout's from ``tf32_fragment_offsets``)."""
    if cutlass.const_expr(dtype is Float32):
        mma_tf32_16x16(acc, s_a + stage, s_kinv + stage, a_offs, b_offs, d_k, umma)
    else:
        for i in cutlass.range_constexpr(d_k // 16):
            box = (i // 4) * (B_T * 64)
            kif = ldmatrix(s_kinv + stage + 2 * (b_offs[i % 4] + box), 4)
            af = ldmatrix(s_a + stage + 2 * (a_offs[i % 4] + box), 4)
            mma_step(acc, af, kif, k_step=0, M=16, N=16, dtype=dtype)


@cute.jit
def decay_fragment_offsets(lane, d_k: cutlass.Constexpr[int], tf32: cutlass.Constexpr[bool]):
    """This lane's ``mma_decay_16x16`` fragment offsets ``(a_offs, kinv_offs)``, hoisted out of
    the chunk loop by the caller: ``tf32_fragment_offsets`` under ``tf32``, else the element
    offsets of the first 16-bit box's four 16-column steps."""
    if cutlass.const_expr(tf32):
        a_offs = tf32_fragment_offsets(lane, d_k, b_operand=False)
        kinv_offs = tf32_fragment_offsets(lane, d_k, b_operand=True)
    else:
        ki_row = lane % 8 + (8 if lane // 16 else 0)
        ki_col = 8 if (lane // 8) % 2 else 0
        a_row = lane % 8 + (8 if (lane // 8) % 2 else 0)
        a_col = 8 if (lane // 8) // 2 else 0
        kinv_offs = [opaque_i32(ki_row * 64 + swz128(ki_row, i * 16 + ki_col)) for i in range(4)]
        a_offs = [opaque_i32(swz128(a_row, a_row * 64 + i * 16 + a_col)) for i in range(4)]
    return a_offs, kinv_offs


def decay_stage_addresses(h_rows, rows: str, kh, s_rows, s_kinv, d_k: int, tf32: bool):
    """``mma_decay_16x16``'s ``(s_a, s_kinv, stage)`` at a decay ring stage, recording the
    reads: the views of the tf32 tiles ``rows + "32"`` of handle ``h_rows`` and ``Kinv`` of
    ``kh`` (stage 0) under ``tf32``, else the 16-bit tiles at ``s_rows`` and ``s_kinv`` and
    ``h_rows``'s stage offset (the rings advance together)."""
    if tf32:
        return (
            h_rows.view(rows + "32", read=True).iterator.toint(),
            kh.view("Kinv", read=True).iterator.toint(),
            0,
        )
    h_rows.access(rows, read=True, by="threads")
    kh.access("Kinv", read=True, by="threads")
    return s_rows, s_kinv, h_rows.index * (d_k * B_T * 2)
