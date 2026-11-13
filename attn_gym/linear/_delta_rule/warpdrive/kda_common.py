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

from attn_gym.warpdrive.thread_sync import tcgen05_after_thread_sync, tcgen05_before_thread_sync

from .ptx import elect_sync, f16x2_to_f32, fmul2, fp32_to_fp16, ldmatrix, mma_step_tf32
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
# more half word through a funnel shift (measured 1.0-1.4x the aligned time). Loads read only
# blocks that hold a byte of the row, so they never leave its allocation. Stores write the
# interior blocks whole and the two partial end blocks element by element, so no byte outside
# the row is written.


@cute.jit
def shift_words(words, shift, count: cutlass.Constexpr[int]):
    """``words[j + shift]`` for ``j < count``, with ``shift`` a runtime value in ``[0, 3]``."""
    odd, high = (shift & 1) != 0, (shift & 2) != 0
    shifted = []
    for j in cutlass.range_constexpr(count):
        low_pair = cutlass.select_(odd, words[j + 1], words[j])
        high_pair = cutlass.select_(odd, words[j + 3], words[j + 2])
        shifted.append(cutlass.select_(high, high_pair, low_pair))
    return shifted


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
def load_state_16(addr, dtype: cutlass.Constexpr, align: cutlass.Constexpr[int] = 16):
    """16 consecutive state values at global ``addr`` (fp32 or bf16) as fp32. Below 16-byte
    ``align``, the row is realigned in registers (NOTE [Unaligned state rows])."""
    vals = []
    if cutlass.const_expr(align < 16):
        blocks = dtype.width // 8  # 16-byte blocks of 16 values
        misalignment = addr & 15
        base = addr - misalignment
        words = []
        for g in cutlass.range_constexpr(blocks):
            words += ldg_v4(base + g * 16)
        # The values spill into one more block unless the row is aligned; reread the last one
        # then, so the load stays on blocks of the row.
        words += ldg_v4(
            base + blocks * 16 - cutlass.select_(misalignment == 0, Int64(16), Int64(0))
        )
        for w in realign_words(words, misalignment, 4 * blocks):
            if cutlass.const_expr(dtype is cutlass.Float32):
                vals.append(as_f32(w))
            else:
                lo, hi = f16x2_to_f32(w, dtype=dtype)
                vals += [lo, hi]
    elif cutlass.const_expr(dtype is cutlass.Float32):
        for g in cutlass.range_constexpr(4):
            for w in ldg_v4(addr + g * 16):
                vals.append(as_f32(w))
    else:
        for g in cutlass.range_constexpr(2):
            for w in ldg_v4(addr + g * 16):
                lo, hi = f16x2_to_f32(w, dtype=dtype)
                vals += [lo, hi]
    return vals


@cute.jit
def store_unaligned_state_32(addr, vals, dtype: cutlass.Constexpr):
    """Store 32 fp32 ``vals`` as ``dtype`` at an element-aligned global ``addr`` (NOTE
    [Unaligned state rows])."""
    element_bytes = dtype.width // 8
    if cutlass.const_expr(dtype is Float32):
        words = [as_i32(v) for v in vals]
    else:
        words = [fp32_to_fp16(vals[2 * j], vals[2 * j + 1], dtype=dtype) for j in range(16)]
    blocks = len(words) // 4
    misalignment = addr & 15
    base = addr - misalignment
    if misalignment == 0:
        for g in cutlass.range_constexpr(blocks):
            stg_v4(base + g * 16, words[4 * g : 4 * g + 4])
    else:
        # Aligned word i holds stream bytes [4i - misalignment, 4i - misalignment + 4): realign
        # the stream behind one block of padding by 16 - misalignment bytes.
        pad = [Int32(0)] * 4
        aligned = realign_words(pad + words + pad, 16 - misalignment, len(words))
        for g in cutlass.range_constexpr(1, blocks):
            stg_v4(base + g * 16, aligned[4 * g : 4 * g + 4])
        # The partial end blocks: the first (16 - misalignment) and last misalignment bytes.
        row = cute.make_tensor(
            cute.make_ptr(dtype, addr, cute.AddressSpace.gmem, assumed_align=element_bytes),
            cute.make_layout(32),
        )
        for e in cutlass.range_constexpr(16 // element_bytes - 1):
            if e * element_bytes + misalignment < 16:
                row[e] = vals[e].to(dtype)
            if e * element_bytes < misalignment:
                row[31 - e] = vals[31 - e].to(dtype)


@cute.jit
def store_state_32(
    addr, offset: int, vals, dtype: cutlass.Constexpr, align: cutlass.Constexpr[int] = 16
):
    """Store 32 fp32 ``vals`` at global ``addr + offset`` (bytes) as ``dtype`` (fp32, or bf16
    rounded to nearest even). Each aligned store adds its constant offset to ``addr`` right before
    it, after the conversion, keeping the instruction order of the former inline form. Below
    16-byte ``align``, the row is realigned in registers (NOTE [Unaligned state rows])."""
    if cutlass.const_expr(align < 16):
        store_unaligned_state_32(addr + offset, vals, dtype)
    elif cutlass.const_expr(dtype is Float32):
        for g in cutlass.range_constexpr(8):
            stg_v4(addr + (offset + g * 16), [as_i32(vals[g * 4 + t]) for t in range(4)])
    else:
        for g in cutlass.range_constexpr(4):
            words = [
                fp32_to_fp16(vals[g * 8 + 2 * t], vals[g * 8 + 2 * t + 1], dtype=dtype)
                for t in range(4)
            ]
            stg_v4(addr + (offset + g * 16), words)


@cute.jit
def beta_residual(v_word, beta0, beta1, ks0, ks1, dtype: cutlass.Constexpr, with_state):
    """``beta * (v - ks)`` for one packed V pair in fp32; only the final pack rounds."""
    v0, v1 = f16x2_to_f32(v_word, dtype=dtype)
    if cutlass.const_expr(with_state):
        v0, v1 = cute.arch.add_packed_f32x2((v0, v1), (-ks0, -ks1))
    y0, y1 = fmul2(beta0, beta1, v0, v1)
    return fp32_to_fp16(y0, y1, dtype=dtype)


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


# ---- TF32 decay operands (fp16 io) -------------------------------------------------------
# fp16 cannot hold the in-chunk decay factors 2^g and 2^-g, whose ratio spans the whole
# 16-token gate sum (about 2^121 at the gate bound), so the fp16 kernels stage the decay operands
# of the in-chunk products as fp32 ``[16, d_k]`` tiles rounded to tf32 and multiply them with
# m16n8k8 tf32 mma.sync (and, in the backward, tcgen05 ``.kind::tf32``). Rows are 16-byte
# chunks of four fp32 columns in one of two layouts:
# - mma.sync only (``umma=False``): rows of ``4 d_k`` bytes; logical chunk
#   ``c = (c4, c3 c2 c1, c0)`` sits at physical chunk ``(c4, c0, c3 c2 c1 ^ row & 7)``, so the
#   eight rows each ``ldmatrix`` matrix reads and the eight chunks ``2 l + c0`` that eight lanes
#   of a row store land in distinct 16-byte bank groups.
# - also a tcgen05 MN-major operand (``umma=True``): the canonical SWIZZLE_128B_BASE32B layout,
#   2048-byte boxes of 32 columns, each 4 atoms of 4 rows x 128 bytes, the 32-byte unit XORed
#   with ``row & 3`` (2-way conflicts on those reads and stores).


@cute.jit
def tf32_chunk(row, chunk):
    """Physical 16-byte chunk of logical ``chunk`` in ``row`` of an mma.sync tf32 tile."""
    return (chunk & 16) | ((chunk & 1) << 3) | (((chunk >> 1) & 7) ^ (row & 7))


@cute.jit
def tf32_offset(row, dim, d_k: cutlass.Constexpr[int], umma: cutlass.Constexpr[bool] = False):
    """Byte offset of fp32 ``(row, dim)`` (``dim`` a multiple of 4) in a tf32 decay tile."""
    if cutlass.const_expr(umma):
        unit = ((dim >> 3) & 3) ^ (row & 3)
        return (
            (dim >> 5) * 2048 + (row >> 2) * 512 + (row & 3) * 128 + (unit << 5)
            + (((dim >> 2) & 1) << 4)
        )  # fmt: skip
    return row * (d_k * 4) + (tf32_chunk(row, dim >> 2) << 4)


@cute.jit
def sts_tf32x8(
    base, row, dim, values, d_k: cutlass.Constexpr[int], umma: cutlass.Constexpr = False
):
    """Round eight fp32 ``values`` of row ``row``, columns ``[dim, dim + 8)``, to tf32 and store
    them into the tile at ``base``."""
    words = [
        inline_ptx("cvt.rna.tf32.f32 $0, $1;", write_only_types=[Int32], read_only_args=[v])
        for v in values
    ]
    for half in cutlass.range_constexpr(2):
        inline_ptx(
            "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
            read_only_args=[
                Int32(base + tf32_offset(row, dim + 4 * half, d_k, umma)),
                *words[4 * half : 4 * half + 4],
            ],
        )


def _tf32_period(umma: bool) -> tuple[int, int]:
    """K steps after which a tile's row offsets repeat, and the bytes the next period adds."""
    return (4, 2048) if umma else (8, 256)


def tf32_fragment_offsets(lane, d_k: int, *, b_operand: bool, umma: bool = False):
    """This lane's ``ldmatrix.x4`` row offsets of the first period of k-steps (see
    ``mma_tf32_16x16``). The four matrices are the A fragment of rows ``[0, 16)`` (rows 0-7 /
    8-15 at columns +0, then at +4) or the B fragment of tokens ``[0, 16)`` (columns +0 / +4 of
    tokens 0-7, then of 8-15)."""
    m, r = lane // 8, lane % 8
    if b_operand:
        row, col = r + 8 * (m // 2), m % 2
    else:
        row, col = r + 8 * (m % 2), m // 2
    period, _ = _tf32_period(umma)
    return [opaque_i32(tf32_offset(row, 8 * ks + 4 * col, d_k, umma)) for ks in range(period)]


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
