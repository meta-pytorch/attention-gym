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


@cute.jit
def load_state_16(addr, dtype: cutlass.Constexpr):
    """16 consecutive state values at global ``addr`` (fp32 or bf16) as fp32."""
    vals = []
    if cutlass.const_expr(dtype is cutlass.Float32):
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


# ---- TF32 decay operands (fp16 io) -------------------------------------------------------
# NOTE [TF32 decay operands]
# With g the per-channel prefix log2 decay of a 16-token chunk, the in-chunk products are
# A[i, j] = (Q[i] * 2^g[i]) . (K[j] * 2^-g[j]) for j <= i, and KK[i, j], which uses K[i] in place
# of Q[i]. On the diagonal the decay cancels exactly in the math, but staging the operands in
# fp16 can turn one into zero and the other into infinity before they multiply: their ratio spans
# the whole 16-token gate sum (about 2^121 at the gate bound). That is why the fp16 kernels form
# these operands (Q_decay or K_decay, and K_inv) in fp32 and round them to tf32 for m16n8k8 tf32
# mma.sync (and, in the backward, tcgen05 ``.kind::tf32``): they need the exponent range, not
# accumulator precision.
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
    """Round eight fp32 ``values`` of row ``row``, columns ``[dim, dim + 8)``, to tf32 and store
    them into the tile at ``base``."""
    words = [cute.arch.cvt_f32_tf32(v) for v in values]
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
