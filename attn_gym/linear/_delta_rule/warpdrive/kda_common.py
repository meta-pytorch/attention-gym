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

from .ptx import elect_sync, f16x2_to_f32, fmul2, fp32_to_fp16
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
def spin_wait(bar, phase):
    """Hint-less ``mbarrier.try_wait.parity`` loop (the upstream kernels' ``spin=True`` wait):
    ptxas keeps it a ``SYNCS.PHASECHK`` retry branch, without the ``NANOSLEEP`` of the
    suspend-hint form that ``cute.arch.mbarrier_wait`` (and so a channel wait) emits. Then
    ``tcgen05.fence::after_thread_sync``, which orders the caller's later tcgen05 operations
    after the wait (PTX ISA 9.7.18.6.3), as a managed wait in a TMEM role does."""
    inline_ptx(
        "{\n\t.reg .pred P1;\n\tLAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [$0], $1;\n\t"
        "@P1 bra.uni DONE;\n\tbra.uni LAB_WAIT;\n\tDONE:\n\t}",
        read_only_args=[Int32(bar.toint()), Int32(phase)],
    )
    tcgen05_after_thread_sync()


@cute.jit
def warp_arrive(bar):
    """One arrival per warp after all its lanes (the ``Warps`` signal): every lane fences its
    tcgen05 operations (``tcgen05.fence::before_thread_sync``), then the warp sync orders every
    lane's prior accesses and completion waits before the elected lane's arrive."""
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
def raw_beta_wait(raw, beta, n, stages: cutlass.Constexpr[int]):
    """Waits on chunk ``n``'s stages of two ``stages``-deep rings that a compute group consumes
    once per chunk, in order, driven by one shared counter through the end accessors, then
    ``tcgen05.fence::after_thread_sync`` as after a managed wait in a TMEM role. Returns the
    stage."""
    stage = n & (stages - 1)
    phase = (n // stages) & 1
    cute.arch.mbarrier_wait(raw.full_barrier(stage), phase)
    cute.arch.mbarrier_wait(beta.full_barrier(stage), phase)
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
