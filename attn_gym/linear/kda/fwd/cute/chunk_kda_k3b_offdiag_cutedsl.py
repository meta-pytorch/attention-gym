# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CuTe DSL K3b: Off-diagonal gated matmul, one pair per CTA, 4 warps.

Matches NV C++ kernel_3b_offdiag_matmul.cu architecture:
  Exact grid: (total_chunks * 6, H, 1) — one CTA per (chunk, pair, head)
  Persistent grid: (chunk_workers * 6, H, 1) — each CTA strides over chunks
  128 threads (4 warps):
      All warps: Phase 1 — fused gating into SMEM
      Warp 0:    Phase 2 — Aqk = sQg @ sKn^T  (MMA)
      Warp 1:    Phase 2 — Akk = sKp @ sKn^T  (MMA)
      All warps: Phase 3 — scaled stores to GMEM

Single-kernel varlen (no is_partial split, no bulk+cleanup dispatch):
  Non-varlen: constexpr-unrolled gating loops, zero guard overhead.
  Varlen: predicated constexpr-offset loads (R2P-style).
          Uses ti_row + _r (constexpr offset) inside predicated blocks:
          loads only execute when _r < actual_row. Invalid loads are
          suppressed at the PTX level (@p LDG) — the address is never
          accessed, so no padding required. Variables default to safe
          values (0 for Q/K, g_ref for G) ensuring exp2(0)=1 and 0*1=0.
"""

import cutlass
from cutlass import Int32, cute
from cutlass.cute.nvgpu import warp

from attn_gym.linear.kda.chunk_schedule import ScheduleKind
from attn_gym.linear.kda.fwd.cute.chunk_schedule import ChunkSchedule
from attn_gym.linear.kda.fwd.cute.chunk_scheduler_cute import (
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)


class ChunkKDAFwdK3bOffdiagCuteDSL:
    WARP_SIZE = 32

    def __init__(
        self,
        BC: int = 16,
        D: int = 128,
        chunk_size: int = 64,
        num_subchunks: int = 4,
        chunk_schedule: ChunkSchedule = ChunkSchedule.DENSE,
        schedule_kind: ScheduleKind = ScheduleKind.STATIC,
        chunk_workers: int = 0,
        use_int64_offsets: bool = False,
    ):
        assert num_subchunks == 4, (
            f"ChunkKDAFwdK3bOffdiagCuteDSL only supports four subchunks, got {num_subchunks}"
        )
        assert chunk_size == num_subchunks * BC, (
            f"chunk_size must equal num_subchunks * BC, got {chunk_size} and "
            f"{num_subchunks} * {BC}"
        )
        if schedule_kind is ScheduleKind.PERSISTENT:
            assert chunk_schedule is ChunkSchedule.RAGGED and chunk_workers > 0, (
                "persistent execution requires a positive ragged chunk-worker grid"
            )
        else:
            assert chunk_workers == 0, "static execution does not use chunk workers"
        self.schedule_kind = schedule_kind
        self.chunk_workers = chunk_workers
        self.use_int64_offsets = use_int64_offsets
        self.BC = BC
        self.D = D
        self.chunk_schedule = chunk_schedule
        self.BT = chunk_size
        self.num_offdiag_blocks = num_subchunks * (num_subchunks - 1) // 2
        self.num_threads = 128
        self.mma_inst_shape = (16, 8, 16)
        self.atom_layout_mnk = (1, 1, 1)

    @cute.jit
    def upcast(self, value):
        """Promote an address operand before its first overflowing multiply."""
        return cutlass.Int64(value) if cutlass.const_expr(self.use_int64_offsets) else value

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mG: cute.Tensor,
        mBeta: cute.Tensor,
        mAqk: cute.Tensor,
        mAkkOD: cute.Tensor,
        scale: cutlass.Float32,
        H: int,
        total_chunks: int,
        cu_seqlens: cute.Tensor | None,
        chunk_offsets: cute.Tensor | None,
        stream,
    ):
        self._dtype: type[cutlass.Numeric] = mQ.element_type
        chunk_axis = (
            self.chunk_workers
            if cutlass.const_expr(self.schedule_kind is ScheduleKind.PERSISTENT)
            else total_chunks
        )
        grid = (chunk_axis * self.num_offdiag_blocks, H, 1)

        smem_k_block_size = 64
        swizzle_bits = 3
        sQK_layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            cute.make_layout((8, smem_k_block_size), stride=(smem_k_block_size, 1)),
        )
        sGated_layout = cute.tile_to_shape(sQK_layout_atom, (self.BC, self.D), (0, 1))
        sGref_layout = cute.make_layout((self.D,), stride=(1,))
        sBeta_layout = cute.make_layout((self.BC,), stride=(1,))

        @cute.struct
        class SharedStorage:
            sQg: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sGated_layout)], 128
            ]
            sKp: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sGated_layout)], 128
            ]
            sKn: cute.struct.Align[
                cute.struct.MemRange[self._dtype, cute.cosize(sGated_layout)], 128
            ]
            sGref: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.D], 128]
            sBeta: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, self.BC], 128]

        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self._dtype, cutlass.Float32, self.mma_inst_shape),
            self.atom_layout_mnk,
            permutation_mnk=(
                self.atom_layout_mnk[0] * self.mma_inst_shape[0],
                self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
                self.mma_inst_shape[2],
            ),
        )

        smem_copy_A = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
            ),
            tiled_mma,
        )
        smem_copy_B = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), self._dtype
            ),
            tiled_mma,
        )

        self.kernel(
            mQ,
            mK,
            mG,
            mBeta,
            mAqk,
            mAkkOD,
            scale,
            cu_seqlens,
            chunk_offsets,
            sGated_layout,
            sGref_layout,
            sBeta_layout,
            smem_copy_A,
            smem_copy_B,
            tiled_mma,
            SharedStorage,
        ).launch(grid=grid, block=[self.num_threads, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mG: cute.Tensor,
        mBeta: cute.Tensor,
        mAqk: cute.Tensor,
        mAkkOD: cute.Tensor,
        scale: cutlass.Float32,
        cu_seqlens: cute.Tensor | None,
        chunk_offsets: cute.Tensor | None,
        sGated_layout: cute.ComposedLayout,
        sGref_layout: cute.Layout,
        sBeta_layout: cute.Layout,
        smem_copy_A: cute.TiledCopy,
        smem_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        block_idx, head_idx, _ = cute.arch.block_idx()

        chunk_idx = block_idx // self.num_offdiag_blocks
        pair_idx = block_idx % self.num_offdiag_blocks

        ri = 1
        if pair_idx >= 1:
            ri = 2
        if pair_idx >= 3:
            ri = 3
        ci = pair_idx - (ri * (ri - 1)) // 2

        # ── Shared memory, allocated once and reused across persistent iterations ──
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQg = storage.sQg.get_tensor(sGated_layout)
        sKp = storage.sKp.get_tensor(sGated_layout)
        sKn = storage.sKn.get_tensor(sGated_layout)
        sGref = storage.sGref.get_tensor(sGref_layout)
        sBeta = storage.sBeta.get_tensor(sBeta_layout)

        if cutlass.const_expr(self.schedule_kind is ScheduleKind.PERSISTENT):
            active_chunks = load_ragged_chunk_count(chunk_offsets)
            # unroll=2 interleaves two chunks' gating loads, restoring the
            # load->use distance the scheduler achieves in the static body
            # (median 46 vs 14 instructions; long_scoreboard 25 -> ~12 cyc/inst).
            # unroll=4 crosses a code-size/register cliff and loses it again.
            for current_chunk in cutlass.range(
                chunk_idx, active_chunks, self.chunk_workers, unroll=2
            ):
                self._process_chunk(
                    mQ,
                    mK,
                    mG,
                    mBeta,
                    mAqk,
                    mAkkOD,
                    scale,
                    cu_seqlens,
                    chunk_offsets,
                    sQg,
                    sKp,
                    sKn,
                    sGref,
                    sBeta,
                    smem_copy_A,
                    smem_copy_B,
                    tiled_mma,
                    current_chunk,
                    pair_idx,
                    ri,
                    ci,
                    head_idx,
                    tidx,
                )
                # Synchronize all warps before the next chunk's gating reuses SMEM.
                cute.arch.sync_threads()
        else:
            self._process_chunk(
                mQ,
                mK,
                mG,
                mBeta,
                mAqk,
                mAkkOD,
                scale,
                cu_seqlens,
                chunk_offsets,
                sQg,
                sKp,
                sKn,
                sGref,
                sBeta,
                smem_copy_A,
                smem_copy_B,
                tiled_mma,
                chunk_idx,
                pair_idx,
                ri,
                ci,
                head_idx,
                tidx,
            )

    @cute.jit
    def _process_chunk(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mG: cute.Tensor,
        mBeta: cute.Tensor,
        mAqk: cute.Tensor,
        mAkkOD: cute.Tensor,
        scale: cutlass.Float32,
        cu_seqlens: cute.Tensor | None,
        chunk_offsets: cute.Tensor | None,
        sQg: cute.Tensor,
        sKp: cute.Tensor,
        sKn: cute.Tensor,
        sGref: cute.Tensor,
        sBeta: cute.Tensor,
        smem_copy_A: cute.TiledCopy,
        smem_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        chunk_idx: Int32,
        pair_idx: Int32,
        ri: Int32,
        ci: Int32,
        head_idx: Int32,
        tidx: Int32,
    ):
        """Gate, multiply, and store one (chunk, pair, head) off-diagonal tile."""
        warp_idx = tidx // self.WARP_SIZE
        lane_idx = tidx % self.WARP_SIZE

        if cutlass.const_expr(self.chunk_schedule is ChunkSchedule.RAGGED):
            active_chunks = load_ragged_chunk_count(chunk_offsets)
            chunk_base = Int32(0)
            eos = Int32(0)
            if chunk_idx < active_chunks:
                _, _, chunk_base, valid_tokens = load_ragged_chunk_work(
                    cu_seqlens,
                    chunk_offsets,
                    Int32(chunk_idx),
                    Int32(self.BT),
                )
                eos = chunk_base + valid_tokens
        else:
            chunk_base = chunk_idx * self.BT
            eos = cute.size(mQ, mode=[0])

        chunk_base = self.upcast(chunk_base)
        ti_row = chunk_base + ri * self.BC
        ti_col = chunk_base + ci * self.BC
        h_offset = head_idx * self.D
        aqk_col = head_idx * self.BT
        akkod_col = head_idx * self.BC * self.BC

        # ══════════════════════════════════════════════════════════
        # Phase 1: Gating into SMEM
        # ══════════════════════════════════════════════════════════
        if cutlass.const_expr(self.chunk_schedule is not ChunkSchedule.DENSE):
            col = tidx
            h_col = h_offset + col
            if chunk_base + self.BT <= eos:
                if tidx < self.D:
                    sGref[tidx] = mG[ti_row, h_col]
                if tidx < self.BC:
                    sBeta[tidx] = cutlass.Float32(mBeta[ti_row + tidx, head_idx])
                for _r in cutlass.range_constexpr(self.BC):
                    g_ref_val = sGref[col]
                    q_val = cutlass.Float32(mQ[ti_row + _r, h_col])
                    k_val_r = cutlass.Float32(mK[ti_row + _r, h_col])
                    g_val_r = mG[ti_row + _r, h_col]
                    gate_r = cute.math.exp2(g_val_r - g_ref_val, fastmath=True)
                    sQg[_r, col] = self._dtype(q_val * gate_r)
                    sKp[_r, col] = self._dtype(k_val_r * gate_r)

                    k_val_c = cutlass.Float32(mK[ti_col + _r, h_col])
                    g_val_c = mG[ti_col + _r, h_col]
                    gate_c = cute.math.exp2(g_ref_val - g_val_c, fastmath=True)
                    sKn[_r, col] = self._dtype(k_val_c * gate_c)
            else:
                # ── Varlen tail: predicated constexpr-offset loads (R2P-style) ──
                actual_row = cutlass.min(self.BC, cutlass.max(eos - ti_row, 0))
                actual_col = cutlass.min(self.BC, cutlass.max(eos - ti_col, 0))
                safe_ti_row = cutlass.min(ti_row, cutlass.max(eos - 1, 0))

                if tidx < self.D and eos > 0:
                    sGref[tidx] = mG[safe_ti_row, h_col]
                if tidx < self.D and eos == 0:
                    # Capacity-only CTAs (eos stays 0) must not touch token storage; a zero
                    # reference gate keeps their SMEM tiles deterministic zeros.
                    sGref[tidx] = mG.element_type(0.0)
                if tidx < self.BC:
                    beta_val = cutlass.Float32(0.0)
                    if tidx < actual_row:
                        beta_val = cutlass.Float32(mBeta[ti_row + tidx, head_idx])
                    sBeta[tidx] = beta_val

                for _r in cutlass.range_constexpr(self.BC):
                    g_ref_val = sGref[col]

                    q_val = cutlass.Float32(0.0)
                    k_val_r = cutlass.Float32(0.0)
                    g_val_r = g_ref_val
                    if _r < actual_row:
                        q_val = cutlass.Float32(mQ[ti_row + _r, h_col])
                        k_val_r = cutlass.Float32(mK[ti_row + _r, h_col])
                        g_val_r = mG[ti_row + _r, h_col]
                    gate_r = cute.math.exp2(g_val_r - g_ref_val, fastmath=True)
                    sQg[_r, col] = self._dtype(q_val * gate_r)
                    sKp[_r, col] = self._dtype(k_val_r * gate_r)

                    k_val_c = cutlass.Float32(0.0)
                    g_val_c = g_ref_val
                    if _r < actual_col:
                        k_val_c = cutlass.Float32(mK[ti_col + _r, h_col])
                        g_val_c = mG[ti_col + _r, h_col]
                    gate_c = cute.math.exp2(g_ref_val - g_val_c, fastmath=True)
                    sKn[_r, col] = self._dtype(k_val_c * gate_c)
        else:
            # ── Non-varlen: constexpr-unrolled, zero guards ──
            if tidx < self.D:
                sGref[tidx] = mG[ti_row, h_offset + tidx]
            if tidx < self.BC:
                sBeta[tidx] = cutlass.Float32(mBeta[ti_row + tidx, head_idx])
            for _r in cutlass.range_constexpr(self.BC):
                col = tidx
                g_ref_val = sGref[col]
                q_val = cutlass.Float32(mQ[ti_row + _r, h_offset + col])
                k_val_r = cutlass.Float32(mK[ti_row + _r, h_offset + col])
                g_val_r = mG[ti_row + _r, h_offset + col]
                gate_r = cute.math.exp2(g_val_r - g_ref_val, fastmath=True)
                sQg[_r, col] = self._dtype(q_val * gate_r)
                sKp[_r, col] = self._dtype(k_val_r * gate_r)
                k_val_c = cutlass.Float32(mK[ti_col + _r, h_offset + col])
                g_val_c = mG[ti_col + _r, h_offset + col]
                gate_c = cute.math.exp2(g_ref_val - g_val_c, fastmath=True)
                sKn[_r, col] = self._dtype(k_val_c * gate_c)

        cute.arch.barrier(barrier_id=1, number_of_threads=self.num_threads)

        # ══════════════════════════════════════════════════════════
        # Phase 2: MMA — Warp 0: Aqk = sQg @ sKn^T
        #                  Warp 1: Akk = sKp @ sKn^T
        # ══════════════════════════════════════════════════════════
        thr_mma = tiled_mma.get_slice(lane_idx)

        thrA = smem_copy_A.get_slice(lane_idx)
        _tCsQg = thrA.partition_S(sQg)
        _tCsKp = thrA.partition_S(sKp)
        _tMmaQg = thr_mma.partition_A(sQg)
        _tMmaKp = thr_mma.partition_A(sKp)
        tCrQg = thr_mma.make_fragment_A(_tMmaQg)
        tCrKp = thr_mma.make_fragment_A(_tMmaKp)
        tCrQgc = thrA.retile(tCrQg)
        tCrKpc = thrA.retile(tCrKp)

        thrB = smem_copy_B.get_slice(lane_idx)
        _tCsKn_B = thrB.partition_S(sKn)
        _tMmaKn = thr_mma.partition_B(sKn)
        tCrKn = thr_mma.make_fragment_B(_tMmaKn)
        tCrKnc = thrB.retile(tCrKn)

        acc_shape = thr_mma.partition_shape_C((self.BC, self.BC))
        acc_Aqk = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc_Akk = cute.make_rmem_tensor(acc_shape, cutlass.Float32)
        acc_Aqk.fill(0.0)
        acc_Akk.fill(0.0)

        num_k_blocks = cute.size(_tCsQg, mode=[2])

        if warp_idx == 0:
            for kb in cutlass.range_constexpr(num_k_blocks):
                cute.copy(smem_copy_A, _tCsQg[None, None, kb], tCrQgc[None, None, kb])
                cute.copy(smem_copy_B, _tCsKn_B[None, None, kb], tCrKnc[None, None, kb])
                cute.gemm(
                    tiled_mma,
                    acc_Aqk,
                    tCrQg[None, None, kb],
                    tCrKn[None, None, kb],
                    acc_Aqk,
                )
        elif warp_idx == 1:
            for kb in cutlass.range_constexpr(num_k_blocks):
                cute.copy(smem_copy_A, _tCsKp[None, None, kb], tCrKpc[None, None, kb])
                cute.copy(smem_copy_B, _tCsKn_B[None, None, kb], tCrKnc[None, None, kb])
                cute.gemm(
                    tiled_mma,
                    acc_Akk,
                    tCrKp[None, None, kb],
                    tCrKn[None, None, kb],
                    acc_Akk,
                )

        # ══════════════════════════════════════════════════════════
        # Phase 3: Scaled stores to GMEM
        # ══════════════════════════════════════════════════════════
        cC = cute.make_identity_tensor((self.BC, self.BC))
        tCcC = thr_mma.partition_C(cC)

        od_row = self.upcast(chunk_idx) * self.num_offdiag_blocks + pair_idx

        if warp_idx == 0:
            out_dtype = mAqk.element_type
            for i in cutlass.range_constexpr(cute.size(acc_Aqk)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                if cutlass.const_expr(self.chunk_schedule is not ChunkSchedule.DENSE):
                    if chunk_base + self.BT <= eos or ti_row + row < eos:
                        mAqk[ti_row + row, aqk_col + ci * self.BC + col] = out_dtype(
                            acc_Aqk[i] * scale
                        )
                else:
                    mAqk[ti_row + row, aqk_col + ci * self.BC + col] = out_dtype(
                        acc_Aqk[i] * scale
                    )
        elif warp_idx == 1:
            for i in cutlass.range_constexpr(cute.size(acc_Akk)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                beta_val = sBeta[row]
                mAkkOD[od_row, akkod_col + row * self.BC + col] = acc_Akk[i] * beta_val
        elif warp_idx == 2:
            # Define the noncausal block while this CTA already owns the
            # corresponding lower block. Aqk is exposed as custom-op tape, so
            # its complete storage must be deterministic for fake/AOT checks.
            out_dtype = mAqk.element_type
            for i in cutlass.range_constexpr(cute.size(acc_Aqk)):
                row = tCcC[i][0]
                col = tCcC[i][1]
                if cutlass.const_expr(self.chunk_schedule is not ChunkSchedule.DENSE):
                    if ti_col + row < eos:
                        mAqk[ti_col + row, aqk_col + ri * self.BC + col] = out_dtype(0.0)
                else:
                    mAqk[ti_col + row, aqk_col + ri * self.BC + col] = out_dtype(0.0)
