# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by Attention Gym in 2026: shared indexer kernel helpers.

"""Shared helpers for the SM100 indexer score and Top-K kernel classes."""

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.warpdrive import Local, ProtocolError, Tmem, WarpSpecialized


class IndexerKernelBase:
    """Offset widening and causal visibility shared by every indexer kernel.

    Subclasses set ``use_int64_offsets`` and ``compress_ratio`` in ``__init__``.
    """

    use_int64_offsets: bool
    compress_ratio: int

    @cute.jit
    def upcast_offset(self, value):
        """Widen origins before address arithmetic in the wide specialization."""
        return Int64(value) if cutlass.const_expr(self.use_int64_offsets) else Int32(value)

    @cute.jit
    def visible_candidates(self, query):
        """Causal candidate count for query; the static ratio keeps r=1 division-free."""
        if cutlass.const_expr(self.compress_ratio == 1):
            return query + 1
        return (query + 1) // self.compress_ratio


@dataclass(frozen=True)
class ScoreParams:
    """Kernel-prologue values the roles read."""

    tiled_mma: cute.TiledMma
    tma_k: cute.CopyAtom
    k: cute.Tensor  # K TMA coordinates: (S, D, B)
    tma_q: cute.CopyAtom
    q: cute.Tensor  # Q TMA coordinates, in the kernel's ``q_operand`` order
    sK: cute.Tensor
    sQ: cute.Tensor
    sW: cute.Tensor | None  # the kernel's ``weight_stage``
    acc_layout: cute.Layout  # every accumulator stage
    weights: cute.Tensor
    scores: cute.Tensor
    pair_start: cutlass.Numeric
    score_scale: Float32
    candidate_bounds: cute.Tensor | None


class IndexerScoreKernelBase(IndexerKernelBase, WarpSpecialized):
    """Launch, prologue and epilogue reduction shared by the score kernels.

    A subclass declares the ``K`` and ``Q`` buffers, its channels with an ``acc`` ring of
    ``acc_stages`` TMEM stages, the ``work`` stream and the roles. It sets ``mma_tile``
    (candidates, heads per MMA, D per stage), the ``k_stages`` and ``q_stages`` of the two
    buffers, and defines ``q_operand`` and ``work_space``.
    """

    name_prefix: str
    heads: int
    head_dim: int
    causal: bool
    contiguous_weight_heads: bool
    mma_tile: tuple[int, int, int]
    k_stages: int
    q_stages: int
    acc_stages = 2

    def get_name(self) -> str:
        """Return a stable name for the shape, mask, weight layout, and offset width."""
        return (
            f"{self.name_prefix}_h{self.heads}_d{self.head_dim}_c{int(self.causal)}_"
            f"r{self.compress_ratio}_i64{int(self.use_int64_offsets)}_"
            f"wh{int(self.contiguous_weight_heads)}"
        )

    def weight_stage(self, smem) -> cute.Tensor | None:
        """SMEM in which the epilogue stages weights; None when it loads them per thread."""
        return None

    @cute.jit
    def accumulator_reader(self, params: ScoreParams, tmem: Tmem, me: Local):
        """Build one epilogue thread's TMEM load of its candidate row of an accumulator stage.

        Returns ``(accumulator, tmem_copy, tmem_thread, candidate, logits)``: every accumulator
        stage, the copy and this thread's slice of it, the thread's candidate within an M128
        tile, and the register fragment receiving its ``mma_tile[1]`` logits.
        """
        if cutlass.const_expr(me.warps != 4):
            raise ProtocolError(
                f"epilogue: one thread per M128 candidate needs 4 warps, got {me.warps}"
            )
        tidx = me.tmem_row
        accumulator = cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        acc_tile = accumulator[(None, None, None, 0)]
        tmem_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.mma_tile[1] // 4)),
            Float32,
        )
        tmem_copy = tcgen05.make_tmem_copy(tmem_atom, acc_tile)
        tmem_thread = tmem_copy.get_slice(tidx)
        coordinates = tmem_thread.partition_D(
            params.tiled_mma.get_slice(tidx).partition_C(
                cute.make_identity_tensor(self.mma_tile[:2])
            )
        )
        logits = cute.make_rmem_tensor(
            tmem_thread.partition_D(cute.make_identity_tensor(acc_tile.shape)).shape, Float32
        )
        assert cute.size(logits) == self.mma_tile[1]
        return accumulator, tmem_copy, tmem_thread, coordinates[0][0], logits

    @cute.jit
    def add_weighted_relu(
        self, sums, logits, weights, first: cutlass.Constexpr, heads: cutlass.Constexpr
    ):
        """Add ``weights[h] * ReLU(logits[h])`` for ``heads`` heads from ``first`` to ``sums``.

        ``sums`` is four independent FP32 chains, each taking every fourth head: they hide add
        latency without packed PTX or cross-thread reductions.
        """
        sum0, sum1, sum2, sum3 = sums
        for group in cutlass.range_constexpr(heads // 4):
            offset = first + group * 4
            x0 = Float32(logits[offset])
            x1 = Float32(logits[offset + 1])
            x2 = Float32(logits[offset + 2])
            x3 = Float32(logits[offset + 3])
            x0 = x0 if x0 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
            x1 = x1 if x1 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
            x2 = x2 if x2 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
            x3 = x3 if x3 > Float32(0.0) else Float32(0.0)  # noqa: FURB136
            sum0 = sum0 + x0 * weights[offset]
            sum1 = sum1 + x1 * weights[offset + 1]
            sum2 = sum2 + x2 * weights[offset + 2]
            sum3 = sum3 + x3 * weights[offset + 3]
        return sum0, sum1, sum2, sum3

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_k: cute.CopyAtom,
        k: cute.Tensor,
        tma_atom_q: cute.CopyAtom,
        q: cute.Tensor,
        weights: cute.Tensor,
        scores: cute.Tensor,
        k_layout: cute.ComposedLayout,
        q_layout: cute.ComposedLayout,
        pair_start,
        score_scale: Float32,
        candidate_bounds: cute.Tensor | None,
        bound,
    ):
        """Bind the K/Q buffers and any weight stage, then run the roles."""
        smem = self.begin_roles(work=bound, layouts={"K": k_layout, "Q": q_layout})
        accumulator_shape = tiled_mma.partition_shape_C(self.mma_tile[:2])
        params = ScoreParams(
            tiled_mma=tiled_mma,
            tma_k=tma_atom_k,
            k=k,
            tma_q=tma_atom_q,
            q=q,
            sK=self.buffer("K"),
            sQ=self.buffer("Q"),
            sW=self.weight_stage(smem),
            acc_layout=tiled_mma.make_fragment_C(
                cute.append(accumulator_shape, self.acc_stages)
            ).layout,
            weights=weights,
            scores=scores,
            pair_start=pair_start,
            score_scale=score_scale,
            candidate_bounds=candidate_bounds,
        )
        self.run_roles(params)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        weights: cute.Tensor,
        scores: cute.Tensor,
        pair_start,
        score_scale: Float32,
        candidate_bounds: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        """Build the TMA descriptors and launch the work items of the existing slab pairs.

        ``candidate_bounds`` selects the packed specialization, which scores only each query's
        interval; None scores its full or causal candidate prefix.
        """
        assert q.element_type == self.dtype
        assert k.element_type == q.element_type and weights.element_type == q.element_type
        assert scores.element_type == Float32
        k_sdb = cute.make_tensor(k.iterator, cute.select(k.layout, mode=[1, 2, 0]))
        tiled_mma = cute.make_tiled_mma(
            tcgen05.MmaF16BF16Op(
                q.element_type,
                Float32,
                (*self.mma_tile[:2], 16),
                tcgen05.CtaGroup.ONE,
                tcgen05.OperandSource.SMEM,
                cute.nvgpu.OperandMajorMode.K,
                cute.nvgpu.OperandMajorMode.K,
            )
        )
        k_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tile, k.element_type, self.k_stages
        )
        q_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tile, q.element_type, self.q_stages
        )
        tma_load = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_atom_k, k_tma = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load,
            k_sdb,
            cute.select(k_layout, mode=[0, 1, 2]),
            self.mma_tile,
            tiled_mma,
        )
        tma_atom_q, q_tma = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load,
            self.q_operand(q),
            cute.select(q_layout, mode=[0, 1, 2]),
            self.mma_tile,
            tiled_mma,
        )
        pair_start = self.upcast_offset(pair_start)
        total_pairs = self.upcast_offset(weights.shape[0]) * cute.ceil_div(
            self.upcast_offset(weights.shape[1]), 2
        )
        # The public launcher slices the slab to the remaining pairs. A longer slab from a
        # direct caller keeps its extra rows untouched.
        pairs = cutlass.min(self.upcast_offset(scores.shape[0]), total_pairs - pair_start)
        bound = self.work.bind(self.work_space(cutlass.Int32(pairs)))
        packed = "_packed" if cutlass.const_expr(candidate_bounds is not None) else ""
        self.kernel.set_name_prefix(f"{self.get_name()}_{q.element_type.__name__.lower()}{packed}")
        self.kernel(
            tiled_mma,
            tma_atom_k,
            k_tma,
            tma_atom_q,
            q_tma,
            weights,
            scores,
            k_layout,
            q_layout,
            pair_start,
            score_scale,
            candidate_bounds,
            bound,
        ).launch(**self.launch_config(bound), stream=stream)
