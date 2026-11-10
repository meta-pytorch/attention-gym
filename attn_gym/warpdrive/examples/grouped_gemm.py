"""Persistent BF16 grouped GEMM ``out[rows_g] = A[rows_g] @ B[g].T`` on SM100.

``A: (total_M, K)`` packs groups back to back; ``B: (G, N, K)`` holds one weight per group.
Device int32 ``offsets: (G,)`` gives each group's end row, as in ``torch._grouped_mm``.
The host knows only G, total_M, N and K. ``GroupedTiles`` decodes
``(group, row0, rows, n_tile)`` work items on the device.

Each operand uses one TMA descriptor for all groups. A and output use row offsets;
B uses the group coordinate of an ``(N, K, G)`` view. No device-side descriptor patching
is needed. Roles and channels follow ``persistent_gemm.py``.

TMA clips at tensor edges, not group boundaries. Output uses a ragged TMA view of shape
``(2^30, N, 2^31 - 1)``: the first mode supplies the clipping edge, and the third has stride N
to select the real row. A tile with ``rows`` valid rows ending at ``end`` starts at
``(2^30 - rows, n, end)``. Stores past the group's end fall outside the view and are dropped.
Ragged A loads can read the next group's rows, but only into accumulator rows never stored.
"""

from dataclasses import dataclass
from typing import NamedTuple

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

from attn_gym.warpdrive import (
    Consumer,
    Local,
    NamedBarrier,
    Producer,
    ProtocolError,
    Ring,
    RoleConfig,
    SingleCta,
    SmemPlan,
    StageView,
    StaticPersistent,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    WarpSpecialized,
    WorkCursor,
    WorkMap,
    WorkStream,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.compat import LayoutEnum
from attn_gym.warpdrive.examples.gemm import _overlaps
from attn_gym.warpdrive.examples.persistent_gemm import (
    CTA_TILE_M,
    EPI_TILE_NS,
    MAX_C_STAGES,
    TILE_K,
    _frozen_roles,
)
from attn_gym.warpdrive.examples.tma_tiles import tma_partition_tiles

RAGGED_ROWS = 2**30  # the row extent of the ragged output view; TMA clips there
RAGGED_SELECT = 2**31 - 1  # extent of its row-selecting third mode
INT32_MAX = 2**31 - 1
LOOKAHEAD = 8  # groups a decode tests at once past its last group
MAX_TILE_M = 2 * CTA_TILE_M  # a CTA pair's MMA tile


class Window(NamedTuple):
    """Decode state: ``group`` owns m-tiles ``[tile_lo, tile_hi)`` and rows ``[row_lo, row_hi)``."""

    group: Int32
    tile_lo: Int32
    tile_hi: Int32
    row_lo: Int32
    row_hi: Int32


@cute.jit
def _seek(window: Window, mt: Int32, offsets: cute.Tensor, total_m: Int32, tile_m: int):
    """Find the group owning m-tile ``mt``, reusing ``window`` without loads if it owns it.

    Walk forward from ``window``, or restart at group 0 for a backward index. Each round
    loads ``LOOKAHEAD`` independent offsets and sums ``ceil(size_g / tile_m)`` tile counts.
    Past the last group, return an empty window covering the rest of the index space.

    Clamp each group end to ``[previous_end, total_m]``. Invalid offsets give wrong routing,
    not out-of-bounds rows, and every starting window gives the same result as group 0.
    """
    if (mt < window.tile_lo) | (mt >= window.tile_hi):
        G = offsets.shape[0]
        back = mt < window.tile_lo
        # Group g is the next to visit; it starts at m-tile `tile` and row `row`.
        g = cutlass.select_(back, Int32(0), window.group + 1)
        tile = cutlass.select_(back, Int32(0), window.tile_hi)
        row = cutlass.select_(back, Int32(0), window.row_hi)
        found = cutlass.Boolean(False)
        while ~found & (g < G):
            for j in cutlass.range_constexpr(LOOKAHEAD):
                end = cutlass.min(cutlass.max(offsets[cutlass.min(g + j, G - 1)], row), total_m)
                next_tile = tile + (end - row + tile_m - 1) // tile_m
                owns = (g + j < G) & (tile <= mt) & (mt < next_tile)
                window = Window(
                    *(
                        cutlass.select_(owns, new, old)
                        for new, old in zip(
                            (g + j, tile, next_tile, row, end), window, strict=True
                        )
                    )
                )
                found = found | owns
                tile, row = next_tile, end
            g = g + LOOKAHEAD
        if ~found:
            window = Window(G - 1, tile, Int32(INT32_MAX), row, row)
    return window


@dataclass(frozen=True)
class GroupedTiles(WorkMap):
    """Decode grouped GEMM items ``(group, row0, rows, n_tile)`` on the device.

    The space is ``(offsets, total_m, n_tiles)`` with G >= 1. Group m-tiles are numbered
    back to back, with n fastest so consecutive items reuse A rows. The host cannot read
    ``sum(ceil(size_g / tile_m))``, so it uses the upper bound ``ceil(total_m / tile_m) + G``
    and drops indices past the last real tile. Keeping padding at the end avoids holes
    that would unbalance static persistence.

    Decode state is the last group's ``Window`` (5 Int32 per decoding warp), including
    the prefix sum of group tile counts. Increasing indices resume that walk; backward
    indices restart at group 0. A fresh or restarted decode takes up to
    ``ceil(G / LOOKAHEAD)`` rounds of offset loads.
    """

    tile_m: int = CTA_TILE_M
    words = 4
    skips = True

    def bind(self, space):
        offsets, total_m, n_tiles = space
        return offsets, Int32(total_m), Int32(cutlass.max(n_tiles, 0))

    def count64(self, space):
        offsets, total_m, n_tiles = space
        m_tiles = (Int64(total_m) + self.tile_m - 1) // self.tile_m + offsets.shape[0]
        return m_tiles * Int64(n_tiles)

    def start(self, space):
        # Group -1's empty window: the first seek searches from group 0.
        return Window(Int32(-1), Int32(0), Int32(0), Int32(0), Int32(0))

    def decode(self, index, space, state):
        offsets, total_m, n_tiles = space
        mt = index // n_tiles
        w = _seek(state, mt, offsets, total_m, self.tile_m)
        row0 = w.row_lo + (mt - w.tile_lo) * self.tile_m
        rows = cutlass.min(w.row_hi - row0, self.tile_m)
        return row0 < w.row_hi, (w.group, row0, rows, index % n_tiles), w

    def get_key(self) -> str:
        return f"grouped{self.tile_m}"

    def describe(self) -> str:
        return f"GroupedTiles(tile_m={self.tile_m}, state=Window: 5 x Int32)"


@dataclass(frozen=True)
class GroupedGemmParams:
    """Layouts, partitions and runtime sizes shared by the grouped GEMM roles."""

    tiled_mma: cute.TiledMma
    tma_a: cute.CopyAtom
    tma_b: cute.CopyAtom
    mA: cute.Tensor  # TMA coordinate tensor of A (total_M, K)
    sA: cute.Tensor  # the A stages, for the per-item TMA partition
    tAsA: StageView  # TMA partitions of the A and B stages: (TMA, stage)
    tBsB: StageView
    tBgB: cute.Tensor  # whole-problem B tiles: (TMA, n_tiles, k_tiles, G)
    tCrA: StageView  # MMA fragments over the SMEM stages: (MMA, MMA_M, MMA_K, stage)
    tCrB: StageView
    num_k_tiles: cutlass.Int32
    acc_layout: cute.Layout  # all accumulator stages
    epi_tile: cute.Tile
    tma_c: cute.CopyAtom
    mC_tma: cute.Tensor  # ragged TMA coordinate tensor of the output (RAGGED_ROWS, N, SELECT)
    mC: cute.Tensor  # the output itself (total_M, N), for the fragment layouts
    sC: cute.Tensor  # epilogue staging buffers


class GroupedGemm(WarpSpecialized):
    def __init__(
        self,
        schedule=None,
        ab_stages: int | None = None,
        acc_stages: int = 2,
        trace: bool = False,
        cluster=None,
        tile_n: int = 256,
        c_stages: int = 2,
        epi_tile_n: int = 32,
        roles: dict[str, RoleConfig] | None = None,
    ):
        """Configure the grouped tile loop, channels and roles.

        ``ab_stages`` defaults to the depth that fills about 192 KB of SMEM.
        ``acc_stages`` counts TMEM accumulators; ``tile_n`` is the output tile width.
        The epilogue uses ``c_stages`` SMEM buffers of ``128 x epi_tile_n``.
        ``roles`` overrides placement and register budgets by name (defaults: 40/40/160).
        """
        self.cluster = cluster or SingleCta()
        cs = self.cluster.pair_size  # CTAs per MMA
        if self.cluster.size != cs:
            raise ValueError(f"GroupedGemm runs on one CTA or a CTA pair, got {self.cluster}")
        if epi_tile_n not in EPI_TILE_NS:
            raise ValueError(f"epi_tile_n must be one of {EPI_TILE_NS}, got {epi_tile_n}")
        if tile_n % epi_tile_n or not 0 < tile_n <= 256 or tile_n // cs % 8:
            raise ValueError(
                f"tile_n must be a multiple of epi_tile_n={epi_tile_n} in (0, 256], got {tile_n}"
            )
        if not 1 <= c_stages <= MAX_C_STAGES:
            raise ValueError(f"c_stages must be in [1, {MAX_C_STAGES}], got {c_stages}")
        self.tile_n = tile_n
        self.epi_tile = (CTA_TILE_M, epi_tile_n)
        self.c_stages = c_stages
        self.roles = roles
        cta_tile = (CTA_TILE_M, tile_n, TILE_K)
        self.mma_tile = (cta_tile[0] * cs, cta_tile[1], cta_tile[2])
        self.dtype = cutlass.BFloat16
        # Attributes read by the CUTLASS epilogue partition helpers.
        self.c_dtype, self.acc_dtype = self.dtype, cutlass.Float32
        self.cta_tile_shape_mnk = cta_tile
        self.c_layout = LayoutEnum.ROW_MAJOR
        self.acc_stages = acc_stages
        stage_bytes = (cta_tile[0] + cta_tile[1] // cs) * cta_tile[2] * 2  # this CTA's copies
        if ab_stages is None:  # fill ~192 KB of SMEM
            ab_stages = 192 * 1024 // stage_bytes
        self.smem_plan = SmemPlan()
        a = self.smem_plan.buffer("A", self.dtype, align=1024)
        b = self.smem_plan.buffer("B", self.dtype, align=1024)
        self.smem_plan.buffer("C", self.dtype, align=128)
        self.ab = Ring(ab_stages, Tma(stage_bytes), Umma, buffer={"A": a, "B": b})
        self.acc = Ring(acc_stages, Umma, Threads)
        # The 4 epilogue warps meet before warp 0 stores a staged subtile and before any warp
        # rewrites it.
        self.epi_sync = NamedBarrier(threads=128)
        self.tmem_plan = TmemPlan().region("acc", cta_tile[1], acc_stages)
        # Tiles have equal work and decode packs them back to back. A static schedule
        # avoids a scheduler warp and workspace without scattering padding among units.
        self.work = WorkStream(
            mapping=GroupedTiles(tile_m=self.mma_tile[0]), schedule=schedule or StaticPersistent()
        )
        self.trace_protocol = trace

    def get_name(self) -> str:
        return (
            f"warpdrive_ggemm_{self.cluster.get_key()}_n{self.tile_n}_ab{self.ab.n}_"
            f"acc{self.acc_stages}_c{self.c_stages}x{self.epi_tile[1]}_"
            f"{self.work.get_key()}{self.trace_key()}"
        )

    @role(warps=4, regs=40)
    @cute.jit
    def load(self, params: GroupedGemmParams, *, ab: Producer, work: WorkCursor):
        ab = ab.views(A=params.tAsA, B=params.tBsB)
        thr_mma = params.tiled_mma.get_slice(self.cluster.coords(self.cluster.rank())[0])
        while work.valid:
            g, row0, _, nt = work.item
            # This item's rows of A, tiled along K: (TM, TK, k_tiles) -> (TMA, k_tiles).
            gA = cute.local_tile(
                cute.domain_offset((row0, 0), params.mA),
                self.mma_tile,
                (0, None, None),
                proj=(1, None, 1),
            )
            _, tAgA = tma_partition_tiles(params.tma_a, params.sA, thr_mma.partition_A(gA))
            tBgB = params.tBgB[None, nt, None, g]
            for k in cutlass.range(params.num_k_tiles, unroll=1):
                with ab.produce() as s:
                    a, b = s.view("A"), s.view("B")
                    cute.copy(params.tma_a, tAgA[None, k], a, tma_bar_ptr=s.barrier)
                    cute.copy(params.tma_b, tBgB[None, k], b, tma_bar_ptr=s.barrier)
            work.advance()
        return ab, work

    @role(warps=5, regs=40, cta="leader")
    @cute.jit
    def mma(
        self,
        params: GroupedGemmParams,
        *,
        ab: Consumer,
        acc: Producer,
        work: WorkCursor,
        tmem: Tmem,
    ):
        tiled_mma = params.tiled_mma
        tCtAcc = cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        ab = ab.views(A=params.tCrA, B=params.tCrB)
        while work.valid:
            with acc.produce() as a:
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k in cutlass.range(params.num_k_tiles, unroll=1):
                    with ab.consume() as s:
                        for kb in cutlass.range_constexpr(cute.size(params.tCrA.tensor, mode=[2])):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc[None, None, None, a.index],
                                s.view("A", None, None, kb),
                                s.view("B", None, None, kb),
                                tCtAcc[None, None, None, a.index],
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            work.advance()
        return ab, acc, work

    @role(warps=range(4), regs=160)
    @cute.jit
    def epilogue(
        self,
        params: GroupedGemmParams,
        *,
        acc: Consumer,
        work: WorkCursor,
        tmem: Tmem,
        me: Local,
    ):
        """Store output through registers, swizzled SMEM and TMA.

        Each of the ``c_stages`` buffers holds one ``epi_tile``. The ragged TMA view
        clips rows at the group's end and columns at N.
        """
        if cutlass.const_expr(me.warps != 4):
            raise ProtocolError(
                f"epilogue: the TMEM load covers 128 rows with 4 warps, got {me.warps}"
            )
        tidx = me.tmem_row
        warp = me.warp
        rank = self.cluster.coords(self.cluster.rank())[0]  # this CTA's rows of the MMA tile
        thr_mma = params.tiled_mma.get_slice(rank)
        tAcc = transform_partitioned_tensor_layout(
            cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        )
        # Whole-problem tiles only size the register fragment; items address their own rows.
        gC_all = transform_partitioned_tensor_layout(
            thr_mma.partition_C(
                cute.local_tile(params.mC, self.mma_tile, (None, None, None), proj=(1, 1, None))
            )
        )
        gC_l = cute.make_tensor(gC_all.iterator, cute.append(gC_all.layout, cute.make_layout(1)))
        t2r, tTR_tAcc, tTR_rAcc = epilogue_tmem_copy_and_partition(
            self, tidx, tAcc, gC_l, params.epi_tile, self.cluster.pair
        )
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.dtype)
        r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(self, t2r, tTR_rC, tidx, params.sC)
        bSG_sC, _ = tma_partition_tiles(
            params.tma_c, params.sC, cute.flat_divide(gC_all, params.epi_tile), tile_rank=2
        )
        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc) - 1)  # (..., SUB, STAGE)
        subtiles = cute.size(tTR_tAcc, mode=[3])
        # Continue buffer rotation across tiles when subtiles is not divisible by c_stages
        # (e.g. tile_n=224 has 7 subtiles at epi_tile_n=32).
        rotate = subtiles % self.c_stages != 0
        first = Int32(0)  # the buffer of this tile's first subtile
        while work.valid:
            _, row0, rows, nt = work.item
            # This CTA's rows of the MMA tile; a pair's peer may hold none of a ragged tile.
            rows = cutlass.max(cutlass.min(rows - CTA_TILE_M * rank, CTA_TILE_M), 0)
            end = row0 + CTA_TILE_M * rank + rows
            # The ragged view's rows [RAGGED_ROWS - rows, RAGGED_ROWS) alias C's rows
            # [end - rows, end); the MMA-tile partition below adds this CTA's row offset.
            mC_item = cute.domain_offset(
                (RAGGED_ROWS - rows - CTA_TILE_M * rank, 0), params.mC_tma[None, None, end]
            )
            gC_tma = cute.flat_divide(
                transform_partitioned_tensor_layout(
                    thr_mma.partition_C(cute.local_tile(mC_item, self.mma_tile[:2], (0, nt)))
                ),
                params.epi_tile,
            )
            _, bSG_gC = tma_partition_tiles(params.tma_c, params.sC, gC_tma, tile_rank=2)
            with acc.consume() as a:
                for i in cutlass.range_constexpr(subtiles):
                    buf = (first + i) % self.c_stages if rotate else i % self.c_stages
                    cute.copy(t2r, tTR_tAcc[None, None, None, i, a.index], tTR_rAcc)
                    tTR_rC.store(tTR_rAcc.load().to(self.dtype))
                    cute.copy(r2s, tRS_rC, tRS_sC[None, None, None, buf])
                    cute.arch.fence_view_async_shared()  # st.shared -> TMA store
                    # All epilogue threads reach both syncs convergently, allowing aligned=True.
                    self.epi_sync.sync(aligned=True)
                    if warp == 0:
                        cute.copy(params.tma_c, bSG_sC[None, buf], bSG_gC[None, 0, i])
                        cute.arch.cp_async_bulk_commit_group()
                        # Buffer `buf` is rewritten c_stages subtiles later; keep at most
                        # c_stages - 1 stores still reading SMEM.
                        cute.arch.cp_async_bulk_wait_group(self.c_stages - 1, read=True)
                    self.epi_sync.sync(aligned=True)
            if cutlass.const_expr(rotate):
                first = (first + subtiles) % self.c_stages
            work.advance()
        if warp == 0:
            cute.arch.cp_async_bulk_wait_group(0, read=True)
        return acc, work

    @cute.kernel
    def kernel(
        self,
        tiled_mma,
        tma_a,
        mA,
        tma_b,
        mB,
        tma_c,
        mC_tma,
        mC,
        a_layout,
        b_layout,
        c_layout,
        bound,
    ):
        self.begin_roles(work=bound, layouts={"A": a_layout, "B": b_layout, "C": c_layout})
        sA, sB, sC = self.buffer("A"), self.buffer("B"), self.buffer("C")
        mma_rank = self.cluster.coords(self.cluster.rank())[0]
        thr_mma = tiled_mma.get_slice(mma_rank)
        # Whole-problem B tiles (BN, BK, n_tiles, k_tiles, G); A tiles are built per item.
        gB = cute.local_tile(mB, self.mma_tile[1:], (None, None, None))
        gA0 = cute.local_tile(mA, self.mma_tile, (0, None, None), proj=(1, None, 1))
        tAsA, _ = tma_partition_tiles(tma_a, sA, thr_mma.partition_A(gA0))
        tBsB, tBgB = tma_partition_tiles(tma_b, sB, thr_mma.partition_B(gB))
        acc_shape = tiled_mma.partition_shape_C(self.mma_tile[:2])
        acc_staged = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))
        params = GroupedGemmParams(
            tiled_mma=tiled_mma,
            tma_a=tma_a,
            tma_b=tma_b,
            mA=mA,
            sA=sA,
            tAsA=StageView(tAsA, axis=1),
            tBsB=StageView(tBsB, axis=1),
            tBgB=tBgB,
            tCrA=StageView(tiled_mma.make_fragment_A(sA), axis=3),
            tCrB=StageView(tiled_mma.make_fragment_B(sB), axis=3),
            num_k_tiles=cute.size(gB, mode=[3]),
            acc_layout=acc_staged.layout,
            epi_tile=self.epi_tile,
            tma_c=tma_c,
            mC_tma=mC_tma,
            mC=mC,
            sC=sC,
        )
        self.run_roles(params)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        offsets: cute.Tensor,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        tile, group = self.mma_tile, self.cluster.cta_group
        op = tcgen05.MmaF16BF16Op(
            self.dtype,
            cutlass.Float32,
            (tile[0], tile[1], 16),
            group,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, tile, self.dtype, self.ab.n)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, tile, self.dtype, self.ab.n)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(group)
        vmnk = self.cluster.layout_vmnk().shape
        tma_a, mA = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op, a, cute.select(a_layout, mode=[0, 1, 2]), tile, tiled_mma, vmnk
        )
        # B as (N, K, G): the group is the batch coordinate of one descriptor.
        b_nkg = cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 2, 0]))
        tma_b, mB = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op, b_nkg, cute.select(b_layout, mode=[0, 1, 2]), tile, tiled_mma, vmnk
        )
        c_layout = sm100_utils.make_smem_layout_epi(
            self.dtype, self.c_layout, self.epi_tile, self.c_stages
        )
        # The ragged output view (see the module docstring); its base lies RAGGED_ROWS rows
        # before C, which a store-only descriptor tolerates.
        row_stride = c.stride[0]
        c_ragged = cute.make_tensor(
            cute.domain_offset((-RAGGED_ROWS, 0), c).iterator,
            cute.make_layout(
                (RAGGED_ROWS, c.shape[1], RAGGED_SELECT), stride=(row_stride, 1, row_stride)
            ),
        )
        tma_c, mC_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            c_ragged,
            cute.select(c_layout, mode=[0, 1]),
            cute.composition(cute.make_identity_layout(c_ragged.shape), self.epi_tile),
        )
        n_tiles = cute.ceil_div(c.shape[1], tile[1])
        bound = self.work.bind((offsets, Int32(a.shape[0]), n_tiles), workspace, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            tiled_mma,
            tma_a,
            mA,
            tma_b,
            mB,
            tma_c,
            mC_tma,
            c,
            a_layout,
            b_layout,
            c_layout,
            bound,
        ).launch(**self.launch_config(bound), stream=stream)


@jit_cache(uncached=lambda op: op.trace_protocol)
def compile_grouped_gemm(op: GroupedGemm):
    """Compile for dynamic total_M, G, N and K.

    K and N must be multiples of 8 for 16-byte TMA strides. Memory and disk caches
    use ``op.__cute_cache_key__()``.
    """
    m, g = cute.sym_int(), cute.sym_int()
    n, k = cute.sym_int(divisibility=8), cute.sym_int(divisibility=8)
    bf16 = cutlass.BFloat16
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (m, k)),
        make_fake_compact_tensor(bf16, (g, n, k)),
        make_fake_compact_tensor(bf16, (m, n)),
        make_fake_compact_tensor(cutlass.Int32, (g,)),
        make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )


def check_grouped_gemm_args(
    a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor, out: torch.Tensor | None
) -> torch.Tensor:
    """Check the operands and return ``out``, allocating it if None.

    ``offsets`` holds nondecreasing group end rows in ``[0, total_M]``. Values are read
    only on the device, not checked here. The kernel clamps invalid offsets to safe rows
    but does not correct the routing.

    ``out`` must not overlap inputs: a store could overwrite rows another tile still loads.
    ``total_M + (G + 1) * tile_m`` must fit in Int32 for decoding.
    """
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("grouped_gemm expects bfloat16 operands")
    if offsets.dtype != torch.int32:
        raise TypeError(f"offsets must be int32, got {offsets.dtype}")
    if a.ndim != 2 or b.ndim != 3 or a.shape[1] != b.shape[2]:
        raise ValueError(f"incompatible shapes {tuple(a.shape)}, {tuple(b.shape)}")
    if a.shape[0] + (b.shape[0] + 1) * MAX_TILE_M > INT32_MAX:
        raise ValueError(f"total_M + (G + 1) * {MAX_TILE_M} must fit in int32")
    if offsets.ndim != 1 or offsets.shape[0] != b.shape[0]:
        raise ValueError(f"offsets must have shape ({b.shape[0]},), got {tuple(offsets.shape)}")
    if not (a.is_contiguous() and b.is_contiguous() and offsets.is_contiguous()):
        raise ValueError("a, b and offsets must be contiguous")
    if a.data_ptr() % 16 or b.data_ptr() % 16:
        raise ValueError("a and b must be 16-byte aligned")
    if a.shape[1] % 8 or b.shape[1] % 8:
        raise ValueError("K and N must be multiples of 8 (16-byte TMA strides)")
    if b.device != a.device or offsets.device != a.device:
        raise ValueError("a, b and offsets must be on one device")
    shape = (a.shape[0], b.shape[1])
    if out is None:
        return a.new_empty(shape)
    if out.dtype != torch.bfloat16:
        raise TypeError(f"out must be bfloat16, got {out.dtype}")
    if out.shape != shape or out.device != a.device:
        raise ValueError(
            f"out must have shape {shape} on {a.device}, got {tuple(out.shape)} on {out.device}"
        )
    if not out.is_contiguous() or out.data_ptr() % 16:
        raise ValueError("out must be contiguous and 16-byte aligned")
    if _overlaps(out, a) or _overlaps(out, b) or _overlaps(out, offsets):
        raise ValueError("out must not overlap a, b or offsets")
    return out


_OPS: dict = {}


def grouped_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    offsets: torch.Tensor,
    out: torch.Tensor | None = None,
    schedule=None,
    max_ctas: int | None = None,
    cluster=None,
    **config,
) -> torch.Tensor:
    """Compute ``out[s_g:e_g] = a[s_g:e_g] @ b[g].T`` for each group.

    Inputs are row-major bf16 ``a: (total_M, K)``, ``b: (G, N, K)`` and int32 group
    end rows ``offsets: (G,)``: ``e_g = offsets[g]``, ``s_0 = 0``, ``s_g = e_{g-1}``.
    Rows past the last offset are unchanged.

    ``schedule`` defaults to ``StaticPersistent()``; ``max_ctas`` caps the persistent grid.
    ``config`` accepts the other ``GroupedGemm`` options: ``tile_n``, ``ab_stages``,
    ``acc_stages``, ``c_stages``, ``epi_tile_n`` and ``roles``.
    """
    out = check_grouped_gemm_args(a, b, offsets, out)
    if out.numel() == 0 or b.shape[0] == 0:  # nothing to compute, or no group covers a row
        return out
    if a.shape[1] == 0:  # no K tiles: the MMA would publish an unwritten accumulator
        covered = torch.arange(out.shape[0], device=out.device) < offsets[-1:]
        return out.masked_fill_(covered[:, None], 0)
    frozen = {k: _frozen_roles(v) if k == "roles" else v for k, v in config.items()}
    key = (schedule, cluster, *sorted(frozen.items()))
    op = _OPS.get(key)
    if op is None:
        op = _OPS[key] = GroupedGemm(schedule=schedule, cluster=cluster, **config)
    fn = compile_grouped_gemm.by_static_key(key, lambda: compile_grouped_gemm(op))
    workspace, sms = op.work.launch_args(a.device)
    fn(a, b, out, offsets, workspace, max_ctas or sms)
    return out


def grouped_gemm_reference(a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor):
    """Per-group fp32 matmuls; rows past the last offset are zero."""
    out = torch.zeros(a.shape[0], b.shape[1], device=a.device, dtype=torch.float32)
    start = 0
    for g, end in enumerate(offsets.tolist()):
        out[start:end] = a[start:end].float() @ b[g].float().T
        start = end
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    g, n, k = 4, 512, 256
    sizes = [300, 0, 128, 77]
    offsets = torch.tensor(sizes, device="cuda", dtype=torch.int32).cumsum(0).to(torch.int32)
    a = torch.randn(sum(sizes), k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(g, n, k, device="cuda", dtype=torch.bfloat16)
    out = grouped_gemm(a, b, offsets)
    err = (out.float() - grouped_gemm_reference(a, b, offsets)).abs().max().item()
    print(f"G={g} sizes={sizes} N={n} K={k}: max abs err {err:.4g}")
    print(GroupedGemm().describe())
