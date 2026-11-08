"""BF16 ``D = A @ B^T`` (``F.linear`` layout) on SM100, written with ``warpdrive``.

One CTA per 128 x ``tile_n`` output tile (a ``WorkStream`` with the ``OneShot`` schedule), K
stage 64, tcgen05 MMA 128 x ``tile_n`` x 16, three A/B SMEM stages; warps 0-3 epilogue, 4 TMA
load, 5 MMA. TMA zero-fills out-of-bounds boxes, so ragged M, N, K need only the epilogue store
predicate. ``tile_n`` (default 256), the stage counts and the roles' placement and register
budgets are constructor arguments.
"""

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym._backends.cute.cache import jit_cache
from attn_gym._backends.cute.utils import compile_tvm_ffi, make_fake_compact_tensor
from attn_gym.warpdrive import (
    Consumer,
    Local,
    OneShot,
    Producer,
    ProtocolError,
    Ring,
    RoleConfig,
    SmemPlan,
    StageView,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    role,
)
from attn_gym.warpdrive.examples.tma_tiles import tma_partition_tiles

TILE_M, TILE_K = 128, 64  # one tcgen05.ld 32x32b lane per row; 128 B K-major rows (SW128)
EPI_COLS = 32  # fp32 columns per tcgen05.ld 32x32b.x32 epilogue subtile


@dataclass(frozen=True)
class GemmParams:
    """Kernel-prologue values the roles read; built once in ``Gemm.kernel``."""

    tiled_mma: cute.TiledMma
    tma_a: cute.CopyAtom
    tma_b: cute.CopyAtom
    tAsA: StageView  # TMA partitions of the A and B stages: (TMA, stage)
    tAgA: cute.Tensor  # whole-problem A tiles: (TMA, m_tiles, k_tiles)
    tBsB: StageView
    tBgB: cute.Tensor  # whole-problem B tiles: (TMA, n_tiles, k_tiles)
    tCrA: StageView  # MMA fragments over the SMEM stages: (MMA, MMA_M, MMA_K, stage)
    tCrB: StageView
    num_k_tiles: cutlass.Int32
    acc_fake: cute.Tensor  # one accumulator stage, for building the TMEM load
    acc_layout: cute.Layout  # all accumulator stages
    epi_tiler: cute.Tile
    tCgC: cute.Tensor  # whole-problem output tiles: (MMA, MMA_M, MMA_N, m_tiles, n_tiles)
    tCcC: cute.Tensor  # their coordinates
    mC: cute.Tensor


class Gemm(WarpSpecialized):
    def __init__(
        self,
        ab_stages: int = 3,
        acc_stages: int = 1,
        trace: bool = False,
        tile_n: int = 256,
        roles: dict[str, RoleConfig] | None = None,
    ):
        """``tile_n``: output tile width, a multiple of ``EPI_COLS`` up to the MMA's 256.
        ``roles``: placement and register budgets by role name over the ``@role`` defaults."""
        if tile_n % EPI_COLS or not 0 < tile_n <= 256:
            raise ValueError(f"tile_n must be a multiple of {EPI_COLS} in (0, 256], got {tile_n}")
        self.dtype = cutlass.BFloat16
        self.tile = (TILE_M, tile_n, TILE_K)
        self.epi_subtiles = tile_n // EPI_COLS
        self.acc_stages = acc_stages
        self.roles = roles
        self.smem_plan = SmemPlan()
        a = self.smem_plan.buffer("A", self.dtype, align=1024)
        b = self.smem_plan.buffer("B", self.dtype, align=1024)
        stage_bytes = (TILE_M + tile_n) * TILE_K * 2
        self.ab = Ring(ab_stages, Tma(stage_bytes), Umma, buffer={"A": a, "B": b})
        self.acc = Ring(acc_stages, Umma, Threads)
        self.tmem_plan = TmemPlan().region("acc", tile_n, acc_stages)
        self.work = WorkStream(schedule=OneShot())  # one tile per CTA, m fastest
        self.trace_protocol = trace

    def get_name(self) -> str:
        n = "" if self.tile[1] == 256 else f"_n{self.tile[1]}"
        return f"warpdrive_gemm_ab{self.ab.n}_acc{self.acc_stages}{n}{self.trace_key()}"

    # ------------------------------------------------------------------ roles

    @role(warps=4, regs=40)
    @cute.jit
    def load(self, params: GemmParams, *, ab: Producer, work: WorkCursor):
        ab = ab.views(A=params.tAsA, B=params.tBsB)
        while work.valid:
            m, n, _ = work.item
            for k in cutlass.range(params.num_k_tiles, unroll=1):
                with ab.produce() as s:
                    cute.copy(
                        params.tma_a, params.tAgA[None, m, k], s.view("A"), tma_bar_ptr=s.barrier
                    )
                    cute.copy(
                        params.tma_b, params.tBgB[None, n, k], s.view("B"), tma_bar_ptr=s.barrier
                    )
            work.advance()
        return ab, work

    @role(warps=5, regs=40)
    @cute.jit
    def mma(
        self, params: GemmParams, *, ab: Consumer, acc: Producer, work: WorkCursor, tmem: Tmem
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
        self, params: GemmParams, *, acc: Consumer, work: WorkCursor, tmem: Tmem, me: Local
    ):
        if cutlass.const_expr(me.warps != 4):
            raise ProtocolError(
                f"epilogue: the TMEM load covers 128 rows with 4 warps, got {me.warps}"
            )
        tidx = me.tmem_row
        t2r = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32),
            cute.zipped_divide(params.acc_fake, params.epi_tiler)[None, 0],
        ).get_slice(tidx)
        tAcc = cute.make_tensor(tmem.ptr("acc"), params.acc_layout)
        rows, cols = params.mC.shape
        while work.valid:
            m, n, _ = work.item
            tCgC = params.tCgC[None, None, None, m, n]
            tDgC = t2r.partition_D(cute.zipped_divide(tCgC, params.epi_tiler))
            tDcC = t2r.partition_D(
                cute.zipped_divide(params.tCcC[None, None, None, m, n], params.epi_tiler)
            )
            rAcc = cute.make_rmem_tensor(tDgC[None, None, 0].shape, cutlass.Float32)
            rC = cute.make_rmem_tensor(tDgC[None, None, 0].shape, self.dtype)
            full_tile = ((m + 1) * self.tile[0] <= rows) & ((n + 1) * self.tile[1] <= cols)
            with acc.consume() as a:
                tDtC = t2r.partition_S(
                    cute.zipped_divide(tAcc[None, None, None, a.index], params.epi_tiler)
                )
                for i in cutlass.range_constexpr(self.epi_subtiles):
                    cute.copy(t2r, tDtC[None, None, i], rAcc)
                    rC.store(rAcc.load().to(self.dtype))
                    if full_tile:
                        cute.autovec_copy(rC, tDgC[None, None, i])
                    else:
                        g, crd = tDgC[None, None, i], tDcC[None, None, i]
                        for j in cutlass.range_constexpr(cute.size(rC)):
                            if cute.elem_less(crd[j], params.mC.shape):
                                g[j] = rC[j]
            work.advance()
        return acc, work

    # ----------------------------------------------------------------- kernel

    @cute.kernel
    def kernel(self, tiled_mma, tma_a, mA, tma_b, mB, mC, a_layout, b_layout, bound):
        self.begin_roles(work=bound, layouts={"A": a_layout, "B": b_layout})
        sA, sB = self.buffer("A"), self.buffer("B")
        thr_mma = tiled_mma.get_slice(0)
        # Whole-problem tile views: (BM, BK, m_tiles, k_tiles) and (BN, BK, n_tiles, k_tiles).
        gA = cute.local_tile(mA, self.tile, (None, None, None), proj=(1, None, 1))
        gB = cute.local_tile(mB, self.tile, (None, None, None), proj=(None, 1, 1))
        tAsA, tAgA = tma_partition_tiles(tma_a, sA, thr_mma.partition_A(gA))
        tBsB, tBgB = tma_partition_tiles(tma_b, sB, thr_mma.partition_B(gB))
        acc_shape = tiled_mma.partition_shape_C(self.tile[:2])
        acc_fake = tiled_mma.make_fragment_C(acc_shape)
        acc_staged = tiled_mma.make_fragment_C(cute.append(acc_shape, self.acc_stages))
        cC = cute.make_identity_tensor(mC.shape)
        params = GemmParams(
            tiled_mma=tiled_mma,
            tma_a=tma_a,
            tma_b=tma_b,
            tAsA=StageView(tAsA, axis=1),
            tAgA=tAgA,
            tBsB=StageView(tBsB, axis=1),
            tBgB=tBgB,
            tCrA=StageView(tiled_mma.make_fragment_A(sA), axis=3),
            tCrB=StageView(tiled_mma.make_fragment_B(sB), axis=3),
            num_k_tiles=cute.size(gA, mode=[3]),
            acc_fake=acc_fake,
            acc_layout=acc_staged.layout,
            epi_tiler=(
                (
                    cute.size(acc_fake, mode=[0, 0]),
                    cute.size(acc_fake, mode=[0, 1]) // self.epi_subtiles,
                ),
            ),
            tCgC=thr_mma.partition_C(
                cute.local_tile(mC, self.tile, (None, None, None), proj=(1, 1, None))
            ),
            tCcC=thr_mma.partition_C(
                cute.local_tile(cC, self.tile, (None, None, None), proj=(1, 1, None))
            ),
            mC=mC,
        )
        self.run_roles(params)

    # ------------------------------------------------------------------- host

    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: cuda.CUstream):
        op = tcgen05.MmaF16BF16Op(
            self.dtype,
            cutlass.Float32,
            (TILE_M, self.tile[1], 16),
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
        )
        tiled_mma = cute.make_tiled_mma(op)
        a_layout = sm100_utils.make_smem_layout_a(tiled_mma, self.tile, self.dtype, self.ab.n)
        b_layout = sm100_utils.make_smem_layout_b(tiled_mma, self.tile, self.dtype, self.ab.n)
        tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_a, mA = cute.nvgpu.make_tiled_tma_atom_A(
            tma_op, a, cute.select(a_layout, mode=[0, 1, 2]), self.tile, tiled_mma
        )
        tma_b, mB = cute.nvgpu.make_tiled_tma_atom_B(
            tma_op, b, cute.select(b_layout, mode=[0, 1, 2]), self.tile, tiled_mma
        )
        m_tiles, n_tiles = cute.ceil_div(c.layout.shape, self.tile[:2])
        bound = self.work.bind((m_tiles, n_tiles, 1))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(tiled_mma, tma_a, mA, tma_b, mB, c, a_layout, b_layout, bound).launch(
            **self.launch_config(bound), stream=stream
        )


@jit_cache(uncached=lambda op: op.trace_protocol)
def compile_gemm(op: Gemm):
    """Compile once with dynamic M, N, K (K and N multiples of 8 for 16-byte TMA strides).
    Cached in memory and on disk under ``op.__cute_cache_key__()``."""
    m = cute.sym_int()
    n, k = cute.sym_int(divisibility=8), cute.sym_int(divisibility=8)
    bf16 = cutlass.BFloat16
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (m, k)),
        make_fake_compact_tensor(bf16, (n, k)),
        make_fake_compact_tensor(bf16, (m, n)),
        name=op.get_name(),
    )


def _byte_span(t: torch.Tensor) -> tuple[int, int]:
    """``[start, stop)`` of the bytes ``t`` addresses (empty for zero elements)."""
    if t.numel() == 0:
        return 0, 0
    extent = 1 + sum((size - 1) * stride for size, stride in zip(t.shape, t.stride()))
    return t.data_ptr(), t.data_ptr() + extent * t.element_size()


def _overlaps(x: torch.Tensor, y: torch.Tensor) -> bool:
    (x0, x1), (y0, y1) = _byte_span(x), _byte_span(y)
    return x0 < y1 and y0 < x1


def check_gemm_args(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None) -> torch.Tensor:
    """Check the operands of ``a @ b.T`` and return ``out`` (allocated when ``None``).

    ``out`` must not share memory with ``a`` or ``b``: one tile's TMA store can overwrite
    operand rows that another tile (another CTA, or a later persistent iteration) still loads.
    """
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise TypeError("gemm expects bfloat16 operands")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"incompatible shapes {tuple(a.shape)}, {tuple(b.shape)}")
    if not (a.is_contiguous() and b.is_contiguous()) or a.data_ptr() % 16 or b.data_ptr() % 16:
        raise ValueError("a and b must be contiguous and 16-byte aligned")
    if a.shape[1] % 8 or b.shape[0] % 8:
        raise ValueError("K and N must be multiples of 8 (16-byte TMA strides)")
    if b.device != a.device:
        raise ValueError(f"a and b must be on one device, got {a.device} and {b.device}")
    shape = (a.shape[0], b.shape[0])
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
    if _overlaps(out, a) or _overlaps(out, b):
        raise ValueError("out must not overlap a or b")
    return out


def gemm(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``a @ b.T`` for row-major bf16 ``a: (M, K)``, ``b: (N, K)``."""
    out = check_gemm_args(a, b, out)
    if out.numel() == 0:
        return out
    if a.shape[1] == 0:  # no K tiles: the MMA would publish an unwritten accumulator
        return out.zero_()
    compile_gemm.by_static_key("default", lambda: compile_gemm(Gemm()))(a, b, out)
    return out


if __name__ == "__main__":
    for m, n, k in [(512, 512, 512), (300, 520, 136), (4097, 2056, 4104)]:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(a.float(), b.float())
        err = (gemm(a, b).float() - ref).abs().max().item()
        print(f"M={m} N={n} K={k}: max abs err {err:.4g}")
    print(Gemm().describe())
