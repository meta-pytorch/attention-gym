# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CuTeDSL forward affine-summary kernel for delta-rule context parallelism.

One local shard of a delta-rule sequence acts on the recurrent state as an affine
map ``H_out = H_in @ A + B``. This kernel computes that summary directly from the
shared per-chunk WY factors (``kg``, ``w``, ``u``, ``cumulative_gate`` as produced
by ``chunk_kda_fwd_intra``) instead of running the full recurrence twice with
zero/identity probe states (see ``examples/linear/delta_rule_context_parallel.py``).

Math (K-first augmented state ``X`` of shape ``[K, V + K]``, fp32):

    X = [0 | I]
    per 64-token chunk c:
        Tmp        = W[c] @ X                 # [BT, V + K]
        Tmp[:, :V] = U[c] - Tmp[:, :V]
        Tmp[:, V:] =      - Tmp[:, V:]
        X          = diag(exp2(gk_last[c])) @ X + Kg[c]^T @ Tmp

The output is the V-first packed transpose ``X^T`` of shape ``[H, V + K, K]``:
rows ``[0:V]`` hold the state bias and rows ``[V:V+K]`` the state transition.
These are distinct from KDA's token-token query/key matrix ``A``.

Terms (see also NOTE [Terminology] in ``attn_gym.linear.context_parallel``):

    chunk         64 tokens (``BT``). The factor kernels lay chunks from each subsequence's
                  first token, so every subsequence has its own chunk grid.
    range         one row of ``bounds``: ``[start, stop)`` of whole chunks inside one
                  subsequence of the span (the staged contract only promises whole
                  subsequences, NOTE [Summary ranges are subsequences] in
                  ``attn_gym.linear.kda.stages``), which gets its own
                  ``[bias; transition]`` summary. ``R`` = number of rows. A range never spans
                  two documents; a row with ``start == stop`` scans nothing and yields the
                  identity.
    column tile   a ``BN``-wide slice of the ``V + K = 256`` augmented-state columns. One
                  CTA owns the ``[K, BN]`` state tile ``X[:, cols]`` for one head and one
                  work item; tiles ``< V / BN`` hold bias columns, the rest transition
                  columns.

The context-parallel routing passes exactly one range per fragment a rank owns: that
fragment's *last* subsequence, or an empty range when it ends its document. A fragment is one
contiguous global token range, so only its last subsequence can continue on another rank and
only its first can be continued; interior subsequences are whole documents that start from
zero. ``R`` therefore equals the rank's fragment count, while each row is one subsequence's
range (NOTE [Terminology], "Summary work is bounded by fragments").

*work item* and *partial* are the ``work`` table contract between the planner and the
kernels, defined in ``attn_gym.linear._delta_rule.triton.work_items``.

Architecture (SM100 warpdrive roles):
  - One CTA per (column tile, head, work item); 6 warps. The recurrence is a serial latency
    chain, so when ``heads * tiles`` leaves SMs idle the wrapper cuts the ranges' chunks into
    as many items as fill the machine (``plan_work_items``) and ``compose_work_items`` folds
    the partials per range, both on the device.
  - CUDA warps (0-3) keep the fp32 ``[K, BN]`` state tile in registers across the item's
    chunks and produce the two SMEM MMA B operands per chunk.
  - Load warp (4) TMA-stages ``w``, ``kg^T``, ``u``, and the chunk-final gate row.
  - MMA warp (5) issues two SS-mode UMMA groups per chunk into TMEM:
      MMA1: W  @ X          -> wx  (BT=64, BN, K=128)
      MMA2: Kg^T @ [U; -wx] -> kt  (K=128, BN, K=64)
    ``U`` rides its own TMA ring straight into an MMA2 B-operand pass, so the
    ``Kg^T @ U`` product runs off the serial recurrence chain; the CUDA warps
    only split ``-wx`` for the remaining accumulate passes.

The fp32-valued MMA B operands (state ``X`` and ``wx``) are split into hi/lo
halves of the I/O dtype and accumulated in two UMMA passes. The A operands
(``w``, ``kg``) and ``u`` retain their original bf16/fp16 storage.

Limitations:
  - B=1, fixed K=V=128, BT=64; any ``T``. Token ranges come from a device ``bounds`` tensor;
    partial last chunks clamp the gate row and zero every over-read UMMA operand in SMEM.
  - Contiguous inputs; SM100 (tcgen05) only.
"""

from dataclasses import dataclass
from typing import NamedTuple

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import cute
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_compact_tensor
from cutlass.cute.typing import Float32, Int32
from torch._subclasses.fake_tensor import FakeTensor

from attn_gym._backends.cute import get_device_properties
from attn_gym._backends.cute.utils import requires_int64_abi
from attn_gym.linear._delta_rule.constants import is_sm100_kda_capability
from attn_gym.linear._delta_rule.triton.work_items import compose_work_items, work_table
from attn_gym.utils import cdiv
from attn_gym.warpdrive import (
    Consumer,
    Local,
    OneShot,
    Producer,
    Ring,
    SmemPlan,
    Threads,
    Tma,
    Tmem,
    TmemPlan,
    Umma,
    Warps,
    WarpSpecialized,
    WorkCursor,
    WorkStream,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, get_compile_target, jit_cache
from attn_gym.warpdrive.compat import LayoutEnum

BT = 64  # chunk size
KEY_DIM = 128
VAL_DIM = 128
SUMMARY_DIM = VAL_DIM + KEY_DIM  # V-first packed [bias; transition] rows
DATA_ALIGN_BYTES = 16

_IO_TYPE_NAMES = {torch.bfloat16: "bf16", torch.float16: "fp16"}
_CUTE_IO_TYPES = {"bf16": cutlass.BFloat16, "fp16": cutlass.Float16}


def operand_view_layout(shape, token_mode: int):
    """Map ``(rows, tokens)`` onto a staged UMMA operand's nested atom/rest modes.

    Regrouping a CuTe identity layout preserves the stage's native coordinate space, including
    its swizzle, without reconstructing compact strides.
    """
    identity = cute.make_identity_layout(shape)
    (atom_0, atom_1), rest_0, rest_1 = identity.shape
    (stride_a0, stride_a1), stride_r0, stride_r1 = identity.stride
    modes = ((atom_0, rest_0), (atom_1, rest_1))
    strides = ((stride_a0, stride_r0), (stride_a1, stride_r1))
    row_mode = 1 - token_mode
    return cute.make_layout(
        (modes[row_mode], modes[token_mode]), stride=(strides[row_mode], strides[token_mode])
    )


@cute.jit
def zero_tail_tokens(smem, stage, valid_rows, tid, io_type, token_mode: cutlass.Constexpr):
    """Zero token rows ``>= valid_rows`` of one staged UMMA operand before the MMA reads it.

    Selecting the over-read rows away in SMEM, rather than cancelling them with a ``0 *``
    factor, keeps NaN/Inf garbage from uninitialized capacity slack out of the result.
    ``token_mode`` names which atom/rest pair of the staged layout spans the tokens; one warp
    (``tid`` in ``[0, 32)``) performs the stores.
    """
    stage_view = smem[None, None, None, stage]
    view = cute.composition(stage_view, operand_view_layout(stage_view.shape, token_mode))
    rows = cute.size(view, mode=[0])
    warp_size = cute.arch.WARP_SIZE
    assert cute.size(view, mode=[1]) == BT, "token modes must tile BT"
    assert rows % warp_size == 0, "operand rows must be a multiple of the warp size"
    for token in cutlass.range(valid_rows, BT, unroll=1):
        for row_block in cutlass.range_constexpr(rows // warp_size):
            view[tid + row_block * warp_size, token] = io_type(0.0)
    cute.arch.fence_view_async_shared()
    cute.arch.sync_warp()


@cute.jit
def range_tail(length):
    """Return the last chunk index and its valid-row count; the count is zero for a full chunk."""
    last_chunk = (length + BT - 1) // BT - 1
    valid_rows = length - last_chunk * BT
    if valid_rows == BT:
        valid_rows = Int32(0)
    return last_chunk, valid_rows


@cute.jit
def tma_partition_a(atom, desc, smem, tile, mma, head):
    """Partition one head of a dense TMA tensor and its stages as an SS-mode MMA A operand."""
    gmem = cute.local_tile(desc, cute.slice_(tile, (None, 0, None)), (None, None, (head, 0)))
    return cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0, 3),
        cute.group_modes(mma.get_slice(0).partition_A(gmem), 0, 3),
    )


@cute.jit
def tma_partition_b(atom, desc, smem, tile, mma, head):
    """Partition one head of a dense TMA tensor and its stages as an SS-mode MMA B operand."""
    gmem = cute.local_tile(desc, cute.slice_(tile, (0, None, None)), (None, None, (head, 0)))
    return cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0, 3),
        cute.group_modes(mma.get_slice(0).partition_B(gmem), 0, 3),
    )


@cute.jit
def tma_partition_gate(atom, desc, smem, head):
    """Partition one head's ``[K, T]`` gate rows and the staged ``[K, 1]`` gate tiles."""
    gmem = cute.flat_divide(desc[None, None, (head, 0)], (KEY_DIM, 1))
    return cpasync.tma_partition(
        atom,
        0,
        cute.make_layout(1),
        cute.group_modes(smem, 0, 2),
        cute.group_modes(gmem, 0, 2),
    )


def tmem_tile_reader(accumulator, tid, columns: int):
    """Build one thread's load of a staged FP32 TMEM accumulator ``columns`` wide.

    Returns the copy, the thread's slice of it, and its partition of every stage.
    """
    atom = cute.make_copy_atom(
        tcgen05.Ld16x256bOp(tcgen05.Repetition(columns // 8), tcgen05.Pack.NONE), Float32
    )
    flat = accumulator[((None, None), 0, 0, None)]
    copy = tcgen05.make_tmem_copy(atom, flat[(None, None, 0)])
    thread_copy = copy.get_slice(tid)
    return copy, thread_copy, thread_copy.partition_S(flat)


def smem_tile_writer(reader, tid, io_type):
    """Build the store of a tile read by ``reader`` into a column-major ``io_type`` stage.

    Returns the copy and the thread's slice of it; the register partition matches the read.
    """
    atom = sm100_utils.get_smem_store_op(LayoutEnum.COL_MAJOR, io_type, Float32, reader)
    copy = cute.make_tiled_copy_D(atom, reader)
    return copy, copy.get_slice(tid)


def store_rounded(value, writer, stage, io_type):
    """Store an FP32 register vector, rounded to ``io_type``, in one SMEM operand stage.

    Returns the rounded registers, used to form the residual when storing the hi half.
    """
    copy, thread_copy = writer
    rounded = cute.make_rmem_tensor(value.shape, io_type)
    rounded.store(value.to(io_type))
    cute.copy(copy, copy.retile(rounded), thread_copy.partition_D(stage))
    return rounded


def _aligned(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize the uncommon contiguous view that misses the TMA alignment."""
    return tensor if tensor.data_ptr() % DATA_ALIGN_BYTES == 0 else tensor.clone()


@cute.jit
def _sequence_feature_head_batch_view(tensor):
    """Re-rank ``[B,T,H,D]`` as ``[T,D,(H,B)]`` for a TMA operand."""
    layout = cute.group_modes(cute.select(tensor.layout, mode=[1, 3, 2, 0]), 2, 4)
    return cute.make_tensor(tensor.iterator, layout)


@cute.jit
def _feature_sequence_head_batch_view(tensor):
    """Re-rank ``[B,T,H,D]`` as ``[D,T,(H,B)]`` for a TMA operand."""
    layout = cute.group_modes(cute.select(tensor.layout, mode=[3, 1, 2, 0]), 2, 4)
    return cute.make_tensor(tensor.iterator, layout)


@cute.jit
def decode_work_row(work_table: cute.Tensor, work: Int32, whole_ranges: cutlass.Constexpr):
    """Return ``(start, length, chunk_begin, chunk_end)`` of work item ``work``.

    Rows are ``(start, chunk_begin, chunk_end, length)`` from ``plan_work_items``, or ``(start,
    stop)`` of ``bounds`` itself under ``whole_ranges`` (one item per range, see
    ``work_items.work_table``). An all-zero row scans nothing.
    """
    start = work_table[work, 0]
    if cutlass.const_expr(whole_ranges):
        length = work_table[work, 1] - start
        return start, length, Int32(0), (length + BT - 1) // BT
    return start, work_table[work, 3], work_table[work, 1], work_table[work, 2]


class TmaOp(NamedTuple):
    """One TMA copy atom and its tensor-map descriptor."""

    atom: cute.CopyAtom
    desc: cute.Tensor


class TmaOps(NamedTuple):
    """TMA operations owned by the affine-summary kernel."""

    w: TmaOp
    kg: TmaOp
    u: TmaOp
    gk: TmaOp


class Mmas(NamedTuple):
    """Tensor-core operations owned by the MMA warp."""

    wx: cute.TiledMma
    kt: cute.TiledMma
    ktu: cute.TiledMma


@dataclass(frozen=True)
class ForwardSummaryParams:
    """Shared layouts and runtime inputs; channel state stays local to each role."""

    work_table: cute.Tensor
    out: cute.Tensor
    mmas: Mmas
    tma: TmaOps
    wx_layout: cute.Layout
    kt_layout: cute.Layout
    sW: cute.Tensor
    sKg: cute.Tensor
    sU: cute.Tensor
    sXb: cute.Tensor
    sXb_store: cute.Tensor
    sTmpb: cute.Tensor
    sTmpb_store: cute.Tensor
    gk_3d: cute.Tensor


class _AffineSummaryFwdOp(WarpSpecialized):
    """Warp-specialized SM100 chunk recurrence over one augmented-state column tile.

    The fp32 ``[K, BN]`` state slice persists in CUDA-warp registers; each chunk
    snapshots it (and later ``Tmp``) into SMEM as hi/lo I/O-dtype halves feeding
    two-pass SS-mode UMMAs.
    """

    def __init__(
        self,
        dtype_name: str,
        heads: int,
        state_bn: int,
        use_int64_offsets: bool,
        whole_ranges: bool,
        fastmath: bool = True,
    ):
        assert state_bn in (32, 64), f"state_bn must be 32 or 64, got {state_bn}"
        assert SUMMARY_DIM % state_bn == 0
        self.dtype_name = dtype_name
        self.io_type = _CUTE_IO_TYPES[dtype_name]
        self.heads = heads
        self.BN = state_bn
        self.use_int64_offsets = use_int64_offsets
        self.fastmath = fastmath
        # With ``whole_ranges`` the work table is ``bounds`` itself, ``[R, 2]`` token ranges,
        # and every row is one item: the ``budget == 1`` case needs no planner and no fold.
        self.whole_ranges = whole_ranges

        # MMA tile shapes (M, N, K).
        # MMA1: w @ X -> wx — w is (BT, K) K-major, X is (K, BN) snapshot in SMEM.
        self.wx_tile = (BT, self.BN, KEY_DIM)
        # MMA2: kg^T @ Tmp -> kt — kg^T is (K, BT) MN-major, Tmp is (BT, BN) in SMEM.
        self.kt_tile = (KEY_DIM, self.BN, BT)

        # TMA pipeline depths. The recurrence is a serial latency chain, so
        # shallow rings are enough for the loads to run ahead of it.
        self.w_depth = 2
        self.kg_depth = 2
        self.u_depth = 2
        self.gk_depth = 2

        self.smem_plan = SmemPlan()
        w = self.smem_plan.buffer("W", self.io_type, align=1024)
        kg = self.smem_plan.buffer("Kg", self.io_type, align=1024)
        xb = self.smem_plan.buffer("Xb", self.io_type, align=1024)
        tmpb = self.smem_plan.buffer("Tmpb", self.io_type, align=1024)
        u = self.smem_plan.buffer("U", self.io_type, align=1024)
        gk = self.smem_plan.buffer("Gk", Float32, align=128)
        xb_store = self.smem_plan.buffer("XbStore", self.io_type, align=1024, overlaps=xb)
        tmpb_store = self.smem_plan.buffer("TmpbStore", self.io_type, align=1024, overlaps=tmpb)
        io_bytes = self.io_type.width // 8
        self.w = Ring(self.w_depth, Tma(BT * KEY_DIM * io_bytes), Umma, buffer=w)
        self.kg = Ring(self.kg_depth, Tma(KEY_DIM * BT * io_bytes), Umma, buffer=kg)
        self.u = Ring(self.u_depth, Tma(self.BN * BT * io_bytes), Umma, buffer=u)
        self.gk = Ring(self.gk_depth, Tma(KEY_DIM * 4), Warps, buffer=gk)
        self.xb = Ring(2, Threads, Umma, buffer={"B": xb, "Store": xb_store})
        self.tmpb = Ring(2, Threads, Umma, buffer={"B": tmpb, "Store": tmpb_store})
        self.wx = Ring(1, Umma, Threads, release_fence="tmem_load")
        self.kt = Ring(1, Umma, Threads, release_fence="tmem_load")
        self.tmem_plan = TmemPlan().region("wx", self.BN).region("kt", self.BN)
        self.work = WorkStream(schedule=OneShot())

    def get_name(self) -> str:
        """Return a stable profiler and artifact name for this specialization."""
        return (
            f"delta_affine_summary_fwd_h{self.heads}_bn{self.BN}_{self.dtype_name}"
            f"_i64{int(self.use_int64_offsets)}_whole{int(self.whole_ranges)}"
            f"_fm{int(self.fastmath)}"
        )

    # ------------------------------------------------------------------
    # Host-side setup (__call__): GMEM views → MMA → TMA → SMEM → launch
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mKg: cute.Tensor,
        mW: cute.Tensor,
        mU: cute.Tensor,
        mG: cute.Tensor,
        mWork: cute.Tensor,
        mOut: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Launch one CTA per (augmented-state column tile, head, work item).

        ``mWork[w] = (start, chunk_begin, chunk_end, length)`` names a run of chunks of one
        range: token ``start`` of the ``[1, T, H, D]`` inputs begins the range, which is
        ``length`` tokens long, and this item scans its chunks ``[chunk_begin, chunk_end)`` from
        ``[0 | I]`` into ``mOut[w]``. An all-zero row scans nothing and stores the identity.
        """
        # Re-rank the dense [1, T, H, D] inputs into the logical TMA views.
        g_w = _sequence_feature_head_batch_view(mW)
        g_kgt = _feature_sequence_head_batch_view(mKg)
        g_u_vt = _feature_sequence_head_batch_view(mU)
        g_gk_k = _feature_sequence_head_batch_view(mG)

        # --- MMA configurations (SS-mode: both operands from SMEM) ---
        # MMA1: w (A, K-major) × X snapshot (B, K-major) → wx.
        mma_wx = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            self.io_type,
            cute.nvgpu.OperandMajorMode.K,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            self.wx_tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        # MMA2: kg^T (A, MN-major) × Tmp (B, K-major) → kt.
        mma_kt = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            self.io_type,
            cute.nvgpu.OperandMajorMode.MN,
            cute.nvgpu.OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            self.kt_tile[:2],
            tcgen05.OperandSource.SMEM,
        )
        # MMA2a: kg^T (A, MN-major) × U (B, MN-major, V-contiguous TMA) → kt.
        # Same accumulator tile as MMA2; only the B major mode differs.
        mma_ktu = sm100_utils.make_trivial_tiled_mma(
            self.io_type,
            self.io_type,
            cute.nvgpu.OperandMajorMode.MN,
            cute.nvgpu.OperandMajorMode.MN,
            Float32,
            tcgen05.CtaGroup.ONE,
            self.kt_tile[:2],
            tcgen05.OperandSource.SMEM,
        )

        # --- SMEM staged layouts ---
        # A operands ride their TMA rings; B operands hold hi/lo halves of the
        # fp32-valued state and Tmp tiles, one pipeline stage per half.
        s_w_staged = sm100_utils.make_smem_layout_a(
            mma_wx,
            self.wx_tile,
            self.io_type,
            self.w_depth,
        )
        s_xb_staged = sm100_utils.make_smem_layout_b(mma_wx, self.wx_tile, self.io_type, 2)
        s_xb_store_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            LayoutEnum.COL_MAJOR,
            (KEY_DIM, self.BN),
            2,
        )
        s_kg_staged = sm100_utils.make_smem_layout_a(
            mma_kt,
            self.kt_tile,
            self.io_type,
            self.kg_depth,
        )
        # u is TMA-loaded directly into its own MMA2a B-operand ring.
        s_u_staged = sm100_utils.make_smem_layout_b(
            mma_ktu,
            self.kt_tile,
            self.io_type,
            self.u_depth,
        )
        s_tmpb_staged = sm100_utils.make_smem_layout_b(mma_kt, self.kt_tile, self.io_type, 2)
        s_tmpb_store_staged = sm100_utils.make_smem_layout_epi(
            self.io_type,
            LayoutEnum.COL_MAJOR,
            (BT, self.BN),
            2,
        )

        clust_lay = cute.tiled_divide(
            cute.make_layout((1, 1, 1)),
            (mma_wx.thr_id.shape,),
        )

        # --- TMA descriptors ---
        tma_ld = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        s_w_one = cute.select(s_w_staged, mode=[0, 1, 2])
        atom_w, desc_w = cute.nvgpu.make_tiled_tma_atom_A(
            tma_ld,
            g_w,
            s_w_one,
            self.wx_tile,
            mma_wx,
            clust_lay.shape,
        )
        s_kg_one = cute.select(s_kg_staged, mode=[0, 1, 2])
        atom_kg, desc_kg = cute.nvgpu.make_tiled_tma_atom_A(
            tma_ld,
            g_kgt,
            s_kg_one,
            self.kt_tile,
            mma_kt,
            clust_lay.shape,
        )
        s_u_one = cute.select(s_u_staged, mode=[0, 1, 2])
        atom_u, desc_u = cute.nvgpu.make_tiled_tma_atom_B(
            tma_ld,
            g_u_vt,
            s_u_one,
            self.kt_tile,
            mma_ktu,
            clust_lay.shape,
        )
        # gk: K×1 fp32 tile (last timestep per chunk).
        s_gk_2d = cute.make_layout((KEY_DIM, 1))
        atom_gk, desc_gk = cpasync.make_tiled_tma_atom(
            tma_ld,
            g_gk_k,
            s_gk_2d,
            (KEY_DIM, 1),
        )

        bound = self.work.bind((SUMMARY_DIM // self.BN, self.heads, Int32(mOut.shape[0])))

        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            Mmas(mma_wx, mma_kt, mma_ktu),
            TmaOps(
                TmaOp(atom_w, desc_w),
                TmaOp(atom_kg, desc_kg),
                TmaOp(atom_u, desc_u),
                TmaOp(atom_gk, desc_gk),
            ),
            mWork,
            mOut,
            {
                "W": s_w_staged,
                "Kg": s_kg_staged,
                "U": s_u_staged,
                "Xb": s_xb_staged,
                "XbStore": s_xb_store_staged,
                "Tmpb": s_tmpb_staged,
                "TmpbStore": s_tmpb_store_staged,
                "Gk": cute.make_layout((KEY_DIM, 1, self.gk_depth), stride=(1, KEY_DIM, KEY_DIM)),
            },
            bound,
        ).launch(
            **self.launch_config(bound, min_blocks_per_mp=1),
            stream=stream,
        )

    # ------------------------------------------------------------------
    # Warp roles
    # ------------------------------------------------------------------

    @role(warps=4)
    @cute.jit
    def run_load(
        self,
        params: ForwardSummaryParams,
        *,
        w: Producer,
        kg: Producer,
        u: Producer,
        gk: Producer,
        work: WorkCursor,
    ):
        """Own the per-chunk TMA G2S loads for w, kg^T, u, and the gate row.

        The range starts at token ``start`` of the stream, so every descriptor is offset along
        its token mode and chunk ``ct`` reads tokens ``start + ct * BT`` onward. A partial last
        chunk clamps the gate row here; the MMA warp zeroes each over-read operand stage.
        """
        sW = params.sW
        sKg = params.sKg
        sU = params.sU
        gk_3d = params.gk_3d
        mma_wx, mma_kt, mma_ktu = params.mmas
        tma = params.tma
        cpasync.prefetch_descriptor(tma.w.atom)
        cpasync.prefetch_descriptor(tma.kg.atom)
        cpasync.prefetch_descriptor(tma.u.atom)
        cpasync.prefetch_descriptor(tma.gk.atom)
        while work.valid:
            col_tile, head, row = work.item
            start, length, chunk_begin, chunk_end = decode_work_row(
                params.work_table, row, self.whole_ranges
            )
            col_base = col_tile * self.BN
            is_value = col_base < VAL_DIM
            desc_w = cute.domain_offset((start, 0, (0, 0)), tma.w.desc)
            desc_kg = cute.domain_offset((0, start, (0, 0)), tma.kg.desc)
            desc_u = cute.domain_offset((0, start, (0, 0)), tma.u.desc)
            desc_gk = cute.domain_offset((0, start, (0, 0)), tma.gk.desc)
            tWs, tWg = tma_partition_a(tma.w.atom, desc_w, sW, self.wx_tile, mma_wx, head)
            tKgs, tKgg = tma_partition_a(tma.kg.atom, desc_kg, sKg, self.kt_tile, mma_kt, head)
            sUp, gUp = tma_partition_b(tma.u.atom, desc_u, sU, self.kt_tile, mma_ktu, head)
            sGKp, gGKp = tma_partition_gate(tma.gk.atom, desc_gk, gk_3d, head)

            for ct in cutlass.range(chunk_begin, chunk_end, unroll=0):
                wh = w.acquire()
                cute.copy(
                    atom=tma.w.atom,
                    src=tWg[None, ct, 0],
                    dst=tWs[None, wh.index],
                    tma_bar_ptr=wh.barrier,
                )
                wh.commit()
                kgh = kg.acquire()
                cute.copy(
                    atom=tma.kg.atom,
                    src=tKgg[None, 0, ct],
                    dst=tKgs[None, kgh.index],
                    tma_bar_ptr=kgh.barrier,
                )
                kgh.commit()
                if is_value:
                    uh = u.acquire()
                    cute.copy(
                        atom=tma.u.atom,
                        src=gUp[None, col_tile, ct],
                        dst=sUp[None, uh.index],
                        tma_bar_ptr=uh.barrier,
                    )
                    uh.commit()
                # The chunk-final gate row: the range's last token when the chunk is partial.
                gk_t = cutlass.min(ct * BT + BT - 1, length - 1)
                gkh = gk.acquire()
                cute.copy(
                    atom=tma.gk.atom,
                    src=gGKp[(None, 0, gk_t)],
                    dst=sGKp[None, gkh.index],
                    tma_bar_ptr=gkh.barrier,
                )
                gkh.commit()
            work.advance()
        return w, kg, u, gk, work

    @role(warps=5)
    @cute.jit
    def run_mma(
        self,
        params: ForwardSummaryParams,
        *,
        w: Consumer,
        kg: Consumer,
        u: Consumer,
        xb: Consumer,
        tmpb: Consumer,
        wx: Producer,
        kt: Producer,
        work: WorkCursor,
        me: Local,
        tmem: Tmem,
    ):
        """Consume operand stages and produce the two TMEM results per chunk.

        The kt accumulator gathers ``Kg^T @ U`` (value tiles, straight from the
        TMA ring, off the recurrence critical path) plus two hi/lo accumulate
        passes over ``Kg^T @ (-wx)``; MMA1 likewise runs two passes over the
        hi/lo state snapshot, recovering ~fp32 operand precision.

        Chunk ``tail_chunk`` holds only ``tail_valid_rows`` range tokens. After each full TMA
        wait, the MMA warp alone zeroes its ``w``, ``kg^T``, and ``u`` stages past that boundary;
        a zero mask would retain NaN/Inf from uninitialized capacity slack. ``zero_tail_tokens``
        fences every writer and synchronizes the warp before the elected UMMA issuer reads it.
        The handle's UMMA release completes those reads before the next TMA fill; the same proxy
        fence also orders the generic tail writes before that fill.
        """
        sW = params.sW
        sKg = params.sKg
        sU = params.sU
        mma_wx, mma_kt, mma_ktu = params.mmas
        t_wx_acc = cute.make_tensor(tmem.ptr("wx"), params.wx_layout)
        t_kt_acc = cute.make_tensor(tmem.ptr("kt"), params.kt_layout)
        t_w_a = mma_wx.make_fragment_A(sW)
        t_xb_b = mma_wx.make_fragment_B(params.sXb)
        t_kg_a = mma_kt.make_fragment_A(sKg)
        t_kg_au = mma_ktu.make_fragment_A(sKg)
        t_u_b = mma_ktu.make_fragment_B(sU)
        t_tmpb_b = mma_kt.make_fragment_B(params.sTmpb)
        mma_tid = me.tid
        while work.valid:
            col_tile, _head, row = work.item
            _start, length, chunk_begin, chunk_end = decode_work_row(
                params.work_table, row, self.whole_ranges
            )
            col_base = col_tile * self.BN
            is_value = col_base < VAL_DIM
            # The range's last chunk may be partial; only the item that scans it must neutralize
            # the over-read rows. An all-zero row has num_chunks == 0 and stores the identity.
            last_chunk, tail_valid_rows = range_tail(length)
            tail_chunk = last_chunk - chunk_begin  # local index of the range's last chunk
            num_chunks = chunk_end - chunk_begin
            for ct in cutlass.range(0, num_chunks, unroll=0):
                is_tail = ct == tail_chunk and tail_valid_rows != 0
                # --- MMA2a: kg^T(SMEM A) × U(SMEM B) → kt(TMEM), value tiles only ---
                kgh = kg.wait()
                if is_tail:
                    zero_tail_tokens(sKg, kgh.index, tail_valid_rows, mma_tid, self.io_type, 1)
                ktd = kt.acquire()
                if is_value:
                    uh = u.wait()
                    if is_tail:
                        zero_tail_tokens(sU, uh.index, tail_valid_rows, mma_tid, self.io_type, 1)
                    for kp in cutlass.range_constexpr(cute.size(t_kg_au, mode=[2])):
                        mma_ktu.set(tcgen05.Field.ACCUMULATE, cutlass.Boolean(kp != 0))
                        cute.gemm(
                            mma_ktu,
                            t_kt_acc[None, None, None, ktd.index],
                            t_kg_au[None, None, kp, kgh.index],
                            t_u_b[None, None, kp, uh.index],
                            t_kt_acc[None, None, None, ktd.index],
                        )
                    uh.release()

                # --- MMA1: w(SMEM A) × X snapshot(SMEM B) → wx(TMEM) ---
                wh = w.wait()
                if is_tail:
                    zero_tail_tokens(sW, wh.index, tail_valid_rows, mma_tid, self.io_type, 0)
                wxd = wx.acquire()
                for split in cutlass.range_constexpr(2):
                    # Each hi/lo half is one pipeline stage, published separately.
                    xbh = xb.wait()
                    xbh.access("B", read=True, by="umma")
                    for kp in cutlass.range_constexpr(cute.size(t_w_a, mode=[2])):
                        mma_wx.set(
                            tcgen05.Field.ACCUMULATE, cutlass.Boolean(split != 0 or kp != 0)
                        )
                        cute.gemm(
                            mma_wx,
                            t_wx_acc[None, None, None, wxd.index],
                            t_w_a[None, None, kp, wh.index],
                            t_xb_b[None, None, kp, xbh.index],
                            t_wx_acc[None, None, None, wxd.index],
                        )
                    xbh.release()
                wxd.commit()
                wh.release()

                # --- MMA2b: kg^T(SMEM A) × -wx hi/lo(SMEM B) → kt(TMEM) ---
                for split in cutlass.range_constexpr(2):
                    tmpbh = tmpb.wait()
                    tmpbh.access("B", read=True, by="umma")
                    for kp in cutlass.range_constexpr(cute.size(t_kg_a, mode=[2])):
                        # The first pass overwrites unless MMA2a already seeded kt.
                        first = split == 0 and kp == 0
                        mma_kt.set(tcgen05.Field.ACCUMULATE, is_value if first else True)
                        cute.gemm(
                            mma_kt,
                            t_kt_acc[None, None, None, ktd.index],
                            t_kg_a[None, None, kp, kgh.index],
                            t_tmpb_b[None, None, kp, tmpbh.index],
                            t_kt_acc[None, None, None, ktd.index],
                        )
                    tmpbh.release()
                ktd.commit()
                kgh.release()
            work.advance()
        return w, kg, u, xb, tmpb, wx, kt, work

    @role(warps=range(4))
    @cute.jit
    def run_state(
        self,
        params: ForwardSummaryParams,
        *,
        xb: Producer,
        tmpb: Producer,
        gk: Consumer,
        wx: Consumer,
        kt: Consumer,
        work: WorkCursor,
        me: Local,
        tmem: Tmem,
    ):
        """Own the fp32 state registers; produce B operands and fold in results."""
        sXb_store = params.sXb_store
        sTmpb_store = params.sTmpb_store
        gk_buf = params.gk_3d[(None, 0, None)]
        t_wx_acc = cute.make_tensor(tmem.ptr("wx"), params.wx_layout)
        t_kt_acc = cute.make_tensor(tmem.ptr("kt"), params.kt_layout)
        local_tid = me.tmem_row

        # MMA1's wx accumulator is (BT, BN) and MMA2's kt is (K, BN).
        tc_t2r_wx, sl_wx, p_t_wx = tmem_tile_reader(t_wx_acc, local_tid, self.BN)
        tc_t2r_kt, sl_kt, p_t_kt = tmem_tile_reader(t_kt_acc, local_tid, self.BN)

        # Identity tensors provide both coordinate maps and range shapes.
        coords_tv = sl_wx.partition_D(cute.make_identity_tensor((BT, self.BN)))
        coords_kv = sl_kt.partition_D(cute.make_identity_tensor((KEY_DIM, self.BN)))

        # The X snapshot has kt's partition and Tmp has wx's.
        xb_writer = smem_tile_writer(tc_t2r_kt, local_tid, self.io_type)
        tmpb_writer = smem_tile_writer(tc_t2r_wx, local_tid, self.io_type)

        while work.valid:
            col_tile, head, row = work.item
            _start, _length, chunk_begin, chunk_end = decode_work_row(
                params.work_table, row, self.whole_ranges
            )
            col_base = col_tile * self.BN
            num_chunks = chunk_end - chunk_begin
            # X starts as this tile's slice of [0 | I]: value-tile columns are zero
            # and identity-tile column (col_base + nc) carries a one at key row kc.
            st = cute.make_rmem_tensor(coords_kv.shape, Float32)
            st.fill(Float32(0.0))
            for ei in cutlass.range(cute.size(st), unroll_full=True):
                kc, nc = coords_kv[ei]
                if col_base + nc == VAL_DIM + kc:
                    st[ei] = Float32(1.0)

            for _ct in cutlass.range(0, num_chunks, unroll=0):
                # ---- Phase 1: R2S X hi/lo → sXb (B operand for MMA1) ----
                # Publish the hi half as soon as it exists so MMA1 starts its first
                # pass while the lo residual is still being formed and stored.
                xbh = xb.acquire()
                xbh.access("Store", write=True, by="threads")
                st_hi = store_rounded(
                    st.load(), xb_writer, sXb_store[(None, None, xbh.index)], self.io_type
                )
                xbh.commit()
                xbh = xb.acquire()
                xbh.access("Store", write=True, by="threads")
                store_rounded(
                    st.load() - st_hi.load().to(Float32),
                    xb_writer,
                    sXb_store[(None, None, xbh.index)],
                    self.io_type,
                )
                xbh.commit()

                # Decay overlaps MMA1; keeping the gate release here keeps its proxy fence
                # off the final kt handoff.
                gkh = gk.wait()
                for ei in cutlass.range(cute.size(st), unroll_full=True):
                    kc, nc = coords_kv[ei]
                    st[ei] = st[ei] * cute.math.exp2(gk_buf[kc, gkh.index], fastmath=self.fastmath)
                gkh.release()

                # ---- Phase 2: T2R wx = W @ X ----
                wxh = wx.wait()
                wx_reg = cute.make_rmem_tensor(coords_tv.shape, Float32)
                cute.copy(tc_t2r_wx, p_t_wx[(None, None, None, wxh.index)], wx_reg)
                wxh.release()

                # ---- Phase 3: R2S -wx hi/lo (the U term accumulates via MMA2a) ----
                tmpbh = tmpb.acquire()
                tmpbh.access("Store", write=True, by="threads")
                tmp = cute.make_rmem_tensor(coords_tv.shape, Float32)
                for ei in cutlass.range_constexpr(cute.size(tmp)):
                    tmp[ei] = Float32(0.0) - wx_reg[ei]
                tmp_hi = store_rounded(
                    tmp.load(), tmpb_writer, sTmpb_store[(None, None, tmpbh.index)], self.io_type
                )
                tmpbh.commit()
                tmpbh = tmpb.acquire()
                tmpbh.access("Store", write=True, by="threads")
                store_rounded(
                    tmp.load() - tmp_hi.load().to(Float32),
                    tmpb_writer,
                    sTmpb_store[(None, None, tmpbh.index)],
                    self.io_type,
                )
                tmpbh.commit()

                # ---- Phase 4: add kt = Kg^T @ Tmp ----
                kth = kt.wait()
                kt_reg = cute.make_rmem_tensor(coords_kv.shape, Float32)
                cute.copy(tc_t2r_kt, p_t_kt[(None, None, None, kth.index)], kt_reg)
                kth.release()
                for ei in cutlass.range_constexpr(cute.size(st)):
                    st[ei] = st[ei] + kt_reg[ei]

            # Epilogue: store the V-first packed transpose out[s, h, col, k] = X[k, col].
            for ei in cutlass.range(cute.size(st), unroll_full=True):
                kc, nc = coords_kv[ei]
                params.out[row, head, col_base + nc, kc] = st[ei]
            work.advance()
        return xb, tmpb, gk, wx, kt, work

    # ------------------------------------------------------------------
    # Device kernel
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        mmas: Mmas,
        tma: TmaOps,
        mWork: cute.Tensor,
        mOut: cute.Tensor,
        smem_layouts: dict,
        bound,
    ):
        """``smem_layouts`` maps each ``smem_plan`` buffer to its staged layout."""
        self.begin_roles(work=bound, layouts=smem_layouts)
        mma_wx, mma_kt, _mma_ktu = mmas
        wx_fragment = mma_wx.make_fragment_C(
            cute.append(mma_wx.partition_shape_C(self.wx_tile[:2]), 1)
        )
        assert tcgen05.find_tmem_tensor_col_offset(wx_fragment) <= self.BN
        kt_fragment = mma_kt.make_fragment_C(
            cute.append(mma_kt.partition_shape_C(self.kt_tile[:2]), 1)
        )
        assert tcgen05.find_tmem_tensor_col_offset(kt_fragment) <= self.BN
        params = ForwardSummaryParams(
            work_table=mWork,
            out=mOut,
            mmas=mmas,
            tma=tma,
            wx_layout=wx_fragment.layout,
            kt_layout=kt_fragment.layout,
            sW=self.buffer("W"),
            sKg=self.buffer("Kg"),
            sU=self.buffer("U"),
            sXb=self.buffer("Xb"),
            sXb_store=self.buffer("XbStore"),
            sTmpb=self.buffer("Tmpb"),
            sTmpb_store=self.buffer("TmpbStore"),
            gk_3d=self.buffer("Gk"),
        )
        self.run_roles(params)


@jit_cache
def _compile_affine_summary(
    dtype_name: str,
    heads: int,
    state_bn: int,
    use_int64_offsets: bool,
    whole_ranges: bool,
    fastmath: bool = True,
):
    """Compile one dtype/head-count specialization from fake tensors."""
    target = get_compile_target()
    if target.device_type != "cuda" or not is_sm100_kda_capability(target.effective_capability):
        raise ValueError(f"affine_summary_fwd requires an SM100/SM103 target; got {target}")
    sym_int = cute.sym_int64 if use_int64_offsets else cute.sym_int
    tokens = sym_int()

    def factor(dtype, last_dim: int):
        return make_fake_compact_tensor(
            dtype,
            (1, tokens, heads, last_dim),
            stride_order=(3, 2, 1, 0),
            assumed_align=DATA_ALIGN_BYTES,
        )

    io_dtype = _CUTE_IO_TYPES[dtype_name]
    work = make_fake_compact_tensor(
        Int32, (sym_int(), 2 if whole_ranges else 4), stride_order=(1, 0)
    )
    out = make_fake_compact_tensor(
        Float32,
        (sym_int(), heads, SUMMARY_DIM, KEY_DIM),
        stride_order=(3, 2, 1, 0),
        assumed_align=DATA_ALIGN_BYTES,
    )
    return compile_tvm_ffi(
        _AffineSummaryFwdOp(
            dtype_name, heads, state_bn, use_int64_offsets, whole_ranges, fastmath
        ),
        factor(io_dtype, KEY_DIM),
        factor(io_dtype, KEY_DIM),
        factor(io_dtype, VAL_DIM),
        factor(Float32, KEY_DIM),
        work,
        out,
    )


# Splitting adds the planner and the fold (two small launches), so a work item must carry
# enough chunks for the shortened chain to pay for them. This floor puts the smallest split at
# the measured crossover, so every split is a win.
MIN_CHUNKS_PER_ITEM = 16
# Widening an otherwise one-wave scan pays for the extra planner/fold at long spans.
MIN_CHUNKS_FOR_WIDE_SPLIT = 128
# Both Triton fallbacks put ranges or work items on grid.z (limit 65535); leave room for
# the work budget, which is at most the SM count. Native OneShot work has a 1-D fallback.
MAX_RANGES = 65535 - 1024


def plan_work_budget(max_chunks: int, work_tiles: int, sm_count: int) -> int:
    """Return how many work items to scan concurrently for a span of at most ``max_chunks``.

    The chunk recurrence is a serial chain per CTA, so when ``work_tiles`` (column tiles times
    heads) leaves SMs idle the chunks are split into items that each scan from ``[0 | I]`` and
    ``compose_work_items`` folds the range maps. ``max_chunks`` is a static upper bound (the span
    length); the items themselves are cut on the device by ``plan_work_items``.
    """
    return max(1, min(max_chunks // MIN_CHUNKS_PER_ITEM, sm_count // work_tiles))


def select_state_columns(max_chunks: int, heads: int, ranges: int, sm_count: int) -> int:
    """Choose the native state tile from span capacity and available time parallelism."""
    narrow_tiles = heads * (SUMMARY_DIM // 32)
    if ranges * narrow_tiles > sm_count:
        return 64
    # BN32 is faster per chunk, but BN64 can split a long otherwise-unsplit recurrence
    # while keeping its active pieces in one wave. Already-split scans keep the narrower tile.
    if max_chunks >= MIN_CHUNKS_FOR_WIDE_SPLIT and sm_count // narrow_tiles == 1:
        return 64
    return 32


@torch.compiler.disable
def build_state_summaries(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    cumulative_gate: torch.Tensor,
    bounds: torch.Tensor,
    *,
    fastmath: bool = True,
) -> torch.Tensor:
    """Compute one packed affine state summary per token range of a stream's WY chunk factors.

    Args:
        kg: Gated keys ``k * exp2(gk_last - gk)``, shape ``[1, T, H, 128]``, bf16 or fp16.
        w: WY factor ``w``, same shape and dtype as ``kg``.
        u: WY pseudo-values ``u``, shape ``[1, T, H, 128]``, same dtype as ``kg``.
        cumulative_gate: Cumulative log2 gates, shape ``[1, T, H, 128]``, fp32.
        bounds: ``int32 [R, 2]`` device tensor of half-open token ranges ``[start, stop)`` with
            ``0 <= start <= stop <= T``; ``start == stop`` yields the identity map. A range is
            exact only over whole chunks of one packed sequence: ``start`` on that sequence's
            64-token chunk grid, ``stop`` on the grid or at the sequence's end (a partial last
            chunk is neutralized in-kernel). The values are not checked (that would sync);
            other ranges return a plausible but wrong map.
        fastmath: Use the flushing approximate ``exp2`` for the per-chunk state decay. Pass
            the same value the local chunk pass used so the summary decays match its own.

    Returns:
        FP32 tensor of shape ``[R, H, 256, 128]``, V-first packed as state bias then state
        transition for ``H_out = H_in @ transition + bias``, one per row of ``bounds``.

    The ranges are read on the device, so the launch is CUDA Graph capturable and replayable
    across different ``bounds`` values of the same shape. The tokens right after a partial last
    chunk are read but dropped by selection, so they may hold anything. Supported scope: B=1,
    fixed K=V=128 and BT=64, contiguous CUDA inputs. SM100 uses the native UMMA kernel; other
    capability 8.0+ targets use Triton. Both backends split long ranges into work items when
    the static budget permits, then compose their partial maps on the device.
    """
    assert kg.ndim == 4, f"kg must be 4D [1, T, H, K], got shape {tuple(kg.shape)}"
    batch, tokens, heads, key_dim = kg.shape
    assert batch == 1, f"build_state_summaries requires B=1, got B={batch}"
    assert tokens > 0, "build_state_summaries requires at least one token"
    assert heads > 0, "build_state_summaries requires at least one head"
    assert key_dim == KEY_DIM, f"build_state_summaries requires K={KEY_DIM}, got {key_dim}"
    assert w.shape == kg.shape, f"w must match kg shape {tuple(kg.shape)}, got {tuple(w.shape)}"
    assert u.shape == (1, tokens, heads, VAL_DIM), (
        f"u must have shape {(1, tokens, heads, VAL_DIM)}, got {tuple(u.shape)}"
    )
    assert cumulative_gate.shape == kg.shape, (
        f"cumulative_gate must match kg shape {tuple(kg.shape)}, "
        f"got {tuple(cumulative_gate.shape)}"
    )
    assert kg.dtype in _IO_TYPE_NAMES, f"kg dtype must be bf16 or fp16, got {kg.dtype}"
    assert w.dtype == kg.dtype and u.dtype == kg.dtype, (
        f"kg/w/u dtypes must match, got {kg.dtype}/{w.dtype}/{u.dtype}"
    )
    assert cumulative_gate.dtype == torch.float32, (
        f"cumulative_gate must be fp32, got {cumulative_gate.dtype}"
    )
    assert bounds.ndim == 2 and bounds.shape[1] == 2 and bounds.shape[0] > 0, (
        f"bounds must have shape [R, 2] with R > 0, got {tuple(bounds.shape)}"
    )
    assert bounds.dtype == torch.int32, f"bounds must be int32, got {bounds.dtype}"

    if isinstance(kg, FakeTensor):
        raise TypeError(
            "build_state_summaries does not support torch.export; run the context-parallel "
            "summary eagerly or under CUDA Graph capture"
        )

    assert kg.is_cuda, "build_state_summaries requires CUDA tensors"
    tensors = (("kg", kg), ("w", w), ("u", u), ("cumulative_gate", cumulative_gate))
    for name, tensor in (*tensors, ("bounds", bounds)):
        assert tensor.device == kg.device, f"{name} must be on {kg.device}, got {tensor.device}"
        assert tensor.is_contiguous(), f"{name} must be contiguous, got strides {tensor.stride()}"
    ranges = bounds.shape[0]
    if ranges > MAX_RANGES:
        raise ValueError(f"at most {MAX_RANGES} ranges per launch, got {ranges}")
    properties = get_device_properties(kg.device)
    capability = (properties.major, properties.minor)
    if capability < (8, 0):
        raise ValueError(f"affine_summary_fwd requires CUDA capability 8.0+, got {capability}")
    portable = not is_sm100_kda_capability(capability)
    sm_count = properties.multi_processor_count
    if portable:
        if capability[0] in (10, 12):
            from attn_gym._backends.triton.utils import configure_triton_allocator

            with torch.cuda.device(kg.device):
                configure_triton_allocator()
        from attn_gym.linear._delta_rule.triton.affine_summary_fwd import (
            _select_block_columns,
            launch_affine_summary_fwd,
        )

        state_bn = _select_block_columns(heads, capability)
    else:
        kg, w, u, cumulative_gate = (_aligned(tensor) for tensor in (kg, w, u, cumulative_gate))
        state_bn = select_state_columns(cdiv(tokens, BT), heads, ranges, sm_count)
    # The bounds live on the device, so the budget comes from the span's chunk count (a static
    # upper bound) and the device cuts the ranges' actual chunks into that many items.
    budget = plan_work_budget(cdiv(tokens, BT), heads * (SUMMARY_DIM // state_bn), sm_count)
    work, range_ids = work_table(bounds, budget)
    partials = torch.empty(
        (work.shape[0], heads, SUMMARY_DIM, KEY_DIM), dtype=torch.float32, device=kg.device
    )
    if portable:
        launch_affine_summary_fwd(
            kg,
            w,
            u,
            cumulative_gate,
            work,
            partials,
            capability,
            whole_ranges=range_ids is None,
            fastmath=fastmath,
        )
    else:
        use_int64_offsets = requires_int64_abi(kg, w, u, cumulative_gate, partials)
        compiled = _compile_affine_summary(
            _IO_TYPE_NAMES[kg.dtype],
            heads,
            state_bn,
            use_int64_offsets,
            whole_ranges=range_ids is None,
            fastmath=fastmath,
        )
        compiled(kg.detach(), w.detach(), u.detach(), cumulative_gate.detach(), work, partials)
    return compose_work_items(partials, range_ids, ranges, reverse=False)
