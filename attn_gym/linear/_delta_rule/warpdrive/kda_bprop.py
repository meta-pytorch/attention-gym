# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Derived from cudnn-frontend v1.30.0 (python/cudnn/linear_attention/frost/kernel/
# kda_bprop_f16.py), rewritten by Attention Gym in 2026 on attn_gym.warpdrive roles and channels
# with CuTeDSL 4.6.2 APIs: one TMA descriptor per tensor replaces the per-sequence descriptor
# arrays, checkpoints are addressed at closed-form rows, and only the natural-log gate the
# drivers compile remains.

"""KDA backward (bprop, BT = 16, per-key-channel decay) for SM100.

Per work item, the gradients of one output head over the chunks ``[write_start, compute_end)``
of one sequence (or chain piece), walked backward (``c = compute_end - 1`` down to
``write_start``). Per chunk, with ``g`` the in-chunk cumulative log2 gate per key channel, ``S``
the checkpoint entering the chunk and ``dS`` the state gradient carried backward in TMEM:
cg0 builds ``K_decay = K 2^g``, ``K_inv = K 2^-g``, ``Q_decay = scale Q 2^g`` and ``2^g_last``;
regmma ``KK = K_decay K_inv^T``, ``T = (I + beta strict(KK))^-1`` and ``dM = dY U^T``; the
epilogue ``A = lower(Q_decay K_inv^T)`` and ``dA = lower(dO U^T)``. The MMA warp then issues,
in upstream's order ((S) SMEM, (T) TMEM operand)::

    K S         = state(S) @ K_decay^T            dQ inter     = state(T) @ dO^T
    dU          = dS(T) @ K_inv + dO^T(S) @ A      dS Q-term   += dO^T(S) @ Q_decay
    U           = Y(T) @ T^-1,  Y = beta (V - K S) dY          = dU(T) @ T^-T
    dK restore  = dS(S) @ U^T                     dS K-term   += -beta dY(T) @ K_decay
    dK inv      = Q_decay^T(S) @ dA - K_decay^T(S) @ beta strict(dM)
    dQ          = dQ inter + K_inv^T(S) @ dA^T     dK decay    = state(T) @ (beta dY)^T
                                                                + K_inv^T(S) @ (beta strict(dM))^T

cg1 stages the TMEM operands and forms ``dV = beta dY``, dbeta (``sum_v dY (V - K S)`` plus the
M-term) and the next chunk's ``dS`` scaled by ``2^g_last``; cg2 assembles dQ and dK from the
parts with their per-channel ``2^(+-g)`` scales and the dA diagonal, and dGate per channel
(reverse-cumsummed over the chunk, plus the state term at its last token),
with upstream's instruction order, so every output matches the upstream kernel bit for bit.
With ``use_dstate_in`` the state gradient starts from ``dstate_in[batch, head]`` when
``compute_end`` is the sequence's last chunk (zero for a split item's warmup); with
``use_dstate0`` the final ``dS`` lands in ``dstate0[dstate_dst, head]``. dq/dk/dv/dgate/dbeta
are stored per output head for the chunks ``[write_start, write_end)``.

Work items are rows of the split-K work table (``work_tables``). The checkpoint entering
chunk ``c`` of item ``(batch b, head)`` is ``checkpoints[cu[b] // 16 + b + c, head]``
(``checkpoint_row``; ``cu`` is the cu_seqlens the work table was built over, so ``cu[b]`` is
the row's ``batch_start``). CTA ``u`` of the grid (one OneShot item each) starts at row ``u``;
the load warp claims later rows from the zeroed scheduler counter and publishes them.

Ragged TMA views replace upstream's per-sequence descriptor arrays: Q/K/V/gate/dO rows past a
sequence end load as zeros (``ragged.RaggedTma``), and outputs store through views that
clip at each sequence end.

Roles (16 warps, the upstream layout and register split)::

    cg0 (0-3, 144)   gate scan, decay operands, beta load, raw Q/K -> TMEM, state -> TMEM
    cg1 (4-7, 168)   dS seed/capture, Y / dU / -beta dY TMEM operands, U / dY / dV tiles, dbeta
    cg2 (8-11, 144)  dQ / dK / dGate assembly from the TMEM accumulators, staging
    regmma (12)      KK, T (blockwise inverse), dM, the dbeta M-term
    mma (13)         every tcgen05 GEMM
    load (14)        Q, K, Gate, V, dO, checkpoint TMA; claims the next work item
    epilogue (15)    A and dA (mma.sync), the dQ / dK / dGate / dV TMA stores (one behind)

Channels (``n`` stages; Threads arrive per thread, as upstream's barriers count)::

    raw      2  load (TMA Q,K,V,Gate)   -> cg0, cg1, cg2           (cg2 reads the exp2 exchange)
    do       2  load (TMA dO)           -> epilogue                 (mma: raw wait, see mma)
    state    1  load (TMA checkpoint)   -> mma (Umma), cg0
    beta     4  cg0 warp 0 (external)   -> regmma, cg1
    decay    2  cg0 (K/Q decay, K inv, 2^g_last) -> mma (Umma), epilogue
    qk_raw   4  cg0 (raw Q/K TMEM)      -> cg2
    state_in 2  cg0 (state TMEM f16)    -> mma (Umma), cg2
    tinv/a/da/dm 2 regmma/epilogue/epilogue/regmma (16 x 16 SMEM tiles) -> mma (Umma)
    da_diag  4  epilogue                -> cg2
    dqk      1  mma (dQ, dK parts TMEM) -> cg2
    dstate   1  cg1 (dS f16 SMEM)       -> mma (Umma), cg2
    dq/dk/dgate 1, dv 2  cg2 / cg1 (staging) -> epilogue (TMA store)  (dv: mma raw wait)
    sched    8  load                    -> every other role

``tsig`` keeps upstream's one-way signals with upstream's counts: the MMA warp <-> cg1 TMEM
recurrence (K S, Y, dU, U, dY, -beta dY, dS), U / dY SMEM tiles, the dbeta M-term, the early
publications of the dQ / dK parts and the decay stage, and the per-item dS store. Each guarded
region is rewritten only after a later handoff of the same chain (noted at ``TSIG``); as
managed rings their reuse waits would sit on the latency-bound per-chunk chain (see
``kda_prefill``).
"""

from dataclasses import dataclass

import cutlass
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, cute
from cutlass.cute.arch import inline_ptx
from cutlass.cute.nvgpu import cpasync, tcgen05

from attn_gym.warpdrive import (
    AFTER_FENCE_COVERED,
    BEFORE_FENCE_COVERED,
    BOTH_FENCES_COVERED,
    Barriers,
    Consumer,
    Local,
    NamedBarrier,
    OneShot,
    PdlWait,
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
    bulk_commit_group,
    bulk_wait_group,
    role,
)
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_strided_tensor
from attn_gym.warpdrive.compat import LayoutEnum
from attn_gym.warpdrive.thread_sync import tcgen05_after_thread_sync, tcgen05_before_thread_sync

from .blockwise_inverse import invert_unit_lower_16x16_fragments
from .contract import tma_divisibility
from .kda_common import (
    B_T,
    CUTE_DTYPES,
    TORCH_NAMES,
    WORK_ITEM_FIELDS,
    as_f32,
    as_i32,
    ldg_v4,
    lds_f32,
    lds_v4,
    lds_v4_f32,
    load_rows,
    mask_pair,
    stg_v4,
    sts_f32,
    swz32,
    swz128,
    umma_commit,
)
from .ptx import (
    elect_sync,
    f16x2_to_f32,
    fadd2,
    ffma2,
    fmul2,
    fp32_to_fp16,
    ldmatrix,
    mma_step,
    opaque_f32_zero,
    shfl_bfly,
    shfl_idx,
    st_shared_v4,
    stmatrix,
    tmem_ld,
    tmem_st,
)
from .ragged import (
    RaggedTma,
    fake_tma_slots,
    make_ragged_tma,
    new_tma_slots,
    ragged_store,
    sequence_descriptors,
    slot_region,
    tma_slot_rows,
)
from .umma import (
    SWIZZLE_32B,
    SWIZZLE_128B,
    MmaDesc,
    desc_opaque,
    instr_desc,
    mma_ss_op,
    mma_ts_all_op,
    smem_desc,
)

DIMS = (64, 128)
LOG2_E = 1.4426950408889634
DSTATE_DST = 9  # work-item field: the sequence whose d_initial_state the item stores
RAW_STAGES = 2
BETA_STAGES = 4
DECAY_STAGES = 2
INTER_STAGES = 2
INTER_TILES = 5  # per stage: A, T, dA, dM, -dM (16 x 16 each, 32-byte swizzle)
TILE = B_T * B_T * 2  # bytes of one 16 x 16 16-bit tile
DIAG_STAGES = 4
QK_RAW_STAGES = 4
DV_STAGES = 2
SCHED_STAGES = 8
# Managed channels wait with the suspend-time hint: the layer's spin wait peels one try and
# branches over the retry when it succeeds, and on this 16-warp kernel the taken branches cost
# more instruction fetch than the hint (A/B: 1.10x -> 1.05x of the upstream kernel at d 128).
# The tsig recurrence keeps upstream's hint-less loop (``_wait``; suspending it: 1.16x).
CHANNEL_WAIT = "suspend"
# The launch bundles compile every KDA kernel at --opt-level 2, which can move where LLVM
# contracts fp32 multiply-adds; the standalone build uses the same level as the bundled kernel.
OPT_LEVEL = 2

# Upstream's one-way signals (``tsig``), with their arrival counts. Producer -> waiters; the
# region each guards is rewritten only after the waiters' later handoff noted here.
#   STATE_K  K S accumulator (MMA commit) -> cg1. The dY GEMM reuses its columns after DU_IN,
#            which cg1 signals after reading K S.
#   DU_ACC, U_ACC, DY_ACC  MMA commits -> cg1; rewritten next chunk after DSTATE_IN / Y_IN /
#            DU_IN waits that cg1 signals after reading them.
#   DQ, DKI, DKR  early dQ / dK-inv / dK-restore publications inside the ``dqk`` handle.
#   Y_IN, DU_IN, NBDY_IN, DSTATE_IN  cg1 TMEM operands -> MMA; rewritten after the next K S /
#            dY / DSTATE_ACC / DSTATE_ACC commit that follows the GEMM reading them.
#   DSTATE_ACC  dS accumulator complete (MMA commit) -> cg1.
#   U_SMEM   sU (cg1) -> MMA, epilogue; DY_SMEM  sdY (cg1) -> regmma. Rewritten after U / dY of
#            the next chunk, issued after the MMA warp waited for dA / dM of this one.
#   DBETA_M  regmma's M-term row sums -> cg1; rewritten after DY_SMEM of the next chunk.
#   DSTATE0  cg1 stored the item's dS -> MMA, before the next item rewrites it.
#   KDI (2)  K decay / K inv of the decay stage -> mma, regmma, cg2 (early ``decay`` commit).
#   SCALE (2) 2^g_last of the decay stage -> cg1 (early ``decay`` commit).
STATE_K, DU_ACC, U_ACC, DY_ACC, DQ, DKI, DKR = 0, 1, 2, 3, 4, 5, 6
Y_IN, DU_IN, NBDY_IN, DSTATE_ACC, DSTATE_IN = 7, 8, 9, 10, 11
U_SMEM, DY_SMEM, DBETA_M, DSTATE0, KDI, SCALE = 12, 13, 14, 15, 16, 18
TSIG_COUNTS = (1, 1, 1, 1, 1, 1, 1, 128, 128, 128, 1, 128, 128, 128, 32, 128, 128, 128, 128, 128)


# ---- small PTX helpers ------------------------------------------------------------------------


@cute.jit
def _wait(bar, phase, fence: cutlass.Constexpr[bool] = True):
    """Hint-less parity wait written as upstream's loop (``while not try_wait``): ptxas places
    the retry out of line, so a satisfied wait falls through instead of taking a branch over a
    peeled retry (an instruction-fetch stall on this kernel's 16 warps). Then
    ``tcgen05.fence::after_thread_sync``: the raw ``tsig`` signals hand TMEM between threads;
    ``fence=False`` omits it where a later wait's fence precedes the caller's next tcgen05
    operation (``AFTER_FENCE_COVERED``)."""
    while not cute.arch.mbarrier_try_wait(bar, phase):
        pass
    if cutlass.const_expr(fence):
        tcgen05_after_thread_sync()


@cute.jit
def before_sync_and_arrive(bar):
    """One thread's arrival on a raw ``tsig`` signal after ``tcgen05.fence::before_thread_sync``,
    which orders its completed TMEM accesses before the arrival (PTX ISA 9.7.18.6.3)."""
    tcgen05_before_thread_sync()
    arrive(bar)


@cute.jit
def arrive(bar):
    """One thread's arrival on a raw ``tsig`` signal without a fence: only where an earlier
    before fence already follows its TMEM accesses (``BEFORE_FENCE_COVERED``)."""
    cute.arch.mbarrier_arrive(bar)


@cute.jit
def _lds_u16_pair(addr0, addr1):
    """Two 16-bit SMEM elements packed into one word (``addr0`` in the low half)."""
    return inline_ptx(
        "{ .reg .b16 l, h; ld.shared.b16 l, [$1]; ld.shared.b16 h, [$2]; mov.b32 $0, {l, h}; }",
        write_only_types=[Int32],
        read_only_args=[Int32(addr0), Int32(addr1)],
    )


@cute.jit
def _sts_16(addr, value, dtype: cutlass.Constexpr):
    """Round one fp32 value to ``dtype`` (16-bit) and store it."""
    tag = "bf16" if cutlass.const_expr(dtype is cutlass.BFloat16) else "f16"
    inline_ptx(
        f"{{ .reg .b16 h; cvt.rn.{tag}.f32 h, $1; st.shared.b16 [$0], h; }}",
        read_only_args=[Int32(addr), value],
    )


@cute.jit
def _half(word, n: cutlass.Constexpr[int], dtype: cutlass.Constexpr):
    """16-bit half ``n`` of a packed word, widened to fp32 (exact). Kept as DSL conversions, as
    upstream writes them: LLVM then folds the widening into the consumer (``fma.rn.f32.bf16``,
    half selectors) instead of a separate unpack."""
    return cutlass.Vector.from_elements((word,), Int32).bitcast(dtype)[n].to(Float32)


@cute.jit
def _lds_wide(addr, dtype: cutlass.Constexpr):
    """One 16-bit SMEM element at byte address ``addr``, widened to fp32 (see ``_half``)."""
    return cute.make_ptr(dtype, addr, cute.AddressSpace.smem, assumed_align=2).load().to(Float32)


@dataclass(frozen=True)
class KdaBpropParams:
    q_load: RaggedTma  # (token, dim, head) loads, zero past each sequence end
    k_load: RaggedTma
    v_load: RaggedTma
    g_load: RaggedTma
    do_load: RaggedTma
    dq_store: RaggedTma  # (token, dim, head) stores, dropping rows past each sequence end
    dk_store: RaggedTma
    dv_store: RaggedTma
    dg_store: RaggedTma
    tma_ckpt: cute.CopyAtom
    mCkpt: cute.Tensor  # (d_v, d_k, head, row)
    box_io: cute.ComposedLayout  # one 16 x 64 16-bit swizzled box
    box_g: cute.ComposedLayout  # one 16 x 128-byte gate box
    box_ckpt: cute.ComposedLayout  # one d_v x 64 16-bit swizzled box
    beta: cute.Tensor  # (token, head)
    dbeta: cute.Tensor
    dstate_in: cute.Tensor | None  # (row, head, d_v, d_k) fp32
    dstate0: cute.Tensor | None
    work_items: cute.Tensor  # (rows, WORK_ITEM_FIELDS)
    work_count: cute.Tensor  # (1,)
    scheduler: cute.Tensor  # (>= 1,) zeroed ticket counter
    q_ratio: Int32  # output heads per q / k / v head
    k_ratio: Int32
    v_ratio: Int32
    scale: Float32
    s_state: Int32  # SMEM byte addresses of the buffers (stage 0)
    s_dstate: Int32
    s_kdecay: Int32
    s_kinv: Int32
    s_qdecay: Int32
    s_scale: Int32
    s_inter: Int32
    s_do: Int32
    s_dv: Int32
    s_q: Int32
    s_k: Int32
    s_v: Int32
    s_g: Int32
    s_dy: Int32
    s_u: Int32
    s_dq: Int32
    s_dk: Int32
    s_dg: Int32
    s_beta: Int32
    s_diag: Int32
    s_red: Int32
    s_betam: Int32


class KdaBprop(WarpSpecialized):
    """Warpdrive KDA backward for ``io`` fp16/bf16 operands and an fp32 or ``io`` gate.

    ``use_initial_state``: the item's first chunk has a checkpoint (else chunk 0 starts from
    zero state). ``use_dstate_in`` / ``use_dstate0``: seed dS from ``dstate_in`` / store the
    final dS to ``dstate0``."""

    def __init__(
        self,
        io: str,
        gate: str,
        d_k: int,
        d_v: int,
        use_initial_state: bool,
        use_dstate_in: bool,
        use_dstate0: bool,
        num_ctas: int,
    ):
        if d_k not in DIMS or d_v not in DIMS:
            raise ValueError(f"the KDA bprop serves d_k, d_v in {DIMS}, got {d_k}, {d_v}")
        if io not in ("fp16", "bf16") or gate not in ("fp32", io):
            raise ValueError(f"io must be fp16/bf16 and the gate fp32 or io, got {io}, {gate}")
        self.io, self.gate, self.d_k, self.d_v = io, gate, d_k, d_v
        self.use_initial_state = use_initial_state
        self.use_dstate_in, self.use_dstate0 = use_dstate_in, use_dstate0
        self.num_ctas = num_ctas
        self.first_state_chunk = 0 if use_initial_state else 1
        self.io_dtype = io_t = CUTE_DTYPES[io]
        self.gate_dtype = CUTE_DTYPES[gate]
        f32 = Float32
        io_b = io_t.width // 8
        plan = self.smem_plan = SmemPlan()
        # Small fp32 arrays first, then upstream's SharedStorage order.
        buf = {
            "Beta": plan.buffer("Beta", f32, align=64),
            "Diag": plan.buffer("Diag", f32, align=64),
            "Red": plan.buffer("Red", f32, align=64),
            "BetaM": plan.buffer("BetaM", f32, align=64),
        }
        for name in ("State", "Dstate", "Kdecay", "Kinv", "Qdecay"):
            buf[name] = plan.buffer(name, io_t, align=1024)
        buf["Scale"] = plan.buffer("Scale", f32, align=16)
        for name in ("Inter", "Do", "Dv", "Q", "K", "V"):
            buf[name] = plan.buffer(name, io_t, align=1024)
        buf["Gate"] = plan.buffer("Gate", f32, align=1024)  # a 16-bit gate fills each half
        for name in ("Dy", "U", "Dq", "Dk"):
            buf[name] = plan.buffer(name, io_t, align=1024)
        buf["Dgate"] = plan.buffer("Dgate", self.gate_dtype, align=1024)
        self.buf = buf

        raw_bytes = B_T * (2 * d_k * io_b + d_v * io_b + d_k * (self.gate_dtype.width // 8))
        # The TMA check reports the loads unverified: it cannot follow the per-item views.
        self.raw = Ring(
            RAW_STAGES,
            Tma(raw_bytes),
            {"cg0": Threads, "cg1": Threads, "cg2": Threads},
            buffer={k: buf[k] for k in ("Q", "K", "V", "Gate")},
            wait=CHANNEL_WAIT,
        )
        # The MMA warp also reads dO (three GEMMs) without releasing it, as upstream: the load
        # warp refills a stage after the epilogue's release, which follows U_SMEM of the same
        # chunk, which cg1 signals after the U GEMM's commit, which covers every earlier GEMM.
        self.do = Ring(
            RAW_STAGES, Tma(B_T * d_v * io_b), Threads, buffer=buf["Do"], wait=CHANNEL_WAIT
        )
        self.state = Ring(
            1,
            Tma(d_k * d_v * io_b),
            {"mma": Umma, "cg0": Threads},
            buffer=buf["State"],
            wait=CHANNEL_WAIT,
        )
        # cg0's warp 0 fills beta (upstream: 32 thread arrivals); regmma and cg1 read it.
        self.beta = Ring(
            BETA_STAGES,
            consumer=Threads,
            external="producer",
            override={"full_counts": 32},
            buffer=buf["Beta"],
            wait=CHANNEL_WAIT,
        )
        # One decay stage holds K decay, K inv, Q decay and 2^g_last; cg0 publishes the first
        # two and the last early (tsig KDI / SCALE) and commits after Q decay.
        self.decay = Ring(
            DECAY_STAGES,
            Threads,
            {"mma": Umma, "epilogue": Threads},
            commit_fence="smem_async",
            buffer={k: buf[k] for k in ("Kdecay", "Kinv", "Qdecay", "Scale")},
            wait=CHANNEL_WAIT,
        )
        # cg2 only reads these TMEM stages: its releases wait for the loads (tcgen05.wait::ld)
        # before cg0 may store the next chunk over them.
        self.qk_raw = Ring(
            QK_RAW_STAGES,
            Threads,
            Threads,
            commit_fence="tmem_store",
            release_fence="tmem_load",
            wait=CHANNEL_WAIT,
        )
        self.state_in = Ring(
            2,
            Threads,
            {"mma": Umma, "cg2": Threads},
            commit_fence="tmem_store",
            release_fence="tmem_load",
            wait=CHANNEL_WAIT,
        )
        # The 16 x 16 tiles share upstream's interleaved ``Inter`` stage (A, T, dA, dM, -dM).
        smem = "smem_async"
        self.tinv = Ring(INTER_STAGES, Threads, Umma, commit_fence=smem, wait=CHANNEL_WAIT)
        self.a = Ring(INTER_STAGES, Threads, Umma, commit_fence=smem, wait=CHANNEL_WAIT)
        self.da = Ring(INTER_STAGES, Threads, Umma, commit_fence=smem, wait=CHANNEL_WAIT)
        self.dm = Ring(INTER_STAGES, Threads, Umma, commit_fence=smem, wait=CHANNEL_WAIT)
        self.da_diag = Ring(DIAG_STAGES, Threads, Threads, buffer=buf["Diag"], wait=CHANNEL_WAIT)
        self.dqk = Ring(1, Umma, Threads, release_fence="tmem_load", wait=CHANNEL_WAIT)
        self.dstate = Ring(
            1,
            Threads,
            {"mma": Umma, "cg2": Threads},
            commit_fence=smem,
            buffer=buf["Dstate"],
            wait=CHANNEL_WAIT,
        )
        # Staging for the epilogue's TMA stores (the async proxy reads them).
        self.dq = Ring(1, Threads, Threads, commit_fence=smem, buffer=buf["Dq"], wait=CHANNEL_WAIT)
        # dK is staged by the same threads right after dQ: dQ's commit fence covers its stores.
        self.dk = Ring(1, Threads, Threads, commit_fence=(), buffer=buf["Dk"], wait=CHANNEL_WAIT)
        self.dgate = Ring(
            1, Threads, Threads, commit_fence=smem, buffer=buf["Dgate"], wait=CHANNEL_WAIT
        )
        # The MMA warp reads dV (dK decay part) without releasing it, as upstream: cg1 rewrites
        # a stage after the epilogue's release, two chunks later, after waiting for K S of that
        # chunk, whose commit covers the dK decay GEMM.
        self.dv = Ring(
            DV_STAGES, Threads, Threads, commit_fence=smem, buffer=buf["Dv"], wait=CHANNEL_WAIT
        )
        self.sched = Ring(SCHED_STAGES, Warps, Warps, payload=(Int32, (1,)), wait=CHANNEL_WAIT)
        self.tsig = Barriers(len(TSIG_COUNTS), counts=list(TSIG_COUNTS))
        self.cg0_sync = NamedBarrier(threads=128)
        self.cg1_sync = NamedBarrier(threads=128)
        self.tmem_plan = (
            TmemPlan()
            .region("dstate", d_k)
            .region("dstate_in", d_k // 2)
            .region("state_in", d_v // 2, 2)
            .region("kstate", B_T)  # K S, then dY
            .region("u", B_T)
            .region("du", B_T)
            .region("dq", B_T)
            .region("dkd", B_T)
            .region("dki", B_T)
            .region("dkr", B_T)
            .region("yin", B_T // 2)  # Y, then -beta dY
            .region("duin", B_T // 2)
            .region("qraw", B_T // 2, QK_RAW_STAGES)
            .region("kraw", B_T // 2, QK_RAW_STAGES)
        )
        self.work = WorkStream(schedule=OneShot(grid="1d"), ready=PdlWait())

        # ---- GEMM geometry (upstream's MmaDesc set) ----------------------------------------
        def desc(m, n, k, *, at=False, bt=False):
            idesc = instr_desc(f32, io_t, io_t, m=m, n=n, a_mn_major=at, b_mn_major=bt)
            return MmaDesc(
                M=m, N=n, K=k, bpe_a=io_b, bpe_b=io_b, idesc=idesc, atranspose=at, btranspose=bt
            )

        self.g_kstate = desc(d_v, B_T, d_k)  # state(S) @ K decay^T; dS input(T) @ K inv
        self.g_do_a = desc(d_v, B_T, B_T, at=True, bt=True)  # dO^T(S) @ A
        self.g_do_qd = desc(d_v, d_k, B_T, at=True, bt=True)  # dO^T(S) @ Q decay
        self.g_t = desc(d_v, B_T, B_T)  # Y(T) @ T^-1
        self.g_t_trans = desc(d_v, B_T, B_T, bt=True)  # dU(T) @ T^-T
        self.g_dy_kd = desc(d_v, d_k, B_T, bt=True)  # -beta dY(T) @ K decay
        self.g_state_t = desc(d_k, B_T, d_v)  # state(T) @ dO^T / (beta dY)^T
        self.g_dstate_u = desc(d_k, B_T, d_v, at=True)  # dS(S) @ U^T
        self.g_kt_d = desc(d_k, B_T, B_T, at=True)  # K inv^T(S) @ dA^T / dM^T
        self.g_kt_dt = desc(d_k, B_T, B_T, at=True, bt=True)  # Q / K decay^T(S) @ dA / -dM

    def tma_slot_rows(self, heads: int) -> int:
        """Rows of the TMA slot buffer: Q, K, V, gate, dO, dQ, dK, dV and dGate, keyed by head."""
        return tma_slot_rows(9, B_T, heads)

    def get_name(self) -> str:
        flags = f"{int(self.use_initial_state)}{int(self.use_dstate_in)}{int(self.use_dstate0)}"
        return f"wd_kda_bprop_{self.io}_g{self.gate}_k{self.d_k}_v{self.d_v}_f{flags}"

    # ---- work decode -------------------------------------------------------------------

    @cute.jit
    def item(self, params: KdaBpropParams, tile):
        """``(batch, head, batch_start, seqlen, write_start, write_end, compute_end)`` of work
        row ``tile`` (chunk-unit bounds; batch_start and seqlen in tokens)."""
        row = params.work_items
        start = row[tile, 6]
        return (
            row[tile, 0],
            row[tile, 1],
            start,
            row[tile, 7] - start,
            row[tile, 2],
            row[tile, 3],
            row[tile, 5],
        )

    @cute.jit
    def next_tile(self, sched, skip_fence=None):
        """The next tile of the item list; ``skip_fence`` as for the consume (its reason is
        ``AFTER_FENCE_COVERED``/``BEFORE_FENCE_COVERED``)."""
        reason = BEFORE_FENCE_COVERED if skip_fence == "before" else AFTER_FENCE_COVERED
        with sched.consume(skip_fence=skip_fence, reason=reason) as h:
            tile = h.data[0]
        return sched, tile

    @cute.jit
    def has_dstate(self, rev):
        """Whether this chunk starts from a state gradient (TMEM dS)."""
        if cutlass.const_expr(self.use_dstate_in):
            return cutlass.Boolean(True)
        return cutlass.Boolean(rev > 0)

    @cute.jit
    def sig_wait(
        self, name: cutlass.Constexpr[str], bar, phase, fence: cutlass.Constexpr[bool] = True
    ):
        """A raw (``tsig`` or unbound channel) wait, as a named trace range when tracing
        (``fence`` as for ``_wait``)."""
        with self.trace_region(name + ".wait"):
            _wait(bar, phase, fence)

    @cute.jit
    def lane_base(self, tmem, warp):
        """TMEM address of lane 0 of this warp's 32-lane subpartition (column 0)."""
        return tmem.base.toint() + ((32 * (warp & 3)) << 16)

    # ---- load (warp 14): Q, K, Gate, V, dO, checkpoints; the next item -----------------

    @role(warps=14, regs=56)
    @cute.jit
    def load(
        self,
        params: KdaBpropParams,
        *,
        raw: Producer,
        do: Producer,
        state: Producer,
        sched: Producer,
        work: WorkCursor,
    ):
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                fields = self.item(params, tile)
                raw, do, state = self.load_item(params, fields, raw, do, state)
                with sched.produce() as sh:
                    if elect_sync():
                        sh.data[0] = Int32(self.num_ctas) + cute.arch.atomic_add(
                            params.scheduler.iterator.llvm_ptr,
                            Int32(1),
                            sem="relaxed",
                            scope="gpu",
                        )
                    cute.arch.sync_warp()
                    tile = sh.data[0]
            work.advance()
        cute.arch.griddepcontrol_launch_dependents()
        return raw, do, state, sched, work

    @cute.jit
    def load_item(self, params: KdaBpropParams, fields, raw, do, state):
        """Q, K, gate, V, dO and checkpoints of every chunk of one item, last chunk first."""
        io, io_b = self.io_dtype, self.io_dtype.width // 8
        g_cols = 128 // (self.gate_dtype.width // 8)
        box = B_T * 128  # bytes of one 16-row 128-byte box
        box_io = params.box_io
        batch, head, start, seqlen, write_start, _we, compute_end = fields
        head_q = head // params.q_ratio
        head_k = head // params.k_ratio
        head_v = head // params.v_ratio
        loads = (params.q_load, params.k_load, params.g_load, params.v_load, params.do_load)
        moves_rows = compute_end > write_start
        descs = sequence_descriptors(loads, start + seqlen, head, B_T, moves_rows)
        q_desc, k_desc, g_desc, v_desc, do_desc = descs
        for rev in cutlass.range(compute_end - write_start, unroll=1):
            chunk = compute_end - 1 - rev
            rows = (start + chunk * B_T, cutlass.min(seqlen - chunk * B_T, B_T))
            with raw.produce() as h:
                bar = h.barrier
                qk_stage = h.index * (self.d_k * B_T * io_b)
                for kb in cutlass.range_constexpr(self.d_k // 64):
                    q_addr = params.s_q + qk_stage + kb * box
                    load_rows(params.q_load, io, q_addr, box_io, rows, head_q, kb, 64, bar, q_desc)
                for kb in cutlass.range_constexpr(self.d_k // 64):
                    k_addr = params.s_k + qk_stage + kb * box
                    load_rows(params.k_load, io, k_addr, box_io, rows, head_k, kb, 64, bar, k_desc)
                g_stage = h.index * (self.d_k * B_T * 4)
                for kb in cutlass.range_constexpr(self.d_k // g_cols):
                    load_rows(
                        params.g_load,
                        self.gate_dtype,
                        params.s_g + g_stage + kb * box,
                        params.box_g,
                        rows,
                        head,
                        kb,
                        g_cols,
                        bar,
                        g_desc,
                    )
                v_stage = h.index * (self.d_v * B_T * io_b)
                for kb in cutlass.range_constexpr(self.d_v // 64):
                    v_addr = params.s_v + v_stage + kb * box
                    load_rows(params.v_load, io, v_addr, box_io, rows, head_v, kb, 64, bar, v_desc)
            with do.produce() as h:
                do_stage = h.index * (self.d_v * B_T * io_b)
                for kb in cutlass.range_constexpr(self.d_v // 64):
                    do_addr = params.s_do + do_stage + kb * box
                    load_rows(
                        params.do_load, io, do_addr, box_io, rows, head, kb, 64, h.barrier, do_desc
                    )
            if chunk >= self.first_state_chunk:
                with state.produce() as h:
                    row = checkpoint_row(start, batch, chunk)
                    for kb in cutlass.range_constexpr(self.d_k // 64):
                        dst = cute.make_tensor(
                            cute.make_ptr(
                                self.io_dtype,
                                params.s_state + kb * (self.d_v * 128),
                                cute.AddressSpace.smem,
                                assumed_align=1024,
                            ),
                            params.box_ckpt,
                        )
                        src = cute.local_tile(
                            params.mCkpt[None, None, head, row], (self.d_v, 64), (0, kb)
                        )
                        s, d = cpasync.tma_partition(
                            params.tma_ckpt,
                            0,
                            cute.make_layout(1),
                            cute.group_modes(dst, 0, 2),
                            cute.group_modes(src, 0, 2),
                        )
                        cute.copy(params.tma_ckpt, d, s, tma_bar_ptr=h.barrier)
        return raw, do, state

    # ---- compute group 0 (warps 0-3): gate scan, decay operands, raw Q/K, state copy --------

    @role(warps=range(4), regs=144)
    @cute.jit
    def cg0(
        self,
        params: KdaBpropParams,
        *,
        raw: Consumer,
        state: Consumer,
        decay: Producer,
        qk_raw: Producer,
        state_in: Producer,
        sched: Consumer,
        work: WorkCursor,
        tsig: Barriers,
        tmem: Tmem,
        me: Local,
    ):
        lane, warp = me.lane, me.warp
        d_k = self.d_k
        channel_rows = d_k // 4
        channel_active = cutlass.Boolean(True)
        channel = warp * 32 + lane
        if cutlass.const_expr(channel_rows < 32):
            channel_active = lane < channel_rows
            channel = warp * channel_rows + (lane & (channel_rows - 1))
        lane_base = self.lane_base(tmem, warp)
        beta_empty = self.barrier_ptr("beta") + BETA_STAGES
        beta_full = self.barrier_ptr("beta")
        base = Int32(0)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                _b, head, start, seqlen, write_start, _we, compute_end = self.item(params, tile)
                n_chunks = compute_end - write_start
                for rev in cutlass.range(n_chunks, unroll=1):
                    n = base + rev
                    chunk = compute_end - 1 - rev
                    chunk_start = chunk * B_T
                    # ---- beta (warp 0): one chunk's scalars, zero past the sequence end --------
                    if warp == 0:
                        bstage = n & (BETA_STAGES - 1)
                        self.sig_wait(
                            "beta_empty", beta_empty + bstage, ((n >> 2) + 1) & 1, fence=False
                        )
                        if lane < B_T:
                            token = chunk_start + lane
                            value = Float32(0.0)
                            if token < seqlen:
                                value = params.beta[start + token, head].to(Float32)
                            sts_f32(params.s_beta + 4 * (bstage * B_T + lane), value)
                        cute.arch.mbarrier_arrive(beta_full + bstage)
                    raw, state, decay, qk_raw, state_in = self.cg0_chunk(
                        params,
                        raw,
                        state,
                        decay,
                        qk_raw,
                        state_in,
                        tsig,
                        chunk,
                        chunk_start,
                        seqlen,
                        lane,
                        warp,
                        channel,
                        channel_active,
                        lane_base,
                        tmem,
                    )
                base += n_chunks
                sched, tile = self.next_tile(sched, skip_fence="before")
            work.advance()
        return raw, state, decay, qk_raw, state_in, sched, work

    @cute.jit
    def cg0_chunk(
        self,
        params,
        raw,
        state,
        decay,
        qk_raw,
        state_in,
        tsig,
        chunk,
        chunk_start,
        seqlen,
        lane,
        warp,
        channel,
        channel_active,
        lane_base,
        tmem,
    ):
        io = self.io_dtype
        d_k, d_v = self.d_k, self.d_v
        dk_halves = d_k // 64
        channel_rows = d_k // 4
        valid = seqlen - chunk_start
        rh = raw.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
        s_q = params.s_q + rh.index * (d_k * B_T * 2)
        s_k = params.s_k + rh.index * (d_k * B_T * 2)
        s_ex = params.s_g + rh.index * (d_k * B_T * 4)  # fp32 exchange over the raw gate stage

        # ---- gate prefix scan: cumulative log2 gate per key channel, exp2 --------------------
        f32_seg = channel >> 5
        f32_dim = channel - f32_seg * 32
        prefix = Float32(0.0)
        exp_g_last = Float32(0.0)
        if cutlass.const_expr(self.gate == "fp32"):
            for pair in cutlass.range(B_T // 2, unroll=1):
                row0 = pair * 2
                row1 = row0 + 1
                g0 = lds_f32(
                    s_ex + 4 * (f32_seg * (B_T * 32) + row0 * 32 + swz128(row0, f32_dim, 4))
                )
                g1 = lds_f32(
                    s_ex + 4 * (f32_seg * (B_T * 32) + row1 * 32 + swz128(row1, f32_dim, 4))
                )
                g0 = g0 * Float32(LOG2_E) if chunk_start + row0 < seqlen else Float32(0.0)
                g1 = g1 * Float32(LOG2_E) if chunk_start + row1 < seqlen else Float32(0.0)
                p0, pair_sum = fadd2(prefix, g0, g0, g1)
                p1 = prefix + pair_sum
                e0 = cute.math.exp2(p0, fastmath=True)
                e1 = cute.math.exp2(p1, fastmath=True)
                # exp2(g) goes in place over this raw gate pair, into slots that other lanes of
                # this warp read (below 32 channel rows, duplicate lanes store equal values to
                # equal slots): the warp sync orders every lane's reads of the pair first. Later
                # pairs touch other rows. Loading all 16 rows before one sync, as cuDNN Frontend
                # v1.32.1 does with a cg0 barrier (v1.30.0 had none), measured 2-5% slower.
                cute.arch.sync_warp()
                sts_f32(
                    s_ex
                    + 4 * (f32_seg * (B_T * 32) + row0 * 32 + swz128(row0 ^ f32_seg, f32_dim, 4)),
                    e0,
                )
                sts_f32(
                    s_ex
                    + 4 * (f32_seg * (B_T * 32) + row1 * 32 + swz128(row1 ^ f32_seg, f32_dim, 4)),
                    e1,
                )
                prefix = p1
                exp_g_last = e1
        else:
            raw_seg = channel >> 6
            raw_col = channel - raw_seg * 64
            graw = cute.make_rmem_tensor((B_T,), Float32)
            for row in cutlass.range_constexpr(B_T):
                g = _lds_wide(
                    s_ex + 2 * (raw_seg * (B_T * 64) + row * 64 + swz128(row, raw_col)),
                    self.gate_dtype,
                )
                graw[row] = g * Float32(LOG2_E) if chunk_start + row < seqlen else Float32(0.0)
            # Every raw gate is read before the exchange stores.
            self.cg0_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
            for pair in cutlass.range_constexpr(B_T // 2):
                row0 = pair * 2
                row1 = row0 + 1
                p0, pair_sum = fadd2(prefix, graw[row0], graw[row0], graw[row1])
                p1 = prefix + pair_sum
                e0 = cute.math.exp2(p0, fastmath=True)
                e1 = cute.math.exp2(p1, fastmath=True)
                if channel_active:
                    sts_f32(
                        s_ex
                        + 4
                        * (f32_seg * (B_T * 32) + row0 * 32 + swz128(row0 ^ f32_seg, f32_dim, 4)),
                        e0,
                    )
                    sts_f32(
                        s_ex
                        + 4
                        * (f32_seg * (B_T * 32) + row1 * 32 + swz128(row1 ^ f32_seg, f32_dim, 4)),
                        e1,
                    )
                prefix = p1
                exp_g_last = e1

        # ---- the decay stage: 2^g_last first (cg1's dS decay) --------------------------------
        dh = decay.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
        stage = dh.index
        if channel_active:
            sts_f32(params.s_scale + 4 * (stage * d_k + channel), exp_g_last)
        arrive(tsig + SCALE + stage)

        # ---- raw Q/K: SMEM -> TMEM ring (channel-major, zero past the sequence end) ---------
        qh = qk_raw.acquire()
        r_seg = channel >> 6
        r_dim = channel - r_seg * 64
        q_words = cute.make_rmem_tensor((B_T // 2,), Int32)
        k_words = cute.make_rmem_tensor((B_T // 2,), Int32)
        for t2 in cutlass.range_constexpr(B_T // 2):
            t0 = 2 * t2
            a0 = 2 * (r_seg * (B_T * 64) + t0 * 64 + swz128(t0, r_dim))
            a1 = 2 * (r_seg * (B_T * 64) + (t0 + 1) * 64 + swz128(t0 + 1, r_dim))
            q_words[t2] = _lds_u16_pair(s_q + a0, s_q + a1)
            k_words[t2] = _lds_u16_pair(s_k + a0, s_k + a1)
        # Q/K rows past the sequence end load as zeros; a tail chunk also masks them, behind
        # one uniform branch.
        ragged = valid < B_T
        if ragged:
            for t2 in cutlass.range_constexpr(B_T // 2):
                q_words[t2] = mask_pair(q_words[t2], Int32(2 * t2), valid)
                k_words[t2] = mask_pair(k_words[t2], Int32(2 * t2), valid)
        qslot = qh.index * (B_T // 2)
        tmem_st("32x32b", lane_base + tmem.cols("qraw") + qslot, [q_words[i] for i in range(8)])
        tmem_st("32x32b", lane_base + tmem.cols("kraw") + qslot, [k_words[i] for i in range(8)])
        qh.commit()
        # Every exp2 exchange store lands before its reads.
        self.cg0_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)

        # ---- raw Q/K rows, exp2(g) of the row -------------------------------------------------
        decay_row = warp * (B_T // 4) + (lane >> 3)
        lane_in_row = lane & 7
        q_rows = cute.make_rmem_tensor((dk_halves * 4,), Int32)
        k_rows = cute.make_rmem_tensor((dk_halves * 4,), Int32)
        for half in cutlass.range_constexpr(dk_halves):
            idx = half * (B_T * 64) + decay_row * 64 + swz128(decay_row, lane_in_row * 8)
            qw = lds_v4(s_q + 2 * idx)
            kw = lds_v4(s_k + 2 * idx)
            for i in cutlass.range_constexpr(4):
                q_rows[half * 4 + i] = qw[i]
                k_rows[half * 4 + i] = kw[i]
        if ragged:
            row_valid = decay_row < valid
            for i in cutlass.range_constexpr(dk_halves * 4):
                q_rows[i] = q_rows[i] if row_valid else Int32(0)
                k_rows[i] = k_rows[i] if row_valid else Int32(0)
        raw_q = []
        raw_k = []
        for i in cutlass.range_constexpr(dk_halves * 4):
            lo, hi = f16x2_to_f32(q_rows[i], dtype=io)
            raw_q += [lo, hi]
            lo, hi = f16x2_to_f32(k_rows[i], dtype=io)
            raw_k += [lo, hi]
        cute.arch.fence_view_async_shared()
        exp_g = []
        for half in cutlass.range_constexpr(dk_halves):
            dim_base = half * 64 + lane_in_row * 8
            for grp in cutlass.range_constexpr(2):
                f_dim = dim_base + grp * 4
                f_seg = f_dim // 32
                idx = (
                    f_seg * (B_T * 32)
                    + decay_row * 32
                    + swz128(decay_row ^ f_seg, f_dim - f_seg * 32, 4)
                )
                exp_g += lds_v4_f32(s_ex + 4 * idx)
        rh.release(skip_fence="before", reason=BEFORE_FENCE_COVERED)

        # ---- K decay = K 2^g, K inv = K 2^-g ----------------------------------------------------
        op_stage = stage * (d_k * B_T * 2)
        for half in cutlass.range_constexpr(dk_halves):
            kd = []
            ki = []
            for p in cutlass.range_constexpr(4):
                r0 = half * 8 + 2 * p
                kd0, kd1 = fmul2(raw_k[r0], raw_k[r0 + 1], exp_g[r0], exp_g[r0 + 1])
                kd.append(fp32_to_fp16(kd0, kd1, dtype=io))
                inv0 = cute.math.rcp(exp_g[r0], approx=True, ftz=True)
                inv1 = cute.math.rcp(exp_g[r0 + 1], approx=True, ftz=True)
                ki0, ki1 = fmul2(raw_k[r0], raw_k[r0 + 1], inv0, inv1)
                ki.append(fp32_to_fp16(ki0, ki1, dtype=io))
            off = op_stage + 2 * (
                half * (B_T * 64) + decay_row * 64 + swz128(decay_row, lane_in_row * 8)
            )
            st_shared_v4(params.s_kinv + off, ki)
            st_shared_v4(params.s_kdecay + off, kd)
        cute.arch.fence_view_async_shared()
        arrive(tsig + KDI + stage)

        # ---- Q decay = (scale Q) 2^g --------------------------------------------------------------
        scale = params.scale
        for half in cutlass.range_constexpr(dk_halves):
            qd = []
            for p in cutlass.range_constexpr(4):
                r0 = half * 8 + 2 * p
                q0, q1 = fmul2(raw_q[r0], raw_q[r0 + 1], scale, scale)
                qd0, qd1 = fmul2(q0, q1, exp_g[r0], exp_g[r0 + 1])
                qd.append(fp32_to_fp16(qd0, qd1, dtype=io))
            off = op_stage + 2 * (
                half * (B_T * 64) + decay_row * 64 + swz128(decay_row, lane_in_row * 8)
            )
            st_shared_v4(params.s_qdecay + off, qd)
        dh.commit(skip_fence="before", reason=BEFORE_FENCE_COVERED)

        # ---- state copy: SMEM -> TMEM f16 (k lanes x v columns) ------------------------------
        sih = state_in.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
        if chunk >= self.first_state_chunk:
            sth = state.wait()
            ldm_row = ((lane >> 4) << 3) + (lane & 7)
            ldm_col = ((lane >> 3) & 1) << 3
            k_base = warp * channel_rows
            k_seg_off = (k_base // 64) * (d_v * 64)
            state_col = lane_base + tmem.cols("state_in") + sih.index * (d_v // 2)
            for blk in cutlass.range_constexpr(d_v // 16):
                dv_row = blk * 16 + ldm_row
                frag_lo = ldmatrix(
                    params.s_state
                    + 2 * (k_seg_off + dv_row * 64 + swz128(dv_row, (k_base + ldm_col) & 63)),
                    4,
                    trans=True,
                )
                tmem_st("16x128b", state_col + blk * 8, frag_lo)
                if cutlass.const_expr(channel_rows == 32):
                    frag_hi = ldmatrix(
                        params.s_state
                        + 2
                        * (k_seg_off + dv_row * 64 + swz128(dv_row, (k_base + 16 + ldm_col) & 63)),
                        4,
                        trans=True,
                    )
                    tmem_st("16x128b", state_col + (16 << 16) + blk * 8, frag_hi)
            cute.arch.fence_view_async_tmem_store()
            sth.release()
        sih.commit(skip_fence="before", reason=BEFORE_FENCE_COVERED)
        return raw, state, decay, qk_raw, state_in

    # ---- compute group 1 (warps 4-7): dS seed/capture, value-side TMEM operands, dbeta -----

    @role(warps=range(4, 8), regs=168)
    @cute.jit
    def cg1(
        self,
        params: KdaBpropParams,
        *,
        raw: Consumer,
        beta: Consumer,
        dv: Producer,
        dstate: Producer,
        sched: Consumer,
        work: WorkCursor,
        tsig: Barriers,
        tmem: Tmem,
        me: Local,
    ):
        lane, warp = me.lane, me.warp
        ov_row = ((lane >> 4) << 3) + (lane & 7)
        ov_col = ((lane >> 3) & 1) << 3
        if cutlass.const_expr(self.d_v == 128):
            value_dim = warp * 32 + lane
            value_base = warp * 32
            row_valid = cutlass.Boolean(True)
        else:
            value_dim = warp * 16 + (lane & 15)
            value_base = warp * 16
            row_valid = lane < 16
        lane_base = self.lane_base(tmem, warp)
        ov_lo = (
            ((value_base + ov_col) >> 6) * (B_T * 64)
            + ov_row * 64
            + swz128(ov_row, (value_base + ov_col) & 63)
        )
        ov_hi = (
            ((value_base + 16 + ov_col) >> 6) * (B_T * 64)
            + ov_row * 64
            + swz128(ov_row, (value_base + 16 + ov_col) & 63)
        )
        geo = (lane, me.tid, warp, lane_base, ov_lo, ov_hi, value_dim, row_valid)
        n_ks = Int32(0)  # completed K S waits (chunks with a checkpoint)
        base = Int32(0)  # chunks of earlier items: the per-chunk signals' epoch
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                batch, head, start, seqlen, write_start, write_end, compute_end = self.item(
                    params, tile
                )
                n_chunks = compute_end - write_start
                if cutlass.const_expr(self.use_dstate_in):  # noqa: SIM102 (static, then dynamic)
                    if n_chunks > 0:
                        dstate = self.seed_dstate(
                            params,
                            dstate,
                            tsig,
                            geo,
                            tmem,
                            base,
                            batch,
                            head,
                            compute_end == cute.ceil_div(seqlen, B_T),
                        )
                for rev in cutlass.range(n_chunks, unroll=1):
                    raw, beta, dv, dstate, n_ks = self.cg1_chunk(
                        params,
                        raw,
                        beta,
                        dv,
                        dstate,
                        tsig,
                        geo,
                        tmem,
                        base + rev,
                        n_ks,
                        rev + 1 < n_chunks,
                        compute_end - 1 - rev,
                        start,
                        seqlen,
                        head,
                        write_end,
                    )
                if cutlass.const_expr(self.use_dstate0):
                    self.store_dstate0(params, geo, tmem, tile, batch, head, n_chunks)
                if n_chunks > 0:
                    before_sync_and_arrive(tsig + DSTATE0)
                base += n_chunks
                sched, tile = self.next_tile(sched)
            work.advance()
        return raw, beta, dv, dstate, sched, work

    @cute.jit
    def seed_dstate(self, params, dstate, tsig, geo, tmem, n, batch, head, seed_true):
        """dS of the item's last chunk: ``dstate_in[batch, head] 2^g_last`` (zero for a split
        item's warmup), as the fp32 accumulator, its fp16 MMA operand and the SMEM tile."""
        _l, _t, _w, lane_base, _lo, _hi, value_dim, row_valid = geo
        dh = dstate.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
        stage = n & 1
        self.sig_wait("scale", tsig + SCALE + stage, (n >> 1) & 1)
        s_scale = params.s_scale + stage * (self.d_k * 4)
        src = params.dstate_in
        addr = (src.iterator + src.layout((batch, head, value_dim, 0))).toint()
        for i in cutlass.range(self.d_k // 16, unroll=1):
            block = cute.make_rmem_tensor((16,), Float32)
            for g in cutlass.range_constexpr(4):
                words = ldg_v4(addr + (i * 16 + g * 4) * 4)
                for t in cutlass.range_constexpr(4):
                    block[g * 4 + t] = as_f32(words[t]) if seed_true else Float32(0.0)
            for g in cutlass.range_constexpr(4):
                f = lds_v4_f32(s_scale + 4 * (i * 16 + g * 4))
                for t in cutlass.range_constexpr(2):
                    block[g * 4 + 2 * t], block[g * 4 + 2 * t + 1] = fmul2(
                        block[g * 4 + 2 * t], block[g * 4 + 2 * t + 1], f[2 * t], f[2 * t + 1]
                    )
            tmem_st(
                "32x32b", lane_base + tmem.cols("dstate") + i * 16, [block[j] for j in range(16)]
            )
            words = [
                fp32_to_fp16(block[2 * p], block[2 * p + 1], dtype=self.io_dtype) for p in range(8)
            ]
            tmem_st("32x32b", lane_base + tmem.cols("dstate_in") + i * 8, words)
        cute.arch.fence_view_async_tmem_store()
        before_sync_and_arrive(tsig + DSTATE_IN)
        for i in cutlass.range_constexpr(self.d_k // 16):
            words = tmem_ld("32x32b", 8, lane_base + tmem.cols("dstate_in") + i * 8, Int32)
            if row_valid:
                for half in cutlass.range_constexpr(2):
                    self.store_dstate_words(
                        params, value_dim, i * 16 + half * 8, words[4 * half :]
                    )
        dh.commit()
        return dstate

    @cute.jit
    def store_dstate_words(self, params, value_dim, d_base: cutlass.Constexpr, words):
        """Eight fp16 dS values ``[value_dim, d_base : d_base + 8]`` into the swizzled tile."""
        addr = (d_base // 64) * (self.d_v * 64) + value_dim * 64 + swz128(value_dim, d_base % 64)
        st_shared_v4(params.s_dstate + 2 * addr, [words[0], words[1], words[2], words[3]])

    @cute.jit
    def cg1_chunk(
        self,
        params,
        raw,
        beta,
        dv,
        dstate,
        tsig,
        geo,
        tmem,
        n,
        n_ks,
        capture,
        chunk,
        start,
        seqlen,
        head,
        write_end,
    ):
        lane, tid, warp, lane_base, ov_lo, ov_hi, value_dim, row_valid = geo
        io = self.io_dtype
        wide = self.d_v == 128
        hi_lanes = lane_base + (16 << 16)
        ph = n & 1  # phase of the per-chunk signals
        chunk_start = chunk * B_T
        valid = seqlen - chunk_start

        # ---- Y = beta (V - K S) ---------------------------------------------------------------
        rh = raw.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
        s_v = params.s_v + rh.index * (self.d_v * B_T * 2)
        v_lo = ldmatrix(s_v + 2 * ov_lo, 4, trans=True)
        v_hi = [Int32(0)] * 4
        if cutlass.const_expr(wide):
            v_hi = ldmatrix(s_v + 2 * ov_hi, 4, trans=True)
        v_words = cute.make_rmem_tensor((8,), Int32)
        for r in cutlass.range_constexpr(4):
            v_words[r] = v_lo[r]
            v_words[4 + r] = v_hi[r]
        if valid < B_T:  # V past the sequence end reads as upstream's zeros (a tail chunk only)
            for r in cutlass.range_constexpr(4):
                token0 = ((r // 2) * 4 + (lane & 3)) * 2
                v_words[r] = mask_pair(v_words[r], Int32(token0), valid)
                if cutlass.const_expr(wide):
                    v_words[4 + r] = mask_pair(v_words[4 + r], Int32(token0), valid)
        v_lo = [v_words[r] for r in range(4)]
        v_hi = [v_words[4 + r] for r in range(4)]
        rh.release()
        bh = beta.wait()
        s_beta = params.s_beta + bh.index * (B_T * 4)
        betas = []
        for half in cutlass.range_constexpr(2):
            token0 = (half * 4 + (lane & 3)) * 2
            betas += [lds_f32(s_beta + 4 * token0), lds_f32(s_beta + 4 * token0 + 4)]
        diff_lo = cute.make_rmem_tensor((8,), Float32)
        diff_hi = cute.make_rmem_tensor((8,), Float32)
        y_lo = cute.make_rmem_tensor((4,), Int32)
        y_hi = cute.make_rmem_tensor((4,), Int32)
        if chunk >= self.first_state_chunk:
            self.sig_wait("state_k", tsig + STATE_K, n_ks & 1)
            n_ks += 1
            ks_lo = tmem_ld("16x256b", 2, lane_base + tmem.cols("kstate"))
            for r in cutlass.range_constexpr(4):
                y_lo[r], diff_lo[2 * r], diff_lo[2 * r + 1] = _beta_residual(
                    v_lo[r], betas[2 * (r // 2)], betas[2 * (r // 2) + 1], ks_lo[2 * r],
                    ks_lo[2 * r + 1], io,
                )  # fmt: skip
            if cutlass.const_expr(wide):
                ks_hi = tmem_ld("16x256b", 2, hi_lanes + tmem.cols("kstate"))
                for r in cutlass.range_constexpr(4):
                    y_hi[r], diff_hi[2 * r], diff_hi[2 * r + 1] = _beta_residual(
                        v_hi[r], betas[2 * (r // 2)], betas[2 * (r // 2) + 1], ks_hi[2 * r],
                        ks_hi[2 * r + 1], io,
                    )  # fmt: skip
        if chunk < self.first_state_chunk:
            for r in cutlass.range_constexpr(4):
                y_lo[r], diff_lo[2 * r], diff_lo[2 * r + 1] = _beta_residual(
                    v_lo[r], betas[2 * (r // 2)], betas[2 * (r // 2) + 1], None, None, io
                )
                if cutlass.const_expr(wide):
                    y_hi[r], diff_hi[2 * r], diff_hi[2 * r + 1] = _beta_residual(
                        v_hi[r], betas[2 * (r // 2)], betas[2 * (r // 2) + 1], None, None, io
                    )
        tmem_st("16x128b", lane_base + tmem.cols("yin"), [y_lo[i] for i in range(4)])
        if cutlass.const_expr(wide):
            tmem_st("16x128b", hi_lanes + tmem.cols("yin"), [y_hi[i] for i in range(4)])
        cute.arch.fence_view_async_tmem_store()
        before_sync_and_arrive(tsig + Y_IN)

        # ---- dU accumulator -> fp16 MMA operand ----------------------------------------------
        self.sig_wait("du_acc", tsig + DU_ACC, ph)
        du_lo = tmem_ld("16x256b", 2, lane_base + tmem.cols("du"))
        du_hi = [Float32(0.0)] * 8
        if cutlass.const_expr(wide):
            du_hi = tmem_ld("16x256b", 2, hi_lanes + tmem.cols("du"))
        tmem_st(
            "16x128b",
            lane_base + tmem.cols("duin"),
            [fp32_to_fp16(du_lo[2 * r], du_lo[2 * r + 1], dtype=io) for r in range(4)],
        )
        if cutlass.const_expr(wide):
            tmem_st(
                "16x128b",
                hi_lanes + tmem.cols("duin"),
                [fp32_to_fp16(du_hi[2 * r], du_hi[2 * r + 1], dtype=io) for r in range(4)],
            )
        cute.arch.fence_view_async_tmem_store()
        before_sync_and_arrive(tsig + DU_IN)

        # ---- U -> sU ---------------------------------------------------------------------------
        self.sig_wait("u_acc", tsig + U_ACC, ph)
        u_lo = tmem_ld("16x256b", 2, lane_base + tmem.cols("u"))
        u_hi = [Float32(0.0)] * 8
        if cutlass.const_expr(wide):
            u_hi = tmem_ld("16x256b", 2, hi_lanes + tmem.cols("u"))
        stmatrix(
            params.s_u + 2 * ov_lo,
            [fp32_to_fp16(u_lo[2 * r], u_lo[2 * r + 1], dtype=io) for r in range(4)],
            trans=True,
        )
        if cutlass.const_expr(wide):
            stmatrix(
                params.s_u + 2 * ov_hi,
                [fp32_to_fp16(u_hi[2 * r], u_hi[2 * r + 1], dtype=io) for r in range(4)],
                trans=True,
            )
        cute.arch.fence_view_async_shared()
        before_sync_and_arrive(tsig + U_SMEM)

        # ---- dY -> sdY -------------------------------------------------------------------------
        self.sig_wait("dy_acc", tsig + DY_ACC, ph)
        dy_lo = tmem_ld("16x256b", 2, lane_base + tmem.cols("kstate"))
        dy_hi = [Float32(0.0)] * 8
        if cutlass.const_expr(wide):
            dy_hi = tmem_ld("16x256b", 2, hi_lanes + tmem.cols("kstate"))
        stmatrix(
            params.s_dy + 2 * ov_lo,
            [fp32_to_fp16(dy_lo[2 * r], dy_lo[2 * r + 1], dtype=io) for r in range(4)],
            trans=True,
        )
        if cutlass.const_expr(wide):
            stmatrix(
                params.s_dy + 2 * ov_hi,
                [fp32_to_fp16(dy_hi[2 * r], dy_hi[2 * r + 1], dtype=io) for r in range(4)],
                trans=True,
            )
        cute.arch.fence_view_async_shared()
        before_sync_and_arrive(tsig + DY_SMEM)

        # ---- beta dY, -beta dY -> TMEM -------------------------------------------------------
        c0 = lds_f32(s_beta + 4 * ((lane & 3) * 2))
        c1 = lds_f32(s_beta + 4 * ((lane & 3) * 2 + 1))
        c8 = lds_f32(s_beta + 4 * ((lane & 3) * 2 + 8))
        c9 = lds_f32(s_beta + 4 * ((lane & 3) * 2 + 9))
        bh.release(skip_fence="before", reason=BEFORE_FENCE_COVERED)
        bdy_lo = cute.make_rmem_tensor((8,), Float32)
        bdy_hi = cute.make_rmem_tensor((8,), Float32)
        for e2 in cutlass.range_constexpr(4):
            e = 2 * e2
            b_lo = c8 if cutlass.const_expr(e >= 4) else c0
            b_hi = c9 if cutlass.const_expr(e >= 4) else c1
            bdy_lo[e], bdy_lo[e + 1] = fmul2(dy_lo[e], dy_lo[e + 1], b_lo, b_hi)
            if cutlass.const_expr(wide):
                bdy_hi[e], bdy_hi[e + 1] = fmul2(dy_hi[e], dy_hi[e + 1], b_lo, b_hi)
        tmem_st(
            "16x128b",
            lane_base + tmem.cols("yin"),
            [fp32_to_fp16(-bdy_lo[2 * r], -bdy_lo[2 * r + 1], dtype=io) for r in range(4)],
        )
        if cutlass.const_expr(wide):
            tmem_st(
                "16x128b",
                hi_lanes + tmem.cols("yin"),
                [fp32_to_fp16(-bdy_hi[2 * r], -bdy_hi[2 * r + 1], dtype=io) for r in range(4)],
            )
        cute.arch.fence_view_async_tmem_store()
        before_sync_and_arrive(tsig + NBDY_IN)

        # ---- dbeta v-term: sum_v dY (V - K S) --------------------------------------------------
        token4 = cute.make_rmem_tensor((4,), Float32)
        for s in cutlass.range_constexpr(4):
            token4[s] = Float32(0.0)
        for j in cutlass.range_constexpr(4):
            slot = 2 * (j // 2)
            lo, hi = ffma2(
                dy_lo[2 * j], dy_lo[2 * j + 1], diff_lo[2 * j], diff_lo[2 * j + 1],
                token4[slot], token4[slot + 1],
            )  # fmt: skip
            if cutlass.const_expr(wide):
                lo, hi = ffma2(
                    dy_hi[2 * j], dy_hi[2 * j + 1], diff_hi[2 * j], diff_hi[2 * j + 1], lo, hi
                )
            token4[slot] = lo
            token4[slot + 1] = hi

        # ---- dV = beta dY -> staging ---------------------------------------------------------------
        vh = dv.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
        s_dv = params.s_dv + vh.index * (B_T * self.d_v * 2)
        stmatrix(
            s_dv + 2 * ov_lo,
            [fp32_to_fp16(bdy_lo[2 * r], bdy_lo[2 * r + 1], dtype=io) for r in range(4)],
            trans=True,
        )
        if cutlass.const_expr(wide):
            stmatrix(
                s_dv + 2 * ov_hi,
                [fp32_to_fp16(bdy_hi[2 * r], bdy_hi[2 * r + 1], dtype=io) for r in range(4)],
                trans=True,
            )
        vh.commit(skip_fence="before", reason=BEFORE_FENCE_COVERED)

        # ---- dbeta = v-term + M-term ----------------------------------------------------------------
        self.cg1_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
        self.sig_wait("dbeta_m", tsig + DBETA_M, ph, fence=False)
        for off in cutlass.range_constexpr(3):
            for s in cutlass.range_constexpr(4):
                token4[s] = token4[s] + shfl_bfly(token4[s], 4 << off)
        if lane < 4:
            for s in cutlass.range_constexpr(4):
                sts_f32(
                    params.s_red + 4 * (warp * B_T + (lane & 3) * 2 + (s % 2) + 8 * (s // 2)),
                    token4[s],
                )
        self.cg1_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)
        if tid < B_T:
            acc = Float32(0.0)
            for w in cutlass.range_constexpr(4):
                acc = acc + lds_f32(params.s_red + 4 * (w * B_T + tid))
            db = acc + lds_f32(params.s_betam + 4 * tid)
            if tid < valid and chunk < write_end:
                params.dbeta[start + chunk_start + tid, head] = db.to(params.dbeta.element_type)
        self.cg1_sync.sync(aligned=True, skip_fence="both", reason=BOTH_FENCES_COVERED)

        # ---- dS for the next chunk: 2^g_last(next) dS, its fp16 operand and SMEM tile -------
        # Fenced only for the dS store after the last chunk (``store_dstate0``) without the
        # scale wait of a capture.
        self.sig_wait("dstate_acc", tsig + DSTATE_ACC, ph, fence=self.use_dstate0)
        if capture:
            dsh = dstate.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
            nxt = n + 1
            stage = nxt & 1
            self.sig_wait("scale", tsig + SCALE + stage, (nxt >> 1) & 1)
            s_scale = params.s_scale + stage * (self.d_k * 4)
            for i in cutlass.range(self.d_k // 32, unroll=1):
                vec = tmem_ld("32x32b", 32, lane_base + tmem.cols("dstate") + i * 32)
                scaled = cute.make_rmem_tensor((32,), Float32)
                for g in cutlass.range_constexpr(8):
                    f = lds_v4_f32(s_scale + 4 * (i * 32 + g * 4))
                    for t in cutlass.range_constexpr(2):
                        k = g * 4 + 2 * t
                        scaled[k], scaled[k + 1] = fmul2(
                            vec[k], vec[k + 1], f[2 * t], f[2 * t + 1]
                        )
                words = [
                    fp32_to_fp16(scaled[2 * p], scaled[2 * p + 1], dtype=io) for p in range(16)
                ]
                tmem_st("32x32b", lane_base + tmem.cols("dstate_in") + i * 16, words)
                tmem_st(
                    "32x32b",
                    lane_base + tmem.cols("dstate") + i * 32,
                    [scaled[j] for j in range(32)],
                )
            cute.arch.fence_view_async_tmem_store()
            before_sync_and_arrive(tsig + DSTATE_IN)
            for i in cutlass.range(self.d_k // 32, unroll=1):
                words = tmem_ld("32x32b", 16, lane_base + tmem.cols("dstate_in") + i * 16, Int32)
                if row_valid:
                    for q in cutlass.range_constexpr(4):
                        d_base = i * 32 + q * 8
                        addr = (
                            (d_base // 64) * (self.d_v * 64)
                            + value_dim * 64
                            + swz128(value_dim, d_base % 64)
                        )
                        st_shared_v4(params.s_dstate + 2 * addr, words[4 * q : 4 * q + 4])
            dsh.commit()
        return raw, beta, dv, dstate, n_ks

    @cute.jit
    def store_dstate0(self, params, geo, tmem, tile, batch, head, n_chunks):
        """The item's dS -> ``dstate0[dstate_dst, head]``; an item with no chunks passes
        ``dstate_in`` (or zeros) through."""
        _l, _t, _w, lane_base, _lo, _hi, value_dim, row_valid = geo
        out = params.dstate0
        dst = params.work_items[tile, DSTATE_DST]
        if n_chunks > 0:
            if dst >= 0:
                addr = (out.iterator + out.layout((dst, head, value_dim, 0))).toint()
                for i in cutlass.range_constexpr(self.d_k // 32):
                    vec = tmem_ld("32x32b", 32, lane_base + tmem.cols("dstate") + i * 32)
                    if row_valid:
                        for g in cutlass.range_constexpr(8):
                            stg_v4(
                                addr + (i * 32 + g * 4) * 4,
                                [as_i32(vec[g * 4 + t]) for t in range(4)],
                            )
                # Before DSTATE0 lets the MMA rewrite dS, or teardown frees TMEM.
                cute.arch.fence_view_async_tmem_load()
        else:
            if row_valid and dst >= 0:
                for kd in cutlass.range(self.d_k, unroll=1):  # cold: empty items
                    if cutlass.const_expr(self.use_dstate_in):
                        out[dst, head, value_dim, kd] = params.dstate_in[
                            batch, head, value_dim, kd
                        ]
                    else:
                        out[dst, head, value_dim, kd] = Float32(0.0)

    # ---- compute group 2 (warps 8-11): dQ / dK / dGate assembly -----------------------------

    @role(warps=range(8, 12), regs=144)
    @cute.jit
    def cg2(
        self,
        params: KdaBpropParams,
        *,
        raw: Consumer,
        qk_raw: Consumer,
        state_in: Consumer,
        dstate: Consumer,
        dqk: Consumer,
        da_diag: Consumer,
        dq: Producer,
        dk: Producer,
        dgate: Producer,
        sched: Consumer,
        work: WorkCursor,
        tsig: Barriers,
        tmem: Tmem,
        me: Local,
    ):
        lane, warp = me.lane, me.warp
        lane_base = self.lane_base(tmem, warp)
        n_dkr = Int32(0)  # completed dK restore waits (chunks with a dS)
        base = Int32(0)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                _b, _h, _s, _len, write_start, _we, compute_end = self.item(params, tile)
                n_chunks = compute_end - write_start
                for rev in cutlass.range(n_chunks, unroll=1):
                    if cutlass.const_expr(self.d_k == 64):
                        raw, qk_raw, state_in, dstate, dqk, da_diag, dq, dk, dgate, n_dkr = (
                            self.cg2_chunk_split(
                                params, raw, qk_raw, state_in, dstate, dqk, da_diag, dq, dk,
                                dgate, tsig, tmem, lane, warp, lane_base, base + rev, n_dkr,
                                rev, compute_end - 1 - rev,
                            )
                        )  # fmt: skip
                    else:
                        raw, qk_raw, state_in, dstate, dqk, da_diag, dq, dk, dgate, n_dkr = (
                            self.cg2_chunk(
                                params, raw, qk_raw, state_in, dstate, dqk, da_diag, dq, dk,
                                dgate, tsig, tmem, lane, warp, lane_base, base + rev, n_dkr,
                                rev, compute_end - 1 - rev,
                            )
                        )  # fmt: skip
                base += n_chunks
                sched, tile = self.next_tile(sched)
            work.advance()
        return raw, qk_raw, state_in, dstate, dqk, da_diag, dq, dk, dgate, sched, work

    @cute.jit
    def cg2_chunk(
        self,
        params,
        raw,
        qk_raw,
        state_in,
        dstate,
        dqk,
        da_diag,
        dq,
        dk,
        dgate,
        tsig,
        tmem,
        lane,
        warp,
        lane_base,
        n,
        n_dkr,
        rev,
        chunk,
    ):
        """One chunk with one key channel per thread (d_k = 128)."""
        io = self.io_dtype
        d_v = self.d_v
        scale = params.scale
        has_dstate = self.has_dstate(rev)
        channel = warp * 32 + lane
        f32_seg = channel >> 5
        f32_dim = channel - f32_seg * 32
        f16_seg = channel >> 6
        f16_dim = channel - f16_seg * 64
        stage = n & 1
        # The exchange is complete.
        self.sig_wait("kdi", tsig + KDI + stage, (n >> 1) & 1, fence=False)

        # ---- per-channel 2^g of every token ---------------------------------------------------
        rh = raw.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
        s_ex = params.s_g + rh.index * (self.d_k * B_T * 4)
        eg = cute.make_rmem_tensor((B_T,), Float32)
        for t in cutlass.range_constexpr(B_T):
            eg[t] = lds_f32(
                s_ex + 4 * (f32_seg * (B_T * 32) + t * 32 + swz128(t ^ f32_seg, f32_dim, 4))
            )
        rh.release()
        qh = qk_raw.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
        qraw_col = lane_base + tmem.cols("qraw") + qh.index * (B_T // 2)
        kraw_col = lane_base + tmem.cols("kraw") + qh.index * (B_T // 2)

        # ---- dGate last: sum_v dS[v, c] S[c, v] ------------------------------------------------
        dgate_last = Float32(0.0)
        sih = state_in.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
        if has_dstate:
            dsh = dstate.wait()
            for trip in cutlass.range(d_v // 32, unroll=1):
                svec = tmem_ld(
                    "32x32b",
                    16,
                    lane_base + tmem.cols("state_in") + sih.index * (d_v // 2) + trip * 16,
                    Int32,
                )
                hacc = cute.make_rmem_tensor((8,), Float32)
                for i in cutlass.range_constexpr(8):
                    hacc[i] = opaque_f32_zero()
                for j in cutlass.range_constexpr(16):
                    v0 = trip * 32 + 2 * j
                    s0 = _half(svec[j], 0, io)
                    s1 = _half(svec[j], 1, io)
                    a0 = f16_seg * (d_v * 64) + v0 * 64 + swz128(2 * j, f16_dim)
                    a1 = f16_seg * (d_v * 64) + (v0 + 1) * 64 + swz128(2 * j + 1, f16_dim)
                    h0 = _lds_wide(params.s_dstate + 2 * a0, io)
                    h1 = _lds_wide(params.s_dstate + 2 * a1, io)
                    hacc[(2 * j) % 8] = hacc[(2 * j) % 8] + h0 * s0
                    hacc[(2 * j + 1) % 8] = hacc[(2 * j + 1) % 8] + h1 * s1
                pa0, pb0 = fadd2(hacc[0], hacc[2], hacc[4], hacc[6])
                pa1, pb1 = fadd2(hacc[1], hacc[3], hacc[5], hacc[7])
                part_a, part_b = fadd2(pa0, pb0, pa1, pb1)
                dgate_last = dgate_last + (part_a + part_b)
            dsh.release()
        sih.release(skip_fence="before", reason=BEFORE_FENCE_COVERED)

        dq_n = cute.make_rmem_tensor((B_T,), Float32)
        dk_n = cute.make_rmem_tensor((B_T,), Float32)
        dg = cute.make_rmem_tensor((B_T,), Float32)
        last_acc = cute.make_rmem_tensor((4,), Float32)
        for i in cutlass.range_constexpr(4):
            last_acc[i] = opaque_f32_zero()
        for t in cutlass.range_constexpr(B_T):
            dk_n[t] = Float32(0.0)

        # ---- dK restore part: 2^-g (dS U^T), dGate last K-dot -----------------------------------
        if has_dstate:
            self.sig_wait("dkr", tsig + DKR, n_dkr & 1)
            n_dkr += 1
            dkr = tmem_ld("32x32b", B_T, lane_base + tmem.cols("dkr"))
            kr = tmem_ld("32x32b", B_T // 2, kraw_col, Int32)
            for t in cutlass.range_constexpr(B_T):
                dk_hat = cute.math.rcp(eg[t], approx=True, ftz=True) * dkr[t]
                dk_n[t] = dk_hat
                k_v = _half(kr[t // 2], t % 2, io)
                last_acc[t % 4] = last_acc[t % 4] + k_v * dk_hat

        # ---- dQ = 2^g scale dQ acc -----------------------------------------------------------------
        self.sig_wait(
            "dq", tsig + DQ, n & 1
        )  # dQ, dK inv and dK restore are early dqk publications
        dqv = tmem_ld("32x32b", B_T, lane_base + tmem.cols("dq"))
        for t2 in cutlass.range_constexpr(B_T // 2):
            t = 2 * t2
            lo, hi = fmul2(eg[t], eg[t + 1], scale, scale)
            dq_n[t], dq_n[t + 1] = fmul2(lo, hi, dqv[t], dqv[t + 1])

        # ---- dK inv part: 2^-g (dA - dM) term ---------------------------------------------------
        self.sig_wait("dki", tsig + DKI, n & 1)
        dki = tmem_ld("32x32b", B_T, lane_base + tmem.cols("dki"))
        for t in cutlass.range_constexpr(B_T):
            dk_n[t] = dk_n[t] + dki[t] * cute.math.rcp(eg[t], approx=True, ftz=True)

        # ---- dK decay part: -2^g term ---------------------------------------------------------------
        dqh = dqk.wait()
        dkd = tmem_ld("32x32b", B_T, lane_base + tmem.cols("dkd"))
        for t in cutlass.range_constexpr(B_T):
            dg[t] = _mul_rn(-eg[t], dkd[t])
            dk_n[t] = _add_rn(dk_n[t], dg[t])
        dqh.release()  # tcgen05.wait::ld, then the parts may be overwritten

        # ---- dA diagonal ---------------------------------------------------------------------------
        ddh = da_diag.wait()
        diag = cute.make_rmem_tensor((B_T,), Float32)
        for t in cutlass.range_constexpr(B_T):
            diag[t] = lds_f32(params.s_diag + 4 * (ddh.index * B_T + t)) * scale
        ddh.release(skip_fence="before", reason=BEFORE_FENCE_COVERED)

        # ---- dGate finalize --------------------------------------------------------------------------
        qf = tmem_ld("32x32b", B_T // 2, qraw_col, Int32)
        kf = tmem_ld("32x32b", B_T // 2, kraw_col, Int32)
        for t in cutlass.range_constexpr(B_T):
            q_v = _half(qf[t // 2], t % 2, io)
            k_v = _half(kf[t // 2], t % 2, io)
            dq_n[t] = dq_n[t] + diag[t] * k_v
            dk_n[t] = dk_n[t] + diag[t] * q_v
            dg[t] = _dgate_term(q_v, k_v, dq_n[t], dk_n[t], dg[t], False)
        dg[B_T - 1] = dg[B_T - 1] + ((last_acc[0] + last_acc[1]) + (last_acc[2] + last_acc[3]))
        qh.release()

        # ---- stage dQ / dK ---------------------------------------------------------------------------
        qsh = dq.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
        ksh = dk.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
        for t in cutlass.range_constexpr(B_T):
            out = 2 * (f16_seg * (B_T * 64) + t * 64 + swz128(t, f16_dim))
            _sts_16(params.s_dq + out, dq_n[t], io)
            _sts_16(params.s_dk + out, dk_n[t], io)
        qsh.commit(skip_fence="before", reason=BEFORE_FENCE_COVERED)
        ksh.commit(skip_fence="before", reason=BEFORE_FENCE_COVERED)

        # ---- dGate: last-chunk state term, reverse cumsum, staging -------------------------------
        if has_dstate:  # noqa: SIM102 (upstream nests the dynamic tests)
            if chunk >= self.first_state_chunk:
                dg[B_T - 1] = dg[B_T - 1] + dgate_last
        suffix = Float32(0.0)
        for rt in cutlass.range_constexpr(B_T):
            t = B_T - 1 - rt
            suffix = suffix + dg[t]
            dg[t] = suffix
        gh = dgate.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
        for t in cutlass.range_constexpr(B_T):
            if cutlass.const_expr(self.gate == "fp32"):
                sts_f32(
                    params.s_dg + 4 * (f32_seg * (B_T * 32) + t * 32 + swz128(t, f32_dim, 4)),
                    dg[t],
                )
            else:
                _sts_16(
                    params.s_dg + 2 * (f16_seg * (B_T * 64) + t * 64 + swz128(t, f16_dim)),
                    dg[t],
                    self.gate_dtype,
                )
        gh.commit(skip_fence="before", reason=BEFORE_FENCE_COVERED)
        return raw, qk_raw, state_in, dstate, dqk, da_diag, dq, dk, dgate, n_dkr

    @cute.jit
    def cg2_chunk_split(
        self,
        params,
        raw,
        qk_raw,
        state_in,
        dstate,
        dqk,
        da_diag,
        dq,
        dk,
        dgate,
        tsig,
        tmem,
        lane,
        warp,
        lane_base,
        n,
        n_dkr,
        rev,
        chunk,
    ):
        """One chunk with two key channels per lane quad (d_k = 64): values ``k = rep * 4 + cs * 2
        + n`` of a lane are channel ``cs`` of the lane's pair, token ``quad * 2 + rep * 8 + n``
        (the 16x256b TMEM fragment)."""
        io = self.io_dtype
        d_v = self.d_v
        scale = params.scale
        has_dstate = self.has_dstate(rev)
        quad = lane & 3
        quad_base = lane - quad
        ch_lo = warp * 16 + (lane >> 2)
        channels = (ch_lo, ch_lo + 8)
        stage = n & 1
        # The exchange is complete.
        self.sig_wait("kdi", tsig + KDI + stage, (n >> 1) & 1, fence=False)

        rh = raw.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
        s_ex = params.s_g + rh.index * (self.d_k * B_T * 4)
        eg = cute.make_rmem_tensor((8,), Float32)
        for cs in cutlass.range_constexpr(2):
            seg = channels[cs] >> 5
            dim = channels[cs] & 31
            for rep in cutlass.range_constexpr(2):
                for m in cutlass.range_constexpr(2):
                    token = quad * 2 + (rep * 8 + m)
                    eg[rep * 4 + cs * 2 + m] = lds_f32(
                        s_ex + 4 * (seg * (B_T * 32) + token * 32 + swz128(token ^ seg, dim, 4))
                    )
        rh.release()
        qh = qk_raw.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
        qraw_col = lane_base + tmem.cols("qraw") + qh.index * (B_T // 2)
        kraw_col = lane_base + tmem.cols("kraw") + qh.index * (B_T // 2)

        # ---- dGate last: sum_v dS[v, c] S[c, v] ------------------------------------------------
        last_frag = cute.make_rmem_tensor((2,), Float32)
        for cs in cutlass.range_constexpr(2):
            last_frag[cs] = Float32(0.0)
        sih = state_in.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
        if has_dstate:
            dsh = dstate.wait()
            svec = tmem_ld(
                "16x256b",
                d_v // 16,
                lane_base + tmem.cols("state_in") + sih.index * (d_v // 2),
                Int32,
            )
            hacc = cute.make_rmem_tensor((8,), Float32)
            for i in cutlass.range_constexpr(8):
                hacc[i] = opaque_f32_zero()
            for cs in cutlass.range_constexpr(2):
                for rep in cutlass.range_constexpr(d_v // 16):
                    for m in cutlass.range_constexpr(2):
                        word_col = quad * 2 + (rep * 8 + m)
                        v0 = word_col * 2
                        word = svec[rep * 4 + cs * 2 + m]
                        h0 = _lds_wide(
                            params.s_dstate + 2 * (v0 * 64 + swz128(v0, channels[cs])), io
                        )
                        h1 = _lds_wide(
                            params.s_dstate + 2 * ((v0 + 1) * 64 + swz128(v0 + 1, channels[cs])),
                            io,
                        )
                        b = cs * 4 + (rep % 2) * 2 + m
                        hacc[b] = hacc[b] + h0 * _half(word, 0, io)
                        hacc[b] = hacc[b] + h1 * _half(word, 1, io)
            for cs in cutlass.range_constexpr(2):
                part = (hacc[cs * 4 + 0] + hacc[cs * 4 + 1]) + (
                    hacc[cs * 4 + 2] + hacc[cs * 4 + 3]
                )
                part = part + shfl_bfly(part, 1)
                part = part + shfl_bfly(part, 2)
                last_frag[cs] = part
            dsh.release()
        sih.release(skip_fence="before", reason=BEFORE_FENCE_COVERED)

        dq_f = cute.make_rmem_tensor((8,), Float32)
        dk_f = cute.make_rmem_tensor((8,), Float32)
        dg = cute.make_rmem_tensor((8,), Float32)
        kdot = cute.make_rmem_tensor((4,), Float32)
        for i in cutlass.range_constexpr(4):
            kdot[i] = opaque_f32_zero()
        for i in cutlass.range_constexpr(8):
            dk_f[i] = Float32(0.0)

        # ---- dK restore part, dGate last K-dot ------------------------------------------------------
        if has_dstate:
            self.sig_wait("dkr", tsig + DKR, n_dkr & 1)
            n_dkr += 1
            dkr = tmem_ld("16x256b", B_T // 8, lane_base + tmem.cols("dkr"))
            kr = tmem_ld("16x256b", B_T // 16, kraw_col, Int32)
            for cs in cutlass.range_constexpr(2):
                for rep in cutlass.range_constexpr(2):
                    src = quad_base + (quad >> 1) + rep * 2
                    w0 = shfl_idx(kr[cs * 2], src)
                    w1 = shfl_idx(kr[cs * 2 + 1], src)
                    k_word = w1 if (quad & 1) == 1 else w0
                    for m in cutlass.range_constexpr(2):
                        k = rep * 4 + cs * 2 + m
                        dk_hat = cute.math.rcp(eg[k], approx=True, ftz=True) * dkr[k]
                        dk_f[k] = dk_hat
                        kdot[cs * 2 + m] = kdot[cs * 2 + m] + _half(k_word, m, io) * dk_hat

        # ---- dQ, dK inv part, dK decay part -------------------------------------------------------
        self.sig_wait("dq", tsig + DQ, n & 1)
        dqv = tmem_ld("16x256b", B_T // 8, lane_base + tmem.cols("dq"))
        for k2 in cutlass.range_constexpr(4):
            k = 2 * k2
            lo, hi = fmul2(eg[k], eg[k + 1], scale, scale)
            dq_f[k], dq_f[k + 1] = fmul2(lo, hi, dqv[k], dqv[k + 1])
        self.sig_wait("dki", tsig + DKI, n & 1)
        dki = tmem_ld("16x256b", B_T // 8, lane_base + tmem.cols("dki"))
        for k in cutlass.range_constexpr(8):
            dk_f[k] = dk_f[k] + dki[k] * cute.math.rcp(eg[k], approx=True, ftz=True)
        dqh = dqk.wait()
        dkd = tmem_ld("16x256b", B_T // 8, lane_base + tmem.cols("dkd"))
        for k in cutlass.range_constexpr(8):
            dg[k] = _mul_rn(-eg[k], dkd[k])
            dk_f[k] = _add_rn(dk_f[k], dg[k])
        dqh.release()

        # ---- dA diagonal --------------------------------------------------------------------------
        ddh = da_diag.wait()
        diag = cute.make_rmem_tensor((4,), Float32)
        for rep in cutlass.range_constexpr(2):
            for m in cutlass.range_constexpr(2):
                token = quad * 2 + (rep * 8 + m)
                diag[rep * 2 + m] = lds_f32(params.s_diag + 4 * (ddh.index * B_T + token)) * scale
        ddh.release(skip_fence="before", reason=BEFORE_FENCE_COVERED)

        # ---- dGate finalize --------------------------------------------------------------------------
        qf = tmem_ld("16x256b", B_T // 16, qraw_col, Int32)
        kf = tmem_ld("16x256b", B_T // 16, kraw_col, Int32)
        for cs in cutlass.range_constexpr(2):
            for rep in cutlass.range_constexpr(2):
                src = quad_base + (quad >> 1) + rep * 2
                q0 = shfl_idx(qf[cs * 2], src)
                q1 = shfl_idx(qf[cs * 2 + 1], src)
                k0 = shfl_idx(kf[cs * 2], src)
                k1 = shfl_idx(kf[cs * 2 + 1], src)
                q_word = q1 if (quad & 1) == 1 else q0
                k_word = k1 if (quad & 1) == 1 else k0
                for m in cutlass.range_constexpr(2):
                    k = rep * 4 + cs * 2 + m
                    q_v = _half(q_word, m, io)
                    k_v = _half(k_word, m, io)
                    dq_f[k] = dq_f[k] + diag[rep * 2 + m] * k_v
                    dk_f[k] = dk_f[k] + diag[rep * 2 + m] * q_v
                    dg[k] = _dgate_term(q_v, k_v, dq_f[k], dk_f[k], dg[k], True)
        for cs in cutlass.range_constexpr(2):
            kd = kdot[cs * 2] + kdot[cs * 2 + 1]
            kd = kd + shfl_bfly(kd, 1)
            kd = kd + shfl_bfly(kd, 2)
            if quad == 3:
                dg[4 + cs * 2 + 1] = dg[4 + cs * 2 + 1] + kd
        qh.release()

        # ---- stage dQ / dK ---------------------------------------------------------------------------
        qsh = dq.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
        ksh = dk.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
        for cs in cutlass.range_constexpr(2):
            for rep in cutlass.range_constexpr(2):
                for m in cutlass.range_constexpr(2):
                    k = rep * 4 + cs * 2 + m
                    token = quad * 2 + (rep * 8 + m)
                    out = 2 * (token * 64 + swz128(token, channels[cs]))
                    _sts_16(params.s_dq + out, dq_f[k], io)
                    _sts_16(params.s_dk + out, dk_f[k], io)
        qsh.commit(skip_fence="before", reason=BEFORE_FENCE_COVERED)
        ksh.commit(skip_fence="before", reason=BEFORE_FENCE_COVERED)

        # ---- dGate: last-chunk state term, reverse cumsum over the quad, staging -----------------
        if has_dstate:  # noqa: SIM102 (upstream nests the dynamic tests)
            if chunk >= self.first_state_chunk:  # noqa: SIM102
                if quad == 3:
                    for cs in cutlass.range_constexpr(2):
                        dg[4 + cs * 2 + 1] = dg[4 + cs * 2 + 1] + last_frag[cs]
        for cs in cutlass.range_constexpr(2):
            pair_sum = cute.make_rmem_tensor((2,), Float32)
            higher = cute.make_rmem_tensor((2,), Float32)
            block_total = cute.make_rmem_tensor((2,), Float32)
            for rep in cutlass.range_constexpr(2):
                pair_sum[rep] = dg[rep * 4 + cs * 2] + dg[rep * 4 + cs * 2 + 1]
                quad_1 = shfl_idx(pair_sum[rep], quad_base + 1)
                quad_2 = shfl_idx(pair_sum[rep], quad_base + 2)
                quad_3 = shfl_idx(pair_sum[rep], quad_base + 3)
                after = quad_1 if quad < 1 else Float32(0.0)
                after = after + (quad_2 if quad < 2 else Float32(0.0))
                after = after + (quad_3 if quad < 3 else Float32(0.0))
                higher[rep] = after
                total = pair_sum[rep] + shfl_bfly(pair_sum[rep], 1)
                total = total + shfl_bfly(total, 2)
                block_total[rep] = total
            tail = dg[4 + cs * 2 + 1] + higher[1]
            dg[4 + cs * 2 + 1] = tail
            dg[4 + cs * 2] = dg[4 + cs * 2] + tail
            head = dg[cs * 2 + 1] + higher[0]
            dg[cs * 2 + 1] = head + block_total[1]
            dg[cs * 2] = (dg[cs * 2] + head) + block_total[1]
        gh = dgate.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
        for cs in cutlass.range_constexpr(2):
            for rep in cutlass.range_constexpr(2):
                for m in cutlass.range_constexpr(2):
                    k = rep * 4 + cs * 2 + m
                    token = quad * 2 + (rep * 8 + m)
                    if cutlass.const_expr(self.gate == "fp32"):
                        seg = channels[cs] >> 5
                        dim = channels[cs] & 31
                        sts_f32(
                            params.s_dg
                            + 4 * (seg * (B_T * 32) + token * 32 + swz128(token, dim, 4)),
                            dg[k],
                        )
                    else:
                        _sts_16(
                            params.s_dg + 2 * (token * 64 + swz128(token, channels[cs])),
                            dg[k],
                            self.gate_dtype,
                        )
        gh.commit(skip_fence="before", reason=BEFORE_FENCE_COVERED)
        return raw, qk_raw, state_in, dstate, dqk, da_diag, dq, dk, dgate, n_dkr

    # ---- register MMA (warp 12): KK, T, dM, the dbeta M-term ------------------------------

    @role(warps=12, regs=56)
    @cute.jit
    def regmma(
        self,
        params: KdaBpropParams,
        *,
        beta: Consumer,
        tinv: Producer,
        dm: Producer,
        sched: Consumer,
        work: WorkCursor,
        tsig: Barriers,
        me: Local,
    ):
        lane = me.lane
        io = self.io_dtype
        frag = _fragment_geometry(lane)
        row_lo = lane >> 2
        row_hi = row_lo + 8
        strict = _tril_mask(lane, strict=True)
        base = Int32(0)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                _b, _h, _s, _len, write_start, _we, compute_end = self.item(params, tile)
                n_chunks = compute_end - write_start
                for rev in cutlass.range(n_chunks, unroll=1):
                    n = base + rev
                    stage = n & 1
                    th = tinv.acquire()
                    # ---- KK = K decay K inv^T ------------------------------------------------------
                    self.sig_wait("kdi", tsig + KDI + stage, (n >> 1) & 1, fence=False)
                    op = stage * (self.d_k * B_T * 2)
                    kk = _mma_tile(params.s_kdecay + op, params.s_kinv + op, frag, self.d_k, io)
                    # ---- L = beta strict(KK), T = (I + L)^-1 ---------------------------------------
                    bh = beta.wait()
                    s_beta = params.s_beta + bh.index * (B_T * 4)
                    beta_lo = lds_f32(s_beta + 4 * row_lo)
                    beta_hi = lds_f32(s_beta + 4 * row_hi)
                    bh.release()
                    l_regs = cute.make_rmem_tensor((8,), Float32)
                    for i in cutlass.range_constexpr(8):
                        b = beta_lo if cutlass.const_expr(i % 4 < 2) else beta_hi
                        lower = kk[i] if (strict >> i) & 1 else Float32(0.0)
                        l_regs[i] = lower * b
                    t = invert_unit_lower_16x16_fragments(l_regs, lane, io)
                    inter = params.s_inter + th.index * (INTER_TILES * TILE)
                    _store_tile(inter + 1 * TILE, frag, t, io)
                    th.commit()
                    # ---- dM = dY U^T, beta strict(dM), -beta strict(dM) ----------------------------
                    mh = dm.acquire()
                    self.sig_wait("dy_smem", tsig + DY_SMEM, n & 1)
                    dmv = _mma_tile(params.s_dy, params.s_u, frag, self.d_v, io)
                    dms = cute.make_rmem_tensor((8,), Float32)
                    for i in cutlass.range_constexpr(8):
                        b = beta_lo if cutlass.const_expr(i % 4 < 2) else beta_hi
                        dms[i] = dmv[i] * b if (strict >> i) & 1 else Float32(0.0)
                    inter = params.s_inter + mh.index * (INTER_TILES * TILE)
                    _store_tile(inter + 3 * TILE, frag, dms, io)
                    neg = cute.make_rmem_tensor((8,), Float32)
                    for i in cutlass.range_constexpr(8):
                        neg[i] = -dms[i]
                    _store_tile(inter + 4 * TILE, frag, neg, io)
                    mh.commit()
                    # ---- M-term: -sum strict(dM . KK) per row ----------------------------------------
                    bsum_lo = Float32(0.0)
                    bsum_hi = Float32(0.0)
                    for i in cutlass.range_constexpr(8):
                        e = dmv[i] * kk[i] if (strict >> i) & 1 else Float32(0.0)
                        if cutlass.const_expr(i % 4 < 2):
                            bsum_lo = bsum_lo + e
                        else:
                            bsum_hi = bsum_hi + e
                    bsum_lo = bsum_lo + shfl_bfly(bsum_lo, 1)
                    bsum_lo = bsum_lo + shfl_bfly(bsum_lo, 2)
                    bsum_hi = bsum_hi + shfl_bfly(bsum_hi, 1)
                    bsum_hi = bsum_hi + shfl_bfly(bsum_hi, 2)
                    if (lane & 3) == 0:
                        sts_f32(params.s_betam + 4 * row_lo, -bsum_lo)
                        sts_f32(params.s_betam + 4 * row_hi, -bsum_hi)
                    cute.arch.fence_view_async_shared()
                    before_sync_and_arrive(tsig + DBETA_M)
                base += n_chunks
                sched, tile = self.next_tile(sched)
            work.advance()
        return beta, tinv, dm, sched, work

    # ---- epilogue (warp 15): A, dA, the one-behind dQ / dK / dGate / dV stores -------------

    @role(warps=15, regs=56)
    @cute.jit
    def epilogue(
        self,
        params: KdaBpropParams,
        *,
        decay: Consumer,
        do: Consumer,
        a: Producer,
        da: Producer,
        da_diag: Producer,
        dq: Consumer,
        dk: Consumer,
        dgate: Consumer,
        dv: Consumer,
        sched: Consumer,
        work: WorkCursor,
        tsig: Barriers,
        me: Local,
    ):
        lane = me.lane
        io = self.io_dtype
        frag = _fragment_geometry(lane)
        incl = _tril_mask(lane, strict=False)
        strict = _tril_mask(lane, strict=True)
        row_lo = lane >> 2
        row_hi = row_lo + 8
        base = Int32(0)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                _b, head, start, seqlen, write_start, write_end, compute_end = self.item(
                    params, tile
                )
                n_chunks = compute_end - write_start
                stores = (params.dq_store, params.dk_store, params.dg_store, params.dv_store)
                descs = sequence_descriptors(stores, start + seqlen, head, B_T, n_chunks > 0)
                pend_start = Int32(0)
                pend_writes = cutlass.Boolean(False)
                for rev in cutlass.range(n_chunks, unroll=1):
                    chunk = compute_end - 1 - rev
                    n = base + rev
                    # ---- A = lower(Q decay K inv^T) ------------------------------------------
                    ah = a.acquire()
                    dh = decay.wait()
                    op = dh.index * (self.d_k * B_T * 2)
                    acc = _mma_tile(params.s_qdecay + op, params.s_kinv + op, frag, self.d_k, io)
                    for i in cutlass.range_constexpr(8):
                        acc[i] = acc[i] if (incl >> i) & 1 else Float32(0.0)
                    inter = params.s_inter + ah.index * (INTER_TILES * TILE)
                    _store_tile(inter, frag, acc, io)
                    ah.commit()
                    dh.release()
                    # ---- dA = lower(dO U^T) ----------------------------------------------------
                    self.sig_wait("u_smem", tsig + U_SMEM, n & 1)
                    oh = do.wait()
                    s_do = params.s_do + oh.index * (self.d_v * B_T * 2)
                    dav = _mma_tile(s_do, params.s_u, frag, self.d_v, io)
                    oh.release()
                    # ---- dA diagonal -> cg2 ----------------------------------------------------
                    gh = da_diag.acquire()
                    for i in cutlass.range_constexpr(8):
                        r = row_hi if cutlass.const_expr(i % 4 >= 2) else row_lo
                        c = (i // 4) * 8 + 2 * (lane & 3) + (i % 2)
                        if r == c:
                            sts_f32(params.s_diag + 4 * (gh.index * B_T + r), dav[i])
                    gh.commit()
                    for i in cutlass.range_constexpr(8):
                        dav[i] = dav[i] if (strict >> i) & 1 else Float32(0.0)
                    dah = da.acquire()
                    inter = params.s_inter + dah.index * (INTER_TILES * TILE)
                    _store_tile(inter + 2 * TILE, frag, dav, io)
                    dah.commit()
                    if rev > 0:
                        dq, dk, dgate, dv = self.store_chunk(
                            params, dq, dk, dgate, dv, pend_writes, pend_start, start, seqlen,
                            head, descs,
                        )  # fmt: skip
                    pend_start = chunk * B_T
                    pend_writes = chunk < write_end
                if n_chunks > 0:  # the last chunk's stores
                    dq, dk, dgate, dv = self.store_chunk(
                        params, dq, dk, dgate, dv, pend_writes, pend_start, start, seqlen,
                        head, descs,
                    )  # fmt: skip
                base += n_chunks
                sched, tile = self.next_tile(sched)
            work.advance()
        bulk_wait_group(0)
        return decay, do, a, da, da_diag, dq, dk, dgate, dv, sched, work

    @cute.jit
    def store_chunk(
        self, params, dq, dk, dgate, dv, writes, chunk_start, start, seqlen, head, descs
    ):
        """TMA-store one staged chunk of dQ, dK, dGate and dV through the item's ``descs`` (rows
        past the sequence end are clipped); warm-up chunks release their stages unstored."""
        rows = (start + chunk_start, cutlass.min(seqlen - chunk_start, B_T))
        dq_desc, dk_desc, dg_desc, dv_desc = descs
        qh = dq.wait()
        if writes:
            self._tma_store(
                params.dq_store, params.s_dq, rows, head, self.d_k, params.box_io, dq_desc
            )
        kh = dk.wait()
        if writes:
            self._tma_store(
                params.dk_store, params.s_dk, rows, head, self.d_k, params.box_io, dk_desc
            )
        gh = dgate.wait()
        if writes:
            self._tma_store(
                params.dg_store,
                params.s_dg,
                rows,
                head,
                self.d_k,
                params.box_g,
                dg_desc,
                dtype=self.gate_dtype,
            )
        vh = dv.wait()
        if writes:
            self._tma_store(
                params.dv_store,
                params.s_dv + vh.index * (B_T * self.d_v * 2),
                rows,
                head,
                self.d_v,
                params.box_io,
                dv_desc,
            )
        bulk_wait_group(3, read=True)
        qh.release()
        bulk_wait_group(2, read=True)
        kh.release()
        bulk_wait_group(1, read=True)
        gh.release()
        bulk_wait_group(0, read=True)
        vh.release()
        return dq, dk, dgate, dv

    @cute.jit
    def _tma_store(
        self,
        store,
        addr,
        rows,
        head,
        dim: cutlass.Constexpr,
        layout,
        desc,
        dtype: cutlass.Constexpr = None,
    ):
        dtype = self.io_dtype if cutlass.const_expr(dtype is None) else dtype
        cols = 128 // (dtype.width // 8)
        token, valid = rows
        for b in cutlass.range_constexpr(dim // cols):
            src = cute.make_tensor(
                cute.make_ptr(
                    dtype, addr + b * (B_T * 128), cute.AddressSpace.smem, assumed_align=1024
                ),
                layout,
            )
            ragged_store(store, src, token, valid, head, b, B_T, cols, desc)
        bulk_commit_group()

    # ---- tcgen05 MMA (warp 13) ------------------------------------------------------------

    @role(warps=13, regs=56)
    @cute.jit
    def mma(
        self,
        params: KdaBpropParams,
        *,
        state: Consumer,
        decay: Consumer,
        state_in: Consumer,
        a: Consumer,
        tinv: Consumer,
        da: Consumer,
        dm: Consumer,
        dstate: Consumer,
        dqk: Producer,
        sched: Consumer,
        work: WorkCursor,
        tsig: Barriers,
        tmem: Tmem,
    ):
        elect = elect_sync()
        d_k, d_v = self.d_k, self.d_v
        t0 = tmem.base.toint()
        col = {
            name: t0 + tmem.cols(name)
            for name in ("dstate", "dstate_in", "state_in", "kstate", "u", "du", "dq", "dkd")
        }
        for name in ("dki", "dkr", "yin", "duin"):
            col[name] = t0 + tmem.cols(name)
        sw128 = SWIZZLE_128B
        d_state0 = smem_desc(params.s_state, 16, 1024, sw128)
        d_kd0 = smem_desc(params.s_kdecay, 16, 1024, sw128)
        d_ki0 = smem_desc(params.s_kinv, 16, 1024, sw128)
        d_do0 = smem_desc(params.s_do, 16, 1024, sw128)
        d_do_t0 = smem_desc(params.s_do, B_T * 128, 1024, sw128)
        d_qd_t0 = smem_desc(params.s_qdecay, B_T * 128, 1024, sw128)
        d_kd_t0 = smem_desc(params.s_kdecay, B_T * 128, 1024, sw128)
        d_ki_t0 = smem_desc(params.s_kinv, B_T * 128, 1024, sw128)
        d_int0 = smem_desc(params.s_inter, 16, 256, SWIZZLE_32B)
        d_dv0 = smem_desc(params.s_dv, 16, 1024, sw128)
        d_dstate_t0 = smem_desc(params.s_dstate, d_v * 128, 1024, sw128)
        d_u0 = smem_desc(params.s_u, 16, 1024, sw128)
        do_seg = (B_T * d_v * 2) >> 4
        op_seg = (B_T * d_k * 2) >> 4
        inter_seg = (INTER_TILES * TILE) >> 4
        slot = TILE >> 4
        do_full = self.barrier_ptr("do")
        dv_full = self.barrier_ptr("dv")
        n_dsin = Int32(0)  # completed DSTATE_IN waits
        n_items = Int32(0)  # completed DSTATE0 waits
        base = Int32(0)
        while work.valid:
            unit, _, _ = work.item
            total = params.work_count[0]
            tile = Int32(unit)
            while tile < total:
                _b, _h, _s, _len, write_start, _we, compute_end = self.item(params, tile)
                n_chunks = compute_end - write_start
                for rev in cutlass.range(n_chunks, unroll=1):
                    n = base + rev
                    chunk = compute_end - 1 - rev
                    stage = n & 1  # every two-stage ring here is chunk-indexed
                    sphase = (n >> 1) & 1
                    has_dstate = self.has_dstate(rev)
                    op_off = stage * op_seg
                    d_do_t = d_do_t0 + stage * do_seg
                    d_qd_t = d_qd_t0 + op_off
                    d_kd_t = d_kd_t0 + op_off
                    d_ki_t = d_ki_t0 + op_off
                    d_int = d_int0 + stage * inter_seg

                    # ---- K S = state(S) @ K decay^T --------------------------------------------
                    self.sig_wait("kdi", tsig + KDI + stage, sphase, fence=False)
                    if chunk >= self.first_state_chunk:
                        sh = state.wait()
                        mma_ss_op(
                            self.g_kstate,
                            desc_opaque(d_state0),
                            d_kd0 + op_off,
                            col["kstate"],
                            False,
                            elect,
                        )
                        umma_commit(tsig + STATE_K, elect)
                        sh.release()

                    # ---- dQ inter = state(T) @ dO^T ---------------------------------------------
                    qh = dqk.acquire(skip_fence="after", reason=AFTER_FENCE_COVERED)
                    sih = state_in.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
                    self.sig_wait(
                        "do_full", do_full + stage, sphase
                    )  # dO landed (see ``self.do``)
                    state_col = col["state_in"] + sih.index * (d_v // 2)
                    if chunk >= self.first_state_chunk:
                        mma_ts_all_op(
                            self.g_state_t,
                            state_col,
                            d_do0 + stage * do_seg,
                            col["dq"],
                            False,
                            elect,
                        )

                    # ---- dU = dS(T) @ K inv + dO^T(S) @ A -------------------------------------
                    if has_dstate:
                        self.sig_wait("dstate_in", tsig + DSTATE_IN, n_dsin & 1)
                        n_dsin += 1
                        mma_ts_all_op(
                            self.g_kstate,
                            col["dstate_in"],
                            d_ki0 + op_off,
                            col["du"],
                            False,
                            elect,
                        )
                    ah = a.wait()
                    mma_ss_op(self.g_do_a, d_do_t, d_int, col["du"], has_dstate, elect)
                    umma_commit(tsig + DU_ACC, elect)
                    ah.release()

                    # ---- dS Q-term += dO^T(S) @ Q decay -----------------------------------------
                    dh = decay.wait()
                    mma_ss_op(self.g_do_qd, d_do_t, d_qd_t, col["dstate"], has_dstate, elect)

                    # ---- U = Y(T) @ T^-1, dY = dU(T) @ T^-T ---------------------------------------
                    th = tinv.wait(skip_fence="after", reason=AFTER_FENCE_COVERED)
                    self.sig_wait("y_in", tsig + Y_IN, n & 1)
                    mma_ts_all_op(self.g_t, col["yin"], d_int + slot, col["u"], False, elect)
                    umma_commit(tsig + U_ACC, elect)
                    self.sig_wait("du_in", tsig + DU_IN, n & 1)
                    mma_ts_all_op(
                        self.g_t_trans, col["duin"], d_int + slot, col["kstate"], False, elect
                    )
                    umma_commit(tsig + DY_ACC, elect)
                    th.release()

                    # ---- dK restore part = dS(S) @ U^T ------------------------------------------
                    self.sig_wait("u_smem", tsig + U_SMEM, n & 1, fence=False)
                    if has_dstate:
                        dsh = dstate.wait()
                        mma_ss_op(
                            self.g_dstate_u,
                            desc_opaque(d_dstate_t0),
                            desc_opaque(d_u0),
                            col["dkr"],
                            False,
                            elect,
                        )
                        umma_commit(tsig + DKR, elect)
                        dsh.release()

                    # ---- dS K-term += -beta dY(T) @ K decay ---------------------------------------
                    self.sig_wait("nbdy_in", tsig + NBDY_IN, n & 1)
                    mma_ts_all_op(self.g_dy_kd, col["yin"], d_kd_t, col["dstate"], True, elect)
                    umma_commit(tsig + DSTATE_ACC, elect)

                    # ---- dK inv part = Q decay^T(S) @ dA, dQ attn += K inv^T(S) @ dA^T ----------
                    dah = da.wait()
                    mma_ss_op(self.g_kt_dt, d_qd_t, d_int + 2 * slot, col["dki"], False, elect)
                    if chunk >= self.first_state_chunk:
                        mma_ss_op(self.g_kt_d, d_ki_t, d_int + 2 * slot, col["dq"], True, elect)
                    if chunk < self.first_state_chunk:
                        mma_ss_op(self.g_kt_d, d_ki_t, d_int + 2 * slot, col["dq"], False, elect)
                    umma_commit(tsig + DQ, elect)
                    dah.release()

                    # ---- dK decay part = state(T) @ (beta dY)^T -----------------------------------
                    self.sig_wait(
                        "dv_full", dv_full + stage, sphase
                    )  # dV staged (see ``self.dv``)
                    if chunk >= self.first_state_chunk:
                        mma_ts_all_op(
                            self.g_state_t,
                            state_col,
                            d_dv0 + stage * do_seg,
                            col["dkd"],
                            False,
                            elect,
                        )
                    sih.release()

                    # ---- dK inv += K decay^T(S) @ -dM, dK decay += K inv^T(S) @ dM^T -----------
                    mh = dm.wait()
                    mma_ss_op(self.g_kt_dt, d_kd_t, d_int + 4 * slot, col["dki"], True, elect)
                    umma_commit(tsig + DKI, elect)
                    if chunk >= self.first_state_chunk:
                        mma_ss_op(self.g_kt_d, d_ki_t, d_int + 3 * slot, col["dkd"], True, elect)
                    if chunk < self.first_state_chunk:
                        mma_ss_op(self.g_kt_d, d_ki_t, d_int + 3 * slot, col["dkd"], False, elect)
                    qh.commit()
                    mh.release()
                    dh.release()
                if n_chunks > 0:  # cg1 stored this item's dS before the next item rewrites it
                    self.sig_wait("dstate0", tsig + DSTATE0, n_items & 1, fence=False)
                    n_items += 1
                base += n_chunks
                sched, tile = self.next_tile(sched)
            work.advance()
        return state, decay, state_in, a, tinv, da, dm, dstate, dqk, sched, work

    # ---- kernel and entry point ---------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        q_load,
        k_load,
        v_load,
        g_load,
        do_load,
        dq_store,
        dk_store,
        dv_store,
        dg_store,
        tma_ckpt,
        mCkpt,
        box_io,
        box_g,
        box_ckpt,
        beta,
        dbeta,
        dstate_in,
        dstate0,
        work_items,
        work_count,
        scheduler,
        q_ratio,
        k_ratio,
        v_ratio,
        scale,
        layouts,
        trace_buffer,
        bound,
    ):
        self.begin_roles(work=bound, layouts=layouts)
        addr = {name: self.buffer(name).iterator.toint() for name in self.buf}
        params = KdaBpropParams(
            q_load=q_load,
            k_load=k_load,
            v_load=v_load,
            g_load=g_load,
            do_load=do_load,
            dq_store=dq_store,
            dk_store=dk_store,
            dv_store=dv_store,
            dg_store=dg_store,
            tma_ckpt=tma_ckpt,
            mCkpt=mCkpt,
            box_io=box_io,
            box_g=box_g,
            box_ckpt=box_ckpt,
            beta=beta,
            dbeta=dbeta,
            dstate_in=dstate_in,
            dstate0=dstate0,
            work_items=work_items,
            work_count=work_count,
            scheduler=scheduler,
            q_ratio=q_ratio,
            k_ratio=k_ratio,
            v_ratio=v_ratio,
            scale=scale,
            s_state=addr["State"],
            s_dstate=addr["Dstate"],
            s_kdecay=addr["Kdecay"],
            s_kinv=addr["Kinv"],
            s_qdecay=addr["Qdecay"],
            s_scale=addr["Scale"],
            s_inter=addr["Inter"],
            s_do=addr["Do"],
            s_dv=addr["Dv"],
            s_q=addr["Q"],
            s_k=addr["K"],
            s_v=addr["V"],
            s_g=addr["Gate"],
            s_dy=addr["Dy"],
            s_u=addr["U"],
            s_dq=addr["Dq"],
            s_dk=addr["Dk"],
            s_dg=addr["Dgate"],
            s_beta=addr["Beta"],
            s_diag=addr["Diag"],
            s_red=addr["Red"],
            s_betam=addr["BetaM"],
        )
        self.run_roles(params, trace_buffer=trace_buffer)

    @cute.jit
    def launch(
        self,
        q: cute.Tensor,  # (tokens, q heads, d_k)
        k: cute.Tensor,  # (tokens, k heads, d_k)
        v: cute.Tensor,  # (tokens, v heads, d_v)
        do: cute.Tensor,  # (tokens, heads, d_v)
        gate: cute.Tensor,  # (tokens, heads, d_k) natural-log decay
        beta: cute.Tensor,  # (tokens, heads)
        checkpoints: cute.Tensor,  # (rows, heads, d_v, d_k), closed-form rows
        dq: cute.Tensor,  # (tokens, heads, d_k): per output head
        dk: cute.Tensor,
        dv: cute.Tensor,  # (tokens, heads, d_v)
        dgate: cute.Tensor,  # like gate
        dbeta: cute.Tensor,  # like beta
        dstate_in: cute.Tensor | None,  # (rows, heads, d_v, d_k) fp32
        dstate0: cute.Tensor | None,
        work_items: cute.Tensor,  # (rows, WORK_ITEM_FIELDS) int32
        work_count: cute.Tensor,  # (1,) int32 rows to run
        scheduler: cute.Tensor,  # (>= 1,) int32, zero
        tma_slots: cute.Tensor,  # (>= self.tma_slot_rows(heads), 16) int64
        scale: Float32,
        stream: cuda.CUstream,
        trace_buffer: cute.Tensor | None = None,
    ):
        """Launch on ``num_ctas`` CTAs (PDL); callable from another ``@cute.jit`` host such as
        a launch bundle, after the kernels that wrote the checkpoints, ``work_items`` and
        ``work_count`` and zeroed ``scheduler``."""
        io, gdt = self.io_dtype, self.gate_dtype
        d_k, d_v = self.d_k, self.d_v
        heads = Int32(gate.shape[1])
        box_io = cute.select(
            sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (B_T, 64), 1), mode=[0, 1]
        )
        g_cols = 128 // (gdt.width // 8)
        box_g = cute.select(
            sm100_utils.make_smem_layout_epi(gdt, LayoutEnum.ROW_MAJOR, (B_T, g_cols), 1),
            mode=[0, 1],
        )
        box_ckpt = cute.select(
            sm100_utils.make_smem_layout_epi(io, LayoutEnum.ROW_MAJOR, (d_v, 64), 1), mode=[0, 1]
        )

        def tdh(t):  # (token, head, dim) -> (token, dim, head)
            return cute.make_tensor(t.iterator, cute.select(t.layout, mode=[0, 2, 1]))

        g2s = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)

        s2g = cpasync.CopyBulkTensorTileS2GOp()

        def ragged(op, t, box, cols, index):  # boxes clipped at each sequence end, keyed by head
            slots = slot_region(tma_slots, index, B_T, heads)
            return make_ragged_tma(op, tdh(t), box, B_T, cols, slots)

        q_load = ragged(g2s, q, box_io, 64, 0)
        k_load = ragged(g2s, k, box_io, 64, 1)
        v_load = ragged(g2s, v, box_io, 64, 2)
        g_load = ragged(g2s, gate, box_g, g_cols, 3)
        do_load = ragged(g2s, do, box_io, 64, 4)
        dq_store = ragged(s2g, dq, box_io, 64, 5)
        dk_store = ragged(s2g, dk, box_io, 64, 6)
        dv_store = ragged(s2g, dv, box_io, 64, 7)
        dg_store = ragged(s2g, dgate, box_g, g_cols, 8)
        ckpt = cute.make_tensor(
            checkpoints.iterator, cute.select(checkpoints.layout, mode=[2, 3, 1, 0])
        )
        tma_ckpt, mCkpt = cpasync.make_tiled_tma_atom(g2s, ckpt, box_ckpt, (d_v, 64))
        layouts = {
            "Beta": cute.make_layout((B_T, BETA_STAGES)),
            "Diag": cute.make_layout((B_T, DIAG_STAGES)),
            "Red": cute.make_layout((4 * B_T,)),
            "BetaM": cute.make_layout((B_T,)),
            "State": cute.make_layout((d_k * d_v, 1)),
            "Dstate": cute.make_layout((d_k * d_v, 1)),
            "Kdecay": cute.make_layout((d_k * B_T, DECAY_STAGES)),
            "Kinv": cute.make_layout((d_k * B_T, DECAY_STAGES)),
            "Qdecay": cute.make_layout((d_k * B_T, DECAY_STAGES)),
            "Scale": cute.make_layout((d_k, DECAY_STAGES)),
            "Inter": cute.make_layout((INTER_TILES * B_T * B_T, INTER_STAGES)),
            "Do": cute.make_layout((d_v * B_T, RAW_STAGES)),
            "Dv": cute.make_layout((d_v * B_T, DV_STAGES)),
            "Q": cute.make_layout((d_k * B_T, RAW_STAGES)),
            "K": cute.make_layout((d_k * B_T, RAW_STAGES)),
            "V": cute.make_layout((d_v * B_T, RAW_STAGES)),
            "Gate": cute.make_layout((d_k * B_T, RAW_STAGES)),
            "Dy": cute.make_layout((d_v * B_T,)),
            "U": cute.make_layout((d_v * B_T,)),
            "Dq": cute.make_layout((d_k * B_T, 1)),
            "Dk": cute.make_layout((d_k * B_T, 1)),
            "Dgate": cute.make_layout((d_k * B_T, 1)),
        }
        bound = self.work.bind((self.num_ctas, 1, 1))
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            q_load,
            k_load,
            v_load,
            g_load,
            do_load,
            dq_store,
            dk_store,
            dv_store,
            dg_store,
            tma_ckpt,
            mCkpt,
            box_io,
            box_g,
            box_ckpt,
            beta,
            dbeta,
            dstate_in,
            dstate0,
            work_items,
            work_count,
            scheduler,
            heads // Int32(q.shape[1]),
            heads // Int32(k.shape[1]),
            heads // Int32(v.shape[1]),
            scale,
            layouts,
            trace_buffer,
            bound,
        ).launch(**self.launch_config(bound), stream=stream, use_pdl=True)

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        do: cute.Tensor,
        gate: cute.Tensor,
        beta: cute.Tensor,
        checkpoints: cute.Tensor,
        dq: cute.Tensor,
        dk: cute.Tensor,
        dv: cute.Tensor,
        dgate: cute.Tensor,
        dbeta: cute.Tensor,
        dstate_in: cute.Tensor | None,
        dstate0: cute.Tensor | None,
        work_items: cute.Tensor,
        work_count: cute.Tensor,
        scheduler: cute.Tensor,
        tma_slots: cute.Tensor,
        scale: Float32,
        trace_buffer: cute.Tensor | None,  # int64 globaltimer profile buffer (tracing only)
        stream: cuda.CUstream,
    ):
        self.launch(
            q,
            k,
            v,
            do,
            gate,
            beta,
            checkpoints,
            dq,
            dk,
            dv,
            dgate,
            dbeta,
            dstate_in,
            dstate0,
            work_items,
            work_count,
            scheduler,
            tma_slots,
            scale,
            stream,
            trace_buffer,
        )


# ---- module helpers -------------------------------------------------------------------------


@cute.jit
def checkpoint_row(batch_start, batch, chunk):
    """Checkpoint row of chunk ``chunk`` of sequence ``batch`` starting at token ``batch_start``
    (``cu[b] // 16 + b + c``; the closed-form padded layout the recompute writes)."""
    return (batch_start >> 4) + batch + chunk


@cute.jit
def _beta_residual(v_word, beta_lo, beta_hi, ks_lo, ks_hi, dtype: cutlass.Constexpr):
    """``(pack(beta (v - ks)), v - ks)`` for one packed V pair (``ks`` None: no state); the
    subtraction and scaling stay fp32, only the MMA operand is rounded."""
    v_lo, v_hi = f16x2_to_f32(v_word, dtype=dtype)
    if cutlass.const_expr(ks_lo is not None):
        v_lo, v_hi = fadd2(v_lo, v_hi, -ks_lo, -ks_hi)
    y_lo, y_hi = fmul2(beta_lo, beta_hi, v_lo, v_hi)
    return fp32_to_fp16(y_lo, y_hi, dtype=dtype), v_lo, v_hi


@cute.jit
def _fragment_geometry(lane):
    """Per-lane ``ldmatrix`` rows/columns of the m16n16k16 A and B operands and the 16 x 16
    32-byte-swizzled ``stmatrix`` offset of an accumulator tile (the upstream lane decode)."""
    b_row = (lane & 7) + (8 if lane >> 4 else 0)
    b_col = 8 if (lane >> 3) & 1 else 0
    a_row = (lane & 7) + (8 if (lane >> 3) & 1 else 0)
    a_col = 8 if lane >> 4 else 0
    t_row = (lane & 7) + (8 if (lane >> 3) & 1 else 0)
    t_col = 8 if lane >> 4 else 0
    return a_row, a_col, b_row, b_col, t_row * B_T + swz32(t_row, t_col)


@cute.jit
def _tril_mask(lane, strict: cutlass.Constexpr[bool]):
    """Bit ``i`` set when accumulator value ``i`` of ``lane`` lies (strictly) below the
    diagonal."""
    mask = Int32(0)
    for i in cutlass.range_constexpr(8):
        r = (lane >> 2) + (8 if i % 4 >= 2 else 0)
        c = (i // 4) * 8 + 2 * (lane & 3) + (i % 2)
        if cutlass.const_expr(strict):
            mask = mask | (Int32(1 << i) if r > c else Int32(0))
        else:
            mask = mask | (Int32(1 << i) if r >= c else Int32(0))
    return mask


@cute.jit
def _mma_tile(a_base, b_base, frag, depth: cutlass.Constexpr[int], dtype: cutlass.Constexpr):
    """``A[16, depth] @ B[16, depth]^T`` with mma.sync from two segment-major 128-byte swizzled
    ``[16 x 64]``-box SMEM tiles; returns the fp32 accumulator fragment."""
    a_row, a_col, b_row, b_col, _t = frag
    acc = cute.make_rmem_tensor((8,), Float32)
    for i in cutlass.range_constexpr(8):
        acc[i] = Float32(0.0)
    for i in cutlass.range_constexpr(depth // 16):
        seg, col = divmod(i * 16, 64)  # a_col, b_col < 16 stay in the 16-column step's box
        af = ldmatrix(a_base + 2 * (seg * (B_T * 64) + a_row * 64 + swz128(a_row, col + a_col)), 4)
        bf = ldmatrix(b_base + 2 * (seg * (B_T * 64) + b_row * 64 + swz128(b_row, col + b_col)), 4)
        mma_step(acc, af, bf, k_step=0, M=16, N=16, dtype=dtype)
    return acc


@cute.jit
def _store_tile(addr, frag, acc, dtype: cutlass.Constexpr):
    """``stmatrix`` one fp32 accumulator fragment as a 16-bit 16 x 16 tile at ``addr``."""
    stmatrix(
        addr + 2 * frag[4],
        [fp32_to_fp16(acc[2 * i], acc[2 * i + 1], dtype=dtype) for i in range(4)],
    )


# ---- standalone entry point (tests) ---------------------------------------------------------


def uncut_work_items(cu_seqlens: torch.Tensor, heads: int):
    """``(work_items, work_count, scheduler)`` of the uncut backward: one item per (sequence,
    head) with any tokens, longest first (the LPT order of ``work_tables``)."""
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    starts = cu_seqlens[:-1].tolist()
    rows = []
    for b, (start, length) in enumerate(zip(starts, lens, strict=True)):
        chunks = -(-length // B_T)
        if chunks:
            for h in range(heads):
                rows.append([b, h, 0, chunks, 0, chunks, start, start + length, b, b])
    rows.sort(key=lambda r: -(r[5] - r[4]))
    device = cu_seqlens.device
    table = torch.tensor(rows or [[0] * WORK_ITEM_FIELDS], dtype=torch.int32, device=device)
    count = torch.tensor([len(rows)], dtype=torch.int32, device=device)
    return table, count, torch.zeros(1, dtype=torch.int32, device=device)


@jit_cache
def compile_kda_bprop(op: KdaBprop, beta_name: str):
    """Compile for dynamic tokens, heads, sequences, checkpoint rows and work rows."""
    io, i32 = op.io_dtype, cutlass.Int32
    tokens, hq, hk, hv, heads, rows, ckpt_rows = (cute.sym_int() for _ in range(7))

    def packed(dtype, h, d):
        return make_fake_strided_tensor(
            dtype, (tokens, h, d), stride_divisibility=tma_divisibility(dtype)
        )

    def state():
        return make_fake_strided_tensor(
            Float32, (cute.sym_int(), heads, op.d_v, op.d_k), stride_divisibility=4
        )

    beta_t = CUTE_DTYPES[beta_name]
    return compile_tvm_ffi(
        op,
        packed(io, hq, op.d_k),
        packed(io, hk, op.d_k),
        packed(io, hv, op.d_v),
        packed(io, heads, op.d_v),
        packed(op.gate_dtype, heads, op.d_k),
        make_fake_strided_tensor(beta_t, (tokens, heads), contiguous_dim=None),
        make_fake_strided_tensor(io, (ckpt_rows, heads, op.d_v, op.d_k), stride_divisibility=8),
        packed(io, heads, op.d_k),
        packed(io, heads, op.d_k),
        packed(io, heads, op.d_v),
        packed(op.gate_dtype, heads, op.d_k),
        make_fake_strided_tensor(beta_t, (tokens, heads), contiguous_dim=None),
        state() if op.use_dstate_in else None,
        state() if op.use_dstate0 else None,
        make_fake_strided_tensor(i32, (rows, WORK_ITEM_FIELDS), contiguous_dim=None),
        make_fake_strided_tensor(i32, (1,)),
        make_fake_strided_tensor(i32, (cute.sym_int(),)),
        fake_tma_slots(),
        Float32(1.0),
        None,
        name=op.get_name(),
        opt_level=OPT_LEVEL,
    )


def kda_bprop(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    checkpoints: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    scale: float,
    use_initial_state: bool,
    d_final_state: torch.Tensor | None = None,
    d_initial_state: torch.Tensor | None = None,
    work: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
):
    """Packed ``[T, H, D]`` KDA backward from the checkpoints entering every chunk
    (``checkpoints[checkpoint_row(cu[b], b, c), h]``, ``[rows, H, d_v, d_k]`` in the io dtype;
    chunk 0's row is read only with ``use_initial_state``). Returns per-output-head ``(dq, dk,
    dv, dgate, dbeta)``; ``d_initial_state`` (fp32 ``[N, H, d_v, d_k]``) receives the state
    gradient. ``work`` overrides the uncut work table ``(items, count, zeroed scheduler)``."""
    if q.dtype not in (torch.float16, torch.bfloat16) or any(
        t.dtype != q.dtype for t in (k, v, do, checkpoints)
    ):
        raise TypeError("q, k, v, do and checkpoints must share a 16-bit dtype")
    heads, d_k, d_v = gate.shape[1], q.shape[-1], v.shape[-1]
    tokens = q.shape[0]

    def per_head(dim, dtype):
        return torch.empty(tokens, heads, dim, dtype=dtype, device=q.device)

    dq, dk, dv = per_head(d_k, q.dtype), per_head(d_k, q.dtype), per_head(d_v, q.dtype)
    dgate, dbeta = torch.empty_like(gate), torch.empty_like(beta)
    op = KdaBprop(
        TORCH_NAMES[q.dtype],
        TORCH_NAMES[gate.dtype],
        d_k,
        d_v,
        use_initial_state,
        d_final_state is not None,
        d_initial_state is not None,
        torch.cuda.get_device_properties(q.device).multi_processor_count,
    )
    table, count, scheduler = work if work is not None else uncut_work_items(cu_seqlens, heads)
    fn = compile_kda_bprop(op, TORCH_NAMES[beta.dtype])
    fn(
        q, k, v, do, gate, beta, checkpoints, dq, dk, dv, dgate, dbeta, d_final_state,
        d_initial_state, table, count, scheduler,
        new_tma_slots(op.tma_slot_rows(heads), q.device), scale, None,
    )  # fmt: skip
    return dq, dk, dv, dgate, dbeta


@cute.jit
def _dgate_term(q_v, k_v, dq, dk, dk_decay, fuse_k: cutlass.Constexpr[bool]):
    """``q dq + k (2 dk_decay - dk)`` with the upstream kernel's rounding. LLVM fuses one of the
    two products into the add: the ``k`` product in the split-channel (d_k = 64) layout, the
    ``q`` product in the one-channel-per-thread layout. Explicit ``.rn`` instructions pin that
    choice rather than leaving it to the contraction heuristics, which follow the register
    dataflow and picked the other product for this kernel's split layout."""
    diff = inline_ptx(
        "{ .reg .f32 t; add.rn.f32 t, $1, $1; sub.rn.f32 $0, t, $2; }",
        write_only_types=[Float32],
        read_only_args=[dk_decay, dk],
    )
    if cutlass.const_expr(fuse_k):
        return _fma_rn(diff, k_v, _mul_rn(dq, q_v))
    return _fma_rn(q_v, dq, _mul_rn(k_v, diff))


@cute.jit
def _mul_rn(a, b):
    return inline_ptx("mul.rn.f32 $0, $1, $2;", write_only_types=[Float32], read_only_args=[a, b])


@cute.jit
def _add_rn(a, b):
    return inline_ptx("add.rn.f32 $0, $1, $2;", write_only_types=[Float32], read_only_args=[a, b])


@cute.jit
def _fma_rn(a, b, c):
    return inline_ptx(
        "fma.rn.f32 $0, $1, $2, $3;", write_only_types=[Float32], read_only_args=[a, b, c]
    )
