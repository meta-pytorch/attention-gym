"""The messages of one Inkling 2 Mini core pass as one persistent warpdrive launch
(``InklingMsgKernel``) and its compile function; ``megakernel`` holds the host side.

Passes after the first read, at every core layer, four fp32 messages: ``ref.block_residual``
of the layer's message queries over two states of the previous pass, its block ``in12`` (the
core's second block input) and its output ``out17``. Each message is a convex mix of the two
RMS-normalized states, so it depends on nothing in the current pass, and neither does the half
of each message gate's down projection that reads it (``ref.message_gate``'s ``h`` is
``down . [rms(x), rms(m)]``). The launch computes both for every (layer, gate) once per pass::

    every CTA   prologue: unweighted RMSNorm of both states (registers)
    MSG  2 per  the query's two scores and their softmax, the message m (written once), then
         gate   32 rows of the gate's down projection x its second half on bf16 rms(m)
                                                               -> fp32 message halves

The visit launches of the pass read ``messages`` and ``mhalf`` (``common``'s GATEMIX). Items
depend on nothing but the prologue: no tagged words, no epoch. Roles as ``visit_kernel``:
warp 8 ``load`` streams 32 x 256 weight stages, warps 0-7 ``compute`` run ``gemv_tile``
(``common.MatrixRowsKernel``).
"""

from dataclasses import dataclass

import cutlass
from cuda.bindings import driver as cuda
from cutlass import BFloat16, Float32, Int32, cute

from attn_gym.warpdrive import Consumer, Local, Producer, StageView, WorkCursor, role
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.examples.inkling_megakernel.common import (
    GATE_RANK,
    GATE_TILES,
    WORDS,
    MatrixRowsKernel,
)
from attn_gym.warpdrive.examples.megakernel.constants import (
    COMPUTE_THREADS,
    COMPUTE_WARPS,
    MMA_N,
    RED_VALUES,
    ROWS,
)
from attn_gym.warpdrive.examples.megakernel.gemv import gemv_tile, owns_row
from attn_gym.warpdrive.examples.megakernel.multi_gemv import finish, gemv_bf16
from attn_gym.warpdrive.examples.megakernel.regmath import cta_sum, fp32, fp32_reciprocal

MSG = 0


@dataclass(frozen=True)
class MsgParams:
    tma: tuple  # TMA atom of the down projections' message halves
    gmem: tuple  # its tile partition: (TMA, row tiles, k tiles)
    sW: StageView
    xs: cute.Tensor  # (1, hidden) bf16: the normalized message
    red: cute.Tensor  # (REDUCTIONS, COMPUTE_WARPS, RED_VALUES) fp32 reductions
    block: cute.Tensor  # (hidden,) bf16 the previous pass's in12
    partial: cute.Tensor  # (hidden,) bf16 the previous pass's out17
    queries: cute.Tensor  # (messages, hidden) fp32 message queries, (layer, gate) major
    messages: cute.Tensor  # (messages, hidden) fp32 output
    mhalf: cute.Tensor  # (messages * GATE_RANK,) fp32 output (multi-token: token-minor)
    dbuf: cute.Tensor | None = (
        None  # multi-token: ``examples.megakernel.multi_gemv.finish``'s buffer
    )


class InklingMsgKernel(MatrixRowsKernel):
    """One core pass's messages and message halves (see the module docstring). Constructor
    arguments are the hidden size and the gates' RMSNorm epsilon; ``stages``: weight ring depth
    in 16 KB stages."""

    def __init__(self, hidden: int, rms_eps: float, stages: int | None = None, tokens: int = 1):
        if not 1 <= tokens <= RED_VALUES // 2:
            raise ValueError(f"tokens must be in [1, {RED_VALUES // 2}], got {tokens}")
        self.init_matrix_rows(hidden, rms_eps, stages, tokens)

    def get_name(self) -> str:
        tokens = f"_t{self.tokens}" if self.tokens > 1 else ""
        return f"warpdrive_inkling_msg_h{self.hidden}{tokens}_s{self.stages}{self.trace_key()}"

    @role(warps=COMPUTE_WARPS)
    @cute.jit
    def load(self, p: MsgParams, *, w: Producer, work: WorkCursor):
        return self.load_matrix_rows(p, w, work)

    @role(warps=range(COMPUTE_WARPS))
    @cute.jit
    def compute(self, p: MsgParams, *, w: Consumer, work: WorkCursor, me: Local):
        if cutlass.const_expr(self.tokens > 1):
            w, work = self.compute_tokens(p, w, work, me)
        else:
            w, work = self.compute_token(p, w, work, me)
        return w, work

    @cute.jit
    def compute_token(self, p: MsgParams, w, work, me):
        tid, warp, lane = me.tid, me.warp, me.lane
        h = self.hidden
        per = h // COMPUTE_THREADS
        inv_h = fp32_reciprocal(h)
        # The two states, RMS-normalized (eps 1e-6, as ref.block_residual).
        x = cute.make_rmem_tensor((2, per), Float32)
        for r in cutlass.range_constexpr(per):
            x[0, r] = Float32(p.block[r * COMPUTE_THREADS + tid])
            x[1, r] = Float32(p.partial[r * COMPUTE_THREADS + tid])
        squares = []
        for n in cutlass.range_constexpr(2):
            ss = Float32(0.0)
            for r in cutlass.range_constexpr(per):
                ss = ss + x[n, r] * x[n, r]
            squares.append(ss)
        squares = cta_sum(p.red, self.csync, squares, 0, warp, lane)
        for n in cutlass.range_constexpr(2):
            rinv = cute.math.rsqrt(squares[n] * inv_h + Float32(1e-6))
            for r in cutlass.range_constexpr(per):
                x[n, r] = x[n, r] * rinv
        while work.valid:
            _op, tile, _b, _row = work.item
            query = tile // GATE_TILES
            dots = [Float32(0.0), Float32(0.0)]
            for r in cutlass.range_constexpr(per):
                q = p.queries[query, r * COMPUTE_THREADS + tid]
                dots[0] = dots[0] + q * x[0, r]
                dots[1] = dots[1] + q * x[1, r]
            dots = cta_sum(p.red, self.csync, dots, 1, warp, lane)
            scale = fp32(h**-0.5)
            s0, s1 = dots[0] * scale, dots[1] * scale
            top = cutlass.max(s0, s1)
            e0, e1 = cute.math.exp(s0 - top), cute.math.exp(s1 - top)
            total = e0 + e1
            p0, p1 = e0 / total, e1 / total
            m = cute.make_rmem_tensor((per,), Float32)
            ss = Float32(0.0)
            for r in cutlass.range_constexpr(per):
                acc = Float32(0.0)
                acc = acc + p0 * x[0, r]
                acc = acc + p1 * x[1, r]
                m[r] = acc
                ss = ss + acc * acc
                if tile % GATE_TILES == 0:
                    p.messages[query, r * COMPUTE_THREADS + tid] = acc
            ss = cta_sum(p.red, self.csync, [ss], 2, warp, lane)[0]
            rinv = cute.math.rsqrt(ss * inv_h + Float32(self.rms_eps))
            for r in cutlass.range_constexpr(per):
                p.xs[0, r * COMPUTE_THREADS + tid] = BFloat16(m[r] * rinv)
            self.csync.sync()
            v, row, w = gemv_tile(w, p.xs, 0, self.hidden_k_tiles, warp, lane)
            if owns_row(lane):
                p.mhalf[tile * ROWS + row] = v
            work.advance()
        return w, work

    @cute.jit
    def compute_tokens(self, p: MsgParams, w, work, me):
        """``compute`` for a block of tokens: the states are token-major; each token's state
        norms, query dots and message norms reduce together (one ``cta_sum`` each for the
        block), the states are reread rather than held, and each tile streams once for every
        token's message. ``messages[query, t * hidden + i]``, ``mhalf[row * tokens + t]``."""
        tid, warp, lane = me.tid, me.warp, me.lane
        h, nt = self.hidden, self.tokens
        per = h // COMPUTE_THREADS
        inv_h = fp32_reciprocal(h)
        squares = []
        for t in cutlass.range_constexpr(nt):
            for state in (p.block, p.partial):
                ss = Float32(0.0)
                for r in cutlass.range_constexpr(per):
                    v = Float32(state[t * h + r * COMPUTE_THREADS + tid])
                    ss = ss + v * v
                squares.append(ss)
        squares = cta_sum(p.red, self.csync, squares, 0, warp, lane)
        rinv = cute.make_rmem_tensor((2 * nt,), Float32)
        for n in cutlass.range_constexpr(2 * nt):
            rinv[n] = cute.math.rsqrt(squares[n] * inv_h + Float32(1e-6))
        scale = fp32(h**-0.5)
        while work.valid:
            _op, tile, _b, _row = work.item
            query = tile // GATE_TILES
            dots = []
            for t in cutlass.range_constexpr(nt):
                d0 = Float32(0.0)
                d1 = Float32(0.0)
                for r in cutlass.range_constexpr(per):
                    k = r * COMPUTE_THREADS + tid
                    q = p.queries[query, k]
                    d0 = d0 + q * (Float32(p.block[t * h + k]) * rinv[2 * t])
                    d1 = d1 + q * (Float32(p.partial[t * h + k]) * rinv[2 * t + 1])
                dots += [d0, d1]
            dots = cta_sum(p.red, self.csync, dots, 1, warp, lane)
            probs = []
            for t in cutlass.range_constexpr(nt):
                s0, s1 = dots[2 * t] * scale, dots[2 * t + 1] * scale
                top = cutlass.max(s0, s1)
                e0, e1 = cute.math.exp(s0 - top), cute.math.exp(s1 - top)
                total = e0 + e1
                probs.append((e0 / total, e1 / total))
            norms = []
            for t in cutlass.range_constexpr(nt):
                ss = Float32(0.0)
                for r in cutlass.range_constexpr(per):
                    k = r * COMPUTE_THREADS + tid
                    m = self.message(p, probs[t], rinv, t, k)
                    ss = ss + m * m
                    if tile % GATE_TILES == 0:
                        p.messages[query, t * h + k] = m
                norms.append(ss)
            norms = cta_sum(p.red, self.csync, norms, 2, warp, lane)
            for t in cutlass.range_constexpr(nt):
                rm = cute.math.rsqrt(norms[t] * inv_h + Float32(self.rms_eps))
                for r in cutlass.range_constexpr(per):
                    k = r * COMPUTE_THREADS + tid
                    p.xs[t, k] = BFloat16(self.message(p, probs[t], rinv, t, k) * rm)
            self.csync.sync()
            rows = [min(t, nt - 1) for t in range(MMA_N)]
            acc, w = gemv_bf16(w, p.xs, rows, 0, self.hidden_k_tiles, warp, lane)
            v = finish(acc, p.dbuf, self.csync, warp, lane, tid)
            t = tid % MMA_N
            if t < nt:
                p.mhalf[(tile * ROWS + tid // MMA_N) * nt + t] = v
            work.advance()
        return w, work

    @cute.jit
    def message(self, p, probs, rinv, t, k):
        """Element ``k`` of token ``t``'s message: the convex mix of its normalized states (in
        ``compute``'s operation order)."""
        acc = Float32(0.0)
        acc = acc + probs[0] * (Float32(p.block[t * self.hidden + k]) * rinv[2 * t])
        acc = acc + probs[1] * (Float32(p.partial[t * self.hidden + k]) * rinv[2 * t + 1])
        return acc

    @cute.kernel
    def kernel(self, tma, gtensor, tensors, bound):
        self.begin_roles(work=bound, layouts=self.matrix_layouts())
        # Materialize the argument tensors in converged code: a first use inside one role's
        # branch would not dominate the other roles' uses.
        block, partial, queries, messages, mhalf = (
            cute.make_tensor(t.iterator, t.layout) for t in tensors
        )
        params = MsgParams(
            **self.matrix_params(tma, gtensor),
            block=block,
            partial=partial,
            queries=queries,
            messages=messages,
            mhalf=mhalf,
        )
        self.run_roles(params)

    @cute.jit
    def __call__(
        self,
        down: cute.Tensor,
        queries: cute.Tensor,
        block: cute.Tensor,
        partial: cute.Tensor,
        messages: cute.Tensor,
        mhalf: cute.Tensor,
        table: cute.Tensor,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        h = self.hidden
        # The message halves: columns hidden .. 2 * hidden of every gate's down projection.
        halves = cute.local_tile(down, (down.shape[0], h), (0, 1))
        atom, gtensor = self.matrix_atom(halves)
        bound = self.work.bind(table, workspace, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(atom, gtensor, (block, partial, queries, messages, mhalf), bound).launch(
            **self.launch_config(bound), stream=stream
        )


@jit_cache
def compile_msg_kernel(op: InklingMsgKernel):
    """One compile serves every pass and core-layer count: the gates' down projections
    ``(messages * GATE_RANK, 2 * hidden)`` bf16, their message queries, both states and the
    outputs are arguments; the table lists the down projections' 32-row tiles."""
    bf16, f32 = BFloat16, Float32
    h, nt = op.hidden, op.tokens
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (cute.sym_int(divisibility=GATE_RANK), 2 * h)),
        make_fake_compact_tensor(f32, (cute.sym_int(), h), assumed_align=4),
        make_fake_compact_tensor(bf16, (nt * h,), assumed_align=2),
        make_fake_compact_tensor(bf16, (nt * h,), assumed_align=2),
        make_fake_compact_tensor(f32, (cute.sym_int(), nt * h), assumed_align=4),
        make_fake_compact_tensor(f32, (cute.sym_int(divisibility=GATE_RANK),), assumed_align=4),
        make_fake_compact_tensor(Int32, (cute.sym_int(), WORDS)),
        make_fake_compact_tensor(Int32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )
