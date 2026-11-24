"""The end of an Inkling 2 Mini decode step as one persistent warpdrive launch
(``InklingHeadKernel``) and its compile function; ``megakernel`` holds the host side.

Batch one, one token. The launch implements ``ref.hidden_states``' last block residual and
``ref.logits_of``::

    every CTA   prologue: block residual of the final query over [blocks..., h], final RMSNorm
                -> bf16 x (SMEM)
    LMHEAD      32 LM-head rows x hidden -> fp32 logits tanh(bf16(row . x) / cap) * cap

The LM head's rows need no padding: TMA zero-fills the last tile's rows past the vocabulary,
whose logits are not stored. Items depend on nothing but the prologue: no tagged words, no
epoch. Roles as ``visit_kernel``: warp 8 ``load`` streams 32 x 256 weight stages, warps 0-7
``compute`` run ``gemv_tile`` (``common.MatrixRowsKernel``).
"""

from dataclasses import dataclass

from cuda.bindings import driver as cuda
from cutlass import BFloat16, Float32, Int32, cute

from attn_gym.warpdrive import Consumer, Local, Producer, StageView, WorkCursor, role
from attn_gym.warpdrive.cache import compile_tvm_ffi, jit_cache, make_fake_compact_tensor
from attn_gym.warpdrive.examples.inkling_megakernel.common import (
    MAX_BLOCKS,
    WORDS,
    MatrixRowsKernel,
)
from attn_gym.warpdrive.examples.megakernel.constants import COMPUTE_WARPS, ROWS, TILE_K
from attn_gym.warpdrive.examples.megakernel.gemv import gemv_tile, owns_row
from attn_gym.warpdrive.examples.megakernel.regmath import bf16_round, fp32_reciprocal
from attn_gym.warpdrive.examples.megakernel.ring import tma_atom

LMHEAD = 0


@dataclass(frozen=True)
class HeadParams:
    tma: tuple  # TMA atom of the LM head
    gmem: tuple  # its tile partition: (TMA, row tiles, k tiles)
    sW: StageView
    xs: cute.Tensor  # (1, hidden) bf16: the final normalized state
    red: cute.Tensor  # (REDUCTIONS, COMPUTE_WARPS, RED_VALUES) fp32 prologue reductions
    blocks: tuple  # MAX_BLOCKS (hidden,) bf16 completed blocks; the first num_blocks are read
    num_blocks: Int32
    h: cute.Tensor  # (hidden,) bf16 the last visit's output (the partial block)
    query: cute.Tensor  # (1, hidden) fp32 final block-residual query
    norms: cute.Tensor  # (1, hidden) bf16 final RMSNorm weight
    logits: cute.Tensor  # (vocab,) fp32 output


class InklingHeadKernel(MatrixRowsKernel):
    """The final block residual, norm, LM head and logit softcap (see the module docstring).
    Constructor arguments are the model sizes and constants the kernel reads; ``stages``:
    weight ring depth in 16 KB stages."""

    def __init__(
        self,
        hidden: int,
        vocab: int,
        rms_eps: float,
        softcap: float,
        stages: int | None = None,
    ):
        self.init_matrix_rows(hidden, rms_eps, stages)
        self.vocab, self.softcap = vocab, softcap
        self.row_tiles = -(-vocab // ROWS)

    def get_name(self) -> str:
        return (
            f"warpdrive_inkling_head_h{self.hidden}_v{self.vocab}_s{self.stages}{self.trace_key()}"
        )

    @role(warps=COMPUTE_WARPS)
    @cute.jit
    def load(self, p: HeadParams, *, w: Producer, work: WorkCursor):
        return self.load_matrix_rows(p, w, work)

    @role(warps=range(COMPUTE_WARPS))
    @cute.jit
    def compute(self, p: HeadParams, *, w: Consumer, work: WorkCursor, me: Local):
        tid, warp, lane = me.tid, me.warp, me.lane
        x = self.load_states(p, p.num_blocks, p.h, tid)
        self.normalize_states(p, x, p.num_blocks, p.query, p.norms, 1, Int32(1), tid, warp, lane)
        cap = self.softcap
        while work.valid:
            _op, tile, _b, _row = work.item
            v, row, w = gemv_tile(w, p.xs, 0, self.hidden_k_tiles, warp, lane)
            index = tile * ROWS + row
            if owns_row(lane) & (index < self.vocab):
                # ref.logits_of: the bf16 LM-head output, softcapped in fp32.
                logit = bf16_round(v) * fp32_reciprocal(cap)
                p.logits[index] = cute.math.tanh(logit) * cap
            work.advance()
        return w, work

    @cute.kernel
    def kernel(self, tma, gtensor, tensors, blocks, bound, num_blocks):
        self.begin_roles(work=bound, layouts=self.core_layouts(1))
        # Materialize the argument tensors in converged code: a first use inside one role's
        # branch would not dominate the other roles' uses.
        h, query, norms, logits = (cute.make_tensor(t.iterator, t.layout) for t in tensors)
        blocks = tuple(cute.make_tensor(t.iterator, t.layout) for t in blocks)
        params = HeadParams(
            **self.matrix_params(tma, gtensor),
            blocks=blocks,
            num_blocks=num_blocks,
            h=h,
            query=query,
            norms=norms,
            logits=logits,
        )
        self.run_roles(params)

    @cute.jit
    def __call__(
        self,
        lm_head: cute.Tensor,
        query: cute.Tensor,
        norms: cute.Tensor,
        h: cute.Tensor,
        block0: cute.Tensor,
        block1: cute.Tensor,
        block2: cute.Tensor,
        block3: cute.Tensor,
        num_blocks: Int32,
        logits: cute.Tensor,
        table: cute.Tensor,
        workspace: cute.Tensor,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        atom, gtensor = tma_atom(lm_head, (ROWS, TILE_K))
        bound = self.work.bind(table, workspace, max_ctas)
        self.kernel.set_name_prefix(self.get_name())
        self.kernel(
            atom,
            gtensor,
            (h, query, norms, logits),
            (block0, block1, block2, block3),
            bound,
            num_blocks,
        ).launch(**self.launch_config(bound), stream=stream)


@jit_cache
def compile_head_kernel(op: InklingHeadKernel):
    """One compile serves every step: the LM head, final query and norm, states and logits are
    arguments; the block count is read at run time."""
    bf16, f32 = BFloat16, Float32
    h = op.hidden
    vector = [make_fake_compact_tensor(bf16, (h,), assumed_align=2) for _ in range(MAX_BLOCKS)]
    return compile_tvm_ffi(
        op,
        make_fake_compact_tensor(bf16, (op.vocab, h)),
        make_fake_compact_tensor(f32, (1, h), assumed_align=4),
        make_fake_compact_tensor(bf16, (1, h), assumed_align=2),
        make_fake_compact_tensor(bf16, (h,), assumed_align=2),  # h
        *vector,
        Int32(1),
        make_fake_compact_tensor(f32, (op.vocab,), assumed_align=4),
        make_fake_compact_tensor(Int32, (cute.sym_int(), WORDS)),
        make_fake_compact_tensor(Int32, (cute.sym_int(),)),
        Int32(1),
        name=op.get_name(),
    )
