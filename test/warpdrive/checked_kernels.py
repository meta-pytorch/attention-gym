"""A GEMM with a planted indexing mistake, for test_warpdrive_checked.py.

As a script, ``checked_kernels.py MISTAKE CHECKED`` runs one 512 x 64 x 128 product and prints the
largest error (a device assert poisons the CUDA context, so each launch gets a process), and
``checked_kernels.py ptx DIR`` compiles the plain GEMM before and after a checked one.
"""

import sys
from pathlib import Path

import cutlass
import torch
from cutlass import Int32, Uint32, cute

from attn_gym.warpdrive import Producer, ProtocolError, WorkCursor, role, unchecked
from attn_gym.warpdrive.examples.gemm import Gemm, GemmParams, compile_gemm
from attn_gym.warpdrive.ptx_check import capture_ptx, read_ptx


class MistakenGemm(Gemm):
    """``mistake``, in the load role:

    * ``tile``: B's tile is taken by the M tile index; ``opted_out`` does so inside
      ``unchecked("bounds")``;
    * ``static``: element 4 of a four-element register tensor is written; ``untaken`` does so
      in a branch no launch takes;
    * ``unsigned``: that tensor is read at the M tile index minus one, unsigned: 2**32 - 1 in
      the first tile;
    * ``base``: a view of the output starts one row before it;
    * ``raises``: tracing the role fails.
    """

    def __init__(self, mistake: str):
        super().__init__(tile_n=64)
        self.mistake = mistake

    def get_name(self) -> str:
        return f"{super().get_name()}_{self.mistake}"

    @role(warps=4, regs=48)
    @cute.jit
    def load(self, params: GemmParams, *, ab: Producer, work: WorkCursor):
        ab = ab.views(A=params.tAsA, B=params.tBsB)
        if cutlass.const_expr(self.mistake == "raises"):
            raise ProtocolError("planted")
        scratch = cute.make_rmem_tensor((4,), Int32)
        if cutlass.const_expr(self.mistake == "static"):
            scratch[4] = Int32(0)
        if cutlass.const_expr(self.mistake == "base"):
            cute.domain_offset((-1, 0), params.mC)
        while work.valid:
            m, n, _ = work.item
            if cutlass.const_expr(self.mistake == "untaken"):
                if m > Int32(1 << 20):
                    scratch[4] = Int32(0)
            if cutlass.const_expr(self.mistake == "unsigned"):
                scratch[0] = scratch[Uint32(m) - Uint32(1)]
            for k in cutlass.range(params.num_k_tiles, unroll=1):
                with ab.produce() as s:
                    cute.copy(
                        params.tma_a, params.tAgA[None, m, k], s.view("A"), tma_bar_ptr=s.barrier
                    )
                    if cutlass.const_expr(self.mistake == "opted_out"):
                        with unchecked("bounds", reason="the wrong tile, on purpose"):
                            tile = params.tBgB[None, m, k]
                    elif cutlass.const_expr(self.mistake == "tile"):
                        tile = params.tBgB[None, m, k]
                    else:
                        tile = params.tBgB[None, n, k]
                    cute.copy(params.tma_b, tile, s.view("B"), tma_bar_ptr=s.barrier)
            work.advance()
        return ab, work


def gemm_ptx(directory: Path, op: Gemm) -> str:
    """The PTX of an uncached compile of ``op`` (a cached kernel has none)."""
    with capture_ptx(directory) as out:
        compile_gemm.__wrapped__(op)
    (path,) = out.glob("*.ptx")
    return read_ptx(path)


if __name__ == "__main__":
    if sys.argv[1] == "ptx":
        base = Path(sys.argv[2])
        before = gemm_ptx(base / "before", Gemm(tile_n=64))
        checked = gemm_ptx(base / "checked", Gemm(tile_n=64, checked=True))
        after = gemm_ptx(base / "after", Gemm(tile_n=64))
        print(f"plain builds equal: {before == after}; checked differs: {checked != before}")
        sys.exit()
    op = MistakenGemm(sys.argv[1], checked=sys.argv[2] == "1")
    torch.manual_seed(0)
    a = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(512, 64, device="cuda", dtype=torch.bfloat16)
    compile_gemm(op)(a, b, out)
    torch.cuda.synchronize()
    error = (out.float() - torch.nn.functional.linear(a.float(), b.float())).abs().max().item()
    print(f"max abs err {error:.3g}", flush=True)
