"""``frost.umma`` descriptors and ``frost.ptx`` TMEM access against a torch GEMM.

One CTA computes ``A @ B^T`` (128 x 64 x 128, 16-bit) with hand-built shared-memory and
instruction descriptors, in both the SMEM-A (``mma_ss``) and TMEM-A (``mma_ts_step``) forms that
the Frost kernels issue, and reads the fp32 accumulator back with ``tmem_ld``.
"""

import pytest
import torch

cutlass = pytest.importorskip("cutlass")
from cutlass import Int32, cute
from cutlass.cute.runtime import from_dlpack

from attn_gym.linear._delta_rule.frost import ptx, umma
from attn_gym.warpdrive.compat import SmemAllocator

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="tcgen05 requires SM100 or SM103",
)

M, N, K = 128, 64, 128


@cute.kernel
def gemm_kernel(a: cute.Tensor, b: cute.Tensor, d: cute.Tensor, ts: cutlass.Constexpr[bool]):
    dtype = a.element_type
    tid = Int32(cute.arch.thread_idx()[0])
    warp = tid // 32
    smem = SmemAllocator()
    # K-major operands as K // 64 boxes of [rows x 64], 128-byte swizzle.
    sA = smem.allocate_tensor(dtype, cute.make_layout(M * K), byte_alignment=1024)
    sB = smem.allocate_tensor(dtype, cute.make_layout(N * K), byte_alignment=1024)
    bar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(1), byte_alignment=8)
    slot = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=4)
    for i in cutlass.range(tid, M * K, 128, unroll=1):
        r, c = i // K, i % K
        sA[(c // 64) * M * 64 + r * 64 + ptx.swizzle_xor_128b(r, c % 64)] = a[r, c]
    for i in cutlass.range(tid, N * K, 128, unroll=1):
        r, c = i // K, i % K
        sB[(c // 64) * N * 64 + r * 64 + ptx.swizzle_xor_128b(r, c % 64)] = b[r, c]
    if tid == 0:
        cute.arch.mbarrier_init(bar.iterator, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.fence_view_async_shared()
    if warp == 0:
        cute.arch.alloc_tmem(256, slot.iterator)
    cute.arch.barrier()
    tmem = slot[0]
    acc = tmem  # columns [0, N)
    a_tmem = tmem + 128  # TS form: A as 16-bit pairs, K / 2 columns
    bpe = dtype.width // 8
    idesc = umma.instr_desc(cutlass.Float32, dtype, dtype, m=M, n=N)
    desc = umma.MmaDesc(M=M, N=N, K=K, bpe_a=bpe, bpe_b=bpe, idesc=idesc)
    if cutlass.const_expr(ts):
        # Thread t owns TMEM lane t: store its A row as packed pairs.
        row = tid
        words = []
        for c in cutlass.range_constexpr(0, K, 2):
            words.append(
                ptx.fp32_to_fp16(
                    a[row, c].to(cutlass.Float32), a[row, c + 1].to(cutlass.Float32), dtype=dtype
                )
            )
        for blk in cutlass.range_constexpr(K // 2 // 8):
            ptx.tmem_st(
                "32x32b", a_tmem + ((32 * warp) << 16) + blk * 8, words[blk * 8 : blk * 8 + 8]
            )
        cute.arch.fence_view_async_tmem_store()
        cute.arch.barrier()
    if warp == 0:
        b_desc = umma.desc_opaque(
            umma.smem_desc(sB.iterator.toint(), 16, 8 * 128, umma.SWIZZLE_128B)
        )
        if cutlass.const_expr(ts):
            for sub in cutlass.range_constexpr(desc.num_subtiles):
                for k in cutlass.range_constexpr(desc.steps_per_subtile):
                    umma.mma_ts_step(
                        desc,
                        a_tmem + sub * desc.steps_per_subtile * desc.tmem_advance_A,
                        b_desc + ((desc.smem_subtile_B * sub) >> 4),
                        acc,
                        k,
                        cutlass.Boolean(sub > 0 or k > 0),
                    )
        else:
            a_desc = umma.desc_opaque(
                umma.smem_desc(sA.iterator.toint(), 16, 8 * 128, umma.SWIZZLE_128B)
            )
            umma.mma_ss(desc, a_desc, b_desc, acc, cutlass.Boolean(False))
        with cute.arch.elect_one():
            cute.nvgpu.tcgen05.commit(bar.iterator)
    cute.arch.mbarrier_wait(bar.iterator, 0)
    vals = ptx.tmem_ld("32x32b", N, tmem + ((32 * warp) << 16))
    cute.arch.fence_view_async_tmem_load()
    for c in cutlass.range_constexpr(N):
        d[tid, c] = vals[c]
    cute.arch.barrier()
    if warp == 0:
        cute.arch.relinquish_tmem_alloc_permit()
        cute.arch.dealloc_tmem(cute.make_ptr(cutlass.Float32, tmem, cute.AddressSpace.tmem), 256)


@cute.jit
def launch(a: cute.Tensor, b: cute.Tensor, d: cute.Tensor, ts: cutlass.Constexpr[bool]):
    gemm_kernel(a, b, d, ts).launch(grid=(1, 1, 1), block=(128, 1, 1), smem=64 * 1024)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("ts", [False, True], ids=["ss", "ts"])
def test_umma_gemm_matches_torch(dtype, ts):
    gen = torch.Generator(device="cuda").manual_seed(0)
    a = torch.randn(M, K, device="cuda", generator=gen).to(dtype)
    b = torch.randn(N, K, device="cuda", generator=gen).to(dtype)
    d = torch.zeros(M, N, device="cuda")
    fn = cute.compile(launch, from_dlpack(a), from_dlpack(b), from_dlpack(d), ts)
    fn(from_dlpack(a), from_dlpack(b), from_dlpack(d))
    torch.testing.assert_close(d, a.float() @ b.float().T, atol=1e-3, rtol=1e-3)
