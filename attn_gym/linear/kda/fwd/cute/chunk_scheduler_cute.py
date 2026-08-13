"""CuTeDSL work decoding for the packed KDA chunk scheduler."""

from __future__ import annotations

import cutlass
import torch
from cuda.bindings import driver as cuda
from cutlass import Int32, cute
from cutlass.cute.runtime import make_fake_compact_tensor

from attn_gym._backends.cute import compile_tvm_ffi, jit_cache
from attn_gym._backends.cute.device import upper_bound


@cute.jit
def load_ragged_chunk_work(
    cu_seqlens: cute.Tensor,
    chunk_offsets: cute.Tensor,
    global_chunk: Int32,
    chunk_size: Int32,
):
    """Decode one known-active global chunk into sequence-local coordinates."""
    num_sequences = Int32(cute.size(chunk_offsets)) - 1
    sequence = (
        upper_bound(
            chunk_offsets,
            global_chunk,
            Int32(0),
            num_sequences + 1,
        )
        - 1
    )
    sequence_offset = Int32(chunk_offsets[sequence])
    local_chunk = global_chunk - sequence_offset
    begin = Int32(cu_seqlens[sequence])
    end = Int32(cu_seqlens[sequence + 1])
    token_start = begin + local_chunk * chunk_size
    valid_tokens = cutlass.min(chunk_size, end - token_start)
    return sequence, local_chunk, token_start, valid_tokens


class ChunkSchedulerDiagnostic:
    """Decode one logical chunk per CTA and broadcast the work across warp groups."""

    threads = 128
    warps = threads // 32
    fields = 5

    @cute.jit
    def __call__(
        self,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        output: cute.Tensor,
        chunk_size: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            work: cute.struct.MemRange[Int32, self.fields]

        self.kernel(
            cu_seqlens,
            chunk_offsets,
            output,
            chunk_size,
            SharedStorage,
        ).launch(
            grid=(cute.size(output, mode=[0]), 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        cu_seqlens: cute.Tensor,
        chunk_offsets: cute.Tensor,
        output: cute.Tensor,
        chunk_size: Int32,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        global_chunk, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = tidx % 32
        num_sequences = Int32(cute.size(chunk_offsets)) - 1

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        work = storage.work.get_tensor(cute.make_layout(self.fields))

        if tidx == 0:
            active_chunks = Int32(chunk_offsets[num_sequences])
            if global_chunk < active_chunks:
                sequence, local_chunk, token_start, valid_tokens = load_ragged_chunk_work(
                    cu_seqlens,
                    chunk_offsets,
                    Int32(global_chunk),
                    chunk_size,
                )

                work[0] = Int32(global_chunk)
                work[1] = sequence
                work[2] = local_chunk
                work[3] = token_start
                work[4] = valid_tokens
            else:
                for field in cutlass.range_constexpr(self.fields):
                    work[field] = Int32(-1)

        cute.arch.sync_threads()
        if lane_idx == 0:
            for field in cutlass.range_constexpr(self.fields):
                output[global_chunk, warp_idx, field] = work[field]
        cute.arch.sync_threads()


@jit_cache
def _compile_chunk_scheduler_diagnostic():
    sequences = cute.sym_int()
    capacity = cute.sym_int()
    cu_seqlens = make_fake_compact_tensor(
        cutlass.Int32,
        (sequences,),
        stride_order=(0,),
        assumed_align=4,
    )
    chunk_offsets = make_fake_compact_tensor(
        cutlass.Int32,
        (sequences,),
        stride_order=(0,),
        assumed_align=4,
    )
    output = make_fake_compact_tensor(
        cutlass.Int32,
        (capacity, ChunkSchedulerDiagnostic.warps, ChunkSchedulerDiagnostic.fields),
        stride_order=(2, 1, 0),
        assumed_align=4,
    )
    return compile_tvm_ffi(
        ChunkSchedulerDiagnostic(),
        cu_seqlens,
        chunk_offsets,
        output,
        Int32(64),
        name="kda_chunk_scheduler_diagnostic",
    )


def decode_ragged_chunk_work_cute(
    cu_seqlens: torch.Tensor,
    chunk_offsets: torch.Tensor,
    capacity: int,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Decode and expose each warp group's scheduler broadcast for validation."""
    if chunk_size != 64:
        raise ValueError(f"the diagnostic scheduler requires chunk_size=64, got {chunk_size}")
    if cu_seqlens.dtype != torch.int32 or chunk_offsets.dtype != torch.int32:
        raise TypeError("cu_seqlens and chunk_offsets must be int32")
    if not cu_seqlens.is_cuda or not chunk_offsets.is_cuda:
        raise ValueError("cu_seqlens and chunk_offsets must be CUDA tensors")
    if cu_seqlens.shape != chunk_offsets.shape:
        raise ValueError("cu_seqlens and chunk_offsets must have the same shape")
    if not cu_seqlens.is_contiguous() or not chunk_offsets.is_contiguous():
        raise ValueError("cu_seqlens and chunk_offsets must be contiguous")

    output = torch.empty(
        (capacity, ChunkSchedulerDiagnostic.warps, ChunkSchedulerDiagnostic.fields),
        dtype=torch.int32,
        device=cu_seqlens.device,
    )
    if capacity:
        _compile_chunk_scheduler_diagnostic()(cu_seqlens, chunk_offsets, output, chunk_size)
    return output


__all__ = ["decode_ragged_chunk_work_cute", "load_ragged_chunk_work"]
