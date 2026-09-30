"""Tests for the CuTeDSL packed KDA chunk-scheduler decode helpers."""

from __future__ import annotations

import pytest
import torch

cutlass = pytest.importorskip("cutlass")

from cuda.bindings import driver as cuda
from cutlass import Int32, cute
from cutlass.cute.runtime import make_fake_compact_tensor

from attn_gym._backends.cute import compile_tvm_ffi, jit_cache
from attn_gym._backends.cute.compat import SmemAllocator
from attn_gym.linear._delta_rule.triton.chunk_scheduler import prepare_ragged_chunk_metadata
from attn_gym.linear.kda.fwd.cute.chunk_scheduler_cute import (
    load_ragged_chunk_count,
    load_ragged_chunk_work,
)


def _expected_tensor(lengths: list[int], chunk_size: int) -> tuple[torch.Tensor, list[int]]:
    """Return CPU-oracle ``(global_chunk, sequence, local_chunk, start, valid)`` rows."""
    rows = []
    offsets = [0]
    for sequence, length in enumerate(lengths):
        begin = offsets[-1]
        offsets.append(begin + length)
        for start in range(begin, begin + length, chunk_size):
            valid = min(chunk_size, begin + length - start)
            rows.append((len(rows), sequence, (start - begin) // chunk_size, start, valid))
    return torch.tensor(rows, dtype=torch.int32), offsets


class ChunkSchedulerDiagnostic:
    """Test one scheduler decode broadcast with a diagnostic-local CTA shape."""

    num_threads = 128
    num_warps = num_threads // cute.arch.WARP_SIZE
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
            block=(self.num_threads, 1, 1),
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
        lane_idx = tidx % cute.arch.WARP_SIZE

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)
        work = storage.work.get_tensor(cute.make_layout(self.fields))

        if tidx == 0:
            active_chunks = load_ragged_chunk_count(chunk_offsets)
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
                # The diagnostic materializes capacity-only CTAs into an uninitialized
                # output. Production kernels instead return without writing.
                for field in cutlass.range_constexpr(self.fields):
                    work[field] = Int32(-1)

        cute.arch.sync_threads()
        if lane_idx == 0:
            for field in cutlass.range_constexpr(self.fields):
                output[global_chunk, warp_idx, field] = work[field]


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
        (
            capacity,
            ChunkSchedulerDiagnostic.num_warps,
            ChunkSchedulerDiagnostic.fields,
        ),
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
        (
            capacity,
            ChunkSchedulerDiagnostic.num_warps,
            ChunkSchedulerDiagnostic.fields,
        ),
        dtype=torch.int32,
        device=cu_seqlens.device,
    )
    if capacity:
        _compile_chunk_scheduler_diagnostic()(cu_seqlens, chunk_offsets, output, chunk_size)
    return output


requires_cute = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="the CuTeDSL KDA scheduler test requires an SM100 or SM103 GPU",
)


@pytest.mark.parametrize(
    "lengths",
    [
        [65, 63],
        [0, 1, 0, 63, 64, 65, 0],
        [1] * 257,
        [4097, 1, 511, 0, 65],
    ],
)
@requires_cute
def test_cute_chunk_scheduler_broadcast_matches_oracle(lengths):
    expected, offsets = _expected_tensor(lengths, 64)
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, offsets[-1], 64)
    actual = decode_ragged_chunk_work_cute(
        cu_seqlens,
        metadata.chunk_offsets,
        metadata.capacity,
    ).cpu()

    for warp in range(actual.shape[1]):
        torch.testing.assert_close(actual[: expected.shape[0], warp], expected)
        torch.testing.assert_close(
            actual[expected.shape[0] :, warp],
            torch.full_like(actual[expected.shape[0] :, warp], -1),
        )


@requires_cute
def test_cute_chunk_scheduler_accepts_zero_capacity():
    cu_seqlens = torch.tensor([0, 0], device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, 0, 64)
    actual = decode_ragged_chunk_work_cute(
        cu_seqlens,
        metadata.chunk_offsets,
        metadata.capacity,
    )

    assert actual.shape == (0, 4, 5)


@requires_cute
def test_cute_chunk_scheduler_cuda_graph_replays_boundaries():
    _, offsets = _expected_tensor([65, 63], 64)
    cu_seqlens = torch.tensor(offsets, device="cuda", dtype=torch.int32)
    metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
    decode_ragged_chunk_work_cute(
        cu_seqlens,
        metadata.chunk_offsets,
        metadata.capacity,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_metadata = prepare_ragged_chunk_metadata(cu_seqlens, 128, 64)
        actual = decode_ragged_chunk_work_cute(
            cu_seqlens,
            captured_metadata.chunk_offsets,
            captured_metadata.capacity,
        )

    cu_seqlens.copy_(torch.tensor([0, 1, 128], device="cuda", dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()

    expected, _ = _expected_tensor([1, 127], 64)
    for warp in range(actual.shape[1]):
        torch.testing.assert_close(actual[: expected.shape[0], warp].cpu(), expected)
        torch.testing.assert_close(
            actual[expected.shape[0] :, warp].cpu(),
            torch.full_like(actual[expected.shape[0] :, warp].cpu(), -1),
        )
