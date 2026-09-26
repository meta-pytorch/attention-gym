"""Interleave local and sparse KV per document for native FA4 varlen offsets.

FA4 expects each document's KV in one contiguous range. A final isolated document
holds inactive capacity, so all physical query rows have valid native offsets.
The inverse permutation reuses the original boundaries without saving an index map.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from attn_gym._backends.triton.utils import _document_ids, ptr_offset, requires_int64_offsets


@triton.jit
def _pack_kv_kernel(
    Local,
    Sparse,
    Packed,
    CuQ,
    CuK,
    ExtendedCuQ,
    CombinedCuKV,
    local_tokens,
    sparse_tokens,
    num_documents,
    stride_local_t,
    stride_local_d,
    stride_sparse_t,
    stride_sparse_d,
    stride_packed_t,
    stride_packed_d,
    D: tl.constexpr,
    PACK: tl.constexpr,
    WIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.arange(0, BLOCK_D)
    if WIDE:
        row = row.to(tl.int64)
        channel = channel.to(tl.int64)

    if row < local_tokens + sparse_tokens:
        if row < local_tokens:
            document = _document_ids(CuQ, row, num_documents, WIDE)
            offset = document.to(tl.int64) if WIDE else document
            packed_row = row + tl.load(CuK + offset)
            unpacked_ptr = Local + ptr_offset((row, channel), (stride_local_t, stride_local_d))
        else:
            sparse_row = row - local_tokens
            document = _document_ids(CuK, sparse_row, num_documents, WIDE)
            offset = document.to(tl.int64) if WIDE else document
            local_end = tl.load(CuQ + tl.minimum(offset + 1, num_documents))
            # Inactive sparse capacity follows all local capacity in the tail document.
            packed_row = sparse_row + tl.where(document < num_documents, local_end, local_tokens)
            unpacked_ptr = Sparse + ptr_offset(
                (sparse_row, channel), (stride_sparse_t, stride_sparse_d)
            )
        packed_ptr = Packed + ptr_offset((packed_row, channel), (stride_packed_t, stride_packed_d))
        if PACK:
            values = tl.load(unpacked_ptr, channel < D, 0)
            tl.store(packed_ptr, values, channel < D)
        else:
            values = tl.load(packed_ptr, channel < D, 0)
            tl.store(unpacked_ptr, values, channel < D)

    if PACK:
        if row <= num_documents:
            query_end = tl.load(CuQ + row)
            sparse_end = tl.load(CuK + row)
            tl.store(ExtendedCuQ + row, query_end)
            tl.store(CombinedCuKV + row, query_end + sparse_end)
        if row == num_documents + 1:
            tl.store(ExtendedCuQ + row, local_tokens)
            tl.store(CombinedCuKV + row, local_tokens + sparse_tokens)


def _launch_pack_kv(
    local: Tensor,
    sparse: Tensor,
    packed: Tensor,
    cu_seqlens: Tensor,
    cu_seqlens_k: Tensor,
    extended_cu_seqlens: Tensor | None,
    combined_cu_seqlens: Tensor | None,
    *,
    pack: bool,
) -> None:
    local_tokens, head_dim = local.shape[2:]
    sparse_tokens = sparse.shape[2]
    num_documents = cu_seqlens.numel() - 1
    with torch.cuda.device(local.device):
        _pack_kv_kernel[(max(local_tokens + sparse_tokens, num_documents + 2),)](
            local,
            sparse,
            packed,
            cu_seqlens,
            cu_seqlens_k,
            extended_cu_seqlens,
            combined_cu_seqlens,
            local_tokens,
            sparse_tokens,
            num_documents,
            *local.stride()[2:],
            *sparse.stride()[2:],
            packed.stride(0),
            packed.stride(2),
            head_dim,
            pack,
            requires_int64_offsets(
                local,
                sparse,
                packed,
                cu_seqlens,
                cu_seqlens_k,
                extended_cu_seqlens,
                combined_cu_seqlens,
            ),
            triton.next_power_of_2(head_dim),
            num_warps=4,
        )


class _PackKV(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, local: Tensor, sparse: Tensor, cu_seqlens: Tensor, cu_seqlens_k: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        local_tokens, head_dim = local.shape[2:]
        sparse_tokens = sparse.shape[2]
        if local_tokens + sparse_tokens > torch.iinfo(torch.int32).max:
            raise ValueError("Combined KV capacity must fit in int32 FA4 offsets.")
        packed = local.new_empty((local_tokens + sparse_tokens, 1, head_dim))
        extended_cu_seqlens = cu_seqlens.new_empty((cu_seqlens.numel() + 1,))
        combined_cu_seqlens = torch.empty_like(extended_cu_seqlens)
        _launch_pack_kv(
            local,
            sparse,
            packed,
            cu_seqlens,
            cu_seqlens_k,
            extended_cu_seqlens,
            combined_cu_seqlens,
            pack=True,
        )
        ctx.save_for_backward(cu_seqlens, cu_seqlens_k)
        ctx.local_shape = local.shape
        ctx.sparse_shape = sparse.shape
        ctx.mark_non_differentiable(extended_cu_seqlens, combined_cu_seqlens)
        ctx.set_materialize_grads(False)
        return packed, extended_cu_seqlens, combined_cu_seqlens

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx, grad_packed: Tensor, _grad_extended: None, _grad_combined: None
    ) -> tuple[Tensor, Tensor, None, None]:
        cu_seqlens, cu_seqlens_k = ctx.saved_tensors
        grad_local = grad_packed.new_empty(ctx.local_shape)
        grad_sparse = grad_packed.new_empty(ctx.sparse_shape)
        _launch_pack_kv(
            grad_local,
            grad_sparse,
            grad_packed,
            cu_seqlens,
            cu_seqlens_k,
            None,
            None,
            pack=False,
        )
        return grad_local, grad_sparse, None, None


def pack_kv(
    local: Tensor, sparse: Tensor, cu_seqlens: Tensor, cu_seqlens_k: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Pack validated [1, 1, T/S, D] KV and append an always-present tail document.

    Returns [T + S, 1, D] KV, extended query offsets, and combined KV offsets.
    Offset tensors are contiguous int32 with matching document counts; their final
    endpoints may precede capacity. No CUDA endpoint is read on the host.
    """
    return _PackKV.apply(local, sparse, cu_seqlens, cu_seqlens_k)
