"""Triton primitives shared by gather-attention kernel schedules."""

from collections.abc import Sequence
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from attn_gym._backends.triton.utils import _document_ids, ptr_offset, requires_int64_offsets


def can_use_shared_kv_schedule(
    query: torch.Tensor,
    sparse_kv: torch.Tensor,
    local_kv: torch.Tensor | None,
    sliding_window_size: int,
    query_rope: torch.Tensor | None = None,
    sparse_rope: torch.Tensor | None = None,
    local_rope: torch.Tensor | None = None,
) -> bool:
    """Check the shared Blackwell forward/backward schedule's layout and shape limits.

    The latent KV is both key and value. Optional key-only RoPE tensors add a second logit
    dot; with RoPE, ``sparse_rope`` (and ``local_rope`` under a window) must be given.
    """
    _, heads, _, head_dim = query.shape
    rope_tensors = (sparse_rope, local_rope) if sliding_window_size else (sparse_rope,)
    rope_ok = query_rope is None or (
        query_rope.shape[-1] in (16, 32, 64, 128)
        and query_rope.dtype == torch.bfloat16
        and all(
            rope is not None
            and rope.dtype == torch.bfloat16
            and rope.shape[-1] == query_rope.shape[-1]
            and rope.stride(1) == 0
            for rope in rope_tensors
        )
    )
    return (
        torch.cuda.get_device_capability(query.device)[0] >= 10
        and query.dtype == torch.bfloat16
        and sparse_kv.stride(1) == 0
        and query.stride(-1) == 1
        and sparse_kv.stride(-1) == 1
        # Without a window the kernels never read local KV, so it may be absent.
        and (sliding_window_size == 0 or (local_kv.stride(1) == 0 and local_kv.stride(-1) == 1))
        and 16 <= heads <= 128
        and head_dim % 16 == 0
        and (head_dim <= 128 or head_dim == 512)
        and rope_ok
        and sliding_window_size <= 2048
    )


class TileConfig(NamedTuple):
    """Launch parameters of a fixed-config (not autotuned) kernel."""

    block_m: int
    block_n: int
    num_warps: int
    num_stages: int
    # Shared memory in BLOCK_D-wide rows: the largest ``metadata.shared / (BLOCK_D *
    # element_size)`` measured for SM86 at D in {64, 128, 256} with Triton 3.8.
    shared_rows: int = 0
    # Tensor-memory columns: the largest ``metadata.tmem_size`` measured on SM103 (GB300) for
    # BF16 and FP32 at D in {64, 128, 256} with Triton 3.8. Only 64-row tiles use tcgen05 MMAs.
    tmem_cols: int = 0


# Tensor-memory columns one CTA may allocate on datacenter Blackwell; Triton rejects kernels
# above it when loading them.
TENSOR_MEMORY_COLUMNS = 512


def select_tiles(
    candidates: Sequence[TileConfig],
    block_d: int,
    element_size: int,
    device: torch.device,
) -> TileConfig:
    """Return the first candidate whose estimated shared and tensor memory fit ``device``.

    Candidates are ordered by preference; the last one is the fallback for tiny budgets.
    """
    properties = torch.cuda.get_device_properties(device)
    shared_budget = properties.shared_memory_per_block_optin
    # SM10x/SM11x accumulate tcgen05 MMAs in tensor memory; other GPUs have none.
    tmem_budget = TENSOR_MEMORY_COLUMNS if properties.major in (10, 11) else float("inf")
    return next(
        (
            c
            for c in candidates
            if c.shared_rows * block_d * element_size <= shared_budget
            and c.tmem_cols <= tmem_budget
        ),
        candidates[-1],
    )


def optional_strides(tensor: torch.Tensor | None) -> tuple[int, ...] | None:
    """Strides of an optional kernel operand; None compiles out its constexpr-guarded uses."""
    return None if tensor is None else tensor.stride()


def tuning_key(tensor: str, *names: str):
    """Key a ``TritonTuner`` by the dtype of argument ``tensor`` and the arguments ``names``.

    Unlike ``triton.autotune``'s implicit dtype key, this also sees through TMA descriptors.
    """

    def key(args):
        value = args[tensor]
        dtype = (value.base if isinstance(value, TensorDescriptor) else value).dtype
        return (dtype, *(args[name] for name in names))

    return key


def prune_wide_backward_configs(configs, args):
    """Bound full-width D=512 gradient tiles without changing smaller-head tuning."""
    if args["D"] != 512:
        return configs
    return [
        config
        for config in configs
        if config.num_warps == 4
        and config.num_stages == 1
        and all(size <= 16 for size in config.kwargs.values())
    ]


@triton.jit
def offset_strides(strides, USE_INT64_OFFSETS: tl.constexpr):
    """Select address width before multiplying runtime strides and indices."""
    result = ()
    for axis in tl.static_range(len(strides)):
        result += (tl.cast(strides[axis], tl.int64 if USE_INT64_OFFSETS else tl.int32),)
    return result


@triton.jit
def load_bhsd(
    tensor_ptr,
    strides,
    batch,
    head,
    positions,
    offsets_d,
    mask,
):
    """Load a tile from gather attention's BHSD tensors."""
    return tl.load(
        tensor_ptr
        + ptr_offset(
            (batch, head, positions[:, None], offsets_d[None, :]),
            strides,
        ),
        mask=mask,
        other=0.0,
    )


@triton.jit
def load_document_bounds(
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    positions,
    num_documents,
    sequence_length,
    USE_INT64_OFFSETS: tl.constexpr,
):
    """Find query and candidate bounds; inactive queries have no sparse candidates."""
    document = _document_ids(cu_seqlens_ptr, positions, num_documents, USE_INT64_OFFSETS)
    offset = document.to(tl.int64) if USE_INT64_OFFSETS else document
    # The right-sided search skips empty documents and returns N for tail/padded positions.
    start = tl.load(cu_seqlens_ptr + offset)
    end = tl.load(
        cu_seqlens_ptr + offset + 1,
        mask=document < num_documents,
        other=sequence_length,
    )
    candidate_start = tl.load(cu_seqlens_k_ptr + offset)
    candidate_end = tl.load(cu_seqlens_k_ptr + tl.minimum(offset + 1, num_documents))
    return start, end, candidate_start, candidate_end


@triton.jit
def store_bhsd(
    tensor_ptr,
    value,
    strides,
    batch,
    head,
    positions,
    offsets_d,
    mask,
):
    """Store a tile to gather attention's BHSD tensors."""
    tl.store(
        tensor_ptr
        + ptr_offset(
            (batch, head, positions[:, None], offsets_d[None, :]),
            strides,
        ),
        value,
        mask=mask,
    )


@triton.jit
def causal_window_mask(
    query_positions,
    key_positions,
    query_mask,
    key_mask,
    window: tl.constexpr,
):
    """Mask valid gather-attention pairs to the causal local window."""
    return (
        query_mask[:, None]
        & key_mask[None, :]
        & (key_positions[None, :] <= query_positions[:, None])
        & (key_positions[None, :] >= query_positions[:, None] - window + 1)
    )


@triton.jit
def online_softmax_update(accumulator, running_max, running_sum, logits, values):
    """Merge one gather-attention tile into FP32 online-softmax state."""
    tile_max = tl.max(logits, axis=1)
    new_max = tl.maximum(running_max, tile_max)
    # An empty prefix with no sink has max=-inf. Keep that state for later tiles,
    # but avoid -inf - -inf while computing this tile's zero contribution.
    safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
    alpha = tl.exp(running_max - safe_max)
    probabilities = tl.exp(logits - safe_max[:, None])
    accumulator *= alpha[:, None]
    accumulator += tl.dot(probabilities.to(values.dtype), values, input_precision="tf32x3")
    running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
    return accumulator, new_max, running_sum


# Sizes and the batch stride of ``kv_indices`` only feed bounds, offsets, and loop trip counts.
# A constexpr TOPK measured no faster on B200 (QSA top-k 2051), so top-k widths share a kernel.
@triton.jit(
    do_not_specialize=["num_documents", "KV_INDICES_BATCH_STRIDE", "S", "SPARSE_SEQ_LEN", "TOPK"]
)
def _resolve_selected_rows(
    kv_indices_ptr,
    rows_ptr,
    counts_ptr,
    cu_seqlens_ptr,
    cu_seqlens_k_ptr,
    num_documents,
    KV_INDICES_BATCH_STRIDE,
    KV_INDICES_STRIDES,
    S,
    SPARSE_SEQ_LEN,
    TOPK,
    HAS_CU_SEQLENS: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Compact one query's valid selections, as global sparse-pool rows, to its row's front."""
    KV_INDICES_STRIDES = offset_strides(
        (KV_INDICES_BATCH_STRIDE, KV_INDICES_STRIDES[0], KV_INDICES_STRIDES[1]),
        USE_INT64_OFFSETS,
    )
    row = tl.program_id(0).to(tl.int64)
    batch = row // S
    query_position = row % S
    candidate_start = 0
    candidate_end = SPARSE_SEQ_LEN
    if HAS_CU_SEQLENS:
        _, _, candidate_start, candidate_end = load_document_bounds(
            cu_seqlens_ptr, cu_seqlens_k_ptr, query_position, num_documents, S, True
        )
    count = 0
    for start in range(0, TOPK, BLOCK_K):
        slots = start + tl.arange(0, BLOCK_K)
        indices = tl.load(
            kv_indices_ptr + ptr_offset((batch, query_position, slots), KV_INDICES_STRIDES),
            mask=slots < TOPK,
            other=-1,
        )
        valid = (indices >= 0) & (indices < candidate_end - candidate_start)
        # Mask local coordinates before adding the document base, including int64 sentinels.
        local = tl.where(valid, indices, 0)
        rows = local + candidate_start
        positions = count + tl.cumsum(valid.to(tl.int32), axis=0) - 1
        tl.store(rows_ptr + row * TOPK + positions, rows.to(rows_ptr.dtype.element_ty), mask=valid)
        count += tl.sum(valid.to(tl.int32), axis=0)
    tl.store(counts_ptr + row, count)


def resolve_selected_rows(
    kv_indices: torch.Tensor,
    sparse_seq_len: int,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_k: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve document-local selections to global pool rows, compacted per query.

    Returns ``(rows, counts)``. ``rows`` [B, S, topk] (int32, or int64 for pools
    of 2**31 rows or more) holds each query's valid selections in their original order
    (repeats included) as global pool rows in its first ``counts[b, s]`` entries; the rest are
    unspecified and never read. The grouped kernels then skip per-gather document checks and
    loop over full tiles of rows without masks, masking only the last, partial tile.
    """
    batch, seq_len, topk = kv_indices.shape
    device = kv_indices.device
    # int32 rows, unless the pool is too long for them.
    rows = torch.empty(
        (batch, seq_len, topk),
        device=device,
        dtype=torch.int32 if sparse_seq_len < 2**31 else torch.int64,
    )
    # The kernel writes every count.
    counts = torch.empty((batch, seq_len), device=device, dtype=torch.int32)
    if topk == 0:
        return rows, counts.zero_()
    has_cu_seqlens = cu_seqlens is not None
    _resolve_selected_rows[(batch * seq_len,)](
        kv_indices,
        rows,
        counts,
        cu_seqlens,
        cu_seqlens_k,
        cu_seqlens.numel() - 1 if has_cu_seqlens else 0,
        KV_INDICES_BATCH_STRIDE=kv_indices.stride(0),
        KV_INDICES_STRIDES=kv_indices.stride()[1:],
        S=seq_len,
        SPARSE_SEQ_LEN=sparse_seq_len,
        TOPK=topk,
        HAS_CU_SEQLENS=has_cu_seqlens,
        USE_INT64_OFFSETS=requires_int64_offsets(kv_indices),
        BLOCK_K=256,
        num_warps=4,
    )
    return rows, counts


@triton.jit
def head_block_heads(head_block, H: tl.constexpr, KV_HEADS: tl.constexpr, BLOCK_H: tl.constexpr):
    """``(kv_head, heads, head_mask)`` for one index of a head-block grid axis.

    With KV_HEADS == 1 the blocks tile all H query heads, which read KV head 0. Otherwise they
    enumerate (KV head, block of its H // KV_HEADS query heads), so each block reads one KV
    head and each gathered K/V row serves the whole block.
    """
    if KV_HEADS == 1:
        kv_head = 0
        heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
        head_mask = heads < H
    else:
        group = H // KV_HEADS
        blocks_per_group = (group + BLOCK_H - 1) // BLOCK_H
        kv_head = head_block // blocks_per_group
        group_offsets = (head_block % blocks_per_group) * BLOCK_H + tl.arange(0, BLOCK_H)
        heads = kv_head * group + group_offsets
        head_mask = group_offsets < group
    return kv_head, heads, head_mask
