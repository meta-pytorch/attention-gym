# Copyright (c) 2025 Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Active-slot cache movement for streaming paged chunk KDA."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from attn_gym._backends.triton.utils import ptr_offset, requires_int64_offsets

_CHUNK_SIZE = tl.constexpr(64)


def _int64_offset_heuristic(*tensor_names: str):
    return {
        "USE_INT64_OFFSETS": lambda args: requires_int64_offsets(
            *(args[name] for name in tensor_names)
        )
    }


def _packed_rows_are_disjoint(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Prove that two dense-row views of the same paged storage do not overlap."""

    def row_layout(tensor: torch.Tensor) -> tuple[int, int, int, int] | None:
        if tensor.ndim == 0 or tensor.shape[0] == 0:
            return None
        expected_stride = 1
        for size, stride in reversed(tuple(zip(tensor.shape[1:], tensor.stride()[1:]))):
            if size != 1 and stride != expected_stride:
                return None
            expected_stride *= size
        element_size = tensor.element_size()
        return (
            tensor.storage_offset() * element_size,
            expected_stride * element_size,
            tensor.stride(0) * element_size,
            tensor.shape[0],
        )

    left_layout = row_layout(left)
    right_layout = row_layout(right)
    if left_layout is None or right_layout is None:
        return False
    left_start, left_size, left_stride, left_rows = left_layout
    right_start, right_size, right_stride, right_rows = right_layout
    if left_rows == right_rows == 1:
        return left_start + left_size <= right_start or right_start + right_size <= left_start
    if left_stride <= 0 or left_stride != right_stride:
        return False
    if left_size > left_stride or right_size > right_stride:
        return False
    right_offset = (right_start - left_start) % left_stride
    return right_offset >= left_size and left_stride - right_offset >= right_size


def _tensors_overlap(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch._C._overlaps(left, right) and not _packed_rows_are_disjoint(left, right)


def _validate_replay_aliases(
    replay_tensors: tuple[tuple[str, torch.Tensor], ...],
    read_only_tensors: tuple[tuple[str, torch.Tensor | None], ...],
) -> None:
    """Reject storage aliasing that would make replay updates order-dependent."""
    for index, (name, tensor) in enumerate(replay_tensors):
        for other_name, other_tensor in replay_tensors[index + 1 :]:
            if _tensors_overlap(tensor, other_tensor):
                raise ValueError(f"replay_state {name} must not alias replay_state {other_name}")
        for other_name, other_tensor in read_only_tensors:
            if other_tensor is not None and _tensors_overlap(tensor, other_tensor):
                raise ValueError(f"replay_state {name} must not alias {other_name}")


@triton.heuristics(
    _int64_offset_heuristic(
        "replay_count",
        "state_indices",
        "has_initial_state",
        "cu_seqlens",
        "tail_counts",
        "prefix_cu_seqlens",
        "pack_cu_seqlens",
    )
)
@triton.jit
def _plan_prefill_kernel(
    replay_count,
    state_indices,
    has_initial_state,
    cu_seqlens,
    tail_counts,
    prefix_cu_seqlens,
    pack_cu_seqlens,
    count_stride_s: tl.constexpr,
    num_sequences,
    use_has_initial_state: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    sequence = tl.arange(0, BLOCK)
    in_bounds = sequence < num_sequences
    sequence_offset = sequence
    if USE_INT64_OFFSETS:
        sequence_offset = sequence_offset.to(tl.int64)

    slot = tl.load(state_indices + sequence_offset, mask=in_bounds, other=0).to(tl.int64)
    count = tl.load(
        replay_count + slot * count_stride_s,
        mask=in_bounds & (slot > 0),
        other=0,
    )
    if use_has_initial_state:
        count = tl.where(
            tl.load(has_initial_state + sequence_offset, mask=in_bounds, other=0),
            count,
            0,
        )
    input_length = tl.load(cu_seqlens + sequence_offset + 1, mask=in_bounds, other=0) - tl.load(
        cu_seqlens + sequence_offset, mask=in_bounds, other=0
    )
    logical_length = tl.where(in_bounds & (slot > 0), count + input_length, 0)
    complete_length = logical_length // _CHUNK_SIZE * _CHUNK_SIZE
    tail_count = logical_length - complete_length

    tl.store(prefix_cu_seqlens, 0)
    tl.store(pack_cu_seqlens, 0)
    tl.store(tail_counts + sequence_offset, tail_count, mask=in_bounds)
    tl.store(
        prefix_cu_seqlens + sequence_offset + 1,
        tl.cumsum(complete_length, axis=0),
        mask=in_bounds,
    )
    tl.store(
        pack_cu_seqlens + sequence_offset + 1,
        tl.cumsum(logical_length, axis=0),
        mask=in_bounds,
    )


@triton.heuristics(
    _int64_offset_heuristic(
        "source",
        "replay_cache",
        "state_indices",
        "input_cu_seqlens",
        "prefix_cu_seqlens",
        "pack_cu_seqlens",
        "tail_counts",
        "prefix",
        "tail",
        "output_route",
    )
)
@triton.jit
def _pack_prefill_kernel(
    source,
    replay_cache,
    state_indices,
    input_cu_seqlens,
    prefix_cu_seqlens,
    pack_cu_seqlens,
    tail_counts,
    prefix,
    tail,
    output_route,
    source_strides: tl.constexpr,
    replay_strides: tl.constexpr,
    prefix_strides: tl.constexpr,
    tail_strides: tl.constexpr,
    num_sequences: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    is_packed: tl.constexpr,
    write_output_routes: tl.constexpr,
    SEARCH_STEPS: tl.constexpr,
    num_elements: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if USE_INT64_OFFSETS:
        offset = offset.to(tl.int64)
    feature_size = num_heads * head_dim
    feature = offset % feature_size
    packed_token = offset // feature_size

    # Find the request owning each token in the compact logical work space. Repeated
    # boundaries naturally skip inactive and empty requests (upper-bound search).
    low = tl.zeros_like(packed_token)
    high = low + num_sequences
    for _ in tl.static_range(SEARCH_STEPS):
        middle = (low + high) // 2
        searching = low < high
        boundary = tl.load(
            pack_cu_seqlens + middle + 1,
            mask=searching & (middle < num_sequences),
            other=0,
        )
        goes_right = searching & (boundary <= packed_token)
        low = tl.where(goes_right, middle + 1, low)
        high = tl.where(searching & ~goes_right, middle, high)
    sequence = low
    in_bounds = (offset < num_elements) & (sequence < num_sequences)
    pack_start = tl.load(
        pack_cu_seqlens + sequence,
        mask=in_bounds,
        other=0,
    ).to(tl.int64)
    logical_token = packed_token - pack_start
    head = feature // head_dim
    dim = feature % head_dim
    slot = tl.load(state_indices + sequence, mask=in_bounds, other=0).to(tl.int64)
    input_start = tl.load(input_cu_seqlens + sequence, mask=in_bounds, other=0).to(tl.int64)
    input_length = tl.load(input_cu_seqlens + sequence + 1, mask=in_bounds, other=0) - input_start
    prefix_start = tl.load(prefix_cu_seqlens + sequence, mask=in_bounds, other=0).to(tl.int64)
    complete_length = (
        tl.load(prefix_cu_seqlens + sequence + 1, mask=in_bounds, other=0) - prefix_start
    )
    tail_count = tl.load(tail_counts + sequence, mask=in_bounds, other=0)
    count = tl.where(slot > 0, complete_length + tail_count - input_length, 0)
    active = in_bounds & (slot > 0) & (logical_token < complete_length + tail_count)

    from_replay = active & (logical_token < count)
    input_token = input_start + logical_token - count
    value = tl.where(
        from_replay,
        tl.load(
            replay_cache + ptr_offset((slot, logical_token, head, dim), replay_strides),
            mask=from_replay,
            other=0.0,
        ),
        tl.load(
            source
            + ptr_offset(
                (
                    0 if is_packed else sequence,
                    input_token if is_packed else input_token - input_start,
                    head,
                    dim,
                ),
                source_strides,
            ),
            mask=active & ~from_replay,
            other=0.0,
        ),
    )

    in_prefix = active & (logical_token < complete_length)
    tl.store(
        prefix
        + ptr_offset(
            (
                0,
                prefix_start + logical_token,
                head,
                dim,
            ),
            prefix_strides,
        ),
        value,
        mask=in_prefix,
    )
    tl.store(
        tail
        + ptr_offset(
            (sequence, logical_token - complete_length, head, dim),
            tail_strides,
        ),
        value,
        mask=active & ~in_prefix,
    )
    if write_output_routes:
        is_input = active & (logical_token >= count)
        output_token = input_start + logical_token - count
        first_feature = feature == 0
        tl.store(
            output_route + output_token,
            tl.where(
                in_prefix,
                prefix_start + logical_token + 1,
                -(sequence * _CHUNK_SIZE + logical_token - complete_length + 1),
            ),
            mask=is_input & first_feature,
        )


@triton.heuristics(_int64_offset_heuristic("prefix", "tail", "output_route", "output"))
@triton.jit
def _scatter_output_kernel(
    prefix,
    tail,
    output_route,
    output,
    feature_size: tl.constexpr,
    num_elements: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if USE_INT64_OFFSETS:
        offset = offset.to(tl.int64)
    in_bounds = offset < num_elements
    output_token = offset // feature_size
    feature = offset % feature_size
    route = tl.load(output_route + output_token, mask=in_bounds, other=0).to(tl.int64)
    from_prefix = in_bounds & (route > 0)
    from_tail = in_bounds & (route < 0)
    prefix_token = route - 1
    tail_token = -route - 1
    value = tl.where(
        from_prefix,
        tl.load(prefix + prefix_token * feature_size + feature, mask=from_prefix, other=0.0),
        tl.load(tail + tail_token * feature_size + feature, mask=from_tail, other=0.0),
    )
    tl.store(
        output + offset,
        value,
        mask=from_prefix | from_tail,
    )


@triton.heuristics(
    _int64_offset_heuristic("tail", "tail_counts", "state_indices", "replay_cache", "replay_count")
)
@triton.jit
def _commit_tail_kernel(
    tail,
    tail_counts,
    state_indices,
    replay_cache,
    replay_count,
    tail_strides: tl.constexpr,
    replay_strides: tl.constexpr,
    count_stride_s: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    write_count: tl.constexpr,
    num_elements: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if USE_INT64_OFFSETS:
        offset = offset.to(tl.int64)
    feature_size = num_heads * head_dim
    sequence_size = _CHUNK_SIZE * feature_size
    in_bounds = offset < num_elements
    sequence = offset // sequence_size
    local_offset = offset % sequence_size
    feature = local_offset % feature_size
    token = local_offset // feature_size
    head = feature // head_dim
    dim = feature % head_dim
    slot = tl.load(state_indices + sequence, mask=in_bounds, other=0).to(tl.int64)
    tail_count = tl.load(tail_counts + sequence, mask=in_bounds, other=0)
    active = in_bounds & (slot > 0) & (token < tail_count)
    tl.store(
        replay_cache + ptr_offset((slot, token, head, dim), replay_strides),
        tl.load(
            tail + ptr_offset((sequence, token, head, dim), tail_strides),
            mask=active,
            other=0.0,
        ),
        mask=active,
    )
    if write_count:
        tl.store(
            replay_count + slot * count_stride_s + local_offset * 0,
            tail_count,
            mask=in_bounds & (local_offset == 0) & (slot > 0),
        )


@triton.heuristics(
    _int64_offset_heuristic("replay_count", "state_indices", "has_initial_state", "active_counts")
)
@triton.jit
def _gather_counts_kernel(
    replay_count,
    state_indices,
    has_initial_state,
    active_counts,
    count_stride_s: tl.constexpr,
    num_active: tl.constexpr,
    use_has_initial_state: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    active_index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if USE_INT64_OFFSETS:
        active_index = active_index.to(tl.int64)
    in_bounds = active_index < num_active
    slot = tl.load(state_indices + active_index, mask=in_bounds, other=0).to(tl.int64)
    count = tl.load(
        replay_count + slot * count_stride_s,
        mask=in_bounds & (slot > 0),
        other=0,
    )
    if use_has_initial_state:
        count = tl.where(
            tl.load(has_initial_state + active_index, mask=in_bounds, other=0),
            count,
            0,
        )
    tl.store(active_counts + active_index, count, mask=in_bounds)


@triton.heuristics(
    _int64_offset_heuristic(
        "replay_cache", "current", "state_indices", "active_counts", "active_cache"
    )
)
@triton.jit
def _update_and_gather_kernel(
    replay_cache,
    current,
    state_indices,
    active_counts,
    active_cache,
    cache_strides: tl.constexpr,
    current_strides: tl.constexpr,
    active_strides: tl.constexpr,
    head_dim: tl.constexpr,
    num_tokens: tl.constexpr,
    feature_size: tl.constexpr,
    num_elements: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if USE_INT64_OFFSETS:
        offset = offset.to(tl.int64)
    in_bounds = offset < num_elements
    feature = offset % feature_size
    remaining = offset // feature_size
    token = remaining % num_tokens
    active_index = remaining // num_tokens

    slot = tl.load(state_indices + active_index, mask=in_bounds, other=0).to(tl.int64)
    active = in_bounds & (slot > 0)
    count = tl.load(active_counts + active_index, mask=in_bounds, other=0)
    head = feature // head_dim
    dim = feature % head_dim
    cache_offset = ptr_offset((slot, token, head, dim), cache_strides)
    current_offset = ptr_offset((active_index, 0, head, dim), current_strides)
    active_offset = ptr_offset((active_index, token, head, dim), active_strides)
    cache_ptr = replay_cache + cache_offset
    is_current = active & (token == count)
    value = tl.load(cache_ptr, mask=active & (token < count), other=0.0)
    value = tl.where(
        is_current,
        tl.load(current + current_offset, mask=is_current, other=0.0),
        value,
    )
    tl.store(cache_ptr, value, mask=is_current)
    tl.store(active_cache + active_offset, value, mask=in_bounds)


@triton.heuristics(
    _int64_offset_heuristic("state_cache", "state_indices", "has_initial_state", "active_state")
)
@triton.jit
def _gather_state_kernel(
    state_cache,
    state_indices,
    has_initial_state,
    active_state,
    state_stride_s: tl.constexpr,
    active_stride_b: tl.constexpr,
    state_size: tl.constexpr,
    num_elements: tl.constexpr,
    use_has_initial_state: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if USE_INT64_OFFSETS:
        offset = offset.to(tl.int64)
    in_bounds = offset < num_elements
    state_offset = offset % state_size
    active_index = offset // state_size
    slot = tl.load(state_indices + active_index, mask=in_bounds, other=0).to(tl.int64)
    active = in_bounds & (slot > 0)
    if use_has_initial_state:
        active &= tl.load(has_initial_state + active_index, mask=in_bounds, other=0)
    tl.store(
        active_state + active_index * active_stride_b + state_offset,
        tl.load(
            state_cache + slot * state_stride_s + state_offset,
            mask=active,
            other=0.0,
        ),
        mask=in_bounds,
    )


@triton.heuristics(
    _int64_offset_heuristic(
        "final_state",
        "state_indices",
        "has_initial_state",
        "active_counts",
        "state_cache",
        "replay_count",
    )
)
@triton.jit
def _commit_state_kernel(
    final_state,
    state_indices,
    has_initial_state,
    active_counts,
    state_cache,
    replay_count,
    final_stride_b: tl.constexpr,
    state_stride_s: tl.constexpr,
    count_stride_s: tl.constexpr,
    state_size: tl.constexpr,
    chunk_size: tl.constexpr,
    num_elements: tl.constexpr,
    use_has_initial_state: tl.constexpr,
    USE_INT64_OFFSETS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if USE_INT64_OFFSETS:
        offset = offset.to(tl.int64)
    in_bounds = offset < num_elements
    state_offset = offset % state_size
    active_index = offset // state_size
    slot = tl.load(state_indices + active_index, mask=in_bounds, other=0).to(tl.int64)
    active = in_bounds & (slot > 0)
    count = tl.load(active_counts + active_index, mask=in_bounds, other=0)
    completes_chunk = count == chunk_size - 1
    fresh_slot = False
    if use_has_initial_state:
        fresh_slot = ~tl.load(has_initial_state + active_index, mask=in_bounds, other=0)
    tl.store(
        state_cache + slot * state_stride_s + state_offset,
        tl.where(
            completes_chunk,
            tl.load(
                final_state + active_index * final_stride_b + state_offset,
                mask=active & completes_chunk,
                other=0.0,
            ),
            0.0,
        ),
        mask=active & (completes_chunk | fresh_slot),
    )
    tl.store(
        replay_count + slot * count_stride_s,
        tl.where(count == chunk_size - 1, 0, count + 1),
        mask=active & (state_offset == 0),
    )


def prepare_paged_chunk_replay(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    replay_q: torch.Tensor,
    replay_k: torch.Tensor,
    replay_v: torch.Tensor,
    replay_gate: torch.Tensor,
    replay_beta: torch.Tensor,
    replay_count: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    """Append one token and gather active replay windows and boundary states."""
    _validate_replay_aliases(
        (
            ("q", replay_q),
            ("k", replay_k),
            ("v", replay_v),
            ("gate", replay_gate),
            ("beta", replay_beta),
            ("count", replay_count),
        ),
        (
            ("q", q),
            ("k", k),
            ("v", v),
            ("gate", gate),
            ("beta", beta),
            ("state_cache", state_cache),
            ("state_indices", state_indices),
            ("has_initial_state", has_initial_state),
        ),
    )
    num_active = q.shape[0]
    active_counts = replay_count.new_empty((num_active,))
    active_state = state_cache.new_empty((num_active, *state_cache.shape[1:]))
    active_caches = tuple(
        cache.new_empty((num_active, *cache.shape[1:]))
        for cache in (replay_q, replay_k, replay_v, replay_gate, replay_beta)
    )
    count_block = 128
    _gather_counts_kernel[(triton.cdiv(num_active, count_block),)](
        replay_count,
        state_indices,
        has_initial_state if has_initial_state is not None else state_indices,
        active_counts,
        count_stride_s=replay_count.stride(0),
        num_active=num_active,
        use_has_initial_state=has_initial_state is not None,
        BLOCK=count_block,
    )

    block = 256
    for cache, current, active_cache in zip(
        (replay_q, replay_k, replay_v, replay_gate, replay_beta),
        (q, k, v, gate, beta),
        active_caches,
        strict=True,
    ):
        feature_size = current.shape[2] * (current.shape[3] if current.ndim == 4 else 1)
        num_elements = num_active * cache.shape[1] * feature_size
        _update_and_gather_kernel[(triton.cdiv(num_elements, block),)](
            cache,
            current,
            state_indices,
            active_counts,
            active_cache,
            cache_strides=(cache.stride() if current.ndim == 4 else (*cache.stride(), 0)),
            current_strides=(current.stride() if current.ndim == 4 else (*current.stride(), 0)),
            active_strides=(
                active_cache.stride() if current.ndim == 4 else (*active_cache.stride(), 0)
            ),
            head_dim=current.shape[-1] if current.ndim == 4 else 1,
            num_tokens=cache.shape[1],
            feature_size=feature_size,
            num_elements=num_elements,
            BLOCK=block,
        )

    state_size = state_cache.shape[1] * state_cache.shape[2] * state_cache.shape[3]
    num_state_elements = num_active * state_size
    _gather_state_kernel[(triton.cdiv(num_state_elements, block),)](
        state_cache,
        state_indices,
        has_initial_state if has_initial_state is not None else state_indices,
        active_state,
        state_stride_s=state_cache.stride(0),
        active_stride_b=active_state.stride(0),
        state_size=state_size,
        num_elements=num_state_elements,
        use_has_initial_state=has_initial_state is not None,
        BLOCK=block,
    )
    return (*active_caches, active_state, active_counts)


def commit_paged_chunk_replay(
    final_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    active_counts: torch.Tensor,
    state_cache: torch.Tensor,
    replay_count: torch.Tensor,
) -> None:
    """Commit completed boundaries and advance active replay counts."""
    num_active = final_state.shape[0]
    state_size = final_state.shape[1] * final_state.shape[2] * final_state.shape[3]
    num_elements = num_active * state_size
    block = 256
    _commit_state_kernel[(triton.cdiv(num_elements, block),)](
        final_state,
        state_indices,
        has_initial_state if has_initial_state is not None else state_indices,
        active_counts,
        state_cache,
        replay_count,
        final_stride_b=final_state.stride(0),
        state_stride_s=state_cache.stride(0),
        count_stride_s=replay_count.stride(0),
        state_size=state_size,
        chunk_size=_CHUNK_SIZE,
        num_elements=num_elements,
        use_has_initial_state=has_initial_state is not None,
        BLOCK=block,
    )


def prepare_paged_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    state_cache: torch.Tensor,
    replay_q: torch.Tensor,
    replay_k: torch.Tensor,
    replay_v: torch.Tensor,
    replay_gate: torch.Tensor,
    replay_beta: torch.Tensor,
    replay_count: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    input_cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Pack complete chunks and tails for graph-safe two-pass replay prefill."""
    _validate_replay_aliases(
        (
            ("q", replay_q),
            ("k", replay_k),
            ("v", replay_v),
            ("gate", replay_gate),
            ("beta", replay_beta),
            ("count", replay_count),
        ),
        (
            ("q", q),
            ("k", k),
            ("v", v),
            ("gate", gate),
            ("beta", beta),
            ("state_cache", state_cache),
            ("state_indices", state_indices),
            ("has_initial_state", has_initial_state),
            ("cu_seqlens", input_cu_seqlens),
        ),
    )
    num_sequences = state_indices.shape[0]
    prefix_capacity = (
        triton.cdiv(
            q.shape[0] * q.shape[1] + (_CHUNK_SIZE - 1) * num_sequences,
            _CHUNK_SIZE,
        )
        * _CHUNK_SIZE
    )
    # Keep all work on grid X: CUDA grid Y is limited to 65,535. This aggregate
    # capacity also avoids repeating the total packed-token bound for every request.
    max_pack_tokens = q.shape[0] * q.shape[1] + (_CHUNK_SIZE - 1) * num_sequences
    tail_counts = replay_count.new_empty((num_sequences,))
    prefix_cu_seqlens = replay_count.new_empty((num_sequences + 1,))
    pack_cu_seqlens = replay_count.new_empty((num_sequences + 1,))
    output_capacity = q.shape[0] * q.shape[1]
    output_route = q.new_zeros((output_capacity,), dtype=torch.int64)
    prefix_caches = tuple(
        cache.new_zeros((1, prefix_capacity, *cache.shape[2:]))
        for cache in (replay_q, replay_k, replay_v, replay_gate, replay_beta)
    )
    tail_caches = tuple(
        cache.new_zeros((num_sequences, _CHUNK_SIZE, *cache.shape[2:]))
        for cache in (replay_q, replay_k, replay_v, replay_gate, replay_beta)
    )
    plan_block = triton.next_power_of_2(num_sequences)
    _plan_prefill_kernel[(1,)](
        replay_count,
        state_indices,
        has_initial_state if has_initial_state is not None else state_indices,
        input_cu_seqlens,
        tail_counts,
        prefix_cu_seqlens,
        pack_cu_seqlens,
        count_stride_s=replay_count.stride(0),
        num_sequences=num_sequences,
        use_has_initial_state=has_initial_state is not None,
        BLOCK=plan_block,
        num_warps=min(8, max(1, plan_block // 32)),
    )

    is_packed = q.shape[0] == 1
    pack_block = 256
    search_steps = num_sequences.bit_length()
    for index, (source, replay_cache, prefix, tail) in enumerate(
        zip(
            (q, k, v, gate, beta),
            (replay_q, replay_k, replay_v, replay_gate, replay_beta),
            prefix_caches,
            tail_caches,
            strict=True,
        )
    ):
        num_heads = source.shape[2]
        head_dim = source.shape[3] if source.ndim == 4 else 1
        num_elements = max_pack_tokens * num_heads * head_dim
        _pack_prefill_kernel[(triton.cdiv(num_elements, pack_block),)](
            source,
            replay_cache,
            state_indices,
            input_cu_seqlens,
            prefix_cu_seqlens,
            pack_cu_seqlens,
            tail_counts,
            prefix,
            tail,
            output_route,
            source_strides=(source.stride() if source.ndim == 4 else (*source.stride(), 0)),
            replay_strides=(
                replay_cache.stride() if source.ndim == 4 else (*replay_cache.stride(), 0)
            ),
            prefix_strides=(prefix.stride() if source.ndim == 4 else (*prefix.stride(), 0)),
            tail_strides=(tail.stride() if source.ndim == 4 else (*tail.stride(), 0)),
            num_sequences=num_sequences,
            num_heads=num_heads,
            head_dim=head_dim,
            is_packed=is_packed,
            write_output_routes=index == 0,
            SEARCH_STEPS=search_steps,
            num_elements=num_elements,
            BLOCK=pack_block,
        )
    return (
        *prefix_caches,
        prefix_cu_seqlens,
        output_route,
        *tail_caches,
        tail_counts,
    )


def gather_paged_chunk_state(
    state_cache: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather post-prefix boundary states for the fixed tail pass."""
    num_sequences = state_indices.shape[0]
    active_state = state_cache.new_empty((num_sequences, *state_cache.shape[1:]))
    state_size = state_cache.shape[1] * state_cache.shape[2] * state_cache.shape[3]
    num_elements = num_sequences * state_size
    block = 256
    _gather_state_kernel[(triton.cdiv(num_elements, block),)](
        state_cache,
        state_indices,
        state_indices,
        active_state,
        state_stride_s=state_cache.stride(0),
        active_stride_b=active_state.stride(0),
        state_size=state_size,
        num_elements=num_elements,
        use_has_initial_state=False,
        BLOCK=block,
    )
    return active_state


def commit_paged_chunk_prefill(
    prefix_output: torch.Tensor,
    tail_output: torch.Tensor,
    output_route: torch.Tensor,
    tail_q: torch.Tensor,
    tail_k: torch.Tensor,
    tail_v: torch.Tensor,
    tail_gate: torch.Tensor,
    tail_beta: torch.Tensor,
    tail_counts: torch.Tensor,
    state_indices: torch.Tensor,
    replay_q: torch.Tensor,
    replay_k: torch.Tensor,
    replay_v: torch.Tensor,
    replay_gate: torch.Tensor,
    replay_beta: torch.Tensor,
    replay_count: torch.Tensor,
    output_template: torch.Tensor,
) -> torch.Tensor:
    """Scatter new-token outputs and retain the final incomplete chunks."""
    output = output_template.new_zeros(output_template.shape)
    feature_size = output.shape[2] * output.shape[3]
    block = 256
    num_elements = output.shape[0] * output.shape[1] * feature_size
    _scatter_output_kernel[(triton.cdiv(num_elements, block),)](
        prefix_output,
        tail_output,
        output_route,
        output,
        feature_size=feature_size,
        num_elements=num_elements,
        BLOCK=block,
    )

    num_sequences = state_indices.shape[0]
    for index, (tail, replay_cache) in enumerate(
        zip(
            (tail_q, tail_k, tail_v, tail_gate, tail_beta),
            (replay_q, replay_k, replay_v, replay_gate, replay_beta),
            strict=True,
        )
    ):
        num_heads = tail.shape[2]
        head_dim = tail.shape[3] if tail.ndim == 4 else 1
        num_elements = num_sequences * _CHUNK_SIZE * num_heads * head_dim
        _commit_tail_kernel[(triton.cdiv(num_elements, block),)](
            tail,
            tail_counts,
            state_indices,
            replay_cache,
            replay_count,
            tail_strides=(tail.stride() if tail.ndim == 4 else (*tail.stride(), 0)),
            replay_strides=(
                replay_cache.stride() if tail.ndim == 4 else (*replay_cache.stride(), 0)
            ),
            count_stride_s=replay_count.stride(0),
            num_heads=num_heads,
            head_dim=head_dim,
            write_count=index == 0,
            num_elements=num_elements,
            BLOCK=block,
        )
    return output


__all__ = [
    "commit_paged_chunk_prefill",
    "commit_paged_chunk_replay",
    "gather_paged_chunk_state",
    "prepare_paged_chunk_prefill",
    "prepare_paged_chunk_replay",
]
